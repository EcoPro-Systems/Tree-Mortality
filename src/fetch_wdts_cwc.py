#!/usr/bin/env python
"""
Canopy water content (CWC) indicators from WDTS AVIRIS-Classic corrected
surface reflectance (Shafron et al. 2025, ORNL DAAC 2391,
doi:10.3334/ORNLDAAC/2391) on an HLS AOI grid.

The reflectance is 15 m ENVI BSQ (224 bands, float32, topographic + BRDF
corrected, ~60 GB per flight line). Only the AOI rows of the needed bands
are read, as one HTTP range request per band (in BSQ a band's rows are
contiguous), which is far faster than GDAL's per-scanline reads. CMR
footprints for these lines are unreliable, so candidate lines are all lines
of the acquisition date(s) and their true extent comes from the ENVI header.

Per 15 m pixel:
  ewt980    equivalent water thickness (cm) from a Beer-Lambert fit of the
            980 nm liquid-water feature, 865-1085 nm excluding 925-970 nm
            (residual vapour): ln R = c0 + c1 (lambda - 975) - Kw(lambda) EWT,
            Kw the PROSPECT-D specific absorption of water (Feret et al. 2017)
  ewt1200   the same over the 1200 nm feature (1100-1265 nm, excluding
            1110-1160 nm)
  ndwi      (R860 - R1240) / (R860 + R1240)  (Gao 1996)
  bd1200    continuum-removed band depth at 1200 nm (continuum 1080-1265 nm)
  ndvi      (R860 - R660) / (R860 + R660)
Pixels are aggregated 2 x 2 to the 30 m AOI grid (15 m pixel edges nest in
the 30 m lattice). Where lines overlap, each 30 m cell takes the line from
the primary date (the one the trait mosaic used) and, among those, the one
where the cell is closest to the swath centre (nearest nadir).

Output: <aoi>_cwc.nc with dims (year, y, x), plus source line per cell.
"""
import os
import re
import time
import click
import numpy as np
import requests
import rasterio
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from util import load_config
from fetch_hls_aoi import aoi_grid

BASE = ('https://data.ornldaac.earthdata.nasa.gov/protected/wdts/'
        'WDTS_AVIRIS-C_L2_corrected/data/')
# Acquisition dates per year: primary first, then fallback (same campaign)
DATES = {2013: ['130612', '130626'], 2014: ['140603'],
         2015: ['150601', '150602'], 2016: ['160621'], 2017: ['170607'],
         2018: ['180622']}
MAX_RUN = 25
WIN_980 = (865, 1085, (925, 970))
WIN_1200 = (1100, 1265, (1110, 1160))
NODATA = -9999.0
RETRIES = 5


def session():
    s = requests.Session()
    s.headers['User-Agent'] = 'ecopro-wdts-cwc'
    return s


def get_range(sess, url, off, n):
    for attempt in range(1, RETRIES + 1):
        try:
            r = sess.get(url, headers={'Range': f'bytes={off}-{off + n - 1}'},
                         timeout=(30, 600))
            if r.status_code == 206 and len(r.content) == n:
                return r.content
            last = f'HTTP {r.status_code}, {len(r.content)} bytes'
        except requests.RequestException as exc:
            last = repr(exc)
        time.sleep(3 * attempt)
    raise RuntimeError(f'{url}: {last}')


def parse_header(text):
    """ENVI header -> dict of the fields we need"""
    h = {}
    for key in ('samples', 'lines', 'bands', 'data type', 'header offset',
                'byte order', 'interleave'):
        m = re.search(rf'^{key}\s*=\s*(\S+)', text, re.M)
        h[key] = m.group(1) if m else None
    m = re.search(r'map info\s*=\s*\{([^}]*)\}', text)
    mi = [v.strip() for v in m.group(1).split(',')]
    h['x0'], h['y0'] = float(mi[3]), float(mi[4])
    h['res'] = float(mi[5])
    h['zone'] = int(mi[7])
    m = re.search(r'wavelength\s*=\s*\{([^}]*)\}', text)
    h['wavelength'] = np.array([float(v) for v in m.group(1).split(',')])
    return h


def find_lines(sess, date):
    """All reflectance lines for a date (v2 preferred), with headers"""
    out = []
    for run in range(1, MAX_RUN + 1):
        for suffix in ('_v2', ''):
            name = f'f{date}t01p00r{run:02d}_refl_corrected_15m{suffix}'
            r = sess.get(BASE + name + '.hdr', timeout=(30, 120))
            if r.status_code == 200 and r.text.startswith('ENVI'):
                h = parse_header(r.text)
                assert h['interleave'] == 'bsq' and h['data type'] == '4'
                assert h['byte order'] == '0' and h['header offset'] == '0'
                h['name'], h['run'], h['date'] = name, run, date
                out.append(h)
                break
    return out


def load_kw(path):
    d = np.loadtxt(path, comments='#')
    return d[:, 0], d[:, 6]


def fit_ewt(R, wl, kw_wl, kw, window):
    """Per-pixel linear LSQ: ln R = c0 + c1 (wl - mid) - Kw EWT"""
    lo, hi, (xlo, xhi) = window
    sel = (wl >= lo) & (wl <= hi) & ~((wl >= xlo) & (wl <= xhi))
    k = np.interp(wl[sel], kw_wl, kw)
    A = np.c_[np.ones(sel.sum()), wl[sel] - wl[sel].mean(), -k]
    pinv = np.linalg.pinv(A)  # (3, nb)
    with np.errstate(invalid='ignore', divide='ignore'):
        lnR = np.log(np.where(R[sel] > 0.005, R[sel], np.nan))
    coef = np.tensordot(pinv, lnR, axes=(1, 0))
    return coef[2]


def band_nearest(wl, target):
    return int(np.argmin(np.abs(wl - target)))


def line_metrics(sess, h, transform, shape, kw_wl, kw, jobs):
    """30 m metrics for one line over the AOI (NaN outside), and the
    distance of each 30 m cell from the swath centre (in 15 m pixels)"""
    res = h['res']
    assert res == 15 and abs(transform.a) == 30
    ax0, ay1 = transform.c, transform.f
    ny15, nx15 = shape[0] * 2, shape[1] * 2
    # Line rows/cols that fall in the AOI (15 m)
    r0 = int(round((h['y0'] - ay1) / res))
    c0 = int(round((ax0 - h['x0']) / res))
    rr0, rr1 = max(r0, 0), min(r0 + ny15, int(h['lines']))
    if rr1 <= rr0:
        return None
    ns, nl = int(h['samples']), int(h['lines'])
    cc0, cc1 = max(c0, 0), min(c0 + nx15, ns)
    if cc1 <= cc0:
        return None
    wl = h['wavelength']
    need = np.flatnonzero(((wl >= 850) & (wl <= 1270)))
    need = sorted(set(need.tolist()) | {band_nearest(wl, 660),
                                         band_nearest(wl, 860),
                                         band_nearest(wl, 1240)})
    url = BASE + h['name'] + '.bin'
    nrow = rr1 - rr0

    def read(b):
        off = (b * nl * ns + rr0 * ns) * 4
        buf = get_range(sess, url, off, nrow * ns * 4)
        return b, np.frombuffer(buf, '<f4').reshape(nrow, ns)[:, cc0:cc1]

    cube = {}
    with ThreadPoolExecutor(jobs) as pool:
        for b, a in pool.map(read, need):
            cube[b] = a
    wsel = wl[need]
    R = np.stack([cube[b] for b in need]).astype(np.float32)
    bad = (R <= NODATA + 1).any(axis=0)
    R[:, bad] = np.nan

    r660, r860, r1240 = (cube[band_nearest(wl, t)].astype(np.float32)
                         for t in (660, 860, 1240))
    with np.errstate(invalid='ignore', divide='ignore'):
        m = {
            'ewt980': fit_ewt(R, wsel, kw_wl, kw, WIN_980),
            'ewt1200': fit_ewt(R, wsel, kw_wl, kw, WIN_1200),
            'ndwi': (r860 - r1240) / (r860 + r1240),
            'ndvi': (r860 - r660) / (r860 + r660),
        }
        # Continuum-removed depth at 1200 nm
        i1, i2, i0 = (np.argmin(np.abs(wsel - t)) for t in (1080, 1265, 1200))
        f = (wsel[i0] - wsel[i1]) / (wsel[i2] - wsel[i1])
        cont = R[i1] * (1 - f) + R[i2] * f
        m['bd1200'] = 1 - R[i0] / cont
    for k in m:
        m[k] = np.where(bad, np.nan, m[k]).astype(np.float32)

    # Swath-centre distance per 15 m pixel (valid pixels in each row)
    valid = ~bad
    cols = np.arange(cc0, cc1)[None, :].repeat(nrow, 0).astype(float)
    with np.errstate(invalid='ignore'):
        centre = np.nanmean(np.where(valid, cols, np.nan), axis=1)
    dist = np.where(valid, np.abs(cols - centre[:, None]), np.nan)

    # Paste into the AOI 15 m canvas, then aggregate 2 x 2 to 30 m
    def to30(a):
        canvas = np.full((ny15, nx15), np.nan, np.float32)
        canvas[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = a
        c = canvas.reshape(shape[0], 2, shape[1], 2)
        with np.errstate(invalid='ignore'):
            return np.nanmean(c, axis=(1, 3)), np.isfinite(c).sum(axis=(1, 3))
    out = {k: to30(v)[0] for k, v in m.items()}
    out['dist'], out['npx'] = to30(dist.astype(np.float32))
    return out


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path),
                default='/Volumes/Earth04/ecopro/wdts')
@click.option('-a', '--aoi', 'name', default='neon_soap_teak',
              show_default=True)
@click.option('-y', '--year', 'years', multiple=True, type=int,
              help='Years to process (default: all in DATES)')
@click.option('-j', '--jobs', default=4, show_default=True)
@click.option('--kw-table', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/wdts/aux/prospect_d_spectra.txt'))
def main(configfile, outputdir, name, years, jobs, kw_table):
    config = load_config(configfile)
    aoi = config['aois'][name]
    transform, shape = aoi_grid(aoi, config['size_m'], config['resolution'])
    kw_wl, kw = load_kw(kw_table)
    sess = session()
    years = list(years) or sorted(DATES)
    keys = ['ewt980', 'ewt1200', 'ndwi', 'bd1200', 'ndvi']
    data = {k: np.full((len(years),) + shape, np.nan, np.float32)
            for k in keys}
    src = np.full((len(years),) + shape, -1, np.int16)
    nadir = np.full((len(years),) + shape, np.nan, np.float32)
    lines_used = {}
    for i, y in enumerate(years):
        cands = []
        for rank, date in enumerate(DATES[y]):
            for h in find_lines(sess, date):
                if h['zone'] != aoi['epsg'] - 32600:
                    continue
                cands.append((rank, h))
        best = np.full(shape, np.inf)
        names = []
        for rank, h in cands:
            t = time.time()
            m = line_metrics(sess, h, transform, shape, kw_wl, kw, jobs)
            if m is None:
                continue
            score = rank * 1e6 + m['dist']
            take = (m['npx'] >= 2) & np.isfinite(m['ewt980']) & (score < best)
            if not take.any():
                continue
            names.append(h['name'])
            for k in keys:
                data[k][i][take] = m[k][take]
            src[i][take] = len(names) - 1
            nadir[i][take] = m['dist'][take]
            best[take] = score[take]
            click.echo(f'[{name}] {y} {h["name"]}: {take.sum()} cells '
                       f'({time.time() - t:.0f} s)')
        lines_used[y] = names
        cov = np.isfinite(data['ewt980'][i]).mean()
        click.echo(f'[{name}] {y}: coverage {cov:.2f}; median EWT980 '
                   f'{np.nanmedian(data["ewt980"][i]):.3f} cm')
    xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    ds = xr.Dataset(
        {**{k: (('year', 'y', 'x'), v) for k, v in data.items()},
         'source_line': (('year', 'y', 'x'), src),
         'nadir_dist': (('year', 'y', 'x'), nadir)},
        coords={'year': years, 'y': ys, 'x': xs},
        attrs={'crs': f'EPSG:{aoi["epsg"]}', 'transform': list(transform)[:6],
               'source': 'doi:10.3334/ORNLDAAC/2391',
               'nadir_dist_units': '15 m pixels from swath centre',
               **{f'lines_{y}': ','.join(v) for y, v in lines_used.items()}})
    out = outputdir / f'{name}_cwc.nc'
    ds.to_netcdf(out.with_suffix('.tmp.nc'),
                 encoding={k: {'zlib': True} for k in ds.data_vars})
    os.replace(out.with_suffix('.tmp.nc'), out)
    click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
