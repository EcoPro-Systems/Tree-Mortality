#!/usr/bin/env python
"""
Does a spaceborne-like imaging-spectroscopy (VSWIR) time series add
drought-recovery skill over a Landsat time series of the same dates,
without lidar? A spaceborne VSWIR mission delivers a time series, as
Landsat does, so this compares equal numbers of dates in the same window
on both sides.

Dates (each AVIRIS-C date with full coverage and public L2 over the AOIs,
and the clear Landsat 8 scene nearest to it on path 42):
  2013 (first drought; cycle-1 responses)
      AVIRIS-C May 3, Jun 12, Jun 26 (ORNL DAAC 2391, corrected 15 m)
      Landsat 8 May 4, Jun 5, Jun 21
  2018 (after the first drought; cycle-2 responses of the pilot AOIs)
      AVIRIS-C Jun 22, Aug 28 (ORNL DAAC 2154: the Aug 28 lines are in no
      other collection, so both dates come from it and share one
      processing chain)
      Landsat 8 Jun 19, Aug 22

Subcommands:
  refl      Cache one date's reflectance on the 30 m AOI grid:
            wdts/sim/<aoi>_refl_<yymmdd>_<product>.nc. Products: wdts
            (2391, as proto_spaceborne_sim.py refl) and avc (2154, ENVI
            BIL, orthocorrected 14-15 m, possibly rotated: the long
            Aug 2018 lines run 22-23 degrees off north). 2154 rows that
            intersect the AOI are read with HTTP range requests, and each
            30 m pixel is the mean of the source pixels whose centres fall
            in it. Where lines overlap, the nearest-nadir line is kept, or
            (--map-year) the line that year's 2403 trait map used.
  emulate   Trait retrievals for every date from one emulator: PLSR from
            log reflectance to the 2403 map of the first date, fitted out
            of fold over 1 km blocks on that date (pixels from the map's
            own line), and each fold's models applied to every date's
            pixels in its held-out blocks. Spectra of later dates are
            interpolated onto the first date's band centres (native) or
            put through each configuration's band responses from their own
            wavelengths. Noise draws are seeded per AOI and date. EWT (the
            Beer-Lambert fit) per date. Traits and EWT are cross-track
            normalized per date along that date's lines. Writes 90 m cell
            tables OUTPUTDIR/cells_<aoi>_<leg>.csv.gz (<config>@<date>:<x>)
            and emulator_<leg>.csv (first-date out-of-fold R² against the
            map).
  compare   Paired 1 km (and 5 km) block bootstrap on one cell table, over
            Env+S (no lidar), of:
              V:<c>    the VSWIR time series of a configuration (every
                       date's traits and EWT, plus the change last - first)
              L        the matched Landsat 8 scenes (bands, NDVI, NDMI,
                       NBR, NIRv per scene, plus the same change)
              V1, L1   one date each (the first VSWIR date, its nearest
                       Landsat scene)
              l8multi  the year's June and Jul-Sep Landsat 8 composites
              M1       the 2403 trait maps of the first date (2013) or the
                       per-date z-scored June 2018 maps (2018)
            Contrasts: V:<c> - L (balanced); L+V:<c> - L (VSWIR beyond
            Landsat); V1:<c> - L1 (one date each); what the extra dates buy
            each side (V:<c> - V1:<c>, L - L1); V:<c> - l8multi (VSWIR
            dates against full-season composites: unbalanced, deployment
            context only); the maps beyond Landsat (L+M1 - L, L1+M1 - L1).
            Cells with every date on both sides (and both composites).
            --structure adds the no-lidar stack of proto_spaceborne_only.py
            (2013 leg) with each time series as the trait block.

The 2013 Jul-Sep composite is one of the years the stress-response slope
(ndmi_sens) is fitted on, and the 2013 scenes fall inside the drought, so
stress response is partly circular for the 2013 leg. The cycle-2 responses
are Landsat 8/9 with a 2017-19 baseline: 2018 Landsat inputs share sensor,
processing and baseline years with them.

    python proto_spaceborne_ts.py refl -a neon_soap_teak -a sierra_nf \\
        --date 130503
    python proto_spaceborne_ts.py refl -a neon_soap_teak -a sierra_nf \\
        --date 180622 --product avc --map-year 2018
    python proto_spaceborne_ts.py emulate $E/hls_results/spaceborne_ts \\
        -a neon_soap_teak --leg 2013 --map-year 2013 \\
        --date 130612:refl2013 --date 130503:refl_130503_wdts \\
        --date 130626:refl_130626_wdts \\
        --emit-uncert $E/emit/EMIT_L2A_RFLUNCERT_..._006.nc
    python proto_spaceborne_ts.py compare $E/hls_results/spaceborne_ts \\
        -a neon_soap_teak --leg 2013 --landsat 20130605 \\
        --landsat 20130504 --landsat 20130621
"""
import os
import re
import time
import zlib
import click
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

import response_common as rc
import proto_forward_pilot as fp
import proto_response_transfer as prt
from fetch_wdts_cwc import (MAX_RUN, WIN_980, session, find_lines, get_range,
                            load_kw, fit_ewt)
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import TRAITS, build, trait_layers, structure_cells
from proto_trait_dynamics import crosstrack_normalize
from proto_spaceborne_sim import (SIM, KW, MAX_TRAIN, good_bands, emit_noise,
                                  configure, degrade, line_cube)
from proto_retrieval_transfer import resample_matrix
from proto_spaceborne_only import (aoi_dirs, landsat_block, save, gains_msg,
                                   spaceborne_stack)
from proto_aviris5_bridge import AVC_BASE, VALID

AVC_SUFFIX = '_corr_v1k1_img'
ROWS = 32  # 2154 rows per range request (all bands of a row are adjacent)
CONFIGS = ['native', 'emit', 'sbg_lo']
TARGETS = {2013: ['ndmi_recovery', 'nirv_recovery', 'ndmi_sens'],
           2018: ['ndmi_recovery', 'nirv_recovery']}
KEYS = ['cell_row', 'cell_col']
LANDSAT_VARS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi',
                'ndmi', 'nbr', 'nirv']


@click.group()
def cli():
    pass


# ---------------------------------------------------------------- refl

def avc_header(sess, name):
    r = sess.get(AVC_BASE + name + '.hdr', timeout=(30, 120))
    if r.status_code != 200 or not r.text.startswith('ENVI'):
        return None
    text = r.text

    def field(k):
        m = re.search(rf'^{k}\s*=\s*(\S+)', text, re.M)
        return m.group(1) if m else None
    assert field('interleave') == 'bil' and field('data type') == '4'
    assert field('byte order') == '0' and field('header offset') in (None,
                                                                     '0')
    mi = [v.strip() for v in re.search(r'map info\s*=\s*\{([^}]*)\}',
                                       text).group(1).split(',')]
    assert float(mi[1]) == 1 and float(mi[2]) == 1, 'reference pixel'
    rot = [float(v.split('=')[1]) for v in mi if v.startswith('rotation')]
    wl = re.search(r'wavelength\s*=\s*\{([^}]*)\}', text).group(1)
    return dict(name=name, samples=int(field('samples')),
                lines=int(field('lines')), bands=int(field('bands')),
                x0=float(mi[3]), y0=float(mi[4]), res=float(mi[5]),
                zone=int(mi[7]), rot=rot[0] if rot else 0.0,
                wavelength=np.array([float(v) for v in wl.split(',')]))


def avc_lines(sess, date):
    """All 2154 reflectance lines of a date, with headers"""
    out = []
    for run in range(1, MAX_RUN + 1):
        h = avc_header(sess, f'f{date}t01p00r{run:02d}{AVC_SUFFIX}')
        if h is not None:
            h['run'] = run
            out.append(h)
    return out


def avc_affine(h):
    """Pixel -> map transform of an ENVI map info with rotation (degrees,
    counterclockwise), as GDAL reads it"""
    th = np.deg2rad(h['rot'])
    c, s, r = np.cos(th), np.sin(th), h['res']
    return Affine(r * c, r * s, h['x0'], r * s, -r * c, h['y0'])


def avc_cube(sess, h, transform, shape, jobs):
    """All bands of one 2154 line on the 30 m AOI grid (mean of the source
    pixels whose centres fall in each 30 m pixel) and each 30 m pixel's
    mean distance from the swath centre (source pixels)"""
    A = avc_affine(h)
    inv = ~A
    H, W = shape
    xs = [transform.c, transform.c + W * transform.a]
    ys = [transform.f, transform.f + H * transform.e]
    cr = np.array([inv * (x, y) for x in xs for y in ys])
    ns, nl, nb = h['samples'], h['lines'], h['bands']
    r0, r1 = max(int(cr[:, 1].min()) - 2, 0), min(int(cr[:, 1].max()) + 3, nl)
    c0, c1 = max(int(cr[:, 0].min()) - 2, 0), min(int(cr[:, 0].max()) + 3, ns)
    if r1 <= r0 or c1 <= c0:
        return None
    url = AVC_BASE + h['name'] + '.bin'
    ref = int(np.argmin(np.abs(h['wavelength'] - 860)))
    S = np.zeros((nb, H * W), np.float32)
    N = np.zeros((nb, H * W), np.uint16)
    Dsum = np.zeros(H * W, np.float64)
    Dn = np.zeros(H * W, np.uint16)
    cols = np.arange(c0, c1)

    def read(a):
        b = min(a + ROWS, r1)
        buf = get_range(sess, url, a * nb * ns * 4, (b - a) * nb * ns * 4)
        X = np.frombuffer(buf, '<f4').reshape(b - a, nb, ns)[:, :, c0:c1]
        return a, b, X.transpose(1, 0, 2)

    with ThreadPoolExecutor(jobs) as pool:
        for a, b, V in pool.map(read, range(r0, r1, ROWS)):
            rr, cc = np.meshgrid(np.arange(a, b) + 0.5, cols + 0.5,
                                 indexing='ij')
            x = A.c + A.a * cc + A.b * rr
            y = A.f + A.d * cc + A.e * rr
            pr = np.floor((y - transform.f) / transform.e).astype(np.int64)
            pc = np.floor((x - transform.c) / transform.a).astype(np.int64)
            ok = (V >= VALID[0]) & (V <= VALID[1])
            v860 = ok[ref] & (V[ref] > 0.005)  # 0-filled swath edges
            with np.errstate(invalid='ignore'):
                centre = np.nanmean(np.where(v860, cc, np.nan), axis=1)
            dist = np.abs(cc - centre[:, None])
            m = v860 & (pr >= 0) & (pr < H) & (pc >= 0) & (pc < W)
            if not m.any():
                continue
            idx = (pr * W + pc)[m]
            lo, n = idx.min(), idx.max() - idx.min() + 1
            idx = idx - lo
            Dsum[lo:lo + n] += np.bincount(idx, dist[m], n)
            Dn[lo:lo + n] += np.bincount(idx, minlength=n).astype(np.uint16)
            for k in range(nb):
                okb = ok[k][m]
                S[k, lo:lo + n] += np.bincount(idx[okb], V[k][m][okb], n)
                N[k, lo:lo + n] += np.bincount(idx[okb], minlength=n) \
                    .astype(np.uint16)
    if not Dn.any():
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        cube = np.where(N > 0, S / N, np.nan).astype(np.float32)
        dist = np.where(Dn > 0, Dsum / Dn, np.nan)
    return cube.reshape(nb, H, W), dist.reshape(H, W)


@cli.command()
@click.option('-a', '--aoi', 'names', multiple=True, required=True)
@click.option('--date', required=True, help='Acquisition date, YYMMDD')
@click.option('--product', type=click.Choice(['wdts', 'avc']),
              default='wdts', show_default=True,
              help='wdts: ORNL DAAC 2391; avc: ORNL DAAC 2154')
@click.option('--map-year', type=int,
              help="Prefer the line this year's 2403 trait map used")
@click.option('-j', '--jobs', default=8, show_default=True)
def refl(names, date, product, map_year, jobs):
    """Cache one date's reflectance mosaic on the 30 m AOI grid"""
    SIM.mkdir(parents=True, exist_ok=True)
    sess = session()
    lines = find_lines(sess, date) if product == 'wdts' else \
        avc_lines(sess, date)
    click.echo(f'{date} {product}: {len(lines)} lines')
    for name in names:
        transform, shape, epsg = rc.aoi_info(name)
        fid = np.zeros(shape, np.int16)
        if map_year:
            fid = xr.open_dataset(rc.E / 'wdts' / f'{name}_traits.nc') \
                .flight_id.sel(year=map_year).values
        mos, wl, names_ = None, None, []
        best = np.full(shape, np.inf)
        src = np.full(shape, -1, np.int16)
        run = np.full(shape, -1, np.int16)
        for h in lines:
            if h['zone'] != epsg - 32600:
                continue
            t = time.time()
            out = (line_cube(sess, h, transform, shape, jobs)
                   if product == 'wdts' else
                   avc_cube(sess, h, transform, shape, jobs))
            if out is None:
                continue
            cube, dist = out
            score = ((fid > 0) & (fid != h['run'])) * 1e5 + dist
            take = np.isfinite(dist) & (score < best)
            if not take.any():
                continue
            if mos is None:
                wl = h['wavelength']
                mos = np.full(cube.shape, np.nan, np.float32)
            assert np.allclose(h['wavelength'], wl)
            mos[:, take] = cube[:, take]
            best[take] = score[take]
            src[take] = len(names_)
            run[take] = h['run']
            names_.append(h['name'])
            click.echo(f'[{name}] {h["name"]}: {take.sum()} px '
                       f'({time.time() - t:.0f} s)')
            del cube
        if mos is None:
            click.echo(f'[{name}] no lines over the AOI')
            continue
        xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
        ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
        doi = {'wdts': '2391', 'avc': '2154'}[product]
        ds = xr.Dataset({'refl': (('wavelength', 'y', 'x'), mos),
                         'source_line': (('y', 'x'), src),
                         'source_line_run': (('y', 'x'), run)},
                        coords={'wavelength': wl, 'y': ys, 'x': xs},
                        attrs={'crs': f'EPSG:{epsg}',
                               'transform': list(transform)[:6],
                               'date': date,
                               'source': f'doi:10.3334/ORNLDAAC/{doi}',
                               'lines': ','.join(names_)})
        out = SIM / f'{name}_refl_{date}_{product}.nc'
        ds.to_netcdf(out.with_suffix('.tmp.nc'),
                     encoding={'refl': {'zlib': True, 'dtype': 'int16',
                                        'scale_factor': 1e-4,
                                        '_FillValue': -32768},
                               'source_line': {'zlib': True},
                               'source_line_run': {'zlib': True}})
        os.replace(out.with_suffix('.tmp.nc'), out)
        click.echo(f'wrote {out}')


# ------------------------------------------------------------- emulate

def predict_dates(Xs, idxs, Y, block, n_comp, seed=0):
    """Fit PLSR per trait on the first date out of fold over blocks; each
    fold's models predict every date's pixels in that fold's held-out
    blocks (blocks with no first-date training pixels: fold 0, which never
    saw them). Returns one (n_pix, n_traits) array per date."""
    has = np.isfinite(Y[idxs[0]]).any(1)
    X0, pix0 = Xs[0][has], idxs[0][has]
    nc = min(n_comp, X0.shape[1])
    rng = np.random.default_rng(seed)
    fold_of, models = {}, []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=rc.N_SPLITS).split(
            X0, groups=block[pix0])):
        fold_of.update({b: f for b in np.unique(block[pix0[te]])})
        if len(tr) > MAX_TRAIN:
            tr = rng.choice(tr, MAX_TRAIN, replace=False)
        ms = []
        for j in range(Y.shape[1]):
            ok = np.isfinite(Y[pix0[tr], j])
            ms.append(PLSRegression(n_components=nc, scale=True).fit(
                X0[tr][ok], Y[pix0[tr], j][ok]))
        models.append(ms)
    out = []
    for X, idx in zip(Xs, idxs):
        f_of = pd.Series(block[idx]).map(fold_of).fillna(0).astype(int) \
            .values
        P = np.full((len(block), Y.shape[1]), np.nan, np.float32)
        for f, ms in enumerate(models):
            sel = f_of == f
            if sel.any():
                for j, m in enumerate(ms):
                    P[idx[sel], j] = m.predict(X[sel]).ravel()
        out.append(P)
    return out


def load_date(name, key, valid):
    ds = xr.open_dataset(SIM / f'{name}_{key}.nc')
    R = ds.refl.values.astype(np.float32)
    wl = ds.wavelength.values
    if 'source_line_run' in ds:
        run = ds.source_line_run.values
    else:  # older caches: the run from the line names
        runs = np.array([int(re.search(r'p00r(\d+)', n).group(1))
                         for n in ds.attrs['lines'].split(',')] + [-1])
        run = runs[ds.source_line.values]  # source_line -1 -> -1
    i860 = int(np.argmin(np.abs(wl - 860)))
    ok = valid & np.isfinite(R[i860]) & (run > 0)
    # bands missing from any line (the non-_v2 lines have no 1323 and
    # 1333 nm data) leave the usable set: native spectra are interpolated
    # across them, band responses renormalized
    good = good_bands(wl) & (np.isfinite(R[:, ok]).mean(1) > 0.99)
    return dict(R=R, wl=wl, run=run, ok=ok, good=good)


@cli.command('emulate')
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'names', multiple=True, required=True)
@click.option('--leg', required=True, help='Name of this set of dates')
@click.option('--map-year', type=int, required=True,
              help='Year of the 2403 map the emulator is fitted to')
@click.option('--date', 'dates', multiple=True, required=True,
              help='LABEL:CACHE (wdts/sim/<aoi>_<CACHE>.nc); the first is '
                   'the date the emulator is fitted on')
@click.option('--last-year', type=int, default=2019, show_default=True,
              help='Disturbance mask end (2019: cycle 1; 2025: cycle 2)')
@click.option('--emit-uncert', type=click.Path(path_type=Path, exists=True),
              required=True, help='EMIT_L2A_RFLUNCERT file (RFL beside it)')
@click.option('--config', 'configs', multiple=True, default=CONFIGS,
              show_default=True)
@click.option('--n-comp', default=10, show_default=True)
@click.option('--scale', default=3, show_default=True)
def emulate_cmd(outputdir, names, leg, map_year, dates, last_year,
                emit_uncert, configs, n_comp, scale):
    """One emulator, every date: per-date traits and EWT, 90 m cells"""
    outputdir.mkdir(parents=True, exist_ok=True)
    rfl = Path(str(emit_uncert).replace('RFLUNCERT', 'RFL'))
    emit = emit_noise(emit_uncert, rfl)
    kw_wl, kw = load_kw(KW)
    specs = [s.split(':') for s in dates]
    stats = []
    for name in names:
        t0 = time.time()
        transform, shape, _ = rc.aoi_info(name)
        env = rc.open_env(name)
        elev = env.elevation.values
        valid = rc.undisturbed(env, last_year)
        tr = xr.open_dataset(rc.E / 'wdts' / f'{name}_traits.nc') \
            .sel(year=map_year)
        fid = tr.flight_id.values
        dd = {label: load_date(name, key, valid) for label, key in specs}
        labels = [label for label, _ in specs]
        first = dd[labels[0]]
        # the emulator learns from pixels whose spectra come from the line
        # the map used there
        train = first['ok'] & (fid > 0) & (first['run'] == fid)
        Y = np.stack([tr[f'{t}_mean'].values for t in TRAITS], -1) \
            .reshape(-1, len(TRAITS)).astype(np.float32)
        Y[~train.ravel()] = np.nan
        ref_wl = first['wl'][first['good']]
        rows, cols = np.indices(shape)
        block = ((rows // 33) * 10000 + cols // 33).ravel()  # ~1 km
        layers = {}
        for cfg in configs:
            Xs, idxs, ewts = [], [], []
            for label in labels:
                x = dd[label]
                if cfg == 'native':
                    W = np.zeros((len(ref_wl), len(x['wl'])))
                    W[:, x['good']] = resample_matrix(x['wl'][x['good']],
                                                      ref_wl)
                    D, owl = degrade(x['R'], W, None, False), ref_wl
                else:
                    W, owl, noise, agg = configure(cfg, x['wl'], x['good'],
                                                   emit)
                    D = degrade(x['R'], W, noise, agg,
                                zlib.crc32(f'{name}:{label}'.encode()))
                X = np.log(np.clip(D.reshape(len(D), -1).T, 1e-3, None))
                idx = np.flatnonzero(x['ok'].ravel() & np.isfinite(X).all(1))
                Xs.append(X[idx])
                idxs.append(idx)
                with np.errstate(invalid='ignore', divide='ignore'):
                    ewts.append(np.where(x['ok'], fit_ewt(
                        D, owl, kw_wl, kw, WIN_980), np.nan)
                        .astype(np.float32))
                del D, X
            assert len({X.shape[1] for X in Xs}) == 1
            P = predict_dates(Xs, idxs, Y, block, n_comp)
            ok = train.ravel() & np.isfinite(P[0]).all(1)
            for j, t in enumerate(TRAITS):
                okj = ok & np.isfinite(Y[:, j])
                stats.append(dict(aoi=name, leg=leg, config=cfg, trait=t,
                                  n=int(okj.sum()),
                                  r2=rc.wr2(Y[okj, j], P[0][okj, j])))
            for label, Pd, ewt in zip(labels, P, ewts):
                run = dd[label]['run']
                for j, t in enumerate(TRAITS):
                    layers[f'{cfg}@{label}:{t}'] = crosstrack_normalize(
                        Pd[:, j].reshape(shape), run, elev, transform)
                layers[f'{cfg}@{label}:ewt'] = crosstrack_normalize(
                    ewt, run, elev, transform)
            s = {x['trait']: x['r2'] for x in stats
                 if x['aoi'] == name and x['config'] == cfg}
            click.echo(f'[{name}] {cfg:7s} {Xs[0].shape[1]:3d} bands; '
                       + ', '.join(f'{label} {len(i)} px' for label, i in
                                   zip(labels, idxs)) + '; first-date R² ' +
                       '  '.join(f'{t} {s[t]:.2f}' for t in
                                 ('Nitrogen', 'LMA', 'Lignin', 'Cellulose')))
            del Xs, P
        cells = rc.cell_table(layers, valid, scale)
        cells.to_csv(outputdir / f'cells_{name}_{leg}.csv.gz', index=False,
                     float_format='%.6g')
        click.echo(f'[{name}] {len(cells)} cells ({time.time() - t0:.0f} s)')
        save(stats, outputdir / f'emulator_{leg}.csv')


# ------------------------------------------------------------- compare

def leg_table(aoi, year, scale):
    """Cells with responses, Env+S and the year's trait maps (M1); (table,
    Env+S columns, valid mask)"""
    if year == 2013:
        rdir, ddir = aoi_dirs(aoi)
        d = build(aoi, scale, rdir, ddir)
        d = d.drop(columns=[c for c in d if c.startswith(('T_', 'W_'))])
        valid = rc.undisturbed(rc.open_env(aoi), 2019)
        t = trait_layers(aoi, valid, scale, None, 2013)
        t = t.rename(columns={f'T_{x}_2013': f'maps:{x}' for x in TRAITS})
        d = d.merge(t[KEYS + [f'maps:{x}' for x in TRAITS]], on=KEYS,
                    how='left')
        return d, ENV + S_WALL, valid
    # cycle 2, pilot AOIs only (proto_forward_pilot.build refuses held-out
    # AOIs); the June 2018 maps, z-scored per date
    d, _ = fp.build(aoi, scale, False)
    d = d.rename(columns={f'T_{x}_18': f'maps:{x}' for x in TRAITS})
    d = d.drop(columns=[c for c in d if c.startswith('T_')])
    return d, prt.ENV + S_WALL, rc.undisturbed(rc.open_env(aoi), 2025)


def timeseries_table(outputdir, aoi, leg, year, scale, scenes, configs,
                     only=()):
    d, es, valid = leg_table(aoi, year, scale)
    cells = pd.read_csv(outputdir / f'cells_{aoi}_{leg}.csv.gz')
    cells = cells.drop(columns=[c for c in cells
                                if c.startswith(('n_px', 'block'))])
    d = d.merge(cells, on=KEYS, how='inner')
    for s in scenes:
        t = landsat_block(aoi, valid, scale, 'l8raw', s)
        t = t.rename(columns={f'l8raw:{v}': f'l8@{s}:{v}'
                              for v in LANDSAT_VARS})
        d = d.merge(t[KEYS + [f'l8@{s}:{v}' for v in LANDSAT_VARS]],
                    on=KEYS, how='left')
    t = landsat_block(aoi, valid, scale, 'l8multi', None, year)
    multi = [c for c in t if c.startswith('l8multi:')]
    d = d.merge(t[KEYS + multi], on=KEYS, how='left')
    # the time series' dates, in the order given at emulate
    labels = [x for x in dict.fromkeys(c.split('@')[1].split(':')[0]
                                       for c in cells if '@' in c)
              if not only or x in only]
    assert not only or len(labels) == len(only), 'unknown --vswir-date'
    first, lo, hi = labels[0], min(labels), max(labels)
    cols, need = {}, []
    for c in configs:
        xs = [x.split(':')[1] for x in cells if x.startswith(f'{c}@{first}:')]
        for x in xs:
            d[f'{c}@chg:{x}'] = d[f'{c}@{hi}:{x}'] - d[f'{c}@{lo}:{x}']
        cols[f'V:{c}'] = [f'{c}@{lab}:{x}' for lab in labels + ['chg']
                          for x in xs]
        cols[f'V1:{c}'] = [f'{c}@{first}:{x}' for x in xs]
        need += [f'{c}@{lab}:Nitrogen' for lab in labels]
    s_lo, s_hi = min(scenes), max(scenes)
    for v in LANDSAT_VARS:
        d[f'l8@chg:{v}'] = d[f'l8@{s_hi}:{v}'] - d[f'l8@{s_lo}:{v}']
    cols['L'] = [f'l8@{s}:{v}' for s in scenes + ['chg']
                 for v in LANDSAT_VARS]
    cols['L1'] = [f'l8@{scenes[0]}:{v}' for v in LANDSAT_VARS]
    cols['l8multi'] = multi
    cols['M1'] = [f'maps:{x}' for x in TRAITS]
    need += [f'l8@{s}:nir' for s in scenes] + ['l8multi:nir_jun',
                                                'l8multi:nir_sum']
    n0 = len(d)
    d = d[d[need].notna().all(1)].copy()
    click.echo(f'[{aoi}] {len(d)} of {n0} cells with every date '
               f'(VSWIR {", ".join(labels)}; Landsat {", ".join(scenes)})')
    return d, es, cols, labels


def ts_ladder(d, es, cols, configs, targets, n_boot):
    fs = {'Env+S': es}
    for k in ['L', 'L1', 'l8multi', 'M1']:
        fs[k] = es + cols[k]
    for c in configs:
        fs[f'V:{c}'] = es + cols[f'V:{c}']
        fs[f'V1:{c}'] = es + cols[f'V1:{c}']
        fs[f'L+V:{c}'] = es + cols['L'] + cols[f'V:{c}']
    fs['L+M1'] = es + cols['L'] + cols['M1']
    fs['L1+M1'] = es + cols['L1'] + cols['M1']
    pairs = [('Env+S', k) for k in fs if k != 'Env+S']
    for c in configs:
        pairs += [('L', f'V:{c}'), ('L', f'L+V:{c}'), ('L1', f'V1:{c}'),
                  (f'V1:{c}', f'V:{c}'), ('l8multi', f'V:{c}')]
    pairs += [('L1', 'L'), ('L', 'L+M1'), ('L1', 'L1+M1')]
    rows = []
    for t in targets:
        sub = d[d[t].notna() & np.isfinite(d[t])]
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                         pairs=pairs, extra_blocks=('block5000',))
        rows += r
        g = {(x['features'], x['compare']): x for x in r
             if x['compare'] and x['boot_blocks'] == 'block1000'}
        base = [x for x in r if x['features'] == 'Env+S'
                and not x['compare']][0]
        click.echo(f'  {t} n={len(sub)} Env+S {base["r2"]:.3f}')
        click.echo(gains_msg(r, [(k, 'Env+S') for k in fs if k != 'Env+S'],
                             []))
        for a, b in pairs:
            if a == 'Env+S':
                continue
            x = g[(b, a)]
            click.echo(f'    {b:16s} - {a:10s} {x["r2"]:+.3f} '
                       f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]')
    return rows


@cli.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--leg', required=True)
@click.option('--year', type=int, help='Year of the dates (default: --leg)')
@click.option('--landsat', 'scenes', multiple=True, required=True,
              help='Landsat 8 scene dates (YYYYMMDD) matched to the VSWIR '
                   'dates, in the same order (the first is L1)')
@click.option('--config', 'configs', multiple=True, default=CONFIGS,
              show_default=True)
@click.option('--vswir-date', 'only', multiple=True,
              help='Use only these VSWIR dates (labels; default all; V1 '
                   'stays the date the emulator was fitted on)')
@click.option('--structure', 'sources', multiple=True,
              help='Lidar source(s) for the no-lidar stack (2013 leg)')
@click.option('--tag', default='', help='Suffix of the output files')
@click.option('-t', '--target', 'targets', multiple=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
def compare(outputdir, aois, leg, year, scenes, configs, only, sources,
            tag, targets, scale, n_boot):
    """VSWIR time series vs the date-matched Landsat time series"""
    year = year or int(leg)
    targets = list(targets) or TARGETS[year]
    scenes = list(scenes)
    rows, stk = [], []
    for aoi in aois:
        d, es, cols, labels = timeseries_table(outputdir, aoi, leg, year,
                                               scale, scenes, configs, only)
        tags = dict(aoi=aoi, leg=leg, vswir=','.join(labels),
                    landsat=','.join(scenes), scale_m=rc.RES * scale)
        click.echo(f'[{aoi}] over Env+S, no lidar')
        rows += [dict(x, **tags) for x in ts_ladder(d, es, cols, configs,
                                                     targets, n_boot)]
        save(rows, outputdir / f'timeseries_{leg}{tag}.csv')
        for src in sources:
            sub = structure_cells(d, aoi, scale, src)
            if len(sub) < 500:
                continue
            click.echo(f'[{aoi}] spaceborne stack, {src}: {len(sub)} cells')
            sc = {c: cols[f'V:{c}'] for c in configs}
            sc['L'] = cols['L']
            stk += [dict(x, source=src, **tags) for x in spaceborne_stack(
                sub, sc, src, targets, ['L'], n_boot)]
            save(stk, outputdir / f'stack_{leg}{tag}.csv')


if __name__ == '__main__':
    cli()
