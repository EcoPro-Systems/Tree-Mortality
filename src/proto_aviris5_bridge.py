#!/usr/bin/env python
"""
Do AVIRIS-Classic and AVIRIS-5 give the same reflectance and canopy water
over the same forest on the same day? July 17, 2025, NEON SOAP/TEAK (and
sierra_nf), from the public L2 collections:

  AVIRIS-C  ORNL DAAC 2154, AVIRIS-Classic L2 reflectance: ENVI BIL, 14.5 m
            orthocorrected, 224 bands. Only the AOI rows are read (HTTP
            range requests; in BIL a row holds every band).
  AVIRIS-5  ORNL DAAC 2484, AV5 L2A OE reflectance: orthorectified NetCDF
            (downloaded whole into $E/aviris_bridge/).

Values outside VALID are set to NaN before averaging (the 2025
AVIRIS-C L2 has isolated non-physical NIR values). Both are area-averaged
onto the 30 m AOI grid (rasterio reproject), and
AVIRIS-5 is convolved to the AVIRIS-C bands (Gaussian responses with the
AVIRIS-C FWHM). Where several lines of one sensor cover a cell, the
nearest-nadir one is kept. On undisturbed forest 90 m cells covered by
both: per-band bias, RMSE and Spearman ρ; NDVI, NDWI and EWT (the
Beer-Lambert fit of fetch_wdts_cwc.py, each sensor on its own bands).

Traits (--traits): the 2403 PLSR coefficients are not public, so one
emulator per AOI (as in proto_spaceborne_sim.py: PLSR from log reflectance
to the 2403 maps) is fitted on the June 2018 AVIRIS-C reflectance cache
(wdts/sim/<aoi>_refl2018.nc, the most recent year with both spectra and a
trait map; pixels from the line the map used) and applied unchanged to
both 2025 mosaics, each interpolated onto the 2018 band centres (AVIRIS-5
after convolution to the AVIRIS-C bands). Both 2025 inputs pass through
the same emulator, so their agreement measures how consistently the two
sensors support one retrieval, not the retrieval's accuracy. Two input
variants: raw (each mosaic as delivered) and matched (each mosaic's per-
band log reflectance rescaled to the mean and SD of the 2018 training
pixels over its own valid forest pixels: a scene-level calibration that
removes per-band level and gain offsets but keeps each sensor's spatial
pattern). Emulators with 5, 10 and 25 PLSR components (--n-comp) test
whether disagreement comes from fine spectral features that differ
between the two processing chains. Reported per number of components,
variant and trait on the same 90 m cells: Spearman ρ between sensors, the
level offset (median AVIRIS-5 minus AVIRIS-C, in SD of the AVIRIS-C
values), each sensor's ρ with the 2018 map, and the emulator's out-of-
fold R² in 2018 (1 km blocks).

Outputs in OUTPUTDIR: bands_<aoi>.csv, indices_<aoi>.csv,
cells_<aoi>.csv, bridge_<aoi>.png and (--traits) traits_<aoi>.csv.

    python proto_aviris5_bridge.py $E/hls_results/aviris5_bridge \
        -a neon_soap_teak -a sierra_nf --traits
"""
import re
import click
import numpy as np
import pandas as pd
import requests
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from affine import Affine
from pathlib import Path
from rasterio.warp import reproject, Resampling
from scipy.stats import spearmanr

import response_common as rc
from fetch_wdts_cwc import (WIN_980, get_range, parse_header, load_kw,
                            fit_ewt)
from proto_spaceborne_sim import (SIM, N_COMP, MAX_TRAIN, good_bands,
                                  srf_matrix, emulate)

AVC_BASE = ('https://data.ornldaac.earthdata.nasa.gov/protected/aviris/'
            'AVIRIS-Classic_L2_Reflectance/data/')
DATA = rc.E / 'aviris_bridge'
LINES = {'neon_soap_teak': {'avc': ['f250717t01p00r08', 'f250717t01p00r09'],
                            'av5': ['AV520250717t171949_006',
                                    'AV520250717t173810_002']},
         'sierra_nf': {'avc': ['f250717t01p00r09', 'f250717t01p00r10',
                               'f250717t01p00r11', 'f250717t01p00r12',
                               'f250717t01p00r13'],
                       'av5': ['AV520250717t175725_004',
                               'AV520250717t181601_004',
                               'AV520250717t183638_003',
                               'AV520250717t181601_005',
                               'AV520250717t175725_003']}}
KW = rc.E / 'wdts' / 'aux' / 'prospect_d_spectra.txt'
ROWS = 64  # rows per range request
VALID = (-0.05, 1.5)  # physical reflectance; outside is set to NaN
TRAIN_YEAR = 2018
FOCUS = ['Nitrogen', 'LMA', 'Lignin', 'Cellulose']
VARIANTS = ['raw', 'matched']


def warp_cube(cube, src_t, src_epsg, transform, shape, epsg):
    """(nb, h, w) -> area average on the AOI grid (NaN outside)"""
    out = np.full((cube.shape[0],) + shape, np.nan, np.float32)
    reproject(cube, out, src_transform=src_t, src_crs=f'EPSG:{src_epsg}',
              src_nodata=np.nan, dst_transform=transform,
              dst_crs=f'EPSG:{epsg}', dst_nodata=np.nan,
              resampling=Resampling.average)
    return out


def nadir_dist(valid):
    """Distance of each valid pixel from the centre of its row's valid
    run (in pixels): a nadir proxy for north-up swaths"""
    cols = np.arange(valid.shape[1])[None, :].repeat(valid.shape[0], 0)
    with np.errstate(invalid='ignore'):
        centre = np.nanmean(np.where(valid, cols, np.nan), axis=1)
    return np.where(valid, np.abs(cols - centre[:, None]), np.nan)


def read_avc(sess, line, transform, shape, epsg):
    hdr = sess.get(AVC_BASE + f'{line}_rfl.hdr', timeout=(30, 120)).text
    h = parse_header(hdr.replace('interleave = bil', 'interleave = bsq'))
    fw = re.search(r'fwhm\s*=\s*\{([^}]*)\}', hdr)
    fwhm = np.array([float(v) for v in fw.group(1).split(',')])
    ns, nl, nb = int(h['samples']), int(h['lines']), int(h['bands'])
    res = h['res']
    assert h['zone'] == epsg - 32600, 'AVIRIS-C line in another UTM zone'
    y1 = transform.f
    y0 = y1 + shape[0] * transform.e
    r0 = max(int(np.floor((h['y0'] - y1) / res)) - 2, 0)
    r1 = min(int(np.ceil((h['y0'] - y0) / res)) + 2, nl)
    if r1 <= r0:
        return None
    url = AVC_BASE + f'{line}_rfl.bin'
    cube = np.empty((nb, r1 - r0, ns), np.float32)
    for a in range(r0, r1, ROWS):
        b = min(a + ROWS, r1)
        buf = get_range(sess, url, a * nb * ns * 4, (b - a) * nb * ns * 4)
        cube[:, a - r0:b - r0] = np.frombuffer(buf, '<f4').reshape(
            b - a, nb, ns).transpose(1, 0, 2)
    cube[(cube < VALID[0]) | (cube > VALID[1])] = np.nan
    src_t = Affine(res, 0, h['x0'], 0, -res, h['y0'] - r0 * res)
    dist = nadir_dist(np.isfinite(cube[int(np.argmin(np.abs(
        h['wavelength'] - 860)))]))
    out = warp_cube(cube, src_t, epsg, transform, shape, epsg)
    d = warp_cube(dist[None].astype(np.float32), src_t, epsg, transform,
                  shape, epsg)[0]
    return h['wavelength'], fwhm, out, d


def read_av5(line, transform, shape, epsg, chunk=48):
    """AV5 L2A orthorectified NetCDF (group reflectance: wavelength x
    northing x easting, WGS84 UTM), warped in band chunks to bound
    memory"""
    import netCDF4
    f = next(DATA.glob(f'{line}_L2A_OE_*_RFL_ORT.nc'))
    nc = netCDF4.Dataset(f)
    g = nc.groups['reflectance']
    wl, fwhm = g['wavelength'][:].data, g['fwhm'][:].data
    x, y = nc['easting'][:].data, nc['northing'][:].data
    src_epsg = int(re.findall(r'AUTHORITY\["EPSG","(\d+)"\]',
                              nc['transverse_mercator'].crs_wkt)[-1])
    rx, ry = float(x[1] - x[0]), float(y[1] - y[0])
    src_t = Affine(rx, 0, x[0] - rx / 2, 0, ry, y[0] - ry / 2)
    var = g['reflectance']
    var.set_auto_mask(False)
    out = np.full((len(wl),) + shape, np.nan, np.float32)
    ref = int(np.argmin(np.abs(wl - 860)))
    dist = None
    for b0 in range(0, len(wl), chunk):
        a = var[b0:b0 + chunk].astype(np.float32)
        a[(a < VALID[0]) | (a > VALID[1])] = np.nan
        out[b0:b0 + chunk] = warp_cube(a, src_t, src_epsg, transform, shape,
                                       epsg)
        if b0 <= ref < b0 + chunk:
            dist = nadir_dist(np.isfinite(a[ref - b0]))
    d = warp_cube(dist[None].astype(np.float32), src_t, src_epsg, transform,
                  shape, epsg)[0]
    return wl, fwhm, out, d


def mosaic(parts):
    """Nearest-nadir mosaic of an iterable of (wl, fwhm, cube, dist) or
    None, folded in one line at a time; returns (wl, fwhm, mosaic)"""
    wl = fwhm = mos = best = None
    for part in parts:
        if part is None:
            continue
        w, f, cube, dist = part
        if mos is None:
            wl, fwhm = w, f
            best = np.full(dist.shape, np.inf)
            mos = np.full(cube.shape, np.nan, np.float32)
        take = np.isfinite(dist) & (dist < best) & np.isfinite(cube).any(0)
        mos[:, take] = cube[:, take]
        best[take] = dist[take]
    return wl, fwhm, mos


def indices(R, wl, kw_wl, kw):
    i = {t: int(np.argmin(np.abs(wl - t))) for t in (660, 860, 1240)}
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'ndvi': (R[i[860]] - R[i[660]]) / (R[i[860]] + R[i[660]]),
                'ndwi': (R[i[860]] - R[i[1240]]) / (R[i[860]] + R[i[1240]]),
                'ewt980': fit_ewt(R, wl, kw_wl, kw, WIN_980)}


def interp_matrix(wl, ref):
    """(len(ref), len(wl)) linear interpolation of spectra onto ref"""
    W = np.zeros((len(ref), len(wl)))
    for i, x in enumerate(ref):
        j = np.searchsorted(wl, x)
        j = min(max(j, 1), len(wl) - 1)
        f = (x - wl[j - 1]) / (wl[j] - wl[j - 1])
        W[i, j - 1], W[i, j] = 1 - f, f
    return W


def log_inputs(R, wl, ref):
    """(n_pix, len(ref)) log reflectance of a cube interpolated from its
    good bands onto ref (NaN where any good band is missing)"""
    g = np.flatnonzero(good_bands(wl) & np.isfinite(R).any((1, 2)))
    W = interp_matrix(wl[g], ref)
    X = R[g].reshape(len(g), -1)
    bad = ~np.isfinite(X).all(0)
    X = (W @ np.where(np.isfinite(X), X, 0)).T
    X = np.log(np.clip(X, 1e-3, None)).astype(np.float32)
    X[bad] = np.nan
    return X


def trait_bridge(aoi, wl_c, C, Fc, valid, scale, n_comps):
    """Emulators (one per number of PLSR components) fitted on the 2018
    AVIRIS-C cache, applied to both 2025 mosaics; agreement on 90 m
    cells"""
    from scipy.stats import spearmanr
    from sklearn.cross_decomposition import PLSRegression
    from fetch_wdts_traits import TRAITS
    shape = valid.shape
    ds = xr.open_dataset(SIM / f'{aoi}_refl{TRAIN_YEAR}.nc')
    wl = ds.wavelength.values
    ref = wl[good_bands(wl)]
    tr = xr.open_dataset(rc.E / 'wdts' / f'{aoi}_traits.nc') \
        .sel(year=TRAIN_YEAR)
    fid = tr.flight_id.values
    Y = np.stack([tr[f'{t}_mean'].values for t in FOCUS], -1) \
        .reshape(-1, len(FOCUS))
    mask = (fid > 0) & (ds.source_line_run.values == fid)
    Y[~mask.ravel()] = np.nan
    X = log_inputs(ds.refl.values.astype(np.float32), wl, ref)
    pix = np.flatnonzero(np.isfinite(X).all(1) & np.isfinite(Y).any(1))
    rows_i, cols_i = np.indices(shape)
    block = ((rows_i // 33) * 10000 + cols_i // 33).ravel()
    rng = np.random.default_rng(0)
    fit = rng.choice(pix, min(MAX_TRAIN, len(pix)), replace=False)
    r2, models = {}, {}
    for nc in n_comps:
        oof = emulate(X[pix], Y[pix], block[pix], nc)
        for j, t in enumerate(FOCUS):
            ok = np.isfinite(Y[pix, j])
            r2[(nc, t)] = rc.wr2(Y[pix, j][ok], oof[ok, j])
            okf = fit[np.isfinite(Y[fit, j])]
            models[(nc, t)] = PLSRegression(
                n_components=min(nc, len(ref)), scale=True).fit(
                X[okf], Y[okf, j])
    mu_t, sd_t = X[fit].mean(0), X[fit].std(0)
    del X
    layers = {}
    for name, R in (('avc', C), ('av5', Fc)):
        Xs = log_inputs(R, wl_c, ref)
        ok = np.isfinite(Xs).all(1)
        on = ok & valid.ravel()
        mu_s, sd_s = Xs[on].mean(0), Xs[on].std(0)
        for v in VARIANTS:
            Xv = Xs[ok] if v == 'raw' else \
                (Xs[ok] - mu_s) / sd_s * sd_t + mu_t
            for (nc, t), m in models.items():
                p = np.full(Xs.shape[0], np.nan, np.float32)
                p[ok] = m.predict(Xv).ravel()
                layers[f'{name}_{v}_{nc}_{t}'] = p.reshape(shape)
    for j, t in enumerate(FOCUS):
        layers[f'map_{t}'] = Y[:, j].reshape(shape)
    d = rc.cell_table(layers, valid, scale)
    rows = []
    for nc, v, t in ((nc, v, t) for nc in n_comps for v in VARIANTS
                     for t in FOCUS):
        c, f = d[f'avc_{v}_{nc}_{t}'], d[f'av5_{v}_{nc}_{t}']
        m = d[f'map_{t}']
        ok = c.notna() & f.notna()
        okm = ok & m.notna()
        rows.append(dict(aoi=aoi, n_comp=nc, inputs=v, trait=t,
                         n=int(ok.sum()), emulator_r2_2018=r2[(nc, t)],
                         rho=spearmanr(c[ok], f[ok])[0],
                         r=np.corrcoef(c[ok], f[ok])[0, 1],
                         avc_median=c[ok].median(), av5_median=f[ok].median(),
                         offset_sd=(f - c)[ok].median() / c[ok].std(),
                         rho_avc_map2018=spearmanr(c[okm], m[okm])[0],
                         rho_av5_map2018=spearmanr(f[okm], m[okm])[0]))
        x = rows[-1]
        click.echo(f'  {nc:2d} comp {v:7s} {t:10s} emulator R² {x["emulator_r2_2018"]:.2f}  '
                   f'AVIRIS-C vs AVIRIS-5 ρ {x["rho"]:.3f}  offset '
                   f'{x["offset_sd"]:+.2f} SD  ρ with 2018 map '
                   f'{x["rho_avc_map2018"]:.2f} / {x["rho_av5_map2018"]:.2f}')
    return pd.DataFrame(rows)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True,
              default=['neon_soap_teak'], show_default=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--traits', is_flag=True,
              help='Also compare emulated traits (fitted on the 2018 cache)')
@click.option('--n-comp', 'n_comps', multiple=True, type=int,
              default=[5, 10, N_COMP], show_default=True,
              help='PLSR components of the trait emulator (repeatable)')
def main(outputdir, aois, scale, traits, n_comps):
    outputdir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    kw_wl, kw = load_kw(KW)
    for aoi in aois:
        transform, shape, epsg = rc.aoi_info(aoi)
        wl_c, fw_c, C = mosaic(read_avc(sess, ln, transform, shape, epsg)
                               for ln in LINES[aoi]['avc'])
        wl_5, fw_5, F = mosaic(read_av5(ln, transform, shape, epsg)
                               for ln in LINES[aoi]['av5'])
        good_c = good_bands(wl_c)
        W, keep = srf_matrix(wl_5, good_bands(wl_5), wl_c, fw_c)
        F5 = F.reshape(F.shape[0], -1)
        Fc = np.full((len(wl_c),) + shape, np.nan, np.float32)
        Fg = np.where(np.isfinite(F5), F5, 0)
        conv = (W @ Fg).reshape((-1,) + shape)
        conv[:, ~np.isfinite(F[W.sum(0) > 0]).all(0)] = np.nan
        Fc[keep] = conv
        del F5, Fg, conv
        Fc[~good_c] = np.nan
        C[~good_c] = np.nan

        env = rc.open_env(aoi)
        valid = rc.undisturbed(env, 2025) & np.isfinite(C).any(0) & \
            np.isfinite(Fc).any(0)
        ic = indices(C, wl_c, kw_wl, kw)
        i5 = indices(F, wl_5, kw_wl, kw)
        layers = {f'avc_{k}': v for k, v in ic.items()}
        layers.update({f'av5_{k}': v for k, v in i5.items()})
        bands = np.flatnonzero(good_c & np.isfinite(Fc).any((1, 2)))
        for b in bands:
            layers[f'c{b}'] = C[b]
            layers[f'f{b}'] = Fc[b]
        d = rc.cell_table(layers, valid, scale)
        d = d.dropna(subset=['avc_ndvi', 'av5_ndvi'])
        click.echo(f'[{aoi}] {len(d)} forest cells covered by both')
        rows = []
        for b in bands:
            x, y = d[f'c{b}'], d[f'f{b}']
            ok = x.notna() & y.notna()
            rows.append(dict(aoi=aoi, band=int(b), wavelength=wl_c[b],
                             n=int(ok.sum()), avc_mean=x[ok].mean(),
                             avc_median=x[ok].median(),
                             av5_median=y[ok].median(),
                             median_rel_diff=((y - x)[ok] / x[ok]).median(),
                             bias=(y - x)[ok].mean(),
                             rel_bias=(y - x)[ok].mean() / x[ok].mean(),
                             rmse=np.sqrt(((y - x)[ok] ** 2).mean()),
                             rho=spearmanr(x[ok], y[ok])[0]))
        bt = pd.DataFrame(rows)
        bt.to_csv(outputdir / f'bands_{aoi}.csv', index=False)
        it = []
        for k in ic:
            x, y = d[f'avc_{k}'], d[f'av5_{k}']
            ok = x.notna() & y.notna()
            it.append(dict(aoi=aoi, index=k, n=int(ok.sum()),
                           avc_median=x[ok].median(),
                           av5_median=y[ok].median(),
                           bias=(y - x)[ok].mean(),
                           rmse=np.sqrt(((y - x)[ok] ** 2).mean()),
                           r=np.corrcoef(x[ok], y[ok])[0, 1],
                           rho=spearmanr(x[ok], y[ok])[0]))
            click.echo(f'  {k:7s} AVIRIS-C {it[-1]["avc_median"]:.3f} '
                       f'AVIRIS-5 {it[-1]["av5_median"]:.3f} '
                       f'bias {it[-1]["bias"]:+.3f} ρ {it[-1]["rho"]:.3f}')
        pd.DataFrame(it).to_csv(outputdir / f'indices_{aoi}.csv', index=False)
        d[['cell_row', 'cell_col'] + list(layers)[:6]].to_csv(
            outputdir / f'cells_{aoi}.csv', index=False)
        vis = bt[(bt.wavelength > 450) & (bt.wavelength < 2400)]
        click.echo(f'  bands 450-2400 nm: median |rel bias| '
                   f'{vis.rel_bias.abs().median():.3f}, median |median rel '
                   f'diff| {vis.median_rel_diff.abs().median():.3f}, median '
                   f'ρ {vis.rho.median():.3f}')
        fig_bridge(bt, d, aoi, outputdir / f'bridge_{aoi}.png')
        if traits:
            del F
            trait_bridge(aoi, wl_c, C, Fc, valid, scale, n_comps).to_csv(
                outputdir / f'traits_{aoi}.csv', index=False)


def fig_bridge(bt, d, aoi, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(bt.wavelength, bt.avc_median, '.', ms=3, label='AVIRIS-C')
    ax[0].plot(bt.wavelength, bt.av5_median, '.', ms=3,
               label='AVIRIS-5 (conv.)')
    ax[0].set_ylabel('median reflectance, forest cells')
    ax[0].legend(fontsize=7)
    ax[1].plot(bt.wavelength, bt.median_rel_diff, '.', ms=3,
               label='median rel. difference')
    ax[1].plot(bt.wavelength, bt.rho, '.', ms=3, label='Spearman ρ')
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_ylim(-0.5, 1.05)
    ax[1].legend(fontsize=7)
    for a in ax[:2]:
        a.set_xlabel('wavelength (nm)')
    ax[2].scatter(d.avc_ewt980, d.av5_ewt980, s=1, alpha=0.2)
    lim = np.nanpercentile(np.r_[d.avc_ewt980, d.av5_ewt980], [1, 99])
    ax[2].plot(lim, lim, 'k', lw=0.6)
    ax[2].set_xlim(lim)
    ax[2].set_ylim(lim)
    ax[2].set_xlabel('EWT980 AVIRIS-C (cm)')
    ax[2].set_ylabel('EWT980 AVIRIS-5 (cm)')
    fig.suptitle(f'{aoi}: AVIRIS-C vs AVIRIS-5, 2025-07-17, 90 m forest '
                 'cells', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
