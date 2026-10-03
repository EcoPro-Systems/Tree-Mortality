#!/usr/bin/env python
"""
Does the trait information that explains drought recovery survive
spaceborne-like sampling (coarser pixels, fewer or wider bands, more
noise)?

Steps (subcommands):

  refl      Read the primary-date WDTS AVIRIS-C corrected reflectance of
            a year (--year, default 2013; ORNL DAAC 2391) with all bands
            over the AOI, aggregate 15 m to the 30 m AOI grid and cache
            wdts/sim/<aoi>_refl<year>.nc. Each pixel takes the line the
            2403 trait mosaic used there (flight_id is the line's run
            number), else the nearest-nadir line, so that spectra and trait
            maps share the view geometry. Lines in another UTM zone than
            the AOI are skipped (source_line_run records each pixel's
            line, so pixels off the trait mosaic's line can be dropped).
  simulate  For each sensor configuration, degrade the spectra, retrieve
            traits with an emulator and EWT with the Beer-Lambert fit, and
            write wdts/sim/<aoi>_{traits,cwc}_<config>.nc in the layout of
            fetch_wdts_traits.py / fetch_wdts_cwc.py, so that
            proto_response_traits.py --traits/--cwc runs unchanged.
            --seed changes the noise draw (outputs <config>_s<seed>).
  compare   Paired block-bootstrap of the difference in trait gain between
            configurations (e.g. emit minus oli), from the out-of-fold
            predictions that proto_response_traits.py --save-preds writes
            for each configuration. All configurations share the cells and
            the base-model predictions, so the difference in gain over the
            base is the difference in R² of the two trait models, scored on
            the same resampled blocks. The l8raw runs (no emulator: the
            Landsat bands and indices themselves as the trait block,
            proto_response_traits.py --raw-block) join the comparison the
            same way.

Configurations:
  native  AVIRIS-C bands (water-vapour bands dropped), 30 m, no added noise
  emit    EMIT band centres and FWHM (from an EMIT L2A file), 60 m, with
          noise drawn from EMIT's own per-band reflectance uncertainty
          (median over vegetated pixels of that scene)
  sbg_hi  10 nm bands, 30 m, the EMIT noise level at 30 m (a pessimistic
          bracket for a 30 m spaceborne imaging spectrometer)
  sbg_lo  10 nm bands, 30 m, half the EMIT noise level
  oli     Landsat 8 OLI reflective bands (boxcar approximations of the
          band passes), 30 m (the multispectral control; no EWT)
  l8      a real Landsat 8 Collection 2 surface-reflectance scene near the
          airborne date (--l8; fetch_landsat_c2_ee.py --scene), through the
          same emulator: the oli control with Landsat's own calibration,
          atmospheric correction, view geometry and registration (no EWT)
60 m data are 2 x 2 means of the 30 m grid, retrieved at 60 m and put back
on the 30 m grid (each 60 m value on its four 30 m pixels).

Trait retrieval: the per-trait PLSR coefficients behind the 2403 maps are
not public, so each configuration gets its own emulator: PLSR from its
spectra to the 2403 2013 trait map (the same mosaic version as the
reflectance), on log reflectance, fitted and applied out of fold over 1 km
blocks (a pixel's
trait is never predicted by a model that saw it). The native emulator's
out-of-fold R² says how closely the emulator reproduces the maps; each
configuration's R² says how much of the map its spectra still carry. At
NEON the native emulator reaches R² 0.81 (N), 0.81 (lignin) and 0.95 (LMA)
over all pixels, and 0.88-0.89 for N and lignin on pixels whose 15 m
subpixels all pass the 2403 QC (the maps average only passing subpixels,
the spectra average all of them). SD
and QC bands are not simulated (NaN), so the reference for gains is the
native configuration, not the full-T runs.

    python proto_spaceborne_sim.py refl -a neon_soap_teak
    python proto_spaceborne_sim.py simulate -a neon_soap_teak \
        --emit-uncert $E/emit/EMIT_L2A_RFLUNCERT_..._006.nc
    python proto_response_traits.py $E/hls_results/spaceborne_sim/emit \
        -a neon_soap_teak --traits $E/wdts/sim/neon_soap_teak_traits_emit.nc \
        --cwc $E/wdts/sim/neon_soap_teak_cwc_emit.nc \
        --structure aso --structure lvis2008 --save-preds
    python proto_spaceborne_sim.py compare -a neon_soap_teak
"""
import os
import time
import click
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

import response_common as rc
from fetch_wdts_cwc import (BASE, DATES, WIN_980, session, find_lines,
                            get_range, load_kw, fit_ewt)
from fetch_wdts_traits import BOX, TRAITS

SIM = rc.E / 'wdts' / 'sim'
YEAR = 2013
BAD = [(0, 400), (1340, 1450), (1790, 1960), (2450, 3000)]
OLI = [(435, 451), (452, 512), (533, 590), (636, 673), (851, 879),
       (1566, 1651), (2107, 2294)]
CONFIGS = ['native', 'emit', 'sbg_hi', 'sbg_lo', 'oli']
L8_BANDS = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
N_COMP = 25
MAX_TRAIN = 60_000
KW = rc.E / 'wdts' / 'aux' / 'prospect_d_spectra.txt'


def good_bands(wl):
    return ~np.any([(wl >= lo) & (wl < hi) for lo, hi in BAD], axis=0)


# ---------------------------------------------------------------- refl

def line_cube(sess, h, transform, shape, jobs):
    """All bands of one line over the AOI, 2 x 2 means to 30 m, and the
    distance of each 30 m pixel from the swath centre (15 m pixels)"""
    res = h['res']
    ny15, nx15 = shape[0] * 2, shape[1] * 2
    r0 = int(round((h['y0'] - transform.f) / res))
    c0 = int(round((transform.c - h['x0']) / res))
    ns, nl = int(h['samples']), int(h['lines'])
    rr0, rr1 = max(r0, 0), min(r0 + ny15, nl)
    cc0, cc1 = max(c0, 0), min(c0 + nx15, ns)
    if rr1 <= rr0 or cc1 <= cc0:
        return None
    nrow = rr1 - rr0
    url = BASE + h['name'] + '.bin'

    def to30(a):
        canvas = np.full((ny15, nx15), np.nan, np.float32)
        canvas[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = a
        c = canvas.reshape(shape[0], 2, shape[1], 2)
        with np.errstate(invalid='ignore'):
            return np.nanmean(c, axis=(1, 3))

    def read(b):
        off = (b * nl * ns + rr0 * ns) * 4
        buf = get_range(sess, url, off, nrow * ns * 4)
        a = np.frombuffer(buf, '<f4').reshape(nrow, ns)[:, cc0:cc1].copy()
        a[a <= -9998] = np.nan
        return b, to30(a)

    nb = len(h['wavelength'])
    ref = int(np.argmin(np.abs(h['wavelength'] - 860)))
    _, a860 = read(ref)
    if not np.isfinite(a860).any():
        return None
    # Swath-centre distance from the valid columns of each 15 m row
    valid = np.isfinite(a860)
    cols = np.arange(shape[1])[None, :].repeat(shape[0], 0).astype(float)
    with np.errstate(invalid='ignore'):
        centre = np.nanmean(np.where(valid, cols, np.nan), axis=1)
    dist = np.where(valid, np.abs(cols - centre[:, None]) * 2, np.nan)
    cube = np.full((nb,) + shape, np.nan, np.float32)
    with ThreadPoolExecutor(jobs) as pool:
        for b, a in pool.map(read, range(nb)):
            cube[b] = a
    return cube, dist


@click.group()
def cli():
    pass


@cli.command()
@click.option('-a', '--aoi', 'name', default='neon_soap_teak',
              show_default=True)
@click.option('-j', '--jobs', default=8, show_default=True)
@click.option('--year', default=YEAR, show_default=True)
def refl(name, jobs, year):
    """Cache one year's reflectance mosaic on the 30 m AOI grid"""
    transform, shape, epsg = rc.aoi_info(name)
    SIM.mkdir(parents=True, exist_ok=True)
    sess = session()
    dates = DATES[BOX[name]][year]
    fid = xr.open_dataset(rc.E / 'wdts' / f'{name}_traits.nc').flight_id \
        .sel(year=year).values
    mos, best, wl, names = None, np.full(shape, np.inf), None, []
    src = np.full(shape, -1, np.int16)
    run = np.full(shape, -1, np.int16)
    for rank, date in enumerate(dates):
        for h in find_lines(sess, date):
            if h['zone'] != epsg - 32600:
                continue  # as in the canopy-water fetch for this AOI
            t = time.time()
            out = line_cube(sess, h, transform, shape, jobs)
            if out is None:
                continue
            cube, dist = out
            other = (fid > 0) & (fid != h['run'])
            score = rank * 1e6 + other * 1e5 + dist
            take = np.isfinite(dist) & (score < best)
            if not take.any():
                continue
            if mos is None:
                wl = h['wavelength']
                mos = np.full(cube.shape, np.nan, np.float32)
            assert np.allclose(h['wavelength'], wl)
            mos[:, take] = cube[:, take]
            best[take] = score[take]
            src[take] = len(names)
            run[take] = h['run'] if rank == 0 else -1
            names.append(h['name'])
            click.echo(f'[{name}] {h["name"]}: {take.sum()} px '
                       f'({time.time() - t:.0f} s)')
    xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    ds = xr.Dataset({'refl': (('wavelength', 'y', 'x'), mos),
                     'source_line': (('y', 'x'), src),
                     'source_line_run': (('y', 'x'), run)},
                    coords={'wavelength': wl, 'y': ys, 'x': xs},
                    attrs={'crs': f'EPSG:{epsg}',
                           'transform': list(transform)[:6],
                           'source': 'doi:10.3334/ORNLDAAC/2391',
                           'lines': ','.join(names)})
    out = SIM / f'{name}_refl{year}.nc'
    ds.to_netcdf(out.with_suffix('.tmp.nc'),
                 encoding={'refl': {'zlib': True, 'dtype': 'int16',
                                    'scale_factor': 1e-4,
                                    '_FillValue': -32768},
                           'source_line': {'zlib': True},
                           'source_line_run': {'zlib': True}})
    os.replace(out.with_suffix('.tmp.nc'), out)
    click.echo(f'wrote {out}')


# ------------------------------------------------------------ simulate

def srf_matrix(wl, good, centres, fwhm=None, boxes=None):
    """(n_out, n_in) band-response weights over the good source bands:
    Gaussian (centres, fwhm) or boxcar (boxes)"""
    W = []
    for i in range(len(boxes) if boxes is not None else len(centres)):
        if boxes is not None:
            lo, hi = boxes[i]
            w = ((wl >= lo) & (wl <= hi)).astype(float)
        else:
            s = fwhm[i] / 2.3548
            w = np.exp(-0.5 * ((wl - centres[i]) / s) ** 2)
        w = np.where(good, w, 0)
        W.append(w / w.sum() if w.sum() > 0 else w)
    W = np.array(W)
    keep = W.sum(1) > 0
    return W[keep], keep


def aggregate2(R):
    """(nb, H, W) 30 m -> 60 m 2 x 2 means, back on the 30 m grid"""
    nb, h, w = R.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    with np.errstate(invalid='ignore'):
        c = np.nanmean(R[:, :h2, :w2].reshape(nb, h2 // 2, 2, w2 // 2, 2),
                       axis=(2, 4))
    out = np.full(R.shape, np.nan, np.float32)
    out[:, :h2, :w2] = np.repeat(np.repeat(c, 2, 1), 2, 2)
    return out


def emit_bands(path):
    b = xr.open_dataset(path, group='sensor_band_parameters')
    u = xr.open_dataset(path)
    var = [v for v in u.data_vars if u[v].ndim == 3][0]
    unc = u[var].values  # (y, x, band)
    return b.wavelengths.values, b.fwhm.values, unc


def emit_noise(path, rfl_path):
    """Median per-band reflectance uncertainty over vegetated pixels"""
    wl, fwhm, unc = emit_bands(path)
    r = xr.open_dataset(rfl_path)
    var = [v for v in r.data_vars if r[v].ndim == 3][0]
    i660, i860 = (int(np.argmin(np.abs(wl - t))) for t in (660, 860))
    r660, r860 = (r[var].isel(bands=i).values for i in (i660, i860))
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = (r860 - r660) / (r860 + r660)
    veg = (ndvi > 0.5) & np.all(np.isfinite(unc), axis=2) & \
        (unc[..., i860] > 0)
    return wl, fwhm, np.median(unc[veg], axis=0)


def configure(cfg, wl, good, emit):
    """(weights, output wavelengths, per-band noise SD, aggregate to 60 m)"""
    W, owl, noise, agg = _configure(cfg, wl, good, emit)
    keep = good_bands(owl)  # no output bands centred in absorption gaps
    return W[keep], owl[keep], None if noise is None else noise[keep], agg


def _configure(cfg, wl, good, emit):
    if cfg == 'native':
        W = np.eye(len(wl))[good]
        return W, wl[good], None, False
    if cfg == 'oli':
        W, keep = srf_matrix(wl, good, None, boxes=OLI)
        return W, np.array([np.mean(b) for b in OLI])[keep], None, False
    ewl, efwhm, enoise = emit
    if cfg == 'emit':
        W, keep = srf_matrix(wl, good, ewl, efwhm)
        return W, ewl[keep], enoise[keep], True
    centres = np.arange(400, 2501, 10.0)
    W, keep = srf_matrix(wl, good, centres, np.full(len(centres), 10.0))
    noise = np.interp(centres[keep], ewl, enoise)
    return W, centres[keep], noise * (1.0 if cfg == 'sbg_hi' else 0.5), False


def degrade(R, W, noise, agg, seed=0):
    """R (nb, H, W) native -> degraded (nb_out, H, W)"""
    nb, h, w = R.shape
    Rg = np.where(np.isfinite(R), R, 0)
    out = (W @ Rg.reshape(nb, -1)).reshape(-1, h, w).astype(np.float32)
    bad = ~np.isfinite(R[W.sum(0) > 0]).all(0) if W.shape[0] else None
    out[:, bad] = np.nan
    if agg:
        out = aggregate2(out)
    if noise is not None:
        rng = np.random.default_rng(seed)
        if agg:  # one draw per 60 m pixel, shared by its four 30 m pixels
            hh, ww = h // 2 + 1, w // 2 + 1
            e = rng.normal(0, 1, (len(noise), hh, ww)).astype(np.float32)
            e = np.repeat(np.repeat(e, 2, 1), 2, 2)[:, :h, :w]
        else:
            e = rng.normal(0, 1, out.shape).astype(np.float32)
        out = out + e * noise[:, None, None].astype(np.float32)
    return out


def landsat_scene(path, shape):
    """(7, H, W) clear-sky Landsat 8 reflectance and band centres"""
    ds = xr.open_dataset(path)
    assert ds.n_clear.shape == shape, 'Landsat scene is not on the AOI grid'
    ok = ds.n_clear.values > 0
    D = np.stack([np.where(ok, ds[b].values, np.nan) for b in L8_BANDS])
    return D.astype(np.float32), np.array([np.mean(b) for b in OLI])


def emulate(X, Y, blocks, n_comp, seed=0):
    """Out-of-fold PLSR predictions of each column of Y from X"""
    P = np.full(Y.shape, np.nan, np.float32)
    rng = np.random.default_rng(seed)
    nc = min(n_comp, X.shape[1])
    for tr, te in GroupKFold(n_splits=rc.N_SPLITS).split(X, groups=blocks):
        if len(tr) > MAX_TRAIN:
            tr = rng.choice(tr, MAX_TRAIN, replace=False)
        for j in range(Y.shape[1]):
            ok = np.isfinite(Y[tr, j])
            m = PLSRegression(n_components=nc, scale=True)
            m.fit(X[tr][ok], Y[tr, j][ok])
            P[te, j] = m.predict(X[te]).ravel()
    return P


@cli.command()
@click.option('-a', '--aoi', 'name', default='neon_soap_teak',
              show_default=True)
@click.option('--emit-uncert', type=click.Path(path_type=Path, exists=True),
              required=True, help='EMIT_L2A_RFLUNCERT file (with its RFL '
                                  'file beside it)')
@click.option('--config', 'configs', multiple=True, default=CONFIGS,
              show_default=True, type=click.Choice(CONFIGS + ['l8']))
@click.option('--l8', 'l8_path', type=click.Path(path_type=Path, exists=True),
              help='Landsat 8 scene for the l8 configuration')
@click.option('--n-comp', default=N_COMP, show_default=True)
@click.option('--seed', default=0, show_default=True,
              help='Noise draw; outputs are named <config>_s<seed> if not 0')
def simulate(name, emit_uncert, configs, l8_path, n_comp, seed):
    """Degraded spectra -> emulated traits and EWT per configuration"""
    transform, shape, epsg = rc.aoi_info(name)
    ds = xr.open_dataset(SIM / f'{name}_refl{YEAR}.nc')
    R = ds.refl.values.astype(np.float32)
    wl = ds.wavelength.values
    good = good_bands(wl)
    rfl = Path(str(emit_uncert).replace('RFLUNCERT', 'RFL'))
    emit = emit_noise(emit_uncert, rfl)
    tr = xr.open_dataset(rc.E / 'wdts' / f'{name}_traits.nc').sel(year=YEAR)
    fid = tr.flight_id.values
    Y = np.stack([tr[f'{t}_mean'].values for t in TRAITS], -1)
    Y[fid <= 0] = np.nan
    rows_i, cols_i = np.indices(shape)
    block = ((rows_i // 33) * 10000 + cols_i // 33)  # ~1 km at 30 m
    kw_wl, kw = load_kw(KW)
    stats = []
    for cfg in configs:
        t0 = time.time()
        if cfg == 'l8':
            if l8_path is None:
                raise click.UsageError('the l8 configuration needs --l8')
            D, owl = landsat_scene(l8_path, shape)
        else:
            W, owl, noise, agg = configure(cfg, wl, good, emit)
            D = degrade(R, W, noise, agg, seed)
        label = cfg if seed == 0 else f'{cfg}_s{seed}'
        # log reflectance: closer to the 2403 maps than raw or vector-
        # normalized spectra (out-of-fold R², native configuration)
        X = np.log(np.clip(D.reshape(len(owl), -1).T, 1e-3, None))
        pix = np.isfinite(X).all(1) & np.isfinite(Y.reshape(-1, len(TRAITS))
                                                  ).any(1)
        P = np.full((X.shape[0], len(TRAITS)), np.nan, np.float32)
        P[pix] = emulate(X[pix], Y.reshape(-1, len(TRAITS))[pix],
                         block.ravel()[pix], n_comp)
        Yf = Y.reshape(-1, len(TRAITS))
        for j, t in enumerate(TRAITS):
            ok = pix & np.isfinite(Yf[:, j])
            stats.append(dict(aoi=name, config=label, trait=t,
                              n=int(ok.sum()),
                              r2=rc.wr2(Yf[ok, j], P[ok, j]),
                              rho=pd.Series(Yf[ok, j]).corr(
                                  pd.Series(P[ok, j]), method='spearman')))
        # EWT (not for Landsat bands: none in the 980 nm window)
        ewt = np.full(shape, np.nan, np.float32)
        if cfg not in ('oli', 'l8'):
            with np.errstate(invalid='ignore', divide='ignore'):
                ewt = fit_ewt(D, owl, kw_wl, kw, WIN_980).astype(np.float32)
        write_outputs(name, label, P.reshape(shape + (len(TRAITS),)), ewt,
                      fid, transform, shape, epsg)
        s = {x['trait']: x for x in stats if x['config'] == label}
        click.echo(f'[{name}] {label:7s} {len(owl):3d} bands  ' + '  '.join(
            f'{t} R² {s[t]["r2"]:.2f}' for t in
            ('Nitrogen', 'LMA', 'Lignin', 'Cellulose', 'Chlorophylls')) +
            f'  median EWT {np.nanmedian(ewt):.3f}  ({time.time() - t0:.0f} s)')
        f = SIM / f'{name}_emulator.csv'
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        if len(old):
            old = old[~old.config.isin([label])]
        pd.concat([old, pd.DataFrame([x for x in stats
                                      if x['config'] == label])]).to_csv(
            f, index=False)


def write_outputs(name, cfg, P, ewt, fid, transform, shape, epsg):
    xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    coords = {'year': [YEAR], 'y': ys, 'x': xs}
    dims = ('year', 'y', 'x')
    nan = np.full((1,) + shape, np.nan, np.float32)
    data = {}
    for j, t in enumerate(TRAITS):
        data[f'{t}_mean'] = (dims, P[None, ..., j])
        data[f'{t}_sd'] = (dims, nan)
    data['qc_fc'] = (dims, np.full((1,) + shape, 255, np.uint8))
    data['flight_id'] = (dims, fid[None].astype(np.int16))
    attrs = {'crs': f'EPSG:{epsg}', 'transform': list(transform)[:6],
             'config': cfg, 'source': 'proto_spaceborne_sim.py emulator'}
    xr.Dataset(data, coords=coords, attrs=attrs).to_netcdf(
        SIM / f'{name}_traits_{cfg}.nc')
    xr.Dataset({'ewt980': (dims, ewt[None]),
                'source_line': (dims, fid[None].astype(np.int16))},
               coords=coords, attrs=attrs).to_netcdf(
        SIM / f'{name}_cwc_{cfg}.nc')


# ------------------------------------------------------------- compare

KEYS = ['cell_row', 'cell_col']
BLOCKS = ['block1000', 'block5000']
# (base, trait model) per kind of run: the gain of each trait model over
# its base, and the paired difference of that model between configurations
COMPARE = {'lidar': [('Env+S+L', 'Env+S+L+Tres'), ('Env+S+L', 'Env+S+L+T'),
                     ('Env+S', 'Env+S+T')],
           'ladder': [('Env+S', 'Env+S+T'), ('Env+S', 'Env+S+Tres')]}


def joined_preds(root, configs, run, name, target, scale_m=90):
    """The configurations' predictions for one target on the cells all of
    them share, as columns <config>:<feature set>"""
    m = None
    for c in configs:
        f = root / c / f'preds_{run}_{name}_{scale_m}m.csv.gz'
        if not f.exists():
            continue
        p = pd.read_csv(f)
        p = p[p.target == target].drop(columns='target')
        p = p.rename(columns={x: f'{c}:{x}' for x in p.columns
                              if x not in KEYS + BLOCKS + ['y']})
        m = p if m is None else m.merge(p.drop(columns=BLOCKS + ['y']),
                                        on=KEYS, how='inner')
    return m


@cli.command()
@click.option('-a', '--aoi', 'name', default='neon_soap_teak',
              show_default=True)
@click.option('--root', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results' / 'spaceborne_sim',
              show_default=True)
@click.option('--config', 'configs', multiple=True,
              default=CONFIGS + ['l8', 'l8raw'], show_default=True)
@click.option('--ref', 'refs', multiple=True, default=['oli', 'l8', 'l8raw'],
              show_default=True, help='Configurations to difference against')
@click.option('-t', '--target', 'targets', multiple=True,
              default=['ndmi_recovery', 'nirv_recovery', 'ndmi_sens',
                       'ndmi_resistance'], show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
def compare(name, root, configs, refs, targets, n_boot):
    """Paired differences in trait gain between configurations"""
    rows = []
    for run in ('lidar_aso', 'lidar_lvis2008', 'ladder'):
        kind = run.split('_')[0]
        for t in targets:
            m = joined_preds(root, configs, run, name, t)
            if m is None or len(m) < 500:
                continue
            have = [c for c in configs if f'{c}:{COMPARE[kind][0][1]}' in m]
            if len(have) < 2:
                continue
            # base models use no traits, so they must agree across configs
            for base in {b for b, _ in COMPARE[kind]}:
                b = m[[f'{c}:{base}' for c in have]].values
                dev = np.abs(b - b[:, :1]).max()
                if dev > 1e-5:
                    click.echo(f'  warning: {run} {t} {base} predictions '
                               f'differ between configurations ({dev:.2g})')
            preds = {x: m[x].values for x in m.columns if ':' in x}
            pairs, meta = [], []
            for base, full in COMPARE[kind]:
                for c in have:
                    pairs.append((f'{have[0]}:{base}', f'{c}:{full}'))
                    meta.append((c, full, 'gain', base))
                    for r in refs:
                        if r in have and r != c:
                            pairs.append((f'{r}:{full}', f'{c}:{full}'))
                            meta.append((c, full, 'diff', r))
            for bcol in BLOCKS:
                bs = rc.bootstrap_r2(m.y.values, preds, m[bcol].values,
                                     n_boot=n_boot, pairs=pairs)
                for (a, b), (c, full, kd, cmp) in zip(pairs, meta):
                    est, lo, hi = bs[f'{b}-{a}']
                    rows.append(dict(aoi=name, run=run, target=t,
                                     boot_blocks=bcol, n=len(m), config=c,
                                     features=full, kind=kd, compare=cmp,
                                     r2=est, lo=lo, hi=hi))
            base, full = COMPARE[kind][0]
            g = {(x['config'], x['kind'], x['compare']): x for x in rows
                 if x['run'] == run and x['target'] == t
                 and x['features'] == full and x['boot_blocks'] == BLOCKS[0]}
            click.echo(f'[{name}] {run} {t} n={len(m)}: {full} over {base}')
            for c in have:
                x = g[(c, 'gain', base)]
                msg = f'    {c:7s} {x["r2"]:+.3f} [{x["lo"]:+.3f},' \
                      f'{x["hi"]:+.3f}]'
                for r in refs:
                    x = g.get((c, 'diff', r))
                    if x:
                        msg += (f'   vs {r} {x["r2"]:+.3f} '
                                f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]')
                click.echo(msg)
    out = root / f'paired_{name}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    click.echo(f'wrote {out}')


if __name__ == '__main__':
    cli()
