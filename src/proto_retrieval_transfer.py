#!/usr/bin/env python
"""
Do trait retrievals carry over to another area or date? A Landsat trait
proxy is trained on AVIRIS trait maps of the same place and date; the
question is whether a spaceborne-like VSWIR retrieval needs that too, or
holds where it was not trained.

Configurations (proto_spaceborne_sim.py; PLSR emulators of the 2403 maps on
log reflectance):
  native   AVIRIS-C bands (resampled to the 2013 NEON band centres), 30 m
  emit     EMIT bands, 60 m, EMIT noise
  sbg_lo   10 nm bands, 30 m, half the EMIT noise
  l8       one Landsat 8 scene near the airborne date (seven bands)
  l8multi  the June (DOY 145-190) and Jul-Sep (DOY 182-273) Landsat 8
           composites of the airborne year (six bands each)
Noise draws are seeded per domain, so source and target never share one.

Domains (AOI, year of the AVIRIS acquisition and trait map) and transfers
(the emulator fitted on the source, applied to the target):
  across areas  neon13 -> sierra13 and back (Yosemite box, same flight
                day and Landsat scene: the mildest transfer)
  across boxes  yos13 (neon13 + sierra13 pooled) -> stan13 (Tahoe box,
                June 4 flight, non-_v2 mosaic, another Landsat path; cycle-1
                responses only)
  across dates  neon13 -> neon18, sierra13 -> sierra18 (June 22, 2018
                flight, non-_v2 mosaic, 2018 Landsat scene); scored on the
                cycle-2 responses of these pilot AOIs
In place: the same configuration's emulator fitted out of fold over 1 km
blocks within the target (as in proto_spaceborne_sim.py simulate).
Pixels: forest pixels whose spectra come from the line the trait mosaic
used (refl caches with source_line_run), with every configuration's
inputs present, so all configurations share the pixels.

Subcommands:
  emulate  Fit and apply the emulators; write 90 m cell tables per target
           domain (<cfg>:<mode>:<trait>, cross-track normalized as in
           proto_response_traits.py, and the raw values with the maps for
           agreement) to OUTPUTDIR/cells_<domain>.csv.gz.
  score    On the cells with responses:
             agreement  Spearman ρ and R² of each configuration's in-place
                        and transferred traits against the AVIRIS maps;
             gain       NDMI/NIRv recovery R² over Env+S with each
                        configuration's in-place and transferred traits,
                        and with the raw Landsat blocks (l8raw, l8multi);
           Paired 1 km block bootstrap of the loss (in place - transferred)
           and of the difference in loss between each VSWIR configuration
           and each Landsat proxy, and of transferred VSWIR against the raw
           Landsat blocks.

    python proto_spaceborne_sim.py refl -a neon_soap_teak --year 2018
    python proto_spaceborne_sim.py refl -a stanislaus
    python proto_retrieval_transfer.py emulate $E/hls_results/retrieval_transfer \\
        --emit-uncert $E/emit/EMIT_L2A_RFLUNCERT_..._006.nc
    python proto_retrieval_transfer.py score $E/hls_results/retrieval_transfer
"""
import time
import zlib
import click
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression

import response_common as rc
import proto_forward_pilot as fp
import proto_response_transfer as prt
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import TRAITS, build
from proto_trait_dynamics import crosstrack_normalize
from proto_spaceborne_sim import (SIM, L8_BANDS, N_COMP, MAX_TRAIN,
                                  good_bands, emit_noise, configure, degrade,
                                  emulate)
from proto_spaceborne_only import aoi_dirs, landsat_block

CONFIGS = ['native', 'emit', 'sbg_lo', 'l8', 'l8multi']
VSWIR = ['native', 'emit', 'sbg_lo']
PROXIES = ['l8', 'l8multi']
DOMAINS = {'neon13': ('neon_soap_teak', 2013), 'sierra13': ('sierra_nf', 2013),
           'stan13': ('stanislaus', 2013), 'neon18': ('neon_soap_teak', 2018),
           'sierra18': ('sierra_nf', 2018)}
SOURCES = {'neon13': ['neon13'], 'sierra13': ['sierra13'],
           'yos13': ['neon13', 'sierra13']}
# target domain: transfer sources applied to it
TRANSFERS = {'neon13': ['sierra13'], 'sierra13': ['neon13'],
             'stan13': ['yos13'], 'neon18': ['neon13'],
             'sierra18': ['sierra13']}
L8_SCENE = {'neon13': '20130621', 'sierra13': '20130621',
            'stan13': '20130628', 'neon18': '20180619',
            'sierra18': '20180619'}
COMPOSITES = ['doy145-190', 'doy182-273']
COMP_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
FOCUS = ['Nitrogen', 'LMA', 'Lignin', 'Cellulose']
TARGETS = ['ndmi_recovery', 'nirv_recovery']
KEYS = ['cell_row', 'cell_col']


# ------------------------------------------------------------- emulate

def last_year(dom):
    """Disturbance mask end: through cycle 1 (2013) or cycle 2 (2018)"""
    return 2019 if DOMAINS[dom][1] == 2013 else 2025


def landsat_inputs(dom, shape):
    """(n_feat, H, W) Landsat reflectance for l8 and l8multi"""
    aoi, year = DOMAINS[dom]
    ds = xr.open_dataset(SIM / f'{aoi}_l8_{L8_SCENE[dom]}.nc')
    ok = ds.n_clear.values > 0
    l8 = np.stack([np.where(ok, ds[b].values, np.nan) for b in L8_BANDS])
    multi = []
    for w in COMPOSITES:
        c = xr.open_dataset(rc.E / 'landsat_composites' /
                            f'{aoi}_c2oli_{w}.nc').sel(year=year)
        ok = c.n_clear.values > 0
        multi += [np.where(ok, c[b].values, np.nan) for b in COMP_BANDS]
    for a in (l8, multi):
        assert a[0].shape == shape
    return {'l8': l8.astype(np.float32),
            'l8multi': np.stack(multi).astype(np.float32)}


def resample_matrix(wl, ref):
    """(len(ref), len(wl)) linear interpolation of spectra onto ref"""
    W = np.zeros((len(ref), len(wl)))
    for i, x in enumerate(ref):
        j = np.searchsorted(wl, x)
        if j == 0 or j == len(wl):
            continue
        f = (x - wl[j - 1]) / (wl[j] - wl[j - 1])
        W[i, j - 1], W[i, j] = 1 - f, f
    return W


class Domain:
    """Spectra, maps and the shared pixel mask of one (AOI, year)"""

    def __init__(self, dom, emit, ref_wl):
        self.dom = dom
        self.aoi, self.year = DOMAINS[dom]
        self.transform, self.shape, _ = rc.aoi_info(self.aoi)
        ds = xr.open_dataset(SIM / f'{self.aoi}_refl{self.year}.nc')
        self.R = ds.refl.values.astype(np.float32)
        self.wl = ds.wavelength.values
        tr = xr.open_dataset(rc.E / 'wdts' / f'{self.aoi}_traits.nc') \
            .sel(year=self.year)
        self.fid = tr.flight_id.values
        Y = np.stack([tr[f'{t}_mean'].values for t in TRAITS], -1)
        mask = self.fid > 0
        if 'source_line_run' in ds:  # spectra from the mosaic's own line
            mask &= ds.source_line_run.values == self.fid
        self.landsat = landsat_inputs(dom, self.shape)
        i860 = int(np.argmin(np.abs(self.wl - 860)))
        mask &= np.isfinite(self.R[i860])
        # bands empty over the domain (the Tahoe 2013 lines have no 1323
        # and 1333 nm data) leave the usable set: native spectra are
        # interpolated across them, band responses renormalized
        fin = np.isfinite(self.R[:, mask]).mean(1)
        self.good = good_bands(self.wl) & (fin > 0.5)
        for a in self.landsat.values():
            mask &= np.isfinite(a).all(0)
        env = rc.open_env(self.aoi)
        self.elev = env.elevation.values
        self.valid = rc.undisturbed(env, last_year(dom))
        self.mask = mask & self.valid
        Y[~self.mask] = np.nan
        self.Y = Y.reshape(-1, len(TRAITS))
        self.emit, self.ref_wl = emit, ref_wl
        rows, cols = np.indices(self.shape)
        self.block = ((rows // 33) * 10000 + cols // 33).ravel()
        self.seed = zlib.crc32(dom.encode())

    def X(self, cfg):
        """(n_pix, n_feat) log inputs of a configuration; NaN off mask"""
        if cfg in PROXIES:
            D = self.landsat[cfg]
        else:
            good = self.good
            if cfg == 'native':
                W = np.zeros((len(self.ref_wl), len(self.wl)))
                W[:, good] = resample_matrix(self.wl[good], self.ref_wl)
                D = degrade(self.R, W, None, False)
            else:
                W, _, noise, agg = configure(cfg, self.wl, good, self.emit)
                D = degrade(self.R, W, noise, agg, self.seed)
        X = np.log(np.clip(D.reshape(len(D), -1).T, 1e-3, None))
        X[~self.mask.ravel()] = np.nan
        return X

    def pix(self, X):
        return np.flatnonzero(np.isfinite(X).all(1) &
                              np.isfinite(self.Y).any(1))


def fit(X, Y, n_comp):
    nc = min(n_comp, X.shape[1])
    models = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        models.append(PLSRegression(n_components=nc, scale=True)
                      .fit(X[ok], Y[ok, j]))
    return models


def apply(models, X, idx):
    P = np.full((X.shape[0], len(models)), np.nan, np.float32)
    for j, m in enumerate(models):
        P[idx, j] = m.predict(X[idx]).ravel()
    return P


def cell_layers(dm, P, tag):
    """Raw and cross-track-normalized trait layers of a prediction"""
    layers = {}
    for j, t in enumerate(TRAITS):
        a = P[:, j].reshape(dm.shape).copy()
        a[dm.fid <= 0] = np.nan
        layers[f'{tag}:{t}:raw'] = a
        layers[f'{tag}:{t}'] = crosstrack_normalize(a, dm.fid, dm.elev,
                                                    dm.transform)
    return layers


@click.group()
def cli():
    pass


@cli.command('emulate')
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--emit-uncert', type=click.Path(path_type=Path, exists=True),
              required=True, help='EMIT_L2A_RFLUNCERT file (RFL beside it)')
@click.option('--config', 'configs', multiple=True, default=CONFIGS,
              show_default=True)
@click.option('--target', 'targets', multiple=True, default=list(TRANSFERS),
              show_default=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--n-comp', default=N_COMP, show_default=True)
def emulate_cmd(outputdir, emit_uncert, configs, targets, scale, n_comp):
    """Fit here, apply there; 90 m cell tables per target domain"""
    outputdir.mkdir(parents=True, exist_ok=True)
    rfl = Path(str(emit_uncert).replace('RFLUNCERT', 'RFL'))
    emit = emit_noise(emit_uncert, rfl)
    ref = xr.open_dataset(SIM / 'neon_soap_teak_refl2013.nc').wavelength.values
    ref_wl = ref[good_bands(ref)]
    need = sorted({s for t in targets for src in TRANSFERS[t]
                   for s in SOURCES[src]})
    rng = np.random.default_rng(0)
    # 1. Training samples of each source domain, per configuration
    train = {}
    for dom in need:
        dm = Domain(dom, emit, ref_wl)
        for cfg in configs:
            X = dm.X(cfg)
            idx = dm.pix(X)
            idx = rng.choice(idx, min(MAX_TRAIN, len(idx)), replace=False)
            train[(dom, cfg)] = (X[idx], dm.Y[idx])
        click.echo(f'[{dom}] training samples drawn')
        del dm
    models = {}
    for src, doms in SOURCES.items():
        if not all((d, configs[0]) in train for d in doms):
            continue
        for cfg in configs:
            parts = [train[(d, cfg)] for d in doms]
            k = MAX_TRAIN // len(parts)
            X = np.concatenate([p[0][:k] for p in parts])
            Y = np.concatenate([p[1][:k] for p in parts])
            models[(src, cfg)] = fit(X, Y, n_comp)
        click.echo(f'[{src}] emulators fitted')
    # 2. In place (out of fold) and transferred predictions per target
    for dom in targets:
        t0 = time.time()
        dm = Domain(dom, emit, ref_wl)
        layers = {f'map:{t}:raw': dm.Y[:, j].reshape(dm.shape)
                  for j, t in enumerate(TRAITS)}
        for cfg in configs:
            X = dm.X(cfg)
            idx = dm.pix(X)
            P = np.full((X.shape[0], len(TRAITS)), np.nan, np.float32)
            P[idx] = emulate(X[idx], dm.Y[idx], dm.block[idx], n_comp)
            layers.update(cell_layers(dm, P, f'{cfg}:in'))
            for src in TRANSFERS[dom]:
                P = apply(models[(src, cfg)], X, idx)
                layers.update(cell_layers(dm, P, f'{cfg}:{src}'))
            click.echo(f'[{dom}] {cfg}: {len(idx)} px')
        cells = rc.cell_table(layers, dm.valid, scale)
        cells.to_csv(outputdir / f'cells_{dom}.csv.gz', index=False,
                     float_format='%.6g')
        click.echo(f'[{dom}] {len(cells)} cells ({time.time() - t0:.0f} s)')


# --------------------------------------------------------------- score

def draws(blocks, n_boot, seed=0):
    """Block index per cell and (n_boot + 1, n_blocks) multiplicities, the
    first row the point estimate"""
    codes, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    M = np.stack([np.ones(len(codes))] +
                 [np.bincount(x, minlength=len(codes)) for x in d])
    return inv, M.astype(float)


def r2_draws(y, p, inv, M):
    nb = M.shape[1]
    agg = lambda v: M @ np.bincount(inv, v, nb)
    n, sy, syy = agg(np.ones_like(y)), agg(y), agg(y * y)
    return 1 - agg((y - p) ** 2) / (syy - sy ** 2 / n)


def rho_draws(y, p, inv, M):
    """Spearman ρ with ranks fixed over all cells (multiplicity-weighted
    Pearson of the ranks)"""
    nb = M.shape[1]
    agg = lambda v: M @ np.bincount(inv, v, nb)
    rx, ry = rankdata(p), rankdata(y)
    n, sx, sy = agg(np.ones_like(y)), agg(rx), agg(ry)
    cov = agg(rx * ry) - sx * sy / n
    return cov / np.sqrt((agg(rx * rx) - sx ** 2 / n) *
                         (agg(ry * ry) - sy ** 2 / n))


def summarize(v):
    return dict(est=v[0], lo=np.percentile(v[1:], 2.5),
                hi=np.percentile(v[1:], 97.5))


def contrasts(stat, configs, src, base=None, raw=None):
    """{name: draws} of the in-place, transferred, loss and difference-in-
    loss contrasts from stat[(config, mode)] (draws); base, if given, is
    subtracted first (gains); raw: {name: draws} of reference blocks"""
    g = (lambda v: v - base) if base is not None else (lambda v: v)
    raw = raw or {}
    out = {}
    for c in configs:
        out[f'{c}:in'] = g(stat[(c, 'in')])
        out[f'{c}:tr'] = g(stat[(c, src)])
        out[f'{c}:loss'] = stat[(c, 'in')] - stat[(c, src)]
    for v in configs:
        if v not in VSWIR:
            continue
        for p in configs:
            if p in PROXIES:
                out[f'{v}-{p}:dloss'] = out[f'{v}:loss'] - out[f'{p}:loss']
                out[f'{v}-{p}:tr'] = stat[(v, src)] - stat[(p, src)]
        for name, r in raw.items():
            out[f'{v}-{name}:tr'] = stat[(v, src)] - r
    for name, r in raw.items():
        out[f'{name}:raw'] = g(r)
    return out


def domain_table(dom, outputdir, scale):
    """Cells with responses, Env+S and the emulated traits; (table, es)"""
    aoi, year = DOMAINS[dom]
    cells = pd.read_csv(outputdir / f'cells_{dom}.csv.gz')
    if year == 2013:
        rdir, ddir = aoi_dirs(aoi)
        d = build(aoi, scale, rdir, ddir)
        d = d.drop(columns=[c for c in d if c.startswith(('T_', 'W_'))])
        es = ENV + S_WALL
    else:  # cycle 2, pilot AOIs only (proto_forward_pilot.build refuses
        # held-out AOIs)
        d, _ = fp.build(aoi, scale, False)
        d = d.drop(columns=[c for c in d if c.startswith('T_')])
        es = prt.ENV + S_WALL
    valid = rc.undisturbed(rc.open_env(aoi), last_year(dom))
    for c in ('l8raw', 'l8multi'):
        t = landsat_block(aoi, valid, scale, c, L8_SCENE[dom], year)
        d = d.merge(t[KEYS + [x for x in t if x.startswith(f'{c}:')]],
                    on=KEYS, how='left')
    d = d.merge(cells.drop(columns=[c for c in cells if c.startswith(
        ('n_px', 'block'))]), on=KEYS, how='inner')
    return d, es


@cli.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--domain', 'domains', multiple=True, default=list(TRANSFERS),
              show_default=True)
@click.option('--config', 'configs', multiple=True, default=CONFIGS,
              show_default=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
def score(outputdir, domains, configs, scale, n_boot):
    """Agreement with the maps and recovery gains, in place vs transferred"""
    agree_rows, gain_rows = [], []
    for dom in domains:
        d, es = domain_table(dom, outputdir, scale)
        for src in TRANSFERS[dom]:
            tags = dict(domain=dom, source=src)
            cols = [f'{c}:{m}:{t}' for c in configs for m in ('in', src)
                    for t in TRAITS]
            d = d[d[cols].notna().all(1)]
            inv, M = draws(d.block1000.values, n_boot)
            click.echo(f'[{dom} <- {src}] {len(d)} cells')
            # Agreement with the AVIRIS maps
            for t in FOCUS:
                y = d[f'map:{t}:raw'].values
                ok = np.isfinite(y)
                for metric, fn in (('rho', rho_draws), ('r2', r2_draws)):
                    stat = {(c, m): fn(y[ok], d[f'{c}:{m}:{t}:raw'].values[ok],
                                       inv[ok], M)
                            for c in configs for m in ('in', src)}
                    for k, v in contrasts(stat, configs, src).items():
                        agree_rows.append(dict(**tags, trait=t, metric=metric,
                                               contrast=k, n=int(ok.sum()),
                                               **summarize(v)))
                g = {x['contrast']: x for x in agree_rows
                     if x['domain'] == dom and x['source'] == src and
                     x['trait'] == t and x['metric'] == 'rho'}
                click.echo(f'  {t:10s} ρ in/tr ' + '  '.join(
                    f'{c} {g[c + ":in"]["est"]:.2f}/{g[c + ":tr"]["est"]:.2f}'
                    for c in configs))
            # Recovery gains over Env+S
            raw = {c: [x for x in d if x.startswith(f'{c}:')
                       and x.count(':') == 1] for c in ('l8raw', 'l8multi')}
            for target in TARGETS:
                sub = d[d[target].notna() & np.isfinite(d[target])]
                si, sM = draws(sub.block1000.values, n_boot)
                y = sub[target].values
                fs = {'Env+S': es}
                for c in configs:
                    for m in ('in', src):
                        fs[(c, m)] = es + [f'{c}:{m}:{t}' for t in TRAITS]
                for c, v in raw.items():  # raw blocks, apart from the
                    fs[f'{c}_raw'] = es + v  # l8multi emulator
                stat = {k: r2_draws(y, rc.oof_predict(sub, v, target,
                                                      'block1000'), si, sM)
                        for k, v in fs.items()}
                base = stat.pop('Env+S')
                rawd = {f'{c}_raw': stat.pop(f'{c}_raw') for c in raw}
                for k, v in contrasts(stat, configs, src, base, rawd).items():
                    gain_rows.append(dict(**tags, target=target, contrast=k,
                                          n=len(sub), r2_base=base[0],
                                          **summarize(v)))
                g = {x['contrast']: x for x in gain_rows
                     if x['domain'] == dom and x['source'] == src and
                     x['target'] == target}
                click.echo(f'  {target} Env+S {base[0]:.3f}  gain in/tr ' +
                           '  '.join(f'{c} {g[c + ":in"]["est"]:+.3f}/'
                                     f'{g[c + ":tr"]["est"]:+.3f}'
                                     for c in configs) +
                           '  ' + '  '.join(
                               f'{c} {g[c + "_raw:raw"]["est"]:+.3f}'
                               for c in raw))
            for name, rows in (('agreement', agree_rows),
                               ('gain', gain_rows)):
                pd.DataFrame([x for x in rows if x['domain'] == dom]).to_csv(
                    outputdir / f'{name}_{dom}.csv', index=False)


if __name__ == '__main__':
    cli()
