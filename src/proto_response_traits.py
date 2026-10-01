#!/usr/bin/env python
"""
Do AVIRIS-C foliar traits and canopy water explain drought response beyond
climate, terrain and stand structure?

Targets (per cell, from proto_response_metrics.py and proto_trait_dynamics
.py): Landsat NDMI/NIRv resistance, recovery, resilience, recovery time and
stress response; AVIRIS EWT resistance (EWT2016/EWT2013); and at NEON the
lidar mortality fraction (trees live in 2013, dead by 2017-18).

1. Nested models (gradient boosting, 1 km spatial-block CV, 95% block-
   bootstrap CIs over 1 and 5 km blocks):
       B1 (baseline greenness) | Env | Env+S | Env+S+T | Env+S+T+W | Env+S+W
   T = 2013 trait means (14 traits), per-pixel SD of LMA/N/chlorophyll and
   the green-fraction QC; T14 adds the 2014 means; W = 2013/2014 EWT.
2. Residual traits: each trait is cross-fitted on Env+S (out-of-fold) and
   the residual replaces it (Env+S+Tres), so only trait information not
   predictable from climate, terrain and structure can add skill. Signs are
   read from partial dependence and within-stratum Spearman ρ.
3. Conditional: the Env+S -> Env+S+T gain within aridity terciles (1981-2010
   CWD) and, at NEON, within SOAP and TEAK; SHAP importances and
   trait x CWD interaction strength for the full model.
4. Lidar check (--structure; replaces 1-3 for that run): on the cells a
   lidar source covers, do traits still add skill once lidar structure (L)
   is in the model? Ladder Env+S | Env+S+T | Env+S+L | Env+S+L+T |
   Env+S+L+W | Env+S+L+Tres, with Tres residualized against Env+S+L.
   Sources:
     neon2013  NEON 2013 lidar trees (Hemming-Schroeder segmentation), cells
               with >= 5 trees tracked through 2017-18 (the mortality cohort)
     aso       ASO 2014-17 structure composite (fetch_lidar_structure.py);
               snow-off flights Oct 2015 (NEON) / Oct 2016 (Sierra NF), so
               it partly records the die-off: a stricter-than-pre-drought
               control
     lvis2008  LVIS Sep 2008 footprints (fetch_lidar_structure.py)
5. Ablations of the recovery gain (--ablation; with --structure the base
   model also includes that lidar source, on its cells). Gain of each
   trait set over the base, for NDMI recovery and alternative definitions:
     features  full T; T without the green-fraction QC and SD bands; N and
               LMA only; the QC band alone
     cells     all; green-dominated (2013 green-fraction QC above its lower
               quartile); no lidar-dead trees in 2013 (NEON lidar cells);
               no post-2015 rise in N or chlorophyll (cells in the top
               quintile of mean(2016-17) - 2015 for either are dropped:
               rises in dying canopy are likely retrieval artifacts)
     targets   ndmi_recovery (post 2017-19 vs drought 2014-16);
               ndmi_recovery_1718 (post 2017-18); ndmi_recovery_late
               (post 2017-19 vs 2015-16); ndmi_rectime; nirv_recovery

6. Spatial-neighbourhood baseline (--neighbour): B0 is the mean response
   of the training cells within --neighbour-radius of a cell, outside its
   own 1 km block, recomputed in every fold (response_common.
   neighbour_mean). Ladder B0 | Env+S | Env+S+B0 | Env+S+T | Env+S+B0+T |
   Env+S+B0+Tres: do traits add skill that spatial autocorrelation of the
   response does not already give?

Traits use the 2013 v2 mosaic (cross-track normalized, as in
proto_trait_dynamics.py). Alternatives for checks: --traits / --cwc read
other trait and canopy-water files (e.g. the non-v2 2013 mosaic, or
simulated retrievals), --trait-year sets the year that forms T (the next
year forms T14), and --line-z also z-scores each trait within each flight
line (removes between-line level offsets, and any real between-line
contrast with them).

    python proto_response_traits.py $E/hls_results/response_traits \
        -a neon_soap_teak --scale 3
    python proto_response_traits.py $E/hls_results/response_traits \
        -a neon_soap_teak --structure neon2013 --structure aso \
        --structure lvis2008
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import partial_dependence

import response_common as rc
from proto_response_metrics import ENV, S_WALL, S_LIDAR, B1
from proto_trait_dynamics import crosstrack_normalize

TRAITS = ['LMA', 'Nitrogen', 'Chlorophylls', 'Cellulose', 'Lignin', 'Fiber',
          'Sugar', 'Starch', 'NSC', 'Calcium', 'Potassium', 'Phosphorus',
          'Sulfur', 'Phenolics']
SD_TRAITS = ['LMA', 'Nitrogen', 'Chlorophylls']
TARGETS = ['ndmi_resistance', 'ndmi_resistance_late', 'ndmi_recovery',
           'ndmi_resilience', 'ndmi_rectime', 'ndmi_sens',
           'nirv_resistance', 'nirv_recovery', 'nirv_resilience',
           'ewt_resistance', 'mort_frac']
ELEV_BAND = 200
MAX_FIT_CELLS = 200_000
S_ASO = ['aso_rh98', 'aso_rh25', 'aso_rh50', 'aso_rh75', 'aso_pai', 'aso_fhd',
         'aso_cv', 'aso_crown_ratio', 'aso_chm_mean', 'aso_chm_max',
         'aso_cover2', 'aso_cover5', 'aso_frac_tall']
S_LVIS = ['lvis_rh100_mean', 'lvis_rh100_max', 'lvis_rh50_mean',
          'lvis_rh25_mean', 'lvis_rh50_ratio', 'lvis_frac_tall']
STRUCTURES = {'neon2013': S_LIDAR, 'aso': S_ASO, 'lvis2008': S_LVIS}
MIN_COVER = 0.7  # fraction of a cell's pixels with lidar


def trait_paths(aoi, traits=None, cwc=None):
    return (traits or rc.E / 'wdts' / f'{aoi}_traits.nc',
            cwc or rc.E / 'wdts' / f'{aoi}_cwc.nc')


def line_z(a, line):
    """z-score a layer within each flight line"""
    out = a.copy()
    for lid in np.unique(line[line >= 0]):
        m = (line == lid) & np.isfinite(a)
        if m.sum() < 100:
            continue
        out[m] = (a[m] - a[m].mean()) / (a[m].std() or 1)
    return out


def trait_year(d):
    """The year that forms T (the earliest T_qcfc_<year> column)"""
    return min(int(c.rsplit('_', 1)[1]) for c in d.columns
               if c.startswith('T_qcfc_'))


def trait_layers(aoi, valid, k, paths=None, y0=2013, per_line=False):
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    ftr, fcwc = paths or trait_paths(aoi)
    tr = xr.open_dataset(ftr)
    cwc = xr.open_dataset(fcwc) if Path(fcwc).exists() else None
    elev = env.elevation.values
    layers = {}
    for y in (y0, y0 + 1):
        if y not in tr.year.values:
            continue
        fid = tr.flight_id.sel(year=y).values
        for t in TRAITS:
            a = tr[f'{t}_mean'].sel(year=y).values.astype(np.float32)
            a[fid <= 0] = np.nan
            a = crosstrack_normalize(a, fid, elev, transform)
            layers[f'T_{t}_{y}'] = line_z(a, fid) if per_line else a
        if y == y0:
            for t in SD_TRAITS:
                a = tr[f'{t}_sd'].sel(year=y).values.astype(np.float32)
                layers[f'T_{t}_sd_{y}'] = np.where(fid > 0, a, np.nan)
        qc = tr.qc_fc.sel(year=y).values.astype(np.float32)
        layers[f'T_qcfc_{y}'] = np.where((fid > 0) & (qc <= 100), qc / 100,
                                         np.nan)
        if cwc is not None and y in cwc.year.values:
            src = cwc.source_line.sel(year=y).values
            ewt = cwc.ewt980.sel(year=y).values.astype(np.float32)
            ewt = crosstrack_normalize(ewt, src, elev, transform)
            layers[f'W_ewt_{y}'] = line_z(ewt, src) if per_line else ewt
    return rc.cell_table(layers, valid, k)


def build(aoi, k, response_dir, dynamics_dir, paths=None, y0=2013,
          per_line=False):
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, 2019)
    t = trait_layers(aoi, valid, k, paths, y0, per_line)
    m = pd.read_csv(response_dir / f'metrics_{aoi}_{rc.RES * k}m.csv')
    m.columns = [c.replace('resistance1516', 'resistance_late')
                 for c in m.columns]
    d = m.merge(t.drop(columns=[c for c in t.columns if c.startswith(
        ('n_px', 'block'))]), on=['cell_row', 'cell_col'], how='inner')
    dyn = dynamics_dir / f'dynamics_{aoi}_{rc.RES * k}m.csv'
    if dyn.exists():
        cols = ['cell_row', 'cell_col', 'ewt_resistance', 'mort_frac',
                'mort_n']
        dd = pd.read_csv(dyn, usecols=lambda c: c in cols)
        d = d.merge(dd, on=['cell_row', 'cell_col'], how='left')
    d['arid'] = pd.qcut(d.cwd_clim, 3, labels=['wet', 'mid', 'dry'])
    if aoi == 'neon_soap_teak':
        # SOAP is the western (lower) half of the AOI, TEAK the eastern
        d['site'] = np.where(d.cell_col < d.cell_col.max() / 2, 'SOAP',
                             'TEAK')
    return d


def structure_cells(d, aoi, k, source):
    """Cells covered by a lidar source, with its structure columns"""
    if source == 'neon2013':
        return d[d.mort_frac.notna()] if 'mort_frac' in d else d.iloc[:0]
    f = rc.E / 'lidar' / f'{aoi}_lidar.nc'
    if not f.exists():
        return d.iloc[:0]
    ds = xr.open_dataset(f)
    cov = 'aso_cov' if source == 'aso' else 'lvis_cov'
    cols = STRUCTURES[source] + [cov]
    if not all(c in ds for c in cols):
        return d.iloc[:0]
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, 2019)
    layers = {c: ds[c].values.astype(np.float32) for c in cols}
    layers[cov] = np.nan_to_num(layers[cov])
    L = rc.cell_table(layers, valid, k)
    L = L[L[cov] >= MIN_COVER][['cell_row', 'cell_col'] + cols]
    return d.drop(columns=[c for c in cols if c in d]).merge(
        L, on=['cell_row', 'cell_col'], how='inner')


def lidar_check(d, source, targets, n_boot):
    """Do traits add skill beyond lidar structure on the covered cells?"""
    L = STRUCTURES[source]
    _, T, W = feature_sets(d)
    es = ENV + S_WALL
    R = pd.DataFrame({c.replace('T_', 'R_'): rc.crossfit_residuals(
        d, c, es + L, 'block1000') for c in T}, index=d.index)
    d = pd.concat([d, R], axis=1)
    fs = {'Env+S': es, 'Env+S+T': es + T, 'Env+S+L': es + L,
          'Env+S+L+T': es + L + T, 'Env+S+L+W': es + L + W,
          'Env+S+L+Tres': es + L + list(R.columns)}
    pairs = [('Env+S', 'Env+S+T'), ('Env+S', 'Env+S+L'),
             ('Env+S+L', 'Env+S+L+T'), ('Env+S+L', 'Env+S+L+W'),
             ('Env+S+L', 'Env+S+L+Tres')]
    rows = []
    for t in targets:
        if t not in d or d[t].notna().sum() < 500:
            continue
        sub = d[d[t].notna() & np.isfinite(d[t])]
        w = 'mort_n' if t == 'mort_frac' else None
        r, _ = rc.ladder(sub, t, fs, 'block1000', weight=w, n_boot=n_boot,
                         pairs=pairs, extra_blocks=('block5000',))
        rows += r
        g = {f'{x["features"]}-{x["compare"]}': x for x in r
             if x['compare'] and x['boot_blocks'] == 'block1000'}
        msg = '  '.join(
            f'{k.split("-")[0].replace("Env+S+", "+")} '
            f'{g[k]["r2"]:+.3f} [{g[k]["lo"]:+.3f},{g[k]["hi"]:+.3f}]'
            for k in ('Env+S+T-Env+S', 'Env+S+L-Env+S',
                      'Env+S+L+T-Env+S+L', 'Env+S+L+Tres-Env+S+L'))
        click.echo(f'  {t:20s} n={len(sub):6d}  {msg}')
    return rows


ABLATION_TARGETS = ['ndmi_recovery', 'ndmi_recovery_1718',
                    'ndmi_recovery_late', 'ndmi_rectime', 'nirv_recovery']
RISE_Q = 0.8


def trait_rise(aoi, valid, k, paths=None):
    """Per cell: mean(2016-17) - 2015 of N and chlorophyll"""
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    tr = xr.open_dataset((paths or trait_paths(aoi))[0])
    elev = env.elevation.values
    layers = {}
    for t in ('Nitrogen', 'Chlorophylls'):
        v = {}
        for y in (2015, 2016, 2017):
            fid = tr.flight_id.sel(year=y).values
            a = tr[f'{t}_mean'].sel(year=y).values.astype(np.float32)
            a[fid <= 0] = np.nan
            v[y] = crosstrack_normalize(a, fid, elev, transform)
        layers[f'rise_{t}'] = (v[2016] + v[2017]) / 2 - v[2015]
    return rc.cell_table(layers, valid, k)[
        ['cell_row', 'cell_col', 'rise_Nitrogen', 'rise_Chlorophylls']]


def run_ablation(d, aoi, k, base, n_boot, label, paths=None):
    """Gain of trait subsets over base, on several cell subsets and
    recovery definitions"""
    d = d.copy()
    y0 = trait_year(d)
    with np.errstate(all='ignore'):
        drought = d[[f'ndmi_{y}' for y in (2014, 2015, 2016)]].mean(1)
        late = d[['ndmi_2015', 'ndmi_2016']].mean(1)
        d['ndmi_recovery_1718'] = d[['ndmi_2017', 'ndmi_2018']].mean(1) \
            - drought
        d['ndmi_recovery_late'] = d[[f'ndmi_{y}' for y in
                                     (2017, 2018, 2019)]].mean(1) - late
    env = rc.open_env(aoi)
    rise = trait_rise(aoi, rc.undisturbed(env, 2019), k, paths)
    d = d.merge(rise, on=['cell_row', 'cell_col'], how='left')
    _, T, _ = feature_sets(d)
    qc = f'T_qcfc_{y0}'
    qc_sd = [qc] + [f'T_{t}_sd_{y0}' for t in SD_TRAITS]
    tsets = {'T': T, 'T-noQC-SD': [c for c in T if c not in qc_sd],
             'N+LMA': [f'T_Nitrogen_{y0}', f'T_LMA_{y0}'],
             'QC': [qc]}
    q = d[qc].quantile(0.25)
    no_rise = ~((d.rise_Nitrogen > d.rise_Nitrogen.quantile(RISE_Q)) |
                (d.rise_Chlorophylls >
                 d.rise_Chlorophylls.quantile(RISE_Q)))
    subsets = {'all': np.ones(len(d), bool),
               'green': (d[qc] > q).values,
               'no_rise': no_rise.values}
    if 'lidar_dead2013' in d:
        subsets['no_dead2013'] = (d.lidar_dead2013 == 0).values
    rows = []
    for t in ABLATION_TARGETS:
        for sname, m in subsets.items():
            sub = d[m & d[t].notna().values & np.isfinite(d[t]).values]
            if len(sub) < 500:
                continue
            fs = {'base': base, **{n: base + c for n, c in tsets.items()}}
            r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                             pairs=[('base', n) for n in tsets])
            for x in r:
                x.update(subset=sname, base_set=label)
            rows += r
            g = {x['features']: x for x in r
                 if x['compare'] and x['boot_blocks'] == 'block1000'}
            click.echo(f'  {t:20s} {sname:12s} n={len(sub):6d}  ' + '  '.join(
                f'{n} {g[n]["r2"]:+.3f} [{g[n]["lo"]:+.3f},'
                f'{g[n]["hi"]:+.3f}]' for n in tsets))
    return rows


def neighbour_ladder(d, targets, n_boot, radius, scale_m):
    """Trait gains over a spatial-neighbourhood baseline (B0)"""
    _, T, _ = feature_sets(d)
    R = residualize(d, T, 'block1000')
    d = pd.concat([d, R], axis=1)
    es = ENV + S_WALL
    fs = {'B0': [], 'Env+S': es, 'Env+S+B0': es, 'Env+S+T': es + T,
          'Env+S+B0+T': es + T, 'Env+S+B0+Tres': es + list(R.columns)}
    pairs = [('B0', 'Env+S'), ('Env+S', 'Env+S+B0'), ('Env+S', 'Env+S+T'),
             ('Env+S+B0', 'Env+S+B0+T'), ('Env+S+B0', 'Env+S+B0+Tres')]
    rows = []
    for t in targets:
        if t not in d or t == 'mort_frac' or d[t].notna().sum() < 500:
            continue
        sub = d[d[t].notna() & np.isfinite(d[t])]

        def nb(df, train, t=t):
            return rc.neighbour_mean(df, t, train, radius, scale_m)[:, None]
        ff = {k: nb for k in fs if 'B0' in k}
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                         pairs=pairs, fold_features=ff)
        rows += r
        g = {f'{x["features"]}-{x["compare"]}': x for x in r
             if x['compare'] and x['boot_blocks'] == 'block1000'}
        a = {x['features']: x['r2'] for x in r
             if not x['compare'] and x['boot_blocks'] == 'block1000'}
        click.echo(f'  {t:20s} B0 {a["B0"]:.3f} Env+S {a["Env+S"]:.3f} '
                   f'Env+S+B0 {a["Env+S+B0"]:.3f}  ' + '  '.join(
                       f'{k} {g[k]["r2"]:+.3f} [{g[k]["lo"]:+.3f},'
                       f'{g[k]["hi"]:+.3f}]' for k in (
                           'Env+S+B0+T-Env+S+B0', 'Env+S+B0+Tres-Env+S+B0')))
    return rows


def feature_sets(d):
    y0 = trait_year(d)
    T = [f'T_{t}_{y0}' for t in TRAITS] + \
        [f'T_{t}_sd_{y0}' for t in SD_TRAITS] + [f'T_qcfc_{y0}']
    T = [c for c in T if d[c].notna().any()]  # simulated files lack SD/QC
    T14 = [f'T_{t}_{y0 + 1}' for t in TRAITS] + [f'T_qcfc_{y0 + 1}']
    W = [c for c in (f'W_ewt_{y0}', f'W_ewt_{y0 + 1}') if c in d]
    es = ENV + S_WALL
    fs = {'B1': B1, 'Env': ENV, 'Env+S': es, 'Env+S+T': es + T}
    if all(c in d for c in T14):
        fs['Env+S+T+T14'] = es + T + T14
    if W:  # canopy water (not yet fetched for every AOI)
        fs.update({'Env+S+W': es + W, 'Env+S+T+W': es + T + W})
    return fs, T, W


def residualize(d, T, blocks):
    out = {}
    for c in T:
        out[c.replace('T_', 'R_')] = rc.crossfit_residuals(
            d, c, ENV + S_WALL, blocks)
    return pd.DataFrame(out, index=d.index)


def pd_sign(model, X, col, grid=20):
    """Direction of the partial dependence: Spearman of PD vs grid"""
    from scipy.stats import spearmanr
    r = partial_dependence(model, X, [col], grid_resolution=grid,
                           kind='average')
    return spearmanr(r['grid_values'][0], r['average'][0])[0], \
        float(np.ptp(r['average'][0]))


def shap_summary(d, cols, target, max_n=20000):
    import shap
    sub = d[d[target].notna()]
    if len(sub) > max_n:
        sub = sub.sample(max_n, random_state=0)
    m = rc.hgb().fit(sub[cols], sub[target])
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(sub[cols])
    imp = pd.Series(np.abs(sv).mean(0), index=cols)
    # trait x CWD interaction: SHAP interaction values on a subsample
    small = sub[cols].iloc[:2000]
    iv = ex.shap_interaction_values(small)
    j = cols.index('cwd_clim')
    inter = pd.Series(np.abs(iv[:, :, j]).mean(0) * 2, index=cols)
    return imp, inter


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--response-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/response')
@click.option('--dynamics-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/trait_dynamics')
@click.option('--scale', 'scales', multiple=True, type=int, default=[3],
              show_default=True)
@click.option('-t', '--target', 'targets', multiple=True,
              default=TARGETS, show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--structure', 'structures', multiple=True,
              type=click.Choice(list(STRUCTURES)),
              help='Run only the lidar check with these sources')
@click.option('--ablation', is_flag=True,
              help='Run only the recovery ablations (with --structure: '
                   'on top of each lidar source)')
@click.option('--traits', 'traits_path', type=click.Path(path_type=Path),
              help='Trait file (default wdts/<aoi>_traits.nc; one AOI)')
@click.option('--cwc', 'cwc_path', type=click.Path(path_type=Path),
              help='Canopy-water file (default wdts/<aoi>_cwc.nc; one AOI)')
@click.option('--trait-year', default=2013, show_default=True,
              help='Year of the traits that form T')
@click.option('--line-z', is_flag=True,
              help='z-score each trait within each flight line')
@click.option('--neighbour', is_flag=True,
              help='Run only the spatial-neighbourhood baseline ladder')
@click.option('--neighbour-radius', default=2000, show_default=True)
@click.option('--gains-only', is_flag=True,
              help='Ladder gains only: skip the conditional gains, residual '
                   'directions and SHAP')
def main(outputdir, aois, response_dir, dynamics_dir, scales, targets,
         n_boot, structures, ablation, traits_path, cwc_path, trait_year,
         line_z, neighbour, neighbour_radius, gains_only):
    if (traits_path or cwc_path) and len(aois) > 1:
        raise click.UsageError('--traits/--cwc take a single --aoi')
    outputdir.mkdir(parents=True, exist_ok=True)
    rows, cond_rows, res_rows, imp_rows = [], [], [], []
    for aoi in aois:
        for k in scales:
            scale_m = rc.RES * k
            paths = trait_paths(aoi, traits_path, cwc_path)
            d = build(aoi, k, response_dir, dynamics_dir, paths, trait_year,
                      line_z)
            if neighbour:
                r = neighbour_ladder(d, targets, n_boot, neighbour_radius,
                                     scale_m)
                for x in r:
                    x.update(aoi=aoi, scale_m=scale_m,
                             radius_m=neighbour_radius)
                f = outputdir / 'neighbour.csv'
                old = pd.read_csv(f) if f.exists() else pd.DataFrame()
                if len(old):
                    old = old[~((old.aoi == aoi) & (old.scale_m == scale_m) &
                                (old.radius_m == neighbour_radius))]
                pd.concat([old, pd.DataFrame(r)]).to_csv(f, index=False)
                continue
            if ablation:
                runs = [('Env+S', d, ENV + S_WALL)]
                for src in structures:
                    sub = structure_cells(d, aoi, k, src)
                    if len(sub) >= 500:
                        runs.append((f'Env+S+L:{src}', sub,
                                     ENV + S_WALL + STRUCTURES[src]))
                for label, dd, base in runs:
                    click.echo(f'[{aoi} {scale_m} m] ablation, base {label}'
                               f': {len(dd)} cells')
                    r = run_ablation(dd, aoi, k, base, n_boot, label,
                                     paths)
                    for x in r:
                        x.update(aoi=aoi, scale_m=scale_m)
                    f = outputdir / 'ablation.csv'
                    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
                    if len(old):
                        old = old[~((old.aoi == aoi) &
                                    (old.scale_m == scale_m) &
                                    (old.base_set == label))]
                    pd.concat([old, pd.DataFrame(r)]).to_csv(f, index=False)
                continue
            if structures:
                for src in structures:
                    sub = structure_cells(d, aoi, k, src)
                    click.echo(f'[{aoi} {scale_m} m] lidar check {src}: '
                               f'{len(sub)} cells')
                    if len(sub) < 500:
                        continue
                    r = lidar_check(sub, src, targets, n_boot)
                    for x in r:
                        x.update(aoi=aoi, scale_m=scale_m, check=src)
                    f = outputdir / f'lidar_check_{src}.csv'
                    old = (pd.read_csv(f) if f.exists() else pd.DataFrame())
                    if len(old):
                        old = old[~((old.aoi == aoi) &
                                    (old.scale_m == scale_m))]
                    pd.concat([old, pd.DataFrame(r)]).to_csv(f, index=False)
                continue
            if len(d) > MAX_FIT_CELLS:
                d = d.sample(MAX_FIT_CELLS, random_state=0)
            fs, T, W = feature_sets(d)
            R = residualize(d, T, 'block1000')
            d = pd.concat([d, R], axis=1)
            fs['Env+S+Tres'] = ENV + S_WALL + list(R.columns)
            d.to_csv(outputdir / f'cells_{aoi}_{scale_m}m.csv', index=False)
            click.echo(f'[{aoi} {scale_m} m] {len(d)} cells')
            for t in targets:
                if t not in d or d[t].notna().sum() < 500:
                    continue
                sub = d[d[t].notna() & np.isfinite(d[t])]
                w = 'mort_n' if t == 'mort_frac' else None
                pairs = [p for p in [
                    ('Env', 'Env+S'), ('Env+S', 'Env+S+T'),
                    ('Env+S+T', 'Env+S+T+T14'), ('Env+S', 'Env+S+W'),
                    ('Env+S+T', 'Env+S+T+W'), ('Env+S', 'Env+S+Tres'),
                    ('B1', 'Env+S')] if p[1] in fs]
                r, _ = rc.ladder(sub, t, fs, 'block1000', weight=w,
                                 n_boot=n_boot, pairs=pairs,
                                 extra_blocks=('block5000',))
                for x in r:
                    x.update(aoi=aoi, scale_m=scale_m)
                rows += r
                gain = {f'{x["features"]}-{x["compare"]}': x for x in r
                        if x['compare'] and x['boot_blocks'] == 'block1000'}
                base = [x for x in r if x['features'] == 'Env+S'
                        and not x['compare']][0]
                click.echo(f'  {t:20s} Env+S {base["r2"]:.3f}  ' + '  '.join(
                    f'{name} {gain[key]["r2"]:+.3f} [{gain[key]["lo"]:+.3f},'
                    f'{gain[key]["hi"]:+.3f}]' for name, key in (
                        ('+T', 'Env+S+T-Env+S'), ('+W', 'Env+S+W-Env+S'),
                        ('+Tres', 'Env+S+Tres-Env+S')) if key in gain))

                if gains_only:
                    pd.DataFrame(rows).to_csv(
                        outputdir / 'response_traits_ladder.csv', index=False)
                    continue

                # Conditional gains
                splits = [('arid', v) for v in ('wet', 'mid', 'dry')]
                if 'site' in sub:
                    splits += [('site', 'SOAP'), ('site', 'TEAK')]
                for col, val in splits:
                    s = sub[sub[col] == val]
                    if len(s) < 500:
                        continue
                    cr, _ = rc.ladder(s, t, {'Env+S': fs['Env+S'],
                                             'Env+S+T': fs['Env+S+T']},
                                      'block1000', weight=w, n_boot=n_boot)
                    for x in cr:
                        if x['compare'] or x['features'] == 'Env+S':
                            x.update(aoi=aoi, scale_m=scale_m, split=col,
                                     group=val)
                            cond_rows.append(x)

                # Residual-trait directions
                m = rc.hgb().fit(sub[fs['Env+S+Tres']], sub[t])
                strata = (sub.arid.astype(str) + '_' +
                          (sub.elevation // ELEV_BAND).astype(int)
                          .astype(str))
                uni = {u['feature']: u for u in rc.within_strata_rho(
                    sub, list(R.columns) + T + W, t, strata)}
                for c in list(R.columns) + T + W:
                    sign, rng = (pd_sign(m, sub[fs['Env+S+Tres']], c)
                                 if c in R.columns else (np.nan, np.nan))
                    u = uni.get(c, {})
                    res_rows.append(dict(aoi=aoi, scale_m=scale_m, target=t,
                                         feature=c, pd_sign=sign,
                                         pd_range=rng, rho=u.get('rho'),
                                         rho_within=u.get('rho_within')))

                # SHAP for the full trait model
                imp, inter = shap_summary(
                    sub, fs.get('Env+S+T+W', fs['Env+S+T']), t)
                for c in imp.index:
                    imp_rows.append(dict(aoi=aoi, scale_m=scale_m, target=t,
                                         feature=c, mean_abs_shap=imp[c],
                                         inter_cwd=inter[c]))
                for name, data in (('response_traits_ladder', rows),
                                   ('response_traits_conditional',
                                    cond_rows),
                                   ('response_traits_residual', res_rows),
                                   ('response_traits_shap', imp_rows)):
                    pd.DataFrame(data).to_csv(outputdir / f'{name}.csv',
                                              index=False)
            fig_ladder(pd.DataFrame(rows), aoi, scale_m,
                       outputdir / f'ladder_{aoi}_{scale_m}m.png')


def fig_ladder(r, aoi, scale_m, path):
    # r['compare'], not r.compare, which is the DataFrame.compare method
    r = r.fillna({'compare': ''})
    r = r[(r.aoi == aoi) & (r.scale_m == scale_m)
          & (r.boot_blocks == 'block1000') & (r['compare'] != '')]
    steps = ['Env+S-Env', 'Env+S+T-Env+S', 'Env+S+W-Env+S',
             'Env+S+Tres-Env+S', 'Env+S+T+W-Env+S+T']
    r = r.assign(step=r.features + '-' + r['compare'])
    r = r[r.step.isin(steps)]
    targets = list(dict.fromkeys(r.target))
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(targets) + 1.5))
    for i, s in enumerate(steps):
        g = r[r.step == s].set_index('target').reindex(targets)
        y = np.arange(len(targets)) + (i - 2) * 0.15
        ax.errorbar(g.r2, y, xerr=[g.r2 - g.lo, g.hi - g.r2], fmt='o',
                    ms=4, label=s.replace('-', ' vs '))
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=8)
    ax.set_xlabel('ΔR² (95% block-bootstrap CI, 1 km blocks)')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_title(f'{aoi} {scale_m} m: gain in R² from each block', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
