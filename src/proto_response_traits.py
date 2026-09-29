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

Traits use the 2013 v2 mosaic (cross-track normalized, as in
proto_trait_dynamics.py).

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


def trait_layers(aoi, valid, k):
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    tr = xr.open_dataset(rc.E / 'wdts' / f'{aoi}_traits.nc')
    f = rc.E / 'wdts' / f'{aoi}_cwc.nc'
    cwc = xr.open_dataset(f) if f.exists() else None
    elev = env.elevation.values
    layers = {}
    for y in (2013, 2014):
        fid = tr.flight_id.sel(year=y).values
        for t in TRAITS:
            a = tr[f'{t}_mean'].sel(year=y).values.astype(np.float32)
            a[fid <= 0] = np.nan
            layers[f'T_{t}_{y}'] = crosstrack_normalize(a, fid, elev,
                                                        transform)
        if y == 2013:
            for t in SD_TRAITS:
                a = tr[f'{t}_sd'].sel(year=y).values.astype(np.float32)
                layers[f'T_{t}_sd_{y}'] = np.where(fid > 0, a, np.nan)
        qc = tr.qc_fc.sel(year=y).values.astype(np.float32)
        layers[f'T_qcfc_{y}'] = np.where((fid > 0) & (qc <= 100), qc / 100,
                                         np.nan)
        if cwc is not None:
            ewt = cwc.ewt980.sel(year=y).values.astype(np.float32)
            layers[f'W_ewt_{y}'] = crosstrack_normalize(
                ewt, cwc.source_line.sel(year=y).values, elev, transform)
    return rc.cell_table(layers, valid, k)


def build(aoi, k, response_dir, dynamics_dir):
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, 2019)
    t = trait_layers(aoi, valid, k)
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


def trait_rise(aoi, valid, k):
    """Per cell: mean(2016-17) - 2015 of N and chlorophyll"""
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    tr = xr.open_dataset(rc.E / 'wdts' / f'{aoi}_traits.nc')
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


def run_ablation(d, aoi, k, base, n_boot, label):
    """Gain of trait subsets over base, on several cell subsets and
    recovery definitions"""
    d = d.copy()
    with np.errstate(all='ignore'):
        drought = d[[f'ndmi_{y}' for y in (2014, 2015, 2016)]].mean(1)
        late = d[['ndmi_2015', 'ndmi_2016']].mean(1)
        d['ndmi_recovery_1718'] = d[['ndmi_2017', 'ndmi_2018']].mean(1) \
            - drought
        d['ndmi_recovery_late'] = d[[f'ndmi_{y}' for y in
                                     (2017, 2018, 2019)]].mean(1) - late
    env = rc.open_env(aoi)
    rise = trait_rise(aoi, rc.undisturbed(env, 2019), k)
    d = d.merge(rise, on=['cell_row', 'cell_col'], how='left')
    _, T, _ = feature_sets(d)
    qc_sd = ['T_qcfc_2013'] + [f'T_{t}_sd_2013' for t in SD_TRAITS]
    tsets = {'T': T, 'T-noQC-SD': [c for c in T if c not in qc_sd],
             'N+LMA': ['T_Nitrogen_2013', 'T_LMA_2013'],
             'QC': ['T_qcfc_2013']}
    q = d.T_qcfc_2013.quantile(0.25)
    no_rise = ~((d.rise_Nitrogen > d.rise_Nitrogen.quantile(RISE_Q)) |
                (d.rise_Chlorophylls >
                 d.rise_Chlorophylls.quantile(RISE_Q)))
    subsets = {'all': np.ones(len(d), bool),
               'green': (d.T_qcfc_2013 > q).values,
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


def feature_sets(d):
    T = [f'T_{t}_2013' for t in TRAITS] + \
        [f'T_{t}_sd_2013' for t in SD_TRAITS] + ['T_qcfc_2013']
    T14 = [f'T_{t}_2014' for t in TRAITS] + ['T_qcfc_2014']
    W = [c for c in ('W_ewt_2013', 'W_ewt_2014') if c in d]
    es = ENV + S_WALL
    fs = {'B1': B1, 'Env': ENV, 'Env+S': es, 'Env+S+T': es + T,
          'Env+S+T+T14': es + T + T14}
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
def main(outputdir, aois, response_dir, dynamics_dir, scales, targets,
         n_boot, structures, ablation):
    outputdir.mkdir(parents=True, exist_ok=True)
    rows, cond_rows, res_rows, imp_rows = [], [], [], []
    for aoi in aois:
        for k in scales:
            scale_m = rc.RES * k
            d = build(aoi, k, response_dir, dynamics_dir)
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
                    r = run_ablation(dd, aoi, k, base, n_boot, label)
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
