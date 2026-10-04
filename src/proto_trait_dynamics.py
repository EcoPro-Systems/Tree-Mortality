#!/usr/bin/env python
"""
AVIRIS-C trait and canopy-water dynamics (2013-2018) vs Landsat change, as
predictors of drought response and mortality.

Part 1, canopy-water trajectories. Per cell, EWT (980 nm Beer-Lambert fit,
fetch_wdts_cwc.py) for each June flight 2013-2018 gives AVIRIS-native
response metrics with 2013 as the base (note 2013 is the drought's second
year):
    ewt_resistance  EWT2016 / EWT2013
    ewt_recovery    mean(EWT2017-18) / EWT2016
    ewt_resilience  mean(EWT2017-18) / EWT2013
These are compared with the Landsat metrics (proto_response_metrics.py) and
with Landsat June NDMI ratios over the same years. Trajectories of EWT, LMA,
N and chlorophyll are plotted by eventual fate (NEON lidar trees: share of
2013-live trees dead by 2017-18).

Part 2, trait change vs multispectral change. Predict
    - the lidar mortality fraction (trees live in 2013, dead 2017-18, NEON);
    - Landsat resilience and recovery (NDMI, NIRv);
    - recovery failure: bottom quintile of NDMI resilience (AUC)
from
    L    Landsat June ΔNDMI, ΔNIRv, ΔNDVI (c2 composites)
    A    AVIRIS ΔEWT, ΔChl, ΔLMA, ΔN (+ ΔQC green fraction)
    L+A, and each on top of Env+S.
Intervals: 2013->2015 (both trait mosaics cross-year calibrated, "_v2") and
2013->2014. Before differencing, each year's traits are cross-track
normalized: per flight line, the trait is regressed on a quadratic in
across-track position plus a quadratic in elevation, and the across-track
part is removed (this targets the view-angle striping seen in ΔLMA).

    python proto_trait_dynamics.py $E/hls_results/trait_dynamics -a neon_soap_teak
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import response_common as rc
from proto_response_metrics import ENV, S_WALL

TRAITS = ['LMA', 'Nitrogen', 'Chlorophylls']
AV_YEARS = list(range(2013, 2019))
INTERVALS = [(2013, 2015), (2013, 2014)]
L_VARS = ['ndmi', 'nirv', 'ndvi']
FATE_BINS = [0, 0.1, 0.5, 1.0001]
FATE_LABELS = ['survived (<10% died)', 'partial (10-50%)', 'died (>50%)']
MIN_TREES = 5


def crosstrack_normalize(a, line, elev, transform):
    """Remove the across-track quadratic trend per flight line.

    The along-track direction is the first principal axis of each line's
    footprint; across-track position is the projection on the second."""
    out = a.copy()
    rows, cols = np.indices(a.shape)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    for lid in np.unique(line[line >= 0]):
        m = (line == lid) & np.isfinite(a) & np.isfinite(elev)
        if m.sum() < 1000:
            continue
        xy = np.column_stack([x[m], y[m]])
        xy = xy - xy.mean(0)
        _, _, vt = np.linalg.svd(xy[::max(1, len(xy) // 5000)],
                                 full_matrices=False)
        u = xy @ vt[1]
        u = u / (np.abs(u).max() or 1)
        e = (elev[m] - elev[m].mean()) / (elev[m].std() or 1)
        X = np.column_stack([np.ones_like(u), u, u * u, e, e * e])
        beta, *_ = np.linalg.lstsq(X, a[m], rcond=None)
        trend = beta[1] * u + beta[2] * (u * u - np.mean(u * u))
        out[m] = a[m] - trend
    return out


def lidar_cohort(transform, shape):
    """Per-pixel counts of cohort-A trees (live 2013, same status 2017 and
    2018) and of those dead by 2017"""
    from proto_predisposition_neon import load_trees, pixel_index
    df = load_trees(rc.E / 'hemming_schroeder2023')
    a = df[(df.live2013 == 1) & (df.live2017 == df.live2018)
           & df.live2017.notna()]
    row, col = pixel_index(a, transform, shape)
    ok = row >= 0
    idx = row[ok] * shape[1] + col[ok]
    n = np.bincount(idx, minlength=shape[0] * shape[1]).astype(float)
    died = np.bincount(idx, weights=(a.live2017.values[ok] == 0),
                       minlength=shape[0] * shape[1])
    n, died = n.reshape(shape), died.reshape(shape)
    n[n == 0], died[n == 0] = np.nan, np.nan
    return n, died


def build(aoi, k, response_dir, variant_june='c2'):
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    tr = xr.open_dataset(rc.E / 'wdts' / f'{aoi}_traits.nc')
    cwc = xr.open_dataset(rc.E / 'wdts' / f'{aoi}_cwc.nc')
    lj = xr.open_dataset(rc.E / 'landsat_composites' /
                         f'{aoi}_{variant_june}_doy145-190.nc')
    elev = env.elevation.values
    valid = rc.undisturbed(env, 2019)
    layers = {}
    for y in AV_YEARS:
        fid = tr.flight_id.sel(year=y).values
        for t in TRAITS:
            a = tr[f'{t}_mean'].sel(year=y).values.astype(np.float32)
            a[fid <= 0] = np.nan
            layers[f'{t}_{y}'] = crosstrack_normalize(a, fid, elev, transform)
            layers[f'{t}_raw_{y}'] = a
        qc = tr.qc_fc.sel(year=y).values.astype(np.float32)
        layers[f'qcfc_{y}'] = np.where((fid > 0) & (qc <= 100), qc / 100,
                                       np.nan)
        ewt = cwc.ewt980.sel(year=y).values.astype(np.float32)
        line = cwc.source_line.sel(year=y).values
        layers[f'ewt_{y}'] = crosstrack_normalize(ewt, line, elev, transform)
        layers[f'ewt_raw_{y}'] = ewt
    for v in L_VARS:
        for y in AV_YEARS:
            layers[f'lj_{v}_{y}'] = lj[v].sel(year=y).values
    if aoi == 'neon_soap_teak':
        layers['hs_n'], layers['hs_died'] = lidar_cohort(transform, shape)
    d = rc.cell_table(layers, valid, k)
    with np.errstate(all='ignore'):
        e = {y: d[f'ewt_{y}'] for y in AV_YEARS}
        d['ewt_resistance'] = e[2016] / e[2013]
        d['ewt_recovery'] = (e[2017] + e[2018]) / 2 / e[2016]
        d['ewt_resilience'] = (e[2017] + e[2018]) / 2 / e[2013]
        n = {y: d[f'lj_ndmi_{y}'] for y in AV_YEARS}
        d['ljndmi_resistance'] = n[2016] - n[2013]
        d['ljndmi_recovery'] = (n[2017] + n[2018]) / 2 - n[2016]
        d['ljndmi_resilience'] = (n[2017] + n[2018]) / 2 - n[2013]
        for a, b in INTERVALS:
            tag = f'd{b % 100}{a % 100}'
            for t in TRAITS + ['ewt', 'qcfc']:
                d[f'A_{t}_{tag}'] = d[f'{t}_{b}'] - d[f'{t}_{a}']
            for t in TRAITS + ['ewt']:
                d[f'Araw_{t}_{tag}'] = d[f'{t}_raw_{b}'] - d[f'{t}_raw_{a}']
            for v in L_VARS:
                d[f'L_{v}_{tag}'] = d[f'lj_{v}_{b}'] - d[f'lj_{v}_{a}']
        if 'hs_n' in d:
            d['mort_frac'] = d.hs_died / d.hs_n
            d['mort_n'] = d.hs_n * d.n_px
            d.loc[d.mort_n < MIN_TREES, ['mort_frac', 'mort_n']] = np.nan
    # Landsat Lloret metrics and Env/S from the response-metrics run
    m = pd.read_csv(response_dir / f'metrics_{aoi}_{rc.RES * k}m.csv')
    keep = ['cell_row', 'cell_col'] + [c for c in m.columns if any(
        c.endswith(s) for s in ('_resistance', '_recovery', '_resilience',
                                '_rectime', '_sens'))] + ENV + S_WALL
    d = d.merge(m[[c for c in keep if c in m.columns]],
                on=['cell_row', 'cell_col'], how='left',
                suffixes=('', '_m'))
    return d


def fig_trajectories(d, path, title):
    d = d[d.mort_frac.notna()]
    fate = pd.cut(d.mort_frac, FATE_BINS, labels=FATE_LABELS, right=False)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for ax, t, lab in zip(axes, ['ewt', 'LMA', 'Nitrogen', 'Chlorophylls'],
                          ['EWT 980 (cm)', 'LMA', 'N', 'Chlorophyll']):
        for f, col in zip(FATE_LABELS, ['tab:green', 'tab:orange',
                                         'tab:red']):
            g = d[fate == f]
            q = np.array([np.nanpercentile(g[f'{t}_{y}'], [25, 50, 75])
                          for y in AV_YEARS])
            ax.plot(AV_YEARS, q[:, 1], '-o', color=col, ms=3,
                    label=f'{f} (n={len(g)})')
            ax.fill_between(AV_YEARS, q[:, 0], q[:, 2], color=col,
                            alpha=0.15)
        ax.axvspan(2012.5, 2016.5, color='orange', alpha=0.07)
        ax.set_title(lab, fontsize=9)
        if t != 'ewt':
            for y in (2013, 2015):
                ax.axvline(y, color='k', lw=0.4, ls=':')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{title}: AVIRIS-C June trajectories by fate of 2013-live '
                 'lidar trees (dotted: cross-year calibrated trait dates)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def auc_oof(d, cols, target, blocks):
    p = rc.oof_predict(d, cols, target, blocks)
    return roc_auc_score(d[target], p), p


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--response-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/response')
@click.option('--scale', 'scales', multiple=True, type=int, default=[1, 3],
              show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--interval', 'intervals', multiple=True,
              help='Only these change intervals (e.g. d1513)')
@click.option('--fold-seed', type=int, default=None,
              help='Shuffle 1 km blocks into folds with this seed (default: '
                   'the deterministic GroupKFold assignment)')
@click.option('--tag', default='',
              help='Suffix of the model output files; a tagged run leaves '
                   'the cell tables and figures alone')
def main(outputdir, aois, response_dir, scales, n_boot, intervals,
         fold_seed, tag):
    rc.FOLD_SEED = fold_seed
    outputdir.mkdir(parents=True, exist_ok=True)
    corr_rows, ladder_rows, auc_rows = [], [], []
    for aoi in aois:
        for k in scales:
            scale_m = rc.RES * k
            d = build(aoi, k, response_dir)
            if not tag:
                d.to_csv(outputdir / f'dynamics_{aoi}_{scale_m}m.csv',
                         index=False)
            click.echo(f'[{aoi} {scale_m} m] {len(d)} cells')
            yrs = {y: np.nanmedian(d[f'ewt_raw_{y}']) for y in AV_YEARS}
            click.echo('  median EWT: ' + ', '.join(
                f'{y} {v:.3f}' for y, v in yrs.items()))

            # Part 1: AVIRIS-native vs Landsat metrics
            pairs = [('ewt_resistance', 'ljndmi_resistance'),
                     ('ewt_recovery', 'ljndmi_recovery'),
                     ('ewt_resilience', 'ljndmi_resilience'),
                     ('ewt_resistance', 'ndmi_resistance'),
                     ('ewt_resilience', 'ndmi_resilience'),
                     ('ewt_resistance', 'nirv_resistance'),
                     ('ewt_resilience', 'nirv_resilience')]
            if 'mort_frac' in d:
                pairs += [(c, 'mort_frac') for c in
                          ['ewt_resistance', 'ewt_resilience',
                           'ljndmi_resistance', 'ndmi_resistance',
                           'ndmi_resilience', 'nirv_resilience']]
            for a, b in pairs:
                ok = d[a].notna() & d[b].notna() & np.isfinite(d[a])
                corr_rows.append(dict(aoi=aoi, scale_m=scale_m, x=a, y=b,
                                      n=int(ok.sum()),
                                      rho=spearmanr(d[a][ok], d[b][ok])[0]))
            if 'mort_frac' in d and not tag:
                fig_trajectories(d, outputdir /
                                 f'trajectories_{aoi}_{scale_m}m.png',
                                 f'{aoi} {scale_m} m')

            # Part 2: trait change vs multispectral change
            targets = ['ndmi_resilience', 'ndmi_recovery',
                       'nirv_resilience', 'nirv_recovery']
            if 'mort_frac' in d:
                targets = ['mort_frac'] + targets
            base = ENV + S_WALL
            for a, b in INTERVALS:
                iv = f'd{b % 100}{a % 100}'
                if intervals and iv not in intervals:
                    continue
                L = [f'L_{v}_{iv}' for v in L_VARS]
                A = [f'A_{t}_{iv}' for t in TRAITS + ['ewt', 'qcfc']]
                Araw = [f'Araw_{t}_{iv}' for t in TRAITS + ['ewt']]
                fs = {'L': L, 'A': A, 'Araw': Araw, 'L+A': L + A,
                      'EnvS': base, 'EnvS+L': base + L,
                      'EnvS+A': base + A, 'EnvS+L+A': base + L + A}
                cmp = [('L', 'A'), ('Araw', 'A'), ('L', 'L+A'),
                       ('EnvS', 'EnvS+L'), ('EnvS', 'EnvS+A'),
                       ('EnvS+L', 'EnvS+L+A')]
                for t in targets:
                    sub = d[d[t].notna() & np.isfinite(d[t])]
                    w = 'mort_n' if t == 'mort_frac' else None
                    rows, _ = rc.ladder(sub, t, fs, 'block1000', weight=w,
                                        n_boot=n_boot, pairs=cmp,
                                        extra_blocks=('block5000',))
                    for r in rows:
                        r.update(aoi=aoi, scale_m=scale_m, interval=iv,
                                 fold_seed=fold_seed)
                    ladder_rows += rows
                    msg = '  '.join(
                        f'{r["features"]}{"-" + r["compare"] if r["compare"] else ""} '
                        f'{r["r2"]:+.3f}' for r in rows
                        if r['boot_blocks'] == 'block1000')
                    click.echo(f'  {iv} {t}: {msg}')
                # Recovery failure: bottom quintile of NDMI resilience
                sub = d[d.ndmi_resilience.notna()].copy()
                sub['fail'] = (sub.ndmi_resilience <=
                               sub.ndmi_resilience.quantile(0.2)).astype(int)
                for name, cols in fs.items():
                    auc, _ = auc_oof(sub, cols, 'fail', 'block1000')
                    auc_rows.append(dict(aoi=aoi, scale_m=scale_m,
                                         interval=iv, fold_seed=fold_seed,
                                         features=name,
                                         target='ndmi_resilience_q20',
                                         auc=auc))
            if not tag:
                pd.DataFrame(corr_rows).to_csv(
                    outputdir / 'ewt_vs_landsat.csv', index=False)
            pd.DataFrame(ladder_rows).to_csv(
                outputdir / f'dynamics_ladder{tag}.csv', index=False)
            pd.DataFrame(auc_rows).to_csv(
                outputdir / f'dynamics_auc{tag}.csv', index=False)


if __name__ == '__main__':
    main()
