#!/usr/bin/env python
"""
Drought response metrics from Landsat and how much of their variance forcing
and stand structure leave unexplained ("same stress, different
trajectories").

Per cell (30, 90 and 270 m), from summer (Jul-Sep) Landsat composites of
NDMI and NIRv:
- resistance  drought (2014-16) vs pre-drought baseline (2008-11);
              resistance_late uses the last two drought years (2015-16)
- recovery    post (2017-19) vs drought
- resilience  post vs baseline
- rectime     years after 2016 until the index is back within 1 SD of the
              baseline (0 = never left it, 4 = not back by 2019, censored)
- sens        stress response: per-cell OLS slope of the index on SPEI4,
              2008-2019
NIRv metrics are ratios; NDMI, which can be near zero, uses differences.

Annual indices are averaged to the cell first, then the metrics are taken,
so a coarse cell's metric is not a mean of noisy pixel ratios.

The noise floor is a placebo metric: the resistance formula applied within
the baseline, mean(2010-11) vs mean(2008-09), where no drought occurred.

Analyses:
1. Variance explained by B1 (baseline greenness), Env (climate normals,
   drought anomalies, terrain), S (structure) and Env+S, spatial-block CV
   (1 km folds), block-bootstrap 95% CIs over 1 km and 5 km blocks.
2. Within-bin spread: cells binned by cumulative 2012-16 CWD anomaly
   (quintiles) x 200 m elevation band x structure tercile.
3. Spatial structure of the Env+S residuals (semivariogram).
4. Figures: within-bin distributions, residual map, and matched-forcing
   neighbour pairs with divergent trajectories.

Pixels are forest (NLCD 2013 41/42/43) with no fire (MTBS, CAL FIRE FRAP
or prescribed burn) 2000-2019 and no FACTS harvest 2005-2019
(--keep-salvage keeps cells whose only harvest was a salvage or sanitation
cut).

    python proto_response_metrics.py $E/hls_results/response -a neon_soap_teak
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import response_common as rc

INDICES = ['ndmi', 'nirv']
BASE, DROUGHT, POST = (2008, 2011), (2014, 2016), (2017, 2019)
PLACEBO = ((2008, 2009), (2010, 2011))
YEARS = list(range(BASE[0], POST[1] + 1))
METRICS = ['resistance', 'recovery', 'resilience', 'rectime', 'sens']
TERRAIN = ['elevation', 'slope', 'northness', 'eastness', 'tpi', 'vrm',
           'rie', 'sapa', 'sdmv', 'adjsd']
ENV = ['cwd_clim', 'ppt_clim', 'tmx_clim', 'cwd_anom_1216', 'cwd_anom_1416',
       'spei4_min_1416', 'tmx_anom_1416', 'ppt_anom_1416'] + TERRAIN
S_WALL = ['glad_h2010', 'tcc2010', 'lf14_evh', 'lf14_evc']
S_LIDAR = ['lidar_n_trees', 'lidar_h_mean', 'lidar_h_p90', 'lidar_h_max',
           'lidar_frac_tall', 'lidar_ca_mean', 'lidar_dead2013']
B1 = ['ndvi_base', 'nirv_base']
ELEV_BAND = 200
MAX_FIT_CELLS = 200_000
ABRUPT_DROP = 0.25  # one-year NDMI drop marking unmapped disturbance
EARLY_DROP = 0.2  # NDMI loss below baseline by 2014, before the die-off


def mean_years(a, years, span):
    i = [years.index(y) for y in range(span[0], span[1] + 1)]
    with np.errstate(all='ignore'):
        return np.nanmean(a[i], axis=0)


def compare(a, b, ratio):
    with np.errstate(all='ignore'):
        return b / a if ratio else b - a


def metrics(series, years, spei, base_w=BASE, drought_w=DROUGHT,
            post_w=POST, placebo_w=PLACEBO):
    """Response metrics per cell. series: {index: (year, n) cell means}"""
    out = {}
    for v, a in series.items():
        ratio = v == 'nirv'
        base = mean_years(a, years, base_w)
        dro = mean_years(a, years, drought_w)
        post = mean_years(a, years, post_w)
        out[f'{v}_base'] = base
        out[f'{v}_resistance'] = compare(base, dro, ratio)
        out[f'{v}_resistance_late'] = compare(
            base, mean_years(a, years, (drought_w[1] - 1, drought_w[1])),
            ratio)
        out[f'{v}_recovery'] = compare(dro, post, ratio)
        out[f'{v}_resilience'] = compare(base, post, ratio)
        out[f'{v}_placebo'] = compare(mean_years(a, years, placebo_w[0]),
                                      mean_years(a, years, placebo_w[1]),
                                      ratio)
        bi = [years.index(y) for y in range(base_w[0], base_w[1] + 1)]
        thr = base - np.nanstd(a[bi], axis=0, ddof=1)
        di = [years.index(y) for y in range(drought_w[0], drought_w[1] + 1)]
        impacted = np.nanmin(a[di], axis=0) < thr
        post_years = list(range(post_w[0], post_w[1] + 1))
        rt = np.full(a.shape[1], len(post_years) + 1.0)
        for j, y in reversed(list(enumerate(post_years))):
            rt[a[years.index(y)] >= thr] = j + 1
        rt[~impacted] = 0
        rt[~np.isfinite(base)] = np.nan
        out[f'{v}_rectime'] = rt
        anom = a - base
        s = spei - spei.mean(0)
        with np.errstate(all='ignore'):
            out[f'{v}_sens'] = (np.nansum(anom * s, 0) /
                                np.nansum(s * s, 0))
    return out


def env_layers(env, years):
    cy = env.year.values.tolist()
    sel = lambda v, span: env[v].sel(year=slice(*span)).values
    cwd, clim = env.cwd.values, env.cwd_clim.values
    out = {k: env[k].values.astype(np.float32)
           for k in ['cwd_clim', 'ppt_clim', 'tmx_clim'] + TERRAIN + S_WALL}
    out['cwd_anom_1216'] = (sel('cwd', (2012, 2016)) - clim).sum(0)
    out['cwd_anom_1416'] = (sel('cwd', (2014, 2016)) - clim).mean(0)
    out['spei4_min_1416'] = sel('spei4', (2014, 2016)).min(0)
    out['tmx_anom_1416'] = (sel('tmx', (2014, 2016)).mean(0)
                            - env.tmx_clim.values)
    out['ppt_anom_1416'] = (sel('ppt', (2014, 2016)).mean(0)
                            / env.ppt_clim.values)
    for k in S_LIDAR:
        if k in env:
            out[k] = env[k].values
    spei = np.stack([env.spei4.sel(year=y).values for y in years])
    return out, spei


def build_cells(aoi, variant, k, keep_salvage=False):
    env = rc.open_env(aoi)
    comp = xr.open_dataset(rc.E / 'landsat_composites' /
                           f'{aoi}_{variant}_doy182-273.nc')
    years = [y for y in YEARS if y in comp.year.values]
    valid = rc.undisturbed(env, POST[1], keep_salvage=keep_salvage)
    stack = {v: comp[v].sel(year=years).values for v in INDICES + ['ndvi']}
    for v in stack:
        valid &= np.isfinite(stack[v]).all(0)
    layers, spei = env_layers(env, years)
    for v, a in stack.items():
        for i, y in enumerate(years):
            layers[f'{v}_{y}'] = a[i]
    for i, y in enumerate(years):
        layers[f'spei4_{y}'] = spei[i]
    d = rc.cell_table(layers, valid, k)
    series = {v: np.stack([d[f'{v}_{y}'].values for y in years])
              for v in INDICES + ['ndvi']}
    sp = np.stack([d[f'spei4_{y}'].values for y in years])
    m = metrics({v: series[v] for v in INDICES}, years, sp)
    m['ndvi_base'] = mean_years(series['ndvi'], years, BASE)
    d = pd.concat([d, pd.DataFrame(m, index=d.index)], axis=1)
    d['elev_band'] = (d.elevation // ELEV_BAND).astype(int)
    return d, years


def feature_sets(d):
    fs = {'B1': B1, 'Env': ENV, 'S': S_WALL, 'Env+S': ENV + S_WALL,
          'Env+S+B1': ENV + S_WALL + B1}
    return fs


def bins(d, target, struct):
    q = pd.qcut(d.cwd_anom_1216, 5, labels=False, duplicates='drop')
    t = pd.qcut(d[struct], 3, labels=False, duplicates='drop')
    key = q.astype(str) + '_' + d.elev_band.astype(str) + '_' + t.astype(str)
    g = d.groupby(key)[target]
    size = g.transform('size')
    ok = size >= 30
    within = d[target][ok] - g.transform('mean')[ok]
    tot = d[target][ok].var()
    return dict(target=target, n_cells=int(ok.sum()), n_bins=int(key[ok]
                .nunique()), total_sd=np.sqrt(tot),
                within_sd=within.std(),
                within_var_frac=within.var() / tot,
                within_iqr_median=g.quantile(0.75)[g.size() >= 30].sub(
                    g.quantile(0.25)[g.size() >= 30]).median()), key


def fig_bins(d, key, target, placebo, path, title):
    top = key.value_counts()
    top = top[top >= 30].index[:12]
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [d[target][key == b].values for b in top]
    ax.violinplot(data, showmedians=True)
    ps = d[placebo].std()
    for i, b in enumerate(top):
        med = np.median(data[i])
        ax.fill_between([i + 0.6, i + 1.4], med - ps, med + ps,
                        color='grey', alpha=0.25, lw=0)
    ax.set_xticks(range(1, len(top) + 1))
    ax.set_xticklabels(top, rotation=45, fontsize=7)
    ax.set_xlabel('CWD-anomaly quintile _ 200 m elevation band _ '
                  'structure tercile')
    ax.set_ylabel(target)
    ax.set_title(f'{title}: within-bin spread (grey: ±1 SD placebo, '
                 f'no-drought noise)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pairs(d, key, target, resid, years, index, path, title, k):
    """Neighbouring cells (< 1 km) in the same bin with the most divergent
    target, plus the map of the Env+S residual.

    Cells with a one-year NDMI drop > ABRUPT_DROP during the drought, or
    that fall more than EARLY_DROP below baseline by 2014 (before the
    2015-16 die-off peak), are not used as examples. These rare
    stand-replacing changes (0.1-0.2% of NEON cells) are disturbances
    missing from the fire and harvest records, not drought responses."""
    d = d.assign(bin=key, resid=resid)
    nd = np.stack([d[f'ndmi_{y}'] for y in years], 1)
    i0, i1 = years.index(BASE[1]), years.index(DROUGHT[1])
    early = nd[:, years.index(2012):years.index(2014) + 1].min(1)
    abrupt = (((-np.diff(nd, axis=1)[:, i0:i1]).max(1) > ABRUPT_DROP)
              | (early < d.ndmi_base.values - EARLY_DROP))
    best = []
    for b, g in d[~abrupt].groupby('bin'):
        if len(g) < 30:
            continue
        if len(g) > 1500:
            g = g.sample(1500, random_state=0)
        x, y = g.cell_col.values * k, g.cell_row.values * k
        v = g[target].values
        i, j = np.argmin(v), np.argmax(v)
        dist = rc.RES * np.hypot(x[:, None] - x[None], y[:, None] - y[None])
        near = dist < 1000
        diff = np.abs(v[:, None] - v[None])
        diff[~near] = -1
        a, c = np.unravel_index(np.argmax(diff), diff.shape)
        best.append((diff[a, c], g.index[a], g.index[c], b))
    best.sort(reverse=True)
    fig = plt.figure(figsize=(12, 7))
    ax0 = fig.add_subplot(1, 2, 1)
    nr, nc = d.cell_row.max() + 1, d.cell_col.max() + 1
    img = np.full((nr, nc), np.nan)
    img[d.cell_row, d.cell_col] = d.resid
    lim = np.nanpercentile(np.abs(img), 98)
    im = ax0.imshow(img, cmap='RdBu', vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax0, shrink=0.6, label=f'{target} residual (Env+S)')
    ax0.set_title(f'{title}: {target} not explained by Env+S', fontsize=9)
    for n, (_, a, c, b) in enumerate(best[:3]):
        axn = fig.add_subplot(3, 2, 2 * n + 2)
        for idx, col in ((a, 'tab:red'), (c, 'tab:blue')):
            ts = [d.loc[idx, f'{index}_{y}'] for y in years]
            axn.plot(years, ts, '-o', color=col, ms=3,
                     label=f'{target} {d.loc[idx, target]:.2f}')
            ax0.plot(d.loc[idx, 'cell_col'], d.loc[idx, 'cell_row'], 'o',
                     mfc='none', mec='k', ms=8)
        axn.axvspan(2012, 2016, color='orange', alpha=0.1)
        axn.set_title(f'bin {b}', fontsize=8)
        axn.legend(fontsize=7)
        axn.set_ylabel(index)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--variant', default='c2l7', show_default=True,
              help='Landsat composite variant (see fetch_landsat_c2_ee.py)')
@click.option('--scale', 'scales', multiple=True, type=int,
              default=[1, 3, 9], show_default=True,
              help='Cell size in 30 m pixels')
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--keep-salvage', is_flag=True)
def main(outputdir, aois, variant, scales, n_boot, keep_salvage):
    outputdir.mkdir(parents=True, exist_ok=True)
    (outputdir / 'variogram_points.csv').unlink(missing_ok=True)
    targets = [f'{v}_{m}' for v in INDICES for m in METRICS]
    ladder_rows, bin_rows, vario_rows = [], [], []
    for aoi in aois:
        for k in scales:
            scale_m = rc.RES * k
            d, years = build_cells(aoi, variant, k, keep_salvage)
            d.to_csv(outputdir / f'metrics_{aoi}_{scale_m}m.csv',
                     index=False)
            fit = (d.sample(MAX_FIT_CELLS, random_state=0)
                   if len(d) > MAX_FIT_CELLS else d)
            click.echo(f'[{aoi} {scale_m} m] {len(d)} cells '
                       f'(fitting {len(fit)})')
            fs = feature_sets(fit)
            for t in targets:
                rows, preds = rc.ladder(fit, t, fs, 'block1000',
                                        n_boot=n_boot,
                                        pairs=[('Env', 'Env+S'),
                                               ('S', 'Env+S'),
                                               ('Env+S', 'Env+S+B1')],
                                        extra_blocks=('block5000',))
                for r in rows:
                    r.update(aoi=aoi, scale_m=scale_m, variant=variant)
                ladder_rows += rows
                r = [r for r in rows if r['features'] == 'Env+S'
                     and r['compare'] == '' and r['boot_blocks'] ==
                     'block1000'][0]
                ph = t.split('_')[0] + '_placebo'
                click.echo(f'  {t:18s} Env+S R2 {r["r2"]:.3f} '
                           f'[{r["lo"]:.3f}, {r["hi"]:.3f}]  '
                           f'sd {fit[t].std():.4f} (placebo '
                           f'{fit[ph].std():.4f})')
                resid = fit.loc[preds.index, t] - preds['Env+S']
                x = fit.loc[preds.index, 'cell_col'].values * scale_m
                y = fit.loc[preds.index, 'cell_row'].values * scale_m
                h, g, cnt, var = rc.semivariogram(x, y, resid.values,
                                                  max_dist=5000)
                ex = rc.fit_exponential(h, g, var)
                vario_rows.append(dict(aoi=aoi, scale_m=scale_m, target=t,
                                       resid_var=var, **ex,
                                       nugget_frac=ex['nugget'] / var))
                pd.DataFrame(dict(h=h, gamma=g, n=cnt)).assign(
                    aoi=aoi, scale_m=scale_m, target=t).to_csv(
                    outputdir / 'variogram_points.csv', mode='a',
                    header=not (outputdir / 'variogram_points.csv')
                    .exists(), index=False)
                for struct in ('glad_h2010', 'lidar_h_mean'):
                    if struct not in fit or fit[struct].notna().mean() < .5:
                        continue
                    sub = fit[fit[struct].notna()]
                    b, key = bins(sub, t, struct)
                    b.update(aoi=aoi, scale_m=scale_m, struct=struct,
                             placebo_sd=sub[ph].std())
                    bin_rows.append(b)
                    if t in ('ndmi_resistance', 'nirv_resilience'):
                        tag = f'{aoi}_{scale_m}m_{t}_{struct}'
                        fig_bins(sub, key, t, ph,
                                 outputdir / f'bins_{tag}.png',
                                 f'{aoi} {scale_m} m')
                        if struct == 'glad_h2010':
                            r_all = resid.reindex(sub.index)
                            fig_pairs(sub, key, t, r_all, years,
                                      t.split('_')[0],
                                      outputdir / f'pairs_{tag}.png',
                                      f'{aoi} {scale_m} m', k)
            pd.DataFrame(ladder_rows).to_csv(
                outputdir / 'response_variance.csv', index=False)
            pd.DataFrame(bin_rows).to_csv(outputdir / 'response_bins.csv',
                                          index=False)
            pd.DataFrame(vario_rows).to_csv(
                outputdir / 'response_variogram.csv', index=False)


if __name__ == '__main__':
    main()
