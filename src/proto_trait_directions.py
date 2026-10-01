#!/usr/bin/env python
"""
Why do trait-recovery directions differ between the Yosemite box (NEON,
sierra_nf) and the Tahoe box (stanislaus)?

In the Yosemite box, residual nitrogen goes with better NDMI recovery and
residual LMA with worse (within-stratum Spearman ρ about +0.19 and -0.17);
in the Tahoe box both are about 0, while the structural-carbon traits
(cellulose, lignin, fiber: worse recovery) agree in both boxes. Three
explanations are tested on the same cells and models as
proto_response_traits.py:

  calibration  the 2013 Tahoe mosaic is not cross-year calibrated (no _v2
               exists for that box). Variants:
                 nov2      Yosemite box with the non-_v2 2013-06-12 mosaic
                           (fetch_wdts_traits.py --date
                           yosemite:2013=20130612 --suffix nov2)
                 year2014  traits from the 2014 acquisition (non-_v2 in
                           both boxes)
                 year2015  traits from 2015 (_v2 in the Yosemite box, not
                           in the Tahoe box; mid-drought, direction only)
                 line_z    each trait z-scored within each flight line
               A single per-date z-score is a monotone transform of one
               mosaic and cannot change these results.
  forest type  LANDFIRE 2014 EVT groups (fetch_forest_type.py): directions
               within the dominant group of each cell (>= 50% of its
               pixels), and a ladder Env+S | Env+S+F | Env+S+T | Env+S+F+T
               with F the group fractions, plus the trait gain within each
               group.
  signal       the Tahoe drought signal is weaker (resistance SD / placebo
               SD 1.6 against 1.9-2.0). In the Yosemite box: (a) iid noise
               added to the target so that its signal share matches the
               Tahoe box (20 draws); (b) the cells with the lowest drought
               forcing (cwd_anom_1416), in subsets whose SD ratio
               approaches the Tahoe one.

Residual traits are cross-fitted on Env+S over all cells (as in
proto_response_traits.py), and ρ is averaged within aridity-tercile x
200 m elevation strata. The 95% CIs come from a 1 km block bootstrap, with
ranks fixed within strata (weighted Pearson on within-stratum ranks).

Outputs in OUTPUTDIR: directions.csv, ladder_ftype.csv, signal.csv and
directions_<target>.png.

    python proto_trait_directions.py $E/hls_results/trait_directions \
        -a neon_soap_teak -a sierra_nf -a stanislaus
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
from scipy.stats import rankdata

import response_common as rc
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import (TRAITS, ELEV_BAND, build, trait_paths,
                                   trait_year)
from fetch_forest_type import FTYPES

TARGETS = ['ndmi_recovery', 'nirv_recovery', 'ndmi_resistance']
FOCUS = ['Nitrogen', 'LMA', 'Lignin', 'Cellulose']
YOSEMITE = ('neon_soap_teak', 'sierra_nf')
VARIANTS = ['baseline', 'nov2', 'year2014', 'year2015', 'line_z']
MIN_GROUP = 1000  # cells for a forest-type subset
N_NOISE = 20
FORCING_Q = (1 / 3, 1 / 2, 2 / 3)


def variant_args(aoi, v):
    """(paths, y0, per_line) for a variant, or None if not applicable"""
    if v == 'nov2':
        if aoi not in YOSEMITE:
            return None
        f = rc.E / 'wdts' / f'{aoi}_traits_nov2.nc'
        return (trait_paths(aoi, f), 2013, False) if f.exists() else None
    y0 = {'year2014': 2014, 'year2015': 2015}.get(v, 2013)
    return trait_paths(aoi), y0, v == 'line_z'


def forest_type(aoi, k):
    f = rc.E / 'env' / f'{aoi}_ftype.nc'
    if not f.exists():
        return None
    ft = xr.open_dataset(f).ftype.values
    env = rc.open_env(aoi)
    layers = {f'F_{g}': (ft == i).astype(np.float32)
              for i, g in enumerate(FTYPES)}
    t = rc.cell_table(layers, rc.undisturbed(env, 2019), k)
    F = [f'F_{g}' for g in FTYPES]
    top = t[F].values.argmax(1)
    t['ftype'] = np.where(t[F].values.max(1) >= 0.5,
                          np.array(FTYPES)[top], 'mixed')
    return t[['cell_row', 'cell_col', 'ftype'] + F]


def residual_traits(d):
    y0 = trait_year(d)
    cols = [f'T_{t}_{y0}' for t in TRAITS]
    R = {f'R_{t}': rc.crossfit_residuals(d, c, ENV + S_WALL, 'block1000')
         for t, c in zip(TRAITS, cols)}
    return pd.DataFrame(R, index=d.index)


def strata_of(d):
    return (d.arid.astype(str) + '_' +
            (d.elevation // ELEV_BAND).astype(int).astype(str))


def rho_within_boot(d, col, target, strata, blocks, n_boot, seed=0,
                    min_n=100):
    """Bootstrap percentiles of the size-weighted within-stratum ρ.

    Ranks are computed once within each stratum; each draw resamples 1 km
    blocks and takes the multiplicity-weighted Pearson of the ranks."""
    ok = (d[col].notna() & d[target].notna()).values
    x, y = d[col].values[ok], d[target].values[ok]
    s, b = strata.values[ok], blocks.values[ok]
    s_codes, s_inv = np.unique(s, return_inverse=True)
    size = np.bincount(s_inv)
    keep = size[s_inv] >= min_n
    x, y, s_inv, b = x[keep], y[keep], s_inv[keep], b[keep]
    if len(x) < 200:
        return np.nan, np.nan
    s_codes, s_inv = np.unique(s_inv, return_inverse=True)
    rx, ry = np.empty(len(x)), np.empty(len(y))
    for j in range(len(s_codes)):
        m = s_inv == j
        rx[m], ry[m] = rankdata(x[m]), rankdata(y[m])
    b_codes, b_inv = np.unique(b, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(b_codes), size=(n_boot, len(b_codes)))
    S = sparse.csr_matrix((np.ones(len(x)), (np.arange(len(x)), s_inv)))
    St = S.T.tocsr()
    rho = []
    for i in range(0, n_boot, 100):  # chunks of draws bound the memory
        W = np.stack([np.bincount(dr, minlength=len(b_codes))
                      for dr in draws[i:i + 100]]).astype(float)[:, b_inv]
        agg = lambda v: (St @ (W * v).T).T
        sw, sx, sy = agg(1.0), agg(rx), agg(ry)
        sxx, syy, sxy = agg(rx * rx), agg(ry * ry), agg(rx * ry)
        with np.errstate(invalid='ignore', divide='ignore'):
            cov = sxy - sx * sy / sw
            vx, vy = sxx - sx ** 2 / sw, syy - sy ** 2 / sw
            r = cov / np.sqrt(vx * vy)
        r = np.where(sw >= min_n / 2, r, np.nan)
        wts = np.where(np.isfinite(r), sw, 0)
        rho.append(np.nansum(np.nan_to_num(r) * wts, 1) / wts.sum(1))
    return tuple(np.percentile(np.concatenate(rho), [2.5, 97.5]))


def directions(d, cols, targets, n_boot, **tags):
    rows = []
    strata = strata_of(d)
    for t in targets:
        if t not in d or d[t].notna().sum() < 500:
            continue
        uni = {u['feature']: u for u in rc.within_strata_rho(
            d, cols, t, strata)}
        for c in cols:
            if c not in uni:
                continue
            lo, hi = (rho_within_boot(d, c, t, strata, d.block1000, n_boot)
                      if n_boot else (np.nan, np.nan))
            u = uni[c]
            rows.append(dict(**tags, target=t, feature=c, n=u['n'],
                             n_strata=u['n_strata'], rho=u['rho'],
                             rho_within=u['rho_within'], lo=lo, hi=hi))
    return rows


def signal_ratio(d, index='ndmi'):
    return d[f'{index}_resistance'].std() / d[f'{index}_placebo'].std()


def signal_checks(d, R, target_ratio, n_boot, **tags):
    """Yosemite-box directions with the drought signal degraded to the
    Tahoe level: (a) iid noise; (b) low-forcing subsets"""
    rows, sig = [], []
    cols = [f'R_{t}' for t in FOCUS]
    d = pd.concat([d, R], axis=1)
    r_y = signal_ratio(d)
    s_y, s_t = 1 - 1 / r_y ** 2, 1 - 1 / target_ratio ** 2
    rng = np.random.default_rng(0)
    for t in ('ndmi_recovery', 'nirv_recovery'):
        v = d[t].var()
        sd = np.sqrt(max(v * (s_y / s_t - 1), 0))
        draws = []
        for i in range(N_NOISE):
            dn = d.copy()
            dn[t] = d[t] + rng.normal(0, sd, len(d))
            draws += directions(dn, cols, [t], n_boot if i == 0 else 0,
                                **tags, subset='noise', draw=i)
        g = pd.DataFrame(draws)
        first = g[g.draw == 0].set_index('feature')
        for c, gg in g.groupby('feature'):
            rows.append(dict(**tags, subset='noise', target=t, feature=c,
                             n=int(gg.n.iloc[0]),
                             n_strata=int(gg.n_strata.iloc[0]),
                             rho=gg.rho.mean(),
                             rho_within=gg.rho_within.mean(),
                             lo=first.loc[c, 'lo'], hi=first.loc[c, 'hi'],
                             draws_lo=gg.rho_within.quantile(0.025),
                             draws_hi=gg.rho_within.quantile(0.975)))
        sig.append(dict(**tags, subset='noise', target=t, sd_ratio=r_y,
                        signal_share=s_y, target_share=s_t, noise_sd=sd,
                        target_sd=np.sqrt(v), n=len(d)))
    for q in FORCING_Q:
        sub = d[d.cwd_anom_1416 <= d.cwd_anom_1416.quantile(q)]
        name = f'low_forcing_{q:.2f}'
        rows += directions(sub, cols, ['ndmi_recovery', 'nirv_recovery'],
                           n_boot, **tags, subset=name)
        sig.append(dict(**tags, subset=name, sd_ratio=signal_ratio(sub),
                        n=len(sub)))
    return rows, sig


def ftype_ladder(d, n_boot, aoi, scale_m):
    rows = []
    y0 = trait_year(d)
    T = [f'T_{t}_{y0}' for t in TRAITS] + [f'T_qcfc_{y0}']
    F = [f'F_{g}' for g in FTYPES]
    es = ENV + S_WALL
    for t in ('ndmi_recovery', 'ndmi_resistance'):
        sub = d[d[t].notna() & np.isfinite(d[t])]
        fs = {'Env+S': es, 'Env+S+F': es + F, 'Env+S+T': es + T,
              'Env+S+F+T': es + F + T}
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                         pairs=[('Env+S', 'Env+S+F'), ('Env+S', 'Env+S+T'),
                                ('Env+S+F', 'Env+S+F+T')])
        rows += [dict(x, aoi=aoi, scale_m=scale_m, group='all') for x in r]
        for g in FTYPES + ['mixed']:
            s = sub[sub.ftype == g]
            if len(s) < MIN_GROUP:
                continue
            r, _ = rc.ladder(s, t, {'Env+S': es, 'Env+S+T': es + T},
                             'block1000', n_boot=n_boot)
            rows += [dict(x, aoi=aoi, scale_m=scale_m, group=g) for x in r]
        g = {x['group']: x for x in rows if x['target'] == t
             and x['compare'] == 'Env+S' and x['features'] == 'Env+S+T'
             and x['aoi'] == aoi}
        click.echo(f'  ladder {t}: ' + '  '.join(
            f'{k} {v["r2"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}] '
            f'n={v["n"]}' for k, v in g.items()))
    return rows


def save(rows, path):
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--variant', 'variants', multiple=True, default=VARIANTS,
              show_default=True, type=click.Choice(VARIANTS))
@click.option('--response-dir', 'response_dirs', multiple=True,
              default=['response', 'response_sierra', 'response_stanislaus'],
              show_default=True,
              help='hls_results subdirs searched for metrics_<aoi>_*.csv')
@click.option('--dynamics-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/trait_dynamics')
@click.option('--scale', default=3, show_default=True)
@click.option('--n-boot', default=500, show_default=True)
@click.option('--skip-ladder', is_flag=True)
def main(outputdir, aois, variants, response_dirs, dynamics_dir, scale,
         n_boot, skip_ladder):
    outputdir.mkdir(parents=True, exist_ok=True)
    scale_m = rc.RES * scale
    rows, lad, sig = [], [], []
    ratios = {}
    for aoi in aois:
        rdir = next(rc.E / 'hls_results' / r for r in response_dirs
                    if (rc.E / 'hls_results' / r /
                        f'metrics_{aoi}_{scale_m}m.csv').exists())
        ft = forest_type(aoi, scale)
        for v in variants:
            args = variant_args(aoi, v)
            if args is None:
                continue
            paths, y0, per_line = args
            d = build(aoi, scale, rdir, dynamics_dir, paths, y0, per_line)
            if ft is not None:
                d = d.merge(ft, on=['cell_row', 'cell_col'], how='left')
                d['ftype'] = d.ftype.fillna('mixed')
            ratios[aoi] = signal_ratio(d)
            click.echo(f'[{aoi} {v}] {len(d)} cells, resistance SD / '
                       f'placebo SD {ratios[aoi]:.2f}')
            R = residual_traits(d)
            dd = pd.concat([d, R], axis=1)
            cols = list(R.columns)
            tags = dict(aoi=aoi, scale_m=scale_m, variant=v)
            rows += directions(dd, cols, TARGETS, n_boot, **tags,
                               subset='all')
            if ft is not None:
                for g in FTYPES + ['mixed']:
                    s = dd[dd.ftype == g]
                    if len(s) >= MIN_GROUP:
                        rows += directions(
                            s, [f'R_{t}' for t in FOCUS], TARGETS[:2],
                            n_boot, **tags, subset=f'ftype_{g}')
            if v == 'baseline' and ft is not None and not skip_ladder:
                lad += ftype_ladder(d, n_boot, aoi, scale_m)
                save(lad, outputdir / 'ladder_ftype.csv')
            if v == 'baseline' and aoi in YOSEMITE and \
                    'stanislaus' in ratios:
                r, s = signal_checks(d, R, ratios['stanislaus'], n_boot,
                                     **tags)
                rows += r
                sig += s
                save(sig, outputdir / 'signal.csv')
            res = pd.DataFrame(rows)
            res = res[(res.aoi == aoi) & (res.variant == v) &
                      (res.target == 'ndmi_recovery')]
            for sub, g in res.groupby('subset', sort=False):
                g = g.set_index('feature')
                click.echo(f'  {sub:22s} n={int(g.n.max()):6d}  ' + '  '.join(
                    f'{t} {g.loc[f"R_{t}", "rho_within"]:+.3f}'
                    f'[{g.loc[f"R_{t}", "lo"]:+.2f},'
                    f'{g.loc[f"R_{t}", "hi"]:+.2f}]'
                    for t in FOCUS if f'R_{t}' in g.index))
            save(rows, outputdir / 'directions.csv')
    for t in TARGETS[:2]:
        fig_directions(pd.DataFrame(rows), t,
                       outputdir / f'directions_{t}.png')


def fig_directions(r, target, path):
    r = r[(r.target == target) & r.feature.isin([f'R_{t}' for t in FOCUS])]
    r = r.assign(row=r.aoi + ' | ' + r.variant + ' | ' + r.subset)
    rows = list(dict.fromkeys(r.row))
    fig, axes = plt.subplots(1, len(FOCUS), sharey=True,
                             figsize=(3.2 * len(FOCUS),
                                      0.22 * len(rows) + 1.5))
    for ax, t in zip(axes, FOCUS):
        g = r[r.feature == f'R_{t}'].set_index('row').reindex(rows)
        y = np.arange(len(rows))
        ax.errorbar(g.rho_within, y, xerr=[g.rho_within - g.lo,
                                           g.hi - g.rho_within],
                    fmt='o', ms=3, lw=0.8, color='0.2')
        ax.axvline(0, color='k', lw=0.6)
        ax.set_title(f'residual {t}', fontsize=9)
        ax.set_xlabel('within-stratum ρ (95% CI)', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize=6)
    axes[0].invert_yaxis()
    fig.suptitle(f'Residual trait vs {target}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
