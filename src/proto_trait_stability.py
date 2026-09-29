#!/usr/bin/env python
"""
How stable are the slow AVIRIS-C foliar traits between June flights?

Models of drought response trained on 2013 traits can only be applied to a
later drought with traits from a later flight (e.g. June 2018, before the
2020-22 drought) if slow leaf-economics traits (N, LMA, lignin) keep their
spatial pattern between dates where the canopy did not change. This script
measures that, per cell, for each year against 2013.

- Traits: WDTS 30 m mosaics (wdts/<aoi>_traits.nc), cross-track normalized
  per flight line as in proto_trait_dynamics.py; raw values are also
  reported. 2013 and 2015 are the cross-year-calibrated "_v2" mosaics.
- Cells: undisturbed NLCD forest (no fire or harvest through 2019). "Stable"
  cells are also in the top tercile of Landsat NDMI resistance (least
  drought loss, proto_response_metrics.py) and, at NEON, have no lidar tree
  mortality (< 5% of 2013-live trees dead by 2017-18, when >= 5 trees).
- Per pair: Pearson r, Spearman ρ, mean bias (later - 2013) in trait units
  and as a fraction of the 2013 spatial SD, and RMSE.
- Sign check: within aridity x elevation strata, Spearman ρ of each year's
  trait with cycle-1 NDMI recovery. If 2018 traits carry the same recovery
  directions as 2013 traits, a 2013-trained model is transferable in sign.

    python proto_trait_stability.py $E/hls_results/trait_stability \\
        -a neon_soap_teak -a sierra_nf
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import response_common as rc
from proto_trait_dynamics import crosstrack_normalize, AV_YEARS

TRAITS = ['Nitrogen', 'LMA', 'Lignin', 'Chlorophylls']
BASE = 2013
RESPONSE_DIRS = {'neon_soap_teak': 'response', 'sierra_nf': 'response_sierra'}
DYNAMICS_DIRS = {'neon_soap_teak': 'trait_dynamics',
                 'sierra_nf': 'trait_dynamics_sierra'}
MAX_DEAD = 0.05
ELEV_BAND = 200


def build(aoi, k):
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    tr = xr.open_dataset(rc.E / 'wdts' / f'{aoi}_traits.nc')
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
    d = rc.cell_table(layers, valid, k)
    res = rc.E / 'hls_results' / RESPONSE_DIRS[aoi]
    m = pd.read_csv(res / f'metrics_{aoi}_{rc.RES * k}m.csv',
                    usecols=lambda c: c in ('cell_row', 'cell_col',
                                            'ndmi_resistance',
                                            'ndmi_recovery', 'cwd_clim',
                                            'elevation'))
    d = d.merge(m, on=['cell_row', 'cell_col'], how='inner')
    dyn = (rc.E / 'hls_results' / DYNAMICS_DIRS[aoi] /
           f'dynamics_{aoi}_{rc.RES * k}m.csv')
    if dyn.exists():
        dd = pd.read_csv(dyn, usecols=lambda c: c in (
            'cell_row', 'cell_col', 'mort_frac'))
        d = d.merge(dd, on=['cell_row', 'cell_col'], how='left')
    top = d.ndmi_resistance >= d.ndmi_resistance.quantile(2 / 3)
    no_mort = d.mort_frac.fillna(0) < MAX_DEAD if 'mort_frac' in d else True
    d['stable'] = top & no_mort
    d['arid'] = pd.qcut(d.cwd_clim, 3, labels=['wet', 'mid', 'dry'])
    return d


def pair_stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 100:
        return dict(n=len(a))
    return dict(n=len(a), r=pearsonr(a, b)[0], rho=spearmanr(a, b)[0],
                bias=float(np.mean(b - a)),
                bias_sd=float(np.mean(b - a) / np.std(a)),
                rmse=float(np.sqrt(np.mean((b - a) ** 2))),
                sd_base=float(np.std(a)), sd_later=float(np.std(b)))


def fig_scatter(d, aoi, scale_m, path):
    s = d[d.stable]
    fig, axes = plt.subplots(1, len(TRAITS), figsize=(4 * len(TRAITS), 4))
    for ax, t in zip(axes, TRAITS):
        a, b = s[f'{t}_{BASE}'], s[f'{t}_2018']
        ok = a.notna() & b.notna()
        ax.hexbin(a[ok], b[ok], gridsize=60, mincnt=1, cmap='viridis',
                  bins='log')
        lo, hi = np.nanpercentile(np.r_[a[ok], b[ok]], [1, 99])
        ax.plot([lo, hi], [lo, hi], 'w--', lw=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f'{t} {BASE}',
               ylabel=f'{t} 2018',
               title=f'ρ = {spearmanr(a[ok], b[ok])[0]:.2f} (n={ok.sum()})')
    fig.suptitle(f'{aoi} {scale_m} m, stable cells, cross-track normalized')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', 'scales', multiple=True, type=int, default=[3, 9],
              show_default=True)
def main(outputdir, aois, scales):
    outputdir.mkdir(parents=True, exist_ok=True)
    rows, sign_rows = [], []
    for aoi in aois:
        for k in scales:
            scale_m = rc.RES * k
            d = build(aoi, k)
            click.echo(f'[{aoi} {scale_m} m] {len(d)} cells, '
                       f'{int(d.stable.sum())} stable')
            for group, g in (('all', d), ('stable', d[d.stable])):
                for t in TRAITS:
                    for norm in ('', 'raw_'):
                        a = g[f'{t}_{norm}{BASE}'].values
                        for y in AV_YEARS:
                            if y == BASE:
                                continue
                            s = pair_stats(a, g[f'{t}_{norm}{y}'].values)
                            rows.append(dict(aoi=aoi, scale_m=scale_m,
                                             cells=group, trait=t,
                                             normalized=norm == '',
                                             base=BASE, year=y, **s))
            strata = (d.arid.astype(str) + '_' +
                      (d.elevation // ELEV_BAND).astype(int).astype(str))
            cols = [f'{t}_{y}' for t in TRAITS for y in AV_YEARS]
            for u in rc.within_strata_rho(d, cols, 'ndmi_recovery', strata):
                t, y = u['feature'].rsplit('_', 1)
                sign_rows.append(dict(aoi=aoi, scale_m=scale_m, trait=t,
                                      year=int(y), **u))
            s = pd.DataFrame(rows)
            s = s[(s.aoi == aoi) & (s.scale_m == scale_m) & s.normalized]
            for group in ('all', 'stable'):
                x = s[s.cells == group].pivot(index='trait', columns='year',
                                              values='rho')
                click.echo(f'  ρ vs {BASE}, {group} cells:\n'
                           + x.round(2).to_string())
            sg = pd.DataFrame(sign_rows)
            sg = sg[(sg.aoi == aoi) & (sg.scale_m == scale_m)]
            click.echo('  within-strata ρ with NDMI recovery:\n' + sg.pivot(
                index='trait', columns='year',
                values='rho_within').round(2).to_string())
            fig_scatter(d, aoi, scale_m,
                        outputdir / f'stability_{aoi}_{scale_m}m.png')
            pd.DataFrame(rows).to_csv(outputdir / 'trait_stability.csv',
                                      index=False)
            pd.DataFrame(sign_rows).to_csv(
                outputdir / 'trait_stability_recovery_sign.csv', index=False)


if __name__ == '__main__':
    main()
