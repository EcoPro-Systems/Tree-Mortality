#!/usr/bin/env python
"""
Do environment-only drought-response models transfer from one drought to
the next?

Two drought cycles, each with Landsat summer (Jul-Sep) response metrics
(proto_response_metrics.metrics) on the same cells:
    cycle 1  baseline 2008-11, drought 2014-16, post 2017-19 (Landsat 7)
    cycle 2  baseline 2017-19, drought 2020-22, post 2023-25 (Landsat 8/9)
Each sensor record is single-family, so sensor offsets largely cancel in
the NIRv ratios and NDMI differences.

Features use cycle-generic names so a model trained on one cycle can be
applied to the other: climate normals, drought-window CWD/SPEI4/Tmax/PPT
anomalies (BCMv8), terrain, pre-2012 structure (Env+S), and baseline
greenness (B1).

Reported: within-cycle 1 km block CV, cycle 1 -> 2 and 2 -> 1 transfer
(absolute R² and Spearman ρ, block-bootstrap CIs on the test cycle).
Cells are forest with no fire (MTBS, FRAP, prescribed) 2000-2025 and no
FACTS harvest 2005-2025, which removes the 2020 Creek Fire.

    python proto_response_transfer.py $E/hls_results/response_transfer \
        -a neon_soap_teak
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import spearmanr

import response_common as rc
from proto_response_metrics import metrics, mean_years, TERRAIN, S_WALL

CYCLES = {
    1: dict(base=(2008, 2011), drought=(2014, 2016), post=(2017, 2019),
            cum=(2012, 2016), variant='c2l7'),
    2: dict(base=(2017, 2019), drought=(2020, 2022), post=(2023, 2025),
            cum=(2020, 2022), variant='c2oli'),
}
ENV = ['cwd_clim', 'ppt_clim', 'tmx_clim', 'cwd_anom_cum', 'cwd_anom_dr',
       'spei4_min_dr', 'tmx_anom_dr', 'ppt_anom_dr'] + TERRAIN
B1 = ['ndvi_base', 'nirv_base']
TARGETS = [f'{v}_{m}' for v in ('ndmi', 'nirv')
           for m in ('resistance', 'recovery', 'resilience')]
INDICES = ['ndmi', 'nirv', 'ndvi']


def build_cycle(aoi, k, c, valid, allow_held_out=False):
    if c == 2:
        rc.check_cycle2(aoi, allow_held_out)
    cy = CYCLES[c]
    env = rc.open_env(aoi)
    comp = xr.open_dataset(rc.E / 'landsat_composites' /
                           f'{aoi}_{cy["variant"]}_doy182-273.nc')
    years = list(range(cy['base'][0], cy['post'][1] + 1))
    stack = {v: comp[v].sel(year=years).values for v in INDICES}
    ok = valid.copy()
    for a in stack.values():
        ok &= np.isfinite(a).all(0)
    sel = lambda v, span: env[v].sel(year=slice(*span)).values
    clim = env.cwd_clim.values
    layers = {v: env[v].values for v in ['cwd_clim', 'ppt_clim', 'tmx_clim']
              + TERRAIN + S_WALL}
    layers['cwd_anom_cum'] = (sel('cwd', cy['cum']) - clim).sum(0)
    layers['cwd_anom_dr'] = (sel('cwd', cy['drought']) - clim).mean(0)
    layers['spei4_min_dr'] = sel('spei4', cy['drought']).min(0)
    layers['tmx_anom_dr'] = (sel('tmx', cy['drought']).mean(0)
                             - env.tmx_clim.values)
    layers['ppt_anom_dr'] = (sel('ppt', cy['drought']).mean(0)
                             / env.ppt_clim.values)
    for v, a in stack.items():
        for i, y in enumerate(years):
            layers[f'{v}_{y}'] = a[i]
    d = rc.cell_table(layers, ok, k)
    series = {v: np.stack([d[f'{v}_{y}'].values for y in years])
              for v in INDICES}
    spei = np.zeros((len(years), len(d)))  # stress response not used here
    m = metrics({v: series[v] for v in ('ndmi', 'nirv')}, years, spei,
                cy['base'], cy['drought'], cy['post'],
                ((cy['base'][0], cy['base'][0]),
                 (cy['base'][1], cy['base'][1])))
    m['ndvi_base'] = mean_years(series['ndvi'], years, cy['base'])
    d = pd.concat([d.drop(columns=[c for c in d.columns if c.startswith(
        tuple(f'{v}_' for v in INDICES))]), pd.DataFrame(m, index=d.index)],
        axis=1)
    if c == 2 and aoi in rc.HELD_OUT:
        # a held-out site is tested on its fixed test cells only (§39)
        from proto_heldout_strata import test_cells
        d = d.merge(test_cells(aoi, k), on=['cell_row', 'cell_col'])
    return d.assign(cycle=c)


def skill(y, p, blocks, n_boot, seed=0):
    b = rc.bootstrap_r2(y, {'p': p}, blocks, n_boot=n_boot, seed=seed)['p']
    return dict(r2=b[0], lo=b[1], hi=b[2], rho=spearmanr(y, p)[0],
                bias=float(np.mean(p - y)))


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', 'scales', multiple=True, type=int, default=[3, 9],
              show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
def main(outputdir, aois, scales, n_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    rows = []
    fsets = {'B1': B1, 'Env': ENV, 'Env+S': ENV + S_WALL,
             'Env+S+B1': ENV + S_WALL + B1}
    for aoi in aois:
        env = rc.open_env(aoi)
        valid = rc.undisturbed(env, 2025)
        for k in scales:
            scale_m = rc.RES * k
            cyc = {c: build_cycle(aoi, k, c, valid) for c in CYCLES}
            key = ['cell_row', 'cell_col']
            common = cyc[1][key].merge(cyc[2][key], on=key)
            for c in cyc:
                cyc[c] = cyc[c].merge(common, on=key).reset_index(drop=True)
                cyc[c].to_csv(outputdir /
                              f'cycle{c}_{aoi}_{scale_m}m.csv', index=False)
            click.echo(f'[{aoi} {scale_m} m] {len(common)} cells in both '
                       'cycles')
            for t in TARGETS:
                for name, cols in fsets.items():
                    for c in CYCLES:
                        d = cyc[c][cyc[c][t].notna()]
                        p = rc.oof_predict(d, cols, t, 'block1000')
                        rows.append(dict(aoi=aoi, scale_m=scale_m, target=t,
                                         features=name, train=c, test=c,
                                         mode='within_cv', n=len(d),
                                         **skill(d[t].values, p,
                                                 d.block1000.values,
                                                 n_boot)))
                    for tr_c, te_c in ((1, 2), (2, 1)):
                        a = cyc[tr_c][cyc[tr_c][t].notna()]
                        b = cyc[te_c][cyc[te_c][t].notna()]
                        m = rc.hgb().fit(a[cols], a[t])
                        p = m.predict(b[cols])
                        rows.append(dict(aoi=aoi, scale_m=scale_m, target=t,
                                         features=name, train=tr_c,
                                         test=te_c, mode='transfer',
                                         n=len(b), **skill(
                                             b[t].values, p,
                                             b.block1000.values, n_boot)))
                    r = [x for x in rows[-4:]]
                    click.echo(
                        f'  {t:16s} {name:9s} within C1 {r[0]["r2"]:.3f} '
                        f'C2 {r[1]["r2"]:.3f} | C1->C2 R2 {r[2]["r2"]:.3f} '
                        f'rho {r[2]["rho"]:.3f} | C2->C1 R2 '
                        f'{r[3]["r2"]:.3f} rho {r[3]["rho"]:.3f}')
                # Does cycle-1 response rank cells for cycle 2?
                x, y = cyc[1][t], cyc[2][t]
                ok = x.notna() & y.notna()
                rows.append(dict(aoi=aoi, scale_m=scale_m, target=t,
                                 features='cycle1_same_metric', train=1,
                                 test=2, mode='persistence', n=int(ok.sum()),
                                 rho=spearmanr(x[ok], y[ok])[0]))
                pd.DataFrame(rows).to_csv(outputdir / 'transfer.csv',
                                          index=False)


if __name__ == '__main__':
    main()
