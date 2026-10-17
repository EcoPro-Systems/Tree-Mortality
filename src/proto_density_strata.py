#!/usr/bin/env python
"""
Do the response signal and the trait gain hold in dense canopies? (cycle 1,
90 m cells, per AOI)

Optical resilience estimates can be unreliable in dense, high-biomass
canopies (Smith & Boers 2023, Nat. Ecol. Evol.). Cells are split into
terciles of canopy cover, per AOI:
  tcc   NLCD 2013 tree canopy cover (wall-to-wall; all cells)
  aso   ASO 2014-17 cover above 5 m (aso_cover5; cells with ASO)
Per tercile:
  - signal-to-noise: SD of NDMI/NIRv resistance and recovery over the SD of
    the placebo (the same contrast within the 2008-11 baseline; §1)
  - Env+S R² of NDMI and NIRv recovery
  - for NDMI recovery, T and Tres over Env+S, and on ASO terciles Tres over
    Env+S+L (L = the ASO block); each tercile fitted on its own cells and
    folds, with residuals fitted within the tercile
  - within-stratum ρ (aridity tercile x 200 m elevation band) of the
    structural-carbon score (mean of z-scored residual 2013 lignin and
    cellulose) with NDMI recovery, residualized on Env+S and Env+S+L

    python proto_density_strata.py $E/hls_results/density_strata \
        -a neon_soap_teak --learner hgb_tuned
"""
import click
import numpy as np
import pandas as pd
from pathlib import Path

import response_common as rc
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import (ELEV_BAND, S_ASO, build, feature_sets,
                                   structure_cells)
from proto_spaceborne_only import aoi_dirs
from proto_trait_directions import rho_within_boot
from proto_forward_pilot import zmean

LADDER_TARGETS = ('ndmi_recovery', 'nirv_recovery')
SIGNAL = [f'{v}_{m}' for v in ('ndmi', 'nirv')
          for m in ('resistance', 'recovery')]
CARBON = ['Lignin', 'Cellulose']
MIN_CELLS = 500


def add_tcc(d, aoi, k):
    env = rc.open_env(aoi)
    t = rc.cell_table({'tcc2013': env.tcc2013.values.astype(np.float32)},
                      rc.undisturbed(env, 2019), k)
    return d.merge(t[['cell_row', 'cell_col', 'tcc2013']],
                   on=['cell_row', 'cell_col'], how='left')


def tercile_rows(s, L, n_boot, dir_boot, learner, **tags):
    """Every per-tercile quantity on the cells s (L: lidar columns or [])"""
    rows = []
    for m in SIGNAL:
        v = s[m].std() / s[m.split('_')[0] + '_placebo'].std()
        rows.append(dict(quantity='signal_ratio', target=m, n=len(s),
                         value=v, learner=learner, **tags))
    _, T, _ = feature_sets(s)
    es = ENV + S_WALL
    resid = {}
    for name, base in (('Env+S', es), ('Env+S+L', es + L)):
        if name == 'Env+S+L' and not L:
            continue
        resid[name] = pd.DataFrame({c.replace('T_', f'R{len(resid)}_'):
                                    rc.crossfit_residuals(s, c, base,
                                                          'block1000')
                                    for c in T}, index=s.index)
    s = pd.concat([s] + list(resid.values()), axis=1)
    fs = {'Env+S': es, 'Env+S+T': es + T,
          'Env+S+Tres': es + list(resid['Env+S'].columns)}
    pairs = [('Env+S', 'Env+S+T'), ('Env+S', 'Env+S+Tres')]
    if L:
        fs.update({'Env+S+L': es + L,
                   'Env+S+L+Tres': es + L + list(resid['Env+S+L'].columns)})
        pairs.append(('Env+S+L', 'Env+S+L+Tres'))
    strata = (s.arid.astype(str) + '_' +
              (s.elevation // ELEV_BAND).astype(int).astype(str))
    for t in LADDER_TARGETS:
        sub = s[s[t].notna()]
        sets = fs if t == 'ndmi_recovery' else {'Env+S': es}
        r, _ = rc.ladder(sub, t, sets, 'block1000', n_boot=n_boot,
                         pairs=pairs if t == 'ndmi_recovery' else [],
                         learner=learner)
        for x in r:
            rows.append(dict(quantity='r2', target=t, n=x['n'],
                             features=x['features'], compare=x['compare'],
                             value=x['r2'], lo=x['lo'], hi=x['hi'],
                             learner=learner, **tags))
    t = 'ndmi_recovery'
    for i, name in enumerate(resid):
        c = f'R{i}_StructC'
        s[c] = zmean(s, [f'R{i}_{x}_2013' for x in CARBON])
        ok = s[t].notna()
        u = rc.within_strata_rho(s[ok], [c], t, strata[ok])[0]
        lo, hi = rho_within_boot(s[ok], c, t, strata[ok], s.block1000[ok],
                                 dir_boot)
        rows.append(dict(quantity='structc_rho', target=t, n=u['n'],
                         features=name, value=u['rho_within'], lo=lo, hi=hi,
                         learner=learner, **tags))
    return rows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', 'k', default=3, show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--dir-boot', default=500, show_default=True)
@click.option('--learner', type=click.Choice(rc.LEARNERS), default='hgb',
              show_default=True)
@click.option('--fold-seed', type=int, default=None)
@click.option('--tag', default='')
def main(outputdir, aois, k, n_boot, dir_boot, learner, fold_seed, tag):
    rc.FOLD_SEED = fold_seed
    outputdir.mkdir(parents=True, exist_ok=True)
    f = outputdir / f'density_strata{tag}.csv'
    for aoi in aois:
        rdir, ddir = aoi_dirs(aoi)
        d = add_tcc(build(aoi, k, rdir, ddir), aoi, k)
        runs = [('tcc', d, 'tcc2013', [])]
        a = structure_cells(d, aoi, k, 'aso')
        if len(a) >= 3 * MIN_CELLS:
            runs.append(('aso', a, 'aso_cover5', S_ASO))
        rows = []
        for strat, dd, col, L in runs:
            dd = dd[dd[col].notna()].copy()
            dd['tercile'], edges = pd.qcut(dd[col], 3, labels=False,
                                           retbins=True)
            click.echo(f'[{aoi}] {strat}: {len(dd)} cells, tercile edges '
                       + ', '.join(f'{e:.2f}' for e in edges))
            for q in range(3):
                s = dd[dd.tercile == q].copy()
                if len(s) < MIN_CELLS:
                    continue
                r = tercile_rows(s, L, n_boot, dir_boot, learner, aoi=aoi,
                                 strata=strat, tercile=q + 1,
                                 cover_lo=edges[q], cover_hi=edges[q + 1],
                                 fold_seed=fold_seed)
                rows += r
                g = {(x['quantity'], x['target'], x.get('features'),
                      x.get('compare')): x for x in r}
                rec = g[('r2', 'ndmi_recovery', 'Env+S', '')]
                click.echo(
                    f'  q{q + 1} n={len(s):5d} SNR rec '
                    f'{g[("signal_ratio", "ndmi_recovery", None, None)]["value"]:.2f}'
                    f'  Env+S {rec["value"]:.3f}  ' + '  '.join(
                        f'{x["features"]}-{x["compare"]} {x["value"]:+.3f} '
                        f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]' for x in r
                        if x['quantity'] == 'r2' and x['compare']
                        and x['target'] == 'ndmi_recovery') + '  ' +
                    '  '.join(f'StructC|{x["features"]} {x["value"]:+.3f}'
                              for x in r if x['quantity'] == 'structc_rho'))
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        if len(old):
            old = old[old.aoi != aoi]
        pd.concat([old, pd.DataFrame(rows)]).to_csv(f, index=False)


if __name__ == '__main__':
    main()
