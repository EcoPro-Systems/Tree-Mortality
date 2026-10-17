#!/usr/bin/env python
"""
Does a slow recovery from the 2012-16 drought mark the trees that died in
the 2020-22 drought? (NEON SOAP/TEAK, 90 m cells)

Targets are the lidar-tree cohorts of Hemming-Schroeder et al. (2023),
gridded by proto_trait_dynamics.lidar_cohort:
  mort2_frac  cohort B: trees live in 2018 and 2019, dead in 2021
  mort_frac   cohort A: trees live in 2013, dead by 2017-18
Both need >= 5 cohort trees in a cell and are weighted by the tree count.

1. Recovery -> next-drought mortality (desoto_ladder.csv). Block-CV ladder
   for mort2_frac on the 2020-22 base, Env+S with the cycle-2 forcing
   (proto_response_transfer.cycle_forcing): + each cycle-1 Landsat NDMI
   response (recovery, resistance, resilience) alone, then all three, then
   + cohort-A mortality; and + the residual recovery, the out-of-fold
   residual of cycle-1 NDMI recovery on cycle-1 Env+S ("recovered less
   than its environment predicts"). Pooled and by site.
2. Response-mortality signs (signs.csv): Spearman ρ of each cycle-1
   response with mort2_frac, raw, within aridity tercile x 200 m elevation
   strata (with a block-bootstrap CI), and with the out-of-fold residual of
   mort2_frac on the base (partial on Env+S).
3. Structural carbon and cycle-1 mortality (carbon_mortality.csv). The
   score is the mean of z-scored residual 2013 lignin and cellulose
   (residualized on cycle-1 Env+S, then on Env+S+L for each lidar source);
   its ρ with cohort-A mortality by site, under the lidar-tree cohort (all
   cells and the common cells) and the NAIP-based cohort of Stovall et al.
   (2019) on the common cells (rc.mortality_target).
4. Does the annual GEDI-fused structure register the 2020-22 die-off
   (--aef-loss; aef_loss.csv)? ρ of the AEF canopy height and cover change
   from 2019 to 2022 and to 2023 (fetch_gedi_fusion.py) with mort2_frac,
   raw and within strata, pooled and by site.

    python proto_recovery_mortality.py $E/hls_results/recovery_mortality
"""
import click
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import response_common as rc
from proto_response_metrics import ENV as ENV1, S_WALL
from proto_response_transfer import ENV as ENV2, cycle_forcing
from proto_response_traits import (ELEV_BAND, STRUCTURES, build,
                                   structure_cells)
from proto_trait_dynamics import lidar_cohort
from proto_trait_directions import rho_within_boot
from proto_forward_pilot import zmean

AOI = 'neon_soap_teak'
MIN_TREES = 5
MIN_SITE = 500
C1 = ['ndmi_recovery', 'ndmi_resistance', 'ndmi_resilience']
C1_SIGNS = C1 + ['nirv_recovery', 'nirv_resistance', 'nirv_resilience']
CARBON = ['Lignin', 'Cellulose']
TARGETS_A = (('hs', 'all'), ('hs', 'common'), ('stovall', 'common'))


def cells(k, response_dir):
    """NEON cells with cohort A and B mortality, the 2020-22 forcing and the
    cycle-1 responses (prefixed c1_) and their forcing"""
    transform, shape, _ = rc.aoi_info(AOI)
    env = rc.open_env(AOI)
    layers = cycle_forcing(AOI, 2)
    for c in 'AB':
        layers[f'n_{c}'], layers[f'died_{c}'] = lidar_cohort(
            transform, shape, c)
    d = rc.cell_table(layers, rc.undisturbed(env, 2021), k)
    with np.errstate(all='ignore'):
        for c, name in (('A', 'mort_frac'), ('B', 'mort2_frac')):
            n = d[f'n_{c}'] * d.n_px
            d[name] = np.where(n >= MIN_TREES, d[f'died_{c}'] / d[f'n_{c}'],
                               np.nan)
            d[name.replace('frac', 'n')] = np.where(n >= MIN_TREES, n, np.nan)
    m = pd.read_csv(response_dir / f'metrics_{AOI}_{rc.RES * k}m.csv')
    c1 = [c for c in ENV1 if c not in ENV2] + C1_SIGNS
    m = m[['cell_row', 'cell_col'] + c1].rename(
        columns={c: f'c1_{c}' for c in c1})
    d = d.merge(m, on=['cell_row', 'cell_col'], how='inner')
    d['arid'] = pd.qcut(d.cwd_clim, 3, labels=['wet', 'mid', 'dry'])
    d['site'] = np.where(d.cell_col < d.cell_col.max() / 2, 'SOAP', 'TEAK')
    d['strata'] = (d.arid.astype(str) + '_' +
                   (d.elevation // ELEV_BAND).astype(int).astype(str))
    return d


def env1(d):
    """Cycle-1 Env+S columns of the cell table"""
    return [c if c in ENV2 else f'c1_{c}' for c in ENV1] + S_WALL


def desoto(d, n_boot, learner, label):
    t, base = 'mort2_frac', ENV2 + S_WALL
    sub = d[d[t].notna()].copy()
    sub['c1_ndmi_recovery_res'] = rc.crossfit_residuals(
        sub, 'c1_ndmi_recovery', env1(sub), 'block1000')
    c1 = [f'c1_{c}' for c in C1]
    fs = {'Env+S': base}
    for c in c1 + ['c1_ndmi_recovery_res', 'mort_frac']:
        fs[f'+{c.replace("c1_", "")}'] = base + [c]
    fs['+c1_all'] = base + c1
    fs['+c1_all+mort_frac'] = base + c1 + ['mort_frac']
    pairs = [('Env+S', n) for n in fs if n != 'Env+S'] + \
        [('+c1_all', '+c1_all+mort_frac')]
    r, _ = rc.ladder(sub, t, fs, 'block1000', weight='mort2_n',
                     n_boot=n_boot, pairs=pairs, extra_blocks=('block5000',),
                     learner=learner)
    for x in r:
        x.update(subset=label)
    g = {x['features']: x for x in r
         if x['compare'] and x['boot_blocks'] == 'block1000'}
    a = [x for x in r if x['features'] == 'Env+S' and not x['compare']
         and x['boot_blocks'] == 'block1000'][0]
    click.echo(f'  {label:5s} n={len(sub):5d} Env+S {a["r2"]:.3f}  ' +
               '  '.join(f'{n} {x["r2"]:+.3f} [{x["lo"]:+.3f},'
                         f'{x["hi"]:+.3f}]' for n, x in g.items()))
    return r, sub


def signs(sub, label, n_boot):
    """ρ of each cycle-1 response with mort2_frac: raw, within strata, and
    with the residual of mort2_frac on the 2020-22 Env+S"""
    sub = sub.copy()
    sub['mort2_res'] = rc.crossfit_residuals(sub, 'mort2_frac',
                                             ENV2 + S_WALL, 'block1000')
    rows = []
    for c in [f'c1_{x}' for x in C1_SIGNS] + ['c1_ndmi_recovery_res']:
        ok = sub[c].notna()
        u = rc.within_strata_rho(sub, [c], 'mort2_frac', sub.strata)[0]
        lo, hi = rho_within_boot(sub[ok], c, 'mort2_frac', sub.strata[ok],
                                 sub.block1000[ok], n_boot)
        rows.append(dict(subset=label, feature=c, n=u['n'], rho=u['rho'],
                         rho_within=u['rho_within'], lo=lo, hi=hi,
                         rho_partial=spearmanr(sub[c][ok],
                                               sub.mort2_res[ok])[0]))
    return rows


def aef_loss(d, k, n_boot):
    import xarray as xr
    g = xr.open_dataset(rc.E / 'structure' / f'{AOI}_gedi_fusion.nc')
    layers = {f'd_{v}_19{y % 100}': g[f'{v}_aef_{y}'].values -
              g[f'{v}_aef_2019'].values
              for v in ('rh98', 'cover') for y in (2022, 2023)}
    t = rc.cell_table(layers, rc.undisturbed(rc.open_env(AOI), 2021), k)
    d = d[d.mort2_frac.notna()].merge(
        t[['cell_row', 'cell_col'] + list(layers)],
        on=['cell_row', 'cell_col'])
    rows = []
    for site in ('all', 'SOAP', 'TEAK'):
        s = d if site == 'all' else d[d.site == site]
        for c in layers:
            u = rc.within_strata_rho(s, [c], 'mort2_frac', s.strata)[0]
            lo, hi = rho_within_boot(s, c, 'mort2_frac', s.strata,
                                     s.block1000, n_boot)
            rows.append(dict(site=site, feature=c, n=u['n'], rho=u['rho'],
                             rho_within=u['rho_within'], lo=lo, hi=hi))
            click.echo(f'  {site:4s} {c:14s} n={u["n"]:5d} raw '
                       f'{u["rho"]:+.3f} within {u["rho_within"]:+.3f} '
                       f'[{lo:+.3f},{hi:+.3f}]')
    return rows


def carbon(k, response_dir, dynamics_dir, n_boot):
    rows = []
    for source, which in TARGETS_A:
        d = build(AOI, k, response_dir, dynamics_dir,
                  mortality_source=source, mortality_cells=which)
        d['strata'] = (d.arid.astype(str) + '_' +
                       (d.elevation // ELEV_BAND).astype(int).astype(str))
        bases = [('Env+S', d, [])]
        for src in STRUCTURES:
            bases.append((f'Env+S+L:{src}', structure_cells(d, AOI, k, src),
                          STRUCTURES[src]))
        for bname, dd, L in bases:
            dd = dd[dd.mort_frac.notna()].copy()
            if len(dd) < MIN_SITE:
                continue
            cols = []
            for x in CARBON:
                dd[f'R_{x}'] = rc.crossfit_residuals(
                    dd, f'T_{x}_2013', ENV1 + S_WALL + L, 'block1000')
                cols.append(f'R_{x}')
            dd['R_StructC'] = zmean(dd, cols)
            for site in ('all', 'SOAP', 'TEAK'):
                s = dd if site == 'all' else dd[dd.site == site]
                if len(s) < MIN_SITE:
                    continue
                for c in cols + ['R_StructC']:
                    u = rc.within_strata_rho(s, [c], 'mort_frac',
                                             s.strata)[0]
                    lo, hi = rho_within_boot(s, c, 'mort_frac', s.strata,
                                             s.block1000, n_boot)
                    rows.append(dict(mortality=source, cells=which,
                                     residual_on=bname, site=site,
                                     feature=c, n=u['n'], rho=u['rho'],
                                     rho_within=u['rho_within'], lo=lo,
                                     hi=hi))
                sc = [x for x in rows[-3:]]
                click.echo(f'  {source}/{which} {bname:17s} {site:4s} '
                           f'n={len(s):5d}  ' + '  '.join(
                               f'{x["feature"][2:]} {x["rho_within"]:+.3f} '
                               f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]'
                               for x in sc))
    return rows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--scale', 'k', default=3, show_default=True)
@click.option('--response-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/response')
@click.option('--dynamics-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/trait_dynamics')
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--dir-boot', default=500, show_default=True)
@click.option('--learner', type=click.Choice(rc.LEARNERS),
              default='hgb_tuned', show_default=True)
@click.option('--skip-carbon', is_flag=True)
@click.option('--aef-loss', 'aef_loss_only', is_flag=True,
              help='Only the GEDI-fused canopy loss vs cohort B (4.)')
@click.option('--fold-seed', type=int, default=None)
@click.option('--tag', default='')
def main(outputdir, k, response_dir, dynamics_dir, n_boot, dir_boot, learner,
         skip_carbon, aef_loss_only, fold_seed, tag):
    rc.FOLD_SEED = fold_seed
    outputdir.mkdir(parents=True, exist_ok=True)
    d = cells(k, response_dir)
    if aef_loss_only:
        pd.DataFrame(aef_loss(d, k, dir_boot)).to_csv(
            outputdir / f'aef_loss{tag}.csv', index=False)
        return
    b = d[d.mort2_frac.notna()]
    click.echo(f'[{AOI} {rc.RES * k} m] cohort B: {len(b)} cells, weighted '
               f'mean {np.average(b.mort2_frac, weights=b.mort2_n):.3f} ' +
               ' '.join(f'{s} {len(g)} '
                        f'{np.average(g.mort2_frac, weights=g.mort2_n):.3f}'
                        for s, g in b.groupby('site')))
    d.to_csv(outputdir / f'cells_{rc.RES * k}m{tag}.csv', index=False)
    lrows, srows = [], []
    for label in ('all', 'SOAP', 'TEAK'):
        dd = d if label == 'all' else d[d.site == label]
        if dd.mort2_frac.notna().sum() < MIN_SITE:
            click.echo(f'  {label}: < {MIN_SITE} cohort-B cells, skipped')
            continue
        r, sub = desoto(dd, n_boot, learner, label)
        lrows += r
        srows += signs(sub, label, dir_boot)
    for x in lrows:
        x.update(learner=learner, fold_seed=fold_seed, scale_m=rc.RES * k)
    pd.DataFrame(lrows).to_csv(outputdir / f'desoto_ladder{tag}.csv',
                               index=False)
    pd.DataFrame(srows).to_csv(outputdir / f'signs{tag}.csv', index=False)
    for x in srows:
        click.echo(f'  ρ {x["subset"]:5s} {x["feature"]:24s} raw '
                   f'{x["rho"]:+.3f}  within {x["rho_within"]:+.3f} '
                   f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]  partial '
                   f'{x["rho_partial"]:+.3f}')
    if not skip_carbon:
        pd.DataFrame(carbon(k, response_dir, dynamics_dir, dir_boot)).to_csv(
            outputdir / f'carbon_mortality{tag}.csv', index=False)


if __name__ == '__main__':
    main()
