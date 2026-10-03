#!/usr/bin/env python
"""
Can traits measured after one drought help predict the response to the
next? A forward-prediction pilot on the NEON AOI (and sierra_nf where
enough cells survive the 2020 Creek Fire mask).

Responses are the Landsat NDMI/NIRv resistance, recovery and resilience of
the two drought cycles (proto_response_transfer.build_cycle):
    cycle 1  baseline 2008-11, drought 2014-16, post 2017-19
    cycle 2  baseline 2017-19, drought 2020-22, post 2023-25
on forest cells with no fire or harvest 2000-2025.

Blocks:
  Env+S  cycle-2 climate normals and drought anomalies, terrain, pre-2012
         wall-to-wall structure (cycle-generic names, as in the transfer
         experiment)
  Leg    legacy: the cycle-1 response metrics of the cell
  T18    June 2018 AVIRIS-C traits (14 means + green-fraction QC), cross-
         track normalized, then standardized per date (z-score over the
         cells; --rank uses percentile ranks instead)
  T13    the June 2013 traits, standardized the same way (reference: the
         trait state before the first drought)

1. Within cycle 2 (1 km block CV, block-bootstrap CIs):
       Env+S | +Leg | +T18 | +Leg+T18 | +T13 | +Leg+T13
2. Across cycles: a model fitted on cycle 1 (Env+S+T13 -> cycle-1
   response) applied to cycle 2 with T18 in the place of T13. The 2018
   levels are offset from 2013 (N about +1.2 SD), so only the per-date
   standardized traits can be swapped. Compared with the same transfer
   without traits.
3. Within-stratum Spearman ρ of 2018 N, LMA and lignin with cycle-2
   responses.

Landsat control (--landsat SCENE, repeatable; --landsat-composites): do
the 2018 traits add to 2018 Landsat reflectance? L1 is the first scene's
bands, NDVI, NDMI, NBR and NIRv (fetch_landsat_c2_ee.py --scene, e.g.
20180619, three days before the June 2018 flight), L2 all scenes with
their change (last - first), LM the year's June and Jul-Sep Landsat 8
composites. On the cells with every scene, paired:
    L1 vs T18 over Env+S and over Env+S+Leg (one date each)
    T18 beyond Env+S+L1, Env+S+Leg+L1, Env+S+Leg+L2 (and +LM)
    T18 vs T13 beyond Env+S+Leg+L1 (post- vs pre-drought traits)
    T18 vs the second scene (or LM), each added beyond Env+S+Leg+L1
and the within-stratum ρ (1 km block-bootstrap CIs) of N, LMA, lignin and
cellulose residualized on Env+S and on Env+S+L1. The cycle-2 responses are
Landsat 8/9 with a 2017-19 baseline, so a 2018 Landsat block shares
sensor, processing and baseline years with them. Outputs landsat_<std>.csv
and landsat_directions_<std>.csv (--tag adds a suffix); the runs without
--landsat are unchanged.

Held-out AOIs (response_common.HELD_OUT) are refused.

    python proto_forward_pilot.py $E/hls_results/forward_pilot \
        -a neon_soap_teak -a sierra_nf
    python proto_forward_pilot.py $E/hls_results/forward_pilot_landsat \
        -a neon_soap_teak -a sierra_nf --landsat 20180619 \
        --landsat 20180822 --landsat-composites
"""
import click
import numpy as np
import pandas as pd
from pathlib import Path

import response_common as rc
from proto_response_metrics import S_WALL
from proto_response_transfer import build_cycle, ENV, skill
from proto_response_traits import TRAITS, ELEV_BAND, trait_layers

TARGETS = [f'{v}_{m}' for v in ('ndmi', 'nirv')
           for m in ('resistance', 'recovery', 'resilience')]
MIN_CELLS = 5000


def standardize(d, cols, rank):
    out = d[cols].copy()
    for c in cols:
        out[c] = (d[c].rank(pct=True) if rank else
                  (d[c] - d[c].mean()) / d[c].std())
    return out


def generic(cols):
    """Trait columns without their year, so 2013 and 2018 can swap"""
    return [c.rsplit('_', 1)[0] if c.startswith('T_') else c for c in cols]


def trait_block(aoi, valid, k, year, rank):
    t = trait_layers(aoi, valid, k, y0=year)
    cols = [f'T_{x}_{year}' for x in TRAITS] + [f'T_qcfc_{year}']
    z = standardize(t, cols, rank)
    z.columns = [c.replace(f'_{year}', '') + f'_{year % 100}' for c in cols]
    return pd.concat([t[['cell_row', 'cell_col']], z], axis=1)


def build(aoi, k, rank):
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, 2025)
    c1 = build_cycle(aoi, k, 1, valid)
    c2 = build_cycle(aoi, k, 2, valid)
    key = ['cell_row', 'cell_col']
    leg = c1[key + TARGETS].rename(columns={t: f'leg_{t}' for t in TARGETS})
    c1r = c1[key + TARGETS + ENV + S_WALL]
    d = c2.merge(leg, on=key)
    for y in (2013, 2018):
        tb = trait_block(aoi, valid, k, y, rank)
        d = d.merge(tb, on=key)
        c1r = c1r.merge(tb, on=key)
    d['arid'] = pd.qcut(d.cwd_clim, 3, labels=['wet', 'mid', 'dry'])
    return d, c1r


LANDSAT_VARS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi',
                'ndmi', 'nbr', 'nirv']
DIR_TRAITS = ['Nitrogen', 'LMA', 'Lignin', 'Cellulose']


def landsat_cols(d, aoi, k, scenes, composites, year=2018):
    """Merge the Landsat scene (l8@<date>:<var>, l8@chg:<var>) and
    composite (l8multi:<var>_<jun|sum>) blocks; returns (table, blocks)"""
    from proto_spaceborne_only import landsat_block
    valid = rc.undisturbed(rc.open_env(aoi), 2025)
    key = ['cell_row', 'cell_col']
    for s in scenes:
        t = landsat_block(aoi, valid, k, 'l8raw', s)
        t = t.rename(columns={f'l8raw:{v}': f'l8@{s}:{v}'
                              for v in LANDSAT_VARS})
        d = d.merge(t[key + [f'l8@{s}:{v}' for v in LANDSAT_VARS]], on=key,
                    how='left')
    blocks = {'L1': [f'l8@{scenes[0]}:{v}' for v in LANDSAT_VARS]}
    need = [f'l8@{s}:nir' for s in scenes]
    if len(scenes) > 1:
        lo, hi = min(scenes), max(scenes)
        for v in LANDSAT_VARS:
            d[f'l8@chg:{v}'] = d[f'l8@{hi}:{v}'] - d[f'l8@{lo}:{v}']
        blocks['L2'] = [f'l8@{s}:{v}' for s in scenes + ['chg']
                        for v in LANDSAT_VARS]
    if composites:
        t = landsat_block(aoi, valid, k, 'l8multi', None, year)
        multi = [c for c in t if c.startswith('l8multi:')]
        d = d.merge(t[key + multi], on=key, how='left')
        blocks['LM'] = multi
        need += ['l8multi:nir_jun', 'l8multi:nir_sum']
    return d[d[need].notna().all(1)].copy(), blocks


def landsat_control(d, aoi, k, scenes, composites, n_boot, dir_boot):
    """Do the 2018 traits add to 2018 Landsat? Paired ladder and residual
    directions on the cells with every Landsat input"""
    from proto_trait_directions import rho_within_boot
    n0 = len(d)
    d, lb = landsat_cols(d, aoi, k, scenes, composites)
    click.echo(f'[{aoi}] Landsat control: {len(d)} of {n0} cells with '
               f'{", ".join(scenes)}' + (' and the composites'
                                         if composites else ''))
    T13 = [f'T_{x}_13' for x in TRAITS] + ['T_qcfc_13']
    T18 = [f'T_{x}_18' for x in TRAITS] + ['T_qcfc_18']
    es = ENV + S_WALL
    legall = [f'leg_{x}' for x in TARGETS]
    L1 = lb['L1']
    fs = {'Env+S': es, 'Env+S+L1': es + L1, 'Env+S+T18': es + T18,
          'Env+S+L1+T18': es + L1 + T18, 'Env+S+Leg': es + legall,
          'Env+S+Leg+L1': es + legall + L1,
          'Env+S+Leg+T18': es + legall + T18,
          'Env+S+Leg+L1+T18': es + legall + L1 + T18,
          'Env+S+Leg+L1+T13': es + legall + L1 + T13}
    pairs = [('Env+S', 'Env+S+L1'), ('Env+S', 'Env+S+T18'),
             ('Env+S+L1', 'Env+S+T18'), ('Env+S+Leg+L1', 'Env+S+Leg+T18'),
             ('Env+S+L1', 'Env+S+L1+T18'), ('Env+S+Leg', 'Env+S+Leg+L1'),
             ('Env+S+Leg+L1', 'Env+S+Leg+L1+T18'),
             ('Env+S+Leg+L1', 'Env+S+Leg+L1+T13'),
             ('Env+S+Leg+L1+T13', 'Env+S+Leg+L1+T18')]
    for b in ('L2', 'LM'):
        if b in lb:
            fs[f'Env+S+Leg+{b}'] = es + legall + lb[b]
            fs[f'Env+S+Leg+{b}+T18'] = es + legall + lb[b] + T18
            # one more Landsat scene (or the composites) vs the 2018 traits,
            # each beyond Env+S+Leg+L1
            pairs += [('Env+S+Leg+L1', f'Env+S+Leg+{b}'),
                      (f'Env+S+Leg+{b}', f'Env+S+Leg+{b}+T18'),
                      (f'Env+S+Leg+{b}', 'Env+S+Leg+L1+T18')]
    rows, drows = [], []
    strata = (d.arid.astype(str) + '_' +
              (d.elevation // ELEV_BAND).astype(int).astype(str))
    R = {f'R_{x}|{bn}': rc.crossfit_residuals(d, f'T_{x}_18', base,
                                              'block1000')
         for x in DIR_TRAITS
         for bn, base in (('Env+S', es), ('Env+S+L1', es + L1))}
    dd = pd.concat([d, pd.DataFrame(R, index=d.index)], axis=1)
    for t in TARGETS:
        sub = d[d[t].notna() & np.isfinite(d[t])]
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                         pairs=pairs, extra_blocks=('block5000',))
        rows += r
        g = {(x['features'], x['compare']): x for x in r
             if x['compare'] and x['boot_blocks'] == 'block1000'}
        base = [x for x in r if x['features'] == 'Env+S'
                and not x['compare']][0]
        click.echo(f'  {t} n={len(sub)} Env+S {base["r2"]:.3f}')
        for a, b in pairs:
            x = g[(b, a)]
            click.echo(f'    {b:22s} - {a:18s} {x["r2"]:+.3f} '
                       f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]')
        if not t.endswith('recovery'):
            continue
        uni = {u['feature']: u for u in rc.within_strata_rho(
            dd, list(R), t, strata)}
        for col in R:
            lo, hi = rho_within_boot(dd, col, t, strata, dd.block1000,
                                     dir_boot)
            x, bn = col[2:].split('|')
            drows.append(dict(target=t, trait=x, residual_on=bn,
                              n=uni[col]['n'],
                              rho_within=uni[col]['rho_within'], lo=lo,
                              hi=hi))
        click.echo('    residual directions: ' + '  '.join(
            f'{x["trait"][:4]}|{x["residual_on"]} {x["rho_within"]:+.2f} '
            f'[{x["lo"]:+.2f},{x["hi"]:+.2f}]' for x in drows
            if x['target'] == t))
    return rows, drows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--rank', is_flag=True,
              help='Per-date percentile ranks instead of z-scores')
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--landsat', 'scenes', multiple=True,
              help='Landsat 8 scene date(s) YYYYMMDD for the Landsat '
                   'control (wdts/sim/<aoi>_l8_<date>.nc); repeatable')
@click.option('--landsat-composites', is_flag=True,
              help='Also the 2018 June and Jul-Sep Landsat 8 composites')
@click.option('--dir-boot', default=500, show_default=True,
              help='Bootstrap draws for the direction CIs')
@click.option('--tag', default='', help='Suffix of the Landsat-control '
                                        'output files')
def main(outputdir, aois, scale, rank, n_boot, scenes, landsat_composites,
         dir_boot, tag):
    outputdir.mkdir(parents=True, exist_ok=True)
    scale_m = rc.RES * scale
    std = 'rank' if rank else 'z'
    ladder_rows, transfer_rows, rho_rows = [], [], []
    if scenes:
        lrows, drows = [], []
        for aoi in aois:
            d, _ = build(aoi, scale, rank)
            tags = dict(aoi=aoi, scale_m=scale_m, standardize=std)
            r, dr = landsat_control(d, aoi, scale, list(scenes),
                                    landsat_composites, n_boot, dir_boot)
            lrows += [dict(x, **tags) for x in r]
            drows += [dict(x, **tags) for x in dr]
            pd.DataFrame(lrows).to_csv(
                outputdir / f'landsat_{std}{tag}.csv', index=False)
            pd.DataFrame(drows).to_csv(
                outputdir / f'landsat_directions_{std}{tag}.csv',
                index=False)
        return
    for aoi in aois:
        d, c1 = build(aoi, scale, rank)
        if len(d) < MIN_CELLS:
            click.echo(f'[{aoi}] only {len(d)} cells; skipped')
            continue
        click.echo(f'[{aoi} {scale_m} m] {len(d)} cells ({std})')
        T13 = [f'T_{x}_13' for x in TRAITS] + ['T_qcfc_13']
        T18 = [f'T_{x}_18' for x in TRAITS] + ['T_qcfc_18']
        es = ENV + S_WALL
        tags = dict(aoi=aoi, scale_m=scale_m, standardize=std)
        for t in TARGETS:
            leg = [f'leg_{t}']
            legall = [f'leg_{x}' for x in TARGETS]
            fs = {'Env+S': es, 'Env+S+Leg': es + legall,
                  'Env+S+T18': es + T18, 'Env+S+Leg+T18': es + legall + T18,
                  'Env+S+T13': es + T13, 'Env+S+Leg+T13': es + legall + T13}
            pairs = [('Env+S', 'Env+S+Leg'), ('Env+S', 'Env+S+T18'),
                     ('Env+S+Leg', 'Env+S+Leg+T18'), ('Env+S', 'Env+S+T13'),
                     ('Env+S+Leg', 'Env+S+Leg+T13'),
                     ('Env+S+T13', 'Env+S+T18')]
            sub = d[d[t].notna() & np.isfinite(d[t])]
            r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                             pairs=pairs)
            ladder_rows += [dict(x, **tags) for x in r]
            g = {f'{x["features"]}-{x["compare"]}': x for x in r
                 if x['compare'] and x['boot_blocks'] == 'block1000'}
            base = [x for x in r if x['features'] == 'Env+S'
                    and not x['compare']][0]
            click.echo(f'  {t:18s} Env+S {base["r2"]:.3f}  ' + '  '.join(
                f'{k.split("-")[0].replace("Env+S+", "+")}'
                f'|{k.split("-")[1].replace("Env+S", "E")} '
                f'{g[k]["r2"]:+.3f} [{g[k]["lo"]:+.3f},{g[k]["hi"]:+.3f}]'
                for k in g))

            # Cycle 1 -> cycle 2 with the trait swap
            a = c1[c1[t].notna()]
            b = sub
            for name, cols_a, cols_b in (
                    ('Env+S', es, es),
                    ('Env+S+T', es + T13, es + T18),
                    ('Env+S+T (2013 traits)', es + T13, es + T13)):
                m = rc.hgb().fit(a[cols_a].set_axis(generic(cols_a), axis=1),
                                 a[t])
                p = m.predict(b[cols_b].set_axis(generic(cols_b), axis=1))
                transfer_rows.append(dict(**tags, target=t, features=name,
                                          n=len(b), **skill(
                                              b[t].values, p,
                                              b.block1000.values, n_boot)))
            tr = transfer_rows[-3:]
            click.echo('    C1->C2 ' + '  '.join(
                f'{x["features"]} R2 {x["r2"]:+.3f} rho {x["rho"]:.3f}'
                for x in tr))

            strata = (sub.arid.astype(str) + '_' +
                      (sub.elevation // ELEV_BAND).astype(int).astype(str))
            cols = [f'T_{x}_{y}' for x in ('Nitrogen', 'LMA', 'Lignin',
                                           'Cellulose') for y in (13, 18)]
            rho_rows += [dict(u, **tags) for u in rc.within_strata_rho(
                sub, cols + leg, t, strata)]
        for name, rows in (('ladder', ladder_rows),
                           ('transfer', transfer_rows),
                           ('directions', rho_rows)):
            pd.DataFrame(rows).to_csv(outputdir / f'{name}_{std}.csv',
                                      index=False)


if __name__ == '__main__':
    main()
