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

Held-out AOIs (response_common.HELD_OUT) are refused.

    python proto_forward_pilot.py $E/hls_results/forward_pilot \
        -a neon_soap_teak -a sierra_nf
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


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--rank', is_flag=True,
              help='Per-date percentile ranks instead of z-scores')
@click.option('--n-boot', default=1000, show_default=True)
def main(outputdir, aois, scale, rank, n_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    scale_m = rc.RES * scale
    std = 'rank' if rank else 'z'
    ladder_rows, transfer_rows, rho_rows = [], [], []
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
