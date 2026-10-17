#!/usr/bin/env python
"""
Does the structural-carbon direction hold beyond canopy water? (cycle 1,
90 m cells, per AOI)

The structural-carbon score (mean of z-scored residual 2013 lignin and
cellulose; §37) goes with slower recovery. Canopy water also tracks
recovery and covaries with leaf structure. On the cells of the canopy-water
check (proto_response_traits.py --water-block), lignin and cellulose are
residualized on B and on B+W (B = Env+S, or Env+S+L for a lidar source),
and the within-stratum ρ (aridity tercile x 200 m elevation band) of each
score with the response is reported with a block-bootstrap CI.

    python proto_water_structc.py $E/hls_results/water_baseline \
        -a neon_soap_teak -a sierra_nf
"""
import click
import pandas as pd
from pathlib import Path

import response_common as rc
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import (ELEV_BAND, STRUCTURES, build,
                                   structure_cells, water_layers)
from proto_spaceborne_only import aoi_dirs
from proto_trait_directions import rho_within_boot
from proto_forward_pilot import zmean

CARBON = ['Lignin', 'Cellulose']
TARGETS = ('ndmi_recovery', 'nirv_recovery', 'mort_frac')
LIDAR = {'neon_soap_teak': ('aso', 'lvis2008'), 'sierra_nf': ('aso',)}


def structc_rows(d, W, base, base_name, dir_boot, **tags):
    """ρ_within of the score residualized on base and on base+W"""
    strata = (d.arid.astype(str) + '_' +
              (d.elevation // ELEV_BAND).astype(int).astype(str))
    rows = []
    for resid_on, cols in ((base_name, base), (f'{base_name}+W', base + W)):
        s = zmean(pd.DataFrame({
            x: rc.crossfit_residuals(d, f'T_{x}_2013', cols, 'block1000')
            for x in CARBON}, index=d.index), CARBON)
        d = d.assign(StructC=s)
        for t in TARGETS:
            if t not in d:
                continue
            ok = d[t].notna() & d.StructC.notna()
            if ok.sum() < 500:
                continue
            u = rc.within_strata_rho(d[ok], ['StructC'], t, strata[ok])[0]
            lo, hi = rho_within_boot(d[ok], 'StructC', t, strata[ok],
                                     d.block1000[ok], dir_boot)
            rows.append(dict(target=t, resid_on=resid_on, n=u['n'],
                             rho_within=u['rho_within'], lo=lo, hi=hi,
                             **tags))
    return rows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('-w', '--water-block', 'blocks', multiple=True,
              default=('aviris', 'brodrick', 'both'), show_default=True)
@click.option('--scale', 'k', default=3, show_default=True)
@click.option('--dir-boot', default=500, show_default=True)
def main(outputdir, aois, blocks, k, dir_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    f = outputdir / 'water_structc.csv'
    for aoi in aois:
        rdir, ddir = aoi_dirs(aoi)
        d0 = build(aoi, k, rdir, ddir)
        rows = []
        for block in blocks:
            d, W = water_layers(d0, aoi, k, block)
            runs = [('none', 'Env+S', d, ENV + S_WALL)]
            for src in LIDAR[aoi]:
                runs.append((src, 'Env+S+L', structure_cells(d, aoi, k, src),
                             ENV + S_WALL + STRUCTURES[src]))
            for check, name, dd, base in runs:
                r = structc_rows(dd, W, base, name, dir_boot, aoi=aoi,
                                 water=block, check=check)
                rows += r
                click.echo(f'[{aoi}] {block} {check} ({len(dd)} cells)  ' +
                           '  '.join(f'{x["target"]}|{x["resid_on"]} '
                                     f'{x["rho_within"]:+.3f} '
                                     f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]'
                                     for x in r))
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        if len(old):
            old = old[old.aoi != aoi]
        pd.concat([old, pd.DataFrame(rows)]).to_csv(f, index=False)


if __name__ == '__main__':
    main()
