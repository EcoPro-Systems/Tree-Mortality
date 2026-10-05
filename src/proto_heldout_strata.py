#!/usr/bin/env python
"""
Elevation strata for the held-out sites, fixed before any of their
2020-22 responses exist.

The cross-drought transfer ranks cells correctly within elevation bands
even where it ranks them in reverse pooled (docs/drought_response.md §21),
so rank skill on the held-out sites is reported pooled and within
elevation quartiles. The quartile cutpoints are computed here, for each
area separately, from elevation alone: 90 m cells (k x k pixels, >= 70%
valid) of NLCD 2013 forest with no fire (2000-) or harvest (2005-) through
2025 (response_common.undisturbed), plus the area's own cell mask where it
has one (env/<aoi>_mask.nc, variable keep; used for areas defined as a box
minus other AOIs). No Landsat response or composite enters, so the
cutpoints cannot depend on the responses they will stratify.

Writes the cutpoints, cell counts and the definition to a YAML file that
is committed with the code (config/heldout_strata.yml).

    python proto_heldout_strata.py ../config/heldout_strata.yml \
        -a stanislaus -a seki -a yosemite_rest

Candidate definitions of an area, written under their own key (--key) to a
separate file so the committed cutpoints stay as they are:
  --conifer        only cells that are >= 50% conifer on their valid pixels
                   (LANDFIRE 2014 EVT, fetch_forest_type.py: the pine,
                   mesic, red-fir and subalpine groups, without meadows)
  --min-elevation  only cells at or above this mean elevation (m)
  --max-elevation  only cells below this mean elevation (m)
  --pool AOI       add another area's cells (with the same filters)

    python proto_heldout_strata.py $E/hls_results/heldout_variants/\
strata_candidates.yml -a yosemite_rest --pool seki --conifer \
        --key yosemite_rest+seki_conifer
"""
import click
import numpy as np
import xarray as xr
import yaml
from datetime import date
from pathlib import Path

import response_common as rc

N_STRATA = 4
LAST_YEAR = 2025
CONIFER = ('pine', 'mesic', 'red_fir', 'subalpine')


def aoi_mask(aoi):
    """The area's own cell mask (True = part of the area), or None"""
    path = rc.E / 'env' / f'{aoi}_mask.nc'
    if not path.exists():
        return None
    return xr.open_dataset(path)['keep'].values.astype(bool)


def conifer_pixels(aoi):
    """Per pixel: True where the LANDFIRE 2014 EVT class is conifer forest
    or woodland (CONIFER groups of fetch_forest_type.py, without meadows)"""
    f = xr.open_dataset(rc.E / 'env' / f'{aoi}_ftype.nc')
    codes = [int(c) for c, g, nm in (r.split('|', 2) for r in
                                      f.attrs['evt_table'].split('; '))
             if g in CONIFER and 'Meadow' not in nm]
    return np.isin(f.lf14_evt.values, codes)


def elevation_cells(aoi, k=3, last_year=LAST_YEAR, conifer=False,
                    min_elevation=None, max_elevation=None):
    """Cells with their mean elevation (m) on undisturbed forest pixels,
    and their conifer share where the area has a forest-type grid"""
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, last_year)
    m = aoi_mask(aoi)
    if m is not None:
        valid &= m
    layers = {'elevation': env.elevation.values}
    if conifer or (rc.E / 'env' / f'{aoi}_ftype.nc').exists():
        layers['conifer'] = conifer_pixels(aoi).astype(np.float32)
    d = rc.cell_table(layers, valid, k)
    if conifer:
        d = d[d.conifer >= 0.5]
    if min_elevation is not None:
        d = d[d.elevation >= min_elevation]
    if max_elevation is not None:
        d = d[d.elevation < max_elevation]
    return d


def elevation_cutpoints(elev, n=N_STRATA):
    """Inner quantile cutpoints (n - 1 values) of cell elevation"""
    return [float(np.quantile(elev, q)) for q in np.arange(1, n) / n]


def elevation_stratum(elev, cuts):
    """0 .. len(cuts) for each elevation (bins closed on the left)"""
    return np.searchsorted(np.asarray(cuts), np.asarray(elev), side='right')


def load_cutpoints(path, aoi):
    with open(path) as f:
        return yaml.safe_load(f)['areas'][aoi]['cutpoints_m']


@click.command()
@click.argument('outputfile', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', default=3, show_default=True)
@click.option('--conifer', is_flag=True,
              help='Only cells >= 50% conifer (LANDFIRE 2014 EVT)')
@click.option('--min-elevation', type=float,
              help='Only cells at or above this elevation (m)')
@click.option('--max-elevation', type=float,
              help='Only cells below this elevation (m)')
@click.option('--pool', 'pools', multiple=True,
              help='Add these areas\' cells to each --aoi')
@click.option('--key', help='Name to write the (single) area under')
def main(outputfile, aois, scale, conifer, min_elevation, max_elevation,
         pools, key):
    if key and len(aois) > 1:
        raise click.UsageError('--key takes a single --aoi')
    out = {'description': (
        f'Elevation quartile cutpoints (m) of {rc.RES * scale} m cells of '
        f'NLCD 2013 forest with no fire or harvest through {LAST_YEAR} '
        f'(>= {rc.MIN_VALID:.0%} of the cell valid); elevation only, '
        f'computed before any 2020-22 response of these areas'),
        'computed': date.today().isoformat(), 'scale_m': rc.RES * scale,
        'areas': {}}
    if outputfile.exists():
        with open(outputfile) as f:
            old = yaml.safe_load(f) or {}
        out['areas'].update(old.get('areas', {}))
    for aoi in aois:
        parts = [elevation_cells(a, scale, conifer=conifer,
                                 min_elevation=min_elevation,
                                 max_elevation=max_elevation)
                 for a in (aoi, *pools)]
        elev = np.concatenate([p.elevation.values for p in parts])
        cuts = elevation_cutpoints(elev)
        s = elevation_stratum(elev, cuts)
        counts = np.bincount(s, minlength=N_STRATA)
        name = key or aoi
        out['areas'][name] = {
            'cells': int(len(elev)),
            'cutpoints_m': [round(c, 1) for c in cuts],
            'cells_per_stratum': [int(c) for c in counts],
            'elevation_range_m': [round(float(elev.min()), 1),
                                  round(float(elev.max()), 1)]}
        if (pools or conifer or min_elevation is not None
                or max_elevation is not None):
            out['areas'][name]['definition'] = {
                'areas': [aoi, *pools], 'conifer': conifer,
                'min_elevation_m': min_elevation,
                'max_elevation_m': max_elevation,
                'cells_per_area': [int(len(p)) for p in parts]}
        click.echo(f'[{name}] {len(elev)} cells; cutpoints ' + ', '.join(
            f'{c:.0f}' for c in cuts) + ' m; per stratum ' +
            ', '.join(map(str, counts)))
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outputfile, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)


if __name__ == '__main__':
    main()
