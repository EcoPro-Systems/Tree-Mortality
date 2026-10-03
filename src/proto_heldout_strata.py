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


def aoi_mask(aoi):
    """The area's own cell mask (True = part of the area), or None"""
    path = rc.E / 'env' / f'{aoi}_mask.nc'
    if not path.exists():
        return None
    return xr.open_dataset(path)['keep'].values.astype(bool)


def elevation_cells(aoi, k=3, last_year=LAST_YEAR):
    """Cells with their mean elevation (m) on undisturbed forest pixels"""
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, last_year)
    m = aoi_mask(aoi)
    if m is not None:
        valid &= m
    return rc.cell_table({'elevation': env.elevation.values}, valid, k)


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
def main(outputfile, aois, scale):
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
        d = elevation_cells(aoi, scale)
        cuts = elevation_cutpoints(d.elevation.values)
        s = elevation_stratum(d.elevation.values, cuts)
        counts = np.bincount(s, minlength=N_STRATA)
        out['areas'][aoi] = {
            'cells': int(len(d)),
            'cutpoints_m': [round(c, 1) for c in cuts],
            'cells_per_stratum': [int(c) for c in counts],
            'elevation_range_m': [round(float(d.elevation.min()), 1),
                                  round(float(d.elevation.max()), 1)]}
        click.echo(f'[{aoi}] {len(d)} cells; cutpoints ' + ', '.join(
            f'{c:.0f}' for c in cuts) + ' m; per stratum ' +
            ', '.join(map(str, counts)))
    with open(outputfile, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)


if __name__ == '__main__':
    main()
