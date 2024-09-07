#!/usr/bin/env python
import json
import click
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
from tqdm import tqdm
from dask.diagnostics import ProgressBar

from util import load_config


def get_climate_features(clim, year, feature_info):
    features = []

    for fbase, finfo in feature_info.items():
        for b in range(finfo['back']):
            cyear = clim.sel(year=(year - b)).assign_coords(year=[year])
            for c in range(1, finfo['cumulative'] + 1):
                fname = f'{fbase}{c}'
                new_name = f'{fname}-{b+1}'
                features.append(
                    cyear[fname].rename(new_name)
                )

    return features


def get_heat_features(heat, year, feature_info):
    features = []

    for fbase, finfo in feature_info.items():
        for b in range(finfo['back']):
            hyear = heat.sel(year=(year - b)).assign_coords(year=[year])
            new_name = f'{fbase}-{b+1}'
            features.append(
                hyear[fbase].rename(new_name)
            )

    return features


def get_basic_features(basic, feature_info):
    return [basic[fname] for fname in feature_info]


def to_samples(mort, clim, topo, heat, year, feature_info, target):

    # Make a new variable to hold copy of the year for each cell
    year_var = mort['id'].copy().rename('year_copy')
    year_var[:] = year
    year_var.attrs['long_name'] = 'year'

    # Merge climate and fold/id features
    features = [
        year_var,
        mort['id'],
        mort['fold'],
        mort['northings'],
        mort['eastings'],
        mort[target].sel(year=year),
    ]
    features += get_climate_features(clim, year, feature_info['climate'])
    features += get_basic_features(topo, feature_info['topography'])
    features += get_heat_features(heat, year, feature_info['heat'])

    feat_ds = xr.merge(features, join='inner', combine_attrs='drop')
    feat_ds = feat_ds.drop_vars(('year', 'spatial_ref',))
    feat_ds = feat_ds.stack(sample=('easting', 'northing'), create_index=False)
    feat_ds = feat_ds.reset_coords(('northing', 'easting'))
    feat_ds = feat_ds.rename({ 'year_copy': 'year' })

    return feat_ds


@click.command()
@click.argument('mortalityfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('climatefile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('topofile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('heatfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(mortalityfile, climatefile, topofile, heatfile, configfile, outputfile):

    config = load_config(configfile)

    years = config['years']
    finfo = config['features']
    chunks = config['chunks']
    target = config['target']

    mort = xr.open_zarr(mortalityfile)
    clim = xr.open_zarr(climatefile)
    topo = xr.open_zarr(topofile)
    heat = xr.open_zarr(heatfile)

    # Fix coordinate mis-match issue due to rounding
    mort = mort.assign_coords(
        easting=np.round(mort.easting.values, 4),
        northing=np.round(mort.northing.values, 4),
    )
    clim = clim.assign_coords(
        easting=np.round(clim.easting.values, 4),
        northing=np.round(clim.northing.values, 4),
    )
    topo = topo.assign_coords(
        easting=np.round(topo.easting.values, 4),
        northing=np.round(topo.northing.values, 4),
    )
    heat = heat.assign_coords(
        easting=np.round(heat.easting.values, 4),
        northing=np.round(heat.northing.values, 4),
    )

    combined = xr.concat(
        [
            to_samples(mort, clim, topo, heat, year, finfo, target)
            for year in tqdm(years, 'Yearly Features')
        ],
        dim='sample', coords='all', compat='identical'
    )

    print(f'Selecting data...')
    good = combined[target].notnull().compute()
    combined = combined.where(good, drop=True)
    for tf in tqdm(finfo['topography'], 'Excluding missing topography'):
        good = combined[tf].notnull().compute()
        combined = combined.where(good, drop=True)

    write_job = combined.chunk(chunks).to_zarr(
        outputfile, mode='w', compute=False, consolidated=True
    )

    print(f'Writing data...')
    with ProgressBar():
        write_job.persist()
    print('Done')


if __name__ == '__main__':
    main()
