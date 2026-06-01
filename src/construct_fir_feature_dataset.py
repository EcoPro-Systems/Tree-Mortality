#!/usr/bin/env python
import json
import click
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from dask.diagnostics import ProgressBar

from util import load_config
from construct_training_dataset import (
    get_climate_features,
    get_heat_features,
)


def to_samples(ds, year, feature_info):

    # Make a new variable to hold copy of the year for each cell
    year_var = ds['id'].copy().rename('year_copy')
    year_var[:] = year
    year_var.attrs['long_name'] = 'year'

    # Merge climate and fold/id features
    features = [
        year_var,
        ds['id'],
        ds['northings'],
        ds['eastings'],
    ]
    features += get_climate_features(ds, year, feature_info['climate'])
    features += get_heat_features(ds, year, feature_info['heat'])

    feat_ds = xr.merge(features, join='inner', combine_attrs='drop')
    feat_ds = feat_ds.drop_vars(('spatial_ref',))
    feat_ds = feat_ds.stack(sample=('easting', 'northing'), create_index=False)
    feat_ds = feat_ds.reset_coords(('northing', 'easting'))
    feat_ds = feat_ds.rename({ 'year_copy': 'year' })

    good = feat_ds['eastings'].notnull().compute()
    feat_ds = feat_ds.where(good, drop=True)

    return feat_ds


def apply_mask(dataset, region_df):
    dataset = dataset.rename({ 'easting': 'x', 'northing': 'y' })
    dataset = dataset.rio.clip(
        region_df.geometry.values, region_df.crs
    )
    dataset = dataset.rename({ 'x': 'easting', 'y': 'northing' })
    return dataset


def append_ne(ds):
    cols = len(ds.easting.values)
    rows = len(ds.northing.values)
    eastings = np.zeros((rows, cols), dtype=float) + ds.easting.values
    northings = (np.zeros((rows, cols), dtype=float).T + ds.northing.values).T
    idx = np.arange(cols * rows, dtype=int).reshape((rows, cols)).T

    dataset = xr.Dataset(
        data_vars={
            'id': (
                ['northing', 'easting'],
                idx.T,
                {
                    'long_name': 'cell identifier',
                }
            ),
            'northings': (
                ['northing', 'easting'],
                northings,
                {
                    'long_name': 'northing coordinate',
                }
            ),
            'eastings': (
                ['northing', 'easting'],
                eastings,
                {
                    'long_name': 'easting coordinate',
                }
            ),
        },
        coords={
            'northing': ds.northing,
            'easting': ds.easting,
        }
    )

    return xr.merge([ds, dataset])


@click.command()
@click.argument('climatefile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('heatfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('firfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(climatefile, heatfile, firfile, configfile, outputfile):

    config = load_config(configfile)

    years = config['years']
    finfo = config['features']
    chunks = config['chunks']

    region_df = gpd.read_file(firfile)

    clim = xr.open_zarr(climatefile)
    heat = xr.open_zarr(heatfile)

    # Fix coordinate mis-match issue due to rounding
    clim = clim.assign_coords(
        easting=np.round(clim.easting.values, 4),
        northing=np.round(clim.northing.values, 4),
    )
    heat = heat.assign_coords(
        easting=np.round(heat.easting.values, 4),
        northing=np.round(heat.northing.values, 4),
    )

    heat = heat.set_coords('spatial_ref').drop_vars('quantile', errors='ignore')
    combined = xr.merge([clim, heat])

    combined = append_ne(combined)

    dataset = apply_mask(combined, region_df)

    combined = xr.concat(
        [
            to_samples(dataset, year, finfo)
            for year in tqdm(years, 'Yearly Features')
        ],
        dim='sample', coords='all', compat='identical'
    )

    write_job = combined.chunk(chunks).to_zarr(
        outputfile, mode='w', compute=False, consolidated=True
    )

    print(f'Writing data...')
    with ProgressBar():
        write_job.persist()
    print('Done')


if __name__ == '__main__':
    main()
