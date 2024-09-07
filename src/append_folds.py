#!/usr/bin/env python
import json
import click
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
from dask.diagnostics import ProgressBar


def make_folds(ds, grid_size, shuffle=False):
    cols = len(ds.easting.values)
    rows = len(ds.northing.values)
    cc = np.arange(cols) // grid_size
    rr = (cc.max() + 1) * (np.arange(rows) // grid_size)

    folds = np.zeros((cols, rows), dtype=int) + cc[..., np.newaxis] + rr
    idx = np.arange(cols * rows, dtype=int).reshape((rows, cols)).T

    eastings = np.zeros((rows, cols), dtype=float) + ds.easting.values
    northings = (np.zeros((rows, cols), dtype=float).T + ds.northing.values).T

    if shuffle:
        np.random.seed(0)
        folds_tmp = folds.ravel()
        np.random.shuffle(folds_tmp)
        folds = folds_tmp.reshape((cols, rows))

    dataset = xr.Dataset(
        data_vars={
            'fold': (
                ['easting', 'northing'],
                folds,
                {
                    'long_name': 'fold',
                }
            ),
            'id': (
                ['easting', 'northing'],
                idx,
                {
                    'long_name': 'cell identifier',
                }
            ),
            'northings': (
                ['easting', 'northing'],
                northings.T,
                {
                    'long_name': 'northing coordinate',
                }
            ),
            'eastings': (
                ['easting', 'northing'],
                eastings.T,
                {
                    'long_name': 'easting coordinate',
                }
            ),
        },
        coords={
            'easting': ds.easting,
            'northing': ds.northing,
        }
    )

    return dataset


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, configfile, outputfile):

    # Load config
    with open(configfile, 'r') as f:
        config = json.load(f)
    grid_size = config['grid_size']
    chunks = config['chunks']
    shuffle = config['shuffle']

    ds = xr.open_zarr(inputfile)

    folds = make_folds(ds, grid_size, shuffle)

    new_dataset = xr.merge([ds, folds], join='exact')

    new_dataset = new_dataset.chunk(chunks)

    write_job = new_dataset.to_zarr(
        outputfile, mode='w', compute=False, consolidated=True
    )

    with ProgressBar():
        write_job.persist()


if __name__ == '__main__':
    main()
