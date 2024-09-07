#!/usr/bin/env python
import json
import click
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
from dask.diagnostics import ProgressBar

CRS = 4326


@click.command()
@click.argument('bcmfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('indexfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(bcmfile, indexfile, outputfile):

    ds = xr.open_zarr(bcmfile)
    idx = xr.open_zarr(indexfile)
    idx = idx.rio.write_crs(CRS)
    idx = idx.rename({
        'latitude': 'y',
        'longitude': 'x',
    })
    ds = ds.rename({
        'easting': 'x',
        'northing': 'y',
    })

    regridded = idx.rio.reproject_match(ds)
    regridded = regridded.rename({
        'x': 'easting',
        'y': 'northing',
    })

    write_job = regridded.to_zarr(
        outputfile, mode='w', compute=False, consolidated=True
    )

    with ProgressBar():
        write_job.persist()


if __name__ == '__main__':
    main()
