#!/usr/bin/env python
import os
import json
import click
import numpy as np
import xarray as xr
import rasterio as rio
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path
from dask.diagnostics import ProgressBar


def apply_masks(dataset, df):

    dataset = dataset.rename({ 'easting': 'x', 'northing': 'y' })
    dataset = dataset.rio.clip(df.geometry.values, df.crs)
    dataset = dataset.rename({ 'x': 'easting', 'y': 'northing' })
    return dataset


@click.command()
@click.argument('projectionfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('shapefile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(projectionfile, shapefile, outputfile):

    shp = gpd.read_file(shapefile)

    ds = xr.open_zarr(projectionfile)

    sf = (1.0 - (ds['mortality_projection'] / 100.))
    sf = np.maximum(0.0, np.minimum(1.0, sf))
    habitat = sf.cumprod(dim='year')

    #habitat.rio.write_crs(pstr, inplace=True)

    habitat = apply_masks(habitat, shp)

    habitat = habitat.chunk({
        'easting': 512,
        'northing': 512,
        'year': -1,
    })

    write_job = habitat.to_zarr(
        outputfile, mode='w', compute=False, consolidated=True
    )

    with ProgressBar():
        write_job.persist()


if __name__ == '__main__':
    main()
