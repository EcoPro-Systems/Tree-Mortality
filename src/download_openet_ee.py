#!/usr/bin/env python
import os
import ee
import geemap
import click
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path


def month_ranges(start_year, end_year):
    # Generate a range of month starts
    dates = pd.date_range(f'{start_year}-01-01', f'{end_year}-12-31', freq='MS')
    
    # For each start, compute the end of the month
    ranges = []
    for start in dates:
        end = start + pd.offsets.MonthEnd(1)
        ranges.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    
    return ranges


@click.command()
@click.argument('outputdir', type=click.Path(
    path_type=Path, exists=True
))
def main(outputdir):

    # Initialize the Earth Engine module.
    ee.Initialize(project='ecopro-1')

    bbox = ee.Geometry.BBox(
        -121.54288583732415, 37.79280555325196,
        -119.22154524430786, 39.549058861605445,
    )

    ranges = month_ranges(2017, 2017)

    for start, end in ranges:
        print(start)

        # Load the dataset
        dataset = (
            ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0')
            .filterDate(start, end)
            .select('et_ensemble_mad')
            .mosaic()
            .clip(bbox)
        )

        ofile = os.path.join(outputdir, f'et_{start}.tif')
        geemap.download_ee_image(
            dataset, filename=ofile, scale=30, region=bbox,
        )


if __name__ == '__main__':
    main()
