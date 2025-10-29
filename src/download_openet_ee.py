#!/usr/bin/env python
import ee
import geemap
import click
import requests
import zipfile
import io
from pathlib import Path


@click.command()
#@click.argument('geomfile', type=click.Path(
#    path_type=Path, exists=True
#))
def main():

    # Initialize the Earth Engine module.
    ee.Initialize(project='ecopro-1')

    bbox = ee.Geometry.BBox(
        -121.54288583732415, 37.79280555325196,
        -119.22154524430786, 39.549058861605445,
    )

    # Load the dataset
    dataset = (
        ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0')
        .filterDate('2011-02-01', '2011-02-02')
        .select('et_ensemble_mad')
        .mosaic()
        .clip(bbox)
    )

    geemap.download_ee_image(
        dataset, filename='et_tile_01b.tif', scale=30, region=bbox,
    )
    exit()

    
    ## Rename bands with their date
    #def rename_band(img):
    #    return img.rename([img.date().format("YYYYMM")])

    #et_image = dataset.map(rename_band).toBands()
    #et_image = dataset.toBands()

    # Request a download URL from EE
    task_url = dataset.getDownloadURL({
        "scale": 270,       # GRIDMET native scale ~4 km
        "region": bbox,
        "format": "GEO_TIFF"
    })
    print(task_url)
    exit()

    # Download the file (EE gives a .zip of GeoTIFFs if many bands)
    response = requests.get(task_url)
    print(response)
    exit()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Extract everything locally
        z.extractall("gee_download")


if __name__ == '__main__':
    main()
