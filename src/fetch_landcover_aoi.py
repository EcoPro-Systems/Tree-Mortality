#!/usr/bin/env python
"""
Fetch NLCD land cover and tree canopy cover for each HLS AOI from the MRLC
WCS, reprojected onto the AOI's 30 m UTM grid (see fetch_hls_aoi.py). Output:

    <outputdir>/<aoi>_landcover.tif  (band 1: NLCD class, band 2: TCC %)

The defaults use 2013 products, i.e., before the 2014-2017 drought die-off,
so the forest mask is not itself affected by the mortality being studied.
"""
import os
import click
import numpy as np
import rasterio
import requests
from pathlib import Path
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling, transform_bounds

from util import load_config
from fetch_hls_aoi import aoi_grid

WCS_URL = 'https://www.mrlc.gov/geoserver/mrlc_download/wcs'
MARGIN = 1000  # m of padding in EPSG:5070 around the AOI


def get_coverage(coverage_id, bounds):
    x0, y0, x1, y1 = bounds
    params = [
        ('service', 'WCS'), ('version', '2.0.1'), ('request', 'GetCoverage'),
        ('coverageId', coverage_id),
        ('subset', f'X({x0 - MARGIN:.0f},{x1 + MARGIN:.0f})'),
        ('subset', f'Y({y0 - MARGIN:.0f},{y1 + MARGIN:.0f})'),
        ('format', 'image/geotiff'),
    ]
    resp = requests.get(WCS_URL, params=params, timeout=300)
    resp.raise_for_status()
    with MemoryFile(resp.content) as mem:
        with mem.open() as ds:
            return ds.read(1), ds.transform, ds.crs


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('--landcover', default='mrlc_download__NLCD_2013_Land_Cover_L48')
@click.option('--canopy', default='mrlc_download__nlcd_tcc_conus_2013_v2021-4')
def main(configfile, outputdir, landcover, canopy):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)

    for name, aoi in config['aois'].items():
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        crs = f'EPSG:{aoi["epsg"]}'
        x0, y1 = transform.c, transform.f
        bounds = (x0, y1 - shape[0] * config['resolution'],
                  x0 + shape[1] * config['resolution'], y1)
        bounds_5070 = transform_bounds(crs, 'EPSG:5070', *bounds)

        out = np.zeros((2,) + shape, dtype=np.uint8)
        for i, (cov, resampling) in enumerate([
            (landcover, Resampling.nearest),
            (canopy, Resampling.bilinear),
        ]):
            src, src_tr, src_crs = get_coverage(cov, bounds_5070)
            reproject(src, out[i], src_transform=src_tr, src_crs=src_crs,
                      dst_transform=transform, dst_crs=crs,
                      resampling=resampling)

        outfile = outputdir / f'{name}_landcover.tif'
        with rasterio.open(
            outfile, 'w', driver='GTiff', height=shape[0], width=shape[1],
            count=2, dtype='uint8', crs=crs, transform=transform,
            compress='deflate',
        ) as dst:
            dst.write(out)
            dst.set_band_description(1, 'nlcd_class')
            dst.set_band_description(2, 'tree_canopy_pct')
            dst.update_tags(landcover=landcover, canopy=canopy)

        classes, counts = np.unique(out[0], return_counts=True)
        frac = dict(zip(classes.tolist(), np.round(counts / counts.sum(), 3)))
        click.echo(f'[{name}] class fractions: {frac}; '
                   f'mean TCC {out[1].mean():.1f}%')


if __name__ == '__main__':
    main()
