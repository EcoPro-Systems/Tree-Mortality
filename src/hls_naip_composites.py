#!/usr/bin/env python
"""
Build annual HLS composites matched to the acquisition date of a reference
NAIP year, for comparison with NAIP-derived dead-tree maps (e.g., Cheng et
al. 2024, from 2020 NAIP).

Each AOI pixel gets a target day of year from the NAIP quarter-quad that
covers it (footprints and dates from the STAC manifest written by
fetch_naip_aoi.py). For every year in the range, the composite is the
per-pixel median of clear observations with |DOY - target| <= window, so
every year is sampled at the same time of year as the reference imagery.
Observations can be restricted to Landsat (L30) or Sentinel-2 (S30).

Output is <outputdir>/<aoi>_naip<year>_<sensors>_w<window>.nc with dims
(year, y, x) for each band/index, plus n_clear and the target_doy map.
"""
import os
import json
import glob
import warnings
import click
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import shape as to_shape
from rasterio.features import rasterize
from rasterio.windows import Window

from util import load_config
from fetch_hls_aoi import aoi_grid
from hls_annual_composites import read_obs, compute_indices, ROLES, INDICES
from hls_flight_composites import list_observations

ROW_CHUNK = 125


def naip_target_doy(manifest, naip_year, transform, shape, crs):
    """Rasterize NAIP quarter-quad acquisition DOY onto the AOI grid"""
    with open(manifest) as f:
        items = json.load(f)['items']
    items = [i for i in items if int(i['properties']['naip:year']) == naip_year]
    gdf = gpd.GeoDataFrame(
        {'doy': [pd.Timestamp(i['properties']['datetime']).dayofyear
                 for i in items]},
        geometry=[to_shape(i['geometry']) for i in items], crs='EPSG:4326',
    ).to_crs(crs)
    doy = rasterize(zip(gdf.geometry, gdf.doy.astype('int16')),
                    out_shape=shape, transform=transform, fill=-1,
                    dtype='int16')
    return doy, sorted(gdf.doy.unique().tolist())


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('hlsdir', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('--naip-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/naip')
@click.option('--naip-year', default=2020, show_default=True)
@click.option('-a', '--aoi', 'aois', multiple=True)
@click.option('-s', '--sensor', 'sensors', multiple=True,
              default=['L30'], show_default=True,
              type=click.Choice(['L30', 'S30']))
@click.option('-w', '--window', default=25, show_default=True,
              help='Half-width (days) around the NAIP acquisition date')
def main(configfile, hlsdir, outputdir, naip_dir, naip_year, aois, sensors,
         window):

    config = load_config(configfile)
    aois = aois or list(config['aois'])
    os.makedirs(outputdir, exist_ok=True)
    variables = ROLES + INDICES
    first, last = config['years']
    years = list(range(first, last + 1))

    for name in aois:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        crs = f'EPSG:{aoi["epsg"]}'
        target, doys = naip_target_doy(naip_dir / name / '_items.json',
                                       naip_year, transform, shape, crs)
        click.echo(f'[{name}] NAIP {naip_year} DOYs {doys}; '
                   f'covered {np.mean(target >= 0):.3f}')
        target = target.astype(np.float32)
        target[target < 0] = np.nan

        obs = list_observations(hlsdir / name, sensors)
        shp = (len(years),) + shape
        out = {v: np.full(shp, np.nan, dtype=np.float32) for v in variables}
        n_clear = np.zeros(shp, dtype=np.int16)

        for yi, year in enumerate(years):
            for r0 in tqdm(range(0, shape[0], ROW_CHUNK), leave=False,
                           desc=f'{name} {year}'):
                h = min(ROW_CHUNK, shape[0] - r0)
                win = Window(0, r0, shape[1], h)
                tgt = target[r0:r0 + h]
                stack = {v: [] for v in variables}
                for doy, paths in obs.get(year, []):
                    near = np.abs(tgt - doy) <= window
                    if not near.any():
                        continue
                    reads = [read_obs(p, win) for p in paths]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        refl = {r: np.nanmean([x[r] for x in reads], axis=0)
                                for r in ROLES}
                    idx = compute_indices(refl)
                    for v in ROLES:
                        stack[v].append(np.where(near, refl[v], np.nan))
                    for v in INDICES:
                        stack[v].append(np.where(near, idx[v], np.nan))
                if not stack['nir']:
                    continue
                rows = slice(r0, r0 + h)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for v in variables:
                        out[v][yi, rows] = np.nanmedian(
                            np.stack(stack[v]), axis=0)
                n_clear[yi, rows] = np.isfinite(
                    np.stack(stack['nir'])).sum(axis=0)
            click.echo(f'[{name}] {year}: valid '
                       f'{np.isfinite(out["ndmi"][yi]).mean():.3f}, median '
                       f'n_clear {np.median(n_clear[yi][n_clear[yi] > 0]) if (n_clear[yi] > 0).any() else 0}')

        x0, y1, res = transform.c, transform.f, transform.a
        ds = xr.Dataset(
            {v: (('year', 'y', 'x'), out[v]) for v in variables},
            coords={'year': years,
                    'y': y1 - res * (np.arange(shape[0]) + 0.5),
                    'x': x0 + res * (np.arange(shape[1]) + 0.5)},
            attrs={'crs': crs, 'transform': list(transform)[:6],
                   'naip_year': naip_year, 'sensors': ','.join(sensors),
                   'window_days': window},
        )
        ds['n_clear'] = (('year', 'y', 'x'), n_clear)
        ds['target_doy'] = (('y', 'x'), np.nan_to_num(target, nan=-1)
                            .astype(np.int16))
        enc = {v: {'zlib': True} for v in ds.data_vars}
        tag = ''.join(sensors)
        outfile = outputdir / f'{name}_naip{naip_year}_{tag}_w{window}.nc'
        ds.to_netcdf(outfile, encoding=enc)
        click.echo(f'[{name}] wrote {outfile}')


if __name__ == '__main__':
    main()
