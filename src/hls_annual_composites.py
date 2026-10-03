#!/usr/bin/env python
"""
Build per-year, per-pixel median composites of HLS reflectance and spectral
indices for each AOI from the per-granule windows written by fetch_hls_aoi.py.

Observations are screened with Fmask (fill, cloud, cloud-adjacent, shadow,
snow/ice, water, high aerosol; same bits as ~/Documents/herd/src/hls_to_daily.py).
The same sensor/date seen from overlapping MGRS tiles is averaged into a
single observation before the median so overlap areas are not double-counted.

Output is <outputdir>/<aoi>.nc with dims (year, y, x) holding the median of
each band and index over the configured day-of-year window, plus n_clear.
"""
import os
import glob
import warnings
import click
import numpy as np
import xarray as xr
import rasterio
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from rasterio.windows import Window

from util import load_config

ROLES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
REF_FILL = -9999
REF_SCALE = 0.0001
QA_FILL = 255
QA_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
QA_AEROSOL_HIGH = (1 << 6) | (1 << 7)
INDICES = ['ndvi', 'ndmi', 'nbr', 'rgi', 'evi']
ROW_CHUNK = 125


def compute_indices(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        return {
            'ndvi': (r['nir'] - r['red']) / (r['nir'] + r['red']),
            'ndmi': (r['nir'] - r['swir1']) / (r['nir'] + r['swir1']),
            'nbr': (r['nir'] - r['swir2']) / (r['nir'] + r['swir2']),
            # Red-green index: rises as needles turn red after mortality
            'rgi': r['red'] / r['green'],
            'evi': 2.5 * (r['nir'] - r['red']) / (
                r['nir'] + 6 * r['red'] - 7.5 * r['blue'] + 1),
        }


def read_obs(path, window):
    with rasterio.open(path) as ds:
        data = ds.read(window=window)
    qa = data[-1].astype(np.int32)
    bad = (qa == QA_FILL) | ((qa & QA_BAD) != 0)
    bad |= (qa & QA_AEROSOL_HIGH) == QA_AEROSOL_HIGH
    refl = {}
    for i, role in enumerate(ROLES):
        raw = data[i]
        arr = np.where((raw == REF_FILL) | bad, np.nan,
                       raw * REF_SCALE).astype(np.float32)
        refl[role] = arr
    return refl


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
@click.option('-a', '--aoi', 'aois', multiple=True)
@click.option('--doy-start', default=182, show_default=True)  # ~Jul 1
@click.option('--doy-end', default=273, show_default=True)  # ~Sep 30
def main(configfile, hlsdir, outputdir, aois, doy_start, doy_end):

    config = load_config(configfile)
    aois = aois or list(config['aois'])
    os.makedirs(outputdir, exist_ok=True)
    variables = ROLES + INDICES

    for name in aois:
        files = sorted(glob.glob(str(hlsdir / name / '*.tif')))
        # Group granules by (year, sensor, date) to merge overlapping tiles
        groups = defaultdict(lambda: defaultdict(list))
        for f in files:
            _, sensor, _, ydoy = os.path.basename(f).split('.')[:4]
            date = datetime.strptime(ydoy[:7], '%Y%j')
            doy = date.timetuple().tm_yday
            if doy_start <= doy <= doy_end:
                groups[date.year][(sensor, ydoy[:7])].append(f)
        years = sorted(groups)

        with rasterio.open(files[0]) as ds:
            shape = ds.shape
            transform = ds.transform
            crs = ds.crs.to_string()

        out = {v: np.full((len(years),) + shape, np.nan, dtype=np.float32)
               for v in variables}
        n_clear = np.zeros((len(years),) + shape, dtype=np.int16)
        n_obs = []

        for yi, year in enumerate(years):
            obs = groups[year]
            n_obs.append(len(obs))
            for r0 in tqdm(range(0, shape[0], ROW_CHUNK),
                           desc=f'{name} {year} ({len(obs)} obs)'):
                window = Window(0, r0, shape[1], min(ROW_CHUNK, shape[0] - r0))
                stack = {v: [] for v in variables}
                for key, paths in obs.items():
                    reads = [read_obs(p, window) for p in paths]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        refl = {
                            role: np.nanmean([r[role] for r in reads], axis=0)
                            for role in ROLES
                        }
                    idx = compute_indices(refl)
                    for v in ROLES:
                        stack[v].append(refl[v])
                    for v in INDICES:
                        stack[v].append(idx[v])
                rows = slice(r0, r0 + window.height)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    for v in variables:
                        out[v][yi, rows] = np.nanmedian(
                            np.stack(stack[v]), axis=0)
                n_clear[yi, rows] = np.isfinite(
                    np.stack(stack['nir'])).sum(axis=0)

        xs = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
        ys = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
        ds = xr.Dataset(
            {v: (('year', 'y', 'x'), out[v]) for v in variables},
            coords={'year': years, 'y': ys, 'x': xs},
            attrs={'crs': crs, 'transform': list(transform)[:6],
                   'doy_start': doy_start, 'doy_end': doy_end},
        )
        ds['n_clear'] = (('year', 'y', 'x'), n_clear)
        ds['n_obs'] = (('year',), np.array(n_obs, dtype=np.int16))
        enc = {v: {'zlib': True} for v in ds.data_vars}
        outfile = outputdir / f'{name}_doy{doy_start}-{doy_end}.nc'
        ds.to_netcdf(outfile, encoding=enc)
        click.echo(f'[{name}] wrote {outfile}; obs per year: '
                   f'{dict(zip(years, n_obs))}')


if __name__ == '__main__':
    main()
