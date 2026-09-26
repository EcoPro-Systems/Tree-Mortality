#!/usr/bin/env python
"""
Build HLS composites matched to each pixel's ADS survey (overflight) date.

For survey year Y, each pixel's flight day of year D comes from the
`flight_doy` layer written by ads_labels_aoi.py (DMSM feature CREATED_DATE, or
the surveyed-area START/END dates). The composite is the per-pixel median of
clear HLS observations with |DOY - D| <= window, taken in:

  prev   year Y-1   (same time of year, before the survey year)
  at     year Y     (around the overflight)
  next   year Y+1   (same time of year, after)
  base   2013       (pre-drought baseline, same time of year)

Comparing at vs prev at the same time of year removes seasonal (phenology)
differences that a fixed Jul-Sep window mixes in when flights range from
early July to late October. Pixels without a flight date are NaN.

Output is <outputdir>/<aoi>_flight_w<window>.nc with dims
(year, lag, y, x) for each band/index, plus n_clear.
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
from hls_annual_composites import read_obs, compute_indices, ROLES, INDICES

LAGS = {'prev': -1, 'at': 0, 'next': 1, 'base': None}
BASE_YEAR = 2013
ROW_CHUNK = 125


def list_observations(aoidir, sensors=('L30', 'S30')):
    """year -> list of (doy, [paths]) with overlapping tiles grouped by
    sensor/date"""
    groups = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(str(aoidir / '*.tif'))):
        _, sensor, _, ydoy = os.path.basename(f).split('.')[:4]
        if sensor not in sensors:
            continue
        date = datetime.strptime(ydoy[:7], '%Y%j')
        groups[date.year][(sensor, date.timetuple().tm_yday)].append(f)
    return {
        year: [(doy, paths) for (_, doy), paths in sorted(obs.items())]
        for year, obs in groups.items()
    }


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
@click.option('--labels', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_labels')
@click.option('-a', '--aoi', 'aois', multiple=True)
@click.option('-w', '--window', default=20, show_default=True,
              help='Half-width (days) around the flight date')
def main(configfile, hlsdir, outputdir, labels, aois, window):

    config = load_config(configfile)
    aois = aois or list(config['aois'])
    os.makedirs(outputdir, exist_ok=True)
    variables = ROLES + INDICES

    for name in aois:
        obs = list_observations(hlsdir / name)
        lab = xr.open_dataset(labels / f'{name}.nc')
        with rasterio.open(glob.glob(str(hlsdir / name / '*.tif'))[0]) as ds:
            shape = ds.shape

        survey_years = [
            int(y) for y in lab.year.values
            if (lab.flight_doy.sel(year=y) >= 0).any() and y > BASE_YEAR
            and y in obs
        ]
        shp = (len(survey_years), len(LAGS)) + shape
        out = {v: np.full(shp, np.nan, dtype=np.float32) for v in variables}
        n_clear = np.zeros(shp, dtype=np.int16)

        for yi, year in enumerate(survey_years):
            fdoy = lab.flight_doy.sel(year=year).values.astype(np.float32)
            fdoy[fdoy < 0] = np.nan
            for li, (lag, off) in enumerate(LAGS.items()):
                src_year = BASE_YEAR if off is None else year + off
                if src_year not in obs:
                    continue
                year_obs = obs[src_year]
                for r0 in tqdm(range(0, shape[0], ROW_CHUNK), leave=False,
                               desc=f'{name} {year} {lag}'):
                    h = min(ROW_CHUNK, shape[0] - r0)
                    window_ = Window(0, r0, shape[1], h)
                    target = fdoy[r0:r0 + h]
                    stack = {v: [] for v in variables}
                    for doy, paths in year_obs:
                        near = np.abs(target - doy) <= window
                        if not near.any():
                            continue
                        reads = [read_obs(p, window_) for p in paths]
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            refl = {
                                role: np.nanmean([r[role] for r in reads],
                                                 axis=0)
                                for role in ROLES
                            }
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
                            out[v][yi, li, rows] = np.nanmedian(
                                np.stack(stack[v]), axis=0)
                    n_clear[yi, li, rows] = np.isfinite(
                        np.stack(stack['nir'])).sum(axis=0)
            valid = np.isfinite(out['ndmi'][yi])
            click.echo(f'[{name}] {year}: valid fraction by lag '
                       f'{dict(zip(LAGS, valid.mean(axis=(1, 2)).round(3)))}')

        ds = xr.Dataset(
            {v: (('year', 'lag', 'y', 'x'), out[v]) for v in variables},
            coords={'year': survey_years, 'lag': list(LAGS),
                    'y': lab.y.values, 'x': lab.x.values},
            attrs=dict(lab.attrs, window_days=window),
        )
        ds['n_clear'] = (('year', 'lag', 'y', 'x'), n_clear)
        enc = {v: {'zlib': True} for v in ds.data_vars}
        outfile = outputdir / f'{name}_flight_w{window}.nc'
        ds.to_netcdf(outfile, encoding=enc)
        click.echo(f'[{name}] wrote {outfile}')


if __name__ == '__main__':
    main()
