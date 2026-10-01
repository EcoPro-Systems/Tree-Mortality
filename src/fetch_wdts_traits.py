#!/usr/bin/env python
"""
Fetch WDTS AVIRIS-Classic foliar trait mosaics (Zheng et al. 2025, ORNL DAAC
2403, doi:10.3334/ORNLDAAC/2403) onto the HLS AOI grids.

The mosaics are 30 m COGs on a 30 m lattice that coincides with the HLS AOI
grids when the UTM zone matches, so AOI windows are read directly (no
resampling). The Tahoe box mosaics are in UTM 11 while the stanislaus AOI is
in UTM 10; those are warped with nearest neighbour (<= 15 m shift). One file
per flight box, date and trait, 12 bands:

  1 mean, 2 sd (200 PLSR permutations)
  3 QC_all, 4 QC_fc (green vegetation fraction >= 0.5), 5 QC_fs (snow <= 0.3),
  6 QC_uncertainty (sd/mean < 0.3), 7 QC_shadow, 8 QC_anomaly,
  9 QC_pixanom, 10 QC_range (within NEON trait range), 11 QC_edge,
  12 flight_id

The mosaics are aggregated from 15 m, so the QC bands are the fraction of
15 m subpixels that pass (0, 0.25, 0.33, ...), and the trait mean is nodata
wherever no subpixel passes, even inside the flight footprint (flight_id >
0). About 30% of the forested AOIs is masked this way, mostly QC_fc ~ 0, i.e.
green vegetation cover < 0.5, which includes dead canopy. The QC fractions
are therefore kept as features in their own right.

For each AOI one early-summer acquisition per year is used (DATES; the
"_v2" Yosemite files are the radiometrically stabilized versions the user
guide recommends for between-year analysis). Output is <aoi>_traits.nc with
dims (year, y, x):

  <trait>_mean, <trait>_sd   float32, NaN where masked or outside footprint
  qc_all, qc_fc, ... qc_edge uint8 percent of 15 m subpixels passing (255
                             outside the footprint), from the LMA file
  flight_id                  int16 (-1 outside the footprint)

--date BOX:YEAR=DATE swaps the acquisition for one year (e.g.
yosemite:2013=20130612, the non-_v2 version of the 2013 primary date), and
--suffix writes <aoi>_traits_<suffix>.nc so the default file is kept.
"""
import os
import re
import time
import click
import numpy as np
import requests
import rasterio
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

from util import load_config
from fetch_hls_aoi import (
    CMR_URL, RETRIES, BACKOFF, gdal_env, aoi_grid, aoi_lonlat_bbox,
    get_with_retry,
)

SHORT_NAME = 'WDTS_AVIRIS-C_foliar_traits_2403'
TRAITS = ['LMA', 'Nitrogen', 'Chlorophylls', 'Cellulose', 'Lignin', 'Fiber',
          'Sugar', 'Starch', 'NSC', 'Calcium', 'Potassium', 'Phosphorus',
          'Sulfur', 'Phenolics']
# Early-summer acquisition per year for each flight box ('_v2' preferred)
DATES = {
    'yosemite': {2013: '20130612_v2', 2014: '20140603', 2015: '20150601_v2',
                 2016: '20160621', 2017: '20170607', 2018: '20180622'},
    'tahoe': {2013: '20130604', 2014: '20140602', 2015: '20150608',
              2016: '20160609', 2017: '20170620', 2018: '20180621'},
}
BOX = {'sierra_nf': 'yosemite', 'neon_soap_teak': 'yosemite',
       'seki': 'yosemite', 'stanislaus': 'tahoe'}
FLAG_BANDS = {'qc_all': 3, 'qc_fc': 4, 'qc_fs': 5, 'qc_uncertainty': 6,
              'qc_shadow': 7, 'qc_anomaly': 8, 'qc_pixanom': 9,
              'qc_range': 10, 'qc_edge': 11}
NODATA = -9999
NAME_RE = re.compile(
    r'traits\.(\w+?)_(\d{8}(?:_v2)?)_(\w+?)_30m(?:_v2)?\.tif$')


def search(session, bbox):
    """{(box, date, trait): url} for mosaics intersecting bbox"""
    params = {'short_name': SHORT_NAME, 'page_size': 2000,
              'bounding_box': ','.join(f'{v:.5f}' for v in bbox)}
    out = {}
    resp = get_with_retry(session, CMR_URL, params)
    for item in resp.json()['items']:
        m = NAME_RE.search(item['umm']['GranuleUR'])
        if m is None:
            continue
        url = [r['URL'] for r in item['umm']['RelatedUrls']
               if r.get('Type') == 'GET DATA'][0]
        out[m.groups()] = url
    return out


def read_bands(url, transform, shape, env, bands, epsg):
    """Read bands of the AOI window (grids coincide); NODATA outside the
    file extent. Mosaics in another UTM zone than the AOI (the Tahoe box is
    in zone 11, the stanislaus AOI in zone 10) are warped onto the AOI grid
    with nearest-neighbour resampling."""
    x0, y1 = transform.c, transform.f
    x1 = x0 + shape[1] * transform.a
    y0 = y1 + shape[0] * transform.e
    for attempt in range(1, RETRIES + 1):
        try:
            with rasterio.Env(**env):
                with rasterio.open('/vsicurl/' + url) as ds:
                    if ds.crs.to_epsg() != epsg:
                        with WarpedVRT(ds, crs=f'EPSG:{epsg}',
                                       transform=transform,
                                       width=shape[1], height=shape[0],
                                       resampling=Resampling.nearest,
                                       src_nodata=NODATA,
                                       nodata=NODATA) as vrt:
                            return vrt.read(bands).astype(np.float32)
                    assert ds.transform.a == transform.a
                    w = from_bounds(x0, y0, x1, y1, ds.transform)
                    r0, c0 = int(round(w.row_off)), int(round(w.col_off))
                    out = np.full((len(bands),) + shape, NODATA, np.float32)
                    rr0, cc0 = max(r0, 0), max(c0, 0)
                    rr1 = min(r0 + shape[0], ds.height)
                    cc1 = min(c0 + shape[1], ds.width)
                    if rr1 > rr0 and cc1 > cc0:
                        out[:, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = ds.read(
                            bands, window=Window(cc0, rr0, cc1 - cc0,
                                                 rr1 - rr0))
                    return out
        except rasterio.errors.RasterioIOError:
            if attempt == RETRIES:
                raise
            time.sleep(BACKOFF * attempt)


def fetch_aoi(name, aoi, config, found, outpath, env, jobs, dates,
              years=None):
    transform, shape = aoi_grid(aoi, config['size_m'], config['resolution'])
    box = BOX[name]
    years = sorted(years or dates)
    jobs_list = [(y, t) for y in years for t in TRAITS]
    for y, t in jobs_list:
        if (box, dates[y], t) not in found:
            raise click.ClickException(f'missing {box} {dates[y]} {t}')

    # Authenticate once, serially, before the parallel reads
    read_bands(found[(box, dates[years[0]], 'LMA')], transform,
               (4, 4), env, [1], aoi['epsg'])

    def run(job):
        y, t = job
        bands = [1, 2]
        if t == 'LMA':
            bands += list(FLAG_BANDS.values()) + [12]
        return job, read_bands(found[(box, dates[y], t)], transform,
                               shape, env, bands, aoi['epsg'])

    ny = len(years)
    data = {}
    for t in TRAITS:
        data[f'{t}_mean'] = np.full((ny,) + shape, np.nan, np.float32)
        data[f'{t}_sd'] = np.full((ny,) + shape, np.nan, np.float32)
    for k in FLAG_BANDS:
        data[k] = np.full((ny,) + shape, 255, np.uint8)
    data['flight_id'] = np.full((ny,) + shape, -1, np.int16)

    with ThreadPoolExecutor(jobs) as pool:
        for (y, t), a in pool.map(run, jobs_list):
            i = years.index(y)
            ok = a[0] != NODATA
            data[f'{t}_mean'][i][ok] = a[0][ok]
            data[f'{t}_sd'][i][ok] = a[1][ok]
            if t == 'LMA':
                fid = a[2 + len(FLAG_BANDS)]
                inside = fid > 0
                for j, k in enumerate(FLAG_BANDS):
                    pct = np.round(100 * a[2 + j]).astype(np.uint8)
                    data[k][i] = np.where(inside, pct, 255)
                data['flight_id'][i] = np.where(inside, fid, -1)
                click.echo(f'  [{name}] {y}: footprint {inside.mean():.2f}, '
                           f'trait valid {ok.mean():.2f}')

    xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    ds = xr.Dataset(
        {k: (('year', 'y', 'x'), v) for k, v in data.items()},
        coords={'year': years, 'y': ys, 'x': xs},
        attrs={'crs': f'EPSG:{aoi["epsg"]}', 'transform': list(transform)[:6],
               'flightbox': box,
               'dates': ','.join(dates[y] for y in years),
               'source': 'doi:10.3334/ORNLDAAC/2403'},
    )
    enc = {k: {'zlib': True, 'complevel': 4} for k in data}
    tmp = outpath.with_suffix('.tmp.nc')
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, outpath)


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
), default='/Volumes/Earth04/ecopro/wdts')
@click.option('-a', '--aoi', 'aois', multiple=True,
              default=['sierra_nf', 'neon_soap_teak'], show_default=True)
@click.option('-j', '--jobs', default=4, show_default=True)
@click.option('--list', 'list_only', is_flag=True,
              help='List available acquisitions per AOI and exit')
@click.option('--overwrite', is_flag=True)
@click.option('--date', 'date_swaps', multiple=True,
              help='Swap one acquisition, as BOX:YEAR=DATE')
@click.option('-y', '--year', 'years', multiple=True, type=int,
              help='Years to fetch (default: all in DATES)')
@click.option('--suffix', default='',
              help='Write <aoi>_traits_<suffix>.nc')
def main(configfile, outputdir, aois, jobs, list_only, overwrite,
         date_swaps, years, suffix):
    box_dates = {b: dict(v) for b, v in DATES.items()}
    for swap in date_swaps:
        m = re.fullmatch(r'(\w+):(\d{4})=(\d{8}(?:_v2)?)', swap)
        if m is None:
            raise click.BadParameter(f'{swap}: expected BOX:YEAR=DATE')
        box_dates[m.group(1)][int(m.group(2))] = m.group(3)
    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    env = gdal_env(str(outputdir / '.cookies.txt'))
    session = requests.Session()
    for name in aois:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        bbox = aoi_lonlat_bbox(transform, shape, aoi['epsg'])
        found = search(session, bbox)
        dates = sorted({(b, d) for b, d, _ in found})
        click.echo(f'[{name}] {len(found)} mosaics; box/dates: ' +
                   ', '.join(f'{b}_{d}' for b, d in dates))
        if list_only:
            continue
        if name not in BOX:
            click.echo(f'[{name}] no WDTS flight box configured; skipping')
            continue
        outpath = outputdir / (f'{name}_traits_{suffix}.nc' if suffix
                               else f'{name}_traits.nc')
        if outpath.exists() and not overwrite:
            click.echo(f'[{name}] {outpath} exists; skipping')
            continue
        fetch_aoi(name, aoi, config, found, outpath, env, jobs,
                  box_dates[BOX[name]], years)
        click.echo(f'[{name}] wrote {outpath}')


if __name__ == '__main__':
    main()
