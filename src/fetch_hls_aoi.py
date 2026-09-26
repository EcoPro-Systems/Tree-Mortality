#!/usr/bin/env python
"""
Fetch HLS (HLSL30/HLSS30 v2.0) reflectance for small AOIs via windowed reads
of the LP DAAC cloud-optimized GeoTIFFs, rather than downloading full 110 km
MGRS tiles. Each granule's AOI window is written to

    <outputdir>/<aoi>/<granule_ur>.tif

as a 7-band int16 GeoTIFF (blue, green, red, nir, swir1, swir2, fmask) on a
fixed per-AOI 30 m UTM grid. HLS tiles in the same UTM zone share a 30 m grid
aligned to multiples of 30 m, so AOI windows are read without resampling;
granules from tiles in a different UTM zone are skipped.

Earthdata credentials come from ~/.netrc. Authentication is done once on a
single request before the parallel reads start, so the workers share the
cookie jar rather than each logging in (URS locks the profile after too many
login attempts). CMR search and retry logic follow ~/Documents/herd/src/fetch_hls.py.

The CMR result is cached per AOI in <outputdir>/<aoi>/_manifest.json, and
completed granules are skipped, so the fetch can be interrupted and re-run.
"""
import os
import re
import json
import time
import click
import numpy as np
import rasterio
import requests
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import load_config

CMR_URL = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json'
GRANULE_RE = re.compile(r'HLS\.(L30|S30)\.T(\w{5})\.(\d{7})T(\d{6})\.')
ROLES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
REF_FILL = -9999
QA_FILL = 255
QA_CLOUD = 1 << 1
QA_ADJACENT = 1 << 2
QA_SHADOW = 1 << 3
QA_AEROSOL_HIGH = (1 << 6) | (1 << 7)
RETRIES = 5
BACKOFF = 3.0


def gdal_env(cookie_file):
    return dict(
        GDAL_HTTP_COOKIEFILE=cookie_file,
        GDAL_HTTP_COOKIEJAR=cookie_file,
        GDAL_HTTP_NETRC='YES',
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='TIF',
        GDAL_HTTP_MAX_RETRY='5',
        GDAL_HTTP_RETRY_DELAY='3',
        VSI_CACHE='FALSE',
    )


def aoi_grid(aoi, size_m, res):
    """Transform/shape of the AOI grid, snapped to the HLS 30 m UTM lattice

    An AOI may override the default square size with its own `size_m`,
    either a scalar or [width, height] in meters.
    """
    size = aoi.get('size_m', size_m)
    w, h = (size, size) if np.isscalar(size) else size
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{aoi["epsg"]}',
                              always_xy=True)
    cx, cy = tr.transform(*aoi['center'])
    x0 = res * np.round((cx - w / 2) / res)
    y1 = res * np.round((cy + h / 2) / res)
    return from_origin(x0, y1, res, res), (int(h // res), int(w // res))


def aoi_lonlat_bbox(transform, shape, epsg):
    tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    x0, y1 = transform.c, transform.f
    x1 = x0 + shape[1] * transform.a
    y0 = y1 + shape[0] * transform.e
    lons, lats = tr.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return min(lons), min(lats), max(lons), max(lats)


def get_with_retry(session, url, params=None, headers=None):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(url, params=params, headers=headers,
                               timeout=(30, 300))
            if resp.status_code >= 500 or resp.status_code == 429:
                last = f'HTTP {resp.status_code}'
            else:
                resp.raise_for_status()
                return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = repr(exc)
        time.sleep(BACKOFF * attempt)
    raise click.ClickException(f'giving up on {url}: {last}')


def search_granules(session, short_name, bbox, years, doy_range):
    """Page CMR for granules intersecting bbox within each year's DOY window"""
    out = []
    for year in range(years[0], years[1] + 1):
        start = datetime.strptime(f'{year}{doy_range[0]:03d}', '%Y%j')
        end = datetime.strptime(f'{year}{doy_range[1]:03d}', '%Y%j')
        params = {
            'short_name': short_name,
            'bounding_box': ','.join(f'{v:.5f}' for v in bbox),
            'temporal': f'{start:%Y-%m-%d}T00:00:00Z,{end:%Y-%m-%d}T23:59:59Z',
            'page_size': 2000,
        }
        search_after = None
        while True:
            headers = (
                {'CMR-Search-After': search_after} if search_after else {}
            )
            resp = get_with_retry(session, CMR_URL, params, headers)
            items = resp.json().get('items', [])
            for item in items:
                umm = item['umm']
                m = GRANULE_RE.search(umm['GranuleUR'])
                if m is None:
                    continue
                cloud = None
                for attr in umm.get('AdditionalAttributes', []):
                    if attr['Name'] == 'CLOUD_COVERAGE':
                        cloud = float(attr['Values'][0])
                urls = {}
                for rel in umm.get('RelatedUrls', []):
                    url = rel.get('URL', '')
                    if rel.get('Type') == 'GET DATA' and url.endswith('.tif'):
                        urls[url.rsplit('.', 2)[-2]] = url
                date = datetime.strptime(m.group(3), '%Y%j')
                out.append({
                    'ur': umm['GranuleUR'], 'short_name': short_name,
                    'tile': m.group(2), 'date': f'{date:%Y-%m-%d}',
                    'cloud': cloud, 'urls': urls,
                })
            search_after = resp.headers.get('CMR-Search-After')
            if not search_after or not items:
                break
    return out


def read_window(url, transform, shape, env):
    """Read the AOI window of one COG band; fill outside the tile footprint

    LP DAAC intermittently answers with a non-TIFF body under load (GDAL then
    reports "not recognized as being in a supported file format"), so reads
    are retried with backoff.
    """
    x0, y1 = transform.c, transform.f
    x1 = x0 + shape[1] * transform.a
    y0 = y1 + shape[0] * transform.e
    for attempt in range(1, RETRIES + 1):
        try:
            with rasterio.Env(**env):
                with rasterio.open('/vsicurl/' + url) as ds:
                    w = from_bounds(x0, y0, x1, y1, ds.transform)
                    w = w.round_offsets().round_lengths()
                    return ds.read(
                        1, window=w, boundless=True, out_shape=shape,
                        fill_value=(ds.nodata if ds.nodata is not None
                                    else REF_FILL),
                    )
        except rasterio.errors.RasterioIOError:
            if attempt == RETRIES:
                raise
            time.sleep(BACKOFF * attempt)


def clear_fraction(qa):
    bad = (qa == QA_FILL)
    bad |= (qa & (QA_CLOUD | QA_ADJACENT | QA_SHADOW)) != 0
    bad |= (qa & QA_AEROSOL_HIGH) == QA_AEROSOL_HIGH
    return 1.0 - bad.mean()


def fetch_granule(g, bands, transform, shape, epsg, outfile, env,
                  min_clear):
    """Returns (status, clear_frac); status in {'ok', 'cloudy'}"""
    qa = read_window(g['urls']['Fmask'], transform, shape, env)
    cf = clear_fraction(qa)
    if cf < min_clear:
        # Record the skip so re-runs don't re-read this granule's Fmask
        Path(str(outfile) + '.cloudy').touch()
        return 'cloudy', cf
    data = np.empty((len(ROLES) + 1,) + shape, dtype=np.int16)
    for i, role in enumerate(ROLES):
        data[i] = read_window(g['urls'][bands[role]], transform, shape, env)
    data[-1] = qa.astype(np.int16)
    tmp = str(outfile) + '.part'
    with rasterio.open(
        tmp, 'w', driver='GTiff', height=shape[0], width=shape[1],
        count=data.shape[0], dtype='int16', crs=f'EPSG:{epsg}',
        transform=transform, nodata=REF_FILL, compress='deflate',
        predictor=2, tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(data)
        for i, name in enumerate(ROLES + ['fmask']):
            dst.set_band_description(i + 1, name)
        dst.update_tags(granule=g['ur'], short_name=g['short_name'],
                        tile=g['tile'], date=g['date'],
                        tile_cloud=str(g['cloud']), aoi_clear=f'{cf:.4f}')
    os.replace(tmp, outfile)
    return 'ok', cf


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('-a', '--aoi', 'aois', multiple=True,
              help='AOI name(s) to fetch (default: all)')
@click.option('-j', '--jobs', default=6, show_default=True,
              help='Parallel granule reads (LP DAAC throttles above ~8)')
@click.option('--refresh-manifest', is_flag=True)
@click.option('--dry-run', is_flag=True, help='Search and report counts only')
def main(configfile, outputdir, aois, jobs, refresh_manifest, dry_run):

    config = load_config(configfile)
    aois = aois or list(config['aois'])
    cookie_file = str((outputdir / '.cookies.txt').absolute())
    os.makedirs(outputdir, exist_ok=True)
    env = gdal_env(cookie_file)
    session = requests.Session()

    for name in aois:
        aoi = config['aois'][name]
        epsg = aoi['epsg']
        zone = f'{epsg % 100:02d}'
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        bbox = aoi_lonlat_bbox(transform, shape, epsg)
        aoidir = outputdir / name
        os.makedirs(aoidir, exist_ok=True)

        manifest = aoidir / '_manifest.json'
        if manifest.exists() and not refresh_manifest:
            with open(manifest) as f:
                granules = json.load(f)['granules']
        else:
            granules = []
            for short_name in config['collections']:
                found = search_granules(session, short_name, bbox,
                                        config['years'], config['doy_range'])
                granules.extend(found)
            with open(manifest, 'w') as f:
                json.dump({'bbox': bbox, 'granules': granules}, f)

        n_all = len(granules)
        granules = [g for g in granules if g['tile'][:2] == zone]
        n_zone = len(granules)
        granules = [g for g in granules if g['cloud'] is None
                    or g['cloud'] <= config['max_tile_cloud']]
        todo = [
            g for g in granules
            if not (aoidir / f'{g["ur"]}.tif').exists()
            and not (aoidir / f'{g["ur"]}.tif.cloudy').exists()
        ]
        tiles = sorted({g['tile'] for g in granules})
        click.echo(
            f'[{name}] {n_all} granules found, {n_zone} in zone {zone}, '
            f'{len(granules)} <= {config["max_tile_cloud"]}% tile cloud, '
            f'{len(todo)} to fetch; tiles: {", ".join(tiles)}'
        )
        if dry_run or not todo:
            continue

        # Authenticate once, serially, before the parallel reads
        bands0 = config['bands'][todo[0]['short_name']]
        read_window(todo[0]['urls'][bands0['red']], transform, shape, env)

        counts = {'ok': 0, 'cloudy': 0, 'error': 0}
        with ThreadPoolExecutor(jobs) as pool:
            futures = {
                pool.submit(
                    fetch_granule, g, config['bands'][g['short_name']],
                    transform, shape, epsg, aoidir / f'{g["ur"]}.tif', env,
                    config['min_clear_frac'],
                ): g for g in todo
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=name):
                try:
                    status, _ = fut.result()
                    counts[status] += 1
                except Exception as exc:
                    counts['error'] += 1
                    tqdm.write(f'{futures[fut]["ur"]}: {exc}')
        click.echo(f'[{name}] {counts}')


if __name__ == '__main__':
    main()
