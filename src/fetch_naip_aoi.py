#!/usr/bin/env python
"""
Download NAIP quarter-quad images (4-band RGBN cloud-optimized GeoTIFFs)
intersecting each HLS AOI from the Microsoft Planetary Computer, as
published (no clipping or resampling). Files are written to

    <outputdir>/<aoi>/<year>/<item_id>.tif   (+ <item_id>.json STAC item)

and a manifest of all items per AOI is kept in <outputdir>/<aoi>/_items.json.
Downloads go to a .part file and are renamed once the byte count matches
Content-Length; completed files are skipped, so the fetch is restartable.

Planetary Computer asset URLs need a short-lived SAS token, which is fetched
from the token API and refreshed before it expires.
"""
import os
import json
import time
import threading
import click
import requests
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from util import load_config
from fetch_hls_aoi import aoi_grid, aoi_lonlat_bbox

STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
TOKEN_URL = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/naip'
RETRIES = 5
BACKOFF = 5.0
CHUNK = 1 << 22


class Token:
    """Thread-safe cached SAS token, refreshed 10 minutes before expiry"""

    def __init__(self):
        self.lock = threading.Lock()
        self.token, self.expiry = None, None

    def get(self):
        with self.lock:
            now = datetime.now(timezone.utc)
            if self.token is None or (self.expiry - now).total_seconds() < 600:
                r = requests.get(TOKEN_URL, timeout=60)
                r.raise_for_status()
                d = r.json()
                self.token = d['token']
                self.expiry = datetime.fromisoformat(
                    d['msft:expiry'].replace('Z', '+00:00'))
            return self.token


def search(bbox, years=None):
    body = {'collections': ['naip'], 'bbox': list(bbox), 'limit': 1000}
    r = requests.post(STAC_URL, json=body, timeout=120)
    r.raise_for_status()
    feats = r.json()['features']
    if years:
        feats = [f for f in feats if int(f['properties']['naip:year']) in years]
    return feats


def download(item, outfile, token):
    href = item['assets']['image']['href']
    for attempt in range(1, RETRIES + 1):
        try:
            with requests.get(f'{href}?{token.get()}', stream=True,
                              timeout=(30, 300)) as r:
                r.raise_for_status()
                size = int(r.headers['Content-Length'])
                if outfile.exists() and outfile.stat().st_size == size:
                    return 'skipped', size
                tmp = Path(str(outfile) + '.part')
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                if tmp.stat().st_size != size:
                    raise IOError(f'short read {tmp.stat().st_size}/{size}')
                os.replace(tmp, outfile)
                return 'ok', size
        except (requests.RequestException, IOError, KeyError) as exc:
            if attempt == RETRIES:
                raise
            time.sleep(BACKOFF * attempt)


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('-a', '--aoi', 'aois', multiple=True)
@click.option('-y', '--year', 'years', multiple=True, type=int,
              help='NAIP year(s) to fetch (default: all available)')
@click.option('-j', '--jobs', default=4, show_default=True)
@click.option('--dry-run', is_flag=True)
def main(configfile, outputdir, aois, years, jobs, dry_run):

    config = load_config(configfile)
    aois = aois or list(config['aois'])
    token = Token()

    for name in aois:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        bbox = aoi_lonlat_bbox(transform, shape, aoi['epsg'])
        items = search(bbox, set(years))
        aoidir = outputdir / name
        os.makedirs(aoidir, exist_ok=True)
        with open(aoidir / '_items.json', 'w') as f:
            json.dump({'bbox': bbox, 'items': items}, f)

        todo = []
        for item in items:
            ydir = aoidir / str(item['properties']['naip:year'])
            os.makedirs(ydir, exist_ok=True)
            with open(ydir / f'{item["id"]}.json', 'w') as f:
                json.dump(item, f)
            outfile = ydir / f'{item["id"]}.tif'
            if not outfile.exists():
                todo.append((item, outfile))
        by_year = {}
        for item in items:
            y = item['properties']['naip:year']
            by_year[y] = by_year.get(y, 0) + 1
        click.echo(f'[{name}] {len(items)} items {dict(sorted(by_year.items()))}'
                   f'; {len(todo)} to download')
        if dry_run or not todo:
            continue

        counts, nbytes = {'ok': 0, 'skipped': 0, 'error': 0}, 0
        with ThreadPoolExecutor(jobs) as pool:
            futures = {pool.submit(download, it, out, token): it
                       for it, out in todo}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=name):
                try:
                    status, size = fut.result()
                    counts[status] += 1
                    nbytes += size
                except Exception as exc:
                    counts['error'] += 1
                    tqdm.write(f'{futures[fut]["id"]}: {exc}')
        click.echo(f'[{name}] {counts}, {nbytes / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
