#!/usr/bin/env python
"""
Fetch NAIP chips from Google Earth Engine (USDA/NAIP/DOQQ) on exactly the
same windows as fetch_naip_chips.py (Planetary Computer), for speed and
consistency comparison. Each chip is requested with ee.data.computePixels on
the Planetary Computer item's native grid (CRS, 0.6 m pixels, origin), so
pixels align one-to-one.

Requires `earthengine authenticate` and a Cloud project registered for Earth
Engine (--project).

    python fetch_naip_chips_ee.py MANIFEST OUTDIR --project <id> \
        -t trees.shp:2016,2018 [--limit 50]
"""
import os
import io
import json
import time
import click
import numpy as np
import rasterio
from tqdm import tqdm
from pathlib import Path
from affine import Affine
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
from fetch_naip_chips import chip_jobs

BANDS = ['R', 'G', 'B', 'N']


def fetch_chip(item, year, r, c, inner, margin, outdir):
    out = outdir / str(year) / f'{item["id"]}__r{r}_c{c}.tif'
    if out.exists():
        return 'skipped', 0
    p = item['properties']
    T = Affine(*p['proj:transform'][:6])
    size = inner + 2 * margin
    x0, y0 = T * (c * inner - margin, r * inner - margin)
    crs = f'EPSG:{p["proj:epsg"]}'
    img = (ee.ImageCollection('USDA/NAIP/DOQQ')
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .filterBounds(ee.Geometry.Point([x0 + size * T.a / 2,
                                            y0 + size * T.e / 2], crs))
           .mosaic().select(BANDS))
    req = {
        'expression': img,
        'fileFormat': 'NUMPY_NDARRAY',
        'grid': {
            'dimensions': {'width': size, 'height': size},
            'affineTransform': {'scaleX': T.a, 'shearX': 0, 'translateX': x0,
                                'shearY': 0, 'scaleY': T.e, 'translateY': y0},
            'crsCode': crs,
        },
    }
    for attempt in range(5):
        try:
            arr = ee.data.computePixels(req)
            break
        except ee.EEException as exc:
            if 'Too Many Requests' not in str(exc) or attempt == 4:
                raise
            time.sleep(2 ** attempt)
    data = np.stack([arr[b] for b in BANDS]).astype(np.uint8)
    os.makedirs(out.parent, exist_ok=True)
    prof = dict(driver='GTiff', height=size, width=size, count=4,
                dtype='uint8', crs=crs,
                transform=Affine(T.a, 0, x0, 0, T.e, y0), compress='deflate',
                tiled=True, blockxsize=256, blockysize=256)
    tmp = str(out) + '.part'
    with rasterio.open(tmp, 'w', **prof) as dst:
        dst.write(data)
    os.replace(tmp, out)
    return 'ok', data.nbytes


@click.command()
@click.argument('manifest', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--project', required=True, help='Cloud project for EE')
@click.option('-t', '--target', 'targets', multiple=True, required=True)
@click.option('--inner', default=512, show_default=True)
@click.option('--margin', default=64, show_default=True)
@click.option('--limit', default=0, help='Only the first N chips (timing)')
@click.option('-j', '--jobs', 'n_jobs', default=8, show_default=True)
def main(manifest, outputdir, project, targets, inner, margin, limit,
         n_jobs):

    ee.Initialize(project=project,
                  opt_url='https://earthengine-highvolume.googleapis.com')
    items = json.load(open(manifest))['items']
    tg = [(t.rsplit(':', 1)[0], [int(y) for y in t.rsplit(':', 1)[1].split(',')])
          for t in targets]
    jobs = chip_jobs(items, tg, inner, margin)
    if limit:
        jobs = jobs[:limit]
    t0, nbytes, counts = time.time(), 0, {'ok': 0, 'skipped': 0, 'error': 0}
    with ThreadPoolExecutor(n_jobs) as pool:
        futs = {pool.submit(fetch_chip, items[k], y, r, c, inner, margin,
                            outputdir): (k, y, r, c) for k, y, r, c in jobs}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                status, nb = f.result()
                counts[status] += 1
                nbytes += nb
            except Exception as exc:
                counts['error'] += 1
                tqdm.write(f'{futs[f]}: {exc}')
    dt = time.time() - t0
    click.echo(f'{counts} in {dt:.0f}s: {counts["ok"] / dt:.2f} chips/s, '
               f'{nbytes / 1e6 / dt:.1f} MB/s uncompressed')


if __name__ == '__main__':
    main()
