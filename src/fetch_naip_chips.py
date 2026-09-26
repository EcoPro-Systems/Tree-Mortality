#!/usr/bin/env python
"""
Fetch small NAIP chips around target locations (e.g., field-validated or
hand-labeled trees) with windowed reads of the Planetary Computer COGs,
instead of downloading whole quarter-quads.

Targets are vector files, each paired with the NAIP year(s) to fetch:

    --target path/to/trees.shp:2016,2018

Each target's centroid is located in the NAIP item(s) of that year (from a
STAC manifest written by fetch_naip_aoi.py). Target locations are binned
into an `inner` x `inner` pixel tiling of each item's native grid; every
occupied tile is read with a `margin` of context on each side (so detections
near a target have full model context) and written as

    <outputdir>/<year>/<item_id>__r<row>_c<col>.tif   (4-band, native CRS)

If the full item already exists locally (under --local-dir/<year>/), it is
read from disk. Existing chips are skipped, so the fetch is restartable.
"""
import os
import json
import time
import click
import numpy as np
import rasterio
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from affine import Affine
from shapely.geometry import shape as to_shape
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_naip_aoi import Token

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='3',
                GDAL_HTTP_MULTIRANGE='YES', GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES')


def chip_jobs(items, targets, inner, margin):
    """(item, year, row, col) for every occupied tile"""
    q = gpd.GeoDataFrame(
        {'k': range(len(items)),
         'year': [int(i['properties']['naip:year']) for i in items]},
        geometry=[to_shape(i['geometry']) for i in items], crs=4326)
    jobs = set()
    for path, years in targets:
        g = gpd.read_file(path)
        g = g[g.geometry.notna()]
        pts = gpd.GeoDataFrame(geometry=g.to_crs(4326).geometry.centroid,
                               crs=4326)
        j = gpd.sjoin(q[q.year.isin(years)], pts, predicate='contains')
        for k, sub in j.groupby('k'):
            it = items[k]
            p = it['properties']
            T = Affine(*p['proj:transform'][:6])
            xy = pts.loc[sub.index_right].to_crs(p['proj:epsg']).geometry
            col, row = ~T * (xy.x.values, xy.y.values)
            for r, c in set(zip((row // inner).astype(int),
                                (col // inner).astype(int))):
                jobs.add((k, int(p['naip:year']), r, c))
    return sorted(jobs)


def fetch_chip(item, year, r, c, inner, margin, outdir, local_dir, token):
    out = outdir / str(year) / f'{item["id"]}__r{r}_c{c}.tif'
    if out.exists():
        return 'skipped'
    local = local_dir / str(year) / f'{item["id"]}.tif' if local_dir else None
    src = str(local) if local and local.exists() else \
        '/vsicurl/' + item['assets']['image']['href'] + '?' + token.get()
    win = Window(c * inner - margin, r * inner - margin,
                 inner + 2 * margin, inner + 2 * margin)
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(src) as ds:
            data = ds.read(window=win, boundless=True, fill_value=0)
            prof = ds.profile.copy()
            prof.update(driver='GTiff', height=data.shape[1],
                        width=data.shape[2], transform=ds.window_transform(win),
                        compress='deflate', tiled=True, blockxsize=256,
                        blockysize=256, photometric=None)
            prof.pop('jpeg_quality', None)
    os.makedirs(out.parent, exist_ok=True)
    tmp = str(out) + '.part'
    with rasterio.open(tmp, 'w', **prof) as dst:
        dst.write(data)
    os.replace(tmp, out)
    return 'local' if src == str(local) else 'remote'


@click.command()
@click.argument('manifest', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-t', '--target', 'targets', multiple=True, required=True,
              help='PATH:YEAR[,YEAR...]')
@click.option('--local-dir', type=click.Path(path_type=Path), default=None,
              help='Directory with already-downloaded full items, by year')
@click.option('--inner', default=512, show_default=True)
@click.option('--margin', default=64, show_default=True)
@click.option('--limit', default=0, help='Only the first N chips (timing)')
@click.option('-j', '--jobs', 'n_jobs', default=8, show_default=True)
def main(manifest, outputdir, targets, local_dir, inner, margin, limit,
         n_jobs):

    items = json.load(open(manifest))['items']
    tg = []
    for t in targets:
        path, years = t.rsplit(':', 1)
        tg.append((path, [int(y) for y in years.split(',')]))
    jobs = chip_jobs(items, tg, inner, margin)
    if limit:
        jobs = jobs[:limit]
    click.echo(f'{len(jobs)} chips of {inner + 2 * margin} px '
               f'({(inner + 2 * margin) * 0.6:.0f} m)')
    token = Token()
    t0 = time.time()
    counts = {'skipped': 0, 'local': 0, 'remote': 0, 'error': 0}
    with ThreadPoolExecutor(n_jobs) as pool:
        futs = {pool.submit(fetch_chip, items[k], y, r, c, inner, margin,
                            outputdir, local_dir, token): (k, y, r, c)
                for k, y, r, c in jobs}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                counts[f.result()] += 1
            except Exception as exc:
                counts['error'] += 1
                tqdm.write(f'{futs[f]}: {exc}')
    dt = time.time() - t0
    click.echo(f'{counts} in {dt:.0f}s: '
               f'{(counts["remote"] + counts["local"]) / dt:.2f} chips/s')


if __name__ == '__main__':
    main()
