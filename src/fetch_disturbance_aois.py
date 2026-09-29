#!/usr/bin/env python
"""
Fire and harvest polygons over the AOIs that MTBS misses (MTBS maps only
fires >= 1000 ac in the West).

- CAL FIRE FRAP historic fire perimeters, all sizes (California Fire
  Perimeters (all)).
- CAL FIRE prescribed fire perimeters (California_Prescribed_Fire_Perimeters
  _Public).
- USFS FACTS timber harvest activities (EDW_TimberHarvest_01: 2001-2010,
  2011-2020, 2021 onward), with activity names, so salvage and sanitation
  cuts after the die-off can be told apart from earlier harvests.

All three are queried from ArcGIS REST services with the AOI lon/lat boxes
and saved as GeoPackages in OUTPUTDIR (EPSG:4326).

    python fetch_disturbance_aois.py ../config/hls_aois.yml $E/fire/disturbance
"""
import time
import click
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

from util import load_config
from fetch_hls_aoi import aoi_grid, aoi_lonlat_bbox

CALFIRE = ('https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/'
           'services/{}/FeatureServer/{}/query')
FACTS = ('https://apps.fs.usda.gov/arcx/rest/services/EDW/'
         'EDW_TimberHarvest_01/MapServer/{}/query')
SOURCES = {
    'frap_fires': [CALFIRE.format('California_Historic_Fire_Perimeters', 0)],
    'rx_fires': [CALFIRE.format('California_Prescribed_Fire_Perimeters'
                                '_Public', 0)],
    'facts_harvest': [FACTS.format(i) for i in (1, 0, 11)],
}
# The FACTS server times out on large pages of full-resolution polygons, so
# it gets small pages, selected fields and ~10 m simplified geometries
PAGE = {'facts_harvest': 100}
FIELDS = {'facts_harvest': 'activity_name,activity_code,treatment_type,'
                           'date_completed,fy_completed,suid'}
SIMPLIFY_DEG = 0.0001
RETRIES = 5


def get(url, params):
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=300)
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError,
                requests.HTTPError):
            if attempt == RETRIES - 1:
                raise
            time.sleep(10 * (attempt + 1))


def query(url, bbox, page=1000, fields='*'):
    frames, off = [], 0
    while True:
        js = get(url, dict(
            where='1=1', geometry=','.join(map(str, bbox)),
            geometryType='esriGeometryEnvelope', inSR=4326,
            spatialRel='esriSpatialRelIntersects', outFields=fields,
            outSR=4326, f='geojson', resultOffset=off,
            resultRecordCount=page, maxAllowableOffset=SIMPLIFY_DEG))
        feats = js.get('features', [])
        if feats:
            frames.append(gpd.GeoDataFrame.from_features(feats, crs=4326))
        off += len(feats)
        if len(feats) < page:
            return frames


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
def main(configfile, outputdir):
    cfg = load_config(configfile)
    outputdir.mkdir(parents=True, exist_ok=True)
    boxes = []
    for aoi in cfg['aois'].values():
        t, s = aoi_grid(aoi, cfg['size_m'], cfg['resolution'])
        boxes.append(aoi_lonlat_bbox(t, s, aoi['epsg']))
    for name, urls in SOURCES.items():
        if (outputdir / f'{name}.gpkg').exists():
            click.echo(f'{name}: exists')
            continue
        frames = [f for u in urls for b in boxes
                  for f in query(u, b, PAGE.get(name, 1000),
                                 FIELDS.get(name, '*'))]
        g = pd.concat(frames, ignore_index=True)
        key = [c for c in g.columns if c.lower() in ('objectid',)]
        g = g.drop_duplicates(subset=key + ['geometry'] if key else None)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        g.to_file(outputdir / f'{name}.gpkg')
        click.echo(f'{name}: {len(g)} polygons')


if __name__ == '__main__':
    main()
