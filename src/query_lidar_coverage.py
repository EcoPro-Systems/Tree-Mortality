#!/usr/bin/env python
"""
Airborne lidar coverage of the study AOIs: which acquisitions exist, when,
and what fraction of each AOI (and of its NLCD forest) they cover.

Sources:
- USGS 3DEP lidar point-cloud work units, from the 3DEP elevation index
  service (layer 24, which has collection start/end dates). The hobu EPT
  boundary file misses some recent work units, so it is not used.
- OpenTopography point-cloud datasets (otCatalog API; catalog footprints,
  some of which are hulls), which hold the pre-2015 USFS Region 5, NCALM and
  NEON prototype collections.
- NASA-JPL Airborne Snow Observatory (ASO) flights merged into the 2014-2017
  structure composite of Ferraz et al. (2020, doi:10.5068/D16T06, Zenodo
  3964981): per-flight tile boundaries from that release.
- LVIS 2008 Sierra Nevada flights (LDS 1.03): cells with at least one
  quality-filtered footprint, from fetch_lidar_structure.py output.
- NEON 2013 lidar trees (Hemming-Schroeder et al. 2023) at SOAP/TEAK: cells
  with segmented trees.

Footprints are rasterized onto the 30 m AOI grids. Output in <outputdir>:
coverage_sources.csv and coverage_<aoi>.png.

    python query_lidar_coverage.py $E/hls_results/lidar_coverage \\
        -a neon_soap_teak -a sierra_nf -a stanislaus
"""
import re
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import requests
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.features import rasterize
from shapely.geometry import shape as to_shape

import response_common as rc
from fetch_hls_aoi import aoi_lonlat_bbox

TDEP_URL = ('https://index.nationalmap.gov/arcgis/rest/services/'
            '3DEPElevationIndex/MapServer/24/query')
OT_URL = 'https://portal.opentopography.org/API/otCatalog'
ASO_TILES = rc.E / 'lidar/aso/meta/ASO_lidar_point_clouds_tiles_boundaries'
ASO_RE = re.compile(r'(\w+?)_(snowoff|snowon)_(\d{4})_(\d{2})_(\d{2})')
FOREST = (41, 42, 43)


def tdep_units(bbox):
    params = {'geometry': ','.join(map(str, bbox)),
              'geometryType': 'esriGeometryEnvelope', 'inSR': 4326,
              'outSR': 4326, 'spatialRel': 'esriSpatialRelIntersects',
              'outFields': 'workunit,collect_start,collect_end,ql',
              'returnGeometry': 'true', 'f': 'geojson'}
    r = requests.get(TDEP_URL, params=params, timeout=120)
    r.raise_for_status()
    g = gpd.GeoDataFrame.from_features(r.json()['features'], crs=4326)
    return pd.DataFrame({
        'source': '3DEP', 'dataset': g.workunit,
        'start': pd.to_datetime(g.collect_start, unit='ms').dt.date,
        'end': pd.to_datetime(g.collect_end, unit='ms').dt.date,
        'detail': g.ql, 'geometry': g.geometry})


def opentopo(bbox):
    params = dict(productFormat='PointCloud', minx=bbox[0], miny=bbox[1],
                  maxx=bbox[2], maxy=bbox[3], detail='true',
                  outputFormat='json', include_federated='false')
    r = requests.get(OT_URL, params=params, timeout=120)
    r.raise_for_status()
    rows = []
    for item in r.json()['Datasets']:
        d = item['Dataset']
        geo = d['spatialCoverage']['geo']['geojson']['features']
        start, _, end = d.get('temporalCoverage', '').partition(' / ')
        rows.append(dict(source='OpenTopography', dataset=d['alternateName'],
                         start=start, end=end or start, detail=d['name'],
                         geometry=to_shape(geo[0]['geometry'])))
    return pd.DataFrame(rows)


def aso_flights():
    rows = []
    for f in sorted(ASO_TILES.glob('*.shp')):
        basin, kind, y, m, dd = ASO_RE.match(f.stem).groups()
        g = gpd.read_file(f).to_crs(4326).make_valid()
        rows.append(dict(source='ASO (Ferraz 2020 composite)',
                         dataset=f'{basin} {kind}', start=f'{y}-{m}-{dd}',
                         end=f'{y}-{m}-{dd}', detail=kind,
                         geometry=list(g)))
    return pd.DataFrame(rows)


def raster_sources(aoi, shape):
    """Coverage masks for sources already gridded on the AOI"""
    out = []
    f = rc.E / 'lidar' / f'{aoi}_lidar.nc'
    if f.exists():
        ds = xr.open_dataset(f)
        if 'lvis_n' in ds:
            out.append(dict(source='LVIS', dataset='LVIS 2008 Sierra Nevada',
                            start='2008-09-21', end='2008-09-26',
                            detail='LDS 1.03, >=1 footprint per 30 m cell',
                            mask=ds.lvis_n.values > 0))
    env = rc.open_env(aoi)
    if 'lidar_n_trees' in env:
        out.append(dict(source='NEON AOP', dataset='NEON 2013 lidar trees',
                        start='2013-06-01', end='2013-06-30',
                        detail='Hemming-Schroeder et al. 2023 segmentation',
                        mask=np.isfinite(env.lidar_n_trees.values)))
    return out


def fig_map(aoi, masks, forest, path):
    n = len(masks)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (name, m) in zip(axes.flat, masks.items()):
        ax.imshow(np.where(forest, 0.25, 0.0) + 0.75 * m, cmap='Greens',
                  vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{name}\n{m.mean():.0%} of AOI', fontsize=8)
    fig.suptitle(f'{aoi}: lidar coverage (dark) over forest (light)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
def main(outputdir, aois):
    outputdir.mkdir(parents=True, exist_ok=True)
    aso = aso_flights() if ASO_TILES.exists() else pd.DataFrame()
    rows = []
    for aoi in aois:
        transform, shape, epsg = rc.aoi_info(aoi)
        bbox = aoi_lonlat_bbox(transform, shape, epsg)
        with rasterio.open(rc.E / 'landcover' / f'{aoi}_landcover.tif') as s:
            forest = np.isin(s.read(1), FOREST)
        polys = pd.concat([tdep_units(bbox), opentopo(bbox), aso],
                          ignore_index=True)
        items = []
        for r in polys.itertuples():
            geoms = r.geometry if isinstance(r.geometry, list) else [
                r.geometry]
            g = gpd.GeoSeries(geoms, crs=4326).to_crs(epsg)
            m = rasterize(list(g), out_shape=shape, transform=transform,
                          fill=0, default_value=1).astype(bool)
            items.append(dict(source=r.source, dataset=r.dataset,
                              start=r.start, end=r.end, detail=r.detail,
                              mask=m))
        items += raster_sources(aoi, shape)
        masks = {}
        for it in items:
            m = it.pop('mask')
            if not m.any():
                continue
            rows.append(dict(aoi=aoi, **it, frac_aoi=m.mean(),
                             frac_forest=m[forest].mean()))
            key = f'{it["dataset"]} ({str(it["start"])[:4]})'
            masks[key] = masks.get(key, False) | m
            click.echo(f'[{aoi}] {it["source"]:28s} {it["dataset"]:36s} '
                       f'{it["start"]} {m.mean():6.1%} of AOI, '
                       f'{m[forest].mean():6.1%} of forest')
        # Combined pre-2015 coverage (structure before the 2015-16 die-off)
        pre = np.zeros(shape, bool)
        for r in [x for x in rows if x['aoi'] == aoi]:
            if str(r['end'])[:4] <= '2014':
                pre |= masks[f'{r["dataset"]} ({str(r["start"])[:4]})']
        rows.append(dict(aoi=aoi, source='all', dataset='any pre-2015',
                         start='', end='2014-12-31', detail='union',
                         frac_aoi=pre.mean(), frac_forest=pre[forest].mean()))
        click.echo(f'[{aoi}] any pre-2015: {pre.mean():.1%} of AOI, '
                   f'{pre[forest].mean():.1%} of forest')
        fig_map(aoi, masks, forest, outputdir / f'coverage_{aoi}.png')
    pd.DataFrame(rows).to_csv(outputdir / 'coverage_sources.csv', index=False)


if __name__ == '__main__':
    main()
