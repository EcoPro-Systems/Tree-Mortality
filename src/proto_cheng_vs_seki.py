#!/usr/bin/env python
"""
Evaluate the Cheng et al. cross-resolution dead-tree model (run on NAIP by
proto_cheng_naip_inference.py) against USGS field data from Sequoia & Kings
Canyon NPs (Das 2024, https://doi.org/10.5066/P9GYXCPG): GPS-located trees
with field-assessed live/dead status.

Datasets and the NAIP year each is paired with:
  MCVValidationNorth2020NAIP   systematic 50x20 m transects, all trees
                               > 40 cm, summer/fall 2020 -> NAIP 2020
                               (4 m2 markers, treated as points).
                               Dead trees are graded: F1-F3 retain dead
                               foliage, T0-T3 retain none (twigs only).
  SpeciesMapCalibrationSouth2020NAIP  points, Dec 2020-Jan 2021 -> NAIP 2020
  SpeciesMapCalibrationNorth2016NAIP  points -> NAIP 2016
  SpeciesMapValidationNorth2019NAIP   crown polygons (TAOs, often several
                               trees), summer 2019 -> NAIP 2018 and 2020

Points: a tree counts as detected if any model dead-crown pixel (energy > 0)
lies within r meters (r = 2, 5, 10; NAIP and GPS positional error are a few
meters). Polygons: fraction of the polygon's pixels flagged dead.

--list-items prints the NAIP STAC item ids (from the fetch_naip_aoi.py
manifest for the `seki` area) needed for these datasets.
"""
import os
import glob
import json
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import shape as to_shape
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from sklearn.metrics import roc_auc_score

DATASETS = {
    'MCV2020': ('MCVValidationNorth2020NAIP', [2020], 'point'),
    'South2020': ('SpeciesMapCalibrationSouth2020NAIP', [2020], 'point'),
    'North2016': ('SpeciesMapCalibrationNorth2016NAIP', [2016], 'point'),
    'Val2019': ('SpeciesMapValidationNorth2019NAIP', [2018, 2020], 'polygon'),
}
RADII = [2, 5, 10]
DEAD_FOLIAGE = {'F1', 'F2', 'F3'}
DEAD_BARE = {'T0', 'T1', 'T2', 'T3'}


def status_class(s):
    if s in ('Live', 'PID'):
        return 'live'
    if s in DEAD_FOLIAGE:
        return 'dead_foliage'
    if s in DEAD_BARE:
        return 'dead_bare'
    if s == 'Dead':
        return 'dead'
    if s == 'Live-Dead':
        return 'mixed'
    return 'other'


def load(field_dir, key):
    fname, years, kind = DATASETS[key]
    g = gpd.read_file(field_dir / f'{fname}.shp')
    g = g[g.geometry.notna()].reset_index(drop=True)
    if kind == 'point':
        g['geometry'] = g.to_crs(26911).geometry.centroid
        g = g.set_crs(26911, allow_override=True)
    g['cls'] = g.Status.map(status_class)
    return g, years, kind


def score(g, kind, files):
    """Point: detection within each radius; polygon: overlap fraction"""
    cols = ([f'det_{r}m' for r in RADII] if kind == 'point'
            else ['frac', 'det_any'])
    out = pd.DataFrame(np.nan, index=g.index, columns=cols)
    for f in files:
        with rasterio.open(f) as ds:
            gg = g.to_crs(ds.crs)
            b = ds.bounds
            pad = max(RADII) + 1
            geoms = gg.geometry if kind == 'polygon' else gg.geometry.buffer(pad)
            inside = ((geoms.bounds.minx > b.left) & (geoms.bounds.maxx < b.right)
                      & (geoms.bounds.miny > b.bottom) & (geoms.bounds.maxy < b.top))
            for i in np.flatnonzero(inside.values & out.iloc[:, 0].isna().values):
                geom = geoms.iloc[i]
                win = from_bounds(*geom.bounds, ds.transform)
                win = win.round_offsets().round_lengths()
                e = ds.read(1, window=win)
                wt = ds.window_transform(win)
                if kind == 'polygon':
                    m = geometry_mask([geom], out_shape=e.shape, invert=True,
                                      transform=wt)
                    if m.sum():
                        out.iloc[i] = [(e[m] > 0).mean(), float((e[m] > 0).any())]
                else:
                    p = gg.geometry.iloc[i]
                    rows, cols_ = np.nonzero(e > 0)
                    if len(rows) == 0:
                        out.iloc[i] = 0.0
                        continue
                    xs, ys = wt * (cols_ + 0.5, rows + 0.5)
                    d = np.hypot(np.asarray(xs) - p.x, np.asarray(ys) - p.y).min()
                    out.iloc[i] = [float(d <= r) for r in RADII]
    return out


def needed_items(field_dir, manifest):
    items = json.load(open(manifest))['items']
    q = gpd.GeoDataFrame(
        {'id': [i['id'] for i in items],
         'year': [int(i['properties']['naip:year']) for i in items]},
        geometry=[to_shape(i['geometry']) for i in items], crs=4326)
    ids = set()
    for key in DATASETS:
        g, years, _ = load(field_dir, key)
        pts = gpd.GeoDataFrame(geometry=g.to_crs(4326).geometry.centroid,
                               crs=4326)
        j = gpd.sjoin(q[q.year.isin(years)], pts, predicate='contains')
        ids |= set(j.id)
    return sorted(ids)


@click.command()
@click.argument('energydir', type=click.Path(path_type=Path))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--field-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/usgs_seki_deadtree_validation'))
@click.option('--manifest', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/naip/seki/_items.json'))
@click.option('--list-items', is_flag=True)
def main(energydir, outputdir, field_dir, manifest, list_items):

    if list_items:
        print(' '.join(f'-i {i}' for i in needed_items(field_dir, manifest)))
        return
    os.makedirs(outputdir, exist_ok=True)
    rows, scored = [], []
    for key, (fname, years, kind) in DATASETS.items():
        g, _, _ = load(field_dir, key)
        for year in years:
            files = sorted(glob.glob(str(energydir / str(year) / '*_energy.tif')))
            s = score(g, kind, files)
            d = pd.concat([g[['Status', 'cls']], s], axis=1).dropna()
            d['dataset'], d['naip_year'] = key, year
            scored.append(d)
            metric_cols = [c for c in s.columns if c.startswith('det')] + \
                (['frac'] if kind == 'polygon' else [])
            for cls, dc in d.groupby('cls'):
                r = dict(dataset=key, naip_year=year, cls=cls, n=len(dc))
                r.update({f'rate_{c}': dc[c].mean() if c != 'frac'
                          else dc[c].mean() for c in metric_cols})
                rows.append(r)
            live = d.cls == 'live'
            dead = d.cls.str.startswith('dead')
            if live.sum() and dead.sum():
                sc = d['frac'] if kind == 'polygon' else d[f'det_{RADII[1]}m']
                rows.append(dict(dataset=key, naip_year=year, cls='AUC dead vs live',
                                 n=int((live | dead).sum()),
                                 auc=roc_auc_score(dead[live | dead],
                                                   sc[live | dead])))
    res = pd.DataFrame(rows)
    res.to_csv(outputdir / 'seki_metrics.csv', index=False)
    pd.concat(scored).to_csv(outputdir / 'seki_scored.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
