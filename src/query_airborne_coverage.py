#!/usr/bin/env python
"""
Inventory of airborne imaging spectroscopy (AVIRIS-Classic, -NG, -3, -5)
flight lines over the AOIs, 2013 onward, and how much of each AOI they cover.

Two sources are combined because neither is complete:
- The AVIRIS Flight Line Locator (ORNL DAAC 2140) flight tables list every
  line flown through Aug 2024, with corner coordinates, whether or not the
  L2 reflectance was ever processed or released. Download
  AVIRIS-{C,NG}_flight_table.csv into LOCATORDIR.
- CMR granules of the public L2 reflectance collections (listed in
  L2_COLLECTIONS). These show which lines have released reflectance, and
  they are the only source for lines after the locator ends.

A line is "flown" if it is in either source and "l2" if a public L2
reflectance granule exists for it. Coverage is the fraction of the AOI
rectangle covered by the union of a date's line footprints.

It also reports the WDTS AVIRIS-C trait mosaic coverage per year
(flight_id > 0, and qc_all > 0 over forest) from <aoi>_traits.nc.

Outputs in OUTPUTDIR: coverage_lines.csv, coverage_dates.csv,
trait_coverage.csv and coverage_map.png.

    python query_airborne_coverage.py ../config/hls_aois.yml \
        $E/aviris_locator $E/hls_results/airborne_coverage
"""
import re
import click
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from util import load_config
from fetch_hls_aoi import CMR_URL, aoi_grid, aoi_lonlat_bbox, get_with_retry

AOIS = ['neon_soap_teak', 'sierra_nf', 'stanislaus']
L2_COLLECTIONS = {
    'WDTS_AVIRIS-C_L2_corrected_2391': 'AVIRIS-C',
    'AVIRIS-Classic_L2_Reflectance_2154': 'AVIRIS-C',
    'AVIRIS-NG_L2_Reflectance_2110': 'AVIRIS-NG',
    'AV3_L2A_RFL_2357': 'AVIRIS-3',
    'AV5_L2A_RFL_2484': 'AVIRIS-5',
}
# Flight line id: f<yymmdd>t01p00r<nn> (C), ang<yyyymmdd>t<hhmmss> (NG),
# AV3/AV5<yyyymmdd>t<hhmmss>_<nnn> (-3/-5)
LINE_RE = re.compile(r'(f\d{6}t\d{2}p\d{2}r\d{2}|ang\d{8}t\d{6}|'
                     r'AV[35]\d{8}t\d{6}_\d{3})')
FOREST = (41, 42, 43)
MIN_FRAC = 0.01


def line_date(line):
    if line.startswith('f'):
        return pd.Timestamp('20' + line[1:7])
    if line.startswith('ang'):
        return pd.Timestamp(line[3:11])
    return pd.Timestamp(line[3:11])


def cmr_granules(session, short_name, bbox, start):
    """All granules of a collection over bbox since start (paged)"""
    params = {'short_name': short_name, 'page_size': 2000,
              'bounding_box': ','.join(f'{v:.5f}' for v in bbox),
              'temporal': f'{start}T00:00:00Z,'}
    headers, out = {}, []
    while True:
        resp = get_with_retry(session, CMR_URL, params, headers)
        items = resp.json().get('items', [])
        out.extend(items)
        after = resp.headers.get('CMR-Search-After')
        if not items or not after:
            return out
        headers = {'CMR-Search-After': after}


def umm_polygon(umm):
    geom = (umm.get('SpatialExtent', {}).get('HorizontalSpatialDomain', {})
            .get('Geometry', {}))
    if geom.get('GPolygons'):
        pts = geom['GPolygons'][0]['Boundary']['Points']
        return Polygon([(p['Longitude'], p['Latitude']) for p in pts])
    if geom.get('BoundingRectangles'):
        r = geom['BoundingRectangles'][0]
        return box(r['WestBoundingCoordinate'], r['SouthBoundingCoordinate'],
                   r['EastBoundingCoordinate'], r['NorthBoundingCoordinate'])
    return None


def locator_lines(locatordir, first_year):
    rows = []
    for inst in ['C', 'NG']:
        f = locatordir / f'AVIRIS-{inst}_flight_table.csv'
        t = pd.read_csv(f, low_memory=False)
        t = t[(t.year >= first_year) & (t.gp_lat1 > -900)]
        for _, r in t.iterrows():
            poly = Polygon([(r[f'gp_lon{k}'], r[f'gp_lat{k}'])
                            for k in range(1, 5)])
            rows.append({'line': r.flight_line, 'instrument': f'AVIRIS-{inst}',
                         'date': pd.Timestamp(r.date), 'site': r.site_name,
                         'geometry': poly, 'source': 'locator'})
    return pd.DataFrame(rows)


def aoi_rect(cfg, name):
    aoi = cfg['aois'][name]
    t, s = aoi_grid(aoi, cfg['size_m'], cfg['resolution'])
    rect = box(t.c, t.f + s[0] * t.e, t.c + s[1] * t.a, t.f)
    to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{aoi["epsg"]}',
                                  always_xy=True)
    return rect, to_utm, aoi_lonlat_bbox(t, s, aoi['epsg'])


def to_utm_poly(poly, tr):
    xs, ys = tr.transform(*poly.exterior.xy)
    p = Polygon(zip(xs, ys))
    return p if p.is_valid else p.buffer(0)


def trait_coverage(wdtsdir, landcoverdir):
    rows = []
    for name in AOIS:
        f = wdtsdir / f'{name}_traits.nc'
        if not f.exists():
            continue
        ds = xr.open_dataset(f)
        with rasterio.open(landcoverdir / f'{name}_landcover.tif') as src:
            forest = np.isin(src.read(1), FOREST)
        for y in ds.year.values:
            fid = ds.flight_id.sel(year=y).values
            qc = ds.qc_all.sel(year=y).values.astype(float)
            foot = fid > 0
            rows.append({
                'aoi': name, 'year': int(y), 'footprint': foot.mean(),
                'forest_footprint': foot[forest].mean(),
                'forest_valid': ((qc > 0) & (qc <= 100) & foot)[forest].mean(),
                'forest_qc_mean': np.nanmean(
                    np.where(foot & (qc <= 100), qc, np.nan)[forest]) / 100,
            })
    return pd.DataFrame(rows)


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('locatordir', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--first-year', default=2013, show_default=True)
@click.option('--wdts-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/wdts')
@click.option('--landcover-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/landcover')
def main(configfile, locatordir, outputdir, first_year, wdts_dir,
         landcover_dir):
    cfg = load_config(configfile)
    outputdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    lines = locator_lines(locatordir, first_year)
    l2 = []
    for name in AOIS:
        _, _, bbox = aoi_rect(cfg, name)
        for short, inst in L2_COLLECTIONS.items():
            for g in cmr_granules(session, short, bbox,
                                  f'{first_year}-01-01'):
                m = LINE_RE.search(g['umm']['GranuleUR'])
                if m:
                    l2.append({'line': m.group(1), 'instrument': inst,
                               'collection': short,
                               'geometry': umm_polygon(g['umm'])})
    l2 = pd.DataFrame(l2).drop_duplicates(['line', 'collection'])
    click.echo(f'{len(lines)} locator lines, {l2.line.nunique()} lines '
               'with public L2 reflectance over the AOIs')

    # Lines only in CMR (after the locator ends, or -3/-5)
    extra = l2[~l2.line.isin(lines.line)].drop_duplicates('line').copy()
    extra['date'] = extra.line.map(line_date)
    extra['site'] = ''
    extra['source'] = 'cmr'
    lines = pd.concat([lines, extra[lines.columns]], ignore_index=True)
    coll = l2.groupby('line').collection.agg(','.join)
    lines['l2'] = lines.line.map(coll).fillna('')

    rows, dates = [], []
    for name in AOIS:
        rect, tr, _ = aoi_rect(cfg, name)
        for _, r in lines.iterrows():
            if r.geometry is None:
                continue
            p = to_utm_poly(r.geometry, tr)
            f = p.intersection(rect).area / rect.area
            if f >= MIN_FRAC:
                rows.append({**r.drop('geometry').to_dict(), 'aoi': name,
                             'frac': round(f, 3), 'poly': p})
    cov = pd.DataFrame(rows)
    for (name, date, inst), g in cov.groupby(['aoi', 'date', 'instrument']):
        rect, _, _ = aoi_rect(cfg, name)
        u = unary_union(list(g.poly)).intersection(rect)
        gl2 = g[g.l2 != '']
        ul2 = (unary_union(list(gl2.poly)).intersection(rect)
               if len(gl2) else None)
        dates.append({
            'aoi': name, 'date': date.date(), 'year': date.year,
            'instrument': inst, 'n_lines': len(g),
            'flown_frac': round(u.area / rect.area, 3),
            'l2_frac': round(ul2.area / rect.area, 3) if ul2 else 0.0,
            'sites': '; '.join(sorted(set(str(s) for s in g.site
                                          if str(s) != 'nan')))[:120],
        })
    dates = pd.DataFrame(dates).sort_values(['aoi', 'date'])
    cov.drop(columns='poly').sort_values(['aoi', 'date', 'line']).to_csv(
        outputdir / 'coverage_lines.csv', index=False)
    dates.to_csv(outputdir / 'coverage_dates.csv', index=False)

    tc = trait_coverage(wdts_dir, landcover_dir)
    tc.round(3).to_csv(outputdir / 'trait_coverage.csv', index=False)

    # Map: AOI rows x year columns (2018 onward), footprints clipped to AOI
    years = list(range(2018, int(dates.year.max()) + 1))
    fig, axes = plt.subplots(len(AOIS), len(years),
                             figsize=(2 * len(years), 2.2 * len(AOIS)))
    colors = {'AVIRIS-C': 'tab:blue', 'AVIRIS-NG': 'tab:green',
              'AVIRIS-3': 'tab:orange', 'AVIRIS-5': 'tab:red'}
    for i, name in enumerate(AOIS):
        rect, _, _ = aoi_rect(cfg, name)
        for j, y in enumerate(years):
            ax = axes[i, j]
            ax.plot(*rect.exterior.xy, 'k-', lw=0.8)
            g = cov[(cov.aoi == name) & (cov.date.dt.year == y)]
            for _, r in g.iterrows():
                p = r.poly.intersection(rect)
                if p.is_empty or p.geom_type != 'Polygon':
                    continue
                ax.fill(*p.exterior.xy, alpha=0.25 if r.l2 else 0.08,
                        color=colors[r.instrument],
                        ec=colors[r.instrument], lw=0.4,
                        ls='-' if r.l2 else '--')
            d = dates[(dates.aoi == name) & (dates.year == y)]
            ax.set_title(f'{y}: {len(d)} date(s)', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4)
               for c in colors.values()]
    fig.legend(handles, list(colors), loc='lower center', ncol=4, fontsize=8)
    fig.suptitle('AVIRIS flight lines over the AOIs (solid: public L2 '
                 'reflectance; faint dashed: flown, no public L2)',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(outputdir / 'coverage_map.png', dpi=150)

    with pd.option_context('display.width', 200, 'display.max_rows', 500,
                           'display.max_colwidth', 60):
        click.echo(dates[dates.year >= 2018].to_string(index=False))
        click.echo(tc.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
