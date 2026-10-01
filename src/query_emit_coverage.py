#!/usr/bin/env python
"""
How often does a spaceborne imaging spectrometer see these forests in the
growing season? EMIT L2A reflectance scenes (LP DAAC EMITL2ARFL, 60 m,
from the ISS, Aug 2022 onward) over the AOIs and a wider Sierra Nevada box,
against the AVIRIS growing-season flights of the same areas.

From CMR only (no downloads): each granule's footprint polygon, start time,
granule-level cloud cover and solar zenith angle. A scene is "usable" for a cell when the
footprint covers it, it is Jun-Sep, the solar zenith angle at the scene
centre is <= --max-sza (the ISS overpass drifts through the day; this keeps
morning and afternoon scenes with sun high enough for steep terrain), and
the granule cloud cover is below --max-cloud. Local time (UTC - 7 h, PDT)
is recorded for reference.
Per-pixel cloud masks are not read, so this is an upper bound on clear
views of a given cell and a lower bound for a scene that is cloudy
elsewhere.

Outputs in OUTPUTDIR:
  emit_granules.csv  granule id, UTC and local time, cloud, AOI coverage
  emit_cells.csv     per AOI, year and criterion: scenes per 90 m cell
                     (median, p10, p90) and the share of cells with >= 1
                     and >= 2 usable scenes
  emit_vs_aviris.csv usable EMIT scenes per cell vs AVIRIS growing-season
                     flight dates (>= 50% of the AOI with public L2) per
                     year, from query_airborne_coverage.py
  emit_map.png       mean usable scenes per year over the Sierra box

    python query_emit_coverage.py ../config/hls_aois.yml \
        $E/hls_results/emit_coverage
"""
import click
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from affine import Affine
from pathlib import Path
from rasterio.features import rasterize

from util import load_config
from query_airborne_coverage import (AOIS, cmr_granules, umm_polygon,
                                     aoi_rect, to_utm_poly)

SHORT_NAME = 'EMITL2ARFL'
START = '2022-08-01'
SIERRA_BOX = (-121.2, 35.6, -118.2, 40.2)  # lon/lat, west slope to crest
MAP_RES = 0.02  # degrees
UTC_OFFSET = -7
SEASON = (6, 9)
CELL = 3  # 90 m cells on the 30 m AOI grid


def granule_table(session, bbox):
    rows = []
    for g in cmr_granules(session, SHORT_NAME, bbox, START):
        u = g['umm']
        t = pd.Timestamp(u['TemporalExtent']['RangeDateTime']
                         ['BeginningDateTime']).tz_convert(None)
        attrs = {a['Name']: a['Values'][0]
                 for a in u.get('AdditionalAttributes', [])}
        cloud = u.get('CloudCover')
        rows.append(dict(granule=u['GranuleUR'], utc=t,
                         local=t + pd.Timedelta(hours=UTC_OFFSET),
                         cloud=float(cloud) if cloud is not None else np.nan,
                         sza=float(attrs.get('SOLAR_ZENITH', np.nan)),
                         geometry=umm_polygon(u)))
    d = pd.DataFrame(rows).drop_duplicates('granule')
    d['year'] = d.local.dt.year
    d['season'] = d.local.dt.month.between(*SEASON)
    return d


def usable(d, max_cloud, max_sza):
    return d.season & (d.sza <= max_sza) & (d.cloud.fillna(100) < max_cloud)


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--max-cloud', 'max_clouds', multiple=True, type=float,
              default=[20, 50], show_default=True)
@click.option('--max-sza', 'max_szas', multiple=True, type=float,
              default=[50, 60], show_default=True)
@click.option('--airborne-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_results/airborne_coverage')
def main(configfile, outputdir, max_clouds, max_szas, airborne_dir):
    cfg = load_config(configfile)
    outputdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    g = granule_table(session, SIERRA_BOX)
    click.echo(f'{len(g)} EMIT granules over the Sierra box since {START}; '
               f'cloud known for {g.cloud.notna().mean():.0%}')

    cells, cov = [], {}
    for name in AOIS:
        rect, tr, _ = aoi_rect(cfg, name)
        x0, y0, x1, y1 = rect.bounds
        res = cfg['resolution'] * CELL
        shape = (int(round((y1 - y0) / res)), int(round((x1 - x0) / res)))
        transform = Affine(res, 0, x0, 0, -res, y1)
        foot = {}
        for i, r in g.iterrows():
            if r.geometry is None:
                continue
            p = to_utm_poly(r.geometry, tr)
            if not p.intersects(rect):
                continue
            f = p.intersection(rect).area / rect.area
            if f < 0.01:
                continue
            foot[i] = rasterize([(p, 1)], shape, transform=transform,
                                fill=0, dtype='uint8').astype(bool)
            g.loc[i, f'cov_{name}'] = f
        cov[name] = foot
        for mc, ms in [(c, z) for c in max_clouds for z in max_szas]:
            ok = usable(g, mc, ms)
            for y in sorted(g.year.unique()):
                n = np.zeros(shape, int)
                for i in foot:
                    if ok[i] and g.year[i] == y:
                        n += foot[i]
                cells.append(dict(aoi=name, year=y, max_cloud=mc, max_sza=ms,
                                  median=np.median(n),
                                  p10=np.percentile(n, 10),
                                  p90=np.percentile(n, 90),
                                  frac_ge1=(n >= 1).mean(),
                                  frac_ge2=(n >= 2).mean(),
                                  scenes=len({i for i in foot if ok[i] and
                                              g.year[i] == y})))
            c = [x for x in cells if x['aoi'] == name
                 and x['max_cloud'] == mc and x['max_sza'] == ms]
            click.echo(f'[{name}] cloud < {mc:.0f}%, SZA <= {ms:.0f}: ' +
                       '  '.join(
                f'{x["year"]} med {x["median"]:.0f} '
                f'(>=1 {x["frac_ge1"]:.0%})' for x in c))
    cells = pd.DataFrame(cells)
    g.drop(columns='geometry').to_csv(outputdir / 'emit_granules.csv',
                                      index=False)
    cells.to_csv(outputdir / 'emit_cells.csv', index=False)

    f = airborne_dir / 'coverage_dates.csv'
    if f.exists():
        a = pd.read_csv(f, parse_dates=['date'])
        a = a[a.date.dt.month.between(*SEASON) & (a.l2_frac >= 0.5)]
        av = (a.groupby(['aoi', 'year']).date.nunique()
              .rename('aviris_dates').reset_index())
        cmp = cells.merge(av, on=['aoi', 'year'], how='outer')
        cmp['aviris_dates'] = cmp.aviris_dates.fillna(0).astype(int)
        cmp.sort_values(['aoi', 'year']).to_csv(
            outputdir / 'emit_vs_aviris.csv', index=False)

    fig_map(g, max_clouds[0], max_szas[0], outputdir / 'emit_map.png', cfg)


def fig_map(g, max_cloud, max_sza, path, cfg):
    w, s, e, n = SIERRA_BOX
    shape = (int(round((n - s) / MAP_RES)), int(round((e - w) / MAP_RES)))
    transform = Affine(MAP_RES, 0, w, 0, -MAP_RES, n)
    ok = usable(g, max_cloud, max_sza)
    total = np.zeros(shape)
    for i, r in g[ok].iterrows():
        if r.geometry is not None:
            total += rasterize([(r.geometry, 1)], shape, transform=transform,
                               fill=0, dtype='uint8')
    years = g.local[ok].dt.year
    ny = years.max() - years.min() + 1
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(total / ny, extent=(w, e, s, n), cmap='viridis')
    fig.colorbar(im, ax=ax, shrink=0.7,
                 label=f'usable EMIT scenes per year (Jun-Sep, SZA <= '
                       f'{max_sza:.0f}°, cloud < {max_cloud:.0f}%)')
    for name in AOIS:
        _, _, bb = aoi_rect(cfg, name)
        ax.add_patch(plt.Rectangle(bb[:2], bb[2] - bb[0], bb[3] - bb[1],
                                   fill=False, ec='w', lw=1))
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_title(f'EMIT growing-season views, {years.min()}-{years.max()}',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
