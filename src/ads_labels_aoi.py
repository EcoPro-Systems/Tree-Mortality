#!/usr/bin/env python
"""
Rasterize raw USFS Aerial Detection Survey (ADS/IDS) polygons onto each HLS
AOI's 30 m UTM grid (see fetch_hls_aoi.py), one layer per survey year.

Inputs are the damage areas, damage points and surveyed areas extracted from
the USFS Region 5 IDS geodatabase (CONUS_Region5_AllYears.gdb), plus MTBS
burned-area perimeters. A pixel is assigned to a polygon if its center falls
inside the exact polygon geometry.

Output is <outputdir>/<aoi>.nc with dims (year, y, x):
  label       int8: -1 not surveyed, 0 surveyed with no damage feature,
              1 mortality polygon, 2 mortality point only (DMSM damage points,
              within `point_radius` m), 3 non-mortality damage only
  poly_id     int32: row of <aoi>_polygons.csv for the most severe mortality
              polygon covering the pixel (-1 if none)
  n_polys     uint8: number of (possibly overlapping) mortality polygons
  tpa_sum     float32: LEGACY_TPA summed over overlapping polygons (legacy)
  pct_sum     float32: PERCENT_MID summed over overlapping polygons, capped
              at 100 (DMSM)
  burned      bool: inside an MTBS perimeter with ignition in that year
  flight_doy  int16: day of year the pixel was surveyed (-1 unknown): the
              tablet CREATED_DATE of the covering DMSM mortality polygon if
              any, otherwise the latest surveyed-area date covering the pixel
              (midpoint of START_DATE/END_DATE)
  flight_doy_min  int16: earliest surveyed-area date covering the pixel, so
              flight_doy - flight_doy_min shows repeat or multi-day coverage
and <outputdir>/<aoi>_polygons.csv with the attributes of every mortality
polygon intersecting the AOI.

Severity differs by survey era: legacy surveys (<= 2016) record trees per acre
(LEGACY_TPA) and no percent affected, while DMSM surveys (>= 2017) record
percent affected classes (PERCENT_MID) and no TPA. Legacy CREATED_DATE values are digitization dates (December),
not observation dates, so only DMSM feature dates are used.
"""
import os
import click
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from rasterio.features import rasterize, MergeAlg

from util import load_config
from fetch_hls_aoi import aoi_grid

KEEP_COLS = [
    'DAMAGE_AREA_ID', 'SURVEY_YEAR', 'HOST_CODE', 'HOST', 'HOST_GROUP',
    'DCA_CODE', 'DCA_COMMON_NAME', 'DAMAGE_TYPE', 'PERCENT_AFFECTED_CODE',
    'PERCENT_AFFECTED', 'PERCENT_MID', 'LEGACY_TPA', 'LEGACY_NO_TREES',
    'LEGACY_SEVERITY', 'AREA_TYPE', 'OBSERVATION_COUNT', 'ACRES',
    'DATA_SOURCE_NAME', 'CREATED_DATE',
]


def survey_doy(sa):
    """Day of year of each surveyed-area polygon (midpoint of start/end)"""
    start = pd.to_datetime(sa.START_DATE, errors='coerce')
    end = pd.to_datetime(sa.END_DATE, errors='coerce')
    end = end.where(end >= start, start).fillna(start)
    start = start.fillna(end)
    mid = start + (end - start) / 2
    return mid.dt.dayofyear


def burn(geoms, transform, shape, fill=0, dtype='uint8'):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.full(shape, fill, dtype=dtype)
    return rasterize(geoms, out_shape=shape, transform=transform, fill=fill,
                     dtype=dtype)


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('--damage-areas', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/usfs_ids/R5_damage_areas_sierra_2012plus.gpkg'))
@click.option('--damage-points', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/usfs_ids/R5_damage_points_sierra_2012plus.gpkg'))
@click.option('--surveyed-areas', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/usfs_ids/R5_surveyed_areas_2012plus.gpkg'))
@click.option('--fire-perimeters', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/fire/mtbs_perimeter_data/mtbs_perims_DD.shp'))
@click.option('--point-radius', default=45.0, show_default=True)
def main(configfile, outputdir, damage_areas, damage_points, surveyed_areas,
         fire_perimeters, point_radius):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    years = list(range(2012, config['years'][1] + 1))

    for name, aoi in config['aois'].items():
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        crs = f'EPSG:{aoi["epsg"]}'
        res = config['resolution']
        x0, y1 = transform.c, transform.f
        aoi_box = box(x0, y1 - shape[0] * res, x0 + shape[1] * res, y1)
        aoi_gs = gpd.GeoSeries([aoi_box], crs=crs)

        def load(path):
            bbox = aoi_gs.to_crs(gpd.read_file(path, rows=1).crs)
            df = gpd.read_file(path, bbox=bbox.iloc[0].buffer(1000))
            return df.to_crs(crs).clip(aoi_box.buffer(500))

        da = load(damage_areas)
        dp = load(damage_points)
        sa = load(surveyed_areas)
        fire = load(fire_perimeters)
        fire['year'] = fire['ig_date'].astype(str).str[:4].astype(int)

        mort = da[da.DAMAGE_TYPE == 'Mortality'].copy()
        mort['severity_key'] = mort.PERCENT_MID.fillna(mort.LEGACY_TPA)
        mort = mort.sort_values(['SURVEY_YEAR', 'severity_key'])
        mort = mort.reset_index(drop=True)
        mort['poly_id'] = np.arange(len(mort), dtype=np.int32)
        mort['aoi_acres'] = mort.geometry.intersection(aoi_box).area / 4046.86
        created = pd.to_datetime(mort.CREATED_DATE, errors='coerce')
        dmsm = (mort.DATA_SOURCE_NAME == 'DMSM_DAMAGE_AREAS')
        mort['created_doy'] = created.dt.dayofyear.where(
            dmsm & (created.dt.year == mort.SURVEY_YEAR))
        sa['doy'] = survey_doy(sa)

        label = np.full((len(years),) + shape, -1, dtype=np.int8)
        poly_id = np.full((len(years),) + shape, -1, dtype=np.int32)
        burned = np.zeros((len(years),) + shape, dtype=bool)
        n_polys = np.zeros((len(years),) + shape, dtype=np.uint8)
        tpa_sum = np.full((len(years),) + shape, np.nan, dtype=np.float32)
        pct_sum = np.full((len(years),) + shape, np.nan, dtype=np.float32)
        flight_doy = np.full((len(years),) + shape, -1, dtype=np.int16)
        flight_doy_min = np.full((len(years),) + shape, -1, dtype=np.int16)

        for i, year in enumerate(years):
            surveyed = burn(sa[sa.SURVEY_YEAR == year].geometry,
                            transform, shape)
            label[i][surveyed == 1] = 0

            s = sa[(sa.SURVEY_YEAR == year) & sa.doy.notna()]
            if len(s):
                # Sorted by date, so the latest (or earliest) survey wins
                for arr, asc in [(flight_doy, True), (flight_doy_min, False)]:
                    ss = s.sort_values('doy', ascending=asc)
                    arr[i] = rasterize(
                        zip(ss.geometry, ss.doy.astype('int16')),
                        out_shape=shape, transform=transform, fill=-1,
                        dtype='int16',
                    )

            other = da[(da.SURVEY_YEAR == year)
                       & (da.DAMAGE_TYPE != 'Mortality')]
            label[i][burn(other.geometry, transform, shape) == 1] = 3

            pts = dp[(dp.SURVEY_YEAR == year)
                     & (dp.DAMAGE_TYPE == 'Mortality')]
            label[i][burn(pts.geometry.buffer(point_radius),
                          transform, shape) == 1] = 2

            # Sorted by severity, so later (more severe) polygons overwrite
            m = mort[mort.SURVEY_YEAR == year]
            if len(m):
                pid = rasterize(
                    zip(m.geometry, m.poly_id), out_shape=shape,
                    transform=transform, fill=-1, dtype='int32',
                )
                poly_id[i] = pid
                label[i][pid >= 0] = 1
                cdoy = m.set_index('poly_id').created_doy
                has = pid >= 0
                vals = cdoy.reindex(pid[has]).values
                fd = flight_doy[i][has]
                flight_doy[i][has] = np.where(np.isfinite(vals), vals, fd)
                n_polys[i] = rasterize(
                    ((g, 1) for g in m.geometry), out_shape=shape,
                    transform=transform, fill=0, dtype='uint8',
                    merge_alg=MergeAlg.add,
                )
                for arr, col, cap in [(tpa_sum, 'LEGACY_TPA', None),
                                      (pct_sum, 'PERCENT_MID', 100)]:
                    mm = m[m[col].notna() & (m[col] >= 0)]
                    if not len(mm):
                        continue
                    v = rasterize(
                        zip(mm.geometry, mm[col].astype('float32')),
                        out_shape=shape, transform=transform, fill=0,
                        dtype='float32', merge_alg=MergeAlg.add,
                    )
                    v[pid < 0] = np.nan
                    arr[i] = v if cap is None else np.minimum(v, cap)

            burned[i] = burn(fire[fire.year == year].geometry,
                             transform, shape) == 1

        xs = x0 + res * (np.arange(shape[1]) + 0.5)
        ys = y1 - res * (np.arange(shape[0]) + 0.5)
        ds = xr.Dataset(
            {
                'label': (('year', 'y', 'x'), label),
                'poly_id': (('year', 'y', 'x'), poly_id),
                'n_polys': (('year', 'y', 'x'), n_polys),
                'tpa_sum': (('year', 'y', 'x'), tpa_sum),
                'pct_sum': (('year', 'y', 'x'), pct_sum),
                'burned': (('year', 'y', 'x'), burned),
                'flight_doy': (('year', 'y', 'x'), flight_doy),
                'flight_doy_min': (('year', 'y', 'x'), flight_doy_min),
            },
            coords={'year': years, 'y': ys, 'x': xs},
            attrs={'crs': crs, 'transform': list(transform)[:6]},
        )
        enc = {v: {'zlib': True} for v in ds.data_vars}
        ds.to_netcdf(outputdir / f'{name}.nc', encoding=enc)

        cols = KEEP_COLS + ['severity_key', 'poly_id', 'aoi_acres',
                            'created_doy']
        mort[cols].to_csv(outputdir / f'{name}_polygons.csv', index=False)

        # Summary: burned positive area vs. polygon acreage clipped to AOI
        px_acres = res * res / 4046.86
        summ = pd.DataFrame({
            'surveyed_frac': (label >= 0).mean(axis=(1, 2)),
            'mort_poly_px_acres': (label == 1).sum(axis=(1, 2)) * px_acres,
            'mort_poly_acres': mort.groupby('SURVEY_YEAR').aoi_acres.sum()
            .reindex(years).fillna(0).values,
            'mort_point_px': (label == 2).sum(axis=(1, 2)),
            'burned_frac': burned.mean(axis=(1, 2)),
            'dated_frac': ((flight_doy >= 0).sum(axis=(1, 2))
                           / np.maximum((label >= 0).sum(axis=(1, 2)), 1)),
            'median_doy': [np.median(f[f >= 0]) if (f >= 0).any() else -1
                           for f in flight_doy],
        }, index=years).round(3)
        click.echo(f'[{name}]\n{summ.to_string()}')


if __name__ == '__main__':
    main()
