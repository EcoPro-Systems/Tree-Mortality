#!/usr/bin/env python
"""
Airborne lidar stand structure on the 30 m AOI grids, from two NASA
airborne lidar sources beyond the NEON 2013 tree segmentation.

1. ASO composite (Ferraz et al. 2020, "From lidar waveforms to vegetation
   products", doi:10.5068/D16T06, Zenodo 3964981). NASA-JPL Airborne Snow
   Observatory lidar from 2014-2017 over the Kings and San Joaquin basins,
   merged into one point cloud per tile, with 10 m structure rasters and a
   5 m CHM (UTM 11N). Per 30 m cell:
     aso_rh25/50/75/98   relative heights (mean of 10 m pixels)
     aso_pai, aso_fhd    plant area index, foliage height diversity
     aso_cv, aso_crown_ratio   height CV, canopy depth / height
     aso_chm_mean/max    5 m CHM mean and max
     aso_cover2/5        fraction of 5 m CHM pixels > 2 m / > 5 m
     aso_frac_tall       fraction of 5 m CHM pixels > 30 m
     aso_cov             fraction of 10 m pixels measured (mask 0 or 3)
     aso_snowoff_year    year of the earliest snow-off flight over the cell
   The snow-off flights are Oct 2015 (Kings, over NEON SOAP/TEAK) and Oct
   2016 (San Joaquin, over Sierra NF), during the 2015-16 die-off, and the
   merge includes 2017 snow-on flights. Structure therefore partly records
   drought mortality (standing dead trees keep their height; their
   foliage is gone).

2. LVIS 2008 (Land, Vegetation and Ice Sensor, LDS 1.03; NASA GSFC,
   Sep 21-26 2008, four days of Sierra Nevada flights; binary .lge files).
   Footprint (~20 m) ground elevation and RH25/50/75/100. Footprints are kept if RH100 is
   0-80 m and the ground elevation agrees with SRTM (after removing the
   median ellipsoid-geoid offset) within 40 m. Per 30 m cell (NaN where
   there is no footprint):
     lvis_n              footprints
     lvis_rh100_mean/max, lvis_rh50_mean, lvis_rh25_mean
     lvis_rh50_ratio     mean RH50/RH100 (footprints with RH100 > 5 m)
     lvis_frac_tall      fraction of footprints with RH100 > 30 m
     lvis_cov            1 where lvis_n > 0

Output is $E/lidar/<aoi>_lidar.nc.

    python fetch_lidar_structure.py -a neon_soap_teak -a sierra_nf
"""
import re
import zipfile
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
from pathlib import Path
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds

import response_common as rc
from aoi_env_layers import cell_centers

LIDAR = rc.E / 'lidar'
ASO = LIDAR / 'aso'
LVIS = LIDAR / 'lvis2008'
ASO_EPSG = 32611
ASO_METRICS = {
    'aso_rh98': 'CanopyHeight_10m', 'aso_rh25': 'RelativeHeight25_10m',
    'aso_rh50': 'RelativeHeight50_10m', 'aso_rh75': 'RelativeHeight75_10m',
    'aso_pai': 'PlantAreaIndex_10m', 'aso_fhd': 'FoliageHeightDiversity_10m',
    'aso_cv': 'CoefficientOfVariation_10m',
    'aso_crown_ratio': 'CanopyRatio_10m'}
ASO_MASK = 'Mask_0Veg_1Outside_2Interp_3NonVeg'
TALL_M = 30
SNOWOFF_RE = re.compile(r'_snowoff_(\d{4})_')


def unzip_once(zf, dest):
    if not dest.exists():
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest)
    return dest


def aoi_bounds(transform, shape):
    x0, y1 = transform.c, transform.f
    return x0, y1 + shape[0] * transform.e, x0 + shape[1] * transform.a, y1


def to_grid(src_arr, src_transform, src_crs, transform, shape, epsg,
            resampling=Resampling.average):
    out = np.full(shape, np.nan, np.float32)
    reproject(src_arr.astype(np.float32), out, src_transform=src_transform,
              src_crs=src_crs, src_nodata=np.nan, dst_transform=transform,
              dst_crs=f'EPSG:{epsg}', dst_nodata=np.nan,
              resampling=resampling)
    return out


def read_window(path, transform, shape, epsg, margin=100):
    """Read the part of a raster covering the AOI (AOI in the raster CRS)"""
    with rasterio.open(path) as src:
        b = aoi_bounds(transform, shape)
        if src.crs.to_epsg() != epsg:
            tr = Transformer.from_crs(f'EPSG:{epsg}', src.crs,
                                      always_xy=True)
            xs, ys = tr.transform([b[0], b[2], b[0], b[2]],
                                  [b[1], b[1], b[3], b[3]])
            b = min(xs), min(ys), max(xs), max(ys)
        b = (b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)
        win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        # masked: nodata and pixels outside the raster become NaN
        a = src.read(1, window=win, boundless=True, masked=True)
        a = a.astype(np.float32).filled(np.nan)
        return a, src.window_transform(win), src.crs


def aso_layers(transform, shape, epsg):
    zips = [ASO / f'Ferraz-ASO_raster_{n}.zip' for n in (
        'forest_traits_and_diversity_metrics', 'CHM')]
    if any(not z.exists() or zipfile.is_zipfile(z) is False for z in zips):
        click.echo('  ASO rasters not downloaded; skipped')
        return {}
    rdir = unzip_once(ASO / 'Ferraz-ASO_raster_forest_traits_and_diversity_'
                      'metrics.zip', ASO / 'forest_metrics')
    files = {p.name: p for p in rdir.rglob('*.tif')}

    def find(key):
        hits = [p for n, p in files.items() if key in n]
        if not hits:
            raise FileNotFoundError(f'no ASO raster matching {key}')
        return hits[0]

    mask, mtr, mcrs = read_window(find(ASO_MASK), transform, shape, epsg)
    measured = np.isin(mask, (0, 3))
    out = {'aso_cov': to_grid(np.where(np.isfinite(mask), measured, np.nan),
                              mtr, mcrs, transform, shape, epsg)}
    if not np.nanmax(out['aso_cov'], initial=0) > 0:
        return {}
    for name, key in ASO_METRICS.items():
        a, tr, crs = read_window(find(key), transform, shape, epsg)
        if a.shape == mask.shape:
            a[~measured] = np.nan
            nonveg = mask == 3
            # non-vegetated pixels: no canopy
            a[nonveg] = 0 if name != 'aso_crown_ratio' else np.nan
        out[name] = to_grid(a, tr, crs, transform, shape, epsg)

    # 5 m CHM tiles
    cdir = unzip_once(ASO / 'Ferraz-ASO_raster_CHM.zip', ASO / 'chm')
    b = aoi_bounds(transform, shape)
    if epsg != ASO_EPSG:
        return out
    tiles = []
    for p in cdir.rglob('*.tif'):
        with rasterio.open(p) as src:
            tb = src.bounds
        if tb.left < b[2] and tb.right > b[0] and tb.bottom < b[3] \
                and tb.top > b[1]:
            tiles.append(p)
    if tiles:
        chm, ctr = merge(tiles, bounds=b, nodata=np.nan, dtype='float32',
                         method='max')
        chm = chm[0]
        chm[chm < 0] = 0
        ok = np.isfinite(chm)
        crs = f'EPSG:{ASO_EPSG}'
        out['aso_chm_mean'] = to_grid(chm, ctr, crs, transform, shape, epsg)
        out['aso_chm_max'] = to_grid(chm, ctr, crs, transform, shape, epsg,
                                     Resampling.max)
        for name, h in (('aso_cover2', 2), ('aso_cover5', 5),
                        ('aso_frac_tall', TALL_M)):
            out[name] = to_grid(np.where(ok, chm > h, np.nan), ctr, crs,
                                transform, shape, epsg)
        click.echo(f'  CHM: {len(tiles)} tiles')

    # earliest snow-off flight over each cell
    year = np.full(shape, np.nan, np.float32)
    bdir = ASO / 'meta/ASO_lidar_point_clouds_tiles_boundaries'
    for f in sorted(bdir.glob('*_snowoff_*.shp')):
        y = int(SNOWOFF_RE.search(f.name).group(1))
        g = gpd.read_file(f).to_crs(epsg).make_valid()
        m = rasterize(list(g.geometry), out_shape=shape, transform=transform,
                      fill=0, default_value=1).astype(bool)
        year[m & ~(year <= y)] = y
    out['aso_snowoff_year'] = year
    return out


# LDS 1.03 ground-elevation (.lge) record, big-endian, 64 bytes (v1.02 plus
# azimuth, incidence angle and range)
LGE_DTYPE = np.dtype([
    ('lfid', '>u4'), ('shot', '>u4'), ('azimuth', '>f4'),
    ('incidence', '>f4'), ('range', '>f4'), ('time', '>f8'),
    ('GLON', '>f8'), ('GLAT', '>f8'), ('ZG', '>f4'), ('RH25', '>f4'),
    ('RH50', '>f4'), ('RH75', '>f4'), ('RH100', '>f4')])


def lvis_shots():
    cache = LVIS / 'lvis2008_shots.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for zf in sorted(LVIS.glob('LVIS_US_CA_day*_VECT_*.zip')):
        with zipfile.ZipFile(zf) as z:
            name = [n for n in z.namelist() if n.endswith('.lge.1.03')][0]
            a = np.frombuffer(z.read(name), dtype=LGE_DTYPE)
        df = pd.DataFrame({c: a[c].astype(np.float64 if c in (
            'GLON', 'GLAT', 'time') else np.float32) for c in (
            'GLON', 'GLAT', 'ZG', 'RH25', 'RH50', 'RH75', 'RH100',
            'incidence', 'time')})
        df['day'] = int(re.search(r'day(\d)', zf.name).group(1))
        frames.append(df)
        click.echo(f'  {zf.name}: {len(df)} shots')
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache)
    return df


def lvis_layers(transform, shape, epsg, elev):
    df = lvis_shots()
    lon = df.GLON.values.copy()
    lon[lon > 180] -= 360
    tr = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = tr.transform(lon, df.GLAT.values)
    col = np.floor((x - transform.c) / transform.a).astype(int)
    row = np.floor((y - transform.f) / transform.e).astype(int)
    inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    d = df[inside].assign(row=row[inside], col=col[inside])
    if d.empty:
        return {}
    d = d[(d.RH100 >= 0) & (d.RH100 <= 80) & np.isfinite(d.ZG)]
    dz = d.ZG.values - elev[d.row.values, d.col.values]
    off = np.nanmedian(dz)
    d = d[np.abs(dz - off) < 40]
    click.echo(f'  LVIS: {inside.sum()} footprints in AOI, {len(d)} kept '
               f'(ZG - SRTM median {off:+.1f} m)')
    d = d.assign(cell=d.row * shape[1] + d.col,
                 ratio=np.where(d.RH100 > 5, d.RH50 / d.RH100, np.nan),
                 tall=(d.RH100 > TALL_M).astype(float))
    g = d.groupby('cell')
    stats = pd.DataFrame({
        'lvis_n': g.size(), 'lvis_rh100_mean': g.RH100.mean(),
        'lvis_rh100_max': g.RH100.max(), 'lvis_rh50_mean': g.RH50.mean(),
        'lvis_rh25_mean': g.RH25.mean(), 'lvis_rh50_ratio': g.ratio.mean(),
        'lvis_frac_tall': g.tall.mean()})
    out = {}
    for c in stats.columns:
        a = np.full(shape[0] * shape[1], np.nan, np.float32)
        a[stats.index.values] = stats[c].values
        out[c] = a.reshape(shape)
    out['lvis_cov'] = np.isfinite(out['lvis_n']).astype(np.float32)
    return out


@click.command()
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
def main(aois):
    for aoi in aois:
        transform, shape, epsg = rc.aoi_info(aoi)
        env = rc.open_env(aoi)
        layers = {}
        click.echo(f'[{aoi}] ASO')
        layers.update(aso_layers(transform, shape, epsg))
        click.echo(f'[{aoi}] LVIS 2008')
        layers.update(lvis_layers(transform, shape, epsg,
                                  env.elevation.values))
        if not layers:
            click.echo(f'[{aoi}] no lidar')
            continue
        for k, v in layers.items():
            if k.endswith('_cov'):
                click.echo(f'  {k}: {np.nanmean(v > 0):.1%} of AOI')
        xs, ys = cell_centers(transform, shape)
        ds = xr.Dataset({k: (('y', 'x'), v) for k, v in layers.items()},
                        coords={'y': ys, 'x': xs},
                        attrs={'crs': f'EPSG:{epsg}',
                               'transform': list(transform)[:6],
                               'aso': 'Ferraz et al. 2020, Zenodo 3964981',
                               'lvis': 'LVIS LDS 1.03, Sep 2008'})
        out = LIDAR / f'{aoi}_lidar.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True} for k in ds.data_vars})
        click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
