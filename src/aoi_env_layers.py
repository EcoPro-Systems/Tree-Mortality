#!/usr/bin/env python
"""
Environment, terrain, structure and mask layers on the 30 m HLS AOI grids,
for the drought-response experiments.

- Climate (BCMv8, 270 m, water years): each 30 m cell takes its nearest BCM
  cell. Annual cwd, aet, pet, ppt, tmx and SPEI1-4 for the requested years,
  plus 1981-2010 climatologies of cwd, ppt and tmx.
- Terrain: the local SRTM-derived layers in topo/generated/ (EPSG:3310,
  about 24 x 31 m) averaged onto the AOI grid.
- Structure, wall to wall (Earth Engine): GLAD forest height 2010,
  LANDFIRE 2014 (LF 1.4.0) EVH and EVC classes, and USFS/NLCD tree canopy
  cover 2010 and 2013 (science product). NLCD 2013 land cover comes from
  landcover/<aoi>_landcover.tif.
- Structure, lidar (NEON SOAP/TEAK only): per-cell summaries of the
  Hemming-Schroeder 2013 tree segmentation (n_trees, height mean/p90/max,
  fraction > 30 m, crown area, dead in 2013), NaN where there are no trees.
- Masks: forest (NLCD 41/42/43); burned area per year from MTBS (fires
  >= 1000 ac), CAL FIRE FRAP (all sizes) and CAL FIRE prescribed burns; USFS
  FACTS harvest per completion year, with salvage and sanitation cuts also
  kept separately (fetch_disturbance_aois.py). FACTS "Natural Changes"
  records are mortality reports, not treatments, and are left out.

Output is <outputdir>/<aoi>_env.nc with static (y, x) layers and
(year, y, x) climate and burned layers. --masks-only keeps NLCD, terrain
and the burned and harvest layers only (no climate or structure), for large
grids that so far need only the forest and disturbance masks (the held-out
sites in config/heldout_aois.yml, with their own --disturbance-dir).

    python aoi_env_layers.py ../config/hls_aois.yml $E/env -a neon_soap_teak
"""
import time
import click
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import geopandas as gpd
from pathlib import Path
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

import ee
from util import load_config
from fetch_hls_aoi import aoi_grid, aoi_lonlat_bbox

E = Path('/Volumes/Earth04/ecopro')
BCM_VARS = ['cwd', 'aet', 'pet', 'ppt', 'tmx']
SPEI_VARS = ['SPEI1', 'SPEI2', 'SPEI3', 'SPEI4']
CLIM_YEARS = (1981, 2010)
TOPO = {'elevation.nc': ['elevation'],
        'slpasp.nc': ['slope_9x9', 'northness_9x9', 'eastness_9x9'],
        'tpi.nc': ['tpi'], 'vrm.nc': ['vrm'], 'rie.nc': ['rie'],
        'sapa.nc': ['sapa'], 'dmv.nc': ['sdmv'], 'adjsd.nc': ['adjSD']}
FOREST = (41, 42, 43)
TALL_M = 30
DISTURBANCE = E / 'fire/disturbance'
SALVAGE = ('Salvage', 'Sanitation')


def cell_centers(transform, shape):
    xs = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
    ys = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    return xs, ys


def bcm_layers(transform, shape, epsg, years):
    """Nearest BCM 270 m cell for every AOI cell"""
    xs, ys = cell_centers(transform, shape)
    X, Y = np.meshgrid(xs, ys)
    tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:3310', always_xy=True)
    bx, by = tr.transform(X, Y)
    ann = xr.open_dataset(E / 'BCMv8/BCMv8_annual.nc4')
    idx = xr.open_dataset(E / 'BCMv8/BCMv8_indexes.nc4')
    east, north = ann.easting.values, ann.northing.values
    # Regular grid: nearest cell by arithmetic on the cell-centre spacing
    ci = np.rint((bx.ravel() - east[0]) / (east[1] - east[0])).astype(int)
    ri = np.rint((by.ravel() - north[0]) / (north[1] - north[0])).astype(int)
    c0, c1, r0, r1 = ci.min(), ci.max() + 1, ri.min(), ri.max() + 1
    ci, ri = (ci - c0).reshape(shape), (ri - r0).reshape(shape)
    win = dict(easting=slice(c0, c1), northing=slice(r0, r1))
    out = {}
    for v in BCM_VARS:
        a = ann[v].isel(**win)
        out[f'{v}_ann'] = a.sel(year=years).values[:, ri, ci]
        if v in ('cwd', 'ppt', 'tmx'):
            clim = a.sel(year=slice(*CLIM_YEARS)).mean('year').values
            out[f'{v}_clim'] = clim[ri, ci]
    for v in SPEI_VARS:
        a = idx[v].isel(**win)
        yy = [y for y in years if y in a.year.values]
        arr = np.full((len(years),) + shape, np.nan, np.float32)
        arr[[years.index(y) for y in yy]] = a.sel(year=yy).values[:, ri, ci]
        out[f'{v.lower()}_ann'] = arr
    out['bcm_cell'] = (ri + r0) * 10000 + (ci + c0)
    return out


def topo_layers(transform, shape, epsg):
    """Average the local SRTM-derived layers onto the AOI grid"""
    xs, ys = cell_centers(transform, shape)
    tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:3310', always_xy=True)
    bx, by = tr.transform(*np.meshgrid(xs[[0, -1]], ys[[0, -1]]))
    pad = 500
    out = {}
    for fname, variables in TOPO.items():
        ds = xr.open_dataset(E / 'topo/generated' / fname)
        e, n = ds.easting.values, ds.northing.values
        dx, dy = e[1] - e[0], n[1] - n[0]
        ci = np.flatnonzero((e > bx.min() - pad) & (e < bx.max() + pad))
        ri = np.flatnonzero((n > by.min() - pad) & (n < by.max() + pad))
        sub = ds.isel(easting=slice(ci[0], ci[-1] + 1),
                      northing=slice(ri[0], ri[-1] + 1))
        src_tr = Affine(dx, 0, e[ci[0]] - dx / 2, 0, dy, n[ri[0]] - dy / 2)
        for v in variables:
            dst = np.full(shape, np.nan, np.float32)
            reproject(sub[v].values.astype(np.float32), dst,
                      src_transform=src_tr, src_crs='EPSG:3310',
                      dst_transform=transform, dst_crs=f'EPSG:{epsg}',
                      src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.average)
            out[v.replace('_9x9', '').lower()] = dst
    return out


def ee_structure(transform, shape, epsg):
    """GLAD 2010 height, LF 1.4.0 EVH/EVC, TCC 2010/2013 on the AOI grid"""
    tcc = (ee.ImageCollection(
        'projects/gtac-data-publish/assets/TCC/Product_Version/2025-6')
        .filter(ee.Filter.eq('study_area', 'CONUS')))

    def tcc_year(y):
        return (tcc.filter(ee.Filter.eq('year', y)).first()
                .select('Science_Percent_Tree_Canopy_Cover')
                .rename(f'tcc{y}').toFloat())

    lf = {k: ee.ImageCollection(f'LANDFIRE/Vegetation/{k}/v1_4_0')
          .filter(ee.Filter.eq('system:index', 'CONUS')).first()
          .rename(f'lf14_{k.lower()}').toFloat() for k in ('EVH', 'EVC')}
    img = (ee.Image('projects/glad/GLCLU2020/Forest_height_2010')
           .rename('glad_h2010').toFloat()
           .addBands([tcc_year(2010), tcc_year(2013), lf['EVH'], lf['EVC']]))
    bands = ['glad_h2010', 'tcc2010', 'tcc2013', 'lf14_evh', 'lf14_evc']
    out = {b: np.full(shape, np.nan, np.float32) for b in bands}
    step = 500
    for r0 in range(0, shape[0], step):
        n = min(step, shape[0] - r0)
        req = {'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
               'grid': {'dimensions': {'width': shape[1], 'height': n},
                        'affineTransform': {
                            'scaleX': transform.a, 'shearX': 0,
                            'translateX': transform.c, 'shearY': 0,
                            'scaleY': transform.e,
                            'translateY': transform.f + r0 * transform.e},
                        'crsCode': f'EPSG:{epsg}'}}
        for attempt in range(5):
            try:
                arr = ee.data.computePixels(req)
                break
            except ee.EEException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        for b in bands:
            out[b][r0:r0 + n] = arr[b]
    # GLAD: 0-60 m forest height; 101+ codes are non-forest/water/no data
    out['glad_h2010'][out['glad_h2010'] > 100] = np.nan
    for b in ('tcc2010', 'tcc2013'):
        out[b][out[b] > 100] = np.nan
    return out


def lidar_structure(transform, shape):
    """Per-cell 2013 tree summaries from the Hemming-Schroeder segmentation"""
    from proto_predisposition_neon import load_trees, pixel_index
    df = load_trees(E / 'hemming_schroeder2023')
    row, col = pixel_index(df, transform, shape)
    df = df[row >= 0].assign(cell=row[row >= 0] * shape[1] + col[row >= 0])
    live = df[df.live2013 == 1]
    g = live.groupby('cell')
    stats = pd.DataFrame({
        'lidar_n_trees': g.size(),
        'lidar_h_mean': g.zmax2013.mean(),
        'lidar_h_p90': g.zmax2013.quantile(0.9),
        'lidar_h_max': g.zmax2013.max(),
        'lidar_frac_tall': g.zmax2013.apply(lambda h: (h > TALL_M).mean()),
        'lidar_ca_mean': g.ca2013.mean(),
        'lidar_ca_sum': g.ca2013.sum(),
    })
    stats['lidar_dead2013'] = (df.groupby('cell').live2013
                               .apply(lambda s: (s == 0).mean()))
    stats['lidar_is_soap'] = live.groupby('cell').is_soap.mean().round()
    out = {}
    for c in stats.columns:
        a = np.full(shape[0] * shape[1], np.nan, np.float32)
        a[stats.index.values] = stats[c].values
        out[c] = a.reshape(shape)
    return out


def burned_layers(transform, shape, epsg, years):
    bbox = aoi_lonlat_bbox(transform, shape, epsg)
    g = gpd.read_file(E / 'fire/mtbs_perimeter_data/mtbs_perims_DD.shp',
                      bbox=bbox).to_crs(f'EPSG:{epsg}')
    g['year'] = pd.to_datetime(g.ig_date).dt.year
    out = np.zeros((len(years),) + shape, np.uint8)
    for i, y in enumerate(years):
        geoms = list(g[g.year == y].geometry)
        if geoms:
            out[i] = rasterize(geoms, out_shape=shape, transform=transform,
                               fill=0, default_value=1)
    return out, g[['event_id', 'incid_name', 'year']]


def disturbance_layers(transform, shape, epsg, years, ddir=DISTURBANCE):
    """FRAP fires, prescribed burns, harvests and salvage cuts per year"""
    def per_year(g):
        g = g[g.geometry.notna()].to_crs(f'EPSG:{epsg}')
        out = np.zeros((len(years),) + shape, np.uint8)
        for i, y in enumerate(years):
            geoms = list(g[g.year == y].geometry)
            if geoms:
                out[i] = rasterize(geoms, out_shape=shape,
                                   transform=transform, fill=0,
                                   default_value=1)
        return out

    frap = gpd.read_file(ddir / 'frap_fires.gpkg')
    frap['year'] = pd.to_numeric(frap.YEAR_, errors='coerce')
    rx = gpd.read_file(ddir / 'rx_fires.gpkg')
    rx['year'] = pd.to_numeric(rx.YEAR_, errors='coerce')
    h = gpd.read_file(ddir / 'facts_harvest.gpkg')
    h['year'] = pd.to_datetime(h.date_completed, unit='ms',
                               errors='coerce').dt.year
    h = h[~h.activity_name.str.startswith('Natural Changes', na=False)]
    salv = h.activity_name.str.startswith(SALVAGE, na=False)
    return {'frap': per_year(frap), 'rx': per_year(rx),
            'harvest': per_year(h), 'salvage': per_year(h[salv])}


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'names', multiple=True, required=True)
@click.option('--years', nargs=2, type=int, default=(2008, 2024),
              show_default=True, help='Climate years (BCMv8 ends 2024)')
@click.option('--burn-years', nargs=2, type=int, default=(2000, 2025),
              show_default=True)
@click.option('--project', default='ecopro-509818', show_default=True)
@click.option('--masks-only', is_flag=True,
              help='NLCD, terrain, burned and harvest layers only')
@click.option('--disturbance-dir', type=click.Path(path_type=Path),
              default=DISTURBANCE, show_default=True,
              help='GeoPackages of fetch_disturbance_aois.py')
def main(configfile, outputdir, names, years, burn_years, project,
         masks_only, disturbance_dir):
    if not masks_only:
        ee.Initialize(project=project,
                      opt_url='https://earthengine-highvolume.googleapis.com')
    config = load_config(configfile)
    outputdir.mkdir(parents=True, exist_ok=True)
    yrs = list(range(years[0], years[1] + 1))
    byrs = list(range(burn_years[0], burn_years[1] + 1))
    for name in names:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        epsg = aoi['epsg']
        t0 = time.time()
        static = {}
        with rasterio.open(E / 'landcover' / f'{name}_landcover.tif') as src:
            lc = src.read()
        static['nlcd2013'] = lc[0].astype(np.float32)
        static['forest'] = np.isin(lc[0], FOREST).astype(np.uint8)
        static.update(topo_layers(transform, shape, epsg))
        click.echo(f'[{name}] terrain done ({time.time() - t0:.0f}s)')
        clim = {}
        if not masks_only:
            static.update(ee_structure(transform, shape, epsg))
            click.echo(f'[{name}] structure done ({time.time() - t0:.0f}s)')
            if name == 'neon_soap_teak':
                static.update(lidar_structure(transform, shape))
                click.echo(f'[{name}] lidar done ({time.time() - t0:.0f}s)')
            clim = bcm_layers(transform, shape, epsg, yrs)
            static['bcm_cell'] = clim.pop('bcm_cell')
            for k in [k for k in clim if k.endswith('_clim')]:
                static[k] = clim.pop(k)
            click.echo(f'[{name}] climate done ({time.time() - t0:.0f}s)')
        burned, fires = burned_layers(transform, shape, epsg, byrs)
        dist = disturbance_layers(transform, shape, epsg, byrs,
                                  disturbance_dir)

        xs, ys = cell_centers(transform, shape)
        ds = xr.Dataset(
            {**{k: (('y', 'x'), v) for k, v in static.items()},
             **{k.replace('_ann', ''): (('year', 'y', 'x'),
                                        v.astype(np.float32))
                for k, v in clim.items()},
             'burned': (('burn_year', 'y', 'x'), burned),
             **{k: (('burn_year', 'y', 'x'), v) for k, v in dist.items()}},
            coords={**({'year': yrs} if clim else {}), 'burn_year': byrs,
                    'y': ys, 'x': xs},
            attrs={'crs': f'EPSG:{epsg}', 'transform': list(transform)[:6],
                   'climate': 'BCMv8 water years, nearest 270 m cell',
                   'clim_years': list(CLIM_YEARS),
                   'fires': '; '.join(f'{r.year} {r.incid_name}'
                                      for r in fires.itertuples())})
        out = outputdir / f'{name}_env.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True} for k in ds.data_vars})
        click.echo(f'wrote {out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
