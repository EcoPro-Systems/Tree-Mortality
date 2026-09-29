#!/usr/bin/env python
"""
Per-year, per-pixel median composites of Landsat Collection 2 Level-2
surface reflectance (Landsat 5 TM, 7 ETM+, 8/9 OLI) on the HLS AOI grids,
computed in Google Earth Engine and pulled with ee.data.computePixels.

HLS starts in 2013, so this supplies the pre-drought (2008-2011) baseline
and a sensor record that runs continuously through the 2012-2016 drought
and its recovery.

- Masking: QA_PIXEL fill, dilated cloud, cirrus, cloud, shadow, snow and
  water bits, any saturated band (QA_RADSAT), reflectance outside [0, 1],
  and haze/smoke: SR_ATMOS_OPACITY > 0.3 (TM/ETM+) or a "high" SR_QA_AEROSOL
  level (OLI). Wildfire smoke (e.g. the 2015 Rough Fire next to SOAP/TEAK)
  otherwise depresses NIRv in single-sensor composites.
- Scaling: SR * 2.75e-5 - 0.2.
- Harmonization: TM/ETM+ reflectance is mapped to OLI with the Roy et al.
  (2016) OLS coefficients (doi:10.1016/j.rse.2015.12.024) so the record is
  comparable to HLS L30 and Landsat 8/9. TM is treated as ETM+.
- Resampling: bilinear onto the AOI's 30 m UTM grid (C2 pixels are offset
  15 m from the HLS lattice).

Variants:
    c2     all sensors (5, 7, 8, 9)
    c2l7   Landsat 7 only (through 2022): one sensor across the 2008-2011
           baseline, the 2012-2016 drought and the recovery. Harmonized L7
           NDMI still runs ~0.015 below the L7+L8 mix in 2013-2019, so
           metrics that span the L5/L7 -> L8 transition should use this.
    c2oli  Landsat 8/9 only (2013 onward), for the 2017-2025 period

Output is <outputdir>/<aoi>_<variant>_doy<start>-<end>.nc with dims
(year, y, x), matching hls_annual_composites.py: the six bands, ndvi, ndmi,
nbr, nirv (= ndvi * nir) and n_clear.

    python fetch_landsat_c2_ee.py ../config/hls_aois.yml $E/landsat_composites \
        -a neon_soap_teak -w 182 273 -w 145 190
"""
import time
import click
import numpy as np
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
from util import load_config
from fetch_hls_aoi import aoi_grid

ROLES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
INDICES = ['ndvi', 'ndmi', 'nbr', 'nirv']
OUT_VARS = ROLES + INDICES + ['n_clear']
SENSORS = {
    'LT05': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
    'LE07': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
    'LC08': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    'LC09': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
}
VARIANTS = {'c2': ['LT05', 'LE07', 'LC08', 'LC09'], 'c2l7': ['LE07'],
            'c2oli': ['LC08', 'LC09']}
# Years with Jul-Sep data; Landsat 7 was decommissioned in 2024
SENSOR_YEARS = {'LT05': (1984, 2011), 'LE07': (1999, 2022),
                'LC08': (2013, 2100), 'LC09': (2022, 2100)}
# Roy et al. (2016) Table 2 OLS, ETM+ -> OLI: OLI = a + b * ETM+
ROY_A = [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]
ROY_B = [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
# QA_PIXEL: fill, dilated cloud, cirrus, cloud, shadow, snow, water
QA_BAD = sum(1 << b for b in (0, 1, 2, 3, 4, 5, 7))
MAX_OPACITY = 0.3  # TM/ETM+ SR_ATMOS_OPACITY above this is hazy
MAX_BYTES = 40_000_000  # computePixels limit is 48 MB per request


def prepare(sensor):
    def fn(img):
        qa = img.select('QA_PIXEL')
        ok = qa.bitwiseAnd(QA_BAD).eq(0).And(
            img.select('QA_RADSAT').eq(0))
        # Haze and smoke, which the cloud bits miss
        if sensor in ('LT05', 'LE07'):
            ok = ok.And(img.select('SR_ATMOS_OPACITY').multiply(0.001)
                        .lte(MAX_OPACITY))
        else:
            ok = ok.And(img.select('SR_QA_AEROSOL').rightShift(6)
                        .bitwiseAnd(3).neq(3))
        sr = (img.select(SENSORS[sensor], ROLES)
              .multiply(2.75e-5).add(-0.2))
        if sensor in ('LT05', 'LE07'):
            sr = sr.multiply(ee.Image.constant(ROY_B)).add(
                ee.Image.constant(ROY_A)).rename(ROLES)
        ok = ok.And(sr.reduce(ee.Reducer.min()).gte(0)).And(
            sr.reduce(ee.Reducer.max()).lte(1))
        ndvi = sr.normalizedDifference(['nir', 'red']).rename('ndvi')
        ndmi = sr.normalizedDifference(['nir', 'swir1']).rename('ndmi')
        nbr = sr.normalizedDifference(['nir', 'swir2']).rename('nbr')
        nirv = ndvi.multiply(sr.select('nir')).rename('nirv')
        out = sr.addBands([ndvi, ndmi, nbr, nirv])
        return out.toFloat().resample('bilinear').updateMask(ok)
    return fn


def composite(sensors, year, doy, region):
    col = None
    for s in sensors:
        if not SENSOR_YEARS[s][0] <= year <= SENSOR_YEARS[s][1]:
            continue
        c = (ee.ImageCollection(f'LANDSAT/{s}/C02/T1_L2')
             .filterBounds(region)
             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
             .filter(ee.Filter.calendarRange(doy[0], doy[1], 'day_of_year'))
             .map(prepare(s)))
        col = c if col is None else col.merge(c)
    med = col.median()
    n = col.select('nir').count().rename('n_clear')
    return med.select(ROLES + INDICES).addBands(n).toFloat()


def fetch_rows(img, transform, crs, r0, nrows, ncols):
    req = {
        'expression': img,
        'fileFormat': 'NUMPY_NDARRAY',
        'grid': {
            'dimensions': {'width': ncols, 'height': nrows},
            'affineTransform': {
                'scaleX': transform.a, 'shearX': 0, 'translateX': transform.c,
                'shearY': 0, 'scaleY': transform.e,
                'translateY': transform.f + r0 * transform.e},
            'crsCode': crs,
        },
    }
    for attempt in range(6):
        try:
            arr = ee.data.computePixels(req)
            break
        except ee.EEException as exc:
            retry = any(s in str(exc) for s in
                        ('Too Many Requests', 'Internal', 'timed out'))
            if not retry or attempt == 5:
                raise
            time.sleep(2 ** attempt)
    return {v: np.asarray(arr[v], dtype=np.float32) for v in OUT_VARS}


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'names', multiple=True,
              help='AOI name(s); default all in the config')
@click.option('-w', '--window', 'windows', multiple=True, nargs=2, type=int,
              default=[(182, 273)], show_default=True,
              help='DOY window (inclusive); repeatable')
@click.option('-v', '--variant', 'variants', multiple=True,
              type=click.Choice(list(VARIANTS)),
              default=['c2', 'c2l7', 'c2oli'],
              show_default=True)
@click.option('--years', nargs=2, type=int, default=(2008, 2025),
              show_default=True)
@click.option('--project', default='ecopro-509818', show_default=True)
@click.option('-j', '--jobs', default=6, show_default=True)
@click.option('--overwrite', is_flag=True)
def main(configfile, outputdir, names, windows, variants, years, project,
         jobs, overwrite):
    ee.Initialize(project=project,
                  opt_url='https://earthengine-highvolume.googleapis.com')
    config = load_config(configfile)
    outputdir.mkdir(parents=True, exist_ok=True)
    yrs = list(range(years[0], years[1] + 1))
    for name in (names or list(config['aois'])):
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        crs = f'EPSG:{aoi["epsg"]}'
        x0, y1 = transform.c, transform.f
        x1, y0 = x0 + shape[1] * transform.a, y1 + shape[0] * transform.e
        region = ee.Geometry.Rectangle([x0, y0, x1, y1], crs, False)
        chunk = max(1, MAX_BYTES // (4 * len(OUT_VARS) * shape[1]))
        for variant in variants:
            for doy in windows:
                outfile = (outputdir /
                           f'{name}_{variant}_doy{doy[0]}-{doy[1]}.nc')
                if outfile.exists() and not overwrite:
                    click.echo(f'{outfile.name} exists; skipping')
                    continue
                out = {v: np.full((len(yrs),) + shape, np.nan, np.float32)
                       for v in OUT_VARS}
                jobs_ = [(yi, y, r0) for yi, y in enumerate(yrs)
                         for r0 in range(0, shape[0], chunk)
                         if any(SENSOR_YEARS[s][0] <= y <= SENSOR_YEARS[s][1]
                                for s in VARIANTS[variant])]
                t0 = time.time()
                with ThreadPoolExecutor(jobs) as pool:
                    futs = {}
                    for yi, y, r0 in jobs_:
                        img = composite(VARIANTS[variant], y, doy, region)
                        n = min(chunk, shape[0] - r0)
                        futs[pool.submit(fetch_rows, img, transform, crs,
                                         r0, n, shape[1])] = (yi, r0, n)
                    for f in as_completed(futs):
                        yi, r0, n = futs[f]
                        arrs = f.result()
                        for v in OUT_VARS:
                            out[v][yi, r0:r0 + n] = arrs[v]
                out['n_clear'] = np.nan_to_num(out['n_clear']).astype(
                    np.int16)
                xs = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
                ys = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
                ds = xr.Dataset(
                    {v: (('year', 'y', 'x'), out[v]) for v in OUT_VARS},
                    coords={'year': yrs, 'y': ys, 'x': xs},
                    attrs={'crs': crs, 'transform': list(transform)[:6],
                           'doy_start': doy[0], 'doy_end': doy[1],
                           'sensors': ','.join(VARIANTS[variant]),
                           'harmonization': 'Roy et al. 2016 ETM+->OLI'},
                )
                enc = {v: {'zlib': True} for v in ds.data_vars}
                ds.to_netcdf(outfile, encoding=enc)
                click.echo(f'{outfile.name}: {len(jobs_)} requests in '
                           f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
