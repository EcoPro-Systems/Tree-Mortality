#!/usr/bin/env python
"""
Annual 30 m GEDI-fused canopy structure maps of California on the 30 m AOI
grids. Two unpublished products, both EPSG:4326 at 0.000269495 deg:

  AEF   GEDI L2/L4 footprints extrapolated with AlphaEarth Foundations
        embeddings by gradient boosting, 2018-2025:
        maps/<year>/Final[/Compressed]/CA_{rh98,cover,agbd}_<year>_v004_*.tif
        band 1 the estimate, band 2 its uncertainty; rh98 in cm and cover in
        % x 100 (Int16, nodata -32768), agbd in Mg/ha
  LSM   GEDI extrapolated with a Landsat model, 2015-2020 and 2022:
        landsat_model/ml_landsat_30.0m_<year>_uspacific.tif, band 1 (canopy
        height, m; the variable is not documented), band 2 alpha (0 =
        no data)

The source files are stored in one-row strips, so only the rows over the
AOI (plus a margin) are read; they are then averaged onto the AOI grid.
The .aux.xml statistics beside them are stale and not used.

Output <outdir>/<aoi>_gedi_fusion.nc (default $E/structure):
  rh98_aef_<y>, cover_aef_<y>, agbd_aef_<y>   (m, %, Mg/ha)
  rh98_aef_unc_<y>, cover_aef_unc_<y>, agbd_aef_unc_<y>
  h_lsm_<y>                                   (m)
and record.json with the source paths and sizes.

    python fetch_gedi_fusion.py --src <maps dir> -a neon_soap_teak \
        -a sierra_nf -a stanislaus
"""
import json
import click
import numpy as np
import rasterio
import xarray as xr
from pathlib import Path
from pyproj import Transformer
from rasterio.windows import from_bounds

import response_common as rc
from aoi_env_layers import cell_centers
from fetch_lidar_structure import aoi_bounds, to_grid

AEF_YEARS = range(2018, 2026)
LSM_YEARS = (2015, 2016, 2017, 2018, 2019, 2020, 2022)
AEF_VARS = {'rh98': 0.01, 'cover': 0.01, 'agbd': 1.0}  # scale to m, %, Mg/ha
MARGIN_DEG = 0.01


def aef_path(src, v, y):
    d = src / 'maps' / str(y) / 'Final'
    hits = sorted(d.glob(f'Compressed/CA_{v}_{y}_v004_*.tif')) or \
        sorted(d.glob(f'CA_{v}_{y}_v004_*.tif'))
    return hits[0] if hits else None


def read_bands(path, transform, shape, epsg):
    """Bands 1-2 over the AOI (geographic window plus a margin), as float
    with nodata as NaN; returns (bands, window transform, crs)"""
    b = aoi_bounds(transform, shape)
    tr = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    xs, ys = tr.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    bb = (min(xs) - MARGIN_DEG, min(ys) - MARGIN_DEG,
          max(xs) + MARGIN_DEG, max(ys) + MARGIN_DEG)
    with rasterio.open(path) as s:
        w = from_bounds(*bb, s.transform).round_offsets().round_lengths()
        a = s.read([1, 2], window=w, boundless=True, masked=True)
        a = a.astype(np.float32).filled(np.nan)
        return a, s.window_transform(w), s.crs


@click.command()
@click.option('--src', type=click.Path(path_type=Path, exists=True),
              required=True, help='Directory holding maps/ and '
                                  'landsat_model/')
@click.option('-o', '--outdir', type=click.Path(path_type=Path),
              default=rc.E / 'structure', show_default=True)
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
def main(src, outdir, aois):
    outdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for y in AEF_YEARS:
        for v in AEF_VARS:
            files[f'{v}_aef_{y}'] = aef_path(src, v, y)
    for y in LSM_YEARS:
        p = src / 'landsat_model' / f'ml_landsat_30.0m_{y}_uspacific.tif'
        files[f'h_lsm_{y}'] = p if p.exists() else None
    (outdir / 'record.json').write_text(json.dumps(
        {k: dict(path=str(p.relative_to(src)), bytes=p.stat().st_size)
         for k, p in files.items() if p}, indent=1))
    for aoi in aois:
        transform, shape, epsg = rc.aoi_info(aoi)
        layers = {}
        for name, p in files.items():
            if p is None:
                click.echo(f'  [{aoi}] {name}: no file')
                continue
            a, wt, crs = read_bands(p, transform, shape, epsg)
            if name.startswith('h_lsm'):
                est, unc = np.where(a[1] > 0, a[0], np.nan), None
            else:
                f = AEF_VARS[name.split('_')[0]]
                est, unc = a[0] * f, a[1] * f
            layers[name] = to_grid(est, wt, crs, transform, shape, epsg)
            if unc is not None:
                u = name.replace('_aef_', '_aef_unc_')
                layers[u] = to_grid(unc, wt, crs, transform, shape, epsg)
            click.echo(f'  [{aoi}] {name}: {np.isfinite(layers[name]).mean():.1%}'
                       f' of AOI, median {np.nanmedian(layers[name]):.1f}')
        xs, ys = cell_centers(transform, shape)
        ds = xr.Dataset({k: (('y', 'x'), v) for k, v in layers.items()},
                        coords={'y': ys, 'x': xs},
                        attrs={'crs': f'EPSG:{epsg}',
                               'transform': list(transform)[:6],
                               'units': 'rh98/h_lsm m, cover %, agbd Mg/ha'})
        out = outdir / f'{aoi}_gedi_fusion.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True} for k in ds.data_vars})
        click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
