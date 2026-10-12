#!/usr/bin/env python
"""
Cell masks of the held-out sites inside the AVIRIS-C Yosemite flight box.

Two of the held-out sites are not whole AOI grids: SEKI (Sequoia and Kings
Canyon National Parks) and the rest of the Yosemite box outside the
neon_soap_teak and sierra_nf AOIs. Both are defined on the 30 m UTM 11
lattice of the trait mosaics:
  footprint      flight_id > 0 in the June 12 2013 (_v2) and June 22 2018
                 trait mosaics (ORNL DAAC 2403), the dates used for the
                 2013 and 2018 traits (fetch_wdts_traits.DATES)
  seki           footprint inside the NPS boundary of SEQU or KICA
  yosemite_rest  footprint outside SEKI and outside the neon_soap_teak and
                 sierra_nf AOI grids
Writes env/<aoi>_mask.nc (keep, uint8, plus the footprint layers) for each
AOI in config/heldout_aois.yml, and caches the park boundary as
geom/seki_boundary.geojson. Forest, fire and harvest masks come later from
the env layers (proto_heldout_strata.py).

    python fetch_heldout_masks.py ../config/heldout_aois.yml $E/env
"""
import click
import geopandas as gpd
import numpy as np
import requests
import xarray as xr
from pathlib import Path
from rasterio.features import rasterize

from util import load_config
from fetch_hls_aoi import aoi_grid, aoi_lonlat_bbox
from fetch_wdts_traits import search, read_bands, gdal_env, NODATA
import response_common as rc

NPS_URL = ('https://services1.arcgis.com/fBc8EJBxQRMcHlei/arcgis/rest/'
           'services/NPS_Land_Resources_Division_Boundary_and_Tract_Data_'
           'Service/FeatureServer/2/query')
UNITS = ('SEQU', 'KICA')
DATES = ('20130612_v2', '20180622')
PILOTS = ('neon_soap_teak', 'sierra_nf')
FLIGHT_ID = 12  # band of the trait mosaics


def seki_boundary(path):
    if not path.exists():
        r = requests.get(NPS_URL, params={
            'where': 'UNIT_CODE IN (' + ','.join(f"'{u}'" for u in UNITS) +
            ')', 'outFields': 'UNIT_CODE,UNIT_NAME', 'outSR': 4326,
            'f': 'geojson'}, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    return gpd.read_file(path)


def footprint(transform, shape, epsg, env):
    found = search(requests.Session(),
                   aoi_lonlat_bbox(transform, shape, epsg))
    out = {}
    for date in DATES:
        a = read_bands(found[('yosemite', date, 'LMA')], transform, shape,
                       env, [FLIGHT_ID], epsg)[0]
        out[date] = (a != NODATA) & (a > 0)
    return out


def box_mask(name, transform, shape):
    """Pixels of this grid inside another AOI's grid (same lattice)"""
    t, s, _ = rc.aoi_info(name)
    x0, y1 = t.c, t.f
    x1, y0 = x0 + s[1] * t.a, y1 + s[0] * t.e
    xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
    ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
    return ((ys >= y0) & (ys < y1))[:, None] & \
        ((xs >= x0) & (xs < x1))[None, :]


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'names', multiple=True,
              default=['seki', 'yosemite_rest'], show_default=True)
def main(configfile, outputdir, names):
    config = load_config(configfile)
    env = gdal_env(str(outputdir / '.cookies.txt'))
    parks = seki_boundary(rc.E / 'geom' / 'seki_boundary.geojson')
    for name in names:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        epsg = aoi['epsg']
        fp = footprint(transform, shape, epsg, env)
        both = fp[DATES[0]] & fp[DATES[1]]
        g = parks.to_crs(epsg)
        park = rasterize([(x, 1) for x in g.geometry], out_shape=shape,
                         transform=transform).astype(bool)
        if name == 'seki':
            keep = both & park
        else:
            keep = both & ~park
            for p in PILOTS:
                keep &= ~box_mask(p, transform, shape)
        ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
        xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
        ds = xr.Dataset(
            {'keep': (('y', 'x'), keep.astype(np.uint8)),
             **{f'footprint_{d[:8]}': (('y', 'x'), fp[d].astype(np.uint8))
                for d in DATES},
             'park': (('y', 'x'), park.astype(np.uint8))},
            coords={'y': ys, 'x': xs},
            attrs={'crs': f'EPSG:{epsg}', 'transform': list(transform)[:6],
                   'definition': ' '.join(__doc__.split('\n')[6:16])})
        out = outputdir / f'{name}_mask.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True}
                                    for k in ds.data_vars})
        km2 = (rc.RES / 1000) ** 2
        click.echo(f'[{name}] {shape[1]} x {shape[0]} px; footprint 2013 '
                   f'{fp[DATES[0]].sum() * km2:.0f} km2, 2018 '
                   f'{fp[DATES[1]].sum() * km2:.0f} km2, both '
                   f'{both.sum() * km2:.0f} km2; kept '
                   f'{keep.sum() * km2:.0f} km2 -> {out}')


if __name__ == '__main__':
    main()
