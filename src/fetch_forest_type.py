#!/usr/bin/env python
"""
Forest type on the 30 m HLS AOI grids: LANDFIRE 2014 (LF 1.4.0) Existing
Vegetation Type (EVT) from Earth Engine, with the ecological systems grouped
into the Sierra Nevada conifer types that differ in host composition:

  pine      dry-mesic mixed conifer, lower montane conifer, Jeffrey /
            ponderosa pine, black oak-conifer (pine-dominated lower
            montane)
  mesic     mesic mixed conifer (white fir and incense cedar dominated)
  red_fir   red fir
  subalpine lodgepole pine and subalpine woodlands
  other     everything else (hardwoods, shrubs, meadows, non-vegetated)

Groups are assigned from the EVT class names (GROUP_RULES, first match).
The class table for the codes present is printed so the mapping can be
checked.

Output is <outputdir>/<aoi>_ftype.nc with lf14_evt (int16 code) and
ftype (uint8 index into FTYPES), plus the code -> name table in the attrs.

    python fetch_forest_type.py ../config/hls_aois.yml $E/env \
        -a neon_soap_teak -a sierra_nf -a stanislaus
"""
import re
import time
import click
import ee
import numpy as np
import xarray as xr
from pathlib import Path

from util import load_config
from fetch_hls_aoi import aoi_grid

FTYPES = ['pine', 'mesic', 'red_fir', 'subalpine', 'other']
GROUP_RULES = [
    ('red_fir', r'Red Fir'),
    ('subalpine', r'Lodgepole|Subalpine|Whitebark|Limber'),
    ('pine', r'Dry-Mesic Mixed Conifer|Lower Montane Conifer|Jeffrey|'
             r'Ponderosa|Oak-Conifer'),
    ('mesic', r'Mesic Mixed Conifer|White Fir'),
]


def group_of(name):
    for g, pat in GROUP_RULES:
        if re.search(pat, name):
            return g
    return 'other'


def evt_image():
    im = (ee.ImageCollection('LANDFIRE/Vegetation/EVT/v1_4_0')
          .filter(ee.Filter.eq('system:index', 'CONUS')).first())
    p = im.getInfo()['properties']
    names = dict(zip(p['EVT_class_values'], p['EVT_class_names']))
    return im.select('EVT').rename('lf14_evt').toInt16(), names


def fetch_codes(img, transform, shape, epsg, step=500):
    out = np.full(shape, -1, np.int16)
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
        out[r0:r0 + n] = arr['lf14_evt']
    return out


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'names', multiple=True, required=True)
@click.option('--project', default='ecopro-509818', show_default=True)
def main(configfile, outputdir, names, project):
    ee.Initialize(project=project,
                  opt_url='https://earthengine-highvolume.googleapis.com')
    config = load_config(configfile)
    img, table = evt_image()
    for name in names:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        codes = fetch_codes(img, transform, shape, aoi['epsg'])
        present, counts = np.unique(codes[codes >= 0], return_counts=True)
        lut = np.full(codes.max() + 1, FTYPES.index('other'), np.uint8)
        rows = []
        for c, n in sorted(zip(present, counts), key=lambda x: -x[1]):
            nm = table.get(int(c), '?')
            g = group_of(nm)
            lut[c] = FTYPES.index(g)
            rows.append(f'{c}|{g}|{nm}')
            if n / codes.size >= 0.005:
                click.echo(f'[{name}] {c:5d} {n / codes.size:6.1%} '
                           f'{g:9s} {nm}')
        ftype = np.where(codes >= 0, lut[np.maximum(codes, 0)], 255)
        xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
        ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
        ds = xr.Dataset(
            {'lf14_evt': (('y', 'x'), codes),
             'ftype': (('y', 'x'), ftype.astype(np.uint8))},
            coords={'y': ys, 'x': xs},
            attrs={'crs': f'EPSG:{aoi["epsg"]}',
                   'transform': list(transform)[:6],
                   'source': 'LANDFIRE LF 1.4.0 EVT (Earth Engine '
                             'LANDFIRE/Vegetation/EVT/v1_4_0)',
                   'ftypes': ','.join(FTYPES),
                   'evt_table': '; '.join(rows)})
        out = outputdir / f'{name}_ftype.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True} for k in ds.data_vars})
        click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
