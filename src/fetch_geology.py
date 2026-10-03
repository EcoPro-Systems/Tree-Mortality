#!/usr/bin/env python
"""
Bedrock substrate on the 30 m HLS AOI grids, from the USGS State Geologic
Map Compilation (SGMC; Horton et al. 2017) polygons for California, which
carry the 1:750,000 Geologic Map of California (Jennings 1977) units. The
generalized lithology (GENERALIZE) is grouped into substrates that differ in
parent material for soils:

  granitic     Igneous, intrusive (Sierra Nevada batholith granodiorite and
               quartz monzonite)
  volcanic     Igneous, volcanic (e.g. Tertiary andesitic pyroclastic and
               volcanic mudflow deposits of the northern Sierra)
  metamorphic  Metamorphic, and mixed igneous and metamorphic units
               (metasedimentary and metavolcanic rocks)
  surficial    Unconsolidated (mainly Quaternary glacial deposits)
  other        water and anything else

The class table of the units present is printed so the grouping can be
checked. At 1:750,000 the map resolves units of roughly a kilometre and
more; small volcanic caps and narrow contacts are generalized.

Output is <outputdir>/<aoi>_geology.nc with substrate (uint8 index into
SUBSTRATES, 255 outside any polygon) and unit (int16 index into the
unit_link table in the attrs).

    python fetch_geology.py ../config/hls_aois.yml $E/env \\
        -a neon_soap_teak -a sierra_nf -a stanislaus
"""
import zipfile
import click
import numpy as np
import requests
import geopandas as gpd
import xarray as xr
from pathlib import Path
from rasterio.features import rasterize

from util import load_config
from fetch_hls_aoi import aoi_grid

URL = 'https://mrdata.usgs.gov/geology/state/shp/CA.zip'
SUBSTRATES = ['granitic', 'volcanic', 'metamorphic', 'surficial', 'other']
GROUP_RULES = [
    ('granitic', 'Igneous, intrusive'),
    ('volcanic', 'Igneous, volcanic'),
    ('metamorphic', 'Metamorphic'),
    ('metamorphic', 'Igneous and Metamorphic'),
    ('surficial', 'Unconsolidated'),
]


def group_of(generalized):
    for g, prefix in GROUP_RULES:
        if str(generalized).startswith(prefix):
            return g
    return 'other'


def polygons(cache):
    shp = cache / 'CA_geol_poly.shp'
    if not shp.exists():
        cache.mkdir(parents=True, exist_ok=True)
        z = cache / 'CA.zip'
        r = requests.get(URL, timeout=(30, 600))
        r.raise_for_status()
        z.write_bytes(r.content)
        with zipfile.ZipFile(z) as f:
            f.extractall(cache)
    return gpd.read_file(shp)


@click.command()
@click.argument('configfile', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'names', multiple=True, required=True)
@click.option('--cache', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/geology', show_default=True)
def main(configfile, outputdir, names, cache):
    config = load_config(configfile)
    g = polygons(cache)
    g['substrate'] = g.GENERALIZE.map(group_of)
    units = sorted(g.UNIT_LINK.unique())
    g['unit'] = g.UNIT_LINK.map({u: i for i, u in enumerate(units)})
    for name in names:
        aoi = config['aois'][name]
        transform, shape = aoi_grid(aoi, config['size_m'],
                                    config['resolution'])
        gg = g.to_crs(f'EPSG:{aoi["epsg"]}')
        x0, y1 = transform.c, transform.f
        x1, y0 = x0 + shape[1] * transform.a, y1 + shape[0] * transform.e
        gg = gg.cx[x0:x1, y0:y1]
        sub = rasterize(((geom, SUBSTRATES.index(s)) for geom, s in
                         zip(gg.geometry, gg.substrate)), out_shape=shape,
                        transform=transform, fill=255, dtype='uint8')
        unit = rasterize(((geom, u) for geom, u in zip(gg.geometry, gg.unit)),
                         out_shape=shape, transform=transform, fill=-1,
                         dtype='int16')
        present, counts = np.unique(unit[unit >= 0], return_counts=True)
        rows = []
        for u, n in sorted(zip(present, counts), key=lambda x: -x[1]):
            r = gg[gg.unit == u].iloc[0]
            rows.append(f'{u}|{units[u]}|{r.substrate}|{r.GENERALIZE}')
            click.echo(f'[{name}] {units[u]:12s} {n / unit.size:6.1%} '
                       f'{r.substrate:11s} {r.GENERALIZE}')
        xs = transform.c + (np.arange(shape[1]) + 0.5) * transform.a
        ys = transform.f + (np.arange(shape[0]) + 0.5) * transform.e
        ds = xr.Dataset(
            {'substrate': (('y', 'x'), sub), 'unit': (('y', 'x'), unit)},
            coords={'y': ys, 'x': xs},
            attrs={'crs': f'EPSG:{aoi["epsg"]}',
                   'transform': list(transform)[:6],
                   'source': f'USGS State Geologic Map Compilation, '
                             f'California ({URL})',
                   'substrates': ','.join(SUBSTRATES),
                   'unit_table': '; '.join(rows)})
        out = outputdir / f'{name}_geology.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True} for k in ds.data_vars})
        click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
