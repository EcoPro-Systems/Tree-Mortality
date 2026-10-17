#!/usr/bin/env python
"""
Dry-season canopy water content (CWC) for California from Brodrick et al.
(2019, Geophysical Research Letters, doi:10.1029/2018GL081108), PANGAEA
doi:10.1594/PANGAEA.897276 (CC BY 4.0).

One statewide GeoTIFF per year (July-August, 1990-2017 without 2012), 30 m,
UTM 10N, mL water per m², nodata -9999, about 1.9 GB each. PANGAEA serves
them from tape: a request fails (503, sometimes 500) until the file is
staged, so the download retries until it starts. Files go to OUTPUTDIR/raw/.

Per AOI, the maps are averaged onto the 30 m AOI grid and written to
OUTPUTDIR/<aoi>_cwc.nc:
  cwc_<year>        CWC (mL/m²) for each year fetched
  pc_cwc_<a><b>     percent change from year a to b, 100 (b - a) / a
                    (Brodrick and Asner 2017's progressive water stress
                    form; Queally et al. 2025 used 2011 -> 2014)
The PANGAEA record is saved as record.json.

    python fetch_brodrick2019.py $E/brodrick2019 -a neon_soap_teak -a sierra_nf
"""
import json
import time
import click
import numpy as np
import requests
import xarray as xr
from pathlib import Path

import response_common as rc
from aoi_env_layers import cell_centers
from fetch_lidar_structure import read_window, to_grid

DOI = 'https://doi.pangaea.de/10.1594/PANGAEA.897276'
URL = 'https://hs.pangaea.de/Maps/CanopyWaterContent/cwc_{}.tif'
YEARS = (2011, 2013, 2014, 2015)
CHANGES = ((2011, 2013), (2011, 2014), (2011, 2015))
NODATA = -9999


def download(url, out, wait_max=7200):
    """Stream url to out, retrying while PANGAEA stages it from tape"""
    if out.exists():
        return
    part = out.with_suffix('.part')
    t0 = time.time()
    while True:
        with requests.get(url, stream=True, timeout=300) as r:
            if r.status_code >= 500 and time.time() - t0 < wait_max:
                time.sleep(int(r.headers.get('retry-after', 30)) + 30)
                continue
            r.raise_for_status()
            n = int(r.headers.get('Content-Length', 0))
            with open(part, 'wb') as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        if n and part.stat().st_size != n:
            raise IOError(f'{url}: {part.stat().st_size} of {n} bytes')
        part.rename(out)
        return


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
def main(outputdir, aois):
    raw = outputdir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    r = requests.get(DOI, params={'format': 'metadata_jsonld'}, timeout=60)
    r.raise_for_status()
    (outputdir / 'record.json').write_text(json.dumps(r.json(), indent=1))
    for y in YEARS:
        click.echo(f'cwc_{y}')
        download(URL.format(y), raw / f'cwc_{y}.tif')
    for aoi in aois:
        transform, shape, epsg = rc.aoi_info(aoi)
        layers = {}
        for y in YEARS:
            a, tr, crs = read_window(raw / f'cwc_{y}.tif', transform, shape,
                                     epsg)
            a[(a == NODATA) | (a < 0)] = np.nan
            layers[f'cwc_{y}'] = to_grid(a, tr, crs, transform, shape, epsg)
            click.echo(f'  [{aoi}] cwc_{y}: '
                       f'{np.isfinite(layers[f"cwc_{y}"]).mean():.1%} of AOI, '
                       f'median {np.nanmedian(layers[f"cwc_{y}"]):.0f} mL/m²')
        with np.errstate(all='ignore'):
            for a, b in CHANGES:
                c0, c1 = layers[f'cwc_{a}'], layers[f'cwc_{b}']
                layers[f'pc_cwc_{a % 100}{b % 100}'] = np.where(
                    c0 > 0, 100 * (c1 - c0) / c0, np.nan).astype(np.float32)
        xs, ys = cell_centers(transform, shape)
        ds = xr.Dataset({k: (('y', 'x'), v) for k, v in layers.items()},
                        coords={'y': ys, 'x': xs},
                        attrs={'crs': f'EPSG:{epsg}',
                               'transform': list(transform)[:6],
                               'source': 'Brodrick et al. 2019, PANGAEA '
                                         '897276', 'units': 'mL/m2'})
        out = outputdir / f'{aoi}_cwc.nc'
        ds.to_netcdf(out, encoding={k: {'zlib': True} for k in ds.data_vars})
        click.echo(f'wrote {out}')


if __name__ == '__main__':
    main()
