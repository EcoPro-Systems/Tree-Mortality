#!/usr/bin/env python
"""
Quick look: does MASTER (WDTS Fall 2020, ORNL DAAC 1940) add anything over
HLS for the Cheng et al. 2020 dead-tree maps and the Hemming-Schroeder NEON
mortality rasters?

MASTER L1B is ~50 m swath data (HDF4, 50 bands 0.46-12.9 um) with per-pixel
lat/lon. For each line intersecting the AOI, pixels are converted to TOA
reflectance (bands 1-25; pi L / (E0 cos sza), the ER-2 flies at ~20 km)
and 11.3 um brightness temperature (inverse Planck at the effective band
center), and binned by pixel center into Cheng's 100 m cells (EPSG:5072).
Where lines overlap, each cell takes the line closest to nadir. Pixels must
fall on NLCD forest not burned (MTBS) 2012-2020 on the AOI grid (the Creek
Fire burned during the campaign).

TOA values over a single flight are fine for ranking cells, but the Fall
2020 lines were flown through Creek Fire smoke (flight-line comments), so
the visible/NIR bands are the least trustworthy.

Outputs (outputdir): master_cells.csv, univariate.csv (Spearman of each
MASTER and HLS 2020 feature with Cheng % dead and HS 2019/2021), model_cv.csv
(Cheng, 5-fold spatial block CV: hls, master, hls+master) and
master_quicklook.pdf.
"""
import os
import re
import json
import subprocess
import click
import numpy as np
import pandas as pd
import requests
import xarray as xr
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pyhdf.SD import SD
from pyproj import Transformer
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages

from util import load_config
from fetch_hls_aoi import CMR_URL, aoi_grid, aoi_lonlat_bbox, get_with_retry
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from proto_hls_vs_cheng import CHENG, FOREST_CLASSES, cheng_window
from proto_hls_vs_hs import load_hs, cells_100m, HS_YEARS, MIN_TREES

COLLECTION = 'MASTER_WDTS_SeptOct_2020_1940'
# Band numbers (1-based): TOA reflectance for VSWIR, BT for TIR
BANDS = {'green': 3, 'red': 5, 'nir': 9, 'swir16': 12, 'swir22': 22}
TIR_BAND = 48  # 11.3 um
C1, C2 = 1.191042e8, 1.4387752e4  # W um^4 m-2 sr-1, um K
HLS_UNI = ['ndvi_raw', 'ndmi_raw', 'nbr_raw', 'rgi_raw', 'swir1_raw',
           'ndvi_base', 'ndmi_base', 'nbr_base', 'rgi_base']
MIN_PX = 2
SITE_SPLIT_LON = -119.15  # SOAP west, TEAK east (neon_soap_teak)


def search(session, bbox, flight):
    params = {'short_name': COLLECTION, 'page_size': 2000,
              'bounding_box': ','.join(f'{v:.5f}' for v in bbox)}
    out = []
    for item in get_with_retry(session, CMR_URL, params).json()['items']:
        ur = item['umm']['GranuleUR']
        if ur.endswith('.hdf') and (flight is None or f'_{flight}_' in ur):
            out.append([r['URL'] for r in item['umm']['RelatedUrls']
                        if r.get('Type') == 'GET DATA'][0])
    return sorted(out)


def download(url, outdir):
    path = outdir / url.rsplit('/', 1)[1]
    if not path.exists():
        cookies = str(outdir / '.cookies')
        subprocess.run(['curl', '-sS', '-n', '-L', '-c', cookies, '-b',
                        cookies, '-o', f'{path}.part', url], check=True)
        os.replace(f'{path}.part', path)
    return path


def read_line(path):
    """Per-pixel features (1D arrays) for one MASTER L1B line"""
    f = SD(str(path))
    get = lambda k: f.select(k).get()
    lat, lon = get('PixelLatitude'), get('PixelLongitude')
    sza = np.deg2rad(get('SolarZenithAngle'))
    vza = get('SensorZenithAngle')
    cal = f.select('CalibratedData')
    scale = np.asarray(cal.attributes()['scale_factor'], np.float64)
    e0 = get('SolarSpectralIrradiance')
    wl = get('EffectiveCentralWavelength_IR_bands')
    bands = sorted(set(BANDS.values()) | {TIR_BAND})
    out = {}
    for b in bands:
        dn = cal[:, b - 1, :].astype(np.float64)
        rad = np.where(dn == -999, np.nan, dn * scale[b - 1])
        if b == TIR_BAND:
            lam = wl[b - 1]
            with np.errstate(all='ignore'):
                out['bt11'] = C2 / (lam * np.log1p(C1 / (lam ** 5 * rad)))
        else:
            name = [k for k, v in BANDS.items() if v == b][0]
            out[name] = np.pi * rad / (e0[b - 1] * np.cos(sza))
    with np.errstate(all='ignore'):
        nd = lambda a, b: (out[a] - out[b]) / (out[a] + out[b])
        out['ndvi'] = nd('nir', 'red')
        out['ndmi'] = nd('nir', 'swir16')
        out['nbr'] = nd('nir', 'swir22')
        out['swir_ratio'] = nd('swir16', 'swir22')
        out['rgi'] = out['red'] / out['green']
    out['vza'] = np.abs(vza)
    ok = (lat != -999) & (lon != -999)
    attrs = f.attributes()
    meta = dict(line=path.name, comment=attrs.get('FlightLineComment', ''),
                area=attrs.get('GeographicArea', ''))
    return lon[ok], lat[ok], {k: v[ok] for k, v in out.items()}, meta


def block_cv(d, feats, target):
    """5-fold spatial block CV (as proto_hls_vs_cheng.evaluate, single AOI)"""
    pred = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(d, groups=d.block):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05)
        m.fit(d.iloc[tr][feats], d.iloc[tr][target])
        pred[te] = m.predict(d.iloc[te][feats])
    return dict(r2=r2_score(d[target], pred),
                spearman=spearmanr(d[target], pred)[0], n=len(d))


def cheng_grid(lab, cheng_dir):
    """The Cheng 100 m window used by cells_100m for this AOI"""
    crs = lab.attrs['crs']
    xs, ys = lab.x.values, lab.y.values
    tr = Transformer.from_crs(crs, 'EPSG:5072', always_xy=True)
    bx, by = tr.transform([xs.min(), xs.max(), xs.min(), xs.max()],
                          [ys.min(), ys.min(), ys.max(), ys.max()])
    a, t5072 = cheng_window(cheng_dir / CHENG['pct_dead'],
                            (min(bx), min(by), max(bx), max(by)))
    return t5072, a.shape


def bin_lines(paths, lab, valid, t5072, shape5072):
    """MASTER features per Cheng cell (index = flat cell index); each cell
    uses the line with the smallest mean view zenith"""
    crs = lab.attrs['crs']
    tr_aoi = rasterio.Affine(*lab.attrs['transform'])
    to_aoi = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    to_5072 = Transformer.from_crs('EPSG:4326', 'EPSG:5072', always_xy=True)
    ncell = shape5072[0] * shape5072[1]
    best, metas = None, []
    for p in paths:
        lon, lat, feats, meta = read_line(p)
        metas.append(meta)
        x, y = to_aoi.transform(lon, lat)
        col, row = ~tr_aoi * (np.asarray(x), np.asarray(y))
        col, row = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = ((row >= 0) & (row < valid.shape[0]) & (col >= 0)
                  & (col < valid.shape[1]))
        keep = np.zeros_like(inside)
        keep[inside] = valid[row[inside], col[inside]]
        ex, ny = to_5072.transform(lon[keep], lat[keep])
        c5, r5 = ~t5072 * (np.asarray(ex), np.asarray(ny))
        c5, r5 = np.floor(c5).astype(int), np.floor(r5).astype(int)
        ok = ((r5 >= 0) & (r5 < shape5072[0]) & (c5 >= 0)
              & (c5 < shape5072[1]))
        idx = (r5 * shape5072[1] + c5)[ok]
        meta['n_pixels_on_aoi_forest'] = int(ok.sum())
        if not ok.any():
            continue
        n = np.bincount(idx, minlength=ncell)
        df = {}
        for k, v in feats.items():
            v = v[keep][ok]
            m = np.isfinite(v)
            with np.errstate(invalid='ignore', divide='ignore'):
                df[f'm_{k}'] = (np.bincount(idx[m], v[m], minlength=ncell)
                                / np.bincount(idx[m], minlength=ncell))
        df = pd.DataFrame(df)
        df['m_npx'] = n
        df['m_line'] = p.name
        df = df[n >= MIN_PX]
        if best is None:
            best = df
        else:
            both = best.index.intersection(df.index)
            better = both[df.loc[both, 'm_vza'].values
                          < best.loc[both, 'm_vza'].values]
            best.loc[better] = df.loc[better]
            best = pd.concat([best, df.loc[df.index.difference(best.index)]])
        print(f'  {p.name}: {int(ok.sum())} forest pixels in AOI '
              f'({meta["comment"]})')
    return best, metas


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('-a', '--aoi', 'name', default='neon_soap_teak',
              show_default=True)
@click.option('--flight', default='2190600', show_default=True,
              help='MASTER flight number to use (None for all)')
@click.option('--master-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/master/MASTER_WDTS_SeptOct_2020')
@click.option('--composites', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_composites')
@click.option('--composite-suffix', default='_naip2020_L30_w25.nc')
@click.option('--labels', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_labels')
@click.option('--landcover', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/landcover')
@click.option('--cheng-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/cheng2024/statewide_2020')
@click.option('--hs-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/hemming_schroeder2023/data/deliverables/raster'))
@click.option('--lines', multiple=True, type=int,
              help='Restrict to these line numbers within the flight')
def main(configfile, outputdir, name, flight, master_dir, composites,
         composite_suffix, labels, landcover, cheng_dir, hs_dir, lines):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(master_dir, exist_ok=True)
    aoi = config['aois'][name]
    transform, shape = aoi_grid(aoi, config['size_m'], config['resolution'])
    bbox = aoi_lonlat_bbox(transform, shape, aoi['epsg'])
    urls = search(requests.Session(), bbox,
                  None if flight == 'None' else flight)
    if lines:
        urls = [u for u in urls if int(re.search(
            r'MASTERL1B_\d+_(\d+)_', u).group(1)) in lines]
    print(f'[{name}] {len(urls)} MASTER L1B lines intersect the AOI bbox')
    paths = [download(u, master_dir) for u in urls]

    comp = xr.open_dataset(composites / f'{name}{composite_suffix}').load()
    lab = xr.open_dataset(labels / f'{name}.nc').load()
    with rasterio.open(landcover / f'{name}_landcover.tif') as ds:
        forest = np.isin(ds.read(1), FOREST_CLASSES)
    years = lab.year.values
    burned20 = lab.burned.sel(year=years[(years >= 2012)
                                         & (years <= 2020)]).any('year').values
    valid = forest & ~burned20

    t5072, shp = cheng_grid(lab, cheng_dir)
    mcells, metas = bin_lines(paths, lab, valid, t5072, shp)
    with open(outputdir / 'lines.json', 'w') as f:
        json.dump(metas, f, indent=1)

    # All forest cells for Cheng; cells with HS trees for the NEON rasters
    cells = cells_100m(name, comp, lab, forest, cheng_dir, 2020, None,
                       ~burned20)
    d = cells.join(mcells, how='inner')
    d['block'] = d.block.astype(str)
    rows_, cols_ = np.divmod(d.index.values, shp[1])
    lon, _ = Transformer.from_crs('EPSG:5072', 'EPSG:4326',
                                  always_xy=True).transform(
        t5072.c + (cols_ + 0.5) * t5072.a, t5072.f + (rows_ + 0.5) * t5072.e)
    d['site'] = np.where(np.asarray(lon) < SITE_SPLIT_LON, 'SOAP', 'TEAK')
    print(f'[{name}] {len(cells)} Cheng cells, {len(d)} with MASTER')
    hs = None
    if name == 'neon_soap_teak':
        ref, trees = load_hs(hs_dir, lab)
        hs = cells_100m(name, comp, lab, forest, cheng_dir, 2020,
                        {f'hs_{y}': ref[y] for y in HS_YEARS},
                        ~burned20 & (trees >= MIN_TREES))
        hs = hs[[f'hs_{y}' for y in HS_YEARS]].join(d, how='inner')
        hs = hs[np.isfinite(hs.hs_2019)]
        print(f'[{name}] {len(hs)} cells with HS: '
              f'{hs.site.value_counts().to_dict()}')
    d.to_csv(outputdir / 'master_cells.csv')

    # Sanity: MASTER vs HLS 2020 for the same quantities
    for m, h in [('m_nir', 'nir_raw'), ('m_ndvi', 'ndvi_raw'),
                 ('m_swir16', 'swir1_raw'), ('m_nbr', 'nbr_raw')]:
        print(f'  sanity {m} vs HLS {h}: rho '
              f'{spearmanr(d[m], d[h], nan_policy="omit")[0]:.3f}')

    mcols = [c for c in d.columns if c.startswith('m_')
             and c not in ('m_line', 'm_npx', 'm_vza')]
    groups = [('cheng_pct_dead', 'all', d)]
    for site, g in d.groupby('site'):
        groups.append(('cheng_pct_dead', site, g))
    if hs is not None:
        for t in ('hs_2019', 'hs_2021'):
            groups.append((t, 'all', hs))
            for site, g in hs.groupby('site'):
                groups.append((t, site, g))
    uni = []
    for t, site, g in groups:
        for f in mcols + HLS_UNI:
            ok = np.isfinite(g[f]) & np.isfinite(g[t])
            uni.append(dict(target=t, site=site, feature=f,
                            n=int(ok.sum()),
                            rho=spearmanr(g[f][ok], g[t][ok])[0]))
    uni = pd.DataFrame(uni)
    uni['column'] = uni.target + ' ' + uni.site
    uni.to_csv(outputdir / 'univariate.csv', index=False)
    print(uni.pivot(index='feature', columns='column',
                    values='rho').round(3).to_string())

    hls = [c for c in cells.columns if c not in
           ('cheng_pct_dead', 'aoi', 'block') and not c.startswith('hs_')]
    rows = []
    for fs, cols in [('hls2020', hls), ('master', mcols),
                     ('hls2020+master', hls + mcols)]:
        rows.append(dict(block_cv(d, cols, 'cheng_pct_dead'), features=fs))
    res = pd.DataFrame(rows)
    res.to_csv(outputdir / 'model_cv.csv', index=False)
    print(res.round(3).to_string())

    with PdfPages(outputdir / 'master_quicklook.pdf') as pdf:
        full = np.full(shp[0] * shp[1], np.nan)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, (col, title) in zip(axes.ravel(), [
                ('cheng_pct_dead', 'Cheng 2020 % dead canopy'),
                ('nir_raw', 'HLS 2020 NIR'),
                ('m_nir', 'MASTER NIR TOA (Oct 15 2020)'),
                ('m_ndmi', 'MASTER NDMI'),
                ('m_nbr', 'MASTER NBR'),
                ('m_bt11', 'MASTER 11.3 um BT (K)')]):
            a = full.copy()
            a[d.index.values] = d[col].values
            a = a.reshape(shp)
            v = np.nanpercentile(a, [2, 98])
            im = ax.imshow(a, vmin=v[0], vmax=v[1], cmap='viridis')
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'{name}: 100 m Cheng cells (forest, unburned '
                     f'2012-2020)')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'maps.png', dpi=110)
        plt.close(fig)

        u = uni.pivot(index='feature', columns='column', values='rho')
        u = u.reindex(mcols + HLS_UNI)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(u.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6,
                       aspect='auto')
        ax.set_xticks(range(u.shape[1]), u.columns, rotation=60, ha='right',
                      fontsize=8)
        ax.set_yticks(range(u.shape[0]), u.index, fontsize=8)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                ax.text(j, i, f'{u.values[i, j]:.2f}', ha='center',
                        va='center', fontsize=7)
        fig.colorbar(im, label='Spearman rho')
        ax.set_title('MASTER (m_) vs HLS 2020 single features')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'univariate.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
