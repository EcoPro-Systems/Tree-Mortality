#!/usr/bin/env python
"""
Multi-year test of HLS/Landsat against the Hemming-Schroeder et al. (2023)
lidar/multispectral 30 m tree-mortality rasters at NEON SOAP and TEAK
(cumulative fraction of trees dead in 2013, 2017, 2018, 2019, 2021; Zenodo
10.5281/zenodo.7938442).

HS rasters are on the Landsat Collection 2 grid (15 m offset from the HLS
MGRS lattice) and are resampled onto the neon_soap_teak AOI grid with
area-weighted averaging. Pixels are restricted to NLCD forest, >= MIN_TREES
trees per 30 m pixel (HS trees_per_pixel), and not burned (MTBS) 2012..Y.

HLS features are "year-relative" so that one model can be applied to any
year Y: for each band/index X (L30 composites matched to the 2020 NAIP date
at every year; see hls_naip_composites.py)
  raw     X(Y)
  base    X(Y) - X(2013)
  d1..d3  X(Y) - X(Y-k)
  cmin/cmax  min/max over 2014..Y of X(y) - X(2013)

Part 1 (NEON only): HistGradientBoosting on cumulative dead fraction (and
new mortality between reference years), evaluated leave-one-year-out with
3 km spatial blocks also withheld, at 30/90/270 m.

Part 2 (transfer): train the same year-relative model on the Cheng et al.
2020 % dead canopy (100 m) in the other AOIs at Y=2020, apply it to NEON at
Y = 2017, 2018, 2019, 2021 and compare (Spearman) with HS aggregated to the
same 100 m cells. Also reports Cheng 2020 vs HS 2019/2021 directly, as the
agreement between the two references.
"""
import os
import click
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from rasterio.warp import reproject, Resampling
from sklearn.metrics import r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.backends.backend_pdf import PdfPages

from util import load_config
from hls_annual_composites import ROLES, INDICES
from proto_hls_vs_cheng import (
    CHENG, FOREST_CLASSES, cheng_window, cell_index, aggregate,
)
from proto_hls_mortality_signal import block_sum

AOI = 'neon_soap_teak'
HS_YEARS = [2013, 2017, 2018, 2019, 2021]
TARGET_YEARS = [2017, 2018, 2019, 2021]
BASE_YEAR = 2013
MIN_TREES = 3
BLOCK_M = 3000
SCALES = [1, 3, 9]
VARIABLES = ROLES + INDICES


def year_features(comp, year, variables=VARIABLES, base_year=BASE_YEAR):
    """Year-relative feature arrays (name -> 2D) for year Y"""
    yrs = set(comp.year.values.tolist())
    out = {}
    for v in variables:
        X = lambda y: comp[v].sel(year=y).values
        base = X(base_year)
        out[f'{v}_raw'] = X(year)
        out[f'{v}_base'] = X(year) - base
        for k in (1, 2, 3):
            out[f'{v}_d{k}'] = (X(year) - X(year - k)
                                if year - k in yrs else np.nan * base)
        hist = np.stack([X(y) - base for y in range(base_year + 1, year + 1)])
        with np.errstate(all='ignore'):
            out[f'{v}_cmin'] = np.nanmin(hist, axis=0)
            out[f'{v}_cmax'] = np.nanmax(hist, axis=0)
    return out


def load_hs(hsdir, lab):
    """HS mortality (year -> 2D) and trees per pixel on the AOI grid"""
    dst_tr = rasterio.Affine(*lab.attrs['transform'])
    shape = (lab.sizes['y'], lab.sizes['x'])
    crs = lab.attrs['crs']

    def warp(path):
        with rasterio.open(path) as ds:
            src = ds.read(1).astype(np.float32)
            src[src == ds.nodata] = np.nan
            dst = np.full(shape, np.nan, np.float32)
            reproject(src, dst, src_transform=ds.transform, src_crs=ds.crs,
                      src_nodata=np.nan, dst_transform=dst_tr, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.average)
            return dst

    ref = {}
    for y in HS_YEARS:
        arrs = [warp(hsdir / f'{site}_mortality_{y}.tif')
                for site in ('soap', 'teak')]
        ref[y] = np.where(np.isfinite(arrs[0]), arrs[0], arrs[1])
    trees = warp(hsdir / 'trees_per_pixel.tif')
    return ref, trees


def block_mean(a, valid, k):
    ok = valid & np.isfinite(a)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (block_sum(np.where(ok, a, 0), k) / block_sum(ok, k),
                block_sum(ok, k))


def part1(comp, lab, forest, ref, trees, res):
    """Leave-one-year-out (+ spatial blocks) at several scales"""
    burned_cum = np.cumsum(lab.burned.values, axis=0) > 0
    lab_years = lab.year.values.tolist()
    rows, uni = [], []
    ny, nx = forest.shape
    for k in SCALES:
        frames = []
        for Y in TARGET_YEARS:
            valid = (forest & (trees >= MIN_TREES) & np.isfinite(ref[Y])
                     & ~burned_cum[lab_years.index(Y)])
            prev = HS_YEARS[HS_YEARS.index(Y) - 1]
            tgt, n = block_mean(ref[Y], valid, k)
            new, _ = block_mean(ref[Y] - ref[prev], valid, k)
            keep = n >= 0.7 * k * k
            feats = {f: block_mean(a, valid, k)[0][keep]
                     for f, a in year_features(comp, Y).items()}
            df = pd.DataFrame(feats)
            df['cum'], df['new'] = tgt[keep], new[keep]
            rr, cc = np.nonzero(keep)
            bs = max(1, int(BLOCK_M / (res * k)))
            df['block'] = (rr // bs) * 10000 + (cc // bs)
            df['year'] = Y
            frames.append(df)
            if k == 1:
                for f in ['ndmi_base', 'nbr_base', 'rgi_base', 'ndmi_cmin',
                          'rgi_cmax', 'ndmi_d1', 'rgi_d1']:
                    ok = np.isfinite(df[f])
                    uni.append(dict(year=Y, feature=f,
                                    rho_cum=spearmanr(df[f][ok],
                                                      df.cum[ok])[0],
                                    rho_new=spearmanr(df[f][ok],
                                                      df.new[ok])[0]))
        d = pd.concat(frames, ignore_index=True)
        feat_cols = [c for c in d.columns
                     if c not in ('cum', 'new', 'block', 'year')]
        blocks = d.block.unique()
        rng = np.random.default_rng(0)
        fold = dict(zip(blocks, rng.integers(0, 5, len(blocks))))
        d['fold'] = d.block.map(fold)
        for target in ['cum', 'new']:
            dt = d[np.isfinite(d[target])]
            for Y in TARGET_YEARS:
                pred = pd.Series(np.nan, index=dt.index)
                for f in range(5):
                    tr = dt[(dt.year != Y) & (dt.fold != f)]
                    te = dt[(dt.year == Y) & (dt.fold == f)]
                    m = HistGradientBoostingRegressor(
                        max_iter=300, learning_rate=0.05)
                    m.fit(tr[feat_cols], tr[target])
                    pred[te.index] = m.predict(te[feat_cols])
                te = dt[dt.year == Y]
                rows.append(dict(scale_m=res * k, target=target, year=Y,
                                 n=len(te),
                                 r2=r2_score(te[target], pred[te.index]),
                                 rho=spearmanr(te[target],
                                               pred[te.index])[0]))
        print(f'part1 scale {res * k} m done')
    return pd.DataFrame(rows), pd.DataFrame(uni)


def cells_100m(name, comp, lab, forest, cheng_dir, year, extra=None,
               extra_valid=None):
    """Aggregate year-relative features (and optional extra 30 m layers) to
    Cheng 100 m cells; return DataFrame of kept cells"""
    from pyproj import Transformer
    crs = lab.attrs['crs']
    xs, ys = lab.x.values, lab.y.values
    tr = Transformer.from_crs(crs, 'EPSG:5072', always_xy=True)
    bx, by = tr.transform([xs.min(), xs.max(), xs.min(), xs.max()],
                          [ys.min(), ys.min(), ys.max(), ys.max()])
    bounds = (min(bx), min(by), max(bx), max(by))
    cheng, t5072 = cheng_window(cheng_dir / CHENG['pct_dead'], bounds)
    idx = cell_index(xs, ys, crs, t5072, cheng.shape)
    ncell = cheng.size
    lab_years = lab.year.values.tolist()
    burned = (np.cumsum(lab.burned.values, axis=0) > 0)[
        lab_years.index(min(year, 2019))]
    valid = forest & ~burned
    if extra_valid is not None:
        valid &= extra_valid
    total = np.bincount(idx[idx >= 0], minlength=ncell)
    nvalid = np.bincount(idx[valid & (idx >= 0)], minlength=ncell)
    keep = (total >= 6) & (nvalid >= 0.7 * total)
    df = pd.DataFrame({f: aggregate(a, valid, idx, ncell)
                       for f, a in year_features(comp, year).items()})
    df['cheng_pct_dead'] = cheng.ravel()
    for k, a in (extra or {}).items():
        df[k] = aggregate(a, valid, idx, ncell)
    df['aoi'] = name
    df['block'] = (np.arange(ncell) // cheng.shape[1] // 30) * 1000 + \
        (np.arange(ncell) % cheng.shape[1]) // 30
    return df[keep & np.isfinite(df.cheng_pct_dead.values)].copy()


def part2(config, composites, labels, landcover, cheng_dir, ref, trees,
          suffix):
    train = []
    for name in config['aois']:
        if name == AOI:
            continue
        comp = xr.open_dataset(composites / f'{name}{suffix}').load()
        lab = xr.open_dataset(labels / f'{name}.nc').load()
        with rasterio.open(landcover / f'{name}_landcover.tif') as ds:
            forest = np.isin(ds.read(1), FOREST_CLASSES)
        train.append(cells_100m(name, comp, lab, forest, cheng_dir, 2020))
    train = pd.concat(train, ignore_index=True)
    feat_cols = [c for c in train.columns
                 if c not in ('cheng_pct_dead', 'aoi', 'block')]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05)
    model.fit(train[feat_cols], train.cheng_pct_dead)

    comp = xr.open_dataset(composites / f'{AOI}{suffix}').load()
    lab = xr.open_dataset(labels / f'{AOI}.nc').load()
    with rasterio.open(landcover / f'{AOI}_landcover.tif') as ds:
        forest = np.isin(ds.read(1), FOREST_CLASSES)
    rows, frames = [], []
    for Y in TARGET_YEARS + [2020]:
        hsv = trees >= MIN_TREES
        extra = {f'hs_{y}': ref[y] for y in HS_YEARS}
        d = cells_100m(AOI, comp, lab, forest, cheng_dir, Y, extra, hsv)
        d = d[np.isfinite(d['hs_2019'])]
        d['pred'] = model.predict(d[feat_cols])
        d['Y'] = Y
        frames.append(d)
        for y_ref in ([Y] if Y in HS_YEARS else [2019, 2021]):
            ok = np.isfinite(d[f'hs_{y_ref}'])
            rows.append(dict(
                comparison=f'Landsat model (trained on Cheng 2020, other '
                           f'AOIs) applied to Y={Y} vs HS {y_ref}',
                n=int(ok.sum()),
                rho=spearmanr(d.pred[ok], d[f'hs_{y_ref}'][ok])[0]))
    d = frames[-1]
    for y_ref in (2019, 2021):
        ok = np.isfinite(d[f'hs_{y_ref}'])
        rows.append(dict(comparison=f'Cheng 2020 vs HS {y_ref} (reference '
                                    f'agreement)', n=int(ok.sum()),
                         rho=spearmanr(d.cheng_pct_dead[ok],
                                       d[f'hs_{y_ref}'][ok])[0]))
        rows.append(dict(comparison=f'Landsat model Y=2020 vs Cheng 2020 '
                                    f'(NEON, unseen AOI)', n=int(ok.sum()),
                         rho=spearmanr(d.pred[ok], d.cheng_pct_dead[ok])[0]))
    return pd.DataFrame(rows).drop_duplicates('comparison'), \
        pd.concat(frames, ignore_index=True)


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
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
def main(configfile, outputdir, composites, composite_suffix, labels,
         landcover, cheng_dir, hs_dir):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    comp = xr.open_dataset(composites / f'{AOI}{composite_suffix}').load()
    lab = xr.open_dataset(labels / f'{AOI}.nc').load()
    with rasterio.open(landcover / f'{AOI}_landcover.tif') as ds:
        forest = np.isin(ds.read(1), FOREST_CLASSES)
    ref, trees = load_hs(hs_dir, lab)
    for y in HS_YEARS:
        v = ref[y][forest & (trees >= MIN_TREES) & np.isfinite(ref[y])]
        print(f'HS {y}: n={v.size} mean dead fraction {v.mean():.3f}')

    res1, uni = part1(comp, lab, forest, ref, trees, config['resolution'])
    res1.to_csv(outputdir / 'neon_loyo.csv', index=False)
    uni.to_csv(outputdir / 'neon_univariate.csv', index=False)
    print(res1.round(3).to_string())
    print(uni.round(3).to_string())

    res2, cells = part2(config, composites, labels, landcover, cheng_dir,
                        ref, trees, composite_suffix)
    res2.to_csv(outputdir / 'transfer.csv', index=False)
    cells.to_csv(outputdir / 'transfer_cells.csv', index=False)
    print(res2.round(3).to_string())

    with PdfPages(outputdir / 'hls_vs_hs.pdf') as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, target in zip(axes, ['cum', 'new']):
            r = res1[res1.target == target]
            for s, g in r.groupby('scale_m'):
                ax.plot(g.year.astype(str), g.rho, 'o-', label=f'{s:.0f} m')
            ax.set_title(f'HS {"cumulative dead fraction" if target == "cum" else "new mortality since previous HS year"}\n'
                         'leave-one-year-out + spatial blocks')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('held-out year')
            ax.legend(fontsize=8)
        axes[0].set_ylabel('Spearman rho (pred vs HS)')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'neon_loyo.png', dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(TARGET_YEARS), figsize=(16, 4))
        for ax, Y in zip(axes, TARGET_YEARS):
            d = cells[(cells.Y == Y) & np.isfinite(cells[f'hs_{Y}'])]
            ax.hexbin(d[f'hs_{Y}'], d.pred, gridsize=40, bins='log',
                      mincnt=1)
            r = spearmanr(d[f'hs_{Y}'], d.pred)[0]
            ax.set_title(f'Y={Y}: rho={r:.2f}')
            ax.set_xlabel(f'HS {Y} dead fraction (100 m)')
        axes[0].set_ylabel('Landsat model trained on Cheng 2020\n'
                           '(other AOIs), applied at year Y')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'transfer.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
