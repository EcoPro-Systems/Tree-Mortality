#!/usr/bin/env python
"""
Compare HLS (Landsat-only L30, or L30+S30) composites with the Cheng et al.
(2024) 2020 statewide dead-tree maps derived from 0.6 m NAIP, and test how
well HLS features predict them.

HLS composites come from hls_naip_composites.py (matched to each pixel's 2020
NAIP acquisition date). 30 m pixels are aggregated onto Cheng's native 100 m
grid (EPSG:5072) by pixel center, so the target is never resampled. Pixels
are restricted to NLCD 2013 forest not burned (MTBS) in 2012-2019; 100 m
cells need >= MIN_VALID of their 30 m pixels valid.

Targets: % dead canopy area per ha, dead-tree density (trees/ha, bias
corrected), and red-stage ratio.

Feature sets (all using only years <= 2020):
  l30_2020   L30 bands/indices in 2020 only
  l30_hist   l30_2020 + change vs 2013 for each year 2014-2020
  hls_hist   as l30_hist, from L30+S30 composites
  ads        ADS mortality-polygon coverage fraction per year 2012-2019
  l30_ads    l30_hist + ads

Models are HistGradientBoostingRegressor, evaluated with 5-fold spatial
block CV (3 km blocks, all AOIs pooled) and leave-one-AOI-out. random_state
is fixed because early stopping uses a random validation split when
n > 10,000.

Outputs (outputdir): cells.csv (features and targets per 100 m cell),
cv_predictions.csv (aoi, row, col and the block-CV out-of-fold prediction
pred_<target>_<feature set> for each cell, in cells.csv row order),
univariate_spearman.csv, model_cv.csv, hls_vs_cheng.pdf.
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
from pyproj import Transformer
from scipy.stats import spearmanr
from rasterio.windows import Window
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.backends.backend_pdf import PdfPages

from util import load_config
from hls_annual_composites import ROLES, INDICES

CHENG = {
    'pct_dead': 'NAIP_2020_CA_owatershed_v8_area_100m_proj_max_clip_mask_percentperha.tif',
    'density': 'NAIP_2020_CA_owatershed_v8_density_100m_proj_clip_mask_biascorrected.tif',
    'red_ratio': 'red_stage_ratio_bydensity_buffer1p_lt1_nobuffer_mask_nocountnodata.tif',
}
FOREST_CLASSES = [41, 42, 43]
BASE_YEAR, REF_YEAR = 2013, 2020
MIN_VALID = 0.7
BLOCK_M = 3000
VARIABLES = ROLES + INDICES


def cheng_window(path, bounds_5072):
    """Read the Cheng raster window covering bounds; return array (NaN
    nodata), window transform"""
    with rasterio.open(path) as ds:
        win = rasterio.windows.from_bounds(*bounds_5072, ds.transform)
        c0, r0 = int(np.floor(win.col_off)) - 1, int(np.floor(win.row_off)) - 1
        win = Window(c0, r0, int(np.ceil(win.width)) + 3,
                     int(np.ceil(win.height)) + 3)
        a = ds.read(1, window=win, boundless=True, fill_value=np.nan)
        a = a.astype(np.float32)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            a[a == ds.nodata] = np.nan
        a[a < -1000] = np.nan
        return a, ds.window_transform(win)


def cell_index(xs, ys, crs, transform_5072, shape_5072):
    """Cheng cell index (flattened) for each AOI pixel center"""
    X, Y = np.meshgrid(xs, ys)
    tr = Transformer.from_crs(crs, 'EPSG:5072', always_xy=True)
    ex, ny = tr.transform(X.ravel(), Y.ravel())
    inv = ~transform_5072
    col, row = inv * (np.asarray(ex), np.asarray(ny))
    col, row = np.floor(col).astype(int), np.floor(row).astype(int)
    ok = (row >= 0) & (row < shape_5072[0]) & (col >= 0) & (col < shape_5072[1])
    idx = np.where(ok, row * shape_5072[1] + col, -1)
    return idx.reshape(X.shape)


def aggregate(values, valid, idx, ncell):
    """Mean of values over valid pixels in each cell"""
    m = valid & np.isfinite(values) & (idx >= 0)
    s = np.bincount(idx[m], weights=values[m], minlength=ncell)
    n = np.bincount(idx[m], minlength=ncell)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / n


def build_aoi(name, comps, lab, landcover, cheng_dir):
    crs = lab.attrs['crs']
    xs, ys = lab.x.values, lab.y.values
    tr = Transformer.from_crs(crs, 'EPSG:5072', always_xy=True)
    bx, by = tr.transform([xs.min(), xs.max(), xs.min(), xs.max()],
                          [ys.min(), ys.min(), ys.max(), ys.max()])
    bounds = (min(bx), min(by), max(bx), max(by))

    targets, t5072 = {}, None
    for key, fname in CHENG.items():
        a, t = cheng_window(cheng_dir / fname, bounds)
        if t5072 is None:
            t5072, shp = t, a.shape
        # All Cheng 100 m layers share one grid (checked: same origin)
        assert np.allclose(list(t)[:6], list(t5072)[:6], atol=1e-3)
        targets[key] = a.ravel()
    ncell = shp[0] * shp[1]
    idx = cell_index(xs, ys, crs, t5072, shp)

    with rasterio.open(landcover) as ds:
        forest = np.isin(ds.read(1), FOREST_CLASSES)
    years = lab.year.values
    burned = lab.burned.sel(year=years[(years >= 2012) & (years < REF_YEAR)])
    valid = forest & ~burned.any('year').values

    total = np.bincount(idx[idx >= 0], minlength=ncell)
    nvalid = np.bincount(idx[valid & (idx >= 0)], minlength=ncell)
    keep = (total >= 6) & (nvalid >= MIN_VALID * total)

    feats = {}
    for tag, comp in comps.items():
        for v in VARIABLES:
            ref = comp[v].sel(year=REF_YEAR).values
            if tag == 'l30':
                feats[f'{tag}_{v}_{REF_YEAR}'] = aggregate(ref, valid, idx,
                                                           ncell)
            base = comp[v].sel(year=BASE_YEAR).values
            for y in range(BASE_YEAR + 1, REF_YEAR + 1):
                d = comp[v].sel(year=y).values - base
                feats[f'{tag}_{v}_d{y}'] = aggregate(d, valid, idx, ncell)
            if tag == 'hls':
                feats[f'{tag}_{v}_{REF_YEAR}'] = aggregate(ref, valid, idx,
                                                           ncell)
    for y in range(2012, REF_YEAR):
        if y in years:
            pos = (lab.label.sel(year=y).values == 1).astype(float)
            feats[f'ads_cov_{y}'] = aggregate(pos, valid, idx, ncell)

    df = pd.DataFrame(feats)
    for k, v in targets.items():
        df[k] = v
    rows, cols = np.divmod(np.arange(ncell), shp[1])
    df['x5072'] = t5072.c + (cols + 0.5) * t5072.a
    df['y5072'] = t5072.f + (rows + 0.5) * t5072.e
    df['row'], df['col'] = rows, cols
    df['aoi'] = name
    df = df[keep & np.isfinite(df.pct_dead.values)].copy()
    df['block'] = (name + '_' + (df.x5072 // BLOCK_M).astype(int).astype(str)
                   + '_' + (df.y5072 // BLOCK_M).astype(int).astype(str))
    return df, (t5072, shp)


def feature_sets(cols):
    l30_2020 = [c for c in cols if c.startswith('l30_') and
                c.endswith(f'_{REF_YEAR}')]
    l30_hist = [c for c in cols if c.startswith('l30_')]
    hls_hist = [c for c in cols if c.startswith('hls_')]
    ads = [c for c in cols if c.startswith('ads_')]
    return {'l30_2020': l30_2020, 'l30_hist': l30_hist,
            'hls_hist': hls_hist, 'ads': ads, 'l30_ads': l30_hist + ads}


def evaluate(df, feats, target):
    d = df[np.isfinite(df[target])]
    out, preds = [], pd.Series(np.nan, index=d.index)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(d, groups=d.block)):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          random_state=0)
        m.fit(d.iloc[tr][feats], d.iloc[tr][target])
        preds.iloc[te] = m.predict(d.iloc[te][feats])
    out.append(dict(cv='block5', held_out='all',
                    r2=r2_score(d[target], preds),
                    spearman=spearmanr(d[target], preds)[0],
                    mae=mean_absolute_error(d[target], preds), n=len(d)))
    for aoi in sorted(d.aoi.unique()):
        tr, te = d[d.aoi != aoi], d[d.aoi == aoi]
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          random_state=0)
        m.fit(tr[feats], tr[target])
        p = m.predict(te[feats])
        out.append(dict(cv='leave_aoi', held_out=aoi,
                        r2=r2_score(te[target], p),
                        spearman=spearmanr(te[target], p)[0],
                        mae=mean_absolute_error(te[target], p), n=len(te)))
    return out, preds


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('--composites', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_composites')
@click.option('--labels', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_labels')
@click.option('--landcover', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/landcover')
@click.option('--cheng-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/cheng2024/statewide_2020')
@click.option('--window', default=25, show_default=True)
def main(configfile, outputdir, composites, labels, landcover, cheng_dir,
         window):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    frames, grids = [], {}
    for name in config['aois']:
        comps = {
            'l30': xr.open_dataset(
                composites / f'{name}_naip{REF_YEAR}_L30_w{window}.nc').load(),
            'hls': xr.open_dataset(
                composites / f'{name}_naip{REF_YEAR}_L30S30_w{window}.nc').load(),
        }
        lab = xr.open_dataset(labels / f'{name}.nc').load()
        df, grids[name] = build_aoi(name, comps, lab,
                                    landcover / f'{name}_landcover.tif',
                                    cheng_dir)
        frames.append(df)
        print(f'[{name}] {len(df)} cells; pct_dead median '
              f'{df.pct_dead.median():.2f}, p90 {df.pct_dead.quantile(.9):.2f}; '
              f'density median {df.density.median():.1f}')
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outputdir / 'cells.csv', index=False)
    fsets = feature_sets(df.columns)

    # Univariate rank correlations
    uni = []
    for target in CHENG:
        for f in fsets['l30_hist'] + fsets['ads']:
            ok = np.isfinite(df[f]) & np.isfinite(df[target])
            for aoi, g in df[ok].groupby('aoi'):
                uni.append(dict(target=target, feature=f, aoi=aoi,
                                rho=spearmanr(g[f], g[target])[0]))
    uni = pd.DataFrame(uni)
    uni.to_csv(outputdir / 'univariate_spearman.csv', index=False)

    res, preds = [], {}
    for target in CHENG:
        for fs, feats in fsets.items():
            rows, p = evaluate(df, feats, target)
            res += [dict(r, target=target, features=fs) for r in rows]
            preds[(target, fs)] = p
            print(f'{target:9s} {fs:9s} block-CV R2 {rows[0]["r2"]:.3f} '
                  f'rho {rows[0]["spearman"]:.3f}')
    res = pd.DataFrame(res)
    res.to_csv(outputdir / 'model_cv.csv', index=False)
    # Out-of-fold predictions, so maps can be redrawn without re-fitting
    oof = df[['aoi', 'row', 'col']].copy()
    for (target, fs), p in preds.items():
        oof[f'pred_{target}_{fs}'] = p.reindex(df.index)
    oof.to_csv(outputdir / 'cv_predictions.csv', index=False)
    make_plots(df, uni, res, preds, grids, outputdir)


def make_plots(df, uni, res, preds, grids, outputdir):
    with PdfPages(outputdir / 'hls_vs_cheng.pdf') as pdf:
        # 1. Model skill by feature set
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, target in zip(axes, CHENG):
            r = res[(res.target == target)]
            piv = r.pivot_table(index='features', columns='held_out',
                                values='r2')
            piv = piv.loc[['ads', 'l30_2020', 'l30_hist', 'hls_hist',
                           'l30_ads']]
            piv.plot.bar(ax=ax, rot=30)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f'{target}: R$^2$ (all = 5-fold 3 km block CV;\n'
                         'AOI = trained on the other two)')
            ax.legend(fontsize=7)
        axes[0].set_ylabel('R$^2$')
        axes[0].set_ylim(-0.5, 1)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'model_r2.png', dpi=150)
        plt.close(fig)

        # 2. Observed vs block-CV predicted
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, target in zip(axes, CHENG):
            p = preds[(target, 'l30_hist')]
            ok = p.notna()
            ax.hexbin(df.loc[ok.index[ok], target], p[ok], gridsize=60,
                      bins='log', mincnt=1)
            lim = np.nanpercentile(df[target], 99.5)
            ax.plot([0, lim], [0, lim], 'r-', lw=0.8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel(f'Cheng 2020 {target}')
            ax.set_ylabel('predicted (Landsat L30, block CV)')
            r = res[(res.target == target) & (res.features == 'l30_hist')
                    & (res.cv == 'block5')].iloc[0]
            ax.set_title(f'R$^2$={r.r2:.2f}, rho={r.spearman:.2f}')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'obs_vs_pred.png', dpi=150)
        plt.close(fig)

        # 3. Maps: Cheng vs predicted vs single L30 feature, per AOI
        p = preds[('pct_dead', 'l30_hist')]
        for aoi, (t, shp) in grids.items():
            d = df[df.aoi == aoi]
            obs = np.full(shp, np.nan)
            pred = np.full(shp, np.nan)
            ndmi = np.full(shp, np.nan)
            obs[d.row, d.col] = d.pct_dead
            pred[d.row, d.col] = p.loc[d.index]
            ndmi[d.row, d.col] = d[f'l30_ndmi_d{REF_YEAR}']
            vmax = np.nanpercentile(obs, 98)
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
            for ax, arr, title, cmap, vr in [
                (axes[0], obs, 'Cheng 2020 % dead canopy', 'magma_r',
                 (0, vmax)),
                (axes[1], pred, 'Predicted from Landsat (block CV)',
                 'magma_r', (0, vmax)),
                (axes[2], ndmi, f'L30 NDMI {REF_YEAR} - {BASE_YEAR}',
                 'RdBu', (-0.2, 0.2)),
            ]:
                im = ax.imshow(arr, cmap=cmap, vmin=vr[0], vmax=vr[1])
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, shrink=0.7)
            fig.suptitle(f'{aoi} (100 m cells, forest, unburned 2012-2019)')
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outputdir / f'map_{aoi}.png', dpi=110)
            plt.close(fig)

        # 4. Top univariate correlations with % dead canopy
        u = uni[uni.target == 'pct_dead'].groupby('feature').rho.mean()
        top = u.abs().sort_values(ascending=False).index[:20]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(range(len(top)), u[top].values)
        ax.set_yticks(range(len(top)), top, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Spearman rho with Cheng % dead canopy (mean over AOIs)')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'univariate_top.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
