#!/usr/bin/env python
"""
Prototype predictive test: how well can HLS features predict ADS mortality
labels in held-out years and held-out AOIs?

Features for survey year Y are the Jul-Sep composite of each band/index at
Y-2, Y-1, Y and Y+1, each relative to the 2013 baseline, plus the raw value
at Y. Using the trajectory (rather than a single year-over-year change)
lets the model handle trees at different stages (green -> red -> gray).

Two targets, both on NLCD forest that is surveyed and not burned (MTBS
2012..Y+1):
  pixel   mortality polygon (1) vs surveyed with no damage feature (0);
          HistGradientBoostingClassifier, AUC
  block   ~1 km blocks (33x33 px): ADS-covered fraction of the block;
          HistGradientBoostingRegressor on block-mean features, R^2 and
          Spearman

Cross-validation: leave-one-year-out (all AOIs) and leave-one-AOI-out.
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
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)

from util import load_config
from proto_hls_mortality_signal import block_sum, FOREST_CLASSES, BASE_YEAR

VARIABLES = ['ndvi', 'ndmi', 'nbr', 'rgi', 'evi', 'red', 'green', 'nir',
             'swir1', 'swir2']
OFFSETS = [-2, -1, 0, 1]
PIXELS_PER_CLASS = 20000
BLOCK = 33


def features(comp, year):
    """Dict of feature name -> 2D array for survey year `year`"""
    yrs = comp.year.values.tolist()
    out = {}
    if 'lag' in comp.dims:
        # Flight-matched composites: prev/at/next relative to the 2013
        # baseline at the same time of year
        c = comp.sel(year=year)
        for var in VARIABLES:
            base = c[var].sel(lag='base').values
            for lag, off in [('prev', -1), ('at', 0), ('next', 1)]:
                out[f'{var}_{off:+d}'] = c[var].sel(lag=lag).values - base
            out[f'{var}_raw'] = c[var].sel(lag='at').values
        return out
    for var in VARIABLES:
        base = comp[var].sel(year=BASE_YEAR).values
        for off in OFFSETS:
            y = year + off
            name = f'{var}_{off:+d}'
            out[name] = (comp[var].sel(year=y).values - base
                         if y in yrs and y != BASE_YEAR
                         else np.full(base.shape, np.nan, np.float32))
        out[f'{var}_raw'] = comp[var].sel(year=year).values
    return out


def block_nanmean(f, mask, k):
    ok = mask & np.isfinite(f)
    with np.errstate(invalid='ignore', divide='ignore'):
        return block_sum(np.where(ok, f, 0), k) / block_sum(ok, k)


def build(config, composites, suffix, labels, landcover, rng):
    pix, blk = [], []
    for name in config['aois']:
        comp = xr.open_dataset(composites / f'{name}{suffix}').load()
        lab = xr.open_dataset(labels / f'{name}.nc').load()
        with rasterio.open(landcover / f'{name}_landcover.tif') as ds:
            forest = np.isin(ds.read(1), FOREST_CLASSES)
        burned_cum = np.cumsum(lab.burned.values, axis=0) > 0
        lab_years = lab.year.values.tolist()

        for year in comp.year.values.tolist():
            if year <= BASE_YEAR or year not in lab_years:
                continue
            yi = lab_years.index(year)
            yn = min(yi + 1, len(lab_years) - 1)
            label = lab.label.values[yi]
            valid = forest & ~burned_cum[yn]
            pos, neg = valid & (label == 1), valid & (label == 0)
            if pos.sum() < 100 or neg.sum() < 100:
                continue
            feats = features(comp, year)
            names = list(feats)
            meta = dict(aoi=name, year=year)

            for cls, mask in [(1, pos), (0, neg)]:
                idx = np.flatnonzero(mask)
                idx = rng.choice(idx, min(len(idx), PIXELS_PER_CLASS),
                                 replace=False)
                df = pd.DataFrame({n: feats[n].ravel()[idx] for n in names})
                df['y'] = cls
                pix.append(df.assign(**meta))

            surveyed = (pos | neg).astype(float)
            n = block_sum(surveyed, BLOCK)
            keep = n >= 0.5 * BLOCK * BLOCK
            if keep.sum() < 10:
                continue
            df = pd.DataFrame({
                nm: block_nanmean(f, surveyed > 0, BLOCK)[keep]
                for nm, f in feats.items()
            })
            df['y'] = block_sum(pos, BLOCK)[keep] / n[keep]
            blk.append(df.assign(**meta))
        print(f'[{name}] features built')
    return pd.concat(pix, ignore_index=True), pd.concat(blk, ignore_index=True)


def cross_validate(df, group, kind, feat_cols):
    rows = []
    for g in sorted(df[group].unique()):
        train, test = df[df[group] != g], df[df[group] == g]
        if len(train) == 0 or (kind == 'pixel' and test.y.nunique() < 2):
            continue
        if kind == 'pixel':
            model = HistGradientBoostingClassifier(max_iter=300,
                                                   learning_rate=0.05)
            model.fit(train[feat_cols], train.y)
            p = model.predict_proba(test[feat_cols])[:, 1]
            rows.append(dict(cv=group, held_out=g, n=len(test),
                             auc=roc_auc_score(test.y, p)))
        else:
            model = HistGradientBoostingRegressor(max_iter=300,
                                                  learning_rate=0.05)
            model.fit(train[feat_cols], train.y)
            p = model.predict(test[feat_cols])
            rows.append(dict(cv=group, held_out=g, n=len(test),
                             r2=r2_score(test.y, p),
                             spearman=spearmanr(test.y, p)[0]))
    return rows


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('--composites', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_composites')
@click.option('--composite-suffix', default='_doy182-273.nc')
@click.option('--labels', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_labels')
@click.option('--landcover', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/landcover')
def main(configfile, outputdir, composites, composite_suffix, labels,
         landcover):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    rng = np.random.default_rng(0)
    pix, blk = build(config, composites, composite_suffix, labels,
                     landcover, rng)
    feat_cols = [c for c in pix.columns if c not in ('y', 'aoi', 'year')]

    results = []
    for kind, df in [('pixel', pix), ('block', blk)]:
        for group in ['year', 'aoi']:
            for r in cross_validate(df, group, kind, feat_cols):
                results.append(dict(r, target=kind))
    res = pd.DataFrame(results)
    res.to_csv(outputdir / 'model_cv.csv', index=False)
    print(res.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    r = res[(res.target == 'pixel') & (res.cv == 'year')]
    axes[0].bar(r.held_out.astype(str), r.auc)
    axes[0].axhline(0.5, color='k', lw=0.5)
    axes[0].set_ylim(0.3, 1)
    axes[0].set_title('Pixel AUC, leave-one-year-out\n'
                      '(ADS polygon vs surveyed no-damage)')
    r = res[(res.target == 'block') & (res.cv == 'year')]
    axes[1].bar(r.held_out.astype(str), r.r2, label='R$^2$')
    axes[1].plot(r.held_out.astype(str), r.spearman, 'ko', label='Spearman')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_title('~1 km block ADS coverage, leave-one-year-out')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(outputdir / 'model_cv.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
