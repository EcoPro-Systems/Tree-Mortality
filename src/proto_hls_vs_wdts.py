#!/usr/bin/env python
"""
Does AVIRIS-Classic imaging spectroscopy add skill over HLS/Landsat for tree
mortality? Uses the WDTS foliar-trait mosaics (ORNL DAAC 2403; fetched onto
the AOI grids by fetch_wdts_traits.py), which cover sierra_nf and
neon_soap_teak (Yosemite flight box) with one early-summer acquisition per
year 2013-2018.

Three feature sets are compared on identical samples and CV folds:

  hls    year-relative HLS L30 features (proto_hls_vs_hs.year_features:
         raw, base = Y - 2013, d1..d3, cmin/cmax), composites matched to the
         2020 NAIP date
  wdts   the same year-relative construction on the 14 WDTS trait means and
         the QC fractions qc_all (15 m subpixels passing all checks), qc_fc
         (green vegetation cover >= 0.5) and qc_shadow
  both   hls + wdts

Samples are NLCD forest, unburned (MTBS) 2012..Y and inside the WDTS flight
footprint in both Y and 2013. Trait means are NaN where WDTS masked every
15 m subpixel (much of it dead or sparse canopy); HGB handles the NaNs and
the qc fractions carry the masking itself as a feature.

Targets:
  A  ADS mortality polygons (label 1 vs surveyed no-damage 0), Y=2014-2018,
     both AOIs; ADS-covered fraction at 30/90/270 m; leave-one-year-out with
     spatial folds withheld, and leave-one-AOI-out; AUC (fraction >= 0.5)
     and Spearman.
  B  Hemming-Schroeder NEON cumulative dead-tree fraction, Y=2017, 2018
     (neon_soap_teak); leave-one-year-out with spatial folds; 30/90/270 m.
  C  Cheng et al. 2020 % dead canopy at 100 m cells, both AOIs; WDTS at 2018
     (its last year), HLS at 2020 and at 2018; 5-fold spatial block CV and
     leave-one-AOI-out (proto_hls_vs_cheng.evaluate).
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
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.backends.backend_pdf import PdfPages

from util import load_config
from proto_hls_vs_cheng import FOREST_CLASSES, evaluate
from proto_hls_vs_hs import (
    year_features, load_hs, block_mean, cells_100m, MIN_TREES,
)

AOIS = ['sierra_nf', 'neon_soap_teak']
WDTS_TRAITS = ['LMA', 'Nitrogen', 'Chlorophylls', 'Cellulose', 'Lignin',
               'Fiber', 'Sugar', 'Starch', 'NSC', 'Calcium', 'Potassium',
               'Phosphorus', 'Sulfur', 'Phenolics']
WDTS_QC = ['qc_all', 'qc_fc', 'qc_shadow']
WDTS_VARS = [f'{t}_mean' for t in WDTS_TRAITS] + WDTS_QC
BASE_YEAR = 2013
ADS_YEARS = [2014, 2015, 2016, 2017, 2018]
HS_YEARS = [2017, 2018]
CHENG_WDTS_YEAR = 2018
SCALES = [1, 3, 9]
BLOCK_M = 3000
NFOLD = 3
MAX_PX = 20000  # pixels sampled per AOI-year at 30 m
FSETS = ['hls', 'wdts', 'both']
UNI_FEATS = (['ndmi_base', 'nbr_base', 'rgi_base', 'ndvi_base', 'ndmi_d1']
             + [f'w_{v}_base' for v in WDTS_VARS] + ['w_qc_fc_d1'])


def load_wdts(path):
    """WDTS traits as float Dataset: trait means, qc fractions (0-1) and the
    flight footprint, NaN outside the footprint"""
    ds = xr.open_dataset(path).load()
    inside = ds.flight_id.values > 0
    out = {f'{t}_mean': ds[f'{t}_mean'] for t in WDTS_TRAITS}
    for q in WDTS_QC:
        out[q] = xr.where(ds.flight_id > 0, ds[q] / 100.0, np.nan).astype(
            np.float32)
    return xr.Dataset(out), dict(zip(ds.year.values.tolist(), inside))


def features(comp, w, year):
    f = year_features(comp, year)
    f.update({f'w_{k}': v for k, v in year_features(
        w, year, WDTS_VARS, BASE_YEAR).items()})
    return f


def fset_cols(cols):
    wd = [c for c in cols if c.startswith('w_')]
    hls = [c for c in cols if c not in wd and not c.startswith('_')]
    return {'hls': hls, 'wdts': wd, 'both': hls + wd}


def safe_auc(y, s):
    y = np.asarray(y) >= 0.5
    if y.all() or not y.any():
        return np.nan
    return roc_auc_score(y, s)


def blocks_frame(feats, target, valid, k, res, extra=None):
    """Block-mean features/target over valid pixels in k x k blocks with
    >= 70% valid; adds spatial block ids (BLOCK_M)"""
    tgt, n = block_mean(target, valid, k)
    keep = n >= 0.7 * k * k
    df = pd.DataFrame({f: block_mean(a, valid, k)[0][keep]
                       for f, a in feats.items()})
    df['_target'] = tgt[keep]
    for name, a in (extra or {}).items():
        df[name] = block_mean(a, valid, k)[0][keep]
    rr, cc = np.nonzero(keep)
    bs = max(1, int(BLOCK_M / (res * k)))
    df['_block'] = (rr // bs) * 10000 + (cc // bs)
    return df


def loyo(d, years, fsets, metric_rows, tag, scale_m):
    """Leave-one-year-out with spatial folds withheld; appends one row per
    feature set and held-out year. d has _target, _year, _fold columns"""
    preds = {}
    for fs, cols in fsets.items():
        pred = pd.Series(np.nan, index=d.index)
        for Y in years:
            for f in range(NFOLD):
                tr = d[(d._year != Y) & (d._fold != f)]
                te = d[(d._year == Y) & (d._fold == f)]
                if len(te) == 0:
                    continue
                m = HistGradientBoostingRegressor(max_iter=300,
                                                  learning_rate=0.05)
                m.fit(tr[cols], tr._target)
                pred[te.index] = m.predict(te[cols])
        preds[fs] = pred
        for Y in years + ['all']:
            te = d if Y == 'all' else d[d._year == Y]
            p = pred[te.index]
            metric_rows.append(dict(
                target=tag, cv='loyo', held_out=Y, scale_m=scale_m,
                features=fs, n=len(te),
                auc=safe_auc(te._target, p) if tag == 'ads' else np.nan,
                rho=spearmanr(te._target, p)[0],
                r2=r2_score(te._target, p)))
    return preds


def leave_aoi(d, fsets, metric_rows, tag, scale_m):
    for fs, cols in fsets.items():
        for aoi in sorted(d._aoi.unique()):
            tr, te = d[d._aoi != aoi], d[d._aoi == aoi]
            m = HistGradientBoostingRegressor(max_iter=300,
                                              learning_rate=0.05)
            m.fit(tr[cols], tr._target)
            p = m.predict(te[cols])
            metric_rows.append(dict(
                target=tag, cv='leave_aoi', held_out=aoi, scale_m=scale_m,
                features=fs, n=len(te),
                auc=safe_auc(te._target, p) if tag == 'ads' else np.nan,
                rho=spearmanr(te._target, p)[0],
                r2=r2_score(te._target, p)))


def univariate(d, tag, scale_m):
    rows = []
    for f in UNI_FEATS:
        if f not in d:
            continue
        for Y, g in list(d.groupby('_year')) + [('all', d)]:
            ok = np.isfinite(g[f]) & np.isfinite(g._target)
            if ok.sum() < 50:
                continue
            rows.append(dict(
                target=tag, scale_m=scale_m, feature=f, year=Y,
                n=int(ok.sum()), frac_finite=ok.mean(),
                rho=spearmanr(g[f][ok], g._target[ok])[0],
                auc=(safe_auc(g._target[ok], g[f][ok])
                     if tag == 'ads' else np.nan)))
    return rows


def part_ads(data, res, rng):
    """(A) ADS mortality-polygon fraction"""
    metrics, uni = [], []
    for k in SCALES:
        frames = []
        for name, (comp, lab, forest, w, inside) in data.items():
            burned_cum = np.cumsum(lab.burned.values, axis=0) > 0
            lab_years = lab.year.values.tolist()
            for Y in ADS_YEARS:
                label = lab.label.values[lab_years.index(Y)]
                valid = (forest & ~burned_cum[lab_years.index(Y)]
                         & inside[Y] & inside[BASE_YEAR]
                         & ((label == 0) | (label == 1)))
                if k == 1 and valid.sum() > MAX_PX:
                    keep = rng.choice(np.flatnonzero(valid), MAX_PX,
                                      replace=False)
                    valid = np.zeros_like(valid)
                    valid.flat[keep] = True
                df = blocks_frame(features(comp, w, Y),
                                  (label == 1).astype(np.float32), valid, k,
                                  res)
                df['_year'], df['_aoi'] = Y, name
                df['_block'] = name + '_' + df._block.astype(str)
                frames.append(df)
        d = pd.concat(frames, ignore_index=True)
        blocks = d._block.unique()
        fold = dict(zip(blocks, rng.integers(0, NFOLD, len(blocks))))
        d['_fold'] = d._block.map(fold)
        fsets = fset_cols(d.columns)
        loyo(d, ADS_YEARS, fsets, metrics, 'ads', res * k)
        leave_aoi(d, fsets, metrics, 'ads', res * k)
        uni += univariate(d, 'ads', res * k)
        print(f'ADS scale {res * k} m: n={len(d)}, positive fraction '
              f'{(d._target >= 0.5).mean():.3f}')
    return metrics, uni


def part_hs(data, hs_dir, res, rng):
    """(B) Hemming-Schroeder NEON cumulative dead fraction"""
    comp, lab, forest, w, inside = data['neon_soap_teak']
    ref, trees = load_hs(hs_dir, lab)
    burned_cum = np.cumsum(lab.burned.values, axis=0) > 0
    lab_years = lab.year.values.tolist()
    metrics, uni = [], []
    for k in SCALES:
        frames = []
        for Y in HS_YEARS:
            valid = (forest & (trees >= MIN_TREES) & np.isfinite(ref[Y])
                     & ~burned_cum[lab_years.index(Y)]
                     & inside[Y] & inside[BASE_YEAR])
            df = blocks_frame(features(comp, w, Y), ref[Y], valid, k, res)
            df['_year'], df['_aoi'] = Y, 'neon_soap_teak'
            frames.append(df)
        d = pd.concat(frames, ignore_index=True)
        blocks = d._block.unique()
        fold = dict(zip(blocks, rng.integers(0, NFOLD, len(blocks))))
        d['_fold'] = d._block.map(fold)
        loyo(d, HS_YEARS, fset_cols(d.columns), metrics, 'hs', res * k)
        uni += univariate(d, 'hs', res * k)
        print(f'HS scale {res * k} m: n={len(d)}')
    return metrics, uni


def part_cheng(data, cheng_dir):
    """(C) Cheng 2020 % dead canopy at 100 m"""
    frames = []
    for name, (comp, lab, forest, w, inside) in data.items():
        extra = {f'w_{k}': v for k, v in year_features(
            w, CHENG_WDTS_YEAR, WDTS_VARS, BASE_YEAR).items()}
        extra.update({f'h18_{k}': v for k, v in
                      year_features(comp, CHENG_WDTS_YEAR).items()})
        d = cells_100m(name, comp, lab, forest, cheng_dir, 2020, extra,
                       inside[CHENG_WDTS_YEAR] & inside[BASE_YEAR])
        d['block'] = name + '_' + d.block.astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    wd = [c for c in d.columns if c.startswith('w_')]
    h18 = [c for c in d.columns if c.startswith('h18_')]
    h20 = [c for c in d.columns if c not in wd + h18 +
           ['cheng_pct_dead', 'aoi', 'block']]
    fsets = {'hls2020': h20, 'hls2018': h18, 'wdts2018': wd,
             'hls2018+wdts2018': h18 + wd, 'hls2020+wdts2018': h20 + wd}
    metrics = []
    for fs, cols in fsets.items():
        rows, _ = evaluate(d, cols, 'cheng_pct_dead')
        metrics += [dict(r, target='cheng', features=fs) for r in rows]
        print(f'Cheng {fs:17s} block-CV R2 {rows[0]["r2"]:.3f} '
              f'rho {rows[0]["spearman"]:.3f}')
    uni = []
    for f in UNI_FEATS:
        for col in ([f] if f.startswith('w_') else [f, f'h18_{f}']):
            if col not in d:
                continue
            ok = np.isfinite(d[col])
            uni.append(dict(target='cheng', scale_m=100, feature=col,
                            year=2020, n=int(ok.sum()),
                            frac_finite=ok.mean(),
                            rho=spearmanr(d[col][ok],
                                          d.cheng_pct_dead[ok])[0]))
    return metrics, uni, d


def make_plots(res, uni, data, outputdir):
    with PdfPages(outputdir / 'hls_vs_wdts.pdf') as pdf:
        # 1. Skill by feature set, per target and scale
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        colors = dict(hls='tab:blue', wdts='tab:orange', both='tab:green')
        for ax, (tag, metric) in zip(axes[:2], [('ads', 'auc'),
                                                ('hs', 'rho')]):
            r = res[(res.target == tag) & (res.cv == 'loyo')
                    & (res.held_out == 'all')]
            for i, fs in enumerate(FSETS):
                g = r[r.features == fs].sort_values('scale_m')
                ax.bar(np.arange(len(g)) + (i - 1) * 0.27, g[metric], 0.27,
                       color=colors[fs], label=fs)
            ax.set_xticks(range(len(g)), [f'{s:.0f} m' for s in g.scale_m])
            ax.set_ylabel(metric)
            ax.set_ylim(0.5 if metric == 'auc' else 0, 1)
            ax.set_title({'ads': 'ADS polygon fraction, 2014-2018\n'
                                 'leave-one-year-out + spatial folds',
                          'hs': 'NEON HS dead fraction, 2017/2018\n'
                                'leave-one-year-out + spatial folds'}[tag])
            ax.legend(fontsize=8)
        r = res[(res.target == 'cheng') & (res.cv == 'block5')]
        axes[2].barh(r.features, r.r2, color='0.5')
        for y, (v, rho) in enumerate(zip(r.r2, r.spearman)):
            axes[2].text(v, y, f' R2 {v:.2f}, rho {rho:.2f}', va='center',
                         fontsize=8)
        axes[2].set_xlim(0, max(0.8, r.r2.max() + 0.3))
        axes[2].set_title('Cheng 2020 % dead canopy (100 m)\n'
                          '5-fold spatial block CV')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'skill.png', dpi=150)
        plt.close(fig)

        # 2. Univariate: base-change of each feature vs each target
        u = uni[(uni.year.astype(str) == 'all') | (uni.target == 'cheng')]
        u = u[u.scale_m.isin([90, 100])]
        piv = u.pivot_table(index='feature', columns='target', values='rho')
        piv = piv.reindex([f for f in UNI_FEATS if f in piv.index])
        fig, ax = plt.subplots(figsize=(6, 9))
        im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6,
                       aspect='auto')
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f'{piv.values[i, j]:.2f}', ha='center',
                        va='center', fontsize=7)
        fig.colorbar(im, label='Spearman rho')
        ax.set_title('Single-feature Spearman (90 m blocks; Cheng 100 m)')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'univariate.png', dpi=150)
        plt.close(fig)

        # 3. Maps 2013 -> 2016 for sierra_nf
        comp, lab, forest, w, inside = data['sierra_nf']
        ext = [comp.x.min().item() - 15, comp.x.max().item() + 15,
               comp.y.min().item() - 15, comp.y.max().item() + 15]
        panels = [
            ('HLS NDMI 2016 - 2013', comp.ndmi, 0.2, 'RdBu'),
            ('WDTS qc_fc 2016 - 2013', w.qc_fc, 0.8, 'RdBu'),
            ('WDTS chlorophyll 2016 - 2013', w.Chlorophylls_mean, 20,
             'RdBu'),
            ('WDTS LMA 2016 - 2013', w.LMA_mean, 80, 'RdBu_r'),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 12))
        pos = (lab.label.sel(year=2016) == 1).values.astype(float)
        for ax, (title, v, lim, cmap) in zip(axes.ravel(), panels):
            dd = (v.sel(year=2016) - v.sel(year=2013)).values
            dd = np.where(forest, dd, np.nan)
            im = ax.imshow(dd, extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
            ax.contour(lab.x, lab.y, pos, levels=[0.5], colors='k',
                       linewidths=0.4)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle('sierra_nf, NLCD forest; black: ADS 2016 mortality '
                     'polygons')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'maps_sierra_nf_2016.png', dpi=110)
        plt.close(fig)


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
@click.option('--wdts-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/wdts')
@click.option('--labels', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_labels')
@click.option('--landcover', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/landcover')
@click.option('--cheng-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/cheng2024/statewide_2020')
@click.option('--hs-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/hemming_schroeder2023/data/deliverables/raster'))
def main(configfile, outputdir, composites, composite_suffix, wdts_dir,
         labels, landcover, cheng_dir, hs_dir):

    config = load_config(configfile)
    os.makedirs(outputdir, exist_ok=True)
    res = config['resolution']
    rng = np.random.default_rng(0)

    data = {}
    for name in AOIS:
        comp = xr.open_dataset(composites / f'{name}{composite_suffix}').load()
        lab = xr.open_dataset(labels / f'{name}.nc').load()
        with rasterio.open(landcover / f'{name}_landcover.tif') as ds:
            forest = np.isin(ds.read(1), FOREST_CLASSES)
        w, inside = load_wdts(wdts_dir / f'{name}_traits.nc')
        assert np.allclose(w.x, lab.x) and np.allclose(w.y, lab.y)
        data[name] = (comp, lab, forest, w, inside)
        cov = {y: round(float(inside[y][forest].mean()), 2) for y in inside}
        print(f'[{name}] WDTS footprint fraction of forest by year: {cov}')

    res_a, uni_a = part_ads(data, res, rng)
    res_b, uni_b = part_hs(data, hs_dir, res, rng)
    res_c, uni_c, cells = part_cheng(data, cheng_dir)

    res_ab = pd.DataFrame(res_a + res_b)
    res_ab.to_csv(outputdir / 'ads_hs_cv.csv', index=False)
    res_c = pd.DataFrame(res_c)
    res_c.to_csv(outputdir / 'cheng_cv.csv', index=False)
    cells.to_csv(outputdir / 'cheng_cells.csv', index=False)
    uni = pd.DataFrame(uni_a + uni_b + uni_c)
    uni.to_csv(outputdir / 'univariate.csv', index=False)

    summ = res_ab[res_ab.held_out.astype(str).isin(['all'])
                  | (res_ab.cv == 'leave_aoi')]
    print(summ.pivot_table(index=['target', 'cv', 'held_out', 'scale_m'],
                           columns='features',
                           values=['auc', 'rho']).round(3).to_string())
    print(res_c.round(3).to_string())
    make_plots(pd.concat([res_ab, res_c], ignore_index=True), uni, data,
               outputdir)


if __name__ == '__main__':
    main()
