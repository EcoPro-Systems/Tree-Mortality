#!/usr/bin/env python
"""
Predisposition test at NEON SOAP/TEAK: do remote-sensing indicators measured
BEFORE trees die predict which trees die, beyond tree size, stand structure,
topography and climate?

Trees and fates come from Hemming-Schroeder et al. (2023): ~1M lidar-
segmented crowns with a live/dead status per year (relative greenness within
the crown from NEON imagery; trees_<year>_rgreen.shp) and per-tree site
covariates (feature_vars_labels.csv). Two cohorts:

  A  2015-17 die-off: live in 2013; died = dead in both 2017 and 2018,
     survived = live in both (trees with inconsistent labels dropped)
  B  2020-22 drought wave: live in 2018 and 2019; died = dead in 2021

Feature groups (sampled at the 30 m pixel of the tree top unless noted):

  S   tree/site: 2013 height and crown area, local density (tpa, meank,
      cover), elevation, slope, aspect, climate normals (ppt, tmax, vpdmax),
      granite, distance to rivers, site
  H   HLS L30 (NAIP-date composites): bands/indices in the pre-mortality
      years and their change
  W   WDTS AVIRIS-C traits (14 trait means + QC fractions): same years
  M   MASTER 2020-10-15 (cohort B only): TOA indices and 11.3 um BT at the
      nearest near-nadir MASTER pixel (~50 m)

Cohort A uses 2013 and 2014 (and their change) as the pre-mortality years;
2015 (June, just before the die-off peaked) is reported separately as a
lead-time sensitivity. Cohort B uses HLS 2019 and 2020, WDTS 2018 (three
years' lead) and MASTER Fall 2020.

Evaluation: HistGradientBoostingClassifier, 5-fold GroupKFold on 1 km
blocks (and leave-one-site-out); AUC and average precision. A "30 m oracle"
scores each tree by the observed death rate of the *other* cohort trees in
its pixel: the ceiling for any predictor that is constant within a 30 m
pixel. Univariate AUCs are reported pooled and averaged within site x 200 m
elevation strata (to separate indicators from the elevation gradient).
"""
import os
import click
import numpy as np
import pandas as pd
import xarray as xr
import pyogrio
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from matplotlib.backends.backend_pdf import PdfPages
from pyproj import Transformer

from hls_annual_composites import ROLES, INDICES
from proto_hls_vs_wdts import WDTS_TRAITS, load_wdts

AOI = 'neon_soap_teak'
S_VARS = ['zmax2013', 'ca2013', 'tpa', 'meank', 'cover', 'elev', 'slope',
          'northness', 'eastness', 'ppt', 'tmax', 'vpdmax', 'granite',
          'd_rivers', 'is_soap']
H_VARS = ROLES + INDICES
W_VARS = [f'{t}_mean' for t in WDTS_TRAITS] + ['qc_all', 'qc_fc']
MASTER_LINES = [1, 2, 3, 4]
MAX_TREES = 200000
BLOCK_M = 1000
ELEV_BAND = 200


def load_trees(hs_dir):
    vec = hs_dir / 'data/deliverables/vector'
    pts = pyogrio.read_dataframe(vec / 'tree_locations_las_intersection.shp',
                                 columns=['treeID', 'ca2013', 'zmax2013'])
    df = pd.DataFrame({'treeID': pts.treeID, 'x': pts.geometry.x,
                       'y': pts.geometry.y, 'ca2013': pts.ca2013,
                       'zmax2013': pts.zmax2013}).set_index('treeID')
    for y in (2013, 2017, 2018, 2019, 2021):
        d = pyogrio.read_dataframe(vec / f'trees_{y}_rgreen.shp',
                                   read_geometry=False,
                                   columns=['treeID', 'live', 'sites'])
        d = d.set_index('treeID')
        df[f'live{y}'] = d.live
        if y == 2013:
            df['site'] = d.sites
    cov = pd.read_csv(vec / 'feature_vars_labels.csv',
                      usecols=['treeID', 'tpa', 'meank', 'cover', 'elev',
                               'slope', 'aspect', 'ppt', 'tmax', 'vpdmax',
                               'granite', 'd_rivers']).set_index('treeID')
    df = df.join(cov)
    df['northness'] = np.cos(df.aspect)
    df['eastness'] = np.sin(df.aspect)
    df['is_soap'] = (df.site == 'SOAP').astype(int)
    return df


def pixel_index(df, transform, shape):
    col, row = ~transform * (df.x.values, df.y.values)
    row, col = np.floor(row).astype(int), np.floor(col).astype(int)
    ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return np.where(ok, row, -1), np.where(ok, col, -1)


def sample(ds, variables, years, row, col, prefix):
    out = {}
    ok = row >= 0
    for v in variables:
        for y in years:
            a = np.full(len(row), np.nan, np.float32)
            a[ok] = ds[v].sel(year=y).values[row[ok], col[ok]]
            out[f'{prefix}{v}_{y}'] = a
        if len(years) == 2:
            out[f'{prefix}{v}_d{years[1] % 100:02d}{years[0] % 100:02d}'] = \
                out[f'{prefix}{v}_{years[1]}'] - out[f'{prefix}{v}_{years[0]}']
    return out


def master_features(df, master_dir, crs):
    """Nearest MASTER pixel (within 40 m) from the most nadir line"""
    from proto_master_quicklook import read_line
    to_utm = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    best = None
    for n in MASTER_LINES:
        path = sorted(master_dir.glob(f'MASTERL1B_2190600_{n:02d}_*.hdf'))[0]
        lon, lat, feats, _ = read_line(path)
        x, y = to_utm.transform(lon, lat)
        dist, i = cKDTree(np.c_[x, y]).query(np.c_[df.x, df.y],
                                             distance_upper_bound=40)
        ok = np.isfinite(dist)
        cur = pd.DataFrame(np.nan, index=df.index,
                           columns=[f'M_{k}' for k in feats])
        for k, v in feats.items():
            cur.loc[ok, f'M_{k}'] = v[i[ok]]
        if best is None:
            best = cur
        else:
            better = cur.M_vza.values < best.M_vza.fillna(np.inf).values
            best.loc[better] = cur.loc[better]
        print(f'  MASTER line {n}: {ok.sum()} trees matched')
    return best.drop(columns=['M_vza'])


def oracle(d):
    """Leave-one-out death rate of the other cohort trees in the pixel"""
    g = d.groupby('pix').died
    s, n = g.transform('sum'), g.transform('size')
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((s - d.died) / (n - 1)).where(n > 1)


def cv(d, groups, rng):
    """Spatial-block CV and leave-one-site-out for each feature group"""
    if len(d) > MAX_TREES:
        d = d.iloc[rng.choice(len(d), MAX_TREES, replace=False)]
    rows, preds = [], {}
    for name, cols in groups.items():
        p = np.full(len(d), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(d, groups=d.block):
            m = HistGradientBoostingClassifier(max_iter=300,
                                               learning_rate=0.05)
            m.fit(d.iloc[tr][cols], d.iloc[tr].died)
            p[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
        preds[name] = p
        pix = pd.DataFrame({'pix': d.pix.values, 'p': p,
                            'died': d.died.values})
        pix = pix.groupby('pix').agg(p=('p', 'mean'), died=('died', 'mean'),
                                     n=('p', 'size'))
        pix = pix[pix.n >= 5]
        rows.append(dict(features=name, cv='block1km', held_out='all',
                         n=len(d), auc=roc_auc_score(d.died, p),
                         ap=average_precision_score(d.died, p),
                         pixel_rho=spearmanr(pix.p, pix.died)[0]))
        for site in ('SOAP', 'TEAK'):
            tr, te = d[d.site != site], d[d.site == site]
            m = HistGradientBoostingClassifier(max_iter=300,
                                               learning_rate=0.05)
            m.fit(tr[cols], tr.died)
            q = m.predict_proba(te[cols])[:, 1]
            rows.append(dict(features=name, cv='leave_site', held_out=site,
                             n=len(te), auc=roc_auc_score(te.died, q),
                             ap=average_precision_score(te.died, q)))
        print(f'  {name:12s} AUC {rows[-3]["auc"]:.3f} AP {rows[-3]["ap"]:.3f}'
              f' pixel rho {rows[-3]["pixel_rho"]:.3f}')
    o = oracle(d)
    ok = o.notna()
    rows.append(dict(features='oracle_30m_pixel', cv='loo_within_pixel',
                     held_out='all', n=int(ok.sum()),
                     auc=roc_auc_score(d.died[ok], o[ok]),
                     ap=average_precision_score(d.died[ok], o[ok])))
    print(f'  30 m oracle AUC {rows[-1]["auc"]:.3f}')
    return rows, d, preds


def univariate(d, cols, cohort):
    rows = []
    strata = d.site + '_' + (d.elev // ELEV_BAND).astype(int).astype(str)
    for c in cols:
        ok = d[c].notna()
        if ok.sum() < 1000:
            continue
        auc = roc_auc_score(d.died[ok], d[c][ok])
        within, wts = [], []
        for s, g in d[ok].groupby(strata[ok]):
            if len(g) >= 500 and 0.05 < g.died.mean() < 0.95:
                within.append(roc_auc_score(g.died, g[c]))
                wts.append(len(g))
        rows.append(dict(cohort=cohort, feature=c, n=int(ok.sum()),
                         auc=auc,
                         auc_within_site_elev=(np.average(within, weights=wts)
                                               if within else np.nan),
                         n_strata=len(within)))
    return rows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--hs-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hemming_schroeder2023')
@click.option('--wdts', 'wdts_path', type=click.Path(path_type=Path),
              default=f'/Volumes/Earth04/ecopro/wdts/{AOI}_traits.nc')
@click.option('--composites', type=click.Path(path_type=Path),
              default=f'/Volumes/Earth04/ecopro/hls_composites/'
                      f'{AOI}_naip2020_L30_w25.nc')
@click.option('--master-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/master/MASTER_WDTS_SeptOct_2020')
def main(outputdir, hs_dir, wdts_path, composites, master_dir):
    os.makedirs(outputdir, exist_ok=True)
    rng = np.random.default_rng(0)
    trees = load_trees(hs_dir)
    print(f'{len(trees)} trees')

    w, _ = load_wdts(wdts_path)
    comp = xr.open_dataset(composites).load()
    transform = rasterio.Affine(*comp.attrs['transform'])
    shape = (comp.sizes['y'], comp.sizes['x'])
    assert np.allclose(w.x, comp.x) and np.allclose(w.y, comp.y)
    row, col = pixel_index(trees, transform, shape)
    trees['pix'] = row * shape[1] + col
    trees['block'] = (trees.site + '_' + (trees.x // BLOCK_M).astype(int)
                      .astype(str) + '_' + (trees.y // BLOCK_M).astype(int)
                      .astype(str))

    feats = {}
    feats.update(sample(comp, H_VARS, [2013, 2014], row, col, 'H_'))
    feats.update(sample(comp, H_VARS, [2015], row, col, 'H_'))
    feats.update(sample(comp, H_VARS, [2019, 2020], row, col, 'H_'))
    feats.update(sample(w, W_VARS, [2013, 2014], row, col, 'W_'))
    feats.update(sample(w, W_VARS, [2015], row, col, 'W_'))
    feats.update(sample(w, W_VARS, [2018], row, col, 'W_'))
    trees = trees.join(pd.DataFrame(feats, index=trees.index))
    trees = trees.join(master_features(trees, master_dir, comp.attrs['crs']))
    trees = trees[(row >= 0) & trees[S_VARS].notna().all(axis=1)]

    # Cohorts
    a = trees[(trees.live2013 == 1) & (trees.live2017 == trees.live2018)
              & trees.live2017.notna()].copy()
    a['died'] = (a.live2017 == 0).astype(int)
    b = trees[(trees.live2018 == 1) & (trees.live2019 == 1)
              & trees.live2021.notna()].copy()
    b['died'] = (b.live2021 == 0).astype(int)
    for n, d in (('A', a), ('B', b)):
        print(f'cohort {n}: {len(d)} trees, died {d.died.mean():.3f}; '
              f'{d.groupby("site").died.mean().round(3).to_dict()}')

    col_of = lambda pfx, yrs: [c for c in trees.columns if c.startswith(pfx)
                               and any(c.endswith(s) for s in yrs)]
    HA = col_of('H_', ['_2013', '_2014', '_d1413'])
    WA = col_of('W_', ['_2013', '_2014', '_d1413'])
    H15 = col_of('H_', ['_2015'])
    W15 = col_of('W_', ['_2015'])
    groups_a = {'S': S_VARS, 'H': HA, 'W': WA, 'S+H': S_VARS + HA,
                'S+W': S_VARS + WA, 'S+H+W': S_VARS + HA + WA,
                'S+H+W+2015': S_VARS + HA + WA + H15 + W15}
    HB = col_of('H_', ['_2019', '_2020', '_d2019'])
    WB = col_of('W_', ['_2018'])
    MB = [c for c in trees.columns if c.startswith('M_')]
    groups_b = {'S': S_VARS, 'H': HB, 'W2018': WB, 'M': MB,
                'S+H': S_VARS + HB, 'S+W2018': S_VARS + WB,
                'S+M': S_VARS + MB, 'S+H+M': S_VARS + HB + MB,
                'S+H+W2018+M': S_VARS + HB + WB + MB}

    res, uni = [], []
    for n, d, groups in (('A', a, groups_a), ('B', b, groups_b)):
        print(f'cohort {n} CV')
        rows, dsub, _ = cv(d, groups, rng)
        res += [dict(r, cohort=n) for r in rows]
        allcols = sorted(set(sum(groups.values(), [])) - {'is_soap'})
        uni += univariate(dsub, allcols, n)
    res = pd.DataFrame(res)
    uni = pd.DataFrame(uni)
    res.to_csv(outputdir / 'predisposition_cv.csv', index=False)
    uni.to_csv(outputdir / 'predisposition_univariate.csv', index=False)
    print(res.round(3).to_string())
    uni['sep'] = (uni.auc_within_site_elev - 0.5).abs()
    for n in ('A', 'B'):
        print(uni[uni.cohort == n].sort_values('sep', ascending=False)
              .head(25).round(3).to_string())

    with PdfPages(outputdir / 'predisposition.pdf') as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, n in zip(axes, ('A', 'B')):
            r = res[(res.cohort == n) & (res.held_out == 'all')]
            ax.barh(r.features, r.auc, color=['0.3' if 'oracle' in f else
                                              'tab:blue' for f in r.features])
            for y, v in enumerate(r.auc):
                ax.text(v, y, f' {v:.3f}', va='center', fontsize=8)
            ax.set_xlim(0.5, 1)
            ax.set_xlabel('AUC (1 km block CV)')
            ax.set_title({'A': 'Cohort A: live 2013 -> dead 2017+2018\n'
                               'features from 2013-2014',
                          'B': 'Cohort B: live 2018-19 -> dead 2021\n'
                               'features 2018-2020'}[n])
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'predisposition_auc.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
