#!/usr/bin/env python
"""
Predisposition test at NEON SOAP/TEAK: do remote-sensing indicators measured
BEFORE trees die predict the *fraction* of trees in a 30 m (or 90 m) cell
that go on to die, beyond stand structure, topography and climate?

Trees and fates come from Hemming-Schroeder et al. (2023): ~1M lidar-
segmented crowns with a live/dead status per year (relative greenness within
the crown from NEON imagery; trees_<year>_rgreen.shp) and per-tree site
covariates (feature_vars_labels.csv). Two cohorts, each target being the
fraction of the cell's cohort trees that died (cells need >= MIN_TREES
cohort trees; models are weighted by that count):

  A  2015-17 die-off: trees live in 2013; died = dead in both 2017 and 2018,
     survived = live in both (inconsistent labels dropped)
  B  2020-22 drought wave: trees live in 2018 and 2019; died = dead in 2021

Feature groups, as cell means over all 2013 lidar trees in the cell:

  S   stand/site: tree count, mean / p90 / max 2013 height, fraction of
      trees > 30 m, mean crown area, fraction already dead in 2013, and
      mean tpa, neighbour distance, cover, elevation, slope, aspect,
      climate normals (ppt, tmax, vpdmax), granite, distance to rivers, site
  H   HLS L30 (NAIP-date composites) bands/indices
  W   WDTS AVIRIS-C traits (14 trait means + QC fractions)
  C   WDTS canopy water indicators from the 15 m reflectance
      (fetch_wdts_cwc.py): EWT at 980 and 1200 nm, NDWI, 1200 nm band depth
  M   MASTER 2020-10-15 at the nearest near-nadir pixel (cohort B only)

Cohort A uses 2013 and 2014 and their change as the pre-mortality years;
2015 (June, as the die-off began) is added separately as a lead-time
sensitivity. Cohort B uses HLS 2019/2020, WDTS 2018 and MASTER Fall 2020.

Evaluation: HistGradientBoostingRegressor on the cell death fraction, 5-fold
GroupKFold on 1 km blocks and leave-one-site-out; tree-weighted R^2 and
Spearman rho. Univariate Spearman is reported pooled and averaged within
site x 200 m elevation strata.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.backends.backend_pdf import PdfPages
from pyproj import Transformer

from hls_annual_composites import ROLES, INDICES
from proto_hls_vs_wdts import WDTS_TRAITS, load_wdts

AOI = 'neon_soap_teak'
TREE_SITE = ['tpa', 'meank', 'cover', 'elev', 'slope', 'northness',
             'eastness', 'ppt', 'tmax', 'vpdmax', 'granite', 'd_rivers',
             'is_soap']
H_VARS = ROLES + INDICES
W_VARS = [f'{t}_mean' for t in WDTS_TRAITS] + ['qc_all', 'qc_fc']
C_VARS = ['ewt980', 'ewt1200', 'ndwi', 'bd1200']
MASTER_LINES = [1, 2, 3, 4]
MIN_TREES = 5
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
    return best.drop(columns=['M_vza'])


def cells(trees, cohort, k, feat_cols):
    """Cell table at k x k pixels: stand/site summaries over all trees,
    feature means, and the cohort death fraction"""
    cid = (trees.row // k) * 100000 + trees.col // k
    t = trees.assign(_tall=(trees.zmax2013 > 30).astype(float),
                     _dead13=(trees.live2013 == 0).astype(float),
                     _lab13=trees.live2013.notna().astype(float))
    g = t.groupby(cid)
    df = pd.DataFrame({
        'n_trees': g.size(),
        'h_mean': g.zmax2013.mean(),
        'h_p90': g.zmax2013.quantile(0.9),
        'h_max': g.zmax2013.max(),
        'frac_tall': g._tall.mean(),
        'ca_mean': g.ca2013.mean(),
        'dead2013': g._dead13.sum() / g._lab13.sum().clip(lower=1),
    })
    df = df.join(g[TREE_SITE + feat_cols].mean())
    c = cohort.groupby(cid.loc[cohort.index]).died
    df['n_cohort'], df['died'] = c.size(), c.mean()
    df = df[df.n_cohort >= MIN_TREES].copy()
    first = g.first()
    df['site'] = first.site.loc[df.index]
    df['block'] = (df.site + '_' + (first.x.loc[df.index] // BLOCK_M)
                   .astype(int).astype(str) + '_' +
                   (first.y.loc[df.index] // BLOCK_M).astype(int).astype(str))
    return df


S_CELL = ['n_trees', 'h_mean', 'h_p90', 'h_max', 'frac_tall', 'ca_mean',
          'dead2013'] + TREE_SITE


def wr2(y, p, w):
    return r2_score(y, p, sample_weight=w)


def cv(d, groups, scale_m, cohort):
    rows = []
    for name, cols in groups.items():
        p = np.full(len(d), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(d, groups=d.block):
            m = HistGradientBoostingRegressor(max_iter=300,
                                              learning_rate=0.05)
            m.fit(d.iloc[tr][cols], d.iloc[tr].died,
                  sample_weight=d.iloc[tr].n_cohort)
            p[te] = m.predict(d.iloc[te][cols])
        rows.append(dict(cohort=cohort, scale_m=scale_m, features=name,
                         cv='block1km', held_out='all', n=len(d),
                         r2=wr2(d.died, p, d.n_cohort),
                         rho=spearmanr(d.died, p)[0]))
        for site in ('SOAP', 'TEAK'):
            tr, te = d[d.site != site], d[d.site == site]
            m = HistGradientBoostingRegressor(max_iter=300,
                                              learning_rate=0.05)
            m.fit(tr[cols], tr.died, sample_weight=tr.n_cohort)
            q = m.predict(te[cols])
            rows.append(dict(cohort=cohort, scale_m=scale_m, features=name,
                             cv='leave_site', held_out=site, n=len(te),
                             r2=wr2(te.died, q, te.n_cohort),
                             rho=spearmanr(te.died, q)[0]))
        r = rows[-3]
        print(f'  [{cohort} {scale_m} m] {name:14s} R2 {r["r2"]:.3f} '
              f'rho {r["rho"]:.3f}  (leave-site SOAP R2 {rows[-2]["r2"]:.3f}'
              f', TEAK {rows[-1]["r2"]:.3f})')
    return rows


def univariate(d, cols, cohort, scale_m):
    rows = []
    strata = d.site + '_' + (d.elev // ELEV_BAND).astype(int).astype(str)
    for c in cols:
        ok = d[c].notna()
        if ok.sum() < 200:
            continue
        within, wts = [], []
        for s, g in d[ok].groupby(strata[ok]):
            if len(g) >= 100 and g.died.std() > 0:
                within.append(spearmanr(g[c], g.died)[0])
                wts.append(len(g))
        rows.append(dict(cohort=cohort, scale_m=scale_m, feature=c,
                         n=int(ok.sum()),
                         rho=spearmanr(d[c][ok], d.died[ok])[0],
                         rho_within_site_elev=(np.average(within, weights=wts)
                                               if within else np.nan),
                         n_strata=len(within)))
    return rows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--hs-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hemming_schroeder2023')
@click.option('--wdts', 'wdts_path', type=click.Path(path_type=Path),
              default=f'/Volumes/Earth04/ecopro/wdts/{AOI}_traits.nc')
@click.option('--cwc', 'cwc_path', type=click.Path(path_type=Path),
              default=f'/Volumes/Earth04/ecopro/wdts/{AOI}_cwc.nc')
@click.option('--composites', type=click.Path(path_type=Path),
              default=f'/Volumes/Earth04/ecopro/hls_composites/'
                      f'{AOI}_naip2020_L30_w25.nc')
@click.option('--master-dir', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/master/MASTER_WDTS_SeptOct_2020')
@click.option('--scale', 'scales', multiple=True, type=int, default=[1, 3],
              show_default=True, help='Cell size in 30 m pixels')
def main(outputdir, hs_dir, wdts_path, cwc_path, composites, master_dir,
         scales):
    os.makedirs(outputdir, exist_ok=True)
    trees = load_trees(hs_dir)
    print(f'{len(trees)} trees')

    w, _ = load_wdts(wdts_path)
    cwc = xr.open_dataset(cwc_path).load()
    comp = xr.open_dataset(composites).load()
    transform = rasterio.Affine(*comp.attrs['transform'])
    shape = (comp.sizes['y'], comp.sizes['x'])
    assert np.allclose(w.x, comp.x) and np.allclose(cwc.x, comp.x)
    assert np.allclose(w.y, comp.y) and np.allclose(cwc.y, comp.y)
    row, col = pixel_index(trees, transform, shape)
    trees['row'], trees['col'] = row, col

    feats = {}
    feats.update(sample(comp, H_VARS, [2013, 2014], row, col, 'H_'))
    feats.update(sample(comp, H_VARS, [2015], row, col, 'H_'))
    feats.update(sample(comp, H_VARS, [2019, 2020], row, col, 'H_'))
    feats.update(sample(w, W_VARS, [2013, 2014], row, col, 'W_'))
    feats.update(sample(w, W_VARS, [2015], row, col, 'W_'))
    feats.update(sample(w, W_VARS, [2018], row, col, 'W_'))
    cy = cwc.year.values.tolist()
    feats.update(sample(cwc, C_VARS, [2013, 2014], row, col, 'C_'))
    for y in (2015, 2018):
        if y in cy:
            feats.update(sample(cwc, C_VARS, [y], row, col, 'C_'))
    if 2015 in cy:
        for v in C_VARS:
            feats[f'C_{v}_d1513'] = feats[f'C_{v}_2015'] - feats[f'C_{v}_2013']
    trees = trees.join(pd.DataFrame(feats, index=trees.index))
    trees = trees.join(master_features(trees, master_dir, comp.attrs['crs']))
    trees = trees[(row >= 0) & trees[TREE_SITE + ['zmax2013']]
                  .notna().all(axis=1)]

    a = trees[(trees.live2013 == 1) & (trees.live2017 == trees.live2018)
              & trees.live2017.notna()].copy()
    a['died'] = (a.live2017 == 0).astype(float)
    b = trees[(trees.live2018 == 1) & (trees.live2019 == 1)
              & trees.live2021.notna()].copy()
    b['died'] = (b.live2021 == 0).astype(float)

    col_of = lambda pfx, yrs: [c for c in trees.columns if c.startswith(pfx)
                               and any(c.endswith(s) for s in yrs)]
    HA = col_of('H_', ['_2013', '_2014', '_d1413'])
    WA = col_of('W_', ['_2013', '_2014', '_d1413'])
    CA = col_of('C_', ['_2013', '_2014', '_d1413'])
    H15, W15 = col_of('H_', ['_2015']), col_of('W_', ['_2015'])
    C15 = col_of('C_', ['_2015', '_d1513'])
    L15 = H15 + W15 + C15
    groups_a = {'S': S_CELL, 'H': HA, 'W': WA, 'C': CA,
                'S+H': S_CELL + HA, 'S+W': S_CELL + WA, 'S+C': S_CELL + CA,
                'S+H+C': S_CELL + HA + CA, 'S+H+W+C': S_CELL + HA + WA + CA,
                'S+H+W+C+2015': S_CELL + HA + WA + CA + L15,
                'S+H+H2015': S_CELL + HA + H15, 'S+W+W2015': S_CELL + WA + W15,
                'S+C+C2015': S_CELL + CA + C15,
                'S+H+H2015+C+C2015': S_CELL + HA + H15 + CA + C15}
    HB = col_of('H_', ['_2019', '_2020', '_d2019'])
    WB = col_of('W_', ['_2018'])
    CB = col_of('C_', ['_2018'])
    MB = [c for c in trees.columns if c.startswith('M_')]
    groups_b = {'S': S_CELL, 'H': HB, 'W2018': WB, 'C2018': CB, 'M': MB,
                'S+H': S_CELL + HB, 'S+W2018': S_CELL + WB,
                'S+C2018': S_CELL + CB, 'S+M': S_CELL + MB,
                'S+H+C2018': S_CELL + HB + CB,
                'S+H+W2018+C2018+M': S_CELL + HB + WB + CB + MB}
    fcols = sorted(set(HA + WA + CA + L15 + HB + WB + CB + MB))

    res, uni = [], []
    for k in scales:
        for n, coh, groups in (('A', a, groups_a), ('B', b, groups_b)):
            d = cells(trees, coh, k, fcols)
            print(f'cohort {n}, {30 * k} m: {len(d)} cells, mean death '
                  f'fraction {np.average(d.died, weights=d.n_cohort):.3f} '
                  f'({d.groupby("site").died.mean().round(3).to_dict()})')
            d.to_csv(outputdir / f'cells_{n}_{30 * k}m.csv')
            res += cv(d, groups, 30 * k, n)
            allc = sorted(set(sum(groups.values(), [])) - {'is_soap'})
            uni += univariate(d, allc, n, 30 * k)
    res = pd.DataFrame(res)
    uni = pd.DataFrame(uni)
    res.to_csv(outputdir / 'predisposition_cv.csv', index=False)
    uni.to_csv(outputdir / 'predisposition_univariate.csv', index=False)
    print(res[res.cv == 'block1km'].pivot_table(
        index=['cohort', 'features'], columns='scale_m',
        values=['r2', 'rho']).round(3).to_string())
    uni['sep'] = uni.rho_within_site_elev.abs()
    for n in ('A', 'B'):
        u = uni[(uni.cohort == n) & (uni.scale_m == 30)]
        print(u.sort_values('sep', ascending=False).head(30)
              .round(3).to_string())

    with PdfPages(outputdir / 'predisposition.pdf') as pdf:
        for k in scales:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, n in zip(axes, ('A', 'B')):
                r = res[(res.cohort == n) & (res.cv == 'block1km')
                        & (res.scale_m == 30 * k)]
                ax.barh(r.features, r.r2, color='tab:blue')
                for yy, (v, rho) in enumerate(zip(r.r2, r.rho)):
                    ax.text(max(v, 0), yy, f' R2 {v:.2f}, rho {rho:.2f}',
                            va='center', fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_xlabel('tree-weighted R2 (1 km block CV)')
                ax.set_title({
                    'A': f'A: fraction of 2013-live trees dead 2017-18\n'
                         f'features 2013-14; {30 * k} m cells',
                    'B': f'B: fraction of 2018-19-live trees dead 2021\n'
                         f'features 2018-20; {30 * k} m cells'}[n])
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outputdir / f'predisposition_r2_{30 * k}m.png',
                        dpi=150)
            plt.close(fig)


if __name__ == '__main__':
    main()
