#!/usr/bin/env python
"""
A second, NAIP-based tree-mortality product at NEON SOAP/TEAK, checked
against the NEON lidar-tracked trees (Hemming-Schroeder et al. 2023, "HS").

Stovall, Shugart & Yang (2019; fetch_stovall2019.py) segmented about 2 M
crowns from the NEON 2013 lidar canopy height model and classified each
crown's NAIP pixels as dead or live in 2009, 2010, 2012, 2014 and 2016. A
crown is dead above 37.5% dead pixels, and a tree is dropped once dead, so
mort_year is each tree's first dead year. The HS cohort used everywhere
else (proto_trait_dynamics.lidar_cohort) is trees live in 2013 with the same
status in 2017 and 2018, died = dead in both. The closest Stovall cohort:
    cohort  mort_year not in {2009, 2010, 2012} (alive through 2012)
    died    mort_year in {2014, 2016}
Stovall ends in 2016, so trees that died in 2017 count only in HS.

Flight-line mask. One oblique, heavily shaded NAIP 2016 flight line at TEAK
makes Stovall's 2016 estimates unreliable there. Queally et al. (2025;
fetch_queally2025.py) published their 30 m grid of both products with that
area masked out; it is on the AOI's 30 m lattice. The mask here is the set
of pixels where both products have trees but the Queally grid is empty, kept
as connected areas of at least MIN_MASK_PX pixels (scattered single empty
pixels are not the flight line).

Outputs (OUTPUTDIR):
    qa.csv                       Stovall trees per site, mort_year x dead,
                                 extent, CRS offset to the HS treetops
    flightline_mask.tif          the mask on the AOI grid
    flightline_quads.csv         mask pixels per NAIP 2016 quad and date
    queally_repro.csv            our 30 m Stovall grid vs the Queally grid
    stovall_<aoi>_<m>m.csv       cells (30, 90, 270 m): stov_frac/stov_n,
                                 HS mort_frac/mort_n (same gridding),
                                 stov_m16 (mean crown dead fraction 2016,
                                 all trees), oblique_frac (flight-line
                                 mask), q_frac (inside the Queally grid),
                                 q_m16/q_m17 (the Queally grid), site
    tree_matches.parquet         each Stovall tree's nearest HS 2017
                                 treetop, inside its 2013 crown or not
    confusion.csv                Stovall died vs HS died on matched trees,
                                 by site x height class x mask
    match_bias.csv               mortality of matched vs unmatched trees
    hs_region.csv                the HS overlap-region summary (their
                                 Table S3 layout)
    agreement.csv                cell-level ρ, r², 1:1 R² and level
                                 difference with 1 km block-bootstrap CIs
    agreement_90m.png            scatter and difference map

    python proto_stovall_check.py $E/hls_results/stovall_check
"""
import json
import click
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import response_common as rc
from proto_predisposition_neon import load_trees, pixel_index
from proto_trait_dynamics import lidar_cohort, MIN_TREES

AOI = 'neon_soap_teak'
STOV = rc.E / 'stovall2019' / 'ALLtrees_v2.csv'
QUEALLY = rc.E / 'queally2025'
HS = rc.E / 'hemming_schroeder2023'
SITE_X = 310000  # SOAP west of this easting, TEAK east (as HS split them)
EARLY = ('2009', '2010', '2012')
DIED = ('2014', '2016')
DEAD_THRESHOLD = 0.375
MIN_MASK_PX = 11  # 1 ha of 30 m pixels
H_BINS = [0, 5, 15, 30, np.inf]
H_LABELS = ['<5', '5-15', '15-30', '>30']
SCALES = (1, 3, 9)


def load_stovall(path=STOV):
    cols = ['zone', 'zmax', 'x', 'y', 'dead', 'mort_year', 'mort2016']
    d = pd.read_csv(path, usecols=cols, dtype={'mort_year': str})
    d['site'] = np.where(d.x < SITE_X, 'SOAP', 'TEAK')
    d['cohort'] = ~d.mort_year.isin(EARLY)
    d['died'] = d.mort_year.isin(DIED)
    d['hclass'] = pd.cut(d.zmax, H_BINS, labels=H_LABELS, right=False)
    return d


def load_hs_crowns():
    """HS trees with cohort-A status, 2017 treetops and 2013 crowns"""
    df = load_trees(HS)
    df['cohort'] = ((df.live2013 == 1) & (df.live2017 == df.live2018)
                    & df.live2017.notna())
    df['died'] = df.cohort & (df.live2017 == 0)
    vec = HS / 'data/deliverables/vector'
    t17 = pyogrio.read_dataframe(vec / 'trees_2017_rgreen.shp',
                                 read_geometry=False,
                                 columns=['treeID', 'xlas2017', 'ylas2017']
                                 ).set_index('treeID')
    df = df.join(t17)
    c13 = pyogrio.read_dataframe(vec / 'trees_2013_rgreen.shp',
                                 columns=['treeID']).set_index('treeID')
    df['crown'] = c13.geometry.reindex(df.index).values
    return df


def qa(st, hs, transform, shape):
    rows = []
    for s, g in st.groupby('site'):
        rows.append(dict(item='trees', site=s, value=len(g)))
        rows.append(dict(item='cohort trees', site=s, value=int(g.cohort.sum())))
        rows.append(dict(item='cohort died frac', site=s,
                         value=g.died[g.cohort].mean()))
    for (y, dd), n in st.groupby(['mort_year', 'dead']).size().items():
        rows.append(dict(item=f'mort_year={y} dead={dd}', site='all',
                         value=n))
    rows.append(dict(item='dead == (mort_year != Live)', site='all',
                     value=float(((st.mort_year != 'Live') ==
                                  (st.dead == 1)).mean())))
    live = st.mort_year == 'Live'
    rows.append(dict(item='Live with mort2016 > 0.375', site='all',
                     value=float((st.mort2016[live] > DEAD_THRESHOLD).mean())))
    rows.append(dict(item='dead 2016 with mort2016 > 0.375', site='all',
                     value=float((st.mort2016[st.mort_year == '2016'] >
                                  DEAD_THRESHOLD).mean())))
    row, _ = pixel_index(st, transform, shape)
    rows.append(dict(item='inside AOI grid', site='all',
                     value=float((row >= 0).mean())))
    for c in ('x', 'y'):
        rows.append(dict(item=f'{c} min', site='all', value=st[c].min()))
        rows.append(dict(item=f'{c} max', site='all', value=st[c].max()))
    # inside the HS study-region polygons
    reg = [pyogrio.read_dataframe(HS / 'data/intermediate/study_region' /
                                  f'{s}_mask_polygon.shp').union_all()
           for s in ('soap', 'teak')]
    inside = shapely.contains_xy(shapely.union_all(reg), st.x.values,
                                 st.y.values)
    st['in_hs_region'] = inside
    rows.append(dict(item='inside HS study region', site='all',
                     value=float(inside.mean())))
    # CRS: offset of tall Stovall trees to the nearest HS 2013 treetop
    tall = st[st.zmax >= 30]
    hst = hs[hs.zmax2013 >= 25]
    dist, i = cKDTree(np.c_[hst.x, hst.y]).query(np.c_[tall.x, tall.y])
    near = dist < 3
    dx = hst.x.values[i[near]] - tall.x.values[near]
    dy = hst.y.values[i[near]] - tall.y.values[near]
    for s in ('SOAP', 'TEAK'):
        m = (tall.site.values[near] == s)
        rows += [dict(item='CRS median dx (HS - Stovall), m', site=s,
                      value=float(np.median(dx[m]))),
                 dict(item='CRS median dy (HS - Stovall), m', site=s,
                      value=float(np.median(dy[m]))),
                 dict(item='CRS median distance, m', site=s,
                      value=float(np.median(dist[near][m]))),
                 dict(item='tall trees with an HS treetop < 3 m', site=s,
                      value=float(near[tall.site.values == s].mean()))]
    return pd.DataFrame(rows)


def counts(df, transform, shape, weights=None):
    row, col = pixel_index(df, transform, shape)
    ok = row >= 0
    idx = row[ok] * shape[1] + col[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    return np.bincount(idx, weights=w,
                       minlength=shape[0] * shape[1]).reshape(shape)


def queally_on_grid(transform, shape):
    """Queally mort_16, mort_17 and site on the AOI grid (same lattice)"""
    with rasterio.open(QUEALLY / 'mortality.tif') as r:
        q = r.read(masked=True).astype(float).filled(np.nan)
        qt = r.transform
    with rasterio.open(QUEALLY / 'sites.tif') as r:
        s = r.read(1, masked=True).astype(float).filled(np.nan)
    c0 = (qt.c - transform.c) / transform.a
    r0 = (qt.f - transform.f) / transform.e
    assert qt.a == transform.a and qt.e == transform.e
    assert c0 == round(c0) and r0 == round(r0), 'not on the AOI lattice'
    c0, r0 = int(c0), int(r0)
    out = np.full((3,) + shape, np.nan)
    inq = np.zeros(shape, bool)
    rs = slice(max(r0, 0), min(r0 + q.shape[1], shape[0]))
    cs = slice(max(c0, 0), min(c0 + q.shape[2], shape[1]))
    src = (slice(rs.start - r0, rs.stop - r0),
           slice(cs.start - c0, cs.stop - c0))
    out[0][rs, cs], out[1][rs, cs] = q[0][src], q[1][src]
    out[2][rs, cs] = s[src]
    inq[rs, cs] = True
    return out[0], out[1], out[2], inq


def flightline_mask(st_all, hs_all, q16, inq):
    """Pixels with trees in both products that the Queally grid leaves
    empty, in connected areas >= MIN_MASK_PX"""
    cand = (st_all > 0) & (hs_all > 0) & inq & ~np.isfinite(q16)
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel())
    keep = size >= MIN_MASK_PX
    keep[0] = False
    return keep[lab], cand


def mask_by_quad(mask, transform, epsg):
    """Mask pixels per NAIP 2016 quad (from the local STAC manifest)"""
    items = json.loads((rc.E / 'naip' / AOI / '_items.json').read_text())
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    rows, cols = np.nonzero(mask)
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    out = []
    for it in items['items']:
        p = it['properties']
        if p.get('naip:year') != '2016':
            continue
        b = it['bbox']
        x0, y0 = tf.transform(b[0], b[1])
        x1, y1 = tf.transform(b[2], b[3])
        m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        out.append(dict(quad=it['id'], date=p['datetime'][:10],
                        mask_px=int(m.sum())))
    return pd.DataFrame(out).sort_values('mask_px', ascending=False)


def cells(layers, valid, k, transform):
    d = rc.cell_table(layers, valid, k)
    d['stov_frac'] = d.st_died / d.st_n
    d['stov_n'] = d.st_n * d.n_px
    d['mort_frac'] = d.hs_died / d.hs_n
    d['mort_n'] = d.hs_n * d.n_px
    d['stov_m16'] = d.st_m16 / d.st_all
    for f, n in (('stov_frac', 'stov_n'), ('mort_frac', 'mort_n')):
        d.loc[~(d[n] >= MIN_TREES), [f, n]] = np.nan
    d.loc[~(d.st_all * d.n_px >= MIN_TREES), 'stov_m16'] = np.nan
    x = transform.c + (d.cell_col + 0.5) * k * transform.a
    d['site'] = np.where(x < SITE_X, 'SOAP', 'TEAK')
    keep = ['cell_row', 'cell_col', 'n_px', 'block1000', 'block5000', 'site',
            'stov_frac', 'stov_n', 'mort_frac', 'mort_n', 'stov_m16',
            'oblique_frac', 'q_frac', 'q_m16', 'q_m17']
    return d[keep]


def match_trees(st, hs):
    """HS method (align_trees/tree_match_02_matching.R): nearest 2017
    treetop, matched if the point lies in that tree's 2013 crown. Also
    flags points inside any 2013 crown (the rule its readme describes)."""
    h = hs[hs.xlas2017.notna() & hs.crown.notna()]
    d, i = cKDTree(np.c_[h.xlas2017, h.ylas2017]).query(np.c_[st.x, st.y])
    crowns = np.asarray(h.crown.values, dtype=object)
    contain = shapely.contains_xy(crowns[i], st.x.values, st.y.values)
    tree = shapely.STRtree(crowns)
    pts = shapely.points(st.x.values, st.y.values)
    pi, _ = tree.query(pts, predicate='within')
    any_c = np.zeros(len(st), bool)
    any_c[pi] = True
    m = pd.DataFrame({
        'zone': st.zone.values, 'x': st.x.values, 'y': st.y.values,
        'zmax': st.zmax.values, 'site': st.site.values,
        'hclass': st.hclass.values, 'dead': st.dead.values,
        'mort_year': st.mort_year.values, 'st_cohort': st.cohort.values,
        'st_died': st.died.values, 'in_hs_region': st.in_hs_region.values,
        'hs_treeID': h.index.values[i], 'd': d, 'contain': contain,
        'any_contain': any_c,
        'hs_live2017': h.live2017.values[i],
        'hs_cohort': h.cohort.values[i], 'hs_died': h.died.values[i],
        'hs_zmax2013': h.zmax2013.values[i]})
    return m


def confusion(m):
    rows = []
    for mask in ('none', 'flightline'):
        mm = m if mask == 'none' else m[~m.oblique]
        b = mm[mm.contain & mm.st_cohort & mm.hs_cohort]
        for site in ('SOAP', 'TEAK', 'all'):
            g0 = b if site == 'all' else b[b.site == site]
            for hc in H_LABELS + ['all']:
                g = g0 if hc == 'all' else g0[g0.hclass == hc]
                if len(g) == 0:
                    continue
                s, h = g.st_died.values, g.hs_died.values
                n = len(g)
                tab = {'both_live': int((~s & ~h).sum()),
                       'stov_died_only': int((s & ~h).sum()),
                       'hs_died_only': int((~s & h).sum()),
                       'both_died': int((s & h).sum())}
                po = (tab['both_live'] + tab['both_died']) / n
                pe = s.mean() * h.mean() + (1 - s.mean()) * (1 - h.mean())
                rows.append(dict(mask=mask, site=site, hclass=hc, n=n, **tab,
                                 stov_rate=s.mean(), hs_rate=h.mean(),
                                 agreement=po,
                                 kappa=(po - pe) / (1 - pe) if pe < 1
                                 else np.nan))
    return pd.DataFrame(rows)


def match_bias(m, hs):
    """Mortality among matched vs unmatched trees, both sides"""
    rows = []
    c = m[m.st_cohort]
    matched_hs = set(m.hs_treeID[m.contain])
    hsc = hs[hs.cohort].copy()
    hsc['matched'] = hsc.index.isin(matched_hs)
    hsc['site'] = np.where(hsc.x < SITE_X, 'SOAP', 'TEAK')
    hsc['hclass'] = pd.cut(hsc.zmax2013, H_BINS, labels=H_LABELS,
                           right=False)
    # HS trees only within the area Stovall covers (its 30 m pixels)
    for site in ('SOAP', 'TEAK', 'all'):
        for hc in H_LABELS + ['all']:
            g = c if site == 'all' else c[c.site == site]
            g = g if hc == 'all' else g[g.hclass == hc]
            if len(g):
                rows.append(dict(product='stovall', site=site, hclass=hc,
                                 n=len(g), matched_frac=g.contain.mean(),
                                 died_matched=g.st_died[g.contain].mean(),
                                 died_unmatched=g.st_died[~g.contain].mean()))
            g = hsc[hsc.in_stov] if site == 'all' else \
                hsc[hsc.in_stov & (hsc.site == site)]
            g = g if hc == 'all' else g[g.hclass == hc]
            if len(g):
                rows.append(dict(product='hs', site=site, hclass=hc,
                                 n=len(g), matched_frac=g.matched.mean(),
                                 died_matched=g.died[g.matched].mean(),
                                 died_unmatched=g.died[~g.matched].mean()))
    return pd.DataFrame(rows)


def hs_region(m):
    """The overlap-region summary in the layout of HS Table S3: Stovall
    `dead` (by 2016, all trees) vs the HS 2017 label, matched =
    contain & label present"""
    r = m[m.in_hs_region]
    rows = []
    for site, g in r.groupby('site'):
        mt = g.contain & g.hs_live2017.notna()
        rows.append(dict(site=site, stovall_trees=len(g),
                         matched=int(mt.sum()),
                         stov_dead=g.dead.mean(),
                         stov_dead_matched=g.dead[mt].mean(),
                         stov_dead_unmatched=g.dead[~mt].mean(),
                         hs_dead2017_matched=1 - g.hs_live2017[mt].mean(),
                         both_dead=int((mt & (g.dead == 1) &
                                        (g.hs_live2017 == 0)).sum()),
                         both_live=int((mt & (g.dead == 0) &
                                        (g.hs_live2017 == 1)).sum())))
    return pd.DataFrame(rows)


def boot_stats(a, b, blocks, n_boot, seed=0):
    """ρ, r² and mean(a - b), each with a block-bootstrap 95% CI"""
    codes, inv = np.unique(blocks, return_inverse=True)
    mult = rc.block_draws(len(codes), n_boot, seed)
    est = (spearmanr(a, b)[0], np.corrcoef(a, b)[0, 1] ** 2,
           np.mean(a - b))
    draws = []
    for mu in mult:
        w = mu[inv]
        keep = w > 0
        rep = np.repeat(np.flatnonzero(keep), w[keep].astype(int))
        aa, bb = a[rep], b[rep]
        draws.append((spearmanr(aa, bb)[0], np.corrcoef(aa, bb)[0, 1] ** 2,
                      np.mean(aa - bb)))
    lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
    return est, lo, hi


def agreement(tabs, n_boot):
    rows = []
    pairs = [('stov_frac', 'mort_frac', 'cohort fractions'),
             ('stov_m16', 'mort_frac', 'crown dead 2016 vs HS'),
             ('q_m16', 'q_m17', 'Queally grid'),
             ('stov_m16', 'q_m16', 'ours vs Queally, Stovall')]
    for k, d in tabs.items():
        for mask in ('none', 'flightline'):
            dd = d if mask == 'none' else d[(d.oblique_frac == 0) &
                                            (d.q_frac == 1)]
            for a, b, label in pairs:
                if label.startswith('ours') and mask == 'flightline':
                    continue
                for site in ('SOAP', 'TEAK', 'all'):
                    g = dd if site == 'all' else dd[dd.site == site]
                    g = g[g[a].notna() & g[b].notna()]
                    if len(g) < 50:
                        continue
                    blocks = g.block1000.values
                    est, lo, hi = boot_stats(g[a].values, g[b].values,
                                             blocks, n_boot)
                    r2 = rc.bootstrap_r2(g[a].values, {'b': g[b].values},
                                         blocks, n_boot=n_boot)['b']
                    rows.append(dict(
                        scale_m=rc.RES * k, mask=mask, site=site, a=a, b=b,
                        pair=label, n=len(g), mean_a=g[a].mean(),
                        mean_b=g[b].mean(),
                        rho=est[0], rho_lo=lo[0], rho_hi=hi[0],
                        r2=est[1], r2_lo=lo[1], r2_hi=hi[1],
                        r2_11=r2[0], r2_11_lo=r2[1], r2_11_hi=r2[2],
                        diff=est[2], diff_lo=lo[2], diff_hi=hi[2]))
                    click.echo(f'  {rc.RES * k:4d} m {mask:10s} {site:4s} '
                               f'{label:26s} n={len(g):6d} ρ={est[0]:.2f} '
                               f'[{lo[0]:.2f},{hi[0]:.2f}] r²={est[1]:.2f} '
                               f'1:1 R²={r2[0]:+.2f} Δ={est[2]:+.3f} '
                               f'[{lo[2]:+.3f},{hi[2]:+.3f}]')
    return pd.DataFrame(rows)


def plot(d, shape, k, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for site, c in (('SOAP', 'C0'), ('TEAK', 'C1')):
        g = d[(d.site == site) & d.stov_frac.notna() & d.mort_frac.notna()
              & (d.oblique_frac == 0) & (d.q_frac == 1)]
        ax[0].scatter(g.mort_frac, g.stov_frac, s=2, alpha=0.3, c=c,
                      label=f'{site} (n={len(g)})')
    ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax[0].set(xlabel='HS cohort died fraction',
              ylabel='Stovall cohort died fraction',
              title=f'{rc.RES * k} m cells, outside the flight-line mask')
    ax[0].legend(markerscale=5)
    grid = np.full((shape[0] // k + 1, shape[1] // k + 1), np.nan)
    ok = d.stov_frac.notna() & d.mort_frac.notna()
    grid[d.cell_row[ok], d.cell_col[ok]] = (d.stov_frac - d.mort_frac)[ok]
    im = ax[1].imshow(grid, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                      interpolation='nearest')
    plt.colorbar(im, ax=ax[1], label='Stovall − HS')
    ax[1].set_title('difference (all cells)')
    mg = np.full(grid.shape, np.nan)
    ok = d.stov_frac.notna() | d.mort_frac.notna()
    mg[d.cell_row[ok], d.cell_col[ok]] = d.oblique_frac[ok]
    im = ax[2].imshow(mg, cmap='viridis', vmin=0, vmax=1,
                      interpolation='nearest')
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title('flight-line mask fraction (cells with trees)')
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--n-boot', default=1000, show_default=True)
@click.option('--skip-matching', is_flag=True,
              help='Skip the tree-level matching (cells and agreement only)')
def main(outputdir, n_boot, skip_matching):
    outputdir.mkdir(parents=True, exist_ok=True)
    transform, shape, epsg = rc.aoi_info(AOI)
    st = load_stovall()
    hs = load_hs_crowns()
    hs = hs[hs.x.notna()]
    click.echo(f'Stovall {len(st)} trees, HS {len(hs)} trees')
    q = qa(st, hs, transform, shape)
    q.to_csv(outputdir / 'qa.csv', index=False)
    click.echo(q.to_string(index=False))

    q16, q17, qsite, inq = queally_on_grid(transform, shape)
    st_all = counts(st, transform, shape)
    hs_all = counts(hs, transform, shape)
    mask, cand = flightline_mask(st_all, hs_all, q16, inq)
    with rasterio.open(outputdir / 'flightline_mask.tif', 'w', driver='GTiff',
                       height=shape[0], width=shape[1], count=1,
                       dtype='uint8', crs=f'EPSG:{epsg}', transform=transform,
                       compress='deflate') as r:
        r.write(mask.astype('uint8'), 1)
    xs = transform.c + (np.nonzero(mask)[1] + 0.5) * transform.a
    click.echo(f'flight-line mask: {mask.sum()} px ({mask.sum() * 0.09:.0f} '
               f'ha; SOAP {(xs < SITE_X).sum()}, TEAK {(xs >= SITE_X).sum()}'
               f'); {cand.sum() - mask.sum()} scattered empty px dropped')
    qd = mask_by_quad(mask, transform, epsg)
    qd.to_csv(outputdir / 'flightline_quads.csv', index=False)
    click.echo(qd.to_string(index=False))

    stc = st[st.cohort]
    hsc = hs[hs.cohort]
    hs_n, hs_died = lidar_cohort(transform, shape)
    layers = {
        'st_n': counts(stc, transform, shape),
        'st_died': counts(stc, transform, shape, stc.died),
        'st_all': st_all,
        'st_m16': counts(st, transform, shape, st.mort2016),
        'hs_n': hs_n, 'hs_died': hs_died,
        'oblique_frac': mask.astype(float), 'q_frac': inq.astype(float),
        'q_m16': q16, 'q_m17': q17}
    assert np.allclose(counts(hsc, transform, shape), hs_n)
    env = rc.open_env(AOI)
    valid = rc.undisturbed(env, 2019)
    tabs = {}
    for k in SCALES:
        d = cells(layers, valid, k, transform)
        d.to_csv(outputdir / f'stovall_{AOI}_{rc.RES * k}m.csv', index=False)
        tabs[k] = d
        click.echo(f'{rc.RES * k} m: {len(d)} cells, Stovall '
                   f'{d.stov_frac.notna().sum()}, HS {d.mort_frac.notna().sum()}'
                   f', both {(d.stov_frac.notna() & d.mort_frac.notna()).sum()}'
                   f', both outside mask '
                   f'{(d.stov_frac.notna() & d.mort_frac.notna() & (d.oblique_frac == 0) & (d.q_frac == 1)).sum()}')
    # Gridding check against the Queally grid at 30 m, all pixels with trees
    ok = (st_all > 0) & np.isfinite(q16)
    m16 = layers['st_m16'][ok] / st_all[ok]
    rep = pd.DataFrame([dict(
        grid='Stovall mean mort2016, 30 m', n=int(ok.sum()),
        rho=spearmanr(m16, q16[ok])[0], r2_11=rc.wr2(q16[ok], m16),
        frac_within_0p01=float(np.mean(np.abs(m16 - q16[ok]) < 0.01)))])
    rep.to_csv(outputdir / 'queally_repro.csv', index=False)
    click.echo(rep.to_string(index=False))

    ag = agreement(tabs, n_boot)
    ag.to_csv(outputdir / 'agreement.csv', index=False)
    plot(tabs[3], shape, 3, outputdir / 'agreement_90m.png')

    if skip_matching:
        return
    m = match_trees(st, hs)
    row, col = pixel_index(m, transform, shape)
    m['oblique'] = np.where(row >= 0, mask[row.clip(0), col.clip(0)], False)
    m.to_parquet(outputdir / 'tree_matches.parquet')
    click.echo(f'matched {m.contain.mean():.3f} of Stovall trees (nearest '
               f'2017 treetop crown), {m.any_contain.mean():.3f} inside any '
               f'2013 crown; HS trees matched more than once: '
               f'{(m.hs_treeID[m.contain].value_counts() > 1).sum()}')
    cf = confusion(m)
    cf.to_csv(outputdir / 'confusion.csv', index=False)
    click.echo(cf[cf.hclass == 'all'].to_string(index=False))
    row, col = pixel_index(hs, transform, shape)
    hs['in_stov'] = np.where(row >= 0, st_all[row.clip(0), col.clip(0)] > 0,
                             False)
    mb = match_bias(m, hs)
    mb.to_csv(outputdir / 'match_bias.csv', index=False)
    click.echo(mb.to_string(index=False))
    hr = hs_region(m)
    hr.to_csv(outputdir / 'hs_region.csv', index=False)
    click.echo(hr.to_string(index=False))


if __name__ == '__main__':
    main()
