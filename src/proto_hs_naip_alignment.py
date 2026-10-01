#!/usr/bin/env python
"""
How well do the Hemming-Schroeder et al. (2023) lidar crowns at NEON
SOAP/TEAK line up with NAIP, and does the offset matter for the crown-level
scores of proto_cheng_vs_hs_labels.py?

Uses the NAIP chips the Cheng model was run on for the HS validation
(naip_chips/neon_soap_teak/<year>) and the model energy rasters from
proto_cheng_naip_inference.py (hls_results/cheng_naip_neon/<year>).

Steps (--step, repeatable; default all, in this order):
  register  Per chip, normalized cross-correlation of NAIP NDVI with a
            canopy mask of the HS lidar treetops (trees >= 5 m, disks of the
            2013 crown area) over +-15 m. The peak is the rigid NAIP-minus-
            lidar offset (positive dx/dy: NAIP features lie east/north of
            the lidar crowns). A rigid shift is a first-order model: relief
            displacement of tall trees also varies within a chip.
  height    On chips with a reliable peak (PSR >= 6), the same offset from
            short (5-15 m), medium (15-30 m) and tall (> 30 m) trees
            separately. Relief displacement moves crown tops radially away
            from the photo nadir by about h * tan(view angle), so taller
            trees should show larger offsets.
  stacked   Model dead mask (energy > 0) cross-correlated with rasterized
            HS 2017 dead and, separately, live crowns, summed over all chips
            and divided by the value expected with no association (model
            density x crown area). The peak of this enrichment surface is
            the systematic model-minus-HS offset and its width the crown-
            level positional scatter.
  rescore   Crown AUC / detection and 30 m sample-pixel agreement as in
            proto_cheng_vs_hs_labels.py, with crowns shifted by: none
            (reproduces crown_metrics.csv), the chip offset (quad median
            where PSR < 6), or the median offset of the chip's NAIP
            quarter-quad.

Outputs (outputdir):
  offsets_<year>.csv            per chip: dx, dy (m), peak r, r at zero
                                shift (r0), peak-to-sidelobe ratio (psr)
  offsets_by_height_<year>.csv  per chip and height class: dx, dy, psr, n
  stacked_<year>.npz            dead / live enrichment surfaces, res (m)
  rescore_crowns.csv, rescore_pixels.csv, crowns_shifted.csv

The HS data are in EPSG:32611 and the chips in EPSG:26911 (NAD83). With
PROJ_NETWORK=ON pyproj applies a grid-based datum shift of about 0.8 m
here; with it off (the default) the two are treated as identical. Set
PROJ_NETWORK=OFF to reproduce the numbers in docs/hls_mortality_prototype.md.

Runtime on a laptop: register ~6 min, height ~3 min, stacked ~5 min,
rescore ~40 min. --max-chips limits each step to the first N chips per year
for a quick test.

Usage:
  python src/proto_hs_naip_alignment.py \\
      /Volumes/Earth04/ecopro/hls_results/hs_naip_alignment \\
      --step register --step height --step stacked
"""
import os
import glob
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.features import rasterize
from scipy.signal import fftconvolve
from scipy.stats import spearmanr, pearsonr
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from proto_cheng_vs_hs_labels import overlap_fraction

STEPS = ['register', 'height', 'stacked', 'rescore']
MAXSHIFT_M = 15.0
# Chips with a weaker correlation peak are unreliable (fall back to the
# quad median in the re-score)
MIN_PSR = 6.0
HEIGHT_BANDS = [('short', 5, 15), ('medium', 15, 30), ('tall', 30, 200)]
SHIFT_MODES = ['none', 'chip', 'quad']


def load_treetops(hs_dir, crs):
    """HS lidar treetops in `crs`: x, y, crown radius from the 2013 crown
    area, and 2013 height"""
    t = gpd.read_file(hs_dir / 'deliverables' / 'vector'
                      / 'tree_locations_las_intersection.shp').to_crs(crs)
    return pd.DataFrame({'x': t.geometry.x.values, 'y': t.geometry.y.values,
                         'r': np.sqrt(t.ca2013.values / np.pi),
                         'h': t.zmax2013.values})


def load_crowns(hs_dir):
    """HS 2017 labelled crowns (live = 1, dead = 0)"""
    t = gpd.read_file(hs_dir / 'training'
                      / 'trees_2017_training_filtered_labeled.shp')
    return t[t.live.isin([0, 1])].reset_index(drop=True)


def canopy_mask(t, transform, shape):
    geoms = [(Point(x, y).buffer(r, 8), 1) for x, y, r in zip(t.x, t.y, t.r)]
    return rasterize(geoms, out_shape=shape, transform=transform,
                     fill=0, dtype='uint8').astype(np.float32)


def xcorr_offset(a, b, maxpix):
    """Shift (drow, dcol) of b that best matches a, by normalized cross-
    correlation over +-maxpix; also returns peak r, r at zero shift and the
    peak-to-sidelobe ratio"""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = fftconvolve(a, b[::-1, ::-1], mode='same') / a.size
    cy, cx = np.array(c.shape) // 2
    w = c[cy - maxpix:cy + maxpix + 1, cx - maxpix:cx + maxpix + 1]
    i, j = np.unravel_index(np.argmax(w), w.shape)
    peak = w[i, j]
    side = w.copy()
    side[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4] = np.nan
    psr = (peak - np.nanmean(side)) / (np.nanstd(side) + 1e-9)
    return i - maxpix, j - maxpix, peak, w[maxpix, maxpix], psr


def read_chip(path):
    """NAIP NDVI, transform, shape, pixel size, bounds and nodata fraction"""
    with rasterio.open(path) as ds:
        img = ds.read().astype(np.float32)
        T, shape, res, b = ds.transform, ds.shape, ds.res[0], ds.bounds
    ndvi = (img[3] - img[0]) / (img[3] + img[0] + 1e-6)
    return ndvi, T, shape, res, b, (img[:3] == 0).all(0).mean()


def near(t, b, pad=10):
    return ((t.x > b.left - pad) & (t.x < b.right + pad)
            & (t.y > b.bottom - pad) & (t.y < b.top + pad))


def chip_shifts(offsets_csv, mode, min_psr=MIN_PSR):
    """{chip: (dx, dy)} crown translation in m for mode 'chip' (the chip's
    offset if its peak is reliable, else its quad median) or 'quad' (quad
    median). Chips without a reliable quad median get no shift"""
    d = pd.read_csv(offsets_csv)
    q = d[d.psr >= min_psr].groupby('quad')[['dx', 'dy']].median()
    d = d.join(q, on='quad', rsuffix='_quad')
    for c in ('dx', 'dy'):
        if mode == 'chip':
            d[f'{c}_shift'] = np.where(d.psr >= min_psr, d[c], d[f'{c}_quad'])
        else:
            d[f'{c}_shift'] = d[f'{c}_quad']
        d[f'{c}_shift'] = d[f'{c}_shift'].fillna(0)
    return dict(zip(d.chip, zip(d.dx_shift, d.dy_shift)))


def register(files, trees):
    rows = []
    t = trees[trees.h >= 5]
    for f in tqdm(files, leave=False):
        ndvi, T, shape, res, b, nodata = read_chip(f)
        k = near(t, b)
        if k.sum() < 200:
            continue
        m = canopy_mask(t[k], T, shape)
        if m.mean() < 0.05 or nodata > 0.01:
            continue
        dr, dc, peak, r0, psr = xcorr_offset(ndvi, m,
                                             int(round(MAXSHIFT_M / res)))
        rows.append(dict(chip=Path(f).stem, quad=Path(f).stem.split('__')[0],
                         dx=dc * res, dy=-dr * res, peak=peak, r0=r0,
                         psr=psr, n_trees=int(k.sum()), cover=m.mean()))
    d = pd.DataFrame(rows)
    shift = np.hypot(d.dx, d.dy)
    good = d.psr >= MIN_PSR
    click.echo(f'  {len(d)} chips; |offset| median {shift.median():.1f} m, '
               f'p90 {shift.quantile(.9):.1f} m; r at zero '
               f'{d.r0.median():.3f} -> at peak {d.peak.median():.3f}')
    click.echo(f'  PSR >= {MIN_PSR:g}: {good.sum()} chips; |offset| median '
               f'{shift[good].median():.1f} m, p90 {shift[good].quantile(.9):.1f} m; '
               f'median (dx, dy) ({d.dx[good].median():+.1f}, '
               f'{d.dy[good].median():+.1f}) m')
    return d


def by_height(chips_dir, off, trees):
    rows = []
    for _, o in tqdm(off.iterrows(), total=len(off), leave=False):
        ndvi, T, shape, res, b, _ = read_chip(chips_dir / f'{o.chip}.tif')
        tt = trees[near(trees, b)]
        r = dict(chip=o.chip, quad=o.quad, dx_all=o.dx, dy_all=o.dy)
        for name, lo, hi in HEIGHT_BANDS:
            s = tt[(tt.h >= lo) & (tt.h < hi)]
            if len(s) < 30:
                continue
            dr, dc, peak, r0, psr = xcorr_offset(
                ndvi, canopy_mask(s, T, shape), int(round(MAXSHIFT_M / res)))
            r.update({f'dx_{name}': dc * res, f'dy_{name}': -dr * res,
                      f'psr_{name}': psr, f'n_{name}': len(s)})
        rows.append(r)
    df = pd.DataFrame(rows)
    for name, _, _ in HEIGHT_BANDS:
        if f'psr_{name}' not in df:
            continue
        # Single height classes have fewer trees, so accept a weaker peak
        ok = df[f'psr_{name}'] >= 5
        sh = np.hypot(df.loc[ok, f'dx_{name}'], df.loc[ok, f'dy_{name}'])
        # Displacement of this height class relative to the whole canopy
        rel = np.hypot(df.loc[ok, f'dx_{name}'] - df.loc[ok, 'dx_all'],
                       df.loc[ok, f'dy_{name}'] - df.loc[ok, 'dy_all'])
        click.echo(f'  {name:6s} n={ok.sum():4d}  |offset| median '
                   f'{sh.median():.1f} m, p90 {sh.quantile(.9):.1f} m; '
                   f'relative to all-canopy offset: median {rel.median():.1f} m, '
                   f'p90 {rel.quantile(.9):.1f} m')
    return df


def stacked(files, crowns):
    acc, exp = {}, {'dead': 0.0, 'live': 0.0}
    by_crs, used, res = {}, set(), None
    for f in tqdm(files, leave=False):
        with rasterio.open(f) as ds:
            b, T, shape, crs = ds.bounds, ds.transform, ds.shape, ds.crs
            if res is None:
                res = ds.res[0]
                maxpix = int(round(MAXSHIFT_M / res))
                acc = {k: np.zeros((2 * maxpix + 1,) * 2) for k in exp}
            if crs not in by_crs:
                by_crs[crs] = crowns.to_crs(crs)
            # Each crown counts once, in the first chip that holds it well
            # inside the edges
            t = by_crs[crs].cx[b.left + 20:b.right - 20,
                               b.bottom + 20:b.top - 20]
            t = t[~t.index.isin(used)]
            if not len(t):
                continue
            used |= set(t.index)
            m = (ds.read(1) > 0).astype(np.float32)
        for k, lv in (('dead', 0), ('live', 1)):
            s = t[t.live == lv]
            if not len(s):
                continue
            d = rasterize([(g, 1) for g in s.geometry], out_shape=shape,
                          transform=T, fill=0, dtype='uint8').astype(np.float32)
            c = fftconvolve(m, d[::-1, ::-1], mode='same')
            cy, cx = np.array(c.shape) // 2
            acc[k] += c[cy - maxpix:cy + maxpix + 1, cx - maxpix:cx + maxpix + 1]
            exp[k] += m.mean() * d.sum()
    enr = {k: acc[k] / exp[k] for k in acc}
    click.echo(f'  {len(used)} labelled crowns')
    for k, e in enr.items():
        i, j = np.unravel_index(np.argmax(e), e.shape)
        dx, dy = (j - maxpix) * res, -(i - maxpix) * res
        # Radius containing half of the excess enrichment above the far-
        # field level (median beyond 12 m from the peak)
        yy, xx = np.mgrid[-maxpix:maxpix + 1, -maxpix:maxpix + 1] * res
        r = np.hypot(xx - dx, yy - dy)
        base = np.median(e[r > 12])
        ex = np.clip(e - base, 0, None)
        order = np.argsort(r.ravel())
        cum = np.cumsum(ex.ravel()[order]) / ex.sum()
        r50 = r.ravel()[order][np.searchsorted(cum, 0.5)]
        click.echo(f'  {k}: enrichment at 0 = {e[maxpix, maxpix]:.2f}, '
                   f'peak {e[i, j]:.2f} at (dx {dx:+.1f}, dy {dy:+.1f}) m, '
                   f'far-field {base:.2f}, half-excess radius {r50:.1f} m')
    return dict(enr, res=res)


def rescore(files, crowns, sites, lab, offsets_csv, year):
    rows, prow = [], []
    dead = (crowns.live == 0).values
    for mode in SHIFT_MODES:
        sh = None if mode == 'none' else chip_shifts(offsets_csv, mode)
        frac = overlap_fraction(crowns.geometry, files, sh)
        crowns[f'frac_{year}_{mode}'] = frac
        ok = np.isfinite(frac)
        for subset, s in (('all', ok), ('high certainty',
                                        ok & (crowns.certainty == 1.0).values)):
            rows.append(dict(year=year, shift=mode, subset=subset,
                             n=int(s.sum()), n_dead=int((dead & s).sum()),
                             auc=roc_auc_score(dead[s], frac[s]),
                             dead_detected=(frac[s & dead] > 0).mean(),
                             live_flagged=(frac[s & ~dead] > 0).mean(),
                             dead_hit_05=(frac[s & dead] >= 0.05).mean(),
                             live_hit_05=(frac[s & ~dead] >= 0.05).mean()))
        sf = overlap_fraction(sites.geometry, files, sh)
        s = sites.assign(model=sf).set_index('sample_id').join(lab)
        ok = np.isfinite(s.model)
        prow.append(dict(year=year, shift=mode, n=int(ok.sum()),
                         spearman=spearmanr(s.frac_dead[ok], s.model[ok])[0],
                         pearson=pearsonr(s.frac_dead[ok], s.model[ok])[0]))
    return rows, prow


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--step', 'steps', multiple=True, default=STEPS,
              show_default=True, type=click.Choice(STEPS))
@click.option('--year', 'years', multiple=True, type=int,
              default=[2016, 2018], show_default=True)
@click.option('--chips-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/naip_chips/neon_soap_teak'))
@click.option('--energy-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/hls_results/cheng_naip_neon'))
@click.option('--hs-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/hemming_schroeder2023/data'))
@click.option('--max-chips', default=0, show_default=True,
              help='If > 0, only use the first N chips per year (testing)')
def main(outputdir, steps, years, chips_dir, energy_dir, hs_dir, max_chips):

    os.makedirs(outputdir, exist_ok=True)
    pd.set_option('display.width', 250)

    def listing(d, pattern):
        f = sorted(glob.glob(str(d / pattern)))
        return f[:max_chips] if max_chips else f

    trees = None
    if 'register' in steps or 'height' in steps:
        with rasterio.open(listing(chips_dir / str(years[0]), '*.tif')[0]) as ds:
            trees = load_treetops(hs_dir, ds.crs)
        click.echo(f'{len(trees)} HS lidar treetops')

    if 'register' in steps:
        for year in years:
            click.echo(f'register {year}:')
            d = register(listing(chips_dir / str(year), '*.tif'), trees)
            d.to_csv(outputdir / f'offsets_{year}.csv', index=False)

    if 'height' in steps:
        for year in years:
            click.echo(f'height {year}:')
            off = pd.read_csv(outputdir / f'offsets_{year}.csv')
            df = by_height(chips_dir / str(year), off[off.psr >= MIN_PSR],
                           trees)
            df.to_csv(outputdir / f'offsets_by_height_{year}.csv', index=False)

    if 'stacked' in steps or 'rescore' in steps:
        crowns = load_crowns(hs_dir)

    if 'stacked' in steps:
        for year in years:
            click.echo(f'stacked {year}:')
            s = stacked(listing(energy_dir / str(year), '*_energy.tif'), crowns)
            np.savez(outputdir / f'stacked_{year}.npz', **s)

    if 'rescore' in steps:
        sites = gpd.read_file(hs_dir / 'training' / 'training_sample_sites.shp')
        sites = sites[sites.sample_id.isin(crowns.sampleid)].reset_index(drop=True)
        lab = crowns.groupby('sampleid').agg(
            frac_dead=('live', lambda v: (v == 0).mean()))
        rows, prow = [], []
        for year in years:
            click.echo(f'rescore {year}')
            r, p = rescore(listing(energy_dir / str(year), '*_energy.tif'),
                           crowns, sites, lab,
                           outputdir / f'offsets_{year}.csv', year)
            rows += r
            prow += p
        crown, pix = pd.DataFrame(rows), pd.DataFrame(prow)
        crown.to_csv(outputdir / 'rescore_crowns.csv', index=False)
        pix.to_csv(outputdir / 'rescore_pixels.csv', index=False)
        crowns.drop(columns='geometry').to_csv(
            outputdir / 'crowns_shifted.csv', index=False)
        click.echo(crown.round(3).to_string(index=False))
        click.echo(pix.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
