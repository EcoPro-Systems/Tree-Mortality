#!/usr/bin/env python
"""
Evaluate the Cheng et al. cross-resolution dead-tree model (run on NAIP by
proto_cheng_naip_inference.py) against the Hemming-Schroeder et al. (2023)
hand labels at NEON SOAP/TEAK: 9,025 lidar-delineated crowns photo-
interpreted as live/dead in 2017 from NEON hyperspectral-derived RGB
(data/training/trees_2017_training_filtered_labeled.shp), grouped in
sampled 30 m Landsat pixels (training_sample_sites.shp).

NAIP brackets the labels: 2016 (Jun 30 - Jul 26) and 2018 (Aug 31 - Sep 8).
Trees dead in 2017 should appear dead in 2018 (recall), and trees live in
2017 should not appear dead in 2016 (false positives); live-2017 trees can
legitimately die by 2018.

Crown level: fraction of each crown's pixels with model energy > 0.
Pixel level: labeled dead fraction per sampled 30 m pixel vs the model's
dead-canopy fraction within that pixel.

--shift chip|quad translates the crowns and sample pixels by the NAIP-minus-
lidar offsets measured by proto_hs_naip_alignment.py (offsets_<year>.csv in
--offsets-dir) before scoring: per chip (quad median where the chip's
correlation peak is weak), or the median of the chip's NAIP quarter-quad.
"""
import os
import glob
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

THRESHOLDS = [0.05, 0.1, 0.25, 0.5]


def overlap_fraction(geoms, energy_files, shifts=None):
    """Fraction of each geometry's pixels with energy > 0 (NaN if no
    raster fully contains it). shifts optionally maps a chip name (energy
    file name without _energy.tif) to a (dx, dy) translation in m applied
    to the geometries before scoring them in that chip"""
    out = np.full(len(geoms), np.nan)
    for f in energy_files:
        with rasterio.open(f) as ds:
            g = geoms.to_crs(ds.crs)
            if shifts:
                dx, dy = shifts.get(Path(f).name[:-len('_energy.tif')],
                                    (0.0, 0.0))
                g = g.translate(dx, dy)
            b = ds.bounds
            inside = ((g.bounds.minx > b.left) & (g.bounds.maxx < b.right)
                      & (g.bounds.miny > b.bottom) & (g.bounds.maxy < b.top))
            for i in np.flatnonzero(inside.values & np.isnan(out)):
                geom = g.iloc[i]
                win = from_bounds(*geom.bounds, ds.transform)
                win = win.round_offsets().round_lengths()
                if win.width < 1 or win.height < 1:
                    continue
                e = ds.read(1, window=win)
                m = geometry_mask([geom], out_shape=e.shape, invert=True,
                                  transform=ds.window_transform(win),
                                  all_touched=False)
                if m.sum():
                    out[i] = (e[m] > 0).mean()
    return out


@click.command()
@click.argument('energydir', type=click.Path(path_type=Path, exists=True))
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--hs-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/hemming_schroeder2023/data/training'))
@click.option('--shift', default='none', show_default=True,
              type=click.Choice(['none', 'chip', 'quad']),
              help='Correct crowns for the measured NAIP-minus-lidar offset')
@click.option('--offsets-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/hls_results/hs_naip_alignment'))
def main(energydir, outputdir, hs_dir, shift, offsets_dir):

    os.makedirs(outputdir, exist_ok=True)
    trees = gpd.read_file(hs_dir / 'trees_2017_training_filtered_labeled.shp')
    trees = trees[trees.live.isin([0, 1])].reset_index(drop=True)
    sites = gpd.read_file(hs_dir / 'training_sample_sites.shp')
    sites = sites[sites.sample_id.isin(trees.sampleid)].reset_index(drop=True)
    trees['high_certainty'] = trees.certainty == 1.0

    rows, prow = [], []
    for year in (2016, 2018):
        files = sorted(glob.glob(str(energydir / str(year) / '*_energy.tif')))
        shifts = None
        if shift != 'none':
            # Imported here: proto_hs_naip_alignment imports this module
            from proto_hs_naip_alignment import chip_shifts
            shifts = chip_shifts(offsets_dir / f'offsets_{year}.csv', shift)
        trees[f'frac_{year}'] = overlap_fraction(trees.geometry, files, shifts)
        sites[f'model_dead_{year}'] = overlap_fraction(sites.geometry, files,
                                                       shifts)
        for subset, t in [('all', trees), ('high certainty', trees[trees.high_certainty])]:
            t = t[np.isfinite(t[f'frac_{year}'])]
            dead = t.live == 0
            r = dict(year=year, subset=subset, n=len(t), n_dead=int(dead.sum()),
                     auc=roc_auc_score(dead, t[f'frac_{year}']))
            for th in THRESHOLDS:
                det = t[f'frac_{year}'] >= th
                tp = (det & dead).sum()
                r[f'recall@{th}'] = tp / dead.sum()
                r[f'precision@{th}'] = tp / max(det.sum(), 1)
                r[f'live_flagged@{th}'] = (det & ~dead).sum() / (~dead).sum()
            rows.append(r)

    lab = trees.groupby('sampleid').agg(
        n_trees=('live', 'size'), frac_dead=('live', lambda v: (v == 0).mean()))
    area = trees.assign(a=trees.area, d=(trees.live == 0) * trees.area)
    lab['area_frac_dead'] = area.groupby('sampleid').d.sum() / \
        area.groupby('sampleid').a.sum()
    s = sites.set_index('sample_id').join(lab, how='inner')
    for year in (2016, 2018):
        ok = np.isfinite(s[f'model_dead_{year}'])
        for ref in ('frac_dead', 'area_frac_dead'):
            prow.append(dict(year=year, reference=ref, n=int(ok.sum()),
                             pearson=pearsonr(s[ref][ok], s[f'model_dead_{year}'][ok])[0],
                             spearman=spearmanr(s[ref][ok], s[f'model_dead_{year}'][ok])[0]))

    crown = pd.DataFrame(rows)
    pix = pd.DataFrame(prow)
    crown.to_csv(outputdir / 'crown_metrics.csv', index=False)
    pix.to_csv(outputdir / 'pixel_metrics.csv', index=False)
    trees.drop(columns='geometry').to_csv(outputdir / 'crowns_scored.csv',
                                           index=False)
    s.drop(columns='geometry').to_csv(outputdir / 'sites_scored.csv')
    pd.set_option('display.width', 250)
    print(crown.round(3).to_string(index=False))
    print(pix.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
