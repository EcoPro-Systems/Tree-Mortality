#!/usr/bin/env python
"""
How much of the lidar structure can imaging spectroscopy and wall-to-wall
products recover, and how much trait-model skill is kept when a model has
no lidar at all?

1. Structure from spectra (out-of-fold gradient boosting, 1 km spatial
   blocks) on the cells each lidar source covers (>= 70% of a cell):
     aso       aso_rh98, aso_cover5, aso_chm_mean (ASO 2014-17 composite)
     lvis2008  lvis_rh100_mean, lvis_rh50_mean (LVIS Sep 2008)
     neon2013  lidar_h_p90, lidar_n_trees (NEON 2013 trees)
   from three input sets:
     spectral  2013 AVIRIS-C traits (means, SD bands, green-fraction QC)
               and 2013/2014 EWT
     wall      GLAD height 2010, TCC 2010, LANDFIRE 2014 EVH/EVC and the
               Landsat baseline NDVI/NIRv
     both      spectral + wall
   Reported: R² and Spearman ρ of the out-of-fold prediction against the
   lidar metric, next to GLAD height alone (ρ).
2. Models without lidar, on the same cells: the structure metrics of the
   source are replaced by their out-of-fold predictions from "both" (Ŝ),
   and the ladder
       Env+S | Env+S+T | Env+S+Ŝ | Env+S+Ŝ+T | Env+S+L | Env+S+L+T
   gives the skill kept without lidar, R²(Env+S+Ŝ+T) / R²(Env+S+L+T).
   Ŝ comes from the same spectra as T, so this measures what a model
   without lidar can do, not whether traits carry information beyond
   structure (that is the lidar check in proto_response_traits.py).

Outputs in OUTPUTDIR: structure_skill.csv, no_lidar_ladder.csv.

    python proto_structure_from_spectra.py \
        $E/hls_results/structure_from_spectra -a neon_soap_teak \
        -a sierra_nf
"""
import click
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import response_common as rc
from proto_response_metrics import ENV, S_WALL, B1
from proto_response_traits import (STRUCTURES, build, structure_cells,
                                   feature_sets)

METRICS = {'aso': ['aso_rh98', 'aso_cover5', 'aso_chm_mean'],
           'lvis2008': ['lvis_rh100_mean', 'lvis_rh50_mean'],
           'neon2013': ['lidar_h_p90', 'lidar_n_trees']}
TARGETS = ['ndmi_recovery', 'nirv_recovery', 'ndmi_sens', 'ndmi_resistance',
           'ndmi_resilience']


def structure_skill(d, source, T, W, aoi, scale_m):
    rows = []
    sets = {'spectral': T + W, 'wall': S_WALL + B1,
            'both': T + W + S_WALL + B1}
    for m in METRICS[source]:
        sub = d[d[m].notna()]
        glad = sub.glad_h2010.notna()
        base = dict(aoi=aoi, scale_m=scale_m, source=source, metric=m,
                    n=len(sub))
        rows.append(dict(base, inputs='glad_h2010', r2=np.nan,
                         rho=spearmanr(sub.glad_h2010[glad], sub[m][glad])[0]))
        for name, cols in sets.items():
            p = rc.oof_predict(sub, cols, m, 'block1000')
            rows.append(dict(base, inputs=name, r2=rc.wr2(sub[m], p),
                             rho=spearmanr(sub[m], p)[0]))
        click.echo(f'  {m:16s} n={len(sub):6d}  ' + '  '.join(
            f'{x["inputs"]} ρ {x["rho"]:.2f}' +
            (f' R² {x["r2"]:.2f}' if np.isfinite(x['r2']) else '')
            for x in rows[-4:]))
    return rows


def no_lidar(d, source, T, W, n_boot, aoi, scale_m):
    L = STRUCTURES[source]
    both = T + W + S_WALL + B1
    Shat = {}
    for c in L:
        ok = d[c].notna()
        p = pd.Series(np.nan, index=d.index)
        p[ok] = rc.oof_predict(d[ok], both, c, 'block1000')
        Shat[f'hat_{c}'] = p
    d = pd.concat([d, pd.DataFrame(Shat)], axis=1)
    H = list(Shat)
    es = ENV + S_WALL
    fs = {'Env+S': es, 'Env+S+T': es + T, 'Env+S+Shat': es + H,
          'Env+S+Shat+T': es + H + T, 'Env+S+L': es + L,
          'Env+S+L+T': es + L + T}
    pairs = [('Env+S', 'Env+S+T'), ('Env+S', 'Env+S+Shat'),
             ('Env+S', 'Env+S+Shat+T'), ('Env+S+Shat', 'Env+S+Shat+T'),
             ('Env+S+L+T', 'Env+S+Shat+T'), ('Env+S', 'Env+S+L+T')]
    rows = []
    for t in TARGETS:
        if t not in d or d[t].notna().sum() < 500:
            continue
        sub = d[d[t].notna() & np.isfinite(d[t])]
        w = 'mort_n' if t == 'mort_frac' else None
        r, _ = rc.ladder(sub, t, fs, 'block1000', weight=w, n_boot=n_boot,
                         pairs=pairs)
        for x in r:
            x.update(aoi=aoi, scale_m=scale_m, source=source)
        rows += r
        a = {x['features']: x['r2'] for x in r
             if not x['compare'] and x['boot_blocks'] == 'block1000'}
        click.echo(f'  {t:16s} n={len(sub):6d}  ' + '  '.join(
            f'{k.replace("Env+S", "E")} {v:.3f}' for k, v in a.items()) +
            f'  kept {a["Env+S+Shat+T"] / a["Env+S+L+T"]:.2f}')
    return rows


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--structure', 'sources', multiple=True,
              default=list(STRUCTURES), show_default=True,
              type=click.Choice(list(STRUCTURES)))
@click.option('--response-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/response')
@click.option('--dynamics-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/trait_dynamics')
@click.option('--scale', default=3, show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
def main(outputdir, aois, sources, response_dir, dynamics_dir, scale,
         n_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    scale_m = rc.RES * scale
    skill_rows, ladder_rows = [], []
    for aoi in aois:
        rdir = response_dir
        if not (rdir / f'metrics_{aoi}_{scale_m}m.csv').exists():
            rdir = rc.E / 'hls_results' / f'response_{aoi.split("_")[0]}'
        d = build(aoi, scale, rdir, dynamics_dir)
        _, T, W = feature_sets(d)
        for src in sources:
            sub = structure_cells(d, aoi, scale, src)
            if len(sub) < 500:
                continue
            click.echo(f'[{aoi} {scale_m} m] {src}: {len(sub)} cells')
            skill_rows += structure_skill(sub, src, T, W, aoi, scale_m)
            ladder_rows += no_lidar(sub, src, T, W, n_boot, aoi, scale_m)
            pd.DataFrame(skill_rows).to_csv(
                outputdir / 'structure_skill.csv', index=False)
            pd.DataFrame(ladder_rows).to_csv(
                outputdir / 'no_lidar_ladder.csv', index=False)


if __name__ == '__main__':
    main()
