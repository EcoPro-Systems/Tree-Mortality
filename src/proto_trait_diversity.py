#!/usr/bin/env python
"""
Does within-cell trait diversity explain drought response beyond mean
traits?

Blocks (2013 AVIRIS-C, on the cells of proto_response_traits.py):
  Tmean  the 14 trait means and the green-fraction QC
  D      diversity: the per-pixel PLSR SD bands of LMA, N and chlorophyll
         (as in T), the spatial SD of N and LMA over the 30 m pixels of a
         cell, and functional dispersion (FDis: mean Euclidean distance of
         the pixels to the cell centroid in AOI-standardized trait space)
         on N + LMA and, where canopy water exists, N + LMA + EWT

Ladder: Env+S | Env+S+Tmean | Env+S+D | Env+S+Tmean+D; with --structure,
the same on top of the lidar structure of that source (on its cells).
Within-stratum ρ of each D feature with each target is also reported.

Outputs in OUTPUTDIR: diversity_ladder.csv, diversity_rho.csv.

    python proto_trait_diversity.py $E/hls_results/trait_diversity \
        -a neon_soap_teak --structure lvis2008 --structure aso
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import response_common as rc
from proto_hls_vs_hs import block_mean
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import (TRAITS, SD_TRAITS, ELEV_BAND, STRUCTURES,
                                   build, structure_cells, trait_paths)
from proto_trait_dynamics import crosstrack_normalize

TARGETS = ['ndmi_resilience', 'ndmi_recovery', 'nirv_resilience',
           'nirv_recovery', 'ndmi_resistance']


def fdis(zs, valid, k):
    """Functional dispersion per k x k cell over pixels valid in every
    standardized layer of zs"""
    ok = valid & np.all([np.isfinite(z) for z in zs], axis=0)
    cent = [np.repeat(np.repeat(block_mean(z, ok, k)[0], k, 0), k, 1)
            for z in zs]
    h, w = ok.shape
    dist = np.sqrt(sum((z[:h // k * k, :w // k * k] -
                        c[:h // k * k, :w // k * k]) ** 2
                       for z, c in zip(zs, cent)))
    full = np.full(ok.shape, np.nan, np.float32)
    full[:h // k * k, :w // k * k] = dist
    return full, ok


def diversity_layers(aoi, k):
    transform, shape, _ = rc.aoi_info(aoi)
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, 2019)
    ftr, fcwc = trait_paths(aoi)
    tr = xr.open_dataset(ftr)
    elev = env.elevation.values
    fid = tr.flight_id.sel(year=2013).values
    z = {}
    for t in ('Nitrogen', 'LMA'):
        a = tr[f'{t}_mean'].sel(year=2013).values.astype(np.float32)
        a[fid <= 0] = np.nan
        a = crosstrack_normalize(a, fid, elev, transform)
        m = valid & np.isfinite(a)
        z[t] = (a - a[m].mean()) / a[m].std()
    if Path(fcwc).exists():
        cwc = xr.open_dataset(fcwc)
        ewt = cwc.ewt980.sel(year=2013).values.astype(np.float32)
        ewt = crosstrack_normalize(ewt, cwc.source_line.sel(year=2013).values,
                                   elev, transform)
        m = valid & np.isfinite(ewt)
        z['EWT'] = (ewt - ewt[m].mean()) / ewt[m].std()
    layers = {}
    layers['D_fdis_NL'], _ = fdis([z['Nitrogen'], z['LMA']], valid, k)
    if 'EWT' in z:
        layers['D_fdis_NLE'], _ = fdis([z['Nitrogen'], z['LMA'], z['EWT']],
                                       valid, k)
    for t in ('Nitrogen', 'LMA'):
        a = z[t]
        mu = np.repeat(np.repeat(block_mean(a, valid, k)[0], k, 0), k, 1)
        h, w = a.shape
        dev = np.full(a.shape, np.nan, np.float32)
        dev[:h // k * k, :w // k * k] = (a[:h // k * k, :w // k * k] -
                                         mu[:h // k * k, :w // k * k]) ** 2
        layers[f'D_pixvar_{t}'] = dev
    t = rc.cell_table(layers, valid, k)
    for c in ('Nitrogen', 'LMA'):
        t[f'D_pixsd_{c}'] = np.sqrt(t.pop(f'D_pixvar_{c}'))
    return t[['cell_row', 'cell_col'] +
             [c for c in t.columns if c.startswith('D_')]]


def run(d, base, targets, n_boot, tags):
    Tmean = [f'T_{t}_2013' for t in TRAITS] + ['T_qcfc_2013']
    D = [f'T_{t}_sd_2013' for t in SD_TRAITS] + \
        [c for c in d.columns if c.startswith('D_')]
    fs = {'base': base, 'base+Tmean': base + Tmean, 'base+D': base + D,
          'base+Tmean+D': base + Tmean + D}
    pairs = [('base', 'base+Tmean'), ('base', 'base+D'),
             ('base+Tmean', 'base+Tmean+D')]
    rows, rho = [], []
    strata = (d.arid.astype(str) + '_' +
              (d.elevation // ELEV_BAND).astype(int).astype(str))
    for t in targets:
        sub = d[d[t].notna() & np.isfinite(d[t])]
        if len(sub) < 500:
            continue
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot, pairs=pairs)
        rows += [dict(x, **tags) for x in r]
        g = {f'{x["features"]}-{x["compare"]}': x for x in r
             if x['compare'] and x['boot_blocks'] == 'block1000'}
        click.echo(f'  {t:18s} n={len(sub):6d}  ' + '  '.join(
            f'{k.split("-")[0].replace("base+", "+")}|'
            f'{k.split("-")[1]} {g[k]["r2"]:+.3f} '
            f'[{g[k]["lo"]:+.3f},{g[k]["hi"]:+.3f}]' for k in g))
        rho += [dict(u, **tags) for u in rc.within_strata_rho(
            sub, D, t, strata[sub.index])]
    return rows, rho


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--structure', 'sources', multiple=True,
              type=click.Choice(list(STRUCTURES)))
@click.option('--dynamics-dir', type=click.Path(path_type=Path),
              default=rc.E / 'hls_results/trait_dynamics')
@click.option('--scale', default=3, show_default=True)
@click.option('--n-boot', default=1000, show_default=True)
def main(outputdir, aois, sources, dynamics_dir, scale, n_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    scale_m = rc.RES * scale
    rows, rho = [], []
    for aoi in aois:
        rdir = rc.E / 'hls_results' / 'response'
        if not (rdir / f'metrics_{aoi}_{scale_m}m.csv').exists():
            rdir = rc.E / 'hls_results' / f'response_{aoi.split("_")[0]}'
        d = build(aoi, scale, rdir, dynamics_dir)
        d = d.merge(diversity_layers(aoi, scale), on=['cell_row', 'cell_col'],
                    how='left')
        runs = [('Env+S', d, ENV + S_WALL)]
        for src in sources:
            sub = structure_cells(d, aoi, scale, src)
            if len(sub) >= 500:
                runs.append((f'Env+S+L:{src}', sub,
                             ENV + S_WALL + STRUCTURES[src]))
        for label, dd, base in runs:
            click.echo(f'[{aoi} {scale_m} m] base {label}: {len(dd)} cells')
            r, u = run(dd, base, TARGETS, n_boot,
                       dict(aoi=aoi, scale_m=scale_m, base_set=label))
            rows += r
            rho += u
            pd.DataFrame(rows).to_csv(outputdir / 'diversity_ladder.csv',
                                      index=False)
            pd.DataFrame(rho).to_csv(outputdir / 'diversity_rho.csv',
                                     index=False)


if __name__ == '__main__':
    main()
