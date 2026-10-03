#!/usr/bin/env python
"""
Why does the cross-drought transfer rank cells in reverse at sierra_nf?

In the forward pilot (proto_forward_pilot.py), a model fitted on cycle-1
responses (2014-16 drought) and applied to cycle 2 (2020-22) ranks cycle-2
cells positively at NEON but in reverse at sierra_nf (Env+S ρ -0.49 to
-0.01), and traits do not rescue it. This script refits the same transfers
on the same cells (proto_forward_pilot.build) and asks where the rank skill
is lost.

1. Covariates per 90 m cell:
     fire_km   distance to the nearest 2020-21 fire perimeter (CAL FIRE
               FRAP; the Creek Fire at sierra_nf). Burned cells are already
               masked; this measures edge, smoke and post-fire effects
     treat_km  distance to the nearest FACTS harvest unit completed 2016-25
               (salvage, sanitation, thinning near the cell)
     ads_c1    USFS Aerial Detection Survey damage in cycle 1: the share of
               the cell's pixels inside a mortality polygon in any of
               2015-17 (hls_labels label == 1; severity units change
               between the legacy and DMSM surveys, so extent is used)
     leg       the cell's own cycle-1 response (legacy)
     dforcing  cycle-2 minus cycle-1 drought CWD anomaly
     elevation, cwd_clim
2. Transfer rank skill, Spearman ρ of cycle-2 response vs the cycle-1
   model's prediction: over all cells, within quartile (or distance) bins
   of each covariate, and after dropping cells near fires, near recent
   treatments or with heavy cycle-1 damage. CIs: 1 km block bootstrap of
   the Pearson correlation of the ranks (ranks fixed, blocks resampled).
3. Drivers: Spearman ρ of each Env+S variable with the response in cycle 1
   and in cycle 2 on the same cells. A variable whose sign flips between
   cycles is a relationship a cycle-1 model cannot carry forward.

Outputs (outputdir): transfer_bins.csv, drivers.csv, cells_<aoi>.csv.gz.

    python proto_transfer_diagnostics.py $E/hls_results/transfer_diagnostics \
        -a neon_soap_teak -a sierra_nf
"""
import click
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import xarray as xr
from pathlib import Path
from scipy.stats import spearmanr

import response_common as rc
from proto_forward_pilot import build, generic
from proto_response_metrics import S_WALL
from proto_response_traits import TRAITS
from proto_response_transfer import ENV

FIRE_YEARS = (2020, 2021)
TREAT_FROM = 2016
ADS_YEARS = (2015, 2016, 2017)
FIRE_BINS = [0, 2, 5, 10, np.inf]
TREAT_BINS = [0, 0.5, 1, 2, np.inf]
MIN_BIN = 300


def cell_xy(d, aoi, k):
    transform, _, epsg = rc.aoi_info(aoi)
    x = transform.c + (d.cell_col.values * k + k / 2) * transform.a
    y = transform.f + (d.cell_row.values * k + k / 2) * transform.e
    return x, y, epsg


def distance_km(x, y, epsg, polys):
    if not len(polys):
        return np.full(len(x), np.inf)
    geom = shapely.union_all(shapely.make_valid(
        polys.to_crs(epsg).geometry.values))
    return shapely.distance(shapely.points(x, y), geom) / 1000


def covariates(d, aoi, k):
    x, y, epsg = cell_xy(d, aoi, k)
    dist = rc.E / 'fire' / 'disturbance'
    fires = gpd.read_file(dist / 'frap_fires.gpkg')
    fires = fires[fires.YEAR_.astype(int).between(*FIRE_YEARS)]
    facts = gpd.read_file(dist / 'facts_harvest.gpkg')
    facts = facts[(facts.fy_completed.astype(float) >= TREAT_FROM) &
                  ~facts.activity_name.str.startswith('Natural Changes')]
    out = pd.DataFrame({'fire_km': distance_km(x, y, epsg, fires),
                        'treat_km': distance_km(x, y, epsg, facts)},
                       index=d.index)
    lab = rc.E / 'hls_labels' / f'{aoi}.nc'
    if lab.exists():
        lbl = xr.open_dataset(lab).label.sel(year=list(ADS_YEARS))
        a = (lbl.values == 1).any(0).astype(np.float32)
        t = rc.cell_table({'ads_c1': a}, np.ones(a.shape, bool), k)
        out = out.join(d[['cell_row', 'cell_col']].merge(
            t[['cell_row', 'cell_col', 'ads_c1']], how='left').set_index(
            d.index)['ads_c1'])
    return out


def rho_ci(y, p, blocks, n_boot, seed=0):
    """Spearman ρ with a block-bootstrap CI (ranks fixed)"""
    ry = pd.Series(y).rank().values
    rp = pd.Series(p).rank().values
    codes, inv = np.unique(blocks, return_inverse=True)
    nb = len(codes)
    s = np.stack([np.bincount(inv, v, nb) for v in (
        np.ones_like(ry), ry, rp, ry * ry, rp * rp, ry * rp)])
    m = np.random.default_rng(seed).integers(0, nb, (n_boot, nb))
    m = np.stack([np.bincount(r, minlength=nb) for r in m]).astype(float)
    n, sy, sp, syy, spp, syp = (m @ s.T).T
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (syp - sy * sp / n) / np.sqrt((syy - sy ** 2 / n) *
                                          (spp - sp ** 2 / n))
    return spearmanr(y, p)[0], *np.nanpercentile(r, [2.5, 97.5])


def transfer_preds(d, c1, t):
    """Cycle-1 models applied to cycle 2, as in proto_forward_pilot"""
    es = ENV + S_WALL
    T13 = [f'T_{x}_13' for x in TRAITS] + ['T_qcfc_13']
    T18 = [f'T_{x}_18' for x in TRAITS] + ['T_qcfc_18']
    a = c1[c1[t].notna()]
    out = {}
    for name, ca, cb in (('Env+S', es, es), ('Env+S+T', es + T13, es + T18)):
        m = rc.hgb().fit(a[ca].set_axis(generic(ca), axis=1), a[t])
        out[name] = m.predict(d[cb].set_axis(generic(cb), axis=1))
    return out


def subsets(d, t):
    """(covariate, group label, mask) for bins and drop filters"""
    out = [('all', 'all', np.ones(len(d), bool))]
    for col, edges in (('fire_km', FIRE_BINS), ('treat_km', TREAT_BINS)):
        b = pd.cut(d[col], edges, right=False)
        for g in b.cat.categories:
            out.append((col, str(g), (b == g).values))
    for col in ('ads_c1', f'leg_{t}', 'dforcing', 'elevation', 'cwd_clim'):
        if col not in d or d[col].nunique() < 4:
            continue
        q = pd.qcut(d[col].rank(method='first'), 4, labels=False)
        for g in range(4):
            out.append((col if col != f'leg_{t}' else 'leg', f'q{g + 1}',
                        (q == g).values))
    out += [('drop', 'fire_km >= 2', (d.fire_km >= 2).values),
            ('drop', 'fire_km >= 5', (d.fire_km >= 5).values),
            ('drop', 'treat_km >= 1', (d.treat_km >= 1).values)]
    if 'ads_c1' in d:
        out.append(('drop', 'ads_c1 == 0', (d.ads_c1 == 0).values))
    leg = d[f'leg_{t}']
    out.append(('drop', 'leg above q25', (leg > leg.quantile(0.25)).values))
    return out


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, required=True)
@click.option('--scale', default=3, show_default=True)
@click.option('-t', '--target', 'targets', multiple=True,
              default=['ndmi_recovery', 'nirv_recovery', 'ndmi_resistance',
                       'ndmi_resilience'], show_default=True)
@click.option('--n-boot', default=500, show_default=True)
def main(outputdir, aois, scale, targets, n_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    bins, drivers = [], []
    for aoi in aois:
        d, c1 = build(aoi, scale, rank=False)
        d = d.reset_index(drop=True)
        d = pd.concat([d, covariates(d, aoi, scale)], axis=1)
        c1i = d[['cell_row', 'cell_col']].merge(c1, how='left')
        d['dforcing'] = d.cwd_anom_dr - c1i.cwd_anom_dr.values
        click.echo(f'[{aoi}] {len(d)} cells; within 2 km of a 2020-21 fire '
                   f'{(d.fire_km < 2).mean():.0%}, within 1 km of a '
                   f'2016+ treatment {(d.treat_km < 1).mean():.0%}')
        for t in targets:
            ok = d[t].notna().values
            p = transfer_preds(d, c1, t)
            for name, pr in p.items():
                d[f'p_{t}_{name}'] = pr
            for cov, g, m in subsets(d, t):
                m = m & ok
                if m.sum() < MIN_BIN:
                    continue
                row = dict(aoi=aoi, target=t, covariate=cov, group=g,
                           n=int(m.sum()))
                for name, pr in p.items():
                    r, lo, hi = rho_ci(d[t].values[m], pr[m],
                                       d.block1000.values[m], n_boot)
                    key = name.replace('Env+S', 'es').replace('+T', 't')
                    row.update({f'rho_{key}': r, f'lo_{key}': lo,
                                f'hi_{key}': hi})
                row['rho_leg'] = spearmanr(d[t].values[m],
                                           d[f'leg_{t}'].values[m])[0]
                bins.append(row)
            a = [x for x in bins if x['aoi'] == aoi and x['target'] == t]
            click.echo(f'  {t:16s} ' + '  '.join(
                f'{x["group"]} {x["rho_es"]:+.2f}/{x["rho_est"]:+.2f}'
                for x in a if x['covariate'] in ('all', 'drop')))
            # Drivers: each Env+S variable vs the response in each cycle
            for v in ENV + S_WALL:
                r1 = spearmanr(c1i[v], c1i[t], nan_policy='omit')[0]
                r2 = spearmanr(d[v], d[t], nan_policy='omit')[0]
                drivers.append(dict(aoi=aoi, target=t, feature=v,
                                    rho_c1=r1, rho_c2=r2,
                                    flip=np.sign(r1) != np.sign(r2)))
        keep = ['cell_row', 'cell_col', 'block1000', 'fire_km', 'treat_km',
                'ads_c1', 'dforcing'] + [c for c in d if c.startswith(
                    ('p_', 'leg_'))] + list(targets)
        d[[c for c in keep if c in d]].to_csv(
            outputdir / f'cells_{aoi}.csv.gz', index=False,
            float_format='%.6g')
        pd.DataFrame(bins).to_csv(outputdir / 'transfer_bins.csv',
                                  index=False)
        pd.DataFrame(drivers).to_csv(outputdir / 'drivers.csv', index=False)


if __name__ == '__main__':
    main()
