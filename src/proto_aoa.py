#!/usr/bin/env python
"""
Area of applicability (Meyer & Pebesma 2021, Methods Ecol. Evol.,
doi:10.1111/2041-210X.13650) of the second-drought model trained on the
pilot AOIs, over the conifer test cells of a held-out AOI. Predictors only:
no 2020-22 response of a held-out AOI is computed or read.

Feature space: the averaged rule's model (proto_forward_pilot.py --rule
--conifer-only): Env+S with the 2020-22 forcing, Leg (the six cycle-1
Landsat responses), T18 (June 2018 traits, z-scored per AOI) and L1 (a June
2018 Landsat 8 scene). Training cells are the pilots' conifer cells, pooled.

1. Each feature is standardized by the training mean and SD and weighted by
   its permutation importance in the tuned GBM (hgb_tuned) fitted on the
   pilots for 2020-22 NDMI recovery (scored on held-out 1 km blocks; NIRv
   recovery weights as a sensitivity check). Negative importances are 0.
2. DI = d_min / (mean pairwise distance between training cells), with d_min
   the weighted Euclidean distance to the nearest training cell; for a
   training cell, to the nearest training cell outside its own 1 km block
   CV fold.
3. The AOA threshold is the upper whisker of the training cells' CV DI
   (Q75 + 1.5 IQR).

Reported (aoa_summary.csv): the share of the held-out test cells inside the
AOA, overall and per frozen elevation stratum (config/heldout_strata.yml),
with and without Leg; and the between-drought shift at the pilots: their
cycle-2 Env+S+T18 cells against cycle-1 Env+S+T13 training (weights from
the cycle-1 NDMI recovery model). Two variants drop the 2020-22 drought
anomalies (Env-anom), and the anomalies and climate normals (Terrain), which
vary mostly between AOIs rather than within them. aoa_drivers.csv lists, for the cells
outside the AOA, the mean weighted standardized difference of each feature
to the nearest training cell.

Every file opened through xarray, pandas or rasterio is recorded
(aoa_inputs.txt), and opening a held-out AOI's 2017-25 Landsat composite
(c2oli) or a cycle-2 table raises.

    python proto_aoa.py $E/hls_results/aoa -a stanislaus
"""
import contextlib
import click
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import response_common as rc
from proto_response_metrics import S_WALL

PILOTS = ('neon_soap_teak', 'sierra_nf')
PILOT_SCENE = '20180619'
SCENES = {'stanislaus': '20180626'}  # the clear L8 scene nearest Jun 19
STRATA = Path(__file__).resolve().parents[1] / 'config/heldout_strata.yml'
TARGETS = ('ndmi_recovery', 'nirv_recovery')
N_PAIR = 5000  # training cells sampled for the mean pairwise distance
N_REPEATS = 5


@contextlib.contextmanager
def guard(log):
    """Record every file opened and refuse a held-out AOI's cycle-2
    response inputs"""
    orig = {(xr, 'open_dataset'): xr.open_dataset,
            (pd, 'read_csv'): pd.read_csv,
            (rasterio, 'open'): rasterio.open}

    def wrap(fn):
        def inner(path, *a, **kw):
            p = str(path)
            if any(h in p for h in rc.HELD_OUT) and (
                    'c2oli' in p or 'cycle2' in p):
                raise AssertionError(f'cycle-2 response input of a '
                                     f'held-out AOI: {p}')
            log.add(p)
            return fn(path, *a, **kw)
        return inner

    for (mod, name), fn in orig.items():
        setattr(mod, name, wrap(fn))
    try:
        yield
    finally:
        for (mod, name), fn in orig.items():
            setattr(mod, name, fn)


def l1_cols(d, scene):
    """Rename a scene's columns to l8:<var> so pilot and test scenes align"""
    return d.rename(columns={c: 'l8:' + c.split(':', 1)[1] for c in d
                             if c.startswith(f'l8@{scene}:')})


def pilot_cells(k):
    """Pilot conifer cells: cycle-2 table with Leg, T13/T18 and L1, and the
    cycle-1 table with T13"""
    from proto_forward_pilot import build, landsat_cols
    c2, c1 = [], []
    for aoi in PILOTS:
        d, c1r = build(aoi, k, rank=False, conifer=True)
        d, _ = landsat_cols(d, aoi, k, [PILOT_SCENE], False)
        c2.append(l1_cols(d, PILOT_SCENE).assign(aoi=aoi))
        c1.append(c1r.assign(aoi=aoi))
    return (pd.concat(c2, ignore_index=True),
            pd.concat(c1, ignore_index=True))


def heldout_cells(aoi, k):
    """Predictors on a held-out AOI's test cells: the 2020-22 forcing and
    wall-to-wall structure, cycle-1 legacy, T18 and the 2018 scene"""
    from proto_forward_pilot import trait_block, TARGETS as LEG
    from proto_heldout_strata import elevation_cells
    from proto_response_transfer import build_cycle, cycle_forcing
    from proto_spaceborne_only import landsat_block
    key = ['cell_row', 'cell_col']
    valid = rc.undisturbed(rc.open_env(aoi), 2025)
    d = rc.cell_table(cycle_forcing(aoi, 2), valid, k)
    # the frozen strata use elevation_cells' elevation (site pixels)
    test = elevation_cells(aoi, k, conifer=True)[key + ['elevation']]
    d = d.merge(test.rename(columns={'elevation': 'stratum_elevation'}),
                on=key)
    c1 = build_cycle(aoi, k, 1, valid)
    d = d.merge(c1[key + LEG].rename(columns={t: f'leg_{t}' for t in LEG}),
                on=key, how='left')
    d = d.merge(trait_block(aoi, valid, k, 2018, False), on=key, how='left')
    scene = SCENES[aoi]
    lb = landsat_block(aoi, valid, k, 'l8raw', scene)
    lb = lb.rename(columns={c: c.replace('l8raw:', 'l8:') for c in lb})
    return d.merge(lb[key + [c for c in lb if c.startswith('l8:')]], on=key,
                   how='left')


def weights(train, cols, target, learner='hgb_tuned'):
    """Permutation importance on held-out blocks (negatives -> 0)"""
    sub = train[train[target].notna()]
    tr, va = rc.inner_split(len(sub), sub.block1000.values, frac=0.2)
    m = rc.fit_model(learner, sub[cols].values[tr], sub[target].values[tr],
                     groups=sub.block1000.values[tr])
    pi = permutation_importance(_Fitted(m), sub[cols].values[va],
                                sub[target].values[va], scoring='r2',
                                n_repeats=N_REPEATS, random_state=0)
    return np.clip(pi.importances_mean, 0, None)


class _Fitted:
    """A fitted model as sklearn's scorers expect it"""
    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.m.predict(X)


def aoa(train, test, cols, w, seed=0):
    """DI of the test cells and the CV DI of the training cells; returns
    (di_test, di_train, threshold, nearest-training-cell differences)"""
    from sklearn.neighbors import NearestNeighbors
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    Xt = ((train[cols] - mu) / sd).values * w
    Xq = ((test[cols] - mu) / sd).values * w
    rng = np.random.default_rng(seed)
    i = rng.choice(len(Xt), min(N_PAIR, len(Xt)), replace=False)
    from scipy.spatial.distance import pdist
    dbar = pdist(Xt[i]).mean()
    di_train = np.full(len(Xt), np.nan)
    for tr, te in GroupKFold(n_splits=rc.N_SPLITS).split(
            Xt, groups=train.block1000.values):
        nn = NearestNeighbors(n_neighbors=1).fit(Xt[tr])
        di_train[te] = nn.kneighbors(Xt[te])[0][:, 0] / dbar
    nn = NearestNeighbors(n_neighbors=1).fit(Xt)
    dist, j = nn.kneighbors(Xq)
    di_test = dist[:, 0] / dbar
    q1, q3 = np.quantile(di_train, [0.25, 0.75])
    thr = q3 + 1.5 * (q3 - q1)
    diff = np.abs(Xq - Xt[j[:, 0]])
    return di_test, di_train, thr, diff


def summary_rows(test, inside, label, strata_cuts=None, **tags):
    from proto_heldout_strata import elevation_stratum
    rows = [dict(cells=label, stratum='all', n=len(test),
                 pct_inside=100 * inside.mean(), **tags)]
    if strata_cuts is not None:
        s = elevation_stratum(test.stratum_elevation.values, strata_cuts)
        for g in range(len(strata_cuts) + 1):
            m = s == g
            rows.append(dict(cells=label, stratum=f'q{g + 1}', n=int(m.sum()),
                             pct_inside=100 * inside[m].mean(), **tags))
    return rows


def run(train, test, cols, target, label, cuts=None, **tags):
    """AOA of test against train on complete cases; returns (summary rows,
    driver rows, threshold)"""
    tr = train.dropna(subset=cols + [target])
    te = test.dropna(subset=cols)
    w = weights(tr, cols, target)
    di, di_tr, thr, diff = aoa(tr, te, cols, w)
    inside = di <= thr
    rows = summary_rows(te, inside, label, cuts, weights=target,
                        n_test_all=len(test), threshold=thr,
                        train_di_median=float(np.median(di_tr)),
                        test_di_median=float(np.median(di)), **tags)
    out = diff[~inside].mean(0) if (~inside).any() else np.zeros(len(cols))
    drivers = [dict(cells=label, weights=target, feature=c, weight=wi,
                    mean_wdiff_outside=v, **tags)
               for c, wi, v in zip(cols, w, out)]
    click.echo(f'  {label:28s} {target:14s} {tags} test {len(te)}/'
               f'{len(test)} complete, inside {100 * inside.mean():.1f}% '
               f'(threshold {thr:.3f})')
    return rows, drivers


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-a', '--aoi', 'aois', multiple=True, default=['stanislaus'],
              show_default=True)
@click.option('--scale', 'k', default=3, show_default=True)
def main(outputdir, aois, k):
    from proto_forward_pilot import TARGETS as LEG
    from proto_heldout_strata import load_cutpoints
    from proto_response_transfer import ENV
    from proto_response_traits import TRAITS
    outputdir.mkdir(parents=True, exist_ok=True)
    opened = set()
    rows, drivers = [], []
    with guard(opened):
        c2, c1 = pilot_cells(k)
        click.echo(f'pilot conifer cells: cycle 2 {len(c2)}, cycle 1 '
                   f'{len(c1)}')
        es = ENV + S_WALL
        leg = [f'leg_{t}' for t in LEG]
        t18 = [f'T_{x}_18' for x in TRAITS] + ['T_qcfc_18']
        l1 = [c for c in c2 if c.startswith('l8:')]
        # the drought anomalies and climate normals come from 4 km
        # climate grids and barely vary within an AOI: two pilot AOIs give
        # about two points in that space. The variants without them show
        # how far the rest of the feature space carries.
        anom = [c for c in ENV if c.endswith('_dr') or c.endswith('_cum')]
        clim = ['cwd_clim', 'ppt_clim', 'tmx_clim']
        no_anom = [c for c in es if c not in anom]
        local = [c for c in no_anom if c not in clim]
        sets = {'Env+S+Leg+T18+L1': es + leg + t18 + l1,
                'Env+S+T18+L1': es + t18 + l1,
                'Env-anom+S+Leg+T18+L1': no_anom + leg + t18 + l1,
                'Terrain+S+Leg+T18+L1': local + leg + t18 + l1}
        for aoi in aois:
            test = heldout_cells(aoi, k)
            cuts = load_cutpoints(STRATA, aoi)
            click.echo(f'[{aoi}] {len(test)} test cells')
            for name, cols in sets.items():
                for t in TARGETS:
                    r, dr = run(c2, test, cols, t, aoi, cuts, features=name)
                    rows += r
                    drivers += dr
        # between droughts at the pilots: cycle-1 Env+S+T13 training,
        # cycle-2 Env+S+T18 cells, with the trait year dropped from names
        t13 = [f'T_{x}_13' for x in TRAITS] + ['T_qcfc_13']
        gen = {c: c.rsplit('_', 1)[0] for c in t13 + t18}
        a = c1.rename(columns=gen)
        b = c2.rename(columns=gen)
        cols = es + [gen[c] for c in t13]
        for p in PILOTS + ('pooled',):
            aa = a if p == 'pooled' else a[a.aoi == p]
            bb = b if p == 'pooled' else b[b.aoi == p]
            for t in TARGETS:
                r, dr = run(aa, bb, cols, t, f'{p} cycle 2 vs 1',
                            features='Env+S+T')
                rows += r
                drivers += dr
    pd.DataFrame(rows).to_csv(outputdir / 'aoa_summary.csv', index=False)
    pd.DataFrame(drivers).to_csv(outputdir / 'aoa_drivers.csv', index=False)
    (outputdir / 'aoa_inputs.txt').write_text('\n'.join(sorted(opened)) +
                                              '\n')
    click.echo(f'{len(opened)} input files; none is a held-out cycle-2 '
               'response input')


if __name__ == '__main__':
    main()
