#!/usr/bin/env python
"""
Model capacity or missing information? Every trait gain in the drought-
response experiments is an R² gain over an untuned gradient-boosting model
(response_common.hgb: 300 iterations, learning rate 0.05). Would a more
flexible learner get the same skill from climate, terrain and structure
alone, absorb the trait gain, or rescue the cross-drought transfer?

Learners (response_common.make_model):
  hgb        the default model of every experiment
  hgb_tuned  gradient boosting with leaves, minimum leaf size and the
             number of iterations tuned on an inner block split of each
             training fold
  mlp        a multilayer perceptron (256-128-64) on standardized inputs,
             early-stopped on an inner block split

Every comparison uses the same learner on both arms, the same cells and
the same 1 km block folds; feature sets are named <set>|<learner> in one
ladder, so differences between learners are paired too (block bootstrap,
1 and 5 km).

Subcommands (pilot AOIs; the held-out AOIs are refused for cycle 2):
  within    Cycle 1 (proto_response_traits.build). Env+S | Env+S+T |
            Env+S+Tres per learner; with --structure, on the cells that
            lidar source covers, Env+S | Env+S+L | Env+S+L+T | Env+S+L+Tres.
            Tres: each trait cross-fitted on the base (Env+S or Env+S+L)
            with hgb, as in the other experiments; --tuned-residuals adds
            Tres* residualized with hgb_tuned (scored with hgb_tuned).
            Contrasts: each learner's trait gains; each set under a learner
            against the same set under hgb (does the ceiling move?); and
            Env+S+T|hgb against Env+S|<learner> (traits on the default
            model against a better learner without them).
  transfer  Environment-only cycle-1 -> cycle-2 transfer per learner
            (proto_response_transfer.build_cycle, cells in both cycles):
            R², Spearman ρ and bias, pooled and within elevation quartiles
            (ρ with 1 km block-bootstrap CIs), next to the within-cycle-2
            block-CV R².
  forward   Within cycle 2 (proto_forward_pilot.build): Env+S | +Leg |
            +T18 | +Leg+T18 per learner, and the same sets across learners.

--subsample keeps whole 1 km blocks (random) up to about that many cells.
Outputs within<tag>.csv, transfer<tag>.csv, forward<tag>.csv.

    python proto_model_capacity.py within $E/hls_results/model_capacity \\
        -a neon_soap_teak -a sierra_nf
    python proto_model_capacity.py within $E/hls_results/model_capacity \\
        -a neon_soap_teak -a sierra_nf --structure aso --tag _aso
    python proto_model_capacity.py transfer $E/hls_results/model_capacity \\
        -a neon_soap_teak -a sierra_nf
    python proto_model_capacity.py forward $E/hls_results/model_capacity \\
        -a neon_soap_teak -a sierra_nf
"""
import time
import click
import numpy as np
import pandas as pd
from pathlib import Path

import response_common as rc
import proto_forward_pilot as fp
import proto_response_transfer as prt
from proto_response_metrics import ENV, S_WALL
from proto_response_traits import (TRAITS, STRUCTURES, build, feature_sets,
                                   structure_cells)
from proto_spaceborne_only import aoi_dirs
from proto_transfer_diagnostics import rho_ci

WITHIN_TARGETS = ['ndmi_recovery', 'ndmi_sens']
CYCLE_TARGETS = ['ndmi_recovery', 'nirv_recovery', 'ndmi_resilience']
KEYS = ['cell_row', 'cell_col']


@click.group()
def cli():
    pass


def options(f):
    for o in reversed([
            click.argument('outputdir', type=click.Path(path_type=Path)),
            click.option('-a', '--aoi', 'aois', multiple=True,
                         required=True),
            click.option('--learner', 'learners', multiple=True,
                         type=click.Choice(rc.LEARNERS),
                         default=rc.LEARNERS, show_default=True),
            click.option('-t', '--target', 'targets', multiple=True),
            click.option('--scale', default=3, show_default=True),
            click.option('--subsample', 'max_cells', type=int,
                         help='Keep random 1 km blocks up to about this '
                              'many cells per AOI'),
            click.option('--n-boot', default=1000, show_default=True),
            click.option('--tag', default='',
                         help='Suffix of the output file')]):
        f = o(f)
    return f


def subsample(d, n, seed=0):
    if not n or len(d) <= n:
        return d
    rng = np.random.default_rng(seed)
    sizes = d.groupby('block1000').size()
    order = rng.permutation(sizes.index.values)
    keep = order[:np.searchsorted(sizes.loc[order].cumsum().values, n) + 1]
    return d[d.block1000.isin(keep)].copy()


def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def learner_ladder(sub, t, sets, learners, pairs, n_boot, fixed=None):
    """One ladder with every set under every learner (<set>|<learner>);
    fixed: {set: learner} for sets fitted with one learner only"""
    fs, lr = {}, {}
    for lname in learners:
        for s, cols in sets.items():
            if fixed and s in fixed:
                continue
            fs[f'{s}|{lname}'] = cols
            lr[f'{s}|{lname}'] = lname
    for s, lname in (fixed or {}).items():
        fs[f'{s}|{lname}'] = sets[s]
        lr[f'{s}|{lname}'] = lname
    pairs = [(a, b) for a, b in pairs if a in fs and b in fs]
    t0 = time.time()
    r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot, pairs=pairs,
                     extra_blocks=('block5000',), learner=lr)
    g = {(x['features'], x['compare']): x for x in r
         if x['boot_blocks'] == 'block1000'}
    click.echo(f'  {t} n={len(sub)} ({time.time() - t0:.0f} s)')
    click.echo('    R²: ' + '  '.join(f'{k} {g[(k, "")]["r2"]:.3f}'
                                      for k in fs))
    for a, b in pairs:
        x = g[(b, a)]
        click.echo(f'    {b:26s} - {a:26s} {x["r2"]:+.3f} '
                   f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]')
    return r


def gain_pairs(steps, learners):
    """Within-learner steps, then each set against itself under hgb, then
    the steps' upper set under hgb against their base under each learner"""
    sets = list(dict.fromkeys(s for st in steps for s in st))
    out = [(f'{a}|{lname}', f'{b}|{lname}') for lname in learners
           for a, b in steps]
    out += [(f'{s}|hgb', f'{s}|{lname}') for lname in learners
            if lname != 'hgb' for s in sets]
    out += [(f'{a}|{lname}', f'{b}|hgb') for lname in learners
            if lname != 'hgb' for a, b in steps]
    return out


@cli.command()
@options
@click.option('--structure', 'source', type=click.Choice(list(STRUCTURES)),
              help='Lidar source: the ladder on the cells it covers')
@click.option('--tuned-residuals', is_flag=True,
              help='Also residualize the traits with hgb_tuned')
def within(outputdir, aois, learners, targets, scale, max_cells, n_boot,
           tag, source, tuned_residuals):
    """Cycle 1: Env+S ceiling and trait gains by learner"""
    outputdir.mkdir(parents=True, exist_ok=True)
    targets = list(targets) or WITHIN_TARGETS
    rows = []
    for aoi in aois:
        rdir, ddir = aoi_dirs(aoi)
        d = build(aoi, scale, rdir, ddir)
        _, T, _ = feature_sets(d)
        es = ENV + S_WALL
        base = es
        if source:
            d = structure_cells(d, aoi, scale, source)
            base = es + STRUCTURES[source]
        d = subsample(d, max_cells)
        click.echo(f'[{aoi}] {len(d)} cells' +
                   (f' covered by {source}' if source else ''))
        t0 = time.time()
        R = pd.DataFrame({c.replace('T_', 'R_'): rc.crossfit_residuals(
            d, c, base, 'block1000') for c in T}, index=d.index)
        if tuned_residuals:
            R = R.join(pd.DataFrame({c.replace('T_', 'Rt_'):
                                     rc.crossfit_residuals(
                d, c, base, 'block1000', learner='hgb_tuned')
                for c in T}, index=d.index))
        click.echo(f'[{aoi}] residuals ({time.time() - t0:.0f} s)')
        d = pd.concat([d, R], axis=1)
        Rh = [c for c in R if c.startswith('R_')]
        Rt = [c for c in R if c.startswith('Rt_')]
        if source:
            sets = {'Env+S': es, 'Env+S+L': base, 'Env+S+L+T': base + T,
                    'Env+S+L+Tres': base + Rh}
            steps = [('Env+S', 'Env+S+L'), ('Env+S+L', 'Env+S+L+T'),
                     ('Env+S+L', 'Env+S+L+Tres')]
        else:
            sets = {'Env+S': es, 'Env+S+T': es + T, 'Env+S+Tres': es + Rh}
            steps = [('Env+S', 'Env+S+T'), ('Env+S', 'Env+S+Tres')]
        top = list(sets)[-1].replace('Tres', '')
        fixed = None
        pairs = gain_pairs(steps, learners)
        if Rt and 'hgb_tuned' in learners:
            sets[f'{top}Tres*'] = base + Rt
            fixed = {f'{top}Tres*': 'hgb_tuned'}
            b = 'Env+S+L' if source else 'Env+S'
            pairs += [(f'{b}|hgb_tuned', f'{top}Tres*|hgb_tuned'),
                      (f'{top}Tres|hgb_tuned', f'{top}Tres*|hgb_tuned')]
        tags = dict(aoi=aoi, structure=source or '', scale_m=rc.RES * scale)
        for t in targets:
            sub = d[d[t].notna() & np.isfinite(d[t])]
            rows += [dict(x, **tags) for x in learner_ladder(
                sub, t, sets, learners, pairs, n_boot, fixed)]
            save(rows, outputdir / f'within{tag}.csv')


@cli.command()
@options
def transfer(outputdir, aois, learners, targets, scale, max_cells, n_boot,
             tag):
    """Environment-only cycle-1 -> cycle-2 transfer by learner"""
    outputdir.mkdir(parents=True, exist_ok=True)
    targets = list(targets) or CYCLE_TARGETS
    es = prt.ENV + S_WALL
    rows = []
    for aoi in aois:
        valid = rc.undisturbed(rc.open_env(aoi), 2025)
        cyc = {c: prt.build_cycle(aoi, scale, c, valid) for c in (1, 2)}
        common = subsample(cyc[1][KEYS + ['block1000']].merge(
            cyc[2][KEYS], on=KEYS), max_cells)[KEYS]
        cyc = {c: x.merge(common, on=KEYS).reset_index(drop=True)
               for c, x in cyc.items()}
        click.echo(f'[{aoi}] {len(common)} cells in both cycles')
        q = pd.qcut(cyc[2].elevation.rank(method='first'), 4, labels=False)
        tags = dict(aoi=aoi, scale_m=rc.RES * scale)
        for t in targets:
            a = cyc[1][cyc[1][t].notna()]
            ok = cyc[2][t].notna().values
            b, qb = cyc[2][ok], q[ok].values
            for lname in learners:
                t0 = time.time()
                m = rc.fit_model(lname, a[es], a[t].values,
                                 groups=a.block1000.values)
                p = m.predict(b[es])
                within_p = rc.oof_predict(b, es, t, 'block1000',
                                          learner=lname)
                y, bl = b[t].values, b.block1000.values
                r = dict(tags, target=t, learner=lname, n=len(b),
                         **prt.skill(y, p, bl, n_boot),
                         within_c2_r2=rc.wr2(y, within_p))
                rq = []
                for g in range(4):
                    s = qb == g
                    rho, lo, hi = rho_ci(y[s], p[s], bl[s], n_boot)
                    r.update({f'rho_q{g + 1}': rho, f'lo_q{g + 1}': lo,
                              f'hi_q{g + 1}': hi})
                    rq.append(f'{rho:+.2f} [{lo:+.2f},{hi:+.2f}]')
                rows.append(r)
                click.echo(f'  {t:16s} {lname:9s} C1->C2 R² {r["r2"]:+.3f} '
                           f'ρ {r["rho"]:+.3f} [within C2 R² '
                           f'{r["within_c2_r2"]:.3f}] | elevation '
                           f'quartiles ρ {" ".join(rq)} '
                           f'({time.time() - t0:.0f} s)')
            save(rows, outputdir / f'transfer{tag}.csv')


@cli.command()
@options
def forward(outputdir, aois, learners, targets, scale, max_cells, n_boot,
            tag):
    """Within cycle 2: legacy and 2018-trait gains by learner"""
    outputdir.mkdir(parents=True, exist_ok=True)
    targets = list(targets) or CYCLE_TARGETS
    es = prt.ENV + S_WALL
    T18 = [f'T_{x}_18' for x in TRAITS] + ['T_qcfc_18']
    leg = [f'leg_{x}' for x in fp.TARGETS]
    sets = {'Env+S': es, 'Env+S+Leg': es + leg, 'Env+S+T18': es + T18,
            'Env+S+Leg+T18': es + leg + T18}
    steps = [('Env+S', 'Env+S+Leg'), ('Env+S', 'Env+S+T18'),
             ('Env+S+Leg', 'Env+S+Leg+T18')]
    rows = []
    for aoi in aois:
        d, _ = fp.build(aoi, scale, False)
        d = subsample(d, max_cells)
        click.echo(f'[{aoi}] {len(d)} cells')
        tags = dict(aoi=aoi, scale_m=rc.RES * scale)
        for t in targets:
            sub = d[d[t].notna() & np.isfinite(d[t])]
            rows += [dict(x, **tags) for x in learner_ladder(
                sub, t, sets, learners, gain_pairs(steps, learners),
                n_boot)]
            save(rows, outputdir / f'forward{tag}.csv')


if __name__ == '__main__':
    cli()
