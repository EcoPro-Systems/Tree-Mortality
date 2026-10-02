#!/usr/bin/env python
"""
Spaceborne-only comparisons, without lidar or airborne trait maps of the
place: does spaceborne-like imaging spectroscopy (VSWIR) add drought-
recovery skill over Landsat, and is the structural-carbon signal one that
needs VSWIR?

Blocks (one table, on the same cells, so every model shares the cells and
folds and every difference in R² is paired by construction):
  native, emit, sbg_hi, sbg_lo, l8
           emulated 2013 trait retrievals (and EWT, except l8) from
           proto_spaceborne_sim.py (wdts/sim/<aoi>_traits_<config>.nc)
  maps     the 2403 trait maps themselves (directions only)
  l8raw    the bands, NDVI, NDMI, NBR and NIRv of one Landsat 8 scene near
           the airborne date (fetch_landsat_c2_ee.py --scene): no airborne
           training at all
  l8multi  the same variables from the June (DOY 145-190) and Jul-Sep
           (DOY 182-273) Landsat 8 composites of the airborne year
           (landsat_composites/<aoi>_c2oli_doy*.nc): a multi-date
           multispectral block, the form in which multispectral trait
           proxies work best (e.g. Liu et al. 2024, Sentinel-2 time series)
The Jul-Sep 2013 composite is one of the twelve years the stress-response
slope (ndmi_sens, 2008-2019) is fitted on, so l8multi is partly circular
for that target; recovery (2017-19 vs 2014-16) does not use 2013.

Subcommands:
  compare  1. Over Env+S, no lidar: Env+S | Env+S+T<c> per block; gains
              over Env+S and paired differences against l8raw and l8multi.
           2. The spaceborne stack, on the cells a lidar source covers:
              Env+S+Ŝ<c>+T<c>, with Ŝ<c> the lidar structure metrics
              predicted out of fold from the same block (T, EWT where it
              exists, wall-to-wall structure and Landsat baseline
              greenness, as in proto_structure_from_spectra.py). Scored
              against the reference Env+S+L+T<native> (lidar and airborne
              retrievals) and against the Landsat stacks.
  carbon   1. Structural carbon only (SC: lignin, cellulose, fiber) and
              leaf economics only (NL: nitrogen, LMA) per emulated block,
              over Env+S, paired against the same traits emulated from real
              Landsat (l8) and against l8raw.
           2. The same blocks on top of Landsat alone (Env+S+l8raw,
              Env+S+l8multi): does the emulated chemistry add what the
              Landsat bands do not carry? l8 is the negative control (its
              traits are a function of the scene's bands).
           3. Residual-trait directions per block: each trait cross-fitted
              on Env+S (and on Env+S+l8raw), then the within-stratum
              Spearman ρ with the response and its 1 km block-bootstrap CI
              (as in proto_trait_directions.py).

Outputs in OUTPUTDIR: headtohead.csv, stack.csv, carbon_ladder.csv,
carbon_directions.csv.

    python proto_spaceborne_only.py compare $E/hls_results/spaceborne_only \\
        -a neon_soap_teak -a sierra_nf
    python proto_spaceborne_only.py carbon $E/hls_results/spaceborne_only \\
        -a neon_soap_teak -a sierra_nf
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import response_common as rc
from proto_response_metrics import ENV, S_WALL, B1
from proto_response_traits import (TRAITS, STRUCTURES, build, structure_cells,
                                   trait_layers)
from proto_trait_directions import strata_of, rho_within_boot

SIM = rc.E / 'wdts' / 'sim'
YEAR = 2013
EMULATED = ['native', 'emit', 'sbg_hi', 'sbg_lo', 'l8']
LANDSAT = ['l8raw', 'l8multi']
COMPOSITES = {'jun': 'doy145-190', 'sum': 'doy182-273'}
SC = ['Lignin', 'Cellulose', 'Fiber']
NL = ['Nitrogen', 'LMA']
TARGETS = ['ndmi_recovery', 'nirv_recovery', 'ndmi_sens']
KEYS = ['cell_row', 'cell_col']


def aoi_dirs(aoi):
    """(response, trait-dynamics) result directories of an AOI"""
    tag = '' if aoi == 'neon_soap_teak' else '_' + aoi.split('_')[0]
    h = rc.E / 'hls_results'
    return h / f'response{tag}', h / f'trait_dynamics{tag}'


def landsat_block(aoi, valid, k, c, scene, year=YEAR):
    """Cell means of the bands and indices of a Landsat scene (l8raw) or of
    the year's June and Jul-Sep composites (l8multi)"""
    if c == 'l8raw':
        srcs = {'': SIM / f'{aoi}_l8_{scene}.nc'}
    else:
        srcs = {f'_{s}': rc.E / 'landsat_composites' / f'{aoi}_c2oli_{w}.nc'
                for s, w in COMPOSITES.items()}
    layers = {}
    for tag, path in srcs.items():
        ds = xr.open_dataset(path)
        if 'year' in ds.dims:
            ds = ds.sel(year=year)
        ok = ds.n_clear.values > 0
        for v in ds.data_vars:
            if v != 'n_clear' and ds[v].ndim == 2:
                layers[f'{c}:{v}{tag}'] = np.where(
                    ok, ds[v].values, np.nan).astype(np.float32)
    return rc.cell_table(layers, valid, k)


def trait_block(aoi, valid, k, c):
    """Cell means of a block's 14 trait means (and EWT) as <c>:<trait>"""
    paths = None if c == 'maps' else (SIM / f'{aoi}_traits_{c}.nc',
                                      SIM / f'{aoi}_cwc_{c}.nc')
    t = trait_layers(aoi, valid, k, paths, YEAR)
    ren = {f'T_{x}_{YEAR}': f'{c}:{x}' for x in TRAITS}
    if c not in ('l8', 'maps'):
        ren[f'W_ewt_{YEAR}'] = f'{c}:ewt'
    return t.rename(columns=ren)


def blocks(aoi, k, configs, scene):
    """Cells with responses, Env+S and B1, and each block's columns
    '<config>:<variable>'; returns (table, {config: columns})"""
    rdir, ddir = aoi_dirs(aoi)
    d = build(aoi, k, rdir, ddir)
    d = d.drop(columns=[c for c in d if c.startswith(('T_', 'W_'))])
    valid = rc.undisturbed(rc.open_env(aoi), 2019)
    cols = {}
    for c in configs:
        t = (landsat_block(aoi, valid, k, c, scene) if c in LANDSAT
             else trait_block(aoi, valid, k, c))
        cols[c] = [x for x in t if x.startswith(f'{c}:')]
        d = d.merge(t[KEYS + cols[c]], on=KEYS, how='left')
    click.echo(f'[{aoi}] {len(d)} cells; ' + ', '.join(
        f'{c} {len(v)}' for c, v in cols.items()))
    return d, cols


def gains_msg(r, names, refs):
    """One line per block: gain over its base and paired differences"""
    g = {(x['features'], x['compare']): x for x in r
         if x['compare'] and x['boot_blocks'] == 'block1000'}
    lines = []
    for name, base in names:
        x = g[(name, base)]
        msg = f'    {name:24s} {x["r2"]:+.3f} [{x["lo"]:+.3f},{x["hi"]:+.3f}]'
        for ref in refs:
            y = g.get((name, ref))
            if y:
                msg += (f'  vs {ref.split(":")[-1]} {y["r2"]:+.3f} '
                        f'[{y["lo"]:+.3f},{y["hi"]:+.3f}]')
        lines.append(msg)
    return '\n'.join(lines)


def save(rows, path):
    """Write rows, keeping the rows of other AOIs already in the file"""
    new = pd.DataFrame(rows)
    if path.exists() and len(new):
        old = pd.read_csv(path)
        new = pd.concat([old[~old.aoi.isin(new.aoi.unique())], new])
    new.to_csv(path, index=False)


@click.group()
def cli():
    pass


def common(f):
    for opt in reversed([
            click.argument('outputdir', type=click.Path(path_type=Path)),
            click.option('-a', '--aoi', 'aois', multiple=True,
                         required=True),
            click.option('--scale', default=3, show_default=True),
            click.option('--scene', default='20130621', show_default=True,
                         help='Date of the l8raw scene (wdts/sim/'
                              '<aoi>_l8_<date>.nc)'),
            click.option('-t', '--target', 'targets', multiple=True,
                         default=TARGETS, show_default=True),
            click.option('--n-boot', default=1000, show_default=True)]):
        f = opt(f)
    return f


# ------------------------------------------------------------- compare

def headtohead(d, cols, targets, refs, n_boot):
    es = ENV + S_WALL
    fs = {'Env+S': es, **{f'Env+S+T:{c}': es + v for c, v in cols.items()}}
    ref_sets = [f'Env+S+T:{r}' for r in refs if r in cols]
    pairs = [('Env+S', f'Env+S+T:{c}') for c in cols] + \
        [(r, f'Env+S+T:{c}') for r in ref_sets for c in cols
         if f'Env+S+T:{c}' != r]
    rows = []
    for t in targets:
        sub = d[d[t].notna() & np.isfinite(d[t])]
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                         pairs=pairs, extra_blocks=('block5000',))
        rows += r
        base = [x for x in r if x['features'] == 'Env+S'
                and not x['compare']][0]
        click.echo(f'  {t} n={len(sub)} Env+S {base["r2"]:.3f}\n' +
                   gains_msg(r, [(f'Env+S+T:{c}', 'Env+S') for c in cols],
                             ref_sets))
    return rows


def spaceborne_stack(d, cols, source, targets, refs, n_boot):
    L = STRUCTURES[source]
    es = ENV + S_WALL
    hats = {}
    for c, v in cols.items():
        for m in L:
            ok = d[m].notna()
            p = pd.Series(np.nan, index=d.index)
            p[ok] = rc.oof_predict(d[ok], v + S_WALL + B1, m, 'block1000')
            hats[f'{c}:hat_{m}'] = p
    d = pd.concat([d, pd.DataFrame(hats)], axis=1)
    ref = 'Env+S+L+T:native'
    fs = {'Env+S': es, ref: es + L + cols['native']}
    for c, v in cols.items():
        fs[f'Env+S+Shat+T:{c}'] = es + [f'{c}:hat_{m}' for m in L] + v
    stacks = [f'Env+S+Shat+T:{c}' for c in cols]
    ref_sets = [f'Env+S+Shat+T:{r}' for r in refs if r in cols]
    pairs = [('Env+S', ref)] + [('Env+S', s) for s in stacks] + \
        [(ref, s) for s in stacks] + \
        [(r, s) for r in ref_sets for s in stacks if s != r]
    rows = []
    for t in targets:
        sub = d[d[t].notna() & np.isfinite(d[t])]
        r, _ = rc.ladder(sub, t, fs, 'block1000', n_boot=n_boot,
                         pairs=pairs, extra_blocks=('block5000',))
        rows += r
        a = {x['features']: x['r2'] for x in r
             if not x['compare'] and x['boot_blocks'] == 'block1000'}
        click.echo(f'  {t} n={len(sub)} Env+S {a["Env+S"]:.3f} '
                   f'reference {a[ref]:.3f}; kept ' + ', '.join(
                       f'{s.split(":")[1]} {a[s] / a[ref]:.2f}'
                       for s in stacks) + '\n' +
                   gains_msg(r, [(s, 'Env+S') for s in stacks],
                             [ref] + ref_sets))
    return rows


@cli.command()
@common
@click.option('--config', 'configs', multiple=True,
              default=EMULATED + LANDSAT, show_default=True)
@click.option('--ref', 'refs', multiple=True, default=LANDSAT,
              show_default=True)
@click.option('--stack-config', 'stack_configs', multiple=True,
              default=['native', 'emit', 'sbg_hi', 'sbg_lo'] + LANDSAT,
              show_default=True)
@click.option('--structure', 'sources', multiple=True,
              default=['aso', 'lvis2008'], show_default=True)
def compare(outputdir, aois, scale, scene, targets, n_boot, configs, refs,
            stack_configs, sources):
    """Spaceborne-like VSWIR vs Landsat without lidar"""
    outputdir.mkdir(parents=True, exist_ok=True)
    h2h, stk = [], []
    for aoi in aois:
        d, cols = blocks(aoi, scale, configs, scene)
        tags = dict(aoi=aoi, scale_m=rc.RES * scale)
        click.echo(f'[{aoi}] over Env+S')
        h2h += [dict(x, **tags) for x in headtohead(d, cols, targets, refs,
                                                    n_boot)]
        save(h2h, outputdir / 'headtohead.csv')
        for src in sources:
            sub = structure_cells(d, aoi, scale, src)
            if len(sub) < 500:
                continue
            click.echo(f'[{aoi}] spaceborne stack, {src}: {len(sub)} cells')
            sc = {c: cols[c] for c in stack_configs}
            stk += [dict(x, source=src, **tags) for x in spaceborne_stack(
                sub, sc, src, targets, refs, n_boot)]
            save(stk, outputdir / 'stack.csv')


# -------------------------------------------------------------- carbon

def carbon_ladder(d, cols, targets, n_boot):
    es = ENV + S_WALL
    emu = [c for c in cols if c not in LANDSAT]
    sub = {f'{b}:{c}': [f'{c}:{x}' for x in names]
           for b, names in (('SC', SC), ('NL', NL)) for c in emu}
    fs = {'Env+S': es, **{k: es + v for k, v in sub.items()}}
    pairs = [('Env+S', k) for k in sub]
    for k in sub:
        b, c = k.split(':')
        if c != 'l8':
            pairs.append((f'{b}:l8', k))
    names = [(k, 'Env+S') for k in sub]
    for r in LANDSAT:
        fs[f'T:{r}'] = es + cols[r]
        pairs.append(('Env+S', f'T:{r}'))
        names.append((f'T:{r}', 'Env+S'))
        for k, v in sub.items():
            fs[f'T:{r}+{k}'] = es + cols[r] + v
            pairs.append((f'T:{r}', f'T:{r}+{k}'))
            names.append((f'T:{r}+{k}', f'T:{r}'))
            b, c = k.split(':')
            pairs.append((f'T:{r}', k))
            if c != 'l8':
                pairs.append((f'T:{r}+{b}:l8', f'T:{r}+{k}'))
    rows = []
    for t in targets:
        s = d[d[t].notna() & np.isfinite(d[t])]
        r, _ = rc.ladder(s, t, fs, 'block1000', n_boot=n_boot, pairs=pairs,
                         extra_blocks=('block5000',))
        rows += r
        g = {(x['features'], x['compare']): x for x in r
             if x['compare'] and x['boot_blocks'] == 'block1000'}
        lines = []
        for name, base in names:
            x = g[(name, base)]
            msg = (f'    {name:22s} over {base:9s} {x["r2"]:+.3f} '
                   f'[{x["lo"]:+.3f},{x["hi"]:+.3f}]')
            b = name.split('+')[-1].split(':')[0]
            ctl = name.rsplit(':', 1)[0] + ':l8'
            y = g.get((name, ctl))
            if y and b in ('SC', 'NL'):
                msg += (f'  vs l8 {y["r2"]:+.3f} '
                        f'[{y["lo"]:+.3f},{y["hi"]:+.3f}]')
            lines.append(msg)
        click.echo(f'  {t} n={len(s)}\n' + '\n'.join(lines))
    return rows


def carbon_directions(d, cols, targets, n_boot):
    strata = strata_of(d)
    emu = [c for c in cols if c not in LANDSAT]
    rows = []
    for bname, base in (('Env+S', ENV + S_WALL),
                        ('Env+S+l8raw', ENV + S_WALL + cols['l8raw'])):
        R = {f'{c}:R_{x}': rc.crossfit_residuals(d, f'{c}:{x}', base,
                                                 'block1000')
             for c in emu for x in SC + NL}
        dd = pd.concat([d, pd.DataFrame(R, index=d.index)], axis=1)
        for t in targets:
            uni = {u['feature']: u for u in rc.within_strata_rho(
                dd, list(R), t, strata)}
            for col in R:
                u = uni[col]
                lo, hi = rho_within_boot(dd, col, t, strata, dd.block1000,
                                         n_boot)
                c, x = col.split(':R_')
                rows.append(dict(residual_on=bname, target=t, config=c,
                                 trait=x, n=u['n'], rho=u['rho'],
                                 rho_within=u['rho_within'], lo=lo, hi=hi))
            click.echo(f'  residual on {bname}, {t}:')
            for c in emu:
                click.echo(f'    {c:7s} ' + '  '.join(
                    f'{x["trait"][:4]} {x["rho_within"]:+.2f} '
                    f'[{x["lo"]:+.2f},{x["hi"]:+.2f}]' for x in rows
                    if x['residual_on'] == bname and x['target'] == t
                    and x['config'] == c))
    return rows


@cli.command()
@common
@click.option('--config', 'configs', multiple=True,
              default=['maps', 'native', 'emit', 'sbg_lo', 'l8'] + LANDSAT,
              show_default=True)
@click.option('--dir-boot', default=500, show_default=True,
              help='Bootstrap draws for the direction CIs')
def carbon(outputdir, aois, scale, scene, targets, n_boot, configs,
           dir_boot):
    """Is structural carbon a VSWIR signal?"""
    outputdir.mkdir(parents=True, exist_ok=True)
    lad, dirs = [], []
    for aoi in aois:
        d, cols = blocks(aoi, scale, configs, scene)
        tags = dict(aoi=aoi, scale_m=rc.RES * scale)
        click.echo(f'[{aoi}] structural carbon and leaf economics blocks')
        ladder_cols = {c: v for c, v in cols.items() if c != 'maps'}
        lad += [dict(x, **tags) for x in carbon_ladder(
            d, ladder_cols, [t for t in targets if t != 'ndmi_sens'],
            n_boot)]
        save(lad, outputdir / 'carbon_ladder.csv')
        click.echo(f'[{aoi}] residual-trait directions')
        dirs += [dict(x, **tags) for x in carbon_directions(
            d, cols, [t for t in targets if t != 'ndmi_sens'], dir_boot)]
        save(dirs, outputdir / 'carbon_directions.csv')


if __name__ == '__main__':
    cli()
