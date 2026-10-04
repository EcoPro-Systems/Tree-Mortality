#!/usr/bin/env python
"""
Collect fold-assignment reruns: for each paired contrast (an R² gain with
its 1 km block-bootstrap CI), the default GroupKFold assignment beside the
runs with blocks shuffled into folds (--fold-seed, files tagged _fold<s>).

The block bootstrap holds the folds fixed, so it does not show how much a
contrast moves when the 1 km blocks are assigned to folds differently. A
contrast is called robust here if its CI excludes 0, with the same sign,
under the default assignment and under every shuffled one.

One CSV per family in OUTPUTDIR (<family>.csv) with, per contrast:
r2/lo/hi under each assignment, the range of the point estimates, and
robust (True/False). Families whose files are missing are skipped.

    python collect_fold_seeds.py $E/hls_results/fold_robustness
"""
import re
import click
import numpy as np
import pandas as pd
from pathlib import Path

import response_common as rc

H = rc.E / 'hls_results'
# family: ([(dir, file stem)], extra key columns)
FAMILIES = {
    'spaceborne_headtohead': ([('spaceborne_only', 'headtohead')], []),
    'spaceborne_stack': ([('spaceborne_only', 'stack')], ['source']),
    'carbon_ladder': ([('spaceborne_only/carbon_neon_soap_teak',
                        'carbon_ladder'),
                       ('spaceborne_only/carbon_sierra_nf', 'carbon_ladder')],
                      []),
    'trait_dynamics': ([('trait_dynamics', 'dynamics_ladder'),
                        ('trait_dynamics_sierra', 'dynamics_ladder')],
                       ['interval']),
    'lidar_neon2013': ([('lidar_check', 'lidar_check_neon2013')], []),
    'lidar_lvis2008': ([('lidar_check', 'lidar_check_lvis2008')], []),
    'lidar_aso': ([('lidar_check', 'lidar_check_aso')], []),
    'neighbour': ([('neighbour_baseline', 'neighbour')], ['radius_m']),
    'forward_pilot': ([('forward_pilot', 'ladder_z')], []),
    'no_lidar': ([('structure_from_spectra', 'no_lidar_ladder')],
                 ['source']),
}
SEED_RE = re.compile(r'_fold(\d+)')


def read_family(locs):
    """All rows of the default and tagged files, with an assignment column
    ('default' or the seed)"""
    frames = []
    for d, stem in locs:
        for f in sorted((H / d).glob(f'{stem}*.csv')):
            rest = f.stem[len(stem):]
            if rest and not SEED_RE.match(rest):
                continue  # another tag, e.g. a different experiment
            x = pd.read_csv(f)
            m = SEED_RE.match(rest)
            x['assignment'] = m.group(1) if m else 'default'
            frames.append(x)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize(df, extra):
    df = df[df['compare'].notna() & (df['compare'] != '')]
    if 'boot_blocks' in df:
        df = df[df.boot_blocks == 'block1000']
    keys = [c for c in ['aoi', 'scale_m'] + extra + ['target', 'features',
                                                       'compare']
            if c in df]
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        g = g.drop_duplicates('assignment', keep='last')
        if 'default' not in set(g.assignment) or len(g) < 2:
            continue
        row = dict(zip(keys, k))
        for _, x in g.sort_values('assignment').iterrows():
            a = x.assignment
            row.update({f'r2_{a}': x.r2, f'lo_{a}': x.lo, f'hi_{a}': x.hi})
        pos, neg = (g.lo > 0).all(), (g.hi < 0).all()
        row.update(n_assignments=len(g), r2_min=g.r2.min(),
                   r2_max=g.r2.max(), spread=g.r2.max() - g.r2.min(),
                   robust=bool(pos or neg))
        rows.append(row)
    return pd.DataFrame(rows)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('-f', '--family', 'families', multiple=True,
              default=list(FAMILIES), show_default=True)
def main(outputdir, families):
    outputdir.mkdir(parents=True, exist_ok=True)
    for name in families:
        locs, extra = FAMILIES[name]
        df = read_family(locs)
        if df is None:
            click.echo(f'{name}: no files')
            continue
        s = summarize(df, extra)
        if not len(s):
            click.echo(f'{name}: no reruns yet')
            continue
        s.to_csv(outputdir / f'{name}.csv', index=False)
        click.echo(f'{name}: {len(s)} contrasts, {int(s.robust.sum())} '
                   f'robust; median spread {np.median(s.spread):.4f}, max '
                   f'{s.spread.max():.4f}')


if __name__ == '__main__':
    main()
