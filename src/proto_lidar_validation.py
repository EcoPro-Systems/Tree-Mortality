#!/usr/bin/env python
"""
Do the ASO composite and LVIS 2008 structure layers (fetch_lidar_structure
.py) agree with the NEON 2013 lidar trees, and does the ASO composite,
flown during the 2015-16 die-off, already record the mortality it would be
used to control for?

1. Agreement: Spearman ρ between matching structure variables over
   undisturbed forest cells at 90 and 270 m (NEON 2013 tree summaries, ASO,
   LVIS, GLAD 2010 height, TCC 2010).
2. Leakage: on the NEON lidar-tree cells, the gain in R² for the lidar
   mortality fraction (trees live in 2013, dead by 2017-18) from adding ASO
   or LVIS structure to Env+S+NEON-2013 structure. A pre-drought source
   (LVIS 2008) should add little once NEON 2013 structure is in; a large ASO
   gain means ASO records the die-off itself. Also the within-strata ρ of
   each ASO variable with the mortality fraction.
3. Maps of the main height variables.

    python proto_lidar_validation.py $E/hls_results/lidar_check
"""
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

import response_common as rc
from proto_response_metrics import ENV, S_WALL, S_LIDAR
from proto_response_traits import S_ASO, S_LVIS, MIN_COVER

AOI = 'neon_soap_teak'
PAIRS = [('lidar_h_mean', 'aso_chm_mean'), ('lidar_h_p90', 'aso_rh98'),
         ('lidar_h_max', 'aso_chm_max'), ('lidar_frac_tall', 'aso_frac_tall'),
         ('lidar_h_mean', 'lvis_rh100_mean'), ('lidar_h_max', 'lvis_rh100_max'),
         ('lidar_frac_tall', 'lvis_frac_tall'),
         ('aso_rh98', 'lvis_rh100_mean'), ('aso_chm_max', 'lvis_rh100_max'),
         ('aso_rh50', 'lvis_rh50_mean'), ('aso_frac_tall', 'lvis_frac_tall'),
         ('glad_h2010', 'aso_rh98'), ('glad_h2010', 'lvis_rh100_mean'),
         ('glad_h2010', 'lidar_h_p90'), ('tcc2010', 'aso_cover5')]
ELEV_BAND = 200


def cells(aoi, k, response_dir, dynamics_dir):
    env = rc.open_env(aoi)
    valid = rc.undisturbed(env, 2019)
    lid = xr.open_dataset(rc.E / 'lidar' / f'{aoi}_lidar.nc')
    nan = np.full(valid.shape, np.nan, np.float32)

    def get(ds, c):  # not every source covers every AOI
        return ds[c].values.astype(np.float32) if c in ds else nan
    layers = {c: get(lid, c) for c in S_ASO + S_LVIS + ['aso_snowoff_year']}
    for c in ('aso_cov', 'lvis_cov'):
        layers[c] = np.nan_to_num(get(lid, c))
    for c in S_LIDAR:
        layers[c] = get(env, c)
    d = rc.cell_table(layers, valid, k)
    d.loc[d.aso_cov < MIN_COVER, S_ASO] = np.nan
    d.loc[d.lvis_cov < MIN_COVER, S_LVIS] = np.nan
    m = pd.read_csv(response_dir / f'metrics_{aoi}_{rc.RES * k}m.csv',
                    usecols=lambda c: c in ENV + S_WALL + ['cell_row',
                                                           'cell_col'])
    d = d.merge(m, on=['cell_row', 'cell_col'], how='inner')
    dyn = dynamics_dir / f'dynamics_{aoi}_{rc.RES * k}m.csv'
    if dyn.exists():
        dd = pd.read_csv(dyn, usecols=lambda c: c in (
            'cell_row', 'cell_col', 'mort_frac', 'mort_n'))
        d = d.merge(dd, on=['cell_row', 'cell_col'], how='left')
    return d


def fig_maps(aoi, path):
    env = rc.open_env(aoi)
    lid = xr.open_dataset(rc.E / 'lidar' / f'{aoi}_lidar.nc')
    panels = [('GLAD height 2010 (m)', env.glad_h2010.values),
              ('ASO RH98 (m)', lid.aso_rh98.values),
              ('ASO snow-off year', lid.aso_snowoff_year.values)]
    if 'lvis_rh100_mean' in lid:
        panels.append(('LVIS 2008 RH100 (m)', lid.lvis_rh100_mean.values))
    if 'lidar_h_p90' in env:
        panels.append(('NEON 2013 tree height p90 (m)',
                       env.lidar_h_p90.values))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6))
    for ax, (title, a) in zip(np.atleast_1d(axes), panels):
        vmax = None if 'year' in title else 50
        im = ax.imshow(a, vmin=None if vmax is None else 0, vmax=vmax,
                       cmap='viridis', interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(aoi)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.option('--n-boot', default=1000, show_default=True)
def main(outputdir, n_boot):
    outputdir.mkdir(parents=True, exist_ok=True)
    E = rc.E / 'hls_results'
    dirs = {AOI: (E / 'response', E / 'trait_dynamics'),
            'sierra_nf': (E / 'response_sierra', E / 'trait_dynamics_sierra')}
    agree, leak, uni = [], [], []
    for aoi, (rdir, ddir) in dirs.items():
        fig_maps(aoi, outputdir / f'structure_maps_{aoi}.png')
        for k in (3, 9):
            if not (rdir / f'metrics_{aoi}_{rc.RES * k}m.csv').exists():
                continue
            d = cells(aoi, k, rdir, ddir)
            for a, b in PAIRS:
                ok = d[a].notna() & d[b].notna()
                if ok.sum() < 100:
                    continue
                agree.append(dict(aoi=aoi, scale_m=rc.RES * k, x=a, y=b,
                                  n=int(ok.sum()),
                                  rho=spearmanr(d[a][ok], d[b][ok])[0],
                                  median_diff=float(np.median(
                                      d[b][ok] - d[a][ok]))))
            if aoi != AOI or k != 3:
                continue
            # Leakage on the NEON lidar-tree cells
            es = ENV + S_WALL
            for src, L in (('aso', S_ASO), ('lvis2008', S_LVIS)):
                sub = d[d.mort_frac.notna() & d[L[0]].notna()]
                if len(sub) < 500:
                    continue
                fs = {'Env+S': es, 'Env+S+Lneon': es + S_LIDAR,
                      'Env+S+L': es + L, 'Env+S+Lneon+L': es + S_LIDAR + L}
                r, _ = rc.ladder(sub, 'mort_frac', fs, 'block1000',
                                 weight='mort_n', n_boot=n_boot,
                                 pairs=[('Env+S', 'Env+S+Lneon'),
                                        ('Env+S', 'Env+S+L'),
                                        ('Env+S+Lneon', 'Env+S+Lneon+L')])
                for x in r:
                    x.update(aoi=aoi, scale_m=rc.RES * k, source=src)
                leak += r
                g = {f'{x["features"]}-{x["compare"]}': x for x in r
                     if x['compare'] and x['boot_blocks'] == 'block1000'}
                click.echo(f'[{aoi} 90 m] mortality, {src} ({len(sub)} '
                           'cells): ' + '  '.join(
                               f'{kk} {v["r2"]:+.3f} [{v["lo"]:+.3f},'
                               f'{v["hi"]:+.3f}]' for kk, v in g.items()))
            strata = ((d.cwd_clim // d.cwd_clim.std()).astype(str) + '_' +
                      (d.elevation // ELEV_BAND).astype(int).astype(str))
            for u in rc.within_strata_rho(d, S_ASO + S_LVIS + S_LIDAR,
                                          'mort_frac', strata):
                uni.append(dict(aoi=aoi, scale_m=rc.RES * k, **u))
    a = pd.DataFrame(agree)
    a.to_csv(outputdir / 'structure_agreement.csv', index=False)
    click.echo(a.round(3).to_string(index=False))
    pd.DataFrame(leak).to_csv(outputdir / 'structure_leakage.csv',
                              index=False)
    u = pd.DataFrame(uni)
    u.to_csv(outputdir / 'structure_vs_mortality.csv', index=False)
    click.echo(u.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
