#!/usr/bin/env python
"""
Prototype analysis: is there a consistent HLS spectral signal associated with
USFS ADS mortality polygons?

For each AOI and survey year Y, compares HLS change features between pixels
inside mortality polygons (label 1) and surveyed forest pixels with no damage
feature (label 0), and a local "ring" control (label 0 pixels 100-500 m from
a mortality polygon). Change features for each band/index X:

  d1   X(Y) - X(Y-1)     change over the year leading up to the survey
  d2   X(Y) - X(Y-2)     two-year change
  lead X(Y+1) - X(Y)     change in the following year
  base X(Y) - X(2013)    change since the pre-drought baseline

With flight-matched composites (--composite-suffix _flight_w<W>.nc) every
term is taken at the pixel's overflight time of year instead: d1 = at - prev,
d2 = next - prev, lead = next - at, base = at - 2013.

Pixels are restricted to NLCD 2013 forest, not burned (MTBS) in 2012..Y+1.

Outputs (in outputdir): signal_metrics.csv (pixel AUCs), polygon_metrics.csv
(per-polygon summaries), and hls_mortality_signal.pdf.
"""
import os
import click
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages

from util import load_config

VARIABLES = ['ndvi', 'ndmi', 'nbr', 'rgi', 'evi', 'red', 'swir1', 'nir']
DELTAS = ['d1', 'd2', 'lead', 'base']
FOREST_CLASSES = [41, 42, 43]
BASE_YEAR = 2013
RING = (100, 500)  # m
MAX_SAMPLES = 200000
FPR = 0.05
SCALES = [1, 3, 9, 33]  # block sizes in pixels (30 m to ~1 km)


def block_sum(arr, k):
    """Sum over non-overlapping k x k blocks (edges trimmed)"""
    ny, nx = (arr.shape[0] // k) * k, (arr.shape[1] // k) * k
    a = arr[:ny, :nx].astype(np.float64)
    return a.reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))


def make_qc_maps(name, comp, lab, outputdir):
    """NDMI/RGI change maps with ADS mortality outlines for the survey year
    with the most mortality pixels in each era"""
    ext = [comp.x.min().item() - 15, comp.x.max().item() + 15,
           comp.y.min().item() - 15, comp.y.max().item() + 15]
    npos = (lab.label == 1).sum(dim=('y', 'x')).to_series()
    comp_years = comp.year.values.tolist()
    choices = []
    for lo, hi in [(2014, 2016), (2017, 2030)]:
        ok = [y for y in comp_years
              if y - 1 in comp_years or 'lag' in comp.dims]
        s = npos[(npos.index >= lo) & (npos.index <= hi)
                 & npos.index.isin(ok)]
        if len(s):
            choices.append(int(s.idxmax()))
    with PdfPages(outputdir / f'qc_maps_{name}.pdf') as pdf:
        for year in choices:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
            c = comp.sel(year=year)
            if 'lag' in c.dims:
                c = c.sel(lag='at')
            rgb = np.stack([c.red, c.green, c.blue], -1)
            axes[0].imshow(np.clip(rgb / 0.15, 0, 1), extent=ext)
            axes[0].set_title(f'{year} Jul-Sep true color')
            for ax, var, cmap, v in [(axes[1], 'ndmi', 'RdBu', 0.15),
                                     (axes[2], 'rgi', 'RdBu_r', 0.2)]:
                dd = delta(comp, var, year, 'd1')
                im = ax.imshow(dd, extent=ext, cmap=cmap, vmin=-v, vmax=v)
                ax.set_title(f'{var} {year} - {year - 1}' + (
                    ' (at overflight date)' if 'lag' in comp.dims else ''))
                fig.colorbar(im, ax=ax, shrink=0.7)
            pos = (lab.label.sel(year=year) == 1).values.astype(float)
            burned = lab.burned.sel(year=slice(2012, year + 1)).any('year')
            for ax in axes:
                ax.contour(lab.x, lab.y, pos, levels=[0.5], colors='k',
                           linewidths=0.5)
                ax.contourf(lab.x, lab.y, burned.values.astype(float),
                            levels=[0.5, 1.5], colors='none',
                            hatches=['////'])
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f'{name}: ADS {year} mortality polygons (black); '
                         f'MTBS burned 2012-{year + 1} (hatched)')
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outputdir / f'qc_map_{name}_{year}.png', dpi=110)
            plt.close(fig)


def host_group(host):
    host = str(host).lower()
    if 'douglas' in host:
        return 'douglas-fir'
    if 'fir' in host:
        return 'true fir'
    if 'pine' in host:
        return 'pine'
    return 'other'


FLIGHT_PAIRS = {
    'd1': ('at', 'prev'), 'd2': ('next', 'prev'),
    'lead': ('next', 'at'), 'base': ('at', 'base'),
}


def delta(comp, var, year, kind):
    yrs = set(comp.year.values.tolist())
    if 'lag' in comp.dims:
        # Flight-matched composites (hls_flight_composites.py): all terms
        # are at the pixel's overflight time of year
        if year not in yrs:
            return None
        a, b = FLIGHT_PAIRS[kind]
        c = comp[var].sel(year=year)
        return (c.sel(lag=a) - c.sel(lag=b)).values
    pairs = {
        'd1': (year, year - 1), 'd2': (year, year - 2),
        'lead': (year + 1, year), 'base': (year, BASE_YEAR),
    }
    a, b = pairs[kind]
    if a not in yrs or b not in yrs or a == b:
        return None
    return (comp[var].sel(year=a) - comp[var].sel(year=b)).values


def auc(pos, neg, rng):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) < 50 or len(neg) < 50:
        return np.nan, len(pos), len(neg)
    if len(pos) > MAX_SAMPLES:
        pos = rng.choice(pos, MAX_SAMPLES, replace=False)
    if len(neg) > MAX_SAMPLES:
        neg = rng.choice(neg, MAX_SAMPLES, replace=False)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg]), len(pos), len(neg)


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputdir', type=click.Path(
    path_type=Path
))
@click.option('--composites', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_composites')
@click.option('--composite-suffix', default='_doy182-273.nc')
@click.option('--labels', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/hls_labels')
@click.option('--landcover', type=click.Path(path_type=Path),
              default='/Volumes/Earth04/ecopro/landcover')
@click.option('-a', '--aoi', 'aois', multiple=True)
def main(configfile, outputdir, composites, composite_suffix, labels,
         landcover, aois):

    config = load_config(configfile)
    aois = aois or list(config['aois'])
    os.makedirs(outputdir, exist_ok=True)
    rng = np.random.default_rng(0)
    res = config['resolution']

    rows, poly_rows = [], []
    for name in aois:
        comp = xr.open_dataset(composites / f'{name}{composite_suffix}').load()
        lab = xr.open_dataset(labels / f'{name}.nc').load()
        polys = pd.read_csv(labels / f'{name}_polygons.csv')
        polys['host_group'] = polys.HOST.map(host_group)
        with rasterio.open(landcover / f'{name}_landcover.tif') as ds:
            forest = np.isin(ds.read(1), FOREST_CLASSES)

        burned_cum = np.cumsum(lab.burned.values, axis=0) > 0
        lab_years = lab.year.values.tolist()
        comp_years = comp.year.values.tolist()

        for year in comp_years:
            if year not in lab_years or year <= BASE_YEAR:
                continue
            yi = lab_years.index(year)
            yn = min(yi + 1, len(lab_years) - 1)
            label = lab.label.values[yi]
            valid = forest & ~burned_cum[yn]
            pos = valid & (label == 1)
            neg = valid & (label == 0)
            if pos.sum() < 50 or neg.sum() < 50:
                continue
            # Ring control: unlabeled surveyed pixels near mortality polygons
            dist = ndimage.distance_transform_edt(label != 1) * res
            ring = neg & (dist >= RING[0]) & (dist <= RING[1])
            far = neg & (dist > 1000)

            pid = lab.poly_id.values[yi]
            host = np.full(label.shape, '', dtype=object)
            host[pid >= 0] = polys.host_group.values[pid[pid >= 0]]
            sev = np.where(np.isfinite(lab.pct_sum.values[yi]),
                           lab.pct_sum.values[yi], lab.tpa_sum.values[yi])
            era = 'legacy' if year <= 2016 else 'dmsm'

            for var in VARIABLES:
                for kind in DELTAS:
                    d = delta(comp, var, year, kind)
                    if d is None:
                        continue
                    base = dict(aoi=name, year=year, era=era, var=var,
                                delta=kind)
                    for ctrl_name, ctrl in [('surveyed', neg),
                                            ('ring', ring), ('far', far)]:
                        a, npos, nneg = auc(d[pos], d[ctrl], rng)
                        rows.append(dict(base, control=ctrl_name,
                                         host='all', auc=a, n_pos=npos,
                                         n_neg=nneg))
                    for hg in ['true fir', 'pine']:
                        a, npos, nneg = auc(d[pos & (host == hg)], d[neg],
                                            rng)
                        rows.append(dict(base, control='surveyed', host=hg,
                                         auc=a, n_pos=npos, n_neg=nneg))
                    ok = pos & np.isfinite(d) & np.isfinite(sev)
                    rho = (spearmanr(d[ok], sev[ok])[0]
                           if ok.sum() > 50 else np.nan)
                    rows.append(dict(base, control='severity_spearman',
                                     host='all', auc=rho, n_pos=int(ok.sum()),
                                     n_neg=0))

            # Polygon-level summaries for the key features
            for var, kind in [('ndmi', 'd1'), ('rgi', 'd1'), ('nbr', 'd1'),
                              ('ndmi', 'base'), ('rgi', 'base')]:
                d = delta(comp, var, year, kind)
                if d is None:
                    continue
                negd = d[neg & np.isfinite(d)]
                # Threshold at FPR on surveyed-negative pixels, in the
                # direction that the index moves with mortality
                sign = -1 if var in ('ndmi', 'nbr', 'ndvi', 'evi') else 1
                thresh = np.quantile(sign * negd, 1 - FPR)
                ok = pos & np.isfinite(d)
                df = pd.DataFrame({'pid': pid[ok], 'd': d[ok],
                                   'changed': sign * d[ok] > thresh,
                                   'sev': sev[ok]})
                g = df.groupby('pid').agg(
                    n_px=('d', 'size'), median_d=('d', 'median'),
                    frac_changed=('changed', 'mean'),
                    sev_px=('sev', 'median'))
                g = g.join(polys.set_index('poly_id')[[
                    'PERCENT_MID', 'LEGACY_TPA', 'host_group',
                    'DCA_COMMON_NAME', 'ACRES', 'OBSERVATION_COUNT']])
                g['aoi'], g['year'], g['era'] = name, year, era
                g['var'], g['delta'] = var, kind
                g['bg_frac_changed'] = FPR
                poly_rows.append(g.reset_index())

            # Multi-scale agreement: in k x k pixel blocks of surveyed,
            # unburned forest, correlate the ADS-covered fraction (and
            # severity-weighted coverage) with the block-mean HLS change
            surveyed = valid & ((label == 0) | (label == 1))
            sev0 = np.where(pos, np.nan_to_num(sev), 0.0)
            for var, kind in [('ndmi', 'd1'), ('rgi', 'd1'), ('nbr', 'd1'),
                              ('ndmi', 'base'), ('rgi', 'base')]:
                d = delta(comp, var, year, kind)
                if d is None:
                    continue
                for k in SCALES:
                    m = surveyed & np.isfinite(d)
                    n = block_sum(m, k)
                    keep = n >= 0.5 * k * k
                    if keep.sum() < 30:
                        continue
                    with np.errstate(invalid='ignore', divide='ignore'):
                        frac = block_sum(pos & m, k)[keep] / n[keep]
                        sevb = block_sum(np.where(m, sev0, 0), k)[keep] / n[keep]
                        dmean = block_sum(np.where(m, d, 0), k)[keep] / n[keep]
                    for what, x in [('frac', frac), ('sev', sevb)]:
                        rows.append(dict(
                            aoi=name, year=year, era=era, var=var,
                            delta=kind, control=f'scale_{what}_spearman',
                            host='all', scale_m=k * res,
                            auc=spearmanr(x, dmean)[0],
                            n_pos=int(keep.sum()), n_neg=0,
                        ))

        make_qc_maps(name, comp, lab, outputdir)
        print(f'[{name}] done')

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outputdir / 'signal_metrics.csv', index=False)
    polym = pd.concat(poly_rows, ignore_index=True)
    polym.to_csv(outputdir / 'polygon_metrics.csv', index=False)
    make_plots(metrics, polym, outputdir)


def make_plots(metrics, polym, outputdir):
    with PdfPages(outputdir / 'hls_mortality_signal.pdf') as pdf:
        m = metrics[(metrics.control == 'surveyed') & (metrics.host == 'all')]
        # 1. AUC heatmap per variable/delta, averaged over AOI-years
        piv = m.groupby(['var', 'delta']).auc.mean().unstack()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.abs(piv.values - 0.5) + 0.5, vmin=0.5, vmax=0.9,
                       cmap='viridis')
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f'{piv.values[i, j]:.2f}', ha='center',
                        va='center', color='w', fontsize=8)
        fig.colorbar(im, label='separability (max(AUC, 1-AUC))')
        ax.set_title('Mean pixel AUC: mortality polygon vs surveyed\n'
                     'no-damage forest (all AOIs and years)')
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'auc_heatmap.png', dpi=150)
        plt.close(fig)

        # 2. AUC by year for key features, per AOI
        key = [('ndmi', 'd1'), ('rgi', 'd1'), ('nbr', 'd1'), ('ndmi', 'base')]
        fig, axes = plt.subplots(1, len(key), figsize=(4 * len(key), 3.5),
                                 sharey=True)
        for ax, (var, kind) in zip(axes, key):
            s = m[(m['var'] == var) & (m.delta == kind)]
            for aoi, sa in s.groupby('aoi'):
                ax.plot(sa.year, sa.auc, 'o-', label=aoi)
            ax.axhline(0.5, color='k', lw=0.5)
            ax.axvline(2016.5, color='gray', ls=':', lw=1)
            ax.set_title(f'{var} {kind}')
            ax.set_xlabel('survey year')
        axes[0].set_ylabel('AUC (polygon vs surveyed)')
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'auc_by_year.png', dpi=150)
        plt.close(fig)

        # 3. Controls and hosts for ndmi d1
        s = metrics[(metrics['var'] == 'ndmi') & (metrics.delta == 'd1')]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for lbl, q in [('vs surveyed', (s.control == 'surveyed')
                        & (s.host == 'all')),
                       ('vs ring 100-500m', s.control == 'ring'),
                       ('vs far >1km', s.control == 'far'),
                       ('true fir only', s.host == 'true fir'),
                       ('pine only', s.host == 'pine')]:
            ss = s[q].groupby('year').auc.mean()
            ax.plot(ss.index, ss.values, 'o-', label=lbl)
        ax.axhline(0.5, color='k', lw=0.5)
        ax.set_ylabel('AUC (mean over AOIs)')
        ax.set_title('ndmi d1: controls and host groups')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(outputdir / 'auc_controls_hosts.png', dpi=150)
        plt.close(fig)

        # 4. Agreement vs spatial scale
        s = metrics[metrics.control.str.startswith('scale_')]
        if len(s):
            key = [('ndmi', 'd1'), ('rgi', 'd1'), ('ndmi', 'base')]
            fig, axes = plt.subplots(1, len(key), figsize=(4.5 * len(key), 3.5),
                                     sharey=True)
            for ax, (var, kind) in zip(axes, key):
                ss = s[(s['var'] == var) & (s.delta == kind)]
                for (ctrl, era), g in ss.groupby(['control', 'era']):
                    gm = g.groupby('scale_m').auc.agg(['mean', 'std'])
                    ax.errorbar(gm.index, gm['mean'], gm['std'], marker='o',
                                capsize=3, label=f'{era}: {ctrl[6:-9]}')
                ax.set_xscale('log')
                ax.axhline(0, color='k', lw=0.5)
                ax.set_xlabel('block size (m)')
                ax.set_title(f'{var} {kind}')
            axes[0].set_ylabel('Spearman(ADS coverage, mean change)\n'
                               'mean +/- sd over AOI-years')
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outputdir / 'scale_agreement.png', dpi=150)
            plt.close(fig)

        # 5. Polygon-level: fraction of changed pixels vs ADS severity
        for var, kind in [('ndmi', 'd1'), ('rgi', 'd1')]:
            p = polym[(polym['var'] == var) & (polym.delta == kind)
                      & (polym.n_px >= 10)]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            leg = p[p.era == 'legacy']
            ax = axes[0]
            ax.scatter(leg.LEGACY_TPA, leg.frac_changed, s=4, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xlabel('ADS LEGACY_TPA (dead trees/acre)')
            ax.set_ylabel(f'fraction of pixels changed ({var} {kind}, '
                          f'{FPR:.0%} FPR)')
            r = spearmanr(leg.LEGACY_TPA, leg.frac_changed)[0]
            ax.set_title(f'Legacy 2014-2016 (n={len(leg)}, rho={r:.2f})')
            ax.axhline(FPR, color='r', ls=':', lw=1)
            dm = p[p.era == 'dmsm']
            ax = axes[1]
            cats = sorted(dm.PERCENT_MID.dropna().unique())
            ax.boxplot([dm.frac_changed[dm.PERCENT_MID == c] for c in cats],
                       labels=[f'{c:g}%' for c in cats], showfliers=False)
            r = spearmanr(dm.PERCENT_MID, dm.frac_changed,
                          nan_policy='omit')[0]
            ax.set_xlabel('ADS percent affected (class midpoint)')
            ax.set_title(f'DMSM 2017+ (n={len(dm)}, rho={r:.2f})')
            ax.axhline(FPR, color='r', ls=':', lw=1)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outputdir / f'polygon_severity_{var}_{kind}.png',
                        dpi=150)
            plt.close(fig)


if __name__ == '__main__':
    main()
