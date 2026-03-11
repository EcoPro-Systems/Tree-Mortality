#!/usr/bin/env python
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from train_rf_model_ray import filter_inf
from util import load_config
from feature_selection import target_remap


def plot_severity_by_year(years, severity):
    years = np.array(years)
    severity = np.array(severity)
    
    unique_severities = sorted(np.unique(severity))
    n_severities = len(unique_severities)
    
    all_colors = {
        1: '#80F29F',  # Light green
        2: '#FFFF80',  # Light yellow
        3: '#FFBF80',  # Light orange
        4: '#FF8080',  # Light red
        5: '#C79FCA',  # Light purple
    }
    
    if n_severities == 3:
        severity_colors = {
            unique_severities[0]: '#80F29F',  # green
            unique_severities[1]: '#FFFF80',  # yellow
            unique_severities[2]: '#FF8080',  # red
        }
    else:
        severity_colors = {s: all_colors[s] for s in unique_severities}
    
    unique_years = sorted(np.unique(years))
    
    # Count entries per year per severity
    counts = defaultdict(lambda: defaultdict(int))
    for y, s in zip(years, severity):
        counts[y][s] += 1
    
    x = np.arange(len(unique_years))
    bar_width = 0.8 / n_severities
    
    fig, ax = plt.subplots(figsize=(max(8, len(unique_years) * 0.9), 5))
    
    for i, sev in enumerate(unique_severities):
        offsets = x - 0.4 + bar_width * i + bar_width / 2
        values = [counts[y][sev] for y in unique_years]
        ax.bar(
            offsets,
            values,
            width=bar_width * 0.9,
            color=severity_colors[sev],
            edgecolor='#888888',
            linewidth=0.6,
            label=f'Severity {sev}',
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(unique_years, fontsize=10)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Entries by Severity and Year', fontsize=14)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.spines[['top', 'right']].set_visible(False)
    
    legend_patches = [
        mpatches.Patch(facecolor=severity_colors[s], edgecolor='#888888', label=f'Severity {s}')
        for s in unique_severities
    ]
    ax.legend(handles=legend_patches, frameon=False)

    return fig


@click.command()
@click.argument('trainingfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(trainingfile, configfile, outputfile):

    config = load_config(configfile)
    target = config['target']
    remap = config.get('target_remap', None)
    feature_names = config['features']

    ds = xr.open_zarr(trainingfile)

    years = ds['year'].values
    ytrn = ds[target].to_numpy()
    ytrn = target_remap(ytrn, remap)
    Xtrn = ds[feature_names].to_array().T
    Xtrn, ytrn = filter_inf(Xtrn, ytrn)

    good = (ytrn > 0)
    Xtrn = Xtrn[good]
    ytrn = ytrn[good].astype(int)
    years = years[good].astype(int)

    fig = plot_severity_by_year(years, ytrn)

    with PdfPages(outputfile) as pdf:
        pdf.savefig(fig)


if __name__ == '__main__':
    main()
