#!/usr/bin/env python
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from train_rf_model_ray import filter_inf
from util import load_config


def plot_features(years, X, y, year, feature_index, feature_name, test_sev):
    relevant = (years != year)
    Xtrn = X[relevant]
    ytrn = y[relevant]

    relevant = (years == year)
    Xtst = X[relevant]
    ytst = y[relevant]

    ftrn = Xtrn[:, feature_index]
    ftst = Xtst[:, feature_index]

    fig, ax = plt.subplots(figsize=(10, 6))
    for sev in sorted(np.unique(ytrn)):
        if feature_name.startswith('SP'):
            fmin = np.quantile(ftrn[ftrn > -99], 0.01)
        else:
            fmin = np.quantile(ftrn, 0.01)
        fmax = np.quantile(ftrn, 0.99)
        bins = np.linspace(fmin, fmax, 101)
        fsub = ftrn[ytrn == sev]
        ax.hist(fsub, bins=bins, density=True, alpha=0.5, label=f'Severity {sev}')

    fsub = ftst[ytst == test_sev]
    ax.hist(fsub, bins=bins, density=True, alpha=0.5, label=f'Test Severity {test_sev}', fc='black')

    ax.set_title(f'Year {year}\n{feature_name}')

    ax.legend(loc='upper right')

    return fig


def fraction_most_frequent_np(xs):
    xs = np.asarray(xs)
    vals, counts = np.unique(xs, return_counts=True)
    return counts.max() / xs.size


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
    feature_names = config['features']

    ds = xr.open_zarr(trainingfile)

    years = ds['year'].values
    ytrn = ds[target].as_numpy()
    Xtrn = ds[feature_names].to_array().T
    Xtrn, ytrn = filter_inf(Xtrn, ytrn)

    good = (ytrn > 0)
    Xtrn = Xtrn[good]
    ytrn = ytrn[good].astype(int)
    years = years[good].astype(int)

    labels = sorted(np.unique(ytrn.values))
    u_years = list(np.unique(years))

    figs = []
    for y in u_years:
        for i, f in enumerate(feature_names):
            for sev in range(1, 6):
                figs.append(plot_features(years, Xtrn, ytrn, y, i, f, sev))

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
