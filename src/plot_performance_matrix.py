#!/usr/bin/env python
import os
import json
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    mean_squared_error as mse, r2_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PiYG = plt.get_cmap('PiYG')
PiYG.set_bad(color=PiYG(0))


PLOT_KWARGS = {
    'r2': {
        'cmap': PiYG,
        'im_kw': dict(vmin=-1, vmax=1),
        'include_values': True,
    },
    'rmse': {
        'cmap': 'plasma',
        'im_kw': dict(vmin=0),
        'include_values': True,
    },
}
PLOT_TITLES = {
    'r2': 'R$^2$',
    'rmse': 'RMSE',
}


def get_matrices(years, metrics):
    all_metrics = set([])
    for k, v in metrics.items():
        all_metrics |= set(v.keys())

    return {
        m: np.array([[
            metrics[yi, yj][m] for yj in years
        ] for yi in years])
        for m in all_metrics
    }


def filter_nan(true, pred):
    good = np.logical_not(np.logical_or(np.isnan(true), np.isnan(pred)))
    return true[good], pred[good]


@click.command()
@click.argument('resultfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(resultfile, outputfile):

    metrics = defaultdict(dict)
    results = np.load(resultfile)
    years = results['years']
    targets = results['targets']
    predictions = results['predictions']

    for yi, year_i in tqdm(list(enumerate(years)), 'Calculating metrics'):
        for yj, year_j in enumerate(years):
            pred = predictions[yi, yj, :]
            true = targets[yi, yj, :]
            true, pred = filter_nan(true, pred)
            metrics[year_i, year_j]['r2'] = r2_score(true, pred)
            #if year_i > year_j: metrics[year_i, year_j]['r2'] = -1
            metrics[year_i, year_j]['rmse'] = mse(true, pred, squared=False)

    matrices = get_matrices(years, metrics)

    figs = []
    for mname, matrix in sorted(matrices.items()):

        fig = plt.figure(figsize=(8, 6))
        figs.append(fig)
        ax = fig.add_subplot(111)

        M = np.ma.masked_where(matrix <= -1, matrix)
        disp = ConfusionMatrixDisplay(M, display_labels=years)
        disp.plot(ax=ax, **PLOT_KWARGS[mname])

        ax.set_xlabel('Testing Year', fontsize=15)
        ax.set_ylabel('Training Year', fontsize=15)
        ax.set_title(PLOT_TITLES[mname], fontsize=18)

    with PdfPages(outputfile) as pdf:
        for fig in figs:
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
