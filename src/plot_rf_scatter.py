#!/usr/bin/env python
import os
import json
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from plot_performance_matrix import filter_nan


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

    figs = []
    for yi, year_i in tqdm(list(enumerate(years)), 'Plotting'):
        pred = predictions[yi, yi, :]
        true = targets[yi, yi, :]
        true, pred = filter_nan(true, pred)
        lim = max(np.max(pred), np.max(true))
        lmin = -0.05 * lim
        lmax = 1.05 * lim

        fig = plt.figure(figsize=(10, 8))
        figs.append(fig)
        ax = fig.add_subplot(111)

        ax.plot(true, pred, 'k.')
        ax.plot([lmin, lmax], [lmin, lmax], 'r:')

        ax.set_xlim(lmin, lmax)
        ax.set_ylim(lmin, lmax)
        ax.set_xlabel('True Mortality (trees / acre)', fontsize=15)
        ax.set_ylabel('Predicted Mortality (trees / acre)', fontsize=15)
        ax.set_title(str(year_i), fontsize=18)

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
