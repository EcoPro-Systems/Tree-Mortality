#!/usr/bin/env python
# coding: utf-8

import os
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import LeaveOneGroupOut
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt

TRAINING_VARIABLES = [
    'PR1-1', 'PRET1-1',
    'SPI1-1', 'SPI1-2', 'SPI1-3', 'SPI1-4','SPI1-5', 'SPI1-6',
    'SPEI1-1', 'SPEI1-2', 'SPEI1-3', 'SPEI1-4','SPEI1-5', 'SPEI1-6',
    'SPI2-1', 'SPI2-2', 'SPI2-3', 'SPI2-4','SPI2-5', 'SPI2-6',
    'SPEI2-1', 'SPEI2-2', 'SPEI2-3', 'SPEI2-4','SPEI2-5', 'SPEI2-6',
    #'SPI3-1', 'SPI3-2', 'SPI3-3', 'SPI3-4','SPI3-5', 'SPI3-6',
    #'SPEI3-1', 'SPEI3-2', 'SPEI3-3', 'SPEI3-4','SPEI3-5', 'SPEI3-6',
    #'SPI4-1', 'SPI4-2', 'SPI4-3', 'SPI4-4','SPI4-5', 'SPI4-6',
    #'SPEI4-1', 'SPEI4-2', 'SPEI4-3', 'SPEI4-4','SPEI4-5', 'SPEI4-6',
    #'SPI5-1', 'SPI5-2', 'SPI5-3', 'SPI5-4','SPI5-5', 'SPI5-6',
    #'SPEI5-1', 'SPEI5-2', 'SPEI5-3', 'SPEI5-4','SPEI5-5', 'SPEI5-6',
    #'SPI6-1', 'SPI6-2', 'SPI6-3', 'SPI6-4','SPI6-5', 'SPI6-6',
    #'SPEI6-1', 'SPEI6-2', 'SPEI6-3', 'SPEI6-4','SPEI6-5', 'SPEI6-6',
    'elevation', 'northness', 'slope', #'rie', 'sapa', 'sdmv', 'tpi', 'vrm',
    'northings', 'eastings',
    'fraction_hot_days-1',
    'fraction_hot_days-2',
    'fraction_hot_days-3',
    'fraction_hot_days-4',
    'fraction_hot_days-5',
    'fraction_hot_days-6',
]

RANGES = [
    (1, (1, 3.0)),
    (2, (3, 10)),
    (3, (10, 30)),
    (4, (30, 50)),
    (5, (50, 100)),
]


def filter_inf(X, y=None, fill=1e1):
    """
    Returns a version of the dataset in which infinite values have been
    replaced with a given finite fill value
    """
    if np.any(np.isnan(X)):
        raise ValueError('NaN entries present but not handled')
        
    Xnew = np.nan_to_num(X, neginf=-fill, posinf=fill)
    
    return Xnew if y is None else (Xnew, y)


def filter_classes(X, y, years):
    return X, y, years
    good = np.logical_or(y == 2, y== 5)
    return X[good], y[good], years[good]


def filter_years(X, y, years):
    return X, y, years
    good = np.logical_or(years == 2016, years == 2018)
    return X[good], y[good], years[good]


def load_training_set(datafile, target='severity_code'):
    """
    Loads a training dataset from the specified file, selecting a specific
    year if given.

    @param datafile: NetCDF4 data file path to load
    @param target: the target variable of the regression
        (default: tpa, trees per acre)

    @return (X, y, years), where X is the feature matrix, y are the
        mortality labels, and years are separate years in the dataset
    """

    # Open dataset and select year (if specified)
    ds = xr.open_dataset(datafile)

    # Drop all variables except for the desired features
    X = ds[TRAINING_VARIABLES].to_array().T

    # Get target mortality values
    y = ds[target].as_numpy().astype(int)

    # Filter infinite values
    X, y = filter_inf(X, y)

    # Get group definitions (years)
    years = ds.year.values.astype(int)

    X, y, years = filter_years(X, y, years)

    X, y, years = filter_classes(X, y, years)
    #y = (y == 2)
    
    return X, y, years


def get_test_year(years, test_idx):
    # Check that there is only a single unique year represented in the test set, and return it
    test_year = np.unique(years[test_idx])
    assert len(test_year) == 1
    return test_year[0]


def print_results(results, metric=balanced_accuracy_score, metric_kwargs=dict(adjusted=True)):
    for year, true_pred in sorted(results.items()):
        score = metric(*true_pred, **metric_kwargs)
        print(f'{year}: {score:.2f}')

    all_results = np.vstack([
        np.column_stack([true, pred]) for true, pred in results.values()
    ])
    overall_score = metric(all_results[:, 0], all_results[:, 1], **metric_kwargs)
    print(f'Overall Score: {overall_score:.2f}')


def severity_confusion(true, pred, ranges, ax, normalize='all', cmap='PiYG', logscale=True):
    labels = [l for l, _ in ranges]
    C = confusion_matrix(true, pred, labels=labels, normalize=normalize)
    cm = colormaps[cmap]

    for i, li in enumerate(labels):
        _, (rl_i, rh_i) = ranges[i]
        for j, lj in enumerate(labels):
            _, (rl_j, rh_j) = ranges[j]
            val = C[i, j] if i == j else -C[i, j]
            val = (val + 1) / 2
            rect = Rectangle(
                (rl_i, rl_j), rh_i - rl_i, rh_j - rl_j,
                edgecolor='k', facecolor=cm(val),
            )
            ax.add_patch(rect)
            if logscale:
                tx = np.sqrt(rh_i * rl_i)
                ty = np.sqrt(rh_j * rl_j)
            else:
                if (rh_i - rl_i) < 10 or (rh_j - rl_j) < 5: continue
                tx = (rh_i + rl_i) / 2
                ty = (rh_j + rl_j) / 2
            ax.text(
                tx, ty, '{c:.1%}'.format(c=C[i, j]),
                ha='center', va='center',
            )

    ax.set_xlim(1, 100)
    ax.set_ylim(1, 100)
    ax.set_xlabel('True Severity', fontsize=14)
    ax.set_ylabel('Predicted Severity', fontsize=14)
    ax.set_aspect('equal')
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')


@click.command()
@click.argument('datafile')
@click.argument('outputfile')
def main(datafile, outputfile):

    # Load the training set by year
    X, y, years = load_training_set(datafile)

    # This is a variant of the random forest classifier
    # You can experiment with alternative models here
    model = ExtraTreesClassifier(
        n_estimators=100, max_depth=5,
        bootstrap=True, n_jobs=-1,
        class_weight='balanced_subsample',
    )

    # Define leave-one-year-out splits
    cross_val = LeaveOneGroupOut()
    splits = list(cross_val.split(X, y, years))

    results_by_year = {}

    for train_idx, test_idx in tqdm(splits, 'Training'):
        test_year = get_test_year(years, test_idx)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_true = y[test_idx]
        results_by_year[test_year] = (y_true, y_pred)

    print_results(results_by_year)

    figs = []
    for y, (true, pred) in sorted(results_by_year.items()):
        for ls in (True, False):
            fig, ax = plt.subplots()
            ax.set_title(y, fontsize=16)
            severity_confusion(true, pred, RANGES, ax, logscale=ls)
            figs.append(fig)

    with PdfPages(outputfile) as pdf:
        for fig in figs:
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
