#!/usr/bin/env python
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    ConfusionMatrixDisplay, confusion_matrix,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from train_rf_model_ray import filter_inf
from util import load_config


def plot_confusion(k, M, labels):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    disp = ConfusionMatrixDisplay(M, display_labels=labels)
    disp.plot(ax=ax)

    ax.set_xlabel('Predicted Severity', fontsize=15)
    ax.set_ylabel('Actual Severity', fontsize=15)

    title = f'Testing Year: {k}'

    ax.set_title(title, fontsize=18)

    return fig


def plot_metrics(years, results, acc_chance, bacc_chance):
    """
    years:       list of years
    results:     list of (accuracy, balanced_accuracy) tuples
    acc_chance:  horizontal line for accuracy chance level
    bacc_chance: horizontal line for balanced accuracy chance level
    """

    years, results = zip(*zip(years, results))

    # Extract the values
    accuracies = [r[0] for r in results]
    balanced_accuracies = [r[1] for r in results]

    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bars and capture the bar containers to extract colors
    bars_acc = ax.bar(x - width/2, accuracies, width, label="Accuracy")
    bars_bacc = ax.bar(x + width/2, balanced_accuracies, width, label="Balanced Accuracy")

    # Get colors from first bar in each series
    acc_color = bars_acc.patches[0].get_facecolor()
    bacc_color = bars_bacc.patches[0].get_facecolor()

    # Add dashed horizontal lines
    ax.axhline(acc_chance, linestyle="--", color=acc_color, label="Accuracy Chance")
    ax.axhline(bacc_chance, linestyle="--", color=bacc_color, label="Balanced Accuracy Chance")

    # Labels and formatting
    ax.set_xlabel("Year")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and Balanced Accuracy by Year")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()

    return fig


def train_model(years, X, y, year, metrics=None):
    if metrics is None:
        metrics = [lambda a, b: balanced_accuracy_score(a, b, adjusted=True)]

    relevant = (years != year)
    Xtrn = X[relevant]
    ytrn = y[relevant]

    relevant = (years == year)
    Xtst = X[relevant]
    ytst = y[relevant]

    model = ExtraTreesClassifier(
        n_estimators=100, max_depth=17,
        bootstrap=True, n_jobs=-1,
        random_state=1,
        criterion="log_loss",
        class_weight='balanced',
    )
    model.fit(Xtrn, ytrn)

    ypred = model.predict(Xtst)

    results = [
        m(ytst, ypred) for m in metrics
    ]

    return results


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

    acc_chance = fraction_most_frequent_np(ytrn)
    bacc_chance = 1.0 / len(labels)

    metrics = [
        lambda a, b: accuracy_score(a, b),
        lambda a, b: balanced_accuracy_score(a, b, adjusted=False),
        lambda a, b: balanced_accuracy_score(a, b, adjusted=True),
        lambda a, b: confusion_matrix(a, b, labels=labels),
    ]

    results = [
        train_model(years, Xtrn, ytrn, year, metrics=metrics)
        for year in tqdm(u_years, 'Training')
    ]

    figs = []

    figs.append(plot_metrics(u_years, results, acc_chance, bacc_chance))

    for year, r in zip(u_years, results):
        figs.append(plot_confusion(year, r[-1], labels))

    with PdfPages(outputfile) as pdf:
        for fig in figs:
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
