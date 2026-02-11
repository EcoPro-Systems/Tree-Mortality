#!/usr/bin/env python
import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from util import load_config


def filter_inf(X, y=None, fill=4):
    if np.any(np.isnan(X)):
        raise ValueError('NaN entries present but not handled')
    Xnew = np.nan_to_num(X, neginf=-fill, posinf=fill)
    if y is None:
        return Xnew
    else:
        return Xnew, y


def rbf_gamma_median_heuristic(X, subsample=None):
    if subsample is not None and subsample < X.shape[0]:
        idx = np.random.choice(X.shape[0], subsample, replace=False)
        X = X[idx]

    dists = pairwise_distances(X, metric="sqeuclidean")
    median_sq_dist = np.median(dists[dists > 0])
    return 1.0 / (2.0 * median_sq_dist)


def K(X, Y, gamma, subsample=2):
    k = rbf_kernel(X[::subsample], Y[::subsample], gamma)
    return np.average(k)


def kernel_to_sq_distance(K):
    diag = np.diag(K)
    return diag[:, None] + diag[None, :] - 2 * K


def plot_mds_with_years(X_embedded, years, figsize=(12, 6)):
    """
    Plot a 2D MDS embedding annotated with year labels.

    Parameters
    ----------
    X_embedded : array-like, shape (n_samples, 2)
        2D embedding from MDS.
    years : array-like, shape (n_samples,)
        Labels for each point (e.g., years like "2020").
    figsize : tuple, optional
        Size of the matplotlib figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = X_embedded[:, 1]
    y = X_embedded[:, 0]

    ax.scatter(x, y)

    for xi, yi, label in zip(x, y, years):
        ax.annotate(
            str(label),
            (xi, yi),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("MDS embedding annotated by year")
    ax.axis("equal")

    return fig


def plot_distance_matrix(D, years, figsize=(7, 6), cmap="viridis"):
    """
    Plot a distance matrix with year labels on both axes.

    Parameters
    ----------
    D : array-like, shape (n_samples, n_samples)
        Distance matrix.
    years : array-like, shape (n_samples,)
        Labels for rows/columns (e.g., years like "2020").
    figsize : tuple, optional
        Size of the matplotlib figure.
    cmap : str, optional
        Matplotlib colormap name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure object.
    """
    D = np.asarray(D)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(D, cmap=cmap)

    # Axis ticks and labels
    ax.set_xticks(np.arange(len(years)))
    ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticklabels(years)

    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Year")
    ax.set_ylabel("Year")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Distance")

    fig.tight_layout()
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
    feature_names = config['features']

    ds = xr.open_zarr(trainingfile)

    years = ds['year'].values
    ytrn = ds[target].as_numpy()
    Xtrn = ds[feature_names].to_array().T
    Xtrn, ytrn = filter_inf(Xtrn, ytrn)

    good = (ytrn > 0)
    Xtrn = Xtrn[good]
    years = years[good].astype(int)
    u_years = list(np.unique(years))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xtrn)

    gamma = rbf_gamma_median_heuristic(X_scaled, subsample=10000)

    Xs = [X_scaled[years == y] for y in u_years]


    kernel = np.array([
        [K(Xi, Xj, gamma) for Xi in Xs]
        for Xj in tqdm(Xs)
    ])

    D = kernel_to_sq_distance(kernel)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        metric=True,
        random_state=0,
        n_init=10,
    )
    X_embedded = mds.fit_transform(D)

    fig1 = plot_mds_with_years(X_embedded, u_years)

    fig2 = plot_distance_matrix(D, u_years)

    figs = [fig1, fig2]

    with PdfPages(outputfile) as pdf:
        for fig in tqdm(figs, 'Saving'):
            pdf.savefig(fig)


if __name__ == '__main__':
    main()
