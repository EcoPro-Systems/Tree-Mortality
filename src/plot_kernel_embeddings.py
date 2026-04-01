#!/usr/bin/env python
"""
plot_year_embeddings.py

Loads a precomputed pairwise kernel matrix (as saved by compute_year_kernel_matrix.py)
and produces 2-D embeddings of all years using two methods:

  MDS     -- Classical MDS applied to all years simultaneously using the full
             pairwise distance matrix derived from the kernel matrix.

  Isomap  -- Isomap fitted on training years only, then out-of-sample projection
             years are embedded via the Nyström / barycentric extension using
             only their distances to the training points.

In both plots:
  - Training years are shown as large white-filled markers with black outlines
    and individual year text labels.
  - Projection years are shown as filled circles coloured by year using a
    sequential colourmap, with a colourbar for reference.

Usage:
    python plot_year_embeddings.py KERNELFILE CONFIGFILE OUTPUTFILE

Arguments:
    KERNELFILE   .npz file produced by compute_year_kernel_matrix.py
                 (arrays: "kernel" and "years")
    CONFIGFILE   JSON config; must contain "projection_years": {"start": int, "end": int}
    OUTPUTFILE   Output PDF (or any format matplotlib supports, e.g. .png, .svg)
"""

import json
import click
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.manifold import MDS, Isomap
from sklearn.metrics import pairwise_distances

from util import load_config


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

TRAIN_MARKER_SIZE   = 90      # scatter point area for training years
PROJ_MARKER_SIZE    = 55      # scatter point area for projection years
TRAIN_LINEWIDTH     = 1.4
PROJ_CMAP           = 'plasma' # sequential colourmap for projection years
TRAIN_FACE_COLOR    = 'white'
TRAIN_EDGE_COLOR    = '#1a1a2e'
LABEL_FONTSIZE      = 7.5
LABEL_OFFSET        = (4, 4)   # points
FIG_WIDTH           = 7.5      # inches per panel
FIG_HEIGHT          = 6.5      # inches per panel
DPI                 = 150


# ---------------------------------------------------------------------------
# Kernel → squared distance
# ---------------------------------------------------------------------------

def kernel_to_sq_distance(K_mat):
    """Convert a kernel matrix to a squared Euclidean distance matrix."""
    diag = np.diag(K_mat)
    return diag[:, None] + diag[None, :] - 2.0 * K_mat


# ---------------------------------------------------------------------------
# Out-of-sample Isomap extension (barycentric / Nyström)
# ---------------------------------------------------------------------------

def isomap_transform_oos(isomap, D_oos_train):
    """
    Embed out-of-sample points using a fitted Isomap model.

    Parameters
    ----------
    isomap : fitted sklearn.manifold.Isomap
        Must have been fitted with metric="precomputed".
    D_oos_train : (n_oos, n_train) array
        Geodesic-distance matrix from each out-of-sample point to every
        training point.  Here we approximate geodesic distances with the
        raw precomputed distances (same approximation used internally by
        sklearn's Isomap for new points via transform()).

    Returns
    -------
    X_oos : (n_oos, n_components) array
    """
    # sklearn's Isomap.transform() accepts a (n_new, n_train) distance matrix
    # when the model was fitted with metric="precomputed".
    return isomap.transform(D_oos_train)


# ---------------------------------------------------------------------------
# Shared plotting helper
# ---------------------------------------------------------------------------

def _setup_fig_ax(title):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12,
                 color='#1a1a2e', fontfamily='monospace')
    ax.set_xlabel('Embedding dimension 1', fontsize=10, color='#3a3a5c')
    ax.set_ylabel('Embedding dimension 2', fontsize=10, color='#3a3a5c')
    ax.tick_params(labelsize=8, colors='#3a3a5c')
    for spine in ax.spines.values():
        spine.set_edgecolor('#ccccdd')
    ax.set_facecolor('#f9f9fc')
    fig.patch.set_facecolor('white')
    return fig, ax


def plot_embedding(ax, X_train, X_proj, train_years, proj_years, cmap_name=PROJ_CMAP):
    """
    Render one embedding panel onto *ax*.

    Parameters
    ----------
    X_train : (n_train, 2)  embedding coordinates for training years
    X_proj  : (n_proj, 2)   embedding coordinates for projection years
    train_years : (n_train,) int year labels
    proj_years  : (n_proj,)  int year labels
    """
    proj_years = np.asarray(proj_years)
    train_years = np.asarray(train_years)

    # -- Projection years: filled circles, coloured by year ----------------
    cmap = mpl.colormaps[cmap_name]
    y_min, y_max = proj_years.min(), proj_years.max()
    norm = mpl.colors.Normalize(vmin=y_min, vmax=y_max)
    colors = cmap(norm(proj_years))

    sc = ax.scatter(
        X_proj[:, 0], X_proj[:, 1],
        c=proj_years,
        cmap=cmap_name,
        norm=norm,
        s=PROJ_MARKER_SIZE,
        zorder=2,
        alpha=0.82,
        linewidths=0,
        label='Projection years',
    )

    # Colourbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.035, aspect=28)
    cbar.set_label('Projection year', fontsize=9, color='#3a3a5c')
    cbar.ax.tick_params(labelsize=8, colors='#3a3a5c')
    cbar.outline.set_edgecolor('#ccccdd')
    # Show only a handful of ticks to keep it clean
    cbar.locator = ticker.MaxNLocator(nbins=6, integer=True)
    cbar.update_ticks()

    # -- Training years: hollow markers with text labels -------------------
    ax.scatter(
        X_train[:, 0], X_train[:, 1],
        s=TRAIN_MARKER_SIZE,
        facecolors=TRAIN_FACE_COLOR,
        edgecolors=TRAIN_EDGE_COLOR,
        linewidths=TRAIN_LINEWIDTH,
        zorder=4,
        label='Training years',
    )
    for xi, yi, yr in zip(X_train[:, 0], X_train[:, 1], train_years):
        ax.annotate(
            str(yr),
            (xi, yi),
            xytext=LABEL_OFFSET,
            textcoords='offset points',
            fontsize=LABEL_FONTSIZE,
            color=TRAIN_EDGE_COLOR,
            fontfamily='monospace',
            zorder=5,
        )

    # -- Legend ------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=TRAIN_FACE_COLOR,
               markeredgecolor=TRAIN_EDGE_COLOR,
               markeredgewidth=TRAIN_LINEWIDTH,
               markersize=8, label='Training years (2012–2023)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cmap(0.65),
               markersize=8, label='Projection years (2024–2100)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0.9,
              edgecolor='#ccccdd', facecolor='white',
              loc='best', borderpad=0.8)

    ax.axis('equal')
    return sc


# ---------------------------------------------------------------------------
# Method 1: MDS on the full distance matrix
# ---------------------------------------------------------------------------

def embed_mds(D, n_init=10, random_state=0):
    """
    Classical (metric) MDS on the full (n_all × n_all) distance matrix.
    Returns (n_all, 2) embedding.
    """
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        metric=True,
        normalized_stress='auto',
        n_init=n_init,
        random_state=random_state,
    )
    return mds.fit_transform(D)


def plot_mds(D, train_years, proj_years):
    all_years  = list(train_years) + list(proj_years)
    n_train    = len(train_years)

    print('  Fitting MDS...')
    X_all = embed_mds(np.sqrt(np.clip(D, 0, None)))   # MDS expects distances, not sq-distances

    X_train = X_all[:n_train]
    X_proj  = X_all[n_train:]

    fig, ax = _setup_fig_ax('Feature-space embedding — MDS (all years)')
    plot_embedding(ax, X_train, X_proj, train_years, proj_years)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Method 2: Isomap trained on training years, projected for future years
# ---------------------------------------------------------------------------

def embed_isomap(D_train, D_proj_train, n_neighbors=5, random_state=0):
    """
    Fit Isomap on training points, then embed projection points out-of-sample.

    Parameters
    ----------
    D_train       : (n_train, n_train)  pairwise distances among training years
    D_proj_train  : (n_proj, n_train)   distances from each proj year to each training year

    Returns
    -------
    X_train : (n_train, 2)
    X_proj  : (n_proj,  2)
    """
    iso = Isomap(
        n_components=2,
        metric='precomputed',
        n_neighbors=n_neighbors,
    )
    print('  Fitting Isomap on training years...')
    X_train = iso.fit_transform(D_train)

    print('  Embedding projection years (out-of-sample)...')
    X_proj = isomap_transform_oos(iso, D_proj_train)

    return X_train, X_proj


def plot_isomap(D, train_years, proj_years, n_neighbors=5):
    n_train = len(train_years)
    n_proj  = len(proj_years)

    D_dist = np.sqrt(np.clip(D, 0, None))   # convert sq-distance → distance

    D_train      = D_dist[:n_train, :n_train]          # (n_train, n_train)
    D_proj_train = D_dist[n_train:, :n_train]           # (n_proj,  n_train)

    X_train, X_proj = embed_isomap(D_train, D_proj_train, n_neighbors=n_neighbors)

    fig, ax = _setup_fig_ax(
        f'Feature-space embedding — Isomap (fitted on training, projected out-of-sample)'
    )
    plot_embedding(ax, X_train, X_proj, train_years, proj_years)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('kernelfile',  type=click.Path(path_type=Path, exists=True))
@click.argument('configfile',  type=click.Path(path_type=Path, exists=True))
@click.argument('outputfile',  type=click.Path(path_type=Path, exists=False))
@click.option('--n-neighbors', default=5, show_default=True,
              help='Number of neighbours for the Isomap graph.')
@click.option('--method', default='both',
              type=click.Choice(['mds', 'isomap', 'both'], case_sensitive=False),
              show_default=True,
              help='Which embedding method(s) to plot.')
def main(kernelfile, configfile, outputfile, n_neighbors, method):
    """
    Plot MDS and/or Isomap embeddings from a precomputed kernel matrix.
    Results are written to OUTPUTFILE (PDF by default; suffix determines format).
    """

    # ------------------------------------------------------------------
    # 1. Load kernel matrix and year labels
    # ------------------------------------------------------------------
    print('Loading kernel matrix...')
    data  = np.load(kernelfile)
    K_mat = data['kernel']    # (n_years, n_years)
    years = data['years']     # (n_years,)  int

    # ------------------------------------------------------------------
    # 2. Determine training vs projection split from config
    # ------------------------------------------------------------------
    config = load_config(configfile)
    proj_start = config['projection_years']['start']

    train_mask = years < proj_start
    proj_mask  = ~train_mask

    train_years = years[train_mask]
    proj_years  = years[proj_mask]

    print(f'Training years : {train_years[0]}–{train_years[-1]} ({len(train_years)} years)')
    print(f'Projection years: {proj_years[0]}–{proj_years[-1]} ({len(proj_years)} years)')

    # ------------------------------------------------------------------
    # 3. Convert kernel → squared distance matrix
    # ------------------------------------------------------------------
    D_sq = kernel_to_sq_distance(K_mat)

    # Reorder so training years come first (they should already be, but be safe)
    order = np.concatenate([np.where(train_mask)[0], np.where(proj_mask)[0]])
    D_sq  = D_sq[np.ix_(order, order)]

    # ------------------------------------------------------------------
    # 4. Produce plots
    # ------------------------------------------------------------------
    output_path = Path(outputfile)
    figs = []

    if method in ('mds', 'both'):
        print('Computing MDS embedding...')
        figs.append(plot_mds(D_sq, train_years, proj_years))

    if method in ('isomap', 'both'):
        print('Computing Isomap embedding...')
        figs.append(plot_isomap(D_sq, train_years, proj_years, n_neighbors=n_neighbors))

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    suffix = output_path.suffix.lower()
    if suffix == '.pdf' or suffix == '':
        save_path = output_path.with_suffix('.pdf')
        with PdfPages(save_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
        print(f'Saved {len(figs)} page(s) to {save_path}')
    else:
        # For image formats, save one file per method
        if len(figs) == 1:
            figs[0].savefig(output_path, bbox_inches='tight', dpi=DPI)
            print(f'Saved to {output_path}')
        else:
            for i, fig in enumerate(figs):
                stem = output_path.stem
                p = output_path.with_name(f'{stem}_{i+1}{output_path.suffix}')
                fig.savefig(p, bbox_inches='tight', dpi=DPI)
                print(f'Saved to {p}')

    print('Done')


if __name__ == '__main__':
    main()
