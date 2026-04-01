#!/usr/bin/env python
"""
compute_year_kernel_matrix.py

Computes a pairwise RBF kernel matrix between per-year feature distributions,
spanning both historical training years (e.g. 2012-2023) and future projection
years (e.g. 2024-2100). The RBF bandwidth is estimated from training data only.
The resulting kernel matrix and year labels are saved to a .npz file.

Usage:
    python compute_year_kernel_matrix.py \\
        TRAININGFILE CLIMATEFILE PROJECTIONFILE FIRFILE CONFIGFILE OUTPUTFILE

Arguments:
    TRAININGFILE    Zarr store containing the training feature dataset
                    (as produced by construct_fir_feature_dataset.py)
    CLIMATEFILE     Zarr store containing historical climate variables
                    (used to construct projection features via get_feature())
    PROJECTIONFILE  Zarr store containing projected climate variables
                    (same structure as CLIMATEFILE but covering future years)
    FIRFILE         Vector file (e.g. GeoJSON/Shapefile) defining the fir-tree
                    region mask -- same file used in construct_fir_feature_dataset.py
    CONFIGFILE      JSON config file.  Must contain at minimum:
                        "features"        : list of feature name strings
                        "projection_years": {"start": int, "end": int}  (end exclusive)
                    The "target" key is read but not used for feature extraction.
    OUTPUTFILE      Path for the output .npz file.  Saved arrays:
                        kernel  -- float64 (n_years, n_years) kernel matrix
                        years   -- int     (n_years,)          corresponding year labels
"""

import re
import json
import click
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

from util import load_config


# ---------------------------------------------------------------------------
# Feature helpers (adapted from apply_rf_model.py)
# ---------------------------------------------------------------------------

def get_feature(merged, year, feature_name):
    """
    Extract a 2-D (northing × easting) numpy array for *feature_name* at *year*
    from a merged climate/topo dataset.

    Handles three feature name patterns:
      - "<BASE><cumulative>-<back>"  e.g. "PREC3-1": cumulative variable from
        year (year - back), selecting the cumulative variant of BASE.
      - A bare variable name present as a key in *merged*.
      - The special names "northings" and "eastings".
    """
    cols = len(merged.easting.values)
    rows = len(merged.northing.values)

    match = re.match(r'([A-Z]+)(\d+)-(\d+)', feature_name)
    if match is not None:
        fbase = match.group(1)
        cumulative = int(match.group(2))
        back = int(match.group(3))
        fname = f'{fbase}{cumulative}'
        cyear = merged.sel(year=(year - back)).assign_coords(year=[year])
        return cyear[fname].to_numpy()

    elif feature_name in merged:
        return merged[feature_name].to_numpy()

    elif feature_name == 'northings':
        northings = (np.zeros((rows, cols), dtype=float).T + merged.northing.values).T
        return northings

    elif feature_name == 'eastings':
        eastings = np.zeros((rows, cols), dtype=float) + merged.easting.values
        return eastings

    else:
        raise ValueError(f'Unhandled feature "{feature_name}"')


# ---------------------------------------------------------------------------
# Region masking (from construct_fir_feature_dataset.py)
# ---------------------------------------------------------------------------

def apply_mask(dataset, region_df):
    """Clip *dataset* to the fir-tree region polygon."""
    dataset = dataset.rename({'easting': 'x', 'northing': 'y'})
    dataset = dataset.rio.clip(
        region_df.geometry.values, region_df.crs
    )
    dataset = dataset.rename({'x': 'easting', 'y': 'northing'})
    return dataset


# ---------------------------------------------------------------------------
# Build a flat feature matrix for one year from projection data
# ---------------------------------------------------------------------------

def extract_projection_features(merged_masked, year, feature_names):
    """
    Return a 2-D float array (n_valid_cells, n_features) for *year* from the
    already-masked projection dataset.  Cells where any feature is NaN are
    dropped, mirroring the behaviour in construct_fir_feature_dataset.py.
    """
    cube = np.dstack([
        get_feature(merged_masked, year, f)
        for f in feature_names
    ])
    X = cube.reshape((-1, len(feature_names)))
    good = ~np.any(np.isnan(X), axis=1)
    return X[good]


# ---------------------------------------------------------------------------
# Kernel helpers (from feature_embeddings_kernel.py)
# ---------------------------------------------------------------------------

def filter_inf(X, fill=4):
    """Replace ±inf with ±fill; raise if NaNs are present."""
    if np.any(np.isnan(X)):
        raise ValueError('NaN entries present in feature matrix')
    return np.nan_to_num(X, neginf=-fill, posinf=fill)


def rbf_gamma_median_heuristic(X, subsample=None):
    """Estimate RBF gamma via the median-of-squared-distances heuristic."""
    if subsample is not None and subsample < X.shape[0]:
        idx = np.random.choice(X.shape[0], subsample, replace=False)
        X = X[idx]
    dists = pairwise_distances(X, metric='sqeuclidean')
    median_sq_dist = np.median(dists[dists > 0])
    return 1.0 / (2.0 * median_sq_dist)


def K(X, Y, gamma, subsample=100):
    """Mean RBF kernel value between (subsampled) rows of X and Y."""
    k = rbf_kernel(X[::subsample], Y[::subsample], gamma)
    return np.average(k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('trainingfile', type=click.Path(path_type=Path, exists=True))
@click.argument('projectionfile', type=click.Path(path_type=Path, exists=True))
@click.argument('firfile',      type=click.Path(path_type=Path, exists=True))
@click.argument('configfile',   type=click.Path(path_type=Path, exists=True))
@click.argument('outputfile',   type=click.Path(path_type=Path, exists=False))
def main(trainingfile, projectionfile, firfile, configfile, outputfile):
    """
    Compute a pairwise RBF kernel matrix across training + projection years
    and save the result to OUTPUTFILE (.npz).
    """

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config = load_config(configfile)
    feature_names = config['features']
    proj_year_range = config['projection_years']
    proj_years = list(range(proj_year_range['start'], proj_year_range['end']))

    # ------------------------------------------------------------------
    # 2. Load fir-tree region mask
    # ------------------------------------------------------------------
    print('Loading fir-tree region mask...')
    region_df = gpd.read_file(firfile)

    # ------------------------------------------------------------------
    # 3. Extract per-year feature matrices from the training zarr
    #    (already masked and stacked by construct_fir_feature_dataset.py)
    # ------------------------------------------------------------------
    print('Loading training features...')
    ds_train = xr.open_zarr(trainingfile)

    train_years_all = ds_train['year'].values.astype(int)
    u_train_years = sorted(np.unique(train_years_all).tolist())

    # Build feature array names list from dataset variables, preserving order
    Xtrn_full = ds_train[feature_names].to_array().values.T   # (n_samples, n_features)
    Xtrn_full = filter_inf(Xtrn_full)

    # Group training samples by year
    train_Xs = {
        y: Xtrn_full[train_years_all == y]
        for y in u_train_years
    }

    # ------------------------------------------------------------------
    # 4. Load projection data, apply fir mask, extract per-year features
    # ------------------------------------------------------------------
    print('Loading and masking projection climate data...')
    proj = xr.open_zarr(projectionfile)

    # Merge climate + projection so that lagged features (e.g. PREC3-1)
    # that straddle the training/projection boundary can be resolved.
    # Inner join keeps only the spatial overlap between the two datasets.
    merged = proj

    # Apply the same fir-tree mask used when building the training set
    merged_masked = apply_mask(merged, region_df)

    print(f'Extracting projection features for {len(proj_years)} years...')
    proj_Xs = {}
    for year in tqdm(proj_years, desc='Projection years'):
        try:
            X = extract_projection_features(merged_masked, year, feature_names)
            X = filter_inf(X)
            proj_Xs[year] = X
        except Exception as e:
            print(f'  Warning: skipping year {year} ({e})')

    valid_proj_years = sorted(proj_Xs.keys())

    # ------------------------------------------------------------------
    # 5. Fit scaler on training data only, then apply to all years
    # ------------------------------------------------------------------
    print('Fitting StandardScaler on training data...')
    X_train_concat = np.vstack(list(train_Xs.values()))
    scaler = StandardScaler()
    scaler.fit(X_train_concat)

    train_Xs_scaled = {y: scaler.transform(X) for y, X in train_Xs.items()}
    proj_Xs_scaled  = {y: scaler.transform(X) for y, X in proj_Xs.items()}

    train_Xs_scaled_concat = np.vstack(list(train_Xs_scaled.values()))

    # ------------------------------------------------------------------
    # 6. Estimate RBF gamma from training data only
    # ------------------------------------------------------------------
    print('Estimating RBF bandwidth from training data...')
    gamma = rbf_gamma_median_heuristic(train_Xs_scaled_concat, subsample=10000)
    print(f'  gamma = {gamma:.6g}')

    # ------------------------------------------------------------------
    # 7. Compute pairwise kernel matrix across ALL years
    # ------------------------------------------------------------------
    all_years = u_train_years + valid_proj_years
    all_Xs = [
        train_Xs_scaled[y] if y in train_Xs_scaled else proj_Xs_scaled[y]
        for y in all_years
    ]

    n = len(all_years)
    print(f'Computing {n}×{n} kernel matrix across {n} years...')
    kernel_matrix = np.zeros((n, n), dtype=np.float64)

    for i in tqdm(range(n), desc='Kernel rows'):
        for j in range(i, n):
            val = K(all_Xs[i], all_Xs[j], gamma)
            kernel_matrix[i, j] = val
            kernel_matrix[j, i] = val   # symmetric

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    years_arr = np.array(all_years, dtype=int)
    np.savez(outputfile, kernel=kernel_matrix, years=years_arr)
    print(f'Saved kernel matrix ({n}×{n}) and year labels to {outputfile}')
    print('Done')


if __name__ == '__main__':
    main()
