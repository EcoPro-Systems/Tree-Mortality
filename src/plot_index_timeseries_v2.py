#!/usr/bin/env python
"""
plot_index_timeseries.py

Plots a multi-page PDF of spatial-mean timeseries for each feature listed in a
YAML config file.  Each page shows:
  - Historical data (solid black line) covering all available years in CLIMATEFILE
  - Two projection scenarios (RCP 4.5, RCP 8.5) covering all years in the
    projection zarrs, so that the overlap with the historical period is visible.

Features are resolved using the same get_feature() pattern as
feature_embeddings_kernel.py (e.g. "SPEI5-6" → base variable "SPEI5", lagged
6 years back).  Spatial averaging is restricted to pixels inside the fir-tree
region defined by FIRFILE.

Usage:
    python plot_index_timeseries.py \\
        CLIMATEFILE PROJFILE1 PROJFILE2 FIRFILE CONFIGFILE OUTPUTFILE

Arguments:
    CLIMATEFILE    Zarr store containing historical climate variables
    PROJFILE1      Zarr store for projection scenario 1 (e.g. RCP 4.5)
    PROJFILE2      Zarr store for projection scenario 2 (e.g. RCP 8.5)
    FIRFILE        Vector file (GeoJSON/Shapefile) defining the fir-tree mask
    CONFIGFILE     YAML config file with at minimum a "features" list and
                   "projection_years": {start, end}
    OUTPUTFILE     Output PDF path
"""

import re
import click
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  – required for .rio accessor
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(configfile):
    with open(configfile) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature helpers  (identical to feature_embeddings_kernel.py)
# ---------------------------------------------------------------------------

def get_feature(merged, year, feature_name):
    """
    Extract a 2-D (northing × easting) numpy array for *feature_name* at *year*
    from a merged climate/topo dataset.

    Handles three feature name patterns:
      - "<BASE><cumulative>-<back>"  e.g. "SPEI5-6": select variable "SPEI5"
        from year (year - back).
      - A bare variable name present as a key in *merged*.
      - The special names "northings" / "eastings".
    """
    cols = len(merged.easting.values)
    rows = len(merged.northing.values)

    match = re.match(r'([A-Za-z]+\d+)-(\d+)', feature_name)
    if match is not None:
        fname = match.group(1)       # e.g. "SPEI5"
        back  = int(match.group(2))  # e.g. 6
        cyear = merged.sel(year=(year - back)).assign_coords(year=[year])
        return cyear[fname].values   # (northing, easting)

    elif feature_name in merged:
        return merged[feature_name].values

    elif feature_name == 'northings':
        return (np.zeros((rows, cols), dtype=float).T + merged.northing.values).T

    elif feature_name == 'eastings':
        return np.zeros((rows, cols), dtype=float) + merged.easting.values

    else:
        raise ValueError(f'Unhandled feature "{feature_name}"')


# ---------------------------------------------------------------------------
# Region masking
# ---------------------------------------------------------------------------

def apply_mask(dataset, region_df):
    """Clip *dataset* to the fir-tree region polygon."""
    dataset = dataset.rename({'easting': 'x', 'northing': 'y'})
    dataset = dataset.rio.write_crs(region_df.crs, inplace=True)
    dataset = dataset.rio.clip(region_df.geometry.values, region_df.crs)
    dataset = dataset.rename({'x': 'easting', 'y': 'northing'})
    return dataset


# ---------------------------------------------------------------------------
# Spatial-mean timeseries extraction
# ---------------------------------------------------------------------------

def spatial_mean_timeseries(ds_masked, feature_name, years):
    """
    Return arrays (valid_years, means) for *feature_name* over *years*.
    NaN pixels (outside mask) are excluded from the mean.
    """
    valid_years, means = [], []
    for year in years:
        try:
            arr = get_feature(ds_masked, year, feature_name)
            mean_val = np.nanmean(arr)
            if np.isfinite(mean_val):
                valid_years.append(year)
                means.append(mean_val)
        except Exception as e:
            print(f'  Warning: skipping year {year} for feature {feature_name}: {e}')
    return np.array(valid_years), np.array(means)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('climatefile',  type=click.Path(path_type=Path, exists=True))
@click.argument('projfile1',    type=click.Path(path_type=Path, exists=True))
@click.argument('projfile2',    type=click.Path(path_type=Path, exists=True))
@click.argument('firfile',      type=click.Path(path_type=Path, exists=True))
@click.argument('configfile',   type=click.Path(path_type=Path, exists=True))
@click.argument('outputfile',   type=click.Path(path_type=Path, exists=False))
def main(climatefile, projfile1, projfile2, firfile, configfile, outputfile):
    """
    Plot spatial-mean feature timeseries (historical + two projection scenarios)
    as a multi-page PDF, one page per feature.
    """

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config = load_config(configfile)
    feature_names = config['features']
    proj_year_range = config['projection_years']
    proj_years = list(range(proj_year_range['start'], proj_year_range['end'] + 1))

    # ------------------------------------------------------------------
    # 2. Load fir-tree region mask
    # ------------------------------------------------------------------
    print('Loading fir-tree region mask...')
    region_df = gpd.read_file(firfile)

    # ------------------------------------------------------------------
    # 3. Load and mask datasets
    # ------------------------------------------------------------------
    print('Loading and masking climate datasets...')
    ds_hist  = apply_mask(xr.open_zarr(climatefile), region_df)
    ds_proj1 = apply_mask(xr.open_zarr(projfile1),   region_df)
    ds_proj2 = apply_mask(xr.open_zarr(projfile2),   region_df)

    hist_years = sorted(int(y) for y in np.unique(ds_hist['year'].values))

    # ------------------------------------------------------------------
    # 4. Build PDF — one page per feature
    # ------------------------------------------------------------------
    print(f'Generating PDF with {len(feature_names)} pages...')
    with PdfPages(outputfile) as pdf:
        for feature in tqdm(feature_names, desc='Features'):
            print(f'\n  Processing feature: {feature}')

            hist_yrs,  hist_vals  = spatial_mean_timeseries(ds_hist,  feature, hist_years)
            proj1_yrs, proj1_vals = spatial_mean_timeseries(ds_proj1, feature, proj_years)
            proj2_yrs, proj2_vals = spatial_mean_timeseries(ds_proj2, feature, proj_years)

            fig, ax = plt.subplots(figsize=(16, 5))

            # Determine full x-axis span
            all_years = np.concatenate([hist_yrs, proj1_yrs, proj2_yrs])
            if len(all_years) == 0:
                print(f'  No data for {feature}, skipping.')
                plt.close(fig)
                continue
            x_min, x_max = int(all_years.min()), int(all_years.max())

            # Shade the historical period for context
            hist_end = hist_yrs[-1] if len(hist_yrs) else proj_years[0]
            ax.axvspan(x_min, hist_end, color='lightgrey', alpha=0.35,
                       label='Historical period')

            # Plot projection scenarios first (so historical sits on top)
            if len(proj2_yrs):
                ax.plot(proj2_yrs, proj2_vals, ':', color='red',   lw=1.8,
                        label='RCP 8.5')
            if len(proj1_yrs):
                ax.plot(proj1_yrs, proj1_vals, '--', color='steelblue', lw=1.8,
                        label='RCP 4.5')

            # Historical on top as solid black
            if len(hist_yrs):
                ax.plot(hist_yrs, hist_vals, '-', color='black', lw=2.0,
                        label='Historical')

            # Vertical line marking start of projections
            ax.axvline(proj_years[0], color='grey', lw=0.8, ls='-')

            ax.set_title(f'{feature}', fontsize=20)
            ax.set_xlim(x_min, x_max)
            ax.set_ylabel(feature, fontsize=16)
            ax.set_xlabel('Year', fontsize=16)
            ax.legend(loc='upper right', fontsize=13, ncol=3)
            ax.tick_params(labelsize=13)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f'\nSaved PDF to {outputfile}')


if __name__ == '__main__':
    main()
