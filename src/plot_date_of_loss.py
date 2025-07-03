#!/usr/bin/env python
import os
import PIL
import json
import click
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


# Required for large high-res background image
PIL.Image.MAX_IMAGE_PIXELS = 250000000


def plot_cutout(ax, ds_crs, plot_crs, plot_kwargs, var, df, extent, names, ax_full):
    df_sub = df.loc[df['PARKNAME'].isin(names)]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    var.plot(
        ax=ax,
        transform=ds_crs,
        add_colorbar=False,
        **plot_kwargs
    )
    ax.add_geometries(
        df_sub['geometry'], crs=plot_crs,
        facecolor='none', edgecolor='black', lw=2
    )
    ax.background_img(
        name='ne2_gray',
        resolution='high',
        extent=extent
    )
    ax.set_title(', '.join(names), fontsize=18)

    ax_full.add_geometries(
        df_sub['geometry'], crs=plot_crs,
        facecolor='none', edgecolor='black', lw=1
    )
    ax_full.add_patch(
        Rectangle(
            xy=[extent[0], extent[2]],
            width=(extent[1] - extent[0]),
            height=(extent[3] - extent[2]),
            facecolor='none', edgecolor='k', lw=2,
            transform=ccrs.PlateCarree()
        )
    )


@click.command()
@click.argument('datafile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
@click.option('-f', '--fraction', default=0.1, type=float)
def main(datafile, configfile, outputfile, fraction):

    with open(configfile, 'r') as f:
        config = json.load(f)

    value = config['value']
    figsize = config['figsize']
    fontsize = config['fontsize']
    extent = config.get('extent', None)
    plot_kwargs = config.get('plot_kwargs', {})
    cbar_kwargs = config.get('cbar_kwargs', {})
    cbar_label = config['cbar_label']
    title_fmt = config.get('title_format', 'Year = {year}')
    background_kwargs = config.get('background_kwargs', None)
    facecolor = config.get('facecolor', None)
    savefig_kwargs = config.get('savefig_kwargs', {})
    nan_to_num_kwargs = config.get('nan_to_num', None)

    plot_crs = ccrs.Mercator()

    ds = xr.open_zarr(datafile)
    try:
        ds_crs = ccrs.Projection(ds.rio.crs)
    except:
        ds_crs = ccrs.PlateCarree()

    var = ds[value]
    critical = (var <= fraction).astype(int)
    first_idx = critical.argmax(dim="year").compute()
    first_year = var.year[first_idx]
    mask = critical.any(dim="year")
    first_year = xr.where(mask, first_year, np.max(var.year.values))
    mask = ~var.isnull().any(dim="year")
    first_year = xr.where(mask, first_year, np.nan)
    var = first_year.compute()

    # Clip infinite values if needed
    if nan_to_num_kwargs is not None:
        var = xr.apply_ufunc(
            np.nan_to_num, var.load(),
            kwargs=nan_to_num_kwargs,
        )

    fig, ax = plt.subplots(
        1, 1,
        figsize=figsize,
        subplot_kw=dict(projection=ccrs.Mercator())
    )
    if extent is not None: ax.set_extent(extent, crs=ccrs.PlateCarree())
    artist = var.plot(
        ax=ax,
        transform=ds_crs,
        add_colorbar=False,
        **plot_kwargs
    )
    title = title_fmt.format(fraction=fraction, pc=100*fraction)
    ax.set_title(title, fontsize=fontsize['title'])
    ax.coastlines()
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    if background_kwargs is not None:
        ax.background_img(**background_kwargs)
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    cbar = fig.colorbar(artist, **cbar_kwargs)
    cbar.set_label(**cbar_label)

    fig.savefig(outputfile, **savefig_kwargs)


if __name__ == '__main__':
    main()
