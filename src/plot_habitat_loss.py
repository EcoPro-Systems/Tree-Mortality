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
@click.argument('boundaryfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('year', type=int)
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(datafile, boundaryfile, year, configfile, outputfile):

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

    df = gpd.read_file(boundaryfile)
    df = df.to_crs(plot_crs)

    ds = xr.open_zarr(datafile)
    try:
        ds_crs = ccrs.Projection(ds.rio.crs)
    except:
        ds_crs = ccrs.PlateCarree()

    var = 100 * ds[value].sel(year=year)

    # Clip infinite values if needed
    if nan_to_num_kwargs is not None:
        var = xr.apply_ufunc(
            np.nan_to_num, var.load(),
            kwargs=nan_to_num_kwargs,
        )

    fig = plt.figure(figsize=figsize)
    ax_full = plt.subplot(2, 2, (2, 4), projection=plot_crs)
    ax_y = plt.subplot(2, 2, 1, projection=plot_crs)
    ax_s = plt.subplot(2, 2, 3, projection=plot_crs)
    if extent is not None: ax_full.set_extent(extent, crs=ccrs.PlateCarree())

    artist = var.plot(
        ax=ax_full,
        transform=ds_crs,
        add_colorbar=False,
        **plot_kwargs
    )
    title = title_fmt.format(year=year)
    fig.suptitle(title, fontsize=fontsize['title'])
    ax_full.set_title(None)

    plot_cutout(
        ax_y, ds_crs, plot_crs, plot_kwargs,
        var, df, [-120.4, -118.5, 37.35, 38.3],
        ('Yosemite',), ax_full
    )

    plot_cutout(
        ax_s, ds_crs, plot_crs, plot_kwargs,
        var, df, [-119.9, -117.6, 36.1, 37.3],
        ('Kings Canyon', 'Sequoia'), ax_full
    )

    if background_kwargs is not None:
        ax_full.background_img(**background_kwargs)

    for ax in (ax_full, ax_s):
        ax.coastlines()
        ax.add_feature(cfeature.STATES.with_scale('50m'))
        if facecolor is not None:
            ax.set_facecolor(facecolor)

    cbar = fig.colorbar(artist, **cbar_kwargs)
    cbar.set_label(**cbar_label)

    plt.tight_layout()

    fig.savefig(outputfile, **savefig_kwargs)


if __name__ == '__main__':
    main()
