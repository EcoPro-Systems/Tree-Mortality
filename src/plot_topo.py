#!/usr/bin/env python
import os
import PIL
import json
import click
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Required for large high-res background image
PIL.Image.MAX_IMAGE_PIXELS = 250000000


@click.command()
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(configfile, outputfile):

    with open(configfile, 'r') as f:
        config = json.load(f)

    figsize = config['figsize']
    fontsize = config['fontsize']
    extent = config.get('extent', None)
    clip = config.get('clip', False)
    plot_kwargs = config.get('plot_kwargs', {})
    cbar_kwargs = config.get('cbar_kwargs', {})
    cbar_label = config['cbar_label']
    title_fmt = config.get('title_format', 'Year = {year}')
    background_kwargs = config.get('background_kwargs', None)
    facecolor = config.get('facecolor', None)
    savefig_kwargs = config.get('savefig_kwargs', {})
    nan_to_num_kwargs = config.get('nan_to_num', None)


    fig, ax = plt.subplots(
        1, 1,
        figsize=figsize,
        subplot_kw=dict(projection=ccrs.Mercator())
    )
    if extent is not None: ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    if background_kwargs is not None:
        ax.background_img(**background_kwargs)
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    fig.savefig(outputfile, **savefig_kwargs)


if __name__ == '__main__':
    main()
