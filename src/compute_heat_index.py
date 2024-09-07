#!/usr/bin/env python
import click
import warnings
import xarray as xr
from pathlib import Path
from dask.distributed import Client

from util import load_config


@click.command()
@click.argument('inputfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('outputfile', type=click.Path(
    path_type=Path, exists=False
))
def main(inputfile, configfile, outputfile):

    warnings.filterwarnings(
        'ignore', r'All-NaN (slice|axis) encountered'
    )

    config = load_config(configfile)
    reference_period = config['reference_period']
    focal_period = config['focal_period']
    mean_threshold = config['mean_threshold']
    quantile_threshold = config['quantile_threshold']

    client = Client()

    ds = xr.open_zarr(inputfile)
    series = ds['t2m_max']

    ref = series.sel({
        'year': slice(*reference_period)
    }).chunk(dict(year=-1))

    means = ref.groupby('dayofyear').mean(dim='year')

    pct = ref.groupby('dayofyear').quantile(quantile_threshold, dim='year')

    focal = series.sel({ 'year': slice(*focal_period) })

    criteria1 = (focal - means) > mean_threshold
    criteria2 = focal > pct

    combined = (criteria1 & criteria2)

    indexes = combined.groupby('year').mean(dim='dayofyear')

    indexes = indexes.rename('fraction_hot_days')
    indexes = indexes.assign_attrs(long_name='Fraction Hot Days', units='1')

    write_job = indexes.to_zarr(
        outputfile, mode='w', compute=False, consolidated=True
    )

    print(f'Writing data, view progress: {client.dashboard_link}')
    write_job.compute()
    print('Done')


if __name__ == '__main__':
    main()
