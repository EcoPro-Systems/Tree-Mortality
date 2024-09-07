#!/usr/bin/env python
import click
import numpy as np
import xarray as xr
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from skops.io import dump


from train_rf_model_ray import filter_inf
from util import load_config


MIDPOINTS = np.array([np.nan, 2.0, 6.5, 20.0, 40.0, 75.0])


def filter_zero(X, y):
    good = (y > 0)
    return X[good], y[good]


@click.command()
@click.argument('trainingfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('modelfile', type=click.Path(
    path_type=Path, exists=False
))
def main(trainingfile, configfile, modelfile):

    config = load_config(configfile)
    target = config['target']
    feature_names = config['features']

    ds = xr.open_zarr(trainingfile)

    ytrn = ds[target].as_numpy()
    Xtrn = ds[feature_names].to_array().T
    Xtrn, ytrn = filter_inf(Xtrn, ytrn)
    Xtrn, ytrn = filter_zero(Xtrn, ytrn)

    # Categorical to regresison targets
    ytrn = MIDPOINTS[ytrn.astype(int)]

    model = ExtraTreesRegressor(
        n_estimators=100, max_depth=5,
        bootstrap=True, n_jobs=-1,
    )
    model.fit(Xtrn, ytrn)

    output = {
        'model': model,
        'features': feature_names,
    }

    with open(modelfile, 'wb') as f:
        dump(output, f)


if __name__ == '__main__':
    main()
