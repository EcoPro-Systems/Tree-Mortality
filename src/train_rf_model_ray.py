#!/usr/bin/env python
import click
import ray
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import product
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score


def filter_inf(X, y=None, fill=1e2):
    if np.any(np.isnan(X)):
        raise ValueError('NaN entries present but not handled')
    Xnew = np.nan_to_num(X, neginf=-fill, posinf=fill)
    if y is None:
        return Xnew
    else:
        return Xnew, y


@ray.remote
def eval_fold(trainingfile, held_out_fold, training_year):
    ds = xr.open_zarr(trainingfile)

    idx = ((ds['year'] == training_year) & (ds['fold'] != held_out_fold)).compute()
    ds_trn= ds.where(idx, drop=True).compute()

    ytrn = ds_trn['tpa'].as_numpy()
    features = ds_trn.drop_vars(
        ('id', 'fold', 'easting', 'northing', 'year', 'tpa')
    )
    feature_names = list(features.keys())
    Xtrn = features.to_array().T
    Xtrn, ytrn = filter_inf(Xtrn, ytrn)

    rf = RandomForestRegressor(max_depth=5)
    rf.fit(Xtrn, ytrn)

    importances = dict(zip(feature_names, rf.feature_importances_))

    idx = (ds['fold'] == held_out_fold).compute()
    ds_tst = ds.where(idx, drop=True).compute()

    ids_tst = ds_tst['id'].as_numpy()
    years_tst = ds_tst['year'].as_numpy()
    ytst = ds_tst['tpa'].as_numpy()
    Xtst = ds_tst.drop_vars(
        ('id', 'fold', 'easting', 'northing', 'year', 'tpa')
    ).to_array().T
    Xtst, ytst = filter_inf(Xtst, ytst)

    ypred = rf.predict(Xtst)

    year_unq = np.unique(years_tst)
    id_unq = np.unique(ids_tst)
    year_idx = np.searchsorted(year_unq, years_tst)
    id_idx = np.searchsorted(id_unq, ids_tst)

    predictions = np.zeros((len(year_unq), len(id_unq)))
    predictions[year_idx, id_idx] = ypred

    targets = np.zeros((len(year_unq), len(id_unq)))
    targets[year_idx, id_idx] = ytst

    return {
        'fold': held_out_fold,
        'train_year': training_year,
        'years': year_unq,
        'ids': id_unq,
        'predictions': predictions,
        'targets': targets,
        'importances': importances,
    }



@click.command()
@click.argument('trainingfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('resultfile', type=click.Path(
    path_type=Path, exists=False
))
def main(trainingfile, resultfile):

    ray.init()

    ds = xr.open_zarr(trainingfile)
    years = np.unique(ds['year']).astype(int)
    folds = np.unique(ds['fold']).astype(int)
    ids = np.unique(ds['id']).astype(int)

    tasks = []
    for year, fold in product(years, folds):
        task = eval_fold.remote(trainingfile, fold, year)
        tasks.append(task)

    results = ray.get(tasks)

    feature_names = sorted(results[0]['importances'].keys())

    predictions = np.zeros((len(years), len(years), len(ids)))
    targets = np.zeros((len(years), len(years), len(ids)))
    importances = np.zeros((len(years), len(folds), len(feature_names)))

    for r in tqdm(results, 'Merging results'):
        tyear_idx = np.searchsorted(years, r['train_year'])
        eyear_idx = np.searchsorted(years, r['years'])
        id_idx = np.searchsorted(ids, r['ids'])
        for y, p, t in zip(eyear_idx, r['predictions'], r['targets']):
            predictions[tyear_idx, y, id_idx] = p
            targets[tyear_idx, y, id_idx] = t

        fold_idx = np.searchsorted(folds, r['fold'])
        importances[tyear_idx, fold_idx, :] = np.array(
            [r['importances'][n] for n in feature_names]
        )

    out = {
        'ids': ids,
        'years': years,
        'predictions': predictions,
        'targets': targets,
        'importances': importances,
        'feature_names': np.asarray(feature_names),
        'folds': np.asarray(folds),
    }

    np.savez_compressed(resultfile, **out)


if __name__ == '__main__':
    main()
