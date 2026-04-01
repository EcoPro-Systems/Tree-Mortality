#!/usr/bin/env python
import json
import click
import warnings
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.stats import randint, uniform
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import (
    cross_validate, GroupKFold, RandomizedSearchCV,
)
from sklearn.metrics import balanced_accuracy_score


from train_rf_model_ray import filter_inf
from util import load_config
from feature_selection import target_remap


warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
)


def summarize_scores(scores):
    return {
        'mean': float(np.average(scores)),
        'std': float(np.std(scores)),
        'n': len(scores),
    }


def nested_group_cv_random_search(X, y, groups, param_distributions, n_iter=10,
                                  inner_splits=3, outer_splits=5, random_state=42):
    """
    Perform nested cross-validation using GroupKFold with RandomizedSearchCV on RandomForestRegressor.

    Parameters:
        X (array-like): Feature matrix
        y (array-like): Target vector
        groups (array-like): Group labels for the samples
        param_distributions (dict): Distributions for hyperparameters
        n_iter (int): Number of parameter settings sampled in RandomizedSearchCV
        inner_splits (int): Number of splits for inner CV
        outer_splits (int): Number of splits for outer CV
        random_state (int): Random seed for reproducibility

    Returns:
        outer_scores (list): R² scores for each outer test fold
        best_params (list): Best parameters found in each inner search
    """
    outer_cv = GroupKFold(n_splits=outer_splits, shuffle=True)
    inner_cv = GroupKFold(n_splits=inner_splits, shuffle=True)

    outer_scores = []
    best_params = []
    importances = []
    records = []

    for train_idx, test_idx in tqdm(list(outer_cv.split(X, y, groups=groups)), 'Evaluating'):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups = groups[train_idx]

        rs = RandomizedSearchCV(
            ExtraTreesClassifier(random_state=random_state, bootstrap=True, n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=inner_cv.split(X_train, y_train, groups=train_groups),
            scoring='balanced_accuracy',
            n_jobs=1,
            random_state=random_state,
        )
        rs.fit(X_train, y_train)


        y_pred = rs.predict(X_test)
        outer_scores.append(balanced_accuracy_score(y_test, y_pred, adjusted=True))
        best_params.append(rs.best_params_)
        importances.append(rs.best_estimator_.feature_importances_)

        # record per-sample results
        for idx, yp, yt in zip(test_idx, y_pred, y_test):
            records.append({
                "index": int(idx),
                "y_true": int(yt),
                "y_pred": int(yp),
            })

    # assemble out-of-fold prediction table
    oof_df = (
        pd.DataFrame(records)
        .sort_values("index")
        .reset_index(drop=True)
    )

    return outer_scores, best_params, importances, oof_df


@click.command()
@click.argument('trainingfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('configfile', type=click.Path(
    path_type=Path, exists=True
))
@click.argument('scorefile', type=click.Path(
    path_type=Path, exists=False
))
def main(trainingfile, configfile, scorefile):

    config = load_config(configfile)
    target = config['target']
    remap = config.get('target_remap', None)
    feature_names = config['features']

    ds = xr.open_zarr(trainingfile)

    years = ds['year'].values
    ytrn = ds[target].to_numpy()
    ytrn = target_remap(ytrn, remap)
    Xtrn = ds[feature_names].to_array().T
    Xtrn, ytrn = filter_inf(Xtrn, ytrn)

    good = (ytrn > 0)
    Xtrn = Xtrn[good]
    ytrn = ytrn[good].astype(int)
    years = years[good].astype(int)

    if 'years' in config:
        good = np.isin(years, config['years'])
        Xtrn = Xtrn[good]
        ytrn = ytrn[good].astype(int)
        years = years[good].astype(int)

    u_years = list(np.unique(years))
    k = len(u_years)

    param_distributions = {
        #'n_estimators': randint(50, 150),                # Number of trees
        'max_depth': [None] + list(range(1, 31, 2)),       # Max depth of trees
        #'min_samples_split': randint(2, 11),               # Min samples to split an internal node
        #'min_samples_leaf': randint(1, 11),                # Min samples required at a leaf node
        #'max_features': ['sqrt', 'log2', None] + list(np.linspace(0.1, 1.0, 5)),  # Features to consider at each split
        #'max_leaf_nodes': [None] + list(range(10, 101, 10)),  # Max leaf nodes
        #'min_weight_fraction_leaf': uniform(0.0, 0.5),      # Min weighted fraction of samples at a leaf
        #'ccp_alpha': uniform(0.0, 0.01),                    # Cost-complexity pruning parameter
        #'max_samples': [None] + list(np.linspace(0.5, 1.0, 6)),  # Max samples for bootstrap if enabled
    }

    #k = 3

    n_iter = 10
    random_state = 42
    outer_scores, best_params, importances, oof_df = nested_group_cv_random_search(
        Xtrn, ytrn, years, param_distributions,
        n_iter=n_iter, inner_splits=(k-1), outer_splits=k, random_state=random_state,
    )

    ba_score = balanced_accuracy_score(oof_df['y_true'], oof_df['y_pred'], adjusted=True)
    print(ba_score)

    importances = [
        dict(zip(feature_names, i))
        for i in importances
    ]

    results = {
        'outer_scores_summary': summarize_scores(outer_scores),
        'pooled_score': ba_score,
        'outer_scores': outer_scores,
        'best_params': best_params,
        'importances': importances,
    }


    cv = GroupKFold(n_splits=k, shuffle=True)
    rs = RandomizedSearchCV(
        ExtraTreesClassifier(random_state=random_state, bootstrap=True, n_jobs=-1),
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv.split(Xtrn, ytrn, groups=years),
        scoring='balanced_accuracy',
        n_jobs=1,
        random_state=random_state,
    )
    rs.fit(Xtrn, ytrn)

    results['overall_best_params'] = rs.best_params_
    results['overall_importances'] = dict(
        zip(feature_names, rs.best_estimator_.feature_importances_)
    )

    with open(scorefile, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
