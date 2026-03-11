#!/usr/bin/env python
import ray
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


@ray.remote(num_cpus=16)
def evaluate_feature(
    feat,
    selected,
    X,
    y,
    groups,
    param_distributions,
    n_iter,
    n_folds,
    random_state,
):
    """
    Evaluate one candidate feature in parallel.
    Returns (feature_name, best_score).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="A worker stopped while some jobs were given to the executor.",
            module="joblib.externals.loky.process_executor",
        )


        cv = GroupKFold(n_splits=n_folds)

        base_model = ExtraTreesClassifier(
            random_state=random_state,
            bootstrap=True,
        )

        current_feats = selected + [feat]
        X_sub = X[current_feats]

        rs = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=16,
            random_state=random_state,
        )

        rs.fit(X_sub, y, groups=groups)

        return feat, rs.best_score_


def save_ranks(scorefile, ranked_features, scores):
    if scorefile is None: return

    results = {
        'ranked_features': ranked_features,
        'scores': scores,
    }

    with open(scorefile, 'w') as f:
        json.dump(results, f, indent=2)


def rank_features_forward_greedy_ray(
    X,
    y,
    groups,
    feature_names,
    param_distributions,
    n_iter=10,
    n_folds=5,
    random_state=42,
    scorefile=None,
):
    """
    Ray-parallel greedy forward feature selection.

    Returns
    -------
    ranked_features : list[int]
    ba_scores : list[float]
    """

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    X = pd.DataFrame(X)
    column_names = list(X.columns)
    name_to_idx = {name: i for i, name in enumerate(X.columns)}

    remaining = set(column_names)
    selected = []
    ranked_features = []
    ba_scores = []

    # Put large objects once in Ray object store
    X_ref = ray.put(X)
    y_ref = ray.put(y)
    groups_ref = ray.put(groups)

    while remaining:
        # Launch parallel jobs for all candidate features
        futures = [
            evaluate_feature.remote(
                feat,
                selected,
                X_ref,
                y_ref,
                groups_ref,
                param_distributions,
                n_iter,
                n_folds,
                random_state,
            )
            for feat in remaining
        ]

        # Gather results
        results = ray.get(futures)

        # Find best feature
        best_feat, best_score = max(results, key=lambda x: x[1])

        # Commit feature
        selected.append(best_feat)
        remaining.remove(best_feat)

        best_feat_name = feature_names[name_to_idx[best_feat]]
        ranked_features.append(best_feat_name)
        ba_scores.append(best_score)

        save_ranks(scorefile, ranked_features, ba_scores)

        print(f"Selected: {best_feat_name} | BA: {best_score:.4f}")

    return ranked_features, ba_scores


def target_remap(trn, remap):
    if remap is None:
        return trn

    return np.array([
        remap.get(i, i)
        for i in trn.astype(int)
    ])


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
    feature_names = feature_names

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

    u_years = list(np.unique(years))
    k = len(u_years)

    param_distributions = {
        'max_depth': [None] + list(range(1, 31, 2)),       # Max depth of trees
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        ranked_features, scores = rank_features_forward_greedy_ray(
            Xtrn, ytrn, years, feature_names, param_distributions,
            n_iter=16, n_folds=k, random_state=42,
            scorefile=scorefile,
        )

    save_ranks(scorefile, ranked_features, scores)


if __name__ == '__main__':
    main()
