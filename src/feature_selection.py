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


@ray.remote
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
            n_jobs=1,
            random_state=random_state,
        )

        rs.fit(X_sub, y, groups=groups)

        return feat, rs.best_score_


def rank_features_forward_greedy_ray(
    X,
    y,
    groups,
    param_distributions,
    n_iter=10,
    n_folds=5,
    random_state=42,
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
    feature_names = list(X.columns)

    remaining = set(feature_names)
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

        ranked_features.append(best_feat)
        ba_scores.append(best_score)

        print(f"Selected: {best_feat} | BA: {best_score:.4f}")

    # Convert names → indices
    name_to_idx = {name: i for i, name in enumerate(X.columns)}
    ranked_indices = [name_to_idx[f] for f in ranked_features]

    return ranked_indices, ba_scores


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
    feature_names = config['features']
    feature_names = feature_names

    ds = xr.open_zarr(trainingfile)

    years = ds['year'].values
    ytrn = ds[target].as_numpy()
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
        ranked_idx, scores = rank_features_forward_greedy_ray(
            Xtrn, ytrn, years, param_distributions,
            n_iter=16, n_folds=k, random_state=42
        )

    ranked_features = [feature_names[i] for i in ranked_idx]

    results = {
        'ranked_features': ranked_features,
        'scores': scores,
    }
    print(results)

    with open(scorefile, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
