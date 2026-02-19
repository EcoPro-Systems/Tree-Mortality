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


def rank_features_forward_greedy(
        X, y, groups, param_distributions, n_iter=10, n_folds=5, random_state=42):
    """
    Greedy forward feature selection with GroupKFold CV.

    Starts with no features and adds the best feature one-by-one.

    Returns
    -------
    ranked_features : list[int]
        Feature indices ordered from most to least important
    r2_scores : list[tuple[int, float]]
        (n_features_selected, mean CV R²) at each step
    """
    X = pd.DataFrame(X)
    n_features = X.shape[1]
    feature_names = list(X.columns)

    cv = GroupKFold(n_splits=n_folds)

    base_model = ExtraTreesClassifier(random_state=random_state, bootstrap=True)

    remaining = set(feature_names)
    selected = []
    ranked_features = []
    ba_scores = []

    while remaining:
        best_feat = None
        best_score = -np.inf

        for feat in tqdm(remaining):
            current_feats = selected + [feat]
            X_sub = X[current_feats]

            rs = RandomizedSearchCV(
                base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="balanced_accuracy",
                cv=cv,
                n_jobs=-1,
                random_state=random_state,
            )
            rs.fit(X_sub, y, groups=groups)
            score = rs.best_score_

            if score > best_score:
                best_score = score
                best_feat = feat

        # Commit the best feature
        selected.append(best_feat)
        remaining.remove(best_feat)

        ranked_features.append(best_feat)
        ba_scores.append(best_score)

    # Convert feature names → indices
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
        ranked_idx, scores = rank_features_forward_greedy(
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
