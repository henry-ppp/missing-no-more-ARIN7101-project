"""MissForest and MICE with random forests (MICE-RF) for numeric tables with NaN gaps.

Both take a NumPy table, return a filled copy that keeps every visible value, and record what happened while
fitting: a training curve (the change between rounds) and each column's feature importances.

- MissForest (Stekhoven & Buehlmann, 2012), our own implementation: start from the column means, then repeatedly
  predict each column's gaps with a random forest trained on the other columns, fewest gaps first. Stop when the
  change between rounds grows for the first time and keep the round before, as in the original paper.
- MICE-RF: scikit-learn's IterativeImputer (chained equations, van Buuren & Groothuis-Oudshoorn, 2011) with
  extremely randomised trees (Geurts, Ernst & Wehenkel, 2006) as the per-column model. LeanIterativeImputer drops
  each forest once it has been used: IterativeImputer keeps every forest it fits (about 20 GB on a 30,000-row
  table), and we never transform new data, so the output is identical.

The change between rounds is the MissForest criterion for both models, so their curves are comparable:
sum over gap cells of (new - previous)^2 / sum over gap cells of new^2.

The shared tables have no category columns, so there is no classification branch.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (makes IterativeImputer importable)
from sklearn.impute import IterativeImputer

from .seeds import derive_seed

N_TREES = 100
MAX_ROUNDS = 10


@dataclass
class FitInfo:
    """What one imputation run did. history: one dict per round (round, change, seconds)."""
    history: list = field(default_factory=list)
    kept_round: int = 0              # the round whose imputation is returned
    stopped_early: bool = False      # True if the stopping rule ended the run before max_rounds
    importances: np.ndarray | None = None   # columns x columns; row j = importance of each column for column j
    seconds: float = 0.0


def change(new: np.ndarray, previous: np.ndarray, gaps: np.ndarray) -> float:
    """The MissForest change between two rounds, over the gap cells."""
    return float(((new - previous)[gaps] ** 2).sum() / max((new[gaps] ** 2).sum(), 1e-300))


def keep_visible(X_missing: np.ndarray, X_filled: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X_missing), X_filled, X_missing)


def _check(X_missing: np.ndarray) -> np.ndarray:
    X = np.asarray(X_missing, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError('Need a 2-D table with at least two columns')
    if np.isinf(X).any():
        raise ValueError('Infinite values in the input')
    if (~np.isnan(X)).sum(axis=0).min() < 2:
        raise ValueError('Every column needs at least two visible values')
    return X


def impute_missforest(X_missing, seed: int, max_features='sqrt', min_samples_leaf: int = 1, n_trees: int = N_TREES,
                      max_rounds: int = MAX_ROUNDS) -> tuple[np.ndarray, FitInfo]:
    X_missing = _check(X_missing)
    start = time.perf_counter()
    gaps = np.isnan(X_missing)
    X = np.where(gaps, np.nanmean(X_missing, axis=0), X_missing)
    order = [j for j in np.argsort(gaps.sum(axis=0), kind='stable') if gaps[:, j].any()]   # fewest gaps first
    info = FitInfo(importances=np.zeros((X.shape[1], X.shape[1])))
    previous_change, previous_importances = np.inf, info.importances.copy()
    for round_ in range(1, max_rounds + 1):
        round_start, X_old = time.perf_counter(), X.copy()
        importances = np.zeros_like(info.importances)
        for j in order:
            visible, missing = ~gaps[:, j], gaps[:, j]
            others = np.delete(X, j, axis=1)
            forest = RandomForestRegressor(n_estimators=n_trees, max_features=max_features,
                                           min_samples_leaf=min_samples_leaf, n_jobs=1,
                                           random_state=derive_seed(seed, 'MissForest', 'forest', round_, int(j)))
            forest.fit(others[visible], X[visible, j])
            X[missing, j] = forest.predict(others[missing])
            importances[j, np.arange(X.shape[1]) != j] = forest.feature_importances_
        step = change(X, X_old, gaps)
        info.history.append({'round': round_, 'change': step, 'seconds': time.perf_counter() - round_start})
        if round_ > 1 and step >= previous_change:
            # the change grew: keep the previous round, as Stekhoven & Buehlmann do
            info.kept_round, info.stopped_early, info.importances = round_ - 1, True, previous_importances
            info.seconds = time.perf_counter() - start
            return keep_visible(X_missing, X_old), info
        previous_change, previous_importances = step, importances
    info.kept_round, info.importances = max_rounds, previous_importances
    info.seconds = time.perf_counter() - start
    return keep_visible(X_missing, X), info


class LeanIterativeImputer(IterativeImputer):
    """IterativeImputer that drops each fitted forest after use and records the change after every round."""

    def fit_transform(self, X, y=None, **params):
        X = np.asarray(X, dtype=float)
        self._gaps = np.isnan(X)
        self._per_round = int(self._gaps.any(axis=0).sum()) if self.skip_complete else X.shape[1]
        self._calls, self._round_start, self._clock = 0, None, time.perf_counter()
        self.history_, self.importances_ = [], np.zeros((X.shape[1], X.shape[1]))
        return super().fit_transform(X, y, **params)

    def _impute_one_feature(self, X_filled, mask_missing_values, feat_idx, neighbor_feat_idx, *args, **kwargs):
        if self._calls % self._per_round == 0:
            self._round_start = X_filled.copy()
        X_filled, estimator = super()._impute_one_feature(X_filled, mask_missing_values, feat_idx, neighbor_feat_idx,
                                                          *args, **kwargs)
        if estimator is not None and hasattr(estimator, 'feature_importances_'):
            row = np.zeros(X_filled.shape[1])
            row[neighbor_feat_idx] = estimator.feature_importances_
            self.importances_[feat_idx] = row
        self._calls += 1
        if self._calls % self._per_round == 0:
            now = time.perf_counter()
            self.history_.append({'round': len(self.history_) + 1,
                                  'change': change(X_filled, self._round_start, self._gaps),
                                  'seconds': now - self._clock})
            self._clock = now
        return X_filled, None


def impute_mice_rf(X_missing, seed: int, max_features=0.5, min_samples_leaf: int = 2, n_trees: int = N_TREES,
                   max_rounds: int = MAX_ROUNDS) -> tuple[np.ndarray, FitInfo]:
    X_missing = _check(X_missing)
    start = time.perf_counter()
    trees = ExtraTreesRegressor(n_estimators=n_trees, max_features=max_features, min_samples_leaf=min_samples_leaf,
                                n_jobs=1, random_state=derive_seed(seed, 'MICE-RF', 'forest'))
    mice = LeanIterativeImputer(estimator=trees, max_iter=max_rounds, skip_complete=True,
                                n_nearest_features=min(40, X_missing.shape[1] - 1),
                                random_state=derive_seed(seed, 'MICE-RF', 'order'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)   # recorded below as stopped_early = False
        X = mice.fit_transform(X_missing)
    info = FitInfo(history=mice.history_, kept_round=int(mice.n_iter_), stopped_early=mice.n_iter_ < max_rounds,
                   importances=mice.importances_, seconds=time.perf_counter() - start)
    return keep_visible(X_missing, X), info


# ---- the models the experiment knows: default settings and the grids tuning tries ----

@dataclass(frozen=True)
class Method:
    impute: object
    default: dict
    grid: dict      # every combination is tried; each hyperparameter: our default and one library default


METHODS = {
    # alternatives: the regression defaults of R's randomForest (p/3 features per split, node size 5)
    'MissForest': Method(impute_missforest, {'max_features': 'sqrt', 'min_samples_leaf': 1},
                         {'max_features': ['sqrt', 1 / 3], 'min_samples_leaf': [1, 5]}),
    # alternatives: scikit-learn's ExtraTreesRegressor defaults (all features, leaf size 1)
    'MICE-RF': Method(impute_mice_rf, {'max_features': 0.5, 'min_samples_leaf': 2},
                      {'max_features': [0.5, 1.0], 'min_samples_leaf': [2, 1]}),
}


class _DataFrameImputer:
    """fit_transform for a DataFrame with NaN gaps; keeps its index and columns. info_ holds the FitInfo."""
    name = None

    def __init__(self, seed: int = 0, n_trees: int = N_TREES, max_rounds: int = MAX_ROUNDS, **params):
        self.seed, self.n_trees, self.max_rounds = seed, n_trees, max_rounds
        self.params = {**METHODS[self.name].default, **params}
        self.info_ = None

    def fit_transform(self, X):
        import pandas as pd
        filled, self.info_ = METHODS[self.name].impute(X.to_numpy(dtype=float), self.seed, n_trees=self.n_trees,
                                                       max_rounds=self.max_rounds, **self.params)
        return pd.DataFrame(filled, index=X.index, columns=X.columns)


class MissForestImputer(_DataFrameImputer):
    name = 'MissForest'


class MiceRFImputer(_DataFrameImputer):
    name = 'MICE-RF'
