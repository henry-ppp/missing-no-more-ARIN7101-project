"""Hide-and-check tuning, using only the incomplete model input (never the evaluation truth).

Each model has two hyperparameters: features tried per split (max_features) and minimum leaf size
(min_samples_leaf). Tuning tries every combination of two values each (imputers.METHODS): the model's default and
one published library default. With so few hyperparameters a small full grid is enough; random search pays off only
when there are many (Bergstra & Bengio, 2012). Earlier runs of these models gained under 0.5% from tuning on a
30,000-row table.

A further share of the visible cells is hidden, every setting fills them in, and the setting with the lowest RMSE on
those cells wins (the default on a tie). The cells the benchmark scores are already hidden in the input, so tuning
never sees them. Tuning can run on a sample of rows to save time on wide tables.
"""
from __future__ import annotations

import itertools

import numpy as np

from .imputers import METHODS
from .seeds import stream

CHECK_SHARE = 0.1     # share of visible cells hidden again for tuning
TUNE_ROWS = 3000


def candidates(method: str, n_columns: int) -> list[dict]:
    """Every grid setting, the default first, skipping any that give the same forest as one already listed."""
    spec = METHODS[method]
    combos = [dict(zip(spec.grid, values)) for values in itertools.product(*spec.grid.values())]
    chosen, seen = [], set()
    for params in [dict(spec.default)] + combos:
        key = _effective(params, n_columns)
        if key not in seen:
            chosen.append(params)
            seen.add(key)
    return chosen


def _effective(params: dict, n_columns: int) -> tuple:
    """Features per split as a count, so 'sqrt' and 1/3 that pick the same number of columns count once."""
    n_predictors = n_columns - 1
    share = params['max_features']
    count = int(np.sqrt(n_predictors)) if share == 'sqrt' else int(share * n_predictors)
    return max(1, count), params['min_samples_leaf']


def tuning_sample(X_missing: np.ndarray, seed: int, n_rows: int | None = TUNE_ROWS) -> np.ndarray:
    """A random sample of rows of the model input (all rows if n_rows is None or not smaller than the table)."""
    if n_rows is None or n_rows >= len(X_missing):
        return X_missing
    rows = np.sort(stream(seed, 'tuning', 'rows').choice(len(X_missing), n_rows, replace=False))
    return X_missing[rows]


def hide_and_check(X_missing: np.ndarray, seed: int, share: float = CHECK_SHARE):
    """Hide a further share of the visible cells. Returns (table with them hidden, mask of the check cells).

    Every row keeps at least one visible value and every column at least two, so the imputers can still run.
    """
    rng = stream(seed, 'tuning', 'check')
    visible = ~np.isnan(X_missing)
    check = visible & (rng.random(X_missing.shape) < share)
    for i in np.flatnonzero(visible.any(axis=1) & ((visible & ~check).sum(axis=1) == 0)):
        check[i, rng.choice(np.flatnonzero(visible[i]))] = False
    for j in np.flatnonzero((visible & ~check).sum(axis=0) < 2):
        check[:, j] = False
    if not check.any():
        raise ValueError('No visible cells to hide for tuning')
    return np.where(check, np.nan, X_missing), check


def score_candidate(method: str, params: dict, X_check: np.ndarray, X_reference: np.ndarray, check: np.ndarray,
                    seed: int, n_trees: int, max_rounds: int) -> dict:
    """Fill the tuning table with one setting and score it on the check cells (RMSE, standardised scale)."""
    filled, info = METHODS[method].impute(X_check, seed, n_trees=n_trees, max_rounds=max_rounds, **params)
    error = filled[check] - X_reference[check]
    return {'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(np.abs(error))),
            'rounds': info.kept_round, 'seconds': info.seconds}
