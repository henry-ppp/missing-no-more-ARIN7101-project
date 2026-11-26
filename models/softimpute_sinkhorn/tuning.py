"""Tuning SoftImpute's lambda by hide-and-check. Sinkhorn has no tuned hyperparameters (the paper fixes them).

Hide-and-check is the random-forest package's rule, with the same seed: a further 10% of the visible cells of the
model input are hidden, every candidate fills them, and the lowest RMSE on them wins. Only the incomplete model
input is used; the cells the benchmark scores stay hidden and the evaluation truth is never opened.
"""
from __future__ import annotations

import time

import numpy as np

from models.random_forest.tuning import CHECK_SHARE, hide_and_check

from .imputers import SOFT_GRID, soft_impute_path

DEFAULT_FRACTION = 0.1    # used only with --no-tune
TIE_TOLERANCE = 0.001     # candidates within 0.1% of the lowest check error count as tied


def tune_softimpute(X_missing: np.ndarray, seed: int, grid=SOFT_GRID, share: float = CHECK_SHARE) -> dict:
    """Score every relative lambda on the check cells; return the winner and every candidate's error."""
    began = time.perf_counter()
    X_check, check = hide_and_check(np.asarray(X_missing, dtype=float), seed, share)
    reference = np.asarray(X_missing, dtype=float)
    candidates = []
    for fraction, filled, info in soft_impute_path(X_check, grid):
        error = filled[check] - reference[check]
        candidates.append({'fraction': fraction, 'rmse': float(np.sqrt(np.mean(error ** 2))),
                           'mae': float(np.mean(np.abs(error))), 'rank': info.details['rank'],
                           'iterations': info.details['iterations'], 'converged': info.details['converged']})
    candidates.sort(key=lambda c: c['fraction'])
    # Small lambdas warm-started from a larger one barely move, so their errors differ only in the 4th or 5th
    # decimal. Take the strongest shrinkage within 0.1% of the lowest error rather than a winner by rounding noise.
    lowest = min(c['rmse'] for c in candidates)
    best = max((c for c in candidates if c['rmse'] <= lowest * (1 + TIE_TOLERANCE)), key=lambda c: c['fraction'])
    edge = best['fraction'] in (min(grid), max(grid))
    return {'params': {'fraction': best['fraction']}, 'check_share': share, 'check_cells': int(check.sum()),
            'candidates': candidates, 'on_grid_edge': edge, 'seconds': time.perf_counter() - began}
