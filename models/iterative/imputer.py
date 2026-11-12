"""Simple iterative imputation.

Missing cells start at their column mean. Each incomplete column is then
predicted from the others with a ridge regression, and that round repeats.
This is the chained-equation (MICE) procedure of van Buuren and
Groothuis-Oudshoorn, with one linear model per column. Observed cells are
left unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IterativeModel:
    """Round-robin ridge imputer, starting from column means."""

    def __init__(self, max_iter: int = 50, tol: float = 1e-3, alpha: float = 1.0):
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.n_iter_: int | None = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if X.shape[0] < 2 or X.shape[1] < 2:
            raise ValueError("iterative imputation needs at least 2 rows and 2 columns")

        observed = X.notna().to_numpy()
        if not observed.any(axis=0).all():
            raise ValueError("every column needs at least one observed value")

        values = X.to_numpy(dtype=float)
        filled = _initial_fill(values, observed)
        ordered = [j for j in range(values.shape[1]) if not observed[:, j].all()]
        limit = self.tol * float(np.max(np.abs(values[observed])))
        self.n_iter_ = 0

        for iteration in range(1, self.max_iter + 1):
            previous = filled.copy()
            for feature in ordered:
                filled = _impute_feature(filled, values, observed, feature, self.alpha)
            self.n_iter_ = iteration
            change = float(np.linalg.norm(filled - previous, ord=np.inf))
            if change < limit:
                break

        filled[observed] = values[observed]
        if not np.isfinite(filled).all():
            raise ValueError("iterative imputation produced a non-finite value")
        return pd.DataFrame(filled, index=X.index, columns=X.columns)


def _initial_fill(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for feature in range(values.shape[1]):
        column = values[observed[:, feature], feature]
        filled[~observed[:, feature], feature] = column.mean()
    return filled


def _impute_feature(
    filled: np.ndarray,
    values: np.ndarray,
    observed: np.ndarray,
    feature: int,
    alpha: float,
) -> np.ndarray:
    train = observed[:, feature]
    missing = ~train
    predictors = np.arange(values.shape[1]) != feature
    train_x = filled[np.ix_(train, predictors)]
    missing_x = filled[np.ix_(missing, predictors)]
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    train_z = (train_x - center) / scale
    missing_z = (missing_x - center) / scale
    design = np.column_stack([np.ones(train_z.shape[0]), train_z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ values[train, feature])
    prediction = np.column_stack([np.ones(missing_z.shape[0]), missing_z]) @ coefficients
    filled[missing, feature] = prediction
    return filled
