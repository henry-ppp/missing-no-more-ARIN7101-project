"""Normalized metrics for comparing imputation errors across datasets.

The project-level shared metrics report raw-scale RMSE/MAE. These helpers add
range-normalized errors so datasets with very different units, such as Breast
Cancer and Yeast, can be compared more fairly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalized_hidden_error(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    mask: pd.DataFrame,
) -> np.ndarray:
    if not (y_true.shape == y_pred.shape == mask.shape):
        raise ValueError("y_true, y_pred, and mask must have the same shape")
    if not y_true.columns.equals(y_pred.columns) or not y_true.columns.equals(mask.columns):
        raise ValueError("y_true, y_pred, and mask must have the same columns")

    hidden = mask.to_numpy().astype(bool)
    if not hidden.any():
        raise ValueError("mask has no hidden cells")

    ranges = (y_true.max(axis=0) - y_true.min(axis=0)).replace(0, 1.0).fillna(1.0)
    raw_error = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    normalized_error = raw_error / ranges.to_numpy(dtype=float)
    return normalized_error[hidden]


def nrmse(y_true: pd.DataFrame, y_pred: pd.DataFrame, mask: pd.DataFrame) -> float:
    """RMSE on hidden cells after dividing each column error by its range."""

    error = _normalized_hidden_error(y_true, y_pred, mask)
    return float(np.sqrt(np.mean(np.square(error))))


def nmae(y_true: pd.DataFrame, y_pred: pd.DataFrame, mask: pd.DataFrame) -> float:
    """MAE on hidden cells after dividing each column error by its range."""

    error = _normalized_hidden_error(y_true, y_pred, mask)
    return float(np.mean(np.abs(error)))
