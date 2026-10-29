"""Score imputations only on cells that were hidden."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hidden_error(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    mask: pd.DataFrame,
) -> np.ndarray:
    if not (y_true.shape == y_pred.shape == mask.shape):
        raise ValueError("y_true, y_pred, and mask must have the same shape")
    hidden = mask.to_numpy().astype(bool)
    if not hidden.any():
        raise ValueError("mask has no hidden cells")
    error = y_true.to_numpy()[hidden] - y_pred.to_numpy()[hidden]
    return error.astype(float)


def rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame, mask: pd.DataFrame) -> float:
    error = _hidden_error(y_true, y_pred, mask)
    return float(np.sqrt(np.mean(np.square(error))))


def mae(y_true: pd.DataFrame, y_pred: pd.DataFrame, mask: pd.DataFrame) -> float:
    error = _hidden_error(y_true, y_pred, mask)
    return float(np.mean(np.abs(error)))
