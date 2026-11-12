"""Score aligned predictions exclusively at artificially hidden cells."""
import numpy as np
import pandas as pd

def validate(X_true, X_pred, mask):
    if not (X_true.shape == X_pred.shape == mask.shape):
        raise ValueError('Matrices must have identical shapes')
    frames = (X_true, X_pred, mask)
    if any(isinstance(x, pd.DataFrame) for x in frames):
        if not all(isinstance(x, pd.DataFrame) for x in frames):
            raise ValueError('Do not mix labeled DataFrames and unlabeled arrays')
        for frame in frames:
            if not frame.index.is_unique or not frame.columns.is_unique:
                raise ValueError('Matrix identifiers must be unique')
            if not frame.index.equals(X_true.index) or not frame.columns.equals(X_true.columns):
                raise ValueError('Row identifiers and feature order must match exactly')
    hidden = np.asarray(mask)
    if hidden.ndim != 2 or not np.isin(hidden, [0, 1]).all():
        raise ValueError('Mask must contain only Boolean or binary values')
    hidden = hidden.astype(bool)
    if not hidden.any():
        raise ValueError('Mask has no hidden cells')
    truth, prediction = np.asarray(X_true, dtype=float), np.asarray(X_pred, dtype=float)
    if not np.isfinite(truth[hidden]).all() or not np.isfinite(prediction[hidden]).all():
        raise ValueError('Evaluation truth and predictions must be finite')
    return truth, prediction, hidden

def _hidden_error(y_true, y_pred, mask):
    truth, prediction, hidden = validate(y_true, y_pred, mask)
    return truth[hidden] - prediction[hidden]

def rmse(y_true, y_pred, mask):
    return float(np.sqrt(np.mean(np.square(_hidden_error(y_true, y_pred, mask)))))

def mae(y_true, y_pred, mask):
    return float(np.mean(np.abs(_hidden_error(y_true, y_pred, mask))))

def r2(y_true, y_pred, mask):
    """Macro mean; skip <2 hidden targets and zero target variance. None if none valid."""
    truth, prediction, hidden = validate(y_true, y_pred, mask)
    scores = []
    for j in range(truth.shape[1]):
        target = truth[hidden[:, j], j]
        if len(target) < 2:
            continue
        denominator = np.square(target - target.mean()).sum()
        if denominator == 0:
            continue
        scores.append(1 - np.square(target - prediction[hidden[:, j], j]).sum() / denominator)
    return float(np.mean(scores)) if scores else None

def relative_improvement(model_rmse, baseline_rmse):
    """Undefined for a perfect (zero-error) mean baseline."""
    return float(1 - model_rmse / baseline_rmse) if baseline_rmse > 0 else None
