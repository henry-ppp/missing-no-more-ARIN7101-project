"""Shared baseline; never replaces the incomplete model input."""
import numpy as np

def mean_impute(X_missing):
    result = X_missing.fillna(X_missing.mean())
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Mean baseline needs finite observations in every column')
    return result
