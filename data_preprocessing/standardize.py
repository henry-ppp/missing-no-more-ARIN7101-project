"""Population Z scores fitted exclusively to observed input cells."""
import numpy as np
import pandas as pd

def fit_observed(X_missing):
    if (X_missing.count() == 0).any():
        raise ValueError('Mask leaves a feature without observations; choose another configuration')
    mean = X_missing.mean()
    std = X_missing.std(ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Nonfinite scaler statistics')
    zero = std == 0
    return {'columns': list(X_missing.columns), 'mean': mean.tolist(),
            'scale': std.mask(zero, 1.0).tolist(), 'zero_variance': zero.tolist(), 'ddof': 0}

def transform(X, stats, inverse=False):
    if list(X.columns) != stats['columns']:
        raise ValueError('Scaler feature order mismatch')
    mean = pd.Series(stats['mean'], index=X.columns)
    scale = pd.Series(stats['scale'], index=X.columns)
    return X * scale + mean if inverse else (X - mean) / scale
