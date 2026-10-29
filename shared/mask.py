"""Hide the same observed cells for every method."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mcar_mask(X: pd.DataFrame, rate: float, seed: int = 0) -> pd.DataFrame:
    """Return a boolean mask that is True where an observed cell is hidden.

    Cells that are already missing stay missing and do not count toward ``rate``.
    """
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    observed = X.notna().to_numpy()
    drop = (np.random.default_rng(seed).random(X.shape) < rate) & observed
    return pd.DataFrame(drop, index=X.index, columns=X.columns)


def apply_mask(X: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``X`` with masked cells set to NaN."""
    if X.shape != mask.shape:
        raise ValueError("X and mask must have the same shape")
    return X.mask(mask)
