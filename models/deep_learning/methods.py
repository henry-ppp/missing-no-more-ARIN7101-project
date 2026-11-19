"""Factory for deep-learning imputation methods."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from models.deep_learning.autoencoder_imputer import DeepAutoencoderImputer
from models.deep_learning.gain_imputer import GAINImputer
from models.deep_learning.mida_imputer import MIDAImputer
from models.deep_learning.miwae_imputer import MIWAEImputer
from models.deep_learning.vae_imputer import VariationalAutoencoderImputer


class Imputer(Protocol):
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


METHODS = {
    "autoencoder": DeepAutoencoderImputer,
    "vae": VariationalAutoencoderImputer,
    "gain": GAINImputer,
    "mida": MIDAImputer,
    "miwae": MIWAEImputer,
}


def create_imputer(
    method: str,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    verbose: bool,
    training_mask_rate: float = 0.2,
) -> Imputer:
    """Create one imputer by method name using shared runner settings."""

    if method not in METHODS:
        valid = ", ".join(sorted(METHODS))
        raise KeyError(f"unknown method {method!r}; choose one of: {valid}")
    return METHODS[method](
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        training_mask_rate=training_mask_rate,
    )
