"""MIDA-style multiple imputation with a denoising autoencoder.

MIDA here reuses the two-mask denoising autoencoder training objective and adds
Monte-Carlo dropout at prediction time. Several stochastic reconstructions are
averaged for the missing cells, giving a lightweight multiple-imputation variant
without changing the shared benchmark/evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from models.deep_learning.autoencoder_imputer import DeepAutoencoderImputer


@dataclass
class MIDAImputer(DeepAutoencoderImputer):
    """Multiple-imputation denoising autoencoder using MC dropout."""

    n_imputations: int = 10

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before transform")
        if self.n_imputations < 1:
            raise ValueError("n_imputations must be positive")
        X = self._validate_input(X)
        if not X.columns.equals(self.columns_):
            raise ValueError("X columns differ from the columns used during fit")

        assert self.mean_ is not None and self.std_ is not None and self.fill_values_ is not None
        observed = X.notna().astype(float)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.mean_) / self.std_).fillna(0.0).astype(float)

        values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
        observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.bool)
        input_values = values_tensor.masked_fill(~observed_tensor, 0.0)

        # Keep dropout active for multiple stochastic imputations. No gradients are
        # tracked, and observed cells are preserved exactly after averaging.
        self.model_.train()
        reconstructions = []
        with torch.no_grad():
            for _ in range(self.n_imputations):
                reconstructions.append(self.model_(input_values, observed_tensor).cpu())
        self.model_.eval()
        reconstructed = torch.stack(reconstructions, dim=0).mean(dim=0).numpy()

        reconstructed = pd.DataFrame(reconstructed, index=X.index, columns=X.columns)
        unscaled = reconstructed * self.std_ + self.mean_
        imputed = X.copy()
        missing = X.isna()
        imputed[missing] = unscaled[missing]
        return imputed.astype(float)

    def checkpoint(self) -> dict[str, object]:
        state = super().checkpoint()
        state["method"] = "mida"
        state["n_imputations"] = self.n_imputations
        return state
