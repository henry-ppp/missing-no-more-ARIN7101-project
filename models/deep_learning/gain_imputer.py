"""GAIN-style adversarial imputer for numeric tabular data.

This is a compact PyTorch implementation inspired by:
Yoon, Jordon and van der Schaar (2018), "GAIN: Missing Data Imputation using
Generative Adversarial Nets".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class GAINImputer:
    """Impute numeric DataFrames with a GAIN-style generator/discriminator.

    GAIN is more expensive and less stable than the autoencoder baselines, but it
    gives the deep-learning section an adversarial method that is directly cited
    in the dataset design document.
    """

    hidden_dims: tuple[int, ...] | None = None
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    hint_rate: float = 0.9
    alpha: float = 10.0
    dropout: float = 0.0
    seed: int = 0
    device: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.columns_: pd.Index | None = None
        self.min_: pd.Series | None = None
        self.max_: pd.Series | None = None
        self.fill_values_: pd.Series | None = None
        self.generator_: _MLP | None = None
        self.discriminator_: _MLP | None = None
        self.history_: list[dict[str, float]] = []

    def fit(self, X: pd.DataFrame) -> "GAINImputer":
        X = self._validate_input(X)
        self.columns_ = X.columns
        self.min_ = X.min(axis=0)
        max_values = X.max(axis=0)
        ranges = (max_values - self.min_).replace(0, 1.0)
        self.max_ = self.min_ + ranges
        self.fill_values_ = X.mean(axis=0).fillna(0.0)

        observed = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.min_) / (self.max_ - self.min_)).clip(0.0, 1.0).fillna(0.0)

        values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
        observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.float32)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        n_features = X.shape[1]
        hidden_dims = self.hidden_dims or self._default_hidden_dims(n_features)
        generator = _MLP(n_features * 2, n_features, hidden_dims, self.dropout).to(device)
        discriminator = _MLP(n_features * 2, n_features, hidden_dims, self.dropout).to(device)
        gen_opt = torch.optim.Adam(generator.parameters(), lr=self.learning_rate)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate)

        loader = DataLoader(
            TensorDataset(values_tensor, observed_tensor),
            batch_size=min(self.batch_size, max(1, len(X))),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            generator.train()
            discriminator.train()
            gen_losses: list[float] = []
            disc_losses: list[float] = []

            for batch_values, batch_observed in loader:
                batch_values = batch_values.to(device)
                batch_observed = batch_observed.to(device)
                noise = torch.rand_like(batch_values)
                corrupt_values = batch_observed * batch_values + (1.0 - batch_observed) * noise

                generated = torch.sigmoid(generator(torch.cat([corrupt_values, batch_observed], dim=1)))
                completed = batch_observed * batch_values + (1.0 - batch_observed) * generated
                hint = self._sample_hint(batch_observed).to(device)

                disc_prob = torch.sigmoid(discriminator(torch.cat([completed.detach(), hint], dim=1)))
                disc_loss = -torch.mean(
                    batch_observed * torch.log(disc_prob + 1e-8)
                    + (1.0 - batch_observed) * torch.log(1.0 - disc_prob + 1e-8)
                )
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()

                generated = torch.sigmoid(generator(torch.cat([corrupt_values, batch_observed], dim=1)))
                completed = batch_observed * batch_values + (1.0 - batch_observed) * generated
                disc_prob = torch.sigmoid(discriminator(torch.cat([completed, hint], dim=1)))
                adversarial_loss = -torch.mean((1.0 - batch_observed) * torch.log(disc_prob + 1e-8))
                reconstruction_loss = torch.sum((batch_observed * batch_values - batch_observed * generated).pow(2)) / batch_observed.sum().clamp_min(1.0)
                gen_loss = adversarial_loss + self.alpha * reconstruction_loss
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()

                gen_losses.append(float(gen_loss.item()))
                disc_losses.append(float(disc_loss.item()))

            mean_gen_loss = float(np.mean(gen_losses)) if gen_losses else float("nan")
            mean_disc_loss = float(np.mean(disc_losses)) if disc_losses else float("nan")
            self.history_.append(
                {"epoch": float(epoch), "generator_loss": mean_gen_loss, "discriminator_loss": mean_disc_loss}
            )
            if self.verbose and (epoch == 1 or epoch % 25 == 0):
                print(
                    f"epoch={epoch:04d} generator_loss={mean_gen_loss:.6f} "
                    f"discriminator_loss={mean_disc_loss:.6f}"
                )

        self.generator_ = generator.to("cpu")
        self.discriminator_ = discriminator.to("cpu")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.generator_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before transform")
        X = self._validate_input(X)
        if not X.columns.equals(self.columns_):
            raise ValueError("X columns differ from the columns used during fit")

        assert self.min_ is not None
        assert self.max_ is not None
        assert self.fill_values_ is not None

        observed = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.min_) / (self.max_ - self.min_)).clip(0.0, 1.0).fillna(0.0)

        self.generator_.eval()
        with torch.no_grad():
            values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
            observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.float32)
            generated = torch.sigmoid(
                self.generator_(torch.cat([values_tensor, observed_tensor], dim=1))
            ).cpu().numpy()

        generated = pd.DataFrame(generated, index=X.index, columns=X.columns)
        unscaled = generated * (self.max_ - self.min_) + self.min_
        imputed = X.copy()
        missing = X.isna()
        imputed[missing] = unscaled[missing]
        return imputed.astype(float)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def checkpoint(self) -> dict[str, object]:
        """Return a serializable checkpoint for the fitted imputer."""

        if self.generator_ is None or self.discriminator_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before checkpoint")
        assert self.min_ is not None
        assert self.max_ is not None
        assert self.fill_values_ is not None
        return {
            "method": "gain",
            "generator_state_dict": self.generator_.state_dict(),
            "discriminator_state_dict": self.discriminator_.state_dict(),
            "columns": list(self.columns_),
            "min": self.min_.to_dict(),
            "max": self.max_.to_dict(),
            "fill_values": self.fill_values_.to_dict(),
            "hidden_dims": self.hidden_dims or self._default_hidden_dims(len(self.columns_)),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "hint_rate": self.hint_rate,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "seed": self.seed,
            "history": self.history_,
        }

    def _sample_hint(self, observed: torch.Tensor) -> torch.Tensor:
        binary = (torch.rand_like(observed) < self.hint_rate).float()
        return binary * observed + (1.0 - binary) * 0.5

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.empty:
            raise ValueError("X must not be empty")
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] != X.shape[1]:
            non_numeric = sorted(set(X.columns) - set(numeric.columns))
            raise ValueError(f"all columns must be numeric; non-numeric columns: {non_numeric}")
        return X.astype(float)

    @staticmethod
    def _default_hidden_dims(n_features: int) -> tuple[int, ...]:
        first = min(128, max(16, n_features * 2))
        second = min(128, max(16, n_features * 2))
        return (first, second)
