"""Denoising autoencoder imputer for numeric tabular data.

The model is intentionally small and CPU-friendly so it can be used for the
course comparison across several datasets and missing rates. It follows the
project protocol: train on the masked table, reconstruct all cells, and evaluate
only the held-out cells with the shared metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            hidden_dims = [max(4, min(128, n_features * 2))]

        encoder_layers: list[nn.Module] = []
        previous = n_features * 2
        for width in hidden_dims:
            encoder_layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            previous = width

        decoder_layers: list[nn.Module] = []
        for width in reversed(hidden_dims[:-1]):
            decoder_layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            previous = width
        decoder_layers.append(nn.Linear(previous, n_features))

        self.network = nn.Sequential(*encoder_layers, *decoder_layers)

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        # Concatenating the observation mask lets the network distinguish an
        # actual zero from a value that was filled with zero before training.
        return self.network(torch.cat([values, observed], dim=1))


@dataclass
class DeepAutoencoderImputer:
    """Impute numeric DataFrames with a denoising autoencoder.

    Parameters are conservative defaults for small tabular datasets. The class
    keeps the same DataFrame index/columns as the input and can be used from a
    script or notebook.
    """

    hidden_dims: tuple[int, ...] | None = None
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.05
    validation_fraction: float = 0.1
    patience: int = 30
    seed: int = 0
    device: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self.columns_: pd.Index | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.fill_values_: pd.Series | None = None
        self.model_: _Autoencoder | None = None
        self.history_: list[dict[str, float]] = []

    def fit(self, X: pd.DataFrame) -> "DeepAutoencoderImputer":
        X = self._validate_input(X)
        self.columns_ = X.columns
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0).replace(0, 1.0)
        self.std_ = std.fillna(1.0)
        self.fill_values_ = self.mean_.fillna(0.0)

        observed = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.mean_) / self.std_).fillna(0.0).astype(np.float32)

        values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
        observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.float32)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        n_features = X.shape[1]
        hidden_dims = self.hidden_dims or self._default_hidden_dims(n_features)
        model = _Autoencoder(n_features, hidden_dims, self.dropout).to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        train_idx, val_idx = self._train_validation_split(len(X))
        train_loader = DataLoader(
            TensorDataset(values_tensor[train_idx], observed_tensor[train_idx]),
            batch_size=min(self.batch_size, max(1, len(train_idx))),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        best_state = None
        best_val = float("inf")
        best_epoch = 0
        self.history_ = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_weight_total = 0.0
            for batch_values, batch_observed in train_loader:
                batch_values = batch_values.to(device)
                batch_observed = batch_observed.to(device)
                prediction = model(batch_values, batch_observed)
                loss = self._masked_mse(prediction, batch_values, batch_observed)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                weight = float(batch_observed.sum().item())
                train_loss_total += float(loss.item()) * max(weight, 1.0)
                train_weight_total += max(weight, 1.0)

            train_loss = train_loss_total / max(train_weight_total, 1.0)
            val_loss = self._evaluate_loss(model, values_tensor, observed_tensor, val_idx, device)
            self.history_.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if self.verbose and (epoch == 1 or epoch % 25 == 0):
                print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            elif epoch - best_epoch >= self.patience:
                if self.verbose:
                    print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.to("cpu")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before transform")
        X = self._validate_input(X)
        if not X.columns.equals(self.columns_):
            raise ValueError("X columns differ from the columns used during fit")

        assert self.mean_ is not None
        assert self.std_ is not None
        assert self.fill_values_ is not None

        observed = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.mean_) / self.std_).fillna(0.0).astype(np.float32)

        self.model_.eval()
        with torch.no_grad():
            values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
            observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.float32)
            reconstructed = self.model_(values_tensor, observed_tensor).cpu().numpy()

        reconstructed = pd.DataFrame(reconstructed, index=X.index, columns=X.columns)
        unscaled = reconstructed * self.std_ + self.mean_
        imputed = X.copy()
        missing = X.isna()
        imputed[missing] = unscaled[missing]
        return imputed.astype(float)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def checkpoint(self) -> dict[str, object]:
        """Return a serializable checkpoint for the fitted imputer."""

        if self.model_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before checkpoint")
        assert self.mean_ is not None
        assert self.std_ is not None
        assert self.fill_values_ is not None
        return {
            "method": "autoencoder",
            "state_dict": self.model_.state_dict(),
            "columns": list(self.columns_),
            "mean": self.mean_.to_dict(),
            "std": self.std_.to_dict(),
            "fill_values": self.fill_values_.to_dict(),
            "hidden_dims": self.hidden_dims or self._default_hidden_dims(len(self.columns_)),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "seed": self.seed,
            "history": self.history_,
        }

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
        first = min(128, max(8, n_features * 2))
        second = min(64, max(4, n_features))
        return (first, second) if first > second else (first,)

    def _train_validation_split(self, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(n_rows)
        if n_rows < 5 or self.validation_fraction <= 0:
            return indices, indices
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        n_val = max(1, int(round(n_rows * self.validation_fraction)))
        n_val = min(n_val, n_rows - 1)
        return indices[n_val:], indices[:n_val]

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        squared_error = (prediction - target).pow(2) * observed
        denominator = observed.sum().clamp_min(1.0)
        return squared_error.sum() / denominator

    def _evaluate_loss(
        self,
        model: _Autoencoder,
        values: torch.Tensor,
        observed: torch.Tensor,
        indices: np.ndarray,
        device: torch.device,
    ) -> float:
        model.eval()
        with torch.no_grad():
            batch_values = values[indices].to(device)
            batch_observed = observed[indices].to(device)
            prediction = model(batch_values, batch_observed)
            loss = self._masked_mse(prediction, batch_values, batch_observed)
        return float(loss.item())
