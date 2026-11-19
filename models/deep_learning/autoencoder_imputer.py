"""Denoising autoencoder imputer for numeric tabular data.

Two-mask protocol:
1. The shared benchmark hides evaluation cells before fitting.
2. This imputer samples an additional training mask from still-observed cells and
   computes the loss only on those training-hidden cells.
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
        hidden_dims = list(hidden_dims) or [max(4, min(128, n_features * 2))]
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
        return self.network(torch.cat([values, observed.to(values.dtype)], dim=1))


@dataclass
class DeepAutoencoderImputer:
    """Impute numeric DataFrames with a two-mask denoising autoencoder."""

    hidden_dims: tuple[int, ...] | None = None
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.05
    training_mask_rate: float = 0.2
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
        if not 0 < self.training_mask_rate < 1:
            raise ValueError("training_mask_rate must be strictly between 0 and 1")
        self.columns_ = X.columns
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0).replace(0, 1.0).fillna(1.0)
        self.fill_values_ = self.mean_.fillna(0.0)

        available = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.mean_) / self.std_).fillna(0.0).astype(np.float32)
        values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
        available_tensor = torch.as_tensor(available.to_numpy(), dtype=torch.bool)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        n_features = X.shape[1]
        hidden_dims = self.hidden_dims or self._default_hidden_dims(n_features)
        model = _Autoencoder(n_features, hidden_dims, self.dropout).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_idx, val_idx = self._train_validation_split(len(X))
        train_loader = DataLoader(
            TensorDataset(values_tensor[train_idx], available_tensor[train_idx]),
            batch_size=min(self.batch_size, max(1, len(train_idx))),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_target_mask = self._sample_training_mask(
            available_tensor[val_idx], torch.Generator().manual_seed(self.seed + 10_000), self.training_mask_rate
        )
        mask_generator = torch.Generator(device=device).manual_seed(self.seed + 1)
        best_state = None
        best_val = float("inf")
        best_epoch = 0
        self.history_ = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss_total = 0.0
            train_weight_total = 0
            for batch_values, batch_available in train_loader:
                batch_values = batch_values.to(device)
                batch_available = batch_available.to(device)
                target_mask = self._sample_training_mask(batch_available, mask_generator, self.training_mask_rate)
                input_observed = batch_available & ~target_mask
                input_values = batch_values.masked_fill(~input_observed, 0.0)
                prediction = model(input_values, input_observed)
                loss = self._masked_mse(prediction, batch_values, target_mask)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                weight = int(target_mask.sum().item())
                train_loss_total += float(loss.item()) * max(weight, 1)
                train_weight_total += max(weight, 1)

            train_loss = train_loss_total / max(train_weight_total, 1)
            val_loss = self._evaluate_loss(model, values_tensor, available_tensor, val_idx, val_target_mask, device)
            self.history_.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
            if self.verbose and (epoch == 1 or epoch % 25 == 0):
                print(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            elif epoch - best_epoch >= self.patience:
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
        assert self.mean_ is not None and self.std_ is not None and self.fill_values_ is not None
        observed = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.mean_) / self.std_).fillna(0.0).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
            observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.bool)
            input_values = values_tensor.masked_fill(~observed_tensor, 0.0)
            reconstructed = self.model_(input_values, observed_tensor).cpu().numpy()
        reconstructed = pd.DataFrame(reconstructed, index=X.index, columns=X.columns)
        unscaled = reconstructed * self.std_ + self.mean_
        imputed = X.copy()
        missing = X.isna()
        imputed[missing] = unscaled[missing]
        return imputed.astype(float)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def checkpoint(self) -> dict[str, object]:
        if self.model_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before checkpoint")
        assert self.mean_ is not None and self.std_ is not None and self.fill_values_ is not None
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
            "training_mask_rate": self.training_mask_rate,
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
    def _sample_training_mask(available: torch.Tensor, generator: torch.Generator, rate: float) -> torch.Tensor:
        target_mask = (torch.rand(available.shape, generator=generator, device=available.device) < rate) & available
        if not target_mask.any() and available.any():
            observed_positions = torch.nonzero(available.reshape(-1), as_tuple=False).reshape(-1)
            chosen = observed_positions[torch.randint(len(observed_positions), (1,), generator=generator, device=available.device)]
            target_mask.reshape(-1)[chosen] = True
        return target_mask

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        if not target_mask.any():
            raise ValueError("No supervised denoising targets")
        return (prediction[target_mask] - target[target_mask]).square().mean()

    def _evaluate_loss(
        self,
        model: _Autoencoder,
        values: torch.Tensor,
        available: torch.Tensor,
        indices: np.ndarray,
        target_mask: torch.Tensor,
        device: torch.device,
    ) -> float:
        model.eval()
        with torch.no_grad():
            batch_values = values[indices].to(device)
            batch_available = available[indices].to(device)
            batch_target_mask = target_mask.to(device)
            input_observed = batch_available & ~batch_target_mask
            input_values = batch_values.masked_fill(~input_observed, 0.0)
            prediction = model(input_values, input_observed)
            loss = self._masked_mse(prediction, batch_values, batch_target_mask)
        return float(loss.item())
