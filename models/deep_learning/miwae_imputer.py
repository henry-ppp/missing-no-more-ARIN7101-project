"""MIWAE-lite imputer for numeric tabular data.

This is a lightweight MIWAE-style implementation for the project benchmark. It
uses the same two-mask protocol as the other deep-learning imputers and optimizes
an importance-weighted variational objective on the additional training-hidden
cells. The decoder is Gaussian with fixed unit variance on standardized data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.deep_learning.vae_imputer import _VariationalAutoencoder


@dataclass
class MIWAEImputer:
    """Missing-data importance-weighted autoencoder imputer."""

    hidden_dims: tuple[int, ...] | None = None
    latent_dim: int | None = None
    importance_samples: int = 10
    prediction_samples: int = 10
    kl_weight: float = 1e-3
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
        self.model_: _VariationalAutoencoder | None = None
        self.history_: list[dict[str, float]] = []

    def fit(self, X: pd.DataFrame) -> "MIWAEImputer":
        X = self._validate_input(X)
        if not 0 < self.training_mask_rate < 1:
            raise ValueError("training_mask_rate must be strictly between 0 and 1")
        if self.importance_samples < 1:
            raise ValueError("importance_samples must be positive")

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
        latent_dim = self.latent_dim or max(2, min(16, n_features // 2))
        model = _VariationalAutoencoder(n_features, hidden_dims, latent_dim, self.dropout).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        train_idx, val_idx = self._train_validation_split(len(X))
        train_loader = DataLoader(
            TensorDataset(values_tensor[train_idx], available_tensor[train_idx]),
            batch_size=min(self.batch_size, max(1, len(train_idx))),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_target_mask = self._sample_training_mask(
            available_tensor[val_idx], torch.Generator().manual_seed(self.seed + 30_000), self.training_mask_rate
        )
        mask_generator = torch.Generator(device=device).manual_seed(self.seed + 4)
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
                loss = self._miwae_loss(model, input_values, input_observed, batch_values, target_mask)
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
            latent_mean, latent_logvar = self.model_.encode(input_values, observed_tensor)
            reconstructions = []
            for _ in range(self.prediction_samples):
                latent = self.model_.reparameterize(latent_mean, latent_logvar)
                reconstructions.append(self.model_.decode(latent).cpu())
            reconstructed = torch.stack(reconstructions, dim=0).mean(dim=0).numpy()
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
        n_features = len(self.columns_)
        return {
            "method": "miwae",
            "state_dict": self.model_.state_dict(),
            "columns": list(self.columns_),
            "mean": self.mean_.to_dict(),
            "std": self.std_.to_dict(),
            "fill_values": self.fill_values_.to_dict(),
            "hidden_dims": self.hidden_dims or self._default_hidden_dims(n_features),
            "latent_dim": self.latent_dim or max(2, min(16, n_features // 2)),
            "importance_samples": self.importance_samples,
            "prediction_samples": self.prediction_samples,
            "kl_weight": self.kl_weight,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "training_mask_rate": self.training_mask_rate,
            "seed": self.seed,
            "history": self.history_,
        }

    def _miwae_loss(
        self,
        model: _VariationalAutoencoder,
        input_values: torch.Tensor,
        input_observed: torch.Tensor,
        target_values: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not target_mask.any():
            raise ValueError("No supervised denoising targets")
        mean, logvar = model.encode(input_values, input_observed)
        log_weights = []
        for _ in range(self.importance_samples):
            z = model.reparameterize(mean, logvar)
            recon = model.decode(z)
            # Gaussian decoder on standardized data. Use per-row average masked
            # reconstruction likelihood so high-dimensional datasets do not
            # dominate the objective purely by feature count.
            target_count = target_mask.sum(dim=1).clamp_min(1)
            log_px = -0.5 * (((recon - target_values).pow(2) * target_mask).sum(dim=1) / target_count)
            log_pz = -0.5 * z.pow(2).sum(dim=1)
            log_qz = -0.5 * (((z - mean).pow(2) / logvar.exp().clamp_min(1e-8)) + logvar).sum(dim=1)
            log_weights.append(log_px + self.kl_weight * (log_pz - log_qz))
        stacked = torch.stack(log_weights, dim=0)
        return -(torch.logsumexp(stacked, dim=0) - np.log(self.importance_samples)).mean()

    def _evaluate_loss(
        self,
        model: _VariationalAutoencoder,
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
            loss = self._miwae_loss(model, input_values, input_observed, batch_values, batch_target_mask)
        return float(loss.item())

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
