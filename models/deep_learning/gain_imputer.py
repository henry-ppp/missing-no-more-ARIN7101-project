"""GAIN-style adversarial imputer for numeric tabular data with two-mask training.

Optimized defaults for the shared V2 benchmark:
- stronger reconstruction weight so generated values stay numerically useful;
- less aggressive hint rate;
- held-out validation mask and early stopping on reconstruction loss.
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
    """Impute numeric DataFrames with a compact two-mask GAIN-style model."""

    hidden_dims: tuple[int, ...] | None = None
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    hint_rate: float = 0.7
    alpha: float = 100.0
    dropout: float = 0.0
    training_mask_rate: float = 0.2
    validation_fraction: float = 0.1
    patience: int = 20
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
        if not 0 < self.training_mask_rate < 1:
            raise ValueError("training_mask_rate must be strictly between 0 and 1")
        if not 0 < self.hint_rate < 1:
            raise ValueError("hint_rate must be strictly between 0 and 1")
        self.columns_ = X.columns
        self.min_ = X.min(axis=0)
        ranges = (X.max(axis=0) - self.min_).replace(0, 1.0)
        self.max_ = self.min_ + ranges
        self.fill_values_ = X.mean(axis=0).fillna(0.0)
        available = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.min_) / (self.max_ - self.min_)).clip(0.0, 1.0).fillna(0.0)
        values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
        available_tensor = torch.as_tensor(available.to_numpy(), dtype=torch.bool)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        n_features = X.shape[1]
        hidden_dims = self.hidden_dims or self._default_hidden_dims(n_features)
        generator = _MLP(n_features * 2, n_features, hidden_dims, self.dropout).to(device)
        discriminator = _MLP(n_features * 2, n_features, hidden_dims, self.dropout).to(device)
        gen_opt = torch.optim.Adam(generator.parameters(), lr=self.learning_rate)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate)

        train_idx, val_idx = self._train_validation_split(len(X))
        loader = DataLoader(
            TensorDataset(values_tensor[train_idx], available_tensor[train_idx]),
            batch_size=min(self.batch_size, max(1, len(train_idx))),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_target_mask = self._sample_training_mask(
            available_tensor[val_idx], torch.Generator().manual_seed(self.seed + 40_000), self.training_mask_rate
        )
        mask_generator = torch.Generator(device=device).manual_seed(self.seed + 3)
        best_state = None
        best_val = float("inf")
        best_epoch = 0
        stale = 0
        self.history_ = []

        for epoch in range(1, self.epochs + 1):
            generator.train()
            discriminator.train()
            gen_losses: list[float] = []
            disc_losses: list[float] = []
            recon_losses: list[float] = []
            for batch_values, batch_available in loader:
                batch_values = batch_values.to(device)
                batch_available = batch_available.to(device)
                target_mask = self._sample_training_mask(batch_available, mask_generator, self.training_mask_rate)
                input_observed = batch_available & ~target_mask
                noise = torch.rand(batch_values.shape, generator=mask_generator, device=device)
                corrupt_values = torch.where(input_observed, batch_values, noise)

                generated = torch.sigmoid(generator(torch.cat([corrupt_values, input_observed.to(batch_values.dtype)], dim=1)))
                completed = torch.where(input_observed, batch_values, generated)
                hint = self._sample_hint(input_observed, mask_generator).to(device)

                # Discriminator: visible cells are real, training-mask cells are generated.
                disc_prob = torch.sigmoid(discriminator(torch.cat([completed.detach(), hint], dim=1)))
                disc_loss = -torch.sum(
                    input_observed * torch.log(disc_prob + 1e-8)
                    + target_mask * torch.log(1.0 - disc_prob + 1e-8)
                ) / (input_observed.sum() + target_mask.sum()).clamp_min(1)
                disc_opt.zero_grad()
                disc_loss.backward()
                disc_opt.step()

                generated = torch.sigmoid(generator(torch.cat([corrupt_values, input_observed.to(batch_values.dtype)], dim=1)))
                completed = torch.where(input_observed, batch_values, generated)
                disc_prob = torch.sigmoid(discriminator(torch.cat([completed, hint], dim=1)))
                adversarial_loss = -torch.sum(target_mask * torch.log(disc_prob + 1e-8)) / target_mask.sum().clamp_min(1)
                reconstruction_loss = (generated[target_mask] - batch_values[target_mask]).square().mean()
                gen_loss = adversarial_loss + self.alpha * reconstruction_loss
                gen_opt.zero_grad()
                gen_loss.backward()
                gen_opt.step()
                gen_losses.append(float(gen_loss.item()))
                disc_losses.append(float(disc_loss.item()))
                recon_losses.append(float(reconstruction_loss.item()))

            val_loss = self._evaluate_loss(generator, values_tensor, available_tensor, val_idx, val_target_mask, device)
            mean_gen_loss = float(np.mean(gen_losses)) if gen_losses else float("nan")
            mean_disc_loss = float(np.mean(disc_losses)) if disc_losses else float("nan")
            mean_recon_loss = float(np.mean(recon_losses)) if recon_losses else float("nan")
            self.history_.append({
                "epoch": float(epoch),
                "generator_loss": mean_gen_loss,
                "discriminator_loss": mean_disc_loss,
                "reconstruction_loss": mean_recon_loss,
                "val_loss": val_loss,
            })
            if self.verbose and (epoch == 1 or epoch % 25 == 0):
                print(
                    f"epoch={epoch:04d} generator_loss={mean_gen_loss:.6f} "
                    f"discriminator_loss={mean_disc_loss:.6f} recon={mean_recon_loss:.6f} val_loss={val_loss:.6f}",
                    flush=True,
                )
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_epoch = epoch
                stale = 0
                best_state = {
                    "generator": {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()},
                    "discriminator": {key: value.detach().cpu().clone() for key, value in discriminator.state_dict().items()},
                }
            else:
                stale += 1
            if stale >= self.patience:
                break

        if best_state is not None:
            generator.load_state_dict(best_state["generator"])
            discriminator.load_state_dict(best_state["discriminator"])
        self.generator_ = generator.to("cpu")
        self.discriminator_ = discriminator.to("cpu")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.generator_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before transform")
        X = self._validate_input(X)
        if not X.columns.equals(self.columns_):
            raise ValueError("X columns differ from the columns used during fit")
        assert self.min_ is not None and self.max_ is not None and self.fill_values_ is not None
        observed = X.notna().astype(np.float32)
        filled = X.fillna(self.fill_values_)
        scaled = ((filled - self.min_) / (self.max_ - self.min_)).clip(0.0, 1.0).fillna(0.0)
        self.generator_.eval()
        with torch.no_grad():
            values_tensor = torch.as_tensor(scaled.to_numpy(), dtype=torch.float32)
            observed_tensor = torch.as_tensor(observed.to_numpy(), dtype=torch.bool)
            input_values = values_tensor.masked_fill(~observed_tensor, 0.0)
            generated = torch.sigmoid(
                self.generator_(torch.cat([input_values, observed_tensor.to(values_tensor.dtype)], dim=1))
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
        if self.generator_ is None or self.discriminator_ is None or self.columns_ is None:
            raise RuntimeError("fit must be called before checkpoint")
        assert self.min_ is not None and self.max_ is not None and self.fill_values_ is not None
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
            "training_mask_rate": self.training_mask_rate,
            "validation_fraction": self.validation_fraction,
            "patience": self.patience,
            "seed": self.seed,
            "history": self.history_,
        }

    def _evaluate_loss(
        self,
        generator: _MLP,
        values: torch.Tensor,
        available: torch.Tensor,
        indices: np.ndarray,
        target_mask: torch.Tensor,
        device: torch.device,
    ) -> float:
        generator.eval()
        with torch.no_grad():
            batch_values = values[indices].to(device)
            batch_available = available[indices].to(device)
            batch_target_mask = target_mask.to(device)
            input_observed = batch_available & ~batch_target_mask
            input_values = batch_values.masked_fill(~input_observed, 0.0)
            generated = torch.sigmoid(generator(torch.cat([input_values, input_observed.to(batch_values.dtype)], dim=1)))
            return float((generated[batch_target_mask] - batch_values[batch_target_mask]).square().mean().item())

    def _train_validation_split(self, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(n_rows)
        if n_rows < 5 or self.validation_fraction <= 0:
            return indices, indices
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        n_val = max(1, int(round(n_rows * self.validation_fraction)))
        n_val = min(n_val, n_rows - 1)
        return indices[n_val:], indices[:n_val]

    def _sample_hint(self, input_observed: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        binary = (torch.rand(input_observed.shape, generator=generator, device=input_observed.device) < self.hint_rate)
        return torch.where(binary, input_observed.to(torch.float32), torch.full(input_observed.shape, 0.5, device=input_observed.device))

    @staticmethod
    def _sample_training_mask(available: torch.Tensor, generator: torch.Generator, rate: float) -> torch.Tensor:
        target_mask = (torch.rand(available.shape, generator=generator, device=available.device) < rate) & available
        if not target_mask.any() and available.any():
            observed_positions = torch.nonzero(available.reshape(-1), as_tuple=False).reshape(-1)
            chosen = observed_positions[torch.randint(len(observed_positions), (1,), generator=generator, device=available.device)]
            target_mask.reshape(-1)[chosen] = True
        return target_mask

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
