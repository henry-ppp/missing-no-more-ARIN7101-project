"""Run a quick deep-learning imputation experiment on one numeric CSV file.

For the full dataset-design benchmark, prefer:

    python -m models.deep_learning.run_benchmark

This script is kept for quick checks on one local CSV.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from models.deep_learning.methods import METHODS, create_imputer
from models.deep_learning.normalized_metrics import nmae, nrmse
from data_preprocessing.mask import apply_mask, mcar_mask
from shared_evaluation.metrics import mae, rmse


DEFAULT_RATES = (0.1, 0.2, 0.3, 0.5)
DEFAULT_METHODS = ("autoencoder", "vae", "gain")


def load_numeric_csv(path: Path, drop_columns: list[str], max_rows: int | None, seed: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    if drop_columns:
        missing = sorted(set(drop_columns) - set(data.columns))
        if missing:
            raise ValueError(f"columns not found in {path}: {missing}")
        data = data.drop(columns=drop_columns)

    numeric = data.select_dtypes(include="number")
    dropped = sorted(set(data.columns) - set(numeric.columns))
    if dropped:
        print(f"Dropping non-numeric columns: {dropped}")

    numeric = numeric.drop_duplicates().dropna(axis=0, how="any")
    if max_rows is not None and len(numeric) > max_rows:
        numeric = numeric.sample(n=max_rows, random_state=seed).sort_index()
    if numeric.empty:
        raise ValueError("no complete numeric rows remain after loading the CSV")
    return numeric.astype(float)


def run_experiment(
    X: pd.DataFrame,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    rates: tuple[float, ...] = DEFAULT_RATES,
    seed: int = 0,
    epochs: int = 300,
    batch_size: int = 128,
    verbose: bool = False,
    save_models: bool = False,
    checkpoint_dir: Path = Path("models/deep_learning/checkpoints"),
    dataset_name: str = "quick_csv",
) -> pd.DataFrame:
    if save_models:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for rate in rates:
        mask = mcar_mask(X, rate=rate, seed=seed)
        X_missing = apply_mask(X, mask)
        hidden_cells = int(mask.to_numpy().sum())
        for method in methods:
            started = time.perf_counter()
            imputer = create_imputer(
                method,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                verbose=verbose,
            )
            imputed = imputer.fit_transform(X_missing)
            elapsed = time.perf_counter() - started

            checkpoint_path = ""
            if save_models:
                checkpoint_path = str(
                    checkpoint_dir / f"{dataset_name}_{method}_rate{rate:g}_seed{seed}_epochs{epochs}.pt"
                )
                torch.save(
                    {
                        "dataset": dataset_name,
                        "missing_rate": rate,
                        "seed": seed,
                        "rows": X.shape[0],
                        "columns": X.shape[1],
                        "imputer": imputer.checkpoint(),
                    },
                    checkpoint_path,
                )

            result = {
                "method": method,
                "missing_rate": rate,
                "rmse": rmse(X, imputed, mask),
                "mae": mae(X, imputed, mask),
                "nrmse": nrmse(X, imputed, mask),
                "nmae": nmae(X, imputed, mask),
                "rows": X.shape[0],
                "columns": X.shape[1],
                "hidden_cells": hidden_cells,
                "seed": seed,
                "epochs": epochs,
                "elapsed_seconds": elapsed,
                "checkpoint_path": checkpoint_path,
            }
            rows.append(result)
            print(
                f"method={method} rate={rate:.1f} rmse={result['rmse']:.6f} "
                f"mae={result['mae']:.6f} nrmse={result['nrmse']:.6f} "
                f"nmae={result['nmae']:.6f} rows={X.shape[0]} columns={X.shape[1]} "
                f"time={elapsed:.1f}s"
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to a dataset CSV file")
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Column to exclude before training; can be repeated for labels/IDs",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=sorted(METHODS),
        help="Deep-learning imputation methods to compare.",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_RATES),
        help="Missing rates to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for masks and training")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=3000,
        help="Maximum complete numeric rows to use; set 0 to disable the cap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/deep_learning/results_quick.csv"),
        help="Where to save the result table",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted model checkpoints for every rate/method run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/deep_learning/checkpoints"),
        help="Directory for saved model checkpoints when --save-models is used.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = None if args.max_rows == 0 else args.max_rows
    X = load_numeric_csv(args.csv, args.drop_column, max_rows=max_rows, seed=args.seed)
    results = run_experiment(
        X,
        methods=tuple(args.methods),
        rates=tuple(args.rates),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        save_models=args.save_models,
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.csv.stem,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
