"""Run the dataset-design deep-learning benchmark.

The benchmark follows ``data/DATASETS_MEETING copy.md`` and the project README:
use complete numeric tables, hide the same cells with MCAR masks, impute with
several deep-learning methods, and score only hidden cells with RMSE/MAE.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from deep_learning.datasets import RECOMMENDED_DATASETS, available_datasets, load_dataset
from deep_learning.methods import METHODS, create_imputer
from deep_learning.normalized_metrics import nmae, nrmse
from shared.mask import apply_mask, mcar_mask
from shared.metrics import mae, rmse


DEFAULT_RATES = (0.1, 0.2, 0.3, 0.5)
DEFAULT_METHODS = ("autoencoder", "vae", "gain")


def run_benchmark(
    datasets: tuple[str, ...] = RECOMMENDED_DATASETS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    rates: tuple[float, ...] = DEFAULT_RATES,
    data_dir: Path = Path("data"),
    seed: int = 0,
    epochs: int = 300,
    batch_size: int = 128,
    max_rows: int | None | str = "spec",
    verbose: bool = False,
    save_models: bool = False,
    checkpoint_dir: Path = Path("deep_learning/checkpoints"),
) -> pd.DataFrame:
    """Run all dataset/method/rate combinations and return a result table."""

    if save_models:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset_name in datasets:
        X = load_dataset(dataset_name, data_dir=data_dir, max_rows=max_rows, seed=seed)
        print(f"\nDataset={dataset_name} rows={X.shape[0]} columns={X.shape[1]}")

        for rate in rates:
            mask = mcar_mask(X, rate=rate, seed=seed)
            X_missing = apply_mask(X, mask)
            hidden_cells = int(mask.to_numpy().sum())

            for method in methods:
                print(f"  method={method} rate={rate:.1f} hidden_cells={hidden_cells}")
                started = time.perf_counter()
                imputer = create_imputer(
                    method,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose,
                )
                imputed = imputer.fit_transform(X_missing)
                elapsed = time.perf_counter() - started

                checkpoint_path = ""
                if save_models:
                    checkpoint_path = str(
                        checkpoint_dir
                        / f"{dataset_name}_{method}_rate{rate:g}_seed{seed}_epochs{epochs}.pt"
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
                    "dataset": dataset_name,
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
                    f"    rmse={result['rmse']:.6f} mae={result['mae']:.6f} "
                    f"nrmse={result['nrmse']:.6f} nmae={result['nmae']:.6f} "
                    f"time={elapsed:.1f}s"
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print the configured dataset choices and exit",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(RECOMMENDED_DATASETS),
        help="Dataset keys to run. Use --list-datasets to see choices.",
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
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing dataset files.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for masks and training.")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs per method.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Override each dataset row limit. If omitted, uses dataset-design limits.",
    )
    parser.add_argument(
        "--no-row-limit",
        action="store_true",
        help="Use all complete numeric rows instead of the dataset-design 3000-row cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deep_learning/results_benchmark.csv"),
        help="Where to save the result table.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted model checkpoints for every dataset/rate/method run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("deep_learning/checkpoints"),
        help="Directory for saved model checkpoints when --save-models is used.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training progress.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        print(available_datasets().to_string(index=False))
        return

    max_rows: int | None | str
    if args.no_row_limit:
        max_rows = None
    elif args.max_rows is not None:
        max_rows = args.max_rows
    else:
        max_rows = "spec"

    results = run_benchmark(
        datasets=tuple(args.datasets),
        methods=tuple(args.methods),
        rates=tuple(args.rates),
        data_dir=args.data_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_rows=max_rows,
        verbose=args.verbose,
        save_models=args.save_models,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
