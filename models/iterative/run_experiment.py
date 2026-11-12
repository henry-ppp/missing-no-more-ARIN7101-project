"""Run the simple iterative imputer on UCI Energy Efficiency.

Meeting protocol: seed 0, missing rates 0.1, 0.2, 0.3 and 0.5, RMSE and MAE
on the hidden numeric cells. The table is the 768 x 10 complete numeric
matrix (eight building features plus heating and cooling load).

    python -m models.iterative.run_experiment
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.mask import apply_mask, mcar_mask
from data_preprocessing.mean_baseline import mean_impute
from models.iterative.imputer import IterativeModel
from shared_evaluation.metrics import mae, relative_improvement, rmse

UCI_ID = 242
DEFAULT_PATH = Path("data/energy_efficiency.csv")
DEFAULT_RATES = (0.1, 0.2, 0.3, 0.5)
EXPECTED_SHAPE = (768, 10)

def _normalized_hidden_error(y_true: pd.DataFrame, y_pred: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
    """Hidden-cell error divided by each column's range. Same definition as the deep-learning helper."""

    hidden = mask.to_numpy().astype(bool)
    ranges = (y_true.max(axis=0) - y_true.min(axis=0)).replace(0, 1.0).to_numpy(dtype=float)
    error = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    return (error / ranges)[hidden]


def load_energy_efficiency(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the complete numeric Energy Efficiency table, downloading it once."""

    if not path.exists():
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise ImportError("Install requirements.txt; ucimlrepo is required") from exc
        frame = fetch_ucirepo(id=UCI_ID).data.original
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    data = pd.read_csv(path)
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    numeric = data.select_dtypes(include="number").dropna(axis=0, how="any")
    if numeric.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"expected Energy Efficiency shape {EXPECTED_SHAPE}, got {numeric.shape} "
            f"with columns {list(numeric.columns)}"
        )
    numeric = numeric.astype(float)
    numeric.index = pd.Index([f"energy_efficiency.csv:{i}" for i in range(len(numeric))], name="row_id")
    return numeric


def run_experiment(
    X: pd.DataFrame,
    rates: tuple[float, ...] = DEFAULT_RATES,
    seed: int = 0,
    max_iter: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rate in rates:
        mask = mcar_mask(X, rate=rate, seed=seed)
        X_missing = apply_mask(X, mask)
        started = time.perf_counter()
        model = IterativeModel(max_iter=max_iter)
        imputed = model.fit_transform(X_missing)
        elapsed = time.perf_counter() - started
        observed = ~mask.to_numpy()
        if not np.allclose(imputed.to_numpy()[observed], X.to_numpy()[observed]):
            raise ValueError("observed cells were modified")
        baseline = mean_impute(X_missing)
        model_rmse = rmse(X, imputed, mask)
        baseline_rmse = rmse(X, baseline, mask)
        normalized_error = _normalized_hidden_error(X, imputed, mask)
        rows.append(
            {
                "dataset": "energy_efficiency",
                "method": "iterative_ridge",
                "missing_rate": rate,
                "rmse": model_rmse,
                "mae": mae(X, imputed, mask),
                "nrmse": float(np.sqrt(np.mean(np.square(normalized_error)))),
                "nmae": float(np.mean(np.abs(normalized_error))),
                "mean_rmse": baseline_rmse,
                "mean_mae": mae(X, baseline, mask),
                "relative_improvement": relative_improvement(model_rmse, baseline_rmse),
                "rows": X.shape[0],
                "columns": X.shape[1],
                "hidden_cells": int(mask.to_numpy().sum()),
                "seed": seed,
                "max_iter": max_iter,
                "n_iter": model.n_iter_,
                "elapsed_seconds": elapsed,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/iterative/results_energy_efficiency.csv"),
    )
    args = parser.parse_args()

    X = load_energy_efficiency(args.data)
    print(f"Loaded {args.data}: {X.shape[0]} rows, {X.shape[1]} columns")
    print("Columns:", ", ".join(X.columns))
    results = run_experiment(X, rates=tuple(args.rates), seed=args.seed, max_iter=args.max_iter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
