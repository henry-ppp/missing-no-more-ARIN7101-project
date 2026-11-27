"""Evaluate all deep-learning predictions with the project V2 evaluator.

This script does not train models. It reads existing predictions from
``data/output`` and writes shared_evaluation v2 reports under
``results/deep_learning/evaluation``. All deep-learning predictions for one
setting are evaluated together so bootstrap intervals and paired differences are
computed consistently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_preprocessing.dataset_config import DATASETS, config_path
from shared_evaluation.evaluate import evaluate_files
from shared_evaluation.result_paths import validate_identifier


PREDICTION_RUNS = {
    "dl_autoencoder": ("dl_autoencoder", "initial"),
    "dl_vae": ("dl_vae", "initial"),
    "dl_mida": ("dl_mida", "initial"),
    "dl_gain_initial": ("dl_gain", "initial"),
    "dl_gain_optimized": ("dl_gain", "optimized"),
    "dl_miwae_initial": ("dl_miwae", "initial"),
    "dl_miwae_optimized": ("dl_miwae", "optimized"),
}

SUMMARY_METRICS = (
    "rmse",
    "mae",
    "r2",
    "relative_improvement",
    "wasserstein",
    "wasserstein_improvement",
)


def prediction_path(output_dir: str | Path, dataset: str, mode: str, rate: float, seed: int, model: str, run_id: str) -> Path:
    return config_path(output_dir, dataset, mode, rate, seed) / model / run_id / "predictions.csv"


def predictions_for_setting(output_dir: str | Path, dataset: str, mode: str, rate: float, seed: int) -> dict[str, Path]:
    predictions: dict[str, Path] = {}
    missing: list[Path] = []
    for name, (folder, run_id) in PREDICTION_RUNS.items():
        path = prediction_path(output_dir, dataset, mode, rate, seed, folder, run_id)
        if path.exists():
            predictions[name] = path
        else:
            missing.append(path)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prediction files for {dataset}/{mode}:\n{missing_text}")
    return predictions


def run_evaluation(
    dataset: str,
    mode: str,
    rate: float,
    seed: int,
    *,
    processed_dir: str = "data/processed",
    output_dir: str = "data/output",
    results_dir: str = "results",
    run_id: str = "all_methods",
    n_boot: int = 1000,
    boot_seed: int = 0,
    downstream: bool = True,
) -> Path:
    validate_identifier(run_id, "Run ID")
    predictions = predictions_for_setting(output_dir, dataset, mode, rate, seed)
    report, path = evaluate_files(
        dataset,
        mode,
        rate,
        seed,
        predictions,
        processed_dir=processed_dir,
        output_dir=results_dir,
        n_boot=n_boot,
        boot_seed=boot_seed,
        run_downstream=downstream,
        model="deep_learning",
        run_id=run_id,
    )
    print(f"Saved {path}")
    return path


def _entry_values(entry: dict) -> dict[str, object]:
    ci = entry.get("ci") or [None, None]
    return {
        "estimate": entry.get("estimate"),
        "bias": entry.get("bias"),
        "bias_corrected": entry.get("bias_corrected"),
        "ci_low": ci[0],
        "ci_high": ci[1],
    }


def write_summary(results_dir: str | Path = "results", run_id: str = "all_methods") -> tuple[Path, Path]:
    root = Path(results_dir) / "deep_learning" / "evaluation"
    reports = sorted(root.glob(f"*/*/rate_*_seed_*/*/evaluation_v2_*.json"))
    rows: list[dict[str, object]] = []
    downstream_rows: list[dict[str, object]] = []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_id") != run_id:
            continue
        setting = {
            "dataset": report["dataset"],
            "mode": report["mode"],
            "rate": report["rate"],
            "seed": report["seed"],
            "run_id": report["run_id"],
        }
        methods = report["accuracy_and_distribution"]["methods"]
        for method, metrics in methods.items():
            if method == "mean":
                continue
            row = {**setting, "method": method, "source": report_path.as_posix()}
            for metric in SUMMARY_METRICS:
                values = _entry_values(metrics[metric])
                for key, value in values.items():
                    suffix = "" if key == "estimate" else f"_{key}"
                    row[f"{metric}{suffix}"] = value
            rows.append(row)
        downstream = report.get("downstream")
        if isinstance(downstream, dict) and "methods" in downstream:
            for method, metrics in downstream["methods"].items():
                if method in {"mean", "true"}:
                    continue
                row = {**setting, "method": method, "source": report_path.as_posix()}
                for metric in ("score", "gap_closed"):
                    values = _entry_values(metrics[metric])
                    for key, value in values.items():
                        suffix = "" if key == "estimate" else f"_{key}"
                        row[f"downstream_{metric}{suffix}"] = value
                downstream_rows.append(row)

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["mode", "dataset", "method"]).reset_index(drop=True)
    summary_path = summary_dir / "benchmark_summary_v2.csv"
    summary.to_csv(summary_path, index=False)

    downstream_summary = pd.DataFrame(downstream_rows)
    if not downstream_summary.empty:
        downstream_summary = downstream_summary.sort_values(["mode", "dataset", "method"]).reset_index(drop=True)
    downstream_path = summary_dir / "downstream_summary_v2.csv"
    downstream_summary.to_csv(downstream_path, index=False)

    if not summary.empty:
        best_rel = (
            summary.sort_values(["dataset", "mode", "relative_improvement"], ascending=[True, True, False])
            .groupby(["dataset", "mode"], as_index=False)
            .first()
        )
        best_rel.to_csv(summary_dir / "best_by_relative_improvement_v2.csv", index=False)
        best_wass = (
            summary.sort_values(["dataset", "mode", "wasserstein_improvement"], ascending=[True, True, False])
            .groupby(["dataset", "mode"], as_index=False)
            .first()
        )
        best_wass.to_csv(summary_dir / "best_by_wasserstein_improvement_v2.csv", index=False)

    print(f"Saved {summary_path}")
    print(f"Saved {downstream_path}")
    return summary_path, downstream_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["california_housing", "credit_card", "gas_turbine", "superconductivity"])
    parser.add_argument("--modes", nargs="+", choices=["quick", "full"], default=["quick", "full"])
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default="all_methods")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--no-downstream", action="store_true")
    args = parser.parse_args()

    for mode in args.modes:
        for dataset in args.datasets:
            run_evaluation(
                dataset,
                mode,
                args.rate,
                args.seed,
                processed_dir=args.processed_dir,
                output_dir=args.output_dir,
                results_dir=args.results_dir,
                run_id=args.run_id,
                n_boot=args.n_boot,
                boot_seed=args.boot_seed,
                downstream=not args.no_downstream,
            )
    write_summary(args.results_dir, args.run_id)


if __name__ == "__main__":
    main()
