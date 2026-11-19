"""Run iterative ridge on the shared benchmark.

Loads only the incomplete standardized matrix. The default is the three Full
tables DAE v2 already scored, at 30% MCAR and seed 0:

    python -m models.iterative.run_experiment --datasets credit_card california_housing gas_turbine --mode full --rate 0.3 --seed 0 --run-id initial
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.dataset_config import DATASETS, MODEL_INPUT, OPTIONAL, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import load_model_input
from models.iterative.imputer import IterativeModel
from shared_evaluation.result_paths import validate_identifier

MODEL = "iterative"
DEFAULT_DATASETS = ("credit_card", "california_housing", "gas_turbine")
HEADLINE = ("relative_improvement", "wasserstein_improvement", "gap_closed")


def prediction_path(output_dir, dataset, mode, rate, seed, run_id) -> Path:
    return config_path(output_dir, dataset, mode, rate, seed) / MODEL / run_id / "predictions.csv"


def load_checked_input(processed_dir, dataset, mode, rate, seed):
    """Incomplete standardized matrix only. Refuses a file whose hash is not the published one."""

    benchmark = config_path(processed_dir, dataset, mode, rate, seed)
    metadata_file = benchmark / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    for key, expected in {"dataset": dataset, "mode": mode, "rate": rate, "seed": seed}.items():
        if metadata[key] != expected:
            raise ValueError(f"Mismatched benchmark metadata: {key}")
    incomplete = load_model_input(benchmark)
    if list(incomplete.columns) != metadata["features"] or incomplete.index.tolist() != metadata["row_ids"]:
        raise ValueError("Shared input identifiers differ from benchmark metadata")
    digest = sha256(benchmark / MODEL_INPUT)
    if digest != metadata["artifact_sha256"][MODEL_INPUT]:
        raise ValueError("Shared input file was modified")
    return benchmark, metadata, incomplete, digest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _headline(report: dict) -> dict:
    accuracy = report["accuracy_and_distribution"]["methods"][MODEL]
    headline = {name: {"estimate": accuracy[name]["estimate"], "ci": accuracy[name]["ci"]}
                for name in ("relative_improvement", "wasserstein_improvement")}
    downstream = report.get("downstream", {})
    if "methods" in downstream:
        gap = downstream["methods"][MODEL]["gap_closed"]
        headline["gap_closed"] = {"estimate": gap["estimate"], "ci": gap["ci"]}
    return headline


def write_summary(results_dir) -> Path:
    root = Path(results_dir) / MODEL
    rows = []
    for file in sorted((root / "experiment").glob("*/*/*/*/experiment.json")):
        report = json.loads(file.read_text(encoding="utf-8"))
        row = {key: report[key] for key in (
            "dataset", "mode", "rate", "seed", "run_id", "rows", "features",
            "max_iter", "tol", "alpha", "n_iter", "fit_seconds")}
        point = report.get("metrics") or report.get("metrics_v1")
        row.update({f"v1_{key}": value for key, value in point.items()})
        row["mean_baseline_rmse"] = report["mean_baseline_rmse"]
        for name in HEADLINE:
            value = report["headline"].get(name)
            row[name] = None if value is None else value["estimate"]
            row[f"{name}_ci_low"] = None if value is None else value["ci"][0]
            row[f"{name}_ci_high"] = None if value is None else value["ci"][1]
        rows.append(row)
    target = root / "evaluation" / "summary" / "benchmark_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["mode", "dataset", "rate", "seed", "run_id"])
    text = frame.to_csv(index=False)
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target


def run(dataset, mode="full", rate=0.3, seed=0, *, max_iter=50, tol=1e-3, alpha=1.0,
        processed_dir="data/processed", output_dir="data/output", results_dir="results",
        run_id="initial", n_boot=1000):
    validate_identifier(run_id, "Run ID")
    benchmark, metadata, incomplete, input_sha = load_checked_input(processed_dir, dataset, mode, rate, seed)
    predictions = prediction_path(output_dir, dataset, mode, rate, seed, run_id)
    experiment_dir = Path(results_dir) / MODEL / "experiment" / dataset / mode / benchmark.name / run_id
    if predictions.parent.exists() or experiment_dir.exists():
        raise FileExistsError(f"Run ID already exists for {dataset}; choose a new --run-id to preserve artifacts")
    predictions.parent.mkdir(parents=True)
    experiment_dir.mkdir(parents=True)

    print(f"Fitting {dataset} {mode}: {incomplete.shape[0]} rows, {incomplete.shape[1]} columns", flush=True)
    started = time.perf_counter()
    model = IterativeModel(max_iter=max_iter, tol=tol, alpha=alpha)
    imputed = model.fit_transform(incomplete)
    elapsed = time.perf_counter() - started
    observed = incomplete.notna().to_numpy()
    if not np.array_equal(imputed.to_numpy()[observed], incomplete.to_numpy()[observed]):
        raise ValueError("observed cells were modified")
    if not np.isfinite(imputed.to_numpy()).all() or imputed.isna().any().any():
        raise ValueError("iterative imputation left a missing or non-finite cell")
    if not imputed.index.equals(incomplete.index) or not imputed.columns.equals(incomplete.columns):
        raise ValueError("predictions changed the row IDs or column order")

    imputed.to_csv(predictions, index_label="row_id")
    prediction_sha = sha256(predictions)
    _write_json(predictions.parent / "metadata.json", {
        "dataset": dataset, "mode": mode, "rate": rate, "seed": seed, "model": MODEL, "run_id": run_id,
        "max_iter": max_iter, "tol": tol, "alpha": alpha, "n_iter": model.n_iter_, "fit_seconds": elapsed,
        "benchmark_metadata_sha256": sha256(benchmark / "metadata.json"),
        "input_sha256": input_sha, "prediction_sha256": prediction_sha})
    print(f"{dataset}: {model.n_iter_} rounds, {elapsed:.1f}s", flush=True)

    # Imported only now, after predictions exist, so fitting never opens evaluation files.
    from shared_evaluation.evaluate import evaluate_files

    saved, evaluation_path = evaluate_files(
        dataset, mode, rate, seed, {MODEL: predictions}, processed_dir, results_dir,
        n_boot=n_boot, boot_seed=seed, run_downstream=True, model=MODEL, run_id=run_id)
    methods = saved["accuracy_and_distribution"]["methods"]
    scores = {name: methods[MODEL][name]["estimate"] for name in (
        "rmse", "mae", "r2", "relative_improvement", "wasserstein", "wasserstein_improvement")}
    report = {
        "dataset": dataset, "mode": mode, "rate": rate, "seed": seed, "run_id": run_id,
        "rows": int(incomplete.shape[0]), "features": int(incomplete.shape[1]),
        "max_iter": max_iter, "tol": tol, "alpha": alpha, "n_iter": model.n_iter_, "fit_seconds": elapsed,
        "metrics": scores, "mean_baseline_rmse": methods["mean"]["rmse"]["estimate"],
        "headline": _headline(saved), "predictions": predictions.as_posix(), "prediction_sha256": prediction_sha,
        "input_sha256": input_sha, "evaluation": evaluation_path.as_posix()}
    _write_json(experiment_dir / "experiment.json", report)
    write_summary(results_dir)
    print(json.dumps({"dataset": dataset, "metrics": scores, "headline": report["headline"]}, allow_nan=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=[*DATASETS, *OPTIONAL], default=list(DEFAULT_DATASETS))
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default="initial")
    args = parser.parse_args()
    validate_identifier(args.run_id, "Run ID")
    failures = []
    for dataset in args.datasets:
        try:
            run(dataset, args.mode, args.rate, args.seed, max_iter=args.max_iter, tol=args.tol, alpha=args.alpha,
                processed_dir=args.processed_dir, output_dir=args.output_dir, results_dir=args.results_dir,
                run_id=args.run_id, n_boot=args.n_boot)
        except Exception as exc:
            traceback.print_exc()
            failures.append({"dataset": dataset, "mode": args.mode, "error": repr(exc)})
    if failures:
        target = Path(args.results_dir) / MODEL / "experiment" / "failures" / f"failures_{args.mode}_{args.run_id}_{time.time_ns()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
