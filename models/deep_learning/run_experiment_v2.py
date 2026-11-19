"""Run existing deep-learning imputers on the shared V2 benchmark.

This runner mirrors ``models.deep_learning_v2.run_experiment`` in data access,
artifact layout, and evaluation. It trains the models in ``models.deep_learning``
on the prepared benchmark input only, saves predictions under ``data/output``,
then invokes the shared evaluator to produce reports under ``results``.

Example from the repository root:

    python -m models.deep_learning.run_experiment_v2 \
      --datasets california_housing credit_card gas_turbine superconductivity \
      --mode quick --rate 0.3 --seed 0 \
      --methods autoencoder vae gain --run-id initial
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_preprocessing.dataset_config import DATASETS, MODEL_INPUT, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import load_model_input
from models.deep_learning.autoencoder_imputer import DeepAutoencoderImputer
from models.deep_learning.gain_imputer import GAINImputer
from models.deep_learning.mida_imputer import MIDAImputer
from models.deep_learning.miwae_imputer import MIWAEImputer
from models.deep_learning.vae_imputer import VariationalAutoencoderImputer
from shared_evaluation.result_paths import validate_identifier


METHODS = {
    "autoencoder": ("dl_autoencoder", DeepAutoencoderImputer),
    "vae": ("dl_vae", VariationalAutoencoderImputer),
    "gain": ("dl_gain", GAINImputer),
    "mida": ("dl_mida", MIDAImputer),
    "miwae": ("dl_miwae", MIWAEImputer),
}


@dataclass(frozen=True)
class DeepLearningV2Config:
    """Training settings aligned with dae_v2 defaults where applicable."""

    epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 0.001
    training_mask_rate: float = 0.3
    validation_fraction: float = 0.1
    patience: int = 20
    dropout: float = 0.05
    seed: int = 0
    device: str = "auto"
    threads: int = 2

    def validate(self) -> None:
        if min(self.epochs, self.batch_size, self.patience, self.threads) < 1:
            raise ValueError("epochs, batch size, patience and threads must be positive")
        if not 0 < self.training_mask_rate < 1:
            raise ValueError("training_mask_rate must be strictly between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        if self.learning_rate <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid learning rate or dropout")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def _set_reproducibility(config: DeepLearningV2Config) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some torch builds/operators may not support strict deterministic mode.
        torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False


def _device_argument(config: DeepLearningV2Config) -> str | None:
    if config.device == "auto":
        return None
    return config.device


def _device_name(config: DeepLearningV2Config) -> str:
    if config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def _make_imputer(method: str, config: DeepLearningV2Config):
    if method not in METHODS:
        valid = ", ".join(sorted(METHODS))
        raise KeyError(f"unknown method {method!r}; choose one of: {valid}")
    _, cls = METHODS[method]
    kwargs = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "training_mask_rate": config.training_mask_rate,
        "seed": config.seed,
        "device": _device_argument(config),
        "verbose": True,
    }
    if method in {"autoencoder", "vae"}:
        kwargs.update(
            validation_fraction=config.validation_fraction,
            patience=config.patience,
        )
    return cls(**kwargs)


def _history_summary(imputer, config: DeepLearningV2Config) -> dict[str, object]:
    history = getattr(imputer, "history_", []) or []
    epochs_completed = len(history) if history else config.epochs
    best_epoch = epochs_completed
    best_validation_loss = None
    if history and "val_loss" in history[0]:
        best = min(history, key=lambda row: row["val_loss"])
        best_epoch = int(best["epoch"])
        best_validation_loss = float(best["val_loss"])
    elif history and "generator_loss" in history[0]:
        # GAIN has no held-out validation objective in this implementation.
        best_epoch = epochs_completed
        best_validation_loss = None
    return {
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


def _write_environment(results_dir: Path, model_id: str) -> None:
    target = results_dir / model_id / "training" / "environment.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name, module in ("torch", torch), ("numpy", np), ("pandas", pd):
        packages[name] = getattr(module, "__version__", None)
    environment = {
        "python": platform.python_version(),
        "os": platform.system(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    target.write_text(json.dumps(environment, indent=2, allow_nan=False), encoding="utf-8")


def write_summary(results_dir: str | Path, model_id: str) -> Path:
    root = Path(results_dir) / model_id
    rows = []
    for file in sorted((root / "training").glob("*/*/rate_*/*/experiment.json")):
        report = json.loads(file.read_text(encoding="utf-8"))
        rows.append(
            {
                key: report[key]
                for key in (
                    "dataset",
                    "mode",
                    "rate",
                    "seed",
                    "run_id",
                    "rows",
                    "features",
                    "rmse",
                    "mae",
                    "r2",
                    "relative_improvement",
                    "mean_baseline_rmse",
                    "best_epoch",
                    "epochs_completed",
                    "best_validation_loss",
                    "runtime_seconds",
                    "device",
                )
            }
        )
    rows.sort(key=lambda row: (row["mode"], row["dataset"], row["rate"], row["seed"], row["run_id"]))
    target = root / "evaluation" / "summary" / "benchmark_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def run(
    method: str,
    dataset: str,
    mode: str = "quick",
    rate: float = 0.3,
    seed: int = 0,
    config: DeepLearningV2Config = DeepLearningV2Config(),
    processed_dir: str = "data/processed",
    output_dir: str = "data/output",
    results_dir: str = "results",
    run_id: str = "initial",
) -> dict[str, object]:
    config.validate()
    validate_identifier(run_id, "Run ID")
    if method not in METHODS:
        valid = ", ".join(sorted(METHODS))
        raise KeyError(f"unknown method {method!r}; choose one of: {valid}")
    model_id, _ = METHODS[method]
    validate_identifier(model_id, "Model identifier")
    _set_reproducibility(config)

    benchmark = config_path(processed_dir, dataset, mode, rate, seed)
    metadata_file = benchmark / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    incomplete = load_model_input(benchmark)
    if list(incomplete.columns) != metadata["features"] or incomplete.index.tolist() != metadata["row_ids"]:
        raise ValueError("Shared input identifiers differ from benchmark metadata")
    if sha256(benchmark / MODEL_INPUT) != metadata["artifact_sha256"][MODEL_INPUT]:
        raise ValueError("Shared input file was modified")

    tag = benchmark.name
    result_folder = Path(results_dir) / model_id / "training" / dataset / mode / tag / run_id
    prediction_folder = config_path(output_dir, dataset, mode, rate, seed) / model_id / run_id
    if result_folder.exists() or prediction_folder.exists():
        raise FileExistsError(
            f"Artifacts already exist for {model_id}/{dataset}/{mode}/{tag}/{run_id}; choose a new --run-id"
        )
    result_folder.mkdir(parents=True)
    prediction_folder.mkdir(parents=True)
    _write_environment(Path(results_dir), model_id)

    print(f"Training {model_id} {dataset} {mode}: {incomplete.shape}", flush=True)
    started = time.perf_counter()
    imputer = _make_imputer(method, config)
    prediction = imputer.fit_transform(incomplete)
    runtime_seconds = time.perf_counter() - started

    if not prediction.index.equals(incomplete.index) or not prediction.columns.equals(incomplete.columns):
        raise ValueError("Prediction identifiers differ from benchmark input")
    if not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise FloatingPointError("Prediction table contains non-finite values")

    training_summary = {
        "config": asdict(config),
        "method": method,
        "model_id": model_id,
        "rows": len(incomplete),
        "features": incomplete.shape[1],
        "device": _device_name(config),
        "device_name": torch.cuda.get_device_name(0) if _device_name(config) == "cuda" else "CPU",
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "runtime_seconds": runtime_seconds,
        "input_sha256": sha256(benchmark / MODEL_INPUT),
        "benchmark_metadata_sha256": sha256(metadata_file),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "objective": "Two-mask denoising: evaluation mask comes from shared benchmark; training loss uses additional masks of observed cells.",
        **_history_summary(imputer, config),
    }

    checkpoint = Path(results_dir) / model_id / "training" / "checkpoints" / dataset / mode / tag / run_id / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dataset": dataset,
            "mode": mode,
            "rate": rate,
            "seed": seed,
            "run_id": run_id,
            "model_id": model_id,
            "method": method,
            "imputer": imputer.checkpoint(),
            "training_summary": training_summary,
        },
        checkpoint,
    )
    pd.DataFrame(getattr(imputer, "history_", [])).to_csv(result_folder / "training_history.csv", index=False)
    (result_folder / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    prediction_path = prediction_folder / "predictions.csv"
    prediction.to_csv(prediction_path, index_label="row_id")
    (prediction_folder / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "mode": mode,
                "rate": rate,
                "seed": seed,
                "model": model_id,
                "method": method,
                "run_id": run_id,
                "benchmark_metadata_sha256": training_summary["benchmark_metadata_sha256"],
                "input_sha256": training_summary["input_sha256"],
                "prediction_sha256": sha256(prediction_path),
                "checkpoint": checkpoint.as_posix(),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    from shared_evaluation.evaluate import evaluate_files

    scores, evaluation_path = evaluate_files(
        dataset,
        mode,
        rate,
        seed,
        prediction_path,
        processed_dir,
        results_dir,
        model=model_id,
        run_id=run_id,
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report = {
        "dataset": dataset,
        "mode": mode,
        "rate": rate,
        "seed": seed,
        "run_id": run_id,
        "rows": len(incomplete),
        "features": incomplete.shape[1],
        **scores,
        "mean_baseline_rmse": evaluation["mean_baseline_rmse"],
        "best_epoch": training_summary["best_epoch"],
        "epochs_completed": training_summary["epochs_completed"],
        "best_validation_loss": training_summary["best_validation_loss"],
        "runtime_seconds": training_summary["runtime_seconds"],
        "device": training_summary["device"],
        "checkpoint": checkpoint.as_posix(),
        "predictions": prediction_path.as_posix(),
        "evaluation": evaluation_path.as_posix(),
        "training_summary": (result_folder / "training_summary.json").as_posix(),
        "config": asdict(config),
        "method": method,
        "model_id": model_id,
    }
    (result_folder / "experiment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_summary(results_dir, model_id)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--datasets", "--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default="initial")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--training-mask-rate", type=float, default=0.3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    config = DeepLearningV2Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_mask_rate=args.training_mask_rate,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        threads=args.threads,
    )
    validate_identifier(args.run_id, "Run ID")
    failures = []
    for method in args.methods:
        for dataset in args.datasets:
            try:
                run(
                    method,
                    dataset,
                    args.mode,
                    args.rate,
                    args.seed,
                    config,
                    args.processed_dir,
                    args.output_dir,
                    args.results_dir,
                    args.run_id,
                )
            except Exception as exc:
                traceback.print_exc()
                failures.append({"method": method, "dataset": dataset, "mode": args.mode, "error": repr(exc)})
    if failures:
        target = (
            Path(args.results_dir)
            / "deep_learning"
            / "training"
            / "failures"
            / f"failures_{args.mode}_{args.run_id}_{time.time_ns()}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
