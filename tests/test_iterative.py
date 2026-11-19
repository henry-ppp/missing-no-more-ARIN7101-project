"""Iterative ridge keeps observed cells, and the runner reads only the shared model input."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import shared_evaluation.evaluate
import shared_evaluation.evaluate_v2
from data_preprocessing.download import sha256
from data_preprocessing.dataset_config import MODEL_INPUT
from models.iterative.imputer import IterativeModel
from models.iterative.run_experiment import prediction_path, run


class IterativeImputerTests(unittest.TestCase):
    def test_observed_cells_are_unchanged(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(40, 5))
        frame = pd.DataFrame(values, columns=[f"x{i}" for i in range(5)])
        frame.iloc[0, 0] = np.nan
        frame.iloc[1:6, 2] = np.nan
        frame.iloc[10:15, 4] = np.nan
        observed = frame.notna().to_numpy()

        imputed = IterativeModel(max_iter=8).fit_transform(frame)

        self.assertTrue(np.array_equal(imputed.to_numpy()[observed], frame.to_numpy()[observed]))
        self.assertTrue(np.isfinite(imputed.to_numpy()).all())
        self.assertFalse(imputed.isna().any().any())
        self.assertTrue(imputed.index.equals(frame.index))
        self.assertTrue(imputed.columns.equals(frame.columns))


class IterativeRunnerTests(unittest.TestCase):
    def test_runner_writes_initial_predictions_from_model_input(self):
        frame = pd.DataFrame(
            [[1.0, 2.0], [np.nan, 0.5], [1.5, np.nan], [0.0, -1.0]],
            index=pd.Index(["r0", "r1", "r2", "r3"], name="row_id"),
            columns=["a", "b"])
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            benchmark = root / "processed/credit_card/quick/rate_0.3_seed_0"
            benchmark.mkdir(parents=True)
            matrix = benchmark / MODEL_INPUT
            frame.to_csv(matrix, index_label="row_id")
            metadata = {
                "dataset": "credit_card", "mode": "quick", "rate": 0.3, "seed": 0,
                "features": ["a", "b"], "row_ids": ["r0", "r1", "r2", "r3"],
                "artifact_sha256": {MODEL_INPUT: sha256(matrix)}}
            (benchmark / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            v1_report = root / "v1.json"
            v1_report.write_text(json.dumps({"mean_baseline_rmse": 1.25}), encoding="utf-8")
            seen = {}

            def fake_v1(dataset, mode, rate, seed, predictions, processed_dir, output_dir, model=None, *, run_id="standalone"):
                seen["v1"] = Path(predictions)
                self.assertTrue(Path(predictions).is_file())
                self.assertEqual((dataset, mode, rate, seed, model, run_id), ("credit_card", "quick", 0.3, 0, "iterative", "initial"))
                return {"rmse": 0.4, "mae": 0.2, "r2": 0.5, "relative_improvement": 0.3}, v1_report

            def fake_v2(dataset, mode, rate, seed, predictions, *args, **kwargs):
                seen["v2"] = predictions
                self.assertEqual(kwargs["model"], "iterative")
                self.assertEqual(kwargs["run_id"], "initial")
                self.assertTrue(kwargs["run_downstream"])
                return {
                    "accuracy_and_distribution": {"methods": {"iterative": {
                        "relative_improvement": {"estimate": 0.3, "ci": [0.1, 0.4]},
                        "wasserstein_improvement": {"estimate": 0.5, "ci": [0.4, 0.6]}}}},
                    "downstream": {"skipped": "synthetic fixture has no target rows"},
                }, root / "v2.json"

            with (
                patch.object(shared_evaluation.evaluate, "evaluate_files", side_effect=fake_v1),
                patch.object(shared_evaluation.evaluate_v2, "evaluate_files", side_effect=fake_v2),
            ):
                report = run("credit_card", mode="quick", rate=0.3, seed=0, max_iter=5,
                             processed_dir=root / "processed", output_dir=root / "output",
                             results_dir=root / "results", run_id="initial", n_boot=0)

            expected = prediction_path(root / "output", "credit_card", "quick", 0.3, 0, "initial")
            self.assertEqual(expected, root / "output/credit_card/quick/rate_0.3_seed_0/iterative/initial/predictions.csv")
            self.assertEqual(seen["v1"], expected)
            self.assertEqual(seen["v2"], {"iterative": expected})
            saved = pd.read_csv(expected).set_index("row_id")
            self.assertEqual(list(saved.columns), ["a", "b"])
            self.assertEqual(saved.index.tolist(), ["r0", "r1", "r2", "r3"])
            self.assertEqual(saved.loc["r0", "a"], 1.0)
            self.assertEqual(saved.loc["r1", "b"], 0.5)
            self.assertTrue(np.isfinite(saved.to_numpy()).all())
            self.assertEqual(report["predictions"], expected.as_posix())
            self.assertTrue((root / "results/iterative/experiment/credit_card/quick/rate_0.3_seed_0/initial/experiment.json").is_file())
            summary = pd.read_csv(root / "results/iterative/evaluation/summary/benchmark_summary.csv")
            self.assertEqual(summary["run_id"].tolist(), ["initial"])
            self.assertTrue((benchmark / "X_cleaned_standardized.csv").exists() is False)

            with (
                patch.object(shared_evaluation.evaluate, "evaluate_files", side_effect=fake_v1),
                patch.object(shared_evaluation.evaluate_v2, "evaluate_files", side_effect=fake_v2),
            ):
                with self.assertRaises(FileExistsError):
                    run("credit_card", mode="quick", rate=0.3, seed=0, processed_dir=root / "processed",
                        output_dir=root / "output", results_dir=root / "results", run_id="initial")


if __name__ == "__main__":
    unittest.main()
