"""Results layout coverage without retraining the published models."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from models.deep_learning_v2.run_experiment import run
from shared_evaluation.evaluate import evaluate_files
from shared_evaluation.result_paths import evaluation_directory, write_report


class ResultPathsTests(unittest.TestCase):
    def test_safe_paths_and_collision(self):
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            target = evaluation_directory(root, 'knn', 'credit_card', 'quick', .3, 0, 'run-1')
            self.assertEqual(target, root / 'knn/evaluation/credit_card/quick/rate_0.3_seed_0/run-1')
            for model, run_id in (('../knn', 'ok'), ('knn', '..'), ('summary', 'ok'), ('CON', 'ok')):
                with self.assertRaises(ValueError):
                    evaluation_directory(root, model, 'credit_card', 'quick', .3, 0, run_id)
            report = target / 'evaluation_v1_test.json'
            write_report(report, {'rmse': 1})
            write_report(report, {'rmse': 1})
            with self.assertRaises(FileExistsError):
                write_report(report, {'rmse': 2})
            self.assertEqual(json.loads(report.read_bytes()), {'rmse': 1})

    def test_runner_writes_one_evaluation(self):
        benchmark = Path('data/processed/credit_card/quick/rate_0.3_seed_0')
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            results = root / 'results'
            fake_summary = {'best_epoch': 1, 'epochs_completed': 1, 'best_validation_loss': .1,
                            'runtime_seconds': 0, 'device': 'cpu'}
            with (
                patch('models.deep_learning_v2.run_experiment.fit', side_effect=lambda x, _: (
                    SimpleNamespace(), fake_summary, [{'epoch': 1}], x.isna().to_numpy())),
                patch('models.deep_learning_v2.run_experiment.predict', side_effect=lambda m, x, *args: x.fillna(0)),
                patch('models.deep_learning_v2.run_experiment.save_checkpoint',
                      side_effect=lambda path, *args: path.write_bytes(b'mock')),
            ):
                report = run('credit_card', results_dir=results, output_dir=root / 'predictions', run_id='smoke')
            relative = Path('credit_card/quick/rate_0.3_seed_0/smoke')
            training = results / 'dae_v2/training' / relative
            self.assertTrue((training / 'experiment.json').is_file())
            self.assertEqual(Path(report['training_summary']), training / 'training_summary.json')
            self.assertTrue((results / 'dae_v2/training/checkpoints' / relative / 'best.pt').is_file())
            summary = pd.read_csv(results / 'dae_v2/evaluation/summary/benchmark_summary.csv')
            self.assertEqual(summary['run_id'].tolist(), ['smoke'])
            out = results / 'dae_v2/evaluation' / relative
            self.assertEqual(Path(report['evaluation']).parent, out)
            self.assertTrue(Path(report['evaluation']).name.startswith('evaluation_v2_'))
            predictions = Path(report['predictions'])
            _, repeated = evaluate_files('credit_card', 'quick', .3, 0, {'dae_v2': predictions},
                                         output_dir=results, n_boot=0, model='dae_v2', run_id='smoke')
            self.assertEqual(repeated, Path(report['evaluation']))
            _, standalone = evaluate_files('credit_card', 'quick', .3, 0, {'mean_probe': benchmark / 'X_mean_baseline.csv'},
                                           output_dir=root / 'standalone', n_boot=0)
            self.assertEqual(standalone.parent, root / 'standalone/credit_card/quick/rate_0.3_seed_0')
            self.assertEqual({p.name for p in results.iterdir()}, {'dae_v2'})
