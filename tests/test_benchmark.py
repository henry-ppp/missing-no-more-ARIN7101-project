"""Offline tests only: no external datasets, downloads or model training."""
import importlib
import json
import io
import subprocess
import sys
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from data_preprocessing.clean import clean, select_rows
from data_preprocessing.dataset_config import DATASETS, config_path
from data_preprocessing.mask import mcar_mask, apply_mask
from data_preprocessing.standardize import fit_observed, transform
from data_preprocessing.mean_baseline import mean_impute
from data_preprocessing.download import download, load_raw
from data_preprocessing.prepare_all import export_configuration, load_model_input, prepare
from shared_evaluation.evaluate import evaluate, evaluate_files
from shared_evaluation.metrics import r2

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.default_rng(9).normal(size=(100, 8)),
            columns=DATASETS['california_housing'].features,
            index=pd.Index([f'row:{i}' for i in range(100)], name='row_id'))

    def test_mask_sampling(self):
        pd.testing.assert_frame_equal(mcar_mask(self.X, .3, 0), mcar_mask(self.X, .3, 0))
        big = pd.DataFrame({'a': range(4000)})
        pd.testing.assert_frame_equal(select_rows(big, 'quick', 3), select_rows(big, 'quick', 3))
        self.assertEqual(len(select_rows(big, 'quick')), 3000)
        self.assertEqual(len(select_rows(big, 'full')), 4000)
        pd.testing.assert_frame_equal(select_rows(self.X, 'quick'), self.X)
        with self.assertRaises(ValueError):
            apply_mask(self.X, mcar_mask(self.X, .3).iloc[::-1])

    def test_clean(self):
        raw = self.X.copy()
        raw.iloc[0, 0] = np.inf
        raw.iloc[1, 1] = np.nan
        result, meta = clean('california_housing', raw)
        self.assertEqual(len(result), 98)
        self.assertEqual(meta['dropped_invalid_rows'], 2)
        raw.iloc[:, 0] = 1
        with self.assertRaises(ValueError):
            clean('california_housing', raw)
        credit = DATASETS['credit_card']
        aliases = ['X1', 'X5'] + [f'X{i}' for i in range(12, 24)]
        raw = pd.DataFrame(np.arange(42).reshape(3, 14), columns=aliases)
        result, _ = clean('credit_card', raw)
        self.assertEqual(tuple(result.columns), credit.features)

    def test_scaling_no_leak(self):
        mask = mcar_mask(self.X, .3)
        missing = apply_mask(self.X, mask)
        stats = fit_observed(missing)
        changed = self.X.mask(mask, 1e12)
        self.assertEqual(stats, fit_observed(apply_mask(changed, mask)))
        z = transform(missing, stats)
        np.testing.assert_allclose(z.mean(), 0, atol=1e-14)
        np.testing.assert_allclose(z.std(ddof=0), 1)
        pd.testing.assert_frame_equal(z.isna(), mask)
        baseline = mean_impute(z)
        np.testing.assert_allclose(baseline.to_numpy()[mask.to_numpy()], 0, atol=1e-14)
        np.testing.assert_allclose(transform(transform(self.X, stats), stats, inverse=True), self.X)
        constant = pd.DataFrame({'x': [1., 1., np.nan]})
        self.assertEqual(fit_observed(constant)['scale'], [1.])
        with self.assertRaises(ValueError):
            fit_observed(pd.DataFrame({'x': [np.nan]}))

    def test_metrics(self):
        truth = pd.DataFrame({'a': [1., 3., 5.], 'b': [2., 4., 8.]})
        mask = truth.notna()
        result = evaluate(truth, truth + 1, mask, truth + 2)
        self.assertEqual(result['rmse'], 1.)
        self.assertEqual(result['mae'], 1.)
        self.assertEqual(result['relative_improvement'], .5)
        expected = np.mean([1 - 3 / np.square(truth[c] - truth[c].mean()).sum() for c in truth])
        self.assertAlmostEqual(result['r2'], expected)
        self.assertIsNone(evaluate(truth, truth, mask, truth)['relative_improvement'])
        self.assertIsNone(r2(truth * 0, truth, mask))
        for invalid in [truth.iloc[::-1], truth[['b', 'a']], truth * np.nan]:
            with self.assertRaises(ValueError):
                evaluate(truth, invalid, mask, truth + 1)
        with self.assertRaises(ValueError):
            evaluate(truth, truth, mask * 2, truth)
        with self.assertRaises(ValueError):
            evaluate(truth, truth, mask & False, truth)
        with self.assertRaises(ValueError):
            evaluate(truth, truth, mask, truth * np.nan)

    def test_file_contract(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            processed = root / 'processed'
            directory = config_path(processed, 'california_housing', 'quick', .3, 0)
            export_configuration(self.X, 'california_housing', 'quick', .3, 0, directory, {}, {})
            with patch('data_preprocessing.prepare_all.read_matrix', wraps=__import__('data_preprocessing.prepare_all', fromlist=['read_matrix']).read_matrix) as reader:
                model_input = load_model_input(directory)
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(reader.call_args.args[0].name, 'X_missing.csv')
            self.assertTrue(model_input.isna().any().any())
            predictions = root / 'mean_predictions.csv'
            model_input.fillna(model_input.mean()).to_csv(predictions)
            results, output = evaluate_files('california_housing', 'quick', .3, 0, predictions, processed, root / 'output')
            self.assertAlmostEqual(results['relative_improvement'], 0, places=12)
            self.assertTrue(output.exists())
            cli = subprocess.run([sys.executable, '-m', 'shared_evaluation.evaluate',
                '--dataset', 'california_housing', '--predictions', str(predictions),
                '--processed-dir', str(processed), '--output-dir', str(root / 'cli-output')],
                capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertEqual(set(json.loads(output.read_text())['metrics']), {'rmse', 'mae', 'r2', 'relative_improvement'})
            with patch('data_preprocessing.prepare_all.download', side_effect=AssertionError('Must reuse cache')):
                self.assertEqual(prepare('california_housing', processed_dir=processed), directory)
            (directory / 'mask_eval.csv').write_text('tampered')
            with self.assertRaises(ValueError):
                evaluate_files('california_housing', 'quick', .3, 0, predictions, processed, root / 'output')

    def test_gas_archive_offline(self):
        payload = io.BytesIO()
        config = DATASETS['gas_turbine']
        with ZipFile(payload, 'w') as archive:
            for year in range(2011, 2016):
                frame = pd.DataFrame(np.arange(22).reshape(2, 11) + year, columns=config.features)
                archive.writestr(f'nested/gt_{year}.csv', frame.to_csv(index=False))
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            with patch('data_preprocessing.download.urlopen', return_value=io.BytesIO(payload.getvalue())) as network:
                download('gas_turbine', temp)
                self.assertEqual(network.call_count, 1)
            combined, hashes = load_raw('gas_turbine', temp)
            self.assertEqual(combined.shape, (10, 11))
            self.assertEqual(len(hashes), 5)
            self.assertEqual(combined.index[0], 'gt_2011.csv:0')
            self.assertEqual(combined.index[-1], 'gt_2015.csv:1')
            with patch('data_preprocessing.download.urlopen', side_effect=AssertionError('No refetch')):
                download('gas_turbine', temp)

    def test_imports(self):
        for name in ('models.knn', 'models.iterative', 'models.random_forest', 'models.deep_learning_v2'):
            importlib.import_module(name)
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed in this verification runtime')
        for name in ('models.deep_learning', 'models.deep_learning.methods', 'models.deep_learning.run_benchmark', 'models.deep_learning.run_experiment'):
            importlib.import_module(name)

if __name__ == '__main__':
    unittest.main()
