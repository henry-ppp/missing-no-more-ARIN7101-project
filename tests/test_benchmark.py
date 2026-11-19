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
from data_preprocessing.prepare_all import export_configuration, load_model_input, prepare, read_matrix
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
                self.assertEqual(reader.call_args.args[0].name, 'X_cleaned_masked_standardized.csv')
            self.assertTrue(model_input.isna().any().any())
            predictions = root / 'mean_predictions.csv'
            model_input.fillna(model_input.mean()).to_csv(predictions)
            results, output = evaluate_files('california_housing', 'quick', .3, 0, {'column_mean': predictions},
                                             processed, root / 'output', n_boot=0, model='column_mean')
            point = results['accuracy_and_distribution']['methods']['column_mean']
            self.assertAlmostEqual(point['relative_improvement']['estimate'], 0, places=12)
            self.assertTrue(output.exists())
            cli = subprocess.run([sys.executable, '-m', 'shared_evaluation.evaluate',
                '--dataset', 'california_housing', '--predictions', f'column_mean={predictions}',
                '--n-boot', '0', '--model', 'column_mean', '--processed-dir', str(processed),
                '--output-dir', str(root / 'cli-output')],
                capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertEqual(set(point), {'rmse', 'mae', 'r2', 'relative_improvement', 'wasserstein', 'wasserstein_improvement'})
            with patch('data_preprocessing.prepare_all.download', side_effect=AssertionError('Must reuse cache')):
                self.assertEqual(prepare('california_housing', processed_dir=processed), directory)
            (directory / 'mask_eval.csv').write_text('tampered')
            with self.assertRaises(ValueError):
                evaluate_files('california_housing', 'quick', .3, 0, {'column_mean': predictions},
                               processed, root / 'output', n_boot=0, model='column_mean')

    def test_fresh_layout_in_both_orders(self):
        dataset = 'california_housing'
        raw = pd.DataFrame(np.random.default_rng(7).normal(size=(3100, 8)), columns=self.X.columns)
        expected = {'X_cleaned.csv', 'X_cleaned_masked.csv', 'X_cleaned_masked_standardized.csv',
                    'X_cleaned_standardized.csv', 'mask_eval.csv', 'X_mean_baseline.csv',
                    'scaler_stats.json', 'metadata.json'}
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            raw_file = root / 'raw' / dataset / 'source.csv'
            raw_file.parent.mkdir(parents=True)
            raw.to_csv(raw_file, index=False)
            for order in (('quick', 'full'), ('full', 'quick')):
                processed = root / order[0]
                for mode in order:
                    target = prepare(dataset, mode, raw_dir=root / 'raw', processed_dir=processed,
                                     skip_download=True)
                    self.assertEqual({path.name for path in target.iterdir()}, expected)
                    complete = read_matrix(target / 'X_cleaned.csv')
                    self.assertEqual(len(complete), 3000 if mode == 'quick' else 3100)
                    mask = read_matrix(target / 'mask_eval.csv').astype(bool)
                    pd.testing.assert_frame_equal(mask, mcar_mask(complete, .3, 0))
                    masked = read_matrix(target / 'X_cleaned_masked.csv')
                    pd.testing.assert_frame_equal(masked, complete.mask(mask))
                    stats = json.loads((target / 'scaler_stats.json').read_text())
                    for key in ('mean', 'scale'):
                        np.testing.assert_allclose(stats[key], fit_observed(masked)[key], atol=1e-14)
                    pd.testing.assert_frame_equal(load_model_input(target), transform(masked, stats))
                    pd.testing.assert_frame_equal(read_matrix(target / 'X_cleaned_standardized.csv'),
                                                  transform(complete, stats))
                full = config_path(processed, dataset, 'full', .3, 0)
                canonical = processed / dataset / 'cleaned' / 'X_cleaned.csv'
                self.assertEqual(canonical.read_bytes(), (full / 'X_cleaned.csv').read_bytes())
                self.assertEqual({p.name for p in canonical.parent.parent.iterdir()}, {'cleaned', 'quick', 'full'})

    def test_published_dae_scores_after_layout_migration(self):
        repo = Path(__file__).resolve().parents[1]
        processed = repo / 'data' / 'processed'
        # Datasets with published DAE runs. HMDA was added after those runs, so it is absent here.
        published = sorted({p.relative_to(repo / 'data' / 'output').parts[0] for p in
                            (repo / 'data' / 'output').glob('*/*/rate_0.3_seed_0/dae_v2/initial/predictions.csv')}
                           & set(DATASETS))
        self.assertTrue(published)
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            for dataset in published:
                for mode in ('quick', 'full'):
                    with self.subTest(dataset=dataset, mode=mode):
                        relative = config_path(Path(), dataset, mode, .3, 0)
                        predictions = repo / 'data' / 'output' / relative / 'dae_v2' / 'initial' / 'predictions.csv'
                        reports = repo / 'results' / 'dae_v2' / 'evaluation' / relative / 'initial'
                        historical, = reports.glob('evaluation_v1_*.json')
                        expected = json.loads(historical.read_text(encoding='utf-8'))
                        saved, _ = evaluate_files(dataset, mode, .3, 0, {'dae_v2': predictions}, processed, temp,
                                                  n_boot=0, model='dae_v2', run_id='initial')
                        point = saved['accuracy_and_distribution']['methods']['dae_v2']
                        for name, score in expected['metrics'].items():
                            self.assertAlmostEqual(point[name]['estimate'], score, places=13)
                        self.assertAlmostEqual(saved['accuracy_and_distribution']['methods']['mean']['rmse']['estimate'],
                                               expected['mean_baseline_rmse'], places=13)

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
