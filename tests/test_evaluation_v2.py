"""Offline tests for shared_evaluation version 2: synthetic tables only, no downloads, no model training."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data_preprocessing.dataset_config import DATASETS, config_path
from data_preprocessing.download import sha256
from data_preprocessing.mask import apply_mask, mcar_mask
from data_preprocessing.mean_baseline import mean_impute
from data_preprocessing.prepare_all import export_configuration, read_matrix
from shared_evaluation.evaluate_v2 import (bootstrap, downstream, evaluate, evaluate_files, holm, load_target,
                                           parse_predictions)
from shared_evaluation.metrics_v2 import rmse, wasserstein


def correlated_table(n=400, d=5, seed=0, columns=None):
    rng = np.random.default_rng(seed)
    cov = 0.6 * np.ones((d, d)) + 0.4 * np.eye(d)
    return pd.DataFrame(rng.multivariate_normal(np.zeros(d), cov, size=n), columns=columns or [f'x{j}' for j in range(d)],
                        index=pd.Index([f'source.csv:{i}' for i in range(n)], name='row_id'))


def noisy(X, mask, sd, seed):
    """Truth plus noise at the hidden cells only."""
    noise = np.random.default_rng(seed).normal(scale=sd, size=X.shape)
    return X + np.where(np.asarray(mask, dtype=bool), noise, 0.0)


def prepared(root, dataset, X, raw_target=None):
    """A prepared configuration in root, optionally with a raw cache holding a target column."""
    raw_hashes = {}
    if raw_target is not None:
        raw = root / 'raw' / dataset
        raw.mkdir(parents=True)
        X.reset_index(drop=True).assign(**raw_target).to_csv(raw / 'source.csv', index=False)
        raw_hashes = {'source.csv': sha256(raw / 'source.csv')}
    directory = config_path(root / 'processed', dataset, 'quick', 0.3, 0)
    export_configuration(X, dataset, 'quick', 0.3, 0, directory, {}, raw_hashes)
    truth = read_matrix(directory / 'evaluation_only' / 'X_true_standardized.csv')   # test-only shortcut
    return directory, truth, read_matrix(directory / 'mask_eval.csv').astype(bool)


class EvaluationV2Tests(unittest.TestCase):
    def setUp(self):
        self.X = correlated_table()
        self.mask = mcar_mask(self.X, 0.3, 0)
        self.baseline = mean_impute(apply_mask(self.X, self.mask))

    def test_independent_of_version_1(self):
        blocked = ("import sys; sys.modules['shared_evaluation.metrics'] = None; "
                   "sys.modules['shared_evaluation.evaluate'] = None; import shared_evaluation.evaluate_v2")
        run = subprocess.run([sys.executable, '-c', blocked], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_matches_version_1_while_it_exists(self):
        try:
            from shared_evaluation.evaluate import evaluate as evaluate_v1
        except ImportError:
            self.skipTest('version 1 has been removed')
        good = noisy(self.X, self.mask, 0.3, 1)
        v1, v2 = evaluate_v1(self.X, good, self.mask, self.baseline), evaluate(self.X, good, self.mask, self.baseline)
        self.assertEqual(v1, {k: v2[k] for k in v1})

    def test_wasserstein_basics(self):
        self.assertEqual(wasserstein(self.X, self.X, self.mask), 0.0)
        # a constant fill is a point mass, so its distance is its mean absolute deviation (per-column MAE)
        hidden = self.mask.to_numpy()
        per_column_mae = [np.abs(self.X.to_numpy()[hidden[:, j], j] - self.baseline.to_numpy()[hidden[:, j], j]).mean()
                          for j in range(self.X.shape[1])]
        self.assertAlmostEqual(wasserstein(self.X, self.baseline, self.mask), np.mean(per_column_mae), places=12)
        self.assertEqual(evaluate(self.X, self.baseline, self.mask, self.baseline)['wasserstein_improvement'], 0.0)

    def test_wasserstein_matches_scipy(self):
        from scipy.stats import wasserstein_distance
        good, hidden = noisy(self.X, self.mask, 0.3, 1), self.mask.to_numpy()
        expected = np.mean([wasserstein_distance(good.to_numpy()[hidden[:, j], j], self.X.to_numpy()[hidden[:, j], j])
                            for j in range(self.X.shape[1])])
        self.assertAlmostEqual(wasserstein(self.X, good, self.mask), expected, places=12)

    def test_wasserstein_measures_distribution_not_order(self):
        good = noisy(self.X, self.mask, 0.3, 1)
        shuffled = good.copy()
        rng = np.random.default_rng(2)
        for j in range(shuffled.shape[1]):
            rows = np.flatnonzero(self.mask.iloc[:, j].to_numpy())
            shuffled.iloc[rows, j] = shuffled.iloc[rng.permutation(rows), j].to_numpy()
        self.assertAlmostEqual(wasserstein(self.X, shuffled, self.mask), wasserstein(self.X, good, self.mask), places=12)
        self.assertGreater(rmse(self.X, shuffled, self.mask), 2 * rmse(self.X, good, self.mask))

    def test_wasserstein_bias_is_corrected(self):
        # truth N(0, 1), imputation truth + N(0, 0.4^2): the population distance is (sqrt(1.16) - 1) * sqrt(2 / pi)
        rng = np.random.default_rng(6)
        X = pd.DataFrame(rng.normal(size=(3000, 3)), columns=list('abc'))
        mask = mcar_mask(X, 0.5, 6)
        pred = X + np.where(mask.to_numpy(), rng.normal(scale=0.4, size=X.shape), 0.0)
        population = (np.sqrt(1.16) - 1) * np.sqrt(2 / np.pi)
        w = bootstrap(X, {'m': pred}, mask, mean_impute(apply_mask(X, mask)), n_boot=300)['methods']['m']['wasserstein']
        self.assertGreater(w['estimate'], population)          # the plug-in value is biased upward
        self.assertGreater(w['bias'], 0)                       # and the bootstrap sees it
        self.assertTrue(w['ci'][0] <= population <= w['ci'][1])
        self.assertTrue(w['ci'][0] <= w['bias_corrected'] <= w['ci'][1])

    def test_bootstrap_point_estimates_match_evaluate(self):
        good = noisy(self.X, self.mask, 0.3, 1)
        result = bootstrap(self.X, {'good': good}, self.mask, self.baseline, n_boot=0)
        for key, value in evaluate(self.X, good, self.mask, self.baseline).items():
            self.assertAlmostEqual(result['methods']['good'][key]['estimate'], value, places=12)
            self.assertEqual(result['methods']['good'][key]['ci'], [None, None])
        self.assertEqual(result['methods']['mean']['relative_improvement']['estimate'], 0.0)
        self.assertEqual(result['rows_resampled'], int(self.mask.to_numpy().any(axis=1).sum()))

    def test_bootstrap_is_reproducible_and_paired(self):
        good = noisy(self.X, self.mask, 0.3, 1)
        first = bootstrap(self.X, {'a': good, 'b': good.copy()}, self.mask, self.baseline, n_boot=50, seed=3)
        self.assertEqual(first, bootstrap(self.X, {'a': good, 'b': good.copy()}, self.mask, self.baseline, n_boot=50, seed=3))
        same = first['differences']['a - b']['rmse']
        self.assertEqual((same['estimate'], same['ci'], same['p_value']), (0.0, [0.0, 0.0], 1.0))

    def test_bootstrap_detects_a_better_method(self):
        good, bad = noisy(self.X, self.mask, 0.1, 1), noisy(self.X, self.mask, 1.0, 2)
        result = bootstrap(self.X, {'good': good, 'bad': bad}, self.mask, self.baseline, n_boot=200)
        diff = result['differences']['good - bad']['rmse']
        self.assertLess(diff['ci'][1], 0)
        self.assertLess(diff['p_value'], 0.05)
        low, high = result['methods']['good']['rmse']['ci']
        self.assertTrue(low <= result['methods']['good']['rmse']['estimate'] <= high)
        self.assertGreater(result['methods']['good']['relative_improvement']['ci'][0], 0)

    def test_bootstrap_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bootstrap(self.X, {'rev': self.X.iloc[::-1]}, self.mask, self.baseline, n_boot=5)
        with self.assertRaises(ValueError):
            bootstrap(self.X, {'mean': self.X}, self.mask, self.baseline, n_boot=5)

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])

    def test_downstream_regression_gap(self):
        beta = np.array([1.0, -0.5, 0.8, 0.0, 0.3])
        y = pd.Series(self.X.to_numpy() @ beta + np.random.default_rng(4).normal(scale=0.5, size=len(self.X)),
                      index=self.X.index)
        tables = {'good': noisy(self.X, self.mask, 0.3, 1), 'mean': self.baseline, 'true': self.X}
        result = downstream(tables, y, 'regression', n_boot=50)
        gap = {name: m['gap_closed']['estimate'] for name, m in result['methods'].items()}
        self.assertAlmostEqual(gap['true'], 1.0)
        self.assertAlmostEqual(gap['mean'], 0.0)
        self.assertTrue(0 < gap['good'] < 1)
        self.assertEqual(result['score_name'], 'r2')

    def test_downstream_classification(self):
        y = pd.Series((self.X.to_numpy() @ np.ones(5) + np.random.default_rng(5).normal(size=len(self.X)) > 0).astype(int),
                      index=self.X.index)
        result = downstream({'mean': self.baseline, 'true': self.X}, y, 'classification', n_boot=20)
        self.assertEqual(result['score_name'], 'auroc')
        self.assertTrue(0.5 < result['methods']['true']['score']['estimate'] <= 1)
        with self.assertRaises(ValueError):
            downstream({'true': self.X}, y, 'classification', n_boot=5)

    def test_load_target(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            folder = Path(temp) / 'credit_card'
            folder.mkdir()
            pd.DataFrame({'X1': [10., 20., 30., 40.], 'Y': [0, 1, 1, 0]}).to_csv(folder / 'source.csv', index=False)
            digest = {'source.csv': sha256(folder / 'source.csv')}
            y = load_target('credit_card', ['source.csv:3', 'source.csv:1'], temp, digest)
            self.assertEqual(y.tolist(), [0.0, 1.0])
            self.assertEqual(y.index.tolist(), ['source.csv:3', 'source.csv:1'])
            with self.assertRaises(ValueError):
                load_target('credit_card', ['source.csv:0'], temp, {'source.csv': '0' * 64})
            with self.assertRaises(FileNotFoundError):
                load_target('superconductivity', ['source.csv:0'], temp)
            with self.assertRaises(ValueError):
                load_target('gas_turbine', ['gt_2011.csv:0'], temp)

    def test_parse_predictions(self):
        parsed = parse_predictions(['data/output/x/dae/predictions.csv', 'vae=other/predictions.csv'])
        self.assertEqual(parsed, {'dae': 'data/output/x/dae/predictions.csv', 'vae': 'other/predictions.csv'})
        with self.assertRaises(ValueError):
            parse_predictions(['a/dae/predictions.csv', 'b/dae/predictions.csv'])

    def test_files_and_command_line(self):
        features = list(DATASETS['california_housing'].features)
        X = correlated_table(n=300, d=len(features), columns=features)
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            directory, truth, mask = prepared(root, 'california_housing', X,
                                              {'MedHouseVal': X.to_numpy() @ np.linspace(1, -1, len(features))})
            (root / 'dae').mkdir()
            noisy(truth, mask, 0.3, 1).to_csv(root / 'dae' / 'predictions.csv', index_label='row_id')
            cli = subprocess.run([sys.executable, '-m', 'shared_evaluation.evaluate_v2', '--dataset', 'california_housing',
                                  '--predictions', str(root / 'dae' / 'predictions.csv'), '--n-boot', '20',
                                  '--downstream', '--raw-dir', str(root / 'raw'),
                                  '--processed-dir', str(root / 'processed'), '--output-dir', str(root / 'output')],
                                 capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            saved = sorted((root / 'output').rglob('evaluation_v2_*.json'))
            self.assertEqual(len(saved), 1)
            report = json.loads(saved[0].read_text())
            self.assertEqual(set(report['accuracy_and_distribution']['methods']), {'dae', 'mean'})
            self.assertEqual(set(report['downstream']['methods']), {'dae', 'mean', 'true'})
            self.assertIn('gap closed', cli.stdout)

    def test_downstream_ignores_observed_cells_a_model_rewrote(self):
        features = list(DATASETS['california_housing'].features)
        X = correlated_table(n=300, d=len(features), columns=features)
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            _, truth, mask = prepared(root, 'california_housing', X,
                                      {'MedHouseVal': X.to_numpy() @ np.linspace(1, -1, len(features))})
            copied = noisy(truth, mask, 0.3, 1)
            rewritten = copied.where(mask, copied + 5.0)   # same guesses, observed cells changed
            copied.to_csv(root / 'copied.csv', index_label='row_id')
            rewritten.to_csv(root / 'rewritten.csv', index_label='row_id')
            report, _ = evaluate_files('california_housing', 'quick', 0.3, 0,
                                       {'copied': root / 'copied.csv', 'rewritten': root / 'rewritten.csv'},
                                       root / 'processed', root / 'output', n_boot=10, run_downstream=True,
                                       raw_dir=root / 'raw')
            scores = report['downstream']['methods']
            self.assertEqual(scores['copied']['score']['estimate'], scores['rewritten']['score']['estimate'])

    def test_dataset_without_target_skips_downstream(self):
        features = list(DATASETS['gas_turbine'].features)
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            directory, _, _ = prepared(root, 'gas_turbine', correlated_table(n=200, d=len(features), columns=features))
            report, _ = evaluate_files('gas_turbine', 'quick', 0.3, 0, {'m': directory / 'X_mean_baseline.csv'},
                                       root / 'processed', root / 'output', n_boot=5, run_downstream=True)
            self.assertIn('skipped', report['downstream'])


if __name__ == '__main__':
    unittest.main()
