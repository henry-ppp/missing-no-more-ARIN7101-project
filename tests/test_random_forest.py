"""Offline tests for models.random_forest: synthetic tables, tiny forests, temporary folders only."""
import json
import sys
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from data_preprocessing.dataset_config import DATASETS, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import export_configuration
from models.random_forest import MiceRFImputer, MissForestImputer, impute_mice_rf, impute_missforest
from models.random_forest.imputers import LeanIterativeImputer
from models.random_forest.run_experiment import Paths, Settings, Tuning, hyperparameter_path, plan, run
from models.random_forest.seeds import derive_seed, stream
from models.random_forest.tuning import candidates, hide_and_check, tuning_sample

TINY = dict(n_trees=8, max_rounds=3)


def correlated(n=300, d=5, seed=0):
    cov = 0.7 * np.ones((d, d)) + 0.3 * np.eye(d)
    return np.random.default_rng(seed).multivariate_normal(np.zeros(d), cov, size=n)


def with_gaps(X, rate=0.3, seed=1):
    mask = np.random.default_rng(seed).random(X.shape) < rate
    return np.where(mask, np.nan, X), mask


class Seeds(unittest.TestCase):
    def test_streams_are_stable_and_separate(self):
        self.assertEqual(derive_seed(0, 'forest', 1, 2), derive_seed(0, 'forest', 1, 2))
        self.assertNotEqual(derive_seed(0, 'forest', 1, 2), derive_seed(0, 'forest', 2, 1))
        self.assertNotEqual(derive_seed(0, 'tuning'), derive_seed(1, 'tuning'))
        self.assertTrue(0 <= derive_seed(0, 'x') < 2**31 - 1)
        np.testing.assert_array_equal(stream(0, 'rows').random(3), stream(0, 'rows').random(3))


class Imputers(unittest.TestCase):
    def setUp(self):
        self.X = correlated()
        self.X_missing, self.mask = with_gaps(self.X)

    def test_both_keep_visible_values_fill_every_gap_and_beat_the_mean(self):
        mean_rmse = np.sqrt(np.mean((np.nanmean(self.X_missing, axis=0) * np.ones_like(self.X) - self.X)[self.mask] ** 2))
        for impute in (impute_missforest, impute_mice_rf):
            with self.subTest(impute.__name__):
                filled, info = impute(self.X_missing, 0, n_trees=25, max_rounds=3)
                np.testing.assert_array_equal(filled[~self.mask], self.X[~self.mask])
                self.assertTrue(np.isfinite(filled).all())
                self.assertLess(np.sqrt(np.mean((filled - self.X)[self.mask] ** 2)), 0.9 * mean_rmse)
                self.assertEqual([h['round'] for h in info.history], list(range(1, len(info.history) + 1)))
                np.testing.assert_allclose(info.importances.sum(axis=1), 1.0)
                np.testing.assert_array_equal(np.diag(info.importances), 0.0)

    def test_same_seed_same_result_other_seed_differs(self):
        for impute in (impute_missforest, impute_mice_rf):
            with self.subTest(impute.__name__):
                a, _ = impute(self.X_missing, 0, **TINY)
                b, _ = impute(self.X_missing, 0, **TINY)
                c, _ = impute(self.X_missing, 1, **TINY)
                np.testing.assert_array_equal(a, b)
                self.assertFalse(np.array_equal(a, c))

    def test_missforest_keeps_the_round_before_the_change_grows(self):
        _, info = impute_missforest(self.X_missing, 0, n_trees=8, max_rounds=10)
        changes = [h['change'] for h in info.history]
        if info.stopped_early:      # the change fell every round until the last, where it grew
            self.assertGreaterEqual(changes[-1], changes[-2])
            self.assertEqual(info.kept_round, len(changes) - 1)
            self.assertTrue(all(changes[i] < changes[i - 1] for i in range(1, len(changes) - 1)))
        else:
            self.assertEqual(info.kept_round, 10)

    def test_lean_iterative_imputer_gives_identical_output(self):
        estimator = ExtraTreesRegressor(n_estimators=10, max_features=0.5, min_samples_leaf=2, n_jobs=1, random_state=3)
        options = dict(max_iter=4, skip_complete=True, n_nearest_features=3, random_state=4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain = IterativeImputer(estimator=estimator, **options).fit_transform(self.X_missing)
            lean = LeanIterativeImputer(estimator=estimator, **options)
            np.testing.assert_array_equal(lean.fit_transform(self.X_missing), plain)
        self.assertEqual(len(lean.history_), lean.n_iter_)

    def test_dataframe_wrappers_keep_labels(self):
        frame = pd.DataFrame(self.X_missing, columns=list('abcde'), index=[f'r{i}' for i in range(len(self.X))])
        for imputer in (MissForestImputer(seed=0, **TINY), MiceRFImputer(seed=0, **TINY)):
            filled = imputer.fit_transform(frame)
            pd.testing.assert_index_equal(filled.index, frame.index)
            pd.testing.assert_index_equal(filled.columns, frame.columns)
            self.assertIsNotNone(imputer.info_)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            impute_missforest(np.array([[1.0, np.inf], [2.0, 3.0], [1.0, 1.0]]), 0)
        with self.assertRaises(ValueError):
            impute_missforest(np.array([[np.nan, 1.0], [np.nan, 2.0], [1.0, 3.0]]), 0)


class TuningSteps(unittest.TestCase):
    def test_candidates_are_the_full_two_by_two_grid_with_the_default_first(self):
        grid = candidates('MissForest', 14)
        self.assertEqual(grid[0], {'max_features': 'sqrt', 'min_samples_leaf': 1})
        self.assertEqual(len(grid), 4)
        self.assertEqual(len({json.dumps(g, sort_keys=True) for g in grid}), 4)
        self.assertEqual(candidates('MICE-RF', 14)[0], {'max_features': 0.5, 'min_samples_leaf': 2})
        self.assertEqual(len(candidates('MICE-RF', 14)), 4)
        # with 8 columns, sqrt(7) and 7/3 both give 2 features per split, so MissForest has only 2 settings
        self.assertEqual(candidates('MissForest', 8), [{'max_features': 'sqrt', 'min_samples_leaf': 1},
                                                       {'max_features': 'sqrt', 'min_samples_leaf': 5}])

    def test_hide_and_check_only_hides_visible_cells(self):
        X_missing, mask = with_gaps(correlated(n=500))
        hidden, check = hide_and_check(X_missing, seed=0, share=0.1)
        self.assertFalse((check & mask).any())
        self.assertTrue(np.isnan(hidden[check]).all())
        np.testing.assert_array_equal(np.isnan(hidden), mask | check)
        self.assertAlmostEqual(check.sum() / (~mask).sum(), 0.1, delta=0.02)
        self.assertTrue((~np.isnan(hidden)).any(axis=1)[(~mask).any(axis=1)].all())

    def test_tuning_sample(self):
        X = correlated(n=500)
        self.assertEqual(len(tuning_sample(X, 0, 100)), 100)
        np.testing.assert_array_equal(tuning_sample(X, 0, 100), tuning_sample(X, 0, 100))
        self.assertIs(tuning_sample(X, 0, 1000), X)
        self.assertIs(tuning_sample(X, 0, None), X)


QUIET = dict(workers=1, log=lambda *_: None)
SMALL = Settings(n_trees=5, max_rounds=2, n_boot=10)
TUNE = Tuning(rows=150)


class Experiment(unittest.TestCase):
    """The whole pipeline on a tiny prepared benchmark, written to a temporary folder."""

    def prepare(self, root, rates=(0.3,)):
        features = list(DATASETS['california_housing'].features)
        X = pd.DataFrame(correlated(n=240, d=len(features)), columns=features,
                         index=pd.Index([f'source.csv:{i}' for i in range(240)], name='row_id'))
        raw = root / 'raw' / 'california_housing'
        raw.mkdir(parents=True)
        X.reset_index(drop=True).assign(MedHouseVal=X.to_numpy() @ np.linspace(1, -1, len(features))).to_csv(
            raw / 'source.csv', index=False)
        for rate in rates:
            export_configuration(X, 'california_housing', 'full', rate, 0,
                                 config_path(root / 'processed', 'california_housing', 'full', rate, 0), {},
                                 {'source.csv': sha256(raw / 'source.csv')})
        return Paths(root / 'processed', root / 'raw', root / 'output', root / 'results' / 'random_forest',
                     root / 'local_models' / 'random_forest')

    @staticmethod
    def trial_files(paths):
        return sorted(paths.local.rglob('candidate_*.json'))

    def test_runs_saves_evaluates_and_resumes(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = self.prepare(Path(temp))
            jobs = plan(['MissForest', 'MICE-RF'], ['california_housing'], 'full', None, 0, 'test', paths)
            self.assertEqual([(j.model, j.rate) for j in jobs], [('MissForest', 0.3), ('MICE-RF', 0.3)])
            # Evaluation version 1 is blocked: the pipeline must only need version 2 (one worker runs in this process).
            with patch.dict(sys.modules, {'shared_evaluation.evaluate': None, 'shared_evaluation.metrics': None}):
                experiments = run(jobs, SMALL, TUNE, paths, **QUIET)
            for job, experiment in zip(jobs, experiments):
                with self.subTest(job.model):
                    self.assertEqual(experiment['seed'], 0)
                    self.assertTrue(job.prediction_path(paths).is_file())
                    for name in ('training_history.csv', 'importances.csv', 'run.json', 'settings.json'):
                        self.assertTrue((job.local_dir(paths) / name).is_file(), name)
                    saved = json.loads(hyperparameter_path(job.model, 'california_housing', 'full', paths).read_text())
                    self.assertEqual(saved['tuned_on']['rate'], 0.3)
                    self.assertEqual(experiment['params'], saved['params'])
                    self.assertEqual(experiment['hyperparameters']['source'], 'tuned')
                    self.assertEqual(len(saved['candidates']), len(candidates(job.model, 8)))
                    reports = list((paths.results / job.model / 'evaluation').rglob('evaluation_v*.json'))
                    self.assertEqual([p.name[:13] for p in reports], ['evaluation_v2'])
                    self.assertIn('gap_closed', experiment['headline'])
                    summary = pd.read_csv(paths.results / job.model / 'evaluation' / 'summary' /
                                          'benchmark_summary.csv')
                    self.assertTrue({'rmse', 'mae', 'r2', 'wasserstein', 'relative_improvement'} <= set(summary))
                    self.assertFalse([c for c in summary if c.startswith('v1_')])
                    predictions = pd.read_csv(job.prediction_path(paths), index_col='row_id')
                    self.assertTrue(np.isfinite(predictions.to_numpy()).all())
            self.assertEqual(len(self.trial_files(paths)), 2 + 4)      # MissForest 2 settings, MICE-RF 4
            paired, = (paths.results / 'comparison' / 'evaluation').rglob('evaluation_v2_*.json')
            differences = json.loads(paired.read_text())['accuracy_and_distribution']['differences']
            self.assertIn('MissForest - MICE-RF', differences)

            stamps = {p: p.stat().st_mtime_ns for p in Path(temp).rglob('*') if p.is_file()}
            run(jobs, SMALL, TUNE, paths, **QUIET)                       # everything is already done
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in Path(temp).rglob('*') if p.is_file()})

            with self.assertRaisesRegex(FileExistsError, 'other settings'):
                run(jobs, Settings(n_trees=6, max_rounds=2, n_boot=10), TUNE, paths, **QUIET)
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in Path(temp).rglob('*') if p.is_file()})

    def test_saved_hyperparameters_are_reused_for_another_rate(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = self.prepare(Path(temp), rates=(0.1, 0.3))
            first, = run(plan(['MICE-RF'], ['california_housing'], 'full', [0.3], 0, 'a', paths), SMALL, TUNE,
                         paths, **QUIET)
            tuned = self.trial_files(paths)
            later, = run(plan(['MICE-RF'], ['california_housing'], 'full', [0.1], 0, 'b', paths), SMALL, TUNE,
                         paths, **QUIET)
            self.assertEqual(self.trial_files(paths), tuned)              # no tuning the second time
            self.assertEqual(later['params'], first['params'])
            self.assertEqual(later['hyperparameters']['tuned_on']['rate'], 0.3)

    def test_two_rates_in_one_run_are_tuned_once(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = self.prepare(Path(temp), rates=(0.1, 0.3))
            experiments = run(plan(['MICE-RF'], ['california_housing'], 'full', None, 0, 'both', paths), SMALL,
                              TUNE, paths, **QUIET)
            self.assertEqual({e['rate'] for e in experiments}, {0.1, 0.3})
            self.assertEqual({e['hyperparameters']['tuned_on']['rate'] for e in experiments}, {0.3})
            self.assertEqual({p.parent.name for p in self.trial_files(paths)}, {'rate_0.3_seed_0'})

    def test_without_tuning_uses_the_defaults(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = self.prepare(Path(temp))
            jobs = plan(['MissForest'], ['california_housing'], 'full', [0.3], 0, 'defaults', paths)
            experiment, = run(jobs, SMALL, Tuning(enabled=False), paths, **QUIET)
            self.assertEqual(experiment['params'], {'max_features': 'sqrt', 'min_samples_leaf': 1})
            self.assertEqual(experiment['hyperparameters'], {'source': 'defaults'})
            self.assertEqual(self.trial_files(paths), [])

    def test_missing_rate_or_mode_is_reported(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            paths = self.prepare(Path(temp))
            with self.assertRaisesRegex(FileNotFoundError, 'no prepared rate 0.5'):
                plan(['MissForest'], ['california_housing'], 'full', [0.5], 0, 'x', paths)
            with self.assertRaisesRegex(FileNotFoundError, 'No prepared quick'):
                plan(['MissForest'], ['california_housing'], 'quick', None, 0, 'x', paths)


if __name__ == '__main__':
    unittest.main()
