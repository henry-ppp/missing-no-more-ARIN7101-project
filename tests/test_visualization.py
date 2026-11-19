"""Offline tests for shared_visualization: synthetic tables only, no downloads, no model training.

shared_evaluation is used here only to write real report files, so a change to its report format makes these
tests fail and points at shared_visualization/results.py. The package itself must never import it.
"""
import json
import subprocess
import sys
import unittest
import warnings
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

import shared_visualization as sv  # noqa: E402
from data_preprocessing.dataset_config import DATASETS, config_path  # noqa: E402
from data_preprocessing.download import sha256  # noqa: E402
from data_preprocessing.prepare_all import export_configuration, read_matrix  # noqa: E402
from shared_evaluation.evaluate import evaluate_files  # noqa: E402
from shared_visualization import style  # noqa: E402
from shared_visualization.results import Results, one_setting, point_values  # noqa: E402
from shared_visualization.stats import (cliques, holm, pairwise_wilcoxon, rank_matrix,  # noqa: E402
                                        smallest_wilcoxon_p)

METHODS = ('missforest', 'mice_rf', 'knn', 'dae_v2', 'vae')


def correlated_table(n, columns, seed=0):
    d = len(columns)
    cov = 0.6 * np.ones((d, d)) + 0.4 * np.eye(d)
    values = np.random.default_rng(seed).multivariate_normal(np.zeros(d), cov, size=n)
    return pd.DataFrame(values, columns=columns, index=pd.Index([f'source.csv:{i}' for i in range(n)], name='row_id'))


def real_report(root, noise=None, n_boot=40):
    """A shared_evaluation version 2 report (with downstream check) for noisy versions of the truth."""
    noise = noise or {'missforest': 0.3, 'dae_v2': 0.6}
    features = list(DATASETS['california_housing'].features)
    X = correlated_table(300, features)
    raw = root / 'raw' / 'california_housing'
    raw.mkdir(parents=True)
    X.reset_index(drop=True).assign(MedHouseVal=X.to_numpy() @ np.linspace(1, -1, len(features))).to_csv(
        raw / 'source.csv', index=False)
    directory = config_path(root / 'processed', 'california_housing', 'quick', 0.3, 0)
    export_configuration(X, 'california_housing', 'quick', 0.3, 0, directory, {}, {'source.csv': sha256(raw / 'source.csv')})
    truth = read_matrix(directory / 'X_cleaned_standardized.csv')
    hidden = read_matrix(directory / 'mask_eval.csv').astype(bool).to_numpy()
    paths = {}
    for i, (name, sd) in enumerate(noise.items()):
        rng = np.random.default_rng(i + 1)
        (root / name).mkdir()
        (truth + np.where(hidden, rng.normal(scale=sd, size=truth.shape), 0.0)).to_csv(
            root / name / 'predictions.csv', index_label='row_id')
        paths[name] = root / name / 'predictions.csv'
    report, path = evaluate_files('california_housing', 'quick', 0.3, 0, paths, root / 'processed', root / 'output',
                                     n_boot=n_boot, run_downstream=True, raw_dir=root / 'raw')
    return report, path, directory


def synthetic_results(datasets=('credit_card', 'gas_turbine'), rates=(0.1, 0.3), methods=METHODS, seed=0):
    """Made-up scores and paired p-values in the tidy format (for drawing tests only)."""
    rng = np.random.default_rng(seed)
    scores, pairs = [], []
    for dataset in datasets:
        for rate in rates:
            values = {m: 0.6 - 0.1 * i - 0.2 * rate + rng.normal(0, 0.02) for i, m in enumerate(methods)}
            setting = dict(dataset=dataset, mode='quick', rate=rate, seed=0)
            for m, v in values.items():
                for metric, shift in (('relative_improvement', 0), ('wasserstein_improvement', 0.15), ('gap_closed', 0.1)):
                    scores.append({**setting, 'method': m, 'metric': metric, 'estimate': v + shift,
                                   'bias_corrected': v + shift, 'ci_low': v + shift - 0.03, 'ci_high': v + shift + 0.03})
                scores.append({**setting, 'method': m, 'metric': 'mae', 'estimate': 1 - (v + 0.05)})
            scores.append({**setting, 'method': 'mean', 'metric': 'mae', 'estimate': 1.0})
            for a, b in combinations(methods, 2):
                for metric in ('relative_improvement', 'wasserstein_improvement', 'gap_closed'):
                    pairs.append({**setting, 'method_a': a, 'method_b': b, 'metric': metric,
                                  'difference': values[a] - values[b], 'p_value': 0.001 if abs(values[a] - values[b]) > 0.05 else 0.4})
    return Results(pd.DataFrame(scores), pd.DataFrame(pairs))


class ReadingReports(unittest.TestCase):
    def test_reads_a_real_version_2_report(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            report, path, _ = real_report(Path(temp))
            results = sv.load_results(Path(temp) / 'output')
        scores = results.scores
        self.assertEqual(set(scores['method']), {'missforest', 'dae_v2', 'mean', 'true'})
        self.assertTrue({'relative_improvement', 'wasserstein_improvement', 'gap_closed', 'downstream_score'}
                        <= set(scores['metric']))
        row = scores[(scores['method'] == 'dae_v2') & (scores['metric'] == 'wasserstein_improvement')].iloc[0]
        entry = report['accuracy_and_distribution']['methods']['dae_v2']['wasserstein_improvement']
        self.assertEqual(row['estimate'], entry['estimate'])
        self.assertEqual([row['ci_low'], row['ci_high']], entry['ci'])
        pair = results.pairs[(results.pairs['metric'] == 'relative_improvement')
                             & results.pairs[['method_a', 'method_b']].isin(['missforest', 'dae_v2']).all(axis=1)]
        self.assertEqual(len(pair), 1)
        self.assertTrue(np.isfinite(pair['p_value'].iloc[0]))

    def test_reads_a_version_1_report_and_renames(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / 'evaluation_v1_x.json'
            path.write_text(json.dumps({'dataset': 'credit_card', 'mode': 'quick', 'rate': 0.3, 'seed': 0,
                                        'model': 'dae_v2_initial', 'mean_baseline_rmse': 1.4,
                                        'metrics': {'rmse': 1.2, 'mae': 0.4, 'r2': 0.3, 'relative_improvement': 0.14}}))
            results = sv.load_results(path, rename={'dae_v2_initial': 'dae_v2'})
        self.assertEqual(set(results.scores['method']), {'dae_v2', 'mean'})
        self.assertTrue(results.pairs.empty)
        self.assertTrue(results.scores['ci_low'].isna().all())

    def test_identical_copies_merge_and_conflicts_raise(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            _, path, _ = real_report(Path(temp))
            once = sv.load_results(path)
            copy = Path(temp) / 'copy' / 'evaluation_v2_copy.json'
            copy.parent.mkdir()
            copy.write_text(path.read_text())
            twice = sv.load_results(path, copy)
            self.assertEqual(len(twice.scores), len(once.scores))
            self.assertEqual(len(twice.pairs), len(once.pairs))
            changed = json.loads(path.read_text())
            changed['accuracy_and_distribution']['methods']['dae_v2']['rmse']['estimate'] += 0.1
            copy.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Different values'):
                sv.load_results(path, copy)

    def test_tidy_tables_round_trip(self):
        results = synthetic_results()
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            scores, pairs = sv.write_tables(results, temp)
            back = sv.load_results(scores, pairs)
        pd.testing.assert_frame_equal(back.scores.drop(columns='source'), results.scores.drop(columns='source'))
        pd.testing.assert_frame_equal(back.pairs.drop(columns='source'), results.pairs.drop(columns='source'))

    def test_unknown_file_is_rejected(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / 'evaluation_other.json'
            path.write_text('{"hello": 1}')
            with self.assertRaisesRegex(ValueError, 'not a shared_evaluation report'):
                sv.load_results(path)

    def test_derived_improvement_and_point_values(self):
        scores = pd.DataFrame([
            dict(dataset='d', mode='quick', rate=0.3, seed=0, method='mean', metric='mae', estimate=2.0),
            dict(dataset='d', mode='quick', rate=0.3, seed=0, method='m', metric='mae', estimate=1.0),
            dict(dataset='d', mode='quick', rate=0.3, seed=0, method='m', metric='wasserstein_improvement',
                 estimate=0.5, bias_corrected=0.6),
            dict(dataset='d', mode='quick', rate=0.3, seed=0, method='m', metric='relative_improvement',
                 estimate=0.2, bias_corrected=0.3)])
        derived = sv.derive_improvement(Results(scores, pd.DataFrame()).scores)
        value = derived[(derived['metric'] == 'mae_improvement')]['estimate']
        self.assertEqual(value.tolist(), [0.5])
        self.assertEqual(len(sv.derive_improvement(derived)), len(derived))
        auto = point_values(derived[derived['metric'] != 'mae_improvement'].iloc[2:4]).tolist()
        self.assertEqual(auto, [0.6, 0.2])

    def test_one_setting_needs_a_choice_when_ambiguous(self):
        results = synthetic_results(rates=(0.1, 0.3))
        with self.assertRaisesRegex(ValueError, 'rate'):
            one_setting(results)
        selected, chosen = one_setting(results, rate=0.3)
        self.assertEqual(chosen, {'mode': 'quick', 'seed': 0, 'rate': 0.3})
        self.assertEqual(set(selected.scores['rate']), {0.3})


class Statistics(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_smallest_wilcoxon_p(self):
        self.assertEqual(smallest_wilcoxon_p(4), 0.125)
        self.assertLess(smallest_wilcoxon_p(16) * 21, 0.05)

    def test_ranks_and_cliques(self):
        values = pd.DataFrame({'a': [3, 3, 3], 'b': [2, 2, 2], 'c': [1, 1, 2]})
        ranks = rank_matrix(values, higher_is_better=True)
        self.assertEqual(ranks['a'].tolist(), [1, 1, 1])
        self.assertEqual(ranks.loc[2, 'b'], 2.5)
        pairwise = pd.DataFrame({'method_a': ['a', 'a', 'b'], 'method_b': ['b', 'c', 'c'], 'p_holm': [0.5, 0.01, 0.5]})
        self.assertEqual(cliques(pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}), pairwise), [['a', 'b'], ['b', 'c']])
        same = pairwise_wilcoxon(pd.DataFrame({'a': [1.0, 2.0], 'b': [1.0, 2.0]}))
        self.assertEqual(same['p_value'].tolist(), [1.0])

    def test_shaded_area_is_the_wasserstein_distance(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=200), rng.gamma(2.0, size=150)
        self.assertAlmostEqual(sv.ecdf_area(a, b), wasserstein_distance(a, b), places=12)
        self.assertAlmostEqual(sv.ecdf_area(a, np.zeros(5)), np.mean(np.abs(a)), places=12)


class Colours(unittest.TestCase):
    def setUp(self):
        style.reset_models()

    def tearDown(self):
        style.reset_models()

    def test_fixed_colours_and_name_variants(self):
        self.assertEqual(style.model_style('MICE-RF').colour, style.PALETTE[1])
        self.assertEqual(style.model_style('mice_rf'), style.model_style('MICE-RF'))
        self.assertEqual(style.ordered_methods(['vae', 'mean', 'knn', 'missforest']), ['missforest', 'knn', 'vae'])

    def test_unknown_models_get_a_spare_colour_once_then_stop(self):
        with self.assertWarns(UserWarning):
            first = style.model_style('new_model')
        self.assertEqual(first.colour, style.PALETTE[7])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(style.model_style('new_model'), first)
        with self.assertRaisesRegex(ValueError, 'No unused palette colour'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            style.model_style('another_model')
        with self.assertRaisesRegex(ValueError, 'palette'):
            style.register_model('x', colour='#123456')

    def test_registering_a_listed_model_keeps_its_look(self):
        before = style.model_style('dae_v2')
        after = style.register_model('dae_v2', label='Denoising AE', family='deep learning')
        self.assertEqual((after.colour, after.marker, after.label), (before.colour, before.marker, 'Denoising AE'))


class Figures(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def assert_uses_colour(self, figure, colour):
        colours = {to_hex(line.get_color()) for ax in figure.axes for line in ax.get_lines()}
        self.assertIn(colour, colours)

    def test_every_figure_renders_and_saves(self):
        results = synthetic_results(datasets=('credit_card', 'gas_turbine', 'superconductivity'),
                                    rates=(0.1, 0.2, 0.3, 0.5))
        plots = {'scoreboard': sv.scoreboard(results, rate=0.3),
                 'tradeoff': sv.accuracy_vs_distribution(results),
                 'missing_rate': sv.missing_rate_curves(results),
                 'ranking': sv.critical_difference(results)}
        colour = style.model_style('missforest').colour
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            for name, plot in plots.items():
                with self.subTest(name):
                    self.assertFalse(plot.data.empty)
                    self.assert_uses_colour(plot.figure, colour)
                    saved = plot.save(Path(temp) / name)
                    self.assertEqual({p.suffix for p in saved}, {'.pdf', '.svg', '.png', '.csv'})
                    self.assertTrue(all(p.stat().st_size > 0 for p in saved))
                    texts = ' '.join(t.get_text() for t in plot.figure.texts)
                    self.assertIn('CC BY 4.0', texts)

    def test_scoreboard_marks_significance_from_paired_p_values(self):
        results = synthetic_results(datasets=('credit_card',), rates=(0.3,), methods=('missforest', 'knn'))
        data = sv.scoreboard(results, metrics=('relative_improvement',)).data.set_index('method')
        self.assertTrue(data.loc['missforest', 'best'])
        self.assertTrue(data.loc['knn', 'significantly_worse'])     # 0.1 apart, p = 0.001
        no_pairs = Results(results.scores, pd.DataFrame())
        with self.assertWarns(UserWarning):
            data = sv.scoreboard(no_pairs, metrics=('relative_improvement',)).data
        self.assertTrue(data['significantly_worse'].isna().all())

    def test_tradeoff_derives_mae_improvement_and_reports_height(self):
        data = sv.accuracy_vs_distribution(synthetic_results(rates=(0.3,))).data
        self.assertIn('mae_improvement', data)
        np.testing.assert_allclose(data['height_above_diagonal'],
                                   data['wasserstein_improvement'] - data['mae_improvement'])
        self.assertTrue(data['kendall_tau'].notna().all())

    def test_ranking_says_when_settings_are_too_few(self):
        few = sv.critical_difference(synthetic_results(datasets=('credit_card', 'gas_turbine'), rates=(0.1, 0.3))).data
        self.assertFalse(few['significance_testable'].any())
        many = synthetic_results(datasets=('credit_card', 'gas_turbine', 'superconductivity', 'california_housing'),
                                 rates=(0.1, 0.2, 0.3, 0.5))
        self.assertTrue(sv.critical_difference(many).data['significance_testable'].all())
        table = sv.rank_table(many)
        self.assertEqual(table.groupby('metric')['average_rank'].sum().round(6).unique().tolist(), [15.0])

    def test_missing_rate_needs_two_rates(self):
        with self.assertRaisesRegex(ValueError, 'two missing rates'):
            sv.missing_rate_curves(synthetic_results(rates=(0.3,)))

    def test_distribution_figure_from_benchmark_files(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            _, _, directory = real_report(Path(temp), n_boot=0)
            truth, imputed = sv.hidden_values(directory / 'X_cleaned_standardized.csv', directory / 'mask_eval.csv',
                                              {'mean': directory / 'X_mean_baseline.csv',
                                               'dae_v2': Path(temp) / 'dae_v2' / 'predictions.csv'}, 'MedInc')
            plot = sv.distribution_figure(truth, imputed, 'MedInc', dataset='california_housing')
            mean_area = plot.data.set_index('method').loc['mean', 'shaded_area']
            self.assertAlmostEqual(mean_area, np.mean(np.abs(truth - imputed['mean'][0])), places=12)
            shuffled = read_matrix(directory / 'mask_eval.csv').iloc[::-1]
            with self.assertRaisesRegex(ValueError, 'rows and columns'):
                sv.hidden_values(directory / 'X_cleaned_standardized.csv', shuffled, {}, 'MedInc')

    def test_training_curves_from_the_results_layout(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            folder = Path(temp) / 'missforest' / 'training' / 'credit_card' / 'quick' / 'rate_0.3_seed_0' / 'run1'
            folder.mkdir(parents=True)
            pd.DataFrame({'round': [1, 2, 3, 4], 'change': [0.5, 0.2, 0.1, 0.15]}).to_csv(
                folder / 'training_history.csv', index=False)
            histories = sv.find_histories(temp, 'missforest')
            self.assertEqual(list(histories), ['Credit card'])
            data = sv.training_curves(histories, 'missforest').data
        self.assertEqual(data['best_step'].tolist(), [3])
        self.assertEqual(data['step_column'].tolist(), ['round'])


class CommandLineAndIndependence(unittest.TestCase):
    def test_command_line_makes_the_supported_figures(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            real_report(Path(temp))
            out = Path(temp) / 'figures'
            cli = subprocess.run([sys.executable, '-m', 'shared_visualization', str(Path(temp) / 'output'),
                                  '--out', str(out), '--formats', 'png'], capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertTrue((out / 'scoreboard_quick_seed_0_rate_0.3.png').is_file())
            self.assertTrue((out / 'accuracy_vs_distribution_quick_seed_0.png').is_file())
            self.assertTrue((out / 'scores.csv').is_file())
            self.assertIn('skipped missing_rate_quick_seed_0', cli.stdout)

    def test_package_never_imports_evaluation_or_model_code(self):
        check = ('import sys, shared_visualization; '
                 'print(sorted({m.split(".")[0] for m in sys.modules} & {"shared_evaluation", "models", "torch"}))')
        result = subprocess.run([sys.executable, '-c', check], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '[]')


if __name__ == '__main__':
    unittest.main()
