"""Evaluation, version 2: accuracy, distribution, confidence intervals and an optional downstream check.

Works for any imputation method; it only reads prediction files and never imports a model. It does not import
evaluate.py or metrics.py, so it keeps working if version 1 is removed.

What it adds to version 1:
- Distribution score: Wasserstein distance at the hidden cells, also as an improvement over mean imputation
  (see metrics_v2.py).
- Confidence intervals without retraining: whole rows are resampled (the cells of one row are imputed together,
  so their errors are correlated). Every model and the mean baseline get the same resampled rows, so
  differences between models are paired and come with a p-value. Intervals and p-values use the basic
  bootstrap (Davison & Hinkley, 1997, ch. 5), which corrects for bias; that matters for the Wasserstein
  distance, whose plug-in value is biased upward. The intervals cover which cells happened to be scored, not
  retraining, model seeds or a new mask, so they are narrower than variation between runs.
- Downstream check (optional): predict each dataset's target from the imputed table with a simple model, using
  cross-validation, and report the share of the gap between mean imputation (0) and the true table (1) that an
  imputation closes. Logistic regression and AUROC for credit card default; linear regression and R^2 for
  California house value and superconductor critical temperature. Gas turbine has no target. The targets are
  not in the prepared data, so they are read from the raw cache data/raw/<dataset>/source.csv.

Command line, from the repository root, once predictions.csv files exist (one or several models):

    python -m shared_evaluation.evaluate_v2 --dataset credit_card --mode quick --rate 0.3 --seed 0 \\
        --predictions data/output/credit_card/quick/rate_0.3_seed_0/dae/predictions.csv \\
                      data/output/credit_card/quick/rate_0.3_seed_0/missforest/predictions.csv \\
        --downstream

A model is named after its folder, or give NAME=PATH. Results go to
data/output/<dataset>/<mode>/rate_<r>_seed_<s>/evaluation_v2_<hash>.json.

In Python:

    from shared_evaluation.evaluate_v2 import evaluate, bootstrap
    evaluate(X_true, X_pred, mask_eval, X_mean_baseline)           # six scores, no intervals
    bootstrap(X_true, {"dae": X_dae, "vae": X_vae}, mask_eval, X_mean_baseline)   # with intervals
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from data_preprocessing.dataset_config import DATASETS, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import read_matrix
from .metrics_v2 import mae, r2, relative_improvement, rmse, validate, wasserstein

SCORES = ('rmse', 'mae', 'r2', 'relative_improvement', 'wasserstein', 'wasserstein_improvement')
BASELINE = 'mean'
TRUE = 'true'
# dataset -> (task, accepted names of the target column in data/raw/<dataset>/source.csv)
TARGETS = {
    'credit_card': ('classification', ('Y', 'default payment next month')),
    'california_housing': ('regression', ('MedHouseVal',)),
    'superconductivity': ('regression', ('critical_temp',)),
}
NAME_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')


# ---- point scores ----

def _raw_scores(truth, prediction, hidden):
    return {'rmse': rmse(truth, prediction, hidden), 'mae': mae(truth, prediction, hidden),
            'r2': r2(truth, prediction, hidden), 'wasserstein': wasserstein(truth, prediction, hidden)}


def _with_improvements(raw, baseline):
    return {**raw, 'relative_improvement': relative_improvement(raw['rmse'], baseline['rmse']),
            'wasserstein_improvement': relative_improvement(raw['wasserstein'], baseline['wasserstein'])}


def evaluate(X_true, X_pred, mask_eval, X_mean_baseline):
    """Version 1's four scores plus the Wasserstein distance and its improvement over the mean baseline."""
    truth, prediction, hidden = validate(X_true, X_pred, mask_eval)
    baseline = validate(X_true, X_mean_baseline, mask_eval)[1]
    return _with_improvements(_raw_scores(truth, prediction, hidden), _raw_scores(truth, baseline, hidden))


# ---- paired row bootstrap ----

def _as_array(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _summary(estimate, values, level, with_p=False):
    """Plug-in estimate, bootstrap bias, bias-corrected value and basic bootstrap interval (and p-value for 0).

    The basic interval is [2*estimate - upper quantile, 2*estimate - lower quantile]. The p-value is two-sided,
    from the same pivot, with the +1 correction so it is never 0.
    """
    values = values[np.isfinite(values)]
    out = {'estimate': estimate, 'bias': None, 'bias_corrected': None, 'ci': [None, None]}
    if with_p:
        out['p_value'] = None
    if estimate is None or not len(values):
        return out
    tail = (1 - level) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    bias = float(values.mean() - estimate)
    out.update(bias=bias, bias_corrected=estimate - bias, ci=[float(2 * estimate - high), float(2 * estimate - low)])
    if with_p:
        pivot = 2 * estimate - values
        below, above = (1 + np.sum(pivot <= 0)) / (1 + len(pivot)), (1 + np.sum(pivot >= 0)) / (1 + len(pivot))
        out['p_value'] = float(min(1.0, 2 * min(below, above)))
    return out


def _paired_bootstrap(replicate, rows, keys, n_boot, seed, level):
    """replicate(row_index) -> {name: {key: value}}. Every name gets the same resampled rows in each draw."""
    point = replicate(rows)
    rng = np.random.default_rng(seed)
    draws = [replicate(rng.choice(rows, size=len(rows), replace=True)) for _ in range(n_boot)]
    values = {n: {k: _as_array([d[n][k] for d in draws]) for k in keys} for n in point}
    methods = {n: {k: _summary(point[n][k], values[n][k], level) for k in keys} for n in point}
    differences = {}
    for a, b in combinations(point, 2):
        differences[f'{a} - {b}'] = {k: _summary(
            None if point[a][k] is None or point[b][k] is None else point[a][k] - point[b][k],
            values[a][k] - values[b][k], level, with_p=True) for k in keys}
    return {'n_boot': n_boot, 'seed': seed, 'level': level, 'interval': 'basic bootstrap',
            'rows_resampled': int(len(rows)), 'methods': methods, 'differences': differences}


def bootstrap(X_true, predictions, mask_eval, X_mean_baseline, n_boot=1000, seed=0, level=0.95):
    """Scores with confidence intervals for each model in predictions ({name: table}) and the mean baseline,
    plus paired differences between every two of them. n_boot=0 gives the point scores only.

    Only rows with at least one hidden cell are resampled; the others never enter a score.
    """
    if BASELINE in predictions or TRUE in predictions:
        raise ValueError(f'"{BASELINE}" and "{TRUE}" are reserved names')
    truth, baseline, hidden = validate(X_true, X_mean_baseline, mask_eval)
    tables = {name: validate(X_true, X_pred, mask_eval)[1] for name, X_pred in predictions.items()}
    tables[BASELINE] = baseline

    def replicate(index):
        t, h = truth[index], hidden[index]
        raw = {name: _raw_scores(t, table[index], h) for name, table in tables.items()}
        return {name: _with_improvements(r, raw[BASELINE]) for name, r in raw.items()}

    return _paired_bootstrap(replicate, np.flatnonzero(hidden.any(axis=1)), SCORES, n_boot, seed, level)


def holm(p_values):
    """Holm's step-down adjustment of a family of p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p, kind='stable')):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


# ---- downstream prediction ----

def load_target(dataset, row_ids, raw_dir='data/raw', raw_sha256=None):
    """The prediction target for the given row IDs, read from the raw cache the prepared data came from.

    Row IDs are "source.csv:<position>", the position in the raw file (see data_preprocessing.download).
    raw_sha256 is metadata.json's record of the raw files; the cache must match it.
    """
    if dataset not in TARGETS:
        raise ValueError(f'{dataset} has no prediction target')
    path = Path(raw_dir) / dataset / 'source.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing. The target is not in the prepared data, so it is read from '
                                'the raw cache; ask whoever prepared the data to share data/raw/.')
    if raw_sha256 is not None and sha256(path) != raw_sha256.get('source.csv'):
        raise ValueError(f'{path} differs from the raw snapshot recorded in metadata.json')
    files, positions = zip(*(str(r).rsplit(':', 1) for r in row_ids))
    if set(files) != {'source.csv'}:
        raise ValueError('Row IDs must have the form source.csv:<position>')
    raw = pd.read_csv(path)
    column = next((c for c in TARGETS[dataset][1] if c in raw.columns), None)
    if column is None:
        raise ValueError(f'{path} has none of the target columns {TARGETS[dataset][1]}')
    y = pd.Series(raw[column].to_numpy(dtype=float)[np.array(positions, dtype=int)],
                  index=pd.Index(list(row_ids), name='row_id'), name=column)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError('Target has missing or non-finite values')
    return y


def downstream(tables, y, task, n_splits=5, seed=0, n_boot=1000, level=0.95):
    """Cross-validated prediction of y from each complete table, with paired row-bootstrap intervals.

    tables ({name: DataFrame}) must include "mean" (mean imputation) and "true" (the true standardised table).
    Every table uses the same folds. The share of the gap closed is (score - score_mean) / (score_true -
    score_mean). The imputers never see y, so nothing leaks; imputation used all rows, not separate folds.
    """
    if BASELINE not in tables or TRUE not in tables:
        raise ValueError(f'tables must include "{BASELINE}" and "{TRUE}"')
    reference = tables[TRUE]
    for name, X in tables.items():
        if not X.index.equals(reference.index) or not X.columns.equals(reference.columns):
            raise ValueError(f'{name}: row IDs and columns must match the true table exactly')
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: tables must be complete and finite')
    if not y.index.equals(reference.index):
        raise ValueError('Target row IDs must match the tables')
    target = y.to_numpy()
    if task == 'classification':
        model, folds = LogisticRegression(max_iter=1000), StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        method, score, score_name = 'predict_proba', roc_auc_score, 'auroc'
    elif task == 'regression':
        model, folds = LinearRegression(), KFold(n_splits, shuffle=True, random_state=seed)
        method, score, score_name = 'predict', r2_score, 'r2'
    else:
        raise ValueError('task must be classification or regression')
    out_of_fold = {}
    for name, X in tables.items():
        p = cross_val_predict(model, X.to_numpy(dtype=float), target, cv=folds, method=method)
        out_of_fold[name] = p[:, 1] if method == 'predict_proba' else p

    def replicate(index):
        s = {name: float(score(target[index], p[index])) for name, p in out_of_fold.items()}
        gap = s[TRUE] - s[BASELINE]
        return {name: {'score': v, 'gap_closed': (v - s[BASELINE]) / gap if gap > 0 else None}
                for name, v in s.items()}

    result = _paired_bootstrap(replicate, np.arange(len(target)), ('score', 'gap_closed'), n_boot, seed, level)
    return {'task': task, 'model': type(model).__name__, 'score_name': score_name, 'n_splits': n_splits, **result}


# ---- files and command line ----

def load_benchmark(dataset, mode, rate, seed, processed_dir='data/processed'):
    """Checks the prepared configuration's metadata and hashes, then returns it with the evaluation files."""
    directory = config_path(processed_dir, dataset, mode, rate, seed)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    for key, expected in {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed}.items():
        if metadata[key] != expected:
            raise ValueError(f'Mismatched benchmark metadata: {key}')
    for filename, digest in metadata['artifact_sha256'].items():
        if sha256(directory / filename) != digest:
            raise ValueError(f'Benchmark artifact was modified: {filename}')
    truth = read_matrix(directory / 'evaluation_only' / 'X_true_standardized.csv')
    if truth.index.tolist() != metadata['row_ids'] or list(truth.columns) != metadata['features']:
        raise ValueError('Ground truth identifiers differ from metadata')
    return directory, metadata, truth, read_matrix(directory / 'mask_eval.csv'), read_matrix(directory / 'X_mean_baseline.csv')


def evaluate_files(dataset, mode, rate, seed, predictions, processed_dir='data/processed', output_dir='data/output',
                   n_boot=1000, boot_seed=0, level=0.95, run_downstream=False, raw_dir='data/raw', n_splits=5):
    """Scores saved prediction files ({model name: path}) for one configuration and writes one JSON report."""
    if not predictions:
        raise ValueError('Give at least one prediction file')
    for name in predictions:
        if not name or set(name) - NAME_CHARS or name in (BASELINE, TRUE):
            raise ValueError(f'Invalid model name {name!r}: letters, digits, _ and - only, not mean or true')
    directory, metadata, truth, mask, baseline = load_benchmark(dataset, mode, rate, seed, processed_dir)
    frames = {name: read_matrix(path) for name, path in predictions.items()}
    report = {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed,
              'predictions': {n: {'path': str(Path(p).resolve()), 'sha256': sha256(p)} for n, p in predictions.items()},
              'benchmark_metadata_sha256': sha256(directory / 'metadata.json'),
              'accuracy_and_distribution': bootstrap(truth, frames, mask, baseline, n_boot, boot_seed, level)}
    if run_downstream:
        if dataset in TARGETS:
            y = load_target(dataset, truth.index.tolist(), raw_dir, metadata.get('raw_sha256'))
            # true values at observed cells (every model's input), each model's guesses at hidden cells
            hidden = mask.astype(bool)
            tables = {name: truth.mask(hidden, frame) for name, frame in frames.items()}
            tables.update({BASELINE: truth.mask(hidden, baseline), TRUE: truth})
            report['downstream'] = downstream(tables, y, TARGETS[dataset][0], n_splits, boot_seed, n_boot, level)
        else:
            report['downstream'] = {'skipped': f'{dataset} has no prediction target'}
    settings = {'predictions': {n: v['sha256'] for n, v in report['predictions'].items()}, 'n_boot': n_boot,
                'boot_seed': boot_seed, 'level': level, 'downstream': run_downstream, 'n_splits': n_splits}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    target = config_path(output_dir, dataset, mode, rate, seed)
    target.mkdir(parents=True, exist_ok=True)
    output = target / f'evaluation_v2_{digest}.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report, output


def parse_predictions(items):
    """["NAME=PATH" or "PATH", ...] -> {name: path}. A bare path is named after its folder."""
    predictions = {}
    for item in items:
        name, path = item.split('=', 1) if '=' in item else (Path(item).parent.name, item)
        if name in predictions:
            raise ValueError(f'Two prediction files are both named {name!r}; use NAME=PATH')
        predictions[name] = path
    return predictions


def _fmt(entry, key='estimate', digits=3):
    est, (lo, hi) = entry[key], entry['ci']
    if est is None:
        return 'n/a'
    return f'{est:.{digits}f}' + (f' [{lo:.{digits}f}, {hi:.{digits}f}]' if lo is not None else '')


def summary_table(report):
    part = report['accuracy_and_distribution']
    lines = [f"Hidden cells, {part['level']:.0%} basic bootstrap CI from {part['n_boot']} row resamples (this mask only).",
             'Wasserstein columns are bias-corrected; the plug-in values are in the JSON.',
             pd.DataFrame({name: {'RMSE': _fmt(s['rmse']), 'vs mean (RMSE)': _fmt(s['relative_improvement']),
                                  'Wasserstein': _fmt(s['wasserstein'], 'bias_corrected'),
                                  'vs mean (Wasserstein)': _fmt(s['wasserstein_improvement'], 'bias_corrected')}
                           for name, s in part['methods'].items()}).T.to_string()]
    down = report.get('downstream')
    if down and 'skipped' in down:
        lines.append(f"Downstream: skipped, {down['skipped']}")
    elif down:
        lines.append(f"Downstream {down['model']}, {down['n_splits']}-fold CV, {down['score_name']}:")
        lines.append(pd.DataFrame({name: {down['score_name']: _fmt(s['score']), 'gap closed': _fmt(s['gap_closed'])}
                                   for name, s in down['methods'].items()}).T.to_string())
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--predictions', nargs='+', required=True, metavar='[NAME=]PATH')
    parser.add_argument('--n-boot', type=int, default=1000, help='Row resamples; 0 gives point scores only.')
    parser.add_argument('--boot-seed', type=int, default=0)
    parser.add_argument('--level', type=float, default=0.95)
    parser.add_argument('--downstream', action='store_true', help='Also run the downstream prediction check.')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()
    try:
        predictions = parse_predictions(args.predictions)
    except ValueError as exc:
        parser.error(str(exc))
    report, output = evaluate_files(args.dataset, args.mode, args.rate, args.seed, predictions, args.processed_dir,
                                    args.output_dir, args.n_boot, args.boot_seed, args.level, args.downstream,
                                    args.raw_dir, args.n_splits)
    print(summary_table(report))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
