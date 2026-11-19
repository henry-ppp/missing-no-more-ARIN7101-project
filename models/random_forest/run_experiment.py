"""Run MissForest and MICE-RF on the shared benchmark, save everything needed later, and evaluate.

One shared seed (0, the benchmark's own) drives every random step, and there are no repeats: the uncertainty comes
from shared_evaluation version 2's bootstrap. From the repository root:

    python -m models.random_forest.run_experiment --plan     # what is done, what is left, rough time
    python -m models.random_forest.run_experiment            # both models, every dataset, full mode, seed 0

Stages, each saved as soon as it finishes so the command can be stopped and restarted:
1. Hyperparameters: each model is tuned once per dataset and mode, by hide-and-check on the model input at one
   missing rate (30% by default), and the winner is saved to
   results/random_forest/<Model>/hyperparameters/<dataset>_<mode>.json. Every run for that dataset and mode, at any
   missing rate, reuses the saved file instead of tuning again. Delete the file to tune again.
2. Final fit: the hyperparameters fill the full model input. Predictions go to the shared location
   data/output/<dataset>/<mode>/<configuration>/<Model>/<run_id>/predictions.csv.
3. Evaluation: shared_evaluation version 2 (bootstrap intervals, Wasserstein, downstream check) scores them.
4. Paired comparison: version 2 scores both models of each setting together, so their difference gets a p-value.

Where things go (<run> = <dataset>/<mode>/<configuration>/<run_id>):
    results/random_forest/<Model>/hyperparameters/<dataset>_<mode>.json      tuned hyperparameters, reused
    results/random_forest/<Model>/experiment/<run>/experiment.json           results of the run
    results/random_forest/<Model>/evaluation/<run>/evaluation_v2_*.json     shared evaluation report
    results/random_forest/<Model>/evaluation/summary/benchmark_summary.csv  one row per run
    results/random_forest/comparison/evaluation/<run>/evaluation_v2_*.json  both models, paired
    local_models/random_forest/<Model>/training/<run>/   not pushed: training_history.csv, importances.csv, run.json
    local_models/random_forest/<Model>/tuning/<dataset>/<mode>/<configuration>/candidate_<k>.json   tuning trials
Fitted forests are not saved: one column's forest is 100-235 MB on the full tables (1-20 GB per run), and the
imputed table, which is all later evaluation needs, is saved instead.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from data_preprocessing.dataset_config import DATASETS, MODEL_INPUT, OPTIONAL, RETIRED, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import load_model_input
from .imputers import MAX_ROUNDS, METHODS, N_TREES
from .tuning import CHECK_SHARE, TUNE_ROWS, candidates, hide_and_check, score_candidate, tuning_sample

SEED = 0
TUNE_RATE = 0.3
HEADLINE = ('relative_improvement', 'wasserstein_improvement')
RAW_SCORES = ('rmse', 'mae', 'r2', 'wasserstein')    # plain estimates from the version 2 report, for the summary
COMPARISON = 'comparison'
# Rough single-core seconds per forest fit, per visible row and per feature tried at each split (M2, 100 trees),
# measured on the full tables. Only used to start the longest runs first and for the --plan estimate.
SECONDS_PER_ROW_FEATURE = {'MissForest': 5e-5, 'MICE-RF': 1.4e-5}


@dataclass(frozen=True)
class Settings:
    """What changes a run's predictions and scores, apart from its hyperparameters. Saved with every run."""
    n_trees: int = N_TREES
    max_rounds: int = MAX_ROUNDS
    n_boot: int = 1000


@dataclass(frozen=True)
class Tuning:
    """How hyperparameters are chosen for a model, dataset and mode that has none saved yet."""
    enabled: bool = True             # False: use each model's defaults
    rows: int | None = TUNE_ROWS     # rows sampled from the model input; None = all rows
    check_share: float = CHECK_SHARE
    rate: float = TUNE_RATE          # missing rate tuned on; if it is not in the run, the smallest rate run


@dataclass(frozen=True)
class Paths:
    processed: Path = Path('data/processed')
    raw: Path = Path('data/raw')
    output: Path = Path('data/output')                 # predictions, as data/README.md specifies
    results: Path = Path('results/random_forest')
    local: Path = Path('local_models/random_forest')


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str
    mode: str
    rate: float
    seed: int
    run_id: str

    def benchmark(self, paths: Paths) -> Path:
        return config_path(paths.processed, self.dataset, self.mode, self.rate, self.seed)

    def _run(self, paths: Paths) -> Path:
        return Path(self.dataset, self.mode, self.benchmark(paths).name, self.run_id)

    def local_dir(self, paths: Paths) -> Path:
        return paths.local / self.model / 'training' / self._run(paths)

    def experiment_dir(self, paths: Paths) -> Path:
        return paths.results / self.model / 'experiment' / self._run(paths)

    def prediction_path(self, paths: Paths) -> Path:
        return config_path(paths.output, self.dataset, self.mode, self.rate, self.seed) / self.model / self.run_id \
            / 'predictions.csv'

    def label(self) -> str:
        return f'{self.model} {self.dataset} {self.mode} {self.rate:g}'


# ---- planning ----

def prepared_rates(paths: Paths, dataset: str, mode: str, seed: int) -> list[float]:
    """Missing rates prepared for a dataset, mode and seed (folders rate_<r>_seed_<seed>)."""
    folders = (paths.processed / dataset / mode).glob(f'rate_*_seed_{seed}')
    return sorted(float(f.name.split('_')[1]) for f in folders if (f / 'metadata.json').is_file())


def plan(models, datasets, mode='full', rates=None, seed=SEED, run_id='initial', paths=Paths()) -> list[Job]:
    jobs = []
    for dataset in datasets:
        found = prepared_rates(paths, dataset, mode, seed)
        for rate in (rates or found):
            if rate not in found:
                raise FileNotFoundError(f'{dataset} {mode} has no prepared rate {rate} for seed {seed} in '
                                        f'{paths.processed}; prepared: {found}')
            jobs += [Job(model, dataset, mode, float(rate), seed, run_id) for model in models]
    if not jobs:
        raise FileNotFoundError(f'No prepared {mode} configurations for seed {seed} in {paths.processed}')
    return jobs


def _shape(job: Job, paths: Paths) -> tuple[int, int]:
    metadata = json.loads((job.benchmark(paths) / 'metadata.json').read_text(encoding='utf-8'))
    return metadata['selected_rows'], len(metadata['features'])


def estimate_seconds(job: Job, settings: Settings, paths: Paths, rows: int | None = None,
                     params: dict | None = None) -> float:
    """An upper bound: every column refitted in every round, with 70% of rows visible."""
    n_rows, n_columns = _shape(job, paths)
    rows = min(rows or n_rows, n_rows)
    predictors = n_columns - 1 if job.model == 'MissForest' else min(40, n_columns - 1)   # MICE-RF: nearest 40
    share = (params or METHODS[job.model].default)['max_features']
    features = max(1, int(np.sqrt(predictors)) if share == 'sqrt' else int(share * predictors))
    return (SECONDS_PER_ROW_FEATURE[job.model] * 0.7 * rows * features * n_columns * settings.max_rounds
            * settings.n_trees / 100)


# ---- reading the model input (only the incomplete matrix, never evaluation files) ----

@lru_cache(maxsize=4)
def _model_input(directory: str, expected_sha: str) -> pd.DataFrame:
    if sha256(Path(directory) / MODEL_INPUT) != expected_sha:
        raise ValueError(f'{directory}/{MODEL_INPUT} differs from the hash in its metadata.json')
    return load_model_input(directory)


def model_input(job: Job, paths: Paths) -> tuple[pd.DataFrame, dict]:
    directory = job.benchmark(paths)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['seed'] != job.seed or metadata['mode'] != job.mode or metadata['dataset'] != job.dataset:
        raise ValueError(f'{directory}/metadata.json does not match {job}')
    return _model_input(str(directory), metadata['artifact_sha256'][MODEL_INPUT]), metadata


# ---- stage 1: hyperparameters, tuned once per model, dataset and mode ----

def hyperparameter_path(model: str, dataset: str, mode: str, paths: Paths) -> Path:
    return paths.results / model / 'hyperparameters' / f'{dataset}_{mode}.json'


def tuning_targets(jobs: list[Job], tuning: Tuning, paths: Paths) -> list[Job]:
    """For each model, dataset and mode in the run without saved hyperparameters, the run to tune on."""
    if not tuning.enabled:
        return []
    groups = {}
    for job in jobs:
        groups.setdefault((job.model, job.dataset, job.mode), {})[job.rate] = job
    return [by_rate[tuning.rate if tuning.rate in by_rate else min(by_rate)]
            for (model, dataset, mode), by_rate in groups.items()
            if not hyperparameter_path(model, dataset, mode, paths).is_file()]


def tuning_tasks(job: Job, paths: Paths) -> list[tuple[Job, int, dict]]:
    _, n_columns = _shape(job, paths)
    return [(job, k, params) for k, params in enumerate(candidates(job.model, n_columns))]


def _trial_path(job: Job, k: int, paths: Paths) -> Path:
    return (paths.local / job.model / 'tuning' / job.dataset / job.mode / job.benchmark(paths).name
            / f'candidate_{k}.json')


def _trial_key(job: Job, params: dict, settings: Settings, tuning: Tuning) -> dict:
    return {'params': params, 'n_trees': settings.n_trees, 'max_rounds': settings.max_rounds, 'rows': tuning.rows,
            'check_share': tuning.check_share, 'seed': job.seed}


def trial_done(job: Job, k: int, params: dict, settings: Settings, tuning: Tuning, paths: Paths) -> bool:
    path = _trial_path(job, k, paths)
    return path.is_file() and json.loads(path.read_text())['settings'] == _trial_key(job, params, settings, tuning)


def run_trial(job: Job, k: int, params: dict, settings: Settings, tuning: Tuning, paths: Paths) -> dict:
    """Score one candidate setting by hide-and-check on the model input (or a sample of its rows), and save it."""
    X, _ = model_input(job, paths)
    sample = tuning_sample(X.to_numpy(dtype=float), job.seed, tuning.rows)
    hidden, check = hide_and_check(sample, job.seed, tuning.check_share)
    scores = score_candidate(job.model, params, hidden, sample, check, job.seed, settings.n_trees, settings.max_rounds)
    record = {'candidate': k, 'settings': _trial_key(job, params, settings, tuning), 'rows': len(sample),
              'check_cells': int(check.sum()), **scores}
    _write_json(_trial_path(job, k, paths), record)
    return record


def save_hyperparameters(job: Job, settings: Settings, tuning: Tuning, paths: Paths) -> dict:
    """Pick the candidate with the lowest check RMSE (the default on a tie) and save it for every missing rate."""
    trials = []
    for _, k, params in tuning_tasks(job, paths):
        if not trial_done(job, k, params, settings, tuning, paths):
            raise RuntimeError(f'Tuning of {job.label()} is not finished (candidate {k})')
        record = json.loads(_trial_path(job, k, paths).read_text())
        trials.append({'params': params, **{key: record[key] for key in ('rmse', 'mae', 'rounds', 'seconds')}})
        rows, check_cells = record['rows'], record['check_cells']
    best = min(range(len(trials)), key=lambda i: (trials[i]['rmse'], i))
    default, chosen = trials[0]['rmse'], trials[best]['rmse']
    saved = {'model': job.model, 'dataset': job.dataset, 'mode': job.mode, 'params': trials[best]['params'],
             'default_params': METHODS[job.model].default, 'grid': METHODS[job.model].grid,
             'tuned_on': {'rate': job.rate, 'seed': job.seed, 'configuration': job.benchmark(paths).name},
             'tuning': {'rows': rows, 'check_share': tuning.check_share, 'check_cells': check_cells,
                        'n_trees': settings.n_trees, 'max_rounds': settings.max_rounds,
                        'trials': _trial_path(job, 0, paths).parent.as_posix()},
             'candidates': trials, 'default_rmse': default, 'chosen_rmse': chosen,
             'gain_percent': 100 * (1 - chosen / default)}
    _write_json(hyperparameter_path(job.model, job.dataset, job.mode, paths), saved)
    return saved


def hyperparameters_for(job: Job, tuning: Tuning, paths: Paths) -> tuple[dict, dict]:
    """(hyperparameters, where they came from) for a run: the saved tuned ones, or the defaults if tuning is off."""
    if not tuning.enabled:
        return dict(METHODS[job.model].default), {'source': 'defaults'}
    path = hyperparameter_path(job.model, job.dataset, job.mode, paths)
    if not path.is_file():
        raise RuntimeError(f'No saved hyperparameters for {job.label()} at {path}')
    saved = json.loads(path.read_text())
    return saved['params'], {'source': 'tuned', 'file': path.as_posix(), 'tuned_on': saved['tuned_on'],
                             'tuning_rows': saved['tuning']['rows'], 'gain_percent': saved['gain_percent']}


# ---- stage 2 and 3: final fit, saving, evaluation ----

def _final_key(job: Job, params: dict, settings: Settings, input_sha: str) -> dict:
    return {'model': job.model, 'dataset': job.dataset, 'mode': job.mode, 'rate': job.rate, 'seed': job.seed,
            'params': params, 'n_trees': settings.n_trees, 'max_rounds': settings.max_rounds,
            'input_sha256': input_sha}


def final_done(job: Job, params: dict | None, settings: Settings, paths: Paths) -> bool:
    """Done if the run was evaluated with these settings (and, if given, these hyperparameters)."""
    experiment = job.experiment_dir(paths) / 'experiment.json'
    if not experiment.is_file():
        return False
    saved = json.loads(experiment.read_text())
    return saved['settings'] == asdict(settings) and (params is None or saved['params'] == params)


def run_final(job: Job, params: dict, source: dict, settings: Settings, paths: Paths) -> dict:
    """Fill the full model input with the given hyperparameters, save predictions and artefacts, then evaluate."""
    X, metadata = model_input(job, paths)
    key = _final_key(job, params, settings, metadata['artifact_sha256'][MODEL_INPUT])
    local, prediction = job.local_dir(paths), job.prediction_path(paths)
    run_file = local / 'run.json'
    saved = json.loads(run_file.read_text()) if run_file.is_file() else None
    if saved is not None and saved['settings'] != key:
        raise FileExistsError(f'{local} was made with other settings or hyperparameters. Use a new --run-id, or '
                              f'delete that folder and {prediction.parent} to replace it.')
    if saved is None or not prediction.is_file() or sha256(prediction) != saved['prediction_sha256']:
        saved = _fit_and_save(job, X, params, key, settings, paths)
    return _evaluate(job, saved, params, source, settings, paths)


def _fit_and_save(job: Job, X: pd.DataFrame, params: dict, key: dict, settings: Settings, paths: Paths) -> dict:
    filled, info = METHODS[job.model].impute(X.to_numpy(dtype=float), job.seed, n_trees=settings.n_trees,
                                             max_rounds=settings.max_rounds, **params)
    if not np.isfinite(filled).all():
        raise FloatingPointError(f'{job.label()}: non-finite imputed values')
    predictions = pd.DataFrame(filled, index=X.index, columns=X.columns)
    prediction = job.prediction_path(paths)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction, index_label='row_id')
    local = job.local_dir(paths)
    local.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(info.history).to_csv(local / 'training_history.csv', index=False)
    pd.DataFrame(info.importances, index=pd.Index(X.columns, name='target'), columns=X.columns).to_csv(
        local / 'importances.csv')
    run = {'settings': key, 'prediction': prediction.as_posix(), 'prediction_sha256': sha256(prediction),
           'kept_round': info.kept_round, 'rounds_run': len(info.history), 'stopped_early': info.stopped_early,
           'fit_seconds': info.seconds, 'environment': _environment()}
    _write_json(prediction.parent / 'metadata.json', {
        'dataset': job.dataset, 'mode': job.mode, 'rate': job.rate, 'seed': job.seed, 'model': job.model,
        'run_id': job.run_id, 'params': params, 'input_sha256': key['input_sha256'],
        'prediction_sha256': run['prediction_sha256'], 'local_artifacts': local.as_posix()})
    _write_json(local / 'run.json', run)
    return run


def _evaluate(job: Job, run: dict, params: dict, source: dict, settings: Settings, paths: Paths) -> dict:
    # Imported only now, after the predictions exist, so the model code never touches evaluation files.
    from shared_evaluation.evaluate import evaluate_files
    prediction = Path(run['prediction'])
    v2, v2_path = evaluate_files(job.dataset, job.mode, job.rate, job.seed, {job.model: prediction}, paths.processed,
                              paths.results, n_boot=settings.n_boot, boot_seed=job.seed, run_downstream=True,
                              raw_dir=paths.raw, model=job.model, run_id=job.run_id)
    accuracy = v2['accuracy_and_distribution']['methods'][job.model]
    headline = {name: {'estimate': accuracy[name]['estimate'], 'ci': accuracy[name]['ci']} for name in HEADLINE}
    downstream = v2.get('downstream', {})
    if 'methods' in downstream:
        gap = downstream['methods'][job.model]['gap_closed']
        headline['gap_closed'] = {'estimate': gap['estimate'], 'ci': gap['ci']}
    experiment = {
        'model': job.model, 'dataset': job.dataset, 'mode': job.mode, 'rate': job.rate, 'seed': job.seed,
        'run_id': job.run_id, 'params': params, 'default_params': METHODS[job.model].default,
        'hyperparameters': source, 'settings': asdict(settings),
        'kept_round': run['kept_round'], 'rounds_run': run['rounds_run'], 'stopped_early': run['stopped_early'],
        'fit_seconds': run['fit_seconds'], 'headline': headline,
        'predictions': prediction.as_posix(), 'prediction_sha256': run['prediction_sha256'],
        'evaluation_v2': v2_path.as_posix(),
        'local_artifacts': job.local_dir(paths).as_posix(), 'environment': run['environment']}
    _write_json(job.experiment_dir(paths) / 'experiment.json', experiment)
    return experiment


# ---- stage 4: paired comparison of the models of each setting ----

def comparison_groups(jobs: list[Job]) -> list[list[Job]]:
    """Jobs of the same setting and run ID, when there are at least two models to compare."""
    groups = {}
    for job in jobs:
        groups.setdefault((job.dataset, job.mode, job.rate, job.seed, job.run_id), []).append(job)
    return [members for members in groups.values() if len(members) > 1]


def comparison_done(members: list[Job], settings: Settings, paths: Paths) -> bool:
    folder = paths.results / COMPARISON / 'evaluation' / members[0]._run(paths)
    wanted = {m.model: sha256(m.prediction_path(paths)) for m in members}
    for report in folder.glob('evaluation_v2_*.json'):
        saved = json.loads(report.read_text())
        if {n: p['sha256'] for n, p in saved['predictions'].items()} == wanted and \
                saved['accuracy_and_distribution']['n_boot'] == settings.n_boot:
            return True
    return False


def compare(members: list[Job], settings: Settings, paths: Paths) -> dict:
    """Evaluate the models of one setting together, so the bootstrap pairs them and gives their differences a p-value.
    Saved under results/random_forest/comparison/evaluation/<run>/."""
    from shared_evaluation.evaluate import evaluate_files
    job = members[0]
    report, _ = evaluate_files(job.dataset, job.mode, job.rate, job.seed,
                            {m.model: m.prediction_path(paths) for m in members}, paths.processed, paths.results,
                            n_boot=settings.n_boot, boot_seed=job.seed, run_downstream=True, raw_dir=paths.raw,
                            model=COMPARISON, run_id=job.run_id)
    return report['accuracy_and_distribution']['differences']


def write_summaries(paths: Paths, models) -> list[Path]:
    """results/random_forest/<Model>/evaluation/summary/benchmark_summary.csv: one row per saved run."""
    written = []
    for model in models:
        rows = []
        for file in sorted((paths.results / model / 'experiment').glob('*/*/*/*/experiment.json')):
            e = json.loads(file.read_text())
            scores = json.loads(Path(e['evaluation_v2']).read_text())['accuracy_and_distribution']['methods'][model]
            row = {k: e[k] for k in ('dataset', 'mode', 'rate', 'seed', 'run_id', 'kept_round', 'rounds_run',
                                     'stopped_early', 'fit_seconds')}
            row.update(params=json.dumps(e['params']), hyperparameters=e['hyperparameters']['source'],
                       tuning_gain_percent=e['hyperparameters'].get('gain_percent'),
                       **{name: scores[name]['estimate'] for name in RAW_SCORES})
            for name, value in e['headline'].items():
                row.update({name: value['estimate'], f'{name}_ci_low': value['ci'][0],
                            f'{name}_ci_high': value['ci'][1]})
            rows.append(row)
        if rows:
            target = paths.results / model / 'evaluation' / 'summary' / 'benchmark_summary.csv'
            text = pd.DataFrame(rows).sort_values(['mode', 'dataset', 'rate', 'run_id']).to_csv(
                index=False, lineterminator='\n')
            current = target.read_text(encoding='utf-8').replace('\r\n', '\n') if target.is_file() else None
            if current != text:     # leave an unchanged file untouched
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode('utf-8'))
            written.append(target)
    return written


# ---- running ----

def clashes(jobs: list[Job], settings: Settings, paths: Paths) -> list[Job]:
    """Runs whose run ID already holds results made with other settings."""
    found = []
    for job in jobs:
        file = job.local_dir(paths) / 'settings.json'
        if file.is_file() and json.loads(file.read_text()) != asdict(settings):
            found.append(job)
    return found


def run(jobs: list[Job], settings: Settings = Settings(), tuning: Tuning = Tuning(), paths: Paths = Paths(),
        workers: int = 1, log=print) -> list[dict]:
    """Run everything that is missing: tuning where no hyperparameters are saved, then final fits with evaluation,
    then paired comparisons. Returns the experiments.

    Refuses before doing anything if a run ID already holds results made with other settings, so saved results are
    never overwritten or mixed.
    """
    clashing = clashes(jobs, settings, paths)
    if clashing:
        raise FileExistsError(f'{len(clashing)} run(s) under run ID {clashing[0].run_id!r} were made with other '
                              f'settings (e.g. {clashing[0].local_dir(paths)}). Use a new --run-id.')
    for job in jobs:
        if not (job.local_dir(paths) / 'settings.json').is_file():
            _write_json(job.local_dir(paths) / 'settings.json', asdict(settings))

    targets = tuning_targets(jobs, tuning, paths)
    if tuning.enabled:
        reused = sorted({(j.model, j.dataset, j.mode) for j in jobs} - {(t.model, t.dataset, t.mode) for t in targets})
        for model, dataset, mode in reused:
            log(f'Reusing saved hyperparameters: {hyperparameter_path(model, dataset, mode, paths)}')
    trials = [t for job in targets for t in tuning_tasks(job, paths)]
    todo = [t for t in trials if not trial_done(*t, settings, tuning, paths)]
    todo.sort(key=lambda t: -estimate_seconds(t[0], settings, paths, tuning.rows, t[2]))
    log(f'Tuning: {len(trials) - len(todo)} of {len(trials)} candidate runs already done, {len(todo)} to run')
    _execute([(run_trial, (*t, settings, tuning, paths)) for t in todo], workers, log,
             lambda task, r: f'{task[1][0].label()} candidate {task[1][1]} {task[1][2]}: check RMSE {r["rmse"]:.4f}')
    for job in targets:
        saved = save_hyperparameters(job, settings, tuning, paths)
        log(f'Saved hyperparameters for {job.model} {job.dataset} {job.mode}: {saved["params"]} '
            f'({saved["gain_percent"]:.2f}% lower check RMSE than the default)')

    chosen = {job: hyperparameters_for(job, tuning, paths) for job in jobs}
    finals = [job for job in jobs if not final_done(job, chosen[job][0], settings, paths)]
    finals.sort(key=lambda job: -estimate_seconds(job, settings, paths, params=chosen[job][0]))
    log(f'Final fits: {len(jobs) - len(finals)} of {len(jobs)} already done, {len(finals)} to run')
    _execute([(run_final, (job, *chosen[job], settings, paths)) for job in finals], workers, log,
             lambda task, e: f'{task[1][0].label()}: RMSE improvement '
                             f'{e["headline"]["relative_improvement"]["estimate"]:.3f}, {e["kept_round"]} rounds, '
                             f'{e["fit_seconds"] / 60:.1f} min')

    groups = comparison_groups(jobs)
    todo_groups = [g for g in groups if not comparison_done(g, settings, paths)]
    log(f'Paired comparisons: {len(groups) - len(todo_groups)} of {len(groups)} already done, {len(todo_groups)} to run')
    _execute([(compare, (g, settings, paths)) for g in todo_groups], workers, log,
             lambda task, d: f'{task[1][0][0].dataset} {task[1][0][0].mode} {task[1][0][0].rate:g}: ' + '; '.join(
                 f'{pair} RMSE improvement p = {v["relative_improvement"]["p_value"]:.3g}' for pair, v in d.items()
                 if 'mean' not in pair))
    for path in write_summaries(paths, sorted({job.model for job in jobs})):
        log(f'Saved {path}')
    return [json.loads((job.experiment_dir(paths) / 'experiment.json').read_text()) for job in jobs]


def _execute(tasks, workers, log, describe):
    """Run (function, arguments) tasks on `workers` processes, one thread each, logging each as it finishes."""
    if not tasks:
        return
    start = time.time()
    with parallel_config(backend='loky', inner_max_num_threads=1):   # one thread per worker: runs go in parallel
        results = Parallel(n_jobs=min(workers, len(tasks)), return_as='generator_unordered')(
            delayed(_numbered)(i, function, arguments) for i, (function, arguments) in enumerate(tasks))
        for k, (i, result) in enumerate(results, 1):
            log(f'  [{k}/{len(tasks)}, {(time.time() - start) / 60:.1f} min] {describe(tasks[i], result)}')


def _numbered(i, function, arguments):
    return i, function(*arguments)


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False, default=_plain), encoding='utf-8')
    temporary.replace(path)      # a run stopped mid-write never leaves a half-written file


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot save {type(value).__name__}')


def _environment() -> dict:
    import sklearn
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__, 'machine': platform.machine(), 'system': platform.system(),
            'code_commit': commit}


def describe_plan(jobs, settings, tuning, paths, workers) -> str:
    """Per run, what is left and an upper bound on its cost; then the total and the wall time with `workers`."""
    lines, trials, fits = [], [], []
    clashing, targets = set(clashes(jobs, settings, paths)), set(tuning_targets(jobs, tuning, paths))
    for job in jobs:
        if job in clashing:
            lines.append(f'  {job.label():45} saved with other settings: use a new --run-id')
            continue
        tune = [estimate_seconds(job, settings, paths, tuning.rows, t[2]) for t in tuning_tasks(job, paths)
                if not trial_done(*t, settings, tuning, paths)] if job in targets else []
        fit = 0.0 if final_done(job, None, settings, paths) else estimate_seconds(job, settings, paths)
        trials += tune
        fits.append(fit)
        hyper = ('tune here' if job in targets else 'defaults' if not tuning.enabled
                 else 'reuse saved' if hyperparameter_path(job.model, job.dataset, job.mode, paths).is_file()
                 else 'from the tuned rate')
        state = 'done' if fit == 0 and not tune else f'tuning <= {sum(tune) / 3600:.1f} h, fit <= {fit / 3600:.1f} h'
        lines.append(f'  {job.label():45} hyperparameters: {hyper:20} {state}')

    def stage(times):   # parallel stage: at least its longest task, at least its share of the workers
        return max(max(times, default=0.0), sum(times) / max(workers, 1))

    wall = (stage(trials) + stage(fits)) * 1.7     # final fits start after tuning; 8 busy cores ran ~1.7x slower
    lines.append(f'Upper bound: {(sum(trials) + sum(fits)) / 3600:.1f} CPU-hours; with {workers} workers about '
                 f'{wall / 3600:.1f} h of wall time. Real runs are shorter: MissForest usually stops before its '
                 'last round.')
    return '\n'.join(lines)


def _rows(value: str) -> int | None:
    return None if value == 'all' else int(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', choices=list(METHODS), default=list(METHODS))
    parser.add_argument('--datasets', nargs='+', choices=[*DATASETS, *OPTIONAL, *RETIRED], default=list(DATASETS),
                        help='Default: the four required datasets. Add hmda_ri_2023 only when that optional dataset is needed.')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full')
    parser.add_argument('--rates', nargs='+', type=float, help='Default: every rate prepared for the seed')
    parser.add_argument('--seed', type=int, default=SEED, help='The one shared seed (default 0)')
    parser.add_argument('--run-id', default='initial')
    parser.add_argument('--n-trees', type=int, default=N_TREES)
    parser.add_argument('--max-rounds', type=int, default=MAX_ROUNDS)
    parser.add_argument('--no-tune', action='store_true', help='Use the default hyperparameters')
    parser.add_argument('--tune-rows', type=_rows, default=TUNE_ROWS,
                        help='Rows of the model input used for tuning, or "all" (default 3000)')
    parser.add_argument('--tune-rate', type=float, default=TUNE_RATE,
                        help='Missing rate to tune on when no hyperparameters are saved (default 0.3)')
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='Runs at the same time, one core each')
    parser.add_argument('--processed-dir', type=Path, default=Paths.processed)
    parser.add_argument('--raw-dir', type=Path, default=Paths.raw)
    parser.add_argument('--output-dir', type=Path, default=Paths.output)
    parser.add_argument('--results-dir', type=Path, default=Paths.results)
    parser.add_argument('--local-dir', type=Path, default=Paths.local)
    parser.add_argument('--plan', action='store_true', help='Only show what is done and what is left')
    args = parser.parse_args(argv)
    from shared_evaluation.result_paths import validate_identifier
    validate_identifier(args.run_id, 'Run ID')
    paths = Paths(args.processed_dir, args.raw_dir, args.output_dir, args.results_dir, args.local_dir)
    settings = Settings(args.n_trees, args.max_rounds, args.n_boot)
    tuning = Tuning(not args.no_tune, args.tune_rows, CHECK_SHARE, args.tune_rate)
    jobs = plan(args.models, args.datasets, args.mode, args.rates, args.seed, args.run_id, paths)
    print(describe_plan(jobs, settings, tuning, paths, args.jobs))
    if not args.plan:
        run(jobs, settings, tuning, paths, args.jobs)


if __name__ == '__main__':
    # Re-import under the package name so worker processes can find the functions they are sent.
    from models.random_forest.run_experiment import main as package_main
    package_main()
