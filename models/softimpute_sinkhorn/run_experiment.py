"""Run SoftImpute and batch Sinkhorn imputation on the shared benchmark, save everything needed later, and evaluate.

One shared seed (0, the benchmark's own) and no repeats: the uncertainty comes from the shared evaluator's bootstrap.
From the repository root:

    python -m models.softimpute_sinkhorn.run_experiment --plan          # what is done and what is left
    python -m models.softimpute_sinkhorn.run_experiment --rates 0.3     # both models, the four required datasets

For each model, dataset and missing rate:
1. Hyperparameters. SoftImpute's lambda is tuned by hide-and-check on the model input (cheap, so it is tuned at every
   rate) and saved to results/softimpute_sinkhorn/SoftImpute/hyperparameters/<dataset>_<mode>_<configuration>.json.
   A rerun reuses the file. Sinkhorn uses the paper's fixed settings.
2. Fit. Predictions go to data/output/<dataset>/<mode>/<configuration>/<Model>/<run_id>/predictions.csv.
3. Evaluation by shared_evaluation.evaluate (bootstrap intervals, Wasserstein, downstream check).
4. Paired comparison: both models of a setting scored together, so their difference gets a p-value.

Where things go (<run> = <dataset>/<mode>/<configuration>/<run_id>):
    results/softimpute_sinkhorn/<Model>/hyperparameters/...json             SoftImpute's tuning, every candidate
    results/softimpute_sinkhorn/<Model>/experiment/<run>/experiment.json    results of the run
    results/softimpute_sinkhorn/<Model>/evaluation/<run>/evaluation_v2_*.json
    results/softimpute_sinkhorn/<Model>/evaluation/summary/benchmark_summary.csv
    results/softimpute_sinkhorn/comparison/evaluation/<run>/evaluation_v2_*.json
    local_models/softimpute_sinkhorn/<Model>/training/<run>/   not pushed: training_history.csv, run.json
Each run is saved as soon as it finishes and skipped on a rerun with the same settings; a run ID reused with
different settings is refused before anything runs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from data_preprocessing.dataset_config import DATASETS, MODEL_INPUT, OPTIONAL, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import load_model_input
from .imputers import (SINKHORN_BATCH, SINKHORN_EPS_SHARE, SINKHORN_LR, SINKHORN_NOISE, SINKHORN_VISITS, SOFT_GRID,
                       SOFT_MAX_ITER, SOFT_TOL, impute_sinkhorn, impute_softimpute, sinkhorn_iterations)
from .tuning import DEFAULT_FRACTION, TIE_TOLERANCE, tune_softimpute

MODELS = ('SoftImpute', 'Sinkhorn')
SEED = 0
COMPARISON = 'comparison'
HEADLINE = ('relative_improvement', 'wasserstein_improvement')
RAW_SCORES = ('rmse', 'mae', 'r2', 'wasserstein')
SINKHORN_SECONDS_PER_ITERATION = 0.04     # measured on the M2 with 4 runs at once; only for --plan


@dataclass(frozen=True)
class Settings:
    """Everything that changes a run's predictions or scores. Saved with every run."""
    n_boot: int = 1000
    tune: bool = True
    soft_grid: tuple = SOFT_GRID
    soft_tol: float = SOFT_TOL
    soft_max_iter: int = SOFT_MAX_ITER
    sinkhorn_visits: float = SINKHORN_VISITS
    sinkhorn_batch: int = SINKHORN_BATCH
    sinkhorn_lr: float = SINKHORN_LR
    sinkhorn_noise: float = SINKHORN_NOISE
    sinkhorn_eps_share: float = SINKHORN_EPS_SHARE

    def of(self, model: str) -> dict:
        """The settings that matter for one model (so changing one model's settings never reruns the other)."""
        prefix = 'soft_' if model == 'SoftImpute' else 'sinkhorn_'
        chosen = {k: v for k, v in asdict(self).items() if k.startswith(prefix) or k == 'n_boot'}
        if model == 'SoftImpute':
            chosen.update(tune=self.tune, tie_tolerance=TIE_TOLERANCE)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in chosen.items()}


@dataclass(frozen=True)
class Paths:
    processed: Path = Path('data/processed')
    raw: Path = Path('data/raw')
    output: Path = Path('data/output')
    results: Path = Path('results/softimpute_sinkhorn')
    local: Path = Path('local_models/softimpute_sinkhorn')


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

    def run_path(self, paths: Paths) -> Path:
        return Path(self.dataset, self.mode, self.benchmark(paths).name, self.run_id)

    def experiment_file(self, paths: Paths) -> Path:
        return paths.results / self.model / 'experiment' / self.run_path(paths) / 'experiment.json'

    def local_dir(self, paths: Paths) -> Path:
        return paths.local / self.model / 'training' / self.run_path(paths)

    def prediction_path(self, paths: Paths) -> Path:
        return config_path(paths.output, self.dataset, self.mode, self.rate, self.seed) / self.model / self.run_id \
            / 'predictions.csv'

    def hyperparameter_file(self, paths: Paths) -> Path:
        return paths.results / self.model / 'hyperparameters' / \
            f'{self.dataset}_{self.mode}_{self.benchmark(paths).name}.json'

    def label(self) -> str:
        return f'{self.model} {self.dataset} {self.mode} {self.rate:g}'


# ---- planning ----

def prepared_rates(paths: Paths, dataset: str, mode: str, seed: int) -> list[float]:
    found = sorted((paths.processed / dataset / mode).glob(f'rate_*_seed_{seed}/metadata.json'))
    return [float(p.parent.name.split('_')[1]) for p in found]


def plan(models, datasets, mode='full', rates=None, seed=SEED, run_id='initial', paths=Paths()) -> list[Job]:
    jobs = []
    for dataset in datasets:
        available = prepared_rates(paths, dataset, mode, seed)
        chosen = available if rates is None else [r for r in rates if r in available]
        missing = [] if rates is None else [r for r in rates if r not in available]
        if missing:
            raise FileNotFoundError(f'{dataset} {mode}: rates {missing} are not prepared for seed {seed}')
        jobs += [Job(m, dataset, mode, r, seed, run_id) for r in chosen for m in models]
    return jobs


def _rows(job: Job, paths: Paths) -> int:
    return len(json.loads((job.benchmark(paths) / 'metadata.json').read_text())['row_ids'])


def estimate_minutes(job: Job, settings: Settings, paths: Paths) -> float:
    if job.model == 'SoftImpute':
        return 2.0
    iterations = sinkhorn_iterations(_rows(job, paths), settings.sinkhorn_visits, settings.sinkhorn_batch)
    return iterations * SINKHORN_SECONDS_PER_ITERATION / 60


# ---- one run: tune, fit, save, evaluate ----

def model_input(job: Job, paths: Paths) -> tuple[pd.DataFrame, str]:
    directory = job.benchmark(paths)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if (metadata['dataset'], metadata['mode'], metadata['seed']) != (job.dataset, job.mode, job.seed):
        raise ValueError(f'{directory}/metadata.json does not match {job}')
    expected = metadata['artifact_sha256'][MODEL_INPUT]
    if sha256(directory / MODEL_INPUT) != expected:
        raise ValueError(f'{directory}/{MODEL_INPUT} differs from the hash in its metadata.json')
    return load_model_input(directory), expected


def hyperparameters(job: Job, X: np.ndarray, input_sha: str, settings: Settings, paths: Paths) -> tuple[dict, dict]:
    if job.model == 'Sinkhorn':
        return {}, {'source': 'paper', 'reference': 'Muzellec et al. (ICML 2020), Section 4'}
    if not settings.tune:
        return {'fraction': DEFAULT_FRACTION}, {'source': 'default'}
    key = {'input_sha256': input_sha, **{k: v for k, v in settings.of(job.model).items() if k != 'n_boot'}}
    path = job.hyperparameter_file(paths)
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved['key'] == key:
            return saved['params'], {'source': 'tuned', 'file': path.as_posix()}
    result = tune_softimpute(X, job.seed, settings.soft_grid)
    _write_json(path, {'model': job.model, 'dataset': job.dataset, 'mode': job.mode, 'rate': job.rate,
                       'seed': job.seed, 'key': key, **result})
    return result['params'], {'source': 'tuned', 'file': path.as_posix()}


def run_job(job: Job, settings: Settings, paths: Paths) -> dict:
    import torch
    torch.set_num_threads(1)
    frame, input_sha = model_input(job, paths)
    X = frame.to_numpy(dtype=float)
    params, source = hyperparameters(job, X, input_sha, settings, paths)
    if job.model == 'SoftImpute':
        filled, info = impute_softimpute(X, job.seed, params['fraction'], grid=settings.soft_grid)
    else:
        filled, info = impute_sinkhorn(X, job.seed, visits=settings.sinkhorn_visits, batch=settings.sinkhorn_batch,
                                       lr=settings.sinkhorn_lr, noise=settings.sinkhorn_noise,
                                       eps_share=settings.sinkhorn_eps_share)
    visible = ~np.isnan(X)
    if not np.isfinite(filled).all() or not np.array_equal(filled[visible], X[visible]):
        raise FloatingPointError(f'{job.label()}: non-finite values or changed visible cells')

    prediction = job.prediction_path(paths)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filled, index=frame.index, columns=frame.columns).to_csv(prediction, index_label='row_id')
    local = job.local_dir(paths)
    local.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(info.history).to_csv(local / 'training_history.csv', index=False)
    prediction_sha = sha256(prediction)
    _write_json(prediction.parent / 'metadata.json', {
        'dataset': job.dataset, 'mode': job.mode, 'rate': job.rate, 'seed': job.seed, 'model': job.model,
        'run_id': job.run_id, 'params': params, 'input_sha256': input_sha, 'prediction_sha256': prediction_sha,
        'local_artifacts': local.as_posix()})
    environment = _environment()
    _write_json(local / 'run.json', {'settings': settings.of(job.model), 'params': params, 'details': info.details,
                                     'fit_seconds': info.seconds, 'environment': environment})

    # Imported only now, after the predictions exist, so the model code never touches evaluation files.
    from shared_evaluation.evaluate import evaluate_files
    report, report_path = evaluate_files(job.dataset, job.mode, job.rate, job.seed, {job.model: prediction},
                                         paths.processed, paths.results, n_boot=settings.n_boot, boot_seed=job.seed,
                                         run_downstream=True, raw_dir=paths.raw, model=job.model, run_id=job.run_id)
    scores = report['accuracy_and_distribution']['methods'][job.model]
    headline = {name: {'estimate': scores[name]['estimate'], 'ci': scores[name]['ci']} for name in HEADLINE}
    if 'methods' in report.get('downstream', {}):
        gap = report['downstream']['methods'][job.model]['gap_closed']
        headline['gap_closed'] = {'estimate': gap['estimate'], 'ci': gap['ci']}
    experiment = {
        'model': job.model, 'dataset': job.dataset, 'mode': job.mode, 'rate': job.rate, 'seed': job.seed,
        'run_id': job.run_id, 'params': params, 'hyperparameters': source, 'settings': settings.of(job.model),
        'details': info.details, 'fit_seconds': info.seconds, 'headline': headline,
        'predictions': prediction.as_posix(), 'prediction_sha256': prediction_sha, 'input_sha256': input_sha,
        'evaluation_v2': report_path.as_posix(), 'local_artifacts': local.as_posix(), 'environment': environment}
    _write_json(job.experiment_file(paths), experiment)
    return experiment


def saved_experiment(job: Job, paths: Paths) -> dict | None:
    file = job.experiment_file(paths)
    return json.loads(file.read_text()) if file.is_file() else None


def is_done(job: Job, settings: Settings, paths: Paths) -> bool:
    saved = saved_experiment(job, paths)
    return saved is not None and saved['settings'] == settings.of(job.model) and \
        job.prediction_path(paths).is_file() and sha256(job.prediction_path(paths)) == saved['prediction_sha256']


def clashes(jobs: list[Job], settings: Settings, paths: Paths) -> list[Job]:
    """Saved runs with this run ID but other settings: rerunning would overwrite them."""
    return [j for j in jobs if (s := saved_experiment(j, paths)) is not None and s['settings'] != settings.of(j.model)]


# ---- paired comparison and summaries ----

def compare(members: list[Job], settings: Settings, paths: Paths) -> dict:
    from shared_evaluation.evaluate import evaluate_files
    job = members[0]
    report, path = evaluate_files(job.dataset, job.mode, job.rate, job.seed,
                                  {m.model: m.prediction_path(paths) for m in members}, paths.processed,
                                  paths.results, n_boot=settings.n_boot, boot_seed=job.seed, run_downstream=True,
                                  raw_dir=paths.raw, model=COMPARISON, run_id=job.run_id)
    return {'path': path.as_posix(), 'differences': report['accuracy_and_distribution']['differences']}


def comparison_done(members: list[Job], settings: Settings, paths: Paths) -> bool:
    folder = paths.results / COMPARISON / 'evaluation' / members[0].run_path(paths)
    wanted = {m.model: sha256(m.prediction_path(paths)) for m in members}
    for report in folder.glob('evaluation_v2_*.json'):
        saved = json.loads(report.read_text())
        if {n: p['sha256'] for n, p in saved['predictions'].items()} == wanted and \
                saved['accuracy_and_distribution']['n_boot'] == settings.n_boot:
            return True
    return False


def write_summaries(paths: Paths, models) -> list[Path]:
    """results/softimpute_sinkhorn/<Model>/evaluation/summary/benchmark_summary.csv: one row per saved run."""
    written = []
    for model in models:
        rows = []
        for file in sorted((paths.results / model / 'experiment').glob('*/*/*/*/experiment.json')):
            e = json.loads(file.read_text())
            scores = json.loads(Path(e['evaluation_v2']).read_text())['accuracy_and_distribution']['methods'][model]
            row = {k: e[k] for k in ('dataset', 'mode', 'rate', 'seed', 'run_id', 'fit_seconds')}
            row.update(params=json.dumps(e['params']), hyperparameters=e['hyperparameters']['source'],
                       **{name: scores[name]['estimate'] for name in RAW_SCORES})
            for name, value in e['headline'].items():
                row.update({name: value['estimate'], f'{name}_ci_low': value['ci'][0],
                            f'{name}_ci_high': value['ci'][1]})
            rows.append(row)
        if rows:
            target = paths.results / model / 'evaluation' / 'summary' / 'benchmark_summary.csv'
            text = pd.DataFrame(rows).sort_values(['mode', 'dataset', 'rate', 'run_id']).to_csv(
                index=False, lineterminator='\n')
            if not target.is_file() or target.read_text(encoding='utf-8').replace('\r\n', '\n') != text:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode('utf-8'))
            written.append(target)
    return written


# ---- running ----

def run(jobs: list[Job], settings: Settings = Settings(), paths: Paths = Paths(), workers: int = 8,
        log=print) -> list[dict]:
    clash = clashes(jobs, settings, paths)
    if clash:
        raise ValueError('These runs are saved with other settings; use a new --run-id: '
                         + ', '.join(j.label() for j in clash))
    todo = sorted([j for j in jobs if not is_done(j, settings, paths)],
                  key=lambda j: -estimate_minutes(j, settings, paths))     # longest first
    log(f'Runs: {len(jobs) - len(todo)} of {len(jobs)} already done, {len(todo)} to run')
    _execute([(run_job, (j, settings, paths)) for j in todo], workers, log,
             lambda task, e: f"{task[1][0].label()}: RMSE improvement {e['headline']['relative_improvement']['estimate']:.3f}, "
                             f"Wasserstein improvement {e['headline']['wasserstein_improvement']['estimate']:.3f}, "
                             f"params {e['params']}, fit {e['fit_seconds'] / 60:.1f} min")
    groups = {}
    for job in jobs:
        groups.setdefault((job.dataset, job.mode, job.rate, job.seed, job.run_id), []).append(job)
    pairs = [g for g in groups.values() if len(g) > 1]
    left = [g for g in pairs if not comparison_done(g, settings, paths)]
    log(f'Paired comparisons: {len(pairs) - len(left)} of {len(pairs)} already done, {len(left)} to run')

    def describe(task, result):
        key = next(iter(result['differences']))
        return f"{task[1][0][0].dataset} {task[1][0][0].mode} {task[1][0][0].rate:g}: {key} RMSE improvement " \
               f"p = {result['differences'][key]['relative_improvement']['p_value']:.3g}"
    _execute([(compare, (g, settings, paths)) for g in left], workers, log, describe)
    for path in write_summaries(paths, sorted({j.model for j in jobs})):
        log(f'Saved {path.as_posix()}')
    return [saved_experiment(j, paths) for j in jobs]


def _execute(tasks, workers, log, describe):
    if not tasks:
        return
    start = time.time()
    with parallel_config(backend='loky', inner_max_num_threads=1):
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
    temporary.replace(path)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot save {type(value).__name__}')


def _environment() -> dict:
    import torch
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'torch': torch.__version__, 'machine': platform.machine(), 'system': platform.system(),
            'code_commit': commit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--datasets', nargs='+', choices=[*DATASETS, *OPTIONAL], default=list(DATASETS),
                        help='Default: the four required datasets')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full')
    parser.add_argument('--rates', nargs='+', type=float, help='Default: every rate prepared for the seed')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--run-id', default='initial')
    parser.add_argument('--sinkhorn-visits', type=float, default=SINKHORN_VISITS,
                        help='Sinkhorn budget: times each row is drawn into a batch')
    parser.add_argument('--no-tune', action='store_true', help=f'SoftImpute with lambda = {DEFAULT_FRACTION} x top')
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--plan', action='store_true', help='Only show what is done and what is left')
    args = parser.parse_args(argv)
    from shared_evaluation.result_paths import validate_identifier
    validate_identifier(args.run_id, 'Run ID')
    settings = Settings(n_boot=args.n_boot, tune=not args.no_tune, sinkhorn_visits=args.sinkhorn_visits)
    paths = Paths()
    jobs = plan(args.models, args.datasets, args.mode, args.rates, args.seed, args.run_id, paths)
    if args.plan:
        for job in jobs:
            state = 'done' if is_done(job, settings, paths) else f'about {estimate_minutes(job, settings, paths):.0f} min'
            print(f'  {job.label():45} {state}')
        return
    run(jobs, settings, paths, workers=args.jobs)


if __name__ == '__main__':
    main()
