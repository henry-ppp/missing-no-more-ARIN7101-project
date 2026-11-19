"""Train DAE V2, save predictions, then invoke independent shared file evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time
import traceback
import pandas as pd
import torch
from data_preprocessing.dataset_config import DATASETS, MODEL_INPUT, config_path
from data_preprocessing.download import sha256
from data_preprocessing.prepare_all import load_model_input
from shared_evaluation.result_paths import validate_identifier
from .train import TrainConfig, fit
from .predict import predict, save_checkpoint


def write_summary(results_dir):
    root = Path(results_dir) / 'dae_v2'
    rows = []
    for file in sorted((root / 'training').glob('*/*/rate_*/*/experiment.json')):
        report = json.loads(file.read_text(encoding='utf-8'))
        rows.append({key: report[key] for key in (
            'dataset', 'mode', 'rate', 'seed', 'run_id', 'rows', 'features',
            'rmse', 'mae', 'r2', 'relative_improvement', 'mean_baseline_rmse',
            'best_epoch', 'epochs_completed', 'best_validation_loss', 'runtime_seconds', 'device')})
    rows.sort(key=lambda row: (row['mode'], row['dataset'], row['rate'], row['seed'], row['run_id']))
    target = root / 'evaluation' / 'summary' / 'benchmark_summary.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def run(dataset, mode='quick', rate=0.3, seed=0, config=TrainConfig(),
        processed_dir='data/processed', output_dir='data/output', results_dir='results', run_id='initial'):
    validate_identifier(run_id, 'Run ID')
    benchmark = config_path(processed_dir, dataset, mode, rate, seed)
    metadata_file = benchmark / 'metadata.json'
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    incomplete = load_model_input(benchmark)
    if list(incomplete.columns) != metadata['features'] or incomplete.index.tolist() != metadata['row_ids']:
        raise ValueError('Shared input identifiers differ from benchmark metadata')
    if sha256(benchmark / MODEL_INPUT) != metadata['artifact_sha256'][MODEL_INPUT]:
        raise ValueError('Shared input file was modified')
    tag = benchmark.name
    result_folder = Path(results_dir) / 'dae_v2' / 'training' / dataset / mode / tag / run_id
    prediction_folder = config_path(output_dir, dataset, mode, rate, seed) / 'dae_v2' / run_id
    if result_folder.exists() or prediction_folder.exists():
        raise FileExistsError('Run ID already exists; choose a new --run-id to preserve artifacts')
    result_folder.mkdir(parents=True)
    prediction_folder.mkdir(parents=True)
    # Training call receives only the incomplete DataFrame and hyperparameters.
    print(f'Training {dataset} {mode}: {incomplete.shape}', flush=True)
    model, training, history, validation_mask = fit(incomplete, config)
    model.feature_names = list(incomplete.columns)
    training['input_sha256'] = sha256(benchmark / MODEL_INPUT)
    training['benchmark_metadata_sha256'] = sha256(metadata_file)
    training['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    checkpoint = Path(results_dir) / 'dae_v2' / 'training' / 'checkpoints' / dataset / mode / tag / run_id / 'best.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint, model, incomplete.columns, training)
    pd.DataFrame(history).to_csv(result_folder / 'training_history.csv', index=False)
    (result_folder / 'training_summary.json').write_text(json.dumps(training, indent=2, allow_nan=False), encoding='utf-8')
    # Heldout positions only (no target values) for exact reproduction; kept local.
    pd.DataFrame(validation_mask, index=incomplete.index, columns=incomplete.columns).astype(int).to_csv(checkpoint.parent / 'validation_mask.csv')
    prediction = predict(model, incomplete, config.batch_size, training['device'])
    prediction_path = prediction_folder / 'predictions.csv'
    prediction.to_csv(prediction_path, index_label='row_id')
    (prediction_folder / 'metadata.json').write_text(json.dumps({
        'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed, 'model': 'dae_v2',
        'run_id': run_id, 'benchmark_metadata_sha256': training['benchmark_metadata_sha256'],
        'input_sha256': training['input_sha256'], 'prediction_sha256': sha256(prediction_path),
        'checkpoint': checkpoint.as_posix()}, indent=2), encoding='utf-8')
    # Only now import the evaluator, after predictions and checkpoint exist.
    from shared_evaluation.evaluate import evaluate_files
    saved, evaluation_path = evaluate_files(dataset, mode, rate, seed, {'dae_v2': prediction_path},
        processed_dir, results_dir, n_boot=0, model='dae_v2', run_id=run_id)
    point = saved['accuracy_and_distribution']['methods']['dae_v2']
    scores = {name: point[name]['estimate'] for name in ('rmse', 'mae', 'r2', 'relative_improvement')}
    report = {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed, 'run_id': run_id,
        'rows': len(incomplete), 'features': incomplete.shape[1], **scores,
        'mean_baseline_rmse': saved['accuracy_and_distribution']['methods']['mean']['rmse']['estimate'],
        **{key: training[key] for key in ('best_epoch', 'epochs_completed', 'best_validation_loss', 'runtime_seconds', 'device')},
        'checkpoint': checkpoint.as_posix(), 'predictions': prediction_path.as_posix(), 'evaluation': evaluation_path.as_posix(),
        'training_summary': (result_folder / 'training_summary.json').as_posix(), 'config': asdict(config)}
    (result_folder / 'experiment.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write_summary(results_dir)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', '--dataset', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--output-dir', default='data/output')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--run-id', default='initial')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--training-mask-rate', type=float, default=0.3)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--hidden-dims', type=int, nargs='+')
    parser.add_argument('--latent-dim', type=int)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--activation', choices=['relu', 'gelu', 'tanh'], default='relu')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    config = TrainConfig(**{key: getattr(args, key) for key in TrainConfig.__dataclass_fields__ if hasattr(args, key)})
    validate_identifier(args.run_id, 'Run ID')
    failures = []
    for dataset in args.datasets:
        try:
            run(dataset, args.mode, args.rate, args.seed, config, args.processed_dir,
                args.output_dir, args.results_dir, args.run_id)
        except Exception as exc:
            traceback.print_exc()
            failures.append({'dataset': dataset, 'mode': args.mode, 'error': repr(exc)})
    if failures:
        target = Path(args.results_dir) / 'dae_v2' / 'training' / 'failures' / f'failures_{args.mode}_{args.run_id}_{time.time_ns()}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(failures, indent=2), encoding='utf-8')
        raise SystemExit(1)

if __name__ == '__main__':
    main()
