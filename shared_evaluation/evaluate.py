"""Independent array and file-based evaluation; never imports a model."""
import argparse
import json
from pathlib import Path
from data_preprocessing.dataset_config import DATASETS, config_path
from data_preprocessing.prepare_all import read_matrix
from data_preprocessing.download import sha256
from .metrics import rmse, mae, r2, relative_improvement

def evaluate(X_true, X_pred, mask_eval, X_mean_baseline):
    model_rmse = rmse(X_true, X_pred, mask_eval)
    baseline_rmse = rmse(X_true, X_mean_baseline, mask_eval)
    return {'rmse': model_rmse, 'mae': mae(X_true, X_pred, mask_eval),
            'r2': r2(X_true, X_pred, mask_eval),
            'relative_improvement': relative_improvement(model_rmse, baseline_rmse)}

def evaluate_files(dataset, mode, rate, seed, predictions, processed_dir='data/processed', output_dir='data/output', model=None):
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
    mask = read_matrix(directory / 'mask_eval.csv')
    baseline = read_matrix(directory / 'X_mean_baseline.csv')
    prediction = read_matrix(predictions)
    result = evaluate(truth, prediction, mask, baseline)
    digest = sha256(predictions)
    name = model or Path(predictions).stem
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Model name must contain only letters, digits, underscore or hyphen')
    target = config_path(output_dir, dataset, mode, rate, seed) / name
    target.mkdir(parents=True, exist_ok=True)
    # Content hash distinguishes different runs of the same model and configuration.
    output = target / f'evaluation_{digest}.json'
    report = {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed, 'model': name,
              'predictions': str(Path(predictions).resolve()), 'prediction_sha256': digest,
              'benchmark_metadata_sha256': sha256(directory / 'metadata.json'), 'metrics': result}
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return result, output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--model')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--output-dir', default='data/output')
    args = parser.parse_args()
    result, output = evaluate_files(**vars(args))
    print(json.dumps(result, allow_nan=False))
    print(f'Saved {output}')

if __name__ == '__main__':
    main()
