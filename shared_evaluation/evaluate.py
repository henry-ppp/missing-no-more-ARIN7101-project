"""Independent array and file-based evaluation; never imports a model."""
import argparse
import json
from pathlib import Path
from data_preprocessing.dataset_config import DATASETS, EVALUATION_TRUTH, config_path
from data_preprocessing.prepare_all import read_matrix
from data_preprocessing.download import sha256
from .result_paths import evaluation_directory, write_report
from .metrics import rmse, mae, r2, relative_improvement

def evaluate(X_true, X_pred, mask_eval, X_mean_baseline):
    model_rmse = rmse(X_true, X_pred, mask_eval)
    baseline_rmse = rmse(X_true, X_mean_baseline, mask_eval)
    return {'rmse': model_rmse, 'mae': mae(X_true, X_pred, mask_eval),
            'r2': r2(X_true, X_pred, mask_eval),
            'relative_improvement': relative_improvement(model_rmse, baseline_rmse)}

def evaluate_files(dataset, mode, rate, seed, predictions, processed_dir='data/processed', output_dir=None, model=None, *, run_id='standalone'):
    if model is None:
        raise ValueError('Specify a model identifier for this report')
    target = evaluation_directory('results' if output_dir is None else output_dir,
                                  model, dataset, mode, rate, seed, run_id)
    directory = config_path(processed_dir, dataset, mode, rate, seed)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    for key, expected in {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed}.items():
        if metadata[key] != expected:
            raise ValueError(f'Mismatched benchmark metadata: {key}')
    for filename, digest in metadata['artifact_sha256'].items():
        if sha256(directory / filename) != digest:
            raise ValueError(f'Benchmark artifact was modified: {filename}')
    truth = read_matrix(directory / EVALUATION_TRUTH)
    if truth.index.tolist() != metadata['row_ids'] or list(truth.columns) != metadata['features']:
        raise ValueError('Ground truth identifiers differ from metadata')
    mask = read_matrix(directory / 'mask_eval.csv')
    baseline = read_matrix(directory / 'X_mean_baseline.csv')
    prediction = read_matrix(predictions)
    result = evaluate(truth, prediction, mask, baseline)
    digest = sha256(predictions)
    # Content hash distinguishes different runs of the same model and configuration.
    output = target / f'evaluation_v1_{digest}.json'
    report = {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed, 'model': model, 'run_id': run_id,
              'predictions': Path(predictions).as_posix(), 'prediction_sha256': digest,
              'benchmark_metadata_sha256': sha256(directory / 'metadata.json'),
              'mean_baseline_rmse': rmse(truth, baseline, mask), 'metrics': result}
    write_report(output, report)
    return result, output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--rate', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--run-id', default='standalone')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--output-dir', help='Override the model results root (default: results)')
    args = parser.parse_args()
    result, output = evaluate_files(**vars(args))
    print(json.dumps(result, allow_nan=False))
    print(f'Saved {output}')

if __name__ == '__main__':
    main()
