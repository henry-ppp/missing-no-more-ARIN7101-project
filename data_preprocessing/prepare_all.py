"""Prepare shared benchmark artifacts; ground truth is evaluator-only."""
import argparse
import json
import platform
import csv
from shutil import copyfile
from importlib.metadata import version, PackageNotFoundError
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .dataset_config import DATASETS, EVALUATION_TRUTH, MODEL_INPUT, OPTIONAL, config_path, get_dataset
from .download import download, load_raw, sha256
from .clean import clean, select_rows
from .mask import mcar_mask, apply_mask
from .standardize import fit_observed, transform
from .mean_baseline import mean_impute

def read_matrix(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        names = next(csv.reader(stream), [])
    if len(names) != len(set(names)):
        raise ValueError('Duplicate CSV column names')
    header = pd.read_csv(path, nrows=0)
    if len(header.columns) < 2 or header.columns[0] != 'row_id':
        raise ValueError(f'{path}: first CSV column must be row_id')
    frame = pd.read_csv(path, dtype={'row_id': str}).set_index('row_id')
    if not frame.index.is_unique or frame.index.isna().any() or not frame.columns.is_unique:
        raise ValueError('Duplicate or missing matrix identifiers')
    return frame

def load_model_input(directory):
    """Return only incomplete standardized input. Never opens evaluator files.

    This API boundary is not OS access control. Do not use complete matrices
    or hidden values in training or model selection.
    """
    return read_matrix(Path(directory) / MODEL_INPUT)

def export_configuration(full, dataset, mode, rate, seed, directory, cleaning, raw_hashes):
    config_path('.', dataset, mode, rate, seed)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f'Configuration exists: {directory}; reuse or select a fresh root')
    selected = select_rows(full, mode, seed)
    mask = mcar_mask(selected, rate, seed)
    if not mask.to_numpy().any():
        raise ValueError('No evaluation-hidden entries')
    missing = apply_mask(selected, mask)
    stats = fit_observed(missing)
    standardized = transform(missing, stats)
    baseline = mean_impute(standardized)
    cleaned = directory.parent.parent / 'cleaned' / 'X_cleaned.csv'
    if cleaned.exists():
        pd.testing.assert_frame_equal(read_matrix(cleaned), full, check_dtype=False, rtol=1e-14, atol=1e-14)
    else:
        cleaned.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(cleaned, index_label='row_id')
    directory.mkdir(parents=True)
    if mode == 'full':
        copyfile(cleaned, directory / 'X_cleaned.csv')
    else:
        selected.to_csv(directory / 'X_cleaned.csv', index_label='row_id')
    missing.to_csv(directory / 'X_cleaned_masked.csv', index_label='row_id')
    transform(selected, stats).to_csv(directory / EVALUATION_TRUTH, index_label='row_id')
    standardized.to_csv(directory / MODEL_INPUT, index_label='row_id')
    baseline.to_csv(directory / 'X_mean_baseline.csv', index_label='row_id')
    mask.astype(int).to_csv(directory / 'mask_eval.csv', index_label='row_id')
    (directory / 'scaler_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    package_versions = {}
    for package in ('scikit-learn', 'ucimlrepo'):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    metadata = {'schema_version': 2, 'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed,
        'dataset_config': asdict(get_dataset(dataset)), 'features': list(selected.columns),
        'row_ids': selected.index.tolist(), 'selected_rows': len(selected),
        'quick_under_limit': mode == 'quick' and len(full) < 3000,
        'cleaning': cleaning, 'raw_sha256': raw_hashes, 'mask_count': int(mask.to_numpy().sum()),
        'sampling': 'default_rng choice without replacement, sorted original positions',
        'standardization': 'Observed only; population std; zero std uses scale=1; no observed values raises error',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, **package_versions}}
    metadata['artifact_sha256'] = {p.relative_to(directory).as_posix(): sha256(p)
        for p in directory.rglob('*') if p.is_file()}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return directory

def prepare(dataset, mode='quick', rate=0.3, seed=0, raw_dir='data/raw', processed_dir='data/processed', skip_download=False):
    destination = config_path(processed_dir, dataset, mode, rate, seed)
    if destination.exists():
        metadata = json.loads((destination / 'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('schema_version') != 2:
            raise ValueError('Unsupported cached schema; use the migrated snapshot or a fresh processed root')
        if any(metadata[k] != v for k, v in {'dataset': dataset, 'mode': mode, 'rate': rate, 'seed': seed}.items()):
            raise ValueError('Cached metadata mismatch')
        for filename, digest in metadata['artifact_sha256'].items():
            if sha256(destination / filename) != digest:
                raise ValueError(f'Changed artifact {filename}; use a fresh processed root')
        cleaned = destination.parent.parent / 'cleaned' / 'X_cleaned.csv'
        if not cleaned.is_file() or (mode == 'full' and sha256(cleaned) != metadata['artifact_sha256']['X_cleaned.csv']):
            raise ValueError('Dataset-level cleaned file is missing or differs from Full')
        return destination
    if not skip_download:
        download(dataset, raw_dir)
    raw, hashes = load_raw(dataset, raw_dir)
    full, cleaning = clean(dataset, raw)
    return export_configuration(full, dataset, mode, rate, seed, destination, cleaning, hashes)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--datasets', '--dataset', nargs='+', choices=[*DATASETS, *OPTIONAL], default=list(DATASETS))
    parser.add_argument('--rate', type=float, nargs='+', default=[0.3])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--skip-download', action='store_true')
    args = parser.parse_args()
    for dataset in args.datasets:
        for rate in args.rate:
            print(prepare(dataset, args.mode, rate, args.seed, args.raw_dir, args.processed_dir, args.skip_download))

if __name__ == '__main__':
    main()
