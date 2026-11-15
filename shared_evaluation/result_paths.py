"""Shared paths for model-owned reports; no dependency on model code."""
import json
import re
from pathlib import Path

from data_preprocessing.dataset_config import config_path


def validate_identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'{label} must contain only letters, digits, underscore or hyphen')
    reserved = {'con', 'prn', 'aux', 'nul', *(f'{prefix}{n}' for prefix in ('com', 'lpt') for n in range(10))}
    if label == 'Model identifier':
        reserved.update(('evaluation', 'training', 'preprocessing', 'summary'))
    if value.lower() in reserved:
        raise ValueError(f'Reserved {label.lower()}: {value}')
    return value


def evaluation_directory(root, model, dataset, mode, rate, seed, run_id):
    model = validate_identifier(model, 'Model identifier')
    run_id = validate_identifier(run_id, 'Run ID')
    return config_path(Path(root) / model / 'evaluation', dataset, mode, rate, seed) / run_id


def write_report(path, report):
    """Reuse identical reports and refuse to replace a report with different content."""
    content = json.dumps(report, indent=2, allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        if json.loads(path.read_bytes()) != json.loads(content):
            raise FileExistsError(f'Report already exists with different content: {path}; choose another run or output directory') from None
    return path
