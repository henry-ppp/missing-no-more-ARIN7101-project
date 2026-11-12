"""Explicit official-source downloads. Imports never access the network."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from .dataset_config import DATASETS

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_raw(dataset, raw_dir='data/raw'):
    config = DATASETS[dataset]
    folder = Path(raw_dir) / dataset
    frames = []
    for filename in config.raw_files:
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; explicitly run without --skip-download to fetch')
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f'Empty source file {path}')
        frame.index = [f'{filename}:{i}' for i in range(len(frame))]
        frames.append(frame)
    return pd.concat(frames), {name: sha256(folder / name) for name in config.raw_files}

def download(dataset, raw_dir='data/raw'):
    from .clean import clean
    config = DATASETS[dataset]
    folder = Path(raw_dir) / dataset
    if all((folder / name).exists() for name in config.raw_files):
        frame, _ = load_raw(dataset, raw_dir)
        clean(dataset, frame)
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if dataset == 'gas_turbine':
            url = 'https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip'
            with urlopen(url, timeout=120) as response:
                content = response.read()
            with ZipFile(BytesIO(content)) as archive:
                for name in config.raw_files:
                    matches = [n for n in archive.namelist() if Path(n).name == name]
                    if len(matches) != 1:
                        raise ValueError(f'Expected exactly one member {name}, found {matches}')
                    data = archive.read(matches[0])
                    frame = pd.read_csv(BytesIO(data))
                    if tuple(frame.columns) != config.features or frame.empty:
                        raise ValueError(f'Unexpected gas schema {name}')
                    (folder / name).write_bytes(data)
            (folder / 'original.zip').write_bytes(content)
        elif dataset == 'california_housing':
            from sklearn.datasets import fetch_california_housing
            bunch = fetch_california_housing(data_home=str(folder / 'sklearn_cache'), as_frame=True)
            bunch.frame.to_csv(folder / 'source.csv', index=False)
        else:
            try:
                from ucimlrepo import fetch_ucirepo
            except ImportError as exc:
                raise ImportError('Install requirements.txt; ucimlrepo is required') from exc
            bunch = fetch_ucirepo(id=config.uci_id)
            bunch.data.original.to_csv(folder / 'source.csv', index=False)
        frame, hashes = load_raw(dataset, raw_dir)
        clean(dataset, frame)
        (folder / 'source_metadata.json').write_text(json.dumps({
            'source': config.source, 'uci_id': config.uci_id, 'rows': len(frame),
            'sha256': hashes, 'storage': 'Original yearly CSV bytes for gas; uncleaned API table export otherwise'
        }, indent=2), encoding='utf-8')
    except Exception as exc:
        raise RuntimeError(f'Download/validation failed for {dataset} from {config.source}: {exc}') from exc
    return folder
