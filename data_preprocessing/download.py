"""Explicit official-source downloads. Imports never access the network."""
import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from .dataset_config import get_dataset

CALIFORNIA_RELEASE = 'https://doi.org/10.17632/hhb6nwfft8.1'
CALIFORNIA_CSV = 'https://data.mendeley.com/public-files/datasets/hhb6nwfft8/files/891e5207-9af4-48d6-b52d-1786310ef8a5/file_downloaded'
CALIFORNIA_SHA256 = '94ef369b909f5c725340b512b7c9a05c1550a554b6592d29303990e3ba521a1f'
HMDA_QUERY = 'https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=RI&years=2023&actions_taken=1'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_url(url, timeout):
    """(bytes, final URL after HTTP redirects); gzip-compressed bodies are decompressed."""
    with urlopen(url, timeout=timeout) as response:
        body, final = response.read(), response.geturl()
    return (gzip.decompress(body) if body[:2] == b'\x1f\x8b' else body), final

def hmda_csv(query=HMDA_QUERY):
    """Bytes of the HMDA Data Browser CSV for a query, and the URL they came from.

    The Data Browser redirects a CSV query to a static file on files.ffiec.cfpb.gov, either by HTTP redirect or with
    a short notice linking to it; the notice is followed once. Responses may be gzip-compressed even when not requested.
    """
    body, url = _read_url(query, 300)
    link = re.search(rb'href="(https://files\.ffiec\.cfpb\.gov/[^"]+\.csv)"', body)
    if link:
        body, url = _read_url(link.group(1).decode(), 300)
    if not body.startswith(b'activity_year,'):
        raise ValueError(f'Unexpected HMDA Data Browser response from {url}')
    return body, url

def load_raw(dataset, raw_dir='data/raw'):
    config = get_dataset(dataset)
    folder = Path(raw_dir) / dataset
    frames = []
    for filename in config.raw_files:
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; explicitly run without --skip-download to fetch')
        frame = pd.read_csv(path, low_memory=False)   # whole-file type inference; values unchanged
        if frame.empty:
            raise ValueError(f'Empty source file {path}')
        frame.index = [f'{filename}:{i}' for i in range(len(frame))]
        frames.append(frame)
    return pd.concat(frames), {name: sha256(folder / name) for name in config.raw_files}

def download(dataset, raw_dir='data/raw'):
    from .clean import clean
    config = get_dataset(dataset)
    folder = Path(raw_dir) / dataset
    if all((folder / name).exists() for name in config.raw_files):
        frame, _ = load_raw(dataset, raw_dir)
        clean(dataset, frame)
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    release = {}
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
            # Licensed, versioned CSV export of the same scikit-learn dataset.
            with urlopen(CALIFORNIA_CSV, timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != CALIFORNIA_SHA256:
                raise ValueError('California Housing release checksum mismatch')
            (folder / 'source.csv').write_bytes(content)
            release = {'release': CALIFORNIA_RELEASE, 'license': 'CC BY 4.0',
                       'storage': 'Byte-identical Deshpande CSV export of scikit-learn California Housing'}
        elif dataset == 'hmda_ri_2023':
            content, file_url = hmda_csv()
            (folder / 'source.csv').write_bytes(content)
            release = {'query': HMDA_QUERY, 'file_url': file_url,
                       'license': 'Public domain (CFPB legal notice, consumerfinance.gov/privacy/website-privacy-policy/); '
                                  'cite the CFPB/FFIEC as the source',
                       'storage': 'Byte-identical HMDA Data Browser CSV (Rhode Island, 2023, originated loans)'}
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
            'sha256': hashes, 'storage': 'Original yearly CSV bytes for gas; uncleaned API table export otherwise',
            **release
        }, indent=2), encoding='utf-8')
    except Exception as exc:
        raise RuntimeError(f'Download/validation failed for {dataset} from {config.source}: {exc}') from exc
    return folder
