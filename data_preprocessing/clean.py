"""Deterministic complete-case cleaning, separate from sampling and masking."""
import numpy as np
import pandas as pd
from .dataset_config import DATASETS

def clean(dataset, raw):
    config = DATASETS[dataset]
    raw = raw.copy()
    if dataset == 'credit_card':
        aliases = {'X1': 'LIMIT_BAL', 'X5': 'AGE'}
        aliases.update({f'X{i+11}': f'BILL_AMT{i}' for i in range(1, 7)})
        aliases.update({f'X{i+17}': f'PAY_AMT{i}' for i in range(1, 7)})
        raw = raw.rename(columns=aliases)
    if not raw.columns.is_unique or not raw.index.is_unique:
        raise ValueError('Source columns and row identifiers must be unique')
    selected = raw.loc[:, list(config.features)].apply(pd.to_numeric, errors='coerce')
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(selected) < 2 or (selected.nunique() < 2).any():
        raise ValueError('Insufficient complete observations or constant selected feature')
    selected.index = selected.index.map(str)
    selected.index.name = 'row_id'
    if not selected.index.is_unique:
        raise ValueError('Row identifiers collide when serialized')
    return selected, {'raw_rows': len(raw), 'clean_rows': len(selected),
                      'dropped_invalid_rows': len(raw) - len(selected),
                      'duplicate_policy': 'retain distinct source observations', 'rules': config.cleaning}

def select_rows(frame, mode='quick', seed=0):
    if mode not in ('quick', 'full'):
        raise ValueError('mode must be quick or full')
    if mode == 'quick' and len(frame) > 3000:
        positions = np.sort(np.random.default_rng(seed).choice(len(frame), 3000, replace=False))
        return frame.iloc[positions].copy()
    return frame.copy()
