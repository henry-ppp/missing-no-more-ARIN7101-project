"""Read saved evaluation reports into two tidy tables. This is the only module that knows report formats.

Every figure takes these two tables, so the figures never depend on how the scores were computed:

scores, one row per setting, method and score:
    dataset, mode, rate, seed, method, metric, estimate, bias_corrected, ci_low, ci_high, source
    (bias_corrected, ci_low and ci_high may be empty)

pairs, one row per paired comparison of two methods on one score (optional; used to mark significance):
    dataset, mode, rate, seed, method_a, method_b, metric, difference, p_value, source
    (difference = method_a - method_b)

Readers exist for shared_evaluation version 1 and version 2 JSON reports. Any other evaluator can write these two
tables as CSV files (write_tables), and every figure works with them unchanged. If a report format changes, only
the matching reader below changes.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SETTING = ['dataset', 'mode', 'rate', 'seed']
SCORE_COLUMNS = SETTING + ['method', 'metric', 'estimate', 'bias_corrected', 'ci_low', 'ci_high', 'source']
PAIR_COLUMNS = SETTING + ['method_a', 'method_b', 'metric', 'difference', 'p_value', 'source']
DOWNSTREAM_NAMES = {'score': 'downstream_score', 'gap_closed': 'gap_closed'}


@dataclass
class Results:
    scores: pd.DataFrame
    pairs: pd.DataFrame

    def __post_init__(self):
        self.scores = _conform(self.scores, SCORE_COLUMNS, 'scores')
        self.pairs = _conform(self.pairs, PAIR_COLUMNS, 'pairs')

    def select(self, **filters) -> 'Results':
        """Keep rows matching every filter, e.g. select(mode='quick', rate=0.3, dataset=['credit_card']).

        A filter given as None is ignored. Filters on columns a table lacks (method, metric) apply to the other.
        """
        return Results(_filter(self.scores, filters), _filter(self.pairs, filters))

    def settings(self, *columns) -> list:
        """Distinct values of setting columns, e.g. settings('mode') or settings('mode', 'rate')."""
        values = self.scores[list(columns)].drop_duplicates().sort_values(list(columns))
        return [tuple(r) if len(columns) > 1 else r[0] for r in values.itertuples(index=False)]


def load_results(*sources, rename: dict | None = None, pattern: str = 'evaluation*.json') -> Results:
    """Read reports from files and folders (folders are searched for evaluation*.json).

    Tidy CSV files (from write_tables) can be given too. rename maps report names to model names, e.g.
    {'dae_v2_initial': 'dae_v2'}. The same score for the same method and setting may appear in several reports
    (the mean baseline appears in every version 2 report); identical values are merged, keeping the copy with an
    interval, and different values raise an error naming the files.
    """
    files = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files.extend(sorted(path.rglob(pattern)))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    if not files:
        raise FileNotFoundError(f'No reports found in {[str(s) for s in sources]}')
    parts = [read_tables(f) if f.suffix == '.csv' else read_report(f) for f in files]
    results = Results(pd.concat([p.scores for p in parts], ignore_index=True),
                      pd.concat([p.pairs for p in parts], ignore_index=True))
    if rename:
        results = _rename(results, rename)
    return Results(_merge(results.scores, SETTING + ['method', 'metric'], 'estimate'),
                   _merge(results.pairs, SETTING + ['method_a', 'method_b', 'metric'], 'p_value'))


def read_report(path) -> Results:
    """One JSON report from shared_evaluation, version 1 or version 2 (detected from its contents)."""
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'accuracy_and_distribution' in report:
        return _from_version_2(report, str(path))
    if 'metrics' in report and 'model' in report:
        return _from_version_1(report, str(path))
    raise ValueError(f'{path}: not a shared_evaluation report (expected version 1 or version 2 JSON)')


def read_tables(path) -> Results:
    """A tidy CSV written by write_tables (or by any evaluator using the same columns). Detects scores or pairs."""
    table = pd.read_csv(path, float_precision='round_trip')
    if 'source' not in table:
        table['source'] = str(path)
    if {'method_a', 'method_b'} <= set(table.columns):
        return Results(pd.DataFrame(columns=SCORE_COLUMNS), table)
    return Results(table, pd.DataFrame(columns=PAIR_COLUMNS))


def write_tables(results: Results, directory) -> tuple[Path, Path]:
    """Write scores.csv and pairs.csv, the format every figure reads."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scores, pairs = directory / 'scores.csv', directory / 'pairs.csv'
    results.scores.to_csv(scores, index=False)
    results.pairs.to_csv(pairs, index=False)
    return scores, pairs


def derive_improvement(scores: pd.DataFrame, metric: str = 'mae', baseline: str = 'mean',
                       name: str | None = None) -> pd.DataFrame:
    """Add 1 - score / baseline score for a lower-is-better score (no interval), e.g. MAE improvement over mean.

    Rows that already exist under the new name are kept as they are.
    """
    name = name or f'{metric}_improvement'
    raw = scores[scores['metric'] == metric]
    base = raw[raw['method'] == baseline].set_index(SETTING)['estimate']
    rows = []
    for row in raw[raw['method'] != baseline].itertuples(index=False):
        key = tuple(getattr(row, c) for c in SETTING)
        if key in base.index and base.loc[key] > 0:
            rows.append({**{c: getattr(row, c) for c in SETTING}, 'method': row.method, 'metric': name,
                         'estimate': 1 - row.estimate / base.loc[key], 'source': row.source})
    added = _conform(pd.DataFrame(rows, columns=SCORE_COLUMNS), SCORE_COLUMNS, 'scores')
    have = scores[scores['metric'] == name].set_index(SETTING + ['method']).index
    added = added[~added.set_index(SETTING + ['method']).index.isin(have)]
    if added.empty:
        return scores
    return _conform(pd.concat([scores, added], ignore_index=True), SCORE_COLUMNS, 'scores')


BIASED_METRICS = ('wasserstein', 'wasserstein_improvement')


def point_values(scores: pd.DataFrame, point: str = 'auto') -> pd.Series:
    """The value to plot for each row.

    'estimate': the plain estimate. 'bias_corrected': the bias-corrected estimate where the report has one.
    'auto': bias-corrected for Wasserstein scores, whose plain value is biased upward, and plain otherwise; this
    matches the table shared_evaluation version 2 prints, so slides and printout agree.
    """
    corrected = scores['bias_corrected'].fillna(scores['estimate'])
    if point == 'estimate':
        return scores['estimate']
    if point == 'bias_corrected':
        return corrected
    if point == 'auto':
        return corrected.where(scores['metric'].isin(BIASED_METRICS), scores['estimate'])
    raise ValueError("point must be 'auto', 'estimate' or 'bias_corrected'")


def one_setting(results: Results, mode=None, rate=None, seed=None, single_rate: bool = True):
    """Narrow results to one mode, one seed and, if single_rate, one missing rate.

    A value left as None is filled in when the reports have only one; otherwise it must be chosen, so figures never
    mix settings silently. Returns (selected results, {column: chosen value}).
    """
    chosen = {}
    wanted = [('mode', mode), ('seed', seed)] + ([('rate', rate)] if single_rate else [])
    for column, value in wanted:
        if value is None:
            options = sorted(results.scores[column].dropna().unique().tolist())
            if len(options) != 1:
                raise ValueError(f'The reports cover several values of {column} ({options}); choose one with '
                                 f'{column}=...')
            value = options[0]
        chosen[column] = value
    if not single_rate and rate is not None:
        chosen['rate'] = rate
    selected = results.select(**chosen)
    if selected.scores.empty:
        raise ValueError(f'No scores for {chosen}')
    return selected, chosen


# ---- readers ----

def _from_version_2(report, source):
    setting = {c: report[c] for c in SETTING}
    scores, pairs = [], []
    sections = [(report['accuracy_and_distribution'], {})]
    if isinstance(report.get('downstream'), dict) and 'methods' in report['downstream']:
        sections.append((report['downstream'], DOWNSTREAM_NAMES))
    for section, names in sections:
        for method, entries in section['methods'].items():
            for metric, entry in entries.items():
                scores.append({**setting, 'method': method, 'metric': names.get(metric, metric),
                               **_entry(entry), 'source': source})
        for label, entries in section.get('differences', {}).items():
            a, b = label.split(' - ', 1)
            for metric, entry in entries.items():
                pairs.append({**setting, 'method_a': a, 'method_b': b, 'metric': names.get(metric, metric),
                              'difference': entry.get('estimate'), 'p_value': entry.get('p_value'), 'source': source})
    return Results(pd.DataFrame(scores, columns=SCORE_COLUMNS), pd.DataFrame(pairs, columns=PAIR_COLUMNS))


def _from_version_1(report, source):
    setting = {c: report[c] for c in SETTING}
    scores = [{**setting, 'method': report['model'], 'metric': metric, 'estimate': value, 'source': source}
              for metric, value in report['metrics'].items()]
    if report.get('mean_baseline_rmse') is not None:
        scores.append({**setting, 'method': 'mean', 'metric': 'rmse', 'estimate': report['mean_baseline_rmse'],
                       'source': source})
    return Results(pd.DataFrame(scores, columns=SCORE_COLUMNS), pd.DataFrame(columns=PAIR_COLUMNS))


def _entry(entry):
    ci = entry.get('ci') or [None, None]
    return {'estimate': entry.get('estimate'), 'bias_corrected': entry.get('bias_corrected'),
            'ci_low': ci[0], 'ci_high': ci[1]}


# ---- table helpers ----

def _conform(table, columns, name):
    table = table.copy()
    required = [c for c in columns if c not in ('bias_corrected', 'ci_low', 'ci_high', 'source', 'difference',
                                                'p_value')]
    missing = [c for c in required if c not in table.columns]
    if missing and len(table):
        raise ValueError(f'{name} table lacks columns {missing}')
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan
    table = table[columns + [c for c in table.columns if c not in columns]]
    for column in ('estimate', 'bias_corrected', 'ci_low', 'ci_high', 'difference', 'p_value', 'rate'):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors='coerce').astype(float)
    if 'seed' in table and len(table):
        table['seed'] = table['seed'].astype(int)
    for column in ('dataset', 'mode', 'method', 'metric', 'method_a', 'method_b'):
        if column in table and len(table):
            table[column] = table[column].astype(str)
    return table.reset_index(drop=True)


def _filter(table, filters):
    keep = pd.Series(True, index=table.index)
    for column, value in filters.items():
        if value is None or column not in table:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if column == 'rate':
            keep &= np.isclose(table[column].to_numpy()[:, None], np.asarray(values, float)[None, :]).any(axis=1)
        else:
            keep &= table[column].isin(values)
    return table[keep].reset_index(drop=True)


def _rename(results, rename):
    scores, pairs = results.scores.copy(), results.pairs.copy()
    scores['method'] = scores['method'].replace(rename)
    pairs['method_a'] = pairs['method_a'].replace(rename)
    pairs['method_b'] = pairs['method_b'].replace(rename)
    return Results(scores, pairs)


def _merge(table, keys, value):
    """One row per key. Identical copies merge (the copy with an interval or p-value wins); conflicts raise."""
    if table.empty:
        return table
    informative = table['ci_low'].notna() if 'ci_low' in table else table['p_value'].notna()
    table = table.assign(_rank=(~informative).astype(int)).sort_values('_rank', kind='stable')
    conflicts = []
    for key, group in table.groupby(keys, sort=False, dropna=False):
        values = group[value].to_numpy(dtype=float)
        known = values[np.isfinite(values)]
        if len(known) > 1 and not np.allclose(known, known[0], rtol=1e-9, atol=1e-12):
            conflicts.append(f'{dict(zip(keys, key))} in {sorted(set(group["source"]))}')
    if conflicts:
        shown = '\n  '.join(conflicts[:5])
        raise ValueError(f'Different values for the same method, setting and score ({len(conflicts)} cases):\n  '
                         f'{shown}\nPass only one report per model and setting, or rename one model '
                         "(rename={'old': 'new'}).")
    merged = table.drop_duplicates(keys, keep='first').drop(columns='_rank')
    return merged.sort_index().reset_index(drop=True)


def missing_pairs_warning(what: str) -> None:
    warnings.warn(f'{what}: the reports have no paired p-values, so significance is not marked. Evaluate all '
                  'models together with shared_evaluation version 2 to get them.', stacklevel=3)
