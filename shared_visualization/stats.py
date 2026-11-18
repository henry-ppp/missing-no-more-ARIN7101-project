"""Statistics the figures need, computed from the score table only (never from predictions).

- holm: Holm (1979) step-down adjustment of a family of p-values.
- rank_matrix / average ranks: rank methods within each setting (1 = best), as in Demsar (2006).
- friedman: Friedman (1937) test that all methods have the same average rank.
- pairwise_wilcoxon: Wilcoxon (1945) signed-rank test for every pair of methods across settings, Holm-adjusted,
  as Benavoli, Corani & Mangili (2016) recommend instead of the Nemenyi test.
- cliques: groups of methods with no significant difference, drawn as bars on the ranking diagram.
- kendall: Kendall's (1938) rank correlation, used for the accuracy-vs-distribution figure.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def holm(p_values) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p, kind='stable')):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def smallest_wilcoxon_p(n: int) -> float:
    """The smallest two-sided p-value the exact signed-rank test can give with n pairs (no ties): 2 / 2^n."""
    return 2.0 / 2 ** n if n > 0 else 1.0


def value_matrix(scores: pd.DataFrame, metric: str, value: str = 'estimate', blocks=('dataset', 'rate')) -> pd.DataFrame:
    """Settings (rows) by methods (columns) for one score. Only settings where every method has a value are kept."""
    table = scores[scores['metric'] == metric].pivot_table(index=list(blocks), columns='method', values=value,
                                                           aggfunc='first')
    return table.dropna(axis=0, how='any')


def rank_matrix(values: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """Rank methods within each setting, 1 = best; ties share the average rank."""
    return values.rank(axis=1, ascending=not higher_is_better, method='average')


def friedman(values: pd.DataFrame) -> float | None:
    """p-value of the Friedman test, or None with fewer than 3 methods or 2 settings."""
    if values.shape[1] < 3 or values.shape[0] < 2:
        return None
    return float(stats.friedmanchisquare(*[values[c].to_numpy() for c in values.columns]).pvalue)


def pairwise_wilcoxon(values: pd.DataFrame) -> pd.DataFrame:
    """Signed-rank test for every pair of methods across settings, with Holm-adjusted p-values."""
    rows = []
    for a, b in combinations(values.columns, 2):
        diff = values[a].to_numpy() - values[b].to_numpy()
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(values[a].to_numpy(), values[b].to_numpy()).pvalue)
        rows.append({'method_a': a, 'method_b': b, 'n': len(diff), 'p_value': p})
    table = pd.DataFrame(rows, columns=['method_a', 'method_b', 'n', 'p_value'])
    table['p_holm'] = holm(table['p_value']) if len(table) else []
    return table


def cliques(average_ranks: pd.Series, pairwise: pd.DataFrame, alpha: float = 0.05) -> list[list[str]]:
    """Maximal runs of methods (in rank order) where no pair differs significantly. Single methods are left out."""
    order = list(average_ranks.sort_values().index)
    different = {frozenset((r.method_a, r.method_b)) for r in pairwise.itertuples() if r.p_holm < alpha}
    runs = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and all(frozenset((order[k], order[j + 1])) not in different for k in range(i, j + 1)):
            j += 1
        if j > i:
            runs.append((i, j))
    maximal = [r for r in runs if not any(o != r and o[0] <= r[0] and r[1] <= o[1] for o in runs)]
    return [order[i:j + 1] for i, j in maximal]


def kendall(x, y) -> tuple[float | None, float | None]:
    """Kendall's tau and its p-value, or (None, None) with fewer than 3 points or a constant score."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    result = stats.kendalltau(x, y)
    return float(result.statistic), float(result.pvalue)
