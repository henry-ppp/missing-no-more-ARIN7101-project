"""Figure 5: overall ranking. Across all settings, which methods are best, and which are tied?

A critical-difference diagram (Demsar, 2006): methods are ranked within each setting (dataset x missing rate,
1 = best) and placed at their average rank. A bar joins methods with no significant difference, using pairwise
Wilcoxon signed-rank tests with the Holm correction, as Benavoli, Corani & Mangili (2016) recommend instead of
the Nemenyi test. The Friedman test of "all methods rank the same" is printed beside each strip.

With few settings no difference can be significant: the exact signed-rank test cannot go below 2 / 2^n with n
settings, and Holm multiplies the smallest p-value by the number of pairs. The figure then says so and shows
average ranks only. Settings on the same dataset at different rates are not independent; say so on the slide.
"""
from __future__ import annotations

import pandas as pd

from .results import Results, one_setting, point_values
from .stats import cliques, friedman, pairwise_wilcoxon, rank_matrix, smallest_wilcoxon_p, value_matrix
from .style import (AXIS, INK, INK_2, SURFACE, Plot, finish, higher_is_better, in_style, metric_label,
                    model_style, new_figure, ordered_methods, source_note)

BLOCKS = ('dataset', 'rate')


def rank_table(results: Results, metrics=('relative_improvement', 'wasserstein_improvement'), mode=None, seed=None,
               methods=None, point: str = 'auto') -> pd.DataFrame:
    """Average rank of each method per score (1 = best), with the number of settings and the Friedman p-value."""
    rows = []
    for metric, values, ranks in _rankings(results, metrics, mode, seed, methods, point):
        p = friedman(values)
        for method, rank in ranks.mean().items():
            rows.append({'metric': metric, 'method': method, 'average_rank': rank, 'settings': len(values),
                         'friedman_p': p})
    return pd.DataFrame(rows)


@in_style
def critical_difference(results: Results, metrics=('relative_improvement', 'wasserstein_improvement'), mode=None,
                        seed=None, methods=None, alpha: float = 0.05, point: str = 'auto',
                        title: str | None = None, note: str | None = None) -> Plot:
    """One critical-difference strip per score, over every dataset and missing rate in the reports."""
    rankings = list(_rankings(results, metrics, mode, seed, methods, point))
    figure, axes = new_figure(len(rankings), 1)
    table = []
    for (metric, values, ranks), ax in zip(rankings, axes[:, 0]):
        n, k = values.shape
        n_pairs = k * (k - 1) // 2
        average = ranks.mean().sort_values()
        # Test pairs only when a significant result is possible at all.
        testable = smallest_wilcoxon_p(n) * n_pairs <= alpha
        groups = cliques(average, pairwise_wilcoxon(values), alpha) if testable else []
        p = friedman(values)
        _strip(ax, average, groups, k)
        status = (f'{n} settings; Friedman {_p_text(p)}' if p is not None else f'{n} settings') + (
            '' if testable else f'. Too few settings for any significant pair: the smallest possible p-value, '
                                f'{smallest_wilcoxon_p(n):.3g}, times {n_pairs} pairs (Holm) exceeds '
                                f'{alpha:g}. Average ranks only.')
        ax.set_title(metric_label(metric), loc='left', pad=26)
        ax.annotate(status, (0, 1), xycoords='axes fraction', xytext=(0, 6), textcoords='offset points',
                    ha='left', va='bottom', color=INK_2, fontsize=12)
        membership = {m: [i for i, g in enumerate(groups) if m in g] for m in average.index}
        table += [{'metric': metric, 'method': m, 'average_rank': r, 'settings': n, 'friedman_p': p,
                   'tied_groups': ' '.join(map(str, membership[m])) or None, 'significance_testable': testable}
                  for m, r in average.items()]
    mode_name = one_setting(results, mode, None, seed, single_rate=False)[1]['mode']
    finish(figure, title or f'Overall ranking across datasets and missing rates: {mode_name.capitalize()} data',
           note if note is not None else source_note(
               results.scores['dataset'].unique(),
               'Rank 1 = best within a setting. Bars join methods with no significant difference (pairwise Wilcoxon, '
               f'Holm, α = {alpha:g}; Benavoli et al., 2016). Diagram after Demšar (2006).'))
    return Plot(figure, pd.DataFrame(table))


def _p_text(p: float) -> str:
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def _rankings(results, metrics, mode, seed, methods, point):
    results, _ = one_setting(results, mode, None, seed, single_rate=False)
    scores = results.scores.assign(value=point_values(results.scores, point))
    methods = ordered_methods(methods or scores['method'].unique())
    for metric in metrics:
        values = value_matrix(scores[scores['method'].isin(methods)], metric, 'value', BLOCKS)
        if values.shape[1] < 2 or values.empty:
            raise ValueError(f'Ranking {metric} needs at least two methods scored in the same settings')
        yield metric, values, rank_matrix(values, higher_is_better(metric))


def _strip(ax, average: pd.Series, groups, k: int):
    """Draw one critical-difference strip: rank axis on top, labels hanging left (best half) and right."""
    margin = 0.45 * max(k - 1, 1)
    ax.set_xlim(1 - margin, k + margin)
    ax.axis('off')
    ax.plot([1, k], [0, 0], color=INK_2, linewidth=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.12], color=INK_2, linewidth=1)
        ax.text(r, 0.2, str(r), ha='center', va='bottom', color=INK_2, fontsize=12)
    bar_top, bar_step = -0.3, 0.18
    for i, group in enumerate(groups):
        y = bar_top - i * bar_step
        ax.plot([average[group].min() - 0.03, average[group].max() + 0.03], [y, y], color=INK, linewidth=4,
                solid_capstyle='round')
    first = bar_top - len(groups) * bar_step - 0.25
    order = list(average.index)
    left, right = order[:(len(order) + 1) // 2], order[(len(order) + 1) // 2:][::-1]
    for side, members in (('left', left), ('right', right)):
        for level, method in enumerate(members):
            rank, style = average[method], model_style(method)
            y = first - level * 0.55
            end = 1 - margin * 0.25 if side == 'left' else k + margin * 0.25
            ax.plot([rank, rank, end], [0, y, y], color=AXIS, linewidth=1, zorder=1)
            ax.plot([rank], [0], marker=style.marker, markersize=10, color=style.colour, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)
            ax.text(end + (-0.05 if side == 'left' else 0.05), y, f'{style.label}  {rank:.2f}' if side == 'left'
                    else f'{rank:.2f}  {style.label}', ha='right' if side == 'left' else 'left', va='center',
                    color=INK, fontsize=13)
    ax.set_ylim(first - max(len(left), len(right)) * 0.55, 0.6)
