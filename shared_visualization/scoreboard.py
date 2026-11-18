"""Figure 2: the scoreboard. How much better than mean imputation is each method, and which gaps are real?

Rows of panels are scores, columns are datasets, and each method is a dot with its bootstrap interval. Methods
keep the same row in every panel (traditional above deep learning). The grey line at 0 is mean imputation.

A filled dot is not significantly worse than the best method in its panel; a hollow dot is. The test is the
report's paired comparison with the best method, Holm-adjusted within the panel. Overlapping intervals can't show
this (Schenker & Gentleman, 2001), which is why it is marked. Dots, not bars: position on a shared scale is read
most accurately (Cleveland & McGill, 1984).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .results import Results, missing_pairs_warning, one_setting, point_values
from .stats import holm
from .style import (GRID, INK, INK_2, MUTED, SLIDE, SURFACE, Plot, dataset_label, finish, higher_is_better,
                    in_style, metric_label, model_style, ordered_datasets, ordered_methods, rate_label,
                    source_note, style_axes)

HEADLINE = ('relative_improvement', 'wasserstein_improvement', 'gap_closed')


@in_style
def scoreboard(results: Results, metrics=HEADLINE, mode=None, rate=None, seed=None, datasets=None, methods=None,
               point: str = 'auto', alpha: float = 0.05, title: str | None = None,
               note: str | None = None) -> Plot:
    """Dots with intervals: one row of panels per score, one panel per dataset, one setting (mode, rate, seed)."""
    results, setting = one_setting(results, mode, rate, seed)
    scores = results.scores.assign(value=point_values(results.scores, point))
    datasets = ordered_datasets(datasets or scores['dataset'].unique())
    methods = ordered_methods(methods or scores['method'].unique())
    if not methods:
        raise ValueError('No methods to plot: the reports only contain the mean and true baselines')
    positions, separators = _positions(methods)

    figure = plt.figure(figsize=SLIDE, layout='constrained')
    rows = np.atleast_1d(figure.subfigures(len(metrics), 1))
    table, significance_known = [], True
    for r, (metric, row) in enumerate(zip(metrics, rows)):
        axes = row.subplots(1, len(datasets), sharex=True, sharey=True, squeeze=False)[0]
        row.suptitle(metric_label(metric), x=0.0, ha='left', fontsize=15, color=INK)
        larger = higher_is_better(metric)
        _row_limits(axes, scores[(scores['metric'] == metric) & scores['method'].isin(methods)
                                 & scores['dataset'].isin(datasets)])
        for dataset, ax in zip(datasets, axes):
            panel = scores[(scores['dataset'] == dataset) & (scores['metric'] == metric)
                           & scores['method'].isin(methods)].set_index('method')
            defined = panel[np.isfinite(panel['value'])]
            flags = None
            if len(defined):
                pairs = results.pairs[(results.pairs['dataset'] == dataset) & (results.pairs['metric'] == metric)]
                flags, best = _significance(defined['value'], pairs, larger, alpha)
                significance_known &= flags is not None
            for method in defined.index:
                entry = defined.loc[method]
                worse = bool(flags.get(method)) if flags else False
                _dot(ax, entry['value'], entry['ci_low'], entry['ci_high'], positions[method],
                     model_style(method).colour, hollow=worse)
                table.append({**setting, 'dataset': dataset, 'metric': metric, 'method': method,
                              'value': entry['value'], 'ci_low': entry['ci_low'], 'ci_high': entry['ci_high'],
                              'best': method == best,
                              'significantly_worse': None if flags is None else worse})
            if not len(defined):
                ax.text(0.5, 0.5, 'Not defined' if len(panel) else 'Not available', transform=ax.transAxes,
                        ha='center', va='center', color=MUTED, fontsize=13)
            ax.axvline(0, color=MUTED, linewidth=1.5, zorder=1)
            for y in separators:
                ax.axhline(y, color=GRID, linewidth=0.8, zorder=0)
            style_axes(ax, 'x')
            if r == 0:
                ax.set_title(dataset_label(dataset))
        axes[0].set_yticks([positions[m] for m in methods], [model_style(m).label for m in methods])
        axes[0].set_ylim(max(positions.values()) + 0.7, -0.7)

    if significance_known:
        legend = ([Line2D([], [], marker='o', linestyle='none', markersize=9, markerfacecolor=INK_2,
                          markeredgecolor=INK_2),
                   Line2D([], [], marker='o', linestyle='none', markersize=9, markerfacecolor=SURFACE,
                          markeredgecolor=INK_2, markeredgewidth=2)],
                  ['Not significantly worse than the best in its panel',
                   f'Significantly worse (paired bootstrap, Holm, α = {alpha:g})'])
    else:
        legend = None
        missing_pairs_warning('scoreboard')
    estimate = {'auto': 'estimate (bias-corrected for Wasserstein)', 'estimate': 'estimate',
                'bias_corrected': 'bias-corrected estimate'}[point]
    finish(figure,
           title or f"How much better than mean imputation? {setting['mode'].capitalize()} data, "
                    f"{rate_label(setting['rate'])}",
           note if note is not None else source_note(
               datasets, f'Dot: {estimate}; line: bootstrap interval over rows from the evaluation report (this mask '
                         'only). 0 = mean imputation.'
                         + ('' if significance_known else ' Significance not marked: no paired p-values.')),
           legend=legend)
    return Plot(figure, pd.DataFrame(table))


def _row_limits(axes, rows):
    """Same x-range for a row of panels: every dot and interval plus 0, with a margin."""
    values = np.concatenate([rows['value'], rows['ci_low'], rows['ci_high'], [0.0]])
    values = values[np.isfinite(values)]
    low, high = values.min(), values.max()
    pad = 0.06 * (high - low or 1)
    axes[0].set_xlim(low - pad, high + pad)


def _positions(methods):
    """y position of each method, with a gap between families. Returns ({method: y}, [separator y])."""
    positions, separators, y, previous = {}, [], 0.0, None
    for method in methods:
        family = model_style(method).family
        if previous is not None and family != previous:
            y += 0.5
            separators.append(y - 0.75)   # halfway between the last row of one family and the first of the next
        positions[method] = y
        previous, y = family, y + 1
    return positions, separators


def _significance(values: pd.Series, pairs: pd.DataFrame, larger: bool, alpha: float):
    """{method: significantly worse than the best?} and the best method; None if a p-value is missing."""
    best = values.idxmax() if larger else values.idxmin()
    others = [m for m in values.index if m != best]
    p_values = []
    for method in others:
        hit = pairs[((pairs['method_a'] == best) & (pairs['method_b'] == method))
                    | ((pairs['method_a'] == method) & (pairs['method_b'] == best))]
        if hit.empty or not np.isfinite(hit['p_value'].iloc[0]):
            return None, best
        p_values.append(float(hit['p_value'].iloc[0]))
    flags = {m: adjusted < alpha for m, adjusted in zip(others, holm(p_values))} if others else {}
    flags[best] = False
    return flags, best


def _dot(ax, value, low, high, y, colour, hollow):
    if np.isfinite(low) and np.isfinite(high):
        ax.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle='round', zorder=2)
    ax.plot([value], [y], marker='o', linestyle='none', markersize=9, zorder=3,
            markerfacecolor=SURFACE if hollow else colour, markeredgecolor=colour if hollow else SURFACE,
            markeredgewidth=2 if hollow else 1.5)
