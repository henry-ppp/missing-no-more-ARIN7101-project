"""Figure 4: the effect of the missing rate. Does the gap between methods change as more values go missing?

One row of panels per score and one panel per dataset; x = missing rate. Every method is a line. To keep seven
lines readable, only the highlighted methods are coloured, with their bootstrap interval as a light band; the
others are grey. By default the highlighted methods are the best traditional and the best deep-learning method
(highest average of the first score). The grey line at 0 is mean imputation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from .results import Results, one_setting, point_values
from .style import (AXIS, FAMILIES, INK, INK_2, MUTED, SLIDE, SURFACE, Plot, dataset_label, finish,
                    higher_is_better, in_style, metric_label, model_style, ordered_datasets, ordered_methods,
                    source_note, style_axes)


@in_style
def missing_rate_curves(results: Results, metrics=('relative_improvement', 'wasserstein_improvement'), mode=None,
                        seed=None, datasets=None, methods=None, highlight=None, point: str = 'auto',
                        title: str | None = None, note: str | None = None) -> Plot:
    """Lines over missing rates for one mode and seed. Needs reports for at least two rates."""
    results, setting = one_setting(results, mode, None, seed, single_rate=False)
    scores = results.scores.assign(value=point_values(results.scores, point))
    rates = sorted(float(r) for r in scores['rate'].unique())
    if len(rates) < 2:
        raise ValueError(f'This figure needs at least two missing rates; the reports have {rates}')
    datasets = ordered_datasets(datasets or scores['dataset'].unique())
    methods = ordered_methods(methods or scores['method'].unique())
    highlight = list(highlight) if highlight is not None else default_highlight(scores, metrics[0], methods)

    figure = plt.figure(figsize=SLIDE, layout='constrained')
    rows = np.atleast_1d(figure.subfigures(len(metrics), 1))
    table = []
    for r, (metric, row) in enumerate(zip(metrics, rows)):
        axes = row.subplots(1, len(datasets), sharex=True, sharey=True, squeeze=False)[0]
        row.suptitle(metric_label(metric), x=0.0, ha='left', fontsize=15, color=INK)
        for c, (dataset, ax) in enumerate(zip(datasets, axes)):
            ax.axhline(0, color=MUTED, linewidth=1.5, zorder=1)
            ordered = [m for m in methods if m not in highlight] + [m for m in methods if m in highlight]
            for method in ordered:
                line = scores[(scores['dataset'] == dataset) & (scores['metric'] == metric)
                              & (scores['method'] == method)].sort_values('rate')
                line = line[np.isfinite(line['value'])]
                if line.empty:
                    continue
                emphasised = method in highlight
                colour = model_style(method).colour if emphasised else AXIS
                if emphasised and line[['ci_low', 'ci_high']].notna().all().all():
                    ax.fill_between(line['rate'], line['ci_low'], line['ci_high'], color=colour, alpha=0.14,
                                    linewidth=0, zorder=2)
                ax.plot(line['rate'], line['value'], color=colour, linewidth=2 if emphasised else 1.5,
                        marker=model_style(method).marker if emphasised else None, markersize=7,
                        markeredgecolor=SURFACE,
                        zorder=4 if emphasised else 3)
                if emphasised and c == len(datasets) - 1:
                    ax.annotate(model_style(method).label, (line['rate'].iloc[-1], line['value'].iloc[-1]),
                                xytext=(6, 0), textcoords='offset points', va='center', fontsize=12, color=INK_2)
                table += [{**{k: v for k, v in setting.items() if k != 'rate'}, 'dataset': dataset,
                           'metric': metric, 'rate': p.rate, 'method': method, 'value': p.value,
                           'ci_low': p.ci_low, 'ci_high': p.ci_high, 'highlighted': emphasised}
                          for p in line.itertuples(index=False)]
            ax.set_xticks(rates)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
            style_axes(ax, 'y')
            if r == 0:
                ax.set_title(dataset_label(dataset))
            if r == len(metrics) - 1:
                ax.set_xlabel('Missing rate')

    handles = [Line2D([], [], color=model_style(m).colour, linewidth=2, marker=model_style(m).marker, markersize=7)
               for m in highlight]
    labels = [model_style(m).label for m in highlight]
    if any(m not in highlight for m in methods):
        handles.append(Line2D([], [], color=AXIS, linewidth=1.5))
        labels.append('Other methods')
    handles.append(Line2D([], [], color=MUTED, linewidth=1.5))
    labels.append('Mean imputation (0)')
    finish(figure, title or f"How the missing rate changes the picture: {setting['mode'].capitalize()} data",
           note if note is not None else source_note(datasets, 'Band: bootstrap interval from the evaluation report '
                                                               '(this mask only).'),
           legend=(handles, labels))
    return Plot(figure, pd.DataFrame(table))


def default_highlight(scores: pd.DataFrame, metric: str, methods) -> list[str]:
    """The best method of each family on average over datasets and rates (largest or smallest, as fits the score)."""
    means = scores[(scores['metric'] == metric) & scores['method'].isin(methods)].groupby('method')['value'].mean()
    chosen = []
    for family in FAMILIES:
        members = means[[model_style(m).family == family for m in means.index]]
        if len(members):
            chosen.append(members.idxmax() if higher_is_better(metric) else members.idxmin())
    return [m for m in methods if m in chosen]
