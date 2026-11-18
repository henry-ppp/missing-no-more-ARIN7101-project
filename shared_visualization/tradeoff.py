"""Figure 3: accuracy vs distribution. Does an accurate method also keep the distribution?

One square panel per dataset. x = MAE improvement over mean, y = Wasserstein improvement over mean, both 0 at
mean imputation. Each dot is a method at one missing rate; dots of one method are joined in rate order.

Why MAE on x: in every column the Wasserstein-1 distance is at most the MAE, because the true row-by-row pairing
is one of the pairings the distance minimises over (Villani, 2009); for mean imputation the two are equal. So the
Wasserstein improvement is at least the column-averaged MAE improvement and the diagonal y = x is the floor that
accuracy guarantees. Height above it is how much better a method keeps the distribution than its accuracy promises.
The reports give MAE pooled over cells, which differs from the column average only slightly under MCAR, so a dot
can sit marginally below the line. Plain estimates are used by default because the floor is exact for them.

Each method has its own colour and shape (triangles = deep learning); with up to four methods the dots are also
labelled. Each panel prints Kendall's tau between the two scores across methods. Evidence that accuracy is not enough is
dots far above the line and methods that swap order between the axes, not a weak correlation alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .results import Results, derive_improvement, one_setting, point_values
from .stats import kendall
from .style import (AXIS, INK_2, MUTED, SURFACE, Plot, dataset_label, finish, in_style,
                    metric_label, model_style, new_figure, ordered_datasets, ordered_methods, source_note,
                    style_axes)

MAX_DIRECT_LABELS = 4   # more labels than this collide; the legend (colour + shape) identifies methods instead


@in_style
def accuracy_vs_distribution(results: Results, x: str = 'mae_improvement', y: str = 'wasserstein_improvement',
                             mode=None, seed=None, rate=None, datasets=None, methods=None, point: str = 'estimate',
                             title: str | None = None, note: str | None = None) -> Plot:
    """Scatter of two scores per dataset, for every missing rate in the reports (or the chosen rate)."""
    results, setting = one_setting(results, mode, rate, seed, single_rate=False)
    scores = results.scores
    derived = x == 'mae_improvement' and not (scores['metric'] == x).any()
    if derived:
        scores = derive_improvement(scores, 'mae', 'mean', x)
    scores = scores.assign(value=point_values(scores, point))
    missing = [m for m in (x, y) if not (scores['metric'] == m).any()]
    if missing:
        raise ValueError(f'The reports have no {missing} scores')
    datasets = ordered_datasets(datasets or scores['dataset'].unique())
    methods = ordered_methods(methods or scores['method'].unique())
    wide = (scores[scores['metric'].isin([x, y]) & scores['method'].isin(methods) & scores['dataset'].isin(datasets)]
            .pivot_table(index=['dataset', 'rate', 'method'], columns='metric', values='value', aggfunc='first')
            .reindex(columns=[x, y]).dropna().reset_index())
    if wide.empty:
        raise ValueError(f'No method has both {x} and {y}')

    values = np.concatenate([wide[x], wide[y], [0.0]])
    low, high = values.min(), values.max()
    pad = 0.06 * (high - low or 1)
    limits = (low - pad, high + pad)
    figure, axes = new_figure(1, len(datasets), size=(13.33, 5.6))
    table = []
    for dataset, ax in zip(datasets, axes[0]):
        panel = wide[wide['dataset'] == dataset]
        ax.plot(limits, limits, color=AXIS, linewidth=1, zorder=1)
        spot = limits[0] + 0.82 * (limits[1] - limits[0])
        ax.annotate('y = x', (spot, spot), xytext=(4, -4), textcoords='offset points', ha='left', va='top',
                    color=MUTED, fontsize=11, rotation=45, rotation_mode='anchor', in_layout=False)
        ax.plot([0], [0], marker='o', linestyle='none', markersize=8, color=MUTED, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        placed = [(0.0, 0.0)]
        _label(ax, 0.0, 0.0, 'Mean', placed, limits)
        taus = []
        for rate, at_rate in panel.groupby('rate'):
            tau, p_value = kendall(at_rate[x], at_rate[y])
            taus.append((rate, tau))
            for row in at_rate.itertuples(index=False):
                table.append({**{k: v for k, v in setting.items() if k != 'rate'}, 'dataset': dataset, 'rate': rate,
                              'method': row.method, x: getattr(row, x), y: getattr(row, y),
                              'height_above_diagonal': getattr(row, y) - getattr(row, x),
                              'kendall_tau': tau, 'kendall_p': p_value})
        for method in methods:
            points = panel[panel['method'] == method].sort_values('rate')
            if points.empty:
                continue
            style = model_style(method)
            if len(points) > 1:
                ax.plot(points[x], points[y], color=style.colour, linewidth=1.2, alpha=0.6, zorder=2)
            ax.plot(points[x], points[y], marker=style.marker, linestyle='none', markersize=10,
                    color=style.colour, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            if panel['method'].nunique() <= MAX_DIRECT_LABELS:
                last = points.iloc[-1]
                _label(ax, last[x], last[y], style.label, placed, limits)
        ax.text(0.97, 0.04, _tau_text(taus), transform=ax.transAxes, ha='right', va='bottom', color=INK_2,
                fontsize=12, linespacing=1.3, in_layout=False)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal')
        ax.set_title(dataset_label(dataset))
        ax.set_xlabel(metric_label(x))
        style_axes(ax, 'both')
    axes[0][0].set_ylabel(metric_label(y))
    legend = ([Line2D([], [], marker=model_style(m).marker, linestyle='none', markersize=10,
                      color=model_style(m).colour, markeredgecolor=SURFACE) for m in methods],
              [model_style(m).label for m in methods])
    rates = sorted(wide['rate'].unique())
    joined = ' Dots of one method are joined from the lowest to the highest missing rate (label at the highest).' \
        if len(rates) > 1 else ''
    finish(figure,
           title or f"Does accuracy predict how well the distribution is kept? {setting['mode'].capitalize()} data",
           note if note is not None else source_note(
               datasets, 'Triangles: deep learning; other shapes: traditional. Wasserstein ≤ MAE in every column '
                         '(Villani, 2009), so y ≥ x up to the small pooled-vs-column-averaged MAE difference. '
                         f'Kendall (1938) τ across methods.{joined}'),
           legend=legend, legend_columns=min(len(methods), 7))
    return Plot(figure, pd.DataFrame(table))


def _tau_text(taus):
    shown = [(rate, tau) for rate, tau in taus if tau is not None]
    if not shown:
        return 'τ: needs 3+ methods'
    if len(taus) == 1:
        return f'Kendall τ = {shown[0][1]:.2f}'
    return '\n'.join(['Kendall τ'] + [f'{rate * 100:g}%: {tau:.2f}' for rate, tau in shown])


def _label(ax, x, y, text, placed, limits):
    """Put a short label beside a point, choosing the side farthest from points already labelled."""
    span = limits[1] - limits[0]
    candidates = [(8, 6, 'left', 'bottom'), (8, -6, 'left', 'top'), (-8, 6, 'right', 'bottom'),
                  (-8, -6, 'right', 'top')]
    def clearance(candidate):
        dx, dy = np.sign(candidate[0]) * 0.06 * span, np.sign(candidate[1]) * 0.04 * span
        spot = (x + dx, y + dy)
        return min(np.hypot(spot[0] - px, spot[1] - py) for px, py in placed) if placed else np.inf
    dx, dy, ha, va = max(candidates, key=clearance)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=12,
                color=INK_2, zorder=5, in_layout=False)
    placed.append((x + np.sign(dx) * 0.06 * span, y + np.sign(dy) * 0.04 * span))
