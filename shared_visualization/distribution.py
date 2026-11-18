"""Figure 1: what the distribution score measures.

For one column, the cumulative distribution of the true hidden values is drawn against each method's guesses.
The shaded area between the two curves equals the Wasserstein-1 distance between them (Vallender, 1974), so the
audience sees the score itself. The printed number is that shaded area, measured from the drawn curves.

This figure needs the values themselves, not the scores, so it reads prediction files: the truth, the evaluation
mask and each method's predictions, all CSV matrices with a row_id first column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .style import (BASELINES, INK, INK_2, MUTED, Plot, dataset_label, finish, in_style, model_style, new_figure,
                    source_note, style_axes)


def hidden_values(truth, mask, predictions: dict, column: str):
    """True and imputed values at the hidden cells of one column.

    truth, mask and each prediction are DataFrames or CSV paths with a row_id first column and identical rows and
    columns (the shared benchmark files: the complete standardised matrix, mask_eval.csv, predictions.csv).
    Returns (true values, {name: imputed values}).
    """
    truth, mask = _matrix(truth), _matrix(mask)
    if column not in truth.columns:
        raise ValueError(f'{column!r} is not a column of the truth matrix')
    _same_layout(truth, mask, 'mask')
    hidden = mask[column].astype(bool).to_numpy()
    if not hidden.any():
        raise ValueError(f'No hidden cells in {column!r}')
    imputed = {}
    for name, prediction in predictions.items():
        prediction = _matrix(prediction)
        _same_layout(truth, prediction, name)
        values = prediction[column].to_numpy(dtype=float)[hidden]
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: non-finite predictions at hidden cells')
        imputed[name] = values
    return truth[column].to_numpy(dtype=float)[hidden], imputed


def ecdf_area(a, b) -> float:
    """Area between the empirical cumulative distributions of a and b (equal to their Wasserstein-1 distance)."""
    grid, fa, fb = _ecdfs(a, b)
    return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * np.diff(grid)))


@in_style
def distribution_figure(true_values, imputed: dict, column: str, dataset: str | None = None,
                        view_quantiles=(0.005, 0.995), title: str | None = None, note: str | None = None) -> Plot:
    """One panel per method: true hidden values (dark line) against the method's guesses (its colour).

    view_quantiles limits the x-axis to the central range of the true values, so a few extreme values don't
    squeeze the picture; the shaded areas are always measured on all values. Use None to show everything.
    """
    if not imputed:
        raise ValueError('Give at least one set of imputed values')
    true_values = np.asarray(true_values, dtype=float)
    figure, axes = new_figure(1, len(imputed), sharey=True)
    rows = []
    for ax, (name, values) in zip(axes[0], imputed.items()):
        colour = MUTED if name in BASELINES else model_style(name).colour
        label = 'Mean imputation' if name == 'mean' else model_style(name).label
        grid, f_true, f_model = _ecdfs(true_values, values)
        area = ecdf_area(true_values, values)
        ax.fill_between(grid, f_true, f_model, step='post', color=colour, alpha=0.18, linewidth=0)
        ax.step(grid, f_true, where='post', color=INK, linewidth=2, label='True hidden values')
        ax.step(grid, f_model, where='post', color=colour, linewidth=2, label=label)
        ax.set_title(label)
        ax.text(0.97, 0.05, f'Shaded area = W₁ = {area:.3f}', transform=ax.transAxes, ha='right', va='bottom',
                color=INK_2, fontsize=13)
        if view_quantiles is not None:
            low, high = np.quantile(true_values, view_quantiles)
            pad = 0.03 * (high - low)
            ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Value (standardised)')
        style_axes(ax, 'y')
        rows.append({'dataset': dataset, 'column': column, 'method': name, 'hidden_cells': len(values),
                     'shaded_area': area})
    axes[0][0].set_ylabel('Share of values at or below')
    handles, labels = axes[0][0].get_legend_handles_labels()
    clipped = '' if view_quantiles is None else (f' x-axis shows the {view_quantiles[0]:.1%}–{view_quantiles[1]:.1%} '
                                                  'range of true values; areas use all values.')
    finish(figure, title or f'What the distribution score measures: {dataset_label(dataset) + ", " if dataset else ""}'
                            f'{column}',
           note if note is not None else source_note([dataset] if dataset else [],
                                                     'Shaded area between the curves = Wasserstein-1 distance '
                                                     '(Vallender, 1974).' + clipped),
           legend=(handles[:1], labels[:1]))
    return Plot(figure, pd.DataFrame(rows))


def _ecdfs(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.union1d(a, b)
    return grid, np.searchsorted(a, grid, side='right') / len(a), np.searchsorted(b, grid, side='right') / len(b)


def _matrix(value):
    if isinstance(value, pd.DataFrame):
        return value
    frame = pd.read_csv(Path(value), dtype={'row_id': str}, float_precision='round_trip')
    if frame.columns[0] != 'row_id':
        raise ValueError(f'{value}: the first column must be row_id')
    return frame.set_index('row_id')


def _same_layout(truth, other, name):
    if not other.index.equals(truth.index) or not other.columns.equals(truth.columns):
        raise ValueError(f'{name}: rows and columns must match the truth matrix exactly, in the same order')
