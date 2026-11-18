"""Optional appendix figure: training curves for any model that writes a training history.

A history is a CSV with one step column (epoch, round, iteration or step) and one or more numeric columns, such as
DAE v2's training_history.csv (epoch, train_masked_mse, validation_masked_mse). MissForest or iterative ridge can
write their change between rounds the same way, so one figure works for every model. Columns whose name contains
"val" are drawn in the model's colour; the others in lighter shades of it. The best step (lowest validation value,
or lowest value of the only column) is marked.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import (INK_2, MUTED, SURFACE, Plot, dataset_label, finish, in_style, model_style, new_figure, tint,
                    style_axes)

STEP_COLUMNS = ('epoch', 'round', 'iteration', 'step')


def find_histories(root, model: str, mode: str | None = None, run_id: str | None = None,
                   filename: str = 'training_history.csv') -> dict[str, Path]:
    """Histories in the shared results layout: <root>/<model>/training/<dataset>/<mode>/<configuration>/<run_id>/.

    Returns {panel title: path}. The title is the dataset, plus mode, configuration or run ID where they vary.
    """
    found = []
    for path in sorted(Path(root, model, 'training').glob(f'*/*/*/*/{filename}')):
        dataset, found_mode, configuration, found_run = path.parts[-5:-1]
        if (mode is None or found_mode == mode) and (run_id is None or found_run == run_id):
            found.append((dataset, found_mode, configuration, found_run, path))
    if not found:
        raise FileNotFoundError(f'No {filename} under {Path(root, model, "training")} for mode={mode}, run_id={run_id}')
    varies = [len({f[i] for f in found}) > 1 for i in range(1, 4)]
    titles = {}
    for dataset, found_mode, configuration, found_run, path in found:
        extra = [v for v, show in zip((found_mode, configuration, found_run), varies) if show]
        titles[', '.join([dataset_label(dataset)] + extra)] = path
    return titles


@in_style
def training_curves(histories: dict, model: str, step: str | None = None, logy: bool = False,
                    title: str | None = None, note: str | None = None) -> Plot:
    """One panel per history ({title: CSV path or DataFrame}), every numeric column as a line against the step."""
    if not histories:
        raise ValueError('Give at least one training history')
    colour = model_style(model).colour
    ncols = min(len(histories), 4)
    nrows = -(-len(histories) // ncols)
    figure, axes = new_figure(nrows, ncols, sharex=False)
    table = []
    for index, (name, history) in enumerate(histories.items()):
        ax = axes[index // ncols][index % ncols]
        frame = history if isinstance(history, pd.DataFrame) else pd.read_csv(history)
        x = step or next((c for c in STEP_COLUMNS if c in frame.columns), frame.columns[0])
        series = [c for c in frame.columns if c != x and pd.api.types.is_numeric_dtype(frame[c])]
        if not series:
            raise ValueError(f'{name}: no numeric columns besides {x!r}')
        validation = [c for c in series if 'val' in c.lower()]
        others = [c for c in series if c not in validation]
        shades = {c: tint(colour, 0.55 + 0.15 * i) for i, c in enumerate(others)}
        shades.update({c: colour for c in validation})
        for column in others + validation:
            ax.plot(frame[x], frame[column], color=shades[column], linewidth=2, label=_pretty(column))
        target = (validation or series)[0]
        best = frame.loc[frame[target].idxmin()]
        ax.axvline(best[x], color=MUTED, linewidth=1, zorder=1)
        ax.plot([best[x]], [best[target]], marker='o', markersize=8, color=colour, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=4)
        right_half = best[x] > (frame[x].min() + frame[x].max()) / 2
        ax.annotate(f'best {x} {best[x]:g}', (best[x], best[target]), xytext=(-8 if right_half else 8, 10),
                    textcoords='offset points', ha='right' if right_half else 'left', fontsize=12, color=INK_2)
        if logy:
            ax.set_yscale('log')
        ax.set_title(name)
        ax.set_xlabel(x.capitalize())
        style_axes(ax, 'y')
        table.append({'panel': name, 'model': model, 'step_column': x, 'best_step': best[x], 'best_column': target,
                      'best_value': best[target], 'steps': len(frame)})
    for index in range(len(histories), nrows * ncols):
        axes[index // ncols][index % ncols].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    finish(figure, title or f'Training curves: {model_style(model).label}',
           note if note is not None else 'Marked: the step with the lowest validation value (lowest value if there '
                                         'is no validation column).',
           legend=(handles, labels))
    return Plot(figure, pd.DataFrame(table))


def _pretty(column: str) -> str:
    return column.replace('_', ' ').capitalize()
