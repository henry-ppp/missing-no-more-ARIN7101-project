"""The shared look: one fixed colour and shape per model, readable labels, slide-sized figures and saving.

Colours come from a palette whose order was checked for colour-blind readers (OKLab distance between neighbours,
see DESIGN.md). A model keeps its colour in every figure. A model that is not listed below gets the next unused
palette colour, keeps it for the rest of the session and triggers a warning asking for it to be registered, so
its colour stays the same next time. Colours are never generated, so there are at most eight models per figure.
"""
from __future__ import annotations

import functools
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---- colours ----

SURFACE = '#fcfcfb'      # figure background
INK = '#0b0b0b'          # main text and the true values
INK_2 = '#52514e'        # secondary text
MUTED = '#898781'        # notes, mean imputation (the 0 line), de-emphasised series
GRID = '#e1e0d9'         # gridlines
AXIS = '#c3c2b7'         # axis lines, reference lines

# Validated categorical order. Neighbouring slots stay distinguishable for colour-blind readers.
PALETTE = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')

BASELINES = ('mean', 'true')   # reference tables in the reports, never plotted as methods
FAMILIES = ('traditional', 'deep learning', 'other')
# Each model also has its own marker, so a scatter stays readable where colours alone could be confused.
# Triangles are deep learning; circles, squares, diamonds and pentagons are traditional.
SPARE_MARKERS = ('X', 'P', 'h', '*')


@dataclass(frozen=True)
class Model:
    label: str
    family: str
    colour: str
    marker: str = 'o'


def normalise(name: str) -> str:
    """'MICE-RF', 'mice_rf' and 'micerf' all become 'micerf'."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


_DEFAULT_MODELS = {
    'missforest': Model('MissForest', 'traditional', PALETTE[0], 'o'),
    'micerf': Model('MICE-RF', 'traditional', PALETTE[1], 's'),
    'knn': Model('kNN', 'traditional', PALETTE[2], 'D'),
    'iterative': Model('Iterative ridge', 'traditional', PALETTE[3], 'p'),
    'daev2': Model('DAE v2', 'deep learning', PALETTE[4], '^'),
    'vae': Model('VAE', 'deep learning', PALETTE[5], 'v'),
    'gain': Model('GAIN', 'deep learning', PALETTE[6], '<'),
}
_ALIASES = {'iterativeridge': 'iterative'}


@dataclass
class _Registry:
    models: dict = field(default_factory=lambda: dict(_DEFAULT_MODELS))
    warned: set = field(default_factory=set)


_registry = _Registry()


def register_model(name: str, label: str | None = None, family: str = 'other', colour: str | None = None,
                   marker: str | None = None) -> Model:
    """Add a model, or change one. The colour must be a palette colour. A model already listed keeps its colour
    and shape unless new ones are given; a new model gets the next unused ones."""
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {FAMILIES}')
    existing = _registry.models.get(_key(name))
    if colour is None:
        colour = existing.colour if existing else _next_free_colour(name)
    elif colour.lower() not in PALETTE:
        raise ValueError(f'Use one of the palette colours {PALETTE}, so every chart stays readable')
    marker = marker or (existing.marker if existing else _next_free_marker())
    model = Model(label or (existing.label if existing else str(name)), family, colour.lower(), marker)
    _registry.models[_key(name)] = model
    return model


def reset_models() -> None:
    """Back to the default list (mainly for tests)."""
    _registry.models = dict(_DEFAULT_MODELS)
    _registry.warned = set()


def model_style(name: str) -> Model:
    """The fixed label, family and colour of a model."""
    key = _key(name)
    if key not in _registry.models:
        model = Model(str(name), 'other', _next_free_colour(name), _next_free_marker())
        _registry.models[key] = model
        if key not in _registry.warned:
            _registry.warned.add(key)
            warnings.warn(f'Model {name!r} has no registered colour; it gets {model.colour} for this session. Call '
                          f'register_model({name!r}, label=..., family=...) or add it to style.py so its colour '
                          'stays the same in every figure.', stacklevel=2)
    return _registry.models[key]


def ordered_methods(names) -> list[str]:
    """Methods in display order: family (traditional, deep learning, other), then palette slot, then name."""
    unique = [n for n in dict.fromkeys(names) if n not in BASELINES]
    styles = {n: model_style(n) for n in unique}
    return sorted(unique, key=lambda n: (FAMILIES.index(styles[n].family), PALETTE.index(styles[n].colour), n))


def _key(name: str) -> str:
    key = normalise(name)
    return _ALIASES.get(key, key)


def _next_free_colour(name) -> str:
    used = {m.colour for m in _registry.models.values()}
    free = [c for c in PALETTE if c not in used]
    if not free:
        raise ValueError(f'No unused palette colour left for {name!r}. Eight models is the readable maximum: plot '
                         'fewer, or register this model with the colour of one that is not in the figure.')
    return free[0]


def _next_free_marker() -> str:
    used = {m.marker for m in _registry.models.values()}
    return next((m for m in SPARE_MARKERS if m not in used), SPARE_MARKERS[0])


def tint(colour: str, amount: float) -> str:
    """The same hue mixed with the background (amount 0 = unchanged, 1 = background)."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    bg = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(c + (b - c) * amount):02x}' for c, b in zip(rgb, bg))


# ---- names shown on figures ----

DATASETS = {
    'credit_card': 'Credit card',
    'california_housing': 'California housing',
    'hmda_ri_2023': 'HMDA mortgages (RI)',
    'gas_turbine': 'Gas turbine',
    'superconductivity': 'Superconductivity',
}
DATASET_SOURCES = {
    'credit_card': 'UCI Default of Credit Card Clients (Yeh)',
    'california_housing': 'California Housing (Pace & Barry; Deshpande CSV export)',
    'hmda_ri_2023': 'HMDA loan-level data, Rhode Island 2023 (CFPB for the FFIEC)',
    'gas_turbine': 'UCI Gas Turbine CO and NOx Emission',
    'superconductivity': 'UCI Superconductivity Data (Hamidieh)',
}
DATASET_LICENSES = {'hmda_ri_2023': 'public domain'}   # CFPB legal notice; every other source: CC BY 4.0
METRICS = {
    'relative_improvement': 'RMSE improvement over mean',
    'mae_improvement': 'MAE improvement over mean',
    'wasserstein_improvement': 'Wasserstein improvement over mean',
    'gap_closed': 'Downstream gap closed',
    'downstream_score': 'Downstream score',
    'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²', 'wasserstein': 'Wasserstein distance',
}
# True: larger is better. Ranking and "best" use this; unknown metrics must be given explicitly.
HIGHER_IS_BETTER = {
    'relative_improvement': True, 'mae_improvement': True, 'wasserstein_improvement': True, 'gap_closed': True,
    'downstream_score': True, 'r2': True, 'rmse': False, 'mae': False, 'wasserstein': False,
}


def dataset_label(name: str) -> str:
    return DATASETS.get(name, str(name).replace('_', ' ').capitalize())


def metric_label(name: str) -> str:
    return METRICS.get(name, str(name).replace('_', ' ').capitalize())


def higher_is_better(metric: str, override: bool | None = None) -> bool:
    if override is not None:
        return override
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f'Say whether larger {metric!r} is better (higher_is_better=True or False)')
    return HIGHER_IS_BETTER[metric]


def ordered_datasets(names) -> list[str]:
    known = list(DATASETS)
    unique = list(dict.fromkeys(names))
    return sorted(unique, key=lambda n: (known.index(n) if n in known else len(known), str(n)))


def rate_label(rate: float) -> str:
    return f'{rate * 100:g}% missing'


def source_note(datasets=(), extra: str | None = None) -> str:
    """'Data: ... (CC BY 4.0).' for the datasets shown, grouped by license, plus a sentence on outside methods or
    statistics."""
    groups = {}
    for d in ordered_datasets(datasets):
        groups.setdefault(DATASET_LICENSES.get(d, 'CC BY 4.0'), []).append(DATASET_SOURCES.get(d, dataset_label(d)))
    sources = [f"{'; '.join(names)} ({license})" for license, names in groups.items()]
    parts = [f"Data: {'; '.join(sources)}." if sources else '']
    if extra:
        parts.append(extra)
    return ' '.join(p for p in parts if p)


# ---- figures ----

SLIDE = (13.33, 7.5)   # inches: a 16:9 slide at its real size, so 14 pt text is 14 pt on the slide

RC = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14, 'xtick.labelsize': 13, 'ytick.labelsize': 14,
    'legend.fontsize': 13, 'figure.titlesize': 18,
    'text.color': INK, 'axes.labelcolor': INK_2, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.titlelocation': 'left', 'axes.titleweight': 'regular', 'axes.titlecolor': INK,
    'legend.frameon': False, 'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
}


def in_style(function):
    """Run a figure function inside the shared style (matplotlib reads it as each element is drawn)."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC):
            return function(*args, **kwargs)
    return wrapper


def new_figure(nrows: int = 1, ncols: int = 1, size=SLIDE, **kwargs):
    """A slide-sized figure. Returns (figure, 2-D array of axes). Call it inside an in_style function."""
    figure, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False, layout='constrained', **kwargs)
    return figure, axes


def finish(figure, title: str | None = None, note: str | None = None, legend=None, legend_columns: int | None = None,
           note_size: float = 11) -> None:
    """Lay out the frame around the panels: title at the top left, the source note along the bottom, and the legend
    (handles, labels) just above the note at the right. The panels get the space that is left, so nothing overlaps.
    """
    width, height = figure.get_size_inches()
    bottom = 0.0
    if note:
        lines = textwrap.wrap(note, int(width * 72 / (note_size * 0.5)))
        figure.text(0.01, 0.012, '\n'.join(lines), ha='left', va='bottom', fontsize=note_size, color=MUTED,
                    linespacing=1.4)
        bottom = 0.012 + (len(lines) * note_size * 1.4 + 6) / 72 / height
    if legend and legend[0]:
        handles, labels = legend
        artist = figure.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.99, bottom),
                               ncols=legend_columns or len(labels), frameon=False, borderaxespad=0)
        extent = artist.get_window_extent(figure.canvas.get_renderer())
        bottom += extent.height / figure.dpi / height + 0.012
    if title:
        figure.suptitle(title, x=0.01, ha='left', color=INK, fontsize=RC['figure.titlesize'])
    engine = figure.get_layout_engine()
    if engine is not None and bottom:
        engine.set(rect=(0, bottom, 1, 1 - bottom))


def style_axes(ax, grid_axis: str | None = 'x') -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, linestyle='-')
        ax.set_axisbelow(True)


@dataclass
class Plot:
    """A finished figure and the table of exactly what it shows (for an appendix table or a check)."""
    figure: object
    data: pd.DataFrame

    def save(self, path, formats=('pdf', 'svg', 'png'), dpi: int = 300, table: bool = True) -> list[Path]:
        """Save as path.pdf, path.svg and path.png (vector files stay sharp on slides), plus path.csv."""
        path = Path(path)
        stem = path.with_suffix('') if path.suffix.lstrip('.') in {*formats, 'csv'} else path
        stem.parent.mkdir(parents=True, exist_ok=True)
        saved = []
        with plt.rc_context(RC):
            for fmt in formats:
                target = stem.with_name(f'{stem.name}.{fmt}')
                self.figure.savefig(target, dpi=dpi, facecolor=SURFACE)
                saved.append(target)
        if table:
            target = stem.with_name(f'{stem.name}.csv')
            self.data.to_csv(target, index=False)
            saved.append(target)
        return saved

    def close(self) -> None:
        plt.close(self.figure)
