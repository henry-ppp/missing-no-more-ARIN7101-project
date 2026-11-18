"""Shared figures for every imputation model, drawn from saved evaluation reports.

    from shared_visualization import load_results, scoreboard
    results = load_results('data/output', 'results')          # every evaluation*.json below these folders
    scoreboard(results, mode='quick', rate=0.3).save('figures/scoreboard_quick')

The package never imports shared_evaluation or any model: it reads report files (results.py is the only place that
knows their format) or tidy CSV tables in the documented columns. See README.md.
"""
from .distribution import distribution_figure, ecdf_area, hidden_values
from .missing_rate import missing_rate_curves
from .ranking import critical_difference, rank_table
from .results import Results, derive_improvement, load_results, one_setting, read_report, read_tables, write_tables
from .scoreboard import HEADLINE, scoreboard
from .style import PALETTE, Plot, model_style, register_model, source_note
from .tradeoff import accuracy_vs_distribution
from .training import find_histories, training_curves

__all__ = [
    'HEADLINE', 'PALETTE', 'Plot', 'Results', 'accuracy_vs_distribution', 'critical_difference', 'derive_improvement',
    'distribution_figure', 'ecdf_area', 'find_histories', 'hidden_values', 'load_results', 'missing_rate_curves',
    'model_style', 'one_setting', 'rank_table', 'read_report', 'read_tables', 'register_model', 'scoreboard',
    'source_note', 'training_curves', 'write_tables',
]
