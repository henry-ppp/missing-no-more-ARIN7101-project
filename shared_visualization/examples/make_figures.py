"""Make every shared figure from the saved evaluation reports: full data, seed 0, the four required datasets.

Run from the repository root, after the paired evaluations in README.md (step 1):

    python -m shared_visualization.examples.make_figures

Comparison figures and their tables go to figures/comparative_plots/, training curves to figures/<model>/, and a
PNG of each figure type is copied into this folder. Every number comes from the reports; nothing is re-scored.
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import pandas as pd  # noqa: E402

from shared_visualization import (accuracy_vs_distribution, critical_difference, distribution_figure,  # noqa: E402
                                  hidden_values, load_results, missing_rate_curves, rank_table, register_model,
                                  scoreboard, training_curves, write_tables)
from shared_visualization.style import PALETTE, dataset_label, source_note  # noqa: E402

REQUIRED = ('credit_card', 'california_housing', 'gas_turbine', 'superconductivity')   # the panel order style.py uses
OUT = Path('figures/comparative_plots')
REPORTS = OUT / 'reports'
EXAMPLES = Path(__file__).resolve().parent
# Only these four models have runs at 10% missing; the missing-rate figure is limited to them.
BOTH_RATES = ['MissForest', 'MICE-RF', 'SoftImpute', 'Sinkhorn']

# style.py lists seven models and keeps the eighth colour spare. The shared figures show twelve, so the other five
# are registered here (names as in the reports). Within a family every colour is unique; a colour used in both
# families always comes with a different shape (pointed shapes are deep learning), and every figure names the
# methods on the axis, beside the mark or in a colour-and-shape legend, so colour is never the only cue.
MODELS = {
    'SoftImpute': dict(label='SoftImpute', family='traditional', colour=PALETTE[7], marker='h'),
    'Sinkhorn': dict(label='Sinkhorn OT', family='traditional', colour=PALETTE[6], marker='8'),
    'dl_autoencoder': dict(label='Autoencoder', family='deep learning', colour=PALETTE[0], marker='>'),
    'dl_mida': dict(label='MIDA', family='deep learning', colour=PALETTE[1], marker='P'),
    'dl_miwae': dict(label='MIWAE', family='deep learning', colour=PALETTE[7], marker='X'),
    'dl_gain': dict(label='GAIN', family='deep learning', colour=PALETTE[6], marker='<'),   # as 'gain' in style.py
    'dl_vae': dict(label='VAE', family='deep learning', colour=PALETTE[5], marker='v'),     # as 'vae' in style.py
}
SHAPES = 'Triangles and crosses: deep learning; round and square shapes: traditional.'


def save(plot, path, examples=False):
    saved = plot.save(path)
    if examples:
        shutil.copy(next(p for p in saved if p.suffix == '.png'), EXAMPLES)
    plot.close()
    print(f'Saved {path}.*')


def comparison_figures():
    results = load_results(REPORTS).select(mode='full', seed=0, dataset=list(REQUIRED))
    write_tables(results, OUT)
    main = results.select(rate=0.3)

    # The row label under each title names the score, so the titles say what it is for.
    scores = {'relative_improvement': ('rmse', 'Accuracy: how much better than mean imputation?'),
              'wasserstein_improvement': ('wasserstein', 'Distribution: how much better than mean imputation?'),
              'gap_closed': ('gap_closed', 'Usefulness for prediction: how much of the gap to the true data is closed?')}
    for metric, (tag, title) in scores.items():
        save(scoreboard(main, (metric,), rate=0.3, title=f'{title} Full data, 30% missing'),
             OUT / f'scoreboard_full_seed_0_rate_0.3_{tag}', examples=tag == 'rmse')

    note = source_note(REQUIRED, f'{SHAPES} Wasserstein ≤ MAE in every column (Villani, 2009), so y ≥ x up to the '
                                 'small pooled-vs-column-averaged MAE difference. Kendall (1938) τ across methods.')
    save(accuracy_vs_distribution(main, note=note, title='Does accuracy predict how well the distribution is kept? '
                                                         'Full data, 30% missing'),
         OUT / 'accuracy_vs_distribution_full_seed_0', examples=True)

    save(critical_difference(main, title='Overall ranking across the four datasets: full data, 30% missing'),
         OUT / 'ranking_full_seed_0', examples=True)
    rank_table(main, metrics=('relative_improvement', 'wasserstein_improvement', 'gap_closed')).to_csv(
        OUT / 'rank_table_full_seed_0.csv', index=False)

    note = source_note(REQUIRED, 'Band: bootstrap interval from the evaluation report (this mask only). Only these '
                                 'four methods have runs at 10% missing; the others were run at 30% only.')
    # The two random forests nearly coincide, so end-of-line labels would overlap; the legend names all four.
    save(missing_rate_curves(results.select(method=BOTH_RATES), methods=BOTH_RATES, highlight=BOTH_RATES, note=note,
                             direct_labels=False),
         OUT / 'missing_rate_full_seed_0', examples=True)


def distribution_picture():
    cfg, out = 'data/processed/credit_card/full/rate_0.3_seed_0', 'data/output/credit_card/full/rate_0.3_seed_0'
    truth, imputed = hidden_values(f'{cfg}/X_cleaned_standardized.csv', f'{cfg}/mask_eval.csv',
                                   {'mean': f'{cfg}/X_mean_baseline.csv',
                                    'MissForest': f'{out}/MissForest/initial/predictions.csv',
                                    'MICE-RF': f'{out}/MICE-RF/initial/predictions.csv',
                                    'dae_v2': f'{out}/dae_v2/initial/predictions.csv'}, column='BILL_AMT1')
    save(distribution_figure(truth, imputed, 'BILL_AMT1', dataset='credit_card'),
         OUT / 'distribution_credit_card_full_BILL_AMT1', examples=True)


def histories(root, model, columns=None):
    """{dataset label: table} for the 30% full runs of the required datasets, in the usual dataset order."""
    found = {}
    for dataset in REQUIRED:
        path = Path(root, model, 'training', dataset, 'full', 'rate_0.3_seed_0', 'initial', 'training_history.csv')
        if path.exists():
            table = pd.read_csv(path)
            found[dataset_label(dataset)] = table[columns] if columns else table
    if not found:
        raise FileNotFoundError(f'No 30% full training histories for {model} under {root}')
    return found


def training_figures():
    # Random-forest and Sinkhorn histories are in local_models/ (not pushed), so these two need the machine that
    # ran them. DAE v2 histories are in results/ and work anywhere.
    for model in ('MissForest', 'MICE-RF'):
        save(training_curves(histories('local_models/random_forest', model, ['round', 'change']), model, logy=True,
                             note='Change between rounds (the stopping rule): the run stops when it grows again and '
                                  'keeps the round before, marked.'),
             f'figures/{model}/training_{model}_full', examples=model == 'MissForest')
    save(training_curves(histories('results', 'dae_v2'), 'dae_v2'), 'figures/dae_v2/training_dae_v2_full',
         examples=True)
    # No Sinkhorn or SoftImpute curves. Sinkhorn's loss compares two random batches, so on credit card and
    # California it is mostly sampling noise; its budget was chosen by hide-and-check instead
    # (models/softimpute_sinkhorn/README.md). SoftImpute's final fit takes 1 to 8 steps along its warm-started path.


def main():
    for name, style in MODELS.items():
        register_model(name, **style)
    comparison_figures()
    distribution_picture()
    training_figures()


if __name__ == '__main__':
    main()
