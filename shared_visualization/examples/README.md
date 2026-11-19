# Example figures

How to remake the six representative figures from the shared results. The PNGs in this folder were made with these
commands (full data, 30% missing, seed 0). Run every command from the repository root, in
the project environment (`conda activate project`, or `pip install -r requirements.txt`). Each figure is saved as PDF,
SVG and PNG, plus a CSV of exactly what it shows.

Output goes to `figures/`. That folder isn't in `.gitignore`, so don't add it to a commit by accident.

| Figure | Step | Output |
| --- | --- | --- |
| Scoreboard | 2 | `figures/comparative_plots/scoreboard_full_seed_0_rate_0.3.*` |
| Accuracy vs distribution | 2 | `figures/comparative_plots/accuracy_vs_distribution_full_seed_0.*` |
| Overall ranking | 2 | `figures/comparative_plots/ranking_full_seed_0.*` |
| Distribution picture | 3 | `figures/comparative_plots/distribution_credit_card_full_BILL_AMT1.*` |
| DAE v2 training curves | 3 | `figures/dae_v2/training_dae_v2_full.*` |
| MissForest training curves | 3 | `figures/MissForest/training_MissForest_full.*` |

## 1. Score all models of each dataset together

The figures read evaluation reports. Scoring every model of a dataset in one call pairs them in the bootstrap, so the
scoreboard can mark which differences are significant. This takes a few minutes.

```bash
for ds in credit_card california_housing superconductivity gas_turbine; do
  out=data/output/$ds/full/rate_0.3_seed_0
  preds=(MissForest=$out/MissForest/initial/predictions.csv MICE-RF=$out/MICE-RF/initial/predictions.csv)
  [ -f $out/dae_v2/initial/predictions.csv ] && preds+=(dae_v2=$out/dae_v2/initial/predictions.csv)
  python -m shared_evaluation.evaluate --dataset $ds --mode full --predictions "${preds[@]}" \
      --downstream --output-dir figures/comparative_plots/reports
done
```

HMDA is optional and has no DAE v2 run yet. Add `hmda_ri_2023` to the loop only when that dataset is needed; it is then scored with the two random forests only.

## 2. Comparison figures

```bash
python -m shared_visualization figures/comparative_plots/reports --out figures/comparative_plots --mode full
```

This also writes `scores.csv`, `pairs.csv` and `rank_table_full_seed_0.csv`. The missing-rate figure is skipped until a
second missing rate is prepared.

## 3. Distribution picture and training curves

These need a choice (a column, a model), so they are made in Python:

```bash
python - <<'EOF'
import matplotlib; matplotlib.use('Agg')
import pandas as pd
from data_preprocessing.dataset_config import DATASETS
from shared_visualization import hidden_values, distribution_figure, find_histories, training_curves

cfg, out = 'data/processed/credit_card/full/rate_0.3_seed_0', 'data/output/credit_card/full/rate_0.3_seed_0'
truth, imputed = hidden_values(f'{cfg}/X_cleaned_standardized.csv', f'{cfg}/mask_eval.csv',
                               {'mean': f'{cfg}/X_mean_baseline.csv',
                                'MissForest': f'{out}/MissForest/initial/predictions.csv',
                                'MICE-RF': f'{out}/MICE-RF/initial/predictions.csv',
                                'dae_v2': f'{out}/dae_v2/initial/predictions.csv'}, column='BILL_AMT1')
distribution_figure(truth, imputed, 'BILL_AMT1', dataset='credit_card').save(
    'figures/comparative_plots/distribution_credit_card_full_BILL_AMT1')

dae = {k: v for k, v in find_histories('results', 'dae_v2', mode='full').items() if v.parts[-5] in DATASETS}
training_curves(dae, 'dae_v2').save('figures/dae_v2/training_dae_v2_full')

rf = find_histories('local_models/random_forest', 'MissForest', mode='full')
training_curves({k: pd.read_csv(p)[['round', 'change']] for k, p in rf.items()}, 'MissForest', logy=True).save(
    'figures/MissForest/training_MissForest_full')
EOF
```

- `v.parts[-5] in DATASETS` keeps curves for the four required datasets. HMDA is optional, so its curves are left out unless that filter is widened.
- Random-forest training histories are in `local_models/`, which isn't pushed, so the MissForest figure can only be
  made on the machine that ran the random forests. DAE v2 histories are in `results/` and work anywhere.
- `[['round', 'change']]` drops the timing column, so only the change per round is drawn.
