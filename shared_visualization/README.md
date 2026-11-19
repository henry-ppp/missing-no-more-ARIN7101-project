# Shared figures

Draws the same slide-ready figures for every imputation model from saved evaluation reports. Every model has one fixed colour and shape in every figure. Why each figure exists and how it was designed: [DESIGN.md](DESIGN.md).

The package never imports `shared_evaluation` or any model. It only reads files: evaluation reports, or two plain CSV tables (below). `results.py` is the only module that knows report formats. If `shared_evaluation` changes its JSON, only `results.py` changes, and `tests/test_visualization.py` will fail until it does.

## Quick start

From the repository root, after `pip install -r requirements.txt`:

1. Write each model's `predictions.csv` as `data/README.md` describes.
2. Score **all models of one setting together**, so the bootstrap pairs every model with every other:

   ```bash
   python -m shared_evaluation.evaluate_v2 --dataset credit_card --mode quick --rate 0.3 --seed 0 --downstream \
       --predictions missforest=data/output/credit_card/quick/rate_0.3_seed_0/missforest/run1/predictions.csv \
                     dae_v2=data/output/credit_card/quick/rate_0.3_seed_0/dae_v2/initial/predictions.csv
   ```

   Use `NAME=PATH`. With the current results layout, a bare path would be named after its run-ID folder.
3. Make every figure the reports support:

   ```bash
   python -m shared_visualization data/output --out figures
   ```

   Folders are searched for `evaluation*.json`. Each figure is saved as PDF and SVG (sharp on slides) and as a 300-dpi PNG, with a CSV of exactly what it shows. A figure the reports can't support is skipped, with the reason printed. Add `--mode quick`, `--seed 0` or `--rename old=new` as needed.

Commands for the six representative figures on the current results: [examples/README.md](examples/README.md).

## In Python

```python
from shared_visualization import load_results, scoreboard, accuracy_vs_distribution

results = load_results('data/output', 'results', rename={'dae_v2_initial': 'dae_v2'})
plot = scoreboard(results, mode='quick', rate=0.3)
plot.save('figures/scoreboard_quick')     # .pdf .svg .png and .csv
plot.data                                 # the numbers shown, as a table
```

Every figure function returns a `Plot` with `.figure` (matplotlib) and `.data`. Each takes `title=` and `note=` to replace the defaults. When the reports cover several modes, seeds or rates, choose with `mode=`, `seed=` and `rate=`. A figure never mixes settings silently.

## The figures

| Function | Question it answers | Needs |
| --- | --- | --- |
| `distribution_figure` | What does the Wasserstein distance measure? The shaded area between the true and imputed cumulative distributions of one column *is* that distance. | The truth, the mask and prediction files: use `hidden_values(...)` first |
| `scoreboard` | How much better than mean imputation is each method, per dataset and score, and which gaps are significant? | Version 2 reports (intervals; paired p-values for the filled and hollow dots) |
| `accuracy_vs_distribution` | Does an accurate method also keep the distribution? x = MAE improvement, y = Wasserstein improvement; y = x is the floor accuracy guarantees. | MAE and Wasserstein for each model and for `mean` (version 2 reports have both) |
| `missing_rate_curves` | Does the gap between methods change as more is missing? | Reports at two or more missing rates |
| `critical_difference` and `rank_table` | Which methods are best overall, and which are tied? | Two or more methods in the same settings. About 10 or more dataset × rate settings before any pair can be significant; with fewer, average ranks only, and the figure says so |
| `training_curves` and `find_histories` | Optional: did training converge? | `training_history.csv` files (below) |

Figure 1 example, from the repository root:

```python
from shared_visualization import hidden_values, distribution_figure
cfg = 'data/processed/credit_card/quick/rate_0.3_seed_0'
truth, imputed = hidden_values(f'{cfg}/X_cleaned_standardized.csv', f'{cfg}/mask_eval.csv',
                               {'mean': f'{cfg}/X_mean_baseline.csv',
                                'dae_v2': 'data/output/credit_card/quick/rate_0.3_seed_0/dae_v2/initial/predictions.csv'},
                               column='BILL_AMT1')
distribution_figure(truth, imputed, 'BILL_AMT1', dataset='credit_card').save('figures/distribution_credit')
```

## Input format (for any evaluator)

`load_results` reads `shared_evaluation` version 1 and version 2 JSON reports. Anything else can feed the figures by writing two CSV files with these columns; `write_tables` writes them from loaded reports:

- `scores.csv`: `dataset, mode, rate, seed, method, metric, estimate`, plus optional `bias_corrected, ci_low, ci_high`. One row per setting, method and score. Include the `mean` baseline's raw scores, so improvements can be derived.
- `pairs.csv` (optional): `dataset, mode, rate, seed, method_a, method_b, metric, difference, p_value`. One row per paired comparison, with `difference = method_a - method_b`. The scoreboard uses it to mark significance.

Score names the figures know: `relative_improvement` (RMSE), `wasserstein_improvement`, `gap_closed`, `mae_improvement` (derived from `mae` when missing), `downstream_score`, `rmse`, `mae`, `r2`, `wasserstein`. Wasserstein scores are plotted bias-corrected and the others as plain estimates, matching the table `evaluate_v2` prints.

## Models, colours and shapes

| Model name in reports | Label | Family | Colour | Shape |
| --- | --- | --- | --- | --- |
| `missforest` | MissForest | traditional | blue | circle |
| `mice_rf` | MICE-RF | traditional | orange | square |
| `knn` | kNN | traditional | aqua | diamond |
| `iterative` / `iterative_ridge` | Iterative ridge | traditional | yellow | pentagon |
| `dae_v2` | DAE v2 | deep learning | pink | triangle up |
| `vae` | VAE | deep learning | green | triangle down |
| `gain` | GAIN | deep learning | violet | triangle left |

Names are matched ignoring case, `-` and `_`. Mean imputation is always the grey 0 line, never a coloured series.

A new model gets the one spare colour (red) and a warning. To keep its look fixed, add it to `_DEFAULT_MODELS` in `style.py`, or call `register_model('name', label='...', family='deep learning')` before plotting. Colours come from a palette checked for colour-blind readers, and are never generated, so at most eight models fit in one figure.

## Training histories

`find_histories('results', 'dae_v2', mode='quick')` finds files at `results/<model>/training/<dataset>/<mode>/<configuration>/<run_id>/training_history.csv`. Each file needs a step column (`epoch`, `round`, `iteration` or `step`) and one or more numeric columns. Columns whose name contains `val` are drawn in the model's colour, the others lighter. A random-forest model can write its change between rounds the same way:

```text
round,change
1,0.41
2,0.12
```

## Tests

```bash
python -m unittest tests.test_visualization -v
```

The tests are offline, using synthetic tables. They also check that importing this package never loads `shared_evaluation`, a model or PyTorch.

## Sources

Figures cite their statistical sources in their notes: Vallender (1974) and Villani (2009) for the Wasserstein distance; Schenker & Gentleman (2001) on overlapping intervals; Holm (1979); Wilcoxon (1945); Friedman (1937); Kendall (1938); Demšar (2006) and Benavoli, Corani & Mangili (2016) for the ranking diagram. Full references are in [DESIGN.md](DESIGN.md). The code uses matplotlib, NumPy, pandas and SciPy.
