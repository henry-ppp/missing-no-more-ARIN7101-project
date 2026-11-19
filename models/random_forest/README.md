# Random-forest imputation

MissForest and MICE with random forests (MICE-RF) on the shared benchmark's **full** tables, with **one shared seed (0)** and no repeats. The uncertainty comes from `shared_evaluation` version 2's bootstrap, not from reruns.

## Run it

From the repository root:

```bash
python -m models.random_forest.run_experiment --plan      # what is done, what is left, rough time
caffeinate -is python -m models.random_forest.run_experiment
```

The second command runs both models on all five datasets in `full` mode, for every missing rate prepared with seed 0 (currently 0.3), using 8 worker processes with one thread each. `caffeinate` keeps a Mac awake; plug the laptop in and keep the lid open. If full Superconductivity is too slow, the meeting note says to use its Quick configuration.

The command can be stopped and restarted. Each tuning trial, final fit and evaluation is saved as soon as it finishes, and a rerun skips work saved with the same settings. Reusing a `--run-id` with different settings is refused before anything runs, so saved results are never mixed or overwritten. Pick a new run ID instead.

**Time.** `--plan` gives an upper bound: 5.3 CPU-hours, about 3.3 hours of wall time on an 8-core M2. It assumes every column is refitted in all 10 rounds; MissForest usually stops earlier. Wall time is set by superconductivity's MissForest (tuning, then a fit of up to 1.3 hours), not by the number of cores. The other datasets finish in minutes.

To tune the three narrow tables on all their rows rather than a sample (about 0.6 hours more), run them separately:

```bash
caffeinate -is python -m models.random_forest.run_experiment --datasets credit_card california_housing gas_turbine --tune-rows all
caffeinate -is python -m models.random_forest.run_experiment --datasets superconductivity
```

Useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--models`, `--datasets` | all | e.g. `--datasets credit_card gas_turbine` |
| `--rates` | every prepared rate | missing rates to run (must be prepared) |
| `--seed` | 0 | the one shared seed; also picks the `rate_<r>_seed_<s>` folders |
| `--run-id` | `initial` | names this run's folders |
| `--n-trees`, `--max-rounds` | 100, 10 | forest size and round limit |
| `--tune-rows` | 3000 | rows of the model input used for tuning, or `all` |
| `--tune-rate` | 0.3 | missing rate to tune on when no hyperparameters are saved |
| `--no-tune` | off | use each model's default hyperparameters |
| `--n-boot` | 1000 | bootstrap resamples in the evaluation |
| `--jobs` | number of cores | runs at the same time |

## Hyperparameters: tuned once, saved, reused

Each model has **two** hyperparameters: features tried per split (`max_features`) and minimum leaf size (`min_samples_leaf`). With only two, a small full grid is enough; random search pays off only with many (Bergstra & Bengio, 2012). Each hyperparameter takes our default and one published library default, giving 4 settings:

| Model | `max_features` | `min_samples_leaf` | Alternatives come from |
| --- | --- | --- | --- |
| MissForest | √p (default), p/3 | 1 (default), 5 | R `randomForest` regression defaults |
| MICE-RF | 50% (default), 100% | 2 (default), 1 | scikit-learn `ExtraTreesRegressor` defaults |

A setting that gives the same forest as another is tried only once. For example, on California (7 predictors) √7 and 7/3 both round to 2 features, so MissForest has 2 settings there.

In your earlier runs, tuning a 12-setting grid gained 0.36% (MissForest) and 0.02% (MICE-RF) on the full 30,000-row credit card table. So a large search isn't worth its cost on these tables.

**How tuning works (hide-and-check).**
- A further 10% of the visible cells of the model input are hidden. A 3,000-row sample is used unless `--tune-rows all`.
- Each setting fills them in, and the lowest RMSE wins; on a tie, the default.
- Tuning reads only the incomplete model input (`load_model_input`). The cells the benchmark scores are already hidden there, and the evaluation truth is never opened.

**Saved and reused.** Each model is tuned **once per dataset and mode**, at one missing rate (30% by default). The winner is saved to:

```text
results/random_forest/<Model>/hyperparameters/<dataset>_<mode>.json
```

The file holds the chosen hyperparameters, the rate and rows they were tuned on, and every candidate's check error. Every later run for that dataset and mode, at **any** missing rate, reads the file and skips tuning. So when the 10%, 20% and 50% data are prepared, `--rates 0.1 0.2 0.5` reuses the 30% tuning. A run that includes several rates and no saved file tunes on 30% only. To tune again, delete the file.

## What else it does

1. **Final fit.** The hyperparameters fill the full model input. Visible cells are kept unchanged, and the run checks that the input matches the hash in its `metadata.json`.
2. **Evaluation.** `shared_evaluation.evaluate` scores the predictions only after they are saved: RMSE, MAE, R², the Wasserstein distance and their improvements over mean imputation, with 95% bootstrap intervals, plus the downstream check.
3. **Paired comparison.** Version 2 scores both models of each setting together, so the MissForest-vs-MICE-RF difference gets a p-value.

**Models.**
- `MissForest` (Stekhoven & Bühlmann, 2012) is our own implementation. It starts from column means, then refits a random forest (Breiman, 2001) for each column, fewest gaps first, from the other columns. It stops when the change between rounds grows and keeps the round before. 100 trees, at most 10 rounds.
- `MICE-RF` is scikit-learn's `IterativeImputer` (chained equations, van Buuren & Groothuis-Oudshoorn, 2011) with extremely randomised trees (Geurts, Ernst & Wehenkel, 2006). It uses `skip_complete=True` and at most 40 nearest features per column.
  - `LeanIterativeImputer` drops each forest after use and gives identical output; plain `IterativeImputer` keeps every forest, about 20 GB on a 30,000-row table.
  - With trees, scikit-learn's stopping rule is rarely met, so MICE-RF usually runs all 10 rounds.

Both models record the MissForest change between rounds as a training curve, plus each column's feature importances.

**One seed.** Seed 0 is the benchmark's own seed: the one that chose the rows and the hidden cells. Every other random step gets its own stream derived from it and a fixed label (`seeds.py`): the tuning sample, the check cells, each forest in each round. So a rerun reproduces every number.

## Where results go

`<run>` is `<dataset>/<mode>/rate_<r>_seed_<s>/<run_id>`:

```text
data/output/<dataset>/<mode>/rate_<r>_seed_<s>/<Model>/<run_id>/
    predictions.csv, metadata.json             where data/README.md says predictions go, for team-wide evaluation
results/random_forest/<Model>/                  Model = MissForest or MICE-RF
    hyperparameters/<dataset>_<mode>.json       tuned once, reused for every missing rate
    experiment/<run>/experiment.json            hyperparameters used, rounds, time, headline scores with intervals
    evaluation/<run>/evaluation_v2_*.json       shared evaluation: intervals, Wasserstein, downstream
    evaluation/summary/benchmark_summary.csv    one row per run (runs before 19 Nov 2026 also have a v1 report)
results/random_forest/comparison/evaluation/<run>/evaluation_v2_*.json   both models, paired
local_models/random_forest/<Model>/             not pushed (.gitignore)
    training/<run>/training_history.csv         change between rounds (the training curve)
    training/<run>/importances.csv              row = imputed column, columns = importance of each predictor
    training/<run>/run.json, settings.json      exact settings, hashes, timings, versions
    tuning/<dataset>/<mode>/rate_<r>_seed_<s>/candidate_<k>.json   each tuning trial
```

**Fitted forests are not saved.** On the full tables one column's 100-tree forest pickles to 100–235 MB, so a single run's forests would take 1–20 GB. Imputation fits once and only needs the filled table, which is saved, so nothing has to be retrained for later evaluation.

The figures read these files directly:

```python
from shared_visualization import load_results, scoreboard, find_histories, training_curves
results = load_results('results/random_forest')            # per-model and paired reports
scoreboard(results, mode='full', rate=0.3).save('figures/rf_scoreboard')
training_curves(find_histories('local_models/random_forest', 'MissForest', mode='full'), 'MissForest').save('figures/rf_training')
```

## Use the imputers directly

```python
from models.random_forest import MissForestImputer, MiceRFImputer
imputer = MissForestImputer(seed=0, min_samples_leaf=5)   # or the saved params: MissForestImputer(seed=0, **params)
filled = imputer.fit_transform(X_missing)     # DataFrame with NaN gaps; same index and columns
imputer.info_.history                         # training curve; imputer.info_.importances for feature importances
```

The array functions `impute_missforest(X, seed, **params)` and `impute_mice_rf(X, seed, **params)` return `(filled, info)`.

## Tests

```bash
python -m unittest tests.test_random_forest -v
```

The tests cover:
- the seeds;
- that both imputers keep visible values, fill every gap, beat mean imputation and are reproducible;
- the MissForest stopping rule;
- that `LeanIterativeImputer` matches scikit-learn;
- the 2×2 grid;
- the whole pipeline on a tiny synthetic benchmark: outputs, evaluation, the paired comparison, saved hyperparameters reused at another rate, a two-rate run tuned once, resuming, and refusing a changed setting.

## Outside sources

- scikit-learn's `RandomForestRegressor`, `ExtraTreesRegressor` and `IterativeImputer` are used as they are.
- MissForest's algorithm follows Stekhoven & Bühlmann (2012); the code, tuning, runner and seeding are ours.
- References:
  - Breiman (2001), *Machine Learning* 45:5–32.
  - Stekhoven & Bühlmann (2012), *Bioinformatics* 28(1):112–118.
  - van Buuren & Groothuis-Oudshoorn (2011), *Journal of Statistical Software* 45(3).
  - Geurts, Ernst & Wehenkel (2006), *Machine Learning* 63:3–42.
  - Bergstra & Bengio (2012), *Journal of Machine Learning Research* 13:281–305.
