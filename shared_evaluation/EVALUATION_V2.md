# Evaluation v2: proposed changes

**Proposal:** score every model on three core criteria: accuracy, distribution and usefulness for prediction. Each one is measured against mean imputation and comes with a 95% confidence interval. The version 1 files stay untouched until we agree.

## Why change

**1. The four version 1 scores all measure the same thing: how close each guess is to its true value. So there is only one single criteria repeated four times**

| v1 score | How it relates to the others |
| --- | --- |
| RMSE | The basic error. |
| Relative improvement | 1 − RMSE / RMSE of mean imputation. Within one setting the mean's RMSE is fixed, so it ranks models exactly like RMSE. |
| MAE | The same errors, absolute instead of squared. It usually ranks models the same way. |
| R² (per column, averaged) | 1 − a column's squared error / that column's variance. Also a rescaled squared error. |

**2. Nothing measures the second half of our question,** whether the filled-in data keeps the original distribution. A model can have a low RMSE by guessing near the middle every time, which squeezes the spread. Mean imputation is the extreme case: every guess in a column is the same number.

**3. One run per setting gives no error bars,** so we can't tell a real difference from luck. Retraining every model many times is too slow.

## Proposed core criteria

All three use the same scale: **0 = mean imputation, higher is better.**

| Question | Score | Why it adds something |
| --- | --- | --- |
| Accuracy: is each guess close to its true value? | RMSE improvement over mean imputation (`relative_improvement`) | We already use it. It has no units, so different datasets can be compared. |
| Distribution: do the guesses as a group have the right spread and shape? | Wasserstein improvement over mean imputation (`wasserstein_improvement`) | It compares all the guesses in a column with all the true values, regardless of which row each one belongs to. It catches models that are accurate but too narrow. |
| Usefulness: can the filled-in table still predict the target? | Share of the gap closed (`gap_closed`), from mean imputation (0) to the true table (1) | It weights each column by how much it matters for a real task, while RMSE weights all columns equally. |

MAE, R² and raw RMSE are still computed and saved, for an appendix.

**The Wasserstein distance in one line:** for each column, sort the guesses and the true values, line them up, and take the average gap; then average over columns. The data is standardised, so the unit is standard deviations. HyperImpute (Jarrett et al., ICML 2022) scores imputation with RMSE plus Wasserstein distance.

**The prediction check:**
- Credit card default uses logistic regression, scored by AUROC.
- California house value and superconductor critical temperature use linear regression, scored by R².
- Gas turbine has no target, so it is skipped.
- It uses 5-fold cross-validation with the same folds for every model.
- It takes the true values at visible cells and each model's guesses at hidden cells.
- GAIN (Yoon et al., ICML 2018) reports a similar check.

## Confidence intervals without retraining

- The scored rows are resampled 1,000 times and every score is recomputed each time. This is called a bootstrap.
- Every model gets the same resampled rows, so we also get the difference between any two models, with an interval and a p-value.
- Whole rows are resampled, not single cells, because the cells in one row are filled in together.
- **Limit:** the intervals only reflect which cells happened to be hidden. They don't include retraining or a new mask, so they are narrower than the variation between separate runs.

## What's in the repo

- New: `metrics_v2.py`, `evaluate_v2.py` and `tests/test_evaluation_v2.py`. `evaluate.py` and `metrics.py` are unchanged.
- Version 2 doesn't import version 1. If we adopt it, we can delete `evaluate.py` and `metrics.py`. We'd also update the "exactly four metrics" line in `data/README.md` and the metric-name check in `tests/test_benchmark.py`.
- It reads only `predictions.csv` files, so it works for every model.

Run it from the repo root, with one or more models:

```bash
python -m shared_evaluation.evaluate_v2 --dataset credit_card --mode quick --rate 0.3 --seed 0 \
    --predictions data/output/credit_card/quick/rate_0.3_seed_0/<model>/predictions.csv [more files] --downstream
```

It prints a table and saves `evaluation_v2_<hash>.json` in that setting's output folder. Leave out `--downstream` to skip the prediction check.

## What we need to decide

1. Do we adopt the three core criteria?
2. The prediction check needs the raw data files (`data/raw/`), because the targets aren't in the prepared data. Can whoever prepared the data share them?
