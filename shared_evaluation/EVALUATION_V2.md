# Evaluation v2: the main protocol

Every model is scored on three core criteria: accuracy, distribution and usefulness for prediction. Each one is measured against mean imputation and comes with a 95% confidence interval. The command is `python -m shared_evaluation.evaluate`.

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

## Core criteria

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
- HMDA uses whether a loan is government-backed (FHA, VA or USDA, from `loan_type`), with logistic regression scored by AUROC.
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

- `evaluate.py`, `metrics_v2.py` and `tests/test_evaluation_v2.py`. RMSE, MAE, R² and RMSE improvement match the older four-score reports. Wasserstein and its improvement are in the same call.
- It reads only `predictions.csv` files, so it works for every model. Older `evaluation_v1_*.json` files stay where they are.

Run it from the repo root with every model of one configuration, so all pairs share the same resamples. Name each file `NAME=PATH`:

```bash
python -m shared_evaluation.evaluate --dataset credit_card --mode quick --rate 0.3 --seed 0 --downstream \
    --predictions dae_v2=data/output/credit_card/quick/rate_0.3_seed_0/dae_v2/initial/predictions.csv \
                  <model>=data/output/credit_card/quick/rate_0.3_seed_0/<model>/<run_id>/predictions.csv
```

It prints a table. Standalone comparisons write to
`data/output/<dataset>/<mode>/<configuration>/evaluation_v2_<hash>.json`.
For a formal report owned by a model, supply `--model MODEL --run-id ID` to
write to `results/<model>/evaluation/<dataset>/<mode>/<configuration>/<run_id>/evaluation_v2_<hash>.json`.
`--output-dir` overrides the corresponding root. Leave out `--downstream`
to skip the prediction check. The check reads its targets from the raw files in `data/raw/`, because the
targets aren't in the prepared data.
