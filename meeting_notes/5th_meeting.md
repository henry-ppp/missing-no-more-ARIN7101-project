# Meeting note

**Project:** Missing No More (ARIN 7101)  
**Date:** 26 November 2026

## Notes

1. Each person commits one markdown file at `docs/slides/<name>.md` covering their own part. Henry and Hamad will build the slide deck from these files. Include anything a slide needs: the line to say, the number to show, and the path to a figure. Due Saturday 28 November.
2. Henry and Hamad make the slide deck. Due Thursday 3 December.
3. The presentation will be rehearsed and recorded on campus on Saturday 5 December, in the afternoon.
4. The presentation is split as follows.

| Part | Who |
| --- | --- |
| Intro | 1 person (TBD) |
| Dataset and metrics | Blith |
| Methods | All 5. Each person introduces their own method. |
| Results, including the comparison | 1 to 2 persons (TBD) |
| Conclusion | 1 person (TBD) |

## What to put in the markdown

Write info for the slide.

- What the method does
- Setup: dataset, Quick or Full, missing rate, seed, etc
- The numbers for the results slide, with the dataset named beside each one. Use the shared scores against mean imputation: RMSE improvement, Wasserstein improvement, and downstream gap closed. 0 is mean imputation and higher is better. Add the 95% bootstrap interval. Gas turbine has no gap closed.
- The figure and the points that figure makes. Shared charts live under `figures/comparative_plots/`.
- Conclusion

Format example only. Replace the blanks with your own results:

```markdown
## Method
MissForest starts from column means, then refits a random forest for each column, fewest gaps first. It stops when the change between rounds grows and keeps the round before. 100 trees, at most 10 rounds. MICE-RF is scikit-learn's iterative imputer with extra trees, also at most 10 rounds.

## Setup
Credit card, California housing, superconductivity, gas turbine. Full, 30% MCAR, seed 0, both models. Use Quick for superconductivity only if Full is too slow.

## Numbers
0 is mean imputation and higher is better. Put the 95% bootstrap interval next to each score. Take the numbers from `results/random_forest/<Model>/evaluation/summary/benchmark_summary.csv`. Gas turbine has no target, so gap closed is not available.

| Dataset | RMSE improvement | Wasserstein improvement | Gap closed |
| --- | --- | --- | --- |
| Credit card | … (…, …) | … (…, …) | … (…, …) |
| California housing | … (…, …) | … (…, …) | … (…, …) |
| Superconductivity | … (…, …) | … (…, …) | … (…, …) |
| Gas turbine | … (…, …) | … (…, …) | not available |

## Figure
`figures/MissForest/training_MissForest_full.png` — change between rounds, which is the stopping rule.
`figures/comparative_plots/distribution_credit_card_full_BILL_AMT1.png` — the shaded gap is the Wasserstein distance for one bill-amount column.

## Conclusion
The intervals only reflect which cells were hidden, not a new training run. Use Full for the headline numbers: California Quick has one extreme `AveOccup` cell, and that cell breaks RMSE and gap closed, etc.
```
