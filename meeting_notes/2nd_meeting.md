# Meeting note

**Project:** Missing No More (ARIN 7101)  
**Date:** 5 November 2026  
**Next meeting:** 12 November 2026

## Decision

We chose either one dataset from the six datasets. Until next meeting, everyone set up their own model and test it on the chosen dataset.

## Work before tonight's meeting

Each method should be set up on the chosen dataset, and each owner could bring initial imputation results.

| Method | Owner | Folder |
| --- | --- | --- |
| Random forest | Blith | `random_forest/` |
| k-nearest neighbours | Hamad | `knn/` |
| Deep learning | Yicheng, Eric | `deep_learning/` |
| Iterative model | Henry | `iterative/` |

Suggested report results with the shared setup: seed `0`, missing rates `0.1`, `0.2`, `0.3`, and `0.5`, and RMSE and MAE on the hidden numeric cells.

## Next meeting

Review the initial results on the chosen dataset and decide what to change before the full comparison.