# Missing No More

Tabular data imputation for ARIN 7101. Work on `dev`. Each person owns one folder and can work there without touching the others.

| Folder | Method | Owner |
| --- | --- | --- |
| `random_forest/` | Random forest | Blith |
| `knn/` | k-nearest neighbours | Hamad |
| `deep_learning_yicheng/` | Deep learning | Yicheng |
| `deep_learning_eric/` | Deep learning | Eric |
| `iterative/` | Iterative model | Henry |

Put shared datasets in `data/`. Those files are not committed.

Compare methods with the same hidden cells. From the repo root:

```python
from shared.mask import apply_mask, mcar_mask
from shared.metrics import mae, rmse

mask = mcar_mask(X, rate=0.2, seed=0)
X_missing = apply_mask(X, mask)
# Impute X_missing with your own method, then score against the original X.
rmse(X, imputed, mask)
mae(X, imputed, mask)
```

Use seed `0` and missing rates `0.1`, `0.2`, `0.3`, and `0.5`. Pass numeric columns. `rmse` and `mae` score only the cells the mask hid.
