# Deep learning imputation

This folder contains the deep-learning part of the project. It follows the
project dataset design in `data/DATASETS_MEETING copy.md`: choose complete
benchmark tables, hide cells with the shared MCAR masks, impute them with several
models, and score only the hidden cells with shared RMSE/MAE.

## Dataset choice from the design document

We should not evaluate on an arbitrary single dataset only. The design document
suggests a suite that covers easy, medium, hard, large/mixed, finance, and quick
small-data cases:

| Dataset key | Why we include it | Expected file in `data/` |
| --- | --- | --- |
| `breast_cancer` | Easy sanity check; strongly correlated numeric features. | `breast_cancer.csv` |
| `wine_white` | Classic benchmark; moderately related chemistry columns. | `winequality-white.csv` |
| `yeast` | Hard case; weak feature relationships. | `yeast.csv` |
| `credit_card_default` | Larger finance dataset with coded categorical/numeric columns. | `credit_card_default.csv` |
| `fred_md` | Finance/economics panel and realistic dependence structure. | `fred_md_complete.csv` |
| `energy_efficiency` | Small complete numeric dataset already added to the project plan. | `energy_efficiency.csv` |

Data files stay in `data/` and are ignored by git. The loader in `datasets.py`
drops labels/IDs/date columns, keeps numeric columns, removes duplicate and
incomplete rows, and caps large datasets at 3,000 rows unless told otherwise.
This matches the design note that experiments should use complete tables and a
sample of at most 3,000 rows.

## Deep-learning methods

The benchmark currently includes three different deep-learning imputation methods:

1. `autoencoder` - denoising autoencoder baseline.
2. `vae` - variational autoencoder, a generative model.
3. `gain` - adversarial GAIN-style model inspired by Yoon et al. (2018).

All methods receive the same incomplete table for each dataset/rate and are
scored with the same hidden-cell mask.

## Data preparation and benchmark commands

From the repository root, list configured datasets:

```bash
python3 -m models.deep_learning.run_benchmark --list-datasets
```

Prepare the public UCI datasets that can be downloaded directly:

```bash
python3 -m models.deep_learning.prepare_datasets --datasets breast_cancer wine_white yeast
```

A good workflow is to run one dataset first, then expand to the others after the
loaders and model settings are confirmed.

First run, one dataset and all methods. Add `--save-models` if fitted model
checkpoints should be kept:

```bash
python3 -m models.deep_learning.run_benchmark \
  --datasets breast_cancer \
  --methods autoencoder vae gain \
  --rates 0.1 0.2 0.3 0.5 \
  --epochs 100 \
  --save-models \
  --output models/deep_learning/results_breast_cancer_first_run.csv
```

Then run the next prepared datasets:

```bash
python3 -m models.deep_learning.run_benchmark \
  --datasets wine_white yeast \
  --methods autoencoder vae gain \
  --rates 0.1 0.2 0.3 0.5 \
  --epochs 100 \
  --output models/deep_learning/results_wine_yeast_first_run.csv
```

Full project settings should use more epochs after all datasets are checked:

```bash
python3 -m models.deep_learning.run_benchmark \
  --datasets breast_cancer wine_white yeast credit_card_default fred_md energy_efficiency \
  --methods autoencoder vae gain \
  --rates 0.1 0.2 0.3 0.5 \
  --seed 0 \
  --epochs 300 \
  --output models/deep_learning/results_benchmark.csv
```

`credit_card_default`, `fred_md`, and `energy_efficiency` are supported by the
loader once their CSV files are placed in `data/` with the expected names.

The result CSV contains:

- dataset
- method
- missing rate
- raw RMSE and MAE
- normalized NRMSE and NMAE, computed by dividing each hidden-cell error by that column's range
- row/column counts
- hidden-cell count
- runtime
- checkpoint path, when `--save-models` is used

Saved checkpoints are written to `models/deep_learning/checkpoints/` and are ignored by
git. Each checkpoint stores the model weights and the preprocessing statistics
needed for that fitted imputer.

## Quick one-CSV runner

For quick debugging on any local numeric CSV:

```bash
python3 -m models.deep_learning.run_experiment data/my_dataset.csv \
  --methods autoencoder vae gain \
  --rates 0.1 0.2 0.3 0.5
```

If the file contains labels or IDs, drop them:

```bash
python3 -m models.deep_learning.run_experiment data/breast_cancer.csv \
  --drop-column diagnosis \
  --methods autoencoder vae gain
```

## Python example

```python
from data_preprocessing.mask import apply_mask, mcar_mask
from shared_evaluation.metrics import mae, rmse
from models.deep_learning.datasets import load_dataset
from models.deep_learning.methods import create_imputer
from models.deep_learning.normalized_metrics import nmae, nrmse

X = load_dataset("breast_cancer")
mask = mcar_mask(X, rate=0.2, seed=0)
X_missing = apply_mask(X, mask)

for method in ["autoencoder", "vae", "gain"]:
    imputer = create_imputer(method, seed=0, epochs=300, batch_size=128, verbose=False)
    imputed = imputer.fit_transform(X_missing)
    print(method, rmse(X, imputed, mask), mae(X, imputed, mask), nrmse(X, imputed, mask), nmae(X, imputed, mask))
```

## Current limitations / next steps

- The shared RMSE/MAE metrics are numeric, so current experiments keep numeric
  columns only.
- Credit-card default includes categorical codes. The first pass treats them as
  numeric/coded features; a stronger later version could one-hot encode or use a
  mixed-type neural model.
- FRED-MD real publication gaps are mentioned in the design document. This code
  covers the simulated complete-table MCAR benchmark first; real-gap evaluation
  can be added once those archived files are available in `data/`.
