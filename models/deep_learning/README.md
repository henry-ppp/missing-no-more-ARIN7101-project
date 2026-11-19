# Deep learning imputation

This folder contains the deep-learning part of the project. Hide cells with the
shared MCAR masks, impute them with several models, and score only the hidden
cells with shared RMSE/MAE.

The current dataset specification is `data/README.md`. The dataset list below
is from the deprecated note `meeting_notes/DEPRECATED_DATASETS_MEETING.md`.
Do not use it.

## Dataset choice from the deprecated note

This section records an old design. Do not choose datasets from it. Use
`data/README.md`.

The old note suggested a suite that covers easy, medium, hard, large/mixed,
finance, and quick small-data cases:

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
This follows the deprecated note: complete tables and a sample of at most
3,000 rows. Do not use that note for the current datasets.

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
  --output results/deep_learning/evaluation/legacy/results_breast_cancer_first_run.csv
```

Then run the next prepared datasets:

```bash
python3 -m models.deep_learning.run_benchmark \
  --datasets wine_white yeast \
  --methods autoencoder vae gain \
  --rates 0.1 0.2 0.3 0.5 \
  --epochs 100 \
  --output results/deep_learning/evaluation/legacy/results_wine_yeast_first_run.csv
```

Full project settings should use more epochs after all datasets are checked:

```bash
python3 -m models.deep_learning.run_benchmark \
  --datasets breast_cancer wine_white yeast credit_card_default fred_md energy_efficiency \
  --methods autoencoder vae gain \
  --rates 0.1 0.2 0.3 0.5 \
  --seed 0 \
  --epochs 300 \
  --output results/deep_learning/evaluation/legacy/results_benchmark.csv
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

## V2 shared-benchmark experiment results

These results use the same shared processed data, mask, output layout, checkpoint layout, and `shared_evaluation.evaluate` scoring style as `models/deep_learning_v2`. They are **not** compared against `dae_v2` here; this section records only the models in `models/deep_learning`.

Run configuration:

- Models: `dl_autoencoder`, `dl_vae`, `dl_gain`, `dl_mida`, `dl_miwae`
- Datasets: `california_housing`, `credit_card`, `gas_turbine`, `superconductivity`
- Modes: `quick`, `full`
- Evaluation missing rate: `0.3`
- Seed: `0`
- Run ID: `initial`
- Epochs: `150`
- Batch size: `256`
- Training mask rate: `0.3`
- Device used here: `cpu`

Main summary file:

```text
results/deep_learning/evaluation/summary/benchmark_summary.csv
```

Per-model summary files:

```text
results/dl_autoencoder/evaluation/summary/benchmark_summary.csv
results/dl_vae/evaluation/summary/benchmark_summary.csv
results/dl_gain/evaluation/summary/benchmark_summary.csv
```

Predictions are saved under:

```text
data/output/<dataset>/<mode>/rate_0.3_seed_0/<model>/initial/predictions.csv
```

Checkpoints are saved under:

```text
results/<model>/training/checkpoints/<dataset>/<mode>/rate_0.3_seed_0/initial/best.pt
```

### V2 benchmark results for `models/deep_learning`

| Model | Dataset | Mode | RMSE | MAE | R2 | Relative improvement |
| --- | --- | --- | --- | --- | --- | --- |
| dl_autoencoder | california_housing | full | 0.683928 | 0.417247 | 0.411225 | 0.251572 |
| dl_gain | california_housing | full | 0.914932 | 0.576012 | -0.041648 | -0.001217 |
| dl_vae | california_housing | full | 0.707714 | 0.432550 | 0.376245 | 0.225543 |
| dl_autoencoder | credit_card | full | 0.726245 | 0.299334 | 0.492035 | 0.278635 |
| dl_gain | credit_card | full | 0.874864 | 0.415654 | 0.255660 | 0.131015 |
| dl_vae | credit_card | full | 0.765253 | 0.303266 | 0.437440 | 0.239889 |
| dl_autoencoder | gas_turbine | full | 0.553843 | 0.350332 | 0.695200 | 0.445655 |
| dl_gain | gas_turbine | full | 0.621652 | 0.420779 | 0.615461 | 0.377785 |
| dl_vae | gas_turbine | full | 0.561126 | 0.356549 | 0.687057 | 0.438366 |
| dl_autoencoder | superconductivity | full | 0.239609 | 0.155955 | 0.942757 | 0.760848 |
| dl_gain | superconductivity | full | 0.512762 | 0.337384 | 0.738289 | 0.488216 |
| dl_vae | superconductivity | full | 0.265984 | 0.172301 | 0.929420 | 0.734523 |
| dl_autoencoder | california_housing | quick | 15.7109 | 0.720732 | 0.308907 | 0.000766 |
| dl_gain | california_housing | quick | 15.7552 | 1.3935 | -2.432474 | -0.002053 |
| dl_vae | california_housing | quick | 15.7184 | 0.771640 | 0.149675 | 0.000287 |
| dl_autoencoder | credit_card | quick | 1.3127 | 0.415341 | 0.349564 | 0.105280 |
| dl_gain | credit_card | quick | 1.6986 | 1.1018 | -0.495721 | -0.157759 |
| dl_vae | credit_card | quick | 1.2977 | 0.378026 | 0.390047 | 0.115496 |
| dl_autoencoder | gas_turbine | quick | 0.593332 | 0.388591 | 0.649412 | 0.399023 |
| dl_gain | gas_turbine | quick | 0.909565 | 0.696187 | 0.165329 | 0.078715 |
| dl_vae | gas_turbine | quick | 0.607485 | 0.398730 | 0.632542 | 0.384687 |
| dl_autoencoder | superconductivity | quick | 0.354503 | 0.241427 | 0.875784 | 0.647927 |
| dl_gain | superconductivity | quick | 0.930167 | 0.685099 | 0.142722 | 0.076209 |
| dl_vae | superconductivity | quick | 0.382462 | 0.258060 | 0.855439 | 0.620160 |

Notes:

- `dl_autoencoder` remains the strongest overall baseline.
- `dl_mida` improves some settings, especially `california_housing` and quick `credit_card`, by averaging MC-dropout imputations.
- `dl_vae` is usually close to `dl_autoencoder`.
- `dl_gain` and the current `dl_miwae` are weaker in this configuration and likely need further hyperparameter tuning before final use.


## Current limitations / next steps

- The shared RMSE/MAE metrics are numeric, so current experiments keep numeric
  columns only.
- Credit-card default includes categorical codes. The first pass treats them as
  numeric/coded features; a stronger later version could one-hot encode or use a
  mixed-type neural model.
- FRED-MD real publication gaps are mentioned in the deprecated note. This code
  covers the simulated complete-table MCAR benchmark first; real-gap evaluation
  can be added once those archived files are available in `data/`.
