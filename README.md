# Missing No More

HKU ARIN7101: Statistics in Artificial Intelligence group project. We compare numerical tabular imputation methods by hiding observed cells completely at random (MCAR). Each method receives the same incomplete matrix and is scored on the same hidden cells. Work is shared on `dev`.

Clone or update `dev` to use the **actual shared raw data and prepared Quick/Full matrices**. The published benchmark uses 30% MCAR and seed 0. Teammates can load the prepared matrices directly without downloading or rerunning preprocessing.

| Data | Published rows | Cleaned rows | Features | Quick rows |
| --- | ---: | ---: | ---: | ---: |
| Credit Card Default | 30,000 | 30,000 | 14 | 3,000 |
| California Housing | 20,640 | 20,640 | 8 | 3,000 |
| Superconductivity | 21,263 | 21,263 | 81 | 3,000 |
| Gas Turbine CO & NOx | 36,733 | 36,733 | 11 | 3,000 |

[Dataset sources, licenses, feature lists and CSV contract](data/README.md) · Sample counts and mask statistics are stored in each configuration's `metadata.json`.

## Repository map

```text
data/
  raw/                  Source CSVs and provenance metadata
  processed/            Full cleaned source plus Quick/Full inputs, masks, scalers, baseline and ground truth
  output/               Model predictions and run metadata
data_preprocessing/     Shared dataset preparation
meeting_notes/          Meeting notes and archived dataset research
models/
  deep_learning/        Existing AE, VAE and GAIN implementations (Yicheng and Eric)
  deep_learning_v2/     Mask-trained Denoising Autoencoder (DAE V2)
  iterative/            Implemented iterative imputer and historical results (Henry)
  knn/                  K-nearest neighbours placeholder (Hamad)
  random_forest/        Random Forest / MissForest placeholder (Blith)
shared_evaluation/      Evaluation version 2 (main protocol); version 1 kept for runners that still call it
results/                Model-owned training and evaluation directories; checkpoints stay local
tests/                  Automated data, model and evaluation checks
```

[Meeting notes](meeting_notes/) record group decisions. The [archived dataset research](meeting_notes/DEPRECATED_DATASETS_MEETING.md) was moved by the team; [data/README.md](data/README.md) is the active dataset specification.

## Environment

Use Python 3.12. The benchmark ran on Windows with Python 3.12.14, PyTorch 2.11.0+cu128, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, SciPy 1.18.1 and ucimlrepo 0.0.7. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# Optional: choose a CUDA build compatible with your NVIDIA driver before installing the other requirements.
python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.ones(2, device='cuda').sum().item()) if torch.cuda.is_available() else print('CUDA unavailable')"
python -m unittest discover -s tests -v
```

Without an NVIDIA-compatible setup, skip the CUDA installation line; the experiment runner uses CPU automatically when CUDA is unavailable, or accepts `--device cpu`.

## Use the shared benchmark

Read `data/processed/<dataset>/<quick|full>/rate_0.3_seed_0/X_cleaned_masked_standardized.csv` with `data_preprocessing.prepare_all.load_model_input`. Fit only on the incomplete matrix. Export standardized predictions with the exact same feature columns and `row_id` index under `data/output/`; do not use the evaluation-only ground truth for training or model selection.

To reproduce the data independently, the pipeline loads the checked-in raw CSVs:

```bash
python -m data_preprocessing.prepare_all --mode quick --skip-download
python -m data_preprocessing.prepare_all --mode full --skip-download
```

These commands verify and reuse the published snapshot. To regenerate into a fresh directory, add `--processed-dir <new-directory>` to both commands. `dataset_config.py` fixes schemas, `download.py` resolves official sources or the pinned California export, and `clean.py` selects numerical columns, drops invalid records and selects Quick/Full rows deterministically. `mask.py` creates the shared evaluation mask, `standardize.py` fits per-feature Z scores using observed cells only, `mean_baseline.py` fills missing cells by observed column mean, and `prepare_all.py` coordinates and exports those steps.

The flow is raw data → dataset-level `cleaned/X_cleaned.csv` → Quick/Full `X_cleaned.csv` → `X_cleaned_masked.csv` → `X_cleaned_masked_standardized.csv`. Evaluation reads the separate complete `X_cleaned_standardized.csv`, transformed with the same scaler fitted to the masked matrix's visible cells. See the [file layout](data/README.md#files-and-commands).

## Evaluate

Evaluation version 2 (`shared_evaluation/evaluate_v2.py`) is the main protocol for every model. It loads the configuration's truth, evaluation mask and mean baseline from `data/processed/`, checks their hashes, and scores only the hidden cells. The three headline scores are all measured against mean imputation (0 = mean imputation, higher is better):

- accuracy: RMSE improvement (`relative_improvement`);
- distribution: Wasserstein improvement (`wasserstein_improvement`);
- usefulness for prediction (`--downstream`): the share of the gap between mean imputation and the true table that an imputation closes (`gap_closed`). Credit card and HMDA use logistic regression scored by AUROC, California and superconductivity use linear regression scored by R², and gas turbine has no target, so it is skipped.

RMSE, MAE and macro per-feature R² are saved too, computed exactly as in version 1. Each score gets a 95% bootstrap interval from 1,000 row resamples. Models scored in the same call share the resamples, so every pair gets a paired difference and p-value. The intervals reflect only which cells were hidden, not retraining or a new mask.

Score all models of one configuration in one call, naming each file `NAME=PATH`:

```bash
python -m shared_evaluation.evaluate_v2 --dataset credit_card --mode full --rate 0.3 --seed 0 --downstream \
    --predictions MissForest=data/output/credit_card/full/rate_0.3_seed_0/MissForest/initial/predictions.csv \
                  dae_v2=data/output/credit_card/full/rate_0.3_seed_0/dae_v2/initial/predictions.csv
```

It prints a table and writes `data/output/<dataset>/<mode>/<configuration>/evaluation_v2_<hash>.json`. Add `--model MODEL --run-id ID` to save a report under `results/<model>/evaluation/<dataset>/<mode>/<configuration>/<run_id>/` instead. In Python, use `evaluate_files`, `evaluate` or `bootstrap` from `shared_evaluation.evaluate_v2`. The reasons for each score are in [EVALUATION_V2.md](shared_evaluation/EVALUATION_V2.md), and figures are drawn from the reports by [shared_visualization](shared_visualization/README.md).

Version 1 (`shared_evaluation/evaluate.py`: RMSE, MAE, R² and RMSE improvement, without intervals) is kept only because existing runners, including DAE V2's, still call it. Their predictions are saved, so score them with version 2 as above; nothing has to be retrained.

## Train and evaluate DAE V2

DAE V2 owns its results under `results/dae_v2/`. The `training/` directory holds
environment information, histories, settings, and experiment records. The
`evaluation/` directory holds V1 or V2 reports and the consolidated benchmark
in `evaluation/summary/`. Both organize runs by dataset, then Quick or Full,
configuration, and run ID. Checkpoints under `training/checkpoints/` are local
and are not published to Git. Predictions remain in `data/output/`; raw and
processed data remain under `data/`. Historical teammate CSVs are retained
under `results/deep_learning/evaluation/legacy/` and
`results/iterative/evaluation/legacy/`.

```bash
python -m models.deep_learning_v2.run_experiment --mode quick --rate 0.3 --seed 0 --run-id reproduction
python -m models.deep_learning_v2.run_experiment --mode full --rate 0.3 --seed 0 --run-id reproduction
```

The MLP encoder-decoder receives zero-filled standardized values and a visibility mask. Each training batch hides more originally observed cells; only those extra hidden cells contribute to masked MSE. Early stopping uses a separate fixed holdout of observed cells excluded from training inputs and targets. At inference only missing cells are replaced. Default settings: Adam, learning rate 0.001, batch size 256, 30% extra training masking, 10% observed validation holdout, ReLU, dropout 0.05, and at most 150 epochs with patience 20. Hidden widths and bottleneck vary with feature count. `--run-id` prevents overwriting earlier runs.

The runner writes a version 1 report for each run. Score its predictions with version 2 as described in [Evaluate](#evaluate).

See [DAE benchmark results](results/dae_v2/evaluation/summary/benchmark_summary.csv) (version 1 scores) and [training histories and configurations](results/dae_v2/training/). The initial experiment uses one seed and a transductive missing-cell holdout; these scores do not measure generalization to unseen rows or establish significance across seeds. Credits and change notices for the datasets are in [data/README.md](data/README.md). The project uses NumPy, pandas, PyTorch, SciPy, scikit-learn and ucimlrepo.

California Housing Quick has an unusually large RMSE because one hidden `AveOccup` value is about 1,314 observed standard deviations from its observed mean. That cell contributes 99.35% of the mean baseline's squared error for this configuration. The published data retain this observation; inspect MAE and macro R² alongside RMSE.

All eight initial DAE runs completed on the RTX 4090 Laptop GPU. DAE improved RMSE in all eight runs for this seed. Full-mode improvement ranged from 23.63% (California Housing) to 71.99% (Superconductivity).
