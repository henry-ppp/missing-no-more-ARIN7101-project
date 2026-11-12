# Shared imputation benchmark data

The four datasets below are fixed. Numbers are original published row counts;
complete-case cleaning can reduce them. No dataset has been downloaded in this change.
Full ordered schemas and exclusions live in `data_preprocessing/dataset_config.py`.

| Key | Official dataset / source | Original rows | Inputs | Raw cache |
| --- | --- | ---: | ---: | --- |
| credit_card | [UCI Default of Credit Card Clients, ID 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 14 | credit_card/source.csv |
| california_housing | [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | 20,640 | 8 | california_housing/source.csv |
| superconductivity | [UCI Superconductivty Data, ID 464](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | 21,263 | 81 | superconductivity/source.csv |
| gas_turbine | [UCI Gas Turbine CO and NOx Emission, ID 551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set) | 36,733 | 11 | gas_turbine/gt_2011.csv through gt_2015.csv |

## Feature and source contract

- Credit: LIMIT_BAL, AGE, BILL_AMT1..6, PAY_AMT1..6 in that order. UCI
  X1/X5/X12..23 aliases map to these names. Exclude ID, outcome, SEX,
  EDUCATION, MARRIAGE and repayment category codes. UCI donation: 2016-01-25.
- California: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
  Latitude, Longitude. Exclude MedHouseVal. Source describes 1990 census data.
- Superconductivity: number_of_elements followed by ten summaries (mean,
  wtd_mean, gmean, wtd_gmean, entropy, wtd_entropy, range, wtd_range, std,
  wtd_std) for each property in official order: atomic_mass, fie, atomic_radius,
  Density, ElectronAffinity, FusionHeat, ThermalConductivity, Valence. Exclude
  critical_temp. unique_m.csv is an alternative representation, never extra rows.
  UCI donation: 2018-10-11.
- Gas: AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX. Exclude year.
  Download URL: https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip
  Require exactly one member for each of five years; concatenate ascending year
  and original within-file order. Include CO and NOX as continuous imputation
  variables. UCI donation: 2019-11-28. The source's chronological split concerns
  supervised emission prediction; this benchmark instead studies transductive
  MCAR reconstruction and does not claim comparable supervised results.

For credit/superconductivity use ucimlrepo.fetch_ucirepo IDs 350/464; California
uses sklearn.datasets.fetch_california_housing. Raw API tables are exported
without cleaning (not claimed to be byte-identical source spreadsheets).
Gas yearly bytes and original ZIP are preserved. Source URL, row count and raw
SHA256 fingerprints identify the downloaded snapshot; upstream datasets are not
pinned to an immutable server version. Downloads are explicit and never run on
import. A complete valid local cache is reused; incomplete caches are refetched.

## Cleaning and reproducibility

Select only configured columns in configured order. Convert numeric values;
drop rows containing nonnumeric, NaN or infinite selected entries. Never fill
genuine missing truth. Retain duplicate numerical observations because they can
represent distinct source records. Reject empty/constant features rather than
silently changing the schema. No log transforms, clipping or arbitrary row cap.
Source filename and zero-based position form stable row IDs.

`quick` chooses at most 3,000 cleaned rows without replacement using NumPy
Generator(seed), retaining original relative order. `full` retains every cleaned
row in source order. Smaller quick datasets retain all rows and record the fact.
Default seed is 0, default missingness .3; rates .1, .2, .3, .5 are supported.
MCAR uses independent Bernoulli draws (realized proportion need not equal rate).
An empty evaluation mask or entirely hidden feature raises an error; masks are
never silently redrawn. All methods must reuse the exported rows and mask.

Order: clean -> sample -> mask -> fit scaler to observed cells only -> transform
-> mean baseline. Population standard deviation uses ddof=0. A zero observed
standard deviation uses scale=1 and is recorded. Standardized NaNs remain NaN.
Mean baseline fills each missing cell using its observed standardized column
mean. It is a separate full matrix, never the shared model input.

## Files and commands

Run from repository root after `pip install -r requirements.txt`:

```bash
python -m data_preprocessing.prepare_all --mode quick
python -m data_preprocessing.prepare_all --mode full --datasets credit_card gas_turbine --rate 0.1 0.2 0.3 0.5 --seed 0
python -m data_preprocessing.prepare_all --mode quick --skip-download --raw-dir data/raw --processed-dir data/processed
```

These commands download data unless --skip-download is given. They are for future
use and were not run against real data during this task. Existing processed
configurations are verified and reused, never remasked for each model. To rebuild,
use a new --processed-dir. Preparation fails on damaged or incomplete caches.

```text
data/
  raw/                     # uncleaned source caches
  processed/<dataset>/evaluation_only/X_clean_full.csv
  processed/<dataset>/<quick|full>/rate_0.3_seed_0/
    X_missing.csv
    X_mean_baseline.csv
    mask_eval.csv          # 1 = artificially hidden, 0 = observed
    scaler_stats.json
    metadata.json          # schema, row IDs, hashes, versions, cleaning, sampling
    evaluation_only/
      X_true_original.csv
      X_true_standardized.csv
  output/<dataset>/<mode>/rate_0.3_seed_0/<model>/
    predictions.csv
    evaluation_<prediction_sha256>.json
```

All CSV matrices start with a `row_id` index column. Models use only:

```python
from data_preprocessing.prepare_all import load_model_input
X_missing = load_model_input('data/processed/credit_card/quick/rate_0.3_seed_0')
# Fit your method on X_missing, preserve its index and columns, then:
# predictions.to_csv('data/output/credit_card/quick/rate_0.3_seed_0/my_model/predictions.csv', index_label='row_id')
```

Create the output parent directory before writing. Predictions must be a full
numerical matrix with identical row IDs, column order and shape in the shared
standardized scale. Fill all evaluation-hidden entries with finite predictions;
observed entries may be copied unchanged. Do not sample, remask or fit an external
scaler independently. Ground truth is accessible only via evaluator code after
predictions exist; never use it for training or hyperparameter selection. The
model loader returns only incomplete observations. The evaluation_only directory
is an API separation, not an operating-system security boundary.

Historical scripts in models/deep_learning preserve the earlier experimental
protocol and datasets/results. They are not runners for this new benchmark. Use
the shared loader and evaluator for all new comparisons. Existing imputer classes
are unchanged; any internal scaling must return predictions in input scale.
Deep learning V2 is deliberately empty.

```bash
python -m shared_evaluation.evaluate --dataset credit_card --mode quick --rate 0.3 --seed 0 --model dae --predictions data/output/credit_card/quick/rate_0.3_seed_0/dae/predictions.csv
```

The CLI supports --processed-dir and --output-dir, verifies metadata/artifact
hashes, rejects misalignment without reordering, prints metrics and writes JSON.
Output names include configuration, model and full prediction hash to distinguish
runs. Model names accept letters, digits, underscore and hyphen. The programmatic
API is `shared_evaluation.evaluate.evaluate(X_true, X_pred, mask_eval,
X_mean_baseline)`; use all DataFrames or all arrays (array callers own alignment).

Exactly four metrics, all over hidden cells in standardized space:
RMSE, MAE, macro per-feature R2, and 1 - RMSE_model/RMSE_mean_baseline.
R2 omits features with fewer than two hidden targets or zero target variance;
returns null/None if none qualify. Relative improvement is null/None for zero
baseline RMSE. Positive improvement beats mean; negative is worse. NaN/infinite
hidden predictions, invalid masks, empty masks and mismatched IDs are errors.

## Verification and attribution

`python -m unittest discover -s tests -v` uses only in-memory synthetic tables and
temporary synthetic fixtures; no downloading or training. Original shared MCAR
and RMSE/MAE code was adopted and extended. NumPy, pandas, scikit-learn,
ucimlrepo, PyTorch and the linked dataset authors must be acknowledged in the
presentation. This infrastructure and benchmark design do not constitute real
experimental results. Course guidance requires original contributions and clear
external attribution, appropriate statistics, a 15-20 minute group video and PDF
slides. Additional statistical analysis across repeated seeds belongs to future
experiments; do not treat a single run as evidence of significance.
