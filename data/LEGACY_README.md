# LEGACY

This is the 18 November 2026 snapshot, when HMDA Rhode Island 2023 replaced Superconductivity. On 19 November 2026 the group put Superconductivity back and kept HMDA, so the current specification is [README.md](README.md).

---

# Shared imputation benchmark data

The four datasets below are fixed. Numbers are original published row counts;
complete-case cleaning retained every row in the published snapshot. The raw CSVs, Quick and Full benchmark files (30% MCAR, seed 0), and DAE predictions are included in Git. Teammates should use the shared matrices directly; no download or preprocessing is required.
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
uses the pinned [Deshpande v1 export](https://doi.org/10.17632/hhb6nwfft8.1), verified value-for-value against scikit-learn fetch_california_housing. Raw UCI API tables are exported
without cleaning (not claimed to be byte-identical source spreadsheets).
Gas yearly bytes are preserved; the redundant ZIP is kept in the local cache only. Source URL, row count and raw
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

On a checkout, the source CSVs are already present. The commands reuse them and only download missing sources when --skip-download is absent. Existing processed
configurations are verified and reused, never remasked for each model. To rebuild,
use a new --processed-dir. Preparation fails on damaged or incomplete caches.

```text
data/
  raw/                     # uncleaned source caches
  processed/<dataset>/cleaned/X_cleaned.csv
  processed/<dataset>/<quick|full>/rate_0.3_seed_0/
    X_cleaned.csv
    X_cleaned_masked.csv
    X_cleaned_masked_standardized.csv
    X_cleaned_standardized.csv
    X_mean_baseline.csv
    mask_eval.csv          # 1 = artificially hidden, 0 = observed
    scaler_stats.json
    metadata.json          # schema, row IDs, hashes, versions, cleaning, sampling
  output/<dataset>/<mode>/rate_0.3_seed_0/<model>/
    predictions.csv
results/<model>/evaluation/<dataset>/<mode>/rate_0.3_seed_0/<run_id>/
  evaluation_v1_<prediction_sha256>.json
```

The preprocessing stages are:

1. `cleaned/X_cleaned.csv`: the entire cleaned dataset in original units.
2. Configuration `X_cleaned.csv`: the selected Quick or Full rows, still complete
   and in original units. Full intentionally keeps a physical, byte-identical
   copy of the dataset-level file.
3. `X_cleaned_masked.csv`: the selected rows with MCAR cells blanked, in original units.
4. `X_cleaned_masked_standardized.csv`: the model input, with the same missing cells.
   Fit `scaler_stats.json` exclusively on the visible cells of the masked matrix.

Evaluation uses `X_cleaned_standardized.csv`, the complete selected rows
transformed with the **same observed-only scaler**, and `mask_eval.csv`.
`X_mean_baseline.csv` fills the standardized input's missing entries with the
observed-column means. `metadata.json` records row IDs, features, source hashes,
and SHA-256 hashes of the other configuration files. For migrated snapshots,
`previous_metadata_sha256` links to the original metadata digest recorded in
historical DAE reports; those reports are preserved unchanged.

All matrices start with a `row_id` index column. Models load only the standardized
incomplete matrix:

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
model loader returns only incomplete observations. This is an API boundary;
the complete matrices are visible on disk.

Historical scripts in models/deep_learning preserve the earlier experimental
protocol and datasets/results. They are not runners for this new benchmark. Use
the shared loader and evaluator for all new comparisons. Existing imputer classes
are unchanged; any internal scaling must return predictions in input scale.
DAE V2 uses the shared loader. See the [project README](../README.md) for commands.

```bash
python -m shared_evaluation.evaluate --dataset credit_card --mode quick --rate 0.3 --seed 0 --model dae_v2 --run-id verification --predictions data/output/credit_card/quick/rate_0.3_seed_0/dae_v2/initial/predictions.csv
```

The CLI requires --model, defaults --run-id to standalone, and supports --processed-dir and --output-dir (override the results root), verifies metadata/artifact
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
presentation. Actual DAE experiment results are in [results/](../results/). Course guidance requires original contributions and clear
external attribution, appropriate statistics, a 15-20 minute group video and PDF
slides. Additional statistical analysis across repeated seeds belongs to future
experiments; do not treat a single run as evidence of significance.

## Dataset redistribution and attribution

These source releases are labeled [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The license applies to the datasets under their respective terms; the corresponding derived artifacts are marked as modified below.

| Dataset | Attribution and release | Changes in this repository |
| --- | --- | --- |
| Credit Card Default | I-Cheng Yeh, [UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) ([source and license](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)) | Raw API table exported to CSV. |
| California Housing | R. Kelley Pace and Ronald Barry (1997), StatLib / 1990 U.S. Census; Vrushali Deshpande, [Scikit-learn CSV Export, v1](https://data.mendeley.com/datasets/hhb6nwfft8/1) | Raw CSV retained byte-for-byte from the licensed release. |
| Superconductivity | Kam Hamidieh, [UCI Superconductivty Data](https://doi.org/10.24432/C53P47) ([source and license](https://archive.ics.uci.edu/dataset/464/superconductivty+data)) | Raw API table exported to CSV. |
| Gas Turbine | [UCI Gas Turbine CO and NOx Emission Data Set](https://doi.org/10.24432/C5WC95) ([source and license](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)) | Five original yearly CSVs retained byte-for-byte. |

For all four sources, the derived files select the documented features, clean and select rows, then mask and scale values. Baseline and DAE prediction matrices additionally fill missing cells. Raw-file checksums and source metadata are next to the CSVs. The complete matrices are publicly visible in Git, so the separation is an experiment protocol: model code may read only the incomplete input and never use truth values for fitting or tuning.
