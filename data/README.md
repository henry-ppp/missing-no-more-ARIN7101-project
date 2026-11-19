# Shared imputation benchmark data

The five datasets below are fixed. On 19 November 2026 the group put Superconductivity back and kept HMDA Rhode Island 2023. Both are required. Superconductivity's 81 collinear columns stay because they show how traditional methods and deep learning differ. HMDA is not an optional extra. The 18 November snapshot, which had replaced Superconductivity with HMDA, is in [LEGACY_README.md](LEGACY_README.md).

Numbers are original row counts. Complete-case cleaning retained every row of credit card, California, Superconductivity and gas turbine, and 17,809 of the 20,714 HMDA loans. The raw CSVs and Quick and Full benchmark files (30% MCAR, seed 0) of all five datasets are in the repository. Teammates should use the shared matrices directly; no download or preprocessing is required. If a full Superconductivity run is too slow, use the Quick configuration already in the repo.
Full ordered schemas and exclusions live in `data_preprocessing/dataset_config.py`.

HMDA's downstream target in `evaluate --downstream` is whether a loan is government-backed (FHA, VA or USDA, from the raw `loan_type` column). Superconductivity's target is critical temperature. Gas turbine has no target and is skipped. No model has HMDA results yet.

| Key | Official dataset / source | Original rows | Inputs | Raw cache |
| --- | --- | ---: | ---: | --- |
| credit_card | [UCI Default of Credit Card Clients, ID 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30,000 | 14 | credit_card/source.csv |
| california_housing | [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) | 20,640 | 8 | california_housing/source.csv |
| hmda_ri_2023 | [FFIEC HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/): Rhode Island, 2023, originated loans | 20,714 | 7 | hmda_ri_2023/source.csv |
| superconductivity | [UCI Superconductivty Data, ID 464](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | 21,263 | 81 | superconductivity/source.csv |
| gas_turbine | [UCI Gas Turbine CO and NOx Emission, ID 551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set) | 36,733 | 11 | gas_turbine/gt_2011.csv through gt_2015.csv |

## Feature and source contract

- Credit: LIMIT_BAL, AGE, BILL_AMT1..6, PAY_AMT1..6 in that order. UCI
  X1/X5/X12..23 aliases map to these names. Exclude ID, outcome, SEX,
  EDUCATION, MARRIAGE and repayment category codes. UCI donation: 2016-01-25.
- California: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
  Latitude, Longitude. Exclude MedHouseVal. Source describes 1990 census data.
- HMDA: loan_amount, loan_to_value_ratio, interest_rate, rate_spread, loan_term,
  property_value, income in that order. Rows: the 2023 loan-level records served by
  the FFIEC HMDA Data Browser for state RI with action_taken = 1 (loan originated).
  Exclude every other column:
  - identifiers and geography (lei, census_tract, county_code, derived_msa-md);
  - applicant demographics (race, ethnicity, sex, age);
  - coded categories (loan_type, loan_purpose, lien_status, ...) and denial reasons;
  - tract-level census summaries (tract_population, tract_minority_population_percent,
    ffiec_msa_md_median_family_income, tract_to_msa_income_percentage,
    tract_owner_occupied_units, tract_one_to_four_family_homes,
    tract_median_age_of_housing_units). They are identical for every loan in a tract,
    so imputation could copy them from neighbours.
  - cost fields that are mostly NA or Exempt (total_loan_costs, total_points_and_fees,
    origination_charges, discount_points, lender_credits; 41-100% missing).
  Units:
  - loan_amount and property_value: dollars, published as the midpoint of their
    $10,000 interval (CFPB disclosure policy for HMDA data from 2018, 84 FR 649);
  - income: thousands of dollars, can be negative;
  - interest_rate and rate_spread: percentage points, rate_spread can be negative;
  - loan_term: months;
  - loan_to_value_ratio: the reported combined loan-to-value ratio, in percent.
  "NA" and "Exempt" are nonnumeric, so the cleaning rule drops those rows; 17,809 remain.
  The 2023 data were published by the CFPB in 2024.
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

For credit and Superconductivity use ucimlrepo.fetch_ucirepo IDs 350 and 464. California
uses the pinned [Deshpande v1 export](https://doi.org/10.17632/hhb6nwfft8.1), verified value-for-value against scikit-learn fetch_california_housing. Raw UCI API tables are exported
without cleaning (not claimed to be byte-identical source spreadsheets).
HMDA uses the Data Browser API, https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=RI&years=2023&actions_taken=1,
which answers with a link to a static CSV on files.ffiec.cfpb.gov. That CSV is stored byte-for-byte as
hmda_ri_2023/source.csv; its row positions form the row IDs.
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
  output/<dataset>/<mode>/rate_0.3_seed_0/evaluation_v2_<hash>.json   # standalone comparison
results/<model>/evaluation/<dataset>/<mode>/rate_0.3_seed_0/<run_id>/
  evaluation_v2_<hash>.json                  # report saved with --model
  evaluation_v1_<prediction_sha256>.json     # older reports, left in place
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

Score predictions with `python -m shared_evaluation.evaluate`.
Score every model of one configuration in one call:

```bash
python -m shared_evaluation.evaluate --dataset credit_card --mode quick --rate 0.3 --seed 0 --downstream \
    --predictions dae_v2=data/output/credit_card/quick/rate_0.3_seed_0/dae_v2/initial/predictions.csv \
                  my_model=data/output/credit_card/quick/rate_0.3_seed_0/my_model/initial/predictions.csv
```

The CLI verifies metadata/artifact hashes, rejects misalignment without
reordering, prints a table and writes one JSON report. Without `--model` the
report is a standalone comparison under `data/output/`; with `--model MODEL
--run-id ID` it goes under `results/MODEL/evaluation/`. `--processed-dir`,
`--raw-dir` and `--output-dir` override the roots. Name each file `NAME=PATH`;
model names accept letters, digits, underscore and hyphen, but not `mean` or
`true`. The programmatic API is `shared_evaluation.evaluate.evaluate(X_true,
X_pred, mask_eval, X_mean_baseline)` for point scores and `bootstrap(...)` for
intervals; use all DataFrames or all arrays (array callers own alignment).

All scores are over hidden cells in standardized space. Headline scores, each
0 for mean imputation and higher for better: RMSE improvement
(1 - RMSE_model/RMSE_mean_baseline), Wasserstein improvement, and, with
`--downstream`, the share of the downstream gap between mean imputation and the
true table that the model closes. RMSE, MAE, macro per-feature R2 and the
Wasserstein distance are also saved. Each score has a 95% basic bootstrap
interval over resampled rows, and every pair of models scored together gets a
paired difference and p-value. R2 omits features with fewer than two hidden
targets or zero target variance; returns null/None if none qualify. Improvements
are null/None for a zero baseline. NaN/infinite hidden predictions, invalid
masks, empty masks and mismatched IDs are errors. Details are in
[EVALUATION_V2.md](../shared_evaluation/EVALUATION_V2.md).

Older `evaluation_v1_*.json` files stay where they are. New reports are `evaluation_v2_<hash>.json`.

## Verification and attribution

`python -m unittest discover -s tests -v` uses only in-memory synthetic tables and
temporary synthetic fixtures; no downloading or training. Original shared MCAR
and RMSE/MAE code was adopted and extended. NumPy, pandas, scikit-learn,
ucimlrepo, PyTorch, the linked dataset authors and the CFPB/FFIEC as HMDA publisher must be acknowledged in the
presentation. Actual DAE experiment results are in [results/](../results/). Course guidance requires original contributions and clear
external attribution, appropriate statistics, a 15-20 minute group video and PDF
slides. Additional statistical analysis across repeated seeds belongs to future
experiments; do not treat a single run as evidence of significance.

## Dataset redistribution and attribution

The credit card, California, Superconductivity and gas turbine source releases are labeled [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The license applies to the datasets under their respective terms; the corresponding derived artifacts are marked as modified below.
The HMDA data have no dataset license, and they don't need one: they are in the public domain. The CFPB's
[legal notice](https://www.consumerfinance.gov/privacy/website-privacy-policy/) (revised December 2023) says "Information
created by the CFPB is in the public domain and you may reproduce, publish, or otherwise use it without the Bureau's
permission" and asks users to cite the Bureau as the source. The data are published under the Home Mortgage Disclosure
Act for public use, with privacy modifications applied before release, and no terms restrict them by use or by country.
Committing, changing and sharing the raw and derived HMDA files is therefore allowed; credit the CFPB/FFIEC as the
source, as in the table below. (A missing license alone would not grant these rights, because unlicensed works are
normally all rights reserved. Here they come from the public-domain notice.)

| Dataset | Attribution and release | Changes in this repository |
| --- | --- | --- |
| Credit Card Default | I-Cheng Yeh, [UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) ([source and license](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)) | Raw API table exported to CSV. |
| California Housing | R. Kelley Pace and Ronald Barry (1997), StatLib / 1990 U.S. Census; Vrushali Deshpande, [Scikit-learn CSV Export, v1](https://data.mendeley.com/datasets/hhb6nwfft8/1) | Raw CSV retained byte-for-byte from the licensed release. |
| HMDA Rhode Island 2023 | Consumer Financial Protection Bureau for the Federal Financial Institutions Examination Council, [HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/), 2023 loan-level data, Rhode Island, originated loans ([public domain](https://www.consumerfinance.gov/privacy/website-privacy-policy/)) | Data Browser CSV retained byte-for-byte. |
| Superconductivity | Kam Hamidieh, [UCI Superconductivty Data](https://doi.org/10.24432/C53P47) ([source and license](https://archive.ics.uci.edu/dataset/464/superconductivty+data)) | Raw API table exported to CSV. |
| Gas Turbine | [UCI Gas Turbine CO and NOx Emission Data Set](https://doi.org/10.24432/C5WC95) ([source and license](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)) | Five original yearly CSVs retained byte-for-byte. |

For all four sources, the derived files select the documented features, clean and select rows, then mask and scale values. Baseline and DAE prediction matrices additionally fill missing cells. Raw-file checksums and source metadata are next to the CSVs. The complete matrices are publicly visible in Git, so the separation is an experiment protocol: model code may read only the incomplete input and never use truth values for fitting or tuning.
