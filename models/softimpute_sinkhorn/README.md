# SoftImpute and optimal-transport (Sinkhorn) imputation

Two methods from families no other model in the project covers, run on the shared benchmark with **one shared seed (0)** and no repeats. The uncertainty comes from the shared evaluator's bootstrap.

- **SoftImpute**: low-rank matrix completion. It tests whether a table's columns share a few underlying directions, which collinear tables such as superconductivity should have. It is the linear counterpart of the autoencoders.
- **Sinkhorn**: batch optimal-transport imputation. Its training objective is a distribution distance, so it speaks directly to the distribution half of the project question.

## Run it

From the repository root:

```bash
python -m models.softimpute_sinkhorn.run_experiment --plan                  # what is done, what is left
caffeinate -is python -m models.softimpute_sinkhorn.run_experiment --rates 0.3
```

The second command runs both models on the four required datasets in `full` mode at 30% missing. Leave out `--rates` to run every prepared rate, and add `--datasets hmda_ri_2023` for the optional dataset. Runs are saved as they finish, skipped on a rerun with the same settings, and a run ID reused with other settings is refused.

| Option | Default | Meaning |
| --- | --- | --- |
| `--models`, `--datasets` | both, the four required | e.g. `--models SoftImpute` |
| `--rates` | every prepared rate | missing rates (must be prepared) |
| `--run-id` | `initial` | names this run's folders |
| `--sinkhorn-visits` | 500 | Sinkhorn's budget (below) |
| `--no-tune` | off | SoftImpute with lambda = 0.1 x the largest singular value |
| `--n-boot` | 1000 | bootstrap resamples in the evaluation |
| `--jobs` | number of cores | runs at the same time |

## The methods

**SoftImpute** (Mazumder, Hastie & Tibshirani, *JMLR* 11, 2010, Algorithm 1). Start with the gaps at 0, the column mean of the standardised data. Then repeat:
1. take the SVD of the filled table;
2. shrink every singular value by lambda, dropping those below it;
3. put the resulting low-rank values back into the gaps.

It stops when the filled values change by less than 1e-5 (relative), or after 500 steps. lambda sets how many dimensions survive.

**Batch Sinkhorn imputation** (Muzellec, Josse, Boyer & Cuturi, *ICML* 2020, Algorithm 1). The missing values are free parameters. Each step:
1. draws two random batches of 128 rows;
2. computes the debiased Sinkhorn divergence between them (an entropic optimal-transport distance; cost |x − y|²/2);
3. moves the batches' missing values down its gradient with RMSprop.

Over many steps any two batches of the filled table come to look like samples of one distribution.

The Sinkhorn divergence uses log-domain Sinkhorn iterations with epsilon-scaling. The gradient comes from one last update (envelope theorem), as in the geomloss library the authors' code uses (Feydy et al., *AISTATS* 2019).

## Tuning

**SoftImpute: lambda is tuned at every missing rate.** It sets the rank, and the right rank differs a lot between tables (in the pilot, 7 of 8 dimensions on California, 59 of 81 on superconductivity). Tuning costs only seconds to minutes.
- **Hide-and-check**, with the random-forest package's rule and seed: a further 10% of the visible cells of the model input are hidden, and the candidates fill them.
- **Candidates**: lambda = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5} × the largest singular value. They are fitted largest first, each warm-started from the previous one (the paper's path algorithm).
- **Winner**: the strongest shrinkage whose check RMSE is within 0.1% of the lowest. Small lambdas warm-started from a larger one barely move, so their errors differ only in the 4th decimal.
- The final fit follows the same path down to the winner, so it is computed exactly as the tuned candidate was.
- Saved to `results/softimpute_sinkhorn/SoftImpute/hyperparameters/<dataset>_<mode>_<configuration>.json` with every candidate's error, and reused by a rerun.

**Sinkhorn: no tuning.** The paper fixes its settings for every dataset "with no further dataset-dependent tuning":
- batch 128;
- RMSprop with step 0.01;
- starting values: the column mean plus N(0, 0.1) noise;
- epsilon = 5% of the median half squared distance between rows of the mean-imputed table, so it adapts to each table.

The one remaining choice is the **iteration budget**, set as visits per row: how often each row is drawn into a batch, which is iterations × 256 / rows. Using visits gives tables of any size the same amount of optimisation. It was chosen once, before the benchmark runs, by hide-and-check on the quick tables (see *Choosing the budget*).

Tuning reads only the incomplete model input (`load_model_input`). The cells the benchmark scores stay hidden, and the evaluation truth is never opened.

## Choosing the budget

The budget was chosen on 26 Nov 2026, before any benchmark run, by hide-and-check on the four quick tables (3,000 rows, 30% missing, seed 0). Sinkhorn ran up to 1,200 visits per row, and the check cells were scored about every 85 visits. Only visible cells were scored.

| Budget (visits per row) | 171 | 256 | 341 | 427 | **512** | 597 | 768 | 853 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Worst shortfall from a table's best RMSE improvement | 0.016 | 0.014 | 0.011 | 0.007 | **0.008** | 0.007 | 0.005 | 0.011 |
| Worst shortfall from a table's best Wasserstein improvement | 0.043 | 0.023 | 0.019 | 0.015 | **0.009** | 0.010 | 0.010 | 0.005 |

- Superconductivity levelled off by about 256 visits and credit card by about 340. California's Wasserstein score levelled off by about 430, and gas turbine's crept up until about 500.
- Past about 500 visits the scores only fluctuate by about ±0.01 with no further gain. So the budget is **500 visits**: the smallest one at which every table is within 0.01 of its best on both scores.
- Cost at full size: iterations = 500 × rows / 256. That is about 40,000 for California and superconductivity, 59,000 for credit card and 72,000 for gas turbine, at about 0.04 s each on the M2.

## Where results go

`<run>` is `<dataset>/<mode>/rate_<r>_seed_<s>/<run_id>`:

```text
data/output/<dataset>/<mode>/rate_<r>_seed_<s>/<Model>/<run_id>/
    predictions.csv, metadata.json               the shared location, for team-wide evaluation
results/softimpute_sinkhorn/<Model>/              Model = SoftImpute or Sinkhorn
    hyperparameters/<dataset>_<mode>_<configuration>.json    SoftImpute's tuning, every candidate
    experiment/<run>/experiment.json             settings, parameters, time, headline scores with intervals
    evaluation/<run>/evaluation_v2_*.json        shared evaluation: intervals, Wasserstein, downstream
    evaluation/summary/benchmark_summary.csv     one row per run
results/softimpute_sinkhorn/comparison/evaluation/<run>/evaluation_v2_*.json   both models, paired
local_models/softimpute_sinkhorn/<Model>/training/<run>/                       not pushed
    training_history.csv                         SoftImpute: change and rank per step; Sinkhorn: loss per 1% of steps
    run.json                                     exact settings, details, timing, versions
```

## Outside sources

- The algorithms follow the two papers above. The code, the tuning, the budget choice and the runner are ours. Seeding and the hide-and-check rule are shared with `models/random_forest`.
- The Sinkhorn authors' repository (github.com/BorisMuzellec/MissingDataOT) has no licence, so none of its code is used.
- PyTorch provides the gradients and RMSprop.
- References:
  - Mazumder, Hastie & Tibshirani (2010), *Journal of Machine Learning Research* 11:2287–2322.
  - Muzellec, Josse, Boyer & Cuturi (2020), *Proceedings of ICML*, PMLR 119:7130–7140.
  - Feydy, Séjourné, Vialard, Amari, Trouvé & Peyré (2019), *Proceedings of AISTATS*, PMLR 89:2681–2690.
