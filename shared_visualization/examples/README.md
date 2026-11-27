# Example figures

How to remake the shared figures: full data, seed 0, the four required datasets. The PNGs in this folder are copies of one figure of each type, made by the two steps below. Run every command from the repository root, in the project environment (`conda activate project`, or `pip install -r requirements.txt`). Each figure is saved as PDF, SVG and PNG, plus a CSV of exactly what it shows.

| Figure | Output |
| --- | --- |
| Scoreboards, one per score: RMSE, Wasserstein, downstream gap closed (all 12 methods, 30% missing) | `figures/comparative_plots/scoreboard_full_seed_0_rate_0.3_{rmse,wasserstein,gap_closed}.*` |
| Accuracy vs distribution (all 12 methods) | `figures/comparative_plots/accuracy_vs_distribution_full_seed_0.*` |
| Overall ranking (all 12 methods) | `figures/comparative_plots/ranking_full_seed_0.*`, `rank_table_full_seed_0.csv` |
| Effect of the missing rate (the four methods run at 10% and 30%) | `figures/comparative_plots/missing_rate_full_seed_0.*` |
| What the Wasserstein distance measures | `figures/comparative_plots/distribution_credit_card_full_BILL_AMT1.*` |
| Training curves: MissForest, MICE-RF, DAE v2 | `figures/<model>/training_<model>_full.*` |
| Every score with its interval, and every paired p-value | `figures/comparative_plots/scores.csv`, `pairs.csv` |

## 1. Score all models of each dataset together

The figures read evaluation reports. Scoring every model of a setting in one call pairs them in the bootstrap, so the scoreboard can mark which differences are significant. This takes about 20 minutes, mostly superconductivity.

```bash
OUT=figures/comparative_plots/reports
for ds in credit_card california_housing superconductivity gas_turbine; do
  o=data/output/$ds/full/rate_0.3_seed_0
  python -m shared_evaluation.evaluate --dataset $ds --mode full --rate 0.3 --downstream --output-dir $OUT \
    --predictions MissForest=$o/MissForest/initial/predictions.csv MICE-RF=$o/MICE-RF/initial/predictions.csv \
      knn=$o/knn/initial/predictions.csv iterative=$o/iterative/initial/predictions.csv \
      SoftImpute=$o/SoftImpute/initial/predictions.csv Sinkhorn=$o/Sinkhorn/initial/predictions.csv \
      dae_v2=$o/dae_v2/initial/predictions.csv dl_autoencoder=$o/dl_autoencoder/initial/predictions.csv \
      dl_gain=$o/dl_gain/optimized/predictions.csv dl_mida=$o/dl_mida/initial/predictions.csv \
      dl_miwae=$o/dl_miwae/optimized/predictions.csv dl_vae=$o/dl_vae/initial/predictions.csv
  o=data/output/$ds/full/rate_0.1_seed_0
  python -m shared_evaluation.evaluate --dataset $ds --mode full --rate 0.1 --downstream --output-dir $OUT \
    --predictions MissForest=$o/MissForest/initial/predictions.csv MICE-RF=$o/MICE-RF/initial/predictions.csv \
      SoftImpute=$o/SoftImpute/initial/predictions.csv Sinkhorn=$o/Sinkhorn/initial/predictions.csv
done
```

- GAIN and MIWAE use their `optimized` runs, made with the current code defaults; the `initial` runs used earlier defaults that `models/deep_learning/README.md` calls weaker. Every other model uses its `initial` run.
- HMDA is optional and only the random forests have it, so it is left out.
- Only MissForest, MICE-RF, SoftImpute and Sinkhorn have runs at 10%. Rates 0.2 and 0.5 are prepared but nobody has run them.

## 2. Make the figures

```bash
python -m shared_visualization.examples.make_figures
```

- **Twelve methods, eight colours.** `style.py` lists seven models and keeps the eighth colour spare for a new one. `make_figures.py` registers the other five. Within a family every colour is unique. A colour used in both families always comes with a different shape (pointed shapes are deep learning), and methods are always named on the figure.
- **One scoreboard per score.** Twelve rows per panel don't fit three rows of panels on one slide at 14 pt, so each score gets its own slide.
- **Ranking.** Four settings (the four datasets at 30%) are too few for any pair to be significant, so the figure shows average ranks only and says so.
- **Training curves.** Random-forest histories are in `local_models/`, which isn't pushed, so those two figures can only be remade on the machine that ran the random forests. DAE v2 histories are in `results/` and work anywhere. There are no Sinkhorn or SoftImpute curves: Sinkhorn's batch loss is mostly sampling noise on two tables (its budget was chosen by hide-and-check), and SoftImpute's final fit takes only 1 to 8 steps.
