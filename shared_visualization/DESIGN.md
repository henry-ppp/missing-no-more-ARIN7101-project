# Shared figures: design

Every model's results go into the same five figures, drawn from saved evaluation reports, with one fixed colour and shape per model. The figures are built in this folder; [README.md](README.md) says how to use them. This note records why each figure looks the way it does, so the team can check the choices before results go on slides.

**Where we are (18 Nov, `dev` at `cab39e3`):** on the shared benchmark only DAE v2 and the mean baseline have predictions, and only at 30% missing. Figures 4 and 5 need more methods and rates before they say anything.

## Rules every figure follows

**One colour and one shape per model, in every chart.** Every score is "improvement over mean imputation", so mean imputation is always 0. It is drawn as a grey line at 0, never as a coloured series. A new model takes the next unused colour, never a made-up one.

| Model | Owner | Family | Colour | Shape |
| --- | --- | --- | --- | --- |
| MissForest | Blith | traditional | blue `#2a78d6` | circle |
| MICE-RF | Blith | traditional | orange `#eb6834` | square |
| kNN | Hamad | traditional | aqua `#1baf7a` | diamond |
| Iterative ridge | Henry | traditional | yellow `#eda100` | pentagon |
| DAE v2 | Eric | deep learning | pink `#e87ba4` | triangle up |
| VAE (if run) | Yicheng, Eric | deep learning | green `#008300` | triangle down |
| GAIN (if run) | Yicheng, Eric | deep learning | violet `#6250d6` | triangle left |
| Mean imputation | shared baseline | | grey `#898781`, the 0 line | |

- **Names, not just colours.** Method names sit on the axis or beside the mark. Aqua, yellow and pink are pale on white (contrast below 3:1), so colour alone can't identify a method.
- **Colour-blind check.** I ran the colours through a colour-blind simulation check, which measures how far apart two colours look (OKLab ΔE; 8 or more is the target):
  - **Rows and lines:** all seven pass when each sits next to its neighbour in the order above. The worst neighbouring pair is yellow and aqua at 9.1.
  - **Scatter:** any two colours can end up side by side, and only the first three stay apart. Orange and green drop to 3.2 for red-green colour blindness. That is why every model also has its own shape, and why the scatter (figure 3) has a colour-and-shape legend.
- **Plain marks.** Thin marks, a solid light grid, one y-axis per chart.
- **Slide-ready output.** Sized for a 16:9 slide at real size: text at least 14 pt, the source note 11 pt. Each figure is saved as PDF/SVG (stays sharp when scaled) and as a 300-dpi PNG, plus a CSV of exactly what it shows.
- **Source note.** A one-line note under each figure gives the data source (UCI; Deshpande for California; CC BY 4.0), any outside method (e.g. scikit-learn's `IterativeImputer`), and the statistical method used. The course requires this split between outside work and ours.
- **One source of numbers.** Numbers come only from the saved reports. The exception is figure 1, which needs the values themselves. Nothing re-computes a score, so slides and reports always agree. Wasserstein scores are shown bias-corrected and the others as plain estimates, as in the table `evaluate` prints.

## The five figures, in the order the talk uses them

### 1. What the distribution score measures (Methods slide)

- **Question:** what is the Wasserstein distance, and why isn't RMSE enough?
- **Form:** pick one column. Plot the cumulative distribution of its true hidden values (dark line) against the imputed values (the model's colour). Use three small panels: mean imputation, one traditional method, one deep method.
- **Why it works:** the shaded area between the two curves *is* the Wasserstein-1 distance for that column (Vallender, 1974), so the audience sees the score itself. Mean imputation shows up as a single vertical jump, because every guess is the same number, and leaves a large area.
- A skewed column, such as a credit-card bill amount, makes the point most clearly.

### 2. Scoreboard (Results slide)

- **Question:** on each dataset, how much better than mean imputation is each method on each headline score, and which differences are real?
- **Form:** each method gets a dot with its 95% interval from evaluate's bootstrap.
  - Grid of panels: datasets go across (4) and the headline scores go down (3: RMSE improvement, Wasserstein improvement, downstream gap closed).
  - Methods are rows in the table order above, traditional above deep learning, and the order never changes between panels.
- **Significance:** a filled dot means "not significantly worse than the best method in this panel"; a hollow dot means significantly worse. This uses evaluate's paired p-values with the Holm correction inside each panel. We mark it because two intervals can overlap even when the paired difference is significant, so readers can't judge this by eye (Schenker & Gentleman, 2001).
- **Why dots and not bars:** position along a shared scale is read most accurately (Cleveland & McGill, 1984).
- Gas turbine has no prediction target, so its downstream panel says "Not available". A panel whose score exists but is undefined (California quick's gap closed, below) says "Not defined".

### 3. Accuracy vs distribution (Results slide; a possible contribution)

- **Question:** if a method is accurate, does it also keep the distribution? In other words, is accuracy a good stand-in for distribution recovery?
- **Form:** one small square panel per dataset, one dot per method per setting (dataset × missing rate).
  - x = MAE improvement over mean, averaged over columns. y = Wasserstein improvement over mean.
  - Both axes put mean imputation at 0 and use the same scale. A thin diagonal line marks y = x.
  - Each model keeps its own colour and shape (triangles are deep learning), with a legend. Dots are also labelled when a panel has four methods or fewer; more labels collide. No intervals here, because the scoreboard already shows them.
  - Each panel prints Kendall's τ between the two scores across methods (mean imputation excluded).
- **Why MAE on the x-axis, not RMSE:**
  - For each column, the Wasserstein distance can never exceed the MAE. Wasserstein is the cheapest way to pair the imputed values with the true values, and the real row-by-row pairing is one of the pairings it considers (Villani, 2009). For mean imputation the two are equal.
  - So the Wasserstein improvement is always at least the MAE improvement, and every dot sits on or above the diagonal. The diagonal is the floor that accuracy guarantees. A dot's height above it shows how much better the method keeps the distribution than its accuracy alone promises.
  - RMSE gives no such floor, and RMSE punishes a few large misses heavily, which moves dots for reasons that have nothing to do with the distribution. On credit card quick, DAE v2 scores 0.12 on RMSE improvement, 0.36 on MAE improvement and 0.56 on Wasserstein improvement.
- **Checked on our data:** the floor held in all 228 columns of the 8 DAE v2 runs. Use the plain Wasserstein value (`estimate` in the JSON), not the bias-corrected one. The floor is exact for the plain value, and the correction can push a dot slightly below the line.
- **What counts as evidence** that accuracy is not enough: dots far above the diagonal, and pairs of methods that swap order between the two axes. A weak correlation does not count, because the floor builds part of the correlation in.
- **Hypothesis to write down before the runs:** random-forest imputers average many trees, which pulls their guesses towards the middle of each column, so their dots sit closer to the floor than the deep models' dots.
- The figure uses the MAE that evaluate saves, which is pooled over all cells rather than averaged by column, so it needs no prediction files and computes no score of its own. On the DAE v2 runs the two versions differ by at most 0.005, so a dot can sit up to about that much below the line. If evaluate later saves a column-averaged MAE improvement as `mae_improvement`, the figure uses it automatically.

### 4. Effect of the missing rate (Results slide)

- **Question:** does the gap between traditional and deep methods change as more values go missing?
- **Form:** one line per method. x = missing rate (10, 20, 30, 50%). y = RMSE improvement in the top row of panels and Wasserstein improvement in the bottom row, with one panel per dataset and each interval drawn as a light band. Only the best traditional and the best deep method are coloured; the rest are grey, so seven lines don't tangle. The GAIN paper (Yoon et al., 2018) reports results across missing rates the same way.
- **Needs rates 0.1, 0.2 and 0.5 prepared** (see below).

### 5. Overall ranking (Conclusions slide)

- **Question:** across all datasets and missing rates, which methods are best overall, and which are tied?
- **Form:** a critical-difference diagram (Demšar, 2006). Each method sits at its average rank on one axis, and a bar joins methods that are not significantly different. Draw those bars from pairwise Wilcoxon signed-rank tests with the Holm correction rather than the older Nemenyi test, as Benavoli et al. (2016) recommend. Use one strip per headline score, and report the Friedman test beside it.
- **Needs enough settings.** Each dataset × missing rate counts once.
  - With only 30% there are 4 settings, and the smallest p-value a Wilcoxon test can give is then 0.125. Holm multiplies the smallest p-value by the number of pairs (21 for seven methods), so about 10 settings are needed before any pair can be significant. With fewer, the figure shows average ranks only and says why; `rank_table` gives the same numbers as a table.
  - With all four rates there are 16 settings. Say on the slide that rates on the same dataset are not independent.

## Optional, for the appendix

- **Per-column heatmap (not built yet).** Rows are methods, columns are features, and the colour shows each column's RMSE improvement: blue better than mean, red worse, grey equal. It would show where a method fails, such as the California `AveOccup` outlier below. No evaluator writes per-column scores yet, and computing them here would duplicate the evaluation code. It can be added once evaluate saves them.
- **Training curves (built).** Loss against epoch, one small panel per dataset. Training and validation lines use two shades of the model's colour, and the best epoch is marked. DAE v2 already writes `training_history.csv` (`epoch, train_masked_mse, validation_masked_mse`) under `results/<model>/training/<dataset>/<mode>/<configuration>/<run_id>/`.
  - **Request to every model owner:** write the same file shape. That means a step column (`epoch`, or the round number for MissForest and iterative ridge) plus one or more named loss or change columns. MissForest's "change between rounds" (its stopping rule) fits directly, so one chart works for every model.

## Left out on purpose

- **Separate RMSE, MAE and R² charts.** They rank the models the same way (see `EVALUATION_V2.md`), so they would be three charts with one message.
- **A runtime chart.** DAE v2 ran on an RTX 4090 GPU, and the random-forest models run on an M2 laptop CPU, so the times can't be compared. A runtime table that names the hardware is the honest alternative.

## Two problems in the current data that affect the figures

**California quick has one extreme cell.** One hidden `AveOccup` value (row `source.csv:19006`) is 1,314 standard deviations from its column mean.
- RMSE: that cell alone makes up 99% of mean imputation's squared error. DAE v2's RMSE improvement is 0.001, with interval [−0.253, 0.001].
- Downstream check: the true table predicts house value *worse* than the mean-imputed table (R² −1.28 against 0.39), so "gap closed" can't be computed. Dropping that one row brings the true table back to 0.55.
- California **full** has neither problem: true 0.60, mean 0.37, and DAE v2 closes 45% [42%, 47%] of the gap.

**Bare prediction paths collide.** Under the new results layout, a bare path such as `.../dae_v2/initial/predictions.csv` is named after its run-ID folder (`initial`). Two models in one evaluate call then clash. Until it's fixed, pass `NAME=PATH`. The fix is one line in `parse_predictions`: name a bare path after the folder above the run ID.

## What we need to decide

1. **Prepare rates 0.1, 0.2 and 0.5?** It's deterministic and uses the committed raw files: `python -m data_preprocessing.prepare_all --mode quick --skip-download --rate 0.1 0.2 0.5` (and `--mode full` if that's the headline). Figures 4 and 5 need it, and figure 3 gets four times as many dots.
2. **Which mode is the headline, quick or full?** Full avoids both California quick problems but costs the most runtime, especially for random forest on superconductivity.
3. **Run evaluate once per setting with every model's predictions together**, so the paired bootstrap covers every pair of models.

`matplotlib` and `scipy`, which these figures use, are now listed in `requirements.txt`.

## References

- Benavoli, A., Corani, G. & Mangili, F. (2016). Should we really use post-hoc tests based on mean-ranks? *Journal of Machine Learning Research*, 17(5), 1–10.
- Cleveland, W. S. & McGill, R. (1984). Graphical perception: theory, experimentation, and application to the development of graphical methods. *Journal of the American Statistical Association*, 79(387), 531–554.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Schenker, N. & Gentleman, J. F. (2001). On judging the significance of differences by examining the overlap between confidence intervals. *The American Statistician*, 55(3), 182–186.
- Vallender, S. S. (1974). Calculation of the Wasserstein distance between probability distributions on the line. *Theory of Probability and Its Applications*, 18(4), 784–786.
- Villani, C. (2009). *Optimal Transport: Old and New*. Springer.
- Yoon, J., Jordon, J. & van der Schaar, M. (2018). GAIN: missing data imputation using generative adversarial nets. *Proceedings of the 35th International Conference on Machine Learning (ICML)*.
