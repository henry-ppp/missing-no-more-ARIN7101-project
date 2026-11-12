# Datasets for the project

The experiment design: Take a complete table, hide some of its values, let each model fill them back in, and measure how far the guesses are from the real values. We compare generative models (e.g. VAE, MIWAE, not-MIWAE and GAIN, etc) with random-forest methods that are known to be hard to beat (MissForest, and MICE with random forests, which we'll call MICE-RF), and KNN.

So we mostly need complete tables, where every true value is known. One of the suggested datasets, FRED-MD, also has real gaps whose true values were published later, which lets us test the models on missing data that nobody simulated.

## How I picked them

We wanted data that would survive the question "why this dataset?" in the Q&A. Every suggested dataset:

- comes with a published paper or official record that says where the data came from.
- has a permanent link (a DOI) or an official host, and a licence.
- has already been used in published imputation research, except FRED-MD (see its entry). The papers I checked are GAIN (ICML 2018), MIWAE (ICML 2019), HI-VAE (*Pattern Recognition* 2020), GRAPE (NeurIPS 2020), Muzellec et al. (ICML 2020) and HyperImpute (ICML 2022). I went through each paper's own dataset list rather than trusting someone else's summary.
- was downloaded and checked for duplicate rows, undocumented codes and similar problems. Anything I found is in the remarks.

## 1. Suggested datasets

These are the five suggested datasets. Breast Cancer, Wine and Yeast go from easy to hard. The Wine datasets may be ignored if time does not allow. In our test run (section 3) the best model beat the column average by about 57% on Breast Cancer, 21% on Wine and only 11% on Yeast, so together they show where the generative models help and where nothing does much. Credit card default adds a mix of numeric and categorical columns and plenty of rows. FRED-MD adds the real gaps. Those two are also our finance data. I already have MissForest results for four of the five (all but Yeast), so the baseline is mostly done.

| # | Dataset | Rows × columns* |
|---|---|---|
| 1 | Wine Quality (white) | 3,961 × 12 |
| 2 | Breast Cancer Wisconsin (Diagnostic) | 569 × 31 |
| 3 | Yeast | 1,453 × 9 |
| 4 | Credit card default (Taiwan) | 29,965 × 24 |
| 5 | FRED-MD | 334 × 111 |
| 6 | Energy Efficiency Dataset | 768 x 8 |

\*After the cleaning described in the remarks. The experiments use a random sample of at most 3,000 of these rows.

### 1. Wine Quality (white)

Lab measurements for 4,898 Portuguese white wines (acidity, sugar, sulphates, alcohol and so on) plus a taste score from 0 to 10 (Cortez et al., 2009). There are eleven numeric columns that are moderately related. It's the most common dataset in imputation papers: MIWAE, Muzellec et al. and HyperImpute use the white wines, and HI-VAE and GRAPE use the wine data too.

- Link: https://doi.org/10.24432/C56S3T
- Remark: 937 rows (19%) are exact duplicates. I drop them, because otherwise a model can fill in a value by copying the row's twin and look better than it is.

### 2. Breast Cancer Wisconsin (Diagnostic)

30 measurements of cell nuclei from images of 569 breast lumps, plus the diagnosis (benign or malignant) (Street, Wolberg & Mangasarian, 1993). Radius, perimeter and area all move together, so every model should clearly beat filling in the column average. If one doesn't, something is wrong with our code. Used by Muzellec et al. GAIN and MIWAE also report a "Breast" dataset, but they don't say which version.

- Link: https://doi.org/10.24432/C5DW2B (also built into scikit-learn as `load_breast_cancer`)
- Remark: clean as published.

### 3. Yeast

1,484 yeast proteins, each described by 8 scores computed from its amino-acid sequence, plus where in the cell it sits (10 locations) (Nakai & Kanehisa, 1992). The columns barely predict each other, so this is the hard case. Used by MIWAE and Muzellec et al. In the MIWAE paper, MissForest did worse here than filling in the column average (error 1.41, against 0.96 for MIWAE). It's the clearest published case of a generative model beating MissForest, so I should check it.

- Link: https://doi.org/10.24432/C5KG68
- Remarks: I drop the protein-name column and 31 duplicate rows. Two columns are almost constant: `erl` is 0.5 in 99% of rows and `pox` is 0 in 99%. I treat both as categories. As numbers, their tiny spread blows up the error, which probably explains MissForest's poor result in the MIWAE paper.

### 4. Credit card default (Taiwan)

30,000 credit-card customers of a Taiwanese bank in 2005: credit limit, age, sex, education, marital status, six months of repayment history, bill and payment amounts, and whether they defaulted the next month (Yeh & Lien, 2009). It's a mix of numeric and categorical columns, and one of our two finance datasets. Used by GAIN and HI-VAE.

- Link: https://doi.org/10.24432/C55S3H
- Remarks: EDUCATION uses the codes 0, 5 and 6 (345 rows) and MARRIAGE uses 0 (54 rows), and none of these codes are documented. I merge them into "other". Bill and payment amounts are very skewed, so it's worth taking logs before training the neural models. I also drop 35 duplicate rows.

### 5. FRED-MD

126 monthly US economic and financial series going back to 1959 (output, jobs, prices, interest rates, credit, the stock market), maintained by the St. Louis Fed (McCracken & Ng, 2016). A new edition comes out every month, and old editions are archived.

We use it in two ways. For simulated gaps, I take 334 months (1992 to 2019) × 111 series that are fully observed after the standard transformations. The real gaps come from series that are published late: the most recent months of each edition have holes that later editions fill in, so I can score each model against the values that were actually published. We've already scored 758 such gaps from 44 editions with MissForest.

- Link: https://www.stlouisfed.org/research/economists/mccracken/fred-databases
- Remarks: the site blocks scripted downloads after a few requests, and I already have the 2015 to 2025 editions saved. Rows are consecutive months, so unlike the other tables they aren't independent of each other. FRED-MD is the only one of the five not used by the imputation papers above. But its source paper fills these same gaps with an EM method, and economists use it widely as a benchmark.

## 2. Optional datasets

These are for when I have time left, want more variety, or need a substitute for one of the five. They went through the same checks. The first five are the ones we'd add first, because each brings something the main five don't. Everything down to Online News has been used in published imputation papers. The last two are finance datasets with a peer-reviewed source, but no peer-reviewed imputation paper uses them yet, so I would have to justify them ourselves.

| Dataset | What it is | Rows × columns* | Remarks |
|---|---|---|---|
| [Adult (Census Income)](https://doi.org/10.24432/C5XW20) | 48,842 adults from the 1994 US census: age, work class, education, marital status, occupation, race, sex, capital gains, weekly hours, country, and whether they earn over $50K (Kohavi, 1996). Used by HI-VAE. | 45,175 × 13 | Mostly categorical, with up to 41 categories. It has real gaps that I can't score but can describe: 7.4% of people are missing occupation, work class or country, and they earn over $50K about half as often (13% against 25%). Drop `fnlwgt` (a survey weight) and `education` (a copy of `education_num`). |
| [California Housing](https://lib.stat.cmu.edu/datasets/houses.zip) | 20,640 neighbourhoods from the 1990 California census: income, house age, rooms, population, location and median house value (Pace & Barry, 1997). Used by Muzellec et al. and HyperImpute. | 20,640 × 9 | House values are capped at $500,001 and house age at 52 years, a real case of high values being hidden, like the MNAR I simulate. Also in scikit-learn as `fetch_california_housing`. The archive states no licence. |
| [Concrete Compressive Strength](https://doi.org/10.24432/C5PK67) | 1,030 lab-tested concrete mixes: the amounts of seven ingredients, the sample's age and its strength (Yeh, 1998). Used by MIWAE, GRAPE, Muzellec et al. and HyperImpute. | 1,005 × 9 | 25 duplicate rows, dropped. A zero means the ingredient wasn't used, so it's a real value, not a missing one. |
| [Letter Recognition](https://doi.org/10.24432/C5ZP40) | 20,000 printed capital letters in 20 fonts, each reduced to 16 whole-number features from 0 to 15, plus the letter (Frey & Slate, 1991). Used by GAIN, HI-VAE and HyperImpute. | 18,668 × 17 | Large, with discrete values and a 26-category letter column. 1,332 duplicate rows (6.7%), dropped. |
| [Spambase](https://doi.org/10.24432/C53G6X) | 4,601 emails collected at Hewlett-Packard Labs in 1999: how often 48 words and 6 characters appear, three measures of capital-letter runs, and a spam label. Used by GAIN, HI-VAE and HyperImpute. | 4,210 × 58 | 77% of the cells are zero. No published paper describes how it was collected, and we'd have to say so. 391 duplicate rows, dropped. The UCI page says it has missing values, but the file has none. |
| [Airfoil Self-Noise](https://doi.org/10.24432/C5VW2C) | NASA wind-tunnel tests of aerofoil blade sections: frequency, angle, chord length, wind speed, thickness and noise level (Brooks, Pope & Marcolini, 1989). Used by Muzellec et al. and HyperImpute. | 1,503 × 6 | The inputs follow the experiment's design (only 4 wind speeds and 6 chord lengths), so they take few distinct values. |
| [Wine Quality (red)](https://doi.org/10.24432/C56S3T) | 1,599 red wines with the same lab measurements as the white set (Cortez et al., 2009). Used by MIWAE, Muzellec et al. and HyperImpute. | 1,359 × 12 | 240 duplicate rows (15%), dropped. Comes in the same download as the white wines. |
| [Combined Cycle Power Plant](https://doi.org/10.24432/C5002N) | 9,568 hourly readings from a gas-and-steam power plant, 2006 to 2011: temperature, pressure, humidity, exhaust vacuum and power output (Tüfekci, 2014). Used by GRAPE. | 9,527 × 5 | The download holds 5 sheets that are shuffled copies of the same data, so use only one. 41 duplicate rows. |
| [Yacht Hydrodynamics](https://doi.org/10.24432/C5XG7R) | 308 towing-tank tests of yacht hull models at Delft: 6 hull-shape and speed settings and the measured resistance (Gerritsma et al., 1981). Used by GRAPE and Muzellec et al. | 308 × 7 | Very small, and each input takes only 5 to 17 distinct values. |
| [Parkinsons](https://doi.org/10.24432/C59C74) | 22 voice measurements from recordings of people with and without Parkinson's disease, plus the diagnosis (Little et al., 2007). Used in the original MissForest paper and by Muzellec et al. | 195 × 23 | 195 recordings from only 31 people (about 6 each), so the rows aren't independent. |
| [Ionosphere](https://doi.org/10.24432/C5W01B) | Radar returns from the upper atmosphere: 34 signal features, labelled good or bad (Sigillito et al., 1989). Used by Muzellec et al. and HyperImpute. | 350 × 34 | Column 2 is always 0, so drop it. Column 1 is 0/1. One duplicate row. |
| [Blood Transfusion Service Center](https://doi.org/10.24432/C5GS39) | 748 donors at a blood service in Hsinchu, Taiwan: months since the last donation, number of donations, total blood given, months since the first donation, and whether they gave blood in March 2007 (Yeh, Yang & Ting, 2009). Used by Muzellec et al. and HyperImpute. | 748 × 4 | Total blood given is exactly 250 × the number of donations, so drop it or it becomes trivial to impute. That leaves only 3 numeric columns. 29% of rows repeat, which is normal for a few small whole numbers, so I keep them. |
| [Libras Movement](https://doi.org/10.24432/C5GC82) | 15 hand movements from Brazilian Sign Language, each traced as 45 (x, y) points (Dias, Peres & Bíscaro, 2009). Used by Muzellec et al. and HyperImpute. | 330 × 91 | 30 duplicate rows, dropped. Very easy to impute, because neighbouring points along a movement are almost the same (MICE-RF error 0.11 in our test run). |
| [Musk (version 1)](https://doi.org/10.24432/C5ZK5B) | 166 shape features for different 3-D poses of molecules, labelled musk or not (Dietterich, Lathrop & Lozano-Pérez, 1997). Used in the original MissForest paper. | 476 × 167 | 476 poses of only 92 molecules, so poses of the same molecule are related. |
| [Online News Popularity](https://doi.org/10.24432/C5NS3V) | 39,644 Mashable articles from two years: word counts, topic, weekday, sentiment and the number of shares (Fernandes, Vinagre & Cortez, 2015). Used by GAIN. | 39,644 × 59 | Drop `url` and `timedelta`, which aren't features. 1,181 articles have no text at all, one row has a word-rate value of 701 where the scale is 0 to 1, and shares are extremely skewed (median 1,400, maximum 843,300). |
| [Polish Companies Bankruptcy](https://doi.org/10.24432/C5F600) | 64 financial ratios for Polish manufacturing firms, 2000 to 2013, and whether they went bankrupt, in five files by forecast horizon (Zięba, Tomczak & Tomczak, 2016). | 8,757 × 63 (3-year file) | Real gaps: 54% of firms in the 3-year file are missing at least one ratio, and those firms went bankrupt three times as often (6.9% against 2.2%). For simulated tests, drop the two most-incomplete ratios and keep the complete rows. The ratios have extreme values, so clip them first. |
| [Taiwanese Bankruptcy Prediction](https://doi.org/10.24432/C5004D) | 95 financial ratios for 6,819 companies listed in Taiwan, 1999 to 2009, from the Taiwan Economic Journal, with 220 bankruptcies. The UCI page names no paper; the data is usually cited to [Liang et al. (2016)](https://doi.org/10.1016/j.ejor.2016.01.012). | 6,819 × 95 | Most ratios are scaled 0 to 1, but 24 columns hold raw values up to 10^10, so clip or log them. One column is constant. |

\*After the cleaning in the remarks. As before, experiments would use a random sample of at most 3,000 rows.

What I left out:
- Kaggle-only datasets, which need an account and come with competition rules.
- Simulated data: Energy Efficiency, Kin8nm and Naval Propulsion.
- Banknote Authentication and Protein Tertiary Structure. No published paper describes either one, and each appears in only one imputation paper.
- South German Credit, which I used in the first MissForest test. Its corrected version is documented only in a technical report, and no imputation paper uses it. Credit card default covers the same ground with a better record.
