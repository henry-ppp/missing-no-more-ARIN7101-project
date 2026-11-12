# Project datasets

This is the dataset list for the project upon discussion.

Sizes are rows × columns after the cleaning in that note. Experiments use a random sample of at most 3,000 of those rows.

| # | Dataset | Size | Source |
| --- | --- | --- | --- |
| 1 | Wine Quality (white) | 3,961 × 12 | https://doi.org/10.24432/C56S3T |
| 2 | Breast Cancer Wisconsin (Diagnostic) | 569 × 31 | https://doi.org/10.24432/C5DW2B |
| 3 | Yeast | 1,453 × 9 | https://doi.org/10.24432/C5KG68 |
| 4 | Credit card default (Taiwan) | 29,965 × 24 | https://doi.org/10.24432/C55S3H |
| 5 | FRED-MD | 334 × 111 | https://www.stlouisfed.org/research/economists/mccracken/fred-databases |
| 6 | Energy Efficiency | 768 × 10 | https://archive.ics.uci.edu/dataset/242/energy+efficiency |

1. **Wine Quality (white).** Lab measurements and a taste score for Portuguese white wines (Cortez et al., 2009). 937 exact duplicate rows are dropped.
2. **Breast Cancer Wisconsin (Diagnostic).** 30 cell-nucleus measurements and the diagnosis (Street, Wolberg & Mangasarian, 1993). Also in scikit-learn as `load_breast_cancer`. Used as published.
3. **Yeast.** Eight sequence scores and the cell location (Nakai & Kanehisa, 1992). The protein-name column and 31 duplicate rows are dropped.
4. **Credit card default (Taiwan).** 30,000 Taiwanese credit-card customers in 2005 (Yeh & Lien, 2009). 35 duplicate rows are dropped.
5. **FRED-MD.** Monthly US economic series from the St. Louis Fed (McCracken & Ng, 2016). The simulated-gap table is 334 months (1992–2019) by 111 fully observed series.
6. **Energy Efficiency.** 768 simulated buildings (Tsanas & Xifara, 2012). Eight building features plus heating load and cooling load, so 10 columns. No missing values. DOI: https://doi.org/10.24432/C51307. UCI id 242.
