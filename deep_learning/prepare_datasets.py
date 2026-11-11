"""Prepare public datasets used by the deep-learning benchmark.

Data files are written to ``data/`` and are ignored by git. The script focuses
on datasets from the project design that are easy to fetch and standardise with
only pandas/numpy:

- Breast Cancer Wisconsin Diagnostic
- Wine Quality white
- Yeast

Other datasets in the design, such as credit-card default and FRED-MD, often
need manually saved files or Excel readers, so the benchmark loader supports them
once their CSV files are placed in ``data/``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd


UCI_WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
UCI_WINE_WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
UCI_YEAST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"


WDBC_COLUMNS = [
    "ID",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


YEAST_COLUMNS = [
    "sequence_name",
    "mcg",
    "gvh",
    "alm",
    "mit",
    "erl",
    "pox",
    "vac",
    "nuc",
    "localization_site",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


def prepare_breast_cancer(data_dir: Path) -> Path:
    raw_path = data_dir / "wdbc.data"
    output_path = data_dir / "breast_cancer.csv"
    download(UCI_WDBC_URL, raw_path)
    data = pd.read_csv(raw_path, header=None, names=WDBC_COLUMNS)
    data = data.drop_duplicates()
    data.to_csv(output_path, index=False)
    return output_path


def prepare_wine_white(data_dir: Path) -> Path:
    output_path = data_dir / "winequality-white.csv"
    download(UCI_WINE_WHITE_URL, output_path)
    # Normalise line endings / verify readability.
    data = pd.read_csv(output_path, sep=";")
    data = data.drop_duplicates()
    data.to_csv(output_path, index=False, sep=";")
    return output_path


def prepare_yeast(data_dir: Path) -> Path:
    raw_path = data_dir / "yeast.data"
    output_path = data_dir / "yeast.csv"
    download(UCI_YEAST_URL, raw_path)
    data = pd.read_csv(raw_path, sep=r"\s+", header=None, names=YEAST_COLUMNS)
    data = data.drop_duplicates()
    data.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where to write prepared CSV files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["breast_cancer", "wine_white", "yeast"],
        choices=["breast_cancer", "wine_white", "yeast"],
        help="Datasets to prepare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparers = {
        "breast_cancer": prepare_breast_cancer,
        "wine_white": prepare_wine_white,
        "yeast": prepare_yeast,
    }
    for dataset in args.datasets:
        output = preparers[dataset](args.data_dir)
        print(f"Prepared {dataset}: {output}")


if __name__ == "__main__":
    main()
