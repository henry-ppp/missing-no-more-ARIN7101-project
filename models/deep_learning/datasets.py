"""Dataset choices and loading rules for the deep-learning experiments.

The choices mirror ``data/DATASETS_MEETING copy.md``. Data files themselves stay
under ``data/`` and are ignored by git.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """How to load one project dataset for numeric imputation experiments."""

    name: str
    path: str
    why: str
    target_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    max_rows: int | None = 3000
    deduplicate: bool = True
    notes: str = ""
    csv_kwargs: dict[str, object] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        path="data/breast_cancer.csv",
        why="Small/easy benchmark with strongly correlated numeric features; useful sanity check.",
        target_columns=("diagnosis", "target"),
        drop_columns=("ID", "id"),
        notes="Use the 30 numeric cell-nuclei measurements; do not impute the diagnosis label or ID.",
    ),
    "wine_white": DatasetSpec(
        name="wine_white",
        path="data/winequality-white.csv",
        why="Classic imputation benchmark with moderately related numeric chemistry columns.",
        notes="Use all 12 numeric columns, including quality, and drop exact duplicate rows.",
        csv_kwargs={"sep": ";"},
    ),
    "yeast": DatasetSpec(
        name="yeast",
        path="data/yeast.csv",
        why="Hard benchmark where columns weakly predict each other; important for testing whether DL helps.",
        target_columns=("class", "localization_site"),
        drop_columns=("sequence_name", "name"),
        notes="Drop protein-name column and duplicate rows; keep numeric score columns only.",
    ),
    "credit_card_default": DatasetSpec(
        name="credit_card_default",
        path="data/credit_card_default.csv",
        why="Larger finance dataset with numeric and coded categorical columns.",
        drop_columns=("ID",),
        notes="Use numeric/coded columns, including the default indicator; later work can add explicit categorical handling.",
    ),
    "fred_md": DatasetSpec(
        name="fred_md",
        path="data/fred_md_complete.csv",
        why="Finance/economics dataset; includes realistic macro-financial dependence structure.",
        target_columns=("sasdate", "date"),
        notes="Use the complete 1992-2019 transformed panel for simulated MCAR gaps.",
        max_rows=None,
    ),
    "energy_efficiency": DatasetSpec(
        name="energy_efficiency",
        path="data/energy_efficiency.csv",
        why="Small complete numeric dataset recently added to the project; quick extra robustness check.",
        target_columns=("Y1", "Y2", "Heating Load", "Cooling Load"),
        notes="Use eight input features if target load columns are present.",
        max_rows=None,
    ),
}


RECOMMENDED_DATASETS: tuple[str, ...] = (
    "breast_cancer",
    "wine_white",
    "yeast",
    "credit_card_default",
    "fred_md",
    "energy_efficiency",
)


def available_datasets() -> pd.DataFrame:
    """Return a human-readable table of dataset choices."""

    return pd.DataFrame(
        [
            {
                "dataset": spec.name,
                "expected_path": spec.path,
                "why": spec.why,
                "notes": spec.notes,
            }
            for spec in DATASETS.values()
        ]
    )


def load_dataset(
    name: str,
    data_dir: Path = Path("data"),
    max_rows: int | None | str = "spec",
    seed: int = 0,
) -> pd.DataFrame:
    """Load and clean one configured dataset.

    The loader intentionally keeps only numeric columns because the shared RMSE
    and MAE metrics expect numeric values. Target/label/ID/date columns are
    dropped before the numeric-only selection.
    """

    if name not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; choose one of: {valid}")

    spec = DATASETS[name]
    path = Path(spec.path)
    if not path.is_absolute():
        path = data_dir / Path(spec.path).name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {name!r} at {path}. Put the raw/cleaned file in data/ or pass --data-dir."
        )

    data = pd.read_csv(path, **spec.csv_kwargs)
    data.columns = [str(column).strip() for column in data.columns]

    columns_to_drop = [
        column
        for column in (*spec.drop_columns, *spec.target_columns)
        if column in data.columns
    ]
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)

    numeric = data.select_dtypes(include="number")
    if spec.deduplicate:
        numeric = numeric.drop_duplicates()
    numeric = numeric.dropna(axis=0, how="any")

    row_limit = spec.max_rows if max_rows == "spec" else max_rows
    if row_limit is not None and len(numeric) > int(row_limit):
        numeric = numeric.sample(n=int(row_limit), random_state=seed).sort_index()

    if numeric.empty:
        raise ValueError(f"{name!r} has no complete numeric rows after cleaning")
    return numeric.astype(float)
