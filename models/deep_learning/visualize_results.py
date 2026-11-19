"""Generate visualizations for models/deep_learning results.

This script registers the project-specific deep-learning model names at runtime
instead of adding them to shared_visualization's global defaults. That keeps the
shared visualization tests and palette rules intact while still producing stable
colours for this project's figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from shared_visualization import (  # noqa: E402
    accuracy_vs_distribution,
    critical_difference,
    load_results,
    missing_rate_curves,
    rank_table,
    scoreboard,
    write_tables,
)
from shared_visualization.style import PALETTE, register_model, reset_models  # noqa: E402


MODEL_STYLES = {
    "dl_autoencoder": ("DL autoencoder", PALETTE[0], "^"),
    "dl_vae": ("DL VAE", PALETTE[1], "v"),
    "dl_mida": ("DL MIDA", PALETTE[2], "X"),
    "dl_gain_initial": ("DL GAIN initial", PALETTE[3], "<"),
    "dl_gain_optimized": ("DL GAIN optimized", PALETTE[4], ">"),
    "dl_miwae_initial": ("DL MIWAE initial", PALETTE[5], "P"),
    "dl_miwae_optimized": ("DL MIWAE optimized", PALETTE[6], "h"),
}


def register_deep_learning_models() -> None:
    reset_models()
    for name, (label, colour, marker) in MODEL_STYLES.items():
        register_model(name, label=label, family="deep learning", colour=colour, marker=marker)


def save_plot(plot, path_prefix: Path, formats: tuple[str, ...]) -> list[Path]:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    saved = plot.save(path_prefix, formats=formats)
    plot.close()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        default="results/deep_learning/evaluation/visualization_all_runs/scores.csv",
        help="Tidy scores CSV generated from deep-learning summaries.",
    )
    parser.add_argument(
        "--pairs",
        default="results/deep_learning/evaluation/visualization_all_runs/pairs.csv",
        help="Tidy pairs CSV generated from deep-learning summaries.",
    )
    parser.add_argument("--out", default="figures/deep_learning", help="Output directory for figures and tables.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], help="Figure formats.")
    args = parser.parse_args()

    register_deep_learning_models()
    results = load_results(args.scores, args.pairs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = list(write_tables(results, out / "tables"))
    formats = tuple(args.formats)

    for mode, seed in results.settings("mode", "seed"):
        part = results.select(mode=mode, seed=seed)
        tag = f"{mode}_seed_{seed}"
        for rate in part.settings("rate"):
            rate_tag = f"rate_{rate:g}"
            saved.extend(
                save_plot(
                    scoreboard(part, metrics=("relative_improvement", "r2"), rate=rate),
                    out / f"scoreboard_{tag}_{rate_tag}",
                    formats,
                )
            )
        try:
            saved.extend(
                save_plot(
                    accuracy_vs_distribution(part),
                    out / f"accuracy_vs_distribution_{tag}",
                    formats,
                )
            )
        except ValueError as exc:
            print(f"skipped accuracy_vs_distribution_{tag}: {exc}")
        try:
            saved.extend(
                save_plot(critical_difference(part), out / f"ranking_{tag}", formats)
            )
        except ValueError as exc:
            print(f"skipped ranking_{tag}: {exc}")
        try:
            table = rank_table(part)
            target = out / f"rank_table_{tag}.csv"
            table.to_csv(target, index=False)
            saved.append(target)
        except ValueError as exc:
            print(f"skipped rank_table_{tag}: {exc}")
        try:
            saved.extend(
                save_plot(missing_rate_curves(part), out / f"missing_rate_{tag}", formats)
            )
        except ValueError as exc:
            print(f"skipped missing_rate_{tag}: {exc}")

    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
