"""Plot reflections with the largest intensity variation across thickness."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_thickness_from_name(path: Path) -> float | None:
    m = re.search(r"_(\d+)nm\.csv$", path.name)
    if m is None:
        return None
    return float(m.group(1))


def _load_pred_tables(pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched pattern: {pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if "intensity" not in df.columns:
            continue
        if "thickness_nm" not in df.columns:
            t = _parse_thickness_from_name(f)
            if t is None:
                raise RuntimeError(f"Could not infer thickness from file name: {f}")
            df["thickness_nm"] = t
        df["source_file"] = str(f)
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid prediction tables with 'intensity' were found.")

    merged = pd.concat(rows, ignore_index=True)
    merged["hkl"] = list(zip(merged["h"].astype(int), merged["k"].astype(int), merged["l"].astype(int)))
    return merged


def _rank_varying_reflections(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("hkl", as_index=False)
        .agg(
            h=("h", "first"),
            k=("k", "first"),
            l=("l", "first"),
            n_thickness=("thickness_nm", "nunique"),
            intensity_min=("intensity", "min"),
            intensity_max=("intensity", "max"),
            intensity_mean=("intensity", "mean"),
            intensity_std=("intensity", "std"),
            mean_S_comb=("S_comb", "mean"),
        )
        .copy()
    )

    stats["intensity_std"] = stats["intensity_std"].fillna(0.0)
    stats["range"] = stats["intensity_max"] - stats["intensity_min"]
    stats["cv"] = stats["intensity_std"] / np.maximum(stats["intensity_mean"], 1e-12)

    # Prioritize broad and practically relevant intensity variation.
    stats["variation_score"] = stats["range"] * np.sqrt(np.maximum(stats["intensity_mean"], 1e-16))
    stats = stats.sort_values(["variation_score", "range"], ascending=False).reset_index(drop=True)
    return stats


def _plot_top_curves(
    long_df: pd.DataFrame,
    ranked: pd.DataFrame,
    top_n: int,
    output_png: Path,
) -> None:
    top = ranked.head(top_n)
    n = len(top)
    if n == 0:
        raise RuntimeError("No reflections available for plotting.")

    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    for ax in axes:
        ax.grid(alpha=0.25)

    for i, row in enumerate(top.itertuples(index=False)):
        ax = axes[i]
        mask = (
            (long_df["h"].astype(int) == int(row.h))
            & (long_df["k"].astype(int) == int(row.k))
            & (long_df["l"].astype(int) == int(row.l))
        )
        g = long_df.loc[mask].sort_values("thickness_nm")

        ax.plot(g["thickness_nm"], g["intensity"], marker="o", linewidth=2.0)
        ax.set_title(
            f"({int(row.h)} {int(row.k)} {int(row.l)})\n"
            f"range={row.range:.3g}, cv={row.cv:.3g}",
            fontsize=9,
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Thickness (nm)")
    fig.supylabel("Predicted intensity")
    fig.suptitle("Reflections with strongest thickness variation", y=1.01)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pattern",
        default="analysis_output/lta_frame0050_pred_t*_*.csv",
        help="Glob pattern for per-thickness prediction tables",
    )
    p.add_argument("--top-n", type=int, default=12, help="Number of reflections to plot")
    p.add_argument(
        "--output-prefix",
        default="analysis_output/lta_frame0050_top_varying",
        help="Prefix for output files",
    )
    p.add_argument(
        "--min-thickness-count",
        type=int,
        default=4,
        help="Minimum number of thickness points required per reflection",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    long_df = _load_pred_tables(args.pattern)
    ranked = _rank_varying_reflections(long_df)
    ranked = ranked[ranked["n_thickness"] >= int(args.min_thickness_count)].copy()

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    ranked_csv = prefix.with_name(prefix.name + "_ranked.csv")
    plot_png = prefix.with_name(prefix.name + "_curves.png")

    ranked.to_csv(ranked_csv, index=False)
    _plot_top_curves(long_df, ranked, top_n=int(args.top_n), output_png=plot_png)

    print(f"rows_total={len(long_df)}")
    print(f"reflections_ranked={len(ranked)}")
    print(f"ranked_csv={ranked_csv}")
    print(f"plot_png={plot_png}")


if __name__ == "__main__":
    main()
