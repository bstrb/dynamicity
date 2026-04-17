#!/usr/bin/env python3
"""Create per-reflection overlay plots across LTA thickness datasets.

This script reads the already generated outputs under:
  real_data_output/LTA_same_reflections/
and creates one plot per HKL with all available datasets overlaid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT = Path("real_data_output/LTA_same_reflections")
DEFAULT_DATASETS = ("LTA_t1", "LTA_t2", "LTA_t3", "LTA_t4")


def _resolve_y(curve: pd.DataFrame, mode: str) -> tuple[str, pd.Series] | None:
    if mode == "normalized":
        if "I_fit_norm" in curve.columns:
            return "I_fit_norm", curve["I_fit_norm"]
        if "I_fit" in curve.columns:
            max_i = curve["I_fit"].max()
            if pd.notna(max_i) and max_i > 0:
                return "I_fit_norm", curve["I_fit"] / max_i
            return "I_fit_norm", pd.Series([float("nan")] * len(curve), index=curve.index)
        return None
    if "I_fit" in curve.columns:
        return "I_fit", curve["I_fit"]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing LTA_same_reflections outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for overlay plots (default: <root>/overlay_plots_raw).",
    )
    parser.add_argument(
        "--x-axis",
        choices=("phi", "frame"),
        default="phi",
        help="X-axis for plots: phi (degrees) or frame number.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subdirectories to include.",
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "normalized"),
        default="raw",
        help="Plot raw I_fit or normalized I_fit_norm.",
    )
    return parser.parse_args()


def _label_with_thickness(dataset: str, curve_df: pd.DataFrame) -> str:
    if "thickness_nm" not in curve_df.columns or curve_df.empty:
        return dataset
    thickness = curve_df["thickness_nm"].iloc[0]
    return f"{dataset} ({thickness:g} nm)"


def _plot_single_hkl(
    h: int,
    k: int,
    l: int,
    root: Path,
    out_dir: Path,
    datasets: list[str],
    x_axis: str,
    mode: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    plotted = 0

    for dataset in datasets:
        curve_path = root / dataset / f"hkl_{h}_{k}_{l}" / "rocking_curve.csv"
        if not curve_path.exists():
            continue

        curve = pd.read_csv(curve_path)
        if curve.empty:
            continue

        good = curve[curve["fit_success"] == True].copy()  # noqa: E712
        if good.empty:
            continue

        y_resolved = _resolve_y(good, mode)
        if y_resolved is None:
            continue
        _, y_values = y_resolved
        good["_plot_y"] = y_values

        x_col = "phi_deg" if x_axis == "phi" and "phi_deg" in good.columns else "frame"
        good = good.sort_values(x_col)
        label = _label_with_thickness(dataset, good)

        ax.plot(good[x_col], good["_plot_y"], marker="o", linewidth=1.5, markersize=3.8, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    mode_text = "Normalized I_fit" if mode == "normalized" else "Raw I_fit"
    ax.set_title(f"Rocking Curve Overlay ({mode_text}) for ({h}, {k}, {l})")
    ax.set_xlabel("phi (deg)" if x_axis == "phi" else "frame")
    ax.set_ylabel("I_fit_norm" if mode == "normalized" else "I_fit (non-normalized)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / f"hkl_{h}_{k}_{l}_overlay_{mode}_{x_axis}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / f"overlay_plots_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reflections_path = root / "reflections_used.csv"
    if not reflections_path.exists():
        raise FileNotFoundError(f"Missing reflections list: {reflections_path}")

    reflections = pd.read_csv(reflections_path)
    required_cols = {"h", "k", "l"}
    if not required_cols.issubset(reflections.columns):
        raise ValueError(f"reflections_used.csv must contain columns: {sorted(required_cols)}")

    plot_index_rows: list[dict[str, str]] = []

    for _, row in reflections.iterrows():
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        out_path = _plot_single_hkl(
            h=h,
            k=k,
            l=l,
            root=root,
            out_dir=out_dir,
            datasets=args.datasets,
            x_axis=args.x_axis,
            mode=args.mode,
        )
        if out_path is not None:
            plot_index_rows.append({"h": h, "k": k, "l": l, "plot_path": str(out_path.resolve())})

    index_df = pd.DataFrame(plot_index_rows)
    index_path = out_dir / "overlay_plot_index.csv"
    index_df.to_csv(index_path, index=False)

    print(f"Saved {len(index_df)} overlay plots to: {out_dir.resolve()}")
    print(f"Plot index: {index_path.resolve()}")


if __name__ == "__main__":
    main()
