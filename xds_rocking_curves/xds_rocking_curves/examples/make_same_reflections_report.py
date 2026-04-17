#!/usr/bin/env python3
"""Build a PDF report for same-reflection rocking-curve comparisons.

The report includes:
- A summary page with run context and top reflections per dataset.
- One page per HKL with per-dataset overlays across datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


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
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="LTA_same_reflections root folder.")
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=None,
        help="Output PDF path (default: <root>/same_reflections_overlay_report_raw.pdf).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subdirectories to include.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("phi", "frame"),
        default="phi",
        help="X-axis for overlays.",
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "normalized"),
        default="raw",
        help="Report raw I_fit or normalized I_fit_norm overlays.",
    )
    return parser.parse_args()


def _hkl_dir_name(h: int, k: int, l: int) -> str:
    return f"hkl_{h}_{k}_{l}"


def _curve_path(root: Path, dataset: str, h: int, k: int, l: int) -> Path:
    return root / dataset / _hkl_dir_name(h, k, l) / "rocking_curve.csv"


def _label(dataset: str, curve_df: pd.DataFrame) -> str:
    if "thickness_nm" in curve_df.columns and not curve_df.empty:
        return f"{dataset} ({curve_df['thickness_nm'].iloc[0]:g} nm)"
    return dataset


def _add_summary_page(
    pdf: PdfPages,
    summary_df: pd.DataFrame,
    datasets: list[str],
    reflections_df: pd.DataFrame,
    mode: str,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape in inches
    mode_title = "Normalized I_fit" if mode == "normalized" else "Raw I_fit"
    fig.suptitle(f"Same-Reflection Rocking Curves Report ({mode_title})", fontsize=16, y=0.98)

    ax_text = fig.add_axes([0.05, 0.52, 0.9, 0.4])
    ax_text.axis("off")

    lines = [
        f"Total reflections compared: {len(reflections_df)}",
        f"Datasets: {', '.join(datasets)}",
        "Metric shown in plots: I_fit_norm, successful fits only"
        if mode == "normalized"
        else "Metric shown in plots: I_fit (non-normalized), successful fits only",
    ]

    if not summary_df.empty and "n_fit_success" in summary_df.columns and "n_relevant_frames" in summary_df.columns:
        frac = (summary_df["n_fit_success"] / summary_df["n_relevant_frames"]).fillna(0.0)
        lines.append(f"Overall median fit success rate: {frac.median():.3f}")

    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=11)

    ax_table = fig.add_axes([0.05, 0.08, 0.9, 0.38])
    ax_table.axis("off")

    table_rows: list[list[str]] = []
    if not summary_df.empty and "median_r2" in summary_df.columns:
        for ds in datasets:
            ds_df = summary_df[summary_df["dataset"] == ds].sort_values("median_r2", ascending=False).head(3)
            for _, row in ds_df.iterrows():
                table_rows.append(
                    [
                        ds,
                        f"({int(row['h'])}, {int(row['k'])}, {int(row['l'])})",
                        f"{row['median_r2']:.4f}",
                        f"{row['fit_success_rate']:.3f}",
                    ]
                )

    if table_rows:
        table = ax_table.table(
            cellText=table_rows,
            colLabels=["Dataset", "Top Reflection", "median_r2", "fit_success_rate"],
            loc="upper left",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.25)
        ax_table.set_title("Top 3 Reflections Per Dataset", fontsize=12, loc="left")
    else:
        ax_table.text(0.0, 1.0, "No summary table available.", va="top", fontsize=11)

    pdf.savefig(fig)
    plt.close(fig)


def _add_reflection_page(
    pdf: PdfPages,
    root: Path,
    datasets: list[str],
    x_axis: str,
    mode: str,
    h: int,
    k: int,
    l: int,
    summary_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0.08, 0.16, 0.84, 0.72])

    plotted = 0
    for ds in datasets:
        path = _curve_path(root, ds, h, k, l)
        if not path.exists():
            continue

        curve = pd.read_csv(path)
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

        ax.plot(good[x_col], good["_plot_y"], marker="o", linewidth=1.5, markersize=3.5, label=_label(ds, good))
        plotted += 1

    mode_text = "Normalized" if mode == "normalized" else "Raw"
    ax.set_title(f"{mode_text} Rocking Curve Overlay for ({h}, {k}, {l})", fontsize=14)
    ax.set_xlabel("phi (deg)" if x_axis == "phi" else "frame")
    ax.set_ylabel("I_fit_norm" if mode == "normalized" else "I_fit (non-normalized)")
    ax.grid(alpha=0.25)
    if plotted > 0:
        ax.legend(loc="best", fontsize=9)

    ax_note = fig.add_axes([0.08, 0.05, 0.84, 0.08])
    ax_note.axis("off")
    if not summary_df.empty:
        rows = summary_df[(summary_df["h"] == h) & (summary_df["k"] == k) & (summary_df["l"] == l)]
        rows = rows.sort_values("thickness_nm")
        if not rows.empty:
            note = " | ".join(
                f"{r.dataset}: median_r2={r.median_r2:.4f}, success={r.fit_success_rate:.3f}"
                for r in rows.itertuples()
            )
            ax_note.text(0.0, 0.7, note, fontsize=9)

    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root
    out_pdf = args.out_pdf or (root / f"same_reflections_overlay_report_{args.mode}.pdf")

    reflections_path = root / "reflections_used.csv"
    summary_path = root / "same_reflections_summary.csv"

    if not reflections_path.exists():
        raise FileNotFoundError(f"Missing reflections file: {reflections_path}")
    reflections = pd.read_csv(reflections_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    required = {"h", "k", "l"}
    if not required.issubset(reflections.columns):
        raise ValueError(f"Expected columns in reflections_used.csv: {sorted(required)}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        _add_summary_page(pdf, summary, args.datasets, reflections, args.mode)
        for _, row in reflections.iterrows():
            _add_reflection_page(
                pdf=pdf,
                root=root,
                datasets=args.datasets,
                x_axis=args.x_axis,
                mode=args.mode,
                h=int(row["h"]),
                k=int(row["k"]),
                l=int(row["l"]),
                summary_df=summary,
            )

    print(f"Report written: {out_pdf.resolve()}")
    print(f"Pages: {len(reflections) + 1}")


if __name__ == "__main__":
    main()
