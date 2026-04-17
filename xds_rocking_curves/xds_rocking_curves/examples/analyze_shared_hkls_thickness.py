#!/usr/bin/env python3
"""Analyze thickness trends on HKLs shared across all datasets.

Inputs are produced by summarize_all_reflections_integrate.py.
Outputs include per-HKL trend metrics, ranked stable/variable lists,
summary stats, and a few diagnostic plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_ROOT = Path("real_data_output/LTA_all_reflections_integrate_summary")
DEFAULT_OUT_SUBDIR = "shared_hkl_analysis"
DATASET_ORDER = ["LTA_t1", "LTA_t2", "LTA_t3", "LTA_t4"]
THICKNESS_MAP = {
    "LTA_t1": 100.0,
    "LTA_t2": 200.0,
    "LTA_t3": 350.0,
    "LTA_t4": 600.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def _load_shared_table(input_root: Path) -> pd.DataFrame:
    all_path = input_root / "all_reflections_all_datasets_long.csv"
    overlap_path = input_root / "hkl_dataset_overlap_counts.csv"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing input table: {all_path}")
    if not overlap_path.exists():
        raise FileNotFoundError(f"Missing overlap table: {overlap_path}")

    all_df = pd.read_csv(all_path)
    overlap = pd.read_csv(overlap_path)
    shared = overlap[overlap["n_datasets_present"] == 4][["h", "k", "l"]].copy()

    merged = all_df.merge(shared, on=["h", "k", "l"], how="inner")
    merged["thickness_nm"] = pd.to_numeric(merged["thickness_nm"], errors="coerce")
    merged["isig"] = pd.to_numeric(merged["isig"], errors="coerce")
    merged["I"] = pd.to_numeric(merged["I"], errors="coerce")
    merged["d_spacing_angstrom"] = pd.to_numeric(merged["d_spacing_angstrom"], errors="coerce")
    merged["dataset"] = pd.Categorical(merged["dataset"], DATASET_ORDER, ordered=True)
    return merged


def _slope_and_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def _compute_per_hkl_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    grouped = df.sort_values(["h", "k", "l", "thickness_nm"]).groupby(["h", "k", "l"], sort=False)
    for (h, k, l), g in grouped:
        x = g["thickness_nm"].to_numpy(dtype=float)
        isig = g["isig"].to_numpy(dtype=float)
        inten = g["I"].to_numpy(dtype=float)

        slope_isig, intercept_isig, r2_isig = _slope_and_r2(x, isig)
        slope_i, intercept_i, r2_i = _slope_and_r2(x, inten)

        mean_isig = float(np.nanmean(isig))
        std_isig = float(np.nanstd(isig, ddof=0))
        cv_isig = std_isig / mean_isig if np.isfinite(mean_isig) and mean_isig != 0 else np.nan

        mean_i = float(np.nanmean(inten))
        std_i = float(np.nanstd(inten, ddof=0))
        cv_i = std_i / mean_i if np.isfinite(mean_i) and mean_i != 0 else np.nan

        rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "n_datasets": int(len(g)),
                "mean_isig": mean_isig,
                "std_isig": std_isig,
                "cv_isig": cv_isig,
                "min_isig": float(np.nanmin(isig)),
                "max_isig": float(np.nanmax(isig)),
                "delta_isig_maxmin": float(np.nanmax(isig) - np.nanmin(isig)),
                "slope_isig_per_nm": slope_isig,
                "intercept_isig": intercept_isig,
                "r2_isig_trend": r2_isig,
                "mean_I": mean_i,
                "std_I": std_i,
                "cv_I": cv_i,
                "min_I": float(np.nanmin(inten)),
                "max_I": float(np.nanmax(inten)),
                "delta_I_maxmin": float(np.nanmax(inten) - np.nanmin(inten)),
                "slope_I_per_nm": slope_i,
                "intercept_I": intercept_i,
                "r2_I_trend": r2_i,
                "mean_d_spacing_angstrom": float(np.nanmean(g["d_spacing_angstrom"].to_numpy(dtype=float))),
            }
        )

    out = pd.DataFrame(rows)
    out["abs_slope_isig_per_nm"] = out["slope_isig_per_nm"].abs()
    out["abs_slope_I_per_nm"] = out["slope_I_per_nm"].abs()
    return out


def _make_plots(shared_df: pd.DataFrame, metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ds in DATASET_ORDER:
        vals = shared_df.loc[shared_df["dataset"] == ds, "isig"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=80, alpha=0.4, density=True, label=ds)
    ax.set_xlabel("I/sigma")
    ax.set_ylabel("density")
    ax.set_title("I/sigma Distribution (Shared HKLs)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hist_isig_shared_hkls.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    vals_by_ds = [
        shared_df.loc[shared_df["dataset"] == ds, "isig"].to_numpy(dtype=float)
        for ds in DATASET_ORDER
    ]
    ax.boxplot(vals_by_ds, labels=DATASET_ORDER, showfliers=False)
    ax.set_ylabel("I/sigma")
    ax.set_title("I/sigma by Dataset (Shared HKLs)")
    fig.tight_layout()
    fig.savefig(out_dir / "box_isig_shared_hkls.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(metrics_df["slope_isig_per_nm"].to_numpy(dtype=float), bins=100, alpha=0.8)
    ax.set_xlabel("slope of I/sigma vs thickness (per nm)")
    ax.set_ylabel("count")
    ax.set_title("Trend Slopes Across Shared HKLs")
    fig.tight_layout()
    fig.savefig(out_dir / "hist_slope_isig_per_nm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(metrics_df["mean_isig"], metrics_df["cv_isig"], s=8, alpha=0.35)
    ax.set_xlabel("mean I/sigma across thickness")
    ax.set_ylabel("CV of I/sigma")
    ax.set_title("Stability vs Signal (Shared HKLs)")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_mean_isig_vs_cv_isig.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    out_dir = args.out_dir or (input_root / DEFAULT_OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = _load_shared_table(input_root)
    shared = shared.sort_values(["h", "k", "l", "dataset"]).reset_index(drop=True)
    shared.to_csv(out_dir / "shared_hkls_long.csv", index=False)

    metrics = _compute_per_hkl_metrics(shared)
    metrics = metrics.sort_values(["cv_isig", "abs_slope_isig_per_nm"], ascending=[True, True]).reset_index(drop=True)
    metrics.to_csv(out_dir / "shared_hkl_trend_metrics.csv", index=False)

    stable = metrics.sort_values(["cv_isig", "abs_slope_isig_per_nm"], ascending=[True, True]).head(args.top_n)
    variable = metrics.sort_values(["cv_isig", "abs_slope_isig_per_nm"], ascending=[False, False]).head(args.top_n)
    steep_neg = metrics.sort_values("slope_isig_per_nm", ascending=True).head(args.top_n)
    steep_pos = metrics.sort_values("slope_isig_per_nm", ascending=False).head(args.top_n)

    stable.to_csv(out_dir / f"top_{args.top_n}_most_stable_hkls.csv", index=False)
    variable.to_csv(out_dir / f"top_{args.top_n}_most_variable_hkls.csv", index=False)
    steep_neg.to_csv(out_dir / f"top_{args.top_n}_steepest_negative_isig_trend.csv", index=False)
    steep_pos.to_csv(out_dir / f"top_{args.top_n}_steepest_positive_isig_trend.csv", index=False)

    dataset_summary = (
        shared.groupby("dataset", as_index=False)
        .agg(
            n_reflections=("h", "count"),
            median_isig=("isig", "median"),
            mean_isig=("isig", "mean"),
            q25_isig=("isig", lambda x: float(pd.Series(x).quantile(0.25))),
            q75_isig=("isig", lambda x: float(pd.Series(x).quantile(0.75))),
            p90_isig=("isig", lambda x: float(pd.Series(x).quantile(0.90))),
            median_I=("I", "median"),
            mean_I=("I", "mean"),
        )
        .copy()
    )
    dataset_summary["thickness_nm"] = dataset_summary["dataset"].map(THICKNESS_MAP)
    dataset_summary = dataset_summary.sort_values("thickness_nm")
    dataset_summary.to_csv(out_dir / "shared_hkl_dataset_summary.csv", index=False)

    _make_plots(shared, metrics, out_dir)

    # Compact markdown handoff for ChatGPT Pro.
    md = []
    md.append("# Shared HKL Thickness-Trend Analysis")
    md.append("")
    md.append(f"Input root: `{input_root}`")
    md.append(f"Output dir: `{out_dir}`")
    md.append("")
    md.append(f"Shared HKLs across all 4 datasets: **{len(metrics)}**")
    md.append(f"Rows in long table: **{len(shared)}**")
    md.append("")
    medians = dataset_summary[["dataset", "median_isig"]].values.tolist()
    md.append("## Median I/sigma by dataset (shared HKLs)")
    for ds, v in medians:
        md.append(f"- {ds}: {float(v):.4f}")
    md.append("")
    md.append("## Exported tables")
    md.append("- shared_hkls_long.csv")
    md.append("- shared_hkl_trend_metrics.csv")
    md.append(f"- top_{args.top_n}_most_stable_hkls.csv")
    md.append(f"- top_{args.top_n}_most_variable_hkls.csv")
    md.append(f"- top_{args.top_n}_steepest_negative_isig_trend.csv")
    md.append(f"- top_{args.top_n}_steepest_positive_isig_trend.csv")
    md.append("- shared_hkl_dataset_summary.csv")
    md.append("")
    md.append("## Exported plots")
    md.append("- hist_isig_shared_hkls.png")
    md.append("- box_isig_shared_hkls.png")
    md.append("- hist_slope_isig_per_nm.png")
    md.append("- scatter_mean_isig_vs_cv_isig.png")

    (out_dir / "CHATGPT_PRO_SHARED_HKL_ANALYSIS.md").write_text("\n".join(md) + "\n")

    print(f"Output dir: {out_dir.resolve()}")
    print(f"Shared HKLs: {len(metrics)}")
    print(f"Long rows: {len(shared)}")
    print(f"Trend metrics: {(out_dir / 'shared_hkl_trend_metrics.csv').resolve()}")


if __name__ == "__main__":
    main()
