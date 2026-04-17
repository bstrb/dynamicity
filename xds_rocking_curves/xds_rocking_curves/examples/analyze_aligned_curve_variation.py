#!/usr/bin/env python3
"""Rank reflections by thickness-dependent intensity and shape variation.

Reads aligned per-reflection curves produced by run_adaptive_curves_for_reflections.py
and computes variation metrics across datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("real_data_output/LTA_reflections_adaptive_top1000_all4"),
        help="Output root from adaptive curve runner.",
    )
    p.add_argument("--top-n", type=int, default=200)
    return p.parse_args()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return np.nan
    if np.allclose(np.std(a), 0.0) or np.allclose(np.std(b), 0.0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def approx_fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate full width at half maximum from centered curve samples."""
    if x.size < 3 or y.size < 3:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return np.nan

    i_peak = int(np.nanargmax(y))
    y_peak = float(y[i_peak])
    if not np.isfinite(y_peak) or y_peak <= 0:
        return np.nan
    half = 0.5 * y_peak

    # Left crossing.
    left_cross = np.nan
    for i in range(i_peak, 0, -1):
        y1, y0 = y[i], y[i - 1]
        if np.isfinite(y1) and np.isfinite(y0) and ((y1 >= half and y0 <= half) or (y1 <= half and y0 >= half)):
            x1, x0 = x[i], x[i - 1]
            if y1 == y0:
                left_cross = float((x1 + x0) / 2.0)
            else:
                t = (half - y0) / (y1 - y0)
                left_cross = float(x0 + t * (x1 - x0))
            break

    # Right crossing.
    right_cross = np.nan
    for i in range(i_peak, len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if np.isfinite(y0) and np.isfinite(y1) and ((y0 >= half and y1 <= half) or (y0 <= half and y1 >= half)):
            x0, x1 = x[i], x[i + 1]
            if y1 == y0:
                right_cross = float((x0 + x1) / 2.0)
            else:
                t = (half - y0) / (y1 - y0)
                right_cross = float(x0 + t * (x1 - x0))
            break

    if not np.isfinite(left_cross) or not np.isfinite(right_cross):
        return np.nan
    return float(right_cross - left_cross)


def main() -> None:
    args = parse_args()
    root = args.root
    aligned_dir = root / "aligned_plots"
    index_path = aligned_dir / "aligned_plot_index.csv"
    curve_summary_path = root / "adaptive_curve_summary.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing: {index_path}")

    index_df = pd.read_csv(index_path)
    curve_summary = pd.read_csv(curve_summary_path) if curve_summary_path.exists() else pd.DataFrame()

    rows: list[dict[str, object]] = []

    for r in index_df.itertuples(index=False):
        h, k, l = int(r.h), int(r.k), int(r.l)
        aligned = pd.read_csv(r.csv_path)
        if aligned.empty:
            continue

        datasets = sorted(aligned["dataset"].unique())
        peaks = (
            aligned.groupby("dataset", as_index=False)["I_peak"].first().rename(columns={"I_peak": "peak_intensity"})
        )
        peak_vals = peaks["peak_intensity"].to_numpy(dtype=float)

        mean_peak = float(np.nanmean(peak_vals))
        std_peak = float(np.nanstd(peak_vals))
        cv_peak = std_peak / mean_peak if np.isfinite(mean_peak) and mean_peak > 0 else np.nan
        delta_peak = float(np.nanmax(peak_vals) - np.nanmin(peak_vals)) if len(peak_vals) else np.nan

        shape_dists = []
        shape_corrs = []
        fwhm_vals = []
        by_ds = {
            ds: aligned[aligned["dataset"] == ds].sort_values("x_centered")["I_smooth_norm"].to_numpy(dtype=float)
            for ds in datasets
        }
        x_by_ds = {
            ds: aligned[aligned["dataset"] == ds].sort_values("x_centered")["x_centered"].to_numpy(dtype=float)
            for ds in datasets
        }
        for ds in datasets:
            fwhm_vals.append(approx_fwhm(x_by_ds[ds], by_ds[ds]))

        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                a = by_ds[datasets[i]]
                b = by_ds[datasets[j]]
                n = min(len(a), len(b))
                if n == 0:
                    continue
                a = a[:n]
                b = b[:n]
                rmse = float(np.sqrt(np.nanmean((a - b) ** 2)))
                corr = safe_corr(a, b)
                shape_dists.append(rmse)
                if np.isfinite(corr):
                    shape_corrs.append(corr)

        mean_shape_rmse = float(np.nanmean(shape_dists)) if shape_dists else np.nan
        mean_shape_corr = float(np.nanmean(shape_corrs)) if shape_corrs else np.nan
        shape_var_score = mean_shape_rmse

        mean_fwhm = float(np.nanmean(np.asarray(fwhm_vals, dtype=float))) if fwhm_vals else np.nan
        std_fwhm = float(np.nanstd(np.asarray(fwhm_vals, dtype=float))) if fwhm_vals else np.nan
        cv_fwhm = std_fwhm / mean_fwhm if np.isfinite(mean_fwhm) and mean_fwhm != 0 else np.nan

        q = curve_summary[(curve_summary["h"] == h) & (curve_summary["k"] == k) & (curve_summary["l"] == l)]
        fail_frac = np.nan
        peak_endpoint_any = np.nan
        n_fit_frames_median = np.nan
        median_r2 = np.nan
        median_rmse = np.nan
        if not q.empty:
            fail_frac = float(np.nanmean(q["n_fit_failed_zeroed"] / q["n_points"]))
            peak_endpoint_any = bool(q["peak_at_endpoint_after_adaptive"].astype(bool).any())
            n_fit_frames_median = float(np.nanmedian(q["n_fit_success"]))

            all_r2 = []
            all_rmse = []
            for row in q.itertuples(index=False):
                curve_path = Path(row.curve_csv)
                if not curve_path.is_absolute():
                    curve_path = Path.cwd() / curve_path
                if not curve_path.exists():
                    continue
                cdf = pd.read_csv(curve_path)
                ok = cdf[cdf["fit_success"] == True]  # noqa: E712
                if ok.empty:
                    continue
                all_r2.extend(pd.to_numeric(ok["r_squared"], errors="coerce").dropna().tolist())
                all_rmse.extend(pd.to_numeric(ok["rmse"], errors="coerce").dropna().tolist())
            if all_r2:
                median_r2 = float(np.nanmedian(np.asarray(all_r2, dtype=float)))
            if all_rmse:
                median_rmse = float(np.nanmedian(np.asarray(all_rmse, dtype=float)))

        rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "n_datasets": int(r.n_datasets),
                "points_per_curve": int(r.points_per_curve),
                "datasets_used": str(r.datasets_used),
                "mean_peak_intensity": mean_peak,
                "std_peak_intensity": std_peak,
                "cv_peak_intensity": cv_peak,
                "delta_peak_intensity": delta_peak,
                "mean_shape_rmse": mean_shape_rmse,
                "mean_shape_corr": mean_shape_corr,
                "shape_variation_score": shape_var_score,
                "mean_fwhm": mean_fwhm,
                "std_fwhm": std_fwhm,
                "cv_fwhm": cv_fwhm,
                "failed_fit_fraction_mean": fail_frac,
                "peak_endpoint_any_dataset": peak_endpoint_any,
                "n_fit_frames_median": n_fit_frames_median,
                "median_r2_successful_fits": median_r2,
                "median_rmse_successful_fits": median_rmse,
                "aligned_csv": str(r.csv_path),
                "aligned_plot": str(r.plot_path),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["shape_variation_score", "cv_peak_intensity"], ascending=[False, False]).reset_index(drop=True)

    out_dir = root / "variation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / "reflection_variation_metrics.csv"
    out.to_csv(all_path, index=False)

    most_vary = out.sort_values(["shape_variation_score", "cv_peak_intensity"], ascending=[False, False]).head(args.top_n)
    least_vary = out.sort_values(["shape_variation_score", "cv_peak_intensity"], ascending=[True, True]).head(args.top_n)

    most_vary.to_csv(out_dir / f"top_{args.top_n}_most_varying_reflections.csv", index=False)
    least_vary.to_csv(out_dir / f"top_{args.top_n}_least_varying_reflections.csv", index=False)

    md_lines = [
        "# Reflection Variation Ranking",
        "",
        f"Input root: `{root}`",
        f"Reflections analyzed: **{len(out)}**",
        "",
        "Metrics:",
        "- intensity variation: `cv_peak_intensity`, `delta_peak_intensity`",
        "- shape variation: `mean_shape_rmse` on smoothed normalized aligned curves",
        "- quality context: failed-fit fraction, endpoint flag, median fitted frames",
        "",
        "Outputs:",
        "- reflection_variation_metrics.csv",
        f"- top_{args.top_n}_most_varying_reflections.csv",
        f"- top_{args.top_n}_least_varying_reflections.csv",
    ]
    (out_dir / "CHATGPT_PRO_REFLECTION_VARIATION.md").write_text("\n".join(md_lines) + "\n")

    print(f"Variation dir: {out_dir.resolve()}")
    print(f"Analyzed reflections: {len(out)}")
    print(f"All metrics: {all_path.resolve()}")


if __name__ == "__main__":
    main()
