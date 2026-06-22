#!/usr/bin/env python3
"""Diagnostic preview for nonself trend correction effects (lambda=0.5 only).

This script is read-only with respect to upstream processing. It consumes the
existing joined component observations table and reports what a lambda=0.5
high-risk trend correction would do per HKL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NONSELF_COMPONENT_COLUMNS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

REQUIRED_COLUMNS = [
    "h",
    "k",
    "l",
    "I_pr",
    "residual",
    "d_spacing",
    "shell_label",
    *NONSELF_COMPONENT_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview nonself trend correction effects using existing joined "
            "component observations (lambda=0.5 only)."
        )
    )
    parser.add_argument(
        "--joined-component-observations",
        required=True,
        type=Path,
        help="Path to sdyn_component_diagnostics/joined_component_observations.csv",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output directory for preview CSVs and plots",
    )
    parser.add_argument("--min-obs", type=int, default=100)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--min-tail-obs", type=int, default=8)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--intensity-floor", type=float, default=1e-9)

    args = parser.parse_args()

    if args.min_obs <= 0:
        raise SystemExit("--min-obs must be > 0")
    if args.min_tail_obs <= 0:
        raise SystemExit("--min-tail-obs must be > 0")
    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0")
    if args.intensity_floor <= 0.0:
        raise SystemExit("--intensity-floor must be > 0")
    if not (0.0 <= args.low_quantile < args.high_quantile <= 1.0):
        raise SystemExit("Expected 0 <= --low-quantile < --high-quantile <= 1")

    # This preview tool intentionally supports only the requested lambda mode.
    if not np.isclose(float(args.lambda_value), 0.5, atol=1e-12):
        raise SystemExit("This preview currently supports only --lambda-value 0.5")

    return args


def hkl_text(h: int, k: int, l: int) -> str:
    return f"({h},{k},{l})"


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def correction_direction(shift: float) -> str:
    if shift > 0.0:
        return "reduce_high_risk"
    if shift < 0.0:
        return "boost_high_risk"
    return "no_shift"


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_joined_component_observations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"--joined-component-observations not found: {path}")

    table = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(
            "joined_component_observations.csv is missing required column(s): "
            f"{missing}"
        )

    out = table.copy()
    out["h"] = pd.to_numeric(out["h"], errors="coerce")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["l"] = pd.to_numeric(out["l"], errors="coerce")
    out["I_pr"] = pd.to_numeric(out["I_pr"], errors="coerce")
    out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    out["d_spacing"] = pd.to_numeric(out["d_spacing"], errors="coerce")

    for col in NONSELF_COMPONENT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["h", "k", "l", "I_pr", "residual"])
    out[["h", "k", "l"]] = out[["h", "k", "l"]].astype("int64")

    out["nonself_mean"] = out[NONSELF_COMPONENT_COLUMNS].mean(axis=1, skipna=True)
    out = out.dropna(subset=["nonself_mean"])
    out["hkl"] = "(" + out["h"].astype(str) + "," + out["k"].astype(str) + "," + out["l"].astype(str) + ")"
    return out


def analyze_per_hkl(
    table: pd.DataFrame,
    min_obs: int,
    low_quantile: float,
    high_quantile: float,
    min_tail_obs: int,
    lambda_value: float,
    intensity_floor: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = table.groupby(["h", "k", "l"], sort=True)
    for (h, k, l), g in grouped:
        n_obs = int(len(g))
        if n_obs < int(min_obs):
            continue

        nonself = pd.to_numeric(g["nonself_mean"], errors="coerce")
        q_low = float(nonself.quantile(low_quantile))
        q_high = float(nonself.quantile(high_quantile))

        low_mask = nonself <= q_low
        high_mask = nonself >= q_high

        g_low = g.loc[low_mask].copy()
        g_high = g.loc[high_mask].copy()

        n_low = int(len(g_low))
        n_high = int(len(g_high))
        if n_low < int(min_tail_obs) or n_high < int(min_tail_obs):
            continue

        median_low_i_pr = float(pd.to_numeric(g_low["I_pr"], errors="coerce").median())
        median_high_i_pr = float(pd.to_numeric(g_high["I_pr"], errors="coerce").median())
        shift = float(median_high_i_pr - median_low_i_pr)

        baseline = max(abs(median_low_i_pr), float(intensity_floor))
        relative_shift_low_baseline = float(shift / baseline)

        rho = spearman_corr(g["nonself_mean"], g["residual"])
        nonself_p05 = float(nonself.quantile(0.05))
        nonself_p95 = float(nonself.quantile(0.95))
        nonself_spread = float(nonself_p95 - nonself_p05)

        x_low = float(pd.to_numeric(g_low["nonself_mean"], errors="coerce").median())
        x_high = float(pd.to_numeric(g_high["nonself_mean"], errors="coerce").median())
        x_span = x_high - x_low
        slope = float(shift / x_span) if np.isfinite(x_span) and abs(x_span) > 0.0 else np.nan

        high_corr_lambda_05 = float(median_high_i_pr - float(lambda_value) * shift)
        delta_lambda_05 = float(-float(lambda_value) * shift)

        rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "hkl": hkl_text(int(h), int(k), int(l)),
                "d_spacing": first_non_null(pd.to_numeric(g["d_spacing"], errors="coerce")),
                "shell_label": str(first_non_null(g["shell_label"])),
                "n_obs": n_obs,
                "n_low": n_low,
                "n_high": n_high,
                "median_low_I_pr": median_low_i_pr,
                "median_high_I_pr": median_high_i_pr,
                "shift": shift,
                "relative_shift_low_baseline": relative_shift_low_baseline,
                "rho": rho,
                "nonself_p05": nonself_p05,
                "nonself_p95": nonself_p95,
                "nonself_spread": nonself_spread,
                "x_low": x_low,
                "x_high": x_high,
                "slope": slope,
                "correction_direction": correction_direction(shift),
                "high_corr_lambda_05": high_corr_lambda_05,
                "delta_lambda_05": delta_lambda_05,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["d_spacing", "h", "k", "l"], ascending=[False, True, True, True]).reset_index(drop=True)
    return out


def rank_tables(preview: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if preview.empty:
        empty = preview.copy()
        return empty, empty, empty

    top_positive = preview.loc[preview["shift"] > 0.0].copy()
    top_positive = top_positive.sort_values(["shift", "n_obs"], ascending=[False, False]).head(top_n)

    top_negative = preview.loc[preview["shift"] < 0.0].copy()
    top_negative = top_negative.sort_values(["shift", "n_obs"], ascending=[True, False]).head(top_n)

    top_abs = preview.copy()
    top_abs["abs_delta_lambda_05"] = pd.to_numeric(top_abs["delta_lambda_05"], errors="coerce").abs()
    top_abs = top_abs.sort_values(["abs_delta_lambda_05", "n_obs"], ascending=[False, False]).head(top_n)
    top_abs = top_abs.drop(columns=["abs_delta_lambda_05"])

    return top_positive, top_negative, top_abs


def plot_ranked_delta_bars(table: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.35 * max(len(table), 1) + 1.5)))

    if table.empty:
        ax.text(0.5, 0.5, "No HKLs to display", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    plot_df = table.copy().reset_index(drop=True)
    y = np.arange(len(plot_df), dtype=float)
    x = pd.to_numeric(plot_df["delta_lambda_05"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    labels = plot_df["hkl"].astype(str).tolist()
    colors = np.where(x >= 0.0, "#1f77b4", "#d62728")

    ax.barh(y, x, color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("delta_lambda_05 = -0.5 * shift")
    ax.set_ylabel("HKL")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(
    out_paths: dict[str, Path],
    preview: pd.DataFrame,
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    top_abs: pd.DataFrame,
) -> dict[str, Path]:
    files = {
        "all_hkls": out_paths["root"] / "nonself_correction_preview_all_hkls.csv",
        "top_positive": out_paths["root"] / "nonself_correction_preview_top_positive.csv",
        "top_negative": out_paths["root"] / "nonself_correction_preview_top_negative.csv",
        "top_abs": out_paths["root"] / "nonself_correction_preview_top_abs.csv",
        "plot_top_abs": out_paths["plots"] / "top_abs_correction_preview_lambda05.png",
        "plot_top_positive": out_paths["plots"] / "top_positive_correction_preview_lambda05.png",
        "plot_top_negative": out_paths["plots"] / "top_negative_correction_preview_lambda05.png",
    }

    preview.to_csv(files["all_hkls"], index=False)
    top_positive.to_csv(files["top_positive"], index=False)
    top_negative.to_csv(files["top_negative"], index=False)
    top_abs.to_csv(files["top_abs"], index=False)

    plot_ranked_delta_bars(
        top_abs,
        files["plot_top_abs"],
        "Top |delta_lambda_05| HKLs (lambda=0.5 correction preview)",
    )
    plot_ranked_delta_bars(
        top_positive,
        files["plot_top_positive"],
        "Top positive shifts: lambda=0.5 correction preview",
    )
    plot_ranked_delta_bars(
        top_negative,
        files["plot_top_negative"],
        "Top negative shifts: lambda=0.5 correction preview",
    )

    return files


def main() -> None:
    args = parse_args()
    out_paths = ensure_output_layout(args.output_root)

    joined = load_joined_component_observations(args.joined_component_observations)

    preview = analyze_per_hkl(
        joined,
        min_obs=int(args.min_obs),
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
        min_tail_obs=int(args.min_tail_obs),
        lambda_value=float(args.lambda_value),
        intensity_floor=float(args.intensity_floor),
    )

    required_output_columns = [
        "h",
        "k",
        "l",
        "hkl",
        "d_spacing",
        "shell_label",
        "n_obs",
        "n_low",
        "n_high",
        "median_low_I_pr",
        "median_high_I_pr",
        "shift",
        "relative_shift_low_baseline",
        "rho",
        "nonself_p05",
        "nonself_p95",
        "nonself_spread",
        "x_low",
        "x_high",
        "slope",
        "correction_direction",
        "high_corr_lambda_05",
        "delta_lambda_05",
    ]
    if preview.empty:
        preview = pd.DataFrame(columns=required_output_columns)
    else:
        preview = preview[required_output_columns]

    top_positive, top_negative, top_abs = rank_tables(preview, top_n=int(args.top_n))
    files = write_outputs(out_paths, preview, top_positive, top_negative, top_abs)

    print("NONSELF_CORRECTION_PREVIEW_OK")
    print(f"joined_rows={len(joined):,}")
    print(f"analyzed_hkls={len(preview):,}")
    print(f"top_positive_count={len(top_positive):,}")
    print(f"top_negative_count={len(top_negative):,}")
    print(f"top_abs_count={len(top_abs):,}")
    print(f"all_hkls_csv={files['all_hkls']}")
    print(f"top_positive_csv={files['top_positive']}")
    print(f"top_negative_csv={files['top_negative']}")
    print(f"top_abs_csv={files['top_abs']}")
    print(f"plot_top_abs={files['plot_top_abs']}")
    print(f"plot_top_positive={files['plot_top_positive']}")
    print(f"plot_top_negative={files['plot_top_negative']}")


if __name__ == "__main__":
    main()
