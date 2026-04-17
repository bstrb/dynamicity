#!/usr/bin/env python3
"""Plot centered rocking-curve comparisons for specific reflections.

For each HKL and dataset:
- Load rocking_curve.csv
- Keep successful fits
- Center x-axis at the max-intensity point
- Enforce equal samples on both sides of the maximum
- Optionally smooth intensity traces

Outputs:
- One aligned CSV per HKL
- One figure per HKL with raw + smoothed overlays (raw and normalized)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("real_data_output/LTA_same_reflections_adaptive_w20")
DEFAULT_DATASETS = ("LTA_t1", "LTA_t2", "LTA_t3", "LTA_t4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root with per-dataset HKL folders.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/specific_reflection_centered_plots).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folders to include.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("phi", "frame"),
        default="phi",
        help="Use phi_deg or frame as x-axis before centering.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Centered moving-average window for smoothing (odd preferred).",
    )
    parser.add_argument(
        "--hkls",
        type=str,
        default="",
        help="Semicolon-separated HKLs, e.g. '3,0,-3;-3,0,3'. If empty, use reflections_used.csv.",
    )
    parser.add_argument(
        "--max-hkls",
        type=int,
        default=0,
        help="Optional limit when reading reflections_used.csv (0 means all).",
    )
    parser.add_argument(
        "--min-points-per-curve",
        type=int,
        default=7,
        help="Minimum symmetric points required per dataset curve to be eligible.",
    )
    parser.add_argument(
        "--min-datasets",
        type=int,
        default=2,
        help="Minimum number of datasets that must pass the point threshold.",
    )
    return parser.parse_args()


def parse_hkls(hkls_text: str, root: Path, max_hkls: int) -> list[tuple[int, int, int]]:
    if hkls_text.strip():
        out: list[tuple[int, int, int]] = []
        for token in hkls_text.split(";"):
            token = token.strip()
            if not token:
                continue
            h, k, l = [int(v.strip()) for v in token.split(",")]
            out.append((h, k, l))
        return out

    reflections_path = root / "reflections_used.csv"
    if not reflections_path.exists():
        raise FileNotFoundError(f"No --hkls provided and missing reflections_used.csv: {reflections_path}")
    refl = pd.read_csv(reflections_path)
    hkls = [(int(r.h), int(r.k), int(r.l)) for r in refl.itertuples(index=False)]
    if max_hkls > 0:
        hkls = hkls[:max_hkls]
    return hkls


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    s = pd.Series(values)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def load_curve(path: Path, x_axis: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df[df["fit_success"] == True].copy()  # noqa: E712
    if df.empty:
        return df

    x_col = "phi_deg" if x_axis == "phi" and "phi_deg" in df.columns else "frame"
    df = df.sort_values(x_col).reset_index(drop=True)
    df["x"] = pd.to_numeric(df[x_col], errors="coerce")
    df["I_fit"] = pd.to_numeric(df["I_fit"], errors="coerce")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["I_fit"])].copy()
    return df


def center_and_trim(df: pd.DataFrame, smooth_window: int) -> tuple[pd.DataFrame, int]:
    peak_pos = int(df["I_fit"].to_numpy(dtype=float).argmax())
    left = peak_pos
    right = len(df) - 1 - peak_pos
    n_side = min(left, right)

    start = peak_pos - n_side
    end = peak_pos + n_side
    cut = df.iloc[start : end + 1].copy().reset_index(drop=True)

    x_peak = float(df.iloc[peak_pos]["x"])
    i_peak = float(df.iloc[peak_pos]["I_fit"])

    cut["x_centered"] = cut["x"] - x_peak
    cut["I_smooth"] = smooth_series(cut["I_fit"].to_numpy(dtype=float), smooth_window)
    cut["I_fit_norm"] = cut["I_fit"] / i_peak if i_peak > 0 else np.nan
    cut["I_smooth_norm"] = cut["I_smooth"] / i_peak if i_peak > 0 else np.nan
    cut["x_peak"] = x_peak
    cut["I_peak"] = i_peak
    return cut, n_side


def align_common_side(curves: dict[str, pd.DataFrame], smooth_window: int) -> tuple[pd.DataFrame, int]:
    centered: dict[str, tuple[pd.DataFrame, int]] = {}
    for ds, df in curves.items():
        centered[ds] = center_and_trim(df, smooth_window)

    common_side = min(v[1] for v in centered.values())
    rows = []
    for ds, (df, _) in centered.items():
        # df currently has symmetric count around its own peak; trim again to common length.
        mid = len(df) // 2
        start = mid - common_side
        end = mid + common_side
        d2 = df.iloc[start : end + 1].copy().reset_index(drop=True)
        d2["dataset"] = ds
        rows.append(d2)

    out = pd.concat(rows, ignore_index=True)
    return out, common_side


def plot_hkl(aligned: pd.DataFrame, h: int, k: int, l: int, out_path: Path) -> None:
    datasets = list(aligned["dataset"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ax_raw, ax_norm = axes

    for ds in datasets:
        g = aligned[aligned["dataset"] == ds].sort_values("x_centered")
        thickness = g["thickness_nm"].iloc[0] if "thickness_nm" in g.columns and not g.empty else np.nan
        label = f"{ds} ({thickness:g} nm)" if np.isfinite(thickness) else ds

        ax_raw.plot(g["x_centered"], g["I_fit"], alpha=0.35, linewidth=1.2)
        ax_raw.plot(g["x_centered"], g["I_smooth"], linewidth=2.0, label=label)

        ax_norm.plot(g["x_centered"], g["I_fit_norm"], alpha=0.35, linewidth=1.2)
        ax_norm.plot(g["x_centered"], g["I_smooth_norm"], linewidth=2.0, label=label)

    ax_raw.set_title("Raw I_fit (faint) + smoothed")
    ax_raw.set_xlabel("x centered at max")
    ax_raw.set_ylabel("I_fit")
    ax_raw.grid(alpha=0.25)

    ax_norm.set_title("Normalized I_fit (faint) + smoothed")
    ax_norm.set_xlabel("x centered at max")
    ax_norm.set_ylabel("I / I_max")
    ax_norm.grid(alpha=0.25)
    ax_norm.legend(loc="best", fontsize=8)

    fig.suptitle(f"Centered symmetric rocking curves for ({h}, {k}, {l})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "specific_reflection_centered_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    hkls = parse_hkls(args.hkls, root, args.max_hkls)
    index_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for h, k, l in hkls:
        curves: dict[str, pd.DataFrame] = {}

        for ds in args.datasets:
            curve_path = root / ds / f"hkl_{h}_{k}_{l}" / "rocking_curve.csv"
            if not curve_path.exists():
                continue
            c = load_curve(curve_path, args.x_axis)
            if c.empty:
                continue
            curves[ds] = c

        if len(curves) < args.min_datasets:
            skipped_rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "reason": "insufficient_datasets_with_curve",
                    "n_datasets_with_curve": len(curves),
                    "n_datasets_eligible": 0,
                }
            )
            continue

        centered_cache: dict[str, tuple[pd.DataFrame, int]] = {
            ds: center_and_trim(df, args.smooth_window) for ds, df in curves.items()
        }

        eligible = {
            ds: pair for ds, pair in centered_cache.items() if (2 * pair[1] + 1) >= args.min_points_per_curve
        }

        if len(eligible) < args.min_datasets:
            skipped_rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "reason": "insufficient_datasets_passing_min_points",
                    "n_datasets_with_curve": len(curves),
                    "n_datasets_eligible": len(eligible),
                }
            )
            continue

        eligible_curves = {ds: curves[ds] for ds in eligible}
        aligned, common_side = align_common_side(eligible_curves, args.smooth_window)

        csv_path = out_dir / f"hkl_{h}_{k}_{l}_centered_aligned.csv"
        aligned.to_csv(csv_path, index=False)

        png_path = out_dir / f"hkl_{h}_{k}_{l}_centered_aligned.png"
        plot_hkl(aligned, h, k, l, png_path)

        index_rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "n_datasets": int(aligned["dataset"].nunique()),
                "common_side_points": int(common_side),
                "points_per_curve": int(2 * common_side + 1),
                "datasets_used": ";".join(sorted(aligned["dataset"].dropna().unique())),
                "csv_path": str(csv_path.resolve()),
                "plot_path": str(png_path.resolve()),
            }
        )

    index_df = pd.DataFrame(index_rows)
    index_path = out_dir / "centered_reflection_plot_index.csv"
    index_df.to_csv(index_path, index=False)

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_path = out_dir / "centered_reflection_skipped.csv"
    skipped_df.to_csv(skipped_path, index=False)

    print(f"Output dir: {out_dir.resolve()}")
    print(f"Reflections processed: {len(index_df)}")
    print(f"Reflections skipped: {len(skipped_df)}")
    print(f"Index: {index_path.resolve()}")
    print(f"Skipped: {skipped_path.resolve()}")


if __name__ == "__main__":
    main()
