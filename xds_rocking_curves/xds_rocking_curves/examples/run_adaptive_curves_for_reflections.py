#!/usr/bin/env python3
"""Adaptive rocking-curve extraction and comparison plots for selected reflections.

Pipeline per dataset/reflection:
1) Predict/fits using local 2D Gaussian on relevant frames (window mode).
2) If a fit fails, intensity defaults to 0 for curve analysis.
3) Adaptively increase window until the max intensity is not at an endpoint.
4) Save final curve + metadata.

Then per reflection:
5) Align curves centered at max intensity across datasets.
6) Enforce equal symmetric point counts around maxima.
7) Plot raw and smoothed overlays across thickness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AnalysisConfig, analyze_single_reflection_dataset


BASE_IMG_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA")
BASE_XDS_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness")
DATASETS = [
    ("LTA_t1", 100.0),
    ("LTA_t2", 200.0),
    ("LTA_t3", 350.0),
    ("LTA_t4", 600.0),
]

DEFAULT_REFLECTIONS = PROJECT_ROOT / "real_data_output" / "LTA_same_reflections_adaptive_w20" / "reflections_used.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "real_data_output" / "LTA_reflections_adaptive_endpoint_safe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflections-csv", type=Path, default=DEFAULT_REFLECTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--x-axis", choices=("phi", "frame"), default="phi")
    parser.add_argument("--relevance-mode", choices=("window", "sg"), default="window")
    parser.add_argument("--sg-threshold", type=float, default=0.02)

    parser.add_argument("--initial-window-half-width", type=int, default=5)
    parser.add_argument("--window-step", type=int, default=4)
    parser.add_argument("--max-window-half-width", type=int, default=25)
    parser.add_argument("--endpoint-margin-points", type=int, default=1)

    parser.add_argument("--patch-half-size", type=int, default=7)
    parser.add_argument("--max-center-shift-px", type=float, default=3.0)
    parser.add_argument("--initial-sigma-px", type=float, default=1.5)
    parser.add_argument("--min-sigma-px", type=float, default=0.5)
    parser.add_argument("--max-sigma-px", type=float, default=6.0)

    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-points-per-curve", type=int, default=7)
    parser.add_argument("--min-datasets", type=int, default=2)
    return parser.parse_args()


def read_reflections(path: Path) -> list[tuple[int, int, int]]:
    tab = pd.read_csv(path)
    need = {"h", "k", "l"}
    if not need.issubset(tab.columns):
        raise ValueError(f"Reflection CSV must contain {sorted(need)}: {path}")
    out: list[tuple[int, int, int]] = []
    for r in tab.itertuples(index=False):
        out.append((int(r.h), int(r.k), int(r.l)))
    return out


def fill_failed_to_zero(curve: pd.DataFrame) -> pd.DataFrame:
    c = curve.copy()
    c["fit_success"] = c["fit_success"].fillna(False).astype(bool)
    c["I_fit_raw"] = pd.to_numeric(c["I_fit"], errors="coerce")
    c["I_fit"] = np.where(c["fit_success"], c["I_fit_raw"], 0.0)
    c["I_fit"] = pd.to_numeric(c["I_fit"], errors="coerce").fillna(0.0)
    max_i = float(c["I_fit"].max()) if not c.empty else np.nan
    c["I_fit_norm"] = c["I_fit"] / max_i if np.isfinite(max_i) and max_i > 0 else np.nan
    return c


def peak_at_endpoint(curve: pd.DataFrame, margin: int, x_col: str) -> tuple[bool, int, int, int]:
    if curve.empty:
        return True, -1, -1, -1
    c = curve.sort_values(x_col).reset_index(drop=True)
    y = pd.to_numeric(c["I_fit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if y.size == 0:
        return True, -1, -1, -1
    i_peak = int(np.argmax(y))
    left_bound = margin
    right_bound = len(c) - 1 - margin
    is_endpoint = i_peak <= left_bound or i_peak >= right_bound
    return is_endpoint, i_peak, left_bound, right_bound


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def center_and_trim(curve: pd.DataFrame, x_col: str, smooth_window: int) -> tuple[pd.DataFrame, int]:
    c = curve.sort_values(x_col).reset_index(drop=True).copy()
    peak_idx = int(np.argmax(c["I_fit"].to_numpy(dtype=float)))
    n_side = min(peak_idx, len(c) - 1 - peak_idx)
    d = c.iloc[peak_idx - n_side : peak_idx + n_side + 1].copy().reset_index(drop=True)

    x_peak = float(c.iloc[peak_idx][x_col])
    i_peak = float(c.iloc[peak_idx]["I_fit"])

    d["x_centered"] = pd.to_numeric(d[x_col], errors="coerce") - x_peak
    d["I_smooth"] = smooth(d["I_fit"].to_numpy(dtype=float), smooth_window)
    d["I_smooth_norm"] = d["I_smooth"] / i_peak if i_peak > 0 else np.nan
    d["x_peak"] = x_peak
    d["I_peak"] = i_peak
    return d, n_side


def align_and_plot_for_hkl(
    h: int,
    k: int,
    l: int,
    dataset_curves: dict[str, pd.DataFrame],
    dataset_thickness: dict[str, float],
    out_dir: Path,
    x_col: str,
    smooth_window: int,
    min_points_per_curve: int,
    min_datasets: int,
) -> tuple[bool, dict[str, object]]:
    centered: dict[str, tuple[pd.DataFrame, int]] = {}
    for ds, c in dataset_curves.items():
        d, n_side = center_and_trim(c, x_col=x_col, smooth_window=smooth_window)
        if (2 * n_side + 1) >= min_points_per_curve:
            centered[ds] = (d, n_side)

    if len(centered) < min_datasets:
        return False, {
            "h": h,
            "k": k,
            "l": l,
            "reason": "insufficient_datasets_passing_min_points",
            "n_datasets_eligible": len(centered),
        }

    common_side = min(v[1] for v in centered.values())
    frames = []
    for ds, (d, _) in centered.items():
        mid = len(d) // 2
        e = d.iloc[mid - common_side : mid + common_side + 1].copy().reset_index(drop=True)
        e["dataset"] = ds
        e["thickness_nm"] = dataset_thickness[ds]
        frames.append(e)

    aligned = pd.concat(frames, ignore_index=True)
    aligned_csv = out_dir / f"hkl_{h}_{k}_{l}_aligned.csv"
    aligned.to_csv(aligned_csv, index=False)

    # Plot raw + smoothed, raw-normalized + smoothed-normalized.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    ax1, ax2 = axes
    for ds in sorted(aligned["dataset"].unique(), key=lambda d: dataset_thickness[d]):
        g = aligned[aligned["dataset"] == ds].sort_values("x_centered")
        label = f"{ds} ({dataset_thickness[ds]:g} nm)"

        ax1.plot(g["x_centered"], g["I_fit"], alpha=0.35, linewidth=1.2)
        ax1.plot(g["x_centered"], g["I_smooth"], linewidth=2.0, label=label)

        ax2.plot(g["x_centered"], g["I_fit_norm"], alpha=0.35, linewidth=1.2)
        ax2.plot(g["x_centered"], g["I_smooth_norm"], linewidth=2.0, label=label)

    ax1.set_title("Raw I_fit (failed fits = 0) + smoothed")
    ax1.set_xlabel("x centered at peak")
    ax1.set_ylabel("I_fit")
    ax1.grid(alpha=0.25)

    ax2.set_title("Normalized I_fit + smoothed")
    ax2.set_xlabel("x centered at peak")
    ax2.set_ylabel("I/I_max")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(f"Adaptive endpoint-safe curves for ({h}, {k}, {l})", y=1.02)
    fig.tight_layout()
    plot_path = out_dir / f"hkl_{h}_{k}_{l}_aligned.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return True, {
        "h": h,
        "k": k,
        "l": l,
        "n_datasets": int(aligned["dataset"].nunique()),
        "datasets_used": ";".join(sorted(aligned["dataset"].unique())),
        "points_per_curve": int(2 * common_side + 1),
        "csv_path": str(aligned_csv.resolve()),
        "plot_path": str(plot_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    reflections = read_reflections(args.reflections_csv)

    output_root = args.output_root
    curves_root = output_root / "curves"
    plots_root = output_root / "aligned_plots"
    output_root.mkdir(parents=True, exist_ok=True)
    curves_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    x_col = "phi_deg" if args.x_axis == "phi" else "frame"
    dataset_thickness = {d: t for d, t in DATASETS}

    adaptive_rows: list[dict[str, object]] = []

    # 1) Adaptive extraction per dataset/reflection.
    for ds, thickness in DATASETS:
        ds_dir = curves_root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)

        xds_dir = BASE_XDS_DIR / ds
        template = str(BASE_IMG_DIR / ds / "image" / "??????.tiff")

        for h, k, l in reflections:
            hkl_dir = ds_dir / f"hkl_{h}_{k}_{l}"
            hkl_dir.mkdir(parents=True, exist_ok=True)

            used_window = args.initial_window_half_width
            endpoint_flag = True
            c_final = pd.DataFrame()

            for w in range(args.initial_window_half_width, args.max_window_half_width + 1, args.window_step):
                result = analyze_single_reflection_dataset(
                    gxparm_path=xds_dir / "GXPARM.XDS",
                    xds_inp_path=xds_dir / "XDS.INP",
                    spot_xds_path=xds_dir / "SPOT.XDS",
                    integrate_hkl_path=xds_dir / "INTEGRATE.HKL",
                    image_glob=None,
                    image_template=template,
                    config=AnalysisConfig(
                        dataset_name=ds,
                        thickness_nm=thickness,
                        hkl=(h, k, l),
                        relevance_mode=args.relevance_mode,
                        sg_threshold=args.sg_threshold,
                        window_half_width=w,
                        patch_half_size=args.patch_half_size,
                        max_center_shift_px=args.max_center_shift_px,
                        initial_sigma_px=args.initial_sigma_px,
                        min_sigma_px=args.min_sigma_px,
                        max_sigma_px=args.max_sigma_px,
                        auto_choose_rotation_sign=True,
                    ),
                    output_dir=hkl_dir,
                )

                c = fill_failed_to_zero(result.curve)
                c = c.sort_values(x_col).reset_index(drop=True)
                is_end, peak_idx, lb, rb = peak_at_endpoint(c, margin=args.endpoint_margin_points, x_col=x_col)

                c_final = c
                used_window = w
                endpoint_flag = is_end

                if not is_end:
                    break

            curve_path = hkl_dir / "rocking_curve_zero_failed.csv"
            c_final.to_csv(curve_path, index=False)

            meta = {
                "dataset": ds,
                "thickness_nm": thickness,
                "h": h,
                "k": k,
                "l": l,
                "relevance_mode": args.relevance_mode,
                "sg_threshold": args.sg_threshold,
                "window_half_width_used": used_window,
                "peak_at_endpoint_after_adaptive": bool(endpoint_flag),
                "x_axis": args.x_axis,
                "endpoint_margin_points": args.endpoint_margin_points,
                "n_points": int(len(c_final)),
                "n_fit_success": int(c_final["fit_success"].sum()) if not c_final.empty else 0,
                "n_fit_failed_zeroed": int((~c_final["fit_success"]).sum()) if not c_final.empty else 0,
                "curve_csv": str(curve_path),
            }
            (hkl_dir / "adaptive_curve_metadata.json").write_text(json.dumps(meta, indent=2))
            adaptive_rows.append(meta)

    pd.DataFrame(adaptive_rows).to_csv(output_root / "adaptive_curve_summary.csv", index=False)

    # 2) Per-reflection aligned plots across thickness.
    index_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for h, k, l in reflections:
        curves: dict[str, pd.DataFrame] = {}
        for ds, _ in DATASETS:
            p = curves_root / ds / f"hkl_{h}_{k}_{l}" / "rocking_curve_zero_failed.csv"
            if not p.exists():
                continue
            c = pd.read_csv(p)
            if c.empty:
                continue
            curves[ds] = c

        ok, info = align_and_plot_for_hkl(
            h=h,
            k=k,
            l=l,
            dataset_curves=curves,
            dataset_thickness=dataset_thickness,
            out_dir=plots_root,
            x_col=x_col,
            smooth_window=args.smooth_window,
            min_points_per_curve=args.min_points_per_curve,
            min_datasets=args.min_datasets,
        )
        if ok:
            index_rows.append(info)
        else:
            skipped_rows.append(info)

    pd.DataFrame(index_rows).to_csv(plots_root / "aligned_plot_index.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(plots_root / "aligned_plot_skipped.csv", index=False)

    print(f"Output root: {output_root.resolve()}")
    print(f"Curves summary: {(output_root / 'adaptive_curve_summary.csv').resolve()}")
    print(f"Aligned plots kept: {len(index_rows)}")
    print(f"Aligned plots skipped: {len(skipped_rows)}")
    print(f"Plot index: {(plots_root / 'aligned_plot_index.csv').resolve()}")


if __name__ == "__main__":
    main()
