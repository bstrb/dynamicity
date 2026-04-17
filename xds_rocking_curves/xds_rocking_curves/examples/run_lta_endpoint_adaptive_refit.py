#!/usr/bin/env python3
"""Adaptive refit for endpoint-peaked rocking curves.

Workflow:
1) Read an existing same-reflections output root (e.g. w12).
2) Detect curves where max I_fit is at first/last successful frame.
3) Refit only those cases with a larger window_half_width.
4) Write updated summary/pivot into a new output root.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AnalysisConfig, analyze_single_reflection_dataset


BASE_IMG_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA")
BASE_XDS_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness")

THICKNESS_BY_DATASET = {
    "LTA_t1": 100.0,
    "LTA_t2": 200.0,
    "LTA_t3": 350.0,
    "LTA_t4": 600.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root",
        type=Path,
        default=PROJECT_ROOT / "real_data_output" / "LTA_same_reflections_wide_w12",
        help="Existing same-reflections output root used to detect endpoint peaks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "real_data_output" / "LTA_same_reflections_adaptive",
        help="New output root for adaptive results.",
    )
    parser.add_argument(
        "--expanded-window-half-width",
        type=int,
        default=20,
        help="Window half-width to use for endpoint-peaked curves.",
    )
    return parser.parse_args()


def summarize_curve(curve: pd.DataFrame) -> dict[str, float | int]:
    if curve.empty:
        return {
            "n_relevant_frames": 0,
            "n_fit_success": 0,
            "fit_success_rate": 0.0,
            "median_r2": np.nan,
            "median_rmse": np.nan,
            "max_I_fit": np.nan,
        }
    fit_ok = curve["fit_success"].fillna(False).astype(bool)
    r2 = pd.to_numeric(curve["r_squared"], errors="coerce")
    rmse = pd.to_numeric(curve["rmse"], errors="coerce")
    i_fit = pd.to_numeric(curve["I_fit"], errors="coerce")
    n_total = int(len(curve))
    n_ok = int(fit_ok.sum())
    return {
        "n_relevant_frames": n_total,
        "n_fit_success": n_ok,
        "fit_success_rate": float(n_ok / n_total) if n_total else 0.0,
        "median_r2": float(np.nanmedian(r2.to_numpy(dtype=float))) if n_total else np.nan,
        "median_rmse": float(np.nanmedian(rmse.to_numpy(dtype=float))) if n_total else np.nan,
        "max_I_fit": float(np.nanmax(i_fit.to_numpy(dtype=float))) if n_total else np.nan,
    }


def is_endpoint_peak(curve_path: Path) -> bool:
    curve = pd.read_csv(curve_path)
    good = curve[curve["fit_success"] == True].copy()  # noqa: E712
    if good.empty:
        return False
    good = good.sort_values("frame")
    idx = good["I_fit"].astype(float).idxmax()
    max_frame = int(good.loc[idx, "frame"])
    return max_frame == int(good["frame"].min()) or max_frame == int(good["frame"].max())


def load_rows(summary_path: Path) -> list[dict[str, object]]:
    table = pd.read_csv(summary_path)
    return [
        {
            "dataset": str(r.dataset),
            "thickness_nm": float(r.thickness_nm),
            "h": int(r.h),
            "k": int(r.k),
            "l": int(r.l),
        }
        for r in table.itertuples(index=False)
    ]


def rebuild_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    out_rows: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        rel_output_dir = output_root.relative_to(PROJECT_ROOT) / dataset / f"hkl_{h}_{k}_{l}"
        curve_path = output_root / dataset / f"hkl_{h}_{k}_{l}" / "rocking_curve.csv"
        curve = pd.read_csv(curve_path)
        stats = summarize_curve(curve)
        out_rows.append({**row, "output_dir": str(rel_output_dir), **stats})

    summary = pd.DataFrame(out_rows)
    summary["ranking_score"] = (
        summary["fit_success_rate"].fillna(0.0) * 0.7
        + summary["median_r2"].fillna(0.0).clip(lower=0.0, upper=1.0) * 0.3
    )
    summary = summary.sort_values(["dataset", "ranking_score"], ascending=[True, False]).reset_index(drop=True)

    summary_path = output_root / "same_reflections_summary.csv"
    summary.to_csv(summary_path, index=False)

    pivot = summary.pivot_table(
        index=["h", "k", "l"],
        columns="dataset",
        values="median_r2",
        aggfunc="first",
    ).reset_index()
    pivot_path = output_root / "same_reflections_median_r2_pivot.csv"
    pivot.to_csv(pivot_path, index=False)


def main() -> None:
    args = parse_args()
    base_root = args.base_root if args.base_root.is_absolute() else (PROJECT_ROOT / args.base_root)
    output_root = args.output_root if args.output_root.is_absolute() else (PROJECT_ROOT / args.output_root)
    base_root = base_root.resolve()
    output_root = output_root.resolve()

    if not base_root.exists():
        raise FileNotFoundError(f"Base root not found: {base_root}")

    summary_path = base_root / "same_reflections_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary in base root: {summary_path}")

    rows = load_rows(summary_path)

    # Start from the base results to preserve all non-target runs.
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(base_root, output_root)

    targets: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        curve_path = output_root / dataset / f"hkl_{h}_{k}_{l}" / "rocking_curve.csv"
        if curve_path.exists() and is_endpoint_peak(curve_path):
            targets.append(row)

    print(f"Base rows: {len(rows)}")
    print(f"Endpoint-peaked targets: {len(targets)}")

    for row in targets:
        dataset = str(row["dataset"])
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        thickness_nm = float(row["thickness_nm"])

        xds_dir = BASE_XDS_DIR / dataset
        tiff_template = str(BASE_IMG_DIR / dataset / "image" / "??????.tiff")
        run_dir = output_root / dataset / f"hkl_{h}_{k}_{l}"

        analyze_single_reflection_dataset(
            gxparm_path=xds_dir / "GXPARM.XDS",
            xds_inp_path=xds_dir / "XDS.INP",
            spot_xds_path=xds_dir / "SPOT.XDS",
            integrate_hkl_path=xds_dir / "INTEGRATE.HKL",
            image_glob=None,
            image_template=tiff_template,
            config=AnalysisConfig(
                dataset_name=dataset,
                thickness_nm=thickness_nm,
                hkl=(h, k, l),
                relevance_mode="window",
                window_half_width=args.expanded_window_half_width,
                patch_half_size=7,
                max_center_shift_px=3.0,
                initial_sigma_px=1.5,
                min_sigma_px=0.5,
                max_sigma_px=6.0,
                auto_choose_rotation_sign=True,
            ),
            output_dir=run_dir,
        )
        print(f"Refit with wider window: {dataset} ({h},{k},{l})")

    rebuild_summary(output_root, rows)
    print(f"Adaptive output root: {output_root}")
    print(f"Summary: {output_root / 'same_reflections_summary.csv'}")
    print(f"Pivot: {output_root / 'same_reflections_median_r2_pivot.csv'}")


if __name__ == "__main__":
    main()
