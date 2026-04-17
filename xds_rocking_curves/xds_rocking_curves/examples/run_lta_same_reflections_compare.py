from __future__ import annotations

import argparse
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

DATASETS = [
    ("LTA_t1", 100.0),
    ("LTA_t2", 200.0),
    ("LTA_t3", 350.0),
    ("LTA_t4", 600.0),
]

SOURCE_REFLECTIONS = (
    PROJECT_ROOT / "real_data_output" / "LTA_t1_candidates" / "candidate_reflections_selected.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "real_data_output" / "LTA_same_reflections"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-half-width",
        type=int,
        default=5,
        help="Half-width (in frames) around the nearest predicted frame to include.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for comparison artifacts.",
    )
    return parser.parse_args()


def load_reflections_in_order(path: Path) -> list[tuple[int, int, int]]:
    table = pd.read_csv(path)
    hkls: list[tuple[int, int, int]] = []
    for row in table.itertuples(index=False):
        hkl = (int(row.h), int(row.k), int(row.l))
        if hkl not in hkls:
            hkls.append(hkl)
    return hkls


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


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    hkls = load_reflections_in_order(SOURCE_REFLECTIONS)
    if not hkls:
        raise RuntimeError(f"No reflections found in {SOURCE_REFLECTIONS}")

    (output_root / "reflections_used.csv").write_text(
        "h,k,l\n" + "\n".join(f"{h},{k},{l}" for h, k, l in hkls) + "\n"
    )

    rows: list[dict[str, object]] = []

    for dataset_name, thickness_nm in DATASETS:
        xds_dir = BASE_XDS_DIR / dataset_name
        tiff_template = str(BASE_IMG_DIR / dataset_name / "image" / "??????.tiff")
        dataset_root = output_root / dataset_name
        dataset_root.mkdir(parents=True, exist_ok=True)

        for hkl in hkls:
            run_dir = dataset_root / f"hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}"
            results = analyze_single_reflection_dataset(
                gxparm_path=xds_dir / "GXPARM.XDS",
                xds_inp_path=xds_dir / "XDS.INP",
                spot_xds_path=xds_dir / "SPOT.XDS",
                integrate_hkl_path=xds_dir / "INTEGRATE.HKL",
                image_glob=None,
                image_template=tiff_template,
                config=AnalysisConfig(
                    dataset_name=dataset_name,
                    thickness_nm=thickness_nm,
                    hkl=hkl,
                    relevance_mode="window",
                    window_half_width=args.window_half_width,
                    patch_half_size=7,
                    max_center_shift_px=3.0,
                    initial_sigma_px=1.5,
                    min_sigma_px=0.5,
                    max_sigma_px=6.0,
                    auto_choose_rotation_sign=True,
                ),
                output_dir=run_dir,
            )
            stats = summarize_curve(results.curve)
            rows.append(
                {
                    "dataset": dataset_name,
                    "thickness_nm": thickness_nm,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "output_dir": str(run_dir),
                    **stats,
                }
            )

    summary = pd.DataFrame(rows)
    summary["ranking_score"] = (
        summary["fit_success_rate"].fillna(0.0) * 0.7
        + summary["median_r2"].fillna(0.0).clip(lower=0.0, upper=1.0) * 0.3
    )
    summary = summary.sort_values(["dataset", "ranking_score"], ascending=[True, False]).reset_index(drop=True)

    summary_path = output_root / "same_reflections_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Per-HKL comparison table across thicknesses.
    compare = summary.pivot_table(
        index=["h", "k", "l"],
        columns="dataset",
        values="median_r2",
        aggfunc="first",
    ).reset_index()
    compare_path = output_root / "same_reflections_median_r2_pivot.csv"
    compare.to_csv(compare_path, index=False)

    print(f"Window half width: {args.window_half_width}")
    print(f"Reflections used: {output_root / 'reflections_used.csv'}")
    print(f"Summary: {summary_path}")
    print(f"Median R2 pivot: {compare_path}")
    for dataset_name, _ in DATASETS:
        subset = summary[summary["dataset"] == dataset_name].head(5)
        print(f"Top 5 for {dataset_name}:")
        print(subset[["h", "k", "l", "fit_success_rate", "median_r2", "n_relevant_frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
