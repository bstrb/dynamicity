#!/usr/bin/env python3
"""Summarize all reflections from INTEGRATE.HKL across LTA datasets.

This script does not run local rocking-curve fitting. It builds a complete
cross-dataset table directly from all INTEGRATE observations and exports rich
summary files for downstream analysis (e.g., ChatGPT Pro).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers import parse_gxparm, parse_integrate_hkl


DATASETS = [
    ("LTA_t1", 100.0),
    ("LTA_t2", 200.0),
    ("LTA_t3", 350.0),
    ("LTA_t4", 600.0),
]

BASE_XDS_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness")
DEFAULT_OUTPUT_ROOT = Path("real_data_output/LTA_all_reflections_integrate_summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-n", type=int, default=500, help="Top-N rows per dataset exported by I/sigma.")
    return parser.parse_args()


def d_spacing_from_hkl(gxparm, h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
    hkls = np.stack([h, k, l], axis=0).astype(float)
    g = gxparm.reciprocal_reference @ hkls
    g_norm = np.linalg.norm(g, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(g_norm > 0, 1.0 / g_norm, np.nan)
    return d


def add_percentile_col(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    values = pd.to_numeric(df[source_col], errors="coerce")
    valid = values.notna()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if valid.any():
        out.loc[valid] = values.loc[valid].rank(pct=True, method="average")
    df[target_col] = out
    return df


def summarize_dataset(df: pd.DataFrame, dataset: str, thickness_nm: float) -> dict[str, float | int | str]:
    isig = pd.to_numeric(df["isig"], errors="coerce")
    intensity = pd.to_numeric(df["I"], errors="coerce")
    dspace = pd.to_numeric(df["d_spacing_angstrom"], errors="coerce")
    return {
        "dataset": dataset,
        "thickness_nm": thickness_nm,
        "n_reflections": int(len(df)),
        "n_positive_I": int((intensity > 0).sum()),
        "n_isig_ge_2": int((isig >= 2).sum()),
        "n_isig_ge_4": int((isig >= 4).sum()),
        "n_isig_ge_8": int((isig >= 8).sum()),
        "median_I": float(np.nanmedian(intensity.to_numpy(dtype=float))),
        "median_isig": float(np.nanmedian(isig.to_numpy(dtype=float))),
        "p90_isig": float(np.nanpercentile(isig.to_numpy(dtype=float), 90)),
        "max_isig": float(np.nanmax(isig.to_numpy(dtype=float))),
        "median_d_spacing_angstrom": float(np.nanmedian(dspace.to_numpy(dtype=float))),
        "min_d_spacing_angstrom": float(np.nanmin(dspace.to_numpy(dtype=float))),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    dataset_stats: list[dict[str, float | int | str]] = []
    top_rows: list[pd.DataFrame] = []

    for dataset, thickness_nm in DATASETS:
        xds_dir = BASE_XDS_DIR / dataset
        gxparm = parse_gxparm(xds_dir / "GXPARM.XDS")
        obs = parse_integrate_hkl(xds_dir / "INTEGRATE.HKL").observations.copy()

        obs["dataset"] = dataset
        obs["thickness_nm"] = thickness_nm
        obs["isig"] = pd.to_numeric(obs["I"], errors="coerce") / pd.to_numeric(obs["sigma"], errors="coerce")
        obs["d_spacing_angstrom"] = d_spacing_from_hkl(
            gxparm,
            obs["h"].to_numpy(dtype=float),
            obs["k"].to_numpy(dtype=float),
            obs["l"].to_numpy(dtype=float),
        )
        obs["resolution_invA"] = np.where(obs["d_spacing_angstrom"] > 0, 1.0 / obs["d_spacing_angstrom"], np.nan)
        obs = add_percentile_col(obs, "isig", "isig_percentile_in_dataset")
        obs = add_percentile_col(obs, "I", "I_percentile_in_dataset")

        obs["hkl"] = obs.apply(lambda r: f"({int(r['h'])},{int(r['k'])},{int(r['l'])})", axis=1)

        cols = [
            "dataset",
            "thickness_nm",
            "h",
            "k",
            "l",
            "hkl",
            "I",
            "sigma",
            "isig",
            "isig_percentile_in_dataset",
            "I_percentile_in_dataset",
            "x_cal",
            "y_cal",
            "z_cal",
            "frame_est",
            "peak",
            "corr",
            "psi",
            "iseg",
            "d_spacing_angstrom",
            "resolution_invA",
        ]
        obs_out = obs[cols].copy()

        ds_dir = output_root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        obs_out.to_csv(ds_dir / "all_reflections_dataset.csv", index=False)

        top = obs_out.sort_values("isig", ascending=False).head(args.top_n).copy()
        top.to_csv(ds_dir / f"top_{args.top_n}_by_isig.csv", index=False)

        dataset_stats.append(summarize_dataset(obs_out, dataset, thickness_nm))
        all_rows.append(obs_out)
        top_rows.append(top)

    all_df = pd.concat(all_rows, ignore_index=True)
    stats_df = pd.DataFrame(dataset_stats)
    top_df = pd.concat(top_rows, ignore_index=True)

    all_df.to_csv(output_root / "all_reflections_all_datasets_long.csv", index=False)
    stats_df.to_csv(output_root / "dataset_overview_stats.csv", index=False)
    top_df.to_csv(output_root / f"top_{args.top_n}_by_isig_all_datasets.csv", index=False)

    # Cross-dataset overlap by exact HKL triplet.
    overlap = (
        all_df.groupby(["h", "k", "l"], as_index=False)["dataset"]
        .nunique()
        .rename(columns={"dataset": "n_datasets_present"})
    )
    overlap = overlap.sort_values(["n_datasets_present", "h", "k", "l"], ascending=[False, True, True, True])
    overlap.to_csv(output_root / "hkl_dataset_overlap_counts.csv", index=False)

    print(f"Output root: {output_root.resolve()}")
    print(f"All reflections long table: {(output_root / 'all_reflections_all_datasets_long.csv').resolve()}")
    print(f"Dataset overview stats: {(output_root / 'dataset_overview_stats.csv').resolve()}")
    print(f"HKL overlap table: {(output_root / 'hkl_dataset_overlap_counts.csv').resolve()}")
    print(f"Total rows across all datasets: {len(all_df)}")


if __name__ == "__main__":
    main()
