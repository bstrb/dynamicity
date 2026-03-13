"""
Proof-of-concept script: run orientation-aware weighting on multiple XDS datasets
with different crystal thicknesses, then correlate sensitivity score `S` with
intensity variation across thickness.

Usage example:

python examples/run_thickness_poc.py \
  --dataset LTA_t1,100,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t1/GXPARM.XDS,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t1/INTEGRATE.HKL \
  --dataset LTA_t2,200,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t2/GXPARM.XDS,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t2/INTEGRATE.HKL \
  --dataset LTA_t3,350,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t3/GXPARM.XDS,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t3/INTEGRATE.HKL \
  --dataset LTA_t4,600,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t4/GXPARM.XDS,/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t4/INTEGRATE.HKL \
  --output results/thickness_poc

Defaults:
- keeps frames 50, 284, 592 if present; adds `--offaxis-per-set` frames with
  largest zone_axis_angle per thickness.
- aggregates reflections present in at least 3 thicknesses.
- variation metric: std(mean_I_by_thickness) / mean(mean_I_by_thickness).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, run_xds_pipeline


@dataclass(slots=True)
class DatasetSpec:
    name: str
    thickness_nm: float
    gxparm: Path
    integrate: Path
    xds_inp: Path | None = None


def parse_dataset_arg(raw: str) -> DatasetSpec:
    parts = raw.split(",")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            "--dataset expects name,thickness_nm,gxparm,integrate[,xds_inp]"
        )
    name = parts[0].strip()
    try:
        thickness = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thickness_nm must be numeric") from exc
    gxparm = Path(parts[2]).expanduser().resolve()
    integrate = Path(parts[3]).expanduser().resolve()
    xds_inp = Path(parts[4]).expanduser().resolve() if len(parts) == 5 else None
    for path in [gxparm, integrate]:
        if not path.exists():
            raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    if xds_inp is not None and not xds_inp.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {xds_inp}")
    return DatasetSpec(name=name, thickness_nm=thickness, gxparm=gxparm, integrate=integrate, xds_inp=xds_inp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thickness POC using orientation-aware weighting on XDS datasets")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_arg,
        help="name,thickness_nm,gxparm,integrate[,xds_inp] (repeat for each thickness)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=[50, 284, 592],
        help="Explicit frame IDs to keep if present (applied per dataset)",
    )
    parser.add_argument(
        "--offaxis-per-set",
        type=int,
        default=4,
        help="Number of additional off-axis frames (largest zone_axis_angle) to sample per dataset",
    )
    parser.add_argument("--dmin", type=float, default=0.8, help="High-resolution limit")
    parser.add_argument("--dmax", type=float, default=20.0, help="Low-resolution limit")
    parser.add_argument(
        "--min-thickness-presence",
        type=int,
        default=3,
        help="Require reflection to be present in at least this many thickness groups",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def select_frames(frame_summary: pd.DataFrame, include_frames: Iterable[int], offaxis_count: int) -> list[int]:
    available = set(frame_summary["frame"].tolist())
    base = [f for f in include_frames if f in available]
    # Pick off-axis frames with largest zone_axis_angle
    if "zone_axis_angle" not in frame_summary.columns:
        return sorted(set(base))
    candidates = (
        frame_summary.loc[~frame_summary["frame"].isin(base)]
        .sort_values("zone_axis_angle", ascending=False)
        .head(offaxis_count)
    )
    picked = base + candidates["frame"].astype(int).tolist()
    return sorted(set(picked))


def run_dataset(spec: DatasetSpec, config: PipelineConfig, include_frames: Iterable[int], offaxis_count: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = run_xds_pipeline(
        gxparm_path=spec.gxparm,
        integrate_path=spec.integrate,
        xds_inp_path=spec.xds_inp,
        config=config,
    )
    frames_to_keep = select_frames(results.frame_summary, include_frames, offaxis_count)
    subset_reflections = results.reflection_table[results.reflection_table["frame"].isin(frames_to_keep)].copy()
    subset_frames = results.frame_summary[results.frame_summary["frame"].isin(frames_to_keep)].copy()

    ds_dir = output_dir / spec.name
    ds_dir.mkdir(parents=True, exist_ok=True)
    subset_reflections.to_csv(ds_dir / "reflection_scores_subset.csv", index=False)
    subset_frames.to_csv(ds_dir / "frame_summary_subset.csv", index=False)

    return subset_reflections, subset_frames


def aggregate_across_thickness(
    reflections_by_set: list[tuple[DatasetSpec, pd.DataFrame]],
    min_thickness_presence: int,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for spec, refl in reflections_by_set:
        grouped = refl.groupby(["h", "k", "l"], as_index=False).agg(
            I_mean=("I", "mean"),
            S_mean=("S", "mean"),
            n_frames=("frame", "nunique"),
        )
        grouped["dataset"] = spec.name
        grouped["thickness_nm"] = spec.thickness_nm
        records.append(grouped)
    merged = pd.concat(records, ignore_index=True)

    intensity_pivot = merged.pivot_table(index=["h", "k", "l"], columns="thickness_nm", values="I_mean")
    score_pivot = merged.pivot_table(index=["h", "k", "l"], columns="thickness_nm", values="S_mean")

    presence = intensity_pivot.notna().sum(axis=1)
    intensity_mean = intensity_pivot.mean(axis=1, skipna=True)
    intensity_std = intensity_pivot.std(axis=1, ddof=0, skipna=True)
    variation = intensity_std / intensity_mean
    score_mean = score_pivot.mean(axis=1, skipna=True)

    summary = pd.DataFrame(
        {
            "h": intensity_pivot.index.get_level_values("h"),
            "k": intensity_pivot.index.get_level_values("k"),
            "l": intensity_pivot.index.get_level_values("l"),
            "mean_intensity": intensity_mean.to_numpy(),
            "variation_std_over_mean": variation.to_numpy(),
            "mean_score_S": score_mean.to_numpy(),
            "thickness_presence": presence.to_numpy(),
        }
    )
    summary = summary[summary["thickness_presence"] >= int(min_thickness_presence)].reset_index(drop=True)
    return summary


def try_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    scatter_path = output_dir / "S_vs_variation.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(summary["mean_score_S"], summary["variation_std_over_mean"], s=6, alpha=0.6, edgecolor="none")
    ax.set_xlabel("Mean sensitivity score S")
    ax.set_ylabel("Intensity variation (std/mean across thickness)")
    ax.set_title("Dynamical sensitivity vs thickness variation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    # High vs low S split at median
    median_S = float(summary["mean_score_S"].median()) if not summary.empty else math.nan
    if not math.isnan(median_S):
        fig, ax = plt.subplots(figsize=(4, 4))
        low = summary.loc[summary["mean_score_S"] <= median_S, "variation_std_over_mean"].dropna()
        high = summary.loc[summary["mean_score_S"] > median_S, "variation_std_over_mean"].dropna()
        ax.boxplot([low, high], tick_labels=["low S", "high S"], showfliers=False)
        ax.set_ylabel("Intensity variation (std/mean)")
        ax.set_title("Variation split by sensitivity")
        fig.tight_layout()
        fig.savefig(output_dir / "variation_boxplot.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(dmin=args.dmin, dmax=args.dmax)

    reflections_by_set: list[tuple[DatasetSpec, pd.DataFrame]] = []
    frame_summaries: list[tuple[DatasetSpec, pd.DataFrame]] = []
    for spec in args.dataset:
        subset_reflections, subset_frames = run_dataset(
            spec=spec,
            config=config,
            include_frames=args.frames,
            offaxis_count=args.offaxis_per_set,
            output_dir=output_dir,
        )
        reflections_by_set.append((spec, subset_reflections))
        frame_summaries.append((spec, subset_frames))

    summary = aggregate_across_thickness(reflections_by_set, min_thickness_presence=args.min_thickness_presence)
    summary.to_csv(output_dir / "hkl_variation_summary.csv", index=False)

    try_plot(summary, output_dir)

    # Minimal console report
    print(f"Saved per-dataset subsets to {output_dir}")
    print(f"Aggregated HKL rows (presence >= {args.min_thickness_presence}): {len(summary)}")
    if not summary.empty:
        print(
            f"Variation median: {summary['variation_std_over_mean'].median():.4f}, "
            f"mean: {summary['variation_std_over_mean'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
