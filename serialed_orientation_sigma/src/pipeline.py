from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .geometry import UnitCell
from .orientation_metrics import OrientationMetricConfig, compute_orientation_metrics
from .parsers import (
    load_orientation_table,
    load_reflection_table,
    load_xds_rotation_series,
    parse_cell_json,
)
from .reflection_metrics import ReflectionMetricConfig, compute_reflection_metrics
from .weighting import WeightingConfig, apply_orientation_aware_weighting


@dataclass(slots=True)
class PipelineConfig:
    """High-level configuration for the full SerialED weighting pipeline."""

    dmin: float = 0.8
    dmax: float | None = 20.0
    sg_threshold: float = 0.02
    zone_axis_limit: int = 4
    hkl_limit: int | None = 20
    max_candidate_reflections: int | None = 50000
    alpha: float = 0.5
    filter_threshold: float | None = None
    xi_scale_nm: float = 200.0
    structure_factor_decay: float = 0.35
    zone_axis_soft_angle_deg: float = 5.0
    combined_formulation: str = "log_multibeam"
    zone_axis_weight: float = 0.25
    n_workers: int = 1
    chunk_size_frames: int = 2000
    voltage_kv: float | None = None
    beam_direction: tuple[float, float, float] | None = None


@dataclass(slots=True)
class PipelineResults:
    """Container for all pipeline outputs."""

    frame_summary: pd.DataFrame
    reflection_table: pd.DataFrame
    cell: UnitCell
    config: PipelineConfig
    metadata: dict[str, Any]


def _orientation_config_from_pipeline(config: PipelineConfig) -> OrientationMetricConfig:
    return OrientationMetricConfig(
        dmin=config.dmin,
        dmax=config.dmax,
        sg_threshold=config.sg_threshold,
        zone_axis_limit=config.zone_axis_limit,
        hkl_limit=config.hkl_limit,
        max_candidate_reflections=config.max_candidate_reflections,
        voltage_kv=config.voltage_kv,
        beam_direction=config.beam_direction,
        n_workers=config.n_workers,
        chunk_size_frames=config.chunk_size_frames,
    )


def _reflection_config_from_pipeline(config: PipelineConfig) -> ReflectionMetricConfig:
    return ReflectionMetricConfig(
        xi_scale_nm=config.xi_scale_nm,
        structure_factor_decay=config.structure_factor_decay,
        zone_axis_soft_angle_deg=config.zone_axis_soft_angle_deg,
        combined_formulation=config.combined_formulation,
        zone_axis_weight=config.zone_axis_weight,
        voltage_kv=config.voltage_kv,
        beam_direction=config.beam_direction,
        n_workers=config.n_workers,
        chunk_size_frames=config.chunk_size_frames,
    )


def _weighting_config_from_pipeline(config: PipelineConfig) -> WeightingConfig:
    return WeightingConfig(alpha=config.alpha, filter_threshold=config.filter_threshold)


def run_pipeline_from_tables(
    orientations: pd.DataFrame,
    reflections: pd.DataFrame,
    cell: UnitCell,
    config: PipelineConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> PipelineResults:
    """Run the full orientation-aware weighting pipeline from in-memory tables."""
    cfg = config or PipelineConfig()
    cell_local = UnitCell(**cell.as_dict())
    if cfg.voltage_kv is not None:
        cell_local.voltage_kv = float(cfg.voltage_kv)
    if cfg.beam_direction is not None:
        cell_local.beam_direction = tuple(float(v) for v in cfg.beam_direction)

    frame_summary = compute_orientation_metrics(
        orientations=orientations,
        cell=cell_local,
        config=_orientation_config_from_pipeline(cfg),
    )
    reflection_metrics = compute_reflection_metrics(
        reflections=reflections,
        orientations=orientations,
        frame_summary=frame_summary,
        cell=cell_local,
        config=_reflection_config_from_pipeline(cfg),
    )
    weighted = apply_orientation_aware_weighting(reflection_metrics, _weighting_config_from_pipeline(cfg))

    frame_stats = (
        weighted.groupby("frame", as_index=False)
        .agg(
            mean_dynamical_score=("S", "mean"),
            mean_d_2beam=("d_2beam", "mean"),
            n_reflections=("S", "size"),
            fraction_kept=("keep", "mean"),
        )
        .sort_values("frame")
    )
    frame_summary = frame_summary.merge(frame_stats, on="frame", how="left")
    frame_summary["fraction_kept"] = frame_summary["fraction_kept"].fillna(1.0)

    return PipelineResults(
        frame_summary=frame_summary,
        reflection_table=weighted,
        cell=cell_local,
        config=cfg,
        metadata=metadata or {},
    )


def run_serialed_csv_pipeline(
    orientations_path: str | Path,
    reflections_path: str | Path,
    cell_path: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResults:
    """Run the full pipeline in SerialED snapshot mode from CSV/TSV files."""
    orientations = load_orientation_table(orientations_path)
    reflections = load_reflection_table(reflections_path)
    cell = parse_cell_json(cell_path)
    metadata = {
        "mode": "serialed_snapshot",
        "orientations_path": str(orientations_path),
        "reflections_path": str(reflections_path),
        "cell_path": str(cell_path),
    }
    return run_pipeline_from_tables(orientations, reflections, cell, config=config, metadata=metadata)


def run_xds_pipeline(
    gxparm_path: str | Path,
    integrate_path: str | Path,
    xds_inp_path: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> PipelineResults:
    """Run the full pipeline in XDS rotation-series mode."""
    orientations, reflections, cell, gxparm_metadata, xds_inp = load_xds_rotation_series(
        gxparm_path=gxparm_path,
        integrate_path=integrate_path,
        xds_inp_path=xds_inp_path,
    )
    metadata = {
        "mode": "xds_rotation_series",
        "gxparm_path": str(gxparm_path),
        "integrate_path": str(integrate_path),
        "xds_inp_path": str(xds_inp_path) if xds_inp_path is not None else None,
        "gxparm": {
            "start_frame": gxparm_metadata.start_frame,
            "phi0_deg": gxparm_metadata.phi0_deg,
            "dphi_deg": gxparm_metadata.dphi_deg,
            "rotation_axis": gxparm_metadata.rotation_axis.tolist(),
            "wavelength_angstrom": gxparm_metadata.wavelength_angstrom,
            "beam_direction": gxparm_metadata.beam_direction.tolist(),
        },
        "xds_inp": xds_inp,
    }
    return run_pipeline_from_tables(orientations, reflections, cell, config=config, metadata=metadata)


def export_results(results: PipelineResults, output_dir: str | Path) -> dict[str, Path]:
    """Write pipeline results to an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_summary_path = output_path / "frame_summary.csv"
    reflection_scores_path = output_path / "reflection_scores.csv"
    merge_weights_path = output_path / "merge_weights.csv"
    metadata_path = output_path / "pipeline_metadata.json"

    results.frame_summary.to_csv(frame_summary_path, index=False)
    results.reflection_table.to_csv(reflection_scores_path, index=False)
    export_columns = [
        "frame",
        "h",
        "k",
        "l",
        "I",
        "sigma",
        "sigma_new",
        "weight_new",
        "keep",
        "S",
        "sg",
        "d_2beam",
        "zone_axis_angle",
        "N_excited",
    ]
    available_columns = [col for col in export_columns if col in results.reflection_table.columns]
    results.reflection_table[available_columns].to_csv(merge_weights_path, index=False)

    metadata_payload = {
        "config": asdict(results.config),
        "cell": results.cell.as_dict(),
        "metadata": results.metadata,
        "n_frames": int(results.frame_summary.shape[0]),
        "n_reflections": int(results.reflection_table.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    return {
        "frame_summary": frame_summary_path,
        "reflection_scores": reflection_scores_path,
        "merge_weights": merge_weights_path,
        "metadata": metadata_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface parser."""
    parser = argparse.ArgumentParser(description="Orientation-aware SerialED weighting pipeline")
    parser.add_argument("--orientations", type=str, help="CSV/TSV/parquet table with per-frame UB matrices")
    parser.add_argument("--reflections", type=str, help="CSV/TSV/parquet table with reflections")
    parser.add_argument("--cell", type=str, help="JSON cell file for snapshot mode")
    parser.add_argument("--gxparm", type=str, help="GXPARM.XDS or XPARM.XDS file")
    parser.add_argument("--integrate", type=str, help="INTEGRATE.HKL file")
    parser.add_argument("--xds-inp", type=str, default=None, help="Optional XDS.INP file")
    parser.add_argument("--dmin", type=float, default=0.8, help="High-resolution limit in angstrom")
    parser.add_argument("--dmax", type=float, default=20.0, help="Low-resolution limit in angstrom")
    parser.add_argument("--sg-threshold", type=float, default=0.02, help="Excitation-error threshold for frame crowding")
    parser.add_argument("--zone-axis-limit", type=int, default=4, help="Maximum integer index for zone-axis search")
    parser.add_argument("--hkl-limit", type=int, default=20, help="Hard cap on Miller-index magnitude for candidate generation")
    parser.add_argument(
        "--max-candidate-reflections",
        type=int,
        default=50000,
        help="Maximum number of candidate shell reflections used for excitation density",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Score-to-sigma weighting strength")
    parser.add_argument("--filter-threshold", type=float, default=None, help="Optional score cutoff for keep mask")
    parser.add_argument("--xi-scale-nm", type=float, default=200.0, help="Extinction-distance scale constant in nm")
    parser.add_argument(
        "--structure-factor-decay",
        type=float,
        default=0.35,
        help="Decay coefficient for the structure-factor proxy",
    )
    parser.add_argument(
        "--combined-formulation",
        type=str,
        default="log_multibeam",
        choices=["log_multibeam", "zone_axis_boost", "weighted_sum"],
        help="Formula used to combine orientation and reflection metrics",
    )
    parser.add_argument("--zone-axis-weight", type=float, default=0.25, help="Zone-axis contribution in alternative score formulas")
    parser.add_argument("--zone-axis-soft-angle-deg", type=float, default=5.0, help="Soft scale for zone-axis proximity transform")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--chunk-size-frames", type=int, default=2000, help="Number of frames per processing chunk")
    parser.add_argument("--voltage-kv", type=float, default=None, help="Override electron accelerating voltage")
    parser.add_argument(
        "--beam-direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("BX", "BY", "BZ"),
        help="Override beam direction in laboratory coordinates",
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    return parser


def run_from_cli(argv: list[str] | None = None) -> PipelineResults:
    """Entry point used by the example command-line script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    snapshot_mode = bool(args.orientations and args.reflections and args.cell)
    xds_mode = bool(args.gxparm and args.integrate)
    if snapshot_mode == xds_mode:
        raise SystemExit(
            "Select exactly one mode: either --orientations/--reflections/--cell or --gxparm/--integrate."
        )

    config = PipelineConfig(
        dmin=args.dmin,
        dmax=args.dmax,
        sg_threshold=args.sg_threshold,
        zone_axis_limit=args.zone_axis_limit,
        hkl_limit=args.hkl_limit,
        max_candidate_reflections=args.max_candidate_reflections,
        alpha=args.alpha,
        filter_threshold=args.filter_threshold,
        xi_scale_nm=args.xi_scale_nm,
        structure_factor_decay=args.structure_factor_decay,
        zone_axis_soft_angle_deg=args.zone_axis_soft_angle_deg,
        combined_formulation=args.combined_formulation,
        zone_axis_weight=args.zone_axis_weight,
        n_workers=args.n_workers,
        chunk_size_frames=args.chunk_size_frames,
        voltage_kv=args.voltage_kv,
        beam_direction=tuple(args.beam_direction) if args.beam_direction is not None else None,
    )

    if snapshot_mode:
        results = run_serialed_csv_pipeline(
            orientations_path=args.orientations,
            reflections_path=args.reflections,
            cell_path=args.cell,
            config=config,
        )
    else:
        results = run_xds_pipeline(
            gxparm_path=args.gxparm,
            integrate_path=args.integrate,
            xds_inp_path=args.xds_inp,
            config=config,
        )

    written = export_results(results, args.output)
    print("Wrote pipeline outputs:")
    for label, path in written.items():
        print(f"  {label}: {path}")
    print(f"Frames: {len(results.frame_summary)}")
    print(f"Reflections: {len(results.reflection_table)}")
    return results
