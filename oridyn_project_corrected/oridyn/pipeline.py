"""End-to-end OriDyn pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import __version__
from .axis_prediction import mark_active_problematic_axes, predict_problematic_axes
from .config import OridynConfig
from .excitation import score_candidate_reflections
from .geometry import orientation_checks
from .hkl_generation import generate_candidate_hkls
from .normalization import normalize_frame_terms, normalize_reflection_terms
from .outputs import summarize_score_terms, write_outputs
from .parallel_scoring import log_progress, score_frames_and_reflections
from .plots import make_standard_plots
from .sigma_model import add_sigma_model, combine_score_terms
from .stream_parser import parse_crystfel_stream
from .summaries import make_information_summaries


def run_pipeline(stream_path: str | Path, output_dir: str | Path, config: OridynConfig) -> dict[str, object]:
    """Run the complete first-version geometry-only scoring workflow."""

    log_progress(f"parsing stream: {stream_path}", config.progress)
    stream = parse_crystfel_stream(stream_path)
    log_progress(
        f"parsed {len(stream.crystal_table)} frame(s) and {len(stream.reflections)} observed reflection(s)",
        config.progress,
    )
    log_progress(f"generating candidate HKLs for {config.dmin:g} <= d <= {config.dmax:g} A", config.progress)
    candidates, candidate_metadata = generate_candidate_hkls(
        stream.unit_cell,
        config.dmin,
        config.dmax,
        hkl_limit=config.hkl_limit,
        max_candidates=config.max_candidates,
        centering=config.centering or stream.unit_cell.centering,
    )
    if candidate_metadata.get("truncated"):
        log_progress(
            f"candidate HKLs: {candidate_metadata['n_candidates']} kept after cap "
            f"({candidate_metadata['n_candidates_before_cap']} before cap)",
            config.progress,
        )
    else:
        log_progress(f"candidate HKLs: {candidate_metadata['n_candidates']}", config.progress)
    log_progress(f"predicting problematic axes up to uvw_max={config.uvw_max}", config.progress)
    axes = predict_problematic_axes(
        candidates,
        uvw_max=config.uvw_max,
        low_order_g0_invA=config.low_order_g0_invA,
        low_order_power=config.low_order_power,
    )
    axes = mark_active_problematic_axes(
        axes,
        max_problematic_axes=config.max_problematic_axes,
        axis_score_min=config.axis_score_min,
    )
    active_axes = axes.loc[axes["used_for_scoring"]].reset_index(drop=True)
    log_progress(
        f"using {len(active_axes)} of {len(axes)} predicted axes for scoring",
        config.progress,
    )
    log_progress("scoring frames and observed reflections", config.progress)
    frame_scores, reflection_scores, scoring_metadata = score_frames_and_reflections(stream, active_axes, candidates, config)
    log_progress("normalizing score terms", config.progress)
    frame_scores, frame_norm_metadata = normalize_frame_terms(frame_scores, config)
    reflection_scores, reflection_norm_metadata = normalize_reflection_terms(reflection_scores, config)
    frame_columns = [
        "frame",
        "assigned_risky_axis",
        "assigned_axis_rank",
        "assigned_axis_angle_deg",
        "nearest_zone_axis",
        "nearest_zone_axis_angle_deg",
        "axis_match",
        "axis_angle_delta_deg",
        "frame_axis_risk_raw",
        "frame_axis_risk_norm",
    ]
    reflection_scores = reflection_scores.merge(frame_scores[frame_columns], on="frame", how="left")
    reflection_scores = combine_score_terms(reflection_scores, config)
    reflection_scores = add_sigma_model(reflection_scores, config)

    frame_scores = _add_reflection_score_summaries(frame_scores, reflection_scores)
    score_terms_summary = summarize_score_terms(reflection_scores)
    log_progress("building information summaries", config.progress)
    information_summaries = make_information_summaries(
        frame_scores,
        reflection_scores,
        axis_sigma_deg=config.axis_sigma_deg,
    )
    log_progress("exporting candidate scores" if config.export_candidates else "skipping candidate score export", config.progress)
    candidate_scores = score_candidate_reflections(stream, candidates, config) if config.export_candidates else None

    metadata = {
        "oridyn_version": __version__,
        "stream_path": str(stream_path),
        "config": config.to_dict(),
        "unit_cell": stream.unit_cell.__dict__,
        "wavelength_angstrom": stream.wavelength_angstrom,
        "n_frames": int(len(stream.crystal_table)),
        "n_observed_reflections": int(len(stream.reflections)),
        "candidate_generation": candidate_metadata,
        "problematic_axes": {
            "n_predicted_axes": int(len(axes)),
            "n_axes_used_for_scoring": int(len(active_axes)),
            "max_problematic_axes": config.max_problematic_axes,
            "axis_score_min": config.axis_score_min,
        },
        "scoring": scoring_metadata,
        "normalization": {
            "reflection_terms": reflection_norm_metadata,
            "frame_terms": frame_norm_metadata,
        },
        "orientation_checks": orientation_checks(stream.crystal_table),
        "score_policy": {
            "target_reflections": "observed/indexed HKLs in the stream",
            "intensity_usage": "observed intensities, peaks, and backgrounds are not used in S_dyn_geom",
            "detector_usage": "detector coordinates are not used in S_dyn_geom",
            "residual_usage": "external residuals are only plotted by a separate command",
        },
    }

    write_outputs(
        output_dir,
        axes,
        frame_scores,
        reflection_scores,
        score_terms_summary,
        metadata,
        candidate_scores=candidate_scores,
        information_summaries=information_summaries,
    )
    log_progress("making plots", config.progress)
    make_standard_plots(reflection_scores, frame_scores, Path(output_dir) / "plots")
    log_progress(f"done: {output_dir}", config.progress)
    return metadata


def run_axes(stream_path: str | Path, output_dir: str | Path, config: OridynConfig) -> pd.DataFrame:
    """Run only candidate generation and cell-mode axis prediction."""

    log_progress(f"parsing stream: {stream_path}", config.progress)
    stream = parse_crystfel_stream(stream_path)
    log_progress(f"generating candidate HKLs for {config.dmin:g} <= d <= {config.dmax:g} A", config.progress)
    candidates, candidate_metadata = generate_candidate_hkls(
        stream.unit_cell,
        config.dmin,
        config.dmax,
        hkl_limit=config.hkl_limit,
        max_candidates=config.max_candidates,
        centering=config.centering or stream.unit_cell.centering,
    )
    log_progress(f"predicting problematic axes up to uvw_max={config.uvw_max}", config.progress)
    axes = predict_problematic_axes(candidates, config.uvw_max, config.low_order_g0_invA, config.low_order_power)
    axes = mark_active_problematic_axes(
        axes,
        max_problematic_axes=config.max_problematic_axes,
        axis_score_min=config.axis_score_min,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    axes.to_csv(output / "problematic_axes.csv", index=False)
    metadata = {
        "oridyn_version": __version__,
        "stream_path": str(stream_path),
        "config": config.to_dict(),
        "candidate_generation": candidate_metadata,
        "problematic_axes": {
            "n_predicted_axes": int(len(axes)),
            "n_axes_used_for_scoring": int(axes["used_for_scoring"].sum()),
            "max_problematic_axes": config.max_problematic_axes,
            "axis_score_min": config.axis_score_min,
        },
        "orientation_checks": orientation_checks(stream.crystal_table),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    log_progress(f"done: {output}", config.progress)
    return axes


def _add_reflection_score_summaries(frame_scores: pd.DataFrame, reflection_scores: pd.DataFrame) -> pd.DataFrame:
    if reflection_scores.empty or "S_dyn_geom" not in reflection_scores:
        out = frame_scores.copy()
        out["n_observed_targets"] = 0
        out["mean_S_dyn_geom"] = 0.0
        out["p95_S_dyn_geom"] = 0.0
        out["mean_sigma_dyn_rel"] = 1.0
        return out
    grouped = reflection_scores.groupby("frame").agg(
        n_observed_targets=("S_dyn_geom", "size"),
        mean_S_dyn_geom=("S_dyn_geom", "mean"),
        p95_S_dyn_geom=("S_dyn_geom", lambda x: float(x.quantile(0.95))),
        mean_sigma_dyn_rel=("sigma_dyn_rel", "mean"),
    )
    return frame_scores.merge(grouped, left_on="frame", right_index=True, how="left").fillna(
        {"n_observed_targets": 0, "mean_S_dyn_geom": 0.0, "p95_S_dyn_geom": 0.0, "mean_sigma_dyn_rel": 1.0}
    )
