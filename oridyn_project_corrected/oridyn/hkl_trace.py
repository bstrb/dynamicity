"""Selected-HKL trajectory scoring across all indexed frames.

This is the exploration layer for OriDyn. It scores a user-provided list of HKLs
across every indexed frame, regardless of whether each HKL was observed in that
frame. The heavy geometry/environment terms are computed once, and the resulting
``hkl_frame_trajectories.csv`` can then be reweighted repeatedly without
rerunning the stream parser or graph-crowding calculations.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .axis_prediction import mark_active_problematic_axes, predict_problematic_axes
from .config import OridynConfig
from .geometry import orientation_checks
from .hkl_generation import generate_candidate_hkls
from .hkl_plots import make_hkl_trace_plots
from .normalization import normalize_frame_terms, normalize_reflection_terms
from .outputs import summarize_score_terms, write_outputs
from .parallel_scoring import log_progress, score_frames_and_reflections
from .plots import make_standard_plots
from .sigma_model import add_sigma_model, combine_score_terms
from .stream_parser import StreamData, parse_crystfel_stream


def load_hkl_targets(path: str | Path) -> pd.DataFrame:
    """Load a selected-HKL CSV/TSV/whitespace table.

    Required columns are ``h``, ``k``, and ``l``. An optional ``label`` or
    ``hkl_label`` column is used in plots. Duplicate HKLs are removed while
    preserving the first label.
    """

    input_path = Path(path)
    dtype = {"label": "string", "hkl_label": "string"}
    if input_path.suffix.lower() in {".tsv", ".tab"}:
        table = pd.read_csv(input_path, sep="\t", dtype=dtype)
    elif input_path.suffix.lower() == ".csv":
        table = pd.read_csv(input_path, dtype=dtype)
    else:
        table = pd.read_csv(input_path, sep=None, engine="python", dtype=dtype)
    required = {"h", "k", "l"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Selected-HKL table is missing required columns: {missing}")
    out = table.copy()
    for col in ("h", "k", "l"):
        out[col] = out[col].astype(int)
    if "hkl_label" not in out.columns:
        if "label" in out.columns:
            out["hkl_label"] = out["label"].astype(str)
        else:
            out["hkl_label"] = "(" + out["h"].astype(str) + " " + out["k"].astype(str) + " " + out["l"].astype(str) + ")"
    out["hkl_label"] = out["hkl_label"].fillna("").astype(str)
    out = out.drop_duplicates(subset=["h", "k", "l"], keep="first").reset_index(drop=True)
    out.insert(0, "target_index", np.arange(len(out), dtype=int))
    return out[["target_index", "h", "k", "l", "hkl_label"]]


def build_hkl_target_table(stream: StreamData, targets: pd.DataFrame) -> pd.DataFrame:
    """Create the frame-by-HKL target table used by the scoring engine."""

    rows: list[dict[str, object]] = []
    for crystal in stream.crystal_table.sort_values("frame").itertuples(index=False):
        crystal_payload = {
            "frame": int(crystal.frame),
            "frame_number": int(crystal.frame_number),
            "chunk_id": int(getattr(crystal, "chunk_id", -1)),
            "crystal_in_chunk": int(getattr(crystal, "crystal_in_chunk", -1)),
            "event": str(getattr(crystal, "event", "")),
            "image_serial": int(getattr(crystal, "image_serial", -1)),
        }
        for target in targets.itertuples(index=False):
            row = dict(crystal_payload)
            row.update(
                {
                    "target_index": int(target.target_index),
                    "target_source": "selected_hkl",
                    "h": int(target.h),
                    "k": int(target.k),
                    "l": int(target.l),
                    "hkl_label": str(target.hkl_label),
                }
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def run_hkl_trace(
    stream_path: str | Path,
    hkl_path: str | Path,
    output_dir: str | Path,
    config: OridynConfig,
) -> dict[str, object]:
    """Score selected HKLs across all indexed frames and write trajectory outputs."""

    log_progress(f"parsing stream: {stream_path}", config.progress)
    stream = parse_crystfel_stream(stream_path)
    targets = load_hkl_targets(hkl_path)
    log_progress(f"loaded {len(targets)} selected HKL target(s)", config.progress)
    target_reflections = build_hkl_target_table(stream, targets)
    trace_stream = StreamData(
        path=stream.path,
        wavelength_angstrom=stream.wavelength_angstrom,
        unit_cell=stream.unit_cell,
        crystal_table=stream.crystal_table,
        reflections=target_reflections,
        detector=stream.detector,
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
    log_progress(f"candidate HKLs: {candidate_metadata.get('n_candidates', len(candidates))}", config.progress)
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
    log_progress(f"using {len(active_axes)} predicted axis/axes for trajectory scoring", config.progress)

    # Selected-HKL trajectories can be very small per frame. Rank or percentile
    # normalization is often clearer for exploration, but we respect the user's
    # config and only compute the score terms once.
    frame_scores, trajectory_scores, scoring_metadata = score_frames_and_reflections(
        trace_stream, active_axes, candidates, config
    )
    frame_scores, frame_norm_metadata = normalize_frame_terms(frame_scores, config)
    trajectory_scores, reflection_norm_metadata = normalize_reflection_terms(trajectory_scores, config)
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
    trajectory_scores = trajectory_scores.merge(frame_scores[frame_columns], on="frame", how="left")
    trajectory_scores = combine_score_terms(trajectory_scores, config)
    trajectory_scores = add_sigma_model(trajectory_scores, config)
    frame_scores = _add_hkl_score_summaries(frame_scores, trajectory_scores)
    score_terms_summary = summarize_score_terms(trajectory_scores)

    metadata = {
        "oridyn_version": __version__,
        "mode": "selected_hkl_trace",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stream_path": str(stream_path),
        "hkl_path": str(hkl_path),
        "config": config.to_dict(),
        "unit_cell": stream.unit_cell.__dict__,
        "wavelength_angstrom": stream.wavelength_angstrom,
        "n_frames": int(len(stream.crystal_table)),
        "n_selected_hkls": int(len(targets)),
        "n_trajectory_rows": int(len(trajectory_scores)),
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
            "target_reflections": "user-selected HKLs scored across every indexed frame",
            "intensity_usage": "observed intensities, peaks, and backgrounds are not used in S_dyn_geom",
            "detector_usage": "detector coordinates are not used in S_dyn_geom",
            "reweighting": "normalized term columns can be recombined later without rerunning geometry",
        },
    }

    out = Path(output_dir)
    write_outputs(
        out,
        axes,
        frame_scores,
        trajectory_scores,
        score_terms_summary,
        metadata,
        candidate_scores=None,
        information_summaries=None,
    )
    trajectory_scores.to_csv(out / "hkl_frame_trajectories.csv", index=False)
    targets.to_csv(out / "selected_hkls.csv", index=False)
    make_standard_plots(trajectory_scores, frame_scores, out / "plots")
    make_hkl_trace_plots(trajectory_scores, out / "plots")
    log_progress(f"done: {out}", config.progress)
    return metadata


def _add_hkl_score_summaries(frame_scores: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty or "S_dyn_geom" not in scores:
        out = frame_scores.copy()
        out["n_selected_hkl_targets"] = 0
        out["mean_selected_hkl_S_dyn_geom"] = 0.0
        out["p95_selected_hkl_S_dyn_geom"] = 0.0
        out["mean_selected_hkl_sigma_dyn_rel"] = 1.0
        return out
    grouped = scores.groupby("frame").agg(
        n_selected_hkl_targets=("S_dyn_geom", "size"),
        mean_selected_hkl_S_dyn_geom=("S_dyn_geom", "mean"),
        p95_selected_hkl_S_dyn_geom=("S_dyn_geom", lambda x: float(x.quantile(0.95))),
        mean_selected_hkl_sigma_dyn_rel=("sigma_dyn_rel", "mean"),
    )
    return frame_scores.merge(grouped, left_on="frame", right_index=True, how="left").fillna(
        {
            "n_selected_hkl_targets": 0,
            "mean_selected_hkl_S_dyn_geom": 0.0,
            "p95_selected_hkl_S_dyn_geom": 0.0,
            "mean_selected_hkl_sigma_dyn_rel": 1.0,
        }
    )
