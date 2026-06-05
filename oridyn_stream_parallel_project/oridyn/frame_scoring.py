"""Frame-level orientation and excitation scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .axis_prediction import unique_zone_axes
from .config import OridynConfig
from .excitation import compute_excited_candidate_nodes
from .geometry import axis_angle_deg, beam_in_direct_coordinates, triplet_label
from .stream_parser import StreamData, reciprocal_matrix_from_row


def compute_frame_scores(
    stream: StreamData,
    problematic_axes: pd.DataFrame,
    candidates: pd.DataFrame,
    config: OridynConfig,
) -> pd.DataFrame:
    """Compute frame-level orientation and candidate excitation metrics."""

    risky_axes = [
        (
            (int(row.u), int(row.v), int(row.w)),
            float(row.axis_score),
            str(row.axis_label),
            int(row.axis_rank) if hasattr(row, "axis_rank") else -1,
        )
        for row in problematic_axes.itertuples(index=False)
    ]
    nearest_axes = unique_zone_axes(config.uvw_max)
    q_span = float(candidates["q_invA"].max() - candidates["q_invA"].min()) if len(candidates) > 1 else 1.0
    rows: list[dict[str, float | int | str]] = []
    for _, crystal_row in stream.crystal_table.iterrows():
        reciprocal = reciprocal_matrix_from_row(crystal_row)
        best_axis = (0, 0, 1)
        best_label = "[0 0 1]"
        best_angle = 180.0
        best_axis_score = 0.0
        best_axis_rank = -1
        best_risk = 0.0
        for axis, axis_score, label, axis_rank in risky_axes:
            angle = axis_angle_deg(reciprocal, axis, config.beam_direction)
            closeness = float(np.exp(-((angle / max(config.axis_sigma_deg, 1e-12)) ** 2)))
            risk = axis_score * closeness
            if risk > best_risk:
                best_axis = axis
                best_label = label
                best_angle = angle
                best_axis_score = axis_score
                best_axis_rank = axis_rank
                best_risk = risk

        nearest_axis = (0, 0, 1)
        nearest_angle = 180.0
        for axis in nearest_axes:
            angle = axis_angle_deg(reciprocal, axis, config.beam_direction)
            if angle < nearest_angle:
                nearest_axis = axis
                nearest_angle = angle

        nodes = compute_excited_candidate_nodes(reciprocal, candidates, stream.wavelength_angstrom, config)
        sum_weight = float(nodes["excitation_weight"].sum()) if not nodes.empty else 0.0
        beam_coords = beam_in_direct_coordinates(reciprocal, config.beam_direction)
        rows.append(
            {
                "frame": int(crystal_row["frame"]),
                "frame_number": int(crystal_row["frame_number"]),
                "event": str(crystal_row.get("event", "")),
                "assigned_risky_axis": best_label,
                "assigned_axis_angle_deg": best_angle,
                "assigned_axis_score": best_axis_score,
                "assigned_axis_rank": best_axis_rank,
                "frame_axis_risk_raw": best_risk,
                "nearest_zone_axis": triplet_label(nearest_axis),
                "nearest_zone_axis_angle_deg": nearest_angle,
                "axis_match": best_label == triplet_label(nearest_axis),
                "axis_angle_delta_deg": best_angle - nearest_angle,
                "beam_direct_u": float(beam_coords[0]),
                "beam_direct_v": float(beam_coords[1]),
                "beam_direct_w": float(beam_coords[2]),
                "n_excited": int(len(nodes)),
                "sum_excitation_weight": sum_weight,
                "excitation_density": float(len(nodes) / max(len(candidates), 1)),
                "resolution_normalized_excitation_density": sum_weight / max(q_span, 1e-12),
                "assigned_risky_axis_u": best_axis[0],
                "assigned_risky_axis_v": best_axis[1],
                "assigned_risky_axis_w": best_axis[2],
            }
        )
    return pd.DataFrame.from_records(rows)
