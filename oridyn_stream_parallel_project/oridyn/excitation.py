"""Reflection excitation scoring at observed or candidate HKLs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OridynConfig
from .geometry import d_spacings_from_q, excitation_error, excitation_weight, hkl_lab_vectors, vector_norms
from .stream_parser import StreamData, reciprocal_matrix_from_row


def compute_excited_candidate_nodes(
    reciprocal_matrix: np.ndarray,
    candidates: pd.DataFrame,
    wavelength_angstrom: float,
    config: OridynConfig,
) -> pd.DataFrame:
    """Return a capped table of softly excited candidate beams for one frame."""

    if candidates.empty:
        return pd.DataFrame(columns=["h", "k", "l", "sg", "excitation_weight"])
    hkls = candidates[["h", "k", "l"]].to_numpy(dtype=int)
    g = hkl_lab_vectors(hkls, reciprocal_matrix)
    sg = excitation_error(g, wavelength_angstrom, config.beam_direction)
    weights = excitation_weight(sg, config.sg0, config.excitation_kernel, config.excitation_lorentzian_power)
    nodes = candidates[["h", "k", "l", "q_invA", "d_angstrom"]].copy()
    nodes["sg"] = sg
    nodes["excitation_weight"] = weights
    nodes = nodes[nodes["excitation_weight"] >= config.neighbor_excitation_min]
    nodes = nodes.sort_values("excitation_weight", ascending=False)
    if len(nodes) > config.max_excited_nodes_per_frame:
        nodes = nodes.head(config.max_excited_nodes_per_frame)
    return nodes.reset_index(drop=True)


def score_observed_reflections(stream: StreamData, config: OridynConfig) -> pd.DataFrame:
    """Score the default target set: observed/indexed HKLs from the stream."""

    observed = stream.reflections.copy()
    if observed.empty:
        return pd.DataFrame()

    keep = [
        col
        for col in (
            "frame",
            "frame_number",
            "chunk_id",
            "crystal_in_chunk",
            "event",
            "image_serial",
            "h",
            "k",
            "l",
            "sigma",
            "fs_px",
            "ss_px",
            "panel",
        )
        if col in observed.columns
    ]
    out_frames: list[pd.DataFrame] = []
    for frame, group in observed[keep].groupby("frame", sort=True):
        crystal_row = stream.crystal_table.loc[stream.crystal_table["frame"] == int(frame)].iloc[0]
        reciprocal = reciprocal_matrix_from_row(crystal_row)
        hkls = group[["h", "k", "l"]].to_numpy(dtype=int)
        g = hkl_lab_vectors(hkls, reciprocal)
        q = vector_norms(g)
        sg = excitation_error(g, stream.wavelength_angstrom, config.beam_direction)
        weights = excitation_weight(sg, config.sg0, config.excitation_kernel, config.excitation_lorentzian_power)
        scored = group.copy()
        scored["target_source"] = "observed"
        scored["q_invA"] = q
        scored["d_angstrom"] = d_spacings_from_q(q)
        scored["sg"] = sg
        scored["excitation_weight"] = weights
        scored["excitation_center"] = weights
        scored["excitation_mean"] = weights
        scored["excitation_max"] = weights
        scored["excitation_integrated"] = weights
        out_frames.append(scored)

    result = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    result.insert(0, "reflection_id", np.arange(len(result), dtype=int))
    return result


def score_candidate_reflections(stream: StreamData, candidates: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    """Optionally export center-orientation excitation for all candidate beams."""

    rows: list[pd.DataFrame] = []
    for _, crystal_row in stream.crystal_table.iterrows():
        reciprocal = reciprocal_matrix_from_row(crystal_row)
        hkls = candidates[["h", "k", "l"]].to_numpy(dtype=int)
        g = hkl_lab_vectors(hkls, reciprocal)
        q = vector_norms(g)
        sg = excitation_error(g, stream.wavelength_angstrom, config.beam_direction)
        weights = excitation_weight(sg, config.sg0, config.excitation_kernel, config.excitation_lorentzian_power)
        frame_df = candidates[["h", "k", "l", "q_invA", "d_angstrom"]].copy()
        frame_df.insert(0, "frame", int(crystal_row["frame"]))
        frame_df.insert(1, "frame_number", int(crystal_row["frame_number"]))
        frame_df["sg"] = sg
        frame_df["q_frame_invA"] = q
        frame_df["excitation_weight"] = weights
        rows.append(frame_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
