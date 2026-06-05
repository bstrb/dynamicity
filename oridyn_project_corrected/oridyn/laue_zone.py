"""Laue-zone classification and crowding terms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OridynConfig
from .excitation import compute_excited_candidate_nodes
from .geometry import parse_triplet_label, triplet_label
from .stream_parser import StreamData, reciprocal_matrix_from_row


def add_laue_zone_terms(
    scores: pd.DataFrame,
    frame_scores: pd.DataFrame,
    stream: StreamData,
    candidates: pd.DataFrame,
    config: OridynConfig,
) -> pd.DataFrame:
    """Add Laue-zone index, masks, and same-zone local crowding."""

    if scores.empty:
        return scores.copy()
    frame_axis = frame_scores.set_index("frame")["assigned_risky_axis"].to_dict()
    out_frames: list[pd.DataFrame] = []
    for frame, group in scores.groupby("frame", sort=True):
        label = str(frame_axis.get(int(frame), "[0 0 1]"))
        axis = parse_triplet_label(label)
        uvw = np.asarray(axis, dtype=int)
        crystal_row = stream.crystal_table.loc[stream.crystal_table["frame"] == int(frame)].iloc[0]
        nodes = compute_excited_candidate_nodes(
            reciprocal_matrix_from_row(crystal_row), candidates, stream.wavelength_angstrom, config
        )
        node_hkl = nodes[["h", "k", "l"]].to_numpy(dtype=int) if not nodes.empty else np.empty((0, 3), dtype=int)
        node_weight = nodes["excitation_weight"].to_numpy(dtype=float) if not nodes.empty else np.empty(0)
        node_laue = node_hkl @ uvw if len(node_hkl) else np.empty(0, dtype=int)

        rows: list[dict[str, float | int | bool | str]] = []
        for target in group.itertuples(index=False):
            hkl = np.asarray([target.h, target.k, target.l], dtype=int)
            laue_n = int(hkl @ uvw)
            abs_n = abs(laue_n)
            same_sum = float(np.sum(node_weight[node_laue == laue_n])) if len(node_laue) else 0.0
            zone_weight = 1.0 / (1.0 + abs_n)
            rows.append(
                {
                    "assigned_zone_axis": triplet_label(axis),
                    "laue_n": laue_n,
                    "abs_laue_n": abs_n,
                    "is_zolz": laue_n == 0,
                    "is_folz": abs_n == 1,
                    "near_zone_law": abs_n <= 1,
                    "same_laue_zone_crowding_raw": float(np.log1p(same_sum)),
                    "laue_zone_risk_raw": float(np.log1p(same_sum) * zone_weight),
                }
            )
        out_frames.append(pd.concat([group.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1))
    return pd.concat(out_frames, ignore_index=True)
