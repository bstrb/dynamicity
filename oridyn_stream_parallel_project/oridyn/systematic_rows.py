"""Systematic-row proxy metrics in HKL graph space."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import OridynConfig
from .excitation import compute_excited_candidate_nodes
from .geometry import canonical_triplet, gcd3, triplet_label
from .stream_parser import StreamData, reciprocal_matrix_from_row


def add_systematic_row_terms(
    scores: pd.DataFrame,
    stream: StreamData,
    candidates: pd.DataFrame,
    config: OridynConfig,
) -> pd.DataFrame:
    """Add first-version row metrics based on primitive HKL directions."""

    if scores.empty:
        return scores.copy()
    out_frames: list[pd.DataFrame] = []
    for frame, group in scores.groupby("frame", sort=True):
        crystal_row = stream.crystal_table.loc[stream.crystal_table["frame"] == int(frame)].iloc[0]
        nodes = compute_excited_candidate_nodes(
            reciprocal_matrix_from_row(crystal_row), candidates, stream.wavelength_angstrom, config
        )
        row_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        row_sums: dict[tuple[int, int, int], float] = defaultdict(float)
        for node in nodes.itertuples(index=False):
            h, k, l = int(node.h), int(node.k), int(node.l)
            step = gcd3(h, k, l)
            if step == 0 or step > config.row_max_steps:
                continue
            direction = canonical_triplet(h, k, l)
            if max(abs(x) for x in direction) > config.row_direction_limit:
                continue
            row_counts[direction] += 1
            row_sums[direction] += float(node.excitation_weight)

        rows: list[dict[str, float | int | str]] = []
        for target in group.itertuples(index=False):
            direction = canonical_triplet(int(target.h), int(target.k), int(target.l))
            count = int(row_counts.get(direction, 0))
            summed = float(row_sums.get(direction, 0.0))
            direction_norm = float(np.linalg.norm(np.asarray(direction, dtype=float)))
            low_order_row = 1.0 / (1.0 + (direction_norm / 1.5) ** config.low_order_power)
            rows.append(
                {
                    "nearest_row_direction": triplet_label(direction, brackets="()"),
                    "row_excited_count": count,
                    "row_excitation_sum": summed,
                    "systematic_row_risk_raw": float(np.log1p(summed) * low_order_row),
                }
            )
        out_frames.append(pd.concat([group.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1))
    return pd.concat(out_frames, ignore_index=True)
