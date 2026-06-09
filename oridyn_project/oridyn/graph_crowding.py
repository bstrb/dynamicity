"""Sparse reciprocal-space graph crowding terms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OridynConfig
from .excitation import compute_excited_candidate_nodes
from .geometry import low_order_prior_from_q
from .stream_parser import StreamData, reciprocal_matrix_from_row


def _delta_prior(delta_hkl: np.ndarray, config: OridynConfig) -> np.ndarray:
    delta_norm = np.linalg.norm(delta_hkl.astype(float), axis=1)
    # Integer-space prior: small difference vectors are favored. The scale is
    # deliberately separate from reciprocal length because this term is local in
    # HKL graph space.
    return 1.0 / (1.0 + (delta_norm / 1.5) ** config.low_order_power)


def add_graph_crowding_terms(
    scores: pd.DataFrame,
    stream: StreamData,
    candidates: pd.DataFrame,
    config: OridynConfig,
) -> pd.DataFrame:
    """Add sparse graph-crowding metrics using excited candidate beams."""

    if scores.empty:
        return scores.copy()
    out_frames: list[pd.DataFrame] = []
    for frame, group in scores.groupby("frame", sort=True):
        crystal_row = stream.crystal_table.loc[stream.crystal_table["frame"] == int(frame)].iloc[0]
        nodes = compute_excited_candidate_nodes(
            reciprocal_matrix_from_row(crystal_row), candidates, stream.wavelength_angstrom, config
        )
        node_hkl = nodes[["h", "k", "l"]].to_numpy(dtype=int) if not nodes.empty else np.empty((0, 3), dtype=int)
        node_weight = nodes["excitation_weight"].to_numpy(dtype=float) if not nodes.empty else np.empty(0)

        rows: list[dict[str, float | str]] = []
        for target in group.itertuples(index=False):
            if nodes.empty:
                rows.append(_empty_graph_row())
                continue
            target_hkl = np.asarray([target.h, target.k, target.l], dtype=int)
            delta = node_hkl - target_hkl[None, :]
            mask = np.max(np.abs(delta), axis=1) <= config.neighbor_hkl_radius
            mask &= np.any(delta != 0, axis=1)
            if not np.any(mask):
                rows.append(_empty_graph_row())
                continue
            neighbor_delta = delta[mask]
            neighbor_weight = node_weight[mask]
            edge_weight = neighbor_weight * _delta_prior(neighbor_delta, config)
            order = np.argsort(edge_weight)[::-1]
            if len(order) > config.max_neighbors_per_reflection:
                order = order[: config.max_neighbors_per_reflection]
            edge_weight = edge_weight[order]
            selected_hkl = node_hkl[mask][order]
            selected_excitation = neighbor_weight[order]
            edge_sum = float(np.sum(edge_weight))
            edge_sq = float(np.sum(edge_weight**2))
            effective = 0.0 if edge_sq <= 0.0 else (edge_sum * edge_sum) / edge_sq
            summary = ";".join(
                f"{int(h)},{int(k)},{int(l)}:{w:.3g}"
                for (h, k, l), w in zip(selected_hkl[:3], edge_weight[:3], strict=False)
            )
            rows.append(
                {
                    "graph_crowding_raw": float(np.log1p(edge_sum)),
                    "sum_neighbor_excitation": float(np.sum(selected_excitation)),
                    "effective_neighbor_count": effective,
                    "max_neighbor_edge_weight": float(np.max(edge_weight)) if len(edge_weight) else 0.0,
                    "top_neighbor_summary": summary,
                }
            )
        out_frames.append(pd.concat([group.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1))
    return pd.concat(out_frames, ignore_index=True)


def _empty_graph_row() -> dict[str, float | str]:
    return {
        "graph_crowding_raw": 0.0,
        "sum_neighbor_excitation": 0.0,
        "effective_neighbor_count": 0.0,
        "max_neighbor_edge_weight": 0.0,
        "top_neighbor_summary": "",
    }
