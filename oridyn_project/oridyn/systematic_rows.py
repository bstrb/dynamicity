"""Affine row-crowding proxy metrics in HKL graph space."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import OridynConfig
from .excitation import compute_excited_candidate_nodes
from .geometry import canonical_triplet, gcd3, iter_primitive_triplets, triplet_label
from .stream_parser import StreamData, reciprocal_matrix_from_row


def add_systematic_row_terms(
    scores: pd.DataFrame,
    stream: StreamData,
    candidates: pd.DataFrame,
    config: OridynConfig,
) -> pd.DataFrame:
    """Add affine row-crowding terms for each observed reflection."""

    if scores.empty:
        return scores.copy()
    out_frames: list[pd.DataFrame] = []
    for frame, group in scores.groupby("frame", sort=True):
        crystal_row = stream.crystal_table.loc[stream.crystal_table["frame"] == int(frame)].iloc[0]
        nodes = compute_excited_candidate_nodes(
            reciprocal_matrix_from_row(crystal_row), candidates, stream.wavelength_angstrom, config
        )
        out_frames.append(add_affine_row_terms_from_nodes(group, nodes, config))
    return pd.concat(out_frames, ignore_index=True)


def add_affine_row_terms_from_nodes(
    scores: pd.DataFrame,
    nodes: pd.DataFrame,
    config: OridynConfig,
) -> pd.DataFrame:
    """Add local affine row-crowding terms using excited candidate nodes."""

    if scores.empty:
        return scores.copy()

    node_hkl = nodes[["h", "k", "l"]].to_numpy(dtype=int) if not nodes.empty else np.empty((0, 3), dtype=int)
    node_weight = nodes["excitation_weight"].to_numpy(dtype=float) if not nodes.empty else np.empty(0)
    allowed_directions = set(_row_directions(int(config.row_direction_limit)))

    rows: list[dict[str, float | int | str]] = []
    for target in scores.itertuples(index=False):
        target_hkl = np.asarray([target.h, target.k, target.l], dtype=int)
        rows.append(_score_best_affine_row(target_hkl, node_hkl, node_weight, allowed_directions, config))

    return pd.concat([scores.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1)


@lru_cache(maxsize=None)
def _row_directions(limit: int) -> tuple[tuple[int, int, int], ...]:
    """Return sign-canonical primitive row directions up to ``limit``."""

    return tuple(iter_primitive_triplets(max(int(limit), 0)))


def _score_best_affine_row(
    target_hkl: np.ndarray,
    node_hkl: np.ndarray,
    node_weight: np.ndarray,
    allowed_directions: set[tuple[int, int, int]],
    config: OridynConfig,
) -> dict[str, float | int | str]:
    if node_hkl.size == 0 or not allowed_directions:
        return _empty_affine_row()

    delta = node_hkl - target_hkl[None, :]
    abs_delta = np.abs(delta)
    nearby = np.max(abs_delta, axis=1) <= int(config.row_max_steps)
    nearby &= np.any(delta != 0, axis=1)
    if not np.any(nearby):
        return _empty_affine_row()

    nearby_delta = delta[nearby]
    nearby_weight = node_weight[nearby]
    total_neighbor_excitation = float(np.sum(nearby_weight))
    if total_neighbor_excitation <= 0.0:
        return _empty_affine_row()

    stats: dict[tuple[int, int, int], dict[str, float | int | bool]] = {}
    for step_delta, weight in zip(nearby_delta, nearby_weight, strict=False):
        step = gcd3(int(step_delta[0]), int(step_delta[1]), int(step_delta[2]))
        if step == 0 or step > int(config.row_max_steps):
            continue
        direction = canonical_triplet(int(step_delta[0]), int(step_delta[1]), int(step_delta[2]))
        if direction not in allowed_directions:
            continue
        signed_step = _signed_step(step_delta, direction)
        if signed_step == 0:
            continue

        row = stats.setdefault(
            direction,
            {"count": 0, "sum": 0.0, "proximity_sum": 0.0, "has_positive": False, "has_negative": False},
        )
        value = float(weight)
        row["count"] = int(row["count"]) + 1
        row["sum"] = float(row["sum"]) + value
        row["proximity_sum"] = float(row["proximity_sum"]) + value / abs(float(signed_step))
        if signed_step > 0:
            row["has_positive"] = True
        else:
            row["has_negative"] = True

    if not stats:
        return _empty_affine_row()

    best_direction = (0, 0, 0)
    best_row: dict[str, float | int | bool] | None = None
    best_risk = -1.0
    best_factors = (0.0, 0.0, 0.0)
    for direction, row in stats.items():
        row_sum = float(row["sum"])
        if row_sum <= 0.0:
            continue
        direction_norm = float(np.linalg.norm(np.asarray(direction, dtype=float)))
        low_index_weight = 1.0 / (1.0 + (direction_norm / 1.5) ** float(config.low_order_power))
        anisotropy = row_sum / max(total_neighbor_excitation, 1e-12)
        continuity = float(row["proximity_sum"]) / row_sum
        two_sidedness = 1.25 if bool(row["has_positive"]) and bool(row["has_negative"]) else 1.0
        risk = row_sum * continuity * two_sidedness * low_index_weight * anisotropy
        if risk > best_risk:
            best_direction = direction
            best_row = row
            best_risk = float(risk)
            best_factors = (float(continuity), float(two_sidedness), float(anisotropy))

    if best_row is None:
        return _empty_affine_row()

    label = triplet_label(best_direction, brackets="()")
    continuity, two_sidedness, anisotropy = best_factors
    return {
        "nearest_row_direction": label,
        "best_affine_row_direction": label,
        "row_excited_count": int(best_row["count"]),
        "row_excitation_sum": float(best_row["sum"]),
        "affine_row_continuity_factor": continuity,
        "affine_row_two_sidedness_factor": two_sidedness,
        "affine_row_anisotropy_factor": anisotropy,
        "affine_row_crowding_raw": max(float(best_risk), 0.0),
        "systematic_row_risk_raw": max(float(best_risk), 0.0),
    }


def _signed_step(delta: np.ndarray, direction: tuple[int, int, int]) -> int:
    for delta_value, direction_value in zip(delta, direction, strict=False):
        if direction_value != 0:
            return int(delta_value) // int(direction_value)
    return 0


def _empty_affine_row() -> dict[str, float | int | str]:
    label = triplet_label((0, 0, 0), brackets="()")
    return {
        "nearest_row_direction": label,
        "best_affine_row_direction": label,
        "row_excited_count": 0,
        "row_excitation_sum": 0.0,
        "affine_row_continuity_factor": 0.0,
        "affine_row_two_sidedness_factor": 0.0,
        "affine_row_anisotropy_factor": 0.0,
        "affine_row_crowding_raw": 0.0,
        "systematic_row_risk_raw": 0.0,
    }
