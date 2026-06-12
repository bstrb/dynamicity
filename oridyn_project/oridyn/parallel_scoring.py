"""Frame-parallel raw scoring for the main OriDyn pipeline."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .axis_prediction import unique_zone_axes
from .config import OridynConfig
from .excitation import compute_excited_candidate_nodes
from .geometry import (
    axis_angle_deg,
    beam_in_direct_coordinates,
    d_spacings_from_q,
    excitation_error,
    excitation_weight,
    hkl_lab_vectors,
    triplet_label,
    vector_norms,
)
from .graph_crowding import _delta_prior, _empty_graph_row
from .self_risk import add_self_risk_terms
from .stream_parser import STREAM_MATRIX_COLUMNS, StreamData
from .systematic_rows import add_affine_row_terms_from_nodes

_WORKER_STATE: dict[str, Any] = {}

OBSERVED_KEEP_COLUMNS = (
    "frame",
    "frame_number",
    "chunk_id",
    "crystal_in_chunk",
    "source_filename",
    "event",
    "image_serial",
    "h",
    "k",
    "l",
    "sigma",
    "fs_px",
    "ss_px",
    "panel",
    "target_index",
    "target_source",
    "hkl_label",
)


class ProgressReporter:
    """Small stderr progress reporter for long command-line runs."""

    def __init__(self, total: int, label: str, enabled: bool = True, min_interval_s: float = 2.0) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.enabled = enabled
        self.min_interval_s = float(min_interval_s)
        self.count = 0
        self.started = time.monotonic()
        self.last_emit = 0.0
        if self.enabled:
            self._emit(force=True)

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        self._emit(force=False)

    def done(self) -> None:
        self.count = self.total
        self._emit(force=True, done=True)

    def _emit(self, force: bool = False, done: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_emit < self.min_interval_s:
            return
        self.last_emit = now
        elapsed = max(now - self.started, 1e-9)
        rate = self.count / elapsed
        if self.total:
            pct = 100.0 * self.count / self.total
            message = f"[oridyn] {self.label}: {self.count}/{self.total} ({pct:5.1f}%), {rate:.2f} frames/s"
        else:
            message = f"[oridyn] {self.label}: 0/0"
        if done:
            message += ", done"
        print(message, file=sys.stderr, flush=True)


def log_progress(message: str, enabled: bool = True) -> None:
    """Print one progress line to stderr."""

    if enabled:
        print(f"[oridyn] {message}", file=sys.stderr, flush=True)


def score_frames_and_reflections(
    stream: StreamData,
    problematic_axes: pd.DataFrame,
    candidates: pd.DataFrame,
    config: OridynConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Score raw frame and reflection terms using frame-level parallelism."""

    tasks = _build_frame_tasks(stream)
    risky_axes = [_axis_tuple_from_row(row) for row in problematic_axes.itertuples(index=False)]
    nearest_axes = unique_zone_axes(config.uvw_max)
    q_span = float(candidates["q_invA"].max() - candidates["q_invA"].min()) if len(candidates) > 1 else 1.0
    workers = _resolve_workers(config.workers)
    if len(tasks) <= 1:
        workers = 1

    reporter = ProgressReporter(len(tasks), f"scoring frames with {workers} worker(s)", enabled=config.progress)
    frame_rows: list[dict[str, float | int | str]] = []
    reflection_tables: list[pd.DataFrame] = []

    if workers == 1:
        for task in tasks:
            frame_row, reflections = _score_one_frame(
                task, candidates, risky_axes, nearest_axes, stream.wavelength_angstrom, config, q_span
            )
            frame_rows.append(frame_row)
            if not reflections.empty:
                reflection_tables.append(reflections)
            reporter.update()
    else:
        context = _multiprocessing_context()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(candidates, risky_axes, nearest_axes, stream.wavelength_angstrom, config, q_span),
        ) as executor:
            futures = [executor.submit(_score_one_frame_from_state, task) for task in tasks]
            for future in as_completed(futures):
                frame_row, reflections = future.result()
                frame_rows.append(frame_row)
                if not reflections.empty:
                    reflection_tables.append(reflections)
                reporter.update()
    reporter.done()

    frame_scores = pd.DataFrame.from_records(frame_rows).sort_values("frame").reset_index(drop=True)
    if reflection_tables:
        reflection_scores = pd.concat(reflection_tables, ignore_index=True)
        reflection_scores = reflection_scores.sort_values(["frame", "h", "k", "l"]).reset_index(drop=True)
        reflection_scores.insert(0, "reflection_id", np.arange(len(reflection_scores), dtype=int))
    else:
        reflection_scores = pd.DataFrame()

    metadata = {
        "parallel_scoring": {
            "workers_requested": int(config.workers),
            "workers_used": int(workers),
            "n_frame_tasks": int(len(tasks)),
            "progress_enabled": bool(config.progress),
        }
    }
    return frame_scores, reflection_scores, metadata


def _init_worker(
    candidates: pd.DataFrame,
    risky_axes: list[tuple[tuple[int, int, int], float, str]],
    nearest_axes: list[tuple[int, int, int]],
    wavelength_angstrom: float,
    config: OridynConfig,
    q_span: float,
) -> None:
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "candidates": candidates,
            "risky_axes": risky_axes,
            "nearest_axes": nearest_axes,
            "wavelength_angstrom": wavelength_angstrom,
            "config": config,
            "q_span": q_span,
        }
    )


def _score_one_frame_from_state(task: tuple[dict[str, Any], pd.DataFrame]) -> tuple[dict[str, Any], pd.DataFrame]:
    return _score_one_frame(
        task,
        _WORKER_STATE["candidates"],
        _WORKER_STATE["risky_axes"],
        _WORKER_STATE["nearest_axes"],
        _WORKER_STATE["wavelength_angstrom"],
        _WORKER_STATE["config"],
        _WORKER_STATE["q_span"],
    )


def _score_one_frame(
    task: tuple[dict[str, Any], pd.DataFrame],
    candidates: pd.DataFrame,
    risky_axes: list[tuple[tuple[int, int, int], float, str, int]],
    nearest_axes: list[tuple[int, int, int]],
    wavelength_angstrom: float,
    config: OridynConfig,
    q_span: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    crystal_row, observations = task
    reciprocal = _reciprocal_from_crystal_dict(crystal_row)
    nodes = compute_excited_candidate_nodes(reciprocal, candidates, wavelength_angstrom, config)
    frame_row, assigned_axis = _score_frame_row(
        crystal_row,
        reciprocal,
        nodes,
        risky_axes,
        nearest_axes,
        config,
        q_span,
        len(candidates),
    )
    reflection_scores = _score_observations(
        observations,
        reciprocal,
        nodes,
        assigned_axis,
        wavelength_angstrom,
        config,
    )
    return frame_row, reflection_scores


def _build_frame_tasks(stream: StreamData) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    observed = stream.reflections.copy()
    keep = [col for col in OBSERVED_KEEP_COLUMNS if col in observed.columns]
    grouped = {
        int(frame): group[keep].copy()
        for frame, group in observed[keep].groupby("frame", sort=False)
    } if not observed.empty else {}
    empty = pd.DataFrame(columns=keep)
    tasks: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for _, crystal_row in stream.crystal_table.sort_values("frame").iterrows():
        frame = int(crystal_row["frame"])
        tasks.append((crystal_row.to_dict(), grouped.get(frame, empty.copy())))
    return tasks


def _reciprocal_from_crystal_dict(crystal_row: dict[str, Any]) -> np.ndarray:
    return np.asarray([crystal_row[col] for col in STREAM_MATRIX_COLUMNS], dtype=float).reshape(3, 3)


def _score_frame_row(
    crystal_row: dict[str, Any],
    reciprocal: np.ndarray,
    nodes: pd.DataFrame,
    risky_axes: list[tuple[tuple[int, int, int], float, str, int]],
    nearest_axes: list[tuple[int, int, int]],
    config: OridynConfig,
    q_span: float,
    candidate_count: int,
) -> tuple[dict[str, Any], tuple[int, int, int]]:
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

    sum_weight = float(nodes["excitation_weight"].sum()) if not nodes.empty else 0.0
    beam_coords = beam_in_direct_coordinates(reciprocal, config.beam_direction)
    return (
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
            "excitation_density": float(len(nodes) / max(candidate_count, 1)),
            "resolution_normalized_excitation_density": sum_weight / max(q_span, 1e-12),
            "assigned_risky_axis_u": best_axis[0],
            "assigned_risky_axis_v": best_axis[1],
            "assigned_risky_axis_w": best_axis[2],
        },
        best_axis,
    )


def _score_observations(
    observations: pd.DataFrame,
    reciprocal: np.ndarray,
    nodes: pd.DataFrame,
    assigned_axis: tuple[int, int, int],
    wavelength_angstrom: float,
    config: OridynConfig,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    scored = observations.copy()
    hkls = scored[["h", "k", "l"]].to_numpy(dtype=int)
    g = hkl_lab_vectors(hkls, reciprocal)
    q = vector_norms(g)
    sg = excitation_error(g, wavelength_angstrom, config.beam_direction)
    weights = excitation_weight(sg, config.sg0, config.excitation_kernel, config.excitation_lorentzian_power)
    if "target_source" not in scored:
        scored["target_source"] = "observed"
    else:
        scored["target_source"] = scored["target_source"].fillna("observed").astype(str)
    scored["q_invA"] = q
    scored["d_angstrom"] = d_spacings_from_q(q)
    scored["sg"] = sg
    scored["excitation_weight"] = weights
    scored["excitation_center"] = weights
    scored["excitation_mean"] = weights
    scored["excitation_max"] = weights
    scored["excitation_integrated"] = weights
    scored = add_self_risk_terms(scored, config)
    scored = _add_graph_terms_from_nodes(scored, nodes, config)
    scored = _add_laue_terms_from_nodes(scored, nodes, assigned_axis)
    scored = add_affine_row_terms_from_nodes(scored, nodes, config)
    return scored


def _add_graph_terms_from_nodes(scores: pd.DataFrame, nodes: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    node_hkl = nodes[["h", "k", "l"]].to_numpy(dtype=int) if not nodes.empty else np.empty((0, 3), dtype=int)
    node_weight = nodes["excitation_weight"].to_numpy(dtype=float) if not nodes.empty else np.empty(0)
    rows: list[dict[str, float | str]] = []
    for target in scores.itertuples(index=False):
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
    return pd.concat([scores.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1)


def _add_laue_terms_from_nodes(
    scores: pd.DataFrame,
    nodes: pd.DataFrame,
    assigned_axis: tuple[int, int, int],
) -> pd.DataFrame:
    uvw = np.asarray(assigned_axis, dtype=int)
    node_hkl = nodes[["h", "k", "l"]].to_numpy(dtype=int) if not nodes.empty else np.empty((0, 3), dtype=int)
    node_weight = nodes["excitation_weight"].to_numpy(dtype=float) if not nodes.empty else np.empty(0)
    node_laue = node_hkl @ uvw if len(node_hkl) else np.empty(0, dtype=int)
    rows: list[dict[str, float | int | bool | str]] = []
    for target in scores.itertuples(index=False):
        hkl = np.asarray([target.h, target.k, target.l], dtype=int)
        laue_n = int(hkl @ uvw)
        abs_n = abs(laue_n)
        same_sum = float(np.sum(node_weight[node_laue == laue_n])) if len(node_laue) else 0.0
        zone_weight = 1.0 / (1.0 + abs_n)
        rows.append(
            {
                "assigned_zone_axis": triplet_label(assigned_axis),
                "laue_n": laue_n,
                "abs_laue_n": abs_n,
                "is_zolz": laue_n == 0,
                "is_folz": abs_n == 1,
                "near_zone_law": abs_n <= 1,
                "same_laue_zone_crowding_raw": float(np.log1p(same_sum)),
                "laue_zone_risk_raw": float(np.log1p(same_sum) * zone_weight),
            }
        )
    return pd.concat([scores.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1)


def _resolve_workers(requested_workers: int) -> int:
    if int(requested_workers) == 0:
        return max(os.cpu_count() or 1, 1)
    return max(int(requested_workers), 1)


def _multiprocessing_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def _axis_tuple_from_row(row: Any) -> tuple[tuple[int, int, int], float, str, int]:
    return (
        (int(row.u), int(row.v), int(row.w)),
        float(row.axis_score),
        str(row.axis_label),
        int(row.axis_rank) if hasattr(row, "axis_rank") else -1,
    )
