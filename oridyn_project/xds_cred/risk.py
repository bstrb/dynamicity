"""Geometry-only OriDyn risk scoring for XDS integrated observations."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ScoreConfig
from .geometry import (
    GeometryLike,
    XdsGeometryModel,
    calculate_observation_risk,
    resolution_from_hkl,
    resolve_geometry,
)
from .xds_parser import iter_integrate_chunks


def enumerate_neighbor_offsets(reciprocal_matrix: np.ndarray, r_cut: float, sigma_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all integer reciprocal-lattice offsets inside r_cut, excluding zero."""

    singular_values = np.linalg.svd(np.asarray(reciprocal_matrix, dtype=float), compute_uv=False)
    min_scale = max(float(np.min(singular_values)), 1e-12)
    limit = int(np.ceil(float(r_cut) / min_scale)) + 1
    offsets: list[tuple[int, int, int]] = []
    distances: list[float] = []
    for dh in range(-limit, limit + 1):
        for dk in range(-limit, limit + 1):
            for dl in range(-limit, limit + 1):
                if dh == 0 and dk == 0 and dl == 0:
                    continue
                delta = np.asarray([dh, dk, dl], dtype=float)
                distance = float(np.linalg.norm(reciprocal_matrix @ delta))
                if distance <= float(r_cut) + 1e-12:
                    offsets.append((dh, dk, dl))
                    distances.append(distance)
    if not offsets:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)
    order = np.lexsort((np.asarray(offsets)[:, 2], np.asarray(offsets)[:, 1], np.asarray(offsets)[:, 0]))
    offset_array = np.asarray(offsets, dtype=int)[order]
    distance_array = np.asarray(distances, dtype=float)[order]
    coupling = np.exp(-((distance_array / max(float(sigma_c), 1e-12)) ** 2))
    return offset_array, distance_array, coupling


def score_integrate_file(
    integrate_path: Path,
    columns: list[str],
    geometry: GeometryLike,
    score: ScoreConfig,
    chunk_size: int,
    workers: int,
    output_path: Path,
    progress: Callable[[int], None] | None = None,
    max_observations: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score INTEGRATE.HKL in chunks and write scored rows incrementally."""

    reference_geometry = geometry.base if isinstance(geometry, XdsGeometryModel) else geometry
    offsets, distances, _coupling = enumerate_neighbor_offsets(
        reference_geometry.reciprocal_matrix_zero,
        score.r_cut,
        score.sigma_c,
    )
    metadata = {
        "neighbor_offset_count": int(len(offsets)),
        "neighbor_distance_min_invA": float(np.min(distances)) if len(distances) else None,
        "neighbor_distance_max_invA": float(np.max(distances)) if len(distances) else None,
        "score": asdict(score),
        "worker_count": int(workers),
        "chunk_size": int(chunk_size),
        "risk_intermediate": str(output_path),
    }
    excluded_frames: list[pd.DataFrame] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if workers <= 1:
        first_write = True
        for chunk_index, (chunk, excluded) in enumerate(
            iter_integrate_chunks(integrate_path, columns, chunk_size, max_observations)
        ):
            if not excluded.empty:
                excluded_frames.append(excluded)
            scored = _score_chunk(chunk, geometry, score, offsets)
            scored.to_csv(output_path, index=False, mode="w" if first_write else "a", header=first_write)
            first_write = False
            if progress:
                progress(len(chunk))
        excluded_all = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else pd.DataFrame()
        return excluded_all, metadata

    pending: dict[int, Any] = {}
    buffered_results: dict[int, pd.DataFrame] = {}
    next_to_write = 0
    first_write = True
    chunk_iter = enumerate(iter_integrate_chunks(integrate_path, columns, chunk_size, max_observations))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        for chunk_index, (chunk, excluded) in chunk_iter:
            if not excluded.empty:
                excluded_frames.append(excluded)
            future = executor.submit(_score_chunk, chunk, geometry, score, offsets)
            pending[chunk_index] = future
            if len(pending) >= workers * 2:
                _drain_completed(pending, buffered_results)
                first_write, next_to_write = _write_ready_results(
                    output_path,
                    buffered_results,
                    next_to_write,
                    first_write,
                    progress,
                )
        while pending:
            _drain_completed(pending, buffered_results, wait_for_one=True)
            first_write, next_to_write = _write_ready_results(
                output_path,
                buffered_results,
                next_to_write,
                first_write,
                progress,
            )
    excluded_all = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else pd.DataFrame()
    return excluded_all, metadata


def _worker_init() -> None:
    """Prevent nested numerical-library oversubscription inside workers."""

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _drain_completed(pending: dict[int, Any], buffered_results: dict[int, pd.DataFrame], wait_for_one: bool = False) -> None:
    """Move completed futures into the ordered result buffer."""

    done, _ = wait(pending.values(), return_when=FIRST_COMPLETED if wait_for_one else FIRST_COMPLETED)
    if not done:
        return
    completed_ids = [idx for idx, future in pending.items() if future in done]
    for idx in completed_ids:
        buffered_results[idx] = pending.pop(idx).result()


def _write_ready_results(
    output_path: Path,
    buffered_results: dict[int, pd.DataFrame],
    next_to_write: int,
    first_write: bool,
    progress: Callable[[int], None] | None,
) -> tuple[bool, int]:
    """Write currently available ordered chunks."""

    while next_to_write in buffered_results:
        scored = buffered_results.pop(next_to_write)
        scored.to_csv(output_path, index=False, mode="w" if first_write else "a", header=first_write)
        first_write = False
        if progress:
            progress(len(scored))
        next_to_write += 1
    return first_write, next_to_write


def _score_chunk(
    chunk: pd.DataFrame,
    geometry: GeometryLike,
    score: ScoreConfig,
    offsets: np.ndarray,
) -> pd.DataFrame:
    """Score one observation chunk."""

    records: list[dict[str, Any]] = []
    for row in chunk.itertuples(index=False):
        hkl = np.asarray([int(row.H), int(row.K), int(row.L)], dtype=int)
        zcal = float(row.ZCAL)
        result = calculate_observation_risk(hkl, zcal, geometry, score, offsets)
        concrete = resolve_geometry(geometry, zcal)
        out = {
            "observation_id": int(row.observation_id),
            "h": int(row.H),
            "k": int(row.K),
            "l": int(row.L),
            "resolution": float(resolution_from_hkl(hkl.reshape(1, 3), geometry)[0]),
            "IOBS": float(row.IOBS),
            "SIGMA": float(row.SIGMA) if hasattr(row, "SIGMA") else np.nan,
            "I_over_SIGMA": _safe_ratio(float(row.IOBS), float(row.SIGMA) if hasattr(row, "SIGMA") else np.nan),
            "PEAK": float(row.PEAK) if hasattr(row, "PEAK") else np.nan,
            "CORR": float(row.CORR) if hasattr(row, "CORR") else np.nan,
            "XCAL": float(row.XCAL) if hasattr(row, "XCAL") else np.nan,
            "YCAL": float(row.YCAL) if hasattr(row, "YCAL") else np.nan,
            "ZCAL": zcal,
            "XOBS": float(row.XOBS) if hasattr(row, "XOBS") else np.nan,
            "YOBS": float(row.YOBS) if hasattr(row, "YOBS") else np.nan,
            "ZOBS": float(row.ZOBS) if hasattr(row, "ZOBS") else np.nan,
            "geometry_source": result.geometry_source,
            "geometry_batch": result.geometry_batch,
            "rotation_angle_deg": result.rotation_angle,
            "z_angle_offset_frames": concrete.angle_offset_frames,
            "q_target_x": result.q_target[0],
            "q_target_y": result.q_target[1],
            "q_target_z": result.q_target[2],
            "s_target": result.s_target,
            "E_target": result.E_target,
            "neighbor_count": result.neighbor_count,
            "R_env": result.R_env,
            "S_risk": result.S_risk,
            "z_pred": result.z_pred,
            "z_residual_frames": result.z_residual_frames,
            "z_residual_degrees": result.z_residual_degrees,
        }
        records.append(out)
    return pd.DataFrame.from_records(records)


def score_dataframe_serial(observations: pd.DataFrame, geometry: GeometryLike, score: ScoreConfig) -> pd.DataFrame:
    """Score an in-memory observation DataFrame, used by tests."""

    reference_geometry = geometry.base if isinstance(geometry, XdsGeometryModel) else geometry
    offsets, _distances, _coupling = enumerate_neighbor_offsets(
        reference_geometry.reciprocal_matrix_zero,
        score.r_cut,
        score.sigma_c,
    )
    return _score_chunk(observations, geometry, score, offsets)


def score_dataframe_chunked(observations: pd.DataFrame, geometry: GeometryLike, score: ScoreConfig, chunk_size: int) -> pd.DataFrame:
    """Score an in-memory DataFrame in chunks, used by tests."""

    frames = []
    reference_geometry = geometry.base if isinstance(geometry, XdsGeometryModel) else geometry
    offsets, _distances, _coupling = enumerate_neighbor_offsets(
        reference_geometry.reciprocal_matrix_zero,
        score.r_cut,
        score.sigma_c,
    )
    for start in range(0, len(observations), chunk_size):
        frames.append(_score_chunk(observations.iloc[start : start + chunk_size], geometry, score, offsets))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0.0:
        return numerator / denominator
    return float("nan")
