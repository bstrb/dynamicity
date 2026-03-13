from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .geometry import (
    UnitCell,
    beam_direction_in_crystal,
    beam_direction_vector,
    compute_excitation_error,
    electron_wavelength_angstrom,
    generate_hkl_grid,
    generate_zone_axes,
    nearest_zone_axis,
)
from .parsers import ORIENTATION_COLUMNS


@dataclass(slots=True)
class OrientationMetricConfig:
    """Configuration for per-frame orientation metrics."""

    dmin: float = 0.8
    dmax: float | None = 20.0
    sg_threshold: float = 0.02
    zone_axis_limit: int = 4
    hkl_limit: int | None = 20
    max_candidate_reflections: int | None = 50000
    voltage_kv: float | None = None
    beam_direction: tuple[float, float, float] | None = None
    n_workers: int = 1
    chunk_size_frames: int = 2000


@dataclass(slots=True)
class OrientationMetricContext:
    """Precomputed arrays reused across all frames."""

    cell: UnitCell
    candidate_hkls: np.ndarray
    candidate_d: np.ndarray
    zone_axes: np.ndarray
    wavelength_angstrom: float
    beam_direction: np.ndarray


def _orientation_row_to_matrix(row: pd.Series) -> np.ndarray:
    return row[ORIENTATION_COLUMNS].to_numpy(dtype=np.float64).reshape(3, 3)


def build_orientation_context(cell: UnitCell, config: OrientationMetricConfig) -> OrientationMetricContext:
    """Precompute zone-axis candidates and reflection shell candidates."""
    wavelength = electron_wavelength_angstrom(config.voltage_kv or cell.voltage_kv)
    beam_dir = beam_direction_vector(config.beam_direction or cell.beam_direction)
    zone_axes = generate_zone_axes(config.zone_axis_limit)
    hkls, d_values = generate_hkl_grid(
        cell=cell,
        dmin=config.dmin,
        dmax=config.dmax,
        hkl_limit=config.hkl_limit,
        include_friedel=True,
        max_reflections=config.max_candidate_reflections,
    )
    return OrientationMetricContext(
        cell=cell,
        candidate_hkls=hkls,
        candidate_d=d_values,
        zone_axes=zone_axes,
        wavelength_angstrom=wavelength,
        beam_direction=beam_dir,
    )


def _compute_frame_metrics_single(
    frame: int,
    phi: float,
    ub: np.ndarray,
    context: OrientationMetricContext,
    config: OrientationMetricConfig,
) -> dict[str, float | int | str]:
    nearest_axis, zone_axis_angle = nearest_zone_axis(
        ub,
        zone_axes=context.zone_axes,
        zone_axis_limit=config.zone_axis_limit,
        beam_direction=context.beam_direction,
    )
    sg = compute_excitation_error(
        ub=ub,
        hkls=context.candidate_hkls,
        wavelength_angstrom=context.wavelength_angstrom,
        beam_direction=context.beam_direction,
    )
    excited = np.abs(sg) < config.sg_threshold
    n_excited = int(np.count_nonzero(excited))
    excitation_density = float(n_excited / max(context.candidate_hkls.shape[0], 1))
    beam_crystal = beam_direction_in_crystal(ub, context.beam_direction)
    return {
        "frame": int(frame),
        "phi": float(phi) if np.isfinite(phi) else np.nan,
        "nearest_zone_axis": f"[{nearest_axis[0]} {nearest_axis[1]} {nearest_axis[2]}]",
        "zone_axis_angle": float(zone_axis_angle),
        "N_excited": n_excited,
        "excitation_density": excitation_density,
        "beam_u": float(beam_crystal[0]),
        "beam_v": float(beam_crystal[1]),
        "beam_w": float(beam_crystal[2]),
        "candidate_reflections": int(context.candidate_hkls.shape[0]),
    }


def _orientation_worker(
    chunk: pd.DataFrame,
    context: OrientationMetricContext,
    config: OrientationMetricConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for _, row in chunk.iterrows():
        ub = _orientation_row_to_matrix(row)
        rows.append(_compute_frame_metrics_single(int(row["frame"]), float(row.get("phi", np.nan)), ub, context, config))
    return pd.DataFrame(rows)


def _frame_chunks(orientations: pd.DataFrame, chunk_size_frames: int) -> list[pd.DataFrame]:
    if chunk_size_frames <= 0:
        return [orientations]
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(orientations), chunk_size_frames):
        chunks.append(orientations.iloc[start : start + chunk_size_frames].copy())
    return chunks


def compute_orientation_metrics(
    orientations: pd.DataFrame,
    cell: UnitCell,
    config: OrientationMetricConfig | None = None,
) -> pd.DataFrame:
    """Compute zone-axis and excitation-density descriptors for each frame."""
    cfg = config or OrientationMetricConfig()
    context = build_orientation_context(cell, cfg)
    chunks = _frame_chunks(orientations.sort_values("frame").reset_index(drop=True), cfg.chunk_size_frames)

    if cfg.n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            outputs = list(executor.map(_orientation_worker, chunks, [context] * len(chunks), [cfg] * len(chunks)))
    else:
        outputs = [_orientation_worker(chunk, context, cfg) for chunk in chunks]

    summary = pd.concat(outputs, ignore_index=True).sort_values("frame").reset_index(drop=True)
    return summary
