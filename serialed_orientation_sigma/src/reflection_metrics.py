from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .geometry import (
    UnitCell,
    compute_excitation_error,
    d_spacing,
    electron_wavelength_angstrom,
)
from .parsers import ORIENTATION_COLUMNS


ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Pb": 82,
    "Bi": 83,
    "U": 92,
}


@dataclass(slots=True)
class ReflectionMetricConfig:
    """Configuration controlling per-reflection sensitivity scoring."""

    xi_scale_nm: float = 200.0
    structure_factor_decay: float = 0.35
    zone_axis_soft_angle_deg: float = 5.0
    combined_formulation: str = "log_multibeam"
    zone_axis_weight: float = 0.25
    voltage_kv: float | None = None
    beam_direction: tuple[float, float, float] | None = None
    n_workers: int = 1
    chunk_size_frames: int = 2000


def _composition_scale(cell: UnitCell) -> float:
    if not cell.composition:
        return 10.0
    total = 0.0
    for element, count in cell.composition.items():
        total += ATOMIC_NUMBERS.get(str(element), 10) * float(count)
    return max(total, 1.0)


def estimate_structure_factor_proxy(
    hkls: np.ndarray,
    cell: UnitCell,
    structure_factor_decay: float = 0.35,
) -> np.ndarray:
    """Estimate a smooth structure-factor proxy for extinction-distance scaling.

    The proxy depends on the scattering magnitude and the overall composition
    scale, but deliberately avoids requiring atomic coordinates.
    """
    d_values = d_spacing(cell, hkls)
    s = 1.0 / (2.0 * np.maximum(d_values, 1e-8))
    composition_scale = _composition_scale(cell)
    proxy = composition_scale * np.exp(-structure_factor_decay * s ** 2) / np.sqrt(1.0 + s ** 2)
    return np.maximum(proxy, 1e-6)


def estimate_extinction_distance(
    hkls: np.ndarray,
    cell: UnitCell,
    xi_scale_nm: float = 200.0,
    structure_factor_decay: float = 0.35,
) -> np.ndarray:
    """Estimate extinction distance in angstrom from a structure-factor proxy."""
    f_proxy = estimate_structure_factor_proxy(hkls, cell, structure_factor_decay=structure_factor_decay)
    return (float(xi_scale_nm) * 10.0) / f_proxy


def zone_axis_proximity_from_angle(angle_deg: np.ndarray | float, soft_angle_deg: float = 5.0) -> np.ndarray:
    """Transform zone-axis distance into a bounded proximity score."""
    arr = np.asarray(angle_deg, dtype=np.float64)
    return 1.0 / (1.0 + arr / max(float(soft_angle_deg), 1e-8))


def combine_dynamical_score(
    d_2beam: np.ndarray,
    n_excited: np.ndarray,
    zone_axis_proximity: np.ndarray,
    formulation: str = "log_multibeam",
    zone_axis_weight: float = 0.25,
) -> np.ndarray:
    """Combine per-reflection and per-frame metrics into a single score."""
    if formulation == "log_multibeam":
        return d_2beam * (1.0 + np.log1p(n_excited))
    if formulation == "zone_axis_boost":
        return d_2beam * (1.0 + np.log1p(n_excited)) * (1.0 + zone_axis_weight * zone_axis_proximity)
    if formulation == "weighted_sum":
        return d_2beam + 0.1 * np.log1p(n_excited) + zone_axis_weight * zone_axis_proximity
    raise ValueError(f"Unknown combined score formulation: {formulation}")


def _reflection_worker(
    chunk: pd.DataFrame,
    cell: UnitCell,
    config: ReflectionMetricConfig,
    wavelength_angstrom: float,
    beam_direction: tuple[float, float, float],
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    for frame, group in chunk.groupby("frame", sort=False):
        ub = group.iloc[0][ORIENTATION_COLUMNS].to_numpy(dtype=np.float64).reshape(3, 3)
        hkls = group[["h", "k", "l"]].to_numpy(dtype=np.float64)
        sg = compute_excitation_error(
            ub=ub,
            hkls=hkls,
            wavelength_angstrom=wavelength_angstrom,
            beam_direction=beam_direction,
        )
        xi_g = estimate_extinction_distance(
            hkls=hkls,
            cell=cell,
            xi_scale_nm=config.xi_scale_nm,
            structure_factor_decay=config.structure_factor_decay,
        )
        d_2beam = 1.0 / (1.0 + (sg * xi_g) ** 2)
        zone_axis_proximity = zone_axis_proximity_from_angle(
            group["zone_axis_angle"].to_numpy(dtype=np.float64),
            soft_angle_deg=config.zone_axis_soft_angle_deg,
        )
        n_excited = group["N_excited"].to_numpy(dtype=np.float64)
        score = combine_dynamical_score(
            d_2beam=d_2beam,
            n_excited=n_excited,
            zone_axis_proximity=zone_axis_proximity,
            formulation=config.combined_formulation,
            zone_axis_weight=config.zone_axis_weight,
        )

        enriched = group.copy()
        enriched["sg"] = sg
        enriched["xi_g"] = xi_g
        enriched["d_2beam"] = d_2beam
        enriched["zone_axis_proximity"] = zone_axis_proximity
        enriched["multi_beam_density"] = enriched["excitation_density"].to_numpy(dtype=np.float64)
        enriched["combined_dynamical_score"] = score
        enriched["S"] = score
        results.append(enriched)
    return pd.concat(results, ignore_index=True)


def _frame_chunks_by_id(frame_ids: np.ndarray, chunk_size_frames: int) -> list[np.ndarray]:
    if chunk_size_frames <= 0:
        return [frame_ids]
    chunks = []
    for start in range(0, len(frame_ids), chunk_size_frames):
        chunks.append(frame_ids[start : start + chunk_size_frames])
    return chunks


def compute_reflection_metrics(
    reflections: pd.DataFrame,
    orientations: pd.DataFrame,
    frame_summary: pd.DataFrame,
    cell: UnitCell,
    config: ReflectionMetricConfig | None = None,
) -> pd.DataFrame:
    """Compute per-reflection dynamical sensitivity metrics."""
    cfg = config or ReflectionMetricConfig()
    orientation_frames = set(orientations["frame"].astype(int))
    reflection_frames = set(reflections["frame"].astype(int))
    missing = sorted(reflection_frames - orientation_frames)
    if missing:
        raise ValueError(f"Missing orientations for reflection frames: {missing[:10]}")

    merged = reflections.merge(orientations, on="frame", how="left", validate="many_to_one")
    merged = merged.merge(frame_summary, on=["frame", "phi"], how="left", validate="many_to_one")
    merged = merged.sort_values(["frame"]).reset_index(drop=True)

    wavelength_angstrom = electron_wavelength_angstrom(cfg.voltage_kv or cell.voltage_kv)
    beam_direction = tuple(float(v) for v in (cfg.beam_direction or cell.beam_direction))

    frame_ids = merged["frame"].drop_duplicates().to_numpy(dtype=int)
    frame_chunks = _frame_chunks_by_id(frame_ids, cfg.chunk_size_frames)
    chunks = [merged[merged["frame"].isin(chunk_ids)].copy() for chunk_ids in frame_chunks]

    if cfg.n_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            outputs = list(
                executor.map(
                    _reflection_worker,
                    chunks,
                    [cell] * len(chunks),
                    [cfg] * len(chunks),
                    [wavelength_angstrom] * len(chunks),
                    [beam_direction] * len(chunks),
                )
            )
    else:
        outputs = [
            _reflection_worker(chunk, cell, cfg, wavelength_angstrom, beam_direction)
            for chunk in chunks
        ]

    return pd.concat(outputs, ignore_index=True)
