"""Geometry and orientation helpers for the OriDyn score path."""

from __future__ import annotations

from math import acos, cos, gcd, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .stream_parser import STREAM_MATRIX_COLUMNS, UnitCell

FloatArray = NDArray[np.float64]


def direct_matrix_from_cell(cell: UnitCell) -> FloatArray:
    """Return direct-space basis vectors as columns in angstrom."""

    alpha = np.deg2rad(cell.alpha)
    beta = np.deg2rad(cell.beta)
    gamma = np.deg2rad(cell.gamma)
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1e-12:
        raise ValueError("Unit-cell gamma is too close to 0 or 180 degrees.")

    a_vec = np.asarray([cell.a, 0.0, 0.0], dtype=float)
    b_vec = np.asarray([cell.b * cos(gamma), cell.b * sin_gamma, 0.0], dtype=float)
    c_x = cell.c * cos(beta)
    c_y = cell.c * (cos(alpha) - cos(beta) * cos(gamma)) / sin_gamma
    c_z = sqrt(max(cell.c * cell.c - c_x * c_x - c_y * c_y, 0.0))
    c_vec = np.asarray([c_x, c_y, c_z], dtype=float)
    return np.column_stack([a_vec, b_vec, c_vec])


def reciprocal_matrix_from_cell(cell: UnitCell) -> FloatArray:
    """Return reciprocal basis columns in inverse angstrom without 2*pi."""

    direct = direct_matrix_from_cell(cell)
    return np.linalg.inv(direct).T


def direct_matrix_from_reciprocal(reciprocal_matrix: ArrayLike) -> FloatArray:
    """Return direct basis columns corresponding to a reciprocal matrix."""

    reciprocal = np.asarray(reciprocal_matrix, dtype=float)
    if reciprocal.shape != (3, 3):
        raise ValueError("reciprocal_matrix must have shape (3, 3).")
    return np.linalg.inv(reciprocal).T


def hkl_lab_vectors(hkls: ArrayLike, reciprocal_matrix: ArrayLike) -> FloatArray:
    """Map HKLs to lab-frame reciprocal vectors."""

    hkl_array = np.asarray(hkls, dtype=float)
    reciprocal = np.asarray(reciprocal_matrix, dtype=float)
    if hkl_array.ndim == 1:
        hkl_array = hkl_array.reshape(1, 3)
    if reciprocal.shape != (3, 3):
        raise ValueError("reciprocal_matrix must have shape (3, 3).")
    return hkl_array @ reciprocal.T


def vector_norms(vectors: ArrayLike) -> FloatArray:
    """Return row-wise Euclidean norms."""

    array = np.asarray(vectors, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.linalg.norm(array, axis=1)


def d_spacings_from_q(q_invA: ArrayLike) -> FloatArray:
    """Convert reciprocal length to d-spacing with safe infinities."""

    q = np.asarray(q_invA, dtype=float)
    return np.divide(1.0, q, out=np.full_like(q, np.inf, dtype=float), where=q > 0.0)


def normalize_vector(vector: ArrayLike) -> FloatArray:
    """Return a unit vector."""

    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("Vector must be non-zero.")
    return array / norm


def excitation_error(
    g_vectors: ArrayLike,
    wavelength_angstrom: float,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Compute an Ewald-sphere excitation-error proxy in inverse angstrom."""

    g = np.asarray(g_vectors, dtype=float)
    if g.ndim == 1:
        g = g.reshape(1, 3)
    beam_unit = normalize_vector(beam_direction)
    k0 = beam_unit / float(wavelength_angstrom)
    return np.linalg.norm(g + k0[None, :], axis=1) - np.linalg.norm(k0)


def excitation_weight(
    sg: ArrayLike,
    sg0: float,
    kernel: str = "gaussian",
    lorentzian_power: float = 2.0,
) -> FloatArray:
    """Convert excitation error to a soft excitation weight."""

    values = np.asarray(sg, dtype=float)
    scale = max(float(sg0), 1e-12)
    x = np.abs(values / scale)
    if kernel == "gaussian":
        return np.exp(-(x**2))
    if kernel == "lorentzian":
        return 1.0 / (1.0 + x**float(lorentzian_power))
    raise ValueError(f"Unknown excitation kernel: {kernel}")


def angular_distance_deg(
    direction: ArrayLike,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
    signless: bool = True,
) -> float:
    """Return angular distance between a direction and the beam."""

    d = normalize_vector(direction)
    beam = normalize_vector(beam_direction)
    dot = float(d @ beam)
    if signless:
        dot = abs(dot)
    dot = min(1.0, max(-1.0, dot))
    return float(np.rad2deg(acos(dot)))


def axis_lab_direction(reciprocal_matrix: ArrayLike, axis: tuple[int, int, int]) -> FloatArray:
    """Map a direct-lattice zone axis to lab coordinates."""

    direct = direct_matrix_from_reciprocal(reciprocal_matrix)
    return direct @ np.asarray(axis, dtype=float)


def axis_angle_deg(
    reciprocal_matrix: ArrayLike,
    axis: tuple[int, int, int],
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> float:
    """Return beam-axis angular distance for one zone axis."""

    return angular_distance_deg(axis_lab_direction(reciprocal_matrix, axis), beam_direction=beam_direction)


def beam_in_direct_coordinates(
    reciprocal_matrix: ArrayLike,
    beam_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Express the lab beam direction in direct-lattice coordinates."""

    direct = direct_matrix_from_reciprocal(reciprocal_matrix)
    coords = np.linalg.solve(direct, normalize_vector(beam_direction))
    norm = float(np.linalg.norm(coords))
    return coords if norm == 0.0 else coords / norm


def gcd3(h: int, k: int, l: int) -> int:
    """Greatest common divisor of three integers."""

    return gcd(gcd(abs(int(h)), abs(int(k))), abs(int(l)))


def canonical_triplet(h: int, k: int, l: int) -> tuple[int, int, int]:
    """Return a primitive, sign-canonical integer triplet."""

    if h == 0 and k == 0 and l == 0:
        return (0, 0, 0)
    div = gcd3(h, k, l)
    out = (int(h // div), int(k // div), int(l // div))
    for value in out:
        if value != 0:
            return tuple(-x for x in out) if value < 0 else out
    return out


def triplet_label(triplet: tuple[int, int, int], brackets: str = "[]") -> str:
    """Format an integer triplet."""

    left, right = brackets[0], brackets[-1]
    return f"{left}{triplet[0]} {triplet[1]} {triplet[2]}{right}"


def parse_triplet_label(label: str) -> tuple[int, int, int]:
    """Parse labels such as ``[1 0 -1]`` or ``(1 0 -1)``."""

    clean = str(label).strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = clean.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Could not parse triplet label: {label}")
    return tuple(int(float(part)) for part in parts)  # type: ignore[return-value]


def orientation_checks(crystal_table: pd.DataFrame) -> dict[str, object]:
    """Summarize basic reciprocal-matrix convention checks for metadata."""

    determinants: list[float] = []
    norms: list[float] = []
    for _, row in crystal_table.iterrows():
        matrix = row[list(STREAM_MATRIX_COLUMNS)].to_numpy(dtype=float).reshape(3, 3)
        determinants.append(float(np.linalg.det(matrix)))
        norms.extend(float(np.linalg.norm(matrix[:, idx])) for idx in range(3))
    det = np.asarray(determinants, dtype=float)
    basis_norms = np.asarray(norms, dtype=float)
    return {
        "reciprocal_matrix_convention": "columns are astar, bstar, cstar in lab-frame A^-1",
        "hkl_mapping": "g_lab = reciprocal_matrix @ [h, k, l]",
        "handedness_positive_fraction": float(np.mean(det > 0.0)) if det.size else None,
        "determinant_min": float(np.min(det)) if det.size else None,
        "determinant_max": float(np.max(det)) if det.size else None,
        "basis_norm_min_invA": float(np.min(basis_norms)) if basis_norms.size else None,
        "basis_norm_max_invA": float(np.max(basis_norms)) if basis_norms.size else None,
    }


def low_order_prior_from_q(q_invA: ArrayLike, g0: float = 0.40, power: float = 1.5) -> FloatArray:
    """Smooth low-order prior for reciprocal vectors."""

    q = np.asarray(q_invA, dtype=float)
    return 1.0 / (1.0 + (q / max(float(g0), 1e-12)) ** float(power))


def iter_primitive_triplets(limit: int) -> Iterable[tuple[int, int, int]]:
    """Yield sign-canonical primitive triplets up to an index limit."""

    seen: set[tuple[int, int, int]] = set()
    for h in range(-limit, limit + 1):
        for k in range(-limit, limit + 1):
            for l in range(-limit, limit + 1):
                triplet = canonical_triplet(h, k, l)
                if triplet == (0, 0, 0) or triplet in seen:
                    continue
                seen.add(triplet)
                yield triplet
