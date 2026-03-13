from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, ceil, cos, gcd, pi, radians, sin, sqrt
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(slots=True)
class UnitCell:
    """Crystallographic unit-cell parameters and optional metadata."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    voltage_kv: float = 200.0
    composition: dict[str, float] = field(default_factory=dict)
    beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the cell."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "voltage_kv": self.voltage_kv,
            "composition": dict(self.composition),
            "beam_direction": list(self.beam_direction),
        }


def as_float_array(values: Sequence[float]) -> FloatArray:
    """Return a float64 numpy array from a sequence."""
    return np.asarray(values, dtype=np.float64)


@njit(cache=True)
def _rowwise_norm(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.empty(array.shape[0], dtype=np.float64)
    for i in range(array.shape[0]):
        out[i] = sqrt(array[i, 0] ** 2 + array[i, 1] ** 2 + array[i, 2] ** 2)
    return out


def unit_vector(vector: Sequence[float]) -> FloatArray:
    """Normalize a vector.

    Parameters
    ----------
    vector:
        Input vector.
    """
    arr = as_float_array(vector)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return arr / norm


def electron_wavelength_angstrom(voltage_kv: float) -> float:
    """Return relativistic electron wavelength in angstrom.

    The formula uses the standard accelerating-voltage approximation with the
    voltage expressed in kilovolts.
    """
    voltage_v = float(voltage_kv) * 1000.0
    return 12.2639 / sqrt(voltage_v * (1.0 + 0.97845e-6 * voltage_v))


def cell_to_direct_matrix(cell: UnitCell) -> FloatArray:
    """Convert unit-cell parameters into a 3x3 direct-space basis matrix.

    The columns of the returned matrix are the direct-space basis vectors in an
    orthonormal Cartesian frame.
    """
    alpha = radians(cell.alpha)
    beta = radians(cell.beta)
    gamma = radians(cell.gamma)

    a_vec = np.array([cell.a, 0.0, 0.0], dtype=np.float64)
    b_vec = np.array([cell.b * cos(gamma), cell.b * sin(gamma), 0.0], dtype=np.float64)

    c_x = cell.c * cos(beta)
    gamma_sin = sin(gamma)
    if abs(gamma_sin) < 1e-12:
        raise ValueError("Gamma angle leads to a singular basis matrix.")
    c_y = cell.c * (cos(alpha) - cos(beta) * cos(gamma)) / gamma_sin
    c_z_sq = cell.c ** 2 - c_x ** 2 - c_y ** 2
    c_z = sqrt(max(c_z_sq, 0.0))
    c_vec = np.array([c_x, c_y, c_z], dtype=np.float64)

    return np.column_stack([a_vec, b_vec, c_vec])


def cell_to_reciprocal_matrix(cell: UnitCell) -> FloatArray:
    """Return the reciprocal basis matrix without the 2pi factor."""
    direct = cell_to_direct_matrix(cell)
    return np.linalg.inv(direct).T


def d_spacing(cell: UnitCell, hkls: npt.ArrayLike) -> FloatArray:
    """Compute d spacing for one or many Miller indices."""
    miller = np.atleast_2d(np.asarray(hkls, dtype=np.float64))
    recip = cell_to_reciprocal_matrix(cell)
    g = miller @ recip.T
    norm_g = np.linalg.norm(g, axis=1)
    with np.errstate(divide="ignore"):
        d = 1.0 / norm_g
    return d


def rotation_matrix_from_axis_angle(axis: Sequence[float], angle_deg: float) -> FloatArray:
    """Return a Rodrigues rotation matrix from axis and angle in degrees."""
    ax = unit_vector(axis)
    angle_rad = radians(angle_deg)
    x, y, z = ax
    c = cos(angle_rad)
    s = sin(angle_rad)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def matrix_from_orientation_row(row: Sequence[float]) -> FloatArray:
    """Build a 3x3 matrix from a flat row-major list of 9 values."""
    arr = np.asarray(row, dtype=np.float64)
    if arr.size != 9:
        raise ValueError("Expected 9 numbers to build a 3x3 orientation matrix.")
    return arr.reshape(3, 3)


def orientation_matrix_to_row(ub: npt.ArrayLike) -> list[float]:
    """Flatten a 3x3 matrix in row-major order."""
    return np.asarray(ub, dtype=np.float64).reshape(9).tolist()


def reconstruct_rotation_series_orientations(
    reference_ub: npt.ArrayLike,
    rotation_axis: Sequence[float],
    frame_ids: Sequence[int],
    phi0_deg: float,
    dphi_deg: float,
    start_frame: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Reconstruct per-frame orientation matrices for an XDS rotation series.

    Parameters
    ----------
    reference_ub:
        Reciprocal basis matrix at the reference frame.
    rotation_axis:
        Rotation axis in the laboratory frame.
    frame_ids:
        Frame identifiers for which orientation matrices should be built.
    phi0_deg:
        Starting rotation angle in degrees.
    dphi_deg:
        Oscillation step per frame in degrees.
    start_frame:
        Frame number corresponding to `phi0_deg`.
    """
    ref = np.asarray(reference_ub, dtype=np.float64)
    frames = np.asarray(frame_ids, dtype=np.int64)
    phis = phi0_deg + (frames - int(start_frame)) * dphi_deg
    mats = np.empty((frames.size, 3, 3), dtype=np.float64)
    for idx, phi in enumerate(phis):
        delta = phi - phi0_deg
        rot = rotation_matrix_from_axis_angle(rotation_axis, delta)
        mats[idx] = rot @ ref
    return phis.astype(np.float64), mats


def beam_direction_vector(beam_direction: Sequence[float] | None = None) -> FloatArray:
    """Return a normalized beam direction vector in the laboratory frame."""
    if beam_direction is None:
        beam_direction = (0.0, 0.0, 1.0)
    return unit_vector(beam_direction)


def beam_direction_in_crystal(ub: npt.ArrayLike, beam_direction: Sequence[float] | None = None) -> FloatArray:
    """Express the beam direction in direct-lattice coordinates.

    The orientation matrix is assumed to map Miller indices to reciprocal-space
    vectors in the laboratory frame. The corresponding direct basis in the lab is
    `inv(UB).T`.
    """
    ub_arr = np.asarray(ub, dtype=np.float64)
    direct_lab = np.linalg.inv(ub_arr).T
    beam_lab = beam_direction_vector(beam_direction)
    coords = np.linalg.solve(direct_lab, beam_lab)
    return unit_vector(coords)


def _canonical_integer_triplet(values: tuple[int, int, int]) -> tuple[int, int, int]:
    """Canonicalize sign so that opposite directions are treated as equivalent."""
    arr = np.array(values, dtype=np.int64)
    nonzero = arr[arr != 0]
    if nonzero.size and nonzero[0] < 0:
        arr *= -1
    return int(arr[0]), int(arr[1]), int(arr[2])


def generate_zone_axes(limit: int = 4) -> IntArray:
    """Generate unique integer zone axes up to a maximum index magnitude.

    The output contains primitive integer directions only, with sign-canonicalized
    vectors so that `[u v w]` and `[-u -v -w]` are not duplicated.
    """
    if limit < 1:
        raise ValueError("zone-axis search limit must be >= 1")

    unique: set[tuple[int, int, int]] = set()
    for u in range(-limit, limit + 1):
        for v in range(-limit, limit + 1):
            for w in range(-limit, limit + 1):
                if u == 0 and v == 0 and w == 0:
                    continue
                divisor = gcd(abs(u), gcd(abs(v), abs(w)))
                if divisor != 1:
                    continue
                unique.add(_canonical_integer_triplet((u, v, w)))
    axes = np.array(sorted(unique), dtype=np.int64)
    return axes


def nearest_zone_axis(
    ub: npt.ArrayLike,
    zone_axes: npt.ArrayLike | None = None,
    zone_axis_limit: int = 4,
    beam_direction: Sequence[float] | None = None,
) -> tuple[tuple[int, int, int], float]:
    """Find the nearest low-index zone axis and the angular distance in degrees."""
    ub_arr = np.asarray(ub, dtype=np.float64)
    axes = np.asarray(zone_axes if zone_axes is not None else generate_zone_axes(zone_axis_limit), dtype=np.int64)
    direct_lab = np.linalg.inv(ub_arr).T
    axes_lab = axes @ direct_lab.T
    axis_norm = np.linalg.norm(axes_lab, axis=1)
    axes_lab = axes_lab / axis_norm[:, None]
    beam = beam_direction_vector(beam_direction)
    dots = np.clip(np.abs(axes_lab @ beam), -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    best = int(np.argmin(angles))
    nearest = tuple(int(x) for x in axes[best])
    return nearest, float(angles[best])


def compute_reciprocal_vectors(ub: npt.ArrayLike, hkls: npt.ArrayLike) -> FloatArray:
    """Map Miller indices to reciprocal-space vectors in the laboratory frame."""
    ub_arr = np.asarray(ub, dtype=np.float64)
    hkl_arr = np.atleast_2d(np.asarray(hkls, dtype=np.float64))
    return hkl_arr @ ub_arr.T


def compute_excitation_error(
    ub: npt.ArrayLike,
    hkls: npt.ArrayLike,
    wavelength_angstrom: float,
    beam_direction: Sequence[float] | None = None,
) -> FloatArray:
    """Compute a compact excitation-error proxy from Ewald-sphere geometry.

    The quantity returned here is

    `sg = |k0 + g| - |k0|`

    where `k0 = beam / wavelength` and `g = UB @ hkl`.
    """
    beam = beam_direction_vector(beam_direction)
    k0 = beam / float(wavelength_angstrom)
    g = compute_reciprocal_vectors(ub, hkls)
    shifted = g + k0[None, :]
    return _rowwise_norm(shifted) - np.linalg.norm(k0)


def generate_hkl_grid(
    cell: UnitCell,
    dmin: float,
    dmax: float | None = None,
    hkl_limit: int | None = None,
    include_friedel: bool = True,
    max_reflections: int | None = None,
) -> tuple[IntArray, FloatArray]:
    """Generate a candidate reflection grid inside a resolution shell.

    Parameters
    ----------
    cell:
        Unit-cell description.
    dmin:
        High-resolution limit in angstrom.
    dmax:
        Low-resolution limit in angstrom. If omitted, all reflections with
        `d >= dmin` are retained.
    hkl_limit:
        Optional hard cap on the index magnitude along each axis.
    include_friedel:
        Whether to keep both Friedel mates.
    max_reflections:
        Optional cap on the number of generated reflections after filtering.
        If the shell contains more reflections, the code keeps the lowest-order
        reflections first because those usually dominate dynamical interactions.
    """
    if dmin <= 0.0:
        raise ValueError("dmin must be positive")

    recip = cell_to_reciprocal_matrix(cell)
    a_star, b_star, c_star = np.linalg.norm(recip, axis=0)
    reciprocal_cutoff = 1.0 / float(dmin)
    bounds = np.ceil(reciprocal_cutoff / np.maximum([a_star, b_star, c_star], 1e-12)).astype(int) + 1
    if hkl_limit is not None:
        bounds = np.minimum(bounds, int(hkl_limit))

    h = np.arange(-bounds[0], bounds[0] + 1, dtype=np.int64)
    k = np.arange(-bounds[1], bounds[1] + 1, dtype=np.int64)
    l = np.arange(-bounds[2], bounds[2] + 1, dtype=np.int64)
    mesh = np.stack(np.meshgrid(h, k, l, indexing="ij"), axis=-1).reshape(-1, 3)
    mesh = mesh[np.any(mesh != 0, axis=1)]

    if not include_friedel:
        canonical: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for triplet in mesh:
            key = _canonical_integer_triplet((int(triplet[0]), int(triplet[1]), int(triplet[2])))
            if key not in seen:
                seen.add(key)
                canonical.append(key)
        mesh = np.asarray(canonical, dtype=np.int64)

    d_values = d_spacing(cell, mesh)
    mask = np.isfinite(d_values) & (d_values >= float(dmin))
    if dmax is not None:
        mask &= d_values <= float(dmax)
    hkls = mesh[mask]
    d_values = d_values[mask]

    if max_reflections is not None and hkls.shape[0] > int(max_reflections):
        order = np.argsort(1.0 / d_values)
        keep = order[: int(max_reflections)]
        hkls = hkls[keep]
        d_values = d_values[keep]

    return hkls.astype(np.int64), d_values.astype(np.float64)
