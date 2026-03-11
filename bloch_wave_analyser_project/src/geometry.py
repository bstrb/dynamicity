"""Geometry and orientation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, pi, sqrt
from typing import Iterable, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .parsers import GXPARMData, UnitCell

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ZoneAxisMatch:
    """Nearest zone-axis description for a frame orientation."""

    axis: tuple[int, int, int]
    label: str
    angle_deg: float


class OrientationModel(Protocol):
    """Protocol for frame-to-orientation mapping.

    This abstraction makes it straightforward to replace the current XDS
    rotation-series reconstruction with independently indexed SerialED frame
    orientations in future work.
    """

    def rotation_matrix(self, frame_index: int, offset: float = 0.0) -> FloatArray:
        """Return the frame rotation matrix.

        Parameters
        ----------
        frame_index:
            Zero-based frame index.
        offset:
            Fractional offset within the frame, e.g. 0.0, 0.5, or 1.0.
        """


@dataclass(frozen=True)
class RotationSeriesOrientationModel:
    """Orientation model reconstructed from ``phi0``, ``dphi``, and rotation axis."""

    gxparm: GXPARMData

    def rotation_matrix(self, frame_index: int, offset: float = 0.0) -> FloatArray:
        angle_deg = self.gxparm.phi0_deg + self.gxparm.dphi_deg * (frame_index + offset)
        return rodrigues_rotation_matrix(self.gxparm.rotation_axis, angle_deg)


def rodrigues_rotation_matrix(axis: ArrayLike, angle_deg: float) -> FloatArray:
    """Return a 3x3 Rodrigues rotation matrix."""

    axis_array = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis_array)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    x, y, z = axis_array / norm
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    outer = np.outer([x, y, z], [x, y, z])
    return c * np.eye(3) + (1.0 - c) * outer + s * cross


def cell_volume(cell: UnitCell) -> float:
    """Compute the unit-cell volume in angstrom cubed."""

    alpha = np.deg2rad(cell.alpha)
    beta = np.deg2rad(cell.beta)
    gamma = np.deg2rad(cell.gamma)
    return cell.a * cell.b * cell.c * sqrt(
        1.0
        - cos(alpha) ** 2
        - cos(beta) ** 2
        - cos(gamma) ** 2
        + 2.0 * cos(alpha) * cos(beta) * cos(gamma)
    )


def generate_candidate_reflections(
    gxparm: GXPARMData,
    dmin_angstrom: float,
    dmax_angstrom: float,
) -> pd.DataFrame:
    """Generate candidate reflections within a requested d-spacing range.

    The reciprocal vectors are computed from the GXPARM reference matrix using the
    same convention as the original browser prototype: ``g_ref = UB_ref @ hkl``.
    """

    if dmin_angstrom <= 0 or dmax_angstrom <= 0:
        raise ValueError("d-spacing limits must be positive.")
    if dmin_angstrom > dmax_angstrom:
        raise ValueError("dmin must be smaller than or equal to dmax.")

    max_cell = max(gxparm.unit_cell.a, gxparm.unit_cell.b, gxparm.unit_cell.c)
    hmax = int(np.ceil(max_cell / dmin_angstrom)) + 1
    qmin = 1.0 / dmax_angstrom
    qmax = 1.0 / dmin_angstrom

    records: list[dict[str, float | int]] = []
    ub_ref = gxparm.reciprocal_reference
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                g_ref = ub_ref @ np.asarray([h, k, l], dtype=float)
                q = float(np.linalg.norm(g_ref))
                if qmin <= q <= qmax:
                    records.append(
                        {
                            "h": h,
                            "k": k,
                            "l": l,
                            "gx_ref": g_ref[0],
                            "gy_ref": g_ref[1],
                            "gz_ref": g_ref[2],
                            "q_invA": q,
                            "d_angstrom": 1.0 / q,
                        }
                    )

    reflections = pd.DataFrame.from_records(records)
    if reflections.empty:
        raise ValueError("No candidate reflections generated for the supplied d-range.")
    reflections["reflection_index"] = np.arange(len(reflections), dtype=int)
    return reflections


def rotate_reference_vectors(rotation_matrix: FloatArray, reference_vectors: FloatArray) -> FloatArray:
    """Rotate reference reciprocal vectors.

    Parameters
    ----------
    rotation_matrix:
        3x3 rotation matrix.
    reference_vectors:
        Array of shape ``(n_reflections, 3)``.
    """

    return reference_vectors @ rotation_matrix.T


def excitation_error(g_vectors: FloatArray, wavelength_angstrom: float) -> FloatArray:
    """Compute the excitation error proxy used by the original HTML script.

    The expression is the radial distance from the Ewald sphere used in the browser
    implementation, not a more advanced linearized ``s_g`` model.
    """

    inv_wavelength = 1.0 / wavelength_angstrom
    gx = g_vectors[:, 0]
    gy = g_vectors[:, 1]
    gz = g_vectors[:, 2]
    return np.sqrt(gx * gx + gy * gy + (inv_wavelength + gz) ** 2) - inv_wavelength


def project_to_detector(g_vectors: FloatArray, gxparm: GXPARMData) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Project reciprocal-space vectors to detector coordinates.

    Returns
    -------
    x_px, y_px, positive_sz_mask
        Detector coordinates in pixels and a mask identifying forward-propagating
        intersections with positive scattering-vector z component.
    """

    inv_wavelength = 1.0 / gxparm.wavelength_angstrom
    sx = g_vectors[:, 0]
    sy = g_vectors[:, 1]
    sz = inv_wavelength + g_vectors[:, 2]
    x_px = gxparm.orgx_px + (sx / sz) * (gxparm.distance_mm / gxparm.pixel_x_mm)
    y_px = gxparm.orgy_px + (sy / sz) * (gxparm.distance_mm / gxparm.pixel_y_mm)
    return x_px, y_px, sz > 0


def inside_detector(x_px: FloatArray, y_px: FloatArray, gxparm: GXPARMData) -> NDArray[np.bool_]:
    """Return a mask for reflections within detector bounds."""

    return (
        (x_px >= 0.0)
        & (x_px < gxparm.detector_nx)
        & (y_px >= 0.0)
        & (y_px < gxparm.detector_ny)
    )


def build_zone_axes(limit: int = 5) -> list[tuple[int, int, int]]:
    """Generate unique low-index zone axes for nearest-zone reporting."""

    def gcd(a: int, b: int) -> int:
        a = abs(a)
        b = abs(b)
        while b:
            a, b = b, a % b
        return a

    axes: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for u in range(-limit, limit + 1):
        for v in range(-limit, limit + 1):
            for w in range(-limit, limit + 1):
                if u == 0 and v == 0 and w == 0:
                    continue
                g = gcd(gcd(abs(u), abs(v)), abs(w))
                uu, vv, ww = u // g, v // g, w // g
                for value in (uu, vv, ww):
                    if value != 0:
                        if value < 0:
                            uu, vv, ww = -uu, -vv, -ww
                        break
                axis = (uu, vv, ww)
                if axis not in seen:
                    seen.add(axis)
                    axes.append(axis)
    return axes


def nearest_zone_axis(
    zone_axes: Iterable[tuple[int, int, int]],
    real_space_reference: FloatArray,
    rotation_matrix: FloatArray,
) -> ZoneAxisMatch:
    """Find the nearest low-index zone axis to the beam direction.

    The GXPARM rows are treated as real-space basis vectors. A real-space direction
    ``[u, v, w]`` is mapped to the lab frame by ``R @ M_ref.T @ [u, v, w]``.
    """

    mt = real_space_reference.T
    rmt = rotation_matrix @ mt
    best_axis: tuple[int, int, int] = (0, 0, 1)
    best_angle = 180.0
    for axis in zone_axes:
        direction = rmt @ np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            continue
        cos_angle = min(1.0, abs(direction[2]) / norm)
        angle_deg = float(np.rad2deg(acos(cos_angle)))
        if angle_deg < best_angle:
            best_axis = axis
            best_angle = angle_deg
    return ZoneAxisMatch(
        axis=best_axis,
        label=f"[{best_axis[0]} {best_axis[1]} {best_axis[2]}]",
        angle_deg=best_angle,
    )


def mark_untrusted_rectangles(
    x_px: FloatArray,
    y_px: FloatArray,
    rectangles: Iterable[tuple[float, float, float, float]] | None,
) -> NDArray[np.bool_]:
    """Return a mask for coordinates that fall inside any untrusted rectangle."""

    mask = np.zeros_like(x_px, dtype=bool)
    if rectangles is None:
        return mask
    for x1, x2, y1, y2 in rectangles:
        mask |= (x_px >= x1) & (x_px <= x2) & (y_px >= y1) & (y_px <= y2)
    return mask
