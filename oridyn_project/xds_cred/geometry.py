"""Reciprocal-space geometry and XDS orientation reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from oridyn.geometry import d_spacings_from_q, excitation_error, reciprocal_matrix_from_cell
from oridyn.stream_parser import UnitCell


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class XdsGeometry:
    """Parsed XPARM/GXPARM geometry."""

    source: str
    starting_frame: int
    starting_angle: float
    oscillation_range: float
    rotation_axis: FloatArray
    wavelength: float
    incident_beam_direction: FloatArray
    space_group_number: int
    unit_cell: tuple[float, float, float, float, float, float]
    direct_matrix: FloatArray
    detector_segment: int
    nx: int
    ny: int
    qx: float
    qy: float
    orgx: float
    orgy: float
    detector_distance: float
    detector_x_axis: FloatArray
    detector_y_axis: FloatArray
    detector_normal: FloatArray
    angle_offset_frames: float = 0.0
    batch_id: str = ""
    batch_start: int | None = None
    batch_end: int | None = None

    @property
    def reciprocal_matrix_zero(self) -> FloatArray:
        """Reciprocal basis columns at XDS zero spindle angle, A^-1."""

        return np.linalg.inv(self.direct_matrix).T

    @property
    def k0(self) -> FloatArray:
        """Incident wavevector in inverse angstrom."""

        return self.incident_beam_direction / self.wavelength


@dataclass(frozen=True)
class GeometryBatch:
    """Locally refined INTEGRATE image-batch geometry."""

    start_image: int
    end_image: int
    direct_matrix: FloatArray
    rotation_axis: FloatArray | None = None
    incident_beam_direction: FloatArray | None = None
    orgx: float | None = None
    orgy: float | None = None
    detector_distance: float | None = None

    @property
    def batch_id(self) -> str:
        return f"{self.start_image}-{self.end_image}"


@dataclass(frozen=True)
class XdsGeometryModel:
    """Geometry provider that can use one matrix or INTEGRATE.LP batch matrices."""

    source: str
    base: XdsGeometry
    batches: tuple[GeometryBatch, ...] = ()
    angle_offset_frames: float = 0.0
    matrix_reference: str = "unit-cell axes are the unrotated crystal at spindle dial 0 degrees"
    angle_equation: str = "phi = STARTING_ANGLE + OSCILLATION_RANGE * (ZCAL - STARTING_FRAME + angle_offset_frames)"
    batch_assignment: str = "nearest integer image to ZCAL"

    def geometry_for_z(self, zcal: float) -> XdsGeometry:
        """Return the concrete geometry matrix used for one fractional ZCAL."""

        if not self.batches:
            return replace(
                self.base,
                source=self.source,
                angle_offset_frames=self.angle_offset_frames,
                batch_id="",
                batch_start=None,
                batch_end=None,
            )
        batch = self.batch_for_z(zcal)
        return replace(
            self.base,
            source=self.source,
            direct_matrix=batch.direct_matrix,
            rotation_axis=batch.rotation_axis if batch.rotation_axis is not None else self.base.rotation_axis,
            incident_beam_direction=(
                batch.incident_beam_direction
                if batch.incident_beam_direction is not None
                else self.base.incident_beam_direction
            ),
            orgx=batch.orgx if batch.orgx is not None else self.base.orgx,
            orgy=batch.orgy if batch.orgy is not None else self.base.orgy,
            detector_distance=(
                batch.detector_distance if batch.detector_distance is not None else self.base.detector_distance
            ),
            angle_offset_frames=self.angle_offset_frames,
            batch_id=batch.batch_id,
            batch_start=batch.start_image,
            batch_end=batch.end_image,
        )

    def batch_for_z(self, zcal: float) -> GeometryBatch:
        """Select the INTEGRATE batch by nearest integer image coordinate."""

        if not self.batches:
            raise ValueError("No batches are available for this geometry model.")
        image = int(math.floor(float(zcal) + 0.5))
        for batch in self.batches:
            if batch.start_image <= image <= batch.end_image:
                return batch
        return min(
            self.batches,
            key=lambda batch: abs(image - 0.5 * (batch.start_image + batch.end_image)),
        )


GeometryLike = XdsGeometry | XdsGeometryModel


@dataclass(frozen=True)
class NeighborContribution:
    """One neighbor term in the OriDyn local reciprocal environment."""

    h: int
    k: int
    l: int
    d_gh: float
    s_neighbor: float
    E_neighbor: float
    C_gh: float
    contribution: float


@dataclass(frozen=True)
class ObservationRiskResult:
    """Transparent observation-level geometry and risk calculation."""

    h: int
    k: int
    l: int
    zcal: float
    rotation_angle: float
    geometry_source: str
    geometry_batch: str
    q_target: tuple[float, float, float]
    s_target: float
    E_target: float
    neighbor_count: int
    R_env: float
    S_risk: float
    z_pred: float
    z_residual_frames: float
    z_residual_degrees: float
    neighbors: tuple[NeighborContribution, ...]
    strongest_neighbors: tuple[NeighborContribution, ...]


def parse_xparm(path: Path, source: str | None = None) -> XdsGeometry:
    """Parse XPARM.XDS/GXPARM.XDS."""

    numbers: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            numbers.extend(float(part) for part in line.split())
    if len(numbers) < 42:
        raise ValueError(f"{path} does not contain enough numeric fields for XPARM parsing.")
    i = 0
    starting_frame = int(numbers[i])
    starting_angle = float(numbers[i + 1])
    oscillation_range = float(numbers[i + 2])
    rotation_axis = normalize(np.asarray(numbers[i + 3 : i + 6], dtype=float))
    i += 6
    wavelength = float(numbers[i])
    beam_vector = normalize(np.asarray(numbers[i + 1 : i + 4], dtype=float))
    i += 4
    space_group_number = int(numbers[i])
    unit_cell = tuple(float(x) for x in numbers[i + 1 : i + 7])
    i += 7
    direct = np.column_stack(
        [
            np.asarray(numbers[i : i + 3], dtype=float),
            np.asarray(numbers[i + 3 : i + 6], dtype=float),
            np.asarray(numbers[i + 6 : i + 9], dtype=float),
        ]
    )
    i += 9
    detector_segment = int(numbers[i])
    nx = int(numbers[i + 1])
    ny = int(numbers[i + 2])
    qx = float(numbers[i + 3])
    qy = float(numbers[i + 4])
    i += 5
    orgx = float(numbers[i])
    orgy = float(numbers[i + 1])
    detector_distance = float(numbers[i + 2])
    i += 3
    detector_x_axis = normalize(np.asarray(numbers[i : i + 3], dtype=float))
    detector_y_axis = normalize(np.asarray(numbers[i + 3 : i + 6], dtype=float))
    detector_normal = normalize(np.asarray(numbers[i + 6 : i + 9], dtype=float))
    return XdsGeometry(
        source=source or path.name,
        starting_frame=starting_frame,
        starting_angle=starting_angle,
        oscillation_range=oscillation_range,
        rotation_axis=rotation_axis,
        wavelength=wavelength,
        incident_beam_direction=beam_vector,
        space_group_number=space_group_number,
        unit_cell=unit_cell,  # type: ignore[arg-type]
        direct_matrix=direct,
        detector_segment=detector_segment,
        nx=nx,
        ny=ny,
        qx=qx,
        qy=qy,
        orgx=orgx,
        orgy=orgy,
        detector_distance=detector_distance,
        detector_x_axis=detector_x_axis,
        detector_y_axis=detector_y_axis,
        detector_normal=detector_normal,
    )


def model_from_geometry(
    geometry: XdsGeometry,
    source: str | None = None,
    angle_offset_frames: float = 0.0,
) -> XdsGeometryModel:
    """Wrap one global XDS geometry as a model."""

    return XdsGeometryModel(
        source=source or geometry.source,
        base=geometry,
        batches=(),
        angle_offset_frames=float(angle_offset_frames),
    )


def parse_integrate_lp_geometry(
    path: Path,
    base_geometry: XdsGeometry,
    angle_offset_frames: float = 1.0,
) -> XdsGeometryModel:
    """Parse batch-refined orientation matrices printed by INTEGRATE.LP."""

    batches: list[GeometryBatch] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"PROCESSING OF IMAGES\s+(\d+)\s+\.\.\.\s+(\d+)", line)
        if match:
            current = {"start_image": int(match.group(1)), "end_image": int(match.group(2))}
            continue
        if current is None:
            continue
        if "COORDINATES OF UNIT CELL A-AXIS" in line:
            current["a"] = _last_three_floats(line)
        elif "COORDINATES OF UNIT CELL B-AXIS" in line:
            current["b"] = _last_three_floats(line)
        elif "COORDINATES OF UNIT CELL C-AXIS" in line:
            current["c"] = _last_three_floats(line)
        elif "LAB COORDINATES OF ROTATION AXIS" in line:
            current["rotation_axis"] = normalize(np.asarray(_last_three_floats(line), dtype=float))
        elif "DIRECT BEAM COORDINATES" in line:
            current["incident_beam_direction"] = normalize(np.asarray(_last_three_floats(line), dtype=float))
        elif "DETECTOR COORDINATES (PIXELS) OF DIRECT BEAM" in line:
            values = _last_two_floats(line)
            current["orgx"] = values[0]
            current["orgy"] = values[1]
        elif "CRYSTAL TO DETECTOR DISTANCE" in line:
            current["detector_distance"] = _last_float(line)
        elif "LAB COORDINATES OF DETECTOR X-AXIS" in line and {"a", "b", "c"} <= set(current):
            batches.append(
                GeometryBatch(
                    start_image=int(current["start_image"]),
                    end_image=int(current["end_image"]),
                    direct_matrix=np.column_stack([current["a"], current["b"], current["c"]]).astype(float),
                    rotation_axis=current.get("rotation_axis"),
                    incident_beam_direction=current.get("incident_beam_direction"),
                    orgx=current.get("orgx"),
                    orgy=current.get("orgy"),
                    detector_distance=current.get("detector_distance"),
                )
            )
            current = None
    if not batches:
        raise ValueError(f"No batch-refined orientation matrices were parsed from {path}.")
    return XdsGeometryModel(
        source="INTEGRATE.LP batch-refined orientation",
        base=base_geometry,
        batches=tuple(batches),
        angle_offset_frames=float(angle_offset_frames),
        matrix_reference="batch-refined unit-cell axes printed by INTEGRATE.LP; axes are unrotated crystal at spindle dial 0 degrees",
        batch_assignment="nearest integer image to fractional ZCAL",
    )


def _last_float(line: str) -> float:
    return float(re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", line)[-1])


def _last_two_floats(line: str) -> list[float]:
    values = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", line)
    return [float(value) for value in values[-2:]]


def _last_three_floats(line: str) -> list[float]:
    values = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?", line)
    return [float(value) for value in values[-3:]]


def normalize(vector: FloatArray) -> FloatArray:
    """Return a normalized vector."""

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def rotation_matrix(axis: FloatArray, angle_degrees: float) -> FloatArray:
    """Rodrigues rotation matrix for a right-handed rotation."""

    unit = normalize(np.asarray(axis, dtype=float))
    theta = math.radians(float(angle_degrees))
    kx, ky, kz = unit
    skew = np.asarray([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=float)
    ident = np.eye(3, dtype=float)
    return ident * math.cos(theta) + (1.0 - math.cos(theta)) * np.outer(unit, unit) + math.sin(theta) * skew


def spindle_angle_at_z(geometry: XdsGeometry, zcal: float) -> float:
    """Return continuous spindle angle for an XDS image coordinate."""

    concrete = resolve_geometry(geometry, zcal)
    return concrete.starting_angle + concrete.oscillation_range * (
        float(zcal) - concrete.starting_frame + concrete.angle_offset_frames
    )


def resolve_geometry(geometry: GeometryLike, zcal: float) -> XdsGeometry:
    """Return a concrete geometry for this observation."""

    if isinstance(geometry, XdsGeometryModel):
        return geometry.geometry_for_z(zcal)
    return geometry


def reciprocal_matrix_at_z(geometry: GeometryLike, zcal: float) -> FloatArray:
    """Reconstruct reciprocal basis columns at fractional ZCAL."""

    concrete = resolve_geometry(geometry, zcal)
    return rotation_matrix(concrete.rotation_axis, spindle_angle_at_z(concrete, zcal)) @ concrete.reciprocal_matrix_zero


def hkl_vectors(hkls: np.ndarray, reciprocal_matrix: FloatArray) -> FloatArray:
    """Map HKL rows to lab-frame reciprocal vectors."""

    hkl_array = np.asarray(hkls, dtype=float)
    if hkl_array.ndim == 1:
        hkl_array = hkl_array.reshape(1, 3)
    return hkl_array @ reciprocal_matrix.T


def resolution_from_hkl(hkls: np.ndarray, geometry: GeometryLike) -> FloatArray:
    """Return d spacing in angstrom for HKL rows."""

    concrete = geometry.base if isinstance(geometry, XdsGeometryModel) else geometry
    q = np.linalg.norm(hkl_vectors(hkls, concrete.reciprocal_matrix_zero), axis=1)
    return d_spacings_from_q(q)


def reciprocal_from_unit_cell(cell: tuple[float, float, float, float, float, float]) -> FloatArray:
    """Return no-2pi reciprocal basis from unit-cell constants."""

    return reciprocal_matrix_from_cell(UnitCell(*cell))


def excitation_at_z(hkl: tuple[int, int, int] | np.ndarray, geometry: GeometryLike, zcal: float) -> float:
    """Calculate excitation error for one HKL at a fractional XDS coordinate."""

    concrete = resolve_geometry(geometry, zcal)
    reciprocal = reciprocal_matrix_at_z(concrete, zcal)
    q = hkl_vectors(np.asarray(hkl, dtype=float), reciprocal)
    return float(excitation_error(q, concrete.wavelength, concrete.incident_beam_direction)[0])


def detector_xy_from_q(q_vector: FloatArray, geometry: XdsGeometry) -> tuple[float, float]:
    """Project a diffracted wavevector onto the XDS detector plane."""

    kout = geometry.k0 + np.asarray(q_vector, dtype=float)
    denominator = float(kout @ geometry.detector_normal)
    if abs(denominator) < 1e-12:
        return (float("nan"), float("nan"))
    x = geometry.orgx + geometry.detector_distance * float(kout @ geometry.detector_x_axis) / (geometry.qx * denominator)
    y = geometry.orgy + geometry.detector_distance * float(kout @ geometry.detector_y_axis) / (geometry.qy * denominator)
    return (x, y)


def predict_z_by_newton(
    hkl: tuple[int, int, int],
    geometry: GeometryLike,
    z_hint: float,
    max_window_frames: float = 30.0,
) -> tuple[float, str]:
    """Predict the nearest Bragg maximum Z coordinate by local Newton iteration."""

    z = float(z_hint)
    lower = z - float(max_window_frames)
    upper = z + float(max_window_frames)
    status = "converged"
    for _ in range(12):
        value = excitation_at_z(hkl, geometry, z)
        if abs(value) < 1e-8:
            return (z, status)
        step = 0.10
        derivative = (excitation_at_z(hkl, geometry, z + step) - excitation_at_z(hkl, geometry, z - step)) / (2.0 * step)
        if not np.isfinite(derivative) or abs(derivative) < 1e-10:
            status = "derivative_too_small"
            break
        candidate = z - value / derivative
        if not (lower <= candidate <= upper):
            status = "outside_prediction_window"
            break
        if abs(candidate - z) < 1e-6:
            return (candidate, status)
        z = candidate
    return (float("nan"), status)


def audit_geometry(
    geometry: GeometryLike,
    observations: pd.DataFrame,
    prediction_window_frames: float = 30.0,
    score: Any | None = None,
    neighbor_offsets: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit how well one geometry reproduces INTEGRATE.HKL coordinates."""

    records: list[dict[str, Any]] = []
    for row in observations.itertuples(index=False):
        hkl = (int(row.H), int(row.K), int(row.L))
        zcal = float(row.ZCAL)
        concrete = resolve_geometry(geometry, zcal)
        if score is not None and neighbor_offsets is not None:
            result = calculate_observation_risk(hkl, zcal, geometry, score, neighbor_offsets)
            q_vector = np.asarray(result.q_target, dtype=float)
            s_target = result.s_target
            z_pred = result.z_pred
            z_status = "converged" if np.isfinite(z_pred) else "not_converged"
        else:
            reciprocal = reciprocal_matrix_at_z(concrete, zcal)
            q_vector = hkl_vectors(np.asarray(hkl, dtype=float), reciprocal)[0]
            s_target = float(excitation_error(q_vector, concrete.wavelength, concrete.incident_beam_direction)[0])
            z_pred, z_status = predict_z_by_newton(hkl, geometry, zcal, prediction_window_frames)
        x_pred, y_pred = detector_xy_from_q(q_vector, concrete)
        records.append(
            {
                "observation_id": int(row.observation_id),
                "geometry_source": concrete.source,
                "geometry_batch": concrete.batch_id,
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "ZCAL": zcal,
                "s_target": s_target,
                "z_pred": z_pred,
                "z_residual": z_pred - zcal if np.isfinite(z_pred) else np.nan,
                "z_residual_degrees": (
                    (z_pred - zcal) * concrete.oscillation_range if np.isfinite(z_pred) else np.nan
                ),
                "z_prediction_status": z_status,
                "XCAL": float(row.XCAL) if hasattr(row, "XCAL") else np.nan,
                "YCAL": float(row.YCAL) if hasattr(row, "YCAL") else np.nan,
                "x_pred": x_pred,
                "y_pred": y_pred,
                "x_residual": x_pred - float(row.XCAL) if hasattr(row, "XCAL") and np.isfinite(x_pred) else np.nan,
                "y_residual": y_pred - float(row.YCAL) if hasattr(row, "YCAL") and np.isfinite(y_pred) else np.nan,
            }
        )
    audit = pd.DataFrame.from_records(records)
    summary = geometry_summary(audit, geometry)
    return audit, summary


def geometry_summary(audit: pd.DataFrame, geometry: XdsGeometry) -> dict[str, Any]:
    """Summarize geometry residuals."""

    return {
        "geometry_source": geometry.source if isinstance(geometry, XdsGeometry) else geometry.source,
        "n_audited": int(len(audit)),
        "space_group_number": (geometry.space_group_number if isinstance(geometry, XdsGeometry) else geometry.base.space_group_number),
        "unit_cell": list(geometry.unit_cell if isinstance(geometry, XdsGeometry) else geometry.base.unit_cell),
        "rotation_axis": (geometry.rotation_axis if isinstance(geometry, XdsGeometry) else geometry.base.rotation_axis).tolist(),
        "beam_direction": (
            geometry.incident_beam_direction
            if isinstance(geometry, XdsGeometry)
            else geometry.base.incident_beam_direction
        ).tolist(),
        "wavelength_angstrom": geometry.wavelength if isinstance(geometry, XdsGeometry) else geometry.base.wavelength,
        "orientation_convention": (
            "B(ZCAL) = R(axis, STARTING_ANGLE + OSCILLATION_RANGE * "
            "(ZCAL - STARTING_FRAME + angle_offset_frames)) @ B_zero"
        ),
        "angle_offset_frames": (
            geometry.angle_offset_frames if isinstance(geometry, XdsGeometry) else geometry.angle_offset_frames
        ),
        "batch_count": 0 if isinstance(geometry, XdsGeometry) else len(geometry.batches),
        "median_abs_s_target_invA": _median_abs(audit["s_target"]),
        "p95_abs_s_target_invA": _quantile_abs(audit["s_target"], 0.95),
        "median_z_residual_frames": _median(audit["z_residual"]),
        "median_abs_z_residual_frames": _median_abs(audit["z_residual"]),
        "p95_abs_z_residual_frames": _quantile_abs(audit["z_residual"], 0.95),
        "median_abs_z_residual_degrees": _median_abs(audit["z_residual_degrees"]),
        "median_abs_x_residual_px": _median_abs(audit["x_residual"]),
        "median_abs_y_residual_px": _median_abs(audit["y_residual"]),
    }


def calculate_observation_risk(
    hkl: tuple[int, int, int] | np.ndarray,
    zcal: float,
    geometry: GeometryLike,
    score: Any,
    neighbor_offsets: np.ndarray,
    strongest_count: int = 10,
) -> ObservationRiskResult:
    """Calculate one transparent observation-level OriDyn risk result."""

    concrete = resolve_geometry(geometry, zcal)
    hkl_array = np.asarray(hkl, dtype=int).reshape(3)
    reciprocal = reciprocal_matrix_at_z(concrete, zcal)
    target_q = hkl_vectors(hkl_array, reciprocal)[0]
    s_target = float(excitation_error(target_q, concrete.wavelength, concrete.incident_beam_direction)[0])
    s0 = max(float(score.s0), 1e-12)
    sigma_c = max(float(score.sigma_c), 1e-12)
    r_cut = float(score.r_cut)
    e_target = float(np.exp(-((s_target / s0) ** 2)))
    neighbor_rows: list[NeighborContribution] = []
    if len(neighbor_offsets):
        neighbor_hkls = hkl_array[None, :] + np.asarray(neighbor_offsets, dtype=int)
        neighbor_q = hkl_vectors(neighbor_hkls, reciprocal)
        distances = np.linalg.norm(neighbor_q - target_q[None, :], axis=1)
        mask = distances <= r_cut + 1e-12
        if np.any(mask):
            selected_hkls = neighbor_hkls[mask]
            selected_q = neighbor_q[mask]
            selected_distances = distances[mask]
            s_neighbors = excitation_error(selected_q, concrete.wavelength, concrete.incident_beam_direction)
            e_neighbors = np.exp(-((s_neighbors / s0) ** 2))
            c_gh = np.exp(-((selected_distances / sigma_c) ** 2))
            contributions = e_neighbors * c_gh
            for hkl_neighbor, distance, s_neighbor, e_neighbor, c_value, contribution in zip(
                selected_hkls,
                selected_distances,
                s_neighbors,
                e_neighbors,
                c_gh,
                contributions,
                strict=True,
            ):
                neighbor_rows.append(
                    NeighborContribution(
                        h=int(hkl_neighbor[0]),
                        k=int(hkl_neighbor[1]),
                        l=int(hkl_neighbor[2]),
                        d_gh=float(distance),
                        s_neighbor=float(s_neighbor),
                        E_neighbor=float(e_neighbor),
                        C_gh=float(c_value),
                        contribution=float(contribution),
                    )
                )
    r_env = float(sum(row.contribution for row in neighbor_rows))
    s_risk = float(e_target * r_env)
    z_pred, _ = predict_z_by_newton(tuple(int(x) for x in hkl_array), concrete, zcal)
    z_residual = z_pred - float(zcal) if np.isfinite(z_pred) else float("nan")
    strongest = tuple(
        sorted(neighbor_rows, key=lambda item: (-item.contribution, item.h, item.k, item.l))[:strongest_count]
    )
    return ObservationRiskResult(
        h=int(hkl_array[0]),
        k=int(hkl_array[1]),
        l=int(hkl_array[2]),
        zcal=float(zcal),
        rotation_angle=spindle_angle_at_z(concrete, zcal),
        geometry_source=concrete.source,
        geometry_batch=concrete.batch_id,
        q_target=(float(target_q[0]), float(target_q[1]), float(target_q[2])),
        s_target=s_target,
        E_target=e_target,
        neighbor_count=len(neighbor_rows),
        R_env=r_env,
        S_risk=s_risk,
        z_pred=z_pred,
        z_residual_frames=z_residual,
        z_residual_degrees=z_residual * concrete.oscillation_range if np.isfinite(z_residual) else float("nan"),
        neighbors=tuple(neighbor_rows),
        strongest_neighbors=strongest,
    )


def choose_geometry(
    summaries: list[dict[str, Any]],
    max_median_abs_z_residual_frames: float,
    max_median_abs_excitation_invA: float,
) -> tuple[str, str]:
    """Choose the geometry with the best ZCAL agreement."""

    usable = [
        item
        for item in summaries
        if np.isfinite(item.get("median_abs_s_target_invA", np.nan))
        and np.isfinite(item.get("median_abs_z_residual_frames", np.nan))
    ]
    if not usable:
        raise RuntimeError("Neither XPARM.XDS nor GXPARM.XDS produced finite geometry-audit residuals.")
    usable.sort(key=lambda item: (item["median_abs_z_residual_frames"], item["median_abs_s_target_invA"], item["geometry_source"]))
    best = usable[0]
    reason = (
        f"Selected {best['geometry_source']} because it had the smallest median absolute Z residual "
        f"({best['median_abs_z_residual_frames']:.4g} frames) among finite geometry audits; "
        f"median |s_g(ZCAL)| was {best['median_abs_s_target_invA']:.4g} A^-1."
    )
    if (
        best["median_abs_z_residual_frames"] > max_median_abs_z_residual_frames
        or best["median_abs_s_target_invA"] > max_median_abs_excitation_invA
    ):
        raise RuntimeError(
            "No geometry reproduced the integration geometry convincingly. "
            f"Best candidate was {best['geometry_source']} with median |Zpred-ZCAL|="
            f"{best['median_abs_z_residual_frames']:.4g} frames and median |s_g|="
            f"{best['median_abs_s_target_invA']:.4g} A^-1."
        )
    return str(best["geometry_source"]), reason


def _median(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _median_abs(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    return float(np.median(arr)) if arr.size else float("nan")


def _quantile_abs(values: pd.Series, quantile: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    return float(np.quantile(arr, quantile)) if arr.size else float("nan")
