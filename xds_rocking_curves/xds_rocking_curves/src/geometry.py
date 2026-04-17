"""Geometry and detector-projection helpers for XDS-based rocking curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from .parsers import GXPARMData, UnitCell, XDSInputData

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DetectorGeometry:
    """Detector geometry in laboratory coordinates."""

    nx: int
    ny: int
    pixel_x_mm: float
    pixel_y_mm: float
    orgx_px: float
    orgy_px: float
    distance_mm: float
    detector_x_axis: FloatArray
    detector_y_axis: FloatArray
    detector_normal: FloatArray
    incident_beam_direction: FloatArray


@dataclass(frozen=True)
class PredictionResult:
    """Prediction output for one reflection on one frame."""

    frame: float
    phi_deg: float
    x_pred: float
    y_pred: float
    sg: float
    on_detector: bool
    valid: bool


@dataclass(frozen=True)
class RotationCalibration:
    """Chosen rotation sign and its validation score."""

    rotation_sign: float
    median_pixel_error: float | None


def normalize(vector: FloatArray) -> FloatArray:
    """Return a unit vector."""

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def rotation_matrix_from_axis_angle(axis: FloatArray, angle_deg: float) -> FloatArray:
    """Return a Rodrigues rotation matrix from axis and angle in degrees."""

    ax = normalize(np.asarray(axis, dtype=np.float64))
    angle_rad = radians(float(angle_deg))
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


def detector_geometry_from_xds(
    gxparm: GXPARMData,
    xds_inp: XDSInputData | None = None,
) -> DetectorGeometry:
    """Combine GXPARM and optional XDS.INP values into a detector model."""

    nx = xds_inp.detector_nx if xds_inp and xds_inp.detector_nx is not None else gxparm.detector_nx
    ny = xds_inp.detector_ny if xds_inp and xds_inp.detector_ny is not None else gxparm.detector_ny
    qx = xds_inp.pixel_x_mm if xds_inp and xds_inp.pixel_x_mm is not None else gxparm.pixel_x_mm
    qy = xds_inp.pixel_y_mm if xds_inp and xds_inp.pixel_y_mm is not None else gxparm.pixel_y_mm
    orgx = xds_inp.orgx_px if xds_inp and xds_inp.orgx_px is not None else gxparm.orgx_px
    orgy = xds_inp.orgy_px if xds_inp and xds_inp.orgy_px is not None else gxparm.orgy_px
    distance = xds_inp.distance_mm if xds_inp and xds_inp.distance_mm is not None else gxparm.distance_mm
    det_x = normalize(xds_inp.detector_x_axis if xds_inp and xds_inp.detector_x_axis is not None else gxparm.detector_x_axis)
    det_y = normalize(xds_inp.detector_y_axis if xds_inp and xds_inp.detector_y_axis is not None else gxparm.detector_y_axis)
    det_n = normalize(np.cross(det_x, det_y))
    beam = normalize(
        xds_inp.incident_beam_direction
        if xds_inp and xds_inp.incident_beam_direction is not None
        else gxparm.incident_beam_direction
    )
    if float(np.dot(beam, det_n)) <= 0.0:
        det_n = -det_n
    return DetectorGeometry(
        nx=nx,
        ny=ny,
        pixel_x_mm=qx,
        pixel_y_mm=qy,
        orgx_px=orgx,
        orgy_px=orgy,
        distance_mm=distance,
        detector_x_axis=det_x,
        detector_y_axis=det_y,
        detector_normal=det_n,
        incident_beam_direction=beam,
    )


def frame_to_phi_deg(
    gxparm: GXPARMData,
    frame_number: float,
    rotation_sign: float = 1.0,
) -> float:
    """Convert a frame number (can be fractional) into spindle angle in degrees."""

    delta_frames = float(frame_number) - float(gxparm.starting_frame)
    return float(gxparm.phi0_deg + rotation_sign * gxparm.dphi_deg * delta_frames)


def reciprocal_basis_for_frame(
    gxparm: GXPARMData,
    frame_number: float,
    rotation_sign: float = 1.0,
) -> FloatArray:
    """Return the reciprocal basis at a frame as a 3x3 matrix with basis columns."""

    phi_deg = frame_to_phi_deg(gxparm, frame_number, rotation_sign=rotation_sign)
    rot = rotation_matrix_from_axis_angle(gxparm.rotation_axis, phi_deg)
    return rot @ gxparm.reciprocal_reference


def reciprocal_vector_for_hkl(
    gxparm: GXPARMData,
    frame_number: float,
    hkl: tuple[int, int, int],
    rotation_sign: float = 1.0,
) -> FloatArray:
    """Map a Miller index to a reciprocal-space vector in the lab frame."""

    basis = reciprocal_basis_for_frame(gxparm, frame_number, rotation_sign=rotation_sign)
    return basis @ np.asarray(hkl, dtype=np.float64)


def excitation_error(
    gxparm: GXPARMData,
    frame_number: float,
    hkl: tuple[int, int, int],
    beam_direction: FloatArray | None = None,
    wavelength_angstrom: float | None = None,
    rotation_sign: float = 1.0,
) -> float:
    """Compact Ewald-sphere excitation-error proxy.

    Returns ``|k0 + g| - |k0|`` in inverse angstrom.
    """

    beam = normalize(np.asarray(beam_direction if beam_direction is not None else gxparm.incident_beam_direction, dtype=np.float64))
    wavelength = float(wavelength_angstrom if wavelength_angstrom is not None else gxparm.wavelength_angstrom)
    k0 = beam / wavelength
    g = reciprocal_vector_for_hkl(gxparm, frame_number, hkl, rotation_sign=rotation_sign)
    return float(np.linalg.norm(k0 + g) - np.linalg.norm(k0))


def diffracted_direction(
    gxparm: GXPARMData,
    frame_number: float,
    hkl: tuple[int, int, int],
    beam_direction: FloatArray | None = None,
    wavelength_angstrom: float | None = None,
    rotation_sign: float = 1.0,
) -> FloatArray:
    """Approximate diffracted-ray direction for one reflection on one frame."""

    beam = normalize(np.asarray(beam_direction if beam_direction is not None else gxparm.incident_beam_direction, dtype=np.float64))
    wavelength = float(wavelength_angstrom if wavelength_angstrom is not None else gxparm.wavelength_angstrom)
    k0 = beam / wavelength
    g = reciprocal_vector_for_hkl(gxparm, frame_number, hkl, rotation_sign=rotation_sign)
    return normalize(k0 + g)


def project_direction_to_detector(direction: FloatArray, detector: DetectorGeometry) -> tuple[float, float, bool]:
    """Project a ray direction onto the detector and return XDS pixel coordinates."""

    d = normalize(np.asarray(direction, dtype=np.float64))
    denom = float(np.dot(d, detector.detector_normal))
    if denom <= 0.0:
        return float("nan"), float("nan"), False

    beam_dir = detector.incident_beam_direction
    beam_intersection = beam_dir * (detector.distance_mm / float(np.dot(beam_dir, detector.detector_normal)))
    t = detector.distance_mm / denom
    point = d * t
    delta = point - beam_intersection
    x_mm = float(np.dot(delta, detector.detector_x_axis))
    y_mm = float(np.dot(delta, detector.detector_y_axis))
    x_px = detector.orgx_px + x_mm / detector.pixel_x_mm
    y_px = detector.orgy_px + y_mm / detector.pixel_y_mm
    on_detector = 1.0 <= x_px <= float(detector.nx) and 1.0 <= y_px <= float(detector.ny)
    return x_px, y_px, on_detector


def predict_reflection_on_frame(
    gxparm: GXPARMData,
    detector: DetectorGeometry,
    hkl: tuple[int, int, int],
    frame_number: float,
    rotation_sign: float = 1.0,
) -> PredictionResult:
    """Predict detector coordinates and excitation error for one frame."""

    phi_deg = frame_to_phi_deg(gxparm, frame_number, rotation_sign=rotation_sign)
    sg = excitation_error(gxparm, frame_number, hkl, rotation_sign=rotation_sign)
    direction = diffracted_direction(gxparm, frame_number, hkl, rotation_sign=rotation_sign)
    x_px, y_px, on_detector = project_direction_to_detector(direction, detector)
    valid = bool(np.isfinite(x_px) and np.isfinite(y_px))
    return PredictionResult(
        frame=float(frame_number),
        phi_deg=phi_deg,
        x_pred=float(x_px),
        y_pred=float(y_px),
        sg=float(sg),
        on_detector=bool(on_detector),
        valid=valid,
    )


def beam_direction_in_crystal(
    gxparm: GXPARMData,
    frame_number: float,
    rotation_sign: float = 1.0,
) -> FloatArray:
    """Express the beam direction in crystal direct-space coordinates."""

    reciprocal_basis = reciprocal_basis_for_frame(gxparm, frame_number, rotation_sign=rotation_sign)
    direct_basis = np.linalg.inv(reciprocal_basis).T
    coords = np.linalg.solve(direct_basis, gxparm.incident_beam_direction)
    return normalize(coords)


def choose_rotation_sign_from_integrate(
    gxparm: GXPARMData,
    detector: DetectorGeometry,
    integrate_observations,
    sample_size: int = 200,
) -> RotationCalibration:
    """Choose the rotation sign by comparing to XCAL/YCAL in INTEGRATE.HKL.

    If no integrate observations are supplied, ``+1`` is returned.
    """

    if integrate_observations is None or len(integrate_observations) == 0:
        return RotationCalibration(rotation_sign=1.0, median_pixel_error=None)

    obs = integrate_observations[["h", "k", "l", "x_cal", "y_cal", "z_cal"]].dropna()
    if obs.empty:
        return RotationCalibration(rotation_sign=1.0, median_pixel_error=None)

    if len(obs) > sample_size:
        obs = obs.sample(sample_size, random_state=0)

    best_sign = 1.0
    best_score = float("inf")
    for sign in (1.0, -1.0):
        errors: list[float] = []
        for row in obs.itertuples(index=False):
            pred = predict_reflection_on_frame(
                gxparm=gxparm,
                detector=detector,
                hkl=(int(row.h), int(row.k), int(row.l)),
                frame_number=float(row.z_cal),
                rotation_sign=sign,
            )
            if not pred.valid:
                continue
            errors.append(float(np.hypot(pred.x_pred - float(row.x_cal), pred.y_pred - float(row.y_cal))))
        if errors:
            score = float(np.median(errors))
            if score < best_score:
                best_score = score
                best_sign = sign
    if not np.isfinite(best_score):
        best_score = None
    return RotationCalibration(rotation_sign=best_sign, median_pixel_error=best_score)
