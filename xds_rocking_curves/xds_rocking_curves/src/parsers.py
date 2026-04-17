"""Parsers for XDS metadata and observation files.

The parser logic is intentionally close to the user's existing GXPARM/XPARM,
INTEGRATE.HKL, and XDS.INP readers, but extended to capture the detector and
beam metadata needed for detector-coordinate prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UnitCell:
    """Unit-cell parameters in angstroms and degrees."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class GXPARMData:
    """Parsed metadata from a GXPARM.XDS or XPARM.XDS file."""

    starting_frame: int
    phi0_deg: float
    dphi_deg: float
    rotation_axis: FloatArray
    wavelength_angstrom: float
    incident_beam_direction: FloatArray
    space_group: int
    unit_cell: UnitCell
    real_space_reference: FloatArray
    reciprocal_reference: FloatArray
    n_segments: int
    detector_nx: int
    detector_ny: int
    pixel_x_mm: float
    pixel_y_mm: float
    orgx_px: float
    orgy_px: float
    distance_mm: float
    detector_x_axis: FloatArray
    detector_y_axis: FloatArray
    detector_normal: FloatArray


@dataclass(frozen=True)
class IntegrateData:
    """Parsed reflection observations from INTEGRATE.HKL."""

    observations: pd.DataFrame
    estimated_n_frames: int
    header: dict[str, Any]


@dataclass(frozen=True)
class SpotData:
    """Parsed records from SPOT.XDS."""

    spots: pd.DataFrame


@dataclass(frozen=True)
class XDSInputData:
    """Optional metadata parsed from XDS.INP."""

    name_template: str | None
    data_range: tuple[int, int] | None
    untrusted_rectangles: list[tuple[float, float, float, float]]
    detector_nx: int | None
    detector_ny: int | None
    pixel_x_mm: float | None
    pixel_y_mm: float | None
    orgx_px: float | None
    orgy_px: float | None
    distance_mm: float | None
    rotation_axis: FloatArray | None
    wavelength_angstrom: float | None
    incident_beam_direction: FloatArray | None
    detector_x_axis: FloatArray | None
    detector_y_axis: FloatArray | None


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalize(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Encountered a zero-length vector while parsing XDS metadata.")
    return vector / norm


def parse_gxparm_text(text: str) -> GXPARMData:
    """Parse GXPARM/XPARM text.

    The layout follows the current XDS documentation. The three unit-cell axis
    lines are treated as a row-wise real-space matrix, mirroring the user's
    existing parser. Its inverse is the reciprocal basis with basis vectors in
    the columns, so a reciprocal-space vector can be computed as ``g = UB @ hkl``.
    """

    lines = _clean_lines(text)
    if len(lines) < 10:
        raise ValueError("GXPARM/XPARM text is too short to parse.")

    offset = 1 if not lines[0][0].isdigit() and not lines[0][0] in "-+" else 0
    numeric_lines = lines[offset:]
    if len(numeric_lines) < 10:
        raise ValueError("GXPARM/XPARM text did not contain enough numeric lines.")

    def as_numbers(index: int) -> list[float]:
        return [float(token) for token in numeric_lines[index].split()]

    line_1 = as_numbers(0)
    line_2 = as_numbers(1)
    line_3 = as_numbers(2)
    a_axis = np.asarray(as_numbers(3), dtype=np.float64)
    b_axis = np.asarray(as_numbers(4), dtype=np.float64)
    c_axis = np.asarray(as_numbers(5), dtype=np.float64)
    line_7 = as_numbers(6)
    line_8 = as_numbers(7)
    line_9 = np.asarray(as_numbers(8), dtype=np.float64)
    line_10 = np.asarray(as_numbers(9), dtype=np.float64)
    if len(numeric_lines) >= 11:
        line_11 = np.asarray(as_numbers(10), dtype=np.float64)
        detector_normal = _normalize(line_11)
    else:
        detector_normal = _normalize(np.cross(line_9, line_10))

    raw_axis = np.asarray(line_1[3:6], dtype=np.float64)
    rotation_axis = _normalize(raw_axis)
    beam_dir = _normalize(np.asarray(line_2[1:4], dtype=np.float64))
    detector_x_axis = _normalize(line_9)
    detector_y_axis = _normalize(line_10)

    real_space_reference = np.asarray([a_axis, b_axis, c_axis], dtype=np.float64)
    reciprocal_reference = np.linalg.inv(real_space_reference)
    unit_cell = UnitCell(
        a=float(line_3[1]),
        b=float(line_3[2]),
        c=float(line_3[3]),
        alpha=float(line_3[4]),
        beta=float(line_3[5]),
        gamma=float(line_3[6]),
    )
    return GXPARMData(
        starting_frame=int(round(line_1[0])),
        phi0_deg=float(line_1[1]),
        dphi_deg=float(line_1[2]),
        rotation_axis=rotation_axis,
        wavelength_angstrom=float(line_2[0]),
        incident_beam_direction=beam_dir,
        space_group=int(round(line_3[0])),
        unit_cell=unit_cell,
        real_space_reference=real_space_reference,
        reciprocal_reference=reciprocal_reference,
        n_segments=int(round(line_7[0])),
        detector_nx=int(round(line_7[1])),
        detector_ny=int(round(line_7[2])),
        pixel_x_mm=float(line_7[3]),
        pixel_y_mm=float(line_7[4]),
        orgx_px=float(line_8[0]),
        orgy_px=float(line_8[1]),
        distance_mm=float(line_8[2]),
        detector_x_axis=detector_x_axis,
        detector_y_axis=detector_y_axis,
        detector_normal=detector_normal,
    )


def parse_gxparm(path: str | Path) -> GXPARMData:
    """Parse a GXPARM/XPARM file from disk."""

    return parse_gxparm_text(Path(path).read_text())


def parse_integrate_text(text: str) -> IntegrateData:
    """Parse INTEGRATE.HKL text using the self-describing header when possible."""

    header: dict[str, Any] = {}
    rows: list[list[float]] = []
    item_map: dict[str, int] = {}
    in_data = False
    item_pattern = re.compile(r"^!ITEM_([^=]+)=(\d+)$")

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            if stripped == "!END_OF_HEADER":
                in_data = True
                continue
            match = item_pattern.match(stripped.replace(" ", ""))
            if match is not None:
                item_map[match.group(1)] = int(match.group(2)) - 1
            elif stripped.startswith("!NAME_TEMPLATE_OF_DATA_FRAMES="):
                header["name_template"] = stripped.split("=", 1)[1].rsplit(" ", 1)[0].strip()
            elif stripped.startswith("!STARTING_FRAME="):
                header["starting_frame"] = int(float(stripped.split("=", 1)[1]))
            elif stripped.startswith("!STARTING_ANGLE="):
                header["starting_angle"] = float(stripped.split("=", 1)[1])
            elif stripped.startswith("!OSCILLATION_RANGE="):
                header["oscillation_range"] = float(stripped.split("=", 1)[1])
            elif stripped.startswith("!NX="):
                parts = stripped.replace("=", " ").split()
                header["nx"] = int(parts[1])
                header["ny"] = int(parts[3])
                header["qx"] = float(parts[5])
                header["qy"] = float(parts[7])
            elif stripped.startswith("!ORGX="):
                parts = stripped.replace("=", " ").split()
                header["orgx"] = float(parts[1])
                header["orgy"] = float(parts[3])
                header["distance"] = float(parts[5])
            elif stripped.startswith("!DIRECTION_OF_DETECTOR_X-AXIS="):
                header["detector_x_axis"] = [float(v) for v in stripped.split("=", 1)[1].split()]
            elif stripped.startswith("!DIRECTION_OF_DETECTOR_Y-AXIS="):
                header["detector_y_axis"] = [float(v) for v in stripped.split("=", 1)[1].split()]
            elif stripped.startswith("!INCIDENT_BEAM_DIRECTION="):
                header["beam_direction"] = [float(v) for v in stripped.split("=", 1)[1].split()]
            continue
        if not in_data:
            continue
        try:
            rows.append([float(token) for token in stripped.split()])
        except ValueError:
            continue

    if not rows:
        raise ValueError("No reflection rows found in INTEGRATE.HKL.")

    array = np.asarray(rows, dtype=np.float64)

    def column_index(*names: str, default: int | None = None) -> int:
        for name in names:
            if name in item_map:
                return item_map[name]
        if default is None:
            raise KeyError(f"None of the requested item names were present: {names}")
        return default

    h = array[:, column_index("H", default=0)].astype(int)
    k = array[:, column_index("K", default=1)].astype(int)
    l = array[:, column_index("L", default=2)].astype(int)
    i_obs = array[:, column_index("IOBS", "I", default=3)]
    sigma = array[:, column_index("SIGMA(IOBS)", "SIGMA", default=4)]
    x_cal = array[:, column_index("XCAL", "XD", default=5)]
    y_cal = array[:, column_index("YCAL", "YD", default=6)]
    z_cal = array[:, column_index("ZCAL", "ZD", default=7)]
    x_obs = array[:, column_index("XOBS", default=12)] if array.shape[1] > 12 else np.full_like(i_obs, np.nan)
    y_obs = array[:, column_index("YOBS", default=13)] if array.shape[1] > 13 else np.full_like(i_obs, np.nan)
    z_obs = array[:, column_index("ZOBS", default=14)] if array.shape[1] > 14 else np.full_like(i_obs, np.nan)
    peak = array[:, column_index("PEAK", default=9)] if array.shape[1] > 9 else np.full_like(i_obs, np.nan)
    corr = array[:, column_index("CORR", default=10)] if array.shape[1] > 10 else np.full_like(i_obs, np.nan)
    psi = array[:, column_index("PSI", default=19)] if array.shape[1] > 19 else np.full_like(i_obs, np.nan)
    iseg = array[:, column_index("ISEG", default=20)].astype(int) if array.shape[1] > 20 else np.ones_like(h)

    observations = pd.DataFrame(
        {
            "h": h,
            "k": k,
            "l": l,
            "I": i_obs,
            "sigma": sigma,
            "x_cal": x_cal,
            "y_cal": y_cal,
            "z_cal": z_cal,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "z_obs": z_obs,
            "peak": peak,
            "corr": corr,
            "psi": psi,
            "iseg": iseg,
            "frame_est": np.floor(z_cal + 0.5).astype(int),
        }
    )
    estimated_n_frames = int(np.floor(float(np.nanmax(z_cal)))) + 1
    return IntegrateData(observations=observations, estimated_n_frames=estimated_n_frames, header=header)


def parse_integrate_hkl(path: str | Path) -> IntegrateData:
    """Parse an INTEGRATE.HKL file from disk."""

    return parse_integrate_text(Path(path).read_text())


def parse_spot_xds_text(text: str) -> SpotData:
    """Parse SPOT.XDS.

    Supports both unindexed spot lists ``x y z intensity (iseg)`` and indexed
    lists ``x y z intensity (iseg) h k l``.
    """

    records: list[dict[str, float | int | None]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        nums = [float(token) for token in parts]
        record: dict[str, float | int | None] = {
            "x": nums[0],
            "y": nums[1],
            "z": nums[2],
            "intensity": nums[3],
            "iseg": 1,
            "h": None,
            "k": None,
            "l": None,
        }
        if len(nums) in {5, 8}:
            record["iseg"] = int(round(nums[4]))
            start = 5
        else:
            start = 4
        if len(nums) - start >= 3:
            record["h"] = int(round(nums[start]))
            record["k"] = int(round(nums[start + 1]))
            record["l"] = int(round(nums[start + 2]))
        records.append(record)

    if not records:
        raise ValueError("No valid spot rows found in SPOT.XDS.")

    spots = pd.DataFrame.from_records(records)
    spots["frame_est"] = np.floor(spots["z"].to_numpy(dtype=float) + 0.5).astype(int)
    spots["indexed"] = spots[["h", "k", "l"]].notna().all(axis=1)
    return SpotData(spots=spots)


def parse_spot_xds(path: str | Path) -> SpotData:
    """Parse SPOT.XDS from disk."""

    return parse_spot_xds_text(Path(path).read_text())


def parse_xds_inp_text(text: str | None) -> XDSInputData:
    """Parse selected fields from XDS.INP text."""

    if not text:
        return XDSInputData(
            name_template=None,
            data_range=None,
            untrusted_rectangles=[],
            detector_nx=None,
            detector_ny=None,
            pixel_x_mm=None,
            pixel_y_mm=None,
            orgx_px=None,
            orgy_px=None,
            distance_mm=None,
            rotation_axis=None,
            wavelength_angstrom=None,
            incident_beam_direction=None,
            detector_x_axis=None,
            detector_y_axis=None,
        )

    def extract_float(line: str, key: str) -> float | None:
        match = re.search(rf"(?:^|\s){re.escape(key)}\s*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", line)
        if match is None:
            return None
        return float(match.group(1))

    def extract_float_array(line: str, key: str) -> FloatArray | None:
        match = re.search(rf"(?:^|\s){re.escape(key)}\s*=\s*([^!]+)", line)
        if match is None:
            return None
        try:
            values = [float(v) for v in match.group(1).split()]
        except ValueError:
            return None
        return np.asarray(values, dtype=np.float64)

    rectangles: list[tuple[float, float, float, float]] = []
    data_range: tuple[int, int] | None = None
    name_template: str | None = None
    detector_nx: int | None = None
    detector_ny: int | None = None
    pixel_x_mm: float | None = None
    pixel_y_mm: float | None = None
    orgx_px: float | None = None
    orgy_px: float | None = None
    distance_mm: float | None = None
    rotation_axis: FloatArray | None = None
    wavelength_angstrom: float | None = None
    incident_beam_direction: FloatArray | None = None
    detector_x_axis: FloatArray | None = None
    detector_y_axis: FloatArray | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("NAME_TEMPLATE_OF_DATA_FRAMES="):
            name_template = line.split("=", 1)[1].strip()
        elif line.startswith("DATA_RANGE="):
            parts = line.split("=", 1)[1].split()
            if len(parts) >= 2:
                data_range = (int(parts[0]), int(parts[1]))
        elif line.startswith("UNTRUSTED_RECTANGLE="):
            parts = line.split("=", 1)[1].split()
            if len(parts) >= 4:
                rectangles.append(tuple(float(parts[i]) for i in range(4)))
        elif line.startswith("NX="):
            value = line.split("=", 1)[1].strip()
            detector_nx = int(float(value))
        elif line.startswith("NY="):
            value = line.split("=", 1)[1].strip()
            detector_ny = int(float(value))
        elif line.startswith("QX="):
            value = line.split("=", 1)[1].strip()
            pixel_x_mm = float(value)
        elif line.startswith("QY="):
            value = line.split("=", 1)[1].strip()
            pixel_y_mm = float(value)
        else:
            value = extract_float(line, "ORGX")
            if value is not None:
                orgx_px = value
            value = extract_float(line, "ORGY")
            if value is not None:
                orgy_px = value
            value = extract_float(line, "DETECTOR_DISTANCE")
            if value is not None:
                distance_mm = value
            value = extract_float(line, "X-RAY_WAVELENGTH")
            if value is not None:
                wavelength_angstrom = value
            vector = extract_float_array(line, "ROTATION_AXIS")
            if vector is not None:
                rotation_axis = _normalize(vector)
            vector = extract_float_array(line, "INCIDENT_BEAM_DIRECTION")
            if vector is not None:
                incident_beam_direction = _normalize(vector)
            vector = extract_float_array(line, "DIRECTION_OF_DETECTOR_X-AXIS")
            if vector is not None:
                detector_x_axis = _normalize(vector)
            vector = extract_float_array(line, "DIRECTION_OF_DETECTOR_Y-AXIS")
            if vector is not None:
                detector_y_axis = _normalize(vector)

    return XDSInputData(
        name_template=name_template,
        data_range=data_range,
        untrusted_rectangles=rectangles,
        detector_nx=detector_nx,
        detector_ny=detector_ny,
        pixel_x_mm=pixel_x_mm,
        pixel_y_mm=pixel_y_mm,
        orgx_px=orgx_px,
        orgy_px=orgy_px,
        distance_mm=distance_mm,
        rotation_axis=rotation_axis,
        wavelength_angstrom=wavelength_angstrom,
        incident_beam_direction=incident_beam_direction,
        detector_x_axis=detector_x_axis,
        detector_y_axis=detector_y_axis,
    )


def parse_xds_inp(path: str | Path) -> XDSInputData:
    """Parse XDS.INP from disk."""

    return parse_xds_inp_text(Path(path).read_text())


def load_optional_xds_inp(path: str | Path | None) -> XDSInputData | None:
    """Load XDS.INP if supplied, otherwise return ``None``."""

    if path is None:
        return None
    return parse_xds_inp(path)
