"""Parsers for XDS and composition inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import FE0

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

    phi0_deg: float
    dphi_deg: float
    rotation_axis: FloatArray
    wavelength_angstrom: float
    space_group: int
    unit_cell: UnitCell
    real_space_reference: FloatArray
    reciprocal_reference: FloatArray
    detector_nx: int
    detector_ny: int
    pixel_x_mm: float
    pixel_y_mm: float
    orgx_px: float
    orgy_px: float
    distance_mm: float


@dataclass(frozen=True)
class IntegrateData:
    """Parsed reflection observations from INTEGRATE.HKL."""

    observations: pd.DataFrame
    estimated_n_frames: int


@dataclass(frozen=True)
class CrystFELStreamData:
    """Parsed CrystFEL stream content for snapshot-style SerialED analysis."""

    wavelength_angstrom: float
    distance_mm: float
    pixel_x_mm: float
    pixel_y_mm: float
    detector_nx: int
    detector_ny: int
    orgx_px: float
    orgy_px: float
    unit_cell: UnitCell
    crystal_table: pd.DataFrame
    reflections: pd.DataFrame


@dataclass(frozen=True)
class PETSProjectData:
    """Parsed PETS project metadata plus integrated reflection rows."""

    pts_path: Path
    rprofall_path: Path
    wavelength_angstrom: float
    aperpixel_invA_per_px: float
    unit_cell: UnitCell
    reciprocal_reference: FloatArray
    detector_nx: int
    detector_ny: int
    orgx_px: float
    orgy_px: float
    imagelist: pd.DataFrame
    rprofall: RProfallData


@dataclass(frozen=True)
class RProfallData:
    """Parsed PETS2 ``.rprofall`` reflection rows."""

    rows: pd.DataFrame
    n_blocks: int


@dataclass(frozen=True)
class XDSInputData:
    """Optional metadata parsed from XDS.INP."""

    untrusted_rectangles: list[tuple[float, float, float, float]]
    data_range: tuple[int, int] | None


@dataclass(frozen=True)
class CompositionEntry:
    """One composition term such as ``24 Si``."""

    count: float
    element: str
    forward_scattering_factor: float


@dataclass(frozen=True)
class CompositionResult:
    """Parsed composition and the Wilson scaling proxy ``sum_fj2``."""

    entries: tuple[CompositionEntry, ...]
    sum_fj2: float


RPROFALL_ROW_WIDTH = 109
RPROFALL_HEADER_RE = re.compile(r"^\s*#\s*(\d+)\s*$")
RPROFALL_SCHEMA: tuple[tuple[str, tuple[int, int], str], ...] = (
    ("h", (0, 4), "int"),
    ("k", (4, 8), "int"),
    ("l", (8, 12), "int"),
    ("resolution", (12, 26), "float"),
    ("excitation", (26, 40), "float"),
    ("rsg", (40, 54), "float"),
    ("iobs", (54, 68), "float"),
    ("sigma", (68, 82), "float"),
    ("icalc", (82, 96), "float"),
    ("frame", (96, 100), "int"),
    ("azimuth", (100, 109), "float"),
)
RPROFALL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "row_id",
    "block_id",
    "h",
    "k",
    "l",
    "resolution",
    "excitation",
    "rsg",
    "iobs",
    "sigma",
    "icalc",
    "frame",
    "azimuth",
)

STREAM_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
STREAM_WAVELENGTH_RE = re.compile(rf"^\s*wavelength\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_CLEN_RE = re.compile(rf"^\s*clen\s*=\s*({STREAM_FLOAT})\s*m")
STREAM_RES_RE = re.compile(rf"^\s*res\s*=\s*({STREAM_FLOAT})")
STREAM_AVERAGE_CLEN_RE = re.compile(rf"^\s*average_camera_length\s*=\s*({STREAM_FLOAT})\s*m")
STREAM_UNITCELL_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_UNITCELL_ANGLE_RE = re.compile(rf"^\s*(al|be|ga)\s*=\s*({STREAM_FLOAT})\s*deg")
STREAM_PANEL_RANGE_RE = re.compile(r"^\s*(p\d+)\/(min_fs|max_fs|min_ss|max_ss)\s*=\s*(-?\d+)")
STREAM_PANEL_CORNER_RE = re.compile(rf"^\s*(p\d+)\/corner_([xy])\s*=\s*({STREAM_FLOAT})")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
STREAM_SERIAL_RE = re.compile(r"^\s*Image serial number:\s*(\d+)")
STREAM_CELL_RE = re.compile(
    rf"^\s*Cell parameters\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+nm,"
    rf"\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+deg"
)
STREAM_VECTOR_RE = re.compile(
    rf"^\s*([abc])star\s*=\s*({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+nm\^-1"
)
STREAM_MATRIX_COLUMNS: tuple[str, ...] = (
    "UB11",
    "UB12",
    "UB13",
    "UB21",
    "UB22",
    "UB23",
    "UB31",
    "UB32",
    "UB33",
)

PETS_IMAGELIST_DEFAULT_HEADER: tuple[str, ...] = (
    "imgname",
    "alpha",
    "beta",
    "domega",
    "alphaorig",
    "betaorig",
    "domegaorig",
    "xcenter",
    "ycenter",
    "intscale",
    "diffbfac",
    "magcorr",
    "elliamp",
    "elliph",
    "paraamp",
    "paraph",
    "useforcalc",
    "dataset",
)


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_fixed_int(field: str) -> int | None:
    token = field.strip()
    if not token:
        return None
    if "*" in token:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _parse_fixed_float(field: str) -> float:
    token = field.strip()
    if not token:
        return float("nan")
    if "*" in token:
        return float("nan")
    try:
        return float(token)
    except ValueError:
        return float("nan")


def _real_space_matrix_from_unit_cell(cell: UnitCell) -> FloatArray:
    """Construct a real-space basis matrix from unit-cell parameters.

    The returned matrix follows the same row-wise convention used by GXPARM
    parsing: each row is one direct basis vector in Cartesian coordinates.
    """

    alpha = np.deg2rad(cell.alpha)
    beta = np.deg2rad(cell.beta)
    gamma = np.deg2rad(cell.gamma)
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) < 1e-12:
        raise ValueError("Unit-cell gamma is too close to 0 or 180 degrees.")

    cos_alpha = float(np.cos(alpha))
    cos_beta = float(np.cos(beta))
    cos_gamma = float(np.cos(gamma))

    a_vec = np.asarray([cell.a, 0.0, 0.0], dtype=float)
    b_vec = np.asarray([cell.b * cos_gamma, cell.b * sin_gamma, 0.0], dtype=float)

    c_x = cell.c * cos_beta
    c_y = cell.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_sq = max(cell.c * cell.c - c_x * c_x - c_y * c_y, 0.0)
    c_vec = np.asarray([c_x, c_y, np.sqrt(c_z_sq)], dtype=float)

    return np.asarray([a_vec, b_vec, c_vec], dtype=float)


def _parse_stream_reflection_row(line: str) -> dict[str, float | int | str] | None:
    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        i_obs = float(parts[3])
        sigma = float(parts[4])
        peak = float(parts[5])
        background = float(parts[6])
        fs_px = float(parts[7])
        ss_px = float(parts[8])
    except ValueError:
        return None

    panel = parts[9] if len(parts) > 9 else ""
    return {
        "h": h,
        "k": k,
        "l": l,
        "I": i_obs,
        "sigma": sigma,
        "peak": peak,
        "background": background,
        "fs_px": fs_px,
        "ss_px": ss_px,
        "panel": panel,
    }


def _rotation_matrix_x_deg(angle_deg: float) -> FloatArray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _rotation_matrix_y_deg(angle_deg: float) -> FloatArray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rotation_matrix_z_deg(angle_deg: float) -> FloatArray:
    theta = np.deg2rad(float(angle_deg))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _pets_rotation_from_angles_deg(alpha_deg: float, beta_deg: float, domega_deg: float) -> FloatArray:
    """Compose PETS per-frame angles into a 3D rotation matrix.

    The convention used here is a practical approximation:
    ``R = Rz(domega) * Ry(beta) * Rx(alpha)``.
    """

    return (
        _rotation_matrix_z_deg(domega_deg)
        @ _rotation_matrix_y_deg(beta_deg)
        @ _rotation_matrix_x_deg(alpha_deg)
    )


def parse_gxparm_text(text: str) -> GXPARMData:
    """Parse GXPARM/XPARM text.

    Notes
    -----
    The line layout mirrors the assumptions used by the original HTML analyser.
    The three rows following the cell line are treated as a real-space reference
    basis matrix whose inverse is used as the reciprocal reference matrix.
    """

    lines = _clean_lines(text)
    if len(lines) < 9:
        raise ValueError("GXPARM/XPARM text is too short to parse.")

    def as_numbers(index: int) -> list[float]:
        return [float(token) for token in lines[index].split()]

    line_1 = as_numbers(1)
    line_2 = as_numbers(2)
    line_3 = as_numbers(3)
    m_ref = np.asarray([as_numbers(4), as_numbers(5), as_numbers(6)], dtype=float)
    line_7 = as_numbers(7)
    line_8 = as_numbers(8)

    raw_axis = np.asarray(line_1[3:6], dtype=float)
    axis_norm = np.linalg.norm(raw_axis)
    if axis_norm == 0:
        raise ValueError("Rotation axis has zero norm.")

    reciprocal_reference = np.linalg.inv(m_ref)
    unit_cell = UnitCell(
        a=float(line_3[1]),
        b=float(line_3[2]),
        c=float(line_3[3]),
        alpha=float(line_3[4]),
        beta=float(line_3[5]),
        gamma=float(line_3[6]),
    )
    return GXPARMData(
        phi0_deg=float(line_1[1]),
        dphi_deg=float(line_1[2]),
        rotation_axis=raw_axis / axis_norm,
        wavelength_angstrom=float(line_2[0]),
        space_group=int(line_3[0]),
        unit_cell=unit_cell,
        real_space_reference=m_ref,
        reciprocal_reference=reciprocal_reference,
        detector_nx=int(line_7[1]),
        detector_ny=int(line_7[2]),
        pixel_x_mm=float(line_7[3]),
        pixel_y_mm=float(line_7[4]),
        orgx_px=float(line_8[0]),
        orgy_px=float(line_8[1]),
        distance_mm=float(line_8[2]),
    )


def parse_gxparm(path: str | Path) -> GXPARMData:
    """Parse a GXPARM/XPARM file from disk."""

    return parse_gxparm_text(Path(path).read_text())


def parse_crystfel_stream_text(text: str) -> CrystFELStreamData:
    """Parse a CrystFEL ``.stream`` file for snapshot-style SerialED analysis.

    Notes
    -----
    The parser keeps one orientation (``astar``, ``bstar``, ``cstar``) per crystal
    block and one frame index per crystal, so downstream analysis can treat each
    indexed crystal as an independent snapshot.
    """

    wavelength_angstrom: float | None = None
    clen_m: float | None = None
    res_px_per_m: float | None = None
    average_camera_lengths_m: list[float] = []

    unit_cell_lengths: dict[str, float] = {}
    unit_cell_angles: dict[str, float] = {}

    panel_ranges: dict[str, dict[str, float]] = {}
    panel_corners: dict[str, dict[str, float]] = {}

    crystal_rows: list[dict[str, float | int | str]] = []
    reflection_rows: list[dict[str, float | int | str]] = []

    in_chunk = False
    in_crystal = False
    in_reflections = False
    current_chunk_id = -1
    current_event: str | None = None
    current_serial: int | None = None
    crystal_counter = 0
    crystal_in_chunk = 0

    current_vectors_nm_invA: dict[str, np.ndarray] = {}
    current_cell_angstrom: UnitCell | None = None
    pending_reflections: list[dict[str, float | int | str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if raw_line.startswith("----- Begin chunk -----"):
            in_chunk = True
            in_crystal = False
            in_reflections = False
            current_chunk_id += 1
            crystal_in_chunk = 0
            current_event = None
            current_serial = None
            continue
        if raw_line.startswith("----- End chunk -----"):
            in_chunk = False
            in_crystal = False
            in_reflections = False
            continue

        if not in_chunk:
            match = STREAM_WAVELENGTH_RE.match(raw_line)
            if match is not None:
                wavelength_angstrom = float(match.group(1))
                continue
            match = STREAM_CLEN_RE.match(raw_line)
            if match is not None:
                clen_m = float(match.group(1))
                continue
            match = STREAM_RES_RE.match(raw_line)
            if match is not None:
                res_px_per_m = float(match.group(1))
                continue
            match = STREAM_UNITCELL_LENGTH_RE.match(raw_line)
            if match is not None:
                unit_cell_lengths[match.group(1)] = float(match.group(2))
                continue
            match = STREAM_UNITCELL_ANGLE_RE.match(raw_line)
            if match is not None:
                unit_cell_angles[match.group(1)] = float(match.group(2))
                continue
            match = STREAM_PANEL_RANGE_RE.match(raw_line)
            if match is not None:
                panel_name = match.group(1)
                field = match.group(2)
                value = float(match.group(3))
                panel_ranges.setdefault(panel_name, {})[field] = value
                continue
            match = STREAM_PANEL_CORNER_RE.match(raw_line)
            if match is not None:
                panel_name = match.group(1)
                axis = match.group(2)
                value = float(match.group(3))
                panel_corners.setdefault(panel_name, {})[axis] = value
                continue
            continue

        match = STREAM_EVENT_RE.match(raw_line)
        if match is not None:
            current_event = match.group(1)
            continue
        match = STREAM_SERIAL_RE.match(raw_line)
        if match is not None:
            current_serial = int(match.group(1))
            continue
        match = STREAM_AVERAGE_CLEN_RE.match(raw_line)
        if match is not None:
            average_camera_lengths_m.append(float(match.group(1)))
            continue

        if raw_line.startswith("--- Begin crystal"):
            in_crystal = True
            in_reflections = False
            current_vectors_nm_invA = {}
            current_cell_angstrom = None
            pending_reflections = []
            crystal_in_chunk += 1
            continue

        if raw_line.startswith("--- End crystal"):
            if not in_crystal:
                continue
            missing_vectors = [axis for axis in ("a", "b", "c") if axis not in current_vectors_nm_invA]
            if missing_vectors:
                raise ValueError(
                    f"Crystal in chunk {current_chunk_id} is missing reciprocal vectors: {missing_vectors}"
                )

            ub_matrix = np.vstack(
                [
                    current_vectors_nm_invA["a"],
                    current_vectors_nm_invA["b"],
                    current_vectors_nm_invA["c"],
                ]
            ) / 10.0

            frame_index = crystal_counter
            frame_number = frame_index + 1
            cell = current_cell_angstrom
            crystal_row: dict[str, float | int | str] = {
                "frame": frame_index,
                "frame_number": frame_number,
                "chunk_id": current_chunk_id,
                "crystal_in_chunk": crystal_in_chunk,
                "event": current_event or f"chunk{current_chunk_id}_crystal{crystal_in_chunk}",
                "image_serial": -1 if current_serial is None else int(current_serial),
                "a_angstrom": float("nan") if cell is None else float(cell.a),
                "b_angstrom": float("nan") if cell is None else float(cell.b),
                "c_angstrom": float("nan") if cell is None else float(cell.c),
                "alpha_deg": float("nan") if cell is None else float(cell.alpha),
                "beta_deg": float("nan") if cell is None else float(cell.beta),
                "gamma_deg": float("nan") if cell is None else float(cell.gamma),
            }
            for idx, value in enumerate(ub_matrix.reshape(-1)):
                crystal_row[STREAM_MATRIX_COLUMNS[idx]] = float(value)
            crystal_rows.append(crystal_row)

            for row in pending_reflections:
                row["frame"] = frame_index
                row["frame_number"] = frame_number
                row["chunk_id"] = current_chunk_id
                row["crystal_in_chunk"] = crystal_in_chunk
                row["event"] = crystal_row["event"]
                row["image_serial"] = crystal_row["image_serial"]
                reflection_rows.append(row)

            crystal_counter += 1
            in_crystal = False
            in_reflections = False
            current_vectors_nm_invA = {}
            current_cell_angstrom = None
            pending_reflections = []
            continue

        if not in_crystal:
            continue

        match = STREAM_CELL_RE.match(raw_line)
        if match is not None:
            current_cell_angstrom = UnitCell(
                a=10.0 * float(match.group(1)),
                b=10.0 * float(match.group(2)),
                c=10.0 * float(match.group(3)),
                alpha=float(match.group(4)),
                beta=float(match.group(5)),
                gamma=float(match.group(6)),
            )
            continue

        match = STREAM_VECTOR_RE.match(raw_line)
        if match is not None:
            current_vectors_nm_invA[match.group(1)] = np.asarray(
                [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                dtype=float,
            )
            continue

        if "Reflections measured after indexing" in raw_line:
            in_reflections = True
            continue
        if "End of reflections" in raw_line:
            in_reflections = False
            continue

        if in_reflections:
            parsed = _parse_stream_reflection_row(raw_line)
            if parsed is not None:
                pending_reflections.append(parsed)

    if not crystal_rows:
        raise ValueError("No indexed crystal blocks found in stream text.")

    if wavelength_angstrom is None:
        raise ValueError("Could not parse stream wavelength.")
    if res_px_per_m is None or res_px_per_m <= 0.0:
        raise ValueError("Could not parse stream detector resolution (res).")

    # Choose the first detector panel if panel metadata exist.
    panel_names = sorted(panel_ranges)
    selected_panel = panel_names[0] if panel_names else None

    detector_nx: int | None = None
    detector_ny: int | None = None
    orgx_px: float | None = None
    orgy_px: float | None = None

    if selected_panel is not None:
        ranges = panel_ranges.get(selected_panel, {})
        corners = panel_corners.get(selected_panel, {})
        required_ranges = ("min_fs", "max_fs", "min_ss", "max_ss")
        if all(key in ranges for key in required_ranges):
            min_fs = int(ranges["min_fs"])
            max_fs = int(ranges["max_fs"])
            min_ss = int(ranges["min_ss"])
            max_ss = int(ranges["max_ss"])
            detector_nx = max_fs - min_fs + 1
            detector_ny = max_ss - min_ss + 1
            if "x" in corners:
                orgx_px = -float(corners["x"]) + min_fs
            if "y" in corners:
                orgy_px = -float(corners["y"]) + min_ss

    reflections_df = pd.DataFrame.from_records(reflection_rows)
    if detector_nx is None or detector_ny is None:
        if reflections_df.empty:
            raise ValueError("Could not determine detector size from stream metadata.")
        detector_nx = int(np.ceil(float(reflections_df["fs_px"].max()))) + 1
        detector_ny = int(np.ceil(float(reflections_df["ss_px"].max()))) + 1
    if orgx_px is None:
        orgx_px = 0.5 * float(detector_nx)
    if orgy_px is None:
        orgy_px = 0.5 * float(detector_ny)

    pixel_mm = 1000.0 / float(res_px_per_m)
    if average_camera_lengths_m:
        distance_mm = 1000.0 * float(np.median(average_camera_lengths_m))
    elif clen_m is not None:
        distance_mm = 1000.0 * float(clen_m)
    else:
        raise ValueError("Could not parse stream camera length.")

    if {"a", "b", "c"} <= unit_cell_lengths.keys() and {"al", "be", "ga"} <= unit_cell_angles.keys():
        unit_cell = UnitCell(
            a=float(unit_cell_lengths["a"]),
            b=float(unit_cell_lengths["b"]),
            c=float(unit_cell_lengths["c"]),
            alpha=float(unit_cell_angles["al"]),
            beta=float(unit_cell_angles["be"]),
            gamma=float(unit_cell_angles["ga"]),
        )
    else:
        crystals_df_tmp = pd.DataFrame.from_records(crystal_rows)
        if crystals_df_tmp[["a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg"]].isna().all(axis=None):
            raise ValueError("Could not parse unit-cell parameters from stream.")
        unit_cell = UnitCell(
            a=float(crystals_df_tmp["a_angstrom"].dropna().median()),
            b=float(crystals_df_tmp["b_angstrom"].dropna().median()),
            c=float(crystals_df_tmp["c_angstrom"].dropna().median()),
            alpha=float(crystals_df_tmp["alpha_deg"].dropna().median()),
            beta=float(crystals_df_tmp["beta_deg"].dropna().median()),
            gamma=float(crystals_df_tmp["gamma_deg"].dropna().median()),
        )

    crystal_df = pd.DataFrame.from_records(crystal_rows)
    crystal_df["frame"] = crystal_df["frame"].astype(int)
    crystal_df["frame_number"] = crystal_df["frame_number"].astype(int)
    crystal_df["chunk_id"] = crystal_df["chunk_id"].astype(int)
    crystal_df["crystal_in_chunk"] = crystal_df["crystal_in_chunk"].astype(int)
    crystal_df["image_serial"] = crystal_df["image_serial"].astype(int)
    for col in STREAM_MATRIX_COLUMNS:
        crystal_df[col] = crystal_df[col].astype(float)

    if reflections_df.empty:
        reflections_df = pd.DataFrame(
            columns=[
                "frame",
                "frame_number",
                "chunk_id",
                "crystal_in_chunk",
                "event",
                "image_serial",
                "h",
                "k",
                "l",
                "I",
                "sigma",
                "peak",
                "background",
                "fs_px",
                "ss_px",
                "panel",
            ]
        )
    else:
        for col in ("frame", "frame_number", "chunk_id", "crystal_in_chunk", "image_serial", "h", "k", "l"):
            reflections_df[col] = reflections_df[col].astype(int)
        for col in ("I", "sigma", "peak", "background", "fs_px", "ss_px"):
            reflections_df[col] = reflections_df[col].astype(float)

    return CrystFELStreamData(
        wavelength_angstrom=float(wavelength_angstrom),
        distance_mm=float(distance_mm),
        pixel_x_mm=float(pixel_mm),
        pixel_y_mm=float(pixel_mm),
        detector_nx=int(detector_nx),
        detector_ny=int(detector_ny),
        orgx_px=float(orgx_px),
        orgy_px=float(orgy_px),
        unit_cell=unit_cell,
        crystal_table=crystal_df.sort_values("frame").reset_index(drop=True),
        reflections=reflections_df.sort_values(["frame", "h", "k", "l"]).reset_index(drop=True),
    )


def parse_crystfel_stream(path: str | Path) -> CrystFELStreamData:
    """Parse a CrystFEL ``.stream`` file from disk."""

    return parse_crystfel_stream_text(Path(path).read_text())


def crystfel_stream_to_analysis_inputs(
    stream_data: CrystFELStreamData,
) -> tuple[GXPARMData, IntegrateData, dict[int, FloatArray]]:
    """Convert parsed stream data to core analysis inputs.

    Returns
    -------
    gxparm:
        Synthetic ``GXPARMData`` carrying detector and unit-cell metadata.
    integrate:
        Observation table built from stream indexed reflections.
    reciprocal_by_frame:
        ``frame_index -> UB`` lookup from stream ``astar``, ``bstar``, ``cstar``.
    """

    if stream_data.crystal_table.empty:
        raise ValueError("stream crystal table is empty.")

    real_space_reference = _real_space_matrix_from_unit_cell(stream_data.unit_cell)
    reciprocal_reference = np.linalg.inv(real_space_reference)

    gxparm = GXPARMData(
        phi0_deg=0.0,
        dphi_deg=0.0,
        rotation_axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        wavelength_angstrom=float(stream_data.wavelength_angstrom),
        space_group=1,
        unit_cell=stream_data.unit_cell,
        real_space_reference=real_space_reference,
        reciprocal_reference=reciprocal_reference,
        detector_nx=int(stream_data.detector_nx),
        detector_ny=int(stream_data.detector_ny),
        pixel_x_mm=float(stream_data.pixel_x_mm),
        pixel_y_mm=float(stream_data.pixel_y_mm),
        orgx_px=float(stream_data.orgx_px),
        orgy_px=float(stream_data.orgy_px),
        distance_mm=float(stream_data.distance_mm),
    )

    reciprocal_by_frame: dict[int, FloatArray] = {}
    for _, row in stream_data.crystal_table.iterrows():
        frame = int(row["frame"])
        matrix = row[list(STREAM_MATRIX_COLUMNS)].to_numpy(dtype=float).reshape(3, 3)
        reciprocal_by_frame[frame] = matrix

    table = stream_data.reflections.copy()
    required = ["h", "k", "l", "I", "sigma", "frame"]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"stream reflections table is missing required columns: {missing}")

    valid = table[required].notna().all(axis=1)
    obs = table.loc[valid, required].copy()
    if obs.empty:
        raise ValueError("No valid stream reflection rows remain after filtering missing h/k/l/I/sigma/frame.")

    obs["h"] = obs["h"].astype(int)
    obs["k"] = obs["k"].astype(int)
    obs["l"] = obs["l"].astype(int)
    obs["I"] = obs["I"].astype(float)
    obs["sigma"] = obs["sigma"].astype(float)
    obs["z_cal"] = obs["frame"].astype(float)
    obs["frame_est"] = np.floor(obs["z_cal"]).astype(int)
    observations = obs[["h", "k", "l", "I", "sigma", "z_cal", "frame_est"]].reset_index(drop=True)
    estimated_n_frames = int(observations["frame_est"].max()) + 1

    integrate = IntegrateData(
        observations=observations,
        estimated_n_frames=estimated_n_frames,
    )
    return gxparm, integrate, reciprocal_by_frame


def _resolve_pets_paths(
    path: str | Path,
    rprofall_path: str | Path | None = None,
) -> tuple[Path, Path]:
    source = Path(path)
    if source.is_dir():
        pts_candidates = sorted(source.glob("*.pts2.backup")) + sorted(source.glob("*.pts2"))
        if not pts_candidates:
            raise ValueError(
                f"Could not find PETS project file (*.pts2.backup or *.pts2) in directory: {source}"
            )
        pts_path = pts_candidates[0]
        directory = source
    else:
        if not source.exists():
            raise ValueError(f"PETS path does not exist: {source}")
        pts_path = source
        directory = source.parent

    if rprofall_path is not None:
        rpf = Path(rprofall_path)
        if not rpf.exists():
            raise ValueError(f"PETS .rprofall path does not exist: {rpf}")
        return pts_path, rpf

    rprofall_candidates = sorted(directory.glob("*.rprofall"))
    if not rprofall_candidates:
        raise ValueError(f"Could not find PETS .rprofall file in directory: {directory}")
    if len(rprofall_candidates) == 1:
        return pts_path, rprofall_candidates[0]

    base_name = pts_path.name
    prefix_candidates: list[str] = []
    if ".pts2" in base_name:
        prefix_candidates.append(base_name.split(".pts2", maxsplit=1)[0])
    if ".ptsopt" in base_name:
        prefix_candidates.append(base_name.split(".ptsopt", maxsplit=1)[0])
    prefix_candidates.append(pts_path.stem)

    for prefix in prefix_candidates:
        matched = [cand for cand in rprofall_candidates if cand.name.startswith(prefix)]
        if len(matched) == 1:
            return pts_path, matched[0]

    candidates_text = ", ".join(cand.name for cand in rprofall_candidates[:8])
    raise ValueError(
        "Found multiple PETS .rprofall files and could not determine which one to use. "
        f"Pass rprofall_path explicitly. Candidates: {candidates_text}"
    )


def _parse_pets_badpixel_max(lines: list[str]) -> tuple[int | None, int | None]:
    in_badpixels = False
    max_x: int | None = None
    max_y: int | None = None
    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        if lower == "badpixels":
            in_badpixels = True
            continue
        if lower == "endbadpixels":
            in_badpixels = False
            continue
        if not in_badpixels:
            continue
        numbers: list[int] = []
        for token in line.split():
            if re.fullmatch(r"[-+]?\d+", token) is None:
                continue
            numbers.append(int(token))
        for idx in range(0, len(numbers) - 1, 2):
            x = numbers[idx]
            y = numbers[idx + 1]
            max_x = x if max_x is None else max(max_x, x)
            max_y = y if max_y is None else max(max_y, y)
    return max_x, max_y


def _parse_pets_imagelist(lines: list[str]) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, str | None]] = []
    in_block = False
    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("imagelistheader"):
            parts = line.split()
            if len(parts) > 1:
                header = [token.strip().lower() for token in parts[1:]]
            continue
        if lower == "imagelist":
            in_block = True
            continue
        if lower == "endimagelist":
            in_block = False
            continue
        if not in_block or not line:
            continue

        values = shlex.split(line)
        if not values:
            continue
        if header is None:
            header = list(PETS_IMAGELIST_DEFAULT_HEADER)
        row: dict[str, str | None] = {}
        for col_idx, col_name in enumerate(header):
            row[col_name] = values[col_idx] if col_idx < len(values) else None
        rows.append(row)

    if not rows:
        raise ValueError("Could not parse PETS imagelist block.")

    table = pd.DataFrame.from_records(rows)
    for column in table.columns:
        if column == "imgname":
            continue
        table[column] = pd.to_numeric(table[column], errors="coerce")

    table["frame"] = np.arange(table.shape[0], dtype=int)
    table["frame_number"] = table["frame"] + 1
    table["image_number"] = (
        table["imgname"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    return table


def _parse_pets_ubmatrix(lines: list[str]) -> FloatArray:
    for line_index, raw_line in enumerate(lines):
        if raw_line.strip().lower() != "ubmatrix":
            continue
        matrix_rows: list[list[float]] = []
        scan_index = line_index + 1
        while scan_index < len(lines) and len(matrix_rows) < 3:
            stripped = lines[scan_index].strip()
            scan_index += 1
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) < 3:
                break
            try:
                row = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
            except ValueError:
                break
            matrix_rows.append(row)
        if len(matrix_rows) == 3:
            return np.asarray(matrix_rows, dtype=float)
    raise ValueError("Could not parse PETS ubmatrix (3x3) from project file.")


def _parse_pets_cell(lines: list[str]) -> UnitCell:
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped.lower().startswith("cell "):
            continue
        tokens = stripped.split()
        if len(tokens) < 7:
            continue
        try:
            return UnitCell(
                a=float(tokens[1]),
                b=float(tokens[2]),
                c=float(tokens[3]),
                alpha=float(tokens[4]),
                beta=float(tokens[5]),
                gamma=float(tokens[6]),
            )
        except ValueError:
            continue
    raise ValueError("Could not parse PETS unit-cell parameters from project file.")


def _infer_pets_detector_size(
    orgx_px: float,
    orgy_px: float,
    badpixel_max_x: int | None,
    badpixel_max_y: int | None,
) -> tuple[int, int]:
    nx_from_center = int(np.ceil(max(orgx_px, 1.0) * 2.0 + 20.0))
    ny_from_center = int(np.ceil(max(orgy_px, 1.0) * 2.0 + 20.0))
    nx_from_badpixels = 0 if badpixel_max_x is None else int(badpixel_max_x) + 1
    ny_from_badpixels = 0 if badpixel_max_y is None else int(badpixel_max_y) + 1
    nx = max(512, nx_from_center, nx_from_badpixels)
    ny = max(512, ny_from_center, ny_from_badpixels)
    return nx, ny


def parse_pets_project(
    path: str | Path,
    rprofall_path: str | Path | None = None,
) -> PETSProjectData:
    """Parse a PETS project directory or ``.pts2(.backup)`` file.

    The parser keeps ``.rprofall`` observations and only the geometry/orientation
    metadata required by the analysis pipeline.
    """

    pts_path, resolved_rprofall_path = _resolve_pets_paths(path, rprofall_path=rprofall_path)
    text = pts_path.read_text()
    lines = text.splitlines()

    wavelength_angstrom: float | None = None
    aperpixel_invA_per_px: float | None = None
    center_x: float | None = None
    center_y: float | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        key = tokens[0].lower()
        if key == "lambda" and len(tokens) >= 2:
            wavelength_angstrom = float(tokens[1])
        elif key == "aperpixel" and len(tokens) >= 2:
            aperpixel_invA_per_px = float(tokens[1])
        elif key == "center" and len(tokens) >= 3:
            if tokens[1].upper() != "AUTO":
                center_x = float(tokens[1])
                center_y = float(tokens[2])

    if wavelength_angstrom is None:
        raise ValueError("Could not parse PETS wavelength ('lambda').")
    if aperpixel_invA_per_px is None:
        raise ValueError("Could not parse PETS reciprocal calibration ('aperpixel').")
    if aperpixel_invA_per_px <= 0.0:
        raise ValueError("PETS 'aperpixel' must be positive.")

    unit_cell = _parse_pets_cell(lines)
    reciprocal_reference = _parse_pets_ubmatrix(lines)
    imagelist = _parse_pets_imagelist(lines)
    rprofall = parse_rprofall(resolved_rprofall_path)

    orgx_px = float(imagelist["xcenter"].dropna().median()) if "xcenter" in imagelist.columns else float("nan")
    orgy_px = float(imagelist["ycenter"].dropna().median()) if "ycenter" in imagelist.columns else float("nan")
    if not np.isfinite(orgx_px) and center_x is not None:
        orgx_px = float(center_x)
    if not np.isfinite(orgy_px) and center_y is not None:
        orgy_px = float(center_y)
    if not np.isfinite(orgx_px):
        orgx_px = 256.0
    if not np.isfinite(orgy_px):
        orgy_px = 256.0

    max_bad_x, max_bad_y = _parse_pets_badpixel_max(lines)
    detector_nx, detector_ny = _infer_pets_detector_size(
        orgx_px=orgx_px,
        orgy_px=orgy_px,
        badpixel_max_x=max_bad_x,
        badpixel_max_y=max_bad_y,
    )

    return PETSProjectData(
        pts_path=pts_path,
        rprofall_path=resolved_rprofall_path,
        wavelength_angstrom=float(wavelength_angstrom),
        aperpixel_invA_per_px=float(aperpixel_invA_per_px),
        unit_cell=unit_cell,
        reciprocal_reference=reciprocal_reference,
        detector_nx=int(detector_nx),
        detector_ny=int(detector_ny),
        orgx_px=float(orgx_px),
        orgy_px=float(orgy_px),
        imagelist=imagelist,
        rprofall=rprofall,
    )


def pets_project_to_analysis_inputs(
    pets_data: PETSProjectData,
) -> tuple[GXPARMData, IntegrateData, dict[int, FloatArray]]:
    """Convert parsed PETS project data to core analysis inputs."""

    integrate = rprofall_to_integrate_data(pets_data.rprofall)
    reciprocal_reference = np.asarray(pets_data.reciprocal_reference, dtype=float)
    if reciprocal_reference.shape != (3, 3):
        raise ValueError("PETS reciprocal reference must be a 3x3 matrix.")

    real_space_reference = np.linalg.inv(reciprocal_reference)
    distance_over_pixel = 1.0 / (
        float(pets_data.aperpixel_invA_per_px) * float(pets_data.wavelength_angstrom)
    )
    pixel_size_mm = 1.0
    distance_mm = distance_over_pixel * pixel_size_mm

    gxparm = GXPARMData(
        phi0_deg=0.0,
        dphi_deg=0.0,
        rotation_axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        wavelength_angstrom=float(pets_data.wavelength_angstrom),
        space_group=1,
        unit_cell=pets_data.unit_cell,
        real_space_reference=real_space_reference,
        reciprocal_reference=reciprocal_reference,
        detector_nx=int(pets_data.detector_nx),
        detector_ny=int(pets_data.detector_ny),
        pixel_x_mm=float(pixel_size_mm),
        pixel_y_mm=float(pixel_size_mm),
        orgx_px=float(pets_data.orgx_px),
        orgy_px=float(pets_data.orgy_px),
        distance_mm=float(distance_mm),
    )

    if pets_data.imagelist.empty:
        raise ValueError("PETS imagelist is empty.")

    angle_table = pets_data.imagelist.reindex(columns=["alpha", "beta", "domega"]).copy()
    for column in ("alpha", "beta", "domega"):
        angle_table[column] = pd.to_numeric(angle_table[column], errors="coerce")
    angle_table = angle_table.ffill().bfill().fillna(0.0)

    first_angles = angle_table.iloc[0]
    first_rotation = _pets_rotation_from_angles_deg(
        alpha_deg=float(first_angles["alpha"]),
        beta_deg=float(first_angles["beta"]),
        domega_deg=float(first_angles["domega"]),
    )
    inverse_first_rotation = np.linalg.inv(first_rotation)

    reciprocal_by_frame: dict[int, FloatArray] = {}
    n_orient_frames = min(int(integrate.estimated_n_frames), angle_table.shape[0])
    for frame_idx in range(n_orient_frames):
        row = angle_table.iloc[frame_idx]
        current_rotation = _pets_rotation_from_angles_deg(
            alpha_deg=float(row["alpha"]),
            beta_deg=float(row["beta"]),
            domega_deg=float(row["domega"]),
        )
        relative_rotation = current_rotation @ inverse_first_rotation
        reciprocal_by_frame[frame_idx] = relative_rotation @ reciprocal_reference

    if not reciprocal_by_frame:
        reciprocal_by_frame[0] = reciprocal_reference

    last_matrix = reciprocal_by_frame[max(reciprocal_by_frame)]
    for frame_idx in range(n_orient_frames, int(integrate.estimated_n_frames)):
        reciprocal_by_frame[frame_idx] = np.asarray(last_matrix, dtype=float)

    return gxparm, integrate, reciprocal_by_frame


def parse_integrate_text(text: str) -> IntegrateData:
    """Parse INTEGRATE.HKL text.

    The original HTML script uses the 8th numeric column (index 7) as the z-like
    frame coordinate. The same convention is preserved here.
    """

    observations: list[dict[str, float | int]] = []
    in_data = False
    max_z_cal = 0.0

    for raw_line in text.splitlines():
        if raw_line.startswith("!END_OF_HEADER"):
            in_data = True
            continue
        if not in_data or raw_line.startswith("!"):
            continue

        parts = raw_line.strip().split()
        if len(parts) < 21:
            continue

        h, k, l = (int(parts[0]), int(parts[1]), int(parts[2]))
        i_obs = float(parts[3])
        sigma = float(parts[4])
        z_cal = float(parts[7])
        max_z_cal = max(max_z_cal, z_cal)
        observations.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "I": i_obs,
                "sigma": sigma,
                "z_cal": z_cal,
                "frame_est": int(np.floor(z_cal)),
            }
        )

    observations_df = pd.DataFrame.from_records(observations)
    if observations_df.empty:
        raise ValueError("No reflection rows found in INTEGRATE.HKL.")

    return IntegrateData(
        observations=observations_df,
        estimated_n_frames=int(np.floor(max_z_cal)) + 1,
    )


def parse_integrate_hkl(path: str | Path) -> IntegrateData:
    """Parse an INTEGRATE.HKL file from disk."""

    return parse_integrate_text(Path(path).read_text())


def parse_rprofall_text(text: str) -> RProfallData:
    """Parse PETS2 ``.rprofall`` text as fixed-width reflection blocks.

    Output columns are:
    ``row_id, block_id, h, k, l, resolution, excitation, rsg, iobs, sigma, icalc, frame, azimuth``.
    """

    rows: list[dict[str, float | int | None]] = []
    block_id: int | None = None
    row_id = 0

    for raw_line in text.splitlines():
        line = raw_line
        if not line.strip():
            continue

        header_match = RPROFALL_HEADER_RE.match(line)
        if header_match is not None:
            block_id = int(header_match.group(1))
            continue

        if block_id is None:
            raise ValueError("Encountered .rprofall data row before any '# <integer>' block header.")

        fixed_line = line[:RPROFALL_ROW_WIDTH].ljust(RPROFALL_ROW_WIDTH)
        row: dict[str, float | int | None] = {"block_id": block_id}
        for field_name, (start, stop), field_type in RPROFALL_SCHEMA:
            raw_field = fixed_line[start:stop]
            if field_type == "int":
                row[field_name] = _parse_fixed_int(raw_field)
            else:
                row[field_name] = _parse_fixed_float(raw_field)

        row_id += 1
        row["row_id"] = row_id
        rows.append(row)

    if not rows:
        raise ValueError("No reflection data rows found in .rprofall text.")

    table = pd.DataFrame.from_records(rows, columns=RPROFALL_OUTPUT_COLUMNS)
    for col in ("row_id", "block_id", "h", "k", "l", "frame"):
        table[col] = table[col].astype("Int64")
    for col in ("resolution", "excitation", "rsg", "iobs", "sigma", "icalc", "azimuth"):
        table[col] = table[col].astype(float)

    return RProfallData(
        rows=table,
        n_blocks=int(table["block_id"].dropna().nunique()),
    )


def parse_rprofall(path: str | Path) -> RProfallData:
    """Parse a PETS2 ``.rprofall`` file from disk."""

    return parse_rprofall_text(Path(path).read_text())


def rprofall_to_integrate_data(data: RProfallData | pd.DataFrame) -> IntegrateData:
    """Convert parsed ``.rprofall`` rows to ``IntegrateData`` for pipeline reuse."""

    table = data.rows.copy() if isinstance(data, RProfallData) else data.copy()
    required = ["h", "k", "l", "iobs", "sigma", "frame"]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"rprofall table is missing required columns: {missing}")

    valid = table[required].notna().all(axis=1)
    obs = table.loc[valid, required].copy()
    if obs.empty:
        raise ValueError("No valid reflection rows remain after filtering missing h/k/l/iobs/sigma/frame.")

    obs["h"] = obs["h"].astype(int)
    obs["k"] = obs["k"].astype(int)
    obs["l"] = obs["l"].astype(int)
    obs["I"] = obs["iobs"].astype(float)
    obs["sigma"] = obs["sigma"].astype(float)
    obs["z_cal"] = obs["frame"].astype(float)
    obs["frame_est"] = np.floor(obs["z_cal"]).astype(int)
    observations = obs[["h", "k", "l", "I", "sigma", "z_cal", "frame_est"]].reset_index(drop=True)

    min_frame = int(observations["frame_est"].min())
    max_frame = int(observations["frame_est"].max())
    estimated_n_frames = max_frame + 1 if min_frame <= 0 else max_frame
    return IntegrateData(observations=observations, estimated_n_frames=estimated_n_frames)


def parse_xds_inp_text(text: str | None) -> XDSInputData:
    """Parse selected fields from XDS.INP text."""

    rectangles: list[tuple[float, float, float, float]] = []
    data_range: tuple[int, int] | None = None
    if not text:
        return XDSInputData(untrusted_rectangles=rectangles, data_range=data_range)

    rect_pattern = re.compile(
        r"UNTRUSTED_RECTANGLE\s*=\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    )
    range_pattern = re.compile(r"DATA_RANGE\s*=\s*(\d+)\s+(\d+)")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        rect_match = rect_pattern.search(line)
        if rect_match:
            rectangles.append(tuple(float(rect_match.group(i)) for i in range(1, 5)))
        range_match = range_pattern.search(line)
        if range_match:
            data_range = (int(range_match.group(1)), int(range_match.group(2)))

    return XDSInputData(untrusted_rectangles=rectangles, data_range=data_range)


def parse_xds_inp(path: str | Path) -> XDSInputData:
    """Parse XDS.INP from disk."""

    return parse_xds_inp_text(Path(path).read_text())


def parse_composition(text: str) -> CompositionResult:
    """Parse a composition string such as ``"4 C, 8 H, 2 O"``.

    Returns
    -------
    CompositionResult
        Parsed entries and ``sum_fj2 = sum_j n_j f_j(0)^2``.
    """

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("Composition string is empty.")

    entries: list[CompositionEntry] = []
    sum_fj2 = 0.0
    pattern = re.compile(r"^(\d+(?:\.\d+)?)\s*([A-Z][a-z]?)$")

    for part in parts:
        match = pattern.match(part)
        if match is None:
            raise ValueError(f"Could not parse composition term: {part!r}")
        count = float(match.group(1))
        element = match.group(2)
        if element not in FE0:
            raise ValueError(f"Unknown element in composition: {element!r}")
        fe0 = FE0[element]
        entries.append(
            CompositionEntry(
                count=count,
                element=element,
                forward_scattering_factor=fe0,
            )
        )
        sum_fj2 += count * fe0 * fe0

    return CompositionResult(entries=tuple(entries), sum_fj2=sum_fj2)


def load_optional_xds_inp(path: str | Path | None) -> XDSInputData | None:
    """Load XDS.INP if a path is supplied, otherwise return ``None``."""

    if path is None:
        return None
    return parse_xds_inp(path)
