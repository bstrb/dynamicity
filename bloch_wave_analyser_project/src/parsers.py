"""Parsers for XDS and composition inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

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
