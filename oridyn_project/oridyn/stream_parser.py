"""Small CrystFEL stream parser for orientation-aware geometry scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
    centering: str = "P"
    lattice_type: str = ""


@dataclass(frozen=True)
class StreamData:
    """Parsed CrystFEL stream data used by the geometry-only pipeline."""

    path: str | None
    wavelength_angstrom: float
    unit_cell: UnitCell
    crystal_table: pd.DataFrame
    reflections: pd.DataFrame
    detector: dict[str, float | int | None]


STREAM_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
STREAM_WAVELENGTH_RE = re.compile(rf"^\s*wavelength\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_CLEN_RE = re.compile(rf"^\s*clen\s*=\s*({STREAM_FLOAT})\s*m")
STREAM_RES_RE = re.compile(rf"^\s*res\s*=\s*({STREAM_FLOAT})")
STREAM_UNITCELL_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_UNITCELL_ANGLE_RE = re.compile(rf"^\s*(al|be|ga)\s*=\s*({STREAM_FLOAT})\s*deg")
STREAM_CENTERING_RE = re.compile(r"^\s*centering\s*=\s*([A-Za-z])")
STREAM_LATTICE_RE = re.compile(r"^\s*lattice_type\s*=\s*([A-Za-z_]+)")
STREAM_PANEL_RANGE_RE = re.compile(r"^\s*(p\d+)\/(min_fs|max_fs|min_ss|max_ss)\s*=\s*(-?\d+)")
STREAM_PANEL_CORNER_RE = re.compile(rf"^\s*(p\d+)\/corner_([xy])\s*=\s*({STREAM_FLOAT})")
STREAM_IMAGE_FILENAME_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
STREAM_SERIAL_RE = re.compile(r"^\s*Image serial number:\s*(\d+)")
STREAM_AVERAGE_CLEN_RE = re.compile(rf"^\s*average_camera_length\s*=\s*({STREAM_FLOAT})\s*m")
STREAM_DET_SHIFT_X_RE = re.compile(rf"^\s*header/float/.*/det_shift_x_mm\s*=\s*({STREAM_FLOAT})")
STREAM_DET_SHIFT_Y_RE = re.compile(rf"^\s*header/float/.*/det_shift_y_mm\s*=\s*({STREAM_FLOAT})")
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


def _parse_reflection_row(line: str) -> dict[str, float | int | str] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        i_obs = float(parts[3])
        sigma = float(parts[4])
    except ValueError:
        return None

    row: dict[str, float | int | str] = {"h": h, "k": k, "l": l, "sigma": sigma}
    # These fields are retained only for optional plotting/provenance. They are
    # not read by the scoring modules.
    if len(parts) >= 9:
        try:
            row.update(
                {
                    "fs_px": float(parts[7]),
                    "ss_px": float(parts[8]),
                    "panel": parts[9] if len(parts) > 9 else "",
                }
            )
        except ValueError:
            pass
    row["_observed_intensity_present"] = 1 if np.isfinite(i_obs) else 0
    return row


def parse_crystfel_stream_text(text: str, path: str | None = None) -> StreamData:
    """Parse the stream metadata needed for geometry-only scoring.

    CrystFEL stores ``astar``, ``bstar``, and ``cstar`` in ``nm^-1``. They are
    converted to ``A^-1`` and stored as columns of a 3x3 reciprocal matrix that
    maps Miller indices to lab-frame reciprocal vectors.
    """

    wavelength: float | None = None
    clen_m: float | None = None
    res_px_per_m: float | None = None
    average_clen_m: list[float] = []
    unit_lengths: dict[str, float] = {}
    unit_angles: dict[str, float] = {}
    centering = "P"
    lattice_type = ""
    panel_ranges: dict[str, dict[str, float]] = {}
    panel_corners: dict[str, dict[str, float]] = {}

    crystal_rows: list[dict[str, float | int | str]] = []
    reflection_rows: list[dict[str, float | int | str]] = []

    in_chunk = False
    in_crystal = False
    in_reflections = False
    chunk_id = -1
    crystal_counter = 0
    crystal_in_chunk = 0
    current_source_filename: str | None = None
    current_event: str | None = None
    current_serial: int | None = None
    current_clen_m: float | None = None
    current_det_shift_x_mm = 0.0
    current_det_shift_y_mm = 0.0
    current_vectors: dict[str, FloatArray] = {}
    current_cell: UnitCell | None = None
    pending_reflections: list[dict[str, float | int | str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if raw_line.startswith("----- Begin chunk -----"):
            in_chunk = True
            in_crystal = False
            in_reflections = False
            chunk_id += 1
            crystal_in_chunk = 0
            current_source_filename = None
            current_event = None
            current_serial = None
            current_clen_m = clen_m
            current_det_shift_x_mm = 0.0
            current_det_shift_y_mm = 0.0
            continue
        if raw_line.startswith("----- End chunk -----"):
            in_chunk = False
            in_crystal = False
            in_reflections = False
            continue

        if not in_chunk:
            if match := STREAM_WAVELENGTH_RE.match(raw_line):
                wavelength = float(match.group(1))
                continue
            if match := STREAM_CLEN_RE.match(raw_line):
                clen_m = float(match.group(1))
                continue
            if match := STREAM_RES_RE.match(raw_line):
                res_px_per_m = float(match.group(1))
                continue
            if match := STREAM_UNITCELL_LENGTH_RE.match(raw_line):
                unit_lengths[match.group(1)] = float(match.group(2))
                continue
            if match := STREAM_UNITCELL_ANGLE_RE.match(raw_line):
                unit_angles[match.group(1)] = float(match.group(2))
                continue
            if match := STREAM_CENTERING_RE.match(raw_line):
                centering = match.group(1).upper()
                continue
            if match := STREAM_LATTICE_RE.match(raw_line):
                lattice_type = match.group(1)
                continue
            if match := STREAM_PANEL_RANGE_RE.match(raw_line):
                panel_ranges.setdefault(match.group(1), {})[match.group(2)] = float(match.group(3))
                continue
            if match := STREAM_PANEL_CORNER_RE.match(raw_line):
                panel_corners.setdefault(match.group(1), {})[match.group(2)] = float(match.group(3))
                continue
            continue

        if match := STREAM_IMAGE_FILENAME_RE.match(raw_line):
            current_source_filename = match.group(1)
            continue
        if match := STREAM_EVENT_RE.match(raw_line):
            current_event = match.group(1)
            continue
        if match := STREAM_SERIAL_RE.match(raw_line):
            current_serial = int(match.group(1))
            continue
        if match := STREAM_AVERAGE_CLEN_RE.match(raw_line):
            current_clen_m = float(match.group(1))
            average_clen_m.append(current_clen_m)
            continue
        if match := STREAM_DET_SHIFT_X_RE.match(raw_line):
            current_det_shift_x_mm = float(match.group(1))
            continue
        if match := STREAM_DET_SHIFT_Y_RE.match(raw_line):
            current_det_shift_y_mm = float(match.group(1))
            continue

        if raw_line.startswith("--- Begin crystal"):
            in_crystal = True
            in_reflections = False
            current_vectors = {}
            current_cell = None
            pending_reflections = []
            crystal_in_chunk += 1
            continue

        if raw_line.startswith("--- End crystal"):
            missing = [axis for axis in ("a", "b", "c") if axis not in current_vectors]
            if missing:
                raise ValueError(f"Crystal in chunk {chunk_id} is missing reciprocal vectors: {missing}")
            reciprocal = np.column_stack([current_vectors["a"], current_vectors["b"], current_vectors["c"]]) / 10.0
            frame = crystal_counter
            frame_number = frame + 1
            row: dict[str, float | int | str] = {
                "frame": frame,
                "frame_number": frame_number,
                "chunk_id": chunk_id,
                "crystal_in_chunk": crystal_in_chunk,
                "source_filename": current_source_filename or "",
                "event": current_event or f"chunk{chunk_id}_crystal{crystal_in_chunk}",
                "image_serial": -1 if current_serial is None else int(current_serial),
                "distance_mm": float("nan") if current_clen_m is None else 1000.0 * float(current_clen_m),
                "det_shift_x_mm": current_det_shift_x_mm,
                "det_shift_y_mm": current_det_shift_y_mm,
            }
            if current_cell is not None:
                row.update(
                    {
                        "a_angstrom": current_cell.a,
                        "b_angstrom": current_cell.b,
                        "c_angstrom": current_cell.c,
                        "alpha_deg": current_cell.alpha,
                        "beta_deg": current_cell.beta,
                        "gamma_deg": current_cell.gamma,
                    }
                )
            for idx, value in enumerate(reciprocal.reshape(-1)):
                row[STREAM_MATRIX_COLUMNS[idx]] = float(value)
            crystal_rows.append(row)
            for reflection in pending_reflections:
                reflection.update(
                    {
                        "frame": frame,
                        "frame_number": frame_number,
                        "chunk_id": chunk_id,
                        "crystal_in_chunk": crystal_in_chunk,
                        "source_filename": row["source_filename"],
                        "event": row["event"],
                        "image_serial": row["image_serial"],
                    }
                )
                reflection_rows.append(reflection)
            crystal_counter += 1
            in_crystal = False
            in_reflections = False
            continue

        if not in_crystal:
            continue
        if match := STREAM_CELL_RE.match(raw_line):
            current_cell = UnitCell(
                a=10.0 * float(match.group(1)),
                b=10.0 * float(match.group(2)),
                c=10.0 * float(match.group(3)),
                alpha=float(match.group(4)),
                beta=float(match.group(5)),
                gamma=float(match.group(6)),
                centering=centering,
                lattice_type=lattice_type,
            )
            continue
        if match := STREAM_VECTOR_RE.match(raw_line):
            current_vectors[match.group(1)] = np.asarray(
                [float(match.group(2)), float(match.group(3)), float(match.group(4))], dtype=float
            )
            continue
        if "Reflections measured after indexing" in raw_line:
            in_reflections = True
            continue
        if "End of reflections" in raw_line:
            in_reflections = False
            continue
        if in_reflections:
            parsed = _parse_reflection_row(raw_line)
            if parsed is not None:
                pending_reflections.append(parsed)

    if not crystal_rows:
        raise ValueError("No indexed crystal blocks found in stream text.")
    if wavelength is None:
        raise ValueError("Could not parse stream wavelength.")

    crystal_df = pd.DataFrame.from_records(crystal_rows).sort_values("frame").reset_index(drop=True)
    for col in ("frame", "frame_number", "chunk_id", "crystal_in_chunk", "image_serial"):
        crystal_df[col] = crystal_df[col].astype(int)
    for col in STREAM_MATRIX_COLUMNS:
        crystal_df[col] = crystal_df[col].astype(float)

    reflections_df = pd.DataFrame.from_records(reflection_rows)
    if reflections_df.empty:
        reflections_df = pd.DataFrame(
            columns=[
                "frame",
                "frame_number",
                "chunk_id",
                "crystal_in_chunk",
                "source_filename",
                "event",
                "image_serial",
                "h",
                "k",
                "l",
                "sigma",
                "fs_px",
                "ss_px",
                "panel",
            ]
        )
    else:
        reflections_df = reflections_df.sort_values(["frame", "h", "k", "l"]).reset_index(drop=True)
        for col in ("frame", "frame_number", "chunk_id", "crystal_in_chunk", "image_serial", "h", "k", "l"):
            reflections_df[col] = reflections_df[col].astype(int)
        for col in ("sigma", "fs_px", "ss_px"):
            if col in reflections_df:
                reflections_df[col] = pd.to_numeric(reflections_df[col], errors="coerce")

    if {"a", "b", "c"} <= unit_lengths.keys() and {"al", "be", "ga"} <= unit_angles.keys():
        cell = UnitCell(
            a=unit_lengths["a"],
            b=unit_lengths["b"],
            c=unit_lengths["c"],
            alpha=unit_angles["al"],
            beta=unit_angles["be"],
            gamma=unit_angles["ga"],
            centering=centering,
            lattice_type=lattice_type,
        )
    else:
        needed = ["a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg"]
        if not set(needed) <= set(crystal_df.columns):
            raise ValueError("Could not parse unit-cell parameters from stream.")
        cell = UnitCell(
            a=float(crystal_df["a_angstrom"].dropna().median()),
            b=float(crystal_df["b_angstrom"].dropna().median()),
            c=float(crystal_df["c_angstrom"].dropna().median()),
            alpha=float(crystal_df["alpha_deg"].dropna().median()),
            beta=float(crystal_df["beta_deg"].dropna().median()),
            gamma=float(crystal_df["gamma_deg"].dropna().median()),
            centering=centering,
            lattice_type=lattice_type,
        )

    detector = _build_detector_metadata(panel_ranges, panel_corners, res_px_per_m, clen_m, average_clen_m)
    return StreamData(
        path=path,
        wavelength_angstrom=float(wavelength),
        unit_cell=cell,
        crystal_table=crystal_df,
        reflections=reflections_df,
        detector=detector,
    )


def _build_detector_metadata(
    panel_ranges: dict[str, dict[str, float]],
    panel_corners: dict[str, dict[str, float]],
    res_px_per_m: float | None,
    clen_m: float | None,
    average_clen_m: list[float],
) -> dict[str, float | int | None]:
    panel = sorted(panel_ranges)[0] if panel_ranges else None
    nx: int | None = None
    ny: int | None = None
    orgx: float | None = None
    orgy: float | None = None
    if panel is not None:
        ranges = panel_ranges.get(panel, {})
        corners = panel_corners.get(panel, {})
        if all(key in ranges for key in ("min_fs", "max_fs", "min_ss", "max_ss")):
            min_fs = int(ranges["min_fs"])
            max_fs = int(ranges["max_fs"])
            min_ss = int(ranges["min_ss"])
            max_ss = int(ranges["max_ss"])
            nx = max_fs - min_fs + 1
            ny = max_ss - min_ss + 1
            orgx = -float(corners["x"]) + min_fs if "x" in corners else None
            orgy = -float(corners["y"]) + min_ss if "y" in corners else None
    pixel_mm = None if res_px_per_m in (None, 0.0) else 1000.0 / float(res_px_per_m)
    distance_mm = None
    if average_clen_m:
        distance_mm = 1000.0 * float(np.median(average_clen_m))
    elif clen_m is not None:
        distance_mm = 1000.0 * float(clen_m)
    return {
        "detector_nx": nx,
        "detector_ny": ny,
        "orgx_px": orgx,
        "orgy_px": orgy,
        "pixel_mm": pixel_mm,
        "distance_mm": distance_mm,
    }


def parse_crystfel_stream(path: str | Path) -> StreamData:
    """Parse a CrystFEL stream from disk."""

    input_path = Path(path)
    return parse_crystfel_stream_text(input_path.read_text(errors="replace"), path=str(input_path))


def reciprocal_matrix_from_row(row: pd.Series) -> FloatArray:
    """Return the 3x3 reciprocal matrix from one crystal table row."""

    return row[list(STREAM_MATRIX_COLUMNS)].to_numpy(dtype=float).reshape(3, 3)
