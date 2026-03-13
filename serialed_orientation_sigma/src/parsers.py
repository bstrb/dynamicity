from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import (
    UnitCell,
    matrix_from_orientation_row,
    orientation_matrix_to_row,
    reconstruct_rotation_series_orientations,
)


ORIENTATION_COLUMNS = [
    "UB11",
    "UB12",
    "UB13",
    "UB21",
    "UB22",
    "UB23",
    "UB31",
    "UB32",
    "UB33",
]


@dataclass(slots=True)
class GXPARMMetadata:
    """Minimal metadata extracted from a GXPARM/XPARM file."""

    start_frame: int
    phi0_deg: float
    dphi_deg: float
    rotation_axis: np.ndarray
    wavelength_angstrom: float
    beam_direction: np.ndarray
    cell: UnitCell
    direct_matrix_lab: np.ndarray
    reference_ub: np.ndarray


def _read_delimited_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV, TSV, whitespace-delimited, or parquet table."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix in {".csv"}:
        return pd.read_csv(file_path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(file_path, sep="\t")
    return pd.read_csv(file_path, sep=None, engine="python")


def parse_cell_json(path: str | Path) -> UnitCell:
    """Read a unit-cell JSON file into a :class:`UnitCell` instance."""
    payload = json.loads(Path(path).read_text())
    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cell JSON is missing keys: {missing}")
    return UnitCell(
        a=float(payload["a"]),
        b=float(payload["b"]),
        c=float(payload["c"]),
        alpha=float(payload["alpha"]),
        beta=float(payload["beta"]),
        gamma=float(payload["gamma"]),
        voltage_kv=float(payload.get("voltage_kv", 200.0)),
        composition={str(k): float(v) for k, v in payload.get("composition", {}).items()},
        beam_direction=tuple(float(v) for v in payload.get("beam_direction", [0.0, 0.0, 1.0])),
    )


def load_orientation_table(path: str | Path) -> pd.DataFrame:
    """Load a SerialED orientation table.

    Expected columns are `frame` and nine matrix values named `UB11` ... `UB33`.
    A `phi` column is optional.
    """
    table = _read_delimited_table(path)
    if "frame" not in table.columns:
        raise ValueError("Orientation table must contain a 'frame' column.")
    missing = [col for col in ORIENTATION_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"Orientation table missing required UB columns: {missing}")
    output = table.copy()
    output["frame"] = output["frame"].astype(int)
    for col in ORIENTATION_COLUMNS:
        output[col] = output[col].astype(float)
    if "phi" in output.columns:
        output["phi"] = output["phi"].astype(float)
    else:
        output["phi"] = np.nan
    return output.sort_values("frame").reset_index(drop=True)


def load_reflection_table(path: str | Path) -> pd.DataFrame:
    """Load a flat reflection table for SerialED snapshot processing."""
    table = _read_delimited_table(path)
    rename_map = {}
    for source, target in {
        "intensity": "I",
        "Intensity": "I",
        "sigI": "sigma",
        "Sigma": "sigma",
    }.items():
        if source in table.columns and target not in table.columns:
            rename_map[source] = target
    if rename_map:
        table = table.rename(columns=rename_map)
    required = ["frame", "h", "k", "l", "I", "sigma"]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"Reflection table missing required columns: {missing}")
    output = table.copy()
    integer_cols = ["frame", "h", "k", "l"]
    for col in integer_cols:
        output[col] = output[col].astype(int)
    float_cols = [col for col in output.columns if col not in integer_cols]
    for col in float_cols:
        output[col] = pd.to_numeric(output[col], errors="coerce")
    output = output.dropna(subset=["I", "sigma"])
    return output.sort_values(["frame", "h", "k", "l"]).reset_index(drop=True)


def parse_xds_inp(path: str | Path) -> dict[str, Any]:
    """Parse a simple XDS.INP key-value file.

    Values are returned either as floats, strings, or lists of floats when a
    whitespace-separated numeric array is detected.
    """
    parsed: dict[str, Any] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        parts = value.split()
        try:
            numeric = [float(part) for part in parts]
        except ValueError:
            parsed[key] = value
        else:
            parsed[key] = numeric[0] if len(numeric) == 1 else numeric
    return parsed


def parse_gxparm(path: str | Path) -> GXPARMMetadata:
    """Parse the numeric payload of a GXPARM.XDS or XPARM.XDS file."""
    numeric_lines: list[list[float]] = []
    for raw in Path(path).read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            numeric_lines.append([float(token) for token in stripped.split()])
        except ValueError:
            continue

    if len(numeric_lines) < 6:
        raise ValueError("GXPARM/XPARM file did not contain enough numeric lines.")

    frame_line = numeric_lines[0]
    beam_line = numeric_lines[1]

    cell_line_idx: int | None = None
    search_stop = min(len(numeric_lines), 12)
    for idx in range(2, search_stop):
        values = numeric_lines[idx]
        if len(values) < 7:
            continue
        a_len, b_len, c_len = values[1], values[2], values[3]
        alpha, beta, gamma = values[4], values[5], values[6]
        if min(a_len, b_len, c_len) <= 0:
            continue
        if not (30.0 <= alpha <= 180.0 and 30.0 <= beta <= 180.0 and 30.0 <= gamma <= 180.0):
            continue
        if idx + 3 >= len(numeric_lines):
            continue
        if any(len(numeric_lines[idx + offset]) < 3 for offset in (1, 2, 3)):
            continue
        cell_line_idx = idx
        break

    if cell_line_idx is None:
        raise ValueError("Failed to locate unit-cell constants in GXPARM/XPARM numeric payload.")

    cell_line = numeric_lines[cell_line_idx]
    a_axis = numeric_lines[cell_line_idx + 1][:3]
    b_axis = numeric_lines[cell_line_idx + 2][:3]
    c_axis = numeric_lines[cell_line_idx + 3][:3]

    start_frame = int(round(frame_line[0]))
    phi0_deg = float(frame_line[1])
    dphi_deg = float(frame_line[2])
    rotation_axis = np.asarray(frame_line[3:6], dtype=np.float64)
    wavelength_angstrom = float(beam_line[0])
    beam_direction = np.asarray(beam_line[1:4], dtype=np.float64)

    cell = UnitCell(
        a=float(cell_line[1]),
        b=float(cell_line[2]),
        c=float(cell_line[3]),
        alpha=float(cell_line[4]),
        beta=float(cell_line[5]),
        gamma=float(cell_line[6]),
        voltage_kv=200.0,
        composition={},
        beam_direction=tuple(float(v) for v in beam_direction),
    )

    direct_matrix_lab = np.column_stack(
        [
            np.asarray(a_axis, dtype=np.float64),
            np.asarray(b_axis, dtype=np.float64),
            np.asarray(c_axis, dtype=np.float64),
        ]
    )
    reference_ub = np.linalg.inv(direct_matrix_lab).T

    return GXPARMMetadata(
        start_frame=start_frame,
        phi0_deg=phi0_deg,
        dphi_deg=dphi_deg,
        rotation_axis=rotation_axis,
        wavelength_angstrom=wavelength_angstrom,
        beam_direction=beam_direction,
        cell=cell,
        direct_matrix_lab=direct_matrix_lab,
        reference_ub=reference_ub,
    )


def estimate_frame_from_z(z_values: np.ndarray | pd.Series) -> np.ndarray:
    """Estimate frame number from XDS `ZD` coordinates."""
    z_array = np.asarray(z_values, dtype=np.float64)
    return np.floor(z_array + 0.5).astype(int)


def load_integrate_hkl(path: str | Path) -> pd.DataFrame:
    """Read an XDS INTEGRATE.HKL file into a flat DataFrame.

    The parser uses the `!ITEM_*` header mapping when present. If the header is
    not available, a fallback positional layout is assumed:

    `h k l I sigma x y z`
    """
    item_map: dict[str, int] = {}
    rows: list[list[float]] = []
    item_pattern = re.compile(r"^!ITEM_([^=]+)=(\d+)$")

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("!"):
            match = item_pattern.match(line.replace(" ", ""))
            if match:
                item_map[match.group(1)] = int(match.group(2)) - 1
            continue
        try:
            rows.append([float(token) for token in line.split()])
        except ValueError:
            continue

    if not rows:
        raise ValueError("INTEGRATE.HKL parser found no numeric reflection rows.")

    array = np.asarray(rows, dtype=np.float64)

    def column_index(*names: str, default: int | None = None) -> int:
        for name in names:
            if name in item_map:
                return item_map[name]
        if default is None:
            raise KeyError(f"None of the XDS item names were found: {names}")
        return default

    h = array[:, column_index("H", default=0)].astype(int)
    k = array[:, column_index("K", default=1)].astype(int)
    l = array[:, column_index("L", default=2)].astype(int)
    intensity = array[:, column_index("IOBS", "I", default=3)]
    sigma = array[:, column_index("SIGMA(IOBS)", "SIGMA", default=4)]
    x = array[:, column_index("XD", "XDET", default=5)] if array.shape[1] > 5 else np.nan
    y = array[:, column_index("YD", "YDET", default=6)] if array.shape[1] > 6 else np.nan
    z = array[:, column_index("ZD", "Z", default=7)] if array.shape[1] > 7 else np.nan

    if np.isscalar(x):
        x = np.full_like(intensity, float(x), dtype=np.float64)
    if np.isscalar(y):
        y = np.full_like(intensity, float(y), dtype=np.float64)
    if np.isscalar(z):
        z = np.full_like(intensity, float(z), dtype=np.float64)

    frame = estimate_frame_from_z(z)
    return pd.DataFrame(
        {
            "frame": frame,
            "h": h,
            "k": k,
            "l": l,
            "I": intensity,
            "sigma": sigma,
            "x": x,
            "y": y,
            "z": z,
        }
    ).sort_values(["frame", "h", "k", "l"]).reset_index(drop=True)


def build_orientation_table(frame_ids: np.ndarray, phis: np.ndarray, matrices: np.ndarray) -> pd.DataFrame:
    """Convert orientation matrices into the standard flat orientation table."""
    rows: list[dict[str, float]] = []
    for frame, phi, matrix in zip(frame_ids, phis, matrices, strict=True):
        flat = orientation_matrix_to_row(matrix)
        row = {"frame": int(frame), "phi": float(phi)}
        for key, value in zip(ORIENTATION_COLUMNS, flat, strict=True):
            row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def orientation_matrix_lookup(orientations: pd.DataFrame) -> dict[int, np.ndarray]:
    """Build a dictionary from frame number to 3x3 orientation matrix."""
    lookup: dict[int, np.ndarray] = {}
    for row in orientations.itertuples(index=False):
        lookup[int(row.frame)] = matrix_from_orientation_row([getattr(row, col) for col in ORIENTATION_COLUMNS])
    return lookup


def load_xds_rotation_series(
    gxparm_path: str | Path,
    integrate_path: str | Path,
    xds_inp_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, UnitCell, GXPARMMetadata, dict[str, Any]]:
    """Load an XDS rotation series and reconstruct per-frame orientations."""
    metadata = parse_gxparm(gxparm_path)
    reflections = load_integrate_hkl(integrate_path)
    xds_inp = parse_xds_inp(xds_inp_path) if xds_inp_path is not None else {}

    if "X-RAY_WAVELENGTH" in xds_inp:
        metadata.wavelength_angstrom = float(xds_inp["X-RAY_WAVELENGTH"])
    if "INCIDENT_BEAM_DIRECTION" in xds_inp:
        direction = xds_inp["INCIDENT_BEAM_DIRECTION"]
        if isinstance(direction, list):
            metadata.beam_direction = np.asarray(direction, dtype=np.float64)
            metadata.cell.beam_direction = tuple(float(v) for v in metadata.beam_direction)
    if "ROTATION_AXIS" in xds_inp:
        axis = xds_inp["ROTATION_AXIS"]
        if isinstance(axis, list):
            metadata.rotation_axis = np.asarray(axis, dtype=np.float64)
    if "OSCILLATION_RANGE" in xds_inp:
        metadata.dphi_deg = float(xds_inp["OSCILLATION_RANGE"])

    frame_ids = np.asarray(sorted(reflections["frame"].unique()), dtype=int)
    phis, matrices = reconstruct_rotation_series_orientations(
        reference_ub=metadata.reference_ub,
        rotation_axis=metadata.rotation_axis,
        frame_ids=frame_ids,
        phi0_deg=metadata.phi0_deg,
        dphi_deg=metadata.dphi_deg,
        start_frame=metadata.start_frame,
    )
    orientations = build_orientation_table(frame_ids, phis, matrices)
    return orientations, reflections, metadata.cell, metadata, xds_inp
