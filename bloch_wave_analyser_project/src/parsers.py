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


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


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
