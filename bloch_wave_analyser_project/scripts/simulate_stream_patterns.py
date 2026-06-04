#!/usr/bin/env python3
"""Simulate detector spot patterns from CrystFEL stream orientations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
BEGIN_CHUNK = "----- Begin chunk"
END_CHUNK = "----- End chunk"
BEGIN_CRYSTAL = "--- Begin crystal"
END_CRYSTAL = "--- End crystal"
EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
SERIAL_RE = re.compile(r"^\s*Image serial number:\s*(\d+)")
WAVELENGTH_RE = re.compile(rf"^\s*wavelength\s*=\s*({FLOAT})\s*A")
CLEN_RE = re.compile(rf"^\s*clen\s*=\s*({FLOAT})\s*m")
AVERAGE_CLEN_RE = re.compile(rf"^\s*average_camera_length\s*=\s*({FLOAT})\s*m")
RES_RE = re.compile(rf"^\s*res\s*=\s*({FLOAT})")
UNIT_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({FLOAT})\s*A")
UNIT_ANGLE_RE = re.compile(rf"^\s*(al|be|ga)\s*=\s*({FLOAT})\s*deg")
PANEL_RANGE_RE = re.compile(r"^\s*(p\d+)\/(min_fs|max_fs|min_ss|max_ss)\s*=\s*(-?\d+)")
PANEL_CORNER_RE = re.compile(rf"^\s*(p\d+)\/corner_([xy])\s*=\s*({FLOAT})")
DET_SHIFT_X_RE = re.compile(rf"^\s*header/float/.*/det_shift_x_mm\s*=\s*({FLOAT})")
DET_SHIFT_Y_RE = re.compile(rf"^\s*header/float/.*/det_shift_y_mm\s*=\s*({FLOAT})")
VECTOR_RE = re.compile(rf"^\s*([abc])star\s*=\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+nm\^-1")


@dataclass(frozen=True)
class UnitCell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class DetectorGeometry:
    wavelength_angstrom: float
    distance_mm: float
    pixel_mm: float
    detector_nx: int
    detector_ny: int
    orgx_px: float
    orgy_px: float
    unit_cell: UnitCell


@dataclass
class StreamOrientation:
    frame_number: int
    chunk_id: int
    crystal_in_chunk: int
    event: str
    image_serial: int
    gstar_invA: np.ndarray
    distance_mm: float
    det_shift_x_mm: float
    det_shift_y_mm: float
    observed: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass(frozen=True)
class SimulatedPattern:
    frame_number: int
    event: str
    x_px: np.ndarray
    y_px: np.ndarray
    hkl: np.ndarray
    sg_invA: np.ndarray
    q_invA: np.ndarray
    score: np.ndarray
    observed: list[dict[str, float | int | str]]
    orgx_px: float
    orgy_px: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="Input CrystFEL .stream file")
    parser.add_argument("--output-dir", required=True, help="Directory for simulated pattern outputs")
    parser.add_argument("--dmin", type=float, default=0.8, help="Minimum simulated d-spacing in Angstrom")
    parser.add_argument("--dmax", type=float, default=20.0, help="Maximum simulated d-spacing in Angstrom")
    parser.add_argument(
        "--excitation-tolerance",
        type=float,
        default=0.01,
        help="Ewald excitation tolerance in 1/A for predicted spot visibility",
    )
    parser.add_argument("--max-spots-per-frame", type=int, default=1800, help="Cap plotted simulated spots per frame")
    parser.add_argument("--max-observed-per-frame", type=int, default=1800, help="Cap overlaid stream spots per frame")
    parser.add_argument("--spot-power", type=float, default=1.5, help="Low-q strength proxy power")
    parser.add_argument("--beam-direction", choices=["plus_z", "minus_z"], default="plus_z")
    parser.add_argument(
        "--det-shift-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Convert stream det_shift_mm to beam-center shift as sign * shift_mm / pixel_mm",
    )
    parser.add_argument(
        "--mirror-x-axis",
        dest="mirror_x_axis",
        action="store_true",
        default=True,
        help="Mirror simulated spots about the detector x-axis (flip y about the beam center)",
    )
    parser.add_argument(
        "--no-mirror-x-axis",
        dest="mirror_x_axis",
        action="store_false",
        help="Disable x-axis mirroring for simulated spots",
    )
    parser.add_argument("--montage-cols", type=int, default=6)
    parser.add_argument("--montage-rows", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional quick-look limit")
    parser.add_argument("--skip-individual", action="store_true", help="Only write montage and CSV outputs")
    parser.add_argument(
        "--show-observed",
        dest="show_observed",
        action="store_true",
        default=True,
        help="Overlay observed spots from the stream (default: on)",
    )
    parser.add_argument(
        "--no-show-observed",
        dest="show_observed",
        action="store_false",
        help="Disable overlay of observed spots",
    )
    return parser


def _real_space_matrix_from_unit_cell(cell: UnitCell) -> np.ndarray:
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


def _parse_reflection_row(line: str) -> dict[str, float | int | str] | None:
    parts = line.split()
    if len(parts) < 9:
        return None
    try:
        return {
            "h": int(parts[0]),
            "k": int(parts[1]),
            "l": int(parts[2]),
            "I": float(parts[3]),
            "sigma": float(parts[4]),
            "fs_px": float(parts[7]),
            "ss_px": float(parts[8]),
            "panel": parts[9] if len(parts) > 9 else "",
        }
    except ValueError:
        return None


def parse_stream(stream_path: Path) -> tuple[DetectorGeometry, list[StreamOrientation]]:
    wavelength: float | None = None
    clen_m: float | None = None
    res_px_per_m: float | None = None
    unit_lengths: dict[str, float] = {}
    unit_angles: dict[str, float] = {}
    panel_ranges: dict[str, dict[str, float]] = {}
    panel_corners: dict[str, dict[str, float]] = {}

    orientations: list[StreamOrientation] = []
    in_chunk = False
    in_crystal = False
    in_reflections = False
    chunk_id = -1
    crystal_in_chunk = 0
    event = ""
    image_serial = -1
    current_clen_m: float | None = None
    det_shift_x_mm = 0.0
    det_shift_y_mm = 0.0
    vectors: dict[str, np.ndarray] = {}
    observed: list[dict[str, float | int | str]] = []

    with stream_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith(BEGIN_CHUNK):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                chunk_id += 1
                crystal_in_chunk = 0
                event = ""
                image_serial = -1
                current_clen_m = clen_m
                det_shift_x_mm = 0.0
                det_shift_y_mm = 0.0
                continue
            if raw_line.startswith(END_CHUNK):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                continue

            if not in_chunk:
                if match := WAVELENGTH_RE.match(raw_line):
                    wavelength = float(match.group(1))
                    continue
                if match := CLEN_RE.match(raw_line):
                    clen_m = float(match.group(1))
                    continue
                if match := RES_RE.match(raw_line):
                    res_px_per_m = float(match.group(1))
                    continue
                if match := UNIT_LENGTH_RE.match(raw_line):
                    unit_lengths[match.group(1)] = float(match.group(2))
                    continue
                if match := UNIT_ANGLE_RE.match(raw_line):
                    unit_angles[match.group(1)] = float(match.group(2))
                    continue
                if match := PANEL_RANGE_RE.match(raw_line):
                    panel_ranges.setdefault(match.group(1), {})[match.group(2)] = float(match.group(3))
                    continue
                if match := PANEL_CORNER_RE.match(raw_line):
                    panel_corners.setdefault(match.group(1), {})[match.group(2)] = float(match.group(3))
                    continue
                continue

            if in_reflections:
                if "End of reflections" in raw_line:
                    in_reflections = False
                    continue
                parsed = _parse_reflection_row(raw_line)
                if parsed is not None:
                    observed.append(parsed)
                continue
            if "Reflections measured after indexing" in raw_line:
                in_reflections = True
                continue

            if match := EVENT_RE.match(raw_line):
                event = match.group(1)
                continue
            if match := SERIAL_RE.match(raw_line):
                image_serial = int(match.group(1))
                continue
            if match := AVERAGE_CLEN_RE.match(raw_line):
                current_clen_m = float(match.group(1))
                continue
            if match := DET_SHIFT_X_RE.match(raw_line):
                det_shift_x_mm = float(match.group(1))
                continue
            if match := DET_SHIFT_Y_RE.match(raw_line):
                det_shift_y_mm = float(match.group(1))
                continue

            if raw_line.startswith(BEGIN_CRYSTAL):
                in_crystal = True
                crystal_in_chunk += 1
                vectors = {}
                observed = []
                continue
            if raw_line.startswith(END_CRYSTAL):
                if {"a", "b", "c"} <= vectors.keys():
                    distance = 1000.0 * float(current_clen_m if current_clen_m is not None else clen_m)
                    orientations.append(
                        StreamOrientation(
                            frame_number=len(orientations) + 1,
                            chunk_id=chunk_id,
                            crystal_in_chunk=crystal_in_chunk,
                            event=event,
                            image_serial=image_serial,
                            gstar_invA=np.column_stack([vectors["a"], vectors["b"], vectors["c"]]) / 10.0,
                            distance_mm=distance,
                            det_shift_x_mm=det_shift_x_mm,
                            det_shift_y_mm=det_shift_y_mm,
                            observed=observed,
                        )
                    )
                in_crystal = False
                continue

            if in_crystal:
                if match := VECTOR_RE.match(raw_line):
                    vectors[match.group(1)] = np.asarray(
                        [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                        dtype=float,
                    )

    missing_header = []
    if wavelength is None:
        missing_header.append("wavelength")
    if clen_m is None:
        missing_header.append("clen")
    if res_px_per_m is None:
        missing_header.append("res")
    if not {"a", "b", "c"} <= unit_lengths.keys() or not {"al", "be", "ga"} <= unit_angles.keys():
        missing_header.append("unit cell")
    if missing_header:
        raise ValueError(f"Could not parse stream header fields: {missing_header}")

    panel = sorted(panel_ranges)[0]
    ranges = panel_ranges[panel]
    corners = panel_corners.get(panel, {})
    detector_nx = int(ranges["max_fs"] - ranges["min_fs"] + 1)
    detector_ny = int(ranges["max_ss"] - ranges["min_ss"] + 1)
    orgx_px = -float(corners.get("x", -0.5 * detector_nx)) + float(ranges["min_fs"])
    orgy_px = -float(corners.get("y", -0.5 * detector_ny)) + float(ranges["min_ss"])
    pixel_mm = 1000.0 / float(res_px_per_m)
    cell = UnitCell(
        a=float(unit_lengths["a"]),
        b=float(unit_lengths["b"]),
        c=float(unit_lengths["c"]),
        alpha=float(unit_angles["al"]),
        beta=float(unit_angles["be"]),
        gamma=float(unit_angles["ga"]),
    )
    geometry = DetectorGeometry(
        wavelength_angstrom=float(wavelength),
        distance_mm=1000.0 * float(clen_m),
        pixel_mm=float(pixel_mm),
        detector_nx=detector_nx,
        detector_ny=detector_ny,
        orgx_px=float(orgx_px),
        orgy_px=float(orgy_px),
        unit_cell=cell,
    )
    return geometry, orientations


def generate_candidate_hkls(cell: UnitCell, dmin: float, dmax: float) -> np.ndarray:
    if dmin <= 0.0 or dmax <= 0.0:
        raise ValueError("dmin and dmax must be positive.")
    if dmin > dmax:
        raise ValueError("dmin must be <= dmax.")

    real_space = _real_space_matrix_from_unit_cell(cell)
    reciprocal_ref = np.linalg.inv(real_space)
    hmax = int(ceil(max(cell.a, cell.b, cell.c) / float(dmin))) + 1
    qmin = 1.0 / float(dmax)
    qmax = 1.0 / float(dmin)
    values = np.arange(-hmax, hmax + 1, dtype=int)
    h, k, l = np.meshgrid(values, values, values, indexing="ij")
    hkls = np.column_stack([h.ravel(), k.ravel(), l.ravel()])
    hkls = hkls[np.any(hkls != 0, axis=1)]
    g_ref = hkls.astype(float) @ reciprocal_ref.T
    q = np.linalg.norm(g_ref, axis=1)
    return hkls[(q >= qmin) & (q <= qmax)]


def simulate_pattern(
    orientation: StreamOrientation,
    geometry: DetectorGeometry,
    hkls: np.ndarray,
    beam: np.ndarray,
    excitation_tolerance: float,
    max_spots: int,
    spot_power: float,
    det_shift_sign: float,
    mirror_x_axis: bool,
) -> SimulatedPattern:
    beam_unit = beam / np.linalg.norm(beam)
    k0 = beam_unit / float(geometry.wavelength_angstrom)
    g_vectors = hkls.astype(float) @ orientation.gstar_invA.T
    q = np.linalg.norm(g_vectors, axis=1)
    shifted = g_vectors + k0[None, :]
    sg = np.linalg.norm(shifted, axis=1) - np.linalg.norm(k0)
    denom = shifted @ beam_unit

    orgx = geometry.orgx_px + float(det_shift_sign) * orientation.det_shift_x_mm / geometry.pixel_mm
    orgy = geometry.orgy_px + float(det_shift_sign) * orientation.det_shift_y_mm / geometry.pixel_mm
    x = orgx + (shifted[:, 0] / denom) * (orientation.distance_mm / geometry.pixel_mm)
    y = orgy + (shifted[:, 1] / denom) * (orientation.distance_mm / geometry.pixel_mm)
    if mirror_x_axis:
        y = 2.0 * orgy - y

    mask = (
        (denom > 0.0)
        & (np.abs(sg) <= float(excitation_tolerance))
        & (x >= 0.0)
        & (x < geometry.detector_nx)
        & (y >= 0.0)
        & (y < geometry.detector_ny)
    )
    selected = np.flatnonzero(mask)
    if selected.size:
        tol = max(float(excitation_tolerance), 1e-12)
        sg_weight = np.exp(-0.5 * np.square(sg[selected] / tol))
        strength = np.power(1.0 / np.maximum(q[selected], 0.02), float(spot_power))
        score = sg_weight * strength
        if selected.size > int(max_spots):
            top_local = np.argpartition(score, -int(max_spots))[-int(max_spots) :]
            selected = selected[top_local]
            score = score[top_local]
        order = np.argsort(score)
        selected = selected[order]
        score = score[order]
    else:
        score = np.asarray([], dtype=float)

    return SimulatedPattern(
        frame_number=orientation.frame_number,
        event=orientation.event,
        x_px=x[selected],
        y_px=y[selected],
        hkl=hkls[selected],
        sg_invA=sg[selected],
        q_invA=q[selected],
        score=score,
        observed=orientation.observed,
        orgx_px=float(orgx),
        orgy_px=float(orgy),
    )


def _observed_subset(pattern: SimulatedPattern, max_observed: int) -> list[dict[str, float | int | str]]:
    if int(max_observed) <= 0:
        return []
    observed = pattern.observed
    if len(observed) <= int(max_observed):
        return observed
    intensities = np.asarray([float(row.get("I", 0.0)) for row in observed], dtype=float)
    order = np.argsort(np.maximum(intensities, 0.0))[-int(max_observed) :]
    return [observed[int(index)] for index in order]


def plot_pattern(
    pattern: SimulatedPattern,
    geometry: DetectorGeometry,
    output_path: Path,
    *,
    max_observed: int,
    show_observed: bool,
    dpi: int,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_pattern_on_axis(
        pattern,
        geometry,
        ax,
        max_observed=max_observed,
        show_observed=show_observed,
        title=title,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def draw_pattern_on_axis(
    pattern: SimulatedPattern,
    geometry: DetectorGeometry,
    ax: plt.Axes,
    *,
    max_observed: int,
    show_observed: bool,
    title: str | None = None,
) -> None:
    ax.set_facecolor("black")
    if pattern.x_px.size:
        score = pattern.score
        score_rel = score / np.max(score) if np.max(score) > 0.0 else np.ones_like(score)
        display_score = np.clip(np.power(score_rel, 0.25), 0.12, 1.0)
        ax.scatter(
            pattern.x_px,
            pattern.y_px,
            s=1.2 + 8.0 * display_score,
            c=display_score,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
            edgecolors="none",
        )

    if show_observed:
        observed = _observed_subset(pattern, max_observed)
        if observed:
            ox = np.asarray([float(row["fs_px"]) for row in observed], dtype=float)
            oy = np.asarray([float(row["ss_px"]) for row in observed], dtype=float)
            oi = np.asarray([max(float(row.get("I", 0.0)), 0.0) for row in observed], dtype=float)
            if oi.size and np.max(oi) > 0.0:
                osize = 9.0 + 28.0 * np.sqrt(oi / np.max(oi))
            else:
                osize = np.full(ox.shape, 12.0)
            ax.scatter(ox, oy, s=osize, facecolors="none", edgecolors="#ff4c4c", linewidths=0.45, alpha=0.85)

    ax.scatter([pattern.orgx_px], [pattern.orgy_px], marker="+", s=30, c="#44aaff", linewidths=0.8)
    ax.set_xlim(0, geometry.detector_nx)
    ax.set_ylim(geometry.detector_ny, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"Frame {pattern.frame_number} sim={pattern.x_px.size} obs={len(pattern.observed)}", fontsize=8)


def write_spot_csv(patterns: list[SimulatedPattern], output_path: Path) -> None:
    fields = ["frame_number", "event", "h", "k", "l", "x_px", "y_px", "sg_invA", "q_invA", "score"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pattern in patterns:
            for hkl, x, y, sg, q, score in zip(
                pattern.hkl,
                pattern.x_px,
                pattern.y_px,
                pattern.sg_invA,
                pattern.q_invA,
                pattern.score,
                strict=True,
            ):
                writer.writerow(
                    {
                        "frame_number": pattern.frame_number,
                        "event": pattern.event,
                        "h": int(hkl[0]),
                        "k": int(hkl[1]),
                        "l": int(hkl[2]),
                        "x_px": float(x),
                        "y_px": float(y),
                        "sg_invA": float(sg),
                        "q_invA": float(q),
                        "score": float(score),
                    }
                )


def write_frame_csv(patterns: list[SimulatedPattern], output_path: Path) -> None:
    fields = ["frame_number", "event", "n_simulated_spots", "n_observed_spots", "orgx_px", "orgy_px"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pattern in patterns:
            writer.writerow(
                {
                    "frame_number": pattern.frame_number,
                    "event": pattern.event,
                    "n_simulated_spots": int(pattern.x_px.size),
                    "n_observed_spots": len(pattern.observed),
                    "orgx_px": pattern.orgx_px,
                    "orgy_px": pattern.orgy_px,
                }
            )


def write_count_plot(patterns: list[SimulatedPattern], output_path: Path, dpi: int) -> None:
    frames = np.asarray([pattern.frame_number for pattern in patterns], dtype=int)
    sim_counts = np.asarray([pattern.x_px.size for pattern in patterns], dtype=int)
    obs_counts = np.asarray([len(pattern.observed) for pattern in patterns], dtype=int)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frames, sim_counts, lw=1.4, label="simulated")
    ax.plot(frames, obs_counts, lw=1.0, label="observed indexed", alpha=0.85)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Spot count")
    ax.set_title("Simulated and observed stream spot counts")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_montages(
    patterns: list[SimulatedPattern],
    geometry: DetectorGeometry,
    output_dir: Path,
    rows: int,
    cols: int,
    max_observed: int,
    show_observed: bool,
    dpi: int,
) -> list[Path]:
    per_page = max(1, int(rows) * int(cols))
    n_pages = int(ceil(len(patterns) / per_page))
    paths: list[Path] = []
    for page in range(n_pages):
        page_patterns = patterns[page * per_page : (page + 1) * per_page]
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.65, rows * 1.65))
        flat_axes = np.atleast_1d(axes).ravel()
        for ax, pattern in zip(flat_axes, page_patterns, strict=False):
            draw_pattern_on_axis(
                pattern,
                geometry,
                ax,
                max_observed=max_observed,
                show_observed=show_observed,
                title=f"{pattern.frame_number}: s{pattern.x_px.size}/o{len(pattern.observed)}",
            )
        for ax in flat_axes[len(page_patterns) :]:
            ax.axis("off")
        fig.suptitle(f"Simulated stream patterns page {page + 1}/{n_pages}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / f"simulated_pattern_montage_page_{page + 1:03d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    args = build_parser().parse_args()
    stream_path = Path(args.stream)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry, orientations = parse_stream(stream_path)
    if args.max_frames is not None:
        orientations = orientations[: max(0, int(args.max_frames))]
    if not orientations:
        raise SystemExit("No indexed stream orientations found.")

    hkls = generate_candidate_hkls(geometry.unit_cell, dmin=float(args.dmin), dmax=float(args.dmax))
    beam = np.asarray([0.0, 0.0, 1.0 if args.beam_direction == "plus_z" else -1.0], dtype=float)
    patterns = [
        simulate_pattern(
            orientation,
            geometry,
            hkls,
            beam,
            excitation_tolerance=float(args.excitation_tolerance),
            max_spots=int(args.max_spots_per_frame),
            spot_power=float(args.spot_power),
            det_shift_sign=float(args.det_shift_sign),
            mirror_x_axis=bool(args.mirror_x_axis),
        )
        for orientation in orientations
    ]

    write_frame_csv(patterns, output_dir / "simulated_pattern_frame_summary.csv")
    write_spot_csv(patterns, output_dir / "simulated_pattern_spots.csv")
    write_count_plot(patterns, output_dir / "simulated_pattern_spot_counts.png", int(args.dpi))
    montage_paths = write_montages(
        patterns,
        geometry,
        output_dir,
        rows=max(1, int(args.montage_rows)),
        cols=max(1, int(args.montage_cols)),
        max_observed=int(args.max_observed_per_frame),
        show_observed=bool(args.show_observed),
        dpi=int(args.dpi),
    )

    individual_paths: list[Path] = []
    if not args.skip_individual:
        pattern_dir = output_dir / "individual_patterns"
        pattern_dir.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            path = pattern_dir / f"simulated_pattern_frame_{pattern.frame_number:04d}.png"
            plot_pattern(
                pattern,
                geometry,
                path,
                max_observed=int(args.max_observed_per_frame),
                show_observed=bool(args.show_observed),
                dpi=int(args.dpi),
            )
            individual_paths.append(path)

    report_lines = [
        f"stream: {stream_path}",
        f"n_orientations: {len(orientations)}",
        f"n_candidate_hkls: {hkls.shape[0]}",
        f"dmin: {float(args.dmin)}",
        f"dmax: {float(args.dmax)}",
        f"excitation_tolerance: {float(args.excitation_tolerance)}",
        f"max_spots_per_frame: {int(args.max_spots_per_frame)}",
        f"beam_direction: {args.beam_direction}",
        f"det_shift_sign: {float(args.det_shift_sign)}",
        f"mirror_x_axis: {bool(args.mirror_x_axis)}",
        f"show_observed: {bool(args.show_observed)}",
        "outputs:",
        "simulated_pattern_frame_summary.csv",
        "simulated_pattern_spots.csv",
        "simulated_pattern_spot_counts.png",
        *(path.name for path in montage_paths),
        f"individual_patterns/*.png ({len(individual_paths)} files)" if individual_paths else "individual_patterns skipped",
    ]
    (output_dir / "simulated_pattern_report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"wrote {len(patterns)} simulated patterns to {output_dir}")


if __name__ == "__main__":
    main()
