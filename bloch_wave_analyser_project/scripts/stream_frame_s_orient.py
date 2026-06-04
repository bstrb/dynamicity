#!/usr/bin/env python3
"""Compute frame-level orientation-only S_orient from a CrystFEL stream."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import (  # noqa: E402
    build_zone_axes,
    excitation_error,
    generate_candidate_reflections,
    inside_detector,
    nearest_zone_axis_from_reciprocal_matrix,
    rotate_reference_vectors,
)
from src.metrics import (  # noqa: E402
    geometry_dynamical_risk,
    orientation_excitation_probability,
    orientation_sg_sigma,
    primitive_hkl_key,
    reciprocal_strength_proxy,
)
from src.parsers import GXPARMData, UnitCell, _real_space_matrix_from_unit_cell  # noqa: E402


FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
WAVELENGTH_RE = re.compile(rf"^\s*wavelength\s*=\s*({FLOAT})\s*A")
CLEN_RE = re.compile(rf"^\s*clen\s*=\s*({FLOAT})\s*m")
RES_RE = re.compile(rf"^\s*res\s*=\s*({FLOAT})")
UNIT_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({FLOAT})\s*A")
UNIT_ANGLE_RE = re.compile(rf"^\s*(al|be|ga)\s*=\s*({FLOAT})\s*deg")
PANEL_RANGE_RE = re.compile(r"^\s*(p\d+)\/(min_fs|max_fs|min_ss|max_ss)\s*=\s*(-?\d+)")
PANEL_CORNER_RE = re.compile(rf"^\s*(p\d+)\/corner_([xy])\s*=\s*({FLOAT})")
EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
SERIAL_RE = re.compile(r"^\s*Image serial number:\s*(\d+)")
DET_SHIFT_X_RE = re.compile(rf"^\s*header/float/.*/det_shift_x_mm\s*=\s*({FLOAT})")
DET_SHIFT_Y_RE = re.compile(rf"^\s*header/float/.*/det_shift_y_mm\s*=\s*({FLOAT})")
AVERAGE_CLEN_RE = re.compile(rf"^\s*average_camera_length\s*=\s*({FLOAT})\s*m")
CELL_RE = re.compile(
    rf"^\s*Cell parameters\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+nm,"
    rf"\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+deg"
)
VECTOR_RE = re.compile(rf"^\s*([abc])star\s*=\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+nm\^-1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="Input CrystFEL .stream file")
    parser.add_argument("--output-dir", required=True, help="Directory for frame summary outputs")
    parser.add_argument("--dmin", type=float, default=0.6, help="Minimum d-spacing in Angstrom")
    parser.add_argument("--dmax", type=float, default=50.0, help="Maximum d-spacing in Angstrom")
    parser.add_argument("--excitation-tolerance", type=float, default=1.5e-3, help="Excitation tolerance in 1/A")
    parser.add_argument(
        "--dynamical-environment-tolerance",
        type=float,
        default=1.0e-2,
        help="Broad excitation window in 1/A used by the legacy geometry score",
    )
    parser.add_argument("--dynamical-neighbor-radius", type=float, default=0.12, help="Neighbor radius in 1/A")
    parser.add_argument("--dynamical-zone-axis-sigma", type=float, default=3.0, help="Zone-axis angular sigma in deg")
    parser.add_argument("--zone-axis-limit", type=int, default=5, help="Maximum low-index zone-axis component")
    parser.add_argument("--orientation-sigma-deg", type=float, default=0.2, help="Orientation uncertainty diagnostic")
    parser.add_argument(
        "--beam-direction",
        choices=["plus_z", "minus_z"],
        default="plus_z",
        help="Beam direction convention for excitation and detector projection",
    )
    parser.add_argument(
        "--det-shift-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Convert per-chunk det_shift_mm to beam-center shift as sign * shift_mm / pixel_mm",
    )
    parser.add_argument("--max-crystals", type=int, default=None, help="Optional debugging limit")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N crystals")
    return parser


def _polar_rotation(ub_frame: np.ndarray, ub_ref: np.ndarray) -> np.ndarray:
    raw = ub_frame @ np.linalg.inv(ub_ref)
    u, _, vh = np.linalg.svd(raw)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _project_to_detector_with_center(
    g_vectors: np.ndarray,
    gxparm: GXPARMData,
    *,
    orgx_px: float,
    orgy_px: float,
    distance_mm: float,
    beam_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_gxparm = replace(
        gxparm,
        orgx_px=float(orgx_px),
        orgy_px=float(orgy_px),
        distance_mm=float(distance_mm),
    )
    beam = np.asarray(beam_direction, dtype=float)
    beam_unit = beam / np.linalg.norm(beam)
    k0 = beam_unit / float(frame_gxparm.wavelength_angstrom)
    shifted = g_vectors + k0[None, :]
    denom = shifted @ beam_unit
    x_px = frame_gxparm.orgx_px + (shifted[:, 0] / denom) * (
        frame_gxparm.distance_mm / frame_gxparm.pixel_x_mm
    )
    y_px = frame_gxparm.orgy_px + (shifted[:, 1] / denom) * (
        frame_gxparm.distance_mm / frame_gxparm.pixel_y_mm
    )
    return x_px, y_px, denom > 0.0


def _unit_cell_from_header(lengths: dict[str, float], angles: dict[str, float]) -> UnitCell:
    missing = sorted({"a", "b", "c"}.difference(lengths) | {"al", "be", "ga"}.difference(angles))
    if missing:
        raise ValueError(f"Could not parse full target unit cell from stream header; missing {missing}")
    return UnitCell(
        a=float(lengths["a"]),
        b=float(lengths["b"]),
        c=float(lengths["c"]),
        alpha=float(angles["al"]),
        beta=float(angles["be"]),
        gamma=float(angles["ga"]),
    )


def _first_crystal_cell(stream_path: Path) -> UnitCell | None:
    with stream_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            match = CELL_RE.match(raw_line)
            if match is None:
                continue
            return UnitCell(
                a=10.0 * float(match.group(1)),
                b=10.0 * float(match.group(2)),
                c=10.0 * float(match.group(3)),
                alpha=float(match.group(4)),
                beta=float(match.group(5)),
                gamma=float(match.group(6)),
            )
    return None


def _detector_from_header(
    panel_ranges: dict[str, dict[str, float]],
    panel_corners: dict[str, dict[str, float]],
) -> tuple[int, int, float, float]:
    panel_names = sorted(panel_ranges)
    if not panel_names:
        raise ValueError("Could not parse panel size from stream geometry.")
    panel = panel_names[0]
    ranges = panel_ranges[panel]
    corners = panel_corners.get(panel, {})
    required = {"min_fs", "max_fs", "min_ss", "max_ss"}
    missing = sorted(required.difference(ranges))
    if missing:
        raise ValueError(f"Panel {panel} is missing range fields: {missing}")
    min_fs = int(ranges["min_fs"])
    max_fs = int(ranges["max_fs"])
    min_ss = int(ranges["min_ss"])
    max_ss = int(ranges["max_ss"])
    detector_nx = max_fs - min_fs + 1
    detector_ny = max_ss - min_ss + 1
    orgx_px = -float(corners.get("x", -0.5 * detector_nx)) + min_fs
    orgy_px = -float(corners.get("y", -0.5 * detector_ny)) + min_ss
    return detector_nx, detector_ny, orgx_px, orgy_px


def _write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(summary["frame_number"], summary["S_orient"], lw=1.0)
    ax.set_xlabel("crystal/frame number")
    ax.set_ylabel("S_orient")
    ax.set_title("Stream frame-level orientation score")
    fig.tight_layout()
    fig.savefig(output_dir / "frame_summary_S_orient.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(summary["S_orient"], bins=80, color="#4b7898", edgecolor="none")
    ax.set_xlabel("S_orient")
    ax.set_ylabel("count")
    ax.set_title("S_orient distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "frame_summary_S_orient_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(summary["n_excited"], summary["S_orient"], s=8, alpha=0.45, edgecolors="none")
    ax.set_xlabel("n_excited")
    ax.set_ylabel("S_orient")
    ax.set_title("Excited candidate count vs S_orient")
    fig.tight_layout()
    fig.savefig(output_dir / "frame_summary_S_orient_vs_n_excited.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(summary["frame_number"], summary["det_shift_x_mm"], lw=0.8)
    axes[0].set_ylabel("det_shift_x_mm")
    axes[1].plot(summary["frame_number"], summary["det_shift_y_mm"], lw=0.8)
    axes[1].set_ylabel("det_shift_y_mm")
    axes[1].set_xlabel("crystal/frame number")
    fig.suptitle("Per-chunk detector shifts")
    fig.tight_layout()
    fig.savefig(output_dir / "frame_summary_detector_shifts.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    stream_path = Path(args.stream)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelength: float | None = None
    clen_m: float | None = None
    res_px_per_m: float | None = None
    unit_lengths: dict[str, float] = {}
    unit_angles: dict[str, float] = {}
    panel_ranges: dict[str, dict[str, float]] = {}
    panel_corners: dict[str, dict[str, float]] = {}

    # First pass: lightweight header parse only.
    with stream_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("----- Begin chunk -----"):
                break
            if match := WAVELENGTH_RE.match(raw_line):
                wavelength = float(match.group(1))
            elif match := CLEN_RE.match(raw_line):
                clen_m = float(match.group(1))
            elif match := RES_RE.match(raw_line):
                res_px_per_m = float(match.group(1))
            elif match := UNIT_LENGTH_RE.match(raw_line):
                unit_lengths[match.group(1)] = float(match.group(2))
            elif match := UNIT_ANGLE_RE.match(raw_line):
                unit_angles[match.group(1)] = float(match.group(2))
            elif match := PANEL_RANGE_RE.match(raw_line):
                panel_ranges.setdefault(match.group(1), {})[match.group(2)] = float(match.group(3))
            elif match := PANEL_CORNER_RE.match(raw_line):
                panel_corners.setdefault(match.group(1), {})[match.group(2)] = float(match.group(3))

    if wavelength is None:
        raise ValueError("Could not parse wavelength from stream header.")
    if clen_m is None:
        raise ValueError("Could not parse clen from stream header.")
    if res_px_per_m is None or res_px_per_m <= 0.0:
        raise ValueError("Could not parse detector res from stream header.")

    cell_source = "stream_header"
    try:
        unit_cell = _unit_cell_from_header(unit_lengths, unit_angles)
    except ValueError:
        unit_cell = _first_crystal_cell(stream_path)
        if unit_cell is None:
            raise
        cell_source = "first_indexed_crystal"
    real_space_reference = _real_space_matrix_from_unit_cell(unit_cell)
    reciprocal_reference = np.linalg.inv(real_space_reference)
    detector_nx, detector_ny, base_orgx_px, base_orgy_px = _detector_from_header(panel_ranges, panel_corners)
    pixel_mm = 1000.0 / float(res_px_per_m)
    base_gxparm = GXPARMData(
        phi0_deg=0.0,
        dphi_deg=0.0,
        rotation_axis=np.asarray([0.0, 0.0, 1.0], dtype=float),
        wavelength_angstrom=float(wavelength),
        space_group=1,
        unit_cell=unit_cell,
        real_space_reference=real_space_reference,
        reciprocal_reference=reciprocal_reference,
        detector_nx=int(detector_nx),
        detector_ny=int(detector_ny),
        pixel_x_mm=float(pixel_mm),
        pixel_y_mm=float(pixel_mm),
        orgx_px=float(base_orgx_px),
        orgy_px=float(base_orgy_px),
        distance_mm=1000.0 * float(clen_m),
    )

    candidate_reflections = generate_candidate_reflections(
        base_gxparm,
        dmin_angstrom=float(args.dmin),
        dmax_angstrom=float(args.dmax),
    ).copy()
    candidate_reflections["strength_proxy"] = reciprocal_strength_proxy(
        candidate_reflections["q_invA"].to_numpy(dtype=float)
    )
    reference_vectors = candidate_reflections[["gx_ref", "gy_ref", "gz_ref"]].to_numpy(dtype=float)
    hkl_tuples = [
        tuple(map(int, hkl))
        for hkl in candidate_reflections[["h", "k", "l"]].itertuples(index=False, name=None)
    ]
    hkl_to_index = {hkl: idx for idx, hkl in enumerate(hkl_tuples)}
    row_keys = [primitive_hkl_key(*hkl) for hkl in hkl_tuples]
    zone_axes = build_zone_axes(limit=int(args.zone_axis_limit))
    beam_direction = np.asarray([0.0, 0.0, 1.0 if args.beam_direction == "plus_z" else -1.0], dtype=float)

    rows: list[dict[str, float | int | str]] = []
    in_chunk = False
    in_crystal = False
    in_reflections = False
    chunk_id = -1
    crystal_in_chunk = 0
    crystal_counter = 0
    current_event = ""
    current_serial = -1
    current_det_shift_x_mm = 0.0
    current_det_shift_y_mm = 0.0
    current_clen_m = clen_m
    current_cell: UnitCell | None = None
    vectors: dict[str, np.ndarray] = {}

    with stream_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                chunk_id += 1
                crystal_in_chunk = 0
                current_event = ""
                current_serial = -1
                current_det_shift_x_mm = 0.0
                current_det_shift_y_mm = 0.0
                current_clen_m = clen_m
                continue
            if raw_line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                continue

            if not in_chunk:
                continue

            if in_reflections:
                if "End of reflections" in raw_line:
                    in_reflections = False
                continue
            if "Reflections measured after indexing" in raw_line:
                in_reflections = True
                continue

            if match := EVENT_RE.match(raw_line):
                current_event = match.group(1)
                continue
            if match := SERIAL_RE.match(raw_line):
                current_serial = int(match.group(1))
                continue
            if match := DET_SHIFT_X_RE.match(raw_line):
                current_det_shift_x_mm = float(match.group(1))
                continue
            if match := DET_SHIFT_Y_RE.match(raw_line):
                current_det_shift_y_mm = float(match.group(1))
                continue
            if match := AVERAGE_CLEN_RE.match(raw_line):
                current_clen_m = float(match.group(1))
                continue

            if raw_line.startswith("--- Begin crystal"):
                in_crystal = True
                crystal_in_chunk += 1
                current_cell = None
                vectors = {}
                continue

            if raw_line.startswith("--- End crystal"):
                if not in_crystal:
                    continue
                missing = [axis for axis in ("a", "b", "c") if axis not in vectors]
                if missing:
                    raise ValueError(f"Crystal {crystal_counter} in chunk {chunk_id} is missing {missing}star.")

                ub_frame = np.column_stack([vectors["a"], vectors["b"], vectors["c"]]) / 10.0
                rotation = _polar_rotation(ub_frame, reciprocal_reference)
                g_mid = rotate_reference_vectors(rotation, reference_vectors)
                sg_mid = excitation_error(g_mid, float(wavelength), beam_direction=beam_direction)
                orgx_px = base_orgx_px + float(args.det_shift_sign) * current_det_shift_x_mm / pixel_mm
                orgy_px = base_orgy_px + float(args.det_shift_sign) * current_det_shift_y_mm / pixel_mm
                distance_mm = 1000.0 * float(current_clen_m)
                x_px, y_px, positive_sz = _project_to_detector_with_center(
                    g_mid,
                    base_gxparm,
                    orgx_px=orgx_px,
                    orgy_px=orgy_px,
                    distance_mm=distance_mm,
                    beam_direction=beam_direction,
                )
                frame_gxparm = replace(
                    base_gxparm,
                    orgx_px=float(orgx_px),
                    orgy_px=float(orgy_px),
                    distance_mm=float(distance_mm),
                )
                within_detector = inside_detector(x_px, y_px, frame_gxparm)
                excited_mask = (np.abs(sg_mid) < float(args.excitation_tolerance)) & positive_sz & within_detector

                zone_axis = nearest_zone_axis_from_reciprocal_matrix(
                    zone_axes,
                    ub_frame,
                    beam_direction=beam_direction,
                )
                if np.any(excited_mask):
                    risk_table = geometry_dynamical_risk(
                        candidate_reflections=candidate_reflections,
                        g_vectors_invA=g_mid,
                        sg_invA=sg_mid,
                        target_mask=excited_mask,
                        environment_mask=positive_sz,
                        zone_axis=zone_axis.axis,
                        zone_axis_angle_deg=zone_axis.angle_deg,
                        hkl_to_index=hkl_to_index,
                        hkl_tuples=hkl_tuples,
                        row_keys=row_keys,
                        strength_proxy=candidate_reflections["strength_proxy"].to_numpy(dtype=float),
                        environment_tolerance_invA=float(args.dynamical_environment_tolerance),
                        neighbor_radius_invA=float(args.dynamical_neighbor_radius),
                        zone_axis_sigma_deg=float(args.dynamical_zone_axis_sigma),
                    )
                    s_orient = float(risk_table["S_dyn"].sum())
                    mean_strength = float(risk_table["strength_proxy"].mean())
                    mean_neighbor_density = float(risk_table["neighbor_density"].mean())
                    mean_row_coupling = float(risk_table["row_coupling"].mean())
                else:
                    s_orient = 0.0
                    mean_strength = 0.0
                    mean_neighbor_density = 0.0
                    mean_row_coupling = 0.0

                orientation_sigma = orientation_sg_sigma(
                    g_vectors_invA=g_mid[excited_mask],
                    wavelength_angstrom=float(wavelength),
                    orientation_sigma_deg=float(args.orientation_sigma_deg),
                    beam_direction=beam_direction,
                )
                p_excited = orientation_excitation_probability(
                    sg_invA=sg_mid[excited_mask],
                    sg_sigma_orient_invA=orientation_sigma,
                    excitation_tolerance_invA=float(args.excitation_tolerance),
                )

                cell = current_cell or unit_cell
                rows.append(
                    {
                        "frame": int(crystal_counter),
                        "frame_number": int(crystal_counter + 1),
                        "chunk_id": int(chunk_id),
                        "crystal_in_chunk": int(crystal_in_chunk),
                        "event": current_event,
                        "image_serial": int(current_serial),
                        "det_shift_x_mm": float(current_det_shift_x_mm),
                        "det_shift_y_mm": float(current_det_shift_y_mm),
                        "orgx_px": float(orgx_px),
                        "orgy_px": float(orgy_px),
                        "distance_mm": float(distance_mm),
                        "cell_a_angstrom": float(cell.a),
                        "cell_b_angstrom": float(cell.b),
                        "cell_c_angstrom": float(cell.c),
                        "zone_axis": zone_axis.label,
                        "zone_axis_angle_deg": float(zone_axis.angle_deg),
                        "n_excited": int(np.count_nonzero(excited_mask)),
                        "S_dyn": float(s_orient),
                        "S_orient": float(s_orient),
                        "mean_strength_proxy": float(mean_strength),
                        "mean_neighbor_density": float(mean_neighbor_density),
                        "mean_row_coupling": float(mean_row_coupling),
                        "mean_orientation_sigma_sg_invA": (
                            float(np.mean(orientation_sigma)) if orientation_sigma.size else 0.0
                        ),
                        "mean_orientation_p_excited": float(np.mean(p_excited)) if p_excited.size else 0.0,
                    }
                )

                crystal_counter += 1
                if args.max_crystals is not None and crystal_counter >= int(args.max_crystals):
                    break
                if int(args.progress_every) > 0 and crystal_counter % int(args.progress_every) == 0:
                    print(f"processed {crystal_counter} crystals", flush=True)
                in_crystal = False
                continue

            if in_crystal:
                if match := CELL_RE.match(raw_line):
                    current_cell = UnitCell(
                        a=10.0 * float(match.group(1)),
                        b=10.0 * float(match.group(2)),
                        c=10.0 * float(match.group(3)),
                        alpha=float(match.group(4)),
                        beta=float(match.group(5)),
                        gamma=float(match.group(6)),
                    )
                    continue
                if match := VECTOR_RE.match(raw_line):
                    vectors[match.group(1)] = np.asarray(
                        [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                        dtype=float,
                    )
                    continue
        else:
            crystal_counter_final = crystal_counter
            crystal_counter = crystal_counter_final

    summary = pd.DataFrame.from_records(rows)
    summary_path = output_dir / "frame_summary.csv"
    summary.to_csv(summary_path, index=False)

    text = [
        f"stream: {stream_path}",
        f"n_crystals: {int(summary.shape[0])}",
        f"dmin_angstrom: {float(args.dmin)}",
        f"dmax_angstrom: {float(args.dmax)}",
        f"n_candidate_reflections: {int(candidate_reflections.shape[0])}",
        f"wavelength_angstrom: {float(wavelength)}",
        f"reference_cell_source: {cell_source}",
        f"reference_cell: {unit_cell.a} {unit_cell.b} {unit_cell.c} {unit_cell.alpha} {unit_cell.beta} {unit_cell.gamma}",
        f"pixel_mm: {float(pixel_mm)}",
        f"detector_nx: {int(detector_nx)}",
        f"detector_ny: {int(detector_ny)}",
        f"base_orgx_px: {float(base_orgx_px)}",
        f"base_orgy_px: {float(base_orgy_px)}",
        f"det_shift_sign: {float(args.det_shift_sign)}",
        f"beam_direction: {args.beam_direction}",
        f"mean_S_orient: {float(summary['S_orient'].mean()) if not summary.empty else float('nan')}",
        f"p95_S_orient: {float(summary['S_orient'].quantile(0.95)) if not summary.empty else float('nan')}",
        f"max_S_orient: {float(summary['S_orient'].max()) if not summary.empty else float('nan')}",
        f"mean_n_excited: {float(summary['n_excited'].mean()) if not summary.empty else float('nan')}",
        f"max_n_excited: {int(summary['n_excited'].max()) if not summary.empty else 'nan'}",
    ]
    (output_dir / "stream_s_orient_summary.txt").write_text("\n".join(text) + "\n")
    _write_plots(summary, output_dir)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {output_dir / 'stream_s_orient_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
