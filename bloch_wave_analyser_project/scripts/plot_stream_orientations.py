#!/usr/bin/env python3
"""Plot lightweight CrystFEL stream orientation diagnostics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import acos, ceil, gcd, sqrt
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
VECTOR_RE = re.compile(rf"^\s*([abc])star\s*=\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s+nm\^-1")


@dataclass(frozen=True)
class OrientationRecord:
    crystal_index: int
    frame_number: int
    chunk_id: int
    crystal_in_chunk: int
    event: str
    image_serial: int
    gstar_invA: np.ndarray
    real_space: np.ndarray
    beam_uvw: np.ndarray
    zone_axis: tuple[int, int, int]
    zone_axis_angle_deg: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, help="Input CrystFEL .stream file")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and PNG diagnostics")
    parser.add_argument("--zone-axis-limit", type=int, default=5, help="Maximum low-index UVW component")
    parser.add_argument(
        "--beam-direction",
        choices=["plus_z", "minus_z"],
        default="plus_z",
        help="Beam direction used for zone-axis matching",
    )
    parser.add_argument("--montage-cols", type=int, default=6, help="Orientation tiles per montage row")
    parser.add_argument("--montage-rows", type=int, default=7, help="Orientation tiles per montage column")
    parser.add_argument("--dpi", type=int, default=180, help="Output PNG resolution")
    return parser


def build_zone_axes(limit: int) -> list[tuple[int, int, int]]:
    axes: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for u in range(-limit, limit + 1):
        for v in range(-limit, limit + 1):
            for w in range(-limit, limit + 1):
                if u == 0 and v == 0 and w == 0:
                    continue
                divisor = gcd(gcd(abs(u), abs(v)), abs(w))
                axis = (u // divisor, v // divisor, w // divisor)
                for value in axis:
                    if value != 0:
                        if value < 0:
                            axis = (-axis[0], -axis[1], -axis[2])
                        break
                if axis not in seen:
                    seen.add(axis)
                    axes.append(axis)
    return axes


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 180.0
    cosang = min(1.0, abs(float(v1 @ v2)) / (n1 * n2))
    return float(np.rad2deg(acos(cosang)))


def nearest_zone_axis(
    real_space: np.ndarray,
    zone_axes: list[tuple[int, int, int]],
    beam: np.ndarray,
) -> tuple[tuple[int, int, int], float]:
    best_axis = (0, 0, 1)
    best_angle = 180.0
    for axis in zone_axes:
        direction = real_space @ np.asarray(axis, dtype=float)
        angle = _angle_deg(direction, beam)
        if angle < best_angle:
            best_axis = axis
            best_angle = angle
    return best_axis, best_angle


def _beam_uvw(real_space: np.ndarray, beam: np.ndarray) -> np.ndarray:
    uvw = np.linalg.solve(real_space, beam)
    norm = float(np.linalg.norm(uvw))
    if norm == 0.0:
        return uvw
    uvw = uvw / norm
    for value in (uvw[2], uvw[1], uvw[0]):
        if abs(float(value)) > 1e-12:
            if value < 0.0:
                uvw = -uvw
            break
    return uvw


def parse_stream_orientations(
    stream_path: Path,
    zone_axes: list[tuple[int, int, int]],
    beam: np.ndarray,
) -> list[OrientationRecord]:
    records: list[OrientationRecord] = []
    in_chunk = False
    in_crystal = False
    chunk_id = -1
    crystal_in_chunk = 0
    event = ""
    image_serial = -1
    vectors: dict[str, np.ndarray] = {}

    with stream_path.open("r", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith(BEGIN_CHUNK):
                in_chunk = True
                in_crystal = False
                chunk_id += 1
                crystal_in_chunk = 0
                event = ""
                image_serial = -1
                vectors = {}
                continue
            if raw_line.startswith(END_CHUNK):
                in_chunk = False
                in_crystal = False
                continue
            if not in_chunk:
                continue

            match = EVENT_RE.match(raw_line)
            if match is not None:
                event = match.group(1)
                continue
            match = SERIAL_RE.match(raw_line)
            if match is not None:
                image_serial = int(match.group(1))
                continue

            if raw_line.startswith(BEGIN_CRYSTAL):
                in_crystal = True
                crystal_in_chunk += 1
                vectors = {}
                continue
            if raw_line.startswith(END_CRYSTAL):
                if {"a", "b", "c"} <= vectors.keys():
                    gstar_invA = np.column_stack([vectors["a"], vectors["b"], vectors["c"]]) / 10.0
                    real_space = np.linalg.inv(gstar_invA).T
                    axis, angle = nearest_zone_axis(real_space, zone_axes, beam)
                    crystal_index = len(records)
                    records.append(
                        OrientationRecord(
                            crystal_index=crystal_index,
                            frame_number=crystal_index + 1,
                            chunk_id=chunk_id,
                            crystal_in_chunk=crystal_in_chunk,
                            event=event,
                            image_serial=image_serial,
                            gstar_invA=gstar_invA,
                            real_space=real_space,
                            beam_uvw=_beam_uvw(real_space, beam),
                            zone_axis=axis,
                            zone_axis_angle_deg=angle,
                        )
                    )
                in_crystal = False
                continue

            if in_crystal:
                match = VECTOR_RE.match(raw_line)
                if match is not None:
                    vectors[match.group(1)] = np.asarray(
                        [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                        dtype=float,
                    )

    return records


def axis_label(axis: tuple[int, int, int]) -> str:
    return f"[{axis[0]} {axis[1]} {axis[2]}]"


def write_csv(records: list[OrientationRecord], output_path: Path) -> None:
    fields = [
        "crystal_index",
        "frame_number",
        "chunk_id",
        "crystal_in_chunk",
        "event",
        "image_serial",
        "zone_axis",
        "zone_axis_u",
        "zone_axis_v",
        "zone_axis_w",
        "zone_axis_angle_deg",
        "beam_u",
        "beam_v",
        "beam_w",
        "astar_x_invA",
        "astar_y_invA",
        "astar_z_invA",
        "bstar_x_invA",
        "bstar_y_invA",
        "bstar_z_invA",
        "cstar_x_invA",
        "cstar_y_invA",
        "cstar_z_invA",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            g = record.gstar_invA
            row = {
                "crystal_index": record.crystal_index,
                "frame_number": record.frame_number,
                "chunk_id": record.chunk_id,
                "crystal_in_chunk": record.crystal_in_chunk,
                "event": record.event,
                "image_serial": record.image_serial,
                "zone_axis": axis_label(record.zone_axis),
                "zone_axis_u": record.zone_axis[0],
                "zone_axis_v": record.zone_axis[1],
                "zone_axis_w": record.zone_axis[2],
                "zone_axis_angle_deg": record.zone_axis_angle_deg,
                "beam_u": record.beam_uvw[0],
                "beam_v": record.beam_uvw[1],
                "beam_w": record.beam_uvw[2],
                "astar_x_invA": g[0, 0],
                "astar_y_invA": g[1, 0],
                "astar_z_invA": g[2, 0],
                "bstar_x_invA": g[0, 1],
                "bstar_y_invA": g[1, 1],
                "bstar_z_invA": g[2, 1],
                "cstar_x_invA": g[0, 2],
                "cstar_y_invA": g[1, 2],
                "cstar_z_invA": g[2, 2],
            }
            writer.writerow(row)


def plot_angle_summary(records: list[OrientationRecord], output_path: Path, dpi: int) -> None:
    frames = np.asarray([record.frame_number for record in records], dtype=int)
    angles = np.asarray([record.zone_axis_angle_deg for record in records], dtype=float)
    labels = [axis_label(record.zone_axis) for record in records]
    unique_labels = sorted(set(labels))
    color_map = {label: plt.cm.tab20(i % 20) for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.scatter(frames, angles, c=colors, s=22, linewidths=0.2, edgecolors="black")
    ax.set_xlabel("Crystal/frame number")
    ax.set_ylabel("Nearest zone-axis angle (deg)")
    ax.set_title("Stream orientations: nearest low-index zone axis")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _lambert_xy(uvw: np.ndarray) -> tuple[float, float]:
    norm = float(np.linalg.norm(uvw))
    if norm == 0.0:
        return 0.0, 0.0
    n = uvw / norm
    if n[2] < 0.0:
        n = -n
    denom = max(1e-12, 1.0 + float(n[2]))
    scale = sqrt(2.0 / denom)
    return float(scale * n[0]), float(scale * n[1])


def plot_pole_figure(records: list[OrientationRecord], output_path: Path, dpi: int) -> None:
    xy = np.asarray([_lambert_xy(record.beam_uvw) for record in records], dtype=float)
    angles = np.asarray([record.zone_axis_angle_deg for record in records], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    circle = plt.Circle((0.0, 0.0), sqrt(2.0), fill=False, color="black", lw=1.0, alpha=0.5)
    ax.add_patch(circle)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=angles, s=28, cmap="viridis_r", edgecolors="black", linewidths=0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.52, 1.52)
    ax.set_ylim(-1.52, 1.52)
    ax.set_xlabel("Beam direction u component")
    ax.set_ylabel("Beam direction v component")
    ax.set_title("Beam direction in crystal UVW coordinates")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    cbar.set_label("Nearest zone-axis angle (deg)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_axis_histogram(records: list[OrientationRecord], output_path: Path, dpi: int) -> None:
    counts: dict[str, int] = {}
    for record in records:
        label = axis_label(record.zone_axis)
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]

    height = max(4.0, 0.24 * len(labels))
    fig, ax = plt.subplots(figsize=(8.5, height))
    ax.barh(np.arange(len(labels)), values, color="#4f7f9f")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Nearest zone-axis counts")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _draw_orientation_tile(ax: plt.Axes, record: OrientationRecord) -> None:
    colors = {"a": "#c63f3f", "b": "#3b8c55", "c": "#3867b7"}
    ax.axhline(0.0, color="0.85", lw=0.5)
    ax.axvline(0.0, color="0.85", lw=0.5)
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="0.8", lw=0.6))
    for index, name in enumerate(("a", "b", "c")):
        vector = record.real_space[:, index]
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        unit = vector / norm
        ax.arrow(
            0.0,
            0.0,
            0.78 * float(unit[0]),
            0.78 * float(unit[1]),
            color=colors[name],
            width=0.008,
            head_width=0.055,
            length_includes_head=True,
            alpha=0.9,
        )
        ax.text(0.88 * float(unit[0]), 0.88 * float(unit[1]), name, color=colors[name], fontsize=6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{record.frame_number}: {axis_label(record.zone_axis)}\n{record.zone_axis_angle_deg:.2f} deg",
        fontsize=6,
        pad=1.5,
    )


def plot_montages(
    records: list[OrientationRecord],
    output_dir: Path,
    rows: int,
    cols: int,
    dpi: int,
) -> list[Path]:
    per_page = max(1, int(rows) * int(cols))
    n_pages = int(ceil(len(records) / per_page))
    written: list[Path] = []
    for page in range(n_pages):
        page_records = records[page * per_page : (page + 1) * per_page]
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.75))
        flat_axes = np.atleast_1d(axes).ravel()
        for ax, record in zip(flat_axes, page_records, strict=False):
            _draw_orientation_tile(ax, record)
        for ax in flat_axes[len(page_records) :]:
            ax.axis("off")
        fig.suptitle(f"Stream orientation montage page {page + 1}/{n_pages}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        path = output_dir / f"orientation_montage_page_{page + 1:03d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    args = build_parser().parse_args()
    stream_path = Path(args.stream)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beam = np.asarray([0.0, 0.0, 1.0 if args.beam_direction == "plus_z" else -1.0], dtype=float)
    zone_axes = build_zone_axes(int(args.zone_axis_limit))
    records = parse_stream_orientations(stream_path, zone_axes, beam)
    if not records:
        raise SystemExit("No indexed crystal orientations found in stream.")

    write_csv(records, output_dir / "orientation_summary.csv")
    plot_angle_summary(records, output_dir / "orientation_zone_axis_angle.png", int(args.dpi))
    plot_pole_figure(records, output_dir / "orientation_pole_figure.png", int(args.dpi))
    plot_axis_histogram(records, output_dir / "orientation_zone_axis_counts.png", int(args.dpi))
    montage_paths = plot_montages(
        records,
        output_dir,
        rows=max(1, int(args.montage_rows)),
        cols=max(1, int(args.montage_cols)),
        dpi=int(args.dpi),
    )

    report_lines = [
        f"stream: {stream_path}",
        f"n_orientations: {len(records)}",
        f"beam_direction: {args.beam_direction}",
        f"zone_axis_limit: {int(args.zone_axis_limit)}",
        "outputs:",
        "orientation_summary.csv",
        "orientation_zone_axis_angle.png",
        "orientation_pole_figure.png",
        "orientation_zone_axis_counts.png",
        *(path.name for path in montage_paths),
    ]
    (output_dir / "orientation_report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"wrote {len(records)} orientations to {output_dir}")


if __name__ == "__main__":
    main()
