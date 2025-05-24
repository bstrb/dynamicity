#!/usr/bin/env python3
"""
misorientation_to_zone_axis.py
--------------------------------
For every indexed chunk in a CrystFEL *.stream* file, find the crystallographic
zone axis [u v w] closest to the incident-beam direction (+z) and the angular
deviation θ.  Output options:

* Nicely formatted table on stdout.
* CSV file (optional).
* θ–rank plot (optional).
* **NEW:** a second *.stream* whose pre-amble and chunks are reordered by θ
           (smallest first) — pass  --sorted-stream <PATH>.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from typing import List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Optional plotting backend
try:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except ModuleNotFoundError:
    _HAVE_MPL = False

# ----------------------------------------------------------------------
# Regex helpers
vec_re = re.compile(
    r"^[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)"
)

event_re = re.compile(r"^Event:\s*(\S+)")

# ----------------------------------------------------------------------
# Core maths


def nearest_zone_axis(
    Mstar: np.ndarray,
    *,
    hmax: int = 3,
    centering: str = "P",
) -> Tuple[Tuple[int, int, int], float]:
    """Return (u, v, w), θ_deg for the zone axis closest to beam (+z)."""
    M = np.linalg.inv(Mstar).T  # direct basis columns in lab coords
    k = np.array([0.0, 0.0, 1.0])  # beam direction

    best_axis = (0, 0, 1)
    best_theta = 180.0
    centering = centering.upper()

    for u in range(-hmax, hmax + 1):
        for v in range(-hmax, hmax + 1):
            for w in range(-hmax, hmax + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                if centering == "I" and (u + v + w) % 2:
                    continue
                if centering == "F" and (u % 2 + v % 2 + w % 2) % 2:
                    continue
                if centering in {"A", "B", "C"}:
                    if centering == "A" and (u + w) % 2:
                        continue
                    if centering == "B" and (v + w) % 2:
                        continue
                    if centering == "C" and (u + v) % 2:
                        continue

                v_lab = M @ np.array([u, v, w], dtype=float)
                cosang = abs(np.dot(v_lab, k)) / np.linalg.norm(v_lab)
                theta = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                if theta < best_theta:
                    best_theta = theta
                    best_axis = (u, v, w)
    return best_axis, best_theta


# ----------------------------------------------------------------------
# Stream parsing


def parse_stream(
    path: pathlib.Path,
) -> Tuple[
    List[str],
    List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, List[str]]],
]:
    """
    Return:
        header_lines  – list[str]     (everything before first chunk)
        chunks        – list of tuples:
            (event_id, a*, b*, c*, chunk_lines)
    """
    header_lines: list[str] = []
    chunks: list[
        Tuple[str, np.ndarray, np.ndarray, np.ndarray, List[str]]
    ] = []

    with path.open() as fh:
        in_chunk = False
        seen_first_chunk = False
        chunk_buf: list[str] = []
        cur_event = None
        a = b = c = None

        for line in fh:
            if line.startswith("----- Begin chunk"):
                in_chunk = True
                seen_first_chunk = True
                chunk_buf = [line]
                cur_event = a = b = c = None
                continue

            if not seen_first_chunk:
                header_lines.append(line)
                continue

            if in_chunk:
                chunk_buf.append(line)

            if line.startswith("----- End chunk ----") and in_chunk:
                in_chunk = False
                if cur_event and a is not None and b is not None and c is not None:
                    chunks.append((cur_event, a, b, c, chunk_buf.copy()))
                continue

            if not in_chunk:
                continue

            m_event = event_re.match(line)
            if m_event:
                cur_event = m_event.group(1)
                continue

            m_vec = vec_re.match(line)
            if m_vec:
                vec = np.array([float(m_vec.group(i)) for i in (1, 2, 3)])
                if line.startswith("astar"):
                    a = vec
                elif line.startswith("bstar"):
                    b = vec
                elif line.startswith("cstar"):
                    c = vec

    return header_lines, chunks


# ----------------------------------------------------------------------
# Main CLI


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compute the angular deviation of each indexed orientation to the "
            "nearest crystallographic zone axis."
        ),
    )
    p.add_argument("stream", type=pathlib.Path, help="CrystFEL .stream file")
    p.add_argument("--hmax", type=int, default=3, help="search |h|,|k|,|l| ≤ hmax")
    p.add_argument(
        "--centering",
        default="P",
        help="lattice centering symbol (P/I/F/A/B/C)",
    )
    p.add_argument("--csv", type=pathlib.Path, help="write CSV output")
    p.add_argument("--plot", type=pathlib.Path, help="save a PNG/PDF plot")
    p.add_argument(
        "--no-sort",
        action="store_true",
        help="keep original chunk order (do not sort)",
    )
    p.add_argument(
        "--sorted-stream",
        type=pathlib.Path,
        help="write a copy of the input .stream with chunks reordered by θ",
    )

    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    header_lines, chunks = parse_stream(args.stream)
    if not chunks:
        sys.exit("[error] No orientation matrices found!")

    results: List[
        Tuple[str, Tuple[int, int, int], float, List[str]]
    ] = []  # (event, uvw, theta, chunk_lines)

    for event, a, b, c, chunk_lines in chunks:
        Mstar = np.column_stack((a, b, c))
        uvw, theta = nearest_zone_axis(
            Mstar, hmax=args.hmax, centering=args.centering
        )
        results.append((event, uvw, theta, chunk_lines))

    if not args.no_sort:
        results.sort(key=lambda x: x[2])  # by θ

    # ------------------------------------------------------------------
    # Pretty print
    print(f"{'Event':<20}  {'[u v w]':<9}  θ (°)")
    print("-" * 40)
    for ev, (u, v, w), th, _ in results:
        print(f"{ev:<20}  [{u:2d} {v:2d} {w:2d}]  {th:7.3f}")

    # ------------------------------------------------------------------
    # CSV
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["Event", "u", "v", "w", "theta_deg"])
            for ev, (u, v, w), th, _ in results:
                wr.writerow([ev, u, v, w, f"{th:.6f}"])
        print(f"[info] CSV written to {args.csv}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Sorted stream (keep header!)
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fh:
            fh.writelines(header_lines)
            for _, _, _, chunk_lines in results:
                fh.writelines(chunk_lines)
        print(
            f"[info] Sorted stream written to {args.sorted_stream}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Plot
    if args.plot:
        if not _HAVE_MPL:
            print("[warn] matplotlib not installed; skipping plot", file=sys.stderr)
        else:
            theta_vals = np.array([th for _, _, th, _ in results])
            ranks = np.arange(1, len(results) + 1)
            uvw_labels = [f"[{u} {v} {w}]" for _, (u, v, w), _, _ in results]
            unique_axes = sorted(set(uvw_labels))
            cmap = plt.get_cmap("tab10", len(unique_axes))
            colors = {ax: cmap(i) for i, ax in enumerate(unique_axes)}

            fig, ax = plt.subplots(figsize=(6, 4))
            for ax_label in unique_axes:
                mask = [lbl == ax_label for lbl in uvw_labels]
                ax.scatter(
                    ranks[mask],
                    theta_vals[mask],
                    label=ax_label,
                    s=15,
                    alpha=0.7,
                    color=colors[ax_label],
                )

            ax.set_xlabel("Rank (sorted by θ)")
            ax.set_ylabel("θ to nearest zone axis (°)")
            ax.set_title("Misorientation to zone axis vs. event rank")
            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.legend(frameon=False, fontsize="small", title="Zone axis")
            fig.tight_layout()
            fig.savefig(args.plot)
            print(f"[info] Plot saved to {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
