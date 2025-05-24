#!/usr/bin/env python3
"""
misorientation_to_zone_axis.py
-------------------------------------------------
Analyse a CrystFEL *.stream* file.

For every indexed chunk it
  • finds the nearest crystallographic zone axis to the beam (+z);
  • reports the misorientation angle θ;
  • computes a **danger score**

        danger = max_i  C_i · exp[−(θ_i / θ0)²]

    where   C_i   = crowdedness (plane count) of crowded axis *i*
            θ_i   = this chunk’s angle to axis *i*
            θ0    = scale parameter (default 3°);

  • prints a table sorted by danger (highest first),
    writes a CSV, a θ-rank plot and a danger-sorted *.stream* copy.

Crowded axes (default: TOP=8) are determined automatically from the first
orientation matrix, taking all reflections with |g| ≤ gmax and counting how
many lie in each zone.  gmax is auto-chosen unless you pass --gmax.

-------------------------------------------------
Usage example
-------------
python misorientation_to_zone_axis.py run.stream             \\
       --hmax 4             \\
       --top-crowded 8      \\
       --theta0 3           \\
       --csv  results.csv   \\
       --plot results.png   \\
       --sorted-stream run_sorted.stream
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except ModuleNotFoundError:
    _HAVE_MPL = False

# ------------------------------------------------------------------ regexes
VEC_RE = re.compile(
    r"^[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)"
)
EVENT_RE = re.compile(r"^Event:\s*(\S+)")
LATTYPE_RE = re.compile(r"^lattice_type\s*=\s*(\S+)", re.I)
CENTERING_RE = re.compile(r"^centering\s*=\s*(\S+)", re.I)
UNIQUE_RE = re.compile(r"^unique_axis\s*=\s*(\S+)", re.I)

# ------------------------------------------------------------------ helpers
def nearest_zone_axis(
    Mstar: np.ndarray,
    *,
    hmax: int,
    centering: str,
) -> Tuple[Tuple[int, int, int], float]:
    """Return (u,v,w), θ° for axis closest to +z."""
    M = np.linalg.inv(Mstar).T          # real-space basis in lab frame
    k = np.array([0.0, 0.0, 1.0])
    best_axis, best_theta = (0, 0, 1), 180.0
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

                v_lab = M @ np.array([u, v, w], float)
                cosang = abs(np.dot(v_lab, k)) / np.linalg.norm(v_lab)
                theta = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                if theta < best_theta:
                    best_theta, best_axis = theta, (u, v, w)
    return best_axis, best_theta


def crowded_axes(
    Mstar: np.ndarray,
    *,
    hmax: int,
    centering: str,
    gmax: float,
    top_n: int,
) -> List[Tuple[Tuple[int, int, int], int]]:
    """Return the `top_n` zone axes with highest plane count."""
    centering = centering.upper()
    # list reflections inside sphere
    refl: List[np.ndarray] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if centering == "I" and (h + k + l) % 2:
                    continue
                if centering == "F" and (h % 2 + k % 2 + l % 2) % 2:
                    continue
                if centering in {"A", "B", "C"}:
                    if centering == "A" and (h + l) % 2:
                        continue
                    if centering == "B" and (k + l) % 2:
                        continue
                    if centering == "C" and (h + k) % 2:
                        continue
                g_cart = np.array([h, k, l], float) @ Mstar  # lab coords
                if np.linalg.norm(g_cart) <= gmax:
                    refl.append(np.array([h, k, l], int))

    # count per axis
    scores: Dict[Tuple[int, int, int], int] = {}
    for u in range(-hmax, hmax + 1):
        for v in range(-hmax, hmax + 1):
            for w in range(-hmax, hmax + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                axis = (u, v, w)
                n = 0
                for hkl in refl:
                    if np.dot(axis, hkl) == 0:
                        n += 1
                scores[axis] = n

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


# ------------------------------------------------------------------ stream parsing
def parse_stream(
    path: pathlib.Path,
) -> Tuple[
    Tuple[str | None, str | None, str | None],
    List[str],
    List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, List[str]]],
]:
    """Return (lattice, centering, unique_axis), header_lines, chunk_list."""
    header, chunks = [], []
    lat_type = centering_lbl = unique_axis = None

    with path.open() as fh:
        in_chunk = seen_first = False
        buf: List[str] = []
        cur_event, a = None, None
        b = c = None
        for line in fh:
            if line.startswith("----- Begin chunk"):
                in_chunk = seen_first = True
                buf = [line]
                cur_event = a = b = c = None
                continue

            if not seen_first:
                header.append(line)
            if in_chunk:
                buf.append(line)

            if line.startswith("----- End chunk ----") and in_chunk:
                in_chunk = False
                if cur_event and a is not None and b is not None and c is not None:
                    chunks.append((cur_event, a, b, c, buf.copy()))
                continue
            if not in_chunk:
                # still allow lattice lines before chunk
                if (m := LATTYPE_RE.match(line)):
                    lat_type = m.group(1).lower()
                if (m := CENTERING_RE.match(line)):
                    centering_lbl = m.group(1).upper()
                if (m := UNIQUE_RE.match(line)):
                    unique_axis = m.group(1).lower()
                continue

            # inside a chunk
            if (m := EVENT_RE.match(line)):
                cur_event = m.group(1)
                continue
            if (m := VEC_RE.match(line)):
                vec = np.array([float(m.group(i)) for i in (1, 2, 3)])
                if line.startswith("astar"):
                    a = vec
                elif line.startswith("bstar"):
                    b = vec
                elif line.startswith("cstar"):
                    c = vec

    return (lat_type, centering_lbl, unique_axis), header, chunks


# ------------------------------------------------------------------ main
def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Rank orientations by combined crowdedness / angle danger."
    )
    p.add_argument("stream", type=pathlib.Path)
    p.add_argument("--hmax", type=int, default=3, help="search |h|≤hmax")
    p.add_argument("--centering", help="override centering symbol (P/I/F/...)")
    p.add_argument("--gmax", type=float, help="reciprocal-radius cut-off (nm⁻¹)")
    p.add_argument(
        "--top-crowded",
        type=int,
        default=8,
        help="keep N most crowded axes (default 8)",
    )
    p.add_argument(
        "--theta0",
        type=float,
        default=3.0,
        help="scale angle θ0 in danger formula (deg, default 3)",
    )
    p.add_argument("--csv", type=pathlib.Path, help="write CSV")
    p.add_argument("--plot", type=pathlib.Path, help="write PNG/PDF")
    p.add_argument("--sorted-stream", type=pathlib.Path)
    p.add_argument("--no-sort", action="store_true")

    args = p.parse_args(argv)

    (lat_type, cent_from_stream, _), header, chunks = parse_stream(args.stream)
    if not chunks:
        sys.exit("[error] No orientation matrices found")

    centering = args.centering or cent_from_stream or "P"

    # representative metric from first chunk
    first_Mstar = np.column_stack(chunks[0][1:4])
    if args.gmax is not None:
        g_cut = args.gmax
    else:
        norms = [np.linalg.norm(first_Mstar[:, i]) for i in range(3)]
        g_cut = 1.2 * min(norms)

    crowded = crowded_axes(
        first_Mstar,
        hmax=args.hmax,
        centering=centering,
        gmax=g_cut,
        top_n=args.top_crowded,
    )

    print(
        f"\nMost crowded zone axes (|g| ≤ {g_cut:.3f} nm⁻¹, "
        f"{lat_type or 'unknown'} {centering}-centred):"
    )
    for i, (ax, npl) in enumerate(crowded, 1):
        u, v, w = ax
        print(f" {i:2d}. [{u:2d} {v:2d} {w:2d}]  planes: {npl}")

    crowd_axes = np.array([ax for ax, _ in crowded])
    crowd_counts = np.array([n for _, n in crowded], float)
    theta0 = args.theta0

    results: List[
        Tuple[str, Tuple[int, int, int], float, List[str], float]
    ] = []  # event, axis, theta, chunk_lines, danger

    k_lab = np.array([0.0, 0.0, 1.0])

    for ev, a, b, c, lines in chunks:
        Mstar = np.column_stack((a, b, c))
        axis, theta = nearest_zone_axis(
            Mstar, hmax=args.hmax, centering=centering
        )

        # angles to crowded axes
        M = np.linalg.inv(Mstar).T
        v_lab = (M @ crowd_axes.T).T  # shape (N,3)
        angles = np.degrees(
            np.arccos(
                np.clip(
                    np.abs(v_lab @ k_lab) / np.linalg.norm(v_lab, axis=1),
                    -1,
                    1,
                )
            )
        )
        scores = crowd_counts * np.exp(-((angles / theta0) ** 2))
        danger = scores.max()

        results.append((ev, axis, theta, lines, danger))

    if not args.no_sort:
        results.sort(key=lambda r: r[4], reverse=True)  # high danger first

    # -------- print table
    # print("\n{:<20} {:>7} {:>11}".format("Event", "θ (°)", "danger"))
    # print("-" * 40)
    # for ev, _, th, _, dg in results:
    #     print(f"{ev:<20} {th:7.3f} {dg:11.2f}")

    # -------- CSV
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                ["Event", "u", "v", "w", "theta_deg", "danger_score"]
            )
            for ev, (u, v, w), th, _, dg in results:
                wr.writerow([ev, u, v, w, f"{th:.6f}", f"{dg:.3f}"])
        print(f"[info] CSV written to {args.csv}", file=sys.stderr)

    # -------- sorted stream
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fh:
            fh.writelines(header)
            for _, _, _, lines, _ in results:
                fh.writelines(lines)
        print(
            f"[info] Sorted stream written to {args.sorted_stream}",
            file=sys.stderr,
        )

    # -------- plot
    if args.plot:
        if not _HAVE_MPL:
            print(
                "[warn] matplotlib not installed; skipping plot",
                file=sys.stderr,
            )
        else:
            danger_vals = np.array([r[4] for r in results])
            ranks = np.arange(1, len(danger_vals) + 1)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(ranks, danger_vals, s=20, alpha=0.8)
            ax.set_xlabel("Rank (danger sorted)")
            ax.set_ylabel("Danger score")
            ax.set_title("Combined crowdedness–angle danger metric")
            ax.grid(True, linestyle=":", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(args.plot)
            print(f"[info] Plot saved to {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
