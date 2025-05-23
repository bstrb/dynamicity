#!/usr/bin/env python3
"""
misorientation_to_zone_axis.py
--------------------------------
Parse a CrystFEL *.stream* file and, **for every indexed chunk**, find the
closest crystallographic zone axis [u v w] and the angular deviation θ
between that axis and the incident‑beam direction (laboratory +z).

**New in this version**
  • Output is **sorted by θ** (smallest first) so the most perfectly
    aligned shots appear at the top.
  • Optional `--plot FILE.png` produces a scatter/histogram figure for
    quick visual inspection.

USAGE
-----
python misorientation_to_zone_axis.py run.stream --hmax 4 \
       --csv results.csv --plot misorientation.png

Arguments
~~~~~~~~~
  streamfile          CrystFEL *.stream* file
  --hmax N            search zone axes with |h|,|k|,|l| ≤ N (default 3)
  --centering C       lattice centering symbol (P, I, F, A, B, C)
  --csv PATH          write a CSV (optional)
  --plot PATH         save a plot (PNG/PDF/...); omit to skip plotting
  --no-sort           keep original order instead of sorting by θ

The CSV has columns: Event, u, v, w, theta_deg
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from typing import List, Sequence, Tuple

import numpy as np

# Plotting imports are local so we can fall back gracefully if matplotlib
# is not present or the user does not request a plot.
try:
    import matplotlib
    matplotlib.use("Agg")  # headless back‑end
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ModuleNotFoundError:
    _HAVE_MPL = False

# ----------------------------------------------------------------------
# Regex helpers to read the stream file
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
                    continue  # body‑centred extinction rule
                if centering == "F" and (u % 2 + v % 2 + w % 2) % 2:
                    continue  # face‑centred rule
                if centering in {"A", "B", "C"}:
                    # e.g. A‑centred: h+l even; use simplistic test
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

def parse_stream(path: pathlib.Path) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (event_id, a*, b*, c*) from a CrystFEL stream"""
    out = []
    with path.open() as fh:
        in_chunk = False
        cur_event = None
        a = b = c = None
        for line in fh:
            if line.startswith("----- Begin chunk"):
                in_chunk = True
                cur_event = a = b = c = None
                continue
            if line.startswith("----- End chunk ----") and in_chunk:
                if cur_event and a is not None and b is not None and c is not None:
                    out.append((cur_event, a, b, c))
                in_chunk = False
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
    return out


# ----------------------------------------------------------------------
# Main CLI

def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Compute angular deviation of each indexed orientation to the nearest zone axis.",
    )
    p.add_argument("stream", type=pathlib.Path, help="CrystFEL .stream file")
    p.add_argument("--hmax", type=int, default=3, help="search |h|,|k|,|l| ≤ hmax (default 3)")
    p.add_argument("--centering", default="P", help="lattice centering symbol (P/I/F/A/B/C)")
    p.add_argument("--csv", type=pathlib.Path, help="write CSV output")
    p.add_argument("--plot", type=pathlib.Path, help="save a PNG/PDF plot of θ vs rank")
    p.add_argument("--no-sort", action="store_true", help="keep original order (do not sort)")

    args = p.parse_args(argv)

    # ------------------------------------------------------------------
    chunks = parse_stream(args.stream)
    if not chunks:
        sys.exit("[error] No orientation matrices found!")

    results: List[Tuple[str, Tuple[int, int, int], float]] = []

    for event, a, b, c in chunks:
        Mstar = np.column_stack((a, b, c))
        uvw, theta = nearest_zone_axis(Mstar, hmax=args.hmax, centering=args.centering)
        results.append((event, uvw, theta))

    if not args.no_sort:
        results.sort(key=lambda x: x[2])  # sort by theta

    # ------------------------------------------------------------------
    # Print nicely to stdout
    print(f"{'Event':<20}  {'[u v w]':<9}  θ (°)")
    print("-" * 40)
    for ev, (u, v, w), th in results:
        print(f"{ev:<20}  [{u:2d} {v:2d} {w:2d}]  {th:7.3f}")

    # CSV output
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["Event", "u", "v", "w", "theta_deg"])
            for ev, (u, v, w), th in results:
                wr.writerow([ev, u, v, w, f"{th:.6f}"])
        print(f"[info] CSV written to {args.csv}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Optional plot
    if args.plot:
        if not _HAVE_MPL:
            print("[warn] matplotlib not installed; skipping plot", file=sys.stderr)
        else:
            # Build arrays for plotting
            theta_vals = np.array([th for _, _, th in results])
            ranks = np.arange(1, len(results) + 1)
            uvw_labels = [f"[{u} {v} {w}]" for _, (u, v, w), _ in results]
            unique_axes = sorted(set(uvw_labels))
            cmap = plt.get_cmap("tab10", len(unique_axes))
            colors = {ax: cmap(i) for i, ax in enumerate(unique_axes)}

            fig, ax = plt.subplots(figsize=(6, 4))
            for ax_label in unique_axes:
                mask = [lbl == ax_label for lbl in uvw_labels]
                ax.scatter(ranks[mask], theta_vals[mask], label=ax_label, s=15, alpha=0.7, color=colors[ax_label])

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
