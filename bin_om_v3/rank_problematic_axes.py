#!/usr/bin/env python3
"""
rank_problematic_axes.py
------------------------

Rank diffraction patterns by proximity to crystallographically
important zone axes, **without any manual reference list**.

Dependencies
------------
    pip install numpy spglib

Usage
-----
    python rank_problematic_axes.py run.stream \
           --theta0 5 --csv ranked.csv --top 40

Options
-------
    --theta0  DEG   Gaussian width defining how far off-axis is still
                    considered “dangerous” (default 5°).
    --top     N     Print only the top-N rows to stdout (CSV always full).
"""

from __future__ import annotations
import argparse, math, pathlib, re, csv, sys
from functools import lru_cache

import numpy as np
import spglib as spg            # symmetry library

# ────────────────────────────────────────────────────────────────────────────
#  1.   REGEXES FOR QUICK STREAM PARSING
# ────────────────────────────────────────────────────────────────────────────
VEC_RE   = re.compile(r"^[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)")
EVENT_RE = re.compile(r"//(\d+)\b")
LAT_RE   = re.compile(r"^lattice_type\s*=\s*(\S+)", re.I)
CENT_RE  = re.compile(r"^centering\s*=\s*(\S+)",   re.I)
UNI_RE   = re.compile(r"^unique_axis\s*=\s*(\S+)", re.I)

# ────────────────────────────────────────────────────────────────────────────
#  2.   STREAM READER
# ────────────────────────────────────────────────────────────────────────────
def read_stream(path: pathlib.Path):
    """Return (lattice_type, centering, unique_axis, {event: M*})."""
    lat = cent = uniq = None
    mats = {}

    with path.open() as fh:
        ev = a = b = c = None
        header_done = False
        for ln in fh:
            if not header_done:
                if   (m := LAT_RE.match(ln)):  lat  = m.group(1).lower()
                elif (m := CENT_RE.match(ln)): cent = m.group(1).upper()
                elif (m := UNI_RE.match(ln)):  uniq = m.group(1).lower()
                if ln.startswith("----- Begin chunk"):
                    header_done = True

            if (m := EVENT_RE.search(ln)):
                ev = m.group(1)
                a = b = c = None

            if (m := VEC_RE.match(ln)):
                vec = np.array([float(m.group(i)) for i in (1,2,3)])
                if   ln.startswith("astar"): a = vec
                elif ln.startswith("bstar"): b = vec
                elif ln.startswith("cstar"): c = vec

            # when we have a full orientation matrix, store it
            if ev and a is not None and b is not None and c is not None:
                mats[ev] = np.column_stack((a, b, c))
                ev = None        # reset until next //event

    if not mats:
        sys.exit("[error] No orientation matrices found in file.")

    return lat, cent, uniq, mats


# ────────────────────────────────────────────────────────────────────────────
#  3.   CANONICAL AXIS GENERATOR (SPGLIB)
# ────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def canonical_axes(lat_type: str, centering: str, uniq_axis: str | None,
                   max_index: int = 1):
    """
    Enumerate all integer axes ( ±u,±v,±w with |u|,|v|,|w| ≤ max_index )
    and merge those that are symmetry-equivalent for the given Bravais
    lattice.  Return a list of primitive (u,v,w) tuples, each representing
    ONE symmetry class.
    """
    # build a dummy conventional cell; only symmetry matters
    # (use simple cubic ~10 Å box, then let spglib attach the requested lattice)
    a = 10.0
    lattice = np.eye(3) * a
    positions = [[0,0,0]]
    numbers = [1]
    cell = (lattice, positions, numbers)

    dataset = spg.get_symmetry_dataset(cell,
                                       symprec=1e-3,
                                       hall_number=None,
                                       return_no_match=False,
                                       symbolic_operations=False,
                                       bravais_type=lat_type + centering,
                                       unique_axis=uniq_axis or "")
    rot_mats = dataset["rotations"]

    # generate candidate axes
    cand = []
    for u in range(-max_index, max_index+1):
        for v in range(-max_index, max_index+1):
            for w in range(-max_index, max_index+1):
                if (u,v,w) == (0,0,0): continue
                cand.append(np.array([u,v,w], int))

    # canonical representative = primitive vector with first non-zero > 0
    def canon(v):
        g = math.gcd(math.gcd(abs(v[0]), abs(v[1])), abs(v[2]))
        v = v // g
        for i in range(3):
            if v[i] != 0:
                if v[i] < 0: v = -v
                break
        return tuple(int(x) for x in v)

    groups = {}
    for v in cand:
        reps = [canon(R @ v) for R in rot_mats]
        key  = min(reps)
        groups.setdefault(key, [])
        groups[key].append(tuple(v))

    return sorted(groups.keys())


# ────────────────────────────────────────────────────────────────────────────
#  4.   MIS-ORIENTATION TO A SET OF AXES
# ────────────────────────────────────────────────────────────────────────────
def min_angle_to_axes(Mstar: np.ndarray, axes: list[tuple[int,int,int]]):
    invT = np.linalg.inv(Mstar).T
    k = np.array([0.,0.,1.])
    best = 180.0
    for u,v,w in axes:
        v_lab = invT @ np.array([u,v,w], float)
        cosang = abs(v_lab @ k) / np.linalg.norm(v_lab)
        theta  = math.degrees(math.acos(max(-1.0,min(1.0,cosang))))
        best   = min(best, theta)
    return best


# ────────────────────────────────────────────────────────────────────────────
#  5.   MAIN CLI
# ────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rank patterns by proximity to canonical zone axes "
                    "for the detected Bravais lattice."
    )
    ap.add_argument("stream", type=pathlib.Path)
    ap.add_argument("--theta0", type=float, default=5.0,
                    help="Gaussian width θ₀ in danger score (deg)")
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=20,
                    help="print only the TOP rows (all are written to CSV)")
    args = ap.parse_args(argv)

    lat, cent, uniq, mats = read_stream(args.stream)
    print(f"[info] lattice_type={lat or 'unknown'}  centering={cent or '?'} "
          f"unique_axis={uniq or '?'}  patterns={len(mats)}")

    axes = canonical_axes(lat or "t", cent or "I", uniq)
    print(f"[info] using {len(axes)} symmetry-reduced axes:"
          f" {', '.join(str(a) for a in axes)}")

    results = []
    for ev, M in mats.items():
        dth  = min_angle_to_axes(M, axes)
        score = math.exp(-(dth/args.theta0)**2)
        results.append((ev, dth, score))

    results.sort(key=lambda r: r[2], reverse=True)

    # ── console output ────────────────────────────────────────────────
    hdr = f"\nRanked by danger   (θ₀ = {args.theta0}°)\n"
    print(hdr + "Event      θmin(°)   score\n" + "─"*30)
    for ev, th, sc in results[:args.top]:
        print(f"{ev:<9} {th:8.2f} {sc:8.3f}")

    # ── CSV ───────────────────────────────────────────────────────────
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["event", "theta_deg", "score"])
            wr.writerows(results)
        print(f"[info] CSV written → {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    import math
    main()
