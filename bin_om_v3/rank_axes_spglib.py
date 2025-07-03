#!/usr/bin/env python3
"""
rank_axes_spglib.py  – lattice-agnostic zone-axis ranking
---------------------------------------------------------
See original docstring for usage.  This version works with any
spglib ≥ 1.16 (tuple API) **and** ≥ 2.3 (dict API).
"""

from __future__ import annotations
import argparse, math, pathlib, re, csv, sys
from functools import lru_cache

import numpy as np
import spglib as spg


# ───── regex helpers ───────────────────────────────────────────────
VEC_RE   = re.compile(r"[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)")
EVENT_RE = re.compile(r"//(\d+)\b")
LAT_RE   = re.compile(r"^lattice_type\s*=\s*(\S+)", re.I)
CENT_RE  = re.compile(r"^centering\s*=\s*(\S+)",   re.I)
UNI_RE   = re.compile(r"^unique_axis\s*=\s*(\S+)", re.I)


# ───── read stream header + matrices ───────────────────────────────
def read_stream(path: pathlib.Path):
    lat = cent = uniq = None
    mats = {}
    with path.open() as fh:
        ev = a = b = c = None
        for ln in fh:
            if (m := LAT_RE.match(ln)): lat  = m.group(1).lower()
            if (m := CENT_RE.match(ln)): cent = m.group(1).upper()
            if (m := UNI_RE.match(ln)): uniq  = m.group(1).lower()

            if (m := EVENT_RE.search(ln)):
                ev = m.group(1); a = b = c = None
            if (m := VEC_RE.match(ln)):
                v = np.array([float(m.group(i)) for i in (1, 2, 3)])
                if   ln.lstrip().startswith("astar"): a = v
                elif ln.lstrip().startswith("bstar"): b = v
                elif ln.lstrip().startswith("cstar"): c = v
            if ev and a is not None and b is not None and c is not None:
                mats[ev] = np.column_stack((a, b, c))
                ev = None
    if not mats:
        sys.exit("no orientation matrices found")
    return lat, cent, uniq, mats

# ─────––  SAFE, MERGE-FREE AXIS LIST  (replace the old axes_for_bravais) ─────
@lru_cache(maxsize=None)
def axes_for_bravais(lat_type: str, centering: str,
                     maxh: int = 1) -> list[tuple[int, int, int]]:
    """
    Return *all* primitive (u v w) with |u|,|v|,|w| ≤ maxh
    (gcd(u,v,w)=1) and canonical sign (first non-zero component > 0).

    No symmetry reduction ⇒ never drops legitimate zone axes.
    """
    axes = []
    for u in range(-maxh, maxh + 1):
        for v in range(-maxh, maxh + 1):
            for w in range(-maxh, maxh + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                if math.gcd(math.gcd(abs(u), abs(v)), abs(w)) != 1:
                    continue                    # keep only primitive
                # canonical sign: first non-zero positive
                g = 1
                tup = (u, v, w)
                for comp in tup:
                    if comp != 0:
                        if comp < 0:
                            tup = tuple(-x for x in tup)
                        break
                axes.append(tup)
    return sorted(set(axes))


# ───── misorientation to axis set ──────────────────────────────────
def min_angle(Mstar, axes):
    invT = np.linalg.inv(Mstar).T
    k = np.array([0., 0., 1.])
    best = 180.0
    for u, v, w in axes:
        v_lab = invT @ np.array([u, v, w], float)
        cang = abs(v_lab @ k) / np.linalg.norm(v_lab)
        best = min(best, math.degrees(math.acos(max(-1., min(1., cang)))))
    return best


# ───── CLI driver ─────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", type=pathlib.Path)
    ap.add_argument("--theta0", type=float, default=5.0,
                    help="Gaussian width θ₀ (deg)")
    ap.add_argument("--maxh", type=int, default=2,
                    help="axes up to |h|≤maxh (1→100/110/001 only)")
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)

    lat, cent, uniq, mats = read_stream(args.stream)
    print(f"[info] lattice_type={lat}  centering={cent} "
          f"patterns={len(mats)}  maxh={args.maxh}")

    axes = axes_for_bravais(lat or "a", cent or "P", args.maxh)
    print(f"[info] {len(axes)} symmetry-distinct axes: "
          f"{', '.join(map(str, axes))}")

    results = []
    for ev, M in mats.items():
        dθ = min_angle(M, axes)
        score = math.exp(-(dθ / args.theta0) ** 2)
        results.append((ev, dθ, score))
    results.sort(key=lambda r: r[2], reverse=True)

    print(f"\nRanked by danger  (θ₀={args.theta0}°)")
    print("Event   θmin(°)  score"); print("─" * 28)
    for ev, th, sc in results[:args.top]:
        print(f"{ev:<7} {th:7.2f} {sc:6.3f}")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            csv.writer(fh).writerows([("event", "theta_deg", "score"), *results])
        print(f"[csv] {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
