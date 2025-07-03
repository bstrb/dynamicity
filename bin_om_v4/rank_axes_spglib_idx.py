#!/usr/bin/env python3
"""
rank_axes_spglib_sort.py
─────────────────────────
Danger score = exp[-(θ/θ0)^2] · exp[α·(|u|+|v|+|w|−1)]

α (index-boost) and θ0 are CLI flags.
Creates optional danger-sorted .stream copy.
"""

from __future__ import annotations
import argparse, math, pathlib, re, csv, sys
from functools import lru_cache
import numpy as np, spglib as spg

# ─── regex helpers ───────────────────────────────────────────────────
VEC_RE   = re.compile(r"[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)")
EVENT_RE = re.compile(r"//(\d+)\b")
LAT_RE   = re.compile(r"^lattice_type\s*=\s*(\S+)", re.I)
CENT_RE  = re.compile(r"^centering\s*=\s*(\S+)",   re.I)

# ─── read stream (header + orientation matrices + raw chunks) ───────
def read_stream(path: pathlib.Path):
    lat = cent = None
    mats, chunks = {}, {}
    header = []
    with path.open() as fh:
        in_chunk, buf = False, []
        ev = a = b = c = None
        for ln in fh:
            if not in_chunk: header.append(ln)
            if (m := LAT_RE.match(ln)):  lat  = m.group(1).lower()
            if (m := CENT_RE.match(ln)): cent = m.group(1).upper()

            if ln.startswith("----- Begin chunk"):
                in_chunk, buf, ev = True, [ln], None
                a = b = c = None
                continue
            if in_chunk: buf.append(ln)
            if ln.startswith("----- End chunk"):
                in_chunk = False
                if ev and a is not None and b is not None and c is not None:
                    mats[ev]   = np.column_stack((a, b, c))
                    chunks[ev] = buf.copy()
                continue
            if in_chunk:
                if (m := EVENT_RE.search(ln)): ev = m.group(1)
                if (m := VEC_RE.match(ln)):
                    vec = np.array([float(m.group(i)) for i in (1, 2, 3)])
                    if ln.lstrip().startswith("astar"): a = vec
                    elif ln.lstrip().startswith("bstar"): b = vec
                    elif ln.lstrip().startswith("cstar"): c = vec
    if not mats:
        sys.exit("no orientation matrices found")
    return lat, cent, header, mats, chunks

# ─── primitive axis list up to |h|≤maxh  (no symmetry merge) ────────
@lru_cache(maxsize=None)
def axes_primitive(maxh: int):
    axes = []
    for u in range(-maxh, maxh + 1):
        for v in range(-maxh, maxh + 1):
            for w in range(-maxh, maxh + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                if math.gcd(math.gcd(abs(u), abs(v)), abs(w)) != 1:
                    continue
                tup = (u, v, w)
                # canonical sign (first non-zero positive)
                for c in tup:
                    if c != 0:
                        if c < 0:
                            tup = tuple(-x for x in tup)
                        break
                axes.append(tup)
    return sorted(set(axes))

# ─── best angle & axis for one pattern ───────────────────────────────
def best_axis_angle(Mstar, axes):
    invT = np.linalg.inv(Mstar).T
    k = np.array([0., 0., 1.])
    best = 180.0; best_axis = (0, 0, 1)
    for u, v, w in axes:
        v_lab = invT @ np.array([u, v, w], float)
        cang  = abs(v_lab @ k) / np.linalg.norm(v_lab)
        theta = math.degrees(math.acos(max(-1., min(1., cang))))
        if theta < best:
            best, best_axis = theta, (u, v, w)
    return best, best_axis

# ─── CLI ─────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("stream", type=pathlib.Path)
    p.add_argument("--theta0", type=float, default=5.0,
                   help="Gaussian half-width θ₀ (deg)")
    p.add_argument("--alpha",  type=float, default=0.40,
                   help="index-boost factor α")
    p.add_argument("--maxh",   type=int,   default=3,
                   help="include axes with |h|≤maxh")
    p.add_argument("--top",    type=int,   default=40)
    p.add_argument("--csv",            type=pathlib.Path)
    p.add_argument("--sorted-stream",  type=pathlib.Path)
    args = p.parse_args(argv)

    lat, cent, header, mats, chunks = read_stream(args.stream)
    axes = axes_primitive(args.maxh)
    print(f"[info] patterns={len(mats)}  α={args.alpha}  θ0={args.theta0}°  "
          f"maxh={args.maxh}  axes={len(axes)}")

    results = []
    for ev, M in mats.items():
        θ, axis = best_axis_angle(M, axes)
        order   = abs(axis[0]) + abs(axis[1]) + abs(axis[2])
        w_idx   = math.exp(args.alpha * (order - 1))              # ★ new
        score   = w_idx * math.exp(- (θ / args.theta0) ** 2)      # ★ new
        results.append((ev, θ, score))

    results.sort(key=lambda r: r[2], reverse=True)

    # ─ print table ─
    if args.top:
        print("\nEvent   θmin(°)  score"); print("─" * 28)
        for ev, th, sc in results[:args.top]:
            print(f"{ev:<7} {th:7.2f} {sc:6.3f}")

    # ─ CSV ─
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            csv.writer(fh).writerows([("event", "theta_deg", "score"), *results])
        print(f"[csv] {args.csv}", file=sys.stderr)

    # ─ danger-sorted stream ─
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fh:
            fh.writelines(header)
            for ev, _, _ in results:
                fh.writelines(chunks[ev])
        print(f"[stream] → {args.sorted_stream}", file=sys.stderr)

if __name__ == "__main__":
    main()
