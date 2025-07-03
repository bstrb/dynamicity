#!/usr/bin/env python3
"""
rank_tI_axes.py  – rank CrystFEL patterns by proximity to the
canonical zone axes of an I-centred tetragonal lattice.

Axes considered
---------------
    [001]                ← c axis
    [100] [010]          ← a / b
    <110> family         ← two-zone axis (4 directions)

score = exp[-(θ / θ0)²]     (θ0 default 5°)
"""

from __future__ import annotations
import argparse, math, pathlib, re, csv, sys
import numpy as np

# ─── regex helpers ──────────────────────────────────────────────────────
VEC_RE   = re.compile(r"[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                      r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)")
EVENT_RE = re.compile(r"//(\d+)\b")

# AXES = [
#     (0,0,1),
#     (1,0,0),(0,1,0),
#     (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0)
# ]
# ------- axes used for scoring ------------------------------------
AXES = [
    (0, 0, 1),                  # [001]
    (1, 0, 0), (0, 1, 0),       # [100] / [010]
    (1, 1, 0), (-1, 1, 0),      # <110> family (4 dirs)
    (1,-1,0), (-1,-1,0),

    # ---- add any “maybe-dangerous” directions below ---------------
    (1, 0, 1), (0, 1, 1),       # <101>
    (-1,0,1), (0,-1,1), (1,0,-1), (0,1,-1),
    (1, 1, 1), (-1, 1, 1),      # <111>
    (2, 1, 0), (1, 2, 0),       # <210>  (optional)
]

K = np.array([0.,0.,1.])

def parse_stream(path: pathlib.Path):
    mats = {}
    with path.open() as fh:
        ev = a = b = c = None
        for ln in fh:
            if (m:=EVENT_RE.search(ln)):
                ev = m.group(1); a=b=c=None
            if (m:=VEC_RE.match(ln)):
                v = np.array([float(m.group(i)) for i in (1,2,3)])
                if   ln.lstrip().startswith("astar"): a=v
                elif ln.lstrip().startswith("bstar"): b=v
                elif ln.lstrip().startswith("cstar"): c=v
            if ev and a is not None and b is not None and c is not None:
                mats[ev] = np.column_stack((a,b,c)); ev=None
    if not mats: sys.exit("no matrices found")
    return mats

def min_angle(M, axes=AXES):
    invT = np.linalg.inv(M).T
    best = 180.0
    for u,v,w in axes:
        v_lab = invT @ np.array([u,v,w],float)
        cosang = abs(v_lab@K)/np.linalg.norm(v_lab)
        theta  = math.degrees(math.acos(max(-1.0,min(1.0,cosang))))
        best   = min(best,theta)
    return best

def cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", type=pathlib.Path)
    ap.add_argument("--theta0", type=float, default=5.0)
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args(argv)

    mats = parse_stream(args.stream)
    results=[]
    for ev,M in mats.items():
        dθ=min_angle(M); score=math.exp(-(dθ/args.theta0)**2)
        results.append((ev,dθ,score))
    results.sort(key=lambda r:r[2], reverse=True)

    print(f"Ranked by zone-axis danger  (θ₀={args.theta0}°)")
    print("Event   θmin(°)  score"); print("─"*28)
    for ev,th,sc in results[:args.top]:
        print(f"{ev:<7} {th:7.2f} {sc:6.3f}")

    if args.csv:
        with args.csv.open("w",newline="") as fh:
            csv.writer(fh).writerows([("event","theta_deg","score"),*results])
        print(f"[csv] {args.csv}",file=sys.stderr)

if __name__=="__main__":
    cli()
