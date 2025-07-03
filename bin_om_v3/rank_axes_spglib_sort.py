#!/usr/bin/env python3
"""
rank_axes_spglib_sort.py  – rank patterns by proximity to canonical zone axes
                               and optionally output a danger-sorted .stream

Dependencies :  numpy  spglib
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
UNI_RE   = re.compile(r"^unique_axis\s*=\s*(\S+)", re.I)

# ─── read header + orientation matrices + raw chunks ───────────────
def read_stream(path: pathlib.Path):
    lat = cent = uniq = None
    mats, chunks = {}, {}
    header_lines = []
    with path.open() as fh:
        ev = a = b = c = None
        buf = []
        in_chunk = False
        for ln in fh:
            if not in_chunk:
                header_lines.append(ln)
            if (m:=LAT_RE.match(ln)): lat  = m.group(1).lower()
            if (m:=CENT_RE.match(ln)): cent = m.group(1).upper()
            if (m:=UNI_RE.match(ln)):  uniq = m.group(1).lower()

            if ln.startswith("----- Begin chunk"):
                in_chunk = True
                buf = [ln]; ev=a=b=c=None
                continue
            if in_chunk:
                buf.append(ln)
            if ln.startswith("----- End chunk"):
                in_chunk=False
                if ev and a is not None and b is not None and c is not None:
                    mats[ev]   = np.column_stack((a,b,c))
                    chunks[ev] = buf.copy()
                continue

            # inside chunk
            if in_chunk:
                if (m:=EVENT_RE.search(ln)): ev=m.group(1)
                if (m:=VEC_RE.match(ln)):
                    v=np.array([float(m.group(i)) for i in (1,2,3)])
                    if ln.lstrip().startswith("astar"): a=v
                    elif ln.lstrip().startswith("bstar"): b=v
                    elif ln.lstrip().startswith("cstar"): c=v

    if not mats:
        sys.exit("no orientation matrices found")
    return lat, cent, uniq, header_lines, mats, chunks

# ─── full primitive-axis list (no symmetry merge) ───────────────────
@lru_cache(maxsize=None)
def axes_primitive(maxh:int):
    axes=[]
    for u in range(-maxh,maxh+1):
        for v in range(-maxh,maxh+1):
            for w in range(-maxh,maxh+1):
                if (u,v,w)==(0,0,0): continue
                if math.gcd(math.gcd(abs(u),abs(v)),abs(w))!=1: continue
                # canonical sign
                tup=(u,v,w)
                for c in tup:
                    if c!=0:
                        if c<0: tup=tuple(-x for x in tup)
                        break
                axes.append(tup)
    return sorted(set(axes))

# ─── misorientation to axis set ─────────────────────────────────────
def min_angle(Mstar, axes):
    invT = np.linalg.inv(Mstar).T
    k = np.array([0.,0.,1.])
    best=180.0
    for u,v,w in axes:
        v_lab = invT @ np.array([u,v,w],float)
        cosang = abs(v_lab@k)/np.linalg.norm(v_lab)
        best=min(best, math.degrees(math.acos(max(-1.,min(1.,cosang)))))
    return best

# ─── CLI driver ─────────────────────────────────────────────────────
def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("stream", type=pathlib.Path)
    ap.add_argument("--theta0", type=float, default=5.0)
    ap.add_argument("--maxh",  type=int,   default=2,
                    help="axes up to |h|≤maxh (1=>100/110/001)")
    ap.add_argument("--csv",   type=pathlib.Path)
    ap.add_argument("--top",   type=int,   default=40)
    ap.add_argument("--sorted-stream", type=pathlib.Path,
                    help="write a danger-sorted copy of the .stream")
    args=ap.parse_args(argv)

    lat,cent,uniq,header,mats,chunks = read_stream(args.stream)
    print(f"[info] patterns={len(mats)}  maxh={args.maxh}")

    axes = axes_primitive(args.maxh)
    print(f"[info] {len(axes)} primitive axes used")

    results=[]
    for ev,M in mats.items():
        dθ=min_angle(M,axes)
        score=math.exp(-(dθ/args.theta0)**2)
        results.append((ev,dθ,score))
    results.sort(key=lambda r:r[2], reverse=True)

    # ── console table ────────────────────────────────────────────
    print(f"\nRanked by danger (θ₀={args.theta0}°)")
    print("Event   θmin(°)  score"); print("─"*28)
    for ev,th,sc in results[:args.top]:
        print(f"{ev:<7} {th:7.2f} {sc:6.3f}")

    # ── CSV output ───────────────────────────────────────────────
    if args.csv:
        with args.csv.open("w",newline="") as fh:
            csv.writer(fh).writerows([("event","theta_deg","score"),*results])
        print(f"[csv] {args.csv}",file=sys.stderr)

    # ── sorted .stream copy ──────────────────────────────────────
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fh:
            fh.writelines(header)
            for ev,_,_ in results:
                fh.writelines(chunks[ev])
        print(f"[stream] written danger-sorted stream → {args.sorted_stream}",
              file=sys.stderr)

if __name__=="__main__":
    main()
