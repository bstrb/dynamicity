#!/usr/bin/env python3
"""
rank_axes_spglib_optc.py
────────────────────────
Danger score uses BOTH mis-orientation angle (θ) and axis “order” k:

      k = |u| + |v| + |w|
      score = exp( - [ (θ/θ0)^2 + (k/k0)^2 ] )

θ0  (deg)  controls how far from an axis is still risky;
k0  sets how strongly high-index axes are penalised
      (k0 ≈ 3 → order-3 axis scores ≈ exp[-1] if θ ≪ θ0).

Outputs:
  • console table (top-N rows)
  • full CSV
  • danger-sorted *.stream* copy

Dependencies:  numpy   spglib   (pip install numpy spglib)
"""

from __future__ import annotations
import argparse, math, pathlib, re, csv, sys
from functools import lru_cache
import numpy as np; import spglib as spg

# ─── helpers ────────────────────────────────────────────────────────
VEC_RE = re.compile(r"[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
                    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)")
EVENT_RE = re.compile(r"//(\d+)\b")

def read_stream(path: pathlib.Path):
    """Return header, {event:M*}, {event:chunk_lines}."""
    header, mats, chunks = [], {}, {}
    with path.open() as fh:
        in_chunk = False; buf=[]
        ev=a=b=c=None
        for ln in fh:
            if not in_chunk: header.append(ln)
            if ln.startswith("----- Begin chunk"):
                in_chunk=True; buf=[ln]; ev=a=b=c=None; continue
            if in_chunk: buf.append(ln)
            if ln.startswith("----- End chunk"):
                in_chunk=False
                if ev and a is not None and b is not None and c is not None:
                    mats[ev]=np.column_stack((a,b,c)); chunks[ev]=buf.copy()
                continue
            if in_chunk:
                if (m:=EVENT_RE.search(ln)): ev=m.group(1)
                if (m:=VEC_RE.match(ln)):
                    vec=np.array([float(m.group(i)) for i in (1,2,3)])
                    if ln.lstrip().startswith("astar"): a=vec
                    elif ln.lstrip().startswith("bstar"): b=vec
                    elif ln.lstrip().startswith("cstar"): c=vec
    if not mats: sys.exit("no orientation matrices found")
    return header, mats, chunks

@lru_cache(maxsize=None)
def axes_primitive(maxh:int):
    axes=[]
    for u in range(-maxh,maxh+1):
        for v in range(-maxh,maxh+1):
            for w in range(-maxh,maxh+1):
                if (u,v,w)==(0,0,0): continue
                if math.gcd(math.gcd(abs(u),abs(v)),abs(w))!=1: continue
                tup=(u,v,w)
                for c in tup:
                    if c!=0:
                        if c<0: tup=tuple(-x for x in tup)
                        break
                axes.append(tup)
    return sorted(set(axes))

def best_axis_angle(Mstar, axes):
    invT = np.linalg.inv(Mstar).T; kvec = np.array([0.,0.,1.])
    best=180.; best_axis=(0,0,1)
    for ax in axes:
        v_lab = invT @ np.array(ax,float)
        theta = math.degrees(math.acos(
                 max(-1.,min(1., abs(v_lab@kvec)/np.linalg.norm(v_lab)))))
        if theta<best: best, best_axis = theta, ax
    return best, best_axis

# ─── CLI ────────────────────────────────────────────────────────────
def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("stream",type=pathlib.Path)
    p.add_argument("--theta0", type=float, default=5.0, help="angle width (deg)")
    p.add_argument("--k0",    type=float, default=3.0, help="index width k0")
    p.add_argument("--maxh",  type=int,   default=3,   help="axis depth")
    p.add_argument("--top",   type=int,   default=40,  help="rows to print")
    p.add_argument("--csv",   type=pathlib.Path)
    p.add_argument("--sorted-stream", type=pathlib.Path)
    args=p.parse_args(argv)

    header, mats, chunks = read_stream(args.stream)
    axes = axes_primitive(args.maxh)
    print(f"[info] patterns={len(mats)}  θ0={args.theta0}°  k0={args.k0}  "
          f"maxh={args.maxh}  axes={len(axes)}")

    results=[]
    for ev,M in mats.items():
        θ, axis = best_axis_angle(M, axes)
        k = abs(axis[0])+abs(axis[1])+abs(axis[2])
        score = math.exp(-((θ/args.theta0)**2 + (k/args.k0)**2))  # ← Option C
        results.append((ev,θ,k,score))
    results.sort(key=lambda r:r[3], reverse=True)

    # console table
    if args.top:
        print("\nEvent   θ(°)  k  score"); print("─"*30)
        for ev,th,k,sc in results[:args.top]:
            print(f"{ev:<7} {th:5.2f} {k:2d} {sc:6.3f}")

    # CSV
    if args.csv:
        with args.csv.open("w",newline="") as fh:
            csv.writer(fh).writerows(
                [("event","theta_deg","order_k","score"),*results])
        print(f"[csv] {args.csv}",file=sys.stderr)

    # sorted stream
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fh:
            fh.writelines(header)
            for ev,_,_,_ in results:
                fh.writelines(chunks[ev])
        print(f"[stream] → {args.sorted_stream}",file=sys.stderr)

if __name__=="__main__":
    main()
