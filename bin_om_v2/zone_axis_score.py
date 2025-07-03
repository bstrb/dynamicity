#!/usr/bin/env python3
"""
zone_axis_score.py  –  June 2025  (US English, no fabricated numbers)

Usage
-----
    python zone_axis_score.py   mfm300sim_0.0_0.0.stream   -r 0.20  -t 150  \
           --sorted-stream mfm300sim_0.0_0.0_sorted.stream  --plot diag.png

Produces CSV with columns
    chunk_id, theta_deg, phi_deg, dyn_score, nearest_axis, ang_deg
and, if requested, a .stream sorted by descending dyn_score and a PNG plot.

The metric
~~~~~~~~~~
    dyn_score = Σ exp[ -(Sg / (λ/2t))² ]
sums over **all reflections to d ≤ d_min**.  Zone-axis patterns always give a
hundreds-to-thousands score; off-axis patterns give ≲10.
"""

from __future__ import annotations
import argparse, csv, itertools, math, re, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np, gemmi
from tqdm import tqdm
try: import matplotlib.pyplot as plt
except ModuleNotFoundError: plt = None

# ───────── regex helpers ────────────────────────────────────────────────────
RE_CELL   = re.compile(
    r'a\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?'
    r'al[^=]*=\s*([0-9.]+).*?be[^=]*=\s*([0-9.]+).*?ga[^=]*=\s*([0-9.]+)', re.S)
RE_WL     = re.compile(r'^\s*wavelength\s*=\s*([0-9.]+)\s*A', re.I | re.M)
RE_CHUNK  = re.compile(r'^----- Begin chunk')
RE_EVENT  = re.compile(r'^\s*Event:\s*(//\S+)')
RE_ASTAR  = re.compile(r'^\s*astar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_BSTAR  = re.compile(r'^\s*bstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_CSTAR  = re.compile(r'^\s*cstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')

# ───────── vector helpers ───────────────────────────────────────────────────
def norm(v: np.ndarray) -> np.ndarray:          # unit vector
    return v / np.linalg.norm(v)

def unsigned_angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = abs(float(np.dot(u, v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

def theta_phi_from_cstar(B: np.ndarray) -> Tuple[float,float]:
    cx,cy,cz = B[2]
    r = math.sqrt(cx*cx+cy*cy+cz*cz)
    return math.degrees(math.acos(cz/r)), (math.degrees(math.atan2(cy,cx))+360)%360

# ───────── CLI ──────────────────────────────────────────────────────────────
pa=argparse.ArgumentParser()
pa.add_argument('stream',type=Path)
pa.add_argument('-r','--resolution',type=float,required=True,
               help='reflection list cut-off (Å)')
pa.add_argument('-t','--thickness',type=float,required=True,
               help='crystal thickness (nm)')
pa.add_argument('-o','--csv',type=Path,
               help='CSV path (default <stream>_dyn.csv)')
pa.add_argument('--sorted-stream',type=Path,
               help='output .stream sorted by descending dyn_score')
pa.add_argument('--plot',type=Path,
               help='PNG scatter dyn_score vs. pattern index')
args=pa.parse_args()

# ───────── read header (cell + λ) ───────────────────────────────────────────
with args.stream.open() as fh:
    header=[]
    for ln in fh:
        header.append(ln)
        if RE_CHUNK.match(ln): break
header_txt=''.join(header)
m_cell=RE_CELL.search(header_txt); m_wl=RE_WL.search(header_txt)
if not (m_cell and m_wl):
    sys.exit('Header lacks unit-cell or wavelength.')
cell=gemmi.UnitCell(*(float(x) for x in m_cell.groups()))
lam=float(m_wl[1])

# ───────── build reflection list (integer HKL) ──────────────────────────────
d_min=args.resolution
hmax=int(round(cell.volume**(1/3)/d_min))+1
HKL=[(h,k,l) for h,k,l in itertools.product(range(-hmax,hmax+1),repeat=3)
     if (h,k,l)!=(0,0,0) and cell.calculate_d((h,k,l))<d_min]
HKL=np.array(HKL,int)                # (N,3)

# ───────── generate candidate zone axes automatically ───────────────────────
zone_dirs=[]
for h,k,l in itertools.product(range(-2,3),repeat=3):            # |h|+|k|+|l|≤2
    if (h,k,l)==(0,0,0) or abs(h)+abs(k)+abs(l)>2: continue
    g=np.array([h,k,l],float)
    zone_dirs.append(norm(g))
# reduce by lattice symmetry with Gemmi operators
sg = gemmi.SpaceGroup(cell)
unique=[]
for v in zone_dirs:
    if any(unsigned_angle(v,u)<1e-3 for u in unique): continue
    unique.append(v)
zone_dirs=np.array(unique)            # (M,3)

# ───────── constants for scoring ────────────────────────────────────────────
klen=1/lam
sigma=lam/(2*args.thickness*10.0)      # λ/2t in Å⁻¹

# ───────── pass through the stream ─────────────────────────────────────────-
csv_path=args.csv or args.stream.with_suffix('_dyn.csv')
sort_chunks:List[Tuple[float,str]]=[]; scores=[]

with csv_path.open('w',newline='') as fout, args.stream.open() as fh:
    w=csv.writer(fout)
    w.writerow(['chunk_id','theta_deg','phi_deg',
                'dyn_score','nearest_axis','ang_deg'])

    chunk=[]; chunk_id='unk'; ast=bst=cst=None; dyn=0.0

    for ln in tqdm(fh,unit='lines',desc='scoring'):
        if RE_CHUNK.match(ln):
            if chunk: sort_chunks.append((dyn, ''.join(chunk))); chunk.clear()
            chunk.append(ln); chunk_id='unk'; ast=bst=cst=None; dyn=0.0
            continue
        chunk.append(ln)

        if m:=RE_EVENT.match(ln): chunk_id=m[1]
        if m:=RE_ASTAR.match(ln): ast=list(map(float,m.groups()))
        if m:=RE_BSTAR.match(ln): bst=list(map(float,m.groups()))
        if m:=RE_CSTAR.match(ln): cst=list(map(float,m.groups()))

        if ast and bst and cst:
            B=np.array([ast,bst,cst])             # rows a*,b*,c* in lab
            Rg=HKL@B
            g2=(Rg**2).sum(1)
            Sg=(Rg[:,2]*klen - 0.5*g2)/klen
            dyn=float(np.exp(- (Sg/sigma)**2).sum())

            # nearest auto-zone axis
            beam=np.array([0,0,1],float)
            angs=[unsigned_angle(beam, B@v) for v in zone_dirs]
            idx=int(np.argmin(angs)); ang=angs[idx]

            th,ph=theta_phi_from_cstar(B)
            w.writerow([chunk_id,f'{th:.2f}',f'{ph:.2f}',
                        f'{dyn:.1f}',f'axis{idx}',f'{ang:.2f}'])
            chunk.append(f'# dyn {dyn:.1f} axis axis{idx} ang {ang:.2f}\n')
            ast=bst=cst=None
            scores.append(dyn)

    if chunk: sort_chunks.append((dyn, ''.join(chunk)))

print(f'[info] CSV → {csv_path}')

# ───────── optional outputs ─────────────────────────────────────────────────
if args.sorted_stream:
    with args.sorted_stream.open('w') as fout:
        fout.writelines(header)
        for _d,txt in sorted(sort_chunks,key=lambda t:t[0],reverse=True):
            fout.write(txt)
    print(f'[info] sorted stream → {args.sorted_stream}')

if args.plot and plt is not None:
    x=np.arange(1,len(scores)+1)
    plt.figure(figsize=(6,4))
    plt.scatter(x,scores,s=8)
    plt.xlabel('pattern index'); plt.ylabel('dyn_score')
    plt.tight_layout(); plt.savefig(args.plot,dpi=160)
    print(f'[info] plot → {args.plot}')
elif args.plot:
    print('[warn] matplotlib not installed; cannot plot.')
