#!/usr/bin/env python3
"""
find_bad_axes.py  –  robust version
Ranks low-index zone-axis directions by a Gaussian dynamical score.

Run e.g.:
    python find_bad_axes.py  mydata.stream \
           --resolution 0.8  --thickness 150  --top 10
"""

from __future__ import annotations
import argparse, itertools, math, re, sys
from pathlib import Path

import numpy as np, gemmi

# ── regex helpers ───────────────────────────────────────────────────────────
RE_WL   = re.compile(r'\bwavelength\s*=\s*([0-9.]+)\s*A', re.I)
RE_CELL = re.compile(
    r'\ba\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?'
    r'al[^=]*=\s*([0-9.]+).*?be[^=]*=\s*([0-9.]+).*?ga[^=]*=\s*([0-9.]+)',
    re.S | re.I)

# ── helpers ─────────────────────────────────────────────────────────────────
def recip_matrix(cell: gemmi.UnitCell) -> np.ndarray:
    a,b,c = cell.a,cell.b,cell.c
    al,be,ga = map(math.radians,(cell.alpha,cell.beta,cell.gamma))
    ax,ay,az = a,0,0
    bx,by,bz = b*math.cos(ga), b*math.sin(ga),0
    cx = c*math.cos(be)
    cy = c*(math.cos(al)-math.cos(be)*math.cos(ga))/math.sin(ga)
    cz = math.sqrt(max(c*c-cx*cx-cy*cy,0))
    aV=np.array([ax,ay,az]); bV=np.array([bx,by,bz]); cV=np.array([cx,cy,cz])
    V=np.dot(aV,np.cross(bV,cV))
    return np.stack([np.cross(bV,cV)/V,
                     np.cross(cV,aV)/V,
                     np.cross(aV,bV)/V])

def gcd3(a:int,b:int,c:int)->int:
    from math import gcd; return gcd(abs(a),gcd(abs(b),abs(c)))

def unique_by_angle(vecs:list[np.ndarray], tol_deg:float=1.0)->list[np.ndarray]:
    uniq=[]
    for v in vecs:
        vn=v/np.linalg.norm(v)
        keep=True
        for u in uniq:
            cos=abs(float(np.dot(vn,u)))
            cos=min(1.0,max(-1.0,cos))        # clip to [-1,1]
            if math.degrees(math.acos(cos)) < tol_deg:
                keep=False; break
        if keep: uniq.append(vn)
    return uniq

def theta_phi(v:np.ndarray)->tuple[float,float]:
    x,y,z=v; r=np.linalg.norm(v)
    return math.degrees(math.acos(z/r)), (math.degrees(math.atan2(y,x))+360)%360

# ── CLI ─────────────────────────────────────────────────────────────────────
ap=argparse.ArgumentParser()
ap.add_argument('stream',type=Path)
ap.add_argument('-r','--resolution',type=float,required=True)
ap.add_argument('-t','--thickness',type=float,required=True)
ap.add_argument('--limit',type=int,default=4,
               help='max |h|+|k|+|l| (default 4)')
ap.add_argument('--top',type=int,default=12)
args=ap.parse_args()

# ── read only the header part of the stream ────────────────────────────────
header_lines=[]
with args.stream.open() as fh:
    for ln in fh:
        header_lines.append(ln)
        if '----- End unit cell -----' in ln:
            break
header=''.join(header_lines)

m_wl  = RE_WL.search(header)
m_cell= RE_CELL.search(header)
if not(m_wl and m_cell):
    sys.exit('Failed to parse wavelength or unit cell from the stream header.')
lam=float(m_wl[1])
cell=gemmi.UnitCell(*map(float,m_cell.groups()))
B=recip_matrix(cell)

# ── reflection list up to d_min ────────────────────────────────────────────
hmax=int(round(cell.volume**(1/3)/args.resolution))+1
gvec=[]
for h,k,l in itertools.product(range(-hmax,hmax+1),repeat=3):
    if (h,k,l)==(0,0,0): continue
    if cell.calculate_d((h,k,l))<args.resolution:
        gvec.append(B@np.array([h,k,l],float))
gvec=np.array(gvec)
klen=1/lam
sigma=lam/(2*args.thickness*10.0)

# ── candidate zone axes ────────────────────────────────────────────────────
cand=[]
for h,k,l in itertools.product(range(-args.limit,args.limit+1),repeat=3):
    if (h,k,l)==(0,0,0) or abs(h)+abs(k)+abs(l)>args.limit: continue
    if gcd3(h,k,l)!=1: continue
    cand.append(np.array([h,k,l],float))
cand=unique_by_angle(cand)

# ── score each axis ────────────────────────────────────────────────────────
rows=[]
for v_int in cand:
    beam=B@v_int; beam/=np.linalg.norm(beam)
    Sg=(gvec@beam -0.5*np.linalg.norm(gvec,axis=1)**2)/klen
    score=float(np.exp(-(Sg/sigma)**2).sum())
    θ,φ=theta_phi(beam)
    h,k,l=(int(round(x)) for x in v_int)
    rows.append((score,(h,k,l),θ,φ))
rows.sort(key=lambda x:x[0],reverse=True)

# ── print table ────────────────────────────────────────────────────────────
print(f'{"h k l":>8}  {"θ°":>6}  {"φ°":>7}  score')
print('-'*35)
for sc,(h,k,l),θ,φ in rows[:args.top]:
    print(f'[{h:2d} {k:2d} {l:2d}] {θ:6.1f} {φ:7.1f} {sc:7.1f}')
