#!/usr/bin/env python3
"""
rank_zone_axes.py  –  robust version
Lists the pure zone-axis directions with the highest many-beam dynamical score.
"""

from __future__ import annotations
import argparse, itertools, math, re, sys
from pathlib import Path
import numpy as np, gemmi

RE_WL   = re.compile(r'^\s*wavelength\s*=\s*([0-9.]+)\s*A', re.I|re.M)
RE_CELL = re.compile(r'a\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?'
                     r'al[^=]*=\s*([0-9.]+).*?be[^=]*=\s*([0-9.]+).*?ga[^=]*=\s*([0-9.]+)',
                     re.S|re.I)

def recip_matrix(cell: gemmi.UnitCell)->np.ndarray:
    a,b,c = cell.a,cell.b,cell.c
    al,be,ga = map(math.radians,(cell.alpha,cell.beta,cell.gamma))
    ax,ay,az = a,0,0
    bx,by,bz = b*math.cos(ga), b*math.sin(ga), 0
    cx = c*math.cos(be)
    cy = c*(math.cos(al)-math.cos(be)*math.cos(ga))/math.sin(ga)
    cz = math.sqrt(max(c*c-cx*cx-cy*cy,0))
    aV=np.array([ax,ay,az]); bV=np.array([bx,by,bz]); cV=np.array([cx,cy,cz])
    V=np.dot(aV,np.cross(bV,cV))
    return np.stack([np.cross(bV,cV)/V,
                     np.cross(cV,aV)/V,
                     np.cross(aV,bV)/V])      # rows a*,b*,c*

def gcd3(a:int,b:int,c:int)->int:
    from math import gcd; return gcd(abs(a),gcd(abs(b),abs(c)))

def enumerate_axes(limit:int)->list[tuple[int,int,int]]:
    cand=[]
    for h,k,l in itertools.product(range(-limit,limit+1),repeat=3):
        if (h,k,l)==(0,0,0) or abs(h)+abs(k)+abs(l)>limit: continue
        if gcd3(h,k,l)!=1: continue
        cand.append((h,k,l))
    return cand

def weight(Sg: np.ndarray, sigma: float)->float:
    return float(np.exp(-(Sg/sigma)**2).sum())

def theta_phi(v:np.ndarray)->tuple[float,float]:
    x,y,z=v; r=np.linalg.norm(v)
    return math.degrees(math.acos(z/r)), (math.degrees(math.atan2(y,x))+360)%360

ap=argparse.ArgumentParser()
ap.add_argument('header',type=Path)
ap.add_argument('-r','--resolution',type=float,required=True)
ap.add_argument('-t','--thickness',type=float,required=True)
ap.add_argument('--limit',type=int,default=4,help='max |h|+|k|+|l| (default 4)')
ap.add_argument('--top',type=int,default=12)
a=ap.parse_args()

hdr=a.header.read_text()
m_wl,m_cell=RE_WL.search(hdr),RE_CELL.search(hdr)
if not(m_wl and m_cell): sys.exit('header parse error')
lam=float(m_wl[1])
cell=gemmi.UnitCell(*map(float,m_cell.groups()))

# reflection list
B=recip_matrix(cell)
hmax=int(round(cell.volume**(1/3)/a.resolution))+1
g=[]
for h,k,l in itertools.product(range(-hmax,hmax+1),repeat=3):
    if (h,k,l)==(0,0,0): continue
    if cell.calculate_d((h,k,l))<a.resolution:
        g.append(B@np.array([h,k,l],float))
g=np.array(g)
klen=1/lam
sigma=lam/(2*a.thickness*10.0)

results=[]
for h,k,l in enumerate_axes(a.limit):
    beam= B@np.array([h,k,l],float)
    beam/=np.linalg.norm(beam)          # unit vector in lab coords
    Sg=(g@beam -0.5*np.linalg.norm(g,axis=1)**2)/klen
    dyn=weight(Sg,sigma)
    θ,φ=theta_phi(beam)
    results.append((dyn,(h,k,l),θ,φ))
results.sort(key=lambda x:x[0],reverse=True)

print(f'{"h k l":>8}  {"θ°":>6}  {"φ°":>7}  {"dyn":>7}')
print('-'*35)
for dyn,(h,k,l),θ,φ in results[:a.top]:
    print(f'[{h:2d} {k:2d} {l:2d}] {θ:6.1f} {φ:7.1f} {dyn:7.1f}')
