#!/usr/bin/env python3
"""
orientation_scoring_stream.py  –  vectorised & robust
"""
from __future__ import annotations
import argparse, itertools, math, re, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import gemmi
try:
    from tqdm import tqdm                    # nice progress bar
except ImportError:                          # fallback if tqdm not installed
    tqdm = lambda x, **k: x

# ---------- stream-parsing helpers (unchanged) --------------------------------
CELL_RE   = re.compile(r'^\s*([abc])\s*=\s*([0-9.]+)\s*A', re.I)
ANGLE_RE  = re.compile(r'^\s*([abg]l?)\s*=\s*([0-9.]+)\s*deg', re.I)
TYPE_RE   = re.compile(r'^\s*lattice_type\s*=\s*(\w+)', re.I)
CENTER_RE = re.compile(r'^\s*centering\s*=\s*([A-Z])', re.I)
UNIAX_RE  = re.compile(r'^\s*unique_axis\s*=\s*([abc])', re.I)

def parse_unit_cell_from_stream(path: Path) -> Tuple[gemmi.UnitCell, gemmi.SpaceGroup]:
    p = {k: None for k in
         ('a','b','c','al','be','ga','lattice_type','centering','unique_axis')}
    with path.open(encoding='utf-8', errors='ignore') as fh:
        in_blk = False
        for line in fh:
            if line.startswith('----- Begin unit cell -----'):
                in_blk = True; continue
            if line.startswith('----- End unit cell -----'):
                break
            if not in_blk: continue
            if m:=CELL_RE.match(line):   p[m[1]] = float(m[2]); continue
            if m:=ANGLE_RE.match(line):
                key={'al':'al','alpha':'al','be':'be','beta':'be',
                     'ga':'ga','gamma':'ga'}[m[1].lower()[:2]]
                p[key]=float(m[2]);continue
            if m:=TYPE_RE.match(line):   p['lattice_type']=m[1].lower();continue
            if m:=CENTER_RE.match(line): p['centering']=m[1].upper();continue
            if m:=UNIAX_RE.match(line):  p['unique_axis']=m[1].lower();continue

    if None in (p['a'],p['b'],p['c']):
        sys.exit('[error] stream lacks unit-cell lengths')

    cell = gemmi.UnitCell(p['a'], p['b'], p['c'], p['al'] or 90, p['be'] or 90, p['ga'] or 90)
    sg   = gemmi.SpaceGroup(infer_sg(p['lattice_type'], p['centering'], p['unique_axis']))
    return cell, sg

def infer_sg(lat:str|None, cen:str|None, ux:str|None)->str:
    if not lat or not cen: return 'P 1'
    lat, cen = lat.lower(), cen.upper()
    if lat=='cubic':       return 'I m -3 m' if cen=='I' else 'P m -3 m'
    if lat=='tetragonal':  return 'I 4/mmm'  if cen=='I' else 'P 4/mmm'
    if lat=='orthorhombic':return {'P':'P m m m','C':'C m m m',
                                   'F':'F m m m','I':'I m m m'}.get(cen,'P m m m')
    if lat=='hexagonal':   return 'P 6/mmm'
    if lat=='monoclinic':  return 'C 1 2/m 1' if (cen=='C'and(ux or'b')=='b') else 'P 1 2/m 1'
    return 'P 1'

# ---------- reciprocal-lattice utilities --------------------------------------
def recip_matrix(cell: gemmi.UnitCell)->np.ndarray:
    a,b,c,al,be,ga = cell.a,cell.b,cell.c, math.radians(cell.alpha),\
                     math.radians(cell.beta), math.radians(cell.gamma)
    ax,ay,az = a,0,0
    bx,by,bz = b*math.cos(ga), b*math.sin(ga), 0
    cx = c*math.cos(be)
    cy = c*(math.cos(al)-math.cos(be)*math.cos(ga))/math.sin(ga)
    cz = math.sqrt(max(c*c - cx*cx - cy*cy, 0))
    aV,bV,cV = np.array([ax,ay,az]), np.array([bx,by,bz]), np.array([cx,cy,cz])
    V   = np.dot(aV, np.cross(bV,cV))
    ast = np.cross(bV,cV)/V; bst = np.cross(cV,aV)/V; cst = np.cross(aV,bV)/V
    return np.stack([ast,bst,cst])          # rows = a*,b*,c*

def reflection_table(cell: gemmi.UnitCell, d_min: float)->np.ndarray:
    B = recip_matrix(cell)
    hmax = int(round(cell.volume**(1/3)/d_min))+1
    g = []
    for h,k,l in itertools.product(range(-hmax,hmax+1),
                                   range(-hmax,hmax+1),
                                   range(-hmax,hmax+1)):
        if h==k==l==0: continue
        if cell.calculate_d((h,k,l)) < d_min:
            g.append(B @ np.array([h,k,l],float))
    return np.stack(g)                      # shape (N,3)

# ---------- orientation scoring (vectorised) ----------------------------------
def rotation(phi,theta):
    rphi = math.radians(phi); rth = math.radians(theta)
    rz = np.array([[math.cos(rphi),-math.sin(rphi),0],
                   [math.sin(rphi), math.cos(rphi),0],
                   [0,0,1]])
    ry = np.array([[ math.cos(rth),0,math.sin(rth)],
                   [0,1,0],
                   [-math.sin(rth),0,math.cos(rth)]])
    return rz@ry

def score(g: np.ndarray, lambda_A: float, t_nm: float,
          step: float)->List[Tuple[float,float,int]]:
    g2   = (g**2).sum(1)                 # |g|²
    t_A  = t_nm*10.0
    half = lambda_A/(2*t_A)
    klen = 1.0/lambda_A
    kvec = np.array([0,0,klen])
    results=[]
    theta_grid = np.arange(0,90+1e-6,step)
    phi_grid   = np.arange(0,360,step)
    for theta in tqdm(theta_grid,desc='θ sweep',unit='°'):
        for phi in phi_grid:
            Rg = (rotation(phi,theta) @ g.T).T
            s  = (Rg[:,2]*klen - 0.5*g2)/klen
            M  = int((np.abs(s)<=half).sum())
            results.append((theta,phi,M))
    return sorted(results,key=lambda x:x[2],reverse=True)

# ---------- CLI / main --------------------------------------------------------
def cli():
    p=argparse.ArgumentParser(description='Rank orientations (stream input)')
    p.add_argument('input',type=Path)
    p.add_argument('-r','--resolution',type=float,required=True)
    p.add_argument('-t','--thickness',type=float,required=True)
    p.add_argument('-s','--step',type=float,default=2.0)
    p.add_argument('-l','--wavelength',type=float,default=0.02508)
    p.add_argument('-n','--top',type=int,default=50)
    return p.parse_args()

def main():
    a=cli()
    cell,_=parse_unit_cell_from_stream(a.input)
    G = reflection_table(cell,a.resolution)
    ranks = score(G,a.wavelength,a.thickness,a.step)
    print(f"{'theta':>8}{'phi':>10}{'M':>8}\n"+'-'*28)
    for th,ph,m in ranks[:a.top]:
        print(f"{th:8.2f}{ph:10.2f}{m:8d}")

if __name__=='__main__': main()
