#!/usr/bin/env python3
"""
rank_vs_refs.py  –  v2.1  (2025-06-23)

• Computes dynamical risk M
• Finds closest reference orientation (mean angle of ±a*, ±b*, ±c*)
• Streams CSV row-by-row
• Optional: sorted .stream  (--sorted-stream)
• Optional: scatter plot     (--plot)
"""

from __future__ import annotations
import argparse, csv, itertools, math, re, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import gemmi
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# ───────── regex helpers ────────────────────────────────────────────────────
RE_CELL = re.compile(
    r'a\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?'
    r'al[^=]*=\s*([0-9.]+).*?be[^=]*=\s*([0-9.]+).*?ga[^=]*=\s*([0-9.]+)',
    re.S)
RE_WL     = re.compile(r'^\s*wavelength\s*=\s*([0-9.]+)\s*A', re.I | re.M)
RE_CHUNK  = re.compile(r'^----- Begin chunk')
RE_EVENT  = re.compile(r'^\s*Event:\s*(\S+)')
RE_ASTAR  = re.compile(r'^\s*astar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_BSTAR  = re.compile(r'^\s*bstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_CSTAR  = re.compile(r'^\s*cstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')

# ───────── helpers ──────────────────────────────────────────────────────────
def angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = abs(float(np.dot(u, v)) /
              (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))

def rms_orientation_diff(B: np.ndarray, R: np.ndarray) -> float:
    """Mean unsigned angle (deg) row-wise between two 3×3 reciprocal bases."""
    return float(np.mean([angle(B[i], R[i]) for i in range(3)]))

def load_refs(path: Path) -> List[np.ndarray]:
    refs=[]
    with path.open() as fh:
        for ln in fh:
            ln=ln.strip()
            if not ln or ln.startswith('#'): continue
            parts=ln.split()
            if len(parts)!=10:
                raise ValueError(f'Bad ref line:\n{ln}')
            refs.append(np.array(list(map(float, parts[1:]))).reshape(3,3))
    if not refs:
        raise ValueError('No references loaded.')
    return refs

def theta_phi_from_cstar(B: np.ndarray) -> Tuple[float,float]:
    """Polar (θ) and azimuth (φ) of c* (row 2)."""
    cx,cy,cz=B[2]
    r=math.sqrt(cx*cx+cy*cy+cz*cz)
    theta=math.degrees(math.acos(cz/r))
    phi=(math.degrees(math.atan2(cy,cx))+360)%360
    return theta,phi

# ───────── CLI ──────────────────────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser()
    p.add_argument('stream',type=Path)
    p.add_argument('-r','--resolution',type=float,required=True)
    p.add_argument('-t','--thickness',type=float,required=True)
    p.add_argument('--refs',type=Path,required=True,
                   help='file with reference orientations (see docs)')
    p.add_argument('-o','--csv',type=Path,
                   help='CSV path (default <stream>_pattern_risk.csv)')
    p.add_argument('--sorted-stream',type=Path)
    p.add_argument('--plot',type=Path)
    return p.parse_args()

# ───────── main ─────────────────────────────────────────────────────────────
def main():
    a=cli()
    refs=load_refs(a.refs)
    s_path=a.stream

    # --- header
    with s_path.open() as fh:
        header=[]
        for ln in fh:
            header.append(ln)
            if RE_CHUNK.match(ln): break
    hdr=''.join(header)
    m_cell,m_wl=RE_CELL.search(hdr),RE_WL.search(hdr)
    if not (m_cell and m_wl): sys.exit('header missing cell or λ')
    cell=gemmi.UnitCell(*(float(x) for x in m_cell.groups()))
    lam=float(m_wl.group(1))

    # --- HKL list
    hmax=int(round(cell.volume**(1/3)/a.resolution))+1
    HKL=np.array([(h,k,l) for h,k,l in itertools.product(range(-hmax,hmax+1),
                                                         repeat=3)
        if (h,k,l)!=(0,0,0) and cell.calculate_d((h,k,l))<a.resolution],int)
    klen=1/lam
    half=lam/(2*a.thickness*10.0)

    csv_path=a.csv or s_path.with_suffix('_pattern_risk.csv')
    sort_chunks:List[Tuple[int,str]]=[]
    M_vals=[]

    with csv_path.open('w',newline='') as fout, s_path.open() as fh:
        w=csv.writer(fout)
        w.writerow(['chunk_id','theta_deg','phi_deg','M','nearest_ref','ang_deg'])

        chunk=[]; chunk_id=None
        ast=bst=cst=None; M=0

        for ln in tqdm(fh,unit='lines',desc='scan'):
            if RE_CHUNK.match(ln):
                if chunk:
                    sort_chunks.append((M, ''.join(chunk)))
                    chunk.clear()
                chunk.append(ln)
                chunk_id=None; ast=bst=cst=None; M=0
                continue

            chunk.append(ln)

            if m:=RE_EVENT.match(ln): chunk_id=f'event:{m[1]}'; continue
            if m:=RE_ASTAR.match(ln): ast=list(map(float,m.groups())); continue
            if m:=RE_BSTAR.match(ln): bst=list(map(float,m.groups())); continue
            if m:=RE_CSTAR.match(ln): cst=list(map(float,m.groups())); continue

            if ast and bst and cst:
                B=np.array([ast,bst,cst])
                Rg=HKL@B; g2=(Rg**2).sum(1)
                s=(Rg[:,2]*klen-0.5*g2)/klen
                M=int((np.abs(s)<=half).sum())

                # orientation match
                dists=[rms_orientation_diff(B,R) for R in refs]
                idx=int(np.argmin(dists)); ang=float(dists[idx])

                theta,phi=theta_phi_from_cstar(B)

                w.writerow([chunk_id or 'unk',f'{theta:.2f}',f'{phi:.2f}',M,
                            f'ref{idx}',f'{ang:.2f}'])
                chunk.append(f'# M {M} ref ref{idx} ang {ang:.2f}\n')

                ast=bst=cst=None
                M_vals.append(M)

        if chunk:
            sort_chunks.append((M, ''.join(chunk)))

    print(f'[info] CSV → {csv_path}')

    if a.sorted_stream:
        with a.sorted_stream.open('w') as fout:
            fout.writelines(header)
            for _m,txt in sorted(sort_chunks,key=lambda t:t[0],reverse=True):
                fout.write(txt)
        print(f'[info] sorted stream → {a.sorted_stream}')

    if a.plot:
        if plt is None:
            print('[warn] matplotlib missing; cannot plot.')
        else:
            plt.figure(figsize=(6,4))
            plt.scatter(range(1,len(M_vals)+1),M_vals,s=8)
            plt.xlabel('pattern index'); plt.ylabel('M')
            plt.tight_layout(); plt.savefig(a.plot,dpi=160)
            print(f'[info] plot → {a.plot}')

if __name__=='__main__':
    main()
