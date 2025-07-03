#!/usr/bin/env python3
"""
rank_patterns_by_orientation.py  (June-2025)
--------------------------------------------

© 2025 Buster Blomberg  –  Public-domain helper

• Reads a CrystFEL *.stream* file
• Computes the “dynamical-risk” metric M for each pattern
• Writes pattern_risk.csv on the fly
• Optionally writes a second stream sorted by M
• Optionally plots M vs. pattern index

Usage
~~~~~
    python rank_patterns_by_orientation.py RUN.stream \
           -r 1.0 -t 150 \
           --sorted-stream RUN_sorted.stream \
           --plot RUN_M_vs_index.png
"""

from __future__ import annotations
import argparse, csv, itertools, math, re, sys
from pathlib import Path
from typing import List

import numpy as np
import gemmi
from tqdm import tqdm

# Optional plotting – only import if user asks for --plot
try:
    import matplotlib.pyplot as plt            # noqa: F401
except ModuleNotFoundError:
    plt = None                                 # handled later

# ───────────────────────── regex helpers ──────────────────────────
RE_CELL = re.compile(
    r'a\s*=\s*([0-9.]+).*?b\s*=\s*([0-9.]+).*?c\s*=\s*([0-9.]+).*?'
    r'al[^=]*=\s*([0-9.]+).*?be[^=]*=\s*([0-9.]+).*?ga[^=]*=\s*([0-9.]+)',
    re.S)

RE_WL     = re.compile(r'^\s*wavelength\s*=\s*([0-9.]+)\s*A', re.I | re.M)
RE_CHUNK  = re.compile(r'^----- Begin chunk')
RE_ASTAR  = re.compile(r'^\s*astar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_BSTAR  = re.compile(r'^\s*bstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_CSTAR  = re.compile(r'^\s*cstar\s*=\s*([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)')
RE_EVENT  = re.compile(r'^\s*Event:\s*(\S+)')

# ───────────────────────── lattice helpers ───────────────────────
def recip_matrix(cell: gemmi.UnitCell) -> np.ndarray:
    a,b,c = cell.a, cell.b, cell.c
    al,be,ga = map(math.radians, (cell.alpha, cell.beta, cell.gamma))
    ax,ay,az = a, 0, 0
    bx,by,bz = b*math.cos(ga), b*math.sin(ga), 0
    cx = c*math.cos(be)
    cy = c*(math.cos(al) - math.cos(be)*math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c*c - cx*cx - cy*cy, 0))
    aV=np.array([ax,ay,az]); bV=np.array([bx,by,bz]); cV=np.array([cx,cy,cz])
    V = np.dot(aV, np.cross(bV, cV))
    return np.stack([np.cross(bV, cV)/V,
                     np.cross(cV, aV)/V,
                     np.cross(aV, bV)/V])

def theta_phi_from_cstar(Brows: np.ndarray):
    cx,cy,cz = Brows[2]
    r = math.sqrt(cx*cx+cy*cy+cz*cz)
    return math.degrees(math.acos(cz/r)), (math.degrees(math.atan2(cy,cx))+360)%360

# ───────────────────────── CLI ───────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument('stream', type=Path)
    p.add_argument('-r','--resolution', type=float, required=True,
                   help='d_min for reflection list (Å)')
    p.add_argument('-t','--thickness', type=float, required=True,
                   help='crystal thickness (nm)')
    p.add_argument('-o','--csv', type=Path,
                   help='CSV path (default <stream>_pattern_risk.csv)')
    p.add_argument('--sorted-stream', type=Path,
                   help='write a second .stream re-ordered by descending M')
    p.add_argument('--plot', type=Path,
                   help='PNG path for M vs. pattern-index scatter')
    return p.parse_args()

# ───────────────────────── main ──────────────────────────────────
def main():
    args = cli()
    s_path: Path = args.stream

    # ---------- header: cell + wavelength ------------------------
    with s_path.open() as fh:
        header=[]
        for ln in fh:
            header.append(ln)
            if RE_CHUNK.match(ln):
                break
    hdr=''.join(header)

    m_cell = RE_CELL.search(hdr)
    if not m_cell:
        sys.exit('Unit-cell block not found.')
    cell = gemmi.UnitCell(*(float(x) for x in m_cell.groups()))

    m_wl = RE_WL.search(hdr)
    if not m_wl:
        sys.exit('wavelength line not found.')
    lam = float(m_wl.group(1))

    # ---------- reflection HKL table -----------------------------
    d_min = args.resolution
    hmax = int(round(cell.volume ** (1/3) / d_min)) + 1
    hkl_table = np.array(
        [(h,k,l) for h,k,l in itertools.product(range(-hmax,hmax+1), repeat=3)
                  if (h,k,l)!=(0,0,0) and cell.calculate_d((h,k,l)) < d_min],
        dtype=int)

    klen = 1.0 / lam
    half = lam / (2 * args.thickness * 10.0)

    # ---------- output paths -------------------------------------
    csv_path = args.csv or s_path.with_suffix('_pattern_risk.csv')
    sort_chunks: List[tuple[int,str]] = []          # (M, chunk_text)
    M_values: List[int] = []                        # for plotting

    # ---------- scan over patterns -------------------------------
    with csv_path.open('w', newline='') as csv_out, s_path.open() as fh:
        writer = csv.writer(csv_out)
        writer.writerow(['chunk_id','theta_deg','phi_deg','M'])

        chunk_lines = []          # accumulate raw lines for sorting
        chunk_id = None
        ast=bst=cst=None
        M = 0                     # current M value (for chunk header)

        for line in tqdm(fh, unit='lines', desc='scan'):
            if RE_CHUNK.match(line):
                # flush previous chunk to sorted list (if any)
                if chunk_lines:
                    sort_chunks.append((M, ''.join(chunk_lines)))
                    chunk_lines.clear()
                chunk_lines.append(line)
                chunk_id = None
                ast=bst=cst=None
                continue

            chunk_lines.append(line)

            if m:=RE_EVENT.match(line):
                chunk_id = f'event:{m.group(1)}'
                continue
            if m:=RE_ASTAR.match(line): ast=list(map(float,m.groups())); continue
            if m:=RE_BSTAR.match(line): bst=list(map(float,m.groups())); continue
            if m:=RE_CSTAR.match(line): cst=list(map(float,m.groups())); continue

            if ast and bst and cst:
                B = np.array([ast, bst, cst])
                theta, phi = theta_phi_from_cstar(B)

                Rg = hkl_table @ B
                g2 = (Rg**2).sum(1)
                s  = (Rg[:,2]*klen - 0.5*g2) / klen
                M  = int((np.abs(s) <= half).sum())
                M_values.append(M)

                writer.writerow([chunk_id or 'unk',
                                 f'{theta:.2f}', f'{phi:.2f}', M])

                # inject the M line into the chunk so the sorted stream keeps it
                chunk_lines.append(f'# M_score {M}\n')
                ast=bst=cst=None

        # push last chunk
        if chunk_lines:
            sort_chunks.append((M, ''.join(chunk_lines)))

    print(f'[info] CSV written → {csv_path}')

    # ---------- optional: write sorted stream --------------------
    if args.sorted_stream:
        with args.sorted_stream.open('w') as fout:
            # header lines first (unchanged order):
            fout.writelines(header)
            for _M, txt in sorted(sort_chunks, key=lambda t: t[0], reverse=True):
                fout.write(txt)
        print(f'[info] sorted stream → {args.sorted_stream}')

    # ---------- optional: plot M vs. index -----------------------
    if args.plot:
        if plt is None:
            print('[warn] matplotlib missing; cannot plot.')
        else:
            x = np.arange(1, len(M_values)+1)
            plt.figure(figsize=(6,4))
            plt.scatter(x, M_values, s=8)
            plt.xlabel('Pattern index (encounter order)')
            plt.ylabel('M  (beams in relrod)')
            plt.tight_layout()
            plt.savefig(args.plot, dpi=160)
            print(f'[info] plot saved → {args.plot}')

if __name__ == '__main__':
    main()
