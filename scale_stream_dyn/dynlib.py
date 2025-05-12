#!/usr/bin/env python3
"""
dynlib.py – crystallographic helpers for dynamical scaling
=========================================================

• Stream parser / writer  (compatible with CrystFEL .stream format)
• Unit-cell + space-group utilities with cctbx **or** pure-numpy fallback
• Resolution & Ewald-sphere maths, D-spacing, s = 1/d, etc.
"""
from __future__ import annotations
import os, re, math, warnings
from collections import defaultdict, namedtuple
from typing import List, Dict, Tuple, Iterable

import numpy as np
from tqdm import tqdm

try:
    # full tool-chain
    from cctbx import crystal, miller, sgtbx
    from cctbx.array_family import flex
except ImportError:
    crystal = miller = sgtbx = None
    warnings.warn("cctbx not found – falling back to minimal symmetry handling")

###############################################################################
# 1.  Reflection data structure & stream parser                               #
###############################################################################
Reflection = namedtuple(
    "Reflection",
    "h k l I sigma peak bkg fs ss panel red flag"
)

FLAG_LOW_REDUNDANCY   = 0x01
FLAG_DYN_OUTLIER      = 0x02           # new bit

RE_BEGIN  = re.compile(r"^----- Begin chunk -----")
RE_END    = re.compile(r"^----- End chunk -----")
RE_EVENT  = re.compile(r"^\s*Event:\s*(.*)")
RE_REF    = re.compile(
    r"^\s*([\-\d]+)\s+([\-\d]+)\s+([\-\d]+)\s+"        # h k l
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"                # I sigma
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"                # peak bkg
    r"([\d\.eE\-]+)\s+([\d\.eE\-]+)\s+"                # fs ss
    r"(\S+)"                                           # panel
)

class Chunk:
    __slots__ = (
        "header", "footer", "reflections",
        "event",  "scale",  "good",
        "_core_log_median",
        "R_sysAbs", "R_Friedel", "p90_log_spread"
    )
    def __init__(self):
        self.header = []
        self.footer = []
        self.reflections = []
        self.event = "unknown"
        self.scale = 1.0
        self.good = True
        # initialize the new metrics so they’re always defined
        self._core_log_median = None
        self.R_sysAbs   = 0.0
        self.R_Friedel  = 0.0
        self.p90_log_spread = 0.0

def parse_stream(path: str) -> Tuple[str, List[Chunk]]:
    """Return (global_header, [chunks])."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # global header (before first 'Begin chunk')
    header_lines, i = [], 0
    while i < len(lines) and not RE_BEGIN.match(lines[i]):
        header_lines.append(lines[i]); i += 1
    gheader = "".join(header_lines)

    chunks: List[Chunk] = []
    while i < len(lines):
        if RE_BEGIN.match(lines[i]):
            ch = Chunk(); i += 1
            # chunk header
            while i < len(lines) and not RE_REF.match(lines[i]):
                ch.header.append(lines[i])
                if (m := RE_EVENT.match(lines[i])):
                    ch.event = m.group(1).strip()
                i += 1
            # reflections
            while i < len(lines) and RE_REF.match(lines[i]):
                m = RE_REF.match(lines[i])
                h, k, l   = map(int, m.group(1, 2, 3))
                I, sig    = map(float, m.group(4, 5))
                peak, bkg = map(float, m.group(6, 7))
                fs, ss    = map(float, m.group(8, 9))
                panel     = m.group(10)
                ch.reflections.append(
                    Reflection(h, k, l, I, sig, peak, bkg, fs, ss, panel, 1, 0)
                )
                i += 1
            # footer
            while i < len(lines) and not RE_END.match(lines[i]):
                ch.footer.append(lines[i]); i += 1
            i += 1       # skip 'End chunk'
            chunks.append(ch)
        else:
            i += 1
    return gheader, chunks

def write_stream(gheader: str,
                 chunks : Iterable[Chunk],
                 path   : str,
                 include_flags: bool = False) -> None:
    """Emit a scaled .stream file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(gheader)
        for ch in chunks:
            fh.write("----- Begin chunk -----\n")
            fh.writelines(ch.header)
            for r in ch.reflections:
                fh.write(
                    f" {r.h:4d} {r.k:4d} {r.l:4d}"
                    f" {r.I:12.2f} {r.sigma:9.2f}"
                    f" {r.peak:10.2f} {r.bkg:10.2f}"
                    f" {r.fs:7.1f} {r.ss:7.1f} {r.panel}"
                )
                if include_flags:
                    fh.write(f"  ;FLAGS={r.flag:02x}")
                fh.write("\n")
            fh.writelines(ch.footer)
            fh.write("----- End chunk -----\n")

###############################################################################
# 2.  Unit cell / symmetry helpers                                            #
###############################################################################
RE_FLOAT = r"([\d\.]+)"
RE_CELL  = {
    "a":  re.compile(fr"^a\s*=\s*{RE_FLOAT}\s*A"),
    "b":  re.compile(fr"^b\s*=\s*{RE_FLOAT}\s*A"),
    "c":  re.compile(fr"^c\s*=\s*{RE_FLOAT}\s*A"),
    "al": re.compile(fr"^al\s*=\s*{RE_FLOAT}\s*deg"),
    "be": re.compile(fr"^be\s*=\s*{RE_FLOAT}\s*deg"),
    "ga": re.compile(fr"^ga\s*=\s*{RE_FLOAT}\s*deg"),
}
RE_SG = re.compile(r"^space_group\s*=\s*(\S+)")

def extract_symmetry(gheader: str,
                     sg_override: str | None = None):
    """Return (unit_cell_tuple, space_group_symbol)  or (None, None)."""
    sg = sg_override or (RE_SG.search(gheader).group(1)
                         if RE_SG.search(gheader) else None)
    if not sg:
        warnings.warn("space_group symbol not found – symmetry tests disabled")
        return None, None
    vals = {k: None for k in RE_CELL}
    for ln in gheader.splitlines():
        for key, pat in RE_CELL.items():
            if (m := pat.match(ln)):
                vals[key] = float(m.group(1))
    if None in vals.values():
        warnings.warn("incomplete unit-cell parameters – symmetry tests disabled")
        return None, None
    return (vals['a'], vals['b'], vals['c'],
            vals['al'], vals['be'], vals['ga']), sg

###############################################################################
# 3.  Resolution, allowed/forbidden, Friedel-mate stuff                       #
###############################################################################

def d_spacing(h, k, l, a, b, c, al, be, ga) -> float:
    """
    Return d-spacing in Å.

    • If cctbx is available we delegate to uctbx.unit_cell.d().
    • Otherwise we fall back to the orthorhombic P-cell formula.
    """
    if crystal:                          # full cctbx installed
        # uctbx is part of cctbx – import lazily so dynlib keeps zero imports
        from cctbx.uctbx import unit_cell as _uctbx_unit_cell
        uc = _uctbx_unit_cell((a, b, c, al, be, ga))
        return uc.d((h, k, l))           # ← 3-line fix
    # ------------------------------------------------------------------ #
    # fallback (orthorhombic approximation – angles assumed 90°)          #
    return 1.0 / math.sqrt(
        (h ** 2) / a ** 2 +
        (k ** 2) / b ** 2 +
        (l ** 2) / c ** 2
    )
# --------------------------------------------------------------------------- #

def s_in_Ainv(d: float) -> float:
    """s = 2 sinθ / λ   but for scaling we usually use 1/d because λ≪d."""
    return 1.0 / d          # ≈ 2sinθ/λ    for electrons λ ≈ 0.02 Å

# sys_absence helpers (needs cctbx)
def is_forbidden(h,k,l, space_group_symbol:str) -> bool:
    if not sgtbx:                       # no symmetry library
        return False
    sg = sgtbx.space_group_info(space_group_symbol).group()
    return sg.is_sys_absent((h,k,l))

###############################################################################
# 4.  DQE / MTF curve loader (csv: "s,I_gain")                                #
###############################################################################
def load_dqe_table(path:str|None):
    """Return interpolator function  D(s)  (→ gain correction factor)."""
    if not path:
        return lambda s: 1.0
    arr = np.loadtxt(path, delimiter=",")
    s_vals, gain = arr[:,0], arr[:,1]
    def D(s:float):
        if s <= s_vals[0]:  return gain[0]
        if s >= s_vals[-1]: return gain[-1]
        i = np.searchsorted(s_vals, s) - 1
        t = (s - s_vals[i]) / (s_vals[i+1]-s_vals[i])
        return gain[i]*(1-t) + gain[i+1]*t
    return D
