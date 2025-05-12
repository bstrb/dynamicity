#!/usr/bin/env python3
"""
parse_crystfel_stream.py

Lightweight parser for CrystFEL .stream files (format ≥ 2.3).

Extracts and exposes in Python:
• Global header:
    – wavelength (Å)
    – camera length (clen, m)
    – detector resolution (res, px⁻¹ m)
    – input unit cell: lattice_type, unique_axis, centering, a,b,c,α,β,γ
• Per-frame data (keyed by Event number):
    – actual cell parameters: a,b,c,α,β,γ
    – reciprocal vectors: astar, bstar, cstar
    – peak-search list: fs_px, ss_px, intensity
    – indexed reflections: h,k,l, I, sigma_I, peak, background, fs_px, ss_px

Usage in Python:
    from parse_crystfel_stream import StreamParser
    parser = StreamParser('data.stream')
    parser.parse()
    # then work with parser.header and parser.frames

Minimal CLI: prints summary only.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Header:
    wavelength: Optional[float] = None
    clen: Optional[float] = None
    res: Optional[float] = None
    lattice_type: Optional[str] = None
    unique_axis: Optional[str] = None
    centering: Optional[str] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    al: Optional[float] = None
    be: Optional[float] = None
    ga: Optional[float] = None

@dataclass
class Peak:
    fs_px: float
    ss_px: float
    intensity: float

@dataclass
class Reflection:
    h: int
    k: int
    l: int
    I: float
    sigma_I: float
    peak: float
    background: float
    fs_px: float
    ss_px: float

@dataclass
class Frame:
    event: str
    cell: Dict[str, float]
    astar: List[float]
    bstar: List[float]
    cstar: List[float]
    peaks: List[Peak] = field(default_factory=list)
    reflections: List[Reflection] = field(default_factory=list)

class StreamParser:
    _FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.header: Header = Header()
        self.frames: List[Frame] = []

    def parse(self) -> None:
        """Parse the .stream file into header and frames."""
        lines = self.filepath.read_text().splitlines()
        i, n = 0, len(lines)
        # parse global header until first chunk
        while i < n and "----- Begin chunk -----" not in lines[i]:
            self._parse_header_line(lines[i]); i += 1
        # parse each chunk
        while i < n:
            if "----- Begin chunk -----" not in lines[i]:
                i += 1; continue
            i += 1
            chunk = []
            while i < n and "----- End chunk -----" not in lines[i]:
                chunk.append(lines[i]); i += 1
            self.frames.append(self._parse_chunk(chunk)); i += 1

    def _parse_header_line(self, line: str) -> None:
        # simple regex matches
        import re
        if m := re.match(fr"^\s*wavelength\s*=\s*({self._FLOAT})\s*A", line):
            self.header.wavelength = float(m.group(1))
        elif m := re.match(fr"^\s*clen\s*=\s*({self._FLOAT})\s*m", line):
            self.header.clen = float(m.group(1))
        elif m := re.match(fr"^\s*res\s*=\s*({self._FLOAT})", line):
            self.header.res = float(m.group(1))
        elif m := re.match(r"^\s*lattice_type\s*=\s*(\S+)", line):
            self.header.lattice_type = m.group(1)
        elif m := re.match(r"^\s*unique_axis\s*=\s*(\S+)", line):
            self.header.unique_axis = m.group(1)
        elif m := re.match(r"^\s*centering\s*=\s*(\S+)", line):
            self.header.centering = m.group(1)
        elif m := re.match(fr"^\s*a\s*=\s*({self._FLOAT})\s*A", line):
            self.header.a = float(m.group(1))
        elif m := re.match(fr"^\s*b\s*=\s*({self._FLOAT})\s*A", line):
            self.header.b = float(m.group(1))
        elif m := re.match(fr"^\s*c\s*=\s*({self._FLOAT})\s*A", line):
            self.header.c = float(m.group(1))
        elif m := re.match(fr"^\s*al\s*=\s*({self._FLOAT})\s*deg", line):
            self.header.al = float(m.group(1))
        elif m := re.match(fr"^\s*be\s*=\s*({self._FLOAT})\s*deg", line):
            self.header.be = float(m.group(1))
        elif m := re.match(fr"^\s*ga\s*=\s*({self._FLOAT})\s*deg", line):
            self.header.ga = float(m.group(1))

    def _parse_chunk(self, chunk: List[str]) -> Frame:
        import re
        event = ""
        cell: Dict[str, float] = {}
        astar = bstar = cstar = []
        peaks: List[Peak] = []
        refls: List[Reflection] = []
        # compile patterns
        ev_re = re.compile(r"^\s*Event:\s*(\S+)")
        cell_re = re.compile(
            fr"^\s*Cell parameters\s+({self._FLOAT})\s+({self._FLOAT})\s+({self._FLOAT})\s+nm,"
            fr"\s+({self._FLOAT})\s+({self._FLOAT})\s+({self._FLOAT})\s+deg"
        )
        vec_re = re.compile(r"^\s*([abc])star\s*=\s*([\s\d\.eE+\-]+)\s*nm\^-1")
        peak_re = re.compile(fr"^\s*({self._FLOAT})\s+({self._FLOAT})\s+{self._FLOAT}\s+({self._FLOAT})")
        refl_re = re.compile(
            fr"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+({self._FLOAT})\s+({self._FLOAT})"
            fr"\s+({self._FLOAT})\s+({self._FLOAT})\s+({self._FLOAT})\s+({self._FLOAT})"
        )
        mode = None
        for line in chunk:
            if m := ev_re.match(line): event = m.group(1)
            elif m := cell_re.match(line):
                cell = dict(a=float(m.group(1)), b=float(m.group(2)), c=float(m.group(3)),
                            al=float(m.group(4)), be=float(m.group(5)), ga=float(m.group(6)))
            elif m := vec_re.match(line):
                vec = [float(x) for x in m.group(2).split()]
                if m.group(1)=='a': astar=vec
                elif m.group(1)=='b': bstar=vec
                else: cstar=vec
            elif "Peaks from peak search" in line: mode='peaks'
            elif "End of peak list" in line: mode=None
            elif "Reflections measured after indexing" in line: mode='refl'
            elif "End of reflections" in line: mode=None
            elif mode=='peaks' and (m:=peak_re.match(line)):
                peaks.append(Peak(float(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif mode=='refl' and (m:=refl_re.match(line)):
                h,k,l = map(int, m.group(1,2,3))
                I, sig, pk, bg, fs, ss = map(float, m.group(4,5,6,7,8,9))
                refls.append(Reflection(h,k,l,I,sig,pk,bg,fs,ss))
        return Frame(event, cell, astar, bstar, cstar, peaks, refls)

# Minimal CLI
if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser(description='Parse CrystFEL .stream')
    ap.add_argument('stream', help='Path to .stream file')
    args = ap.parse_args()
    parser = StreamParser(args.stream)
    parser.parse()
    hdr = parser.header
    print(f'Parsed header: wavelength={hdr.wavelength} Å, clen={hdr.clen} m, res={hdr.res} px/m')
    print(f'Parsed frames: {len(parser.frames)}')
