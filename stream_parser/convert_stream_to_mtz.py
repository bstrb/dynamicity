#!/usr/bin/env python3
"""
convert_stream_to_mtz.py
------------------------
CrystFEL .stream  ➜  MTZ chunks (duplicates OK).

• Parses only the first 5 columns (H K L I SIGI).
• Writes one MTZ per N frames, preserving duplicate HKLs.
• Uses low-level iotbx.mtz API so no dedup happens.

2025-06-27
"""

from __future__ import annotations
import argparse, itertools, re, sys
from pathlib import Path
from typing import Iterable, List, Tuple

from cctbx.array_family import flex
import iotbx.mtz as mtz


# ─── simple containers ──────────────────────────────────────────────────────
class UnitCell:
    def __init__(self, a,b,c, al,be,ga):
        self.a,self.b,self.c = a,b,c
        self.al,self.be,self.ga = al,be,ga
    def as_tuple(self):                       # 6-tuple needed by add_crystal
        return (self.a,self.b,self.c,self.al,self.be,self.ga)
    def __str__(self):
        return (f"{self.a:.3f} {self.b:.3f} {self.c:.3f} Å, "
                f"{self.al:.2f} {self.be:.2f} {self.ga:.2f} °")

class Reflection:
    __slots__=("h","k","l","i","sigi")
    def __init__(self,h,k,l_,I,SIG):
        self.h,self.k,self.l = h,k,l_
        self.i,self.sigi     = I,SIG


# ─── regex helpers ──────────────────────────────────────────────────────────
_HEADER_NUM  = re.compile(  # numeric “key = value” (unit allowed)
    r"^\s*(\w+)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:\s+\S+)?")
_REFL_LINE   = re.compile(  # first five numeric cols only
    r"\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+([-0-9.+eE]+)\s+([-0-9.+eE]+)")


# ─── parsing ────────────────────────────────────────────────────────────────
def parse_header(lines:Iterable[str]) -> Tuple[dict,UnitCell]:
    geom, cell = {}, {}
    in_geom = in_uc = False
    for ln in lines:
        if ln.startswith("----- Begin geometry file -----"): in_geom=True;  continue
        if ln.startswith("----- End geometry file -----"):   in_geom=False; continue
        if ln.startswith("----- Begin unit cell -----"):     in_uc=True;    continue
        if ln.startswith("----- End unit cell -----"):       in_uc=False;   break
        if in_geom or in_uc:
            m=_HEADER_NUM.match(ln)
            if m:
                key,val=m.groups(); val=float(val)
                (geom if in_geom else cell)[key]=val

    uc = UnitCell(
        cell.get("a",1), cell.get("b",cell.get("a",1)), cell.get("c",1),
        cell.get("al",90), cell.get("be",90), cell.get("ga",90)
    )
    return geom, uc


def iterate_frames(path:Path) -> Iterable[List[Reflection]]:
    with path.open("r",encoding="utf-8",errors="ignore") as fh:
        in_tbl=False; buffer=[]
        for ln in fh:
            if ln.startswith("----- Begin chunk -----"): buffer=[]
            elif ln.startswith("----- End chunk -----"):
                if buffer: yield buffer
            elif ln.lstrip().startswith("Reflections measured"): in_tbl=True
            elif ln.lstrip().startswith("End of reflections"):   in_tbl=False
            elif in_tbl:
                m=_REFL_LINE.match(ln)
                if m:
                    h,k,l_,I,SIG = m.groups()
                    buffer.append(Reflection(
                        int(h), int(k), int(l_), float(I), float(SIG)))


# ─── MTZ writer ─────────────────────────────────────────────────────────────
def write_mtz(refs:List[Reflection], cell:UnitCell,
              out_path:Path, sg_symbol:str="P1") -> None:
    n=len(refs)
    obj = mtz.object()
    obj.set_title(out_path.stem)

    # create crystal & dataset
    cryst = obj.add_crystal("CRYST","PROJECT", cell.as_tuple())
    cryst.set_space_group_name(sg_symbol)
    dset  = cryst.add_dataset("RAW")          # wavelength=None by default

    obj.set_n_refl(n)                         # allocate rows

    def add_column(name, typ, data):
        col = dset.add_column(name, typ)
        if typ=="H":
            col.set_ints (flex.int(data))
        else:
            col.set_reals(flex.double(data))

    add_column("H",    "H", [r.h for r in refs])
    add_column("K",    "H", [r.k for r in refs])
    add_column("L",    "H", [r.l for r in refs])
    add_column("I",    "J", [r.i for r in refs])
    add_column("SIGI", "Q", [r.sigi for r in refs])

    obj.write(str(out_path))


# ─── driver ────────────────────────────────────────────────────────────────
def process(stream:Path, chunk:int, out_dir:Path) -> None:
    if not stream.is_file():
        sys.exit(f"[ERROR] No such .stream file: {stream}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with stream.open("r",encoding="utf-8",errors="ignore") as fh:
        header=itertools.takewhile(lambda l: not l.startswith("----- Begin chunk"), fh)
        geom, cell = parse_header(header)

    print("[INFO] Geometry parameters:",
          ", ".join(f"{k}={v}" for k,v in geom.items()))
    print("[INFO] Target unit cell   :", cell)

    frames=iterate_frames(stream)
    idx=1
    while True:
        group=list(itertools.islice(frames, chunk))
        if not group: break
        refl=list(itertools.chain.from_iterable(group))

        out=out_dir/f"chunk_{idx:03d}.mtz"
        write_mtz(refl, cell, out)
        print(f"[OK]  Wrote {len(refl)} reflections → {out}")
        idx+=1

    print("[DONE] All MTZ chunks written.")


# ─── CLI ────────────────────────────────────────────────────────────────────
def main(argv=None):
    p=argparse.ArgumentParser(description="CrystFEL stream → MTZ (duplicates OK)")
    p.add_argument("--stream", required=True, help=".stream file path")
    p.add_argument("--chunk-size", type=int, required=True, help="Frames per MTZ")
    p.add_argument("--out-dir", default="mtz/", help="Output directory")
    args=p.parse_args(argv)

    process(Path(args.stream).expanduser(),
            args.chunk_size,
            Path(args.out_dir).expanduser())

if __name__ == "__main__":
    main()
