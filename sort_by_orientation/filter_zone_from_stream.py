#!/usr/bin/env python3
"""
Filter (near-)ZOLZ (or any zone) reflections from a CrystFEL .stream file.

Removes reflection rows that satisfy |h*u + k*v + l*w| <= tolerance.
- Streams the file for scalability (only buffers one crystal block at a time).
- Preserves all non-reflection lines exactly as they are.
- Patches `num_reflections = ...` per filtered crystal.
- Can limit filtering to the first N chunks and/or first N crystals; the rest are copied untouched.

Usage examples:
  # Classic ZOLZ (l == 0), no tolerance, filter all chunks
  python3 filter_zone_from_stream.py --in in.stream --out out.stream

  # Drop a band around ZOLZ: |l| <= 1
  python3 filter_zone_from_stream.py --in in.stream --out out.stream --tolerance 1

  # Custom zone [1 0 1] with tolerance 2, only first 10 chunks
  python3 filter_zone_from_stream.py --in in.stream --out out.stream --zone 1 0 1 --tolerance 2 --limit-chunks 10

  # Limit by first 200 crystals regardless of chunks
  python3 filter_zone_from_stream.py --in in.stream --out out.stream --limit-crystals 200
"""
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

# ---------- Regexes (compiled once) ----------
BEGIN_CHUNK = re.compile(r"^\s*-{5}\s*Begin chunk\s*-{5}\s*$")
END_CHUNK   = re.compile(r"^\s*-{5}\s*End chunk\s*-{5}\s*$")

BEGIN_CRYSTAL = re.compile(r"^\s*---\s*Begin crystal")
END_CRYSTAL   = re.compile(r"^\s*---\s*End crystal")

BEGIN_REFL = re.compile(r"^\s*Reflections measured after indexing\s*$")
END_REFL   = re.compile(r"^\s*End of reflections\s*$")

NUM_REFL_LINE = re.compile(r"^\s*num_reflections\s*=\s*(\d+)\s*(?:\r?\n)?$")

# A reflection row begins with three integers (h k l). We keep this liberal to preserve spacing verbatim.
REFLECTION_ROW_PREFIX = re.compile(r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\b")
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remove (near-)ZOLZ reflections from a CrystFEL .stream.")
    p.add_argument("--in",  dest="inp",  required=True, help="Input .stream path")
    p.add_argument("--out", dest="outp", required=True, help="Output .stream path")
    p.add_argument("--zone", nargs=3, type=int, default=[0, 0, 1],
                   metavar=("u", "v", "w"),
                   help="Fallback zone axis [u v w] as integers. Default [0 0 1] (ZOLZ -> l == 0).")
    p.add_argument("--zone-list", dest="zone_list", type=str, default=None,
                   help="Path to text file with one [u v w] per line (same order as crystals). Extra text on a line is ignored.")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="Drop if |h*u + k*v + l*w| <= tolerance (float). Default 0.0.")
    p.add_argument("--limit-crystals", type=int, default=None,
                   help="Filter only the first N crystals (counting from the start of the file).")
    return p.parse_args()
def parse_zone_list_file(path: Path) -> List[Tuple[int,int,int]]:
    """
    Parse lines like: 'sim.h5 //1058 -> [1 0 0] score=...'
    Returns a list of (u,v,w) for crystals in order.
    Lines without a bracketed triple are skipped.
    """
    import re
    triple_re = re.compile(r"\[([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\]")
    zones: List[Tuple[int,int,int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = triple_re.search(ln)
            if m:
                zones.append(tuple(map(int, m.groups())))
    return zones

def dot_int(h:int, k:int, l:int, zone:Tuple[int,int,int]) -> int:
    u, v, w = zone
    return h*u + k*v + l*w

def process_stream(inp: Path, outp: Path,
                   zone: Tuple[int,int,int],
                   tolerance: float,
                   limit_crystals: Optional[int],
                   zone_list: Optional[List[Tuple[int,int,int]]] = None) -> None:
    """
    Stream input -> output, buffering only the current crystal block to patch num_reflections.
    If zone_list is provided, use its i-th [u v w] for the i-th crystal; otherwise use `zone`.
    """
    crystals_seen = 0
    in_crystal = False
    in_reflections = False
    filter_active_for_this_crystal = True

    # Per-crystal zone (updated at each crystal start)
    zone_for_this_crystal: Tuple[int,int,int] = zone

    crystal_buf: List[str] = []
    num_refl_line_idx: Optional[int] = None
    kept_reflections_in_crystal = 0

    def should_filter_crystal() -> bool:
        if limit_crystals is None:
            return True
        return crystals_seen < limit_crystals

    def begin_crystal(line: str):
        nonlocal in_crystal, in_reflections, crystal_buf, num_refl_line_idx
        nonlocal kept_reflections_in_crystal, filter_active_for_this_crystal, crystals_seen
        nonlocal zone_for_this_crystal

        in_crystal = True
        in_reflections = False
        crystals_seen += 1
        filter_active_for_this_crystal = should_filter_crystal()

        # Choose zone from list if available; fallback to global
        if zone_list and (crystals_seen - 1) < len(zone_list):
            zone_for_this_crystal = zone_list[crystals_seen - 1]
        else:
            zone_for_this_crystal = zone

        crystal_buf = [line]
        num_refl_line_idx = None
        kept_reflections_in_crystal = 0

    def flush_crystal(outf):
        nonlocal crystal_buf, num_refl_line_idx, kept_reflections_in_crystal
        if not crystal_buf:
            return
        if filter_active_for_this_crystal and num_refl_line_idx is not None:
            original = crystal_buf[num_refl_line_idx]
            nl = "\n" if not original.endswith("\r\n") else "\r\n"
            crystal_buf[num_refl_line_idx] = f"num_reflections = {kept_reflections_in_crystal}{nl}"
        outf.writelines(crystal_buf)
        crystal_buf = []
        num_refl_line_idx = None
        kept_reflections_in_crystal = 0

    with inp.open("r", encoding="utf-8", errors="replace") as inf, \
         outp.open("w", encoding="utf-8", newline="") as outf:

        for line in inf:
            if not in_crystal:
                if BEGIN_CRYSTAL.match(line):
                    begin_crystal(line)
                else:
                    outf.write(line)
                continue

            # inside a crystal (buffering)
            if num_refl_line_idx is None and NUM_REFL_LINE.match(line):
                num_refl_line_idx = len(crystal_buf)

            if BEGIN_REFL.match(line):
                in_reflections = True
                crystal_buf.append(line)
                continue

            if END_REFL.match(line):
                in_reflections = False
                crystal_buf.append(line)
                continue

            if END_CRYSTAL.match(line):
                crystal_buf.append(line)
                flush_crystal(outf)
                in_crystal = False
                in_reflections = False
                continue

            if in_reflections and filter_active_for_this_crystal:
                m = REFLECTION_ROW_PREFIX.match(line)
                if m:
                    try:
                        h, k, l = map(int, m.groups())
                    except Exception:
                        crystal_buf.append(line)
                        continue
                    # Use per-crystal zone here:
                    val = abs(dot_int(h, k, l, zone_for_this_crystal))
                    if val <= tolerance:
                        # drop this reflection row
                        continue
                    kept_reflections_in_crystal += 1
                    crystal_buf.append(line)
                else:
                    crystal_buf.append(line)
            else:
                crystal_buf.append(line)

        if in_crystal and crystal_buf:
            flush_crystal(outf)
def main():
    args = parse_args()
    inp = Path(args.inp)
    outp = Path(args.outp)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    zl: Optional[List[Tuple[int,int,int]]] = None
    if args.zone_list:
        p = Path(args.zone_list)
        if not p.exists():
            raise SystemExit(f"Zone list not found: {p}")
        zl = parse_zone_list_file(p)

    process_stream(
        inp=inp,
        outp=outp,
        zone=tuple(args.zone),
        tolerance=args.tolerance,  # no int() cast
        limit_crystals=args.limit_crystals,
        zone_list=zl,
    )

if __name__ == "__main__":
    main()
