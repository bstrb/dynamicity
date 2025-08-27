#!/usr/bin/env python3
"""
stream_to_cell.py — Extract the unit cell block from a CrystFEL stream header
and write it as a standalone CrystFEL .cell file.

Usage:
  python3 stream_to_cell.py --stream /path/to/file.stream --outdir /path/to/out
  # (also works with .stream.gz)
"""

import argparse
import gzip
import os
import re
import sys
from typing import List

BEGIN_RE = re.compile(r"Begin\s+unit\s+cell", re.IGNORECASE)
END_RE   = re.compile(r"End\s+unit\s+cell",   re.IGNORECASE)

def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def extract_unit_cell_lines(stream_path: str) -> List[str]:
    """
    Returns the lines inside the '----- Begin unit cell -----' block,
    excluding the Begin/End marker lines themselves.
    """
    in_block = False
    cell_lines: List[str] = []
    with open_maybe_gz(stream_path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not in_block:
                if BEGIN_RE.search(line):
                    in_block = True
                continue
            # in_block == True
            if END_RE.search(line):
                break
            cell_lines.append(line)

    if not cell_lines:
        raise RuntimeError(
            "Unit cell block not found. Ensure the stream header contains "
            "'----- Begin unit cell -----' ... '----- End unit cell -----'."
        )
    return cell_lines

def ensure_has_version_header(lines: List[str]) -> List[str]:
    """
    Ensure the first non-empty, non-comment line is:
      'CrystFEL unit cell file version 1.0'
    If not present, prepend it.
    """
    # Find first substantive line
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(";"):  # comment
            continue
        # Is it already the version header?
        if s.lower().startswith("crystfel unit cell file version"):
            return lines
        break
    # Prepend the header
    return ["CrystFEL unit cell file version 1.0"] + ([""] if lines and lines[0].strip() else []) + lines

def write_cell_file(outdir: str, lines: List[str], name: str = "cell.cell") -> str:
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Extract unit cell from a CrystFEL stream header.")
    ap.add_argument("--stream", "-s", required=True, help="Path to .stream (or .stream.gz)")
    ap.add_argument("--outdir", "-o", required=True, help="Directory to write cell.cell")
    ap.add_argument("--outfile", default="cell.cell", help="Output filename (default: cell.cell)")
    args = ap.parse_args()

    if not os.path.exists(args.stream):
        print(f"ERROR: stream not found: {args.stream}", file=sys.stderr)
        sys.exit(2)

    try:
        cell_lines = extract_unit_cell_lines(args.stream)
        cell_lines = ensure_has_version_header(cell_lines)
        out_path = write_cell_file(args.outdir, cell_lines, args.outfile)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote unit cell from stream header to: {out_path}")

if __name__ == "__main__":
    main()
