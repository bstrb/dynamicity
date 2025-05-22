#!/usr/bin/env python3
"""
misorientation_from_stream.py
-----------------------------
Parse a CrystFEL *.stream* file and compute the angular misorientation
between the indexed orientation of each chunk and a reference orientation.

USAGE
-----
python misorientation_from_stream.py streamfile.stream \
        --ref "0 0.8422826 0  -0.8422826 0 0  0 0 0.8422826" \
        --csv results.csv

  * If --ref is omitted the script will read the FIRST occurrence of
    astar/bstar/cstar in the file and use that as the reference.
  * --csv is optional; if given a CSV is written in addition to stdout.

The --ref string must contain NINE numbers:
    (astar_x astar_y astar_z  bstar_x …  cstar_z)
i.e. the three reciprocal-lattice vectors in column order, space-separated.
"""
import argparse
import pathlib
from pathlib import Path
import re
import sys
from typing import List, Tuple

import numpy as np
import csv

# ---------- helpers ---------------------------------------------------------
vec_re = re.compile(
    r'^[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+'
    r'([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+'
    r'([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)'
)

event_re = re.compile(r'^Event:\s*(\S+)')

def misorientation_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                       M_refinv: np.ndarray) -> float:
    """
    Compute the misorientation angle (in degrees) between orientation matrix
    M = [a b c] and reference using
        R = M * M_ref^{-1};  angle = acos((tr(R) - 1)/2)
    The input vectors must already be numpy 3-vectors.
    """
    M = np.column_stack((a, b, c))
    R = M @ M_refinv
    # Numerical safety
    cosang = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# ---------- main ------------------------------------------------------------
def parse_stream(path: pathlib.Path) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return a list of (event_id, astar, bstar, cstar) from the stream file.
    astar, bstar, cstar are np.array shape-(3,).
    """
    out = []
    with path.open() as fh:
        in_chunk = False
        cur_event = None
        a = b = c = None
        for line in fh:
            # Beginning of a chunk?
            if line.startswith("----- Begin chunk"):
                in_chunk = True
                cur_event = a = b = c = None
                continue
            # End of a chunk -> flush
            if line.startswith("----- End chunk -----") and in_chunk:
                if cur_event and a is not None and b is not None and c is not None:
                    out.append((cur_event, a, b, c))
                in_chunk = False
                continue
            if not in_chunk:
                continue

            # Inside a chunk
            m_event = event_re.match(line)
            if m_event:
                cur_event = m_event.group(1)
                continue
            m_vec = vec_re.match(line)
            if m_vec:
                vec = np.array([float(m_vec.group(i)) for i in (1, 2, 3)])
                if line.startswith("astar"):
                    a = vec
                elif line.startswith("bstar"):
                    b = vec
                elif line.startswith("cstar"):
                    c = vec
    return out


def main() -> None:
    stream = Path("/Users/xiaodong/Desktop/simulations/LTA/simulation-12/from_file_-512.5_-512.5.stream")
    ref = "0 0.8422826 0 -0.8422826 0 0 0 0 0.8422826"
    csv_results = Path("results.csv")

    # Read orientations from file
    chunks = parse_stream(stream)
    if not chunks:
        sys.exit("No orientation matrices found!")

    # Determine reference matrix
    if ref:
        ref_vals = np.fromstring(ref, sep=' ')
        if ref_vals.size != 9:
            sys.exit("--ref must contain 9 numbers (a*, b*, c*)")
        ref_vecs = [ref_vals[i*3:(i+1)*3] for i in range(3)]
    else:
        # Use the first chunk
        ref_vecs = chunks[0][1:4]
        print(f"[info] Using event {chunks[0][0]} as reference", file=sys.stderr)


    M_ref = np.column_stack(ref_vecs)   # shape (3, 3)
    M_refinv = np.linalg.inv(M_ref)     # works 


    results = []
    for event, a, b, c in chunks:
        ang = misorientation_deg(a, b, c, M_refinv)
        results.append((event, ang))

    # Pretty print to stdout
    print(f"{'Event':<20}  Misorientation (°)")
    print("-"*36)
    for ev, ang in results:
        print(f"{ev:<20}  {ang:8.3f}")

    # Optional CSV
    if csv_results:
        with csv_results.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["Event", "Misorientation_deg"])
            wr.writerows(results)
        print(f"[info] CSV written to {csv_results}", file=sys.stderr)


if __name__ == "__main__":
    main()
