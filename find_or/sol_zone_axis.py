#!/usr/bin/env python3
"""
sol_zone_axis.py  —  report the zone axis [h k l] for each entry in a
CrystFEL .sol file (cubic cell only, edge a hard-coded or via --a).

Quick run:
    python sol_zone_axis.py orientation_matrices.sol
Options:
    --a   unit-cell edge in Å  (default 11.8725)
    --thr rounding tolerance  (default 0.30)
"""
from __future__ import annotations
import numpy as np
import argparse, math, pathlib, sys

# ----------------------------------------------------------------------
def gcd3(a: int, b: int, c: int) -> int:
    """Greatest common divisor of three integers, zeros allowed."""
    return math.gcd(math.gcd(abs(a), abs(b)), abs(c))

def nearest_zone_axis(invG_nm: np.ndarray,
                      beam: tuple[float,float,float] = (0.0, 0.0, 1.0),
                      thr: float = 0.30) -> tuple[int,int,int] | None:
    """
    Convert beam direction to fractional indices, rescale so the
    smallest non-zero component is 1, then round to nearest integers.

    Returns (h,k,l) reduced to lowest terms or None if outside tolerance.
    """
    hkl = invG_nm @ np.asarray(beam, dtype=float)

    # find scaling factor = smallest non-zero |component|
    nz = np.abs(hkl) > 1e-8
    if not np.any(nz):
        return None
    scale = np.min(np.abs(hkl[nz]))
    hkl_scaled = hkl / scale

    hkl_round = np.rint(hkl_scaled)
    if np.linalg.norm(hkl_scaled - hkl_round) > thr:
        return None

    h, k, l = map(int, hkl_round)
    g = gcd3(h, k, l) or 1
    return (h // g, k // g, l // g)

# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Determine zone axes from a CrystFEL .sol file "
                    "(cubic lattice).")
    p.add_argument("solfile", help=".sol text file")
    p.add_argument("--a", type=float, default=11.8725,
                   help="cubic cell edge a in Å (default 11.8725)")
    p.add_argument("--thr", type=float, default=0.30,
                   help="tolerance for rounding to integer hkl (default 0.30)")
    return p.parse_args()

# ----------------------------------------------------------------------
def main():
    args  = parse_args()
    inv_a = 10.0 / args.a          # 1/a in nm⁻¹
    thr   = args.thr
    beam  = (0.0, 0.0, 1.0)        # lab Z

    with pathlib.Path(args.solfile).expanduser().open() as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            tok = line.split()
            if len(tok) < 13:
                sys.stderr.write(f"Malformed line skipped:\n{line}")
                continue

            img, idx = tok[0], tok[1]
            try:
                nums = list(map(float, tok[2:11]))   # nine G* components
            except ValueError:
                sys.stderr.write(f"Non-numeric values in line:\n{line}")
                continue

            Gstar = np.column_stack((nums[0:3], nums[3:6], nums[6:9]))
            try:
                invG_nm = np.linalg.inv(Gstar) * inv_a   # dimensionless
            except np.linalg.LinAlgError:
                sys.stderr.write(f"Singular G* for {img} {idx}; skipped\n")
                continue

            hkl = nearest_zone_axis(invG_nm, beam, thr)
            if hkl is None:
                print(f"{img} {idx}  —  no integral zone within tolerance")
            else:
                print(f"{img} {idx}  →  [{hkl[0]} {hkl[1]} {hkl[2]}]")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
