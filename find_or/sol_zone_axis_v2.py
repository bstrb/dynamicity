#!/usr/bin/env python3
"""
sol_zone_axis_v2.py  —  report the zone axis [h k l] for each orientation
                     listed in a CrystFEL *.sol file.

Handles any orthogonal unit cell (α = β = γ = 90°) by scaling the
fractional indices component-wise with (1/a, 1/b, 1/c).

-----------------------------------------------------------------------
Example usage
-------------
# cubic (a = 11.8725 Å)
$ python sol_zone_axis_v2.py LTA_orientations.sol --a 11.8725

# tetragonal I-centred (a = b = 15.1215 Å, c = 12.0683 Å)
$ python sol_zone_axis_v2.py MFM300_tI.orientations.sol --a 15.1215 --c 12.0683

# orthorhombic (all three different)
$ python sol_zone_axis_v2.py somefile.sol --a 12.3 --b 14.1 --c 7.9 --thr 0.25
"""
from __future__ import annotations
import numpy as np
import argparse, math, pathlib, sys

# ----------------------------------------------------------------------
def gcd3(a: int, b: int, c: int) -> int:
    """Greatest common divisor of three integers, zeros allowed."""
    return math.gcd(math.gcd(abs(a), abs(b)), abs(c))

def nearest_zone_axis(invG_nm: np.ndarray,
                      beam: tuple[float,float,float],
                      axes_nm: np.ndarray,
                      thr: float = 0.30) -> tuple[int,int,int] | None:
    """
    Project *beam* through invG, divide by (a,b,c) to get fractional
    indices, rescale so the smallest non-zero component is 1, then round.

    Parameters
    ----------
    invG_nm : 3×3 ndarray
        Inverse reciprocal-basis matrix with nm units.
    beam : length-3 iterable
        Beam direction in lab frame (default = (0,0,1)).
    axes_nm : ndarray, shape (3,)
        Direct-cell edges (a,b,c) in nm.
    thr : float
        RMS tolerance between fractional and rounded indices.

    Returns
    -------
    Reduced (h,k,l) triple or None if outside tolerance.
    """
    hkl_real = invG_nm @ np.asarray(beam, float)         # nm
    hkl_frac = hkl_real / axes_nm                        # dimensionless

    nz = np.abs(hkl_frac) > 1e-8                         # non-zero comps
    if not np.any(nz):
        return None
    scale = np.min(np.abs(hkl_frac[nz]))                 # smallest non-zero
    hkl_scaled = hkl_frac / scale

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
                    "(orthogonal lattices).")
    p.add_argument("solfile", help=".sol text file to read")
    p.add_argument("--a", type=float, required=True,
                   help="cell edge a in Å (required)")
    p.add_argument("--b", type=float,
                   help="cell edge b in Å (defaults to --a)")
    p.add_argument("--c", type=float,
                   help="cell edge c in Å (defaults to --a if omitted)")
    p.add_argument("--thr", type=float, default=0.30,
                   help="tolerance for rounding to integer hkl (default 0.30)")
    p.add_argument("--beam", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                   metavar=("X", "Y", "Z"),
                   help="lab-frame beam direction (default 0 0 1)")
    return p.parse_args()

# ----------------------------------------------------------------------
def main():
    args   = parse_args()
    a_nm   = args.a / 10.0
    b_nm   = (args.b or args.a) / 10.0
    c_nm   = (args.c or args.a) / 10.0
    axes_nm = np.array([a_nm, b_nm, c_nm])
    inv_axes_nm = 1.0 / axes_nm           # needed later
    beam  = tuple(args.beam)
    thr   = args.thr

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
                nums = list(map(float, tok[2:11]))       # nine G* numbers
            except ValueError:
                sys.stderr.write(f"Non-numeric line skipped:\n{line}")
                continue

            Gstar = np.column_stack((nums[0:3], nums[3:6], nums[6:9]))
            try:
                invG_nm = np.linalg.inv(Gstar)           # nm
            except np.linalg.LinAlgError:
                sys.stderr.write(f"Singular G* for {img} {idx}; skipped\n")
                continue

            hkl = nearest_zone_axis(invG_nm, beam, axes_nm, thr)
            if hkl is None:
                print(f"{img} {idx}  —  no integral zone within tolerance")
            else:
                print(f"{img} {idx}  →  [{hkl[0]} {hkl[1]} {hkl[2]}]")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
