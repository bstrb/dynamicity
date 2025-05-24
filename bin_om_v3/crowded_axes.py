#!/usr/bin/env python3
"""
crowded_axes.py
--------------------------------------------------------------------
Crowded-axis detector for any Bravais lattice (P, I, F, A, B, C, R).

Input (CLI):
  --system         triclinic / monoclinic / orthorhombic / tetragonal /
                   trigonal / hexagonal / cubic
  --centering      P I F A B C R   (R = rhombohedral hex setting)
  --unique-axis    a|b|c   (needed only for monoclinic / some ortho)
  --hmax N         enumerate Miller indices |h|,|k|,|l| ≤ N   (default 3)
  --gmax nm^-1     optional reciprocal-sphere cut-off
  --cell           a b c α β γ     six numbers (Å and deg) if you want
                   |g| calculations; otherwise counts are index-based
  --top N          how many axes to print (default 8)

Output:
  Table: [u v w]   plane-count

The program applies only the **centring extinction rule**; it does *not*
try to expand space-group symmetry (that is enough to judge crowdedness).

Examples
--------
1. Body-centred tetragonal, index radius 4, no metric:
   crowded_axes.py --system tetragonal --centering I --hmax 4

2. Monoclinic C, b-unique, with metric and |g|≤1.0 nm⁻¹:
   crowded_axes.py --system monoclinic --centering C --unique-axis b \\
                   --cell 12.1 10.3 9.7 90 92.5 90 --gmax 1.0
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------- extinction tests --------------------------- #
def passes_centering(h: int, k: int, l: int, cent: str) -> bool:
    """Return True if (h k l) is allowed by the centring symbol."""
    cent = cent.upper()
    if cent == "P":
        return True
    if cent == "I":
        return (h + k + l) % 2 == 0
    if cent == "F":
        return (h % 2 + k % 2 + l % 2) % 2 == 0
    if cent == "A":
        return (h + l) % 2 == 0
    if cent == "B":
        return (k + l) % 2 == 0
    if cent == "C":
        return (h + k) % 2 == 0
    if cent == "R":
        # rhombohedral, hexagonal axes: -h + k + l ≡ 0 (mod 3)
        return (-h + k + l) % 3 == 0
    raise ValueError(f"Unknown centring: {cent}")

# ----------------------------- |g| calculation ---------------------------- #
def reciprocal_vector(h: int, k: int, l: int,
                      a: float, b: float, c: float,
                      alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return reciprocal vector (nm^-1) in Cartesian lab frame."""
    # convert to nanometres to match gmax units
    a *= 0.1; b *= 0.1; c *= 0.1
    alpha_r, beta_r, gamma_r = map(math.radians, (alpha, beta, gamma))
    vol = (a * b * c *
           math.sqrt(1 - math.cos(alpha_r)**2
                       - math.cos(beta_r)**2
                       - math.cos(gamma_r)**2
                       + 2 * math.cos(alpha_r)
                           * math.cos(beta_r)
                           * math.cos(gamma_r)))
    astar = (b * c * math.sin(alpha_r)) / vol
    bstar = (a * c * math.sin(beta_r)) / vol
    cstar = (a * b * math.sin(gamma_r)) / vol
    # Cartesian components (hex/rhombohedral need a full transform, but
    # magnitude is sufficient here, so we return the length directly)
    g_mag = math.sqrt((h * astar)**2 + (k * bstar)**2 + (l * cstar)**2
                      + 2 * h*k*astar*bstar*math.cos(gamma_r)
                      + 2 * h*l*astar*cstar*math.cos(beta_r)
                      + 2 * k*l*bstar*cstar*math.cos(alpha_r))
    return g_mag  # nm^-1

# ---------------------------- crowdedness core --------------------------- #
def crowded_axes(system: str,
                 centering: str,
                 hmax: int,
                 unique_axis: str | None = None,
                 gmax: float | None = None,
                 cell: Tuple[float, float, float, float, float, float] | None = None,
                 top_n: int = 8
                 ) -> List[Tuple[Tuple[int, int, int], int]]:
    """Return (axis, plane_count) sorted descending."""
    centering = centering.upper()
    if centering not in "PIFABCR":
        raise ValueError("centering must be one of P I F A B C R")

    # reflection set
    refl: List[Tuple[int, int, int]] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if not passes_centering(h, k, l, centering):
                    continue
                if gmax and cell:
                    g_len = reciprocal_vector(h, k, l, *cell)
                    if g_len > gmax:
                        continue
                refl.append((h, k, l))

    # zone-axis list
    scores: Dict[Tuple[int, int, int], int] = {}
    for u in range(-hmax, hmax + 1):
        for v in range(-hmax, hmax + 1):
            for w in range(-hmax, hmax + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                axis = (u, v, w)
                # optional gcd filter (primitive axis)
                g = math.gcd(math.gcd(abs(u), abs(v)), abs(w))
                if g > 1:
                    continue
                n = 0
                for h, k, l in refl:
                    if h * u + k * v + l * w == 0:
                        n += 1
                scores[axis] = n

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]

# ---------------------------- CLI handling ------------------------------- #
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser("crowded_axes.py")
    ap.add_argument("--system", required=True,
                    choices=["triclinic", "monoclinic", "orthorhombic",
                             "tetragonal", "trigonal", "hexagonal", "cubic"])
    ap.add_argument("--centering", required=True,
                    help="P I F A B C R")
    ap.add_argument("--unique-axis",
                    help="a|b|c  (needed only for monoclinic / some ortho)")
    ap.add_argument("--hmax", type=int, default=3)
    ap.add_argument("--gmax", type=float,
                    help="reciprocal-sphere radius nm^-1 (optional)")
    ap.add_argument("--cell", nargs=6, type=float,
                    metavar=("a", "b", "c", "alpha", "beta", "gamma"),
                    help="cell parameters Å deg (optional)")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args(argv)

    if args.system == "monoclinic" and not args.unique_axis:
        ap.error("monoclinic needs --unique-axis a|b|c")

    cell = tuple(args.cell) if args.cell else None

    ranked = crowded_axes(system=args.system,
                          centering=args.centering,
                          unique_axis=args.unique_axis,
                          hmax=args.hmax,
                          gmax=args.gmax,
                          cell=cell,
                          top_n=args.top)

    print(f"\nMost crowded zone axes "
          f"({args.system} {args.centering}, hmax={args.hmax})")
    if args.gmax:
        print(f"gmax = {args.gmax:.3f} nm^-1")
    print("-" * 34)
    for i, (axis, n) in enumerate(ranked, 1):
        u, v, w = axis
        print(f"{i:2d}. [{u:2d} {v:2d} {w:2d}]   planes: {n}")
    print()

if __name__ == "__main__":
    main()
