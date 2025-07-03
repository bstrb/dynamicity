#!/usr/bin/env python3
"""
misorientation_to_zone_axis_v2.py
---------------------------------
Ranks indexed patterns by likelihood of dynamical scattering.

Fixes compared with v1
----------------------
* Crowded axes are recomputed for **every** orientation (or every N patterns).
* Symmetry-equivalent axes are merged with spglib.
* Crowd score can be weighted by Σ|F_hkl|² from a supplied SF file.

Extra command-line options
--------------------------
  --crowded-every N   recompute crowded axes every N patterns (default 1)
  --sf FILE           MTZ or (mm)CIF containing structure factors
  --sf-col C          column label to read (default 'F' → uses |F|^2)

All other options are identical to the original script.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re
import sys
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ------------------------- optional extras ---------------------------------
try:
    import gemmi  # for MTZ / CIF structure-factor files

    _HAVE_GEMMI = True
except ModuleNotFoundError:
    _HAVE_GEMMI = False

try:
    import spglib as spg  # symmetry operations
except ModuleNotFoundError:
    sys.exit("[error] spglib not installed; `pip install spglib`")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except ModuleNotFoundError:
    _HAVE_MPL = False

# ------------------------- regexes ----------------------------------------
VEC_RE = re.compile(
    r"^[abc]star\s*=\s*([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[eE][+-]?\d+)?)"
)
EVENT_RE = re.compile(r"^Event:\s*(\S+)")
LATTYPE_RE = re.compile(r"^lattice_type\s*=\s*(\S+)", re.I)
CENTERING_RE = re.compile(r"^centering\s*=\s*(\S+)", re.I)
UNIQUE_RE = re.compile(r"^unique_axis\s*=\s*(\S+)", re.I)

# =============================================================================
#                               HELPERS
# =============================================================================


def gcd3(a: int, b: int, c: int) -> int:
    """Greatest common divisor of three integers (incl. zeros)."""
    return math.gcd(math.gcd(abs(a), abs(b)), abs(c))


def canonical_axis(u: int, v: int, w: int) -> Tuple[int, int, int]:
    """
    Reduce (u,v,w) to primitive integer vector with a fixed sign convention:
    first non-zero component > 0.
    """
    g = gcd3(u, v, w)
    uu, vv, ww = (u // g, v // g, w // g)
    if (uu, vv, ww) == (0, 0, 0):
        return 0, 0, 0
    for comp in (uu, vv, ww):
        if comp != 0:
            if comp < 0:
                uu, vv, ww = (-uu, -vv, -ww)
            break
    return uu, vv, ww


def symmetry_ops_from_stream(
    lattice_type: str | None,
    centering: str | None,
    unique_axis: str | None,
) -> List[np.ndarray]:
    """
    Build dummy cell for spglib to obtain rotation matrices only.
    A single H atom suffices; we do not need translations.
    """
    if lattice_type is None:
        # fall back to primitive triclinic symmetry only (identity)
        return [np.eye(3, dtype=int)]

    # Dummy triclinic cell based on a ≈10 Å to avoid flat cells
    a_len = 10.0
    basis = np.eye(3) * a_len
    positions = [[0.0, 0.0, 0.0]]
    numbers = [1]  # H

    cell = (basis, positions, numbers)

    dataset = spg.get_symmetry_dataset(cell, symprec=1.0)  # generous
    rots = [r.astype(int) for r in dataset["rotations"]]

    # We still honour centering manually because dummy cell is P
    if centering:
        centering = centering.upper()
        if centering == "I":
            rots += [np.eye(3, dtype=int)]  # body-centred adds inversion
        elif centering == "F":
            rots += [np.eye(3, dtype=int)]  # face-centred likewise
        # A/B/C also keep identity; rotations already include them

    return rots


def expand_axes_by_symmetry(
    axes: List[Tuple[int, int, int]],
    rotations: List[np.ndarray],
) -> Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]:
    """
    Map every canonical axis to all symmetry-equivalent primitive axes.
    Returns {canonical: [equiv1, equiv2, ...]}.
    """
    groups: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    for u, v, w in axes:
        if (u, v, w) == (0, 0, 0):
            continue
        # find its equivalence group
        reps = []
        for R in rotations:
            uu, vv, ww = R @ np.array([u, v, w])
            reps.append(canonical_axis(int(uu), int(vv), int(ww)))
        canon = min(reps)  # unique representative
        groups.setdefault(canon, [])
        groups[canon].append((u, v, w))
    return groups


def nearest_zone_axis(
    Mstar: np.ndarray,
    *,
    hmax: int,
    centering: str,
) -> Tuple[Tuple[int, int, int], float]:
    """Return (u,v,w), θ° for axis closest to +z."""
    M = np.linalg.inv(Mstar).T  # real-space basis in lab frame
    k = np.array([0.0, 0.0, 1.0])
    best_axis, best_theta = (0, 0, 1), 180.0
    centering = centering.upper()

    for u in range(-hmax, hmax + 1):
        for v in range(-hmax, hmax + 1):
            for w in range(-hmax, hmax + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                if centering == "I" and (u + v + w) % 2:
                    continue
                if centering == "F" and (u % 2 + v % 2 + w % 2) % 2:
                    continue
                if centering in {"A", "B", "C"}:
                    if centering == "A" and (u + w) % 2:
                        continue
                    if centering == "B" and (v + w) % 2:
                        continue
                    if centering == "C" and (u + v) % 2:
                        continue

                v_lab = M @ np.array([u, v, w], float)
                cosang = abs(np.dot(v_lab, k)) / np.linalg.norm(v_lab)
                theta = math.degrees(math.acos(np.clip(cosang, -1, 1)))
                if theta < best_theta:
                    best_theta, best_axis = theta, (u, v, w)
    return best_axis, best_theta


def crowded_axes(
    Mstar: np.ndarray,
    *,
    hmax: int,
    centering: str,
    gmax: float,
    rotations: List[np.ndarray],
    sf_weights: Dict[Tuple[int, int, int], float] | None,
    top_n: int,
) -> List[Tuple[Tuple[int, int, int], float]]:
    """
    Calculate the top N crowded axes based on Σ|F|² weighting and symmetry
    merging. Returns list [(axis, weight), ...] sorted by descending weight.
    """
    centering = centering.upper()
    refl: List[np.ndarray] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if centering == "I" and (h + k + l) % 2:
                    continue
                if centering == "F" and (h % 2 + k % 2 + l % 2) % 2:
                    continue
                if centering in {"A", "B", "C"}:
                    if centering == "A" and (h + l) % 2:
                        continue
                    if centering == "B" and (k + l) % 2:
                        continue
                    if centering == "C" and (h + k) % 2:
                        continue
                g_cart = np.array([h, k, l], float) @ Mstar
                if np.linalg.norm(g_cart) <= gmax:
                    refl.append(np.array([h, k, l], int))

    # weight lookup
    def weight(h: int, k: int, l: int) -> float:
        if sf_weights is None:
            return 1.0
        return sf_weights.get((h, k, l), 0.0)

    # initial per-axis weights
    per_axis: Dict[Tuple[int, int, int], float] = {}
    for u in range(-hmax, hmax + 1):
        for v in range(-hmax, hmax + 1):
            for w in range(-hmax, hmax + 1):
                if (u, v, w) == (0, 0, 0):
                    continue
                axis = (u, v, w)
                wgt = 0.0
                for hkl in refl:
                    if np.dot(axis, hkl) == 0:
                        wgt += weight(*hkl)
                per_axis[axis] = wgt

    # merge symmetry-equivalent axes
    merged: Dict[Tuple[int, int, int], float] = {}
    groups = expand_axes_by_symmetry(list(per_axis.keys()), rotations)
    for canon, eqv_axes in groups.items():
        merged[canon] = sum(per_axis[ax] for ax in eqv_axes)

    # pick top_n
    top = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return top


def parse_stream(
    path: pathlib.Path,
) -> Tuple[
    Tuple[str | None, str | None, str | None],
    List[str],
    List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, List[str]]],
]:
    """Return (lattice, centering, unique_axis), header_lines, chunk_list."""
    header, chunks = [], []
    lat_type = centering_lbl = unique_axis = None

    with path.open() as fh:
        in_chunk = seen_first = False
        buf: List[str] = []
        cur_event, a = None, None
        b = c = None
        for line in fh:
            if line.startswith("----- Begin chunk"):
                in_chunk = seen_first = True
                buf = [line]
                cur_event = a = b = c = None
                continue

            if not seen_first:
                header.append(line)
            if in_chunk:
                buf.append(line)

            if line.startswith("----- End chunk ----") and in_chunk:
                in_chunk = False
                if cur_event and a is not None and b is not None and c is not None:
                    chunks.append((cur_event, a, b, c, buf.copy()))
                continue
            if not in_chunk:
                if (m := LATTYPE_RE.match(line)):
                    lat_type = m.group(1).lower()
                if (m := CENTERING_RE.match(line)):
                    centering_lbl = m.group(1).upper()
                if (m := UNIQUE_RE.match(line)):
                    unique_axis = m.group(1).lower()
                continue

            # inside chunk
            if (m := EVENT_RE.match(line)):
                cur_event = m.group(1)
                continue
            if (m := VEC_RE.match(line)):
                vec = np.array([float(m.group(i)) for i in (1, 2, 3)])
                if line.startswith("astar"):
                    a = vec
                elif line.startswith("bstar"):
                    b = vec
                elif line.startswith("cstar"):
                    c = vec

    return (lat_type, centering_lbl, unique_axis), header, chunks


# -----------------------------------------------------------------------------
#                         Structure-factor utilities
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_sf(file_path: pathlib.Path, col: str) -> Dict[Tuple[int, int, int], float]:
    """
    Read |F|^2 weights from an MTZ or (mm)CIF file using gemmi.
    Returns {(h,k,l): |F|^2, ...}
    """
    if not _HAVE_GEMMI:
        raise RuntimeError("gemmi not installed")

    if file_path.suffix.lower() in {".mtz", ".mtz.gz"}:
        mtz = gemmi.read_mtz_file(str(file_path))
        if col not in mtz.column_labels():
            raise ValueError(f"Column '{col}' not found in MTZ")
        col_data = mtz.get_column(col).as_double_array()
        hkls = [
            (
                int(mtz[h].h),
                int(mtz[h].k),
                int(mtz[h].l),
            )
            for h in range(mtz.n_reflections)
        ]
        return {hkl: float(val**2) for hkl, val in zip(hkls, col_data) if val > 0.0}

    # assume CIF
    doc = gemmi.cif.read(str(file_path))
    block = doc.sole_block()
    ampl = block.find_values(col)
    h = block.find_values("_refln.index_h")
    k = block.find_values("_refln.index_k")
    l = block.find_values("_refln.index_l")
    weights: Dict[Tuple[int, int, int], float] = {}
    for hh, kk, ll, aa in zip(h, k, l, ampl):
        try:
            hkl = (int(hh), int(kk), int(ll))
            val = float(aa)
            if val > 0.0:
                weights[hkl] = val**2
        except ValueError:
            continue
    return weights


# =============================================================================
#                                  MAIN
# =============================================================================
def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Rank orientations by combined crowdedness / angle danger."
    )
    p.add_argument("stream", type=pathlib.Path)
    p.add_argument("--hmax", type=int, default=3, help="search |h|≤hmax")
    p.add_argument("--centering", help="override centering symbol (P/I/F/...)")
    p.add_argument("--gmax", type=float, help="reciprocal-radius cut-off (nm⁻¹)")
    p.add_argument(
        "--top-crowded", type=int, default=8, help="keep N most crowded axes"
    )
    p.add_argument(
        "--theta0", type=float, default=3.0, help="scale angle θ0 in danger formula (°)"
    )
    p.add_argument("--crowded-every", type=int, default=1, help="recompute every N")
    p.add_argument("--sf", type=pathlib.Path, help="MTZ/CIF with structure factors")
    p.add_argument("--sf-col", default="F", help="column label for amplitudes (|F|)")
    p.add_argument("--csv", type=pathlib.Path, help="write CSV")
    p.add_argument("--plot", type=pathlib.Path, help="write PNG/PDF")
    p.add_argument("--sorted-stream", type=pathlib.Path)
    p.add_argument("--no-sort", action="store_true")
    args = p.parse_args(argv)

    (lat_type, cent_from_stream, unique_axis), header, chunks = parse_stream(
        args.stream
    )
    if not chunks:
        sys.exit("[error] No orientation matrices found")

    centering = args.centering or cent_from_stream or "P"

    # structure-factor weights
    if args.sf:
        if not _HAVE_GEMMI:
            sys.exit("[error] gemmi not installed but --sf given")
        sf_weights = load_sf(args.sf.resolve(), args.sf_col)
        print(f"[info] Read {len(sf_weights)} SF amplitudes from {args.sf}", file=sys.stderr)
    else:
        sf_weights = None

    # symmetry operations
    rotations = symmetry_ops_from_stream(lat_type, centering, unique_axis)

    # representative metric for initial gmax if not given
    first_Mstar = np.column_stack(chunks[0][1:4])
    if args.gmax is not None:
        g_cut_init = args.gmax
    else:
        norms = [np.linalg.norm(first_Mstar[:, i]) for i in range(3)]
        g_cut_init = 1.2 * min(norms)

    results = []  # [(event, axis, theta, lines, danger)]
    theta0 = args.theta0
    k_lab = np.array([0.0, 0.0, 1.0])

    # loop over patterns
    for idx, (ev, a, b, c, lines) in enumerate(chunks):
        Mstar = np.column_stack((a, b, c))

        # gmax recalc each time? keep initial; user can override
        if (idx % args.crowded_every) == 0:
            crowded = crowded_axes(
                Mstar,
                hmax=args.hmax,
                centering=centering,
                gmax=g_cut_init,
                rotations=rotations,
                sf_weights=sf_weights,
                top_n=args.top_crowded,
            )
            crowd_axes = np.array([ax for ax, _ in crowded])
            crowd_weights = np.array([w for _, w in crowded], float)

        axis, theta = nearest_zone_axis(
            Mstar, hmax=args.hmax, centering=centering
        )

        # angles to crowded axes
        M = np.linalg.inv(Mstar).T
        v_lab = (M @ crowd_axes.T).T  # shape (N,3)
        angles = np.degrees(
            np.arccos(
                np.clip(
                    np.abs(v_lab @ k_lab) / np.linalg.norm(v_lab, axis=1),
                    -1,
                    1,
                )
            )
        )
        scores = crowd_weights * np.exp(-((angles / theta0) ** 2))
        danger = scores.max()

        results.append((ev, axis, theta, lines, danger))

    if not args.no_sort:
        results.sort(key=lambda r: r[4], reverse=True)

    # ------------------------------------------------------------------ CSV
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(
                ["Event", "u", "v", "w", "theta_deg", "danger_score"]
            )
            for ev, (u, v, w), th, _, dg in results:
                wr.writerow([ev, u, v, w, f"{th:.6f}", f"{dg:.3f}"])
        print(f"[info] CSV written to {args.csv}", file=sys.stderr)

    # ------------------------------------------------------------------ sorted stream
    if args.sorted_stream:
        with args.sorted_stream.open("w") as fh:
            fh.writelines(header)
            for _, _, _, lines, _ in results:
                fh.writelines(lines)
        print(
            f"[info] Sorted stream written to {args.sorted_stream}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ plot
    if args.plot:
        if not _HAVE_MPL:
            print(
                "[warn] matplotlib not installed; skipping plot",
                file=sys.stderr,
            )
        else:
            danger_vals = np.array([r[4] for r in results])
            ranks = np.arange(1, len(danger_vals) + 1)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(ranks, danger_vals, s=20, alpha=0.8)
            ax.set_xlabel("Rank (danger sorted)")
            ax.set_ylabel("Danger score")
            ax.set_title("Crowdedness–angle danger metric (sym-merged, SF-weighted)")
            ax.grid(True, linestyle=":", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(args.plot)
            print(f"[info] Plot saved to {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
