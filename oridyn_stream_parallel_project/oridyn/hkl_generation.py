"""Candidate HKL generation and centering rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import hkl_lab_vectors, reciprocal_matrix_from_cell
from .stream_parser import UnitCell


def allowed_by_centering(h: int, k: int, l: int, centering: str = "P") -> bool:
    """Return whether an HKL passes simple lattice-centering extinction rules."""

    code = (centering or "P").upper()
    if code == "P":
        return True
    if code == "A":
        return ((k + l) % 2) == 0
    if code == "B":
        return ((h + l) % 2) == 0
    if code == "C":
        return ((h + k) % 2) == 0
    if code == "I":
        return ((h + k + l) % 2) == 0
    if code == "F":
        return (h % 2) == (k % 2) == (l % 2)
    if code == "R":
        return ((h - k) % 3 == 0) and ((k - l) % 3 == 0)
    return True


def generate_candidate_hkls(
    cell: UnitCell,
    dmin: float,
    dmax: float,
    hkl_limit: int | None = None,
    max_candidates: int | None = None,
    centering: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate candidate HKLs in a d-spacing interval."""

    if dmin <= 0.0 or dmax <= 0.0:
        raise ValueError("dmin and dmax must be positive.")
    if dmin > dmax:
        raise ValueError("dmin must be smaller than or equal to dmax.")

    reciprocal = reciprocal_matrix_from_cell(cell)
    qmin = 1.0 / float(dmax)
    qmax = 1.0 / float(dmin)
    basis_norms = np.linalg.norm(reciprocal, axis=0)
    bounds = np.ceil(qmax / np.maximum(basis_norms, 1e-12)).astype(int) + 2
    if hkl_limit is not None:
        bounds = np.minimum(bounds, int(hkl_limit))
    hmax, kmax, lmax = (int(x) for x in bounds)
    use_centering = centering if centering is not None else cell.centering

    records: list[dict[str, float | int]] = []
    for h in range(-hmax, hmax + 1):
        for k in range(-kmax, kmax + 1):
            for l in range(-lmax, lmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not allowed_by_centering(h, k, l, use_centering):
                    continue
                q = float(np.linalg.norm(hkl_lab_vectors((h, k, l), reciprocal)[0]))
                if qmin <= q <= qmax:
                    records.append({"h": h, "k": k, "l": l, "q_invA": q, "d_angstrom": 1.0 / q})

    candidates = pd.DataFrame.from_records(records)
    truncated = False
    n_before_cap = int(len(candidates))
    if candidates.empty:
        raise ValueError("No candidate HKLs generated for the supplied d-spacing limits.")
    candidates = candidates.sort_values(["q_invA", "h", "k", "l"]).reset_index(drop=True)
    if max_candidates is not None and len(candidates) > int(max_candidates):
        candidates = candidates.head(int(max_candidates)).copy()
        truncated = True
    candidates["candidate_index"] = np.arange(len(candidates), dtype=int)
    metadata = {
        "dmin": float(dmin),
        "dmax": float(dmax),
        "qmin_invA": qmin,
        "qmax_invA": qmax,
        "hkl_bounds": [hmax, kmax, lmax],
        "centering": use_centering,
        "n_candidates_before_cap": n_before_cap,
        "n_candidates": int(len(candidates)),
        "max_candidates": max_candidates,
        "truncated": truncated,
    }
    return candidates, metadata
