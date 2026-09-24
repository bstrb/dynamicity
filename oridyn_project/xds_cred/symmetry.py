"""Space-group symmetry grouping for signed XDS observations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def symmetry_backend() -> str:
    """Return the available crystallographic symmetry backend."""

    try:
        import gemmi  # noqa: F401

        return "gemmi"
    except Exception:
        pass
    try:
        from cctbx import sgtbx  # noqa: F401

        return "cctbx"
    except Exception:
        return "none"


def require_symmetry_backend() -> str:
    """Require gemmi or cctbx for symmetry canonicalization."""

    backend = symmetry_backend()
    if backend == "none":
        raise RuntimeError(
            "No crystallographic symmetry backend is available. Install gemmi or run in an environment with cctbx."
        )
    return backend


@lru_cache(maxsize=None)
def operation_matrices(space_group_number: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Return rotational operations for the space group."""

    backend = require_symmetry_backend()
    if backend == "gemmi":
        return _gemmi_operation_matrices(space_group_number)
    return _cctbx_operation_matrices(space_group_number)


def canonical_hkl(hkl: tuple[int, int, int], space_group_number: int) -> tuple[int, int, int]:
    """Return lexicographic canonical representative under true SG rotations only."""

    h = np.asarray(hkl, dtype=int)
    mates = []
    for matrix_tuple in operation_matrices(space_group_number):
        matrix = np.asarray(matrix_tuple, dtype=int)
        mate = tuple(int(x) for x in matrix @ h)
        mates.append(mate)
    return min(mates)


def assign_symmetry_ids(observations: pd.DataFrame, space_group_number: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach canonical symmetry identifiers and group multiplicities."""

    backend = require_symmetry_backend()
    unique_hkls = observations[["h", "k", "l"]].drop_duplicates()
    canon_map: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for row in unique_hkls.itertuples(index=False):
        key = (int(row.h), int(row.k), int(row.l))
        canon_map[key] = canonical_hkl(key, space_group_number)
    out = observations.copy()
    canonical = [canon_map[(int(row.h), int(row.k), int(row.l))] for row in out.itertuples(index=False)]
    out["sym_h"] = [item[0] for item in canonical]
    out["sym_k"] = [item[1] for item in canonical]
    out["sym_l"] = [item[2] for item in canonical]
    out["symmetry_id"] = [f"{item[0]},{item[1]},{item[2]}" for item in canonical]
    multiplicities = out.groupby("symmetry_id", sort=True).size().rename("multiplicity")
    out = out.join(multiplicities, on="symmetry_id")
    groups = (
        out.groupby("symmetry_id", sort=True)
        .agg(
            sym_h=("sym_h", "first"),
            sym_k=("sym_k", "first"),
            sym_l=("sym_l", "first"),
            multiplicity=("observation_id", "size"),
            median_IOBS=("IOBS", "median"),
            median_S_risk=("S_risk", "median"),
            min_resolution=("resolution", "min"),
            max_resolution=("resolution", "max"),
        )
        .reset_index()
    )
    metadata = {
        "space_group_number": int(space_group_number),
        "symmetry_backend": backend,
        "operation_count": len(operation_matrices(space_group_number)),
        "friedel_policy": "No Friedel equivalence added beyond actual space-group rotational operations.",
    }
    return out, groups, metadata


def _gemmi_operation_matrices(space_group_number: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    import gemmi

    group = gemmi.SpaceGroup(int(space_group_number))
    matrices: list[tuple[tuple[int, int, int], ...]] = []
    for op in group.operations():
        if hasattr(op, "apply_to_hkl"):
            e1 = np.asarray(op.apply_to_hkl((1, 0, 0)), dtype=int)
            e2 = np.asarray(op.apply_to_hkl((0, 1, 0)), dtype=int)
            e3 = np.asarray(op.apply_to_hkl((0, 0, 1)), dtype=int)
            matrix = np.column_stack([e1, e2, e3])
        else:  # pragma: no cover - defensive for older gemmi
            rot = np.asarray(op.rot, dtype=float)
            den = float(getattr(op, "DEN", 1))
            matrix = np.rint(rot / den).astype(int)
        matrices.append(_matrix_tuple(matrix))
    return tuple(dict.fromkeys(matrices))


def _cctbx_operation_matrices(space_group_number: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    from cctbx import sgtbx

    group = sgtbx.space_group_info(number=int(space_group_number)).group()
    matrices: list[tuple[tuple[int, int, int], ...]] = []
    for op in group.all_ops():
        matrix = _matrix_from_hkl_expression(op.r().as_hkl())
        matrices.append(_matrix_tuple(matrix))
    return tuple(dict.fromkeys(matrices))


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(int(x) for x in row) for row in matrix)


def _matrix_from_hkl_expression(expression: str) -> np.ndarray:
    """Parse cctbx reciprocal-index expressions such as ``h,l,-k``."""

    rows: list[list[int]] = []
    for term in expression.replace(" ", "").split(","):
        coeffs = [0, 0, 0]
        sign = -1 if term.startswith("-") else 1
        symbol = term[1:] if term.startswith(("-", "+")) else term
        if symbol not in {"h", "k", "l"}:
            raise ValueError(f"Unsupported cctbx HKL operation term: {term}")
        coeffs[{"h": 0, "k": 1, "l": 2}[symbol]] = sign
        rows.append(coeffs)
    return np.asarray(rows, dtype=int)
