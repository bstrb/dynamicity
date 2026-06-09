"""Problematic zone-axis prediction from cell and candidate HKLs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import iter_primitive_triplets, low_order_prior_from_q, triplet_label


def unique_zone_axes(uvw_max: int) -> list[tuple[int, int, int]]:
    """Return sign-canonical primitive zone axes up to ``uvw_max``."""

    return list(iter_primitive_triplets(int(uvw_max)))


def predict_problematic_axes(
    candidates: pd.DataFrame,
    uvw_max: int = 5,
    low_order_g0_invA: float = 0.40,
    low_order_power: float = 1.5,
) -> pd.DataFrame:
    """Rank axes by ZOLZ density and low-order weighting from the cell only."""

    if candidates.empty:
        raise ValueError("Cannot predict axes without candidate HKLs.")
    hkl = candidates[["h", "k", "l"]].to_numpy(dtype=int)
    q = candidates["q_invA"].to_numpy(dtype=float)
    priors = low_order_prior_from_q(q, g0=low_order_g0_invA, power=low_order_power)

    rows: list[dict[str, float | int | str]] = []
    for axis in unique_zone_axes(uvw_max):
        uvw = np.asarray(axis, dtype=int)
        in_zone = (hkl @ uvw) == 0
        count = int(np.sum(in_zone))
        if count == 0:
            weighted = 0.0
            q_min = np.nan
        else:
            weighted = float(np.sum(priors[in_zone]))
            q_min = float(np.min(q[in_zone]))
        rows.append(
            {
                "u": axis[0],
                "v": axis[1],
                "w": axis[2],
                "axis_label": triplet_label(axis),
                "zolz_count": count,
                "zolz_weighted_count": weighted,
                "min_zolz_q_invA": q_min,
                "axis_complexity": abs(axis[0]) + abs(axis[1]) + abs(axis[2]),
            }
        )

    axes = pd.DataFrame.from_records(rows)
    max_weight = float(axes["zolz_weighted_count"].max()) if not axes.empty else 0.0
    axes["axis_score"] = 0.0 if max_weight <= 0.0 else axes["zolz_weighted_count"] / max_weight
    axes = axes.sort_values(
        ["axis_score", "zolz_weighted_count", "axis_complexity", "u", "v", "w"],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    axes["axis_rank"] = np.arange(1, len(axes) + 1, dtype=int)
    return axes


def mark_active_problematic_axes(
    axes: pd.DataFrame,
    max_problematic_axes: int | None = 50,
    axis_score_min: float = 0.0,
) -> pd.DataFrame:
    """Mark the ranked axes that are allowed to contribute to scoring."""

    if axes.empty:
        raise ValueError("No problematic axes are available to mark for scoring.")
    out = axes.copy()
    if "axis_rank" not in out:
        out = out.sort_values("axis_score", ascending=False).reset_index(drop=True)
        out["axis_rank"] = np.arange(1, len(out) + 1, dtype=int)
    active = out["axis_score"].to_numpy(dtype=float) >= float(axis_score_min)
    if max_problematic_axes is not None:
        active &= out["axis_rank"].to_numpy(dtype=int) <= int(max_problematic_axes)
    out["used_for_scoring"] = active
    if not bool(out["used_for_scoring"].any()):
        raise ValueError("No problematic axes passed max_problematic_axes/axis_score_min filters.")
    return out
