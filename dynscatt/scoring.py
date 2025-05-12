
# =====================================================================
# file: scoring.py  ── metric aggregation + weighting
# =====================================================================
from __future__ import annotations

from typing import Optional

import pandas as pd

from metrics import (
    equiv_spread,
    wilson_fit,
    wilson_deviation,
    intensity_rank_metric,
)

DEFAULT_WEIGHTS = {
    "m_equiv": 0.50,
    "m_wilson": 0.30,
    "m_rank": 0.20,
}


def add_metrics(
    df: pd.DataFrame,
    *,
    sg_symbol: str,
    cell,
    min_I: float = 0.0,
    min_sn: float = 0.0,
) -> pd.DataFrame:
    """Compute per‑reflection metric columns.

    `min_I`   – absolute intensity threshold (values ≤ ignored for Wilson fit)
    `min_sn`  – signal‑to‑noise (I/σ) threshold (ignored if 0)
    """
    df = df.copy()

    # -----------------------------------------------------------------
    # 1) symmetry spread (all reflections)
    # -----------------------------------------------------------------
    df["m_equiv"] = equiv_spread(df, sg_symbol)

    # -----------------------------------------------------------------
    # 2) Wilson deviation – *fit* only to reasonably strong reflections
    # -----------------------------------------------------------------
    mask = (df.I > min_I) & ((df.I / df.sigI) > min_sn)
    if mask.sum() < 10:  # sanity guard
        mask = df.I > df.I.median()
    scale, B = wilson_fit(df[mask], cell)
    df["m_wilson"] = wilson_deviation(df, cell, scale, B).abs()

    # -----------------------------------------------------------------
    # 3) rank metric (all reflections)
    # -----------------------------------------------------------------
    df["m_rank"] = intensity_rank_metric(df)

    return df


def combine_scores(df: pd.DataFrame, weights: Optional[dict[str, float]] = None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    score = sum(w[k] * df[k] for k in w)
    df = df.copy()
    df["dyn_score"] = score
    df["w_dyn"] = 1.0 / (1.0 + df.dyn_score)
    return df