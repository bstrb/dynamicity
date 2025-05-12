
# =====================================================================
# file: metrics.py  ── dynamical metrics
# =====================================================================
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss

from symmetry import assign_equiv_classes

# ---------------------------------------------------------------------
# 1) Symmetry‑equivalent intensity spread
# ---------------------------------------------------------------------

def equiv_spread(df: pd.DataFrame, sg_symbol: str) -> pd.Series:
    df = assign_equiv_classes(df, sg_symbol)
    mean_I = df.groupby("equiv_id")["I"].transform("mean")
    return (df.I - mean_I).abs() / mean_I.clip(lower=1e-6)

# ---------------------------------------------------------------------
# 2) Wilson plot (⟨I⟩ vs. s²) deviation
# ---------------------------------------------------------------------

try:
    import gemmi
except ImportError:
    gemmi = None


def _calc_d_spacing(row: pd.Series, cell: Tuple[float, float, float, float, float, float]) -> float:
    """d‑spacing in Å for reflection (h,k,l).  Uses Gemmi if present, otherwise
    orthorhombic approximation (angles 90°)."""
    a, b, c, al, be, ga = cell
    if gemmi:
        uc = gemmi.UnitCell(a, b, c, al, be, ga)
        return uc.calculate_d((int(row.h), int(row.k), int(row.l)))
    # fallback (approx)
    return 1.0 / math.sqrt(
        (row.h ** 2) / a ** 2 +
        (row.k ** 2) / b ** 2 +
        (row.l ** 2) / c ** 2
    )


def wilson_fit(df: pd.DataFrame, cell: Tuple[float, float, float, float, float, float]):
    s2 = []
    lnI = []
    for _, r in df.iterrows():
        d = _calc_d_spacing(r, cell)
        s2.append((1.0 / (2.0 * d)) ** 2)
        lnI.append(math.log(max(r.I, 1e-6)))
    A = np.vstack([np.ones_like(s2), -2.0 * np.array(s2)]).T
    ln_scale, B = np.linalg.lstsq(A, np.array(lnI), rcond=None)[0]
    return math.exp(ln_scale), B


def wilson_deviation(df: pd.DataFrame, cell, scale, B):
    dev = []
    for _, r in df.iterrows():
        d = _calc_d_spacing(r, cell)
        s2 = (1.0 / (2.0 * d)) ** 2
        I_pred = scale * math.exp(-2.0 * B * s2)
        dev.append((r.I - I_pred) / I_pred)
    return pd.Series(dev, index=df.index)

# ---------------------------------------------------------------------
# 3) Rank‑based strong‑beam metric
# ---------------------------------------------------------------------

def intensity_rank_metric(df: pd.DataFrame) -> pd.Series:
    ranks = ss.rankdata(df.I.values, method="average") / len(df)
    z = ss.norm.ppf(ranks.clip(1e-4, 1-1e-4))
    return z / z.max()