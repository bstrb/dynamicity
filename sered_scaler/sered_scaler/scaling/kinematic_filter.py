# ================================================================
# file: sered_scaler/scaling/kinematic_filter.py
# ================================================================

"""Empirical Z‑score filter for discarding the most dynamical reflections."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from pandas import DataFrame

__all__ = ["zscore_filter", "add_zscore_column"]


def add_zscore_column(reflections: DataFrame, scale_result, *, cutoff: float = 2.5) -> None:
    """Attach *zscore* and *weight* columns **in‑place**.

    Parameters
    ----------
    reflections : DataFrame
        Table returned from *stream_to_dfs()* (or a subset after joins).
    scale_result : ScaleResult
        Output of *provisional_scale()*.
    cutoff : float, default 2.5
        |z| above which a reflection is considered non‑kinematic.
    """
    frames = reflections["event"].astype("category").cat.codes.to_numpy()
    hkl    = (reflections[["h","k","l"]]
              .astype("category")
              .apply(lambda s: s.cat.codes)
              .astype(int))
    refl_compact = (hkl["h"] * (hkl["k"].max()+1) * (hkl["l"].max()+1)
                    + hkl["k"] * (hkl["l"].max()+1)
                    + hkl["l"])

    pred = scale_result.log_s[frames] + scale_result.log_F2[refl_compact]
    z    = (np.log(np.clip(reflections.I.to_numpy(dtype=float), 1e-3, None)) - pred) / scale_result.model.scale_

    reflections["zscore"] = z
    reflections["weight"] = (np.abs(z) < cutoff).astype(float)


def zscore_filter(reflections: DataFrame, *, cutoff: float = 2.5) -> DataFrame:
    """Return **copy** of *reflections* with *weight* column (1 = kinematic)."""
    from .provisional import provisional_scale  # local import to avoid cycles

    scale_res = provisional_scale(reflections)
    reflections = reflections.copy()
    add_zscore_column(reflections, scale_res, cutoff=cutoff)
    return reflections
