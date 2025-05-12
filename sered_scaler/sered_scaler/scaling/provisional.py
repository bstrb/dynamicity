
# ================================================================
# file: sered_scaler/scaling/provisional.py
# ================================================================

"""First‑pass robust log‑scale fit using *HuberRegressor*.

Exposes one public function – *provisional_scale()* – which returns a
``ScaleResult`` dataclass holding per‑frame scale factors and provisional
log(F²) values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pandas import DataFrame
from sklearn.linear_model import HuberRegressor
from scipy import sparse

from ..utils.design_matrix import one_hot, stack

__all__ = ["ScaleResult", "provisional_scale"]

@dataclass
class ScaleResult:
    log_s: np.ndarray  # per‑frame log‑scale
    log_F2: np.ndarray  # per‑reflection log‑F²
    model: HuberRegressor  # fitted model (kept for .scale_ and later reuse)


def provisional_scale(reflections: DataFrame) -> ScaleResult:
    """Robustly scale *reflections* across frames.

    Parameters
    ----------
    reflections : DataFrame
        Must at least contain columns ``event, h, k, l, I``.

    Returns
    -------
    ScaleResult
        Dataclass with fitted parameters and the underlying model.
    """
    # --- encode categorical factors ---------------------------------------------------
    frames = reflections["event"].astype("category").cat.codes  # int key 0..n_f-1
    hkl    = reflections[["h", "k", "l"]].astype("category")
    hkl_key = hkl.apply(lambda s: s.cat.codes).astype(int)
    # build a unique integer per (h,k,l)
    refl_compact = (hkl_key["h"] * (hkl["k"].cat.categories.size * hkl["l"].cat.categories.size)
                    + hkl_key["k"] * hkl["l"].cat.categories.size
                    + hkl_key["l"])

    n_f = frames.max() + 1
    n_r = refl_compact.max() + 1

    X = stack([
        one_hot(frames, n_f),
        one_hot(refl_compact, n_r)
    ]).tocsr()

    y = np.log(np.clip(reflections.I.to_numpy(dtype=float), 1e-3, None))

    model = HuberRegressor()
    model.fit(X, y)

    log_params = model.coef_
    log_s  = log_params[:n_f]
    log_F2 = log_params[n_f:]

    return ScaleResult(log_s=log_s, log_F2=log_F2, model=model)
