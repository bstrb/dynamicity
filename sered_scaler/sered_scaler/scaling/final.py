
# ================================================================
# file: sered_scaler/scaling/final.py
# ================================================================

"""Weighted re‑scale & merge to produce kinematic F² table."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import Ridge
from scipy import sparse

from ..utils.design_matrix import one_hot, stack

__all__ = ["weighted_merge"]

def weighted_merge(reflections: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Run WLS on *reflections* that already carry a *weight* column.

    Returns
    -------
    merged : DataFrame
        Unique (h,k,l) rows with *F2* column (merged kinematic intensity).
    frame_scales : DataFrame
        Per‑frame scale factors (s_i) for diagnostic plots.
    """
    if "weight" not in reflections.columns:
        raise ValueError("reflections must have a 'weight' column (run kinematic filter first)")

    frames = reflections["event"].astype("category").cat.codes
    hkl    = reflections[["h","k","l"]].astype("category")
    hkl_key = hkl.apply(lambda s: s.cat.codes).astype(int)
    refl_compact = (hkl_key["h"] * (hkl["k"].cat.categories.size * hkl["l"].cat.categories.size)
                    + hkl_key["k"] * hkl["l"].cat.categories.size
                    + hkl_key["l"])

    n_f = frames.max() + 1
    n_r = refl_compact.max() + 1

    X = stack([
        one_hot(frames, n_f),
        one_hot(refl_compact, n_r)
    ]).tocsr()
    y = np.log(np.clip(reflections.I.to_numpy(float), 1e-3, None))
    W = sparse.diags(reflections.weight.to_numpy(float))

    model = Ridge(alpha=1e-3, fit_intercept=False)
    model.fit(W @ X, W @ y)

    log_params = model.coef_
    log_s  = log_params[:n_f]
    log_F2 = log_params[n_f:]

    # ── build output tables ───────────────────────────────────────────────
    frame_scales = (
        reflections[["event"]]
        .drop_duplicates()
        .assign(scale=np.exp(log_s))
        .reset_index(drop=True)
    )

    def _safe_avg(group: DataFrame):
        w   = group.weight.to_numpy(float)
        f2  = group.F2.to_numpy(float)
        return f2.mean() if w.sum() == 0 else np.average(f2, weights=w)

    # merged = (
    #     reflections.assign(F2=np.exp(log_F2[refl_compact]))
    #     .groupby(["h", "k", "l"], as_index=False)
    #     .apply(_safe_avg)
    #     .rename(columns={0: "F2"})
    # )
    group = reflections.assign(
        F2=np.exp(log_F2[refl_compact])
    ).groupby(["h","k","l"])

    def mean_sigma(group: DataFrame):
        w  = group.weight.to_numpy(float)
        f  = group.F2.to_numpy(float)
        mu = np.average(f, weights=w) if w.sum() else f.mean()

        if "weight_var" in group.columns:
            wvar = group.weight_var.to_numpy(float)
            num  = ((f - mu) ** 2 * wvar).sum()
            den  = (w.sum() ** 2) if w.sum() else 1.0
            sig  = np.sqrt(num / den) if den else 0.05 * mu
        else:  # z‑score path
            sig = 0.05 * mu

        return pd.Series({"F2": mu, "sigF2": sig})

    # def mean_and_sig(d):
    #     w = d.weight.to_numpy(float)
    #     f = d.F2.to_numpy(float)
    #     mu = np.average(f, weights=w) if w.sum() else f.mean()
    #     var = np.average((f-mu)**2, weights=w) if w.sum() > 1 else 0.0
    #     return pd.Series({"F2": mu, "sigF2": var**0.5})

    merged = group.apply(mean_sigma).reset_index()

    return merged, frame_scales
