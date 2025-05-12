# ================================================================
# file: sered_scaler/scaling/bayesian_filter.py
# ================================================================

"""Bayesian Student‑t mixture weighting *chunked to avoid OOM*.

Memory blow‑ups happened because we built a (draws × N_obs) array.  We
now compute responsibilities in **100 000‑row batches**, so peak RAM is
`draws × 1e5 × 8 bytes ≈ 120 MB` even for draws=150.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t
import pymc as pm

from .provisional import provisional_scale

__all__ = ["mixture_filter"]


_BATCH = 100_000   # observations per chunk when computing weights


def mixture_filter(
    reflections: pd.DataFrame,
    *,
    max_iter: int = 8000,
    draws: int = 300,
    batch: int = _BATCH,
) -> pd.DataFrame:
    """Return reflections **copy** with `weight` column.

    Parameters
    ----------
    batch : int
        Number of observations processed at once when computing
        responsibilities.  Lower if you still hit "Killed".
    """
    # ---- deterministic μ_ij from provisional scale --------------------
    scale_res = provisional_scale(reflections)

    frames = reflections["event"].astype("category").cat.codes.to_numpy()
    hkl_cat = reflections[["h", "k", "l"]].astype("category")
    hkl_key = hkl_cat.apply(lambda s: s.cat.codes).astype(int)
    refl_compact = (
        hkl_key["h"] * (hkl_cat["k"].cat.categories.size * hkl_cat["l"].cat.categories.size)
        + hkl_key["k"] * hkl_cat["l"].cat.categories.size
        + hkl_key["l"]
    )

    mu_pred = scale_res.log_s[frames] + scale_res.log_F2[refl_compact]
    y_obs   = np.log(np.clip(reflections.I.to_numpy(float), 1e-3, None))

    # ---- ADVI mixture fit --------------------------------------------
    with pm.Model() as model:
        sigma_k = pm.HalfNormal("sigma_k", 1.0)
        sigma_d = pm.HalfNormal("sigma_d", 3.0)
        pi      = pm.Beta("pi", 1, 1)

        comp_k = pm.StudentT.dist(nu=4, mu=mu_pred, sigma=sigma_k)
        comp_d = pm.StudentT.dist(nu=4, mu=mu_pred, sigma=sigma_d)
        pm.Mixture("obs", w=[pi, 1-pi], comp_dists=[comp_k, comp_d], observed=y_obs)

        approx = pm.fit(max_iter, method="advi", progressbar=False)
        idata  = approx.sample(draws)

    post = idata.posterior[["pi", "sigma_k", "sigma_d"]].stack(sample=("chain", "draw"))
    pi_s   = post["pi"].values.astype(float)
    sigk_s = post["sigma_k"].values.astype(float)
    sigd_s = post["sigma_d"].values.astype(float)

    # ---- batched responsibility calculation --------------------------
    n_obs = len(reflections)
    weights = np.empty(n_obs, dtype=float)
    for start in range(0, n_obs, batch):
        end = min(start + batch, n_obs)
        mu_b = mu_pred[start:end]
        y_b  = y_obs[start:end]

        z_k = (y_b[None, :] - mu_b) / sigk_s[:, None]
        z_d = (y_b[None, :] - mu_b) / sigd_s[:, None]
        pk  = t.pdf(z_k, df=4) / sigk_s[:, None]
        pd  = t.pdf(z_d, df=4) / sigd_s[:, None]
        w   = (pi_s[:, None] * pk) / (pi_s[:, None] * pk + (1 - pi_s)[:, None] * pd)
        weights[start:end] = w.mean(axis=0)

    out = reflections.copy()
    out["weight"] = weights
    return out
