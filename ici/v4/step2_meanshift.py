# step2_meanshift.py
# -----------------------------------------------------------------------------
# Mean-shift Step-2 for center proposal:
#   - Operates only on SUCCESS points (dx,dy, wrmsd)
#   - Quickly jumps to the densest, lowest-wrmsd mode of the local cluster
#   - Robust to arcs and unstable Hessians
#
# Public API:
#   - Step2MeanShiftConfig: lightweight config dataclass
#   - propose_step2_meanshift(successes_w, failures, tried, R, min_spacing_mm, rng, cfg)
#
# Where:
#   successes_w : list[tuple[float, float, float]]  # (dx, dy, wrmsd) for successes only
#   failures    : list[tuple[float, float]]         # (dx, dy) for failed trials (optional)
#   tried       : np.ndarray shape (N,2) of all already-tried (dx,dy) points (optional)
#   R           : float  # search radius (mm)
#   min_spacing_mm : float # min spacing (mm)
#   rng         : np.random.Generator | random.Random | None
#   cfg         : Step2MeanShiftConfig
#
# Returns:
#   (ndx, ndy, reason: str)
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import random

import numpy as np


@dataclass
class Step2MeanShiftConfig:
    # Neighborhood / selection
    k_nearest: int = 40            # how many nearest successes to use
    q_best_for_seed: int = 12      # how many best by wrmsd for centroid seed

    # Weights: wrmsd -> weight
    wrmsd_eps: float = 0.02        # stabilizer to avoid 1/0
    wrmsd_power: float = 2.0       # w_i ~ 1 / (eps + wrmsd_i)^p

    # Kernel bandwidth
    bandwidth_scale: float = 1.3   # h = scale * median pairwise distance in the K set
    min_bandwidth: float = 1e-6    # hard floor (mm) to avoid zero

    # Iterations / stopping
    max_iters: int = 6
    tol_mm: float = 0.003          # stop if move < tol_mm

    # Spacing handling
    jitter_trials: int = 3         # if spacing blocks, try up to N tiny jitters
    jitter_sigma_frac: float = 0.5 # jitter sigma = 0.5 * min_spacing

    # Disk clamp
    stay_inside_R: bool = True


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    # Returns upper-triangular distances (flattened) to estimate a robust scale
    n = X.shape[0]
    if n <= 1:
        return np.array([0.0], dtype=float)
    dists = []
    for i in range(n - 1):
        di = np.linalg.norm(X[i+1:] - X[i], axis=1)
        dists.append(di)
    return np.concatenate(dists) if dists else np.array([0.0], dtype=float)


def _k_nearest_indices(X: np.ndarray, x0: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(X - x0, axis=1)
    idx = np.argpartition(d, min(k, len(d)-1))[:k]
    # sort the selected ones by distance
    idx = idx[np.argsort(d[idx])]
    return idx


def _soft_centroid_best_q(successes: np.ndarray, wrmsd: np.ndarray, q: int, eps: float) -> np.ndarray:
    """ Weighted centroid of the best-q by WRMSD with weights 1/(eps+wrmsd). """
    q = min(q, successes.shape[0])
    order = np.argsort(wrmsd)[:q]
    Xq = successes[order]
    Wq = 1.0 / (eps + wrmsd[order])
    wsum = float(np.sum(Wq))
    if wsum <= 0:
        return Xq.mean(axis=0)
    return (Xq * Wq[:, None]).sum(axis=0) / wsum


def _gauss(u: np.ndarray) -> np.ndarray:
    # standard Gaussian kernel K(u) = exp(-0.5 u^2)
    return np.exp(-0.5 * (u ** 2))


def _meanshift(X: np.ndarray, w: np.ndarray, h: float, x0: np.ndarray,
               tol: float, max_iters: int) -> np.ndarray:
    """One cluster mean-shift starting at x0 with weights w and bandwidth h."""
    x = x0.copy()
    h = max(h, 1e-12)
    inv_h = 1.0 / h
    for _ in range(max_iters):
        u = np.linalg.norm((X - x), axis=1) * inv_h
        k = _gauss(u)
        wk = w * k
        s = float(np.sum(wk))
        if s <= 0:
            break
        x_new = (X * wk[:, None]).sum(axis=0) / s
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def _project_to_disk(x: np.ndarray, R: float) -> np.ndarray:
    r = float(np.linalg.norm(x))
    if r <= R:
        return x
    return x * (R / r)


def _respect_min_spacing(x: np.ndarray, tried: Optional[np.ndarray], min_spacing_mm: float) -> bool:
    if tried is None or len(tried) == 0:
        return True
    d = np.linalg.norm(tried - x[None, :], axis=1)
    return bool(np.all(d >= min_spacing_mm - 1e-12))


def _maybe_jitter_to_clear_spacing(x: np.ndarray, tried: Optional[np.ndarray],
                                   min_spacing_mm: float,
                                   rng: Optional[object],
                                   max_trials: int,
                                   sigma_frac: float) -> Optional[np.ndarray]:
    if _respect_min_spacing(x, tried, min_spacing_mm):
        return x
    if max_trials <= 0:
        return None
    # RNG adapter
    if rng is None:
        rr = random.random
        rn = lambda: random.gauss(0.0, 1.0)
    elif hasattr(rng, "normal"):
        rr = rng.random if hasattr(rng, "random") else random.random
        rn = lambda: float(rng.normal(0.0, 1.0))
    else:
        rr = rng.random
        rn = lambda: random.gauss(0.0, 1.0)

    sigma = sigma_frac * min_spacing_mm
    for _ in range(max_trials):
        xj = x + np.array([rn() * sigma, rn() * sigma], dtype=float)
        if _respect_min_spacing(xj, tried, min_spacing_mm):
            return xj
    return None


def propose_step2_meanshift(
    successes_w: List[Tuple[float, float, float]],
    failures: Optional[List[Tuple[float, float]]] = None,
    tried: Optional[np.ndarray] = None,
    R: float = 0.05,
    min_spacing_mm: float = 0.001,
    rng: Optional[object] = None,
    cfg: Optional[Step2MeanShiftConfig] = None,
) -> Tuple[float, float, str]:
    """
    Mean-shift Step-2 proposal. Returns (dx, dy, reason).
    """
    if cfg is None:
        cfg = Step2MeanShiftConfig()

    if len(successes_w) == 0:
        # Fallback: if no successes, propose small random step toward center
        x = np.zeros(2, dtype=float)
        x = _maybe_jitter_to_clear_spacing(x, tried, min_spacing_mm, rng,
                                           cfg.jitter_trials, cfg.jitter_sigma_frac) or x
        return float(x[0]), float(x[1]), "step2_ms_fallback_no_success"

    # Prepare arrays
    S = np.array([(dx, dy) for (dx, dy, _) in successes_w], dtype=float)
    W = np.array([wr for (_, _, wr) in successes_w], dtype=float)

    # Seed at weighted centroid of best-q by wrmsd
    x_seed = _soft_centroid_best_q(S, W, cfg.q_best_for_seed, cfg.wrmsd_eps)

    # Find K nearest successes to the seed (local neighborhood)
    if S.shape[0] <= cfg.k_nearest:
        idx_loc = np.arange(S.shape[0], dtype=int)
    else:
        idx_loc = _k_nearest_indices(S, x_seed, cfg.k_nearest)

    Xloc = S[idx_loc]
    Wloc = W[idx_loc]

    # Compute weights for mean-shift: w_i = 1/(eps + wrmsd)^p
    w_ms = 1.0 / np.power(cfg.wrmsd_eps + np.maximum(Wloc, 0.0), cfg.wrmsd_power)

    # Bandwidth from pairwise scale
    dists = _pairwise_dists(Xloc)
    med = float(np.median(dists)) if dists.size else 0.0
    h = max(cfg.bandwidth_scale * max(med, 1e-12), cfg.min_bandwidth)

    # Run mean-shift starting at x_seed
    x_star = _meanshift(Xloc, w_ms, h, x_seed, tol=cfg.tol_mm, max_iters=cfg.max_iters)

    # Enforce search disk
    if cfg.stay_inside_R:
        x_star = _project_to_disk(x_star, R)

    # Respect spacing, with a couple of tiny jitters if needed
    x_star2 = _maybe_jitter_to_clear_spacing(
        x_star, tried, min_spacing_mm, rng, cfg.jitter_trials, cfg.jitter_sigma_frac
    )
    if x_star2 is None:
        # As a fallback, shrink bandwidth and try once more
        h2 = max(0.7 * h, cfg.min_bandwidth)
        x_star_alt = _meanshift(Xloc, w_ms, h2, x_seed, tol=cfg.tol_mm, max_iters=cfg.max_iters)
        if cfg.stay_inside_R:
            x_star_alt = _project_to_disk(x_star_alt, R)
        x_star2 = _maybe_jitter_to_clear_spacing(
            x_star_alt, tried, min_spacing_mm, rng, cfg.jitter_trials, cfg.jitter_sigma_frac
        )

    if x_star2 is None:
        # If still blocked, return the nearest allowed point on spacing boundary:
        # move from x_seed toward x_star until just above spacing threshold.
        # Simple bisection along the segment.
        if tried is None or tried.shape[0] == 0:
            x_ret = x_star
        else:
            a = x_seed.copy()
            b = x_star.copy()
            for _ in range(20):
                m = 0.5 * (a + b)
                if _respect_min_spacing(m, tried, min_spacing_mm):
                    a = m
                else:
                    b = m
            x_ret = a
        reason = "step2_ms_spacing_clamped"
        return float(x_ret[0]), float(x_ret[1]), reason

    reason = "step2_meanshift_jump"
    return float(x_star2[0]), float(x_star2[1]), reason
