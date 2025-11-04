# step2_bayes.py
# -----------------------------------------------------------------------------
# Step-2 Bayesian GP proposer: minimize posterior mean μ(x) of WRMSD in 2D.
# - Uses only SUCCESS points (dx,dy,wrmsd) for the GP.
# - Adds a soft failure penalty field (Gaussian bumps).
# - Normalizes inputs by R and outputs by median(y).
# - Pure NumPy; Matérn 3/2 kernel; analytic ∇μ; multi-start gradient descent
#   with backtracking; projection to disk; deterministic spacing slide.
#
# Public API:
#   - Step2BayesConfig
#   - propose_step2_bayes(successes_w, failures, tried, R, min_spacing_mm, cfg)
#
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np

# ------------------------- config -------------------------

@dataclass
class Step2BayesConfig:
    # Training set / normalization
    max_train_successes: int = 100      # use the best N successes by WRMSD
    wrmsd_eps: float = 1e-6             # to avoid 1/0 in scaling
    y_noise_frac: float = 0.03           # σ_n = y_scale * y_noise_frac (nugget)
    # Kernel: Matérn 3/2  k(r)= (1 + √3 r/ℓ) exp(-√3 r/ℓ)
    ell_scale: float = 0.6               # ℓ = ell_scale * median pairwise distance (in normalized coords)
    ell_min: float = 0.15
    # Failure penalty field (in *physical* mm, not normalized)
    fail_bump_sigma_fracR: float = 0.50  # σ = frac * R
    fail_bump_amp_frac: float = 0.08     # amplitude = frac * median(success wrmsd)
    # Optimizer
    gd_steps: int = 80
    gd_lr_init: float = 0.25
    gd_backtrack: float = 0.5
    gd_armijo: float = 1e-4
    gd_tol_mm: float = 0.0015
    # Seeds
    seed_top_k: int = 8                  # start from top-k best successes
    seed_extra_softbest: int = 10        # soft-argmin centroid from best M successes
    # Constraints
    stay_inside_R: bool = True
    spacing_slide_bisect_iters: int = 24
    spacing_slide_allow_ratio: float = 1.6  # slide if block < 1.6 * min_spacing


# ------------------------- helpers -------------------------

def _project_to_disk(x: np.ndarray, R: float) -> np.ndarray:
    r = float(np.linalg.norm(x))
    if r <= R or R <= 0:
        return x
    return x * (R / r)

def _respect_min_spacing(x: np.ndarray, tried: Optional[np.ndarray], min_spacing_mm: float) -> bool:
    if tried is None or tried.size == 0 or min_spacing_mm <= 0:
        return True
    d = np.linalg.norm(tried - x[None, :], axis=1)
    return bool(np.all(d >= (min_spacing_mm - 1e-12)))

def _slide_along_segment_to_feasible(a: np.ndarray, b: np.ndarray, tried: Optional[np.ndarray],
                                     min_spacing_mm: float, iters: int) -> np.ndarray:
    lo = a.copy(); hi = b.copy()
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if _respect_min_spacing(m, tried, min_spacing_mm):
            lo = m
        else:
            hi = m
    return lo

def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n <= 1:
        return np.array([0.0], float)
    out = []
    for i in range(n-1):
        out.append(np.linalg.norm(X[i+1:] - X[i], axis=1))
    return np.concatenate(out) if out else np.array([0.0], float)

def _soft_centroid_best_q(S: np.ndarray, y: np.ndarray, q: int) -> np.ndarray:
    q = max(1, min(q, S.shape[0]))
    order = np.argsort(y)[:q]
    Xq = S[order]
    w = 1.0 / (1e-6 + y[order])
    return (Xq * w[:, None]).sum(axis=0) / float(np.sum(w))

# ------------------------- kernel & GP -------------------------

def _matern32_and_grad(X: np.ndarray, x: np.ndarray, ell: float):
    """
    Compute Matérn 3/2 kernel vector k(x, X) and its gradient w.r.t. x.
    X: (N,2), x: (2,), ell>0, inputs are normalized coords.
    Returns k: (N,), dkdx: (N,2)
    """
    # r = ||x - X|| / ell
    d = X - x[None, :]
    r = np.linalg.norm(d, axis=1) + 1e-12
    s = np.sqrt(3.0) * (r / ell)
    k = (1.0 + s) * np.exp(-s)
    # ∂k/∂r = ∂/∂r [ (1+s) e^{-s} ] = - (3 r / ell^2) * (1/(√3 r/ell) - ???) -> derive carefully:
    # Derivation shortcut: k(r) = (1+s) e^{-s}, s = c r, c = √3/ell
    # dk/dr = c e^{-s} - (1+s) c e^{-s} = c e^{-s} (1 - 1 - s) = - c s e^{-s} = -(√3/ell) * s * e^{-s}
    # But s = c r, so dk/dr = - (√3/ell) * (c r) * e^{-s} = -(3 r / ell^2) * e^{-s}
    dk_dr = -(3.0 * r / (ell * ell)) * np.exp(-s)
    # ∇r = (x - X)/r with a minus sign because r = ||x - X||, ∂r/∂x = (x - X)/r
    # So ∇k = dk/dr * ∂r/∂x = dk/dr * (x - X)/r
    grad = (dk_dr / r)[:, None] * (x[None, :] - X)
    return k, grad

def _build_gp(Sn: np.ndarray, yn: np.ndarray, ell: float, sigma_n: float):
    """ Precompute (K + σ^2 I)^{-1} y and training X for posterior queries. """
    N = Sn.shape[0]
    # K_ij = k(||xi-xj||)
    d = Sn[:, None, :] - Sn[None, :, :]
    r = np.linalg.norm(d, axis=2) + 1e-12
    s = np.sqrt(3.0) * (r / ell)
    K = (1.0 + s) * np.exp(-s)
    K[np.arange(N), np.arange(N)] += (sigma_n * sigma_n)
    try:
        L = np.linalg.cholesky(K)
        # α = K^{-1} y via cho_solve
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        cho = ("chol", L)
    except np.linalg.LinAlgError:
        # Fallback to explicit inverse (small N)
        K_inv = np.linalg.pinv(K)
        alpha = K_inv @ yn
        cho = ("inv", K_inv)
    return alpha, cho

def _posterior_mean_and_grad(xn: np.ndarray, Sn: np.ndarray, alpha, cho, ell: float):
    # k(x,Sn) and grad w.r.t x
    k, grad_k = _matern32_and_grad(Sn, xn, ell)
    mu = k @ alpha
    # mu grad: ∇μ = (∂k/∂x)^T α
    g = grad_k.T @ alpha
    return float(mu), g

# ------------------------- penalty field -------------------------

def _failure_penalty(x_mm: np.ndarray,
                     failures: Optional[List[Tuple[float, float]]],
                     R: float,
                     amp: float,
                     sigma_fracR: float) -> Tuple[float, np.ndarray]:
    """Gaussian bumps centered at failures in *mm* coords. Return value and grad (in mm coords)."""
    if not failures:
        return 0.0, np.zeros(2, float)
    sigma = max(1e-9, sigma_fracR * R)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    val = 0.0
    grad = np.zeros(2, float)
    x = x_mm
    for fx, fy in failures:
        d = x - np.array([fx, fy], float)
        r2 = float(d[0]*d[0] + d[1]*d[1])
        e = math.exp(- r2 * inv2s2)
        val += e
        grad += e * (-d * (1.0 / (sigma * sigma)))
    return amp * val, amp * grad

# ------------------------- optimizer -------------------------

def _minimize_mu(start_x_mm: np.ndarray,
                 Sn_mm: np.ndarray,
                 yn: np.ndarray,
                 alpha, cho,
                 R: float,
                 ell: float,
                 y_scale: float,
                 failures: Optional[List[Tuple[float, float]]],
                 tried: Optional[np.ndarray],
                 min_spacing_mm: float,
                 cfg: Step2BayesConfig) -> Tuple[np.ndarray, float]:
    """
    Gradient descent with backtracking on f(x)= μ(x)/y_scale + penalty(x), in *mm* coords.
    """
    # Normalize helper (mm -> normalized)
    def to_norm(xmm): return xmm / max(R, 1e-12)
    def from_norm(xn): return xn * max(R, 1e-12)

    x = start_x_mm.copy()
    lr = cfg.gd_lr_init
    best_f = float("inf")
    best_x = x.copy()

    for _ in range(cfg.gd_steps):
        # μ and ∇μ in normalized coords
        xn = to_norm(x)
        mu, gmu_n = _posterior_mean_and_grad(xn, Sn_mm / max(R, 1e-12), alpha, cho, ell)
        # scale back gradient to *mm* coords: dμ/dx_mm = (1/R) * dμ/dx_n
        gmu_mm = gmu_n / max(R, 1e-12)

        # penalty (mm coords)
        p_val, p_grad = _failure_penalty(x, failures, R,
                                         amp=cfg.fail_bump_amp_frac * y_scale,
                                         sigma_fracR=cfg.fail_bump_sigma_fracR)

        f = (mu * y_scale) / y_scale + p_val  # simplify: f = mu + penalty; (mu already normalized by y_scale)
        g = gmu_mm + p_grad

        # track best
        if f < best_f:
            best_f = f
            best_x = x.copy()

        # Stop if gradient is tiny in mm
        if np.linalg.norm(g) < cfg.gd_tol_mm / max(R, 1e-12) * 0.5:
            break

        # Backtracking line search
        t = lr
        accept = False
        fx = f
        for _bt in range(12):
            cand = x - t * g
            # project to disk
            cand = _project_to_disk(cand, R)
            # evaluate objective at cand
            xn_c = to_norm(cand)
            mu_c, gmu_n_c = _posterior_mean_and_grad(xn_c, Sn_mm / max(R, 1e-12), alpha, cho, ell)
            p_val_c, _ = _failure_penalty(cand, failures, R,
                                          amp=cfg.fail_bump_amp_frac * y_scale,
                                          sigma_fracR=cfg.fail_bump_sigma_fracR)
            f_c = mu_c + p_val_c
            # Armijo
            if f_c <= fx - cfg.gd_armijo * t * np.dot(g, g):
                accept = True
                x = cand
                break
            t *= cfg.gd_backtrack

        if not accept:
            lr *= 0.5
            if lr < 1e-3:
                break

    # spacing handling (deterministic slide if lightly blocked)
    if not _respect_min_spacing(best_x, tried, min_spacing_mm):
        # slide from weighted centroid toward best_x until feasible
        # centroid of *training* points (mm)
        # weights inverse to y (smaller wrmsd => larger weight)
        w = 1.0 / (1e-6 + yn)
        seed = (Sn_mm * w[:, None]).sum(axis=0) / float(np.sum(w))
        # if it's only lightly blocked, slide deterministically
        if tried is not None and tried.size:
            d_block = float(np.min(np.linalg.norm(tried - best_x[None, :], axis=1)))
        else:
            d_block = float("inf")
        if np.isfinite(d_block) and d_block < cfg.spacing_slide_allow_ratio * min_spacing_mm:
            best_x = _slide_along_segment_to_feasible(seed, best_x, tried, min_spacing_mm, cfg.spacing_slide_bisect_iters)

    return best_x, best_f

# ------------------------- main API -------------------------

def propose_step2_bayes(
    successes_w: List[Tuple[float, float, float]],
    failures: Optional[List[Tuple[float, float]]] = None,
    tried: Optional[np.ndarray] = None,
    R: float = 0.05,
    min_spacing_mm: float = 0.001,
    cfg: Optional[Step2BayesConfig] = None,
) -> Tuple[float, float, str]:
    """
    Return (dx_mm, dy_mm, reason). Deterministic given inputs.
    """
    if cfg is None:
        cfg = Step2BayesConfig()

    if len(successes_w) == 0:
        return 0.0, 0.0, "step2_bayes_fallback_center_no_success"

    # Sort by WRMSD ascending; keep best N
    S_all = np.array([(dx, dy) for (dx, dy, _) in successes_w], float)
    y_all = np.array([wr for (_, _, wr) in successes_w], float)
    order = np.argsort(y_all)
    keep = order[: min(cfg.max_train_successes, len(order))]
    S = S_all[keep]
    y = y_all[keep]

    # Normalize outputs
    y_scale = float(np.median(y) + cfg.wrmsd_eps)
    yn = y / y_scale

    # Normalize inputs by R for GP lengthscale selection (we still optimize in mm)
    Sn_n = S / max(R, 1e-12)
    # median pairwise distance in normalized coords
    mpd = float(np.median(_pairwise_dists(Sn_n))) if Sn_n.shape[0] > 1 else 0.5
    ell = max(cfg.ell_min, cfg.ell_scale * max(mpd, 1e-6))

    # Nugget (noise)
    sigma_n = cfg.y_noise_frac

    # Precompute GP system
    alpha, cho = _build_gp(Sn_n, yn, ell, sigma_n)

    # Build seed set (in mm coords)
    seeds = []
    topk = min(cfg.seed_top_k, S.shape[0])
    # best K individual successes
    for j in range(topk):
        seeds.append(S[j])
    # soft-argmin centroid of best M
    M = min(cfg.seed_extra_softbest, S.shape[0])
    seeds.append(_soft_centroid_best_q(S, y, M))

    # Multi-start local descent; keep the best
    best_val = float("inf")
    best_pt = seeds[0].copy()
    for s in seeds:
        x_star, f_star = _minimize_mu(
            start_x_mm=np.array(s, float),
            Sn_mm=S,
            yn=yn,
            alpha=alpha,
            cho=cho,
            R=R,
            ell=ell,
            y_scale=y_scale,
            failures=failures,
            tried=tried,
            min_spacing_mm=min_spacing_mm,
            cfg=cfg,
        )
        if f_star < best_val:
            best_val = f_star
            best_pt = x_star.copy()

    # Keep inside disk and spacing
    if cfg.stay_inside_R:
        best_pt = _project_to_disk(best_pt, R)
    if not _respect_min_spacing(best_pt, tried, min_spacing_mm):
        # final deterministic slide from centroid
        w = 1.0 / (1e-6 + yn)
        seed = (S * w[:, None]).sum(axis=0) / float(np.sum(w))
        best_pt = _slide_along_segment_to_feasible(seed, best_pt, tried, min_spacing_mm, cfg.spacing_slide_bisect_iters)

    return float(best_pt[0]), float(best_pt[1]), "step2_bayes_jump"
