#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts_hillmap.py — CSV parser + two‑phase proposer (HillMap → Gradient Mapper)

Overview
--------
Step 1 (HillMap exploration; few parameters):
  • Start with a wide, shallow 2D Gaussian HILL centered at the FIRST tried shift of the event.
    - This encodes the prior belief of higher index probability near that start.
  • Each SUCCESSFUL index adds another Gaussian HILL (same σ as failures).
  • Each UNSUCCESSFUL index adds a smaller Gaussian DENT (negative amplitude) at that point.
  • Next probe is drawn RANDOMLY but PROPORTIONALLY to this landscape
    (with a small global exploration floor and a minimum spacing to avoid duplicates).
  • Continue until we have N_success successes → switch to Step 2.

Step 2 (Local wRMSD mapping via quadratic gradient search):
  • Fit a local quadratic surface w(x,y) ≈ a x^2 + b y^2 + c xy + d x + e y + f to the SUCCESS points.
  • Move from the current best success along the NEGATIVE GRADIENT of this quadratic
    by a short step (fraction of radius). This is robust even if successes lie on one side.
  • If the last few proposals were unindexed, fall back to Step 1 (tunable small integer).
  • All moves obey a circular bound of radius R (points outside are projected to the disk).

CSV format preserved:
  "#/abs/path event <int>"
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
"""

from __future__ import annotations
import argparse, os, sys, math
from typing import Dict, List, Tuple, Optional
import numpy as np

# -------------------------- CSV utils --------------------------

def _fmt6(x: float) -> str:
    return f"{x:.6f}"

def _float_or_blank(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "":
        return None
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def parse_log(log_path: str):
    entries = []
    latest_run = -1
    with open(log_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        current_key = None
        for ln in f:
            if ln.startswith("#/"):
                try:
                    path_part, ev_part = ln[1:].rsplit(" event ", 1)
                    ev = int(ev_part.strip())
                    current_key = (os.path.abspath(path_part.strip()), ev)
                    entries.append((current_key, None))
                except Exception:
                    entries.append((None, ("RAW", ln.rstrip("\n"))))
                continue
            parts = [p.strip() for p in ln.rstrip("\n").split(",")]
            while len(parts) < 7:
                parts.append("")
            run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = parts[:7]
            try:
                run_n = int(run_s)
                latest_run = max(latest_run, run_n)
            except Exception:
                entries.append((None, ("RAW", ln.rstrip("\n"))))
                continue
            entries.append((None, (run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s)))
    return entries, latest_run

def write_log(log_path: str, entries) -> None:
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm\n")
        for key, row in entries:
            if key is not None:
                f.write("#" + key[0] + f" event {key[1]}\n")
            elif row is not None and len(row) >= 2 and row[0] == "RAW":
                f.write(row[1] + "\n")
            elif row is not None:
                f.write(",".join(row) + "\n")
    os.replace(tmp_path, log_path)

# -------------------------- Geometry & kernels --------------------------

def _uniform_points_in_disk(rng: np.random.Generator, n: int, R: float) -> np.ndarray:
    """Uniform random points in a disk of radius R centered at (0,0)."""
    u = rng.random(n)
    r = R * np.sqrt(u)
    theta = rng.random(n) * 2.0 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1).astype(float)

def _gauss2(xy: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """2D isotropic Gaussian G(x)=exp(-||x-c||^2 / (2σ^2)); vectorized for xy shape (N,2)."""
    if sigma <= 0:
        return np.zeros((xy.shape[0],), dtype=float)
    d2 = np.sum((xy - center[None, :])**2, axis=1)
    return np.exp(-0.5 * d2 / (sigma * sigma))

def _min_dist(xy: np.ndarray, tried: np.ndarray) -> np.ndarray:
    """Return min distance from each xy[i] to any point in tried (shape (M,2))."""
    if tried.size == 0:
        return np.full((xy.shape[0],), np.inf, dtype=float)
    diff = xy[:, None, :] - tried[None, :, :]
    d2 = np.sum(diff * diff, axis=2)  # (N, M)
    return np.sqrt(np.min(d2, axis=1))

def _project_to_disk(p: np.ndarray, R: float) -> np.ndarray:
    n = np.hypot(p[0], p[1])
    if n <= R or n == 0.0:
        return p
    return p * (R / n)

# -------------------------- Quadratic fit for Step 2 --------------------------

def _fit_quadratic(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Fit w ≈ a x^2 + b y^2 + c x y + d x + e y + f using least squares.
    Returns coefficients [a, b, c, d, e, f] and a flag indicating if the fit is well-conditioned.
    """
    n = X.shape[0]
    A = np.column_stack([X[:,0]**2, X[:,1]**2, X[:,0]*X[:,1], X[:,0], X[:,1], np.ones(n)])
    try:
        beta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        ok = (rank == 6)
        return beta, ok
    except np.linalg.LinAlgError:
        return np.zeros(6, float), False

def _quad_gradient(beta: np.ndarray, x: float, y: float) -> np.ndarray:
    a, b, c, d, e, f = beta
    gx = 2.0*a*x + c*y + d
    gy = c*x + 2.0*b*y + e
    return np.array([gx, gy], float)

def _quad_minimizer(beta: np.ndarray) -> Optional[np.ndarray]:
    a, b, c, d, e, f = beta
    H = np.array([[2.0*a, c],[c, 2.0*b]], float)
    g = np.array([d, e], float)
    # Solve H [x;y] + g = 0  ->  [x;y] = - H^{-1} g
    try:
        sol = -np.linalg.solve(H, g)
        return sol
    except np.linalg.LinAlgError:
        return None

def _is_pd_hessian(beta: np.ndarray) -> bool:
    a, b, c, d, e, f = beta
    # Hessian positive definite: a>0, b>0, and 4ab - c^2 > 0
    return (a > 0.0) and (b > 0.0) and ((4.0*a*b - c*c) > 1e-12)

# -------------------------- Step 1: HillMap sampler --------------------------

def _step1_hillmap(
    rng: np.random.Generator,
    successes: List[Tuple[float, float]],
    failures: List[Tuple[float, float]],
    first_shift: Tuple[float, float],
    R: float,
    n_cand: int,
    sigma0: float,
    sigma: float,
    amp_prior: float,
    amp_success: float,
    amp_fail: float,
    explore_floor: float,
    min_spacing: float,
    tried: np.ndarray,
) -> Tuple[float, float, str]:
    """
    Build probability field: F = amp_prior * G0 + sum(amp_success * G_succ) - sum(amp_fail * G_fail)
    Sample candidate proportional to max(F,0) + explore_floor, with min spacing.
    """
    C = _uniform_points_in_disk(rng, n_cand, R)
    # Min spacing
    dmin = _min_dist(C, tried)
    mask = dmin >= (min_spacing - 1e-12)
    if not np.any(mask):
        mask = dmin >= 0.0
    C = C[mask]
    if C.shape[0] == 0:
        return 0.0, 0.0, "no_candidates"

    F = np.zeros((C.shape[0],), dtype=float)
    # Prior hill at first shift
    c0 = np.array([first_shift[0], first_shift[1]], float)
    F += amp_prior * _gauss2(C, c0, sigma0)

    # Success hills
    if successes:
        S = np.array(successes, float)
        for i in range(S.shape[0]):
            F += amp_success * _gauss2(C, S[i, :], sigma)

    # Failure dents
    if failures:
        for fx, fy in failures:
            F -= amp_fail * _gauss2(C, np.array([fx, fy], float), sigma)

    # Convert to weights with exploration floor
    w = np.maximum(F, 0.0) + float(explore_floor)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        j = int(rng.integers(0, C.shape[0]))
        return float(C[j,0]), float(C[j,1]), "step1_uniform_fallback"

    w = w / s
    j = int(rng.choice(C.shape[0], p=w))
    return float(C[j,0]), float(C[j,1]), "step1_hillmap"

# -------------------------- Step 2: Gradient mapper --------------------------

def _recent_unindexed_streak(trials_sorted: List[Tuple[int,float,float,int,Optional[float]]]) -> int:
    streak = 0
    for _, _, _, idx, _ in reversed(trials_sorted):
        if idx == 0:
            streak += 1
        else:
            break
    return streak

def _step2_gradient_mapper(
    rng: np.random.Generator,
    successes_w: List[Tuple[float, float, float]],
    R: float,
    step_frac: float,
    min_spacing: float,
    tried: np.ndarray,
) -> Tuple[float, float, str]:
    """
    Fit quadratic to success points; step from incumbent along negative gradient by step_frac * R.
    If quadratic is well-conditioned and PD, optionally move directly toward its minimizer (clipped).
    """
    X = np.array([[x, y] for (x, y, _) in successes_w], float)
    y = np.array([w for (_, _, w) in successes_w], float)

    # Best incumbent
    jbest = int(np.argmin(y))
    xb = X[jbest, :].copy()

    beta, ok = _fit_quadratic(X, y)
    if not ok:
        # fallback: small jitter around incumbent
        for _ in range(64):
            step = step_frac * R
            jitter = rng.standard_normal(2)
            jitter = jitter / (np.linalg.norm(jitter) + 1e-12)
            cand = xb - step * jitter
            cand = _project_to_disk(cand, R)
            if _min_dist(cand[None,:], tried)[0] >= min_spacing:
                return float(cand[0]), float(cand[1]), "step2_quad_fit_fallback_jitter"
        return float(xb[0]), float(xb[1]), "step2_quad_fit_fallback_incumbent"

    # Prefer moving toward quadratic minimizer if PD & inside disk
    reason = "step2_grad_descent"
    if _is_pd_hessian(beta):
        xm = _quad_minimizer(beta)
        if xm is not None and np.isfinite(xm).all():
            xm = _project_to_disk(xm, R)
            # Take a step toward xm from xb
            direction = xm - xb
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                step = step_frac * R
                move = direction / norm * min(step, norm)
                cand = xb + move
                if _min_dist(cand[None,:], tried)[0] >= min_spacing:
                    return float(cand[0]), float(cand[1]), "step2_toward_quad_min"
            # If too close or same point, fall through to gradient step

    # Gradient at incumbent, step opposite gradient
    g = _quad_gradient(beta, xb[0], xb[1])
    gnorm = float(np.linalg.norm(g))
    if gnorm < 1e-12:
        # tiny gradient; nudge in random direction
        for _ in range(64):
            step = step_frac * R
            jitter = rng.standard_normal(2)
            jitter = jitter / (np.linalg.norm(jitter) + 1e-12)
            cand = xb - step * jitter
            cand = _project_to_disk(cand, R)
            if _min_dist(cand[None,:], tried)[0] >= min_spacing:
                return float(cand[0]), float(cand[1]), "step2_grad_tiny_random"
        return float(xb[0]), float(xb[1]), "step2_grad_tiny_incumbent"
    direction = - g / gnorm
    cand = xb + (step_frac * R) * direction
    cand = _project_to_disk(cand, R)
    if _min_dist(cand[None,:], tried)[0] >= min_spacing:
        return float(cand[0]), float(cand[1]), reason

    # If spacing violated, try rotating direction a bit to find a free spot
    angle = np.arctan2(direction[1], direction[0])
    for k in range(1, 13):
        th = angle + (k * (np.pi / 12.0))  # +/-15°, 30°, ...
        alt_dir = np.array([np.cos(th), np.sin(th)], float)
        cand = xb + (step_frac * R) * alt_dir
        cand = _project_to_disk(cand, R)
        if _min_dist(cand[None,:], tried)[0] >= min_spacing:
            return float(cand[0]), float(cand[1]), "step2_grad_rotated"
    return float(xb[0]), float(xb[1]), "step2_no_free_dir"

# -------------------------- Main per-event proposal --------------------------

def propose_event_hillmap(
    trials_sorted: List[Tuple[int,float,float,int,Optional[float]]],
    R: float,
    rng: np.random.Generator,
    # Step 1 params
    N_success_to_step2: int,
    sigma0_frac: float,
    sigma_frac: float,
    prior_amp_frac: float,
    success_amp: float,
    fail_amp_frac: float,
    step1_candidates: int,
    explore_floor: float,
    min_spacing: float,
    # Step 2 params
    step2_step_frac: float,
    back_to_step1_streak: int,
) -> Tuple[float, float, str]:
    """Return (next_dx, next_dy, reason_str)."""
    tried = np.array([[dx, dy] for (_, dx, dy, _, _) in trials_sorted], float) if len(trials_sorted)>0 else np.zeros((0,2), float)
    successes = []
    successes_w = []
    failures = []
    for (_, dx, dy, indexed, wr) in trials_sorted:
        if indexed == 1 and wr is not None and math.isfinite(wr):
            successes.append((dx, dy))
            successes_w.append((dx, dy, float(wr)))
        elif indexed == 0:
            failures.append((dx, dy))

    # First tried shift for prior center
    if len(trials_sorted) > 0:
        first_dx, first_dy = trials_sorted[0][1], trials_sorted[0][2]
    else:
        first_dx, first_dy = 0.0, 0.0

    sigma0 = float(max(1e-12, sigma0_frac * R))
    sigma = float(max(1e-12, sigma_frac * R))
    amp_prior = float(prior_amp_frac) * float(success_amp)
    amp_success = float(success_amp)
    amp_fail = float(fail_amp_frac) * float(success_amp)

    # Step 2 conditions
    if len(successes_w) >= max(3, N_success_to_step2) and _recent_unindexed_streak(trials_sorted) < back_to_step1_streak:
        ndx, ndy, r = _step2_gradient_mapper(
            rng=rng,
            successes_w=successes_w,
            R=R,
            step_frac=step2_step_frac,
            min_spacing=min_spacing,
            tried=tried,
        )
        return ndx, ndy, r

    # Step 1
    ndx, ndy, r = _step1_hillmap(
        rng=rng,
        successes=successes,
        failures=failures,
        first_shift=(first_dx, first_dy),
        R=R,
        n_cand=step1_candidates,
        sigma0=sigma0,
        sigma=sigma,
        amp_prior=amp_prior,
        amp_success=amp_success,
        amp_fail=amp_fail,
        explore_floor=explore_floor,
        min_spacing=min_spacing,
        tried=tried,
    )
    return ndx, ndy, r

# -------------------------- CLI --------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-event proposer: HillMap (success-hills + failure-dents) + Quadratic Gradient Mapper."
    )
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Circular bound radius for |(dx,dy)|.")
    ap.add_argument("--seed", type=int, default=1337)

    # Step 1 (HillMap) — minimal, well-described knobs
    ap.add_argument("--step1-N-success", type=int, default=3,
                    help="Number of successes required to enter Step 2 (>=3). Lower = map sooner.")
    ap.add_argument("--sigma0-frac", type=float, default=1,
                    help="Initial prior hill width as fraction of radius (e.g., 0.6 → σ0 = 0.6*R). Bigger = cling near first point longer.")
    ap.add_argument("--sigma-frac", type=float, default=0.3,
                    help="Shared Gaussian width for success hills and failure dents as fraction of radius. Bigger = broader influence.")
    ap.add_argument("--prior-amp-frac", type=float, default=1.50,
                    help="Initial prior hill amplitude as a fraction of success amplitude.")
    ap.add_argument("--success-amp", type=float, default=1.0,
                    help="Amplitude for success hills. Keep at 1.0 and tune others relative to it.")
    ap.add_argument("--fail-amp-frac", type=float, default=0.30,
                    help="Failure dent amplitude as a fraction of success amplitude (smaller than 1).")
    ap.add_argument("--step1-candidates", type=int, default=4096,
                    help="Number of random candidates in disk before picking one by HillMap weights.")
    ap.add_argument("--explore-floor", type=float, default=0.10,
                    help="Minimum probability mass everywhere to avoid getting stuck. Bigger = more global exploration.")
    ap.add_argument("--min-spacing-mm", type=float, default=0.01,
                    help="Minimum allowed distance from any previously tried point.")

    # Step 2 (Gradient Mapper) — minimal knobs
    ap.add_argument("--step2-step-frac", type=float, default=0.20,
                    help="Step length as fraction of radius when following negative gradient (e.g., 0.2 → step = 0.2*R).")
    ap.add_argument("--back-to-step1-streak", type=int, default=5,
                    help="Consecutive unindexed proposals in Step 2 that trigger a fallback to Step 1.")

    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr); return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr); return 2

    history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    latest_rows_by_key: Dict[Tuple[str,int], List[int]] = {}
    current_key: Optional[Tuple[str,int]] = None
    for idx, (key, row) in enumerate(entries):
        if key is not None:
            current_key = key; continue
        if row is None or len(row) == 0 or row[0] == "RAW":
            continue
        run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = row[:7]
        try:
            run_n = int(run_s)
        except Exception:
            continue
        dx = float(dx_s) if dx_s else 0.0
        dy = float(dy_s) if dy_s else 0.0
        indexed = int(idx_s or "0")
        wr = _float_or_blank(wr_s)
        if current_key is None:
            continue
        history.setdefault(current_key, []).append((run_n, dx, dy, indexed, wr))
        if run_n == latest_run:
            latest_rows_by_key.setdefault(current_key, []).append(idx)

    rng_master = np.random.default_rng(int(args.seed))
    n_new = 0

    for key, rows in latest_rows_by_key.items():
        trials = sorted(history.get(key, []), key=lambda t: t[0])
        # Per-event RNG for reproducibility
        key_seed = int(abs(hash((args.seed, key[0], key[1]))) % (2**32 - 1))
        rng = np.random.default_rng(key_seed)

        ndx, ndy, reason = propose_event_hillmap(
            trials_sorted=trials,
            R=float(args.radius_mm),
            rng=rng,
            # Step 1
            N_success_to_step2=max(3, int(args.step1_N_success)),
            sigma0_frac=float(args.sigma0_frac),
            sigma_frac=float(args.sigma_frac),
            prior_amp_frac=float(args.prior_amp_frac),
            success_amp=float(args.success_amp),
            fail_amp_frac=float(args.fail_amp_frac),
            step1_candidates=int(args.step1_candidates),
            explore_floor=float(args.explore_floor),
            min_spacing=float(args.min_spacing_mm),
            # Step 2
            step2_step_frac=float(args.step2_step_frac),
            back_to_step1_streak=int(args.back_to_step1_streak),
        )

        for row_idx in rows:
            row = list(entries[row_idx][1])
            if len(row) < 7:
                row += [""] * (7 - len(row))
            if row[0].isdigit():
                row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy)); n_new += 1
                entries[row_idx] = (None, tuple(row))
        # print(f"[diag:{os.path.basename(key[0])}|event{key[1]}] reason={reason}")

    write_log(log_path, entries)
    print(f"[propose] {n_new} new proposals")
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
