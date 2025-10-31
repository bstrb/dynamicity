#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
propose_next_shifts_raindrop.py — CSV parser + two-phase "Raindrop→Mapper" proposer

Step 1 (Raindrop exploration):
  - Start with a shallow, *wide* inverted Gaussian pool centered at the *first tried shift*.
  - Each successful index adds an inverted Gaussian "drop".
  - Each unsuccessful index adds a smaller positive Gaussian "hill".
  - Sample candidates proportionally to this landscape (with an exploration floor and min spacing).
  - Continue until we have N_success successes forming a triangle with angles within [θ_min, θ_max].

Step 2 (Mapper: local wRMSD descent):
  - Fit a smooth GP on successful (dx,dy) → wRMSD points.
  - Add a *slight* failure influence via a low-weight Gaussian bump field (same σ as hills).
  - Propose the next point by moving from the best-success point in the direction that
    reduces the GP mean + failure bump; step length is a fraction of the radius.
  - If recent unindexed streak ≥ back_to_step1, fall back to Step 1.

This script preserves the CSV format used by your previous tool:
  "#/abs/path event <int>"
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
'''
from __future__ import annotations

import argparse, os, sys, math, hashlib
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

# -------------------------- Geometry & sampling helpers --------------------------

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

def _inside_disk(xy: np.ndarray, R: float) -> np.ndarray:
    return (xy[:, 0]**2 + xy[:, 1]**2) <= (R * R + 1e-15)

def _min_dist(xy: np.ndarray, tried: np.ndarray) -> np.ndarray:
    """Return min distance from each xy[i] to any point in tried (shape (M,2))."""
    if tried.size == 0:
        return np.full((xy.shape[0],), np.inf, dtype=float)
    # broadcasting distance
    diff = xy[:, None, :] - tried[None, :, :]
    d2 = np.sum(diff * diff, axis=2)  # (N, M)
    return np.sqrt(np.min(d2, axis=1))

def _triangle_angles_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float]:
    """Internal angles (deg) of triangle (p1,p2,p3); returns (a,b,c)."""
    def side(a, b): return float(np.linalg.norm(a - b))
    A = side(p2, p3); B = side(p1, p3); C = side(p1, p2)
    # Law of cosines with clipping to avoid nan
    def angle(opposite, adj1, adj2):
        if adj1 <= 1e-15 or adj2 <= 1e-15:
            return 0.0
        cosv = (adj1*adj1 + adj2*adj2 - opposite*opposite) / (2.0 * adj1 * adj2)
        cosv = max(-1.0, min(1.0, cosv))
        return math.degrees(math.acos(cosv))
    a = angle(A, B, C); b = angle(B, A, C); c = angle(C, A, B)
    return a, b, c

# -------------------------- Simple GP for Step 2 --------------------------

def _rbf_aniso(X1, X2, lsx, lsy):
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _auto_ls(X, y, R):
    k = max(3, int(0.4 * X.shape[0]))
    idx = np.argsort(y)[:k]
    Xk = X[idx]
    sx = float(np.std(Xk[:,0], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    sy = float(np.std(Xk[:,1], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    lsx = float(np.clip(1.25 * sx, 0.004 * R, 0.25 * R))
    lsy = float(np.clip(1.25 * sy, 0.004 * R, 0.25 * R))
    return lsx, lsy

def _gp_fit_predict(X, y, Xstar, R, noise=1e-6):
    """Zero-mean GP on (dx,dy)->w with ARD RBF. Returns mu (N*,), var (N*,)."""
    if len(X) == 0:
        raise ValueError("GP requires at least one observation")
    y = np.asarray(y, dtype=float)
    ymean = float(np.mean(y))
    yc = y - ymean
    lsx, lsy = _auto_ls(np.asarray(X, float), y, R)
    K = _rbf_aniso(X, X, lsx, lsy) + (float(noise) * np.eye(len(X)))
    jitter = 1e-12
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(K + jitter * np.eye(len(X)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
    Ks = _rbf_aniso(X, Xstar, lsx, lsy)
    mu = Ks.T @ alpha + ymean
    v = np.linalg.solve(L, Ks)
    var = np.maximum(0.0, np.maximum(1e-12, 1.0 - np.sum(v * v, axis=0)))
    return mu, var

# -------------------------- Core proposal logic --------------------------

def _step1_raindrop_proposal(
    rng: np.random.Generator,
    successes: List[Tuple[float, float]],
    failures: List[Tuple[float, float]],
    first_shift: Tuple[float, float],
    R: float,
    n_cand: int,
    sigma0: float,
    sigma: float,
    drop_amp: float,
    hill_frac: float,
    prior_amp_frac: float,
    explore_floor: float,
    min_spacing: float,
    tried: np.ndarray,
) -> Tuple[float, float, str]:
    """Sample candidate from probability landscape formed by drops/hills/prior within disk radius R."""
    # Generate candidate points uniformly in disk
    C = _uniform_points_in_disk(rng, n_cand, R)
    # Enforce min spacing
    dmin = _min_dist(C, tried)
    mask = dmin >= (min_spacing - 1e-12)
    if not np.any(mask):
        mask = dmin >= 0.0
    C = C[mask]
    if C.shape[0] == 0:
        return 0.0, 0.0, "no_candidates"

    # Build field F = prior + sum(drops) - sum(hills)
    F = np.zeros((C.shape[0],), dtype=float)
    # prior
    c0 = np.array([first_shift[0], first_shift[1]], float)
    F += prior_amp_frac * drop_amp * _gauss2(C, c0, sigma0)
    # successes: inverted => increase F to increase sampling prob
    if successes:
        S = np.array(successes, float)
        for i in range(S.shape[0]):
            F += drop_amp * _gauss2(C, S[i, :], sigma)
    # failures: subtract smaller amplitude
    if failures:
        for fx, fy in failures:
            F -= (hill_frac * drop_amp) * _gauss2(C, np.array([fx, fy], float), sigma)

    # Convert to weights + exploration floor
    w = np.maximum(F, 0.0) + float(explore_floor)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        j = int(rng.integers(0, C.shape[0]))
        return float(C[j,0]), float(C[j,1]), "step1_uniform_fallback"

    w = w / s
    j = int(rng.choice(C.shape[0], p=w))
    return float(C[j,0]), float(C[j,1]), "step1_raindrop"

def _angles_ok(pts3: np.ndarray, amin: float, amax: float) -> bool:
    a, b, c = _triangle_angles_deg(pts3[0], pts3[1], pts3[2])
    return (a >= amin and b >= amin and c >= amin and a <= amax and b <= amax and c <= amax)

def _recent_unindexed_streak(trials_sorted: List[Tuple[int,float,float,int,Optional[float]]]) -> int:
    streak = 0
    for _, _, _, idx, _ in reversed(trials_sorted):
        if idx == 0:
            streak += 1
        else:
            break
    return streak

def _step2_mapper_proposal(
    rng: np.random.Generator,
    successes_w: List[Tuple[float, float, float]],
    failures: List[Tuple[float, float]],
    R: float,
    sigma: float,
    step_frac: float,
    fail_bump_frac: float,
    ring_dirs: int,
    min_spacing: float,
    tried: np.ndarray,
) -> Tuple[float, float, str]:
    """
    Propose next point by moving from current best along the direction that reduces
    GP mean (plus slight failure bump). Uses a ring of candidate directions.
    """
    X = np.array([[x, y] for (x, y, _) in successes_w], float)
    y = np.array([w for (_, _, w) in successes_w], float)

    # Best incumbent
    jbest = int(np.argmin(y))
    xb = X[jbest, :].copy()
    # Candidate ring radii (adaptive: try longer then shorter if needed)
    r1 = step_frac * R
    r2 = 0.5 * r1
    thetas = np.linspace(0.0, 2.0*np.pi, num=ring_dirs, endpoint=False)
    cir1 = np.stack([xb[0] + r1*np.cos(thetas), xb[1] + r1*np.sin(thetas)], axis=1)
    cir2 = np.stack([xb[0] + r2*np.cos(thetas), xb[1] + r2*np.sin(thetas)], axis=1)
    C = np.concatenate([cir1, cir2], axis=0)
    # Project to disk if needed
    norms = np.sqrt(C[:,0]**2 + C[:,1]**2)
    over = norms > R
    if np.any(over):
        C[over, 0] = C[over, 0] * (R / norms[over])
        C[over, 1] = C[over, 1] * (R / norms[over])
    # Enforce min spacing
    dmin = _min_dist(C, tried)
    mask = dmin >= (min_spacing - 1e-12)
    C = C[mask]
    if C.shape[0] == 0:
        # fallback: jitter around best
        for _ in range(64):
            jitter = (rng.standard_normal(2) * (0.25 * r2))
            cand = xb + jitter
            if cand[0]**2 + cand[1]**2 <= R*R and _min_dist(cand[None,:], tried)[0] >= min_spacing:
                return float(cand[0]), float(cand[1]), "step2_fallback_jitter"
        return float(xb[0]), float(xb[1]), "step2_fallback_incumbent"

    # GP mean on candidates
    mu, _ = _gp_fit_predict(X, y, C, R, noise=1e-6)
    # Slight failure bumps (scaled to median wRMSD)
    med_w = float(np.median(y)) if y.size > 0 else 1.0
    bump_amp = float(fail_bump_frac) * med_w
    if failures:
        H = np.zeros((C.shape[0],), dtype=float)
        for fx, fy in failures:
            H += _gauss2(C, np.array([fx, fy], float), sigma)
        mu = mu + bump_amp * H

    j = int(np.argmin(mu))
    return float(C[j,0]), float(C[j,1]), "step2_mapper"

# -------------------------- Main per-event proposal --------------------------

def propose_event_raindrop(
    trials_sorted: List[Tuple[int,float,float,int,Optional[float]]],
    R: float,
    rng: np.random.Generator,
    # Step 1 params
    N_success_to_step2: int,
    tri_angle_min_deg: float,
    tri_angle_max_deg: float,
    sigma0_frac: float,
    sigma_drop_frac: float,
    prior_amp_frac: float,
    drop_amp: float,
    hill_frac: float,
    step1_candidates: int,
    explore_floor: float,
    min_spacing: float,
    # Step 2 params
    step2_step_frac: float,
    step2_ring_dirs: int,
    step2_fail_bump_frac: float,
    back_to_step1_streak: int,
) -> Tuple[float, float, str]:
    """Return (next_dx, next_dy, reason_str) or ('done','done',reason)"""
    # Gather tried, successes, failures
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

    # First tried shift for prior
    if len(trials_sorted) > 0:
        first_dx, first_dy = trials_sorted[0][1], trials_sorted[0][2]
    else:
        first_dx, first_dy = 0.0, 0.0

    sigma0 = float(max(1e-12, sigma0_frac * R))
    sigma = float(max(1e-12, sigma_drop_frac * R))

    # Decide if Step 2 conditions are met
    reason_prefix = ""
    if len(successes_w) >= max(3, N_success_to_step2):
        # check triangle non-colinear via angle constraints
        # pick 3 best successes by wRMSD and test angles
        SW = sorted(successes_w, key=lambda t: t[2])
        base3 = np.array([[SW[i][0], SW[i][1]] for i in range(3)], float)
        if _angles_ok(base3, tri_angle_min_deg, tri_angle_max_deg):
            # Check recent unindexed streak
            if _recent_unindexed_streak(trials_sorted) >= back_to_step1_streak:
                reason_prefix = "fallback_to_step1_after_unindexed_streak"
            else:
                # STEP 2
                ndx, ndy, r = _step2_mapper_proposal(
                    rng=rng,
                    successes_w=successes_w,
                    failures=failures,
                    R=R,
                    sigma=sigma,
                    step_frac=step2_step_frac,
                    fail_bump_frac=step2_fail_bump_frac,
                    ring_dirs=step2_ring_dirs,
                    min_spacing=min_spacing,
                    tried=tried,
                )
                return ndx, ndy, r

    # STEP 1
    ndx, ndy, r = _step1_raindrop_proposal(
        rng=rng,
        successes=successes,
        failures=failures,
        first_shift=(first_dx, first_dy),
        R=R,
        n_cand=step1_candidates,
        sigma0=sigma0,
        sigma=sigma,
        drop_amp=drop_amp,
        hill_frac=hill_frac,
        prior_amp_frac=prior_amp_frac,
        explore_floor=explore_floor,
        min_spacing=min_spacing,
        tried=tried,
    )
    if reason_prefix:
        r = reason_prefix + "|" + r
    return ndx, ndy, r

# -------------------------- CLI --------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-event proposer: Two-phase Raindrop→Mapper using CSV history (no external deps).")
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")
    # Domain (circular bounds)
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Disk radius bound for |(dx,dy)|.")
    ap.add_argument("--seed", type=int, default=1337)

    # Step 1 (Raindrop) minimal params
    ap.add_argument("--step1-N-success", type=int, default=3, help="Number of successes required to enter Step 2 (>=3).")
    ap.add_argument("--triangle-angle-min-deg", type=float, default=30.0, help="Min internal angle (deg) for the success triangle.")
    ap.add_argument("--triangle-angle-max-deg", type=float, default=120.0, help="Max internal angle (deg) for the success triangle.")
    ap.add_argument("--sigma0-frac", type=float, default=0.60, help="Initial shallow pool width as fraction of radius.")
    ap.add_argument("--sigma-drop-frac", type=float, default=0.18, help="Drop/Hill Gaussian width as fraction of radius.")
    ap.add_argument("--prior-amp-frac", type=float, default=0.30, help="Initial pool amplitude relative to drop_amp.")
    ap.add_argument("--drop-amp", type=float, default=1.0, help="Amplitude for success drops; hills use hill-frac * drop-amp.")
    ap.add_argument("--hill-frac", type=float, default=0.40, help="Amplitude fraction for failure hills relative to drops.")
    ap.add_argument("--step1-candidates", type=int, default=4096, help="Number of random candidate points in disk for Step 1 sampling.")
    ap.add_argument("--explore-floor", type=float, default=0.10, help="Minimum probability mass for global exploration.")
    ap.add_argument("--min-spacing-mm", type=float, default=0.0001, help="Minimum distance to previous tried points.")

    # Step 2 (Mapper) minimal params
    ap.add_argument("--step2-step-frac", type=float, default=0.20, help="Step length as fraction of radius for ring search.")
    ap.add_argument("--step2-ring-dirs", type=int, default=5, help="Number of directions on ring for Step 2.")
    ap.add_argument("--step2-fail-bump-frac", type=float, default=0.10, help="Failure bump strength as fraction of median wRMSD.")
    ap.add_argument("--back-to-step1-streak", type=int, default=5, help="Consecutive unindexed proposals in Step 2 to fall back to Step 1.")

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
    n_new, n_done = 0, 0

    for key, rows in latest_rows_by_key.items():
        trials = sorted(history.get(key, []), key=lambda t: t[0])
        # Per-event RNG for reproducibility/diversity
        key_seed = int(abs(hash((args.seed, key[0], key[1]))) % (2**32 - 1))
        rng = np.random.default_rng(key_seed)

        ndx, ndy, reason = propose_event_raindrop(
            trials_sorted=trials,
            R=float(args.radius_mm),
            rng=rng,
            # Step 1
            N_success_to_step2=max(3, int(args.step1_N_success)),
            tri_angle_min_deg=float(args.triangle_angle_min_deg),
            tri_angle_max_deg=float(args.triangle_angle_max_deg),
            sigma0_frac=float(args.sigma0_frac),
            sigma_drop_frac=float(args.sigma_drop_frac),
            prior_amp_frac=float(args.prior_amp_frac),
            drop_amp=float(args.drop_amp),
            hill_frac=float(args.hill_frac),
            step1_candidates=int(args.step1_candidates),
            explore_floor=float(args.explore_floor),
            min_spacing=float(args.min_spacing_mm),
            # Step 2
            step2_step_frac=float(args.step2_step_frac),
            step2_ring_dirs=int(args.step2_ring_dirs),
            step2_fail_bump_frac=float(args.step2_fail_bump_frac),
            back_to_step1_streak=int(args.back_to_step1_streak),
        )

        for row_idx in rows:
            row = list(entries[row_idx][1])
            if len(row) < 7:
                row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):
                    row[5] = "done"; row[6] = "done"; n_done += 1
                else:
                    row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy)); n_new += 1
                entries[row_idx] = (None, tuple(row))
        # print(f"[diag:{os.path.basename(key[0])}|event{key[1]}] reason={reason}")

    write_log(log_path, entries)
    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
