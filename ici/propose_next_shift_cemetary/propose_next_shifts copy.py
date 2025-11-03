#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py

Per-frame proposer with three phases:
  (1) Expanding RING search from the initial center until the first successful index (finite wRMSD).
  (2) On each success (until N successes are accumulated), RESTART a slightly smaller RING
      centered very close to that successful shift (offset = frac * current r_step).
  (3) After >= N successes, ADAPTIVELY MAP the local minimum of the wRMSD surface using
      trust-region Bayesian optimization (GP + EI) augmented with a gradient step and a
      quadratic-fit candidate. Continue proposing per run until convergence criteria are met.

Minimal knobs:
  • Ring geometry: --r-max, --r-step, --k-base
  • Ring shrink & re-center: --ring-shrink, --center-offset-frac
  • Switch to mapping: --min-good-for-best  (>=3)
  • Mapping BO: --bo-candidates, --bo-noise, --bo-guard-abs
  • Convergence/“done”: --done-ei, --done-step-mm, --done-wrmsd

CSV compatibility
-----------------
Keeps grouped CSV format:
  "#/abs/path/to/file event <int>"
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm

Only rows belonging to the LATEST run are updated, per event (frame).

Author: ChatGPT
License: MIT
"""

from __future__ import annotations
import argparse, os, sys, math, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np

# ------------------------------ Utilities ------------------------------

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

def _hash_angle(seed: int, key: Tuple[str,int]) -> float:
    """Stable per-event base angle in [0, 2π) for ring sampling."""
    h = hashlib.sha1(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "little")
    frac = (v & ((1<<53)-1)) / float(1<<53)
    return 2.0 * math.pi * frac

GOLDEN_ANGLE = 2.0 * math.pi * (1.0 - (math.sqrt(5.0)-1.0)/2.0)  # ~2.399963229...

def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    """How many angular samples to use for a given ring radius r."""
    return max(4, int(math.ceil(k_base * (r / max(r_max, 1e-12)))))

# ------------------------------ CSV I/O (grouped) ------------------------------

def parse_log(log_path: str):
    """Return list of (key,row) and latest_run. key=(abs_path,event_id) on header lines."""
    entries = []
    latest_run = -1
    with open(log_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
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

# ------------------------------ RING search core ------------------------------

def _ring_candidate(center: Tuple[float,float],
                    tried_set: set,
                    r_max: float, r_step: float, k_base: float,
                    base_angle: float, phase: float,
                    global_clip: float) -> Optional[Tuple[float,float]]:
    """
    Iterate radii r=k*r_step up to r_max; for each r, sample n_angles_for_radius(r) angles.
    Return first candidate not tried (6-dec grid) and inside [-global_clip, +global_clip]^2.
    """
    cx, cy = center
    if r_step <= 0 or r_max <= 0:
        return None
    max_k = int(math.floor(r_max / r_step + 1e-9))
    for k in range(1, max_k + 1):
        r = k * r_step
        n = n_angles_for_radius(r, r_max, k_base)
        for i in range(n):
            theta = base_angle + phase + 2.0 * math.pi * (i / n)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            x = float(np.clip(x, -global_clip, global_clip))
            y = float(np.clip(y, -global_clip, global_clip))
            keyfmt = (_fmt6(x), _fmt6(y))
            if keyfmt in tried_set:
                continue
            return (x, y)
    return None

# ------------------------------ GP + EI helpers ------------------------------

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
    """Zero-mean GP with RBF kernel (amp=1); returns (mu, var)."""
    if len(X) == 0:
        raise ValueError("GP requires at least one observation")
    ymean = float(np.mean(y))
    yc = y - ymean
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
    var = np.maximum(0.0, 1.0 - np.sum(v * v, axis=0))  # amp=1 prior
    return mu, var

def _phi(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)

def _Phi(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

def _expected_improvement(mu: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
    """EI for minimization."""
    std = np.sqrt(np.maximum(var, 1e-16))
    z = (y_best - mu - xi) / std
    ei = (y_best - mu - xi) * _Phi(z) + std * _phi(z)
    ei[var < 1e-30] = 0.0
    return ei

def _auto_ls(X: np.ndarray, y: np.ndarray, R: float) -> Tuple[float,float]:
    """Data-driven lengthscales from top ~40% points, clamped to [0.004R, 0.20R]."""
    k = max(3, int(0.4 * X.shape[0]))
    idx = np.argsort(y)[:k]
    Xk = X[idx]
    sx = float(np.std(Xk[:,0], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    sy = float(np.std(Xk[:,1], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    lsx = float(np.clip(1.25 * sx, 0.004 * R, 0.20 * R))
    lsy = float(np.clip(1.25 * sy, 0.004 * R, 0.20 * R))
    return lsx, lsy

def _gradient_step_candidate(bx, by, rho, X, y, lsx, lsy, noise) -> Optional[Tuple[float,float]]:
    """One negative-gradient step of GP posterior mean at incumbent."""
    h = 0.25 * rho
    P = np.array([[bx + h, by], [bx - h, by], [bx, by + h], [bx, by - h]], float)
    mu, _ = _gp_fit_predict(X, y, P, lsx, lsy, noise)
    dmu_dx = float((mu[0] - mu[1]) / (2*h))
    dmu_dy = float((mu[2] - mu[3]) / (2*h))
    gnorm = math.hypot(dmu_dx, dmu_dy)
    if not np.isfinite(gnorm) or gnorm < 1e-12:
        return None
    step = 0.5 * rho
    nx = float(np.clip(bx - step * dmu_dx/gnorm, bx - rho, bx + rho))
    ny = float(np.clip(by - step * dmu_dy/gnorm, by - rho, by + rho))
    return (nx, ny)

def _quad_minimizer(X: np.ndarray, y: np.ndarray):
    """Fit 2D quadratic and return minimizer if Hessian PD."""
    if X.shape[0] < 6: return None
    x, z = X[:,0], X[:,1]
    M = np.column_stack([x*x, z*z, x*z, x, z, np.ones_like(x)])
    try:
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a,b,c,d,e,f = [float(v) for v in coef]
    H = np.array([[2*a, c],[c, 2*b]], float)
    g = np.array([d, e], float)
    try:
        if np.any(np.linalg.eigvalsh(H) <= 0): return None
        sol = -np.linalg.solve(H, g)
        return float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        return None

def _filter_seen(Xcand: np.ndarray, Xseen: np.ndarray, eps: float) -> np.ndarray:
    """Drop candidates within eps of any previously tried shift."""
    if Xseen.size == 0 or Xcand.size == 0:
        return np.ones(Xcand.shape[0], dtype=bool)
    keep = np.ones(Xcand.shape[0], dtype=bool)
    for i, p in enumerate(Xcand):
        if not keep[i]:
            continue
        d = np.hypot(Xseen[:,0] - p[0], Xseen[:,1] - p[1])
        if np.any(d < eps):
            keep[i] = False
    return keep

# ------------------------------ Adaptive surface step ------------------------------

def _adaptive_surface_step(X_good: np.ndarray, y_good: np.ndarray,
                           R: float, bx: float, by: float,
                           bo_candidates: int, bo_guard_abs: float,
                           bo_noise: float, rng: np.random.Generator):
    """
    One adaptive BO step near the incumbent:
      • TR radius rho = 60th percentile of distances to (bx,by), clamped.
      • Candidates: local Gaussian cloud + ring points + gradient step + quad minimizer.
      • Relevance guard: keep if mu <= best+bo_guard_abs or var high.
      • Acquisition: EI * exp(-(dist/rho)^2).
    Returns (nx, ny, meta_dict).
    """
    # Trust region
    d = np.sqrt((X_good[:,0]-bx)**2 + (X_good[:,1]-by)**2)
    if d.size >= 3 and np.all(np.isfinite(d)):
        rho = float(np.quantile(d, 0.60))
    else:
        rho = 0.02
    rho = float(np.clip(rho, 0.006, min(0.04, R)))  # 6–40 µm typically works well

    # GP hyperparams from data
    lsx, lsy = _auto_ls(X_good, y_good, R)

    # Candidate set
    Cl = max(60, int(0.90 * bo_candidates))
    Cg = max(1, bo_candidates - Cl)
    sx = 0.30 * rho; sy = 0.30 * rho
    X_loc = np.column_stack([rng.normal(bx, sx, Cl), rng.normal(by, sy, Cl)])
    X_loc[:,0] = np.clip(X_loc[:,0], bx - rho, bx + rho)
    X_loc[:,1] = np.clip(X_loc[:,1], by - rho, by + rho)

    k_ring = 16
    thetas = np.linspace(0, 2*np.pi, k_ring, endpoint=False)
    ring = np.column_stack([bx + 0.5*rho*np.cos(thetas), by + 0.5*rho*np.sin(thetas)])

    X_glb = np.column_stack([rng.uniform(bx - rho, bx + rho, Cg),
                             rng.uniform(by - rho, by + rho, Cg)])

    Xc = np.vstack([X_loc, ring, X_glb])

    # Add gradient-step and quadratic minimizer candidates
    grad = _gradient_step_candidate(bx, by, rho, X_good, y_good, lsx, lsy, bo_noise)
    if grad is not None: Xc = np.vstack([Xc, np.array([[grad[0], grad[1]]])])
    qmin = _quad_minimizer(X_good, y_good)
    if qmin is not None:
        qx, qy = float(np.clip(qmin[0], bx - rho, bx + rho)), float(np.clip(qmin[1], by - rho, by + rho))
        Xc = np.vstack([Xc, np.array([[qx, qy]])])

    # Predict and acquire
    mu, var = _gp_fit_predict(X_good, y_good, Xc, lsx, lsy, bo_noise)
    y_best = float(np.min(y_good))
    ei = _expected_improvement(mu, var, y_best, xi=0.0)

    # Relevance guard (drop obviously bad points unless uncertain)
    keep = (mu <= y_best + bo_guard_abs) | (var >= 0.05)  # uncertainty gate
    if np.any(keep):
        Xc, mu, var, ei = Xc[keep], mu[keep], var[keep], ei[keep]

    # Locality weight to discourage far-edge picks when we already have a basin
    dist = np.sqrt((Xc[:,0]-bx)**2 + (Xc[:,1]-by)**2)
    w = np.exp(- (dist / max(rho, 1e-9))**2)
    score = ei * w

    if Xc.shape[0] == 0:
        return (bx, by, {"rho": rho, "ei_max": 0.0, "guard_pruned_all": True})

    j = int(np.argmax(score))
    return (float(Xc[j,0]), float(Xc[j,1]), {
        "rho": rho, "ei_max": float(ei[j]), "lsx": float(lsx), "lsy": float(lsy)
    })

# ------------------------------ Per-event proposal ------------------------------

def propose_event(trials_sorted: List[Tuple[int,float,float,int,Optional[float]]],
                  r_max0: float, r_step0: float, k_base: float,
                  ring_shrink: float, center_offset_frac: float,
                  min_good_for_best: int, global_clip: float,
                  base_angle: float, seed: int,
                  bo_candidates: int, bo_guard_abs: float, bo_noise: float,
                  done_ei: float, done_step_mm: float, done_wrmsd: float
                  ) -> Tuple[object, object, str]:
    """
    trials_sorted: [(run_n, dx, dy, indexed, wrmsd)] sorted by run_n
    Returns: (next_dx, next_dy, reason) where next_* may be "done".
    """
    tried_set = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)
    successes = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
    n_succ = len(successes)

    # Initial center = earliest det_shift for this event, else (0,0)
    if len(trials_sorted) > 0:
        init_cx, init_cy = trials_sorted[0][1], trials_sorted[0][2]
    else:
        init_cx, init_cy = 0.0, 0.0

    # ----------------- Phases 1 & 2: expanding rings -----------------
    if n_succ < max(3, min_good_for_best):
        # Compute ring params for current stage s = n_succ (shrink after each success)
        s = n_succ
        r_max = r_max0 * (ring_shrink ** s)
        r_step = max(1e-6, r_step0 * (ring_shrink ** s))

        # Ring center: initial center for s==0, else near last success with small offset
        if s == 0:
            cx, cy = init_cx, init_cy
        else:
            last_sx, last_sy, _ = successes[-1]
            eps = center_offset_frac * r_step  # fraction of CURRENT r_step
            theta = base_angle + s * GOLDEN_ANGLE * 0.5
            cx = last_sx + eps * math.cos(theta + math.pi/2.0)
            cy = last_sy + eps * math.sin(theta + math.pi/2.0)

        phase = s * GOLDEN_ANGLE  # vary angular phase each stage

        cand = _ring_candidate(center=(cx,cy),
                               tried_set=tried_set,
                               r_max=r_max, r_step=r_step, k_base=k_base,
                               base_angle=base_angle, phase=phase,
                               global_clip=global_clip)
        if cand is None:
            return ("done", "done", "ring_exhausted")
        return (float(cand[0]), float(cand[1]), "ring_stage")

    # ----------------- Phase 3: Adaptive surface mapping (BO) -----------------
    X_good = np.array([[dx,dy] for (dx,dy,wr) in successes], float)
    y_good = np.array([wr for (dx,dy,wr) in successes], float)

    jbest = int(np.argmin(y_good))
    bx, by = float(X_good[jbest,0]), float(X_good[jbest,1])
    ybest = float(y_good[jbest])

    rng = np.random.default_rng(abs(hash((base_angle, seed))) % (2**32 - 1))

    # Build candidate set and pick acquisition winner
    nx, ny, meta = _adaptive_surface_step(
        X_good=X_good, y_good=y_good, R=global_clip, bx=bx, by=by,
        bo_candidates=bo_candidates, bo_guard_abs=bo_guard_abs,
        bo_noise=bo_noise, rng=rng
    )

    # Convergence / relevance controls
    proposed_step = math.hypot(nx - bx, ny - by)
    ei_max = float(meta.get("ei_max", 0.0))

    if (ei_max < done_ei) and ((ybest <= done_wrmsd) or (proposed_step < done_step_mm)):
        return ("done", "done", "map_converged")

    # Avoid exact duplicate of any previously tried shift
    if (_fmt6(nx), _fmt6(ny)) in tried_set:
        dx = bx - nx; dy = by - ny
        norm = math.hypot(dx, dy)
        if norm > 1e-12:
            nx += 0.25 * dx; ny += 0.25 * dy

    # Clip to global square and return
    nx = float(np.clip(nx, -global_clip, global_clip))
    ny = float(np.clip(ny, -global_clip, global_clip))
    return (float(nx), float(ny), "map_adaptive_bo")

# ------------------------------ Main ------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-frame center-shift proposer: expanding ring -> (on each success) smaller ring -> (after N successes) adaptive surface mapping until done.")
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")

    # Global clamp for absolute shift domain (square [-R,R]^2).
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Global half-width limit for |dx| and |dy|.")

    # Ring parameters
    ap.add_argument("--r-max", type=float, default=0.05, help="Initial ring maximum radius (mm).")
    ap.add_argument("--r-step", type=float, default=0.01, help="Ring radial increment (mm).")
    ap.add_argument("--k-base", type=float, default=20.0, help="Angular density scale (angles per ring ∝ k_base * r/r_max).")
    ap.add_argument("--ring-shrink", type=float, default=0.90, help="Factor to reduce r_max and r_step after each success (slightly < 1).")
    ap.add_argument("--center-offset-frac", type=float, default=0.10, help="Offset of new ring center = frac * current r_step (to avoid cycling same rays).")

    # Switch to adaptive mapping after this many successful points (>=3).
    ap.add_argument("--min-good-for-best", type=int, default=4, help="Number of successful (indexed + finite wRMSD) points before adaptive mapping.")

    # Mapping / BO (minimal knobs)
    ap.add_argument("--bo-candidates", type=int, default=800, help="Number of adaptive mapping candidates.")
    ap.add_argument("--bo-noise", type=float, default=1e-4, help="GP observation noise for wRMSD.")
    ap.add_argument("--bo-guard-abs", type=float, default=0.10, help="Keep candidates with mu <= best+guard or high variance.")

    # Done / relevance tolerance
    ap.add_argument("--done-ei", type=float, default=5e-3, help="Stop mapping if EI_max below this.")
    ap.add_argument("--done-step-mm", type=float, default=1e-3, help="Stop if proposed step (from incumbent) is smaller than this (mm).")
    ap.add_argument("--done-wrmsd", type=float, default=0.10, help="Stop if incumbent best wRMSD <= this.")

    # RNG
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr)
        return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr)
        return 2

    # Gather history and latest-run row indices per event
    history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    latest_rows_by_key: Dict[Tuple[str,int], List[int]] = {}
    current_key: Optional[Tuple[str,int]] = None
    for idx, (key, row) in enumerate(entries):
        if key is not None:
            current_key = key
            continue
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

    # Propose per event in latest run
    n_new, n_done = 0, 0
    for key, rows in latest_rows_by_key.items():
        trials = sorted(history.get(key, []), key=lambda t: t[0])
        base_angle = _hash_angle(args.seed, key)

        ndx, ndy, reason = propose_event(
            trials_sorted=trials,
            r_max0=float(args.r_max),
            r_step0=float(args.r_step),
            k_base=float(args.k_base),
            ring_shrink=float(args.ring_shrink),
            center_offset_frac=float(args.center_offset_frac),
            min_good_for_best=max(3, int(args.min_good_for_best)),
            global_clip=float(args.radius_mm),
            base_angle=base_angle,
            seed=int(args.seed),
            bo_candidates=int(args.bo_candidates),
            bo_guard_abs=float(args.bo_guard_abs),
            bo_noise=float(args.bo_noise),
            done_ei=float(args.done_ei),
            done_step_mm=float(args.done_step_mm),
            done_wrmsd=float(args.done_wrmsd),
        )

        for row_idx in rows:
            row = list(entries[row_idx][1])
            if len(row) < 7:
                row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):  # e.g., "done"
                    row[5] = "done"; row[6] = "done"
                    n_done += 1
                else:
                    row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy))
                    n_new += 1
                entries[row_idx] = (None, tuple(row))
        # Per-event diagnostics
        # print(f"[diag:{os.path.basename(key[0])}|event{key[1]}] reason={reason}")

    write_log(log_path, entries)
    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
