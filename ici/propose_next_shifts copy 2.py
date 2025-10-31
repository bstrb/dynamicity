#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py  —  per-frame expanding-ring + gradient-driven surface mapping

Phases
------
1) Expanding ring from the initial center until the first success.
2) On each success (until N successes), restart a slightly smaller ring, centered near
   that success (offset = frac * current r_step).
3) After >= N successes, adaptively map the local minimum using a **gradient-driven**
   step on the GP posterior mean surface, with a small backtracking line-search.
   We also try a quadratic-fit (Newton-like) candidate and pick the one with lower
   predicted mean. No patience/shrink knobs.

Minimal knobs
-------------
• Ring: --r-max, --r-step, --k-base, --ring-shrink, --center-offset-frac
• Switch: --min-good-for-best (>=3)
• Mapping: --bo-step-mm (max step), --bo-noise (GP noise), --bo-backtrack (∈(0,1)), --bo-trials
• Done: --done-step-mm, --done-wrmsd, --done-grad

CSV format preserved:
  "#/abs/path event <int>"
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
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

GOLDEN_ANGLE = 2.0 * math.pi * (1.0 - (math.sqrt(5.0)-1.0)/2.0)  # ~2.399963

def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    return max(4, int(math.ceil(k_base * (r / max(r_max, 1e-12)))))

# ------------------------------ CSV I/O (grouped) ------------------------------

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

# ------------------------------ RING search core ------------------------------

def _ring_candidate(center: Tuple[float,float],
                    tried_set: set,
                    r_max: float, r_step: float, k_base: float,
                    base_angle: float, phase: float,
                    global_clip: float) -> Optional[Tuple[float,float]]:
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

# ------------------------------ GP helpers ------------------------------

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
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
    var = np.maximum(0.0, 1.0 - np.sum(v * v, axis=0))
    return mu, var

def _auto_ls(X: np.ndarray, y: np.ndarray, R: float) -> Tuple[float,float]:
    k = max(3, int(0.4 * X.shape[0]))
    idx = np.argsort(y)[:k]
    Xk = X[idx]
    sx = float(np.std(Xk[:,0], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    sy = float(np.std(Xk[:,1], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    lsx = float(np.clip(1.25 * sx, 0.004 * R, 0.20 * R))
    lsy = float(np.clip(1.25 * sy, 0.004 * R, 0.20 * R))
    return lsx, lsy

def _gp_mean_grad_at(bx, by, X, y, lsx, lsy, noise):
    """Return (mu, dmu_dx, dmu_dy) at (bx,by) via central differences on GP mean."""
    # a robust h picks based on data scale
    R = max(1e-3, 0.02)
    h = 0.25 * R  # we will rescale by lengthscales to be safe
    # choose per-dim finite-diff step relative to lengthscale
    hx = max(1e-4, 0.1 * lsx); hy = max(1e-4, 0.1 * lsy)
    P = np.array([[bx + hx, by], [bx - hx, by], [bx, by + hy], [bx, by - hy]], float)
    mu, _ = _gp_fit_predict(X, y, P, lsx, lsy, noise)
    dmu_dx = float((mu[0] - mu[1]) / (2*hx))
    dmu_dy = float((mu[2] - mu[3]) / (2*hy))
    # also return mean at incumbent
    P0 = np.array([[bx, by]], float)
    mu0, _ = _gp_fit_predict(X, y, P0, lsx, lsy, noise)
    return float(mu0[0]), dmu_dx, dmu_dy

def _quad_minimizer(X: np.ndarray, y: np.ndarray):
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

# ------------------------------ Gradient-driven mapping ------------------------------

def _gradient_mapping_step(X_good: np.ndarray, y_good: np.ndarray,
                           bx: float, by: float, R: float,
                           bo_step_mm: float, bo_backtrack: float, bo_trials: int,
                           bo_noise: float, rng) -> Tuple[float,float,dict]:
    """One step toward the minimum using GP-mean gradient + small backtracking line-search.
       Also try a quadratic-fit minimizer candidate; pick the best predicted mean.
    """
    lsx, lsy = _auto_ls(X_good, y_good, R)
    mu0, gx, gy = _gp_mean_grad_at(bx, by, X_good, y_good, lsx, lsy, bo_noise)
    gnorm = math.hypot(gx, gy)
    # Trust-region from data geometry (60th percentile)
    d = np.sqrt((X_good[:,0]-bx)**2 + (X_good[:,1]-by)**2)
    if d.size >= 3 and np.all(np.isfinite(d)):
        rho = float(np.quantile(d, 0.60))
    else:
        rho = 0.02
    rho = float(np.clip(rho, 0.006, min(0.04, R)))
    # Step budget limited by rho and bo_step_mm
    step0 = min(0.5 * rho, max(1e-4, bo_step_mm))
    cand_list = []

    if gnorm > 1e-12:
        dx = -gx / gnorm; dy = -gy / gnorm
        # small backtracking line search on GP mean
        s = step0
        for _ in range(max(1, bo_trials)):
            nx = float(np.clip(bx + s*dx, -R, R))
            ny = float(np.clip(by + s*dy, -R, R))
            cand_list.append((nx, ny))
            s *= float(np.clip(bo_backtrack, 0.3, 0.95))

    # Quadratic-fit candidate (Newton-like)
    qmin = _quad_minimizer(X_good, y_good)
    if qmin is not None:
        qx = float(np.clip(qmin[0], bx - rho, bx + rho))
        qy = float(np.clip(qmin[1], by - rho, by + rho))
        cand_list.append((qx, qy))

    # Always consider a tiny local probe to escape flatness
    tiny = 0.15 * step0
    ang = rng.uniform(0, 2*np.pi)
    cand_list.append((float(np.clip(bx + tiny*math.cos(ang), -R, R)),
                      float(np.clip(by + tiny*math.sin(ang), -R, R))))

    # Score by predicted mean
    Xc = np.array(cand_list, float)
    mu, _ = _gp_fit_predict(X_good, y_good, Xc, lsx, lsy, bo_noise)
    j = int(np.argmin(mu))
    return float(Xc[j,0]), float(Xc[j,1]), {
        "mu0": float(mu0), "mu_cand": float(mu[j]), "gnorm": float(gnorm), "rho": float(rho),
        "lsx": float(lsx), "lsy": float(lsy)
    }

# ------------------------------ Per-event proposal ------------------------------

def propose_event(trials_sorted: List[Tuple[int,float,float,int,Optional[float]]],
                  r_max0: float, r_step0: float, k_base: float,
                  ring_shrink: float, center_offset_frac: float,
                  min_good_for_best: int, global_clip: float,
                  base_angle: float, seed: int,
                  bo_step_mm: float, bo_backtrack: float, bo_trials: int, bo_noise: float,
                  done_step_mm: float, done_wrmsd: float, done_grad: float
                  ) -> Tuple[object, object, str]:
    tried_set = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)
    successes = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
    n_succ = len(successes)

    # Initial center
    if len(trials_sorted) > 0:
        init_cx, init_cy = trials_sorted[0][1], trials_sorted[0][2]
    else:
        init_cx, init_cy = 0.0, 0.0

    # ----------------- Phases 1 & 2: expanding rings -----------------
    if n_succ < max(3, min_good_for_best):
        s = n_succ
        r_max = r_max0 * (ring_shrink ** s)
        r_step = max(1e-6, r_step0 * (ring_shrink ** s))
        if s == 0:
            cx, cy = init_cx, init_cy
        else:
            last_sx, last_sy, _ = successes[-1]
            eps = center_offset_frac * r_step
            theta = base_angle + s * GOLDEN_ANGLE * 0.5
            cx = last_sx + eps * math.cos(theta + math.pi/2.0)
            cy = last_sy + eps * math.sin(theta + math.pi/2.0)
        phase = s * GOLDEN_ANGLE
        cand = _ring_candidate((cx,cy), tried_set, r_max, r_step, k_base, base_angle, phase, global_clip)
        if cand is None:
            return ("done", "done", "ring_exhausted")
        return (float(cand[0]), float(cand[1]), "ring_stage")

    # ----------------- Phase 3: Gradient-driven mapping -----------------
    X_good = np.array([[dx,dy] for (dx,dy,wr) in successes], float)
    y_good = np.array([wr for (dx,dy,wr) in successes], float)

    jbest = int(np.argmin(y_good))
    bx, by = float(X_good[jbest,0]), float(X_good[jbest,1])
    ybest = float(y_good[jbest])

    rng = np.random.default_rng(abs(hash((base_angle, seed))) % (2**32 - 1))

    nx, ny, meta = _gradient_mapping_step(
        X_good=X_good, y_good=y_good, bx=bx, by=by, R=global_clip,
        bo_step_mm=bo_step_mm, bo_backtrack=bo_backtrack, bo_trials=bo_trials,
        bo_noise=bo_noise, rng=rng
    )

    # Convergence controls (no patience/shrink): if step tiny or grad tiny, and we already good enough
    gnorm = float(meta.get("gnorm", 0.0))
    proposed_step = math.hypot(nx - bx, ny - by)
    if ((proposed_step < done_step_mm) or (gnorm < done_grad)) and (ybest <= done_wrmsd):
        return ("done", "done", "map_converged")

    # Avoid duplicates
    if (_fmt6(nx), _fmt6(ny)) in tried_set:
        dx = bx - nx; dy = by - ny
        norm = math.hypot(dx, dy)
        if norm > 1e-12:
            nx += 0.25 * dx; ny += 0.25 * dy

    nx = float(np.clip(nx, -global_clip, global_clip))
    ny = float(np.clip(ny, -global_clip, global_clip))
    return (float(nx), float(ny), "map_grad")

# ------------------------------ Main ------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-frame proposer: expanding ring -> shrinking rings on successes -> gradient-driven surface mapping.")
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Global half-width limit for |dx| and |dy|.")

    # Ring parameters
    ap.add_argument("--r-max", type=float, default=0.05, help="Initial ring maximum radius (mm).")
    ap.add_argument("--r-step", type=float, default=0.01, help="Ring radial increment (mm).")
    ap.add_argument("--k-base", type=float, default=20.0, help="Angular density (angles per ring ∝ k_base * r/r_max).")
    ap.add_argument("--ring-shrink", type=float, default=0.90, help="Factor to reduce r_max and r_step after each success.")
    ap.add_argument("--center-offset-frac", type=float, default=0.10, help="Offset of new ring center = frac * current r_step.")

    # Switch to mapping after this many successful points (>=3).
    ap.add_argument("--min-good-for-best", type=int, default=4, help="Number of successful (indexed + finite wRMSD) points before mapping.")

    # Gradient-driven mapping (minimal knobs)
    ap.add_argument("--bo-step-mm", type=float, default=0.01, help="Max step length for gradient move (mm).")
    ap.add_argument("--bo-backtrack", type=float, default=0.6, help="Backtracking factor (0.3–0.95).")
    ap.add_argument("--bo-trials", type=int, default=4, help="Number of backtracking candidates to test (>=1).")
    ap.add_argument("--bo-noise", type=float, default=1e-4, help="GP observation noise for wRMSD.")

    # Done criteria (no patience)
    ap.add_argument("--done-step-mm", type=float, default=1e-3, help="Stop if proposed step from incumbent is smaller than this (mm).")
    ap.add_argument("--done-wrmsd", type=float, default=0.10, help="Stop if incumbent best wRMSD <= this.")
    ap.add_argument("--done-grad", type=float, default=5e-3, help="Stop if GP-mean gradient norm is below this (mm^-1).")

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
            bo_step_mm=float(args.bo_step_mm),
            bo_backtrack=float(args.bo_backtrack),
            bo_trials=int(args.bo_trials),
            bo_noise=float(args.bo_noise),
            done_step_mm=float(args.done_step_mm),
            done_wrmsd=float(args.done_wrmsd),
            done_grad=float(args.done_grad),
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
        print(f"[diag:{os.path.basename(key[0])}|event{key[1]}] reason={reason}")

    write_log(log_path, entries)
    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
