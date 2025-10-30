#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py

Per-frame proposer that combines:
  1) Adaptive expanding ring search (until enough successful points exist), then
  2) Local Bayesian Optimization (GP + EI) with an extra "sample-gradient" step.

Design goals
------------
• Frame-by-frame (per-event) proposals for the **latest run only**.
• Start with **ring search** from an initial center. While no index+finite-wRMSD
  results exist, keep expanding the ring.
• Once the **first success** is found, start a **local ring** around that success,
  with a small deterministic shift of the ring center to avoid sampling on a
  single circle; repeat until we have enough successful points (>= bo_min_good).
• When we have enough good points, switch to **BO** in a trust region around the
  current best: EI candidates + one extra candidate produced by a negative-
  gradient step of the GP posterior mean ("adaptive sample gradient search").
• Minimal tunables; most behavior adapts to the data.

CSV compatibility
-----------------
Preserves the grouped CSV format with section headers:
  "#/abs/path/to/file event <int>"
and data rows with columns:
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm

Usage (same style as your previous scripts)
-------------------------------------------
python3 propose_next_shifts_ring_bo.py --run-root <run_root> \
  [--r-max 0.05 --r-step 0.01 --k-base 20] \
  [--bo-min-good 4 --bo-lsx 0.02 --bo-lsy 0.02 --bo-noise 1e-4 --bo-candidates 800 --bo-ei-eps 1e-3] \
  [--seed 1337]

Author: ChatGPT
License: MIT
"""

from __future__ import annotations
import argparse, os, sys, math, statistics, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np

# ------------------------------ Small utilities ------------------------------

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

def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    """How many angular samples to use for a given ring radius r."""
    return max(1, math.ceil(k_base * (r / max(r_max, 1e-12))))

# ------------------------------ GP + EI (anisotropic RBF) ------------------------------

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
    """Zero-mean GP with RBF kernel (amplitude=1); returns (mu, var)."""
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

# ------------------------------ CSV I/O (grouped) ------------------------------

def parse_log(log_path: str):
    """Return list of (key,row) and latest_run. key=(abs_path,event_id) on header lines."""
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

# ------------------------------ Ring search logic ------------------------------

def _propose_expanding_ring(center: Tuple[float,float],
                            tried_set: set,
                            r_max: float, r_step: float, k_base: float,
                            angle_base: float,
                            grid_tol: float = 5e-7) -> Optional[Tuple[float,float]]:
    """
    Deterministic: iterate radii = r_step, 2*r_step, ..., r_max and angles based on n_angles_for_radius.
    Return the first candidate not already tried (w.r.t 6-decimal grid, with a small absolute tol).
    """
    cx, cy = center
    max_k = int(math.ceil(r_max / max(r_step, 1e-12)))
    for k in range(1, max_k + 1):
        r = k * r_step
        n = n_angles_for_radius(r, r_max, k_base)
        for i in range(n):
            theta = angle_base + 2.0 * math.pi * (i / n)
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            # Clip to global square box [-r_max, r_max] in each axis
            x = float(np.clip(x, -r_max, r_max))
            y = float(np.clip(y, -r_max, r_max))
            keyfmt = (_fmt6(x), _fmt6(y))
            if keyfmt in tried_set:
                continue
            # absolute tolerance to avoid tiny repeats
            if any(abs(float(t[0]) - x) < grid_tol and abs(float(t[1]) - y) < grid_tol for t in tried_set):
                continue
            return (x, y)
    return None

# ------------------------------ BO logic ------------------------------

def _trust_region_radius(best_xy: Tuple[float,float], XY_good: np.ndarray) -> float:
    """
    Trust-region half-width from data: clamp 2*median distance from best in [rho_min, rho_max].
    """
    d = np.sqrt(np.sum((XY_good - np.array(best_xy))**2, axis=1))
    if d.size >= 3 and np.all(np.isfinite(d)):
        rho = 2.0 * float(np.median(d))
    else:
        rho = 0.02
    return float(np.clip(rho, 0.006, 0.05))  # mm

def _bo_candidates(bx, by, rho, n, rng):
    """Local candidates around best + small global sprinkle inside [-rho, +rho] square."""
    Cl = max(50, int(0.85 * n))
    Cg = max(1, n - Cl)
    sx = 0.40 * rho
    sy = 0.40 * rho
    xs_loc = rng.normal(bx, sx, Cl)
    ys_loc = rng.normal(by, sy, Cl)
    xs_loc = np.clip(xs_loc, bx - rho, bx + rho)
    ys_loc = np.clip(ys_loc, by - rho, by + rho)
    xs_glb = rng.uniform(bx - rho, bx + rho, Cg)
    ys_glb = rng.uniform(by - rho, by + rho, Cg)
    return np.vstack([np.column_stack([xs_loc, ys_loc]), np.column_stack([xs_glb, ys_glb])])

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

def _gp_mean_at(points: np.ndarray, X: np.ndarray, y: np.ndarray, lsx: float, lsy: float, noise: float) -> np.ndarray:
    mu, _ = _gp_fit_predict(X, y, points, lsx, lsy, noise)
    return mu

def _gradient_step_candidate(bx, by, rho, X, y, lsx, lsy, noise) -> Optional[Tuple[float,float]]:
    """
    One negative-gradient step of the GP posterior mean at the incumbent.
    Finite differences of GP mean; step length ~ 0.5*rho.
    """
    h = 0.25 * rho
    P = np.array([[bx + h, by], [bx - h, by], [bx, by + h], [bx, by - h]], float)
    muP = _gp_mean_at(P, X, y, lsx, lsy, noise)
    dmu_dx = (muP[0] - muP[1]) / (2*h)
    dmu_dy = (muP[2] - muP[3]) / (2*h)
    gnorm = math.hypot(float(dmu_dx), float(dmu_dy))
    if not np.isfinite(gnorm) or gnorm < 1e-12:
        return None
    step = 0.5 * rho
    nx = float(bx - step * (dmu_dx / gnorm))
    ny = float(by - step * (dmu_dy / gnorm))
    nx = float(np.clip(nx, bx - rho, bx + rho))
    ny = float(np.clip(ny, by - rho, by + rho))
    return (nx, ny)

# ------------------------------ Main per-event proposal ------------------------------

def propose_for_latest(entries, latest_run: int,
                       r_max: float, r_step: float, k_base: float,
                       bo_min_good: int,
                       bo_lsx: float, bo_lsy: float, bo_noise: float,
                       bo_candidates: int, bo_ei_eps: float,
                       seed: int,
                       converge_tol: float = 1e-4):

    history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    latest_rows_by_key: Dict[Tuple[str,int], List[int]] = {}
    current_key: Optional[Tuple[str,int]] = None

    for idx, (key, row) in enumerate(entries):
        if key is not None:
            current_key = key
            continue
        if row is None or len(row) == 0:
            continue
        if row[0] == "RAW":
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

    proposals: Dict[Tuple[str,int], Tuple[object, object, str]] = {}  # ndx, ndy, reason

    for key, trials in history.items():
        if key not in latest_rows_by_key:
            continue
        trials_sorted = sorted(trials, key=lambda t: t[0])  # by run_n

        # Tried points (all) & good points (successful with wrmsd)
        tried_set = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)
        good = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]

        # Initial center = earliest det_shift for this event, else (0,0)
        if len(trials_sorted) > 0:
            init_cx, init_cy = trials_sorted[0][1], trials_sorted[0][2]
        else:
            init_cx, init_cy = 0.0, 0.0

        base_angle = _hash_angle(seed, key)

        # ----------------- Phase 1: global ring until first success -----------------
        if len(good) == 0:
            nd = _propose_expanding_ring((init_cx, init_cy), tried_set, r_max, r_step, k_base, base_angle)
            if nd is None:
                proposals[key] = ("done", "done", "ring_exhausted_no_success")
            else:
                proposals[key] = (nd[0], nd[1], "ring_global")
            continue

        # ----------------- Phase 2: local rings around success until enough points -----------------
        if len(good) < max(3, bo_min_good):
            # Choose the most recent success as local center; add small deterministic shift
            last_success_idx = max(i for i,(_,_,_,ind,wr) in enumerate(trials_sorted) if ind and (wr is not None) and math.isfinite(wr))
            last_sx, last_sy = trials_sorted[last_success_idx][1], trials_sorted[last_success_idx][2]
            # Small shift = 0.25 * r_step in direction given by base_angle (per event)
            eps = 0.25 * r_step
            scx = last_sx + eps * math.cos(base_angle + 0.5*math.pi)  # perpendicular
            scy = last_sy + eps * math.sin(base_angle + 0.5*math.pi)
            nd = _propose_expanding_ring((scx, scy), tried_set, r_max, r_step, k_base, base_angle)
            if nd is None:
                # Fallback: try ring centered exactly at last success
                nd = _propose_expanding_ring((last_sx, last_sy), tried_set, r_max, r_step, k_base, base_angle + math.pi/7.0)
            if nd is None:
                proposals[key] = ("done", "done", "local_ring_exhausted_preBO")
            else:
                proposals[key] = (nd[0], nd[1], "ring_local")
            continue

        # ----------------- Phase 3: BO with adaptive sample-gradient -----------------
        XY_good = np.array([[dx,dy] for (dx,dy,wr) in good], float)
        wr_good = np.array([wr for (dx,dy,wr) in good], float)

        jbest = int(np.argmin(wr_good))
        bx, by = float(XY_good[jbest,0]), float(XY_good[jbest,1])
        y_best = float(wr_good[jbest])

        # Trust region
        rho = _trust_region_radius((bx,by), XY_good)
        xmin, xmax = bx - rho, bx + rho
        ymin, ymax = by - rho, by + rho

        # Candidate set: EI candidates + one gradient-step candidate
        rng = np.random.default_rng(abs(hash((key[0], key[1], seed))) % (2**32 - 1))
        Xc = _bo_candidates(bx, by, rho, int(bo_candidates), rng)

        # add gradient-step candidate
        grad_xy = _gradient_step_candidate(bx, by, rho, XY_good, wr_good, bo_lsx, bo_lsy, bo_noise)
        if grad_xy is not None:
            Xc = np.vstack([Xc, np.array([[grad_xy[0], grad_xy[1]]], float)])

        # Drop near-duplicates
        Xseen = np.array([[dx,dy] for (_,dx,dy,_,_) in trials_sorted], float) if len(trials_sorted) > 0 else np.empty((0,2), float)
        keep = _filter_seen(Xc, Xseen, eps=0.001 * max(rho, 1e-6))
        Xc = Xc[keep] if np.any(keep) else Xc

        if Xc.shape[0] == 0:
            proposals[key] = ("done", "done", "bo_no_novel_candidates")
            continue

        mu, var = _gp_fit_predict(XY_good, wr_good, Xc, bo_lsx, bo_lsy, bo_noise)
        ei = _expected_improvement(mu, var, y_best, xi=0.0)
        if ei.size == 0:
            proposals[key] = ("done", "done", "bo_ei_empty")
            continue

        j = int(np.argmax(ei))
        ei_max = float(ei[j])

        if not np.isfinite(ei_max) or ei_max < float(bo_ei_eps):
            proposals[key] = ("done", "done", "bo_ei_small")
            continue

        nx, ny = float(Xc[j,0]), float(Xc[j,1])

        # Tiny-step / duplicate safety
        if math.hypot(nx - bx, ny - by) < converge_tol:
            proposals[key] = ("done", "done", "bo_converged_tiny_step")
            continue
        if (_fmt6(nx), _fmt6(ny)) in tried_set:
            proposals[key] = ("done", "done", "bo_duplicate_point")
            continue

        proposals[key] = (nx, ny, "bo_ei")

    # ----------------- Write proposals into latest run only -----------------
    n_new, n_done = 0, 0
    for key, idx_list in latest_rows_by_key.items():
        if key not in proposals:
            continue
        ndx, ndy, _ = proposals[key]
        for row_idx in idx_list:
            row = list(entries[row_idx][1])
            if len(row) < 7:
                row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):  # "done"
                    row[5] = "done"; row[6] = "done"; n_done += 1
                else:
                    row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy)); n_new += 1
                entries[row_idx] = (None, tuple(row))

    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    # Diagnostics per event
    # for key, (_, _, reason) in proposals.items():
    #     print(f"[diag:{os.path.basename(key[0])}|event{key[1]}] reason={reason}")

    return entries

# ------------------------------ CLI ------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-frame center-shift proposer: adaptive ring + BO with sample-gradient.")
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")

    # Ring search (global and local)
    ap.add_argument("--r-max", type=float, default=0.05, help="Max absolute shift magnitude (mm).")
    ap.add_argument("--r-step", type=float, default=0.01, help="Ring radial increment (mm).")
    ap.add_argument("--k-base", type=float, default=20.0, help="Angular density scale for ring (samples ∝ k_base * r/r_max).")

    # BO switch & model
    ap.add_argument("--bo-min-good", type=int, default=4, help="Min successful points before switching to BO (>=3).")
    ap.add_argument("--bo-lsx", type=float, default=0.02, help="GP lengthscale in x (mm).")
    ap.add_argument("--bo-lsy", type=float, default=0.02, help="GP lengthscale in y (mm).")
    ap.add_argument("--bo-noise", type=float, default=1e-4, help="GP observation noise.")
    ap.add_argument("--bo-candidates", type=int, default=800, help="Number of EI candidates.")
    ap.add_argument("--bo-ei-eps", type=float, default=1e-3, help="Stop BO when EI < this.")

    # Misc
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--converge-tol", type=float, default=1e-4, help="Stop if proposed step from incumbent is smaller than this (mm).")

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

    updated_entries = propose_for_latest(
        entries=entries,
        latest_run=latest_run,
        r_max=float(args.r_max),
        r_step=float(args.r_step),
        k_base=float(args.k_base),
        bo_min_good=max(3, int(args.bo_min_good)),
        bo_lsx=float(args.bo_lsx),
        bo_lsy=float(args.bo_lsy),
        bo_noise=float(args.bo_noise),
        bo_candidates=int(args.bo_candidates),
        bo_ei_eps=float(args.bo_ei_eps),
        seed=int(args.seed),
        converge_tol=float(args.converge_tol),
    )

    write_log(log_path, updated_entries)
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
