#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py  —  Hybrid Ring + Robust BO (GP+EI)

Purpose
-------
Drop-in replacement for your proposer that keeps the original grouped CSV I/O
and "apply to latest run only" behavior, while combining the strongest parts
from your original ring+local-refine and advanced pure-BO variants:

1) Expanding ring until the first successful, finite wRMSD ("indexed") frame.
2) Robust Bayesian Optimization (anisotropic RBF GP + Expected Improvement)
   for local refinement, with:
   - Mixed local/global candidate sampling around the incumbent.
   - Adaptive trust region (rho) from median distance of successful points.
   - Optional annulus constraint when a likely radius is learned.
   - Guard-rail filtering of EI candidates (bad mean unless variance large).
   - One-time outward radial push if the first success is too central.
   - Tiny cross seeding when data are very sparse.
   - Backtrack to incumbent if a clear regression was just observed.
   - Data-driven "done" check using a robust good-band threshold.

CLI (minimal knobs)
-------------------
  --r-max, --r-step, --k-base     Ring geometry
  --bo-...                        GP/BO knobs (lengthscales, noise, candidates, guards)
  --bo-rho-...                    Trust region half-size bounds
  --annulus-half                  Half-width of annulus (mm) around median radius of successes
  --robust-done-*                 Stopping rule sensitivity

CSV Format (unchanged)
----------------------
Header line:
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
Section headers:
  #/abs/path/to/file event <int>

Each data row belongs to the most recent section header.

"""

from __future__ import annotations
import argparse
import hashlib
import math
import os
import sys
import statistics
from typing import Dict, List, Tuple, Optional

import numpy as np
from math import erf, sqrt

# ----------------- Defaults -----------------
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
R_MAX_DEFAULT = 0.05
R_STEP_DEFAULT = 0.01
K_BASE_DEFAULT = 20.0
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 5e-4

# ----------------- GP + EI -----------------

class BO2DConfig:
    def __init__(self,
                 lsx=0.02, lsy=0.02, noise=3e-4,
                 candidates=700, ei_eps=2e-3, rng_seed=1337,
                 local_frac=0.85, local_sigma_scale=1.2,
                 max_step_mm=None,
                 mu_guard=0.10, var_guard=0.08):
        self.lsx = float(lsx)
        self.lsy = float(lsy)
        self.noise = float(noise)
        self.candidates = int(candidates)
        self.ei_eps = float(ei_eps)
        self.rng = np.random.default_rng(int(rng_seed))
        self.local_frac = float(local_frac)
        self.local_sigma_scale = float(local_sigma_scale)
        self.max_step_mm = None if max_step_mm is None else float(max_step_mm)
        # guard rails for candidate pruning
        self.mu_guard = float(mu_guard)
        self.var_guard = float(var_guard)
        # optional annulus constraint (center, r_med, half_width)
        self.ann_center = None   # (cx, cy)
        self.ann_r_med = None
        self.ann_half = None


def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))


def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
    # Center y for stability
    import numpy as _np
    ymean = float(_np.mean(y))
    yc = y - ymean
    K = _rbf_aniso(X, X, lsx, lsy) + (noise * _np.eye(len(X)))
    jitter = 1e-12
    try:
        L = _np.linalg.cholesky(K)
    except _np.linalg.LinAlgError:
        L = _np.linalg.cholesky(K + jitter * _np.eye(len(X)))
    alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, yc))
    Ks = _rbf_aniso(X, Xstar, lsx, lsy)
    mu = Ks.T @ alpha + ymean
    v = _np.linalg.solve(L, Ks)
    var = _np.maximum(0.0, 1.0 - _np.sum(v * v, axis=0))  # amplitude=1.0
    return mu, var


def _phi(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (z * z))


def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def expected_improvement(mu: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
    # Minimize: improvement = y_best - Y
    std = np.sqrt(np.maximum(var, 1e-16))
    z = (y_best - mu - xi) / std
    ei = (y_best - mu - xi) * _Phi(z) + std * _phi(z)
    ei[var < 1e-30] = 0.0
    return ei


def bo2d_propose(tried_xy: np.ndarray,
                 tried_wrmsd: np.ndarray,
                 bounds: tuple,
                 config: BO2DConfig,
                 tried_tol: float = 1e-6):
    """
    Returns:
        (next_xy, ei_max): tuple, or (None, 0.0) if no improvement expected.
    """
    assert tried_xy.ndim == 2 and tried_xy.shape[1] == 2, "tried_xy must be (n,2)"
    assert tried_wrmsd.ndim == 1 and tried_wrmsd.shape[0] == tried_xy.shape[0], "shape mismatch"

    (xmin, xmax), (ymin, ymax) = bounds
    X = tried_xy.astype(float)
    y = tried_wrmsd.astype(float)

    y_best = float(np.min(y))
    jbest = int(np.argmin(y))
    bx, by = float(X[jbest,0]), float(X[jbest,1])

    # Candidate sampler: local Gaussian around best + global uniform
    C = max(10, int(config.candidates))
    Cg = max(1, int((1.0 - config.local_frac) * C))
    Cl = C - Cg

    parts = []
    if Cl > 0 and len(X) > 0:
        sx = config.local_sigma_scale * max(config.lsx, 1e-9)
        sy = config.local_sigma_scale * max(config.lsy, 1e-9)
        xs_loc = config.rng.normal(bx, sx, Cl)
        ys_loc = config.rng.normal(by, sy, Cl)
        xs_loc = np.clip(xs_loc, xmin, xmax)
        ys_loc = np.clip(ys_loc, ymin, ymax)
        parts.append(np.column_stack([xs_loc, ys_loc]))

    if Cg > 0:
        xs_glb = config.rng.uniform(xmin, xmax, Cg)
        ys_glb = config.rng.uniform(ymin, ymax, Cg)
        parts.append(np.column_stack([xs_glb, ys_glb]))

    Xc = np.vstack(parts) if parts else np.empty((0,2), float)

    # Optional annulus restriction relative to center
    if (config.ann_center is not None) and (config.ann_r_med is not None) and (config.ann_half is not None) and (Xc.size > 0):
        cx, cy = config.ann_center
        r = np.sqrt((Xc[:,0]-cx)**2 + (Xc[:,1]-cy)**2)
        ann_ok = (r >= (config.ann_r_med - config.ann_half)) & (r <= (config.ann_r_med + config.ann_half))
        if np.any(ann_ok):
            Xc = Xc[ann_ok]

    # Drop candidates too close to tried points
    if len(X) > 0 and Xc.shape[0] > 0:
        d2 = np.sum((Xc[:, None, :] - X[None, :, :]) ** 2, axis=2)
        mind = np.sqrt(np.min(d2, axis=1))
        mask = mind > tried_tol
        Xc = Xc[mask]
        if Xc.shape[0] == 0:
            return None, 0.0

    mu, var = _gp_fit_predict(X, y, Xc, config.lsx, config.lsy, config.noise)
    ei = expected_improvement(mu, var, y_best, xi=0.0)

    # Guard-rail pruning: keep if mean not far above best OR uncertainty large
    if Xc.shape[0] > 0:
        keep = (mu <= (y_best + config.mu_guard)) | (var >= config.var_guard)
        if np.any(keep):
            Xc, mu, var, ei = Xc[keep], mu[keep], var[keep], ei[keep]

    if ei.size == 0:
        return None, 0.0

    j = int(np.argmax(ei))
    ei_max = float(ei[j])
    next_xy = (float(Xc[j, 0]), float(Xc[j, 1]))
    return next_xy, ei_max


# ----------------- Utilities -----------------
def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    return max(1, math.ceil(k_base * (r / max(r_max, 1e-9))))


def _hash_angle(seed: int, key: Tuple[str, int]) -> float:
    h = hashlib.sha256()
    h.update(f"{seed}|{key[0]}|{key[1]}".encode("utf-8"))
    val = int.from_bytes(h.digest()[:8], "big")
    frac = (val & ((1<<53)-1)) / float(1<<53)
    return 2.0 * math.pi * frac


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


def _mad(xs: List[float]) -> float:
    if not xs:
        return 0.0
    med = statistics.median(xs)
    dev = [abs(x - med) for x in xs]
    return statistics.median(dev)


# ------------- CSV parsing/writing -------------
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


# ------------- Ring proposer -------------
def pick_ring_probe(state, r_step: float, r_max: float, k_base: float):
    r = (state['ring_step'] + 1) * r_step
    if r > r_max + 1e-12:
        state['give_up'] = True
        return state['ring_cx'], state['ring_cy']
    n = n_angles_for_radius(r, r_max, k_base)
    state['ring_angle_idx'] = (state['ring_angle_idx'] + 1) % n
    theta = (state['ring_angle_base'] or 0.0) + 2.0 * math.pi * state['ring_angle_idx'] / n
    ndx = state['ring_cx'] + r * math.cos(theta)
    ndy = state['ring_cy'] + r * math.sin(theta)
    if state['ring_angle_idx'] == n - 1:
        state['ring_step'] += 1
        state['ring_angle_idx'] = -1
    return ndx, ndy


# ------------- Main per-event proposal -------------
def propose_for_latest(entries, latest_run: int,
                       r_max, r_step, k_base,
                       seed, converge_tol,
                       bo_cfg: BO2DConfig,
                       bo_max_step_mm=None,
                       bo_rho_init=0.04, bo_rho_min=0.001, bo_rho_max=0.06,
                       radial_push_r_thresh=0.03,
                       radial_push_min=0.012,
                       small_cross_delta=0.006,
                       backtrack_delta=0.10,
                       annulus_half=0.015,
                       robust_done_window=5,
                       robust_done_improve=0.01,
                       robust_done_mad_mult=2.0):
    """
    Returns updated entries.
    """
    # ---- Build history and latest-row indices ----
    history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    latest_rows_by_key: Dict[Tuple[str,int], List[int]] = {}
    current_key: Optional[Tuple[str,int]] = None

    # Also collect "done" per-event best wrmsd to form a robust threshold
    per_event_best_wr = {}

    # First pass: group data and track per-event best wr
    key_latest_seen_run: Dict[Tuple[str,int], int] = {}
    for idx, (key, row) in enumerate(entries):
        if key is not None:
            current_key = key
            key_latest_seen_run[key] = -1
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
        key_latest_seen_run[current_key] = max(key_latest_seen_run[current_key], run_n)
        if run_n == latest_run:
            latest_rows_by_key.setdefault(current_key, []).append(idx)

        # track best wr so far for this event
        if indexed and (wr is not None) and math.isfinite(wr):
            prev = per_event_best_wr.get(current_key, None)
            if prev is None or wr < prev:
                per_event_best_wr[current_key] = wr

    # Identify events that are currently "done" (for robust band stats)
    done_event_bests = []
    for key, idx_list in latest_rows_by_key.items():
        if not idx_list:
            continue
        last_row = entries[idx_list[-1]][1]
        if last_row is None or last_row[0] == "RAW":
            continue
        ndx_s, ndy_s = last_row[5], last_row[6]
        if (ndx_s == "done") and (ndy_s == "done"):
            if key in per_event_best_wr:
                done_event_bests.append(per_event_best_wr[key])

    pool = done_event_bests if done_event_bests else list(per_event_best_wr.values())
    if pool:
        band_med = statistics.median(pool)
        band_mad = _mad(pool)
        good_band_upper = band_med + robust_done_mad_mult * band_mad
    else:
        good_band_upper = float("inf")

    proposals: Dict[Tuple[str,int], Tuple[object, object]] = {}

    # ---- Per-event proposal ----
    for key, trials in history.items():
        if key not in latest_rows_by_key:
            continue

        trials_sorted = sorted(trials, key=lambda t: t[0])  # by run_n

        # Initialize state
        state = {
            'phase': 'ring',
            'ring_cx': trials_sorted[0][1] if trials_sorted else 0.0,
            'ring_cy': trials_sorted[0][2] if trials_sorted else 0.0,
            'ring_step': 0,
            'ring_angle_idx': -1,
            'ring_angle_base': _hash_angle(seed, key),  # deterministic per (image,event)
            'give_up': False,
        }

        # Count ring attempts before first success
        k_ring = 0
        for (_, dx_t, dy_t, indexed_t, wr_t) in trials_sorted:
            if indexed_t and (wr_t is not None) and math.isfinite(wr_t):
                break
            if abs(dx_t - state['ring_cx']) < 1e-12 and abs(dy_t - state['ring_cy']) < 1e-12:
                continue
            k_ring += 1

        if k_ring > 0:
            for _ in range(k_ring):
                r_tmp = (state['ring_step'] + 1) * r_step
                if r_tmp > r_max + 1e-12:
                    state['give_up'] = True
                    break
                n_tmp = n_angles_for_radius(r_tmp, r_max, k_base)
                state['ring_angle_idx'] = (state['ring_angle_idx'] + 1) % n_tmp
                if state['ring_angle_idx'] == n_tmp - 1:
                    state['ring_step'] += 1
                    state['ring_angle_idx'] = -1

        # Build arrays of successful (indexed & finite wrmsd) trials
        good = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
        tried_points = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)

        # Ring phase
        if len(good) == 0:
            ndx, ndy = pick_ring_probe(state, r_step, r_max, k_base) if not state['give_up'] else (state['ring_cx'], state['ring_cy'])
            if state['give_up']:
                proposals[key] = ("done", "done")
                continue
            for _ in range(200):
                keyfmt = (_fmt6(ndx), _fmt6(ndy))
                if keyfmt not in tried_points:
                    proposals[key] = (ndx, ndy)
                    break
                ndx, ndy = pick_ring_probe(state, r_step, r_max, k_base)
            else:
                proposals[key] = ("done", "done")
            continue

        # ---------------- BO phase (local refinement) ----------------
        tried_xy = np.array([[dx,dy] for (dx,dy,wr) in good], float)
        tried_wr = np.array([wr for (dx,dy,wr) in good], float)

        # Current best so far
        jbest = int(np.argmin(tried_wr))
        best_xy = (float(tried_xy[jbest,0]), float(tried_xy[jbest,1]))
        y_best = float(tried_wr[jbest])

        # Robust "done" guard
        if len(tried_wr) >= 3 and y_best <= good_band_upper:
            recent_bests = np.minimum.accumulate(tried_wr)[-min(robust_done_window, len(tried_wr)):]
            if (recent_bests[0] - recent_bests[-1]) < robust_done_improve:
                proposals[key] = ("done", "done")
                continue

        # Backtrack rule
        last_dx, last_dy, last_wr = good[-1]
        if last_wr > (y_best + backtrack_delta):
            bx, by = best_xy
            if (_fmt6(bx), _fmt6(by)) not in tried_points:
                proposals[key] = (bx, by)
                continue
            step = max(small_cross_delta, 0.5*(bo_cfg.lsx + bo_cfg.lsy))
            for cx_, cy_ in [(bx+step,by), (bx-step,by), (bx,by+step), (bx,by-step)]:
                if (_fmt6(cx_), _fmt6(cy_)) not in tried_points:
                    proposals[key] = (cx_, cy_)
                    break
            else:
                proposals[key] = ("done", "done")
            continue

        # Ring center -> best vector and radius
        cx_center, cy_center = float(state['ring_cx']), float(state['ring_cy'])
        bx, by = best_xy
        vx, vy = (bx - cx_center), (by - cy_center)
        r_best = math.hypot(vx, vy)

        # One-time radial outward push when first success very central / data sparse
        need_radial_push = (len(tried_xy) <= 2) or (r_best < radial_push_r_thresh)
        if need_radial_push:
            if r_best < 1e-12:
                vx, vy = -1.0, -1.0
                r_best = math.sqrt(2.0)
            ux, uy = (vx / r_best, vy / r_best)

            ls_mean = 0.5 * (bo_cfg.lsx + bo_cfg.lsy)
            hard_cap = 0.02
            step_cap = float(bo_max_step_mm) if (bo_max_step_mm is not None) else ls_mean
            push = max(r_step, min(step_cap, hard_cap))
            if r_best < 0.02:
                push = max(push, radial_push_min)

            cand_r = min(r_best + push, float(r_max))
            cand_xy = (cx_center + cand_r * ux, cy_center + cand_r * uy)
            keyfmt_cand = (_fmt6(cand_xy[0]), _fmt6(cand_xy[1]))
            if keyfmt_cand not in tried_points:
                proposals[key] = (cand_xy[0], cand_xy[1])
                continue

        # Tiny cross when very sparse
        if len(tried_xy) == 1:
            delta = max(small_cross_delta, 0.5*(bo_cfg.lsx + bo_cfg.lsy))
            for cx_, cy_ in [(best_xy[0]+delta, best_xy[1]), (best_xy[0]-delta, best_xy[1]),
                             (best_xy[0], best_xy[1]+delta), (best_xy[0], best_xy[1]-delta)]:
                kf = (_fmt6(cx_), _fmt6(cy_))
                if kf not in tried_points:
                    proposals[key] = (cx_, cy_)
                    break
            if key in proposals:
                continue

        # Trust-region
        dists = np.sqrt(np.sum((tried_xy - np.array(best_xy))**2, axis=1))
        if len(dists) >= 3:
            rho = float(np.clip(2.0 * np.median(dists), bo_rho_min, bo_rho_max))
        else:
            rho = float(np.clip(bo_rho_init, bo_rho_min, bo_rho_max))

        xmin, xmax = best_xy[0] - rho, best_xy[0] + rho
        ymin, ymax = best_xy[1] - rho, best_xy[1] + rho

        # Annulus
        radii = np.sqrt((tried_xy[:,0]-cx_center)**2 + (tried_xy[:,1]-cy_center)**2)
        r_med = float(np.median(radii)) if len(radii) > 0 else float("nan")
        if len(tried_xy) >= 3 and math.isfinite(r_med):
            bo_cfg.ann_center = (cx_center, cy_center)
            bo_cfg.ann_r_med = r_med
            bo_cfg.ann_half = float(annulus_half)
        else:
            bo_cfg.ann_center = bo_cfg.ann_r_med = bo_cfg.ann_half = None

        next_xy, ei_max = bo2d_propose(tried_xy, tried_wr, ((xmin, xmax), (ymin, ymax)), bo_cfg)

        if (next_xy is None) or (ei_max < bo_cfg.ei_eps):
            proposals[key] = ("done", "done")
            continue

        if math.hypot(next_xy[0]-best_xy[0], next_xy[1]-best_xy[1]) < converge_tol:
            proposals[key] = ("done", "done")
            continue

        if (bo_max_step_mm is not None) and (next_xy is not None):
            bx, by = best_xy
            dxs, dys = next_xy[0] - bx, next_xy[1] - by
            r = math.hypot(dxs, dys)
            cap = float(bo_max_step_mm)
            if r > cap and r > 0.0:
                s = cap / r
                next_xy = (bx + s * dxs, by + s * dys)

        if (_fmt6(next_xy[0]), _fmt6(next_xy[1])) in tried_points:
            proposals[key] = ("done", "done")
            continue

        proposals[key] = (next_xy[0], next_xy[1])

    # ---- Apply proposals ----
    n_new, n_done = 0, 0
    for key, idx_list in latest_rows_by_key.items():
        if key not in proposals:
            continue
        ndx, ndy = proposals[key]
        for row_idx in idx_list:
            row = list(entries[row_idx][1])
            if len(row) < 7:
                row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):  # done
                    row[5] = "done"
                    row[6] = "done"
                    n_done += 1
                else:
                    row[5] = _fmt6(float(ndx))
                    row[6] = _fmt6(float(ndy))
                    n_new += 1
                entries[row_idx] = (None, tuple(row))

    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose next center shifts using ring (until first success) then robust BO around the incumbent.")
    ap.add_argument("--run-root", default=None, help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.")
    ap.add_argument("--r-max", type=float, default=R_MAX_DEFAULT)
    ap.add_argument("--r-step", type=float, default=R_STEP_DEFAULT)
    ap.add_argument("--k-base", type=float, default=K_BASE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--converge-tol", type=float, default=CONVERGE_TOL_DEFAULT)
    ap.add_argument("--sidecar", help="If provided, write proposals to this CSV instead of rewriting the log.")

    # GP/BO params
    ap.add_argument("--bo-lengthscale-x", type=float, default=0.02, help="RBF lengthscale in x (mm)")
    ap.add_argument("--bo-lengthscale-y", type=float, default=0.02, help="RBF lengthscale in y (mm)")
    ap.add_argument("--bo-noise", type=float, default=3e-4, help="GP nugget noise in wrmsd^2 units")
    ap.add_argument("--bo-candidates", type=int, default=700, help="Number of EI candidates")
    ap.add_argument("--bo-ei-eps", type=float, default=2e-3, help="EI threshold to mark done")

    # Candidate sampling mix
    ap.add_argument("--bo-local-frac", type=float, default=0.85, help="Fraction of EI candidates sampled near current best (0..1).")
    ap.add_argument("--bo-local-sigma-scale", type=float, default=1.2, help="Local Gaussian sigma = scale * lengthscale per axis.")
    ap.add_argument("--bo-max-step-mm", type=float, default=0.010, help="Hard cap on |step| from current best (mm) in BO phase.")

    # Trust region box half-sizes
    ap.add_argument("--bo-rho-init", type=float, default=0.04, help="Initial half-size (mm) for BO bounds around current best.")
    ap.add_argument("--bo-rho-min", type=float, default=0.001, help="Minimum BO half-size (mm).")
    ap.add_argument("--bo-rho-max", type=float, default=0.06, help="Maximum BO half-size (mm).")

    # Guard-rail thresholds
    ap.add_argument("--bo-mu-guard", type=float, default=0.10, help="Reject EI candidates with predicted mean > y_best + mu_guard unless var large.")
    ap.add_argument("--bo-var-guard", type=float, default=0.08, help="Allow EI candidates with variance >= var_guard even if mean looks worse.")

    # Radial push / seeding / backtrack
    ap.add_argument("--radial-push-r-thresh", type=float, default=0.03, help="If best radius < this, perform one outward radial push.")
    ap.add_argument("--radial-push-min", type=float, default=0.012, help="Minimum outward push when best is very central.")
    ap.add_argument("--small-cross-delta", type=float, default=0.006, help="Half-step for tiny cross seeding around incumbent (mm).")
    ap.add_argument("--backtrack-delta", type=float, default=0.10, help="If last_wr > best + delta, snap back toward best / cross.")

    # Annulus and robust-done
    ap.add_argument("--annulus-half", type=float, default=0.015, help="Half-width of annulus (mm) around median radius of successes.")
    ap.add_argument("--robust-done-window", type=int, default=5, help="Recent window length for improvement check.")
    ap.add_argument("--robust-done-improve", type=float, default=0.01, help="Required improvement over recent window to keep going.")
    ap.add_argument("--robust-done-mad-mult", type=float, default=2.0, help="Good-band upper = median + MAD*mult from DONE events.")

    args = ap.parse_args(argv)

    run_root = os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_ROOT))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr)
        return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr)
        return 2

    bo_cfg = BO2DConfig(
        lsx=args.bo_lengthscale_x,
        lsy=args.bo_lengthscale_y,
        noise=args.bo_noise,
        candidates=args.bo_candidates,
        ei_eps=args.bo_ei_eps,
        rng_seed=args.seed,
        local_frac=args.bo_local_frac,
        local_sigma_scale=args.bo_local_sigma_scale,
        max_step_mm=args.bo_max_step_mm,
        mu_guard=args.bo_mu_guard,
        var_guard=args.bo_var_guard,
    )

    updated_entries = propose_for_latest(
        entries=entries,
        latest_run=latest_run,
        r_max=float(args.r_max),
        r_step=float(args.r_step),
        k_base=float(args.k_base),
        seed=int(args.seed),
        converge_tol=float(args.converge_tol),
        bo_cfg=bo_cfg,
        bo_max_step_mm=args.bo_max_step_mm,
        bo_rho_init=args.bo_rho_init,
        bo_rho_min=args.bo_rho_min,
        bo_rho_max=args.bo_rho_max,
        radial_push_r_thresh=args.radial_push_r_thresh,
        radial_push_min=args.radial_push_min,
        small_cross_delta=args.small_cross_delta,
        backtrack_delta=args.backtrack_delta,
        annulus_half=args.annulus_half,
        robust_done_window=args.robust_done_window,
        robust_done_improve=args.robust_done_improve,
        robust_done_mad_mult=args.robust_done_mad_mult,
    )

    if args.sidecar:
        with open(args.sidecar, "w", encoding="utf-8") as f:
            f.write("real_h5_path,event,run_n,next_dx_mm,next_dy_mm\n")
            current_key = None
            for (key,row) in updated_entries:
                if key is not None:
                    current_key = key; continue
                if row is None or row[0] == "RAW": continue
                run_n = int(row[0])
                if run_n != latest_run: continue
                next_dx, next_dy = row[5], row[6]
                if next_dx == "" and next_dy == "": continue
                f.write(f"{current_key[0]},{current_key[1]},{run_n},{next_dx},{next_dy}\n")
        print(f"[propose] Wrote proposals to {args.sidecar}")
    else:
        write_log(log_path, updated_entries)
        print(f"[propose] Updated {log_path} with next_* for run_{latest_run:03d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
