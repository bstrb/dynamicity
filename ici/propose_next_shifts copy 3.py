#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py  — ring → robust BO (GP+EI) with guard rails

Drop-in replacement that preserves your grouped CSV format and
"apply to latest run only" behavior, but with sturdier defaults
and a few additions that helped in your logs:

Additions / behavior:
- Ring search until first indexed+finite wRMSD, with deterministic angle base per (image,event).
- One-shot outward radial push when first successes are too central.
- Tiny cross seeding around the incumbent when data are very sparse.
- Anisotropic GP (RBF) with nugget; mixed local/global EI sampling.
- Trust region that adapts from median distances to incumbent.
- Optional annulus constraint (median radius ± half-width).
- Guard rails for EI: drop candidates with bad predicted mean unless var is large.
- Backtrack rule: if recent wrmsd regresses a lot, snap back to best/cross.
- Robust "done": best inside good band (median+MAD*mult) AND little recent improvement.

CLI is intentionally minimal; all knobs have conservative defaults.
"""

from __future__ import annotations
import argparse, hashlib, math, os, sys, statistics
from typing import Dict, List, Tuple, Optional

import numpy as np
from math import erf, sqrt

# ----------------- Defaults -----------------
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
R_MAX_DEFAULT = 0.060      # ring limit [mm]
R_STEP_DEFAULT = 0.020     # ring step [mm]
K_BASE_DEFAULT = 20.0      # ring: #angles at radius r ~ K_BASE * r/R_MAX
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 5e-4

# ----------------- GP + EI -----------------
class BO2DConfig:
    def __init__(self,
                 lsx=0.015, lsy=0.015, noise=3e-4,
                 candidates=600, ei_eps=2e-3, rng_seed=1337,
                 local_frac=0.85, local_sigma_scale=1.1,
                 max_step_mm=0.010,
                 mu_guard=0.08, var_guard=0.06):
        self.lsx = float(lsx)
        self.lsy = float(lsy)
        self.noise = float(noise)
        self.candidates = int(candidates)
        self.ei_eps = float(ei_eps)
        self.rng = np.random.default_rng(int(rng_seed))
        self.local_frac = float(local_frac)
        self.local_sigma_scale = float(local_sigma_scale)
        self.max_step_mm = float(max_step_mm) if max_step_mm is not None else None
        # EI guard-rails
        self.mu_guard = float(mu_guard)
        self.var_guard = float(var_guard)
        # optional annulus
        self.ann_center = None   # (cx, cy)
        self.ann_r_med = None
        self.ann_half = None

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx*dx + dy*dy))

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
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
    var = _np.maximum(0.0, 1.0 - _np.sum(v * v, axis=0))
    return mu, var

def _phi(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (z * z))

def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

def expected_improvement(mu: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
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
    (xmin, xmax), (ymin, ymax) = bounds
    X = tried_xy.astype(float)
    y = tried_wrmsd.astype(float)

    y_best = float(np.min(y))
    jbest = int(np.argmin(y))
    bx, by = float(X[jbest,0]), float(X[jbest,1])

    # Mixture sampling
    C = max(10, int(config.candidates))
    Cg = max(1, int((1.0 - config.local_frac) * C))
    Cl = C - Cg

    parts = []
    if Cl > 0 and len(X) > 0:
        sx = config.local_sigma_scale * max(config.lsx, 1e-9)
        sy = config.local_sigma_scale * max(config.lsy, 1e-9)
        xs_loc = config.rng.normal(bx, sx, Cl)
        ys_loc = config.rng.normal(by, sy, Cl)
        parts.append(np.column_stack([np.clip(xs_loc, xmin, xmax),
                                      np.clip(ys_loc, ymin, ymax)]))
    if Cg > 0:
        xs = config.rng.uniform(xmin, xmax, Cg)
        ys = config.rng.uniform(ymin, ymax, Cg)
        parts.append(np.column_stack([xs, ys]))
    Xc = np.vstack(parts) if parts else np.empty((0,2), float)

    # Optional annulus restriction
    if (config.ann_center is not None) and (config.ann_r_med is not None) and (config.ann_half is not None) and (Xc.size > 0):
        cx, cy = config.ann_center
        r = np.sqrt((Xc[:,0]-cx)**2 + (Xc[:,1]-cy)**2)
        ann_ok = (r >= (config.ann_r_med - config.ann_half)) & (r <= (config.ann_r_med + config.ann_half))
        if np.any(ann_ok):
            Xc = Xc[ann_ok]

    # Drop too-close candidates
    if len(X) > 0 and Xc.shape[0] > 0:
        d2 = np.sum((Xc[:, None, :] - X[None, :, :]) ** 2, axis=2)
        mind = np.sqrt(np.min(d2, axis=1))
        Xc = Xc[mind > tried_tol]
        if Xc.shape[0] == 0:
            return None, 0.0

    mu, var = _gp_fit_predict(X, y, Xc, config.lsx, config.lsy, config.noise)
    ei = expected_improvement(mu, var, y_best, xi=0.0)

    # Guard-rail: keep if mean looks reasonable or variance is large
    if Xc.shape[0] > 0:
        keep = (mu <= (y_best + config.mu_guard)) | (var >= config.var_guard)
        if np.any(keep):
            Xc, mu, var, ei = Xc[keep], mu[keep], var[keep], ei[keep]
        if Xc.shape[0] == 0:
            return None, 0.0

    j = int(np.argmax(ei))
    return (float(Xc[j,0]), float(Xc[j,1])), float(ei[j])

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
    dev = [abs(x-med) for x in xs]
    return statistics.median(dev)

# ------------- CSV I/O -------------
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
                       bo_rho_init=0.035, bo_rho_min=0.003, bo_rho_max=0.060,
                       radial_push_r_thresh=0.030,
                       radial_push_min=0.012,
                       small_cross_delta=0.006,
                       backtrack_delta=0.060,
                       annulus_half=0.012,
                       robust_done_window=6,
                       robust_done_improve=0.005,
                       robust_done_mad_mult=2.0):
    """
    Update entries in-place with next_dx/dy for the latest run only.
    """
    # Build history
    history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    latest_rows_by_key: Dict[Tuple[str,int], List[int]] = {}
    current_key: Optional[Tuple[str,int]] = None
    per_event_best_wr = {}

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

        if indexed and (wr is not None) and math.isfinite(wr):
            prev = per_event_best_wr.get(current_key)
            if prev is None or wr < prev:
                per_event_best_wr[current_key] = wr

    # Robust "good band" threshold from DONE events (fallback: all bests)
    done_event_bests = []
    for key, idx_list in latest_rows_by_key.items():
        if not idx_list: continue
        last_row = entries[idx_list[-1]][1]
        if last_row is None or last_row[0] == "RAW": continue
        if last_row[5] == "done" and last_row[6] == "done" and key in per_event_best_wr:
            done_event_bests.append(per_event_best_wr[key])
    pool = done_event_bests if done_event_bests else list(per_event_best_wr.values())
    if pool:
        band_med = statistics.median(pool); band_mad = _mad(pool)
        good_band_upper = band_med + robust_done_mad_mult * band_mad
    else:
        good_band_upper = float("inf")

    proposals: Dict[Tuple[str,int], Tuple[object, object]] = {}

    # Per-event proposal
    for key, trials in history.items():
        if key not in latest_rows_by_key:
            continue
        trials_sorted = sorted(trials, key=lambda t: t[0])
        ring_cx = trials_sorted[0][1] if trials_sorted else 0.0
        ring_cy = trials_sorted[0][2] if trials_sorted else 0.0
        state = {
            'ring_cx': ring_cx, 'ring_cy': ring_cy,
            'ring_step': 0, 'ring_angle_idx': -1,
            'ring_angle_base': _hash_angle(seed, key),
            'give_up': False,
        }

        # Advance ring counters to mirror history until first success
        for (_, dx_t, dy_t, ind_t, wr_t) in trials_sorted:
            if ind_t and (wr_t is not None) and math.isfinite(wr_t):
                break
            if abs(dx_t - ring_cx) < 1e-12 and abs(dy_t - ring_cy) < 1e-12:
                continue
            r_tmp = (state['ring_step'] + 1) * r_step
            if r_tmp > r_max + 1e-12:
                state['give_up'] = True
                break
            n_tmp = n_angles_for_radius(r_tmp, r_max, k_base)
            state['ring_angle_idx'] = (state['ring_angle_idx'] + 1) % n_tmp
            if state['ring_angle_idx'] == n_tmp - 1:
                state['ring_step'] += 1
                state['ring_angle_idx'] = -1

        good = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
        tried_points = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)

        # Ring phase
        if len(good) == 0:
            if state['give_up']:
                proposals[key] = ("done", "done"); continue
            for _ in range(200):
                ndx, ndy = pick_ring_probe(state, r_step, r_max, k_base)
                if (_fmt6(ndx), _fmt6(ndy)) not in tried_points:
                    proposals[key] = (ndx, ndy); break
            else:
                proposals[key] = ("done", "done")
            continue

        # BO phase
        tried_xy = np.array([[dx,dy] for (dx,dy,wr) in good], float)
        tried_wr = np.array([wr for (dx,dy,wr) in good], float)
        jbest = int(np.argmin(tried_wr))
        best_xy = (float(tried_xy[jbest,0]), float(tried_xy[jbest,1]))
        y_best = float(tried_wr[jbest])

        # Robust done
        if len(tried_wr) >= 3 and y_best <= good_band_upper:
            best_seq = []
            m = float('inf')
            for w in tried_wr:
                m = min(m, float(w)); best_seq.append(m)
            recent = best_seq[-min(robust_done_window, len(best_seq)):]
            if (recent[0] - recent[-1]) < robust_done_improve:
                proposals[key] = ("done", "done"); continue

        # Backtrack on regression
        last_dx, last_dy, last_wr = good[-1]
        if last_wr > (y_best + backtrack_delta):
            bx, by = best_xy
            if (_fmt6(bx), _fmt6(by)) not in tried_points:
                proposals[key] = (bx, by); continue
            step = max(small_cross_delta, 0.5*(bo_cfg.lsx + bo_cfg.lsy))
            for cx_, cy_ in [(bx+step,by), (bx-step,by), (bx,by+step), (bx,by-step)]:
                if (_fmt6(cx_), _fmt6(cy_)) not in tried_points:
                    proposals[key] = (cx_, cy_); break
            else:
                proposals[key] = ("done", "done")
            continue

        # Center→best vector & radius
        cx_center, cy_center = ring_cx, ring_cy
        bx, by = best_xy
        vx, vy = (bx - cx_center), (by - cy_center)
        r_best = math.hypot(vx, vy)
        if r_best < 1e-12:
            vx, vy = -1.0, -1.0; r_best = math.sqrt(2.0)
        ux, uy = (vx / r_best, vy / r_best)

        # One-shot outward push if too central or too few points
        need_push = (len(tried_xy) <= 2) or (r_best < radial_push_r_thresh)
        if need_push:
            ls_mean = 0.5 * (bo_cfg.lsx + bo_cfg.lsy)
            hard_cap = 0.02
            step_cap = float(bo_cfg.max_step_mm) if (bo_cfg.max_step_mm is not None) else ls_mean
            push = max(r_step, min(step_cap, hard_cap))
            if r_best < 0.02:
                push = max(push, radial_push_min)
            cand_r = min(r_best + push, float(r_max))
            cand_xy = (cx_center + cand_r * ux, cy_center + cand_r * uy)
            if (_fmt6(cand_xy[0]), _fmt6(cand_xy[1])) not in tried_points:
                proposals[key] = (cand_xy[0], cand_xy[1]); continue

        # Tiny cross when single success
        if len(tried_xy) == 1:
            delta = max(small_cross_delta, 0.5*(bo_cfg.lsx + bo_cfg.lsy))
            for cx_, cy_ in [(bx+delta,by), (bx-delta,by), (bx,by+delta), (bx,by-delta)]:
                if (_fmt6(cx_), _fmt6(cy_)) not in tried_points:
                    proposals[key] = (cx_, cy_); break
            if key in proposals: continue

        # Trust region around best
        dists = np.sqrt(np.sum((tried_xy - np.array(best_xy))**2, axis=1))
        if len(dists) >= 3:
            rho = float(np.clip(2.0 * np.median(dists), bo_rho_min, bo_rho_max))
        else:
            rho = float(np.clip(bo_rho_init, bo_rho_min, bo_rho_max))
        xmin, xmax = best_xy[0] - rho, best_xy[0] + rho
        ymin, ymax = best_xy[1] - rho, best_xy[1] + rho

        # Annulus constraint (median radius ± half)
        radii = np.sqrt((tried_xy[:,0]-cx_center)**2 + (tried_xy[:,1]-cy_center)**2)
        r_med = float(np.median(radii)) if len(tried_xy) >= 3 else None
        if r_med and math.isfinite(r_med):
            bo_cfg.ann_center = (cx_center, cy_center)
            bo_cfg.ann_r_med = r_med
            bo_cfg.ann_half = float(annulus_half)
        else:
            bo_cfg.ann_center = bo_cfg.ann_r_med = bo_cfg.ann_half = None

        next_xy, ei_max = bo2d_propose(tried_xy, tried_wr, ((xmin, xmax), (ymin, ymax)), bo_cfg)
        if (next_xy is None) or (ei_max < bo_cfg.ei_eps):
            proposals[key] = ("done", "done"); continue

        # Hard cap from incumbent
        if (bo_cfg.max_step_mm is not None):
            dxs, dys = next_xy[0] - bx, next_xy[1] - by
            r = math.hypot(dxs, dys)
            cap = float(bo_cfg.max_step_mm)
            if r > cap and r > 0.0:
                s = cap / r
                next_xy = (bx + s*dxs, by + s*dys)

        # Tiny-move stop
        if math.hypot(next_xy[0]-bx, next_xy[1]-by) < converge_tol:
            proposals[key] = ("done", "done"); continue

        if (_fmt6(next_xy[0]), _fmt6(next_xy[1])) in tried_points:
            proposals[key] = ("done", "done"); continue

        proposals[key] = (next_xy[0], next_xy[1])

    # Apply proposals to latest-run rows
    n_new, n_done = 0, 0
    for key, idx_list in latest_rows_by_key.items():
        if key not in proposals: continue
        ndx, ndy = proposals[key]
        for row_idx in idx_list:
            row = list(entries[row_idx][1])
            if len(row) < 7: row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):
                    row[5] = "done"; row[6] = "done"; n_done += 1
                else:
                    row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy)); n_new += 1
                entries[row_idx] = (None, tuple(row))

    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    return entries

# ------------- CLI -------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose next detector shifts: ring until first success, then robust BO (GP+EI).")
    ap.add_argument("--run-root", default=None, help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.")
    ap.add_argument("--r-max", type=float, default=R_MAX_DEFAULT)
    ap.add_argument("--r-step", type=float, default=R_STEP_DEFAULT)
    ap.add_argument("--k-base", type=float, default=K_BASE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--converge-tol", type=float, default=CONVERGE_TOL_DEFAULT)
    ap.add_argument("--sidecar", help="If provided, write proposals to this CSV instead of rewriting the log.")

    # GP/BO params
    ap.add_argument("--bo-lengthscale-x", type=float, default=0.015)
    ap.add_argument("--bo-lengthscale-y", type=float, default=0.015)
    ap.add_argument("--bo-noise", type=float, default=3e-4)
    ap.add_argument("--bo-candidates", type=int, default=600)
    ap.add_argument("--bo-ei-eps", type=float, default=2e-3)
    ap.add_argument("--bo-local-frac", type=float, default=0.85)
    ap.add_argument("--bo-local-sigma-scale", type=float, default=1.1)
    ap.add_argument("--bo-max-step-mm", type=float, default=0.010)

    # Trust region
    ap.add_argument("--bo-rho-init", type=float, default=0.035)
    ap.add_argument("--bo-rho-min", type=float, default=0.003)
    ap.add_argument("--bo-rho-max", type=float, default=0.060)

    # Radial push / cross / backtrack
    ap.add_argument("--radial-push-r-thresh", type=float, default=0.030)
    ap.add_argument("--radial-push-min", type=float, default=0.012)
    ap.add_argument("--small-cross-delta", type=float, default=0.006)
    ap.add_argument("--backtrack-delta", type=float, default=0.060)

    # Annulus + done
    ap.add_argument("--annulus-half", type=float, default=0.012)
    ap.add_argument("--robust-done-window", type=int, default=6)
    ap.add_argument("--robust-done-improve", type=float, default=0.005)
    ap.add_argument("--robust-done-mad-mult", type=float, default=2.0)

    args = ap.parse_args(argv)

    run_root = os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_ROOT))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr); return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr); return 2

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
    )

    # Guard thresholds can be tuned via env if needed
    mu_guard = float(os.environ.get("PROPOSE_MU_GUARD", bo_cfg.mu_guard))
    var_guard = float(os.environ.get("PROPOSE_VAR_GUARD", bo_cfg.var_guard))
    bo_cfg.mu_guard = mu_guard
    bo_cfg.var_guard = var_guard

    updated_entries = propose_for_latest(
        entries=entries,
        latest_run=latest_run,
        r_max=float(args.r_max),
        r_step=float(args.r_step),
        k_base=float(args.k_base),
        seed=int(args.seed),
        converge_tol=float(args.converge_tol),
        bo_cfg=bo_cfg,
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
            for (key,row) in entries:
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
