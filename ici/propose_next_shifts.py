
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py

Drop-in replacement for propose_next_shifts that preserves the original
grouped CSV parsing and "apply to latest run only" behavior, but replaces
the local Nelder–Mead refinement with Bayesian Optimization (GP + EI).

Keeps:
- ring search until first indexed+finite wrmsd
- same CSV columns and section headers
- same CLI for root discovery and geometry

Adds:
- BO flags for the local refinement phase

"""
from __future__ import annotations
import argparse, hashlib, math, os, sys
from typing import Dict, List, Tuple, Optional

import numpy as np

# ----------------- Defaults -----------------
# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
R_MAX_DEFAULT = 0.05
R_STEP_DEFAULT = 0.01
K_BASE_DEFAULT = 20.0
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 1e-4


# ----------------- BO2D code adapted from ici/bo2d_optimizer.py -----------------

from math import erf, sqrt

class BO2DConfig:
    def __init__(self, lsx=0.02, lsy=0.02, noise=1e-4, candidates=800, ei_eps=1e-3, rng_seed=1337):
        self.lsx = float(lsx)
        self.lsy = float(lsy)
        self.noise = float(noise)
        self.candidates = int(candidates)
        self.ei_eps = float(ei_eps)
        self.rng = np.random.default_rng(int(rng_seed))

# --------- Gaussian Process (RBF anisotropic) + EI ---------

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
    # Cholesky with tiny jitter fallback
    jitter = 1e-12
    try:
        L = _np.linalg.cholesky(K)
    except _np.linalg.LinAlgError:
        L = _np.linalg.cholesky(K + jitter * _np.eye(len(X)))
    alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, yc))
    Ks = _rbf_aniso(X, Xstar, lsx, lsy)
    mu = Ks.T @ alpha + ymean
    v = _np.linalg.solve(L, Ks)
    # Kernel amplitude assumed 1.0
    var = _np.maximum(0.0, 1.0 - _np.sum(v * v, axis=0))
    return mu, var

def _phi(z: np.ndarray) -> np.ndarray:
    # standard normal PDF
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * (z * z))

def _Phi(z: np.ndarray) -> np.ndarray:
    # use math.erf element-wise (NumPy lacks np.erf on some builds)
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
    Args:
        tried_xy: (n,2) past (dx,dy) in mm (finite wrmsd only).
        tried_wrmsd: (n,) past wrmsd values (lower is better).
        bounds: ((xmin,xmax),(ymin,ymax))
        config: BO2DConfig
        tried_tol: distance threshold to consider a point already tried.

    Returns:
        (next_xy, ei_max): tuple, or (None, 0.0) if no improvement expected.
    """
    assert tried_xy.ndim == 2 and tried_xy.shape[1] == 2, "tried_xy must be (n,2)"
    assert tried_wrmsd.ndim == 1 and tried_wrmsd.shape[0] == tried_xy.shape[0], "shape mismatch"

    (xmin, xmax), (ymin, ymax) = bounds
    X = tried_xy.astype(float)
    y = tried_wrmsd.astype(float)

    # Fit GP on tried points
    y_best = float(np.min(y))

    # Sample random candidates
    C = max(10, int(config.candidates))
    xs = config.rng.uniform(xmin, xmax, C)
    ys = config.rng.uniform(ymin, ymax, C)
    Xc = np.column_stack([xs, ys])

    # Drop candidates too close to tried points
    if len(X) > 0:
        d2 = np.sum((Xc[:, None, :] - X[None, :, :]) ** 2, axis=2)
        mind = np.sqrt(np.min(d2, axis=1))
        mask = mind > tried_tol
        Xc = Xc[mask]
        if Xc.shape[0] == 0:
            return None, 0.0

    mu, var = _gp_fit_predict(X, y, Xc, config.lsx, config.lsy, config.noise)
    ei = expected_improvement(mu, var, y_best, xi=0.0)

    if ei.size == 0:
        return None, 0.0

    j = int(np.argmax(ei))
    ei_max = float(ei[j])
    next_xy = (float(Xc[j, 0]), float(Xc[j, 1]))
    return next_xy, ei_max

# ----------------- Utilities from original behavior -----------------
def n_angles_for_radius(r: float, r_max: float, k_base: float) -> int:
    return max(1, math.ceil(k_base * (r / max(r_max, 1e-9))))

def _hash_angle(seed: int, key: Tuple[str, int]) -> float:
    h = hashlib.sha256()
    h.update(f"{seed}|{key[0]}|{key[1]}".encode("utf-8"))
    import struct
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

# ------------- CSV parsing/writing (compatible with original) -------------
def parse_log(log_path: str):
    entries = []
    latest_run = -1
    with open(log_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for ln in f:
            if ln.startswith("#/"):
                try:
                    path_part, ev_part = ln[1:].rsplit(" event ", 1)
                    ev = int(ev_part.strip())
                    key = (os.path.abspath(path_part.strip()), ev)
                    entries.append((key, None))
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

# ------------- Ring proposer (same behavior) -------------
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
                       bo_cfg):
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

    proposals: Dict[Tuple[str,int], Tuple[object, object]] = {}

    for key, trials in history.items():
        if key not in latest_rows_by_key:
            continue

        trials_sorted = sorted(trials, key=lambda t: t[0])  # by run_n

        # Initialize state like the original script
        state = {
            'phase': 'ring',
            'ring_cx': trials_sorted[0][1] if trials_sorted else 0.0,
            'ring_cy': trials_sorted[0][2] if trials_sorted else 0.0,
            'ring_step': 0,
            'ring_angle_idx': -1,
            'ring_angle_base': _hash_angle(seed, key),
            'give_up': False,
        }

        # Count ring attempts before first success
        k_ring = 0
        saw_success = False
        for (_, dx_t, dy_t, indexed_t, wr_t) in trials_sorted:
            if indexed_t and (wr_t is not None) and math.isfinite(wr_t):
                saw_success = True
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

        # Decide which phase we're in
        if len(good) == 0:
            # Still in ring exploration
            ndx, ndy = pick_ring_probe(state, r_step, r_max, k_base) if not state['give_up'] else (state['ring_cx'], state['ring_cy'])
            if state['give_up']:
                proposals[key] = ("done", "done")
                continue
            # ensure novelty
            for _ in range(200):
                keyfmt = (_fmt6(ndx), _fmt6(ndy))
                if keyfmt not in tried_points:
                    proposals[key] = (ndx, ndy)
                    break
                ndx, ndy = pick_ring_probe(state, r_step, r_max, k_base)
            else:
                proposals[key] = ("done", "done")
            continue

        # BO phase (local refinement)
        tried_xy = np.array([[dx,dy] for (dx,dy,wr) in good], float)
        tried_wr = np.array([wr for (dx,dy,wr) in good], float)

        # Bounds centered at ring center; can be adjusted to mechanical limits
        xmin, xmax = state['ring_cx'] - r_max, state['ring_cx'] + r_max
        ymin, ymax = state['ring_cy'] - r_max, state['ring_cy'] + r_max
        next_xy, ei_max = bo2d_propose(tried_xy, tried_wr, ((xmin, xmax), (ymin, ymax)), bo_cfg)

        if (next_xy is None) or (ei_max < bo_cfg.ei_eps):
            proposals[key] = ("done", "done")
            continue

        # Also stop if step from best is tiny
        jbest = int(np.argmin(tried_wr))
        best_xy = (float(tried_xy[jbest,0]), float(tried_xy[jbest,1]))
        if math.hypot(next_xy[0]-best_xy[0], next_xy[1]-best_xy[1]) < converge_tol:
            proposals[key] = ("done", "done")
            continue

        # Avoid repeats
        if (_fmt6(next_xy[0]), _fmt6(next_xy[1])) in tried_points:
            proposals[key] = ("done", "done")
            continue

        proposals[key] = (next_xy[0], next_xy[1])

    # Apply proposals to latest-run rows
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
    ap = argparse.ArgumentParser(description="Propose next center shifts using ring for unindexed and BO for indexed frames.")
    ap.add_argument("--run-root", default=None, help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.")
    ap.add_argument("--r-max", type=float, default=R_MAX_DEFAULT)
    ap.add_argument("--r-step", type=float, default=R_STEP_DEFAULT)
    ap.add_argument("--k-base", type=float, default=K_BASE_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--converge-tol", type=float, default=CONVERGE_TOL_DEFAULT)
    ap.add_argument("--sidecar", help="If provided, write proposals to this CSV instead of rewriting the log.")

    # BO params
    ap.add_argument("--bo-lengthscale-x", type=float, default=0.02, help="RBF lengthscale in x (mm)")
    ap.add_argument("--bo-lengthscale-y", type=float, default=0.02, help="RBF lengthscale in y (mm)")
    ap.add_argument("--bo-noise", type=float, default=1e-4, help="GP nugget noise in wrmsd^2 units")
    ap.add_argument("--bo-candidates", type=int, default=800, help="Number of random EI candidates")
    ap.add_argument("--bo-ei-eps", type=float, default=1e-3, help="EI threshold to mark done")
    ap.add_argument("--bo-max-evals-local", type=int, default=40, help="(Reserved) Max BO evals after first success")

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

    bo_cfg = BO2DConfig(lsx=args.bo_lengthscale_x,
                        lsy=args.bo_lengthscale_y,
                        noise=args.bo_noise,
                        candidates=args.bo_candidates,
                        ei_eps=args.bo_ei_eps,
                        rng_seed=args.seed)

    updated_entries = propose_for_latest(
        entries=entries,
        latest_run=latest_run,
        r_max=float(args.r_max),
        r_step=float(args.r_step),
        k_base=float(args.k_base),
        seed=int(args.seed),
        converge_tol=float(args.converge_tol),
        bo_cfg=bo_cfg,
    )

    if args.sidecar:
        with open(args.sidecar, "w", encoding="utf-8") as f:
            f.write("real_h5_path,event,run_n,next_dx_mm,next_dy_mm\n")
            # We must reconstruct latest_rows_by_key again or just rescan entries:
            # Simpler: scan entries for rows with this latest run and dump updated next_*.
            current_key = None
            for (key,row) in entries:
                if key is not None:
                    current_key = key
                    continue
                if row is None or row[0] == "RAW":
                    continue
                run_n = int(row[0])
                if run_n != latest_run:
                    continue
                next_dx, next_dy = row[5], row[6]
                if next_dx == "" and next_dy == "":
                    continue
                f.write(f"{current_key[0]},{current_key[1]},{run_n},{next_dx},{next_dy}\n")
        print(f"[propose] Wrote proposals to {args.sidecar}")
    else:
        write_log(log_path, updated_entries)
        print(f"[propose] Updated {log_path} with next_* for run_{latest_run:03d}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
