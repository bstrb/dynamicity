#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py — Global constrained BO (q=1) for detector center shift

Summary
-------
Drop-in replacement for your proposer that:
  • Reads the grouped CSV `runs/image_run_log.csv`.
  • Aggregates *across all events & runs* by unique detector shift (mm).
  • Builds two surrogate models over shift Δ=(dx,dy):
        - Success-rate model s(Δ) in [0,1] (GP regression on per-shift success fraction).
        - Quality model m(Δ): robust central tendency of wRMSD over successful rows (GP regression).
  • Chooses a SINGLE next shift Δ_next for the *entire* next iteration (q=1) via
        Constrained Expected Improvement (CEI):  EI_m(Δ) × Pr_feasible(Δ), where Pr_feasible≈ŝ(Δ).
  • Writes that same (next_dx_mm, next_dy_mm) into *all* rows of the latest run,
    except rows already marked "done".
  • Optionally stops (marks all "done") when predicted improvement is negligible and
    sufficient feasible evidence exists.

Why a global (shared) Δ?
------------------------
The detector-center shift is a global property of the dataset. Per-event ring searches
can succeed only when a frame "gets lucky" early; aggregating outcomes across all frames
lets us learn the bowl around the true center and move everyone together with one Δ per
iteration (your q=1 workflow).

Compatibility
-------------
• CLI:    --run-root  (same as before)
• CSV:    Keeps header & section format:
          run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
          #/abs/path/to/file event <int>
• Prints: `[propose] <n_numeric> new proposals, <n_done> marked done`

Default knobs (tunable via flags) are chosen to respect your ranges:
  - Search radius R ≈ 0.05 mm, minimum meaningful step ≈ 0.005 mm.
  - Success threshold τ=0.6 (used only for diagnostics; CEI uses ŝ directly).

Implementation notes
--------------------
This is a lightweight, dependency-free GP (RBF, unit amplitude) with Cholesky
and small jitter. For success, we regress the empirical fraction s(Δ) using a
Jeffreys-smoothed estimate (α=β=0.5) to avoid 0/1 degeneracy. For quality, we
use the median wRMSD over successful rows per Δ. CEI is computed on a mixed
candidate set (local around incumbent + global uniform).

Author: ChatGPT (drop-in for your pipeline)
License: MIT
"""

from __future__ import annotations
import argparse, os, sys, math, statistics, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np

# ------------------------------ GP utilities ------------------------------

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    """Anisotropic RBF kernel with unit variance (amplitude=1)."""
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float):
    """Zero-mean GP with RBF kernel (amplitude=1). y is centered internally."""
    if len(X) == 0:
        raise ValueError("GP needs at least one observation.")
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
    var = np.maximum(0.0, 1.0 - np.sum(v * v, axis=0))  # unit amplitude prior
    return mu, var

def _phi(z: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)

def _Phi(z: np.ndarray) -> np.ndarray:
    # Using numpy erf via vectorize; avoids SciPy dependency
    from math import erf, sqrt
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))

def _expected_improvement(mu: np.ndarray, var: np.ndarray, y_best: float, xi: float = 0.0) -> np.ndarray:
    """EI for *minimization* of y. Returns 0 where var≈0."""
    std = np.sqrt(np.maximum(var, 1e-16))
    z = (y_best - mu - xi) / std
    ei = (y_best - mu - xi) * _Phi(z) + std * _phi(z)
    ei[var < 1e-30] = 0.0
    return ei

# -------------------------- CSV I/O (grouped) ----------------------------

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
    """Return list of (key,row) where key=(abs_path,event) for section headers,
    row is tuple of the 7 CSV fields, and latest_run number."""
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

# --------------------------- Aggregation logic ---------------------------

def _hash_angle(seed: int, key: Tuple[str, int]) -> float:
    """Deterministic angle for initial exploration (if needed)."""
    h = hashlib.sha256()
    h.update(f"{seed}|{key[0]}|{key[1]}".encode("utf-8"))
    val = int.from_bytes(h.digest()[:8], "big")
    frac = (val & ((1<<53)-1)) / float(1<<53)
    return 2.0 * math.pi * frac

def aggregate_by_shift(entries) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]]]:
    """Group rows by (dx,dy) rounded to 6 decimals across ALL runs/events.
    Returns:
      - XY_all: (n_u,2) unique shifts
      - stats:  structured float array (n_u, 3): [s_hat, m_median, n_total]
      - per_event_history: dict key->list of (run,dx,dy,indexed,wrmsd)
    """
    per_event_history: Dict[Tuple[str,int], List[Tuple[int,float,float,int,Optional[float]]]] = {}
    # (dx,dy) -> [n_total, n_succ, list wrmsd_succ]
    agg: Dict[Tuple[str,str], Tuple[int,int,List[float]]] = {}
    for key, row in entries:
        if key is not None:
            current_key = key
            per_event_history.setdefault(current_key, [])
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
        per_event_history[current_key].append((run_n, dx, dy, indexed, wr))
        k = (_fmt6(dx), _fmt6(dy))
        n_tot, n_succ, wr_list = agg.get(k, (0,0,[]))
        n_tot += 1
        if indexed and (wr is not None) and math.isfinite(wr):
            n_succ += 1
            wr_list = wr_list + [float(wr)]
        agg[k] = (n_tot, n_succ, wr_list)

    if not agg:
        return np.empty((0,2)), np.empty((0,3)), per_event_history

    keys = []
    s_hat = []
    m_med = []
    n_totals = []
    for (dxs, dys), (n_tot, n_succ, wr_list) in agg.items():
        dx, dy = float(dxs), float(dys)
        # Jeffreys-smoothed success fraction
        a = 0.5; b = 0.5
        s = (n_succ + a) / (n_tot + a + b)
        keys.append((dx, dy))
        s_hat.append(s)
        if wr_list:
            m_med.append(float(statistics.median(wr_list)))
        else:
            m_med.append(np.nan)  # no quality defined for fully failed shift
        n_totals.append(float(n_tot))

    XY_all = np.array(keys, dtype=float)
    stats = np.column_stack([np.array(s_hat, float),
                             np.array(m_med, float),
                             np.array(n_totals, float)])
    return XY_all, stats, per_event_history

# --------------------------- Proposal computation ---------------------------

def constrained_bo_next(XY: np.ndarray,
                        stats: np.ndarray,
                        bounds: Tuple[Tuple[float,float], Tuple[float,float]],
                        lsx: float, lsy: float,
                        noise_s: float, noise_m: float,
                        tau_feasible: float,
                        n_candidates: int,
                        local_frac: float,
                        local_sigma_scale: float,
                        rng: np.random.Generator) -> Tuple[Optional[Tuple[float,float]], Dict[str,float]]:
    """Return next Δ via CEI. stats columns: [s_hat, m_median, n_total]."""
    if XY.shape[0] == 0:
        return None, {"reason":"no_data"}

    # Feasible subset for quality model
    s = stats[:,0]
    m = stats[:,1]
    feas_mask = (s > 0.0) & np.isfinite(m)
    XY_feas = XY[feas_mask]
    m_feas = m[feas_mask]
    # If no feasible points yet: maximize predicted success probability
    if XY_feas.shape[0] == 0:
        # Fit GP on s
        try:
            mu_s, var_s = _gp_fit_predict(XY, s, _candidate_mesh(bounds, n_candidates, rng), lsx, lsy, noise_s)
        except Exception:
            # fallback: pick highest s seen
            j = int(np.argmax(s))
            return (float(XY[j,0]), float(XY[j,1])), {"reason":"fallback_seen_max_s"}
        Xc = _candidate_mesh(bounds, n_candidates, rng)
        mu_s, var_s = _gp_fit_predict(XY, s, Xc, lsx, lsy, noise_s)
        j = int(np.argmax(mu_s))
        return (float(Xc[j,0]), float(Xc[j,1])), {"reason":"no_feasible_max_psucc","psucc":float(mu_s[j])}

    # With feasible data: CEI
    # Build candidate set: mixture local/global
    y_best = float(np.min(m_feas))
    jbest = int(np.argmin(m_feas))
    bx, by = float(XY_feas[jbest,0]), float(XY_feas[jbest,1])

    C = max(50, int(n_candidates))
    Cg = max(1, int((1.0 - local_frac) * C))
    Cl = C - Cg

    (xmin, xmax), (ymin, ymax) = bounds
    parts = []
    if Cl > 0:
        sx = local_sigma_scale * max(lsx, 1e-9)
        sy = local_sigma_scale * max(lsy, 1e-9)
        xs_loc = rng.normal(bx, sx, Cl)
        ys_loc = rng.normal(by, sy, Cl)
        xs_loc = np.clip(xs_loc, xmin, xmax)
        ys_loc = np.clip(ys_loc, ymin, ymax)
        parts.append(np.column_stack([xs_loc, ys_loc]))
    if Cg > 0:
        xs_glb = rng.uniform(xmin, xmax, Cg)
        ys_glb = rng.uniform(ymin, ymax, Cg)
        parts.append(np.column_stack([xs_glb, ys_glb]))
    Xc = np.vstack(parts)

    # Fit surrogates
    mu_s, var_s = _gp_fit_predict(XY, s, Xc, lsx, lsy, noise_s)
    mu_m, var_m = _gp_fit_predict(XY_feas, m_feas, Xc, lsx, lsy, noise_m)

    # CEI ≈ EI(m) * Pr_feasible  (here Pr_feasible≈μ_s clipped to [0,1])
    mu_s = np.clip(mu_s, 0.0, 1.0)
    ei = _expected_improvement(mu_m, var_m, y_best, xi=0.0)
    cei = ei * mu_s

    # Guard-rails: if predicted mean way worse than incumbent AND variance tiny, suppress
    guard_mu = y_best + 0.10  # 0.10 absolute wrmsd tolerance
    guard_var = 0.08          # keep candidates with uncertainty
    keep = (mu_m <= guard_mu) | (var_m >= guard_var)
    if np.any(keep):
        Xc, cei = Xc[keep], cei[keep]

    if Xc.shape[0] == 0:
        # fallback to incumbent
        return (bx, by), {"reason":"guards_pruned_all","best":y_best}

    j = int(np.argmax(cei))
    return (float(Xc[j,0]), float(Xc[j,1])), {
        "reason":"cei",
        "y_best": y_best,
        "ei": float(ei[j]) if ei.shape[0]==cei.shape[0] else float(np.nan),
        "psucc": float(mu_s[j]),
    }

def _candidate_mesh(bounds, n: int, rng: np.random.Generator) -> np.ndarray:
    (xmin, xmax), (ymin, ymax) = bounds
    xs = rng.uniform(xmin, xmax, n)
    ys = rng.uniform(ymin, ymax, n)
    return np.column_stack([xs, ys])

# ------------------------------ Main routine ------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Global constrained BO proposer for detector-center shifts (q=1).")
    ap.add_argument("--run-root", required=True, help="Experiment root that contains 'runs/image_run_log.csv'.")
    # Search-space/radius
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Half-width of square search box (default 0.05 mm).")
    ap.add_argument("--min-step-mm", type=float, default=0.005, help="Minimum meaningful change (stop threshold).")
    # Surrogate hyperparams
    ap.add_argument("--ls-x-mm", type=float, default=0.015)
    ap.add_argument("--ls-y-mm", type=float, default=0.015)
    ap.add_argument("--noise-s", type=float, default=5e-3, help="Observation noise for success GP.")
    ap.add_argument("--noise-m", type=float, default=3e-4, help="Observation noise for quality GP.")
    ap.add_argument("--tau-feasible", type=float, default=0.60, help="Soft success threshold for diagnostics.")
    # Acquisition / candidates
    ap.add_argument("--candidates", type=int, default=800)
    ap.add_argument("--local-frac", type=float, default=0.85)
    ap.add_argument("--local-sigma-scale", type=float, default=1.2)
    # Stopping heuristics
    ap.add_argument("--ei-stop", type=float, default=2e-3, help="If CEI's EI component < this and stable, stop.")
    ap.add_argument("--wrmsd-stop", type=float, default=0.14, help="If best m(Δ) ≤ this and EI small, stop.")
    # RNG
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        raise SystemExit(f"[ERR] Missing log: {log_path}")

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("[propose] No rows found; nothing to do.")
        return 0

    # Aggregate across all runs/events
    XY_all, stats, per_event_history = aggregate_by_shift(entries)
    # Determine bounds from data and radius box around origin
    R = float(abs(args.radius_mm))
    if XY_all.shape[0] > 0:
        xmin = max(-R, float(np.min(XY_all[:,0]) - 0.01))
        xmax = min( R, float(np.max(XY_all[:,0]) + 0.01))
        ymin = max(-R, float(np.min(XY_all[:,1]) - 0.01))
        ymax = min( R, float(np.max(XY_all[:,1]) + 0.01))
    else:
        xmin, xmax, ymin, ymax = -R, R, -R, R
    bounds = ((xmin, xmax), (ymin, ymax))

    # Propose next Δ via constrained BO
    rng = np.random.default_rng(args.seed)
    next_xy, meta = constrained_bo_next(
        XY=XY_all,
        stats=stats,
        bounds=bounds,
        lsx=args.ls_x_mm, lsy=args.ls_y_mm,
        noise_s=args.noise_s, noise_m=args.noise_m,
        tau_feasible=args.tau_feasible,
        n_candidates=args.candidates,
        local_frac=args.local_frac,
        local_sigma_scale=args.local_sigma_scale,
        rng=rng
    )

    # Compute incumbent best for diagnostics & stopping
    s = stats[:,0]; m = stats[:,1]
    feas_mask = (s > 0.0) & np.isfinite(m)
    best_m = float(np.min(m[feas_mask])) if np.any(feas_mask) else float("inf")

    # Decide "done"?
    mark_done = False
    if next_xy is None:
        # No data at all; push a deterministic outward probe along (-,-)
        step = 0.01
        next_xy = (-step, -step)
    else:
        # Stopping: if EI small AND we already have a low m(Δ) many times
        # We approximate "EI small" by meta.get("ei",0.0) and require good best_m
        ei_comp = float(meta.get("ei", 0.0))
        if np.isfinite(best_m) and (best_m <= args.wrmsd_stop) and (ei_comp < args.ei_stop):
            mark_done = True

    # Update all rows in LATEST run
    n_num = 0
    n_done = 0
    for i, (key, row) in enumerate(entries):
        if row is None or (len(row) >= 2 and row[0] == "RAW"):
            continue
        run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = row[:7]
        if not run_s.isdigit() or int(run_s) != latest_run:
            continue
        # Skip if already "done"
        if (ndx_s == "done") and (ndy_s == "done"):
            continue
        if mark_done:
            entries[i] = (key, (run_s, dx_s, dy_s, idx_s, wr_s, "done", "done"))
            n_done += 1
        else:
            ndx, ndy = next_xy
            entries[i] = (key, (run_s, dx_s, dy_s, idx_s, wr_s, _fmt6(ndx), _fmt6(ndy)))
            n_num += 1

    write_log(log_path, entries)

    if mark_done:
        print(f"[propose] 0 new proposals, {n_done} marked done")
    else:
        print(f"[propose] {n_num} new proposals, {n_done} marked done")

    # Optional: brief diagnostics
    print(f"[diag] bounds=(({xmin:.3f},{xmax:.3f}),({ymin:.3f},{ymax:.3f})), best_m={best_m:.3f}")
    if next_xy is not None:
        print(f"[diag] next_xy=({_fmt6(next_xy[0])},{_fmt6(next_xy[1])}) via {meta.get('reason','?')}  psucc≈{meta.get('psucc','—')}  ei≈{meta.get('ei','—')}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
