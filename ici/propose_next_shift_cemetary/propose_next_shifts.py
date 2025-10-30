#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py — Per-event constrained BO (q=1) for detector center shift
Self-adapting (no extra knobs): per-frame lengthscales + trust region inferred from data.

Summary
-------
• Reads grouped CSV `runs/image_run_log.csv` (sections: "#/path event <int>").
• For the LATEST run: propose a distinct (next_dx_mm,next_dy_mm) per event (frame),
  using ONLY that event’s history (falls back gracefully when cold).
• CEI acquisition with:
    - Per-event, data-driven GP lengthscales (no tuning).
    - Per-event trust region (focus box) around the incumbent best, computed from distances
      to the incumbent; automatically narrows as more good results accumulate.
    - Quadratic-fit "snap" candidate toward the local bowl minimum when reliable.
    - Duplicate avoidance and step-capping to keep moves smooth.
• Writes proposals only into the latest run’s rows for each event; marks "done"
  per event when EI is small and best wRMSD is already good.

CLI is unchanged; defaults are sensible. Existing flags like --ls-x-mm are used for cold-start only.
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

def aggregate_by_shift_global(entries):
    """Across ALL events: group by (dx,dy) rounded to 6 decimals.
    Returns XY_all (n,2) and stats_all (n,3: s_hat, m_median, n_total)."""
    agg: Dict[Tuple[str,str], Tuple[int,int,List[float]]] = {}
    for key, row in entries:
        if key is not None:
            continue
        if row is None or len(row) == 0 or row[0] == "RAW":
            continue
        run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = row[:7]
        try:
            _ = int(run_s)
        except Exception:
            continue
        dx = float(dx_s) if dx_s else 0.0
        dy = float(dy_s) if dy_s else 0.0
        indexed = int(idx_s or "0")
        wr = _float_or_blank(wr_s)
        k = (_fmt6(dx), _fmt6(dy))
        n_tot, n_succ, wr_list = agg.get(k, (0,0,[]))
        n_tot += 1
        if indexed and (wr is not None) and math.isfinite(wr):
            n_succ += 1
            wr_list = wr_list + [float(wr)]
        agg[k] = (n_tot, n_succ, wr_list)

    if not agg:
        return np.empty((0,2)), np.empty((0,3))

    keys = []
    s_hat = []
    m_med = []
    n_totals = []
    for (dxs, dys), (n_tot, n_succ, wr_list) in agg.items():
        dx, dy = float(dxs), float(dys)
        a = 0.5; b = 0.5  # Jeffreys
        s = (n_succ + a) / (n_tot + a + b)
        keys.append((dx, dy))
        s_hat.append(s)
        m_med.append(float(statistics.median(wr_list)) if wr_list else np.nan)
        n_totals.append(float(n_tot))
    XY_all = np.array(keys, dtype=float)
    stats_all = np.column_stack([np.array(s_hat, float),
                                 np.array(m_med, float),
                                 np.array(n_totals, float)])
    return XY_all, stats_all

def aggregate_by_shift_for_event(entries, key_filter: Tuple[str,int]):
    """For a specific event key: group by (dx,dy) rounded to 6 decimals."""
    agg: Dict[Tuple[str,str], Tuple[int,int,List[float]]] = {}
    current_key = None
    for key, row in entries:
        if key is not None:
            current_key = key
            continue
        if row is None or len(row) == 0 or row[0] == "RAW":
            continue
        if current_key != key_filter:
            continue
        run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = row[:7]
        try:
            _ = int(run_s)
        except Exception:
            continue
        dx = float(dx_s) if dx_s else 0.0
        dy = float(dy_s) if dy_s else 0.0
        indexed = int(idx_s or "0")
        wr = _float_or_blank(wr_s)
        k = (_fmt6(dx), _fmt6(dy))
        n_tot, n_succ, wr_list = agg.get(k, (0,0,[]))
        n_tot += 1
        if indexed and (wr is not None) and math.isfinite(wr):
            n_succ += 1
            wr_list = wr_list + [float(wr)]
        agg[k] = (n_tot, n_succ, wr_list)

    if not agg:
        return np.empty((0,2)), np.empty((0,3))

    keys, s_hat, m_med, n_totals = [], [], [], []
    for (dxs, dys), (n_tot, n_succ, wr_list) in agg.items():
        dx, dy = float(dxs), float(dys)
        a = 0.5; b = 0.5
        s = (n_succ + a) / (n_tot + a + b)
        keys.append((dx, dy))
        s_hat.append(s)
        m_med.append(float(statistics.median(wr_list)) if wr_list else np.nan)
        n_totals.append(float(n_tot))
    XY = np.array(keys, dtype=float)
    stats = np.column_stack([np.array(s_hat, float),
                             np.array(m_med, float),
                             np.array(n_totals, float)])
    return XY, stats

# --------------------------- Adaptive helpers ----------------------------

def _auto_event_scales(XY_feas: np.ndarray, m_feas: np.ndarray, bounds) -> tuple:
    """
    Compute per-event GP lengthscales and a trust-region half-width (focus box)
    directly from data — no extra knobs.
    """
    (xmin, xmax), (ymin, ymax) = bounds
    R = max(xmax, -xmin, ymax, -ymin)

    # Top-k best points (30% or at least 6) to measure local spread
    k = max(6, int(0.30 * XY_feas.shape[0]))
    idx = np.argpartition(m_feas, min(k-1, len(m_feas)-1))[:k]
    Xbest = XY_feas[idx]

    # Robust spreads → event-specific GP lengthscales (clamped)
    sx = float(np.std(Xbest[:, 0], ddof=1)) if Xbest.shape[0] > 1 else 0.10 * R
    sy = float(np.std(Xbest[:, 1], ddof=1)) if Xbest.shape[0] > 1 else 0.10 * R
    lsx = float(np.clip(1.5 * sx, 0.005 * R, 0.30 * R))
    lsy = float(np.clip(1.5 * sy, 0.005 * R, 0.30 * R))

    # Incumbent and distances
    jbest = int(np.argmin(m_feas))
    bx, by = float(XY_feas[jbest, 0]), float(XY_feas[jbest, 1])
    d = np.hypot(XY_feas[:,0] - bx, XY_feas[:,1] - by)

    # Trust-region half-width = 60th percentile of distances (clamped)
    q60 = float(np.quantile(d, 0.60)) if d.size else 0.10 * R
    half = float(np.clip(q60, 2.0 * min(lsx, lsy), 0.80 * R))

    # Focused inner bounds
    fxmin = max(xmin, bx - half); fxmax = min(xmax, bx + half)
    fymin = max(ymin, by - half); fymax = min(ymax, by + half)
    focused_bounds = ((fxmin, fxmax), (fymin, fymax))
    return (lsx, lsy, (bx, by), half, focused_bounds)

def _quad_minimizer(X: np.ndarray, y: np.ndarray):
    """
    Fit y ≈ a x^2 + b y^2 + c x y + d x + e y + f and return minimizer if Hessian is PD.
    """
    if X.shape[0] < 6:
        return None
    x, z = X[:,0], X[:,1]
    M = np.column_stack([x*x, z*z, x*z, x, z, np.ones_like(x)])
    try:
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c, d, e, f = [float(v) for v in coef]
    H = np.array([[2*a, c], [c, 2*b]], float)
    g = np.array([d, e], float)
    try:
        eig = np.linalg.eigvalsh(H)
        if np.any(eig <= 0):
            return None
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

# --------------------------- Proposal computation ---------------------------

def _candidate_mesh(bounds, n: int, rng: np.random.Generator) -> np.ndarray:
    (xmin, xmax), (ymin, ymax) = bounds
    xs = rng.uniform(xmin, xmax, n)
    ys = rng.uniform(ymin, ymax, n)
    return np.column_stack([xs, ys])

def constrained_bo_next(XY: np.ndarray,
                        stats: np.ndarray,
                        bounds: Tuple[Tuple[float,float], Tuple[float,float]],
                        lsx: float, lsy: float,            # used only when cold
                        noise_s: float, noise_m: float,
                        tau_feasible: float,
                        n_candidates: int,
                        local_frac: float,                 # used only when cold
                        local_sigma_scale: float,          # used only when cold
                        rng: np.random.Generator) -> Tuple[Optional[Tuple[float,float]], Dict[str,float]]:
    """Return next Δ via CEI. stats columns: [s_hat, m_median, n_total]."""
    if XY.shape[0] == 0:
        return None, {"reason":"no_data"}

    s = stats[:,0]
    m = stats[:,1]
    feas_mask = (s > 0.0) & np.isfinite(m)
    XY_feas = XY[feas_mask]
    m_feas = m[feas_mask]

    (xmin, xmax), (ymin, ymax) = bounds
    R = max(xmax, -xmin, ymax, -ymin)

    # -------------- COLD START: no feasible data yet --------------
    if XY_feas.shape[0] == 0:
        Xc = _candidate_mesh(bounds, n_candidates, rng)
        mu_s, _ = _gp_fit_predict(XY, s, Xc, lsx, lsy, noise_s)
        j = int(np.argmax(mu_s))
        return (float(Xc[j,0]), float(Xc[j,1])), {"reason":"no_feasible_max_psucc","psucc":float(mu_s[j])}

    # -------------- WARM START: self-adapting per event --------------
    # Auto lengthscales + focus window from data
    lsx_ev, lsy_ev, (bx, by), half, fbounds = _auto_event_scales(XY_feas, m_feas, bounds)
    fxmin, fxmax = fbounds[0]; fymin, fymax = fbounds[1]

    # Adaptive mixing: more local as feasible count grows (→ ~0.98)
    n_feas = XY_feas.shape[0]
    base_local = 0.80 + 0.18 * (1.0 - np.exp(-n_feas / 25.0))
    Cl = max(50, int(n_candidates * base_local))
    Cg = max(1, n_candidates - Cl)

    parts = []

    # (a) Quadratic-fit minimizer (deterministic) if inside focus
    qmin = _quad_minimizer(XY_feas, m_feas)
    if qmin is not None:
        qx, qy = qmin
        qx = float(np.clip(qx, fxmin, fxmax))
        qy = float(np.clip(qy, fymin, fymax))
        parts.append(np.array([[qx, qy]], dtype=float))

    # (b) Local cloud around incumbent, scale from trust region (no CLI)
    sx = max(0.4 * half, 2.0 * lsx_ev)
    sy = max(0.4 * half, 2.0 * lsy_ev)
    xs_loc = rng.normal(bx, sx, Cl)
    ys_loc = rng.normal(by, sy, Cl)
    xs_loc = np.clip(xs_loc, fxmin, fxmax)
    ys_loc = np.clip(ys_loc, fymin, fymax)
    parts.append(np.column_stack([xs_loc, ys_loc]))

    # (c) Small global sprinkle for escape
    xs_glb = rng.uniform(xmin, xmax, Cg)
    ys_glb = rng.uniform(ymin, ymax, Cg)
    parts.append(np.column_stack([xs_glb, ys_glb]))

    Xc = np.vstack(parts)

    # Drop near-duplicate already-tried shifts
    eps = max(0.01 * half, 0.001 * R)  # mm
    keep = _filter_seen(Xc, XY, eps)
    Xc = Xc[keep] if np.any(keep) else Xc

    # Fit surrogates with auto lengthscales
    mu_s, var_s = _gp_fit_predict(XY, s, Xc, lsx_ev, lsy_ev, noise_s)
    mu_m, var_m = _gp_fit_predict(XY_feas, m_feas, Xc, lsx_ev, lsy_ev, noise_m)

    # CEI ≈ EI(m) * Pr_feasible
    y_best = float(np.min(m_feas))
    mu_s = np.clip(mu_s, 0.0, 1.0)
    ei = _expected_improvement(mu_m, var_m, y_best, xi=0.0)
    cei = ei * mu_s

    # Guard-rails: keep plausible or uncertain; keep arrays in sync
    guard_mu = y_best + 0.10
    guard_var = 0.08
    keep = (mu_m <= guard_mu) | (var_m >= guard_var)
    if np.any(keep):
        Xc    = Xc[keep]
        ei    = ei[keep]
        mu_s  = mu_s[keep]
        mu_m  = mu_m[keep]
        var_m = var_m[keep]
        cei   = cei[keep]

    if Xc.shape[0] == 0:
        return (bx, by), {"reason":"guards_pruned_all","best":y_best}

    j = int(np.argmax(cei))
    nx, ny = float(Xc[j,0]), float(Xc[j,1])

    # Step cap: do not move more than 0.7 * half from incumbent in one go
    step = math.hypot(nx - bx, ny - by)
    max_step = 0.70 * half
    if step > max_step and step > 0:
        scale = max_step / step
        nx = bx + scale * (nx - bx)
        ny = by + scale * (ny - by)

    return (nx, ny), {
        "reason": "cei_adaptive",
        "y_best": y_best,
        "ei": float(ei[j]),
        "psucc": float(mu_s[j]),
        "half_focus": float(half),
        "lsx_ev": float(lsx_ev), "lsy_ev": float(lsy_ev),
    }

# ------------------------------ Main routine ------------------------------

def _per_event_seed(base_seed: int, key: Tuple[str,int]) -> int:
    """Stable per-event RNG seed so different events don't collide."""
    h = hashlib.sha1(f"{key[0]}|{key[1]}".encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "little")
    return base_seed ^ (v & 0x7FFFFFFF)

def main(argv=None):
    ap = argparse.ArgumentParser(description="Constrained BO proposer for detector-center shifts — per event (self-adapting).")
    ap.add_argument("--run-root", required=True, help="Experiment root that contains 'runs/image_run_log.csv'.")
    # Search-space/radius (global clamp)
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Half-width of square search box (default 0.05 mm).")
    ap.add_argument("--min-step-mm", type=float, default=1e-4, help="Minimum meaningful change (stop threshold).")
    # Surrogate hyperparams (used mainly for cold start)
    ap.add_argument("--ls-x-mm", type=float, default=0.01)
    ap.add_argument("--ls-y-mm", type=float, default=0.01)
    ap.add_argument("--noise-s", type=float, default=5e-3, help="Observation noise for success GP.")
    ap.add_argument("--noise-m", type=float, default=3e-4, help="Observation noise for quality GP.")
    ap.add_argument("--tau-feasible", type=float, default=1.0, help="Soft success threshold for diagnostics.")
    # Acquisition / candidates (counts only; mixing becomes adaptive after warm start)
    ap.add_argument("--candidates", type=int, default=1200)
    ap.add_argument("--local-frac", type=float, default=0.98)
    ap.add_argument("--local-sigma-scale", type=float, default=0.5)
    # Stopping heuristics (per event)
    ap.add_argument("--ei-stop", type=float, default=5e-2, help="Stop event if EI component < this and best WRMSD is good.")
    ap.add_argument("--wrmsd-stop", type=float, default=0.1, help="Stop event if best m(Δ) ≤ this and EI small.")
    # RNG
    ap.add_argument("--seed", type=int, default=23)
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

    # Global aggregate (only for cold-start bounds)
    XY_all_global, stats_all_global = aggregate_by_shift_global(entries)

    # Determine which events appear in the latest run
    event_keys_latest: List[Tuple[str,int]] = []
    seen = set()
    current_key = None
    for key, row in entries:
        if key is not None:
            current_key = key
            continue
        if row is None or (len(row) >= 2 and row[0] == "RAW"):
            continue
        run_s = row[0]
        if run_s.isdigit() and int(run_s) == latest_run:
            if current_key is not None and current_key not in seen:
                seen.add(current_key)
                event_keys_latest.append(current_key)

    # Precompute global bounds within radius (for cold events)
    R = float(abs(args.radius_mm))
    if XY_all_global.shape[0] > 0:
        gxmin = max(-R, float(np.min(XY_all_global[:,0]) - 0.01))
        gxmax = min( R, float(np.max(XY_all_global[:,0]) + 0.01))
        gymin = max(-R, float(np.min(XY_all_global[:,1]) - 0.01))
        gymax = min( R, float(np.max(XY_all_global[:,1]) + 0.01))
    else:
        gxmin, gxmax, gymin, gymax = -R, R, -R, R
    global_bounds = ((gxmin, gxmax), (gymin, gymax))

    # Propose per event
    per_event_next: Dict[Tuple[str,int], Tuple[bool, Tuple[float,float], Dict[str,float]]] = {}
    for ev_key in event_keys_latest:
        # Aggregate this event's history only
        XY_ev, stats_ev = aggregate_by_shift_for_event(entries, ev_key)

        # Event-specific bounds (fallback to global bounds if empty)
        if XY_ev.shape[0] > 0:
            xmin = max(-R, float(np.min(XY_ev[:,0]) - 0.01))
            xmax = min( R, float(np.max(XY_ev[:,0]) + 0.01))
            ymin = max(-R, float(np.min(XY_ev[:,1]) - 0.01))
            ymax = min( R, float(np.max(XY_ev[:,1]) + 0.01))
            bounds_ev = ((xmin, xmax), (ymin, ymax))
        else:
            bounds_ev = global_bounds

        # If event has no history, emit a deterministic small probe
        if XY_ev.shape[0] == 0:
            next_xy = (-0.010, -0.010)
            per_event_next[ev_key] = (False, next_xy, {"reason":"deterministic_bootstrap"})
            continue

        # Per-event RNG
        seed_ev = _per_event_seed(args.seed, ev_key)
        rng_ev = np.random.default_rng(seed_ev)

        # Propose via CEI for THIS event (adaptive)
        next_xy, meta = constrained_bo_next(
            XY=XY_ev,
            stats=stats_ev,
            bounds=bounds_ev,
            lsx=args.ls_x_mm, lsy=args.ls_y_mm,
            noise_s=args.noise_s, noise_m=args.noise_m,
            tau_feasible=args.tau_feasible,
            n_candidates=args.candidates,
            local_frac=args.local_frac,
            local_sigma_scale=args.local_sigma_scale,
            rng=rng_ev
        )

        # Stopping (per event) based on event's best_m and EI
        s = stats_ev[:,0]; m = stats_ev[:,1]
        feas_mask = (s > 0.0) & np.isfinite(m)
        best_m = float(np.min(m[feas_mask])) if np.any(feas_mask) else float("inf")

        mark_done = False
        if next_xy is None:
            next_xy = (-0.010, -0.010)
        else:
            ei_comp = float(meta.get("ei", 0.0))
            if np.isfinite(best_m) and (best_m <= args.wrmsd_stop) and (ei_comp < args.ei_stop):
                mark_done = True

        per_event_next[ev_key] = (mark_done, next_xy, {"best_m": best_m, **meta})

    # Write back only rows of latest run, per event
    n_num = 0
    n_done = 0
    current_key = None
    for i, (key, row) in enumerate(entries):
        if key is not None:
            current_key = key
            continue
        if row is None or (len(row) >= 2 and row[0] == "RAW"):
            continue
        run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = row[:7]
        if not run_s.isdigit() or int(run_s) != latest_run:
            continue
        if current_key not in per_event_next:
            continue  # safety
        # Skip if already "done"
        if (ndx_s == "done") and (ndy_s == "done"):
            continue

        mark_done, (ndx, ndy), _ = per_event_next[current_key]
        if mark_done:
            entries[i] = (key, (run_s, dx_s, dy_s, idx_s, wr_s, "done", "done"))
            n_done += 1
        else:
            entries[i] = (key, (run_s, dx_s, dy_s, idx_s, wr_s, _fmt6(ndx), _fmt6(ndy)))
            n_num += 1

    write_log(log_path, entries)

    # Diagnostics
    n_events = len(event_keys_latest)
    n_events_done = sum(1 for k,(d,_,_) in per_event_next.items() if d)
    print(f"[propose] events in latest run: {n_events}  (done now: {n_events_done})")
    print(f"[propose] {n_num} new proposals, {n_done} marked done")

    for key in event_keys_latest:
        mark_done, (dx,dy), meta = per_event_next[key]
        base = os.path.basename(key[0])
        tag = f"{base}|event{key[1]}"
        print(f"[diag:{tag}] next_xy=({_fmt6(dx)},{_fmt6(dy)}) via {meta.get('reason','?')} "
              f"best_m≈{meta.get('best_m','—')} ei≈{meta.get('ei','—')} "
              f"psucc≈{meta.get('psucc','—')} half≈{meta.get('half_focus','—')} "
              f"lsx≈{meta.get('lsx_ev','—')} lsy≈{meta.get('lsy_ev','—')} "
              f"{'DONE' if mark_done else ''}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
