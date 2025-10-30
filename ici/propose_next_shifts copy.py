#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts_purebo.py

Pure Bayesian Optimization (no expanding ring). Uses both indexed and
unindexed outcomes from prior runs for each (image,event):

- Indexed rows with finite wRMSD -> true observations (y = wRMSD).
- Unindexed rows -> weak negatives via pseudo-observation
  y = (global_median_good or fallback) + penalty_offset.

Behavior highlights (few knobs, robust defaults):
- Small isotropic GP (RBF) with nugget.
- Candidate sampler: 90% local around incumbent best (sigma = lengthscale), 10% global within a trust box.
- Trust box (+/- bo-rho) around the incumbent (or around initial center if no incumbent yet).
- Per-step cap (--bo-max-step-mm) from the incumbent.
- Guard rails: drop EI candidates with predicted mean >> current best unless variance is large.
- Data-driven 'done': if best sits in robust "good band" (median + 2*MAD over DONE events' bests;
  fallback to all current bests if none DONE) and recent improvement < 0.01, stop.
- Minimal seeding: optional 3x3 grid around the initial center if we have < 3 observations.

CLI (minimal knobs):
  --bo-lengthscale   Isotropic lengthscale [mm]
  --bo-noise         GP nugget in wRMSD^2
  --bo-candidates    EI candidates per propose
  --bo-ei-eps        EI threshold to stop refining
  --bo-rho           Trust-region half-size [mm]
  --bo-max-step-mm   Hard cap on step from incumbent [mm] (optional)
  --penalty-offset   Additive penalty for unindexed pseudo-y (default 0.30)
  --seed-grid        If set, seed a 3x3 grid around start center when obs < 3
  --seed-delta       Half-spacing for seeding grid [mm] (default 0.006)

CSV I/O: compatible with image_run_log.csv format used in your pipeline.
"""
from __future__ import annotations
import argparse, math, os, sys, statistics
from typing import Dict, List, Tuple, Optional
import numpy as np
from math import erf, sqrt

# ----------------- Defaults -----------------
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 5e-4  # stop if tiny move from best

# ----------------- GP + EI -----------------
class BO2DConfig:
    def __init__(self, lengthscale=0.02, noise=3e-4, candidates=400, ei_eps=2e-3,
                 rng_seed=1337, rho=0.04, max_step_mm: Optional[float]=0.010):
        self.lengthscale = float(lengthscale)
        self.noise = float(noise)
        self.candidates = int(candidates)
        self.ei_eps = float(ei_eps)
        self.rho = float(rho)
        self.max_step_mm = float(max_step_mm) if max_step_mm is not None else None
        self.rng = np.random.default_rng(int(rng_seed))
        # fixed heuristics
        self.local_frac = 0.90
        self.mu_guard = 0.08
        self.var_guard = 0.05

class SPSA2DConfig:
    def __init__(self,
                 a=0.006, A=10.0, alpha=0.602,     # step-size schedule a_k = a/(k+A)^alpha
                 c=0.010, gamma=0.101,             # perturb schedule  c_k = c/(k+1)^gamma
                 rho=0.05,                         # trust-region half-size [mm]
                 max_step_mm=0.012,                # hard cap on |step| from incumbent [mm]
                 retries=1,                        # retries per arm when forcing baseline refresh
                 rng_seed=1337):
        self.a=float(a); self.A=float(A); self.alpha=float(alpha)
        self.c=float(c); self.gamma=float(gamma)
        self.rho=float(rho); self.max_step_mm=float(max_step_mm) if max_step_mm is not None else None
        self.retries=int(retries)
        self.rng = np.random.default_rng(int(rng_seed))

def _rbf_iso(X1: np.ndarray, X2: np.ndarray, ls: float) -> np.ndarray:
    d2 = np.sum(((X1[:, None, :] - X2[None, :, :]) / max(ls,1e-12))**2, axis=2)
    return np.exp(-0.5 * d2)

def _gp_fit_predict(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, ls: float, noise: float):
    import numpy as _np
    ymean = float(_np.mean(y))
    yc = y - ymean
    K = _rbf_iso(X, X, ls) + (noise * _np.eye(len(X)))
    jitter = 1e-12
    try:
        L = _np.linalg.cholesky(K)
    except _np.linalg.LinAlgError:
        L = _np.linalg.cholesky(K + jitter * _np.eye(len(X)))
    alpha = _np.linalg.solve(L.T, _np.linalg.solve(L, yc))
    Ks = _rbf_iso(X, Xstar, ls)
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

def bo2d_propose(tried_xy: np.ndarray, tried_y: np.ndarray, box: Tuple[Tuple[float,float],Tuple[float,float]],
                 cfg: BO2DConfig, tried_tol: float = 1e-6):
    (xmin, xmax), (ymin, ymax) = box
    X = tried_xy.astype(float)
    y = tried_y.astype(float)
    y_best = float(np.min(y))
    jbest = int(np.argmin(y))
    bx, by = float(X[jbest,0]), float(X[jbest,1])

    # Candidates: 90% local Gaussian, 10% global in box
    C = max(10, int(cfg.candidates))
    Cg = max(1, int((1.0 - cfg.local_frac) * C))
    Cl = C - Cg
    parts = []
    if Cl > 0:
        xs = cfg.rng.normal(bx, cfg.lengthscale, Cl)
        ys = cfg.rng.normal(by, cfg.lengthscale, Cl)
        parts.append(np.column_stack([np.clip(xs, xmin, xmax), np.clip(ys, ymin, ymax)]))
    if Cg > 0:
        xs = cfg.rng.uniform(xmin, xmax, Cg)
        ys = cfg.rng.uniform(ymin, ymax, Cg)
        parts.append(np.column_stack([xs, ys]))
    Xc = np.vstack(parts) if parts else np.empty((0,2), float)

    if len(X) > 0 and Xc.shape[0] > 0:
        d2 = np.sum((Xc[:, None, :] - X[None, :, :]) ** 2, axis=2)
        mind = np.sqrt(np.min(d2, axis=1))
        Xc = Xc[mind > tried_tol]
        if Xc.shape[0] == 0:
            return None, 0.0

    mu, var = _gp_fit_predict(X, y, Xc, cfg.lengthscale, cfg.noise)
    ei = expected_improvement(mu, var, y_best, xi=0.0)

    # Guard rails: drop very-bad-mean unless uncertainty large
    keep = (mu <= (y_best + cfg.mu_guard)) | (var >= cfg.var_guard)
    if np.any(keep):
        Xc, mu, var, ei = Xc[keep], mu[keep], var[keep], ei[keep]
    if ei.size == 0:
        return None, 0.0

    j = int(np.argmax(ei))
    next_xy = (float(Xc[j,0]), float(Xc[j,1]))
    return next_xy, float(ei[j])
def _incumbent_from_history(trials_sorted):
    # Return (best_xy, best_y) over indexed rows with finite wr
    best_xy, best_y = None, float("inf")
    for (_, dx, dy, ind, wr) in trials_sorted:
        if ind and wr is not None and math.isfinite(wr):
            if wr < best_y:
                best_y = float(wr); best_xy = (float(dx), float(dy))
    return best_xy, best_y

def _success_rate_recent(trials_sorted, window=10):
    zs = []
    for (_,_,_,ind,wr) in trials_sorted[-window:]:
        zs.append(1 if (ind and wr is not None and math.isfinite(wr)) else 0)
    return (sum(zs)/len(zs)) if zs else 0.0

def spsa_propose_for_event(trials_sorted,
                           cfg: SPSA2DConfig,
                           st: Optional[dict],
                           tried_points_set,
                           k_global: int):
    """
    Returns (proposal_xy, updated_state_dict, mark_done_bool).
    We use one-sided SPSA around incumbent with baseline replicate caching.
    """
    # 1) Find incumbent and (optional) baseline y0 at that point
    best_xy, best_y = _incumbent_from_history(trials_sorted)
    if best_xy is None:
        # No successes yet: propose small step toward local mean of all tried points to raise success prob
        mx = np.mean([dx for (_,dx,_,_,_) in trials_sorted]) if trials_sorted else 0.0
        my = np.mean([dy for (_,_,dy,_,_) in trials_sorted]) if trials_sorted else 0.0
        return (mx, my), st or {}, False

    bx, by = best_xy
    y0 = best_y

    # 2) Pull/initialize state
    st = dict(st or {})
    if "k" not in st:
        st["k"] = 0
        st["seed"] = k_global
        st["x0"], st["y0"] = y0, y0  # store last baseline value
        st["bx"], st["by"] = bx, by

    # 3) Schedules
    k = int(st["k"])
    a_k = cfg.a / ((k + cfg.A) ** cfg.alpha)
    c_k = cfg.c / ((k + 1) ** cfg.gamma)

    # 4) Trust box around incumbent
    xmin, xmax = bx - cfg.rho, bx + cfg.rho
    ymin, ymax = by - cfg.rho, by + cfg.rho

    # 5) Generate a fresh Δ and candidate (+ arm). We’ll rely on one-sided SPSA (baseline at incumbent).
    rng = cfg.rng
    Δ = rng.choice(np.array([-1.0, 1.0]), size=(2,), replace=True)
    x_plus = (np.clip(bx + c_k * Δ[0], xmin, xmax),
              np.clip(by + c_k * Δ[1], ymin, ymax))

    # Avoid duplicates
    if (_fmt6(x_plus[0]), _fmt6(x_plus[1])) in tried_points_set:
        # jitter once; if still duplicate, mark done
        for _ in range(5):
            Δ = rng.choice(np.array([-1.0, 1.0]), size=(2,), replace=True)
            x_plus = (np.clip(bx + c_k * Δ[0], xmin, xmax),
                      np.clip(by + c_k * Δ[1], ymin, ymax))
            if (_fmt6(x_plus[0]), _fmt6(x_plus[1])) not in tried_points_set:
                break
        else:
            return ("done","done"), st, True

    # 6) Optional staged policy: if recent success rate is low, shrink region
    p_recent = _success_rate_recent(trials_sorted, window=10)
    if p_recent < 0.2:
        shrink = 0.6
        x_plus = (bx + shrink*(x_plus[0]-bx), by + shrink*(x_plus[1]-by))

    # 7) Cap absolute move from incumbent
    if cfg.max_step_mm is not None:
        dxs, dys = x_plus[0]-bx, x_plus[1]-by
        r = math.hypot(dxs, dys); cap = float(cfg.max_step_mm)
        if r > cap and r > 0.0:
            s = cap / r
            x_plus = (bx + s*dxs, by + s*dys)

    # Update state for next round (we’ll recompute gradient once we see y_plus in the next call)
    st["k"] = k + 1
    st["bx"], st["by"] = bx, by
    # Note: If you later want to *move* the center using one-sided gradient (y_plus - y0)/(c_k*Δ),
    # you can compute that in propose_for_latest by detecting the last proposed x_plus and its outcome.

    return x_plus, st, False

# ----------------- Utils & CSV -----------------
# ---- Lightweight per-event state persisted in RAW #STATE lines ----
def _state_tag(key):  # key = (abs_path, event_int)
    return f"#STATE {key[0]} :: event {key[1]} :: "

def read_states(entries):
    """Return dict[key] = state_dict parsed from #STATE lines."""
    states = {}
    for key, row in entries:
        if key is None and row is not None and len(row)>=2 and row[0]=="RAW":
            ln = row[1]
            if ln.startswith("#STATE "):
                try:
                    header, payload = ln.split("::", 2)[-1], ln.split("::", 2)[-0]  # just to not crash if format changes
                except Exception:
                    payload = ln
                try:
                    # Expected format: "#STATE <path> :: event <ev> :: k=<int> x0=<float> y0=<float> bx=<float> by=<float> seed=<int>"
                    parts = ln.split("::")
                    # parts[0] = "#STATE <path> ", parts[1] = " event <ev> ", parts[2] = " k=... ..."
                    ev = int(parts[1].strip().split()[-1])
                    # We need to recover <path> to form the key; since entries keep sections, we'll assign on next section change in propose()
                except Exception:
                    pass
    return states  # we will re-fill properly inside proposer

def parse_state_line(line):
    # returns dict of parsed fields; resilient to missing keys
    out = {}
    toks = line.split("::")[-1].strip().split()
    for t in toks:
        if "=" in t:
            k,v = t.split("=",1)
            try:
                out[k]=float(v) if "." in v or "e" in v.lower() else int(v)
            except Exception:
                out[k]=v
    return out

def format_state_line(key, st):
    # st: dict with k, x0, y0, bx, by, seed
    return (_state_tag(key) +
            f"k={st.get('k',0)} x0={st.get('x0',0.0):.6f} y0={st.get('y0',float('nan')):.6f} "
            f"bx={st.get('bx',0.0):.6f} by={st.get('by',0.0):.6f} seed={int(st.get('seed',0))}\n")

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

# ----------------- Main per-event proposal -----------------
def propose_for_latest(entries, latest_run: int,
                       seed: int,
                       bo_cfg: BO2DConfig,
                       penalty_offset: float,
                       seed_grid: bool,
                       seed_delta: float,
                       algo: str = "bo",
                       spsa_params: Optional[dict] = None):
    """
    Adjusted proposer with two algorithms:
      - 'bo'   : your existing BO (unchanged behavior)
      - 'spsa' : failure-aware one-sided SPSA that assumes success probability increases near the minimum

    SPSA state is persisted as RAW lines starting with '#STATE ' so each call can continue schedules.
    """

    # ----------------- Local helpers just for SPSA-path -----------------

    def _last_next_for_event(entries, key, latest_run):
        # Returns (next_dx, next_dy) for this event at latest_run if present, else (None, None)
        last = None
        current_key = None
        for (k, row) in entries:
            if k is not None:
                current_key = k
                continue
            if current_key != key or row is None or row[0] == "RAW":
                continue
            if row[0].isdigit() and int(row[0]) == latest_run:
                # row format: run_n, dx, dy, indexed, wrmsd, next_dx, next_dy
                if len(row) >= 7:
                    ndx = row[5].strip(); ndy = row[6].strip()
                    last = (float(ndx), float(ndy)) if (ndx and ndy) else (None, None)
        return last if last is not None else (None, None)

    def _same_point(a, b, tol=1e-6):
        if a is None or b is None or isinstance(a, str) or isinstance(b, str):
            return False
        x1,y1 = a; x2,y2 = b
        return (abs(x1 - x2) <= tol) and (abs(y1 - y2) <= tol)

    def _fmt_state_line(key, st: dict) -> str:
        # st keys used: k, x0, y0, bx, by, seed
        return (f"#STATE {key[0]} :: event {key[1]} :: "
                f"k={int(st.get('k', 0))} "
                f"x0={float(st.get('x0', 0.0)):.6f} "
                f"y0={float(st.get('y0', float('nan'))):.6f} "
                f"bx={float(st.get('bx', 0.0)):.6f} "
                f"by={float(st.get('by', 0.0)):.6f} "
                f"seed={int(st.get('seed', 0))}")

    def _parse_state_line(line: str) -> dict:
        # robustly parse " :: k=... x0=... y0=... bx=... by=... seed=..."
        out = {}
        tail = line.split("::")[-1]
        for tok in tail.strip().split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    out[k] = float(v) if any(c in v for c in ".eE") else int(v)
                except Exception:
                    out[k] = v
        return out

    def _incumbent_from_history(trials_sorted):
        # Return best successful (indexed, finite wr) as (xy, y)
        best_xy, best_y = None, float("inf")
        for (_, dx, dy, ind, wr) in trials_sorted:
            if ind and wr is not None and math.isfinite(wr):
                if wr < best_y:
                    best_y = float(wr); best_xy = (float(dx), float(dy))
        return best_xy, best_y

    def _success_rate_recent(trials_sorted, window=10):
        zs = [(1 if (ind and wr is not None and math.isfinite(wr)) else 0)
              for (_,_,_,ind,wr) in trials_sorted[-window:]]
        return (sum(zs)/len(zs)) if zs else 0.0

    # Default SPSA knobs (can be overridden via spsa_params)
    spsa_defaults = dict(a=0.006, A=10.0, alpha=0.602,
                         c=0.010, gamma=0.101,
                         rho=0.05, max_step_mm=0.012,
                         retries=1, rng_seed=seed)
    sp = {**spsa_defaults, **(spsa_params or {})}
    rng = np.random.default_rng(int(sp["rng_seed"]))

    # ----------------- Build history (unchanged) -----------------
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

    # robust band for 'done'
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
        good_band_upper = band_med + 2.0 * band_mad
    else:
        good_band_upper = float("inf")

    # global median of "good" (indexed) to set pseudo-y for BO path (unchanged idea)
    pseudo_base = statistics.median(pool) if pool else 0.5
    pseudo_y = 2 * float(pseudo_base)

    # ----------------- Load any existing per-event SPSA state from RAW lines -----------------
    states_by_key: Dict[Tuple[str,int], dict] = {}
    current_key = None
    for key, row in entries:
        if key is not None:
            current_key = key; continue
        if row is None or len(row)==0 or row[0]!="RAW": continue
        ln = row[1]
        if current_key and ln.startswith("#STATE "):
            # Validate that line belongs to this current_key by event id if present
            try:
                if f"event {current_key[1]}" in ln:
                    states_by_key[current_key] = _parse_state_line(ln)
            except Exception:
                pass

    proposals: Dict[Tuple[str,int], Tuple[object, object]] = {}
    state_lines_to_append: Dict[Tuple[str,int], str] = {}

    # ----------------- Per-event proposing -----------------
    for key, trials in history.items():
        if key not in latest_rows_by_key:
            continue

        trials_sorted = sorted(trials, key=lambda t: t[0])
        tried_points = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)

        # ---------- SPSA branch ----------
        if algo.lower() == "spsa":
            # Incumbent from successful (indexed) measurements
            best_xy, best_y = _incumbent_from_history(trials_sorted)

            if best_xy is None:
                # No successes yet → aim toward local mean of tried points (raise success probability)
                if trials_sorted:
                    mx = np.mean([dx for (_,dx,_,_,_) in trials_sorted])
                    my = np.mean([dy for (_,_,dy,_,_) in trials_sorted])
                    proposals[key] = (float(mx), float(my))
                else:
                    # Fallback to (0,0) if truly empty
                    proposals[key] = (0.0, 0.0)
                continue

            bx, by = best_xy
            y0 = best_y

            # Recover/update per-event state
            st_prev = states_by_key.get(key, {})
            k = int(st_prev.get("k", 0))
            # Schedules
            a_k = sp["a"] / ((k + sp["A"]) ** sp["alpha"])
            c_k = sp["c"] / ((k + 1) ** sp["gamma"])

            # Trust box around incumbent
            xmin, xmax = bx - sp["rho"], bx + sp["rho"]
            ymin, ymax = by - sp["rho"], by + sp["rho"]

            # Draw a Rademacher perturbation and propose one-sided +arm
            Δx, Δy = rng.choice(np.array([-1.0, 1.0])), rng.choice(np.array([-1.0, 1.0]))
            x_plus = (np.clip(bx + c_k * Δx, xmin, xmax),
                      np.clip(by + c_k * Δy, ymin, ymax))

            # Avoid duplicates (retry a few jitter attempts)
            tries = 0
            while (_fmt6(x_plus[0]), _fmt6(x_plus[1])) in tried_points and tries < 5:
                Δx, Δy = rng.choice(np.array([-1.0, 1.0])), rng.choice(np.array([-1.0, 1.0]))
                x_plus = (np.clip(bx + c_k * Δx, xmin, xmax),
                          np.clip(by + c_k * Δy, ymin, ymax))
                tries += 1
            if (_fmt6(x_plus[0]), _fmt6(x_plus[1])) in tried_points:
                proposals[key] = ("done", "done")
                continue

            # If recent success rate is poor, shrink step toward incumbent
            p_recent = _success_rate_recent(trials_sorted, window=10)
            if p_recent < 0.4:
                shrink = 0.6
                x_plus = (bx + shrink*(x_plus[0]-bx), by + shrink*(x_plus[1]-by))

            # Cap absolute move from incumbent
            if sp["max_step_mm"] is not None:
                dxs, dys = x_plus[0]-bx, x_plus[1]-by
                r = math.hypot(dxs, dys); cap = float(sp["max_step_mm"])
                if r > cap and r > 0.0:
                    s = cap / r
                    x_plus = (bx + s*dxs, by + s*dys)

            proposals[key] = (float(x_plus[0]), float(x_plus[1]))

            # Persist updated state
            st_new = {
                "k": k + 1,
                "x0": y0, "y0": y0,
                "bx": bx, "by": by,
                "seed": sp["rng_seed"]
            }
            state_lines_to_append[key] = _fmt_state_line(key, st_new)
            continue  # next key

        # ---------- BO branch (your current behavior) ----------
        # Build BO dataset (both indexed and unindexed)
        obs_xy = []
        obs_y = []
        for (_, dx, dy, ind, wr) in trials_sorted:
            if ind and (wr is not None) and math.isfinite(wr):
                obs_xy.append((dx, dy))
                obs_y.append(float(wr))
            else:
                obs_xy.append((dx, dy))
                obs_y.append(pseudo_y)

        # If no observations at all (unlikely), seed center from first row
        if len(obs_xy) == 0:
            start_cx, start_cy = (trials_sorted[0][1], trials_sorted[0][2]) if trials_sorted else (0.0, 0.0)
            proposals[key] = (start_cx, start_cy)
            continue

        X = np.array(obs_xy, float)
        y = np.array(obs_y, float)

        # Incumbent best (by y)
        jbest = int(np.argmin(y))
        best_xy = (float(X[jbest,0]), float(X[jbest,1]))
        y_best = float(y[jbest])

        # robust 'done'
        indexed_wr = [wr for (_,_,_,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
        if indexed_wr:
            best_seq = []
            m = float('inf')
            for (_,_,_,ind,wr) in trials_sorted:
                if ind and (wr is not None) and math.isfinite(wr):
                    m = min(m, float(wr)); best_seq.append(m)
            if len(best_seq) >= 3:
                recent = best_seq[-min(5, len(best_seq)):]
                if (recent[0] - recent[-1]) < 0.01 and (y_best <= good_band_upper):
                    proposals[key] = ("done", "done"); continue

        # Optional tiny seeding if too few points (<3)
        if seed_grid and (len(obs_xy) < 3):
            made = False
            start_cx, start_cy = (trials_sorted[0][1], trials_sorted[0][2]) if trials_sorted else (0.0, 0.0)
            for sx in (-seed_delta, 0.0, seed_delta):
                for sy in (-seed_delta, 0.0, seed_delta):
                    cand = (_fmt6(start_cx+sx), _fmt6(start_cy+sy))
                    if cand not in tried_points:
                        proposals[key] = (start_cx+sx, start_cy+sy); made = True
                        break
                if made: break
            if made: continue  # we proposed a seed

        # BO trust box around incumbent (or start if best is still at start)
        bx, by = best_xy
        xmin, xmax = bx - bo_cfg.rho, bx + bo_cfg.rho
        ymin, ymax = by - bo_cfg.rho, by + bo_cfg.rho

        next_xy, ei_max = bo2d_propose(X, y, ((xmin, xmax), (ymin, ymax)), bo_cfg)
        if (next_xy is None) or (ei_max < bo_cfg.ei_eps):
            proposals[key] = ("done", "done"); continue

        if math.hypot(next_xy[0]-bx, next_xy[1]-by) < CONVERGE_TOL_DEFAULT:
            proposals[key] = ("done", "done"); continue

        if (bo_cfg.max_step_mm is not None):
            dxs, dys = next_xy[0] - bx, next_xy[1] - by
            r = math.hypot(dxs, dys)
            cap = float(bo_cfg.max_step_mm)
            if r > cap and r > 0.0:
                s = cap / r
                next_xy = (bx + s*dxs, by + s*dys)

        if (_fmt6(next_xy[0]), _fmt6(next_xy[1])) in tried_points:
            proposals[key] = ("done", "done"); continue

        proposals[key] = (next_xy[0], next_xy[1])

    # ----------------- Apply proposals to latest-run rows (unchanged style) -----------------
    n_new, n_done = 0, 0
    for key, idx_list in latest_rows_by_key.items():
        if key not in proposals:
            continue
        ndx, ndy = proposals[key]
        for row_idx in idx_list:
            row = list(entries[row_idx][1])
            if len(row) < 7: row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):  # "done"
                    row[5] = "done"; row[6] = "done"; n_done += 1
                else:
                    row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy)); n_new += 1
                entries[row_idx] = (None, tuple(row))

    # Append/update SPSA state lines (we just append at EOF; parser preserves RAW lines)
    if state_lines_to_append:
        for key, stline in state_lines_to_append.items():
            entries.append((None, ("RAW", stline)))

    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    return entries

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose next detector shifts (BO or failure-aware SPSA).")
    ap.add_argument("--run-root", default=None, help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--sidecar", help="If provided, write proposals to this CSV instead of rewriting the log.")

    # ---- BO knobs (existing) ----
    ap.add_argument("--bo-lengthscale", type=float, default=0.02, help="Isotropic RBF lengthscale (mm)." )
    ap.add_argument("--bo-noise", type=float, default=3e-4, help="GP nugget noise in wrmsd^2 units." )
    ap.add_argument("--bo-candidates", type=int, default=500, help="Number of EI candidates." )
    ap.add_argument("--bo-ei-eps", type=float, default=2e-3, help="EI threshold to mark done." )
    ap.add_argument("--bo-rho", type=float, default=0.05, help="Trust region half-size (mm) around incumbent." )
    ap.add_argument("--bo-max-step-mm", type=float, default=0.012, help="Hard cap on |step| from incumbent (mm)." )

    # ---- Pseudo-observation control (existing) ----
    ap.add_argument("--penalty-offset", type=float, default=1, help="Additive penalty for unindexed pseudo-wRMSD.")
    ap.add_argument("--seed-grid", action="store_true", help="Seed a 3x3 grid around start when observations < 3.")
    ap.add_argument("--seed-delta", type=float, default=0.006, help="Half-spacing (mm) for the seeding grid.")

    # ---- Algorithm choice ----
    ap.add_argument("--algo", choices=["bo","spsa"], default="spsa",
                    help="Proposal algorithm: 'bo' (Bayesian Optimization) or 'spsa' (failure-aware SPSA).")
    print(f"[propose] algo={args.algo}")

    # ---- SPSA knobs ----
    ap.add_argument("--spsa-a", type=float, default=0.01, help="SPSA step-size scale a.")
    ap.add_argument("--spsa-A", type=float, default=10.0, help="SPSA stability constant A.")
    ap.add_argument("--spsa-alpha", type=float, default=0.602, help="SPSA step-size exponent alpha.")
    ap.add_argument("--spsa-c", type=float, default=0.02, help="SPSA perturbation scale c.")
    ap.add_argument("--spsa-gamma", type=float, default=0.101, help="SPSA perturbation exponent gamma.")
    ap.add_argument("--spsa-rho", type=float, default=0.1, help="Trust-region half-size (mm) around incumbent (SPSA).")
    ap.add_argument("--spsa-max-step-mm", type=float, default=0.012, help="Hard cap on |step| from incumbent (SPSA).")
    ap.add_argument("--spsa-retries", type=int, default=1, help="(Reserved) Retries per arm when forcing baseline (not used in proposer).")

    args = ap.parse_args(argv)

    run_root = os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_ROOT))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr); return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr); return 2

    bo_cfg = BO2DConfig(lengthscale=args.bo_lengthscale,
                        noise=args.bo_noise,
                        candidates=args.bo_candidates,
                        ei_eps=args.bo_ei_eps,
                        rng_seed=args.seed,
                        rho=args.bo_rho,
                        max_step_mm=args.bo_max_step_mm)

    spsa_params = dict(
        a=args.spsa_a, A=args.spsa_A, alpha=args.spsa_alpha,
        c=args.spsa_c, gamma=args.spsa_gamma,
        rho=args.spsa_rho, max_step_mm=args.spsa_max_step_mm,
        retries=args.spsa_retries, rng_seed=args.seed
    )

    updated_entries = propose_for_latest(entries, latest_run,
                                         seed=int(args.seed),
                                         bo_cfg=bo_cfg,
                                         penalty_offset=float(args.penalty_offset),
                                         seed_grid=bool(args.seed_grid),
                                         seed_delta=float(args.seed_delta),
                                         algo=args.algo,
                                         spsa_params=spsa_params)

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
