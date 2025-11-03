#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts_turbo.py — CSV parser + TuRBO-style GP-EI proposer (drop-in)

What this does
--------------
- Parses the same CSV as your original script.
- For each (path,event) group, collects past (dx, dy, indexed, wRMSD).
- If too little signal yet: explores with low-discrepancy random points inside bounds.
- Otherwise: fits a small GP (ARD RBF + noise), builds K trust regions around the
  top-K incumbents, and maximizes **Expected Improvement (EI)** within each region
  via randomized multistart sampling. Picks the best candidate overall.
- Optional “done” checks (tiny step, tiny GP grad, good enough wRMSD) to write "done".

Key differences vs your ring/bracketing code
--------------------------------------------
- Fully model-based: proposals come from maximizing EI using a GP posterior.
- TuRBO-style exploration: several small boxes (trust regions) that expand/contract
  implicitly by sizing them from local point scales; multi-start gives global reach.
- Keeps your CSV format and console diagnostics identical in spirit.

CSV format preserved:
  "#/abs/path event <int>"
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
"""

from __future__ import annotations
import argparse, os, sys, math, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np

from scipy.special import erf  # at the top

def _norm_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

# -------------------------- utils & parsing (unchanged spirit) --------------------------

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
    h = hashlib.sha1(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "little")
    frac = (v & ((1<<53)-1)) / float(1<<53)
    return 2.0 * math.pi * frac

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

# ------------------------------- small GP + EI ---------------------------------

def _rbf_aniso(X1, X2, lsx, lsy):
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_fit_predict(X, y, Xstar, lsx, lsy, noise):
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

def _auto_ls(X, y, R):
    # Use spread of top-k to set ARD lengthscales (robust-ish)
    k = max(3, int(0.4 * X.shape[0]))
    idx = np.argsort(y)[:k]
    Xk = X[idx]
    sx = float(np.std(Xk[:,0], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    sy = float(np.std(Xk[:,1], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    lsx = float(np.clip(1.25 * sx, 0.004 * R, 0.20 * R))
    lsy = float(np.clip(1.25 * sy, 0.004 * R, 0.20 * R))
    return lsx, lsy

def _expected_improvement(mu, var, f_best, xi=1e-6):
    """Expected Improvement (minimization) using SciPy erf."""
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    sigma = np.sqrt(np.maximum(var, 1e-16))
    delta = f_best - mu - xi
    z = delta / sigma

    Phi = _norm_cdf(z)
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

    ei = delta * Phi + sigma * phi
    ei = np.where(sigma < 1e-12, 0.0, ei)
    return np.maximum(ei, 0.0)


def _gp_grad_norm(bx, by, X, y, R, noise):
    if X.shape[0] < 4:
        return float("inf")
    lsx, lsy = _auto_ls(X, y, R)
    hx = max(1e-4, 0.1 * lsx); hy = max(1e-4, 0.1 * lsy)
    P = np.array([[bx + hx, by], [bx - hx, by], [bx, by + hy], [bx, by - hy]], float)
    mu, _ = _gp_fit_predict(X, y, P, lsx, lsy, noise)
    dmu_dx = float((mu[0] - mu[1]) / (2 * hx))
    dmu_dy = float((mu[2] - mu[3]) / (2 * hy))
    return float(math.hypot(dmu_dx, dmu_dy))

# ------------------------- TuRBO-style trust regions ---------------------------

def _low_disc_points_in_box(rng, n, center, half_side, R):
    # Cranley–Patterson rotated grid -> low-discrepancy-ish sampling in 2D box
    # box = [center - half_side, center + half_side], clipped to [-R, R]
    u = rng.random((n, 2))
    pts = center + (2.0 * u - 1.0) * half_side
    pts = np.clip(pts, -R, R)
    return pts

def _trust_region_sizes(X, y, centers, R, min_side_frac, max_side_frac):
    # Size each TR from local spread of nearby good points (robust iqr).
    sizes = []
    for c in centers:
        d = np.sqrt(((X - c)**2).sum(axis=1))
        # Focus on nearest neighbors (top-q by y then nearest in space)
        q = max(5, min(20, len(X)//2))
        good_idx = np.argsort(y)[:q]
        d_good = d[good_idx]
        if d_good.size >= 4:
            iqr = float(np.subtract(*np.percentile(d_good, [75, 25])))
            s = 2.0 * max(iqr, 1e-6)
        else:
            s = 0.1 * R
        s = float(np.clip(s, min_side_frac * R, max_side_frac * R))
        sizes.append(s)
    return np.array(sizes, float)

def _ei_in_trust_region(rng, X, y, R, center, half_side, noise, n_cand):
    # Build ARD GP, sample candidates in TR, compute EI, pick best
    lsx, lsy = _auto_ls(X, y, R)
    fc = float(np.min(y))
    C = _low_disc_points_in_box(rng, n_cand, center, half_side, R)
    mu, var = _gp_fit_predict(X, y, C, lsx, lsy, noise)
    ei = _expected_improvement(mu, var, fc)
    j = int(np.argmax(ei))
    return C[j], float(ei[j])

# ------------------------------ main proposer ---------------------------------

def propose_event_turbo(trials_sorted,
                        global_clip,
                        rng: np.random.Generator,
                        min_good_for_best: int,
                        turbo_K: int,
                        turbo_min_side_frac: float,
                        turbo_max_side_frac: float,
                        turbo_candidates_per_TR: int,
                        turbo_noise: float,
                        explore_candidates: int,
                        done_step_mm: float,
                        done_w: float,
                        done_grad: float):
    """
    Return (next_dx, next_dy, reason_str) or ('done','done',reason)
    """
    # Track tried coordinates to avoid repeats
    tried_set = set((_fmt6(dx), _fmt6(dy)) for (_, dx, dy, _, _) in trials_sorted)
    # Keep only successful & finite wRMSD for GP fitting
    successes = [(dx, dy, wr) for (_, dx, dy, ind, wr) in trials_sorted
                 if ind and (wr is not None) and math.isfinite(wr)]

    # Fallback center (first dx,dy or (0,0))
    if len(trials_sorted) > 0:
        init_cx, init_cy = trials_sorted[0][1], trials_sorted[0][2]
    else:
        init_cx, init_cy = 0.0, 0.0

    # Not enough good data yet -> space-filling exploration in box
    if len(successes) < max(3, min_good_for_best):
        for _ in range(5):  # a few tries to avoid duplicates
            cand = _low_disc_points_in_box(rng, 1, np.array([init_cx, init_cy]), 0.50 * global_clip, global_clip)[0]
            key = (_fmt6(float(cand[0])), _fmt6(float(cand[1])))
            if key not in tried_set:
                return float(cand[0]), float(cand[1]), "explore_warmup"
        # as last resort jitter around center
        ang = rng.uniform(0, 2*np.pi); rad = 0.2 * global_clip
        nx = float(np.clip(init_cx + rad * math.cos(ang), -global_clip, global_clip))
        ny = float(np.clip(init_cy + rad * math.sin(ang), -global_clip, global_clip))
        return nx, ny, "explore_warmup_jitter"

    # Prepare GP training data
    X = np.array([[dx, dy] for (dx, dy, wr) in successes], float)
    y = np.array([wr for (_, _, wr) in successes], float)
    jbest = int(np.argmin(y))
    bx, by = float(X[jbest, 0]), float(X[jbest, 1])
    ybest = float(y[jbest])

    # Early "done" checks: small modeled gradient + tiny step, and good-enough wRMSD
    gnorm = _gp_grad_norm(bx, by, X, y, global_clip, turbo_noise)
    if (ybest <= done_w) and (gnorm < done_grad):
        return "done", "done", "map_converged"

    # Choose K trust-region centers = top-K incumbents (by y then spatial diversity)
    order = np.argsort(y)
    centers = []
    for idx in order:
        c = X[idx]
        if not centers:
            centers.append(c)
        else:
            # ensure some diversity among centers
            d = np.sqrt(((np.array(centers) - c)**2).sum(axis=1))
            if (d.min() if len(d)>0 else 1e9) > 1e-6 * global_clip:
                centers.append(c)
        if len(centers) >= turbo_K:
            break
    centers = np.array(centers, float)

    # Size each trust region from local spread
    half_sides = 0.5 * _trust_region_sizes(X, y, centers, global_clip,
                                           min_side_frac=turbo_min_side_frac,
                                           max_side_frac=turbo_max_side_frac)

    # Evaluate EI in each TR via randomized candidates; keep the best overall
    best_cand = None
    best_ei = -1.0
    for ci in range(centers.shape[0]):
        cand, ei_val = _ei_in_trust_region(rng, X, y, global_clip,
                                           center=centers[ci],
                                           half_side=half_sides[ci],
                                           noise=turbo_noise,
                                           n_cand=turbo_candidates_per_TR)
        if ei_val > best_ei:
            best_ei = ei_val
            best_cand = cand

    # Add a few global candidates (exploration) and keep the best EI among all
    if explore_candidates > 0:
        Cg = _low_disc_points_in_box(rng, explore_candidates,
                                     center=np.array([0.0, 0.0]),
                                     half_side=global_clip,
                                     R=global_clip)
        lsx, lsy = _auto_ls(X, y, global_clip)
        mu_g, var_g = _gp_fit_predict(X, y, Cg, lsx, lsy, turbo_noise)
        ei_g = _expected_improvement(mu_g, var_g, float(np.min(y)))
        jg = int(np.argmax(ei_g))
        if float(ei_g[jg]) > best_ei:
            best_cand = Cg[jg]
            best_ei = float(ei_g[jg])

    # If the best candidate is already tried, nudge toward the incumbent
    if best_cand is None:
        return "done", "done", "no_candidate"
    nx, ny = float(best_cand[0]), float(best_cand[1])
    if (_fmt6(nx), _fmt6(ny)) in tried_set:
        dx = bx - nx; dy = by - ny
        norm = math.hypot(dx, dy)
        if norm > 1e-12:
            nx += 0.25 * dx; ny += 0.25 * dy

    nx = float(np.clip(nx, -global_clip, global_clip))
    ny = float(np.clip(ny, -global_clip, global_clip))

    # Final "done" check: microscopic step relative to incumbent and good ybest
    step = math.hypot(nx - bx, ny - by)
    if (step < done_step_mm) and (ybest <= done_w) and (gnorm < done_grad):
        return "done", "done", "map_converged_small_step"

    return nx, ny, "turbo_gp_ei"

# ----------------------------------- CLI --------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-event proposer: TuRBO-style GP-EI using CSV history (drop-in replacement).")
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Global half-width limit for |dx| and |dy|.")
    # Convergence knobs (same semantics as original)
    ap.add_argument("--min-good-for-best", type=int, default=2, help="Min successes (indexed + finite wRMSD) before model-based TuRBO.")
    ap.add_argument("--bo-noise", type=float, default=1e-4, help="GP observation noise for wRMSD.")
    ap.add_argument("--done-step-mm", type=float, default=1e-3, help="Stop if proposed step from incumbent is smaller than this (mm).")
    ap.add_argument("--done-wrmsd", type=float, default=0.10, help="Stop if incumbent best wRMSD <= this.")
    ap.add_argument("--done-grad", type=float, default=3e-3, help="Stop if GP-mean gradient norm is below this (mm^-1).")
    ap.add_argument("--seed", type=int, default=1337)
    # TuRBO-specific knobs
    ap.add_argument("--turbo-K", type=int, default=3, help="Number of trust region centers (top-K incumbents).")
    ap.add_argument("--turbo-min-side-frac", type=float, default=0.1, help="Min TR side as fraction of radius (per side).")
    ap.add_argument("--turbo-max-side-frac", type=float, default=1.0, help="Max TR side as fraction of radius (per side).")
    ap.add_argument("--turbo-candidates-per-TR", type=int, default=512, help="Random candidate count per TR for EI search.")
    ap.add_argument("--turbo-explore-candidates", type=int, default=256, help="Extra global candidates to avoid premature focus.")
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
        # separate rng per key for reproducibility/diversity
        key_seed = int(abs(hash((args.seed, key[0], key[1]))) % (2**32 - 1))
        rng = np.random.default_rng(key_seed)

        ndx, ndy, reason = propose_event_turbo(
            trials_sorted=trials,
            global_clip=float(args.radius_mm),
            rng=rng,
            min_good_for_best=max(3, int(args.min_good_for_best)),
            turbo_K=int(args.turbo_K),
            turbo_min_side_frac=float(args.turbo_min_side_frac),
            turbo_max_side_frac=float(args.turbo_max_side_frac),
            turbo_candidates_per_TR=int(args.turbo_candidates_per_TR),
            turbo_noise=float(args.bo_noise),
            explore_candidates=int(args.turbo_explore_candidates),
            done_step_mm=float(args.done_step_mm),
            done_w=float(args.done_wrmsd),
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
        # print(f"[diag:{os.path.basename(key[0])}|event{key[1]}] reason={reason}")

    write_log(log_path, entries)
    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
