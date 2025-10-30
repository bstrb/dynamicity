#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts_ring_then_best.py

Per-frame proposer that combines:
  (1) Expanding ring search until the first successful index (finite wRMSD).
  (2) Upon each success (until enough successes), RESTART a smaller expanding ring
      centered very close to that success (offset = frac * current r_step).
  (3) Once we have >= N successful points (tunable, >=3), pick the "most likely
      best" point: the minimizer of the GP posterior mean (local grid) a.k.a.
      the lowest predicted wRMSD given the current surface; propose that.

Minimal knobs. Everything else adapts to the data.
- Ring: r_max, r_step, k_base
- Shrink: ring_shrink (applied to both r_max and r_step after each success)
- Offset: center_offset_frac (fraction of current r_step)
- Switch: min_good_for_best (>=3)

CSV compatibility
-----------------
Keeps grouped CSV format:
  "#/abs/path event <int>"
  run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm

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

# ------------------------------ Ring search core ------------------------------

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

# ------------------------------ GP mean minimizer (final pick) ------------------------------

def _rbf_aniso(X1: np.ndarray, X2: np.ndarray, lsx: float, lsy: float) -> np.ndarray:
    dx = (X1[:, None, 0] - X2[None, :, 0]) / max(lsx, 1e-12)
    dy = (X1[:, None, 1] - X2[None, :, 1]) / max(lsy, 1e-12)
    return np.exp(-0.5 * (dx * dx + dy * dy))

def _gp_mean(X: np.ndarray, y: np.ndarray, Xstar: np.ndarray, lsx: float, lsy: float, noise: float) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("GP requires data")
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
    return mu

def _auto_ls(X: np.ndarray, y: np.ndarray, R: float) -> Tuple[float,float]:
    """Automatic, data-driven lengthscales from the spread of the top points."""
    k = max(3, int(0.4 * X.shape[0]))
    idx = np.argsort(y)[:k]
    Xk = X[idx]
    sx = float(np.std(Xk[:,0], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    sy = float(np.std(Xk[:,1], ddof=1)) if Xk.shape[0] > 1 else 0.10 * R
    # Clamp to reasonable range vs global radius
    lsx = float(np.clip(1.25 * sx, 0.004 * R, 0.20 * R))
    lsy = float(np.clip(1.25 * sy, 0.004 * R, 0.20 * R))
    return lsx, lsy

def _pick_best_from_surface(X_good: np.ndarray, y_good: np.ndarray,
                            R: float,
                            grid_half: float,
                            grid_n: int = 33,
                            noise: float = 1e-4) -> Tuple[float,float]:
    """
    Build GP on good points and minimize its mean on a square grid centered at the incumbent best.
    Returns (nx, ny). If GP fails, return the incumbent best.
    """
    jbest = int(np.argmin(y_good))
    bx, by = float(X_good[jbest,0]), float(X_good[jbest,1])
    try:
        lsx, lsy = _auto_ls(X_good, y_good, R)
        xs = np.linspace(bx - grid_half, bx + grid_half, grid_n)
        ys = np.linspace(by - grid_half, by + grid_half, grid_n)
        GX, GY = np.meshgrid(xs, ys)
        Xgrid = np.column_stack([GX.ravel(), GY.ravel()])
        mu = _gp_mean(X_good, y_good, Xgrid, lsx, lsy, noise).reshape(GX.shape)
        j = int(np.argmin(mu))
        gy, gx = divmod(j, mu.shape[1])
        nx, ny = float(GX[gy, gx]), float(GY[gy, gx])
        return nx, ny
    except Exception:
        return bx, by

# ------------------------------ Proposal per event ------------------------------

def propose_event(trials_sorted: List[Tuple[int,float,float,int,Optional[float]]],
                  r_max0: float, r_step0: float, k_base: float,
                  ring_shrink: float, center_offset_frac: float,
                  min_good_for_best: int, global_clip: float,
                  base_angle: float, seed: int) -> Tuple[object, object, str]:
    """
    trials_sorted: [(run_n, dx, dy, indexed, wrmsd)] sorted by run_n
    Returns: (next_dx, next_dy, reason) where next_* may be "done".
    """
    tried_set = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)
    successes = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
    n_succ = len(successes)

    # Stage-dependent ring parameters
    # stage 0: before first success; center = initial det_shift (or 0,0)
    if len(trials_sorted) > 0:
        init_cx, init_cy = trials_sorted[0][1], trials_sorted[0][2]
    else:
        init_cx, init_cy = 0.0, 0.0

    # Determine stage index s (number of successes so far, capped)
    s = n_succ

    # If we have fewer than min_good_for_best successes: RING PHASE
    if n_succ < max(3, min_good_for_best):
        # Compute current ring parameters
        shrink_pow = max(0, s)  # shrink after each success
        r_max = r_max0 * (ring_shrink ** shrink_pow)
        r_step = max(1e-6, r_step0 * (ring_shrink ** shrink_pow))

        # Ring center
        if s == 0:
            cx, cy = init_cx, init_cy
        else:
            # latest success
            last_sx, last_sy, _ = successes[-1]
            eps = center_offset_frac * (r_step0 * (ring_shrink ** (s-1)))
            # offset along a rotated base angle per stage to avoid revisiting same rays
            theta = base_angle + s * GOLDEN_ANGLE * 0.5
            cx = last_sx + eps * math.cos(theta + math.pi/2.0)
            cy = last_sy + eps * math.sin(theta + math.pi/2.0)

        # Phase shift per stage to vary angles
        phase = s * GOLDEN_ANGLE

        cand = _ring_candidate(center=(cx,cy),
                               tried_set=tried_set,
                               r_max=r_max, r_step=r_step, k_base=k_base,
                               base_angle=base_angle, phase=phase,
                               global_clip=global_clip)
        if cand is None:
            return ("done", "done", "ring_exhausted")
        return (float(cand[0]), float(cand[1]), "ring_stage")

    # Otherwise: FINAL PICK from surface (GP mean minimizer)
    X_good = np.array([[dx,dy] for (dx,dy,wr) in successes], float)
    y_good = np.array([wr for (dx,dy,wr) in successes], float)
    jbest = int(np.argmin(y_good))
    bx, by = float(X_good[jbest,0]), float(X_good[jbest,1])

    # Define a local grid half-width from the distribution of distances to best
    d = np.sqrt(np.sum((X_good - np.array([bx,by]))**2, axis=1))
    if d.size >= 3 and np.all(np.isfinite(d)):
        half = float(np.quantile(d, 0.60))
    else:
        half = 0.02
    half = float(np.clip(half, 0.006, 0.04))

    nx, ny = _pick_best_from_surface(X_good, y_good, R=global_clip, grid_half=half, grid_n=33, noise=1e-4)
    nx = float(np.clip(nx, -global_clip, global_clip))
    ny = float(np.clip(ny, -global_clip, global_clip))

    if (_fmt6(nx), _fmt6(ny)) in tried_set:
        # If predicted min already tried, nudge slightly toward the incumbent best
        dx = bx - nx; dy = by - ny
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            # tiny random dither (stable per event)
            rng = np.random.default_rng(abs(hash((base_angle, seed))) % (2**32-1))
            nx += 1e-3 * (rng.random() - 0.5)
            ny += 1e-3 * (rng.random() - 0.5)
        else:
            nx += 0.25 * dx; ny += 0.25 * dy
        nx = float(np.clip(nx, -global_clip, global_clip))
        ny = float(np.clip(ny, -global_clip, global_clip))

    return (float(nx), float(ny), "best_from_surface")

# ------------------------------ Main ------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-frame center-shift proposer: expanding ring -> (on success) smaller ring -> (after N successes) best-from-surface.")
    ap.add_argument("--run-root", required=True, help="Path that contains 'runs/image_run_log.csv'.")

    # Global clamp for absolute shift domain (square [-R,R]^2).
    ap.add_argument("--radius-mm", type=float, default=0.05, help="Global half-width limit for |dx| and |dy|.")

    # Ring parameters
    ap.add_argument("--r-max", type=float, default=0.05, help="Initial ring maximum radius (mm).")
    ap.add_argument("--r-step", type=float, default=0.01, help="Ring radial increment (mm).")
    ap.add_argument("--k-base", type=float, default=20.0, help="Angular density scale (angles per ring ∝ k_base * r/r_max).")
    ap.add_argument("--ring-shrink", type=float, default=0.50, help="Factor to reduce r_max and r_step after each success (slightly < 1).")
    ap.add_argument("--center-offset-frac", type=float, default=0.1, help="Offset of new ring center = frac * current r_step (to avoid cycling same rays).")

    # Switch to final pick after this many successful points (>=3).
    ap.add_argument("--min-good-for-best", type=int, default=5, help="Number of successful (indexed + finite wRMSD) points before selecting best from surface.")

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
