# -*- coding: utf-8 -*-
"""
propose_next_shifts.py

Goal: find detector center shifts (dx, dy) that MINIMIZE wRMSD per (image,event),
even when many evaluations fail to index (missing wRMSD).

This hybrid strategy has three phases:
  0) Seeding (if no successes yet):
     - Expanding ring around the initial center to raise success probability.

  1) Feasibility-first guidance (if few successes, < N_succ_for_quad):
     - Train a simple logistic model P(success | x) using all attempts.
     - Propose the next point by moving from the incumbent toward the local
       maximum of P(success) (hill-climb) within a TRUST REGION and STEP CAP.

  2) Local quadratic fit on wRMSD (once we have enough successes):
     - Fit a ridge-regularized quadratic model y = b0 + b1 x + b2 y + b3 x^2 + b4 xy + b5 y^2
       on the last M successful points (x=dx, y=dy).
     - Compute the stationary point; if positive-definite, move toward it.
     - Otherwise, fallback to a cautious line search toward the best linear-descent direction.
     - Always respect TRUST REGION and STEP CAP and avoid duplicates.

Guard rails:
  - If predicted next point is too close to a tried point, jitter a few times; else mark done.
  - If recent best improvement is tiny and best wRMSD sits inside a robust "good band", mark done.

CLI is intentionally minimal; defaults are tuned for mm-scale shifts and target wRMSD ~0.1–0.2.
"""

from __future__ import annotations
import argparse, math, os, sys, statistics, random
from typing import Dict, List, Tuple, Optional
import numpy as np

# ----------------- Defaults -----------------
DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
SEED_DEFAULT = 1337
CONVERGE_TOL_DEFAULT = 5e-4  # stop if tiny move from best

# ----------------- Config -----------------
class HybridCFG:
    def __init__(self,
        rho=0.040,                # trust-region half-size [mm]
        max_step_mm=0.010,        # hard cap on |step| from incumbent [mm]
        seed_radii=(0.006,0.012), # initial expanding ring radii [mm]
        seed_dirs=8,              # number of directions around ring
        succ_window=10,           # window for recent success rate
        succ_shrink_thresh=0.30,  # shrink step if success rate below this
        shrink_factor=0.6,
        N_succ_for_quad=6,        # min #successes to enable quadratic fit
        quad_M_recent=20,         # use up to M most-recent successes for fit
        quad_ridge=1e-3,          # ridge on quadratic normal equations
        line_search_steps=(1.0, 0.6, 0.35),  # backtracking fractions
        rng_seed=SEED_DEFAULT
    ):
        self.rho=float(rho)
        self.max_step_mm=float(max_step_mm) if max_step_mm is not None else None
        self.seed_radii=tuple(float(r) for r in seed_radii)
        self.seed_dirs=int(seed_dirs)
        self.succ_window=int(succ_window)
        self.succ_shrink_thresh=float(succ_shrink_thresh)
        self.shrink_factor=float(shrink_factor)
        self.N_succ_for_quad=int(N_succ_for_quad)
        self.quad_M_recent=int(quad_M_recent)
        self.quad_ridge=float(quad_ridge)
        self.line_search_steps=tuple(float(s) for s in line_search_steps)
        self.rng = np.random.default_rng(int(rng_seed))

# ----------------- Utils -----------------
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

# ----------------- Local helpers -----------------
def _incumbent_from_history(trials_sorted):
    # Return best successful (indexed, finite wr) as (xy, y)
    best_xy, best_y = None, float("inf")
    for (_, dx, dy, ind, wr) in trials_sorted:
        if ind and wr is not None and math.isfinite(wr):
            if wr < best_y:
                best_y = float(wr); best_xy = (float(dx), float(dy))
    return best_xy, best_y

def _recent_success_rate(trials_sorted, window=10):
    zs = [(1 if (ind and wr is not None and math.isfinite(wr)) else 0)
          for (_,_,_,ind,wr) in trials_sorted[-window:]]
    return (sum(zs)/len(zs)) if zs else 0.0

def _design_matrix_quadratic(xs, ys):
    # Columns: [1, x, y, x^2, x*y, y^2]
    X = np.column_stack([
        np.ones_like(xs),
        xs, ys,
        xs*xs, xs*ys, ys*ys
    ])
    return X

def _fit_quadratic_ridge(points, ridge=1e-3):
    # points: list of (x, y, wrmsd) with wrmsd finite
    arr = np.array(points, float)
    xs, ys, ws = arr[:,0], arr[:,1], arr[:,2]
    X = _design_matrix_quadratic(xs, ys)
    y = ws
    # Solve (X^T X + λI) β = X^T y
    XT = X.T
    H = XT @ X
    H += ridge * np.eye(H.shape[0])
    beta = np.linalg.solve(H, XT @ y)
    # Hessian of quadratic wrt (x,y): [[2*b3, b4], [b4, 2*b5]]
    b0,b1,b2,b3,b4,b5 = beta
    Hxy = np.array([[2*b3, b4],[b4, 2*b5]], float)
    g = np.array([b1, b2], float)  # gradient at origin
    # Stationary point solves Hxy * [x;y] + g = 0 -> [x;y] = -Hxy^{-1} g
    try:
        xstar = -np.linalg.solve(Hxy, g)
        # Check PD Hessian
        eig = np.linalg.eigvalsh(Hxy)
        is_min = np.all(eig > 1e-8)
    except np.linalg.LinAlgError:
        xstar = None; is_min = False
    return beta, Hxy, g, xstar, is_min

def _clip_trust_and_step(bx, by, tx, ty, rho, max_step):
    xmin, xmax = bx - rho, bx + rho
    ymin, ymax = by - rho, by + rho
    tx = float(np.clip(tx, xmin, xmax))
    ty = float(np.clip(ty, ymin, ymax))
    if max_step is not None:
        dx, dy = (tx - bx), (ty - by)
        r = math.hypot(dx, dy)
        if r > max_step and r > 0.0:
            s = max_step / r
            tx, ty = (bx + s*dx, by + s*dy)
    return tx, ty

def _avoid_duplicates(candidate, tried_set, jitter=0.0008, attempts=6):
    x, y = candidate
    if (_fmt6(x), _fmt6(y)) not in tried_set:
        return candidate, True
    # jitter a few times
    for _ in range(attempts):
        jx = x + (random.random()*2-1)*jitter
        jy = y + (random.random()*2-1)*jitter
        if (_fmt6(jx), _fmt6(jy)) not in tried_set:
            return (jx, jy), True
    return candidate, False

def _robust_done_criterion(trials_sorted, good_band_upper):
    # If best-sequence improved < 0.01 over last few successes and best <= good_band, mark done.
    indexed_wr = [wr for (_,_,_,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
    if not indexed_wr:
        return False
    best_seq = []
    m = float('inf')
    for (_,_,_,ind,wr) in trials_sorted:
        if ind and (wr is not None) and math.isfinite(wr):
            m = min(m, float(wr)); best_seq.append(m)
    if len(best_seq) >= 3:
        recent = best_seq[-min(5, len(best_seq)):]
        if (recent[0] - recent[-1]) < 0.01 and (recent[-1] <= good_band_upper):
            return True
    return False

# ----------------- Core proposer -----------------
def propose_for_latest(entries, latest_run: int, cfg: HybridCFG):
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

    # ----------------- Make proposals per event -----------------
    n_new, n_done = 0, 0
    for key, trials in history.items():
        if key not in latest_rows_by_key:
            continue
        trials_sorted = sorted(trials, key=lambda t: t[0])
        tried_points = set((_fmt6(dx), _fmt6(dy)) for (_,dx,dy,_,_) in trials_sorted)

        # Early 'done' check
        if _robust_done_criterion(trials_sorted, good_band_upper):
            ndx, ndy = ("done", "done")
        else:
            # Incumbent among successes
            best_xy, best_y = _incumbent_from_history(trials_sorted)

            if best_xy is None:
                # -------- Phase 0: Seeding (expanding ring) --------
                # Start from first row center
                start_cx, start_cy = (trials_sorted[0][1], trials_sorted[0][2]) if trials_sorted else (0.0, 0.0)
                proposed = None
                for r in cfg.seed_radii:
                    for kdir in range(cfg.seed_dirs):
                        ang = (2*math.pi * kdir) / cfg.seed_dirs
                        tx = start_cx + r*math.cos(ang)
                        ty = start_cy + r*math.sin(ang)
                        tx, ty = _clip_trust_and_step(start_cx, start_cy, tx, ty, cfg.rho, cfg.max_step_mm)
                        (tx, ty), ok = _avoid_duplicates((tx,ty), tried_points)
                        if ok:
                            proposed = (tx, ty); break
                    if proposed is not None: break
                if proposed is None:
                    ndx, ndy = ("done","done")
                else:
                    ndx, ndy = proposed

            else:
                bx, by = best_xy

                # Count successes
                succ_points = [(dx,dy,wr) for (_,dx,dy,ind,wr) in trials_sorted if ind and (wr is not None) and math.isfinite(wr)]
                # Recent success rate to optionally shrink steps
                p_recent = _recent_success_rate(trials_sorted, window=cfg.succ_window)

                if len(succ_points) >= cfg.N_succ_for_quad:
                    # -------- Phase 2: Quadratic fit on wRMSD --------
                    # Use up to M most-recent successes
                    recent_succ = succ_points[-cfg.quad_M_recent:]
                    beta, Hxy, g, xstar, is_min = _fit_quadratic_ridge(recent_succ, ridge=cfg.quad_ridge)

                    if xstar is not None and is_min:
                        tx, ty = float(xstar[0]), float(xstar[1])
                    else:
                        # Fallback: move opposite gradient (steepest descent at incumbent ~ g + H*[bx,by])
                        # Evaluate gradient at incumbent for quadratic: ∇ = g + Hxy @ [bx,by]
                        grad = g + Hxy @ np.array([bx, by], float)
                        if np.linalg.norm(grad) > 0:
                            dirn = -grad / np.linalg.norm(grad)
                        else:
                            dirn = np.array([0.0,0.0])
                        # Try a short backtracking line search
                        tx, ty = bx, by
                        moved = False
                        for s in cfg.line_search_steps:
                            cand = (bx + s*cfg.rho*dirn[0], by + s*cfg.rho*dirn[1])
                            cand = _clip_trust_and_step(bx, by, cand[0], cand[1], cfg.rho, cfg.max_step_mm)
                            (cand, ok) = _avoid_duplicates(cand, tried_points)
                            if ok:
                                tx, ty = cand; moved = True; break
                        if not moved:
                            ndx, ndy = ("done","done")
                            tx, ty = bx, by  # not used

                    # Apply success-rate shrink
                    if isinstance(tx, float) and isinstance(ty, float) and p_recent < cfg.succ_shrink_thresh:
                        tx = bx + cfg.shrink_factor*(tx - bx)
                        ty = by + cfg.shrink_factor*(ty - by)

                    if isinstance(tx, float) and isinstance(ty, float):
                        tx, ty = _clip_trust_and_step(bx, by, tx, ty, cfg.rho, cfg.max_step_mm)
                        (tx, ty), ok = _avoid_duplicates((tx,ty), tried_points)
                        ndx, ndy = (tx, ty) if ok else ("done","done")
                    else:
                        ndx, ndy = ("done","done")

                else:
                    # -------- Phase 1: Feasibility-first guidance --------
                    # Train a tiny logistic model on success = 1(indexed&finite wr), 0 otherwise
                    pts = np.array([(dx,dy) for (_,dx,dy,_,_) in trials_sorted], float)
                    ylab = np.array([1 if (ind and (wr is not None) and math.isfinite(wr)) else 0
                                     for (_,_,_,ind,wr) in trials_sorted], float)
                    # Design matrix [1, x, y]
                    X = np.column_stack([np.ones(len(pts)), pts[:,0], pts[:,1]])
                    # Ridge-regularized IRLS (few steps)
                    w = np.zeros(3)
                    lam = 1e-2
                    for _ in range(5):
                        z = X @ w
                        p = 1/(1+np.exp(-np.clip(z, -20, 20)))
                        W = p*(1-p)
                        # Avoid zero weights
                        W = np.maximum(W, 1e-4)
                        # IRLS: (X^T W X + λI) w = X^T W z_tilde
                        ztilde = z + (ylab - p)/W
                        XTWX = X.T @ (W[:,None]*X) + lam*np.eye(3)
                        XTWz = X.T @ (W*ztilde)
                        try:
                            w = np.linalg.solve(XTWX, XTWz)
                        except np.linalg.LinAlgError:
                            break
                    # Move along gradient of p at incumbent
                    # ∇p = p(1-p) * [w1, w2] at (bx,by)
                    pb = 1/(1+np.exp(-(w[0] + w[1]*bx + w[2]*by)))
                    gradp = pb*(1-pb) * np.array([w[1], w[2]], float)
                    if np.linalg.norm(gradp) > 0:
                        dirn = gradp/np.linalg.norm(gradp)
                    else:
                        # Random small direction if flat
                        dirn = np.array([1.0,0.0])
                    # Proposed target is a short step to increase success probability
                    step = 0.6*cfg.rho
                    tx, ty = bx + step*dirn[0], by + step*dirn[1]
                    if p_recent < cfg.succ_shrink_thresh:
                        tx = bx + cfg.shrink_factor*(tx - bx)
                        ty = by + cfg.shrink_factor*(ty - by)
                    tx, ty = _clip_trust_and_step(bx, by, tx, ty, cfg.rho, cfg.max_step_mm)
                    (tx, ty), ok = _avoid_duplicates((tx,ty), tried_points)
                    ndx, ndy = (tx, ty) if ok else ("done","done")

        # Apply to latest-run rows for this event
        for row_idx in latest_rows_by_key[key]:
            row = list(entries[row_idx][1])
            if len(row) < 7: row += [""] * (7 - len(row))
            if row[0].isdigit():
                if isinstance(ndx, str):  # "done"
                    row[5] = "done"; row[6] = "done"; n_done += 1
                else:
                    row[5] = _fmt6(float(ndx)); row[6] = _fmt6(float(ndy)); n_new += 1
                entries[row_idx] = (None, tuple(row))

    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    return entries

# ----------------- Main -----------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid proposer: feasibility-first + local quadratic fit for wRMSD.")
    ap.add_argument("--run-root", default=None, help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--sidecar", help="If provided, write proposals to this CSV instead of rewriting the log.")

    # Trust region and steps
    ap.add_argument("--rho", type=float, default=0.040, help="Trust-region half-size (mm).")
    ap.add_argument("--max-step-mm", type=float, default=0.010, help="Hard cap on |step| from incumbent (mm).")

    # Seeding
    ap.add_argument("--seed-radii", type=str, default="0.006,0.012", help="Comma-separated ring radii (mm).")
    ap.add_argument("--seed-dirs", type=int, default=8, help="Number of directions for ring seeding.")

    # Success-aware moves
    ap.add_argument("--succ-window", type=int, default=10, help="Window for recent success rate.")
    ap.add_argument("--succ-shrink-thresh", type=float, default=0.30, help="Shrink step if recent success rate falls below this.")
    ap.add_argument("--shrink-factor", type=float, default=0.6, help="Factor to shrink moves when success rate is low.")

    # Quadratic
    ap.add_argument("--N-succ-for-quad", type=int, default=6, help="Minimum # of successes before enabling quadratic fit.")
    ap.add_argument("--quad-M-recent", type=int, default=20, help="Use up to M most-recent successes for quadratic fit.")
    ap.add_argument("--quad-ridge", type=float, default=1e-3, help="Ridge strength for quadratic fit.")
    ap.add_argument("--line-search-steps", type=str, default="1.0,0.6,0.35", help="Comma-separated backtracking fractions.")

    args = ap.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_root = os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_ROOT))
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr); return 2

    # Build config
    seed_radii = tuple(float(s) for s in args.seed_radii.split(",")) if args.seed_radii else (0.006,0.012)
    line_steps = tuple(float(s) for s in args.line_search_steps.split(",")) if args.line_search_steps else (1.0,0.6,0.35)
    cfg = HybridCFG(
        rho=args.rho,
        max_step_mm=args.max_step_mm,
        seed_radii=seed_radii,
        seed_dirs=args.seed_dirs,
        succ_window=args.succ_window,
        succ_shrink_thresh=args.succ_shrink_thresh,
        shrink_factor=args.shrink_factor,
        N_succ_for_quad=args.N_succ_for_quad,
        quad_M_recent=args.quad_M_recent,
        quad_ridge=args.quad_ridge,
        line_search_steps=line_steps,
        rng_seed=args.seed
    )

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr); return 2

    updated_entries = propose_for_latest(entries, latest_run, cfg=cfg)

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