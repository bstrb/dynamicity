
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py — Uses external Step-1 (HillMap) module with fixed 2σ=R; Step-2 kept as before.
"""
from __future__ import annotations
import argparse, os, sys, math, hashlib
from typing import List, Tuple, Optional, Dict
import numpy as np

from step1_hillmap import Trial as Step1Trial, Step1Params, propose_step1

def _fmt6(x: float) -> str:
    return f"{x:.6f}"

def _finite_float_or_none(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan","none"}: return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _group_blocks(lines: List[str]) -> List[Tuple[str, List[str]]]:
    blocks = []
    cur_header = None
    cur = []
    for ln in lines:
        if ln.startswith("#/") and " event " in ln:
            if cur:
                blocks.append((cur_header, cur))
            cur_header = ln.rstrip("\n")
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append((cur_header, cur))
    return blocks

def _parse_event_header(header_line: str) -> Tuple[str, int]:
    try:
        prefix, ev = header_line[1:].split(" event ")
        return prefix.strip(), int(ev.strip())
    except Exception:
        return header_line.strip("# \n"), -1

def read_log(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def write_log(path: str, lines: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    os.replace(tmp, path)

def parse_blocks(lines: List[str]) -> Tuple[List[Tuple[str, List[str]]], int]:
    blocks = _group_blocks(lines)
    latest_run = -1
    for _, block in blocks:
        for ln in block:
            if not ln or ln.startswith("#") or ln.strip().startswith("run_n"):
                continue
            try:
                rn = int(ln.split(",")[0].strip())
                latest_run = max(latest_run, rn)
            except Exception:
                pass
    return blocks, latest_run

def update_block_latest_run(block_lines: List[str], latest_run: int, new_dx: Optional[float], new_dy: Optional[float]) -> List[str]:
    out = []
    for ln in block_lines:
        if not ln or ln.startswith("#") or ln.strip().startswith("run_n"):
            out.append(ln); continue
        parts = [p.strip() for p in ln.rstrip("\n").split(",")]
        if len(parts) < 7: parts += [""] * (7 - len(parts))
        try:
            rn = int(parts[0])
        except Exception:
            out.append(ln); continue
        if rn == latest_run:
            if new_dx is None and new_dy is None:
                parts[5] = "done"; parts[6] = "done"
            else:
                parts[5] = _fmt6(float(new_dx)); parts[6] = _fmt6(float(new_dy))
            ln = ",".join(parts) + "\n"
        out.append(ln)
    return out

# ---------- Step-2 (unchanged essentials) ----------

def _pd_project_2x2(H: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    w, V = np.linalg.eigh((H + H.T) * 0.5)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def _irls_quadratic_fit(successes: List[Tuple[float,float,float]], ridge: float = 1e-6, iters: int = 5) -> Tuple[np.ndarray, float]:
    X, y = [], []
    for x, yv, w in successes:
        X.append([x*x, yv*yv, x*yv, x, yv, 1.0])
        y.append(w)
    X = np.asarray(X, float); y = np.asarray(y, float)
    wts = np.ones_like(y)
    med = np.median(y); mad = np.median(np.abs(y - med)) + 1e-12
    scale = max(1.4826 * mad, 1e-12)
    for _ in range(iters):
        W = np.diag(wts)
        XtW = X.T @ W
        A = XtW @ X + ridge * np.eye(6)
        b = XtW @ y
        coef = np.linalg.solve(A, b)
        pred = X @ coef
        r = y - pred
        k = 1.345 * scale
        wts = np.where(np.abs(r) <= k, 1.0, k / (np.abs(r) + 1e-12))
    return coef, scale

def _quad_predict_grad(coef: np.ndarray, x: float, y: float) -> Tuple[float, np.ndarray, np.ndarray]:
    a,b,c,d,e,f = coef
    w = a*x*x + b*y*y + c*x*y + d*x + e*y + f
    gx = 2*a*x + c*y + d
    gy = 2*b*y + c*x + e
    H = np.array([[2*a, c],[c, 2*b]], float)
    return w, np.array([gx, gy], float), H

def _bound_disk(p: np.ndarray, R: float) -> np.ndarray:
    r = float(np.linalg.norm(p))
    if r <= R: return p
    if r == 0: return np.array([R, 0.0], float)
    return p * (R / r)

def _filter_spacing(cands: np.ndarray, tried: np.ndarray, min_spacing: float) -> np.ndarray:
    if tried.size == 0: return cands
    dif = cands[:, None, :] - tried[None, :, :]
    d2 = np.sum(dif*dif, axis=2)
    ok = np.all(d2 >= (min_spacing*min_spacing), axis=1)
    return cands[ok, :]

def _softargmin_xy(successes: List[Tuple[float,float,float]], tau: Optional[float] = None) -> np.ndarray:
    ws = np.array([w for _,_,w in successes], float)
    wmin = float(np.min(ws))
    if tau is None or tau <= 0: tau = max(1e-6, 0.5 * np.std(ws) + 1e-6)
    weights = np.exp(-(ws - wmin) / tau); weights /= np.sum(weights)
    xs = np.array([x for x,_,_ in successes], float)
    ys = np.array([y for _,y,_ in successes], float)
    return np.array([np.sum(weights*xs), np.sum(weights*ys)], float)

class Step2Params:
    def __init__(self, R, step_frac, direct_frac, fail_bump_frac, min_spacing, done_step_mm, done_grad):
        self.R = float(R); self.step_frac = float(step_frac); self.direct_frac = float(direct_frac)
        self.fail_bump_frac = float(fail_bump_frac); self.min_spacing = float(min_spacing)
        self.done_step_mm = float(done_step_mm); self.done_grad = float(done_grad)

def propose_step2(successes: List[Tuple[float,float,float]], failures: List[Tuple[float,float]], tried: np.ndarray, params: Step2Params, rng: np.random.Generator):
    best_idx = int(np.argmin([w for *_, w in successes]))
    xb = np.array([successes[best_idx][0], successes[best_idx][1]], float)

    coef, _ = _irls_quadratic_fit(successes, ridge=1e-6, iters=5)
    wb, gb, H = _quad_predict_grad(coef, xb[0], xb[1])
    H = _pd_project_2x2(H, eps=1e-9)

    try:
        xm = -np.linalg.solve(H, gb) + xb
    except Exception:
        xm = xb.copy()

    dist = float(np.linalg.norm(xm - xb))
    gnorm = float(np.linalg.norm(gb))
    if dist <= params.done_step_mm and gnorm <= params.done_grad:
        return None, None, "step2_done_converged"

    R = params.R
    def _toward(target: np.ndarray):
        d = target - xb
        L = float(np.linalg.norm(d))
        if L <= params.direct_frac * R:
            return _bound_disk(target.copy(), R), "toward_target_direct"
        if L == 0:
            return _bound_disk(xb.copy(), R), "toward_target_zero"
        step = xb + d * (params.step_frac * R / (L + 1e-12))
        return _bound_disk(step, R), "toward_target_step"

    cands = []
    p1, tag1 = _toward(_bound_disk(xm, R)); cands.append(("toward_minimizer_" + tag1, p1))
    xs = _softargmin_xy(successes)
    p2, tag2 = _toward(_bound_disk(xs, R)); cands.append(("toward_softargmin_" + tag2, p2))
    g = -gb; Lg = float(np.linalg.norm(g))
    p3 = xb + (g / (Lg + 1e-12)) * (params.step_frac * R) if Lg > 0 else xb.copy()
    cands.append(("along_neg_grad", _bound_disk(p3, R)))

    c_xy = np.stack([p for _,p in cands], axis=0)
    c_xy = _filter_spacing(c_xy, tried, params.min_spacing)
    cands_kept = [(tag, p) for (tag, p) in cands if any((p == c).all() for c in c_xy)]

    if len(c_xy) == 0:
        for a_deg in range(15, 181, 15):
            a = np.deg2rad(a_deg)
            Rm = np.array([[np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]], float)
            grot = Rm @ (-gb if Lg > 0 else np.array([1.0, 0.0]))
            step = xb + grot / (np.linalg.norm(grot) + 1e-12) * (params.step_frac * R)
            p = _bound_disk(step, R)
            c_xy2 = _filter_spacing(p[None, :], tried, params.min_spacing)
            if c_xy2.shape[0] > 0:
                cands_kept = [("grad_rotated", p)]; break

    if len(cands_kept) == 0:
        for _ in range(16):
            jitter = rng.normal(0.0, params.step_frac * R * 0.2, size=(2,))
            p = _bound_disk(xb + jitter, R)
            c_xy2 = _filter_spacing(p[None, :], tried, params.min_spacing)
            if c_xy2.shape[0] > 0:
                cands_kept = [("jitter", p)]; break

    if len(cands_kept) == 0:
        return xb[0], xb[1], "step2_no_feasible_found"

    def _pred_w(p: np.ndarray) -> float:
        w, _, _ = _quad_predict_grad(coef, float(p[0]), float(p[1]))
        return float(w)

    bumps_sigma = 0.5 * R
    bumps_amp = params.fail_bump_frac * (np.median([w for *_, w in successes]) if len(successes) else 1.0)
    def _fail_bump(p: np.ndarray) -> float:
        if not failures: return 0.0
        s = 0.0
        for fx, fy in failures:
            dx = p[0]-fx; dy = p[1]-fy
            s += math.exp(-0.5*(dx*dx+dy*dy)/(bumps_sigma*bumps_sigma))
        return bumps_amp * s

    best = None
    for tag, p in cands_kept:
        sc = _pred_w(p) + _fail_bump(p)
        if (best is None) or (sc < best[0]):
            best = (sc, tag, p)
    _, tag, p = best
    return float(p[0]), float(p[1]), "step2_commit:" + tag

def _recent_unindexed_streak(trials: List[Tuple[int,float,float,int,Optional[float]]]) -> int:
    s = 0
    for _, _, _, idx, _ in reversed(trials):
        if idx == 0: s += 1
        else: break
    return s

def _make_seed(abs_path: str, event_id: int, global_seed: int) -> int:
    key = f"{abs_path}::{event_id}::{global_seed}"
    return int.from_bytes(hashlib.blake2s(key.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFF

def propose_event(trials_sorted, R, rng,
                  step1_A0, step1_hill_frac, step1_drop_frac, step1_candidates, step1_explore_floor, min_spacing, allow_spacing_relax,
                  step2_step_frac, step2_direct_frac, step2_fail_bump_frac, back_to_step1_streak, done_step_mm, done_grad, N_success_to_step2):
    successes_w = []
    failures = []
    tried = []
    for _, dx, dy, idx, wr in trials_sorted:
        tried.append((dx, dy))
        if idx == 1 and _finite_float_or_none(wr) is not None:
            successes_w.append((dx, dy, float(wr)))
        elif idx == 0:
            failures.append((dx, dy))
    tried = np.array(tried, float) if tried else np.empty((0,2), float)

    n_succ = len(successes_w)
    if n_succ >= max(3, N_success_to_step2) and _recent_unindexed_streak(trials_sorted) < back_to_step1_streak:
        ndx, ndy, reason = propose_step2(successes_w, failures, tried,
                                         Step2Params(R, step2_step_frac, step2_direct_frac, step2_fail_bump_frac, min_spacing, done_step_mm, done_grad),
                                         rng)
        return ndx, ndy, reason

    first_center = (trials_sorted[0][1], trials_sorted[0][2]) if trials_sorted else (0.0, 0.0)
    s1_trials = [Step1Trial(dx, dy, idx, (_finite_float_or_none(wr))) for _, dx, dy, idx, wr in trials_sorted]
    s1_params = Step1Params(radius_mm=R, rng_seed=int(rng.integers(0, 2**31-1)), n_candidates=int(step1_candidates),
                            A0=float(step1_A0), hill_amp_frac=float(step1_hill_frac), drop_amp_frac=float(step1_drop_frac),
                            explore_floor=float(step1_explore_floor), min_spacing_mm=float(min_spacing),
                            first_attempt_center_mm=first_center, allow_spacing_relax=bool(allow_spacing_relax))
    res = propose_step1(s1_trials, s1_params)
    if res.done:
        return None, None, res.reason
    else:
        x, y = res.proposal_xy_mm
        return float(x), float(y), res.reason

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose next shifts per event (Step-1 imported, Step-2 local).")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--radius-mm", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--step1-A0", type=float, default=1.0)
    ap.add_argument("--step1-hill-amp-frac", type=float, default=0.8)
    ap.add_argument("--step1-drop-amp-frac", type=float, default=0.6)
    ap.add_argument("--step1-candidates", type=int, default=8192)
    ap.add_argument("--step1-explore-floor", type=float, default=1e-6)
    ap.add_argument("--step1-allow-spacing-relax", action="store_true")

    ap.add_argument("--step2-step-frac", type=float, default=0.5, help="Fraction of R to step along") 
    ap.add_argument("--step2-direct-frac", type=float, default=0.1)
    ap.add_argument("--step2-fail-bump-frac", type=float, default=0.01)
    ap.add_argument("--back-to-step1-streak", type=int, default=5)
    ap.add_argument("--done-step-mm", type=float, default=0.05)
    ap.add_argument("--done-grad", type=float, default=0.05)

    ap.add_argument("--min-spacing-mm", type=float, default=0.001)

    args = ap.parse_args(argv)

    log_path = os.path.join(args.run_root, "runs", "image_run_log.csv")
    if not os.path.isfile(log_path):
        print(f"ERROR: not found: {log_path}", file=sys.stderr); return 2

    lines = read_log(log_path)
    blocks, latest_run = parse_blocks(lines)
    if latest_run < 0:
        print("ERROR: Could not determine latest run_n in CSV.", file=sys.stderr); return 2

    new_lines: List[str] = []
    n_new, n_done = 0, 0

    for header, block in blocks:
        abs_path, event_id = _parse_event_header(header if header else "#/unknown event -1")

        rows = []
        for ln in block:
            s = ln.strip()
            # Skip comment lines or the global header line if it appears inside the block
            if (not s) or s.startswith("#") or s.startswith("run_n"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 7:
                parts += [""] * (7 - len(parts))
            # First field must be an int run number; otherwise skip
            try:
                rn = int(parts[0])
            except Exception:
                continue
            # Parse shifts
            try:
                dx = float(parts[1]); dy = float(parts[2])
            except Exception:
                dx, dy = 0.0, 0.0
            idx = 1 if parts[3] == "1" else 0
            wrs = parts[4]
            wr = None
            try:
                wr = float(wrs) if wrs not in ("", "nan", "None", "none") else None
            except Exception:
                wr = None
            rows.append((rn, dx, dy, idx, wr))

        trials_sorted = sorted(rows, key=lambda t: t[0])
        key_seed = _make_seed(abs_path, event_id, int(args.seed))
        rng = np.random.default_rng(key_seed)

        ndx, ndy, reason = propose_event(
            trials_sorted=trials_sorted,
            R=float(args.radius_mm),
            rng=rng,
            step1_A0=float(args.step1_A0),
            step1_hill_frac=float(args.step1_hill_amp_frac),
            step1_drop_frac=float(args.step1_drop_amp_frac),
            step1_candidates=int(args.step1_candidates),
            step1_explore_floor=float(args.step1_explore_floor),
            min_spacing=float(args.min_spacing_mm),
            allow_spacing_relax=bool(args.step1_allow_spacing_relax),
            step2_step_frac=float(args.step2_step_frac),
            step2_direct_frac=float(args.step2_direct_frac),
            step2_fail_bump_frac=float(args.step2_fail_bump_frac),
            back_to_step1_streak=int(args.back_to_step1_streak),
            done_step_mm=float(args.done_step_mm),
            done_grad=float(args.done_grad),
            N_success_to_step2=3,
        )

        updated = update_block_latest_run(block_lines=block, latest_run=latest_run, new_dx=ndx, new_dy=ndy)
        if ndx is None and ndy is None: n_done += 1
        else: n_new += 1
        new_lines.extend(updated)

    write_log(log_path, new_lines)
    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
