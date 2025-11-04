#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py — Uses external Step-1 (HillMap); Step-2 imported from step2_meanshift.py (mean-shift jump) or step2_bayes.py (Bayesian optimization jump)
"""
from __future__ import annotations
import argparse, os, sys, math, hashlib
from typing import List, Tuple, Optional, Dict
import numpy as np

from step1_hillmap import Trial as Step1Trial, Step1Params, propose_step1
# NEW: import the mean-shift and bayes Step-2 module
from step2_meanshift import propose_step2_meanshift, Step2MeanShiftConfig
from step2_bayes import propose_step2_bayes, Step2BayesConfig 

# ------------------------ CSV helpers ------------------------

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

# ------------------------ Small utils ------------------------

def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _recent_unindexed_streak(trials: List[Tuple[int,float,float,int,Optional[float]]]) -> int:
    s = 0
    for _, _, _, idx, _ in reversed(trials):
        if idx == 0: s += 1
        else: break
    return s

def _make_seed(abs_path: str, event_id: int, global_seed: int) -> int:
    key = f"{abs_path}::{event_id}::{global_seed}"
    return int.from_bytes(hashlib.blake2s(key.encode(), digest_size=8).digest(), "big") & 0x7FFFFFFF

# ------------------------ Main proposer ------------------------

def propose_event(step2algo , trials_sorted, R, rng,
                  step1_A0, step1_hill_frac, step1_drop_frac, step1_candidates, step1_explore_floor, min_spacing, allow_spacing_relax,
                  step2_step_frac, step2_direct_frac, step2_fail_bump_frac, back_to_step1_streak, done_step_mm, done_grad, N4step2,
                  # NEW: mean-shift config passthrough
                  ms_k_nearest: int = 40,
                  ms_q_best_seed: int = 12,
                  ms_wrmsd_eps: float = 0.02,
                  ms_wrmsd_power: float = 2.0,
                  ms_bandwidth_scale: float = 1.3,
                  ms_max_iters: int = 6,
                  ms_tol_mm: float = 0.003,
                  ms_jitter_trials: int = 3,
                  ms_jitter_sigma_frac: float = 0.5,
                  ):
    """
    Return (next_dx_mm, next_dy_mm, reason). If done, returns (None, None, "done_*").
    """
    # Parse trials into successes/failures/tried
    successes_w = []
    failures = []
    tried = []
    for _, dx, dy, idx, wr in trials_sorted:
        tried.append((dx, dy))
        if idx == 1 and _finite(wr):
            successes_w.append((dx, dy, float(wr)))
        elif idx == 0:
            failures.append((dx, dy))
    tried = np.array(tried, float) if tried else np.empty((0,2), float)

    # Gate: only Step-2 if enough signal and not in a miss-streak
    n_succ = len(successes_w)
    if n_succ >= N4step2 and _recent_unindexed_streak(trials_sorted) < back_to_step1_streak:
        # --- Step-2: call the imported mean-shift proposer ---
        if step2algo == "meanshift":
            ms_cfg = Step2MeanShiftConfig(
                k_nearest=ms_k_nearest,
                q_best_for_seed=ms_q_best_seed,
                wrmsd_eps=ms_wrmsd_eps,
                wrmsd_power=ms_wrmsd_power,
                bandwidth_scale=ms_bandwidth_scale,
                max_iters=ms_max_iters,
                tol_mm=ms_tol_mm,
                jitter_trials=ms_jitter_trials,
                jitter_sigma_frac=ms_jitter_sigma_frac,
                stay_inside_R=True,
            )
            ndx, ndy, reason = propose_step2_meanshift(
                successes_w=successes_w,
                failures=failures,
                tried=tried,
                R=R,
                min_spacing_mm=min_spacing,
                rng=rng,
                cfg=ms_cfg,
            )
        elif step2algo == "bayes":  # step2algo == "bayes"
            bayes_cfg = Step2BayesConfig(
                max_train_successes=100,
                y_noise_frac=0.03,
                ell_scale=0.6,
                ell_min=0.15,
                fail_bump_sigma_fracR=0.50,
                fail_bump_amp_frac=0.08,
                gd_steps=80,
                gd_lr_init=0.25,
                gd_backtrack=0.5,
                gd_armijo=1e-4,
                gd_tol_mm=0.0015,
                seed_top_k=8,
                seed_extra_softbest=10,
                stay_inside_R=True,
                spacing_slide_bisect_iters=24,
                spacing_slide_allow_ratio=1.6,
            )
            ndx, ndy, reason = propose_step2_bayes(
                successes_w=successes_w,
                failures=failures,
                tried=tried,
                R=R,
                min_spacing_mm=min_spacing,
                cfg=bayes_cfg,
            )
        elif step2algo == "none":
            ndx, ndy, reason = None, None, "done_step2_disabled"
        return ndx, ndy, reason

    # --- Step-1 exploration (your HillMap module) ---
    first_center = (trials_sorted[0][1], trials_sorted[0][2]) if trials_sorted else (0.0, 0.0)
    s1_trials = [Step1Trial(dx, dy, idx, (float(wr) if _finite(wr) else None)) for _, dx, dy, idx, wr in trials_sorted]
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

# ------------------------ CLI runner ------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Propose next shifts per event (Step-1 imported, Step-2 = mean-shift).")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--radius-mm", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)

    # Step 1 knobs
    ap.add_argument("--step1-A0", type=float, default=1.0)
    ap.add_argument("--step1-hill-amp-frac", type=float, default=0.8)
    ap.add_argument("--step1-drop-amp-frac", type=float, default=0.5)
    ap.add_argument("--step1-candidates", type=int, default=8192)
    ap.add_argument("--step1-explore-floor", type=float, default=1e-6)
    ap.add_argument("--step1-allow-spacing-relax", action="store_true")

    ap.add_argument("--step2-algorithm", type=str, default="none",
                choices=["bayes", "meanshift", "none"], help="'bayes','meanshift' or 'none' to disable optimization after Step 1")

    # (Legacy) Step-2 knobs mean-shift or Bayes
    ap.add_argument("--N4-step2", type=int, default=2)
    ap.add_argument("--step2-step-frac", type=float, default=0.5, help="(unused by mean-shift)")
    ap.add_argument("--step2-direct-frac", type=float, default=0.1, help="(unused by mean-shift)")
    ap.add_argument("--step2-fail-bump-frac", type=float, default=0.01, help="(unused by mean-shift)")
    ap.add_argument("--back-to-step1-streak", type=int, default=5)
    ap.add_argument("--done-step-mm", type=float, default=0.05, help="(unused by mean-shift)")
    ap.add_argument("--done-grad", type=float, default=0.05, help="(unused by mean-shift)")

    ap.add_argument("--min-spacing-mm", type=float, default=0.001)

    # NEW: mean-shift config flags
    ap.add_argument("--ms-k-nearest", type=int, default=40)
    ap.add_argument("--ms-q-best-seed", type=int, default=12)
    ap.add_argument("--ms-wrmsd-eps", type=float, default=0.02)
    ap.add_argument("--ms-wrmsd-power", type=float, default=2.0)
    ap.add_argument("--ms-bandwidth-scale", type=float, default=1.3)
    ap.add_argument("--ms-max-iters", type=int, default=6)
    ap.add_argument("--ms-tol-mm", type=float, default=0.003)
    ap.add_argument("--ms-jitter-trials", type=int, default=3)
    ap.add_argument("--ms-jitter-sigma-frac", type=float, default=0.5)

    args = ap.parse_args(argv)

    log_path = os.path.join(args.run_root, "image_run_log.csv")
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
            if (not s) or s.startswith("#") or s.startswith("run_n"):
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 7:
                parts += [""] * (7 - len(parts))
            try:
                rn = int(parts[0])
            except Exception:
                continue
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
        key_seed = int(hashlib.blake2s(f"{abs_path}::{event_id}::{int(args.seed)}".encode(), digest_size=8).hexdigest(), 16) & 0x7FFFFFFF
        rng = np.random.default_rng(key_seed)

        ndx, ndy, reason = propose_event(
            step2algo=args.step2_algorithm,
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
            N4step2=args.N4_step2,
            ms_k_nearest=int(args.ms_k_nearest),
            ms_q_best_seed=int(args.ms_q_best_seed),
            ms_wrmsd_eps=float(args.ms_wrmsd_eps),
            ms_wrmsd_power=float(args.ms_wrmsd_power),
            ms_bandwidth_scale=float(args.ms_bandwidth_scale),
            ms_max_iters=int(args.ms_max_iters),
            ms_tol_mm=float(args.ms_tol_mm),
            ms_jitter_trials=int(args.ms_jitter_trials),
            ms_jitter_sigma_frac=float(args.ms_jitter_sigma_frac),
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
