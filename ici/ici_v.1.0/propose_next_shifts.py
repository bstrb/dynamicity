#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py — Step-1 (HillMap) exploration; Step-2 can be:
  - dxdy: apply refined det shifts from per_frame_dx_dy.csv of the latest successful run (one-shot reindex at refined center, then done)
  - meanshift: local mean-shift optimization (optional)
  - bayes: Bayesian optimization (optional)
"""
from __future__ import annotations
import argparse, os, sys, math, hashlib
from typing import List, Tuple, Optional
import numpy as np
import h5py

from step1_hillmap_wrmsd import Trial as Step1Trial, Step1Params, propose_step1
from step2_dxdy import propose_step2_dxdy, Step2DxDyConfig

# ------------------------ CSV helpers ------------------------

def _fmt6(x: float) -> str:
    return f"{x:.6f}"

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
    """Returns (hdf5_path, event_id)."""
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
            if (not ln) or ln.startswith("#") or ln.strip().startswith("run_n"):
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
        if (not ln) or ln.startswith("#") or ln.strip().startswith("run_n"):
            out.append(ln); continue
        parts = [p.strip() for p in ln.rstrip("\n").split(",")]
        if len(parts) < 7:
            parts += [""] * (7 - len(parts))
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
        if idx == 0:
            s += 1
        else:
            break
    return s

# ------------------------- Convergence checks ------------------------

def wrmsd_recurring_convergence(successes_w, N=5, tol=0.1):
    """
    Detects convergence when the best wRMSD is repeatedly reached.
    tol: allowed relative difference (e.g., 0.1 = 10%)
    """
    wr = [w for _,_,w in successes_w if w is not None]
    if len(wr) < N:
        return False, {}

    wr_recent = np.array(wr[-N:], float)
    best = np.min(wr_recent)
    rel_improvement = (np.min(wr) - best) / max(best, 1e-9)

    # count how many times we got within tol * best
    near_best = np.sum(wr_recent <= best * (1 + tol))

    converged = (rel_improvement < tol) and (near_best >= 2)
    return converged, {
        "best": best,
        "rel_improvement": rel_improvement,
        "near_best_count": int(near_best),
        "N": N,
    }

def wrmsd_median_convergence(successes_w, N=5, rel_tol=0.05):
    """
    Detects convergence when median wRMSD stops changing significantly.
    Compares medians of two consecutive N-length windows; converged if relative change < rel_tol.
    """
    wr = [w for _,_,w in successes_w if w is not None]
    if len(wr) < N*2:
        return False
    med1 = np.median(wr[-N:])
    med2 = np.median(wr[-2*N:-N])
    rel = abs(med1 - med2) / max(med2, 1e-9)
    return rel < rel_tol

def wrmsd_no_improvement(successes_w, N=5, eps=0.01):
    """
    successes_w: list[(run, dx, wrmsd)] with wrmsd for indexed frames
    Returns True if best wRMSD over the last N successes has not improved
    by at least eps relative to the best before that window.
    """
    wr = [w for _, _, w in successes_w if w is not None]
    if len(wr) < N + 1:
        return False
    prev_best = min(wr[:-N])
    recent_best = min(wr[-N:])
    rel_gain = (prev_best - recent_best) / max(prev_best, 1e-9)
    return rel_gain < eps


def wrmsd_stability(successes_w, N=6, rel_std_tol=0.02):
    """
    Returns True if the relative std of the last N successful wRMSDs is ≤ rel_std_tol.
    Uses relative std = std/mean; requires at least N successes.
    """
    wr = [w for _, _, w in successes_w if w is not None]
    if len(wr) < N:
        return False
    recent = wr[-N:]
    mu = float(np.mean(recent))
    sd = float(np.std(recent))
    if not np.isfinite(mu) or mu <= 0:
        return False
    rel_std = sd / mu
    return rel_std <= rel_std_tol


# ------------------------ Main proposer ------------------------
def propose_event(
    step2algo: str,
    trials_sorted: List[Tuple[int,float,float,int,Optional[float]]],
    R: float,
    rng: np.random.Generator,
    step1_A0: float,
    step1_hill_frac: float,
    step1_drop_frac: float,
    step1_candidates: int,
    step1_explore_floor: float,
    min_spacing: float,
    allow_spacing_relax: bool,
    back_to_step1_streak: int,
    N_conv: int = 3,
    recurring_tol: float = 0.1,
    median_rel_tol: float = 0.05,
    λ = 0.5,  # damping factor for dxdy, if set 1.0 = no damping
    done_on_streak_successes: int = 2,
    done_on_streak_length: int = 6,
    noimprove_N: int = 5,
    noimprove_eps: float = 0.01,
    stability_N: int = 6,
    stability_std: float = 0.02,
    beta: float = 10.0,
    event_abs_path: str = "",
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Return (next_dx_mm, next_dy_mm, reason). If done, returns (None, None, "done_*").
    trials_sorted: list of (run_n, dx, dy, indexed{0/1}, wrmsd{float|None})
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
    tried = np.array(tried, float) if tried else np.empty((0, 2), float)

    # --- NEW: mark done if we already have enough successes AND a long unindexed streak
    s_streak = _recent_unindexed_streak(trials_sorted)
    if len(successes_w) >= done_on_streak_successes and s_streak >= done_on_streak_length:
        return None, None, f"done_unindexed_streak(k={s_streak}, succ={len(successes_w)})"

    # --- NEW: mark done if best wRMSD hasn’t improved by ≥ eps over last N successes
    if wrmsd_no_improvement(successes_w, N=noimprove_N, eps=noimprove_eps):
        return None, None, f"done_no_improve(N={noimprove_N}, eps={noimprove_eps:g})"
    
    # --- NEW: mark done if recent wRMSD is statistically stable (low relative std)
    if wrmsd_stability(successes_w, N=stability_N, rel_std_tol=stability_std):
        return None, None, f"done_stable_wrmsd(N={stability_N}, relstd≤{stability_std:g})"

    # Gate for Step-2
    if s_streak < back_to_step1_streak:

        if step2algo == "dxdy":
            dxdy_cfg = Step2DxDyConfig(col_dx="dx", col_dy="dy")
            ndx, ndy, reason_dxdy = propose_step2_dxdy(
                successes_w=successes_w,
                failures=failures,
                tried=tried,
                R=R,
                min_spacing_mm=min_spacing,
                event_dir=event_abs_path,
                cfg=dxdy_cfg,
            )

            # If CSV missing/invalid -> fall back to Step-1 exploration
            if ndx is None or ndy is None:
                first_center = (trials_sorted[0][1], trials_sorted[0][2])
                s1_trials = [
                    Step1Trial(
                        dx - first_center[0],
                        dy - first_center[1],
                        idx,
                        (float(wr) if _finite(wr) else None),
                    )
                    for _, dx, dy, idx, wr in trials_sorted
                ]
                hist_salt = len(trials_sorted)
                rng_seed = (int(rng.integers(0, 2**31 - 1)) ^ (hist_salt * 0x9E3779B1)) & 0x7fffffff

                s1_params = Step1Params(
                    radius_mm=R,
                    rng_seed=rng_seed,
                    n_candidates=int(step1_candidates),
                    A0=float(step1_A0),
                    hill_amp_frac=float(step1_hill_frac),
                    drop_amp_frac=float(step1_drop_frac),
                    explore_floor=float(step1_explore_floor),
                    min_spacing_mm=float(min_spacing),
                    first_attempt_center_mm=first_center,
                    allow_spacing_relax=bool(allow_spacing_relax),
                )
                res = propose_step1(s1_trials, s1_params, beta=beta)
                if res.done:
                    return None, None, "done_step1_fallback"
                x, y = res.proposal_xy_mm
                return float(x), float(y), "step1_fallback_due_to_dxdy_error"

            # Pull last / previous tried centers (unchanged)
            last_dx  = trials_sorted[-1][1] if trials_sorted else 0.0
            last_dy  = trials_sorted[-1][2] if trials_sorted else 0.0
            last_idx = trials_sorted[-1][3] if trials_sorted else 0
            prev_dx  = trials_sorted[-2][1] if len(trials_sorted) >= 2 else 0.0
            prev_dy  = trials_sorted[-2][2] if len(trials_sorted) >= 2 else 0.0

            # Was the refined one-shot already applied on the last run?
            # In subtract mode, applied center = prev − refined.
            # Use a tolerance that survives 6-decimal rounding in the log.
            eps = float(min_spacing)

            # Compare the shift we actually applied to λ * ndx/ndy
            applied_dx = prev_dx - last_dx
            applied_dy = prev_dy - last_dy
            already_applied = (
                len(trials_sorted) >= 2
                and math.hypot(applied_dx - λ * float(ndx), applied_dy - λ * float(ndy)) <= eps
            )

            if already_applied:
                if last_idx == 1:
                    # --- New: check convergence of reindexing shifts using run history ---
                    # Need at least 2 runs to have two applied refinements to compare
                    if len(trials_sorted) >= 2:
                        # Extract the last two *applied* shifts (i.e. centers we actually used)
                        prev_shift = np.array([trials_sorted[-2][1], trials_sorted[-2][2]], float)
                        last_shift = np.array([trials_sorted[-1][1], trials_sorted[-1][2]], float)
                        shift_delta = np.linalg.norm(last_shift - prev_shift)

                        conv_median = wrmsd_median_convergence(successes_w, N=N_conv, rel_tol=median_rel_tol)
                        conv_recur, _ = wrmsd_recurring_convergence(successes_w, N=N_conv, tol=recurring_tol)

                        # Also apply the global convergence checks here
                        # Also apply the global convergence checks here
                        noimp = wrmsd_no_improvement(successes_w, N=noimprove_N, eps=noimprove_eps)
                        stable = wrmsd_stability(successes_w, N=stability_N, rel_std_tol=stability_std)
                        if noimp or stable:
                            return None, None, (
                                f"done_dxdy_converged(global: noimp={noimp}, stable={stable})"
                            )


                        if shift_delta <= eps or conv_median or conv_recur:
                            return None, None, (
                                f"done_dxdy_converged(shiftΔ={shift_delta:.4g}, "
                                f"hist={conv_median}, recur={conv_recur})"
                            )
                        else:
                            # Not yet converged: keep refining at new refined center
                            return (last_dx - λ * float(ndx)), (last_dy - λ * float(ndy)), "dxdy_continue_refinement"
                    else:
                        # Only one refinement so far; allow at least one more iteration
                        return (last_dx - λ * float(ndx)), (last_dy - λ * float(ndy)), "dxdy_continue_refinement"

                else:
                    # Refined attempt failed => back to Step-1 exploration
                    first_center = (trials_sorted[0][1], trials_sorted[0][2])

                    s1_trials = [
                        Step1Trial(
                            dx - first_center[0],
                            dy - first_center[1],
                            idx,
                            (float(wr) if _finite(wr) else None),
                        )
                        for _, dx, dy, idx, wr in trials_sorted
                    ]
                    hist_salt = len(trials_sorted)
                    rng_seed = (int(rng.integers(0, 2**31 - 1)) ^ (hist_salt * 0x9E3779B1)) & 0x7fffffff

                    s1_params = Step1Params(
                        radius_mm=R,
                        rng_seed=rng_seed,
                        n_candidates=int(step1_candidates),
                        A0=float(step1_A0),
                        hill_amp_frac=float(step1_hill_frac),
                        drop_amp_frac=float(step1_drop_frac),
                        explore_floor=float(step1_explore_floor),
                        min_spacing_mm=float(min_spacing),
                        first_attempt_center_mm=first_center,
                        allow_spacing_relax=bool(allow_spacing_relax),
                    )
                    res = propose_step1(s1_trials, s1_params, beta=beta)
                    if res.done:
                        return None, None, "done_after_dxdy_fail_step1"
                    x, y = res.proposal_xy_mm
                    return float(x), float(y), "step1_after_dxdy_fail"
                
            return (last_dx - λ * float(ndx)), (last_dy - λ * float(ndy)), f"dxdy_damped_refine(lambda={λ:.2f})"

        elif step2algo == "none":
            return None, None, "done_step2_disabled"

    # --- Step-1 exploration (HillMap) ---
    first_center = (trials_sorted[0][1], trials_sorted[0][2])
    # Convert prior trials to the LOCAL frame (relative to first_center)
    s1_trials = [
        Step1Trial(
            dx - first_center[0],
            dy - first_center[1],
            idx,
            (float(wr) if _finite(wr) else None),
        )
        for _, dx, dy, idx, wr in trials_sorted
    ]
    hist_salt = len(trials_sorted)
    rng_seed = (int(rng.integers(0, 2**31 - 1)) ^ (hist_salt * 0x9E3779B1)) & 0x7fffffff

    s1_params = Step1Params(
        radius_mm=R,
        rng_seed=rng_seed,
        n_candidates=int(step1_candidates),
        A0=float(step1_A0),
        hill_amp_frac=float(step1_hill_frac),
        drop_amp_frac=float(step1_drop_frac),
        explore_floor=float(step1_explore_floor),
        min_spacing_mm=float(min_spacing),
        first_attempt_center_mm=first_center,   # seed stays absolute
        allow_spacing_relax=bool(allow_spacing_relax),
    )

    res = propose_step1(s1_trials, s1_params, beta=beta)
    x, y = res.proposal_xy_mm      # still absolute (because step1 adds seed internally)
    if res.done:
        return None, None, res.reason
    else:
        x, y = res.proposal_xy_mm
        return float(x), float(y), res.reason


# ------------------------ CLI runner ------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Propose next shifts per event. Step-1: HillMap. "
            "Step-2: 'dxdy' (one-shot refined center), 'meanshift', 'bayes', or 'none'."
        )
    )
    # Required arguments
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--radius-mm", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-spacing-mm", type=float, default=0.0005)

    # Hillmap search knobs
    ap.add_argument("--step1-A0", type=float, default=2.0)
    ap.add_argument("--step1-hill-amp-frac", type=float, default=5.0)
    ap.add_argument("--step1-drop-amp-frac", type=float, default=0.1)
    ap.add_argument("--step1-candidates", type=int, default=8192)
    ap.add_argument("--step1-explore-floor", type=float, default=1e-5)
    ap.add_argument("--step1-allow-spacing-relax", action="store_true")
    ap.add_argument("--beta", type=float, default=10.0)

    # Convergence / done criteria
    ap.add_argument("--back-to-step1-streak", type=int, default=1,
                    help="If we have this many or more consecutive unindexed results, go back to Step-1 exploration.")
    ap.add_argument("--N-conv", type=int, default=3,
                    help="Number of recent successful results used for recurring and median convergence checks.")
    ap.add_argument("--recurring-tol", type=float, default=0.1,
                    help="Relative tolerance for recurring best-wRMSD convergence check.")
    ap.add_argument("--median-rel-tol", type=float, default=0.1,
                    help="Relative tolerance for median wRMSD convergence check.")
    ap.add_argument("--damping-factor", type=float, default=0.8,
                    help="Damping factor λ for dxdy refined shifts; 1.0 = no damping.")
    ap.add_argument("--done-on-streak-successes", type=int, default=2, help="Require at least this many indexed successes for an event before a long unindexed streak can mark it done.")
    ap.add_argument("--done-on-streak-length", type=int, default=5, help="If we have >= --done-on-streak-successes successes and observe this many consecutive unindexed results, mark event done.")
    ap.add_argument("--noimprove-N", type=int, default=2,
                    help="Window size (count of successful results) for no-improvement check.")
    ap.add_argument("--noimprove-eps", type=float, default=0.02, help="Minimum relative improvement (e.g., 0.05 = 5%) in best wRMSD over the last --noimprove-N successes; otherwise mark done.")
    ap.add_argument("--stability-N", type=int, default=3,
                    help="Number of most recent successful results used for wRMSD stability check.")
    ap.add_argument("--stability-std", type=float, default=0.05,
                    help="Relative std threshold for stability: std(wRMSD_recent)/mean(wRMSD_recent) ≤ threshold ⇒ done.")


    ap.add_argument(
        "--step2-algorithm",
        type=str,
        default="dxdy",
        choices=["dxdy", "none"],
        help="'dxdy' = apply per_frame_dx_dy.csv once; 'none' disables Step-2 and marks event done once indexed.",
    )

    args = ap.parse_args(argv)

    log_path = os.path.join(args.run_root, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print(f"ERROR: not found: {log_path}", file=sys.stderr)
        return 2

    lines = read_log(log_path)
    blocks, latest_run = parse_blocks(lines)
    if latest_run < 0:
        print("ERROR: Could not determine latest run_n in CSV.", file=sys.stderr)
        return 2

    new_lines: List[str] = []
    n_new, n_done = 0, 0

    for header, block in blocks:
        h5_path, event_id = _parse_event_header(header if header else "#/unknown event -1")
        # Skip non-event blocks (preamble or malformed)
        if event_id < 0:
            new_lines.extend(block)  # pass through unchanged
            continue

        # Parse rows of the block into (rn, dx, dy, idx, wr)
        rows: List[Tuple[int, float, float, int, Optional[float]]] = []
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
            wr = None
            try:
                wrs = parts[4]
                wr = float(wrs) if wrs not in ("", "nan", "None", "none") else None
            except Exception:
                wr = None
            rows.append((rn, dx, dy, idx, wr))

        trials_sorted = sorted(rows, key=lambda t: t[0])
        # --- Ensure the event has a first attempted center in the log ---
        # If the block has no rows yet for this event, synthesize the first trial
        # from the per-frame det shifts in the source HDF5 (never (0,0)).
        if not trials_sorted:
            try:
                with h5py.File(h5_path, "r") as f:
                    seed_dx = float(f["/entry/data/det_shift_x_mm"][int(event_id)])
                    seed_dy = float(f["/entry/data/det_shift_y_mm"][int(event_id)])
            except KeyError as e:
                raise KeyError(f"Missing det shift datasets in {h5_path}: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed reading seed for {h5_path} event {event_id}: {e}")

            # Append one data line to this block so trials_sorted[0] exists.
            # Columns: run_n, dx, dy, idx, wrmsd, next_dx, next_dy
            first_line = f"{latest_run},{seed_dx:.6f},{seed_dy:.6f},0,,,\n"
            block.append(first_line)
            rows.append((latest_run, seed_dx, seed_dy, 0, None))
            trials_sorted = sorted(rows, key=lambda t: t[0])


        # Stable per-event RNG (unchanged)
        key_seed = int(
            hashlib.blake2s(f"{h5_path}::{event_id}::{int(args.seed)}".encode(), digest_size=8).hexdigest(),
            16,
        ) & 0x7FFFFFFF
        rng = np.random.default_rng(key_seed)

        # Resolve the event directory for dxdy from the latest SUCCESSFUL run of this event
        succ_runs = [rn for (rn, _dx, _dy, idx, wr) in trials_sorted if idx == 1 and (wr is not None)]
        event_dir_for_dxdy = ""
        if succ_runs:
            event_last_rn = trials_sorted[-1][0]
            prev_succs = [rn for rn in succ_runs if rn < event_last_rn]
            use_rn = max(prev_succs) if prev_succs else max(succ_runs)
            event_dir_for_dxdy = os.path.join(
                args.run_root, f"run_{use_rn:03d}", f"event_{int(event_id):06d}"
            )

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
            back_to_step1_streak=int(args.back_to_step1_streak),
            N_conv=int(args.N_conv),
            recurring_tol=args.recurring_tol,
            median_rel_tol=args.median_rel_tol,
            λ=args.damping_factor,
            done_on_streak_successes=int(args.done_on_streak_successes),
            done_on_streak_length=int(args.done_on_streak_length),
            noimprove_N=int(args.noimprove_N),
            noimprove_eps=float(args.noimprove_eps),
            stability_N=int(args.stability_N),
            stability_std=float(args.stability_std),
            beta=float(args.beta),
            # IMPORTANT: for dxdy this must be the actual event dir of the success run
            event_abs_path=event_dir_for_dxdy,
        )

        updated = update_block_latest_run(block_lines=block, latest_run=latest_run, new_dx=ndx, new_dy=ndy)
        if ndx is None and ndy is None:
            n_done += 1
        else:
            n_new += 1
        new_lines.extend(updated)

    write_log(log_path, new_lines)
    print(f"[propose] {n_new} new proposals, {n_done} marked done")
    # print(f"[propose] Updated {log_path} for run_{latest_run:03d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
