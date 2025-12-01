#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propose_next_shifts.py

Implements a two-step proposal logic per event based on a global image_run_log.csv
and a JSON sidecar (image_run_state.json).

Decision table (Option A):

    IF indexed_this_run == 1:
        IF dx/dy is refinable:
            → Step 2 (dxdy refinement)
        ELSE:
            → Step 1 (Hillmap, weighted by previous successes/failures)
    ELSE:
        IF ever_indexed == 0:
            → Step 1 "pure" Hillmap (no positive hills; only exploration + penalties)
        ELSE:
            → Step 1 weighted Hillmap (Boltzmann hills for successes, negative hills for fails)

The CSV schema (one section per image/event):

    run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm,next_reason

where
    - indexed is an *ever-indexed* sticky flag (0/1)
    - wrmsd is per-run wRMSD (blank if not indexed for that run)
    - next_* are filled by this script for the NEXT run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from step1_hillmap_wrmsd import Trial as Step1Trial, Step1Params, propose_step1
from step2_dxdy import Step2DxDyConfig, propose_step2_dxdy


# ------------------------ Small utilities ------------------------


def _fmt6(x: float) -> str:
    return f"{x:.6f}"


def _finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


# ------------------------ CSV helpers ------------------------


def _group_blocks(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Group the log into blocks:
        ("#/abs/path/to/file.h5 event 123", [header_line, csv_line1, csv_line2, ...])
    """
    blocks: List[Tuple[str, List[str]]] = []
    cur_header: Optional[str] = None
    cur: List[str] = []
    for ln in lines:
        if ln.startswith("#/") and " event " in ln:
            if cur:
                blocks.append((cur_header or "", cur))
            cur_header = ln.rstrip("\n")
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append((cur_header or "", cur))
    return blocks

def update_csv_with_proposals(
    log_path: str,
    proposals: Dict[Tuple[str, int], Tuple[str, str, str]],
) -> None:
    """
    Modify the last row for each (h5,event) section in image_run_log.csv and fill:
        next_dx_mm, next_dy_mm, next_reason

    proposals[(abs_h5_path, ev)] = (next_dx, next_dy, reason)
    where next_dx/next_dy are strings (either numeric or "done").

    This version is O(N) in the number of lines in the log. It preserves
    the exact line ordering of the input file.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out: List[str] = []
    section_header_re = re.compile(r"^#(?P<path>/.*)\s+event\s+(?P<ev>\d+)")
    current_key: Optional[Tuple[str, int]] = None
    last_csv_idx: Optional[int] = None  # index in `out` of last CSV line for current section

    def apply_proposal_at_index(idx: Optional[int], key: Optional[Tuple[str, int]]) -> None:
        """
        If idx points at a CSV line for an event that has a proposal,
        patch out[idx] in place.
        """
        if idx is None or key not in proposals:
            return

        line = out[idx]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            parts += [""] * (8 - len(parts))
        else:
            parts = parts[:8]

        ndx, ndy, reason = proposals[key]

        parts[5] = ndx if ndx not in (None, "", "None") else "done"
        parts[6] = ndy if ndy not in (None, "", "None") else "done"
        parts[7] = reason or ""

        out[idx] = ",".join(parts) + "\n"

    for line in lines:
        m = section_header_re.match(line)
        if m:
            # New section → finalize last CSV for previous section
            apply_proposal_at_index(last_csv_idx, current_key)

            h5_path = os.path.abspath(m.group("path"))
            ev = int(m.group("ev"))
            current_key = (h5_path, ev)
            last_csv_idx = None

            out.append(line)
            continue

        # Inside a section: track the last non-empty, non-comment CSV line
        if current_key and not line.startswith("#") and line.strip():
            last_csv_idx = len(out)

        out.append(line)

    # EOF: finalize last section
    apply_proposal_at_index(last_csv_idx, current_key)

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(out)


def _parse_event_header(header_line: str) -> Tuple[str, int]:
    """Return (hdf5_path, event_id) from '#/path event 123'."""
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
    """
    Group into (header, block_lines) and detect latest run_n
    by scanning *all* data lines (not just from the end),
    because the log is grouped by event.
    """
    blocks = _group_blocks(lines)

    latest_run = -1
    for ln in lines:
        s = ln.strip()
        if (not s) or s.startswith("#") or s.startswith("run_n"):
            continue
        try:
            first_field = s.split(",", 1)[0].strip()
            rn = int(first_field)
            if rn > latest_run:
                latest_run = rn
        except Exception:
            continue

    return blocks, latest_run


# ------------------------- Convergence helpers ------------------------


def _recent_unindexed_streak(trials: List[Tuple[int, float, float, int, Optional[float]]]) -> int:
    """
    Count consecutive unindexed trials at the end (wrmsd is None).
    """
    s = 0
    for _, _, _, _, wrmsd in reversed(trials):
        if wrmsd is None:
            s += 1
        else:
            break
    return s


def wrmsd_no_improvement(successes_w, N=5, eps=0.01):
    """
    successes_w: list[(dx, dy, wrmsd)] for indexed trials.

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
    Returns True if the relative std of the last N successful wRMSDs
    is ≤ rel_std_tol. Uses std/mean.
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


def wrmsd_recurring_convergence(successes_w, N=5, tol=0.1):
    """
    Detects convergence when the best wRMSD is repeatedly reached.
    tol: allowed relative difference (e.g., 0.1 = 10%).
    """
    wr = [w for _, _, w in successes_w if w is not None]
    if len(wr) < N:
        return False, {}

    wr_recent = np.array(wr[-N:], float)
    best = np.min(wr_recent)
    rel_improvement = (np.min(wr) - best) / max(best, 1e-9)

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
    Detect convergence when median wRMSD stops changing significantly.
    Compares medians of two consecutive N-length windows; converged if relative change < rel_tol.
    """
    wr = [w for _, _, w in successes_w if w is not None]
    if len(wr) < N * 2:
        return False
    med1 = np.median(wr[-N:])
    med2 = np.median(wr[-2 * N : -N])
    rel = abs(med1 - med2) / max(med2, 1e-9)
    return rel < rel_tol


# ------------------------ Proposal logic ------------------------


def propose_event(
    step2algo: str,
    trials_sorted: List[Tuple[int, float, float, int, Optional[float]]],
    R: float,
    rng: np.random.Generator,
    step1_A0: float,
    step1_hill_frac: float,
    step1_drop_frac: float,
    step1_candidates: int,
    step1_explore_floor: float,
    min_spacing: float,
    allow_spacing_relax: bool,
    # Convergence / done options
    done_on_streak_successes: int = 2,
    done_on_streak_length: int = 5,
    noimprove_N: int = 2,
    noimprove_eps: float = 0.02,
    stability_N: int = 3,
    stability_std: float = 0.05,
    N_conv: int = 3,
    recurring_tol: float = 0.1,
    median_rel_tol: float = 0.1,
    # dxdy refinement
    λ: float = 0.8,
    event_abs_path: str = "",
    beta: float = 10.0,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Return (next_dx_mm, next_dy_mm, reason). If done, returns (None, None, 'done_*').

    trials_sorted: list of (run_n, dx, dy, idx_this_run, wrmsd{float|None})
    """

    if not trials_sorted:
        # Should not happen in practice because we always synthesize a seed trial,
        # but guard anyway.
        return None, None, "done_no_trials"

    # Parse history into successes, failures, tried
    successes_w: List[Tuple[float, float, float]] = []
    failures: List[Tuple[float, float]] = []
    tried: List[Tuple[float, float]] = []

    for _, dx, dy, idx, wr in trials_sorted:
        tried.append((dx, dy))
        if _finite(wr):
            successes_w.append((dx, dy, float(wr)))
        elif wr is None:
            failures.append((dx, dy))

    tried_arr = np.array(tried, float) if tried else np.empty((0, 2), float)

    ever_indexed = bool(successes_w)
    last_wrmsd = trials_sorted[-1][4]
    indexed_this_run = _finite(last_wrmsd)
    s_streak = _recent_unindexed_streak(trials_sorted)

    # ------------ Global "done" criteria (wRMSD-based) ------------

    # 1) Done if we had enough successes and a long unindexed streak
    if ever_indexed and s_streak >= done_on_streak_length and len(successes_w) >= done_on_streak_successes:
        return None, None, f"done_unindexed_streak(k={s_streak}, succ={len(successes_w)})"

    # 2) Done if no improvement in best wRMSD over last N successes
    if wrmsd_no_improvement(successes_w, N=noimprove_N, eps=noimprove_eps):
        return None, None, f"done_no_improve(N={noimprove_N}, eps={noimprove_eps:g})"

    # 3) Done if wRMSD is statistically stable
    if wrmsd_stability(successes_w, N=stability_N, rel_std_tol=stability_std):
        return None, None, f"done_stable_wrmsd(N={stability_N}, relstd≤{stability_std:g})"

    # ------------ Step-2 zone (Option A: only last run indexed) ------------

    if indexed_this_run:
        if step2algo == "dxdy":
            # Attempt dx/dy refinement from per_frame_dx_dy.csv
            dxdy_cfg = Step2DxDyConfig(col_dx="dx", col_dy="dy")
            ndx, ndy, reason_dxdy = propose_step2_dxdy(
                successes_w=successes_w,
                failures=failures,
                tried=tried_arr,
                R=R,
                min_spacing_mm=min_spacing,
                event_dir=event_abs_path,
                cfg=dxdy_cfg,
            )

            # If dx/dy information is missing or invalid, fall back to Step 1 weighted (Case 4)
            if ndx is None or ndy is None:
                # Step-1 relative coordinates (local frame)
                first_center = (trials_sorted[0][1], trials_sorted[0][2])

                # Build trials for Step-1 with an EVER-indexed flag
                ever_flag = 0
                s1_trials = []
                for _, dx, dy, idx, wr in trials_sorted:
                    if _finite(wr):
                        ever_flag = 1  # once indexed, stay indexed for all subsequent Step-1 logic
                    s1_trials.append(
                        Step1Trial(
                            dx - first_center[0],
                            dy - first_center[1],
                            ever_flag,
                            (float(wr) if _finite(wr) else None),
                        )
                    )

                hist_salt = len(trials_sorted)
                rng_seed = (int(rng.integers(0, 2**31 - 1)) ^ (hist_salt * 0x9E3779B1)) & 0x7FFFFFFF

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
                    return None, None, "done_step1_after_dxdy_unavailable"
                x, y = res.proposal_xy_mm
                return float(x), float(y), "step1_weighted_after_dxdy_unavailable"

            # We have a refined dx/dy
            last_dx = trials_sorted[-1][1]
            last_dy = trials_sorted[-1][2]
            prev_dx = trials_sorted[-2][1] if len(trials_sorted) >= 2 else last_dx
            prev_dy = trials_sorted[-2][2] if len(trials_sorted) >= 2 else last_dy

            # Applied center shift in last refinement step
            applied_dx = prev_dx - last_dx
            applied_dy = prev_dy - last_dy
            eps = float(min_spacing)

            # If we've already applied λ * ndx/ndy, check for convergence
            already_applied = (
                len(trials_sorted) >= 2
                and math.hypot(applied_dx - λ * float(ndx), applied_dy - λ * float(ndy)) <= eps
            )

            if already_applied:
                # Check if dxdy refinement itself has converged
                if _finite(last_wrmsd):
                    if len(trials_sorted) >= 2:
                        prev_shift = np.array([prev_dx, prev_dy], float)
                        last_shift = np.array([last_dx, last_dy], float)
                        shift_delta = float(np.linalg.norm(last_shift - prev_shift))

                        conv_median = wrmsd_median_convergence(successes_w, N=N_conv, rel_tol=median_rel_tol)
                        conv_recur, _ = wrmsd_recurring_convergence(successes_w, N=N_conv, tol=recurring_tol)

                        noimp = wrmsd_no_improvement(successes_w, N=noimprove_N, eps=noimprove_eps)
                        stable = wrmsd_stability(successes_w, N=stability_N, rel_std_tol=stability_std)

                        if noimp or stable:
                            return None, None, f"done_dxdy_converged(global: noimp={noimp}, stable={stable})"

                        if shift_delta <= eps or conv_median or conv_recur:
                            return None, None, (
                                f"done_dxdy_converged(shiftΔ={shift_delta:.4g}, "
                                f"hist={conv_median}, recur={conv_recur})"
                            )
                        else:
                            # continue refining around refined center
                            return (last_dx - λ * float(ndx)), (last_dy - λ * float(ndy)), "dxdy_continue_refinement"
                    else:
                        # Only one refinement so far
                        return (last_dx - λ * float(ndx)), (last_dy - λ * float(ndy)), "dxdy_continue_refinement"
                else:
                    # Last attempt after refinement failed (no wRMSD) → fall back to Step 1
                    first_center = (trials_sorted[0][1], trials_sorted[0][2])

                    # Build trials for Step-1 with an EVER-indexed flag
                    ever_flag = 0
                    s1_trials = []
                    for _, dx, dy, idx, wr in trials_sorted:
                        if _finite(wr):
                            ever_flag = 1  # once indexed, stay indexed for all subsequent Step-1 logic
                        s1_trials.append(
                            Step1Trial(
                                dx - first_center[0],
                                dy - first_center[1],
                                ever_flag,
                                (float(wr) if _finite(wr) else None),
                            )
                        )

                    hist_salt = len(trials_sorted)
                    rng_seed = (int(rng.integers(0, 2**31 - 1)) ^ (hist_salt * 0x9E3779B1)) & 0x7FFFFFFF

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

            # Default dxdy refinement step (first or continued)
            return (last_dx - λ * float(ndx)), (last_dy - λ * float(ndy)), f"dxdy_damped_refine(lambda={λ:.2f})"

        elif step2algo == "none":
            # Step 2 disabled, but we have an indexed solution:
            # simplest behavior: mark event done immediately after first success.
            return None, None, "done_step2_disabled"

    # ------------ Step-1 exploration (HillMap) ------------
    first_center = (trials_sorted[0][1], trials_sorted[0][2])

    # Build trials for Step-1 with an EVER-indexed flag
    ever_flag = 0
    s1_trials = []
    for _, dx, dy, idx, wr in trials_sorted:
        if _finite(wr):
            ever_flag = 1  # once indexed, stay indexed for all subsequent Step-1 logic
        s1_trials.append(
            Step1Trial(
                dx - first_center[0],
                dy - first_center[1],
                ever_flag,
                (float(wr) if _finite(wr) else None),
            )
        )

    hist_salt = len(trials_sorted)
    rng_seed = (int(rng.integers(0, 2**31 - 1)) ^ (hist_salt * 0x9E3779B1)) & 0x7FFFFFFF

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


    # Label for diagnostics
    if not ever_indexed:
        mode = "step1_pure_no_index_yet"
    elif not indexed_this_run:
        mode = "step1_weighted_after_index"
    else:
        # last run indexed but we are not in Step-2 zone (e.g. step2algo != 'dxdy')
        mode = "step1_weighted_after_index"

    res = propose_step1(s1_trials, s1_params, beta=beta)
    if res.done:
        return None, None, f"done_{mode}"
    x, y = res.proposal_xy_mm
    return float(x), float(y), mode


# ------------------------ Sidecar state helpers ------------------------


def _default_event_state() -> Dict:
    return {
        "trials": [],              # list of [run, dx, dy, idx_this_run, wr]
        "proposal_history": [],    # list of [run, next_dx, next_dy, reason]
        "latest_status": ["", ""],
        "last_run": -1,
        "ever_indexed": False,     # sticky flag
    }


def load_state(state_path: str) -> Dict:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state not a dict")
        if "events" not in state:
            state["events"] = {}
        if "last_global_run" not in state:
            state["last_global_run"] = -1
        return state
    except FileNotFoundError:
        return {"last_global_run": -1, "events": {}}
    except Exception:
        # If corrupted, start fresh
        return {"last_global_run": -1, "events": {}}


def save_state(state_path: str, state: Dict) -> None:
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)


# ------------------------ CLI runner ------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Propose next shifts per event. Step-1: HillMap. "
            "Step-2: 'dxdy' (one-shot refined center) or 'none'."
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
    ap.add_argument("--done-on-streak-successes", type=int, default=2,
                    help="Require at least this many indexed successes for an event "
                         "before a long unindexed streak can mark it done.")
    ap.add_argument("--done-on-streak-length", type=int, default=5,
                    help="If we have >= --done-on-streak-successes successes and observe this many "
                         "consecutive unindexed results, mark event done.")
    ap.add_argument("--noimprove-N", type=int, default=2,
                    help="Window size (count of successful results) for no-improvement check.")
    ap.add_argument("--noimprove-eps", type=float, default=0.02,
                    help="Minimum relative improvement in best wRMSD over the last --noimprove-N successes.")
    ap.add_argument("--stability-N", type=int, default=3,
                    help="Number of most recent successful results used for wRMSD stability check.")
    ap.add_argument("--stability-std", type=float, default=0.05,
                    help="Relative std threshold for stability: std(wRMSD_recent)/mean(wRMSD_recent) ≤ threshold ⇒ done.")
    ap.add_argument("--N-conv", type=int, default=3,
                    help="Number of recent successful results used for recurring and median convergence checks.")
    ap.add_argument("--recurring-tol", type=float, default=0.1,
                    help="Relative tolerance for recurring best-wRMSD convergence check.")
    ap.add_argument("--median-rel-tol", type=float, default=0.1,
                    help="Relative tolerance for median wRMSD convergence check.")

    # dxdy refinement
    ap.add_argument("--damping-factor", type=float, default=0.8,
                    help="Damping factor λ for dxdy refined shifts; 1.0 = no damping.")

    ap.add_argument(
        "--step2-algorithm",
        type=str,
        default="dxdy",
        choices=["dxdy", "none"],
        help="'dxdy' = apply per_frame_dx_dy.csv; 'none' disables Step-2 and marks event done once indexed.",
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
    # --- Detect column order from header line ---
    header_map: Dict[str, int] = {}
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("run_n,"):
            cols = [c.strip() for c in s.split(",")]
            header_map = {name: i for i, name in enumerate(cols)}
            break

    if not header_map:
        raise RuntimeError("image_run_log.csv missing header line with column names")

    def col(name: str, default: int) -> int:
        """Safe column-index accessor with a fallback."""
        return header_map.get(name, default)

    COL_RUN     = col("run_n", 0)
    COL_DX      = col("det_shift_x_mm", 1)
    COL_DY      = col("det_shift_y_mm", 2)
    COL_INDEXED = col("indexed", 3)
    COL_WRMSD   = col("wrmsd", 4)
    COL_NEXT_DX = col("next_dx_mm", 5)
    COL_NEXT_DY = col("next_dy_mm", 6)
    # COL_REASON could be added if needed; not required here.

    max_col = max(COL_RUN, COL_DX, COL_DY, COL_INDEXED, COL_WRMSD, COL_NEXT_DX, COL_NEXT_DY)

    state_path = os.path.join(args.run_root, "image_run_state.json")
    state = load_state(state_path)

    # If the log got truncated or reset, drop the sidecar and rebuild.
    if state.get("last_global_run", -1) > latest_run:
        state = {"last_global_run": -1, "events": {}}
        
    # NEW: if we've already processed this latest_run once and the log
    # hasn't advanced, skip proposing again to avoid reusing the same run_n.
    if state.get("last_global_run", -1) == latest_run:
        print(f"[propose] latest_run={latest_run} already processed; no new rows in log. Skipping proposals.")
        return 0

    events_state: Dict[str, Dict] = state.setdefault("events", {})

    n_new, n_done = 0, 0

    # Cache per-HDF5 det_shift arrays so we don't re-open files repeatedly
    det_shift_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_seed_shift(h5_path: str, event_id: int) -> Tuple[float, float]:
        """
        Get (det_shift_x_mm, det_shift_y_mm) for an event from the HDF5 file.
        """
        h5_path_abs = os.path.abspath(h5_path)
        if h5_path_abs not in det_shift_cache:
            try:
                with h5py.File(h5_path_abs, "r") as f:
                    dx_arr = np.array(f["/entry/data/det_shift_x_mm"], dtype=float)
                    dy_arr = np.array(f["/entry/data/det_shift_y_mm"], dtype=float)
                det_shift_cache[h5_path_abs] = (dx_arr, dy_arr)
            except KeyError as e:
                raise KeyError(f"Missing det shift datasets in {h5_path_abs}: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed reading det shifts from {h5_path_abs}: {e}")
        dx_arr, dy_arr = det_shift_cache[h5_path_abs]
        return float(dx_arr[event_id]), float(dy_arr[event_id])

    # Process each event block
    for header, block in blocks:
        h5_path, event_id = _parse_event_header(header if header else "#/unknown event -1")
        if event_id < 0:
            continue

        key = f"{os.path.abspath(h5_path)}::{int(event_id)}"
        ev_state = events_state.get(key)
        if ev_state is None:
            ev_state = _default_event_state()
            events_state[key] = ev_state

        ev_state.setdefault("trials", [])
        ev_state.setdefault("latest_status", ["", ""])
        ev_state.setdefault("last_run", -1)
        ev_state.setdefault("proposal_history", [])
        ev_state.setdefault("ever_indexed", False)

        last_run_ev: int = int(ev_state.get("last_run", -1))
        latest_status = ev_state.get("latest_status", ["", ""])

        # Parse *new* rows (rn > last_run_ev) for this event
    
        for ln in block:
            s = ln.strip()
            if (not s) or s.startswith("#") or s.startswith("run_n"):
                continue

            parts = [p.strip() for p in s.split(",")]
            if len(parts) <= max_col:
                parts += [""] * (max_col + 1 - len(parts))

            # --- Use header-derived column indices ---
            try:
                rn = int(parts[COL_RUN])
            except Exception:
                continue
            if rn <= last_run_ev:
                continue

            # detector shifts for this run
            try:
                dx = float(parts[COL_DX])
                dy = float(parts[COL_DY])
            except Exception:
                dx, dy = 0.0, 0.0

            # wrmsd for THIS run (per-run success flag)
            wr: Optional[float]
            try:
                wrs = parts[COL_WRMSD]
                wr = float(wrs) if wrs not in ("", "nan", "NaN", "None", "none", "") else None
            except Exception:
                wr = None

            # per-run indexing flag: THIS run produced a wRMSD or not
            idx_this_run = 1 if (wr is not None and math.isfinite(wr)) else 0

            # sticky ever-indexed from CSV as backup
            idx_ever_csv = 0
            try:
                idxs = parts[COL_INDEXED]
                if idxs not in ("", "nan", "NaN", "None", "none", ""):
                    idx_ever_csv = int(float(idxs))
            except Exception:
                idx_ever_csv = 0

            # store in JSON state
            ev_state["trials"].append([rn, dx, dy, idx_this_run, wr])
            ev_state["last_run"] = rn
            ev_state["latest_status"] = [parts[COL_NEXT_DX], parts[COL_NEXT_DY]]

            ever_prev = bool(ev_state.get("ever_indexed", False))
            ever_from_csv = bool(idx_ever_csv != 0)
            ev_state["ever_indexed"] = bool(ever_prev or idx_this_run == 1 or ever_from_csv)

            last_run_ev = rn
            latest_status = ev_state["latest_status"]


        # If we have no trials at all yet (fresh state), synthesize from HDF5 shifts
        if not ev_state["trials"]:
            seed_dx, seed_dy = get_seed_shift(h5_path, int(event_id))
            ev_state["trials"].append([latest_run, seed_dx, seed_dy, 0, None])
            ev_state["last_run"] = latest_run
            ev_state["latest_status"] = ["", ""]
            last_run_ev = latest_run
            latest_status = ev_state["latest_status"]

        # If already marked done, skip proposing
        nx_raw, ny_raw = latest_status
        if str(nx_raw).lower() == "done" and str(ny_raw).lower() == "done":
            n_done += 1
            continue

        # Sort by run number
        trials_sorted = sorted(
            [
                (int(rn), float(dx), float(dy), int(idx), (float(wr) if wr is not None else None))
                for (rn, dx, dy, idx, wr) in ev_state["trials"]
            ],
            key=lambda t: t[0],
        )

        # Stable per-event RNG
        key_seed = int(
            hashlib.blake2s(f"{h5_path}::{event_id}::{int(args.seed)}".encode(), digest_size=8).hexdigest(),
            16,
        ) & 0x7FFFFFFF
        rng = np.random.default_rng(key_seed)

        # Directory for dxdy: pick the run directory of the last successful run
        succ_runs = [rn for (rn, _dx, _dy, idx, wr) in trials_sorted if _finite(wr)]
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
            done_on_streak_successes=int(args.done_on_streak_successes),
            done_on_streak_length=int(args.done_on_streak_length),
            noimprove_N=int(args.noimprove_N),
            noimprove_eps=float(args.noimprove_eps),
            stability_N=int(args.stability_N),
            stability_std=float(args.stability_std),
            N_conv=int(args.N_conv),
            recurring_tol=float(args.recurring_tol),
            median_rel_tol=float(args.median_rel_tol),
            λ=float(args.damping_factor),
            event_abs_path=event_dir_for_dxdy,
            beta=float(args.beta),
        )

        if ndx is None and ndy is None:
            n_done += 1
            ev_state["latest_status"] = ["done", "done"]
        else:
            n_new += 1
            ev_state["latest_status"] = [_fmt6(float(ndx)), _fmt6(float(ndy))]

        # Record proposal history for this event (proposal belongs to NEXT run)
        run_number = latest_run + 1
        if ndx is None and ndy is None:
            ev_state["proposal_history"].append([run_number, "done", "done", reason])
        else:
            ev_state["proposal_history"].append([run_number, _fmt6(float(ndx)), _fmt6(float(ndy)), reason])

    # Apply proposals into CSV
    proposals_dict: Dict[Tuple[str, int], Tuple[str, str, str]] = {}
    for key, ev_state in state["events"].items():
        if not ev_state["proposal_history"]:
            continue
        run_n, ndx, ndy, reason = ev_state["proposal_history"][-1]
        h5_path, ev_id = key.split("::")
        proposals_dict[(os.path.abspath(h5_path), int(ev_id))] = (ndx, ndy, reason)

    update_csv_with_proposals(log_path, proposals_dict)

    # Save JSON sidecar
    state["last_global_run"] = latest_run
    save_state(state_path, state)

    print(f"[propose] {n_new} new proposals, {n_done} marked done/skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
