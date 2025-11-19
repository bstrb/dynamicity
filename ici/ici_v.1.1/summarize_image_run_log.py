#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_image_run_log.py

Summarize per-run indexing performance from image_run_log.csv and
current proposal/done status from image_run_state.json (sidecar).

Outputs lines like:

  [summary] Runs: first=0, current=2, previous=1
  [summary] Index rate: first=92.308%, previous=95.604%, current=98.901%, Δ(first→curr)=6.593, Δ(prev→curr)=3.297
  [summary] wRMSD mean: first=0.112, previous=0.121, current=0.135, Δ(first→curr)=0.023, Δ(prev→curr)=0.014
  [summary] wRMSD median: ...
  [summary] Proposals: due to unindexed (Hillmap search) = N1
  [summary] Proposals: due to local optimization (CrystFEL Refine or wRMSD-Boltzmann-weighted Hillmap search) = N2
  [summary] Done events: count=Ndone, wRMSD mean=..., median=...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (not math.isfinite(x))):
        return "—"
    return f"{x:.3f}%"


def _fmt_float(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (not math.isfinite(x))):
        return "—"
    return f"{x:.3f}"


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def parse_log(log_path: str):
    """
    Parse image_run_log.csv into per-run statistics.

    Assumes columns:
      run_n, dx_mm, dy_mm, indexed(0/1), wrmsd, next_dx, next_dy
    Comments start with '#'.
    """
    if not os.path.isfile(log_path):
        print(f"[summary] No log at {log_path}", file=sys.stderr)
        return {}, []

    per_run: Dict[int, Dict[str, object]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        header_seen = False
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("#"):
                continue
            if not header_seen:
                # header line
                header_seen = True
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 5:
                continue
            try:
                rn = int(parts[0])
            except Exception:
                continue
            try:
                idx = int(parts[3])
            except Exception:
                idx = 0
            wr = None
            if parts[4] not in ("", "nan", "NaN", "None", "none"):
                try:
                    wv = float(parts[4])
                    if math.isfinite(wv):
                        wr = wv
                except Exception:
                    wr = None

            d = per_run.setdefault(rn, {"rows": 0, "indexed": 0, "wr": []})
            d["rows"] = int(d["rows"]) + 1
            if idx == 1:
                d["indexed"] = int(d["indexed"]) + 1
                if wr is not None:
                    d["wr"].append(wr)

    runs = sorted(per_run.keys())
    return per_run, runs


def metrics_for_run(per_run: Dict[int, Dict[str, object]], rn: int):
    d = per_run.get(rn)
    if not d:
        return None, None, None
    rows = int(d["rows"])
    idx = int(d["indexed"])
    wr_list: List[float] = d["wr"]  # type: ignore

    if rows <= 0:
        rate = None
    else:
        rate = 100.0 * idx / rows

    if not wr_list:
        mean = None
        med = None
    else:
        mean = float(np.mean(wr_list))
        med = float(np.median(wr_list))

    return rate, mean, med


def summarize_state(run_root: str):
    """
    Read sidecar image_run_state.json and compute:
      - number of events currently 'done'
      - mean/median of best wRMSD over done events
      - count of proposals due to unindexed vs local optimization
    """
    state_path = os.path.join(run_root, "image_run_state.json")
    if not os.path.isfile(state_path):
        # Fallback: no sidecar; print zeros but keep layout
        print("[summary] Proposals: due to unindexed (Hillmap search) = 0")
        print(
            "[summary] Proposals: due to local optimization "
            "(CrystFEL Refine or wRMSD-Boltzmann-weighted Hillmap search) = 0"
        )
        print("[summary] Done events: count=0, wRMSD mean=—, median=—")
        return

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        print(f"[summary] WARNING: failed to read {state_path}: {e}", file=sys.stderr)
        print("[summary] Proposals: due to unindexed (Hillmap search) = 0")
        print(
            "[summary] Proposals: due to local optimization "
            "(CrystFEL Refine or wRMSD-Boltzmann-weighted Hillmap search) = 0"
        )
        print("[summary] Done events: count=0, wRMSD mean=—, median=—")
        return

    events = state.get("events", {})
    if not isinstance(events, dict):
        events = {}

    n_prop_unindexed = 0
    n_prop_local = 0
    done_wr: List[float] = []
    done_count = 0

    for ev_key, ev_state in events.items():
        if not isinstance(ev_state, dict):
            continue
        trials = ev_state.get("trials", [])
        if not isinstance(trials, list) or not trials:
            continue

        # Each trial: [run, dx, dy, idx, wr]
        try:
            trials_sorted = sorted(
                [
                    (int(rn), float(dx), float(dy), int(idx), (float(wr) if _finite(wr) else None))
                    for rn, dx, dy, idx, wr in trials
                ],
                key=lambda t: t[0],
            )
        except Exception:
            continue

        latest_status = ev_state.get("latest_status", ["", ""])
        if not isinstance(latest_status, list) or len(latest_status) < 2:
            latest_status = ["", ""]

        is_done = (
            str(latest_status[0]).lower() == "done"
            and str(latest_status[1]).lower() == "done"
        )

        # Collect best wRMSD over successes for this event
        wr_success = [wr for (_rn, _dx, _dy, idx, wr) in trials_sorted if idx == 1 and wr is not None]
        best_wr = min(wr_success) if wr_success else None

        if is_done:
            done_count += 1
            if best_wr is not None and _finite(best_wr):
                done_wr.append(float(best_wr))
            continue

        # Still active → classify proposal reason based on last trial
        last_rn, last_dx, last_dy, last_idx, last_wr = trials_sorted[-1]
        if last_idx == 0 or (last_wr is None):
            n_prop_unindexed += 1
        else:
            n_prop_local += 1

    print(f"[summary] Proposals: due to unindexed (Hillmap search) = {n_prop_unindexed}")
    print(
        "[summary] Proposals: due to local optimization "
        f"(CrystFEL Refine or wRMSD-Boltzmann-weighted Hillmap search) = {n_prop_local}"
    )

    if done_count == 0 or not done_wr:
        print("[summary] Done events: count=0, wRMSD mean=—, median=—")
    else:
        mean_wr = float(np.mean(done_wr))
        med_wr = float(np.median(done_wr))
        print(
            "[summary] Done events: count={cnt}, wRMSD mean={m}, median={md}".format(
                cnt=done_count,
                m=_fmt_float(mean_wr),
                md=_fmt_float(med_wr),
            )
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize image_run_log.csv + image_run_state.json for indexing quality and proposal/done status."
    )
    ap.add_argument("--run-root", required=True, help="Path to runs_YYYYMMDD_HHMMSS folder.")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    log_path = os.path.join(run_root, "image_run_log.csv")

    per_run, runs = parse_log(log_path)
    if not runs:
        print("[summary] No data rows found in image_run_log.csv")
        # still try to show done/proposals from state (if present)
        summarize_state(run_root)
        return 0

    first = runs[0]
    curr = runs[-1]
    prev = runs[-2] if len(runs) >= 2 else None

    # Compute metrics
    rate_first, mean_first, med_first = metrics_for_run(per_run, first)
    rate_curr, mean_curr, med_curr = metrics_for_run(per_run, curr)
    if prev is not None:
        rate_prev, mean_prev, med_prev = metrics_for_run(per_run, prev)
    else:
        rate_prev = mean_prev = med_prev = None

    # Print run indices
    if prev is not None:
        print(f"[summary] Runs: first={first}, current={curr}, previous={prev}")
    else:
        print(f"[summary] Runs: first={first}, current={curr}")

    # Index rates
    if rate_first is None or rate_curr is None:
        d_first_curr = None
    else:
        d_first_curr = rate_curr - rate_first

    if rate_prev is None or rate_curr is None:
        d_prev_curr = None
    else:
        d_prev_curr = rate_curr - rate_prev

    print(
        "[summary] Index rate: first={rf}, previous={rp}, current={rc}, "
        "Δ(first→curr)={df}, Δ(prev→curr)={dp}".format(
            rf=_fmt_pct(rate_first),
            rp=_fmt_pct(rate_prev),
            rc=_fmt_pct(rate_curr),
            df=_fmt_float(d_first_curr),
            dp=_fmt_float(d_prev_curr),
        )
    )

    # wRMSD mean
    if mean_first is None or mean_curr is None:
        d_first_curr_mean = None
    else:
        d_first_curr_mean = mean_curr - mean_first

    if mean_prev is None or mean_curr is None:
        d_prev_curr_mean = None
    else:
        d_prev_curr_mean = mean_curr - mean_prev

    print(
        "[summary] wRMSD mean: first={mf}, previous={mp}, current={mc}, "
        "Δ(first→curr)={df}, Δ(prev→curr)={dp}".format(
            mf=_fmt_float(mean_first),
            mp=_fmt_float(mean_prev),
            mc=_fmt_float(mean_curr),
            df=_fmt_float(d_first_curr_mean),
            dp=_fmt_float(d_prev_curr_mean),
        )
    )

    # wRMSD median
    if med_first is None or med_curr is None:
        d_first_curr_med = None
    else:
        d_first_curr_med = med_curr - med_first

    if med_prev is None or med_curr is None:
        d_prev_curr_med = None
    else:
        d_prev_curr_med = med_curr - med_prev

    print(
        "[summary] wRMSD median: first={mf}, previous={mp}, current={mc}, "
        "Δ(first→curr)={df}, Δ(prev→curr)={dp}".format(
            mf=_fmt_float(med_first),
            mp=_fmt_float(med_prev),
            mc=_fmt_float(med_curr),
            df=_fmt_float(d_first_curr_med),
            dp=_fmt_float(d_prev_curr_med),
        )
    )

    # Proposals + done from sidecar
    summarize_state(run_root)

    return 0

if __name__ == "__main__":
    sys.exit(main())
