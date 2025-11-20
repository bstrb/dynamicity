#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_image_run_log.py

Per-run summary of an iterative indexing/refinement pipeline.

For each run R in image_run_log.csv, prints:

  * Index rate of the FIRST run (using only trials with run_n == first_run)
  * Cumulative index rate up to PREVIOUS run (using best result per event)
  * Cumulative index rate up to CURRENT run (using best result per event)

  * wRMSD mean/median of FIRST run (best result per event at run_n == first_run)
  * Cumulative wRMSD mean/median up to PREVIOUS run (best result per event)
  * Cumulative wRMSD mean/median up to CURRENT run (best result per event)

  * Number of DONE events at this run, fraction of total frames,
    and their internal wRMSD mean/median (best result per done event)

  * Number of proposals issued at this run due to:
      - never-indexed events, and
      - refinement / Boltzmann search map (events that had indexed before)
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import re
import statistics
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable


SECTION_RE = re.compile(r"^#(?P<path>/.+?)\s+event\s+(?P<ev>\d+)\s*$")

@dataclass
class Trial:
    run: int
    indexed: bool
    wrmsd: Optional[float]
    next_dx: str
    next_dy: str

EventsDict = Dict[Tuple[str, int], List[Trial]]

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_sidecar(run_root: str):
    state_path = os.path.join(run_root, "image_run_state.json")
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return {}

    events = state.get("events", {})
    sidecar = {}

    for key, ev_state in events.items():
        try:
            h5, ev = key.split("::", 1)
            ev = int(ev)
        except Exception:
            continue

        latest_status = ev_state.get("latest_status", ["",""])
        last_run = int(ev_state.get("last_run", -1))
        proposal_history = ev_state.get("proposal_history", [])

        sidecar[(os.path.abspath(h5), ev)] = {
            "latest_status": latest_status,
            "last_run": last_run,
            "proposal_history": proposal_history,
        }

    return sidecar


def parse_log_rows(path: str) -> Tuple[EventsDict, List[int]]:
    """
    Parse grouped image_run_log.csv into an events dict:

        events[(h5_path, event_id)] = [Trial(...), ...]

    Uses section headers of the form:
        #/abs/path/to/file.h5 event 123
    to identify which image/event the following rows belong to.
    """
    events: EventsDict = {}
    runs_seen = set()

    current_key: Optional[Tuple[str, int]] = None
    warned_orphan_rows = False

    with open(path, "r", encoding="utf-8") as f:
        # Read header line (we don't actually need the names here)
        header = f.readline()

        for line in f:
            if not line.strip():
                continue

            if line.startswith("#"):
                m = SECTION_RE.match(line)
                if m:
                    h5_path = os.path.abspath(m.group("path").strip())
                    ev = int(m.group("ev"))
                    current_key = (h5_path, ev)
                # other comment lines are ignored
                continue

            # Data row
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                parts += [""] * (7 - len(parts))

            run_str, dx_str, dy_str, idx_str, wr_str, ndx_str, ndy_str = parts[:7]

            try:
                run = int(run_str)
            except ValueError:
                # malformed run number, skip
                continue

            runs_seen.add(run)

            if current_key is None:
                # Data rows without a section header – skip, but warn once.
                if not warned_orphan_rows:
                    print(
                        "[summary] Warning: data row(s) found before any "
                        "'#/path event N' header; skipping those rows.",
                        file=sys.stderr,
                    )
                    warned_orphan_rows = True
                continue

            indexed = (idx_str == "1")

            wrmsd: Optional[float]
            if wr_str in ("", "nan", "NaN", "None"):
                wrmsd = None
            else:
                try:
                    wr_val = float(wr_str)
                    wrmsd = wr_val if math.isfinite(wr_val) else None
                except ValueError:
                    wrmsd = None

            trial = Trial(
                run=run,
                indexed=indexed,
                wrmsd=wrmsd,
                next_dx=ndx_str,
                next_dy=ndy_str,
            )

            events.setdefault(current_key, []).append(trial)

    # Sort trials for each event by run number
    for key in events:
        events[key].sort(key=lambda t: t.run)

    runs_sorted = sorted(runs_seen)
    return events, runs_sorted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _successes(trials: Iterable[Trial]) -> List[Trial]:
    """Filter trials that are successful (indexed and finite wRMSD)."""
    out: List[Trial] = []
    for t in trials:
        if not t.indexed:
            continue
        if t.wrmsd is None:
            continue
        if not math.isfinite(t.wrmsd):
            continue
        out.append(t)
    return out


def _best_wrmsd(trials: Iterable[Trial]) -> Optional[float]:
    """Return minimum wRMSD among successful trials, or None if no successes."""
    succ = _successes(trials)
    if not succ:
        return None
    return min(t.wrmsd for t in succ if t.wrmsd is not None)


def _fmt_float(x: Optional[float], ndigits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{ndigits}f}"


def _fmt_frac(num: Optional[int], den: Optional[int]) -> str:
    if num is None or den is None or den == 0:
        return "n/a"
    return f"{num}/{den} ({100.0 * num / den:.1f}%)"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def first_run_stats(events: EventsDict, first_run: int):
    """
    Stats using only trials with run == first_run.

    Returns:
        (index_rate, wrmsd_mean, wrmsd_median)
    """
    n_events_with_trial = 0
    n_indexed_events = 0
    wrmsds: List[float] = []

    for _key, trials in events.items():
        subset = [t for t in trials if t.run == first_run]
        if not subset:
            continue

        n_events_with_trial += 1

        best = _best_wrmsd(subset)
        if best is not None:
            n_indexed_events += 1
            wrmsds.append(best)

    if n_events_with_trial == 0:
        idx_rate = None
    else:
        idx_rate = n_indexed_events / n_events_with_trial

    if wrmsds:
        wr_mean = statistics.fmean(wrmsds)
        wr_median = statistics.median(wrmsds)
    else:
        wr_mean = wr_median = None

    return idx_rate, wr_mean, wr_median

def cumulative_stats(
    events: EventsDict,
    upto_run: int,
    total_events_global: int,
    sidecar=None,
):
    """
    Cumulative stats using all trials with run <= upto_run.

    Returns:
        (
          idx_rate, wr_mean, wr_median,
          done_count, done_fraction, done_wr_mean, done_wr_median
        )
    """
    n_events_with_trials = 0
    n_indexed_events = 0
    wrmsds_best: List[float] = []

    done_count = 0
    done_wrmsds: List[float] = []

    for key, trials in events.items():
        subset = [t for t in trials if t.run <= upto_run]
        if not subset:
            continue

        n_events_with_trials += 1

        # best wRMSD per event so far
        best = _best_wrmsd(subset)
        if best is not None:
            n_indexed_events += 1
            wrmsds_best.append(best)

        # ---------------- DONE DETECTION ----------------
        is_done = False

        # 1) Prefer sidecar
        if sidecar is not None and key in sidecar:
            info = sidecar[key]
            latest_status = info["latest_status"]
            last_run = info["last_run"]

            sx = (latest_status[0] or "").lower()
            sy = (latest_status[1] or "").lower()
            if last_run <= upto_run and sx == "done" and sy == "done":
                is_done = True

        # 2) Fallback to CSV (almost never used, but safe)
        if not is_done:
            last_trial = subset[-1]
            ndx = (last_trial.next_dx or "").strip().lower()
            ndy = (last_trial.next_dy or "").strip().lower()
            if ndx == "done" and ndy == "done":
                is_done = True

        if is_done:
            done_count += 1
            if best is not None:
                done_wrmsds.append(best)

    # index rate
    if n_events_with_trials == 0:
        idx_rate = None
    else:
        idx_rate = n_indexed_events / n_events_with_trials

    # done fraction
    if total_events_global > 0:
        done_fraction = done_count / total_events_global
    else:
        done_fraction = None

    # wrmsd
    if wrmsds_best:
        wr_mean = statistics.fmean(wrmsds_best)
        wr_median = statistics.median(wrmsds_best)
    else:
        wr_mean = wr_median = None

    # wrmsd for done events
    if done_wrmsds:
        done_wr_mean = statistics.fmean(done_wrmsds)
        done_wr_median = statistics.median(done_wrmsds)
    else:
        done_wr_mean = done_wr_median = None

    return (
        idx_rate,
        wr_mean,
        wr_median,
        done_count,
        done_fraction,
        done_wr_mean,
        done_wr_median,
    )
def proposal_counts_for_run(events: EventsDict, run: int, sidecar):
    """
    Count proposals for a given run, based on the *final* proposal recorded per event.
    Only the last proposal with rn == run is meaningful.

    Classification:
        - reason.startswith("step1") → Hillmap / unindexed
        - reason.startswith("dxdy")  → refinement / Boltzmann
        - reason.startswith("done")  → not a proposal
    """

    ring = 0
    refine = 0

    for key in events:
        if key not in sidecar:
            continue

        ph = sidecar[key].get("proposal_history", [])
        # Filter proposals for THIS run
        entries = [e for e in ph if e[0] == run]

        if not entries:
            continue

        # Only the LAST entry matters
        (_, ndx, ndy, reason) = entries[-1]
        reason = (reason or "").lower()

        if reason.startswith("done"):
            # Not a proposal
            continue

        if reason.startswith("step1"):
            ring += 1
        elif reason.startswith("dxdy"):
            refine += 1
        else:
            # Safe fallback: count as refine
            refine += 1

    return ring, refine

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize image_run_log.csv per run."
    )
    ap.add_argument(
        "--run-root",
        required=True,
        help="Root directory containing image_run_log.csv",
    )
    args = ap.parse_args(argv)

    log_path = os.path.join(args.run_root, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print(f"[summary] No log file found: {log_path}")
        return 0

    events, runs = parse_log_rows(log_path)
    if not runs:
        print("[summary] No data rows found.")
        return 0

    total_events_global = len(events)
    first_run = runs[0]

    sidecar = load_sidecar(args.run_root)

    # First-run stats (constant w.r.t. run)
    first_idx, first_wr_mean, first_wr_median = first_run_stats(events, first_run)

    # Per-run summaries
    run = runs[-1]
    i = len(runs) - 1

    prev_run = runs[i - 1] if i > 0 else None

    # Cumulative stats up to previous run
    if prev_run is not None:
        (
            prev_idx,
            prev_wr_mean,
            prev_wr_median,
            prev_done_count,
            prev_done_fraction,
            prev_done_wr_mean,
            prev_done_wr_median,
        ) = cumulative_stats(events, prev_run, total_events_global, sidecar=sidecar)
        
    else:
        prev_idx = prev_wr_mean = prev_wr_median = None
        prev_done_count = None
        prev_done_fraction = None
        prev_done_wr_mean = prev_done_wr_median = None

    # Cumulative stats up to current run
    (
        curr_idx,
        curr_wr_mean,
        curr_wr_median,
        curr_done_count,
        curr_done_fraction,
        curr_done_wr_mean,
        curr_done_wr_median,
    ) = cumulative_stats(events, run, total_events_global, sidecar=sidecar)
    
    # Proposals for this run
    prop_never, prop_refine = proposal_counts_for_run(events, run+1, sidecar)
    
    # --- Compact, Column-Style Summary -------------------------------------

    print("[summary] Runs: first={}, current={}{}".format(
        f"{first_run:03d}",
        f"{run:03d}",
        f", previous={prev_run:03d}" if prev_run is not None else ""
    ))

    # helper for percentage formatting
    def _fmt_pct(x):
        if x is None:
            return "—"
        return f"{x*100:.2f}%"

    # deltas
    def _delta(a, b):
        if a is None or b is None:
            return "—"
        d = b - a
        return f"{d:+.4f}"

    print("[summary] Index rate:  first={}, previous={}, current={},  Δ(first→curr)={}, Δ(prev→curr)={}".format(
        _fmt_pct(first_idx),
        _fmt_pct(prev_idx),
        _fmt_pct(curr_idx),
        _delta(first_idx, curr_idx),
        _delta(prev_idx, curr_idx),
    ))

    print("[summary] wRMSD mean:   first={}, previous={}, current={},  Δ(first→curr)={}, Δ(prev→curr)={}".format(
        _fmt_float(first_wr_mean),
        _fmt_float(prev_wr_mean),
        _fmt_float(curr_wr_mean),
        _delta(first_wr_mean, curr_wr_mean),
        _delta(prev_wr_mean, curr_wr_mean),
    ))

    print("[summary] wRMSD median: first={}, previous={}, current={},  Δ(first→curr)={}, Δ(prev→curr)={}".format(
        _fmt_float(first_wr_median),
        _fmt_float(prev_wr_median),
        _fmt_float(curr_wr_median),
        _delta(first_wr_median, curr_wr_median),
        _delta(prev_wr_median, curr_wr_median),
    ))

    print("[summary] Proposals: unindexed(Hillmap)={}, refine/Boltzmann={}".format(
        prop_never, prop_refine
    ))

    print("[summary] Done events: {}/{} ({:.1f}%),  wRMSD mean={}, median={}".format(
        curr_done_count,
        total_events_global,
        curr_done_fraction * 100 if curr_done_fraction is not None else 0.0,
        _fmt_float(curr_done_wr_mean),
        _fmt_float(curr_done_wr_median),
    ))

    return 0

if __name__ == "__main__":
    sys.exit(main())
