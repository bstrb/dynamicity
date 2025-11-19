#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_image_run_log.py

Reads runs/image_run_log.csv (your grouped log with event headers) and prints:
- first run and current indexed percent, plus increase from first and from previous run
- first run and current mean/median wRMSD, plus change from first and from previous run
- number of proposed next steps due to unindexed frames (ring) and due to local optimization (BO)
- number of done events and their internal mean/median wRMSD

Optional: write a CSV/JSON summary artifact.

Usage:
  python3 summarize_image_run_log.py --run-root <root> [--out-csv summary.csv] [--out-json summary.json]
"""
import argparse, os, math, json, statistics as stats
from collections import defaultdict


def parse_log(log_path):
    rows = []
    current_event = None
    with open(log_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("#/"):
                # header line: "#/abs/path event <num>"
                try:
                    left, evs = ln[1:].rsplit(" event ", 1)
                    src = os.path.abspath(left.strip())
                    ev  = int(evs.strip())
                    current_event = (src, ev)
                except Exception:
                    current_event = None
                continue

            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 7 or not parts[0].isdigit():
                continue

            run = int(parts[0])

            def _f(s):
                try:
                    return float(s) if s != "" else None
                except Exception:
                    return None

            det_dx = _f(parts[1]); det_dy = _f(parts[2])
            indexed = int(parts[3]) if parts[3] else 0
            wr = _f(parts[4])
            next_dx_raw, next_dy_raw = parts[5], parts[6]

            def _f_next(s):
                if s in ("", "done"):
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            next_dx = _f_next(next_dx_raw)
            next_dy = _f_next(next_dy_raw)

            src, ev = (current_event if current_event else (None, None))
            rows.append({
                "src": src,
                "event": ev,
                "sec": (src, ev),  # <-- unique section key
                "run": run,
                "det_dx": det_dx, "det_dy": det_dy,
                "indexed": indexed, "wrmsd": wr,
                "next_dx": next_dx, "next_dy": next_dy,
                "next_dx_raw": next_dx_raw, "next_dy_raw": next_dy_raw
            })
    return rows


def mean_safe(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return (sum(vals) / len(vals)) if vals else None


def median_safe(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return (stats.median(vals) if vals else None)


def group_rows_by_section(rows):
    """
    Returns:
      by_sec: {sec: [rows sorted by run]}
      runs:   sorted unique run numbers
    """
    by_sec = defaultdict(list)
    runs = set()
    for r in rows:
        by_sec[r["sec"]].append(r)
        runs.add(r["run"])
    for sec in by_sec:
        by_sec[sec].sort(key=lambda x: x["run"])
    return by_sec, sorted(runs)


def run_stats_cumulative_grouped(by_sec, run_cutoff: int):
    """
    Cumulative-as-of-run stats using pre-grouped rows:
      - events_present: sections that have appeared up to run_cutoff
      - success_count: sections with ≥1 success up to run_cutoff
      - wrmsd_mean/median: mean/median of per-section BEST (min) wRMSD up to run_cutoff
    """
    upto_secs = []
    best_wr_by_sec = {}

    for sec, g in by_sec.items():
        seen = False
        best_wr = None
        for r in g:
            if r["run"] > run_cutoff:
                break
            seen = True
            if r["indexed"] == 1 and (r["wrmsd"] is not None) and not math.isnan(r["wrmsd"]):
                if best_wr is None or r["wrmsd"] < best_wr:
                    best_wr = r["wrmsd"]

        if seen:
            upto_secs.append(sec)
            if best_wr is not None:
                best_wr_by_sec[sec] = best_wr

    events_present = len(upto_secs)
    success_count = len(best_wr_by_sec)
    idx_pct = (success_count / events_present * 100.0) if events_present else None

    wr_values = list(best_wr_by_sec.values())
    wr_mean = (sum(wr_values) / len(wr_values)) if wr_values else None
    wr_median = (stats.median(wr_values) if wr_values else None)

    return {
        "events_present": events_present,
        "success_count": success_count,
        "index_percent": idx_pct,
        "wrmsd_mean": wr_mean,
        "wrmsd_median": wr_median,
    }


def proposal_breakdown_current_grouped(by_sec, run_cutoff: int):
    """
    Returns (ring_props, bo_props) counting SECTIONS (src,event):
      - If section is done at/before run_cutoff -> not counted
      - Else if section has any success up to run_cutoff -> counts toward BO
      - Else -> counts toward ring
    """
    ring_props = 0
    bo_props = 0

    for sec, g in by_sec.items():
        # g is sorted by run; consider only rows up to cutoff
        g_upto = [r for r in g if r["run"] <= run_cutoff]
        if not g_upto:
            continue

        last = g_upto[-1]
        is_done = (last["next_dx_raw"] == "done") and (last["next_dy_raw"] == "done")
        if is_done:
            continue

        has_success = any(
            (row["indexed"] == 1) and (row["wrmsd"] is not None) and (not math.isnan(row["wrmsd"]))
            for row in g_upto
        )

        if has_success:
            bo_props += 1
        else:
            ring_props += 1

    return ring_props, bo_props


def done_events_summary_grouped(by_sec):
    done_wrmsd = []

    for sec, g in by_sec.items():
        last = g[-1]
        if last["next_dx_raw"] == "done" and last["next_dy_raw"] == "done":
            wrs = [
                r["wrmsd"] for r in g
                if r["indexed"] == 1 and r["wrmsd"] is not None and not math.isnan(r["wrmsd"])
            ]
            if wrs:
                done_wrmsd.append(min(wrs))

    n_done = len(done_wrmsd)
    return n_done, mean_safe(done_wrmsd), median_safe(done_wrmsd)


def fmt(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" + ("%" if pct else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Experiment root that contains runs/image_run_log.csv")
    ap.add_argument("--log", default=None, help="Optional explicit path to image_run_log.csv")
    ap.add_argument("--out-csv", default=None, help="Optional path to write a one-row summary CSV")
    ap.add_argument("--out-json", default=None, help="Optional path to write a JSON summary")
    args = ap.parse_args()

    runs_dir = os.path.abspath(os.path.expanduser(args.run_root))
    log_path = args.log or os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print(f"[summary] ERROR: missing {log_path}")
        return 2

    rows = parse_log(log_path)
    if not rows:
        print(f"[summary] ERROR: no data rows in {log_path}")
        return 2

    by_sec, all_runs = group_rows_by_section(rows)
    first_run = all_runs[0]
    current_run = all_runs[-1]
    prev_run = all_runs[-2] if len(all_runs) > 1 else None

    first = run_stats_cumulative_grouped(by_sec, first_run)
    curr  = run_stats_cumulative_grouped(by_sec, current_run)
    prev  = run_stats_cumulative_grouped(by_sec, prev_run) if prev_run is not None else None

    # deltas
    def delta(a, b):
        if a is None or b is None or any(isinstance(v, float) and math.isnan(v) for v in (a, b)):
            return None
        return b - a

    idx_inc_first_to_curr = delta(first["index_percent"], curr["index_percent"])
    idx_inc_prev_to_curr  = delta(prev["index_percent"],  curr["index_percent"]) if prev else None

    wr_mean_delta_first_to_curr   = delta(first["wrmsd_mean"],   curr["wrmsd_mean"])
    wr_mean_delta_prev_to_curr    = delta(prev["wrmsd_mean"],    curr["wrmsd_mean"]) if prev else None
    wr_median_delta_first_to_curr = delta(first["wrmsd_median"], curr["wrmsd_median"])
    wr_median_delta_prev_to_curr  = delta(prev["wrmsd_median"],  curr["wrmsd_median"]) if prev else None

    ring_props, bo_props = proposal_breakdown_current_grouped(by_sec, current_run)
    n_done, wr_done_mean, wr_done_median = done_events_summary_grouped(by_sec)

    summary = {
        "first_run": first_run,
        "current_run": current_run,
        "previous_run": prev_run,
        "index_percent_first": first["index_percent"],
        "index_percent_current": curr["index_percent"],
        "delta_index_pct_first_to_current": idx_inc_first_to_curr,
        "delta_index_pct_prev_to_current": idx_inc_prev_to_curr,
        "wrmsd_mean_first": first["wrmsd_mean"],
        "wrmsd_mean_current": curr["wrmsd_mean"],
        "delta_wrmsd_mean_first_to_current": wr_mean_delta_first_to_curr,
        "delta_wrmsd_mean_prev_to_current": wr_mean_delta_prev_to_curr,
        "wrmsd_median_first": first["wrmsd_median"],
        "wrmsd_median_current": curr["wrmsd_median"],
        "delta_wrmsd_median_first_to_current": wr_median_delta_first_to_curr,
        "delta_wrmsd_median_prev_to_current": wr_median_delta_prev_to_curr,
        "proposals_unindexed_ring": ring_props,
        "proposals_local_bo": bo_props,
        "done_events_count": n_done,
        "done_events_wrmsd_mean": wr_done_mean,
        "done_events_wrmsd_median": wr_done_median,
    }

    # pretty print
    print("[summary] Runs: first={}, current={}{}".format(
        first_run, current_run, f", previous={prev_run}" if prev_run is not None else ""))

    print("[summary] Index rate: first={}, previous={}, current={}, Δ(first→curr)={}, Δ(prev→curr)={}".format(
        fmt(first["index_percent"], pct=True),
        fmt(prev["index_percent"], pct=True) if prev_run is not None else "—",
        fmt(curr["index_percent"], pct=True),
        fmt(idx_inc_first_to_curr, pct=False),
        fmt(idx_inc_prev_to_curr, pct=False) if prev_run is not None else "—",
    ))

    print("[summary] wRMSD mean: first={}, previous={}, current={}, Δ(first→curr)={}, Δ(prev→curr)={}".format(
        fmt(first["wrmsd_mean"]),
        fmt(prev["wrmsd_mean"]) if prev_run is not None else "—",
        fmt(curr["wrmsd_mean"]),
        fmt(wr_mean_delta_first_to_curr),
        fmt(wr_mean_delta_prev_to_curr) if prev_run is not None else "—",
    ))

    print("[summary] wRMSD median: first={}, previous={}, current={}, Δ(first→curr)={}, Δ(prev→curr)={}".format(
        fmt(first["wrmsd_median"]),
        fmt(prev["wrmsd_median"]) if prev_run is not None else "—",
        fmt(curr["wrmsd_median"]),
        fmt(wr_median_delta_first_to_curr),
        fmt(wr_median_delta_prev_to_curr) if prev_run is not None else "—",
    ))

    print("[summary] Proposals: due to unindexed (Hillmap search) = {}".format(
        ring_props))
    print("[summary] Proposals: due to local optimization (CrystFEL Refine or wRMSD-Boltzmann-weighted Hillmap search) = {}".format(
        bo_props))

    print("[summary] Done events: count={}, wRMSD mean={}, median={}".format(
        n_done, fmt(wr_done_mean), fmt(wr_done_median)))

    # optional artifacts
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader(); w.writerow(summary)
        print(f"[summary] wrote {args.out_csv}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[summary] wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
