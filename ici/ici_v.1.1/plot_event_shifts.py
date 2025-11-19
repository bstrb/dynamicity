#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_event_shifts.py

Parse an image_run_log.csv (with interleaved "#<abs_h5_path> event <N>" headers)
and plot tried (dx,dy) shifts per (image,event). If wRMSD values are present,
color points by wRMSD and highlight the best (finite) wRMSD.

Examples:
  # single event
  python plot_event_shifts.py --log image_run_log.csv --event 67 --outdir ./plots

  # list keys
  python plot_event_shifts.py --log image_run_log.csv --list-keys

  # all events (optionally filter by filename substring) + mark true center
  python plot_event_shifts.py --log image_run_log.csv --all-events --image-substr sim1.h5 --true-center -0.028 -0.028 --outdir ./plots

  # all events but skip those without any finite wRMSD
  python plot_event_shifts.py --log image_run_log.csv --all-events --skip-no-wrmsd
"""
import argparse
import math
import os
import re
from typing import Dict, List, Tuple, Optional

# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.tri import Triangulation

Row = Tuple[int, float, float, int, Optional[float], Optional[float], Optional[float]]  # run, dx, dy, indexed, wrmsd, next_dx, next_dy

def _abs(s: str) -> str:
    return os.path.abspath(os.path.expanduser(s.strip()))

def _flt(s: str) -> Optional[float]:
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def parse_image_run_log(log_path: str) -> Dict[Tuple[str, int], List[Row]]:
    groups: Dict[Tuple[str,int], List[Row]] = {}
    cur: Optional[Tuple[str,int]] = None
    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("#"):
                m = re.match(r"#(?P<path>/.+?)\s+event\s+(?P<ev>\d+)\s*$", ln)
                if m:
                    cur = (_abs(m.group("path")), int(m.group("ev")))
                    groups.setdefault(cur, [])
                else:
                    cur = None
                continue
            if cur is None or ln.startswith("run_n,"):
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 7:
                continue
            run_n = int(parts[0]) if parts[0].isdigit() else int(re.sub(r"\D+","", parts[0]) or 0)
            dx = _flt(parts[1]); dy = _flt(parts[2])
            indexed = int(parts[3]) if parts[3].isdigit() else (1 if parts[3] == "1" else 0)
            wrmsd = _flt(parts[4])
            ndx = None if parts[5] in ("", "done") else _flt(parts[5])
            ndy = None if parts[6] in ("", "done") else _flt(parts[6])
            if dx is None or dy is None:
                continue
            groups[cur].append((run_n, dx, dy, indexed, wrmsd, ndx, ndy))
    # sort by run within each group
    for k in list(groups.keys()):
        groups[k].sort(key=lambda r: r[0])
    return groups

def iter_groups_for_args(groups, event: Optional[int], image_substr: Optional[str], all_events: bool):
    keys = []
    if all_events:
        for (path, ev) in groups.keys():
            if image_substr and (image_substr.lower() not in path.lower()):
                continue
            keys.append((path, ev))
    else:
        for (path, ev) in groups.keys():
            if (event is not None) and (ev != event):
                continue
            if image_substr and (image_substr.lower() not in path.lower()):
                continue
            keys.append((path, ev))
    keys.sort(key=lambda k: (k[1], k[0]))
    return keys

def plot_group(path: str,
               event: int,
               rows: List[Row],
               outdir: str,
               dpi: int = 150,
               annotate: bool = True,
               true_center: Optional[Tuple[float,float]] = None,
               event_only_title: bool = False) -> str:
    xs_abs = [r[1] for r in rows]
    ys_abs = [r[2] for r in rows]
    runs   = [r[0] for r in rows]
    wrs    = [r[4] for r in rows]
    idx    = [r[3] for r in rows]

    # center at first run
    x0, y0 = xs_abs[0], ys_abs[0]
    xs = [x - x0 for x in xs_abs]
    ys = [y - y0 for y in ys_abs]

    finite_mask = [(w is not None) for w in wrs]
    xs_f = [x for x, m in zip(xs, finite_mask) if m]
    ys_f = [y for y, m in zip(ys, finite_mask) if m]
    wrs_f = [w for w in wrs if w is not None]

    xs_nf = [x for x, m in zip(xs, finite_mask) if not m]
    ys_nf = [y for y, m in zip(ys, finite_mask) if not m]

    ok_mask = [(ii == 1 and (w is not None)) for ii, w in zip(idx, wrs)]
    xs_ok = [x for x, m in zip(xs, ok_mask) if m]
    ys_ok = [y for y, m in zip(ys, ok_mask) if m]
    wrs_ok = [w for w, m in zip(wrs, ok_mask) if m]

    plt.figure(figsize=(6, 6))
    # path
    plt.plot(xs, ys, marker="o", linewidth=1.0)
    plt.scatter([0.0], [0.0], marker="s", s=60, facecolors="none", edgecolors="black", zorder=5)
    plt.text(0.0, 0.0, " start", fontsize=8, ha="left", va="bottom")

    # 3) Heatmap (tricontourf) of wRMSD for successfully indexed points
    def _has_area(xs, ys, eps=1e-6):
        if len(xs) < 3:
            return False
        X = np.column_stack([xs, ys]) - np.mean([xs, ys], axis=1)
        # rank-2 in 2D means non-collinear
        s = np.linalg.svd(X, compute_uv=False)
        return (len(s) >= 2) and (s[1] > eps)

    if len(wrs_ok) >= 3 and _has_area(xs_ok, ys_ok, eps=1e-6):
        try:
            tri = Triangulation(xs_ok, ys_ok)
            cs = plt.tricontourf(tri, wrs_ok, levels=12, alpha=0.35)
            cb_hm = plt.colorbar(cs)
            cb_hm.set_label("wRMSD (indexed only)")
        except Exception:
            # Degenerate geometry or qhull still unhappy — skip heatmap.
            pass
    # else: not enough area → skip heatmap gracefully

    # colored points by wRMSD
    if wrs_f:
        sc = plt.scatter(xs_f, ys_f, c=wrs_f, zorder=6)
        if len(wrs_ok) < 3:
            cb = plt.colorbar(sc)
            cb.set_label("wRMSD")
        # highlight best (prefer indexed)
        best_cand = [i for i,(w,ii) in enumerate(zip(wrs, idx)) if (w is not None and ii == 1)]
        if not best_cand:
            best_cand = [i for i,w in enumerate(wrs) if w is not None]
        bi = min(best_cand, key=lambda i: wrs[i])
        plt.scatter([xs[bi]], [ys[bi]], marker="*", s=160, zorder=7)

    if xs_nf:
        plt.scatter(xs_nf, ys_nf, marker="x", zorder=6)

    if annotate:
        for x, y, rn in zip(xs, ys, runs):
            plt.text(x, y, str(rn), fontsize=8, ha="left", va="bottom")

    # optional true center marker (convert absolute to centered coords)
    if true_center is not None:
        tcx, tcy = true_center
        plt.scatter([tcx - x0], [tcy - y0], marker="+", s=100, linewidths=1.5, zorder=7)
        plt.text(tcx - x0, tcy - y0, " true", fontsize=8, ha="left", va="bottom")

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("det_shift_x_mm (centered at first run)")
    plt.ylabel("det_shift_y_mm (centered at first run)")

    # --- Title behavior ---
    if event_only_title:
        title = f"event {event}"
    else:
        title = f"{os.path.basename(path)} — event {event} (centered at first run)"
    plt.title(title)

    # Fixed limits
    plt.xlim(-0.05, 0.05)
    plt.ylim(-0.05, 0.05)

    plt.tight_layout()

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"{base}_event_{event}_shifts_centered.png")
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser(description="Plot dx,dy shifts per (image,event) from image_run_log.csv, with optional wRMSD coloring and best highlight.")
    ap.add_argument("--log", required=True, help="Path to image_run_log.csv")
    ap.add_argument("--event", type=int, default=None, help="Event number to plot. If omitted with --all-events, plots all.")
    ap.add_argument("--all-events", action="store_true", help="Plot ALL events in the log (optionally filtered by --image-substr).")
    ap.add_argument("--image-substr", type=str, default=None, help="Substring filter for the image path (case-insensitive)")
    ap.add_argument("--outdir", type=str, default=None, help="Directory to save PNG(s). Default: '<log_dir>/plots_event_shifts'")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--first_only", action="store_true", help="If multiple (image,event) groups match, only plot the first")
    ap.add_argument("--list_keys", action="store_true", help="List available (image,event) keys and exit")
    ap.add_argument("--no_annotate", action="store_true", help="Do not annotate run numbers next to points")
    ap.add_argument("--true-center", type=float, nargs=2, metavar=("CX","CY"),
                    help="Optional marker (mm) for known true center (e.g. --true-center -0.028 -0.028)")
    ap.add_argument("--skip-out-of-limits", action="store_true",
                    help="If set, skip plotting any (image,event) groups where all points lie outside the fixed x/y limits (-0.05 to 0.05 mm).")
    ap.add_argument("--skip-no-wrmsd", action="store_true",
                    help="If set, skip plotting any (image,event) groups with no finite wRMSD values.")
    args = ap.parse_args()

    groups = parse_image_run_log(_abs(args.log))
    
    # Derive default outdir if not provided
    if args.outdir is None:
        log_dir = os.path.dirname(os.path.abspath(args.log))
        args.outdir = os.path.join(log_dir, "plots_event_shifts")
    os.makedirs(args.outdir, exist_ok=True)

    if args.list_keys:
        print("# Available (image,event) keys:")
        for (path, ev) in sorted(groups.keys(), key=lambda k: (k[1], k[0])):
            print(f"{ev}\t{path}")
        return 0

    keys = iter_groups_for_args(groups, args.event, args.image_substr, args.all_events)
    if not keys:
        if args.all_events:
            print("No events matched filters (try removing --image-substr).")
        else:
            print("No matching (image,event) groups. Use --list-keys to inspect available entries, or add --all-events.")
        return 2

    plotted = []
    for (p, ev) in keys:
        rows = groups[(p, ev)]

        # Optional skip if all points outside fixed x/y limits
        if args.skip_out_of_limits:
            xs = [r[1] for r in rows]
            ys = [r[2] for r in rows]
            if all((abs(x - xs[0]) > 0.05 or abs(y - ys[0]) > 0.05) for x, y in zip(xs, ys)):
                print(f"Skipped (all points outside x/y limits): event {ev}\t{p}")
                if args.first_only:
                    break
                continue

        # Optional skip if no finite wRMSD present
        if args.skip_no_wrmsd:
            has_wrmsd = any((w is not None) for (_, _, _, _, w, _, _) in rows)
            if not has_wrmsd:
                print(f"Skipped (no wRMSD): event {ev}\t{p}")
                if args.first_only:
                    break
                continue

        out_png = plot_group(
            p, ev, rows, args.outdir, dpi=args.dpi,
            annotate=(not args.no_annotate),
            true_center=(tuple(args.true_center) if args.true_center else None),
            event_only_title=args.all_events  # <-- event-only title when plotting all events
        )
        print(f"Wrote: {out_png}")
        plotted.append(out_png)
        if args.first_only:
            break

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
