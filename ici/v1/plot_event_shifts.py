
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_event_shifts.py

Parse an image_run_log.csv (with interleaved "#<abs_h5_path> event <N>" headers)
and plot all tried (dx,dy) shifts for a selected (image,event). If wRMSD values
are present, color points by wRMSD and highlight the best (finite) wRMSD.

Example usage:
  python plot_event_shifts.py --log /path/to/image_run_log.csv --event 67 --image-substr sim.h5 --outdir ./plots
  python plot_event_shifts.py --log image_run_log.csv --event 67 --list-keys

Notes:
- The CSV body rows are expected to be:
    run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
- The script groups history by (real_h5_path, event). Use --list-keys to see what exists.
- If multiple (path,event) groups match the given --event and optional --image-substr,
  all of them will be plotted (one PNG per group) unless --first-only is provided.
"""

import argparse
import math
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

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
            dx = _flt(parts[1])
            dy = _flt(parts[2])
            indexed = int(parts[3]) if parts[3].isdigit() else (1 if parts[3] == "1" else 0)
            wrmsd = _flt(parts[4])
            ndx = None if parts[5] in ("", "done") else _flt(parts[5])
            ndy = None if parts[6] in ("", "done") else _flt(parts[6])
            if dx is None or dy is None:
                continue
            groups[cur].append((run_n, dx, dy, indexed, wrmsd, ndx, ndy))
    # sort rows by run_n within each group
    for k in list(groups.keys()):
        groups[k].sort(key=lambda r: r[0])
    return groups

def pick_groups(groups, event: int, image_substr: Optional[str]) -> List[Tuple[str,int]]:
    keys = []
    for (path, ev) in groups.keys():
        if ev != event:
            continue
        if image_substr and (image_substr.lower() not in path.lower()):
            continue
        keys.append((path, ev))
    return keys

# def plot_group(path: str, event: int, rows: List[Row], outdir: str, dpi: int = 150, annotate: bool = True) -> str:
#     xs = [r[1] for r in rows]
#     ys = [r[2] for r in rows]
#     runs = [r[0] for r in rows]
#     wrs = [r[4] for r in rows]
#     idx = [r[3] for r in rows]

#     # Prepare color mapping for wRMSD: only pass finite values; others will be plotted separately
#     finite_mask = [ (w is not None) for w in wrs ]
#     xs_f = [x for x, m in zip(xs, finite_mask) if m]
#     ys_f = [y for y, m in zip(ys, finite_mask) if m]
#     wrs_f = [w for w in wrs if w is not None]

#     xs_nf = [x for x, m in zip(xs, finite_mask) if not m]
#     ys_nf = [y for y, m in zip(ys, finite_mask) if not m]

#     plt.figure(figsize=(6,6))

#     # Draw tried positions in order
#     plt.plot(xs, ys, marker="o")  # default style, connects in order

#     # Overlay: color by wRMSD for finite values with a colorbar
#     sc = None
#     if wrs_f:
#         sc = plt.scatter(xs_f, ys_f, c=wrs_f)  # default colormap, default style
#         cb = plt.colorbar(sc)
#         cb.set_label("wRMSD")

#         # Highlight best (finite) wRMSD
#         best_idx = min(range(len(wrs)), key=lambda i: float("inf") if wrs[i] is None else wrs[i])
#         plt.scatter([xs[best_idx]], [ys[best_idx]], marker="*", s=160)  # larger star

#     # Overlay: non-finite wrmsd (if any)
#     if xs_nf:
#         plt.scatter(xs_nf, ys_nf, marker="x")  # default style

#     # Annotate run numbers to see sequence
#     if annotate:
#         for x, y, rn in zip(xs, ys, runs):
#             plt.text(x, y, str(rn), fontsize=8, ha="left", va="bottom")

#     # Axes and aspect
#     plt.axhline(0, linewidth=0.8)
#     plt.axvline(0, linewidth=0.8)
#     plt.gca().set_aspect("equal", adjustable="box")
#     plt.xlabel("det_shift_x_mm")
#     plt.ylabel("det_shift_y_mm")
#     title = f"{os.path.basename(path)} — event {event}"
#     plt.title(title)
#     plt.tight_layout()

#     # Save
#     base = os.path.splitext(os.path.basename(path))[0]
#     out_png = os.path.join(outdir, f"{base}_event_{event}_shifts.png")
#     os.makedirs(outdir, exist_ok=True)
#     plt.savefig(out_png, dpi=dpi)
#     plt.close()
#     return out_png
def plot_group(path: str, event: int, rows: List[Row], outdir: str, dpi: int = 150, annotate: bool = True) -> str:
    # Unpack columns
    xs_abs = [r[1] for r in rows]
    ys_abs = [r[2] for r in rows]
    runs   = [r[0] for r in rows]
    wrs    = [r[4] for r in rows]
    idx    = [r[3] for r in rows]

    # 1) Recenter at the FIRST run point instead of (0,0)
    x0, y0 = xs_abs[0], ys_abs[0]
    xs = [x - x0 for x in xs_abs]
    ys = [y - y0 for y in ys_abs]

    # Masks and splits
    finite_mask = [(w is not None) for w in wrs]
    xs_f = [x for x, m in zip(xs, finite_mask) if m]
    ys_f = [y for y, m in zip(ys, finite_mask) if m]
    wrs_f = [w for w in wrs if w is not None]

    xs_nf = [x for x, m in zip(xs, finite_mask) if not m]
    ys_nf = [y for y, m in zip(ys, finite_mask) if not m]

    # Successfully indexed + finite-wRMSD for the heatmap
    ok_mask = [(i == 1 and (w is not None)) for i, w in zip(idx, wrs)]
    xs_ok = [x for x, m in zip(xs, ok_mask) if m]
    ys_ok = [y for y, m in zip(ys, ok_mask) if m]
    wrs_ok = [w for w, m in zip(wrs, ok_mask) if m]

    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    plt.figure(figsize=(6, 6))

    # 2) Plot the progression (always), centered at first run
    #    Connect in order so the search path is clear
    plt.plot(xs, ys, marker="o", linewidth=1.0)

    # Mark the first run point explicitly at (0,0)
    plt.scatter([0.0], [0.0], marker="s", s=60, facecolors="none", edgecolors="black", zorder=5)
    plt.text(0.0, 0.0, " start", fontsize=8, ha="left", va="bottom")

    # 3) Heatmap (tricontourf) of wRMSD for successfully indexed points
    #    (transparent overlay so the path and points remain visible)
    if len(wrs_ok) >= 3:
        tri = Triangulation(xs_ok, ys_ok)
        # Use multiple levels; alpha so it doesn't overpower the path
        cs = plt.tricontourf(tri, wrs_ok, levels=12, alpha=0.35)
        cb_hm = plt.colorbar(cs)
        cb_hm.set_label("wRMSD (indexed only)")

    # 4) Scatter color for all finite wRMSD points (keeps your original coloring)
    sc = None
    if wrs_f:
        sc = plt.scatter(xs_f, ys_f, c=wrs_f, zorder=6)
        # Only add a second colorbar if we didn't already add the heatmap one
        if len(wrs_ok) < 3:
            cb = plt.colorbar(sc)
            cb.set_label("wRMSD")

        # Highlight best finite wRMSD, preferring indexed points
        best_candidates = [i for i, (w, ii) in enumerate(zip(wrs, idx)) if (w is not None and ii == 1)]
        if not best_candidates:
            best_candidates = [i for i, w in enumerate(wrs) if w is not None]
        best_idx = min(best_candidates, key=lambda i: wrs[i])
        plt.scatter([xs[best_idx]], [ys[best_idx]], marker="*", s=160, zorder=7)

    # 5) Non-finite wRMSD, if any
    if xs_nf:
        plt.scatter(xs_nf, ys_nf, marker="x", zorder=6)

    # 6) Annotate run numbers (optional)
    if annotate:
        for x, y, rn in zip(xs, ys, runs):
            plt.text(x, y, str(rn), fontsize=8, ha="left", va="bottom")

    # Axes / aspect
    # Crosshairs now show the *first-run-centered* axes
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("det_shift_x_mm (centered at first run)")
    plt.ylabel("det_shift_y_mm (centered at first run)")
    title = f"{os.path.basename(path)} — event {event} (centered at first run)"
    plt.title(title)
    plt.tight_layout()

    # Save
    base = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(outdir, f"{base}_event_{event}_shifts_centered.png")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    return out_png

def main():
    ap = argparse.ArgumentParser(description="Plot dx,dy shifts per (image,event) from image_run_log.csv, with optional wRMSD coloring and best highlight.")
    ap.add_argument("--log", required=True, help="Path to image_run_log.csv")
    ap.add_argument("--event", type=int, required=True, help="Event number to plot")
    ap.add_argument("--image-substr", type=str, default=None, help="Substring filter for the image path (case-insensitive)")
    ap.add_argument("--outdir", type=str, default=".", help="Directory to save PNG(s)")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--first_only", action="store_true", help="If multiple (image,event) groups match, only plot the first")
    ap.add_argument("--list_keys", action="store_true", help="List available (image,event) keys and exit")
    ap.add_argument("--no_annotate", action="store_true", help="Do not annotate run numbers next to points")
    args = ap.parse_args()

    log_path = _abs(args.log)
    groups = parse_image_run_log(log_path)

    if args.list_keys:
        print("# Available (image,event) keys:")
        for (path, ev) in sorted(groups.keys(), key=lambda k: (k[1], k[0])):
            print(f"{ev}\t{path}")
        return 0

    keys = pick_groups(groups, args.event, args.image_substr)
    if not keys:
        print("No matching (image,event) groups. Use --list-keys to inspect available entries.")
        return 2

    plotted = []
    for k in keys:
        rows = groups[k]
        out_png = plot_group(k[0], k[1], rows, args.outdir, dpi=args.dpi, annotate=(not args.no_annotate))
        print(f"Wrote: {out_png}")
        plotted.append(out_png)
        if args.first_only:
            break

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
