#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_next_run_from_log.py
Create a new run folder (lst + overlays + sh) **from runs/image_run_log.csv**.

Default scope:
- Use ONLY entries whose latest row in the log has run_n == max(run_n in log).
  (Pass --scope all to include all keys' latest rows.)

Rerun policy (log-only, no extra state needed):
- Include an image/event if in its latest log row:
  * indexed == 0 (still unindexed), OR
  * the "next_dx_mm,next_dy_mm" differ from the last "det_shift_x_mm,det_shift_y_mm"
    by > 1e-12 (i.e., there is a concrete next proposal to try).

Overlays' shifts are taken from the latest log row's **next_dx_mm,next_dy_mm**.
"""
from __future__ import annotations
import os, sys, csv, math, re, shlex, argparse
from typing import Dict, Tuple, List

import h5py  # just to validate ELINK target existence

DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
IMAGES_DS = "/entry/data/images"

def resolve_real_source(h5_path: str) -> str:
    return os.path.abspath(h5_path)

def create_overlay(src_h5: str, overlay_path: str) -> int:
    """Create an overlay with ExternalLink to images and det_shift datasets (float64, shape (N,))."""
    import h5py, os
    from pathlib import Path
    src_h5 = os.path.abspath(src_h5)
    overlay_path = os.path.abspath(overlay_path)
    Path(os.path.dirname(overlay_path)).mkdir(parents=True, exist_ok=True)
    if os.path.exists(overlay_path):
        os.remove(overlay_path)
    with h5py.File(src_h5, "r") as src, h5py.File(overlay_path, "w") as ov:
        g_entry = ov.require_group("/entry")
        g_data = g_entry.require_group("data")
        # External link to images
        g_data["images"] = h5py.ExternalLink(src_h5, IMAGES_DS)
        N = src[IMAGES_DS].shape[0]
        g_data.create_dataset("det_shift_x_mm", shape=(N,), dtype="f8")
        g_data.create_dataset("det_shift_y_mm", shape=(N,), dtype="f8")
    return 0

def write_shifts_mm(h5_overlay_path: str, indices: List[int], dx_mm: List[float], dy_mm: List[float]) -> None:
    import h5py, numpy as np
    with h5py.File(h5_overlay_path, "r+") as ov:
        x = ov["/entry/data/det_shift_x_mm"]; y = ov["/entry/data/det_shift_y_mm"]
        idx = np.asarray(indices, dtype=int)
        x[idx] = np.asarray(dx_mm, dtype=float)
        y[idx] = np.asarray(dy_mm, dtype=float)

def _latest_entries_from_global_log(log_path: str, scope: str="last") -> Dict[Tuple[str,int], Dict[str,str]]:
    latest: Dict[Tuple[str,int], Dict[str,str]] = {}
    if not os.path.exists(log_path):
        return latest
    # first pass: find max run_n if scope == 'last'
    max_run = -1
    if scope == "last":
        cur_key = None
        with open(log_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                if ln.startswith("#/"):
                    try:
                        path_part, ev_part = ln[1:].rsplit(" event ", 1)
                        cur_key = (os.path.abspath(path_part.strip()), int(ev_part.strip()))
                    except Exception:
                        cur_key = None
                    continue
                if ln.startswith("run_n,") or cur_key is None:
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 7: continue
                try:
                    r = int(parts[0]); 
                    if r > max_run: max_run = r
                except Exception:
                    pass
    # second pass: collect latest row per key (respecting scope)
    cur_key = None
    with open(log_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if ln.startswith("#/"):
                try:
                    path_part, ev_part = ln[1:].rsplit(" event ", 1)
                    cur_key = (os.path.abspath(path_part.strip()), int(ev_part.strip()))
                except Exception:
                    cur_key = None
                continue
            if ln.startswith("run_n,") or cur_key is None:
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 7: continue
            run_n = parts[0]
            if scope == "last" and (run_n != str(max_run)):
                continue
            latest[cur_key] = {
                "run_n": parts[0],
                "dx": parts[1],
                "dy": parts[2],
                "indexed": parts[3],
                "wrmsd": parts[4],
                "next_dx": parts[5],
                "next_dy": parts[6],
            }
    return latest

def _should_rerun(row: Dict[str,str]) -> bool:
    try:
        indexed = int(row.get("indexed","0") or 0)
    except Exception:
        indexed = 0
    try:
        dx = float(row.get("dx","") or "nan")
        dy = float(row.get("dy","") or "nan")
        ndx = float(row.get("next_dx","") or "nan")
        ndy = float(row.get("next_dy","") or "nan")
    except Exception:
        return False
    # Policy: rerun if unindexed OR we have a concrete next that differs
    if indexed == 0:
        return True
    # next differs more than 1e-12
    if (math.isfinite(ndx) and math.isfinite(ndy)) and (abs(ndx - dx) > 1e-12 or abs(ndy - dy) > 1e-12):
        return True
    return False

def prepare_next_from_log(run_root: str, scope: str="last") -> str:
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.exists(log_path):
        raise SystemExit("No global log found: " + log_path)

    latest = _latest_entries_from_global_log(log_path, scope=scope)
    if not latest:
        raise SystemExit("No eligible entries parsed from global log (scope=%s)." % scope)

    # filter reruns
    rerun_pairs = [(real, ev) for (real, ev), row in latest.items() if _should_rerun(row)]
    if not rerun_pairs:
        print("No pairs qualify for rerun under current policy; nothing created.")
        return ""

    # next run id
    max_n = -1
    for name in os.listdir(runs_dir):
        m = re.match(r"^run_(\d{3})$", name)
        if m:
            n = int(m.group(1))
            if n > max_n: max_n = n
    next_n = max_n + 1
    next_run_dir = os.path.join(runs_dir, f"run_{next_n:03d}")
    os.makedirs(next_run_dir, exist_ok=True)

    # group by src and create overlays with NEXT shifts
    by_src: Dict[str, List[int]] = {}
    for real, ev in rerun_pairs:
        by_src.setdefault(real, []).append(ev)

    overlays_dir = os.path.join(next_run_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    overlay_map: Dict[str, str] = {}

    for src, events in by_src.items():
        base = os.path.basename(src)
        overlay_path = os.path.join(overlays_dir, f"overlay_{base}")
        create_overlay(src, overlay_path)
        overlay_map[src] = overlay_path
        # collect next dx/dy by event order
        events_sorted = sorted(events)
        dx_vals = [float(latest[(src, ev)]["next_dx"]) for ev in events_sorted]
        dy_vals = [float(latest[(src, ev)]["next_dy"]) for ev in events_sorted]
        write_shifts_mm(overlay_path, events_sorted, dx_vals, dy_vals)

    # lst
    lst_next = os.path.join(next_run_dir, f"lst_{next_n:03d}.lst")
    with open(lst_next, "w", encoding="utf-8") as f:
        for src, events in sorted(by_src.items()):
            for ev in sorted(events):
                f.write(f"{overlay_map[src]} //{ev}\n")

    # sh (reuse sh_000.sh geometry/cell/flags)
    def parse_sh(sh_path: str):
        geom = cell = None; flags: List[str] = []
        with open(sh_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): continue
                toks = shlex.split(ln)
                if toks and "indexamajig" in toks[0]:
                    i=1
                    while i<len(toks):
                        t=toks[i]
                        if t=="-g": geom=toks[i+1]; i+=2; continue
                        if t=="-p": cell=toks[i+1]; i+=2; continue
                        if t in ("-i","-o"): i+=2; continue
                        flags.append(t); i+=1
                    break
        return geom, cell, flags

    sh0 = os.path.join(runs_dir, "run_000", "sh_000.sh")
    geom, cell, flags = parse_sh(sh0)
    sh_next = os.path.join(next_run_dir, f"sh_{next_n:03d}.sh")
    stream_next = os.path.join(next_run_dir, f"stream_{next_n:03d}.stream")
    with open(sh_next, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        cmd = ["indexamajig", "-g", geom, "-i", lst_next, "-o", stream_next, "-p", cell, *flags]
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    os.chmod(sh_next, 0o755)

    print(f"Prepared next run run_{next_n:03d} from global log (scope={scope})")
    print(f"  lst: {lst_next}")
    print(f"  sh:  {sh_next}")
    return next_run_dir

def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare next run folder (lst + overlays + sh) from runs/image_run_log.csv")
    ap.add_argument("--run-root", help="Path to run root (parent of runs/)")
    ap.add_argument("--scope", default="last", help="'last' (default) or 'all' to include all latest rows")
    args = ap.parse_args(argv)

    # ✅ safer default handling
    run_root = os.path.abspath(os.path.expanduser(args.run_root)) if args.run_root else DEFAULT_ROOT
    scope = args.scope

    prepare_next_from_log(run_root, scope=scope)
    return 0

if __name__ == "__main__":
    sys.exit(main())
