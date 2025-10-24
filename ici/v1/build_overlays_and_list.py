#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_overlays_and_list.py

From image_run_log.csv (latest run only), create per-source overlay HDF5 files
(named with the next run number) and a lst_XXX.lst that lists lines like:

  /path/to/overlay_run_number.h5 //event_number

Notes:
- Skips entries whose next_* fields are "done".
- Overlays are created in runs/run_{next:03d}/
- Each overlay filename is <src_basename>__run_{next:03d}.h5
- Requires overlay_elink.py to be importable (create_overlay, write_shifts_mm)
"""
from __future__ import annotations
import argparse, os, sys, math

DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"

try:
    from overlay_elink import create_overlay, write_shifts_mm
except Exception:
    print("ERROR: Could not import overlay_elink.py (create_overlay, write_shifts_mm). "
          "Place overlay_elink.py next to this script or add it to PYTHONPATH.", file=sys.stderr)
    raise

def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def parse_log(log_path: str):
    """
    Returns:
      entries: list of tuples (type, payload)
               type == 'section' -> payload = (real_path, event:int)
               type == 'row'     -> payload = (run_n:int, dx:str, dy:str, indexed:str, wrmsd:str, next_dx:str, next_dy:str)
      latest_run: int
    """
    entries = []
    latest_run = -1
    with open(log_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for ln in f:
            if ln.startswith("#/"):
                try:
                    path_part, ev_part = ln[1:].rsplit(" event ", 1)
                    ev = int(ev_part.strip())
                    entries.append(("section", (_abs(path_part.strip()), ev)))
                except Exception:
                    pass
                continue
            parts = [p.strip() for p in ln.rstrip("\n").split(",")]
            while len(parts) < 7:
                parts.append("")
            run_s, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = parts[:7]
            try:
                run_n = int(run_s)
                latest_run = max(latest_run, run_n)
            except Exception:
                continue
            entries.append(("row", (run_n, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s)))
    return entries, latest_run

def collect_latest_numeric_proposals(entries, latest_run: int):
    """
    From latest run only, collect numeric proposals per (real_path, event).
    Returns mapping: real_path -> { event_idx: (next_dx_float, next_dy_float) }
    """
    by_src = {}
    current_key = None
    last_next = {}  # key -> (ndx, ndy) strings, keep last seen in the section
    for typ, payload in entries:
        if typ == "section":
            current_key = payload  # (real_path, event)
            continue
        if typ != "row" or current_key is None:
            continue
        run_n, dx_s, dy_s, idx_s, wr_s, ndx_s, ndy_s = payload
        if run_n != latest_run:
            continue
        # keep last row’s proposal for this key
        last_next[current_key] = (ndx_s, ndy_s)

    # Convert strings to floats and filter "done"
    for (rp, ev), (ndx_s, ndy_s) in last_next.items():
        if not ndx_s or not ndy_s:
            continue
        if ndx_s == "done" or ndy_s == "done":
            continue
        try:
            ndx = float(ndx_s); ndy = float(ndy_s)
            if not (math.isfinite(ndx) and math.isfinite(ndy)):
                continue
        except Exception:
            continue
        by_src.setdefault(rp, {})[int(ev)] = (ndx, ndy)

    return by_src

def ensure_overlay_for_run(src_path: str, run_dir: str, next_run: int) -> str:
    """
    Create (or overwrite) an overlay for src_path under run_dir.
    Name: <basename>_overlay_{nextoverlay:03d}.h5
    Returns absolute overlay path.
    """
    os.makedirs(run_dir, exist_ok=True)
    base = os.path.basename(src_path)
    root, _ = os.path.splitext(base)
    overlay_name = f"{root}_overlay_{next_run:03d}.h5"
    overlay_path = os.path.join(run_dir, overlay_name)
    # Create/overwrite and seed arrays
    create_overlay(src_path, overlay_path)
    return _abs(overlay_path)

def write_all_shifts(run_dir: str, next_run: int, proposals_by_src: dict):
    """
    For each source, write its proposed shifts to its overlay in run_dir.
    Returns: dict src_path -> overlay_path
    """
    overlay_paths = {}
    for src_path, ev2shift in proposals_by_src.items():
        if not ev2shift:
            continue
        overlay_path = ensure_overlay_for_run(src_path, run_dir, next_run)
        overlay_paths[src_path] = overlay_path
        # Prepare vectors
        indices = sorted(ev2shift.keys())
        dx = [ev2shift[i][0] for i in indices]
        dy = [ev2shift[i][1] for i in indices]
        write_shifts_mm(overlay_path, indices, dx, dy)
    return overlay_paths

def write_lst(lst_path: str, proposals_by_src: dict, overlay_paths: dict):
    """
    Write lines like:
      /path/to/overlay_run_number.h5 //event_number
    """
    n = 0
    with open(lst_path, "w", encoding="utf-8") as f:
        for src_path, ev2shift in sorted(proposals_by_src.items()):
            if not ev2shift:
                continue
            ov = overlay_paths.get(src_path)
            if not ov:
                continue
            for ev in sorted(ev2shift.keys()):
                f.write(f"{ov} //{ev}\n")
                n += 1
    return n

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create run-named overlays and a .lst with '<overlay> //<event>' lines from latest proposals.")
    ap.add_argument("--run-root", default=None,
                    help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.")
    args = ap.parse_args(argv)

    run_root = _abs(args.run_root or DEFAULT_ROOT)
    runs_dir = os.path.join(run_root, "runs")
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if not os.path.isfile(log_path):
        print("ERROR: missing runs/image_run_log.csv", file=sys.stderr)
        return 2

    entries, latest_run = parse_log(log_path)
    if latest_run < 0:
        print("ERROR: no data rows found in image_run_log.csv", file=sys.stderr)
        return 2

    proposals_by_src = collect_latest_numeric_proposals(entries, latest_run)
    if not proposals_by_src:
        print(f"[overlay] No numeric proposals in run_{latest_run:03d} (all 'done' or missing). Nothing to do.")
        return 0

    next_run = latest_run + 1
    next_run_dir = os.path.join(runs_dir, f"run_{next_run:03d}")
    os.makedirs(next_run_dir, exist_ok=True)

    # Create per-source overlays inside the next run folder, named with run number
    overlay_paths = write_all_shifts(next_run_dir, next_run, proposals_by_src)

    # Create the list file with the desired format
    lst_path = os.path.join(next_run_dir, f"lst_{next_run:03d}.lst")
    n_lines = write_lst(lst_path, proposals_by_src, overlay_paths)

    print(f"[overlay] Created/updated {len(overlay_paths)} overlay files in {next_run_dir}")
    print(f"[overlay] Wrote {n_lines} lines to {lst_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
