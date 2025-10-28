#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_overlays_and_list.py 

From image_run_log.csv (latest run only), create per-source overlay HDF5 files
(named with the next run number) and a lst_XXX.lst that lists lines like:

  /path/to/overlay_run_number.h5 //event_number

Additions in this adjusted version:
- Persist a JSON + TSV mapping from overlay .h5 → original .h5 in each run folder:
    runs/run_XXX/overlay_to_original.json
    runs/run_XXX/overlay_to_original.tsv
- (Optional) Tag each overlay file with an HDF5 attribute 'overlay_original_path'
  if h5py is available.

Notes:
- Skips entries whose next_* fields are "done".
- Overlays are created in runs/run_{next:03d}/
- Each overlay filename is <src_basename>_overlay_{next:03d}.h5
- Requires overlay_elink.py to be importable (create_overlay, write_shifts_mm)
"""
from __future__ import annotations

import argparse, os, sys, math, json

# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
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

            # ensure we have at least 7 entries
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

def tag_overlay_with_original(overlay_path: str, original_path: str) -> None:
    """Optionally store the original path as an attribute in the overlay HDF5 file.
    Skips silently if h5py is unavailable or file isn't writeable.
    """
    try:
        import h5py
    except Exception:
        return
    try:
        with h5py.File(overlay_path, "r+") as h5:
            h5.attrs["overlay_original_path"] = original_path
    except Exception:
        # non-fatal
        pass

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
    # (optional) embed original path as HDF5 attribute
    try:
        tag_overlay_with_original(overlay_path, _abs(src_path))
    except Exception:
        pass
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

def write_overlay_mapping(run_dir: str, overlay_paths: dict) -> str:
    """
    Persist mapping files that relate overlays to originals within the run folder.
    Writes:
      - overlay_to_original.json : {"/abs/overlay.h5": "/abs/original.h5", ...}
      - overlay_to_original.tsv  : tab-delimited table for quick inspection
    Returns the path to the JSON file.
    """
    os.makedirs(run_dir, exist_ok=True)
    map_json = os.path.join(run_dir, "overlay_to_original.json")
    map_tsv  = os.path.join(run_dir, "overlay_to_original.tsv")

    overlay_to_src = { _abs(ov): _abs(src) for src, ov in overlay_paths.items() }

    with open(map_json, "w", encoding="utf-8") as f:
        json.dump(overlay_to_src, f, indent=2)

    with open(map_tsv, "w", encoding="utf-8") as f:
        f.write("# overlay_path\toriginal_path\n")
        for ov, src in sorted(overlay_to_src.items()):
            f.write(f"{ov}\t{src}\n")
    return map_json

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Create run-named overlays and a .lst with '<overlay> //<event>' lines from latest proposals. Also writes overlay→original mapping files.")
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

    # Persist the overlay→original mapping (JSON + TSV)
    map_json_path = write_overlay_mapping(next_run_dir, overlay_paths)

    print(f"[overlay] Created/updated {len(overlay_paths)} overlay files in {next_run_dir}")
    print(f"[overlay] Wrote {n_lines} lines to {lst_path}")
    print(f"[overlay] Wrote overlay→original map to {map_json_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
