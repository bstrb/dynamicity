#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_image_run_log.py

Ingest the latest chunk_metrics_###.csv from runs/run_### and append rows to
runs/image_run_log.csv using the existing CSV schema:
run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm

This script *does not* compute next_*; it leaves those two fields blank for the
newly appended rows. A follow-up script will fill them in.
"""
from __future__ import annotations
import argparse, csv, math, os, re, sys
from typing import Dict, List, Tuple, Optional
import h5py

DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
IMAGES_DS = "/entry/data/images"

def resolve_real_source(h5_path: str) -> str:
    """Return the real HDF5 path if images dataset is an ExternalLink; else the input path."""
    ap = os.path.abspath(h5_path)
    try:
        with h5py.File(ap, "r") as f:
            link = f.get(IMAGES_DS, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                return os.path.abspath(link.filename)
    except Exception:
        pass
    return ap

def _src_from_image_col(img: str) -> str:
    s = img.strip()
    if "//" in s:
        h5, _ = s.split("//", 1)
        return h5.strip()
    return s

def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _find_latest_run_dir(runs_dir: str) -> Tuple[int, str]:
    last_n, last_dir = -1, ""
    if not os.path.isdir(runs_dir):
        return -1, ""
    for name in os.listdir(runs_dir):
        m = re.match(r"^run_(\d{3})$", name)
        if m:
            n = int(m.group(1))
            if n > last_n:
                last_n = n
                last_dir = os.path.join(runs_dir, name)
    return last_n, last_dir

def _existing_keys(log_path: str) -> set:
    """Return set of (abs_path, event) pairs already present as section headers in log."""
    seen = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.startswith("#/"):
                    try:
                        path_part, ev_part = ln[1:].rsplit(" event ", 1)
                        ev = int(ev_part.strip())
                        seen.add((os.path.abspath(path_part.strip()), ev))
                    except Exception:
                        pass
    return seen

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Append latest run rows to runs/image_run_log.csv (no next_*)."
    )
    ap.add_argument(
        "--run-root",
        default=None,
        help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.",
    )
    args = ap.parse_args(argv)

    # ✅ If --run-root is omitted or empty, fall back to DEFAULT_ROOT
    run_root = os.path.abspath(os.path.expanduser(args.run_root or DEFAULT_ROOT))

    runs_dir = os.path.join(run_root, "runs")
    os.makedirs(runs_dir, exist_ok=True)


    last_n, last_run_dir = _find_latest_run_dir(runs_dir)
    if last_n < 0 or not last_run_dir:
        print("ERROR: no run_* folders found", file=sys.stderr)
        return 2

    metrics_path = os.path.join(last_run_dir, f"chunk_metrics_{last_n:03d}.csv")
    if not os.path.isfile(metrics_path):
        print("ERROR: missing latest metrics", file=sys.stderr)
        return 2

    latest_rows = _read_csv_rows(metrics_path)

    # Prepare header and where to write
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    write_header = not os.path.exists(log_path)
    if write_header:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm\n")

    seen = _existing_keys(log_path)

    # Append new section headers and rows
    out_lines: List[str] = []
    appended_rows = 0

    for row in latest_rows:
        img = (row.get("image") or "").strip()
        evs = (row.get("event") or "").strip()
        if not evs.isdigit():
            continue
        ev = int(evs)
        real = resolve_real_source(_src_from_image_col(img))

        key = (real, ev)
        if key not in seen:
            out_lines.append("#" + real + f" event {ev}\n")
            seen.add(key)

        # Tried values
        try:
            dx = float(row.get("det_shift_x_mm", "0") or 0.0)
        except Exception:
            dx = 0.0
        try:
            dy = float(row.get("det_shift_y_mm", "0") or 0.0)
        except Exception:
            dy = 0.0
        indexed = int(row.get("indexed", "0") or 0)
        wrmsd = row.get("wrmsd", "")
        wr_out = ""
        try:
            wv = float(wrmsd) if wrmsd not in ("", None) else float("nan")
            if math.isfinite(wv):
                wr_out = f"{wv:.6f}"
        except Exception:
            pass

        out_lines.append(f"{last_n},{dx:.6f},{dy:.6f},{indexed},{wr_out},,\n")
        appended_rows += 1

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"[log] Appended {appended_rows} rows to {log_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
