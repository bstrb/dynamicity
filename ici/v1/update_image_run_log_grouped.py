#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_image_run_log_grouped.py

Ingest the latest chunk_metrics_###.csv from runs/run_### and append rows to
runs/image_run_log.csv using the existing CSV schema:
run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm

This version *groups* rows by image-event sections so that new run entries are
inserted into the correct section rather than all being appended at the end.
If a section (image-event) doesn't exist yet, it is created at the end.

Duplicates (identical run lines within a given section) are avoided.

Notes
-----
- Section headers are lines of the form:
  "#/abs/path/to/file.h5 event 123"
- The CSV has a single header line at the very top.
- "next_*" fields are intentionally left blank; a follow-up script will fill them.
"""
from __future__ import annotations
import argparse, csv, math, os, re, sys
from typing import Dict, List, Tuple, Optional, OrderedDict
import h5py
from collections import OrderedDict as OD

DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
IMAGES_DS = "/entry/data/images"

CSV_HEADER = "run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm\n"
SECTION_RE = re.compile(r"^#(?P<path>/.*)\s+event\s+(?P<ev>\d+)\s*$")

def resolve_real_source(h5_path: str) -> str:
    # \"\"\"Return the real HDF5 path if images dataset is an ExternalLink; else the input path.\"\"\"
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

def _parse_log_into_sections(lines: List[str]) -> Tuple[str, "OD[Tuple[str,int], List[str]]"]:
    # \"\"\"
    # Parse existing log lines into:
    #   - top_header: the CSV header line (or CSV_HEADER if missing)
    #   - sections: OrderedDict mapping (abs_path, event) -> list of lines
    #     (includes the section header line as the first element).
    # Any non-section lines appearing before the first section (besides header) are preserved after header.
    # \"\"\"
    sections: "OD[Tuple[str,int], List[str]]" = OD()
    i = 0
    top_header = CSV_HEADER
    preface: List[str] = []

    if lines:
        # First line should be the CSV header; if not, we'll insert one.
        if lines[0].strip().lower().startswith("run_n,"):
            top_header = lines[0]
            i = 1
        else:
            # Keep whatever was there as preface to preserve file content
            preface.append(lines[0])
            i = 1

    current_key: Optional[Tuple[str,int]] = None

    def finalize_preface():
        if preface:
            # Attach preface as a pseudo-section under a special key to preserve order
            sections[("__PREFACE__", -1)] = preface.copy()

    # Scan remaining lines
    for line in lines[i:]:
        m = SECTION_RE.match(line)
        if m:
            if current_key is None and preface:
                finalize_preface()
                preface.clear()
            path = os.path.abspath(m.group("path").strip())
            ev = int(m.group("ev"))
            current_key = (path, ev)
            sections.setdefault(current_key, []).append(line)
        else:
            if current_key is None:
                preface.append(line)
            else:
                sections[current_key].append(line)

    if current_key is None and preface:
        finalize_preface()

    return top_header, sections

def _ensure_section(sections: "OD[Tuple[str,int], List[str]]", key: Tuple[str,int]) -> None:
    if key not in sections:
        sections[key] = [f"#{key[0]} event {key[1]}\n"]

def _existing_run_lines_in_section(section_lines: List[str]) -> set:
    # \"\"\"Return set of run lines (string after header) present in this section, to avoid duplicates.\"\"\"
    existing = set()
    for ln in section_lines:
        if ln.startswith("#"):
            continue
        s = ln.strip()
        if s:  # non-empty CSV line
            existing.add(s)
    return existing

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Append latest run rows to runs/image_run_log.csv grouped by image-event (no next_*)."
    )
    ap.add_argument(
        "--run-root",
        default=None,
        help="Path to run root that contains 'runs/'. Defaults to DEFAULT_ROOT if omitted.",
    )
    args = ap.parse_args(argv)

    # If --run-root is omitted or empty, fall back to DEFAULT_ROOT
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

    # Load existing log (if any)
    log_path = os.path.join(runs_dir, "image_run_log.csv")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            existing_lines = f.readlines()
    else:
        existing_lines = []

    header_line, sections = _parse_log_into_sections(existing_lines)
    if not header_line.strip().lower().startswith("run_n,"):
        header_line = CSV_HEADER  # enforce correct header

    # Track order of new sections to append deterministically
    new_section_order: List[Tuple[str,int]] = []

    appended_rows = 0

    for row in latest_rows:
        img = (row.get("image") or "").strip()
        evs = (row.get("event") or "").strip()
        if not evs.isdigit():
            continue
        ev = int(evs)
        real = resolve_real_source(_src_from_image_col(img))
        key = (real, ev)

        # Prepare CSV row fields
        def _to_float(val, default=0.0) -> float:
            try:
                return float(val)
            except Exception:
                return float(default)

        dx = _to_float(row.get("det_shift_x_mm", 0.0))
        dy = _to_float(row.get("det_shift_y_mm", 0.0))
        try:
            indexed = int(row.get("indexed", 0) or 0)
        except Exception:
            indexed = 0

        wr_out = ""
        wrmsd = row.get("wrmsd", "")
        try:
            wv = float(wrmsd) if wrmsd not in ("", None) else float("nan")
            if math.isfinite(wv):
                wr_out = f"{wv:.6f}"
        except Exception:
            pass

        csv_line = f"{last_n},{dx:.6f},{dy:.6f},{indexed},{wr_out},,\n"

        # Create section if missing
        if key not in sections:
            _ensure_section(sections, key)
            new_section_order.append(key)

        # Avoid duplicates within the section
        existing_set = _existing_run_lines_in_section(sections[key])
        if csv_line.strip() not in existing_set:
            sections[key].append(csv_line)
            appended_rows += 1

    # Reassemble file with preserved order:
    # 1) header
    # 2) any preface (stored under ("__PREFACE__", -1)) if present
    # 3) existing sections in their current order
    # 4) new sections (if any), in the order they were first encountered above
    out_lines: List[str] = [header_line if header_line.endswith("\n") else header_line + "\n"]

    # Extract and write preface first (if any)
    preface_key = ("__PREFACE__", -1)
    if preface_key in sections:
        out_lines.extend(sections[preface_key])

    # Existing section order as stored
    for key, lines in list(sections.items()):
        if key == preface_key:
            continue
        out_lines.extend(lines)

    # For any brand-new section that might not have been in the initial sections order
    # (This is largely redundant because we already inserted sections in the OD and appended there,
    # but we keep this logic in case someone modifies insertion behavior above.)
    for key in new_section_order:
        if key in sections:
            # Ensure it's already in out_lines; if not, append
            # (Check by identity of the section header)
            header = f"#{key[0]} event {key[1]}\n"
            # Only append if the header isn't already present
            if header not in out_lines:
                out_lines.extend(sections[key])
 
    # Write back
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"[log] Appended {appended_rows} new rows into grouped sections in {log_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
