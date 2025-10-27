#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_early_break_from_log.py  (improvement-only incremental)

Update early_break.stream incrementally from image_run_log.csv with this policy:
  - For each (image,event) group, look at the **latest** row.
  - If the latest row's finite wRMSD is **strictly lower** than any prior finite wRMSD
    for that group, **replace** the group's chunk with the latest run's chunk.
  - If no prior finite wRMSD exists for that group and the latest is finite,
    **append** the chunk (first success).
  - Otherwise, **do nothing** for that group.
If no groups require replacement/append, the existing early_break.stream file is left unchanged.

We still rewrite the whole file only when there is at least one change, using a temp file + atomic replace.
"""

from __future__ import annotations
import os
import sys
import re
import math
import argparse
import glob
from typing import Dict, Tuple, List, Optional

HeaderKey = Tuple[str, int]  # (abs_image_path, event)

DEFAULT_RUN_ROOT = "/home/bubl3932/files/ici_trials/runs"
DEFAULT_LOG_NAME = "image_run_log.csv"
DEFAULT_OUT_NAME = "early_break.stream"
EPS = 1e-12  # strict improvement

# ----------------------------- Helpers -----------------------------

def _abs(s: str) -> str:
    return os.path.abspath(os.path.expanduser(s.strip()))

def _flt(s: str) -> Optional[float]:
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def run_stream_path(run_root: str, run_n: int) -> str:
    return os.path.join(run_root, f"run_{run_n:03d}", f"stream_{run_n:03d}.stream")

# ------------------------- Log Parsing ----------------------------

def parse_image_run_log(log_path: str) -> Dict[HeaderKey, List[dict]]:
    """
    Parse interleaved headers and CSV rows of the global log:

      #/abs/path.h5 event 40
      run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
      0,0.3285,0.1037,1,0.9361,0.3385,0.1037
      ...

    Returns: {(abs_image_path, event): [ {'run_n': int, 'wrmsd': float|None}, ... ]}
             rows are in file order (ascending time), so last element is "latest".
    """
    groups: Dict[HeaderKey, List[dict]] = {}
    cur: Optional[HeaderKey] = None

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue

            if ln.startswith("#/"):
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

            # wrmsd at parts[4]
            wr = _flt(parts[4])
            try:
                rn = int(parts[0])
            except Exception:
                rn = int(re.sub(r"\D+", "", parts[0]) or 0)
            groups[cur].append({"run_n": rn, "wrmsd": wr})

    return groups

def decide_updates(groups: Dict[HeaderKey, List[dict]]) -> Dict[HeaderKey, int]:
    """
    For each group, decide if the latest row qualifies:
      - latest wrmsd is finite
      - and (no prior finite wrmsd) -> append
        OR (latest wrmsd < min_prior - EPS) -> replace
    Return dict of {key -> run_n_of_latest} for groups that need updating/appending.
    """
    out: Dict[HeaderKey, int] = {}
    for key, rows in groups.items():
        if not rows:
            continue
        latest = rows[-1]
        w_latest = latest["wrmsd"]
        if w_latest is None:
            continue  # nothing to do

        # prior finite minima
        prior = [r["wrmsd"] for r in rows[:-1] if r["wrmsd"] is not None]
        if not prior:
            # first success -> append
            out[key] = latest["run_n"]
            continue
        min_prior = min(prior)
        if w_latest < (min_prior - EPS):
            # strict improvement -> replace
            out[key] = latest["run_n"]
    return out

# --------------------- Stream Parsing / Indexing -------------------

def parse_stream(file_path: str):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_end = 0
    for i, ln in enumerate(lines):
        if "Begin chunk" in ln:
            header_end = i
            break
    header = lines[:header_end] if header_end > 0 else []

    bounds = []
    start = None
    for i, ln in enumerate(lines):
        if "Begin chunk" in ln:
            start = i
        elif "End chunk" in ln and start is not None:
            bounds.append((start, i + 1))
            start = None
    return bounds, lines, header

def chunk_key_and_event(lines_slice: List[str]) -> Optional[HeaderKey]:
    img_path = None
    ev = None
    for ln in lines_slice:
        if "Image filename:" in ln:
            img_path = ln.split("Image filename:", 1)[1].strip()
        elif "Image file:" in ln:
            img_path = ln.split("Image file:", 1)[1].strip()
        elif ln.strip().startswith("Event:"):
            tail = ln.split("Event:", 1)[1].strip().lstrip("/")
            try:
                ev = int(tail)
            except Exception:
                pass
    if img_path is None or ev is None:
        return None
    return (_abs(img_path), int(ev))

def load_chunk_for(run_root: str, rn: int, key: HeaderKey) -> Optional[List[str]]:
    stream_path = run_stream_path(run_root, rn)
    if not os.path.isfile(stream_path):
        print(f"[WARN] Missing stream for run_{rn:03d}: {stream_path}")
        return None
    bounds, lines, _ = parse_stream(stream_path)
    # Build a quick index for this stream
    index: Dict[HeaderKey, Tuple[int,int]] = {}
    for a, b in bounds:
        k = chunk_key_and_event(lines[a:b])
        if k is not None:
            index[k] = (a, b)
    seg = index.get(key)
    if seg is None:
        print(f"[WARN] No chunk for {key} in run_{rn:03d}")
        return None
    a, b = seg
    return lines[a:b]

# --------------------------- Main Builder -------------------------

def build_early_break_incremental(run_root: str,
                                  log_name: str = DEFAULT_LOG_NAME,
                                  out_name: str = DEFAULT_OUT_NAME) -> str:
    run_root = _abs(run_root)
    log_path = os.path.join(run_root, log_name)
    out_path = os.path.join(run_root, out_name)
    out_tmp = out_path + ".tmp"

    if not os.path.isfile(log_path):
        raise SystemExit(f"[ERR] No log at {log_path}")

    groups = parse_image_run_log(log_path)
    plans = decide_updates(groups)  # {key -> latest_run}

    # If file doesn't exist yet and nothing to add, nothing to do
    if not os.path.isfile(out_path) and not plans:
        print("[early-break] No updates and no existing file; nothing to do.")
        return out_path

    # Parse existing early_break (if present)
    existing_bounds, existing_lines, existing_header = [], [], []
    if os.path.isfile(out_path):
        try:
            existing_bounds, existing_lines, existing_header = parse_stream(out_path)
        except Exception as e:
            print(f"[WARN] Could not parse existing {out_path}: {e}")

    # Build map of existing chunks
    existing_map: Dict[HeaderKey, Tuple[int,int]] = {}
    for (a, b) in existing_bounds:
        k = chunk_key_and_event(existing_lines[a:b])
        if k is not None:
            existing_map[k] = (a, b)

    # Resolve new/updated chunks
    new_chunks: Dict[HeaderKey, List[str]] = {}
    for key, rn in plans.items():
        seg = load_chunk_for(run_root, rn, key)
        if seg is not None:
            new_chunks[key] = seg

    # If after resolution we still have nothing to change, do nothing
    if not new_chunks:
        print("[early-break] No improvements or first-success rows detected; leaving file unchanged.")
        return out_path

    # Decide header to write
    header_to_write = existing_header
    if not header_to_write:
        # attempt to borrow a header from any contributing stream
        for rn in set(plans.values()):
            sp = run_stream_path(run_root, rn)
            if os.path.isfile(sp):
                try:
                    _, lines, hdr = parse_stream(sp)
                    if hdr:
                        header_to_write = hdr
                        break
                except Exception:
                    pass

    # Write merged result to temp
    replaced = 0
    appended = 0
    total_chunks = 0

    with open(out_tmp, "w", encoding="utf-8") as wf:
        if header_to_write:
            wf.writelines(header_to_write)

        # Write existing chunks, replacing those with improvements
        if existing_bounds and existing_lines:
            for (a, b) in existing_bounds:
                key = chunk_key_and_event(existing_lines[a:b])
                if key is None:
                    wf.writelines(existing_lines[a:b])
                    total_chunks += 1
                    continue
                if key in new_chunks:
                    wf.writelines(new_chunks.pop(key))  # replace
                    replaced += 1
                else:
                    wf.writelines(existing_lines[a:b])   # keep
                total_chunks += 1

        # Append remaining new chunks (first successes)
        for key in sorted(new_chunks.keys(), key=lambda t: (t[0], t[1])):
            wf.writelines(new_chunks[key])
            appended += 1
            total_chunks += 1

    # Atomic replace
    os.replace(out_tmp, out_path)
    print(f"[early-break] Updated {out_path}: replaced {replaced}, appended {appended}, total chunks now {total_chunks}.")
    return out_path

# ------------------------------ CLI --------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(description="Incrementally update early_break.stream based on *improvement-only* policy.")
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Path containing image_run_log.csv and run_*/")
    ap.add_argument("--log-name", default=DEFAULT_LOG_NAME, help="Log filename (default: image_run_log.csv)")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME, help="Output filename (default: early_break.stream)")
    args = ap.parse_args(argv)

    run_root = _abs(args.run_root)
    build_early_break_incremental(run_root, args.log_name, args.out_name)
    return 0

if __name__ == "__main__":
    sys.exit(main())
