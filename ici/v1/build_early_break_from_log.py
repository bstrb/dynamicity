#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_early_break_from_log.py  —  MP version (read+prep in parallel, safe merge)

Policy (incremental, improvement-only):
  - For each (image,event) group, consider the latest row in image_run_log.csv.
  - If latest finite wRMSD < best prior finite wRMSD: REPLACE that group's chunk.
  - If no prior finite wRMSD and latest is finite: APPEND that group's chunk.
  - Otherwise: NO CHANGE for that group.
If nothing changes, the output file is left untouched.

Multiprocessing:
  - Chunk extraction from per-run stream files is parallelized.
  - Each worker writes the extracted chunk to a temp file and returns its path.
  - The main process concatenates header + existing/replaced/appended chunks
    deterministically into a temporary output, then atomically replaces the target.

CLI:
  --run-root   path to 'runs/' directory (containing image_run_log.csv and run_*/)
  --log-name   name of csv (default: image_run_log.csv)
  --out-name   output stream name (default: early_break.stream)
  --workers    number of worker processes (default: os.cpu_count())

Example:
  python build_early_break_from_log.py --run-root /data/exp/runs --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List, Optional

HeaderKey = Tuple[str, int]  # (abs_image_path, event)
EPS = 1e-12

DEFAULT_RUN_ROOT = "."
DEFAULT_LOG_NAME = "image_run_log.csv"
DEFAULT_OUT_NAME = "early_break.stream"


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


def _hash_key(key: HeaderKey) -> str:
    h = hashlib.sha1(f"{key[0]}|{key[1]}".encode("utf-8")).hexdigest()
    return h[:16]


# ------------------------- Log Parsing ----------------------------

def parse_image_run_log(log_path: str) -> Dict[HeaderKey, List[dict]]:
    """
    Parse interleaved headers and CSV rows of the global log:

      #/abs/path.h5 event 40
      run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
      0,0.3285,0.1037,1,0.9361,0.3385,0.1037
      ...

    Returns: {(abs_image_path, event): [ {'run_n': int, 'wrmsd': float|None}, ... ]}
             rows in file order (ascending), so last element is "latest".
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

def parse_stream_chunks(stream_path: str):
    """
    Return (bounds, lines, header_lines)
      - bounds: list of (start_idx, end_idx) covering each chunk
      - lines: all lines
      - header_lines: everything before first "Begin chunk"
    """
    with open(stream_path, "r", encoding="utf-8", errors="ignore") as f:
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


def chunk_key_from_slice(lines_slice: List[str]) -> Optional[HeaderKey]:
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


def map_chunk_ids_by_image_event(lines: List[str], bounds: List[Tuple[int, int]]) -> Dict[HeaderKey, int]:
    mapping: Dict[HeaderKey, int] = {}
    for cid, (a, b) in enumerate(bounds):
        key = chunk_key_from_slice(lines[a:b])
        if key is not None:
            mapping[key] = cid
    return mapping


def load_chunk_for(run_root: str, rn: int, key: HeaderKey) -> Optional[List[str]]:
    """
    Extract a single chunk (as list of lines) for (image,event) from run rn.
    """
    stream_path = run_stream_path(run_root, rn)
    if not os.path.isfile(stream_path):
        print(f"[WARN] Missing stream for run_{rn:03d}: {stream_path}")
        return None
    try:
        bounds, lines, _ = parse_stream_chunks(stream_path)
        mapping = map_chunk_ids_by_image_event(lines, bounds)
        cid = mapping.get(key)
        if cid is None:
            print(f"[WARN] No chunk for {key} in run_{rn:03d}")
            return None
        a, b = bounds[cid]
        return lines[a:b]
    except Exception as e:
        print(f"[WARN] Failed to parse {stream_path}: {e}")
        return None


# ---- Worker: load and spill chunk to temp file (for parallel IO) ----

def _worker_load_to_temp(args) -> Tuple[HeaderKey, Optional[str]]:
    run_root, rn, key, temp_dir = args
    seg = load_chunk_for(run_root, rn, key)
    if seg is None:
        return key, None
    # Write to a unique temp file
    h = _hash_key(key)
    tmp_path = os.path.join(temp_dir, f"{rn:03d}_{h}.chunk")
    with open(tmp_path, "w", encoding="utf-8") as wf:
        wf.writelines(seg)
    return key, tmp_path


# --------------------------- Main Builder -------------------------

def build_early_break_incremental(
    run_root: str,
    log_name: str = DEFAULT_LOG_NAME,
    out_name: str = DEFAULT_OUT_NAME,
    workers: Optional[int] = None,
) -> str:

    run_root = _abs(run_root)
    log_path = os.path.join(run_root, log_name)
    out_path = os.path.join(run_root, out_name)
    out_tmp = out_path + ".tmp"

    if not os.path.isfile(log_path):
        raise SystemExit(f"[ERR] No log at {log_path}")

    groups = parse_image_run_log(log_path)
    plans = decide_updates(groups)  # {key -> latest_run}

    # If no file and nothing to add, nothing to do
    if not os.path.isfile(out_path) and not plans:
        print("[early-break] No updates and no existing file; nothing to do.")
        return out_path

    # Parse existing early_break (if present)
    existing_bounds, existing_lines, existing_header = [], [], []
    if os.path.isfile(out_path):
        try:
            existing_bounds, existing_lines, existing_header = parse_stream_chunks(out_path)
        except Exception as e:
            print(f"[WARN] Could not parse existing {out_path}: {e}")

    # Build map of existing chunks
    existing_map: Dict[HeaderKey, Tuple[int,int]] = {}
    for (a, b) in existing_bounds:
        k = chunk_key_from_slice(existing_lines[a:b])
        if k is not None:
            existing_map[k] = (a, b)

    # Resolve new/updated chunks in parallel: spill each to temp file
    temp_dir = tempfile.mkdtemp(prefix="early_break_", dir=run_root)
    try:
        new_chunk_files: Dict[HeaderKey, str] = {}
        if plans:
            # default workers: all cores
            if workers is None or workers <= 0:
                workers = max(1, multiprocessing.cpu_count())
            # Clamp to number of tasks
            w = min(workers, max(1, len(plans)))
            print(f"[multi] Using {w} workers for {len(plans)} chunk(s)")
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = {
                    ex.submit(_worker_load_to_temp, (run_root, rn, key, temp_dir)): key
                    for key, rn in plans.items()
                }
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        k, tmp = fut.result()
                        if tmp is not None:
                            new_chunk_files[k] = tmp
                    except Exception as e:
                        print(f"[WARN] Worker failed for {key}: {e}")

        # After resolution, if nothing to change, leave file as-is
        if not new_chunk_files:
            print("[early-break] No improvements or first-success rows detected; leaving file unchanged.")
            return out_path

        # Decide header to write
        header_to_write = existing_header
        if not header_to_write:
            # attempt to borrow a header from any run stream we just used
            # (read first contributing temp file's source header via its run number)
            try:
                any_key = next(iter(new_chunk_files.keys()))
                rn_guess = plans[any_key]
                sp = run_stream_path(run_root, rn_guess)
                _, _, hdr = parse_stream_chunks(sp)
                if hdr:
                    header_to_write = hdr
            except Exception:
                pass

        # Merge deterministically
        replaced = 0
        appended = 0
        total_chunks = 0

        with open(out_tmp, "w", encoding="utf-8") as wf:
            if header_to_write:
                wf.writelines(header_to_write)

            # 1) write existing chunks, replacing where needed
            if existing_bounds and existing_lines:
                for (a, b) in existing_bounds:
                    key = chunk_key_from_slice(existing_lines[a:b])
                    if key is None:
                        wf.writelines(existing_lines[a:b])
                        total_chunks += 1
                        continue
                    tmp = new_chunk_files.pop(key, None)
                    if tmp is not None:
                        # replace
                        with open(tmp, "r", encoding="utf-8") as rf:
                            shutil.copyfileobj(rf, wf)
                        replaced += 1
                    else:
                        # keep
                        wf.writelines(existing_lines[a:b])
                    total_chunks += 1

            # 2) append remaining new chunks (first successes)
            if new_chunk_files:
                for key in sorted(new_chunk_files.keys(), key=lambda t: (t[0], t[1])):
                    with open(new_chunk_files[key], "r", encoding="utf-8") as rf:
                        shutil.copyfileobj(rf, wf)
                    appended += 1
                    total_chunks += 1

        # Atomic replace
        os.replace(out_tmp, out_path)
        print(f"[early-break] Merge complete: wrote {total_chunks} chunk(s) to {out_path} "
              f"(replaced {replaced}, appended {appended}).")
        return out_path

    finally:
        # cleanup temp dir
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ------------------------------ CLI --------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="Incrementally update early_break.stream (improvement-only) with parallel chunk extraction."
    )
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT,
                    help="Path to 'runs/' containing image_run_log.csv and run_*/")
    ap.add_argument("--log-name", default=DEFAULT_LOG_NAME,
                    help=f"Log filename (default: {DEFAULT_LOG_NAME})")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                    help=f"Output stream filename (default: {DEFAULT_OUT_NAME})")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of worker processes (default: all cores)")
    args = ap.parse_args(argv)

    run_root = _abs(args.run_root)
    build_early_break_incremental(run_root, args.log_name, args.out_name, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
