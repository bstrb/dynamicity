#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_early_break_from_log.py

Fast builder for early_break.stream from image_run_log.csv, optimized for very
large .stream files.

Policy (per (image, event) group):
  - Consider only finite numeric wRMSD values.
  - If the *last* row of that group's block has the minimal finite wRMSD,
    select that row's run_n.
  - Copy the matching chunk (identified by Image filename + Event) from
    {run_root}/run_{run:03d}/stream_{run:03d}.stream into {run_root}/early_break.stream.
  - Write the header (everything above the first "Begin chunk") exactly once,
    taken from the *first* successfully loaded run (lowest run number).

I/O & Speed:
  - Groups selections by run, opens each run's stream once.
  - Builds a per-run in-memory index: {(abs_image_path, event) -> chunk_id}.
  - Uses ProcessPoolExecutor to parse stream files in parallel (CPU-bound parsing).

Usage:
  python build_early_break_from_log.py
  python build_early_break_from_log.py --run-root /home/bubl3932/files/ici_trials/runs
  python build_early_break_from_log.py --workers 8
"""

from __future__ import annotations
import os
import sys
import re
import math
import argparse
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

HeaderKey = Tuple[str, int]  # (abs_image_path, event)

DEFAULT_RUN_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004/runs"
# DEFAULT_RUN_ROOT = "/home/bubl3932/files/ici_trials/runs"
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


# ------------------------- Log Parsing ----------------------------

def parse_image_run_log(log_path: str) -> Dict[HeaderKey, List[dict]]:
    """
    Parse interleaved headers and CSV rows of the global log:

      #/abs/path.h5 event 40
      run_n,det_shift_x_mm,det_shift_y_mm,indexed,wrmsd,next_dx_mm,next_dy_mm
      0,0.3285,0.1037,1,0.9361,0.3385,0.1037
      ...

    Returns: {(abs_image_path, event): [ {'run_n': str, 'wrmsd': str}, ... ]}
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

            groups[cur].append({
                "run_n": parts[0],
                "wrmsd": parts[4],
            })

    return groups


def choose_groups_where_last_is_min(groups: Dict[HeaderKey, List[dict]]) -> List[Tuple[HeaderKey, int]]:
    """
    For each group: find minimal finite wRMSD; if it's on the LAST row,
    select that row's run_n (int).
    """
    chosen: List[Tuple[HeaderKey, int]] = []
    for key, rows in groups.items():
        wrs = [_flt(r.get("wrmsd", "")) for r in rows]
        valid = [(i, w) for i, w in enumerate(wrs) if w is not None]
        if not valid:
            continue
        min_idx, _ = min(valid, key=lambda t: t[1])
        last_idx = len(rows) - 1
        if min_idx != last_idx:
            continue
        rn_str = rows[last_idx]["run_n"]
        try:
            rn = int(rn_str)
        except Exception:
            rn = int(re.sub(r"\D+", "", rn_str) or 0)
        chosen.append((key, rn))
    return chosen


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
 

def map_chunk_ids_by_image_event(lines: List[str], bounds: List[Tuple[int, int]]) -> Dict[HeaderKey, int]:
    """
    Build {(abs_image_path, event) -> chunk_id} once per stream.
    Accepts 'Image filename:' or 'Image file:' and 'Event: //N' or 'Event: N'.
    """
    mapping: Dict[HeaderKey, int] = {}
    for cid, (a, b) in enumerate(bounds):
        img_path = None
        ev = None
        for ln in lines[a:b]:
            if "Image filename:" in ln:
                img_path = ln.split("Image filename:", 1)[1].strip()
            elif "Image file:" in ln:
                img_path = ln.split("Image file:", 1)[1].strip()
            elif ln.strip().startswith("Event:"):
                tail = ln.split("Event:", 1)[1].strip()
                tail = tail.lstrip("/")  # allow "Event: //40"
                try:
                    ev = int(tail)
                except Exception:
                    pass
        if img_path is not None and ev is not None:
            mapping[(_abs(img_path), ev)] = cid
    return mapping


def load_run_stream(run_root: str, rn: int):
    """
    Parse and index one run's stream file.

    Returns (rn, bounds, lines, header, mapping) on success; None on failure.
    This function is defined at module top-level to be picklable for multiprocessing.
    """
    src = run_stream_path(run_root, rn)
    if not os.path.isfile(src):
        print(f"[WARN] Missing stream: {src}")
        return None
    try:
        bounds, lines, header = parse_stream_chunks(src)
        mapping = map_chunk_ids_by_image_event(lines, bounds)
        return (rn, bounds, lines, header, mapping)
    except Exception as e:
        print(f"[WARN] Failed to parse {src}: {e}")
        return None


# --------------------------- Main Builder -------------------------

def build_early_break(run_root: str,
                      log_name: str = DEFAULT_LOG_NAME,
                      out_name: str = DEFAULT_OUT_NAME,
                      workers: Optional[int] = None) -> str:
    run_root = _abs(run_root)
    log_path = os.path.join(run_root, log_name)
    out_path = os.path.join(run_root, out_name)

    if not os.path.isfile(log_path):
        raise SystemExit(f"[ERR] No log at {log_path}")

    groups = parse_image_run_log(log_path)
    picks = choose_groups_where_last_is_min(groups)
    if not picks:
        with open(out_path, "w", encoding="utf-8") as f:
            pass
        print("[early-break] No groups selected (last row not minimal wRMSD).")
        return out_path

    # Group selections by run
    from collections import defaultdict
    by_run: Dict[int, List[HeaderKey]] = defaultdict(list)
    for (img, ev), rn in picks:
        by_run[rn].append((img, ev))

    run_list = sorted(by_run.keys())
    if workers is None or workers <= 0:
        workers = max(1, (os.cpu_count() or 1))

    # Load and index streams in parallel (processes)
    loaded = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_run_stream, run_root, rn): rn for rn in run_list}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                rn, bounds, lines, header, mapping = res
                loaded[rn] = (bounds, lines, header, mapping)

    # Write deterministically in ascending rn
    # wrote_header = False
    # total = 0
    # with open(out_path, "w", encoding="utf-8") as out_f:
    #     for rn in run_list:
    #         if rn not in loaded:
    #             continue
    #         bounds, lines, header, mapping = loaded[rn]

    #         if not wrote_header and header:
    #             out_f.writelines(header)
    #             wrote_header = True

    #         for (img, ev) in sorted(by_run[rn], key=lambda t: (t[0], t[1])):
    #             key = (_abs(img), int(ev))
    #             cid = mapping.get(key)
    #             if cid is None:
    #                 print(f"[WARN] No chunk for ({img}, event {ev}) in run_{rn:03d}")
    #                 continue
    #             a, b = bounds[cid]
    #             out_f.writelines(lines[a:b])
    #             total += 1

    # print(f"[early-break] Selected {len(picks)} groups; wrote {total} chunk(s) to {out_path}")
    # return out_path

    # -------------------- Build new chunks in memory --------------------
    # Collect the selected chunks' text per (abs_image_path, event).
    new_chunks: Dict[HeaderKey, List[str]] = {}
    first_loaded_header: List[str] = []
    for rn in run_list:
        if rn not in loaded:
            continue
        bounds, lines, header, mapping = loaded[rn]
        if not first_loaded_header and header:
            first_loaded_header = header
        for (img, ev) in sorted(by_run[rn], key=lambda t: (t[0], t[1])):
            key = (_abs(img), int(ev))
            cid = mapping.get(key)
            if cid is None:
                print(f"[WARN] No chunk for ({img}, event {ev}) in run_{rn:03d}")
                continue
            a, b = bounds[cid]
            new_chunks[key] = lines[a:b]

    # -------------------- Load existing early_break (if any) --------------------
    existing_bounds: List[Tuple[int, int]] = []
    existing_lines: List[str] = []
    existing_header: List[str] = []
    existing_mapping: Dict[HeaderKey, int] = {}

    if os.path.isfile(out_path):
        try:
            existing_bounds, existing_lines, existing_header = parse_stream_chunks(out_path)
            existing_mapping = map_chunk_ids_by_image_event(existing_lines, existing_bounds)
        except Exception as e:
            print(f"[WARN] Failed to parse existing {out_path}: {e}")

    # -------------------- Merge & write --------------------
    # Policy:
    # - Header: keep existing header if present; otherwise use header from first loaded run.
    # - For each existing chunk (in its current order):
    #       if a new chunk for that (image,event) exists -> write the new chunk (replace)
    #       else -> keep the existing chunk
    # - After that, append any remaining new chunks (those that didn't exist before),
    #       in deterministic order by (abs_image_path, event).
    wrote_header = False
    total_written = 0
    replaced = 0
    appended = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        header_to_write = existing_header if existing_header else first_loaded_header
        if header_to_write:
            out_f.writelines(header_to_write)
            wrote_header = True

        # Write existing chunks, replacing where we have a new version
        if existing_bounds and existing_lines:
            for cid, (a, b) in enumerate(existing_bounds):
                # Extract this existing chunk's (image,event) by scanning its lines
                img_path = None
                ev = None
                for ln in existing_lines[a:b]:
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
                key = (_abs(img_path), int(ev)) if (img_path is not None and ev is not None) else None

                if key and key in new_chunks:
                    out_f.writelines(new_chunks.pop(key))  # replace
                    replaced += 1
                else:
                    out_f.writelines(existing_lines[a:b])   # keep
                total_written += 1

        # Append remaining new chunks (those not in the existing file)
        if new_chunks:
            for key in sorted(new_chunks.keys(), key=lambda t: (t[0], t[1])):
                out_f.writelines(new_chunks[key])
                total_written += 1
                appended += 1

    print(
        f"[early-break] Merge complete: wrote {total_written} chunk(s) to {out_path} "
        f"(replaced {replaced}, appended {appended})."
    )

# ------------------------------ CLI --------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="Build early_break.stream from image_run_log.csv (last-row-min wRMSD), optimized for large streams."
    )
    ap.add_argument(
        "--run-root",
        default=DEFAULT_RUN_ROOT,
        help=f"Path containing image_run_log.csv and run_*/ (default: {DEFAULT_RUN_ROOT})",
    )
    ap.add_argument(
        "--log-name",
        default=DEFAULT_LOG_NAME,
        help=f"Log filename inside run-root (default: {DEFAULT_LOG_NAME})",
    )
    ap.add_argument(
        "--out-name",
        default=DEFAULT_OUT_NAME,
        help=f"Output filename inside run-root (default: {DEFAULT_OUT_NAME})",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel processes to parse streams (default: CPU count)",
    )

    args = ap.parse_args(argv)
    run_root = _abs(args.run_root)
    build_early_break(run_root, args.log_name, args.out_name, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
