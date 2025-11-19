#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_early_break_from_log.py

New implementation using the sidecar image_run_state.json created
by propose_next_shifts.py.

Builds two streams:

  1) early_break.stream
     - For every (image,event) with at least one finite wRMSD, pick the run
       where that event reached its BEST (minimum) wRMSD.
     - Extract the corresponding chunk from stream_RRR.stream and write it.
     - This is what the orchestrator later renames to done.stream.

  2) only_done_events.stream
     - Same best-wRMSD rule, but only for events whose latest_status
       == ('done','done') in the sidecar.

Notes:
  - image_run_log.csv is no longer parsed; state comes from image_run_state.json.
  - If the sidecar is missing or empty, nothing is built and any existing
    stream files are left untouched.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List, Optional

HeaderKey = Tuple[str, int]  # (abs_image_path, event_id)

DEFAULT_LOG_NAME = "image_run_log.csv"     # kept for CLI compatibility (unused)
DEFAULT_OUT_NAME = "early_break.stream"    # name used by orchestrator
STATE_NAME = "image_run_state.json"
ONLY_DONE_NAME = "only_done_events.stream"

# ----------------------------- Helpers -----------------------------

def _abs(s: str) -> str:
    return os.path.abspath(os.path.expanduser(s.strip()))

def run_stream_path(run_root: str, run_n: int) -> str:
    return os.path.join(run_root, f"run_{run_n:03d}", f"stream_{run_n:03d}.stream")

def load_state(state_path: str) -> Dict:
    """
    Load image_run_state.json as produced by propose_next_shifts.py.
    Falls back to an empty state if missing or corrupted.
    """
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            raise ValueError("state not a dict")
        if "events" not in state:
            state["events"] = {}
        if "last_global_run" not in state:
            state["last_global_run"] = -1
        return state
    except FileNotFoundError:
        print(f"[early-break] WARNING: no sidecar at {state_path}, nothing to build.")
        return {"last_global_run": -1, "events": {}}
    except Exception as e:
        print(f"[early-break] WARNING: could not read {state_path} ({e}), treating as empty.")
        return {"last_global_run": -1, "events": {}}

# --------------------- Stream Parsing / Indexing -------------------

def parse_stream_chunks(stream_path: str):
    """
    Return (bounds, lines, header) for a CrystFEL .stream file.

    bounds: list[(start_idx, end_idx)] for each chunk
    lines:  list of all lines in the file
    header: lines before the first "Begin chunk"
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
    """
    Derive (abs_image_path, event_id) from a chunk's lines.
    """
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
    """
    Build mapping from (abs_path,event) to chunk index.
    """
    mapping: Dict[HeaderKey, int] = {}
    for cid, (a, b) in enumerate(bounds):
        key = chunk_key_from_slice(lines[a:b])
        if key is not None:
            mapping[key] = cid
    return mapping

def extract_chunks_for_run(run_root: str, rn: int, keys: List[HeaderKey]) -> Dict[HeaderKey, str]:
    """
    Worker: open stream_{rn}.stream once, build index, and return {key: chunk_text}.
    """
    stream_path = run_stream_path(run_root, rn)
    if not os.path.isfile(stream_path):
        return {}

    bounds, lines, _ = parse_stream_chunks(stream_path)
    index = map_chunk_ids_by_image_event(lines, bounds)
    out: Dict[HeaderKey, str] = {}
    for key in keys:
        cid = index.get(key)
        if cid is not None:
            a, b = bounds[cid]
            out[key] = "".join(lines[a:b])
    return out

# --------------------------- Plan building -------------------------

def build_plans_from_state(events_state: Dict[str, Dict]) -> Tuple[Dict[HeaderKey,int], Dict[HeaderKey,int]]:
    """
    From image_run_state["events"], build:
      - plans_all:  {HeaderKey -> best_run_n} for all events with ≥1 finite wRMSD
      - plans_done: {HeaderKey -> best_run_n} but only for events whose
                    latest_status == ('done','done')

    Here:
      - state key is "abs_path::event_id"
      - trials: [ [run, dx, dy, idx, wr], ... ]
    """
    plans_all: Dict[HeaderKey, int] = {}
    plans_done: Dict[HeaderKey, int] = {}

    for k_str, ev_state in events_state.items():
        trials = ev_state.get("trials", [])
        latest_status = ev_state.get("latest_status", ["", ""])

        if not trials:
            continue

        # state key → HeaderKey (abs_path, event_id)
        try:
            path_str, ev_str = k_str.rsplit("::", 1)
            img_path = _abs(path_str)
            ev_id = int(ev_str)
        except Exception:
            continue
        key: HeaderKey = (img_path, ev_id)

        # collect finite wRMSDs
        finite = [(rn, wr) for (rn, _dx, _dy, _idx, wr) in trials if wr is not None]
        if not finite:
            continue

        # find best (minimum) wRMSD and its run
        best_run, best_wr = min(finite, key=lambda t: t[1])

        # record in all-plans
        plans_all[key] = int(best_run)

        # done-only?
        nx, ny = latest_status
        if str(nx).lower() == "done" and str(ny).lower() == "done":
            plans_done[key] = int(best_run)

    return plans_all, plans_done

# --------------------------- Stream builder ------------------------

def build_stream_for_plans(
    run_root: str,
    plans: Dict[HeaderKey, int],
    out_path: str,
    workers: int,
    label: str,
) -> str:
    """
    Given plans {HeaderKey -> best_run_n}, build a new stream file at out_path
    containing:
      - header cloned from any contributing run's stream file
      - one chunk for each HeaderKey (from the selected run)

    The file is rebuilt from scratch.
    """
    out_path = _abs(out_path)
    out_tmp = out_path + ".tmp"

    if not plans:
        if os.path.isfile(out_path):
            print(f"[early-break] No events to write for {label}; leaving existing {os.path.basename(out_path)}.")
            return out_path
        else:
            print(f"[early-break] No events to write for {label}; nothing created.")
            return out_path

    # Group planned updates by run number
    plans_by_run: Dict[int, List[HeaderKey]] = {}
    for key, rn in plans.items():
        plans_by_run.setdefault(int(rn), []).append(key)

    if workers is None or workers <= 0:
        workers = max(1, multiprocessing.cpu_count())
    w = min(workers, max(1, len(plans_by_run)))
    print(f"[multi] Building {os.path.basename(out_path)} ({label}) with {w} workers "
          f"for {len(plans)} event(s) across {len(plans_by_run)} run(s).")

    new_chunks_text: Dict[HeaderKey, str] = {}

    # --- parallel extraction per run ---
    with ProcessPoolExecutor(max_workers=w) as ex:
        futs = []
        for rn, keys in plans_by_run.items():
            futs.append(ex.submit(extract_chunks_for_run, run_root, rn, keys))

        for fut, (rn, keys) in zip(futs, plans_by_run.items()):
            try:
                res = fut.result()
                new_chunks_text.update(res)
            except Exception as e:
                print(f"[WARN] Worker failed for run_{rn:03d}: {e}")

    # Borrow header from any contributing stream
    header_to_write: List[str] = []
    try:
        any_rn = next(iter(plans_by_run.keys()))
        sp = run_stream_path(run_root, any_rn)
        _, _, hdr = parse_stream_chunks(sp)
        if hdr:
            header_to_write = hdr
    except Exception as e:
        print(f"[WARN] Could not borrow header from run streams ({e}); writing chunks only.")

    # Now write new stream file from scratch
    total_chunks = 0
    with open(out_tmp, "w", encoding="utf-8") as wf:
        if header_to_write:
            wf.writelines(header_to_write)

        # sort keys for deterministic ordering
        pending_keys = list(plans.keys())
        for key in sorted(pending_keys, key=lambda t: (t[0], t[1])):
            text = new_chunks_text.get(key)
            if text is None:
                continue
            wf.write(text)
            total_chunks += 1

    os.replace(out_tmp, out_path)
    print(f"[early-break] Wrote {total_chunks} chunk(s) to {out_path} ({label})")
    return out_path

# ------------------------------ CLI --------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description=(
            "Build early_break.stream (best wRMSD per event, all events)\n"
            "and only_done_events.stream (best per event, done events only)\n"
            "from image_run_state.json."
        )
    )
    ap.add_argument("--run-root", required=True,
                    help="Path to 'runs/' containing run_*/ and image_run_state.json")
    ap.add_argument("--log-name", default=DEFAULT_LOG_NAME,
                    help=f"Kept for compatibility; ignored (default: {DEFAULT_LOG_NAME})")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                    help=f"Output name for ALL-best stream (default: {DEFAULT_OUT_NAME})")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of worker processes (default: all cores)")
    args = ap.parse_args(argv)

    run_root = _abs(args.run_root)
    state_path = os.path.join(run_root, STATE_NAME)
    state = load_state(state_path)
    events_state = state.get("events", {})

    if not events_state:
        print("[early-break] No events in state; nothing to build.")
        return 0

    plans_all, plans_done = build_plans_from_state(events_state)

    # early_break.stream (all events, best wrmsd per event)
    all_path = os.path.join(run_root, args.out_name)
    build_stream_for_plans(run_root, plans_all, all_path, workers=args.workers, label="all events")

    # only_done_events.stream (subset for done events)
    done_only_path = os.path.join(run_root, ONLY_DONE_NAME)
    build_stream_for_plans(run_root, plans_done, done_only_path, workers=args.workers, label="done-only")

    return 0


if __name__ == "__main__":
    sys.exit(main())
