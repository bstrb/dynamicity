#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_early_break_from_log.py  —  MP (per-run workers, no per-chunk temp explosion)

Policy (incremental, improvement-only):
  - For each (image,event) group, consider the latest row in image_run_log.csv.
  - If latest finite wRMSD < best prior finite wRMSD: REPLACE that group's chunk.
  - If no prior finite wRMSD and latest is finite: APPEND that group's chunk.
  - Otherwise: NO CHANGE for that group.
If nothing changes, the output file is left untouched.

Multiprocessing (safe & efficient):
  - Group work by RUN number.
  - Each worker opens **one** stream file (stream_RRR.stream), builds an index,
    extracts **all** requested chunks for that run, and returns a mapping
    {key: chunk_text}.
  - Avoids thousands of temp files and repeated re-reading of the same stream.

CLI:
  --run-root   path to 'runs/' directory (containing image_run_log.csv and run_*/)
  --log-name   name of csv (default: image_run_log.csv)
  --out-name   output stream name (default: early_break.stream)
  --workers    number of worker processes (default: os.cpu_count())
  --spill-to-disk  (optional) if set, workers write ONE temp file per run and return its path,
                   further reducing RAM use while keeping temp files bounded by #runs.

Example:
  python build_early_break_from_log.py --run-root /data/exp/runs --workers 8
"""

from __future__ import annotations

import argparse
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
    out: Dict[HeaderKey, int] = {}
    for key, rows in groups.items():
        if not rows:
            continue
        latest = rows[-1]
        w_latest = latest["wrmsd"]
        if w_latest is None:
            continue

        prior = [r["wrmsd"] for r in rows[:-1] if r["wrmsd"] is not None]
        if not prior:
            out[key] = latest["run_n"]
            continue
        min_prior = min(prior)
        if w_latest < (min_prior - EPS):
            out[key] = latest["run_n"]
    return out


# --------------------- Stream Parsing / Indexing -------------------

def parse_stream_chunks(stream_path: str):
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


def extract_chunks_for_run(run_root: str, rn: int, keys: List[HeaderKey]):
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


def extract_chunks_for_run_spill(run_root: str, rn: int, keys: List[HeaderKey], temp_dir: str):
    """
    Worker variant: write a single temp file per RUN (not per chunk), and return
    a manifest {key: (tmp_path, offset, length)}. Main process concatenates from these.
    """
    stream_path = run_stream_path(run_root, rn)
    if not os.path.isfile(stream_path):
        return {}

    bounds, lines, _ = parse_stream_chunks(stream_path)
    index = map_chunk_ids_by_image_event(lines, bounds)

    tmp_path = os.path.join(temp_dir, f"run_{rn:03d}.chunks")
    with open(tmp_path, "wb") as wf:
        manifest = {}
        pos = 0
        for key in keys:
            cid = index.get(key)
            if cid is None:
                continue
            a, b = bounds[cid]
            data = "".join(lines[a:b]).encode("utf-8")
            wf.write(data)
            manifest[key] = (tmp_path, pos, len(data))
            pos += len(data)
    return manifest


# --------------------------- Main Builder -------------------------

def build_early_break_incremental(
    run_root: str,
    log_name: str = DEFAULT_LOG_NAME,
    out_name: str = DEFAULT_OUT_NAME,
    workers: Optional[int] = None,
    spill_to_disk: bool = False,
) -> str:

    run_root = _abs(run_root)
    log_path = os.path.join(run_root, log_name)
    out_path = os.path.join(run_root, out_name)
    out_tmp = out_path + ".tmp"

    if not os.path.isfile(log_path):
        raise SystemExit(f"[ERR] No log at {log_path}")

    groups = parse_image_run_log(log_path)
    plans = decide_updates(groups)  # {key -> latest_run}

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

    existing_map: Dict[HeaderKey, Tuple[int,int]] = {}
    for (a, b) in existing_bounds:
        k = chunk_key_from_slice(existing_lines[a:b])
        if k is not None:
            existing_map[k] = (a, b)

    # Group planned updates by run number
    plans_by_run: Dict[int, List[HeaderKey]] = {}
    for key, rn in plans.items():
        plans_by_run.setdefault(rn, []).append(key)

    # Resolve new/updated chunks in parallel per-run
    if workers is None or workers <= 0:
        workers = max(1, multiprocessing.cpu_count())
    w = min(workers, max(1, len(plans_by_run)))
    if plans_by_run:
        print(f"[multi] Using {w} workers for {sum(len(v) for v in plans_by_run.values())} chunk(s) across {len(plans_by_run)} run(s)")

    new_chunks_text: Dict[HeaderKey, str] = {}        # default in-memory
    new_chunks_manifest: Dict[HeaderKey, Tuple[str,int,int]] = {}  # when spill_to_disk=True

    temp_dir = tempfile.mkdtemp(prefix="early_break_", dir=run_root) if spill_to_disk else None
    try:
        if plans_by_run:
            with ProcessPoolExecutor(max_workers=w) as ex:
                futs = []
                if spill_to_disk:
                    for rn, keys in plans_by_run.items():
                        futs.append(ex.submit(extract_chunks_for_run_spill, run_root, rn, keys, temp_dir))
                else:
                    for rn, keys in plans_by_run.items():
                        futs.append(ex.submit(extract_chunks_for_run, run_root, rn, keys))

                for fut, (rn, keys) in zip(futs, plans_by_run.items()):
                    try:
                        res = fut.result()
                        if spill_to_disk:
                            # res: {key: (tmp_path, offset, length)}
                            new_chunks_manifest.update(res)
                        else:
                            # res: {key: text}
                            new_chunks_text.update(res)
                    except Exception as e:
                        print(f"[WARN] Worker failed for run_{rn:03d}: {e}")

        # Nothing to change?
        if not new_chunks_text and not new_chunks_manifest:
            print("[early-break] No improvements or first-success rows detected; leaving file unchanged.")
            return out_path

        # Decide header
        header_to_write = existing_header
        if not header_to_write:
            # Borrow header from any contributing stream
            try:
                any_rn = next(iter(plans_by_run.keys()))
                sp = run_stream_path(run_root, any_rn)
                _, _, hdr = parse_stream_chunks(sp)
                if hdr:
                    header_to_write = hdr
            except Exception:
                pass

        replaced = 0
        appended = 0
        total_chunks = 0

        with open(out_tmp, "w", encoding="utf-8") as wf:
            if header_to_write:
                wf.writelines(header_to_write)

            # 1) existing chunks (replace where needed)
            if existing_bounds and existing_lines:
                for (a, b) in existing_bounds:
                    key = chunk_key_from_slice(existing_lines[a:b])
                    if key is None:
                        wf.writelines(existing_lines[a:b])
                        total_chunks += 1
                        continue

                    if spill_to_disk and key in new_chunks_manifest:
                        path, off, ln = new_chunks_manifest.pop(key)
                        with open(path, "rb") as rf:
                            rf.seek(off)
                            wf.write(rf.read(ln).decode("utf-8"))
                        replaced += 1
                    elif (not spill_to_disk) and key in new_chunks_text:
                        wf.write(new_chunks_text.pop(key))
                        replaced += 1
                    else:
                        wf.writelines(existing_lines[a:b])
                    total_chunks += 1

            # 2) append remaining new chunks (first successes)
            pending_keys = list(new_chunks_manifest.keys()) if spill_to_disk else list(new_chunks_text.keys())
            for key in sorted(pending_keys, key=lambda t: (t[0], t[1])):
                if spill_to_disk:
                    path, off, ln = new_chunks_manifest[key]
                    with open(path, "rb") as rf:
                        rf.seek(off)
                        wf.write(rf.read(ln).decode("utf-8"))
                else:
                    wf.write(new_chunks_text[key])
                appended += 1
                total_chunks += 1

        os.replace(out_tmp, out_path)
        print(f"[early-break] Merge complete: wrote {total_chunks} chunk(s) to {out_path} "
              f"(replaced {replaced}, appended {appended}).")
        return out_path

    finally:
        if temp_dir:
            try: shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception: pass


# ------------------------------ CLI --------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(
        description="Incrementally update early_break.stream (improvement-only) with parallel per-run extraction."
    )
    ap.add_argument("--run-root", required=True,
                    help="Path to 'runs/' containing image_run_log.csv and run_*/")
    ap.add_argument("--log-name", default=DEFAULT_LOG_NAME,
                    help=f"Log filename (default: {DEFAULT_LOG_NAME})")
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME,
                    help=f"Output stream filename (default: {DEFAULT_OUT_NAME})")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of worker processes (default: all cores)")
    ap.add_argument("--spill-to-disk", action="store_true",
                    help="Use ONE temp file per run (instead of RAM) to hold extracted chunks")
    args = ap.parse_args(argv)

    run_root = _abs(args.run_root)
    build_early_break_incremental(run_root, args.log_name, args.out_name,
                                  workers=args.workers, spill_to_disk=args.spill_to_disk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
