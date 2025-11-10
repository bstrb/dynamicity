#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_stream_paths.py

Rewrite 'Image filename:' lines in an indexamajig .stream so they point to the
*original* .h5 using the overlay_to_original mapping produced by
build_overlays_and_list.py.

Now fully argparse-driven so it works for ANY run without editing the file.

USAGE EXAMPLES
--------------
# Preferred (explicit):
python3 fix_stream_paths.py --run-dir /path/to/runs/run_007 --run 007

# Or, if you keep run-root and run number separately:
python3 fix_stream_paths.py --run-root /path/to/exp_root --run 007

# Auto-detect the .stream inside a run folder (if exactly one stream_*.stream exists):
python3 fix_stream_paths.py --run-dir /path/to/runs/run_007

# Operate in-place instead of writing *_fixed.stream:
python3 fix_stream_paths.py --run-dir /path/to/runs/run_007 --run 007 --inplace
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, Tuple

# Back-compat defaults (only used when *no* CLI args are supplied)
DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_RUN = "001"  # only used if you call the script with absolutely no args

def _load_mapping(run_dir: str) -> Dict[str, str]:
    """
    Load overlay->original mapping.
    Prefer JSON; fall back to TSV if present.
    Keys are normalized to absolute paths.
    """
    json_path = os.path.join(run_dir, "overlay_to_original.json")
    tsv_path  = os.path.join(run_dir, "overlay_to_original.tsv")

    mapping: Dict[str, str] = {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # normalize keys
        mapping = { os.path.abspath(k): v for k, v in raw.items() }
        return mapping

    if os.path.isfile(tsv_path):
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                ov, orig = parts[0], parts[1]
                mapping[os.path.abspath(ov)] = orig
        return mapping

    raise FileNotFoundError(f"overlay_to_original.(json|tsv) not found in: {run_dir}")

def _detect_stream_path(run_dir: str, run: str = None) -> Tuple[str, str]:
    """
    Return (in_stream_path, run_str). If run is None, try to auto-detect a single stream_*.stream
    in the run_dir. If multiple exist and run is None, raise.
    """
    if run:
        run = f"{int(run):03d}"
        candidate = os.path.join(run_dir, f"stream_{run}.stream")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Expected stream file not found: {candidate}")
        return candidate, run

    # No run specified: try to find a single stream_*.stream
    streams = sorted(glob.glob(os.path.join(run_dir, "stream_*.stream")))
    if len(streams) == 0:
        raise FileNotFoundError(f"No stream_*.stream found in {run_dir}")
    if len(streams) > 1:
        raise RuntimeError(f"Multiple stream_*.stream files found in {run_dir}. Specify --run.")
    # Extract NNN from stream_NNN.stream
    base = os.path.basename(streams[0])
    try:
        run_guess = base.split("_", 1)[1].split(".")[0]
    except Exception:
        run_guess = "000"
    return streams[0], run_guess

def _swap_line(line: str, mapping: Dict[str, str]) -> str:
    """
    Replace the path in 'Image filename:' (or 'Image filename =') lines if present in mapping.
    """
    if not (line.startswith("Image filename:") or line.startswith("Image filename =")):
        return line
    # Split on ':' or '=' once
    if ":" in line:
        _, rhs = line.split(":", 1)
        prefix = "Image filename:"
    else:
        _, rhs = line.split("=", 1)
        prefix = "Image filename ="
    path = rhs.strip()
    key_abs = os.path.abspath(path)
    new = mapping.get(key_abs)
    if not new:
        # Also try direct string key (in case mapping stored non-abs paths)
        new = mapping.get(path)
    if not new:
        return line
    return f"{prefix} {new}\n"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fix 'Image filename' paths in an indexamajig .stream using overlay_to_original mapping.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--run-dir", help="Path to runs/run_XXX/")
    group.add_argument("--run-root", help="Experiment root containing 'runs/' (use with --run)")
    ap.add_argument("--run", help="Run number NNN (e.g. 007). Required if you use --run-root. Optional with --run-dir (auto-detects if a single stream exists).")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input stream file instead of writing *_fixed.stream")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    no_args = (len(sys.argv) <= 1) if argv is None else (len(argv) == 0)
    if no_args:
        run_dir = os.path.join(os.path.abspath(DEFAULT_ROOT), "runs", f"run_{DEFAULT_RUN}")
        in_stream, run = _detect_stream_path(run_dir, run=DEFAULT_RUN)
    else:
        if args.run_dir:
            run_dir = os.path.abspath(args.run_dir)
        else:
            if not args.run_root or not args.run:
                ap.error("When using --run-root you must also provide --run NNN")
            run_dir = os.path.join(os.path.abspath(args.run_root), "runs", f"run_{int(args.run):03d}")

        in_stream, run = _detect_stream_path(run_dir, run=args.run)

    mapping = _load_mapping(run_dir)

    if args.inplace:
        out_stream = in_stream
    else:
        base, ext = os.path.splitext(in_stream)
        out_stream = f"{base}_fixed{ext}"

    # Process the stream
    with open(in_stream, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    with open(out_stream, "w", encoding="utf-8") as fout:
        for ln in lines:
            fout.write(_swap_line(ln, mapping))

    print(f"[stream-fix] Run dir: {run_dir}")
    print(f"[stream-fix] Stream in : {in_stream}")
    print(f"[stream-fix] Stream out: {out_stream}")
    print(f"[stream-fix] Mapping  : overlay_to_original.(json|tsv)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
