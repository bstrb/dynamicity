#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_stream_paths.py

Rewrites 'Image filename:' lines in a .stream so they point to the *original* .h5
using overlay_to_original.json produced by build_overlays_and_list.py.

Behavior:
  - If run with **no arguments**, use hardcoded defaults:
      DEFAULT_ROOT and run (below).
      Run directory assumed to be: <DEFAULT_ROOT>/runs/run_<run>/
      Input:  stream_<run>.stream
      Output: stream_<run>_fixed.stream
  - If arguments are provided:
      Provide either --run-dir OR --run-root.
        * --run-dir points directly to runs/run_XXX/
        * --run-root points to the experiment root containing 'runs/',
          and this script uses the *hardcoded* 'run' variable to select run_XXX.

Example:
  python3 fix_stream_paths.py                # uses defaults
  python3 fix_stream_paths.py --run-root /data/sim_004
  python3 fix_stream_paths.py --run-dir /data/sim_004/runs/run_001
"""

import os, sys, json, argparse

# --- Hardcoded trial defaults (as requested) ---
# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_RUN = "003"
run = DEFAULT_RUN
DEFAULT_RUN_DIR = os.path.join(os.path.abspath(DEFAULT_ROOT), "runs", f"run_{run}")
# ------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fix 'Image filename' paths in an indexamajig .stream using overlay_to_original.json")
    ap.add_argument("--run-root", help="Experiment root containing 'runs/' (uses hardcoded 'run' for run_XXX)")
    ap.add_argument("--run-dir", help="Path directly to runs/run_XXX/")
    args = ap.parse_args(argv)

    # Decide run_dir + filenames based on whether arguments were provided
    if argv is None:
        # When called as a script without explicit argv, sys.argv[1:] is used by argparse;
        # to detect 'no args', re-check sys.argv length here.
        no_args = (len(sys.argv) <= 1)
    else:
        no_args = (len(argv) == 0)

    if no_args:
        run_dir = DEFAULT_RUN_DIR
    else:
        if args.run_dir:
            run_dir = os.path.abspath(args.run_dir)
        elif args.run_root:
            run_dir = os.path.join(os.path.abspath(args.run_root), "runs", f"run_{run}")
        else:
            print("Provide --run-root or --run-dir, or run with no args for defaults.", file=sys.stderr)
            return 2

    # Inputs/outputs derived from run_dir and hardcoded 'run'
    map_json = os.path.join(run_dir, "overlay_to_original.json")
    in_stream = os.path.join(run_dir, f"stream_{run}.stream")
    # out_stream = os.path.join(run_dir, f"stream_{run}_fixed.stream")
    out_stream = in_stream

    if not os.path.isfile(map_json):
        print(f"ERROR: mapping file not found: {map_json}", file=sys.stderr)
        return 2
    if not os.path.isfile(in_stream):
        print(f"ERROR: stream file not found: {in_stream}", file=sys.stderr)
        return 2

    with open(map_json, "r", encoding="utf-8") as f:
        ov2orig = json.load(f)

    # normalize keys to absolute paths
    ov2orig_norm = { os.path.abspath(k): v for k, v in ov2orig.items() }

    def swap_path(line: str) -> str:
        if not line.startswith("Image filename:"):
            return line
        path = line.split(":", 1)[1].strip()
        key = os.path.abspath(path)
        new = ov2orig_norm.get(key)
        if not new:
            return line
        return f"Image filename: {new}\n"

    with open(in_stream, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    with open(out_stream, "w", encoding="utf-8") as fout:
        fout.writelines(swap_path(ln) for ln in lines)

    print(f"[stream-fix] Run dir: {run_dir}")
    print(f"[stream-fix] Mapping used: {map_json}")
    print(f"[stream-fix] Updated stream written to: {out_stream}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
