#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_run_only.py

Step 2: run the command script (sh_000.sh) for run_000 and return.

Defaults (no args):
  RUN ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
  RUN DIR  = <RUN ROOT>/runs/run_000
  SHELL    = /bin/bash
"""

import argparse, os, sys, subprocess
from pathlib import Path

DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_RUN_DIR = os.path.join(DEFAULT_ROOT, "runs", "run_000")
DEFAULT_SH = "sh_000.sh"

def build_ap():
    ap = argparse.ArgumentParser(description="Run sh_000.sh (indexamajig) for run_000.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-root", help="Root folder that contains runs/run_000.")
    ap.add_argument("--run-dir", help="Explicit run directory (e.g., .../runs/run_000).")
    ap.add_argument("--shell", default="/bin/bash", help="Shell to execute the .sh script.")
    return ap

def run_script(run_dir: str, shell: str) -> int:
    sh = os.path.join(run_dir, DEFAULT_SH)
    if not os.path.isfile(sh):
        print(f"ERROR: not found: {sh}", file=sys.stderr)
        return 2
    out = os.path.join(run_dir, "idx.stdout")
    err = os.path.join(run_dir, "idx.stderr")
    print(f"Running: {sh}")
    with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
        proc = subprocess.run([shell, sh], stdout=fo, stderr=fe)
    print(f"Return code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"WARNING: non-zero return (see {err})")
    return proc.returncode

def main(argv):
    ap = build_ap()
    args = ap.parse_args(argv)

    if len(argv) == 0:
        run_dir = DEFAULT_RUN_DIR
        shell = "/bin/bash"
    else:
        if args.run_dir:
            run_dir = args.run_dir
        elif args.run_root:
            run_dir = os.path.join(args.run_root, "runs", "run_000")
        else:
            print("Provide --run-root or --run-dir, or run with no args for defaults.", file=sys.stderr)
            return 2
        shell = args.shell

    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")
    rc = run_script(run_dir, shell)
    return rc

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
