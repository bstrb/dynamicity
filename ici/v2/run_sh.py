#     sys.exit(main(sys.argv[1:]))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_sh.py

Usage:
  python run_sh.py [--run-root <path>] [--run <run>]

Defaults (no args):
  RUN ROOT = "/home/bubl3932/files/ici_trials"
  RUN      = "000"
  RUN DIR  = <RUN ROOT>/runs/run_<RUN>
  SHELL    = /bin/bash
"""

import argparse, os, sys, subprocess
from pathlib import Path

DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_RUN = "001"


def build_ap():
    ap = argparse.ArgumentParser(
        description="Run sh_<run>.sh (indexamajig) for the selected run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--run-root",
        help='Root folder that contains runs/run_<run> (e.g., ".../sim_004")',
        default=None,
    )
    ap.add_argument(
        "--run",
        help='Run identifier (e.g., "000", "3", "12"). Will be zero-padded to width 3.',
        default=None,
    )
    return ap


def normalize_run(run: str) -> str:
    """Return zero-padded run string (e.g. '0' -> '000')."""
    try:
        return f"{int(run):03d}"
    except (TypeError, ValueError):
        # If not numeric, return as-is (e.g. custom labels)
        return str(run)


def run_script(run_dir: str, run: str) -> int:
    """Execute sh_<run>.sh inside run_dir using /bin/bash. Capture stdout/stderr."""
    sh = os.path.join(run_dir, f"sh_{run}.sh")
    if not os.path.isfile(sh):
        print(f"ERROR: not found: {sh}", file=sys.stderr)
        return 2

    out = os.path.join(run_dir, "idx.stdout")
    err = os.path.join(run_dir, "idx.stderr")

    print(f"Running: {sh}")
    with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
        proc = subprocess.run(["/bin/bash", sh], stdout=fo, stderr=fe, cwd=run_dir)

    print(f"Return code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"WARNING: non-zero return (see {err})")
    return proc.returncode

def main(argv):
    ap = build_ap()
    args = ap.parse_args(argv)

    # Resolve parameters: use defaults if not provided
    run_root = args.run_root if args.run_root else DEFAULT_ROOT
    run = normalize_run(args.run if args.run else DEFAULT_RUN)

    # Construct run directory from root and run
    run_dir = os.path.join(run_root, f"run_{run}")
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run root : {os.path.abspath(os.path.expanduser(run_root))}")
    print(f"Run      : {run}")
    print(f"Run dir  : {run_dir}")

    rc = run_script(run_dir, run)
    return rc

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
