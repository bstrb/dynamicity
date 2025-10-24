#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_next_run_sh.py

Duplicate the previous run's .sh script and update the -i (input .lst)
and -o (output .stream) paths for the next run number.
Keeps all other parameters identical.

Usage examples:
  python3 copy_next_run_sh.py
  python3 copy_next_run_sh.py --run 0
  python3 copy_next_run_sh.py --run-root "/path/to/sim_004" --run 5
"""
import os, re, sys, argparse

# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_RUN = "000"  # will be zero-padded to width 3 at runtime


def _normalize_run(run: str) -> str:
    """Zero-pad numeric runs to width 3 (e.g. '0' -> '000'); pass through non-numeric."""
    try:
        return f"{int(run):03d}"
    except (TypeError, ValueError):
        return str(run)


def copy_and_update_sh(run_root: str, run_str: str) -> str:
    """Copy sh_<run>.sh from run_<run> to run_<run+1>, updating -i/-o paths."""
    runs_dir = os.path.join(run_root, "runs")
    try:
        last_n = int(run_str)
    except ValueError:
        print(f"ERROR: --run must be numeric for this script (got '{run_str}').", file=sys.stderr)
        sys.exit(2)

    next_n = last_n + 1
    prev_dir = os.path.join(runs_dir, f"run_{last_n:03d}")
    next_dir = os.path.join(runs_dir, f"run_{next_n:03d}")
    os.makedirs(next_dir, exist_ok=True)

    # Prefer sh_{last_n:03d}.sh if present; else first *.sh
    candidate = os.path.join(prev_dir, f"sh_{last_n:03d}.sh")
    prev_sh = candidate if os.path.isfile(candidate) else None
    if prev_sh is None:
        for f in os.listdir(prev_dir):
            if f.endswith(".sh"):
                prev_sh = os.path.join(prev_dir, f)
                break
    if not prev_sh:
        print(f"ERROR: No .sh file found in {prev_dir}", file=sys.stderr)
        sys.exit(2)

    new_sh = os.path.join(next_dir, f"sh_{next_n:03d}.sh")

    with open(prev_sh, "r", encoding="utf-8") as f:
        text = f.read()

    # Replace the input/output paths for the next run
    # Keep the -i/-o flags intact and only swap the path following them.
    pattern_i = re.compile(rf"(\s-i\s+).*/runs/run_{last_n:03d}/lst_{last_n:03d}\.lst\b")
    pattern_o = re.compile(rf"(\s-o\s+).*/runs/run_{last_n:03d}/stream_{last_n:03d}\.stream\b")

    text_new = text
    text_new = re.sub(
        pattern_i,
        rf"\1{run_root}/runs/run_{next_n:03d}/lst_{next_n:03d}.lst",
        text_new,
    )
    text_new = re.sub(
        pattern_o,
        rf"\1{run_root}/runs/run_{next_n:03d}/stream_{next_n:03d}.stream",
        text_new,
    )

    with open(new_sh, "w", encoding="utf-8") as f:
        f.write(text_new)

    # Copy execute permissions from previous
    try:
        st = os.stat(prev_sh)
        os.chmod(new_sh, st.st_mode)
    except Exception as e:
        print(f"WARNING: could not copy permissions from {prev_sh} -> {new_sh}: {e}", file=sys.stderr)

    print(f"[copy-sh] Created {new_sh} from {prev_sh}")
    return new_sh


def main(argv=None):
    ap = argparse.ArgumentParser(description="Copy sh_<run>.sh to next run and update -i/-o paths.")
    ap.add_argument("--run-root", help='Root folder that contains runs/run_<run>')
    ap.add_argument("--run", help='Run identifier (e.g., "0", "3", "12", "003"); zero-padded to width 3')
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root if args.run_root else DEFAULT_ROOT))
    run_str = _normalize_run(args.run if args.run else DEFAULT_RUN)

    print(f"Run root : {run_root}")
    print(f"Run      : {run_str}  (creating next: {int(run_str):03d} -> {int(run_str)+1:03d})")
    print(f"Prev dir : {os.path.join(run_root, 'runs', f'run_{int(run_str):03d}')}")
    print(f"Next dir : {os.path.join(run_root, 'runs', f'run_{int(run_str)+1:03d}')}")

    new_sh = copy_and_update_sh(run_root, run_str)
    print(f"[copy-sh] Done: {new_sh}")


if __name__ == "__main__":
    main()
