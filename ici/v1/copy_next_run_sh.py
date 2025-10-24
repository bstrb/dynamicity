#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_next_run_sh.py

Duplicate the previous run's .sh script and update the -i (input .lst)
and -o (output .stream) paths for the next run number.
Keeps all other parameters identical.
""" 
import os, re, shutil, sys

DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_RUN = 3
# DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"

def copy_and_update_sh(run_root):
    runs_dir = os.path.join(run_root, "runs")
    # last_n = find_latest_run(runs_dir)
    last_n = DEFAULT_RUN-1
    if last_n < 0:
        print("ERROR: No run_### folders found.")
        sys.exit(2)

    next_n = last_n + 1
    prev_dir = os.path.join(runs_dir, f"run_{last_n:03d}")
    next_dir = os.path.join(runs_dir, f"run_{next_n:03d}")
    os.makedirs(next_dir, exist_ok=True)

    # Find the .sh file inside the previous run
    prev_sh = None
    for f in os.listdir(prev_dir):
        if f.endswith(".sh"):
            prev_sh = os.path.join(prev_dir, f)
            break
    if not prev_sh:
        print(f"ERROR: No .sh file found in {prev_dir}")
        sys.exit(2)

    new_sh = os.path.join(next_dir, f"sh_{next_n:03d}.sh")

    with open(prev_sh, "r", encoding="utf-8") as f:
        text = f.read()

    # Replace all occurrences of run_### consistently
    pattern_i = re.compile(rf"(-i\s+.*/runs/run_{last_n:03d}/lst_{last_n:03d}\.lst)")
    pattern_o = re.compile(rf"(-o\s+.*/runs/run_{last_n:03d}/stream_{last_n:03d}\.stream)")

    text_new = text
    text_new = re.sub(pattern_i,
                      f"-i {run_root}/runs/run_{next_n:03d}/lst_{next_n:03d}.lst",
                      text_new)
    text_new = re.sub(pattern_o,
                      f"-o {run_root}/runs/run_{next_n:03d}/stream_{next_n:03d}.stream",
                      text_new)

    with open(new_sh, "w", encoding="utf-8") as f:
        f.write(text_new)

    # Copy execute permissions from previous
    st = os.stat(prev_sh)
    os.chmod(new_sh, st.st_mode)

    print(f"[copy-sh] Created {new_sh} from {prev_sh}")
    return new_sh

def main():
    run_root = os.path.abspath(os.path.expanduser(DEFAULT_ROOT))
    copy_and_update_sh(run_root)

if __name__ == "__main__":
    main()
