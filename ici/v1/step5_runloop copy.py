#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_runloop.py

Orchestrates the iterative pipeline:
  For latest run_NNN:
    1) Run its sh_NNN.sh (indexamajig)
    2) Evaluate stream with step3_evaluate_stream.py (copy/rename trick)
    3) Prepare next run using step4_prepare_next_run_overlay_elink.py
  Repeat until no new run is created (all converged/exhausted).

Defaults (no flags):
  RUN_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
  SHELL    = "/bin/bash"
  STEP3    = "step3_evaluate_stream.py"
  STEP4    = "step4_prepare_next_run_overlay_elink.py"
"""

from __future__ import annotations
import argparse, os, sys, re, subprocess, shutil
from pathlib import Path

DEFAULT_RUN_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
DEFAULT_SHELL = "/bin/bash"
DEFAULT_STEP3 = "step3_evaluate_stream.py"
DEFAULT_STEP4 = "step4_prepare_next_run_overlay_elink.py"

def find_latest_run(run_root: str):
    runs_dir = os.path.join(run_root, "runs")
    max_n = -1; max_path = ""
    if not os.path.isdir(runs_dir):
        return -1, ""
    for name in os.listdir(runs_dir):
        m = re.match(r"^run_(\d{3})$", name)
        if not m: continue
        n = int(m.group(1))
        if n > max_n:
            max_n = n; max_path = os.path.join(runs_dir, name)
    return max_n, max_path

def run_shell_script(run_dir: str, shell: str) -> int:
    # Expect sh_NNN.sh in run_dir
    base = os.path.basename(run_dir)
    m = re.match(r"run_(\d{3})$", base)
    if not m:
        print(f"ERROR: run dir name must be run_XXX: {run_dir}", file=sys.stderr); return 2
    n = int(m.group(1))
    sh_path = os.path.join(run_dir, f"sh_{n:03d}.sh")
    if not os.path.isfile(sh_path):
        print(f"ERROR: not found: {sh_path}", file=sys.stderr); return 2
    out = os.path.join(run_dir, "idx.stdout")
    err = os.path.join(run_dir, "idx.stderr")
    print(f"[run] {sh_path}")
    with open(out, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
        proc = subprocess.run([shell, sh_path], stdout=fo, stderr=fe)
    print(f"[run] return code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"[run] WARNING non-zero return, see {err}")
    return proc.returncode

def evaluate_stream_with_step3(run_dir: str, step3_path: str) -> int:
    """
    step3_evaluate_stream.py expects stream_000.stream and writes *_000.* outputs.
    For run_NNN, we temporarily copy stream_{NNN}.stream to stream_000.stream,
    run step3, then rename outputs back to *_NNN.* names.
    """
    base = os.path.basename(run_dir)
    m = re.match(r"run_(\d{3})$", base)
    if not m:
        print(f"ERROR: run dir name must be run_XXX: {run_dir}", file=sys.stderr); return 2
    n = int(m.group(1))
    stream_src = os.path.join(run_dir, f"stream_{n:03d}.stream")
    stream_tmp = os.path.join(run_dir, "stream_000.stream")
    if not os.path.isfile(stream_src):
        print(f"ERROR: not found: {stream_src}", file=sys.stderr); return 2

    # Copy stream to expected name
    if os.path.exists(stream_tmp):
        os.remove(stream_tmp)
    shutil.copy2(stream_src, stream_tmp)

    # Call step3
    print(f"[eval] {step3_path} --run-dir {run_dir}")
    proc = subprocess.run([sys.executable, step3_path, "--run-dir", run_dir])
    if proc.returncode != 0:
        print(f"[eval] step3 returned {proc.returncode}", file=sys.stderr); return proc.returncode

    # Rename outputs to *_NNN.*
    mapping = {
        "chunk_metrics_000.csv": f"chunk_metrics_{n:03d}.csv",
        "summary_000.txt": f"summary_{n:03d}.txt",
        "parse_debug_000.txt": f"parse_debug_{n:03d}.txt",
    }
    for src, dst in mapping.items():
        s = os.path.join(run_dir, src)
        d = os.path.join(run_dir, dst)
        if os.path.exists(s):
            if os.path.exists(d): os.remove(d)
            os.replace(s, d)
            print(f"[eval] wrote {d}")
        else:
            print(f"[eval] WARNING expected {s} not found", file=sys.stderr)

    # Clean up temporary stream copy (optional keep for debugging)
    try:
        os.remove(stream_tmp)
    except Exception:
        pass

    return 0

def prepare_next_run(step4_path: str, run_root: str) -> bool:
    """
    Calls step4_prepare_next_run_overlay_elink.py. Returns True if it created a new run.
    """
    # Get latest before
    before_n, _ = find_latest_run(run_root)
    print(f"[prep] before latest run: {before_n:03d}")
    proc = subprocess.run([sys.executable, step4_path, "--run-root", run_root])
    if proc.returncode != 0:
        print(f"[prep] step4 returned {proc.returncode}", file=sys.stderr)
        # Still check if a new run was made
    after_n, _ = find_latest_run(run_root)
    print(f"[prep] after  latest run: {after_n:03d}")
    return after_n > before_n

def main(argv=None):
    ap = argparse.ArgumentParser(description="Run-Loop orchestrator for index → evaluate → prepare-next.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Root containing runs/")
    ap.add_argument("--shell", default=DEFAULT_SHELL, help="Shell to execute sh_NNN.sh")
    ap.add_argument("--step3", default=DEFAULT_STEP3, help="Path to step3_evaluate_stream.py")
    ap.add_argument("--step4", default=DEFAULT_STEP4, help="Path to step4_prepare_next_run_overlay_elink.py")
    ap.add_argument("--max-iters", type=int, default=100, help="Safety cap for iterations")
    args = ap.parse_args(argv)

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    shell = args.shell
    step3 = os.path.abspath(os.path.expanduser(args.step3))
    step4 = os.path.abspath(os.path.expanduser(args.step4))

    print("=== Step 5 Run-Loop ===")
    print(f"run_root: {run_root}")
    print(f"shell:    {shell}")
    print(f"step3:    {step3}")
    print(f"step4:    {step4}")

    iters = 0
    while iters < args.max_iters:
        iters += 1
        print(f"\n[loop] iteration {iters}")
        n, run_dir = find_latest_run(run_root)
        if n < 0:
            print("No runs found. Create run_000 first."); return 2

        # 1) Execute sh_NNN.sh for latest run
        rc = run_shell_script(run_dir, shell)
        if rc != 0:
            print("Stopping due to indexamajig failure"); return rc

        # 2) Evaluate stream with step3 (copy/rename trick)
        rc = evaluate_stream_with_step3(run_dir, step3)
        if rc != 0:
            print("Stopping due to evaluation failure"); return rc

        # 3) Prepare next run
        made = prepare_next_run(step4, run_root)
        if not made:
            print("\nAll frames converged or exhausted Rmax. Terminating loop.")
            return 0

    print("Reached max iterations without convergence. Exiting.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
