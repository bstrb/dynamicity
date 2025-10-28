#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ici_orchestrator.py

Strict orchestration per requested loop:
When runs exist (including after first init), iterate:
  1) propose_next_shifts.py
  2) if all next_* == done for latest run -> break
  3) else:
       build_overlays_and_list.py
       (re)detect latest run in folder
       copy_next_run_sh.py on latest
       run_sh.py on latest
       fix_stream_paths.py on latest (inplace)
       evaluate_stream.py on latest
       update_image_run_log_grouped.py
       build_early_break_from_log.py
"""
from glob import glob
import argparse, os, re, subprocess, sys
import time, datetime, threading
from typing import List, Tuple

# -------- Default config MacOS (applies ONLY when run with NO CLI args) --------
# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_004"
# DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300-VIII.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/MFM300-VIII.cell"
# DEFAULT_H5   = DEFAULT_ROOT + "/sim.h5"

# -------- Default config WSL (applies ONLY when run with NO CLI args) --------
DEFAULT_ROOT = "/home/bubl3932/files/ici_trials"
DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300.geom"
DEFAULT_CELL = DEFAULT_ROOT + "/MFM300.cell"
DEFAULT_H5   = DEFAULT_ROOT + "/MFM300.h5"

# DEFAULT_ROOT = "/home/bubl3932/files/UOX1"
# DEFAULT_GEOM = DEFAULT_ROOT + "/UOX.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/UOX.cell"
# DEFAULT_H5   = DEFAULT_ROOT + "/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5"

# Propose Next Shifts defaults
# Expanding ring search parameters
R_MAX_DEFAULT = 0.06                # in mm 1,07 pixels with 17 857 resolution
R_STEP_DEFAULT = 0.02               # in mm about 0,36 pixels with 17 857 resolution
K_BASE_DEFAULT = 20.0               # controls initial number of evals per radius, decrease for sparser sampling
SEED_DEFAULT = 1337                 # for reproducibility
CONVERGE_TOL_DEFAULT = 1e-4         # increase to stop when moves are tiny try 5e-4 or 1e-3 (in mm).
MAX_ITERS_DEFAULT = 100             # safety cap on max iterations

# BO parameters
BO_LENGTHSCALE_X_DEFAULT = 0.02     # in mm about 0.36 pixels with 17 857 resolution. increase for smoother surrogate,
BO_LENGTHSCALE_Y_DEFAULT = 0.02     # i.e less exploration. Try 0.03–0.05 if your wRMSD surface is gentle.

BO_NOISE_DEFAULT = 1e-4             # in wrmsd^2 units
BO_CANDIDATES_DEFAULT = 800         # pick greedier steps to converge in fewer iterations i.e 200–400 instead of 800–1000.
BO_EI_EPS_DEFAULT = 1e-3            # to stop earlier. Try 2e-3 or 5e-3 instead of 1e-3.
BO_MAX_EVALS_LOCAL_DEFAULT = 40     # number of BO evaluations per iteration

# Quick recipes
# Finish ultra-fast (may miss tiny gains):
# --bo-ei-eps 5e-3 --converge-tol 1e-3 --bo-candidates 200 --bo-bounds-frac 0.5 --bo-lengthscale-x 0.04 --bo-lengthscale-y 0.04

# Balanced:
# --bo-ei-eps 3e-3 --converge-tol 5e-4 --bo-candidates 400 --bo-bounds-frac 0.75 --bo-lengthscale-x 0.03 --bo-lengthscale-y 0.03

# Thorough (slower):
# --bo-ei-eps 1e-3 --converge-tol 1e-4 --bo-candidates 1000 --bo-bounds-frac 1.0 --bo-lengthscale-x 0.02 --bo-lengthscale-y 0.02

# Default indexamajig / xgandalf / integration flags


DEFAULT_FLAGS = [
    # Peakfinding
    "--peaks=cxi",
    # "--peaks=peakfinder9",
    # "--min-snr-biggest-pix=1",
    # "--min-snr-peak-pix=6",
    # "--min-snr=1",
    # "--min-sig=11",
    # "--min-peak-over-neighbour=-inf",
    # "--local-bg-radius=3",
    # Other
    "-j", "24",
    "--min-peaks=15",
    "--tolerance=10,10,10,5",
    "--xgandalf-sampling-pitch=5",
    "--xgandalf-grad-desc-iterations=1",
    "--xgandalf-tolerance=0.02",
    "--int-radius=4,5,9",
    "--no-retry",
    "--no-half-pixel-shift",
    "--no-non-hits-in-stream",
    # "--fix-profile-radius=70000000",
    "--indexing=xgandalf",
    "--integration=rings",
]

# --- begin: tiny tee-logger and timer (Change #1) --------------------

class _Tee:
    """Mirror writes to both real stream and a file handle."""
    def __init__(self, real_stream, fh):
        self.real = real_stream
        self.fh = fh
        self._lock = threading.Lock()
    def write(self, data):
        with self._lock:
            self.real.write(data)
            self.fh.write(data)
    def flush(self):
        with self._lock:
            self.real.flush()
            self.fh.flush()

class OrchestratorRunLogger:
    """
    Context manager that:
      - creates runs/<timestamp>_orchestrator.log
      - tees stdout/stderr to that file
      - records start/end timestamps and total wall time
    """
    def __init__(self, runs_dir: str):
        os.makedirs(runs_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(runs_dir, f"{ts}_orchestrator.log")
        self._fh = None
        self._t0 = None
        self._old_out = None
        self._old_err = None

    def __enter__(self):
        self._fh = open(self.log_path, "w", encoding="utf-8")
        self._t0 = time.perf_counter()
        start_iso = datetime.datetime.now().isoformat(timespec="seconds")
        self._fh.write(f"[orchestrator] start={start_iso}\n")
        self._fh.flush()
        # tee stdout/stderr
        self._old_out, self._old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, self._fh)
        sys.stderr = _Tee(sys.stderr, self._fh)
        print(f"[orchestrator] logging to: {self.log_path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        # restore
        sys.stdout, sys.stderr = self._old_out, self._old_err
        elapsed = time.perf_counter() - self._t0
        end_iso = datetime.datetime.now().isoformat(timespec="seconds")
        self._fh.write(f"[orchestrator] end={end_iso} elapsed_sec={elapsed:.3f}\n")
        self._fh.close()
        if exc:
            print(f"[orchestrator] ERROR: {exc_type.__name__}: {exc}")
        print(f"[orchestrator] total elapsed: {elapsed:.3f}s; log: {self.log_path}")
# --- end: tiny tee-logger and timer ----------------------------------

def runs_dir(run_root: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(run_root)), "runs")

def list_run_numbers(run_root: str) -> List[int]:
    rd = runs_dir(run_root)
    if not os.path.isdir(rd):
        return []
    nums = []
    for name in os.listdir(rd):
        m = re.fullmatch(r"run_(\d{3})", name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(nums)

def latest_run(run_root: str) -> Tuple[int, str]:
    nums = list_run_numbers(run_root)
    return (nums[-1], os.path.join(runs_dir(run_root), f"run_{nums[-1]:03d}")) if nums else (-1, "")

def run_py(script: str, args: List[str], check: bool = True) -> int:
    cmd = ["python3", script, *args]
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    if check and rc != 0:
        raise SystemExit(f"[ERR] {script} exited with {rc}")
    return rc

def detect_latest_run_from_log(log_path: str) -> int:
    try:
        latest = -1
        with open(log_path, "r", encoding="utf-8") as f:
            _ = f.readline()  # header
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if parts and parts[0].isdigit():
                    v = int(parts[0])
                    if v > latest:
                        latest = v
        return latest
    except Exception:
        return -1

def all_next_done_for_latest(log_path: str, latest: int) -> bool:
    if latest < 0:
        return False
    any_row = False
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            _ = f.readline()
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 7 or not parts[0].isdigit():
                    continue
                if int(parts[0]) != latest:
                    continue
                any_row = True
                if not (parts[-2] == "done" and parts[-1] == "done"):
                    return False
        return any_row
    except FileNotFoundError:
        return False

# def do_init_sequence(run_root: str):
#     print("[phase] No runs detected -> initializing run_000")
#     run_py(
#             "no_run_prep_singlelist.py",
#             [
#                 "--run-root", run_root,
#                 "--geom", DEFAULT_GEOM,
#                 "--cell", DEFAULT_CELL,
#                 # optional: you can omit this since default is "indexamajig"
#                 "--indexamajig", "indexamajig",
#                 # positional sources (one or many)
#                 DEFAULT_H5,
#                 # everything after `--` gets forwarded as indexamajig flags
#                 "--", *DEFAULT_FLAGS,
#             ],
#         )
#     run_py("run_sh.py", ["--run-root", run_root, "--run", "000"], check=False)
#     run_py("evaluate_stream.py", ["--run-root", run_root, "--run", "000"], check=False)
#     run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
#     run_py("build_early_break_from_log.py", ["--run-root", os.path.join(run_root, "runs")])
#     print("[done] Initialization cycle complete. Proceeding to loop...")

def do_init_sequence(run_root: str, h5_sources: list):
    print("[phase] No runs detected -> initializing run_000")
    sources = []
    for s in h5_sources:
        matches = sorted(glob(s))
        sources.extend(matches if matches else [s])

    print(f"[init] using {len(sources)} HDF5 source(s):", *sources, sep="\n  ")

    run_py(
            "no_run_prep_singlelist.py",
            [
                "--run-root", run_root,
                "--geom", DEFAULT_GEOM,
                "--cell", DEFAULT_CELL,
                # optional: you can omit this since default is "indexamajig"
                "--indexamajig", "indexamajig",
                # positional sources (one or many)
                DEFAULT_H5,
                # everything after `--` gets forwarded as indexamajig flags
                "--", *DEFAULT_FLAGS,
            ],
        )
    run_py("run_sh.py", ["--run-root", run_root, "--run", "000"], check=False)
    run_py("evaluate_stream.py", ["--run-root", run_root, "--run", "000"], check=False)
    run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
    run_py("build_early_break_from_log.py", ["--run-root", os.path.join(run_root, "runs")])
    print("[done] Initialization cycle complete. Proceeding to loop...")

# def iterate_until_done(run_root: str, max_iters: int, skip_fix_stream: bool):
def iterate_until_done(run_root: str, max_iters: int):
    rd = runs_dir(run_root)
    it = 0
    while it < max_iters:
        it += 1

        # print(f"\n[loop] Iteration {it}", flush=True)

        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[loop] Iteration {it} started at {ts}", flush=True)

        # 1) propose_next_shifts.py
        run_py("propose_next_shifts.py", ["--run-root", run_root, "--r-max", str(R_MAX_DEFAULT), "--r-step", str(R_STEP_DEFAULT), "--k-base", str(K_BASE_DEFAULT), "--seed", str(SEED_DEFAULT), "--converge-tol", str(CONVERGE_TOL_DEFAULT), "--bo-lengthscale-x", str(BO_LENGTHSCALE_X_DEFAULT), "--bo-lengthscale-y", str(BO_LENGTHSCALE_Y_DEFAULT), "--bo-noise", str(BO_NOISE_DEFAULT), "--bo-candidates", str(BO_CANDIDATES_DEFAULT), "--bo-ei-eps", str(BO_EI_EPS_DEFAULT), "--bo-max-evals-local", str(BO_MAX_EVALS_LOCAL_DEFAULT)])

        # evaluate stop condition based on latest run in the *log*
        log_path = os.path.join(rd, "image_run_log.csv")
        latest_in_log = detect_latest_run_from_log(log_path)
        if latest_in_log < 0:
            print("[warn] No rows detected in image_run_log.csv; stopping.")
            break

        # # 2) if all next_* for latest are 'done' -> break
        # if all_next_done_for_latest(log_path, latest_in_log):
        #     print(f"[stop] All next_* entries for run_{latest_in_log:03d} are 'done'.")
        #     break
        # Modified 2024-06-06: rename early_break.stream → done.stream if exists, then break
        if all_next_done_for_latest(log_path, latest_in_log):
            print(f"[stop] All next_* entries for run_{latest_in_log:03d} are 'done'.")

            # rename early_break.stream → done.stream if it exists
            early_stream = os.path.join(rd, "early_break.stream")
            done_stream = os.path.join(rd, "done.stream")
            if os.path.exists(early_stream):
                os.rename(early_stream, done_stream)
                print(f"[rename] {early_stream} → {done_stream}")

            break


        # 3) else build overlays & list for next iteration
        run_py("build_overlays_and_list.py", ["--run-root", run_root])

        # Re-detect LATEST RUN IN FOLDER (as requested)
        latest_num, latest_dir = latest_run(run_root)
        if latest_num < 0:
            print("[err] No run folders found after overlays; aborting.")
            break

        run_str = f"{latest_num:03d}"

        # copy_next_run_sh.py on latest
        run_py("copy_next_run_sh.py", ["--run-root", run_root, "--run", run_str], check=False)

        # run_sh.py on latest
        run_py("run_sh.py", ["--run-root", run_root, "--run", run_str], check=False)

        # fix_stream_paths.py on latest (inplace)
        _ = run_py("fix_stream_paths.py", ["--run-dir", latest_dir, "--run", run_str, "--inplace"], check=False)

        # evaluate_stream.py on latest
        run_py("evaluate_stream.py", ["--run-root", run_root, "--run", run_str], check=False)

        # update_image_run_log_grouped.py (this is the file we have)
        run_py("update_image_run_log_grouped.py", ["--run-root", run_root])

        # build_early_break_from_log.py
        run_py("build_early_break_from_log.py", ["--run-root", os.path.join(run_root, "runs")])

    else:
        print(f"[stop] Reached max-iters={max_iters} without satisfying 'done'.")
    print("[done] Orchestration complete.")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Orchestrate SerialED iterative runs using provided helper scripts.")
    ap.add_argument("--run-root", default=DEFAULT_ROOT, help="Experiment root that contains 'runs/'.")
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS_DEFAULT, help="Safety cap on loop iterations.")
    ap.add_argument(
        "--h5", nargs="+", default=[DEFAULT_H5],
        help="One or more HDF5 sources or globs (e.g., sim_001.h5 sim_002.h5 or sim_*.h5)"
    )


    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    run_root = os.path.abspath(os.path.expanduser(args.run_root))
    os.makedirs(runs_dir(run_root), exist_ok=True)

    # --- begin: wrap whole orchestration with logger (Change #2) ---
    with OrchestratorRunLogger(runs_dir(run_root)):
        # If no runs, initialize run_000 first
        if not list_run_numbers(run_root):
            # do_init_sequence(run_root)
            do_init_sequence(run_root, args.h5)

        # Iterate until all next_* == done
        iterate_until_done(run_root, args.max_iters)
    # --- end: wrap whole orchestration with logger ------------------

if __name__ == "__main__":
    sys.exit(main())
