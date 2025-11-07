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

DEFAULT_MAX_ITERS = 20

# Default paths
# DEFAULT_ROOT = "/home/bubl3932/files/simulations/MFM300-VIII_tI/sim_002"
# DEFAULT_GEOM = DEFAULT_ROOT + "/4135627.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/4135627.cell"
# DEFAULT_H5   = [DEFAULT_ROOT + "/sim.h5"]

DEFAULT_ROOT = "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038"
DEFAULT_GEOM = DEFAULT_ROOT + "/MFM.geom"
DEFAULT_CELL = DEFAULT_ROOT + "/MFM.cell"
DEFAULT_H5   = [DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038.h5"]

# Default paths
# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_010"
# DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300-VIII.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/MFM300-VIII.cell"
# DEFAULT_H5   = [DEFAULT_ROOT + "/sim.h5"]

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
        self.log_path = os.path.join(runs_dir, "orchestrator.log")
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
    # run_root is already the timestamped runs folder
    return os.path.abspath(os.path.expanduser(run_root))


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
    import subprocess, sys
    cmd = ["python3", script, *args]
    print(f"[RUN] {' '.join(cmd)}", flush=True)

    # Capture child stdout/stderr, reprint to our stdout
    # so it flows through the Tee into both console + log.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    try:
        for line in proc.stdout:
            # forward exactly as the child wrote it
            print(line, end="")  # our _Tee will duplicate to file + console
    finally:
        proc.stdout.close()
    rc = proc.wait()

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

def do_init_sequence(run_root: str, geom: str, cell: str, h5_sources: list):
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
                "--geom", geom,
                "--cell", cell,
                *sources,
            ],
        )
    run_py("create_run_sh.py", ["--run-root", run_root, "--geom", geom, "--cell", cell, "--run", "000",
        "--",  # everything after this is passed directly into indexamajig
        *DEFAULT_FLAGS,], check=False)
    run_py("run_sh.py", ["--run-root", run_root, "--run", "000"], check=False)
    run_py("evaluate_stream.py", ["--run-root", run_root, "--run", "000"], check=False)
    run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
    run_py("summarize_image_run_log.py", ["--run-root", run_root,])
    run_py("build_early_break_from_log.py", ["--run-root", run_root])
    print("[done] Initialization cycle complete. Proceeding to loop...")

def iterate_until_done(run_root, max_iters=DEFAULT_MAX_ITERS):
    rd = runs_dir(run_root)
    it = 0
    while it < max_iters:
        it += 1

        # print(f"\n[loop] Iteration {it}", flush=True)

        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[loop] Iteration {it} started at {ts}", flush=True)

        # 1) propose_next_shifts.py
        run_py("propose_next_shifts.py", ["--run-root", run_root])

        # evaluate stop condition based on latest run in the *log*
        log_path = os.path.join(rd, "image_run_log.csv")
        latest_in_log = detect_latest_run_from_log(log_path)
        if latest_in_log < 0:
            print("[warn] No rows detected in image_run_log.csv; stopping.")
            break

        # # 2) if all next_* for latest are 'done' -> break
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

        # run_py("copy_next_run_sh.py", ["--run-root", run_root, "--run", run_str], check=False)
        run_py("create_run_sh.py", ["--run-root", run_root, "--geom", DEFAULT_GEOM, "--cell", DEFAULT_CELL, "--run", run_str,
        "--",  # everything after this is passed directly into indexamajig
        *DEFAULT_FLAGS], check=False)
        run_py("run_sh.py", ["--run-root", run_root, "--run", run_str], check=False)
        _ = run_py("fix_stream_paths.py", ["--run-dir", latest_dir, "--run", run_str, "--inplace"], check=False)
        run_py("evaluate_stream.py", ["--run-root", run_root, "--run", run_str], check=False)
        run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
        run_py("summarize_image_run_log.py", ["--run-root", run_root,])
        run_py("build_early_break_from_log.py", ["--run-root", run_root])

    else:
        print(f"[stop] Reached max-iters={max_iters} without satisfying 'done'.")
    print("[done] Orchestration complete.")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Orchestrate SerialED iterative runs using provided helper scripts.")
    ap.add_argument("--run-root", default=DEFAULT_ROOT, help="Experiment root that contains 'runs/'.")
    ap.add_argument("--geom", default=DEFAULT_GEOM, help="Geometry file for initialization.")
    ap.add_argument("--cell", default=DEFAULT_CELL, help="Cell file for initialization.")
    ap.add_argument("--h5", nargs="+", default=DEFAULT_H5, help="One or more HDF5 sources or globs (e.g., sim_001.h5 sim_002.h5 or sim_*.h5)")
    ap.add_argument("--flags", nargs="*", default=DEFAULT_FLAGS, help="Additional indexamajig / xgandalf / integration flags for initialization.")

    args = ap.parse_args(argv if argv is not None else sys.argv[1:])


    exp_root = os.path.abspath(os.path.expanduser(args.run_root))

    # one timestamp per orchestration
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = os.path.join(exp_root, f"runs_{ts}")
    os.makedirs(runs_dir(sess), exist_ok=True)

    # --- begin: wrap whole orchestration with logger (Change #2) ---
    with OrchestratorRunLogger(runs_dir(sess)):
        # If no runs, initialize run_000 first
        if not list_run_numbers(sess):
            # do_init_sequence(run_root)
            do_init_sequence(sess, args.geom, args.cell, args.h5)

        # Iterate until all next_* == done
        iterate_until_done(sess)

    # --- end: wrap whole orchestration with logger ------------------

if __name__ == "__main__":
    sys.exit(main())
