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


# Default paths
# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_012"
# DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300-VIII.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/MFM300-VIII.cell"
# DEFAULT_H5   = [DEFAULT_ROOT + "/sim1.h5",
#                 DEFAULT_ROOT + "/sim2.h5",
#                 DEFAULT_ROOT + "/sim3.h5"]

DEFAULT_ROOT = "/home/bubl3932/files/MFM300_VIII/MP15_3x100"
DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300-VIII.geom"
DEFAULT_CELL = DEFAULT_ROOT + "/MFM300-VIII.cell"
DEFAULT_H5   = [DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_1712_min_15peaks_100.h5",
                DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_1822_min_15peaks_100.h5",
                DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038_min_15peaks_100.h5"]
                
DEFAULT_MAX_ITERS = 20          # maximum number of iterations
DEFAULT_NUM_CPU   = 24           # default number of parallel jobs set to os.cpu_count() for max

# Default propose next shift parameters
radius_mm        = 0.05          # search radius 
min_spacing_mm   = 5e-4         # minimum spacing between shifts
N_conv           = 3            # minimum number o events to consider convergence
recurring_tol    = 0.1          # tolerance for recurring shifts (0.1 = 10%)
median_rel_tol   = 0.1          # median relative tolerance for convergence (0.1 = 10%)
noimprove_N      = 2            # number of iterations with no improvement to consider convergence
noimprove_eps    = 0.02         # minimum improvement for noimprove trigger (0.02 = 2%)
stability_N      = 3            # number of iterations to consider for stability
stability_std    = 0.05         # standard deviation threshold for stability (0.05 = 5%)
done_on_streak_successes = 2    # number of  successes to consider done after unindexed streak
done_on_streak_length   = 5     # length of streak to consider done when at least done_on_streak_successes
λ                = 0.8          # damping factor for refinded shift updates

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
    "-j", "1",
    "--min-peaks=15",
    "--tolerance=10,10,10,5",
    "--xgandalf-sampling-pitch=5",
    "--xgandalf-grad-desc-iterations=1",
    "--xgandalf-tolerance=0.02",
    "--int-radius=4,5,9",
    "--no-retry",
    "--no-half-pixel-shift",
    "--no-non-hits-in-stream",
    "--fix-profile-radius=70000000",
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
    from tqdm import tqdm

    # cmd = ["python3", script, *args]
    # print(f"[RUN] {' '.join(cmd)}", flush=True)
    is_run_sh = (script == "run_sh.py")

    if is_run_sh:
        # force unbuffered stdout from the child
        cmd = ["python3", "-u", script, *args]
    else:
        cmd = ["python3", script, *args]

    # print(f"[RUN] {' '.join(cmd)}", flush=True)

    # Normal behavior for all scripts except run_sh.py
    is_run_sh = (script == "run_sh.py")

    if not is_run_sh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        try:
            for line in proc.stdout:
                # BEFORE:
                # print(line, end="")
                # AFTER:
                print(line, end="", flush=True)
        finally:
            proc.stdout.close()
        rc = proc.wait()
        if check and rc != 0:
            raise SystemExit(f"[ERR] {script} exited with {rc}")
        return rc

    # --- Special Case: run_sh.py → use single-line tqdm progress bar ---
    total_events = None
    event_re = re.compile(r"\brunning\s+(\d+)\s+event\b", re.I)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    pbar = None
    try:
        for line in proc.stdout:
            if "__EVENT_DONE__" in line:
                if pbar is not None:
                    pbar.update(1)
                continue

            # BEFORE:
            # print(line, end="")
            # AFTER:
            print(line, end="", flush=True)

            m = event_re.search(line)
            if m and pbar is None:
                total_events = int(m.group(1))
                real_stdout = getattr(sys, "__stdout__", sys.stdout)
                pbar = tqdm(
                    total=total_events,
                    desc="[run_sh] Indexing",
                    unit="evt",
                    ncols=80,
                    ascii=True,
                    dynamic_ncols=False,
                    leave=False,
                    file=real_stdout,
                    bar_format="{desc}: |{bar}| {percentage:3.0f}% ETA {remaining}",
                )
                continue
    finally:
        if pbar is not None:
            pbar.close()
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

def do_init_sequence(run_root: str, geom: str, cell: str, h5_sources: list, jobs=os.cpu_count()):
    print("[phase] Initializing run_000")
    sources = []
    for s in h5_sources:
        matches = sorted(glob(s))
        sources.extend(matches if matches else [s])

    print(f"[init] using {len(sources)} HDF5 source(s):", *sources, sep="\n  ")
    print(f"[init] using following indexamajig flags:")
    print(f" {' '.join(DEFAULT_FLAGS)}")
    
    # Print convergence parameters once at initialization
    print("[init] Convergence parameters:")
    print(f" radius_mm = {radius_mm}, min_spacing_mm = {min_spacing_mm}, N_conv = {N_conv}, recurring_tol = {recurring_tol}, median_rel_tol = {median_rel_tol}, noimprove_N = {noimprove_N}, noimprove_eps = {noimprove_eps}, stability_N = {stability_N}, stability_std = {stability_std}, done_on_streak_successes = {done_on_streak_successes}, done_on_streak_length = {done_on_streak_length}, damping_factor (λ) = {λ}")

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
    run_py("run_sh.py", ["--run-root", run_root, "--run", "000", "--jobs", str(jobs)], check=False)
    run_py("evaluate_stream.py", ["--run-root", run_root, "--run", "000"], check=False)
    run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
    run_py("summarize_image_run_log.py", ["--run-root", run_root,])
    run_py("build_early_break_from_log.py", ["--run-root", run_root])
    print("[done] Initialization cycle complete. Proceeding to loop...")

def iterate_until_done(run_root, max_iters=10, jobs=os.cpu_count()):
    rd = runs_dir(run_root)
    it = 0
    while it < max_iters:
        it += 1

        # print(f"\n[loop] Iteration {it}", flush=True)

        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[loop] Iteration {it} started at {ts}", flush=True)

        # 1) Propose next shifts
        # run_py("propose_next_shifts.py", ["--run-root", run_root])
        run_py(
            "propose_next_shifts.py",
            [
                "--run-root", run_root,
                "--radius-mm", str(radius_mm),
                "--min-spacing-mm", str(min_spacing_mm),
                "--N-conv", str(N_conv),
                "--recurring-tol", str(recurring_tol),
                "--median-rel-tol", str(median_rel_tol),
                "--noimprove-N", str(noimprove_N),
                "--noimprove-eps", str(noimprove_eps),
                "--stability-N", str(stability_N),
                "--stability-std", str(stability_std),
                "--done-on-streak-successes", str(done_on_streak_successes),
                "--done-on-streak-length", str(done_on_streak_length),
                "--damping-factor", str(λ),
                "--step2-algorithm", "dxdy",
            ]
        )


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
        run_py("run_sh.py", ["--run-root", run_root, "--run", run_str, "--jobs", str(jobs)], check=False)
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
    ap.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS, help="Maximum number of iterations before stopping.")
    ap.add_argument("--jobs", type=str, default=DEFAULT_NUM_CPU, help="Number of parallel jobs for indexamajig/xgandalf.")

    ap.add_argument("--radius-mm", type=float, default=radius_mm)
    ap.add_argument("--min-spacing-mm", type=float, default=min_spacing_mm)
    ap.add_argument("--N-conv", type=int, default=N_conv)
    ap.add_argument("--recurring-tol", type=float, default=recurring_tol)
    ap.add_argument("--median-rel-tol", type=float, default=median_rel_tol)
    ap.add_argument("--noimprove-N", type=int, default=noimprove_N)
    ap.add_argument("--noimprove-eps", type=float, default=noimprove_eps)
    ap.add_argument("--stability-N", type=int, default=stability_N)
    ap.add_argument("--stability-std", type=float, default=stability_std)
    ap.add_argument("--done-on-streak-successes", type=int, default=done_on_streak_successes)
    ap.add_argument("--done-on-streak-length", type=int, default=done_on_streak_length)
    ap.add_argument("--damping-factor", type=float, default=λ)

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
            do_init_sequence(sess, args.geom, args.cell, args.h5, jobs=args.jobs)

        # Iterate until all next_* == done
        iterate_until_done(sess, max_iters=args.max_iters, jobs=args.jobs)

    # --- end: wrap whole orchestration with logger ------------------

if __name__ == "__main__":
    sys.exit(main())
