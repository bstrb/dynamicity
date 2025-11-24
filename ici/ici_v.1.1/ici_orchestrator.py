#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ici_orchestrator.py
Orchestrate SerialED iterative runs using provided helper scripts.
loop:
    - create overlays and list for next iteration
    - run indexamajig/xgandalf/integration
    - evaluate streams
    - update image_run_log.csv
    - propose next det shifts
    - summarize log
    - build early_break_from_log (one with all best index per image/event and one with only done image/events)
    until done or max iterations reached

"""
from glob import glob
import argparse, os, re, subprocess, sys
import time, datetime, threading
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# DEFAULT_ROOT = ""
# DEFAULT_GEOM = ""
# DEFAULT_CELL = ""
# DEFAULT_H5 = []
# Default paths

# DEFAULT_ROOT = "/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_012"
# DEFAULT_GEOM = DEFAULT_ROOT + "/MFM300-VIII.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/MFM300-VIII.cell"
# DEFAULT_H5   = [DEFAULT_ROOT + "/sim2.h5"]
# DEFAULT_H5   = [DEFAULT_ROOT + "/sim1.h5",
#                 DEFAULT_ROOT + "/sim2.h5",
#                 DEFAULT_ROOT + "/sim3.h5"]

DEFAULT_ROOT = "/home/bubl3932/files/MFM300_VIII/MP15_3x100"
DEFAULT_GEOM = DEFAULT_ROOT + "/MFM.geom"
DEFAULT_CELL = DEFAULT_ROOT + "/MFM.cell"
DEFAULT_H5   = [DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_1712_min_15peaks_100.h5",
                DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_1822_min_15peaks_100.h5",
                DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038_min_15peaks_100.h5"]

# DEFAULT_ROOT = "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038"
# DEFAULT_GEOM = DEFAULT_ROOT + "/MFM.geom"
# DEFAULT_CELL = DEFAULT_ROOT + "/MFM.cell"
# DEFAULT_H5 = [DEFAULT_ROOT + "/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038_min_15peaks.h5"]

DEFAULT_MAX_ITERS = 20
DEFAULT_NUM_CPU = os.cpu_count()

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
λ                = 0.8          # damping factor for refined det shift updates

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

class TimestampingStream:
    """
    Wraps a stream and prefixes each full line with a wall-clock timestamp.

    Works with arbitrary chunks (no assumption that writes end with '\n').
    """
    def __init__(self, real_stream):
        self.real = real_stream
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data):
        if not data:
            return
        with self._lock:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.real.write(f"[{ts}] {line}\n")

    def flush(self):
        with self._lock:
            if self._buf:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.real.write(f"[{ts}] {self._buf}")
                self._buf = ""
            self.real.flush()

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

        # tee stdout/stderr and add timestamps per line
        self._old_out, self._old_err = sys.stdout, sys.stderr

        tee_out = _Tee(sys.stdout, self._fh)
        tee_err = _Tee(sys.stderr, self._fh)

        sys.stdout = TimestampingStream(tee_out)
        sys.stderr = TimestampingStream(tee_err)

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
    import subprocess
    from tqdm import tqdm

    is_run_sh = (script == "run_sh.py")

    script_path = os.path.join(SCRIPT_DIR, script)
    if is_run_sh:
        # force unbuffered stdout from the child for progress markers
        cmd = ["python3", "-u", script_path, *args]
    else:
        cmd = ["python3", script_path, *args]

    # Normal behavior for all scripts except run_sh.py
    if not is_run_sh:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            print(f"[ERR] Failed to launch {script}: {e}", file=sys.stderr)
            if check:
                raise SystemExit(2)
            return 2

        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        rc = proc.wait()
        if check and rc != 0:
            raise SystemExit(f"[ERR] {script} exited with {rc}")
        return rc

    # --- Special Case: run_sh.py → use single-line tqdm progress bar ---
    total_events = None
    event_re = re.compile(r"\brunning\s+(\d+)\s+event\b", re.I)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        print(f"[ERR] Failed to launch {script}: {e}", file=sys.stderr)
        if check:
            raise SystemExit(2)
        return 2

    pbar = None
    try:
        for line in proc.stdout:
            if "__EVENT_DONE__" in line:
                if pbar is not None:
                    pbar.update(1)
                continue

            print(line, end="", flush=True)

            m = event_re.search(line)
            if m and pbar is None:
                total_events = int(m.group(1))
                real_stdout = getattr(sys, "__stdout__", sys.stdout)
                pbar = tqdm(
                    total=total_events,
                    desc="Indexing",
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
        if proc.stdout is not None:
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
            _ = f.readline()  # skip header
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 7 or not parts[0].isdigit():
                    continue
                if int(parts[0]) != latest:
                    continue
                any_row = True

                # Explicitly look at *next_dx_mm* and *next_dy_mm* columns
                next_dx = parts[5].lower()
                next_dy = parts[6].lower()
                if not (next_dx == "done" and next_dy == "done"):
                    return False
        return any_row
    except FileNotFoundError:
        return False


# def do_init_sequence(run_root: str, geom: str, cell: str, h5_sources: list, jobs=os.cpu_count()):
def do_init_sequence(
    run_root: str,
    geom: str,
    cell: str,
    h5_sources: list,
    flags: List[str],
    params: dict | None = None,
    jobs=os.cpu_count()
):
    params = params or {}
    required_keys = [
        "radius_mm", "min_spacing_mm", "N_conv",
        "recurring_tol", "median_rel_tol",
        "noimprove_N", "noimprove_eps",
        "stability_N", "stability_std",
        "done_on_streak_successes", "done_on_streak_length",
        "damping_factor",
    ]

    for k in required_keys:
        if k not in params:
            raise SystemExit(f"[ERR] Missing required parameter '{k}' in params dict.")

    radius_mm        = params["radius_mm"]          # search radius
    min_spacing_mm   = params["min_spacing_mm"]     # minimum spacing between shifts
    N_conv           = params["N_conv"]             # minimum number o events to consider convergence
    recurring_tol    = params["recurring_tol"]      # tolerance for recurring shifts (0.1 = 10%)
    median_rel_tol   = params["median_rel_tol"]     # median relative tolerance for convergence (0.1 = 10%)
    noimprove_N      = params["noimprove_N"]        # number of iterations with no improvement to consider convergence
    noimprove_eps    = params["noimprove_eps"]      # minimum improvement for noimprove trigger (0.02 = 2%)
    stability_N      = params["stability_N"]        # number of iterations to consider for stability
    stability_std    = params["stability_std"]      # standard deviation threshold for stability (0.05 = 5%)
    done_on_streak_successes = params["done_on_streak_successes"]  # number of successful streaks to consider done
    done_on_streak_length = params["done_on_streak_length"]  # length of streak to consider done
    λ = params["damping_factor"]                    # damping factor (λ) for refined det shift updates     

    print("[phase] Initializing first run (run_000)...")
    sources = []
    for s in h5_sources:
        matches = sorted(glob(s))
        sources.extend(matches if matches else [s])

    print(f"[init] Using {len(sources)} HDF5 source(s):", *sources, sep="\n")
    print(f"[init] Geometry file: {geom}")
    print(f"[init] Cell file: {cell}")
    print(f"[init] CrystFEL indexamajig flags:")
    print(f"{' '.join(flags)}")
    
    # Print convergence parameters once at initialization
    print("[init] Convergence parameters:")
    print(f"radius_mm = {radius_mm}, min_spacing_mm = {min_spacing_mm}, N_conv = {N_conv}, recurring_tol = {recurring_tol}, median_rel_tol = {median_rel_tol}, noimprove_N = {noimprove_N}, noimprove_eps = {noimprove_eps}, stability_N = {stability_N}, stability_std = {stability_std}, done_on_streak_successes = {done_on_streak_successes}, done_on_streak_length = {done_on_streak_length}, damping_factor (λ) = {λ}")

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
        *flags,], check=False)
    run_py("run_sh.py", ["--run-root", run_root, "--run", "000", "--jobs", str(jobs)], check=False)
    run_py("evaluate_stream.py", ["--run-root", run_root, "--run", "000"], check=False)
    run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
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
    run_py("summarize_image_run_log.py", ["--run-root", run_root,])
    run_py("build_early_break_from_log.py", ["--run-root", run_root])
    print("[done] Initialization cycle complete. Proceeding to loop...")

def iterate_until_done(
    run_root: str,
    geom: str,
    cell: str,
    max_iters: int,
    flags: List[str],
    params: dict | None = None,
    jobs=os.cpu_count()
):
    params = params or {}
    required_keys = [
        "radius_mm", "min_spacing_mm", "N_conv",
        "recurring_tol", "median_rel_tol",
        "noimprove_N", "noimprove_eps",
        "stability_N", "stability_std",
        "done_on_streak_successes", "done_on_streak_length",
        "damping_factor",
    ]

    for k in required_keys:
        if k not in params:
            raise SystemExit(f"[ERR] Missing required parameter '{k}' in params dict.")

    radius_mm        = params["radius_mm"]          # search radius
    min_spacing_mm   = params["min_spacing_mm"]     # minimum spacing between shifts
    N_conv           = params["N_conv"]             # minimum number o events to consider convergence
    recurring_tol    = params["recurring_tol"]      # tolerance for recurring shifts (0.1 = 10%)
    median_rel_tol   = params["median_rel_tol"]     # median relative tolerance for convergence (0.1 = 10%)
    noimprove_N      = params["noimprove_N"]        # number of iterations with no improvement to consider convergence
    noimprove_eps    = params["noimprove_eps"]      # minimum improvement for noimprove trigger (0.02 = 2%)
    stability_N      = params["stability_N"]        # number of iterations to consider for stability
    stability_std    = params["stability_std"]      # standard deviation threshold for stability (0.05 = 5%)
    done_on_streak_successes = params["done_on_streak_successes"]  # number of successful streaks to consider done
    done_on_streak_length = params["done_on_streak_length"]  # length of streak to consider done
    λ = params["damping_factor"]                    # damping factor (λ) for refined det shift updates     

    rd = runs_dir(run_root)
    it = 0
    while it < max_iters:
        it += 1


        # import datetime
        # ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        print(f"==================== Iteration {it} started ====================", flush=True)

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

        # # 3) else build overlays & list for next iteration
        run_py("build_overlays_and_list.py", ["--run-root", run_root])

        # Decide next run index from the *log*, not by reusing the latest folder
        latest_in_log = detect_latest_run_from_log(log_path)
        if latest_in_log < 0:
            print("[err] No runs found in image_run_log.csv; aborting.")
            break

        next_run_num = latest_in_log + 1
        run_str = f"{next_run_num:03d}"
        latest_dir = os.path.join(runs_dir(run_root), f"run_{run_str}")
        os.makedirs(latest_dir, exist_ok=True)

        # run_py("copy_next_run_sh.py", ["--run-root", run_root, "--run", run_str], check=False)
        run_py("create_run_sh.py", ["--run-root", run_root, "--geom", geom, "--cell", cell, "--run", run_str,
        "--",  # everything after this is passed directly into indexamajig
        *flags], check=False)
        run_py("run_sh.py", ["--run-root", run_root, "--run", run_str, "--jobs", str(jobs)], check=False)
        # NEW: if no stream file exists, stop cleanly instead of crashing downstream
        stream_candidate = os.path.join(latest_dir, f"stream_{run_str}.stream")
        if not os.path.exists(stream_candidate):
            print(f"[warn] No per-event streams found for run_{run_str}: {stream_candidate} not found.")
            print("[stop] No successful indexing in latest iteration; stopping orchestration.")
            break
        _ = run_py("fix_stream_paths.py", ["--run-dir", latest_dir, "--run", run_str, "--inplace"], check=False)
        run_py("evaluate_stream.py", ["--run-root", run_root, "--run", run_str], check=False)
        run_py("update_image_run_log_grouped.py", ["--run-root", run_root])
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
        run_py("summarize_image_run_log.py", ["--run-root", run_root,])
        run_py("build_early_break_from_log.py", ["--run-root", run_root])

    else:
        print(f"[stop] Reached max-iters={max_iters-1} without satisfying 'done'.")
    print("[done] Orchestration complete.")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Orchestrate SerialED iterative runs using provided helper scripts.")
    ap.add_argument("--run-root", default=None, help="Experiment root that contains 'runs/'.")
    ap.add_argument("--geom", default=None, help="Geometry file for initialization.")
    ap.add_argument("--cell", default=None, help="Cell file for initialization.")
    ap.add_argument("--h5", nargs="+", default=None,
                    help="One or more HDF5 sources or globs (e.g., sim_001.h5 sim_002.h5 or sim_*.h5)")
    ap.add_argument("--flags", nargs="*", default=DEFAULT_FLAGS,
                    help="Additional indexamajig / xgandalf / integration flags for initialization.")
    ap.add_argument("--max-iters", type=int, default=20,
                    help="Maximum number of iterations before stopping.")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="Number of parallel jobs during indexing and refinement.")

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
    params = {
        "radius_mm": args.radius_mm,
        "min_spacing_mm": args.min_spacing_mm,
        "N_conv": args.N_conv,
        "recurring_tol": args.recurring_tol,
        "median_rel_tol": args.median_rel_tol,
        "noimprove_N": args.noimprove_N,
        "noimprove_eps": args.noimprove_eps,
        "stability_N": args.stability_N,
        "stability_std": args.stability_std,
        "done_on_streak_successes": args.done_on_streak_successes,
        "done_on_streak_length": args.done_on_streak_length,
        "damping_factor": args.damping_factor,
    }

    # ---------- basic input validation (fail fast, nice messages) ----------

    # max iterations
    if args.max_iters <= 0:
        print("[ERR] --max-iters must be at least 1.", file=sys.stderr)
        return 2

    # experiment root: may not exist yet; parent must exist and be writable
    exp_root = os.path.abspath(os.path.expanduser(args.run_root))
    parent = os.path.dirname(exp_root) or "."
    if not os.path.isdir(parent):
        print(f"[ERR] Parent directory of run-root does not exist: {parent}", file=sys.stderr)
        return 2
    if not os.access(parent, os.W_OK):
        print(f"[ERR] Parent directory of run-root is not writable: {parent}", file=sys.stderr)
        return 2

    # geometry / cell files must exist
    geom = os.path.abspath(os.path.expanduser(args.geom))
    cell = os.path.abspath(os.path.expanduser(args.cell))

    if not geom or not os.path.isfile(geom):
        print(f"[ERR] Geometry file not found: {geom}", file=sys.stderr)
        print("      Please provide a valid --geom path to a CrystFEL .geom file.", file=sys.stderr)
        return 2

    if not cell or not os.path.isfile(cell):
        print(f"[ERR] Cell file not found: {cell}", file=sys.stderr)
        print("      Please provide a valid --cell path to a CrystFEL .cell file.", file=sys.stderr)
        return 2

    # HDF5 sources must be non-empty and at least one pattern must match
    h5_sources = args.h5 if isinstance(args.h5, (list, tuple)) else [args.h5]
    h5_sources = [s for s in h5_sources if s]

    if not h5_sources:
        print(
            "[ERR] No HDF5 sources provided. Use --h5 file1.h5 file2.h5 or patterns like sim_*.h5",
            file=sys.stderr,
        )
        return 2

    matched_any = False
    unmatched_patterns = []
    for s in h5_sources:
        pattern = os.path.abspath(os.path.expanduser(s))
        if glob(pattern):
            matched_any = True
        else:
            unmatched_patterns.append(s)

    if not matched_any:
        print("[ERR] None of the provided --h5 patterns matched any files on disk.", file=sys.stderr)
        print("      Patterns checked:", file=sys.stderr)
        for s in h5_sources:
            print(f"        {s}", file=sys.stderr)
        return 2

    if unmatched_patterns:
        print("[WARN] Some --h5 patterns did not match any files (they may be ignored downstream):",
              file=sys.stderr)
        for s in unmatched_patterns:
            print(f"        {s}", file=sys.stderr)

    # ---------- create session directory ----------

    # one timestamped subfolder per orchestration
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sess = os.path.join(exp_root, f"runs_{ts}")
    os.makedirs(runs_dir(sess), exist_ok=True)

    # ---------- run orchestration with logging ----------

    with OrchestratorRunLogger(runs_dir(sess)):
        if not list_run_numbers(sess):
            do_init_sequence(sess, geom, cell, h5_sources, flags=args.flags, params=params, jobs=args.jobs)

        iterate_until_done(sess, geom, cell, max_iters=args.max_iters+1, flags=args.flags, params=params, jobs=args.jobs)


    return 0

if __name__ == "__main__":
    sys.exit(main())
