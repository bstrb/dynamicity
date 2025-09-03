# coseda/gandalf_iterator.py
from __future__ import annotations

import os
import sys
import glob
import shutil
import atexit
import signal
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import h5py

__all__ = [
    "DEFAULT_PEAKFINDER_OPTIONS",
    "INDEXING_FLAGS",
    "count_images_in_h5_folder",
    "estimate_passes",
    "run_gandalf_iterator",
    "cleanup_temp_dirs",
]

# -------------------- housekeeping --------------------

def cleanup_temp_dirs() -> None:
    """Remove indexamajig* directories and mille-data.bin* files in CWD."""
    for d in glob.glob("indexamajig*"):
        p = Path(d)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for f in glob.glob("mille-data.bin*"):
        p = Path(f)
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass

atexit.register(cleanup_temp_dirs)

def _handle_signal(_sig, _frame):
    cleanup_temp_dirs()
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# -------------------- constants --------------------

# Use list form to make GUI <-> backend concatenation easy
DEFAULT_PEAKFINDER_OPTIONS = {
    "cxi": ["--peaks=cxi"],
    "peakfinder9": [
        "--peaks=peakfinder9",
        "--min-snr-biggest-pix=7",
        "--min-snr-peak-pix=6",
        "--min-snr=5",
        "--min-sig=11",
        "--min-peak-over-neighbour=-inf",
        "--local-bg-radius=3",
    ],
    "peakfinder8": [
        "--peaks=peakfinder8",
        "--threshold=800",
        "--min-snr=5",
        "--min-pix-count=2",
        "--max-pix-count=200",
        "--local-bg-radius=3",
        "--min-res=0",
        "--max-res=1200",
    ],
}

INDEXING_FLAGS: List[str] = ["--indexing=xgandalf", "--integration=rings"]

_PROGRESS_RE = re.compile(r"^\s*(\d+)\s+images\s+processed\b", re.IGNORECASE)

# -------------------- HDF5 utilities --------------------

def count_images_in_h5_folder(folder: str) -> Tuple[int, List[Tuple[str, int]]]:
    """
    Count total images from /entry/data/images in all .h5 files in `folder` (recursive).
    Returns (total, [(path, n_images_per_file), ...]).
    """
    h5_paths = sorted(glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True))
    total = 0
    per_file: List[Tuple[str, int]] = []
    for path in h5_paths:
        n_for_file = 0
        try:
            with h5py.File(path, "r") as f:
                if "/entry/data/images" in f:
                    ds = f["/entry/data/images"]
                    if getattr(ds, "ndim", 0) >= 3:
                        n_for_file = int(ds.shape[0])
        except Exception:
            n_for_file = 0
        total += n_for_file
        per_file.append((path, n_for_file))
    return total, per_file

# -------------------- pass estimation --------------------

def estimate_passes(max_radius: float, step: float) -> int:
    """
    Estimate how many centre-shift runs will be done for (max_radius, step),
    assuming a circular mask on a square grid and including origin (0,0).
    """
    if step <= 0 or max_radius <= 0:
        return 1
    import math
    r, s = float(max_radius), float(step)
    k = int(math.ceil(r / s))
    count = 0
    for ix in range(-k, k + 1):
        for iy in range(-k, k + 1):
            if (ix * s) ** 2 + (iy * s) ** 2 <= r * r + 1e-9:
                count += 1
    return max(1, count)

# -------------------- backend runner --------------------
# from coseda_gandalf_radial_iterator_helpers import gandalf_iterator

def run_gandalf_iterator(
    *,
    geom_file: str,
    cell_file: str,
    input_folder: str,
    output_base: str = "Xtal",
    threads: int = 24,
    max_radius: float = 0.0,
    step: float = 0.1,
    extra_flags: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    line_callback: Optional[Callable[[str], None]] = None,
    python_executable: Optional[str] = None,
    env_extra: Optional[dict] = None,
) -> int:
    """
    Run the gandalf radial iterator in a subprocess (no Qt). Progress is reported via callbacks.

    progress_callback(done, total): overall progress across all passes
    line_callback(line): raw stdout (merged with stderr)
    Returns subprocess return code.
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(input_folder)
    if not os.path.isfile(geom_file):
        raise FileNotFoundError(geom_file)
    if not os.path.isfile(cell_file):
        raise FileNotFoundError(cell_file)

    total_images, _ = count_images_in_h5_folder(input_folder)
    per_pass_total = max(0, total_images)
    passes_total = estimate_passes(max_radius, step)
    overall_max = max(1, per_pass_total * passes_total)

    # Tiny runner so we can import the iterator in the child process
    runner_code = textwrap.dedent(f"""
        import sys
        import os
        
        # Add the gandalf_iterations directory to Python path
        script_dir = r"{os.path.dirname(os.path.abspath(__file__))}"
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        try:
            # Vendored inside COSEDA
            # from coseda.gandalf_iterations.gandalf_radial_iterator import gandalf_iterator
            from coseda_gandalf_radial_iterator_helpers import gandalf_iterator
        except ImportError as e:
            sys.stderr.write(f"Import error: {{e}}\\n")
            sys.stderr.flush()
            sys.exit(1)

        if __name__ == "__main__":
            args = sys.argv[1:]
            if len(args) < 7:
                sys.stderr.write("Runner expects: geom cell input_folder output_base threads max_radius step [extra flags...]\\n")
                sys.stderr.flush()
                sys.exit(2)
            geom, cell, input_folder, output_base, threads, max_radius, step, *extra_flags = args
            gandalf_iterator(
                geom, cell, input_folder, output_base,
                int(threads),
                max_radius=float(max_radius),
                step=float(step),
                extra_flags=extra_flags,
            )
    """).lstrip()

    tmp = tempfile.NamedTemporaryFile("w", delete=False, prefix="run_gandalf_", suffix=".py")
    tmp.write(runner_code)
    tmp.flush()
    tmp.close()
    runner_path = tmp.name

    flags = list(extra_flags or [])
    argv = [
        (python_executable or sys.executable),
        runner_path,
        geom_file, cell_file, input_folder, output_base,
        str(threads), str(max_radius), str(step),
        *flags,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # help child import project modules
    parent_pp = env.get("PYTHONPATH", "")
    extra_paths = [p for p in sys.path if isinstance(p, str) and p]
    env["PYTHONPATH"] = (parent_pp + (os.pathsep if parent_pp else "")) + os.pathsep.join(extra_paths)
    if env_extra:
        env.update(env_extra)

    proc = subprocess.Popen(
        argv,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    passes_done = 0
    prev_processed_this_pass = 0

    try:
        for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.rstrip("\r\n")
            if line_callback:
                line_callback(line)

            m = _PROGRESS_RE.match(line)
            if not m:
                continue

            try:
                processed = int(m.group(1))
            except Exception:
                processed = None

            if processed is None:
                continue

            # new pass if counter drops
            if processed < prev_processed_this_pass:
                passes_done = min(passes_done + 1, max(0, passes_total - 1))
            prev_processed_this_pass = processed

            cur = min(processed, per_pass_total) if per_pass_total > 0 else processed
            overall_done = passes_done * per_pass_total + cur
            if overall_done > overall_max:
                overall_done = overall_max

            if progress_callback:
                progress_callback(overall_done, overall_max)
    finally:
        rc = proc.wait()
        try:
            Path(runner_path).unlink(missing_ok=True)
        except Exception:
            pass
        cleanup_temp_dirs()

    return rc
