# coseda/coseda/gandalf_radial_iterator.py
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

# NEW — add with other imports
from coseda.io import log_start, log_result, handle_input, parse_config, shoutout


import h5py

__all__ = [
    "DEFAULT_PEAKFINDER_OPTIONS",
    "INDEXING_FLAGS",
    "count_images_in_h5_folder",
    "estimate_passes",
    "run_gandalf_iterator",
    "cleanup_temp_dirs",
    "gandalf_iterator", 
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
    cancel_check: Optional[Callable[[], bool]] = None,
) -> int:
    """
    Run the gandalf radial iterator in a subprocess (no Qt). Progress is reported via callbacks.

    progress_callback(done, total): overall progress across all passes
    line_callback(line): raw stdout (merged with stderr)
    cancel_check(): return True to request cooperative cancellation

    Returns subprocess return code (130 if cancelled via SIGINT).
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(input_folder)
    if not os.path.isfile(geom_file):
        raise FileNotFoundError(geom_file)
    if not os.path.isfile(cell_file):
        raise FileNotFoundError(cell_file)

    shoutout(
        f"[gandalf_iterator] start: folder={input_folder}, output_base={output_base}, "
        f"threads={threads}, max_radius={max_radius}, step={step}, flags={extra_flags or []}"
    )

    total_images, _ = count_images_in_h5_folder(input_folder)
    per_pass_total = max(0, total_images)
    passes_total = estimate_passes(max_radius, step)
    overall_max = max(1, per_pass_total * passes_total)

    runner_code = textwrap.dedent(f"""
        import sys
        import os

        script_dir = r"{os.path.dirname(os.path.abspath(__file__))}"
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        try:
            from gandalf_radial_iterator import gandalf_iterator
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
    parent_pp = env.get("PYTHONPATH", "")
    extra_paths = [p for p in sys.path if isinstance(p, str) and p]
    env["PYTHONPATH"] = (parent_pp + (os.pathsep if parent_pp else "")) + os.pathsep.join(extra_paths)
    if env_extra:
        env.update(env_extra)

    popen_kwargs = dict(
        args=argv,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        popen_kwargs["preexec_fn"] = os.setsid  # start a new session/process group

    proc = subprocess.Popen(**popen_kwargs)  # type: ignore[arg-type]

    def _send_sigint_to_group(p: subprocess.Popen):
        try:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass

    passes_done = 0
    prev_processed_this_pass = 0
    rc: int = 0

    try:
        for raw_line in proc.stdout:  # type: ignore[union-attr]
            if cancel_check and cancel_check():
                shoutout("[gandalf_iterator] cancellation requested; sending SIGINT to process group…")
                _send_sigint_to_group(proc)
                rc = 130
                break

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
        try:
            if proc.poll() is None:
                if cancel_check and cancel_check():
                    _send_sigint_to_group(proc)
                rc_wait = proc.wait()
                if rc == 0:
                    rc = rc_wait
            else:
                rc = proc.returncode
        finally:
            try:
                Path(runner_path).unlink(missing_ok=True)
            except Exception:
                pass
            cleanup_temp_dirs()

    shoutout(f"[gandalf_iterator] finished with return code {rc}")
    return rc


# -------------------- HELPER FUNCTIONS

import math
from tqdm import tqdm
def gandalf_iterator(geomfile_path, 
                     cellfile_path, 
                     input_path, 
                     output_file_base, 
                     num_threads, 
                     max_radius=1, 
                     step=0.1, 
                     extra_flags=None
                     ):
    """
    Run CrystFEL's 'indexamajig' on a grid of beam centers.
    """
    listfile_path = list_h5_files(input_path)
    output_folder = os.path.join(input_path, f"xgandalf_iterations_max_radius_{max_radius}_step_{step}")
    os.makedirs(output_folder, exist_ok=True)

    xy_pairs = list(generate_sorted_grid_points(max_radius=max_radius, step=step))
    shoutout(f"Resulting streamfiles will be saved in {output_folder}")
    res = extract_resolution(geomfile_path)
    mm_per_pixel = 1000 / res
    
    for x, y in tqdm(xy_pairs, desc="Processing XY pairs"):
        shoutout(f"Running for pixel shifts x = {x}, y = {y}")
        output_path = f"{output_folder}/{output_file_base}_{x}_{y}.stream"
        shift_x = x * mm_per_pixel
        shift_y = y * mm_per_pixel
        try:
            perturb_det_shifts(listfile_path, shift_x, shift_y)
            run_indexamajig(geomfile_path, listfile_path, cellfile_path, output_path, num_threads, extra_flags=extra_flags)
        except KeyboardInterrupt:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)
            shoutout("Process interrupted by user.")
            break
        except subprocess.CalledProcessError as e:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)
            shoutout(f"Error during indexamajig execution: {e}")
            break
        except Exception as e:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)
            shoutout(f"Unexpected error: {e}")
            break
        else:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)
def list_h5_files(input_path):
    """
    Creates or replaces a 'list.lst' file in the specified input_path directory.
    The file contains the full paths of all files ending with '.h5' in the directory,
    sorted alphabetically.
    """
    listfile_path = os.path.join(input_path, 'list.lst')
    try:
        h5_files = [
            file for file in os.listdir(input_path)
            if file.endswith('.h5') and os.path.isfile(os.path.join(input_path, file))
        ]
        h5_files_sorted = sorted(h5_files, key=lambda x: x.lower())
        with open(listfile_path, 'w') as list_file:
            for file in h5_files_sorted:
                full_path = os.path.join(input_path, file)
                list_file.write(full_path + '\n')
        shoutout(f"'list.lst' has been created with {len(h5_files_sorted)} entries at {listfile_path}")
    except FileNotFoundError:
        shoutout(f"The directory '{input_path}' does not exist.")
    except PermissionError:
        shoutout(f"Permission denied when accessing '{input_path}'.")
    except Exception as e:
        shoutout(f"An unexpected error occurred: {e}")
    return listfile_path

def run_indexamajig(geomfile_path, listfile_path, cellfile_path, output_path, num_threads, extra_flags=None):

    if extra_flags is None:
        extra_flags = []

    # Create a list of command parts
    command_parts = [
        "indexamajig",
        "-g", geomfile_path,
        "-i", listfile_path,
        "-o", output_path,
        "-p", cellfile_path,
        "-j", str(num_threads)
    ]

    # Append any extra flags provided by the user.
    command_parts.extend(extra_flags)

    # Join the parts into a single command string.
    base_command = " ".join(command_parts)
    subprocess.run(base_command, shell=True, check=True)
    
def extract_resolution(geom_file_path: str) -> float:
    """
    Extracts the resolution value from a geometry file.

    The geometry file is expected to contain a line like:
      res = 17857.14285714286

    Parameters:
        geom_file_path (str): The file path to the geometry file.

    Returns:
        float: The resolution value extracted from the file.

    Raises:
        ValueError: If the resolution value is not found or is invalid.
    """
    with open(geom_file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace
            line = line.strip()
            # Skip comment lines
            if line.startswith(";"):
                continue
            # Look for the resolution line
            if line.startswith("res"):
                parts = line.split("=")
                if len(parts) >= 2:
                    res_str = parts[1].strip()
                    try:
                        return float(res_str)
                    except ValueError:
                        raise ValueError(f"Invalid resolution value: {res_str}")
    raise ValueError("Resolution not found in the geometry file.")

def perturb_det_shifts(file_list, x_pert, y_pert):
    """
    Adds x_pert to entry/data/det_shift_x_mm and y_pert to entry/data/det_shift_y_mm
    in each .h5 file listed in file_list.lst.
    """
    with open(file_list, 'r') as f:
        h5_files = [line.strip() for line in f if line.strip()]

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r+') as h5f:
            x_data = h5f["entry/data/det_shift_x_mm"][...]
            y_data = h5f["entry/data/det_shift_y_mm"][...]
            x_data += x_pert
            y_data += y_pert
            h5f["entry/data/det_shift_x_mm"][:] = x_data
            h5f["entry/data/det_shift_y_mm"][:] = y_data

    shoutout(f"  => Applied x shift of {x_pert} mm, y shift of {y_pert} mm")


def grid_points_in_circle(x_center, y_center, max_radius, step=0.5):
    """
    Generate all grid points inside a circle with the given center and maximum radius.
    The grid is defined by the specified step size (granularity) and the coordinates are 
    rounded to a number of decimals determined by the step size.
    
    Args:
        x_center, y_center: Coordinates of the circle center.
        max_radius: Maximum radius from the center.
        step: Grid spacing.
        
    Returns:
        A list of (x, y) tuples that lie within the circle.
    """
    # Determine the number of decimals for rounding based on the step size.
    decimals = max(0, -int(math.floor(math.log10(step))))
    
    points = []
    max_i = int(math.ceil(max_radius / step))
    
    for i in range(-max_i, max_i + 1):
        for j in range(-max_i, max_i + 1):
            # Compute and round the grid coordinates
            x = round(x_center + i * step, decimals)
            y = round(y_center + j * step, decimals)
            # Check if the point is within the circle
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= max_radius ** 2:
                points.append((x, y))
    return points

def generate_sorted_grid_points(max_radius, step=0.5):
    """
    Generate all grid points (with the given granularity) within a circle defined by max_radius,
    round them based on the step size, and sort them in order of increasing radial distance.
    """
    x_center, y_center = 0, 0
    points = grid_points_in_circle(x_center, y_center, max_radius, step)
    points.sort(key=lambda pt: (pt[0] - x_center) ** 2 + (pt[1] - y_center) ** 2)
    shoutout(f"Generated {len(points)} grid points in the circle.")
    return points
