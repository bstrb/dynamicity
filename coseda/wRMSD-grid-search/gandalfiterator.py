import os
import h5py

# --- VENDORED GANDALF ITERATOR BEGIN ---

def gandalf_iterator(
    geomfile_path: str,
    cellfile_path: str,
    input_path: str,
    output_file_base: str,
    num_threads: int,
    max_radius: float,
    step: float,
    extra_flags=None,
):
    """
    Run CrystFEL 'indexamajig' over a filled circular grid of center shifts.

    For each (dx, dy) in *pixels*:
      1) Convert to millimeters using geometry 'res' (mm/px = 1000/res).
      2) Apply det_shift_x_mm += dx_mm and det_shift_y_mm += dy_mm (in-place) to all HDF5s listed.
      3) Run indexamajig to write a unique .stream for this pass.
      4) Revert the applied shifts (subtract the same deltas), even on errors.

    Emits progress lines and [XY-PASS] i/N markers for the GUI.
    """
    import sys, subprocess, math
    from pathlib import Path

    # ---- Validate required inputs ----
    geom = Path(geomfile_path)
    cell = Path(cellfile_path)
    input_p = Path(input_path)

    if not geom.is_file():
        print(f"ERROR: geometry file missing: {geom}", file=sys.stderr); return 2
    if not cell.is_file():
        print(f"ERROR: cell file missing: {cell}", file=sys.stderr); return 2
    if not input_p.exists():
        print(f"ERROR: input path missing: {input_p}", file=sys.stderr); return 2

    # ---- Output directory & base name ----
    outbase = Path(output_file_base)
    if outbase.is_absolute():
        out_dir = outbase.parent
        out_base_name = outbase.name
    else:
        base_dir = input_p if input_p.is_dir() else input_p.parent
        out_dir = base_dir / f"xgandalf_iterations_max_radius_{max_radius}_step_{step}"
        out_base_name = outbase.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resulting streamfiles will be saved in {out_dir}")

    # ---- Prepare list.lst (or accept provided list file) ----
    if input_p.is_file():
        list_path = input_p
        print(f"Using provided list file: {list_path}")
    else:
        list_path = input_p / "list.lst"
        try:
            if not list_path.exists():
                exts = ("*.h5", "*.hdf5", "*.cxi")
                hits = []
                for ext in exts:
                    hits.extend(sorted(input_p.glob(ext)))
                hits = [str(p.resolve()) for p in hits]
                list_path.write_text("\n".join(hits) + ("\n" if hits else ""), encoding="utf-8")
                print(f"'list.lst' has been created with {len(hits)} entries at {list_path}")
            else:
                print(f"Using existing list file: {list_path}")
        except Exception as e:
            print(f"WARNING: could not write list.lst: {e}", file=sys.stderr)

    # ---- Build grid: step-lattice points inside circle, sorted radially ----
    grid_pts: list[tuple[float, float]] = []
    if max_radius <= 0 or step <= 0:
        grid_pts = [(0.0, 0.0)]
    else:
        try:
            decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 6
            decimals = min(6, decimals)
        except Exception:
            decimals = 6

        limit = int(math.ceil(max_radius / step))
        R2 = max_radius * max_radius
        pts_set = set()
        for i in range(-limit, limit + 1):
            xi = round(i * step, decimals)
            xi2 = xi * xi
            if xi2 > R2 + 1e-12:
                continue
            for j in range(-limit, limit + 1):
                yj = round(j * step, decimals)
                if xi2 + yj * yj <= R2 + 1e-12:
                    pts_set.add((xi, yj))

        grid_pts = sorted(
            pts_set,
            key=lambda p: (p[0]*p[0] + p[1]*p[1], abs(p[0]) + abs(p[1]), p[0], p[1])
        )
        if (0.0, 0.0) in grid_pts:
            grid_pts.remove((0.0, 0.0))
            grid_pts.insert(0, (0.0, 0.0))

    print(f"Generated {len(grid_pts)} grid points in the circle.")

    # ---- Convert px→mm using geometry 'res' ----
    try:
        res = extract_resolution(str(geom))  # expects a line "res = <float>"
        mm_per_pixel = 1000.0 / float(res)
        print(f"[geom] res={res} → mm_per_pixel={mm_per_pixel}")
    except Exception as e:
        print(f"ERROR: failed to read 'res' from geometry: {e}", file=sys.stderr)
        return 2

    # ---- Handle optional indexamajig path flag; pass through remaining flags ----
    extra_flags = list(extra_flags or [])
    idxamajig_bin = "indexamajig"
    pruned_flags: list[str] = []
    for f in extra_flags:
        if f.startswith("--indexamajig-path="):
            idxamajig_bin = f.split("=", 1)[1].strip() or idxamajig_bin
        else:
            pruned_flags.append(f)
    extra_flags = pruned_flags

    # ---- Main loop: apply shift, run, revert ----
    total = len(grid_pts)
    passes_done = 0

    for (dx_px, dy_px) in grid_pts:
        print(f"Running for pixel shifts x = {dx_px}, y = {dy_px}")
        # Convert pixel shifts to millimeters
        sx_mm = float(dx_px) * mm_per_pixel
        sy_mm = float(dy_px) * mm_per_pixel
        print(f"  => Applying det_shift Δx={sx_mm} mm, Δy={sy_mm} mm")

        stream_name = (
            f"{out_base_name}_{dx_px:.3f}_{dy_px:.3f}.stream"
            .replace("+", "")
            .replace("-0.000", "0.000")
        )
        stream_path = (out_dir / stream_name).resolve()

        # 1) Apply in-place shift
        try:
            perturb_det_shifts(str(list_path), sx_mm, sy_mm)
        except Exception as e:
            print(f"ERROR: failed to apply det_shift ({sx_mm},{sy_mm}) mm: {e}", file=sys.stderr)
            # skip this pass and continue with the next
            continue

        # 2) Run indexamajig and 3) Revert in finally
        try:
            cmd = [
                idxamajig_bin,
                "-g", str(geom),
                "-i", str(list_path),
                "-o", str(stream_path),
                "-p", str(cell),
                "-j", str(num_threads),
                *extra_flags,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                print(f"ERROR: '{idxamajig_bin}' not found. Set it on PATH or use --indexamajig-path=...", file=sys.stderr)
                return 2
            except Exception as e:
                print(f"ERROR: failed to launch indexamajig: {e}", file=sys.stderr)
                return 2

            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line.rstrip())
            except Exception as e:
                print(f"WARNING: error while reading indexamajig output: {e}", file=sys.stderr)

            rc = proc.wait()
            if rc != 0:
                print(f"Error during indexamajig execution (exit {rc}).", file=sys.stderr)
            else:
                passes_done += 1
                print(f"[XY-PASS] {passes_done}/{total} completed -> {stream_path}")

        finally:
            # Always revert the applied shift
            try:
                perturb_det_shifts(str(list_path), -sx_mm, -sy_mm)
                print(f"  => Reverted det_shift Δx={-sx_mm} mm, Δy={-sy_mm} mm")
            except Exception as e:
                print(f"WARNING: failed to revert det_shift: {e}", file=sys.stderr)

    return 0


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

def run_indexamajig(geomfile_path, listfile_path, cellfile_path, output_path, num_threads, extra_flags=None):
    if extra_flags is None:
        extra_flags = []

    cmd = [
        "indexamajig",
        "-g", geomfile_path,
        "-i", listfile_path,
        "-o", output_path,
        "-p", cellfile_path,
        "-j", str(num_threads),
        *extra_flags,
    ]
    # inherit stdout/stderr so QProcess sees lines immediately
    import subprocess
    subprocess.run(cmd, check=True)

def perturb_det_shifts(file_list, x_pert, y_pert):
    """
    Usage: python perturb_shifts.py file_list.lst x_perturbation y_perturbation

    This script adds x_perturbation to entry/data/det_shift_x_mm
    and y_perturbation to entry/data/det_shift_y_mm in each .h5 file
    listed in file_list.lst.
    """

    # Read the .lst file to get list of .h5 files
    with open(file_list, 'r') as f:
        h5_files = [line.strip() for line in f if line.strip()]

    # Loop over .h5 files and apply perturbation
    for h5_file in h5_files:
        # print(f"Processing: {h5_file}")
        with h5py.File(h5_file, 'r+') as h5f:
            # Load existing datasets
            x_data = h5f["entry/data/det_shift_x_mm"][...]
            y_data = h5f["entry/data/det_shift_y_mm"][...]

            # Apply perturbations
            x_data += x_pert
            y_data += y_pert

            # Write updated values back
            h5f["entry/data/det_shift_x_mm"][:] = x_data
            h5f["entry/data/det_shift_y_mm"][:] = y_data

    print(f"  => Applied x shift of {x_pert} mm, y shift of {y_pert} mm")

def list_h5_files(input_path):
    """
    Creates or replaces a 'list.lst' file in the specified input_path directory.
    The file contains the full paths of all files ending with '.h5' in the directory,
    sorted alphabetically.
    
    Args:
        input_path (str): The directory path where '.h5' files are located.
    """
    # Path to the list file
    listfile_path = os.path.join(input_path, 'list.lst')
    
    try:
        # List all .h5 files
        h5_files = [file for file in os.listdir(input_path) if file.endswith('.h5') and os.path.isfile(os.path.join(input_path, file))]
        
        # Sort the list alphabetically
        h5_files_sorted = sorted(h5_files, key=lambda x: x.lower())  # Case-insensitive sorting
        
        # Open the list file in write mode to overwrite if it exists
        with open(listfile_path, 'w') as list_file:
            for file in h5_files_sorted:
                full_path = os.path.join(input_path, file)
                list_file.write(full_path + '\n')
        
        print(f"'list.lst' has been created with {len(h5_files_sorted)} entries at {listfile_path}")
    
    except FileNotFoundError:
        print(f"The directory '{input_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied when accessing '{input_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return listfile_path

# # --- VENDORED GANDALF ITERATOR END ---