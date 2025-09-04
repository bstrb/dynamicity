import os
import subprocess
import math
import glob
import h5py
from dataclasses import dataclass
from typing import List, Tuple
from coseda.io import log_start, log_result, handle_input, parse_config, shoutout
import shutil
from typing import List, Tuple, Dict, Any

def generate_sorted_grid_points(max_radius, step=0.01):
    """
    Generate all (dx, dy) points on a grid within a circle of radius max_radius,
    sorted by increasing distance from the origin. Coordinates are rounded
    to the precision implied by step.
    """
    points = []
    # Determine decimal places from step
    step_str = f"{step:.6f}".rstrip('0').rstrip('.')
    decimals = 0
    if '.' in step_str:
        decimals = len(step_str.split('.')[1])

    # Number of steps in each direction
    r_steps = int(math.floor(max_radius / step))
    for i in range(-r_steps, r_steps + 1):
        for j in range(-r_steps, r_steps + 1):
            x = i * step
            y = j * step
            if math.hypot(x, y) <= max_radius:
                points.append((round(x, decimals), round(y, decimals)))

    # Sort by squared distance
    points.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    return points

def generate_offset_pairs(step=0.01, layers=1):
    """
    Generate (dx, dy) offset pairs on a grid within a circle of radius layers*step,
    sorted by increasing distance from the origin.
    """
    max_radius = layers * step
    for dx, dy in generate_sorted_grid_points(max_radius, step):
        yield dx, dy

def modify_geometry_file(template_file_path, modified_file_path, x, y):
    """Modify the geometry file with new x, y values."""
    with open(template_file_path, 'r') as file:
        lines = file.readlines()

    with open(modified_file_path, 'w') as file:
        for line in lines:
            if line.startswith("p0/corner_x"):
                file.write(f"p0/corner_x = {x}\n")
            elif line.startswith("p0/corner_y"):
                file.write(f"p0/corner_y = {y}\n")
            else:
                file.write(line)


def create_trial_output_folder(base_dir: str, prefix: str, x: float, y: float) -> str:
    """
    Create and return a trial-specific output folder under base_dir.
    The folder name will be '{prefix}_x{x}_y{y}'.
    """
    trial_folder = os.path.join(base_dir, prefix)
    os.makedirs(trial_folder, exist_ok=True)
    return trial_folder

def run_indexamajig(x, y, geomfile_path, cellfile_path, listfile_path, output_file_base,
                    num_threads, indexing_method, min_peaks,
                    xgandalf_tolerance, xgandalf_sampling_pitch,
                    xgandalf_min_lattice_vector_length, xgandalf_max_lattice_vector_length,
                    xgandalf_iterations, tolerance):

    output_file = f"{output_file_base}.stream"
    output_path = os.path.join(os.path.dirname(listfile_path), output_file)
    listfile = listfile_path

    # assemble core index-only command
    base_command = (
        f"indexamajig -g {geomfile_path} "
        f"-i {listfile_path} "
        f"-o {output_path} "
        f"-j {num_threads} "
        f"-p {cellfile_path} "
        f"--indexing={indexing_method} "
        "--no-retry --no-revalidate "
        "--no-half-pixel-shift "
        "--no-check-cell "
        "--peaks=cxi "
        f"--min-peaks={min_peaks} "
        f"--xgandalf-tolerance={xgandalf_tolerance} "
        f"--xgandalf-sampling-pitch={xgandalf_sampling_pitch} "
        f"--xgandalf-min-lattice-vector-length={xgandalf_min_lattice_vector_length} "
        f"--xgandalf-max-lattice-vector-length={xgandalf_max_lattice_vector_length} "
        f"--xgandalf-grad-desc-iterations={xgandalf_iterations} "
        f"--tolerance={tolerance} "
        "--no-image-data"
    )

    subprocess.run(base_command, shell=True, check=True)

def get_default_panel_corner(h5_path: str) -> Tuple[float, float]:
    """
    Open the given .h5, read the dataset at /entry/data/images,
    and return the default (x0, y0) offsets = (-x_dim/2, -y_dim/2).
    """
    with h5py.File(h5_path, 'r') as f:
        dset = f['/entry/data/images']
        x_dim, y_dim = dset.shape[1], dset.shape[2]
        return -x_dim/2.0, -y_dim/2.0

def generate_list_file(
    h5_file_paths: List[str],
    output_dir: str,
    filename: str = 'list.lst'
) -> str:
    """
    Write a list file of HDF5 paths into the given output directory.

    Args:
        h5_file_paths: Iterable of full paths to .h5 files.
        output_dir: Directory where to write the list file.
        filename: Name of the list file (default 'list.lst').

    Returns:
        The path to the generated list file.
    """
    # ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    list_path = os.path.join(output_dir, filename)
    with open(list_path, 'w') as lf:
        for path in h5_file_paths:
            lf.write(path + '\n')
    return list_path

def gandalf_iterator(geomfile_path, cellfile_path, h5_file, output_file_base,
                     num_threads, indexing_method, resolution_push, integration_method,
                     int_radius, min_peaks, xgandalf_tolerance, xgandalf_sampling_pitch,
                     xgandalf_min_vector_length, xgandalf_max_vector_length,
                     xgandalf_iterations, tolerance):
    # determine default panel corner offsets from the given HDF5 file
    xdefault, ydefault = get_default_panel_corner(h5_file)
    print(f"Running for initial x={xdefault}, y={ydefault}")

    try:
        modify_geometry_file(geomfile_path, geomfile_path, xdefault, ydefault)
        run_indexamajig(
            xdefault, ydefault, geomfile_path, cellfile_path, h5_file,
            output_file_base, num_threads, indexing_method,
            min_peaks, xgandalf_tolerance,
            xgandalf_sampling_pitch, xgandalf_min_vector_length,
            xgandalf_max_vector_length, xgandalf_iterations, tolerance
        )
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        exit()
    except subprocess.CalledProcessError as e:
        print(f"Error during initial indexamajig execution: {e}")
        exit()
    except Exception as e:
        print(f"Unexpected error during initial execution: {e}")
        exit()

    # Continue with radial offsets around default center
    offsets = generate_offset_pairs(step=0.01, layers=2)
    for dx, dy in offsets:
        x = xdefault + dx
        y = ydefault + dy
        print(f"Running for x={x}, y={y}")
        try:
            modify_geometry_file(geomfile_path, geomfile_path, x, y)
            run_indexamajig(
                x, y, geomfile_path, cellfile_path, h5_file,
                output_file_base, num_threads, indexing_method,
                min_peaks, xgandalf_tolerance,
                xgandalf_sampling_pitch, xgandalf_min_vector_length,
                xgandalf_max_vector_length, xgandalf_iterations, tolerance
            )
        except KeyboardInterrupt:
            print("Process interrupted by user.")
            break
        except subprocess.CalledProcessError as e:
            print(f"Error during indexamajig execution: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


@dataclass
class IndexTask:
    x: float
    y: float
    geomfile_path: str
    cellfile_path: str
    h5_file: str
    output_dir: str
    output_file_base: str
    num_threads: int
    indexing_method: str
    resolution_push: float
    integration_method: str
    int_radius: float
    min_peaks: int
    xgandalf_tolerance: float
    xgandalf_sampling_pitch: float
    xgandalf_min_vector_length: float
    xgandalf_max_vector_length: float
    xgandalf_iterations: int
    tolerance: float

def build_index_tasks(
    geomfile_path: str,
    cellfile_path: str,
    h5_file: str,
    output_dir: str,
    output_file_base: str,
    num_threads: int,
    indexing_method: str,
    resolution_push: float,
    integration_method: str,
    int_radius: float,
    min_peaks: int,
    xgandalf_tolerance: float,
    xgandalf_sampling_pitch: float,
    xgandalf_min_vector_length: float,
    xgandalf_max_vector_length: float,
    xgandalf_iterations: int,
    tolerance: float,
    step: float = 0.1,
    layers: int = 1
) -> List[IndexTask]:
    # Compute the true default panel corner from the HDF5 file
    xdefault, ydefault = get_default_panel_corner(h5_file)

    tasks: List[IndexTask] = []
    # First task: default center
    tasks.append(IndexTask(
        xdefault, ydefault,
        geomfile_path, cellfile_path, h5_file,
        output_dir, output_file_base,
        num_threads, indexing_method, resolution_push,
        integration_method, int_radius, min_peaks,
        xgandalf_tolerance, xgandalf_sampling_pitch,
        xgandalf_min_vector_length, xgandalf_max_vector_length,
        xgandalf_iterations, tolerance
    ))
    # Subsequent tasks: sub-pixel offsets around default
    for dx, dy in generate_offset_pairs(step=step, layers=layers):
        tasks.append(IndexTask(
            xdefault + dx, ydefault + dy,
            geomfile_path, cellfile_path, h5_file,
            output_dir, output_file_base,
            num_threads, indexing_method, resolution_push,
            integration_method, int_radius, min_peaks,
            xgandalf_tolerance, xgandalf_sampling_pitch,
            xgandalf_min_vector_length, xgandalf_max_vector_length,
            xgandalf_iterations, tolerance
        ))
    return tasks

def execute_tasks(
    tasks: List[IndexTask],
    geom_template: str
):
    """
    Only geometry file is regenerated per trial. Cell and list files are written once.
    """
    print("=== Offsets to be tried ===")
    x0, y0 = get_default_panel_corner(tasks[0].h5_file)
    for t in tasks:
        dx = t.x - x0
        dy = t.y - y0
        print(f"  dx={dx:+.2f}  dy={dy:+.2f}")
    print("===========================\n")

    output_dir = tasks[0].output_dir
    cellfile_dst = os.path.join(output_dir, "cellfile.cell")
    listfile_dst = os.path.join(output_dir, "list.lst")

    # Only write/copy once
    shutil.copy(tasks[0].cellfile_path, cellfile_dst)
    with open(listfile_dst, 'w') as lf:
        lf.write(tasks[0].h5_file + '\n')

    total = len(tasks)
    for idx, t in enumerate(tasks, start=1):
        dx = t.x - x0
        dy = t.y - y0

        geom_name = f"geometry_dx{dx:+.2f}_dy{dy:+.2f}.geom"
        stream_base = os.path.join(output_dir, f"xgandalf_run_dx{dx:+.2f}_dy{dy:+.2f}")
        geom_path = os.path.join(output_dir, geom_name)
        stream_file = f"{stream_base}.stream"       

        shutil.copy(geom_template, geom_path)
        modify_geometry_file(geom_path, geom_path, t.x, t.y)

        print(f"[{idx}/{total}] Running dx={dx:+.2f}, dy={dy:+.2f} -> {os.path.basename(stream_base)}.stream")

        try:
            run_indexamajig(
                t.x, t.y,
                geom_path,
                cellfile_dst,
                listfile_dst,
                stream_base,  # produces .stream output
                t.num_threads,
                t.indexing_method,
                t.min_peaks,
                t.xgandalf_tolerance,
                t.xgandalf_sampling_pitch,
                t.xgandalf_min_vector_length,
                t.xgandalf_max_vector_length,
                t.xgandalf_iterations,
                t.tolerance
            )
        except subprocess.CalledProcessError as e:
            print(f"Task {idx} failed: {e}")
        except Exception as e:
            print(f"Task {idx} unexpected error: {e}")
        else:
            print(f"Task {idx} completed successfully.")

def run_offset_sweep(
    geom_template: str,
    cell_file: str,
    list_file: str,
    output_dir: str,
    step: float,
    layers: int,
    num_threads: int,
    indexing_method: str,
    min_peaks: int,
    xgandalf_tolerance: float,
    xgandalf_sampling_pitch: float,
    xgandalf_min_lattice_vector_length: float,
    xgandalf_max_lattice_vector_length: float,
    xgandalf_iterations: int,
    tolerance: str,
) -> List[Dict[str, Any]]:
    """
    Runs a grid sweep of geometry offsets and indexes each, skipping
    any run whose .stream file already exists.
    Returns a list of dicts with keys: dx, dy, folder (stream path or None), and status.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine the default panel corner from the first HDF5 in the list
    with open(list_file, 'r') as lf:
        first_h5 = lf.readline().strip()
    x0, y0 = get_default_panel_corner(first_h5)

    # generate and print plan
    offsets = list(generate_offset_pairs(step, layers))
    print("Testing offsets:")
    for dx, dy in offsets:
        print(f"  dx={dx:+.2f}  dy={dy:+.2f}")

    results = []
    for dx, dy in offsets:
        # absolute corner positions
        x = x0 + dx
        y = y0 + dy

        base = f"idx_dx{dx:+.2f}_dy{dy:+.2f}"
        out_base = os.path.join(output_dir, base)
        stream_file = f"{out_base}.stream"

        # skip existing
        if os.path.exists(stream_file):
            print(f"Skipping dx={dx:+.2f}, dy={dy:+.2f}: already have {base}.stream")
            results.append({
                'dx': dx,
                'dy': dy,
                'folder': None,
                'status': 'skipped'
            })
            continue

        # prepare geometry file
        geom_filename = os.path.basename(geom_template)
        geom_path = os.path.join(output_dir, geom_filename)
        shutil.copy(geom_template, geom_path)
        modify_geometry_file(geom_path, geom_path, x, y)

        try:
            folder = run_indexamajig(
                x, y,
                geom_path,
                cell_file,
                list_file,
                out_base,
                num_threads,
                indexing_method,
                min_peaks,
                xgandalf_tolerance,
                xgandalf_sampling_pitch,
                xgandalf_min_lattice_vector_length,
                xgandalf_max_lattice_vector_length,
                xgandalf_iterations,
                tolerance
            )
            status = 'ok'
        except Exception as e:
            folder = None
            status = f"fail: {e}"

        results.append({
            'dx': dx,
            'dy': dy,
            'folder': folder,
            'status': status
        })

    return results