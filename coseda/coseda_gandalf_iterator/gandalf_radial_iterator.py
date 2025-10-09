import os
import h5py
import math
import subprocess
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

    Args:
        x (float): Initial beam center X coordinate in pixels.
        y (float): Initial beam center Y coordinate in pixels.
        geomfile_path (str): Path to the .geom file.
        cellfile_path (str): Path to the .cell file containing cell parameters.
        input_path (str): Path to the folder where .h5 files reside (and where output is stored).
        output_file_base (str): Base name for output files (e.g., 'LTA'); final filenames will be 'base_x_y.h5'.
        num_threads (int): Number of CPU threads to use.
        max_radius (float): Maximum radius for the grid search, in pixels.
        step (float): Grid step size in pixels (the smaller, the finer the grid).
        extra_flags (list): Additional command-line flags to pass to 'indexamajig'.

    Returns:
        None. Outputs multiple .stream and .h5 files in the input_path folder.

    Notes:
        - The function performs a radial scan of beam centers around (x, y).
        - Each new (x, y) is processed with the same CrystFEL parameters.
        - Make sure CrystFEL is installed and in your PATH.
    """
    listfile_path = list_h5_files(input_path)
    output_folder = os.path.join(input_path, f"xgandalf_iterations_max_radius_{max_radius}_step_{step}")
    os.makedirs(output_folder, exist_ok = True)

    xy_pairs = list(generate_sorted_grid_points(max_radius=max_radius, step=step))
    print(f"Resulting streamfiles will be saved in {output_folder}")
    res = extract_resolution(geomfile_path)
    mm_per_pixel = 1000/res
    
    for x, y in tqdm(xy_pairs, desc="Processing XY pairs"):
        print(f"Running for pixel shifts x = {x}, y = {y}")
        output_path = f"{output_folder}/{output_file_base}_{x}_{y}.stream"
        # Convert pixel shifts to millimeters.
        shift_x = x * mm_per_pixel
        shift_y = y * mm_per_pixel
        try:
            perturb_det_shifts(listfile_path, shift_x, shift_y)  # Apply the shifts
            run_indexamajig(geomfile_path, listfile_path, cellfile_path, output_path, num_threads, extra_flags=extra_flags)
        except KeyboardInterrupt:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)  # Always reset the shifts
            print("Process interrupted by user.")
            break
        except subprocess.CalledProcessError as e:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)  # Always reset the shifts
            print(f"Error during indexamajig execution: {e}")
            break
        except Exception as e:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)  # Always reset the shifts
            print(f"Unexpected error: {e}")
            break
        else:
            perturb_det_shifts(listfile_path, -shift_x, -shift_y)

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
    round them based on the step size, and sort them in order of increasing radial distance from the center.
    
    Returns:
        List of (x, y) tuples sorted from the center outward.
    """
    x_center, y_center = 0,0
    points = grid_points_in_circle(x_center, y_center, max_radius, step)
    # Sort by the squared distance from the center (no need for square roots)
    points.sort(key=lambda pt: (pt[0] - x_center) ** 2 + (pt[1] - y_center) ** 2)
    print(f"Generated {len(points)} grid points in the circle.")
    return points