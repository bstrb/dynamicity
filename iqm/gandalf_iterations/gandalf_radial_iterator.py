import os
import subprocess
from tqdm import tqdm

from .list_h5_files import list_h5_files
from .run_indexamajig import run_indexamajig
from .extract_resolution import extract_resolution
from .perturb_det_shifts import perturb_det_shifts
from .uniform_radial_xy_pairs import generate_sorted_grid_points

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


