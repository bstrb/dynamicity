import os
import subprocess
import shutil

def convert_hkl_to_mtz(
    output_dir,
    cellfile_path,
    hkl_filename="crystfel.hkl",
    mtz_filename="output.mtz"
):
    """
    Convert an HKL file to an MTZ file using the 'get_hkl' command-line tool.

    This function will look for the input HKL file (by default named 'crystfel.hkl') 
    in 'output_dir', and then run 'get_hkl' to produce the specified MTZ file (by 
    default 'output.mtz'). It will also write standard output and standard error logs
    to 'stdout.log' and 'stderr.log' in the same directory.

    :param output_dir: The directory containing the HKL file and where the MTZ file 
                       and logs will be stored.
    :type output_dir: str
    :param cellfile_path: The path to the cell file used by 'get_hkl'.
    :type cellfile_path: str
    :param hkl_filename: The name of the input HKL file. Defaults to 'crystfel.hkl'.
    :type hkl_filename: str, optional
    :param mtz_filename: The name for the output MTZ file. Defaults to 'output.mtz'.
    :type mtz_filename: str, optional

    :raises ValueError: If 'get_hkl' is not found in the PATH or if expected files 
                        do not exist.
    :raises subprocess.CalledProcessError: If the 'get_hkl' process fails.
    """

    # Check if get_hkl is available in the PATH
    if shutil.which("get_hkl") is None:
        raise ValueError("The command 'get_hkl' is not found in the PATH.")

    # Construct full paths
    input_hkl_path = os.path.join(output_dir, hkl_filename)
    output_mtz_path = os.path.join(output_dir, mtz_filename)
    stdout_log = os.path.join(output_dir, "stdout.log")
    stderr_log = os.path.join(output_dir, "stderr.log")

    # Validate that the input directory exists
    if not os.path.isdir(output_dir):
        raise ValueError(f"The output directory '{output_dir}' does not exist.")

    # Check if the HKL file exists
    if not os.path.isfile(input_hkl_path):
        raise ValueError(f"The input HKL file '{input_hkl_path}' does not exist.")

    # Check if the cell file exists
    if not os.path.isfile(cellfile_path):
        raise ValueError(f"The cell file '{cellfile_path}' does not exist.")

    # Build the get_hkl command
    hkl2mtz_cmd = [
        "get_hkl",
        "-i", input_hkl_path,
        "-o", output_mtz_path,
        "-p", cellfile_path,
        "--output-format=mtz"
    ]

    try:
        # Open logs in append mode
        with open(stdout_log, "a") as stdout, open(stderr_log, "a") as stderr:
            print(f"[INFO] Converting {hkl_filename} to {mtz_filename} in directory: {output_dir}")
            subprocess.run(hkl2mtz_cmd, stdout=stdout, stderr=stderr, check=True)
            print(f"[INFO] Conversion to {mtz_filename} completed successfully in: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert {hkl_filename} to {mtz_filename} in {output_dir}: {e}")
        raise


# Example usage:
if __name__ == "__main__":
    # Replace these paths with valid ones in your environment
    example_output_dir = "/home/bubl3932/files/LauraPacoste_Dynamical-filtering-Buster/with-refine/ReOx-WT_merge_res_25-1.5"
    example_cellfile_path = os.path.join(example_output_dir, "cell.cell")
    try:
        convert_hkl_to_mtz(
            output_dir=example_output_dir,
            cellfile_path=example_cellfile_path,
            hkl_filename="crystfel.hkl",
            mtz_filename="output.mtz"
        )
        print("Conversion completed without errors.")
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"Conversion failed: {exc}")
