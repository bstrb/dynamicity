#!/usr/bin/env python3
import os

def format_number(value, is_sigma=False):
    """
    Format a float to a maximum of 7 characters (excluding left padding)
    so that when right adjusted in an 8-character field it meets the SHELX style.
    
    Rules:
      - For nonzero values:
          * If the integer part has 6 digits, output with no decimals but add a trailing period.
          * If the integer part has 5 digits, output with 1 decimal.
          * If the integer part has 4 or fewer digits, output with 2 decimals.
      - For zero:
          * For intensity (is_sigma False): return "0."
          * For sigma (is_sigma True): return "0.00"
    """
    if value == 0:
        return "0.00" if is_sigma else "0."
    
    int_part = abs(int(value))
    digits = len(str(int_part))
    
    if digits >= 6:
        decimals = 0
    elif digits == 5:
        decimals = 1
    else:
        decimals = 2

    formatted = f"{value:.{decimals}f}"
    
    # Ensure a trailing period exists if there are no decimals.
    if decimals == 0 and "." not in formatted:
        if len(formatted) + 1 <= 7:
            formatted = formatted + "."
    
    # If rounding makes the string too long, fallback by trimming.
    if len(formatted) > 7:
        formatted = formatted[:7]
    return formatted

def convert_hkl_crystfel_to_shelx(input_dir: str):
    """
    Convert an crystfel formatted HKL file to an SHELX compatible HKL.

    This function will look for the input HKL file (by default named 'crystfel.hkl') 
    in 'input_dir', and then  produce the specified hkl file (by default 'shelx.hkl').

    :param input_dir: The directory containing the crystfel.hkl file and where the shelx dir
                    containing the shelx.hkl file will be created.
    """
    input_filename = os.path.join(input_dir,"crystfel.hkl")
    output_filename = os.path.join(os.path.dirname(input_filename), "shelx/shelx.hkl")
    print(f"[INFO] Converting {os.path.basename(input_filename)} to {os.path.basename(output_filename)} in directory: {os.path.dirname(input_filename)}")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    reflections = []
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    in_reflections = False
    for line in lines:
        if not in_reflections and "sigma(I)" in line:
            in_reflections = True
            continue

        if in_reflections:
            if line.strip() == "" or line.strip().startswith("End"):
                break

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                I = float(parts[3])
                sigma = float(parts[5])
            except ValueError:
                continue

            reflections.append((h, k, l, I, sigma))

    with open(output_filename, 'w') as out:
        # Each reflection is written with fixed width fields.
        for h, k, l, I, sigma in reflections:
            I_str = format_number(I, is_sigma=False)
            sigma_str = format_number(sigma, is_sigma=True)
            out.write(f"{h:4d}{k:4d}{l:4d}{I_str:>8}{sigma_str:>8}{1:>4}\n")
        # Append final line with zeros.
        final_I = format_number(0, is_sigma=False)
        final_sigma = format_number(0, is_sigma=True)
        out.write(f"{0:4d}{0:4d}{0:4d}{final_I:>8}{final_sigma:>8}{0:>4}\n")

    print(f"[INFO] Conversion to {os.path.basename(output_filename)} completed successfully in: {os.path.join(os.path.dirname(input_filename))}/shelx")
  
if __name__ == "__main__":

    input_dir = "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/zero_fitering/filtered_metrics_merge"
    convert_hkl_crystfel_to_shelx(input_dir)
    input_dir = "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/zero_fitering/filtered_metrics_sorted_5000_first_xtals_noZOLZ_tol0.1_merge"
    convert_hkl_crystfel_to_shelx(input_dir)
    input_dir = "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/xgandalf_iterations_max_radius_0.5_step_0.2/zero_fitering/filtered_metrics_sorted_50000_first_xtals_noZOLZ_tol0.1_merge"
    convert_hkl_crystfel_to_shelx(input_dir)
