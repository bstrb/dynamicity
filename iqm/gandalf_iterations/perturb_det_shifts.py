#!/usr/bin/env python3

import h5py

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
