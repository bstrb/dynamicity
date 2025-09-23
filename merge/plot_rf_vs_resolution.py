#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# --------- parsing helpers ---------

def find_last_cycle_line(content: str):
    """
    Returns the character index at the start of the last ':Cycle <n>' header.
    """
    cycle_pattern = re.compile(r'^:Cycle\s+\d+', re.MULTILINE)
    matches = list(cycle_pattern.finditer(content))
    return matches[-1].start() if matches else None

def extract_data_from_section(content: str, last_cycle_line: int):
    """
    From the last cycle section:
      - find line containing 'Rf_used'
      - then find the next '$$' line -> start of data block
      - read rows until the following '$$'
    Returns a list of lines containing numeric data.
    """
    lines = content[last_cycle_line:].splitlines()
    numerical_section = []
    rf_used_found = False
    in_data_block = False
    delimiter_pattern = re.compile(r'^\$\$')

    for line in lines:
        if not rf_used_found and 'Rf_used' in line:
            rf_used_found = True
            continue
        if rf_used_found and not in_data_block and delimiter_pattern.match(line):
            in_data_block = True
            continue
        if in_data_block:
            if delimiter_pattern.match(line):
                break
            stripped = line.strip()
            if stripped:
                numerical_section.append(stripped)

    return numerical_section if numerical_section else None

def format_extracted_data(numerical_section):
    """
    Converts lines of space-separated numbers into an NxM numpy array.
    Expects at least 6 columns so that column 0 = resolution, column 5 = Rf_used.
    """
    rows = []
    for line in numerical_section:
        # split on any whitespace; ignore multiple spaces
        parts = re.split(r'\s+', line.strip())
        # skip lines that aren't fully numeric
        try:
            vals = list(map(float, parts))
        except ValueError:
            continue
        rows.append(vals)

    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    if arr.shape[1] < 6:
        raise ValueError(
            f"Parsed table has {arr.shape[1]} columns; expected at least 6 "
            f"(need col0=resolution, col5=Rf_used)."
        )
    return arr

def load_rf_table(txt_path: str):
    """
    Reads a refmac-style log .txt and returns (resolution, rf_used) arrays from the last cycle block.
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    last_pos = find_last_cycle_line(content)
    if last_pos is None:
        raise RuntimeError(f"No ':Cycle' section found in {txt_path}")

    block = extract_data_from_section(content, last_pos)
    if block is None:
        raise RuntimeError(f"No numerical block after 'Rf_used' in {txt_path}")

    data = format_extracted_data(block)
    if data is None:
        raise RuntimeError(f"Failed to parse numerical data in {txt_path}")

    resolution = data[:, 0]   # column 0
    rf_used   = data[:, 5]    # column 5 (Rf_used)
    return resolution, rf_used

# --------- plotting ---------

def plot_two_runs(path_a: str, path_b: str, label_a: str = None, label_b: str = None, out_png: str = None):
    res_a, rf_a = load_rf_table(path_a)
    res_b, rf_b = load_rf_table(path_b)

    # Convert 1/d^2 -> d (Å)
    res_a = 1.0 / np.sqrt(res_a)
    res_b = 1.0 / np.sqrt(res_b)

    if label_a is None:
        label_a = os.path.basename(os.path.dirname(path_a))
    if label_b is None:
        label_b = os.path.basename(os.path.dirname(path_b))

    plt.figure(figsize=(8, 5))
    plt.plot(res_a, rf_a, marker='o', linestyle='-', label=label_a)
    plt.plot(res_b, rf_b, marker='o', linestyle='-', label=label_b)

    plt.title("Rf_used vs Resolution")
    plt.xlabel("Resolution (Å)")
    plt.ylabel("Rf_used")
    plt.legend()
    # Crystallography convention: highest resolution at left
    ax = plt.gca()
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.4)

    if out_png:
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        print(f"Saved figure: {out_png}")

    plt.tight_layout()
    plt.show()

# --------- inputs (your exact two files) ---------

path1 = "/home/bubl3932/files/UOX1/xgandalf_iterations_max_radius_0.71_step_0.5/metrics_run_20250922-151741/filtered_metrics_sorted_noZOLZ_tol0.1_merge_res_20-1.3/metrics_run_20250922-151741_output_bins_20_minres_2.txt"
path2 = "/home/bubl3932/files/UOX1/xgandalf_iterations_max_radius_0.71_step_0.5/metrics_run_20250922-151741/filtered_metrics_merge_res_20-1.3/metrics_run_20250922-151741_output_bins_20_minres_2.txt"

# Optional output image (uncomment to save):
# out_png = "/home/bubl3932/files/UOX1/rf_used_vs_resolution.png"
out_png = None

if __name__ == "__main__":
    plot_two_runs(
        path1,
        path2,
        label_a="filtered_metrics_sorted_noZOLZ_tol0.1_merge_res_20-1.3",
        label_b="filtered_metrics_sorted_merge_res_20-1.3",
        out_png=out_png
    )
