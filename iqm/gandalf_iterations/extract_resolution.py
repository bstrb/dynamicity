#!/usr/bin/env python3

import sys

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

if __name__ == "__main__":
    
    geom_file = "/Users/xiaodong/Desktop/simulations/LTA/LTAsim.geom"
    try:
        resolution = extract_resolution(geom_file)
        print(f"mm per pixel: {1000/resolution}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
