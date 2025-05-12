#!/usr/bin/env python3
"""
make_hdf5_stack_with_shifts.py
––––––––––––––––––––––––––––––
Create a NeXus‑style HDF5 file that contains

  /entry/data/images        (uint16, n_imgs × ny × nx)
  /entry/data/center_x      (float64, n_imgs)
  /entry/data/center_y      (float64, n_imgs)
  /entry/data/det_shift_x_mm
  /entry/data/det_shift_y_mm

using the image stack in  *.tiff  files and the metadata that resides
in an associated  *.pts2  file.
"""

import pathlib
import re
import h5py
import numpy as np
import tifffile as tf   #  pip install tifffile h5py numpy

# ─────────────────────────────── USER SETTINGS ────────────────────────────────
imgs_dir     = pathlib.Path("/home/bubl3932/files/MFM300_VIII/Al_check_0p26/tiff")
file_pattern = "Al_check_0p26_{:04d}.tiff"      # <basename>_{0000…0119}.tiff
pts2_file    = pathlib.Path("/home/bubl3932/files/MFM300_VIII/Al_check_0p26/PETS.pts2")
out_h5       = pathlib.Path("/home/bubl3932/files/MFM300_VIII/Al_check_0p26/Al_check_0p26_det_shifts.h5")  # output file name

n_imgs              = 120                # number of frames to read
pixels_per_meter    = 17857.14285714286 # camera calibration
compression_opts    = dict(compression="gzip",
                           compression_opts=4,
                           chunks=True)
# ───────────────────────────────────────────────────────────────────────────────


def load_stack(directory: pathlib.Path,
               pattern: str,
               n: int) -> tuple[np.ndarray, int, int, np.dtype]:
    """Read n TIFFs into a NumPy 3‑D array and return (stack, ny, nx, dtype)."""
    first = tf.imread(directory / pattern.format(0))
    ny, nx = first.shape
    dtype  = first.dtype
    stack  = np.empty((n, ny, nx), dtype=dtype)
    stack[0] = first
    for i in range(1, n):
        stack[i] = tf.imread(directory / pattern.format(i))
    return stack, ny, nx, dtype


def parse_centers(pts2: pathlib.Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract xcenter / ycenter columns (8th & 9th) from the imagelist section."""
    rgx = re.compile(r'^"tiff/[^"]+"\s+(.+)$')     # match any line that starts with "tiff/…"
    cx = np.empty(n, dtype=np.float64)
    cy = np.empty(n, dtype=np.float64)

    with pts2.open() as fh:
        idx = 0
        for line in fh:
            if line.lstrip().startswith('#') or not line.strip():
                continue
            m = rgx.match(line)
            if not m:
                continue
            fields = m.group(1).split()
            cx[idx] = float(fields[6])   # zero‑based index 7  → xcenter
            cy[idx] = float(fields[7])   # zero‑based index 8  → ycenter
            idx += 1
    if idx != n:
        raise RuntimeError(f"Expected {n} image lines, parsed {idx}")
    return cx, cy


def main():
    # 1) images & centers
    stack, ny, nx, _dtype = load_stack(imgs_dir, file_pattern, n_imgs)
    center_x, center_y    = parse_centers(pts2_file, n_imgs)

    # 2) detector‑shifts  (pixel → metre → mm, negated so that
    #    a positive pixel shift right/down becomes a *negative* mm shift)
    presumed_center_x = nx / 2.0
    presumed_center_y = ny / 2.0

    det_shift_x_mm = -((center_x - presumed_center_x) / pixels_per_meter) * 1_000.0
    det_shift_y_mm = -((center_y - presumed_center_y) / pixels_per_meter) * 1_000.0

    # 3) write HDF5
    with h5py.File(out_h5, "w") as h5:
        grp_data = h5.create_group("entry").create_group("data")

        grp_data.create_dataset("images",
                                data=stack,
                                maxshape=(None, ny, nx),
                                **compression_opts)

        grp_data.create_dataset("center_x",
                                data=center_x,
                                maxshape=(None,),
                                **compression_opts)

        grp_data.create_dataset("center_y",
                                data=center_y,
                                maxshape=(None,),
                                **compression_opts)

        grp_data.create_dataset("det_shift_x_mm",
                                data=det_shift_x_mm,
                                maxshape=(None,),
                                **compression_opts)

        grp_data.create_dataset("det_shift_y_mm",
                                data=det_shift_y_mm,
                                maxshape=(None,),
                                **compression_opts)

        # — optional but handy NeXus metadata —
        grp_data.attrs["NX_class"] = "NXdata"
        grp_data["images"].attrs["signal"] = 1
        grp_data["images"].attrs["interpretation"] = "image"
        for name in ("center_x", "center_y"):
            grp_data[name].attrs["units"] = "pixel"
        for name in ("det_shift_x_mm", "det_shift_y_mm"):
            grp_data[name].attrs["units"] = "millimetre"

    print(f"✓  Wrote {out_h5} with:")
    print(f"   images          : {stack.shape}  uint{stack.dtype.itemsize*8}")
    print(f"   center_x / y    : {center_x.shape}")
    print(f"   det_shift_x / y : {det_shift_x_mm.shape}")


if __name__ == "__main__":
    main()
