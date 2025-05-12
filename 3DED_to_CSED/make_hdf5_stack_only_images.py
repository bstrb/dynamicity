#!/usr/bin/env python3
# save as make_hdf5_stack.py  and run with `python make_hdf5_stack.py`
import pathlib
import h5py
import numpy as np
import tifffile as tf      # pip install tifffile
import mrcfile             # pip install mrcfile

# ------------------------------------------------------------------------------
# USER SETTINGS — adjust only if your folder / names differ
# ------------------------------------------------------------------------------
imgs_dir   = pathlib.Path("/home/bubl3932/files/MFM300_VIII/MFM300_VIII_2")
base_name  = "MFM300_VIII_2_{:04d}"      # no extension here
file_ext   = ".mrc"                     # choose either ".tiff" or ".mrc"
n_imgs     = 120
out_h5     = imgs_dir / f"{base_name.split('{{')[0]}.h5"
compression = dict(compression="gzip", compression_opts=4, chunks=True)
# ------------------------------------------------------------------------------

def read_image(path: pathlib.Path) -> np.ndarray:
    """Read a single 2D image from TIFF or MRC."""
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return tf.imread(path)
    elif ext == ".mrc":
        with mrcfile.open(path, permissive=True) as mrc:
            # mrc.data may be memmap’d — copy it into RAM
            return np.array(mrc.data)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}")

# ---------- 1) read first image to discover shape / dtype ---------------------
first_path = imgs_dir / f"{base_name.format(0)}{file_ext}"
first_im   = read_image(first_path)
if first_im.ndim != 2:
    raise RuntimeError(f"Expected 2D images, got {first_im.ndim}D array from {first_path}")
ny, nx    = first_im.shape
dtype     = first_im.dtype

# ---------- 2) pre-allocate image stack arrays -------------------------------
stack = np.empty((n_imgs, ny, nx), dtype=dtype)

# ---------- 3) load all images ------------------------------------------------
for i in range(n_imgs):
    path = imgs_dir / f"{base_name.format(i)}{file_ext}"
    img  = read_image(path)
    if img.shape != (ny, nx):
        raise RuntimeError(f"Image {path} has shape {img.shape}, expected {(ny, nx)}")
    stack[i] = img

# ---------- 4) write HDF5 -----------------------------------------------------
with h5py.File(out_h5, "w") as h5:
    grp_data = h5.create_group("entry").create_group("data")
    dset_img = grp_data.create_dataset(
        "images",
        data=stack,
        maxshape=(None, ny, nx),
        **compression
    )
    # Helpful metadata (optional)
    grp_data.attrs["NX_class"]         = "NXdata"
    dset_img.attrs["interpretation"]   = "image"
    dset_img.attrs["signal"]           = 1

print(f"✓  Wrote {out_h5} with {n_imgs} images  ({ny}×{nx})")
