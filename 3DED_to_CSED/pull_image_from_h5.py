import h5py
import numpy as np
from PIL import Image
from pathlib import Path

# ------------------------------------------------------------------
# 1.  File & dataset bookkeeping
# ------------------------------------------------------------------
h5_path   = Path("/home/bubl3932/files/MFM300_VIII/MFM300_VIII_spot2_20250408_1511/"
                 "MFM300_VIII_spot2_20250408_1511.h5")
dataset   = "/entry/data/images"      # internal HDF5 path
frame_id  = 15068                     # the frame you want
out_file  = Path("pattern_15068.tif") # will be created next to the notebook

# ------------------------------------------------------------------
# 2.  Read the single frame into RAM
# ------------------------------------------------------------------
with h5py.File(h5_path, "r") as f:
    dset = f[dataset]                # <HDF5 dataset object>
    if frame_id >= len(dset):
        raise IndexError(f"Frame {frame_id} is out of range: dataset has {len(dset)} frames")
    img  = dset[frame_id]            # numpy array, whatever dtype your camera stored
print("Raw shape:", img.shape, img.dtype)

# ------------------------------------------------------------------
# 3.  Guarantee 1024 × 1024 (skip if already the right size)
# ------------------------------------------------------------------
target_size = (1024, 1024)
if img.shape != target_size:
    # a) Simple centre‑crop, or
    # b) binning/re‑sampling.  Pick ONE that fits your workflow.
    #
    # ----- a) Centre‑crop -----
    y0 = (img.shape[0] - target_size[0]) // 2
    x0 = (img.shape[1] - target_size[1]) // 2
    img = img[y0:y0+1024, x0:x0+1024]

    # ----- b) Bin down by integer factor if exactly 2k × 2k, for example -----
    # factor = img.shape[0] // 1024
    # img = img.reshape(1024, factor, 1024, factor).mean(axis=(1,3))

# ------------------------------------------------------------------
# 4.  Convert to 8‑bit for comfortable painting in GUI tools
# ------------------------------------------------------------------
img8 = img.astype(np.float32)
p0, p99 = np.percentile(img8, (0.5, 99.5))   # robust stretch
img8 = np.clip((img8 - p0) / (p99 - p0), 0, 1) * 255
img8 = img8.astype(np.uint8)

# ------------------------------------------------------------------
# 5.  Save as an 8‑bit TIFF
# ------------------------------------------------------------------
Image.fromarray(img8).save(out_file)
print("Wrote", out_file.resolve())
