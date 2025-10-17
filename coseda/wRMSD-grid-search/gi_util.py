# coseda/gi_util.py
from __future__ import annotations
import os, h5py
from pathlib import Path
from datetime import datetime

def estimate_grid_points(max_radius_px: float, step_px: float) -> int:
    import math
    if step_px <= 0.0 or max_radius_px < 0.0: return 0
    if max_radius_px == 0.0: return 1
    try:
        decimals = max(0, -int(math.floor(math.log10(step_px)))) if step_px > 0 else 6
        decimals = min(6, decimals)
    except Exception:
        decimals = 6
    limit = int(math.ceil(max_radius_px / step_px))
    R2 = max_radius_px * max_radius_px
    count = 0
    for i in range(-limit, limit + 1):
        x = round(i * step_px, decimals); x2 = x * x
        if x2 > R2 + 1e-12: continue
        for j in range(-limit, limit + 1):
            y = round(j * step_px, decimals)
            if x2 + y*y <= R2 + 1e-12: count += 1
    return count

def count_images_in_h5_folder(folder: Path) -> int:
    total = 0
    for ext in ("*.h5", "*.hdf5", "*.cxi"):
        for h5path in folder.glob(ext):
            try:
                with h5py.File(h5path, "r") as h5f:
                    if "/entry/data/images" in h5f:
                        ds = h5f["/entry/data/images"]
                        if ds.ndim >= 3:
                            total += int(ds.shape[0])
            except Exception:
                pass
    return total

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def timestamp_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"indexingintegration_{ts}"

def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def append_line(widget_or_self, text: str) -> None:
    try:
        widget_or_self.appendPlainText(text)
    except Exception:
        try:
            widget_or_self.append(text)
        except Exception:
            print(text)
