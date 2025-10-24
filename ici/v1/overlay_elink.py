#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_elink.py

Create/manage overlay HDF5s that:
- expose /entry/data/images via ExternalLink to the source HDF5 dataset
- expose (if present) peaks via ExternalLink at:
    /entry/data/peakTotalIntensity
    /entry/data/peakXPosRaw
    /entry/data/peakYPosRaw
- hold writable per-image /entry/data/det_shift_x_mm and _y_mm arrays (float64)
- copy initial seeds from source if present, else zeros

Requires: h5py >= 3.x
"""
from __future__ import annotations
import os
from typing import Optional, Sequence, Dict
import numpy as np
import h5py

IMAGES_DS = "/entry/data/images"
SHIFT_X_DS = "/entry/data/det_shift_x_mm"
SHIFT_Y_DS = "/entry/data/det_shift_y_mm"

# Peaks (raw detector-frame pixels), shape: (n_images, n_peaks)
PEAK_I_DS = "/entry/data/peakTotalIntensity"
PEAK_X_DS = "/entry/data/peakXPosRaw"
PEAK_Y_DS = "/entry/data/peakYPosRaw"
NPEAKS_DS = "/entry/data/nPeaks"


# ---------------- Utils ----------------
def _ensure_parent(path: str) -> None:
    p = os.path.dirname(os.path.abspath(path))
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _copy_initial_seeds(src: h5py.File, ov: h5py.File) -> None:
    n_images = src[IMAGES_DS].shape[0]
    x = ov[SHIFT_X_DS]; y = ov[SHIFT_Y_DS]
    if SHIFT_X_DS in src and SHIFT_Y_DS in src:
        xs = src[SHIFT_X_DS][...]; ys = src[SHIFT_Y_DS][...]
        if xs.shape != (n_images,) or ys.shape != (n_images,):
            raise ValueError(f"Seed arrays wrong shape: {xs.shape}, {ys.shape}; expected {(n_images,)}")
        x[...] = xs; y[...] = ys
    else:
        x[...] = 0.0; y[...] = 0.0


def _link_dataset(ov: h5py.File, ov_path: str, target_filename: str, target_path: str) -> None:
    """
    Create/overwrite an ExternalLink at ov_path pointing to target_filename::target_path.
    Ensures parent groups exist.
    """
    parts = [p for p in ov_path.split("/") if p]  # ignore leading "/"
    g = ov["/"]
    for name in parts[:-1]:
        g = g.require_group(name)
    final = parts[-1]
    if final in g:
        del g[final]
    g[final] = h5py.ExternalLink(target_filename, target_path)


def _find_peak_paths(src: h5py.File, n_images: int) -> Dict[str, Optional[str]]:
    """
    Discover peak datasets anywhere in the source file by basename:
      'peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw'
    Return dict {'I': abs_path_or_None, 'X': abs_path_or_None, 'Y': abs_path_or_None}.
    Validates first dimension == n_images and (when present) that X/Y/I share the same second dimension.
    """
    found = {"I": None, "X": None, "Y": None}
    shapes: Dict[str, tuple] = {}

    def maybe_take(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        base = name.split("/")[-1]
        # Normalize to absolute path
        abs_name = "/" + name.lstrip("/")
        if base == "peakTotalIntensity":
            found["I"] = abs_name; shapes["I"] = obj.shape
        elif base == "peakXPosRaw":
            found["X"] = abs_name; shapes["X"] = obj.shape
        elif base == "peakYPosRaw":
            found["Y"] = abs_name; shapes["Y"] = obj.shape
        elif base == "nPeaks":
            found["N"] = abs_name; shapes["N"] = obj.shape

    src.visititems(maybe_take)

    # If none found, bail quietly
    if not any(found.values()):
        return found

    # Validate first dimension (when present)
    for k in ("X", "Y", "I"):
        if found[k]:
            s = shapes[k]
            if len(s) < 1 or s[0] != n_images:
                raise ValueError(f"Peak dataset {found[k]} has shape {s}, expected first dim = {n_images}")

    # Validate shared second dimension across those present
    dims = [shapes[k][1] for k in ("X", "Y", "I") if k in shapes and len(shapes[k]) >= 2]
    if len(dims) >= 2 and not all(d == dims[0] for d in dims):
        raise ValueError(f"Peak arrays second dimension mismatch: {shapes}")

    return found


# ---------------- Public API ----------------
def create_overlay(h5_src_path: str, h5_overlay_path: str) -> int:
    """
    Create overlay file (overwrite if exists). Returns number of images N.
    - /entry/data/images: ExternalLink -> source
    - /entry/data/det_shift_x_mm, _y_mm: float64 arrays shape (N,)
    - If present in source (anywhere), link peaks and expose them at:
        /entry/data/peakTotalIntensity
        /entry/data/peakXPosRaw
        /entry/data/peakYPosRaw
    """
    h5_src_path = os.path.abspath(h5_src_path)
    h5_overlay_path = os.path.abspath(h5_overlay_path)
    _ensure_parent(h5_overlay_path)
    if os.path.exists(h5_overlay_path):
        os.remove(h5_overlay_path)

    with h5py.File(h5_src_path, "r") as src, h5py.File(h5_overlay_path, "w") as ov:
        # Validate and get N
        if IMAGES_DS not in src:
            raise KeyError(f"Missing dataset in source: {IMAGES_DS}")
        N = src[IMAGES_DS].shape[0]

        # create groups
        g_entry = ov.require_group("/entry")
        g_data  = g_entry.require_group("data")

        # External link to images (use absolute file path for robustness)
        if "images" in g_data:
            del g_data["images"]
        g_data["images"] = h5py.ExternalLink(h5_src_path, IMAGES_DS)

        # Shift arrays (persisted in overlay)
        if "det_shift_x_mm" in g_data: del g_data["det_shift_x_mm"]
        if "det_shift_y_mm" in g_data: del g_data["det_shift_y_mm"]
        g_data.create_dataset("det_shift_x_mm", shape=(N,), dtype="f8")
        g_data.create_dataset("det_shift_y_mm", shape=(N,), dtype="f8")
        _copy_initial_seeds(src, ov)

        # Optional peaks: auto-discover actual source paths, then expose at standard overlay paths
        peak_src = _find_peak_paths(src, N)  # {'I': '/.../peakTotalIntensity', 'X': '/.../peakXPosRaw', 'Y': '/.../peakYPosRaw'}
        linked_any = False
        if peak_src["X"]:
            _link_dataset(ov, PEAK_X_DS, h5_src_path, peak_src["X"]); linked_any = True
        if peak_src["Y"]:
            _link_dataset(ov, PEAK_Y_DS, h5_src_path, peak_src["Y"]); linked_any = True
        if peak_src["I"]:
            _link_dataset(ov, PEAK_I_DS, h5_src_path, peak_src["I"]); linked_any = True
        if peak_src["N"]:
            _link_dataset(ov, NPEAKS_DS, h5_src_path, peak_src["N"]); linked_any = True

        # (optional) annotate units if peaks are present
        if linked_any:
            try:
                ov[PEAK_X_DS].attrs["units"] = "pixel"
                ov[PEAK_Y_DS].attrs["units"] = "pixel"
                ov[PEAK_I_DS].attrs["description"] = "peak total intensity"
                ov[NPEAKS_DS].attrs["description"] = "number of peaks"
            except Exception:
                pass

    return N


def write_shifts_mm(h5_overlay_path: str, indices: Sequence[int], dx_mm: Sequence[float], dy_mm: Sequence[float]) -> None:
    if len(indices) != len(dx_mm) or len(indices) != len(dy_mm):
        raise ValueError("indices, dx_mm, dy_mm must have same length")
    with h5py.File(h5_overlay_path, "r+") as ov:
        x = ov[SHIFT_X_DS]; y = ov[SHIFT_Y_DS]
        idx = np.asarray(indices, dtype=int)
        x[idx] = np.asarray(dx_mm, dtype=float)
        y[idx] = np.asarray(dy_mm, dtype=float)
