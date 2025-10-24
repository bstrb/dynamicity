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
from typing import Optional, Sequence
import numpy as np
import h5py

IMAGES_DS = "/entry/data/images"
SHIFT_X_DS = "/entry/data/det_shift_x_mm"
SHIFT_Y_DS = "/entry/data/det_shift_y_mm"

# Peaks (raw detector-frame pixels), shape: (n_images, n_peaks)
PEAK_I_DS = "/entry/data/peakTotalIntensity"
PEAK_X_DS = "/entry/data/peakXPosRaw"
PEAK_Y_DS = "/entry/data/peakYPosRaw"

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

def _link_if_present(src: h5py.File, ov: h5py.File, src_path: str, ov_path: Optional[str] = None) -> bool:
    """
    Create an ExternalLink in 'ov' pointing to 'src_path' in 'src' if it exists.
    Returns True if linked, False otherwise.
    """
    if ov_path is None:
        ov_path = src_path
    try:
        # verify existence & shape
        _ = src[src_path]
    except Exception:
        return False
    # ensure parent groups exist in overlay
    parts = [p for p in ov_path.split("/") if p]  # ignore leading "/"
    g = ov["/"]
    for name in parts[:-1]:
        g = g.require_group(name)
    # install the link at the final name
    final_name = parts[-1]
    if final_name in g:
        del g[final_name]
    g[final_name] = h5py.ExternalLink(src.filename, src_path)
    return True

def _validate_optional_peaks(src: h5py.File) -> None:
    """
    If peaks are present in the source, perform light validation:
    - they share the same first dimension (n_images) as images
    - X/Y/Intensity shapes agree with each other
    """
    if not (PEAK_X_DS in src and PEAK_Y_DS in src and PEAK_I_DS in src):
        return
    n_img = src[IMAGES_DS].shape[0]
    sx = src[PEAK_X_DS].shape
    sy = src[PEAK_Y_DS].shape
    si = src[PEAK_I_DS].shape
    if not (sx[0] == sy[0] == si[0] == n_img):
        raise ValueError(
            f"Peaks first dimension must match images: "
            f"images={n_img}, peakX={sx}, peakY={sy}, peakI={si}"
        )
    if not (sx[1] == sy[1] == si[1]):
        raise ValueError(
            f"Peak arrays second dimension (n_peaks) must agree: "
            f"peakX={sx}, peakY={sy}, peakI={si}"
        )

def create_overlay(h5_src_path: str, h5_overlay_path: str) -> int:
    """
    Create overlay file (overwrite if exists). Returns number of images N.
    - /entry/data/images: ExternalLink -> source
    - /entry/data/det_shift_x_mm, _y_mm: float64 arrays shape (N,)
    - If present in source, link peaks:
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

        # External link to images
        if "images" in g_data:
            del g_data["images"]
        g_data["images"] = h5py.ExternalLink(h5_src_path, IMAGES_DS)

        # Shift arrays (persisted in overlay)
        if "det_shift_x_mm" in g_data: del g_data["det_shift_x_mm"]
        if "det_shift_y_mm" in g_data: del g_data["det_shift_y_mm"]
        g_data.create_dataset("det_shift_x_mm", shape=(N,), dtype="f8")
        g_data.create_dataset("det_shift_y_mm", shape=(N,), dtype="f8")
        _copy_initial_seeds(src, ov)

        # Optional peaks: link at the SAME path names as source
        _validate_optional_peaks(src)
        linked_any = False
        linked_any |= _link_if_present(src, ov, PEAK_X_DS)
        linked_any |= _link_if_present(src, ov, PEAK_Y_DS)
        linked_any |= _link_if_present(src, ov, PEAK_I_DS)

        # (optional) annotate units if peaks are present
        if linked_any:
            try:
                ov[PEAK_X_DS].attrs["units"] = "pixel"
                ov[PEAK_Y_DS].attrs["units"] = "pixel"
                ov[PEAK_I_DS].attrs["description"] = "peak total intensity"
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
