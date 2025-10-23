#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_elink.py

Create/manage overlay HDF5s that:
- expose /entry/data/images via ExternalLink to the source HDF5 dataset
- hold writable per-image /entry/data/det_shift_x_mm and _y_mm arrays (float64)
- copy initial seeds from source if present, else zeros

Requires: h5py >= 3.x
"""
from __future__ import annotations
import os
from typing import Optional, Sequence, Tuple
import numpy as np
import h5py

IMAGES_DS = "/entry/data/images"
SHIFT_X_DS = "/entry/data/det_shift_x_mm"
SHIFT_Y_DS = "/entry/data/det_shift_y_mm"

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

def create_overlay(h5_src_path: str, h5_overlay_path: str) -> int:
    """
    Create overlay file (overwrite if exists). Returns number of images N.
    - /entry/data/images: ExternalLink -> source
    - /entry/data/det_shift_x_mm, _y_mm: float64 arrays shape (N,)
    """
    h5_src_path = os.path.abspath(h5_src_path)
    h5_overlay_path = os.path.abspath(h5_overlay_path)
    _ensure_parent(h5_overlay_path)
    if os.path.exists(h5_overlay_path):
        os.remove(h5_overlay_path)

    with h5py.File(h5_src_path, "r") as src, h5py.File(h5_overlay_path, "w") as ov:
        g_entry = ov.require_group("/entry")
        g_data  = g_entry.require_group("data")
        # External link to images
        g_data["images"] = h5py.ExternalLink(h5_src_path, IMAGES_DS)
        # shift arrays
        N = src[IMAGES_DS].shape[0]
        g_data.create_dataset("det_shift_x_mm", shape=(N,), dtype="f8")
        g_data.create_dataset("det_shift_y_mm", shape=(N,), dtype="f8")
        _copy_initial_seeds(src, ov)
    return N

def write_shifts_mm(h5_overlay_path: str, indices: Sequence[int], dx_mm: Sequence[float], dy_mm: Sequence[float]) -> None:
    if len(indices) != len(dx_mm) or len(indices) != len(dy_mm):
        raise ValueError("indices, dx_mm, dy_mm must have same length")
    with h5py.File(h5_overlay_path, "r+") as ov:
        x = ov[SHIFT_X_DS]; y = ov[SHIFT_Y_DS]
        idx = np.asarray(indices, dtype=int)
        x[idx] = np.asarray(dx_mm, dtype=float)
        y[idx] = np.asarray(dy_mm, dtype=float)
