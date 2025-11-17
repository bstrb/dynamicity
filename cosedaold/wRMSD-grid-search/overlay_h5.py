#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overlay_h5.py
Create and manage overlay HDF5s that:
- expose /entry/data/images via ExternalLink (or VDS) to the source HDF5
- hold writable per-image /entry/data/det_shift_x_mm and _y_mm arrays (float64)

Original .h5 files remain read-only. We write only to overlays.

Requires: h5py >= 3.x
"""

from __future__ import annotations
import os
from typing import Iterable, Optional, Tuple, Sequence

import numpy as np
import h5py


IMAGES_DS = "/entry/data/images"
SHIFT_X_DS = "/entry/data/det_shift_x_mm"
SHIFT_Y_DS = "/entry/data/det_shift_y_mm"


def ensure_parent(path: str) -> None:
    p = os.path.dirname(os.path.abspath(path))
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _copy_initial_seeds(src: h5py.File, ov: h5py.File) -> None:
    """Copy seed shifts from source to overlay if present; else zeros."""
    n_images = src[IMAGES_DS].shape[0]
    x = ov[SHIFT_X_DS]
    y = ov[SHIFT_Y_DS]

    if SHIFT_X_DS in src and SHIFT_Y_DS in src:
        xs = src[SHIFT_X_DS][...]
        ys = src[SHIFT_Y_DS][...]
        if xs.shape != (n_images,) or ys.shape != (n_images,):
            raise ValueError(
                f"Seed arrays in source have wrong shape: "
                f"{xs.shape}, {ys.shape}; expected {(n_images,)}"
            )
        x[...] = xs
        y[...] = ys
    else:
        x[...] = 0.0
        y[...] = 0.0


def create_overlay(
    h5_src_path: str,
    h5_overlay_path: str,
    use_vds: bool = True,
) -> int:
    """
    Create overlay file (overwrite if exists). Returns number of images N.

    - /entry/data/images: ExternalLink to source (default) or VDS if use_vds=True
    - /entry/data/det_shift_x_mm and _y_mm: writable float64 arrays shape (N,)
    - seeds copied from source if present, else zeros
    """
    h5_src_path = os.path.abspath(h5_src_path)
    h5_overlay_path = os.path.abspath(h5_overlay_path)
    ensure_parent(h5_overlay_path)

    # (Re)create overlay
    if os.path.exists(h5_overlay_path):
        os.remove(h5_overlay_path)

    with h5py.File(h5_src_path, "r") as src, h5py.File(h5_overlay_path, "w") as ov:
        # create groups
        g_entry = ov.require_group("/entry")
        g_data = g_entry.require_group("data")

        # images link
        if not use_vds:
            # ExternalLink: simplest and widely supported
            g_data["images"] = h5py.ExternalLink(h5_src_path, IMAGES_DS)
        else:
            # VDS alternative (if ELINK is restricted in your env)
            # Build a simple VDS that maps the whole first axis
            N, H, W = src[IMAGES_DS].shape
            layout = h5py.VirtualLayout(shape=(N, H, W), dtype=src[IMAGES_DS].dtype)
            vsource = h5py.VirtualSource(h5_src_path, IMAGES_DS, shape=(N, H, W))
            layout[:, :, :] = vsource
            g_data.create_virtual_dataset("images", layout, fillvalue=0)

        # writable shift arrays
        N = src[IMAGES_DS].shape[0]
        g_data.create_dataset("det_shift_x_mm", shape=(N,), dtype="f8")
        g_data.create_dataset("det_shift_y_mm", shape=(N,), dtype="f8")

        # copy initial seeds (if any)
        _copy_initial_seeds(src, ov)

    return N


def write_shifts_mm(
    h5_overlay_path: str,
    indices: Sequence[int],
    dx_mm: Sequence[float],
    dy_mm: Sequence[float],
) -> None:
    """Set per-image absolute shifts (mm) for given indices in overlay."""
    if len(indices) != len(dx_mm) or len(indices) != len(dy_mm):
        raise ValueError("indices, dx_mm, dy_mm must have same length")
    with h5py.File(h5_overlay_path, "r+") as ov:
        x = ov[SHIFT_X_DS]
        y = ov[SHIFT_Y_DS]
        x[np.asarray(indices, dtype=int)] = np.asarray(dx_mm, dtype=float)
        y[np.asarray(indices, dtype=int)] = np.asarray(dy_mm, dtype=float)


def get_seed_shifts_mm(
    h5_overlay_path: str,
    indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read current shifts (mm) from overlay. If indices is None, read all.
    Returns (x_mm, y_mm) arrays.
    """
    with h5py.File(h5_overlay_path, "r") as ov:
        x = ov[SHIFT_X_DS][...]
        y = ov[SHIFT_Y_DS][...]
    if indices is None:
        return x, y
    idx = np.asarray(indices, dtype=int)
    return x[idx], y[idx]


def reset_shifts_to_values(
    h5_overlay_path: str,
    indices: Sequence[int],
    dx_mm: Sequence[float],
    dy_mm: Sequence[float],
) -> None:
    """Reset listed indices to given values (often back to seed)."""
    write_shifts_mm(h5_overlay_path, indices, dx_mm, dy_mm)


def zero_shifts(
    h5_overlay_path: str,
    indices: Optional[Sequence[int]] = None,
) -> None:
    """Set listed indices (or all if None) to 0 mm."""
    with h5py.File(h5_overlay_path, "r+") as ov:
        N = ov[SHIFT_X_DS].shape[0]
        if indices is None:
            ov[SHIFT_X_DS][...] = 0.0
            ov[SHIFT_Y_DS][...] = 0.0
        else:
            idx = np.asarray(indices, dtype=int)
            ov[SHIFT_X_DS][idx] = 0.0
            ov[SHIFT_Y_DS][idx] = 0.0
            
def read_seed_shifts_mm_from_src(h5_src_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read seed shifts directly from the source HDF5. If not present, return zeros.
    """
    h5_src_path = os.path.abspath(h5_src_path)
    with h5py.File(h5_src_path, "r") as f:
        N = f[IMAGES_DS].shape[0]
        if SHIFT_X_DS in f and SHIFT_Y_DS in f:
            xs = f[SHIFT_X_DS][...]
            ys = f[SHIFT_Y_DS][...]
            if xs.shape != (N,) or ys.shape != (N,):
                raise ValueError(f"Seed arrays in source have wrong shape: {xs.shape}, {ys.shape}; expected {(N,)}")
            return xs, ys
        else:
            return np.zeros((N,), dtype="f8"), np.zeros((N,), dtype="f8")
