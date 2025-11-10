#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter an HDF5 by /entry/data/nPeaks >= MIN_PEAKS.
Preserves alignment across all per-frame datasets whose first dimension equals /entry/data/images.shape[0].
Everything else is copied as-is. Compression/chunking/attrs preserved.

Usage:
  python3 filter_by_npeaks.py input.h5 --min-peaks 15 [--images-path /entry/data/images] [--npeaks-path /entry/data/nPeaks] [--out OUTPUT.h5]

If --out is not given, writes: <basename>_min_15peaks.h5
"""

import argparse, os
import numpy as np
import h5py

def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v

def make_chunks_like(src_chunks, new_len):
    if src_chunks is None:
        return None
    ch = list(src_chunks)
    if len(ch) >= 1:
        ch[0] = min(max(1, ch[0]), max(1, new_len))
    return tuple(ch)

def create_like(dst_group, name, src_dset, new_shape):
    return dst_group.create_dataset(
        name,
        shape=new_shape,
        dtype=src_dset.dtype,
        compression=src_dset.compression,
        compression_opts=src_dset.compression_opts,
        shuffle=src_dset.shuffle,
        fletcher32=src_dset.fletcher32,
        scaleoffset=src_dset.scaleoffset,
        chunks=make_chunks_like(src_dset.chunks, new_shape[0] if len(new_shape) else 0),
        fillvalue=src_dset.fillvalue,
        maxshape=None if src_dset.maxshape is None else tuple([new_shape[i] if src_dset.maxshape[i] is not None else None for i in range(len(new_shape))])
    )

def recursive_copy_filtered(src, dst, frame_mask, n_frames, images_path):
    # Walk groups/datasets; filter datasets whose axis-0 == n_frames
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            # g = dst.create_group(key)
            g = dst.require_group(key)
            copy_attrs(item, g)
            recursive_copy_filtered(item, g, frame_mask, n_frames, images_path)
        elif isinstance(item, h5py.Dataset):
            dset = item
            # Identify per-frame dataset: same first axis length as images
            if dset.ndim >= 1 and dset.shape[0] == n_frames:
                # Filter along axis 0 using boolean mask; write in chunks to keep memory low
                new_shape = (int(frame_mask.sum()),) + dset.shape[1:]
                out = create_like(dst, key, dset, new_shape)
                copy_attrs(dset, out)

                # Fancy boolean indexing can be memory heavy; copy in slices
                # Strategy: iterate in blocks over indices where frame_mask is True
                idx = np.nonzero(frame_mask)[0]
                # Choose a block size (tune if needed)
                block = 2048
                wptr = 0
                for s in range(0, len(idx), block):
                    sel = idx[s:s+block]
                    data = dset[sel, ...]
                    out[wptr:wptr+len(sel), ...] = data
                    wptr += len(sel)
            else:
                # Copy entire dataset verbatim
                out = create_like(dst, key, dset, dset.shape)
                copy_attrs(dset, out)
                # Copy in chunks
                if dset.ndim == 0:
                    out[()] = dset[()]
                else:
                    # read/write in chunks along axis 0 if possible
                    total0 = dset.shape[0] if dset.ndim >= 1 else 1
                    block = 2048 if dset.ndim >= 1 else 1
                    w = 0
                    while w < total0:
                        e = min(total0, w + block)
                        if dset.ndim >= 1:
                            out[w:e, ...] = dset[w:e, ...]
                        w = e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_h5")
    ap.add_argument("--min-peaks", type=int, default=15)
    ap.add_argument("--images-path", default="/entry/data/images")
    ap.add_argument("--npeaks-path", default="/entry/data/nPeaks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = os.path.abspath(args.input_h5)
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = args.out or os.path.join(os.path.dirname(in_path), f"{base}_min_{args.min_peaks}peaks.h5")

    with h5py.File(in_path, "r") as f_in:
        if args.images_path not in f_in:
            raise KeyError(f"Missing images dataset: {args.images_path}")
        if args.npeaks_path not in f_in:
            raise KeyError(f"Missing nPeaks dataset: {args.npeaks_path}")

        n_frames = f_in[args.images_path].shape[0]
        npeaks = f_in[args.npeaks_path][...]
        if npeaks.shape[0] != n_frames:
            raise ValueError("nPeaks length does not match images first dimension")

        frame_mask = (npeaks >= args.min_peaks)
        keep_count = int(frame_mask.sum())
        if keep_count == 0:
            raise RuntimeError(f"No frames satisfy nPeaks >= {args.min_peaks}")

        with h5py.File(out_path, "w") as f_out:
            # File-level attributes + provenance
            copy_attrs(f_in, f_out)
            f_out.attrs["_source_file"] = in_path
            f_out.attrs["_min_peaks"] = args.min_peaks
            f_out.attrs["_kept_frames_count"] = keep_count
            # Also store kept indices for downstream
            f_out.create_dataset("/entry/data/kept_frame_indices", data=np.nonzero(frame_mask)[0], dtype=np.int64)

            # Mirror group structure & copy datasets (filtered or full as appropriate)
            recursive_copy_filtered(f_in, f_out, frame_mask, n_frames, args.images_path)

    print(f"[filter_by_npeaks] Wrote: {out_path}")

if __name__ == "__main__":
    main()
