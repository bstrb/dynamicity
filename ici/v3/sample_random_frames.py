#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly sample N frames (without replacement) from an HDF5 (typically the filtered output).
Preserves alignment across all per-frame datasets. Compression/chunking/attrs preserved.

Usage:
  python3 sample_random_frames.py input.h5 --count 5000 [--seed 1337] [--images-path /entry/data/images] [--out OUTPUT.h5]

If --out is not given, the script tries to use provenance to emit:
  <original_basename>_min_<min_peaks>peaks_<count>.h5
falling back to the input basename if provenance attrs are missing.
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

def recursive_copy_sampled(src, dst, idx, n_frames):
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            # g = dst.create_group(key)
            g = dst.require_group(key)
            copy_attrs(item, g)
            recursive_copy_sampled(item, g, idx, n_frames)
        elif isinstance(item, h5py.Dataset):
            dset = item
            if dset.ndim >= 1 and dset.shape[0] == n_frames:
                new_shape = (len(idx),) + dset.shape[1:]
                out = create_like(dst, key, dset, new_shape)
                copy_attrs(dset, out)

                block = 2048
                wptr = 0
                for s in range(0, len(idx), block):
                    sel = idx[s:s+block]
                    data = dset[sel, ...]
                    out[wptr:wptr+len(sel), ...] = data
                    wptr += len(sel)
            else:
                out = create_like(dst, key, dset, dset.shape)
                copy_attrs(dset, out)
                if dset.ndim == 0:
                    out[()] = dset[()]
                else:
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
    ap.add_argument("--count", type=int, required=True, help="Number of frames to sample (without replacement)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--images-path", default="/entry/data/images")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = os.path.abspath(args.input_h5)
    with h5py.File(in_path, "r") as f_in:
        if args.images_path not in f_in:
            raise KeyError(f"Missing images dataset: {args.images_path}")
        n_frames = f_in[args.images_path].shape[0]
        if args.count <= 0 or args.count > n_frames:
            raise ValueError(f"--count must be in [1, {n_frames}]")

        # Figure out provenance for naming
        orig = f_in.attrs.get("_source_file", None)
        min_peaks = f_in.attrs.get("_min_peaks", 15)
        # Prefer original basename if available, else this file
        base_for_name = os.path.splitext(os.path.basename(orig if orig else in_path))[0]

        if args.out is None:
            out_name = f"{base_for_name}_min_{int(min_peaks)}peaks_{args.count}.h5"
            out_path = os.path.join(os.path.dirname(in_path), out_name)
        else:
            out_path = args.out

        # Build random sample indices (without replacement)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_frames, size=args.count, replace=False)
        idx.sort()  # keep ascending for nicer disk access

        with h5py.File(out_path, "w") as f_out:
            copy_attrs(f_in, f_out)
            # Update/augment provenance
            f_out.attrs["_source_file"] = orig if orig else in_path
            f_out.attrs["_min_peaks"] = int(min_peaks)
            f_out.attrs["_sampled_count"] = int(args.count)
            f_out.attrs["_sample_seed"] = int(args.seed)
            f_out.create_dataset("/entry/data/sample_frame_indices", data=idx.astype(np.int64), dtype=np.int64)

            recursive_copy_sampled(f_in, f_out, idx, n_frames)

    print(f"[sample_random_frames] Wrote: {out_path}")

if __name__ == "__main__":
    main()
