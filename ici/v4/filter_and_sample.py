#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter + sample in ONE pass.

Keeps frames with /entry/data/nPeaks >= MIN_PEAKS, then randomly samples N (without replacement)
from those kept, and writes an output HDF5 that preserves per-frame alignment and dataset
compression/chunking/attributes.

Usage:
  python3 filter_and_sample.py input.h5 --count 5000 \
      [--min-peaks 15] [--seed 1337] \
      [--images-path /entry/data/images] [--npeaks-path /entry/data/nPeaks] \
      [--out /path/to/output.h5]

If --out is omitted, the filename is:
  <original_basename>_min_<MIN>peaks_<COUNT>.h5
"""

import argparse, os, sys
import numpy as np
import h5py

# ---------- helpers ----------

def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v

def make_chunks_like(src_chunks, new_len_axis0):
    if src_chunks is None:
        return None
    ch = list(src_chunks)
    if len(ch) >= 1:
        # keep same chunk size along axis 0 but cap to new length (must be >=1)
        ch[0] = max(1, min(ch[0], new_len_axis0 if new_len_axis0 is not None else ch[0]))
    return tuple(ch)

def create_like(dst_group, name, src_dset, new_shape):
    # Some properties might be None or not supported for all dtypes; be defensive
    kwargs = dict(
        shape=new_shape,
        dtype=src_dset.dtype,
        compression=src_dset.compression,
        compression_opts=src_dset.compression_opts,
        shuffle=src_dset.shuffle,
        fletcher32=src_dset.fletcher32,
        chunks=make_chunks_like(src_dset.chunks, new_shape[0] if len(new_shape) else None),
        fillvalue=src_dset.fillvalue,
    )
    # scaleoffset may be None or int; only pass if present (older h5py can be picky)
    try:
        so = src_dset.scaleoffset
        if so is not None:
            kwargs["scaleoffset"] = so
    except Exception:
        pass

    # Respect maxshape if present; clamp axis-0 to new length when not None
    try:
        if src_dset.maxshape is not None:
            maxshape = list(src_dset.maxshape)
            if len(maxshape) >= 1 and maxshape[0] is not None:
                maxshape[0] = new_shape[0]
            kwargs["maxshape"] = tuple(maxshape)
    except Exception:
        pass

    return dst_group.create_dataset(name, **kwargs)

def coalesce_indices(sorted_idx):
    """[(start, end)] half-open ranges covering sorted indices (ascending)."""
    if len(sorted_idx) == 0:
        return []
    ranges = []
    s = sorted_idx[0]
    prev = s
    for x in sorted_idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((s, prev + 1))
            s = x
            prev = x
    ranges.append((s, prev + 1))
    return ranges

def is_per_frame_dataset(dset, n_frames):
    return (dset.ndim >= 1) and (dset.shape[0] == n_frames)

def recursive_copy_sampled(src_group, dst_group, n_frames, sampled_idx_sorted):
    """Copy group tree. For datasets with axis-0 == n_frames, copy only sampled indices (range-coalesced)."""
    for key in src_group:
        item = src_group[key]
        if isinstance(item, h5py.Group):
            g = dst_group.require_group(key)
            copy_attrs(item, g)
            recursive_copy_sampled(item, g, n_frames, sampled_idx_sorted)
        elif isinstance(item, h5py.Dataset):
            dset = item
            if is_per_frame_dataset(dset, n_frames):
                new_shape = (len(sampled_idx_sorted),) + dset.shape[1:]
                out = create_like(dst_group, key, dset, new_shape)
                copy_attrs(dset, out)

                # Copy in contiguous ranges for efficiency
                ranges = coalesce_indices(sampled_idx_sorted)
                wptr = 0
                for (rs, re) in ranges:
                    block_len = re - rs
                    chunk = dset[rs:re, ...]   # read contiguous block
                    out[wptr:wptr + block_len, ...] = chunk
                    wptr += block_len
            else:
                # Copy whole dataset verbatim (stream in chunks along axis 0 where possible)
                out = create_like(dst_group, key, dset, dset.shape)
                copy_attrs(dset, out)
                if dset.ndim == 0:
                    out[()] = dset[()]
                else:
                    # Chunked bulk copy along axis 0
                    total0 = dset.shape[0]
                    step = dset.chunks[0] if (dset.chunks and dset.ndim >= 1) else 2048
                    pos = 0
                    while pos < total0:
                        end = min(total0, pos + step)
                        out[pos:end, ...] = dset[pos:end, ...]
                        pos = end

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Filter by nPeaks then sample N frames in one pass.")
    ap.add_argument("input_h5")
    ap.add_argument("--count", type=int, required=True, help="Number of frames to sample (without replacement)")
    ap.add_argument("--min-peaks", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--images-path", default="/entry/data/images")
    ap.add_argument("--npeaks-path", default="/entry/data/nPeaks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = os.path.abspath(args.input_h5)
    if not os.path.isfile(in_path):
        print(f"ERROR: not found: {in_path}", file=sys.stderr)
        return 2

    with h5py.File(in_path, "r") as f_in:
        if args.images_path not in f_in:
            raise KeyError(f"Missing images dataset: {args.images_path}")
        if args.npeaks_path not in f_in:
            raise KeyError(f"Missing nPeaks dataset: {args.npeaks_path}")

        n_frames = f_in[args.images_path].shape[0]
        npeaks = f_in[args.npeaks_path][...]
        if npeaks.shape[0] != n_frames:
            raise ValueError("nPeaks length does not match images first dimension")

        keep_mask = (npeaks >= args.min_peaks)
        kept_idx = np.nonzero(keep_mask)[0]
        kept = kept_idx.size
        if kept == 0:
            raise RuntimeError(f"No frames satisfy nPeaks >= {args.min_peaks}")
        if args.count <= 0 or args.count > kept:
            raise ValueError(f"--count must be in [1, {kept}] (kept after filtering)")

        # Sample WITHOUT replacement from kept frames
        rng = np.random.default_rng(args.seed)
        sampled = rng.choice(kept_idx, size=args.count, replace=False)
        sampled.sort()  # ascending for better I/O

        # Output name
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = args.out or os.path.join(
            os.path.dirname(in_path),
            f"{base}_min_{int(args.min_peaks)}peaks_{int(args.count)}.h5"
        )

        with h5py.File(out_path, "w") as f_out:
            # File-level attrs & provenance
            copy_attrs(f_in, f_out)
            f_out.attrs["_source_file"] = in_path
            f_out.attrs["_min_peaks"] = int(args.min_peaks)
            f_out.attrs["_kept_frames_after_filter"] = int(kept)
            f_out.attrs["_sampled_count"] = int(args.count)
            f_out.attrs["_sample_seed"] = int(args.seed)

            # Store indices used (to allow tracing back)
            # write these first (creates /entry and /entry/data ahead of recursion)
            f_out.require_group("/entry/data")
            f_out["/entry/data/kept_frame_indices"] = kept_idx.astype(np.int64)
            f_out["/entry/data/sample_frame_indices"] = sampled.astype(np.int64)

            # Mirror hierarchy and copy data, sampling per-frame datasets
            recursive_copy_sampled(f_in, f_out, n_frames, sampled)

    print(f"[filter_and_sample] Filtered kept={kept}, sampled={args.count}")
    print(f"[filter_and_sample] Wrote: {out_path}")

if __name__ == "__main__":
    sys.exit(main())
