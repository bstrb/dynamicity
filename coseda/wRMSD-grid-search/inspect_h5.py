#!/usr/bin/env python3
import argparse, os, re, sys, textwrap
from collections import defaultdict
import numpy as np

try:
    import h5py
except ImportError:
    print("This script requires 'h5py'. Install with: pip install h5py", file=sys.stderr)
    sys.exit(1)

DET_SHIFT_CANDIDATES = {
    "x": ["det_shift_x_mm", "detector_shift_x_mm", "shift_x_mm", "det_shift_x"],
    "y": ["det_shift_y_mm", "detector_shift_y_mm", "shift_y_mm", "det_shift_y"],
}

LST_LINE_RE = re.compile(r"^(?P<path>.+?)\s*//\s*(?P<index>\d+)\s*$")

def human_dtype(dt):
    try:
        return str(dt)
    except Exception:
        return repr(dt)

def walk_h5(obj, prefix=""):
    if isinstance(obj, h5py.Dataset):
        ds = obj
        info = {
            "path": prefix,
            "dtype": human_dtype(ds.dtype),
            "shape": ds.shape,
            "chunks": getattr(ds, "chunks", None),
            "compression": getattr(ds, "compression", None),
            "compression_opts": getattr(ds, "compression_opts", None),
        }
        print(f"▸ DATASET {info['path']}: shape={info['shape']} dtype={info['dtype']}"
              f" chunks={info['chunks']} comp={info['compression']}:{info['compression_opts']}")
        # Print a few attributes
        n_attrs = min(6, len(ds.attrs))
        if n_attrs:
            print("    attrs:")
            for k in list(ds.attrs.keys())[:n_attrs]:
                v = ds.attrs[k]
                # Keep attribute printouts compact
                sv = v if isinstance(v, (str, bytes)) else np.array(v).tolist()
                if isinstance(sv, list) and len(sv) > 8:
                    sv = sv[:8] + ["..."]
                print(f"      - {k}: {sv}")
    elif isinstance(obj, h5py.Group):
        grp = obj
        if prefix != "/":
            print(f"● GROUP   {prefix}")
        keys = sorted(grp.keys())
        for k in keys:
            child = grp[k]
            path = prefix.rstrip("/") + "/" + k if prefix != "/" else "/" + k
            walk_h5(child, path)

def find_image_stack_candidates(f):
    """
    Heuristics: datasets with ndim>=3 or (ndim==2 and first dim>1),
    where the first axis likely enumerates images.
    """
    cands = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            if not shape:
                return
            if (len(shape) >= 3 and shape[0] > 1) or (len(shape) == 2 and shape[0] > 1):
                # Favor larger total size, prefer names containing 'image' or 'data'
                weight = np.prod(shape)
                name_score = 1
                lname = name.lower()
                if "image" in lname or "data" in lname:
                    name_score += 1
                cands.append((name_score, weight, name, shape))
    f.visititems(visit)
    cands.sort(reverse=True)  # by name_score then weight
    return cands

def locate_shift_arrays(f):
    """Return possible x/y shift datasets and shapes."""
    hits = {"x": [], "y": []}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1].lower()
            for axis, patterns in DET_SHIFT_CANDIDATES.items():
                if any(base == p.lower() for p in patterns):
                    hits[axis].append((name, obj.shape, human_dtype(obj.dtype)))
    f.visititems(visit)
    return hits

def parse_lst(lst_path):
    entries = []
    unknown = []
    with open(lst_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            m = LST_LINE_RE.match(s)
            if not m:
                unknown.append(s)
                continue
            path = m.group("path")
            idx = int(m.group("index"))
            entries.append((path, idx))
    return entries, unknown

def validate_lst(entries):
    """Group by HDF5 path."""
    grouped = defaultdict(list)
    for p, i in entries:
        grouped[os.path.abspath(p)].append(i)
    return grouped

def main():
    ap = argparse.ArgumentParser(
        description="Inspect HDF5 structure, detect image stack & det_shift arrays, and (optionally) validate a .lst file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("h5", nargs="?", help="Path to an HDF5 file to inspect.")
    ap.add_argument("--lst", help="Optional .lst file to validate (format: '/path/to/file.h5 //image_index').")
    ap.add_argument("--stats", action="store_true", help="Compute basic stats for det_shift arrays (min/max/mean).")
    args = ap.parse_args()

    if not args.h5 and not args.lst:
        ap.print_help(sys.stderr)
        sys.exit(2)

    if args.h5:
        h5_path = os.path.abspath(args.h5)
        if not os.path.exists(h5_path):
            print(f"[ERROR] HDF5 not found: {h5_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== FILE: {h5_path} ===")
        with h5py.File(h5_path, "r") as f:
            # Walk structure
            print("\n--- Structure ---")
            walk_h5(f["/"], "/")

            # Image stack candidates
            print("\n--- Image stack candidates (likely image axis = 0) ---")
            cands = find_image_stack_candidates(f)
            if not cands:
                print("No obvious image stack datasets found.")
            else:
                for rank, (name_score, weight, name, shape) in enumerate(cands[:8], 1):
                    print(f"{rank}. {name}: shape={shape}")

            # Shift arrays
            print("\n--- Detector shift datasets (candidates) ---")
            hits = locate_shift_arrays(f)
            if not hits["x"] and not hits["y"]:
                print("No obvious det_shift_x_mm / det_shift_y_mm datasets found.")
            else:
                if hits["x"]:
                    for name, shape, dt in hits["x"]:
                        print(f"X: {name}  shape={shape}  dtype={dt}")
                if hits["y"]:
                    for name, shape, dt in hits["y"]:
                        print(f"Y: {name}  shape={shape}  dtype={dt}")

            # Optional stats
            if args.stats:
                print("\n--- Shift array stats ---")
                for axis in ("x", "y"):
                    for name, shape, dt in hits[axis]:
                        try:
                            d = f[name][...]
                            if d.size > 5_000_000:
                                print(f"{axis.upper()}: {name}: array too large for stats (size={d.size})")
                                continue
                            finite = np.isfinite(d)
                            if not finite.any():
                                print(f"{axis.upper()}: {name}: no finite values")
                                continue
                            vals = d[finite]
                            print(f"{axis.upper()}: {name}: min={np.min(vals):.6g}  max={np.max(vals):.6g}  mean={np.mean(vals):.6g}")
                        except Exception as e:
                            print(f"{axis.upper()}: {name}: error reading stats: {e}")

    # .lst validation (can run with or without --h5)
    if args.lst:
        lst_path = os.path.abspath(args.lst)
        if not os.path.exists(lst_path):
            print(f"\n[ERROR] .lst not found: {lst_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\n=== LST: {lst_path} ===")
        entries, unknown = parse_lst(lst_path)
        if unknown:
            print("Lines that did NOT match 'path //index' format:")
            for s in unknown[:10]:
                print("  ", s)
            if len(unknown) > 10:
                print(f"  ... (+{len(unknown)-10} more)")
        if not entries:
            print("No valid entries parsed from the .lst.")
            sys.exit(0)

        grouped = validate_lst(entries)
        print(f"\n--- Summary ---")
        for p, idxs in grouped.items():
            print(f"{p}: {len(idxs)} entries")

        # Validate indices against the likely image stack length
        print("\n--- Validating indices against HDF5 image stacks ---")
        for p, idxs in grouped.items():
            if not os.path.exists(p):
                print(f"[ERROR] Missing HDF5: {p}")
                continue
            try:
                with h5py.File(p, "r") as f:
                    cands = find_image_stack_candidates(f)
                    if not cands:
                        print(f"{p}: no image stack candidates; cannot validate indices.")
                        continue
                    # Pick top candidate
                    _, _, ds_name, shape = cands[0]
                    n_images = shape[0]
                    bad = [i for i in idxs if i < 0 or i >= n_images]
                    if bad:
                        print(f"{p}: BAD indices (out of range 0..{n_images-1}): {sorted(set(bad))[:15]}")
                    else:
                        print(f"{p}: OK (all indices within 0..{n_images-1}) using dataset {ds_name}")
            except Exception as e:
                print(f"{p}: error inspecting file: {e}")

if __name__ == "__main__":
    main()
