#!/usr/bin/env python3
import argparse
import h5py
import numpy as np   # <-- NEW

IMAGES_PATH = "/entry/data/images"


def copy_attrs(src_obj, dst_obj):
    """Copy all attributes from src_obj to dst_obj."""
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def create_placeholder_dataset(src_dset, dst_group, name, compression="gzip", fillvalue=0):
    """
    Create a placeholder dataset with same shape/dtype as src_dset
    but without copying the original data. Uses compression so the
    file stays small, and fills with zeros so HDF5/CrystFEL see a
    real, initialized dataset.
    """
    print(f"  -> Creating placeholder for dataset '{src_dset.name}'")

    kwargs = {
        "shape": src_dset.shape,
        "dtype": src_dset.dtype,
        "compression": compression,
        # we won't rely on fillvalue alone anymore
    }

    # Preserve chunking if present
    if src_dset.chunks is not None:
        kwargs["chunks"] = src_dset.chunks

    # Preserve maxshape if present (e.g. extensible datasets)
    if src_dset.maxshape is not None:
        kwargs["maxshape"] = src_dset.maxshape

    # Preserve common filters if present
    if src_dset.scaleoffset is not None:
        kwargs["scaleoffset"] = src_dset.scaleoffset
    if src_dset.shuffle:
        kwargs["shuffle"] = True
    if src_dset.fletcher32:
        kwargs["fletcher32"] = True

    dst_dset = dst_group.create_dataset(name, **kwargs)

    # IMPORTANT: actually fill with zeros so CrystFEL sees a real dataset
    # Zeros compress extremely well with gzip, so file size will still drop a lot.
    print("    Filling placeholder with zeros (this may take a moment)...")
    dst_dset[...] = np.zeros(src_dset.shape, dtype=src_dset.dtype)

    copy_attrs(src_dset, dst_dset)


def copy_group(src_group, dst_group, base_path="/", compression="gzip", fillvalue=0):
    """
    Recursively copy src_group into dst_group.

    Everything is copied verbatim except the dataset at IMAGES_PATH,
    which is replaced by a placeholder dataset.
    """
    for name, item in src_group.items():
        # Absolute path for this object
        if base_path == "/" or base_path == "":
            full_path = "/" + name
        else:
            full_path = base_path.rstrip("/") + "/" + name

        if isinstance(item, h5py.Group):
            print(f"Entering group: {full_path}")
            new_grp = dst_group.create_group(name)
            copy_attrs(item, new_grp)
            copy_group(
                item,
                new_grp,
                base_path=full_path,
                compression=compression,
                fillvalue=fillvalue,
            )

        elif isinstance(item, h5py.Dataset):
            if full_path == IMAGES_PATH:
                create_placeholder_dataset(
                    item,
                    dst_group,
                    name,
                    compression=compression,
                    fillvalue=fillvalue,
                )
            else:
                print(f"Copying dataset with data: {full_path}")
                src_group.file.copy(item, dst_group, name=name)

        else:
            print(f"Skipping unsupported object type at {full_path} ({type(item)})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy an HDF5 file while replacing /entry/data/images "
            "with a compressed placeholder dataset."
        )
    )
    parser.add_argument("input", help="Input HDF5 file (full data)")
    parser.add_argument("output", help="Output HDF5 file (images replaced by placeholders)")
    parser.add_argument(
        "--compression",
        default="gzip",
        help="Compression for placeholder dataset (default: gzip)",
    )
    parser.add_argument(
        "--fillvalue",
        type=float,
        default=0.0,
        help="(Unused now) Fill value for placeholder dataset (default: 0.0)",
    )

    args = parser.parse_args()

    print("Input file: ", args.input)
    print("Output file:", args.output)
    print("Images dataset path:", IMAGES_PATH)
    print("Compression:", args.compression)
    print()

    with h5py.File(args.input, "r") as f_in, h5py.File(args.output, "w") as f_out:
        # Copy root attributes
        copy_attrs(f_in, f_out)

        # Recursively copy everything
        copy_group(
            f_in,
            f_out,
            base_path="/",
            compression=args.compression,
            fillvalue=args.fillvalue,
        )

    print("\nDone.")
    print(f"File written to: {args.output}")
    print(f"Dataset {IMAGES_PATH} was replaced by a placeholder;")
    print("all other datasets (center_x, det_shift_x_mm, peaks, etc.) were copied with data.")


if __name__ == "__main__":
    main()


# python make_light_h5.py /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038_min_15peaks.h5 /home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038_min_15peaks_light.h5
