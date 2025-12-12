#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, Set, List

import h5py


ATTR_NAME = "original_file_path"


def extract_h5_paths_from_sol(sol_path: Path) -> List[str]:
    """
    Parse a .sol file and return the list of HDF5 paths found as the first
    token on each non-empty, non-comment line.

    Example .sol line:
      /path/to/file.h5 //0 +0.0000000 +0.6613101 ... tIc
    """
    paths: List[str] = []

    with sol_path.open("r") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            # First token up to whitespace is the HDF5 path
            first = s.split()[0]
            if first.lower().endswith(".h5"):
                paths.append(first)

    return paths


def try_get_original_from_attr(noimg_path: str) -> str:
    """
    Open no_image_file.h5 and try to read ATTR_NAME from the root.

    Returns the original file path if found and existing, otherwise ''.
    """
    if not os.path.exists(noimg_path):
        return ""

    try:
        with h5py.File(noimg_path, "r") as f:
            if ATTR_NAME in f.attrs:
                orig = f.attrs[ATTR_NAME]
                if isinstance(orig, bytes):
                    orig = orig.decode("utf-8", errors="ignore")
                orig = str(orig)
            else:
                return ""
    except Exception:
        return ""

    if os.path.exists(orig):
        return orig
    else:
        print(f"  Attribute {ATTR_NAME!r} found in {noimg_path!r} but file does not exist: {orig!r}")
        return ""


def prompt_for_original(noimg_path: str) -> str:
    """
    Prompt the user to provide the path to the original full-image file.h5
    corresponding to the given no_image_file.h5 path.

    Keeps asking until a valid existing file is provided or the user enters
    an empty line to skip.
    """
    print(f"\nNo original full-image file found automatically for:")
    print(f"  no-image file: {noimg_path}")
    while True:
        user = input("Enter path to the original full image file.h5 (empty to skip): ").strip()
        if user == "":
            print("  -> Skipping this entry.")
            return ""
        if os.path.exists(user):
            return os.path.abspath(user)
        else:
            print(f"  Path does not exist: {user!r}. Please try again.")


def build_mapping(noimg_paths: Set[str]) -> Dict[str, str]:
    """
    For each no_image_file.h5 path, try to find the original full-image
    file.h5 using the stored attribute. If not found, prompt the user.

    Returns a mapping {noimg_path -> original_path}. Entries which could not be
    resolved are omitted.
    """
    mapping: Dict[str, str] = {}

    for noimg in sorted(noimg_paths):
        print(f"Processing {noimg!r} ...")

        # 1) Try attribute on the no-image file
        orig = try_get_original_from_attr(noimg)

        # 2) If still unknown, ask the user
        if not orig:
            orig = prompt_for_original(noimg)

        if orig:
            mapping[noimg] = orig
            print(f"  -> Using original full-image file: {orig}")
        else:
            print(f"  WARNING: No original file specified for {noimg!r}; "
                  f"this path will NOT be replaced in the .sol")

    return mapping


def rewrite_sol(sol_path: Path, mapping: Dict[str, str]) -> None:
    """
    Rewrite sol_path in place, replacing each key in mapping with its value.
    Writes a backup file with '.bak' appended to the original suffix.
    """
    text = sol_path.read_text()

    backup = sol_path.with_suffix(sol_path.suffix + ".bak")
    backup.write_text(text)

    # Replace longer paths first to avoid substring collisions
    for old, new in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(old, new)

    sol_path.write_text(text)
    print(f"\nRewrote {sol_path} (backup: {backup})")


def write_lst(lst_path: Path, mapping: Dict[str, str]) -> None:
    """
    Write a .lst file containing one line per unique original full-image HDF5 file.
    """
    originals = sorted(set(mapping.values()))
    lst_path.parent.mkdir(parents=True, exist_ok=True)
    with lst_path.open("w") as fh:
        for p in originals:
            fh.write(p + "\n")

    print(f"Wrote {len(originals)} entries to {lst_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Relink a .sol file from no_image_file.h5 paths to their original "
            "full-image file.h5 paths, using HDF5 attributes and user prompts. "
            "Also writes a corresponding .lst."
        )
    )
    parser.add_argument(
        "sol",
        help="Input .sol file (will be modified in place; a .bak backup is created)",
    )
    parser.add_argument(
        "--lst-out",
        help="Output .lst file (default: <sol_basename>.lst)",
    )

    args = parser.parse_args()
    sol_path = Path(args.sol).resolve()

    if not sol_path.exists():
        raise SystemExit(f".sol file does not exist: {sol_path}")

    lst_path: Path
    if args.lst_out is not None:
        lst_path = Path(args.lst_out).resolve()
    else:
        base, _ = os.path.splitext(str(sol_path))
        lst_path = Path(base + ".lst")

    # 1) Collect all HDF5 paths from the .sol
    h5_paths = extract_h5_paths_from_sol(sol_path)
    if not h5_paths:
        print("No .h5 paths found in .sol; nothing to do.")
        return

    unique_noimg_paths: Set[str] = set(h5_paths)
    print(f"Found {len(unique_noimg_paths)} unique .h5 paths in {sol_path}")

    # 2) Build mapping noimg -> original
    mapping = build_mapping(unique_noimg_paths)
    if not mapping:
        print("No mappings resolved; not modifying .sol or writing .lst.")
        return

    # 3) Rewrite .sol in place (with backup)
    rewrite_sol(sol_path, mapping)

    # 4) Write .lst listing the original full-image HDF5 files
    write_lst(lst_path, mapping)


if __name__ == "__main__":
    main()
