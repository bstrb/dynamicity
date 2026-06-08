#!/usr/bin/env python3

import argparse
import math
import re
from typing import List, Optional, Tuple

"""
filters reflections in a CrystFEL .stream file by absolute d-spacing, using the astar/bstar/cstar vectors in each crystal block to compute d for each reflection. Writes a new .stream file with only the reflections in the specified resolution range, and optionally drops crystals that have no reflections after filtering.
python3 res_cut_stream.py \
  --input "path/to/input.stream" \
  --output "path/to/output.stream" \
  --lowres lowest_trustworthy_resolution \
  --highres highest_trustworthy_resolution \
  --drop-empty-crystals

"""
# python3 res_cut_stream.py \
#   --input "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII.stream" \
#   --output "/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300-VIII_cut_20-0_3.stream" \
#   --lowres 20 \
#   --highres 0.3 \
#   --drop-empty-crystals

FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
VEC_RE = re.compile(
    rf"^\s*(astar|bstar|cstar)\s*=\s*({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})"
)


def parse_vec(line: str) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    m = VEC_RE.match(line)
    if not m:
        return None
    name = m.group(1)
    vec = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return name, vec


def d_spacing_angstrom(
    h: int,
    k: int,
    l: int,
    astar: Tuple[float, float, float],
    bstar: Tuple[float, float, float],
    cstar: Tuple[float, float, float],
) -> Tuple[float, float]:
    gx = h * astar[0] + k * bstar[0] + l * cstar[0]
    gy = h * astar[1] + k * bstar[1] + l * cstar[1]
    gz = h * astar[2] + k * bstar[2] + l * cstar[2]

    g_nm_inv = math.sqrt(gx * gx + gy * gy + gz * gz)

    if g_nm_inv == 0:
        return math.inf, g_nm_inv

    d_angstrom = 10.0 / g_nm_inv
    return d_angstrom, g_nm_inv


def process_crystal_block(
    block: List[str],
    lowres: float,
    highres: float,
    update_resolution_limit: bool = True,
) -> Tuple[List[str], int, int]:
    astar = bstar = cstar = None

    for line in block:
        parsed = parse_vec(line)
        if parsed is None:
            continue

        name, vec = parsed
        if name == "astar":
            astar = vec
        elif name == "bstar":
            bstar = vec
        elif name == "cstar":
            cstar = vec

    if astar is None or bstar is None or cstar is None:
        return block, 0, 0

    refl_start = None
    refl_end = None

    for i, line in enumerate(block):
        if line.strip() == "Reflections measured after indexing":
            refl_start = i
            break

    if refl_start is None:
        return block, 0, 0

    for i in range(refl_start + 1, len(block)):
        if line_is_end_of_reflections(block[i]):
            refl_end = i
            break

    if refl_end is None:
        return block, 0, 0

    reflection_indices = set()
    keep_indices = set()
    kept_g_values = []

    for i in range(refl_start + 1, refl_end):
        line = block[i]
        parts = line.split()

        if len(parts) < 3:
            continue

        try:
            h = int(parts[0])
            k = int(parts[1])
            l = int(parts[2])
        except ValueError:
            continue

        reflection_indices.add(i)

        d_angstrom, g_nm_inv = d_spacing_angstrom(h, k, l, astar, bstar, cstar)

        if highres <= d_angstrom <= lowres:
            keep_indices.add(i)
            kept_g_values.append(g_nm_inv)

    old_count = len(reflection_indices)
    new_count = len(keep_indices)

    new_block = []
    max_g = max(kept_g_values) if kept_g_values else None

    for i, line in enumerate(block):
        if i in reflection_indices and i not in keep_indices:
            continue

        stripped = line.strip()

        if stripped.startswith("num_reflections ="):
            new_block.append(f"num_reflections = {new_count}\n")
            continue

        if update_resolution_limit and stripped.startswith("diffraction_resolution_limit ="):
            if max_g is not None and max_g > 0:
                new_d = 10.0 / max_g
                new_block.append(
                    f"diffraction_resolution_limit = {max_g:.2f} nm^-1 or {new_d:.2f} A\n"
                )
            else:
                new_block.append("diffraction_resolution_limit = 0.00 nm^-1 or inf A\n")
            continue

        new_block.append(line)

    return new_block, old_count, new_count


def line_is_end_of_reflections(line: str) -> bool:
    return line.strip() == "End of reflections"


def filter_stream(
    input_path: str,
    output_path: str,
    lowres: float,
    highres: float,
    drop_empty_crystals: bool,
    update_resolution_limit: bool,
) -> None:
    total_crystals = 0
    kept_crystals = 0
    total_reflections = 0
    kept_reflections = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as inp, \
         open(output_path, "w", encoding="utf-8") as out:

        inside_crystal = False
        crystal_block = []

        for line in inp:
            if line.strip() == "--- Begin crystal":
                inside_crystal = True
                crystal_block = [line]
                continue

            if inside_crystal:
                crystal_block.append(line)

                if line.strip() == "--- End crystal":
                    total_crystals += 1

                    new_block, old_count, new_count = process_crystal_block(
                        crystal_block,
                        lowres=lowres,
                        highres=highres,
                        update_resolution_limit=update_resolution_limit,
                    )

                    total_reflections += old_count
                    kept_reflections += new_count

                    if new_count > 0 or not drop_empty_crystals:
                        out.writelines(new_block)
                        kept_crystals += 1

                    inside_crystal = False
                    crystal_block = []

                continue

            out.write(line)

        if inside_crystal:
            raise RuntimeError("Input stream ended inside an unfinished crystal block.")

    print(f"Input stream:        {input_path}")
    print(f"Output stream:       {output_path}")
    print(f"Resolution window:   {lowres:.3f} Å to {highres:.3f} Å")
    print(f"Crystals kept:       {kept_crystals} / {total_crystals}")
    print(f"Reflections kept:    {kept_reflections} / {total_reflections}")
    print(f"Reflections removed: {total_reflections - kept_reflections}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter CrystFEL stream reflection tables by absolute d-spacing."
    )
    parser.add_argument("--input", required=True, help="Input CrystFEL .stream file")
    parser.add_argument("--output", required=True, help="Output filtered .stream file")
    parser.add_argument("--lowres", type=float, required=True, help="Largest d-spacing to keep, in Å")
    parser.add_argument("--highres", type=float, required=True, help="Smallest d-spacing to keep, in Å")
    parser.add_argument(
        "--drop-empty-crystals",
        action="store_true",
        help="Remove crystal blocks where no reflections remain after filtering",
    )
    parser.add_argument(
        "--keep-original-resolution-limit",
        action="store_true",
        help="Do not update diffraction_resolution_limit after filtering",
    )

    args = parser.parse_args()

    if args.highres <= 0:
        raise ValueError("--highres must be positive")

    if args.lowres <= args.highres:
        raise ValueError("--lowres must be larger than --highres, e.g. --lowres 20 --highres 0.4")

    filter_stream(
        input_path=args.input,
        output_path=args.output,
        lowres=args.lowres,
        highres=args.highres,
        drop_empty_crystals=args.drop_empty_crystals,
        update_resolution_limit=not args.keep_original_resolution_limit,
    )


if __name__ == "__main__":
    main()