#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple, Optional

LAT_CODE = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "rhombohedral": "h",
    "hexagonal": "h",
    "cubic": "c",
    # NOTE: trigonal intentionally not supported yet (see compute_bravais)
}

AXIS_MAP = {
    "monoclinic": "b",
    "tetragonal": "c",
    "hexagonal": "c",
    # "trigonal": "c",
    "rhombohedral": "c",
    # orthorhombic, cubic, triclinic → no unique_axis entry
}


def parse_unit_cell(
    lines: List[str],
) -> Tuple[str, str, Optional[str]]:
    """
    Parse ONLY lattice_type, centering and unique_axis from the unit cell header block:

    ----- Begin unit cell -----
    ...
    lattice_type = tetragonal
    unique_axis  = c
    centering    = I
    ...
    ----- End unit cell -----
    """
    in_uc = False
    lattice_type = None
    centering = None
    unique_axis = None

    for line in lines:
        s = line.strip()
        if s == "----- Begin unit cell -----":
            in_uc = True
            continue
        if s == "----- End unit cell -----":
            break
        if not in_uc:
            continue

        if s.startswith("lattice_type"):
            _, rhs = s.split("=", 1)
            lattice_type = rhs.strip()
            continue
        if s.startswith("centering"):
            _, rhs = s.split("=", 1)
            centering = rhs.strip()
            continue
        if s.startswith("unique_axis"):
            _, rhs = s.split("=", 1)
            unique_axis = rhs.strip()
            continue

    if lattice_type is None or centering is None:
        raise ValueError("Could not find lattice_type and/or centering in unit cell header block")

    return lattice_type, centering, unique_axis


def compute_bravais(lattice_type: str, centering: str, unique_axis: Optional[str]) -> str:
    """
    Compute bravais code like 'tIc' from lattice_type, centering and the
    fixed AXIS_MAP. unique_axis is currently unused but parsed for future use.

    Trigonal is intentionally not supported yet.
    """
    lt = lattice_type.lower().strip()

    if lt == "trigonal":
        raise NotImplementedError("lattice_type = trigonal is not supported yet in bravais mapping")

    if lt not in LAT_CODE:
        raise ValueError(f"Unknown lattice_type '{lattice_type}' for bravais mapping")

    lat_code = LAT_CODE[lt]
    axis_code = AXIS_MAP.get(lt)

    if axis_code:
        return f"{lat_code}{centering}{axis_code}"
    else:
        return f"{lat_code}{centering}"


def parse_reciprocal_line(line: str) -> List[float]:
    """
    Parse a line like:
        astar = -0.2089414 +0.2798239 +0.5632466 nm^-1

    Returns [a1, a2, a3] as floats.
    """
    if "=" not in line:
        raise ValueError(f"Malformed reciprocal line (no '='): {line!r}")

    _, rhs = line.split("=", 1)
    rhs = rhs.strip()

    if "nm^-1" in rhs:
        rhs = rhs.split("nm^-1", 1)[0].strip()

    parts = rhs.split()
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            pass

    if len(values) != 3:
        raise ValueError(f"Expected 3 numeric components in reciprocal line, got {len(values)}: {line!r}")

    return values


def extract_geometry_block(lines: List[str]) -> Optional[str]:
    """
    Extract the geometry block between:

      ----- Begin geometry file -----
      ...
      ----- End geometry file -----

    Returns the block as a single string including newlines, or None if not found.
    """
    in_geom = False
    buf: List[str] = []

    for line in lines:
        s = line.strip()
        if s == "----- Begin geometry file -----":
            in_geom = True
            continue
        if s == "----- End geometry file -----":
            break
        if in_geom:
            buf.append(line)

    if buf:
        return "".join(buf)
    return None


def extract_cell_block(lines: List[str]) -> Optional[str]:
    """
    Extract the unit cell block between:

      ----- Begin unit cell -----
      ...
      ----- End unit cell -----

    Returns the block as a single string including newlines, or None if not found.
    """
    in_cell = False
    buf: List[str] = []

    for line in lines:
        s = line.strip()
        if s == "----- Begin unit cell -----":
            in_cell = True
            continue
        if s == "----- End unit cell -----":
            break
        if in_cell:
            buf.append(line)

    if buf:
        return "".join(buf)
    return None


def stream_to_sol(
    stream_path: str,
    sol_path: Optional[str] = None,
    geom_path: Optional[str] = None,
    cell_path: Optional[str] = None,
) -> None:
    """
    Convert a CrystFEL stream file to a .sol file.

    Also:
      - Copies the geometry header block into a .geom file.
      - Copies the unit cell header block into a .cell file.

    .sol lines:
        <h5_path> //<Event> <9 a*/b*/c* components> <det_shift_x_mm> <det_shift_y_mm> <bravais>
    """
    with open(stream_path, "r") as fh:
        lines = fh.readlines()

    # Default output paths
    if sol_path is None:
        base, _ = os.path.splitext(stream_path)
        sol_path = base + ".sol"

    if geom_path is None:
        base, _ = os.path.splitext(stream_path)
        geom_path = base + ".geom"

    if cell_path is None:
        base, _ = os.path.splitext(stream_path)
        cell_path = base + ".cell"

    # 1) Copy geometry header block to .geom
    geom_block = extract_geometry_block(lines)
    if geom_block:
        with open(geom_path, "w") as gfh:
            gfh.write(geom_block)
        print(f"Wrote geometry header block to {geom_path}")
    else:
        print("No 'Begin geometry file'/'End geometry file' block found; not writing .geom")

    # 2) Copy unit cell header block to .cell
    cell_block = extract_cell_block(lines)
    if cell_block:
        with open(cell_path, "w") as cfh:
            cfh.write(cell_block)
        print(f"Wrote unit cell header block to {cell_path}")
    else:
        print("No 'Begin unit cell'/'End unit cell' block found; not writing .cell")

    # 3) Parse unit cell (for bravais code only)
    lattice_type, centering, unique_axis = parse_unit_cell(lines)
    bravais = compute_bravais(lattice_type, centering, unique_axis)

    # 4) Build .sol from chunks
    with open(sol_path, "w") as out:
        in_chunk = False
        in_crystal = False

        current_image: Optional[str] = None
        current_event: Optional[int] = None
        det_x: Optional[float] = None
        det_y: Optional[float] = None

        astar: Optional[List[float]] = None
        bstar: Optional[List[float]] = None
        cstar: Optional[List[float]] = None

        def flush_crystal() -> None:
            """
            Write one line for the current crystal, if everything needed is present.
            """
            nonlocal astar, bstar, cstar
            if (
                current_image is None
                or current_event is None
                or det_x is None
                or det_y is None
                or astar is None
                or bstar is None
                or cstar is None
            ):
                return

            components = astar + bstar + cstar
            line_matrix = " ".join(f"{v:+.7f}" for v in components)

            out.write(
                f"{current_image} //{current_event} "
                f"{line_matrix} "
                f"{det_x:.6f} {det_y:.6f} {bravais}\n"
            )

        for line in lines:
            s = line.strip()

            # Chunk boundaries
            if s == "----- Begin chunk -----":
                in_chunk = True
                in_crystal = False

                # Reset chunk-level info
                current_image = None
                current_event = None
                det_x = None
                det_y = None

                astar = bstar = cstar = None
                continue

            if s == "----- End chunk -----":
                in_chunk = False
                in_crystal = False
                astar = bstar = cstar = None
                continue

            if not in_chunk:
                continue

            # Inside a chunk
            if s.startswith("Image filename:"):
                _, rhs = s.split(":", 1)
                current_image = rhs.strip()
                continue

            if s.startswith("Event:"):
                _, rhs = s.split(":", 1)
                rhs = rhs.strip()
                if rhs.startswith("//"):
                    rhs = rhs[2:]
                try:
                    current_event = int(rhs)
                except ValueError:
                    current_event = None
                continue

            if s.startswith("header/float//entry/data/det_shift_x_mm"):
                _, rhs = s.split("=", 1)
                det_x = float(rhs.strip())
                continue

            if s.startswith("header/float//entry/data/det_shift_y_mm"):
                _, rhs = s.split("=", 1)
                det_y = float(rhs.strip())
                continue

            # Crystal boundaries inside chunk
            if s == "--- Begin crystal":
                in_crystal = True
                astar = bstar = cstar = None
                continue

            if s == "--- End crystal":
                flush_crystal()
                in_crystal = False
                astar = bstar = cstar = None
                continue

            if not in_crystal:
                continue

            # Inside crystal block
            if s.startswith("astar"):
                astar = parse_reciprocal_line(s)
                continue
            if s.startswith("bstar"):
                bstar = parse_reciprocal_line(s)
                continue
            if s.startswith("cstar"):
                cstar = parse_reciprocal_line(s)
                continue

    print(f"Wrote .sol to {sol_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CrystFEL stream file into a .sol file and extract header .geom and .cell."
    )
    parser.add_argument("stream", help="Input CrystFEL .stream file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .sol file (default: <stream_basename>.sol)",
    )
    parser.add_argument(
        "--geom-out",
        help="Output .geom file (default: <stream_basename>.geom)",
    )
    parser.add_argument(
        "--cell-out",
        help="Output .cell file (default: <stream_basename>.cell)",
    )

    args = parser.parse_args()
    stream_path: str = args.stream
    sol_path: Optional[str] = args.output
    geom_path: Optional[str] = args.geom_out
    cell_path: Optional[str] = args.cell_out

    stream_to_sol(stream_path, sol_path, geom_path, cell_path)


if __name__ == "__main__":
    main()
