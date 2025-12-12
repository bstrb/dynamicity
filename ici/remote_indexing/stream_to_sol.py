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


def _parse_all_floats(s: str) -> List[float]:
    values: List[float] = []
    for tok in s.replace("=", " ").split():
        try:
            values.append(float(tok))
        except ValueError:
            continue
    return values


def _parse_first_float(s: str) -> Optional[float]:
    vals = _parse_all_floats(s)
    return vals[0] if vals else None


def parse_unit_cell(lines: List[str]) -> Tuple[str, str, Optional[str], Optional[Tuple[float, float, float, float, float, float]]]:
    """
    Parse the unit cell header block:

    ----- Begin unit cell -----
    lattice_type = tetragonal
    unique_axis = c
    centering = I
    a = ...
    b = ...
    c = ...
    alpha = ...
    beta = ...
    gamma = ...
    ...
    ----- End unit cell -----

    Returns:
      lattice_type, centering, unique_axis, (a,b,c,alpha,beta,gamma) or None if
      the metric parameters can't be found.
    """
    in_uc = False
    lattice_type = None
    centering = None
    unique_axis = None

    a = b = c = alpha = beta = gamma = None

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
            # lattice_type = tetragonal
            _, rhs = s.split("=", 1)
            lattice_type = rhs.strip()
        elif s.startswith("centering"):
            # centering = I
            _, rhs = s.split("=", 1)
            centering = rhs.strip()
        elif s.startswith("unique_axis"):
            # unique_axis = c
            _, rhs = s.split("=", 1)
            unique_axis = rhs.strip()
        else:
            sl = s.lower()

            # Try "parameters = a b c alpha beta gamma" style
            if "parameters" in sl:
                vals = _parse_all_floats(s)
                if len(vals) >= 6:
                    a, b, c, alpha, beta, gamma = vals[:6]
                    continue

            # Try separate lines like "a = 88.0", etc.
            if sl.startswith("a"):
                maybe = _parse_first_float(s)
                if maybe is not None:
                    a = maybe
                    continue
            if sl.startswith("b"):
                maybe = _parse_first_float(s)
                if maybe is not None:
                    b = maybe
                    continue
            if sl.startswith("c"):
                maybe = _parse_first_float(s)
                if maybe is not None:
                    c = maybe
                    continue
            if sl.startswith("alpha"):
                maybe = _parse_first_float(s)
                if maybe is not None:
                    alpha = maybe
                    continue
            if sl.startswith("beta"):
                maybe = _parse_first_float(s)
                if maybe is not None:
                    beta = maybe
                    continue
            if sl.startswith("gamma"):
                maybe = _parse_first_float(s)
                if maybe is not None:
                    gamma = maybe
                    continue

    if lattice_type is None or centering is None:
        raise ValueError("Could not find lattice_type and/or centering in unit cell header block")

    cell_params: Optional[Tuple[float, float, float, float, float, float]]
    if None not in (a, b, c, alpha, beta, gamma):
        cell_params = (a, b, c, alpha, beta, gamma)
    else:
        cell_params = None

    return lattice_type, centering, unique_axis, cell_params


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
    # Remove left-hand label
    if "=" not in line:
        raise ValueError(f"Malformed reciprocal line (no '='): {line!r}")

    _, rhs = line.split("=", 1)
    rhs = rhs.strip()

    # Remove trailing unit, e.g. 'nm^-1'
    if "nm^-1" in rhs:
        rhs = rhs.split("nm^-1", 1)[0].strip()

    parts = rhs.split()
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            # Ignore tokens that aren't floats
            pass

    if len(values) != 3:
        raise ValueError(f"Expected 3 numeric components in reciprocal line, got {len(values)}: {line!r}")

    return values


def extract_geometry_block(lines: List[str]) -> Optional[str]:
    """
    Extract the geometry block between:

      ----- Begin geometry -----
      ...
      ----- End geometry -----

    Returns the block as a single string including newlines, or None if not found.
    """
    in_geom = False
    buf: List[str] = []

    for line in lines:
        s = line.strip()
        if s == "----- Begin geometry -----":
            in_geom = True
            continue
        if s == "----- End geometry -----":
            break
        if in_geom:
            buf.append(line)

    if buf:
        return "".join(buf)
    return None


def write_cell_file(
    cell_path: str,
    lattice_type: str,
    centering: str,
    cell_params: Tuple[float, float, float, float, float, float],
) -> None:
    """
    Write a simple CrystFEL .cell file from unit cell parameters.
    """
    a, b, c, alpha, beta, gamma = cell_params
    with open(cell_path, "w") as fh:
        fh.write(f"lattice_type {lattice_type}\n")
        fh.write(f"centering {centering}\n")
        fh.write(f"a {a:.6f}\n")
        fh.write(f"b {b:.6f}\n")
        fh.write(f"c {c:.6f}\n")
        fh.write(f"alpha {alpha:.6f}\n")
        fh.write(f"beta {beta:.6f}\n")
        fh.write(f"gamma {gamma:.6f}\n")


def stream_to_sol(
    stream_path: str,
    sol_path: Optional[str] = None,
    geom_path: Optional[str] = None,
    cell_path: Optional[str] = None,
) -> None:
    """
    Convert a CrystFEL stream file to a .sol file.

    Each crystal in each chunk becomes one line in the .sol file:
        <h5_path> //<Event> <9 a*/b*/c* components> <det_shift_x_mm> <det_shift_y_mm> <bravais>

    - Unit cell (for bravais) is taken from the header 'unit cell' block.
    - Multi-crystal chunks produce multiple lines, duplicating the Event number.
    - Chunks with no crystals are skipped.
    - Optionally writes:
        * geom_path: geometry block from 'Begin geometry'/'End geometry'
        * cell_path: CrystFEL .cell from unit cell parameters (if present)
    """
    with open(stream_path, "r") as fh:
        lines = fh.readlines()

    lattice_type, centering, unique_axis, cell_params = parse_unit_cell(lines)
    bravais = compute_bravais(lattice_type, centering, unique_axis)

    if sol_path is None:
        base, _ = os.path.splitext(stream_path)
        sol_path = base + ".sol"

    # Default geom/cell paths if requested but not specified
    if geom_path is None:
        base, _ = os.path.splitext(stream_path)
        geom_path = base + ".geom"

    if cell_path is None:
        base, _ = os.path.splitext(stream_path)
        cell_path = base + ".cell"

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

            # Flatten astar, bstar, cstar into 9 components
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

                # Crystal-level (but we reset again at each 'Begin crystal')
                astar = bstar = cstar = None
                continue

            if s == "----- End chunk -----":
                in_chunk = False
                in_crystal = False
                astar = bstar = cstar = None
                continue

            if not in_chunk:
                # Ignore everything outside chunks (we already parsed unit cell)
                continue

            # Inside a chunk
            if s.startswith("Image filename:"):
                # Image filename: /path/to/file.h5
                _, rhs = s.split(":", 1)
                current_image = rhs.strip()
                continue

            if s.startswith("Event:"):
                # Event: //16
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
                # header/float//entry/data/det_shift_x_mm = 0.317321
                _, rhs = s.split("=", 1)
                det_x = float(rhs.strip())
                continue

            if s.startswith("header/float//entry/data/det_shift_y_mm"):
                # header/float//entry/data/det_shift_y_mm = 0.092506
                _, rhs = s.split("=", 1)
                det_y = float(rhs.strip())
                continue

            # Crystal boundaries inside chunk
            if s == "--- Begin crystal":
                in_crystal = True
                astar = bstar = cstar = None
                continue

            if s == "--- End crystal":
                # Only write if we have a full set of a*/b*/c*
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

    # After writing .sol, write geometry (if present)
    geom_block = extract_geometry_block(lines)
    if geom_block:
        with open(geom_path, "w") as gfh:
            gfh.write(geom_block)
        print(f"Wrote geometry block to {geom_path}")
    else:
        print("No 'Begin geometry'/'End geometry' block found; not writing .geom")

    # After writing .sol, write cell if we have parameters
    if cell_params is not None:
        write_cell_file(cell_path, lattice_type, centering, cell_params)
        print(f"Wrote unit cell to {cell_path}")
    else:
        print("Unit cell metric parameters not found in header; not writing .cell")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CrystFEL stream file into a .sol file (+ optional .geom and .cell)"
    )
    parser.add_argument(
        "stream",
        help="Input CrystFEL .stream file",
    )
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
        help="Output .cell file (default: <stream_basename>.cell, only if unit cell parameters are present)",
    )

    args = parser.parse_args()
    stream_path: str = args.stream
    sol_path: Optional[str] = args.output
    geom_path: Optional[str] = args.geom_out
    cell_path: Optional[str] = args.cell_out

    stream_to_sol(stream_path, sol_path, geom_path, cell_path)


if __name__ == "__main__":
    main()
