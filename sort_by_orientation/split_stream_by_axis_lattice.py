#!/usr/bin/env python3
"""
split_stream_by_axis_lattice.py
===============================

CrystFEL .stream utilities where **each chunk is one event**:

  • Auto-detect lattice (Bravais) from the stream **header** (best effort).
  • Use hard-coded "problematic" zone axes per lattice (as specified by user).
  • For every **chunk/event**, compute the closest axis angle to the beam
    using astar/bstar/cstar from the crystal block (if present).
  • Three operations:
      1) --sort-angle         : write a copy of the input with events
                                sorted by ascending angle deviation.
                                Supports --include-unindexed and --count-split.
      2) --angle-bins / --angle-split : split into angle-deviation bins.
      3) (optional earlier mode removed: frame bins by IDs; not needed here.)

Conventions
-----------
• Each **chunk** is an event/frame. If no astar/bstar/cstar is found, the chunk
  is treated as **unindexed** (angle = +inf).
• Angle bins use (low, high] except the first bin, which is [low, high].
• Sorting ties are broken by original order (seq); angle equality keeps stability.

Examples
--------
Sorted copy (low→high angle), unindexed at end, plus a text report:
  python split_stream_by_axis_lattice.py INPUT.stream sorted_by_angle.stream \
    --sort-angle --report closest_axis_angles.txt

Sorted copy, split every 20k events:
  python split_stream_by_axis_lattice.py INPUT.stream sorted_by_angle.stream \
    --sort-angle --count-split 20000

Angle bins (like your current workflow):
  python split_stream_by_axis_lattice.py INPUT.stream out_angle.stream \
    --angle-bins 0,12.5,50 --report closest_axis_angles.txt
"""

from __future__ import annotations
import argparse, math, re, unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np

# ---------- stream markers (read binary to track byte offsets) ----------
BEGIN_CHUNK_B = b"----- Begin chunk"
END_CHUNK_B   = b"----- End chunk"
EVENT_FIELD_B = b"Event:"

# ---------- regex (text mode decode per-line) ----------
FLOAT_RE = r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
AXIS_LINE_RE = {
    'astar': re.compile(rf"astar\s*=\s*{FLOAT_RE}\s+{FLOAT_RE}\s+{FLOAT_RE}", re.I),
    'bstar': re.compile(rf"bstar\s*=\s*{FLOAT_RE}\s+{FLOAT_RE}\s+{FLOAT_RE}", re.I),
    'cstar': re.compile(rf"cstar\s*=\s*{FLOAT_RE}\s+{FLOAT_RE}\s+{FLOAT_RE}", re.I),
}
IMG_FN_RE  = re.compile(r'^Image filename:\s*(\S+)')
# allow IDs like //41-1, //113-1 etc.
EVENT_ANY_RE = re.compile(r'^Event:\s*//\s*([^\s]+)')

# ---------- axes per lattice (you can extend these as you like) ----------
def _axes(*triples: str) -> List[Tuple[int,int,int]]:
    out = []
    for s in triples:
        s2 = re.sub(r'(?<=\d)-', ' -', s)   # "1-10" -> "1 -1 0"
        nums = re.findall(r"-?\d+", s2)
        if len(nums) == 1 and len(nums[0]) == 3:
            nums = list(nums[0])
        if len(nums) != 3:
            raise ValueError(f"Bad axis spec in table: {s}")
        out.append((int(nums[0]), int(nums[1]), int(nums[2])))
    return out

AXES_BY_LATTICE: Dict[str, List[Tuple[int,int,int]]] = {
    # No rotational symmetry → no unique “high-symmetry” directions.
    # Still, use the basis axes as the minimal, practical primaries.
    "triclinic": _axes(
        "1 0 0","-1 0 0",
        "0 1 0","0 -1 0",
        "0 0 1","0 0 -1"
    ),

    # 2/m with unique b: [010] is the only proper rotation axis (2-fold).
    # [100] and [001] are the other lattice axes (not rotationally special,
    # but standard primaries to complement [010]).
    "monoclinic": _axes(
        "0 1 0","0 -1 0",     # <010>  (2-fold axis)
        "1 0 0","-1 0 0",     # <100>
        "0 0 1","0 0 -1"      # <001>
    ),

    # mmm: three orthogonal 2-fold axes → the three basis directions are the primaries.
    "orthorhombic": _axes(
        "1 0 0","-1 0 0",     # <100>
        "0 1 0","0 -1 0",     # <010>
        "0 0 1","0 0 -1"      # <001>
    ),

    # # 4/mmm: principal 4-fold along c; two inequivalent 2-folds in the basal plane.
    "tetragonal": _axes(
        "0 0 1","0 0 -1",     # <001>  (4-fold)
        "1 0 0","-1 0 0",     # <100>  (2-fold, along a)
        "1 1 0","-1 -1 0"     # <110>  (2-fold, diagonal in basal plane)
    ),

    # trigonal in hexagonal setting (3m/−3m): 3-fold along c; 2-folds in the basal net.
    "trigonal": _axes(
        "0 0 1","0 0 -1",     # <001>  (3-fold)
        "1 0 0","-1 0 0",     # <100>  (basal 2-fold type A)
        "1 1 0","-1 -1 0"     # <110>  (basal 2-fold type B; not equivalent to <100>)
    ),

    # rhombohedral (r setting): 3-fold along the rhombohedral axis; pick one basal 2-fold family.
    # (In r axes, <100>, <010>, and <1-10> are cycled by the 3-fold; one representative suffices.)
    "rhombohedral": _axes(
        "0 0 1","0 0 -1",     # <001>_r  (3-fold)
        "1 0 0","-1 0 0"      # one basal 2-fold family in r basis
    ),

    # 6/mmm: 6-fold along c; two inequivalent basal 2-folds (<100> and <110>).
    "hexagonal": _axes(
        "0 0 1","0 0 -1",     # <001>  (6-fold)
        "1 0 0","-1 0 0",     # <100>  (basal 2-fold type A)
        "1 1 0","-1 -1 0"     # <110>  (basal 2-fold type B)
    ),

    # m-3m: the classic trio—4-fold, 3-fold, 2-fold.
    "cubic": _axes(
        "0 0 1","0 0 -1",     # <001>  (4-fold)
        "1 1 1","-1 -1 -1",   # <111>  (3-fold)
        "1 1 0","-1 -1 0"     # <110>  (2-fold)
    )
}

LATTICE_ALIASES = {
    "triclinic": ["triclinic", "a triclinic", "p1", " 1 "],
    "monoclinic": ["monoclinic", "2/m", " p2 ", " p21 ", " c2/m "],
    "orthorhombic": ["orthorhombic", "mmm", "222", "p212121", "p2221", "pna21", "fddd"],
    "tetragonal": ["tetragonal", "4/mmm", "4mm", "p4", "i4", "p42", "i41", "p4mm", "p42/mnm", "i4/mmm"],
    "trigonal": ["trigonal", "rhombohedral", "3m", "r-3m", "r3m", "p3", "p-3", "p31m", "p3m1", "p-31m", "p-3m1"],
    "hexagonal": ["hexagonal", "6/mmm", "p6", "p63", "p6/mmm", "p63/mmc"],
    "cubic": ["cubic", "m-3m", "m3m", "fd-3m", "fm-3m", "pm-3m", "im-3m", "ia-3d", "p432", "i432", "f432"],
}

def _normalise_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def read_header_text(stream_path: Path) -> str:
    header = []
    with stream_path.open("rb") as fh:
        while True:
            line = fh.readline()
            if not line or line.startswith(BEGIN_CHUNK_B):
                break
            try:
                header.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                header.append(line.decode("latin-1", errors="replace"))
    return _normalise_text("".join(header))

def detect_lattice_from_header(header_text: str) -> Optional[str]:
    t = header_text
    def any_token(tokens: List[str]) -> bool:
        return any(tok in t for tok in tokens)
    if any_token(LATTICE_ALIASES["cubic"]):        return "cubic"
    if any_token(LATTICE_ALIASES["hexagonal"]):    return "hexagonal"
    if any_token(LATTICE_ALIASES["tetragonal"]):   return "tetragonal"
    if any_token(LATTICE_ALIASES["orthorhombic"]): return "orthorhombic"
    if any_token(LATTICE_ALIASES["monoclinic"]):   return "monoclinic"
    if any_token(LATTICE_ALIASES["trigonal"]):     return "trigonal"  # rhombohedral alias
    if any_token(LATTICE_ALIASES["triclinic"]):    return "triclinic"
    return None

# ---------- math ----------
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1u = v1 / np.linalg.norm(v1)
    v2u = v2 / np.linalg.norm(v2)
    cosang = float(np.clip(abs(np.dot(v1u, v2u)), -1.0, 1.0))   # ±axis equivalence
    return float(np.degrees(np.arccos(cosang)))

# ---------- chunk meta ----------
class ChunkMeta:
    __slots__ = ("seq", "img_base", "event", "angle", "best_axis", "start", "end")
    def __init__(self, seq: int, img_base: str, event: str, start: int, end: int):
        self.seq       : int  = seq
        self.img_base  : str  = img_base
        self.event     : str  = event
        self.start     : int  = start
        self.end       : int  = end
        self.angle     : float = math.inf
        self.best_axis : Optional[Tuple[int,int,int]] = None

# ---------- pass 1: index all chunks (compute per-chunk angle) ----------
def index_chunks(
    stream_path: Path,
    axes: List[Tuple[int,int,int]],
    beam_xyz: np.ndarray
) -> Tuple[bytes, List[ChunkMeta]]:
    header = bytearray()
    chunks: List[ChunkMeta] = []

    with stream_path.open("rb") as fh:
        in_header = True
        in_chunk = False
        start_offset = 0

        seq = 0
        img_base: Optional[str] = None
        event_s : Optional[str] = None
        have_astar = have_bstar = have_cstar = False
        astar = bstar = cstar = None

        while True:
            pos_before = fh.tell()
            line_b = fh.readline()
            if not line_b:
                break  # EOF

            if in_header:
                if line_b.startswith(BEGIN_CHUNK_B):
                    in_header = False
                    in_chunk  = True
                    start_offset = pos_before
                    img_base = None
                    event_s = None
                    have_astar = have_bstar = have_cstar = False
                    astar = bstar = cstar = None
                else:
                    header += line_b
                continue

            if line_b.startswith(BEGIN_CHUNK_B):
                in_chunk  = True
                start_offset = pos_before
                img_base = None
                event_s = None
                have_astar = have_bstar = have_cstar = False
                astar = bstar = cstar = None
                continue

            if line_b.startswith(END_CHUNK_B):
                end_offset = fh.tell()
                # finalize this chunk
                img_out = img_base or "unknown_image"
                evt_out = event_s or "?"
                cm = ChunkMeta(seq, img_out, evt_out, start_offset, end_offset)

                if have_astar and have_bstar and have_cstar:
                    Gstar = np.column_stack((astar, bstar, cstar))
                    try:
                        _ = float(np.linalg.det(Gstar))  # sanity
                        A_real = np.linalg.inv(Gstar).T  # <-- REAL-SPACE BASIS
                        best_ang = math.inf
                        best_axis = None
                        for uvw in axes:
                            axis_vec = A_real @ np.asarray(uvw, float)  # real-space zone dir
                            ang = angle_between(axis_vec, beam_xyz)
                            if ang < best_ang:
                                best_ang = ang
                                best_axis = uvw
                        cm.angle = best_ang
                        cm.best_axis = best_axis
                    except Exception:
                        pass  # leave as unindexed/inf

                chunks.append(cm)
                seq += 1
                in_chunk = False
                continue

            # within chunk: parse metadata of interest
            if in_chunk:
                text = line_b.decode(errors="replace")
                if (m := IMG_FN_RE.match(text)):
                    img_base = Path(m.group(1)).name
                    continue
                if (m := EVENT_ANY_RE.match(text)):
                    event_s = m.group(1)
                    continue
                for key, rx in AXIS_LINE_RE.items():
                    m = rx.match(text)
                    if m:
                        vec = tuple(float(x) for x in m.groups())
                        if key == 'astar':
                            astar = np.asarray(vec, float); have_astar = True
                        elif key == 'bstar':
                            bstar = np.asarray(vec, float); have_bstar = True
                        elif key == 'cstar':
                            cstar = np.asarray(vec, float); have_cstar = True

    return bytes(header), chunks

# ---------- helpers for bin logic ----------
def parse_angle_bins(split: Optional[float], bins: Optional[Sequence[float]]) -> List[Tuple[float,float]]:
    if split is not None:
        if split <= 0:
            raise ValueError("--angle-split must be positive")
        out, start = [], 0.0
        while True:
            end = start + split
            out.append((start, end))
            if end > 180.0:
                out.append((end, math.inf))
                break
            start = end
        return out
    assert bins is not None and len(bins) >= 2
    if list(bins) != sorted(bins):
        raise ValueError("--angle-bins must be ascending")
    out = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    out.append((bins[-1], math.inf))
    return out

def in_bin_float(val: float, low: float, high: float, first: bool) -> bool:
    return (low <= val <= high) if first else (low < val <= high)

# ---------- write: angle-binned outputs ----------
def write_angle_bins(
    src_path: Path,
    header: bytes,
    chunks: List[ChunkMeta],
    out_prefix: Path,
    bins: List[Tuple[float,float]],
    report_path: Optional[Path] = None
) -> None:
    # report (optional)
    rep_fh = None
    if report_path is not None:
        rep_fh = report_path.open("w", encoding="utf-8")
        for cm in chunks:
            if math.isfinite(cm.angle) and cm.best_axis is not None:
                ax = cm.best_axis
                print(f"{cm.img_base} //{cm.event} -> [{ax[0]} {ax[1]} {ax[2]}] angle={cm.angle:.2f}°", file=rep_fh)
            else:
                print(f"{cm.img_base} //{cm.event} -> unindexed", file=rep_fh)

    # open bin files lazily
    handles: List[Tuple[Tuple[float,float], Path, "object"]] = []

    def get_handle(low: float, high: float):
        for (l,h), path, fh in handles:
            if l == low and h == high:
                return fh
        suffix = f"angle_{low}-{('inf' if math.isinf(high) else high)}"
        out_path = out_prefix.with_name(f"{out_prefix.stem}_{suffix}{out_prefix.suffix}")
        fh = out_path.open("wb"); fh.write(header)
        handles.append(((low,high), out_path, fh))
        return fh

    # sort by angle to make bins deterministic
    indexed = [cm for cm in chunks if math.isfinite(cm.angle)]
    indexed.sort(key=lambda cm: (cm.angle, cm.seq))

    with src_path.open("rb") as src:
        for cm in indexed:
            placed = False
            for j,(low,high) in enumerate(bins):
                first = (j == 0)
                if in_bin_float(cm.angle, low, high, first):
                    dst = get_handle(low, high)
                    src.seek(cm.start); dst.write(src.read(cm.end - cm.start))
                    placed = True
                    break
            if not placed:
                raise RuntimeError(f"No angle bin matched {cm.angle}° for {cm.img_base}//{cm.event}")

    for (_, _), path, fh in handles:
        fh.close()
        print(f"Wrote → {path}")
    if rep_fh:
        rep_fh.close()

# ---------- write: sorted copy (by angle) ----------
def write_sorted_by_angle(
    src_path: Path,
    header: bytes,
    chunks: List[ChunkMeta],
    out_base: Path,
    *,
    include_unindexed: str = "end",    # 'end'|'start'|'drop'
    count_split: Optional[int] = None  # N chunks per part, None = single file
) -> None:
    if include_unindexed not in {"end","start","drop"}:
        raise ValueError("--include-unindexed must be one of: end, start, drop")
    if count_split is not None and count_split <= 0:
        raise ValueError("--count-split must be a positive integer")

    indexed   = [cm for cm in chunks if math.isfinite(cm.angle)]
    unindexed = [cm for cm in chunks if not math.isfinite(cm.angle)]

    indexed.sort(key=lambda cm: (cm.angle, cm.seq))
    # keep original order for unindexed (stable on seq)
    unindexed.sort(key=lambda cm: cm.seq)

    if include_unindexed == "start":
        order = unindexed + indexed
    elif include_unindexed == "end":
        order = indexed + unindexed
    else:  # drop
        order = indexed

    # batching
    groups: List[List[ChunkMeta]]
    if count_split is None:
        groups = [order]
    else:
        groups = [order[i:i+count_split] for i in range(0, len(order), count_split)]

    with src_path.open("rb") as src:
        for gi, grp in enumerate(groups):
            out_path = (out_base if count_split is None
                        else out_base.with_name(f"{out_base.stem}_part{gi+1}{out_base.suffix}"))
            with out_path.open("wb") as dst:
                dst.write(header)
                for cm in grp:
                    src.seek(cm.start); dst.write(src.read(cm.end - cm.start))

# ---------- CLI ----------
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    ap.add_argument("input_stream", type=Path)

    # Make 'output' optional to preserve backward compatibility AND allow omission
    ap.add_argument(
        "output",
        type=Path,
        nargs="?",  # <-- optional now
        help=("For --sort-angle: output .stream filename (optional: defaults to <input>_sorted.stream). "
              "For angle binning: prefix (optional: defaults to <input>_bin).")
    )

    ap.add_argument("--beam", type=float, nargs=3, metavar=("X","Y","Z"),
                    default=(0.0, 0.0, 1.0),
                    help="Lab-frame beam direction (default 0 0 1)")

    # choose axes: auto by lattice (from header), or manual override
    ap.add_argument("--lattice", type=str, default="auto",
                    choices=["auto","triclinic","monoclinic","orthorhombic","tetragonal","trigonal","rhombohedral","hexagonal","cubic"],
                    help="Axis family based on lattice (default: auto-detect from header).")
    ap.add_argument("--axes", nargs="+", metavar="UVW",
                    help="Manual axes (3 ints or '110' strings). If given, --lattice is ignored.")
    ap.add_argument("--printaxes", action="store_true",
                    help="Print the axes that will be used and exit.")

    # operation modes
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sort-angle", action="store_true",
                      help="Write a copy of the input with chunks sorted by ascending angle deviation.")
    mode.add_argument("--angle-split", type=float,
                      help="Angle bin width in degrees (e.g., 5).")
    mode.add_argument("--angle-bins",  type=str,
                      help="Comma-separated angle edges, e.g., 0,12.5,50")

    # options
    ap.add_argument("--include-unindexed", choices=("end","start","drop"),
                    default="end",
                    help="Where unindexed chunks go in --sort-angle (default: end).")
    ap.add_argument("--count-split", type=int,
                    help="After --sort-angle, split into files of N chunks each (optional).")
    ap.add_argument("--report", action="store_true",
                    help="Write per-chunk report '<img> //<event> -> [uvw] angle=…'.")

    args = ap.parse_args(argv)

    # decide axes
    if args.axes:
        axes_user: List[Tuple[int,int,int]] = []
        for s in args.axes:
            s2 = re.sub(r'(?<=\d)-', ' -', s)
            nums = re.findall(r"-?\d+", s2)
            if len(nums) == 1 and len(nums[0]) == 3:
                nums = list(nums[0])
            if len(nums) != 3:
                raise SystemExit(f"Bad axis specification: '{s}'")
            axes_user.append((int(nums[0]), int(nums[1]), int(nums[2])))
        axes_name = "custom"
    else:
        if args.lattice == "auto":
            header_text = read_header_text(args.input_stream)
            lattice = detect_lattice_from_header(header_text)
            if lattice is None:
                raise SystemExit(
                    "Could not auto-detect lattice from header. "
                    "Use --lattice {triclinic|monoclinic|orthorhombic|tetragonal|trigonal|rhombohedral|hexagonal|cubic} "
                    "or provide --axes …"
                )
            axes_user = AXES_BY_LATTICE["trigonal" if lattice == "rhombohedral" else lattice]
            axes_name = lattice
        else:
            lattice = "trigonal" if args.lattice == "rhombohedral" else args.lattice
            axes_user = AXES_BY_LATTICE[lattice]
            axes_name = args.lattice

    if args.printaxes:
        print(f"Using axis set for: {axes_name}")
        for uvw in axes_user:
            print(f"[{uvw[0]} {uvw[1]} {uvw[2]}]")
        return

    # index pass
    beam_xyz = np.asarray(args.beam, float)
    header_bytes, chunk_list = index_chunks(args.input_stream, axes_user, beam_xyz)

    # Define the report filename relative to the input_stream
    report_path = args.input_stream.with_name(args.input_stream.stem + "_report.txt")


    # optional report
    if args.report:
        valid = []
        invalid = []
        for cm in chunk_list:
            if math.isfinite(cm.angle) and cm.best_axis is not None:
                valid.append(cm)
            else:
                invalid.append(cm)

        # sort valid ones by angle (ascending)
        valid.sort(key=lambda cm: cm.angle)

        with report_path.open("w", encoding="utf-8") as rep:
            for cm in valid:
                ax = cm.best_axis
                rep.write(f"{cm.img_base} //{cm.event} -> "
                        f"[{ax[0]} {ax[1]} {ax[2]}] angle={cm.angle:.2f}°\n")

            # write the unindexed ones last
            for cm in invalid:
                rep.write(f"{cm.img_base} //{cm.event} -> unindexed\n")

        print(f"Wrote report → {report_path}")


    # ---- derive defaults for outputs when 'output' not provided ----
    # For --sort-angle: a concrete .stream filename
    default_sorted_path = args.input_stream.with_name(args.input_stream.stem + "_sorted.stream")
    # For binning: a prefix (no extension; your writer will add suffixes/extensions)
    default_bin_prefix = args.input_stream.with_name(args.input_stream.stem + "_bin")

    # perform operation
    if args.sort_angle:
        out_base = args.output if args.output is not None else default_sorted_path
        write_sorted_by_angle(
            args.input_stream,
            header_bytes,
            chunk_list,
            out_base=out_base,
            include_unindexed=args.include_unindexed,
            count_split=args.count_split
        )
        if args.output is None:
            print(f"Wrote sorted stream → {out_base}")
    else:
        # angle binning
        if args.angle_bins:
            bins = [float(x) for x in args.angle_bins.split(",")]
            angle_bins = parse_angle_bins(None, bins)
        elif args.angle_split is not None:
            angle_bins = parse_angle_bins(args.angle_split, None)
        else:
            raise SystemExit("Provide --angle-bins or --angle-split, or use --sort-angle.")

        out_prefix = args.output if args.output is not None else default_bin_prefix
        write_angle_bins(
            args.input_stream,
            header_bytes,
            chunk_list,
            out_prefix=out_prefix,
            bins=angle_bins,
            report_path=None  # report already handled above if requested
        )
        if args.output is None:
            print(f"Wrote angle-binned outputs with prefix → {out_prefix}")

if __name__ == "__main__":
    main()
