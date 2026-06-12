#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_bad_keys(scores_path: str, threshold: float, chunksize: int = 1_000_000):
    """
    Return set of (frame, h, k, l) keys where sigma_dyn_rel >= threshold.
    Uses chunks so large OriDyn CSVs do not need to be fully loaded.
    """
    usecols = ["frame", "h", "k", "l", "sigma_dyn_rel"]
    bad = set()
    total_rows = 0
    bad_rows = 0

    for chunk in pd.read_csv(scores_path, usecols=usecols, chunksize=chunksize):
        total_rows += len(chunk)

        mask = pd.to_numeric(chunk["sigma_dyn_rel"], errors="coerce") >= threshold
        sub = chunk.loc[mask, ["frame", "h", "k", "l"]].dropna()

        bad_rows += len(sub)

        for row in sub.itertuples(index=False, name=None):
            frame, h, k, l = row
            bad.add((int(frame), int(h), int(k), int(l)))

    return bad, total_rows, bad_rows


def parse_reflection_hkl(line: str):
    """
    Return (h,k,l) if this looks like a CrystFEL reflection row, otherwise None.
    Reflection rows start with h k l I sigma ...
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return None

    return h, k, l


def filter_stream(stream_path: str, output_path: str, bad_keys: set[tuple[int, int, int, int]]):
    stream_path = Path(stream_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_chunk = False
    in_crystal = False
    in_reflections = False

    chunk_id = -1
    crystal_counter = 0

    reflection_rows_seen = 0
    reflection_rows_removed = 0
    reflection_rows_kept = 0
    crystals_seen = 0
    chunks_seen = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as inp, \
         output_path.open("w", encoding="utf-8") as out:

        for raw_line in inp:
            line = raw_line.rstrip("\n")

            if line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                chunk_id += 1
                chunks_seen += 1
                out.write(raw_line)
                continue

            if line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                out.write(raw_line)
                continue

            if line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                out.write(raw_line)
                continue

            if line.startswith("--- End crystal"):
                if in_crystal:
                    crystal_counter += 1
                    crystals_seen += 1
                in_crystal = False
                in_reflections = False
                out.write(raw_line)
                continue

            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                out.write(raw_line)
                continue

            if in_reflections and "End of reflections" in line:
                in_reflections = False
                out.write(raw_line)
                continue

            if in_chunk and in_crystal and in_reflections:
                hkl = parse_reflection_hkl(line)
                if hkl is not None:
                    reflection_rows_seen += 1
                    h, k, l = hkl
                    key = (crystal_counter, h, k, l)

                    if key in bad_keys:
                        reflection_rows_removed += 1
                        continue

                    reflection_rows_kept += 1
                    out.write(raw_line)
                    continue

            out.write(raw_line)

    return {
        "chunks_seen": chunks_seen,
        "crystals_seen": crystals_seen,
        "reflection_rows_seen": reflection_rows_seen,
        "reflection_rows_kept": reflection_rows_kept,
        "reflection_rows_removed": reflection_rows_removed,
    }


def main():
    p = argparse.ArgumentParser(
        description="Filter CrystFEL stream reflections using OriDyn sigma_dyn_rel threshold."
    )
    p.add_argument("--stream", required=True, help="Input CrystFEL stream.")
    p.add_argument("--scores", required=True, help="OriDyn reflection_scores.csv.")
    p.add_argument("--output", required=True, help="Output filtered stream.")
    p.add_argument(
        "--sigma-dyn-rel-min",
        type=float,
        required=True,
        help="Remove reflections with sigma_dyn_rel >= this value.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="CSV chunk size for reading large reflection_scores.csv.",
    )

    args = p.parse_args()

    bad_keys, total_rows, bad_rows = load_bad_keys(
        args.scores,
        threshold=args.sigma_dyn_rel_min,
        chunksize=args.chunksize,
    )

    stats = filter_stream(args.stream, args.output, bad_keys)

    print("Scores rows read:", total_rows)
    print("Scores rows above threshold:", bad_rows)
    print("Unique keys to remove:", len(bad_keys))
    print("Chunks seen:", stats["chunks_seen"])
    print("Crystals seen:", stats["crystals_seen"])
    print("Reflection rows seen:", stats["reflection_rows_seen"])
    print("Reflection rows removed:", stats["reflection_rows_removed"])
    print("Reflection rows kept:", stats["reflection_rows_kept"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
