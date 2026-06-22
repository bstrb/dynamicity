#!/usr/bin/env python3
"""Filter CrystFEL stream reflections by non-self OriDyn risk with 4/mmm safety grouping.

This tool removes high non-self-risk measured reflection observations from a stream
before partialator. Matching is done using signed HKLs and source_filename+event.
Filtering safety decisions are made on canonical 4/mmm HKL groups.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


SOURCE_COLUMNS = (
    "source_filename",
    "target_source",
    "source",
    "filename",
    "image_filename",
    "image",
    "file",
    "Image filename",
)

NONSELF_COMPONENT_COLUMNS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

FOCUS_FAMILIES = [
    (0, 0, 4),
    (0, 0, 6),
    (0, 0, 8),
    (0, 0, 10),
    (0, 0, 12),
    (0, 0, 14),
    (0, 0, 16),
    (16, 0, 0),
    (6, 0, 0),
    (3, 3, 0),
    (4, 4, 0),
    (4, 0, 0),
    (0, 4, 0),
    (7, 7, 0),
]

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter measured stream reflections using non-self OriDyn risk with signed-key matching "
            "and canonical 4/mmm group safety checks."
        )
    )
    parser.add_argument("--stream", required=True, type=Path, help="Input CrystFEL .stream file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder for filtered stream and diagnostics")
    parser.add_argument("--pointgroup", default="4/mmm", help="Safety-group point group (default: 4/mmm)")
    parser.add_argument("--remove-top-fraction", type=float, default=0.50)
    parser.add_argument("--min-obs-all", type=int, default=100)
    parser.add_argument("--min-obs-kept", type=int, default=50)
    parser.add_argument("--min-nonself-spread", type=float, default=0.05)
    parser.add_argument("--max-crystal-removed-fraction", type=float, default=0.50)
    parser.add_argument("--min-crystal-reflections-kept", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--max-events", type=int, default=None, help="Optional smoke-test limit on chunks/events")
    parser.add_argument("--max-reflections", type=int, default=None, help="Optional smoke-test limit on measured reflections")
    parser.add_argument(
        "--hkl-list",
        type=str,
        default=None,
        help='Optional semicolon-separated signed HKLs, e.g. "0,0,4;16,0,0;3,3,0"',
    )
    parser.add_argument("--scores-chunksize", type=int, default=1000000)
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if args.pointgroup != "4/mmm":
        raise SystemExit("This script currently supports only --pointgroup 4/mmm")
    if not (0.0 <= float(args.remove_top_fraction) <= 1.0):
        raise SystemExit("--remove-top-fraction must be in [0, 1]")
    if int(args.min_obs_all) < 1:
        raise SystemExit("--min-obs-all must be >= 1")
    if int(args.min_obs_kept) < 1:
        raise SystemExit("--min-obs-kept must be >= 1")
    if float(args.min_nonself_spread) < 0.0:
        raise SystemExit("--min-nonself-spread must be >= 0")
    if not (0.0 <= float(args.max_crystal_removed_fraction) <= 1.0):
        raise SystemExit("--max-crystal-removed-fraction must be in [0, 1]")
    if int(args.min_crystal_reflections_kept) < 0:
        raise SystemExit("--min-crystal-reflections-kept must be >= 0")
    if int(args.progress_every) < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.max_events is not None and int(args.max_events) < 1:
        raise SystemExit("--max-events must be >= 1 when provided")
    if args.max_reflections is not None and int(args.max_reflections) < 1:
        raise SystemExit("--max-reflections must be >= 1 when provided")
    if int(args.scores_chunksize) < 1:
        raise SystemExit("--scores-chunksize must be >= 1")

    return args


def normalize_source(value: Any) -> str:
    return str(value).strip()


def normalize_event(value: Any) -> str:
    return str(value).strip()


def hkl_text(h: int, k: int, l: int) -> str:
    return f"({h},{k},{l})"


def canonicalize_4mmm(h: int, k: int, l: int) -> tuple[int, int, int]:
    """Canonicalize signed HKL into a tetragonal 4/mmm family representative."""
    ah = abs(int(h))
    ak = abs(int(k))
    al = abs(int(l))
    return max(ah, ak), min(ah, ak), al


def parse_hkl_list(text: str | None) -> set[tuple[int, int, int]] | None:
    if text is None or not str(text).strip():
        return None

    out: set[tuple[int, int, int]] = set()
    tokens = [tok.strip() for tok in str(text).split(";") if tok.strip()]
    for token in tokens:
        parts = [p.strip() for p in token.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"Invalid --hkl-list entry: {token!r}")
        try:
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError as exc:
            raise SystemExit(f"Invalid --hkl-list entry (non-integer HKL): {token!r}") from exc
        out.add((h, k, l))
    return out


def parse_reflection_hkl(line: str) -> tuple[int, int, int] | None:
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


def build_key(source: str, event: str, h: int, k: int, l: int) -> str:
    return f"{source}\t{event}\t{int(h)}\t{int(k)}\t{int(l)}"


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [col for col in SOURCE_COLUMNS if col in header]
    if not candidates:
        raise SystemExit(
            "Scores file does not contain a recognized source column. "
            f"Checked {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )
    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for col in candidates:
        if looks_like_source_filename(sample[col]):
            return col
    return candidates[0]


def ensure_output_layout(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "filtered_stream": root / "filtered_nonself_top50_4mmm.stream",
        "decisions_csv": root / "filter_decisions_per_4mmm_hkl.csv",
        "removed_csv": root / "removed_stream_reflections.csv",
        "kept_csv": root / "kept_stream_reflections.csv",
        "unmatched_csv": root / "unmatched_stream_reflections.csv",
        "per_crystal_csv": root / "per_crystal_filter_summary.csv",
        "readme": root / "README_nonself_stream_filter_top50_4mmm.txt",
    }


def collect_stream_keys(
    stream_path: Path,
    hkl_filter: set[tuple[int, int, int]] | None,
    max_events: int | None,
    max_reflections: int | None,
    progress_every: int,
) -> dict[str, Any]:
    log("Stage 1/5: scanning stream for reflection keys and limits")

    in_chunk = False
    in_crystal = False
    in_reflections = False
    stop_after_chunk = False

    current_source = ""
    current_event = ""
    chunks_read = 0
    crystals_read = 0
    reflections_read = 0
    reflections_considered = 0
    stream_keys: set[str] = set()
    cutoff_line: int | None = None
    last_line_no = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            last_line_no = line_no
            line = raw_line.rstrip("\n")

            if line.startswith("----- Begin chunk -----"):
                if max_events is not None and chunks_read >= int(max_events):
                    cutoff_line = line_no - 1
                    log(
                        "Reached --max-events limit; "
                        f"processing prefix ends at line {cutoff_line:,}"
                    )
                    break
                in_chunk = True
                in_crystal = False
                in_reflections = False
                current_source = ""
                current_event = ""
                chunks_read += 1
                continue

            if line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                if stop_after_chunk:
                    cutoff_line = line_no
                    log(
                        "Reached --max-reflections limit; "
                        f"processing prefix ends at line {cutoff_line:,}"
                    )
                    break
                continue

            if in_chunk and (m := STREAM_IMAGE_RE.match(line)):
                current_source = normalize_source(m.group(1))
                continue

            if in_chunk and (m := STREAM_EVENT_RE.match(line)):
                current_event = normalize_event(m.group(1))
                continue

            if line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                crystals_read += 1
                continue

            if line.startswith("--- End crystal"):
                in_crystal = False
                in_reflections = False
                continue

            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                continue

            if in_reflections and "End of reflections" in line:
                in_reflections = False
                continue

            if in_chunk and in_crystal and in_reflections:
                parsed = parse_reflection_hkl(line)
                if parsed is None:
                    continue

                h, k, l = parsed
                reflections_read += 1
                if (hkl_filter is None) or ((h, k, l) in hkl_filter):
                    reflections_considered += 1
                    stream_keys.add(build_key(current_source, current_event, h, k, l))

                if reflections_read % int(progress_every) == 0:
                    log(
                        "Stage 1 progress: "
                        f"chunks={chunks_read:,}, crystals={crystals_read:,}, "
                        f"reflections={reflections_read:,}, considered={reflections_considered:,}, "
                        f"unique_keys={len(stream_keys):,}"
                    )

                if max_reflections is not None and reflections_read >= int(max_reflections):
                    stop_after_chunk = True

    truncated = cutoff_line is not None
    if truncated:
        log(
            "Stage 1 complete (prefix mode): "
            f"cutoff_line={cutoff_line:,}, chunks={chunks_read:,}, crystals={crystals_read:,}, "
            f"reflections={reflections_read:,}, considered={reflections_considered:,}, unique_keys={len(stream_keys):,}"
        )
    else:
        log(
            "Stage 1 complete (full scan): "
            f"lines={last_line_no:,}, chunks={chunks_read:,}, crystals={crystals_read:,}, "
            f"reflections={reflections_read:,}, considered={reflections_considered:,}, unique_keys={len(stream_keys):,}"
        )

    return {
        "stream_keys": stream_keys,
        "chunks_read": int(chunks_read),
        "crystals_read": int(crystals_read),
        "reflections_read": int(reflections_read),
        "reflections_considered": int(reflections_considered),
        "cutoff_line": cutoff_line,
        "truncated": bool(truncated),
    }


def load_scores_for_stream_keys(
    scores_path: Path,
    stream_keys: set[str],
    scores_chunksize: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], str]:
    log("Stage 2/5: loading OriDyn scores and matching stream keys")

    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l", *NONSELF_COMPONENT_COLUMNS]
    missing = [col for col in required if col not in header]
    if missing:
        raise SystemExit(f"Scores file is missing required columns: {missing}")

    usecols = list(dict.fromkeys(required))
    rows_read = 0
    matched_rows = 0
    chunk_count = 0
    chunks: list[pd.DataFrame] = []

    if not stream_keys:
        log("No stream keys collected; score matching map will be empty")
        return {}, {
            "score_rows_read": 0,
            "score_rows_with_nonself": 0,
            "score_rows_matching_stream_keys": 0,
            "score_unique_matching_keys": 0,
            "score_duplicate_key_rows": 0,
        }, source_column

    for chunk in pd.read_csv(scores_path, usecols=usecols, chunksize=int(scores_chunksize)):
        chunk_count += 1
        rows_read += len(chunk)

        chunk[source_column] = chunk[source_column].map(normalize_source)
        chunk["event"] = chunk["event"].map(normalize_event)

        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        for col in NONSELF_COMPONENT_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.dropna(subset=[source_column, "event", "h", "k", "l"])
        if chunk.empty:
            if chunk_count == 1 or chunk_count % 5 == 0:
                log(
                    "Stage 2 progress: "
                    f"chunks={chunk_count:,}, score_rows_read={rows_read:,}, matched_rows={matched_rows:,}"
                )
            continue

        chunk = chunk.astype({"h": "int64", "k": "int64", "l": "int64"})
        chunk["nonself_mean"] = chunk[NONSELF_COMPONENT_COLUMNS].mean(axis=1, skipna=True)
        chunk = chunk[np.isfinite(chunk["nonself_mean"].to_numpy(dtype=float))].copy()
        if chunk.empty:
            if chunk_count == 1 or chunk_count % 5 == 0:
                log(
                    "Stage 2 progress: "
                    f"chunks={chunk_count:,}, score_rows_read={rows_read:,}, matched_rows={matched_rows:,}"
                )
            continue

        key_series = (
            chunk[source_column].astype(str)
            + "\t"
            + chunk["event"].astype(str)
            + "\t"
            + chunk["h"].astype(str)
            + "\t"
            + chunk["k"].astype(str)
            + "\t"
            + chunk["l"].astype(str)
        )
        mask = key_series.isin(stream_keys)
        if bool(mask.any()):
            sub = chunk.loc[mask, [source_column, "event", "h", "k", "l", *NONSELF_COMPONENT_COLUMNS, "nonself_mean"]].copy()
            sub["key"] = key_series.loc[mask].to_numpy()
            chunks.append(sub)
            matched_rows += len(sub)

        if chunk_count == 1 or chunk_count % 5 == 0:
            log(
                "Stage 2 progress: "
                f"chunks={chunk_count:,}, score_rows_read={rows_read:,}, matched_rows={matched_rows:,}"
            )

    if not chunks:
        log("No score rows matched stream keys")
        return {}, {
            "score_rows_read": int(rows_read),
            "score_rows_with_nonself": 0,
            "score_rows_matching_stream_keys": 0,
            "score_unique_matching_keys": 0,
            "score_duplicate_key_rows": 0,
        }, source_column

    matched_table = pd.concat(chunks, ignore_index=True)
    dup_rows = int(matched_table.duplicated("key", keep=False).sum())

    grouped = matched_table.groupby("key", as_index=False).agg(
        source_filename=(source_column, "first"),
        event=("event", "first"),
        h=("h", "first"),
        k=("k", "first"),
        l=("l", "first"),
        graph_crowding_norm=("graph_crowding_norm", "median"),
        same_laue_zone_crowding_norm=("same_laue_zone_crowding_norm", "median"),
        systematic_row_risk_norm=("systematic_row_risk_norm", "median"),
        frame_axis_risk_norm=("frame_axis_risk_norm", "median"),
        nonself_mean=("nonself_mean", "median"),
    )

    score_map: dict[str, dict[str, Any]] = {}
    for row in grouped.itertuples(index=False):
        score_map[str(row.key)] = {
            "source_filename": str(row.source_filename),
            "event": str(row.event),
            "h": int(row.h),
            "k": int(row.k),
            "l": int(row.l),
            "graph_crowding_norm": float(row.graph_crowding_norm),
            "same_laue_zone_crowding_norm": float(row.same_laue_zone_crowding_norm),
            "systematic_row_risk_norm": float(row.systematic_row_risk_norm),
            "frame_axis_risk_norm": float(row.frame_axis_risk_norm),
            "nonself_mean": float(row.nonself_mean),
        }

    stats = {
        "score_rows_read": int(rows_read),
        "score_rows_with_nonself": int(len(matched_table)),
        "score_rows_matching_stream_keys": int(len(matched_table)),
        "score_unique_matching_keys": int(len(grouped)),
        "score_duplicate_key_rows": int(dup_rows),
    }

    log(
        "Stage 2 complete: "
        f"rows_read={rows_read:,}, matched_rows={len(matched_table):,}, "
        f"unique_keys={len(grouped):,}, duplicate_rows={dup_rows:,}"
    )
    return score_map, stats, source_column


def collect_reflection_observations(
    stream_path: Path,
    score_map: dict[str, dict[str, Any]],
    hkl_filter: set[tuple[int, int, int]] | None,
    cutoff_line: int | None,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    log("Stage 3/5: matching stream reflections to scores and building observation tables")

    in_chunk = False
    in_crystal = False
    in_reflections = False

    current_source = ""
    current_event = ""
    chunk_index = -1
    crystal_index = -1

    matched_records: list[dict[str, Any]] = []
    unmatched_records: list[dict[str, Any]] = []
    crystal_stats: dict[int, dict[str, Any]] = {}

    chunks_read = 0
    crystals_read = 0
    measured_reflections_read = 0
    considered_reflections = 0
    matched_reflections = 0
    unmatched_reflections = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if cutoff_line is not None and line_no > int(cutoff_line):
                break

            line = raw_line.rstrip("\n")

            if line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                current_source = ""
                current_event = ""
                chunk_index += 1
                chunks_read += 1
                continue

            if line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                continue

            if in_chunk and (m := STREAM_IMAGE_RE.match(line)):
                current_source = normalize_source(m.group(1))
                continue

            if in_chunk and (m := STREAM_EVENT_RE.match(line)):
                current_event = normalize_event(m.group(1))
                continue

            if line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                crystal_index += 1
                crystals_read += 1
                crystal_stats[crystal_index] = {
                    "crystal_index": int(crystal_index),
                    "chunk_index": int(chunk_index),
                    "source_filename": str(current_source),
                    "event": str(current_event),
                    "measured_reflections_read": 0,
                    "considered_reflections": 0,
                    "matched_reflections": 0,
                    "unmatched_reflections": 0,
                }
                continue

            if line.startswith("--- End crystal"):
                in_crystal = False
                in_reflections = False
                continue

            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                continue

            if in_reflections and "End of reflections" in line:
                in_reflections = False
                continue

            if in_chunk and in_crystal and in_reflections:
                parsed = parse_reflection_hkl(line)
                if parsed is None:
                    continue

                h, k, l = parsed
                measured_reflections_read += 1
                crystal_stats[crystal_index]["measured_reflections_read"] += 1

                if hkl_filter is not None and (h, k, l) not in hkl_filter:
                    if measured_reflections_read % int(progress_every) == 0:
                        log(
                            "Stage 3 progress: "
                            f"chunks={chunks_read:,}, crystals={crystals_read:,}, measured={measured_reflections_read:,}, "
                            f"considered={considered_reflections:,}, matched={matched_reflections:,}, unmatched={unmatched_reflections:,}"
                        )
                    continue

                considered_reflections += 1
                crystal_stats[crystal_index]["considered_reflections"] += 1
                key = build_key(current_source, current_event, h, k, l)
                canon = canonicalize_4mmm(h, k, l)

                if key in score_map:
                    matched_reflections += 1
                    crystal_stats[crystal_index]["matched_reflections"] += 1
                    s = score_map[key]
                    matched_records.append(
                        {
                            "source_filename": str(current_source),
                            "event": str(current_event),
                            "h": int(h),
                            "k": int(k),
                            "l": int(l),
                            "hkl_signed": hkl_text(int(h), int(k), int(l)),
                            "h_canon": int(canon[0]),
                            "k_canon": int(canon[1]),
                            "l_canon": int(canon[2]),
                            "hkl_canon": hkl_text(int(canon[0]), int(canon[1]), int(canon[2])),
                            "nonself_mean": float(s["nonself_mean"]),
                            "graph_crowding_norm": float(s["graph_crowding_norm"]),
                            "same_laue_zone_crowding_norm": float(s["same_laue_zone_crowding_norm"]),
                            "systematic_row_risk_norm": float(s["systematic_row_risk_norm"]),
                            "frame_axis_risk_norm": float(s["frame_axis_risk_norm"]),
                            "crystal_index": int(crystal_index),
                            "chunk_index": int(chunk_index),
                            "reflection_line_number": int(line_no),
                            "score_key": key,
                        }
                    )
                else:
                    unmatched_reflections += 1
                    crystal_stats[crystal_index]["unmatched_reflections"] += 1
                    unmatched_records.append(
                        {
                            "source_filename": str(current_source),
                            "event": str(current_event),
                            "h": int(h),
                            "k": int(k),
                            "l": int(l),
                            "hkl_signed": hkl_text(int(h), int(k), int(l)),
                            "h_canon": int(canon[0]),
                            "k_canon": int(canon[1]),
                            "l_canon": int(canon[2]),
                            "hkl_canon": hkl_text(int(canon[0]), int(canon[1]), int(canon[2])),
                            "crystal_index": int(crystal_index),
                            "chunk_index": int(chunk_index),
                            "reflection_line_number": int(line_no),
                            "reason": "no_score_match_signed_key",
                        }
                    )

                if measured_reflections_read % int(progress_every) == 0:
                    log(
                        "Stage 3 progress: "
                        f"chunks={chunks_read:,}, crystals={crystals_read:,}, measured={measured_reflections_read:,}, "
                        f"considered={considered_reflections:,}, matched={matched_reflections:,}, unmatched={unmatched_reflections:,}"
                    )

    matched_columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "hkl_signed",
        "h_canon",
        "k_canon",
        "l_canon",
        "hkl_canon",
        "nonself_mean",
        "graph_crowding_norm",
        "same_laue_zone_crowding_norm",
        "systematic_row_risk_norm",
        "frame_axis_risk_norm",
        "crystal_index",
        "chunk_index",
        "reflection_line_number",
        "score_key",
    ]
    unmatched_columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "hkl_signed",
        "h_canon",
        "k_canon",
        "l_canon",
        "hkl_canon",
        "crystal_index",
        "chunk_index",
        "reflection_line_number",
        "reason",
    ]
    crystal_columns = [
        "crystal_index",
        "chunk_index",
        "source_filename",
        "event",
        "measured_reflections_read",
        "considered_reflections",
        "matched_reflections",
        "unmatched_reflections",
    ]

    matched_df = pd.DataFrame.from_records(matched_records, columns=matched_columns)
    unmatched_df = pd.DataFrame.from_records(unmatched_records, columns=unmatched_columns)
    crystal_df = pd.DataFrame.from_records(list(crystal_stats.values()), columns=crystal_columns)
    if not crystal_df.empty:
        crystal_df = crystal_df.sort_values(["crystal_index"]).reset_index(drop=True)

    stats = {
        "chunks_read": int(chunks_read),
        "crystals_read": int(crystals_read),
        "measured_reflections_read": int(measured_reflections_read),
        "considered_reflections": int(considered_reflections),
        "matched_reflections": int(matched_reflections),
        "unmatched_reflections": int(unmatched_reflections),
    }

    log(
        "Stage 3 complete: "
        f"chunks={chunks_read:,}, crystals={crystals_read:,}, measured={measured_reflections_read:,}, "
        f"considered={considered_reflections:,}, matched={matched_reflections:,}, unmatched={unmatched_reflections:,}"
    )
    return matched_df, unmatched_df, crystal_df, stats


def build_candidate_group_filter(
    matched_df: pd.DataFrame,
    remove_top_fraction: float,
    min_obs_all: int,
    min_obs_kept: int,
    min_nonself_spread: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Stage 4/5: computing 4/mmm group-level candidate filtering")

    if matched_df.empty:
        empty = pd.DataFrame(
            columns=[
                "h_canon",
                "k_canon",
                "l_canon",
                "hkl_canon",
                "n_obs_all",
                "n_obs_kept",
                "n_obs_removed",
                "removed_fraction",
                "nonself_p10",
                "nonself_p90",
                "nonself_spread",
                "filter_applied",
                "filter_reason",
            ]
        )
        out = matched_df.copy()
        out["remove_candidate"] = False
        out["remove_final"] = False
        out["restored_by_crystal_safety"] = False
        return out, empty

    df = matched_df.copy()
    df["remove_candidate"] = False
    df["remove_final"] = False
    df["restored_by_crystal_safety"] = False

    decision_rows: list[dict[str, Any]] = []
    grouped = df.groupby(["h_canon", "k_canon", "l_canon"], sort=True)

    for (h_c, k_c, l_c), group in grouped:
        n_obs_all = int(len(group))
        nonself = pd.to_numeric(group["nonself_mean"], errors="coerce")
        nonself_p10 = float(nonself.quantile(0.10))
        nonself_p90 = float(nonself.quantile(0.90))
        nonself_spread = float(nonself_p90 - nonself_p10)

        n_obs_removed_candidate = int(math.floor(float(remove_top_fraction) * float(n_obs_all)))
        n_obs_kept_candidate = int(n_obs_all - n_obs_removed_candidate)

        reasons: list[str] = []
        if n_obs_all < int(min_obs_all):
            reasons.append("n_obs_all_below_min")
        if n_obs_kept_candidate < int(min_obs_kept):
            reasons.append("n_obs_kept_below_min")
        if nonself_spread < float(min_nonself_spread):
            reasons.append("nonself_spread_below_min")
        if n_obs_removed_candidate <= 0:
            reasons.append("remove_fraction_zero")

        candidate_applied = len(reasons) == 0
        if candidate_applied:
            remove_index = (
                group.sort_values(["nonself_mean", "reflection_line_number"], ascending=[False, True])
                .head(n_obs_removed_candidate)
                .index
            )
            df.loc[remove_index, "remove_candidate"] = True
            reason = "candidate_applied"
        else:
            reason = ";".join(reasons)

        decision_rows.append(
            {
                "h_canon": int(h_c),
                "k_canon": int(k_c),
                "l_canon": int(l_c),
                "hkl_canon": hkl_text(int(h_c), int(k_c), int(l_c)),
                "n_obs_all": int(n_obs_all),
                "nonself_p10": float(nonself_p10),
                "nonself_p90": float(nonself_p90),
                "nonself_spread": float(nonself_spread),
                "candidate_removed": int(n_obs_removed_candidate if candidate_applied else 0),
                "candidate_applied": bool(candidate_applied),
                "initial_reason": str(reason),
            }
        )

    decisions = pd.DataFrame(decision_rows)
    log(
        "Stage 4 candidate decisions: "
        f"groups={len(decisions):,}, candidate_applied={int(decisions['candidate_applied'].sum()):,}, "
        f"candidate_removed={int(df['remove_candidate'].sum()):,}"
    )
    return df, decisions


def apply_crystal_safety(
    df: pd.DataFrame,
    crystal_df: pd.DataFrame,
    max_crystal_removed_fraction: float,
    min_crystal_reflections_kept: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    log("Stage 4b/5: enforcing crystal-level safety restoration")

    if crystal_df.empty:
        out = df.copy()
        out["remove_final"] = out["remove_candidate"]
        out["restored_by_crystal_safety"] = False
        per_crystal = pd.DataFrame(
            columns=[
                "crystal_index",
                "chunk_index",
                "source_filename",
                "event",
                "measured_reflections_read",
                "considered_reflections",
                "matched_reflections",
                "unmatched_reflections",
                "candidate_removed",
                "final_removed",
                "restored_count",
                "final_removed_fraction_of_measured",
                "total_kept_final",
                "crystal_safety_required",
                "crystal_safety_passed",
            ]
        )
        return out, per_crystal, 0

    out = df.copy()
    out["remove_final"] = out["remove_candidate"]
    out["restored_by_crystal_safety"] = False

    rows: list[dict[str, Any]] = []
    crystals_requiring_restoration = 0

    for record in crystal_df.itertuples(index=False):
        crystal_index = int(record.crystal_index)
        total_measured = int(record.measured_reflections_read)

        crystal_mask = out["crystal_index"] == crystal_index
        candidate_removed = int(out.loc[crystal_mask, "remove_candidate"].sum()) if bool(crystal_mask.any()) else 0
        final_removed = int(candidate_removed)
        restored_count = 0
        safety_required = False
        safety_passed = True

        if total_measured > 0 and candidate_removed > 0:
            require_keep_floor = total_measured >= int(min_crystal_reflections_kept)

            def violates(removed_count: int) -> bool:
                kept_total = int(total_measured - removed_count)
                removed_fraction = float(removed_count / total_measured)
                too_many_removed = removed_fraction > float(max_crystal_removed_fraction)
                too_few_kept = require_keep_floor and kept_total < int(min_crystal_reflections_kept)
                return bool(too_many_removed or too_few_kept)

            if violates(final_removed):
                safety_required = True
                crystals_requiring_restoration += 1

                restore_order = (
                    out.loc[crystal_mask & out["remove_final"]]
                    .sort_values(["nonself_mean", "reflection_line_number"], ascending=[True, True])
                    .index
                )

                for ridx in restore_order:
                    if not violates(final_removed):
                        break
                    out.at[ridx, "remove_final"] = False
                    out.at[ridx, "restored_by_crystal_safety"] = True
                    final_removed -= 1
                    restored_count += 1

                safety_passed = not violates(final_removed)

        total_kept_final = int(total_measured - final_removed)
        final_removed_fraction = float(final_removed / total_measured) if total_measured > 0 else 0.0

        rows.append(
            {
                "crystal_index": crystal_index,
                "chunk_index": int(record.chunk_index),
                "source_filename": str(record.source_filename),
                "event": str(record.event),
                "measured_reflections_read": total_measured,
                "considered_reflections": int(record.considered_reflections),
                "matched_reflections": int(record.matched_reflections),
                "unmatched_reflections": int(record.unmatched_reflections),
                "candidate_removed": int(candidate_removed),
                "final_removed": int(final_removed),
                "restored_count": int(restored_count),
                "final_removed_fraction_of_measured": float(final_removed_fraction),
                "total_kept_final": int(total_kept_final),
                "crystal_safety_required": bool(safety_required),
                "crystal_safety_passed": bool(safety_passed),
            }
        )

    per_crystal = pd.DataFrame(rows).sort_values(["crystal_index"]).reset_index(drop=True)
    log(
        "Stage 4b complete: "
        f"candidate_removed={int(out['remove_candidate'].sum()):,}, "
        f"final_removed={int(out['remove_final'].sum()):,}, "
        f"restored={int(out['restored_by_crystal_safety'].sum()):,}, "
        f"crystals_requiring_restoration={crystals_requiring_restoration:,}"
    )
    return out, per_crystal, int(crystals_requiring_restoration)


def finalize_group_decisions(df: pd.DataFrame, base_decisions: pd.DataFrame) -> pd.DataFrame:
    if base_decisions.empty:
        return pd.DataFrame(
            columns=[
                "h_canon",
                "k_canon",
                "l_canon",
                "hkl_canon",
                "n_obs_all",
                "n_obs_kept",
                "n_obs_removed",
                "removed_fraction",
                "nonself_p10",
                "nonself_p90",
                "nonself_spread",
                "filter_applied",
                "filter_reason",
            ]
        )

    grouped = (
        df.groupby(["h_canon", "k_canon", "l_canon", "hkl_canon"], sort=True)
        .agg(
            n_obs_all=("nonself_mean", "size"),
            nonself_p10=("nonself_mean", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.10))),
            nonself_p90=("nonself_mean", lambda x: float(pd.to_numeric(x, errors="coerce").quantile(0.90))),
            candidate_removed=("remove_candidate", "sum"),
            n_obs_removed=("remove_final", "sum"),
        )
        .reset_index()
    )
    grouped["nonself_spread"] = grouped["nonself_p90"] - grouped["nonself_p10"]
    grouped["n_obs_kept"] = grouped["n_obs_all"] - grouped["n_obs_removed"]
    grouped["removed_fraction"] = np.where(
        grouped["n_obs_all"] > 0,
        grouped["n_obs_removed"] / grouped["n_obs_all"],
        0.0,
    )

    out = grouped.merge(
        base_decisions[["h_canon", "k_canon", "l_canon", "candidate_applied", "initial_reason"]],
        on=["h_canon", "k_canon", "l_canon"],
        how="left",
    )

    filter_applied: list[bool] = []
    filter_reason: list[str] = []
    for row in out.itertuples(index=False):
        candidate_applied = bool(row.candidate_applied)
        candidate_removed = int(row.candidate_removed)
        final_removed = int(row.n_obs_removed)

        if not candidate_applied:
            filter_applied.append(False)
            filter_reason.append(str(row.initial_reason))
            continue

        if final_removed <= 0:
            filter_applied.append(False)
            if candidate_removed > 0:
                filter_reason.append("restored_all_by_crystal_safety")
            else:
                filter_reason.append("remove_fraction_zero")
            continue

        filter_applied.append(True)
        if final_removed < candidate_removed:
            filter_reason.append("applied_with_crystal_restoration")
        else:
            filter_reason.append("applied")

    out["filter_applied"] = filter_applied
    out["filter_reason"] = filter_reason

    out = out[
        [
            "h_canon",
            "k_canon",
            "l_canon",
            "hkl_canon",
            "n_obs_all",
            "n_obs_kept",
            "n_obs_removed",
            "removed_fraction",
            "nonself_p10",
            "nonself_p90",
            "nonself_spread",
            "filter_applied",
            "filter_reason",
        ]
    ].sort_values(["n_obs_removed", "nonself_spread", "h_canon", "k_canon", "l_canon"], ascending=[False, False, True, True, True])

    return out.reset_index(drop=True)


def build_focus_family_table(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h, k, l in FOCUS_FAMILIES:
        h_c, k_c, l_c = canonicalize_4mmm(h, k, l)
        row = decisions[
            (decisions["h_canon"] == h_c)
            & (decisions["k_canon"] == k_c)
            & (decisions["l_canon"] == l_c)
        ]
        if row.empty:
            rows.append(
                {
                    "requested_hkl": hkl_text(h, k, l),
                    "hkl_canon": hkl_text(h_c, k_c, l_c),
                    "present": False,
                    "n_obs_all": 0,
                    "n_obs_removed": 0,
                    "n_obs_kept": 0,
                    "nonself_spread": np.nan,
                    "filter_applied": False,
                    "filter_reason": "not_present",
                }
            )
            continue

        r = row.iloc[0]
        rows.append(
            {
                "requested_hkl": hkl_text(h, k, l),
                "hkl_canon": str(r["hkl_canon"]),
                "present": True,
                "n_obs_all": int(r["n_obs_all"]),
                "n_obs_removed": int(r["n_obs_removed"]),
                "n_obs_kept": int(r["n_obs_kept"]),
                "nonself_spread": float(r["nonself_spread"]),
                "filter_applied": bool(r["filter_applied"]),
                "filter_reason": str(r["filter_reason"]),
            }
        )

    return pd.DataFrame(rows)


def write_filtered_stream(
    stream_path: Path,
    output_path: Path,
    remove_line_numbers: set[int],
    cutoff_line: int | None,
    progress_every: int,
) -> dict[str, int]:
    log("Stage 5/5: writing filtered stream")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    reflections_seen = 0
    reflections_removed = 0
    reflections_kept = 0

    in_chunk = False
    in_crystal = False
    in_reflections = False

    with stream_path.open("r", encoding="utf-8", errors="replace") as inp, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line_no, raw_line in enumerate(inp, start=1):
            if cutoff_line is not None and line_no > int(cutoff_line):
                break

            line_count += 1
            line = raw_line.rstrip("\n")

            if line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
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
                parsed = parse_reflection_hkl(line)
                if parsed is not None:
                    reflections_seen += 1
                    if int(line_no) in remove_line_numbers:
                        reflections_removed += 1
                    else:
                        reflections_kept += 1
                        out.write(raw_line)

                    if reflections_seen % int(progress_every) == 0:
                        log(
                            "Stage 5 progress: "
                            f"reflection_lines_seen={reflections_seen:,}, removed={reflections_removed:,}, kept={reflections_kept:,}"
                        )
                    continue

            out.write(raw_line)

    log(
        "Stage 5 complete: "
        f"line_count={line_count:,}, reflection_lines_seen={reflections_seen:,}, "
        f"removed={reflections_removed:,}, kept={reflections_kept:,}"
    )
    return {
        "line_count_written": int(line_count),
        "reflection_lines_seen": int(reflections_seen),
        "reflection_lines_removed": int(reflections_removed),
        "reflection_lines_kept": int(reflections_kept),
    }


def write_readme(
    path: Path,
    args: argparse.Namespace,
    outputs: dict[str, Path],
    source_column: str,
    scan_stats: dict[str, Any],
    score_stats: dict[str, int],
    collect_stats: dict[str, int],
    decisions: pd.DataFrame,
    per_crystal: pd.DataFrame,
    crystals_requiring_restoration: int,
) -> None:
    unchanged = decisions.loc[~decisions["filter_applied"]].copy() if not decisions.empty else decisions
    reason_counts = Counter(unchanged["filter_reason"].astype(str).tolist()) if not unchanged.empty else Counter()

    top_removed = decisions.sort_values(["n_obs_removed", "n_obs_all"], ascending=[False, False]).head(20)
    top_spread = decisions.sort_values(["nonself_spread", "n_obs_all"], ascending=[False, False]).head(20)
    focus_table = build_focus_family_table(decisions)

    lines: list[str] = []
    lines.append("Non-self stream filtering with 4/mmm canonical safety")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"- stream: {args.stream}")
    lines.append(f"- scores: {args.scores}")
    lines.append(f"- score_source_column_selected: {source_column}")
    lines.append(f"- pointgroup: {args.pointgroup}")
    lines.append(f"- remove_top_fraction: {args.remove_top_fraction}")
    lines.append(f"- min_obs_all: {args.min_obs_all}")
    lines.append(f"- min_obs_kept: {args.min_obs_kept}")
    lines.append(f"- min_nonself_spread: {args.min_nonself_spread}")
    lines.append(f"- max_crystal_removed_fraction: {args.max_crystal_removed_fraction}")
    lines.append(f"- min_crystal_reflections_kept: {args.min_crystal_reflections_kept}")
    if args.hkl_list:
        lines.append(f"- hkl_list_filter: {args.hkl_list}")
    if args.max_events is not None:
        lines.append(f"- max_events: {args.max_events}")
    if args.max_reflections is not None:
        lines.append(f"- max_reflections: {args.max_reflections}")
    if bool(scan_stats.get("truncated", False)):
        lines.append(f"- prefix_mode_cutoff_line: {scan_stats.get('cutoff_line')}")

    lines.append("")
    lines.append("Output filtered stream path:")
    lines.append(f"- {outputs['filtered_stream']}")
    lines.append("")
    lines.append("Summary counts:")
    lines.append(f"- total_chunks_read: {collect_stats['chunks_read']:,}")
    lines.append(f"- total_crystals_read: {collect_stats['crystals_read']:,}")
    lines.append(f"- total_measured_reflections_read: {collect_stats['measured_reflections_read']:,}")
    lines.append(f"- total_measured_reflections_considered: {collect_stats['considered_reflections']:,}")
    lines.append(f"- total_measured_reflections_matched_to_scores: {collect_stats['matched_reflections']:,}")
    lines.append(f"- total_unmatched_measured_reflections: {collect_stats['unmatched_reflections']:,}")

    total_removed = int(decisions["n_obs_removed"].sum()) if not decisions.empty else 0
    total_kept = int(decisions["n_obs_kept"].sum()) if not decisions.empty else 0
    lines.append(f"- total_reflections_removed: {total_removed:,}")
    lines.append(f"- total_reflections_kept: {total_kept:,}")
    lines.append(f"- total_canonical_4mmm_groups: {len(decisions):,}")
    lines.append(f"- groups_filter_applied: {int(decisions['filter_applied'].sum()) if not decisions.empty else 0:,}")
    lines.append(
        f"- groups_left_unchanged: {int((~decisions['filter_applied']).sum()) if not decisions.empty else 0:,}"
    )
    lines.append(f"- crystals_requiring_restoration: {crystals_requiring_restoration:,}")

    lines.append("")
    lines.append("Reason counts for unchanged groups:")
    if reason_counts:
        for reason, count in reason_counts.most_common():
            lines.append(f"- {reason}: {count:,}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("Score matching stats:")
    lines.append(f"- score_rows_read: {score_stats.get('score_rows_read', 0):,}")
    lines.append(f"- score_rows_matching_stream_keys: {score_stats.get('score_rows_matching_stream_keys', 0):,}")
    lines.append(f"- score_unique_matching_keys: {score_stats.get('score_unique_matching_keys', 0):,}")
    lines.append(f"- score_duplicate_key_rows: {score_stats.get('score_duplicate_key_rows', 0):,}")

    lines.append("")
    lines.append("Top 20 canonical HKL groups by number removed:")
    if top_removed.empty:
        lines.append("(none)")
    else:
        lines.extend(top_removed.to_string(index=False).splitlines())

    lines.append("")
    lines.append("Top 20 canonical HKL groups by nonself spread:")
    if top_spread.empty:
        lines.append("(none)")
    else:
        lines.extend(top_spread.to_string(index=False).splitlines())

    lines.append("")
    lines.append("Requested family diagnostics:")
    if focus_table.empty:
        lines.append("(none)")
    else:
        lines.extend(focus_table.to_string(index=False).splitlines())

    lines.append("")
    lines.append("Generated files:")
    for key in [
        "filtered_stream",
        "decisions_csv",
        "removed_csv",
        "kept_csv",
        "unmatched_csv",
        "per_crystal_csv",
        "readme",
    ]:
        lines.append(f"- {outputs[key]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    hkl_filter = parse_hkl_list(args.hkl_list)
    outputs = ensure_output_layout(args.output_root)

    log("Starting non-self stream filtering workflow")
    log(f"Input stream: {args.stream}")
    log(f"Input scores: {args.scores}")
    log("Output paths:")
    for key in [
        "filtered_stream",
        "decisions_csv",
        "removed_csv",
        "kept_csv",
        "unmatched_csv",
        "per_crystal_csv",
        "readme",
    ]:
        log(f"  - {key}: {outputs[key]}")

    scan_stats = collect_stream_keys(
        stream_path=args.stream,
        hkl_filter=hkl_filter,
        max_events=args.max_events,
        max_reflections=args.max_reflections,
        progress_every=args.progress_every,
    )

    score_map, score_stats, source_column = load_scores_for_stream_keys(
        scores_path=args.scores,
        stream_keys=scan_stats["stream_keys"],
        scores_chunksize=args.scores_chunksize,
    )

    matched_df, unmatched_df, crystal_df, collect_stats = collect_reflection_observations(
        stream_path=args.stream,
        score_map=score_map,
        hkl_filter=hkl_filter,
        cutoff_line=scan_stats["cutoff_line"],
        progress_every=args.progress_every,
    )

    candidate_df, base_decisions = build_candidate_group_filter(
        matched_df=matched_df,
        remove_top_fraction=float(args.remove_top_fraction),
        min_obs_all=int(args.min_obs_all),
        min_obs_kept=int(args.min_obs_kept),
        min_nonself_spread=float(args.min_nonself_spread),
    )

    final_df, per_crystal, crystals_requiring_restoration = apply_crystal_safety(
        df=candidate_df,
        crystal_df=crystal_df,
        max_crystal_removed_fraction=float(args.max_crystal_removed_fraction),
        min_crystal_reflections_kept=int(args.min_crystal_reflections_kept),
    )

    decisions = finalize_group_decisions(final_df, base_decisions)

    removed_df = final_df.loc[final_df["remove_final"]].copy()
    kept_df = final_df.loc[~final_df["remove_final"]].copy()

    output_columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "hkl_signed",
        "h_canon",
        "k_canon",
        "l_canon",
        "hkl_canon",
        "nonself_mean",
        "graph_crowding_norm",
        "same_laue_zone_crowding_norm",
        "systematic_row_risk_norm",
        "frame_axis_risk_norm",
        "crystal_index",
        "reflection_line_number",
    ]

    unmatched_columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "hkl_signed",
        "h_canon",
        "k_canon",
        "l_canon",
        "hkl_canon",
        "crystal_index",
        "reflection_line_number",
        "reason",
    ]

    decisions.to_csv(outputs["decisions_csv"], index=False)
    removed_df.reindex(columns=output_columns).to_csv(outputs["removed_csv"], index=False)
    kept_df.reindex(columns=output_columns).to_csv(outputs["kept_csv"], index=False)
    unmatched_df.reindex(columns=unmatched_columns).to_csv(outputs["unmatched_csv"], index=False)
    per_crystal.to_csv(outputs["per_crystal_csv"], index=False)

    remove_line_numbers = set(removed_df["reflection_line_number"].astype(int).tolist()) if not removed_df.empty else set()
    write_stats = write_filtered_stream(
        stream_path=args.stream,
        output_path=outputs["filtered_stream"],
        remove_line_numbers=remove_line_numbers,
        cutoff_line=scan_stats["cutoff_line"],
        progress_every=args.progress_every,
    )

    write_readme(
        path=outputs["readme"],
        args=args,
        outputs=outputs,
        source_column=source_column,
        scan_stats=scan_stats,
        score_stats=score_stats,
        collect_stats=collect_stats,
        decisions=decisions,
        per_crystal=per_crystal,
        crystals_requiring_restoration=crystals_requiring_restoration,
    )

    log("Workflow complete")
    print("NONSELF_STREAM_FILTER_4MMM_OK", flush=True)
    print(f"chunks_read={collect_stats['chunks_read']:,}", flush=True)
    print(f"crystals_read={collect_stats['crystals_read']:,}", flush=True)
    print(f"measured_reflections_read={collect_stats['measured_reflections_read']:,}", flush=True)
    print(f"considered_reflections={collect_stats['considered_reflections']:,}", flush=True)
    print(f"matched_reflections={collect_stats['matched_reflections']:,}", flush=True)
    print(f"unmatched_reflections={collect_stats['unmatched_reflections']:,}", flush=True)
    print(f"removed_reflections={int(decisions['n_obs_removed'].sum()) if not decisions.empty else 0:,}", flush=True)
    print(f"kept_reflections={int(decisions['n_obs_kept'].sum()) if not decisions.empty else 0:,}", flush=True)
    print(f"filtered_stream={outputs['filtered_stream']}", flush=True)
    print(f"decisions_csv={outputs['decisions_csv']}", flush=True)
    print(f"removed_csv={outputs['removed_csv']}", flush=True)
    print(f"kept_csv={outputs['kept_csv']}", flush=True)
    print(f"unmatched_csv={outputs['unmatched_csv']}", flush=True)
    print(f"per_crystal_csv={outputs['per_crystal_csv']}", flush=True)
    print(f"readme={outputs['readme']}", flush=True)
    print(
        "write_stage_reflection_lines="
        f"seen:{write_stats['reflection_lines_seen']:,},"
        f"removed:{write_stats['reflection_lines_removed']:,},"
        f"kept:{write_stats['reflection_lines_kept']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
