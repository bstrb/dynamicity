#!/usr/bin/env python3
"""Split complete CrystFEL stream chunks by frame-level OriDyn risk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


BEGIN_CHUNK = b"----- Begin chunk -----"
END_CHUNK = b"----- End chunk -----"
IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.*?)\s*$")
EVENT_RE = re.compile(r"^\s*Event:\s*(.*?)\s*$")


@dataclass(frozen=True)
class StreamChunk:
    chunk_index: int
    source_filename: str
    event: str
    start_offset: int
    end_offset: int


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CrystFEL stream into low/high frame-risk stream files while preserving "
            "complete original chunks."
        )
    )
    parser.add_argument("--stream", required=True, type=Path, help="Input CrystFEL .stream file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder")
    parser.add_argument("--progress-every", type=int, default=100000)

    parser.add_argument("--graph-column", default="graph_crowding_norm")
    parser.add_argument("--frame-column", default="frame_axis_risk_norm")
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--frame-weight", type=float, default=0.5)
    parser.add_argument("--high-fraction", type=float, default=0.5)
    parser.add_argument("--odd-extra", choices=["low", "high"], default="low")
    parser.add_argument(
        "--include-unmatched-in",
        choices=["low", "high", "both", "none"],
        default="none",
        help="Where to copy chunks without matching frame risk (default: none)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write diagnostics but do not write stream files")

    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if int(args.progress_every) < 1:
        raise SystemExit("--progress-every must be >= 1")
    if not math.isfinite(float(args.graph_weight)):
        raise SystemExit("--graph-weight must be finite")
    if not math.isfinite(float(args.frame_weight)):
        raise SystemExit("--frame-weight must be finite")
    if not (0.0 < float(args.high_fraction) < 1.0):
        raise SystemExit("--high-fraction must be > 0 and < 1")

    return args


def output_paths(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "low_stream": output_root / "low_risk_frames_50.stream",
        "high_stream": output_root / "high_risk_frames_50.stream",
        "frame_summary": output_root / "frame_risk_summary.csv",
        "split_summary": output_root / "split_summary.txt",
        "unmatched_chunks": output_root / "unmatched_stream_chunks.csv",
    }


def is_marker(raw_line: bytes, marker: bytes) -> bool:
    return raw_line.rstrip(b"\r\n") == marker


def scan_stream_chunks(stream_path: Path, progress_every: int) -> tuple[bytes, list[StreamChunk]]:
    log("Stage 1/4: scanning stream chunks")

    header = bytearray()
    chunks: list[StreamChunk] = []
    in_chunk = False
    chunk_index = -1
    start_offset = 0
    current_source = ""
    current_event = ""
    seen_first_chunk = False

    with stream_path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if raw_line == b"":
                break

            if not in_chunk:
                if is_marker(raw_line, BEGIN_CHUNK):
                    seen_first_chunk = True
                    in_chunk = True
                    chunk_index += 1
                    start_offset = line_start
                    current_source = ""
                    current_event = ""
                elif not seen_first_chunk:
                    header.extend(raw_line)
                continue

            text_line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            image_match = IMAGE_RE.match(text_line)
            if image_match:
                current_source = normalize_text(image_match.group(1))
                continue

            event_match = EVENT_RE.match(text_line)
            if event_match:
                current_event = normalize_text(event_match.group(1))
                continue

            if is_marker(raw_line, END_CHUNK):
                chunks.append(
                    StreamChunk(
                        chunk_index=chunk_index,
                        source_filename=current_source,
                        event=current_event,
                        start_offset=start_offset,
                        end_offset=handle.tell(),
                    )
                )
                in_chunk = False
                if len(chunks) % int(progress_every) == 0:
                    log(f"Stage 1 progress: chunks={len(chunks):,}")

    if in_chunk:
        raise SystemExit(f"Stream ended before closing chunk index {chunk_index}")

    log(f"Stage 1 complete: chunks={len(chunks):,}")
    return bytes(header), chunks


def load_frame_risks(
    scores_path: Path,
    graph_column: str,
    frame_column: str,
    graph_weight: float,
    frame_weight: float,
    progress_every: int,
) -> pd.DataFrame:
    log("Stage 2/4: aggregating frame risk from scores")

    header = list(pd.read_csv(scores_path, nrows=0).columns)
    required = ["source_filename", "event", graph_column, frame_column]
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"Scores file is missing required columns: {missing}")

    usecols = list(dict.fromkeys(required))
    chunksize = max(int(progress_every), 100000)
    rows_read = 0
    valid_rows = 0
    next_progress = int(progress_every)
    aggregates: dict[tuple[str, str], dict[str, float]] = {}

    for chunk in pd.read_csv(scores_path, usecols=usecols, chunksize=chunksize):
        rows_read += len(chunk)
        chunk["source_filename"] = chunk["source_filename"].map(normalize_text)
        chunk["event"] = chunk["event"].map(normalize_text)
        chunk[graph_column] = pd.to_numeric(chunk[graph_column], errors="coerce")
        chunk[frame_column] = pd.to_numeric(chunk[frame_column], errors="coerce")
        chunk = chunk[chunk["source_filename"] != ""].copy()
        if chunk.empty:
            while rows_read >= next_progress:
                log(f"Stage 2 progress: score_rows_read={rows_read:,}, valid_risk_rows={valid_rows:,}")
                next_progress += int(progress_every)
            continue

        risk = float(graph_weight) * chunk[graph_column] + float(frame_weight) * chunk[frame_column]
        finite_mask = np.isfinite(risk.to_numpy(dtype=float))
        chunk = chunk.loc[finite_mask].copy()
        if chunk.empty:
            while rows_read >= next_progress:
                log(f"Stage 2 progress: score_rows_read={rows_read:,}, valid_risk_rows={valid_rows:,}")
                next_progress += int(progress_every)
            continue

        chunk["_frame_split_risk"] = risk.loc[chunk.index].to_numpy(dtype=float)
        valid_rows += len(chunk)
        grouped = (
            chunk.groupby(["source_filename", "event"], dropna=False)
            .agg(
                n_score_rows=("_frame_split_risk", "size"),
                graph_sum=(graph_column, "sum"),
                frame_sum=(frame_column, "sum"),
                risk_sum=("_frame_split_risk", "sum"),
            )
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            key = (str(row.source_filename), str(row.event))
            acc = aggregates.setdefault(key, {"n": 0.0, "graph_sum": 0.0, "frame_sum": 0.0, "risk_sum": 0.0})
            acc["n"] += float(row.n_score_rows)
            acc["graph_sum"] += float(row.graph_sum)
            acc["frame_sum"] += float(row.frame_sum)
            acc["risk_sum"] += float(row.risk_sum)

        while rows_read >= next_progress:
            log(f"Stage 2 progress: score_rows_read={rows_read:,}, valid_risk_rows={valid_rows:,}")
            next_progress += int(progress_every)

    records: list[dict[str, Any]] = []
    for (source, event), acc in aggregates.items():
        n = int(acc["n"])
        if n < 1:
            continue
        records.append(
            {
                "source_filename": source,
                "event": event,
                "n_score_rows": n,
                "mean_graph_crowding_norm": acc["graph_sum"] / n,
                "mean_frame_axis_risk_norm": acc["frame_sum"] / n,
                "mean_frame_split_risk": acc["risk_sum"] / n,
            }
        )

    frame_risks = pd.DataFrame.from_records(
        records,
        columns=[
            "source_filename",
            "event",
            "n_score_rows",
            "mean_graph_crowding_norm",
            "mean_frame_axis_risk_norm",
            "mean_frame_split_risk",
        ],
    )
    log(f"Stage 2 complete: frame_risk_rows={len(frame_risks):,}, valid_risk_rows={valid_rows:,}")
    return frame_risks


def compute_high_count(n_matched: int, high_fraction: float, odd_extra: str) -> int:
    if n_matched <= 0:
        return 0
    if abs(float(high_fraction) - 0.5) < 1e-12 and n_matched % 2 == 1:
        return (n_matched // 2) + (1 if odd_extra == "high" else 0)
    return max(0, min(n_matched, int(math.floor(float(high_fraction) * float(n_matched)))))


def assign_splits(
    chunks: list[StreamChunk],
    frame_risks: pd.DataFrame,
    high_fraction: float,
    odd_extra: str,
) -> tuple[pd.DataFrame, set[int], set[int], set[int]]:
    log("Stage 3/4: matching stream chunks and assigning splits")

    risk_by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame_risks.itertuples(index=False):
        risk_by_frame[(str(row.source_filename), str(row.event))] = {
            "n_score_rows": int(row.n_score_rows),
            "mean_graph_crowding_norm": float(row.mean_graph_crowding_norm),
            "mean_frame_axis_risk_norm": float(row.mean_frame_axis_risk_norm),
            "mean_frame_split_risk": float(row.mean_frame_split_risk),
        }

    matched: list[tuple[float, int]] = []
    chunk_risk: dict[int, dict[str, Any]] = {}
    unmatched_indices: set[int] = set()

    for chunk in chunks:
        values = risk_by_frame.get((chunk.source_filename, chunk.event))
        if values is None:
            unmatched_indices.add(chunk.chunk_index)
            continue
        chunk_risk[chunk.chunk_index] = values
        matched.append((float(values["mean_frame_split_risk"]), chunk.chunk_index))

    matched_sorted = sorted(matched, key=lambda item: (item[0], item[1]))
    high_n = compute_high_count(len(matched_sorted), high_fraction, odd_extra)
    low_n = len(matched_sorted) - high_n
    low_indices = {idx for _, idx in matched_sorted[:low_n]}
    high_indices = {idx for _, idx in matched_sorted[low_n:]}
    rank_by_chunk = {idx: rank for rank, (_, idx) in enumerate(matched_sorted, start=1)}

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        values = chunk_risk.get(chunk.chunk_index)
        if values is None:
            rows.append(
                {
                    "source_filename": chunk.source_filename,
                    "event": chunk.event,
                    "chunk_index": int(chunk.chunk_index),
                    "n_score_rows": 0,
                    "mean_graph_crowding_norm": np.nan,
                    "mean_frame_axis_risk_norm": np.nan,
                    "mean_frame_split_risk": np.nan,
                    "risk_rank_low_to_high": np.nan,
                    "assigned_split": "unmatched",
                }
            )
            continue

        assigned = "low" if chunk.chunk_index in low_indices else "high"
        rows.append(
            {
                "source_filename": chunk.source_filename,
                "event": chunk.event,
                "chunk_index": int(chunk.chunk_index),
                "n_score_rows": int(values["n_score_rows"]),
                "mean_graph_crowding_norm": float(values["mean_graph_crowding_norm"]),
                "mean_frame_axis_risk_norm": float(values["mean_frame_axis_risk_norm"]),
                "mean_frame_split_risk": float(values["mean_frame_split_risk"]),
                "risk_rank_low_to_high": int(rank_by_chunk[chunk.chunk_index]),
                "assigned_split": assigned,
            }
        )

    summary = pd.DataFrame.from_records(
        rows,
        columns=[
            "source_filename",
            "event",
            "chunk_index",
            "n_score_rows",
            "mean_graph_crowding_norm",
            "mean_frame_axis_risk_norm",
            "mean_frame_split_risk",
            "risk_rank_low_to_high",
            "assigned_split",
        ],
    )
    log(
        "Stage 3 complete: "
        f"matched_chunks={len(matched_sorted):,}, low={len(low_indices):,}, "
        f"high={len(high_indices):,}, unmatched={len(unmatched_indices):,}"
    )
    return summary, low_indices, high_indices, unmatched_indices


def copy_byte_range(source_handle: Any, dest_handle: Any, start: int, end: int, buffer_size: int = 1024 * 1024) -> None:
    source_handle.seek(int(start))
    remaining = int(end) - int(start)
    while remaining > 0:
        data = source_handle.read(min(buffer_size, remaining))
        if data == b"":
            raise SystemExit("Unexpected EOF while copying stream chunk")
        dest_handle.write(data)
        remaining -= len(data)


def selected_for_output(
    base_indices: set[int],
    unmatched_indices: set[int],
    include_unmatched_in: str,
    split: str,
) -> set[int]:
    selected = set(base_indices)
    if include_unmatched_in == split or include_unmatched_in == "both":
        selected.update(unmatched_indices)
    return selected


def write_stream_file(
    stream_path: Path,
    output_path: Path,
    header: bytes,
    chunks: list[StreamChunk],
    selected_indices: set[int],
) -> None:
    by_index = {chunk.chunk_index: chunk for chunk in chunks}
    ordered_indices = sorted(selected_indices)
    with stream_path.open("rb") as source_handle, output_path.open("wb") as dest_handle:
        dest_handle.write(header)
        for idx in ordered_indices:
            chunk = by_index[idx]
            copy_byte_range(source_handle, dest_handle, chunk.start_offset, chunk.end_offset)


def risk_stats(frame_summary: pd.DataFrame, split: str) -> dict[str, float | None]:
    values = pd.to_numeric(
        frame_summary.loc[frame_summary["assigned_split"] == split, "mean_frame_split_risk"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def format_stat(value: float | None) -> str:
    return "NA" if value is None else f"{value:.8g}"


def write_split_summary(
    path: Path,
    args: argparse.Namespace,
    outputs: dict[str, Path],
    total_chunks: int,
    low_indices: set[int],
    high_indices: set[int],
    unmatched_indices: set[int],
    low_output_indices: set[int],
    high_output_indices: set[int],
    frame_summary: pd.DataFrame,
) -> None:
    low_stats = risk_stats(frame_summary, "low")
    high_stats = risk_stats(frame_summary, "high")
    lines = [
        "OriDyn frame-risk stream split summary",
        "",
        f"input stream path: {args.stream}",
        f"scores path: {args.scores}",
        f"low-risk output stream path: {outputs['low_stream']}",
        f"high-risk output stream path: {outputs['high_stream']}",
        "",
        f"total stream chunks: {total_chunks}",
        f"matched chunks: {len(low_indices) + len(high_indices)}",
        f"unmatched chunks: {len(unmatched_indices)}",
        f"low-risk chunk count: {len(low_indices)}",
        f"high-risk chunk count: {len(high_indices)}",
        f"low output chunk count: {len(low_output_indices)}",
        f"high output chunk count: {len(high_output_indices)}",
        "",
        "low-risk split risk stats:",
        f"  mean: {format_stat(low_stats['mean'])}",
        f"  median: {format_stat(low_stats['median'])}",
        f"  min: {format_stat(low_stats['min'])}",
        f"  max: {format_stat(low_stats['max'])}",
        "",
        "high-risk split risk stats:",
        f"  mean: {format_stat(high_stats['mean'])}",
        f"  median: {format_stat(high_stats['median'])}",
        f"  min: {format_stat(high_stats['min'])}",
        f"  max: {format_stat(high_stats['max'])}",
        "",
        f"graph column: {args.graph_column}",
        f"frame column: {args.frame_column}",
        f"graph weight: {float(args.graph_weight)}",
        f"frame weight: {float(args.frame_weight)}",
        f"high fraction: {float(args.high_fraction)}",
        f"odd extra: {args.odd_extra}",
        f"include unmatched in: {args.include_unmatched_in}",
        f"unmatched chunks included anywhere: {args.include_unmatched_in != 'none'}",
        f"dry run: {bool(args.dry_run)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs = output_paths(args.output_root)

    log("Starting frame-risk stream split")
    log(f"Output root: {args.output_root}")
    log(f"Low-risk stream: {outputs['low_stream']}")
    log(f"High-risk stream: {outputs['high_stream']}")
    log(f"Frame summary CSV: {outputs['frame_summary']}")
    log(f"Split summary TXT: {outputs['split_summary']}")
    log(f"Unmatched chunks CSV: {outputs['unmatched_chunks']}")

    header, chunks = scan_stream_chunks(args.stream, progress_every=int(args.progress_every))
    frame_risks = load_frame_risks(
        args.scores,
        graph_column=str(args.graph_column),
        frame_column=str(args.frame_column),
        graph_weight=float(args.graph_weight),
        frame_weight=float(args.frame_weight),
        progress_every=int(args.progress_every),
    )
    frame_summary, low_indices, high_indices, unmatched_indices = assign_splits(
        chunks,
        frame_risks,
        high_fraction=float(args.high_fraction),
        odd_extra=str(args.odd_extra),
    )

    unmatched_df = frame_summary.loc[
        frame_summary["assigned_split"] == "unmatched",
        ["chunk_index", "source_filename", "event"],
    ].copy()
    frame_summary.to_csv(outputs["frame_summary"], index=False)
    unmatched_df.to_csv(outputs["unmatched_chunks"], index=False)

    low_output_indices = selected_for_output(low_indices, unmatched_indices, str(args.include_unmatched_in), "low")
    high_output_indices = selected_for_output(high_indices, unmatched_indices, str(args.include_unmatched_in), "high")

    write_split_summary(
        outputs["split_summary"],
        args,
        outputs,
        total_chunks=len(chunks),
        low_indices=low_indices,
        high_indices=high_indices,
        unmatched_indices=unmatched_indices,
        low_output_indices=low_output_indices,
        high_output_indices=high_output_indices,
        frame_summary=frame_summary,
    )

    if bool(args.dry_run):
        log("Dry run requested; stream files were not written")
    else:
        log("Stage 4/4: writing selected complete stream chunks")
        write_stream_file(args.stream, outputs["low_stream"], header, chunks, low_output_indices)
        write_stream_file(args.stream, outputs["high_stream"], header, chunks, high_output_indices)
        log("Stage 4 complete: stream files written")

    log("Diagnostics written")


if __name__ == "__main__":
    main()
