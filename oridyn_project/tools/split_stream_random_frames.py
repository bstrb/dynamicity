#!/usr/bin/env python3
"""Randomly split complete CrystFEL stream chunks into two control streams."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

from split_stream_by_frame_risk import format_stat
from split_stream_by_frame_risk import log
from split_stream_by_frame_risk import normalize_text
from split_stream_by_frame_risk import scan_stream_chunks
from split_stream_by_frame_risk import selected_for_output
from split_stream_by_frame_risk import write_stream_file


GRAPH_DIAGNOSTIC_COLUMN = "graph_crowding_norm"
FRAME_DIAGNOSTIC_COLUMN = "frame_axis_risk_norm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly split matched CrystFEL stream chunks into A/B complete-frame "
            "control streams while preserving original chunk text."
        )
    )
    parser.add_argument("--stream", required=True, type=Path, help="Input CrystFEL .stream file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv for frame matching")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=0.5,
        help="Fraction of matched chunks assigned to split B; the remainder go to split A",
    )
    parser.add_argument("--odd-extra", choices=["A", "B"], default="A")
    parser.add_argument("--include-unmatched-in", choices=["A", "B", "both", "none"], default="none")
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--dry-run", action="store_true", help="Write diagnostics but do not write stream files")

    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if not (0.0 < float(args.split_fraction) < 1.0):
        raise SystemExit("--split-fraction must be > 0 and < 1")
    if int(args.progress_every) < 1:
        raise SystemExit("--progress-every must be >= 1")

    return args


def output_paths(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "A_stream": output_root / "random_A_frames_50.stream",
        "B_stream": output_root / "random_B_frames_50.stream",
        "frame_summary": output_root / "random_frame_split_summary.csv",
        "split_summary": output_root / "random_split_summary.txt",
        "unmatched_chunks": output_root / "unmatched_stream_chunks.csv",
    }


def load_frame_diagnostics(scores_path: Path, progress_every: int) -> pd.DataFrame:
    log("Stage 2/4: aggregating frame matching diagnostics from scores")

    header = list(pd.read_csv(scores_path, nrows=0).columns)
    required = ["source_filename", "event"]
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"Scores file is missing required columns: {missing}")

    has_graph = GRAPH_DIAGNOSTIC_COLUMN in header
    has_frame = FRAME_DIAGNOSTIC_COLUMN in header
    optional_columns = [column for column in [GRAPH_DIAGNOSTIC_COLUMN, FRAME_DIAGNOSTIC_COLUMN] if column in header]
    usecols = [*required, *optional_columns]

    chunksize = max(int(progress_every), 100000)
    rows_read = 0
    next_progress = int(progress_every)
    aggregates: dict[tuple[str, str], dict[str, float]] = {}

    for chunk in pd.read_csv(scores_path, usecols=usecols, chunksize=chunksize):
        rows_read += len(chunk)
        chunk["source_filename"] = chunk["source_filename"].map(normalize_text)
        chunk["event"] = chunk["event"].map(normalize_text)
        chunk = chunk[chunk["source_filename"] != ""].copy()

        if has_graph:
            chunk[GRAPH_DIAGNOSTIC_COLUMN] = pd.to_numeric(chunk[GRAPH_DIAGNOSTIC_COLUMN], errors="coerce")
            graph_mask = np.isfinite(chunk[GRAPH_DIAGNOSTIC_COLUMN].to_numpy(dtype=float))
            chunk["_graph_sum"] = chunk[GRAPH_DIAGNOSTIC_COLUMN].where(graph_mask, 0.0)
            chunk["_graph_n"] = graph_mask.astype(int)
        else:
            chunk["_graph_sum"] = 0.0
            chunk["_graph_n"] = 0

        if has_frame:
            chunk[FRAME_DIAGNOSTIC_COLUMN] = pd.to_numeric(chunk[FRAME_DIAGNOSTIC_COLUMN], errors="coerce")
            frame_mask = np.isfinite(chunk[FRAME_DIAGNOSTIC_COLUMN].to_numpy(dtype=float))
            chunk["_frame_sum"] = chunk[FRAME_DIAGNOSTIC_COLUMN].where(frame_mask, 0.0)
            chunk["_frame_n"] = frame_mask.astype(int)
        else:
            chunk["_frame_sum"] = 0.0
            chunk["_frame_n"] = 0

        grouped = (
            chunk.groupby(["source_filename", "event"], dropna=False)
            .agg(
                n_score_rows=("source_filename", "size"),
                graph_sum=("_graph_sum", "sum"),
                graph_n=("_graph_n", "sum"),
                frame_sum=("_frame_sum", "sum"),
                frame_n=("_frame_n", "sum"),
            )
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            key = (str(row.source_filename), str(row.event))
            acc = aggregates.setdefault(
                key,
                {
                    "n": 0.0,
                    "graph_sum": 0.0,
                    "graph_n": 0.0,
                    "frame_sum": 0.0,
                    "frame_n": 0.0,
                },
            )
            acc["n"] += float(row.n_score_rows)
            acc["graph_sum"] += float(row.graph_sum)
            acc["graph_n"] += float(row.graph_n)
            acc["frame_sum"] += float(row.frame_sum)
            acc["frame_n"] += float(row.frame_n)

        while rows_read >= next_progress:
            log(f"Stage 2 progress: score_rows_read={rows_read:,}, frame_keys={len(aggregates):,}")
            next_progress += int(progress_every)

    records: list[dict[str, Any]] = []
    for (source, event), acc in aggregates.items():
        graph_mean = acc["graph_sum"] / acc["graph_n"] if acc["graph_n"] > 0 else np.nan
        frame_mean = acc["frame_sum"] / acc["frame_n"] if acc["frame_n"] > 0 else np.nan
        records.append(
            {
                "source_filename": source,
                "event": event,
                "n_score_rows": int(acc["n"]),
                "mean_graph_crowding_norm": graph_mean,
                "mean_frame_axis_risk_norm": frame_mean,
            }
        )

    frame_diagnostics = pd.DataFrame.from_records(
        records,
        columns=[
            "source_filename",
            "event",
            "n_score_rows",
            "mean_graph_crowding_norm",
            "mean_frame_axis_risk_norm",
        ],
    )
    log(
        "Stage 2 complete: "
        f"frame_keys={len(frame_diagnostics):,}, "
        f"graph_diagnostic_available={has_graph}, frame_diagnostic_available={has_frame}"
    )
    return frame_diagnostics


def compute_b_count(n_matched: int, split_fraction: float, odd_extra: str) -> int:
    if n_matched <= 0:
        return 0
    if abs(float(split_fraction) - 0.5) < 1e-12 and n_matched % 2 == 1:
        return (n_matched // 2) + (1 if odd_extra == "B" else 0)
    return max(0, min(n_matched, int(math.floor(float(split_fraction) * float(n_matched)))))


def assign_random_splits(
    chunks: list[Any],
    frame_diagnostics: pd.DataFrame,
    seed: int,
    split_fraction: float,
    odd_extra: str,
    include_unmatched_in: str,
    dry_run: bool,
) -> tuple[pd.DataFrame, set[int], set[int], set[int], set[int], set[int]]:
    log("Stage 3/4: matching chunks and assigning random A/B splits")

    diagnostics_by_frame: dict[tuple[str, str], dict[str, Any]] = {}
    for row in frame_diagnostics.itertuples(index=False):
        diagnostics_by_frame[(str(row.source_filename), str(row.event))] = {
            "n_score_rows": int(row.n_score_rows),
            "mean_graph_crowding_norm": float(row.mean_graph_crowding_norm)
            if pd.notna(row.mean_graph_crowding_norm)
            else np.nan,
            "mean_frame_axis_risk_norm": float(row.mean_frame_axis_risk_norm)
            if pd.notna(row.mean_frame_axis_risk_norm)
            else np.nan,
        }

    matched_indices: list[int] = []
    unmatched_indices: set[int] = set()
    chunk_values: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        values = diagnostics_by_frame.get((chunk.source_filename, chunk.event))
        if values is None:
            unmatched_indices.add(int(chunk.chunk_index))
            continue
        matched_indices.append(int(chunk.chunk_index))
        chunk_values[int(chunk.chunk_index)] = values

    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(np.array(matched_indices, dtype=int)) if matched_indices else np.array([], dtype=int)
    b_count = compute_b_count(len(matched_indices), split_fraction, odd_extra)
    a_count = len(matched_indices) - b_count
    a_indices = {int(idx) for idx in shuffled[:a_count]}
    b_indices = {int(idx) for idx in shuffled[a_count:]}
    random_order_by_chunk = {int(idx): order for order, idx in enumerate(shuffled.tolist(), start=1)}

    a_output_indices = selected_for_output(a_indices, unmatched_indices, include_unmatched_in, "A")
    b_output_indices = selected_for_output(b_indices, unmatched_indices, include_unmatched_in, "B")

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        idx = int(chunk.chunk_index)
        values = chunk_values.get(idx)
        if values is None:
            assigned_split = "unmatched"
            n_score_rows = 0
            graph_mean = np.nan
            frame_mean = np.nan
            random_order = np.nan
        else:
            assigned_split = "A" if idx in a_indices else "B"
            n_score_rows = int(values["n_score_rows"])
            graph_mean = values["mean_graph_crowding_norm"]
            frame_mean = values["mean_frame_axis_risk_norm"]
            random_order = int(random_order_by_chunk[idx])

        if idx in a_output_indices and idx in b_output_indices:
            included_in_output = "both"
        elif idx in a_output_indices:
            included_in_output = "A"
        elif idx in b_output_indices:
            included_in_output = "B"
        else:
            included_in_output = "none"

        rows.append(
            {
                "chunk_index": idx,
                "source_filename": chunk.source_filename,
                "event": chunk.event,
                "matched": values is not None,
                "n_score_rows": n_score_rows,
                "mean_graph_crowding_norm": graph_mean,
                "mean_frame_axis_risk_norm": frame_mean,
                "random_order": random_order,
                "assigned_split": assigned_split,
                "included_in_output": included_in_output,
                "seed": int(seed),
                "dry_run": bool(dry_run),
            }
        )

    summary = pd.DataFrame.from_records(
        rows,
        columns=[
            "chunk_index",
            "source_filename",
            "event",
            "matched",
            "n_score_rows",
            "mean_graph_crowding_norm",
            "mean_frame_axis_risk_norm",
            "random_order",
            "assigned_split",
            "included_in_output",
            "seed",
            "dry_run",
        ],
    )

    log(
        "Stage 3 complete: "
        f"matched_chunks={len(matched_indices):,}, A={len(a_indices):,}, "
        f"B={len(b_indices):,}, unmatched={len(unmatched_indices):,}"
    )
    return summary, a_indices, b_indices, unmatched_indices, a_output_indices, b_output_indices


def diagnostic_stats(frame_summary: pd.DataFrame, split: str, column: str) -> dict[str, float | None]:
    values = pd.to_numeric(frame_summary.loc[frame_summary["assigned_split"] == split, column], errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def write_random_split_summary(
    path: Path,
    args: argparse.Namespace,
    outputs: dict[str, Path],
    total_chunks: int,
    a_indices: set[int],
    b_indices: set[int],
    unmatched_indices: set[int],
    a_output_indices: set[int],
    b_output_indices: set[int],
    frame_summary: pd.DataFrame,
) -> None:
    graph_a = diagnostic_stats(frame_summary, "A", "mean_graph_crowding_norm")
    graph_b = diagnostic_stats(frame_summary, "B", "mean_graph_crowding_norm")
    frame_a = diagnostic_stats(frame_summary, "A", "mean_frame_axis_risk_norm")
    frame_b = diagnostic_stats(frame_summary, "B", "mean_frame_axis_risk_norm")

    lines = [
        "OriDyn random frame split summary",
        "",
        f"input stream path: {args.stream}",
        f"scores path: {args.scores}",
        f"random A output stream path: {outputs['A_stream']}",
        f"random B output stream path: {outputs['B_stream']}",
        "",
        f"total stream chunks: {total_chunks}",
        f"matched chunks: {len(a_indices) + len(b_indices)}",
        f"unmatched chunks: {len(unmatched_indices)}",
        f"A chunk count: {len(a_indices)}",
        f"B chunk count: {len(b_indices)}",
        f"A output chunk count: {len(a_output_indices)}",
        f"B output chunk count: {len(b_output_indices)}",
        "",
        f"seed: {int(args.seed)}",
        f"split fraction assigned to B: {float(args.split_fraction)}",
        f"odd extra: {args.odd_extra}",
        f"include unmatched in: {args.include_unmatched_in}",
        f"unmatched chunks included anywhere: {args.include_unmatched_in != 'none'}",
        f"dry run: {bool(args.dry_run)}",
        "",
        "A graph_crowding_norm diagnostic stats:",
        f"  mean: {format_stat(graph_a['mean'])}",
        f"  median: {format_stat(graph_a['median'])}",
        f"  min: {format_stat(graph_a['min'])}",
        f"  max: {format_stat(graph_a['max'])}",
        "",
        "B graph_crowding_norm diagnostic stats:",
        f"  mean: {format_stat(graph_b['mean'])}",
        f"  median: {format_stat(graph_b['median'])}",
        f"  min: {format_stat(graph_b['min'])}",
        f"  max: {format_stat(graph_b['max'])}",
        "",
        "A frame_axis_risk_norm diagnostic stats:",
        f"  mean: {format_stat(frame_a['mean'])}",
        f"  median: {format_stat(frame_a['median'])}",
        f"  min: {format_stat(frame_a['min'])}",
        f"  max: {format_stat(frame_a['max'])}",
        "",
        "B frame_axis_risk_norm diagnostic stats:",
        f"  mean: {format_stat(frame_b['mean'])}",
        f"  median: {format_stat(frame_b['median'])}",
        f"  min: {format_stat(frame_b['min'])}",
        f"  max: {format_stat(frame_b['max'])}",
        "",
        "Assignment note: diagnostic columns are summarized only; random assignment uses only the seed and matched chunks.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs = output_paths(args.output_root)

    log("Starting random frame control split")
    log(f"Output root: {args.output_root}")
    log(f"Random A stream: {outputs['A_stream']}")
    log(f"Random B stream: {outputs['B_stream']}")
    log(f"Frame summary CSV: {outputs['frame_summary']}")
    log(f"Split summary TXT: {outputs['split_summary']}")
    log(f"Unmatched chunks CSV: {outputs['unmatched_chunks']}")

    header, chunks = scan_stream_chunks(args.stream, progress_every=int(args.progress_every))
    frame_diagnostics = load_frame_diagnostics(args.scores, progress_every=int(args.progress_every))
    (
        frame_summary,
        a_indices,
        b_indices,
        unmatched_indices,
        a_output_indices,
        b_output_indices,
    ) = assign_random_splits(
        chunks,
        frame_diagnostics,
        seed=int(args.seed),
        split_fraction=float(args.split_fraction),
        odd_extra=str(args.odd_extra),
        include_unmatched_in=str(args.include_unmatched_in),
        dry_run=bool(args.dry_run),
    )

    unmatched_df = frame_summary.loc[
        frame_summary["assigned_split"] == "unmatched",
        ["chunk_index", "source_filename", "event"],
    ].copy()
    frame_summary.to_csv(outputs["frame_summary"], index=False)
    unmatched_df.to_csv(outputs["unmatched_chunks"], index=False)

    write_random_split_summary(
        outputs["split_summary"],
        args,
        outputs,
        total_chunks=len(chunks),
        a_indices=a_indices,
        b_indices=b_indices,
        unmatched_indices=unmatched_indices,
        a_output_indices=a_output_indices,
        b_output_indices=b_output_indices,
        frame_summary=frame_summary,
    )

    if bool(args.dry_run):
        log("Dry run requested; stream files were not written")
    else:
        log("Stage 4/4: writing selected complete stream chunks")
        write_stream_file(args.stream, outputs["A_stream"], header, chunks, a_output_indices)
        write_stream_file(args.stream, outputs["B_stream"], header, chunks, b_output_indices)
        log("Stage 4 complete: stream files written")

    log("Diagnostics written")


if __name__ == "__main__":
    main()
