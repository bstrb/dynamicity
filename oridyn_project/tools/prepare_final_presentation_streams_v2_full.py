#!/usr/bin/env python3
"""Prepare final presentation v2_full CrystFEL stream filtering set.

This is a lightweight stream-preparation driver only. It does not run merging,
partialator, refinement, QC, or SHELXL.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, TextIO

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from filter_stream_by_geometry_coupling_v2_keep_fraction import (  # noqa: E402
    DEFAULT_SCORE_COLUMN,
    HKL_COLUMNS,
    KEY_COLUMNS,
    SCORE_CHUNKSIZE,
    STREAM_EVENT_RE,
    STREAM_IMAGE_RE,
    build_key,
    normalize_event,
    normalize_score_chunk,
    normalize_source,
    parse_reflection_hkl,
    require_columns,
)


DATA_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_STREAM = DATA_ROOT / "MFM300-VIII_cut_20-0_3.stream"
DEFAULT_BASE_REFLECTION_SCORES = DATA_ROOT / "oridyn_full_cut_20-0_3/reflection_scores.csv"
DEFAULT_V2_SCORES = (
    DATA_ROOT
    / "geometry_coupling_v2_full_filter_keep90_80_20260623"
    / "v2_scores"
    / "geometry_coupling_v2_scores.csv"
)
DEFAULT_OUTPUT_ROOT = (
    DATA_ROOT
    / "final_presentation_processing_2026_07_03"
    / "stream_filtering_v2_full"
)
DEFAULT_PROGRESS_EVERY = 1_000_000


@dataclass(frozen=True)
class StreamSpec:
    name: str
    mode: str
    keep_fraction: float
    output_name: str
    random_seed: int | None = None


FILTER_SPECS = [
    StreamSpec("random_50", "random_50", 0.50, "random_50.stream", random_seed=1),
    StreamSpec("low_10", "low_fraction", 0.10, "low_10.stream"),
    StreamSpec("low_20", "low_fraction", 0.20, "low_20.stream"),
    StreamSpec("low_30", "low_fraction", 0.30, "low_30.stream"),
    StreamSpec("low_40", "low_fraction", 0.40, "low_40.stream"),
    StreamSpec("low_50", "low_fraction", 0.50, "low_50.stream"),
    StreamSpec("high_50", "high_fraction", 0.50, "high_50.stream"),
    StreamSpec("low_60", "low_fraction", 0.60, "low_60.stream"),
    StreamSpec("low_70", "low_fraction", 0.70, "low_70.stream"),
    StreamSpec("low_80", "low_fraction", 0.80, "low_80.stream"),
    StreamSpec("low_90", "low_fraction", 0.90, "low_90.stream"),
]


SUMMARY_COLUMNS = [
    "dataset_name",
    "output_stream_path",
    "input_stream_path",
    "score_file_path",
    "filtering_mode",
    "keep_fraction",
    "random_seed",
    "number_of_chunks_read",
    "number_of_chunks_written",
    "number_of_reflections_read",
    "number_of_reflections_written",
    "fraction_of_reflections_retained",
    "number_of_unique_signed_HKLs_before_filtering",
    "number_of_unique_signed_HKLs_after_filtering",
    "number_of_observations_successfully_matched_to_v2_full_scores",
    "number_of_observations_missing_scores",
    "missing_score_observations_were",
    "command_used_to_generate_stream",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM, help="Original full CrystFEL stream")
    parser.add_argument(
        "--v2-scores-csv",
        type=Path,
        default=DEFAULT_V2_SCORES,
        help="geometry_coupling_v2_scores.csv containing trust_risk_v2_full_norm",
    )
    parser.add_argument(
        "--base-reflection-scores-csv",
        type=Path,
        default=DEFAULT_BASE_REFLECTION_SCORES,
        help="Base reflection_scores.csv used to document v2 score regeneration",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument(
        "--missing-score-policy",
        choices=["keep", "remove"],
        default="keep",
        help="How to handle stream observations that are absent from the v2 score table",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final streams and audit files in --output-root",
    )
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.v2_scores_csv.exists():
        raise SystemExit(f"--v2-scores-csv not found: {args.v2_scores_csv}")
    if not args.base_reflection_scores_csv.exists():
        raise SystemExit(f"--base-reflection-scores-csv not found: {args.base_reflection_scores_csv}")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    return args


def log(message: str, handle: TextIO | None = None) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if handle is not None:
        handle.write(line + "\n")
        handle.flush()


def command_text(args: argparse.Namespace) -> str:
    parts = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stream",
        str(args.stream),
        "--v2-scores-csv",
        str(args.v2_scores_csv),
        "--base-reflection-scores-csv",
        str(args.base_reflection_scores_csv),
        "--output-root",
        str(args.output_root),
        "--score-column",
        str(args.score_column),
        "--random-seed",
        str(int(args.random_seed)),
        "--progress-every",
        str(int(args.progress_every)),
        "--missing-score-policy",
        str(args.missing_score_policy),
        "--overwrite",
    ]
    return shlex.join(parts)


def v2_generation_command(args: argparse.Namespace) -> str:
    parts = [
        sys.executable,
        str(PROJECT_ROOT / "tools/compute_geometry_coupling_v2_scores.py"),
        "--scores-csv",
        str(args.base_reflection_scores_csv),
        "--outdir",
        str(args.v2_scores_csv.parent),
        "--workers",
        "0",
        "--progress-every-frames",
        "1000000",
        "--overwrite",
    ]
    return shlex.join(parts)


def expected_output_paths(output_root: Path) -> list[Path]:
    paths = [output_root / "full.stream"]
    paths.extend(output_root / spec.output_name for spec in FILTER_SPECS)
    paths.extend(
        [
            output_root / "filtering_summary.csv",
            output_root / "filtering_manifest.json",
            output_root / "filtering_commands.sh",
            output_root / "filtering_audit.log",
        ]
    )
    return paths


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    blocked = [path for path in expected_output_paths(output_root) if path.exists() or path.is_symlink()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}\nUse --overwrite if intended.")
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in blocked:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def csv_has_column(path: Path, column: str) -> bool:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return column in header


def link_or_copy_full_stream(source: Path, output: Path, audit: TextIO) -> str:
    try:
        output.symlink_to(source)
        log(f"Created full.stream symlink: {output} -> {source}", audit)
        return "symlink"
    except OSError as exc:
        log(f"Symlink failed ({exc}); copying full stream instead", audit)
        shutil.copy2(source, output)
        log(f"Copied full stream: {output}", audit)
        return "copy"


def load_score_table(scores_path: Path, score_column: str, audit: TextIO) -> tuple[pd.DataFrame, dict[str, int]]:
    header = pd.read_csv(scores_path, nrows=0).columns.tolist()
    require_columns(header, [*KEY_COLUMNS, score_column], "v2 scores CSV")
    usecols = [*KEY_COLUMNS, score_column]
    chunks: list[pd.DataFrame] = []
    stats = {
        "score_rows_read": 0,
        "score_rows_after_cleanup": 0,
        "duplicate_score_key_rows": 0,
        "duplicate_score_keys": 0,
        "score_rows_after_deduplication": 0,
    }

    log(f"Loading v2 scores from {scores_path}", audit)
    for idx, chunk in enumerate(pd.read_csv(scores_path, usecols=usecols, chunksize=SCORE_CHUNKSIZE), start=1):
        stats["score_rows_read"] += int(len(chunk))
        clean = normalize_score_chunk(chunk, score_column)
        stats["score_rows_after_cleanup"] += int(len(clean))
        if not clean.empty:
            chunks.append(clean)
        if idx % 5 == 0:
            log(
                "Score load progress: "
                f"rows_read={stats['score_rows_read']:,}, "
                f"rows_after_cleanup={stats['score_rows_after_cleanup']:,}",
                audit,
            )

    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if table.empty:
        raise SystemExit("No usable v2 score rows were loaded")

    duplicated = table.duplicated(KEY_COLUMNS, keep=False)
    stats["duplicate_score_key_rows"] = int(duplicated.sum())
    stats["duplicate_score_keys"] = (
        int(table.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicated.any() else 0
    )
    if duplicated.any():
        log(
            "Warning: duplicate exact observation keys in v2 scores; keeping the first row per key",
            audit,
        )
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    stats["score_rows_after_deduplication"] = int(len(table))
    log(f"Loaded usable v2 score rows: {len(table):,}", audit)
    return table, stats


def keep_count(n_observations: int, fraction: float) -> int:
    if n_observations <= 0:
        return 0
    return max(1, int(math.floor(float(n_observations) * float(fraction))))


def build_selection_lookup(
    scores: pd.DataFrame,
    score_column: str,
    random_seed: int,
    audit: TextIO,
) -> tuple[dict[tuple[str, str, int, int, int], int], dict[str, Any]]:
    log("Ranking observations per signed HKL by v2_full score", audit)
    work = scores.sort_values([*HKL_COLUMNS, score_column, "source_filename", "event"], kind="mergesort").reset_index(
        drop=True
    )
    grouped = work.groupby(HKL_COLUMNS, sort=False)
    n_total = grouped[score_column].transform("size").to_numpy(dtype=np.int64)
    rank_low = grouped.cumcount().to_numpy(dtype=np.int64)
    keep_masks: dict[str, np.ndarray] = {}

    for spec in FILTER_SPECS:
        if spec.mode == "low_fraction":
            n_keep = np.maximum(1, np.floor(n_total.astype(float) * spec.keep_fraction).astype(np.int64))
            keep_masks[spec.name] = rank_low < n_keep
        elif spec.mode == "high_fraction":
            n_keep = np.maximum(1, np.floor(n_total.astype(float) * spec.keep_fraction).astype(np.int64))
            keep_masks[spec.name] = rank_low >= (n_total - n_keep)
        elif spec.mode == "random_50":
            keep_masks[spec.name] = np.zeros(len(work), dtype=bool)
        else:
            raise SystemExit(f"Unsupported filtering mode: {spec.mode}")

    rng = np.random.default_rng(int(random_seed))
    random_mask = keep_masks["random_50"]
    hkl_count = 0
    singleton_hkls = 0
    low_high_overlap_hkls = 0
    low_high_overlap_observations = 0
    for hkl, indices in grouped.indices.items():
        hkl_count += 1
        indices = np.asarray(indices, dtype=np.int64)
        n_obs = int(len(indices))
        if n_obs == 1:
            singleton_hkls += 1
        n_keep = keep_count(n_obs, 0.50)
        if n_keep > 0:
            chosen = rng.choice(indices, size=n_keep, replace=False)
            random_mask[chosen] = True
        if 2 * n_keep > n_obs:
            low_high_overlap_hkls += 1
            low_high_overlap_observations += int(2 * n_keep - n_obs)
        if hkl_count % 100_000 == 0:
            log(f"Selection progress: signed_hkls={hkl_count:,}", audit)

    keep_mask_bits = np.zeros(len(work), dtype=np.uint16)
    selection_counts: dict[str, int] = {}
    for bit_idx, spec in enumerate(FILTER_SPECS):
        mask = keep_masks[spec.name]
        keep_mask_bits[mask] |= np.uint16(1 << bit_idx)
        selection_counts[spec.name] = int(mask.sum())

    key_to_keep_mask: dict[tuple[str, str, int, int, int], int] = {}
    for row, mask in zip(work.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), keep_mask_bits, strict=True):
        source, event, h, k, l = row
        key_to_keep_mask[(str(source), str(event), int(h), int(k), int(l))] = int(mask)

    stats = {
        "score_rows_ranked": int(len(work)),
        "unique_signed_hkls_in_score_table": int(hkl_count),
        "singleton_signed_hkls_in_score_table": int(singleton_hkls),
        "selection_rounding": "floor(n_observations * keep_fraction), with a minimum of one kept observation per non-empty signed HKL",
        "random_seed": int(random_seed),
        "selected_score_rows_by_dataset": selection_counts,
        "low50_high50_count_delta_max_by_design": 0,
        "low50_high50_overlap_hkls_due_to_minimum_one": int(low_high_overlap_hkls),
        "low50_high50_overlap_observations_due_to_minimum_one": int(low_high_overlap_observations),
        "key_lookup_entries": int(len(key_to_keep_mask)),
    }
    log(
        "Selection lookup ready: "
        f"keys={stats['key_lookup_entries']:,}, signed_hkls={hkl_count:,}, "
        f"singletons={singleton_hkls:,}",
        audit,
    )
    return key_to_keep_mask, stats


def open_filter_handles(output_root: Path) -> dict[str, TextIO]:
    return {spec.name: (output_root / spec.output_name).open("w", encoding="utf-8") for spec in FILTER_SPECS}


def write_filtered_streams(
    stream_path: Path,
    output_root: Path,
    key_to_keep_mask: dict[tuple[str, str, int, int, int], int],
    missing_score_policy: str,
    progress_every: int,
    audit: TextIO,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    handles = open_filter_handles(output_root)
    by_spec = {
        spec.name: {
            "reflections_written": 0,
            "removed_reflections": 0,
            "unique_after": set(),
            "per_hkl_counts": defaultdict(int),
        }
        for spec in FILTER_SPECS
    }
    global_stats: dict[str, Any] = {
        "chunks_read": 0,
        "chunk_end_markers_read": 0,
        "reflections_read": 0,
        "matched_observations": 0,
        "missing_score_observations": 0,
        "unique_before": set(),
    }

    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False

    log("Writing filtered streams in one pass over the input stream", audit)
    try:
        with stream_path.open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                line = raw_line.rstrip("\n")
                if "Begin chunk" in line:
                    global_stats["chunks_read"] += 1
                    chunk_source = ""
                    chunk_event = ""
                    current_source = ""
                    current_event = ""
                    in_crystal = False
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if "End chunk" in line:
                    global_stats["chunk_end_markers_read"] += 1
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                image_match = STREAM_IMAGE_RE.match(line)
                if image_match:
                    if in_crystal:
                        current_source = normalize_source(image_match.group(1))
                    else:
                        chunk_source = normalize_source(image_match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                event_match = STREAM_EVENT_RE.match(line)
                if event_match:
                    if in_crystal:
                        current_event = normalize_event(event_match.group(1))
                    else:
                        chunk_event = normalize_event(event_match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if "Begin crystal" in line:
                    current_source = chunk_source
                    current_event = chunk_event
                    in_crystal = True
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if "End crystal" in line:
                    in_crystal = False
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if in_crystal and "Reflections measured after indexing" in line:
                    in_reflections = True
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if in_reflections and line.startswith("End of reflections"):
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if in_crystal and in_reflections:
                    hkl = parse_reflection_hkl(line)
                    if hkl is not None:
                        h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
                        hkl_key = (h, k, l)
                        global_stats["reflections_read"] += 1
                        global_stats["unique_before"].add(hkl_key)
                        key = build_key(current_source, current_event, h, k, l)
                        keep_mask = key_to_keep_mask.get(key)
                        matched = keep_mask is not None
                        if matched:
                            global_stats["matched_observations"] += 1
                        else:
                            global_stats["missing_score_observations"] += 1

                        for bit_idx, spec in enumerate(FILTER_SPECS):
                            keep = bool(int(keep_mask) & (1 << bit_idx)) if matched else missing_score_policy == "keep"
                            if keep:
                                handles[spec.name].write(raw_line)
                                by_spec[spec.name]["reflections_written"] += 1
                                by_spec[spec.name]["unique_after"].add(hkl_key)
                                by_spec[spec.name]["per_hkl_counts"][hkl_key] += 1
                            else:
                                by_spec[spec.name]["removed_reflections"] += 1

                        if global_stats["reflections_read"] % int(progress_every) == 0:
                            log(
                                "Stream write progress: "
                                f"reflections={global_stats['reflections_read']:,}, "
                                f"matched={global_stats['matched_observations']:,}, "
                                f"missing_scores={global_stats['missing_score_observations']:,}",
                                audit,
                            )
                        continue

                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()

    log(
        "Finished stream rewrite: "
        f"chunks={global_stats['chunks_read']:,}, "
        f"reflections={global_stats['reflections_read']:,}, "
        f"matched={global_stats['matched_observations']:,}, "
        f"missing_scores={global_stats['missing_score_observations']:,}",
        audit,
    )
    return by_spec, global_stats


def build_summary_rows(
    args: argparse.Namespace,
    full_mode: str,
    by_spec: dict[str, dict[str, Any]],
    global_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    command = command_text(args)
    chunks_read = int(global_stats["chunks_read"])
    reflections_read = int(global_stats["reflections_read"])
    unique_before = int(len(global_stats["unique_before"]))
    matched = int(global_stats["matched_observations"])
    missing = int(global_stats["missing_score_observations"])
    missing_policy = "kept" if args.missing_score_policy == "keep" else "removed"

    rows: list[dict[str, Any]] = [
        {
            "dataset_name": "full",
            "output_stream_path": str(args.output_root / "full.stream"),
            "input_stream_path": str(args.stream),
            "score_file_path": str(args.v2_scores_csv),
            "filtering_mode": f"full_{full_mode}",
            "keep_fraction": 1.0,
            "random_seed": "",
            "number_of_chunks_read": chunks_read,
            "number_of_chunks_written": chunks_read,
            "number_of_reflections_read": reflections_read,
            "number_of_reflections_written": reflections_read,
            "fraction_of_reflections_retained": 1.0,
            "number_of_unique_signed_HKLs_before_filtering": unique_before,
            "number_of_unique_signed_HKLs_after_filtering": unique_before,
            "number_of_observations_successfully_matched_to_v2_full_scores": matched,
            "number_of_observations_missing_scores": missing,
            "missing_score_observations_were": "not_filtered",
            "command_used_to_generate_stream": command,
        }
    ]

    spec_by_name = {spec.name: spec for spec in FILTER_SPECS}
    for spec in FILTER_SPECS:
        stats = by_spec[spec.name]
        written = int(stats["reflections_written"])
        rows.append(
            {
                "dataset_name": spec.name,
                "output_stream_path": str(args.output_root / spec.output_name),
                "input_stream_path": str(args.stream),
                "score_file_path": str(args.v2_scores_csv),
                "filtering_mode": spec.mode,
                "keep_fraction": float(spec.keep_fraction),
                "random_seed": int(args.random_seed) if spec.random_seed is not None else "",
                "number_of_chunks_read": chunks_read,
                "number_of_chunks_written": chunks_read,
                "number_of_reflections_read": reflections_read,
                "number_of_reflections_written": written,
                "fraction_of_reflections_retained": float(written / max(reflections_read, 1)),
                "number_of_unique_signed_HKLs_before_filtering": unique_before,
                "number_of_unique_signed_HKLs_after_filtering": int(len(stats["unique_after"])),
                "number_of_observations_successfully_matched_to_v2_full_scores": matched,
                "number_of_observations_missing_scores": missing,
                "missing_score_observations_were": missing_policy,
                "command_used_to_generate_stream": command,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_commands(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        "",
        "# Regenerate the final presentation stream filtering set.",
        command_text(args),
        "",
        "# If the v2 score CSV must be regenerated first, run this existing project command.",
        "# It creates geometry_coupling_v2_scores.csv with trust_risk_v2_full_norm.",
        "# " + v2_generation_command(args),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def validate_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    full_mode: str,
    by_spec: dict[str, dict[str, Any]],
    global_stats: dict[str, Any],
    audit: TextIO,
) -> dict[str, Any]:
    validations: dict[str, Any] = {}
    chunk_markers_ok = int(global_stats["chunks_read"]) > 0 and int(global_stats["chunks_read"]) == int(
        global_stats["chunk_end_markers_read"]
    )
    rows_by_name = {str(row["dataset_name"]): row for row in rows}

    existence = {}
    for name, row in rows_by_name.items():
        path = Path(str(row["output_stream_path"]))
        existence[name] = {
            "exists": path.exists() or path.is_symlink(),
            "non_empty": path.stat().st_size > 0 if path.exists() or path.is_symlink() else False,
            "chunk_markers_valid": bool(chunk_markers_ok),
        }
    validations["stream_existence_and_markers"] = existence

    full_path = args.output_root / "full.stream"
    validations["full_stream"] = {
        "mode": full_mode,
        "is_symlink": full_path.is_symlink(),
        "points_to_input_stream": bool(full_path.samefile(args.stream)) if full_path.exists() else False,
    }

    fraction_checks = {}
    for spec in FILTER_SPECS:
        row = rows_by_name[spec.name]
        actual = float(row["fraction_of_reflections_retained"])
        expected = float(spec.keep_fraction)
        tolerance = 0.03
        fraction_checks[spec.name] = {
            "expected_keep_fraction": expected,
            "actual_fraction_retained": actual,
            "tolerance": tolerance,
            "ok": abs(actual - expected) <= tolerance,
        }
    validations["reflection_fraction_checks"] = fraction_checks

    low_counts = by_spec["low_50"]["per_hkl_counts"]
    high_counts = by_spec["high_50"]["per_hkl_counts"]
    unique_before = global_stats["unique_before"]
    max_delta = 0
    hkl_delta_gt_one = 0
    for hkl in unique_before:
        delta = abs(int(low_counts.get(hkl, 0)) - int(high_counts.get(hkl, 0)))
        max_delta = max(max_delta, delta)
        if delta > 1:
            hkl_delta_gt_one += 1
    validations["low50_high50_size_complementarity"] = {
        "max_abs_count_delta_per_signed_hkl": int(max_delta),
        "signed_hkls_with_delta_gt_one": int(hkl_delta_gt_one),
        "ok": int(hkl_delta_gt_one) == 0,
    }

    monotonic_specs = sorted(
        (spec for spec in FILTER_SPECS if spec.mode == "low_fraction"),
        key=lambda spec: spec.keep_fraction,
    )
    monotonic_names = [spec.name for spec in monotonic_specs]
    monotonic_counts = [int(rows_by_name[name]["number_of_reflections_written"]) for name in monotonic_names]
    validations["low_fraction_monotonicity"] = {
        "datasets": monotonic_names,
        "reflection_counts": monotonic_counts,
        "ok": all(a <= b for a, b in zip(monotonic_counts, monotonic_counts[1:])),
    }

    validations["all_requested_streams_validated"] = all(
        item["exists"] and item["non_empty"] and item["chunk_markers_valid"] for item in existence.values()
    )
    validations["all_fraction_checks_passed"] = all(item["ok"] for item in fraction_checks.values())
    validations["all_validation_checks_passed"] = bool(
        validations["all_requested_streams_validated"]
        and validations["full_stream"]["points_to_input_stream"]
        and validations["all_fraction_checks_passed"]
        and validations["low50_high50_size_complementarity"]["ok"]
        and validations["low_fraction_monotonicity"]["ok"]
    )

    for key, value in validations.items():
        log(f"Validation {key}: {value}", audit)
    return validations


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    full_mode: str,
    score_stats: dict[str, int],
    selection_stats: dict[str, Any],
    validations: dict[str, Any],
    base_has_score_column: bool,
    v2_has_score_column: bool,
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "final presentation stream filtering set only; no merging, partialator, SHELXL, or refinement run",
        "paths": {
            "input_stream": str(args.stream),
            "base_reflection_scores_csv": str(args.base_reflection_scores_csv),
            "v2_scores_csv": str(args.v2_scores_csv),
            "output_root": str(args.output_root),
        },
        "score_column": str(args.score_column),
        "base_reflection_scores_has_score_column": bool(base_has_score_column),
        "v2_scores_csv_has_score_column": bool(v2_has_score_column),
        "v2_score_generation_command_if_needed": v2_generation_command(args),
        "matching": {
            "key": "source_filename + event + signed h,k,l",
            "canonicalize_hkls_to_4mmm": False,
            "source_normalization": "strip whitespace",
            "event_normalization": "strip whitespace and leading //",
        },
        "selection": {
            "score_sort": "ascending v2_full risk for low_fraction, descending tail for high_fraction",
            "rounding": selection_stats.get("selection_rounding", ""),
            "missing_score_policy": str(args.missing_score_policy),
            "random_seed": int(args.random_seed),
            "filter_specs": [
                {
                    "dataset": spec.name,
                    "mode": spec.mode,
                    "keep_fraction": float(spec.keep_fraction),
                    "output_name": spec.output_name,
                    "random_seed": int(args.random_seed) if spec.random_seed is not None else None,
                }
                for spec in FILTER_SPECS
            ],
        },
        "stream_rewrite_behavior": {
            "full_stream_mode": full_mode,
            "observation_level_filtering": True,
            "whole_chunks_removed": False,
            "empty_chunks_or_crystals_after_filtering": "kept to preserve CrystFEL metadata and syntax",
            "non_reflection_stream_text_modified": False,
        },
        "existing_project_scripts_reused_or_documented": {
            "filter_helper": str(PROJECT_ROOT / "tools/filter_stream_by_geometry_coupling_v2_keep_fraction.py"),
            "v2_score_generation": str(PROJECT_ROOT / "tools/compute_geometry_coupling_v2_scores.py"),
            "v2_workflow_wrapper": str(PROJECT_ROOT / "tools/run_oridyn_v2_score_filter_split_workflow.py"),
            "workflow_note": str(PROJECT_ROOT / "codex_notes/run_v2_score_filter_split_workflow_20260623.md"),
        },
        "commands": {
            "regenerate_streams": command_text(args),
            "regenerate_v2_scores_if_needed": v2_generation_command(args),
        },
        "score_load_stats": score_stats,
        "selection_stats": selection_stats,
        "summary_rows": rows,
        "validations": validations,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_compact_summary(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("dataset", "dataset_name"),
        ("mode", "filtering_mode"),
        ("keep", "keep_fraction"),
        ("refs_written", "number_of_reflections_written"),
        ("retained", "fraction_of_reflections_retained"),
        ("unique_hkls", "number_of_unique_signed_HKLs_after_filtering"),
        ("missing", "number_of_observations_missing_scores"),
    ]
    rendered = []
    for row in rows:
        rendered_row = {}
        for label, key in columns:
            value = row[key]
            if key == "fraction_of_reflections_retained":
                value = f"{float(value):.6f}"
            rendered_row[label] = str(value)
        rendered.append(rendered_row)
    widths = {
        label: max(len(label), *(len(row[label]) for row in rendered))
        for label, _key in columns
    }
    header = "  ".join(label.ljust(widths[label]) for label, _key in columns)
    sep = "  ".join("-" * widths[label] for label, _key in columns)
    print(header)
    print(sep)
    for row in rendered:
        print("  ".join(row[label].ljust(widths[label]) for label, _key in columns))


def main() -> int:
    args = parse_args()
    prepare_output_root(args.output_root, bool(args.overwrite))

    audit_path = args.output_root / "filtering_audit.log"
    with audit_path.open("w", encoding="utf-8") as audit:
        log("Starting final presentation stream filtering preparation", audit)
        log(f"Input stream: {args.stream}", audit)
        log(f"V2 score CSV: {args.v2_scores_csv}", audit)
        log(f"Output root: {args.output_root}", audit)

        base_has_score_column = csv_has_column(args.base_reflection_scores_csv, args.score_column)
        v2_has_score_column = csv_has_column(args.v2_scores_csv, args.score_column)
        if not v2_has_score_column:
            raise SystemExit(f"{args.v2_scores_csv} does not contain {args.score_column}")
        if not base_has_score_column:
            log(
                f"Base reflection score file lacks {args.score_column}; "
                "the existing v2 scorer creates the separate geometry_coupling_v2_scores.csv file.",
                audit,
            )
            log(f"Regeneration command: {v2_generation_command(args)}", audit)

        full_mode = link_or_copy_full_stream(args.stream, args.output_root / "full.stream", audit)
        score_table, score_stats = load_score_table(args.v2_scores_csv, args.score_column, audit)
        key_to_keep_mask, selection_stats = build_selection_lookup(
            score_table,
            str(args.score_column),
            int(args.random_seed),
            audit,
        )
        del score_table

        by_spec, global_stats = write_filtered_streams(
            args.stream,
            args.output_root,
            key_to_keep_mask,
            str(args.missing_score_policy),
            int(args.progress_every),
            audit,
        )
        del key_to_keep_mask

        rows = build_summary_rows(args, full_mode, by_spec, global_stats)
        write_summary_csv(args.output_root / "filtering_summary.csv", rows)
        write_commands(args.output_root / "filtering_commands.sh", args)
        validations = validate_outputs(args, rows, full_mode, by_spec, global_stats, audit)
        write_manifest(
            args.output_root / "filtering_manifest.json",
            args,
            rows,
            full_mode,
            score_stats,
            selection_stats,
            validations,
            base_has_score_column,
            v2_has_score_column,
        )

        log(f"Wrote summary CSV: {args.output_root / 'filtering_summary.csv'}", audit)
        log(f"Wrote manifest JSON: {args.output_root / 'filtering_manifest.json'}", audit)
        log(f"Wrote regeneration commands: {args.output_root / 'filtering_commands.sh'}", audit)
        log("Final presentation stream filtering preparation complete", audit)

    print_compact_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
