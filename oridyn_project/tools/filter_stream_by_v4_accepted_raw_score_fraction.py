#!/usr/bin/env python3
"""Filter a CrystFEL stream by high v4 raw score among partialator-accepted observations only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
DEFAULT_SCORE_COLUMN = "local_crowding_target_gated_raw"
DEFAULT_KEEP_FRACTIONS = [0.90, 0.80, 0.70, 0.60, 0.50]
DEFAULT_MIN_ACCEPTED_KEEP = 20
DEFAULT_PROGRESS_EVERY = 1_000_000
DEFAULT_SCORES_CHUNKSIZE = 500_000

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="Input CrystFEL stream")
    parser.add_argument("--accepted-scores", required=True, type=Path, help="Accepted-only v4 raw score CSV")
    parser.add_argument("--output-root", required=True, type=Path, help="Output directory for filtered stream variants")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN, help="Raw v4 score column; higher is worse")
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--min-accepted-keep", type=int, default=DEFAULT_MIN_ACCEPTED_KEEP)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--scores-chunksize", type=int, default=DEFAULT_SCORES_CHUNKSIZE)
    parser.add_argument("--max-events", type=int, default=None, help="Smoke-test limit on stream crystal blocks/events")
    parser.add_argument("--summarize-only", action="store_true", help="Build selection tables, but do not write stream variants")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.stream.is_file():
        raise SystemExit(f"--stream must be a file: {args.stream}")
    if not args.accepted_scores.is_file():
        raise SystemExit(f"--accepted-scores must be a CSV file: {args.accepted_scores}")
    if int(args.min_accepted_keep) < 1:
        raise SystemExit("--min-accepted-keep must be >= 1")
    if int(args.progress_every) < 1:
        raise SystemExit("--progress-every must be >= 1")
    if int(args.scores_chunksize) < 1:
        raise SystemExit("--scores-chunksize must be >= 1")
    if args.max_events is not None and int(args.max_events) < 1:
        raise SystemExit("--max-events must be >= 1 when provided")

    keep_fractions = []
    for value in args.keep_fractions:
        fraction = float(value)
        if not np.isfinite(fraction) or not (0.0 < fraction <= 1.0):
            raise SystemExit("--keep-fractions values must satisfy 0 < fraction <= 1")
        keep_fractions.append(fraction)
    args.keep_fractions = sorted(set(keep_fractions), reverse=True)
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_source(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_event(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def build_key(source: Any, event: Any, h: int, k: int, l: int) -> tuple[str, str, int, int, int]:
    return normalize_source(source), normalize_event(event), int(h), int(k), int(l)


def percent_label(fraction: float) -> int:
    return int(round(float(fraction) * 100.0))


def variant_name(fraction: float) -> str:
    return f"v4_accepted_raw_score_keep{percent_label(fraction):02d}"


def output_stream_name(fraction: float) -> str:
    return f"MFM300_VIII_v4_accepted_raw_score_keep{percent_label(fraction):02d}.stream"


def output_paths(output_root: Path, keep_fractions: list[float]) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for fraction in keep_fractions:
        variant = variant_name(fraction)
        paths[variant] = {
            "stream": output_root / output_stream_name(fraction),
            "summary_json": output_root / f"{variant}_summary.json",
            "removed_by_hkl_csv": output_root / f"{variant}_removed_by_hkl.csv",
        }
    return paths


def ensure_outputs(output_root: Path, paths: dict[str, dict[str, Path]], summarize_only: bool, overwrite: bool) -> None:
    blocked = []
    if not summarize_only:
        for variant_paths in paths.values():
            for path in variant_paths.values():
                if path.exists():
                    blocked.append(path)
    for name in ["filter_sweep_summary.csv", "accepted_hkl_summary.csv", "run_metadata.json"]:
        path = output_root / name
        if path.exists():
            blocked.append(path)
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked[:20])
        more = "" if len(blocked) <= 20 else f"\n  ... and {len(blocked) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}{more}\nUse --overwrite if intended.")
    output_root.mkdir(parents=True, exist_ok=True)


def require_columns(header: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


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


def collect_smoke_keys(
    stream_path: Path,
    max_events: int,
    progress_every: int,
) -> tuple[set[tuple[str, str, int, int, int]], dict[str, int]]:
    keys: set[tuple[str, str, int, int, int]] = set()
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    crystals_seen = 0
    observations_seen = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if "Begin chunk" in line:
                chunk_source = ""
                chunk_event = ""
                current_source = ""
                current_event = ""
                in_crystal = False
                in_reflections = False
                continue
            if match := STREAM_IMAGE_RE.match(line):
                if in_crystal:
                    current_source = normalize_source(match.group(1))
                else:
                    chunk_source = normalize_source(match.group(1))
                continue
            if match := STREAM_EVENT_RE.match(line):
                if in_crystal:
                    current_event = normalize_event(match.group(1))
                else:
                    chunk_event = normalize_event(match.group(1))
                continue
            if "Begin crystal" in line:
                crystals_seen += 1
                if crystals_seen > int(max_events):
                    break
                current_source = chunk_source
                current_event = chunk_event
                in_crystal = True
                in_reflections = False
                continue
            if "End crystal" in line:
                in_crystal = False
                in_reflections = False
                continue
            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                continue
            if in_reflections and line.startswith("End of reflections"):
                in_reflections = False
                continue
            if in_crystal and in_reflections:
                hkl = parse_reflection_hkl(line)
                if hkl is None:
                    continue
                observations_seen += 1
                keys.add(build_key(current_source, current_event, *hkl))
                if observations_seen % int(progress_every) == 0:
                    log(f"Smoke key collection: observations={observations_seen:,}, keys={len(keys):,}")

    return keys, {
        "max_events": int(max_events),
        "smoke_crystals_seen": int(min(crystals_seen, int(max_events))),
        "smoke_observations_seen": int(observations_seen),
        "smoke_unique_keys": int(len(keys)),
    }


def normalize_score_chunk(chunk: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = chunk.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = out[HKL_COLUMNS].isna().any(axis=1) | out[score_column].isna()
    out = out.loc[~bad].copy()
    if out.empty:
        return out
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out.loc[:, [*KEY_COLUMNS, score_column]].copy()


def filter_chunk_to_keys(chunk: pd.DataFrame, key_filter: set[tuple[str, str, int, int, int]]) -> pd.DataFrame:
    if chunk.empty:
        return chunk
    keys = [
        build_key(source, event, int(h), int(k), int(l))
        for source, event, h, k, l in chunk.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    return chunk.loc[[key in key_filter for key in keys]].copy()


def load_accepted_scores(
    scores_path: Path,
    score_column: str,
    key_filter: set[tuple[str, str, int, int, int]] | None,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(scores_path, nrows=0).columns.tolist()
    require_columns(header, [*KEY_COLUMNS, score_column], "accepted-only v4 score CSV")
    usecols = [*KEY_COLUMNS, score_column]
    chunks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {
        "score_rows_read": 0,
        "score_rows_after_cleanup": 0,
        "score_rows_after_smoke_key_filter": 0,
    }
    for idx, chunk in enumerate(pd.read_csv(scores_path, usecols=usecols, chunksize=int(chunksize)), start=1):
        stats["score_rows_read"] += int(len(chunk))
        normalized = normalize_score_chunk(chunk, score_column)
        stats["score_rows_after_cleanup"] += int(len(normalized))
        if key_filter is not None:
            normalized = filter_chunk_to_keys(normalized, key_filter)
            stats["score_rows_after_smoke_key_filter"] += int(len(normalized))
        if not normalized.empty:
            chunks.append(normalized)
        if idx == 1 or idx % 5 == 0:
            extra = ""
            if key_filter is not None:
                extra = f", matched_smoke_rows={stats['score_rows_after_smoke_key_filter']:,}"
            log(f"Accepted-score CSV scan: chunks={idx:,}, rows_read={stats['score_rows_read']:,}{extra}")
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    return table, stats


def value_quantiles(values: pd.Series) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {"q10": None, "median": None, "q90": None}
    q10, q50, q90 = np.quantile(finite, [0.10, 0.50, 0.90])
    return {"q10": float(q10), "median": float(q50), "q90": float(q90)}


def build_filter_masks(
    accepted: pd.DataFrame,
    score_column: str,
    keep_fractions: list[float],
    min_accepted_keep: int,
) -> tuple[
    dict[tuple[str, str, int, int, int], int],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    if accepted.empty:
        raise SystemExit("No usable accepted score rows were loaded")

    work = accepted.copy()
    duplicated = work.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(duplicated.sum())
    duplicate_keys = int(work.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact observation keys in accepted scores; keeping first row per key")
        work = work.drop_duplicates(KEY_COLUMNS, keep="first").copy()

    grouped = work.groupby(HKL_COLUMNS, sort=False, dropna=False)
    accepted_hkl_summary = grouped.agg(
        n_accepted=(score_column, "size"),
        raw_score_min=(score_column, "min"),
        raw_score_q10=(score_column, lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.10))),
        raw_score_median=(score_column, "median"),
        raw_score_q90=(score_column, lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.90))),
        raw_score_max=(score_column, "max"),
    ).reset_index()
    accepted_hkl_summary["low_count_no_filter"] = accepted_hkl_summary["n_accepted"] < int(min_accepted_keep)

    work = work.merge(
        accepted_hkl_summary.loc[:, [*HKL_COLUMNS, "n_accepted", "low_count_no_filter"]],
        on=HKL_COLUMNS,
        how="left",
    )
    work = work.sort_values(
        [*HKL_COLUMNS, score_column, "source_filename", "event"],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    work["rank_low_to_high_within_hkl"] = work.groupby(HKL_COLUMNS, sort=False).cumcount().astype("int64")

    remove_mask_bits = np.zeros(len(work), dtype=np.uint16)
    variant_rows: list[dict[str, Any]] = []
    removed_by_hkl_tables: dict[str, pd.DataFrame] = {}

    for bit_idx, keep_fraction in enumerate(keep_fractions):
        bit = np.uint16(1 << bit_idx)
        variant = variant_name(keep_fraction)
        target_keep = np.ceil(work["n_accepted"].to_numpy(dtype=float) * float(keep_fraction)).astype("int64")
        keep_n = np.maximum(target_keep, int(min_accepted_keep))
        keep_n = np.minimum(keep_n, work["n_accepted"].to_numpy(dtype="int64"))
        keep = work["low_count_no_filter"].to_numpy(dtype=bool)
        keep = keep | (work["rank_low_to_high_within_hkl"].to_numpy(dtype="int64") < keep_n)
        remove = ~keep
        remove_mask_bits[remove] |= bit

        work[f"target_keep_{variant}"] = target_keep
        work[f"keep_n_{variant}"] = keep_n
        work[f"remove_{variant}"] = remove
        removed = work.loc[remove]
        kept = work.loc[~remove]
        filtered_hkl_count = int(removed.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]) if not removed.empty else 0

        removed_stats = value_quantiles(removed[score_column])
        kept_stats = value_quantiles(kept[score_column])
        variant_rows.append(
            {
                "variant": variant,
                "keep_fraction": float(keep_fraction),
                "score_column": score_column,
                "score_direction": "higher_score_is_worse",
                "accepted_observations": int(len(work)),
                "accepted_observations_removed_by_selection": int(remove.sum()),
                "accepted_observations_kept_by_selection": int((~remove).sum()),
                "accepted_fraction_removed_by_selection": float(remove.mean()) if len(remove) else 0.0,
                "unique_signed_hkls_total": int(accepted_hkl_summary.shape[0]),
                "unique_signed_hkls_filtered": filtered_hkl_count,
                "unique_signed_hkls_low_count_kept_unchanged": int(accepted_hkl_summary["low_count_no_filter"].sum()),
                "min_accepted_keep": int(min_accepted_keep),
                "removed_raw_score_q10": removed_stats["q10"],
                "removed_raw_score_median": removed_stats["median"],
                "removed_raw_score_q90": removed_stats["q90"],
                "kept_raw_score_q10": kept_stats["q10"],
                "kept_raw_score_median": kept_stats["median"],
                "kept_raw_score_q90": kept_stats["q90"],
            }
        )

        per_hkl = accepted_hkl_summary.copy()
        per_hkl["keep_fraction"] = float(keep_fraction)
        per_hkl["target_keep"] = np.ceil(per_hkl["n_accepted"].to_numpy(dtype=float) * float(keep_fraction)).astype("int64")
        per_hkl["keep_n"] = np.minimum(
            np.maximum(per_hkl["target_keep"].to_numpy(dtype="int64"), int(min_accepted_keep)),
            per_hkl["n_accepted"].to_numpy(dtype="int64"),
        )
        per_hkl["remove_n"] = np.where(
            per_hkl["n_accepted"].to_numpy(dtype="int64") < int(min_accepted_keep),
            0,
            np.maximum(0, per_hkl["n_accepted"].to_numpy(dtype="int64") - per_hkl["keep_n"].to_numpy(dtype="int64")),
        ).astype("int64")
        per_hkl["kept_accepted_after_filter"] = per_hkl["n_accepted"].astype("int64") - per_hkl["remove_n"].astype("int64")
        removed_by_hkl_tables[variant] = per_hkl.loc[per_hkl["remove_n"] > 0].sort_values(
            ["remove_n", "h", "k", "l"], ascending=[False, True, True, True]
        )

    key_to_mask: dict[tuple[str, str, int, int, int], int] = {}
    for row, mask in zip(work.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), remove_mask_bits, strict=True):
        source, event, h, k, l = row
        key_to_mask[(str(source), str(event), int(h), int(k), int(l))] = int(mask)

    selection_stats: dict[str, Any] = {
        "accepted_rows_after_duplicate_cleanup": int(len(work)),
        "duplicate_score_rows": duplicate_rows,
        "duplicate_score_keys": duplicate_keys,
        "key_to_mask_entries": int(len(key_to_mask)),
        "unique_signed_hkls_total": int(accepted_hkl_summary.shape[0]),
        "unique_signed_hkls_low_count_kept_unchanged": int(accepted_hkl_summary["low_count_no_filter"].sum()),
        "min_accepted_keep": int(min_accepted_keep),
        "rules": [
            "Only accepted-only v4 score rows are candidates for removal.",
            "Stream observations absent from the accepted-only table are kept unchanged.",
            "Per signed HKL: if n_accepted < min_accepted_keep, remove zero observations.",
            "Otherwise target_keep=ceil(keep_fraction*n_accepted), keep_n=max(target_keep,min_accepted_keep), remove_n=max(0,n_accepted-keep_n).",
            "High raw local_crowding_target_gated_raw observations are removed; low raw scores are kept.",
            "Signed HKLs are preserved exactly; no 4/mmm canonicalization is applied.",
        ],
    }
    return key_to_mask, pd.DataFrame.from_records(variant_rows), accepted_hkl_summary, removed_by_hkl_tables, selection_stats


def write_stream_variants(
    stream_path: Path,
    paths: dict[str, dict[str, Path]],
    key_to_mask: dict[tuple[str, str, int, int, int], int],
    keep_fractions: list[float],
    progress_every: int,
    max_events: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[tuple[int, int, int], int]], dict[tuple[int, int, int], int]]:
    variants = [variant_name(fraction) for fraction in keep_fractions]
    variant_bits = {variant: 1 << idx for idx, variant in enumerate(variants)}
    handles = {variant: paths[variant]["stream"].open("w", encoding="utf-8") for variant in variants}
    stats = {
        variant: {
            "total_stream_observations_seen": 0,
            "accepted_matched_observations": 0,
            "nonaccepted_or_unmatched_observations": 0,
            "kept_observations": 0,
            "removed_observations": 0,
        }
        for variant in variants
    }
    removed_by_hkl = {variant: defaultdict(int) for variant in variants}
    matched_hkl_counts: dict[tuple[int, int, int], int] = defaultdict(int)

    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    chunks_seen = 0
    crystals_seen = 0
    stream_observations_seen = 0
    matched_observations = 0
    unmatched_observations = 0

    try:
        with stream_path.open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                line = raw_line.rstrip("\n")
                if "Begin chunk" in line:
                    chunks_seen += 1
                    chunk_source = ""
                    chunk_event = ""
                    current_source = ""
                    current_event = ""
                    in_crystal = False
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue
                if match := STREAM_IMAGE_RE.match(line):
                    if in_crystal:
                        current_source = normalize_source(match.group(1))
                    else:
                        chunk_source = normalize_source(match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue
                if match := STREAM_EVENT_RE.match(line):
                    if in_crystal:
                        current_event = normalize_event(match.group(1))
                    else:
                        chunk_event = normalize_event(match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue
                if "Begin crystal" in line:
                    crystals_seen += 1
                    if max_events is not None and crystals_seen > int(max_events):
                        log(f"Reached --max-events {max_events}; stopping stream rewrite early")
                        break
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
                        stream_observations_seen += 1
                        key = build_key(current_source, current_event, *hkl)
                        mask = key_to_mask.get(key)
                        matched = mask is not None
                        if matched:
                            matched_observations += 1
                            matched_hkl_counts[(int(hkl[0]), int(hkl[1]), int(hkl[2]))] += 1
                        else:
                            unmatched_observations += 1
                        for variant in variants:
                            stats[variant]["total_stream_observations_seen"] += 1
                            if matched:
                                stats[variant]["accepted_matched_observations"] += 1
                            else:
                                stats[variant]["nonaccepted_or_unmatched_observations"] += 1
                            remove = matched and bool(int(mask) & int(variant_bits[variant]))
                            if remove:
                                stats[variant]["removed_observations"] += 1
                                removed_by_hkl[variant][(int(hkl[0]), int(hkl[1]), int(hkl[2]))] += 1
                            else:
                                handles[variant].write(raw_line)
                                stats[variant]["kept_observations"] += 1
                        if stream_observations_seen % int(progress_every) == 0:
                            removed_text = ", ".join(
                                f"{variant} removed={stats[variant]['removed_observations']:,}" for variant in variants
                            )
                            log(
                                "Stream progress: "
                                f"observations={stream_observations_seen:,}, "
                                f"accepted_matched={matched_observations:,}, "
                                f"nonaccepted_or_unmatched={unmatched_observations:,}, "
                                f"{removed_text}"
                            )
                        continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()

    stream_stats = {
        "chunks_seen": int(chunks_seen),
        "crystals_seen": int(min(crystals_seen, int(max_events)) if max_events is not None else crystals_seen),
        "stream_observations_seen": int(stream_observations_seen),
        "stream_accepted_matched_observations": int(matched_observations),
        "stream_nonaccepted_or_unmatched_observations": int(unmatched_observations),
    }
    rows = []
    for variant, fraction in zip(variants, keep_fractions, strict=True):
        row = dict(stats[variant])
        row["variant"] = variant
        row["keep_fraction"] = float(fraction)
        row["removed_fraction_of_stream"] = row["removed_observations"] / max(row["total_stream_observations_seen"], 1)
        row["removed_fraction_of_accepted_matched"] = row["removed_observations"] / max(row["accepted_matched_observations"], 1)
        row["output_stream_path"] = str(paths[variant]["stream"])
        rows.append(row)
    return pd.DataFrame.from_records(rows), stream_stats, removed_by_hkl, matched_hkl_counts


def write_removed_by_hkl_csvs(
    paths: dict[str, dict[str, Path]],
    removed_by_hkl_tables: dict[str, pd.DataFrame],
    stream_removed_by_hkl: dict[str, dict[tuple[int, int, int], int]],
    matched_hkl_counts: dict[tuple[int, int, int], int],
) -> dict[str, int]:
    affected_counts = {}
    for variant, table in removed_by_hkl_tables.items():
        out = table.copy()
        if not out.empty:
            stream_removed = []
            stream_matched = []
            for row in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None):
                hkl = (int(row[0]), int(row[1]), int(row[2]))
                stream_removed.append(int(stream_removed_by_hkl[variant].get(hkl, 0)))
                stream_matched.append(int(matched_hkl_counts.get(hkl, 0)))
            out["stream_removed_observations"] = stream_removed
            out["stream_accepted_matched_observations"] = stream_matched
        else:
            out = pd.DataFrame(
                columns=[
                    "h",
                    "k",
                    "l",
                    "n_accepted",
                    "keep_fraction",
                    "target_keep",
                    "keep_n",
                    "remove_n",
                    "kept_accepted_after_filter",
                    "stream_removed_observations",
                    "stream_accepted_matched_observations",
                ]
            )
        out.to_csv(paths[variant]["removed_by_hkl_csv"], index=False)
        affected_counts[variant] = int(len(out))
    return affected_counts


def write_variant_jsons(
    paths: dict[str, dict[str, Path]],
    sweep_summary: pd.DataFrame,
    stream_stats: dict[str, Any],
    selection_stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    rows = {str(row.variant): row._asdict() for row in sweep_summary.itertuples(index=False)}
    for variant, variant_paths in paths.items():
        payload = {
            "variant": variant,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "stream": str(args.stream),
                "accepted_scores": str(args.accepted_scores),
            },
            "score_column": str(args.score_column),
            "keep_fractions": [float(value) for value in args.keep_fractions],
            "min_accepted_keep": int(args.min_accepted_keep),
            "selection_stats": selection_stats,
            "stream_stats": stream_stats,
            "variant_summary": rows.get(variant, {}),
            "outputs": {name: str(path) for name, path in variant_paths.items()},
        }
        variant_paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_metadata(
    output_root: Path,
    args: argparse.Namespace,
    score_load_stats: dict[str, Any],
    selection_stats: dict[str, Any],
    stream_stats: dict[str, Any] | None,
) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "accepted_scores": str(args.accepted_scores),
        },
        "output_root": str(output_root),
        "score_column": str(args.score_column),
        "score_direction": "higher_score_is_worse",
        "keep_fractions": [float(value) for value in args.keep_fractions],
        "min_accepted_keep": int(args.min_accepted_keep),
        "progress_every": int(args.progress_every),
        "scores_chunksize": int(args.scores_chunksize),
        "max_events": None if args.max_events is None else int(args.max_events),
        "summarize_only": bool(args.summarize_only),
        "score_load_stats": score_load_stats,
        "selection_stats": selection_stats,
        "stream_stats": stream_stats,
        "warnings": [
            "This is an observation-level v4 raw-score filter on partialator-accepted observations only.",
            "No v4 score normalization or log1p score is used for selection.",
            "No within-HKL spread or q10/q90 gate is used.",
            "Observation matching uses exact source_filename + normalized event + signed h,k,l.",
            "HKLs are not canonicalized to 4/mmm.",
            "Stream observations absent from the accepted-only score table are kept unchanged.",
        ],
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = output_paths(args.output_root, args.keep_fractions)
    ensure_outputs(args.output_root, paths, bool(args.summarize_only), bool(args.overwrite))
    log(f"Output root: {args.output_root}")
    for fraction in args.keep_fractions:
        variant = variant_name(fraction)
        log(f"Prepared variant keep_fraction={fraction:.3f}: {paths[variant]['stream']}")

    smoke_key_filter = None
    smoke_stats: dict[str, int] = {}
    if args.max_events is not None:
        log(f"Collecting smoke-test keys for first {int(args.max_events)} stream events/crystals")
        smoke_key_filter, smoke_stats = collect_smoke_keys(args.stream, int(args.max_events), int(args.progress_every))
        log(f"Smoke key collection done: keys={len(smoke_key_filter):,}")

    log("Loading accepted-only v4 raw score table")
    accepted, score_load_stats = load_accepted_scores(args.accepted_scores, str(args.score_column), smoke_key_filter, int(args.scores_chunksize))
    score_load_stats.update(smoke_stats)
    log(f"Loaded accepted scores: rows={len(accepted):,}")

    log("Selecting high raw-score accepted observations for removal")
    key_to_mask, variant_selection, accepted_hkl_summary, removed_by_hkl_tables, selection_stats = build_filter_masks(
        accepted,
        str(args.score_column),
        args.keep_fractions,
        int(args.min_accepted_keep),
    )
    log(
        "Selection ready: "
        f"accepted_hkls={selection_stats['unique_signed_hkls_total']:,}, "
        f"low_count_hkls={selection_stats['unique_signed_hkls_low_count_kept_unchanged']:,}, "
        f"lookup_keys={selection_stats['key_to_mask_entries']:,}"
    )

    accepted_hkl_summary.to_csv(args.output_root / "accepted_hkl_summary.csv", index=False)

    stream_stats = None
    if args.summarize_only:
        log("Summarize-only mode: not writing stream variants")
        for variant, table in removed_by_hkl_tables.items():
            table.to_csv(paths[variant]["removed_by_hkl_csv"], index=False)
        sweep_summary = variant_selection.copy()
        sweep_summary["output_stream_path"] = [str(paths[str(row.variant)]["stream"]) for row in sweep_summary.itertuples(index=False)]
    else:
        log("Writing filtered stream variants in one pass over the original stream")
        stream_summary, stream_stats, stream_removed_by_hkl, matched_hkl_counts = write_stream_variants(
            args.stream,
            paths,
            key_to_mask,
            args.keep_fractions,
            int(args.progress_every),
            None if args.max_events is None else int(args.max_events),
        )
        affected_counts = write_removed_by_hkl_csvs(paths, removed_by_hkl_tables, stream_removed_by_hkl, matched_hkl_counts)
        stream_summary["number_of_signed_hkls_affected"] = stream_summary["variant"].map(affected_counts).astype("int64")
        sweep_summary = stream_summary.merge(variant_selection, on=["variant", "keep_fraction"], how="left")
        write_variant_jsons(paths, sweep_summary, stream_stats, selection_stats, args)

    sweep_summary.to_csv(args.output_root / "filter_sweep_summary.csv", index=False)
    write_run_metadata(args.output_root, args, score_load_stats, selection_stats, stream_stats)

    log("V4 accepted raw-score fraction filter sweep complete")
    for variant_paths in paths.values():
        if not args.summarize_only:
            print(f"Wrote: {variant_paths['stream']}")
            print(f"Wrote: {variant_paths['summary_json']}")
        print(f"Wrote: {variant_paths['removed_by_hkl_csv']}")
    print(f"Wrote: {args.output_root / 'accepted_hkl_summary.csv'}")
    print(f"Wrote: {args.output_root / 'filter_sweep_summary.csv'}")
    print(f"Wrote: {args.output_root / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())