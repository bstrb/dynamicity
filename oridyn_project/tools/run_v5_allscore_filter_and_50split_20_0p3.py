#!/usr/bin/env python3
"""Run v5 all-score accepted-observation filters and 50% diagnostic splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
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
DEFAULT_SCORE_COLUMN = "nonself_local_excitation_raw"
DEFAULT_KEEP_FRACTIONS = [0.90, 0.80, 0.70, 0.60, 0.50]
DEFAULT_MIN_ACCEPTED_KEEP = 20
DEFAULT_MIN_ACCEPTED_FOR_50SPLIT = 40
DEFAULT_SEED = 1
DEFAULT_SCORES_CHUNKSIZE = 500_000
DEFAULT_PROGRESS_EVERY = 1_000_000
SELECTED_HKLS = [(0, 4, 0), (10, -7, 3), (0, 27, 5), (6, 6, 2), (8, 4, 0), (9, 3, 0), (8, 6, 0), (7, 5, 0)]

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    kind: str
    stream_name: str
    keep_fraction: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--v5-scores", required=True, type=Path)
    parser.add_argument("--accepted-scores", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--min-accepted-keep", type=int, default=DEFAULT_MIN_ACCEPTED_KEEP)
    parser.add_argument("--min-accepted-for-50split", type=int, default=DEFAULT_MIN_ACCEPTED_FOR_50SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-events", type=int, default=None, help="Tiny smoke-test limit on stream crystal blocks/events")
    parser.add_argument("--scores-chunksize", type=int, default=DEFAULT_SCORES_CHUNKSIZE)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    args = parser.parse_args()

    if not args.stream.is_file():
        raise SystemExit(f"--stream must be a file: {args.stream}")
    if not args.v5_scores.is_file():
        raise SystemExit(f"--v5-scores must be a CSV file: {args.v5_scores}")
    if not args.accepted_scores.is_file():
        raise SystemExit(f"--accepted-scores must be a CSV file: {args.accepted_scores}")
    if int(args.min_accepted_keep) < 1:
        raise SystemExit("--min-accepted-keep must be >= 1")
    if int(args.min_accepted_for_50split) < 1:
        raise SystemExit("--min-accepted-for-50split must be >= 1")
    if int(args.scores_chunksize) < 1:
        raise SystemExit("--scores-chunksize must be >= 1")
    if int(args.progress_every) < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.max_events is not None and int(args.max_events) < 1:
        raise SystemExit("--max-events must be >= 1 when supplied")

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


def variant_specs(keep_fractions: list[float], seed: int) -> list[VariantSpec]:
    specs = [
        VariantSpec(
            variant=f"v5_allscore_lowrisk_keep{percent_label(fraction):02d}",
            kind="lowrisk",
            stream_name=f"MFM300_VIII_v5_allscore_lowrisk_keep{percent_label(fraction):02d}.stream",
            keep_fraction=float(fraction),
        )
        for fraction in keep_fractions
    ]
    specs.extend(
        [
            VariantSpec("v5_allscore_low50", "low50", "MFM300_VIII_v5_allscore_low50_accepted.stream"),
            VariantSpec("v5_allscore_high50", "high50", "MFM300_VIII_v5_allscore_high50_accepted.stream"),
            VariantSpec(
                f"v5_allscore_random50_seed{int(seed)}",
                "random50",
                f"MFM300_VIII_v5_allscore_random50_seed{int(seed)}_accepted.stream",
            ),
        ]
    )
    return specs


def output_paths(output_root: Path, specs: list[VariantSpec]) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for spec in specs:
        paths[spec.variant] = {
            "stream": output_root / spec.stream_name,
            "kept_keys_csv": output_root / f"{spec.variant}_kept_accepted_keys.csv",
            "removed_keys_csv": output_root / f"{spec.variant}_removed_accepted_keys.csv",
            "per_hkl_summary_csv": output_root / f"{spec.variant}_per_hkl_summary.csv",
        }
    return paths


def ensure_outputs(output_root: Path, paths: dict[str, dict[str, Path]], overwrite: bool) -> None:
    blocked = []
    for variant_paths in paths.values():
        for path in variant_paths.values():
            if path.exists():
                blocked.append(path)
    for name in [
        "v5_allscore_filter_and_50split_summary.csv",
        "v5_allscore_filter_and_50split_summary.md",
        "v5_allscore_selected_hkl_summary.csv",
        "run_metadata.json",
    ]:
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


def collect_smoke_keys(stream_path: Path, max_events: int, progress_every: int) -> tuple[set[tuple[str, str, int, int, int]], dict[str, int]]:
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


def normalize_key_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[~out[HKL_COLUMNS].isna().any(axis=1)].copy()
    if out.empty:
        return out.loc[:, KEY_COLUMNS].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out.loc[:, KEY_COLUMNS].copy()


def normalize_score_chunk(chunk: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = normalize_key_chunk(chunk)
    if out.empty:
        out[score_column] = pd.Series(dtype=float)
        return out.loc[:, [*KEY_COLUMNS, score_column]].copy()
    out[score_column] = pd.to_numeric(chunk.loc[out.index, score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.loc[~out[score_column].isna()].copy()
    return out.loc[:, [*KEY_COLUMNS, score_column]].copy()


def filter_chunk_to_keys(chunk: pd.DataFrame, key_filter: set[tuple[str, str, int, int, int]]) -> pd.DataFrame:
    if chunk.empty:
        return chunk
    keys = [
        build_key(source, event, int(h), int(k), int(l))
        for source, event, h, k, l in chunk.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    return chunk.loc[[key in key_filter for key in keys]].copy()


def load_accepted_keys(accepted_path: Path, key_filter: set[tuple[str, str, int, int, int]] | None, chunksize: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(accepted_path, nrows=0).columns.tolist()
    require_columns(header, KEY_COLUMNS, "accepted-only table")
    chunks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {
        "accepted_rows_read": 0,
        "accepted_rows_after_key_cleanup": 0,
        "accepted_rows_after_smoke_key_filter": 0,
        "accepted_duplicate_rows": 0,
        "accepted_duplicate_keys": 0,
        "accepted_unique_keys": 0,
    }
    for idx, chunk in enumerate(pd.read_csv(accepted_path, usecols=KEY_COLUMNS, chunksize=int(chunksize)), start=1):
        stats["accepted_rows_read"] += int(len(chunk))
        normalized = normalize_key_chunk(chunk)
        stats["accepted_rows_after_key_cleanup"] += int(len(normalized))
        if key_filter is not None:
            normalized = filter_chunk_to_keys(normalized, key_filter)
            stats["accepted_rows_after_smoke_key_filter"] += int(len(normalized))
        if not normalized.empty:
            chunks.append(normalized)
        if idx == 1 or idx % 5 == 0:
            extra = ""
            if key_filter is not None:
                extra = f", matched_smoke_rows={stats['accepted_rows_after_smoke_key_filter']:,}"
            log(f"Accepted-key CSV scan: chunks={idx:,}, rows_read={stats['accepted_rows_read']:,}{extra}")
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=KEY_COLUMNS)
    duplicated = table.duplicated(KEY_COLUMNS, keep=False)
    stats["accepted_duplicate_rows"] = int(duplicated.sum())
    stats["accepted_duplicate_keys"] = int(table.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if stats["accepted_duplicate_rows"] else 0
    if stats["accepted_duplicate_rows"]:
        log("Warning: duplicate exact observation keys in accepted-only table; keeping first key")
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    stats["accepted_unique_keys"] = int(len(table))
    return table, stats


def load_accepted_v5_scores(v5_scores_path: Path, accepted_keys: pd.DataFrame, score_column: str, chunksize: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(v5_scores_path, nrows=0).columns.tolist()
    require_columns(header, [*KEY_COLUMNS, score_column], "v5 score CSV")
    if accepted_keys.empty:
        raise SystemExit("No usable accepted keys were loaded")
    usecols = [*KEY_COLUMNS, score_column]
    chunks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {"v5_score_rows_read": 0, "v5_score_rows_after_cleanup": 0, "accepted_v5_matched_rows": 0}
    for idx, chunk in enumerate(pd.read_csv(v5_scores_path, usecols=usecols, chunksize=int(chunksize)), start=1):
        stats["v5_score_rows_read"] += int(len(chunk))
        normalized = normalize_score_chunk(chunk, score_column)
        stats["v5_score_rows_after_cleanup"] += int(len(normalized))
        if normalized.empty:
            continue
        matched = normalized.merge(accepted_keys, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        stats["accepted_v5_matched_rows"] += int(len(matched))
        if not matched.empty:
            chunks.append(matched)
        if idx == 1 or idx % 5 == 0:
            log(
                "V5 score CSV scan: "
                f"chunks={idx:,}, rows_read={stats['v5_score_rows_read']:,}, "
                f"accepted_matches={stats['accepted_v5_matched_rows']:,}"
            )
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    stats["accepted_keys_without_v5_score"] = int(max(0, len(accepted_keys) - table.loc[:, KEY_COLUMNS].drop_duplicates().shape[0]))
    return table, stats


def stable_tie_break_values(work: pd.DataFrame, seed: int) -> np.ndarray:
    key_frame = work.loc[:, KEY_COLUMNS].copy()
    hashes = pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype=np.uint64)
    seed_mix = np.uint64((int(seed) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    return hashes ^ seed_mix


def rank_within_hkl(work: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> np.ndarray:
    ordered = work.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    ranks = np.empty(len(work), dtype=np.int64)
    rank_values = ordered.groupby(HKL_COLUMNS, sort=False).cumcount().to_numpy(dtype=np.int64)
    ranks[ordered.index.to_numpy(dtype=np.int64)] = rank_values
    return ranks


def score_stats(values: pd.Series) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {"min": None, "median": None, "max": None}
    return {"min": float(np.min(finite)), "median": float(np.median(finite)), "max": float(np.max(finite))}


def build_per_hkl_summary(work: pd.DataFrame, spec: VariantSpec, score_column: str, threshold: int) -> pd.DataFrame:
    remove_column = f"remove_{spec.variant}"
    temp = work.loc[:, [*HKL_COLUMNS, score_column, "zero_score", "positive_score", "n_accepted", remove_column]].copy()
    temp["kept"] = ~temp[remove_column].to_numpy(dtype=bool)
    temp["removed"] = temp[remove_column].to_numpy(dtype=bool)
    temp["zero_kept"] = temp["zero_score"].to_numpy(dtype=bool) & temp["kept"].to_numpy(dtype=bool)
    temp["zero_removed"] = temp["zero_score"].to_numpy(dtype=bool) & temp["removed"].to_numpy(dtype=bool)
    temp["positive_kept"] = temp["positive_score"].to_numpy(dtype=bool) & temp["kept"].to_numpy(dtype=bool)
    temp["positive_removed"] = temp["positive_score"].to_numpy(dtype=bool) & temp["removed"].to_numpy(dtype=bool)
    grouped = temp.groupby(HKL_COLUMNS, sort=False)
    summary = grouped.agg(
        n_accepted=(score_column, "size"),
        n_kept=("kept", "sum"),
        n_removed=("removed", "sum"),
        n_zero_before=("zero_score", "sum"),
        n_zero_kept=("zero_kept", "sum"),
        n_zero_removed=("zero_removed", "sum"),
        n_positive_before=("positive_score", "sum"),
        n_positive_kept=("positive_kept", "sum"),
        n_positive_removed=("positive_removed", "sum"),
        score_min_before=(score_column, "min"),
        score_median_before=(score_column, "median"),
        score_max_before=(score_column, "max"),
    ).reset_index()
    summary["variant"] = spec.variant
    summary["keep_fraction"] = spec.keep_fraction
    summary["split_kind"] = spec.kind
    summary["low_count_unchanged"] = summary["n_accepted"].to_numpy(dtype=np.int64) < int(threshold)
    summary["changed"] = summary["n_removed"].to_numpy(dtype=np.int64) > 0
    return summary


def build_selection_tables(
    accepted_v5: pd.DataFrame,
    specs: list[VariantSpec],
    score_column: str,
    min_accepted_keep: int,
    min_accepted_for_50split: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    if accepted_v5.empty:
        raise SystemExit("No accepted observations matched to v5 scores were loaded")
    work = accepted_v5.copy().reset_index(drop=True)
    duplicated = work.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(duplicated.sum())
    duplicate_keys = int(work.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact observation keys after accepted/v5 join; keeping first row per key")
        work = work.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)

    work[score_column] = pd.to_numeric(work[score_column], errors="coerce").astype(float)
    work["zero_score"] = work[score_column].to_numpy(dtype=float) == 0.0
    work["positive_score"] = work[score_column].to_numpy(dtype=float) > 0.0
    work["tie_break"] = stable_tie_break_values(work, seed)
    work["n_accepted"] = work.groupby(HKL_COLUMNS, sort=False)[score_column].transform("size").astype("int64")
    work["rank_low"] = rank_within_hkl(work, [*HKL_COLUMNS, score_column, "tie_break"], [True, True, True, True, True])
    work["rank_high"] = rank_within_hkl(work, [*HKL_COLUMNS, score_column, "tie_break"], [True, True, True, False, True])
    work["rank_random"] = rank_within_hkl(work, [*HKL_COLUMNS, "tie_break"], [True, True, True, True])

    remove_mask_bits = np.zeros(len(work), dtype=np.uint16)
    per_hkl_tables: dict[str, pd.DataFrame] = {}
    variant_rows: list[dict[str, Any]] = []
    n_accepted = work["n_accepted"].to_numpy(dtype=np.int64)

    for bit_idx, spec in enumerate(specs):
        bit = np.uint16(1 << bit_idx)
        if spec.kind == "lowrisk":
            target_keep = np.ceil(n_accepted.astype(float) * float(spec.keep_fraction)).astype(np.int64)
            keep_n = np.minimum(np.maximum(target_keep, int(min_accepted_keep)), n_accepted)
            keep = (n_accepted < int(min_accepted_keep)) | (work["rank_low"].to_numpy(dtype=np.int64) < keep_n)
            threshold = int(min_accepted_keep)
        elif spec.kind == "low50":
            eligible = n_accepted >= int(min_accepted_for_50split)
            keep_n = np.floor(n_accepted.astype(float) / 2.0).astype(np.int64)
            keep = ~eligible | (work["rank_low"].to_numpy(dtype=np.int64) < keep_n)
            threshold = int(min_accepted_for_50split)
        elif spec.kind == "high50":
            eligible = n_accepted >= int(min_accepted_for_50split)
            keep_n = np.floor(n_accepted.astype(float) / 2.0).astype(np.int64)
            keep = ~eligible | (work["rank_high"].to_numpy(dtype=np.int64) < keep_n)
            threshold = int(min_accepted_for_50split)
        elif spec.kind == "random50":
            eligible = n_accepted >= int(min_accepted_for_50split)
            keep_n = np.floor(n_accepted.astype(float) / 2.0).astype(np.int64)
            keep = ~eligible | (work["rank_random"].to_numpy(dtype=np.int64) < keep_n)
            threshold = int(min_accepted_for_50split)
        else:
            raise ValueError(f"Unknown variant kind: {spec.kind}")
        remove = ~keep
        work[f"remove_{spec.variant}"] = remove
        remove_mask_bits[remove] |= bit
        per_hkl = build_per_hkl_summary(work, spec, score_column, threshold)
        per_hkl_tables[spec.variant] = per_hkl
        kept = ~remove
        zero = work["zero_score"].to_numpy(dtype=bool)
        positive = work["positive_score"].to_numpy(dtype=bool)
        variant_rows.append(
            {
                "variant": spec.variant,
                "keep_fraction": spec.keep_fraction,
                "split_kind": spec.kind,
                "accepted_kept": int(kept.sum()),
                "accepted_removed": int(remove.sum()),
                "fraction_accepted_removed": float(remove.sum() / max(len(work), 1)),
                "zero_score_accepted_total": int(zero.sum()),
                "zero_score_kept": int((zero & kept).sum()),
                "zero_score_removed": int((zero & remove).sum()),
                "positive_score_accepted_total": int(positive.sum()),
                "positive_score_kept": int((positive & kept).sum()),
                "positive_score_removed": int((positive & remove).sum()),
                "signed_hkls_total": int(per_hkl.shape[0]),
                "signed_hkls_changed": int(per_hkl["changed"].sum()),
                "signed_hkls_low_count_unchanged": int(per_hkl["low_count_unchanged"].sum()),
                "min_accepted_after_filter": int(per_hkl["n_kept"].min()) if not per_hkl.empty else 0,
            }
        )

    selection_stats = {
        "accepted_v5_rows_after_duplicate_cleanup": int(len(work)),
        "duplicate_accepted_v5_rows": duplicate_rows,
        "duplicate_accepted_v5_keys": duplicate_keys,
        "unique_signed_hkls_total": int(work.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]),
        "zero_score_accepted_total": int(work["zero_score"].sum()),
        "positive_score_accepted_total": int(work["positive_score"].sum()),
        "seed": int(seed),
        "rules": [
            "Uses existing v5 score CSV only; does not recompute observation scores.",
            "Exact observation key is source_filename + normalized event + signed h,k,l.",
            "Signed HKLs are preserved exactly; no 4/mmm canonicalization is applied.",
            "Zero-score observations are included in all-score ranking and are valid low-risk observations.",
            "Only accepted v5-matched observations are ranked for removal/keeping.",
            "Nonaccepted or unmatched stream observations are kept unchanged in every output stream.",
        ],
    }
    return work, remove_mask_bits, per_hkl_tables, pd.DataFrame.from_records(variant_rows), selection_stats


def write_key_tables(paths: dict[str, dict[str, Path]], work: pd.DataFrame, specs: list[VariantSpec], score_column: str) -> dict[str, dict[str, int]]:
    key_stats: dict[str, dict[str, int]] = {}
    columns = [*KEY_COLUMNS, score_column, "zero_score", "positive_score", "tie_break", "n_accepted", "rank_low", "rank_high", "rank_random"]
    for spec in specs:
        remove = work[f"remove_{spec.variant}"].to_numpy(dtype=bool)
        kept = work.loc[~remove, columns].copy()
        removed = work.loc[remove, columns].copy()
        kept.to_csv(paths[spec.variant]["kept_keys_csv"], index=False)
        removed.to_csv(paths[spec.variant]["removed_keys_csv"], index=False)
        key_stats[spec.variant] = {"kept_key_rows": int(len(kept)), "removed_key_rows": int(len(removed))}
    return key_stats


def write_stream_variants(
    stream_path: Path,
    paths: dict[str, dict[str, Path]],
    specs: list[VariantSpec],
    key_to_mask: dict[tuple[str, str, int, int, int], int],
    progress_every: int,
    max_events: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variant_bits = {spec.variant: 1 << idx for idx, spec in enumerate(specs)}
    handles = {spec.variant: paths[spec.variant]["stream"].open("w", encoding="utf-8") for spec in specs}
    stats = {
        spec.variant: {
            "total_stream_observations_seen": 0,
            "accepted_v5_matched_observations": 0,
            "nonaccepted_or_unmatched_observations_kept": 0,
            "stream_kept_observations": 0,
            "stream_removed_observations": 0,
        }
        for spec in specs
    }

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
                        else:
                            unmatched_observations += 1
                        for spec in specs:
                            row = stats[spec.variant]
                            row["total_stream_observations_seen"] += 1
                            if matched:
                                row["accepted_v5_matched_observations"] += 1
                            else:
                                row["nonaccepted_or_unmatched_observations_kept"] += 1
                            remove = matched and bool(int(mask) & int(variant_bits[spec.variant]))
                            if remove:
                                row["stream_removed_observations"] += 1
                            else:
                                handles[spec.variant].write(raw_line)
                                row["stream_kept_observations"] += 1
                        if stream_observations_seen % int(progress_every) == 0:
                            removed_text = ", ".join(
                                f"{spec.variant} removed={stats[spec.variant]['stream_removed_observations']:,}" for spec in specs
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

    rows = []
    for spec in specs:
        row = dict(stats[spec.variant])
        row["variant"] = spec.variant
        row["output_stream"] = str(paths[spec.variant]["stream"])
        rows.append(row)
    stream_stats = {
        "chunks_seen": int(chunks_seen),
        "crystals_seen": int(min(crystals_seen, int(max_events)) if max_events is not None else crystals_seen),
        "stream_observations_seen": int(stream_observations_seen),
        "stream_accepted_v5_matched_observations": int(matched_observations),
        "stream_nonaccepted_or_unmatched_observations": int(unmatched_observations),
    }
    return pd.DataFrame.from_records(rows), stream_stats


def hkl_mask(table: pd.DataFrame, hkl: tuple[int, int, int]) -> pd.Series:
    h, k, l = hkl
    return (table["h"] == h) & (table["k"] == k) & (table["l"] == l)


def selected_hkl_summary(work: pd.DataFrame, specs: list[VariantSpec], score_column: str) -> pd.DataFrame:
    rows = []
    for hkl in SELECTED_HKLS:
        base = work.loc[hkl_mask(work, hkl)].copy()
        for spec in specs:
            remove_column = f"remove_{spec.variant}"
            if base.empty:
                kept = base
                removed = base
            else:
                kept = base.loc[~base[remove_column].to_numpy(dtype=bool)]
                removed = base.loc[base[remove_column].to_numpy(dtype=bool)]
            kept_stats = score_stats(kept[score_column] if not kept.empty else pd.Series(dtype=float))
            removed_stats = score_stats(removed[score_column] if not removed.empty else pd.Series(dtype=float))
            rows.append(
                {
                    "variant": spec.variant,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "n_accepted_before": int(len(base)),
                    "n_kept": int(len(kept)),
                    "n_removed": int(len(removed)),
                    "n_zero_before": int(base["zero_score"].sum()) if not base.empty else 0,
                    "n_zero_kept": int(kept["zero_score"].sum()) if not kept.empty else 0,
                    "n_zero_removed": int(removed["zero_score"].sum()) if not removed.empty else 0,
                    "score_min_kept": kept_stats["min"],
                    "score_median_kept": kept_stats["median"],
                    "score_max_kept": kept_stats["max"],
                    "score_min_removed": removed_stats["min"],
                    "score_median_removed": removed_stats["median"],
                    "score_max_removed": removed_stats["max"],
                }
            )
    return pd.DataFrame.from_records(rows)


def write_summaries(
    output_root: Path,
    specs: list[VariantSpec],
    stream_summary: pd.DataFrame,
    variant_selection_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
) -> pd.DataFrame:
    summary = stream_summary.merge(variant_selection_summary, on="variant", how="left")
    columns = [
        "variant",
        "total_stream_observations_seen",
        "accepted_v5_matched_observations",
        "nonaccepted_or_unmatched_observations_kept",
        "accepted_kept",
        "accepted_removed",
        "fraction_accepted_removed",
        "zero_score_accepted_total",
        "zero_score_kept",
        "zero_score_removed",
        "positive_score_accepted_total",
        "positive_score_kept",
        "positive_score_removed",
        "signed_hkls_total",
        "signed_hkls_changed",
        "signed_hkls_low_count_unchanged",
        "min_accepted_after_filter",
        "output_stream",
    ]
    summary = summary.loc[:, columns]
    summary.to_csv(output_root / "v5_allscore_filter_and_50split_summary.csv", index=False)
    selected_summary.to_csv(output_root / "v5_allscore_selected_hkl_summary.csv", index=False)

    lines = [
        "# V5 All-Score Accepted-Observation Filter and 50-Split Summary",
        "",
        "This workflow uses the existing v5 observation score table only. It does not recompute scores, run partialator, merge, or refine.",
        "",
        "## Global Summary",
        "| variant | accepted kept | accepted removed | fraction removed | zero kept | zero removed | positive kept | positive removed | changed HKLs | min accepted after |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {int(row.accepted_kept):,} | {int(row.accepted_removed):,} | "
            f"{float(row.fraction_accepted_removed):.6g} | {int(row.zero_score_kept):,} | {int(row.zero_score_removed):,} | "
            f"{int(row.positive_score_kept):,} | {int(row.positive_score_removed):,} | {int(row.signed_hkls_changed):,} | "
            f"{int(row.min_accepted_after_filter):,} |"
        )
    lines.extend(
        [
            "",
            "## Selected HKLs",
            "| variant | h | k | l | n before | n kept | n removed | zero before | zero kept | zero removed | kept median | removed median |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in selected_summary.itertuples(index=False):
        kept_median = "" if pd.isna(row.score_median_kept) else f"{float(row.score_median_kept):.6g}"
        removed_median = "" if pd.isna(row.score_median_removed) else f"{float(row.score_median_removed):.6g}"
        lines.append(
            f"| {row.variant} | {int(row.h)} | {int(row.k)} | {int(row.l)} | {int(row.n_accepted_before)} | "
            f"{int(row.n_kept)} | {int(row.n_removed)} | {int(row.n_zero_before)} | {int(row.n_zero_kept)} | "
            f"{int(row.n_zero_removed)} | {kept_median} | {removed_median} |"
        )
    lines.extend(["", "## Output Streams"])
    for spec in specs:
        stream_path = summary.loc[summary["variant"] == spec.variant, "output_stream"].iloc[0]
        lines.append(f"- `{spec.variant}`: `{stream_path}`")
    (output_root / "v5_allscore_filter_and_50split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_run_metadata(
    output_root: Path,
    args: argparse.Namespace,
    specs: list[VariantSpec],
    accepted_key_stats: dict[str, Any],
    score_load_stats: dict[str, Any],
    selection_stats: dict[str, Any],
    stream_stats: dict[str, Any],
    key_table_stats: dict[str, dict[str, int]],
) -> None:
    payload = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "v5_scores": str(args.v5_scores),
            "accepted_scores": str(args.accepted_scores),
        },
        "output_root": str(output_root),
        "score_column": str(args.score_column),
        "keep_fractions": [float(value) for value in args.keep_fractions],
        "min_accepted_keep": int(args.min_accepted_keep),
        "min_accepted_for_50split": int(args.min_accepted_for_50split),
        "seed": int(args.seed),
        "max_events": None if args.max_events is None else int(args.max_events),
        "variants": [spec.__dict__ for spec in specs],
        "accepted_key_stats": accepted_key_stats,
        "score_load_stats": score_load_stats,
        "selection_stats": selection_stats,
        "stream_stats": stream_stats,
        "key_table_stats": key_table_stats,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_dry_run(args: argparse.Namespace, specs: list[VariantSpec], paths: dict[str, dict[str, Path]]) -> None:
    print("Dry run: no key joins, score ranking, stream rewriting, or output writing will run.")
    print(f"Output root: {args.output_root}")
    for spec in specs:
        print(f"stream: {paths[spec.variant]['stream']}")
        print(f"kept keys: {paths[spec.variant]['kept_keys_csv']}")
        print(f"removed keys: {paths[spec.variant]['removed_keys_csv']}")
        print(f"per-HKL summary: {paths[spec.variant]['per_hkl_summary_csv']}")
    print(f"global summary CSV: {args.output_root / 'v5_allscore_filter_and_50split_summary.csv'}")
    print(f"global summary markdown: {args.output_root / 'v5_allscore_filter_and_50split_summary.md'}")
    print(f"selected-HKL summary CSV: {args.output_root / 'v5_allscore_selected_hkl_summary.csv'}")


def main() -> int:
    args = parse_args()
    specs = variant_specs(args.keep_fractions, int(args.seed))
    paths = output_paths(args.output_root, specs)
    if args.dry_run:
        print_dry_run(args, specs, paths)
        return 0
    ensure_outputs(args.output_root, paths, bool(args.overwrite))

    log(f"Output root: {args.output_root}")
    smoke_key_filter = None
    smoke_stats: dict[str, int] = {}
    if args.max_events is not None:
        log(f"Collecting smoke-test keys for first {int(args.max_events)} crystals")
        smoke_key_filter, smoke_stats = collect_smoke_keys(args.stream, int(args.max_events), int(args.progress_every))
        log(f"Smoke key collection done: keys={len(smoke_key_filter):,}")

    log("Loading P1 iter1 accepted observation keys")
    accepted_keys, accepted_key_stats = load_accepted_keys(args.accepted_scores, smoke_key_filter, int(args.scores_chunksize))
    accepted_key_stats.update(smoke_stats)
    log(f"Loaded accepted keys: unique_keys={len(accepted_keys):,}")

    log("Joining accepted keys to existing v5 score table")
    accepted_v5, score_load_stats = load_accepted_v5_scores(args.v5_scores, accepted_keys, str(args.score_column), int(args.scores_chunksize))
    log(f"Loaded accepted v5 scores: rows={len(accepted_v5):,}")

    log("Building all-score low-risk filters and 50% split selections")
    selection_table, remove_mask_bits, per_hkl_tables, variant_selection_summary, selection_stats = build_selection_tables(
        accepted_v5,
        specs,
        str(args.score_column),
        int(args.min_accepted_keep),
        int(args.min_accepted_for_50split),
        int(args.seed),
    )
    for spec in specs:
        per_hkl_tables[spec.variant].to_csv(paths[spec.variant]["per_hkl_summary_csv"], index=False)

    log("Writing accepted key tables")
    key_table_stats = write_key_tables(paths, selection_table, specs, str(args.score_column))

    log("Preparing stream removal lookup")
    key_to_mask: dict[tuple[str, str, int, int, int], int] = {}
    for row, mask in zip(selection_table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), remove_mask_bits, strict=True):
        source, event, h, k, l = row
        key_to_mask[(str(source), str(event), int(h), int(k), int(l))] = int(mask)

    log("Writing all output stream variants in one pass over the original stream")
    stream_summary, stream_stats = write_stream_variants(
        args.stream,
        paths,
        specs,
        key_to_mask,
        int(args.progress_every),
        None if args.max_events is None else int(args.max_events),
    )

    selected_summary = selected_hkl_summary(selection_table, specs, str(args.score_column))
    global_summary = write_summaries(args.output_root, specs, stream_summary, variant_selection_summary, selected_summary)
    write_run_metadata(args.output_root, args, specs, accepted_key_stats, score_load_stats, selection_stats, stream_stats, key_table_stats)

    log("V5 all-score filter and 50-split workflow complete")
    for spec in specs:
        print(f"Wrote: {paths[spec.variant]['stream']}")
        print(f"Wrote: {paths[spec.variant]['kept_keys_csv']}")
        print(f"Wrote: {paths[spec.variant]['removed_keys_csv']}")
        print(f"Wrote: {paths[spec.variant]['per_hkl_summary_csv']}")
    print(f"Wrote: {args.output_root / 'v5_allscore_filter_and_50split_summary.csv'}")
    print(f"Wrote: {args.output_root / 'v5_allscore_filter_and_50split_summary.md'}")
    print(f"Wrote: {args.output_root / 'v5_allscore_selected_hkl_summary.csv'}")
    print(f"Wrote: {args.output_root / 'run_metadata.json'}")
    print(global_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())