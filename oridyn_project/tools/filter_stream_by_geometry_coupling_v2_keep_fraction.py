#!/usr/bin/env python3
"""Filter a CrystFEL stream by a geometry-coupling v2 norm score column.

This is a separate v2 stream-preparation tool. It does not use or modify the
old successful geometry-trust score pipeline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
DEFAULT_KEEP_FRACTIONS = [0.90, 0.80]
DEFAULT_SCORE_COLUMN = "trust_risk_v2_full_norm"
DEFAULT_PROGRESS_EVERY = 1_000_000
SCORE_CHUNKSIZE = 500_000

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="Original CrystFEL stream")
    parser.add_argument("--v2-scores-csv", required=True, type=Path, help="Output from compute_geometry_coupling_v2_scores.py")
    parser.add_argument("--output-root", required=True, type=Path, help="Fresh v2 filtering output directory")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN, help="V2 norm column to filter by")
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--min-obs-per-hkl", type=int, default=10)
    parser.add_argument(
        "--keep-low-count-hkls",
        action="store_true",
        help="Keep matched HKLs with fewer than --min-obs-per-hkl observations unchanged",
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--max-events", type=int, default=None, help="Smoke-test limit on stream crystal blocks/events")
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.v2_scores_csv.exists():
        raise SystemExit(f"--v2-scores-csv not found: {args.v2_scores_csv}")
    if args.min_obs_per_hkl < 1:
        raise SystemExit("--min-obs-per-hkl must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.max_events is not None and args.max_events < 1:
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
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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


def score_label(score_column: str) -> str:
    label = str(score_column)
    for prefix in ("trust_risk_", "manybeam_coupling_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    for suffix in ("_norm", "_raw"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_") or "score"


def variant_name(score_column: str, fraction: float) -> str:
    return f"geometry_coupling_{score_label(score_column)}_keep{percent_label(fraction)}"


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in columns]
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
    return out


def filter_chunk_to_keys(chunk: pd.DataFrame, key_filter: set[tuple[str, str, int, int, int]]) -> pd.DataFrame:
    if chunk.empty:
        return chunk
    keys = [
        build_key(source, event, int(h), int(k), int(l))
        for source, event, h, k, l in chunk.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    return chunk.loc[[key in key_filter for key in keys]].copy()


def load_v2_scores(
    scores_path: Path,
    score_column: str,
    key_filter: set[tuple[str, str, int, int, int]] | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    header = pd.read_csv(scores_path, nrows=0).columns.tolist()
    require_columns(header, [*KEY_COLUMNS, score_column], "v2 scores CSV")
    usecols = [*KEY_COLUMNS, score_column]
    stats = {
        "score_rows_read": 0,
        "score_rows_after_cleanup": 0,
        "score_rows_after_smoke_key_filter": 0,
    }
    if key_filter is None:
        raw = pd.read_csv(scores_path, usecols=usecols)
        stats["score_rows_read"] = int(len(raw))
        table = normalize_score_chunk(raw, score_column)
        stats["score_rows_after_cleanup"] = int(len(table))
    else:
        chunks = []
        for idx, chunk in enumerate(pd.read_csv(scores_path, usecols=usecols, chunksize=SCORE_CHUNKSIZE), start=1):
            stats["score_rows_read"] += int(len(chunk))
            normalized = normalize_score_chunk(chunk, score_column)
            stats["score_rows_after_cleanup"] += int(len(normalized))
            filtered = filter_chunk_to_keys(normalized, key_filter)
            if not filtered.empty:
                chunks.append(filtered)
                stats["score_rows_after_smoke_key_filter"] += int(len(filtered))
            if idx % 10 == 0:
                log(
                    "V2 score CSV scan: "
                    f"rows_read={stats['score_rows_read']:,}, "
                    f"matched_smoke_rows={stats['score_rows_after_smoke_key_filter']:,}"
                )
        table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    return table, stats


def build_filter_masks(
    scored: pd.DataFrame,
    score_column: str,
    keep_fractions: list[float],
    min_obs_per_hkl: int,
    keep_low_count_hkls: bool,
) -> tuple[dict[tuple[str, str, int, int, int], int], pd.DataFrame, dict[str, Any]]:
    work = scored.sort_values([*HKL_COLUMNS, score_column, "source_filename", "event"], kind="mergesort").reset_index(drop=True)
    duplicated = work.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(duplicated.sum())
    duplicate_keys = int(work.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact observation keys in v2 scores; keeping first row per key")
        work = work.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    grouped = work.groupby(HKL_COLUMNS, sort=False)
    work["_n_total"] = grouped[score_column].transform("size").astype("int64")
    work["_risk_rank"] = grouped.cumcount().astype("int64")
    work["_low_count_hkl"] = work["_n_total"] < int(min_obs_per_hkl)

    remove_mask_bits = np.zeros(len(work), dtype=np.uint16)
    variant_rows = []
    for bit_idx, keep_fraction in enumerate(keep_fractions):
        bit = np.uint16(1 << bit_idx)
        n_keep = np.floor(work["_n_total"].to_numpy(dtype=float) * float(keep_fraction)).astype("int64")
        keep = work["_risk_rank"].to_numpy(dtype=np.int64) < n_keep
        if keep_low_count_hkls:
            keep = np.where(work["_low_count_hkl"].to_numpy(dtype=bool), True, keep)
        remove = ~keep
        remove_mask_bits[remove] |= bit
        variant_rows.append(
            {
                "variant": variant_name(score_column, keep_fraction),
                "score_column": score_column,
                "keep_fraction": float(keep_fraction),
                "keep_percent_label": percent_label(keep_fraction),
                "scored_rows_removed_by_selection": int(remove.sum()),
                "scored_rows_kept_by_selection": int((~remove).sum()),
                "scored_fraction_removed_by_selection": float(remove.mean()) if len(remove) else 0.0,
            }
        )

    key_to_mask: dict[tuple[str, str, int, int, int], int] = {}
    for row, mask in zip(work.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), remove_mask_bits, strict=True):
        source, event, h, k, l = row
        key_to_mask[(str(source), str(event), int(h), int(k), int(l))] = int(mask)

    stats = {
        "scored_hkls": int(work[HKL_COLUMNS].drop_duplicates().shape[0]),
        "low_count_hkls": int(work.loc[work["_low_count_hkl"], HKL_COLUMNS].drop_duplicates().shape[0]),
        "low_count_hkls_kept_unchanged": int(work.loc[work["_low_count_hkl"], HKL_COLUMNS].drop_duplicates().shape[0])
        if keep_low_count_hkls
        else 0,
        "duplicate_score_rows": duplicate_rows,
        "duplicate_score_keys": duplicate_keys,
        "key_to_mask_entries": int(len(key_to_mask)),
    }
    return key_to_mask, pd.DataFrame.from_records(variant_rows), stats


def output_paths(output_root: Path, score_column: str, keep_fractions: list[float]) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for fraction in keep_fractions:
        name = variant_name(score_column, fraction)
        paths[name] = {
            "stream": output_root / f"{name}.stream",
            "summary_json": output_root / f"{name}_summary.json",
            "removed_by_hkl_csv": output_root / f"{name}_removed_by_hkl.csv",
        }
    return paths


def ensure_no_overwrite(output_root: Path, paths: dict[str, dict[str, Path]]) -> None:
    blocked = []
    for variant_paths in paths.values():
        for path in variant_paths.values():
            if path.exists():
                blocked.append(path)
    for name in ["filter_sweep_summary.csv", "run_metadata.json"]:
        path = output_root / name
        if path.exists():
            blocked.append(path)
    if blocked:
        formatted = "\n".join(f"  {path}" for path in blocked[:20])
        more = "" if len(blocked) <= 20 else f"\n  ... and {len(blocked) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}{more}")
    output_root.mkdir(parents=True, exist_ok=True)


def write_stream_variants(
    stream_path: Path,
    paths: dict[str, dict[str, Path]],
    key_to_mask: dict[tuple[str, str, int, int, int], int],
    score_column: str,
    keep_fractions: list[float],
    progress_every: int,
    max_events: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[tuple[int, int, int], int]], dict[tuple[int, int, int], int]]:
    variants = [variant_name(score_column, fraction) for fraction in keep_fractions]
    variant_bits = {variant: 1 << idx for idx, variant in enumerate(variants)}
    handles = {variant: paths[variant]["stream"].open("w", encoding="utf-8") for variant in variants}
    stats = {
        variant: {
            "total_observations_seen": 0,
            "matched_observations": 0,
            "unmatched_observations": 0,
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
                            stats[variant]["total_observations_seen"] += 1
                            if matched:
                                stats[variant]["matched_observations"] += 1
                            else:
                                stats[variant]["unmatched_observations"] += 1
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
                                f"matched={matched_observations:,}, unmatched={unmatched_observations:,}, "
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
        "stream_matched_observations": int(matched_observations),
        "stream_unmatched_observations": int(unmatched_observations),
    }
    rows = []
    for variant, fraction in zip(variants, keep_fractions, strict=True):
        row = dict(stats[variant])
        row["variant"] = variant
        row["score_column"] = score_column
        row["keep_fraction"] = float(fraction)
        row["removed_fraction"] = row["removed_observations"] / max(row["total_observations_seen"], 1)
        row["removed_fraction_of_matched"] = row["removed_observations"] / max(row["matched_observations"], 1)
        row["output_stream_path"] = str(paths[variant]["stream"])
        rows.append(row)
    return pd.DataFrame.from_records(rows), stream_stats, removed_by_hkl, matched_hkl_counts


def write_removed_by_hkl_csvs(
    paths: dict[str, dict[str, Path]],
    removed_by_hkl: dict[str, dict[tuple[int, int, int], int]],
    matched_hkl_counts: dict[tuple[int, int, int], int],
) -> dict[str, int]:
    affected_counts = {}
    for variant, counts in removed_by_hkl.items():
        rows = []
        for hkl, n_removed in counts.items():
            n_matched = int(matched_hkl_counts.get(hkl, 0))
            h, k, l = hkl
            rows.append(
                {
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "matched_observations_for_hkl": n_matched,
                    "removed_observations": int(n_removed),
                    "kept_matched_observations": int(max(n_matched - int(n_removed), 0)),
                    "removed_fraction_of_matched_hkl": float(int(n_removed) / max(n_matched, 1)),
                }
            )
        table = pd.DataFrame.from_records(rows)
        if not table.empty:
            table = table.sort_values(["removed_observations", "h", "k", "l"], ascending=[False, True, True, True])
        else:
            table = pd.DataFrame(
                columns=[
                    "h",
                    "k",
                    "l",
                    "matched_observations_for_hkl",
                    "removed_observations",
                    "kept_matched_observations",
                    "removed_fraction_of_matched_hkl",
                ]
            )
        table.to_csv(paths[variant]["removed_by_hkl_csv"], index=False)
        affected_counts[variant] = int(len(table))
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
                "v2_scores_csv": str(args.v2_scores_csv),
            },
            "score_column": str(args.score_column),
            "selection": {
                "min_obs_per_hkl": int(args.min_obs_per_hkl),
                "keep_low_count_hkls": bool(args.keep_low_count_hkls),
            },
            "selection_stats": selection_stats,
            "stream_stats": stream_stats,
            "variant_summary": rows.get(variant, {}),
            "outputs": {name: str(path) for name, path in variant_paths.items()},
        }
        variant_paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_metadata(
    output_root: Path,
    args: argparse.Namespace,
    score_load_stats: dict[str, int],
    selection_stats: dict[str, Any],
    stream_stats: dict[str, Any],
) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "v2_scores_csv": str(args.v2_scores_csv),
        },
        "output_root": str(output_root),
        "score_column": str(args.score_column),
        "keep_fractions": [float(value) for value in args.keep_fractions],
        "min_obs_per_hkl": int(args.min_obs_per_hkl),
        "keep_low_count_hkls": bool(args.keep_low_count_hkls),
        "progress_every": int(args.progress_every),
        "max_events": None if args.max_events is None else int(args.max_events),
        "score_load_stats": score_load_stats,
        "selection_stats": selection_stats,
        "stream_stats": stream_stats,
        "warnings": [
            "This is cSerialED geometry-coupling v2 filtering, not an intensity correction.",
            "Observation matching uses exact source_filename + event + signed h,k,l.",
            "HKLs are not canonicalized.",
            "Unmatched stream observations are kept by default and counted.",
        ],
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = output_paths(args.output_root, args.score_column, args.keep_fractions)
    ensure_no_overwrite(args.output_root, paths)
    log(f"Output root: {args.output_root}")
    for fraction in args.keep_fractions:
        name = variant_name(args.score_column, fraction)
        log(f"Prepared output stream for keep_fraction={fraction:.3f}: {paths[name]['stream']}")

    smoke_key_filter = None
    smoke_stats: dict[str, int] = {}
    if args.max_events is not None:
        log(f"Collecting smoke-test keys for first {int(args.max_events)} stream events/crystals")
        smoke_key_filter, smoke_stats = collect_smoke_keys(args.stream, int(args.max_events), int(args.progress_every))
        log(f"Smoke key collection done: keys={len(smoke_key_filter):,}")

    log("Loading v2 score table")
    score_table, score_load_stats = load_v2_scores(args.v2_scores_csv, args.score_column, smoke_key_filter)
    score_load_stats.update(smoke_stats)
    log(f"Loaded v2 scores: rows={len(score_table):,}")

    log("Selecting per-HKL lowest-risk observations for each keep fraction")
    key_to_mask, variant_selection, selection_stats = build_filter_masks(
        score_table,
        str(args.score_column),
        args.keep_fractions,
        int(args.min_obs_per_hkl),
        bool(args.keep_low_count_hkls),
    )
    log(
        "Selection ready: "
        f"scored_hkls={selection_stats['scored_hkls']:,}, "
        f"low_count_hkls={selection_stats['low_count_hkls']:,}, "
        f"lookup_keys={selection_stats['key_to_mask_entries']:,}"
    )

    log("Writing filtered v2 stream variants in one stream pass")
    sweep_summary, stream_stats, removed_by_hkl, matched_hkl_counts = write_stream_variants(
        args.stream,
        paths,
        key_to_mask,
        str(args.score_column),
        args.keep_fractions,
        int(args.progress_every),
        None if args.max_events is None else int(args.max_events),
    )
    affected_counts = write_removed_by_hkl_csvs(paths, removed_by_hkl, matched_hkl_counts)
    sweep_summary["number_of_signed_hkls_affected"] = sweep_summary["variant"].map(affected_counts).astype("int64")
    sweep_summary["number_of_low_count_hkls_kept_unchanged"] = int(selection_stats["low_count_hkls_kept_unchanged"])
    sweep_summary = sweep_summary.merge(variant_selection, on=["variant", "score_column", "keep_fraction"], how="left")
    sweep_summary.to_csv(args.output_root / "filter_sweep_summary.csv", index=False)
    write_variant_jsons(paths, sweep_summary, stream_stats, selection_stats, args)
    write_run_metadata(args.output_root, args, score_load_stats, selection_stats, stream_stats)

    log("V2 filtering sweep complete")
    for variant_paths in paths.values():
        print(f"Wrote: {variant_paths['stream']}")
        print(f"Wrote: {variant_paths['summary_json']}")
        print(f"Wrote: {variant_paths['removed_by_hkl_csv']}")
    print(f"Wrote: {args.output_root / 'filter_sweep_summary.csv'}")
    print(f"Wrote: {args.output_root / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
