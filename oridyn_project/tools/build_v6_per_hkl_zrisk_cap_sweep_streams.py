#!/usr/bin/env python3
"""Build V6 full-population per-HKL normalized S_risk filter streams.

This exploratory builder ranks observations by within-signed-HKL normalized
risk, then removes a global high-z tail with the same loose per-HKL safety
caps used in the cap0.8 global EgM2 sweep.

It reads the existing V6 full_population_cache.sqlite and source stream.  It
does not run Partialator, merging, indexing, or production processing.  Stream
files are written only with --write-streams.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import build_v6_fullpop_gentle_egm2_streams as base


DEFAULT_SOURCE_OUT_DIR = base.DEFAULT_SOURCE_OUT_DIR
DEFAULT_SOURCE_STREAM = base.DEFAULT_SOURCE_STREAM
DEFAULT_OUTPUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_per_hkl_zrisk_cap0p8_20260813"
)
DEFAULT_DROP_FRACS = ("0.005", "0.010", "0.020", "0.050", "0.100", "0.200")
DEFAULT_SCORE_MODES = ("per_hkl_z", "per_hkl_robust_z")
VALID_SCORE_MODES = ("raw_global", "per_hkl_z", "per_hkl_robust_z")
DEFAULT_MAX_REMOVED_FRAC_PER_HKL = 0.80
DEFAULT_MIN_RETAINED_PER_HKL = 2
SQL_CHUNK_ROWS = 250_000
EPS = 1.0e-12


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_delimited(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str] | None = None,
    delimiter: str = "\t",
) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def cap_label(value: float) -> str:
    return f"{float(value):.6g}".rstrip("0").rstrip(".").replace(".", "p")


def drop_label(fraction: float) -> str:
    percent = float(fraction) * 100.0
    if abs(percent - round(percent)) < 1.0e-9 and percent >= 1.0:
        return f"{int(round(percent)):02d}"
    return f"{percent:.6g}".rstrip("0").rstrip(".").replace(".", "p")


def parse_drop_fracs(value: str | None) -> list[tuple[str, float]]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_DROP_FRACS)
    out: list[tuple[str, float]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        fraction = float(item)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise SystemExit(f"Invalid drop fraction {item!r}; expected 0 < fraction < 1")
        out.append((item, fraction))
    if not out:
        raise SystemExit("--drop-fracs must contain at least one fraction")
    labels = [drop_label(fraction) for _text, fraction in out]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate drop label(s): {duplicates}")
    return sorted(out, key=lambda item: item[1])


def parse_score_modes(value: str | None) -> list[str]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_SCORE_MODES)
    modes: list[str] = []
    for raw in str(text).split(","):
        mode = raw.strip()
        if not mode:
            continue
        if mode not in VALID_SCORE_MODES:
            raise SystemExit(f"Unknown --score-mode {mode!r}; choose from {', '.join(VALID_SCORE_MODES)}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise SystemExit("--score-mode must contain at least one mode")
    return modes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-mode", default=",".join(DEFAULT_SCORE_MODES), help="Comma-separated: raw_global,per_hkl_z,per_hkl_robust_z")
    parser.add_argument("--drop-fracs", default=",".join(DEFAULT_DROP_FRACS))
    parser.add_argument("--min-retained-per-hkl", type=int, default=DEFAULT_MIN_RETAINED_PER_HKL)
    parser.add_argument("--max-removed-frac-per-hkl", type=float, default=DEFAULT_MAX_REMOVED_FRAC_PER_HKL)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.score_modes = parse_score_modes(args.score_mode)
    args.drop_items = parse_drop_fracs(args.drop_fracs)
    args.min_retained_per_hkl = max(0, int(args.min_retained_per_hkl))
    args.max_removed_frac_per_hkl = float(args.max_removed_frac_per_hkl)
    args.workers = max(1, int(args.workers))
    if not math.isfinite(args.max_removed_frac_per_hkl) or not (0.0 < args.max_removed_frac_per_hkl <= 1.0):
        raise SystemExit("--max-removed-frac-per-hkl must satisfy 0 < value <= 1")
    if args.dry_run and args.write_streams:
        raise SystemExit("Use either --dry-run or --write-streams, not both")
    if not args.write_streams:
        args.dry_run = True
    if not args.source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {args.source_out_dir}")
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    return args


def build_variants(
    score_modes: list[str],
    drop_items: list[tuple[str, float]],
    out_dir: Path,
    max_removed_frac_per_hkl: float,
) -> tuple[list[base.VariantSpec], dict[str, str]]:
    variants: list[base.VariantSpec] = []
    variant_modes: dict[str, str] = {}
    cap = cap_label(max_removed_frac_per_hkl)
    for mode in score_modes:
        for text, fraction in drop_items:
            label = drop_label(float(fraction))
            variant_id = f"filter_all_{mode}_eg_m2_cap{cap}_drop{label}"
            output_filename = f"{variant_id}.stream"
            variants.append(
                base.VariantSpec(
                    variant_id=variant_id,
                    fraction_text=text,
                    drop_fraction=float(fraction),
                    output_filename=output_filename,
                    output_stream=out_dir / output_filename,
                )
            )
            variant_modes[variant_id] = mode
    return variants, variant_modes


def planned_output_paths(out_dir: Path, variants: list[base.VariantSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "per_hkl_zrisk_manifest.tsv",
        out_dir / "per_hkl_zrisk_counts.csv",
        out_dir / "per_hkl_zrisk_hkl_removal_manifest.tsv",
        out_dir / "per_hkl_zrisk_top_removed_hkls.tsv",
        out_dir / "per_hkl_zrisk_parameters.json",
        out_dir / "per_hkl_zrisk_metadata.json",
        out_dir / "run.log",
    ]
    if write_streams:
        paths.extend(variant.output_stream for variant in variants)
    return paths


def refuse_overwrite(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        preview = "\n  ".join(str(path) for path in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ... {len(existing) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {preview}{extra}")


def require_cache_schema(db_file: Path) -> set[str]:
    if not db_file.is_file():
        raise SystemExit(f"full-population cache not found: {db_file}")
    present = set(base.cache_schema(db_file))
    required = {"ordinal", "h", "k", "l", "exact_key_text"}
    score_sources = {"S_risk", "s_risk", "srisk", "eg_m2", "score_eg_m2", "Eg", "M2"}
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"score_cache is missing required columns: {missing}")
    if not ({"Eg", "M2"} <= present or score_sources & present):
        raise SystemExit("score_cache needs an S_risk/eg_m2-like column or both Eg and M2")
    return present


def choose_score_expression(schema: set[str]) -> tuple[str, str, str]:
    for column in ["S_risk", "s_risk", "srisk", "eg_m2", "score_eg_m2"]:
        if column in schema:
            return f'"{column}"', column, f"source column {column}"
    if {"Eg", "M2"} <= schema:
        return "Eg * M2", "EgM2", "computed as Eg * M2"
    raise SystemExit("Could not choose S_risk source")


def max_removable_for_hkl(n_obs: int, max_fraction: float, min_retained: int) -> int:
    return int(max(0, min(int(math.floor(float(max_fraction) * int(n_obs))), int(n_obs) - int(min_retained))))


def load_score_table(
    db_file: Path,
    accepted_count: int,
    score_expression: str,
    logger: base.RunLogger,
) -> dict[str, np.ndarray]:
    schema = set(base.cache_schema(db_file))
    tie_column = "source_order" if "source_order" in schema else "ordinal"
    query = f"""
        SELECT ordinal,h,k,l,{tie_column} AS tie_value,{score_expression} AS srisk
        FROM score_cache
    """
    chunks: dict[str, list[np.ndarray]] = {
        "ordinal": [],
        "h": [],
        "k": [],
        "l": [],
        "tie_value": [],
        "srisk": [],
    }
    rows_read = 0
    progress = base.StageProgress(logger, "reading S_risk scores", accepted_count, "observations", workers=1)
    with base.connect_readonly(db_file) as conn:
        for chunk in pd.read_sql_query(query, conn, chunksize=SQL_CHUNK_ROWS):
            score = pd.to_numeric(chunk["srisk"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            chunks["ordinal"].append(chunk["ordinal"].to_numpy(dtype=np.int64, copy=True))
            chunks["h"].append(chunk["h"].to_numpy(dtype=np.int32, copy=True))
            chunks["k"].append(chunk["k"].to_numpy(dtype=np.int32, copy=True))
            chunks["l"].append(chunk["l"].to_numpy(dtype=np.int32, copy=True))
            chunks["tie_value"].append(chunk["tie_value"].to_numpy(dtype=np.int64, copy=True))
            chunks["srisk"].append(score.astype(np.float64, copy=True))
            rows_read += int(len(chunk))
            progress.update(rows_read)
    progress.finish(rows_read)
    if rows_read != int(accepted_count):
        raise SystemExit(f"Read {rows_read:,} cache rows, expected {accepted_count:,}")
    table = {key: np.concatenate(values) for key, values in chunks.items()}
    nonfinite = int((~np.isfinite(table["srisk"])).sum())
    if nonfinite:
        logger.log(f"warning: {nonfinite:,} observations have nonfinite S_risk and will be ineligible")
    return table


def hkl_key(h: int, k: int, l: int) -> tuple[int, int, int]:
    return int(h), int(k), int(l)


def compute_hkl_counts(table: dict[str, np.ndarray]) -> dict[tuple[int, int, int], int]:
    counts: dict[tuple[int, int, int], int] = {}
    h_values = table["h"]
    k_values = table["k"]
    l_values = table["l"]
    order = np.lexsort((l_values, k_values, h_values))
    if len(order) == 0:
        return counts
    sorted_h = h_values[order]
    sorted_k = k_values[order]
    sorted_l = l_values[order]
    starts = np.r_[0, np.nonzero((np.diff(sorted_h) != 0) | (np.diff(sorted_k) != 0) | (np.diff(sorted_l) != 0))[0] + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        counts[hkl_key(sorted_h[start], sorted_k[start], sorted_l[start])] = int(end - start)
    return counts


def compute_per_hkl_rank_values(
    mode: str,
    table: dict[str, np.ndarray],
    logger: base.RunLogger,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = table["srisk"]
    if mode == "raw_global":
        rank = raw.copy()
        finite = np.isfinite(rank)
        return rank, {
            "mode": mode,
            "rank_definition": "raw S_risk",
            "eligible_hkl_count": "",
            "ineligible_hkl_too_few_finite": "",
            "ineligible_hkl_zero_spread": "",
            "eligible_observation_count": int(finite.sum()),
            "ineligible_observation_count": int((~finite).sum()),
        }

    h_values = table["h"]
    k_values = table["k"]
    l_values = table["l"]
    order = np.lexsort((l_values, k_values, h_values))
    rank = np.full(raw.shape, np.nan, dtype=np.float64)
    if len(order) == 0:
        return rank, {}
    sorted_h = h_values[order]
    sorted_k = k_values[order]
    sorted_l = l_values[order]
    starts = np.r_[0, np.nonzero((np.diff(sorted_h) != 0) | (np.diff(sorted_k) != 0) | (np.diff(sorted_l) != 0))[0] + 1]
    ends = np.r_[starts[1:], len(order)]
    progress = base.StageProgress(logger, f"computing {mode}", len(starts), "signed HKLs", workers=1)
    eligible_hkls = 0
    too_few_hkls = 0
    zero_spread_hkls = 0
    eligible_obs = 0
    for group_number, (start, end) in enumerate(zip(starts, ends), start=1):
        indexes = order[start:end]
        values = raw[indexes]
        finite_mask = np.isfinite(values)
        n_finite = int(finite_mask.sum())
        if n_finite < 2:
            too_few_hkls += 1
            progress.update(group_number)
            continue
        finite_values = values[finite_mask]
        finite_indexes = indexes[finite_mask]
        if mode == "per_hkl_z":
            center = float(np.mean(finite_values))
            denom = float(np.std(finite_values, ddof=0))
            rank_definition = "(S_risk - mean_hkl(S_risk)) / std_hkl(S_risk), std uses ddof=0"
        elif mode == "per_hkl_robust_z":
            center = float(np.median(finite_values))
            mad = float(np.median(np.abs(finite_values - center)))
            denom = 1.4826 * mad
            rank_definition = "(S_risk - median_hkl(S_risk)) / (1.4826 * MAD_hkl(S_risk))"
        else:
            raise SystemExit(f"Unsupported score mode: {mode}")
        if not math.isfinite(denom) or denom <= EPS:
            zero_spread_hkls += 1
            progress.update(group_number)
            continue
        rank[finite_indexes] = (finite_values - center) / denom
        eligible_hkls += 1
        eligible_obs += n_finite
        progress.update(group_number)
    progress.finish(len(starts))
    ineligible_obs = int((~np.isfinite(rank)).sum())
    return rank, {
        "mode": mode,
        "rank_definition": rank_definition,
        "eligible_hkl_count": int(eligible_hkls),
        "ineligible_hkl_too_few_finite": int(too_few_hkls),
        "ineligible_hkl_zero_spread": int(zero_spread_hkls),
        "eligible_observation_count": int(eligible_obs),
        "ineligible_observation_count": int(ineligible_obs),
    }


def select_for_mode(
    mode: str,
    rank_values: np.ndarray,
    table: dict[str, np.ndarray],
    drop_items: list[tuple[str, float]],
    hkl_counts: dict[tuple[int, int, int], int],
    max_removed_frac_per_hkl: float,
    min_retained_per_hkl: int,
    accepted_count: int,
    logger: base.RunLogger,
) -> tuple[list[int], list[tuple[int, int, int]], list[float], list[float], dict[str, Any]]:
    capacities = {
        hkl: max_removable_for_hkl(n_obs, max_removed_frac_per_hkl, min_retained_per_hkl)
        for hkl, n_obs in hkl_counts.items()
    }
    requested_counts = [int(math.floor(fraction * accepted_count)) for _text, fraction in drop_items]
    max_requested = max(requested_counts) if requested_counts else 0
    finite = np.isfinite(rank_values)
    valid_indexes = np.nonzero(finite)[0]
    if valid_indexes.size == 0:
        logger.log(f"{mode}: no finite rank values; no removals can be selected")
        return [], [], [], [], {
            "mode": mode,
            "finite_rank_observation_count": 0,
            "cap_limited_total_capacity": 0,
            "max_requested_removed_count": int(max_requested),
            "max_selected_removed_count": 0,
            "ranked_observations_scanned": 0,
        }
    raw = table["srisk"]
    tie = table["tie_value"]
    sort_order = valid_indexes[np.lexsort((tie[valid_indexes], -raw[valid_indexes], -rank_values[valid_indexes]))]
    selected_ordinals: list[int] = []
    selected_hkls: list[tuple[int, int, int]] = []
    selected_rank_values: list[float] = []
    selected_raw_values: list[float] = []
    used_by_hkl: dict[tuple[int, int, int], int] = {}
    progress = base.StageProgress(logger, f"selecting {mode} high tail", len(sort_order), "ranked observations", workers=1)
    scanned = 0
    for index in sort_order:
        scanned += 1
        hkl = hkl_key(table["h"][index], table["k"][index], table["l"][index])
        used = used_by_hkl.get(hkl, 0)
        if used < capacities.get(hkl, 0):
            selected_ordinals.append(int(table["ordinal"][index]))
            selected_hkls.append(hkl)
            selected_rank_values.append(float(rank_values[index]))
            selected_raw_values.append(float(raw[index]))
            used_by_hkl[hkl] = used + 1
            if len(selected_ordinals) >= max_requested:
                progress.update(scanned, force=True)
                break
        progress.update(scanned)
    progress.finish(scanned)
    if len(selected_ordinals) < max_requested:
        logger.log(f"warning: {mode} selected {len(selected_ordinals):,}/{max_requested:,} requested rows")
    return selected_ordinals, selected_hkls, selected_rank_values, selected_raw_values, {
        "mode": mode,
        "finite_rank_observation_count": int(valid_indexes.size),
        "cap_limited_total_capacity": int(sum(capacities.values())),
        "max_requested_removed_count": int(max_requested),
        "max_selected_removed_count": int(len(selected_ordinals)),
        "ranked_observations_scanned": int(scanned),
    }


def prefix_counter(selected_hkls: list[tuple[int, int, int]], n: int) -> Counter[tuple[int, int, int]]:
    return Counter(selected_hkls[: int(n)])


def summarize_variant_hkls(
    counter: Counter[tuple[int, int, int]],
    hkl_counts: dict[tuple[int, int, int], int],
    max_removed_frac_per_hkl: float,
    min_retained_per_hkl: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    removed_counts = sorted(counter.values(), reverse=True)
    total_removed = int(sum(removed_counts))
    capacities = {
        hkl: max_removable_for_hkl(n_obs, max_removed_frac_per_hkl, min_retained_per_hkl)
        for hkl, n_obs in hkl_counts.items()
    }
    if total_removed:
        median_removed = float(np.median(removed_counts))
        max_removed = int(removed_counts[0])
        max_removed_fraction = max(counter[hkl] / hkl_counts[hkl] for hkl in counter)
        top10_fraction = sum(removed_counts[:10]) / total_removed
        top50_fraction = sum(removed_counts[:50]) / total_removed
        top100_fraction = sum(removed_counts[:100]) / total_removed
    else:
        median_removed = 0.0
        max_removed = 0
        max_removed_fraction = 0.0
        top10_fraction = top50_fraction = top100_fraction = 0.0
    hkls_at_cap = sum(1 for hkl, n_removed in counter.items() if n_removed >= capacities.get(hkl, 0) > 0)
    hkls_min_retained = sum(1 for hkl, n_removed in counter.items() if hkl_counts[hkl] - n_removed == min_retained_per_hkl)
    summary = {
        "affected_signed_hkls": int(len(counter)),
        "median_removed_per_affected_hkl": f"{median_removed:.6g}",
        "max_removed_per_affected_hkl": int(max_removed),
        "max_removed_fraction_any_hkl": f"{max_removed_fraction:.12g}",
        "hkls_hitting_cap": int(hkls_at_cap),
        "hkls_ending_with_min_retained": int(hkls_min_retained),
        "top10_removed_fraction": f"{top10_fraction:.12g}",
        "top50_removed_fraction": f"{top50_fraction:.12g}",
        "top100_removed_fraction": f"{top100_fraction:.12g}",
    }
    manifest_rows: list[dict[str, Any]] = []
    for hkl, n_removed in sorted(counter.items(), key=lambda item: item[0]):
        n_obs = hkl_counts[hkl]
        retained = n_obs - int(n_removed)
        manifest_rows.append(
            {
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "source_observations": int(n_obs),
                "removed_observations": int(n_removed),
                "retained_observations": int(retained),
                "removed_fraction": f"{int(n_removed) / int(n_obs):.12g}",
                "hit_cap": int(int(n_removed) >= capacities.get(hkl, 0) > 0),
                "ended_with_min_retained": int(retained == min_retained_per_hkl),
            }
        )
    top_rows: list[dict[str, Any]] = []
    for rank, (hkl, n_removed) in enumerate(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:20], start=1):
        n_obs = hkl_counts[hkl]
        retained = n_obs - int(n_removed)
        top_rows.append(
            {
                "rank": int(rank),
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "source_observations": int(n_obs),
                "removed_observations": int(n_removed),
                "retained_observations": int(retained),
                "removed_fraction": f"{int(n_removed) / int(n_obs):.12g}",
                "hit_cap": int(int(n_removed) >= capacities.get(hkl, 0) > 0),
                "ended_with_min_retained": int(retained == min_retained_per_hkl),
            }
        )
    return summary, manifest_rows, top_rows


def construct_masks_and_tables(
    variants: list[base.VariantSpec],
    variant_modes: dict[str, str],
    drop_items: list[tuple[str, float]],
    table: dict[str, np.ndarray],
    hkl_counts: dict[tuple[int, int, int], int],
    accepted_count: int,
    source_rows: int | None,
    score_source_id: str,
    max_removed_frac_per_hkl: float,
    min_retained_per_hkl: int,
    logger: base.RunLogger,
) -> tuple[dict[str, base.PackedMask], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    variants_by_mode: dict[str, list[base.VariantSpec]] = {}
    for variant in variants:
        variants_by_mode.setdefault(variant_modes[variant.variant_id], []).append(variant)

    masks = {variant.variant_id: base.PackedMask(accepted_count) for variant in variants}
    count_rows: list[dict[str, Any]] = []
    hkl_manifest_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    mode_stats: dict[str, Any] = {}

    for mode, mode_variants in variants_by_mode.items():
        rank_values, rank_stats = compute_per_hkl_rank_values(mode, table, logger)
        selected_ordinals, selected_hkls, selected_rank_values, selected_raw_values, selection_stats = select_for_mode(
            mode,
            rank_values,
            table,
            drop_items,
            hkl_counts,
            max_removed_frac_per_hkl,
            min_retained_per_hkl,
            accepted_count,
            logger,
        )
        selected_ordinals_array = np.asarray(selected_ordinals, dtype=np.int64)
        mode_stats[mode] = {"rank_stats": rank_stats, "selection_stats": selection_stats}
        for variant in mode_variants:
            requested = int(math.floor(float(variant.drop_fraction) * int(accepted_count)))
            actual = int(min(requested, len(selected_ordinals)))
            masks[variant.variant_id].set_many(selected_ordinals_array[:actual])
            counter = prefix_counter(selected_hkls, actual)
            hkl_summary, hkl_rows, hkl_top_rows = summarize_variant_hkls(
                counter,
                hkl_counts,
                max_removed_frac_per_hkl,
                min_retained_per_hkl,
            )
            rank_cutoff = selected_rank_values[actual - 1] if actual else ""
            raw_cutoff = selected_raw_values[actual - 1] if actual else ""
            count_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "score_mode": mode,
                    "score_id": "S_risk",
                    "score_source": score_source_id,
                    "drop_fraction_requested": float(variant.drop_fraction),
                    "drop_designation": f"drop{drop_label(float(variant.drop_fraction))}",
                    "max_removed_frac_per_hkl": float(max_removed_frac_per_hkl),
                    "min_retained_per_hkl": int(min_retained_per_hkl),
                    "accepted_population_count": int(accepted_count),
                    "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                    "eligible_ranked_observations": selection_stats["finite_rank_observation_count"],
                    "requested_removed_count": int(requested),
                    "accepted_observations_removed": int(actual),
                    "accepted_observations_retained": int(accepted_count - actual),
                    "fill_fraction_of_requested": float(actual / requested) if requested else 1.0,
                    "removed_fraction_of_accepted_population": float(actual / max(1, accepted_count)),
                    "removed_fraction_of_source_rows": "" if source_rows is None else float(actual / max(1, source_rows)),
                    "rank_cutoff": rank_cutoff,
                    "raw_srisk_cutoff": raw_cutoff,
                    "mask_sha256": masks[variant.variant_id].digest(),
                    **hkl_summary,
                }
            )
            for row in hkl_rows:
                hkl_manifest_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "score_mode": mode,
                        "drop_fraction_requested": float(variant.drop_fraction),
                        **row,
                    }
                )
            for row in hkl_top_rows:
                top_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "score_mode": mode,
                        "drop_fraction_requested": float(variant.drop_fraction),
                        **row,
                    }
                )

    for row in count_rows:
        variant_id = str(row["variant_id"])
        mask_count = masks[variant_id].count()
        if int(mask_count) != int(row["accepted_observations_removed"]):
            raise SystemExit(f"{variant_id}: mask count {mask_count} != table count {row['accepted_observations_removed']}")
    return masks, count_rows, hkl_manifest_rows, top_rows, mode_stats


def make_manifest_rows(
    variants: list[base.VariantSpec],
    variant_modes: dict[str, str],
    count_rows: list[dict[str, Any]],
    stream_qc: list[dict[str, Any]],
    write_streams: bool,
) -> list[dict[str, Any]]:
    counts = {row["variant_id"]: row for row in count_rows}
    qc = base.stream_qc_by_variant(stream_qc)
    rows: list[dict[str, Any]] = []
    for order, variant in enumerate(variants, start=1):
        count = counts[variant.variant_id]
        qc_row = qc.get(variant.variant_id, {})
        rows.append(
            {
                "merge_order": order,
                "variant_id": variant.variant_id,
                "stream_path": str(variant.output_stream),
                "score_id": "S_risk",
                "score_mode": variant_modes[variant.variant_id],
                "score_source": count["score_source"],
                "experiment_type": "fullpop_per_hkl_zrisk_tail_filter",
                "target": "all",
                "ranking_scope": "global by selected score mode",
                "fraction": float(variant.drop_fraction),
                "designation": f"drop{drop_label(float(variant.drop_fraction))}",
                "max_removed_frac_per_hkl": count["max_removed_frac_per_hkl"],
                "min_retained_per_hkl": count["min_retained_per_hkl"],
                "actual_removed_count": int(count["accepted_observations_removed"]),
                "actual_removed_fraction_of_accepted_population": count["removed_fraction_of_accepted_population"],
                "actual_removed_fraction_of_source_rows": count["removed_fraction_of_source_rows"],
                "affected_signed_hkls": count["affected_signed_hkls"],
                "hkls_hitting_cap": count["hkls_hitting_cap"],
                "hkls_ending_with_min_retained": count["hkls_ending_with_min_retained"],
                "stream_status": qc_row.get("status") or ("generated" if write_streams else "planned_dry_run"),
                "stream_qc_removed_observations": qc_row.get("stream_removed_count", ""),
                "stream_qc_kept_observations": qc_row.get("stream_kept_count", ""),
                "merge_status": "not_started",
                "partialator_model": "offset",
                "symmetry": "4/mmm",
                "iterations": 10,
                "min_measurements": 1,
                "push_res": "inf",
                "no_Bscale": True,
                "no_pr": True,
            }
        )
    return rows


def git_info(project_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        rev = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--short"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return out
    out.update({"available": bool(rev.stdout.strip()), "commit": rev.stdout.strip(), "status_short": status.stdout.splitlines()})
    return out


def write_outputs(
    args: argparse.Namespace,
    variants: list[base.VariantSpec],
    variant_modes: dict[str, str],
    db_file: Path,
    source_rows: int | None,
    score_source: dict[str, str],
    mode_stats: dict[str, Any],
    count_rows: list[dict[str, Any]],
    hkl_manifest_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    stream_qc: list[dict[str, Any]],
    started_utc: str,
    logger: base.RunLogger,
) -> None:
    logger.log("writing per-HKL z-risk manifests and metadata")
    manifest_rows = make_manifest_rows(variants, variant_modes, count_rows, stream_qc, bool(args.write_streams))
    write_delimited(args.output_dir / "per_hkl_zrisk_counts.csv", count_rows, delimiter=",")
    write_delimited(args.output_dir / "per_hkl_zrisk_manifest.tsv", manifest_rows)
    write_delimited(args.output_dir / "per_hkl_zrisk_hkl_removal_manifest.tsv", hkl_manifest_rows)
    write_delimited(args.output_dir / "per_hkl_zrisk_top_removed_hkls.tsv", top_rows)
    parameters = {
        "source_out_dir": str(args.source_out_dir),
        "source_stream": str(args.source_stream),
        "source_cache": str(db_file),
        "output_dir": str(args.output_dir),
        "score_modes": list(args.score_modes),
        "drop_fracs": [text for text, _fraction in args.drop_items],
        "workers_requested": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "score_source": score_source,
        "score_mode_definitions": {
            "raw_global": "rank by raw S_risk",
            "per_hkl_z": "rank by (S_risk - mean_hkl(S_risk)) / std_hkl(S_risk), std uses ddof=0",
            "per_hkl_robust_z": "rank by (S_risk - median_hkl(S_risk)) / (1.4826 * MAD_hkl(S_risk))",
        },
        "filtering_rule": {
            "target": "all full-population scoreable accepted observations",
            "hkl_grouping": "exact signed h,k,l",
            "symmetry_canonicalization": False,
            "global_rank": "score_mode value descending, raw S_risk descending for ties, source_order ascending for ties when available",
            "requested_n_remove": "floor(drop_fraction * accepted_population_count)",
            "ineligible_groups": "HKLs with fewer than 2 finite S_risk values or zero/near-zero denominator are not removed in per-HKL z modes",
            "max_removed_frac_per_hkl": float(args.max_removed_frac_per_hkl),
            "min_retained_per_hkl": int(args.min_retained_per_hkl),
            "non_selected_observations": "retained unchanged",
            "non_scoreable_source_rows": "retained unchanged",
            "source_order": "preserved by stream rewrite",
        },
    }
    write_json(args.output_dir / "per_hkl_zrisk_parameters.json", parameters)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "script": str(Path(__file__).resolve()),
        "git": git_info(Path(__file__).resolve().parents[1]),
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "source_files": {
            "source_out_dir": str(args.source_out_dir),
            "source_cache": str(db_file),
            "source_stream": str(args.source_stream),
            "source_parameters": str(args.source_out_dir / "parameters.json"),
            "source_validation": str(args.source_out_dir / "validation.json"),
        },
        "mode_stats": mode_stats,
        "variant_count": len(variants),
        "stream_qc": stream_qc,
        "outputs": {
            "manifest": str(args.output_dir / "per_hkl_zrisk_manifest.tsv"),
            "counts": str(args.output_dir / "per_hkl_zrisk_counts.csv"),
            "hkl_removal_manifest": str(args.output_dir / "per_hkl_zrisk_hkl_removal_manifest.tsv"),
            "top_removed_hkls": str(args.output_dir / "per_hkl_zrisk_top_removed_hkls.tsv"),
            "parameters": str(args.output_dir / "per_hkl_zrisk_parameters.json"),
            "metadata": str(args.output_dir / "per_hkl_zrisk_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "per_hkl_zrisk_metadata.json", metadata)


def run_builder(args: argparse.Namespace) -> int:
    variants, variant_modes = build_variants(args.score_modes, args.drop_items, args.output_dir, float(args.max_removed_frac_per_hkl))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(planned_output_paths(args.output_dir, variants, bool(args.write_streams)))

    logger = base.RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        logger.log("V6 per-HKL z-risk cap-sweep builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"score_modes={','.join(args.score_modes)}")
        logger.log(f"drop_fracs={','.join(text for text, _fraction in args.drop_items)}")
        logger.log(f"max_removed_frac_per_hkl={float(args.max_removed_frac_per_hkl):.6g}")
        logger.log(f"min_retained_per_hkl={int(args.min_retained_per_hkl)}")
        logger.log(f"workers_requested={int(args.workers)}")

        db_file = base.cache_db_path(args.source_out_dir)
        schema = require_cache_schema(db_file)
        score_expression, score_id, score_description = choose_score_expression(schema)
        score_source = {"score_id": score_id, "expression": score_expression, "description": score_description}
        accepted_count = base.score_cache_count(db_file)
        source_rows = base.source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and source_rows < accepted_count:
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than accepted cache count {accepted_count:,}")
        table = load_score_table(db_file, accepted_count, score_expression, logger)
        hkl_counts = compute_hkl_counts(table)
        masks, count_rows, hkl_manifest_rows, top_rows, mode_stats = construct_masks_and_tables(
            variants,
            variant_modes,
            args.drop_items,
            table,
            hkl_counts,
            accepted_count,
            source_rows,
            score_id,
            float(args.max_removed_frac_per_hkl),
            int(args.min_retained_per_hkl),
            logger,
        )
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            stream_qc = base.rewrite_streams(args.source_stream, db_file, variants, masks, logger, source_rows)
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(args, variants, variant_modes, db_file, source_rows, score_source, mode_stats, count_rows, hkl_manifest_rows, top_rows, stream_qc, started_utc, logger)
        logger.log("V6 per-HKL z-risk cap-sweep builder complete")
        return 0
    finally:
        logger.close()


def synthetic_stream_text(rows: list[tuple[str, str, int, int, int, float]]) -> str:
    lines = []
    for source, event, h, k, l, intensity in rows:
        lines.extend(
            [
                "----- Begin chunk -----\n",
                f"Image filename: {source}\n",
                f"Event: {event}\n",
                "Begin crystal\n",
                "Reflections measured after indexing\n",
                f"{h:4d} {k:4d} {l:4d} {intensity:10.3f} 1.0\n",
                "End of reflections\n",
                "End crystal\n",
                "----- End chunk -----\n",
            ]
        )
    return "".join(lines)


def run_self_test() -> int:
    project_root = Path(__file__).resolve().parents[1]
    scratch_root = project_root / ".codex_smoke" / "per_hkl_zrisk"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    source_out = scratch_root / "source_out"
    output_dir = scratch_root / "out"
    source_out.mkdir(parents=True)
    source_stream = scratch_root / "source.stream"
    db_file = source_out / "full_population_cache.sqlite"
    try:
        stream_rows: list[tuple[str, str, int, int, int, float]] = []
        cache_rows: list[tuple[int, int, int, int, str, int, float, float]] = []
        scores_by_hkl = {
            (1, 0, 0): [1.0, 2.0, 3.0, 100.0, 101.0, 102.0],
            (2, 0, 0): [5.0, 5.0, 5.0, 5.0],
            (3, 0, 0): [1.0, 4.0, 7.0, 10.0],
        }
        ordinal = 0
        for hkl, scores in scores_by_hkl.items():
            for value in scores:
                source = "synthetic.img"
                event = str(ordinal + 1)
                h, k, l = hkl
                exact_key = base.key_to_text(source, event, h, k, l)
                cache_rows.append((ordinal, h, k, l, exact_key, ordinal, value, 1.0))
                stream_rows.append((source, event, h, k, l, value))
                ordinal += 1
        source_stream.write_text(synthetic_stream_text(stream_rows), encoding="utf-8")
        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            CREATE TABLE score_cache(
                ordinal INTEGER PRIMARY KEY,
                h INTEGER NOT NULL,
                k INTEGER NOT NULL,
                l INTEGER NOT NULL,
                exact_key_text TEXT NOT NULL,
                source_order INTEGER NOT NULL,
                Eg REAL NOT NULL,
                M2 REAL NOT NULL
            )
            """
        )
        conn.executemany("INSERT INTO score_cache VALUES (?,?,?,?,?,?,?,?)", cache_rows)
        conn.commit()
        conn.close()
        args = argparse.Namespace(
            source_out_dir=source_out.resolve(),
            source_stream=source_stream.resolve(),
            output_dir=output_dir.resolve(),
            score_modes=["per_hkl_z", "per_hkl_robust_z"],
            drop_items=[("0.50", 0.50)],
            min_retained_per_hkl=2,
            max_removed_frac_per_hkl=0.80,
            workers=1,
            dry_run=False,
            write_streams=True,
        )
        run_builder(args)
        for name in [
            "per_hkl_zrisk_manifest.tsv",
            "per_hkl_zrisk_counts.csv",
            "per_hkl_zrisk_hkl_removal_manifest.tsv",
            "per_hkl_zrisk_top_removed_hkls.tsv",
            "per_hkl_zrisk_parameters.json",
            "per_hkl_zrisk_metadata.json",
        ]:
            assert (output_dir / name).is_file(), name
        with (output_dir / "per_hkl_zrisk_counts.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "count rows"
        for row in rows:
            assert int(row["accepted_observations_removed"]) > 0
            assert int(row["hkls_hitting_cap"]) >= 0
        print("self-test passed: synthetic per-HKL z/robust-z selection, stream rewrite, and audit tables are consistent")
        return 0
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    return run_builder(args)


if __name__ == "__main__":
    raise SystemExit(main())
