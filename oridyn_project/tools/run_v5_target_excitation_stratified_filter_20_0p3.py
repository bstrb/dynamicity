#!/usr/bin/env python3
"""Run v5 keep filters stratified by target excitation tertile."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from run_v5_allscore_filter_and_50split_20_0p3 import (
    DEFAULT_MIN_ACCEPTED_KEEP,
    DEFAULT_PROGRESS_EVERY,
    DEFAULT_SCORES_CHUNKSIZE,
    HKL_COLUMNS,
    KEY_COLUMNS,
    collect_smoke_keys,
    log,
    normalize_event,
    normalize_key_chunk,
    normalize_source,
    percent_label,
    require_columns,
    stable_tie_break_values,
    write_stream_variants,
)


BASE = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524")
DEFAULT_STREAM = BASE / "MFM300-VIII_cut_20-0_3.stream"
DEFAULT_V5_SCORES = (
    BASE
    / "oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705"
    / "geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"
)
DEFAULT_ACCEPTED_SCORES = (
    BASE
    / "oridyn_v4_local_crowding_raw_20_0p3_20260704"
    / "partialator_survivor_mask"
    / "p1_iter1_20260705T1214"
    / "v4_p1_iter1_partialator_survivors_only_scores.csv"
)
DEFAULT_OUTPUT_ROOT = BASE / "oridyn_v5_target_excitation_stratified_filter_20_0p3_20260712"
DEFAULT_SCORE_COLUMN = "nonself_local_excitation_raw"
DEFAULT_TARGET_COLUMN = "target_excitation_Eg"
DEFAULT_KEEP_FRACTIONS = [0.95, 0.90]
DEFAULT_SEED = 1
TERTILE_LABELS = ("low", "mid", "high")


@dataclass(frozen=True)
class StratifiedVariant:
    variant: str
    kind: str
    stream_name: str
    keep_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--v5-scores", type=Path, default=DEFAULT_V5_SCORES)
    parser.add_argument("--accepted-scores", type=Path, default=DEFAULT_ACCEPTED_SCORES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--min-accepted-keep", type=int, default=DEFAULT_MIN_ACCEPTED_KEEP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scores-chunksize", type=int, default=DEFAULT_SCORES_CHUNKSIZE)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--max-events", type=int, default=None, help="Tiny smoke-test limit on stream crystal blocks/events")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for label in ("stream", "v5_scores", "accepted_scores"):
        path = getattr(args, label)
        if not path.is_file():
            raise SystemExit(f"--{label.replace('_', '-')} must be a file: {path}")
    if int(args.min_accepted_keep) < 1:
        raise SystemExit("--min-accepted-keep must be >= 1")
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


def variant_specs(keep_fractions: list[float], seed: int) -> list[StratifiedVariant]:
    specs: list[StratifiedVariant] = []
    for fraction in keep_fractions:
        pct = percent_label(fraction)
        specs.append(
            StratifiedVariant(
                variant=f"v5_target_excitation_stratified_keep{pct:02d}",
                kind="v5_stratified",
                stream_name=f"MFM300_VIII_v5_target_excitation_stratified_keep{pct:02d}.stream",
                keep_fraction=float(fraction),
            )
        )
        specs.append(
            StratifiedVariant(
                variant=f"v5_target_excitation_stratified_random_keep{pct:02d}_seed{int(seed)}",
                kind="matched_random",
                stream_name=f"MFM300_VIII_v5_target_excitation_stratified_random_keep{pct:02d}_seed{int(seed)}.stream",
                keep_fraction=float(fraction),
            )
        )
    return specs


def output_paths(output_root: Path, specs: list[StratifiedVariant]) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for spec in specs:
        paths[spec.variant] = {
            "stream": output_root / spec.stream_name,
            "kept_keys_csv": output_root / f"{spec.variant}_kept_accepted_keys.csv",
            "removed_keys_csv": output_root / f"{spec.variant}_removed_accepted_keys.csv",
        }
    return paths


def ensure_outputs(output_root: Path, paths: dict[str, dict[str, Path]], overwrite: bool) -> None:
    blocked: list[Path] = []
    for variant_paths in paths.values():
        for path in variant_paths.values():
            if path.exists():
                blocked.append(path)
    for name in [
        "v5_target_excitation_stratified_filter_summary.csv",
        "v5_target_excitation_stratified_filter_global_summary.csv",
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


def load_accepted_observations(
    accepted_path: Path,
    key_filter: set[tuple[str, str, int, int, int]] | None,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(accepted_path, nrows=0).columns.tolist()
    require_columns(header, KEY_COLUMNS, "accepted-only table")
    usecols = [*KEY_COLUMNS]
    has_partiality = "partiality" in header
    if has_partiality:
        usecols.append("partiality")

    chunks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {
        "accepted_rows_read": 0,
        "accepted_rows_after_key_cleanup": 0,
        "accepted_rows_after_smoke_key_filter": 0,
        "accepted_duplicate_rows": 0,
        "accepted_duplicate_keys": 0,
        "accepted_unique_keys": 0,
        "partiality_column_available": bool(has_partiality),
    }
    for idx, chunk in enumerate(pd.read_csv(accepted_path, usecols=usecols, chunksize=int(chunksize)), start=1):
        stats["accepted_rows_read"] += int(len(chunk))
        normalized = normalize_key_chunk(chunk)
        stats["accepted_rows_after_key_cleanup"] += int(len(normalized))
        if has_partiality:
            normalized["partiality"] = pd.to_numeric(chunk.loc[normalized.index, "partiality"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        else:
            normalized["partiality"] = np.nan
        if key_filter is not None:
            key_mask = [
                (source, event, int(h), int(k), int(l)) in key_filter
                for source, event, h, k, l in normalized.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
            ]
            normalized = normalized.loc[key_mask].copy()
            stats["accepted_rows_after_smoke_key_filter"] += int(len(normalized))
        if not normalized.empty:
            chunks.append(normalized.loc[:, [*KEY_COLUMNS, "partiality"]].copy())
        if idx == 1 or idx % 5 == 0:
            extra = ""
            if key_filter is not None:
                extra = f", matched_smoke_rows={stats['accepted_rows_after_smoke_key_filter']:,}"
            log(f"Accepted CSV scan: chunks={idx:,}, rows_read={stats['accepted_rows_read']:,}{extra}")

    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=[*KEY_COLUMNS, "partiality"])
    duplicated = table.duplicated(KEY_COLUMNS, keep=False)
    stats["accepted_duplicate_rows"] = int(duplicated.sum())
    stats["accepted_duplicate_keys"] = int(table.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicated.any() else 0
    if duplicated.any():
        log("Warning: duplicate exact observation keys in accepted-only table; keeping first key")
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    stats["accepted_unique_keys"] = int(len(table))
    return table.reset_index(drop=True), stats


def normalize_v5_chunk(chunk: pd.DataFrame, score_column: str, target_column: str) -> pd.DataFrame:
    out = chunk.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out[target_column] = pd.to_numeric(out[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.loc[~out[[*HKL_COLUMNS, score_column, target_column]].isna().any(axis=1)].copy()
    if out.empty:
        return out.loc[:, [*KEY_COLUMNS, score_column, target_column]].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out.loc[:, [*KEY_COLUMNS, score_column, target_column]].copy()


def load_accepted_v5_table(
    v5_scores_path: Path,
    accepted: pd.DataFrame,
    score_column: str,
    target_column: str,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(v5_scores_path, nrows=0).columns.tolist()
    require_columns(header, [*KEY_COLUMNS, score_column, target_column], "v5 score CSV")
    if accepted.empty:
        raise SystemExit("No usable accepted observations were loaded")
    usecols = [*KEY_COLUMNS, score_column, target_column]
    chunks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {
        "v5_rows_read": 0,
        "v5_rows_after_cleanup": 0,
        "accepted_v5_matched_rows": 0,
    }
    accepted_keys = accepted.loc[:, [*KEY_COLUMNS, "partiality"]].copy()
    for idx, chunk in enumerate(pd.read_csv(v5_scores_path, usecols=usecols, chunksize=int(chunksize)), start=1):
        stats["v5_rows_read"] += int(len(chunk))
        normalized = normalize_v5_chunk(chunk, score_column, target_column)
        stats["v5_rows_after_cleanup"] += int(len(normalized))
        if normalized.empty:
            continue
        matched = normalized.merge(accepted_keys, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        stats["accepted_v5_matched_rows"] += int(len(matched))
        if not matched.empty:
            chunks.append(matched)
        if idx == 1 or idx % 5 == 0:
            log(
                "V5 CSV scan: "
                f"chunks={idx:,}, rows_read={stats['v5_rows_read']:,}, "
                f"accepted_matches={stats['accepted_v5_matched_rows']:,}"
            )
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=[*KEY_COLUMNS, score_column, target_column, "partiality"])
    stats["accepted_keys_without_v5_score"] = int(max(0, len(accepted) - table.loc[:, KEY_COLUMNS].drop_duplicates().shape[0]))
    return table, stats


def rank_within_groups(work: pd.DataFrame, group_columns: list[str], sort_columns: list[str], ascending: list[bool]) -> np.ndarray:
    ordered = work.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    ranks = np.empty(len(work), dtype=np.int64)
    rank_values = ordered.groupby(group_columns, sort=False).cumcount().to_numpy(dtype=np.int64)
    ranks[ordered.index.to_numpy(dtype=np.int64)] = rank_values
    return ranks


def assign_target_tertiles(work: pd.DataFrame, target_column: str) -> pd.DataFrame:
    out = work.copy().reset_index(drop=True)
    ordered = out.sort_values([*HKL_COLUMNS, target_column, "tie_break"], ascending=[True, True, True, True, True], kind="mergesort")
    rank = ordered.groupby(HKL_COLUMNS, sort=False).cumcount().to_numpy(dtype=np.int64)
    n = ordered.groupby(HKL_COLUMNS, sort=False)[target_column].transform("size").to_numpy(dtype=np.int64)
    tertile = np.minimum((rank * 3) // np.maximum(n, 1), 2).astype(np.int64)
    assigned = np.empty(len(out), dtype=np.int64)
    assigned[ordered.index.to_numpy(dtype=np.int64)] = tertile
    out["excitation_tertile_index"] = assigned
    out["excitation_tertile"] = pd.Categorical.from_codes(assigned, TERTILE_LABELS, ordered=True)
    return out


def allocate_removals_by_tertile(counts: list[int], total_remove: int) -> list[int]:
    capacities = [max(0, int(count) - 1) if int(count) > 0 else 0 for count in counts]
    total_remove = min(max(0, int(total_remove)), int(sum(capacities)))
    if total_remove == 0 or sum(counts) == 0:
        return [0, 0, 0]
    weights = np.asarray(counts, dtype=float) / float(sum(counts))
    ideal = weights * float(total_remove)
    removals = [min(int(np.floor(value)), cap) for value, cap in zip(ideal, capacities, strict=True)]
    remaining = int(total_remove - sum(removals))
    remainders = [float(value - np.floor(value)) for value in ideal]
    while remaining > 0:
        best_idx = None
        best_key = None
        for idx, (current, cap) in enumerate(zip(removals, capacities, strict=True)):
            if current >= cap:
                continue
            key = (remainders[idx], counts[idx], -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        removals[best_idx] += 1
        remaining -= 1
        remainders[best_idx] = 0.0
    return removals


def build_budget_table(work: pd.DataFrame, keep_fractions: list[float], min_accepted_keep: int) -> pd.DataFrame:
    group_columns = [*HKL_COLUMNS, "excitation_tertile_index"]
    segment_counts = work.groupby(group_columns, observed=True, sort=False).size().rename("tertile_total").reset_index()
    pivot = segment_counts.pivot_table(
        index=HKL_COLUMNS,
        columns="excitation_tertile_index",
        values="tertile_total",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for tertile_idx in range(len(TERTILE_LABELS)):
        if tertile_idx not in pivot.columns:
            pivot[tertile_idx] = 0

    rows: list[dict[str, Any]] = []
    for row in pivot.to_dict("records"):
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        tertile_counts = [int(row.get(tertile_idx, 0)) for tertile_idx in range(len(TERTILE_LABELS))]
        n_total = int(sum(tertile_counts))
        for fraction in sorted(set(float(value) for value in keep_fractions), reverse=True):
            if n_total < int(min_accepted_keep):
                keep_n = n_total
            else:
                keep_n = min(max(int(np.ceil(n_total * float(fraction))), int(min_accepted_keep)), n_total)
            total_remove = int(n_total - keep_n)
            tertile_removals = allocate_removals_by_tertile(tertile_counts, total_remove)
            for tertile_idx, (count, remove_n) in enumerate(zip(tertile_counts, tertile_removals, strict=True)):
                rows.append(
                    {
                        "keep_fraction": float(fraction),
                        "h": h,
                        "k": k,
                        "l": l,
                        "excitation_tertile_index": int(tertile_idx),
                        "excitation_tertile": TERTILE_LABELS[tertile_idx],
                        "tertile_total": int(count),
                        "tertile_remove_budget": int(remove_n),
                        "hkl_total": int(n_total),
                        "hkl_keep_target": int(keep_n),
                        "hkl_remove_budget": int(total_remove),
                    }
                )
    return pd.DataFrame.from_records(rows)


def build_selection_tables(
    accepted_v5: pd.DataFrame,
    specs: list[StratifiedVariant],
    score_column: str,
    target_column: str,
    min_accepted_keep: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if accepted_v5.empty:
        raise SystemExit("No partialator-accepted observations matched the v5 score table")
    work = accepted_v5.copy().reset_index(drop=True)
    duplicated = work.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(duplicated.sum())
    duplicate_keys = int(work.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact observation keys after accepted/v5 join; keeping first row per key")
        work = work.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)

    log(f"Preparing selection table: rows={len(work):,}")
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce").astype(float)
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce").astype(float)
    work["partiality"] = pd.to_numeric(work["partiality"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work["tie_break"] = stable_tie_break_values(work, int(seed))
    work["n_accepted"] = work.groupby(HKL_COLUMNS, sort=False)[score_column].transform("size").astype("int64")
    log("Assigning target-excitation tertiles within signed HKLs")
    work = assign_target_tertiles(work, target_column)
    log("Ranking v5 score within target-excitation tertiles")
    work["rank_v5_desc_in_tertile"] = rank_within_groups(
        work,
        [*HKL_COLUMNS, "excitation_tertile_index"],
        [*HKL_COLUMNS, "excitation_tertile_index", score_column, "tie_break"],
        [True, True, True, True, False, True],
    )
    log("Ranking matched-random order within target-excitation tertiles")
    work["rank_random_in_tertile"] = rank_within_groups(
        work,
        [*HKL_COLUMNS, "excitation_tertile_index"],
        [*HKL_COLUMNS, "excitation_tertile_index", "tie_break"],
        [True, True, True, True, True],
    )

    log("Building per-HKL/per-tertile removal budgets")
    budget_summary = build_budget_table(work, [float(spec.keep_fraction) for spec in specs], int(min_accepted_keep))
    budget_columns: dict[float, str] = {}
    merge_columns = [*HKL_COLUMNS, "excitation_tertile_index"]
    for fraction in sorted({float(spec.keep_fraction) for spec in specs}, reverse=True):
        column = f"remove_budget_keep{percent_label(fraction):02d}"
        budget_columns[float(fraction)] = column
        budget_slice = budget_summary.loc[
            budget_summary["keep_fraction"].to_numpy(dtype=float) == float(fraction),
            [*merge_columns, "tertile_remove_budget"],
        ].rename(columns={"tertile_remove_budget": column})
        work = work.merge(budget_slice, on=merge_columns, how="left", sort=False, validate="many_to_one")
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0).astype("int64")

    remove_mask_bits = np.zeros(len(work), dtype=np.uint16)
    summary_tables: list[pd.DataFrame] = []
    variant_rows: list[dict[str, Any]] = []
    unique_signed_hkls_total = int(work.loc[:, HKL_COLUMNS].drop_duplicates().shape[0])
    log(f"Applying variant masks and summaries for {len(specs)} variants across {unique_signed_hkls_total:,} signed HKLs")
    for bit_idx, spec in enumerate(specs):
        log(f"Selecting removals for {spec.variant}")
        bit = np.uint16(1 << bit_idx)
        remove_counts = work[budget_columns[float(spec.keep_fraction)]].to_numpy(dtype=np.int64)
        if spec.kind == "v5_stratified":
            remove = work["rank_v5_desc_in_tertile"].to_numpy(dtype=np.int64) < remove_counts
        elif spec.kind == "matched_random":
            remove = work["rank_random_in_tertile"].to_numpy(dtype=np.int64) < remove_counts
        else:
            raise ValueError(f"Unknown variant kind: {spec.kind}")
        work[f"remove_{spec.variant}"] = remove
        remove_mask_bits[remove] |= bit
        summary_tables.append(build_stratified_summary(work, spec, score_column, target_column))
        kept = ~remove
        variant_rows.append(
            {
                "variant": spec.variant,
                "kind": spec.kind,
                "keep_fraction": float(spec.keep_fraction),
                "accepted_kept": int(kept.sum()),
                "accepted_removed": int(remove.sum()),
                "fraction_accepted_removed": float(remove.sum() / max(len(work), 1)),
                "signed_hkls_total": unique_signed_hkls_total,
                "signed_hkls_changed": int(work.loc[remove, HKL_COLUMNS].drop_duplicates().shape[0]),
                "min_accepted_after_filter": int((work.assign(_kept=kept).groupby(HKL_COLUMNS, sort=False)["_kept"].sum()).min()),
            }
        )

    selection_stats = {
        "accepted_v5_rows_after_duplicate_cleanup": int(len(work)),
        "duplicate_accepted_v5_rows": duplicate_rows,
        "duplicate_accepted_v5_keys": duplicate_keys,
        "unique_signed_hkls_total": unique_signed_hkls_total,
        "seed": int(seed),
        "target_column": target_column,
        "score_column": score_column,
        "min_accepted_keep": int(min_accepted_keep),
        "rules": [
            "Uses existing v5 score CSV only; v5 is not multiplied by target_excitation_Eg.",
            "Only partialator-accepted observations are considered for removal.",
            "Exact observation key is source_filename + normalized event + signed h,k,l.",
            "Signed HKLs are preserved exactly; no symmetry canonicalization is applied.",
            "Within each signed HKL, observations are split into low/mid/high target_excitation_Eg tertiles.",
            "The per-HKL keep budget follows the existing min-retained rule.",
            "Removal budget is distributed proportionally across non-empty tertiles, keeping at least one observation in each.",
            "V5 variants remove the highest nonself_local_excitation_raw observations within each tertile.",
            "Random variants remove exactly the same number per signed HKL and excitation tertile as the paired v5 variant.",
            "Nonaccepted or unmatched stream observations are kept unchanged in every output stream.",
        ],
    }
    return (
        work,
        remove_mask_bits,
        pd.concat(summary_tables, ignore_index=True),
        pd.DataFrame.from_records(variant_rows),
        budget_summary,
        selection_stats,
    )


def finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.mean()) if len(numeric) else np.nan


def finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.median()) if len(numeric) else np.nan


def build_stratified_summary(
    work: pd.DataFrame,
    spec: StratifiedVariant,
    score_column: str,
    target_column: str,
) -> pd.DataFrame:
    remove_column = f"remove_{spec.variant}"
    group_columns = [*HKL_COLUMNS, "excitation_tertile_index", "excitation_tertile"]

    def stats_for(subset: pd.DataFrame, label: str) -> pd.DataFrame:
        columns = [
            *group_columns,
            f"target_excitation_Eg_mean_{label}",
            f"target_excitation_Eg_median_{label}",
            f"partiality_mean_{label}",
            f"partiality_median_{label}",
            f"v5_score_mean_{label}",
            f"v5_score_median_{label}",
        ]
        if subset.empty:
            return pd.DataFrame(columns=columns)
        return (
            subset.groupby(group_columns, observed=True, sort=False)
            .agg(
                **{
                    f"target_excitation_Eg_mean_{label}": (target_column, "mean"),
                    f"target_excitation_Eg_median_{label}": (target_column, "median"),
                    f"partiality_mean_{label}": ("partiality", "mean"),
                    f"partiality_median_{label}": ("partiality", "median"),
                    f"v5_score_mean_{label}": (score_column, "mean"),
                    f"v5_score_median_{label}": (score_column, "median"),
                }
            )
            .reset_index()
        )

    base = work.loc[:, [*group_columns, target_column, "partiality", score_column, remove_column]].copy()
    base["_kept"] = ~base[remove_column].to_numpy(dtype=bool)
    base["_removed"] = base[remove_column].to_numpy(dtype=bool)
    counts = (
        base.groupby(group_columns, observed=True, sort=False)
        .agg(n_total=(score_column, "size"), n_kept=("_kept", "sum"), n_removed=("_removed", "sum"))
        .reset_index()
    )
    out = counts.merge(stats_for(base, "before"), on=group_columns, how="left", sort=False)
    out = out.merge(stats_for(base.loc[base["_kept"]], "kept"), on=group_columns, how="left", sort=False)
    out = out.merge(stats_for(base.loc[base["_removed"]], "removed"), on=group_columns, how="left", sort=False)
    out.insert(0, "keep_fraction", float(spec.keep_fraction))
    out.insert(0, "kind", spec.kind)
    out.insert(0, "variant", spec.variant)
    out["n_total"] = out["n_total"].astype("int64")
    out["n_kept"] = out["n_kept"].astype("int64")
    out["n_removed"] = out["n_removed"].astype("int64")
    return out


def write_key_tables(
    paths: dict[str, dict[str, Path]],
    work: pd.DataFrame,
    specs: list[StratifiedVariant],
    score_column: str,
    target_column: str,
) -> dict[str, dict[str, int]]:
    key_stats: dict[str, dict[str, int]] = {}
    columns = [
        *KEY_COLUMNS,
        target_column,
        score_column,
        "partiality",
        "excitation_tertile",
        "tie_break",
        "n_accepted",
        "rank_v5_desc_in_tertile",
        "rank_random_in_tertile",
    ]
    for spec in specs:
        remove = work[f"remove_{spec.variant}"].to_numpy(dtype=bool)
        kept = work.loc[~remove, columns].copy()
        removed = work.loc[remove, columns].copy()
        kept.to_csv(paths[spec.variant]["kept_keys_csv"], index=False)
        removed.to_csv(paths[spec.variant]["removed_keys_csv"], index=False)
        key_stats[spec.variant] = {"kept_key_rows": int(len(kept)), "removed_key_rows": int(len(removed))}
    return key_stats


def write_run_metadata(
    output_root: Path,
    args: argparse.Namespace,
    specs: list[StratifiedVariant],
    accepted_stats: dict[str, Any],
    v5_stats: dict[str, Any],
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
        "target_column": str(args.target_column),
        "keep_fractions": [float(value) for value in args.keep_fractions],
        "min_accepted_keep": int(args.min_accepted_keep),
        "seed": int(args.seed),
        "max_events": None if args.max_events is None else int(args.max_events),
        "variants": [spec.__dict__ for spec in specs],
        "accepted_stats": accepted_stats,
        "v5_stats": v5_stats,
        "selection_stats": selection_stats,
        "stream_stats": stream_stats,
        "key_table_stats": key_table_stats,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_dry_run(args: argparse.Namespace, specs: list[StratifiedVariant], paths: dict[str, dict[str, Path]]) -> None:
    print("Dry run: no key joins, stratified selection, stream rewriting, or output writing will run.")
    print(f"Output root: {args.output_root}")
    for spec in specs:
        print(f"stream: {paths[spec.variant]['stream']}")
        print(f"kept keys: {paths[spec.variant]['kept_keys_csv']}")
        print(f"removed keys: {paths[spec.variant]['removed_keys_csv']}")
    print(f"summary CSV: {args.output_root / 'v5_target_excitation_stratified_filter_summary.csv'}")
    print(f"global summary CSV: {args.output_root / 'v5_target_excitation_stratified_filter_global_summary.csv'}")
    print(f"budget CSV: {args.output_root / 'v5_target_excitation_stratified_filter_budget.csv'}")
    print(f"metadata JSON: {args.output_root / 'run_metadata.json'}")


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
    smoke_stats: dict[str, Any] = {}
    if args.max_events is not None:
        log(f"Collecting smoke-test keys for first {int(args.max_events)} crystals")
        smoke_key_filter, smoke_stats = collect_smoke_keys(args.stream, int(args.max_events), int(args.progress_every))
        log(f"Smoke key collection done: keys={len(smoke_key_filter):,}")

    log("Loading partialator-accepted observation keys and partiality")
    accepted, accepted_stats = load_accepted_observations(args.accepted_scores, smoke_key_filter, int(args.scores_chunksize))
    accepted_stats.update(smoke_stats)
    log(f"Loaded accepted observations: unique_keys={len(accepted):,}")

    log("Joining accepted observations to existing v5 score table")
    accepted_v5, v5_stats = load_accepted_v5_table(
        args.v5_scores,
        accepted,
        str(args.score_column),
        str(args.target_column),
        int(args.scores_chunksize),
    )
    log(f"Loaded accepted v5 observations: rows={len(accepted_v5):,}")

    log("Building target-excitation-stratified v5 and matched-random selections")
    selection_table, remove_mask_bits, stratified_summary, global_summary, budget_summary, selection_stats = build_selection_tables(
        accepted_v5,
        specs,
        str(args.score_column),
        str(args.target_column),
        int(args.min_accepted_keep),
        int(args.seed),
    )
    stratified_summary.to_csv(args.output_root / "v5_target_excitation_stratified_filter_summary.csv", index=False)
    global_summary.to_csv(args.output_root / "v5_target_excitation_stratified_filter_global_summary.csv", index=False)
    budget_summary.to_csv(args.output_root / "v5_target_excitation_stratified_filter_budget.csv", index=False)

    log("Writing accepted key tables")
    key_table_stats = write_key_tables(paths, selection_table, specs, str(args.score_column), str(args.target_column))

    log("Preparing stream removal lookup")
    key_to_mask: dict[tuple[str, str, int, int, int], int] = {}
    for row, mask in zip(selection_table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), remove_mask_bits, strict=True):
        source, event, h, k, l = row
        key_to_mask[(str(source), str(event), int(h), int(k), int(l))] = int(mask)

    log("Writing stratified filter and matched-random stream variants")
    stream_summary, stream_stats = write_stream_variants(
        args.stream,
        paths,
        specs,
        key_to_mask,
        int(args.progress_every),
        None if args.max_events is None else int(args.max_events),
    )
    stream_summary.to_csv(args.output_root / "v5_target_excitation_stratified_filter_stream_summary.csv", index=False)
    write_run_metadata(args.output_root, args, specs, accepted_stats, v5_stats, selection_stats, stream_stats, key_table_stats)

    log("V5 target-excitation-stratified filter workflow complete")
    for spec in specs:
        print(f"Wrote: {paths[spec.variant]['stream']}")
        print(f"Wrote: {paths[spec.variant]['kept_keys_csv']}")
        print(f"Wrote: {paths[spec.variant]['removed_keys_csv']}")
    print(f"Wrote: {args.output_root / 'v5_target_excitation_stratified_filter_summary.csv'}")
    print(f"Wrote: {args.output_root / 'v5_target_excitation_stratified_filter_global_summary.csv'}")
    print(f"Wrote: {args.output_root / 'v5_target_excitation_stratified_filter_budget.csv'}")
    print(f"Wrote: {args.output_root / 'v5_target_excitation_stratified_filter_stream_summary.csv'}")
    print(f"Wrote: {args.output_root / 'run_metadata.json'}")
    print(global_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
