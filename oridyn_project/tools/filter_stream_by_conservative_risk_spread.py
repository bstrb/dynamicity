#!/usr/bin/env python3
"""Conservative signed-HKL OriDyn stream filtering by within-reflection risk spread.

This tool removes only the highest-risk observations from signed HKLs that pass
explicit safety gates: enough observations, enough kept observations, sufficient
within-HKL score spread, and an allowed resolution range. Ineligible HKLs and
observations with missing scores are kept unchanged.
"""

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

DEFAULT_SCORE_COLUMN = "badness_v3_excited_lowcoupling_shell_norm"
DEFAULT_FALLBACK_SCORE_COLUMN = "trust_risk_v3_target_gated_shell_norm"
DEFAULT_KEEP_FRACTIONS = [0.90, 0.80, 0.70, 0.60, 0.50]
DEFAULT_PROGRESS_EVERY = 1_000_000
DEFAULT_SCORES_CHUNKSIZE = 500_000

D_ANGSTROM_CANDIDATES = ["d_angstrom", "d_for_shell_angstrom", "d_A", "d"]
INV_NM_CANDIDATES = ["inv_nm", "inv_nm_for_shell", "q_inv_nm"]
EG_COLUMN = "target_excitation_Eg"
SHELL_COLUMN = "resolution_shell_index"

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="Input CrystFEL stream")
    parser.add_argument("--scores", required=True, type=Path, help="V3 score CSV")
    parser.add_argument("--output-root", required=True, type=Path, help="Fresh conservative filtering output directory")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN, help="Primary score/badness column")
    parser.add_argument(
        "--fallback-score-column",
        default=DEFAULT_FALLBACK_SCORE_COLUMN,
        help="Score column to use when --score-column is absent",
    )
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=DEFAULT_KEEP_FRACTIONS)
    parser.add_argument("--min-obs-to-filter", type=int, default=20)
    parser.add_argument("--min-keep-obs", type=int, default=10)
    parser.add_argument("--min-risk-spread-q90-q10", type=float, default=0.20)
    parser.add_argument("--min-risk-iqr", type=float, default=0.05)
    parser.add_argument("--filter-low-resolution-only", action="store_true")
    parser.add_argument("--max-filter-inv-nm", type=float, default=14.0)
    parser.add_argument(
        "--min-filter-d-angstrom",
        type=float,
        default=None,
        help="Equivalent conservative resolution gate: filter only if d_angstrom >= this value",
    )
    parser.add_argument(
        "--higher-score-is-worse",
        dest="higher_score_is_worse",
        action="store_true",
        default=True,
        help="Keep low scores and remove high scores (default; use for badness/risk columns)",
    )
    parser.add_argument(
        "--lower-score-is-worse",
        dest="higher_score_is_worse",
        action="store_false",
        help="Keep high scores and remove low scores",
    )
    parser.add_argument("--summarize-only", action="store_true", help="Only report HKL eligibility; do not write streams")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--scores-chunksize", type=int, default=DEFAULT_SCORES_CHUNKSIZE)
    parser.add_argument("--max-events", type=int, default=None, help="Smoke-test limit on stream crystal blocks/events")
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if int(args.min_obs_to_filter) < 1:
        raise SystemExit("--min-obs-to-filter must be >= 1")
    if int(args.min_keep_obs) < 1:
        raise SystemExit("--min-keep-obs must be >= 1")
    if float(args.min_risk_spread_q90_q10) < 0.0:
        raise SystemExit("--min-risk-spread-q90-q10 must be >= 0")
    if float(args.min_risk_iqr) < 0.0:
        raise SystemExit("--min-risk-iqr must be >= 0")
    if float(args.max_filter_inv_nm) <= 0.0:
        raise SystemExit("--max-filter-inv-nm must be > 0")
    if args.min_filter_d_angstrom is not None and float(args.min_filter_d_angstrom) <= 0.0:
        raise SystemExit("--min-filter-d-angstrom must be > 0")
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


def variant_name(fraction: float) -> str:
    return f"v3_conservative_spread_keep{percent_label(fraction):02d}"


def output_stream_name(fraction: float) -> str:
    return f"MFM300_VIII_v3_conservative_spread_keep{percent_label(fraction):02d}.stream"


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


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


def first_present(header: list[str], candidates: list[str]) -> str | None:
    for column in candidates:
        if column in header:
            return column
    return None


def resolve_score_column(header: list[str], args: argparse.Namespace) -> tuple[str, bool]:
    if args.score_column in header:
        return str(args.score_column), False
    if args.fallback_score_column and args.fallback_score_column in header:
        return str(args.fallback_score_column), True
    available = ", ".join(header)
    raise SystemExit(
        f"Score column {args.score_column!r} was not found, and fallback "
        f"{args.fallback_score_column!r} was not found. Available columns: {available}"
    )


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


def normalize_score_chunk(
    chunk: pd.DataFrame,
    score_column: str,
    d_column: str | None,
    inv_nm_column: str | None,
    has_eg: bool,
    has_shell: bool,
) -> pd.DataFrame:
    out = chunk.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if d_column is not None:
        d_values = pd.to_numeric(out[d_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        d_values = pd.Series(np.nan, index=out.index, dtype="float64")
    if inv_nm_column is not None:
        inv_values = pd.to_numeric(out[inv_nm_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        inv_values = pd.Series(np.nan, index=out.index, dtype="float64")

    d_missing = d_values.isna() & inv_values.notna() & (inv_values > 0)
    d_values = d_values.where(~d_missing, 10.0 / inv_values)
    inv_missing = inv_values.isna() & d_values.notna() & (d_values > 0)
    inv_values = inv_values.where(~inv_missing, 10.0 / d_values)

    out["d_angstrom"] = d_values
    out["inv_nm"] = inv_values
    if has_eg:
        out[EG_COLUMN] = pd.to_numeric(out[EG_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if has_shell:
        shell = out[SHELL_COLUMN]
        out["resolution_shell"] = shell.where(shell.notna(), "missing").astype(str)
    else:
        out["resolution_shell"] = "missing"

    bad_hkl = out[HKL_COLUMNS].isna().any(axis=1)
    out = out.loc[~bad_hkl].copy()
    if out.empty:
        return out
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")

    keep_cols = [*KEY_COLUMNS, score_column, "d_angstrom", "inv_nm", "resolution_shell"]
    if has_eg:
        keep_cols.append(EG_COLUMN)
    return out.loc[:, keep_cols].copy()


def filter_chunk_to_keys(
    chunk: pd.DataFrame,
    key_filter: set[tuple[str, str, int, int, int]],
) -> pd.DataFrame:
    if chunk.empty:
        return chunk
    keys = [
        build_key(source, event, int(h), int(k), int(l))
        for source, event, h, k, l in chunk.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    return chunk.loc[[key in key_filter for key in keys]].copy()


def load_scores(
    scores_path: Path,
    score_column: str,
    key_filter: set[tuple[str, str, int, int, int]] | None,
    chunksize: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(scores_path, nrows=0).columns.tolist()
    require_columns(header, [*KEY_COLUMNS, score_column], "score CSV")
    d_column = first_present(header, D_ANGSTROM_CANDIDATES)
    inv_nm_column = first_present(header, INV_NM_CANDIDATES)
    has_eg = EG_COLUMN in header
    has_shell = SHELL_COLUMN in header
    usecols = list(dict.fromkeys([*KEY_COLUMNS, score_column, d_column, inv_nm_column, EG_COLUMN if has_eg else None, SHELL_COLUMN if has_shell else None]))
    usecols = [column for column in usecols if column is not None]
    chunks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {
        "score_rows_read": 0,
        "score_rows_after_hkl_cleanup": 0,
        "score_rows_after_smoke_key_filter": 0,
        "d_angstrom_column": d_column,
        "inv_nm_column": inv_nm_column,
        "has_target_excitation_Eg": bool(has_eg),
        "has_resolution_shell_index": bool(has_shell),
    }
    for idx, chunk in enumerate(pd.read_csv(scores_path, usecols=usecols, chunksize=int(chunksize)), start=1):
        stats["score_rows_read"] += int(len(chunk))
        normalized = normalize_score_chunk(chunk, score_column, d_column, inv_nm_column, has_eg, has_shell)
        stats["score_rows_after_hkl_cleanup"] += int(len(normalized))
        if key_filter is not None:
            normalized = filter_chunk_to_keys(normalized, key_filter)
            stats["score_rows_after_smoke_key_filter"] += int(len(normalized))
        if not normalized.empty:
            chunks.append(normalized)
        if idx == 1 or idx % 5 == 0:
            extra = ""
            if key_filter is not None:
                extra = f", matched_smoke_rows={stats['score_rows_after_smoke_key_filter']:,}"
            log(f"Score CSV scan: chunks={idx:,}, rows_read={stats['score_rows_read']:,}{extra}")
    if chunks:
        table = pd.concat(chunks, ignore_index=True)
    else:
        table = pd.DataFrame(columns=[*KEY_COLUMNS, score_column, "d_angstrom", "inv_nm", "resolution_shell"])
        if has_eg:
            table[EG_COLUMN] = pd.Series(dtype="float64")
    return table, stats


def quantile_columns(grouped: pd.core.groupby.generic.DataFrameGroupBy, value_column: str, prefix: str) -> pd.DataFrame:
    q = grouped[value_column].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    q = q.rename(columns={0.10: f"{prefix}_q10", 0.25: f"{prefix}_q25", 0.50: f"{prefix}_q50", 0.75: f"{prefix}_q75", 0.90: f"{prefix}_q90"})
    return q.reset_index()


def value_quantiles(values: pd.Series) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {"q10": None, "median": None, "q90": None}
    q10, q50, q90 = np.quantile(finite, [0.10, 0.50, 0.90])
    return {"q10": float(q10), "median": float(q50), "q90": float(q90)}


def assign_primary_reason(groups: pd.DataFrame) -> pd.Series:
    reason = pd.Series("eligible", index=groups.index, dtype="object")
    reason = reason.mask(groups["missing_score"], "missing_score")
    reason = reason.mask((reason == "eligible") & groups["outside_resolution_range"], "outside_resolution_range")
    reason = reason.mask((reason == "eligible") & groups["low_count"], "low_count")
    reason = reason.mask((reason == "eligible") & groups["low_risk_spread"], "low_risk_spread")
    return reason


def build_hkl_summary(scored: pd.DataFrame, score_column: str, args: argparse.Namespace) -> pd.DataFrame:
    grouped = scored.groupby(HKL_COLUMNS, sort=False, dropna=False)
    base = grouped.agg(
        n_obs=(score_column, "size"),
        n_missing_score=(score_column, lambda values: int(pd.to_numeric(values, errors="coerce").isna().sum())),
        d_angstrom=("d_angstrom", "median"),
        inv_nm=("inv_nm", "median"),
        resolution_shell=("resolution_shell", "first"),
    ).reset_index()
    risk_q = quantile_columns(grouped, score_column, "risk")
    summary = base.merge(risk_q, on=HKL_COLUMNS, how="left")
    summary["risk_spread_q90_q10"] = summary["risk_q90"] - summary["risk_q10"]
    summary["risk_iqr"] = summary["risk_q75"] - summary["risk_q25"]

    if EG_COLUMN in scored.columns:
        eg_q = quantile_columns(grouped, EG_COLUMN, "target_excitation_Eg")
        keep = ["target_excitation_Eg_q10", "target_excitation_Eg_q50", "target_excitation_Eg_q90"]
        summary = summary.merge(eg_q.loc[:, [*HKL_COLUMNS, *keep]], on=HKL_COLUMNS, how="left")

    summary["missing_score"] = summary["n_missing_score"] > 0
    summary["low_count"] = (summary["n_obs"] < int(args.min_obs_to_filter)) | (summary["n_obs"] <= int(args.min_keep_obs))
    summary["low_risk_spread"] = (summary["risk_spread_q90_q10"] < float(args.min_risk_spread_q90_q10)) | (
        summary["risk_iqr"] < float(args.min_risk_iqr)
    )
    summary["low_risk_spread"] = summary["low_risk_spread"].fillna(True)

    if args.filter_low_resolution_only:
        if args.min_filter_d_angstrom is not None:
            resolution_ok = summary["d_angstrom"].notna() & (summary["d_angstrom"] >= float(args.min_filter_d_angstrom))
        else:
            resolution_ok = summary["inv_nm"].notna() & (summary["inv_nm"] <= float(args.max_filter_inv_nm))
        summary["outside_resolution_range"] = ~resolution_ok
    else:
        summary["outside_resolution_range"] = False

    summary["primary_reason"] = assign_primary_reason(summary)
    summary["eligible_for_filtering"] = summary["primary_reason"] == "eligible"
    return summary


def add_risk_deciles(work: pd.DataFrame, score_column: str) -> pd.DataFrame:
    out = work.copy()
    out["risk_decile"] = "missing"
    finite = out[score_column].notna()
    if bool(finite.any()):
        ranks = out.loc[finite, score_column].rank(method="first", pct=True)
        decile = np.ceil(ranks.to_numpy(dtype=float) * 10.0).astype(int)
        decile = np.clip(decile, 1, 10)
        out.loc[finite, "risk_decile"] = [f"q{value:02d}" for value in decile]
    return out


def build_filter_masks(
    scored: pd.DataFrame,
    score_column: str,
    args: argparse.Namespace,
) -> tuple[
    dict[tuple[str, str, int, int, int], int],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    if scored.empty:
        raise SystemExit("No usable score rows were loaded")

    work = scored.copy()
    duplicated = work.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(duplicated.sum())
    duplicate_keys = int(work.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact observation keys in scores; keeping first row per key")
        work = work.drop_duplicates(KEY_COLUMNS, keep="first").copy()

    hkl_summary = build_hkl_summary(work, score_column, args)
    merge_cols = [
        *HKL_COLUMNS,
        "n_obs",
        "d_angstrom",
        "inv_nm",
        "resolution_shell",
        "risk_q10",
        "risk_q25",
        "risk_q50",
        "risk_q75",
        "risk_q90",
        "risk_spread_q90_q10",
        "risk_iqr",
        "missing_score",
        "low_count",
        "low_risk_spread",
        "outside_resolution_range",
        "primary_reason",
        "eligible_for_filtering",
    ]
    if "target_excitation_Eg_q10" in hkl_summary.columns:
        merge_cols.extend(["target_excitation_Eg_q10", "target_excitation_Eg_q50", "target_excitation_Eg_q90"])
    work = work.merge(hkl_summary.loc[:, merge_cols], on=HKL_COLUMNS, how="left", suffixes=("", "_hkl"))
    work = add_risk_deciles(work, score_column)

    ascending_score = bool(args.higher_score_is_worse)
    work = work.sort_values(
        [*HKL_COLUMNS, score_column, "source_filename", "event"],
        ascending=[True, True, True, ascending_score, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    work["risk_rank_within_hkl"] = work.groupby(HKL_COLUMNS, sort=False).cumcount().astype("int64")

    remove_mask_bits = np.zeros(len(work), dtype=np.uint16)
    variant_rows: list[dict[str, Any]] = []
    removed_by_hkl_tables: dict[str, pd.DataFrame] = {}

    for bit_idx, keep_fraction in enumerate(args.keep_fractions):
        bit = np.uint16(1 << bit_idx)
        variant = variant_name(keep_fraction)
        n_keep = np.ceil(work["n_obs"].to_numpy(dtype=float) * float(keep_fraction)).astype("int64")
        n_keep = np.maximum(n_keep, int(args.min_keep_obs))
        n_keep = np.minimum(n_keep, work["n_obs"].to_numpy(dtype="int64"))
        keep = ~work["eligible_for_filtering"].to_numpy(dtype=bool)
        keep = keep | work[score_column].isna().to_numpy(dtype=bool)
        keep = keep | (work["risk_rank_within_hkl"].to_numpy(dtype="int64") < n_keep)
        remove = ~keep
        remove_mask_bits[remove] |= bit

        work[f"n_keep_{variant}"] = n_keep
        work[f"remove_{variant}"] = remove
        removed = work.loc[remove]
        kept = work.loc[~remove]
        removed_by_shell = removed.groupby("resolution_shell", dropna=False).size().astype(int).to_dict()
        removed_by_decile = removed.groupby("risk_decile", dropna=False).size().astype(int).to_dict()
        filtered_hkl_count = int(removed.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]) if not removed.empty else 0

        row = {
            "variant": variant,
            "keep_fraction": float(keep_fraction),
            "score_column": score_column,
            "score_direction": "higher_score_is_worse" if args.higher_score_is_worse else "lower_score_is_worse",
            "scored_observations": int(len(work)),
            "scored_observations_removed_by_selection": int(remove.sum()),
            "scored_observations_kept_by_selection": int((~remove).sum()),
            "unique_signed_hkls_total": int(hkl_summary.shape[0]),
            "unique_signed_hkls_filtered": filtered_hkl_count,
            "unique_signed_hkls_kept_unchanged_because_low_count": int((hkl_summary["primary_reason"] == "low_count").sum()),
            "unique_signed_hkls_kept_unchanged_because_low_risk_spread": int((hkl_summary["primary_reason"] == "low_risk_spread").sum()),
            "unique_signed_hkls_kept_unchanged_because_outside_resolution_range": int((hkl_summary["primary_reason"] == "outside_resolution_range").sum()),
            "unique_signed_hkls_kept_unchanged_because_missing_score": int((hkl_summary["primary_reason"] == "missing_score").sum()),
            "removed_observations_by_resolution_shell": json.dumps({str(k): int(v) for k, v in removed_by_shell.items()}, sort_keys=True),
            "removed_observations_by_risk_decile": json.dumps({str(k): int(v) for k, v in removed_by_decile.items()}, sort_keys=True),
        }
        for prefix, rows in [("removed", removed), ("kept", kept)]:
            stats = value_quantiles(rows[score_column])
            row[f"{prefix}_risk_q10"] = stats["q10"]
            row[f"{prefix}_risk_median"] = stats["median"]
            row[f"{prefix}_risk_q90"] = stats["q90"]
            if EG_COLUMN in work.columns:
                eg_stats = value_quantiles(rows[EG_COLUMN])
                row[f"{prefix}_target_excitation_Eg_q10"] = eg_stats["q10"]
                row[f"{prefix}_target_excitation_Eg_median"] = eg_stats["median"]
                row[f"{prefix}_target_excitation_Eg_q90"] = eg_stats["q90"]
        variant_rows.append(row)

        hkl_removed = removed.groupby(HKL_COLUMNS, as_index=False).size().rename(columns={"size": "removed_observations"})
        hkl_table = hkl_summary.merge(hkl_removed, on=HKL_COLUMNS, how="left")
        hkl_table["removed_observations"] = hkl_table["removed_observations"].fillna(0).astype("int64")
        hkl_table["keep_fraction"] = float(keep_fraction)
        hkl_table["n_keep"] = np.minimum(
            np.maximum(np.ceil(hkl_table["n_obs"].to_numpy(dtype=float) * float(keep_fraction)).astype("int64"), int(args.min_keep_obs)),
            hkl_table["n_obs"].to_numpy(dtype="int64"),
        )
        hkl_table["kept_observations"] = hkl_table["n_obs"].astype("int64") - hkl_table["removed_observations"].astype("int64")
        hkl_table = hkl_table.loc[hkl_table["removed_observations"] > 0].copy()
        removed_by_hkl_tables[variant] = hkl_table.sort_values(
            ["removed_observations", "h", "k", "l"], ascending=[False, True, True, True]
        )

    key_to_mask: dict[tuple[str, str, int, int, int], int] = {}
    for row, mask in zip(work.loc[:, KEY_COLUMNS].itertuples(index=False, name=None), remove_mask_bits, strict=True):
        source, event, h, k, l = row
        key_to_mask[(str(source), str(event), int(h), int(k), int(l))] = int(mask)

    reason_counts = hkl_summary["primary_reason"].value_counts().to_dict()
    selection_stats = {
        "score_rows_after_duplicate_cleanup": int(len(work)),
        "duplicate_score_rows": duplicate_rows,
        "duplicate_score_keys": duplicate_keys,
        "key_to_mask_entries": int(len(key_to_mask)),
        "unique_signed_hkls_total": int(hkl_summary.shape[0]),
        "unique_signed_hkls_eligible_for_filtering": int(hkl_summary["eligible_for_filtering"].sum()),
        "unique_signed_hkls_by_primary_reason": {str(k): int(v) for k, v in reason_counts.items()},
        "raw_unique_signed_hkls_low_count": int(hkl_summary["low_count"].sum()),
        "raw_unique_signed_hkls_low_risk_spread": int(hkl_summary["low_risk_spread"].sum()),
        "raw_unique_signed_hkls_outside_resolution_range": int(hkl_summary["outside_resolution_range"].sum()),
        "raw_unique_signed_hkls_missing_score": int(hkl_summary["missing_score"].sum()),
    }
    return key_to_mask, pd.DataFrame.from_records(variant_rows), hkl_summary, removed_by_hkl_tables, selection_stats


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


def ensure_no_overwrite(output_root: Path, paths: dict[str, dict[str, Path]], summarize_only: bool) -> None:
    if summarize_only:
        return
    blocked = []
    for variant_paths in paths.values():
        for path in variant_paths.values():
            if path.exists():
                blocked.append(path)
    for name in ["conservative_filter_sweep_summary.csv", "run_metadata.json"]:
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
    keep_fractions: list[float],
    progress_every: int,
    max_events: int | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[tuple[int, int, int], int]], dict[tuple[int, int, int], int]]:
    variants = [variant_name(fraction) for fraction in keep_fractions]
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
        row["keep_fraction"] = float(fraction)
        row["removed_fraction"] = row["removed_observations"] / max(row["total_observations_seen"], 1)
        row["removed_fraction_of_matched"] = row["removed_observations"] / max(row["matched_observations"], 1)
        row["output_stream_path"] = str(paths[variant]["stream"])
        rows.append(row)
    return pd.DataFrame.from_records(rows), stream_stats, removed_by_hkl, matched_hkl_counts


def write_removed_by_hkl_csvs(
    paths: dict[str, dict[str, Path]],
    selected_removed_tables: dict[str, pd.DataFrame],
    stream_removed_by_hkl: dict[str, dict[tuple[int, int, int], int]],
    matched_hkl_counts: dict[tuple[int, int, int], int],
) -> dict[str, int]:
    affected_counts: dict[str, int] = {}
    for variant, table in selected_removed_tables.items():
        out = table.copy()
        if out.empty:
            out = pd.DataFrame(
                columns=[
                    "h",
                    "k",
                    "l",
                    "n_obs",
                    "n_keep",
                    "removed_observations",
                    "stream_removed_observations",
                    "stream_matched_observations_for_hkl",
                ]
            )
        else:
            stream_counts = []
            matched_counts = []
            for h, k, l in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None):
                hkl = (int(h), int(k), int(l))
                stream_counts.append(int(stream_removed_by_hkl.get(variant, {}).get(hkl, 0)))
                matched_counts.append(int(matched_hkl_counts.get(hkl, 0)))
            out["stream_removed_observations"] = stream_counts
            out["stream_matched_observations_for_hkl"] = matched_counts
        out.to_csv(paths[variant]["removed_by_hkl_csv"], index=False)
        affected_counts[variant] = int(len(out))
    return affected_counts


def write_variant_jsons(
    paths: dict[str, dict[str, Path]],
    sweep_summary: pd.DataFrame,
    stream_stats: dict[str, Any],
    selection_stats: dict[str, Any],
    args: argparse.Namespace,
    score_column_used: str,
    fallback_used: bool,
) -> None:
    rows = {str(row.variant): row._asdict() for row in sweep_summary.itertuples(index=False)}
    for variant, variant_paths in paths.items():
        payload = {
            "variant": variant,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "stream": str(args.stream),
                "scores": str(args.scores),
            },
            "score_column_requested": str(args.score_column),
            "fallback_score_column": str(args.fallback_score_column),
            "score_column_used": score_column_used,
            "fallback_score_column_used": bool(fallback_used),
            "selection": selection_config(args),
            "selection_stats": selection_stats,
            "stream_stats": stream_stats,
            "variant_summary": rows.get(variant, {}),
            "outputs": {name: str(path) for name, path in variant_paths.items()},
        }
        variant_paths["summary_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def selection_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "keep_fractions": [float(value) for value in args.keep_fractions],
        "min_obs_to_filter": int(args.min_obs_to_filter),
        "min_keep_obs": int(args.min_keep_obs),
        "min_risk_spread_q90_q10": float(args.min_risk_spread_q90_q10),
        "min_risk_iqr": float(args.min_risk_iqr),
        "filter_low_resolution_only": bool(args.filter_low_resolution_only),
        "max_filter_inv_nm": float(args.max_filter_inv_nm),
        "min_filter_d_angstrom": None if args.min_filter_d_angstrom is None else float(args.min_filter_d_angstrom),
        "higher_score_is_worse": bool(args.higher_score_is_worse),
    }


def write_run_metadata(
    output_root: Path,
    args: argparse.Namespace,
    score_column_used: str,
    fallback_used: bool,
    score_load_stats: dict[str, Any],
    selection_stats: dict[str, Any],
    stream_stats: dict[str, Any],
) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "scores": str(args.scores),
        },
        "output_root": str(output_root),
        "score_column_requested": str(args.score_column),
        "fallback_score_column": str(args.fallback_score_column),
        "score_column_used": score_column_used,
        "fallback_score_column_used": bool(fallback_used),
        "selection": selection_config(args),
        "progress_every": int(args.progress_every),
        "scores_chunksize": int(args.scores_chunksize),
        "max_events": None if args.max_events is None else int(args.max_events),
        "score_load_stats": score_load_stats,
        "selection_stats": selection_stats,
        "stream_stats": stream_stats,
        "warnings": [
            "Observation matching uses exact source_filename + event + signed h,k,l.",
            "HKLs are not canonicalized.",
            "Unmatched stream observations and missing-score observations are kept by default.",
            "This is a separate conservative v3 filtering experiment and does not modify score generation.",
        ],
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def summarize_only_report(hkl_summary: pd.DataFrame, selection_stats: dict[str, Any], args: argparse.Namespace) -> None:
    print("Conservative v3 spread-filter eligibility summary")
    print(f"Score-table signed HKLs: {selection_stats['unique_signed_hkls_total']:,}")
    print(f"Eligible signed HKLs: {selection_stats['unique_signed_hkls_eligible_for_filtering']:,}")
    for reason, count in selection_stats["unique_signed_hkls_by_primary_reason"].items():
        print(f"Primary reason {reason}: {int(count):,}")
    print(f"min_obs_to_filter: {int(args.min_obs_to_filter)}")
    print(f"min_keep_obs: {int(args.min_keep_obs)}")
    print(f"min_risk_spread_q90_q10: {float(args.min_risk_spread_q90_q10):.3f}")
    print(f"min_risk_iqr: {float(args.min_risk_iqr):.3f}")
    if args.filter_low_resolution_only:
        if args.min_filter_d_angstrom is not None:
            print(f"Filtering only reflections with d >= {float(args.min_filter_d_angstrom):.3f} Å")
        else:
            print(f"Filtering only reflections with 1/d <= {float(args.max_filter_inv_nm):.3f} nm^-1")
    else:
        print("Resolution gate disabled")
    cols = [
        "h",
        "k",
        "l",
        "n_obs",
        "d_angstrom",
        "inv_nm",
        "risk_spread_q90_q10",
        "risk_iqr",
        "primary_reason",
    ]
    preview = hkl_summary.loc[:, [column for column in cols if column in hkl_summary.columns]].head(20)
    print("\nFirst 20 signed-HKL eligibility rows:")
    print(preview.to_string(index=False))


def main() -> int:
    args = parse_args()
    header = pd.read_csv(args.scores, nrows=0).columns.tolist()
    score_column_used, fallback_used = resolve_score_column(header, args)
    if fallback_used:
        log(f"Primary score column {args.score_column!r} absent; using fallback {score_column_used!r}")
    else:
        log(f"Using score column: {score_column_used}")
    if args.higher_score_is_worse:
        log("Score direction: higher values are worse; eligible HKLs keep the lowest scores")
    else:
        log("Score direction: lower values are worse; eligible HKLs keep the highest scores")
    if args.filter_low_resolution_only:
        if args.min_filter_d_angstrom is not None:
            log(f"Filtering only reflections with d >= {float(args.min_filter_d_angstrom):.3f} Å")
        else:
            log(f"Filtering only reflections with 1/d <= {float(args.max_filter_inv_nm):.3f} nm^-1")
    else:
        log("Resolution gate disabled; all resolution ranges are eligible if other gates pass")

    smoke_key_filter = None
    smoke_stats: dict[str, int] = {}
    if args.max_events is not None:
        log(f"Collecting smoke-test keys for first {int(args.max_events)} stream events/crystals")
        smoke_key_filter, smoke_stats = collect_smoke_keys(args.stream, int(args.max_events), int(args.progress_every))
        log(f"Smoke key collection done: keys={len(smoke_key_filter):,}")

    log("Loading score table")
    score_table, score_load_stats = load_scores(args.scores, score_column_used, smoke_key_filter, int(args.scores_chunksize))
    score_load_stats.update(smoke_stats)
    log(f"Loaded scores: rows={len(score_table):,}")

    log("Building conservative per-signed-HKL eligibility and removal masks")
    key_to_mask, variant_selection, hkl_summary, removed_by_hkl_tables, selection_stats = build_filter_masks(
        score_table, score_column_used, args
    )
    log(
        "Eligibility ready: "
        f"signed_hkls={selection_stats['unique_signed_hkls_total']:,}, "
        f"eligible={selection_stats['unique_signed_hkls_eligible_for_filtering']:,}, "
        f"lookup_keys={selection_stats['key_to_mask_entries']:,}"
    )

    if args.summarize_only:
        summarize_only_report(hkl_summary, selection_stats, args)
        return 0

    paths = output_paths(args.output_root, args.keep_fractions)
    ensure_no_overwrite(args.output_root, paths, bool(args.summarize_only))
    log(f"Output root: {args.output_root}")
    for fraction in args.keep_fractions:
        variant = variant_name(fraction)
        log(f"Prepared output stream for keep_fraction={fraction:.3f}: {paths[variant]['stream']}")

    log("Writing conservative stream variants in one stream pass")
    stream_summary, stream_stats, stream_removed_by_hkl, matched_hkl_counts = write_stream_variants(
        args.stream,
        paths,
        key_to_mask,
        args.keep_fractions,
        int(args.progress_every),
        None if args.max_events is None else int(args.max_events),
    )
    affected_counts = write_removed_by_hkl_csvs(paths, removed_by_hkl_tables, stream_removed_by_hkl, matched_hkl_counts)
    stream_summary["number_of_signed_hkls_affected"] = stream_summary["variant"].map(affected_counts).fillna(0).astype("int64")
    sweep_summary = stream_summary.merge(variant_selection, on=["variant", "keep_fraction"], how="left")
    sweep_summary.to_csv(args.output_root / "conservative_filter_sweep_summary.csv", index=False)
    write_variant_jsons(paths, sweep_summary, stream_stats, selection_stats, args, score_column_used, fallback_used)
    write_run_metadata(args.output_root, args, score_column_used, fallback_used, score_load_stats, selection_stats, stream_stats)

    log("Conservative v3 spread filtering sweep complete")
    for variant_paths in paths.values():
        print(f"Wrote: {variant_paths['stream']}")
        print(f"Wrote: {variant_paths['summary_json']}")
        print(f"Wrote: {variant_paths['removed_by_hkl_csv']}")
    print(f"Wrote: {args.output_root / 'conservative_filter_sweep_summary.csv'}")
    print(f"Wrote: {args.output_root / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())