#!/usr/bin/env python3
"""Diagnose enhancement-feed shifts in partialator unmerged observations.

This standalone diagnostic joins CrystFEL partialator --unmerged-output rows to
reflection_scores_with_enh_feed.csv by source_filename + event + signed h,k,l.
It then asks, within each signed HKL, whether high enhancement-feed observations
have larger unmerged intensity contribution than low enhancement-feed
observations.

No symmetry canonicalization is applied and no intensities are changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
UNMERGED_FLAGGED_RE = re.compile(r"^\s*Flagged:\s*(\S+)\s*$", re.IGNORECASE)
TOP_ROWS = 100
SUMMARY_TABLE_ROWS = 12
POOR_MATCH_RATE = 0.50
STRENGTH_CLASSES = ["weak", "middle", "strong"]
STRENGTH_SPOTLIGHT_COLUMNS = [
    "h",
    "k",
    "l",
    "n_obs",
    "n_low",
    "n_high",
    "relative_shift_Iweight",
    "relative_shift_I",
    "relative_shift_weight",
    "Iweight_low_median",
    "Iweight_high_median",
    "enh_feed_raw_p95_minus_median",
]
SHIFT_COLUMNS = [
    "h",
    "k",
    "l",
    "n_obs",
    "n_low",
    "n_high",
    "I_low_median",
    "I_high_median",
    "weight_low_median",
    "weight_high_median",
    "Iweight_low_median",
    "Iweight_high_median",
    "relative_shift_I",
    "relative_shift_weight",
    "relative_shift_Iweight",
    "enh_feed_raw_median",
    "enh_feed_raw_p95",
    "enh_feed_raw_p95_minus_median",
    "fraction_rank_gt_0.95",
    "graph_crowding_norm_median",
    "frame_axis_risk_norm_median",
]
STRENGTH_SUMMARY_COLUMNS = [
    "local_strength_class",
    "n_HKLs",
    "median_relative_shift_Iweight",
    "fraction_positive_Iweight",
    "median_relative_shift_I",
    "fraction_positive_I",
    "median_relative_shift_weight",
    "fraction_positive_weight",
]

REQUIRED_SCORE_COLUMNS = [
    *KEY_COLUMNS,
    "enh_feed_raw",
    "enh_feed_rank_frame",
]
OPTIONAL_SCORE_COLUMNS = [
    "enh_feed_norm_frame",
    "enh_feed_log_raw",
    "enh_feed_frame_p95_raw",
    "enh_feed_norm_frame_clipped",
    "enh_feed_n_predicted_q",
    "enh_feed_n_valid_paths",
    "enh_feed_top_path_score",
    "graph_crowding_norm",
    "frame_axis_risk_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmerged-hkl", required=True, type=Path, help="partialator --unmerged-output HKL file")
    parser.add_argument("--scores-csv", required=True, type=Path, help="reflection_scores_with_enh_feed.csv")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--stream", type=Path, default=None)
    parser.add_argument("--symmetry", default="unknown")
    parser.add_argument("--partialator-model", default="unknown")
    parser.add_argument("--partialator-iterations", default="unknown")
    parser.add_argument(
        "--partialator-post-refinement",
        choices=["enabled", "disabled", "unknown"],
        default="unknown",
    )
    parser.add_argument(
        "--partialator-bscale",
        choices=["enabled", "disabled", "unknown"],
        default="unknown",
    )
    parser.add_argument("--notes", default="")

    parser.add_argument(
        "--allow-basename-fallback",
        action="store_true",
        help="If exact source path matching is poor, retry matching by basename + event + signed HKL.",
    )
    parser.add_argument(
        "--include-flagged-crystals",
        action="store_true",
        help="Include crystal blocks marked 'Flagged: yes' instead of excluding them.",
    )
    parser.add_argument(
        "--include-row-flags",
        action="store_true",
        help="Include rows flagged partiality_too_small or nan_esd instead of excluding them.",
    )
    parser.add_argument("--min-obs", type=int, default=50)
    parser.add_argument("--low-rank-max", type=float, default=0.20)
    parser.add_argument("--high-rank-min", type=float, default=0.90)
    parser.add_argument("--min-tail-obs", type=int, default=10)
    args = parser.parse_args()

    if not args.unmerged_hkl.exists():
        raise SystemExit(f"--unmerged-hkl not found: {args.unmerged_hkl}")
    if not args.scores_csv.exists():
        raise SystemExit(f"--scores-csv not found: {args.scores_csv}")
    if args.stream is not None and not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if int(args.min_obs) < 1:
        raise SystemExit("--min-obs must be >= 1")
    if int(args.min_tail_obs) < 1:
        raise SystemExit("--min-tail-obs must be >= 1")
    if not (0.0 <= float(args.low_rank_max) <= 1.0):
        raise SystemExit("--low-rank-max must be between 0 and 1")
    if not (0.0 <= float(args.high_rank_min) <= 1.0):
        raise SystemExit("--high-rank-min must be between 0 and 1")
    if float(args.low_rank_max) >= float(args.high_rank_min):
        raise SystemExit("--low-rank-max must be smaller than --high-rank-min")
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def prepare_output_dir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise SystemExit(f"{outdir} exists and is not empty; pass --overwrite to reuse it.")
    outdir.mkdir(parents=True, exist_ok=True)


def normalize_source(value: Any) -> str:
    return str(value).strip()


def normalize_event(value: Any) -> str:
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def source_basename(value: Any) -> str:
    text = normalize_source(value)
    return Path(text).name


def parse_flagged_value(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "t", "1"}


def normalize_key_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=HKL_COLUMNS)
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def duplicate_summary(table: pd.DataFrame, key_columns: list[str]) -> dict[str, int]:
    if table.empty:
        return {"duplicate_key_rows": 0, "duplicate_keys": 0}
    duplicated = table.duplicated(key_columns, keep=False)
    return {
        "duplicate_key_rows": int(duplicated.sum()),
        "duplicate_keys": int(table.loc[duplicated, key_columns].drop_duplicates().shape[0]),
    }


def parse_unmerged_observations(
    path: Path,
    include_flagged_crystals: bool,
    include_row_flags: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    current_crystal_id: int | str | None = None
    current_source = ""
    current_event = ""
    current_flagged = False

    stats = {
        "crystal_blocks_seen": 0,
        "unmerged_reflection_rows_seen": 0,
        "unmerged_reflection_rows_kept": 0,
        "excluded_flagged_crystal_rows": 0,
        "excluded_partiality_too_small_rows": 0,
        "excluded_nan_esd_rows": 0,
        "excluded_nonfinite_intensity_or_weight_rows": 0,
        "malformed_unmerged_rows": 0,
    }

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                stats["crystal_blocks_seen"] += 1
                token = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else ""
                try:
                    current_crystal_id = int(token)
                except ValueError:
                    current_crystal_id = token
                current_source = ""
                current_event = ""
                current_flagged = False
                continue

            filename_match = UNMERGED_FILENAME_RE.match(line)
            if filename_match:
                current_source = normalize_source(filename_match.group(1))
                current_event = normalize_event(filename_match.group(2) or "")
                continue

            flagged_match = UNMERGED_FLAGGED_RE.match(line)
            if flagged_match:
                current_flagged = parse_flagged_value(flagged_match.group(1))
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                intensity = float(parts[3])
                weight = float(parts[4])
            except ValueError:
                stats["malformed_unmerged_rows"] += 1
                continue

            stats["unmerged_reflection_rows_seen"] += 1
            row_flags = " ".join(parts[5:]).strip()
            row_flags_lower = row_flags.lower()

            has_partiality_too_small = "partiality_too_small" in row_flags_lower
            has_nan_esd = "nan_esd" in row_flags_lower

            if current_flagged and not include_flagged_crystals:
                stats["excluded_flagged_crystal_rows"] += 1
                continue
            if not include_row_flags and (has_partiality_too_small or has_nan_esd):
                if has_partiality_too_small:
                    stats["excluded_partiality_too_small_rows"] += 1
                if has_nan_esd:
                    stats["excluded_nan_esd_rows"] += 1
                continue
            if not np.isfinite(intensity) or not np.isfinite(weight):
                stats["excluded_nonfinite_intensity_or_weight_rows"] += 1
                continue

            rows.append(
                {
                    "crystal_id": current_crystal_id,
                    "source_filename": current_source,
                    "event": current_event,
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "I_unmerged": float(intensity),
                    "partialator_weight": float(weight),
                    "I_times_weight": float(intensity * weight),
                    "row_flags": row_flags,
                    "crystal_flagged": bool(current_flagged),
                    "unmerged_line_number": int(line_number),
                }
            )
            stats["unmerged_reflection_rows_kept"] += 1

    columns = [
        "crystal_id",
        *KEY_COLUMNS,
        "I_unmerged",
        "partialator_weight",
        "I_times_weight",
        "row_flags",
        "crystal_flagged",
        "unmerged_line_number",
    ]
    table = pd.DataFrame.from_records(rows, columns=columns)
    if not table.empty:
        table = normalize_key_columns(table)
        table["unmerged_row_id"] = np.arange(len(table), dtype=np.int64)
        counts = table.groupby(KEY_COLUMNS, sort=False).size().rename("unmerged_key_n_rows").reset_index()
        table = table.merge(counts, on=KEY_COLUMNS, how="left")
    else:
        table["unmerged_row_id"] = pd.Series(dtype="int64")
        table["unmerged_key_n_rows"] = pd.Series(dtype="int64")

    stats.update({f"unmerged_{key}": value for key, value in duplicate_summary(table, KEY_COLUMNS).items()})
    return table, stats


def load_scores(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [column for column in REQUIRED_SCORE_COLUMNS if column not in header]
    if missing:
        raise SystemExit(f"--scores-csv is missing required column(s): {missing}")

    usecols = [column for column in [*REQUIRED_SCORE_COLUMNS, *OPTIONAL_SCORE_COLUMNS] if column in header]
    scores = pd.read_csv(path, usecols=usecols)
    scores = normalize_key_columns(scores)
    scores["score_source_filename"] = scores["source_filename"]

    numeric_columns = [
        column
        for column in scores.columns
        if column not in {"source_filename", "score_source_filename", "event"}
    ]
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores[HKL_COLUMNS] = scores[HKL_COLUMNS].astype("int64")

    counts = scores.groupby(KEY_COLUMNS, sort=False).size().rename("score_key_n_rows").reset_index()
    scores = scores.merge(counts, on=KEY_COLUMNS, how="left")
    stats = duplicate_summary(scores, KEY_COLUMNS)
    stats = {f"scores_{key}": value for key, value in stats.items()}
    stats["scores_rows_loaded"] = int(len(scores))
    stats["scores_unique_keys"] = int(scores[KEY_COLUMNS].drop_duplicates().shape[0])
    return scores, stats


def deduplicate_scores(scores: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    return scores.drop_duplicates(key_columns, keep="first").copy()


def join_with_scores(
    unmerged: pd.DataFrame,
    scores: pd.DataFrame,
    allow_basename_fallback: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if unmerged.empty:
        return unmerged.copy(), {
            "match_mode": "exact",
            "exact_matches": 0,
            "exact_match_rate": 0.0,
            "basename_fallback_attempted": False,
            "basename_matches": 0,
            "basename_match_rate": 0.0,
        }

    exact_scores = deduplicate_scores(scores, KEY_COLUMNS)
    exact_join = unmerged.merge(exact_scores, on=KEY_COLUMNS, how="left", suffixes=("", "_score"))
    exact_join["score_matched"] = exact_join["enh_feed_raw"].notna()
    exact_matches = int(exact_join["score_matched"].sum())
    exact_rate = float(exact_matches / max(len(unmerged), 1))

    stats: dict[str, Any] = {
        "match_mode": "exact",
        "exact_matches": exact_matches,
        "exact_match_rate": exact_rate,
        "basename_fallback_attempted": False,
        "basename_matches": 0,
        "basename_match_rate": 0.0,
        "basename_fallback_used": False,
        "basename_fallback_reason": "",
    }

    if exact_rate >= POOR_MATCH_RATE or not allow_basename_fallback:
        if exact_rate < POOR_MATCH_RATE:
            stats["basename_fallback_reason"] = (
                "Exact match rate was poor, but --allow-basename-fallback was not set."
            )
        exact_join["match_mode"] = "exact"
        return exact_join, stats

    basename_unmerged = unmerged.copy()
    basename_scores = scores.copy()
    basename_unmerged["source_basename"] = basename_unmerged["source_filename"].map(source_basename)
    basename_scores["source_basename"] = basename_scores["source_filename"].map(source_basename)
    basename_key = ["source_basename", "event", "h", "k", "l"]
    basename_scores = deduplicate_scores(basename_scores, basename_key)
    basename_join = basename_unmerged.merge(
        basename_scores,
        on=basename_key,
        how="left",
        suffixes=("", "_score"),
    )
    basename_join["score_matched"] = basename_join["enh_feed_raw"].notna()
    basename_matches = int(basename_join["score_matched"].sum())
    basename_rate = float(basename_matches / max(len(unmerged), 1))
    stats.update(
        {
            "basename_fallback_attempted": True,
            "basename_matches": basename_matches,
            "basename_match_rate": basename_rate,
        }
    )

    if basename_matches > exact_matches:
        stats["match_mode"] = "basename"
        stats["basename_fallback_used"] = True
        stats["basename_fallback_reason"] = (
            f"Exact match rate {exact_rate:.3f} was below {POOR_MATCH_RATE:.2f}; "
            "basename fallback improved the match count."
        )
        basename_join["source_filename"] = basename_join["source_filename"].map(normalize_source)
        basename_join["match_mode"] = "basename"
        return basename_join, stats

    stats["basename_fallback_reason"] = (
        f"Exact match rate {exact_rate:.3f} was below {POOR_MATCH_RATE:.2f}; "
        "basename fallback did not improve the match count."
    )
    exact_join["match_mode"] = "exact"
    return exact_join, stats


def finite_matched_observations(joined: pd.DataFrame) -> pd.DataFrame:
    required = ["score_matched", "I_unmerged", "partialator_weight", "I_times_weight", "enh_feed_raw", "enh_feed_rank_frame"]
    if joined.empty:
        return joined.copy()
    mask = joined["score_matched"].astype(bool)
    for column in required[1:]:
        mask &= pd.to_numeric(joined[column], errors="coerce").map(np.isfinite)
    return joined.loc[mask].copy()


def median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def relative_shift(high: float, low: float, floor: float) -> float:
    if not np.isfinite(high) or not np.isfinite(low):
        return np.nan
    return float((high - low) / max(abs(low), floor))


def analyze_hkl_shifts(
    joined: pd.DataFrame,
    min_obs: int,
    low_rank_max: float,
    high_rank_min: float,
    min_tail_obs: int,
) -> pd.DataFrame:
    matched = finite_matched_observations(joined)
    rows: list[dict[str, Any]] = []
    if matched.empty:
        return pd.DataFrame(columns=SHIFT_COLUMNS)

    for (h, k, l), group in matched.groupby(HKL_COLUMNS, sort=True):
        n_obs = int(len(group))
        if n_obs < int(min_obs):
            continue
        low = group.loc[group["enh_feed_rank_frame"] <= float(low_rank_max)]
        high = group.loc[group["enh_feed_rank_frame"] >= float(high_rank_min)]
        n_low = int(len(low))
        n_high = int(len(high))
        if n_low < int(min_tail_obs) or n_high < int(min_tail_obs):
            continue

        i_low = median(low["I_unmerged"])
        i_high = median(high["I_unmerged"])
        weight_low = median(low["partialator_weight"])
        weight_high = median(high["partialator_weight"])
        iweight_low = median(low["I_times_weight"])
        iweight_high = median(high["I_times_weight"])
        raw_median = median(group["enh_feed_raw"])
        raw_p95 = quantile(group["enh_feed_raw"], 0.95)

        row: dict[str, Any] = {
            "h": int(h),
            "k": int(k),
            "l": int(l),
            "n_obs": n_obs,
            "n_low": n_low,
            "n_high": n_high,
            "I_low_median": i_low,
            "I_high_median": i_high,
            "weight_low_median": weight_low,
            "weight_high_median": weight_high,
            "Iweight_low_median": iweight_low,
            "Iweight_high_median": iweight_high,
            "relative_shift_I": relative_shift(i_high, i_low, 1.0),
            "relative_shift_weight": relative_shift(weight_high, weight_low, 1e-6),
            "relative_shift_Iweight": relative_shift(iweight_high, iweight_low, 1.0),
            "enh_feed_raw_median": raw_median,
            "enh_feed_raw_p95": raw_p95,
            "enh_feed_raw_p95_minus_median": raw_p95 - raw_median if np.isfinite(raw_p95) and np.isfinite(raw_median) else np.nan,
            "fraction_rank_gt_0.95": float((group["enh_feed_rank_frame"] > 0.95).mean()),
        }
        if "graph_crowding_norm" in group.columns:
            row["graph_crowding_norm_median"] = median(group["graph_crowding_norm"])
        if "frame_axis_risk_norm" in group.columns:
            row["frame_axis_risk_norm_median"] = median(group["frame_axis_risk_norm"])
        rows.append(row)

    return pd.DataFrame.from_records(rows, columns=SHIFT_COLUMNS)


def add_local_strength_class(shifts: pd.DataFrame) -> pd.DataFrame:
    out = shifts.copy()
    out["local_strength_class"] = pd.NA
    if out.empty or "Iweight_low_median" not in out.columns:
        return out

    baseline = pd.to_numeric(out["Iweight_low_median"], errors="coerce")
    finite = baseline.map(np.isfinite)
    ordered_index = baseline.loc[finite].sort_values(kind="mergesort").index
    n_values = len(ordered_index)
    if n_values == 0:
        return out

    class_ids = np.minimum((np.arange(n_values) * len(STRENGTH_CLASSES)) // n_values, len(STRENGTH_CLASSES) - 1)
    labels = [STRENGTH_CLASSES[int(class_id)] for class_id in class_ids]
    out.loc[ordered_index, "local_strength_class"] = labels
    return out


def fraction_positive(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float((values > 0.0).mean()) if not values.empty else np.nan


def summarize_local_strength_classes(shifts_with_strength: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strength_class in STRENGTH_CLASSES:
        group = shifts_with_strength.loc[shifts_with_strength["local_strength_class"] == strength_class]
        rows.append(
            {
                "local_strength_class": strength_class,
                "n_HKLs": int(len(group)),
                "median_relative_shift_Iweight": median(group["relative_shift_Iweight"]) if not group.empty else np.nan,
                "fraction_positive_Iweight": fraction_positive(group["relative_shift_Iweight"]) if not group.empty else np.nan,
                "median_relative_shift_I": median(group["relative_shift_I"]) if not group.empty else np.nan,
                "fraction_positive_I": fraction_positive(group["relative_shift_I"]) if not group.empty else np.nan,
                "median_relative_shift_weight": median(group["relative_shift_weight"]) if not group.empty else np.nan,
                "fraction_positive_weight": fraction_positive(group["relative_shift_weight"]) if not group.empty else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows, columns=STRENGTH_SUMMARY_COLUMNS)


def strength_spotlight_tables(shifts_with_strength: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weak_positive = shifts_with_strength.loc[shifts_with_strength["local_strength_class"] == "weak"].copy()
    strong_negative = shifts_with_strength.loc[shifts_with_strength["local_strength_class"] == "strong"].copy()
    weak_positive = weak_positive.sort_values("relative_shift_Iweight", ascending=False).head(TOP_ROWS)
    strong_negative = strong_negative.sort_values("relative_shift_Iweight", ascending=True).head(TOP_ROWS)
    return weak_positive, strong_negative


def format_stats(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return (
        f"median={float(values.median()):.6g}, "
        f"p95={float(values.quantile(0.95)):.6g}, "
        f"max={float(values.max()):.6g}"
    )


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = SUMMARY_TABLE_ROWS) -> str:
    if table is None or table.empty:
        return "_No rows._"
    view = table.loc[:, [column for column in columns if column in table.columns]].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if not isinstance(value, (list, tuple, dict, str, bytes)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def write_metadata(
    outdir: Path,
    args: argparse.Namespace,
    unmerged_stats: dict[str, Any],
    score_stats: dict[str, Any],
    join_stats: dict[str, Any],
    analysis_stats: dict[str, Any],
) -> None:
    metadata = {
        "script": Path(__file__).name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "unmerged_hkl": str(args.unmerged_hkl),
            "scores_csv": str(args.scores_csv),
            "stream": None if args.stream is None else str(args.stream),
        },
        "partialator_metadata": {
            "symmetry": str(args.symmetry),
            "model": str(args.partialator_model),
            "iterations": str(args.partialator_iterations),
            "post_refinement": str(args.partialator_post_refinement),
            "bscale": str(args.partialator_bscale),
            "notes": str(args.notes),
        },
        "filters": {
            "include_flagged_crystals": bool(args.include_flagged_crystals),
            "include_row_flags": bool(args.include_row_flags),
            "min_obs": int(args.min_obs),
            "low_rank_max": float(args.low_rank_max),
            "high_rank_min": float(args.high_rank_min),
            "min_tail_obs": int(args.min_tail_obs),
        },
        "matching": {
            "allow_basename_fallback": bool(args.allow_basename_fallback),
            **join_stats,
        },
        "stats": {
            **unmerged_stats,
            **score_stats,
            **analysis_stats,
        },
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=jsonable), encoding="utf-8")


def write_summary(
    outdir: Path,
    args: argparse.Namespace,
    unmerged_stats: dict[str, Any],
    score_stats: dict[str, Any],
    join_stats: dict[str, Any],
    joined: pd.DataFrame,
    shifts: pd.DataFrame,
    strength_summary: pd.DataFrame,
    weak_positive: pd.DataFrame,
    strong_negative: pd.DataFrame,
) -> None:
    matched = finite_matched_observations(joined)
    positive = shifts.sort_values("relative_shift_Iweight", ascending=False) if not shifts.empty else shifts
    negative = shifts.sort_values("relative_shift_Iweight", ascending=True) if not shifts.empty else shifts
    table_columns = [
        "h",
        "k",
        "l",
        "n_obs",
        "n_low",
        "n_high",
        "relative_shift_Iweight",
        "Iweight_low_median",
        "Iweight_high_median",
        "enh_feed_raw_p95_minus_median",
    ]

    lines = [
        "# Enhancement-Feed Shift From Partialator Unmerged",
        "",
        "## Scope And Caveats",
        "",
        "- This script does not assume a specific symmetry, partiality model, number of iterations, PR setting, or B-scale setting.",
        "- The interpretation depends on the partialator settings supplied by the user.",
        "- If symmetry is not `1`, signed-HKL orientation-specific interpretation may be compromised because symmetry handling can canonicalize or duplicate observations.",
        "- If post-refinement was enabled, orientation/partiality refinement may absorb or reshape the orientation-dependent effect.",
        "- For the clean OriDyn feed diagnostic, recommended metadata is: symmetry `1`, model `offset`, iterations `1`, post-refinement `disabled`, B-scale `disabled`.",
        "- Column 4 in the unmerged file is treated as `I_unmerged`; column 5 is treated as `partialator_weight`, a CrystFEL combined weight/partiality/scale-like value, not sigma.",
        "- `I_times_weight` is computed as `I_unmerged * partialator_weight`.",
        "- No symmetry canonicalization, filtering-by-score, reweighting, or intensity modification is performed.",
        "",
        "## Supplied Metadata",
        "",
        f"- Stream: `{args.stream}`",
        f"- Symmetry: `{args.symmetry}`",
        f"- Partialator model: `{args.partialator_model}`",
        f"- Partialator iterations: `{args.partialator_iterations}`",
        f"- Post-refinement: `{args.partialator_post_refinement}`",
        f"- B-scale: `{args.partialator_bscale}`",
        f"- Notes: {args.notes if args.notes else 'n/a'}",
        "",
        "## Counts",
        "",
        f"- Unmerged reflection rows seen: {unmerged_stats.get('unmerged_reflection_rows_seen', 0)}",
        f"- Unmerged rows kept after filters: {unmerged_stats.get('unmerged_reflection_rows_kept', 0)}",
        f"- Score rows loaded: {score_stats.get('scores_rows_loaded', 0)}",
        f"- Exact duplicate unmerged key rows: {unmerged_stats.get('unmerged_duplicate_key_rows', 0)}",
        f"- Exact duplicate unmerged keys: {unmerged_stats.get('unmerged_duplicate_keys', 0)}",
        f"- Exact duplicate score key rows: {score_stats.get('scores_duplicate_key_rows', 0)}",
        f"- Exact duplicate score keys: {score_stats.get('scores_duplicate_keys', 0)}",
        f"- Match mode used: {join_stats.get('match_mode', 'unknown')}",
        f"- Exact matches: {join_stats.get('exact_matches', 0)} ({join_stats.get('exact_match_rate', 0.0):.3%})",
        f"- Basename fallback attempted: {join_stats.get('basename_fallback_attempted', False)}",
        f"- Basename fallback used: {join_stats.get('basename_fallback_used', False)}",
        f"- Basename matches: {join_stats.get('basename_matches', 0)} ({join_stats.get('basename_match_rate', 0.0):.3%})",
        f"- Matched finite observations used for tail analysis: {len(matched)}",
        f"- Signed HKLs passing tail filters: {len(shifts)}",
        "",
        "## Filters",
        "",
        f"- Include flagged crystals: {bool(args.include_flagged_crystals)}",
        f"- Include row flags (`partiality_too_small`, `nan_esd`): {bool(args.include_row_flags)}",
        f"- Min observations per signed HKL: {int(args.min_obs)}",
        f"- Low-feed tail: `enh_feed_rank_frame <= {float(args.low_rank_max):.3g}`",
        f"- High-feed tail: `enh_feed_rank_frame >= {float(args.high_rank_min):.3g}`",
        f"- Min observations in each tail: {int(args.min_tail_obs)}",
        "",
        "## Joined Score Distributions",
        "",
        f"- I_unmerged: {format_stats(matched['I_unmerged']) if not matched.empty else 'n/a'}",
        f"- partialator_weight: {format_stats(matched['partialator_weight']) if not matched.empty else 'n/a'}",
        f"- I_times_weight: {format_stats(matched['I_times_weight']) if not matched.empty else 'n/a'}",
        f"- enh_feed_raw: {format_stats(matched['enh_feed_raw']) if not matched.empty else 'n/a'}",
        f"- enh_feed_rank_frame: {format_stats(matched['enh_feed_rank_frame']) if not matched.empty else 'n/a'}",
        "",
    ]
    if join_stats.get("basename_fallback_reason"):
        lines.extend(["## Match Note", "", f"- {join_stats['basename_fallback_reason']}", ""])

    if str(args.symmetry).strip() not in {"1", "unknown", ""}:
        lines.extend(
            [
                "## Symmetry Warning",
                "",
                f"- Supplied symmetry is `{args.symmetry}` rather than `1`; signed-HKL orientation-specific interpretation may be compromised.",
                "",
            ]
        )
    if args.partialator_post_refinement == "enabled":
        lines.extend(
            [
                "## Post-Refinement Warning",
                "",
                "- Post-refinement was marked enabled, so orientation/partiality refinement may have absorbed or reshaped the orientation-dependent effect.",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Positive I_times_weight Shifts",
            "",
            markdown_table(positive, table_columns),
            "",
            "## Top Negative I_times_weight Shifts",
            "",
            markdown_table(negative, table_columns),
            "",
            "## Local-Strength Proof-Of-Concept",
            "",
            "Weak, middle, and strong are crude tertiles of `Iweight_low_median` across signed HKLs passing the tail filters.",
            "This is used only as a proof-of-concept local-strength stratification, not as a final physical classification.",
            "",
            markdown_table(strength_summary, STRENGTH_SUMMARY_COLUMNS, max_rows=len(STRENGTH_CLASSES)),
            "",
            "### Weak Positive And Strong Negative I_times_weight Shift Spotlights",
            "",
            "Top weak positive rows are weak-class HKLs sorted by `relative_shift_Iweight` descending.",
            "Top strong negative rows are strong-class HKLs sorted by `relative_shift_Iweight` ascending.",
            "",
            "#### Top Weak Positive I_times_weight Shifts",
            "",
            markdown_table(weak_positive, STRENGTH_SPOTLIGHT_COLUMNS, max_rows=12),
            "",
            "#### Top Strong Negative I_times_weight Shifts",
            "",
            markdown_table(strong_negative, STRENGTH_SPOTLIGHT_COLUMNS, max_rows=12),
            "",
            "## Output Files",
            "",
            "- `joined_enh_feed_partialator_observations.csv`",
            "- `enh_feed_shift_by_signed_hkl.csv`",
            "- `enh_feed_shift_by_signed_hkl_with_strength_class.csv`",
            "- `top_positive_Iweight_shifts.csv`",
            "- `top_negative_Iweight_shifts.csv`",
            "- `top_weak_positive_Iweight_shifts.csv`",
            "- `top_strong_negative_Iweight_shifts.csv`",
            "- `summary.md`",
            "- `run_metadata.json`",
        ]
    )
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.outdir, args.overwrite)

    log("Parsing partialator unmerged observations")
    unmerged, unmerged_stats = parse_unmerged_observations(
        args.unmerged_hkl,
        include_flagged_crystals=bool(args.include_flagged_crystals),
        include_row_flags=bool(args.include_row_flags),
    )
    log(
        "Unmerged parse complete: "
        f"seen={unmerged_stats['unmerged_reflection_rows_seen']:,}, "
        f"kept={unmerged_stats['unmerged_reflection_rows_kept']:,}, "
        f"duplicate_keys={unmerged_stats['unmerged_duplicate_keys']:,}"
    )

    log("Loading enhancement-feed score CSV")
    scores, score_stats = load_scores(args.scores_csv)
    log(
        "Score load complete: "
        f"rows={score_stats['scores_rows_loaded']:,}, "
        f"duplicate_keys={score_stats['scores_duplicate_keys']:,}"
    )

    log("Joining unmerged observations to enhancement-feed scores")
    joined, join_stats = join_with_scores(unmerged, scores, bool(args.allow_basename_fallback))
    joined.to_csv(args.outdir / "joined_enh_feed_partialator_observations.csv", index=False)
    log(
        "Join complete: "
        f"mode={join_stats['match_mode']}, "
        f"exact_matches={join_stats['exact_matches']:,}, "
        f"basename_matches={join_stats['basename_matches']:,}"
    )

    log("Computing signed-HKL high-vs-low enhancement-feed shifts")
    shifts = analyze_hkl_shifts(
        joined,
        min_obs=int(args.min_obs),
        low_rank_max=float(args.low_rank_max),
        high_rank_min=float(args.high_rank_min),
        min_tail_obs=int(args.min_tail_obs),
    )
    if not shifts.empty:
        shifts = shifts.sort_values(["relative_shift_Iweight", "n_obs"], ascending=[False, False])
    shifts.to_csv(args.outdir / "enh_feed_shift_by_signed_hkl.csv", index=False)
    shifts_with_strength = add_local_strength_class(shifts)
    strength_summary = summarize_local_strength_classes(shifts_with_strength)
    weak_positive, strong_negative = strength_spotlight_tables(shifts_with_strength)
    shifts_with_strength.to_csv(args.outdir / "enh_feed_shift_by_signed_hkl_with_strength_class.csv", index=False)
    weak_positive.to_csv(args.outdir / "top_weak_positive_Iweight_shifts.csv", index=False)
    strong_negative.to_csv(args.outdir / "top_strong_negative_Iweight_shifts.csv", index=False)
    shifts.sort_values("relative_shift_Iweight", ascending=False).head(TOP_ROWS).to_csv(
        args.outdir / "top_positive_Iweight_shifts.csv",
        index=False,
    )
    shifts.sort_values("relative_shift_Iweight", ascending=True).head(TOP_ROWS).to_csv(
        args.outdir / "top_negative_Iweight_shifts.csv",
        index=False,
    )

    analysis_stats = {
        "joined_rows": int(len(joined)),
        "matched_rows": int(joined["score_matched"].sum()) if "score_matched" in joined else 0,
        "matched_finite_rows_for_analysis": int(len(finite_matched_observations(joined))),
        "signed_hkls_passing_tail_filters": int(len(shifts)),
    }
    write_summary(
        args.outdir,
        args,
        unmerged_stats,
        score_stats,
        join_stats,
        joined,
        shifts,
        strength_summary,
        weak_positive,
        strong_negative,
    )
    write_metadata(args.outdir, args, unmerged_stats, score_stats, join_stats, analysis_stats)

    print("Local-strength proof-of-concept summary:")
    print(markdown_table(strength_summary, STRENGTH_SUMMARY_COLUMNS, max_rows=len(STRENGTH_CLASSES)))
    print(f"Wrote: {args.outdir / 'joined_enh_feed_partialator_observations.csv'}")
    print(f"Wrote: {args.outdir / 'enh_feed_shift_by_signed_hkl.csv'}")
    print(f"Wrote: {args.outdir / 'enh_feed_shift_by_signed_hkl_with_strength_class.csv'}")
    print(f"Wrote: {args.outdir / 'top_positive_Iweight_shifts.csv'}")
    print(f"Wrote: {args.outdir / 'top_negative_Iweight_shifts.csv'}")
    print(f"Wrote: {args.outdir / 'top_weak_positive_Iweight_shifts.csv'}")
    print(f"Wrote: {args.outdir / 'top_strong_negative_Iweight_shifts.csv'}")
    print(f"Wrote: {args.outdir / 'summary.md'}")
    print(f"Wrote: {args.outdir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
