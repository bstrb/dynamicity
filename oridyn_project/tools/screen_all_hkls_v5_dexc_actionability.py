#!/usr/bin/env python3
"""Screen all signed HKLs for v5 excitation-deficit actionability.

The script consumes existing Partialator-accepted observations, existing v5
scores, and the existing merged intensity table. It does not recompute v5,
rewrite streams, run Partialator, merge, or refine structures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_and_crossvalidate_v5_dexc_correction_candidates as cvmod  # noqa: E402
from diagnose_v5_excitation_imbalance_orientation_broad_hkl_v2 import (  # noqa: E402
    DEFAULT_ACCEPTED,
    DEFAULT_CHUNKSIZE,
    DEFAULT_COUPLING_COLUMN,
    DEFAULT_SCORE_COLUMN,
    DEFAULT_SG_COLUMN,
    DEFAULT_STREAM,
    DEFAULT_STRENGTH_TABLE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_V5_SCORES,
    HKL_COLUMNS,
    KEY_COLUMNS,
    add_dexc_columns,
    add_intensity_responses,
    add_reciprocal_descriptors,
    choose_intensity_columns,
    four_mmm_orbit_id,
    four_mmm_orbit_variants,
    normalize_key_columns,
    parse_stream_unit_cell,
    partial_spearman,
    read_crystfel_hkl_strength,
    read_header,
    reciprocal_matrix_from_cell,
    require_columns,
    robust_slope,
    spearman_with_p,
)


BASE = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524")
DEFAULT_OUT_DIR = BASE / "oridyn_v5_dexc_all_hkl_actionability_20_0p3_20260713"

ORIGINAL_RESPONSE = "corrected_intensity_residual"
DEFICIT_NORM = "excitation_deficit_norm"
DEFICIT_RAW = "excitation_deficit_raw"
V5_SCORE = "nonself_local_excitation_raw"
TARGET_EXCITATION = "target_excitation_Eg"
PARTIALITY = "partiality"
COUPLING_SUM = "coupling_sum_raw"
DESCRIPTORS = [DEFICIT_NORM, DEFICIT_RAW, V5_SCORE]
MODEL_NAMES = [
    "linear_excitation_deficit_norm",
    "linear_excitation_deficit_raw",
    "linear_v5_score",
    "interaction_deficit_norm_x_v5",
    "high_deficit_threshold",
    "deficit_quintile",
    "piecewise_linear_deficit",
]

HH0_SEEDS = [(index, index, 0) for index in range(1, 12)]
SUSPECTED_GAIN_SEEDS = [(2, 2, 4), (6, 0, 1), (5, 0, 3)]
BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-scores", type=Path, default=DEFAULT_V5_SCORES)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--strength-table", type=Path, default=DEFAULT_STRENGTH_TABLE)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM, help="Used only for unit-cell resolution descriptors.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--coupling-column", default=DEFAULT_COUPLING_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--sg-column", default=DEFAULT_SG_COLUMN)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--candidate-file", type=Path, default=None, help="CSV of exact signed h,k,l rows to validate directly.")
    parser.add_argument("--screen-dir", type=Path, default=None, help="Completed screen directory used only for candidate metadata with --candidate-file.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Process workers for detailed per-HKL stages; defaults to all logical CPUs.")
    parser.add_argument("--screen-only", action="store_true", help="Stop after all-HKL screen and candidate selection outputs.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap on detailed-validation candidates after cheap-screen ranking.")
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--min-nonzero-coupling", type=int, default=100)
    parser.add_argument("--min-descriptor-spread", type=float, default=1.0e-4)
    parser.add_argument("--min-abs-slope", type=float, default=0.25)
    parser.add_argument("--min-abs-rho", type=float, default=0.08)
    parser.add_argument("--min-finite-response", type=int, default=30)
    parser.add_argument("--stable-sign-fraction", type=float, default=0.80)
    parser.add_argument("--min-correction-delta-mae", type=float, default=0.0)
    parser.add_argument("--min-filter-random-beat-fraction", type=float, default=0.60)
    parser.add_argument("--max-high-quality-degradation", type=float, default=0.02)
    parser.add_argument("--max-accepted-rows", type=int, default=None, help="Smoke-test limit; reads only this many accepted rows.")
    parser.add_argument("--max-v5-rows", type=int, default=None, help="Smoke-test limit; reads only this many v5 rows.")
    parser.add_argument("--plot-dpi", type=int, default=170)
    args = parser.parse_args()

    for label, path in [("--accepted", args.accepted), ("--v5-scores", args.v5_scores), ("--strength-table", args.strength_table), ("--stream", args.stream)]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if args.candidate_file is not None and not args.candidate_file.is_file():
        raise SystemExit(f"--candidate-file not found: {args.candidate_file}")
    if args.screen_dir is not None:
        if args.candidate_file is None:
            raise SystemExit("--screen-dir requires --candidate-file")
        if not args.screen_dir.is_dir():
            raise SystemExit(f"--screen-dir not found: {args.screen_dir}")
        for filename in ["all_hkl_screen.csv", "candidate_hkls.csv"]:
            if not (args.screen_dir / filename).is_file():
                raise SystemExit(f"--screen-dir missing {filename}: {args.screen_dir / filename}")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    if int(args.splits) < 1:
        raise SystemExit("--splits must be >= 1")
    if not (0.50 <= float(args.train_fraction) < 1.0):
        raise SystemExit("--train-fraction must satisfy 0.50 <= value < 1.0")
    if int(args.min_nonzero_coupling) < 1:
        raise SystemExit("--min-nonzero-coupling must be >= 1")
    if args.max_candidates is not None and int(args.max_candidates) < 1:
        raise SystemExit("--max-candidates must be >= 1 when provided")
    if args.max_accepted_rows is not None and int(args.max_accepted_rows) < 1:
        raise SystemExit("--max-accepted-rows must be >= 1 when provided")
    if args.max_v5_rows is not None and int(args.max_v5_rows) < 1:
        raise SystemExit("--max-v5-rows must be >= 1 when provided")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class StageProgress:
    def __init__(self, stage: str, total: int | None = None, unit: str = "items", min_interval: float = 30.0, per_item_until: int = 20):
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.min_interval = float(min_interval)
        self.per_item_until = int(per_item_until)
        self.start_time = time.monotonic()
        self.last_report_time = self.start_time
        self.completed = 0
        total_text = f"total={self.total:,} {self.unit}" if self.total is not None else f"total=unknown {self.unit}"
        log(f"{self.stage} start: {total_text}")

    def _message(self, completed: int, suffix: str = "progress") -> str:
        elapsed = max(time.monotonic() - self.start_time, 1.0e-9)
        rate = completed / elapsed if elapsed > 0.0 else np.nan
        parts = [f"{self.stage} {suffix}: completed={completed:,}"]
        if self.total is not None:
            pct = 100.0 * completed / max(1, self.total)
            parts[0] += f"/{self.total:,} ({pct:.1f}%)"
        parts.append(f"elapsed={format_duration(elapsed)}")
        parts.append(f"rate={rate:.2f} {self.unit}/s")
        if self.total is not None and completed > 0 and completed < self.total and rate > 0.0:
            eta = (self.total - completed) / rate
            parts.append(f"ETA={format_duration(eta)}")
        return "; ".join(parts)

    def update(self, completed: int, force: bool = False) -> None:
        completed = int(completed)
        now = time.monotonic()
        should_report = force or completed == self.total or completed <= self.per_item_until or (now - self.last_report_time) >= self.min_interval
        if should_report:
            log(self._message(completed))
            self.last_report_time = now
        self.completed = completed

    def advance(self, amount: int = 1, force: bool = False) -> None:
        self.update(self.completed + int(amount), force=force)

    def finish(self, completed: int | None = None) -> None:
        final_completed = self.completed if completed is None else int(completed)
        log(self._message(final_completed, suffix="complete"))


def set_worker_numeric_threads() -> None:
    for name in BLAS_THREAD_ENV_VARS:
        os.environ[name] = "1"


def worker_initializer() -> None:
    set_worker_numeric_threads()


def effective_worker_count(args: argparse.Namespace, item_count: int) -> int:
    if int(item_count) <= 0:
        return 0
    return max(1, min(int(args.workers), int(item_count)))


def hkl_label(hkl: tuple[int, int, int]) -> str:
    return f"({hkl[0]},{hkl[1]},{hkl[2]})"


def unit_cell_to_dict(cell: Any) -> dict[str, float]:
    return {name: float(getattr(cell, name)) for name in ["a", "b", "c", "alpha", "beta", "gamma"]}


def sign_value(value: Any, eps: float = 1.0e-12) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(numeric) or abs(numeric) <= eps:
        return 0
    return 1 if numeric > 0 else -1


def first_finite(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.iloc[0]) if len(finite) else np.nan


def q_values(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {f"{prefix}_q10": np.nan, f"{prefix}_q50": np.nan, f"{prefix}_q90": np.nan}
    q10, q50, q90 = np.quantile(values.to_numpy(dtype=float), [0.10, 0.50, 0.90])
    return {f"{prefix}_q10": float(q10), f"{prefix}_q50": float(q50), f"{prefix}_q90": float(q90)}


def normalize_hkl_table(table: pd.DataFrame, label: str) -> pd.DataFrame:
    require_columns(list(table.columns), HKL_COLUMNS, label)
    out = table.copy()
    for column in HKL_COLUMNS:
        numeric = pd.to_numeric(out[column], errors="coerce")
        if numeric.isna().any():
            raise SystemExit(f"{label} contains non-integer or missing {column} values")
        out[column] = numeric.astype(int)
    out["hkl"] = [hkl_label(tuple(map(int, row))) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
    return out


def load_candidate_table(path: Path) -> pd.DataFrame:
    table = normalize_hkl_table(pd.read_csv(path), "candidate file")
    if table.empty:
        raise SystemExit(f"--candidate-file contains no signed HKL rows: {path}")
    table["candidate_file_order"] = np.arange(len(table), dtype=int)
    duplicate_mask = table.duplicated(HKL_COLUMNS, keep="first")
    if duplicate_mask.any():
        log(f"Candidate file duplicate signed HKLs: dropping {int(duplicate_mask.sum()):,} duplicate row(s), keeping first occurrence")
        table = table.loc[~duplicate_mask].copy().reset_index(drop=True)
        table["candidate_file_order"] = np.arange(len(table), dtype=int)
    computed_orbits = pd.Series([four_mmm_orbit_id(int(row.h), int(row.k), int(row.l)) for row in table.loc[:, HKL_COLUMNS].itertuples(index=False)], index=table.index)
    if "four_mmm_orbit_id" in table.columns:
        table["four_mmm_orbit_id"] = table["four_mmm_orbit_id"].where(table["four_mmm_orbit_id"].notna(), computed_orbits)
    else:
        table["four_mmm_orbit_id"] = computed_orbits
    table["candidate_file_selected"] = True
    return table


def candidate_hkl_set(candidate_table: pd.DataFrame | None) -> set[tuple[int, int, int]] | None:
    if candidate_table is None:
        return None
    return {tuple(map(int, row)) for row in candidate_table.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)}


def filter_to_candidate_hkls(table: pd.DataFrame, wanted: set[tuple[int, int, int]] | None) -> pd.DataFrame:
    if wanted is None:
        return table
    numeric = table.loc[:, HKL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    row_hkls = list(zip(numeric["h"].to_numpy(), numeric["k"].to_numpy(), numeric["l"].to_numpy(), strict=False))
    mask = np.asarray([tuple(map(int, hkl)) in wanted if all(np.isfinite(hkl)) else False for hkl in row_hkls], dtype=bool)
    return table.loc[mask].copy()


def merge_hkl_metadata(base: pd.DataFrame, metadata: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if metadata.empty:
        return base
    meta = normalize_hkl_table(metadata, source_label).drop_duplicates(HKL_COLUMNS, keep="first")
    payload = [column for column in meta.columns if column not in [*HKL_COLUMNS, "hkl"]]
    rename = {column: f"{column}_{source_label}" for column in payload if column in base.columns}
    merged = base.merge(meta.loc[:, [*HKL_COLUMNS, *payload]].rename(columns=rename), on=HKL_COLUMNS, how="left", sort=False, validate="one_to_one")
    for column, renamed in rename.items():
        if column in {"four_mmm_orbit_id", "pilot_role"}:
            merged[column] = merged[column].where(merged[column].notna(), merged[renamed])
    return merged


def add_screen_metrics(screen: pd.DataFrame, args: argparse.Namespace, recompute: bool = True) -> pd.DataFrame:
    out = normalize_hkl_table(screen, "screen table")
    for column in ["accepted_count", "nonzero_coupling_count", "zero_coupling_count", "finite_response_count"]:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    computed_orbits = pd.Series([four_mmm_orbit_id(int(row.h), int(row.k), int(row.l)) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False)], index=out.index)
    if "four_mmm_orbit_id" in out.columns:
        out["four_mmm_orbit_id"] = out["four_mmm_orbit_id"].where(out["four_mmm_orbit_id"].notna(), computed_orbits)
    else:
        out["four_mmm_orbit_id"] = computed_orbits
    for descriptor in DESCRIPTORS:
        for column in [f"slope_{descriptor}", f"spearman_{descriptor}", f"{descriptor}_spread_q90_q10"]:
            if column not in out.columns:
                out[column] = np.nan
        shape_column = f"{descriptor}_quintile_shape"
        if shape_column not in out.columns:
            out[shape_column] = "insufficient"
    if not recompute:
        if "screen_max_abs_slope" not in out.columns:
            out["screen_max_abs_slope"] = np.nan
        if "screen_max_abs_spearman" not in out.columns:
            out["screen_max_abs_spearman"] = np.nan
        if "screen_nonflat_quintile_trend" not in out.columns:
            out["screen_nonflat_quintile_trend"] = False
        if "candidate_score" not in out.columns:
            out["candidate_score"] = np.nan
        if "cheap_screen_pass" not in out.columns:
            out["cheap_screen_pass"] = False
        if "cheap_screen_rejection_reason" not in out.columns:
            out["cheap_screen_rejection_reason"] = "candidate_file_mode_not_screened"
        return out
    max_abs_slope = out[[f"slope_{descriptor}" for descriptor in DESCRIPTORS]].abs().max(axis=1, skipna=True)
    max_abs_rho = out[[f"spearman_{descriptor}" for descriptor in DESCRIPTORS]].abs().max(axis=1, skipna=True)
    nonflat_shape = pd.Series(False, index=out.index)
    for descriptor in DESCRIPTORS:
        nonflat_shape |= ~out[f"{descriptor}_quintile_shape"].fillna("insufficient").isin(["flat", "insufficient"])
    spread_ok = pd.Series(False, index=out.index)
    for descriptor in DESCRIPTORS:
        spread_ok |= pd.to_numeric(out[f"{descriptor}_spread_q90_q10"], errors="coerce").fillna(0.0) >= float(args.min_descriptor_spread)
    out["screen_max_abs_slope"] = max_abs_slope
    out["screen_max_abs_spearman"] = max_abs_rho
    out["screen_nonflat_quintile_trend"] = nonflat_shape
    out["candidate_score"] = max_abs_rho.fillna(0.0) + 0.05 * max_abs_slope.fillna(0.0) + 0.25 * nonflat_shape.astype(float)
    out["cheap_screen_pass"] = (
        (out["nonzero_coupling_count"] >= int(args.min_nonzero_coupling))
        & (out["finite_response_count"] >= int(args.min_finite_response))
        & spread_ok
        & ((max_abs_slope >= float(args.min_abs_slope)) | (max_abs_rho >= float(args.min_abs_rho)) | nonflat_shape)
    )
    reasons = []
    for _, row in out.iterrows():
        current = []
        if int(row["nonzero_coupling_count"]) < int(args.min_nonzero_coupling):
            current.append("low_nonzero_coupling")
        if int(row["finite_response_count"]) < int(args.min_finite_response):
            current.append("low_finite_response")
        if not bool(spread_ok.loc[row.name]):
            current.append("low_descriptor_spread")
        if not (float(row.get("screen_max_abs_slope", 0.0) or 0.0) >= float(args.min_abs_slope) or float(row.get("screen_max_abs_spearman", 0.0) or 0.0) >= float(args.min_abs_rho) or bool(row.get("screen_nonflat_quintile_trend", False))):
            current.append("no_meaningful_effect_signal")
        reasons.append(";".join(current))
    out["cheap_screen_rejection_reason"] = reasons
    return out


def known_family_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in HH0_SEEDS:
        for hkl in sorted(four_mmm_orbit_variants(*seed)):
            rows.append({"h": hkl[0], "k": hkl[1], "l": hkl[2], "known_family": "hh0", "known_family_seed": hkl_label(seed)})
    for seed in SUSPECTED_GAIN_SEEDS:
        for hkl in sorted(four_mmm_orbit_variants(*seed)):
            rows.append({"h": hkl[0], "k": hkl[1], "l": hkl[2], "known_family": "suspected_gain", "known_family_seed": hkl_label(seed)})
    out = pd.DataFrame.from_records(rows).drop_duplicates(HKL_COLUMNS + ["known_family_seed"]).reset_index(drop=True)
    out["hkl"] = [hkl_label(tuple(map(int, row))) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
    return out


def load_accepted_table_limited(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    header = read_header(path)
    require_columns(header, KEY_COLUMNS, "accepted survivor table")
    choice = choose_intensity_columns(header)
    optional = ["partiality", "I_unmerged", choice.column, *choice.scale_columns]
    if choice.sigma_column is not None:
        optional.append(choice.sigma_column)
    optional = list(dict.fromkeys(column for column in optional if column in header))
    usecols = [*KEY_COLUMNS, *optional]
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_after_hkl_filter = 0
    wanted_hkls = getattr(args, "candidate_hkl_set", None)
    progress = StageProgress("Accepted survivor scan", total=args.max_accepted_rows, unit="rows")
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=int(args.chunksize)), start=1):
        if args.max_accepted_rows is not None:
            remaining = int(args.max_accepted_rows) - rows_read
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        work = filter_to_candidate_hkls(work, wanted_hkls)
        rows_after_hkl_filter += int(len(work))
        if work.empty:
            progress.update(rows_read, force=chunk_index == 1)
            if args.max_accepted_rows is not None and rows_read >= int(args.max_accepted_rows):
                break
            continue
        work["observation_intensity"] = pd.to_numeric(work[choice.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work["I_unmerged"] = pd.to_numeric(work["I_unmerged"], errors="coerce").replace([np.inf, -np.inf], np.nan) if "I_unmerged" in work.columns else np.nan
        work["sigma"] = pd.to_numeric(work[choice.sigma_column], errors="coerce").replace([np.inf, -np.inf], np.nan) if choice.sigma_column is not None else np.nan
        work["partiality"] = pd.to_numeric(work["partiality"], errors="coerce").replace([np.inf, -np.inf], np.nan) if "partiality" in work.columns else np.nan
        for scale_column in choice.scale_columns:
            work[scale_column] = pd.to_numeric(work[scale_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunks.append(work.loc[:, [*KEY_COLUMNS, "observation_intensity", "I_unmerged", "sigma", "partiality", *choice.scale_columns]].copy())
        progress.update(rows_read, force=chunk_index == 1)
        if args.max_accepted_rows is not None and rows_read >= int(args.max_accepted_rows):
            break
    progress.finish(rows_read)
    if not chunks:
        raise SystemExit("Accepted survivor table yielded no usable rows")
    table = pd.concat(chunks, ignore_index=True)
    duplicate_mask = table.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = int(table.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    metadata = {
        "rows_read": int(rows_read),
        "candidate_hkl_filter_applied": wanted_hkls is not None,
        "rows_after_candidate_hkl_filter_before_dedup": int(rows_after_hkl_filter),
        "rows_after_key_cleanup": int(len(table)),
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
        "intensity_column_used": choice.column,
        "sigma_column_used": choice.sigma_column,
        "scale_columns_available": list(choice.scale_columns),
        "intensity_choice_note": choice.note,
    }
    return table, choice, metadata


def load_accepted_v5_join_limited(accepted: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = read_header(args.v5_scores)
    required = [*KEY_COLUMNS, args.score_column, args.coupling_column, args.target_column, args.sg_column]
    require_columns(header, required, "v5 score CSV")
    accepted_payload = accepted.copy()
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_after_hkl_filter = 0
    rows_after_cleanup = 0
    rows_matched = 0
    wanted_hkls = getattr(args, "candidate_hkl_set", None)
    progress = StageProgress("V5/accepted exact-key join scan", total=args.max_v5_rows, unit="rows")
    for chunk_index, chunk in enumerate(pd.read_csv(args.v5_scores, usecols=required, chunksize=int(args.chunksize)), start=1):
        if args.max_v5_rows is not None:
            remaining = int(args.max_v5_rows) - rows_read
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        work = filter_to_candidate_hkls(work, wanted_hkls)
        rows_after_hkl_filter += int(len(work))
        if work.empty:
            progress.update(rows_read, force=chunk_index == 1)
            if args.max_v5_rows is not None and rows_read >= int(args.max_v5_rows):
                break
            continue
        for column in [args.score_column, args.coupling_column, args.target_column, args.sg_column]:
            work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=[args.score_column, args.coupling_column, args.target_column])
        rows_after_cleanup += int(len(work))
        matched = work.merge(accepted_payload, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        rows_matched += int(len(matched))
        if not matched.empty:
            chunks.append(matched)
        progress.update(rows_read, force=chunk_index == 1)
        if args.max_v5_rows is not None and rows_read >= int(args.max_v5_rows):
            break
    progress.finish(rows_read)
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not table.empty:
        duplicate_mask = table.duplicated(KEY_COLUMNS, keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_keys = int(table.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
        if duplicate_rows:
            table = table.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    else:
        duplicate_rows = 0
        duplicate_keys = 0
    metadata = {
        "v5_rows_read": int(rows_read),
        "candidate_hkl_filter_applied": wanted_hkls is not None,
        "v5_rows_after_candidate_hkl_filter": int(rows_after_hkl_filter),
        "v5_rows_after_cleanup": int(rows_after_cleanup),
        "accepted_v5_matched_rows": int(len(table)),
        "duplicate_join_rows": duplicate_rows,
        "duplicate_join_keys": duplicate_keys,
        "accepted_keys_without_v5_score": int(max(0, len(accepted) - table.loc[:, KEY_COLUMNS].drop_duplicates().shape[0])) if not table.empty else int(len(accepted)),
    }
    return table, metadata


def add_excitation_deficit_raw(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    coupling = pd.to_numeric(out[COUPLING_SUM], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    deficit = pd.to_numeric(out[DEFICIT_NORM], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    out[DEFICIT_RAW] = np.where(np.isfinite(coupling) & (coupling != 0.0), coupling * deficit, 0.0)
    return out


def prepare_observations(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    accepted, intensity_choice, accepted_meta = load_accepted_table_limited(args.accepted, args)
    joined, join_meta = load_accepted_v5_join_limited(accepted, args)
    if joined.empty:
        raise SystemExit("No accepted observations matched v5 scores under the current limits")
    v5_args = SimpleNamespace(score_column=args.score_column, coupling_column=args.coupling_column, target_column=args.target_column, sg_column=args.sg_column)
    joined = add_dexc_columns(joined, v5_args)
    joined = add_excitation_deficit_raw(joined)

    strength, strength_meta = read_crystfel_hkl_strength(args.strength_table)
    cell = parse_stream_unit_cell(args.stream)
    reciprocal = reciprocal_matrix_from_cell(cell)
    strength = add_reciprocal_descriptors(strength, reciprocal)
    strength["resolution_angstrom"] = strength["d_spacing_angstrom"]
    payload = [column for column in [*HKL_COLUMNS, "merged_intensity_or_Fobs", "merged_sigma", "merged_nmeas", "d_spacing_angstrom", "resolution_angstrom", "g_norm_invA", "reciprocal_direction_class", "angle_to_cstar_deg", "symmetry_orbit_id"] if column in strength.columns]
    observations = joined.merge(strength.loc[:, payload], on=HKL_COLUMNS, how="left", validate="many_to_one")
    observations = add_intensity_responses(observations, strength, tuple(intensity_choice.scale_columns))
    observations["four_mmm_orbit_id"] = [four_mmm_orbit_id(int(row.h), int(row.k), int(row.l)) for row in observations.loc[:, HKL_COLUMNS].itertuples(index=False)]
    metadata = {"accepted_table": accepted_meta, "v5_join": join_meta, "strength": strength_meta, "unit_cell": unit_cell_to_dict(cell)}
    return observations, list(intensity_choice.scale_columns), metadata


def classify_quintile_shape(medians: list[float]) -> str:
    values = np.asarray([value for value in medians if np.isfinite(value)], dtype=float)
    if len(values) < 3:
        return "insufficient"
    value_range = float(np.max(values) - np.min(values))
    scale = max(float(np.nanmedian(np.abs(values))), 1.0)
    tol = 0.02 * scale
    if value_range <= tol:
        return "flat"
    diffs = np.diff(values)
    abs_steps = np.abs(diffs)
    step_total = float(np.sum(abs_steps))
    if step_total > 0.0 and float(np.max(abs_steps) / step_total) >= 0.65:
        return "threshold-like"
    middle = values[1:-1]
    if len(middle) and ((np.nanmedian(middle) + tol < min(values[0], values[-1])) or (np.nanmedian(middle) - tol > max(values[0], values[-1]))):
        return "U-shaped"
    if np.all(diffs >= -tol) or np.all(diffs <= tol):
        return "monotonic"
    return "nonmonotonic"


def descriptor_quintiles(group: pd.DataFrame, descriptor: str) -> dict[str, Any]:
    nonzero = group.loc[pd.to_numeric(group[COUPLING_SUM], errors="coerce") > 0.0].copy()
    prefix = descriptor
    if len(nonzero) < 5 or nonzero[descriptor].nunique(dropna=True) < 2:
        return {f"{prefix}_quintile_medians": "", f"{prefix}_quintile_shape": "insufficient"}
    labels = pd.qcut(nonzero[descriptor].rank(method="first"), q=min(5, len(nonzero)), labels=False, duplicates="drop")
    medians = nonzero.assign(_q=np.asarray(labels, dtype=int)).groupby("_q", sort=True)[ORIGINAL_RESPONSE].median().to_numpy(dtype=float)
    return {f"{prefix}_quintile_medians": ";".join(f"{float(value):.6g}" for value in medians), f"{prefix}_quintile_shape": classify_quintile_shape(list(medians))}


def summarize_hkl(group: pd.DataFrame) -> dict[str, Any]:
    hkl = tuple(map(int, group.loc[:, HKL_COLUMNS].iloc[0].tolist()))
    nonzero = group.loc[pd.to_numeric(group[COUPLING_SUM], errors="coerce") > 0.0].copy()
    response = pd.to_numeric(group[ORIGINAL_RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan)
    row: dict[str, Any] = {
        "h": hkl[0],
        "k": hkl[1],
        "l": hkl[2],
        "hkl": hkl_label(hkl),
        "four_mmm_orbit_id": four_mmm_orbit_id(*hkl),
        "accepted_count": int(len(group)),
        "nonzero_coupling_count": int(len(nonzero)),
        "zero_coupling_count": int(len(group) - len(nonzero)),
        "finite_response_count": int(response.notna().sum()),
        "merged_intensity_or_Fobs": first_finite(group["merged_intensity_or_Fobs"]),
        "resolution_angstrom": first_finite(group["resolution_angstrom"]),
        "median_target_excitation": float(pd.to_numeric(group[TARGET_EXCITATION], errors="coerce").median()),
        "median_partiality": float(pd.to_numeric(group[PARTIALITY], errors="coerce").median()),
        "median_v5_score": float(pd.to_numeric(group[V5_SCORE], errors="coerce").median()),
        "q90_v5_score": float(pd.to_numeric(group[V5_SCORE], errors="coerce").quantile(0.90)),
    }
    row.update(q_values(nonzero[DEFICIT_NORM], "excitation_deficit"))
    row["excitation_deficit_spread_q90_q10"] = row["excitation_deficit_q90"] - row["excitation_deficit_q10"] if np.isfinite(row["excitation_deficit_q90"]) and np.isfinite(row["excitation_deficit_q10"]) else np.nan
    controls = nonzero.loc[:, [TARGET_EXCITATION, PARTIALITY]] if len(nonzero) else pd.DataFrame(columns=[TARGET_EXCITATION, PARTIALITY])
    for descriptor in DESCRIPTORS:
        row[f"{descriptor}_spread_q90_q10"] = np.nan
        if len(nonzero):
            qs = q_values(nonzero[descriptor], descriptor)
            row.update(qs)
            row[f"{descriptor}_spread_q90_q10"] = qs[f"{descriptor}_q90"] - qs[f"{descriptor}_q10"] if np.isfinite(qs[f"{descriptor}_q90"]) and np.isfinite(qs[f"{descriptor}_q10"]) else np.nan
        descriptor_values = pd.to_numeric(nonzero[descriptor], errors="coerce").replace([np.inf, -np.inf], np.nan) if len(nonzero) else pd.Series(dtype=float)
        response_values = pd.to_numeric(nonzero[ORIGINAL_RESPONSE], errors="coerce").replace([np.inf, -np.inf], np.nan) if len(nonzero) else pd.Series(dtype=float)
        finite_pair_count = int((descriptor_values.notna() & response_values.notna()).sum())
        row[f"slope_{descriptor}"] = robust_slope(descriptor_values, response_values) if finite_pair_count >= 3 else np.nan
        row[f"spearman_{descriptor}"] = spearman_with_p(response_values, descriptor_values)[0] if finite_pair_count >= 3 else np.nan
        row[f"partial_spearman_{descriptor}_ctrl_Eg_partiality"] = partial_spearman(response_values, descriptor_values, controls) if finite_pair_count >= 5 else np.nan
        row.update(descriptor_quintiles(group, descriptor))
    return row


def build_all_hkl_screen(observations: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    grouped = observations.groupby(HKL_COLUMNS, sort=False)
    progress = StageProgress("All-HKL cheap-screen summarization", total=grouped.ngroups, unit="HKLs")
    rows = []
    for _, group in grouped:
        rows.append(summarize_hkl(group))
        progress.advance()
    progress.finish(len(rows))
    screen = pd.DataFrame.from_records(rows)
    known = known_family_table()
    screen = screen.merge(known.loc[:, [*HKL_COLUMNS, "known_family", "known_family_seed"]].drop_duplicates(HKL_COLUMNS), on=HKL_COLUMNS, how="outer", validate="one_to_one")
    screen = add_screen_metrics(screen, args)
    return screen.sort_values(["cheap_screen_pass", "candidate_score", "nonzero_coupling_count"], ascending=[False, False, False]).reset_index(drop=True)


def build_candidate_observation_screen(observations: pd.DataFrame, candidate_table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    grouped = observations.groupby(HKL_COLUMNS, sort=False)
    progress = StageProgress("Candidate-HKL observation summarization", total=grouped.ngroups, unit="HKLs")
    rows = []
    for _, group in grouped:
        rows.append(summarize_hkl(group))
        progress.advance()
    progress.finish(len(rows))
    observed_screen = pd.DataFrame.from_records(rows) if rows else pd.DataFrame(columns=HKL_COLUMNS)
    screen = merge_hkl_metadata(candidate_table.copy(), observed_screen, "observed")
    return add_screen_metrics(screen, args, recompute=False).sort_values("candidate_file_order", ascending=True).reset_index(drop=True)


def load_screen_dir_candidate_metadata(screen_dir: Path, candidate_table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    progress = StageProgress("Completed-screen metadata load", total=2, unit="files", per_item_until=2)
    screen = candidate_table.copy()
    screen = merge_hkl_metadata(screen, pd.read_csv(screen_dir / "all_hkl_screen.csv"), "all_hkl_screen")
    progress.advance(force=True)
    screen = merge_hkl_metadata(screen, pd.read_csv(screen_dir / "candidate_hkls.csv"), "candidate_hkls")
    progress.advance(force=True)
    progress.finish(2)
    return add_screen_metrics(screen, args).sort_values("candidate_file_order", ascending=True).reset_index(drop=True)


def choose_candidates(screen: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = screen.copy()
    eligible = out.loc[out["cheap_screen_pass"].astype(bool)].sort_values(["candidate_score", "nonzero_coupling_count"], ascending=[False, False])
    selected_index = eligible.index.tolist()
    if args.max_candidates is not None:
        selected_index = selected_index[: int(args.max_candidates)]
    out["detailed_validation_selected"] = False
    out.loc[selected_index, "detailed_validation_selected"] = True
    out["candidate_output_reason"] = np.where(out["detailed_validation_selected"], "selected_for_detailed_validation", np.where(out["cheap_screen_pass"], "passes_screen_not_selected_due_to_max_candidates", np.where(out["known_family"].notna(), "known_family_below_threshold", "not_candidate")))
    keep = out["detailed_validation_selected"] | out["cheap_screen_pass"] | out["known_family"].notna()
    return out.loc[keep].copy().reset_index(drop=True)


def choose_candidate_file_candidates(screen: pd.DataFrame) -> pd.DataFrame:
    out = screen.copy().sort_values("candidate_file_order", ascending=True).reset_index(drop=True)
    out["detailed_validation_selected"] = True
    out["candidate_output_reason"] = "candidate_file"
    return out


def clean_actionability_frame(group: pd.DataFrame, response: str, covariates: list[str]) -> pd.DataFrame:
    required = [response, *covariates, *DESCRIPTORS, COUPLING_SUM, TARGET_EXCITATION, PARTIALITY]
    if "I_robust_z_within_hkl" in group.columns:
        required.append("I_robust_z_within_hkl")
    keep = [column for column in dict.fromkeys([*HKL_COLUMNS, "source_filename", "event", "closest_uvw", "continuous_uvw_x", "continuous_uvw_y", "continuous_uvw_z", *required]) if column in group.columns]
    out = group.loc[:, keep].copy()
    numeric_columns = [response, *covariates, *DESCRIPTORS, COUPLING_SUM, TARGET_EXCITATION, PARTIALITY, "I_robust_z_within_hkl"]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.loc[out[COUPLING_SUM] > 0.0].dropna(subset=[response, *covariates, *DESCRIPTORS]).reset_index(drop=True)


def fit_effect_model(train: pd.DataFrame, train_residual: np.ndarray, test: pd.DataFrame, model_name: str) -> dict[str, Any]:
    residual = np.asarray(train_residual, dtype=float)
    params: dict[str, Any] = {"model_name": model_name}
    if model_name == "linear_excitation_deficit_norm":
        feature_train = pd.to_numeric(train[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float)
        feature_test = pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float)
    elif model_name == "linear_excitation_deficit_raw":
        feature_train = pd.to_numeric(train[DEFICIT_RAW], errors="coerce").to_numpy(dtype=float)
        feature_test = pd.to_numeric(test[DEFICIT_RAW], errors="coerce").to_numpy(dtype=float)
    elif model_name == "linear_v5_score":
        feature_train = pd.to_numeric(train[V5_SCORE], errors="coerce").to_numpy(dtype=float)
        feature_test = pd.to_numeric(test[V5_SCORE], errors="coerce").to_numpy(dtype=float)
    elif model_name == "interaction_deficit_norm_x_v5":
        feature_train = pd.to_numeric(train[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float) * pd.to_numeric(train[V5_SCORE], errors="coerce").to_numpy(dtype=float)
        feature_test = pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float) * pd.to_numeric(test[V5_SCORE], errors="coerce").to_numpy(dtype=float)
    elif model_name == "high_deficit_threshold":
        deficit_train = pd.to_numeric(train[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float)
        deficit_test = pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float)
        breakpoint = float(np.nanquantile(deficit_train, 0.90))
        feature_train = np.maximum(0.0, deficit_train - breakpoint)
        feature_test = np.maximum(0.0, deficit_test - breakpoint)
        params["breakpoint"] = breakpoint
    elif model_name == "piecewise_linear_deficit":
        deficit_train = pd.to_numeric(train[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float)
        deficit_test = pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float)
        best: dict[str, Any] | None = None
        for quantile in [0.25, 0.50, 0.75]:
            breakpoint = float(np.nanquantile(deficit_train, quantile))
            center = float(np.nanmedian(deficit_train))
            design = np.column_stack([deficit_train - center, np.maximum(0.0, deficit_train - breakpoint)])
            beta = cvmod.huber_irls(np.column_stack([np.ones(len(design)), design]), residual)
            pred = design @ beta[1:]
            mae = float(np.nanmean(np.abs(residual - pred)))
            if best is None or mae < best["mae"]:
                best = {"mae": mae, "breakpoint": breakpoint, "center": center, "beta": beta[1:]}
        if best is None:
            return {"predicted_effect_test": np.full(len(test), np.nan), "slope_sign": 0, **params}
        feature_test_matrix = np.column_stack([deficit_test - best["center"], np.maximum(0.0, deficit_test - best["breakpoint"])])
        return {"predicted_effect_test": feature_test_matrix @ best["beta"], "slope_sign": sign_value(best["beta"][0]), "beta": float(best["beta"][0]), "breakpoint": best["breakpoint"], **params}
    elif model_name == "deficit_quintile":
        offsets = cvmod.train_quintile_offsets_from_residuals(train, residual)
        baseline_zero = np.zeros(len(test), dtype=float)
        corrected = cvmod.apply_quintile_offsets_to_residuals(test, baseline_zero, offsets)
        predicted = -corrected
        finite_offsets = [float(value) for value in offsets.get("offsets", {}).values() if np.isfinite(value)]
        return {"predicted_effect_test": predicted, "slope_sign": sign_value(finite_offsets[-1] - finite_offsets[0]) if len(finite_offsets) >= 2 else 0, "n_bins": int(offsets.get("n_bins", 0)), **params}
    else:
        raise ValueError(f"Unknown model: {model_name}")

    center = float(np.nanmedian(feature_train))
    beta = cvmod.robust_slope_simple(feature_train - center, residual)
    return {"predicted_effect_test": beta * (feature_test - center), "slope_sign": sign_value(beta), "beta": beta, "center": center, **params}


def shuffled_train_for_model(train: pd.DataFrame, model_name: str, rng: np.random.Generator) -> pd.DataFrame:
    shuffled = train.copy()
    columns = [DEFICIT_NORM]
    if "raw" in model_name:
        columns = [DEFICIT_RAW]
    elif "v5" in model_name and "interaction" not in model_name:
        columns = [V5_SCORE]
    elif "interaction" in model_name:
        columns = [DEFICIT_NORM, V5_SCORE]
    for column in columns:
        values = shuffled[column].to_numpy(dtype=float).copy()
        rng.shuffle(values)
        shuffled[column] = values
    return shuffled


def subset_delta(test: pd.DataFrame, before: np.ndarray, after: np.ndarray, column: str, train_threshold: float) -> tuple[int, float]:
    values = pd.to_numeric(test[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & (values >= train_threshold)
    if int(mask.sum()) < 3:
        return int(mask.sum()), np.nan
    return int(mask.sum()), float(np.nanmean(np.abs(before[mask])) - np.nanmean(np.abs(after[mask])))


def worker_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(splits=int(args.splits), train_fraction=float(args.train_fraction), seed=int(args.seed))


def observation_group_for_hkl(observations: pd.DataFrame, hkl: tuple[int, int, int]) -> pd.DataFrame:
    return observations.loc[(observations["h"].astype(int) == hkl[0]) & (observations["k"].astype(int) == hkl[1]) & (observations["l"].astype(int) == hkl[2])].copy()


def crossvalidate_one_candidate(group: pd.DataFrame, candidate: dict[str, Any], scale_columns: list[str], args: SimpleNamespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    covariates = [TARGET_EXCITATION, PARTIALITY, *scale_columns]
    hkl = (int(candidate["h"]), int(candidate["k"]), int(candidate["l"]))
    frame = clean_actionability_frame(group, cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates)
    if len(frame) < 10:
        return [{"h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "status": "insufficient_data", "n_usable": int(len(frame))}]
    for split in range(int(args.splits)):
        train_idx, test_idx = cvmod.deterministic_stratified_split(frame, split, float(args.train_fraction), int(args.seed))
        if len(train_idx) < 5 or len(test_idx) < 3:
            rows.append({"h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl), "split": split, "status": "split_too_small", "n_usable": int(len(frame)), "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
            continue
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        train, test, denominator, norm_note = cvmod.add_train_normalized_response(train, test)
        baseline = cvmod.fit_model(train, cvmod.CV_RESPONSE_COLUMN, covariates)
        train_pred = cvmod.predict_model(baseline, train)
        test_pred = cvmod.predict_model(baseline, test)
        y_train = pd.to_numeric(train[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
        y_test = pd.to_numeric(test[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
        residual_train = y_train - train_pred
        residual_test = y_test - test_pred
        train_high_target = float(pd.to_numeric(train[TARGET_EXCITATION], errors="coerce").quantile(0.75))
        train_high_partiality = float(pd.to_numeric(train[PARTIALITY], errors="coerce").quantile(0.75))
        rng = np.random.default_rng(int(args.seed) + 7919 * (split + 1) + 97 * sum(abs(v) for v in hkl))
        for model_name in MODEL_NAMES:
            fit = fit_effect_model(train, residual_train, test, model_name)
            effect = np.asarray(fit["predicted_effect_test"], dtype=float)
            corrected_residual = residual_test - effect
            shuffled_train = shuffled_train_for_model(train, model_name, rng)
            sham_fit = fit_effect_model(shuffled_train, residual_train, test, model_name)
            sham_effect = np.asarray(sham_fit["predicted_effect_test"], dtype=float)
            sham_residual = residual_test - sham_effect
            before = cvmod.residual_metrics(y_test, residual_test, pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float))
            after = cvmod.residual_metrics(y_test, corrected_residual, pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float))
            sham = cvmod.residual_metrics(y_test, sham_residual, pd.to_numeric(test[DEFICIT_NORM], errors="coerce").to_numpy(dtype=float))
            high_target_n, high_target_delta = subset_delta(test, residual_test, corrected_residual, TARGET_EXCITATION, train_high_target)
            high_partiality_n, high_partiality_delta = subset_delta(test, residual_test, corrected_residual, PARTIALITY, train_high_partiality)
            response_corrected = y_test - effect
            rows.append(
                {
                    "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl),
                    "four_mmm_orbit_id": candidate.get("four_mmm_orbit_id", four_mmm_orbit_id(*hkl)),
                    "split": split, "status": "ok", "model_name": model_name,
                    "n_train": int(len(train)), "n_test": int(len(test)),
                    "response_normalization_denominator_train": denominator,
                    "response_normalization_source": norm_note,
                    "slope_sign": int(fit.get("slope_sign", 0)),
                    "beta_or_primary_slope": fit.get("beta", np.nan),
                    "heldout_mae_before": before["mae"],
                    "heldout_mae_after": after["mae"],
                    "heldout_delta_mae": before["mae"] - after["mae"],
                    "heldout_relative_mae_improvement": (before["mae"] - after["mae"]) / before["mae"] if np.isfinite(before["mae"]) and before["mae"] > 0 else np.nan,
                    "heldout_mad_before": before["median_abs_residual"],
                    "heldout_mad_after": after["median_abs_residual"],
                    "heldout_delta_mad": before["median_abs_residual"] - after["median_abs_residual"],
                    "heldout_relative_mad_improvement": (before["median_abs_residual"] - after["median_abs_residual"]) / before["median_abs_residual"] if np.isfinite(before["median_abs_residual"]) and before["median_abs_residual"] > 0 else np.nan,
                    "residual_deficit_rho_before": before["residual_spearman_vs_deficit"],
                    "residual_deficit_rho_after": after["residual_spearman_vs_deficit"],
                    "abs_rho_reduction": abs(before["residual_spearman_vs_deficit"]) - abs(after["residual_spearman_vs_deficit"]),
                    "sham_delta_mae": before["mae"] - sham["mae"],
                    "beats_sham_mae": bool((before["mae"] - after["mae"]) > (before["mae"] - sham["mae"])),
                    "high_target_n_test": high_target_n,
                    "high_target_delta_mae": high_target_delta,
                    "high_partiality_n_test": high_partiality_n,
                    "high_partiality_delta_mae": high_partiality_delta,
                    "mean_intensity_shift_after_minus_before": float(np.nanmean(response_corrected) - np.nanmean(y_test)),
                    "median_intensity_shift_after_minus_before": float(np.nanmedian(response_corrected) - np.nanmedian(y_test)),
                }
            )
    return rows


def crossvalidation_worker(task: tuple[int, dict[str, Any], pd.DataFrame, list[str], SimpleNamespace]) -> tuple[int, list[dict[str, Any]]]:
    index, candidate, group, scale_columns, args = task
    return index, crossvalidate_one_candidate(group, candidate, scale_columns, args)


def run_candidate_crossvalidation(observations: pd.DataFrame, candidates: pd.DataFrame, scale_columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    selected = candidates.loc[candidates["detailed_validation_selected"].astype(bool)].copy().reset_index(drop=True)
    worker_count = effective_worker_count(args, len(selected))
    progress = StageProgress("Candidate crossvalidation", total=len(selected), unit="HKLs")
    if selected.empty:
        progress.finish(0)
        return pd.DataFrame()
    tasks = []
    worker_namespace = worker_args(args)
    for index, candidate in selected.iterrows():
        hkl = (int(candidate.h), int(candidate.k), int(candidate.l))
        tasks.append((index, candidate.to_dict(), observation_group_for_hkl(observations, hkl), list(scale_columns), worker_namespace))
    results: dict[int, list[dict[str, Any]]] = {}
    use_processes = int(args.workers) > 1
    if use_processes:
        set_worker_numeric_threads()
        with ProcessPoolExecutor(max_workers=worker_count, initializer=worker_initializer) as executor:
            futures = [executor.submit(crossvalidation_worker, task) for task in tasks]
            for future in as_completed(futures):
                index, rows = future.result()
                results[int(index)] = rows
                progress.advance()
    else:
        for task in tasks:
            index, rows = crossvalidation_worker(task)
            results[int(index)] = rows
            progress.advance()
    progress.finish(len(selected))
    ordered_rows = [row for index in sorted(results) for row in results[index]]
    return pd.DataFrame.from_records(ordered_rows)


def choose_best_models(crossvalidation: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if crossvalidation.empty or "status" not in crossvalidation.columns:
        return pd.DataFrame()
    ok = crossvalidation.loc[crossvalidation["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for hkl, hkl_group in ok.groupby(HKL_COLUMNS, sort=False):
        model_rows = []
        for model_name, group in hkl_group.groupby("model_name", sort=False):
            slopes = pd.to_numeric(group["slope_sign"], errors="coerce").dropna().astype(int)
            nonzero_slopes = slopes.loc[slopes != 0]
            mode_sign = int(nonzero_slopes.mode().iloc[0]) if len(nonzero_slopes) else 0
            stability = float((slopes == mode_sign).mean()) if mode_sign else 0.0
            model_rows.append(
                {
                    "model_name": model_name,
                    "median_delta_mae": float(pd.to_numeric(group["heldout_delta_mae"], errors="coerce").median()),
                    "median_relative_mae_improvement": float(pd.to_numeric(group["heldout_relative_mae_improvement"], errors="coerce").median()),
                    "median_delta_mad": float(pd.to_numeric(group["heldout_delta_mad"], errors="coerce").median()),
                    "fraction_splits_improve_mae": float((pd.to_numeric(group["heldout_delta_mae"], errors="coerce") > 0.0).mean()),
                    "fraction_beats_sham_mae": float(group["beats_sham_mae"].astype(bool).mean()),
                    "slope_sign": mode_sign,
                    "slope_sign_stability": stability,
                    "slope_ci_low": float(np.nanquantile(pd.to_numeric(group["beta_or_primary_slope"], errors="coerce"), 0.025)),
                    "slope_ci_high": float(np.nanquantile(pd.to_numeric(group["beta_or_primary_slope"], errors="coerce"), 0.975)),
                    "median_high_target_delta_mae": float(pd.to_numeric(group["high_target_delta_mae"], errors="coerce").median()),
                    "median_high_partiality_delta_mae": float(pd.to_numeric(group["high_partiality_delta_mae"], errors="coerce").median()),
                    "median_abs_mean_intensity_shift": float(pd.to_numeric(group["mean_intensity_shift_after_minus_before"], errors="coerce").abs().median()),
                    "median_abs_median_intensity_shift": float(pd.to_numeric(group["median_intensity_shift_after_minus_before"], errors="coerce").abs().median()),
                    "n_ok_splits": int(len(group)),
                }
            )
        ranked = sorted(model_rows, key=lambda row: (row["median_delta_mae"], row["fraction_beats_sham_mae"], row["slope_sign_stability"]), reverse=True)
        best = ranked[0]
        rows.append({"h": int(hkl[0]), "k": int(hkl[1]), "l": int(hkl[2]), "hkl": hkl_label(tuple(map(int, hkl))), **{f"best_{key}": value for key, value in best.items()}, "all_model_ranking": json.dumps(ranked, sort_keys=True)})
    return pd.DataFrame.from_records(rows)


def apply_filter_metrics(residual: np.ndarray, response: np.ndarray, keep: np.ndarray, high_target: np.ndarray, high_partiality: np.ndarray) -> dict[str, float]:
    if int(keep.sum()) < 3:
        return {"mae": np.nan, "mad": np.nan, "std": np.nan, "mean_shift": np.nan, "median_shift": np.nan, "high_target_retention": np.nan, "high_partiality_retention": np.nan}
    kept_resid = residual[keep]
    kept_response = response[keep]
    return {
        "mae": float(np.nanmean(np.abs(kept_resid))),
        "mad": cvmod.mad(kept_resid),
        "std": float(np.nanstd(kept_resid)),
        "mean_shift": float(np.nanmean(kept_response) - np.nanmean(response)),
        "median_shift": float(np.nanmedian(kept_response) - np.nanmedian(response)),
        "high_target_retention": float((keep & high_target).sum() / max(1, high_target.sum())),
        "high_partiality_retention": float((keep & high_partiality).sum() / max(1, high_partiality.sum())),
    }


def filter_simulation_one_best(group: pd.DataFrame, best: dict[str, Any], scale_columns: list[str], args: SimpleNamespace) -> list[dict[str, Any]]:
    covariates = [TARGET_EXCITATION, PARTIALITY, *scale_columns]
    rows: list[dict[str, Any]] = []
    hkl = (int(best["h"]), int(best["k"]), int(best["l"]))
    model_name = str(best["best_model_name"])
    frame = clean_actionability_frame(group, cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates)
    for split in range(int(args.splits)):
        train_idx, test_idx = cvmod.deterministic_stratified_split(frame, split, float(args.train_fraction), int(args.seed) + 37)
        if len(train_idx) < 5 or len(test_idx) < 10:
            continue
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        train, test, _, _ = cvmod.add_train_normalized_response(train, test)
        baseline = cvmod.fit_model(train, cvmod.CV_RESPONSE_COLUMN, covariates)
        train_resid = pd.to_numeric(train[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float) - cvmod.predict_model(baseline, train)
        response_test = pd.to_numeric(test[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
        residual_test = response_test - cvmod.predict_model(baseline, test)
        fit = fit_effect_model(train, train_resid, test, model_name)
        effect = np.asarray(fit["predicted_effect_test"], dtype=float)
        rng = np.random.default_rng(int(args.seed) + 4001 * (split + 1) + 13 * sum(abs(v) for v in hkl))
        target_threshold = float(pd.to_numeric(train[TARGET_EXCITATION], errors="coerce").quantile(0.75))
        partiality_threshold = float(pd.to_numeric(train[PARTIALITY], errors="coerce").quantile(0.75))
        high_target = pd.to_numeric(test[TARGET_EXCITATION], errors="coerce").to_numpy(dtype=float) >= target_threshold
        high_partiality = pd.to_numeric(test[PARTIALITY], errors="coerce").to_numpy(dtype=float) >= partiality_threshold
        base_mae = float(np.nanmean(np.abs(residual_test)))
        for filter_name, fraction in [("abs_predicted_distortion_top_5pct", 0.05), ("abs_predicted_distortion_top_10pct", 0.10), ("harmful_one_sided_tail_5pct", 0.05), ("harmful_one_sided_tail_10pct", 0.10)]:
            remove_n = max(1, int(round(len(test) * fraction)))
            if filter_name.startswith("abs_"):
                order = np.argsort(np.nan_to_num(np.abs(effect), nan=-np.inf))[::-1]
            else:
                direction = sign_value(best.get("best_slope_sign", 0))
                score = effect if direction >= 0 else -effect
                order = np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]
            remove = np.zeros(len(test), dtype=bool)
            remove[order[:remove_n]] = True
            random_remove = np.zeros(len(test), dtype=bool)
            random_remove[rng.choice(len(test), size=remove_n, replace=False)] = True
            keep = ~remove
            random_keep = ~random_remove
            metrics = apply_filter_metrics(residual_test, response_test, keep, high_target, high_partiality)
            random_metrics = apply_filter_metrics(residual_test, response_test, random_keep, high_target, high_partiality)
            rows.append(
                {
                    "h": hkl[0], "k": hkl[1], "l": hkl[2], "hkl": hkl_label(hkl),
                    "four_mmm_orbit_id": best.get("four_mmm_orbit_id", four_mmm_orbit_id(*hkl)),
                    "split": split, "best_model_name": model_name, "filter_name": filter_name,
                    "n_test": int(len(test)), "n_removed": int(remove_n),
                    "baseline_mae_full_test": base_mae,
                    "filtered_mae": metrics["mae"], "random_mae": random_metrics["mae"],
                    "filter_delta_mae_vs_full_positive_improves": base_mae - metrics["mae"],
                    "filter_delta_mae_vs_random_positive_improves": random_metrics["mae"] - metrics["mae"],
                    "filtered_mad": metrics["mad"], "random_mad": random_metrics["mad"],
                    "filter_delta_mad_vs_random_positive_improves": random_metrics["mad"] - metrics["mad"],
                    "filtered_residual_std": metrics["std"], "random_residual_std": random_metrics["std"],
                    "mean_intensity_shift": metrics["mean_shift"], "median_intensity_shift": metrics["median_shift"],
                    "high_target_retention": metrics["high_target_retention"],
                    "high_partiality_retention": metrics["high_partiality_retention"],
                    "beats_matched_random_mae": bool(metrics["mae"] < random_metrics["mae"]),
                    "beats_matched_random_mad": bool(metrics["mad"] < random_metrics["mad"]),
                }
            )
    return rows


def filter_simulation_worker(task: tuple[int, dict[str, Any], pd.DataFrame, list[str], SimpleNamespace]) -> tuple[int, list[dict[str, Any]]]:
    index, best, group, scale_columns, args = task
    return index, filter_simulation_one_best(group, best, scale_columns, args)


def run_filter_simulation(observations: pd.DataFrame, best_models: pd.DataFrame, scale_columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    if best_models.empty:
        return pd.DataFrame()
    best_models = best_models.reset_index(drop=True)
    worker_count = effective_worker_count(args, len(best_models))
    progress = StageProgress("Candidate filtering simulation", total=len(best_models), unit="HKLs")
    tasks = []
    worker_namespace = worker_args(args)
    for index, best in best_models.iterrows():
        hkl = (int(best.h), int(best.k), int(best.l))
        tasks.append((index, best.to_dict(), observation_group_for_hkl(observations, hkl), list(scale_columns), worker_namespace))
    results: dict[int, list[dict[str, Any]]] = {}
    use_processes = int(args.workers) > 1
    if use_processes:
        set_worker_numeric_threads()
        with ProcessPoolExecutor(max_workers=worker_count, initializer=worker_initializer) as executor:
            futures = [executor.submit(filter_simulation_worker, task) for task in tasks]
            for future in as_completed(futures):
                index, rows = future.result()
                results[int(index)] = rows
                progress.advance()
    else:
        for task in tasks:
            index, rows = filter_simulation_worker(task)
            results[int(index)] = rows
            progress.advance()
    progress.finish(len(best_models))
    ordered_rows = [row for index in sorted(results) for row in results[index]]
    return pd.DataFrame.from_records(ordered_rows)


def summarize_filtering(filtering: pd.DataFrame) -> pd.DataFrame:
    if filtering.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for hkl, hkl_group in filtering.groupby(HKL_COLUMNS, sort=False):
        best_rows = []
        for filter_name, group in hkl_group.groupby("filter_name", sort=False):
            best_rows.append(
                {
                    "filter_name": filter_name,
                    "median_delta_mae_vs_random": float(pd.to_numeric(group["filter_delta_mae_vs_random_positive_improves"], errors="coerce").median()),
                    "fraction_beats_random_mae": float(group["beats_matched_random_mae"].astype(bool).mean()),
                    "median_high_target_retention": float(pd.to_numeric(group["high_target_retention"], errors="coerce").median()),
                    "median_high_partiality_retention": float(pd.to_numeric(group["high_partiality_retention"], errors="coerce").median()),
                }
            )
        ranked = sorted(best_rows, key=lambda row: (row["fraction_beats_random_mae"], row["median_delta_mae_vs_random"]), reverse=True)
        rows.append({"h": int(hkl[0]), "k": int(hkl[1]), "l": int(hkl[2]), "hkl": hkl_label(tuple(map(int, hkl))), **{f"best_filter_{key}": value for key, value in ranked[0].items()}, "all_filter_ranking": json.dumps(ranked, sort_keys=True)})
    return pd.DataFrame.from_records(rows)


def run_symmetry_transfer(observations: pd.DataFrame, best_models: pd.DataFrame, scale_columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    if best_models.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    covariates = [TARGET_EXCITATION, PARTIALITY, *scale_columns]
    best_lookup = {tuple(map(int, row[:3])): model for row, model in zip(best_models.loc[:, HKL_COLUMNS].itertuples(index=False, name=None), best_models["best_model_name"], strict=False)}
    total_pairs = int(sum(len(group) * (len(group) - 1) for _, group in best_models.groupby("four_mmm_orbit_id", sort=False)))
    progress = StageProgress("Symmetry-orbit transfer checks", total=total_pairs, unit="directed pairs")
    for orbit_id, group in best_models.groupby("four_mmm_orbit_id", sort=False):
        hkls = [tuple(map(int, row)) for row in group.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
        for train_hkl, test_hkl in combinations(hkls, 2):
            for a, b in [(train_hkl, test_hkl), (test_hkl, train_hkl)]:
                if best_lookup.get(a) != best_lookup.get(b):
                    rows.append({"four_mmm_orbit_id": orbit_id, "train_hkl": hkl_label(a), "test_hkl": hkl_label(b), "status": "best_model_mismatch"})
                    progress.advance()
                    continue
                train_group = observations.loc[(observations["h"].astype(int) == a[0]) & (observations["k"].astype(int) == a[1]) & (observations["l"].astype(int) == a[2])]
                test_group = observations.loc[(observations["h"].astype(int) == b[0]) & (observations["k"].astype(int) == b[1]) & (observations["l"].astype(int) == b[2])]
                train = clean_actionability_frame(train_group, cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates)
                test = clean_actionability_frame(test_group, cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates)
                if len(train) < int(args.min_nonzero_coupling) or len(test) < int(args.min_nonzero_coupling):
                    rows.append({"four_mmm_orbit_id": orbit_id, "train_hkl": hkl_label(a), "test_hkl": hkl_label(b), "status": "insufficient_data", "n_train": int(len(train)), "n_test": int(len(test))})
                    progress.advance()
                    continue
                train, test, _, _ = cvmod.add_train_normalized_response(train, test)
                baseline = cvmod.fit_model(train, cvmod.CV_RESPONSE_COLUMN, covariates)
                train_resid = pd.to_numeric(train[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float) - cvmod.predict_model(baseline, train)
                y_test = pd.to_numeric(test[cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float)
                test_resid = y_test - cvmod.predict_model(baseline, test)
                fit = fit_effect_model(train, train_resid, test, str(best_lookup[a]))
                corrected = test_resid - np.asarray(fit["predicted_effect_test"], dtype=float)
                rows.append(
                    {
                        "four_mmm_orbit_id": orbit_id, "train_hkl": hkl_label(a), "test_hkl": hkl_label(b),
                        "status": "ok", "best_model_name": best_lookup[a], "n_train": int(len(train)), "n_test": int(len(test)),
                        "transfer_delta_mae": float(np.nanmean(np.abs(test_resid)) - np.nanmean(np.abs(corrected))),
                        "transfer_improves_mae": bool(np.nanmean(np.abs(corrected)) < np.nanmean(np.abs(test_resid))),
                    }
                )
                progress.advance()
    progress.finish(total_pairs)
    return pd.DataFrame.from_records(rows)


def build_orbit_summary(decision_base: pd.DataFrame, transfer: pd.DataFrame, filtering_summary: pd.DataFrame) -> pd.DataFrame:
    if decision_base.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    filter_lookup = {tuple(map(int, row[:3])): data for row, data in zip(filtering_summary.loc[:, HKL_COLUMNS].itertuples(index=False, name=None), filtering_summary.to_dict("records"), strict=False)} if not filtering_summary.empty else {}
    for orbit_id, group in decision_base.groupby("four_mmm_orbit_id", sort=False):
        signs = [sign_value(value) for value in group.get("best_slope_sign", pd.Series(dtype=float))]
        nonzero = [value for value in signs if value != 0]
        sign_agreement = float(Counter(nonzero).most_common(1)[0][1] / len(nonzero)) if nonzero else np.nan
        models = group.get("best_model_name", pd.Series(dtype=object)).dropna().astype(str)
        model_agreement = float(models.value_counts().iloc[0] / len(models)) if len(models) else np.nan
        transfer_group = transfer.loc[transfer["four_mmm_orbit_id"] == orbit_id] if not transfer.empty and "four_mmm_orbit_id" in transfer.columns else pd.DataFrame()
        ok_transfer = transfer_group.loc[transfer_group.get("status", pd.Series(dtype=object)) == "ok"] if not transfer_group.empty else pd.DataFrame()
        rows.append(
            {
                "four_mmm_orbit_id": orbit_id,
                "n_signed_hkls_in_decision_table": int(len(group)),
                "slope_sign_agreement_fraction": sign_agreement,
                "best_model_agreement_fraction": model_agreement,
                "n_transfer_tests_ok": int(len(ok_transfer)),
                "fraction_transfer_improves_mae": float(ok_transfer["transfer_improves_mae"].astype(bool).mean()) if not ok_transfer.empty else np.nan,
                "n_correction_ready_mates": int(group["correction_ready_flag"].astype(bool).sum()) if "correction_ready_flag" in group else 0,
                "n_filtering_ready_mates": int(sum(bool(filter_lookup.get(tuple(map(int, row)), {}).get("filtering_ready_flag", False)) for row in group.loc[:, HKL_COLUMNS].itertuples(index=False, name=None))),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_decision_table(screen: pd.DataFrame, candidates: pd.DataFrame, best_models: pd.DataFrame, filtering_summary: pd.DataFrame, transfer: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base = candidates.copy()
    if not best_models.empty:
        best_to_merge = best_models.drop(columns=["four_mmm_orbit_id"], errors="ignore")
        base = base.merge(best_to_merge, on=HKL_COLUMNS + ["hkl"], how="left", validate="one_to_one")
    if not filtering_summary.empty:
        filtering_to_merge = filtering_summary.drop(columns=["four_mmm_orbit_id"], errors="ignore")
        base = base.merge(filtering_to_merge, on=HKL_COLUMNS + ["hkl"], how="left", validate="one_to_one")
    transfer_support: dict[str, bool] = {}
    if not transfer.empty and "status" in transfer.columns:
        for hkl_text, group in transfer.loc[transfer["status"] == "ok"].groupby("test_hkl", sort=False):
            transfer_support[hkl_text] = bool(group["transfer_improves_mae"].astype(bool).any())
    classifications = []
    directions = []
    correction_flags = []
    filtering_flags = []
    for _, row in base.iterrows():
        n_nonzero = int(row.get("nonzero_coupling_count", 0) or 0)
        if n_nonzero < int(args.min_nonzero_coupling):
            classifications.append("insufficient data")
            directions.append("none")
            correction_flags.append(False)
            filtering_flags.append(False)
            continue
        best_delta = float(row.get("best_median_delta_mae", np.nan))
        sign_stability = float(row.get("best_slope_sign_stability", np.nan))
        beats_sham = float(row.get("best_fraction_beats_sham_mae", np.nan))
        high_target = float(row.get("best_median_high_target_delta_mae", np.nan))
        high_part = float(row.get("best_median_high_partiality_delta_mae", np.nan))
        high_quality_ok = (not np.isfinite(high_target) or high_target >= -float(args.max_high_quality_degradation)) and (not np.isfinite(high_part) or high_part >= -float(args.max_high_quality_degradation))
        correction_ready = bool(np.isfinite(best_delta) and best_delta > float(args.min_correction_delta_mae) and sign_stability >= float(args.stable_sign_fraction) and beats_sham >= 0.60 and high_quality_ok)
        filter_delta = float(row.get("best_filter_median_delta_mae_vs_random", np.nan))
        filter_fraction = float(row.get("best_filter_fraction_beats_random_mae", np.nan))
        target_retention = float(row.get("best_filter_median_high_target_retention", np.nan))
        partiality_retention = float(row.get("best_filter_median_high_partiality_retention", np.nan))
        filtering_ready = bool(np.isfinite(filter_delta) and filter_delta > 0.0 and filter_fraction >= float(args.min_filter_random_beat_fraction) and (not np.isfinite(target_retention) or target_retention >= 0.85) and (not np.isfinite(partiality_retention) or partiality_retention >= 0.85))
        correction_flags.append(correction_ready)
        filtering_flags.append(filtering_ready)
        model = str(row.get("best_model_name", ""))
        slope_sign = sign_value(row.get("best_slope_sign", 0))
        if model in {"deficit_quintile", "high_deficit_threshold", "piecewise_linear_deficit"}:
            direction = "nonlinear/threshold correction only"
        elif slope_sign > 0:
            direction = "downward correction at high deficit"
        elif slope_sign < 0:
            direction = "upward correction at high deficit"
        else:
            direction = "none"
        directions.append(direction)
        if correction_ready and filtering_ready:
            classifications.append("correction-and-filtering candidate")
        elif correction_ready:
            classifications.append("correction-ready")
        elif filtering_ready:
            classifications.append("filtering-ready only")
        elif bool(row.get("cheap_screen_pass", False)) or np.isfinite(best_delta):
            classifications.append("promising but uncertain")
        else:
            classifications.append("no supported effect")
    base["correction_ready_flag"] = correction_flags
    base["filtering_ready_flag"] = filtering_flags
    base["symmetry_transfer_supported"] = [bool(transfer_support.get(str(hkl), False)) for hkl in base["hkl"]]
    base["recommended_correction_direction"] = directions
    base["actionability_classification"] = classifications
    return base


def make_plots(out_dir: Path, screen: pd.DataFrame, best_models: pd.DataFrame, decision: pd.DataFrame, dpi: int) -> list[Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.hist(pd.to_numeric(screen["candidate_score"], errors="coerce").dropna(), bins=60, color="0.25")
    ax.set_xlabel("cheap-screen candidate score")
    ax.set_ylabel("signed HKLs")
    path = plot_dir / "cheap_screen_candidate_score_histogram.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    paths.append(path)
    if not best_models.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        counts = best_models["best_model_name"].value_counts()
        ax.bar(counts.index.astype(str), counts.to_numpy())
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("best-model HKL count")
        path = plot_dir / "best_model_counts.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths.append(path)
    if not decision.empty and "actionability_classification" in decision.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        counts = decision["actionability_classification"].value_counts()
        ax.bar(counts.index.astype(str), counts.to_numpy())
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("signed HKL count")
        path = plot_dir / "actionability_class_counts.png"
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        paths.append(path)
    return paths


def write_readme(out_dir: Path, args: argparse.Namespace, metadata: dict[str, Any], plot_count: int) -> None:
    lines = [
        "# V5 Excitation-Deficit All-HKL Actionability Screen",
        "",
        "Screens signed HKLs for orientation-dependent correction and filtering actionability using existing accepted observations, existing v5 scores, and existing merged intensities.",
        "",
        "No v5 recomputation, stream rewrite, Partialator run, merge, or refinement is performed.",
        "",
        "## Leakage Controls",
        "",
        "Detailed validation uses train-only intensity normalization, train-only baseline fitting, train-only descriptor fitting, and train-only thresholds/quintile/correction parameters.",
        "",
        "## Cheap-Screen Thresholds",
        "",
        f"- min nonzero coupling: `{args.min_nonzero_coupling}`",
        f"- min descriptor spread: `{args.min_descriptor_spread}`",
        f"- min abs slope: `{args.min_abs_slope}`",
        f"- min abs Spearman rho: `{args.min_abs_rho}`",
        f"- max candidates: `{args.max_candidates}`",
        "",
        "## Outputs",
        "",
        "- `all_hkl_screen.csv`",
        "- `candidate_hkls.csv`",
        "- `candidate_crossvalidation.csv`",
        "- `candidate_best_model.csv`",
        "- `candidate_filter_simulation.csv`",
        "- `candidate_symmetry_orbit_summary.csv`",
        "- `actionability_decision_table.csv`",
        "- `run_metadata.json`",
        f"- compact aggregate plots: `{plot_count}` PNG files under `plots/`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(out_dir: Path, args: argparse.Namespace, metadata: dict[str, Any], outputs: dict[str, Path]) -> None:
    run_metadata = {
        "command": "tools/screen_all_hkls_v5_dexc_actionability.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "candidate_hkl_set"},
        "inputs": {"accepted": str(args.accepted), "v5_scores": str(args.v5_scores), "strength_table": str(args.strength_table), "stream": str(args.stream)},
        "thresholds": {"min_nonzero_coupling": args.min_nonzero_coupling, "min_descriptor_spread": args.min_descriptor_spread, "min_abs_slope": args.min_abs_slope, "min_abs_rho": args.min_abs_rho, "max_candidates": args.max_candidates},
        "metadata": metadata,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "leakage_controls": {"normalization": "training HKL median only", "baseline_fit": "training observations only", "descriptor_fit": "training residuals only", "test_observations_used_for_fitting_or_thresholds": False},
        "did_not_run": ["v5 recomputation", "stream rewrite", "Partialator", "merge", "structure refinement"],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(screen: pd.DataFrame, candidates: pd.DataFrame, best_models: pd.DataFrame, filtering: pd.DataFrame, transfer: pd.DataFrame, decision: pd.DataFrame) -> None:
    screened = int(len(screen))
    passed = int(screen["cheap_screen_pass"].astype(bool).sum()) if "cheap_screen_pass" in screen else 0
    correction_ready = int(decision["correction_ready_flag"].astype(bool).sum()) if not decision.empty and "correction_ready_flag" in decision else 0
    filtering_ready = int(decision["filtering_ready_flag"].astype(bool).sum()) if not decision.empty and "filtering_ready_flag" in decision else 0
    symmetry_supported = int(decision["symmetry_transfer_supported"].astype(bool).sum()) if not decision.empty and "symmetry_transfer_supported" in decision else 0
    print("\nV5 Dexc actionability screen summary:")
    print(f"  screened signed HKLs: {screened:,}")
    print(f"  cheap-screen candidates: {passed:,}")
    print(f"  correction-ready: {correction_ready:,}")
    print(f"  filtering-ready: {filtering_ready:,}")
    print(f"  supported by symmetry transfer: {symmetry_supported:,}")
    if not decision.empty:
        cols = ["hkl", "best_model_name", "best_median_delta_mae", "best_slope_sign_stability", "recommended_correction_direction", "actionability_classification"]
        print("\nTop 20 adjustment candidates:")
        if "best_median_delta_mae" in decision.columns:
            adjustment = decision.sort_values("best_median_delta_mae", ascending=False)
        else:
            adjustment = decision.sort_values("candidate_score", ascending=False) if "candidate_score" in decision.columns else decision
        print(adjustment.loc[:, [column for column in cols if column in adjustment.columns]].head(20).to_string(index=False))
        filter_cols = ["hkl", "best_filter_filter_name", "best_filter_median_delta_mae_vs_random", "best_filter_fraction_beats_random_mae", "actionability_classification"]
        print("\nTop 20 filtering candidates:")
        if "best_filter_median_delta_mae_vs_random" in decision.columns:
            print(decision.sort_values("best_filter_median_delta_mae_vs_random", ascending=False).loc[:, [column for column in filter_cols if column in decision.columns]].head(20).to_string(index=False))
        else:
            print("  No filtering simulation rows were produced.")


def report_candidate_counts(candidates: pd.DataFrame, args: argparse.Namespace) -> None:
    selected = candidates.loc[candidates["detailed_validation_selected"].astype(bool)] if "detailed_validation_selected" in candidates.columns else candidates
    candidate_count = int(len(selected))
    orbit_count = int(selected["four_mmm_orbit_id"].nunique(dropna=True)) if "four_mmm_orbit_id" in selected.columns else 0
    worker_count = effective_worker_count(args, candidate_count)
    log(f"Candidate signed-HKL count: {candidate_count:,}")
    log(f"Candidate orbit count: {orbit_count:,}")
    log(f"Selected worker count: {worker_count:,}")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_table = load_candidate_table(args.candidate_file) if args.candidate_file is not None else None
    args.candidate_hkl_set = candidate_hkl_set(candidate_table)
    if candidate_table is not None:
        log(f"Candidate-file mode enabled: preserving {len(candidate_table):,} exact signed HKL(s) from {args.candidate_file}")
    log("Loading accepted observations and exact-key v5 scores")
    observations, scale_columns, metadata = prepare_observations(args)
    log(f"Matched accepted/v5 observations: {len(observations):,}")
    if candidate_table is not None:
        log(f"Retained accepted/v5 observation count after candidate filtering: {len(observations):,}")
        if args.screen_dir is not None:
            log(f"Reusing completed screen metadata from {args.screen_dir}")
            screen = load_screen_dir_candidate_metadata(args.screen_dir, candidate_table, args)
        else:
            log("Building candidate-only observation metadata")
            screen = build_candidate_observation_screen(observations, candidate_table, args)
        candidates = choose_candidate_file_candidates(screen)
    else:
        log("Building cheap all-HKL screen")
        screen = build_all_hkl_screen(observations, args)
        candidates = choose_candidates(screen, args)
    metadata["candidate_file_mode"] = {
        "enabled": candidate_table is not None,
        "candidate_file": str(args.candidate_file) if args.candidate_file is not None else None,
        "screen_dir": str(args.screen_dir) if args.screen_dir is not None else None,
        "candidate_signed_hkl_count": int(len(candidate_table)) if candidate_table is not None else None,
        "candidate_orbit_count": int(candidate_table["four_mmm_orbit_id"].nunique(dropna=True)) if candidate_table is not None and "four_mmm_orbit_id" in candidate_table.columns else None,
        "retained_accepted_v5_observation_count": int(len(observations)) if candidate_table is not None else None,
    }
    report_candidate_counts(candidates, args)

    if args.screen_only:
        crossvalidation = pd.DataFrame()
        best_models = pd.DataFrame()
        filtering = pd.DataFrame()
        filtering_summary = pd.DataFrame()
        transfer = pd.DataFrame()
        decision = candidates.copy()
        decision["correction_ready_flag"] = False
        decision["filtering_ready_flag"] = False
        decision["symmetry_transfer_supported"] = False
        decision["recommended_correction_direction"] = "none"
        decision["actionability_classification"] = np.where(decision["cheap_screen_pass"].astype(bool), "promising but uncertain", np.where(decision["nonzero_coupling_count"] < int(args.min_nonzero_coupling), "insufficient data", "no supported effect"))
        orbit_summary = pd.DataFrame()
    else:
        log("Running leakage-free candidate crossvalidation")
        crossvalidation = run_candidate_crossvalidation(observations, candidates, scale_columns, args)
        log("Choosing best held-out model per candidate HKL")
        best_models = choose_best_models(crossvalidation, args)
        if not best_models.empty:
            best_models = best_models.merge(candidates.loc[:, [*HKL_COLUMNS, "hkl", "four_mmm_orbit_id"]], on=HKL_COLUMNS + ["hkl"], how="left", validate="one_to_one")
        log("Running filtering simulation for candidate best models")
        filtering = run_filter_simulation(observations, best_models, scale_columns, args)
        filtering_summary = summarize_filtering(filtering)
        log("Running symmetry-orbit transfer checks")
        transfer = run_symmetry_transfer(observations, best_models, scale_columns, args)
        decision = build_decision_table(screen, candidates, best_models, filtering_summary, transfer, args)
        orbit_summary = build_orbit_summary(decision, transfer, filtering_summary)

    outputs = {
        "all_hkl_screen": args.out_dir / "all_hkl_screen.csv",
        "candidate_hkls": args.out_dir / "candidate_hkls.csv",
        "candidate_crossvalidation": args.out_dir / "candidate_crossvalidation.csv",
        "candidate_best_model": args.out_dir / "candidate_best_model.csv",
        "candidate_filter_simulation": args.out_dir / "candidate_filter_simulation.csv",
        "candidate_symmetry_orbit_summary": args.out_dir / "candidate_symmetry_orbit_summary.csv",
        "actionability_decision_table": args.out_dir / "actionability_decision_table.csv",
        "README": args.out_dir / "README.md",
        "run_metadata": args.out_dir / "run_metadata.json",
    }
    screen.to_csv(outputs["all_hkl_screen"], index=False)
    candidates.to_csv(outputs["candidate_hkls"], index=False)
    crossvalidation.to_csv(outputs["candidate_crossvalidation"], index=False)
    best_models.to_csv(outputs["candidate_best_model"], index=False)
    filtering.to_csv(outputs["candidate_filter_simulation"], index=False)
    orbit_summary.to_csv(outputs["candidate_symmetry_orbit_summary"], index=False)
    decision.to_csv(outputs["actionability_decision_table"], index=False)
    plots = make_plots(args.out_dir, screen, best_models, decision, int(args.plot_dpi))
    write_readme(args.out_dir, args, metadata, len(plots))
    write_metadata(args.out_dir, args, metadata, outputs)
    print_summary(screen, candidates, best_models, filtering, transfer, decision)
    print(f"\nOutputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())