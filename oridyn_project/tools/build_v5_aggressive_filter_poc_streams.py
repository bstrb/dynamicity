#!/usr/bin/env python3
"""Build aggressive orientation-aware v5 filtering POC streams.

This script consumes completed broad v5 actionability outputs and prepares an
aggressive intervention intended to be large enough to detect during merging if
the orientation-aware prediction is useful. It preserves exact signed h,k,l and
never uses merged intensity or Fobs as ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import screen_all_hkls_v5_dexc_actionability as actionmod  # noqa: E402


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
ABS_FILTER_NAME = "abs_predicted_distortion_top_10pct"
DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR = 6_732_955
DEFAULT_CHUNKSIZE = 500_000
DISALLOWED_EVIDENCE_COLUMNS = ["actionability_classification", "merged_intensity_or_Fobs", "Fobs", "candidate_score", "cheap_screen_pass"]
BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-summary-dir", type=Path, required=True)
    parser.add_argument("--broad-actionability-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--v5-scores", type=Path, required=True)
    parser.add_argument("--input-stream", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--filter-fractions", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--exclude-extreme-high-resolution", action="store_true")
    parser.add_argument("--min-usable-mates", type=int, default=2)
    parser.add_argument("--min-positive-mate-fraction", type=float, default=0.75)
    parser.add_argument("--min-orbit-median-delta", type=float, default=0.02)
    parser.add_argument("--min-stable-sign-fraction", type=float, default=0.75)
    parser.add_argument("--min-symmetry-support-fraction", type=float, default=0.25)
    parser.add_argument("--min-signed-hkl-sign-stability", type=float, default=0.80)
    parser.add_argument("--min-filter-beats-random-fraction", type=float, default=0.80)
    parser.add_argument("--min-remaining-observations", type=int, default=30)
    args = parser.parse_args()

    for label, path in [
        ("--broad-summary-dir", args.broad_summary_dir),
        ("--broad-actionability-dir", args.broad_actionability_dir),
    ]:
        if not path.is_dir():
            raise SystemExit(f"{label} not found: {path}")
    for label, path in [
        ("--manifest", args.manifest),
        ("--accepted", args.accepted),
        ("--v5-scores", args.v5_scores),
        ("--input-stream", args.input_stream),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    for filename in ["broad_orbit_summary.csv", "broad_signed_hkl_results.csv"]:
        if not (args.broad_summary_dir / filename).is_file():
            raise SystemExit(f"Missing input: {args.broad_summary_dir / filename}")
    for filename in ["candidate_best_model.csv", "candidate_filter_simulation.csv"]:
        if not (args.broad_actionability_dir / filename).is_file():
            raise SystemExit(f"Missing input: {args.broad_actionability_dir / filename}")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    if int(args.min_usable_mates) < 1:
        raise SystemExit("--min-usable-mates must be >= 1")
    if int(args.min_remaining_observations) < 1:
        raise SystemExit("--min-remaining-observations must be >= 1")
    fractions = []
    for value in args.filter_fractions:
        fraction = float(value)
        if not np.isfinite(fraction) or not (0.0 < fraction < 1.0):
            raise SystemExit("--filter-fractions values must satisfy 0 < fraction < 1")
        fractions.append(fraction)
    args.filter_fractions = sorted(set(fractions))
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
    def __init__(self, stage: str, total: int | None, unit: str = "items"):
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.start_time = time.monotonic()
        self.last_report_time = self.start_time
        self.completed = 0
        total_text = f"{self.total:,} {self.unit}" if self.total is not None else f"unknown {self.unit}"
        log(f"{self.stage} start: total={total_text}")

    def update(self, completed: int, force: bool = False) -> None:
        completed = int(completed)
        now = time.monotonic()
        if not force and completed > 20 and (now - self.last_report_time) < 30.0 and completed != self.total:
            self.completed = completed
            return
        elapsed = max(now - self.start_time, 1.0e-9)
        rate = completed / elapsed
        message = f"{self.stage} progress: completed={completed:,}"
        if self.total is not None:
            pct = 100.0 * completed / max(1, self.total)
            message += f"/{self.total:,} ({pct:.1f}%)"
        message += f"; elapsed={format_duration(elapsed)}; rate={rate:.2f} {self.unit}/s"
        if self.total is not None and completed > 0 and completed < self.total and rate > 0.0:
            message += f"; ETA={format_duration((self.total - completed) / rate)}"
        log(message)
        self.last_report_time = now
        self.completed = completed

    def advance(self, amount: int = 1) -> None:
        self.update(self.completed + int(amount))

    def finish(self, completed: int | None = None) -> None:
        final = self.completed if completed is None else int(completed)
        elapsed = max(time.monotonic() - self.start_time, 1.0e-9)
        rate = final / elapsed
        message = f"{self.stage} complete: completed={final:,}"
        if self.total is not None:
            pct = 100.0 * final / max(1, self.total)
            message += f"/{self.total:,} ({pct:.1f}%)"
        message += f"; elapsed={format_duration(elapsed)}; rate={rate:.2f} {self.unit}/s"
        log(message)


def set_worker_numeric_threads() -> None:
    for name in BLAS_THREAD_ENV_VARS:
        os.environ[name] = "1"


def worker_initializer() -> None:
    set_worker_numeric_threads()


def percent_label(fraction: float) -> int:
    return int(round(float(fraction) * 100.0))


def hkl_label(h: int, k: int, l: int) -> str:
    return f"({int(h)},{int(k)},{int(l)})"


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
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def build_key(source: Any, event: Any, h: int, k: int, l: int) -> tuple[str, str, int, int, int]:
    return normalize_source(source), normalize_event(event), int(h), int(k), int(l)


def key_to_text(key: tuple[str, str, int, int, int]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}|{key[4]}"


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], byteorder="big", signed=False)


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def read_csv_maybe_empty(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def normalize_hkl_table(table: pd.DataFrame, label: str) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    require_columns(table, HKL_COLUMNS, label)
    out = table.copy()
    for column in HKL_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise SystemExit(f"{label} contains missing/noninteger {column} values")
        out[column] = values.astype(int)
    out["hkl"] = [hkl_label(row.h, row.k, row.l) for row in out.loc[:, HKL_COLUMNS].itertuples(index=False)]
    return out


def to_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    text = values.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def finite_numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index)
    return pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def selected_hkl_set(table: pd.DataFrame) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in table.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)}


def filter_to_hkls(table: pd.DataFrame, hkls: set[tuple[int, int, int]]) -> pd.DataFrame:
    if table.empty:
        return table
    numeric = table.loc[:, HKL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mask = [
        tuple(map(int, row)) in hkls if all(np.isfinite(row)) else False
        for row in numeric.itertuples(index=False, name=None)
    ]
    return table.loc[np.asarray(mask, dtype=bool)].copy()


def parse_reflection_hkl(line: str) -> tuple[int, int, int] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return None
    return h, k, l


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orbit = pd.read_csv(args.broad_summary_dir / "broad_orbit_summary.csv", low_memory=False)
    signed = normalize_hkl_table(pd.read_csv(args.broad_summary_dir / "broad_signed_hkl_results.csv", low_memory=False), "broad_signed_hkl_results.csv")
    best = normalize_hkl_table(pd.read_csv(args.broad_actionability_dir / "candidate_best_model.csv", low_memory=False), "candidate_best_model.csv")
    filtering = normalize_hkl_table(read_csv_maybe_empty(args.broad_actionability_dir / "candidate_filter_simulation.csv"), "candidate_filter_simulation.csv")
    manifest = normalize_hkl_table(pd.read_csv(args.manifest, low_memory=False), "manifest")
    return orbit, signed, best, filtering, manifest


def build_filter_evidence(filtering: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = [*HKL_COLUMNS, "filter_name", "filter_delta_mae_vs_random_positive_improves", "beats_matched_random_mae"]
    require_columns(filtering, required, "candidate_filter_simulation.csv")
    use = filtering.loc[filtering["filter_name"].astype(str) == ABS_FILTER_NAME].copy()
    if use.empty:
        return pd.DataFrame(columns=[*HKL_COLUMNS, "filter_abs10_median_delta_mae_vs_random", "filter_abs10_fraction_beats_random", "filter_abs10_n_rows"])
    use["filter_delta_mae_vs_random_positive_improves"] = finite_numeric(use, "filter_delta_mae_vs_random_positive_improves")
    use["beats_matched_random_mae"] = to_bool_series(use["beats_matched_random_mae"])
    grouped = use.groupby(HKL_COLUMNS, sort=False)
    return grouped.agg(
        filter_abs10_median_delta_mae_vs_random=("filter_delta_mae_vs_random_positive_improves", "median"),
        filter_abs10_fraction_beats_random=("beats_matched_random_mae", "mean"),
        filter_abs10_n_rows=("filter_name", "size"),
    ).reset_index()


def select_orbits_and_hkls(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    orbit, signed, best, filtering, manifest = load_inputs(args)
    orbit_required = [
        "four_mmm_orbit_id",
        "usable_mate_count",
        "fraction_positive_heldout_delta_mae",
        "median_heldout_delta_mae",
        "stable_sign_fraction",
        "symmetry_transfer_supported_fraction",
    ]
    require_columns(orbit, orbit_required, "broad_orbit_summary.csv")
    signed_required = [*HKL_COLUMNS, "four_mmm_orbit_id", "best_median_delta_mae", "best_slope_sign_stability"]
    require_columns(signed, signed_required, "broad_signed_hkl_results.csv")
    require_columns(best, [*HKL_COLUMNS, "best_model_name"], "candidate_best_model.csv")
    require_columns(manifest, [*HKL_COLUMNS, "resolution_angstrom", "resolution_shell_10_label", "resolution_shell_20_label"], "manifest")

    if args.exclude_extreme_high_resolution and "extreme_high_resolution_5pct" not in manifest.columns:
        raise SystemExit("--exclude-extreme-high-resolution requires manifest column extreme_high_resolution_5pct")
    if "extreme_high_resolution_5pct" not in manifest.columns:
        manifest["extreme_high_resolution_5pct"] = False
    if "extreme_low_resolution_1pct" not in manifest.columns:
        manifest["extreme_low_resolution_1pct"] = False

    orbit_work = orbit.copy()
    for column in orbit_required[1:]:
        orbit_work[column] = finite_numeric(orbit_work, column)
    orbit_mask = (
        (orbit_work["usable_mate_count"] >= int(args.min_usable_mates))
        & (orbit_work["fraction_positive_heldout_delta_mae"] >= float(args.min_positive_mate_fraction))
        & (orbit_work["median_heldout_delta_mae"] >= float(args.min_orbit_median_delta))
        & (orbit_work["stable_sign_fraction"] >= float(args.min_stable_sign_fraction))
        & (orbit_work["symmetry_transfer_supported_fraction"] >= float(args.min_symmetry_support_fraction))
    )
    selected_orbits = orbit_work.loc[orbit_mask].copy().reset_index(drop=True)

    signed_base = signed.drop(columns=["best_model_name"] if "best_model_name" in signed.columns else [])
    signed_work = signed_base.merge(
        best.loc[:, [*HKL_COLUMNS, "best_model_name"]].drop_duplicates(HKL_COLUMNS, keep="first"),
        on=HKL_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    manifest_payload = [column for column in [*HKL_COLUMNS, "resolution_angstrom", "reciprocal_resolution", "resolution_shell_10_index", "resolution_shell_10_label", "resolution_shell_20_index", "resolution_shell_20_label", "extreme_low_resolution_1pct", "extreme_high_resolution_5pct"] if column in manifest.columns]
    manifest_meta = manifest.loc[:, manifest_payload].drop_duplicates(HKL_COLUMNS, keep="first")
    signed_work = signed_work.merge(manifest_meta, on=HKL_COLUMNS, how="left", validate="one_to_one", suffixes=("", "_manifest"))
    for column in [value for value in manifest_payload if value not in HKL_COLUMNS]:
        manifest_column = f"{column}_manifest"
        if manifest_column not in signed_work.columns:
            continue
        if column in signed_work.columns:
            signed_work[column] = signed_work[column].where(signed_work[column].notna(), signed_work[manifest_column])
        else:
            signed_work[column] = signed_work[manifest_column]
        signed_work = signed_work.drop(columns=[manifest_column])
    signed_work = signed_work.merge(build_filter_evidence(filtering, args), on=HKL_COLUMNS, how="left", validate="one_to_one")
    signed_work["best_median_delta_mae"] = finite_numeric(signed_work, "best_median_delta_mae")
    signed_work["best_slope_sign_stability"] = finite_numeric(signed_work, "best_slope_sign_stability")
    signed_work["filter_abs10_median_delta_mae_vs_random"] = finite_numeric(signed_work, "filter_abs10_median_delta_mae_vs_random")
    signed_work["filter_abs10_fraction_beats_random"] = finite_numeric(signed_work, "filter_abs10_fraction_beats_random")
    signed_work["extreme_high_resolution_5pct"] = to_bool_series(signed_work["extreme_high_resolution_5pct"])
    signed_work["extreme_low_resolution_1pct"] = to_bool_series(signed_work["extreme_low_resolution_1pct"])

    selected_orbits = selected_orbits.drop(columns=[column for column in DISALLOWED_EVIDENCE_COLUMNS if column in selected_orbits.columns])
    signed_work = signed_work.drop(columns=[column for column in DISALLOWED_EVIDENCE_COLUMNS if column in signed_work.columns])
    selected_orbit_ids = set(selected_orbits["four_mmm_orbit_id"].astype(str))
    signed_work["excluded_by_extreme_high_resolution_flag"] = bool(args.exclude_extreme_high_resolution) & signed_work["extreme_high_resolution_5pct"].to_numpy(dtype=bool)
    signed_mask = (
        signed_work["four_mmm_orbit_id"].astype(str).isin(selected_orbit_ids)
        & (signed_work["best_median_delta_mae"] > 0.0)
        & (signed_work["best_slope_sign_stability"] >= float(args.min_signed_hkl_sign_stability))
        & (signed_work["filter_abs10_median_delta_mae_vs_random"] > 0.0)
        & (signed_work["filter_abs10_fraction_beats_random"] >= float(args.min_filter_beats_random_fraction))
        & ~signed_work["excluded_by_extreme_high_resolution_flag"]
    )
    selected_signed = signed_work.loc[signed_mask].copy().sort_values(HKL_COLUMNS).reset_index(drop=True)
    if selected_orbits.empty or selected_signed.empty:
        log("Selection produced no signed HKLs; thresholds were not relaxed")

    metadata = {
        "selected_orbit_count": int(len(selected_orbits)),
        "selected_signed_hkl_count": int(len(selected_signed)),
        "excluded_extreme_high_resolution_5pct_count": int(signed_work["excluded_by_extreme_high_resolution_flag"].sum()),
        "selection_rules": {
            "uses_actionability_classification": False,
            "uses_merged_intensity_or_Fobs": False,
            "uses_Fobs": False,
            "uses_candidate_score": False,
            "uses_cheap_screen_pass": False,
            "preserves_exact_signed_hkl": True,
        },
    }
    return selected_orbits, selected_signed, metadata


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_accepted_selected(args: argparse.Namespace, selected: pd.DataFrame, include_intensity: bool) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    selected_set = selected_hkl_set(selected)
    header = read_header(args.accepted)
    actionmod.require_columns(header, KEY_COLUMNS, "accepted table")
    scale_columns: list[str] = []
    usecols = list(KEY_COLUMNS)
    intensity_column = None
    sigma_column = None
    if include_intensity:
        choice = actionmod.choose_intensity_columns(header)
        intensity_column = choice.column
        sigma_column = choice.sigma_column
        scale_columns = list(choice.scale_columns)
        optional = ["partiality", "I_unmerged", choice.column, *choice.scale_columns]
        if choice.sigma_column is not None:
            optional.append(choice.sigma_column)
        usecols = list(dict.fromkeys([*KEY_COLUMNS, *[column for column in optional if column in header]]))
    chunks: list[pd.DataFrame] = []
    progress = StageProgress("Accepted selected-HKL scan", total=None, unit="chunks")
    rows_read = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.accepted, usecols=usecols, chunksize=DEFAULT_CHUNKSIZE), start=1):
        rows_read += int(len(chunk))
        work = actionmod.normalize_key_columns(chunk)
        work = filter_to_hkls(work, selected_set)
        if not work.empty:
            if include_intensity:
                assert intensity_column is not None
                work["observation_intensity"] = pd.to_numeric(work[intensity_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                work["partiality"] = pd.to_numeric(work["partiality"], errors="coerce").replace([np.inf, -np.inf], np.nan) if "partiality" in work.columns else np.nan
                work["I_unmerged"] = pd.to_numeric(work["I_unmerged"], errors="coerce").replace([np.inf, -np.inf], np.nan) if "I_unmerged" in work.columns else np.nan
                work["sigma"] = pd.to_numeric(work[sigma_column], errors="coerce").replace([np.inf, -np.inf], np.nan) if sigma_column is not None else np.nan
                for column in scale_columns:
                    work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                chunks.append(work.loc[:, [*KEY_COLUMNS, "observation_intensity", "partiality", "I_unmerged", "sigma", *scale_columns]].copy())
            else:
                chunks.append(work.loc[:, KEY_COLUMNS].copy())
        progress.update(chunk_index, force=chunk_index == 1)
    progress.finish(chunk_index if 'chunk_index' in locals() else 0)
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=[*KEY_COLUMNS, "observation_intensity", "partiality", "I_unmerged", "sigma", *scale_columns] if include_intensity else KEY_COLUMNS)
    duplicated = table.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        duplicate_keys = int(table.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0])
        raise SystemExit(f"Accepted table contains duplicate exact observation keys for selected HKLs: {duplicate_keys}")
    stats = {
        "accepted_rows_read": int(rows_read),
        "selected_accepted_observations": int(len(table)),
        "selected_signed_hkls_with_accepted_observations": int(table.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]) if len(table) else 0,
        "intensity_column": intensity_column,
        "scale_columns": scale_columns,
    }
    return table.reset_index(drop=True), scale_columns, stats


def load_v5_for_accepted(args: argparse.Namespace, accepted: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = read_header(args.v5_scores)
    required = [*KEY_COLUMNS, actionmod.DEFAULT_SCORE_COLUMN, actionmod.DEFAULT_COUPLING_COLUMN, actionmod.DEFAULT_TARGET_COLUMN, actionmod.DEFAULT_SG_COLUMN]
    actionmod.require_columns(header, required, "v5 score CSV")
    selected_set = selected_hkl_set(accepted)
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    progress = StageProgress("V5 selected-HKL exact-key join", total=None, unit="chunks")
    for chunk_index, chunk in enumerate(pd.read_csv(args.v5_scores, usecols=required, chunksize=DEFAULT_CHUNKSIZE), start=1):
        rows_read += int(len(chunk))
        work = actionmod.normalize_key_columns(chunk)
        work = filter_to_hkls(work, selected_set)
        if work.empty:
            progress.update(chunk_index, force=chunk_index == 1)
            continue
        for column in [actionmod.DEFAULT_SCORE_COLUMN, actionmod.DEFAULT_COUPLING_COLUMN, actionmod.DEFAULT_TARGET_COLUMN, actionmod.DEFAULT_SG_COLUMN]:
            work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=[actionmod.DEFAULT_SCORE_COLUMN, actionmod.DEFAULT_COUPLING_COLUMN, actionmod.DEFAULT_TARGET_COLUMN])
        matched = work.merge(accepted, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        if not matched.empty:
            chunks.append(matched)
        progress.update(chunk_index, force=chunk_index == 1)
    progress.finish(chunk_index if 'chunk_index' in locals() else 0)
    table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if table.empty:
        raise SystemExit("No accepted selected observations matched v5 scores")
    duplicated = table.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        duplicate_keys = int(table.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0])
        raise SystemExit(f"Accepted/v5 join contains duplicate exact observation keys: {duplicate_keys}")
    v5_args = SimpleNamespace(
        score_column=actionmod.DEFAULT_SCORE_COLUMN,
        coupling_column=actionmod.DEFAULT_COUPLING_COLUMN,
        target_column=actionmod.DEFAULT_TARGET_COLUMN,
        sg_column=actionmod.DEFAULT_SG_COLUMN,
    )
    table = actionmod.add_dexc_columns(table, v5_args)
    table = actionmod.add_excitation_deficit_raw(table)
    stats = {
        "v5_rows_read": int(rows_read),
        "accepted_v5_matched_rows": int(len(table)),
        "accepted_keys_without_v5_score": int(max(0, len(accepted) - table.loc[:, KEY_COLUMNS].drop_duplicates().shape[0])),
    }
    return table.reset_index(drop=True), stats


def hkl_counts(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=[*HKL_COLUMNS, "n_accepted"])
    return table.groupby(HKL_COLUMNS, sort=False).size().reset_index(name="n_accepted")


def add_counts_to_selected(selected: pd.DataFrame, accepted: pd.DataFrame) -> pd.DataFrame:
    counts = hkl_counts(accepted)
    out = selected.merge(counts, on=HKL_COLUMNS, how="left", validate="one_to_one")
    out["n_accepted"] = pd.to_numeric(out["n_accepted"], errors="coerce").fillna(0).astype(int)
    return out


def removal_count(n_accepted: int, fraction: float, min_remaining: int) -> int:
    return int(max(0, min(int(np.floor(float(fraction) * int(n_accepted))), int(n_accepted) - int(min_remaining))))


def remaining_distribution(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "min": int(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "max": int(clean.max()),
    }


def audit_outputs(args: argparse.Namespace, selected_orbits: pd.DataFrame, selected: pd.DataFrame, accepted: pd.DataFrame, selection_metadata: dict[str, Any]) -> None:
    selected = add_counts_to_selected(selected, accepted)
    selected.to_csv(args.out_dir / "selected_signed_hkls.csv", index=False)
    selected_orbits.to_csv(args.out_dir / "selected_orbits.csv", index=False)
    selected_observations = int(selected["n_accepted"].sum()) if len(selected) else 0
    shell_rows: list[dict[str, Any]] = []
    for shell_count in [10, 20]:
        label_column = f"resolution_shell_{shell_count}_label"
        index_column = f"resolution_shell_{shell_count}_index"
        if label_column not in selected.columns:
            continue
        for key, group in selected.groupby([index_column, label_column], sort=True, dropna=False):
            row = {
                "shell_count": shell_count,
                "shell_index": key[0],
                "shell_label": key[1],
                "selected_signed_hkl_count": int(len(group)),
                "selected_accepted_observation_count": int(group["n_accepted"].sum()),
            }
            for fraction in args.filter_fractions:
                row[f"predicted_remove_drop{percent_label(fraction)}"] = int(sum(removal_count(n, fraction, args.min_remaining_observations) for n in group["n_accepted"]))
            shell_rows.append(row)
    shell_audit = pd.DataFrame.from_records(shell_rows)
    shell_audit.to_csv(args.out_dir / "aggressive_filter_audit_by_shell.csv", index=False)
    fraction_rows = []
    for fraction in args.filter_fractions:
        remove_total = int(sum(removal_count(n, fraction, args.min_remaining_observations) for n in selected["n_accepted"]))
        remaining = selected["n_accepted"].astype(int) - [removal_count(n, fraction, args.min_remaining_observations) for n in selected["n_accepted"]]
        fraction_rows.append(
            {
                "fraction": float(fraction),
                "predicted_removed": remove_total,
                "removal_fraction_all_accepted_6732955": float(remove_total / DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR),
                "removal_fraction_selected_observations": float(remove_total / selected_observations) if selected_observations else 0.0,
                "remaining_observations_per_signed_hkl": remaining_distribution(pd.Series(remaining)),
            }
        )
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_orbit_count": int(len(selected_orbits)),
        "selected_signed_hkl_count": int(len(selected)),
        "selected_accepted_observation_count": selected_observations,
        "selected_counts_by_shell": shell_rows,
        "selected_resolution_range": {
            "min_resolution_angstrom": float(selected["resolution_angstrom"].min()) if len(selected) else None,
            "max_resolution_angstrom": float(selected["resolution_angstrom"].max()) if len(selected) else None,
        },
        "selected_low_resolution_1pct_count": int(to_bool_series(selected.get("extreme_low_resolution_1pct", pd.Series(dtype=bool))).sum()) if len(selected) else 0,
        "excluded_extreme_high_resolution_5pct_count": int(selection_metadata.get("excluded_extreme_high_resolution_5pct_count", 0)),
        "fractions": fraction_rows,
        "thresholds_relaxed_automatically": False,
        "scientific_constraints": selection_metadata.get("selection_rules", {}),
    }
    (args.out_dir / "aggressive_filter_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    log(f"Selected orbit count: {len(selected_orbits):,}")
    log(f"Selected signed-HKL count: {len(selected):,}")
    log(f"Selected accepted-observation count: {selected_observations:,}")
    if len(selected):
        log(f"Selected resolution range: {selected['resolution_angstrom'].min():.6g}..{selected['resolution_angstrom'].max():.6g} A")
    log(f"Selected low-resolution 1% count: {audit['selected_low_resolution_1pct_count']:,}")
    log(f"Excluded extreme-high-resolution 5% count: {audit['excluded_extreme_high_resolution_5pct_count']:,}")
    for fraction_row in fraction_rows:
        log(
            f"Predicted removals drop{percent_label(fraction_row['fraction'])}: {fraction_row['predicted_removed']:,}; "
            f"fraction_all={fraction_row['removal_fraction_all_accepted_6732955']:.6g}; "
            f"fraction_selected={fraction_row['removal_fraction_selected_observations']:.6g}; "
            f"remaining_distribution={json.dumps(fraction_row['remaining_observations_per_signed_hkl'], sort_keys=True)}"
        )
    if not shell_audit.empty:
        for shell_count in [10, 20]:
            sub = shell_audit.loc[shell_audit["shell_count"] == shell_count]
            log(f"Selected counts by {shell_count}-shell: " + ", ".join(f"{row.shell_label}={int(row.selected_signed_hkl_count)}" for row in sub.itertuples(index=False)))


def prediction_worker(task: tuple[int, dict[str, Any], pd.DataFrame, list[str]]) -> tuple[int, int, pd.DataFrame]:
    index, selected_row, group, scale_columns = task
    hkl = (int(selected_row["h"]), int(selected_row["k"]), int(selected_row["l"]))
    model_name = str(selected_row["best_model_name"])
    covariates = [actionmod.TARGET_EXCITATION, actionmod.PARTIALITY, *scale_columns]
    frame = actionmod.clean_actionability_frame(group, actionmod.cvmod.CV_RESPONSE_SOURCE_COLUMN, covariates)
    if len(frame) < 5:
        raise RuntimeError(f"Insufficient model frame rows for {hkl_label(*hkl)}: {len(frame)}")
    train, pred_frame, _, _ = actionmod.cvmod.add_train_normalized_response(frame, frame)
    baseline = actionmod.cvmod.fit_model(train, actionmod.cvmod.CV_RESPONSE_COLUMN, covariates)
    train_resid = pd.to_numeric(train[actionmod.cvmod.CV_RESPONSE_COLUMN], errors="coerce").to_numpy(dtype=float) - actionmod.cvmod.predict_model(baseline, train)
    fit = actionmod.fit_effect_model(train, train_resid, pred_frame, model_name)
    pred = pred_frame.loc[:, KEY_COLUMNS].copy()
    pred["predicted_distortion"] = np.asarray(fit["predicted_effect_test"], dtype=float)
    pred["abs_predicted_distortion"] = np.abs(pred["predicted_distortion"].to_numpy(dtype=float))
    pred["best_model_name"] = model_name
    pred["worker_pid"] = os.getpid()
    all_rows = group.loc[:, KEY_COLUMNS].copy()
    out = all_rows.merge(pred, on=KEY_COLUMNS, how="left", validate="one_to_one")
    out["predicted_distortion"] = pd.to_numeric(out["predicted_distortion"], errors="coerce").fillna(0.0)
    out["abs_predicted_distortion"] = pd.to_numeric(out["abs_predicted_distortion"], errors="coerce").fillna(0.0)
    out["best_model_name"] = out["best_model_name"].fillna(model_name)
    out["worker_pid"] = out["worker_pid"].fillna(os.getpid()).astype(int)
    return index, os.getpid(), out


def fit_predictions_parallel(args: argparse.Namespace, observations: pd.DataFrame, selected: pd.DataFrame, scale_columns: list[str]) -> tuple[pd.DataFrame, list[int]]:
    if selected.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "predicted_distortion", "abs_predicted_distortion", "best_model_name", "worker_pid"]), []
    actual_workers = min(int(args.workers), int(len(selected)))
    log(f"Requested worker count: {int(args.workers):,}")
    log(f"Actual worker count: {actual_workers:,}")
    tasks = []
    for index, row in selected.reset_index(drop=True).iterrows():
        hkl = (int(row.h), int(row.k), int(row.l))
        group = observations.loc[(observations["h"].astype(int) == hkl[0]) & (observations["k"].astype(int) == hkl[1]) & (observations["l"].astype(int) == hkl[2])].copy()
        tasks.append((int(index), row.to_dict(), group, list(scale_columns)))
    progress = StageProgress("Per-HKL model fitting and prediction", total=len(tasks), unit="HKLs")
    results: dict[int, pd.DataFrame] = {}
    pids: set[int] = set()
    set_worker_numeric_threads()
    if actual_workers > 1:
        with ProcessPoolExecutor(max_workers=actual_workers, initializer=worker_initializer) as executor:
            futures = [executor.submit(prediction_worker, task) for task in tasks]
            for future in as_completed(futures):
                index, pid, frame = future.result()
                results[index] = frame
                pids.add(int(pid))
                progress.advance()
    else:
        for task in tasks:
            index, pid, frame = prediction_worker(task)
            results[index] = frame
            pids.add(int(pid))
            progress.advance()
    progress.finish(len(tasks))
    log("Worker PIDs: " + ", ".join(str(pid) for pid in sorted(pids)))
    prediction = pd.concat([results[index] for index in sorted(results)], ignore_index=True) if results else pd.DataFrame()
    duplicated = prediction.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        duplicate_keys = int(prediction.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0])
        raise SystemExit(f"Prediction table contains duplicate exact observation keys: {duplicate_keys}")
    return prediction, sorted(pids)


def rank_predictions(predictions: pd.DataFrame, fraction: float, random_seed: int, min_remaining: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = predictions.copy().reset_index(drop=True)
    work["key_text"] = [key_to_text(tuple(row)) for row in work.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)]
    work["random_tie"] = [stable_u64(f"{random_seed}|{percent_label(fraction)}|{text}") for text in work["key_text"]]
    aggressive_rows = []
    random_rows = []
    for hkl, group in work.groupby(HKL_COLUMNS, sort=False):
        n_accepted = int(len(group))
        remove_n = removal_count(n_accepted, fraction, min_remaining)
        if remove_n <= 0:
            continue
        aggressive = group.sort_values(["abs_predicted_distortion", "source_filename", "event", "h", "k", "l"], ascending=[False, True, True, True, True, True], kind="mergesort").head(remove_n).copy()
        random = group.sort_values(["random_tie", "source_filename", "event", "h", "k", "l"], ascending=[True, True, True, True, True, True], kind="mergesort").head(remove_n).copy()
        aggressive["fraction"] = float(fraction)
        random["fraction"] = float(fraction)
        aggressive["remove_n_for_hkl"] = remove_n
        random["remove_n_for_hkl"] = remove_n
        aggressive_rows.append(aggressive)
        random_rows.append(random)
    columns = [*KEY_COLUMNS, "fraction", "predicted_distortion", "abs_predicted_distortion", "best_model_name", "remove_n_for_hkl"]
    aggressive_table = pd.concat(aggressive_rows, ignore_index=True) if aggressive_rows else pd.DataFrame(columns=columns)
    random_table = pd.concat(random_rows, ignore_index=True) if random_rows else pd.DataFrame(columns=columns)
    return aggressive_table.loc[:, columns], random_table.loc[:, columns]


def key_set(table: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
    return {
        build_key(source, event, int(h), int(k), int(l))
        for source, event, h, k, l in table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    }


def validate_removal_tables(removals: dict[str, pd.DataFrame], selected: pd.DataFrame) -> None:
    selected_keys = selected_hkl_set(selected)
    for variant, table in removals.items():
        duplicated = table.duplicated(KEY_COLUMNS, keep=False) if not table.empty else pd.Series(dtype=bool)
        if duplicated.any():
            raise SystemExit(f"{variant} has duplicate exact removal keys")
        removed_hkls = selected_hkl_set(table) if not table.empty else set()
        extra = removed_hkls - selected_keys
        if extra:
            raise SystemExit(f"{variant} would remove unselected signed HKLs: {sorted(extra)[:10]}")


def build_removal_outputs(args: argparse.Namespace, predictions: pd.DataFrame, selected: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    removals: dict[str, pd.DataFrame] = {}
    qc_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    selected_meta = selected.loc[:, [*HKL_COLUMNS, "resolution_shell_10_label", "resolution_shell_20_label", "resolution_angstrom"]].drop_duplicates(HKL_COLUMNS, keep="first")
    for fraction in args.filter_fractions:
        label = percent_label(fraction)
        aggressive, random = rank_predictions(predictions, fraction, args.random_seed, args.min_remaining_observations)
        removals[f"aggressive_drop{label}"] = aggressive
        removals[f"random_drop{label}"] = random
        aggressive.to_csv(args.out_dir / f"removal_keys_aggressive_drop{label}.csv", index=False)
        random.to_csv(args.out_dir / f"removal_keys_random_drop{label}.csv", index=False)
        for hkl, group in predictions.groupby(HKL_COLUMNS, sort=False):
            hkl_tuple = tuple(map(int, hkl))
            aggressive_count = int(((aggressive["h"] == hkl_tuple[0]) & (aggressive["k"] == hkl_tuple[1]) & (aggressive["l"] == hkl_tuple[2])).sum()) if not aggressive.empty else 0
            random_count = int(((random["h"] == hkl_tuple[0]) & (random["k"] == hkl_tuple[1]) & (random["l"] == hkl_tuple[2])).sum()) if not random.empty else 0
            n_accepted = int(len(group))
            qc_rows.append(
                {
                    "fraction": float(fraction),
                    "h": hkl_tuple[0],
                    "k": hkl_tuple[1],
                    "l": hkl_tuple[2],
                    "n_accepted_predictions": n_accepted,
                    "expected_remove_n": removal_count(n_accepted, fraction, args.min_remaining_observations),
                    "aggressive_removed": aggressive_count,
                    "random_removed": random_count,
                    "aggressive_random_counts_equal": aggressive_count == random_count,
                    "remaining_after_aggressive": n_accepted - aggressive_count,
                    "min_remaining_observations": int(args.min_remaining_observations),
                }
            )
        for variant_name, table in [(f"aggressive_drop{label}", aggressive), (f"random_drop{label}", random)]:
            if table.empty:
                continue
            enriched = table.merge(selected_meta, on=HKL_COLUMNS, how="left", validate="many_to_one")
            for shell_column in ["resolution_shell_10_label", "resolution_shell_20_label"]:
                for shell_label_value, shell_group in enriched.groupby(shell_column, sort=True, dropna=False):
                    shell_rows.append({"variant": variant_name, "fraction": float(fraction), "shell_column": shell_column, "shell_label": shell_label_value, "removed_observation_count": int(len(shell_group)), "signed_hkl_count": int(shell_group.loc[:, HKL_COLUMNS].drop_duplicates().shape[0])})
            for (source, event), frame_group in table.groupby(["source_filename", "event"], sort=False):
                frame_rows.append({"variant": variant_name, "fraction": float(fraction), "source_filename": source, "event": event, "removed_observation_count": int(len(frame_group))})
    validate_removal_tables(removals, selected)
    qc = pd.DataFrame.from_records(qc_rows)
    if not qc.empty and not qc["aggressive_random_counts_equal"].all():
        raise SystemExit("Aggressive/random per-HKL removal counts differ")
    return removals, qc, pd.DataFrame.from_records(shell_rows), pd.DataFrame.from_records(frame_rows)


def stream_output_path(out_dir: Path, variant: str, seed: int) -> Path:
    if variant.startswith("aggressive_drop"):
        label = variant.replace("aggressive_drop", "")
        return out_dir / f"aggressive_absdist_drop{label}.stream"
    if variant.startswith("random_drop"):
        label = variant.replace("random_drop", "")
        return out_dir / f"random_matched_drop{label}_seed{int(seed)}.stream"
    raise ValueError(f"Unknown variant: {variant}")


def write_stream_variants(input_stream: Path, out_dir: Path, removals: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    variant_keys = {variant: key_set(table) for variant, table in removals.items()}
    for variant, keys in variant_keys.items():
        if len(keys) != len(removals[variant]):
            raise SystemExit(f"{variant} has duplicate exact removal keys")
    variant_paths = {variant: stream_output_path(out_dir, variant, seed) for variant in removals}
    handles = {variant: path.open("w", encoding="utf-8") for variant, path in variant_paths.items()}
    found_counts = {variant: Counter() for variant in removals}
    stats = {
        variant: {"requested_removals": len(keys), "removed_observations": 0, "kept_observations": 0, "total_reflection_rows_seen": 0}
        for variant, keys in variant_keys.items()
    }
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    progress = StageProgress("Stream rewriting", total=None, unit="reflection rows")
    rows_seen = 0
    try:
        with input_stream.open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                line = raw_line.rstrip("\n")
                if "Begin chunk" in line:
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
                if match := STREAM_FILENAME_RE.match(line):
                    source_name = normalize_source(match.group(1))
                    event_name = normalize_event(match.group(2)) if match.group(2) is not None else ""
                    if in_crystal:
                        current_source = source_name
                        if event_name:
                            current_event = event_name
                    else:
                        chunk_source = source_name
                        if event_name:
                            chunk_event = event_name
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
                hkl = parse_reflection_hkl(line) if in_crystal and in_reflections else None
                if hkl is not None:
                    key = build_key(current_source, current_event, *hkl)
                    rows_seen += 1
                    for variant, handle in handles.items():
                        stats[variant]["total_reflection_rows_seen"] += 1
                        if key in variant_keys[variant]:
                            found_counts[variant][key] += 1
                            stats[variant]["removed_observations"] += 1
                        else:
                            handle.write(raw_line)
                            stats[variant]["kept_observations"] += 1
                    progress.update(rows_seen)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()
    progress.finish(rows_seen)
    qc_rows = []
    for variant, keys in variant_keys.items():
        missing = [key for key in keys if found_counts[variant][key] == 0]
        duplicated = [key for key in keys if found_counts[variant][key] > 1]
        if missing or duplicated:
            raise SystemExit(f"{variant}: requested stream removal keys missing={len(missing)}, duplicated={len(duplicated)}")
        row = dict(stats[variant])
        if row["removed_observations"] != row["requested_removals"]:
            raise SystemExit(f"{variant}: stream reflection-row difference {row['removed_observations']} != requested removals {row['requested_removals']}")
        row["variant"] = variant
        row["output_stream"] = str(variant_paths[variant])
        row["all_requested_keys_found_exactly_once"] = True
        row["stream_reflection_row_difference_equals_requested_removals"] = True
        qc_rows.append(row)
    return pd.DataFrame.from_records(qc_rows)


def write_readme(out_dir: Path) -> None:
    text = """# V5 Aggressive Orientation-Aware Filter POC Streams

This run selects broad v5 actionability orbits using numeric held-out/orbit/filtering evidence, not merged intensity/Fobs, candidate scores, cheap-screen flags, or actionability classifications.

Generation mode refits the selected existing model family per exact signed HKL using accepted observations only to rank observations by absolute predicted distortion. Random controls remove the same number of exact observations per signed HKL.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_partialator_script(out_dir: Path, fractions: list[float], seed: int) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", "# Fill in your Partialator/merge command for each generated stream."]
    for fraction in fractions:
        label = percent_label(fraction)
        lines.append(f"echo aggressive_absdist_drop{label}.stream")
        lines.append(f"echo random_matched_drop{label}_seed{int(seed)}.stream")
    path = out_dir / "run_partialator_comparison.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log("Selecting orbits and signed HKLs from broad numeric evidence")
    selected_orbits, selected_signed, selection_metadata = select_orbits_and_hkls(args)
    selected_orbits.to_csv(args.out_dir / "selected_orbits.csv", index=False)
    selected_signed.to_csv(args.out_dir / "selected_signed_hkls.csv", index=False)
    accepted, scale_columns, accepted_stats = load_accepted_selected(args, selected_signed, include_intensity=not args.audit_only)
    log(f"Selected orbit count: {len(selected_orbits):,}")
    log(f"Selected signed-HKL count: {len(selected_signed):,}")
    log(f"Selected accepted-observation count: {len(accepted):,}")
    audit_outputs(args, selected_orbits, selected_signed, accepted, selection_metadata)
    if args.audit_only:
        return 0

    observations, v5_stats = load_v5_for_accepted(args, accepted)
    predictions, worker_pids = fit_predictions_parallel(args, observations, selected_signed, scale_columns)
    predictions.to_csv(args.out_dir / "selected_observation_predictions.csv", index=False)
    removals, per_hkl_qc, shell_summary, frame_summary = build_removal_outputs(args, predictions, selected_signed)
    per_hkl_qc.to_csv(args.out_dir / "per_hkl_removal_qc.csv", index=False)
    shell_summary.to_csv(args.out_dir / "per_shell_removal_summary.csv", index=False)
    frame_summary.to_csv(args.out_dir / "per_frame_removal_summary.csv", index=False)
    stream_qc = write_stream_variants(args.input_stream, args.out_dir, removals, args.random_seed)
    stream_qc.to_csv(args.out_dir / "stream_rewrite_qc.csv", index=False)
    if not per_hkl_qc.empty and not per_hkl_qc["aggressive_random_counts_equal"].all():
        raise SystemExit("QC failed: aggressive/random removals differ per signed HKL")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "accepted_stats": accepted_stats,
        "v5_stats": v5_stats,
        "selection_metadata": selection_metadata,
        "requested_worker_count": int(args.workers),
        "actual_worker_count": int(min(int(args.workers), max(1, len(selected_signed)))) if len(selected_signed) else 0,
        "worker_pids": worker_pids,
        "scientific_constraints": {
            "uses_merged_intensity_or_Fobs_as_ground_truth": False,
            "uses_actionability_classification": False,
            "uses_candidate_score_or_cheap_screen_pass": False,
            "preserves_exact_signed_hkl": True,
            "canonicalizes_signed_observations": False,
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    write_readme(args.out_dir)
    write_partialator_script(args.out_dir, args.filter_fractions, args.random_seed)
    log(f"Outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())