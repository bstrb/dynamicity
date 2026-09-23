#!/usr/bin/env python3
"""Build count-balanced v5 crowding/target-excitation corner subset streams."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_v5_aggressive_filter_poc_streams as agmod  # noqa: E402


KEY_COLUMNS = agmod.KEY_COLUMNS
HKL_COLUMNS = agmod.HKL_COLUMNS
M_COLUMN = agmod.actionmod.DEFAULT_SCORE_COLUMN
EG_COLUMN = agmod.actionmod.DEFAULT_TARGET_COLUMN
DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR = agmod.DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR
DEFAULT_CHUNKSIZE = agmod.DEFAULT_CHUNKSIZE

CORNER_SPECS = {
    "lowM_lowEg": {
        "stream": "corner_lowM_lowEg.stream",
        "m_ascending": True,
        "eg_ascending": True,
    },
    "lowM_highEg": {
        "stream": "corner_lowM_highEg.stream",
        "m_ascending": True,
        "eg_ascending": False,
    },
    "highM_lowEg": {
        "stream": "corner_highM_lowEg.stream",
        "m_ascending": False,
        "eg_ascending": True,
    },
    "highM_highEg": {
        "stream": "corner_highM_highEg.stream",
        "m_ascending": False,
        "eg_ascending": False,
    },
}
RANDOM_SUBSET = "random_corner_matched"
SUBSET_ORDER = [*CORNER_SPECS.keys(), RANDOM_SUBSET]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--v5-scores", type=Path, required=True)
    parser.add_argument("--input-stream", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.30)
    parser.add_argument("--min-balanced-per-hkl", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for label, path in [
        ("--manifest", args.manifest),
        ("--accepted", args.accepted),
        ("--v5-scores", args.v5_scores),
        ("--input-stream", args.input_stream),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if not (0.0 < float(args.tail_fraction) <= 0.5):
        raise SystemExit("--tail-fraction must satisfy 0 < value <= 0.5")
    if int(args.min_balanced_per_hkl) < 1:
        raise SystemExit("--min-balanced-per-hkl must be >= 1")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is nonempty; use --overwrite to replace files: {args.out_dir}")
    return args


def log(message: str) -> None:
    agmod.log(message)


def json_default(value: Any) -> Any:
    return agmod.json_default(value)


def require_columns_from_header(header: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def count_csv_data_rows(path: Path, label: str) -> int:
    total_bytes = int(path.stat().st_size)
    progress = agmod.StageProgress(f"Counting {label} rows", total=total_bytes, unit="bytes")
    newline_count = 0
    bytes_read = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
            progress.update(bytes_read, force=bytes_read == len(chunk))
    if total_bytes > 0 and last_byte != b"\n":
        newline_count += 1
    progress.finish(bytes_read)
    return max(0, int(newline_count) - 1)


def key_tuple_from_row(row: Any) -> tuple[str, str, int, int, int]:
    return agmod.build_key(row.source_filename, row.event, int(row.h), int(row.k), int(row.l))


def key_text_from_frame(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [agmod.key_to_text(tuple(row)) for row in frame.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)],
        index=frame.index,
        dtype=object,
    )


def key_set(table: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
    if table.empty:
        return set()
    return {
        agmod.build_key(source, event, int(h), int(k), int(l))
        for source, event, h, k, l in table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    }


def distribution(values: pd.Series) -> dict[str, Any]:
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


def read_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    progress = agmod.StageProgress("Reading manifest", total=1, unit="files")
    manifest = agmod.normalize_hkl_table(pd.read_csv(path, low_memory=False), "manifest")
    progress.update(1, force=True)
    progress.finish(1)
    agmod.require_columns(manifest, HKL_COLUMNS, "manifest")
    duplicate_mask = manifest.duplicated(HKL_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise SystemExit(f"Manifest contains duplicate exact signed HKLs: {int(duplicate_mask.sum())}")
    metadata = {
        "manifest_rows": int(len(manifest)),
        "manifest_signed_hkls": int(manifest.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]),
        "manifest_metadata_columns": [column for column in manifest.columns if column not in [*HKL_COLUMNS, "hkl"]],
    }
    return manifest.reset_index(drop=True), metadata


def load_accepted_manifest_keys(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    log("Validating accepted table columns")
    header = pd.read_csv(args.accepted, nrows=0).columns.tolist()
    require_columns_from_header(header, KEY_COLUMNS, "accepted table")
    manifest_hkls = agmod.selected_hkl_set(manifest)
    total_rows = count_csv_data_rows(args.accepted, "accepted")
    progress = agmod.StageProgress("Reading accepted keys and restricting to manifest HKLs", total=total_rows, unit="rows")
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_after_manifest = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.accepted, usecols=KEY_COLUMNS, chunksize=DEFAULT_CHUNKSIZE), start=1):
        rows_read += int(len(chunk))
        work = agmod.actionmod.normalize_key_columns(chunk)
        work = agmod.filter_to_hkls(work, manifest_hkls)
        rows_after_manifest += int(len(work))
        if not work.empty:
            chunks.append(work.loc[:, KEY_COLUMNS].copy())
        progress.update(rows_read, force=chunk_index == 1)
    progress.finish(rows_read)
    accepted = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=KEY_COLUMNS)
    duplicate_mask = accepted.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = int(accepted.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0])
        raise SystemExit(f"Accepted table contains duplicate exact observation keys after manifest restriction: {duplicate_keys}")
    stats = {
        "accepted_rows_read": int(rows_read),
        "accepted_rows_after_manifest_hkl_restriction": int(rows_after_manifest),
        "accepted_unique_exact_keys_after_manifest_hkl_restriction": int(len(accepted)),
    }
    return accepted.reset_index(drop=True), stats


def load_v5_joined_descriptors(args: argparse.Namespace, accepted: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    log("Validating v5 score table columns")
    header = pd.read_csv(args.v5_scores, nrows=0).columns.tolist()
    required = [*KEY_COLUMNS, M_COLUMN, EG_COLUMN]
    require_columns_from_header(header, required, "v5 score CSV")
    if accepted.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, M_COLUMN, EG_COLUMN]), {
            "v5_rows_read": 0,
            "accepted_v5_finite_descriptor_rows": 0,
            "accepted_keys_without_finite_v5_descriptors": 0,
        }
    manifest_hkls = agmod.selected_hkl_set(accepted)
    total_rows = count_csv_data_rows(args.v5_scores, "v5 score")
    progress = agmod.StageProgress("Joining v5 descriptors to exact accepted keys", total=total_rows, unit="rows")
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_after_manifest = 0
    rows_after_finite = 0
    rows_matched = 0
    accepted_payload = accepted.loc[:, KEY_COLUMNS].copy()
    for chunk_index, chunk in enumerate(pd.read_csv(args.v5_scores, usecols=required, chunksize=DEFAULT_CHUNKSIZE), start=1):
        rows_read += int(len(chunk))
        work = agmod.actionmod.normalize_key_columns(chunk)
        work = agmod.filter_to_hkls(work, manifest_hkls)
        rows_after_manifest += int(len(work))
        if not work.empty:
            work[M_COLUMN] = pd.to_numeric(work[M_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
            work[EG_COLUMN] = pd.to_numeric(work[EG_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
            work = work.dropna(subset=[M_COLUMN, EG_COLUMN])
            rows_after_finite += int(len(work))
            matched = work.merge(accepted_payload, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
            rows_matched += int(len(matched))
            if not matched.empty:
                chunks.append(matched.loc[:, [*KEY_COLUMNS, M_COLUMN, EG_COLUMN]].copy())
        progress.update(rows_read, force=chunk_index == 1)
    progress.finish(rows_read)
    joined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=[*KEY_COLUMNS, M_COLUMN, EG_COLUMN])
    duplicate_mask = joined.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_keys = int(joined.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0])
        raise SystemExit(f"Accepted/v5 descriptor join contains duplicate exact observation keys: {duplicate_keys}")
    stats = {
        "v5_rows_read": int(rows_read),
        "v5_rows_after_manifest_hkl_restriction": int(rows_after_manifest),
        "v5_rows_after_finite_descriptor_filter": int(rows_after_finite),
        "accepted_v5_finite_descriptor_rows": int(rows_matched),
        "accepted_keys_without_finite_v5_descriptors": int(max(0, len(accepted) - joined.loc[:, KEY_COLUMNS].drop_duplicates().shape[0])),
    }
    return joined.reset_index(drop=True), stats


def select_extreme_rows(group: pd.DataFrame, mask: pd.Series, subset_name: str, n_balanced: int) -> pd.DataFrame:
    spec = CORNER_SPECS[subset_name]
    subset = group.loc[mask].copy()
    if subset.empty or n_balanced <= 0:
        return subset.head(0)
    ordered = subset.sort_values(
        [M_COLUMN, EG_COLUMN, "exact_key_text"],
        ascending=[bool(spec["m_ascending"]), bool(spec["eg_ascending"]), True],
        kind="mergesort",
    ).head(int(n_balanced)).copy()
    ordered["subset"] = subset_name
    return ordered


def random_rows_for_group(group: pd.DataFrame, seed: int, n_balanced: int) -> pd.DataFrame:
    if group.empty or n_balanced <= 0:
        return group.head(0)
    work = group.copy()
    work["random_score"] = [agmod.stable_u64(f"{int(seed)}|corner-random|{text}") for text in work["exact_key_text"]]
    out = work.sort_values(["random_score", "exact_key_text"], ascending=[True, True], kind="mergesort").head(int(n_balanced)).copy()
    check = work.sort_values(["random_score", "exact_key_text"], ascending=[True, True], kind="mergesort").head(int(n_balanced))
    if list(out["exact_key_text"]) != list(check["exact_key_text"]):
        raise RuntimeError("Deterministic random selection changed within the same worker")
    out["subset"] = RANDOM_SUBSET
    return out


def corner_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame, float, int, int]) -> tuple[int, int, dict[str, Any], list[dict[str, Any]]]:
    index, hkl, group, tail_fraction, min_balanced, seed = task
    h, k, l = map(int, hkl)
    group = group.copy().reset_index(drop=True)
    qc: dict[str, Any] = {
        "h": h,
        "k": k,
        "l": l,
        "total_eligible_accepted_observations": int(len(group)),
        "q_M_low": np.nan,
        "q_M_high": np.nan,
        "q_Eg_low": np.nan,
        "q_Eg_high": np.nan,
        "raw_lowM_lowEg_count": 0,
        "raw_lowM_highEg_count": 0,
        "raw_highM_lowEg_count": 0,
        "raw_highM_highEg_count": 0,
        "n_balanced": 0,
        "included": False,
        "selected_lowM_lowEg_count": 0,
        "selected_lowM_highEg_count": 0,
        "selected_highM_lowEg_count": 0,
        "selected_highM_highEg_count": 0,
        "selected_random_count": 0,
        "worker_pid": os.getpid(),
    }
    if group.empty:
        return index, os.getpid(), qc, []

    group[M_COLUMN] = pd.to_numeric(group[M_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
    group[EG_COLUMN] = pd.to_numeric(group[EG_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
    group = group.dropna(subset=[M_COLUMN, EG_COLUMN]).reset_index(drop=True)
    qc["total_eligible_accepted_observations"] = int(len(group))
    if group.empty:
        return index, os.getpid(), qc, []

    group["exact_key_text"] = key_text_from_frame(group)
    m_values = group[M_COLUMN].to_numpy(dtype=float)
    eg_values = group[EG_COLUMN].to_numpy(dtype=float)
    q_m_low = float(np.quantile(m_values, float(tail_fraction)))
    q_m_high = float(np.quantile(m_values, 1.0 - float(tail_fraction)))
    q_eg_low = float(np.quantile(eg_values, float(tail_fraction)))
    q_eg_high = float(np.quantile(eg_values, 1.0 - float(tail_fraction)))
    qc.update({"q_M_low": q_m_low, "q_M_high": q_m_high, "q_Eg_low": q_eg_low, "q_Eg_high": q_eg_high})

    low_m = group[M_COLUMN] <= q_m_low
    high_m = (group[M_COLUMN] >= q_m_high) & (group[M_COLUMN] > 0.0)
    low_eg = group[EG_COLUMN] <= q_eg_low
    high_eg = group[EG_COLUMN] >= q_eg_high
    masks = {
        "lowM_lowEg": low_m & low_eg,
        "lowM_highEg": low_m & high_eg,
        "highM_lowEg": high_m & low_eg,
        "highM_highEg": high_m & high_eg,
    }
    raw_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    qc["raw_lowM_lowEg_count"] = raw_counts["lowM_lowEg"]
    qc["raw_lowM_highEg_count"] = raw_counts["lowM_highEg"]
    qc["raw_highM_lowEg_count"] = raw_counts["highM_lowEg"]
    qc["raw_highM_highEg_count"] = raw_counts["highM_highEg"]
    n_balanced = int(min(raw_counts.values())) if raw_counts else 0
    qc["n_balanced"] = n_balanced
    if n_balanced < int(min_balanced):
        return index, os.getpid(), qc, []

    selected_frames: list[pd.DataFrame] = []
    for subset_name, mask in masks.items():
        selected = select_extreme_rows(group, mask, subset_name, n_balanced)
        selected_frames.append(selected)
        qc[f"selected_{subset_name}_count"] = int(len(selected))
    random_selected = random_rows_for_group(group, int(seed), n_balanced)
    selected_frames.append(random_selected)
    qc["selected_random_count"] = int(len(random_selected))
    qc["included"] = True

    selected_all = pd.concat(selected_frames, ignore_index=True)
    selected_all["n_balanced"] = n_balanced
    selected_all["tail_fraction"] = float(tail_fraction)
    selected_all["q_M_low"] = q_m_low
    selected_all["q_M_high"] = q_m_high
    selected_all["q_Eg_low"] = q_eg_low
    selected_all["q_Eg_high"] = q_eg_high
    keep = ["subset", *KEY_COLUMNS, M_COLUMN, EG_COLUMN, "n_balanced", "tail_fraction", "q_M_low", "q_M_high", "q_Eg_low", "q_Eg_high", "exact_key_text"]
    return index, os.getpid(), qc, selected_all.loc[:, keep].to_dict("records")


def build_tasks(observations: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> list[tuple[int, tuple[int, int, int], pd.DataFrame, float, int, int]]:
    grouped = {tuple(map(int, hkl)): group.copy() for hkl, group in observations.groupby(HKL_COLUMNS, sort=False)}
    tasks = []
    for index, hkl in enumerate(manifest.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)):
        hkl_tuple = tuple(map(int, hkl))
        group = grouped.get(hkl_tuple, pd.DataFrame(columns=[*KEY_COLUMNS, M_COLUMN, EG_COLUMN]))
        tasks.append((index, hkl_tuple, group.loc[:, [*KEY_COLUMNS, M_COLUMN, EG_COLUMN]].copy(), float(args.tail_fraction), int(args.min_balanced_per_hkl), int(args.seed)))
    return tasks


def select_corners(observations: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    tasks = build_tasks(observations, manifest, args)
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(), []
    actual_workers = min(int(args.workers), len(tasks))
    log(f"Requested worker count: {int(args.workers):,}")
    log(f"Actual worker count: {actual_workers:,}")
    progress = agmod.StageProgress("Per-HKL corner selection and random matching", total=len(tasks), unit="HKLs")
    agmod.set_worker_numeric_threads()
    results: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    worker_pids: set[int] = set()
    if actual_workers > 1:
        with ProcessPoolExecutor(max_workers=actual_workers, initializer=agmod.worker_initializer) as executor:
            futures = [executor.submit(corner_worker, task) for task in tasks]
            for future in as_completed(futures):
                index, pid, qc, records = future.result()
                results[int(index)] = (qc, records)
                worker_pids.add(int(pid))
                progress.advance()
    else:
        for task in tasks:
            index, pid, qc, records = corner_worker(task)
            results[int(index)] = (qc, records)
            worker_pids.add(int(pid))
            progress.advance()
    progress.finish(len(tasks))
    log("Worker PIDs: " + ", ".join(str(pid) for pid in sorted(worker_pids)))
    qc = pd.DataFrame.from_records([results[index][0] for index in sorted(results)])
    selected_records = [record for index in sorted(results) for record in results[index][1]]
    selected = pd.DataFrame.from_records(selected_records)
    return qc, selected, sorted(worker_pids)


def validate_balanced_selection(per_hkl_qc: pd.DataFrame, selected: pd.DataFrame) -> None:
    progress = agmod.StageProgress("Random matching and count-balance validation", total=len(per_hkl_qc), unit="HKLs")
    expected_totals = {subset: 0 for subset in SUBSET_ORDER}
    selected_counts = selected.groupby(["subset", *HKL_COLUMNS], sort=False).size() if not selected.empty else pd.Series(dtype=int)
    for idx, row in enumerate(per_hkl_qc.itertuples(index=False), start=1):
        hkl = (int(row.h), int(row.k), int(row.l))
        n_balanced = int(row.n_balanced)
        if bool(row.included):
            counts = [
                int(row.selected_lowM_lowEg_count),
                int(row.selected_lowM_highEg_count),
                int(row.selected_highM_lowEg_count),
                int(row.selected_highM_highEg_count),
                int(row.selected_random_count),
            ]
            if counts != [n_balanced] * len(SUBSET_ORDER):
                raise SystemExit(f"Unequal per-HKL subset counts for signed HKL {hkl}: {counts} vs n_balanced={n_balanced}")
            for subset in SUBSET_ORDER:
                observed = int(selected_counts.get((subset, *hkl), 0))
                if observed != n_balanced:
                    raise SystemExit(f"Selected table count mismatch for {subset} {hkl}: {observed} != {n_balanced}")
                expected_totals[subset] += n_balanced
        progress.update(idx, force=idx == 1)
    progress.finish(len(per_hkl_qc))
    if len(set(expected_totals.values())) > 1:
        raise SystemExit(f"Unequal total counts between outputs: {expected_totals}")
    for subset in SUBSET_ORDER:
        subset_table = selected.loc[selected["subset"] == subset] if not selected.empty else selected
        if not subset_table.empty and subset_table.duplicated(KEY_COLUMNS, keep=False).any():
            raise SystemExit(f"Duplicate exact observation keys selected for subset {subset}")


def enrich_qc_with_manifest(per_hkl_qc: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in manifest.columns
        if column not in ["hkl"] and column not in per_hkl_qc.columns
    ]
    if not metadata_columns:
        return per_hkl_qc
    return per_hkl_qc.merge(manifest.loc[:, [*HKL_COLUMNS, *metadata_columns]], on=HKL_COLUMNS, how="left", validate="one_to_one")


def audit_summary(manifest: pd.DataFrame, per_hkl_qc: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace, stats: dict[str, Any], worker_pids: list[int]) -> dict[str, Any]:
    included = per_hkl_qc.loc[per_hkl_qc["included"].astype(bool)].copy()
    balanced_per_output = int(included["n_balanced"].sum()) if not included.empty else 0
    raw_counts = {
        "lowM_lowEg": int(per_hkl_qc["raw_lowM_lowEg_count"].sum()) if not per_hkl_qc.empty else 0,
        "lowM_highEg": int(per_hkl_qc["raw_lowM_highEg_count"].sum()) if not per_hkl_qc.empty else 0,
        "highM_lowEg": int(per_hkl_qc["raw_highM_lowEg_count"].sum()) if not per_hkl_qc.empty else 0,
        "highM_highEg": int(per_hkl_qc["raw_highM_highEg_count"].sum()) if not per_hkl_qc.empty else 0,
    }
    selected_counts = {subset: int((selected["subset"] == subset).sum()) if not selected.empty else 0 for subset in SUBSET_ORDER}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eligible_signed_hkls": int(len(manifest)),
        "included_signed_hkls": int(len(included)),
        "excluded_signed_hkls_with_zero_or_insufficient_balanced_count": int(len(manifest) - len(included)),
        "raw_corner_counts": raw_counts,
        "selected_counts_by_output": selected_counts,
        "balanced_observations_per_output": balanced_per_output,
        "fraction_of_all_6732955_accepted_observations": float(balanced_per_output / DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR),
        "n_balanced_per_included_hkl": distribution(included["n_balanced"] if not included.empty else pd.Series(dtype=float)),
        "tail_fraction": float(args.tail_fraction),
        "min_balanced_per_hkl": int(args.min_balanced_per_hkl),
        "seed": int(args.seed),
        "worker_pids": worker_pids,
        "input_stats": stats,
        "scientific_constraints": {
            "uses_intensities_or_response_variables_for_selection": False,
            "uses_merged_intensity_or_Fobs": False,
            "uses_prediction_errors_or_fitted_distortion": False,
            "preserves_exact_signed_hkl": True,
            "canonicalizes_observations_to_4mmm": False,
            "descriptor_M": M_COLUMN,
            "descriptor_Eg": EG_COLUMN,
        },
    }


def write_audit_outputs(out_dir: Path, per_hkl_qc: pd.DataFrame, selected: pd.DataFrame, audit: dict[str, Any], args: argparse.Namespace) -> None:
    progress = agmod.StageProgress("Writing selection tables and audit metadata", total=4, unit="files")
    per_hkl_qc.to_csv(out_dir / "per_hkl_corner_qc.csv", index=False)
    progress.advance()
    selected_out = selected.copy()
    if not selected_out.empty:
        selected_out = selected_out.sort_values(["subset", *HKL_COLUMNS, "source_filename", "event"], kind="mergesort")
    selected_out.to_csv(out_dir / "selected_corner_observations.csv", index=False)
    progress.advance()
    (out_dir / "corner_selection_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "audit": audit,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    progress.finish(4)


def stream_path_for_subset(out_dir: Path, subset: str, seed: int) -> Path:
    if subset in CORNER_SPECS:
        return out_dir / str(CORNER_SPECS[subset]["stream"])
    if subset == RANDOM_SUBSET:
        return out_dir / f"random_corner_matched_seed{int(seed)}.stream"
    raise ValueError(f"Unknown subset: {subset}")


def selected_key_sets(selected: pd.DataFrame) -> dict[str, set[tuple[str, str, int, int, int]]]:
    out: dict[str, set[tuple[str, str, int, int, int]]] = {}
    for subset in SUBSET_ORDER:
        subset_table = selected.loc[selected["subset"] == subset] if not selected.empty else pd.DataFrame(columns=KEY_COLUMNS)
        keys = key_set(subset_table)
        if len(keys) != len(subset_table):
            raise SystemExit(f"Duplicate exact observation keys requested for subset {subset}")
        out[subset] = keys
    return out


def write_subset_streams(input_stream: Path, out_dir: Path, selected: pd.DataFrame, seed: int) -> pd.DataFrame:
    key_sets = selected_key_sets(selected)
    output_paths = {subset: stream_path_for_subset(out_dir, subset, seed) for subset in SUBSET_ORDER}
    handles = {subset: path.open("w", encoding="utf-8") for subset, path in output_paths.items()}
    found_counts = {subset: Counter() for subset in SUBSET_ORDER}
    stats = {
        subset: {
            "subset": subset,
            "requested_observations": int(len(key_sets[subset])),
            "source_stream_reflection_rows_seen": 0,
            "output_reflection_rows_written": 0,
            "source_requested_keys_found": 0,
            "output_stream": str(output_paths[subset]),
        }
        for subset in SUBSET_ORDER
    }

    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    bytes_read = 0
    progress = agmod.StageProgress("Scanning source stream and rewriting subset streams", total=int(input_stream.stat().st_size), unit="bytes")
    try:
        with input_stream.open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                bytes_read += len(raw_line.encode("utf-8", errors="replace"))
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
                    progress.update(bytes_read)
                    continue
                if match := agmod.STREAM_IMAGE_RE.match(line):
                    if in_crystal:
                        current_source = agmod.normalize_source(match.group(1))
                    else:
                        chunk_source = agmod.normalize_source(match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                if match := agmod.STREAM_EVENT_RE.match(line):
                    if in_crystal:
                        current_event = agmod.normalize_event(match.group(1))
                    else:
                        chunk_event = agmod.normalize_event(match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                if match := agmod.STREAM_FILENAME_RE.match(line):
                    source_name = agmod.normalize_source(match.group(1))
                    event_name = agmod.normalize_event(match.group(2)) if match.group(2) is not None else ""
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
                    progress.update(bytes_read)
                    continue
                if "Begin crystal" in line:
                    current_source = chunk_source
                    current_event = chunk_event
                    in_crystal = True
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                if "End crystal" in line:
                    in_crystal = False
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                if in_crystal and "Reflections measured after indexing" in line:
                    in_reflections = True
                    for handle in handles.values():
                        handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                if in_reflections and line.startswith("End of reflections"):
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                hkl = agmod.parse_reflection_hkl(line) if in_crystal and in_reflections else None
                if hkl is not None:
                    key = agmod.build_key(current_source, current_event, *hkl)
                    for subset, handle in handles.items():
                        stats[subset]["source_stream_reflection_rows_seen"] += 1
                        if key in key_sets[subset]:
                            found_counts[subset][key] += 1
                            stats[subset]["source_requested_keys_found"] += 1
                            stats[subset]["output_reflection_rows_written"] += 1
                            handle.write(raw_line)
                    progress.update(bytes_read)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
                progress.update(bytes_read)
    finally:
        for handle in handles.values():
            handle.close()
    progress.finish(bytes_read)

    rows = []
    for subset in SUBSET_ORDER:
        missing = [key for key in key_sets[subset] if found_counts[subset][key] == 0]
        duplicated = [key for key in key_sets[subset] if found_counts[subset][key] > 1]
        if missing or duplicated:
            raise SystemExit(f"{subset}: requested source stream keys missing={len(missing)}, duplicated={len(duplicated)}")
        row = dict(stats[subset])
        if row["source_requested_keys_found"] != row["requested_observations"]:
            raise SystemExit(f"{subset}: source found {row['source_requested_keys_found']} keys but requested {row['requested_observations']}")
        row["source_requested_keys_found_exactly_once"] = True
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def scan_output_stream(path: Path, requested_keys: set[tuple[str, str, int, int, int]]) -> dict[str, Any]:
    found = Counter()
    unexpected = 0
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    reflection_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            line = raw_line.rstrip("\n")
            if "Begin chunk" in line:
                chunk_source = ""
                chunk_event = ""
                current_source = ""
                current_event = ""
                in_crystal = False
                in_reflections = False
                continue
            if match := agmod.STREAM_IMAGE_RE.match(line):
                if in_crystal:
                    current_source = agmod.normalize_source(match.group(1))
                else:
                    chunk_source = agmod.normalize_source(match.group(1))
                continue
            if match := agmod.STREAM_EVENT_RE.match(line):
                if in_crystal:
                    current_event = agmod.normalize_event(match.group(1))
                else:
                    chunk_event = agmod.normalize_event(match.group(1))
                continue
            if match := agmod.STREAM_FILENAME_RE.match(line):
                source_name = agmod.normalize_source(match.group(1))
                event_name = agmod.normalize_event(match.group(2)) if match.group(2) is not None else ""
                if in_crystal:
                    current_source = source_name
                    if event_name:
                        current_event = event_name
                else:
                    chunk_source = source_name
                    if event_name:
                        chunk_event = event_name
                continue
            if "Begin crystal" in line:
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
            hkl = agmod.parse_reflection_hkl(line) if in_crystal and in_reflections else None
            if hkl is not None:
                reflection_rows += 1
                key = agmod.build_key(current_source, current_event, *hkl)
                if key in requested_keys:
                    found[key] += 1
                else:
                    unexpected += 1
    missing = [key for key in requested_keys if found[key] == 0]
    duplicated = [key for key in requested_keys if found[key] > 1]
    return {
        "output_reflection_rows_verified": int(reflection_rows),
        "output_unexpected_reflection_rows": int(unexpected),
        "output_requested_keys_missing": int(len(missing)),
        "output_requested_keys_duplicated": int(len(duplicated)),
        "output_requested_keys_found_exactly_once": not missing and not duplicated and unexpected == 0,
    }


def validate_output_streams(out_dir: Path, selected: pd.DataFrame, seed: int, stream_qc: pd.DataFrame) -> pd.DataFrame:
    key_sets = selected_key_sets(selected)
    rows = []
    progress = agmod.StageProgress("Final exact-key validation of output streams", total=len(SUBSET_ORDER), unit="streams")
    for idx, subset in enumerate(SUBSET_ORDER, start=1):
        path = stream_path_for_subset(out_dir, subset, seed)
        result = scan_output_stream(path, key_sets[subset])
        if not result["output_requested_keys_found_exactly_once"]:
            raise SystemExit(f"Output stream exact-key validation failed for {subset}: {result}")
        rows.append({"subset": subset, **result})
        progress.update(idx, force=idx == 1)
    progress.finish(len(SUBSET_ORDER))
    validated = pd.DataFrame.from_records(rows)
    merged = stream_qc.merge(validated, on="subset", how="left", validate="one_to_one")
    if not (merged["requested_observations"].astype(int) == merged["output_reflection_rows_verified"].astype(int)).all():
        raise SystemExit("An output stream has a reflection-row count different from its requested exact-key count")
    if merged["requested_observations"].nunique(dropna=False) > 1:
        raise SystemExit("Unequal total counts between the five output streams")
    return merged


def write_readme(out_dir: Path) -> None:
    text = """# V5 Crowding / Target-Excitation Corner Streams

This experiment builds five count-balanced observation-subset streams from accepted observations whose exact signed HKL appears in the broad manifest.

Selection uses only two geometry descriptors from the v5 score table: `nonself_local_excitation_raw` and `target_excitation_Eg`. It does not use observation intensity, merged intensity, Fobs, prediction error, fitted distortion, or any response variable. Exact signed `h,k,l` and exact observation keys are preserved.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def print_audit_summary(audit: dict[str, Any]) -> None:
    log(f"Eligible signed HKLs: {audit['eligible_signed_hkls']:,}")
    log(f"Included signed HKLs: {audit['included_signed_hkls']:,}")
    log(f"Excluded signed HKLs with zero/insufficient balanced count: {audit['excluded_signed_hkls_with_zero_or_insufficient_balanced_count']:,}")
    log("Raw corner counts: " + ", ".join(f"{name}={count:,}" for name, count in audit["raw_corner_counts"].items()))
    log(f"Balanced observations per output: {audit['balanced_observations_per_output']:,}")
    log(f"Fraction of all 6,732,955 accepted observations: {audit['fraction_of_all_6732955_accepted_observations']:.6g}")
    log("n_balanced per included HKL: " + json.dumps(audit["n_balanced_per_included_hkl"], sort_keys=True))


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    agmod.set_worker_numeric_threads()

    manifest, manifest_stats = read_manifest(args.manifest)
    accepted, accepted_stats = load_accepted_manifest_keys(args, manifest)
    observations, v5_stats = load_v5_joined_descriptors(args, accepted)
    per_hkl_qc, selected, worker_pids = select_corners(observations, manifest, args)
    per_hkl_qc = enrich_qc_with_manifest(per_hkl_qc, manifest)
    validate_balanced_selection(per_hkl_qc, selected)
    stats = {"manifest": manifest_stats, "accepted": accepted_stats, "v5": v5_stats}
    audit = audit_summary(manifest, per_hkl_qc, selected, args, stats, worker_pids)
    write_audit_outputs(args.out_dir, per_hkl_qc, selected, audit, args)
    print_audit_summary(audit)
    if not args.audit_only:
        stream_qc = write_subset_streams(args.input_stream, args.out_dir, selected, int(args.seed))
        stream_qc = validate_output_streams(args.out_dir, selected, int(args.seed), stream_qc)
        stream_qc.to_csv(args.out_dir / "stream_rewrite_qc.csv", index=False)
    else:
        pd.DataFrame(columns=["subset", "requested_observations", "output_stream"]).to_csv(args.out_dir / "stream_rewrite_qc.csv", index=False)
    write_readme(args.out_dir)
    log(f"Outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())