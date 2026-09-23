#!/usr/bin/env python3
"""Build high-Eg excitation-matched crowding filter streams.

This is a narrow production script for testing whether reciprocal-space
crowding is harmful among similarly well-excited observations.  It uses only
exact signed observation keys, target excitation Eg as a matching variable, and
nonself_local_excitation_raw as the tested crowding variable.
"""

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

BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
for _thread_env_name in BLAS_THREAD_ENV_VARS:
    os.environ.setdefault(_thread_env_name, "1")

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_v5_aggressive_filter_poc_streams as agmod  # noqa: E402
import build_v5_crowding_target_excitation_corner_streams as cornermod  # noqa: E402


KEY_COLUMNS = agmod.KEY_COLUMNS
HKL_COLUMNS = agmod.HKL_COLUMNS
M_COLUMN = cornermod.M_COLUMN
EG_COLUMN = cornermod.EG_COLUMN
ABS_SG_COLUMN = "abs_sg"
EG_CLIPPED_COLUMN = "Eg_clipped_for_abs_sg"
EG_CLIP_APPLIED_COLUMN = "Eg_clip_applied_for_abs_sg"
EXACT_KEY_TEXT_COLUMN = "exact_key_text"
DEFAULT_SG0 = 0.0013180204579645218
DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR = agmod.DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR
DEFAULT_CHUNKSIZE = agmod.DEFAULT_CHUNKSIZE
MODE_ORDER = {"crowding": 0, "random": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--v5-scores", type=Path, required=True)
    parser.add_argument("--input-stream", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--high-eg-fraction", type=float, default=0.30)
    parser.add_argument("--excitation-block-size", type=int, default=10)
    parser.add_argument("--min-final-block-size", type=int, default=5)
    parser.add_argument("--min-high-eg-observations", type=int, default=10)
    parser.add_argument("--min-crowding-range", type=float, default=0.0)
    parser.add_argument("--filter-fractions", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    parser.add_argument("--min-remaining-per-block", type=int, default=2)
    parser.add_argument("--sg0", type=float, default=DEFAULT_SG0)
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
    if args.out_dir.exists() and not args.out_dir.is_dir():
        raise SystemExit(f"--out-dir exists but is not a directory: {args.out_dir}")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory exists and is nonempty; use --overwrite to replace files: {args.out_dir}")
    if not (0.0 < float(args.high_eg_fraction) <= 1.0):
        raise SystemExit("--high-eg-fraction must satisfy 0 < value <= 1")
    if int(args.excitation_block_size) < 1:
        raise SystemExit("--excitation-block-size must be >= 1")
    if int(args.min_final_block_size) < 1:
        raise SystemExit("--min-final-block-size must be >= 1")
    if int(args.min_high_eg_observations) < 1:
        raise SystemExit("--min-high-eg-observations must be >= 1")
    if not np.isfinite(float(args.min_crowding_range)) or float(args.min_crowding_range) < 0.0:
        raise SystemExit("--min-crowding-range must be finite and >= 0")
    if int(args.min_remaining_per_block) < 0:
        raise SystemExit("--min-remaining-per-block must be >= 0")
    if not np.isfinite(float(args.sg0)) or float(args.sg0) <= 0.0:
        raise SystemExit("--sg0 must be finite and > 0")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")

    fractions: list[float] = []
    labels: set[int] = set()
    for value in args.filter_fractions:
        fraction = float(value)
        if not np.isfinite(fraction) or not (0.0 < fraction < 1.0):
            raise SystemExit("--filter-fractions values must satisfy 0 < fraction < 1")
        label = agmod.percent_label(fraction)
        if label in labels:
            raise SystemExit("--filter-fractions contain duplicate percent labels after rounding")
        labels.add(label)
        fractions.append(fraction)
    args.filter_fractions = sorted(fractions)
    return args


def log(message: str) -> None:
    agmod.log(message)


def json_default(value: Any) -> Any:
    return agmod.json_default(value)


def key_text_from_frame(frame: pd.DataFrame) -> pd.Series:
    return cornermod.key_text_from_frame(frame)


def key_set(table: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
    return cornermod.key_set(table)


def numeric_distribution(values: Any) -> dict[str, Any]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    return {
        "n": int(len(clean)),
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def min_median_max(prefix: str, values: pd.Series | np.ndarray) -> dict[str, float | None]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {f"{prefix}_min": None, f"{prefix}_median": None, f"{prefix}_max": None}
    return {
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_max": float(clean.max()),
    }


def read_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    return cornermod.read_manifest(path)


def load_accepted_manifest_keys(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    return cornermod.load_accepted_manifest_keys(args, manifest)


def load_v5_joined_eligible(args: argparse.Namespace, accepted: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    log("Validating v5 score table columns")
    header = pd.read_csv(args.v5_scores, nrows=0).columns.tolist()
    required = [*KEY_COLUMNS, M_COLUMN, EG_COLUMN]
    cornermod.require_columns_from_header(header, required, "v5 score CSV")
    if accepted.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, M_COLUMN, EG_COLUMN, ABS_SG_COLUMN, EG_CLIPPED_COLUMN, EG_CLIP_APPLIED_COLUMN, EXACT_KEY_TEXT_COLUMN]), {
            "v5_rows_read": 0,
            "v5_rows_after_manifest_hkl_restriction": 0,
            "v5_rows_after_finite_M_Eg_filter": 0,
            "v5_rows_after_Eg_domain_filter": 0,
            "eligible_accepted_v5_rows": 0,
            "accepted_keys_without_eligible_v5_descriptors": 0,
            "eg_clipped_for_abs_sg_count": 0,
        }

    selected_hkls = agmod.selected_hkl_set(accepted)
    total_rows = cornermod.count_csv_data_rows(args.v5_scores, "v5 score")
    progress = agmod.StageProgress("Joining V5 descriptors to exact accepted keys", total=total_rows, unit="rows")
    chunks: list[pd.DataFrame] = []
    accepted_payload = accepted.loc[:, KEY_COLUMNS].copy()
    rows_read = 0
    rows_after_hkl = 0
    rows_after_finite = 0
    rows_after_eg_domain = 0
    rows_matched = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.v5_scores, usecols=required, chunksize=DEFAULT_CHUNKSIZE), start=1):
        rows_read += int(len(chunk))
        work = agmod.actionmod.normalize_key_columns(chunk)
        work = agmod.filter_to_hkls(work, selected_hkls)
        rows_after_hkl += int(len(work))
        if not work.empty:
            work[M_COLUMN] = pd.to_numeric(work[M_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
            work[EG_COLUMN] = pd.to_numeric(work[EG_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = work.loc[work[M_COLUMN].notna() & work[EG_COLUMN].notna()].copy()
            rows_after_finite += int(len(finite))
            finite = finite.loc[(finite[EG_COLUMN] > 0.0) & (finite[EG_COLUMN] <= 1.0)].copy()
            rows_after_eg_domain += int(len(finite))
            if not finite.empty:
                matched = finite.merge(accepted_payload, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
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

    if not joined.empty:
        eg = pd.to_numeric(joined[EG_COLUMN], errors="coerce").to_numpy(dtype=float)
        clipped = np.clip(eg, np.finfo(float).tiny, 1.0)
        joined[EG_CLIPPED_COLUMN] = clipped
        joined[EG_CLIP_APPLIED_COLUMN] = clipped != eg
        joined[ABS_SG_COLUMN] = float(args.sg0) * np.sqrt(-np.log(clipped))
        if not np.isfinite(joined[ABS_SG_COLUMN].to_numpy(dtype=float)).all():
            raise SystemExit("Nonfinite abs_sg values after Eg clipping")
        joined[EXACT_KEY_TEXT_COLUMN] = key_text_from_frame(joined)
        joined = joined.sort_values([*HKL_COLUMNS, EXACT_KEY_TEXT_COLUMN], kind="mergesort").reset_index(drop=True)
    else:
        joined[EG_CLIPPED_COLUMN] = pd.Series(dtype=float)
        joined[EG_CLIP_APPLIED_COLUMN] = pd.Series(dtype=bool)
        joined[ABS_SG_COLUMN] = pd.Series(dtype=float)
        joined[EXACT_KEY_TEXT_COLUMN] = pd.Series(dtype=object)

    stats = {
        "v5_rows_read": int(rows_read),
        "v5_rows_after_manifest_hkl_restriction": int(rows_after_hkl),
        "v5_rows_after_finite_M_Eg_filter": int(rows_after_finite),
        "v5_rows_after_Eg_domain_filter_0_lt_Eg_le_1": int(rows_after_eg_domain),
        "eligible_accepted_v5_rows": int(rows_matched),
        "accepted_keys_without_eligible_v5_descriptors": int(max(0, len(accepted) - joined.loc[:, KEY_COLUMNS].drop_duplicates().shape[0])),
        "eg_clipped_for_abs_sg_count": int(joined[EG_CLIP_APPLIED_COLUMN].sum()) if not joined.empty else 0,
        "eg_clipping_rule": "np.clip(Eg, np.finfo(float).tiny, 1.0) for abs_sg only after filtering 0 < Eg <= 1",
    }
    return joined, stats


def split_excitation_blocks(high_pool: pd.DataFrame, block_size: int, min_final_size: int) -> list[pd.DataFrame]:
    ordered = high_pool.sort_values([ABS_SG_COLUMN, EXACT_KEY_TEXT_COLUMN], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    blocks = [ordered.iloc[start : start + int(block_size)].copy() for start in range(0, len(ordered), int(block_size))]
    if len(blocks) > 1 and len(blocks[-1]) < int(min_final_size):
        blocks[-2] = pd.concat([blocks[-2], blocks[-1]], ignore_index=True)
        blocks = blocks[:-1]
    if len(blocks) == 1 and len(blocks[0]) < int(min_final_size):
        return []
    return blocks


def removal_count_for_block(n_block: int, fraction: float, min_remaining: int) -> int:
    raw = int(np.floor(float(fraction) * int(n_block)))
    cap = int(n_block) - int(min_remaining)
    return int(min(raw, cap))


def block_random_order(block: pd.DataFrame, seed: int, hkl: tuple[int, int, int], block_id: int) -> pd.DataFrame:
    block_seed = agmod.stable_u64(f"{int(seed)}|{int(hkl[0])}|{int(hkl[1])}|{int(hkl[2])}|block{int(block_id)}")
    work = block.copy()
    work["_random_rank"] = [agmod.stable_u64(f"{block_seed}|{text}") for text in work[EXACT_KEY_TEXT_COLUMN]]
    ordered = work.sort_values(["_random_rank", EXACT_KEY_TEXT_COLUMN], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    check = work.sort_values(["_random_rank", EXACT_KEY_TEXT_COLUMN], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    if list(ordered[EXACT_KEY_TEXT_COLUMN]) != list(check[EXACT_KEY_TEXT_COLUMN]):
        raise RuntimeError(f"Nondeterministic random ordering detected for {hkl} block {block_id}")
    return ordered


def append_removal_records(
    records: list[dict[str, Any]],
    rows: pd.DataFrame,
    mode: str,
    fraction: float,
    hkl: tuple[int, int, int],
    block_id: int,
    block_size: int,
) -> None:
    label = agmod.percent_label(fraction)
    for rank, row in enumerate(rows.itertuples(index=False), start=1):
        payload = row._asdict()
        records.append(
            {
                "mode": mode,
                "fraction": float(fraction),
                "drop_label": f"drop{label}",
                "variant": f"{mode}_drop{label}",
                "h": int(hkl[0]),
                "k": int(hkl[1]),
                "l": int(hkl[2]),
                "block_id": int(block_id),
                "block_size": int(block_size),
                "selection_rank_in_block": int(rank),
                "source_filename": payload["source_filename"],
                "event": payload["event"],
                M_COLUMN: float(payload[M_COLUMN]),
                EG_COLUMN: float(payload[EG_COLUMN]),
                EG_CLIPPED_COLUMN: float(payload[EG_CLIPPED_COLUMN]),
                EG_CLIP_APPLIED_COLUMN: bool(payload[EG_CLIP_APPLIED_COLUMN]),
                ABS_SG_COLUMN: float(payload[ABS_SG_COLUMN]),
                EXACT_KEY_TEXT_COLUMN: payload[EXACT_KEY_TEXT_COLUMN],
            }
        )


def selection_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame, dict[str, Any]]) -> tuple[int, int, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    index, hkl, group, params = task
    hkl = tuple(map(int, hkl))
    fractions = [float(value) for value in params["filter_fractions"]]
    labels = [agmod.percent_label(value) for value in fractions]
    hkl_qc: dict[str, Any] = {
        "h": hkl[0],
        "k": hkl[1],
        "l": hkl[2],
        "eligible_accepted_observations": int(len(group)),
        "high_eg_pool_size": 0,
        "n_excitation_blocks": 0,
        "n_actionable_blocks": 0,
        "observations_in_actionable_blocks": 0,
        "n_skipped_no_contrast_blocks": 0,
        "n_skipped_no_contrast_observations": 0,
        "included": False,
        "excluded_reason": "",
        "worker_pid": os.getpid(),
    }
    for label in labels:
        hkl_qc[f"oriented_removal_count_drop{label}"] = 0
        hkl_qc[f"random_removal_count_drop{label}"] = 0

    if group.empty:
        hkl_qc["excluded_reason"] = "no_eligible_observations"
        return index, os.getpid(), hkl_qc, [], []

    group = group.copy().reset_index(drop=True)
    if group.duplicated(KEY_COLUMNS, keep=False).any():
        raise RuntimeError(f"Duplicate exact observation key within signed HKL {hkl}")
    n_high = int(np.floor(float(params["high_eg_fraction"]) * int(len(group))))
    hkl_qc["high_eg_pool_size"] = int(n_high)
    if n_high < int(params["min_high_eg_observations"]):
        hkl_qc["excluded_reason"] = "high_eg_pool_below_minimum"
        return index, os.getpid(), hkl_qc, [], []

    high_pool = group.sort_values([EG_COLUMN, EXACT_KEY_TEXT_COLUMN], ascending=[False, True], kind="mergesort").head(n_high).copy()
    blocks = split_excitation_blocks(high_pool, int(params["excitation_block_size"]), int(params["min_final_block_size"]))
    hkl_qc["n_excitation_blocks"] = int(len(blocks))
    if not blocks:
        hkl_qc["excluded_reason"] = "no_excitation_block_meets_minimum_size"
        return index, os.getpid(), hkl_qc, [], []

    hkl_qc["included"] = True
    block_rows: list[dict[str, Any]] = []
    removal_records: list[dict[str, Any]] = []
    for block_id, block in enumerate(blocks, start=1):
        block = block.copy().reset_index(drop=True)
        n_block = int(len(block))
        m_values = pd.to_numeric(block[M_COLUMN], errors="coerce").to_numpy(dtype=float)
        eg_values = pd.to_numeric(block[EG_COLUMN], errors="coerce").to_numpy(dtype=float)
        abs_sg_values = pd.to_numeric(block[ABS_SG_COLUMN], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(m_values).all() and np.isfinite(eg_values).all() and np.isfinite(abs_sg_values).all()):
            raise RuntimeError(f"Nonfinite selected M, Eg, or abs_sg in {hkl} block {block_id}")
        m_range = float(np.max(m_values) - np.min(m_values)) if n_block else 0.0
        actionable = bool(m_range > float(params["min_crowding_range"]))
        block_qc: dict[str, Any] = {
            "h": hkl[0],
            "k": hkl[1],
            "l": hkl[2],
            "block_id": int(block_id),
            "block_size": int(n_block),
            "high_eg_pool_size": int(n_high),
            "actionable": actionable,
            "skipped_reason": "" if actionable else "no_crowding_contrast",
            "M_range": float(m_range),
            "worker_pid": os.getpid(),
        }
        block_qc.update(min_median_max("Eg", block[EG_COLUMN]))
        block_qc.update(min_median_max("abs_sg", block[ABS_SG_COLUMN]))
        block_qc.update(min_median_max("M", block[M_COLUMN]))
        for label in labels:
            block_qc[f"removal_count_drop{label}"] = 0
            block_qc[f"oriented_count_drop{label}"] = 0
            block_qc[f"random_count_drop{label}"] = 0

        if not actionable:
            hkl_qc["n_skipped_no_contrast_blocks"] += 1
            hkl_qc["n_skipped_no_contrast_observations"] += n_block
            block_rows.append(block_qc)
            continue

        hkl_qc["n_actionable_blocks"] += 1
        hkl_qc["observations_in_actionable_blocks"] += n_block
        oriented_order = block.sort_values([M_COLUMN, ABS_SG_COLUMN, EXACT_KEY_TEXT_COLUMN], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
        random_order = block_random_order(block, int(params["seed"]), hkl, block_id)
        for fraction, label in zip(fractions, labels):
            n_remove = removal_count_for_block(n_block, fraction, int(params["min_remaining_per_block"]))
            if n_remove < 1:
                raise RuntimeError(
                    f"Actionable block {hkl} block {block_id} cannot remove at least one observation "
                    f"for fraction={fraction}; block_size={n_block}, min_remaining_per_block={params['min_remaining_per_block']}"
                )
            oriented_rows = oriented_order.head(n_remove)
            random_rows = random_order.head(n_remove)
            block_qc[f"removal_count_drop{label}"] = int(n_remove)
            block_qc[f"oriented_count_drop{label}"] = int(len(oriented_rows))
            block_qc[f"random_count_drop{label}"] = int(len(random_rows))
            hkl_qc[f"oriented_removal_count_drop{label}"] += int(len(oriented_rows))
            hkl_qc[f"random_removal_count_drop{label}"] += int(len(random_rows))
            append_removal_records(removal_records, oriented_rows, "crowding", fraction, hkl, block_id, n_block)
            append_removal_records(removal_records, random_rows, "random", fraction, hkl, block_id, n_block)
        block_rows.append(block_qc)

    return index, os.getpid(), hkl_qc, block_rows, removal_records


def build_selection_tasks(observations: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> list[tuple[int, tuple[int, int, int], pd.DataFrame, dict[str, Any]]]:
    columns = [*KEY_COLUMNS, M_COLUMN, EG_COLUMN, EG_CLIPPED_COLUMN, EG_CLIP_APPLIED_COLUMN, ABS_SG_COLUMN, EXACT_KEY_TEXT_COLUMN]
    grouped = {tuple(map(int, hkl)): group.loc[:, columns].copy() for hkl, group in observations.groupby(HKL_COLUMNS, sort=False)}
    params = {
        "high_eg_fraction": float(args.high_eg_fraction),
        "excitation_block_size": int(args.excitation_block_size),
        "min_final_block_size": int(args.min_final_block_size),
        "min_high_eg_observations": int(args.min_high_eg_observations),
        "min_crowding_range": float(args.min_crowding_range),
        "filter_fractions": [float(value) for value in args.filter_fractions],
        "min_remaining_per_block": int(args.min_remaining_per_block),
        "seed": int(args.seed),
    }
    tasks: list[tuple[int, tuple[int, int, int], pd.DataFrame, dict[str, Any]]] = []
    empty = pd.DataFrame(columns=columns)
    for index, hkl in enumerate(manifest.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)):
        hkl_tuple = tuple(map(int, hkl))
        tasks.append((int(index), hkl_tuple, grouped.get(hkl_tuple, empty).copy(), params))
    return tasks


def select_removals(observations: pd.DataFrame, manifest: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], int]:
    tasks = build_selection_tasks(observations, manifest, args)
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], 0
    actual_workers = min(int(args.workers), int(len(tasks)))
    log(f"Requested worker count: {int(args.workers):,}")
    log(f"Actual worker count: {actual_workers:,}")
    progress = agmod.StageProgress(
        "Per-HKL selection: constructing high-Eg pools, excitation blocks, oriented removals, and matched-random removals",
        total=len(tasks),
        unit="HKLs",
    )
    agmod.set_worker_numeric_threads()
    results: dict[int, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    worker_pids: set[int] = set()
    if actual_workers > 1:
        with ProcessPoolExecutor(max_workers=actual_workers, initializer=agmod.worker_initializer) as executor:
            futures = [executor.submit(selection_worker, task) for task in tasks]
            for future in as_completed(futures):
                index, pid, hkl_qc, block_rows, removal_rows = future.result()
                results[int(index)] = (hkl_qc, block_rows, removal_rows)
                worker_pids.add(int(pid))
                progress.advance()
    else:
        for task in tasks:
            index, pid, hkl_qc, block_rows, removal_rows = selection_worker(task)
            results[int(index)] = (hkl_qc, block_rows, removal_rows)
            worker_pids.add(int(pid))
            progress.advance()
    progress.finish(len(tasks))
    log("Worker PIDs: " + ", ".join(str(pid) for pid in sorted(worker_pids)))

    hkl_rows = [results[index][0] for index in sorted(results)]
    block_rows = [row for index in sorted(results) for row in results[index][1]]
    removal_rows = [row for index in sorted(results) for row in results[index][2]]
    per_hkl_qc = pd.DataFrame.from_records(hkl_rows)
    per_block_qc = pd.DataFrame.from_records(block_rows)
    selected = pd.DataFrame.from_records(removal_rows)
    return per_hkl_qc, per_block_qc, selected, sorted(worker_pids), actual_workers


def enrich_hkl_qc_with_manifest(per_hkl_qc: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [column for column in manifest.columns if column not in ["hkl"] and column not in per_hkl_qc.columns]
    if not metadata_columns:
        return per_hkl_qc
    return per_hkl_qc.merge(manifest.loc[:, [*HKL_COLUMNS, *metadata_columns]], on=HKL_COLUMNS, how="left", validate="one_to_one")


def empty_selected_removals_frame(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "mode",
            "fraction",
            "drop_label",
            "variant",
            *HKL_COLUMNS,
            "block_id",
            "block_size",
            "selection_rank_in_block",
            "source_filename",
            "event",
            M_COLUMN,
            EG_COLUMN,
            EG_CLIPPED_COLUMN,
            EG_CLIP_APPLIED_COLUMN,
            ABS_SG_COLUMN,
            EXACT_KEY_TEXT_COLUMN,
        ]
    )


def sorted_selected_removals(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    work = selected.copy()
    work["_mode_order"] = work["mode"].map(MODE_ORDER).fillna(99).astype(int)
    work["_drop_order"] = pd.to_numeric(work["fraction"], errors="coerce")
    work = work.sort_values(
        ["_mode_order", "_drop_order", *HKL_COLUMNS, "block_id", "selection_rank_in_block", EXACT_KEY_TEXT_COLUMN],
        kind="mergesort",
    ).drop(columns=["_mode_order", "_drop_order"])
    return work.reset_index(drop=True)


def validate_count_and_matching(per_hkl_qc: pd.DataFrame, per_block_qc: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> None:
    progress = agmod.StageProgress("Validating count equality, nesting, and Eg matching diagnostics", total=3, unit="checks")
    labels = [agmod.percent_label(value) for value in args.filter_fractions]
    if not per_block_qc.empty:
        for label in labels:
            actionable = per_block_qc.loc[per_block_qc["actionable"].astype(bool)].copy()
            if not actionable.empty:
                if not (actionable[f"oriented_count_drop{label}"].astype(int) == actionable[f"removal_count_drop{label}"].astype(int)).all():
                    raise SystemExit(f"Oriented count mismatch in at least one block for drop{label}")
                if not (actionable[f"random_count_drop{label}"].astype(int) == actionable[f"removal_count_drop{label}"].astype(int)).all():
                    raise SystemExit(f"Random count mismatch in at least one block for drop{label}")
    progress.advance()

    if not per_hkl_qc.empty:
        for label in labels:
            oriented = per_hkl_qc[f"oriented_removal_count_drop{label}"].astype(int)
            random = per_hkl_qc[f"random_removal_count_drop{label}"].astype(int)
            if not (oriented == random).all():
                raise SystemExit(f"Oriented/random per-HKL removal counts differ for drop{label}")
            if int(oriented.sum()) != int(random.sum()):
                raise SystemExit(f"Oriented/random total removal counts differ for drop{label}")
    progress.advance()

    if not selected.empty:
        for column in [M_COLUMN, EG_COLUMN, ABS_SG_COLUMN]:
            values = pd.to_numeric(selected[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.isna().any():
                raise SystemExit(f"Selected removals contain nonfinite {column}")
        for mode in ["crowding", "random"]:
            previous: set[tuple[str, str, int, int, int]] | None = None
            for fraction in sorted(args.filter_fractions):
                subset = selected.loc[(selected["mode"] == mode) & (np.isclose(selected["fraction"], float(fraction)))].copy()
                keys = key_set(subset) if not subset.empty else set()
                if len(keys) != len(subset):
                    raise SystemExit(f"Duplicate exact observation keys in {mode} drop{agmod.percent_label(fraction)}")
                if previous is not None and not previous.issubset(keys):
                    raise SystemExit(f"Non-nested removal sets for {mode} at drop{agmod.percent_label(fraction)}")
                previous = keys
    progress.advance()
    progress.finish(3)


def removal_tables_by_variant(selected: pd.DataFrame, fractions: list[float]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for fraction in fractions:
        label = agmod.percent_label(fraction)
        for mode in ["crowding", "random"]:
            table = selected.loc[(selected["mode"] == mode) & (np.isclose(selected["fraction"], float(fraction)))].copy() if not selected.empty else pd.DataFrame(columns=KEY_COLUMNS)
            if not table.empty:
                table = table.loc[:, [*KEY_COLUMNS, "mode", "fraction", "drop_label", "variant", "block_id", M_COLUMN, EG_COLUMN, ABS_SG_COLUMN, EXACT_KEY_TEXT_COLUMN]].copy()
            tables[f"{mode}_drop{label}"] = table
    return tables


def desired_stream_path(out_dir: Path, mode: str, label: int, seed: int) -> Path:
    if mode == "crowding":
        return out_dir / f"highEg_Egmatched_crowding_drop{label}.stream"
    if mode == "random":
        return out_dir / f"random_highEg_Egmatched_drop{label}_seed{int(seed)}.stream"
    raise ValueError(f"Unknown mode: {mode}")


def agmod_variant_name(mode: str, label: int) -> str:
    if mode == "crowding":
        return f"aggressive_drop{label}"
    if mode == "random":
        return f"random_drop{label}"
    raise ValueError(f"Unknown mode: {mode}")


def stream_scan_absence(path: Path, requested_keys: set[tuple[str, str, int, int, int]]) -> dict[str, Any]:
    found = Counter()
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
    requested_remaining = int(sum(found.values()))
    return {
        "output_reflection_rows_verified": int(reflection_rows),
        "requested_removal_keys_remaining_in_output": requested_remaining,
        "requested_removal_keys_absent_in_output": requested_remaining == 0,
    }


def write_full_filter_streams(input_stream: Path, out_dir: Path, selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    tables = removal_tables_by_variant(selected, list(args.filter_fractions))
    agmod_tables: dict[str, pd.DataFrame] = {}
    variant_map: dict[str, dict[str, Any]] = {}
    for fraction in args.filter_fractions:
        label = agmod.percent_label(fraction)
        for mode in ["crowding", "random"]:
            source_variant = f"{mode}_drop{label}"
            ag_variant = agmod_variant_name(mode, label)
            agmod_tables[ag_variant] = tables[source_variant]
            variant_map[ag_variant] = {
                "variant": source_variant,
                "mode": mode,
                "fraction": float(fraction),
                "drop_label": f"drop{label}",
                "desired_path": desired_stream_path(out_dir, mode, label, int(args.seed)),
            }
    stream_qc = agmod.write_stream_variants(input_stream, out_dir, agmod_tables, int(args.seed))
    rows: list[dict[str, Any]] = []
    progress = agmod.StageProgress("Final exact-key validation of output streams", total=len(stream_qc), unit="streams")
    for idx, row in enumerate(stream_qc.itertuples(index=False), start=1):
        old_path = Path(row.output_stream)
        mapped = variant_map[str(row.variant)]
        new_path = Path(mapped["desired_path"])
        if old_path != new_path:
            old_path.replace(new_path)
        requested_table = tables[str(mapped["variant"])]
        requested_keys = key_set(requested_table) if not requested_table.empty else set()
        scan = stream_scan_absence(new_path, requested_keys)
        expected_rows = int(row.total_reflection_rows_seen) - int(row.requested_removals)
        if scan["requested_removal_keys_remaining_in_output"] != 0:
            raise SystemExit(f"Requested removals remain in output stream {new_path}")
        if int(scan["output_reflection_rows_verified"]) != expected_rows:
            raise SystemExit(
                f"Output stream removed an unrequested key or has a row-count mismatch: {new_path}; "
                f"observed={scan['output_reflection_rows_verified']} expected={expected_rows}"
            )
        out_row = row._asdict()
        out_row.update({key: value for key, value in mapped.items() if key != "desired_path"})
        out_row["output_stream"] = str(new_path)
        out_row["output_reflection_rows_expected"] = int(expected_rows)
        out_row["nonrequested_reflections_preserved_by_count"] = True
        out_row.update(scan)
        rows.append(out_row)
        progress.update(idx, force=idx == 1)
    progress.finish(len(stream_qc))
    return pd.DataFrame.from_records(rows)


def selected_distribution(selected: pd.DataFrame, mode: str, fraction: float, column: str) -> dict[str, Any]:
    if selected.empty:
        return numeric_distribution([])
    subset = selected.loc[(selected["mode"] == mode) & (np.isclose(selected["fraction"], float(fraction))), column]
    return numeric_distribution(subset)


def matching_diagnostics(selected: pd.DataFrame, fractions: list[float]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for fraction in fractions:
        label = f"drop{agmod.percent_label(fraction)}"
        crowd_abs = selected_distribution(selected, "crowding", fraction, ABS_SG_COLUMN)
        random_abs = selected_distribution(selected, "random", fraction, ABS_SG_COLUMN)
        crowd_eg = selected_distribution(selected, "crowding", fraction, EG_COLUMN)
        random_eg = selected_distribution(selected, "random", fraction, EG_COLUMN)
        median_crowd = crowd_abs["median"]
        median_random = random_abs["median"]
        abs_diff = None if median_crowd is None or median_random is None else abs(float(median_crowd) - float(median_random))
        rel_diff = None
        if abs_diff is not None:
            denom = max(abs(float(median_random)), np.finfo(float).tiny)
            rel_diff = float(abs_diff / denom)
        rows[label] = {
            "crowding_abs_sg": crowd_abs,
            "random_abs_sg": random_abs,
            "crowding_Eg": crowd_eg,
            "random_Eg": random_eg,
            "absolute_difference_median_abs_sg": abs_diff,
            "relative_difference_median_abs_sg_vs_random": rel_diff,
        }
    return rows


def audit_summary(
    manifest: pd.DataFrame,
    per_hkl_qc: pd.DataFrame,
    per_block_qc: pd.DataFrame,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    input_stats: dict[str, Any],
    worker_pids: list[int],
    actual_workers: int,
) -> dict[str, Any]:
    labels = [agmod.percent_label(value) for value in args.filter_fractions]
    included = per_hkl_qc.loc[per_hkl_qc["included"].astype(bool)].copy() if not per_hkl_qc.empty else pd.DataFrame()
    eligible_hkls = per_hkl_qc.loc[per_hkl_qc["eligible_accepted_observations"].astype(int) > 0].copy() if not per_hkl_qc.empty else pd.DataFrame()
    excluded = per_hkl_qc.loc[~per_hkl_qc["included"].astype(bool)].copy() if not per_hkl_qc.empty else pd.DataFrame()
    excluded_reasons = excluded["excluded_reason"].fillna("").replace("", "none").value_counts().to_dict() if not excluded.empty else {}
    actionable = per_block_qc.loc[per_block_qc["actionable"].astype(bool)].copy() if not per_block_qc.empty else pd.DataFrame()
    skipped = per_block_qc.loc[~per_block_qc["actionable"].astype(bool)].copy() if not per_block_qc.empty else pd.DataFrame()

    removal_counts: dict[str, Any] = {}
    for fraction, label in zip(args.filter_fractions, labels):
        oriented_count = int((selected["mode"].eq("crowding") & np.isclose(selected["fraction"], float(fraction))).sum()) if not selected.empty else 0
        random_count = int((selected["mode"].eq("random") & np.isclose(selected["fraction"], float(fraction))).sum()) if not selected.empty else 0
        removal_counts[f"drop{label}"] = {
            "fraction": float(fraction),
            "oriented_removal_count": oriented_count,
            "random_removal_count": random_count,
            "counts_equal": oriented_count == random_count,
            "oriented_removal_fraction_all_6732955_accepted": float(oriented_count / DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR),
            "random_removal_fraction_all_6732955_accepted": float(random_count / DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR),
        }

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "broad_manifest_signed_hkls": int(len(manifest)),
        "eligible_signed_hkls": int(len(eligible_hkls)),
        "included_signed_hkls": int(len(included)),
        "excluded_signed_hkls": int(len(excluded)),
        "excluded_signed_hkls_by_reason": {str(key): int(value) for key, value in excluded_reasons.items()},
        "total_accepted_observations_in_input": int(input_stats["accepted"].get("accepted_rows_read", 0)),
        "accepted_observation_denominator_all_accepted": int(DEFAULT_ACCEPTED_OBSERVATION_DENOMINATOR),
        "eligible_accepted_observations_with_finite_M_Eg_and_valid_Eg": int(input_stats["v5"].get("eligible_accepted_v5_rows", 0)),
        "total_high_Eg_observations": int(per_hkl_qc["high_eg_pool_size"].sum()) if not per_hkl_qc.empty else 0,
        "included_high_Eg_observations": int(included["high_eg_pool_size"].sum()) if not included.empty else 0,
        "actionable_block_observations": int(actionable["block_size"].sum()) if not actionable.empty else 0,
        "skipped_no_contrast_blocks": int(len(skipped)),
        "skipped_no_contrast_observations": int(skipped["block_size"].sum()) if not skipped.empty else 0,
        "removal_counts": removal_counts,
        "per_hkl_count_distributions": {
            "eligible_accepted_observations": numeric_distribution(per_hkl_qc["eligible_accepted_observations"] if not per_hkl_qc.empty else []),
            "high_eg_pool_size": numeric_distribution(per_hkl_qc["high_eg_pool_size"] if not per_hkl_qc.empty else []),
            "n_excitation_blocks": numeric_distribution(per_hkl_qc["n_excitation_blocks"] if not per_hkl_qc.empty else []),
            "n_actionable_blocks": numeric_distribution(per_hkl_qc["n_actionable_blocks"] if not per_hkl_qc.empty else []),
            "observations_in_actionable_blocks": numeric_distribution(per_hkl_qc["observations_in_actionable_blocks"] if not per_hkl_qc.empty else []),
        },
        "per_block_count_distributions": {
            "block_size": numeric_distribution(per_block_qc["block_size"] if not per_block_qc.empty else []),
            "M_range": numeric_distribution(per_block_qc["M_range"] if not per_block_qc.empty else []),
            "actionable_block_size": numeric_distribution(actionable["block_size"] if not actionable.empty else []),
        },
        "Eg_abs_sg_matching_diagnostics": matching_diagnostics(selected, list(args.filter_fractions)),
        "eg_clipping": {
            "sg0": float(args.sg0),
            "eg_clipped_for_abs_sg_count": int(input_stats["v5"].get("eg_clipped_for_abs_sg_count", 0)),
            "clipping_rule": input_stats["v5"].get("eg_clipping_rule"),
        },
        "requested_worker_count": int(args.workers),
        "actual_worker_count": int(actual_workers),
        "worker_pids": [int(pid) for pid in worker_pids],
        "input_stats": input_stats,
        "selection_parameters": {
            "high_eg_fraction": float(args.high_eg_fraction),
            "excitation_block_size": int(args.excitation_block_size),
            "min_final_block_size": int(args.min_final_block_size),
            "min_high_eg_observations": int(args.min_high_eg_observations),
            "min_crowding_range": float(args.min_crowding_range),
            "filter_fractions": [float(value) for value in args.filter_fractions],
            "min_remaining_per_block": int(args.min_remaining_per_block),
            "seed": int(args.seed),
        },
        "scientific_constraints": {
            "descriptor_M": M_COLUMN,
            "matching_descriptor_Eg": EG_COLUMN,
            "uses_intensities_or_response_variables_for_selection": False,
            "uses_merged_intensity_or_Fobs": False,
            "uses_prediction_errors_or_fitted_distortion": False,
            "combines_Eg_and_M_into_one_score": False,
            "preserves_exact_signed_hkl": True,
            "canonicalizes_observations_to_4mmm": False,
            "exact_observation_key": "source_filename + normalized event + signed h,k,l",
        },
    }


def write_readme(out_dir: Path, args: argparse.Namespace) -> None:
    text = f"""# V5 High-Eg Excitation-Matched Crowding Filter Streams

This run tests crowding within similarly well-excited observations.  It first
selects a rank-based high-Eg pool per exact signed HKL, then forms contiguous
blocks in absolute excitation error (`abs_sg = sg0 * sqrt(-log(Eg))`) and removes
the highest `nonself_local_excitation_raw` observations within each actionable
block.  Matched-random controls remove the same number of observations from the
same signed-HKL excitation block.

Selection uses only `{M_COLUMN}` and `{EG_COLUMN}` from the v5 score table.  It
does not use measured intensity, merged intensity, Fobs, prediction errors, or
fitted distortion.  Full-stream outputs preserve all nonrequested observations
and all CrystFEL metadata.

Seed: {int(args.seed)}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_audit_outputs(
    out_dir: Path,
    per_hkl_qc: pd.DataFrame,
    per_block_qc: pd.DataFrame,
    selected: pd.DataFrame,
    audit: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    progress = agmod.StageProgress("Writing QC tables and audit metadata", total=5, unit="files")
    per_hkl_qc.to_csv(out_dir / "per_hkl_filter_qc.csv", index=False)
    progress.advance()
    per_block_qc.to_csv(out_dir / "per_excitation_block_qc.csv", index=False)
    progress.advance()
    sorted_selected_removals(selected).to_csv(out_dir / "selected_removal_observations.csv", index=False)
    progress.advance()
    (out_dir / "excitation_matched_filter_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "audit": audit,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    progress.finish(5)


def write_audit_only_stream_qc(out_dir: Path) -> None:
    columns = [
        "variant",
        "mode",
        "fraction",
        "drop_label",
        "requested_removals",
        "removed_observations",
        "kept_observations",
        "total_reflection_rows_seen",
        "output_stream",
    ]
    pd.DataFrame(columns=columns).to_csv(out_dir / "stream_rewrite_qc.csv", index=False)


def print_audit_summary(audit: dict[str, Any]) -> None:
    log(f"Eligible signed HKLs: {audit['eligible_signed_hkls']:,}")
    log(f"Included signed HKLs: {audit['included_signed_hkls']:,}")
    log(f"Excluded signed HKLs: {audit['excluded_signed_hkls']:,} ({audit['excluded_signed_hkls_by_reason']})")
    log(f"Total accepted observations in input: {audit['total_accepted_observations_in_input']:,}")
    log(f"Total high-Eg observations: {audit['total_high_Eg_observations']:,}")
    log(f"Actionable-block observations: {audit['actionable_block_observations']:,}")
    log(f"Skipped no-contrast blocks: {audit['skipped_no_contrast_blocks']:,}")
    for label, row in audit["removal_counts"].items():
        log(
            f"{label}: oriented={row['oriented_removal_count']:,}, random={row['random_removal_count']:,}, "
            f"fraction_all={row['oriented_removal_fraction_all_6732955_accepted']:.6g}"
        )
    for label, row in audit["Eg_abs_sg_matching_diagnostics"].items():
        log(
            f"{label}: median abs_sg oriented/random="
            f"{row['crowding_abs_sg']['median']}/{row['random_abs_sg']['median']}; "
            f"abs_diff={row['absolute_difference_median_abs_sg']}"
        )
    log("Worker PIDs: " + ", ".join(str(pid) for pid in audit["worker_pids"]))


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    agmod.set_worker_numeric_threads()

    manifest, manifest_stats = read_manifest(args.manifest)
    accepted, accepted_stats = load_accepted_manifest_keys(args, manifest)
    observations, v5_stats = load_v5_joined_eligible(args, accepted)
    per_hkl_qc, per_block_qc, selected, worker_pids, actual_workers = select_removals(observations, manifest, args)
    if selected.empty:
        selected = empty_selected_removals_frame(args)
    per_hkl_qc = enrich_hkl_qc_with_manifest(per_hkl_qc, manifest)
    validate_count_and_matching(per_hkl_qc, per_block_qc, selected, args)

    input_stats = {"manifest": manifest_stats, "accepted": accepted_stats, "v5": v5_stats}
    audit = audit_summary(manifest, per_hkl_qc, per_block_qc, selected, args, input_stats, worker_pids, actual_workers)
    write_audit_outputs(args.out_dir, per_hkl_qc, per_block_qc, selected, audit, args)
    print_audit_summary(audit)

    if args.audit_only:
        write_audit_only_stream_qc(args.out_dir)
    else:
        stream_qc = write_full_filter_streams(args.input_stream, args.out_dir, selected, args)
        stream_qc.to_csv(args.out_dir / "stream_rewrite_qc.csv", index=False)

    write_readme(args.out_dir, args)
    log(f"Outputs written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
