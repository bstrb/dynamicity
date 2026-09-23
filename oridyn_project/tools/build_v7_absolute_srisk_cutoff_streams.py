#!/usr/bin/env python3
"""Build V7 absolute S_risk cutoff streams and matched random controls.

This script reads an existing V7 geometric-risk SQLite cache and the source
CrystFEL stream.  It removes accepted observations above absolute raw S_risk
cutoffs inside the analysed resolution domain d >= 0.5 A, while preserving all
other source-stream rows.  It also builds deterministic globally matched random
controls with the same removal count as each cutoff stream.

It does not run Partialator, merging, refinement, or modify the V7 cache.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_v7_geometric_risk_cache as v7


DATASET_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_SOURCE_STREAM = DATASET_ROOT / "MFM300-VIII_cut_20-0_3.stream"
DEFAULT_V7_CACHE = (
    DATASET_ROOT
    / "oridyn_v7_geometric_risk_s0_0p002_rcut0p20_20260903"
    / "v7_geometric_risk_cache.sqlite"
)
DEFAULT_OUTPUT_DIR = (
    DATASET_ROOT
    / "oridyn_v7_absolute_Srisk_filter_streams_s0_0p002_sig0p05_rcut0p20_20260904"
)

DEFAULT_CUTOFFS = [0.050, 0.020, 0.010, 0.005]
DEFAULT_SEED = 20260904
DEFAULT_MIN_RETAINED_PER_HKL = 2
DEFAULT_MAX_REMOVED_FRAC_PER_HKL = 0.80
DEFAULT_D_MIN_ANALYSIS = 0.5
DEFAULT_CHUNK_ROWS = 250_000
DEFAULT_PROGRESS_EVERY = 250_000
THRESHOLD_EPSILON = 1.0e-300
SHELLS = [
    ("0.5-0.6", 0.5, 0.6),
    ("0.6-0.8", 0.6, 0.8),
    ("0.8-1.0", 0.8, 1.0),
    ("1.0-1.2", 1.0, 1.2),
    ("1.2-1.5", 1.2, 1.5),
    ("1.5-2.0", 1.5, 2.0),
    ("2.0-3.0", 2.0, 3.0),
    ("3.0-5.0", 3.0, 5.0),
    (">5.0", 5.0, math.inf),
]
Q_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 100]
Q_NAMES = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p97p5", "p99", "p99p5", "p99p9", "max"]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    stream_type: str
    cutoff: float
    stream_path: Path
    variant_dir: Path
    seed: int | None


class RunLogger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = log_path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


class PackedMask:
    def __init__(self, n_bits: int):
        self.n_bits = int(n_bits)
        self.array = np.zeros((self.n_bits + 7) // 8, dtype=np.uint8)

    def set_many(self, indexes: np.ndarray) -> None:
        indexes = np.asarray(indexes, dtype=np.int64)
        if indexes.size == 0:
            return
        if int(indexes.min()) < 0 or int(indexes.max()) >= self.n_bits:
            raise IndexError(f"mask index out of range: min={int(indexes.min())}, max={int(indexes.max())}, n_bits={self.n_bits}")
        byte_indexes = np.right_shift(indexes, 3)
        bit_values = np.left_shift(1, np.bitwise_and(indexes, 7)).astype(np.uint8, copy=False)
        np.bitwise_or.at(self.array, byte_indexes, bit_values)

    def get(self, index: int) -> bool:
        idx = int(index)
        if idx < 0 or idx >= self.n_bits:
            return False
        return bool(int(self.array[idx >> 3]) & (1 << (idx & 7)))

    def count(self) -> int:
        return int(np.unpackbits(self.array, bitorder="little")[: self.n_bits].sum())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--v7-cache", type=Path, default=DEFAULT_V7_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoffs", default=",".join(f"{x:.3f}" for x in DEFAULT_CUTOFFS), help="Comma-separated raw V7 S_risk cutoffs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-retained-per-hkl", type=int, default=DEFAULT_MIN_RETAINED_PER_HKL)
    parser.add_argument("--max-removed-frac-per-hkl", type=float, default=DEFAULT_MAX_REMOVED_FRAC_PER_HKL)
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN_ANALYSIS, help="Analysed resolution-domain minimum d-spacing in A.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Recorded for provenance; stream rewrite is sequential for order fidelity.")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.source_stream = args.source_stream.expanduser().resolve()
    args.v7_cache = args.v7_cache.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.source_stream.is_file():
        raise SystemExit(f"Source stream not found: {args.source_stream}")
    if not args.v7_cache.is_file():
        raise SystemExit(f"V7 cache not found: {args.v7_cache}")
    cutoffs = []
    for text in str(args.cutoffs).split(","):
        text = text.strip()
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit("--cutoffs values must be positive finite numbers")
        cutoffs.append(value)
    if not cutoffs:
        raise SystemExit("--cutoffs must include at least one cutoff")
    args.cutoff_values = sorted(set(cutoffs), reverse=True)
    if args.min_retained_per_hkl < 0:
        raise SystemExit("--min-retained-per-hkl must be >= 0")
    if not (0.0 < float(args.max_removed_frac_per_hkl) <= 1.0):
        raise SystemExit("--max-removed-frac-per-hkl must satisfy 0 < value <= 1")
    if not math.isfinite(args.d_min) or args.d_min <= 0:
        raise SystemExit("--d-min must be positive")
    args.chunk_rows = max(1, int(args.chunk_rows))
    args.progress_every = max(1, int(args.progress_every))
    args.workers = max(1, int(args.workers))
    return args


def cutoff_label(cutoff: float) -> str:
    return f"{cutoff:.3f}".replace(".", "p")


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def git_info() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return out
    out["available"] = rev.returncode == 0
    out["commit"] = rev.stdout.strip() if rev.returncode == 0 else None
    out["status_short"] = status.stdout.splitlines() if status.returncode == 0 else []
    return out


def source_index(source: str) -> str:
    match = re.search(r"_(\d+)\.h5$", str(source))
    return match.group(1) if match else ""


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


def open_cache(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def hkl_key(h: int, k: int, l: int) -> tuple[int, int, int]:
    return int(h), int(k), int(l)


def hkl_capacity(n_obs: int, max_removed_frac: float, min_retained: int) -> tuple[int, str]:
    by_fraction = int(math.floor(float(max_removed_frac) * int(n_obs)))
    by_min_retained = max(0, int(n_obs) - int(min_retained))
    cap = max(0, min(by_fraction, by_min_retained))
    if cap == by_fraction and by_fraction < by_min_retained:
        reason = "max_removed_fraction"
    elif cap == by_min_retained and by_min_retained < by_fraction:
        reason = "min_retained"
    elif cap == 0:
        reason = "zero_capacity"
    else:
        reason = "both_equal"
    return cap, reason


def parse_reciprocal_metrics(source_stream: Path, logger: RunLogger) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    metrics: dict[tuple[str, str], np.ndarray] = {}
    duplicate_keys = 0
    duplicate_metric_max_delta = 0.0
    crystals = 0
    for crystal in v7.iter_stream_crystals(source_stream):
        crystals += 1
        key = (v7.normalize_source(crystal.source_filename), v7.normalize_event(crystal.event))
        if key in metrics:
            duplicate_keys += 1
            duplicate_metric_max_delta = max(duplicate_metric_max_delta, float(np.max(np.abs(metrics[key] - crystal.reciprocal_matrix))))
        else:
            metrics[key] = np.asarray(crystal.reciprocal_matrix, dtype=np.float64)
        if crystals == 1 or crystals % 5_000 == 0:
            logger.log(f"parsed stream crystals={crystals:,}, unique source/events={len(metrics):,}")
    return metrics, {
        "stream_crystals_seen": int(crystals),
        "stream_unique_source_events": int(len(metrics)),
        "stream_duplicate_source_event_keys": int(duplicate_keys),
        "stream_duplicate_metric_max_abs_delta": float(duplicate_metric_max_delta),
    }


def structured_hkl(h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
    out = np.empty(len(h), dtype=[("h", np.int32), ("k", np.int32), ("l", np.int32)])
    out["h"] = h.astype(np.int32, copy=False)
    out["k"] = k.astype(np.int32, copy=False)
    out["l"] = l.astype(np.int32, copy=False)
    return out


def unique_hkl_count(h: np.ndarray, k: np.ndarray, l: np.ndarray) -> int:
    if len(h) == 0:
        return 0
    return int(np.unique(structured_hkl(h, k, l)).shape[0])


def qstats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {name: math.nan for name in Q_NAMES}
    qs = np.percentile(arr, Q_PERCENTILES)
    return {name: float(value) for name, value in zip(Q_NAMES, qs)}


def metric_quantile_rows(label: str, values_by_metric: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in values_by_metric.items():
        row = {"population": label, "metric": metric, "n": int(len(values))}
        row.update(qstats(values))
        rows.append(row)
    return rows


def shell_label_for_d(d_spacing: float) -> str:
    for label, lo, hi in SHELLS:
        if math.isinf(hi):
            if d_spacing >= lo:
                return label
        elif lo <= d_spacing < hi:
            return label
    return "outside"


def load_hkl_counts(conn: sqlite3.Connection, logger: RunLogger) -> dict[tuple[int, int, int], int]:
    logger.log("loading total accepted observation counts per signed HKL")
    counts: dict[tuple[int, int, int], int] = {}
    for h, k, l, n_obs in conn.execute("SELECT h,k,l,COUNT(*) FROM v7_score_cache GROUP BY h,k,l"):
        counts[hkl_key(h, k, l)] = int(n_obs)
    logger.log(f"loaded signed-HKL counts: {len(counts):,}")
    return counts


def load_domain_arrays(
    conn: sqlite3.Connection,
    stream_metrics: dict[tuple[str, str], np.ndarray],
    args: argparse.Namespace,
    logger: RunLogger,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], dict[str, Any]]:
    total_rows = int(conn.execute("SELECT COUNT(*) FROM v7_score_cache").fetchone()[0])
    source_event_to_id: dict[tuple[str, str], int] = {}
    source_event_labels: list[tuple[str, str]] = []
    reciprocal_by_id: list[np.ndarray | None] = []
    missing_metric_keys: set[tuple[str, str]] = set()

    parts: dict[str, list[np.ndarray]] = {
        "source_order": [],
        "source_event_id": [],
        "h": [],
        "k": [],
        "l": [],
        "E_g": [],
        "R_env": [],
        "S_risk": [],
        "d_spacing": [],
        "reciprocal_radius": [],
    }
    rows_seen = 0
    rows_included = 0
    rows_missing_metric = 0
    rows_invalid_radius = 0
    rows_outside_resolution = 0
    max_source_order = 0
    min_source_order = None

    logger.log("reading V7 cache and computing per-row d-spacing for analysed domain")
    cursor = conn.execute("SELECT source_filename,event,h,k,l,source_order,E_g,R_env,S_risk FROM v7_score_cache ORDER BY ordinal")
    start = time.monotonic()
    while True:
        rows = cursor.fetchmany(int(args.chunk_rows))
        if not rows:
            break
        n = len(rows)
        rows_seen += n
        sources = [v7.normalize_source(row[0]) for row in rows]
        events = [v7.normalize_event(row[1]) for row in rows]
        h = np.fromiter((int(row[2]) for row in rows), dtype=np.int32, count=n)
        k = np.fromiter((int(row[3]) for row in rows), dtype=np.int32, count=n)
        l = np.fromiter((int(row[4]) for row in rows), dtype=np.int32, count=n)
        source_order = np.fromiter((int(row[5]) for row in rows), dtype=np.int64, count=n)
        eg = np.fromiter((float(row[6]) for row in rows), dtype=np.float64, count=n)
        renv = np.fromiter((float(row[7]) for row in rows), dtype=np.float64, count=n)
        srisk = np.fromiter((float(row[8]) for row in rows), dtype=np.float64, count=n)
        max_source_order = max(max_source_order, int(source_order.max()))
        so_min = int(source_order.min())
        min_source_order = so_min if min_source_order is None else min(min_source_order, so_min)

        se_ids = np.empty(n, dtype=np.int32)
        for i, key in enumerate(zip(sources, events)):
            sid = source_event_to_id.get(key)
            if sid is None:
                sid = len(source_event_labels)
                source_event_to_id[key] = sid
                source_event_labels.append(key)
                reciprocal_by_id.append(stream_metrics.get(key))
                if reciprocal_by_id[-1] is None:
                    missing_metric_keys.add(key)
            se_ids[i] = sid

        radius = np.full(n, np.nan, dtype=np.float64)
        hkl = np.column_stack([h.astype(np.float64), k.astype(np.float64), l.astype(np.float64)])
        for sid in np.unique(se_ids):
            idx = np.flatnonzero(se_ids == sid)
            reciprocal = reciprocal_by_id[int(sid)]
            if reciprocal is None:
                rows_missing_metric += int(len(idx))
                continue
            gvec = hkl[idx] @ reciprocal.T
            radius[idx] = np.linalg.norm(gvec, axis=1)
        valid_radius = np.isfinite(radius) & (radius > 0.0)
        d_spacing = np.full(n, np.nan, dtype=np.float64)
        d_spacing[valid_radius] = 1.0 / radius[valid_radius]
        include = valid_radius & (d_spacing >= float(args.d_min)) & (radius <= (1.0 / float(args.d_min) + 1.0e-12))
        rows_invalid_radius += int(np.count_nonzero(~valid_radius))
        rows_outside_resolution += int(np.count_nonzero(valid_radius & ~include))

        if np.any(include):
            parts["source_order"].append(source_order[include].astype(np.int64, copy=True))
            parts["source_event_id"].append(se_ids[include].astype(np.int32, copy=True))
            parts["h"].append(h[include].astype(np.int32, copy=True))
            parts["k"].append(k[include].astype(np.int32, copy=True))
            parts["l"].append(l[include].astype(np.int32, copy=True))
            parts["E_g"].append(eg[include].astype(np.float64, copy=True))
            parts["R_env"].append(renv[include].astype(np.float64, copy=True))
            parts["S_risk"].append(srisk[include].astype(np.float64, copy=True))
            parts["d_spacing"].append(d_spacing[include].astype(np.float64, copy=True))
            parts["reciprocal_radius"].append(radius[include].astype(np.float64, copy=True))
            rows_included += int(np.count_nonzero(include))

        if rows_seen == n or rows_seen % 1_000_000 < n or rows_seen == total_rows:
            elapsed = max(time.monotonic() - start, 1.0e-9)
            rate = rows_seen / elapsed
            eta = (total_rows - rows_seen) / max(rate, 1.0e-9)
            logger.log(
                f"cache read rows={rows_seen:,}/{total_rows:,} ({100.0 * rows_seen / max(1, total_rows):.1f}%), "
                f"included={rows_included:,}, rate={rate:,.0f} rows/s, eta={eta / 60.0:.1f} min"
            )

    arrays = {name: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64) for name, chunks in parts.items()}
    for name in ("source_order",):
        arrays[name] = arrays[name].astype(np.int64, copy=False)
    for name in ("source_event_id", "h", "k", "l"):
        arrays[name] = arrays[name].astype(np.int32, copy=False)

    stats = {
        "cache_rows_read": int(rows_seen),
        "cache_total_rows": int(total_rows),
        "included_rows_d_ge_min": int(rows_included),
        "d_min_A": float(args.d_min),
        "rows_missing_stream_metric": int(rows_missing_metric),
        "missing_metric_key_count": int(len(missing_metric_keys)),
        "rows_invalid_radius": int(rows_invalid_radius),
        "rows_outside_resolution_domain": int(rows_outside_resolution),
        "min_source_order": int(min_source_order or 0),
        "max_source_order": int(max_source_order),
        "source_event_keys_in_cache": int(len(source_event_labels)),
    }
    logger.log(
        f"included domain rows={rows_included:,}; unique source/events={len(source_event_labels):,}; "
        f"source_order range={stats['min_source_order']:,}..{stats['max_source_order']:,}"
    )
    return arrays, source_event_labels, stats


def prepare_outputs(args: argparse.Namespace, variants: list[Variant]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_outputs = [
        args.output_dir / "v7_absolute_cutoff_stream_summary.tsv",
        args.output_dir / "v7_absolute_cutoff_random_control_summary.tsv",
        args.output_dir / "v7_absolute_cutoff_removed_hkl_summary.tsv",
        args.output_dir / "v7_absolute_cutoff_removed_source_event_summary.tsv",
        args.output_dir / "v7_absolute_cutoff_notes.md",
        args.output_dir / "v7_absolute_cutoff_partialator_commands.sh",
        args.output_dir / "v7_absolute_cutoff_run_metadata.json",
    ]
    variant_outputs = []
    for variant in variants:
        variant_outputs.extend(
            [
                variant.stream_path,
                variant.variant_dir / "removal_manifest.tsv",
                variant.variant_dir / "removal_summary.json",
                variant.variant_dir / "removed_score_quantiles.tsv",
                variant.variant_dir / "top_removed_hkl_counts.tsv",
                variant.variant_dir / "top_removed_source_event_counts.tsv",
                variant.variant_dir / "removed_l_distribution.tsv",
                variant.variant_dir / "removed_resolution_distribution.tsv",
            ]
        )
    existing = [path for path in [*root_outputs, *variant_outputs] if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(str(path) for path in existing[:50])
        extra = "" if len(existing) <= 50 else f"\n... and {len(existing) - 50} more"
        raise SystemExit(f"Refusing to overwrite existing outputs; pass --overwrite if intentional:\n{joined}{extra}")
    if args.overwrite:
        for variant in variants:
            if variant.variant_dir.exists():
                shutil.rmtree(variant.variant_dir)
        for path in root_outputs:
            if path.exists():
                path.unlink()
    for variant in variants:
        variant.variant_dir.mkdir(parents=True, exist_ok=True)


def build_variants(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    for cutoff in args.cutoff_values:
        label = cutoff_label(cutoff)
        high_id = f"v7_abs_srisk_gt_{label}"
        high_dir = args.output_dir / high_id
        variants.append(Variant(high_id, "v7_cutoff", cutoff, high_dir / f"{high_id}.stream", high_dir, None))
        random_id = f"random_matched_v7_abs_srisk_gt_{label}_seed{int(args.seed)}"
        random_dir = args.output_dir / random_id
        variants.append(Variant(random_id, "matched_random", cutoff, random_dir / f"{random_id}.stream", random_dir, int(args.seed)))
    return variants


def select_with_caps(
    order: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    l: np.ndarray,
    hkl_counts: dict[tuple[int, int, int], int],
    capacities: dict[tuple[int, int, int], tuple[int, str]],
    *,
    target_n: int | None,
) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    if target_n is not None and int(target_n) <= 0:
        return np.empty(0, dtype=np.int64), {}
    selected: list[int] = []
    removed_by_hkl: dict[tuple[int, int, int], int] = {}
    for idx_raw in order:
        idx = int(idx_raw)
        key = hkl_key(int(h[idx]), int(k[idx]), int(l[idx]))
        n_total = hkl_counts.get(key, 0)
        if n_total <= 0:
            continue
        cap, _reason = capacities[key]
        used = removed_by_hkl.get(key, 0)
        if used >= cap:
            continue
        selected.append(idx)
        removed_by_hkl[key] = used + 1
        if target_n is not None and len(selected) >= int(target_n):
            break
    if target_n is not None and len(selected) != int(target_n):
        raise SystemExit(f"Could not select requested random control count {target_n:,}; selected {len(selected):,}")
    return np.asarray(selected, dtype=np.int64), removed_by_hkl


def select_variants(
    arrays: dict[str, np.ndarray],
    hkl_counts: dict[tuple[int, int, int], int],
    args: argparse.Namespace,
    variants: list[Variant],
    logger: RunLogger,
) -> tuple[dict[str, np.ndarray], dict[str, dict[tuple[int, int, int], int]], dict[str, dict[tuple[int, int, int], int]], dict[str, Any]]:
    h = arrays["h"]
    k = arrays["k"]
    l = arrays["l"]
    srisk = arrays["S_risk"]
    source_order = arrays["source_order"]
    n_domain = len(srisk)
    capacities = {
        key: hkl_capacity(n_obs, float(args.max_removed_frac_per_hkl), int(args.min_retained_per_hkl))
        for key, n_obs in hkl_counts.items()
    }
    selected_by_variant: dict[str, np.ndarray] = {}
    removed_by_variant_hkl: dict[str, dict[tuple[int, int, int], int]] = {}
    candidates_by_variant_hkl: dict[str, dict[tuple[int, int, int], int]] = {}
    random_rng = np.random.default_rng(int(args.seed))
    random_order_master = random_rng.permutation(n_domain)

    for variant in variants:
        if variant.stream_type == "v7_cutoff":
            candidates = np.flatnonzero(srisk > float(variant.cutoff))
            order = candidates[np.lexsort((source_order[candidates], -srisk[candidates]))]
            candidate_hkls, candidate_counts = np.unique(structured_hkl(h[candidates], k[candidates], l[candidates]), return_counts=True)
            candidates_by_variant_hkl[variant.variant_id] = {
                hkl_key(int(rec["h"]), int(rec["k"]), int(rec["l"])): int(count)
                for rec, count in zip(candidate_hkls, candidate_counts)
            }
            selected, removed_by_hkl = select_with_caps(order, h, k, l, hkl_counts, capacities, target_n=None)
            selected_by_variant[variant.variant_id] = selected
            removed_by_variant_hkl[variant.variant_id] = removed_by_hkl
            logger.log(
                f"selected {variant.variant_id}: raw_candidates={len(candidates):,}, "
                f"removed_after_caps={len(selected):,}, affected_hkls={len(removed_by_hkl):,}"
            )
        else:
            mate_id = variant.variant_id.replace("random_matched_", "").replace(f"_seed{int(args.seed)}", "")
            target_n = len(selected_by_variant[mate_id])
            candidate_hkls, candidate_counts = np.unique(structured_hkl(h, k, l), return_counts=True)
            candidates_by_variant_hkl[variant.variant_id] = {
                hkl_key(int(rec["h"]), int(rec["k"]), int(rec["l"])): int(count)
                for rec, count in zip(candidate_hkls, candidate_counts)
            }
            selected, removed_by_hkl = select_with_caps(random_order_master, h, k, l, hkl_counts, capacities, target_n=target_n)
            selected_by_variant[variant.variant_id] = selected
            removed_by_variant_hkl[variant.variant_id] = removed_by_hkl
            logger.log(
                f"selected {variant.variant_id}: domain_candidates={n_domain:,}, "
                f"matched_removed={len(selected):,}, affected_hkls={len(removed_by_hkl):,}"
            )

    stats = {
        "capacity_hkl_count": int(len(capacities)),
        "total_capacity_accepted_population": int(sum(cap for cap, _reason in capacities.values())),
    }
    return selected_by_variant, removed_by_variant_hkl, candidates_by_variant_hkl, stats


def make_masks(
    variants: list[Variant],
    arrays: dict[str, np.ndarray],
    selected_by_variant: dict[str, np.ndarray],
    source_mask_bits: int,
    logger: RunLogger,
) -> tuple[dict[str, PackedMask], dict[int, str]]:
    masks: dict[str, PackedMask] = {}
    expected_key_by_source_order: dict[int, str] = {}
    h = arrays["h"]
    k = arrays["k"]
    l = arrays["l"]
    source_order = arrays["source_order"]
    source_event_id = arrays["source_event_id"]
    source_event_labels = arrays["_source_event_labels"]
    for variant in variants:
        idx = selected_by_variant[variant.variant_id]
        mask = PackedMask(source_mask_bits)
        orders = source_order[idx].astype(np.int64, copy=False)
        mask.set_many(orders)
        if mask.count() != len(idx):
            raise SystemExit(f"{variant.variant_id}: source_order mask has {mask.count():,} bits but selected {len(idx):,} rows")
        masks[variant.variant_id] = mask
        for row_idx in idx:
            so = int(source_order[int(row_idx)])
            src, ev = source_event_labels[int(source_event_id[int(row_idx)])]
            key = v7.key_to_text(src, ev, int(h[int(row_idx)]), int(k[int(row_idx)]), int(l[int(row_idx)]))
            previous = expected_key_by_source_order.get(so)
            if previous is not None and previous != key:
                raise SystemExit(f"source_order collision {so}: {previous} != {key}")
            expected_key_by_source_order[so] = key
        logger.log(f"built source-order mask for {variant.variant_id}: removals={mask.count():,}")
    return masks, expected_key_by_source_order


def selected_rows(
    arrays: dict[str, np.ndarray],
    source_event_labels: list[tuple[str, str]],
    selected_idx: np.ndarray,
    *,
    sort_by_source_order: bool = True,
) -> Iterator[dict[str, Any]]:
    order = selected_idx[np.argsort(arrays["source_order"][selected_idx], kind="mergesort")] if sort_by_source_order else selected_idx
    for n, idx_raw in enumerate(order, start=1):
        idx = int(idx_raw)
        source, event = source_event_labels[int(arrays["source_event_id"][idx])]
        h = int(arrays["h"][idx])
        k = int(arrays["k"][idx])
        l = int(arrays["l"][idx])
        yield {
            "removed_row_number": n,
            "source_index": source_index(source),
            "source_filename": source,
            "event": event,
            "h": h,
            "k": k,
            "l": l,
            "exact_key_text": v7.key_to_text(source, event, h, k, l),
            "source_order": int(arrays["source_order"][idx]),
            "E_g": float(arrays["E_g"][idx]),
            "R_env": float(arrays["R_env"][idx]),
            "S_risk": float(arrays["S_risk"][idx]),
            "d_spacing_A": float(arrays["d_spacing"][idx]),
            "reciprocal_radius_Ainv": float(arrays["reciprocal_radius"][idx]),
        }


def summarize_removed_hkls(
    removed_by_hkl: dict[tuple[int, int, int], int],
    candidate_by_hkl: dict[tuple[int, int, int], int],
    hkl_counts: dict[tuple[int, int, int], int],
    args: argparse.Namespace,
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    total_removed = sum(removed_by_hkl.values())
    items = sorted(removed_by_hkl.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        items = items[:top_n]
    for rank, (key, removed) in enumerate(items, start=1):
        n_obs = hkl_counts.get(key, 0)
        cap, reason = hkl_capacity(n_obs, float(args.max_removed_frac_per_hkl), int(args.min_retained_per_hkl))
        h, k, l = key
        rows.append(
            {
                "rank": rank,
                "h": h,
                "k": k,
                "l": l,
                "removed_observations": int(removed),
                "candidate_observations": int(candidate_by_hkl.get(key, 0)),
                "total_accepted_observations_for_hkl": int(n_obs),
                "retained_accepted_observations_for_hkl": int(n_obs - removed),
                "removed_fraction_for_hkl": float(removed / n_obs) if n_obs else math.nan,
                "capacity": int(cap),
                "capacity_reason": reason,
                "hit_capacity": int(removed >= cap > 0),
                "protected_by_min_retained": int(candidate_by_hkl.get(key, 0) > cap and reason == "min_retained"),
                "fraction_of_all_removed": float(removed / total_removed) if total_removed else math.nan,
            }
        )
    return rows


def summarize_source_events(
    arrays: dict[str, np.ndarray],
    source_event_labels: list[tuple[str, str]],
    selected_idx: np.ndarray,
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    source_event_id = arrays["source_event_id"][selected_idx]
    unique, counts = np.unique(source_event_id, return_counts=True)
    order = np.argsort(counts)[::-1]
    if top_n is not None:
        order = order[:top_n]
    total = int(len(selected_idx))
    rows = []
    for rank, pos in enumerate(order, start=1):
        sid = int(unique[int(pos)])
        count = int(counts[int(pos)])
        source, event = source_event_labels[sid]
        rows.append(
            {
                "rank": rank,
                "source_index": source_index(source),
                "source_filename": source,
                "event": event,
                "removed_observations": count,
                "fraction_of_all_removed": float(count / total) if total else math.nan,
            }
        )
    return rows


def summarize_l_distribution(arrays: dict[str, np.ndarray], selected_idx: np.ndarray) -> list[dict[str, Any]]:
    selected_l = arrays["l"][selected_idx]
    unique, counts = np.unique(selected_l, return_counts=True)
    total = int(len(selected_idx))
    rows = []
    for l_value, count in sorted(zip(unique, counts), key=lambda item: int(item[0])):
        rows.append({"l": int(l_value), "removed_observations": int(count), "fraction_of_all_removed": float(count / total) if total else math.nan})
    return rows


def summarize_resolution_distribution(arrays: dict[str, np.ndarray], selected_idx: np.ndarray) -> list[dict[str, Any]]:
    d = arrays["d_spacing"][selected_idx]
    total = int(len(selected_idx))
    rows = []
    for label, lo, hi in SHELLS:
        mask = (d >= lo) if math.isinf(hi) else ((d >= lo) & (d < hi))
        count = int(np.count_nonzero(mask))
        rows.append(
            {
                "resolution_shell_A": label,
                "d_min_inclusive_A": lo,
                "d_max_exclusive_A": "" if math.isinf(hi) else hi,
                "removed_observations": count,
                "fraction_of_all_removed": float(count / total) if total else math.nan,
            }
        )
    return rows


def write_variant_artifacts(
    variant: Variant,
    arrays: dict[str, np.ndarray],
    source_event_labels: list[tuple[str, str]],
    selected_idx: np.ndarray,
    removed_by_hkl: dict[tuple[int, int, int], int],
    candidate_by_hkl: dict[tuple[int, int, int], int],
    hkl_counts: dict[tuple[int, int, int], int],
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_fields = [
        "removed_row_number",
        "source_index",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "exact_key_text",
        "source_order",
        "E_g",
        "R_env",
        "S_risk",
        "d_spacing_A",
        "reciprocal_radius_Ainv",
    ]
    write_tsv(variant.variant_dir / "removal_manifest.tsv", selected_rows(arrays, source_event_labels, selected_idx), manifest_fields)

    selected_mask = np.zeros(len(arrays["S_risk"]), dtype=bool)
    selected_mask[selected_idx] = True
    retained_mask = ~selected_mask
    q_rows = []
    q_rows.extend(
        metric_quantile_rows(
            "removed",
            {
                "E_g": arrays["E_g"][selected_idx],
                "R_env": arrays["R_env"][selected_idx],
                "S_risk": arrays["S_risk"][selected_idx],
                "d_spacing_A": arrays["d_spacing"][selected_idx],
                "reciprocal_radius_Ainv": arrays["reciprocal_radius"][selected_idx],
            },
        )
    )
    q_rows.extend(
        metric_quantile_rows(
            "retained_analysed_domain",
            {
                "E_g": arrays["E_g"][retained_mask],
                "R_env": arrays["R_env"][retained_mask],
                "S_risk": arrays["S_risk"][retained_mask],
                "d_spacing_A": arrays["d_spacing"][retained_mask],
                "reciprocal_radius_Ainv": arrays["reciprocal_radius"][retained_mask],
            },
        )
    )
    write_tsv(variant.variant_dir / "removed_score_quantiles.tsv", q_rows, ["population", "metric", "n", *Q_NAMES])

    hkl_rows = summarize_removed_hkls(removed_by_hkl, candidate_by_hkl, hkl_counts, args, top_n=50)
    source_rows = summarize_source_events(arrays, source_event_labels, selected_idx, top_n=50)
    write_tsv(variant.variant_dir / "top_removed_hkl_counts.tsv", hkl_rows, ["rank", "h", "k", "l", "removed_observations", "candidate_observations", "total_accepted_observations_for_hkl", "retained_accepted_observations_for_hkl", "removed_fraction_for_hkl", "capacity", "capacity_reason", "hit_capacity", "protected_by_min_retained", "fraction_of_all_removed"])
    write_tsv(variant.variant_dir / "top_removed_source_event_counts.tsv", source_rows, ["rank", "source_index", "source_filename", "event", "removed_observations", "fraction_of_all_removed"])
    write_tsv(variant.variant_dir / "removed_l_distribution.tsv", summarize_l_distribution(arrays, selected_idx), ["l", "removed_observations", "fraction_of_all_removed"])
    write_tsv(variant.variant_dir / "removed_resolution_distribution.tsv", summarize_resolution_distribution(arrays, selected_idx), ["resolution_shell_A", "d_min_inclusive_A", "d_max_exclusive_A", "removed_observations", "fraction_of_all_removed"])
    write_json(variant.variant_dir / "removal_summary.json", summary)
    return hkl_rows[:20], source_rows[:20]


def make_summary(
    variant: Variant,
    arrays: dict[str, np.ndarray],
    selected_idx: np.ndarray,
    removed_by_hkl: dict[tuple[int, int, int], int],
    candidate_by_hkl: dict[tuple[int, int, int], int],
    hkl_counts: dict[tuple[int, int, int], int],
    args: argparse.Namespace,
    stream_qc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_removed = int(len(selected_idx))
    source_event_count = int(np.unique(arrays["source_event_id"][selected_idx]).shape[0]) if n_removed else 0
    l0_count = int(np.count_nonzero(arrays["l"][selected_idx] == 0)) if n_removed else 0
    source_event_id = arrays["source_event_id"][selected_idx]
    se_counts = np.unique(source_event_id, return_counts=True)[1] if n_removed else np.asarray([], dtype=np.int64)
    top10_source_event_removed = int(np.sum(np.sort(se_counts)[::-1][:10])) if n_removed else 0
    hkls_hitting_capacity = 0
    hkls_protected_by_min_retained = 0
    for key, removed in removed_by_hkl.items():
        n_obs = hkl_counts[key]
        cap, reason = hkl_capacity(n_obs, float(args.max_removed_frac_per_hkl), int(args.min_retained_per_hkl))
        candidate_count = candidate_by_hkl.get(key, 0)
        if removed >= cap > 0 and candidate_count >= cap:
            hkls_hitting_capacity += 1
        if candidate_count > cap and reason == "min_retained":
            hkls_protected_by_min_retained += 1
    summary = {
        "variant_id": variant.variant_id,
        "stream_type": variant.stream_type,
        "cutoff": float(variant.cutoff),
        "seed": variant.seed,
        "stream_path": str(variant.stream_path),
        "raw_candidate_count_before_safety_caps": int(sum(candidate_by_hkl.values())),
        "final_removed_observation_count": n_removed,
        "final_removed_fraction_of_all_accepted_observations": float(n_removed / max(1, int(args.total_cache_rows))),
        "final_removed_fraction_of_analysed_d_ge_0p5A_observations": float(n_removed / max(1, int(args.analysed_domain_rows))),
        "affected_signed_hkls": int(len(removed_by_hkl)),
        "affected_source_events": source_event_count,
        "hkls_hitting_max_removal_safety_cap": int(hkls_hitting_capacity),
        "hkls_protected_by_min_retained_rule": int(hkls_protected_by_min_retained),
        "l0_removed_observation_count": l0_count,
        "l0_fraction_among_removed_observations": float(l0_count / n_removed) if n_removed else math.nan,
        "top10_source_event_removed_observations": top10_source_event_removed,
        "top10_source_event_fraction_of_removals": float(top10_source_event_removed / n_removed) if n_removed else math.nan,
        "min_retained_per_hkl": int(args.min_retained_per_hkl),
        "max_removed_frac_per_hkl": float(args.max_removed_frac_per_hkl),
        "analysed_d_min_A": float(args.d_min),
        "selection_rule": "S_risk cutoff within d>=0.5A domain, then per-HKL caps" if variant.stream_type == "v7_cutoff" else "deterministic random from d>=0.5A domain, same total count as cutoff stream, then per-HKL caps",
    }
    if stream_qc:
        summary.update({f"stream_qc_{key}": value for key, value in stream_qc.items()})
    return summary


def rewrite_streams(
    source_stream: Path,
    variants: list[Variant],
    masks: dict[str, PackedMask],
    expected_key_by_source_order: dict[int, str],
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, dict[str, Any]]:
    logger.log(f"rewriting source stream once into {len(variants)} filtered streams")
    handles = {variant.variant_id: variant.stream_path.open("w", encoding="utf-8") for variant in variants}
    stats = {
        variant.variant_id: {
            "stream_removed_count": 0,
            "stream_kept_reflection_rows": 0,
            "stream_total_reflection_rows_seen": 0,
            "removed_keys_matched_cache": 0,
            "removed_key_mismatches": 0,
            "begin_chunk_count": 0,
            "end_chunk_count": 0,
            "begin_crystal_count": 0,
            "end_crystal_count": 0,
            "begin_reflection_table_count": 0,
            "end_reflection_table_count": 0,
            "stream_valid_by_structure": True,
        }
        for variant in variants
    }
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_chunk = False
    in_crystal = False
    in_reflections = False
    reflection_row = 0
    started = time.monotonic()
    variant_ids = [variant.variant_id for variant in variants]

    def write_all(raw_line: str) -> None:
        for handle in handles.values():
            handle.write(raw_line)

    try:
        with source_stream.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if "Begin chunk" in line:
                    in_chunk = True
                    chunk_source = ""
                    chunk_event = ""
                    current_source = ""
                    current_event = ""
                    in_crystal = False
                    in_reflections = False
                    for stat in stats.values():
                        stat["begin_chunk_count"] += 1
                    write_all(raw_line)
                    continue
                if "End chunk" in line:
                    in_chunk = False
                    in_crystal = False
                    in_reflections = False
                    for stat in stats.values():
                        stat["end_chunk_count"] += 1
                    write_all(raw_line)
                    continue
                if match := v7.IMAGE_RE.match(line):
                    if in_crystal:
                        current_source = v7.normalize_source(match.group(1))
                    else:
                        chunk_source = v7.normalize_source(match.group(1))
                    write_all(raw_line)
                    continue
                if match := v7.EVENT_RE.match(line):
                    if in_crystal:
                        current_event = v7.normalize_event(match.group(1))
                    else:
                        chunk_event = v7.normalize_event(match.group(1))
                    write_all(raw_line)
                    continue
                if match := v7.FILENAME_RE.match(line):
                    parsed_source = v7.normalize_source(match.group(1))
                    parsed_event = v7.normalize_event(match.group(2)) if match.group(2) is not None else ""
                    if in_crystal:
                        current_source = parsed_source
                        if parsed_event:
                            current_event = parsed_event
                    else:
                        chunk_source = parsed_source
                        if parsed_event:
                            chunk_event = parsed_event
                    write_all(raw_line)
                    continue
                if "Begin crystal" in line:
                    current_source = chunk_source
                    current_event = chunk_event
                    in_crystal = True
                    in_reflections = False
                    for stat in stats.values():
                        stat["begin_crystal_count"] += 1
                    write_all(raw_line)
                    continue
                if "End crystal" in line:
                    in_crystal = False
                    in_reflections = False
                    for stat in stats.values():
                        stat["end_crystal_count"] += 1
                    write_all(raw_line)
                    continue
                if in_crystal and "Reflections measured after indexing" in line:
                    in_reflections = True
                    for stat in stats.values():
                        stat["begin_reflection_table_count"] += 1
                    write_all(raw_line)
                    continue
                if in_reflections and line.startswith("End of reflections"):
                    in_reflections = False
                    for stat in stats.values():
                        stat["end_reflection_table_count"] += 1
                    write_all(raw_line)
                    continue
                hkl = parse_reflection_hkl(line) if in_chunk and in_crystal and in_reflections else None
                if hkl is None:
                    write_all(raw_line)
                    continue

                reflection_row += 1
                removed_any = False
                expected_key = expected_key_by_source_order.get(reflection_row)
                actual_key = None
                for variant_id in variant_ids:
                    stat = stats[variant_id]
                    stat["stream_total_reflection_rows_seen"] += 1
                    if masks[variant_id].get(reflection_row):
                        stat["stream_removed_count"] += 1
                        removed_any = True
                        if expected_key is not None:
                            if actual_key is None:
                                actual_key = v7.key_to_text(current_source, current_event, *hkl)
                            if actual_key == expected_key:
                                stat["removed_keys_matched_cache"] += 1
                            else:
                                stat["removed_key_mismatches"] += 1
                    else:
                        stat["stream_kept_reflection_rows"] += 1
                        handles[variant_id].write(raw_line)
                if removed_any and expected_key is None:
                    raise SystemExit(f"Internal error: source_order {reflection_row} removed but no expected cache key recorded")
                if reflection_row % int(args.progress_every) == 0:
                    elapsed = max(time.monotonic() - started, 1.0e-9)
                    rate = reflection_row / elapsed
                    logger.log(f"stream rewrite reflection_rows={reflection_row:,}, rate={rate:,.0f} rows/s")
    finally:
        for handle in handles.values():
            handle.close()

    for variant in variants:
        stat = stats[variant.variant_id]
        expected_removed = masks[variant.variant_id].count()
        if int(stat["stream_removed_count"]) != expected_removed:
            raise SystemExit(f"{variant.variant_id}: stream removed {stat['stream_removed_count']:,} but expected {expected_removed:,}")
        if int(stat["removed_key_mismatches"]) != 0:
            raise SystemExit(f"{variant.variant_id}: removed key mismatches {stat['removed_key_mismatches']:,}")
        if int(stat["removed_keys_matched_cache"]) != expected_removed:
            raise SystemExit(f"{variant.variant_id}: removed key match count {stat['removed_keys_matched_cache']:,} but expected {expected_removed:,}")
        valid = (
            stat["begin_chunk_count"] == stat["end_chunk_count"]
            and stat["begin_crystal_count"] == stat["end_crystal_count"]
            and stat["begin_reflection_table_count"] == stat["end_reflection_table_count"]
            and variant.stream_path.is_file()
            and variant.stream_path.stat().st_size > 0
        )
        stat["stream_valid_by_structure"] = bool(valid)
        if not valid:
            raise SystemExit(f"{variant.variant_id}: output stream failed structural validation counters")
        logger.log(
            f"validated stream {variant.variant_id}: removed={stat['stream_removed_count']:,}, "
            f"kept={stat['stream_kept_reflection_rows']:,}, path={variant.stream_path}"
        )
    return stats


def validate_selection(
    variants: list[Variant],
    arrays: dict[str, np.ndarray],
    selected_by_variant: dict[str, np.ndarray],
    removed_by_variant_hkl: dict[str, dict[tuple[int, int, int], int]],
    hkl_counts: dict[tuple[int, int, int], int],
    args: argparse.Namespace,
    logger: RunLogger,
) -> None:
    for variant in variants:
        selected = selected_by_variant[variant.variant_id]
        if selected.size:
            min_d = float(np.min(arrays["d_spacing"][selected]))
            if min_d < float(args.d_min) - 1.0e-12:
                raise SystemExit(f"{variant.variant_id}: selected row outside d-spacing domain: min_d={min_d}")
        for key, removed in removed_by_variant_hkl[variant.variant_id].items():
            n_obs = hkl_counts[key]
            cap, _reason = hkl_capacity(n_obs, float(args.max_removed_frac_per_hkl), int(args.min_retained_per_hkl))
            if removed > cap:
                raise SystemExit(f"{variant.variant_id}: HKL {key} removed {removed}, exceeds cap {cap}")
            if n_obs - removed < int(args.min_retained_per_hkl):
                raise SystemExit(f"{variant.variant_id}: HKL {key} retained {n_obs - removed}, below min retained")
    for cutoff in args.cutoff_values:
        label = cutoff_label(cutoff)
        high_id = f"v7_abs_srisk_gt_{label}"
        random_id = f"random_matched_v7_abs_srisk_gt_{label}_seed{int(args.seed)}"
        if len(selected_by_variant[high_id]) != len(selected_by_variant[random_id]):
            raise SystemExit(f"{random_id}: random control count does not match {high_id}")
    logger.log("selection validation passed: random counts match, no safety-cap violations, all selected rows are in d-spacing domain")


def command_quote(path: Path | str) -> str:
    text = str(path)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def partialator_command_for_stream(stream_path: Path, threads: int) -> str:
    out_dir = Path(str(stream_path).removesuffix(".stream") + "_partialator_results")
    return "\n".join(
        [
            f"mkdir -p {command_quote(out_dir)} {command_quote(out_dir / 'qc_stats')} {command_quote(out_dir / 'pr-logs')}",
            f"python3 /home/bubl3932/projects/dynamicity/merge/stream_to_cell.py --stream {command_quote(stream_path)} --outdir {command_quote(out_dir)}",
            " ".join(
                [
                    "partialator",
                    command_quote(stream_path),
                    "--model=offset",
                    f"-j {int(threads)}",
                    "-o",
                    command_quote(out_dir / "crystfel.hkl"),
                    "-y 4/mmm",
                    "--min-measurements=1",
                    "--push-res=inf",
                    "--iterations=10",
                    "--harvest-file",
                    command_quote(out_dir / "parameters.json"),
                    "--log-folder",
                    command_quote(out_dir / "pr-logs"),
                    "--polarisation=none",
                    "--max-adu=inf",
                    "--min-res=inf",
                    "--no-Bscale",
                    "--no-pr",
                    ">",
                    command_quote(out_dir / "partialator_stdout.log"),
                    "2>",
                    command_quote(out_dir / "partialator_stderr.log"),
                ]
            ),
        ]
    )


def write_partialator_commands(args: argparse.Namespace, variants: list[Variant]) -> str:
    path = args.output_dir / "v7_absolute_cutoff_partialator_commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated commands only. This script was not run by the V7 cutoff builder.",
        "",
    ]
    for variant in variants:
        lines.append(f"echo 'Merging {variant.variant_id}'")
        lines.append(partialator_command_for_stream(variant.stream_path, int(args.workers)))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def write_root_outputs(
    args: argparse.Namespace,
    variants: list[Variant],
    summaries: dict[str, dict[str, Any]],
    global_hkl_rows: list[dict[str, Any]],
    global_source_rows: list[dict[str, Any]],
    run_stats: dict[str, Any],
) -> None:
    summary_fields = [
        "variant_id",
        "stream_type",
        "cutoff",
        "seed",
        "stream_path",
        "raw_candidate_count_before_safety_caps",
        "final_removed_observation_count",
        "final_removed_fraction_of_all_accepted_observations",
        "final_removed_fraction_of_analysed_d_ge_0p5A_observations",
        "affected_signed_hkls",
        "affected_source_events",
        "hkls_hitting_max_removal_safety_cap",
        "hkls_protected_by_min_retained_rule",
        "l0_removed_observation_count",
        "l0_fraction_among_removed_observations",
        "top10_source_event_removed_observations",
        "top10_source_event_fraction_of_removals",
        "stream_qc_stream_removed_count",
        "stream_qc_stream_kept_reflection_rows",
        "stream_qc_stream_total_reflection_rows_seen",
        "stream_qc_stream_valid_by_structure",
    ]
    cutoff_rows = [summaries[v.variant_id] for v in variants if v.stream_type == "v7_cutoff"]
    random_rows = [summaries[v.variant_id] for v in variants if v.stream_type == "matched_random"]
    write_tsv(args.output_dir / "v7_absolute_cutoff_stream_summary.tsv", cutoff_rows, summary_fields)
    write_tsv(args.output_dir / "v7_absolute_cutoff_random_control_summary.tsv", random_rows, summary_fields)
    write_tsv(args.output_dir / "v7_absolute_cutoff_removed_hkl_summary.tsv", global_hkl_rows, ["variant_id", "stream_type", "cutoff", "rank", "h", "k", "l", "removed_observations", "candidate_observations", "total_accepted_observations_for_hkl", "retained_accepted_observations_for_hkl", "removed_fraction_for_hkl", "capacity", "capacity_reason", "hit_capacity", "protected_by_min_retained", "fraction_of_all_removed"])
    write_tsv(args.output_dir / "v7_absolute_cutoff_removed_source_event_summary.tsv", global_source_rows, ["variant_id", "stream_type", "cutoff", "rank", "source_index", "source_filename", "event", "removed_observations", "fraction_of_all_removed"])
    command_path = write_partialator_commands(args, variants)
    write_json(
        args.output_dir / "v7_absolute_cutoff_run_metadata.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "source_stream": str(args.source_stream),
            "v7_cache": str(args.v7_cache),
            "output_dir": str(args.output_dir),
            "cutoffs": [float(x) for x in args.cutoff_values],
            "seed": int(args.seed),
            "min_retained_per_hkl": int(args.min_retained_per_hkl),
            "max_removed_frac_per_hkl": float(args.max_removed_frac_per_hkl),
            "d_min_A": float(args.d_min),
            "workers_requested": int(args.workers),
            "resolution_method": "cache lacked d_spacing/radius columns; computed |g|=||B*[h,k,l]|| using source stream astar/bstar/cstar converted from nm^-1 to A^-1, then d=1/|g|",
            "run_stats": run_stats,
            "partialator_command_file": command_path,
            "git": git_info(),
        },
    )

    notes = [
        "# V7 Absolute S_risk Cutoff Streams",
        "",
        f"- Source stream: `{args.source_stream}`",
        f"- V7 cache: `{args.v7_cache}`",
        f"- Output directory: `{args.output_dir}`",
        f"- Resolution domain: rows with `d_spacing >= {float(args.d_min):.3g} A` only; outside-domain rows are retained unchanged.",
        "- d-spacing method: the V7 cache had no stored `d_spacing` or reciprocal-radius columns, so `|g|` was computed from each source/event stream reciprocal basis `astar/bstar/cstar` converted from `nm^-1` to `A^-1`; `d = 1 / |g|`.",
        f"- Safety caps: exact signed HKL, min retained per HKL `{int(args.min_retained_per_hkl)}`, max removed fraction per HKL `{float(args.max_removed_frac_per_hkl):.3g}`.",
        f"- Matched random controls: deterministic seed `{int(args.seed)}`, globally matched to the cutoff stream removal count inside the same analysed domain, with the same HKL safety caps.",
        "",
        "## Cutoff Streams",
        "| variant | cutoff | removed | fraction all accepted | fraction analysed domain | affected HKLs | affected source/events | l=0 fraction | top10 source/event fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cutoff_rows:
        notes.append(
            f"| `{row['variant_id']}` | > {float(row['cutoff']):.3f} | {int(row['final_removed_observation_count']):,} | "
            f"{100.0 * float(row['final_removed_fraction_of_all_accepted_observations']):.3f}% | "
            f"{100.0 * float(row['final_removed_fraction_of_analysed_d_ge_0p5A_observations']):.3f}% | "
            f"{int(row['affected_signed_hkls']):,} | {int(row['affected_source_events']):,} | "
            f"{100.0 * float(row['l0_fraction_among_removed_observations']):.3f}% | "
            f"{100.0 * float(row['top10_source_event_fraction_of_removals']):.3f}% |"
        )
    notes.extend(["", "## Random Controls"])
    notes.append("| variant | matched cutoff | removed | affected HKLs | affected source/events | l=0 fraction |")
    notes.append("|---|---:|---:|---:|---:|---:|")
    for row in random_rows:
        notes.append(
            f"| `{row['variant_id']}` | > {float(row['cutoff']):.3f} | {int(row['final_removed_observation_count']):,} | "
            f"{int(row['affected_signed_hkls']):,} | {int(row['affected_source_events']):,} | "
            f"{100.0 * float(row['l0_fraction_among_removed_observations']):.3f}% |"
        )
    notes.extend(
        [
            "",
            "## Validation",
            "- Every selected removal row came from the V7 cache and had `d_spacing >= 0.5 A`.",
            "- During stream rewrite, every removed source-order row was checked against the expected cache exact key.",
            "- Random controls remove exactly the same number of observations as their matched cutoff streams.",
            "- No signed HKL violates the min-retained or max-removed-fraction safety caps.",
            "- Output stream structure counters are balanced because the source stream is copied verbatim except for selected reflection rows.",
            "",
            "## Partialator Commands",
            f"- Command file: `{command_path}`",
            "- Commands were written but not run.",
            "",
            "## Stream Paths",
        ]
    )
    for variant in variants:
        notes.append(f'- `{variant.stream_path}`')
    (args.output_dir / "v7_absolute_cutoff_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = build_variants(args)
    prepare_outputs(args, variants)
    logger = RunLogger(args.output_dir / "v7_absolute_cutoff_run.log")
    started = time.monotonic()
    try:
        logger.log("V7 absolute S_risk cutoff stream build start")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"v7_cache={args.v7_cache}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"cutoffs={args.cutoff_values}; seed={args.seed}; workers={args.workers}")
        conn = open_cache(args.v7_cache)
        schema = {row["name"] for row in conn.execute("PRAGMA table_info(v7_score_cache)")}
        for column in ["source_filename", "event", "h", "k", "l", "source_order", "E_g", "R_env", "S_risk"]:
            if column not in schema:
                raise SystemExit(f"V7 cache missing required column: {column}")
        total_cache_rows = int(conn.execute("SELECT COUNT(*) FROM v7_score_cache").fetchone()[0])
        args.total_cache_rows = total_cache_rows
        source_order_min, source_order_max, source_order_distinct = conn.execute(
            "SELECT MIN(source_order), MAX(source_order), COUNT(DISTINCT source_order) FROM v7_score_cache"
        ).fetchone()
        if int(source_order_distinct) != total_cache_rows:
            raise SystemExit("V7 cache source_order values are not unique; refusing source-order stream rewrite")
        logger.log(f"cache rows={total_cache_rows:,}; source_order range={int(source_order_min):,}..{int(source_order_max):,}")
        hkl_counts = load_hkl_counts(conn, logger)
        stream_metrics, stream_stats = parse_reciprocal_metrics(args.source_stream, logger)
        arrays, source_event_labels, domain_stats = load_domain_arrays(conn, stream_metrics, args, logger)
        conn.close()
        arrays["_source_event_labels"] = source_event_labels  # type: ignore[assignment]
        args.analysed_domain_rows = int(len(arrays["S_risk"]))
        selected_by_variant, removed_by_variant_hkl, candidates_by_variant_hkl, capacity_stats = select_variants(arrays, hkl_counts, args, variants, logger)
        validate_selection(variants, arrays, selected_by_variant, removed_by_variant_hkl, hkl_counts, args, logger)
        source_mask_bits = int(source_order_max) + 1
        masks, expected_key_by_source_order = make_masks(variants, arrays, selected_by_variant, source_mask_bits, logger)
        stream_qc = rewrite_streams(args.source_stream, variants, masks, expected_key_by_source_order, args, logger)

        summaries: dict[str, dict[str, Any]] = {}
        global_hkl_rows: list[dict[str, Any]] = []
        global_source_rows: list[dict[str, Any]] = []
        for variant in variants:
            summary = make_summary(
                variant,
                arrays,
                selected_by_variant[variant.variant_id],
                removed_by_variant_hkl[variant.variant_id],
                candidates_by_variant_hkl[variant.variant_id],
                hkl_counts,
                args,
                stream_qc.get(variant.variant_id),
            )
            summaries[variant.variant_id] = summary
            hkl_rows, source_rows = write_variant_artifacts(
                variant,
                arrays,
                source_event_labels,
                selected_by_variant[variant.variant_id],
                removed_by_variant_hkl[variant.variant_id],
                candidates_by_variant_hkl[variant.variant_id],
                hkl_counts,
                summary,
                args,
            )
            for row in hkl_rows:
                global_hkl_rows.append({"variant_id": variant.variant_id, "stream_type": variant.stream_type, "cutoff": variant.cutoff, **row})
            for row in source_rows:
                global_source_rows.append({"variant_id": variant.variant_id, "stream_type": variant.stream_type, "cutoff": variant.cutoff, **row})
            logger.log(f"wrote variant artifacts: {variant.variant_id}")

        run_stats = {
            "elapsed_seconds": float(time.monotonic() - started),
            "stream": stream_stats,
            "domain": domain_stats,
            "capacity": capacity_stats,
            "variant_ids": [variant.variant_id for variant in variants],
            "stream_qc": stream_qc,
        }
        write_root_outputs(args, variants, summaries, global_hkl_rows, global_source_rows, run_stats)
        logger.log(f"wrote root summaries and notes: {args.output_dir}")
        logger.log(f"partialator command file: {args.output_dir / 'v7_absolute_cutoff_partialator_commands.sh'}")
        logger.log("V7 absolute S_risk cutoff stream build complete; Partialator/merging were not run")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
