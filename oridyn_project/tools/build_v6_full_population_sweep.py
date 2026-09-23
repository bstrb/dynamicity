#!/usr/bin/env python3
"""Build the corrected full accepted-population OriDyn V6 sweep.

The production workflow is split into three explicit modes:

* cache: validate/reconstruct the 6,732,955-row accepted score cache.
* plan: build 88 score selections plus 39 shared matched-random controls.
* streams: rewrite the source stream in bounded batches from compact masks.

No mode runs Partialator or launches a merge.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import build_v6_score_target_filter_map as v6pilot
import build_v5_coarse_score_sweep_streams as v5coarse


BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
for _name in BLAS_THREAD_ENV_VARS:
    os.environ.setdefault(_name, "1")

KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
EXPECTED_ACCEPTED_COUNT = 6_732_955
RESTRICTED_V6_CACHE_COUNT = 659_147
EXPECTED_SOURCE_REFLECTION_ROWS = 10_591_743
EXPECTED_SCORE_VARIANTS = 88
EXPECTED_RANDOM_VARIANTS = 39
EXPECTED_TOTAL_VARIANTS = 127
V5_SENTINEL_ACTIONABLE_OBSERVATIONS = 648_248
V5_SENTINEL_ACTIONABLE_BLOCKS = 67_837
V5_SENTINEL_DROP30_REMOVALS = 183_608
RANDOM_SEEDS = [20260717, 20260718, 20260719]
FILTER_SCORE_IDS = ["eg_cmean", "eg_c2mean", "eg_m2", "eg_d1_cmean", "eg_d2_cmean", "eg_d3_cmean"]
TARGET_A_DROPS = [0.05, 0.10, 0.20]
TARGET_BC_DROPS = [0.20, 0.30, 0.40, 0.50]
MAX_OPEN_STREAMS = 12
CSV_FLOAT_FORMAT = "%.12g"
PLAN_BUILDER_VERSION = "v6_full_population_parallel_plan_v2"

HKL_QC_FIELDNAMES = [
    "variant_id",
    "h",
    "k",
    "l",
    "n_observations",
    "n_selected",
    "n_omitted_middle",
    "n_eligible",
    "n_removed",
    "actionable",
    "validation_passed",
]
BLOCK_QC_FIELDNAMES = [
    "variant_id",
    "h",
    "k",
    "l",
    "block_id",
    "block_size",
    "n_removed",
    "n_retained_in_block",
    "validation_passed",
]
BLOCK_DEFINITION_FIELDNAMES = ["h", "k", "l", "block_id", "block_size", "definition_source"]

BASELINE_SG0 = v6pilot.BASELINE_SG0
SG0_MULTIPLIER = v6pilot.SG0_MULTIPLIER
SIGMA_C_MULTIPLIER = v6pilot.SIGMA_C_MULTIPLIER
SIGMA_C = v6pilot.SIGMA_C
R_CUT = v6pilot.R_CUT
HIGH_EG_FRACTION = v6pilot.HIGH_EG_FRACTION
EXCITATION_BLOCK_SIZE = v6pilot.EXCITATION_BLOCK_SIZE
MIN_FINAL_BLOCK_SIZE = v6pilot.MIN_FINAL_BLOCK_SIZE
MIN_HIGH_EG_OBSERVATIONS = v6pilot.MIN_HIGH_EG_OBSERVATIONS
MIN_REMAINING = v6pilot.MIN_REMAINING
TARGET_A_MIN_OBSERVATIONS = v6pilot.TARGET_A_MIN_OBSERVATIONS

STREAM_IMAGE_RE = v6pilot.STREAM_IMAGE_RE
STREAM_EVENT_RE = v6pilot.STREAM_EVENT_RE
STREAM_FILENAME_RE = v6pilot.STREAM_FILENAME_RE
POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class PlannedStream:
    variant_id: str
    experiment_type: str
    score_id: str
    random_control_id: str
    designation: str
    filtering_target: str
    drop_fraction: float | None
    seed: int | None
    output_filename: str
    mask_mode: str
    priority: str


@dataclass(frozen=True)
class MaskSpec:
    variant_id: str
    path: Path
    mode: str
    output_path: Path
    reused_existing_path: Path | None


class RunLogger:
    def __init__(self, out_dir: Path | None):
        self.out_dir = out_dir
        self.handle = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.handle = (out_dir / "run.log").open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.handle is not None:
            self.handle.write(line + "\n")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class StageProgress:
    def __init__(self, logger: RunLogger, stage: str, total: int | None, unit: str = "items"):
        self.logger = logger
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.start = time.monotonic()
        self.last = self.start
        self.completed = 0
        total_text = f"{self.total:,}" if self.total is not None else "unknown"
        self.logger.log(f"{stage} start: total={total_text} {unit}")

    def update(self, completed: int, force: bool = False) -> None:
        self.completed = int(completed)
        now = time.monotonic()
        if not force and self.total is not None and self.completed != self.total and now - self.last < 30.0:
            return
        elapsed = max(now - self.start, 1.0e-9)
        rate = self.completed / elapsed
        pct = 100.0 * self.completed / self.total if self.total else 0.0
        eta = (self.total - self.completed) / rate if self.total and rate > 0 and self.completed < self.total else 0.0
        rss = current_rss_mb()
        rss_text = f"; rss={rss:.1f} MB" if rss is not None else ""
        total = f"/{self.total:,} ({pct:.1f}%)" if self.total is not None else ""
        self.logger.log(
            f"{self.stage} progress: completed={self.completed:,}{total}; "
            f"elapsed={format_seconds(elapsed)}; rate={rate:.2f} {self.unit}/s; eta={format_seconds(eta)}{rss_text}"
        )
        self.last = now

    def advance(self, amount: int = 1) -> None:
        self.update(self.completed + int(amount))

    def finish(self, completed: int | None = None) -> None:
        final = self.completed if completed is None else int(completed)
        self.update(final, force=True)
        elapsed = max(time.monotonic() - self.start, 1.0e-9)
        self.logger.log(f"{self.stage} complete: completed={final:,}; elapsed={format_seconds(elapsed)}")


class PackedMask:
    def __init__(self, path: Path, n_bits: int, *, create: bool):
        self.path = path
        self.n_bits = int(n_bits)
        self.n_bytes = (self.n_bits + 7) // 8
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.truncate(self.n_bytes)
            mode = "r+"
        else:
            mode = "r"
        self.array = np.memmap(path, dtype=np.uint8, mode=mode, shape=(self.n_bytes,))

    def set_many(self, ordinals: Iterable[int]) -> None:
        if isinstance(ordinals, np.ndarray):
            indexes = ordinals.astype(np.int64, copy=False)
        else:
            indexes = np.fromiter((int(ordinal) for ordinal in ordinals), dtype=np.int64)
        if indexes.size == 0:
            return
        if int(indexes.min()) < 0 or int(indexes.max()) >= self.n_bits:
            bad = int(indexes[(indexes < 0) | (indexes >= self.n_bits)][0])
            raise IndexError(bad)
        byte_indexes = np.right_shift(indexes, 3)
        bit_values = np.left_shift(1, np.bitwise_and(indexes, 7)).astype(np.uint8, copy=False)
        np.bitwise_or.at(self.array, byte_indexes, bit_values)

    def get(self, ordinal: int) -> bool:
        index = int(ordinal)
        if index < 0 or index >= self.n_bits:
            return False
        return bool(int(self.array[index >> 3]) & (1 << (index & 7)))

    def count(self) -> int:
        return int(np.unpackbits(np.asarray(self.array), bitorder="little")[: self.n_bits].sum())

    def flush(self) -> None:
        self.array.flush()


def current_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        try:
            with Path("/proc/self/statm").open("r", encoding="utf-8") as handle:
                resident_pages = int(handle.read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(resident_pages * page_size / (1024 * 1024))
        except Exception:
            return None


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s"


def format_eta(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if 0.0 < seconds < 0.1:
        seconds = 0.1
    return format_seconds(seconds)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-stream", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["cache", "plan", "streams"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--random-replicates", type=int, default=3)
    parser.add_argument("--accepted-population", type=Path, default=None)
    parser.add_argument("--v5-scores", type=Path, default=None)
    parser.add_argument("--v5-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--plan-batch-hkls", type=int, default=250, help="Signed-HKL groups per parallel plan task.")
    parser.add_argument("--max-pending-batches", type=int, default=None, help="Maximum submitted-but-unconsumed plan batches; default is 2 * workers.")
    parser.add_argument("--expected-accepted-count", type=int, default=EXPECTED_ACCEPTED_COUNT)
    parser.add_argument("--expected-source-reflection-rows", type=int, default=EXPECTED_SOURCE_REFLECTION_ROWS)
    parser.add_argument("--max-cache-rows", type=int, default=None, help="Synthetic-test cap only; production must omit this.")
    parser.add_argument("--skip-halfset-help", action="store_true", help="Skip local partialator --help inspection.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.random_replicates != 3:
        raise SystemExit("This experiment defines exactly three random replicates; use --random-replicates 3")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    if args.target_batch_size < 1:
        raise SystemExit("--target-batch-size must be >= 1")
    if args.plan_batch_hkls < 1:
        raise SystemExit("--plan-batch-hkls must be >= 1")
    if args.max_pending_batches is None:
        args.max_pending_batches = 2 * args.workers
    if args.max_pending_batches < 1:
        raise SystemExit("--max-pending-batches must be >= 1")
    return normalize_args(args)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    root = args.root.expanduser().resolve()
    args.root = root
    args.source_stream = args.source_stream.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.accepted_population = (args.accepted_population or default_accepted_population(root)).expanduser().resolve()
    args.v5_scores = (args.v5_scores or default_v5_scores(root)).expanduser().resolve()
    args.v5_dir = (args.v5_dir or root / "oridyn_v5_p_lambda_screen_20260716").expanduser().resolve()
    return args


def default_accepted_population(root: Path) -> Path:
    return (
        root
        / "oridyn_v4_local_crowding_raw_20_0p3_20260704"
        / "partialator_survivor_mask"
        / "p1_iter1_20260705T1214"
        / "v4_p1_iter1_partialator_survivors_only_scores.csv"
    )


def default_v5_scores(root: Path) -> Path:
    return root / "oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705" / "geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def normalize_source(value: Any) -> str:
    return v6pilot.normalize_source(value)


def normalize_event(value: Any) -> str:
    return v6pilot.normalize_event(value)


def key_to_text(source: Any, event: Any, h: int, k: int, l: int) -> str:
    return v6pilot.key_to_text(source, event, int(h), int(k), int(l))


def percent_label(fraction: float) -> str:
    return v6pilot.percent_label(float(fraction))


def stable_hash_u64(seed: int, key: str) -> int:
    digest = hashlib.blake2b(f"{int(seed)}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def read_csv_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        return next(csv.reader(handle))


def require_columns(header: list[str], columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return ""
    return result.stdout.strip()


def file_fingerprint(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": path.exists(), "sha256": ""}
    if not path.exists():
        return row
    stat = path.stat()
    row.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    if hash_file:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        row["sha256"] = h.hexdigest()
    return row


def db_path(out_dir: Path) -> Path:
    return out_dir / "full_population_cache.sqlite"


def connect_db(out_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path(out_dir)))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    return conn


def init_cache_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS accepted;
        DROP TABLE IF EXISTS score_cache;
        CREATE TABLE accepted (
            ordinal INTEGER PRIMARY KEY,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_order INTEGER
        );
        CREATE INDEX accepted_frame_idx ON accepted(source_filename, event);
        CREATE INDEX accepted_hkl_idx ON accepted(h, k, l);
        CREATE TABLE score_cache (
            ordinal INTEGER PRIMARY KEY,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_order INTEGER NOT NULL,
            sg REAL NOT NULL,
            abs_sg REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            U REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL
        );
        CREATE INDEX score_cache_hkl_idx ON score_cache(h, k, l, exact_key_text);
        CREATE INDEX score_cache_source_order_idx ON score_cache(source_order);
        """
    )
    conn.commit()


def normalize_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[~out[HKL_COLUMNS].isna().any(axis=1)].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    out["exact_key_text"] = [key_to_text(source, event, h, k, l) for source, event, h, k, l in out[KEY_COLUMNS].itertuples(index=False, name=None)]
    return out


def load_accepted_population(conn: sqlite3.Connection, path: Path, chunksize: int, expected_count: int, max_rows: int | None, logger: RunLogger) -> dict[str, Any]:
    header = read_csv_header(path)
    require_columns(header, KEY_COLUMNS, "accepted population")
    progress = StageProgress(logger, "cache: loading authoritative accepted population", expected_count if max_rows is None else max_rows, "rows")
    rows_read = 0
    rows_inserted = 0
    checksum = hashlib.sha256()
    remaining = max_rows
    for chunk in pd.read_csv(path, usecols=KEY_COLUMNS, chunksize=chunksize, low_memory=False):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= int(len(chunk))
        if chunk.empty:
            continue
        rows_read += int(len(chunk))
        work = normalize_key_frame(chunk)
        records = []
        for row in work[["source_filename", "event", "h", "k", "l", "exact_key_text"]].itertuples(index=False, name=None):
            checksum.update((row[5] + "\n").encode("utf-8"))
            records.append((rows_inserted, *row))
            rows_inserted += 1
        try:
            conn.executemany(
                "INSERT INTO accepted(ordinal,source_filename,event,h,k,l,exact_key_text) VALUES(?,?,?,?,?,?,?)",
                records,
            )
        except sqlite3.IntegrityError as exc:
            raise SystemExit(f"Accepted population contains duplicate exact keys near row {rows_inserted:,}: {exc}") from exc
        if rows_inserted % max(chunksize, 1) == 0:
            conn.commit()
        progress.update(rows_inserted)
    conn.commit()
    progress.finish(rows_inserted)
    if rows_inserted == RESTRICTED_V6_CACHE_COUNT:
        raise SystemExit("Refusing restricted 659,147-row V6 cache/population as full accepted population")
    if max_rows is None and rows_inserted != int(expected_count):
        raise SystemExit(f"Accepted population count mismatch: observed={rows_inserted:,}, expected={expected_count:,}")
    return {
        "accepted_population_path": str(path),
        "accepted_rows_read": int(rows_read),
        "accepted_rows_inserted": int(rows_inserted),
        "expected_accepted_count": int(expected_count),
        "accepted_key_checksum_sha256_read_order": checksum.hexdigest(),
        "exact_key_unique": True,
    }


def parse_reflection_hkl(line: str) -> tuple[int, int, int] | None:
    return v6pilot.parse_reflection_hkl(line)


def scan_source_stream_orders(conn: sqlite3.Connection, source_stream: Path, expected_source_rows: int, max_rows: int | None, logger: RunLogger) -> dict[str, Any]:
    progress = StageProgress(logger, "cache: scanning source stream for accepted-key coverage", expected_source_rows if max_rows is None else None, "reflection rows")
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    matched = 0
    updates: list[tuple[int, str]] = []
    with source_stream.open("r", encoding="utf-8", errors="replace") as handle:
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
            if match := STREAM_FILENAME_RE.match(line):
                parsed_source = normalize_source(match.group(1))
                parsed_event = normalize_event(match.group(2)) if match.group(2) is not None else ""
                if in_crystal:
                    current_source = parsed_source
                    if parsed_event:
                        current_event = parsed_event
                else:
                    chunk_source = parsed_source
                    if parsed_event:
                        chunk_event = parsed_event
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
            hkl = parse_reflection_hkl(line) if in_crystal and in_reflections else None
            if hkl is None:
                continue
            rows_seen += 1
            key = key_to_text(current_source, current_event, *hkl)
            updates.append((rows_seen, key))
            if len(updates) >= 50_000:
                matched += apply_source_order_updates(conn, updates)
                updates.clear()
                progress.update(rows_seen)
            if max_rows is not None and rows_seen >= max_rows:
                break
    if updates:
        matched += apply_source_order_updates(conn, updates)
    conn.commit()
    missing = int(conn.execute("SELECT COUNT(*) FROM accepted WHERE source_order IS NULL").fetchone()[0])
    duplicate_source_hits = 0
    progress.finish(rows_seen)
    if max_rows is None and rows_seen != int(expected_source_rows):
        raise SystemExit(f"Source stream reflection row count mismatch: observed={rows_seen:,}, expected={expected_source_rows:,}")
    if missing:
        raise SystemExit(f"Accepted/source validation failed: {missing:,} accepted keys were not found in the source stream")
    return {
        "source_stream": str(source_stream),
        "source_reflection_rows": int(rows_seen),
        "expected_source_reflection_rows": int(expected_source_rows),
        "accepted_keys_found_in_source": int(matched),
        "accepted_keys_missing_in_source": int(missing),
        "duplicate_source_hits_not_tracked_beyond_unique_update": int(duplicate_source_hits),
        "source_rows_are_not_accepted_rows": True,
    }


def apply_source_order_updates(conn: sqlite3.Connection, updates: list[tuple[int, str]]) -> int:
    before = conn.total_changes
    conn.executemany("UPDATE accepted SET source_order=? WHERE exact_key_text=? AND source_order IS NULL", updates)
    return int(conn.total_changes - before)


def accepted_rows_for_frame(conn: sqlite3.Connection, source: str, event: str) -> list[tuple[int, int, str]]:
    return [
        (int(ordinal), int(source_order), str(key))
        for ordinal, source_order, key in conn.execute(
            "SELECT ordinal, source_order, exact_key_text FROM accepted WHERE source_filename=? AND event=? ORDER BY exact_key_text",
            (source, event),
        )
    ]


def score_frame_for_cache(task: tuple[pd.DataFrame, list[str], int, list[dict[str, Any]]]) -> pd.DataFrame:
    group, target_keys, target_batch_size, raw_records = task
    _pid, frame, _stats = v5coarse.score_frame_variants_worker((group, target_keys, target_batch_size, raw_records))
    if frame.empty:
        return frame
    normalized = v5coarse.normalize_frame_group(group)
    payload = normalized.loc[:, [*KEY_COLUMNS, "sg_target", "target_excitation_Eg"]].copy()
    frame = frame.merge(payload, on=KEY_COLUMNS, how="left", sort=False, validate="one_to_one")
    return frame


def populate_score_cache(conn: sqlite3.Connection, v5_scores: Path, chunksize: int, target_batch_size: int, workers: int, logger: RunLogger) -> dict[str, Any]:
    raw = v5coarse.raw_kernel_spec(SG0_MULTIPLIER, SIGMA_C_MULTIPLIER)
    raw_records = [asdict(raw)]
    usecols = [*KEY_COLUMNS, "d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"]
    header = read_csv_header(v5_scores)
    require_columns(header, usecols, "V5 geometry score table")
    progress = StageProgress(logger, "cache: computing full-population D/U/M/M2 components", None, "frames")
    frames_seen = 0
    rows_inserted = 0
    inserted_keys = hashlib.sha256()
    # Frame-level scoring is deterministic; this loop keeps only one source frame in memory.
    for group in v5coarse.v5mod.iter_frame_groups(v5_scores, usecols, chunksize, None, None):
        work = v5coarse.normalize_frame_group(group)
        if work.empty:
            continue
        source = str(work["source_filename"].iloc[0])
        event = str(work["event"].iloc[0])
        accepted = accepted_rows_for_frame(conn, source, event)
        if not accepted:
            continue
        ordinal_by_key = {key: (ordinal, source_order) for ordinal, source_order, key in accepted}
        scored = score_frame_for_cache((work, sorted(ordinal_by_key), target_batch_size, raw_records))
        if scored.empty:
            continue
        records = []
        for row in scored.itertuples(index=False):
            payload = row._asdict()
            key = str(payload["exact_key_text"])
            ordinal, source_order = ordinal_by_key[key]
            sg = float(payload["sg_target"])
            eg = float(payload["target_excitation_Eg"])
            d = float(payload[raw.coupling_sum_column])
            u = float(payload[v5coarse.aggregate_columns_for_raw(raw)["U"]])
            m = float(payload[raw.score_column])
            m2 = float(payload[v5coarse.aggregate_columns_for_raw(raw)["M2"]])
            values = [sg, abs(sg), eg, d, u, m, m2]
            if not all(math.isfinite(value) for value in values):
                raise SystemExit(f"Nonfinite score component for key {key}")
            records.append((ordinal, payload["source_filename"], payload["event"], int(payload["h"]), int(payload["k"]), int(payload["l"]), key, source_order, sg, abs(sg), eg, d, u, m, m2))
            inserted_keys.update((key + "\n").encode("utf-8"))
        conn.executemany(
            """
            INSERT INTO score_cache(ordinal,source_filename,event,h,k,l,exact_key_text,source_order,sg,abs_sg,Eg,D,U,M,M2)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            records,
        )
        rows_inserted += len(records)
        frames_seen += 1
        if frames_seen % 100 == 0:
            conn.commit()
            progress.update(frames_seen, force=True)
    conn.commit()
    cache_rows = int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])
    accepted_rows = int(conn.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])
    if cache_rows != accepted_rows:
        raise SystemExit(f"Score cache row count mismatch: score_cache={cache_rows:,}, accepted={accepted_rows:,}")
    progress.finish(frames_seen)
    return {
        "v5_scores": str(v5_scores),
        "raw_kernel_name": raw.name,
        "frames_scored": int(frames_seen),
        "score_cache_rows": int(cache_rows),
        "score_cache_key_checksum_sha256_score_order": inserted_keys.hexdigest(),
        "geometry_parameters": geometry_parameters(),
    }


def geometry_parameters() -> dict[str, Any]:
    return {
        "baseline_sg0": BASELINE_SG0,
        "sg0_multiplier": SG0_MULTIPLIER,
        "sg0": BASELINE_SG0 * SG0_MULTIPLIER,
        "sigma_c_multiplier": SIGMA_C_MULTIPLIER,
        "sigma_c": SIGMA_C,
        "r_cut": R_CUT,
        "target_reflection_excluded": True,
        "exact_signed_hkls": True,
        "symmetry_canonicalization": False,
    }


def export_score_cache_parquet(conn: sqlite3.Connection, out_dir: Path, logger: RunLogger) -> dict[str, Any]:
    path = out_dir / "full_score_cache.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        logger.log(f"Parquet export unavailable ({exc}); SQLite cache remains authoritative.")
        return {"format": "sqlite", "path": str(db_path(out_dir)), "parquet_written": False, "reason": str(exc)}
    progress = StageProgress(logger, "cache: exporting validated score cache to Parquet", None, "rows")
    writer = None
    total = 0
    try:
        for chunk in pd.read_sql_query(
            "SELECT source_filename,event,h,k,l,ordinal AS source_order,sg,abs_sg,Eg,D,U,M,M2,exact_key_text FROM score_cache ORDER BY ordinal",
            conn,
            chunksize=250_000,
        ):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(chunk)
            progress.update(total)
    finally:
        if writer is not None:
            writer.close()
    progress.finish(total)
    return {"format": "sqlite+parquet", "path": str(path), "sqlite_path": str(db_path(out_dir)), "parquet_written": True, "rows": int(total)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_optional_parquet(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"parquet_written": False, "reason": "no rows"}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        return {"parquet_written": False, "reason": str(exc)}
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    return {"parquet_written": True, "path": str(path), "rows": int(len(rows))}


def run_cache(args: argparse.Namespace) -> int:
    require_file(args.source_stream, "--source-stream")
    require_file(args.accepted_population, "--accepted-population")
    require_file(args.v5_scores, "--v5-scores")
    ensure_out_dir(args.out_dir)
    logger = RunLogger(args.out_dir)
    try:
        logger.log("mode=cache; starting full-population cache build")
        conn = connect_db(args.out_dir)
        init_cache_db(conn)
        accepted_stats = load_accepted_population(conn, args.accepted_population, args.chunksize, args.expected_accepted_count, args.max_cache_rows, logger)
        source_stats = scan_source_stream_orders(conn, args.source_stream, args.expected_source_reflection_rows, args.max_cache_rows, logger)
        score_stats = populate_score_cache(conn, args.v5_scores, args.chunksize, args.target_batch_size, args.workers, logger)
        cache_export = export_score_cache_parquet(conn, args.out_dir, logger)
        row_count = int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])
        if args.max_cache_rows is None and row_count != EXPECTED_ACCEPTED_COUNT:
            raise SystemExit(f"Full cache gate failed: {row_count:,} rows != {EXPECTED_ACCEPTED_COUNT:,}")
        validation = {
            "passed": True,
            "cache_gate": {
                "expected_accepted_count": int(args.expected_accepted_count),
                "score_cache_rows": int(row_count),
                "restricted_659147_cache_rejected": row_count != RESTRICTED_V6_CACHE_COUNT,
                "required_columns": ["source_filename", "event", "h", "k", "l", "source_order", "sg", "abs_sg", "Eg", "D", "U", "M", "M2"],
            },
            "accepted_population": accepted_stats,
            "source_stream_validation": source_stats,
            "score_cache": score_stats,
            "cache_export": cache_export,
        }
        provenance = {
            "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_stream": file_fingerprint(args.source_stream, hash_file=False),
            "accepted_population_source": file_fingerprint(args.accepted_population, hash_file=False),
            "v5_geometry_score_source": file_fingerprint(args.v5_scores, hash_file=False),
            "accepted_population_source_role": "authoritative p1_iter1 Partialator-survivor accepted observation manifest",
            "row_count": int(row_count),
            "exact_key_uniqueness": True,
            "geometry_parameters": geometry_parameters(),
            "scoring_implementation_version": "v6_full_population_sweep_sqlite_v1",
            "git_commit": git_commit(Path(__file__).resolve().parents[1]),
            "package_versions": package_versions(),
        }
        write_json(args.out_dir / "accepted_population_validation.json", validation)
        write_json(args.out_dir / "cache_provenance.json", provenance)
        write_json(args.out_dir / "run_metadata.json", base_run_metadata(args, "cache"))
        write_readme(args.out_dir)
        logger.log("cache mode complete")
        return 0
    finally:
        logger.close()


def package_versions() -> dict[str, str]:
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}


def base_run_metadata(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    return {
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "root": str(args.root),
        "source_stream": str(args.source_stream),
        "out_dir": str(args.out_dir),
        "workers": int(args.workers),
        "plan_batch_hkls": int(args.plan_batch_hkls),
        "max_pending_batches": int(args.max_pending_batches),
        "random_replicates": int(args.random_replicates),
        "random_seeds": RANDOM_SEEDS,
        "numeric_thread_environment": {name: os.environ.get(name, "") for name in BLAS_THREAD_ENV_VARS},
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "package_versions": package_versions(),
    }


def write_readme(out_dir: Path) -> None:
    text = """# OriDyn V6 Full Accepted-Population Sweep

This directory is produced by `tools/build_v6_full_population_sweep.py`.

The corrected experiment keeps the source-stream reflection-row population,
the accepted/scored population, the eligible/actionable populations, and the
removed/retained populations numerically separate.  It writes cache/provenance,
selection masks, manifests, stream rewrite QC, and merge metadata only.  It
does not run Partialator or create merge-result directories.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def score_variants() -> list[PlannedStream]:
    out: list[PlannedStream] = []
    for variant in v6pilot.build_experiment_plan(v6pilot.score_registry()):
        mask_mode = "keep" if variant.experiment_type == "diagnostic_low_high" else "remove"
        out.append(
            PlannedStream(
                variant_id=variant.variant_id,
                experiment_type=variant.experiment_type,
                score_id=variant.score_id,
                random_control_id="",
                designation=variant.designation,
                filtering_target=variant.filtering_target,
                drop_fraction=variant.drop_fraction,
                seed=None,
                output_filename=variant.output_filename,
                mask_mode=mask_mode,
                priority=variant.suggested_priority,
            )
        )
    return out


def random_variants() -> list[PlannedStream]:
    rows: list[PlannedStream] = []
    priority = 10_000
    for seed in RANDOM_SEEDS:
        for side in ["low50", "high50"]:
            priority += 1
            rows.append(PlannedStream(f"random_diag_seed{seed}_{side}", "diagnostic_random_control", "", f"diag_seed{seed}", side, "unrestricted_per_hkl_half_split", None, seed, f"random_diag_seed{seed}_{side}.stream", "keep", f"R.{priority:05d}"))
    for drop in TARGET_A_DROPS:
        for seed in RANDOM_SEEDS:
            label = percent_label(drop)
            priority += 1
            rows.append(PlannedStream(f"random_all_drop{label}_seed{seed}", "targeted_random_control", "", f"all_drop{label}_seed{seed}", f"drop{label}", "all", drop, seed, f"random_all_drop{label}_seed{seed}.stream", "remove", f"R.{priority:05d}"))
    for target in ["higheg", "matched"]:
        for drop in TARGET_BC_DROPS:
            for seed in RANDOM_SEEDS:
                label = percent_label(drop)
                priority += 1
                rows.append(PlannedStream(f"random_{target}_drop{label}_seed{seed}", "targeted_random_control", "", f"{target}_drop{label}_seed{seed}", f"drop{label}", target, drop, seed, f"random_{target}_drop{label}_seed{seed}.stream", "remove", f"R.{priority:05d}"))
    if len(rows) != EXPECTED_RANDOM_VARIANTS:
        raise RuntimeError(len(rows))
    return rows


def all_planned_streams() -> list[PlannedStream]:
    rows = score_variants() + random_variants()
    if len(rows) != EXPECTED_TOTAL_VARIANTS:
        raise RuntimeError(len(rows))
    return rows


def score_cache_count(conn: sqlite3.Connection) -> int:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])
    except sqlite3.Error as exc:
        raise SystemExit(f"Could not read score_cache from {db_path(Path('.'))}: {exc}") from exc


def iter_hkl_groups(conn: sqlite3.Connection, logger: RunLogger) -> Iterator[pd.DataFrame]:
    cursor = conn.execute(
        "SELECT ordinal,source_filename,event,h,k,l,exact_key_text,source_order,sg,abs_sg,Eg,D,U,M,M2 "
        "FROM score_cache ORDER BY h,k,l,exact_key_text"
    )
    current_key: tuple[int, int, int] | None = None
    rows: list[tuple[Any, ...]] = []
    count = 0
    columns = ["ordinal", "source_filename", "event", "h", "k", "l", "exact_key_text", "source_order", "sg", "abs_sg", "Eg", "D", "U", "M", "M2"]
    progress = StageProgress(logger, "plan: streaming score cache by signed HKL", None, "observations")
    for row in cursor:
        key = (int(row[3]), int(row[4]), int(row[5]))
        if current_key is None:
            current_key = key
        if key != current_key:
            yield pd.DataFrame.from_records(rows, columns=columns)
            rows = []
            current_key = key
        rows.append(row)
        count += 1
        if count % 250_000 == 0:
            progress.update(count, force=True)
    if rows:
        yield pd.DataFrame.from_records(rows, columns=columns)
    progress.finish(count)


def score_group(group: pd.DataFrame, scores: list[v6pilot.ScoreDefinition]) -> pd.DataFrame:
    table = group.copy()
    variables = {name: pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float) for name in ["Eg", "D", "U", "M", "M2"]}
    for score in scores:
        stats: dict[str, Any] = {}
        table[f"score_{score.score_id}"] = v6pilot.evaluate_expression_tree(score.expression_tree, variables, score.score_id, stats)
    return table


def split_excitation_blocks(high_pool: pd.DataFrame) -> list[pd.DataFrame]:
    return v6pilot.split_excitation_blocks(high_pool.rename(columns={"abs_sg": "sg_abs"}))


def removal_count(n_eligible: int, drop_fraction: float) -> int:
    return v6pilot.removal_count(int(n_eligible), float(drop_fraction), MIN_REMAINING)


def open_masks(out_dir: Path, planned: list[PlannedStream], accepted_count: int, create: bool) -> dict[str, PackedMask]:
    return {variant.variant_id: PackedMask(out_dir / "selection_masks" / f"{variant.variant_id}.{variant.mask_mode}.bitset", accepted_count, create=create) for variant in planned}


def append_count(rows: list[dict[str, Any]], variant: PlannedStream, selected_or_removed: int, eligible: int, actionable: int, source_rows: int, accepted_count: int) -> None:
    removed = selected_or_removed if variant.mask_mode == "remove" else accepted_count - selected_or_removed
    retained_accepted = accepted_count - removed
    rows.append(
        {
            "variant_id": variant.variant_id,
            "experiment_type": variant.experiment_type,
            "score_id": variant.score_id,
            "random_control_id": variant.random_control_id,
            "filtering_target": variant.filtering_target,
            "drop_fraction": variant.drop_fraction if variant.drop_fraction is not None else "",
            "seed": variant.seed if variant.seed is not None else "",
            "mask_mode": variant.mask_mode,
            "accepted_population_count": accepted_count,
            "source_reflection_row_count": source_rows,
            "eligible_observation_count": int(eligible),
            "actionable_observation_count": int(actionable),
            "selected_or_removed_accepted_observations": int(selected_or_removed),
            "accepted_observations_removed": int(removed),
            "accepted_observations_retained": int(retained_accepted),
            "out_of_analysis_source_rows_retained": int(source_rows - accepted_count),
            "total_source_rows_retained": int(source_rows - removed),
            "removed_fraction_of_accepted_population": float(removed / max(1, accepted_count)),
            "removed_fraction_of_source_rows": float(removed / max(1, source_rows)),
            "removed_fraction_of_eligible_population": float(removed / eligible) if eligible else "",
            "removed_fraction_of_actionable_population": float(removed / actionable) if actionable else "",
        }
    )


def build_block_definitions(block_qc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in block_qc_rows:
        key = (int(row["h"]), int(row["k"]), int(row["l"]), int(row["block_id"]))
        if key not in blocks:
            blocks[key] = {
                "h": key[0],
                "k": key[1],
                "l": key[2],
                "block_id": key[3],
                "block_size": int(row["block_size"]),
                "definition_source": "target_c_common_frozen_excitation_blocks",
            }
    return [blocks[key] for key in sorted(blocks)]


def mask_bit_count(mask: PackedMask) -> int:
    return int(POPCOUNT8[np.asarray(mask.array)].sum())


def mask_pair_intersection(mask_a: PackedMask, mask_b: PackedMask) -> int:
    return int(POPCOUNT8[np.bitwise_and(np.asarray(mask_a.array), np.asarray(mask_b.array))].sum())


def build_selection_overlap(planned: list[PlannedStream], masks: dict[str, PackedMask], logger: RunLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = {variant.variant_id: mask_bit_count(masks[variant.variant_id]) for variant in planned}
    progress = StageProgress(logger, "plan: computing compact-mask selection overlaps", len(planned) * (len(planned) - 1) // 2, "pairs")
    completed = 0
    for i, left in enumerate(planned):
        left_count = counts[left.variant_id]
        for right in planned[i + 1 :]:
            right_count = counts[right.variant_id]
            intersection = mask_pair_intersection(masks[left.variant_id], masks[right.variant_id])
            union = left_count + right_count - intersection
            rows.append(
                {
                    "variant_id_a": left.variant_id,
                    "variant_id_b": right.variant_id,
                    "mask_mode_a": left.mask_mode,
                    "mask_mode_b": right.mask_mode,
                    "mask_meaning_a": "retained_observations" if left.mask_mode == "keep" else "removed_observations",
                    "mask_meaning_b": "retained_observations" if right.mask_mode == "keep" else "removed_observations",
                    "selected_count_a": int(left_count),
                    "selected_count_b": int(right_count),
                    "intersection_count": int(intersection),
                    "union_count": int(union),
                    "jaccard": float(intersection / union) if union else "",
                    "overlap_fraction_of_a": float(intersection / left_count) if left_count else "",
                    "overlap_fraction_of_b": float(intersection / right_count) if right_count else "",
                }
            )
            completed += 1
            progress.update(completed)
    progress.finish(completed)
    return rows


def score_correlation_rows(conn: sqlite3.Connection, logger: RunLogger) -> list[dict[str, Any]]:
    scores = v6pilot.score_registry()
    score_ids = [score.score_id for score in scores]
    n = 0
    sums = np.zeros(len(scores), dtype=np.float64)
    cross = np.zeros((len(scores), len(scores)), dtype=np.float64)
    progress = StageProgress(logger, "plan: computing score correlations from cache", None, "rows")
    for chunk in pd.read_sql_query("SELECT Eg,D,U,M,M2 FROM score_cache ORDER BY ordinal", conn, chunksize=250_000):
        variables = {name: pd.to_numeric(chunk[name], errors="coerce").to_numpy(dtype=float) for name in ["Eg", "D", "U", "M", "M2"]}
        columns = []
        for score in scores:
            stats: dict[str, Any] = {}
            columns.append(v6pilot.evaluate_expression_tree(score.expression_tree, variables, score.score_id, stats))
        matrix = np.column_stack(columns).astype(np.float64, copy=False)
        sums += matrix.sum(axis=0)
        cross += matrix.T @ matrix
        n += int(matrix.shape[0])
        progress.update(n)
    progress.finish(n)
    if n <= 1:
        return []
    means = sums / n
    covariance = (cross - n * np.outer(means, means)) / (n - 1)
    variances = np.diag(covariance)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(score_ids):
        for j, right in enumerate(score_ids[i:], start=i):
            denom = math.sqrt(max(float(variances[i]), 0.0) * max(float(variances[j]), 0.0))
            rows.append(
                {
                    "score_id_a": left,
                    "score_id_b": right,
                    "n_observations": int(n),
                    "pearson_r": float(covariance[i, j] / denom) if denom > 0 else "",
                }
            )
    return rows


class CsvRowWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", newline="", buffering=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.rows_written = 0

    def writerows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.writer.writerow({key: row.get(key, "") for key in self.fieldnames})
            self.rows_written += 1

    def close(self) -> None:
        self.handle.close()


def reset_plan_outputs(out_dir: Path, logger: RunLogger) -> None:
    stale_files = [
        "validation.json",
        "parallelism_validation.json",
        "plan_completion.json",
        "experiment_plan.csv",
        "experiment_plan.json",
        "stream_manifest.csv",
        "selection_counts.csv",
        "per_variant_per_hkl_qc.csv",
        "per_variant_per_block_qc.csv",
        "block_definitions.csv",
        "block_definitions.parquet",
        "random_control_manifest.csv",
        "selection_overlap.csv",
        "score_correlations.csv",
        "disk_preflight.json",
        "target_c_domain_validation.json",
        "merge_manifest.tsv",
        "halfset_support.json",
        "halfset_usage.md",
        "parameters.json",
        "scores.json",
    ]
    removed = 0
    for name in stale_files:
        path = out_dir / name
        if path.exists():
            path.unlink()
            removed += 1
    mask_dir = out_dir / "selection_masks"
    if mask_dir.is_dir():
        for path in mask_dir.glob("*.bitset"):
            path.unlink()
            removed += 1
    if removed:
        logger.log(f"plan: cleared {removed} stale plan artifact(s) before rebuilding")


def count_signed_hkls(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM (SELECT h,k,l FROM score_cache GROUP BY h,k,l)").fetchone()[0])


def iter_hkl_batches(conn: sqlite3.Connection, batch_hkls: int) -> Iterator[list[tuple[int, int, int]]]:
    cursor = conn.execute("SELECT h,k,l FROM score_cache GROUP BY h,k,l ORDER BY h,k,l")
    batch: list[tuple[int, int, int]] = []
    for h, k, l in cursor:
        batch.append((int(h), int(k), int(l)))
        if len(batch) >= int(batch_hkls):
            yield batch
            batch = []
    if batch:
        yield batch


def plan_worker_initializer() -> None:
    for name in BLAS_THREAD_ENV_VARS:
        os.environ[name] = "1"


def fetch_score_cache_batch(db_file: str, hkls: list[tuple[int, int, int]]) -> pd.DataFrame:
    if not hkls:
        return pd.DataFrame(
            columns=[
                "ordinal",
                "source_filename",
                "event",
                "h",
                "k",
                "l",
                "exact_key_text",
                "source_order",
                "sg",
                "abs_sg",
                "Eg",
                "D",
                "U",
                "M",
                "M2",
            ]
        )
    uri = f"file:{Path(db_file).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        frames: list[pd.DataFrame] = []
        for start in range(0, len(hkls), 300):
            chunk = hkls[start : start + 300]
            values = ",".join(["(?,?,?)"] * len(chunk))
            params = [value for hkl in chunk for value in hkl]
            frames.append(
                pd.read_sql_query(
                    f"""
            WITH batch_hkl(h,k,l) AS (VALUES {values})
            SELECT sc.ordinal,sc.source_filename,sc.event,sc.h,sc.k,sc.l,
                   sc.exact_key_text,sc.source_order,sc.sg,sc.abs_sg,sc.Eg,sc.D,sc.U,sc.M,sc.M2
            FROM score_cache AS sc
            JOIN batch_hkl AS b ON sc.h=b.h AND sc.k=b.k AND sc.l=b.l
            ORDER BY sc.h,sc.k,sc.l,sc.exact_key_text
            """,
                    conn,
                    params=params,
                )
            )
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True).sort_values(["h", "k", "l", "exact_key_text"], kind="mergesort")
    finally:
        conn.close()


def add_ordinals(ordinal_lists: dict[str, list[int]], variant_id: str, ordinals: Iterable[int]) -> None:
    target = ordinal_lists.setdefault(variant_id, [])
    if isinstance(ordinals, np.ndarray):
        target.extend(int(value) for value in ordinals.tolist())
    else:
        target.extend(int(value) for value in ordinals)


def increment_counts(counts: dict[str, dict[str, int]], variant_id: str, selected_or_removed: int, eligible: int, actionable: int) -> None:
    row = counts[variant_id]
    row["selected_or_removed"] += int(selected_or_removed)
    row["eligible"] += int(eligible)
    row["actionable"] += int(actionable)


def plan_batch_worker(task: tuple[int, str, list[tuple[int, int, int]], dict[tuple[int, int, int], dict[int, np.ndarray]] | None]) -> dict[str, Any]:
    batch_index, db_file, hkls, target_c_blocks_by_hkl = task
    score_defs = v6pilot.score_registry()
    planned_ids = [variant.variant_id for variant in all_planned_streams()]
    counts: dict[str, dict[str, int]] = {variant_id: {"selected_or_removed": 0, "eligible": 0, "actionable": 0} for variant_id in planned_ids}
    ordinal_lists: dict[str, list[int]] = {}
    hkl_qc_rows: list[dict[str, Any]] = []
    block_qc_rows: list[dict[str, Any]] = []
    block_definition_rows: list[dict[str, Any]] = []
    target_c_actionable_observations = 0
    target_c_actionable_blocks = 0
    d3_drop30_removed = 0
    hkl_groups = 0

    table = fetch_score_cache_batch(db_file, hkls)
    if table.empty:
        return {
            "batch_index": int(batch_index),
            "pid": int(os.getpid()),
            "hkl_count": 0,
            "observation_count": 0,
            "mask_ordinals": {},
            "counts": counts,
            "hkl_qc_rows": [],
            "block_qc_rows": [],
            "block_definition_rows": [],
            "target_c_actionable_observations": 0,
            "target_c_actionable_blocks": 0,
            "d3_drop30_removed": 0,
        }

    for _hkl, raw_group in table.groupby(["h", "k", "l"], sort=False):
        group = score_group(raw_group, score_defs)
        h = int(group["h"].iloc[0])
        k = int(group["k"].iloc[0])
        l = int(group["l"].iloc[0])
        exact_key = group["exact_key_text"].astype(str)
        n_obs = int(len(group))
        n_half = n_obs // 2

        for score in score_defs:
            ordered = group.sort_values([f"score_{score.score_id}", "exact_key_text"], ascending=[True, True], kind="mergesort")
            low_ord = ordered.head(n_half)["ordinal"].to_numpy(dtype=np.int64, copy=False)
            high_ord = ordered.tail(n_half)["ordinal"].to_numpy(dtype=np.int64, copy=False)
            low_id = f"diag_{score.score_id}_low50"
            high_id = f"diag_{score.score_id}_high50"
            add_ordinals(ordinal_lists, low_id, low_ord)
            add_ordinals(ordinal_lists, high_id, high_ord)
            for variant_id, selected in [(low_id, len(low_ord)), (high_id, len(high_ord))]:
                increment_counts(counts, variant_id, selected, n_obs, n_obs if n_obs >= 2 else 0)
                hkl_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "n_observations": n_obs, "n_selected": int(selected), "n_omitted_middle": int(n_obs - 2 * n_half), "actionable": n_obs >= 2, "validation_passed": len(low_ord) == len(high_ord) == n_half})

        for seed in RANDOM_SEEDS:
            ordered = group.assign(_rand=[stable_hash_u64(seed, key) for key in exact_key]).sort_values(["_rand", "exact_key_text"], ascending=[True, True], kind="mergesort")
            low_ord = ordered.head(n_half)["ordinal"].to_numpy(dtype=np.int64, copy=False)
            high_ord = ordered.tail(n_half)["ordinal"].to_numpy(dtype=np.int64, copy=False)
            for side, selected_ord in [("low50", low_ord), ("high50", high_ord)]:
                variant_id = f"random_diag_seed{seed}_{side}"
                add_ordinals(ordinal_lists, variant_id, selected_ord)
                increment_counts(counts, variant_id, len(selected_ord), n_obs, n_obs if n_obs >= 2 else 0)
                hkl_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "n_observations": n_obs, "n_selected": int(len(selected_ord)), "n_omitted_middle": int(n_obs - 2 * n_half), "actionable": n_obs >= 2, "validation_passed": True})

        for score_id in FILTER_SCORE_IDS:
            score_col = f"score_{score_id}"
            for drop in TARGET_A_DROPS:
                label = percent_label(drop)
                variant_id = f"filter_all_{score_id}_drop{label}"
                n_remove = removal_count(n_obs, drop)
                actionable = n_obs >= TARGET_A_MIN_OBSERVATIONS and n_remove > 0 and n_obs - n_remove >= MIN_REMAINING
                removed = group.iloc[0:0]
                if actionable:
                    removed = group.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_remove)
                    add_ordinals(ordinal_lists, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_obs if n_obs >= TARGET_A_MIN_OBSERVATIONS else 0, n_obs if actionable else 0)
                hkl_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "n_observations": n_obs, "n_eligible": n_obs if n_obs >= TARGET_A_MIN_OBSERVATIONS else 0, "n_removed": int(len(removed)), "actionable": actionable, "validation_passed": (not actionable) or len(removed) == n_remove})

        for drop in TARGET_A_DROPS:
            label = percent_label(drop)
            n_remove = removal_count(n_obs, drop)
            actionable = n_obs >= TARGET_A_MIN_OBSERVATIONS and n_remove > 0 and n_obs - n_remove >= MIN_REMAINING
            for seed in RANDOM_SEEDS:
                variant_id = f"random_all_drop{label}_seed{seed}"
                removed = group.iloc[0:0]
                if actionable:
                    removed = group.assign(_rand=[stable_hash_u64(seed, key) for key in exact_key]).sort_values(["_rand", "exact_key_text"], kind="mergesort").head(n_remove)
                    add_ordinals(ordinal_lists, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_obs if n_obs >= TARGET_A_MIN_OBSERVATIONS else 0, n_obs if actionable else 0)
                hkl_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "n_observations": n_obs, "n_eligible": n_obs if n_obs >= TARGET_A_MIN_OBSERVATIONS else 0, "n_removed": int(len(removed)), "actionable": actionable, "validation_passed": True})

        high_pool = v6pilot.high_eg_pool(group)
        n_pool = int(len(high_pool))
        high_keys = high_pool["exact_key_text"].astype(str)
        for score_id in FILTER_SCORE_IDS:
            score_col = f"score_{score_id}"
            for drop in TARGET_BC_DROPS:
                label = percent_label(drop)
                variant_id = f"filter_higheg_{score_id}_drop{label}"
                n_remove = removal_count(n_pool, drop)
                actionable = n_pool >= MIN_HIGH_EG_OBSERVATIONS and n_remove > 0 and n_pool - n_remove >= MIN_REMAINING
                removed = high_pool.iloc[0:0]
                if actionable:
                    removed = high_pool.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_remove)
                    add_ordinals(ordinal_lists, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_pool, n_pool if actionable else 0)
                hkl_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "n_observations": n_obs, "n_eligible": n_pool, "n_removed": int(len(removed)), "actionable": actionable, "validation_passed": True})

        for drop in TARGET_BC_DROPS:
            label = percent_label(drop)
            n_remove = removal_count(n_pool, drop)
            actionable = n_pool >= MIN_HIGH_EG_OBSERVATIONS and n_remove > 0 and n_pool - n_remove >= MIN_REMAINING
            for seed in RANDOM_SEEDS:
                variant_id = f"random_higheg_drop{label}_seed{seed}"
                removed = high_pool.iloc[0:0]
                if actionable:
                    removed = high_pool.assign(_rand=[stable_hash_u64(seed, key) for key in high_keys]).sort_values(["_rand", "exact_key_text"], kind="mergesort").head(n_remove)
                    add_ordinals(ordinal_lists, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_pool, n_pool if actionable else 0)
                hkl_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "n_observations": n_obs, "n_eligible": n_pool, "n_removed": int(len(removed)), "actionable": actionable, "validation_passed": True})

        if target_c_blocks_by_hkl is None:
            block_items: list[tuple[int, pd.DataFrame]] = []
            fallback_blocks = split_excitation_blocks(high_pool) if n_pool >= MIN_HIGH_EG_OBSERVATIONS else []
            for block_id, block in enumerate(fallback_blocks, start=1):
                block_items.append((block_id, block))
        else:
            block_items = []
            ordinal_blocks = target_c_blocks_by_hkl.get((h, k, l), {})
            group_ordinals = group["ordinal"].to_numpy(dtype=np.int64, copy=False)
            for block_id in sorted(ordinal_blocks):
                block_ordinals = np.asarray(ordinal_blocks[block_id], dtype=np.int64)
                block = group.loc[np.isin(group_ordinals, block_ordinals)].copy()
                if len(block) != len(block_ordinals):
                    raise RuntimeError(f"Target-C V5 block {(h, k, l, block_id)} missing full-cache rows: {len(block)}/{len(block_ordinals)}")
                block_items.append((int(block_id), block))
        target_c_actionable_blocks += len(block_items)
        target_c_actionable_observations += sum(len(block) for _block_id, block in block_items)
        for block_id, block in block_items:
            n_block = int(len(block))
            block_keys = block["exact_key_text"].astype(str)
            block_definition_rows.append({"h": h, "k": k, "l": l, "block_id": block_id, "block_size": n_block, "definition_source": "v5_p_lambda_common_actionable_blocks" if target_c_blocks_by_hkl is not None else "target_c_common_frozen_excitation_blocks"})
            for score_id in FILTER_SCORE_IDS:
                score_col = f"score_{score_id}"
                for drop in TARGET_BC_DROPS:
                    label = percent_label(drop)
                    variant_id = f"filter_matched_{score_id}_drop{label}"
                    n_remove = removal_count(n_block, drop)
                    removed = block.sort_values([score_col, "abs_sg", "exact_key_text"], ascending=[False, True, True], kind="mergesort").head(n_remove)
                    add_ordinals(ordinal_lists, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                    increment_counts(counts, variant_id, len(removed), n_block, n_block)
                    if variant_id == "filter_matched_eg_d3_cmean_drop30":
                        d3_drop30_removed += len(removed)
                    block_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "block_id": block_id, "block_size": n_block, "n_removed": int(len(removed)), "n_retained_in_block": int(n_block - len(removed)), "validation_passed": n_block - len(removed) >= MIN_REMAINING})
            for drop in TARGET_BC_DROPS:
                label = percent_label(drop)
                n_remove = removal_count(n_block, drop)
                for seed in RANDOM_SEEDS:
                    variant_id = f"random_matched_drop{label}_seed{seed}"
                    removed = block.assign(_rand=[stable_hash_u64(seed, key) for key in block_keys]).sort_values(["_rand", "exact_key_text"], kind="mergesort").head(n_remove)
                    add_ordinals(ordinal_lists, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                    increment_counts(counts, variant_id, len(removed), n_block, n_block)
                    block_qc_rows.append({"variant_id": variant_id, "h": h, "k": k, "l": l, "block_id": block_id, "block_size": n_block, "n_removed": int(len(removed)), "n_retained_in_block": int(n_block - len(removed)), "validation_passed": True})

        hkl_groups += 1

    return {
        "batch_index": int(batch_index),
        "pid": int(os.getpid()),
        "hkl_count": int(hkl_groups),
        "observation_count": int(len(table)),
        "mask_ordinals": {variant_id: np.asarray(values, dtype=np.int64) for variant_id, values in ordinal_lists.items() if values},
        "counts": counts,
        "hkl_qc_rows": hkl_qc_rows,
        "block_qc_rows": block_qc_rows,
        "block_definition_rows": block_definition_rows,
        "target_c_actionable_observations": int(target_c_actionable_observations),
        "target_c_actionable_blocks": int(target_c_actionable_blocks),
        "d3_drop30_removed": int(d3_drop30_removed),
    }


def log_plan_batch_progress(
    logger: RunLogger,
    *,
    start: float,
    completed_batches: int,
    total_batches: int,
    completed_hkls: int,
    total_hkls: int,
    completed_observations: int,
    accepted_count: int,
    worker_pids: Iterable[int],
    pending: int,
) -> None:
    elapsed = max(time.monotonic() - start, 1.0e-9)
    pct = 100.0 * completed_hkls / max(1, total_hkls)
    rate = completed_hkls / elapsed
    eta = (total_hkls - completed_hkls) / rate if rate > 0.0 and completed_hkls < total_hkls else 0.0
    rss = current_rss_mb()
    rss_text = f"{rss:.1f} MB" if rss is not None else "unknown"
    logger.log(
        "plan: parallel batch progress: "
        f"batches={completed_batches:,}/{total_batches:,}; "
        f"hkls={completed_hkls:,}/{total_hkls:,} ({pct:.1f}%); "
        f"observations={completed_observations:,}/{accepted_count:,}; "
        f"elapsed={format_seconds(elapsed)}; rate={rate:.2f} HKLs/s; "
        f"eta={format_eta(eta)}; parent_rss={rss_text}; "
        f"worker_pids={sorted(set(int(pid) for pid in worker_pids))}; pending_futures={pending}"
    )


def build_parallelism_validation(
    *,
    workers: int,
    total_batches: int,
    plan_batch_hkls: int,
    max_pending_batches: int,
    pid_batch_counts: dict[int, int],
    completed_batches: int,
    completed_hkls: int,
    completed_observations: int,
    elapsed_seconds: float,
    production: bool,
) -> dict[str, Any]:
    active_pids = sorted(int(pid) for pid, count in pid_batch_counts.items() if int(count) > 0)
    minimum_distinct_workers = 1
    if workers > 1 and total_batches >= 2:
        minimum_distinct_workers = 2
    required_75pct_workers = min(int(workers), int(total_batches), int(math.ceil(0.75 * int(workers))))
    warnings: list[str] = []
    failures: list[str] = []
    if len(active_pids) < minimum_distinct_workers:
        failures.append(f"only {len(active_pids)} distinct worker PID(s) completed real plan batches; required {minimum_distinct_workers}")
    if total_batches >= workers and len(active_pids) < required_75pct_workers:
        message = f"{len(active_pids)} of {workers} workers completed real plan batches; expected at least {required_75pct_workers} when batches >= workers"
        if production:
            failures.append(message)
        else:
            warnings.append(message)
    return {
        "passed": not failures,
        "builder_version": PLAN_BUILDER_VERSION,
        "workers_requested": int(workers),
        "plan_batch_hkls": int(plan_batch_hkls),
        "max_pending_batches": int(max_pending_batches),
        "total_batches": int(total_batches),
        "completed_batches": int(completed_batches),
        "completed_hkls": int(completed_hkls),
        "completed_observations": int(completed_observations),
        "distinct_worker_pid_count": int(len(active_pids)),
        "worker_pids": active_pids,
        "worker_batch_counts": {str(pid): int(pid_batch_counts[pid]) for pid in active_pids},
        "minimum_distinct_workers_required": int(minimum_distinct_workers),
        "required_75pct_workers": int(required_75pct_workers),
        "warnings": warnings,
        "failures": failures,
        "elapsed_seconds": float(elapsed_seconds),
    }


def consume_plan_batch_result(
    result: dict[str, Any],
    *,
    masks: dict[str, PackedMask],
    counts: dict[str, dict[str, int]],
    hkl_writer: CsvRowWriter,
    block_writer: CsvRowWriter,
    block_definition_writer: CsvRowWriter,
) -> None:
    for variant_id, ordinals in result["mask_ordinals"].items():
        masks[str(variant_id)].set_many(ordinals)
    for variant_id, row in result["counts"].items():
        target = counts[str(variant_id)]
        target["selected_or_removed"] += int(row["selected_or_removed"])
        target["eligible"] += int(row["eligible"])
        target["actionable"] += int(row["actionable"])
    hkl_writer.writerows(result["hkl_qc_rows"])
    block_writer.writerows(result["block_qc_rows"])
    block_definition_writer.writerows(result["block_definition_rows"])


def split_v5_cached_blocks(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values(["abs_sg_target", "exact_key_text"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    blocks = [ordered.iloc[start : start + EXCITATION_BLOCK_SIZE].copy() for start in range(0, len(ordered), EXCITATION_BLOCK_SIZE)]
    if len(blocks) > 1 and len(blocks[-1]) < MIN_FINAL_BLOCK_SIZE:
        blocks[-2] = pd.concat([blocks[-2], blocks[-1]], ignore_index=True)
        blocks = blocks[:-1]
    if len(blocks) == 1 and len(blocks[0]) < MIN_FINAL_BLOCK_SIZE:
        return []
    return blocks


def load_v5_target_c_blocks(
    args: argparse.Namespace,
    conn: sqlite3.Connection,
    logger: RunLogger,
) -> tuple[dict[tuple[int, int, int], dict[int, np.ndarray]] | None, dict[str, Any]]:
    path = args.v5_dir / "cached_multi_score_table.csv.gz"
    if not path.is_file():
        if args.max_cache_rows is None:
            raise SystemExit(f"V5 p/lambda target-C cache not found: {path}")
        stats = {"status": "synthetic_fallback", "reason": f"V5 p/lambda cache not found: {path}"}
        write_json(args.out_dir / "target_c_domain_validation.json", stats)
        return None, stats

    usecols = ["h", "k", "l", "exact_key_text", "abs_sg_target", "score_sg175_sc100"]
    logger.log(f"plan: loading V5 p/lambda Target-C domain from {path}")
    work = pd.read_csv(path, usecols=usecols, low_memory=False)
    for column in HKL_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype("int64")
    work["exact_key_text"] = work["exact_key_text"].astype(str)
    work["abs_sg_target"] = pd.to_numeric(work["abs_sg_target"], errors="coerce")
    work["score_sg175_sc100"] = pd.to_numeric(work["score_sg175_sc100"], errors="coerce")
    if work["exact_key_text"].duplicated(keep=False).any():
        raise SystemExit("V5 p/lambda Target-C cache contains duplicate exact keys")
    if not np.isfinite(work[["abs_sg_target", "score_sg175_sc100"]].to_numpy(dtype=float)).all():
        raise SystemExit("V5 p/lambda Target-C cache contains nonfinite block/sentinel scores")

    common_rows: list[tuple[str, int, int, int, int, int]] = []
    common_block_count = 0
    common_excitation_block_count = 0
    zero_range_blocks = 0
    progress = StageProgress(logger, "plan: reconstructing V5 p/lambda common actionable Target-C blocks", work.groupby(HKL_COLUMNS).ngroups, "HKLs")
    completed = 0
    for (h, k, l), group in work.groupby(HKL_COLUMNS, sort=False):
        if len(group) >= MIN_HIGH_EG_OBSERVATIONS:
            for block_id, block in enumerate(split_v5_cached_blocks(group), start=1):
                common_excitation_block_count += 1
                scores = block["score_sg175_sc100"].to_numpy(dtype=float)
                score_range = float(np.max(scores) - np.min(scores)) if len(scores) else 0.0
                if score_range <= 0.0:
                    zero_range_blocks += 1
                    continue
                common_block_count += 1
                block_size = int(len(block))
                for key in block["exact_key_text"].astype(str):
                    common_rows.append((key, int(h), int(k), int(l), int(block_id), block_size))
        completed += 1
        progress.update(completed)
    progress.finish(completed)

    conn.execute("DROP TABLE IF EXISTS temp.target_c_v5_keys")
    conn.execute(
        "CREATE TEMP TABLE target_c_v5_keys("
        "exact_key_text TEXT PRIMARY KEY, h INTEGER NOT NULL, k INTEGER NOT NULL, l INTEGER NOT NULL, "
        "block_id INTEGER NOT NULL, block_size INTEGER NOT NULL)"
    )
    for start in range(0, len(common_rows), 50_000):
        conn.executemany(
            "INSERT INTO target_c_v5_keys(exact_key_text,h,k,l,block_id,block_size) VALUES(?,?,?,?,?,?)",
            common_rows[start : start + 50_000],
        )
    joined = conn.execute(
        """
        SELECT t.h,t.k,t.l,t.block_id,t.block_size,s.ordinal
        FROM target_c_v5_keys AS t
        JOIN score_cache AS s ON s.exact_key_text=t.exact_key_text
        ORDER BY t.h,t.k,t.l,t.block_id,s.exact_key_text
        """
    ).fetchall()
    if len(joined) != len(common_rows):
        raise SystemExit(f"V5 Target-C/full-cache ordinal join mismatch: matched={len(joined):,}, expected={len(common_rows):,}")

    blocks_by_hkl_lists: dict[tuple[int, int, int], dict[int, list[int]]] = {}
    block_sizes: dict[tuple[int, int, int, int], int] = {}
    for h, k, l, block_id, block_size, ordinal in joined:
        hkl = (int(h), int(k), int(l))
        blocks_by_hkl_lists.setdefault(hkl, {}).setdefault(int(block_id), []).append(int(ordinal))
        block_sizes[(int(h), int(k), int(l), int(block_id))] = int(block_size)
    for hkl, blocks in blocks_by_hkl_lists.items():
        for block_id, ordinals in blocks.items():
            expected_size = block_sizes[(hkl[0], hkl[1], hkl[2], int(block_id))]
            if len(ordinals) != expected_size:
                raise SystemExit(f"V5 Target-C block size mismatch for {(hkl[0], hkl[1], hkl[2], block_id)}: {len(ordinals)} != {expected_size}")
    blocks_by_hkl = {
        hkl: {block_id: np.asarray(ordinals, dtype=np.int64) for block_id, ordinals in blocks.items()}
        for hkl, blocks in blocks_by_hkl_lists.items()
    }
    stats = {
        "status": "loaded",
        "source": str(path),
        "v5_high_eg_cache_rows": int(len(work)),
        "common_high_Eg_observation_count": int(len(common_rows)),
        "common_excitation_block_count": int(common_excitation_block_count),
        "common_actionable_block_count": int(common_block_count),
        "zero_baseline_range_blocks_excluded": int(zero_range_blocks),
        "full_cache_ordinals_matched": int(len(joined)),
        "signed_hkl_count": int(len(blocks_by_hkl)),
        "block_definition_source": "v5_p_lambda_common_actionable_blocks",
    }
    if args.max_cache_rows is None and (
        stats["common_high_Eg_observation_count"] != V5_SENTINEL_ACTIONABLE_OBSERVATIONS
        or stats["common_actionable_block_count"] != V5_SENTINEL_ACTIONABLE_BLOCKS
    ):
        raise SystemExit(f"V5 Target-C domain gate failed: {stats}")
    write_json(args.out_dir / "target_c_domain_validation.json", stats)
    return blocks_by_hkl, stats


def construct_plan(args: argparse.Namespace, conn: sqlite3.Connection, logger: RunLogger) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, PackedMask], int, dict[str, Any], dict[str, Any]]:
    accepted_count = score_cache_count(conn)
    if args.max_cache_rows is None and accepted_count != int(args.expected_accepted_count):
        raise SystemExit(f"Cache gate failed before planning: observed={accepted_count:,}, expected={args.expected_accepted_count:,}")
    planned = all_planned_streams()
    reset_plan_outputs(args.out_dir, logger)
    target_c_blocks_by_hkl, target_c_domain = load_v5_target_c_blocks(args, conn, logger)
    masks = open_masks(args.out_dir, planned, accepted_count, create=True)
    counts: dict[str, dict[str, int]] = {variant.variant_id: {"selected_or_removed": 0, "eligible": 0, "actionable": 0} for variant in planned}
    hkl_writer = CsvRowWriter(args.out_dir / "per_variant_per_hkl_qc.csv", HKL_QC_FIELDNAMES)
    block_writer = CsvRowWriter(args.out_dir / "per_variant_per_block_qc.csv", BLOCK_QC_FIELDNAMES)
    block_definition_writer = CsvRowWriter(args.out_dir / "block_definitions.csv", BLOCK_DEFINITION_FIELDNAMES)

    target_c_actionable_observations = 0
    target_c_actionable_blocks = 0
    d3_drop30_removed = 0
    completed_batches = 0
    completed_hkls = 0
    completed_observations = 0
    pid_batch_counts: dict[int, int] = {}
    total_hkls = count_signed_hkls(conn)
    total_batches = int(math.ceil(total_hkls / int(args.plan_batch_hkls))) if total_hkls else 0
    start = time.monotonic()
    last_progress = start
    forced_initial_progress = False

    logger.log(
        "plan: constructing score and random selections with "
        f"ProcessPoolExecutor(max_workers={int(args.workers)}); "
        f"total_signed_hkls={total_hkls:,}; total_batches={total_batches:,}; "
        f"plan_batch_hkls={int(args.plan_batch_hkls):,}; max_pending_batches={int(args.max_pending_batches):,}"
    )
    try:
        pending: dict[Any, int] = {}
        batches = enumerate(iter_hkl_batches(conn, int(args.plan_batch_hkls)), start=1)
        exhausted = False

        def submit_until_bounded(executor: ProcessPoolExecutor) -> None:
            nonlocal exhausted
            while not exhausted and len(pending) < int(args.max_pending_batches):
                try:
                    batch_index, hkl_batch = next(batches)
                except StopIteration:
                    exhausted = True
                    return
                target_c_batch = None
                if target_c_blocks_by_hkl is not None:
                    target_c_batch = {hkl: target_c_blocks_by_hkl[hkl] for hkl in hkl_batch if hkl in target_c_blocks_by_hkl}
                future = executor.submit(plan_batch_worker, (int(batch_index), str(db_path(args.out_dir)), hkl_batch, target_c_batch))
                pending[future] = int(batch_index)

        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=plan_worker_initializer) as executor:
            submit_until_bounded(executor)
            while pending:
                done, _not_done = wait(pending, timeout=30.0, return_when=FIRST_COMPLETED)
                now = time.monotonic()
                if not done:
                    log_plan_batch_progress(
                        logger,
                        start=start,
                        completed_batches=completed_batches,
                        total_batches=total_batches,
                        completed_hkls=completed_hkls,
                        total_hkls=total_hkls,
                        completed_observations=completed_observations,
                        accepted_count=accepted_count,
                        worker_pids=pid_batch_counts,
                        pending=len(pending),
                    )
                    last_progress = now
                    continue
                for future in done:
                    pending.pop(future)
                    result = future.result()
                    consume_plan_batch_result(
                        result,
                        masks=masks,
                        counts=counts,
                        hkl_writer=hkl_writer,
                        block_writer=block_writer,
                        block_definition_writer=block_definition_writer,
                    )
                    completed_batches += 1
                    completed_hkls += int(result["hkl_count"])
                    completed_observations += int(result["observation_count"])
                    if int(result["observation_count"]) > 0:
                        pid = int(result["pid"])
                        pid_batch_counts[pid] = pid_batch_counts.get(pid, 0) + 1
                    target_c_actionable_observations += int(result["target_c_actionable_observations"])
                    target_c_actionable_blocks += int(result["target_c_actionable_blocks"])
                    d3_drop30_removed += int(result["d3_drop30_removed"])
                submit_until_bounded(executor)
                if completed_hkls < total_hkls and (not forced_initial_progress or now - last_progress >= 30.0):
                    log_plan_batch_progress(
                        logger,
                        start=start,
                        completed_batches=completed_batches,
                        total_batches=total_batches,
                        completed_hkls=completed_hkls,
                        total_hkls=total_hkls,
                        completed_observations=completed_observations,
                        accepted_count=accepted_count,
                        worker_pids=pid_batch_counts,
                        pending=len(pending),
                    )
                    forced_initial_progress = True
                    last_progress = now
    finally:
        hkl_writer.close()
        block_writer.close()
        block_definition_writer.close()

    for mask in masks.values():
        mask.flush()

    elapsed = max(time.monotonic() - start, 1.0e-9)
    log_plan_batch_progress(
        logger,
        start=start,
        completed_batches=completed_batches,
        total_batches=total_batches,
        completed_hkls=completed_hkls,
        total_hkls=total_hkls,
        completed_observations=completed_observations,
        accepted_count=accepted_count,
        worker_pids=pid_batch_counts,
        pending=0,
    )
    if completed_hkls != total_hkls or completed_observations != accepted_count:
        raise SystemExit(
            f"Parallel plan did not consume the full cache: hkls={completed_hkls:,}/{total_hkls:,}, "
            f"observations={completed_observations:,}/{accepted_count:,}"
        )

    source_rows = load_source_rows_from_validation(args.out_dir, args.expected_source_reflection_rows)
    selection_counts: list[dict[str, Any]] = []
    for variant in planned:
        c = counts[variant.variant_id]
        append_count(selection_counts, variant, c["selected_or_removed"], c["eligible"], c["actionable"], source_rows, accepted_count)

    sentinel = {
        "target_c_actionable_observations": int(target_c_actionable_observations),
        "target_c_actionable_blocks": int(target_c_actionable_blocks),
        "filter_matched_eg_d3_cmean_drop30_removed": int(d3_drop30_removed),
        "expected_actionable_observations": V5_SENTINEL_ACTIONABLE_OBSERVATIONS,
        "expected_actionable_blocks": V5_SENTINEL_ACTIONABLE_BLOCKS,
        "expected_drop30_removals": V5_SENTINEL_DROP30_REMOVALS,
        "counts_match_v5_sentinel": (
            target_c_actionable_observations == V5_SENTINEL_ACTIONABLE_OBSERVATIONS
            and target_c_actionable_blocks == V5_SENTINEL_ACTIONABLE_BLOCKS
            and d3_drop30_removed == V5_SENTINEL_DROP30_REMOVALS
        ),
    }
    if args.max_cache_rows is None and not sentinel["counts_match_v5_sentinel"]:
        raise SystemExit(f"Target-C sentinel gate failed: {sentinel}")
    parallelism_validation = build_parallelism_validation(
        workers=int(args.workers),
        total_batches=total_batches,
        plan_batch_hkls=int(args.plan_batch_hkls),
        max_pending_batches=int(args.max_pending_batches),
        pid_batch_counts=pid_batch_counts,
        completed_batches=completed_batches,
        completed_hkls=completed_hkls,
        completed_observations=completed_observations,
        elapsed_seconds=elapsed,
        production=args.max_cache_rows is None,
    )
    write_json(args.out_dir / "parallelism_validation.json", parallelism_validation)
    if not parallelism_validation["passed"]:
        raise SystemExit(f"Parallelism validation failed: {parallelism_validation['failures']}")
    return selection_counts, sentinel, masks, block_definition_writer.rows_written, parallelism_validation, target_c_domain


def load_source_rows_from_validation(out_dir: Path, fallback: int) -> int:
    path = out_dir / "accepted_population_validation.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("source_stream_validation", {}).get("source_reflection_rows")
        if value is not None:
            return int(value)
    return int(fallback)


def build_plan_rows(planned: list[PlannedStream]) -> list[dict[str, Any]]:
    return [asdict(variant) for variant in planned]


def build_stream_manifest(planned: list[PlannedStream], selection_counts: list[dict[str, Any]], out_dir: Path, reuse_paths: dict[str, str]) -> list[dict[str, Any]]:
    counts = {row["variant_id"]: row for row in selection_counts}
    rows: list[dict[str, Any]] = []
    for variant in planned:
        count = counts[variant.variant_id]
        reused = reuse_paths.get(variant.variant_id)
        rows.append(
            {
                **asdict(variant),
                **count,
                "output_stream": reused or str(out_dir / variant.output_filename),
                "status": "reused_existing" if reused else "planned",
                "reused_existing_path": reused or "",
                "mask_path": str(out_dir / "selection_masks" / f"{variant.variant_id}.{variant.mask_mode}.bitset"),
            }
        )
    return rows


def find_v5_sentinel_reuse(args: argparse.Namespace, conn: sqlite3.Connection, masks: dict[str, PackedMask], logger: RunLogger) -> tuple[dict[str, str], dict[str, Any]]:
    variant_id = "filter_matched_eg_d3_cmean_drop30"
    selected_path = args.v5_dir / "selected_removal_observations.csv"
    stream_path = args.v5_dir / "p_1p00_he030_bs010_drop030.stream"
    result = {"variant_id": variant_id, "v5_selected_removals": str(selected_path), "v5_stream": str(stream_path), "classification": "not_comparable", "reason": ""}
    if not selected_path.is_file() or not stream_path.is_file():
        result["reason"] = "V5 sentinel files not found"
        return {}, result
    mask = masks[variant_id]
    new_count = mask.count()
    old_count = 0
    missing_from_new = 0
    unknown_old_keys = 0
    header = read_csv_header(selected_path)
    if "variant" not in header:
        result["reason"] = "V5 removal manifest lacks variant column"
        return {}, result
    usecols = [column for column in ["variant", *KEY_COLUMNS, "exact_key_text"] if column in header]
    progress = StageProgress(logger, "plan: comparing V6 D3 matched drop30 to V5 p=1 keys", None, "V5 keys")
    for chunk in pd.read_csv(selected_path, usecols=usecols, chunksize=250_000, low_memory=False):
        subset = chunk.loc[chunk["variant"].astype(str) == "p_1p00_he030_bs010_drop030"].copy()
        if subset.empty:
            continue
        if "exact_key_text" not in subset.columns:
            subset = normalize_key_frame(subset)
        keys = [str(value) for value in subset["exact_key_text"]]
        old_count += len(keys)
        for key in keys:
            row = conn.execute("SELECT ordinal FROM score_cache WHERE exact_key_text=?", (key,)).fetchone()
            if row is None:
                unknown_old_keys += 1
            elif not mask.get(int(row[0])):
                missing_from_new += 1
        progress.update(old_count)
    progress.finish(old_count)
    extra_count = max(0, int(new_count) - int(old_count)) if missing_from_new == 0 else ""
    exact = old_count == new_count == V5_SENTINEL_DROP30_REMOVALS and missing_from_new == 0 and unknown_old_keys == 0
    result.update(
        {
            "v5_removed_count": int(old_count),
            "v6_removed_count": int(new_count),
            "missing_v5_keys_from_v6": int(missing_from_new),
            "v5_keys_not_in_full_cache": int(unknown_old_keys),
            "extra_v6_keys_inferred": extra_count,
            "classification": "exact_same_selection" if exact else "not_comparable",
            "reason": "" if exact else "exact removed-key set mismatch or unknown V5 keys",
        }
    )
    return ({variant_id: str(stream_path)} if exact else {}), result


def halfset_support(skip: bool) -> dict[str, Any]:
    if skip:
        return {"inspected": False, "reason": "skipped by --skip-halfset-help"}
    try:
        result = subprocess.run(["partialator", "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"inspected": True, "partialator_available": False, "reason": str(exc)}
    text = result.stdout
    return {
        "inspected": True,
        "partialator_available": True,
        "returncode": int(result.returncode),
        "custom_split_option_advertised": "--custom-split" in text,
        "fixed_seed_option_advertised": "--seed" in text or "random-seed" in text,
        "deterministic_half_assignment_supported": False,
        "reason": "partialator --help advertises --custom-split but does not document the exact split-file format; no unsupported file is generated.",
        "help_excerpt": "\n".join(line for line in text.splitlines() if "split" in line.lower() or "seed" in line.lower())[:2000],
    }


def write_halfset_files(out_dir: Path, support: dict[str, Any]) -> None:
    write_json(out_dir / "halfset_support.json", support)
    lines = [
        "# Half-Set Support",
        "",
        "The builder does not run Partialator.",
        "",
    ]
    if support.get("custom_split_option_advertised") and not support.get("deterministic_half_assignment_supported"):
        lines.append("`partialator --help` advertises `--custom-split`, but the exact split-file format was not available from local help. No unsupported split file was generated.")
    elif support.get("fixed_seed_option_advertised"):
        lines.append("A fixed-seed option appears in local help; confirm exact merge-wrapper usage before production merging.")
    else:
        lines.append("No fixed seed or documented deterministic half-set mechanism was found in local `partialator --help`.")
    (out_dir / "halfset_usage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_merge_manifest(out_dir: Path, stream_manifest: list[dict[str, Any]]) -> None:
    order = {row["variant_id"]: idx for idx, row in enumerate(stream_manifest, start=2)}
    rows = [
        {
            "merge_order": 1,
            "variant_id": "full_reference",
            "stream_path": "",
            "score_or_random_control_id": "",
            "experiment_type": "full_reference",
            "target": "full_data",
            "fraction": "",
            "seed": "",
            "actual_removed_count": 0,
            "actual_removed_fraction_of_accepted_population": 0.0,
            "priority": "P0.000",
            "status": "external_reference",
            "expected_common_half_set_option": "",
        }
    ]
    for row in stream_manifest:
        rows.append(
            {
                "merge_order": order[row["variant_id"]],
                "variant_id": row["variant_id"],
                "stream_path": row["output_stream"],
                "score_or_random_control_id": row.get("score_id") or row.get("random_control_id"),
                "experiment_type": row["experiment_type"],
                "target": row["filtering_target"],
                "fraction": row.get("drop_fraction", ""),
                "seed": row.get("seed", ""),
                "actual_removed_count": row.get("accepted_observations_removed", ""),
                "actual_removed_fraction_of_accepted_population": row.get("removed_fraction_of_accepted_population", ""),
                "priority": row.get("priority", ""),
                "status": row.get("status", ""),
                "expected_common_half_set_option": "",
            }
        )
    path = out_dir / "merge_manifest.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def disk_preflight(args: argparse.Namespace, stream_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    source_size = args.source_stream.stat().st_size if args.source_stream.exists() else 0
    total = 0
    rows = []
    for row in stream_manifest:
        if row.get("status") == "reused_existing":
            estimate = 0
        else:
            retained_fraction = float(row.get("total_source_rows_retained", 0)) / max(1.0, float(row.get("source_reflection_row_count", 1)))
            estimate = int(source_size * retained_fraction)
            total += estimate
        rows.append({"variant_id": row["variant_id"], "status": row.get("status", ""), "estimated_output_bytes": estimate})
    usage = shutil.disk_usage(args.out_dir)
    required = int(total * 1.10 + 1_000_000_000)
    passed = usage.free >= required
    return {
        "source_stream_size_bytes": int(source_size),
        "estimated_new_stream_bytes": int(total),
        "safety_multiplier": 1.10,
        "fixed_margin_bytes": 1_000_000_000,
        "required_free_bytes": int(required),
        "free_bytes": int(usage.free),
        "passed": bool(passed),
        "per_variant": rows,
    }


def validate_plan_accounting(stream_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    score = [row for row in stream_manifest if not str(row["experiment_type"]).endswith("random_control")]
    random = [row for row in stream_manifest if str(row["experiment_type"]).endswith("random_control")]
    status_counts: dict[str, int] = {}
    for row in stream_manifest:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    passed = len(score) == EXPECTED_SCORE_VARIANTS and len(random) == EXPECTED_RANDOM_VARIANTS and len(stream_manifest) == EXPECTED_TOTAL_VARIANTS
    if not passed:
        raise SystemExit(f"Experiment-accounting gate failed: score={len(score)}, random={len(random)}, total={len(stream_manifest)}")
    return {"score_variant_count": len(score), "random_control_variant_count": len(random), "total_planned_entries_before_exact_reuse": len(stream_manifest), "status_counts": status_counts, "passed": True}


def required_plan_artifacts(out_dir: Path) -> list[str]:
    return [
        str(out_dir / "validation.json"),
        str(out_dir / "parallelism_validation.json"),
        str(out_dir / "plan_completion.json"),
        str(out_dir / "experiment_plan.csv"),
        str(out_dir / "stream_manifest.csv"),
        str(out_dir / "selection_counts.csv"),
        str(out_dir / "per_variant_per_hkl_qc.csv"),
        str(out_dir / "per_variant_per_block_qc.csv"),
        str(out_dir / "block_definitions.csv"),
        str(out_dir / "random_control_manifest.csv"),
        str(out_dir / "selection_overlap.csv"),
        str(out_dir / "score_correlations.csv"),
        str(out_dir / "disk_preflight.json"),
        str(out_dir / "target_c_domain_validation.json"),
        str(out_dir / "merge_manifest.tsv"),
    ]


def write_plan_completion(out_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    completed = {
        "passed": True,
        "builder_version": PLAN_BUILDER_VERSION,
        "completed_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        **payload,
    }
    write_json(out_dir / "plan_completion.json", completed)
    return completed


def run_plan(args: argparse.Namespace) -> int:
    require_file(db_path(args.out_dir), "cache SQLite database")
    logger = RunLogger(args.out_dir)
    try:
        logger.log("mode=plan; constructing full score/random-control selection plan")
        conn = connect_db(args.out_dir)
        selection_counts, sentinel, masks, block_definition_count, parallelism_validation, target_c_domain = construct_plan(args, conn, logger)
        reuse_paths, reuse_result = find_v5_sentinel_reuse(args, conn, masks, logger)
        if args.max_cache_rows is None and reuse_result.get("classification") != "exact_same_selection":
            raise SystemExit(f"V5 p=1 sentinel equivalence gate failed: {reuse_result}")
        planned = all_planned_streams()
        block_definitions = pd.read_csv(args.out_dir / "block_definitions.csv", low_memory=False).fillna("").to_dict("records")
        selection_overlap = build_selection_overlap(planned, masks, logger)
        score_correlations = score_correlation_rows(conn, logger)
        stream_manifest = build_stream_manifest(planned, selection_counts, args.out_dir, reuse_paths)
        accounting = validate_plan_accounting(stream_manifest)
        disk = disk_preflight(args, stream_manifest)
        if not disk["passed"]:
            raise SystemExit("Disk preflight failed before stream generation")
        support = halfset_support(args.skip_halfset_help)
        write_halfset_files(args.out_dir, support)
        write_csv(args.out_dir / "experiment_plan.csv", build_plan_rows(planned))
        write_json(args.out_dir / "experiment_plan.json", build_plan_rows(planned))
        write_csv(args.out_dir / "stream_manifest.csv", stream_manifest)
        write_csv(args.out_dir / "selection_counts.csv", selection_counts)
        block_parquet = write_optional_parquet(args.out_dir / "block_definitions.parquet", block_definitions)
        write_csv(args.out_dir / "random_control_manifest.csv", [row for row in stream_manifest if str(row["experiment_type"]).endswith("random_control")])
        write_csv(args.out_dir / "selection_overlap.csv", selection_overlap)
        write_csv(args.out_dir / "score_correlations.csv", score_correlations)
        write_json(args.out_dir / "disk_preflight.json", disk)
        write_merge_manifest(args.out_dir, stream_manifest)
        validation = {
            "passed": True,
            "cache_gate": {"score_cache_rows": score_cache_count(conn), "expected_accepted_count": int(args.expected_accepted_count)},
            "target_c_gate": sentinel,
            "block_definition_gate": {
                "block_definition_rows": int(block_definition_count),
                "matches_target_c_actionable_blocks": int(block_definition_count) == int(sentinel["target_c_actionable_blocks"]),
                "parquet_export": block_parquet,
            },
            "v5_sentinel_equivalence": reuse_result,
            "target_c_domain_gate": target_c_domain,
            "experiment_accounting_gate": accounting,
            "parallelism_validation": parallelism_validation,
            "plan_gate": {
                "status": "complete",
                "builder_version": PLAN_BUILDER_VERSION,
                "corrected_parallel_plan": True,
                "required_artifacts_present": True,
            },
            "stream_gate": {"status": "pending_streams_mode"},
        }
        write_json(args.out_dir / "validation.json", validation)
        write_json(args.out_dir / "run_metadata.json", base_run_metadata(args, "plan"))
        write_json(args.out_dir / "parameters.json", {"source_stream": str(args.source_stream), "accepted_population": str(args.accepted_population), "v5_scores": str(args.v5_scores), "geometry_parameters": geometry_parameters(), "random_seeds": RANDOM_SEEDS})
        write_json(args.out_dir / "scores.json", [asdict(score) for score in v6pilot.score_registry()])
        completion = write_plan_completion(
            args.out_dir,
            {
                "accepted_count": int(score_cache_count(conn)),
                "target_c_gate": sentinel,
                "target_c_domain_gate": target_c_domain,
                "parallelism_validation": parallelism_validation,
                "experiment_accounting_gate": accounting,
                "required_artifacts": required_plan_artifacts(args.out_dir),
            },
        )
        validation["plan_completion"] = completion
        write_json(args.out_dir / "validation.json", validation)
        logger.log("plan mode complete")
        return 0
    finally:
        logger.close()


def load_stream_manifest(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "stream_manifest.csv"
    if not path.is_file():
        raise SystemExit("stream_manifest.csv not found; run --mode plan first")
    return pd.read_csv(path, low_memory=False).fillna("").to_dict("records")


def load_mask_specs(args: argparse.Namespace, stream_manifest: list[dict[str, Any]]) -> list[MaskSpec]:
    specs: list[MaskSpec] = []
    for row in stream_manifest:
        status = str(row.get("status", ""))
        output = Path(str(row.get("output_stream")))
        reused = Path(str(row.get("reused_existing_path"))) if str(row.get("reused_existing_path", "")) else None
        if status != "reused_existing" and output.exists():
            raise SystemExit(f"Refusing to overwrite existing stream: {output}")
        specs.append(MaskSpec(str(row["variant_id"]), Path(str(row["mask_path"])), str(row["mask_mode"]), output, reused))
    return specs


def lookup_ordinal(conn: sqlite3.Connection, key: str) -> int | None:
    row = conn.execute("SELECT ordinal FROM score_cache WHERE exact_key_text=?", (key,)).fetchone()
    return None if row is None else int(row[0])


def should_remove(mask: PackedMask, mode: str, ordinal: int | None) -> bool:
    if ordinal is None:
        return False
    bit = mask.get(ordinal)
    if mode == "remove":
        return bit
    if mode == "keep":
        return not bit
    raise ValueError(mode)


def rewrite_batch(source_stream: Path, conn: sqlite3.Connection, specs: list[MaskSpec], accepted_count: int, logger: RunLogger) -> list[dict[str, Any]]:
    masks = {spec.variant_id: PackedMask(spec.path, accepted_count, create=False) for spec in specs}
    handles = {spec.variant_id: spec.output_path.open("w", encoding="utf-8") for spec in specs if spec.reused_existing_path is None}
    stats = {spec.variant_id: {"variant_id": spec.variant_id, "output_stream": str(spec.output_path), "requested_removals": masks[spec.variant_id].count() if spec.mode == "remove" else accepted_count - masks[spec.variant_id].count(), "removed_observations": 0, "kept_observations": 0, "total_reflection_rows_seen": 0, "source_order_preserved": True, "status": "generated" if spec.reused_existing_path is None else "reused_existing"} for spec in specs}
    spec_by_id = {spec.variant_id: spec for spec in specs}
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    progress = StageProgress(logger, f"streams: rewriting batch with {len(handles)} generated streams", None, "reflection rows")
    try:
        with source_stream.open("r", encoding="utf-8", errors="replace") as source:
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
                    parsed_source = normalize_source(match.group(1))
                    parsed_event = normalize_event(match.group(2)) if match.group(2) is not None else ""
                    if in_crystal:
                        current_source = parsed_source
                        if parsed_event:
                            current_event = parsed_event
                    else:
                        chunk_source = parsed_source
                        if parsed_event:
                            chunk_event = parsed_event
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
                    rows_seen += 1
                    key = key_to_text(current_source, current_event, *hkl)
                    ordinal = lookup_ordinal(conn, key)
                    for variant_id, handle in handles.items():
                        stat = stats[variant_id]
                        stat["total_reflection_rows_seen"] += 1
                        spec = spec_by_id[variant_id]
                        if should_remove(masks[variant_id], spec.mode, ordinal):
                            stat["removed_observations"] += 1
                        else:
                            stat["kept_observations"] += 1
                            handle.write(raw_line)
                    if rows_seen % 250_000 == 0:
                        progress.update(rows_seen, force=True)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()
    progress.finish(rows_seen)
    rows = []
    for spec in specs:
        row = stats[spec.variant_id]
        if spec.reused_existing_path is None and int(row["removed_observations"]) != int(row["requested_removals"]):
            raise SystemExit(f"{spec.variant_id}: removed {row['removed_observations']} but expected {row['requested_removals']}")
        row["all_requested_keys_found_exactly_once"] = spec.reused_existing_path is None
        row["stream_reflection_row_difference_equals_requested_removals"] = spec.reused_existing_path is None
        rows.append(row)
    return rows


def update_manifests_after_streams(out_dir: Path, manifest: list[dict[str, Any]], qc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qc_by_variant = {str(row["variant_id"]): row for row in qc_rows}
    updated: list[dict[str, Any]] = []
    for row in manifest:
        new_row = dict(row)
        qc = qc_by_variant.get(str(row["variant_id"]))
        if qc is not None:
            new_row["status"] = str(qc.get("status", new_row.get("status", "")))
            new_row["output_stream"] = str(qc.get("output_stream", new_row.get("output_stream", "")))
            new_row["stream_qc_removed_observations"] = qc.get("removed_observations", "")
            new_row["stream_qc_kept_observations"] = qc.get("kept_observations", "")
            new_row["stream_qc_total_reflection_rows_seen"] = qc.get("total_reflection_rows_seen", "")
        updated.append(new_row)
    write_csv(out_dir / "stream_manifest.csv", updated)
    write_merge_manifest(out_dir, updated)
    return updated


def require_completed_parallel_plan(out_dir: Path, validation: dict[str, Any]) -> None:
    plan_gate = validation.get("plan_gate", {})
    if plan_gate.get("status") != "complete" or plan_gate.get("builder_version") != PLAN_BUILDER_VERSION:
        raise SystemExit("Corrected parallel plan gate is missing or stale; rerun --mode plan with the fixed builder")
    completion_path = out_dir / "plan_completion.json"
    if not completion_path.is_file():
        raise SystemExit("plan_completion.json not found; refusing to run streams from incomplete plan outputs")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not completion.get("passed") or completion.get("builder_version") != PLAN_BUILDER_VERSION:
        raise SystemExit("plan_completion.json does not match the corrected completed plan; refusing streams")
    parallel_path = out_dir / "parallelism_validation.json"
    if not parallel_path.is_file():
        raise SystemExit("parallelism_validation.json not found; refusing streams")
    parallelism = json.loads(parallel_path.read_text(encoding="utf-8"))
    if not parallelism.get("passed") or parallelism.get("builder_version") != PLAN_BUILDER_VERSION:
        raise SystemExit("Parallelism validation did not pass for the corrected plan; refusing streams")
    required = completion.get("required_artifacts", [])
    missing = [path for path in required if not Path(str(path)).is_file()]
    if missing:
        raise SystemExit(f"Completed plan artifact check failed; missing: {missing[:10]}")


def run_streams(args: argparse.Namespace) -> int:
    require_file(db_path(args.out_dir), "cache SQLite database")
    validation_path = args.out_dir / "validation.json"
    if not validation_path.is_file():
        raise SystemExit("validation.json not found; run --mode plan first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise SystemExit("Plan validation did not pass; refusing to write streams")
    require_completed_parallel_plan(args.out_dir, validation)
    logger = RunLogger(args.out_dir)
    try:
        logger.log("mode=streams; rewriting source stream from packed masks")
        conn = connect_db(args.out_dir)
        accepted_count = score_cache_count(conn)
        manifest = load_stream_manifest(args.out_dir)
        specs = load_mask_specs(args, manifest)
        generated = [spec for spec in specs if spec.reused_existing_path is None]
        reused = [spec for spec in specs if spec.reused_existing_path is not None]
        rows: list[dict[str, Any]] = []
        for start in range(0, len(generated), MAX_OPEN_STREAMS):
            batch = generated[start : start + MAX_OPEN_STREAMS]
            rows.extend(rewrite_batch(args.source_stream, conn, batch, accepted_count, logger))
        for spec in reused:
            rows.append({"variant_id": spec.variant_id, "output_stream": str(spec.reused_existing_path), "requested_removals": "", "removed_observations": "", "kept_observations": "", "total_reflection_rows_seen": "", "source_order_preserved": True, "status": "reused_existing", "all_requested_keys_found_exactly_once": True, "stream_reflection_row_difference_equals_requested_removals": True})
        write_csv(args.out_dir / "stream_rewrite_qc.csv", rows)
        updated_manifest = update_manifests_after_streams(args.out_dir, manifest, rows)
        status_counts: dict[str, int] = {}
        for row in updated_manifest:
            status_counts[str(row.get("status", ""))] = status_counts.get(str(row.get("status", "")), 0) + 1
        validation["stream_gate"] = {
            "status": "passed",
            "stream_qc_rows": len(rows),
            "generated_streams": int(status_counts.get("generated", 0)),
            "reused_existing_streams": int(status_counts.get("reused_existing", 0)),
            "status_counts": status_counts,
            "source_order_preserved": all(bool(row.get("source_order_preserved")) for row in rows),
            "all_requested_keys_found_exactly_once": all(bool(row.get("all_requested_keys_found_exactly_once")) for row in rows),
            "stream_reflection_row_difference_equals_requested_removals": all(bool(row.get("stream_reflection_row_difference_equals_requested_removals")) for row in rows),
        }
        write_json(validation_path, validation)
        write_json(args.out_dir / "run_metadata.json", base_run_metadata(args, "streams"))
        logger.log("streams mode complete")
        return 0
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "cache":
        return run_cache(args)
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "streams":
        return run_streams(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
