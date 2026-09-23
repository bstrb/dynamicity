#!/usr/bin/env python3
"""Build directional-intensity full-population EgM2 filter streams.

This is an aggressive proof-of-concept builder.  It combines the existing
full-population EgM2 score with a directional geometry proxy and a local
excitation-weighted intensity residual.  It never runs Partialator or merging;
stream files are written only with --write-streams.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

for _env_name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_env_name, "1")

import numpy as np
import pandas as pd


DEFAULT_SOURCE_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_full_population_sweep_20260717"
)
DEFAULT_SOURCE_STREAM = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "MFM300-VIII_cut_20-0_3.stream"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_fullpop_directional_intensity_20260811"
)

FAMILY_ORDER = ["enhance_highJ", "deplete_lowJ", "directional_consistent_union", "directional_extreme"]
DEFAULT_FAMILIES = tuple(FAMILY_ORDER)
DEFAULT_FRACTIONS = ("0.01", "0.02", "0.05", "0.10", "0.15", "0.20")

SIGMA_C = 0.050
R_CUT = 0.150
ROBUST_MAD_SCALE = 1.4826
EPS = 1.0e-12
MIN_LOCAL_NEIGHBORS = 3
MIN_RETAINED_PER_HKL = 10
MAX_REMOVE_PER_HKL_FRACTION = 0.20
RISK_Z_THRESHOLD = 1.0
ENHANCE_J_THRESHOLD = 1.0
DEPLETE_J_THRESHOLD = -1.0
EXTREME_ABS_J_THRESHOLD = 2.0

PLAN_BATCH_HKLS = 250
FRAME_FUTURE_MULTIPLIER = 3
MAX_OPEN_STREAMS = 8

HKL_COLUMNS = ["h", "k", "l"]
STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")
UNIT_CELL_RE = re.compile(
    r"^\s*(a|b|c|al|be|ga)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z]+)?"
)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    family: str
    fraction_text: str
    fraction: float
    output_filename: str
    output_stream: Path


@dataclass(frozen=True)
class FrameRecord:
    source_filename: str
    event: str
    h: int
    k: int
    l: int
    intensity: float
    exact_key_text: str


class RunLogger:
    def __init__(self, out_dir: Path | None) -> None:
        self.handle = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.handle = (out_dir / "run.log").open("x", encoding="utf-8", buffering=1)

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
    def __init__(
        self,
        logger: RunLogger,
        stage: str,
        total: int | None,
        unit: str,
        workers: int,
        progress_every: float,
    ) -> None:
        self.logger = logger
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.workers = int(workers)
        self.progress_every = float(progress_every)
        self.completed = 0
        self.started = time.monotonic()
        self.last = self.started
        total_text = f"{self.total:,} {unit}" if self.total is not None else f"unknown {unit}"
        self.logger.log(f"{stage}: started ({total_text}; workers={self.workers}; rss={rss_mb():.1f} MB)")

    def update(self, completed: int, force: bool = False) -> None:
        self.completed = int(completed)
        now = time.monotonic()
        if not force and now - self.last < self.progress_every:
            return
        self.last = now
        elapsed = max(1.0e-9, now - self.started)
        rate = self.completed / elapsed
        if self.total:
            pct = 100.0 * self.completed / max(1, self.total)
            eta = (self.total - self.completed) / rate if rate > 0.0 else float("inf")
            self.logger.log(
                f"{self.stage}: {self.completed:,}/{self.total:,} {self.unit} "
                f"({pct:.1f}%), elapsed={format_duration(elapsed)}, "
                f"rate={rate:,.1f}/s, eta={format_duration(eta)}, "
                f"rss={rss_mb():.1f} MB, workers={self.workers}"
            )
        else:
            self.logger.log(
                f"{self.stage}: {self.completed:,} {self.unit}, "
                f"elapsed={format_duration(elapsed)}, rate={rate:,.1f}/s, "
                f"rss={rss_mb():.1f} MB, workers={self.workers}"
            )

    def finish(self, completed: int | None = None) -> None:
        if completed is not None:
            self.completed = int(completed)
        self.update(self.completed, force=True)
        self.logger.log(f"{self.stage}: finished")


POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


class PackedMask:
    def __init__(self, n_bits: int) -> None:
        self.n_bits = int(n_bits)
        self.array = np.zeros((self.n_bits + 7) // 8, dtype=np.uint8)

    def set_many(self, ordinals: Iterable[int] | np.ndarray) -> None:
        if isinstance(ordinals, np.ndarray):
            indexes = ordinals.astype(np.int64, copy=False)
        else:
            indexes = np.fromiter((int(value) for value in ordinals), dtype=np.int64)
        if indexes.size == 0:
            return
        if indexes.min() < 0 or indexes.max() >= self.n_bits:
            raise IndexError("Mask ordinal outside score-cache bounds")
        byte_indexes = indexes // 8
        bit_values = (1 << (indexes % 8)).astype(np.uint8)
        np.bitwise_or.at(self.array, byte_indexes, bit_values)

    def get(self, ordinal: int | None) -> bool:
        if ordinal is None:
            return False
        index = int(ordinal)
        if index < 0 or index >= self.n_bits:
            return False
        return bool(self.array[index // 8] & (1 << (index % 8)))

    def count(self) -> int:
        return int(POPCOUNT8[self.array].sum())

    def digest(self) -> str:
        return hashlib.sha256(self.array.tobytes()).hexdigest()


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None, delimiter: str = ",") -> None:
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


def key_to_text(source: Any, event: Any, h: int, k: int, l: int) -> str:
    return f"{normalize_source(source)}|{normalize_event(event)}|{int(h)}|{int(k)}|{int(l)}"


def parse_reflection_row(line: str) -> tuple[int, int, int, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
        intensity = float(parts[3])
        float(parts[4])
    except ValueError:
        return None
    if not math.isfinite(intensity):
        raise RuntimeError(f"Nonfinite reflection intensity encountered: {line[:120]}")
    return h, k, l, intensity


def stream_key_from_context(current_source: str, current_event: str, hkl: tuple[int, int, int]) -> str:
    return key_to_text(current_source, current_event, int(hkl[0]), int(hkl[1]), int(hkl[2]))


def parse_fraction_items(value: str | None) -> list[tuple[str, float]]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_FRACTIONS)
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        fraction = float(item)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise SystemExit(f"Invalid drop fraction {item!r}; expected 0 < fraction < 1")
        if item in seen:
            raise SystemExit(f"Duplicate fraction label: {item}")
        seen.add(item)
        out.append((item, fraction))
    if not out:
        raise SystemExit("--fractions must contain at least one value")
    return out


def parse_families(value: str | None) -> list[str]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_FAMILIES)
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(text).split(","):
        family = raw.strip()
        if not family:
            continue
        if family not in FAMILY_ORDER:
            raise SystemExit(f"Unknown family {family!r}; expected one of {', '.join(FAMILY_ORDER)}")
        if family not in seen:
            out.append(family)
            seen.add(family)
    if not out:
        raise SystemExit("--families must contain at least one value")
    return out


def fraction_label(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("+"):
        stripped = stripped[1:]
    return stripped.replace(".", "p").replace("-", "m")


def build_variants(families: list[str], fractions: list[tuple[str, float]], out_dir: Path) -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for family in families:
        for fraction_text, fraction in fractions:
            variant_id = f"directional_{family}_drop{fraction_label(fraction_text)}"
            variants.append(
                VariantSpec(
                    variant_id=variant_id,
                    family=family,
                    fraction_text=fraction_text,
                    fraction=float(fraction),
                    output_filename=f"{variant_id}.stream",
                    output_stream=out_dir / f"{variant_id}.stream",
                )
            )
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--fractions", default=",".join(DEFAULT_FRACTIONS))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    parser.add_argument("--progress-every", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.family_items = parse_families(args.families)
    args.fraction_items = parse_fraction_items(args.fractions)
    args.workers = max(1, int(args.workers))
    args.progress_every = max(0.1, float(args.progress_every))
    if args.dry_run and args.write_streams:
        raise SystemExit("Use either --dry-run or --write-streams, not both")
    if not args.write_streams:
        args.dry_run = True
    if not args.source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {args.source_out_dir}")
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    return args


def work_db_path(out_dir: Path) -> Path:
    return out_dir / ".directional_intensity_work.sqlite"


def planned_output_paths(out_dir: Path, variants: list[VariantSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "directional_intensity_manifest.tsv",
        out_dir / "directional_intensity_counts.csv",
        out_dir / "directional_intensity_removed_keys.tsv.gz",
        out_dir / "directional_intensity_parameters.json",
        out_dir / "directional_intensity_metadata.json",
        out_dir / "run.log",
        work_db_path(out_dir),
    ]
    if write_streams:
        paths.extend(variant.output_stream for variant in variants)
    return paths


def prepare_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(str(path) for path in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ... {len(existing) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {preview}{extra}")
    if overwrite:
        for path in existing:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                raise SystemExit(f"Refusing to overwrite non-file output path: {path}")


def cache_db_path(source_out_dir: Path) -> Path:
    return source_out_dir / "full_population_cache.sqlite"


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def connect_work(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def cache_schema(db_file: Path) -> list[str]:
    with connect_readonly(db_file) as conn:
        return [str(row[1]) for row in conn.execute("PRAGMA table_info(score_cache)").fetchall()]


def require_cache_schema(db_file: Path) -> list[str]:
    present = cache_schema(db_file)
    required = {"ordinal", "source_filename", "event", "h", "k", "l", "exact_key_text", "source_order", "Eg", "D", "M", "M2"}
    missing = sorted(required - set(present))
    if missing:
        raise SystemExit(
            "full_population_cache.sqlite score_cache is missing required directional-intensity columns: "
            f"{missing}. M and D are required for A=M/(D+eps)."
        )
    return present


def score_cache_stats(db_file: Path) -> dict[str, int]:
    with connect_readonly(db_file) as conn:
        row = conn.execute("SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM score_cache").fetchone()
    count = int(row[0])
    min_ordinal = 0 if row[1] is None else int(row[1])
    max_ordinal = -1 if row[2] is None else int(row[2])
    if min_ordinal < 0:
        raise SystemExit(f"Negative score-cache ordinal encountered: {min_ordinal}")
    return {"count": count, "min_ordinal": min_ordinal, "max_ordinal": max_ordinal, "mask_bits": max(max_ordinal + 1, count)}


def count_signed_hkls(db_file: Path) -> int:
    with connect_readonly(db_file) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM (SELECT h,k,l FROM score_cache GROUP BY h,k,l)").fetchone()[0])


def source_reflection_row_count(source_out_dir: Path) -> int | None:
    for path in [source_out_dir / "selection_counts.csv", source_out_dir / "stream_manifest.csv"]:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in ["source_reflection_row_count", "stream_qc_total_reflection_rows_seen"]:
                    value = row.get(key)
                    if value not in (None, ""):
                        return int(float(value))
    validation = read_json(source_out_dir / "validation.json")
    for key_path in [("source_stream_validation", "source_reflection_rows"), ("cache_gate", "source_reflection_rows")]:
        current: Any = validation
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return int(current)
    return None


def init_work_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE hkl_stats (
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            n_obs INTEGER NOT NULL,
            egm2_median REAL NOT NULL,
            egm2_mad REAL NOT NULL,
            risk_denom REAL NOT NULL,
            max_remove INTEGER NOT NULL,
            PRIMARY KEY (h,k,l)
        );
        CREATE TABLE candidates (
            family TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            intensity REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL,
            egm2 REAL NOT NULL,
            risk_z REAL NOT NULL,
            geom_delta REAL NOT NULL,
            j_resid REAL NOT NULL,
            rank_score REAL NOT NULL,
            PRIMARY KEY (family, ordinal)
        );
        CREATE TABLE selected (
            variant_id TEXT NOT NULL,
            family TEXT NOT NULL,
            fraction REAL NOT NULL,
            fraction_label TEXT NOT NULL,
            selection_rank INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            intensity REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL,
            egm2 REAL NOT NULL,
            risk_z REAL NOT NULL,
            geom_delta REAL NOT NULL,
            j_resid REAL NOT NULL,
            rank_score REAL NOT NULL,
            PRIMARY KEY (variant_id, ordinal)
        );
        """
    )
    conn.commit()


def iter_hkl_batches(db_file: Path, batch_hkls: int = PLAN_BATCH_HKLS) -> Iterator[list[tuple[int, int, int]]]:
    with connect_readonly(db_file) as conn:
        cursor = conn.execute("SELECT h,k,l FROM score_cache GROUP BY h,k,l ORDER BY h,k,l")
        batch: list[tuple[int, int, int]] = []
        for h, k, l in cursor:
            batch.append((int(h), int(k), int(l)))
            if len(batch) >= int(batch_hkls):
                yield batch
                batch = []
        if batch:
            yield batch


def fetch_hkl_batch(db_file: str, hkls: list[tuple[int, int, int]]) -> pd.DataFrame:
    if not hkls:
        return pd.DataFrame(columns=["h", "k", "l", "Eg", "M2"])
    conn = connect_readonly(Path(db_file))
    try:
        frames: list[pd.DataFrame] = []
        for start in range(0, len(hkls), 300):
            chunk = hkls[start : start + 300]
            values = ",".join(["(?,?,?)"] * len(chunk))
            params = [value for hkl in chunk for value in hkl]
            frames.append(
                pd.read_sql_query(
                    f"""
                    WITH batch_hkl(h,k,l) AS (VALUES {values})
                    SELECT sc.h,sc.k,sc.l,sc.Eg,sc.M2
                    FROM score_cache AS sc
                    JOIN batch_hkl AS b ON sc.h=b.h AND sc.k=b.k AND sc.l=b.l
                    ORDER BY sc.h,sc.k,sc.l
                    """,
                    conn,
                    params=params,
                )
            )
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    finally:
        conn.close()


def worker_hkl_stats(task: tuple[str, list[tuple[int, int, int]]]) -> tuple[list[tuple[int, int, int, int, float, float, float, int]], dict[str, int]]:
    db_file, hkls = task
    table = fetch_hkl_batch(db_file, hkls)
    rows: list[tuple[int, int, int, int, float, float, float, int]] = []
    stats = {"hkl_count": 0, "observation_count": int(len(table)), "finite_count": 0}
    if table.empty:
        return rows, stats
    eg = pd.to_numeric(table["Eg"], errors="coerce").to_numpy(dtype=float)
    m2 = pd.to_numeric(table["M2"], errors="coerce").to_numpy(dtype=float)
    egm2 = eg * m2
    if not np.isfinite(egm2).all():
        raise RuntimeError("Nonfinite EgM2 encountered while computing per-HKL risk statistics")
    table = table.assign(egm2=egm2)
    for hkl, group in table.groupby(HKL_COLUMNS, sort=False):
        values = group["egm2"].to_numpy(dtype=float, copy=False)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        denom = float(ROBUST_MAD_SCALE * mad + EPS)
        n_obs = int(len(values))
        max_remove = int(max(0, min(math.floor(MAX_REMOVE_PER_HKL_FRACTION * n_obs), n_obs - MIN_RETAINED_PER_HKL)))
        rows.append((int(hkl[0]), int(hkl[1]), int(hkl[2]), n_obs, median, mad, denom, max_remove))
        stats["hkl_count"] += 1
        stats["finite_count"] += int(len(values))
    return rows, stats


def populate_hkl_stats(
    cache_db: Path,
    work_conn: sqlite3.Connection,
    workers: int,
    logger: RunLogger,
    progress_every: float,
) -> dict[str, Any]:
    total_hkls = count_signed_hkls(cache_db)
    batches = list(iter_hkl_batches(cache_db))
    progress = StageProgress(logger, "computing per-HKL EgM2 risk statistics", total_hkls, "HKLs", workers, progress_every)
    completed_hkls = 0
    completed_observations = 0
    work_conn.execute("BEGIN")
    try:
        if workers > 1 and len(batches) > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [executor.submit(worker_hkl_stats, (str(cache_db), batch)) for batch in batches]
                for future in as_completed(futures):
                    rows, stats = future.result()
                    work_conn.executemany(
                        "INSERT INTO hkl_stats(h,k,l,n_obs,egm2_median,egm2_mad,risk_denom,max_remove) VALUES(?,?,?,?,?,?,?,?)",
                        rows,
                    )
                    completed_hkls += int(stats["hkl_count"])
                    completed_observations += int(stats["observation_count"])
                    progress.update(completed_hkls)
        else:
            for batch in batches:
                rows, stats = worker_hkl_stats((str(cache_db), batch))
                work_conn.executemany(
                    "INSERT INTO hkl_stats(h,k,l,n_obs,egm2_median,egm2_mad,risk_denom,max_remove) VALUES(?,?,?,?,?,?,?,?)",
                    rows,
                )
                completed_hkls += int(stats["hkl_count"])
                completed_observations += int(stats["observation_count"])
                progress.update(completed_hkls)
        work_conn.commit()
    except Exception:
        work_conn.rollback()
        raise
    progress.finish(completed_hkls)
    work_conn.execute("CREATE INDEX idx_hkl_stats_limit ON hkl_stats(h,k,l,max_remove)")
    work_conn.commit()
    return {
        "signed_hkl_count": int(total_hkls),
        "completed_hkls": int(completed_hkls),
        "completed_observations": int(completed_observations),
        "batch_count": int(len(batches)),
    }


def parse_unit_cell_from_stream(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    in_unit_cell = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("----- Begin unit cell"):
                in_unit_cell = True
                continue
            if line.startswith("----- End unit cell"):
                break
            if not in_unit_cell:
                continue
            match = UNIT_CELL_RE.match(line)
            if not match:
                continue
            key, raw_value, unit = match.groups()
            value = float(raw_value)
            if key in {"a", "b", "c"}:
                unit_text = (unit or "A").lower()
                if unit_text == "nm":
                    value *= 10.0
                elif unit_text not in {"a", "angstrom", "angstroms"}:
                    raise SystemExit(f"Unsupported unit-cell length unit {unit!r} in {path}")
            values[key] = value
    required = {"a", "b", "c", "al", "be", "ga"}
    missing = sorted(required - set(values))
    if missing:
        raise SystemExit(
            "Could not parse a complete unit cell from the source stream; "
            f"missing {missing}. Directional local residuals require reciprocal-space dq."
        )
    return values


def reciprocal_metric_from_cell(cell: dict[str, float]) -> np.ndarray:
    a = float(cell["a"])
    b = float(cell["b"])
    c = float(cell["c"])
    alpha = math.radians(float(cell["al"]))
    beta = math.radians(float(cell["be"]))
    gamma = math.radians(float(cell["ga"]))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise SystemExit("Unit-cell gamma angle is degenerate; cannot compute reciprocal metric")
    avec = np.array([a, 0.0, 0.0], dtype=float)
    bvec = np.array([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=float)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        raise SystemExit("Unit cell gives nonpositive c-axis z component; cannot compute reciprocal metric")
    cvec = np.array([cx, cy, math.sqrt(cz_sq)], dtype=float)
    direct = np.column_stack([avec, bvec, cvec])
    reciprocal = np.linalg.inv(direct).T
    metric = reciprocal.T @ reciprocal
    if not np.isfinite(metric).all():
        raise SystemExit("Nonfinite reciprocal metric computed from source stream unit cell")
    return metric


def dq_from_delta(delta_hkl: np.ndarray, metric: np.ndarray) -> np.ndarray:
    squared = np.einsum("...i,ij,...j->...", delta_hkl.astype(float), metric, delta_hkl.astype(float))
    return np.sqrt(np.clip(squared, 0.0, None))


def coupling_kernel(dq: np.ndarray) -> np.ndarray:
    values = np.asarray(dq, dtype=float)
    kernel = np.exp(-0.5 * (values / SIGMA_C) ** 2)
    kernel = np.where(values <= R_CUT, kernel, 0.0)
    return np.where(np.isfinite(kernel), kernel, 0.0)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(mask)) == 0:
        return float("nan")
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    total = float(np.sum(w))
    if total <= 0.0:
        return float("nan")
    cdf = np.cumsum(w)
    return float(v[int(np.searchsorted(cdf, 0.5 * total, side="left"))])


def local_j_residuals(hkls: np.ndarray, j_values: np.ndarray, metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(j_values))
    residuals = np.full(n, np.nan, dtype=float)
    neighbor_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        delta = hkls - hkls[i]
        nonself = np.any(delta != 0, axis=1)
        dq = dq_from_delta(delta, metric)
        weights = coupling_kernel(dq)
        valid = nonself & (weights > 0.0) & np.isfinite(j_values)
        count = int(np.sum(valid))
        neighbor_counts[i] = count
        if count < MIN_LOCAL_NEIGHBORS:
            continue
        neighbor_j = j_values[valid]
        neighbor_w = weights[valid]
        med = weighted_median(neighbor_j, neighbor_w)
        if not math.isfinite(med):
            continue
        mad = weighted_median(np.abs(neighbor_j - med), neighbor_w)
        if not math.isfinite(mad):
            continue
        residuals[i] = (float(j_values[i]) - med) / (ROBUST_MAD_SCALE * mad + EPS)
    return residuals, neighbor_counts


def query_cache_rows(conn: sqlite3.Connection, keys: list[str]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for start in range(0, len(keys), 700):
        chunk = keys[start : start + 700]
        values = ",".join(["(?)"] * len(chunk))
        frames.append(
            pd.read_sql_query(
                f"""
                WITH frame_keys(exact_key_text) AS (VALUES {values})
                SELECT sc.ordinal,sc.exact_key_text,sc.source_filename,sc.event,sc.h,sc.k,sc.l,sc.Eg,sc.D,sc.M,sc.M2,
                       hs.n_obs AS hkl_n_obs,hs.risk_denom,hs.egm2_median,hs.max_remove
                FROM frame_keys AS fk
                JOIN score_cache AS sc ON sc.exact_key_text=fk.exact_key_text
                JOIN hkl_stats AS hs ON hs.h=sc.h AND hs.k=sc.k AND hs.l=sc.l
                ORDER BY sc.exact_key_text
                """,
                conn,
                params=chunk,
            )
        )
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def family_candidate_rows(row: pd.Series) -> list[tuple[Any, ...]]:
    risk_z = float(row["risk_z"])
    geom_delta = float(row["geom_delta"])
    j_resid = float(row["j_resid"])
    if not all(math.isfinite(value) for value in [risk_z, geom_delta, j_resid]):
        return []
    if risk_z < RISK_Z_THRESHOLD:
        return []
    egm2 = float(row["egm2"])
    rank_base = egm2 * abs(geom_delta)
    common = (
        int(row["ordinal"]),
        str(row["exact_key_text"]),
        str(row["source_filename"]),
        str(row["event"]),
        int(row["h"]),
        int(row["k"]),
        int(row["l"]),
        float(row["intensity"]),
        float(row["Eg"]),
        float(row["D"]),
        float(row["M"]),
        float(row["M2"]),
        egm2,
        risk_z,
        geom_delta,
        j_resid,
    )
    rows: list[tuple[Any, ...]] = []
    enhance = geom_delta > 0.0 and j_resid >= ENHANCE_J_THRESHOLD
    deplete = geom_delta < 0.0 and j_resid <= DEPLETE_J_THRESHOLD
    if enhance:
        rows.append(("enhance_highJ", *common, rank_base * j_resid))
    if deplete:
        rows.append(("deplete_lowJ", *common, rank_base * abs(j_resid)))
    if enhance or deplete:
        rows.append(("directional_consistent_union", *common, rank_base * abs(j_resid)))
        if abs(j_resid) >= EXTREME_ABS_J_THRESHOLD:
            rows.append(("directional_extreme", *common, rank_base * abs(j_resid)))
    return rows


def worker_score_frame(task: tuple[str, str, list[dict[str, Any]], list[list[float]]]) -> dict[str, Any]:
    cache_db, work_db, records_payload, metric_payload = task
    records = [FrameRecord(**record) for record in records_payload]
    metric = np.asarray(metric_payload, dtype=float)
    stats: dict[str, Any] = {
        "frames": 1,
        "reflection_rows": int(len(records)),
        "joined_rows": 0,
        "missing_cache_rows": 0,
        "finite_residual_rows": 0,
        "too_few_neighbor_rows": 0,
        "candidate_rows": 0,
        "candidate_rows_by_family": {family: 0 for family in FAMILY_ORDER},
    }
    if not records:
        return {"candidate_rows": [], "stats": stats}
    frame = pd.DataFrame([record.__dict__ for record in records])
    if "intensity" not in frame or frame["intensity"].isna().all():
        raise RuntimeError("Frame contains no parsable intensities")

    conn = sqlite3.connect(f"file:{Path(cache_db).resolve()}?mode=ro", uri=True)
    conn.execute("ATTACH DATABASE ? AS work", (str(Path(work_db).resolve()),))
    try:
        cache_rows = query_cache_rows(conn, frame["exact_key_text"].astype(str).tolist())
    finally:
        conn.close()
    if cache_rows.empty:
        stats["missing_cache_rows"] = int(len(frame))
        return {"candidate_rows": [], "stats": stats}
    joined = cache_rows.merge(frame.loc[:, ["exact_key_text", "intensity"]], on="exact_key_text", how="left", sort=False, validate="one_to_one")
    stats["joined_rows"] = int(len(joined))
    stats["missing_cache_rows"] = int(len(frame) - len(joined))
    required_float_columns = ["Eg", "D", "M", "M2", "intensity", "risk_denom", "egm2_median"]
    for column in required_float_columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    if joined[required_float_columns].isna().any().any():
        raise RuntimeError("Nonfinite joined intensity/cache values encountered; cannot compute directional residuals")
    if (joined["D"].to_numpy(dtype=float) < 0.0).any():
        raise RuntimeError("Negative D coupling sum encountered; cannot compute A=M/(D+eps)")

    eg = joined["Eg"].to_numpy(dtype=float)
    m2 = joined["M2"].to_numpy(dtype=float)
    m = joined["M"].to_numpy(dtype=float)
    d = joined["D"].to_numpy(dtype=float)
    intensity = joined["intensity"].to_numpy(dtype=float)
    egm2 = eg * m2
    risk_z = (egm2 - joined["egm2_median"].to_numpy(dtype=float)) / joined["risk_denom"].to_numpy(dtype=float)
    geom_delta = m / (d + EPS) - eg
    j_values = intensity * eg
    if not np.isfinite(j_values).all():
        raise RuntimeError("Nonfinite J=I*Eg encountered; cannot compute local residuals")
    hkls = joined.loc[:, HKL_COLUMNS].to_numpy(dtype=np.int64)
    j_resid, neighbor_counts = local_j_residuals(hkls, j_values, metric)
    joined = joined.assign(egm2=egm2, risk_z=risk_z, geom_delta=geom_delta, j_resid=j_resid, local_neighbor_count=neighbor_counts)
    stats["finite_residual_rows"] = int(np.isfinite(j_resid).sum())
    stats["too_few_neighbor_rows"] = int((neighbor_counts < MIN_LOCAL_NEIGHBORS).sum())

    candidate_rows: list[tuple[Any, ...]] = []
    for _idx, row in joined.iterrows():
        rows = family_candidate_rows(row)
        candidate_rows.extend(rows)
        for item in rows:
            stats["candidate_rows_by_family"][str(item[0])] += 1
    stats["candidate_rows"] = int(len(candidate_rows))
    return {"candidate_rows": candidate_rows, "stats": stats}


def candidate_insert_sql() -> str:
    return """
        INSERT OR IGNORE INTO candidates(
            family,ordinal,exact_key_text,source_filename,event,h,k,l,intensity,Eg,D,M,M2,
            egm2,risk_z,geom_delta,j_resid,rank_score
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """


def merge_scan_stats(total: dict[str, Any], stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        if isinstance(value, dict):
            target = total.setdefault(key, {family: 0 for family in FAMILY_ORDER})
            for subkey, subvalue in value.items():
                target[subkey] = int(target.get(subkey, 0)) + int(subvalue)
        else:
            total[key] = int(total.get(key, 0)) + int(value)


def frame_to_payload(records: list[FrameRecord]) -> list[dict[str, Any]]:
    return [record.__dict__ for record in records]


def iter_stream_frames(path: Path, logger: RunLogger, progress_every: float) -> Iterator[list[FrameRecord]]:
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    current_records: list[FrameRecord] = []
    rows_seen = 0
    progress = StageProgress(logger, "parsing source stream for intensities", None, "reflection rows", 1, progress_every)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if "Begin chunk" in line:
                if current_records:
                    yield current_records
                    current_records = []
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
                if current_records:
                    yield current_records
                    current_records = []
                current_source = chunk_source
                current_event = chunk_event
                in_crystal = True
                in_reflections = False
                continue
            if "End crystal" in line:
                in_crystal = False
                in_reflections = False
                if current_records:
                    yield current_records
                    current_records = []
                continue
            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                continue
            if in_reflections and line.startswith("End of reflections"):
                in_reflections = False
                if current_records:
                    yield current_records
                    current_records = []
                continue
            parsed = parse_reflection_row(line) if in_crystal and in_reflections else None
            if parsed is None:
                continue
            h, k, l, intensity = parsed
            rows_seen += 1
            key = stream_key_from_context(current_source, current_event, (h, k, l))
            current_records.append(FrameRecord(current_source, current_event, h, k, l, intensity, key))
            progress.update(rows_seen)
    if current_records:
        yield current_records
    progress.finish(rows_seen)


def drain_frame_futures(
    futures: set[Any],
    work_conn: sqlite3.Connection,
    scan_stats: dict[str, Any],
    progress: StageProgress,
    completed_frames: int,
    force_all: bool = False,
) -> tuple[set[Any], int]:
    if not futures:
        return futures, completed_frames
    done: set[Any]
    if force_all:
        done = set(futures)
    else:
        done, _pending = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        result = future.result()
        candidate_rows = result["candidate_rows"]
        if candidate_rows:
            work_conn.executemany(candidate_insert_sql(), candidate_rows)
        merge_scan_stats(scan_stats, result["stats"])
        completed_frames += int(result["stats"].get("frames", 0))
        progress.update(completed_frames)
    futures -= done
    return futures, completed_frames


def populate_candidates(
    cache_db: Path,
    work_db: Path,
    work_conn: sqlite3.Connection,
    source_stream: Path,
    metric: np.ndarray,
    workers: int,
    logger: RunLogger,
    progress_every: float,
) -> dict[str, Any]:
    scan_stats: dict[str, Any] = {
        "frames": 0,
        "reflection_rows": 0,
        "joined_rows": 0,
        "missing_cache_rows": 0,
        "finite_residual_rows": 0,
        "too_few_neighbor_rows": 0,
        "candidate_rows": 0,
        "candidate_rows_by_family": {family: 0 for family in FAMILY_ORDER},
    }
    progress = StageProgress(logger, "computing directional intensity candidates", None, "frames", workers, progress_every)
    max_pending = max(1, int(workers) * FRAME_FUTURE_MULTIPLIER)
    completed_frames = 0
    work_conn.execute("BEGIN")
    try:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures: set[Any] = set()
                for frame_records in iter_stream_frames(source_stream, logger, progress_every):
                    if not frame_records:
                        continue
                    futures.add(
                        executor.submit(
                            worker_score_frame,
                            (str(cache_db), str(work_db), frame_to_payload(frame_records), metric.tolist()),
                        )
                    )
                    while len(futures) >= max_pending:
                        futures, completed_frames = drain_frame_futures(futures, work_conn, scan_stats, progress, completed_frames)
                while futures:
                    futures, completed_frames = drain_frame_futures(
                        futures, work_conn, scan_stats, progress, completed_frames, force_all=True
                    )
        else:
            for frame_records in iter_stream_frames(source_stream, logger, progress_every):
                result = worker_score_frame((str(cache_db), str(work_db), frame_to_payload(frame_records), metric.tolist()))
                candidate_rows = result["candidate_rows"]
                if candidate_rows:
                    work_conn.executemany(candidate_insert_sql(), candidate_rows)
                merge_scan_stats(scan_stats, result["stats"])
                completed_frames += int(result["stats"].get("frames", 0))
                progress.update(completed_frames)
        work_conn.commit()
    except Exception:
        work_conn.rollback()
        raise
    progress.finish(completed_frames)
    if int(scan_stats["reflection_rows"]) <= 0:
        raise SystemExit("No reflection intensities were parsed from the source stream")
    if int(scan_stats["joined_rows"]) <= 0:
        raise SystemExit("No parsed stream intensities could be joined to full_population_cache.sqlite exact keys")
    if int(scan_stats["finite_residual_rows"]) <= 0:
        raise SystemExit("No local neighbour residuals could be computed; check unit cell and local-neighbour density")
    work_conn.executescript(
        """
        CREATE INDEX idx_candidates_family_rank ON candidates(family, rank_score DESC, risk_z DESC, exact_key_text);
        CREATE INDEX idx_candidates_hkl ON candidates(family,h,k,l);
        CREATE INDEX idx_candidates_ordinal ON candidates(ordinal);
        """
    )
    work_conn.commit()
    return scan_stats


def select_variant(
    conn: sqlite3.Connection,
    variant: VariantSpec,
    hkl_limits: dict[tuple[int, int, int], int],
) -> dict[str, Any]:
    eligible = int(conn.execute("SELECT COUNT(*) FROM candidates WHERE family=?", (variant.family,)).fetchone()[0])
    target_remove = int(math.floor(float(variant.fraction) * eligible))
    selected_per_hkl: dict[tuple[int, int, int], int] = {}
    selected = 0
    exhausted = True
    cursor = conn.execute(
        """
        SELECT ordinal,exact_key_text,source_filename,event,h,k,l,intensity,Eg,D,M,M2,egm2,risk_z,geom_delta,j_resid,rank_score
        FROM candidates
        WHERE family=?
        ORDER BY rank_score DESC, risk_z DESC, exact_key_text ASC
        """,
        (variant.family,),
    )
    insert_rows: list[tuple[Any, ...]] = []
    for row in cursor:
        if selected >= target_remove:
            exhausted = False
            break
        hkl = (int(row[4]), int(row[5]), int(row[6]))
        limit = int(hkl_limits.get(hkl, 0))
        if selected_per_hkl.get(hkl, 0) >= limit:
            continue
        selected += 1
        selected_per_hkl[hkl] = selected_per_hkl.get(hkl, 0) + 1
        insert_rows.append((variant.variant_id, variant.family, float(variant.fraction), variant.fraction_text, selected, *row))
        if len(insert_rows) >= 50_000:
            conn.executemany(
                """
                INSERT INTO selected(
                    variant_id,family,fraction,fraction_label,selection_rank,ordinal,exact_key_text,source_filename,event,h,k,l,
                    intensity,Eg,D,M,M2,egm2,risk_z,geom_delta,j_resid,rank_score
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                insert_rows,
            )
            insert_rows = []
    if insert_rows:
        conn.executemany(
            """
            INSERT INTO selected(
                variant_id,family,fraction,fraction_label,selection_rank,ordinal,exact_key_text,source_filename,event,h,k,l,
                intensity,Eg,D,M,M2,egm2,risk_z,geom_delta,j_resid,rank_score
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            insert_rows,
        )
    selected_hkls = int(len(selected_per_hkl))
    capped_hkls = int(sum(1 for hkl, value in selected_per_hkl.items() if value >= int(hkl_limits.get(hkl, 0))))
    return {
        "variant_id": variant.variant_id,
        "family": variant.family,
        "fraction": float(variant.fraction),
        "fraction_label": variant.fraction_text,
        "eligible_observation_count": int(eligible),
        "target_remove_count": int(target_remove),
        "actual_removed_count": int(selected),
        "selection_exhausted_by_hkl_safety": bool(exhausted and selected < target_remove),
        "selected_hkl_groups": selected_hkls,
        "selected_hkl_groups_at_cap": capped_hkls,
    }


def hkl_limits_from_work(conn: sqlite3.Connection) -> dict[tuple[int, int, int], int]:
    return {
        (int(h), int(k), int(l)): int(max_remove)
        for h, k, l, max_remove in conn.execute("SELECT h,k,l,max_remove FROM hkl_stats")
    }


def run_selection(
    work_conn: sqlite3.Connection,
    variants: list[VariantSpec],
    logger: RunLogger,
    progress_every: float,
) -> list[dict[str, Any]]:
    logger.log("loading per-HKL safety limits")
    hkl_limits = hkl_limits_from_work(work_conn)
    progress = StageProgress(logger, "selecting globally ranked directional removals", len(variants), "variants", 1, progress_every)
    rows: list[dict[str, Any]] = []
    work_conn.execute("BEGIN")
    try:
        for idx, variant in enumerate(variants, start=1):
            rows.append(select_variant(work_conn, variant, hkl_limits))
            progress.update(idx)
        work_conn.commit()
    except Exception:
        work_conn.rollback()
        raise
    progress.finish(len(variants))
    work_conn.execute("CREATE INDEX idx_selected_variant ON selected(variant_id, selection_rank)")
    work_conn.execute("CREATE INDEX idx_selected_ordinal ON selected(ordinal)")
    work_conn.commit()
    return rows


def write_removed_keys(conn: sqlite3.Connection, path: Path, logger: RunLogger, progress_every: float) -> int:
    total = int(conn.execute("SELECT COUNT(*) FROM selected").fetchone()[0])
    progress = StageProgress(logger, "writing removed exact keys", total, "rows", 1, progress_every)
    written = 0
    fieldnames = [
        "variant_id",
        "family",
        "fraction",
        "fraction_label",
        "selection_rank",
        "ordinal",
        "exact_key_text",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "intensity",
        "Eg",
        "D",
        "M",
        "M2",
        "egm2",
        "risk_z",
        "geom_delta",
        "j_resid",
        "rank_score",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fieldnames)
        for row in conn.execute(
            """
            SELECT variant_id,family,fraction,fraction_label,selection_rank,ordinal,exact_key_text,source_filename,event,
                   h,k,l,intensity,Eg,D,M,M2,egm2,risk_z,geom_delta,j_resid,rank_score
            FROM selected
            ORDER BY variant_id, selection_rank
            """
        ):
            writer.writerow(row)
            written += 1
            progress.update(written)
    progress.finish(written)
    return written


def lookup_ordinal(conn: sqlite3.Connection, key: str) -> int | None:
    row = conn.execute("SELECT ordinal FROM score_cache WHERE exact_key_text=?", (key,)).fetchone()
    return None if row is None else int(row[0])


def build_masks_from_selected(
    work_conn: sqlite3.Connection,
    variants: list[VariantSpec],
    mask_bits: int,
    logger: RunLogger,
    progress_every: float,
) -> dict[str, PackedMask]:
    masks = {variant.variant_id: PackedMask(mask_bits) for variant in variants}
    total = int(work_conn.execute("SELECT COUNT(*) FROM selected").fetchone()[0])
    progress = StageProgress(logger, "building selected-row masks", total, "rows", 1, progress_every)
    completed = 0
    for variant_id, ordinal in work_conn.execute("SELECT variant_id, ordinal FROM selected ORDER BY variant_id"):
        masks[str(variant_id)].set_many([int(ordinal)])
        completed += 1
        progress.update(completed)
    progress.finish(completed)
    return masks


def rewrite_stream_batch(
    source_stream: Path,
    cache_db: Path,
    variants: list[VariantSpec],
    masks: dict[str, PackedMask],
    logger: RunLogger,
    source_rows: int | None,
    progress_every: float,
) -> list[dict[str, Any]]:
    for variant in variants:
        if variant.output_stream.exists():
            raise SystemExit(f"Refusing to overwrite existing output stream: {variant.output_stream}")
    handles = {variant.variant_id: variant.output_stream.open("w", encoding="utf-8") for variant in variants}
    stats = {
        variant.variant_id: {
            "variant_id": variant.variant_id,
            "output_stream": str(variant.output_stream),
            "planned_removed_count": masks[variant.variant_id].count(),
            "stream_removed_count": 0,
            "stream_kept_count": 0,
            "stream_reflection_rows_seen": 0,
            "source_order_preserved": True,
            "status": "generated",
        }
        for variant in variants
    }
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    progress = StageProgress(logger, f"rewriting stream batch ({len(variants)} outputs)", source_rows, "reflection rows", 1, progress_every)
    conn = connect_readonly(cache_db)
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
                parsed = parse_reflection_row(line) if in_crystal and in_reflections else None
                if parsed is not None:
                    rows_seen += 1
                    h, k, l, _intensity = parsed
                    key = stream_key_from_context(current_source, current_event, (h, k, l))
                    ordinal = lookup_ordinal(conn, key)
                    for variant_id, handle in handles.items():
                        row = stats[variant_id]
                        row["stream_reflection_rows_seen"] += 1
                        if masks[variant_id].get(ordinal):
                            row["stream_removed_count"] += 1
                        else:
                            handle.write(raw_line)
                            row["stream_kept_count"] += 1
                    progress.update(rows_seen)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        conn.close()
        for handle in handles.values():
            handle.close()
    progress.finish(rows_seen)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        row = stats[variant.variant_id]
        if int(row["stream_removed_count"]) != int(row["planned_removed_count"]):
            raise SystemExit(
                f"{variant.variant_id}: stream rewrite removed {row['stream_removed_count']} rows "
                f"but planned {row['planned_removed_count']}"
            )
        rows.append(row)
    return rows


def rewrite_streams(
    source_stream: Path,
    cache_db: Path,
    variants: list[VariantSpec],
    masks: dict[str, PackedMask],
    logger: RunLogger,
    source_rows: int | None,
    progress_every: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = StageProgress(logger, "rewriting directional intensity streams", len(variants), "streams", 1, progress_every)
    for start in range(0, len(variants), MAX_OPEN_STREAMS):
        batch = variants[start : start + MAX_OPEN_STREAMS]
        rows.extend(rewrite_stream_batch(source_stream, cache_db, batch, masks, logger, source_rows, progress_every))
        progress.update(min(len(rows), len(variants)))
    progress.finish(len(variants))
    return rows


def stream_qc_by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["variant_id"]): row for row in rows}


def selected_mask_digests(conn: sqlite3.Connection, variants: list[VariantSpec], mask_bits: int) -> dict[str, str]:
    digests: dict[str, str] = {}
    for variant in variants:
        mask = PackedMask(mask_bits)
        ordinals = [int(row[0]) for row in conn.execute("SELECT ordinal FROM selected WHERE variant_id=?", (variant.variant_id,))]
        mask.set_many(ordinals)
        digests[variant.variant_id] = mask.digest()
    return digests


def make_count_rows(
    variants: list[VariantSpec],
    selection_rows: list[dict[str, Any]],
    accepted_count: int,
    source_rows: int | None,
    mask_digests: dict[str, str],
) -> list[dict[str, Any]]:
    by_id = {row["variant_id"]: row for row in selection_rows}
    out: list[dict[str, Any]] = []
    for variant in variants:
        row = dict(by_id.get(variant.variant_id, {}))
        removed = int(row.get("actual_removed_count", 0))
        out.append(
            {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "score_base": "EgM2 = Eg * M2",
                "direction_proxy": "A=M/(D+eps); geom_delta=A-Eg",
                "local_intensity_proxy": "J=I*Eg; J_resid=(J_g-weighted_median_C(J_q))/(1.4826*weighted_MAD_C(J_q)+eps)",
                "fraction": float(variant.fraction),
                "fraction_label": variant.fraction_text,
                "eligible_observation_count": int(row.get("eligible_observation_count", 0)),
                "target_remove_count": int(row.get("target_remove_count", 0)),
                "accepted_observations_removed": removed,
                "accepted_observations_retained": int(accepted_count - removed),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "removed_fraction_of_eligible_domain": float(removed / max(1, int(row.get("eligible_observation_count", 0)))),
                "removed_fraction_of_scoreable_population": float(removed / max(1, accepted_count)),
                "removed_fraction_of_source_rows": "" if source_rows is None else float(removed / max(1, source_rows)),
                "selected_hkl_groups": int(row.get("selected_hkl_groups", 0)),
                "selected_hkl_groups_at_cap": int(row.get("selected_hkl_groups_at_cap", 0)),
                "selection_exhausted_by_hkl_safety": bool(row.get("selection_exhausted_by_hkl_safety", False)),
                "mask_sha256": mask_digests.get(variant.variant_id, ""),
            }
        )
    return out


def make_manifest_rows(
    variants: list[VariantSpec],
    count_rows: list[dict[str, Any]],
    stream_qc: list[dict[str, Any]],
    write_streams: bool,
) -> list[dict[str, Any]]:
    counts = {row["variant_id"]: row for row in count_rows}
    qc = stream_qc_by_variant(stream_qc)
    rows: list[dict[str, Any]] = []
    for order, variant in enumerate(variants, start=1):
        count = counts[variant.variant_id]
        qc_row = qc.get(variant.variant_id, {})
        rows.append(
            {
                "merge_order": order,
                "variant_id": variant.variant_id,
                "family": variant.family,
                "stream_path": str(variant.output_stream),
                "score_base": "eg_m2",
                "score_formula": "Eg * M2",
                "experiment_type": "fullpop_directional_intensity_filter",
                "target": "all",
                "fraction": float(variant.fraction),
                "fraction_label": variant.fraction_text,
                "eligible_observation_count": count["eligible_observation_count"],
                "actual_removed_count": count["accepted_observations_removed"],
                "actual_removed_fraction_of_scoreable_population": count["removed_fraction_of_scoreable_population"],
                "actual_removed_fraction_of_source_rows": count["removed_fraction_of_source_rows"],
                "stream_status": qc_row.get("status") or ("generated" if write_streams else "planned_dry_run"),
                "stream_qc_removed_observations": qc_row.get("stream_removed_count", ""),
                "stream_qc_kept_observations": qc_row.get("stream_kept_count", ""),
                "source_order_preserved": qc_row.get("source_order_preserved", True),
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
    variants: list[VariantSpec],
    cache_db: Path,
    cache_columns: list[str],
    cache_stats: dict[str, int],
    source_rows: int | None,
    unit_cell: dict[str, float],
    reciprocal_metric: np.ndarray,
    hkl_stats: dict[str, Any],
    scan_stats: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    removed_keys_count: int,
    stream_qc: list[dict[str, Any]],
    mask_digests: dict[str, str],
    started_utc: str,
    logger: RunLogger,
) -> None:
    logger.log("writing directional intensity manifests and metadata")
    count_rows = make_count_rows(variants, selection_rows, int(cache_stats["count"]), source_rows, mask_digests)
    manifest_rows = make_manifest_rows(variants, count_rows, stream_qc, bool(args.write_streams))
    write_csv(args.output_dir / "directional_intensity_counts.csv", count_rows)
    write_csv(args.output_dir / "directional_intensity_manifest.tsv", manifest_rows, delimiter="\t")
    parameters = {
        "source_out_dir": str(args.source_out_dir),
        "source_stream": str(args.source_stream),
        "source_cache": str(cache_db),
        "output_dir": str(args.output_dir),
        "families": args.family_items,
        "fractions": [text for text, _fraction in args.fraction_items],
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "overwrite": bool(args.overwrite),
        "score_base": {"score_id": "eg_m2", "formula": "Eg * M2"},
        "direction_proxy": "A = M/(D + eps); geom_delta = A - Eg",
        "local_intensity_proxy": "J = I * Eg; weighted local residual uses C(g-q) weights within the same source/event frame",
        "kernel": {
            "formula": "C(g-q)=exp[-0.5*(dq/sigma_c)^2] for dq<=r_cut, otherwise 0",
            "dq": "sqrt((q-g)^T G* (q-g)) from the source-stream unit cell reciprocal metric",
            "sigma_c_A_inv": SIGMA_C,
            "r_cut_A_inv": R_CUT,
            "self_coupling_excluded": True,
        },
        "thresholds": {
            "risk_z": RISK_Z_THRESHOLD,
            "enhance_highJ": {"geom_delta": ">0", "j_resid": f">={ENHANCE_J_THRESHOLD}"},
            "deplete_lowJ": {"geom_delta": "<0", "j_resid": f"<={DEPLETE_J_THRESHOLD}"},
            "directional_extreme_abs_j_resid": EXTREME_ABS_J_THRESHOLD,
            "min_local_neighbors": MIN_LOCAL_NEIGHBORS,
        },
        "filtering": {
            "domain": "full-population all-domain scoreable accepted observations joined to source-stream intensities",
            "selection_scope": "global within eligible observations for each family/fraction",
            "hkl_key": "exact signed h,k,l; no symmetry canonicalization",
            "max_remove_per_hkl_fraction": MAX_REMOVE_PER_HKL_FRACTION,
            "min_retained_per_hkl": MIN_RETAINED_PER_HKL,
            "source_order": "preserved by stream-line rewriting",
        },
    }
    write_json(args.output_dir / "directional_intensity_parameters.json", parameters)
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
            "source_cache": str(cache_db),
            "source_stream": str(args.source_stream),
            "source_parameters": str(args.source_out_dir / "parameters.json"),
            "source_validation": str(args.source_out_dir / "validation.json"),
        },
        "source_cache_columns": cache_columns,
        "cache_stats": cache_stats,
        "source_reflection_row_count": "" if source_rows is None else int(source_rows),
        "unit_cell_A_deg": unit_cell,
        "reciprocal_metric_A_inv_sq": reciprocal_metric,
        "hkl_stats": hkl_stats,
        "candidate_scan_stats": scan_stats,
        "selection_stats": selection_rows,
        "removed_keys_count": int(removed_keys_count),
        "stream_qc": stream_qc,
        "validation": {
            "intensities_parsed": int(scan_stats.get("reflection_rows", 0)) > 0,
            "intensities_joined_to_cache": int(scan_stats.get("joined_rows", 0)) > 0,
            "local_residuals_computed": int(scan_stats.get("finite_residual_rows", 0)) > 0,
            "source_order_preserved": all(bool(row.get("source_order_preserved", True)) for row in stream_qc) if stream_qc else True,
        },
        "outputs": {
            "manifest": str(args.output_dir / "directional_intensity_manifest.tsv"),
            "counts": str(args.output_dir / "directional_intensity_counts.csv"),
            "removed_keys": str(args.output_dir / "directional_intensity_removed_keys.tsv.gz"),
            "parameters": str(args.output_dir / "directional_intensity_parameters.json"),
            "metadata": str(args.output_dir / "directional_intensity_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "directional_intensity_metadata.json", metadata)


def main() -> int:
    args = parse_args()
    variants = build_variants(args.family_items, args.fraction_items, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_outputs(planned_output_paths(args.output_dir, variants, bool(args.write_streams)), bool(args.overwrite))

    logger = RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    work_db = work_db_path(args.output_dir)
    work_conn: sqlite3.Connection | None = None
    try:
        logger.log("full-pop directional intensity filter builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"families={','.join(args.family_items)}")
        logger.log(f"fractions={','.join(text for text, _fraction in args.fraction_items)}")
        logger.log(f"variants={len(variants)}")
        logger.log(f"workers={int(args.workers)}")

        cache_db = cache_db_path(args.source_out_dir)
        if not cache_db.is_file():
            raise SystemExit(f"full-population cache not found: {cache_db}")
        cache_columns = require_cache_schema(cache_db)
        cache_stats = score_cache_stats(cache_db)
        if int(cache_stats["count"]) <= 0:
            raise SystemExit("score_cache is empty")
        source_rows = source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and int(source_rows) < int(cache_stats["count"]):
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than accepted cache count {cache_stats['count']:,}")

        unit_cell = parse_unit_cell_from_stream(args.source_stream)
        metric = reciprocal_metric_from_cell(unit_cell)
        logger.log(
            "kernel: gaussian C(g-q)=exp[-0.5*(dq/sigma_c)^2], "
            f"sigma_c={SIGMA_C:.6g} A^-1, r_cut={R_CUT:.6g} A^-1, q!=g"
        )

        work_conn = connect_work(work_db)
        init_work_db(work_conn)
        hkl_stats = populate_hkl_stats(cache_db, work_conn, int(args.workers), logger, float(args.progress_every))
        scan_stats = populate_candidates(
            cache_db,
            work_db,
            work_conn,
            args.source_stream,
            metric,
            int(args.workers),
            logger,
            float(args.progress_every),
        )
        selection_rows = run_selection(work_conn, variants, logger, float(args.progress_every))
        removed_keys_count = write_removed_keys(
            work_conn,
            args.output_dir / "directional_intensity_removed_keys.tsv.gz",
            logger,
            float(args.progress_every),
        )
        mask_digests = selected_mask_digests(work_conn, variants, int(cache_stats["mask_bits"]))
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            masks = build_masks_from_selected(work_conn, variants, int(cache_stats["mask_bits"]), logger, float(args.progress_every))
            stream_qc = rewrite_streams(args.source_stream, cache_db, variants, masks, logger, source_rows, float(args.progress_every))
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(
            args,
            variants,
            cache_db,
            cache_columns,
            cache_stats,
            source_rows,
            unit_cell,
            metric,
            hkl_stats,
            scan_stats,
            selection_rows,
            removed_keys_count,
            stream_qc,
            mask_digests,
            started_utc,
            logger,
        )
        logger.log("full-pop directional intensity filter builder complete")
        return 0
    finally:
        if work_conn is not None:
            work_conn.close()
        if work_db.exists():
            work_db.unlink()
        wal = Path(str(work_db) + "-wal")
        shm = Path(str(work_db) + "-shm")
        for sidecar in [wal, shm]:
            if sidecar.exists():
                sidecar.unlink()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
