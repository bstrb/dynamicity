#!/usr/bin/env python3
"""Build targeted V6 formula-comparison stream plans.

This script creates a small, standardized comparison between two score formulas:

* eg_m2:     S = Eg * sum_q(Eq*C(g-q)^2)
* eg_c2mean: S = Eg * sum_q(Eq*C(g-q)^2) / sum_q(Eq)

It has two modes:

* plan:    read the existing V6 full-population cache, build removal masks, and
           write manifests/metadata/overlap tables.
* streams: rewrite the source CrystFEL stream from the validated plan masks.

It never runs Partialator, stream generation from raw data, or merging.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
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


BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
for _name in BLAS_THREAD_ENV_VARS:
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd


DEFAULT_SOURCE_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_full_population_sweep_20260717"
)
DEFAULT_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_formula_comparison_20260809"
)

SCORE_IDS = ("eg_m2", "eg_c2mean")
TARGET_DROPS = {
    "all": (0.02, 0.03, 0.04, 0.05, 0.07, 0.10),
    "higheg": (0.10, 0.15, 0.20, 0.25, 0.30),
    "matched": (0.10, 0.15, 0.20, 0.25, 0.30),
}
EXPECTED_VARIANT_COUNT = 32

KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
HIGH_EG_FRACTION = 0.30
EXCITATION_BLOCK_SIZE = 10
MIN_FINAL_BLOCK_SIZE = 5
MIN_HIGH_EG_OBSERVATIONS = 10
MIN_REMAINING = 2
TARGET_ALL_MIN_OBSERVATIONS = 10
PLAN_BATCH_HKLS = 250
MAX_PENDING_FACTOR = 2
MAX_OPEN_STREAMS = 8
CSV_FLOAT_FORMAT = "%.12g"
EXPECTED_ACCEPTED_COUNT = 6_732_955
EXPECTED_SOURCE_REFLECTION_ROWS = 10_591_743

OUTPUT_PLAN = "formula_comparison_plan.csv"
OUTPUT_COUNTS = "formula_comparison_selection_counts.csv"
OUTPUT_OVERLAP = "formula_comparison_overlap.csv"
OUTPUT_STREAM_MANIFEST = "formula_comparison_stream_manifest.csv"
OUTPUT_MERGE_MANIFEST = "formula_comparison_merge_manifest.tsv"
OUTPUT_PARAMETERS = "formula_comparison_parameters.json"
OUTPUT_METADATA = "formula_comparison_metadata.json"

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")

POPCOUNT8 = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    experiment_type: str
    score_id: str
    score_formula: str
    designation: str
    filtering_target: str
    high_eg_fraction: float | None
    block_size: int | None
    drop_fraction: float
    output_filename: str
    mask_mode: str
    priority: str


@dataclass(frozen=True)
class MaskSpec:
    variant_id: str
    path: Path
    mode: str
    output_path: Path


class StageProgress:
    def __init__(self, stage: str, total: int | None, unit: str, workers: int = 1, min_interval: float = 5.0):
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.workers = int(workers)
        self.min_interval = float(min_interval)
        self.start = time.monotonic()
        self.last = self.start
        self.completed = 0
        total_text = f"{self.total:,}" if self.total is not None else "unknown"
        log(f"{self.stage} start: total={total_text} {self.unit}; workers={self.workers}")

    def update(self, completed: int, force: bool = False) -> None:
        self.completed = int(completed)
        now = time.monotonic()
        if not force and self.total is not None and self.completed != self.total and now - self.last < self.min_interval:
            return
        if not force and self.total is None and now - self.last < self.min_interval:
            return
        elapsed = max(now - self.start, 1.0e-9)
        rate = self.completed / elapsed
        parts = [f"{self.stage} progress: completed={self.completed:,}"]
        if self.total is not None:
            pct = 100.0 * self.completed / max(1, self.total)
            parts.append(f"/{self.total:,} ({pct:.1f}%)")
        parts.append(f"; elapsed={format_seconds(elapsed)}")
        parts.append(f"; rate={rate:.2f} {self.unit}/s")
        if self.total is not None and 0 < self.completed < self.total and rate > 0.0:
            parts.append(f"; ETA={format_seconds((self.total - self.completed) / rate)}")
        rss = current_rss_mb()
        if rss is not None:
            parts.append(f"; parent_rss={rss:.1f} MB")
        parts.append(f"; workers={self.workers}")
        log("".join(parts))
        self.last = now

    def advance(self, amount: int = 1) -> None:
        self.update(self.completed + int(amount))

    def finish(self, completed: int | None = None) -> None:
        final = self.completed if completed is None else int(completed)
        self.update(final, force=True)
        elapsed = max(time.monotonic() - self.start, 1.0e-9)
        log(f"{self.stage} complete: completed={final:,}; elapsed={format_seconds(elapsed)}; workers={self.workers}")


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

    def set_many(self, ordinals: Iterable[int] | np.ndarray) -> None:
        if isinstance(ordinals, np.ndarray):
            indexes = ordinals.astype(np.int64, copy=False)
        else:
            indexes = np.fromiter((int(value) for value in ordinals), dtype=np.int64)
        if indexes.size == 0:
            return
        if int(indexes.min()) < 0 or int(indexes.max()) >= self.n_bits:
            raise IndexError(f"Mask ordinal outside 0..{self.n_bits - 1}")
        byte_indexes = np.right_shift(indexes, 3)
        bit_values = np.left_shift(1, np.bitwise_and(indexes, 7)).astype(np.uint8, copy=False)
        np.bitwise_or.at(self.array, byte_indexes, bit_values)

    def get(self, ordinal: int | None) -> bool:
        if ordinal is None:
            return False
        index = int(ordinal)
        if index < 0 or index >= self.n_bits:
            return False
        return bool(int(self.array[index >> 3]) & (1 << (index & 7)))

    def count(self) -> int:
        return int(POPCOUNT8[np.asarray(self.array)].sum())

    def flush(self) -> None:
        self.array.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=["plan", "streams"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def current_rss_mb() -> float | None:
    try:
        with Path("/proc/self/statm").open("r", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(resident_pages * page_size / (1024 * 1024))
    except Exception:
        return None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def percent_label(fraction: float) -> str:
    return f"{int(round(float(fraction) * 100.0)):02d}"


def score_formula(score_id: str) -> str:
    if score_id == "eg_m2":
        return "Eg * sum_q(Eq*C(g-q)^2)"
    if score_id == "eg_c2mean":
        return "Eg * sum_q(Eq*C(g-q)^2) / sum_q(Eq)"
    raise ValueError(score_id)


def score_expanded_formula(score_id: str) -> str:
    if score_id == "eg_m2":
        return "Eg * M2"
    if score_id == "eg_c2mean":
        return "Eg * M2 / U"
    raise ValueError(score_id)


def build_variants() -> list[VariantSpec]:
    rows: list[VariantSpec] = []
    priority = 0
    for target in ("all", "higheg", "matched"):
        for score_id in SCORE_IDS:
            for drop in TARGET_DROPS[target]:
                priority += 1
                label = percent_label(drop)
                variant_id = f"filter_{target}_{score_id}_drop{label}"
                rows.append(
                    VariantSpec(
                        variant_id=variant_id,
                        experiment_type="targeted_filter",
                        score_id=score_id,
                        score_formula=score_formula(score_id),
                        designation=f"drop{label}",
                        filtering_target=target,
                        high_eg_fraction=HIGH_EG_FRACTION if target in {"higheg", "matched"} else None,
                        block_size=EXCITATION_BLOCK_SIZE if target == "matched" else None,
                        drop_fraction=float(drop),
                        output_filename=f"{variant_id}.stream",
                        mask_mode="remove",
                        priority=f"F.{priority:03d}",
                    )
                )
    if len(rows) != EXPECTED_VARIANT_COUNT:
        raise SystemExit(f"Internal plan error: built {len(rows)} variants, expected {EXPECTED_VARIANT_COUNT}")
    return rows


def plan_output_paths(out_dir: Path, variants: list[VariantSpec]) -> list[Path]:
    paths = [
        out_dir / OUTPUT_PLAN,
        out_dir / OUTPUT_COUNTS,
        out_dir / OUTPUT_OVERLAP,
        out_dir / OUTPUT_STREAM_MANIFEST,
        out_dir / OUTPUT_MERGE_MANIFEST,
        out_dir / OUTPUT_PARAMETERS,
        out_dir / OUTPUT_METADATA,
    ]
    paths.extend(out_dir / "selection_masks" / f"{variant.variant_id}.remove.bitset" for variant in variants)
    paths.extend(out_dir / variant.output_filename for variant in variants)
    return paths


def refuse_existing_plan_outputs(out_dir: Path, variants: list[VariantSpec]) -> None:
    existing = [path for path in plan_output_paths(out_dir, variants) if path.exists()]
    if existing:
        preview = "\n  ".join(str(path) for path in existing[:20])
        raise SystemExit(
            "Refusing to overwrite existing formula-comparison outputs. "
            "Move/archive the existing OUT directory first. Existing paths include:\n  "
            + preview
        )


def archive_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    target = archive_dir / f"{stamp}_{path.name}"
    suffix = 1
    while target.exists():
        target = archive_dir / f"{stamp}_{suffix}_{path.name}"
        suffix += 1
    path.replace(target)
    return target


def write_json(path: Path, payload: Any, *, overwrite: bool = False, archive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if archive:
            archive_existing(path)
        elif not overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None, *, overwrite: bool = False, archive: bool = False, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if archive:
            archive_existing(path)
        elif not overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {path}")
    if fieldnames is None:
        fieldnames = collect_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=json_default)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def file_fingerprint(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        row.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return row


def git_info(project_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"project_root": str(project_root)}
    try:
        head = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        info["head"] = head.stdout.strip()
    except OSError as exc:
        info["head_error"] = str(exc)
    try:
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--short"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        info["status_short"] = status.stdout.splitlines()
    except OSError as exc:
        info["status_error"] = str(exc)
    return info


def package_versions() -> dict[str, str]:
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def cache_db_path(source_out_dir: Path) -> Path:
    return source_out_dir / "full_population_cache.sqlite"


def load_source_context(source_out_dir: Path) -> dict[str, Any]:
    if not source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {source_out_dir}")
    parameters = read_json(source_out_dir / "parameters.json")
    validation = read_json(source_out_dir / "validation.json")
    provenance = read_json(source_out_dir / "cache_provenance.json")
    target_c_validation = read_json(source_out_dir / "target_c_domain_validation.json")
    source_stream = Path(str(parameters.get("source_stream") or provenance.get("source_stream", {}).get("path") or ""))
    source_cache = cache_db_path(source_out_dir)
    target_c_source = Path(str(target_c_validation.get("source") or validation.get("target_c_domain_gate", {}).get("source") or ""))
    if not source_stream.is_file():
        raise SystemExit(f"Source stream not found from source OUT metadata: {source_stream}")
    if not source_cache.is_file():
        raise SystemExit(f"Source full-population cache not found: {source_cache}")
    if not target_c_source.is_file():
        raise SystemExit(f"Matched target block source cache not found: {target_c_source}")
    return {
        "source_out_dir": str(source_out_dir),
        "parameters": parameters,
        "validation": validation,
        "cache_provenance": provenance,
        "target_c_domain_validation": target_c_validation,
        "source_stream": source_stream,
        "source_cache": source_cache,
        "target_c_source": target_c_source,
    }


def score_cache_count(db_file: Path) -> int:
    with connect_readonly(db_file) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])


def source_reflection_row_count(source_out_dir: Path) -> int:
    path = source_out_dir / "selection_counts.csv"
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("source_reflection_row_count")
                if value:
                    return int(float(value))
    return EXPECTED_SOURCE_REFLECTION_ROWS


def cache_schema(db_file: Path) -> list[str]:
    with connect_readonly(db_file) as conn:
        return [str(row[1]) for row in conn.execute("PRAGMA table_info(score_cache)").fetchall()]


def require_cache_schema(db_file: Path) -> None:
    required = {"ordinal", "source_filename", "event", "h", "k", "l", "exact_key_text", "source_order", "abs_sg", "Eg", "U", "M2"}
    present = set(cache_schema(db_file))
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"Source score_cache is missing required columns: {missing}")


def count_signed_hkls(db_file: Path) -> int:
    with connect_readonly(db_file) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM (SELECT h,k,l FROM score_cache GROUP BY h,k,l)").fetchone()[0])


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


def fetch_score_cache_batch(db_file: str, hkls: list[tuple[int, int, int]]) -> pd.DataFrame:
    if not hkls:
        return pd.DataFrame(columns=["ordinal", "h", "k", "l", "exact_key_text", "abs_sg", "Eg", "U", "M2"])
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
                    SELECT sc.ordinal,sc.h,sc.k,sc.l,sc.exact_key_text,sc.abs_sg,sc.Eg,sc.U,sc.M2
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


def add_score_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    eg = pd.to_numeric(out["Eg"], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(out["U"], errors="coerce").to_numpy(dtype=float)
    m2 = pd.to_numeric(out["M2"], errors="coerce").to_numpy(dtype=float)
    eg_m2 = eg * m2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        eg_c2mean = np.divide(eg_m2, u, out=np.zeros_like(eg_m2, dtype=float), where=u != 0.0)
    if not np.isfinite(eg_m2).all() or not np.isfinite(eg_c2mean).all():
        raise SystemExit("Nonfinite score values encountered while evaluating formula comparison scores")
    out["score_eg_m2"] = eg_m2
    out["score_eg_c2mean"] = eg_c2mean
    return out


def high_eg_pool(group: pd.DataFrame) -> pd.DataFrame:
    n_high = int(math.floor(HIGH_EG_FRACTION * int(len(group))))
    if n_high <= 0:
        return group.iloc[0:0].copy()
    return group.sort_values(["Eg", "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_high).copy()


def removal_count(n_eligible: int, drop_fraction: float) -> int:
    raw = int(math.floor(float(drop_fraction) * int(n_eligible)))
    cap = int(n_eligible) - int(MIN_REMAINING)
    return int(max(0, min(raw, cap)))


def init_counts(variant_ids: Iterable[str]) -> dict[str, dict[str, int]]:
    return {variant_id: {"selected_or_removed": 0, "eligible": 0, "actionable": 0} for variant_id in variant_ids}


def increment_counts(counts: dict[str, dict[str, int]], variant_id: str, selected_or_removed: int, eligible: int, actionable: int) -> None:
    row = counts[variant_id]
    row["selected_or_removed"] += int(selected_or_removed)
    row["eligible"] += int(eligible)
    row["actionable"] += int(actionable)


def add_ordinals(target: dict[str, list[int]], variant_id: str, ordinals: Iterable[int] | np.ndarray) -> None:
    values = target.setdefault(variant_id, [])
    if isinstance(ordinals, np.ndarray):
        values.extend(int(value) for value in ordinals.tolist())
    else:
        values.extend(int(value) for value in ordinals)


def all_higheg_variant_ids() -> list[str]:
    return [variant.variant_id for variant in build_variants() if variant.filtering_target in {"all", "higheg"}]


def plan_all_higheg_batch_worker(task: tuple[int, str, list[tuple[int, int, int]]]) -> dict[str, Any]:
    batch_index, db_file, hkls = task
    table = fetch_score_cache_batch(db_file, hkls)
    counts = init_counts(all_higheg_variant_ids())
    ordinals: dict[str, list[int]] = {}
    hkl_count = 0
    if table.empty:
        return {"batch_index": batch_index, "hkl_count": 0, "observation_count": 0, "counts": counts, "mask_ordinals": {}}
    table = add_score_columns(table)
    for _hkl, raw_group in table.groupby(HKL_COLUMNS, sort=False):
        group = raw_group.copy()
        n_obs = int(len(group))
        for score_id in SCORE_IDS:
            score_col = f"score_{score_id}"
            for drop in TARGET_DROPS["all"]:
                label = percent_label(drop)
                variant_id = f"filter_all_{score_id}_drop{label}"
                n_remove = removal_count(n_obs, drop)
                actionable = n_obs >= TARGET_ALL_MIN_OBSERVATIONS and n_remove > 0 and n_obs - n_remove >= MIN_REMAINING
                removed = group.iloc[0:0]
                if actionable:
                    removed = group.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_remove)
                    add_ordinals(ordinals, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_obs if n_obs >= TARGET_ALL_MIN_OBSERVATIONS else 0, n_obs if actionable else 0)

        pool = high_eg_pool(group)
        n_pool = int(len(pool))
        for score_id in SCORE_IDS:
            score_col = f"score_{score_id}"
            for drop in TARGET_DROPS["higheg"]:
                label = percent_label(drop)
                variant_id = f"filter_higheg_{score_id}_drop{label}"
                n_remove = removal_count(n_pool, drop)
                actionable = n_pool >= MIN_HIGH_EG_OBSERVATIONS and n_remove > 0 and n_pool - n_remove >= MIN_REMAINING
                removed = pool.iloc[0:0]
                if actionable:
                    removed = pool.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_remove)
                    add_ordinals(ordinals, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_pool, n_pool if actionable else 0)
        hkl_count += 1
    return {
        "batch_index": int(batch_index),
        "hkl_count": int(hkl_count),
        "observation_count": int(len(table)),
        "counts": counts,
        "mask_ordinals": {variant_id: np.asarray(values, dtype=np.int64) for variant_id, values in ordinals.items() if values},
    }


def split_v5_cached_blocks(group: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = group.sort_values(["abs_sg_target", "exact_key_text"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    blocks = [ordered.iloc[start : start + EXCITATION_BLOCK_SIZE].copy() for start in range(0, len(ordered), EXCITATION_BLOCK_SIZE)]
    if len(blocks) > 1 and len(blocks[-1]) < MIN_FINAL_BLOCK_SIZE:
        blocks[-2] = pd.concat([blocks[-2], blocks[-1]], ignore_index=True)
        blocks = blocks[:-1]
    if len(blocks) == 1 and len(blocks[0]) < MIN_FINAL_BLOCK_SIZE:
        return []
    return blocks


def read_source_block_definitions(source_out_dir: Path) -> list[tuple[int, int, int, int, int]]:
    path = source_out_dir / "block_definitions.csv"
    if not path.is_file():
        raise SystemExit(f"Source block_definitions.csv not found: {path}")
    rows: list[tuple[int, int, int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["h"]), int(row["k"]), int(row["l"]), int(row["block_id"]), int(row["block_size"])))
    return sorted(rows)


def block_signature(rows: Iterable[tuple[int, int, int, int, int]]) -> str:
    digest = hashlib.sha256()
    for h, k, l, block_id, block_size in sorted(rows):
        digest.update(f"{h},{k},{l},{block_id},{block_size}\n".encode("utf-8"))
    return digest.hexdigest()


def reconstruct_target_c_members(target_c_source: Path, source_out_dir: Path, logger_workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["h", "k", "l", "exact_key_text", "abs_sg_target", "score_sg175_sc100"]
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    duplicate_count = 0
    progress = StageProgress("plan: loading matched target-domain cache", None, "rows", workers=1)
    rows_seen = 0
    for chunk in pd.read_csv(target_c_source, usecols=usecols, chunksize=250_000, low_memory=False, compression="infer"):
        chunk = chunk.copy()
        for column in HKL_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("int64")
        chunk["exact_key_text"] = chunk["exact_key_text"].astype(str)
        chunk["abs_sg_target"] = pd.to_numeric(chunk["abs_sg_target"], errors="coerce")
        chunk["score_sg175_sc100"] = pd.to_numeric(chunk["score_sg175_sc100"], errors="coerce")
        keys = chunk["exact_key_text"].tolist()
        duplicate_count += sum(1 for key in keys if key in seen)
        seen.update(keys)
        frames.append(chunk)
        rows_seen += len(chunk)
        progress.update(rows_seen)
    progress.finish(rows_seen)
    if duplicate_count:
        raise SystemExit(f"Matched target-domain cache contains duplicate exact keys: {duplicate_count}")
    work = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)
    if not np.isfinite(work[["abs_sg_target", "score_sg175_sc100"]].to_numpy(dtype=float)).all():
        raise SystemExit("Matched target-domain cache contains nonfinite block-defining scores")

    member_rows: list[dict[str, Any]] = []
    block_rows: list[tuple[int, int, int, int, int]] = []
    common_excitation_block_count = 0
    zero_range_blocks = 0
    progress = StageProgress("plan: reconstructing matched block definitions", work.groupby(HKL_COLUMNS).ngroups, "HKLs", workers=logger_workers)
    completed = 0
    for (h, k, l), group in work.groupby(HKL_COLUMNS, sort=False):
        if len(group) >= MIN_HIGH_EG_OBSERVATIONS:
            for block_id, block in enumerate(split_v5_cached_blocks(group), start=1):
                common_excitation_block_count += 1
                values = block["score_sg175_sc100"].to_numpy(dtype=float)
                score_range = float(np.max(values) - np.min(values)) if len(values) else 0.0
                if score_range <= 0.0:
                    zero_range_blocks += 1
                    continue
                block_size = int(len(block))
                block_rows.append((int(h), int(k), int(l), int(block_id), block_size))
                for key in block["exact_key_text"].astype(str):
                    member_rows.append(
                        {
                            "exact_key_text": key,
                            "h": int(h),
                            "k": int(k),
                            "l": int(l),
                            "block_id": int(block_id),
                            "block_size": block_size,
                        }
                    )
        completed += 1
        progress.update(completed)
    progress.finish(completed)

    source_blocks = read_source_block_definitions(source_out_dir)
    reconstructed_blocks = sorted(block_rows)
    if reconstructed_blocks != source_blocks:
        source_set = set(source_blocks)
        reconstructed_set = set(reconstructed_blocks)
        missing = sorted(source_set - reconstructed_set)[:10]
        extra = sorted(reconstructed_set - source_set)[:10]
        raise SystemExit(
            "Reconstructed matched block definitions do not match source V6 block_definitions.csv; "
            f"missing={missing}, extra={extra}"
        )

    members = pd.DataFrame.from_records(member_rows)
    stats = {
        "target_c_source": str(target_c_source),
        "v5_high_eg_cache_rows": int(len(work)),
        "common_high_Eg_observation_count": int(len(members)),
        "common_excitation_block_count": int(common_excitation_block_count),
        "common_actionable_block_count": int(len(reconstructed_blocks)),
        "zero_baseline_range_blocks_excluded": int(zero_range_blocks),
        "signed_hkl_count": int(members.groupby(HKL_COLUMNS).ngroups) if not members.empty else 0,
        "source_block_definition_count": int(len(source_blocks)),
        "reconstructed_block_definition_count": int(len(reconstructed_blocks)),
        "block_definition_signature_sha256": block_signature(reconstructed_blocks),
        "matches_source_block_definitions": True,
    }
    return members, stats


def join_target_c_members_to_cache(members: pd.DataFrame, db_file: Path, workers: int) -> pd.DataFrame:
    if members.empty:
        raise SystemExit("Matched target-domain reconstruction produced no member rows")
    keys = members["exact_key_text"].astype(str).tolist()
    lookup: dict[str, tuple[int, int, int, int, float, float, float, float]] = {}
    progress = StageProgress("plan: joining matched target-domain rows to full cache", len(keys), "keys", workers=1)
    done = 0
    with connect_readonly(db_file) as conn:
        for start in range(0, len(keys), 900):
            chunk = keys[start : start + 900]
            placeholders = ",".join(["?"] * len(chunk))
            query = (
                "SELECT exact_key_text,ordinal,h,k,l,abs_sg,Eg,U,M2 "
                f"FROM score_cache WHERE exact_key_text IN ({placeholders})"
            )
            for key, ordinal, h, k, l, abs_sg, eg, u, m2 in conn.execute(query, chunk):
                lookup[str(key)] = (int(ordinal), int(h), int(k), int(l), float(abs_sg), float(eg), float(u), float(m2))
            done += len(chunk)
            progress.update(done)
    progress.finish(done)
    if len(lookup) != len(keys):
        missing = [key for key in keys if key not in lookup][:10]
        raise SystemExit(f"Matched target-domain/full-cache exact-key join mismatch: matched={len(lookup):,}, expected={len(keys):,}, missing={missing}")

    rows: list[dict[str, Any]] = []
    progress = StageProgress("plan: materializing matched block table", len(keys), "rows", workers=workers)
    for idx, row in enumerate(members.itertuples(index=False), start=1):
        ordinal, h, k, l, abs_sg, eg, u, m2 = lookup[str(row.exact_key_text)]
        if (h, k, l) != (int(row.h), int(row.k), int(row.l)):
            raise SystemExit(f"Matched target-domain HKL mismatch for {row.exact_key_text}")
        rows.append(
            {
                "exact_key_text": str(row.exact_key_text),
                "h": h,
                "k": k,
                "l": l,
                "block_id": int(row.block_id),
                "block_size": int(row.block_size),
                "ordinal": ordinal,
                "abs_sg": abs_sg,
                "Eg": eg,
                "U": u,
                "M2": m2,
            }
        )
        progress.update(idx)
    progress.finish(len(rows))
    return add_score_columns(pd.DataFrame.from_records(rows))


def matched_variant_ids() -> list[str]:
    return [variant.variant_id for variant in build_variants() if variant.filtering_target == "matched"]


def matched_selection_chunk_worker(task: tuple[int, pd.DataFrame]) -> dict[str, Any]:
    chunk_index, table = task
    counts = init_counts(matched_variant_ids())
    ordinals: dict[str, list[int]] = {}
    block_count = 0
    for _block_key, block in table.groupby(["h", "k", "l", "block_id"], sort=False):
        n_block = int(len(block))
        for score_id in SCORE_IDS:
            score_col = f"score_{score_id}"
            for drop in TARGET_DROPS["matched"]:
                label = percent_label(drop)
                variant_id = f"filter_matched_{score_id}_drop{label}"
                n_remove = removal_count(n_block, drop)
                removed = block.sort_values([score_col, "abs_sg", "exact_key_text"], ascending=[False, True, True], kind="mergesort").head(n_remove)
                add_ordinals(ordinals, variant_id, removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
                increment_counts(counts, variant_id, len(removed), n_block, n_block)
        block_count += 1
    return {
        "chunk_index": int(chunk_index),
        "block_count": int(block_count),
        "observation_count": int(len(table)),
        "counts": counts,
        "mask_ordinals": {variant_id: np.asarray(values, dtype=np.int64) for variant_id, values in ordinals.items() if values},
    }


def iter_block_chunks(table: pd.DataFrame, blocks_per_chunk: int = 800) -> Iterator[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    block_count = 0
    for _block_key, block in table.groupby(["h", "k", "l", "block_id"], sort=False):
        frames.append(block.copy())
        block_count += 1
        if block_count >= blocks_per_chunk:
            yield pd.concat(frames, ignore_index=True)
            frames = []
            block_count = 0
    if frames:
        yield pd.concat(frames, ignore_index=True)


def merge_count_dicts(left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]) -> None:
    for variant_id, values in right.items():
        target = left.setdefault(variant_id, {"selected_or_removed": 0, "eligible": 0, "actionable": 0})
        for key in ("selected_or_removed", "eligible", "actionable"):
            target[key] += int(values.get(key, 0))


def apply_result_masks(masks: dict[str, PackedMask], result: dict[str, Any]) -> None:
    for variant_id, ordinals in result.get("mask_ordinals", {}).items():
        masks[str(variant_id)].set_many(ordinals)


def open_masks(out_dir: Path, variants: list[VariantSpec], accepted_count: int, create: bool) -> dict[str, PackedMask]:
    return {
        variant.variant_id: PackedMask(out_dir / "selection_masks" / f"{variant.variant_id}.remove.bitset", accepted_count, create=create)
        for variant in variants
    }


def run_parallel_all_higheg(db_file: Path, masks: dict[str, PackedMask], variants: list[VariantSpec], workers: int) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    target_variants = [variant for variant in variants if variant.filtering_target in {"all", "higheg"}]
    counts = init_counts(variant.variant_id for variant in target_variants)
    total_hkls = count_signed_hkls(db_file)
    progress = StageProgress("plan: selecting all/high-Eg removals", total_hkls, "HKLs", workers=workers)
    completed_hkls = 0
    completed_batches = 0
    completed_observations = 0
    worker_pids: set[int] = set()
    pending: set[Any] = set()
    max_pending = max(1, workers * MAX_PENDING_FACTOR)
    batches = enumerate(iter_hkl_batches(db_file, PLAN_BATCH_HKLS), start=1)
    exhausted = False
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_initializer) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    batch_index, hkls = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(plan_all_higheg_batch_worker, (batch_index, str(db_file), hkls)))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                merge_count_dicts(counts, result["counts"])
                apply_result_masks(masks, result)
                completed_hkls += int(result["hkl_count"])
                completed_observations += int(result["observation_count"])
                completed_batches += 1
                if "pid" in result:
                    worker_pids.add(int(result["pid"]))
                progress.update(completed_hkls)
    progress.finish(completed_hkls)
    return counts, {
        "completed_hkls": int(completed_hkls),
        "completed_observations": int(completed_observations),
        "completed_batches": int(completed_batches),
        "total_hkls": int(total_hkls),
    }


def run_parallel_matched(block_table: pd.DataFrame, masks: dict[str, PackedMask], variants: list[VariantSpec], workers: int) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    target_variants = [variant for variant in variants if variant.filtering_target == "matched"]
    counts = init_counts(variant.variant_id for variant in target_variants)
    total_blocks = int(block_table.groupby(["h", "k", "l", "block_id"]).ngroups)
    progress = StageProgress("plan: selecting matched-block removals", total_blocks, "blocks", workers=workers)
    completed_blocks = 0
    completed_chunks = 0
    completed_observations = 0
    chunk_iter = enumerate(iter_block_chunks(block_table), start=1)
    pending: set[Any] = set()
    max_pending = max(1, workers * MAX_PENDING_FACTOR)
    exhausted = False
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_initializer) as pool:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    chunk_index, chunk = next(chunk_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(matched_selection_chunk_worker, (chunk_index, chunk)))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                merge_count_dicts(counts, result["counts"])
                apply_result_masks(masks, result)
                completed_blocks += int(result["block_count"])
                completed_observations += int(result["observation_count"])
                completed_chunks += 1
                progress.update(completed_blocks)
    progress.finish(completed_blocks)
    return counts, {
        "completed_blocks": int(completed_blocks),
        "completed_observations": int(completed_observations),
        "completed_chunks": int(completed_chunks),
        "total_blocks": int(total_blocks),
    }


def worker_initializer() -> None:
    for name in BLAS_THREAD_ENV_VARS:
        os.environ[name] = "1"


def build_selection_count_rows(
    variants: list[VariantSpec],
    counts: dict[str, dict[str, int]],
    masks: dict[str, PackedMask],
    accepted_count: int,
    source_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        count = counts[variant.variant_id]
        removed = masks[variant.variant_id].count()
        selected_or_removed = int(count["selected_or_removed"])
        if removed != selected_or_removed:
            raise SystemExit(f"Mask/count mismatch for {variant.variant_id}: mask={removed:,}, counts={selected_or_removed:,}")
        retained_accepted = int(accepted_count - removed)
        rows.append(
            {
                "variant_id": variant.variant_id,
                "experiment_type": variant.experiment_type,
                "score_id": variant.score_id,
                "score_formula": variant.score_formula,
                "filtering_target": variant.filtering_target,
                "drop_fraction": variant.drop_fraction,
                "mask_mode": variant.mask_mode,
                "accepted_population_count": int(accepted_count),
                "source_reflection_row_count": int(source_rows),
                "eligible_observation_count": int(count["eligible"]),
                "actionable_observation_count": int(count["actionable"]),
                "selected_or_removed_accepted_observations": int(selected_or_removed),
                "accepted_observations_removed": int(removed),
                "accepted_observations_retained": retained_accepted,
                "out_of_analysis_source_rows_retained": int(source_rows - accepted_count),
                "total_source_rows_retained": int(source_rows - removed),
                "removed_fraction_of_accepted_population": float(removed / max(1, accepted_count)),
                "removed_fraction_of_source_rows": float(removed / max(1, source_rows)),
                "removed_fraction_of_eligible_population": float(removed / count["eligible"]) if count["eligible"] else "",
                "removed_fraction_of_actionable_population": float(removed / count["actionable"]) if count["actionable"] else "",
            }
        )
    return rows


def mask_intersection(mask_a: PackedMask, mask_b: PackedMask) -> int:
    return int(POPCOUNT8[np.bitwise_and(np.asarray(mask_a.array), np.asarray(mask_b.array))].sum())


def build_overlap_rows(variants: list[VariantSpec], masks: dict[str, PackedMask]) -> list[dict[str, Any]]:
    by_key = {(variant.filtering_target, percent_label(variant.drop_fraction), variant.score_id): variant for variant in variants}
    rows: list[dict[str, Any]] = []
    for target, drops in TARGET_DROPS.items():
        for drop in drops:
            label = percent_label(drop)
            left = by_key[(target, label, "eg_m2")]
            right = by_key[(target, label, "eg_c2mean")]
            left_count = masks[left.variant_id].count()
            right_count = masks[right.variant_id].count()
            intersection = mask_intersection(masks[left.variant_id], masks[right.variant_id])
            union = int(left_count + right_count - intersection)
            rows.append(
                {
                    "filtering_target": target,
                    "drop_fraction": float(drop),
                    "designation": f"drop{label}",
                    "eg_m2_variant_id": left.variant_id,
                    "eg_c2mean_variant_id": right.variant_id,
                    "eg_m2_removed_count": int(left_count),
                    "eg_c2mean_removed_count": int(right_count),
                    "intersection_removed_count": int(intersection),
                    "union_removed_count": int(union),
                    "jaccard_removed_sets": float(intersection / union) if union else 1.0,
                    "same_removed_count": bool(left_count == right_count),
                    "matched_pair": bool(target == "matched"),
                }
            )
    return rows


def validate_pairing(selection_counts: list[dict[str, Any]], overlap_rows: list[dict[str, Any]], target_c_stats: dict[str, Any]) -> dict[str, Any]:
    by_variant = {row["variant_id"]: row for row in selection_counts}
    failures: list[str] = []
    for overlap in overlap_rows:
        left = by_variant[overlap["eg_m2_variant_id"]]
        right = by_variant[overlap["eg_c2mean_variant_id"]]
        for key in ("eligible_observation_count", "actionable_observation_count", "accepted_observations_removed"):
            if int(left[key]) != int(right[key]):
                failures.append(f"{overlap['filtering_target']} {overlap['designation']} unpaired {key}: eg_m2={left[key]} eg_c2mean={right[key]}")
        if not overlap["same_removed_count"]:
            failures.append(f"{overlap['filtering_target']} {overlap['designation']} has unequal removed counts")
    matched_rows = [row for row in overlap_rows if row["matched_pair"]]
    if len(matched_rows) != len(TARGET_DROPS["matched"]):
        failures.append(f"Expected {len(TARGET_DROPS['matched'])} matched overlap pairs, found {len(matched_rows)}")
    if not target_c_stats.get("matches_source_block_definitions"):
        failures.append("Matched block definitions do not match source OUT block_definitions.csv")
    return {
        "passed": not failures,
        "failures": failures,
        "paired_target_drop_count": int(len(overlap_rows)),
        "matched_pair_count": int(len(matched_rows)),
        "target_domains_identical_by_pair": not failures,
        "selection_counts_paired": not failures,
        "matched_block_definitions_identical": bool(target_c_stats.get("matches_source_block_definitions")),
    }


def build_plan_rows(variants: list[VariantSpec]) -> list[dict[str, Any]]:
    return [asdict(variant) for variant in variants]


def build_stream_manifest(
    out_dir: Path,
    variants: list[VariantSpec],
    selection_counts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = {row["variant_id"]: row for row in selection_counts}
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.append(
            {
                **asdict(variant),
                **counts[variant.variant_id],
                "output_stream": str(out_dir / variant.output_filename),
                "status": "planned",
                "mask_path": str(out_dir / "selection_masks" / f"{variant.variant_id}.remove.bitset"),
                "stream_qc_removed_observations": "",
                "stream_qc_kept_observations": "",
                "stream_qc_total_reflection_rows_seen": "",
            }
        )
    return rows


def partialator_merge_settings() -> dict[str, Any]:
    return {
        "model": "offset",
        "symmetry": "4/mmm",
        "iterations": 10,
        "min_measurements": 1,
        "push_res": "inf",
        "threads": 24,
        "polarisation": "none",
        "max_adu": "inf",
        "min_res": "inf",
        "no_Bscale": True,
        "no_pr": True,
        "qc_lowres": 20.0,
        "qc_highres": 0.35,
        "source": "/home/bubl3932/projects/dynamicity/merge/run_merge_qc.sh",
    }


def build_merge_manifest(stream_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = partialator_merge_settings()
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(stream_manifest, start=1):
        rows.append(
            {
                "merge_order": idx,
                "variant_id": row["variant_id"],
                "stream_path": row["output_stream"],
                "score_id": row["score_id"],
                "score_formula": row["score_formula"],
                "experiment_type": row["experiment_type"],
                "target": row["filtering_target"],
                "fraction": row["drop_fraction"],
                "actual_removed_count": row["accepted_observations_removed"],
                "actual_removed_fraction_of_accepted_population": row["removed_fraction_of_accepted_population"],
                "priority": row["priority"],
                "stream_status": row["status"],
                "merge_status": "not_started",
                "partialator_model": settings["model"],
                "symmetry": settings["symmetry"],
                "iterations": settings["iterations"],
                "min_measurements": settings["min_measurements"],
                "push_res": settings["push_res"],
                "no_Bscale": settings["no_Bscale"],
                "no_pr": settings["no_pr"],
            }
        )
    return rows


def disk_preflight(source_stream: Path, stream_manifest: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    source_size = int(source_stream.stat().st_size)
    total = 0
    per_variant: list[dict[str, Any]] = []
    for row in stream_manifest:
        retained_fraction = float(row["total_source_rows_retained"]) / max(1.0, float(row["source_reflection_row_count"]))
        estimate = int(round(source_size * retained_fraction))
        total += estimate
        per_variant.append({"variant_id": row["variant_id"], "estimated_output_bytes": estimate, "retained_fraction": retained_fraction})
    usage = shutil.disk_usage(out_dir if out_dir.exists() else out_dir.parent)
    required = int(math.ceil(total * 1.10 + 1_000_000_000))
    return {
        "source_stream_size_bytes": source_size,
        "estimated_new_stream_bytes": int(total),
        "safety_multiplier": 1.10,
        "fixed_margin_bytes": 1_000_000_000,
        "required_free_bytes": int(required),
        "free_bytes": int(usage.free),
        "passed": bool(usage.free >= required),
        "per_variant": per_variant,
    }


def manual_plan_command(source_out_dir: Path, out_dir: Path, workers: int) -> str:
    return (
        "python -u tools/build_v6_formula_comparison_streams.py "
        f"--source-out-dir {source_out_dir} --out-dir {out_dir} --mode plan --workers {workers}"
    )


def manual_streams_command(source_out_dir: Path, out_dir: Path, workers: int) -> str:
    return (
        "python -u tools/build_v6_formula_comparison_streams.py "
        f"--source-out-dir {source_out_dir} --out-dir {out_dir} --mode streams --workers {workers}"
    )


def partialator_template_command(out_dir: Path) -> str:
    return (
        "while IFS=$'\\t' read -r merge_order variant_id stream_path score_id rest; do\n"
        "  [ \"$merge_order\" = \"merge_order\" ] && continue\n"
        "  [ -s \"$stream_path\" ] || { echo \"missing stream: $stream_path\" >&2; continue; }\n"
        "  run_id=$(date +%Y%m%dT%H%M)\n"
        "  merge_out=\"${stream_path%.stream}_partialator_results_${run_id}\"\n"
        "  mkdir -p \"$merge_out/pr-logs\" \"$merge_out/qc_stats\"\n"
        "  python3 /home/bubl3932/projects/dynamicity/merge/stream_to_cell.py --stream \"$stream_path\" --outdir \"$merge_out\"\n"
        "  partialator \"$stream_path\" --model=offset -j 24 -o \"$merge_out/crystfel.hkl\" -y 4/mmm "
        "--min-measurements=1 --push-res=inf --iterations=10 --harvest-file=\"$merge_out/parameters.json\" "
        "--log-folder=\"$merge_out/pr-logs\" --polarisation=none --max-adu=inf --min-res=inf --no-Bscale --no-pr "
        "> \"$merge_out/partialator_stdout.log\" 2> \"$merge_out/partialator_stderr.log\"\n"
        f"done < {out_dir / OUTPUT_MERGE_MANIFEST}"
    )


def build_parameters_payload(args: argparse.Namespace, context: dict[str, Any], target_c_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    geometry = context.get("parameters", {}).get("geometry_parameters") or context.get("cache_provenance", {}).get("geometry_parameters") or {}
    return {
        "source_out_dir": str(args.source_out_dir),
        "out_dir": str(args.out_dir),
        "mode": args.mode,
        "workers": int(args.workers),
        "source_stream": str(context["source_stream"]),
        "source_cache": str(context["source_cache"]),
        "target_c_source": str(context["target_c_source"]),
        "geometry_parameters": geometry,
        "filtering_parameters": {
            "targets": TARGET_DROPS,
            "score_ids": SCORE_IDS,
            "high_Eg_fraction": HIGH_EG_FRACTION,
            "excitation_block_size": EXCITATION_BLOCK_SIZE,
            "min_final_block_size": MIN_FINAL_BLOCK_SIZE,
            "min_high_Eg_observations": MIN_HIGH_EG_OBSERVATIONS,
            "min_remaining": MIN_REMAINING,
            "target_all_min_observations": TARGET_ALL_MIN_OBSERVATIONS,
            "tie_breakers": {
                "all": "score descending, exact_key_text ascending",
                "higheg": "Eg descending/exact_key_text ascending for high-Eg pool; score descending/exact_key_text ascending for removal",
                "matched": "source matched blocks; score descending, abs_sg ascending, exact_key_text ascending",
            },
        },
        "score_definitions": {
            "eg_m2": {
                "formula": score_formula("eg_m2"),
                "expanded_formula": score_expanded_formula("eg_m2"),
            },
            "eg_c2mean": {
                "formula": score_formula("eg_c2mean"),
                "expanded_formula": score_expanded_formula("eg_c2mean"),
                "zero_denominator_behavior": "score set to 0 where U=sum_q(Eq) is 0",
            },
            "coupling": {
                "C": "C(g-q) = exp[-0.5*(dq/sigma_c)^2] for dq <= r_cut, otherwise 0",
                "dq": "dq = sqrt((q-g)^T G* (q-g))",
                "sigma_c": "0.050 A^-1",
                "r_cut": "0.150 A^-1",
                "self_coupling": "q != g",
            },
        },
        "merge_settings": partialator_merge_settings(),
        "target_c_stats": target_c_stats or {},
    }


def build_metadata_payload(
    args: argparse.Namespace,
    context: dict[str, Any],
    accepted_count: int,
    source_rows: int,
    selection_counts: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    target_c_stats: dict[str, Any],
    validation: dict[str, Any],
    disk: dict[str, Any],
    stage_stats: dict[str, Any],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "project_root": str(project_root),
        "git": git_info(project_root),
        "package_versions": package_versions(),
        "numeric_thread_environment": {name: os.environ.get(name) for name in BLAS_THREAD_ENV_VARS},
        "parameters": build_parameters_payload(args, context, target_c_stats),
        "source_files": {
            "source_out_dir": str(args.source_out_dir),
            "source_stream": file_fingerprint(context["source_stream"]),
            "source_cache": file_fingerprint(context["source_cache"]),
            "target_c_source": file_fingerprint(context["target_c_source"]),
            "source_parameters": file_fingerprint(args.source_out_dir / "parameters.json"),
            "source_validation": file_fingerprint(args.source_out_dir / "validation.json"),
            "source_cache_provenance": file_fingerprint(args.source_out_dir / "cache_provenance.json"),
            "source_block_definitions": file_fingerprint(args.source_out_dir / "block_definitions.csv"),
        },
        "source_cache": {
            "accepted_count": int(accepted_count),
            "source_reflection_rows": int(source_rows),
            "expected_accepted_count": EXPECTED_ACCEPTED_COUNT,
            "expected_source_reflection_rows": EXPECTED_SOURCE_REFLECTION_ROWS,
        },
        "plan_validation": validation,
        "target_c_stats": target_c_stats,
        "stage_stats": stage_stats,
        "disk_preflight": disk,
        "output_files": {
            "plan": str(args.out_dir / OUTPUT_PLAN),
            "selection_counts": str(args.out_dir / OUTPUT_COUNTS),
            "overlap": str(args.out_dir / OUTPUT_OVERLAP),
            "stream_manifest": str(args.out_dir / OUTPUT_STREAM_MANIFEST),
            "merge_manifest": str(args.out_dir / OUTPUT_MERGE_MANIFEST),
            "parameters": str(args.out_dir / OUTPUT_PARAMETERS),
            "metadata": str(args.out_dir / OUTPUT_METADATA),
            "selection_masks": str(args.out_dir / "selection_masks"),
        },
        "row_counts": {
            "selection_counts": len(selection_counts),
            "overlap_rows": len(overlap_rows),
        },
        "manual_commands": {
            "plan": manual_plan_command(args.source_out_dir, args.out_dir, int(args.workers)),
            "streams": manual_streams_command(args.source_out_dir, args.out_dir, int(args.workers)),
            "partialator_template": partialator_template_command(args.out_dir),
        },
    }


def run_plan(args: argparse.Namespace) -> int:
    variants = build_variants()
    refuse_existing_plan_outputs(args.out_dir, variants)
    context = load_source_context(args.source_out_dir)
    source_cache = Path(context["source_cache"])
    source_stream = Path(context["source_stream"])
    require_cache_schema(source_cache)
    accepted_count = score_cache_count(source_cache)
    if accepted_count != EXPECTED_ACCEPTED_COUNT:
        log(f"Warning: accepted_count={accepted_count:,} differs from expected {EXPECTED_ACCEPTED_COUNT:,}")
    source_rows = source_reflection_row_count(args.source_out_dir)
    log(f"Using source cache: {source_cache}")
    log(f"Using source stream: {source_stream}")
    log(f"Writing formula comparison OUT: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    masks = open_masks(args.out_dir, variants, accepted_count, create=True)

    all_counts, all_stats = run_parallel_all_higheg(source_cache, masks, variants, int(args.workers))
    target_members, target_c_stats = reconstruct_target_c_members(Path(context["target_c_source"]), args.source_out_dir, int(args.workers))
    block_table = join_target_c_members_to_cache(target_members, source_cache, int(args.workers))
    matched_counts, matched_stats = run_parallel_matched(block_table, masks, variants, int(args.workers))
    counts: dict[str, dict[str, int]] = {}
    merge_count_dicts(counts, all_counts)
    merge_count_dicts(counts, matched_counts)
    for mask in masks.values():
        mask.flush()

    selection_counts = build_selection_count_rows(variants, counts, masks, accepted_count, source_rows)
    overlap_rows = build_overlap_rows(variants, masks)
    validation = validate_pairing(selection_counts, overlap_rows, target_c_stats)
    if not validation["passed"]:
        raise SystemExit("Plan validation failed:\n  " + "\n  ".join(validation["failures"]))
    stream_manifest = build_stream_manifest(args.out_dir, variants, selection_counts)
    merge_manifest = build_merge_manifest(stream_manifest)
    disk = disk_preflight(source_stream, stream_manifest, args.out_dir)
    stage_stats = {"all_higheg": all_stats, "matched": matched_stats}

    parameters = build_parameters_payload(args, context, target_c_stats)
    metadata = build_metadata_payload(
        args,
        context,
        accepted_count,
        source_rows,
        selection_counts,
        overlap_rows,
        target_c_stats,
        validation,
        disk,
        stage_stats,
    )
    log("Writing plan outputs")
    write_csv(args.out_dir / OUTPUT_PLAN, build_plan_rows(variants))
    write_csv(args.out_dir / OUTPUT_COUNTS, selection_counts)
    write_csv(args.out_dir / OUTPUT_OVERLAP, overlap_rows)
    write_csv(args.out_dir / OUTPUT_STREAM_MANIFEST, stream_manifest)
    write_csv(args.out_dir / OUTPUT_MERGE_MANIFEST, merge_manifest, delimiter="\t")
    write_json(args.out_dir / OUTPUT_PARAMETERS, parameters)
    write_json(args.out_dir / OUTPUT_METADATA, metadata)
    log("Plan mode complete")
    return 0


def read_csv_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: value for key, value in row.items()} for row in csv.DictReader(handle, delimiter=delimiter)]


def load_stream_manifest(out_dir: Path) -> list[dict[str, str]]:
    return read_csv_rows(out_dir / OUTPUT_STREAM_MANIFEST)


def load_mask_specs(manifest: list[dict[str, str]]) -> list[MaskSpec]:
    specs: list[MaskSpec] = []
    for row in manifest:
        specs.append(
            MaskSpec(
                variant_id=str(row["variant_id"]),
                path=Path(str(row["mask_path"])),
                mode=str(row["mask_mode"]),
                output_path=Path(str(row["output_stream"])),
            )
        )
    return specs


def lookup_ordinal(conn: sqlite3.Connection, key: str) -> int | None:
    row = conn.execute("SELECT ordinal FROM score_cache WHERE exact_key_text=?", (key,)).fetchone()
    return None if row is None else int(row[0])


def should_remove(mask: PackedMask, mode: str, ordinal: int | None) -> bool:
    if ordinal is None:
        return False
    if mode != "remove":
        raise ValueError(f"Unsupported mask mode for this script: {mode}")
    return mask.get(ordinal)


def rewrite_stream_batch(source_stream: Path, conn: sqlite3.Connection, specs: list[MaskSpec], accepted_count: int, workers: int) -> list[dict[str, Any]]:
    for spec in specs:
        if spec.output_path.exists():
            raise SystemExit(f"Refusing to overwrite existing stream: {spec.output_path}")
    masks = {spec.variant_id: PackedMask(spec.path, accepted_count, create=False) for spec in specs}
    handles = {spec.variant_id: spec.output_path.open("w", encoding="utf-8") for spec in specs}
    stats = {
        spec.variant_id: {
            "variant_id": spec.variant_id,
            "output_stream": str(spec.output_path),
            "requested_removals": masks[spec.variant_id].count(),
            "removed_observations": 0,
            "kept_observations": 0,
            "total_reflection_rows_seen": 0,
            "source_order_preserved": True,
            "status": "generated",
        }
        for spec in specs
    }
    spec_by_id = {spec.variant_id: spec for spec in specs}
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    progress = StageProgress(f"streams: rewriting batch of {len(specs)} streams", None, "reflection rows", workers=workers)
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
                    progress.update(rows_seen)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()
    progress.finish(rows_seen)
    out: list[dict[str, Any]] = []
    for spec in specs:
        row = stats[spec.variant_id]
        if int(row["removed_observations"]) != int(row["requested_removals"]):
            raise SystemExit(f"{spec.variant_id}: removed {row['removed_observations']} but expected {row['requested_removals']}")
        row["all_requested_keys_found_exactly_once"] = True
        row["stream_reflection_row_difference_equals_requested_removals"] = True
        out.append(row)
    return out


def require_valid_plan(out_dir: Path) -> dict[str, Any]:
    metadata = read_json(out_dir / OUTPUT_METADATA)
    validation = metadata.get("plan_validation", {})
    if not metadata or not validation.get("passed"):
        raise SystemExit(f"Valid formula-comparison plan metadata not found or failed: {out_dir / OUTPUT_METADATA}")
    for filename in (OUTPUT_PLAN, OUTPUT_COUNTS, OUTPUT_OVERLAP, OUTPUT_STREAM_MANIFEST, OUTPUT_MERGE_MANIFEST, OUTPUT_PARAMETERS):
        if not (out_dir / filename).is_file():
            raise SystemExit(f"Plan artifact missing; rerun --mode plan: {out_dir / filename}")
    return metadata


def update_stream_outputs(args: argparse.Namespace, manifest: list[dict[str, str]], qc_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    qc_by_variant = {str(row["variant_id"]): row for row in qc_rows}
    updated_manifest: list[dict[str, Any]] = []
    for row in manifest:
        new_row: dict[str, Any] = dict(row)
        qc = qc_by_variant.get(str(row["variant_id"]))
        if qc is not None:
            new_row["status"] = "generated"
            new_row["stream_qc_removed_observations"] = qc["removed_observations"]
            new_row["stream_qc_kept_observations"] = qc["kept_observations"]
            new_row["stream_qc_total_reflection_rows_seen"] = qc["total_reflection_rows_seen"]
        updated_manifest.append(new_row)
    updated_merge = build_merge_manifest(updated_manifest)
    metadata = dict(metadata)
    metadata["stream_generation"] = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "qc_rows": qc_rows,
        "status_counts": {status: sum(1 for row in updated_manifest if row.get("status") == status) for status in sorted({str(row.get("status")) for row in updated_manifest})},
        "passed": all(bool(row.get("all_requested_keys_found_exactly_once")) and bool(row.get("stream_reflection_row_difference_equals_requested_removals")) for row in qc_rows),
    }
    write_csv(args.out_dir / OUTPUT_STREAM_MANIFEST, updated_manifest, archive=True)
    write_csv(args.out_dir / OUTPUT_MERGE_MANIFEST, updated_merge, archive=True, delimiter="\t")
    write_json(args.out_dir / OUTPUT_METADATA, metadata, archive=True)


def run_streams(args: argparse.Namespace) -> int:
    metadata = require_valid_plan(args.out_dir)
    context = load_source_context(args.source_out_dir)
    source_stream = Path(context["source_stream"])
    source_cache = Path(context["source_cache"])
    accepted_count = score_cache_count(source_cache)
    manifest = load_stream_manifest(args.out_dir)
    specs = load_mask_specs(manifest)
    missing_masks = [str(spec.path) for spec in specs if not spec.path.is_file()]
    if missing_masks:
        raise SystemExit(f"Missing mask file(s); rerun --mode plan: {missing_masks[:10]}")
    existing_streams = [str(spec.output_path) for spec in specs if spec.output_path.exists()]
    if existing_streams:
        raise SystemExit("Refusing to overwrite existing stream output(s):\n  " + "\n  ".join(existing_streams[:20]))
    qc_rows: list[dict[str, Any]] = []
    with connect_readonly(source_cache) as conn:
        for start in range(0, len(specs), MAX_OPEN_STREAMS):
            batch = specs[start : start + MAX_OPEN_STREAMS]
            qc_rows.extend(rewrite_stream_batch(source_stream, conn, batch, accepted_count, int(args.workers)))
    update_stream_outputs(args, manifest, qc_rows, metadata)
    log("Streams mode complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "streams":
        return run_streams(args)
    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
