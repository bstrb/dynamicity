#!/usr/bin/env python3
"""Build ultra-gentle full-population EgM2 all-domain filter streams.

The script reads an existing V6 full-population score cache and plans/removes a
small per-HKL fraction of high EgM2 observations.  It does not run Partialator
or merging, and real stream files are written only with --write-streams.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

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
    "oridyn_v6_fullpop_gentle_egm2_20260810"
)
DEFAULT_FRACTIONS = ("0.001", "0.002", "0.003", "0.004", "0.005", "0.0075", "0.010")

KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
MIN_REMAINING = 2
PLAN_BATCH_HKLS = 250
MAX_OPEN_STREAMS = 8
CSV_FLOAT_FORMAT = "%.12g"

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    fraction_text: str
    drop_fraction: float
    output_filename: str
    output_stream: Path


class RunLogger:
    def __init__(self, out_dir: Path | None) -> None:
        self.out_dir = out_dir
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
    def __init__(self, logger: RunLogger, stage: str, total: int | None, unit: str = "items", workers: int = 1) -> None:
        self.logger = logger
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.workers = int(workers)
        self.completed = 0
        self.started = time.monotonic()
        self.last = self.started
        total_text = f"{self.total:,} {unit}" if self.total is not None else f"unknown {unit}"
        self.logger.log(f"{stage}: started ({total_text}; workers={self.workers})")

    def update(self, completed: int, force: bool = False) -> None:
        self.completed = int(completed)
        now = time.monotonic()
        if not force and now - self.last < 5.0:
            return
        self.last = now
        elapsed = max(1.0e-9, now - self.started)
        rate = self.completed / elapsed
        if self.total:
            pct = 100.0 * self.completed / max(1, self.total)
            eta = (self.total - self.completed) / rate if rate > 0 else float("inf")
            self.logger.log(
                f"{self.stage}: {self.completed:,}/{self.total:,} {self.unit} "
                f"({pct:.1f}%), elapsed={format_duration(elapsed)}, "
                f"rate={rate:,.1f}/s, eta={format_duration(eta)}, workers={self.workers}"
            )
        else:
            self.logger.log(
                f"{self.stage}: {self.completed:,} {self.unit}, "
                f"elapsed={format_duration(elapsed)}, rate={rate:,.1f}/s, workers={self.workers}"
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
            raise IndexError("Mask ordinal outside full-population cache bounds")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


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


def stream_key_from_context(current_source: str, current_event: str, hkl: tuple[int, int, int]) -> str:
    return key_to_text(current_source, current_event, int(hkl[0]), int(hkl[1]), int(hkl[2]))


def parse_fraction_items(value: str | None) -> list[tuple[str, float]]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_FRACTIONS)
    out: list[tuple[str, float]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        fraction = float(item)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise SystemExit(f"Invalid drop fraction {item!r}; expected 0 < fraction < 1")
        out.append((item, fraction))
    if not out:
        raise SystemExit("--fractions must contain at least one fraction")
    seen: set[str] = set()
    for label, _fraction in out:
        if label in seen:
            raise SystemExit(f"Duplicate fraction label: {label}")
        seen.add(label)
    return out


def fraction_label(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("+"):
        stripped = stripped[1:]
    return stripped.replace(".", "p").replace("-", "m")


def build_variants(fractions: list[tuple[str, float]], out_dir: Path) -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for text, fraction in fractions:
        label = fraction_label(text)
        variant_id = f"fullpop_eg_m2_drop{label}"
        output_filename = f"{variant_id}.stream"
        variants.append(
            VariantSpec(
                variant_id=variant_id,
                fraction_text=text,
                drop_fraction=float(fraction),
                output_filename=output_filename,
                output_stream=out_dir / output_filename,
            )
        )
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fractions", default=",".join(DEFAULT_FRACTIONS))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    args = parser.parse_args()

    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.fraction_items = parse_fraction_items(args.fractions)
    args.workers = max(1, int(args.workers))
    if args.dry_run and args.write_streams:
        raise SystemExit("Use either --dry-run or --write-streams, not both")
    if not args.write_streams:
        args.dry_run = True
    if not args.source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {args.source_out_dir}")
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    return args


def planned_output_paths(out_dir: Path, variants: list[VariantSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "gentle_egm2_manifest.tsv",
        out_dir / "gentle_egm2_counts.csv",
        out_dir / "parameters.json",
        out_dir / "run_metadata.json",
        out_dir / "run.log",
    ]
    if write_streams:
        paths.extend(variant.output_stream for variant in variants)
    return paths


def refuse_overwrite(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        preview = "\n  ".join(str(path) for path in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ... {len(existing) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {preview}{extra}")


def cache_db_path(source_out_dir: Path) -> Path:
    return source_out_dir / "full_population_cache.sqlite"


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def cache_schema(db_file: Path) -> list[str]:
    with connect_readonly(db_file) as conn:
        return [str(row[1]) for row in conn.execute("PRAGMA table_info(score_cache)").fetchall()]


def require_cache_schema(db_file: Path) -> None:
    required = {"ordinal", "h", "k", "l", "exact_key_text", "source_order", "Eg", "M2"}
    present = set(cache_schema(db_file))
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"full_population_cache.sqlite score_cache is missing required columns: {missing}")


def score_cache_count(db_file: Path) -> int:
    with connect_readonly(db_file) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM score_cache").fetchone()[0])


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
    for key_path in [
        ("source_stream_validation", "source_reflection_rows"),
        ("cache_gate", "source_reflection_rows"),
    ]:
        current: Any = validation
        for key in key_path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return int(current)
    return None


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
        return pd.DataFrame(columns=["ordinal", "h", "k", "l", "exact_key_text", "Eg", "M2"])
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
                    SELECT sc.ordinal,sc.h,sc.k,sc.l,sc.exact_key_text,sc.Eg,sc.M2
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


def removal_count(n_obs: int, drop_fraction: float) -> int:
    raw = int(math.floor(float(drop_fraction) * int(n_obs)))
    cap = int(n_obs) - MIN_REMAINING
    return int(max(0, min(raw, cap)))


def worker_select_batch(task: tuple[str, list[tuple[int, int, int]], list[dict[str, Any]]]) -> dict[str, Any]:
    db_file, hkls, variants_payload = task
    variants = [VariantSpec(output_stream=Path(row["output_stream"]), **{key: value for key, value in row.items() if key != "output_stream"}) for row in variants_payload]
    table = fetch_score_cache_batch(db_file, hkls)
    counts = {
        variant.variant_id: {
            "hkl_groups": 0,
            "candidate_observation_count": 0,
            "actionable_observation_count": 0,
            "removed_count": 0,
        }
        for variant in variants
    }
    ordinals: dict[str, list[int]] = {variant.variant_id: [] for variant in variants}
    if table.empty:
        return {"counts": counts, "ordinals": {}, "hkl_count": 0, "observation_count": 0}
    eg = pd.to_numeric(table["Eg"], errors="coerce").to_numpy(dtype=float)
    m2 = pd.to_numeric(table["M2"], errors="coerce").to_numpy(dtype=float)
    score = eg * m2
    if not np.isfinite(score).all():
        raise SystemExit("Nonfinite EgM2 score encountered in full-population cache")
    table = table.assign(score_eg_m2=score)
    hkl_count = 0
    for _hkl, group in table.groupby(HKL_COLUMNS, sort=False):
        hkl_count += 1
        n_obs = int(len(group))
        for variant in variants:
            row = counts[variant.variant_id]
            row["hkl_groups"] += 1
            row["candidate_observation_count"] += n_obs
            n_remove = removal_count(n_obs, variant.drop_fraction)
            if n_remove > 0:
                row["actionable_observation_count"] += n_obs
                row["removed_count"] += n_remove
                removed = group.sort_values(["score_eg_m2", "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_remove)
                ordinals[variant.variant_id].extend(int(value) for value in removed["ordinal"].to_numpy(dtype=np.int64, copy=False))
    return {
        "counts": counts,
        "ordinals": {key: np.asarray(values, dtype=np.int64) for key, values in ordinals.items() if values},
        "hkl_count": int(hkl_count),
        "observation_count": int(len(table)),
    }


def construct_masks(
    db_file: Path,
    variants: list[VariantSpec],
    accepted_count: int,
    workers: int,
    logger: RunLogger,
) -> tuple[dict[str, PackedMask], dict[str, dict[str, int]], dict[str, Any]]:
    total_hkls = count_signed_hkls(db_file)
    batches = list(iter_hkl_batches(db_file))
    variant_payload = [asdict(variant) for variant in variants]
    masks = {variant.variant_id: PackedMask(accepted_count) for variant in variants}
    counts = {
        variant.variant_id: {
            "hkl_groups": 0,
            "candidate_observation_count": 0,
            "actionable_observation_count": 0,
            "removed_count": 0,
        }
        for variant in variants
    }
    progress = StageProgress(logger, "selecting gentle EgM2 removals", total_hkls, "HKLs", workers=workers)
    completed_hkls = 0
    completed_observations = 0
    if workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(worker_select_batch, (str(db_file), batch, variant_payload)) for batch in batches]
            for future in as_completed(futures):
                result = future.result()
                completed_hkls += int(result["hkl_count"])
                completed_observations += int(result["observation_count"])
                for variant_id, row in result["counts"].items():
                    for key, value in row.items():
                        counts[variant_id][key] += int(value)
                for variant_id, ordinal_array in result["ordinals"].items():
                    masks[variant_id].set_many(ordinal_array)
                progress.update(completed_hkls)
    else:
        for batch in batches:
            result = worker_select_batch((str(db_file), batch, variant_payload))
            completed_hkls += int(result["hkl_count"])
            completed_observations += int(result["observation_count"])
            for variant_id, row in result["counts"].items():
                for key, value in row.items():
                    counts[variant_id][key] += int(value)
            for variant_id, ordinal_array in result["ordinals"].items():
                masks[variant_id].set_many(ordinal_array)
            progress.update(completed_hkls)
    progress.finish(completed_hkls)
    for variant in variants:
        mask_count = masks[variant.variant_id].count()
        if int(mask_count) != int(counts[variant.variant_id]["removed_count"]):
            raise SystemExit(f"{variant.variant_id}: mask count {mask_count} != selected count {counts[variant.variant_id]['removed_count']}")
    stats = {
        "signed_hkl_count": int(total_hkls),
        "completed_hkls": int(completed_hkls),
        "completed_observations": int(completed_observations),
        "batch_count": int(len(batches)),
        "workers": int(workers),
    }
    return masks, counts, stats


def lookup_ordinal(conn: sqlite3.Connection, key: str) -> int | None:
    row = conn.execute("SELECT ordinal FROM score_cache WHERE exact_key_text=?", (key,)).fetchone()
    return None if row is None else int(row[0])


def rewrite_stream_batch(
    source_stream: Path,
    db_file: Path,
    variants: list[VariantSpec],
    masks: dict[str, PackedMask],
    logger: RunLogger,
    source_rows: int | None,
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
    progress = StageProgress(logger, f"rewriting stream batch ({len(variants)} outputs)", source_rows, "reflection rows", workers=1)
    conn = connect_readonly(db_file)
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
                    key = stream_key_from_context(current_source, current_event, hkl)
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
    db_file: Path,
    variants: list[VariantSpec],
    masks: dict[str, PackedMask],
    logger: RunLogger,
    source_rows: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = StageProgress(logger, "rewriting gentle EgM2 streams", len(variants), "streams", workers=1)
    for start in range(0, len(variants), MAX_OPEN_STREAMS):
        batch = variants[start : start + MAX_OPEN_STREAMS]
        rows.extend(rewrite_stream_batch(source_stream, db_file, batch, masks, logger, source_rows))
        progress.update(min(len(rows), len(variants)))
    progress.finish(len(variants))
    return rows


def stream_qc_by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["variant_id"]): row for row in rows}


def make_count_rows(
    variants: list[VariantSpec],
    counts: dict[str, dict[str, int]],
    masks: dict[str, PackedMask],
    accepted_count: int,
    source_rows: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        count = counts[variant.variant_id]
        removed = masks[variant.variant_id].count()
        retained_accepted = int(accepted_count - removed)
        rows.append(
            {
                "variant_id": variant.variant_id,
                "score_id": "eg_m2",
                "score_formula": "Eg * M2",
                "filtering_target": "all",
                "drop_fraction": float(variant.drop_fraction),
                "drop_fraction_label": variant.fraction_text,
                "accepted_population_count": int(accepted_count),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "signed_hkl_groups": int(count["hkl_groups"]),
                "candidate_observation_count": int(count["candidate_observation_count"]),
                "actionable_observation_count": int(count["actionable_observation_count"]),
                "accepted_observations_removed": int(removed),
                "accepted_observations_retained": int(retained_accepted),
                "out_of_analysis_source_rows_retained": "" if source_rows is None else int(source_rows - accepted_count),
                "total_source_rows_retained": "" if source_rows is None else int(source_rows - removed),
                "removed_fraction_of_accepted_population": float(removed / max(1, accepted_count)),
                "removed_fraction_of_source_rows": "" if source_rows is None else float(removed / max(1, source_rows)),
                "removed_fraction_of_candidate_domain": float(removed / max(1, count["candidate_observation_count"])),
                "removed_fraction_of_actionable_population": float(removed / count["actionable_observation_count"]) if count["actionable_observation_count"] else "",
                "mask_sha256": masks[variant.variant_id].digest(),
            }
        )
    return rows


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
                "stream_path": str(variant.output_stream),
                "score_id": "eg_m2",
                "score_formula": "Eg * M2",
                "experiment_type": "fullpop_gentle_filter",
                "target": "all",
                "fraction": float(variant.drop_fraction),
                "fraction_label": variant.fraction_text,
                "actual_removed_count": int(count["accepted_observations_removed"]),
                "actual_removed_fraction_of_accepted_population": count["removed_fraction_of_accepted_population"],
                "actual_removed_fraction_of_source_rows": count["removed_fraction_of_source_rows"],
                "stream_status": qc_row.get("status") or ("generated" if write_streams else "planned_dry_run"),
                "stream_qc_removed_observations": qc_row.get("stream_removed_count", ""),
                "stream_qc_kept_observations": qc_row.get("stream_kept_count", ""),
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
    db_file: Path,
    accepted_count: int,
    source_rows: int | None,
    selection_stats: dict[str, Any],
    counts: dict[str, dict[str, int]],
    masks: dict[str, PackedMask],
    stream_qc: list[dict[str, Any]],
    started_utc: str,
    logger: RunLogger,
) -> None:
    logger.log("writing manifests and metadata")
    count_rows = make_count_rows(variants, counts, masks, accepted_count, source_rows)
    manifest_rows = make_manifest_rows(variants, count_rows, stream_qc, bool(args.write_streams))
    write_csv(args.output_dir / "gentle_egm2_counts.csv", count_rows)
    write_csv(args.output_dir / "gentle_egm2_manifest.tsv", manifest_rows, delimiter="\t")
    parameters = {
        "source_out_dir": str(args.source_out_dir),
        "source_stream": str(args.source_stream),
        "source_cache": str(db_file),
        "output_dir": str(args.output_dir),
        "fractions": [text for text, _fraction in args.fraction_items],
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "score_definition": {"score_id": "eg_m2", "formula": "Eg * M2"},
        "candidate_domain": "full-population all-domain scoreable accepted observations from full_population_cache.sqlite",
        "filtering_rule": {
            "grouping": "exact signed h,k,l",
            "rank": "Eg*M2 descending, exact_key_text ascending for ties",
            "n_remove": "floor(drop_fraction * n_obs), capped to leave at least 2 observations",
            "min_remaining": MIN_REMAINING,
            "symmetry_canonicalization": False,
            "non_selected_observations": "retained unchanged",
            "non_scoreable_source_rows": "retained unchanged",
        },
    }
    write_json(args.output_dir / "parameters.json", parameters)
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
            "source_cache": str(db_file),
            "source_stream": str(args.source_stream),
            "source_parameters": str(args.source_out_dir / "parameters.json"),
            "source_validation": str(args.source_out_dir / "validation.json"),
        },
        "accepted_population_count": int(accepted_count),
        "source_reflection_row_count": "" if source_rows is None else int(source_rows),
        "selection_stats": selection_stats,
        "variant_count": len(variants),
        "stream_qc": stream_qc,
        "outputs": {
            "manifest": str(args.output_dir / "gentle_egm2_manifest.tsv"),
            "counts": str(args.output_dir / "gentle_egm2_counts.csv"),
            "parameters": str(args.output_dir / "parameters.json"),
            "metadata": str(args.output_dir / "run_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "run_metadata.json", metadata)


def main() -> int:
    args = parse_args()
    variants = build_variants(args.fraction_items, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(planned_output_paths(args.output_dir, variants, bool(args.write_streams)))

    logger = RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        logger.log("full-pop gentle EgM2 stream builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"fractions={','.join(text for text, _fraction in args.fraction_items)}")
        logger.log(f"workers={int(args.workers)}")

        db_file = cache_db_path(args.source_out_dir)
        if not db_file.is_file():
            raise SystemExit(f"full-population cache not found: {db_file}")
        require_cache_schema(db_file)
        accepted_count = score_cache_count(db_file)
        source_rows = source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and source_rows < accepted_count:
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than accepted cache count {accepted_count:,}")

        masks, counts, selection_stats = construct_masks(db_file, variants, accepted_count, int(args.workers), logger)
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            stream_qc = rewrite_streams(args.source_stream, db_file, variants, masks, logger, source_rows)
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(args, variants, db_file, accepted_count, source_rows, selection_stats, counts, masks, stream_qc, started_utc, logger)
        logger.log("full-pop gentle EgM2 stream builder complete")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
