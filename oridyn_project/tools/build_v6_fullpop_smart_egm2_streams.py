#!/usr/bin/env python3
"""Build smart full-population EgM2 filter streams and controls.

The script reads an existing V6 full-population score cache, plans smart
within-HKL EgM2 removals, and optionally rewrites stream files.  It does not run
Partialator or merging.  Stream files are written only with --write-streams.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    "oridyn_v6_fullpop_smart_egm2_20260811"
)
DEFAULT_FRACTIONS = ("0.002", "0.005", "0.010", "0.020", "0.030")
DEFAULT_Z_THRESHOLDS = ("1", "2", "3")
DEFAULT_MIN_REDUNDANCIES = ("20", "50", "100")
DEFAULT_GATE_PAIRS = ((1.0, 20), (2.0, 20), (2.0, 50), (3.0, 20), (3.0, 50))
DEFAULT_RANDOM_SEEDS = ("1", "2", "3")

HKL_COLUMNS = ["h", "k", "l"]
MIN_REMAINING = 2
PLAN_BATCH_HKLS = 250
MAX_OPEN_STREAMS = 8
ROBUST_MAD_SCALE = 1.4826
ROBUST_EPS = 1.0e-12
SUPPORTED_RESOLUTION_GATES = ("all", "low", "middle", "high", "exclude_outermost")

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    base_variant_id: str
    variant_role: str
    fraction_text: str
    drop_fraction: float
    z_threshold: float
    min_redundancy: int
    resolution_gate: str
    random_seed: int | None
    output_filename: str
    output_stream: Path


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
        self.logger.log(f"{stage}: started ({total_text}; workers={self.workers}; rss={rss_mb():.1f} MB)")

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


def parse_float_items(value: str | None, default_items: tuple[str, ...], name: str) -> list[tuple[str, float]]:
    text = value if value is not None and str(value).strip() else ",".join(default_items)
    out: list[tuple[str, float]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        parsed = float(item)
        if not math.isfinite(parsed):
            raise SystemExit(f"Invalid {name} value {item!r}")
        out.append((item, parsed))
    if not out:
        raise SystemExit(f"--{name} must contain at least one value")
    return out


def parse_fraction_items(value: str | None) -> list[tuple[str, float]]:
    items = parse_float_items(value, DEFAULT_FRACTIONS, "fractions")
    seen: set[str] = set()
    for label, fraction in items:
        if fraction <= 0.0 or fraction >= 1.0:
            raise SystemExit(f"Invalid drop fraction {label!r}; expected 0 < fraction < 1")
        if label in seen:
            raise SystemExit(f"Duplicate fraction label: {label}")
        seen.add(label)
    return items


def parse_int_items(value: str | None, default_items: tuple[str, ...], name: str) -> list[int]:
    text = value if value is not None and str(value).strip() else ",".join(default_items)
    out: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < MIN_REMAINING:
            raise SystemExit(f"Invalid {name} value {item!r}; expected >= {MIN_REMAINING}")
        out.append(parsed)
    if not out:
        raise SystemExit(f"--{name} must contain at least one value")
    return out


def parse_seed_items(value: str | None) -> list[int]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_RANDOM_SEEDS)
    out: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise SystemExit(f"Invalid random-seeds value {item!r}; expected >= 0")
        out.append(parsed)
    if not out:
        raise SystemExit("--random-seeds must contain at least one value")
    return out


def fraction_label(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("+"):
        stripped = stripped[1:]
    return stripped.replace(".", "p").replace("-", "m")


def number_label(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p").replace("-", "m")


def gate_pairs_from_args(args: argparse.Namespace) -> list[tuple[float, int]]:
    if args.z_thresholds is None and args.min_redundancies is None:
        return list(DEFAULT_GATE_PAIRS)
    z_items = parse_float_items(args.z_thresholds, DEFAULT_Z_THRESHOLDS, "z-thresholds")
    min_redundancies = parse_int_items(args.min_redundancies, DEFAULT_MIN_REDUNDANCIES, "min-redundancies")
    pairs: list[tuple[float, int]] = []
    seen: set[tuple[float, int]] = set()
    for _z_text, z_threshold in z_items:
        if z_threshold < 0:
            raise SystemExit("--z-thresholds values must be >= 0")
        for min_redundancy in min_redundancies:
            pair = (float(z_threshold), int(min_redundancy))
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    return pairs


def build_variants(
    fractions: list[tuple[str, float]],
    gate_pairs: list[tuple[float, int]],
    random_seeds: list[int],
    out_dir: Path,
) -> tuple[list[VariantSpec], list[VariantSpec], list[VariantSpec]]:
    targeted: list[VariantSpec] = []
    randoms: list[VariantSpec] = []
    low_risk: list[VariantSpec] = []
    for z_threshold, min_redundancy in gate_pairs:
        z_label = number_label(z_threshold)
        for fraction_text, fraction in fractions:
            drop_label = fraction_label(fraction_text)
            base = f"smart_egm2_z{z_label}_minred{int(min_redundancy)}_drop{drop_label}"
            targeted.append(
                VariantSpec(
                    variant_id=base,
                    base_variant_id=base,
                    variant_role="targeted_high_egm2",
                    fraction_text=fraction_text,
                    drop_fraction=float(fraction),
                    z_threshold=float(z_threshold),
                    min_redundancy=int(min_redundancy),
                    resolution_gate="all",
                    random_seed=None,
                    output_filename=f"{base}.stream",
                    output_stream=out_dir / f"{base}.stream",
                )
            )
            low_id = f"{base}_low_risk"
            low_risk.append(
                VariantSpec(
                    variant_id=low_id,
                    base_variant_id=base,
                    variant_role="low_risk",
                    fraction_text=fraction_text,
                    drop_fraction=float(fraction),
                    z_threshold=float(z_threshold),
                    min_redundancy=int(min_redundancy),
                    resolution_gate="all",
                    random_seed=None,
                    output_filename=f"{low_id}.stream",
                    output_stream=out_dir / f"{low_id}.stream",
                )
            )
            for seed in random_seeds:
                random_id = f"{base}_matched_random_seed{int(seed)}"
                randoms.append(
                    VariantSpec(
                        variant_id=random_id,
                        base_variant_id=base,
                        variant_role="matched_random",
                        fraction_text=fraction_text,
                        drop_fraction=float(fraction),
                        z_threshold=float(z_threshold),
                        min_redundancy=int(min_redundancy),
                        resolution_gate="all",
                        random_seed=int(seed),
                        output_filename=f"{random_id}.stream",
                        output_stream=out_dir / f"{random_id}.stream",
                    )
                )
    all_ids = [variant.variant_id for variant in targeted + low_risk + randoms]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Internal error: duplicate variant ids generated")
    return targeted, low_risk, randoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fractions", default=",".join(DEFAULT_FRACTIONS))
    parser.add_argument("--z-thresholds", default=None)
    parser.add_argument("--min-redundancies", default=None)
    parser.add_argument("--random-seeds", default=",".join(DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    args = parser.parse_args()

    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.fraction_items = parse_fraction_items(args.fractions)
    args.gate_pairs = gate_pairs_from_args(args)
    args.random_seed_items = parse_seed_items(args.random_seeds)
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
        out_dir / "smart_egm2_plan.csv",
        out_dir / "smart_egm2_manifest.tsv",
        out_dir / "smart_egm2_counts.csv",
        out_dir / "smart_egm2_random_manifest.tsv",
        out_dir / "smart_egm2_parameters.json",
        out_dir / "smart_egm2_metadata.json",
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


def require_cache_schema(db_file: Path) -> list[str]:
    present = cache_schema(db_file)
    required = {"ordinal", "h", "k", "l", "exact_key_text", "source_order", "Eg", "M2"}
    missing = sorted(required - set(present))
    if missing:
        raise SystemExit(f"full_population_cache.sqlite score_cache is missing required columns: {missing}")
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


def resolution_support(schema: list[str]) -> dict[str, Any]:
    candidates = ["d_spacing", "d", "resolution", "res", "shell", "resolution_shell"]
    found = [column for column in candidates if column in set(schema)]
    if not found:
        return {
            "available": False,
            "active_gate": "all",
            "supported_gates": list(SUPPORTED_RESOLUTION_GATES),
            "note": "No d-spacing or shell column found in score_cache; resolution-aware filtering left at gate=all.",
        }
    return {
        "available": True,
        "active_gate": "all",
        "supported_gates": list(SUPPORTED_RESOLUTION_GATES),
        "available_columns": found,
        "note": "Resolution metadata was detected, but this moderate default run uses gate=all.",
    }


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


def removal_count(eligible_count: int, total_hkl_obs: int, drop_fraction: float) -> int:
    raw = int(math.floor(float(drop_fraction) * int(eligible_count)))
    cap = int(total_hkl_obs) - MIN_REMAINING
    return int(max(0, min(raw, cap, int(eligible_count))))


def stable_seed(seed: int, base_variant_id: str, h: int, k: int, l: int) -> int:
    payload = f"{int(seed)}|{base_variant_id}|{int(h)}|{int(k)}|{int(l)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF


def empty_count_row(variant: VariantSpec) -> dict[str, int]:
    return {
        "hkl_groups_seen": 0,
        "hkl_groups_passing_min_redundancy": 0,
        "hkl_groups_with_eligible_observations": 0,
        "hkl_groups_with_removal": 0,
        "candidate_observation_count": 0,
        "eligible_observation_count": 0,
        "actionable_observation_count": 0,
        "removed_count": 0,
        "per_hkl_count_mismatches": 0 if variant.variant_role in {"matched_random", "low_risk"} else 0,
    }


def worker_select_batch(task: tuple[str, list[tuple[int, int, int]], list[dict[str, Any]]]) -> dict[str, Any]:
    db_file, hkls, variants_payload = task
    variants = [
        VariantSpec(output_stream=Path(row["output_stream"]), **{key: value for key, value in row.items() if key != "output_stream"})
        for row in variants_payload
    ]
    targeted_by_base = {variant.base_variant_id: variant for variant in variants if variant.variant_role == "targeted_high_egm2"}
    controls_by_base: dict[str, list[VariantSpec]] = {}
    for variant in variants:
        if variant.variant_role != "targeted_high_egm2":
            controls_by_base.setdefault(variant.base_variant_id, []).append(variant)

    table = fetch_score_cache_batch(db_file, hkls)
    counts = {variant.variant_id: empty_count_row(variant) for variant in variants}
    ordinals: dict[str, list[int]] = {variant.variant_id: [] for variant in variants}
    if table.empty:
        return {"counts": counts, "ordinals": {}, "hkl_count": 0, "observation_count": 0}

    eg = pd.to_numeric(table["Eg"], errors="coerce").to_numpy(dtype=float)
    m2 = pd.to_numeric(table["M2"], errors="coerce").to_numpy(dtype=float)
    score = eg * m2
    if not np.isfinite(score).all():
        raise RuntimeError("Nonfinite EgM2 score encountered in full-population cache")
    table = table.assign(score_eg_m2=score)

    hkl_count = 0
    for hkl, group in table.groupby(HKL_COLUMNS, sort=False):
        hkl_count += 1
        h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        n_obs = int(len(group))
        scores = group["score_eg_m2"].to_numpy(dtype=float, copy=False)
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        denom = ROBUST_MAD_SCALE * mad + ROBUST_EPS
        z_scores = (scores - median) / denom
        group_with_z = group.assign(egm2_z=z_scores)

        for target in targeted_by_base.values():
            target_row = counts[target.variant_id]
            target_row["hkl_groups_seen"] += 1
            target_row["candidate_observation_count"] += n_obs
            if n_obs < target.min_redundancy:
                for control in controls_by_base.get(target.base_variant_id, []):
                    counts[control.variant_id]["hkl_groups_seen"] += 1
                    counts[control.variant_id]["candidate_observation_count"] += n_obs
                continue

            eligible = group_with_z[group_with_z["egm2_z"] >= float(target.z_threshold)]
            eligible_count = int(len(eligible))
            target_row["hkl_groups_passing_min_redundancy"] += 1
            target_row["eligible_observation_count"] += eligible_count
            if eligible_count > 0:
                target_row["hkl_groups_with_eligible_observations"] += 1
            n_remove = removal_count(eligible_count, n_obs, target.drop_fraction)
            if n_remove > 0:
                target_row["hkl_groups_with_removal"] += 1
                target_row["actionable_observation_count"] += eligible_count
                target_row["removed_count"] += n_remove
                removed_target = eligible.sort_values(["score_eg_m2", "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_remove)
                ordinals[target.variant_id].extend(int(value) for value in removed_target["ordinal"].to_numpy(dtype=np.int64, copy=False))

            for control in controls_by_base.get(target.base_variant_id, []):
                control_row = counts[control.variant_id]
                control_row["hkl_groups_seen"] += 1
                control_row["candidate_observation_count"] += n_obs
                control_row["hkl_groups_passing_min_redundancy"] += 1
                control_row["eligible_observation_count"] += eligible_count
                if eligible_count > 0:
                    control_row["hkl_groups_with_eligible_observations"] += 1
                if n_remove <= 0:
                    continue
                control_row["hkl_groups_with_removal"] += 1
                control_row["actionable_observation_count"] += eligible_count
                selected: pd.DataFrame
                if control.variant_role == "low_risk":
                    selected = eligible.sort_values(["score_eg_m2", "exact_key_text"], ascending=[True, True], kind="mergesort").head(n_remove)
                elif control.variant_role == "matched_random":
                    rng = np.random.default_rng(stable_seed(int(control.random_seed or 0), control.base_variant_id, h, k, l))
                    positions = rng.choice(np.arange(eligible_count), size=n_remove, replace=False)
                    selected = eligible.iloc[np.sort(positions)]
                else:
                    raise RuntimeError(f"Unknown control role: {control.variant_role}")
                if int(len(selected)) != int(n_remove):
                    control_row["per_hkl_count_mismatches"] += 1
                control_row["removed_count"] += int(len(selected))
                ordinals[control.variant_id].extend(int(value) for value in selected["ordinal"].to_numpy(dtype=np.int64, copy=False))
    return {
        "counts": counts,
        "ordinals": {key: np.asarray(values, dtype=np.int64) for key, values in ordinals.items() if values},
        "hkl_count": int(hkl_count),
        "observation_count": int(len(table)),
    }


def construct_masks(
    db_file: Path,
    variants: list[VariantSpec],
    mask_bits: int,
    workers: int,
    logger: RunLogger,
) -> tuple[dict[str, PackedMask], dict[str, dict[str, int]], dict[str, Any]]:
    total_hkls = count_signed_hkls(db_file)
    batches = list(iter_hkl_batches(db_file))
    variant_payload = [asdict(variant) for variant in variants]
    masks = {variant.variant_id: PackedMask(mask_bits) for variant in variants}
    counts = {variant.variant_id: empty_count_row(variant) for variant in variants}
    progress = StageProgress(logger, "planning smart EgM2 removals", total_hkls, "HKLs", workers=workers)
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

    mismatches: list[str] = []
    for variant in variants:
        mask_count = masks[variant.variant_id].count()
        selected_count = int(counts[variant.variant_id]["removed_count"])
        if mask_count != selected_count:
            raise SystemExit(f"{variant.variant_id}: mask count {mask_count} != selected count {selected_count}")
        if int(counts[variant.variant_id].get("per_hkl_count_mismatches", 0)) != 0:
            mismatches.append(variant.variant_id)
    if mismatches:
        raise SystemExit(f"Control per-HKL removal-count validation failed for: {', '.join(mismatches[:20])}")
    stats = {
        "signed_hkl_count": int(total_hkls),
        "completed_hkls": int(completed_hkls),
        "completed_observations": int(completed_observations),
        "batch_count": int(len(batches)),
        "workers": int(workers),
        "rss_mb_after_selection": rss_mb(),
        "control_per_hkl_removal_counts_validated": True,
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
    progress = StageProgress(logger, "rewriting smart EgM2 streams", len(variants), "streams", workers=1)
    for start in range(0, len(variants), MAX_OPEN_STREAMS):
        batch = variants[start : start + MAX_OPEN_STREAMS]
        rows.extend(rewrite_stream_batch(source_stream, db_file, batch, masks, logger, source_rows))
        progress.update(min(len(rows), len(variants)))
    progress.finish(len(variants))
    return rows


def stream_qc_by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["variant_id"]): row for row in rows}


def make_plan_rows(variants: list[VariantSpec]) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": variant.variant_id,
            "base_variant_id": variant.base_variant_id,
            "variant_role": variant.variant_role,
            "score_id": "eg_m2",
            "score_formula": "Eg * M2",
            "z_threshold": variant.z_threshold,
            "min_redundancy": variant.min_redundancy,
            "resolution_gate": variant.resolution_gate,
            "drop_fraction": variant.drop_fraction,
            "drop_fraction_label": variant.fraction_text,
            "random_seed": "" if variant.random_seed is None else int(variant.random_seed),
            "stream_path": str(variant.output_stream),
        }
        for variant in variants
    ]


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
        rows.append(
            {
                "variant_id": variant.variant_id,
                "base_variant_id": variant.base_variant_id,
                "variant_role": variant.variant_role,
                "score_id": "eg_m2",
                "score_formula": "Eg * M2",
                "filtering_target": "all",
                "z_threshold": variant.z_threshold,
                "min_redundancy": variant.min_redundancy,
                "resolution_gate": variant.resolution_gate,
                "drop_fraction": float(variant.drop_fraction),
                "drop_fraction_label": variant.fraction_text,
                "random_seed": "" if variant.random_seed is None else int(variant.random_seed),
                "accepted_population_count": int(accepted_count),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "hkl_groups_seen": int(count["hkl_groups_seen"]),
                "hkl_groups_passing_min_redundancy": int(count["hkl_groups_passing_min_redundancy"]),
                "hkl_groups_with_eligible_observations": int(count["hkl_groups_with_eligible_observations"]),
                "hkl_groups_with_removal": int(count["hkl_groups_with_removal"]),
                "candidate_observation_count": int(count["candidate_observation_count"]),
                "eligible_observation_count": int(count["eligible_observation_count"]),
                "actionable_observation_count": int(count["actionable_observation_count"]),
                "accepted_observations_removed": int(removed),
                "accepted_observations_retained": int(accepted_count - removed),
                "out_of_analysis_source_rows_retained": "" if source_rows is None else int(source_rows - accepted_count),
                "total_source_rows_retained": "" if source_rows is None else int(source_rows - removed),
                "removed_fraction_of_eligible_domain": float(removed / max(1, count["eligible_observation_count"])),
                "removed_fraction_of_scoreable_population": float(removed / max(1, accepted_count)),
                "removed_fraction_of_source_rows": "" if source_rows is None else float(removed / max(1, source_rows)),
                "per_hkl_count_mismatches": int(count.get("per_hkl_count_mismatches", 0)),
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
                "base_variant_id": variant.base_variant_id,
                "variant_role": variant.variant_role,
                "stream_path": str(variant.output_stream),
                "score_id": "eg_m2",
                "score_formula": "Eg * M2",
                "experiment_type": "fullpop_smart_filter",
                "target": "all",
                "z_threshold": variant.z_threshold,
                "min_redundancy": variant.min_redundancy,
                "resolution_gate": variant.resolution_gate,
                "fraction": float(variant.drop_fraction),
                "fraction_label": variant.fraction_text,
                "random_seed": "" if variant.random_seed is None else int(variant.random_seed),
                "eligible_observation_count": int(count["eligible_observation_count"]),
                "actual_removed_count": int(count["accepted_observations_removed"]),
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


def validation_summary(
    variants: list[VariantSpec],
    counts: dict[str, dict[str, int]],
    stream_qc: list[dict[str, Any]],
) -> dict[str, Any]:
    control_mismatches = {
        variant.variant_id: int(counts[variant.variant_id].get("per_hkl_count_mismatches", 0))
        for variant in variants
        if variant.variant_role in {"matched_random", "low_risk"}
    }
    qc = stream_qc_by_variant(stream_qc)
    return {
        "matched_controls_exact_same_per_hkl_removal_counts": all(value == 0 for value in control_mismatches.values()),
        "control_per_hkl_count_mismatches": control_mismatches,
        "source_order_preserved": all(bool(row.get("source_order_preserved", True)) for row in qc.values()) if qc else True,
        "stream_rewrite_checked": bool(stream_qc),
    }


def write_outputs(
    args: argparse.Namespace,
    variants: list[VariantSpec],
    random_variants: list[VariantSpec],
    db_file: Path,
    accepted_count: int,
    source_rows: int | None,
    cache_stats: dict[str, int],
    cache_columns: list[str],
    selection_stats: dict[str, Any],
    counts: dict[str, dict[str, int]],
    masks: dict[str, PackedMask],
    stream_qc: list[dict[str, Any]],
    started_utc: str,
    logger: RunLogger,
) -> None:
    logger.log("writing smart EgM2 manifests and metadata")
    plan_rows = make_plan_rows(variants)
    count_rows = make_count_rows(variants, counts, masks, accepted_count, source_rows)
    manifest_rows = make_manifest_rows(variants, count_rows, stream_qc, bool(args.write_streams))
    random_ids = {variant.variant_id for variant in random_variants}
    random_manifest_rows = [row for row in manifest_rows if row["variant_id"] in random_ids]
    write_csv(args.output_dir / "smart_egm2_plan.csv", plan_rows)
    write_csv(args.output_dir / "smart_egm2_counts.csv", count_rows)
    write_csv(args.output_dir / "smart_egm2_manifest.tsv", manifest_rows, delimiter="\t")
    write_csv(args.output_dir / "smart_egm2_random_manifest.tsv", random_manifest_rows, delimiter="\t")

    parameters = {
        "source_out_dir": str(args.source_out_dir),
        "source_stream": str(args.source_stream),
        "source_cache": str(db_file),
        "output_dir": str(args.output_dir),
        "fractions": [text for text, _fraction in args.fraction_items],
        "gate_pairs": [{"z_threshold": z, "min_redundancy": minred} for z, minred in args.gate_pairs],
        "random_seeds": args.random_seed_items,
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "score_definition": {"score_id": "eg_m2", "formula": "Eg * M2"},
        "candidate_domain": "full-population all-domain scoreable accepted observations from full_population_cache.sqlite",
        "filtering_rule": {
            "grouping": "exact signed h,k,l",
            "symmetry_canonicalization": False,
            "score_gap_gate": "z = (Eg*M2 - median_hkl) / (1.4826*MAD_hkl + 1e-12); eligible if z >= threshold",
            "redundancy_gate": "signed HKL group is filterable only when n_obs >= min_redundancy",
            "resolution_gate": "all",
            "n_remove": "floor(drop_fraction * eligible_count), capped to leave at least 2 observations in the signed HKL",
            "targeted_selection": "highest Eg*M2 among eligible observations; exact_key_text ascending for ties",
            "matched_random_controls": "same eligible domain and same per-HKL removal counts, deterministic per seed/HKL",
            "low_risk_controls": "lowest Eg*M2 among eligible observations with same per-HKL removal counts",
            "non_selected_observations": "retained unchanged",
            "non_scoreable_source_rows": "retained unchanged",
            "source_stream_order": "preserved by line-order rewriting",
        },
    }
    write_json(args.output_dir / "smart_egm2_parameters.json", parameters)
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
        "source_cache_columns": cache_columns,
        "resolution_gate_support": resolution_support(cache_columns),
        "cache_stats": cache_stats,
        "accepted_population_count": int(accepted_count),
        "source_reflection_row_count": "" if source_rows is None else int(source_rows),
        "selection_stats": selection_stats,
        "validation": validation_summary(variants, counts, stream_qc),
        "variant_count": len(variants),
        "targeted_variant_count": sum(1 for variant in variants if variant.variant_role == "targeted_high_egm2"),
        "low_risk_variant_count": sum(1 for variant in variants if variant.variant_role == "low_risk"),
        "random_variant_count": len(random_variants),
        "stream_qc": stream_qc,
        "outputs": {
            "plan": str(args.output_dir / "smart_egm2_plan.csv"),
            "manifest": str(args.output_dir / "smart_egm2_manifest.tsv"),
            "counts": str(args.output_dir / "smart_egm2_counts.csv"),
            "random_manifest": str(args.output_dir / "smart_egm2_random_manifest.tsv"),
            "parameters": str(args.output_dir / "smart_egm2_parameters.json"),
            "metadata": str(args.output_dir / "smart_egm2_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "smart_egm2_metadata.json", metadata)


def main() -> int:
    args = parse_args()
    targeted, low_risk, randoms = build_variants(args.fraction_items, args.gate_pairs, args.random_seed_items, args.output_dir)
    variants = targeted + low_risk + randoms
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(planned_output_paths(args.output_dir, variants, bool(args.write_streams)))

    logger = RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        logger.log("full-pop smart EgM2 stream builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"fractions={','.join(text for text, _fraction in args.fraction_items)}")
        logger.log(f"gate_pairs={args.gate_pairs}")
        logger.log(f"random_seeds={args.random_seed_items}")
        logger.log(f"variants={len(variants)} ({len(targeted)} targeted, {len(low_risk)} low-risk, {len(randoms)} random)")
        logger.log(f"workers={int(args.workers)}")

        db_file = cache_db_path(args.source_out_dir)
        if not db_file.is_file():
            raise SystemExit(f"full-population cache not found: {db_file}")
        cache_columns = require_cache_schema(db_file)
        cache_stats = score_cache_stats(db_file)
        accepted_count = int(cache_stats["count"])
        if accepted_count <= 0:
            raise SystemExit("score_cache is empty")
        source_rows = source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and source_rows < accepted_count:
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than accepted cache count {accepted_count:,}")
        logger.log(f"resolution gate status: {resolution_support(cache_columns)['note']}")

        masks, counts, selection_stats = construct_masks(db_file, variants, int(cache_stats["mask_bits"]), int(args.workers), logger)
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            stream_qc = rewrite_streams(args.source_stream, db_file, variants, masks, logger, source_rows)
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(
            args,
            variants,
            randoms,
            db_file,
            accepted_count,
            source_rows,
            cache_stats,
            cache_columns,
            selection_stats,
            counts,
            masks,
            stream_qc,
            started_utc,
            logger,
        )
        logger.log("full-pop smart EgM2 stream builder complete")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
