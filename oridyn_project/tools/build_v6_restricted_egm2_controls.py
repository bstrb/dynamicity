#!/usr/bin/env python3
"""Build control streams for the old restricted V6 EgM2 filtering pilot.

This script reconstructs the old restricted score-cache candidate domains and
creates matched control removal sets.  It never runs Partialator or merging.
Real stream files are written only when --write-streams is supplied.
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
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_OLD_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_score_target_filter_map_20260716"
)
DEFAULT_SOURCE_STREAM = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "MFM300-VIII_cut_20-0_3.stream"
)
DEFAULT_TARGETS = (
    "filter_all_eg_m2_drop05",
    "filter_higheg_eg_m2_drop20",
    "filter_higheg_eg_m2_drop30",
)
DEFAULT_CONTROLS = ("matched_random", "low_risk", "eg_only", "score_shuffled")
DEFAULT_RANDOM_SEEDS = (1, 2, 3, 4, 5)

KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
COLUMN_ALIASES = {
    "Eg": ["Eg", "target_excitation_Eg"],
    "M2": ["M2", "excitation_coupling_sq_sum_sg175_sc100"],
}
STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")
TARGET_RE = re.compile(r"^filter_(all|higheg)_eg_m2_drop(\d+)$")

HIGH_EG_FRACTION = 0.30
TARGET_ALL_MIN_OBSERVATIONS = 10
MIN_HIGH_EG_OBSERVATIONS = 10
MIN_REMAINING = 2
MAX_OPEN_STREAMS = 8
CSV_FLOAT_FORMAT = "%.12g"


@dataclass(frozen=True)
class TargetSpec:
    variant_id: str
    filtering_target: str
    drop_fraction: float
    high_eg_fraction: float | None
    expected_removed_count: int | None
    old_retained_count: int | None
    old_removed_fraction_of_scoreable_cache: float | None


@dataclass(frozen=True)
class ControlSpec:
    control_id: str
    target_variant_id: str
    control_type: str
    seed: int | None
    output_filename: str
    output_stream: Path


@dataclass
class TargetSummary:
    target_variant_id: str
    filtering_target: str
    drop_fraction: float
    hkl_groups: int = 0
    actionable_hkl_groups: int = 0
    scoreable_observations: int = 0
    candidate_observation_count: int = 0
    actionable_candidate_observation_count: int = 0
    planned_removed_count: int = 0
    expected_removed_count: int | None = None
    candidate_signature_sha256: str = ""
    per_hkl_removal_signature_sha256: str = ""


class RunLogger:
    def __init__(self, out_dir: Path | None, progress_every: int = 100_000) -> None:
        self.out_dir = out_dir
        self.progress_every = int(progress_every)
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
    def __init__(self, logger: RunLogger, stage: str, total: int | None, unit: str = "items") -> None:
        self.logger = logger
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.completed = 0
        self.started = time.monotonic()
        self.last = self.started
        total_text = f"{self.total:,} {unit}" if self.total is not None else f"unknown {unit}"
        self.logger.log(f"{stage}: started ({total_text})")

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
            remaining = max(0, self.total - self.completed)
            eta = remaining / rate if rate > 0 else float("inf")
            self.logger.log(
                f"{self.stage}: {self.completed:,}/{self.total:,} {self.unit} "
                f"({pct:.1f}%), elapsed={format_duration(elapsed)}, "
                f"rate={rate:,.1f}/s, eta={format_duration(eta)}"
            )
        else:
            self.logger.log(
                f"{self.stage}: {self.completed:,} {self.unit}, "
                f"elapsed={format_duration(elapsed)}, rate={rate:,.1f}/s"
            )

    def advance(self, amount: int = 1) -> None:
        self.update(self.completed + int(amount))

    def finish(self, completed: int | None = None) -> None:
        if completed is not None:
            self.completed = int(completed)
        self.update(self.completed, force=True)
        self.logger.log(f"{self.stage}: finished")


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
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
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


def add_exact_key_text(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise SystemExit(f"Table contains noninteger {column} values")
        out[column] = values.astype("int64")
    out["exact_key_text"] = [
        key_to_text(source, event, h, k, l)
        for source, event, h, k, l in out.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    return out


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


def percent_label(fraction: float) -> str:
    return f"{int(round(float(fraction) * 100.0)):02d}"


def parse_csv_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if out == ["all"]:
        return list(default)
    return out


def parse_seed_list(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return list(DEFAULT_RANDOM_SEEDS)
    seeds: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise SystemExit("--random-seeds must contain at least one integer seed")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument("--random-seeds", default=",".join(str(seed) for seed in DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    args.old_dir = args.old_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.targets = parse_csv_list(args.targets, DEFAULT_TARGETS)
    args.controls = parse_csv_list(args.controls, DEFAULT_CONTROLS)
    args.random_seeds = parse_seed_list(args.random_seeds)
    args.workers = max(1, int(args.workers))
    args.progress_every = max(1, int(args.progress_every))
    if args.dry_run and args.write_streams:
        raise SystemExit("Use either --dry-run or --write-streams, not both")
    if not args.write_streams:
        args.dry_run = True
    if not args.old_dir.is_dir():
        raise SystemExit(f"--old-dir not found: {args.old_dir}")
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    allowed_controls = set(DEFAULT_CONTROLS)
    unknown_controls = sorted(set(args.controls) - allowed_controls)
    if unknown_controls:
        raise SystemExit(f"Unknown --controls value(s): {unknown_controls}; allowed={sorted(allowed_controls)}")
    unknown_targets = [target for target in args.targets if not TARGET_RE.match(target)]
    if unknown_targets:
        raise SystemExit(f"Unsupported --targets value(s): {unknown_targets}; expected filter_all/filter_higheg eg_m2 drop variants")
    return args


def find_column(header: list[str], canonical: str) -> str:
    for candidate in COLUMN_ALIASES[canonical]:
        if candidate in header:
            return candidate
    raise SystemExit(f"Cache is missing required {canonical} column; tried {COLUMN_ALIASES[canonical]}")


def load_old_context(old_dir: Path) -> dict[str, Any]:
    parameters = read_json(old_dir / "parameters.json")
    validation = read_json(old_dir / "validation.json")
    run_metadata = read_json(old_dir / "run_metadata.json")
    cache = Path(str(parameters.get("cache") or ""))
    if not cache.is_file():
        raise SystemExit(f"Old-dir parameters.json cache is missing or unreadable: {cache}")
    return {
        "parameters": parameters,
        "validation": validation,
        "run_metadata": run_metadata,
        "cache": cache.expanduser().resolve(),
    }


def load_target_specs(old_dir: Path, target_ids: list[str]) -> list[TargetSpec]:
    manifest_path = old_dir / "stream_manifest.csv"
    manifest_by_id: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("variant_id"):
                    manifest_by_id[str(row["variant_id"])] = row

    specs: list[TargetSpec] = []
    for target_id in target_ids:
        match = TARGET_RE.match(target_id)
        if not match:
            raise SystemExit(f"Cannot parse target variant id: {target_id}")
        parsed_target = match.group(1)
        parsed_drop = int(match.group(2)) / 100.0
        row = manifest_by_id.get(target_id, {})
        filtering_target = str(row.get("filtering_target") or parsed_target)
        drop_fraction = float(row.get("drop_fraction") or parsed_drop)
        if filtering_target != parsed_target:
            raise SystemExit(f"{target_id}: manifest filtering_target={filtering_target!r} disagrees with variant id")
        high_text = row.get("high_Eg_fraction") or row.get("high_eg_fraction") or ""
        high_fraction = float(high_text) if str(high_text).strip() else None
        expected = int(row["selected_or_removed_count"]) if str(row.get("selected_or_removed_count", "")).strip() else None
        retained = int(row["retained_count"]) if str(row.get("retained_count", "")).strip() else None
        removed_fraction = (
            float(row["global_accepted_observation_fraction_removed"])
            if str(row.get("global_accepted_observation_fraction_removed", "")).strip()
            else None
        )
        specs.append(
            TargetSpec(
                variant_id=target_id,
                filtering_target=filtering_target,
                drop_fraction=drop_fraction,
                high_eg_fraction=high_fraction,
                expected_removed_count=expected,
                old_retained_count=retained,
                old_removed_fraction_of_scoreable_cache=removed_fraction,
            )
        )
    return specs


def build_control_specs(targets: list[TargetSpec], controls: list[str], seeds: list[int], out_dir: Path) -> list[ControlSpec]:
    specs: list[ControlSpec] = []
    for target in targets:
        if "matched_random" in controls:
            for seed in seeds:
                control_id = f"{target.variant_id}__matched_random_seed{int(seed):03d}"
                specs.append(
                    ControlSpec(
                        control_id=control_id,
                        target_variant_id=target.variant_id,
                        control_type="matched_random",
                        seed=int(seed),
                        output_filename=f"{control_id}.stream",
                        output_stream=out_dir / f"{control_id}.stream",
                    )
                )
        if "low_risk" in controls:
            control_id = f"{target.variant_id}__low_risk"
            specs.append(
                ControlSpec(
                    control_id=control_id,
                    target_variant_id=target.variant_id,
                    control_type="low_risk",
                    seed=None,
                    output_filename=f"{control_id}.stream",
                    output_stream=out_dir / f"{control_id}.stream",
                )
            )
        if "eg_only" in controls:
            control_id = f"{target.variant_id}__eg_only"
            specs.append(
                ControlSpec(
                    control_id=control_id,
                    target_variant_id=target.variant_id,
                    control_type="eg_only",
                    seed=None,
                    output_filename=f"{control_id}.stream",
                    output_stream=out_dir / f"{control_id}.stream",
                )
            )
        if "score_shuffled" in controls:
            seed = int(seeds[0])
            control_id = f"{target.variant_id}__score_shuffled_seed{seed:03d}"
            specs.append(
                ControlSpec(
                    control_id=control_id,
                    target_variant_id=target.variant_id,
                    control_type="score_shuffled",
                    seed=seed,
                    output_filename=f"{control_id}.stream",
                    output_stream=out_dir / f"{control_id}.stream",
                )
            )
    return specs


def planned_output_paths(out_dir: Path, controls: list[ControlSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "control_manifest.tsv",
        out_dir / "control_summary.csv",
        out_dir / "pilot_domain_audit.md",
        out_dir / "parameters.json",
        out_dir / "run_metadata.json",
        out_dir / "run.log",
    ]
    if write_streams:
        paths.extend(control.output_stream for control in controls)
    return paths


def refuse_overwrite(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        preview = "\n  ".join(str(path) for path in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ... {len(existing) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {preview}{extra}")


def load_score_cache(cache_path: Path, logger: RunLogger) -> tuple[pd.DataFrame, dict[str, Any]]:
    progress = StageProgress(logger, "loading restricted score cache", total=1, unit="files")
    header = pd.read_csv(cache_path, nrows=0, compression="infer").columns.tolist()
    missing_keys = [column for column in KEY_COLUMNS if column not in header]
    if missing_keys:
        raise SystemExit(f"Cache is missing exact-key columns: {missing_keys}")
    column_map = {name: find_column(header, name) for name in ["Eg", "M2"]}
    usecols = sorted(set(KEY_COLUMNS + list(column_map.values()) + (["exact_key_text"] if "exact_key_text" in header else [])))
    table = pd.read_csv(cache_path, usecols=usecols, low_memory=False, compression="infer")
    table = add_exact_key_text(table)
    duplicates = table.duplicated("exact_key_text", keep=False)
    if duplicates.any():
        raise SystemExit(f"Cache contains duplicate exact observation keys: {int(duplicates.sum())} duplicate rows")
    for canonical, source_column in column_map.items():
        values = pd.to_numeric(table[source_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            raise SystemExit(f"Cache contains nonfinite values in {source_column} for {canonical}")
        table[canonical] = values.astype(float)
    table["score_eg_m2"] = table["Eg"].to_numpy(dtype=float) * table["M2"].to_numpy(dtype=float)
    if not np.isfinite(table["score_eg_m2"].to_numpy(dtype=float)).all():
        raise SystemExit("Cache produced nonfinite EgM2 scores")
    table = table.sort_values(["h", "k", "l", "exact_key_text"], kind="mergesort").reset_index(drop=True)
    progress.finish(1)
    return table, {
        "cache_rows": int(len(table)),
        "column_map": column_map,
        "exact_signed_hkl_count": int(table.loc[:, HKL_COLUMNS].drop_duplicates().shape[0]),
    }


def removal_count(n_eligible: int, drop_fraction: float, min_remaining: int = MIN_REMAINING) -> int:
    raw = int(math.floor(float(drop_fraction) * int(n_eligible)))
    cap = int(n_eligible) - int(min_remaining)
    return int(min(raw, cap))


def stable_seed(base_seed: int, control_id: str, hkl: tuple[int, int, int]) -> int:
    text = f"{base_seed}|{control_id}|{hkl[0]},{hkl[1]},{hkl[2]}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def sorted_indices(keys: list[str], values: list[float], descending: bool) -> list[int]:
    if descending:
        return sorted(range(len(keys)), key=lambda idx: (-values[idx], keys[idx]))
    return sorted(range(len(keys)), key=lambda idx: (values[idx], keys[idx]))


def hkl_candidate_indices(
    target: TargetSpec,
    keys: list[str],
    eg: list[float],
) -> tuple[list[int], int, bool]:
    n_obs = len(keys)
    if target.filtering_target == "all":
        candidate = list(range(n_obs)) if n_obs >= TARGET_ALL_MIN_OBSERVATIONS else []
        n_remove = removal_count(len(candidate), target.drop_fraction) if candidate else 0
        actionable = bool(n_obs >= TARGET_ALL_MIN_OBSERVATIONS and n_remove > 0 and len(candidate) - n_remove >= MIN_REMAINING)
        return candidate, n_remove if actionable else 0, actionable
    if target.filtering_target == "higheg":
        n_high = int(math.floor(HIGH_EG_FRACTION * n_obs))
        order = sorted_indices(keys, eg, descending=True)
        candidate = order[:n_high]
        n_remove = removal_count(len(candidate), target.drop_fraction) if candidate else 0
        actionable = bool(len(candidate) >= MIN_HIGH_EG_OBSERVATIONS and n_remove > 0 and len(candidate) - n_remove >= MIN_REMAINING)
        return candidate, n_remove if actionable else 0, actionable
    raise ValueError(f"Unsupported filtering target: {target.filtering_target}")


def select_control_keys(
    control: ControlSpec,
    hkl: tuple[int, int, int],
    candidate: list[int],
    n_remove: int,
    keys: list[str],
    eg: list[float],
    score: list[float],
) -> list[str]:
    if n_remove <= 0:
        return []
    if control.control_type == "matched_random":
        ordered_pool = sorted(candidate, key=lambda idx: keys[idx])
        rng = np.random.default_rng(stable_seed(int(control.seed or 0), control.control_id, hkl))
        picked = rng.choice(len(ordered_pool), size=n_remove, replace=False)
        return [keys[ordered_pool[int(pos)]] for pos in picked]
    if control.control_type == "low_risk":
        ordered = sorted(candidate, key=lambda idx: (score[idx], keys[idx]))
        return [keys[idx] for idx in ordered[:n_remove]]
    if control.control_type == "eg_only":
        ordered = sorted(candidate, key=lambda idx: (-eg[idx], keys[idx]))
        return [keys[idx] for idx in ordered[:n_remove]]
    if control.control_type == "score_shuffled":
        base_order = sorted(range(len(keys)), key=lambda idx: keys[idx])
        base_scores = np.asarray([score[idx] for idx in base_order], dtype=float)
        rng = np.random.default_rng(stable_seed(int(control.seed or 0), control.control_id, hkl))
        shuffled = rng.permutation(base_scores)
        shuffled_by_index = {idx: float(shuffled[pos]) for pos, idx in enumerate(base_order)}
        ordered = sorted(candidate, key=lambda idx: (-shuffled_by_index[idx], keys[idx]))
        return [keys[idx] for idx in ordered[:n_remove]]
    raise ValueError(f"Unsupported control type: {control.control_type}")


def group_payloads(cache: pd.DataFrame) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for hkl, group in cache.groupby(HKL_COLUMNS, sort=False):
        payloads.append(
            {
                "hkl": tuple(int(value) for value in hkl),
                "keys": group["exact_key_text"].astype(str).tolist(),
                "eg": pd.to_numeric(group["Eg"], errors="raise").astype(float).tolist(),
                "score": pd.to_numeric(group["score_eg_m2"], errors="raise").astype(float).tolist(),
            }
        )
    return payloads


def batch_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def process_group_batch(
    batch: list[dict[str, Any]],
    target_dicts: list[dict[str, Any]],
    control_dicts: list[dict[str, Any]],
) -> dict[str, Any]:
    targets = [TargetSpec(**item) for item in target_dicts]
    controls = [ControlSpec(output_stream=Path(item["output_stream"]), **{k: v for k, v in item.items() if k != "output_stream"}) for item in control_dicts]
    controls_by_target: dict[str, list[ControlSpec]] = {}
    for control in controls:
        controls_by_target.setdefault(control.target_variant_id, []).append(control)

    remove_lists: dict[str, list[str]] = {control.control_id: [] for control in controls}
    summaries: dict[str, dict[str, int]] = {
        target.variant_id: {
            "hkl_groups": 0,
            "actionable_hkl_groups": 0,
            "scoreable_observations": 0,
            "candidate_observation_count": 0,
            "actionable_candidate_observation_count": 0,
            "planned_removed_count": 0,
        }
        for target in targets
    }
    hkl_rows: list[tuple[str, int, int, int, int, int, int, bool]] = []

    for payload in batch:
        hkl = tuple(payload["hkl"])
        keys = list(payload["keys"])
        eg = list(payload["eg"])
        score = list(payload["score"])
        for target in targets:
            candidate, n_remove, actionable = hkl_candidate_indices(target, keys, eg)
            summary = summaries[target.variant_id]
            summary["hkl_groups"] += 1
            summary["scoreable_observations"] += len(keys)
            summary["candidate_observation_count"] += len(candidate)
            if actionable:
                summary["actionable_hkl_groups"] += 1
                summary["actionable_candidate_observation_count"] += len(candidate)
                summary["planned_removed_count"] += n_remove
            hkl_rows.append((target.variant_id, int(hkl[0]), int(hkl[1]), int(hkl[2]), len(candidate), n_remove, len(keys), bool(actionable)))
            if n_remove <= 0:
                continue
            for control in controls_by_target.get(target.variant_id, []):
                remove_lists[control.control_id].extend(select_control_keys(control, hkl, candidate, n_remove, keys, eg, score))

    return {"remove_lists": remove_lists, "summaries": summaries, "hkl_rows": hkl_rows}


def signature_from_hkl_rows(hkl_rows: list[tuple[str, int, int, int, int, int, int, bool]], target_id: str, mode: str) -> str:
    digest = hashlib.sha256()
    for row in sorted((row for row in hkl_rows if row[0] == target_id), key=lambda item: (item[1], item[2], item[3])):
        if mode == "candidate":
            payload = f"{row[1]},{row[2]},{row[3]},{row[4]}\n"
        elif mode == "removal":
            payload = f"{row[1]},{row[2]},{row[3]},{row[5]}\n"
        else:
            raise ValueError(mode)
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def construct_control_selections(
    cache: pd.DataFrame,
    targets: list[TargetSpec],
    controls: list[ControlSpec],
    workers: int,
    logger: RunLogger,
) -> tuple[dict[str, set[str]], dict[str, TargetSummary], list[tuple[str, int, int, int, int, int, int, bool]]]:
    groups = group_payloads(cache)
    batches = batch_items(groups, 200)
    target_dicts = [asdict(target) for target in targets]
    control_dicts = [asdict(control) for control in controls]
    progress = StageProgress(logger, "constructing control removal sets", total=len(batches), unit="group batches")
    remove_sets: dict[str, set[str]] = {control.control_id: set() for control in controls}
    summary_acc: dict[str, dict[str, int]] = {
        target.variant_id: {
            "hkl_groups": 0,
            "actionable_hkl_groups": 0,
            "scoreable_observations": 0,
            "candidate_observation_count": 0,
            "actionable_candidate_observation_count": 0,
            "planned_removed_count": 0,
        }
        for target in targets
    }
    hkl_rows: list[tuple[str, int, int, int, int, int, int, bool]] = []

    if workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_group_batch, batch, target_dicts, control_dicts) for batch in batches]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                for control_id, keys in result["remove_lists"].items():
                    remove_sets[control_id].update(keys)
                for target_id, row in result["summaries"].items():
                    for key, value in row.items():
                        summary_acc[target_id][key] += int(value)
                hkl_rows.extend(result["hkl_rows"])
                progress.update(completed)
    else:
        for completed, batch in enumerate(batches, start=1):
            result = process_group_batch(batch, target_dicts, control_dicts)
            for control_id, keys in result["remove_lists"].items():
                remove_sets[control_id].update(keys)
            for target_id, row in result["summaries"].items():
                for key, value in row.items():
                    summary_acc[target_id][key] += int(value)
            hkl_rows.extend(result["hkl_rows"])
            progress.update(completed)

    progress.finish(len(batches))

    summaries: dict[str, TargetSummary] = {}
    targets_by_id = {target.variant_id: target for target in targets}
    for target_id, values in summary_acc.items():
        target = targets_by_id[target_id]
        summaries[target_id] = TargetSummary(
            target_variant_id=target_id,
            filtering_target=target.filtering_target,
            drop_fraction=float(target.drop_fraction),
            hkl_groups=int(values["hkl_groups"]),
            actionable_hkl_groups=int(values["actionable_hkl_groups"]),
            scoreable_observations=int(values["scoreable_observations"]),
            candidate_observation_count=int(values["candidate_observation_count"]),
            actionable_candidate_observation_count=int(values["actionable_candidate_observation_count"]),
            planned_removed_count=int(values["planned_removed_count"]),
            expected_removed_count=target.expected_removed_count,
            candidate_signature_sha256=signature_from_hkl_rows(hkl_rows, target_id, "candidate"),
            per_hkl_removal_signature_sha256=signature_from_hkl_rows(hkl_rows, target_id, "removal"),
        )
    for control in controls:
        target_summary = summaries[control.target_variant_id]
        if len(remove_sets[control.control_id]) != target_summary.planned_removed_count:
            raise SystemExit(
                f"{control.control_id}: control removed-key set has {len(remove_sets[control.control_id])} "
                f"unique keys but target planned {target_summary.planned_removed_count}"
            )
    return remove_sets, summaries, hkl_rows


def read_old_hkl_qc_summaries(old_dir: Path, target_ids: list[str], logger: RunLogger) -> dict[str, dict[str, Any]]:
    path = old_dir / "per_variant_per_hkl_qc.csv"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    wanted = set(target_ids)
    usecols = ["variant_id", "n_observations", "n_eligible", "n_removed", "actionable"]
    out: dict[str, dict[str, Any]] = {
        target_id: {
            "hkl_rows": 0,
            "candidate_observation_count": 0,
            "actionable_candidate_observation_count": 0,
            "planned_removed_count": 0,
            "actionable_hkl_groups": 0,
        }
        for target_id in target_ids
    }
    progress = StageProgress(logger, "validating against old per-HKL QC", total=None, unit="rows")
    rows_seen = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        rows_seen += len(chunk)
        chunk = chunk[chunk["variant_id"].astype(str).isin(wanted)]
        for target_id, group in chunk.groupby("variant_id", sort=False):
            record = out[str(target_id)]
            record["hkl_rows"] += int(len(group))
            n_eligible = pd.to_numeric(group["n_eligible"], errors="coerce").fillna(0).astype(int)
            n_removed = pd.to_numeric(group["n_removed"], errors="coerce").fillna(0).astype(int)
            actionable = group["actionable"].astype(str).str.lower().isin(["true", "1", "yes"])
            record["candidate_observation_count"] += int(n_eligible.sum())
            record["actionable_candidate_observation_count"] += int(n_eligible[actionable].sum())
            record["planned_removed_count"] += int(n_removed.sum())
            record["actionable_hkl_groups"] += int(actionable.sum())
        progress.update(rows_seen)
    progress.finish(rows_seen)
    return {"available": True, "path": str(path), "by_target": out}


def validate_target_summaries(
    targets: list[TargetSpec],
    summaries: dict[str, TargetSummary],
    old_hkl_qc: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for target in targets:
        summary = summaries[target.variant_id]
        if target.expected_removed_count is not None and summary.planned_removed_count != int(target.expected_removed_count):
            raise SystemExit(
                f"{target.variant_id}: reconstructed removal count {summary.planned_removed_count} "
                f"does not match old manifest count {target.expected_removed_count}"
            )
        if old_hkl_qc.get("available"):
            old_row = old_hkl_qc.get("by_target", {}).get(target.variant_id)
            if old_row:
                for key in ["candidate_observation_count", "actionable_candidate_observation_count", "planned_removed_count"]:
                    if int(old_row[key]) != int(getattr(summary, key)):
                        raise SystemExit(
                            f"{target.variant_id}: reconstructed {key}={getattr(summary, key)} "
                            f"does not match old per-HKL QC {old_row[key]}"
                        )
        else:
            warnings.append(f"Old per-HKL QC was not available: {old_hkl_qc.get('path')}")
    return warnings


def rewrite_stream_batch(
    source_stream: Path,
    controls: list[ControlSpec],
    remove_sets: dict[str, set[str]],
    logger: RunLogger,
) -> list[dict[str, Any]]:
    for control in controls:
        if control.output_stream.exists():
            raise SystemExit(f"Refusing to overwrite existing output stream: {control.output_stream}")
    handles = {control.control_id: control.output_stream.open("w", encoding="utf-8") for control in controls}
    stats: dict[str, dict[str, Any]] = {
        control.control_id: {
            "control_id": control.control_id,
            "output_stream": str(control.output_stream),
            "planned_removed_count": int(len(remove_sets[control.control_id])),
            "stream_removed_count": 0,
            "stream_kept_count": 0,
            "stream_reflection_rows_seen": 0,
            "source_order_preserved": True,
            "status": "generated",
        }
        for control in controls
    }
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    progress = StageProgress(logger, f"rewriting stream batch ({len(controls)} controls)", total=None, unit="reflection rows")
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
                    for control_id, handle in handles.items():
                        row = stats[control_id]
                        row["stream_reflection_rows_seen"] += 1
                        if key in remove_sets[control_id]:
                            row["stream_removed_count"] += 1
                        else:
                            handle.write(raw_line)
                            row["stream_kept_count"] += 1
                    progress.update(rows_seen)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()
    progress.finish(rows_seen)
    rows: list[dict[str, Any]] = []
    for control in controls:
        row = stats[control.control_id]
        if int(row["stream_removed_count"]) != int(row["planned_removed_count"]):
            raise SystemExit(
                f"{control.control_id}: stream rewrite removed {row['stream_removed_count']} rows "
                f"but planned {row['planned_removed_count']}"
            )
        rows.append(row)
    return rows


def rewrite_streams(
    source_stream: Path,
    controls: list[ControlSpec],
    remove_sets: dict[str, set[str]],
    logger: RunLogger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = StageProgress(logger, "rewriting control streams", total=len(controls), unit="controls")
    for start in range(0, len(controls), MAX_OPEN_STREAMS):
        batch = controls[start : start + MAX_OPEN_STREAMS]
        rows.extend(rewrite_stream_batch(source_stream, batch, remove_sets, logger))
        progress.update(min(len(rows), len(controls)))
    progress.finish(len(controls))
    return rows


def stream_qc_by_control(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["control_id"]): row for row in rows}


def make_manifest_rows(
    controls: list[ControlSpec],
    targets: dict[str, TargetSpec],
    summaries: dict[str, TargetSummary],
    remove_sets: dict[str, set[str]],
    source_stream: Path,
    source_rows: int | None,
    scoreable_rows: int,
    stream_qc: list[dict[str, Any]],
    write_streams: bool,
) -> list[dict[str, Any]]:
    qc = stream_qc_by_control(stream_qc)
    rows: list[dict[str, Any]] = []
    for control in controls:
        target = targets[control.target_variant_id]
        summary = summaries[control.target_variant_id]
        planned_removed = int(len(remove_sets[control.control_id]))
        qc_row = qc.get(control.control_id, {})
        status = str(qc_row.get("status") or ("generated" if write_streams else "planned_dry_run"))
        rows.append(
            {
                "control_id": control.control_id,
                "target_variant_id": target.variant_id,
                "control_type": control.control_type,
                "seed": "" if control.seed is None else int(control.seed),
                "source_stream": str(source_stream),
                "output_stream": str(control.output_stream),
                "filtering_target": target.filtering_target,
                "score_id": "eg_m2",
                "score_formula": "Eg * M2",
                "drop_fraction": float(target.drop_fraction),
                "high_Eg_fraction": "" if target.high_eg_fraction is None else float(target.high_eg_fraction),
                "scoreable_observation_count": int(scoreable_rows),
                "candidate_observation_count": int(summary.candidate_observation_count),
                "actionable_candidate_observation_count": int(summary.actionable_candidate_observation_count),
                "planned_removed_count": planned_removed,
                "stream_removed_count": qc_row.get("stream_removed_count", ""),
                "stream_kept_count": qc_row.get("stream_kept_count", ""),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "removed_fraction_of_source_rows": "" if source_rows is None else planned_removed / max(1, int(source_rows)),
                "removed_fraction_of_scoreable_cache": planned_removed / max(1, int(scoreable_rows)),
                "removed_fraction_of_candidate_domain": planned_removed / max(1, int(summary.candidate_observation_count)),
                "removed_fraction_of_actionable_candidate_domain": planned_removed / max(1, int(summary.actionable_candidate_observation_count)),
                "per_hkl_removal_signature_sha256": summary.per_hkl_removal_signature_sha256,
                "candidate_signature_sha256": summary.candidate_signature_sha256,
                "status": status,
            }
        )
    return rows


def make_summary_rows(targets: list[TargetSpec], summaries: dict[str, TargetSummary], old_hkl_qc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    old_by_target = old_hkl_qc.get("by_target", {}) if old_hkl_qc.get("available") else {}
    for target in targets:
        summary = summaries[target.variant_id]
        old_row = old_by_target.get(target.variant_id, {})
        rows.append(
            {
                "target_variant_id": target.variant_id,
                "filtering_target": target.filtering_target,
                "drop_fraction": float(target.drop_fraction),
                "high_Eg_fraction": "" if target.high_eg_fraction is None else float(target.high_eg_fraction),
                "hkl_groups": int(summary.hkl_groups),
                "actionable_hkl_groups": int(summary.actionable_hkl_groups),
                "scoreable_observations": int(summary.scoreable_observations),
                "candidate_observation_count": int(summary.candidate_observation_count),
                "actionable_candidate_observation_count": int(summary.actionable_candidate_observation_count),
                "planned_removed_count": int(summary.planned_removed_count),
                "old_manifest_removed_count": "" if target.expected_removed_count is None else int(target.expected_removed_count),
                "old_per_hkl_qc_candidate_count": old_row.get("candidate_observation_count", ""),
                "old_per_hkl_qc_actionable_candidate_count": old_row.get("actionable_candidate_observation_count", ""),
                "old_per_hkl_qc_removed_count": old_row.get("planned_removed_count", ""),
                "candidate_signature_sha256": summary.candidate_signature_sha256,
                "per_hkl_removal_signature_sha256": summary.per_hkl_removal_signature_sha256,
            }
        )
    return rows


def source_reflection_rows_from_validation(validation: dict[str, Any]) -> int | None:
    for path in [
        ("source_stream_validation", "source_reflection_rows"),
        ("stream_preflight", "source_reflection_rows"),
    ]:
        current: Any = validation
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return int(current)
    return None


def scoreable_rows_from_validation(validation: dict[str, Any]) -> int | None:
    for path in [
        ("cache_stats", "cache_rows"),
        ("cache_metadata", "cache_rows_expected_from_metadata"),
    ]:
        current: Any = validation
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return int(current)
    return None


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
    out.update(
        {
            "available": bool(rev.stdout.strip()),
            "commit": rev.stdout.strip(),
            "status_short": status.stdout.splitlines(),
        }
    )
    return out


def make_audit_markdown(
    old_context: dict[str, Any],
    targets: list[TargetSpec],
    summaries: dict[str, TargetSummary],
    manifest_rows: list[dict[str, Any]],
    source_rows: int | None,
    scoreable_rows: int,
    write_streams: bool,
) -> str:
    validation = old_context["validation"]
    unmatched = (
        validation.get("source_stream_validation", {}).get("source_reflection_rows_without_cache_score")
        if isinstance(validation.get("source_stream_validation"), dict)
        else None
    )
    rows = [
        "# Restricted EgM2 Control Stream Audit",
        "",
        "## Scope",
        "",
        "This directory contains matched controls for the old restricted V6 EgM2 filtering pilot. "
        "The script reconstructs candidate domains from the old score cache and metadata; it does not run Partialator, merging, indexing, or stream generation beyond optional line-by-line control stream rewriting.",
        "",
        "## Restricted Score-Cache Domain",
        "",
        f"- Old score cache: `{old_context['cache']}`",
        f"- Scoreable restricted-cache observations: {scoreable_rows:,}",
        f"- Source stream reflection rows: {'unknown' if source_rows is None else f'{source_rows:,}'}",
        f"- Source rows without cache scores retained unchanged: {'unknown' if unmatched is None else f'{int(unmatched):,}'}",
        "- Exact observation key: `source_filename|event|signed h|signed k|signed l`.",
        "- Symmetry canonicalization is not used; signed HKLs are preserved.",
        "",
        "## Candidate Domains",
        "",
        "`all` means all scoreable observations in the restricted cache, grouped by exact signed HKL. "
        "The old `filter_all_eg_m2_drop05` label applies a 5% per-HKL drop rule within this restricted cache, with floor rounding and a minimum of two retained observations.",
        "",
        "`higheg` means the top `floor(0.30 * n_obs)` observations by `Eg` within each exact signed HKL. "
        "It is not a global Eg threshold.  Removal is then applied within that high-Eg pool, again with floor rounding and the min-2-retained rule.",
        "",
        "Non-scoreable stream reflections and observations outside the candidate domain are retained unchanged in every control stream.",
        "",
        "## Target Removal Counts",
        "",
        "| target | domain | nominal drop | candidate obs | actionable candidate obs | removed | removed / source rows | removed / candidate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        summary = summaries[target.variant_id]
        removed = summary.planned_removed_count
        source_pct = "" if source_rows is None else f"{100.0 * removed / max(1, source_rows):.3f}%"
        candidate_pct = f"{100.0 * removed / max(1, summary.candidate_observation_count):.3f}%"
        rows.append(
            f"| `{target.variant_id}` | {target.filtering_target} | {100.0 * target.drop_fraction:.1f}% | "
            f"{summary.candidate_observation_count:,} | {summary.actionable_candidate_observation_count:,} | "
            f"{removed:,} | {source_pct} | {candidate_pct} |"
        )
    rows.extend(
        [
            "",
            "## Why Old drop05 Removes Less Than Full-Population drop05",
            "",
            "The old pilot applied `drop05` only to the 659,147 scoreable rows in the restricted V5-derived cache, while retaining the much larger set of source-stream rows without cache scores unchanged. "
            "The newer V6 formula-comparison `drop05` applies the analogous filtering to the broader full-population accepted cache of 6,732,955 observations. "
            "That larger candidate population is why the newer `filter_all_eg_m2_drop05` removed 229,253 observations, whereas the old restricted pilot removed 19,978.",
            "",
            "## Controls",
            "",
            f"- Mode: {'stream generation' if write_streams else 'dry run; stream paths planned but not written'}",
            "- `matched_random`: same per-HKL removal counts as the target, sampled uniformly from the same candidate pool for each seed.",
            "- `low_risk`: same per-HKL removal counts, removing lowest EgM2 scores.",
            "- `eg_only`: same per-HKL removal counts, removing highest Eg observations.",
            "- `score_shuffled`: same per-HKL removal counts after shuffling EgM2 within each signed HKL with the first configured seed.",
            "",
            "## Control Manifest Preview",
            "",
            "| control_id | target | type | seed | planned removed | status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in manifest_rows[:40]:
        rows.append(
            f"| `{row['control_id']}` | `{row['target_variant_id']}` | {row['control_type']} | "
            f"{row['seed']} | {int(row['planned_removed_count']):,} | {row['status']} |"
        )
    if len(manifest_rows) > 40:
        rows.append(f"| ... | ... | ... | ... | ... | {len(manifest_rows) - 40} more controls |")
    rows.append("")
    return "\n".join(rows)


def write_outputs(
    args: argparse.Namespace,
    old_context: dict[str, Any],
    targets: list[TargetSpec],
    controls: list[ControlSpec],
    cache_stats: dict[str, Any],
    old_hkl_qc: dict[str, Any],
    summaries: dict[str, TargetSummary],
    remove_sets: dict[str, set[str]],
    stream_qc: list[dict[str, Any]],
    warnings: list[str],
    started_utc: str,
    logger: RunLogger,
) -> None:
    source_rows = source_reflection_rows_from_validation(old_context["validation"])
    scoreable_rows = int(cache_stats["cache_rows"])
    target_by_id = {target.variant_id: target for target in targets}
    manifest_rows = make_manifest_rows(
        controls,
        target_by_id,
        summaries,
        remove_sets,
        args.source_stream,
        source_rows,
        scoreable_rows,
        stream_qc,
        bool(args.write_streams),
    )
    summary_rows = make_summary_rows(targets, summaries, old_hkl_qc)
    logger.log("writing control manifests and metadata")
    write_csv(args.output_dir / "control_manifest.tsv", manifest_rows, delimiter="\t")
    write_csv(args.output_dir / "control_summary.csv", summary_rows)
    (args.output_dir / "pilot_domain_audit.md").write_text(
        make_audit_markdown(old_context, targets, summaries, manifest_rows, source_rows, scoreable_rows, bool(args.write_streams)),
        encoding="utf-8",
    )
    parameters = {
        "old_dir": str(args.old_dir),
        "source_stream": str(args.source_stream),
        "output_dir": str(args.output_dir),
        "targets": [target.variant_id for target in targets],
        "controls": list(args.controls),
        "random_seeds": list(args.random_seeds),
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "progress_every": int(args.progress_every),
        "restricted_domain_definition": {
            "all": "all scoreable restricted-cache observations grouped by exact signed HKL",
            "higheg": "top floor(0.30*n_obs) by Eg within each exact signed HKL",
            "exact_key": "source_filename|event|signed h|signed k|signed l",
            "symmetry_canonicalization": False,
            "min_remaining": MIN_REMAINING,
            "target_all_min_observations": TARGET_ALL_MIN_OBSERVATIONS,
            "min_high_Eg_observations": MIN_HIGH_EG_OBSERVATIONS,
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
        "old_context_files": {
            "parameters": str(args.old_dir / "parameters.json"),
            "validation": str(args.old_dir / "validation.json"),
            "run_metadata": str(args.old_dir / "run_metadata.json"),
            "stream_manifest": str(args.old_dir / "stream_manifest.csv"),
            "per_variant_per_hkl_qc": str(args.old_dir / "per_variant_per_hkl_qc.csv"),
        },
        "old_cache": str(old_context["cache"]),
        "cache_stats": cache_stats,
        "old_hkl_qc_validation": old_hkl_qc,
        "target_summaries": {key: asdict(value) for key, value in summaries.items()},
        "control_count": len(controls),
        "stream_qc": stream_qc,
        "warnings": warnings,
        "outputs": {
            "control_manifest": str(args.output_dir / "control_manifest.tsv"),
            "control_summary": str(args.output_dir / "control_summary.csv"),
            "pilot_domain_audit": str(args.output_dir / "pilot_domain_audit.md"),
            "parameters": str(args.output_dir / "parameters.json"),
            "run_metadata": str(args.output_dir / "run_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "run_metadata.json", metadata)


def main() -> int:
    args = parse_args()
    old_context = load_old_context(args.old_dir)
    targets = load_target_specs(args.old_dir, args.targets)
    controls = build_control_specs(targets, list(args.controls), list(args.random_seeds), args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(planned_output_paths(args.output_dir, controls, bool(args.write_streams)))

    logger = RunLogger(args.output_dir, progress_every=int(args.progress_every))
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        logger.log("restricted EgM2 control builder started")
        logger.log(f"old_dir={args.old_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"targets={','.join(target.variant_id for target in targets)}")
        logger.log(f"controls={','.join(args.controls)}; random_seeds={','.join(str(seed) for seed in args.random_seeds)}")
        logger.log(f"workers={args.workers}")

        cache, cache_stats = load_score_cache(Path(old_context["cache"]), logger)
        validation_rows = scoreable_rows_from_validation(old_context["validation"])
        if validation_rows is not None and int(validation_rows) != int(cache_stats["cache_rows"]):
            raise SystemExit(f"Cache row count {cache_stats['cache_rows']} differs from old validation {validation_rows}")

        remove_sets, summaries, _hkl_rows = construct_control_selections(cache, targets, controls, int(args.workers), logger)
        old_hkl_qc = read_old_hkl_qc_summaries(args.old_dir, [target.variant_id for target in targets], logger)
        warnings = validate_target_summaries(targets, summaries, old_hkl_qc)

        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            stream_qc = rewrite_streams(args.source_stream, controls, remove_sets, logger)
        else:
            logger.log("dry-run mode: control stream files were not written")

        write_outputs(
            args,
            old_context,
            targets,
            controls,
            cache_stats,
            old_hkl_qc,
            summaries,
            remove_sets,
            stream_qc,
            warnings,
            started_utc,
            logger,
        )
        logger.log("restricted EgM2 control builder complete")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
