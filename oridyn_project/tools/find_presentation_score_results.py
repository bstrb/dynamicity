#!/usr/bin/env python3
"""Compare presentation score candidates from existing merge results only.

This script is intentionally read-only with respect to experiment/result data.
It reads existing summary CSV/JSON files, optionally scans nearby already
completed pilot result directories, and writes a compact presentation comparison
inside the selected OUT directory.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_full_population_sweep_20260717"
)

REFERENCE = {
    "cc12": 0.9970285,
    "rsplit": 5.45,
    "snr": 17.00,
    "completeness": 99.889625,
    "redundancy": 534.39,
}

OUTPUT_CSV = "presentation_score_comparison.csv"
OUTPUT_MD = "presentation_score_comparison.md"
OUTPUT_METADATA = "presentation_score_comparison_metadata.json"

CURRENT_INPUT_FILES = [
    "merge_summary_nonrandom.csv",
    "merge_filter_ranked.csv",
    "merge_diag_low_high_pairs.csv",
    "merge_missing_or_unfinished.csv",
    "experiment_plan.csv",
    "selection_counts.csv",
    "validation.json",
    "stream_manifest.csv",
    "merge_manifest.tsv",
]

NEARBY_SUMMARY_FILES = [
    "merge_filter_ranked.csv",
    "merge_summary_nonrandom.csv",
    "selected_merge_runs.csv",
    "summary.csv",
    "per_variant_filter_summary.csv",
    "score_comparison_by_current_reference.csv",
]

FLOAT_RE = re.compile(r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?")
PARTIALATOR_RE = re.compile(r"partialator_results", re.IGNORECASE)
MAX_TEXT_BYTES = 2_000_000
PROGRESS_INTERVAL_SECONDS = 3.0

SCORE_LABELS = {
    "eg_c2mean": "Eg<C^2>_E",
    "eg_m2": "Eg*M2",
}


@dataclass
class SourceTracker:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, path: Path, role: str, exists: bool | None = None, truncated: bool = False) -> None:
        path = path.resolve()
        if exists is None:
            exists = path.exists()
        record: dict[str, Any] = {
            "path": str(path),
            "role": role,
            "exists": bool(exists),
            "truncated": bool(truncated),
        }
        if exists:
            try:
                stat = path.stat()
                record["size_bytes"] = stat.st_size
                record["mtime"] = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
            except OSError as exc:
                record["stat_error"] = str(exc)
        with self.lock:
            existing = self.records.get(str(path))
            if existing is None:
                self.records[str(path)] = record
            else:
                roles = set(str(existing.get("role", "")).split(";"))
                roles.add(role)
                existing["role"] = ";".join(sorted(role for role in roles if role))
                existing["exists"] = existing.get("exists") or record["exists"]
                existing["truncated"] = existing.get("truncated") or record["truncated"]
                for key in ("size_bytes", "mtime", "stat_error"):
                    if key in record:
                        existing[key] = record[key]

    def as_list(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(self.records.values(), key=lambda row: row["path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-nearby",
        action="store_true",
        help="Also scan sibling restricted-pilot/V5 result directories for existing candidate results.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    match = FLOAT_RE.search(text)
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return value


def read_text_bounded(path: Path, tracker: SourceTracker, role: str, max_bytes: int = MAX_TEXT_BYTES) -> tuple[str, bool]:
    exists = path.exists()
    if not exists:
        tracker.add(path, role, exists=False)
        return "", False
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        tracker.add(path, role, exists=True)
        return f"READ_ERROR: {exc}", False
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    tracker.add(path, role, exists=True, truncated=truncated)
    return data.decode("utf-8", errors="replace"), truncated


def iter_csv_rows(path: Path, tracker: SourceTracker, role: str, delimiter: str = ",") -> Iterable[dict[str, str]]:
    if not path.exists():
        tracker.add(path, role, exists=False)
        return
    tracker.add(path, role, exists=True)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            yield {clean_text(key): clean_text(value) for key, value in row.items() if key is not None}


def load_json_bounded(path: Path, tracker: SourceTracker, role: str) -> dict[str, Any]:
    text, truncated = read_text_bounded(path, tracker, role)
    if not text or truncated:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def result_variant_from_name(name: str) -> str:
    match = PARTIALATOR_RE.search(name)
    if not match:
        return name
    return name[: match.start()].rstrip("_-.")


def metadata_key(row: dict[str, Any]) -> str:
    for key in ("variant_id", "variant", "stream_stem", "stream_name"):
        value = clean_text(row.get(key))
        if value:
            return value
    output_filename = clean_text(row.get("output_filename") or row.get("stream_path"))
    if output_filename:
        return Path(output_filename).stem
    return ""


def load_metadata_tables(out_dir: Path, tracker: SourceTracker) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for filename in ("experiment_plan.csv", "selection_counts.csv", "stream_manifest.csv"):
        path = out_dir / filename
        for row in iter_csv_rows(path, tracker, f"current metadata {filename}"):
            key = metadata_key(row)
            if not key:
                continue
            target = tables.setdefault(key, {})
            for col, value in row.items():
                if value and col not in target:
                    target[col] = value
                elif value and f"{filename}__{col}" not in target:
                    target[f"{filename}__{col}"] = value
    manifest = out_dir / "merge_manifest.tsv"
    for row in iter_csv_rows(manifest, tracker, "current metadata merge_manifest.tsv", delimiter="\t"):
        key = metadata_key(row)
        if not key:
            continue
        target = tables.setdefault(key, {})
        for col, value in row.items():
            if value and col not in target:
                target[col] = value
            elif value and f"merge_manifest.tsv__{col}" not in target:
                target[f"merge_manifest.tsv__{col}"] = value
    return tables


def enrich_from_metadata(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key, value in metadata.items():
        if value and not clean_text(out.get(key)):
            out[key] = value
    return out


def score_id_from_row(row: dict[str, Any]) -> str:
    for key in ("score", "score_id", "score_or_random_control_id"):
        value = clean_text(row.get(key)).lower()
        if value:
            return value
    variant = clean_text(row.get("variant")).lower()
    if re.search(r"(^|_)eg_c2mean($|_)", variant):
        return "eg_c2mean"
    if re.search(r"(^|_)eg_m2($|_)", variant):
        return "eg_m2"
    return ""


def score_kind(row: dict[str, Any]) -> str:
    score_id = score_id_from_row(row)
    variant = clean_text(row.get("variant")).lower()
    if score_id == "eg_c2mean" or re.search(r"(^|_)eg_c2mean($|_)", variant):
        return "eg_c2mean"
    if score_id == "eg_m2" or re.search(r"(^|_)eg_m2($|_)", variant):
        return "eg_m2"
    return ""


def is_filter_row(row: dict[str, Any]) -> bool:
    variant = clean_text(row.get("variant")).lower()
    experiment_type = clean_text(row.get("experiment_type")).lower()
    classification = clean_text(row.get("classification")).lower()
    return (
        variant.startswith("filter_")
        or "_drop" in variant
        or classification.startswith("filter_")
        or "filter" in experiment_type
    )


def is_diag_row(row: dict[str, Any]) -> bool:
    variant = clean_text(row.get("variant")).lower()
    experiment_type = clean_text(row.get("experiment_type")).lower()
    return variant.startswith("diag_") or "diagnostic" in experiment_type


def classify_scope(base: Path, out_dir: Path) -> str:
    if base.resolve() == out_dir.resolve():
        return "current_v6_full_population"
    return "earlier_restricted_pilot_or_v5"


def metric(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = to_float(row.get(name))
        if value is not None:
            return value
    return None


def compute_deltas(row: dict[str, Any]) -> None:
    cc12 = metric(row, "cc12")
    rsplit = metric(row, "rsplit")
    snr = metric(row, "snr", "I_over_sigma")
    completeness = metric(row, "completeness")
    redundancy = metric(row, "redundancy", "multiplicity")
    if cc12 is not None:
        row["delta_cc12_vs_ref"] = cc12 - REFERENCE["cc12"]
    if rsplit is not None:
        row["delta_rsplit_vs_ref"] = rsplit - REFERENCE["rsplit"]
    if snr is not None:
        row["delta_snr_vs_ref"] = snr - REFERENCE["snr"]
    if completeness is not None:
        row["delta_completeness_vs_ref"] = completeness - REFERENCE["completeness"]
    if redundancy is not None:
        row["delta_redundancy_vs_ref"] = redundancy - REFERENCE["redundancy"]
    row["improves_both"] = bool(cc12 is not None and rsplit is not None and cc12 > REFERENCE["cc12"] and rsplit < REFERENCE["rsplit"])


def normalize_candidate_row(
    row: dict[str, Any],
    *,
    scope_category: str,
    source_group: str,
    source_file: Path,
    analysis_kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metadata = metadata or {}
    row = enrich_from_metadata(row, metadata)
    kind = score_kind(row)
    if not kind:
        return None
    variant = clean_text(row.get("variant") or row.get("variant_id"))
    if not variant:
        return None
    out: dict[str, Any] = {
        "scope_category": scope_category,
        "source_group": source_group,
        "analysis_kind": analysis_kind,
        "score_label": SCORE_LABELS[kind],
        "score_id": kind,
        "variant": variant,
        "target_family": clean_text(row.get("target_family") or row.get("filtering_target") or row.get("target")),
        "drop_fraction": metric(row, "drop_fraction", "metadata_drop_fraction", "fraction"),
        "designation": clean_text(row.get("designation")),
        "status": clean_text(row.get("status") or "completed"),
        "cc12": metric(row, "cc12"),
        "rsplit": metric(row, "rsplit"),
        "snr": metric(row, "snr", "I_over_sigma"),
        "completeness": metric(row, "completeness"),
        "redundancy": metric(row, "redundancy", "multiplicity"),
        "result_dir": clean_text(row.get("result_dir") or row.get("merge_dir")),
        "source_file": str(source_file),
    }
    for key in (
        "cc12_source_file",
        "rsplit_source_file",
        "snr_source_file",
        "completeness_source_file",
        "redundancy_source_file",
    ):
        if clean_text(row.get(key)):
            out[key] = row[key]
    if not out["target_family"]:
        variant_lower = variant.lower()
        if variant_lower.startswith("filter_all_"):
            out["target_family"] = "all"
        elif variant_lower.startswith("filter_higheg_"):
            out["target_family"] = "higheg"
        elif variant_lower.startswith("filter_matched_"):
            out["target_family"] = "matched"
    if out["drop_fraction"] is None:
        match = re.search(r"_drop(\d+)", variant.lower())
        if match:
            out["drop_fraction"] = int(match.group(1)) / 100.0
    compute_deltas(out)
    return out


def normalize_diag_row(row: dict[str, Any], source_file: Path, source_group: str, scope_category: str) -> dict[str, Any] | None:
    score = clean_text(row.get("score") or row.get("score_id")).lower()
    if score not in SCORE_LABELS:
        return None
    out: dict[str, Any] = {
        "scope_category": scope_category,
        "source_group": source_group,
        "analysis_kind": "diagnostic_low_high_pair",
        "score_label": SCORE_LABELS[score],
        "score_id": score,
        "variant": clean_text(row.get("score")),
        "source_file": str(source_file),
        "low_variant": clean_text(row.get("low_variant")),
        "high_variant": clean_text(row.get("high_variant")),
        "low_cc12": metric(row, "low_cc12"),
        "high_cc12": metric(row, "high_cc12"),
        "delta_cc12_low_high": metric(row, "delta_cc12"),
        "low_rsplit": metric(row, "low_rsplit"),
        "high_rsplit": metric(row, "high_rsplit"),
        "delta_rsplit_low_high": metric(row, "delta_rsplit"),
        "low_snr": metric(row, "low_snr"),
        "high_snr": metric(row, "high_snr"),
        "delta_snr_low_high": metric(row, "delta_snr"),
        "status": "completed",
    }
    return out


def parse_current_v6(out_dir: Path, tracker: SourceTracker) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata_tables = load_metadata_tables(out_dir, tracker)
    validation = load_json_bounded(out_dir / "validation.json", tracker, "current metadata validation.json")

    for filename in ("merge_summary_nonrandom.csv", "merge_filter_ranked.csv", "merge_diag_low_high_pairs.csv", "merge_missing_or_unfinished.csv"):
        # Register even absent files as part of the requested starting set.
        path = out_dir / filename
        if not path.exists():
            tracker.add(path, f"current input {filename}", exists=False)

    filter_source = out_dir / "merge_filter_ranked.csv"
    if filter_source.exists():
        for row in iter_csv_rows(filter_source, tracker, "current filter summary"):
            if not is_filter_row(row):
                continue
            variant = clean_text(row.get("variant"))
            normalized = normalize_candidate_row(
                row,
                scope_category="current_v6_full_population",
                source_group=out_dir.name,
                source_file=filter_source,
                analysis_kind="filter_result",
                metadata=metadata_tables.get(variant, {}),
            )
            if normalized:
                rows.append(normalized)
    else:
        summary_source = out_dir / "merge_summary_nonrandom.csv"
        for row in iter_csv_rows(summary_source, tracker, "current nonrandom summary"):
            if not is_filter_row(row):
                continue
            variant = clean_text(row.get("variant"))
            normalized = normalize_candidate_row(
                row,
                scope_category="current_v6_full_population",
                source_group=out_dir.name,
                source_file=summary_source,
                analysis_kind="filter_result",
                metadata=metadata_tables.get(variant, {}),
            )
            if normalized:
                rows.append(normalized)

    diag_source = out_dir / "merge_diag_low_high_pairs.csv"
    for row in iter_csv_rows(diag_source, tracker, "current diagnostic summary"):
        normalized = normalize_diag_row(row, diag_source, out_dir.name, "current_v6_full_population")
        if normalized:
            rows.append(normalized)

    missing_source = out_dir / "merge_missing_or_unfinished.csv"
    for row in iter_csv_rows(missing_source, tracker, "current missing/unfinished summary"):
        if not is_filter_row(row):
            continue
        variant = clean_text(row.get("variant"))
        normalized = normalize_candidate_row(
            row,
            scope_category="current_v6_full_population",
            source_group=out_dir.name,
            source_file=missing_source,
            analysis_kind="missing_or_unfinished",
            metadata=metadata_tables.get(variant, {}),
        )
        if normalized:
            rows.append(normalized)

    return rows, {
        "metadata_variant_count": len(metadata_tables),
        "validation_top_level_keys": sorted(validation.keys())[:50],
    }


def parse_metric_from_text(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = to_float(match.group(1))
            if value is not None:
                return value
    return None


def parse_result_dir(result_dir: Path, base: Path, tracker: SourceTracker) -> dict[str, Any]:
    variant = result_variant_from_name(result_dir.name)
    row: dict[str, Any] = {
        "variant": variant,
        "result_dir": str(result_dir),
        "status": "completed",
    }
    files = {
        "qc_stats/check_hkl_completeness.log": "nearby completeness qc",
        "qc_stats/compare_cc12.log": "nearby cc12 qc",
        "qc_stats/compare_rsplit.log": "nearby rsplit qc",
        "metadata_and_outputs.txt": "nearby result metadata",
        "parameters.json": "nearby result parameters",
    }
    source_files: list[str] = []
    for rel, role in files.items():
        path = result_dir / rel
        text, _ = read_text_bounded(path, tracker, role)
        if not text:
            continue
        source_files.append(rel)
        if rel.endswith("check_hkl_completeness.log"):
            row["snr"] = parse_metric_from_text(text, [r"Overall\s+<snr>\s*=\s*(" + FLOAT_RE.pattern + r")"])
            row["redundancy"] = parse_metric_from_text(text, [r"Overall\s+redundancy\s*=\s*(" + FLOAT_RE.pattern + r")"])
            row["completeness"] = parse_metric_from_text(text, [r"Overall\s+completeness\s*=\s*(" + FLOAT_RE.pattern + r")"])
            row["measurements"] = parse_metric_from_text(text, [r"(\d+)\s+measurements\s+in\s+total"])
            row["reflections"] = parse_metric_from_text(text, [r"(\d+)\s+reflections\s+in\s+total"])
            row["snr_source_file"] = rel
            row["redundancy_source_file"] = rel
            row["completeness_source_file"] = rel
        elif rel.endswith("compare_cc12.log"):
            row["cc12"] = parse_metric_from_text(text, [r"Overall\s+CC\s*=\s*(" + FLOAT_RE.pattern + r")"])
            row["cc12_source_file"] = rel
        elif rel.endswith("compare_rsplit.log"):
            row["rsplit"] = parse_metric_from_text(text, [r"Overall\s+Rsplit\s*=\s*(" + FLOAT_RE.pattern + r")"])
            row["rsplit_source_file"] = rel
    row["scanned_stat_files"] = ";".join(source_files)
    if row.get("cc12") is None or row.get("rsplit") is None:
        row["status"] = "unfinished_or_missing_final_stats"
    return row


def candidate_result_dirs(base: Path) -> list[Path]:
    dirs: list[Path] = []
    for path in base.glob("*partialator_results*"):
        if not path.is_dir():
            continue
        variant = result_variant_from_name(path.name).lower()
        if re.search(r"(^|_)eg_c2mean($|_)", variant) or re.search(r"(^|_)eg_m2($|_)", variant):
            dirs.append(path)
    return sorted(dirs)


def nearby_dirs(out_dir: Path) -> list[Path]:
    parent = out_dir.parent
    if not parent.exists():
        return []
    dirs: list[Path] = []
    for path in parent.iterdir():
        if not path.is_dir() or path.resolve() == out_dir.resolve():
            continue
        name = path.name
        if name.startswith("oridyn_v6_score_target_filter_map_") or name.startswith("oridyn_v5_"):
            dirs.append(path)
    return sorted(dirs)


def load_nearby_metadata(base: Path, tracker: SourceTracker) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for filename in ("experiment_plan.csv", "per_variant_selection_summary.csv", "score_variant_parameters.csv"):
        path = base / filename
        for row in iter_csv_rows(path, tracker, f"nearby metadata {filename}"):
            key = metadata_key(row)
            if not key:
                continue
            target = tables.setdefault(key, {})
            for col, value in row.items():
                if value and col not in target:
                    target[col] = value
    for filename in ("validation.json", "run_metadata.json"):
        load_json_bounded(base / filename, tracker, f"nearby metadata {filename}")
    return tables


def parse_nearby_base(base: Path, out_dir: Path, tracker: SourceTracker, workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = load_nearby_metadata(base, tracker)
    scope = classify_scope(base, out_dir)

    used_summary = False
    for filename in NEARBY_SUMMARY_FILES:
        path = base / filename
        if not path.exists():
            tracker.add(path, f"nearby summary {filename}", exists=False)
            continue
        delimiter = "\t" if path.suffix == ".tsv" else ","
        found = 0
        for source_row in iter_csv_rows(path, tracker, f"nearby summary {filename}", delimiter=delimiter):
            if not is_filter_row(source_row):
                continue
            variant = clean_text(source_row.get("variant") or source_row.get("variant_id") or source_row.get("stream_stem"))
            source_row["variant"] = variant
            normalized = normalize_candidate_row(
                source_row,
                scope_category=scope,
                source_group=base.name,
                source_file=path,
                analysis_kind="filter_result",
                metadata=metadata.get(variant, {}),
            )
            if normalized:
                rows.append(normalized)
                found += 1
        if found:
            used_summary = True

    # Some restricted pilot folders have result dirs but no compact merge summary.
    existing_variants = {row["variant"] for row in rows if row.get("source_group") == base.name}
    dirs = [path for path in candidate_result_dirs(base) if result_variant_from_name(path.name) not in existing_variants]
    if not dirs:
        return rows

    log(f"{base.name}: parsing {len(dirs)} existing candidate result dirs with {workers} workers")
    started = time.monotonic()
    last_progress = started
    parsed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(parse_result_dir, path, base, tracker): path for path in dirs}
        for future in as_completed(future_map):
            parsed += 1
            parsed_row = future.result()
            variant = clean_text(parsed_row.get("variant"))
            normalized = normalize_candidate_row(
                parsed_row,
                scope_category=scope,
                source_group=base.name,
                source_file=Path(parsed_row.get("result_dir", "")),
                analysis_kind="filter_result",
                metadata=metadata.get(variant, {}),
            )
            if normalized:
                rows.append(normalized)
            now = time.monotonic()
            if now - last_progress >= PROGRESS_INTERVAL_SECONDS and len(dirs) > 10:
                elapsed = max(now - started, 1.0e-9)
                log(f"{base.name}: parsed {parsed}/{len(dirs)} dirs ({parsed / elapsed:.1f}/s)")
                last_progress = now
    if used_summary:
        log(f"{base.name}: supplemented summary rows with direct parsing where needed")
    return rows


def parse_nearby(out_dir: Path, tracker: SourceTracker, workers: int) -> tuple[list[dict[str, Any]], list[str]]:
    bases = nearby_dirs(out_dir)
    rows: list[dict[str, Any]] = []
    scanned: list[str] = []
    if not bases:
        return rows, scanned
    log(f"Scanning {len(bases)} nearby sibling output directories")
    for base in bases:
        scanned.append(str(base))
        rows.extend(parse_nearby_base(base, out_dir, tracker, workers))
    return rows, scanned


def sort_filter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        cc12 = metric(row, "cc12")
        rsplit = metric(row, "rsplit")
        return (
            row.get("scope_category") != "current_v6_full_population",
            row.get("score_id") != "eg_c2mean",
            not bool(row.get("improves_both")),
            -(cc12 if cc12 is not None else -999.0),
            rsplit if rsplit is not None else 999.0,
            clean_text(row.get("variant")),
        )

    return sorted(rows, key=key)


def best_balanced_current(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    current = [
        row
        for row in rows
        if row.get("scope_category") == "current_v6_full_population"
        and row.get("analysis_kind") == "filter_result"
        and metric(row, "cc12") is not None
        and metric(row, "rsplit") is not None
    ]
    if not current:
        return None
    # Prefer CC-improved rows, then the lowest Rsplit, then the highest CC1/2.
    return sorted(
        current,
        key=lambda row: (
            metric(row, "cc12") <= REFERENCE["cc12"],
            metric(row, "rsplit") if metric(row, "rsplit") is not None else 999.0,
            -(metric(row, "cc12") or -999.0),
        ),
    )[0]


def recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filters = [row for row in rows if row.get("analysis_kind") == "filter_result"]
    c2_both = [row for row in filters if row.get("score_id") == "eg_c2mean" and row.get("improves_both")]
    m2_both = [row for row in filters if row.get("score_id") == "eg_m2" and row.get("improves_both")]
    current_best = best_balanced_current(rows)

    if c2_both:
        best = sort_filter_rows(c2_both)[0]
        return {
            "recommendation": "Use Eg<C^2>_E",
            "reason": "An existing filtering result improves both CC1/2 and Rsplit versus the full-data reference.",
            "supporting_variant": best.get("variant"),
            "supporting_scope": best.get("scope_category"),
        }
    if current_best and current_best.get("score_id") == "eg_c2mean":
        return {
            "recommendation": "Use Eg<C^2>_E as the current full-population result, with caveat",
            "reason": "No Eg<C^2>_E both-improved result was found, but it is the best balanced current full-population candidate by CC-improved/lowest-Rsplit ordering.",
            "supporting_variant": current_best.get("variant"),
            "supporting_scope": current_best.get("scope_category"),
        }
    if m2_both:
        best = sort_filter_rows(m2_both)[0]
        return {
            "recommendation": "Stick with Eg*M2",
            "reason": "No Eg<C^2>_E both-improved result was found, while Eg*M2 has an existing clean both-improved restricted-pilot result.",
            "supporting_variant": best.get("variant"),
            "supporting_scope": best.get("scope_category"),
        }
    return {
        "recommendation": "No decisive score recommendation",
        "reason": "No existing Eg<C^2>_E or Eg*M2 filtering result improves both CC1/2 and Rsplit in the scanned inputs.",
        "supporting_variant": "",
        "supporting_scope": "",
    }


def collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "scope_category",
        "source_group",
        "analysis_kind",
        "score_label",
        "score_id",
        "variant",
        "target_family",
        "drop_fraction",
        "designation",
        "status",
        "cc12",
        "rsplit",
        "snr",
        "completeness",
        "redundancy",
        "delta_cc12_vs_ref",
        "delta_rsplit_vs_ref",
        "delta_snr_vs_ref",
        "improves_both",
        "low_variant",
        "high_variant",
        "low_cc12",
        "high_cc12",
        "delta_cc12_low_high",
        "low_rsplit",
        "high_rsplit",
        "delta_rsplit_low_high",
        "low_snr",
        "high_snr",
        "delta_snr_low_high",
        "result_dir",
        "source_file",
        "cc12_source_file",
        "rsplit_source_file",
        "snr_source_file",
        "completeness_source_file",
        "redundancy_source_file",
    ]
    columns: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if any(key in row for row in rows):
            columns.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = collect_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in columns})


def fmt_float(value: Any, digits: int = 7) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if abs(number) >= 10:
        return f"{number:.3f}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    if not selected:
        return "_None found._\n"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in selected:
        values: list[str] = []
        for label, key in columns:
            value = row.get(key, "")
            if isinstance(value, bool):
                values.append("yes" if value else "no")
            elif isinstance(value, float):
                values.append(fmt_float(value))
            else:
                values.append(clean_text(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, rows: list[dict[str, Any]], rec: dict[str, Any], args: argparse.Namespace) -> None:
    current_filters = sort_filter_rows(
        [
            row
            for row in rows
            if row.get("scope_category") == "current_v6_full_population"
            and row.get("analysis_kind") == "filter_result"
        ]
    )
    nearby_filters = sort_filter_rows(
        [
            row
            for row in rows
            if row.get("scope_category") == "earlier_restricted_pilot_or_v5"
            and row.get("analysis_kind") == "filter_result"
        ]
    )
    diagnostics = [
        row
        for row in rows
        if row.get("scope_category") == "current_v6_full_population"
        and row.get("analysis_kind") == "diagnostic_low_high_pair"
    ]
    missing = [
        row
        for row in rows
        if row.get("analysis_kind") == "missing_or_unfinished"
    ]
    table_cols = [
        ("score", "score_label"),
        ("variant", "variant"),
        ("target", "target_family"),
        ("drop", "drop_fraction"),
        ("CC1/2", "cc12"),
        ("Rsplit", "rsplit"),
        ("SNR", "snr"),
        ("dCC", "delta_cc12_vs_ref"),
        ("dRsplit", "delta_rsplit_vs_ref"),
        ("both", "improves_both"),
    ]
    diag_cols = [
        ("score", "score_label"),
        ("low CC1/2", "low_cc12"),
        ("high CC1/2", "high_cc12"),
        ("dCC low-high", "delta_cc12_low_high"),
        ("low Rsplit", "low_rsplit"),
        ("high Rsplit", "high_rsplit"),
        ("dRsplit low-high", "delta_rsplit_low_high"),
        ("low SNR", "low_snr"),
        ("high SNR", "high_snr"),
        ("dSNR", "delta_snr_low_high"),
    ]
    missing_cols = [
        ("score", "score_label"),
        ("variant", "variant"),
        ("target", "target_family"),
        ("drop", "drop_fraction"),
        ("status", "status"),
    ]
    text = f"""# Presentation Score Comparison

Generated: {now_iso()}

OUT: `{args.out_dir}`

Reference: CC1/2 `{REFERENCE['cc12']}`, Rsplit `{REFERENCE['rsplit']}`, SNR `{REFERENCE['snr']}`, completeness `{REFERENCE['completeness']}`, redundancy `{REFERENCE['redundancy']}`.

## Reciprocal-Space Coupling Definition

`C(g-q) = exp[-0.5*(dq/sigma_c)^2]` for `dq <= r_cut`, otherwise `0`.

`dq = sqrt((q-g)^T G* (q-g))`.

`sigma_c = 0.050 A^-1`; `r_cut = 0.150 A^-1`; `q != g`.

## Recommendation

**{rec['recommendation']}**: {rec['reason']}

Supporting variant: `{rec.get('supporting_variant') or 'none'}`; scope: `{rec.get('supporting_scope') or 'none'}`.

## A) Current Full-Population V6 Filtering Results

{markdown_table(current_filters, table_cols, limit=30)}

## Current V6 Diagnostic Low/High Pairs

{markdown_table(diagnostics, diag_cols, limit=10)}

## Current Missing/Unfinished Candidate Rows

{markdown_table(missing, missing_cols, limit=30)}

## B) Earlier Restricted-Pilot/V5 Results

{markdown_table(nearby_filters, table_cols, limit=40)}
"""
    path.write_text(text, encoding="utf-8")


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    rec: dict[str, Any],
    current_metadata: dict[str, Any],
    nearby_scanned: list[str],
    tracker: SourceTracker,
) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('scope_category')}::{row.get('analysis_kind')}::{row.get('score_id')}"
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "generated_at": now_iso(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "parameters": {
            "out_dir": str(args.out_dir),
            "include_nearby": bool(args.include_nearby),
            "workers": int(args.workers),
        },
        "reference": REFERENCE,
        "score_definitions": {
            "Eg<C^2>_E": "S(g) = Eg * sum_q(Eq*C(g-q)^2) / sum_q(Eq)",
            "Eg*M2": "Eg*M2; variants identified by score_id/variant eg_m2",
            "C": "C(g-q) = exp[-0.5*(dq/sigma_c)^2] for dq <= r_cut, otherwise 0; dq = sqrt((q-g)^T G* (q-g)); sigma_c = 0.050 A^-1; r_cut = 0.150 A^-1; q != g.",
        },
        "recommendation": rec,
        "row_count": len(rows),
        "counts": counts,
        "current_metadata": current_metadata,
        "nearby_directories_scanned": nearby_scanned,
        "source_files_used": tracker.as_list(),
        "outputs": {
            "csv": str(args.out_dir / OUTPUT_CSV),
            "markdown": str(args.out_dir / OUTPUT_MD),
            "metadata_json": str(args.out_dir / OUTPUT_METADATA),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if not out_dir.exists():
        raise SystemExit(f"OUT directory does not exist: {out_dir}")
    tracker = SourceTracker()
    for filename in CURRENT_INPUT_FILES:
        path = out_dir / filename
        if not path.exists():
            tracker.add(path, f"requested current input {filename}", exists=False)

    log("Reading current V6 summary and metadata files")
    rows, current_metadata = parse_current_v6(out_dir, tracker)

    nearby_scanned: list[str] = []
    if args.include_nearby:
        nearby_rows, nearby_scanned = parse_nearby(out_dir, tracker, args.workers)
        rows.extend(nearby_rows)

    rows = sort_filter_rows([row for row in rows if row.get("analysis_kind") == "filter_result"]) + [
        row for row in rows if row.get("analysis_kind") != "filter_result"
    ]
    rec = recommendation(rows)

    csv_path = out_dir / OUTPUT_CSV
    md_path = out_dir / OUTPUT_MD
    metadata_path = out_dir / OUTPUT_METADATA
    log(f"Writing {csv_path}")
    write_csv(csv_path, rows)
    log(f"Writing {md_path}")
    write_markdown(md_path, rows, rec, args)
    log(f"Writing {metadata_path}")
    write_metadata(
        metadata_path,
        args=args,
        rows=rows,
        rec=rec,
        current_metadata=current_metadata,
        nearby_scanned=nearby_scanned,
        tracker=tracker,
    )
    log(f"Done. Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
