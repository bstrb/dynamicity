#!/usr/bin/env python3
"""Summarize V6 full-population Partialator merge results.

The script is intentionally tolerant of merges still running.  It discovers
Partialator result directories, extracts final QC metrics from the files that
are actually present, joins sweep metadata, and writes compact CSV/Markdown
outputs into the sweep directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_full_population_sweep_20260717"
)

CORE_METRICS = ("completeness", "redundancy", "snr", "cc12", "rsplit")
METRIC_COLUMNS = (
    "completeness",
    "redundancy",
    "multiplicity",
    "snr",
    "I_over_sigma",
    "cc12",
    "rsplit",
    "nref",
    "reflections",
    "possible_reflections",
    "measurements",
    "point_group",
    "symmetry",
)
REFERENCE_DEFAULTS = {
    "completeness": 99.889625,
    "redundancy": 534.39,
    "snr": 17.00,
    "cc12": 0.9970285,
    "rsplit": 5.45,
}

SCAN_SUFFIXES = {".log", ".txt", ".json", ".csv", ".md", ".tsv", ".cell"}
SKIP_SCAN_DIRS = {"pr-logs", "__pycache__", ".git"}
MAX_SCAN_BYTES = 50_000_000

FLOAT_TOKEN = r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?"
PARTIALATOR_RE = re.compile(r"partialator_results", re.IGNORECASE)


@dataclass
class MetricHit:
    value: Any
    source_file: str
    priority: int
    raw_text: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-randoms", action="store_true")
    parser.add_argument("--reference-cc12", type=float, default=REFERENCE_DEFAULTS["cc12"])
    parser.add_argument("--reference-rsplit", type=float, default=REFERENCE_DEFAULTS["rsplit"])
    parser.add_argument("--reference-snr", type=float, default=REFERENCE_DEFAULTS["snr"])
    parser.add_argument("--reference-completeness", type=float, default=REFERENCE_DEFAULTS["completeness"])
    parser.add_argument("--reference-redundancy", type=float, default=REFERENCE_DEFAULTS["redundancy"])
    return parser.parse_args()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("x", "").replace("X", "").replace("%", "")
    text = text.replace("\u00d7", "")
    match = re.search(FLOAT_TOKEN, text)
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1.0e-6:
        return None
    return rounded


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = collect_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in columns})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=json_default)
    return value


def collect_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "variant",
        "result_dir",
        "status",
        "classification",
        "target_family",
        "score",
        "drop_fraction",
        "designation",
        *METRIC_COLUMNS,
        "delta_completeness_vs_ref",
        "delta_redundancy_vs_ref",
        "delta_snr_vs_ref",
        "delta_cc12_vs_ref",
        "delta_rsplit_vs_ref",
        *[f"{metric}_source_file" for metric in METRIC_COLUMNS],
        "metadata_sources",
    ]
    seen: set[str] = set()
    columns: list[str] = []
    for key in preferred:
        if any(key in row for row in rows) and key not in seen:
            columns.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen and not key.endswith("_source_priority"):
                columns.append(key)
                seen.add(key)
    return columns


def sort_key_text(value: Any) -> str:
    return clean_text(value).lower()


def safe_float(value: Any, missing: float = float("nan")) -> float:
    number = to_float(value)
    return number if number is not None else missing


def safe_float_sort(value: Any, missing: float) -> float:
    number = to_float(value)
    return number if number is not None else missing


def result_variant_from_name(name: str) -> str:
    match = PARTIALATOR_RE.search(name)
    if not match:
        return name
    return name[: match.start()].rstrip("_-.")


def classify_variant(variant: str, metadata: dict[str, Any] | None = None) -> str:
    if variant.startswith("random_"):
        return "random skipped"
    if variant.startswith("diag_"):
        return "diag"
    if variant.startswith("filter_all_"):
        return "filter_all"
    if variant.startswith("filter_higheg_"):
        return "filter_higheg"
    if variant.startswith("filter_matched_"):
        return "filter_matched"

    metadata = metadata or {}
    target = clean_text(metadata.get("filtering_target") or metadata.get("target")).lower()
    experiment_type = clean_text(metadata.get("experiment_type")).lower()
    if "random" in experiment_type:
        return "random skipped"
    if "diagnostic" in experiment_type:
        return "diag"
    if target == "all":
        return "filter_all"
    if target == "higheg":
        return "filter_higheg"
    if target == "matched":
        return "filter_matched"
    return "other"


def target_family_from_classification(classification: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    target = clean_text(metadata.get("filtering_target") or metadata.get("target")).lower()
    if target in {"all", "higheg", "matched"}:
        return target
    if classification == "filter_all":
        return "all"
    if classification == "filter_higheg":
        return "higheg"
    if classification == "filter_matched":
        return "matched"
    return ""


def parse_score(variant: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in ("score_id", "score_or_random_control_id"):
        value = clean_text(metadata.get(key))
        if value:
            return value
    if variant.startswith("diag_"):
        match = re.match(r"diag_(.+)_(?:low50|high50)$", variant)
        return match.group(1) if match else variant.removeprefix("diag_")
    match = re.match(r"filter_(?:all|higheg|matched)_(.+)_drop\d+$", variant)
    return match.group(1) if match else ""


def parse_drop_fraction(variant: str, metadata: dict[str, Any] | None = None) -> float | None:
    metadata = metadata or {}
    value = to_float(metadata.get("drop_fraction") or metadata.get("fraction"))
    if value is not None:
        return value
    match = re.search(r"_drop(\d+)", variant)
    if not match:
        return None
    digits = match.group(1)
    return int(digits) / 100.0


def parse_designation(variant: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    designation = clean_text(metadata.get("designation"))
    if designation:
        return designation
    match = re.search(r"_(low50|high50)$", variant)
    return match.group(1) if match else ""


def source_priority(result_dir: Path, path: Path) -> int:
    rel = path.relative_to(result_dir).as_posix()
    if rel == "qc_stats/check_hkl_completeness.log":
        return 0
    if rel == "qc_stats/compare_cc12.log":
        return 0
    if rel == "qc_stats/compare_rsplit.log":
        return 0
    if rel.startswith("qc_stats/"):
        return 5
    if rel == "metadata_and_outputs.txt":
        return 10
    if rel == "parameters.json":
        return 20
    if rel.endswith(".json"):
        return 25
    if "partialator_stderr" in rel:
        return 60
    if "partialator_stdout" in rel:
        return 30
    return 40


def candidate_stat_files(result_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(result_dir):
        dirnames[:] = [name for name in dirnames if name not in SKIP_SCAN_DIRS]
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_SCAN_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return sorted(files, key=lambda path: (source_priority(result_dir, path), path.relative_to(result_dir).as_posix()))


def set_hit(hits: dict[str, MetricHit], metric: str, value: Any, source_file: str, priority: int, raw_text: str = "") -> None:
    if metric in {"point_group", "symmetry"}:
        text = clean_text(value)
        if not text:
            return
        parsed_value: Any = text
    elif metric in {"nref", "reflections", "possible_reflections", "measurements"}:
        parsed_int = to_int(value)
        if parsed_int is None:
            return
        parsed_value = parsed_int
    else:
        number = to_float(value)
        if number is None:
            return
        parsed_value = number
    current = hits.get(metric)
    if current is not None and current.priority <= priority:
        return
    hits[metric] = MetricHit(parsed_value, source_file, priority, raw_text)


def set_cc12_hit(
    hits: dict[str, MetricHit],
    value: Any,
    source_file: str,
    priority: int,
    raw_text: str = "",
    percent_scale: bool = False,
) -> None:
    number = to_float(value)
    if number is None:
        return
    if percent_scale or abs(number) > 1.5:
        number = number / 100.0
    set_hit(hits, "cc12", number, source_file, priority, raw_text)


def parse_text_metrics(text: str, source_file: str, priority: int, hits: dict[str, MetricHit]) -> None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if match := re.search(r"(\d+)\s+measurements\s+in\s+total", stripped, re.IGNORECASE):
            set_hit(hits, "measurements", match.group(1), source_file, priority, stripped)
        if match := re.search(r"(\d+)\s+reflections\s+in\s+total", stripped, re.IGNORECASE):
            set_hit(hits, "reflections", match.group(1), source_file, priority, stripped)
        if match := re.search(r"(\d+)\s+reflections\s+possible", stripped, re.IGNORECASE):
            set_hit(hits, "possible_reflections", match.group(1), source_file, priority, stripped)
        if match := re.search(r"(\d+)\s+reflection\s+pairs\s+accepted", stripped, re.IGNORECASE):
            set_hit(hits, "nref", match.group(1), source_file, priority, stripped)
        if match := re.search(r"nref\s*[:=]\s*(\d+)", stripped, re.IGNORECASE):
            set_hit(hits, "nref", match.group(1), source_file, priority, stripped)

        if match := re.search(r"Overall\s+<snr>\s*=\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "snr", match.group(1), source_file, priority, stripped)
            set_hit(hits, "I_over_sigma", match.group(1), source_file, priority, stripped)
        if match := re.search(r"\b(?:SNR|I[_ /-]*over[_ /-]*sigma|I\s*/\s*sigma(?:\(I\))?)\b\s*[:=]\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "snr", match.group(1), source_file, priority, stripped)
            set_hit(hits, "I_over_sigma", match.group(1), source_file, priority, stripped)

        if match := re.search(r"Overall\s+redundancy\s*=\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "redundancy", match.group(1), source_file, priority, stripped)
            set_hit(hits, "multiplicity", match.group(1), source_file, priority, stripped)
        if match := re.search(r"\b(?:redundancy|multiplicity)\b\s*[:=]\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "redundancy", match.group(1), source_file, priority, stripped)
            set_hit(hits, "multiplicity", match.group(1), source_file, priority, stripped)

        if match := re.search(r"Overall\s+completeness\s*=\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "completeness", match.group(1), source_file, priority, stripped)
        if match := re.search(r"\bcompleteness\b\s*[:=]\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "completeness", match.group(1), source_file, priority, stripped)

        if match := re.search(r"Overall\s+CC\s*=\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_cc12_hit(hits, match.group(1), source_file, priority, stripped)
        if match := re.search(r"\b(?:CC1/2|CC\s*1/2|CChalf|CC\s*half|CC1\/2)\b\s*[:=]\s*(" + FLOAT_TOKEN + r")\s*(%)?", stripped, re.IGNORECASE):
            set_cc12_hit(hits, match.group(1), source_file, priority, stripped, percent_scale=bool(match.group(2)))

        if match := re.search(r"Overall\s+Rsplit\s*=\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "rsplit", match.group(1), source_file, priority, stripped)
        if match := re.search(r"\bRsplit\b\s*[:=]\s*(" + FLOAT_TOKEN + r")", stripped, re.IGNORECASE):
            set_hit(hits, "rsplit", match.group(1), source_file, priority, stripped)

        if match := re.search(r"\b(?:SYM|symmetry|point\s+group|point_group)\b\s*[:=]\s*([A-Za-z0-9_/\-+ ]+)", stripped, re.IGNORECASE):
            symmetry = match.group(1).strip()
            if symmetry and len(symmetry) <= 40:
                set_hit(hits, "symmetry", symmetry, source_file, priority, stripped)
                set_hit(hits, "point_group", symmetry, source_file, priority, stripped)


def flatten_json(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(item, next_prefix)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from flatten_json(item, next_prefix)
    else:
        yield prefix, value


def parse_json_metrics(text: str, source_file: str, priority: int, hits: dict[str, MetricHit]) -> None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    for key, value in flatten_json(data):
        key_norm = key.lower().replace("-", "_")
        if "completeness" in key_norm:
            set_hit(hits, "completeness", value, source_file, priority, key)
        elif "redundancy" in key_norm or "multiplicity" in key_norm:
            set_hit(hits, "redundancy", value, source_file, priority, key)
            set_hit(hits, "multiplicity", value, source_file, priority, key)
        elif "i_over_sigma" in key_norm or "i/sigma" in key_norm or key_norm.endswith(".snr") or key_norm == "snr":
            set_hit(hits, "snr", value, source_file, priority, key)
            set_hit(hits, "I_over_sigma", value, source_file, priority, key)
        elif "cc12" in key_norm or "cc1/2" in key_norm or "cchalf" in key_norm:
            set_cc12_hit(hits, value, source_file, priority, key)
        elif "rsplit" in key_norm:
            set_hit(hits, "rsplit", value, source_file, priority, key)
        elif key_norm.endswith("nref") or key_norm.endswith("n_ref"):
            set_hit(hits, "nref", value, source_file, priority, key)
        elif key_norm.endswith("reflections") or key_norm.endswith("reflection_count"):
            set_hit(hits, "reflections", value, source_file, priority, key)
        elif key_norm.endswith("measurements") or key_norm.endswith("measurement_count"):
            set_hit(hits, "measurements", value, source_file, priority, key)
        elif key_norm.endswith("sym") or "symmetry" in key_norm or "point_group" in key_norm:
            text_value = clean_text(value)
            if text_value and len(text_value) <= 40:
                set_hit(hits, "symmetry", text_value, source_file, priority, key)
                set_hit(hits, "point_group", text_value, source_file, priority, key)


def parse_result_dir(result_dir: Path) -> dict[str, Any]:
    variant = result_variant_from_name(result_dir.name)
    hits: dict[str, MetricHit] = {}
    scanned_files: list[str] = []
    read_errors: list[str] = []

    for path in candidate_stat_files(result_dir):
        rel = path.relative_to(result_dir).as_posix()
        priority = source_priority(result_dir, path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            read_errors.append(f"{rel}: {exc}")
            continue
        scanned_files.append(rel)
        if path.suffix.lower() == ".json":
            parse_json_metrics(text, rel, priority, hits)
        parse_text_metrics(text, rel, priority, hits)

    row: dict[str, Any] = {
        "variant": variant,
        "result_dir": str(result_dir),
        "result_dir_name": result_dir.name,
        "scanned_stat_file_count": len(scanned_files),
        "scanned_stat_files": ";".join(scanned_files),
    }
    if read_errors:
        row["read_errors"] = "; ".join(read_errors)

    for metric, hit in sorted(hits.items()):
        row[metric] = hit.value
        row[f"{metric}_source_file"] = hit.source_file
        row[f"{metric}_source_priority"] = hit.priority

    if "snr" in row and "I_over_sigma" not in row:
        row["I_over_sigma"] = row["snr"]
        row["I_over_sigma_source_file"] = row.get("snr_source_file", "")
    if "redundancy" in row and "multiplicity" not in row:
        row["multiplicity"] = row["redundancy"]
        row["multiplicity_source_file"] = row.get("redundancy_source_file", "")
    if "point_group" in row and "symmetry" not in row:
        row["symmetry"] = row["point_group"]
        row["symmetry_source_file"] = row.get("point_group_source_file", "")
    if "symmetry" in row and "point_group" not in row:
        row["point_group"] = row["symmetry"]
        row["point_group_source_file"] = row.get("symmetry_source_file", "")

    final_core_metrics = {
        metric
        for metric in CORE_METRICS
        if metric in row and int(row.get(f"{metric}_source_priority", 999)) < 60
    }
    missing_core = [metric for metric in CORE_METRICS if metric not in final_core_metrics]
    row["missing_core_metrics"] = ";".join(missing_core)
    if not scanned_files:
        row["status"] = "unfinished_no_stat_files"
    elif not missing_core:
        row["status"] = "completed"
    elif not final_core_metrics:
        row["status"] = "unfinished_no_final_stats"
    else:
        row["status"] = "partial_stats"
    return row


def read_table(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def load_metadata(out_dir: Path) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, Any]]:
    metadata_sources: dict[str, dict[str, dict[str, str]]] = {}
    for filename in ("experiment_plan.csv", "selection_counts.csv", "stream_manifest.csv", "merge_manifest.tsv"):
        path = out_dir / filename
        rows = read_table(path)
        by_variant: dict[str, dict[str, str]] = {}
        for row in rows:
            variant = clean_text(row.get("variant_id"))
            if variant:
                by_variant[variant] = {key: clean_text(value) for key, value in row.items()}
        if by_variant:
            metadata_sources[filename] = by_variant

    validation: dict[str, Any] = {}
    validation_path = out_dir / "validation.json"
    if validation_path.exists():
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            validation = {"read_error": str(exc)}
    return metadata_sources, validation


def metadata_for_variant(metadata_sources: dict[str, dict[str, dict[str, str]]], variant: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    sources: list[str] = []
    reserved = {
        "variant",
        "result_dir",
        "status",
        "classification",
        "score",
        "target_family",
        "drop_fraction",
        *METRIC_COLUMNS,
    }
    for source_name, by_variant in metadata_sources.items():
        row = by_variant.get(variant)
        if not row:
            continue
        sources.append(source_name)
        for key, value in row.items():
            if key == "variant_id" or value == "":
                continue
            out_key = f"metadata_{key}" if key in reserved else key
            if out_key not in merged or merged[out_key] == "":
                merged[out_key] = value
            elif str(merged[out_key]) != value:
                merged[f"{source_name}__{key}"] = value
    merged["metadata_sources"] = ";".join(sources)
    return merged


def planned_variants(metadata_sources: dict[str, dict[str, dict[str, str]]]) -> set[str]:
    variants: set[str] = set()
    for by_variant in metadata_sources.values():
        variants.update(by_variant)
    variants.discard("full_reference")
    return variants


def add_metadata_and_derived(row: dict[str, Any], metadata_sources: dict[str, dict[str, dict[str, str]]], reference: dict[str, float]) -> dict[str, Any]:
    variant = clean_text(row.get("variant"))
    metadata = metadata_for_variant(metadata_sources, variant)
    classification = classify_variant(variant, metadata)
    enriched = {**row, **metadata}
    enriched["classification"] = classification
    enriched["target_family"] = target_family_from_classification(classification, metadata)
    enriched["score"] = parse_score(variant, metadata)
    drop_fraction = parse_drop_fraction(variant, metadata)
    if drop_fraction is not None:
        enriched["drop_fraction"] = drop_fraction
    designation = parse_designation(variant, metadata)
    if designation:
        enriched["designation"] = designation

    metric_to_ref = {
        "completeness": "completeness",
        "redundancy": "redundancy",
        "snr": "snr",
        "cc12": "cc12",
        "rsplit": "rsplit",
    }
    for metric, ref_key in metric_to_ref.items():
        value = to_float(enriched.get(metric))
        if value is not None:
            enriched[f"delta_{metric}_vs_ref"] = value - reference[ref_key]
    return enriched


def discover_result_dirs(out_dir: Path) -> list[Path]:
    return sorted(
        [path for path in out_dir.iterdir() if path.is_dir() and "partialator_results" in path.name],
        key=lambda path: path.name,
    )


def parse_all_results(out_dir: Path, metadata_sources: dict[str, dict[str, dict[str, str]]], reference: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for result_dir in discover_result_dirs(out_dir):
        parsed = parse_result_dir(result_dir)
        rows.append(add_metadata_and_derived(parsed, metadata_sources, reference))
    return rows


def is_random_row(row: dict[str, Any]) -> bool:
    return clean_text(row.get("variant")).startswith("random_") or row.get("classification") == "random skipped"


def complete_rows(rows: list[dict[str, Any]], include_randoms: bool) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("status") == "completed"]
    if not include_randoms:
        selected = [row for row in selected if not is_random_row(row)]
    return selected


def missing_or_unfinished_rows(
    all_result_rows: list[dict[str, Any]],
    metadata_sources: dict[str, dict[str, dict[str, str]]],
    reference: dict[str, float],
    include_randoms: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    result_by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in all_result_rows:
        result_by_variant.setdefault(clean_text(row.get("variant")), []).append(row)
        if row.get("status") != "completed":
            rows.append(dict(row))

    for variant in sorted(planned_variants(metadata_sources)):
        if variant in result_by_variant:
            continue
        base = {
            "variant": variant,
            "result_dir": "",
            "result_dir_name": "",
            "status": "missing_result_dir",
            "missing_core_metrics": ";".join(CORE_METRICS),
        }
        rows.append(add_metadata_and_derived(base, metadata_sources, reference))

    if not include_randoms:
        rows = [row for row in rows if not is_random_row(row)]
    return sorted(rows, key=lambda row: (sort_key_text(row.get("classification")), sort_key_text(row.get("variant"))))


def diagnostic_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_score: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("classification") != "diag":
            continue
        designation = clean_text(row.get("designation")).lower()
        if designation not in {"low50", "high50"}:
            continue
        score = clean_text(row.get("score")) or clean_text(row.get("variant"))
        by_score.setdefault(score, {})[designation] = row

    pair_rows: list[dict[str, Any]] = []
    for score, pair in sorted(by_score.items()):
        low = pair.get("low50")
        high = pair.get("high50")
        if not low or not high:
            continue
        out: dict[str, Any] = {
            "score": score,
            "low_variant": low.get("variant", ""),
            "high_variant": high.get("variant", ""),
            "low_cc12": low.get("cc12"),
            "high_cc12": high.get("cc12"),
            "delta_cc12": delta(low.get("cc12"), high.get("cc12")),
            "low_rsplit": low.get("rsplit"),
            "high_rsplit": high.get("rsplit"),
            "delta_rsplit": delta(low.get("rsplit"), high.get("rsplit")),
            "low_snr": low.get("snr"),
            "high_snr": high.get("snr"),
            "delta_snr": delta(low.get("snr"), high.get("snr")),
        }
        pair_rows.append(out)
    return sorted(pair_rows, key=lambda row: safe_float_sort(row.get("delta_cc12"), float("-inf")), reverse=True)


def delta(left: Any, right: Any) -> float | None:
    left_f = to_float(left)
    right_f = to_float(right)
    if left_f is None or right_f is None:
        return None
    return left_f - right_f


def filter_ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filters = [dict(row) for row in rows if clean_text(row.get("classification")).startswith("filter_")]
    filters.sort(
        key=lambda row: (
            {"all": 0, "higheg": 1, "matched": 2}.get(clean_text(row.get("target_family")), 99),
            -safe_float_sort(row.get("cc12"), float("-inf")),
            safe_float_sort(row.get("rsplit"), float("inf")),
            sort_key_text(row.get("score")),
            safe_float_sort(row.get("drop_fraction"), float("inf")),
            sort_key_text(row.get("variant")),
        )
    )

    family_counts: dict[str, int] = {}
    family_score_counts: dict[tuple[str, str], int] = {}
    for row in filters:
        family = clean_text(row.get("target_family"))
        score = clean_text(row.get("score"))
        family_counts[family] = family_counts.get(family, 0) + 1
        family_score_counts[(family, score)] = family_score_counts.get((family, score), 0) + 1
        row["rank_within_family"] = family_counts[family]
        row["rank_within_family_score"] = family_score_counts[(family, score)]
    return filters


def best_filter_per_family(ranked_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in ranked_rows:
        family = clean_text(row.get("target_family"))
        if family not in {"all", "higheg", "matched"}:
            continue
        current = best.get(family)
        if current is None or filter_better(row, current):
            best[family] = row
    return [best[family] for family in ("all", "higheg", "matched") if family in best]


def filter_better(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        safe_float_sort(left.get("cc12"), float("-inf")),
        -safe_float_sort(left.get("rsplit"), float("inf")),
        safe_float_sort(left.get("snr"), float("-inf")),
    ) > (
        safe_float_sort(right.get("cc12"), float("-inf")),
        -safe_float_sort(right.get("rsplit"), float("inf")),
        safe_float_sort(right.get("snr"), float("-inf")),
    )


def top_by(rows: list[dict[str, Any]], metric: str, descending: bool, limit: int = 10) -> list[dict[str, Any]]:
    if descending:
        ranked = sorted(rows, key=lambda row: safe_float_sort(row.get(metric), float("-inf")), reverse=True)
    else:
        ranked = sorted(rows, key=lambda row: safe_float_sort(row.get(metric), float("inf")))
    return [row for row in ranked if to_float(row.get(metric)) is not None][:limit]


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        values = [format_markdown_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if abs(value) < 1 and value != 0:
            return f"{value:.7f}".rstrip("0").rstrip(".")
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|")


def write_markdown_summary(
    path: Path,
    completed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    top_cc12 = top_by(completed, "cc12", descending=True)
    top_rsplit = top_by(completed, "rsplit", descending=False)
    best_filters = best_filter_per_family(filter_rows)
    lines = [
        "# V6 Merge Summary",
        "",
        f"Generated: {metadata['generated_at']}",
        f"OUT: `{metadata['out_dir']}`",
        "",
        "## Counts",
        "",
        markdown_table(
            [
                {"name": "completed non-random result dirs", "count": metadata["counts"]["completed_in_summary"]},
                {"name": "unfinished/missing non-random variants", "count": metadata["counts"]["missing_or_unfinished_in_output"]},
                {"name": "skipped random planned variants", "count": metadata["counts"]["planned_random_variants"]},
            ],
            ["name", "count"],
        ),
        "",
        "## Top 10 by CC1/2",
        "",
        markdown_table(top_cc12, ["variant", "classification", "score", "drop_fraction", "cc12", "rsplit", "snr"]),
        "",
        "## Top 10 by Rsplit",
        "",
        markdown_table(top_rsplit, ["variant", "classification", "score", "drop_fraction", "rsplit", "cc12", "snr"]),
        "",
        "## Diagnostic Low/High Pairs",
        "",
        markdown_table(
            diag_rows,
            ["score", "low_cc12", "high_cc12", "delta_cc12", "low_rsplit", "high_rsplit", "delta_rsplit", "low_snr", "high_snr", "delta_snr"],
        ),
        "",
        "## Best Filter Per Family",
        "",
        markdown_table(best_filters, ["target_family", "variant", "score", "drop_fraction", "cc12", "rsplit", "snr"]),
        "",
        "## Missing or Unfinished",
        "",
        markdown_table(missing[:40], ["variant", "classification", "status", "score", "drop_fraction", "result_dir_name", "missing_core_metrics"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_metadata_json(
    out_dir: Path,
    reference: dict[str, float],
    all_rows: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    metadata_sources: dict[str, dict[str, dict[str, str]]],
    validation: dict[str, Any],
    output_files: dict[str, str],
    include_randoms: bool,
) -> dict[str, Any]:
    planned = planned_variants(metadata_sources)
    planned_random = {variant for variant in planned if variant.startswith("random_")}
    discovered_random = [row for row in all_rows if is_random_row(row)]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "include_randoms": include_randoms,
        "reference": reference,
        "counts": {
            "result_dirs_discovered": len(all_rows),
            "completed_result_dirs_discovered": sum(1 for row in all_rows if row.get("status") == "completed"),
            "completed_in_summary": len(completed),
            "missing_or_unfinished_in_output": len(missing),
            "planned_variants": len(planned),
            "planned_random_variants": len(planned_random),
            "discovered_random_result_dirs": len(discovered_random),
        },
        "metadata_sources": {name: len(rows) for name, rows in metadata_sources.items()},
        "validation": validation,
        "outputs": output_files,
    }


def print_report(completed: list[dict[str, Any]], missing: list[dict[str, Any]], diag_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]], output_files: dict[str, str]) -> None:
    print("\nFiles created:")
    for path in output_files.values():
        print(f"- {path}")

    print(f"\nCompleted non-random result dirs parsed: {len(completed)}")
    print(f"Unfinished/missing non-random variants: {len(missing)}")

    print("\nTop 10 variants by cc12:")
    print(markdown_table(top_by(completed, "cc12", descending=True), ["variant", "cc12", "rsplit", "snr", "classification"]))

    print("\nTop 10 variants by rsplit:")
    print(markdown_table(top_by(completed, "rsplit", descending=False), ["variant", "rsplit", "cc12", "snr", "classification"]))

    print("\nDiagnostic low/high table sorted by delta cc12:")
    print(markdown_table(diag_rows, ["score", "low_cc12", "high_cc12", "delta_cc12", "low_rsplit", "high_rsplit", "delta_rsplit", "low_snr", "high_snr", "delta_snr"]))

    print("\nBest filter variant per family:")
    print(markdown_table(best_filter_per_family(filter_rows), ["target_family", "variant", "score", "drop_fraction", "cc12", "rsplit", "snr"]))


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    if not out_dir.is_dir():
        raise SystemExit(f"OUT directory does not exist: {out_dir}")

    reference = {
        "cc12": args.reference_cc12,
        "rsplit": args.reference_rsplit,
        "snr": args.reference_snr,
        "completeness": args.reference_completeness,
        "redundancy": args.reference_redundancy,
    }
    metadata_sources, validation = load_metadata(out_dir)
    all_rows = parse_all_results(out_dir, metadata_sources, reference)
    completed = complete_rows(all_rows, include_randoms=args.include_randoms)
    completed.sort(
        key=lambda row: (
            sort_key_text(row.get("classification")),
            sort_key_text(row.get("target_family")),
            sort_key_text(row.get("score")),
            safe_float_sort(row.get("drop_fraction"), float("inf")),
            sort_key_text(row.get("variant")),
        )
    )
    missing = missing_or_unfinished_rows(all_rows, metadata_sources, reference, include_randoms=args.include_randoms)
    diag_rows = diagnostic_pairs(completed)
    filter_rows = filter_ranked(completed)

    output_files = {
        "summary_csv": str(out_dir / "merge_summary_nonrandom.csv"),
        "summary_md": str(out_dir / "merge_summary_nonrandom.md"),
        "diag_pairs_csv": str(out_dir / "merge_diag_low_high_pairs.csv"),
        "filter_ranked_csv": str(out_dir / "merge_filter_ranked.csv"),
        "missing_csv": str(out_dir / "merge_missing_or_unfinished.csv"),
        "metadata_json": str(out_dir / "merge_summary_metadata.json"),
    }

    write_csv(Path(output_files["summary_csv"]), completed)
    write_csv(
        Path(output_files["diag_pairs_csv"]),
        diag_rows,
        [
            "score",
            "low_variant",
            "high_variant",
            "low_cc12",
            "high_cc12",
            "delta_cc12",
            "low_rsplit",
            "high_rsplit",
            "delta_rsplit",
            "low_snr",
            "high_snr",
            "delta_snr",
        ],
    )
    write_csv(Path(output_files["filter_ranked_csv"]), filter_rows)
    write_csv(Path(output_files["missing_csv"]), missing)

    metadata_json = build_metadata_json(
        out_dir,
        reference,
        all_rows,
        completed,
        missing,
        metadata_sources,
        validation,
        output_files,
        args.include_randoms,
    )
    write_markdown_summary(Path(output_files["summary_md"]), completed, missing, diag_rows, filter_rows, metadata_json)
    Path(output_files["metadata_json"]).write_text(json.dumps(metadata_json, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")

    print_report(completed, missing, diag_rows, filter_rows, output_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
