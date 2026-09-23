#!/usr/bin/env python3
"""Summarize existing OriDyn V5 Partialator merge results.

This script is read-only with respect to configured experiment directories.  It
discovers existing streams and merge-result directories, parses completed QC
outputs, records missing/incomplete runs, resolves duplicate merge runs per
stream, joins available score metadata, and writes compact summary tables.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "crystfel.hkl",
    "crystfel.hkl1",
    "crystfel.hkl2",
    "parameters.json",
    "qc_stats/check_hkl_completeness.log",
    "qc_stats/check_shell.tsv",
    "qc_stats/compare_cc12.log",
    "qc_stats/compare_cc12_shell.tsv",
    "qc_stats/compare_rsplit.log",
    "qc_stats/compare_rsplit_shell.tsv",
]

OPTIONAL_FILES = [
    "cell.cell",
    "metadata_and_outputs.txt",
    "partialator_stdout.log",
    "partialator_stderr.log",
    "pr-logs",
    "qc_stats",
    "shelx",
]

METADATA_FILES = [
    "experiment_plan.csv",
    "experiment_plan.json",
    "score_variant_parameters.csv",
    "scores.json",
    "summary.csv",
    "per_variant_filter_summary.csv",
    "score_comparison_by_current_reference.csv",
    "parameters.json",
    "run_metadata.json",
    "sweep_audit.json",
    "validation.json",
]

METRIC_NAMES = [
    "completeness",
    "redundancy",
    "snr",
    "cc12",
    "rsplit",
    "merged_reflection_count",
    "observation_count",
    "low_resolution_limit_A",
    "high_resolution_limit_A",
    "number_of_shells",
]

GLOBAL_NUMERIC_COLUMNS = [
    *METRIC_NAMES,
    "delta_cc12",
    "delta_rsplit",
    "delta_redundancy",
    "delta_snr",
    "delta_completeness",
]

FLOAT_RE = re.compile(r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?|[-+]?inf|nan", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"(20\d{6}T\d{4,6}|20\d{2}[-_]\d{2}[-_]\d{2}[T_ -]\d{2}[:_-]?\d{2}(?::?\d{2})?)")
PARTIALATOR_SUFFIX_RE = re.compile(r"_partialator_results(?:_.*)?$")


@dataclass
class Logger:
    lines: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.lines.append(line)
        print(line, flush=True)

    def progress(self, stage: str, completed: int, total: int, started: float) -> None:
        elapsed = max(time.monotonic() - started, 1.0e-9)
        rate = completed / elapsed if elapsed > 0 else float("nan")
        pct = (100.0 * completed / total) if total else 100.0
        eta = ((total - completed) / rate) if rate > 0 and completed < total else 0.0
        self.log(
            f"{stage}: {completed}/{total} ({pct:.1f}%) complete; "
            f"elapsed={format_seconds(elapsed)}; rate={rate:.2f}/s; eta={format_seconds(eta)}"
        )

    def ensure_run_log(self, out_dir: Path) -> None:
        path = out_dir / "run.log"
        try:
            if path.exists() and path.stat().st_size > 0:
                return
            path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        except OSError:
            return


@dataclass(frozen=True)
class StreamRecord:
    experiment_order: int
    experiment_name: str
    experiment_dir: str
    stream_order: int
    stream_filename: str
    stream_stem: str
    stream_path: str


@dataclass(frozen=True)
class RunCandidate:
    experiment_order: int
    experiment_name: str
    experiment_dir: str
    merge_order: int
    merge_dir: str
    merge_dir_name: str
    association_method: str
    association_candidate: str
    association_warning: str
    stream_order: int | None
    stream_filename: str
    stream_stem: str
    stream_path: str
    stream_exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


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
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def now_iso_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    rounded = int(round(number))
    return rounded if abs(number - rounded) < 1.0e-6 else None


def finite_or_nan(value: Any) -> float:
    number = to_float(value)
    return float(number) if number is not None else float("nan")


def bool_from_any(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "no", "n", "0", "off", ""}:
        return False
    return None


def decimal_places(text: str) -> int | None:
    match = FLOAT_RE.search(text)
    if not match:
        return None
    token = match.group(0)
    if "e" in token.lower():
        return None
    if "." not in token:
        return 0
    return len(token.split(".", 1)[1].rstrip("% "))


def read_json_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, f"missing file: {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not parse JSON {path}: {exc}"


def read_text_lines(path: Path) -> tuple[list[str], str | None]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle], None
    except FileNotFoundError:
        return [], f"missing file: {path}"
    except OSError as exc:
        return [], f"could not read {path}: {exc}"


def nested_get(mapping: Any, keys: Iterable[str]) -> Any:
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def find_first_key(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        for value in mapping.values():
            found = find_first_key(value, key)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = find_first_key(value, key)
            if found is not None:
                return found
    return None


def first_nonmissing(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if str(value).strip() == "":
            continue
        return value
    return None


def set_metric(row: dict[str, Any], metric: str, value: Any, source: str, precision: str, priority: int) -> None:
    number = to_float(value)
    if number is None and metric not in {"number_of_shells", "merged_reflection_count", "observation_count"}:
        return
    current_priority = row.get(f"{metric}_source_priority")
    if current_priority is not None and int(current_priority) <= priority:
        return
    if metric in {"number_of_shells", "merged_reflection_count", "observation_count"}:
        parsed_int = to_int(value)
        if parsed_int is None:
            return
        row[metric] = parsed_int
    else:
        row[metric] = number
    row[f"{metric}_source_file"] = source
    row[f"{metric}_precision"] = precision
    row[f"{metric}_source_priority"] = priority


def load_config(path: Path) -> list[str]:
    data, error = read_json_file(path)
    if error:
        raise SystemExit(error)
    if isinstance(data, list):
        dirs = data
    elif isinstance(data, dict) and isinstance(data.get("source_directories"), list):
        dirs = data["source_directories"]
    else:
        raise SystemExit("Config must be a JSON list or contain a source_directories list")
    if not dirs:
        raise SystemExit("Config contains no source directories")
    clean = []
    for idx, item in enumerate(dirs, start=1):
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(f"Config source directory #{idx} is not a non-empty string")
        clean.append(item.strip())
    return clean


def parse_timestamp_text(text: str) -> tuple[str, float]:
    match = TIMESTAMP_RE.search(text)
    if not match:
        return "", 0.0
    raw = match.group(1).replace("_", "-")
    candidates = []
    if re.fullmatch(r"20\d{6}T\d{4,6}", raw):
        if len(raw) == 13:
            candidates.append((raw, "%Y%m%dT%H%M"))
        else:
            candidates.append((raw, "%Y%m%dT%H%M%S"))
    candidates.extend(
        [
            (raw.replace("_", "T"), "%Y-%m-%dT%H:%M:%S"),
            (raw.replace("_", "T"), "%Y-%m-%dT%H:%M"),
            (raw.replace(" ", "T"), "%Y-%m-%dT%H:%M:%S"),
            (raw.replace(" ", "T"), "%Y-%m-%dT%H:%M"),
            (raw, "%Y-%m-%dT%H-%M"),
        ]
    )
    for candidate, fmt in candidates:
        try:
            dt = datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat(timespec="minutes").replace("+00:00", "Z"), dt.timestamp()
        except ValueError:
            continue
    return match.group(1), 0.0


def timestamp_from_value(value: Any) -> tuple[str, float]:
    if value is None:
        return "", 0.0
    text = str(value).strip()
    if not text:
        return "", 0.0
    for candidate in [text, text.replace(" ", "T")]:
        try:
            dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat(timespec="minutes"), dt.timestamp()
        except ValueError:
            pass
    return parse_timestamp_text(text)


def fallback_stream_stem(merge_dir_name: str) -> str:
    return PARTIALATOR_SUFFIX_RE.sub("", merge_dir_name)


def stream_path_from_metadata(path: Path) -> str:
    lines, error = read_text_lines(path)
    if error:
        return ""
    for line in lines:
        match = re.match(r"\s*STREAM\s*:\s*(\S.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def associate_merge_dir(
    merge_dir: Path,
    streams_by_path: dict[str, StreamRecord],
    streams_by_stem: dict[str, StreamRecord],
) -> tuple[StreamRecord | None, str, str, str]:
    params_path = merge_dir / "parameters.json"
    data, _ = read_json_file(params_path)
    if data is not None:
        candidate = find_first_key(data, "stream_file")
        if candidate is not None:
            candidate_text = str(candidate).strip()
            if candidate_text in streams_by_path:
                return streams_by_path[candidate_text], "parameters_json_stream_file", candidate_text, ""
            return None, "parameters_json_stream_file_unmatched", candidate_text, "parameters.json stream_file did not match a discovered top-level stream exactly"

    metadata_candidate = stream_path_from_metadata(merge_dir / "metadata_and_outputs.txt")
    if metadata_candidate:
        if metadata_candidate in streams_by_path:
            return streams_by_path[metadata_candidate], "metadata_and_outputs_stream", metadata_candidate, ""
        return None, "metadata_and_outputs_stream_unmatched", metadata_candidate, "metadata_and_outputs.txt STREAM did not match a discovered top-level stream exactly"

    stem = fallback_stream_stem(merge_dir.name)
    if stem in streams_by_stem:
        return streams_by_stem[stem], "merge_dir_stem_fallback", stem, "associated by merge-directory stem fallback"
    return None, "merge_dir_stem_fallback_unmatched", stem, "merge-directory stem fallback did not match a discovered top-level stream"


def discover_experiment(order: int, experiment_dir: str) -> tuple[list[StreamRecord], list[RunCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    exp_path = Path(experiment_dir)
    experiment_name = exp_path.name
    warnings: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    streams: list[StreamRecord] = []
    candidates: list[RunCandidate] = []
    if not exp_path.is_dir():
        coverage_rows.append(
            {
                "experiment_order": order,
                "experiment_name": experiment_name,
                "experiment_dir": str(exp_path),
                "experiment_exists": False,
                "stream_order": "",
                "stream_filename": "",
                "stream_stem": "",
                "stream_path": "",
                "merge_run_count": 0,
                "merge_status": "experiment_directory_missing",
                "merge_dirs_json": "[]",
            }
        )
        warnings.append(make_warning("discovery", experiment_name, str(exp_path), "", "", "error", "Configured experiment directory is missing", str(exp_path)))
        return streams, candidates, coverage_rows, warnings

    stream_paths = sorted(path for path in exp_path.glob("*.stream") if path.is_file())
    for stream_order, stream in enumerate(stream_paths, start=1):
        streams.append(
            StreamRecord(
                experiment_order=order,
                experiment_name=experiment_name,
                experiment_dir=str(exp_path),
                stream_order=stream_order,
                stream_filename=stream.name,
                stream_stem=stream.stem,
                stream_path=str(stream),
            )
        )
    streams_by_path = {stream.stream_path: stream for stream in streams}
    streams_by_stem = {stream.stream_stem: stream for stream in streams}

    merge_dirs = sorted(path for path in exp_path.glob("*_partialator_results_*") if path.is_dir())
    runs_by_stream: dict[str, list[str]] = defaultdict(list)
    for merge_order, merge_dir in enumerate(merge_dirs, start=1):
        stream, method, candidate, warning = associate_merge_dir(merge_dir, streams_by_path, streams_by_stem)
        if stream is not None:
            runs_by_stream[stream.stream_path].append(str(merge_dir))
            stream_order = stream.stream_order
            stream_filename = stream.stream_filename
            stream_stem = stream.stream_stem
            stream_path = stream.stream_path
            stream_exists = True
        else:
            stream_order = None
            stream_filename = f"{candidate}.stream" if candidate and not str(candidate).endswith(".stream") else Path(str(candidate)).name
            stream_stem = Path(stream_filename).stem if stream_filename else fallback_stream_stem(merge_dir.name)
            stream_path = str(exp_path / stream_filename) if stream_filename else ""
            stream_exists = False
            warnings.append(make_warning("association", experiment_name, str(exp_path), stream_path, str(merge_dir), "warning", warning, str(merge_dir)))
        candidates.append(
            RunCandidate(
                experiment_order=order,
                experiment_name=experiment_name,
                experiment_dir=str(exp_path),
                merge_order=merge_order,
                merge_dir=str(merge_dir),
                merge_dir_name=merge_dir.name,
                association_method=method,
                association_candidate=candidate,
                association_warning=warning,
                stream_order=stream_order,
                stream_filename=stream_filename,
                stream_stem=stream_stem,
                stream_path=stream_path,
                stream_exists=stream_exists,
            )
        )

    for stream in streams:
        merge_dirs_for_stream = runs_by_stream.get(stream.stream_path, [])
        count = len(merge_dirs_for_stream)
        status = "no_merge_result" if count == 0 else "one_merge_result" if count == 1 else "multiple_merge_results"
        coverage_rows.append(
            {
                "experiment_order": order,
                "experiment_name": experiment_name,
                "experiment_dir": str(exp_path),
                "experiment_exists": True,
                "stream_order": stream.stream_order,
                "stream_filename": stream.stream_filename,
                "stream_stem": stream.stream_stem,
                "stream_path": stream.stream_path,
                "merge_run_count": count,
                "merge_status": status,
                "merge_dirs_json": compact_json(merge_dirs_for_stream),
            }
        )
    if not streams:
        coverage_rows.append(
            {
                "experiment_order": order,
                "experiment_name": experiment_name,
                "experiment_dir": str(exp_path),
                "experiment_exists": True,
                "stream_order": "",
                "stream_filename": "",
                "stream_stem": "",
                "stream_path": "",
                "merge_run_count": len(merge_dirs),
                "merge_status": "no_top_level_streams",
                "merge_dirs_json": compact_json([str(path) for path in merge_dirs]),
            }
        )
    return streams, candidates, coverage_rows, warnings


def make_warning(
    stage: str,
    experiment_name: str,
    experiment_dir: str,
    stream_path: str,
    merge_dir: str,
    severity: str,
    message: str,
    source_file: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "stream_path": stream_path,
        "merge_results_dir": merge_dir,
        "severity": severity,
        "message": message,
        "source_file": source_file,
    }


def parse_check_hkl_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines, error = read_text_lines(path)
    metrics: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    if error:
        return metrics, [{"severity": "error", "message": error, "source_file": str(path)}]
    for line in lines:
        if match := re.search(r"(\d+)\s+measurements\s+in\s+total", line):
            metrics["observation_count"] = int(match.group(1))
            metrics["observation_count_precision"] = "integer_log"
        elif match := re.search(r"(\d+)\s+reflections\s+in\s+total", line):
            metrics["merged_reflection_count"] = int(match.group(1))
            metrics["merged_reflection_count_precision"] = "integer_log"
        elif match := re.search(r"Overall\s+<snr>\s*=\s*(\S+)", line, re.IGNORECASE):
            metrics["snr"] = to_float(match.group(1))
            metrics["snr_precision"] = f"log_decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(r"Overall\s+redundancy\s*=\s*(\S+)", line, re.IGNORECASE):
            metrics["redundancy"] = to_float(match.group(1))
            metrics["redundancy_precision"] = f"log_decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(r"Overall\s+completeness\s*=\s*(\S+)", line, re.IGNORECASE):
            metrics["completeness"] = to_float(match.group(1))
            metrics["completeness_precision"] = f"log_decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(r"1/d\s+goes\s+from\s+(\S+)\s+to\s+(\S+)\s+nm", line, re.IGNORECASE):
            min_inv = to_float(match.group(1))
            max_inv = to_float(match.group(2))
            if min_inv and max_inv and min_inv > 0 and max_inv > 0:
                metrics["low_resolution_limit_A"] = 10.0 / min_inv
                metrics["high_resolution_limit_A"] = 10.0 / max_inv
                metrics["resolution_precision"] = "computed_from_log_invnm"
    return metrics, warnings


def parse_compare_log(path: Path, metric: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines, error = read_text_lines(path)
    metrics: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    if error:
        return metrics, [{"severity": "error", "message": error, "source_file": str(path)}]
    for line in lines:
        if metric == "cc12" and (match := re.search(r"Overall\s+CC\s*=\s*(\S+)", line, re.IGNORECASE)):
            metrics["cc12"] = to_float(match.group(1))
            metrics["cc12_precision"] = f"log_decimal_places_{decimal_places(match.group(1))}"
        elif metric == "rsplit" and (match := re.search(r"Overall\s+Rsplit\s*=\s*(\S+)", line, re.IGNORECASE)):
            metrics["rsplit"] = to_float(match.group(1))
            metrics["rsplit_precision"] = f"log_decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(
            r"Accepted\s+resolution\s+range:\s+(\S+)\s+to\s+(\S+)\s+nm\^-1\s+\((\S+)\s+to\s+(\S+)\s+Angstroms\)",
            line,
            re.IGNORECASE,
        ):
            low_a = to_float(match.group(3))
            high_a = to_float(match.group(4))
            if low_a is not None and high_a is not None:
                metrics["low_resolution_limit_A"] = low_a
                metrics["high_resolution_limit_A"] = high_a
                metrics["resolution_precision"] = f"accepted_range_log_decimal_places_{decimal_places(match.group(3))}_{decimal_places(match.group(4))}"
        elif match := re.search(r"(\d+)\s+reflection\s+pairs\s+accepted", line, re.IGNORECASE):
            metrics.setdefault("merged_reflection_count", int(match.group(1)))
            metrics.setdefault("merged_reflection_count_precision", "integer_log_reflection_pairs")
    return metrics, warnings


def parse_metadata_headlines(path: Path) -> dict[str, Any]:
    lines, error = read_text_lines(path)
    metrics: dict[str, Any] = {}
    if error:
        return metrics
    for line in lines:
        if match := re.search(r"Completeness:\s*(\S+)%", line, re.IGNORECASE):
            metrics["completeness"] = to_float(match.group(1))
            metrics["completeness_precision"] = "rounded_metadata_summary"
        elif match := re.search(r"Redundancy:\s*(\S+)", line, re.IGNORECASE):
            metrics["redundancy"] = to_float(match.group(1).rstrip("xX"))
            metrics["redundancy_precision"] = "rounded_metadata_summary"
        elif match := re.search(r"SNR:\s*(\S+)", line, re.IGNORECASE):
            metrics["snr"] = to_float(match.group(1))
            metrics["snr_precision"] = "rounded_metadata_summary"
        elif match := re.search(r"CC1/2:\s*(\S+)", line, re.IGNORECASE):
            metrics["cc12"] = to_float(match.group(1))
            metrics["cc12_precision"] = "metadata_summary"
        elif match := re.search(r"Rsplit:\s*(\S+)", line, re.IGNORECASE):
            metrics["rsplit"] = to_float(match.group(1))
            metrics["rsplit_precision"] = "rounded_metadata_summary"
        elif match := re.search(r"Range:\s*(\S+)-(\S+)\s*[AÅ];\s*shells=(\d+)", line, re.IGNORECASE):
            metrics["low_resolution_limit_A"] = to_float(match.group(1))
            metrics["high_resolution_limit_A"] = to_float(match.group(2))
            metrics["number_of_shells"] = int(match.group(3))
            metrics["resolution_precision"] = "metadata_range_summary"
    return metrics


def numbers_from_line(line: str) -> list[float]:
    values: list[float] = []
    for match in FLOAT_RE.finditer(line):
        value = to_float(match.group(0))
        if value is not None:
            values.append(value)
    return values


def parse_shell_table(path: Path, kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lines, error = read_text_lines(path)
    warnings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if error:
        return rows, [{"severity": "error", "message": error, "source_file": str(path)}]
    header = next((line for line in lines if line.strip()), "")
    normalized_header = re.sub(r"\s+", " ", header.strip().lower())
    if kind == "check_shell" and not all(token in normalized_header for token in ["compl", "meas", "snr", "min 1/nm", "max 1/nm"]):
        warnings.append({"severity": "warning", "message": f"Unexpected check_shell.tsv header: {header}", "source_file": str(path)})
    if kind == "cc12_shell" and not all(token in normalized_header for token in ["cc", "nref", "min 1/nm", "max 1/nm"]):
        warnings.append({"severity": "warning", "message": f"Unexpected compare_cc12_shell.tsv header: {header}", "source_file": str(path)})
    if kind == "rsplit_shell" and not all(token in normalized_header for token in ["rsplit", "nref", "min 1/nm", "max 1/nm"]):
        warnings.append({"severity": "warning", "message": f"Unexpected compare_rsplit_shell.tsv header: {header}", "source_file": str(path)})

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or not re.match(r"[-+.\d]", stripped):
            continue
        values = numbers_from_line(stripped)
        if kind == "check_shell":
            if len(values) < 11:
                warnings.append({"severity": "error", "message": f"Could not parse check_shell row with {len(values)} numeric fields: {stripped}", "source_file": str(path)})
                continue
            inv_center, nref, possible, compl, meas, red, snr, mean_i, d_a, min_inv, max_inv = values[:11]
            rows.append(
                {
                    "shell_index": len(rows) + 1,
                    "reciprocal_resolution_1_per_nm": inv_center,
                    "check_nref": int(round(nref)),
                    "possible_reflections": int(round(possible)),
                    "completeness": compl,
                    "observation_count": int(round(meas)),
                    "redundancy": red,
                    "snr": snr,
                    "mean_i": mean_i,
                    "shell_center_resolution_A": d_a,
                    "min_invnm": min_inv,
                    "max_invnm": max_inv,
                    "check_shell_source_file": str(path),
                }
            )
        elif kind == "cc12_shell":
            if len(values) < 6:
                warnings.append({"severity": "error", "message": f"Could not parse CC1/2 shell row with {len(values)} numeric fields: {stripped}", "source_file": str(path)})
                continue
            inv_center, cc, nref, d_a, min_inv, max_inv = values[:6]
            rows.append(
                {
                    "shell_index": len(rows) + 1,
                    "reciprocal_resolution_1_per_nm": inv_center,
                    "cc12": cc,
                    "cc12_nref": int(round(nref)),
                    "shell_center_resolution_A": d_a,
                    "min_invnm": min_inv,
                    "max_invnm": max_inv,
                    "compare_cc12_shell_source_file": str(path),
                }
            )
        elif kind == "rsplit_shell":
            if len(values) < 6:
                warnings.append({"severity": "error", "message": f"Could not parse Rsplit shell row with {len(values)} numeric fields: {stripped}", "source_file": str(path)})
                continue
            inv_center, rsplit, nref, d_a, min_inv, max_inv = values[:6]
            rows.append(
                {
                    "shell_index": len(rows) + 1,
                    "reciprocal_resolution_1_per_nm": inv_center,
                    "rsplit": rsplit,
                    "rsplit_nref": int(round(nref)),
                    "shell_center_resolution_A": d_a,
                    "min_invnm": min_inv,
                    "max_invnm": max_inv,
                    "compare_rsplit_shell_source_file": str(path),
                }
            )
    if not rows:
        warnings.append({"severity": "error", "message": f"No shell rows parsed from {path.name}", "source_file": str(path)})
    return rows, warnings


def boundary_key(row: dict[str, Any], places: int = 6) -> tuple[float, float] | None:
    min_inv = to_float(row.get("min_invnm"))
    max_inv = to_float(row.get("max_invnm"))
    if min_inv is None or max_inv is None:
        return None
    return (round(min_inv, places), round(max_inv, places))


def shell_scheme_id_from_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    boundaries: list[tuple[float, float]] = []
    for row in sorted(rows, key=lambda item: (int(item.get("shell_index") or 0), float(item.get("min_invnm") or 0.0))):
        key = boundary_key(row)
        if key is not None:
            boundaries.append(key)
    payload = compact_json(boundaries)
    if not boundaries:
        return "", "[]"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"shellscheme_{digest}", payload


def index_by_boundary(rows: list[dict[str, Any]], label: str, warnings: list[dict[str, Any]], source_file: str) -> dict[tuple[float, float], dict[str, Any]]:
    out: dict[tuple[float, float], dict[str, Any]] = {}
    duplicates: list[tuple[float, float]] = []
    for row in rows:
        key = boundary_key(row)
        if key is None:
            warnings.append({"severity": "error", "message": f"{label} row lacks explicit shell boundaries", "source_file": source_file})
            continue
        if key in out:
            duplicates.append(key)
        out[key] = row
    if duplicates:
        warnings.append({"severity": "error", "message": f"{label} has duplicate shell boundary keys: {duplicates[:5]}", "source_file": source_file})
    return out


def join_shell_tables(
    candidate: RunCandidate,
    check_rows: list[dict[str, Any]],
    cc_rows: list[dict[str, Any]],
    rsplit_rows: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    check_index = index_by_boundary(check_rows, "check_shell.tsv", warnings, str(Path(candidate.merge_dir) / "qc_stats/check_shell.tsv")) if check_rows else {}
    cc_index = index_by_boundary(cc_rows, "compare_cc12_shell.tsv", warnings, str(Path(candidate.merge_dir) / "qc_stats/compare_cc12_shell.tsv")) if cc_rows else {}
    rsplit_index = index_by_boundary(rsplit_rows, "compare_rsplit_shell.tsv", warnings, str(Path(candidate.merge_dir) / "qc_stats/compare_rsplit_shell.tsv")) if rsplit_rows else {}
    keys = sorted(set(check_index) | set(cc_index) | set(rsplit_index))
    joined: list[dict[str, Any]] = []
    for idx, key in enumerate(keys, start=1):
        base = dict(check_index.get(key) or cc_index.get(key) or rsplit_index.get(key) or {})
        base["shell_index"] = idx
        if key[0] > 0:
            base["shell_lower_resolution_A"] = 10.0 / key[0]
        else:
            base["shell_lower_resolution_A"] = np.nan
        if key[1] > 0:
            base["shell_upper_resolution_A"] = 10.0 / key[1]
        else:
            base["shell_upper_resolution_A"] = np.nan
        base["min_invnm"] = key[0]
        base["max_invnm"] = key[1]
        parser_warnings: list[str] = []
        if key not in check_index:
            parser_warnings.append("missing check_shell.tsv row for shell boundaries")
        else:
            base.update(check_index[key])
        if key not in cc_index:
            parser_warnings.append("missing compare_cc12_shell.tsv row for shell boundaries")
        else:
            cc = cc_index[key]
            for field_name in ["cc12", "cc12_nref", "compare_cc12_shell_source_file"]:
                base[field_name] = cc.get(field_name)
        if key not in rsplit_index:
            parser_warnings.append("missing compare_rsplit_shell.tsv row for shell boundaries")
        else:
            rs = rsplit_index[key]
            for field_name in ["rsplit", "rsplit_nref", "compare_rsplit_shell_source_file"]:
                base[field_name] = rs.get(field_name)
        nrefs = [base.get(name) for name in ["check_nref", "cc12_nref", "rsplit_nref"] if base.get(name) not in (None, "")]
        if len(set(nrefs)) > 1:
            parser_warnings.append(f"nref mismatch across shell files: {nrefs}")
        base["shell_join_method"] = "explicit_boundaries"
        base["parser_warnings"] = "; ".join(parser_warnings)
        joined.append(base)
    if check_rows and cc_rows and set(check_index) != set(cc_index):
        warnings.append({"severity": "warning", "message": "check_shell.tsv and compare_cc12_shell.tsv shell boundaries do not match exactly", "source_file": str(Path(candidate.merge_dir) / "qc_stats")})
    if check_rows and rsplit_rows and set(check_index) != set(rsplit_index):
        warnings.append({"severity": "warning", "message": "check_shell.tsv and compare_rsplit_shell.tsv shell boundaries do not match exactly", "source_file": str(Path(candidate.merge_dir) / "qc_stats")})
    scheme_id, boundary_json = shell_scheme_id_from_rows(joined)
    scheme = {
        "shell_scheme_id": scheme_id,
        "shell_boundary_json": boundary_json,
        "number_of_shells": len(joined),
        "shell_join_method": "explicit_boundaries" if joined else "",
    }
    return joined, scheme


def parse_partialator_parameters(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data, error = read_json_file(path)
    if error:
        return {}, [{"severity": "error", "message": error, "source_file": str(path)}]
    wrapper = data.get("merge_wrapper", {}) if isinstance(data, dict) else {}
    merging = data.get("merging", {}) if isinstance(data, dict) else {}
    row: dict[str, Any] = {}
    row["symmetry"] = first_nonmissing(wrapper.get("symmetry"), merging.get("symmetry"))
    row["partialator_model"] = first_nonmissing(wrapper.get("model"), merging.get("partiality_model"))
    row["partialator_iterations"] = first_nonmissing(wrapper.get("iterations"), merging.get("num_iterations"))
    row["minimum_measurements"] = first_nonmissing(wrapper.get("min_measurements"), merging.get("min_measurements_per_unique_reflection"))
    row["thread_count"] = first_nonmissing(wrapper.get("threads"))
    disable_pr = bool_from_any(wrapper.get("disable_pr"))
    post_refine = bool_from_any(merging.get("post_refine"))
    if disable_pr is not None:
        row["pr_enabled"] = not disable_pr
    elif post_refine is not None:
        row["pr_enabled"] = post_refine
    no_bscale = bool_from_any(wrapper.get("no_bscale"))
    bscale = bool_from_any(merging.get("Bscale"))
    if no_bscale is not None:
        row["bscale_enabled"] = not no_bscale
    elif bscale is not None:
        row["bscale_enabled"] = bscale
    row["parameters_stream_file"] = first_nonmissing(wrapper.get("stream_file"), find_first_key(data, "stream_file"))
    row["run_started"] = first_nonmissing(wrapper.get("run_started"), find_first_key(data, "run_started"))
    row["run_id"] = first_nonmissing(wrapper.get("run_id"), find_first_key(data, "run_id"))
    row["lowres_parameter_A"] = to_float(first_nonmissing(wrapper.get("lowres"), wrapper.get("low_resolution_A")))
    row["highres_parameter_A"] = to_float(first_nonmissing(wrapper.get("highres"), wrapper.get("high_resolution_A")))
    row["partialator_command"] = wrapper.get("partialator_command")
    return row, []


def parse_run(candidate: RunCandidate) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    merge_dir = Path(candidate.merge_dir)
    warnings: list[dict[str, Any]] = []
    row: dict[str, Any] = {
        "experiment_order": candidate.experiment_order,
        "experiment_name": candidate.experiment_name,
        "experiment_dir": candidate.experiment_dir,
        "stream_order": candidate.stream_order if candidate.stream_order is not None else "",
        "stream_filename": candidate.stream_filename,
        "stream_path": candidate.stream_path,
        "stream_stem": candidate.stream_stem,
        "stream_exists": candidate.stream_exists,
        "merge_order": candidate.merge_order,
        "merge_results_dir": str(merge_dir),
        "merge_results_name": merge_dir.name,
        "association_method": candidate.association_method,
        "association_candidate": candidate.association_candidate,
        "association_warning": candidate.association_warning,
        "selected": False,
        "selection_reason": "",
    }
    for metric in METRIC_NAMES:
        row[metric] = np.nan
        row[f"{metric}_source_file"] = ""
        row[f"{metric}_precision"] = ""

    missing_required = [rel for rel in REQUIRED_FILES if not (merge_dir / rel).exists()]
    missing_optional = [rel for rel in OPTIONAL_FILES if not (merge_dir / rel).exists()]
    row["required_files_present"] = len(missing_required) == 0
    row["missing_required_files"] = "; ".join(missing_required)
    row["missing_optional_files"] = "; ".join(missing_optional)
    for rel in missing_required:
        warnings.append(make_warning("parse", candidate.experiment_name, candidate.experiment_dir, candidate.stream_path, str(merge_dir), "error", f"Missing required output: {rel}", str(merge_dir / rel)))
    for rel in missing_optional:
        warnings.append(make_warning("parse", candidate.experiment_name, candidate.experiment_dir, candidate.stream_path, str(merge_dir), "warning", f"Missing optional output: {rel}", str(merge_dir / rel)))

    param_row, param_warnings = parse_partialator_parameters(merge_dir / "parameters.json")
    row.update(param_row)
    for warning in param_warnings:
        warnings.append(make_warning("parse", candidate.experiment_name, candidate.experiment_dir, candidate.stream_path, str(merge_dir), warning["severity"], warning["message"], warning["source_file"]))

    metadata_metrics = parse_metadata_headlines(merge_dir / "metadata_and_outputs.txt")
    check_metrics, check_warnings = parse_check_hkl_log(merge_dir / "qc_stats/check_hkl_completeness.log")
    cc_metrics, cc_warnings = parse_compare_log(merge_dir / "qc_stats/compare_cc12.log", "cc12")
    rsplit_metrics, rsplit_warnings = parse_compare_log(merge_dir / "qc_stats/compare_rsplit.log", "rsplit")
    for warning in [*check_warnings, *cc_warnings, *rsplit_warnings]:
        warnings.append(make_warning("parse", candidate.experiment_name, candidate.experiment_dir, candidate.stream_path, str(merge_dir), warning["severity"], warning["message"], warning["source_file"]))

    for metric in ["completeness", "redundancy", "snr", "merged_reflection_count", "observation_count"]:
        if metric in check_metrics:
            set_metric(row, metric, check_metrics[metric], str(merge_dir / "qc_stats/check_hkl_completeness.log"), check_metrics.get(f"{metric}_precision", "full_precision_log"), 10)
    if "cc12" in cc_metrics:
        set_metric(row, "cc12", cc_metrics["cc12"], str(merge_dir / "qc_stats/compare_cc12.log"), cc_metrics.get("cc12_precision", "full_precision_log"), 10)
    if "rsplit" in rsplit_metrics:
        set_metric(row, "rsplit", rsplit_metrics["rsplit"], str(merge_dir / "qc_stats/compare_rsplit.log"), rsplit_metrics.get("rsplit_precision", "log"), 10)
    for metric_source, source_file in [(cc_metrics, "compare_cc12.log"), (rsplit_metrics, "compare_rsplit.log"), (check_metrics, "check_hkl_completeness.log")]:
        if "low_resolution_limit_A" in metric_source:
            set_metric(row, "low_resolution_limit_A", metric_source["low_resolution_limit_A"], str(merge_dir / "qc_stats" / source_file), metric_source.get("resolution_precision", "log"), 10 if source_file.startswith("compare") else 20)
            set_metric(row, "high_resolution_limit_A", metric_source["high_resolution_limit_A"], str(merge_dir / "qc_stats" / source_file), metric_source.get("resolution_precision", "log"), 10 if source_file.startswith("compare") else 20)
    for metric in ["completeness", "redundancy", "snr", "cc12", "rsplit", "low_resolution_limit_A", "high_resolution_limit_A", "number_of_shells"]:
        if metric in metadata_metrics:
            set_metric(row, metric, metadata_metrics[metric], str(merge_dir / "metadata_and_outputs.txt"), metadata_metrics.get(f"{metric}_precision", metadata_metrics.get("resolution_precision", "metadata_fallback")), 50)
    if row.get("low_resolution_limit_A") is np.nan or pd.isna(row.get("low_resolution_limit_A")):
        if row.get("lowres_parameter_A") is not None:
            set_metric(row, "low_resolution_limit_A", row["lowres_parameter_A"], str(merge_dir / "parameters.json"), "parameters_json_fallback", 60)
    if row.get("high_resolution_limit_A") is np.nan or pd.isna(row.get("high_resolution_limit_A")):
        if row.get("highres_parameter_A") is not None:
            set_metric(row, "high_resolution_limit_A", row["highres_parameter_A"], str(merge_dir / "parameters.json"), "parameters_json_fallback", 60)

    check_shell_rows, check_shell_warnings = parse_shell_table(merge_dir / "qc_stats/check_shell.tsv", "check_shell")
    cc_shell_rows, cc_shell_warnings = parse_shell_table(merge_dir / "qc_stats/compare_cc12_shell.tsv", "cc12_shell")
    rsplit_shell_rows, rsplit_shell_warnings = parse_shell_table(merge_dir / "qc_stats/compare_rsplit_shell.tsv", "rsplit_shell")
    local_shell_warnings = [*check_shell_warnings, *cc_shell_warnings, *rsplit_shell_warnings]
    for warning in local_shell_warnings:
        warnings.append(make_warning("parse_shell", candidate.experiment_name, candidate.experiment_dir, candidate.stream_path, str(merge_dir), warning["severity"], warning["message"], warning["source_file"]))
    shell_rows, shell_scheme = join_shell_tables(candidate, check_shell_rows, cc_shell_rows, rsplit_shell_rows, local_shell_warnings)
    for warning in local_shell_warnings:
        if not any(existing.get("message") == warning["message"] and existing.get("source_file") == warning["source_file"] for existing in warnings):
            warnings.append(make_warning("parse_shell", candidate.experiment_name, candidate.experiment_dir, candidate.stream_path, str(merge_dir), warning["severity"], warning["message"], warning["source_file"]))

    if shell_rows:
        set_metric(row, "number_of_shells", len(shell_rows), str(merge_dir / "qc_stats/check_shell.tsv"), "counted_shell_rows", 5)
    row.update(shell_scheme)
    timestamp_iso, timestamp_sort = timestamp_from_value(first_nonmissing(row.get("run_started"), row.get("run_id"), merge_dir.name))
    if not timestamp_iso:
        try:
            timestamp_sort = merge_dir.stat().st_mtime
            timestamp_iso = datetime.fromtimestamp(timestamp_sort).astimezone().isoformat(timespec="minutes")
        except OSError:
            timestamp_sort = 0.0
            timestamp_iso = ""
    row["merge_timestamp"] = timestamp_iso
    row["merge_timestamp_sort"] = timestamp_sort
    row["parsed_metric_count"] = sum(0 if pd.isna(row.get(metric)) else 1 for metric in ["completeness", "redundancy", "snr", "cc12", "rsplit"])
    row["required_present_count"] = len(REQUIRED_FILES) - len(missing_required)
    row["completeness_score_for_selection"] = int(row["required_present_count"]) * 10 + int(row["parsed_metric_count"]) * 5 + (5 if shell_rows else 0)

    parse_errors = [warning for warning in warnings if warning.get("severity") == "error"]
    core_metrics_present = all(not pd.isna(row.get(metric)) for metric in ["completeness", "redundancy", "snr", "cc12", "rsplit"])
    row["parsing_valid"] = len(parse_errors) == 0 and core_metrics_present and bool(shell_rows)
    row["merge_valid"] = bool(row["required_files_present"]) and bool(row["parsing_valid"])
    row["warning_messages"] = "; ".join(warning["message"] for warning in warnings if warning.get("severity") == "warning")
    row["error_messages"] = "; ".join(warning["message"] for warning in warnings if warning.get("severity") == "error")
    row["parser_warning_count"] = sum(1 for warning in warnings if warning.get("severity") == "warning")
    row["parser_error_count"] = sum(1 for warning in warnings if warning.get("severity") == "error")

    enriched_shell_rows: list[dict[str, Any]] = []
    for shell in shell_rows:
        enriched = {
            "experiment_order": candidate.experiment_order,
            "experiment_name": candidate.experiment_name,
            "experiment_dir": candidate.experiment_dir,
            "stream_order": candidate.stream_order if candidate.stream_order is not None else "",
            "stream_stem": candidate.stream_stem,
            "stream_path": candidate.stream_path,
            "merge_results_dir": str(merge_dir),
            "merge_timestamp": row["merge_timestamp"],
            "shell_scheme_id": row.get("shell_scheme_id", ""),
            **shell,
        }
        enriched_shell_rows.append(enriched)
    return row, enriched_shell_rows, shell_scheme, warnings


def bounded_map(
    items: list[Any],
    workers: int,
    func: Callable[[Any], Any],
    logger: Logger,
    stage: str,
) -> list[Any]:
    total = len(items)
    if total == 0:
        logger.progress(stage, 0, 0, time.monotonic())
        return []
    logger.log(f"{stage}: start; total={total}; workers={workers}; bounded_submission={max(1, workers * 2)}")
    started = time.monotonic()
    results: list[Any] = []
    max_pending = max(1, workers * 2)
    iterator = iter(items)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for _ in range(min(max_pending, total)):
            pending.add(executor.submit(func, next(iterator)))
        exhausted = False
        next_report = 1
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                completed += 1
                if completed >= next_report or completed == total:
                    logger.progress(stage, completed, total, started)
                    next_report = min(total, max(next_report + 1, completed + max(1, total // 20)))
                if not exhausted:
                    try:
                        pending.add(executor.submit(func, next(iterator)))
                    except StopIteration:
                        exhausted = True
    logger.progress(stage, total, total, started)
    logger.log(f"{stage}: complete")
    return results


def flatten_scalar_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            nested = flatten_scalar_dict(value, name)
            for nested_key, nested_value in nested.items():
                out[nested_key] = nested_value
        elif isinstance(value, list):
            continue
        else:
            out[name] = value
            out[str(key)] = value
    return out


def extract_json_records(data: Any, source_file: str, parent_scalars: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    parent = dict(parent_scalars or {})
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                row = {**parent, **flatten_scalar_dict(item)}
                row["_source_file"] = source_file
                row["_row_index"] = idx
                row["_raw_json"] = compact_json(item)
                records.append(row)
                for key, value in item.items():
                    if isinstance(value, (list, dict)):
                        records.extend(extract_json_records(value, source_file, {**parent, **flatten_scalar_dict(item)}))
    elif isinstance(data, dict):
        scalars = {**parent, **flatten_scalar_dict(data)}
        row_added = False
        for key, value in data.items():
            if isinstance(value, list):
                row_added = True
                records.extend(extract_json_records(value, source_file, scalars))
            elif isinstance(value, dict):
                records.extend(extract_json_records(value, source_file, scalars))
        if not row_added:
            scalars["_source_file"] = source_file
            scalars["_row_index"] = 0
            scalars["_raw_json"] = compact_json(data)
            records.append(scalars)
    return records


def load_metadata_records(experiment_dir: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exp_path = Path(experiment_dir)
    records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for filename in METADATA_FILES:
        path = exp_path / filename
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".csv":
                table = pd.read_csv(path, low_memory=False)
                for idx, row in table.iterrows():
                    data = {str(key): (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
                    data["_source_file"] = str(path)
                    data["_row_index"] = int(idx)
                    data["_raw_json"] = compact_json({key: value for key, value in data.items() if not key.startswith("_")})
                    records.append(data)
            elif path.suffix.lower() == ".json":
                data, error = read_json_file(path)
                if error:
                    warnings.append(make_warning("metadata", exp_path.name, str(exp_path), "", "", "warning", error, str(path)))
                    continue
                records.extend(extract_json_records(data, str(path)))
        except Exception as exc:  # noqa: BLE001 - metadata should not abort the summary.
            warnings.append(make_warning("metadata", exp_path.name, str(exp_path), "", "", "warning", f"Could not load metadata file {path}: {exc}", str(path)))
    return records, warnings


def value_as_join_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def metadata_match_keys(record: dict[str, Any], stream_path: str, stream_filename: str, stream_stem: str) -> list[str]:
    keys: list[str] = []
    for key, value in record.items():
        if key.startswith("_"):
            continue
        text = value_as_join_text(value)
        if not text:
            continue
        key_lower = key.lower()
        if text == stream_path and (text.endswith(".stream") or "stream" in key_lower):
            keys.append(f"{key}=exact_stream_path")
        elif text == stream_filename and ("stream" in key_lower or key_lower in {"expected_output_filename", "output_filename", "filename"}):
            keys.append(f"{key}=exact_stream_filename")
        elif text.endswith(".stream") and Path(text).name == stream_filename and not Path(text).is_absolute():
            keys.append(f"{key}=exact_relative_stream_filename")
        elif text == stream_stem and key_lower in {"stream_stem", "variant", "variant_name", "experiment", "name", "configuration", "config_name"}:
            keys.append(f"{key}=exact_stream_stem")
    return keys


CANONICAL_SYNONYMS = {
    "variant_name": ["variant", "variant_name", "experiment", "name"],
    "score_name": ["score_name"],
    "family": ["score_family", "family"],
    "formula": ["formula"],
    "expression_tree": ["expression_tree_json", "expression_tree", "tree"],
    "p": ["p", "p_value"],
    "lambda": ["lambda", "lambda_value"],
    "alpha": ["alpha"],
    "sg0_multiplier": ["sg0_multiplier", "sg_multiplier"],
    "sigma_c_multiplier": ["sigma_c_multiplier", "sigma_multiplier"],
    "sigma_c": ["sigma_c"],
    "cutoff_radius": ["cutoff_radius", "r_cut"],
    "filter_name": ["filter_name"],
    "high_eg_fraction": ["high_eg_fraction"],
    "block_size": ["block_size", "excitation_block_size"],
    "drop_fraction": ["drop_fraction"],
    "removed_observation_count": ["removed_observation_count", "actual_removal_count", "expected_removal_count", "removed_count"],
    "accepted_observation_fraction_removed": ["accepted_observation_fraction_removed", "accepted_fraction_removed", "removal_fraction_all_6732955_accepted"],
    "reference_designation": ["reference_designation", "reference_role", "reference", "is_reference", "control", "internal_control"],
    "cache_path": ["cache_path", "score_cache_dir", "cached_score_table", "cache"],
    "production_datetime": ["created_utc", "created", "created_at", "run_started", "date", "datetime", "timestamp"],
}


def canonical_metadata_fields(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source = Path(str(record.get("_source_file", ""))).name
    for canonical, synonyms in CANONICAL_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in record and value_as_join_text(record.get(synonym)):
                if canonical == "score_name" and synonym == "name" and source != "score_variant_parameters.csv":
                    continue
                out[canonical] = record.get(synonym)
                if canonical == "reference_designation":
                    out["reference_designation_field"] = synonym
                break
    if "score_name" not in out and source == "score_variant_parameters.csv" and value_as_join_text(record.get("name")):
        out["score_name"] = record.get("name")
    if "variant_name" in out and source == "score_variant_parameters.csv" and out.get("variant_name") == out.get("score_name"):
        out.pop("variant_name", None)
    return out


def merge_metadata_rows(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for record in records:
        canonical = canonical_metadata_fields(record)
        source = Path(str(record.get("_source_file", ""))).name
        for key, value in canonical.items():
            text = value_as_join_text(value)
            if not text:
                continue
            if key not in merged or not value_as_join_text(merged[key]):
                merged[key] = value
                merged[f"{key}_metadata_source"] = source
            elif value_as_join_text(merged[key]) != text:
                conflicts.append(f"{key}: kept {merged[key]!r} from {merged.get(f'{key}_metadata_source')}, saw {value!r} from {source}")
    return merged, conflicts


def join_metadata_for_stream(
    experiment_name: str,
    experiment_dir: str,
    stream_path: str,
    stream_filename: str,
    stream_stem: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ambiguous: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    by_file: dict[str, list[tuple[dict[str, Any], list[str]]]] = defaultdict(list)
    for record in records:
        keys = metadata_match_keys(record, stream_path, stream_filename, stream_stem)
        if keys:
            by_file[str(record.get("_source_file", ""))].append((record, keys))
    for source_file, entries in by_file.items():
        unique_rows: dict[int, tuple[dict[str, Any], list[str]]] = {}
        for record, keys in entries:
            unique_rows[int(record.get("_row_index", -1))] = (record, keys)
        if len(unique_rows) == 1:
            matched.append(next(iter(unique_rows.values()))[0])
        elif len(unique_rows) > 1:
            ambiguous.append(
                {
                    "experiment_name": experiment_name,
                    "experiment_dir": experiment_dir,
                    "stream_path": stream_path,
                    "stream_stem": stream_stem,
                    "metadata_file": source_file,
                    "match_count": len(unique_rows),
                    "matched_join_keys_json": compact_json([keys for _, keys in unique_rows.values()]),
                    "message": "More than one metadata row matched this stream exactly; skipped this file for the stream",
                }
            )
    merged, conflicts = merge_metadata_rows(matched)

    score_name = value_as_join_text(merged.get("score_name"))
    if score_name:
        score_rows: list[dict[str, Any]] = []
        for record in records:
            if Path(str(record.get("_source_file", ""))).name != "score_variant_parameters.csv":
                continue
            if value_as_join_text(record.get("name")) == score_name or value_as_join_text(record.get("score_name")) == score_name:
                score_rows.append(record)
        unique_score_rows = {int(record.get("_row_index", -1)): record for record in score_rows}
        already_sources = {str(record.get("_source_file", "")) for record in matched}
        if len(unique_score_rows) == 1 and str(next(iter(unique_score_rows.values())).get("_source_file", "")) not in already_sources:
            score_merged, score_conflicts = merge_metadata_rows([next(iter(unique_score_rows.values()))])
            for key, value in score_merged.items():
                if key not in merged or not value_as_join_text(merged.get(key)):
                    merged[key] = value
            conflicts.extend(score_conflicts)
            matched.append(next(iter(unique_score_rows.values())))
        elif len(unique_score_rows) > 1:
            ambiguous.append(
                {
                    "experiment_name": experiment_name,
                    "experiment_dir": experiment_dir,
                    "stream_path": stream_path,
                    "stream_stem": stream_stem,
                    "metadata_file": str(Path(experiment_dir) / "score_variant_parameters.csv"),
                    "match_count": len(unique_score_rows),
                    "matched_join_keys_json": compact_json([f"name={score_name}"]),
                    "message": "More than one score_variant_parameters row matched the already joined score_name",
                }
            )

    merged["metadata_join_status"] = "matched" if matched else "no_metadata_match"
    merged["metadata_sources"] = "; ".join(sorted({Path(str(record.get("_source_file", ""))).name for record in matched if record.get("_source_file")}))
    merged["metadata_conflicts_json"] = compact_json(conflicts)
    merged["metadata_raw_json"] = compact_json(
        [
            {
                "source_file": Path(str(record.get("_source_file", ""))).name,
                "row_index": record.get("_row_index"),
                "raw": json.loads(record.get("_raw_json", "{}")),
            }
            for record in matched
        ]
    )
    return merged, ambiguous


def join_metadata(
    runs: list[dict[str, Any]],
    streams: list[StreamRecord],
    logger: Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    logger.log("joining score metadata: start")
    started = time.monotonic()
    records_by_experiment: dict[str, list[dict[str, Any]]] = {}
    warnings: list[dict[str, Any]] = []
    for idx, exp_dir in enumerate(sorted({stream.experiment_dir for stream in streams}), start=1):
        records, local_warnings = load_metadata_records(exp_dir)
        records_by_experiment[exp_dir] = records
        warnings.extend(local_warnings)
        logger.progress("joining score metadata: loaded metadata files", idx, len(set(stream.experiment_dir for stream in streams)), started)

    metadata_by_stream: dict[tuple[str, str], dict[str, Any]] = {}
    ambiguous: list[dict[str, Any]] = []
    for stream in streams:
        metadata, local_ambiguous = join_metadata_for_stream(
            stream.experiment_name,
            stream.experiment_dir,
            stream.stream_path,
            stream.stream_filename,
            stream.stream_stem,
            records_by_experiment.get(stream.experiment_dir, []),
        )
        metadata_by_stream[(stream.experiment_dir, stream.stream_path)] = metadata
        ambiguous.extend(local_ambiguous)
    for run in runs:
        metadata = metadata_by_stream.get((run["experiment_dir"], run["stream_path"]), {})
        run.update(metadata)
    logger.log("joining score metadata: complete")
    return runs, ambiguous, warnings


def select_duplicate_runs(runs: list[dict[str, Any]], streams: list[StreamRecord], logger: Logger) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    logger.log("resolving duplicate runs: start")
    runs_by_stream: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("stream_path"):
            runs_by_stream[(run["experiment_dir"], run["stream_path"])].append(run)

    selected_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for stream in streams:
        key = (stream.experiment_dir, stream.stream_path)
        group = runs_by_stream.get(key, [])
        if not group:
            continue
        ranked = sorted(
            group,
            key=lambda row: (
                bool(row.get("merge_valid")),
                int(row.get("completeness_score_for_selection") or 0),
                float(row.get("merge_timestamp_sort") or 0.0),
                str(row.get("merge_results_name") or ""),
            ),
            reverse=True,
        )
        selected = ranked[0]
        selected["selected"] = True
        if len(group) == 1:
            selected["selection_reason"] = "only_merge_run_selected"
        elif selected.get("merge_valid"):
            selected["selection_reason"] = "selected_latest_complete_parseable_run"
        else:
            selected["selection_reason"] = "selected_best_incomplete_run_no_complete_parseable_run"
        for other in group:
            if other is selected:
                continue
            other["selected"] = False
            if selected.get("merge_valid") and not other.get("merge_valid"):
                other["selection_reason"] = "not_selected_incomplete_or_failed_duplicate"
            elif selected.get("merge_valid") == other.get("merge_valid") and float(selected.get("merge_timestamp_sort") or 0.0) >= float(other.get("merge_timestamp_sort") or 0.0):
                other["selection_reason"] = "not_selected_older_duplicate"
            else:
                other["selection_reason"] = "not_selected_less_complete_duplicate"
        selected_rows.append(selected)
        if len(group) > 1:
            duplicate_rows.append(
                {
                    "experiment_name": stream.experiment_name,
                    "experiment_dir": stream.experiment_dir,
                    "stream_order": stream.stream_order,
                    "stream_filename": stream.stream_filename,
                    "stream_stem": stream.stream_stem,
                    "stream_path": stream.stream_path,
                    "duplicate_merge_count": len(group),
                    "selected_merge_results_dir": selected["merge_results_dir"],
                    "selected_merge_valid": selected.get("merge_valid"),
                    "selection_reason": selected.get("selection_reason"),
                    "all_merge_results_dirs_json": compact_json([row["merge_results_dir"] for row in sorted(group, key=lambda item: str(item.get("merge_results_dir")))]),
                }
            )
    logger.log("resolving duplicate runs: complete")
    return runs, selected_rows, duplicate_rows


def identify_reference(selected: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    references: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        if row.get("merge_valid"):
            by_experiment[str(row.get("experiment_dir"))].append(row)
    for exp_dir, rows in by_experiment.items():
        candidates: dict[str, dict[str, Any]] = {}
        methods: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            reference_value = value_as_join_text(row.get("reference_designation")).lower()
            reference_field = value_as_join_text(row.get("reference_designation_field")).lower()
            source = value_as_join_text(row.get("reference_designation_metadata_source")).lower()
            if ("reference" in reference_field or reference_field == "is_reference") and reference_value in {"true", "yes", "1", "ref", "reference"}:
                candidates[row["stream_path"]] = row
                methods[row["stream_path"]].add("explicit_metadata_reference_field")
            if "experiment_plan" in source and ("reference" in reference_field or reference_field == "is_reference") and reference_value in {"true", "yes", "1", "ref", "reference"}:
                candidates[row["stream_path"]] = row
                methods[row["stream_path"]].add("explicit_experiment_plan_reference_entry")
            if reference_value in {"reference", "ref"}:
                candidates[row["stream_path"]] = row
                methods[row["stream_path"]].add("explicit_metadata_reference_designation")
            stem = str(row.get("stream_stem", ""))
            if re.search(r"(^|_)ref($|_)", stem):
                candidates[row["stream_path"]] = row
                methods[row["stream_path"]].add("unambiguous_ref_stream_name")
        if len(candidates) == 1:
            stream_path, row = next(iter(candidates.items()))
            row["reference_identification_method"] = ";".join(sorted(methods[stream_path]))
            references[exp_dir] = row
        elif len(candidates) > 1:
            first = rows[0]
            warnings.append(
                make_warning(
                    "reference",
                    str(first.get("experiment_name", "")),
                    exp_dir,
                    "",
                    "",
                    "warning",
                    f"Multiple reference candidates found; no experiment-local reference selected: {sorted(candidates)}",
                    exp_dir,
                )
            )
    return references, warnings


def add_global_deltas(runs: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> None:
    reference_snapshots = {
        exp_dir: {
            "stream_path": reference.get("stream_path", ""),
            "stream_stem": reference.get("stream_stem", ""),
            "reference_identification_method": reference.get("reference_identification_method", ""),
            "metrics": {metric: reference.get(metric) for metric in ["cc12", "rsplit", "redundancy", "snr", "completeness"]},
        }
        for exp_dir, reference in references.items()
    }
    for row in runs:
        row["reference_stream_path"] = ""
        row["reference_stream_stem"] = ""
        row["reference_identification_method"] = ""
        for metric in ["cc12", "rsplit", "redundancy", "snr", "completeness"]:
            row[f"delta_{metric}"] = np.nan
        if not row.get("selected") or not row.get("merge_valid"):
            continue
        reference = reference_snapshots.get(str(row.get("experiment_dir")))
        if not reference:
            continue
        row["reference_stream_path"] = reference["stream_path"]
        row["reference_stream_stem"] = reference["stream_stem"]
        row["reference_identification_method"] = reference.get("reference_identification_method", "")
        for metric in ["cc12", "rsplit", "redundancy", "snr", "completeness"]:
            value = to_float(row.get(metric))
            ref_value = to_float(reference["metrics"].get(metric))
            if value is not None and ref_value is not None:
                row[f"delta_{metric}"] = value - ref_value


def selected_shell_rows(shell_rows: list[dict[str, Any]], selected_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_dirs = {row["merge_results_dir"]: row for row in selected_runs}
    out: list[dict[str, Any]] = []
    for shell in shell_rows:
        selected = selected_dirs.get(shell.get("merge_results_dir"))
        if not selected or not selected.get("merge_valid"):
            continue
        enriched = dict(shell)
        enriched.update(
            {
                "score_variant_name": selected.get("variant_name") or selected.get("score_name") or selected.get("stream_stem"),
                "score_name": selected.get("score_name", ""),
                "family": selected.get("family", ""),
                "formula": selected.get("formula", ""),
                "selected": True,
            }
        )
        out.append(enriched)
    return out


def validate_shell_schemes(selected_runs: list[dict[str, Any]], logger: Logger) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logger.log("validating shell schemes: start")
    by_scheme: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_runs:
        if row.get("merge_valid") and row.get("shell_scheme_id"):
            by_scheme[(row["experiment_dir"], row["shell_scheme_id"])].append(row)
    scheme_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    schemes_by_experiment: dict[str, set[str]] = defaultdict(set)
    for (exp_dir, scheme_id), rows in sorted(by_scheme.items()):
        schemes_by_experiment[exp_dir].add(scheme_id)
        first = rows[0]
        scheme_rows.append(
            {
                "experiment_name": first.get("experiment_name", ""),
                "experiment_dir": exp_dir,
                "shell_scheme_id": scheme_id,
                "number_of_shells": first.get("number_of_shells"),
                "selected_merge_count": len(rows),
                "stream_stems_json": compact_json([row.get("stream_stem") for row in rows]),
                "shell_boundary_json": first.get("shell_boundary_json", "[]"),
                "scheme_consistency_within_experiment": "",
            }
        )
    for exp_dir, schemes in schemes_by_experiment.items():
        status = "consistent" if len(schemes) <= 1 else "mismatch"
        for row in scheme_rows:
            if row["experiment_dir"] == exp_dir:
                row["scheme_consistency_within_experiment"] = status
        if len(schemes) > 1:
            first = next(row for row in selected_runs if row.get("experiment_dir") == exp_dir)
            warnings.append(make_warning("shell_scheme", first.get("experiment_name", ""), exp_dir, "", "", "warning", f"Selected valid merges use {len(schemes)} shell schemes; shellwise deltas only compare matching schemes", exp_dir))
    logger.log("validating shell schemes: complete")
    return scheme_rows, warnings


def calculate_shell_deltas(
    shell_rows: list[dict[str, Any]],
    selected_runs: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    logger: Logger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    logger.log("calculating reference differences: start")
    rows_by_dir = {row["merge_results_dir"]: row for row in selected_runs}
    shells_by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shell in shell_rows:
        shells_by_dir[str(shell.get("merge_results_dir"))].append(shell)
    deltas: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for row in selected_runs:
        if not row.get("merge_valid"):
            continue
        reference = references.get(str(row.get("experiment_dir")))
        if not reference:
            continue
        if row.get("shell_scheme_id") != reference.get("shell_scheme_id"):
            warnings.append(make_warning("reference_delta", row.get("experiment_name", ""), row.get("experiment_dir", ""), row.get("stream_path", ""), row.get("merge_results_dir", ""), "warning", "Skipped shell deltas because candidate and reference shell_scheme_id differ", row.get("merge_results_dir", "")))
            continue
        ref_shells = {boundary_key(shell): shell for shell in shells_by_dir.get(reference["merge_results_dir"], [])}
        for shell in shells_by_dir.get(row["merge_results_dir"], []):
            key = boundary_key(shell)
            ref_shell = ref_shells.get(key)
            if ref_shell is None:
                warnings.append(make_warning("reference_delta", row.get("experiment_name", ""), row.get("experiment_dir", ""), row.get("stream_path", ""), row.get("merge_results_dir", ""), "warning", "Reference shell missing matching explicit boundaries", row.get("merge_results_dir", "")))
                continue
            out = {
                "experiment_name": row.get("experiment_name"),
                "experiment_dir": row.get("experiment_dir"),
                "stream_stem": row.get("stream_stem"),
                "score_variant_name": row.get("variant_name") or row.get("score_name") or row.get("stream_stem"),
                "merge_results_dir": row.get("merge_results_dir"),
                "reference_stream_stem": reference.get("stream_stem"),
                "reference_merge_results_dir": reference.get("merge_results_dir"),
                "reference_identification_method": reference.get("reference_identification_method", ""),
                "shell_scheme_id": row.get("shell_scheme_id"),
                "shell_index": shell.get("shell_index"),
                "shell_lower_resolution_A": shell.get("shell_lower_resolution_A"),
                "shell_upper_resolution_A": shell.get("shell_upper_resolution_A"),
                "shell_center_resolution_A": shell.get("shell_center_resolution_A"),
                "reciprocal_resolution_1_per_nm": shell.get("reciprocal_resolution_1_per_nm"),
            }
            for metric, delta_name in [
                ("cc12", "delta_shell_cc12"),
                ("rsplit", "delta_shell_rsplit"),
                ("redundancy", "delta_shell_redundancy"),
                ("snr", "delta_shell_snr"),
                ("completeness", "delta_shell_completeness"),
                ("check_nref", "delta_shell_nref"),
            ]:
                value = to_float(shell.get(metric))
                ref_value = to_float(ref_shell.get(metric))
                out[delta_name] = value - ref_value if value is not None and ref_value is not None else np.nan
            deltas.append(out)
    logger.log("calculating reference differences: complete")
    return deltas, warnings


def summarize_variant_shell_deltas(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not deltas:
        return []
    table = pd.DataFrame(deltas)
    rows: list[dict[str, Any]] = []
    group_cols = ["experiment_name", "experiment_dir", "stream_stem", "score_variant_name", "merge_results_dir", "reference_stream_stem", "shell_scheme_id"]
    for keys, group in table.groupby(group_cols, dropna=False, sort=False):
        cc = pd.to_numeric(group["delta_shell_cc12"], errors="coerce").dropna()
        rs = pd.to_numeric(group["delta_shell_rsplit"], errors="coerce").dropna()
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "comparable_shell_count": int(len(group)),
                "shells_with_improved_cc12": int((cc > 0).sum()),
                "shells_with_worsened_cc12": int((cc < 0).sum()),
                "shells_with_improved_rsplit": int((rs < 0).sum()),
                "shells_with_worsened_rsplit": int((rs > 0).sum()),
                "median_shellwise_cc12_difference": float(cc.median()) if len(cc) else np.nan,
                "median_shellwise_rsplit_difference": float(rs.median()) if len(rs) else np.nan,
                "min_shellwise_cc12_difference": float(cc.min()) if len(cc) else np.nan,
                "max_shellwise_cc12_difference": float(cc.max()) if len(cc) else np.nan,
                "min_shellwise_rsplit_difference": float(rs.min()) if len(rs) else np.nan,
                "max_shellwise_rsplit_difference": float(rs.max()) if len(rs) else np.nan,
            }
        )
        rows.append(row)
    return rows


def rank_selected(selected_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selected_runs if row.get("merge_valid") and to_float(row.get("cc12")) is not None]
    ranked_rows: list[dict[str, Any]] = []

    def sort_key(row: dict[str, Any]) -> tuple[float, float, int, int]:
        cc = to_float(row.get("cc12"))
        rs = to_float(row.get("rsplit"))
        return (-(cc if cc is not None else -np.inf), rs if rs is not None else np.inf, int(row.get("experiment_order") or 0), int(row.get("stream_order") or 0))

    for scope, group in [("global", rows)]:
        for rank, row in enumerate(sorted(group, key=sort_key), start=1):
            ranked_rows.append(ranking_row(scope, rank, row))
    by_exp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_exp[str(row.get("experiment_dir"))].append(row)
    for exp_dir, group in by_exp.items():
        for rank, row in enumerate(sorted(group, key=sort_key), start=1):
            ranked_rows.append(ranking_row("experiment", rank, row, exp_dir))
    return ranked_rows


def ranking_row(scope: str, rank: int, row: dict[str, Any], scope_experiment_dir: str = "") -> dict[str, Any]:
    return {
        "ranking_scope": scope,
        "scope_experiment_dir": scope_experiment_dir,
        "rank": rank,
        "experiment_order": row.get("experiment_order"),
        "stream_order": row.get("stream_order"),
        "experiment_name": row.get("experiment_name"),
        "experiment_dir": row.get("experiment_dir"),
        "stream_stem": row.get("stream_stem"),
        "score_variant_name": row.get("variant_name") or row.get("score_name") or row.get("stream_stem"),
        "merge_results_dir": row.get("merge_results_dir"),
        "cc12": row.get("cc12"),
        "rsplit": row.get("rsplit"),
        "completeness": row.get("completeness"),
        "redundancy": row.get("redundancy"),
        "snr": row.get("snr"),
        "note": "Ranking uses CC1/2 descending, then Rsplit ascending; tiny numerical differences are not interpreted as significant.",
    }


def dataframe_from_rows(rows: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    if rows:
        table = pd.DataFrame(rows)
    else:
        table = pd.DataFrame(columns=columns or [])
    if columns:
        for column in columns:
            if column not in table.columns:
                table[column] = np.nan
        extra = [column for column in table.columns if column not in columns]
        table = table.loc[:, [*columns, *extra]]
    for column in table.columns:
        if column.endswith("_source_priority") or column == "merge_timestamp_sort":
            table = table.drop(columns=[column])
    return table


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    table = dataframe_from_rows(rows, columns)
    table.to_csv(path, index=False)
    return table


def make_missing_merges(coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in coverage_rows if row.get("experiment_exists") is True and row.get("stream_path") and int(row.get("merge_run_count") or 0) == 0]


def build_report(
    out_dir: Path,
    validation: dict[str, Any],
    rankings: list[dict[str, Any]],
    selected_runs: list[dict[str, Any]],
    shell_summaries: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    output_paths: dict[str, Path],
) -> str:
    top_cc = sorted([row for row in selected_runs if row.get("merge_valid") and to_float(row.get("cc12")) is not None], key=lambda row: -(to_float(row.get("cc12")) or -np.inf))[:10]
    top_rs = sorted([row for row in selected_runs if row.get("merge_valid") and to_float(row.get("rsplit")) is not None], key=lambda row: (to_float(row.get("rsplit")) or np.inf))[:10]
    lines = [
        "# OriDyn V5 Merge Results Summary",
        "",
        "## Coverage",
        f"- Configured experiment directories: {validation['counts']['configured_experiment_directories']}",
        f"- Present/missing experiment directories: {validation['counts']['present_experiment_directories']} / {validation['counts']['missing_experiment_directories']}",
        f"- Discovered streams: {validation['counts']['discovered_streams']}",
        f"- Streams with successful selected merges: {validation['counts']['streams_with_successful_selected_merges']}",
        f"- Streams without merges: {validation['counts']['streams_without_merges']}",
        f"- Streams with duplicate merge runs: {validation['counts']['streams_with_duplicate_merge_runs']}",
        f"- Selected valid merges: {validation['counts']['selected_valid_merges']}",
        f"- Parsing failures/errors: {validation['counts']['parse_errors']}",
        f"- Shell-scheme mismatch experiments: {validation['counts']['shell_scheme_mismatch_experiments']}",
        "",
        "## Top Global Variants by CC1/2",
    ]
    for row in top_cc:
        lines.append(f"- {row.get('experiment_name')} / {row.get('stream_stem')}: CC1/2={row.get('cc12')}, Rsplit={row.get('rsplit')}")
    if not top_cc:
        lines.append("- None.")
    lines.extend(["", "## Top Global Variants by Rsplit"])
    for row in top_rs:
        lines.append(f"- {row.get('experiment_name')} / {row.get('stream_stem')}: Rsplit={row.get('rsplit')}, CC1/2={row.get('cc12')}")
    if not top_rs:
        lines.append("- None.")
    lines.extend(["", "## Top Variants Within Each Experiment"])
    experiment_rankings = [row for row in rankings if row.get("ranking_scope") == "experiment" and int(row.get("rank") or 0) <= 3]
    if experiment_rankings:
        current = None
        for row in experiment_rankings:
            if row.get("experiment_name") != current:
                current = row.get("experiment_name")
                lines.append(f"- {current}")
            lines.append(f"  - rank {row.get('rank')}: {row.get('stream_stem')} CC1/2={row.get('cc12')} Rsplit={row.get('rsplit')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Reference Comparisons"])
    if references:
        for ref in references.values():
            lines.append(f"- {ref.get('experiment_name')}: reference {ref.get('stream_stem')} ({ref.get('reference_identification_method')})")
    else:
        lines.append("- No explicit or unambiguous experiment-local references were identified.")
    if shell_summaries:
        lines.append("")
        lines.append("## Shellwise Improvement Counts")
        for row in shell_summaries[:50]:
            lines.append(
                f"- {row.get('experiment_name')} / {row.get('stream_stem')}: "
                f"CC1/2 improved/worsened shells={row.get('shells_with_improved_cc12')}/{row.get('shells_with_worsened_cc12')}; "
                f"Rsplit improved/worsened shells={row.get('shells_with_improved_rsplit')}/{row.get('shells_with_worsened_rsplit')}"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- Factual extracted results come from existing machine-readable QC files and raw logs where available.",
            "- Rounded metadata headline values are used only when the more precise QC/log value is absent.",
            "- Missing data, parser warnings, inferred merge-directory stem associations, and ambiguous metadata joins are reported in the output tables.",
            "- Shellwise deltas are calculated only when candidate and reference shell_scheme_id match.",
            "",
            "## Output Tables",
        ]
    )
    for name, path in output_paths.items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def choose_plot_variants(rows: list[dict[str, Any]], reference: dict[str, Any] | None = None, limit: int = 12) -> list[str]:
    selected: list[str] = []
    if reference:
        selected.append(reference.get("merge_results_dir", ""))
    by_cc = sorted([row for row in rows if to_float(row.get("cc12")) is not None], key=lambda row: -(to_float(row.get("cc12")) or -np.inf))[:5]
    by_rs = sorted([row for row in rows if to_float(row.get("rsplit")) is not None], key=lambda row: to_float(row.get("rsplit")) or np.inf)[:5]
    for row in [*by_cc, *by_rs]:
        selected.append(row.get("merge_results_dir", ""))
    deduped = []
    for item in selected:
        if item and item not in deduped:
            deduped.append(item)
    return deduped[:limit]


def generate_plots(
    out_dir: Path,
    selected_runs: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
    shell_deltas: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    logger: Logger,
) -> list[dict[str, Any]]:
    logger.log("generating plots: start")
    plotted: list[dict[str, Any]] = []
    valid = [row for row in selected_runs if row.get("merge_valid") and to_float(row.get("cc12")) is not None and to_float(row.get("rsplit")) is not None]
    if len(valid) < 2:
        logger.log("generating plots: skipped; fewer than two valid selected merges with CC1/2 and Rsplit")
        return plotted
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.log(f"generating plots: skipped; matplotlib unavailable: {exc}")
        return plotted

    global_path = out_dir / "global_cc12_vs_rsplit.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([to_float(row.get("rsplit")) for row in valid], [to_float(row.get("cc12")) for row in valid], s=24, alpha=0.75)
    ax.set_xlabel("Rsplit (%)")
    ax.set_ylabel("CC1/2")
    ax.set_title("Selected valid merges")
    fig.tight_layout()
    fig.savefig(global_path, dpi=150)
    plt.close(fig)
    plotted.append({"plot": str(global_path), "plot_type": "global_cc12_vs_rsplit", "variants_json": compact_json([row.get("stream_stem") for row in valid])})

    shells_by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shell in shell_rows:
        shells_by_dir[str(shell.get("merge_results_dir"))].append(shell)
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_experiment[str(row.get("experiment_dir"))].append(row)
    for exp_dir, rows in by_experiment.items():
        if len(rows) < 2:
            continue
        reference = references.get(exp_dir)
        dirs = choose_plot_variants(rows, reference)
        if len(dirs) < 2:
            continue
        exp_name = rows[0].get("experiment_name", Path(exp_dir).name)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(exp_name))
        for metric, ylabel, filename in [
            ("cc12", "CC1/2", f"{safe_name}_cc12_vs_resolution.png"),
            ("rsplit", "Rsplit (%)", f"{safe_name}_rsplit_vs_resolution.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted_variants: list[str] = []
            for merge_dir in dirs:
                run = next((row for row in rows if row.get("merge_results_dir") == merge_dir), None)
                if run is None:
                    continue
                shells = sorted(shells_by_dir.get(merge_dir, []), key=lambda item: int(item.get("shell_index") or 0))
                xs = [to_float(shell.get("shell_center_resolution_A")) for shell in shells]
                ys = [to_float(shell.get(metric)) for shell in shells]
                if sum(value is not None for value in xs) < 2 or sum(value is not None for value in ys) < 2:
                    continue
                ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=str(run.get("stream_stem")))
                plotted_variants.append(str(run.get("stream_stem")))
            if plotted_variants:
                ax.set_xlabel("Resolution (A)")
                ax.set_ylabel(ylabel)
                ax.set_title(str(exp_name))
                ax.invert_xaxis()
                ax.legend(fontsize=7)
                fig.tight_layout()
                path = out_dir / filename
                fig.savefig(path, dpi=150)
                plotted.append({"plot": str(path), "plot_type": metric, "experiment_name": exp_name, "variants_json": compact_json(plotted_variants)})
            plt.close(fig)

    if shell_deltas:
        delta_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in shell_deltas:
            delta_by_experiment[str(row.get("experiment_dir"))].append(row)
        for exp_dir, rows in delta_by_experiment.items():
            if len({row.get("merge_results_dir") for row in rows}) < 2:
                continue
            exp_name = rows[0].get("experiment_name", Path(exp_dir).name)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(exp_name))
            for metric, ylabel, filename in [
                ("delta_shell_cc12", "Delta shell CC1/2", f"{safe_name}_delta_cc12_vs_resolution.png"),
                ("delta_shell_rsplit", "Delta shell Rsplit (%)", f"{safe_name}_delta_rsplit_vs_resolution.png"),
            ]:
                fig, ax = plt.subplots(figsize=(8, 5))
                plotted_variants = []
                for merge_dir, group in pd.DataFrame(rows).groupby("merge_results_dir", sort=False):
                    group = group.sort_values("shell_index")
                    xs = [to_float(value) for value in group["shell_center_resolution_A"]]
                    ys = [to_float(value) for value in group[metric]]
                    if sum(value is not None for value in xs) < 2 or sum(value is not None for value in ys) < 2:
                        continue
                    label = str(group["stream_stem"].iloc[0])
                    ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=label)
                    plotted_variants.append(label)
                if plotted_variants:
                    ax.axhline(0.0, color="black", linewidth=0.8)
                    ax.set_xlabel("Resolution (A)")
                    ax.set_ylabel(ylabel)
                    ax.set_title(str(exp_name))
                    ax.invert_xaxis()
                    ax.legend(fontsize=7)
                    fig.tight_layout()
                    path = out_dir / filename
                    fig.savefig(path, dpi=150)
                    plotted.append({"plot": str(path), "plot_type": metric, "experiment_name": exp_name, "variants_json": compact_json(plotted_variants)})
                plt.close(fig)
    logger.log("generating plots: complete")
    return plotted


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.replace("\n", " "), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import matplotlib

        versions["matplotlib"] = matplotlib.__version__
    except Exception:  # noqa: BLE001
        versions["matplotlib"] = "unavailable"
    return versions


def git_commit(script_path: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(script_path.parent.parent), "rev-parse", "HEAD"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def validate_outputs(
    runs: list[dict[str, Any]],
    selected_runs: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    parse_warnings: list[dict[str, Any]],
    shell_scheme_rows: list[dict[str, Any]],
    output_tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    errors: list[str] = []
    selected_counter = Counter((row.get("experiment_dir"), row.get("stream_path")) for row in selected_runs)
    duplicate_selected = [key for key, count in selected_counter.items() if count > 1]
    if duplicate_selected:
        errors.append(f"Selected stream has more than one selected merge: {duplicate_selected[:5]}")
    for row in selected_runs:
        if row.get("merge_results_dir") and not Path(str(row["merge_results_dir"])).exists():
            errors.append(f"Selected merge path does not exist: {row['merge_results_dir']}")
    for row in runs:
        for column in GLOBAL_NUMERIC_COLUMNS:
            value = row.get(column)
            if value is None or value == "" or pd.isna(value):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"Non-numeric value in {column}: {value!r}")
                continue
            if not math.isfinite(number):
                errors.append(f"Non-finite value in {column}: {value!r}")
    for shell in shell_rows:
        if not shell.get("experiment_name") or not shell.get("merge_results_dir"):
            errors.append("Shell row lacks experiment_name or merge_results_dir")
        lower = to_float(shell.get("shell_lower_resolution_A"))
        upper = to_float(shell.get("shell_upper_resolution_A"))
        if lower is not None and upper is not None and lower < upper:
            errors.append(f"Shell lower-resolution boundary is smaller than upper boundary for {shell.get('merge_results_dir')} shell {shell.get('shell_index')}")
    for name, table in output_tables.items():
        if len(table) and "experiment_name" in table.columns and table["experiment_name"].isna().all():
            errors.append(f"{name} rows do not retain experiment_name")
    counts = {
        "configured_experiment_directories": len({row.get("experiment_dir") for row in coverage_rows}),
        "present_experiment_directories": len({row.get("experiment_dir") for row in coverage_rows if row.get("experiment_exists") is True}),
        "missing_experiment_directories": len({row.get("experiment_dir") for row in coverage_rows if row.get("experiment_exists") is False}),
        "discovered_streams": sum(1 for row in coverage_rows if row.get("stream_path")),
        "discovered_merge_runs": len(runs),
        "streams_with_successful_selected_merges": sum(1 for row in selected_runs if row.get("merge_valid")),
        "streams_without_merges": sum(1 for row in coverage_rows if row.get("stream_path") and int(row.get("merge_run_count") or 0) == 0),
        "streams_with_duplicate_merge_runs": len(duplicate_rows),
        "selected_merge_runs": len(selected_runs),
        "selected_valid_merges": sum(1 for row in selected_runs if row.get("merge_valid")),
        "parse_warnings": sum(1 for row in parse_warnings if row.get("severity") == "warning"),
        "parse_errors": sum(1 for row in parse_warnings if row.get("severity") == "error"),
        "shell_rows": len(shell_rows),
        "shell_scheme_mismatch_experiments": len({row.get("experiment_dir") for row in shell_scheme_rows if row.get("scheme_consistency_within_experiment") == "mismatch"}),
    }
    table_counts = {name: int(len(table)) for name, table in output_tables.items()}
    return {"status": "ok" if not errors else "validation_errors", "errors": errors, "counts": counts, "table_counts": table_counts}


def main() -> int:
    args = parse_args()
    logger = Logger()
    script_path = Path(__file__).resolve()
    fatal = False
    source_dirs: list[str] = []
    out_dir = args.out_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"Could not create --out-dir {out_dir}: {exc}") from exc

    try:
        started = time.monotonic()
        logger.log("loading configuration: start")
        source_dirs = load_config(args.config)
        logger.progress("loading configuration", 1, 1, started)
        logger.log("loading configuration: complete")

        logger.log("validating experiment directories and discovering streams/merge directories: start")
        started = time.monotonic()
        all_streams: list[StreamRecord] = []
        run_candidates: list[RunCandidate] = []
        coverage_rows: list[dict[str, Any]] = []
        parse_warnings: list[dict[str, Any]] = []
        for order, exp_dir in enumerate(source_dirs, start=1):
            streams, candidates, coverage, warnings = discover_experiment(order, exp_dir)
            all_streams.extend(streams)
            run_candidates.extend(candidates)
            coverage_rows.extend(coverage)
            parse_warnings.extend(warnings)
            logger.progress("discovering streams and merge directories", order, len(source_dirs), started)
        logger.log("validating experiment directories and discovering streams/merge directories: complete")

        logger.log("parsing global statistics: start")
        logger.log("parsing shellwise statistics: start")
        parse_results = bounded_map(run_candidates, args.workers, parse_run, logger, "parsing merge result directories")
        runs: list[dict[str, Any]] = []
        all_shell_rows: list[dict[str, Any]] = []
        raw_shell_schemes: list[dict[str, Any]] = []
        for run_row, shell_rows, shell_scheme, warnings in parse_results:
            runs.append(run_row)
            all_shell_rows.extend(shell_rows)
            raw_shell_schemes.append(shell_scheme)
            parse_warnings.extend(warnings)
        runs = sorted(runs, key=lambda row: (int(row.get("experiment_order") or 0), int(row.get("stream_order") or 10**9), int(row.get("merge_order") or 0), str(row.get("merge_results_dir"))))
        logger.log("parsing global statistics: complete")
        logger.log("parsing shellwise statistics: complete")

        runs, ambiguous_metadata, metadata_warnings = join_metadata(runs, all_streams, logger)
        parse_warnings.extend(metadata_warnings)

        runs, selected_runs, duplicate_rows = select_duplicate_runs(runs, all_streams, logger)
        references, reference_warnings = identify_reference(selected_runs)
        parse_warnings.extend(reference_warnings)
        add_global_deltas(runs, references)
        selected_runs = [row for row in runs if row.get("selected")]

        selected_shell = selected_shell_rows(all_shell_rows, selected_runs)
        shell_scheme_rows, shell_scheme_warnings = validate_shell_schemes(selected_runs, logger)
        parse_warnings.extend(shell_scheme_warnings)
        shell_deltas, delta_warnings = calculate_shell_deltas(selected_shell, selected_runs, references, logger)
        parse_warnings.extend(delta_warnings)
        variant_shell_summary = summarize_variant_shell_deltas(shell_deltas)
        rankings = rank_selected(selected_runs)
        missing_rows = make_missing_merges(coverage_rows)

        output_paths = {
            "parameters.json": out_dir / "parameters.json",
            "experiment_coverage.csv": out_dir / "experiment_coverage.csv",
            "all_merge_runs.csv": out_dir / "all_merge_runs.csv",
            "selected_merge_runs.csv": out_dir / "selected_merge_runs.csv",
            "missing_merges.csv": out_dir / "missing_merges.csv",
            "duplicate_merges.csv": out_dir / "duplicate_merges.csv",
            "ambiguous_metadata_joins.csv": out_dir / "ambiguous_metadata_joins.csv",
            "global_results.csv": out_dir / "global_results.csv",
            "global_rankings.csv": out_dir / "global_rankings.csv",
            "shell_results.csv": out_dir / "shell_results.csv",
            "shell_schemes.csv": out_dir / "shell_schemes.csv",
            "shell_deltas_vs_reference.csv": out_dir / "shell_deltas_vs_reference.csv",
            "variant_shell_summary.csv": out_dir / "variant_shell_summary.csv",
            "parse_warnings.csv": out_dir / "parse_warnings.csv",
            "validation.json": out_dir / "validation.json",
            "report.md": out_dir / "report.md",
            "run.log": out_dir / "run.log",
        }

        plotted = generate_plots(out_dir, selected_runs, selected_shell, shell_deltas, references, logger)

        logger.log("writing outputs: start")
        table_outputs: dict[str, pd.DataFrame] = {}
        table_outputs["experiment_coverage.csv"] = write_csv(output_paths["experiment_coverage.csv"], coverage_rows)
        table_outputs["all_merge_runs.csv"] = write_csv(output_paths["all_merge_runs.csv"], runs)
        table_outputs["selected_merge_runs.csv"] = write_csv(output_paths["selected_merge_runs.csv"], selected_runs)
        table_outputs["missing_merges.csv"] = write_csv(output_paths["missing_merges.csv"], missing_rows)
        table_outputs["duplicate_merges.csv"] = write_csv(output_paths["duplicate_merges.csv"], duplicate_rows)
        table_outputs["ambiguous_metadata_joins.csv"] = write_csv(output_paths["ambiguous_metadata_joins.csv"], ambiguous_metadata)
        table_outputs["global_results.csv"] = write_csv(output_paths["global_results.csv"], runs)
        table_outputs["global_rankings.csv"] = write_csv(output_paths["global_rankings.csv"], rankings)
        table_outputs["shell_results.csv"] = write_csv(output_paths["shell_results.csv"], selected_shell)
        table_outputs["shell_schemes.csv"] = write_csv(output_paths["shell_schemes.csv"], shell_scheme_rows)
        table_outputs["shell_deltas_vs_reference.csv"] = write_csv(output_paths["shell_deltas_vs_reference.csv"], shell_deltas)
        table_outputs["variant_shell_summary.csv"] = write_csv(output_paths["variant_shell_summary.csv"], variant_shell_summary)
        table_outputs["parse_warnings.csv"] = write_csv(output_paths["parse_warnings.csv"], parse_warnings)

        validation = validate_outputs(runs, selected_runs, selected_shell, coverage_rows, duplicate_rows, parse_warnings, shell_scheme_rows, table_outputs)
        parameters = {
            "generation_date_local": now_iso_local(),
            "source_script_path": str(script_path),
            "git_commit": git_commit(script_path),
            "python_version": sys.version.replace("\n", " "),
            "package_versions": package_versions(),
            "config_path": str(args.config),
            "configured_source_directories": source_dirs,
            "output_directory": str(out_dir),
            "worker_count": args.workers,
            "parser_assumptions": [
                "Top-level *.stream files define the stream universe for coverage.",
                "Merge directories are top-level directories matching *_partialator_results_*.",
                "Merge-to-stream association uses parameters.json stream_file, then metadata_and_outputs.txt STREAM, then merge-directory stem fallback.",
                "Global CC1/2 is parsed from compare_cc12.log and never averaged from shell values.",
                "Global Rsplit is parsed from compare_rsplit.log and never averaged from shell values.",
                "Completeness, redundancy, SNR, observations, and reflection counts prefer check_hkl_completeness.log.",
                "metadata_and_outputs.txt headline values are fallback values only.",
                "Shell joins use explicit normalized reciprocal-resolution shell boundaries; row-order joins are not used without boundary validation.",
            ],
            "reference_selection_rules": [
                "Use explicit metadata reference fields or designations when unambiguous.",
                "Use an unambiguous stream stem containing a standalone ref token.",
                "Do not infer a reference from best metric values.",
            ],
            "duplicate_selection_rules": [
                "Prefer parseable runs with all required outputs present.",
                "Among equally complete runs, select the latest timestamp.",
                "Retain all non-selected duplicate runs in all_merge_runs.csv.",
                "Never select a newer failed/incomplete run over an older complete parseable run.",
            ],
            "counts": validation["counts"],
            "plots": plotted,
        }
        output_paths["parameters.json"].write_text(json.dumps(parameters, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        output_paths["validation.json"].write_text(json.dumps(validation, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        report = build_report(out_dir, validation, rankings, selected_runs, variant_shell_summary, references, output_paths)
        output_paths["report.md"].write_text(report, encoding="utf-8")
        logger.log("writing outputs: complete")

        if validation["counts"]["selected_valid_merges"] == 0:
            logger.log("fatal: no valid selected merge result could be parsed")
            fatal = True
        if validation["errors"]:
            logger.log(f"validation completed with {len(validation['errors'])} error(s); see validation.json")
        else:
            logger.log("validation completed without structural errors")
    finally:
        logger.ensure_run_log(out_dir)

    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
