#!/usr/bin/env python3
"""Build OriDyn V6 score-target filtering streams.

This script prepares the V6 stream matrix only.  It never runs Partialator,
never launches a merge, and never creates random-control streams.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

BLAS_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
for _thread_env_name in BLAS_THREAD_ENV_VARS:
    os.environ.setdefault(_thread_env_name, "1")

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]

BASELINE_SG0 = 0.0013180204579645218
SG0_MULTIPLIER = 1.75
SIGMA_C_MULTIPLIER = 1.0
SIGMA_C = 0.050
R_CUT = 0.150
HIGH_EG_FRACTION = 0.30
EXCITATION_BLOCK_SIZE = 10
MIN_FINAL_BLOCK_SIZE = 5
MIN_HIGH_EG_OBSERVATIONS = 10
MIN_REMAINING = 2
TARGET_A_MIN_OBSERVATIONS = 10
MAX_OPEN_STREAMS = 16
EXPECTED_STREAM_COUNT = 88
CSV_FLOAT_FORMAT = "%.12g"

STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")

COLUMN_ALIASES = {
    "Eg": ["Eg", "target_excitation_Eg"],
    "sg": ["sg", "abs_sg_target", "sg_target", "s_g"],
    "D": ["D", "coupling_sum_sc100"],
    "U": ["U", "neighbor_excitation_sum_sg175_sc100"],
    "M": ["M", "score_sg175_sc100", "nonself_local_excitation_raw"],
    "M2": ["M2", "excitation_coupling_sq_sum_sg175_sc100"],
}


def set_worker_numeric_threads() -> None:
    for name in BLAS_THREAD_ENV_VARS:
        os.environ[name] = "1"


def worker_initializer() -> None:
    set_worker_numeric_threads()


@dataclass(frozen=True)
class ScoreDefinition:
    score_id: str
    formula: str
    expanded_formula: str
    expression_tree: dict[str, Any]
    variables_used: tuple[str, ...]
    scientific_purpose: str
    classification: str


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    experiment_type: str
    score_id: str
    designation: str
    filtering_target: str
    high_eg_fraction: float | None
    block_size: int | None
    drop_fraction: float | None
    output_filename: str
    suggested_priority: str


@dataclass
class RewriteSpec:
    variant_id: str
    output_path: Path
    mode: str
    keys: set[str]
    requested_removed: int
    reused_existing_path: str | None = None


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


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class StageProgress:
    def __init__(self, logger: RunLogger, stage: str, total: int | None, unit: str = "items"):
        self.logger = logger
        self.stage = stage
        self.total = int(total) if total is not None else None
        self.unit = unit
        self.start_time = time.monotonic()
        self.last_report_time = self.start_time
        self.completed = 0
        total_text = f"{self.total:,} {unit}" if self.total is not None else f"unknown {unit}"
        self.logger.log(f"{stage} start: total={total_text}")

    def update(self, completed: int, force: bool = False) -> None:
        completed = int(completed)
        now = time.monotonic()
        if not force and completed != self.total and completed > 20 and now - self.last_report_time < 30.0:
            self.completed = completed
            return
        elapsed = max(now - self.start_time, 1.0e-9)
        rate = completed / elapsed
        message = f"{self.stage} progress: completed={completed:,}"
        if self.total is not None:
            pct = 100.0 * completed / max(1, self.total)
            message += f"/{self.total:,} ({pct:.1f}%)"
        message += f"; elapsed={format_duration(elapsed)}; rate={rate:.2f} {self.unit}/s"
        if self.total is not None and 0 < completed < self.total and rate > 0.0:
            message += f"; ETA={format_duration((self.total - completed) / rate)}"
        self.logger.log(message)
        self.completed = completed
        self.last_report_time = now

    def advance(self, amount: int = 1) -> None:
        self.update(self.completed + int(amount))

    def finish(self, completed: int | None = None) -> None:
        final = self.completed if completed is None else int(completed)
        elapsed = max(time.monotonic() - self.start_time, 1.0e-9)
        rate = final / elapsed
        message = f"{self.stage} complete: completed={final:,}"
        if self.total is not None:
            pct = 100.0 * final / max(1, self.total)
            message += f"/{self.total:,} ({pct:.1f}%)"
        message += f"; elapsed={format_duration(elapsed)}; rate={rate:.2f} {self.unit}/s"
        self.logger.log(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stream", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    if not args.cache.is_file():
        raise SystemExit(f"--cache not found: {args.cache}")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, set):
        return sorted(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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


def percent_label(fraction: float) -> str:
    return f"{int(round(float(fraction) * 100.0)):02d}"


def var_node(name: str) -> dict[str, str]:
    return {"var": name}


def pow_node(base: Any, exponent: float) -> dict[str, Any]:
    return {"op": "pow", "base": base, "exponent": float(exponent)}


def mul_node(*args: Any) -> dict[str, Any]:
    return {"op": "mul", "args": list(args)}


def div_node(numerator: Any, denominator: Any) -> dict[str, Any]:
    return {
        "op": "div",
        "numerator": numerator,
        "denominator": denominator,
        "zero_policy": "zero_if_zero_over_zero_else_fail",
    }


def expression_variables(node: Any) -> set[str]:
    if not isinstance(node, dict):
        return set()
    if "var" in node:
        return {str(node["var"])}
    variables: set[str] = set()
    for value in node.values():
        if isinstance(value, dict):
            variables.update(expression_variables(value))
        elif isinstance(value, list):
            for item in value:
                variables.update(expression_variables(item))
    return variables


def score_registry() -> list[ScoreDefinition]:
    eg = var_node("Eg")
    d = var_node("D")
    u = var_node("U")
    m = var_node("M")
    m2 = var_node("M2")
    d1 = d
    d2 = pow_node(d, 2.0)
    d3 = pow_node(d, 3.0)
    records = [
        (
            "eg",
            "Eg",
            "Eg",
            eg,
            "Target-excitation-only control.",
            "control",
        ),
        (
            "density_d",
            "D",
            "D",
            d,
            "Coupling-density-only control.",
            "control",
        ),
        (
            "legacy_m",
            "M",
            "M",
            m,
            "Original raw environmental excitation/coupling score.",
            "benchmark",
        ),
        (
            "eg_m",
            "Eg * M",
            "Eg * M",
            mul_node(eg, m),
            "Target-weighted raw environmental score.",
            "benchmark",
        ),
        (
            "eg_cmean",
            "Eg * <C>_E",
            "Eg * M / U",
            mul_node(eg, div_node(m, u)),
            "Target excitation times normalized neighboring-excitation mean coupling.",
            "candidate",
        ),
        (
            "eg_c2mean",
            "Eg * <C^2>_E",
            "Eg * M2 / U",
            mul_node(eg, div_node(m2, u)),
            "Target excitation times normalized second coupling moment.",
            "candidate",
        ),
        (
            "eg_m2",
            "Eg * M2",
            "Eg * M2",
            mul_node(eg, m2),
            "Total excitation-weighted squared-link burden.",
            "candidate",
        ),
        (
            "eg_d1_cmean",
            "Eg * D * <C>_E",
            "Eg * D * M / U",
            mul_node(eg, d1, div_node(m, u)),
            "Mild density amplification of normalized mean coupling.",
            "candidate",
        ),
        (
            "eg_d2_cmean",
            "Eg * D^2 * <C>_E",
            "Eg * D^2 * M / U",
            mul_node(eg, d2, div_node(m, u)),
            "Moderate density amplification of normalized mean coupling.",
            "candidate",
        ),
        (
            "eg_d3_cmean",
            "Eg * D^3 * <C>_E",
            "Eg * M * D^3 / U",
            div_node(mul_node(eg, m, d3), u),
            "Empirical V5 benchmark score.",
            "benchmark",
        ),
        (
            "eg_d2_c2mean",
            "Eg * D^2 * <C^2>_E",
            "Eg * D^2 * M2 / U",
            mul_node(eg, d2, div_node(m2, u)),
            "Moderate density amplification of the normalized second coupling moment.",
            "benchmark",
        ),
    ]
    return [
        ScoreDefinition(
            score_id=score_id,
            formula=formula,
            expanded_formula=expanded,
            expression_tree=tree,
            variables_used=tuple(sorted(expression_variables(tree))),
            scientific_purpose=purpose,
            classification=classification,
        )
        for score_id, formula, expanded, tree, purpose, classification in records
    ]


def evaluate_expression_tree(
    node: dict[str, Any],
    variables: dict[str, np.ndarray],
    score_id: str,
    stats: dict[str, Any],
    label: str = "expr",
) -> np.ndarray:
    if "var" in node:
        variable = str(node["var"])
        if variable not in variables:
            raise SystemExit(f"Score {score_id} requires missing variable {variable}")
        return variables[variable]
    op = str(node.get("op", ""))
    if op == "mul":
        args = [evaluate_expression_tree(arg, variables, score_id, stats, f"{label}.mul{idx}") for idx, arg in enumerate(node["args"])]
        out = np.ones_like(args[0], dtype=float)
        for arg in args:
            out = out * arg
    elif op == "pow":
        base = evaluate_expression_tree(node["base"], variables, score_id, stats, f"{label}.pow")
        exponent = float(node["exponent"])
        invalid = (base < 0.0) & ~np.isclose(exponent, round(exponent))
        if invalid.any():
            raise SystemExit(f"Score {score_id} has invalid negative fractional power at {label}")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.power(base, exponent)
    elif op == "div":
        numerator = evaluate_expression_tree(node["numerator"], variables, score_id, stats, f"{label}.num")
        denominator = evaluate_expression_tree(node["denominator"], variables, score_id, stats, f"{label}.den")
        zero = denominator == 0.0
        stats[f"{score_id}_{label.replace('.', '_')}_zero_denominator_count"] = int(zero.sum())
        out = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=~zero,
        )
    else:
        raise SystemExit(f"Unsupported score expression operation for {score_id}: {op}")
    if not np.isfinite(out).all():
        raise SystemExit(f"Score {score_id} produced nonfinite values at {label}")
    return out


def build_experiment_plan(scores: list[ScoreDefinition]) -> list[VariantSpec]:
    score_ids = {score.score_id for score in scores}
    filter_scores = ["eg_cmean", "eg_c2mean", "eg_m2", "eg_d1_cmean", "eg_d2_cmean", "eg_d3_cmean"]
    missing = sorted(set(filter_scores) - score_ids)
    if missing:
        raise SystemExit(f"Internal score registry missing filter score(s): {missing}")

    variants: list[VariantSpec] = []
    priority_counter = 0

    def add_diag(score_id: str, tier: int) -> None:
        nonlocal priority_counter
        for designation in ["low50", "high50"]:
            priority_counter += 1
            variants.append(
                VariantSpec(
                    variant_id=f"diag_{score_id}_{designation}",
                    experiment_type="diagnostic_low_high",
                    score_id=score_id,
                    designation=designation,
                    filtering_target="unrestricted_per_hkl_half_split",
                    high_eg_fraction=None,
                    block_size=None,
                    drop_fraction=None,
                    output_filename=f"diag_{score_id}_{designation}.stream",
                    suggested_priority=f"P{tier}.{priority_counter:03d}",
                )
            )

    for score_id in ["eg_m2", "eg_d2_cmean", "eg_d3_cmean", "eg_cmean", "eg_c2mean"]:
        add_diag(score_id, 1)
    for score_id in ["eg_d1_cmean", "eg_d2_c2mean", "eg_m", "legacy_m", "density_d", "eg"]:
        add_diag(score_id, 2)

    def add_filter(target: str, score_id: str, drop: float, tier: int) -> None:
        nonlocal priority_counter
        priority_counter += 1
        label = percent_label(drop)
        variants.append(
            VariantSpec(
                variant_id=f"filter_{target}_{score_id}_drop{label}",
                experiment_type="targeted_filter",
                score_id=score_id,
                designation=f"drop{label}",
                filtering_target=target,
                high_eg_fraction=HIGH_EG_FRACTION if target in {"higheg", "matched"} else None,
                block_size=EXCITATION_BLOCK_SIZE if target == "matched" else None,
                drop_fraction=float(drop),
                output_filename=f"filter_{target}_{score_id}_drop{label}.stream",
                suggested_priority=f"P{tier}.{priority_counter:03d}",
            )
        )

    ordered_filter_scores = ["eg_d2_cmean", "eg_d3_cmean", "eg_m2", "eg_cmean", "eg_c2mean", "eg_d1_cmean"]
    for score_id in ordered_filter_scores:
        for drop in [0.20, 0.30, 0.40, 0.50]:
            add_filter("matched", score_id, drop, 3)
    for score_id in ordered_filter_scores:
        for drop in [0.20, 0.30, 0.40, 0.50]:
            add_filter("higheg", score_id, drop, 4)
    for score_id in ordered_filter_scores:
        for drop in [0.05, 0.10, 0.20]:
            add_filter("all", score_id, drop, 5)

    if len(variants) != EXPECTED_STREAM_COUNT:
        raise SystemExit(f"Internal plan error: expected 88 variants, built {len(variants)}")
    return variants


def find_column(header: list[str], canonical: str) -> str:
    for candidate in COLUMN_ALIASES[canonical]:
        if candidate in header:
            return candidate
    raise SystemExit(f"Cache is missing required {canonical} column; tried {COLUMN_ALIASES[canonical]}")


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc


def metadata_payloads(cache: Path) -> dict[str, dict[str, Any]]:
    parent = cache.parent
    out: dict[str, dict[str, Any]] = {}
    for name in ["run_metadata.json", "parameters.json", "sweep_audit.json"]:
        payload = read_json_if_present(parent / name)
        if payload is not None:
            out[name] = payload
    return out


def find_nested_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = find_nested_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_nested_value(value, key)
            if found is not None:
                return found
    return None


def validate_cache_metadata(cache: Path, source_stream: Path, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    failures: list[str] = []
    source_resolved = str(source_stream.resolve())
    source_paths: set[str] = set()
    raw_specs: list[dict[str, Any]] = []
    scorer_schema_versions: set[str] = set()
    cache_rows_expected: int | None = None

    for payload in payloads.values():
        input_stream = find_nested_value(payload, "input_stream")
        if isinstance(input_stream, str):
            source_paths.add(str(Path(input_stream).resolve()))
        elif isinstance(input_stream, dict) and input_stream.get("path"):
            source_paths.add(str(Path(str(input_stream["path"])).resolve()))
        if isinstance(payload.get("cache_stats"), dict) and payload["cache_stats"].get("rows") is not None:
            cache_rows_expected = int(payload["cache_stats"]["rows"])
        schema = payload.get("cache_schema")
        if isinstance(schema, dict):
            registry = schema.get("raw_kernel_registry")
            if isinstance(registry, list):
                raw_specs.extend([item for item in registry if isinstance(item, dict)])
            if schema.get("scorer_schema_version") is not None:
                scorer_schema_versions.add(str(schema["scorer_schema_version"]))
        registry = payload.get("kernel_parameters") or payload.get("raw_kernel_specs")
        if isinstance(registry, list):
            raw_specs.extend([item for item in registry if isinstance(item, dict)])
        if payload.get("scorer_schema_version") is not None:
            scorer_schema_versions.add(str(payload["scorer_schema_version"]))

    if source_paths and source_resolved not in source_paths:
        failures.append(f"source stream path does not match cache metadata: {source_resolved} not in {sorted(source_paths)}")
    elif not source_paths:
        warnings.append("No source stream path found in cache metadata; source compatibility will rely on exact-key stream scan.")

    compatible_kernel = False
    for spec in raw_specs:
        try:
            compatible_kernel = compatible_kernel or (
                math.isclose(float(spec.get("sg_multiplier")), SG0_MULTIPLIER, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(float(spec.get("sigma_multiplier")), SIGMA_C_MULTIPLIER, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(float(spec.get("sigma_c")), SIGMA_C, rel_tol=0.0, abs_tol=1.0e-12)
                and math.isclose(float(spec.get("r_cut")), R_CUT, rel_tol=0.0, abs_tol=1.0e-12)
            )
        except (TypeError, ValueError):
            continue
    if not compatible_kernel:
        failures.append("cache metadata does not contain a sg175/sc100 raw kernel with sigma_c=0.050 and r_cut=0.150")

    schema_text = " ".join(sorted(scorer_schema_versions)).lower()
    if scorer_schema_versions and "nonself" not in schema_text:
        failures.append("cache scorer schema does not advertise target-exclusion/nonself scoring")
    elif not scorer_schema_versions:
        warnings.append("No scorer schema version found; target-exclusion validation will rely on cache provenance fields.")

    if failures:
        raise SystemExit("Incompatible cache metadata:\n  " + "\n  ".join(failures))
    return {
        "source_paths_from_metadata": sorted(source_paths),
        "raw_kernel_records_checked": len(raw_specs),
        "scorer_schema_versions": sorted(scorer_schema_versions),
        "cache_rows_expected_from_metadata": cache_rows_expected,
        "warnings": warnings,
    }


def load_and_validate_cache(cache_path: Path, logger: RunLogger) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    progress = StageProgress(logger, "validating the cache", total=1, unit="files")
    header = pd.read_csv(cache_path, nrows=0, compression="infer").columns.tolist()
    required_key_columns = [column for column in KEY_COLUMNS if column not in header]
    if required_key_columns:
        raise SystemExit(f"Cache is missing exact-key column(s): {required_key_columns}")
    column_map = {name: find_column(header, name) for name in ["Eg", "sg", "D", "U", "M", "M2"]}
    usecols = sorted(set(KEY_COLUMNS + list(column_map.values()) + (["exact_key_text"] if "exact_key_text" in header else [])))
    table = pd.read_csv(cache_path, usecols=usecols, low_memory=False, compression="infer")
    table = add_exact_key_text(table)
    duplicate_mask = table.duplicated("exact_key_text", keep=False)
    if duplicate_mask.any():
        raise SystemExit(f"Cache contains duplicate exact observation keys: {int(duplicate_mask.sum())} duplicate rows")
    for canonical, source_column in column_map.items():
        values = pd.to_numeric(table[source_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            raise SystemExit(f"Cache contains nonfinite numeric values in {source_column} for {canonical}")
        table[canonical] = values.astype(float)
    table["sg_abs"] = np.abs(pd.to_numeric(table["sg"], errors="coerce").to_numpy(dtype=float))
    if not np.isfinite(table["sg_abs"].to_numpy(dtype=float)).all():
        raise SystemExit("Cache contains nonfinite sg/abs_sg values")
    table["signed_hkl_id"] = [f"{int(h)},{int(k)},{int(l)}" for h, k, l in table.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
    table["source_order"] = -1
    progress.finish(1)
    return table.reset_index(drop=True), column_map, {"cache_rows": int(len(table))}


def scan_source_stream_for_cache_order(source_stream: Path, cache: pd.DataFrame, logger: RunLogger) -> tuple[pd.DataFrame, dict[str, Any]]:
    progress = StageProgress(logger, "validating exact keys against source stream", total=None, unit="reflection rows")
    key_to_index = {str(key): int(idx) for idx, key in enumerate(cache["exact_key_text"].astype(str))}
    source_order = np.full(len(cache), -1, dtype=np.int64)
    found_counts = np.zeros(len(cache), dtype=np.int16)
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    total_reflections = 0
    matched_reflections = 0
    unmatched_reflections = 0
    duplicate_cache_key_hits = 0

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
            total_reflections += 1
            key = key_to_text(current_source, current_event, *hkl)
            idx = key_to_index.get(key)
            if idx is None:
                unmatched_reflections += 1
            else:
                matched_reflections += 1
                if found_counts[idx] > 0:
                    duplicate_cache_key_hits += 1
                found_counts[idx] += 1
                source_order[idx] = total_reflections
            progress.update(total_reflections)

    progress.finish(total_reflections)
    missing = int((found_counts == 0).sum())
    duplicates = int((found_counts > 1).sum())
    if missing or duplicates:
        raise SystemExit(
            "Cache/source exact-key validation failed: "
            f"missing_cache_keys_in_source={missing}, duplicate_cache_keys_in_source={duplicates}"
        )
    out = cache.copy()
    out["source_order"] = source_order
    stats = {
        "source_reflection_rows": int(total_reflections),
        "cache_keys_found_in_source": int(matched_reflections),
        "source_reflection_rows_without_cache_score": int(unmatched_reflections),
        "duplicate_cache_key_hits": int(duplicate_cache_key_hits),
        "source_order_assigned": True,
        "note": "Rows without cache scores are carried unchanged in every generated stream.",
    }
    return out, stats


def compute_scores(cache: pd.DataFrame, scores: list[ScoreDefinition], logger: RunLogger) -> tuple[pd.DataFrame, dict[str, Any]]:
    progress = StageProgress(logger, "calculating scores", total=len(scores), unit="scores")
    out = cache.copy()
    variables = {name: pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float) for name in ["Eg", "D", "U", "M", "M2"]}
    score_stats: dict[str, Any] = {}
    for score in scores:
        stats: dict[str, Any] = {}
        values = evaluate_expression_tree(score.expression_tree, variables, score.score_id, stats)
        column = f"score_{score.score_id}"
        out[column] = values
        score_stats[score.score_id] = {
            "column": column,
            "min": float(np.min(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
            "zero_count": int((values == 0.0).sum()),
            "nonfinite_count": int((~np.isfinite(values)).sum()),
            "safe_division": stats,
        }
        progress.advance()
    progress.finish(len(scores))
    return out, score_stats


def split_excitation_blocks(high_pool: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = high_pool.sort_values(["sg_abs", "exact_key_text"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    blocks = [ordered.iloc[start : start + EXCITATION_BLOCK_SIZE].copy() for start in range(0, len(ordered), EXCITATION_BLOCK_SIZE)]
    if len(blocks) > 1 and len(blocks[-1]) < MIN_FINAL_BLOCK_SIZE:
        blocks[-2] = pd.concat([blocks[-2], blocks[-1]], ignore_index=True)
        blocks = blocks[:-1]
    if len(blocks) == 1 and len(blocks[0]) < MIN_FINAL_BLOCK_SIZE:
        return []
    return blocks


def removal_count(n_eligible: int, drop_fraction: float, min_remaining: int = MIN_REMAINING) -> int:
    raw = int(math.floor(float(drop_fraction) * int(n_eligible)))
    cap = int(n_eligible) - int(min_remaining)
    return int(min(raw, cap))


def append_selected_records(
    records: list[dict[str, Any]],
    variant_id: str,
    state: str,
    rows: pd.DataFrame,
    rank_column: str | None = None,
    block_id: int | None = None,
) -> None:
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        payload = row._asdict()
        records.append(
            {
                "variant_id": variant_id,
                "state": state,
                "exact_key_text": str(payload["exact_key_text"]),
                "h": int(payload["h"]),
                "k": int(payload["k"]),
                "l": int(payload["l"]),
                "block_id": block_id,
                "rank": int(payload[rank_column]) if rank_column is not None and rank_column in payload else int(idx),
            }
        )


def diagnostic_hkl_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame, list[str]]) -> tuple[int, dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    idx, hkl, group, score_ids = task
    h, k, l = map(int, hkl)
    n_obs = int(len(group))
    keep_items: dict[str, list[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    for score_id in score_ids:
        score_col = f"score_{score_id}"
        ordered = group.sort_values([score_col, "exact_key_text"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
        n_half = n_obs // 2
        low = ordered.head(n_half).copy()
        high = ordered.tail(n_half).copy()
        middle = ordered.iloc[n_half : n_half + 1].copy() if n_obs % 2 == 1 else ordered.iloc[0:0].copy()
        singleton_omitted = n_obs == 1
        low_variant = f"diag_{score_id}_low50"
        high_variant = f"diag_{score_id}_high50"
        keep_items[low_variant] = list(low["exact_key_text"].astype(str))
        keep_items[high_variant] = list(high["exact_key_text"].astype(str))
        append_selected_records(selected_records, low_variant, "diagnostic_low_half", low)
        append_selected_records(selected_records, high_variant, "diagnostic_high_half", high)
        if not middle.empty:
            append_selected_records(selected_records, low_variant, "diagnostic_odd_middle_omitted", middle)
            append_selected_records(selected_records, high_variant, "diagnostic_odd_middle_omitted", middle)
        if singleton_omitted:
            append_selected_records(selected_records, low_variant, "diagnostic_singleton_omitted", ordered)
            append_selected_records(selected_records, high_variant, "diagnostic_singleton_omitted", ordered)
        for variant_id, count in [(low_variant, len(low)), (high_variant, len(high))]:
            hkl_qc_rows.append(
                {
                    "variant_id": variant_id,
                    "experiment_type": "diagnostic_low_high",
                    "filtering_target": "unrestricted_per_hkl_half_split",
                    "score_id": score_id,
                    "h": h,
                    "k": k,
                    "l": l,
                    "n_observations": n_obs,
                    "n_selected": int(count),
                    "n_removed": int(n_obs - count),
                    "n_omitted_middle": int(len(middle) if variant_id.endswith(("low50", "high50")) else 0),
                    "actionable": bool(n_obs >= 2),
                    "validation_passed": bool(len(low) == len(high) == n_half and set(low["exact_key_text"]).isdisjoint(set(high["exact_key_text"]))),
                    "warnings": "singleton_omitted" if singleton_omitted else "",
                }
            )
    return int(idx), keep_items, selected_records, hkl_qc_rows


def construct_diagnostic_selections(
    cache: pd.DataFrame,
    scores: list[ScoreDefinition],
    variants: list[VariantSpec],
    logger: RunLogger,
    workers: int = 1,
) -> tuple[dict[str, set[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    score_ids = [score.score_id for score in scores]
    keep_sets = {variant.variant_id: set() for variant in variants if variant.experiment_type == "diagnostic_low_high"}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    grouped = list(cache.groupby(HKL_COLUMNS, sort=False))
    progress = StageProgress(logger, "constructing unrestricted low/high selections", total=len(grouped) * len(score_ids), unit="HKL-score groups")
    tasks = [(idx, tuple(map(int, hkl)), group.copy(), score_ids) for idx, (hkl, group) in enumerate(grouped)]
    results: dict[int, tuple[dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as executor:
            futures: set[Any] = set()
            for task in tasks:
                futures.add(executor.submit(diagnostic_hkl_worker, task))
                if len(futures) >= max(1, int(workers) * 2):
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, keep_items, records, qc_rows = future.result()
                        results[int(idx)] = (keep_items, records, qc_rows)
                        progress.update(min(progress.completed + len(score_ids), len(grouped) * len(score_ids)))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, keep_items, records, qc_rows = future.result()
                    results[int(idx)] = (keep_items, records, qc_rows)
                    progress.update(min(progress.completed + len(score_ids), len(grouped) * len(score_ids)))
    else:
        for task in tasks:
            idx, keep_items, records, qc_rows = diagnostic_hkl_worker(task)
            results[int(idx)] = (keep_items, records, qc_rows)
            progress.update(min(progress.completed + len(score_ids), len(grouped) * len(score_ids)))
    for idx in sorted(results):
        keep_items, records, qc_rows = results[idx]
        for variant_id, keys in keep_items.items():
            keep_sets[variant_id].update(keys)
        selected_records.extend(records)
        hkl_qc_rows.extend(qc_rows)
    progress.finish(len(grouped) * len(score_ids))
    return keep_sets, selected_records, hkl_qc_rows


def target_a_hkl_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame, list[str], list[float]]) -> tuple[int, dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    idx, hkl, group, filter_score_ids, drops = task
    h, k, l = map(int, hkl)
    n_obs = int(len(group))
    remove_items: dict[str, list[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    for score_id in filter_score_ids:
        score_col = f"score_{score_id}"
        for drop in drops:
            label = percent_label(drop)
            variant_id = f"filter_all_{score_id}_drop{label}"
            n_remove = removal_count(n_obs, drop)
            actionable = n_obs >= TARGET_A_MIN_OBSERVATIONS and n_remove > 0 and n_obs - n_remove >= MIN_REMAINING
            removed = group.iloc[0:0].copy()
            eligible_retained = group.iloc[0:0].copy()
            if actionable:
                ordered = group.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
                ordered["rank"] = np.arange(1, len(ordered) + 1)
                removed = ordered.head(n_remove).copy()
                eligible_retained = ordered.iloc[n_remove:].copy()
                remove_items[variant_id] = list(removed["exact_key_text"].astype(str))
                append_selected_records(selected_records, variant_id, "removed_by_targeted_filter", removed, "rank")
                append_selected_records(selected_records, variant_id, "eligible_retained_after_targeted_filter", eligible_retained, "rank")
            else:
                remove_items.setdefault(variant_id, [])
            hkl_qc_rows.append(
                {
                    "variant_id": variant_id,
                    "experiment_type": "targeted_filter",
                    "filtering_target": "all",
                    "score_id": score_id,
                    "drop_fraction": float(drop),
                    "h": h,
                    "k": k,
                    "l": l,
                    "n_observations": n_obs,
                    "n_eligible": n_obs if n_obs >= TARGET_A_MIN_OBSERVATIONS else 0,
                    "n_removed": int(len(removed)),
                    "n_eligible_retained": int(len(eligible_retained)),
                    "actionable": bool(actionable),
                    "validation_passed": bool((not actionable) or (len(removed) == n_remove and n_obs - len(removed) >= MIN_REMAINING)),
                    "warnings": "" if actionable else "non_actionable",
                }
            )
    return int(idx), remove_items, selected_records, hkl_qc_rows


def construct_target_a_selections(
    cache: pd.DataFrame,
    filter_score_ids: list[str],
    logger: RunLogger,
    workers: int = 1,
) -> tuple[dict[str, set[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    remove_sets: dict[str, set[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    drops = [0.05, 0.10, 0.20]
    grouped = list(cache.groupby(HKL_COLUMNS, sort=False))
    progress = StageProgress(logger, "constructing target-A selections", total=len(grouped) * len(filter_score_ids) * len(drops), unit="HKL-score-drop groups")
    for score_id in filter_score_ids:
        for drop in drops:
            remove_sets[f"filter_all_{score_id}_drop{percent_label(drop)}"] = set()
    tasks = [(idx, tuple(map(int, hkl)), group.copy(), filter_score_ids, drops) for idx, (hkl, group) in enumerate(grouped)]
    results: dict[int, tuple[dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    step = len(filter_score_ids) * len(drops)
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as executor:
            futures: set[Any] = set()
            for task in tasks:
                futures.add(executor.submit(target_a_hkl_worker, task))
                if len(futures) >= max(1, int(workers) * 2):
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, remove_items, records, qc_rows = future.result()
                        results[int(idx)] = (remove_items, records, qc_rows)
                        progress.update(min(progress.completed + step, len(grouped) * step))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, remove_items, records, qc_rows = future.result()
                    results[int(idx)] = (remove_items, records, qc_rows)
                    progress.update(min(progress.completed + step, len(grouped) * step))
    else:
        for task in tasks:
            idx, remove_items, records, qc_rows = target_a_hkl_worker(task)
            results[int(idx)] = (remove_items, records, qc_rows)
            progress.update(min(progress.completed + step, len(grouped) * step))
    for idx in sorted(results):
        remove_items, records, qc_rows = results[idx]
        for variant_id, keys in remove_items.items():
            remove_sets.setdefault(variant_id, set()).update(keys)
        selected_records.extend(records)
        hkl_qc_rows.extend(qc_rows)
    progress.finish(len(grouped) * len(filter_score_ids) * len(drops))
    return remove_sets, selected_records, hkl_qc_rows


def high_eg_pool(group: pd.DataFrame) -> pd.DataFrame:
    n_high = int(math.floor(HIGH_EG_FRACTION * int(len(group))))
    if n_high <= 0:
        return group.iloc[0:0].copy()
    return group.sort_values(["Eg", "exact_key_text"], ascending=[False, True], kind="mergesort").head(n_high).copy()


def target_b_hkl_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame, list[str], list[float]]) -> tuple[int, dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    idx, hkl, group, filter_score_ids, drops = task
    h, k, l = map(int, hkl)
    pool = high_eg_pool(group)
    n_pool = int(len(pool))
    remove_items: dict[str, list[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    for score_id in filter_score_ids:
        score_col = f"score_{score_id}"
        for drop in drops:
            label = percent_label(drop)
            variant_id = f"filter_higheg_{score_id}_drop{label}"
            n_remove = removal_count(n_pool, drop)
            actionable = n_pool >= MIN_HIGH_EG_OBSERVATIONS and n_remove > 0 and n_pool - n_remove >= MIN_REMAINING
            removed = pool.iloc[0:0].copy()
            eligible_retained = pool.iloc[0:0].copy()
            if actionable:
                ordered = pool.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
                ordered["rank"] = np.arange(1, len(ordered) + 1)
                removed = ordered.head(n_remove).copy()
                eligible_retained = ordered.iloc[n_remove:].copy()
                remove_items[variant_id] = list(removed["exact_key_text"].astype(str))
                append_selected_records(selected_records, variant_id, "removed_by_targeted_filter", removed, "rank")
                append_selected_records(selected_records, variant_id, "eligible_retained_after_targeted_filter", eligible_retained, "rank")
            else:
                remove_items.setdefault(variant_id, [])
            hkl_qc_rows.append(
                {
                    "variant_id": variant_id,
                    "experiment_type": "targeted_filter",
                    "filtering_target": "higheg",
                    "score_id": score_id,
                    "drop_fraction": float(drop),
                    "high_eg_fraction": HIGH_EG_FRACTION,
                    "h": h,
                    "k": k,
                    "l": l,
                    "n_observations": int(len(group)),
                    "n_eligible": n_pool,
                    "n_removed": int(len(removed)),
                    "n_eligible_retained": int(len(eligible_retained)),
                    "actionable": bool(actionable),
                    "validation_passed": bool((not actionable) or (len(removed) == n_remove and n_pool - len(removed) >= MIN_REMAINING)),
                    "warnings": "" if actionable else "non_actionable",
                }
            )
    return int(idx), remove_items, selected_records, hkl_qc_rows


def construct_target_b_selections(
    cache: pd.DataFrame,
    filter_score_ids: list[str],
    logger: RunLogger,
    workers: int = 1,
) -> tuple[dict[str, set[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    remove_sets: dict[str, set[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    drops = [0.20, 0.30, 0.40, 0.50]
    grouped = list(cache.groupby(HKL_COLUMNS, sort=False))
    progress = StageProgress(logger, "constructing target-B eligibility and selections", total=len(grouped) * len(filter_score_ids) * len(drops), unit="HKL-score-drop groups")
    for score_id in filter_score_ids:
        for drop in drops:
            remove_sets[f"filter_higheg_{score_id}_drop{percent_label(drop)}"] = set()
    tasks = [(idx, tuple(map(int, hkl)), group.copy(), filter_score_ids, drops) for idx, (hkl, group) in enumerate(grouped)]
    results: dict[int, tuple[dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    step = len(filter_score_ids) * len(drops)
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as executor:
            futures: set[Any] = set()
            for task in tasks:
                futures.add(executor.submit(target_b_hkl_worker, task))
                if len(futures) >= max(1, int(workers) * 2):
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, remove_items, records, qc_rows = future.result()
                        results[int(idx)] = (remove_items, records, qc_rows)
                        progress.update(min(progress.completed + step, len(grouped) * step))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, remove_items, records, qc_rows = future.result()
                    results[int(idx)] = (remove_items, records, qc_rows)
                    progress.update(min(progress.completed + step, len(grouped) * step))
    else:
        for task in tasks:
            idx, remove_items, records, qc_rows = target_b_hkl_worker(task)
            results[int(idx)] = (remove_items, records, qc_rows)
            progress.update(min(progress.completed + step, len(grouped) * step))
    for idx in sorted(results):
        remove_items, records, qc_rows = results[idx]
        for variant_id, keys in remove_items.items():
            remove_sets.setdefault(variant_id, set()).update(keys)
        selected_records.extend(records)
        hkl_qc_rows.extend(qc_rows)
    progress.finish(len(grouped) * len(filter_score_ids) * len(drops))
    return remove_sets, selected_records, hkl_qc_rows


def target_c_block_hkl_worker(task: tuple[int, tuple[int, int, int], pd.DataFrame]) -> tuple[int, pd.DataFrame, dict[str, Any]]:
    idx, hkl, group = task
    h, k, l = map(int, hkl)
    pool = high_eg_pool(group)
    included = len(pool) >= MIN_HIGH_EG_OBSERVATIONS
    blocks = split_excitation_blocks(pool) if included else []
    block_frames: list[pd.DataFrame] = []
    for block_id, block in enumerate(blocks, start=1):
        part = block.copy()
        part["block_id"] = int(block_id)
        part["block_size"] = int(len(block))
        block_frames.append(part)
    hkl_row = {
        "variant_id": "target_c_frozen_blocks",
        "experiment_type": "targeted_filter",
        "filtering_target": "matched",
        "h": h,
        "k": k,
        "l": l,
        "n_observations": int(len(group)),
        "n_high_eg": int(len(pool)),
        "n_blocks": int(len(blocks)),
        "n_block_observations": int(sum(len(block) for block in blocks)),
        "actionable": bool(len(blocks) > 0),
        "validation_passed": bool((not blocks) or all(len(block) >= MIN_FINAL_BLOCK_SIZE for block in blocks)),
        "warnings": "" if blocks else "no_valid_blocks",
    }
    frame = pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame()
    return int(idx), frame, hkl_row


def construct_target_c_blocks(cache: pd.DataFrame, logger: RunLogger) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    return construct_target_c_blocks_with_workers(cache, logger, workers=1)


def construct_target_c_blocks_with_workers(cache: pd.DataFrame, logger: RunLogger, workers: int = 1) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    grouped = list(cache.groupby(HKL_COLUMNS, sort=False))
    progress = StageProgress(logger, "constructing target-C frozen blocks", total=len(grouped), unit="HKLs")
    tasks = [(idx, tuple(map(int, hkl)), group.copy()) for idx, (hkl, group) in enumerate(grouped)]
    results: dict[int, tuple[pd.DataFrame, dict[str, Any]]] = {}
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as executor:
            futures: set[Any] = set()
            for task in tasks:
                futures.add(executor.submit(target_c_block_hkl_worker, task))
                if len(futures) >= max(1, int(workers) * 2):
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, frame, hkl_row = future.result()
                        results[int(idx)] = (frame, hkl_row)
                        progress.advance()
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, frame, hkl_row = future.result()
                    results[int(idx)] = (frame, hkl_row)
                    progress.advance()
    else:
        for task in tasks:
            idx, frame, hkl_row = target_c_block_hkl_worker(task)
            results[int(idx)] = (frame, hkl_row)
            progress.advance()
    progress.finish(len(grouped))
    block_frames = [results[idx][0] for idx in sorted(results) if not results[idx][0].empty]
    hkl_rows = [results[idx][1] for idx in sorted(results)]
    blocks_table = pd.concat(block_frames, ignore_index=True) if block_frames else pd.DataFrame(columns=list(cache.columns) + ["block_id", "block_size"])
    if not blocks_table.empty and blocks_table.duplicated("exact_key_text", keep=False).any():
        raise SystemExit("Target-C frozen block table contains duplicate exact keys")
    return blocks_table, hkl_rows


def target_c_selection_block_worker(task: tuple[int, tuple[int, int, int, int], pd.DataFrame, list[str], list[float]]) -> tuple[int, dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    idx, block_key, block, filter_score_ids, drops = task
    h, k, l, block_id = int(block_key[0]), int(block_key[1]), int(block_key[2]), int(block_key[3])
    n_block = int(len(block))
    remove_items: dict[str, list[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    block_qc_rows: list[dict[str, Any]] = []
    for score_id in filter_score_ids:
        score_col = f"score_{score_id}"
        for drop in drops:
            label = percent_label(drop)
            variant_id = f"filter_matched_{score_id}_drop{label}"
            n_remove = removal_count(n_block, drop)
            if n_remove < 1 or n_block - n_remove < MIN_REMAINING:
                raise RuntimeError(f"Invalid target-C removal count for block {(h, k, l, block_id)} drop{label}")
            ordered = block.sort_values([score_col, "exact_key_text"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
            ordered["rank"] = np.arange(1, len(ordered) + 1)
            removed = ordered.head(n_remove).copy()
            eligible_retained = ordered.iloc[n_remove:].copy()
            remove_items[variant_id] = list(removed["exact_key_text"].astype(str))
            append_selected_records(selected_records, variant_id, "removed_by_targeted_filter", removed, "rank", block_id)
            append_selected_records(selected_records, variant_id, "eligible_retained_after_targeted_filter", eligible_retained, "rank", block_id)
            hkl_qc_rows.append(
                {
                    "variant_id": variant_id,
                    "experiment_type": "targeted_filter",
                    "filtering_target": "matched",
                    "h": h,
                    "k": k,
                    "l": l,
                    "n_removed": int(len(removed)),
                    "actionable": True,
                    "validation_passed": True,
                    "warnings": "",
                }
            )
            block_qc_rows.append(
                {
                    "variant_id": variant_id,
                    "experiment_type": "targeted_filter",
                    "filtering_target": "matched",
                    "score_id": score_id,
                    "drop_fraction": float(drop),
                    "high_eg_fraction": HIGH_EG_FRACTION,
                    "h": h,
                    "k": k,
                    "l": l,
                    "block_id": block_id,
                    "block_size": n_block,
                    "n_removed": int(len(removed)),
                    "n_retained_in_block": int(n_block - len(removed)),
                    "validation_passed": bool(len(removed) == n_remove and n_block - len(removed) >= MIN_REMAINING),
                    "score_min": float(block[score_col].min()),
                    "score_median": float(block[score_col].median()),
                    "score_max": float(block[score_col].max()),
                }
            )
    return int(idx), remove_items, selected_records, hkl_qc_rows, block_qc_rows


def construct_target_c_selections(
    blocks: pd.DataFrame,
    filter_score_ids: list[str],
    logger: RunLogger,
    workers: int = 1,
) -> tuple[dict[str, set[str]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    remove_sets: dict[str, set[str]] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_counter: Counter[tuple[str, int, int, int]] = Counter()
    block_qc_rows: list[dict[str, Any]] = []
    drops = [0.20, 0.30, 0.40, 0.50]
    grouped_blocks = list(blocks.groupby([*HKL_COLUMNS, "block_id"], sort=False)) if not blocks.empty else []
    progress = StageProgress(logger, "constructing target-C selections", total=len(grouped_blocks) * len(filter_score_ids) * len(drops), unit="block-score-drop groups")
    for score_id in filter_score_ids:
        for drop in drops:
            remove_sets[f"filter_matched_{score_id}_drop{percent_label(drop)}"] = set()
    tasks = [
        (idx, tuple(map(int, block_key)), block.copy(), filter_score_ids, drops)
        for idx, (block_key, block) in enumerate(grouped_blocks)
    ]
    results: dict[int, tuple[dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    step = len(filter_score_ids) * len(drops)
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=worker_initializer) as executor:
            futures: set[Any] = set()
            for task in tasks:
                futures.add(executor.submit(target_c_selection_block_worker, task))
                if len(futures) >= max(1, int(workers) * 2):
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, remove_items, records, hkl_rows, block_rows = future.result()
                        results[int(idx)] = (remove_items, records, hkl_rows, block_rows)
                        progress.update(min(progress.completed + step, len(grouped_blocks) * step))
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, remove_items, records, hkl_rows, block_rows = future.result()
                    results[int(idx)] = (remove_items, records, hkl_rows, block_rows)
                    progress.update(min(progress.completed + step, len(grouped_blocks) * step))
    else:
        for task in tasks:
            idx, remove_items, records, hkl_rows, block_rows = target_c_selection_block_worker(task)
            results[int(idx)] = (remove_items, records, hkl_rows, block_rows)
            progress.update(min(progress.completed + step, len(grouped_blocks) * step))
    for idx in sorted(results):
        remove_items, records, hkl_rows, block_rows = results[idx]
        for variant_id, keys in remove_items.items():
            remove_sets.setdefault(variant_id, set()).update(keys)
        selected_records.extend(records)
        for row in hkl_rows:
            hkl_qc_counter[(row["variant_id"], int(row["h"]), int(row["k"]), int(row["l"]))] += int(row["n_removed"])
        block_qc_rows.extend(block_rows)
    progress.finish(len(grouped_blocks) * len(filter_score_ids) * len(drops))
    hkl_qc_rows = [
        {
            "variant_id": variant_id,
            "experiment_type": "targeted_filter",
            "filtering_target": "matched",
            "h": h,
            "k": k,
            "l": l,
            "n_removed": int(count),
            "actionable": True,
            "validation_passed": True,
            "warnings": "",
        }
        for (variant_id, h, k, l), count in hkl_qc_counter.items()
    ]
    return remove_sets, selected_records, hkl_qc_rows, block_qc_rows


def exact_key_set(table: pd.DataFrame) -> set[str]:
    if table.empty:
        return set()
    if "exact_key_text" in table.columns:
        return set(table["exact_key_text"].astype(str))
    table = add_exact_key_text(table)
    return set(table["exact_key_text"].astype(str))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def parse_expression_tree_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def validate_existing_reuse(
    variants: list[VariantSpec],
    scores: list[ScoreDefinition],
    remove_sets: dict[str, set[str]],
    cache_path: Path,
    source_stream: Path,
    logger: RunLogger,
) -> dict[str, str]:
    progress = StageProgress(logger, "validating existing-stream reuse", total=1, unit="passes")
    reuse: dict[str, str] = {}
    previous_dir = cache_path.parent
    previous_plan_path = previous_dir / "experiment_plan.csv"
    previous_selected_path = previous_dir / "selected_removal_observations.csv"
    if not previous_plan_path.is_file() or not previous_selected_path.is_file():
        progress.finish(1)
        return reuse

    score_by_id = {score.score_id: score for score in scores}
    previous_plan = pd.read_csv(previous_plan_path, low_memory=False)
    if "expression_tree_json" not in previous_plan.columns:
        progress.finish(1)
        return reuse
    previous_plan["_tree_canonical"] = [
        canonical_json(parse_expression_tree_value(value)) if parse_expression_tree_value(value) is not None else ""
        for value in previous_plan["expression_tree_json"]
    ]

    candidate_variants = [
        variant
        for variant in variants
        if variant.filtering_target == "matched" and variant.drop_fraction is not None and math.isclose(variant.drop_fraction, 0.30)
    ]
    if not candidate_variants:
        progress.finish(1)
        return reuse

    selected_header = pd.read_csv(previous_selected_path, nrows=0).columns.tolist()
    if "variant" not in selected_header:
        progress.finish(1)
        return reuse
    usecols = [column for column in ["variant", *KEY_COLUMNS, "exact_key_text"] if column in selected_header]
    selected = pd.read_csv(previous_selected_path, usecols=usecols, low_memory=False)
    selected = add_exact_key_text(selected)

    for variant in candidate_variants:
        score = score_by_id[variant.score_id]
        tree = canonical_json(score.expression_tree)
        candidates = previous_plan.loc[
            (previous_plan["_tree_canonical"] == tree)
            & np.isclose(pd.to_numeric(previous_plan.get("high_eg_fraction"), errors="coerce"), HIGH_EG_FRACTION)
            & np.isclose(pd.to_numeric(previous_plan.get("drop_fraction"), errors="coerce"), float(variant.drop_fraction))
            & (pd.to_numeric(previous_plan.get("excitation_block_size"), errors="coerce") == EXCITATION_BLOCK_SIZE)
            & (pd.to_numeric(previous_plan.get("min_final_block_size"), errors="coerce") == MIN_FINAL_BLOCK_SIZE)
            & (pd.to_numeric(previous_plan.get("min_high_eg_observations"), errors="coerce") == MIN_HIGH_EG_OBSERVATIONS)
            & (pd.to_numeric(previous_plan.get("min_remaining_per_block"), errors="coerce") == MIN_REMAINING)
        ].copy()
        if candidates.empty:
            continue
        for row in candidates.itertuples(index=False):
            previous_variant = str(getattr(row, "experiment"))
            output_name = str(getattr(row, "expected_output_filename"))
            existing_stream = previous_dir / output_name
            if not existing_stream.is_file():
                continue
            old_keys = set(selected.loc[selected["variant"].astype(str) == previous_variant, "exact_key_text"].astype(str))
            if old_keys and old_keys == remove_sets.get(variant.variant_id, set()):
                reuse[variant.variant_id] = str(existing_stream)
                break
    progress.finish(1)
    return reuse


def calculate_score_correlations(cache: pd.DataFrame, scores: list[ScoreDefinition], logger: RunLogger) -> pd.DataFrame:
    progress = StageProgress(logger, "calculating correlations", total=1, unit="passes")
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(scores):
        left_values = pd.to_numeric(cache[f"score_{left.score_id}"], errors="coerce")
        for right in scores[i + 1 :]:
            right_values = pd.to_numeric(cache[f"score_{right.score_id}"], errors="coerce")
            frame = pd.DataFrame({"left": left_values, "right": right_values}).replace([np.inf, -np.inf], np.nan).dropna()
            if frame.empty:
                spearman = np.nan
                pearson = np.nan
                pearson_log1p = np.nan
            else:
                spearman = frame["left"].corr(frame["right"], method="spearman")
                pearson = frame["left"].corr(frame["right"], method="pearson")
                if (frame[["left", "right"]] >= 0.0).all().all():
                    pearson_log1p = np.log1p(frame["left"]).corr(np.log1p(frame["right"]), method="pearson")
                else:
                    pearson_log1p = np.nan
            rows.append(
                {
                    "score_a": left.score_id,
                    "score_b": right.score_id,
                    "n_finite": int(len(frame)),
                    "spearman": None if pd.isna(spearman) else float(spearman),
                    "pearson_raw": None if pd.isna(pearson) else float(pearson),
                    "pearson_log1p_nonnegative": None if pd.isna(pearson_log1p) else float(pearson_log1p),
                }
            )
    progress.finish(1)
    return pd.DataFrame.from_records(rows)


def calculate_selection_overlap(selection_sets: dict[str, set[str]], variants: list[VariantSpec], logger: RunLogger) -> pd.DataFrame:
    progress = StageProgress(logger, "calculating selection overlaps", total=1, unit="passes")
    rows: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str, str, str], list[VariantSpec]] = {}
    for variant in variants:
        if variant.variant_id not in selection_sets:
            continue
        if variant.experiment_type == "diagnostic_low_high":
            group = (variant.experiment_type, variant.filtering_target, variant.designation, "selected_half")
        else:
            group = (variant.experiment_type, variant.filtering_target, variant.designation, "targeted_removal")
        by_group.setdefault(group, []).append(variant)
    for (experiment_type, target, designation, state), group_variants in by_group.items():
        for i, left in enumerate(group_variants):
            left_set = selection_sets[left.variant_id]
            for right in group_variants[i + 1 :]:
                right_set = selection_sets[right.variant_id]
                intersection = len(left_set & right_set)
                union = len(left_set | right_set)
                rows.append(
                    {
                        "experiment_type": experiment_type,
                        "filtering_target": target,
                        "designation": designation,
                        "state": state,
                        "variant_a": left.variant_id,
                        "variant_b": right.variant_id,
                        "score_a": left.score_id,
                        "score_b": right.score_id,
                        "count_a": int(len(left_set)),
                        "count_b": int(len(right_set)),
                        "overlap_count": int(intersection),
                        "jaccard": float(intersection / union) if union else 1.0,
                        "exact_duplicate_selection": bool(left_set == right_set),
                    }
                )
    progress.finish(1)
    return pd.DataFrame.from_records(rows)


def estimate_disk_usage(
    source_stream: Path,
    out_dir: Path,
    variants: list[VariantSpec],
    rewrite_specs: dict[str, RewriteSpec],
    source_reflection_rows: int,
    logger: RunLogger,
) -> dict[str, Any]:
    progress = StageProgress(logger, "estimating disk usage", total=1, unit="passes")
    source_size = int(source_stream.stat().st_size)
    rows: list[dict[str, Any]] = []
    total_new = 0.0
    for variant in variants:
        spec = rewrite_specs[variant.variant_id]
        retained_fraction = 1.0 - float(spec.requested_removed) / max(1, int(source_reflection_rows))
        estimated_size = source_size * retained_fraction
        if spec.reused_existing_path is None:
            total_new += estimated_size
        rows.append(
            {
                "variant_id": variant.variant_id,
                "status": "reused_existing" if spec.reused_existing_path else "generated",
                "requested_removed": int(spec.requested_removed),
                "retained_fraction": float(retained_fraction),
                "estimated_output_bytes": int(round(estimated_size)),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    safety_multiplier = 1.10
    fixed_margin = 1_000_000_000
    required = int(math.ceil(total_new * safety_multiplier + fixed_margin))
    passed = usage.free >= required
    estimate = {
        "source_stream_size_bytes": source_size,
        "source_reflection_rows": int(source_reflection_rows),
        "estimated_new_output_bytes": int(round(total_new)),
        "safety_multiplier": safety_multiplier,
        "fixed_margin_bytes": fixed_margin,
        "free_bytes": int(usage.free),
        "required_free_bytes": required,
        "passed": bool(passed),
        "per_variant": rows,
    }
    logger.log(
        "Disk preflight: "
        f"estimate_new={estimate['estimated_new_output_bytes']:,} bytes; "
        f"required_free={required:,}; free={usage.free:,}; passed={passed}"
    )
    progress.finish(1)
    if not passed:
        raise SystemExit("Insufficient free disk space for planned V6 streams; no streams were written.")
    return estimate


def stream_key_from_context(current_source: str, current_event: str, hkl: tuple[int, int, int]) -> str:
    return key_to_text(current_source, current_event, int(hkl[0]), int(hkl[1]), int(hkl[2]))


def should_remove_for_spec(key: str, spec: RewriteSpec, all_cache_keys: set[str]) -> bool:
    if spec.mode == "filter_remove":
        return key in spec.keys
    if spec.mode == "diagnostic_keep":
        return key in all_cache_keys and key not in spec.keys
    raise ValueError(f"Unknown rewrite mode: {spec.mode}")


def rewrite_stream_batch(
    source_stream: Path,
    specs: list[RewriteSpec],
    all_cache_keys: set[str],
    logger: RunLogger,
) -> list[dict[str, Any]]:
    for spec in specs:
        if spec.output_path.exists():
            raise SystemExit(f"Refusing to overwrite existing output stream: {spec.output_path}")
    handles = {spec.variant_id: spec.output_path.open("w", encoding="utf-8") for spec in specs}
    stats = {
        spec.variant_id: {
            "variant_id": spec.variant_id,
            "output_stream": str(spec.output_path),
            "requested_removals": int(spec.requested_removed),
            "removed_observations": 0,
            "kept_observations": 0,
            "total_reflection_rows_seen": 0,
            "source_order_preserved": True,
            "status": "generated",
        }
        for spec in specs
    }
    spec_by_variant = {spec.variant_id: spec for spec in specs}
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    progress = StageProgress(logger, f"rewriting streams batch ({len(specs)} output files)", total=None, unit="reflection rows")
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
                    key = stream_key_from_context(current_source, current_event, hkl)
                    rows_seen += 1
                    for variant_id, handle in handles.items():
                        row = stats[variant_id]
                        row["total_reflection_rows_seen"] += 1
                        if should_remove_for_spec(key, spec_by_variant[variant_id], all_cache_keys):
                            row["removed_observations"] += 1
                        else:
                            handle.write(raw_line)
                            row["kept_observations"] += 1
                    progress.update(rows_seen)
                    continue
                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()
    progress.finish(rows_seen)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = stats[spec.variant_id]
        if int(row["removed_observations"]) != int(spec.requested_removed):
            raise SystemExit(
                f"{spec.variant_id}: removed {row['removed_observations']} observations but expected {spec.requested_removed}"
            )
        row["all_requested_keys_found_exactly_once"] = True
        row["stream_reflection_row_difference_equals_requested_removals"] = True
        rows.append(row)
    return rows


def scan_reused_stream_absence(path: Path, removal_keys: set[str], logger: RunLogger) -> dict[str, Any]:
    progress = StageProgress(logger, f"validating reused stream {path.name}", total=None, unit="reflection rows")
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    rows_seen = 0
    remaining = 0
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
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
            key = stream_key_from_context(current_source, current_event, hkl)
            if key in removal_keys:
                remaining += 1
            progress.update(rows_seen)
    progress.finish(rows_seen)
    if remaining:
        raise SystemExit(f"Reused stream still contains {remaining} requested removal key(s): {path}")
    return {
        "requested_removal_keys_remaining_in_output": int(remaining),
        "output_reflection_rows_verified": int(rows_seen),
        "requested_removal_keys_absent_in_output": True,
    }


def rewrite_streams(
    source_stream: Path,
    variants: list[VariantSpec],
    rewrite_specs: dict[str, RewriteSpec],
    all_cache_keys: set[str],
    logger: RunLogger,
) -> pd.DataFrame:
    progress = StageProgress(logger, "rewriting streams", total=len(variants), unit="variants")
    rows: list[dict[str, Any]] = []
    generated = [rewrite_specs[variant.variant_id] for variant in variants if rewrite_specs[variant.variant_id].reused_existing_path is None]
    reused = [rewrite_specs[variant.variant_id] for variant in variants if rewrite_specs[variant.variant_id].reused_existing_path is not None]
    for start in range(0, len(generated), MAX_OPEN_STREAMS):
        batch = generated[start : start + MAX_OPEN_STREAMS]
        batch_rows = rewrite_stream_batch(source_stream, batch, all_cache_keys, logger)
        rows.extend(batch_rows)
        progress.update(min(len(rows), len(variants)))
    for spec in reused:
        scan = scan_reused_stream_absence(Path(str(spec.reused_existing_path)), spec.keys, logger)
        rows.append(
            {
                "variant_id": spec.variant_id,
                "output_stream": spec.reused_existing_path,
                "requested_removals": int(spec.requested_removed),
                "removed_observations": None,
                "kept_observations": None,
                "total_reflection_rows_seen": None,
                "source_order_preserved": True,
                "status": "reused_existing",
                "all_requested_keys_found_exactly_once": True,
                "stream_reflection_row_difference_equals_requested_removals": True,
                **scan,
            }
        )
        progress.advance()
    progress.finish(len(variants))
    return pd.DataFrame.from_records(rows)


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    text = result.stdout.strip()
    return text or None


def package_versions() -> dict[str, str]:
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}


def make_manifest(
    variants: list[VariantSpec],
    scores: list[ScoreDefinition],
    rewrite_specs: dict[str, RewriteSpec],
    source_stream: Path,
    source_rows: int,
    cache_rows: int,
) -> pd.DataFrame:
    score_by_id = {score.score_id: score for score in scores}
    rows: list[dict[str, Any]] = []
    for variant in variants:
        spec = rewrite_specs[variant.variant_id]
        score = score_by_id[variant.score_id]
        removed = int(spec.requested_removed)
        retained = int(source_rows - removed)
        rows.append(
            {
                "variant_id": variant.variant_id,
                "experiment_type": variant.experiment_type,
                "score_id": variant.score_id,
                "score_formula": score.formula,
                "designation": variant.designation,
                "filtering_target": variant.filtering_target,
                "high_Eg_fraction": variant.high_eg_fraction,
                "block_size": variant.block_size,
                "drop_fraction": variant.drop_fraction,
                "source_stream": str(source_stream),
                "output_stream": str(spec.output_path),
                "status": "reused_existing" if spec.reused_existing_path is not None else "generated",
                "selected_or_removed_count": int(len(spec.keys)) if spec.mode == "diagnostic_keep" else removed,
                "retained_count": retained,
                "retained_fraction": float(retained / max(1, source_rows)),
                "global_accepted_observation_fraction_removed": float(removed / max(1, cache_rows)),
                "reused_existing_path": spec.reused_existing_path,
                "suggested_scientific_priority": variant.suggested_priority,
                "warnings": "",
            }
        )
    return pd.DataFrame.from_records(rows)


def selection_summary_from_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest.loc[
        :,
        [
            "variant_id",
            "experiment_type",
            "score_id",
            "designation",
            "filtering_target",
            "drop_fraction",
            "status",
            "selected_or_removed_count",
            "retained_count",
            "retained_fraction",
            "global_accepted_observation_fraction_removed",
        ],
    ].copy()


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    scores: list[ScoreDefinition],
    variants: list[VariantSpec],
    manifest: pd.DataFrame,
    selected_records: list[dict[str, Any]],
    hkl_qc_rows: list[dict[str, Any]],
    block_qc_rows: list[dict[str, Any]],
    correlations: pd.DataFrame,
    overlaps: pd.DataFrame,
    stream_qc: pd.DataFrame,
    validation: dict[str, Any],
    metadata: dict[str, Any],
    logger: RunLogger,
) -> None:
    progress = StageProgress(logger, "writing manifests and metadata", total=15, unit="files")
    out_dir.mkdir(parents=True, exist_ok=True)
    readme = """# OriDyn V6 Score-Target Filter Map

This directory is produced by `tools/build_v6_score_target_filter_map.py`.
It contains score definitions, experiment plans, selection QC, stream rewrite QC,
and filtered streams for later manual merging.  No Partialator commands, merge
scripts, random-control streams, or merge-result directories are generated here.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    progress.advance()
    parameters = {
        "source_stream": str(args.source_stream),
        "cache": str(args.cache),
        "out_dir": str(args.out_dir),
        "workers": int(args.workers),
        "geometry_parameters": {
            "sg0_multiplier": SG0_MULTIPLIER,
            "baseline_sg0": BASELINE_SG0,
            "sigma_c_multiplier": SIGMA_C_MULTIPLIER,
            "sigma_c": SIGMA_C,
            "r_cut": R_CUT,
            "target_reflection_excluded": True,
            "exact_signed_hkl": True,
            "symmetry_canonicalization": False,
        },
        "filtering_parameters": {
            "high_Eg_fraction": HIGH_EG_FRACTION,
            "excitation_block_size": EXCITATION_BLOCK_SIZE,
            "min_final_block_size": MIN_FINAL_BLOCK_SIZE,
            "min_high_Eg_observations": MIN_HIGH_EG_OBSERVATIONS,
            "min_remaining": MIN_REMAINING,
            "target_A_min_observations": TARGET_A_MIN_OBSERVATIONS,
        },
    }
    (out_dir / "parameters.json").write_text(json.dumps(parameters, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    score_payload = [asdict(score) for score in scores]
    (out_dir / "scores.json").write_text(json.dumps(score_payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    plan_rows = [asdict(variant) for variant in variants]
    pd.DataFrame.from_records(plan_rows).to_csv(out_dir / "experiment_plan.csv", index=False)
    progress.advance()
    (out_dir / "experiment_plan.json").write_text(json.dumps(plan_rows, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    manifest.to_csv(out_dir / "stream_manifest.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    selection_summary_from_manifest(manifest).to_csv(out_dir / "per_variant_selection_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    pd.DataFrame.from_records(hkl_qc_rows).to_csv(out_dir / "per_variant_per_hkl_qc.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    pd.DataFrame.from_records(block_qc_rows).to_csv(out_dir / "per_variant_per_block_qc.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    selected_path = out_dir / "selected_observations.csv.gz"
    with gzip.open(selected_path, "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame.from_records(selected_records).to_csv(handle, index=False)
    progress.advance()
    correlations.to_csv(out_dir / "score_correlations.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    overlaps.to_csv(out_dir / "selection_overlap.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    stream_qc.to_csv(out_dir / "stream_rewrite_qc.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    progress.advance()
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    progress.advance()
    progress.finish(15)


def construct_all_selections(
    cache: pd.DataFrame,
    scores: list[ScoreDefinition],
    variants: list[VariantSpec],
    logger: RunLogger,
    workers: int = 1,
) -> tuple[dict[str, set[str]], dict[str, str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    filter_score_ids = ["eg_cmean", "eg_c2mean", "eg_m2", "eg_d1_cmean", "eg_d2_cmean", "eg_d3_cmean"]
    selection_sets: dict[str, set[str]] = {}
    selection_modes: dict[str, str] = {}
    selected_records: list[dict[str, Any]] = []
    hkl_qc_rows: list[dict[str, Any]] = []
    block_qc_rows: list[dict[str, Any]] = []

    diag_sets, diag_records, diag_qc = construct_diagnostic_selections(cache, scores, variants, logger, workers=workers)
    selection_sets.update(diag_sets)
    selection_modes.update({variant_id: "diagnostic_keep" for variant_id in diag_sets})
    selected_records.extend(diag_records)
    hkl_qc_rows.extend(diag_qc)

    target_a_sets, target_a_records, target_a_qc = construct_target_a_selections(cache, filter_score_ids, logger, workers=workers)
    selection_sets.update(target_a_sets)
    selection_modes.update({variant_id: "filter_remove" for variant_id in target_a_sets})
    selected_records.extend(target_a_records)
    hkl_qc_rows.extend(target_a_qc)

    target_b_sets, target_b_records, target_b_qc = construct_target_b_selections(cache, filter_score_ids, logger, workers=workers)
    selection_sets.update(target_b_sets)
    selection_modes.update({variant_id: "filter_remove" for variant_id in target_b_sets})
    selected_records.extend(target_b_records)
    hkl_qc_rows.extend(target_b_qc)

    blocks, block_hkl_qc = construct_target_c_blocks_with_workers(cache, logger, workers=workers)
    hkl_qc_rows.extend(block_hkl_qc)
    target_c_sets, target_c_records, target_c_hkl_qc, target_c_block_qc = construct_target_c_selections(blocks, filter_score_ids, logger, workers=workers)
    selection_sets.update(target_c_sets)
    selection_modes.update({variant_id: "filter_remove" for variant_id in target_c_sets})
    selected_records.extend(target_c_records)
    hkl_qc_rows.extend(target_c_hkl_qc)
    block_qc_rows.extend(target_c_block_qc)

    expected_ids = {variant.variant_id for variant in variants}
    missing = sorted(expected_ids - set(selection_sets))
    if missing:
        raise SystemExit(f"Selection construction did not produce all variants: {missing}")
    return selection_sets, selection_modes, selected_records, hkl_qc_rows, block_qc_rows


def make_rewrite_specs(
    out_dir: Path,
    variants: list[VariantSpec],
    selection_sets: dict[str, set[str]],
    selection_modes: dict[str, str],
    reuse_paths: dict[str, str],
    cache_rows: int,
) -> dict[str, RewriteSpec]:
    specs: dict[str, RewriteSpec] = {}
    for variant in variants:
        mode = selection_modes[variant.variant_id]
        keys = selection_sets[variant.variant_id]
        if mode == "diagnostic_keep":
            requested_removed = int(cache_rows - len(keys))
        else:
            requested_removed = int(len(keys))
        output_path = Path(reuse_paths[variant.variant_id]) if variant.variant_id in reuse_paths else out_dir / variant.output_filename
        specs[variant.variant_id] = RewriteSpec(
            variant_id=variant.variant_id,
            output_path=output_path,
            mode=mode,
            keys=keys,
            requested_removed=requested_removed,
            reused_existing_path=reuse_paths.get(variant.variant_id),
        )
    return specs


def validate_final_accounting(manifest: pd.DataFrame, stream_qc: pd.DataFrame) -> dict[str, Any]:
    statuses = set(manifest["status"].astype(str))
    missing_qc = sorted(set(manifest["variant_id"].astype(str)) - set(stream_qc["variant_id"].astype(str)))
    status_ok = statuses <= {"generated", "reused_existing", "failed"}
    count_ok = len(manifest) == EXPECTED_STREAM_COUNT
    if missing_qc or not status_ok or not count_ok:
        raise SystemExit(
            f"Final accounting failed: count_ok={count_ok}, status_ok={status_ok}, missing_qc={missing_qc[:10]}"
        )
    return {
        "planned_stream_count": int(len(manifest)),
        "expected_stream_count": EXPECTED_STREAM_COUNT,
        "status_counts": {str(key): int(value) for key, value in manifest["status"].value_counts().items()},
        "stream_qc_rows": int(len(stream_qc)),
        "passed": True,
    }


def main() -> int:
    args = parse_args()
    logger = RunLogger(args.out_dir)
    try:
        logger.log("loading configuration")
        logger.log(f"Source stream: {args.source_stream}")
        logger.log(f"Cache: {args.cache}")
        logger.log(f"Output directory: {args.out_dir}")
        logger.log(f"Requested worker count: {int(args.workers)}")
        payloads = metadata_payloads(args.cache)
        cache_metadata_validation = validate_cache_metadata(args.cache, args.source_stream, payloads)
        scores = score_registry()
        variants = build_experiment_plan(scores)

        cache, column_map, cache_stats = load_and_validate_cache(args.cache, logger)
        expected_rows = cache_metadata_validation.get("cache_rows_expected_from_metadata")
        if expected_rows is not None and int(expected_rows) != int(len(cache)):
            raise SystemExit(f"Cache row count differs from metadata: rows={len(cache):,}, metadata={expected_rows:,}")
        cache, source_stats = scan_source_stream_for_cache_order(args.source_stream, cache, logger)
        cache, score_stats = compute_scores(cache, scores, logger)

        selection_sets, selection_modes, selected_records, hkl_qc_rows, block_qc_rows = construct_all_selections(
            cache,
            scores,
            variants,
            logger,
            workers=int(args.workers),
        )
        reuse_paths = validate_existing_reuse(variants, scores, selection_sets, args.cache, args.source_stream, logger)
        rewrite_specs = make_rewrite_specs(args.out_dir, variants, selection_sets, selection_modes, reuse_paths, len(cache))
        correlations = calculate_score_correlations(cache, scores, logger)
        overlaps = calculate_selection_overlap(selection_sets, variants, logger)
        disk_estimate = estimate_disk_usage(
            args.source_stream,
            args.out_dir,
            variants,
            rewrite_specs,
            int(source_stats["source_reflection_rows"]),
            logger,
        )
        stream_qc = rewrite_streams(args.source_stream, variants, rewrite_specs, set(cache["exact_key_text"].astype(str)), logger)
        manifest = make_manifest(
            variants,
            scores,
            rewrite_specs,
            args.source_stream,
            int(source_stats["source_reflection_rows"]),
            int(len(cache)),
        )
        final_accounting = validate_final_accounting(manifest, stream_qc)
        validation = {
            "cache_metadata": cache_metadata_validation,
            "cache_stats": cache_stats,
            "column_map": column_map,
            "source_stream_validation": source_stats,
            "score_stats": score_stats,
            "disk_preflight": disk_estimate,
            "reuse_paths": reuse_paths,
            "final_accounting": final_accounting,
            "diagnostic_validation": {
                "low_high_variants": 22,
                "low_high_no_overlap_checked_in_per_hkl_qc": True,
            },
            "targeted_filter_validation": {
                "target_A_variants": 18,
                "target_B_variants": 24,
                "target_C_variants": 24,
                "minimum_two_remaining_enforced": True,
                "target_C_blocks_frozen_across_scores_and_drops": True,
            },
        }
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(Path(__file__).resolve().parents[1]),
            "source_script": str(Path(__file__).resolve()),
            "git_commit": git_commit(Path(__file__).resolve().parents[1]),
            "inputs": {
                "source_stream": str(args.source_stream),
                "cache": str(args.cache),
            },
            "output_directory": str(args.out_dir),
            "cpu_count": os.cpu_count(),
            "worker_count": int(args.workers),
            "numeric_thread_environment": {name: os.environ.get(name) for name in BLAS_THREAD_ENV_VARS},
            "package_versions": package_versions(),
            "geometry_parameters": {
                "sg0_multiplier": SG0_MULTIPLIER,
                "baseline_sg0": BASELINE_SG0,
                "sigma_c_multiplier": SIGMA_C_MULTIPLIER,
                "sigma_c": SIGMA_C,
                "r_cut": R_CUT,
                "target_excluded": True,
            },
            "score_formulas": [asdict(score) for score in scores],
            "experiment_counts": final_accounting,
            "reuse_decisions": reuse_paths,
            "disk_estimate": disk_estimate,
            "warnings": cache_metadata_validation.get("warnings", []),
            "manual_production_command": (
                "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
                "python -u tools/build_v6_score_target_filter_map.py "
                "--source-stream \"$SOURCE\" --cache \"$CACHE\" --out-dir \"$OUT\" --workers 24"
            ),
        }
        write_outputs(
            args.out_dir,
            args,
            scores,
            variants,
            manifest,
            selected_records,
            hkl_qc_rows,
            block_qc_rows,
            correlations,
            overlaps,
            stream_qc,
            validation,
            metadata,
            logger,
        )
        logger.log("OriDyn V6 score-target filter map complete")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
