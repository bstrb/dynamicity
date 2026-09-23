#!/usr/bin/env python3
"""Complete read-only audit for the OriDyn V6 score/filter experiment.

This script audits the 88 planned V6 score streams, and when present also audits
the corrected full-population surface with 39 generated random-control streams.
It does not regenerate streams, rerun filtering, run Partialator, or modify any
existing experiment files.  All artifacts are written beneath --out-dir.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import audit_v6_matched_drop30 as common


AUDIT_DIRNAME = "oridyn_v6_complete_audit_20260717"
EXPECTED_VARIANT_COUNT = 88
SOURCE_ACCEPTED_OBSERVATIONS = 6_732_955
RESTRICTED_V6_CACHE_ROWS = 659_147
EXPECTED_FULL_POPULATION_RANDOM_COUNT = 39
EXPECTED_FULL_POPULATION_TOTAL_COUNT = 127
FULL_POPULATION_ACCEPTED_OBSERVATIONS = 6_732_955
RANDOM_SEEDS = [20260717, 20260718, 20260719]
V5_MATCHED_EXPECTED = {
    "actionable_observations": 648_248,
    "actionable_blocks": 67_837,
    "removed": 183_608,
    "fraction_of_accepted": 0.027270,
}

DIAGNOSTIC_SCORES = [
    "eg",
    "density_d",
    "legacy_m",
    "eg_m",
    "eg_cmean",
    "eg_c2mean",
    "eg_m2",
    "eg_d1_cmean",
    "eg_d2_cmean",
    "eg_d3_cmean",
    "eg_d2_c2mean",
]
FILTER_SCORES = [
    "eg_cmean",
    "eg_c2mean",
    "eg_m2",
    "eg_d1_cmean",
    "eg_d2_cmean",
    "eg_d3_cmean",
]
TARGET_DROPS = {
    "all": [0.05, 0.10, 0.20],
    "higheg": [0.20, 0.30, 0.40, 0.50],
    "matched": [0.20, 0.30, 0.40, 0.50],
}
SCORE_FORMULAS = {
    "eg": "Eg",
    "density_d": "D",
    "legacy_m": "M",
    "eg_m": "Eg * M",
    "eg_cmean": "Eg * M / U",
    "eg_c2mean": "Eg * M2 / U",
    "eg_m2": "Eg * M2",
    "eg_d1_cmean": "Eg * D * M / U",
    "eg_d2_cmean": "Eg * D^2 * M / U",
    "eg_d3_cmean": "Eg * D^3 * M / U",
    "eg_d2_c2mean": "Eg * D^2 * M2 / U",
}
TARGET_LABELS = {
    "all": "target_A_all",
    "higheg": "target_B_higheg",
    "matched": "target_C_matched",
}
REQUIRED_OUTPUT_FILES = [
    "complete_audit_summary.md",
    "complete_audit_summary.json",
    "experiment_accounting.csv",
    "stream_validation.csv",
    "selection_counts.csv",
    "diagnostic_pair_results.csv",
    "filter_global_metrics.csv",
    "filter_sweeps.csv",
    "global_metrics_all_variants.csv",
    "global_deltas_vs_full.csv",
    "shell_boundaries.csv",
    "shell_metrics_all_variants.csv",
    "shell_deltas_vs_full.csv",
    "shell_diagnostic_low_high_deltas.csv",
    "shell_variant_summary.csv",
    "score_family_comparisons.csv",
    "selection_overlap.csv",
    "v5_v6_equivalence.csv",
    "random_control_comparability.csv",
    "oriented_vs_random_results.csv",
    "merge_settings_audit.csv",
    "halfset_audit.csv",
    "variant_rankings.csv",
    "pareto_front.csv",
    "missing_or_failed_variants.csv",
    "warnings.csv",
    "run_metadata.json",
    "run.log",
]
PLOT_FILES = [
    "diagnostic_cc12_low_vs_high.png",
    "diagnostic_rsplit_low_vs_high.png",
    "diagnostic_cc12_separation.png",
    "diagnostic_rsplit_separation.png",
    "filter_sweep_target_A_cc12_vs_removed_fraction.png",
    "filter_sweep_target_A_rsplit_vs_removed_fraction.png",
    "filter_sweep_target_A_snr_vs_removed_fraction.png",
    "filter_sweep_target_A_redundancy_vs_removed_fraction.png",
    "filter_sweep_target_B_cc12_vs_removed_fraction.png",
    "filter_sweep_target_B_rsplit_vs_removed_fraction.png",
    "filter_sweep_target_B_snr_vs_removed_fraction.png",
    "filter_sweep_target_B_redundancy_vs_removed_fraction.png",
    "filter_sweep_target_C_cc12_vs_removed_fraction.png",
    "filter_sweep_target_C_rsplit_vs_removed_fraction.png",
    "filter_sweep_target_C_snr_vs_removed_fraction.png",
    "filter_sweep_target_C_redundancy_vs_removed_fraction.png",
    "shell_heatmap_cc12_delta_vs_full.png",
    "shell_heatmap_rsplit_delta_vs_full.png",
    "shell_heatmap_snr_delta_vs_full.png",
    "focused_shell_curves_cc12.png",
    "focused_shell_curves_rsplit.png",
    "metric_improvement_vs_observations_removed.png",
    "score_selection_jaccard_matrix.png",
    "cc12_vs_rsplit_pareto.png",
    "retained_redundancy_vs_rsplit.png",
    "retained_redundancy_vs_cc12.png",
]
MERGE_REQUIRED_FILES = common.MERGE_REQUIRED_FILES


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    experiment_type: str
    score_id: str
    designation: str
    filtering_target: str
    drop_fraction: float | None
    output_filename: str
    expected_priority_group: str


@dataclass(frozen=True)
class MergeRecord:
    label: str
    variant_id: str
    source: str
    stream_path: Path | None
    merge_dir: Path | None
    candidate_count: int = 0
    ambiguous: bool = False


class AuditLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.lines: list[str] = []
        self.stage_start_times: dict[str, float] = {}

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self.lines.append(line)
        print(line, flush=True)

    def stage_start(self, index: int, total: int, name: str, total_items: int | None = None) -> None:
        label = f"{index:02d}/{total} {name}"
        self.stage_start_times[name] = time.monotonic()
        suffix = f"; total={total_items:,}" if total_items is not None else ""
        self.log(f"stage start: {label}{suffix}")

    def progress(self, name: str, completed: int, total: int) -> None:
        elapsed = max(time.monotonic() - self.stage_start_times.get(name, time.monotonic()), 1e-9)
        rate = completed / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * completed / total if total else 100.0
        eta = (total - completed) / rate if rate > 0 and completed < total else 0.0
        self.log(
            f"{name}: completed={completed:,}/{total:,} ({pct:.1f}%); "
            f"elapsed={format_seconds(elapsed)}; rate={rate:.2f}/s; eta={format_seconds(eta)}"
        )

    def stage_done(self, name: str, completed: int | None = None, total: int | None = None) -> None:
        elapsed = max(time.monotonic() - self.stage_start_times.get(name, time.monotonic()), 1e-9)
        count_text = ""
        if completed is not None and total is not None:
            count_text = f"; completed={completed:,}/{total:,}"
        self.log(f"stage complete: {name}{count_text}; elapsed={format_seconds(elapsed)}")

    def finalize(self) -> None:
        path = self.out_dir / "run.log"
        if path.exists() and path.stat().st_size > 0:
            return
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{sec:04.1f}s"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v6-dir", type=Path, default=None)
    parser.add_argument("--source-stream", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--historical-search-root", type=Path, default=None)
    parser.add_argument("--skip-stream-scan", action="store_true", help="Skip independent reflection-row scans of all streams.")
    parser.add_argument("--skip-plots", action="store_true", help="Write tables only; placeholder plot records are still logged.")
    parser.add_argument("--no-shell-recalculation", action="store_true", help="Do not run check_hkl/compare_hkl fallback when shell tables are missing or incompatible.")
    parser.add_argument("--hash-halfsets", action="store_true", help="Stream SHA256 hashes for hkl1/hkl2 half-set files.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    root = args.root.expanduser().resolve()
    args.root = root
    args.v6_dir = (args.v6_dir or root / "oridyn_v6_score_target_filter_map_20260716").expanduser().resolve()
    args.source_stream = (args.source_stream or root / "MFM300-VIII_cut_20-0_3.stream").expanduser().resolve()
    args.historical_search_root = (args.historical_search_root or root).expanduser().resolve()
    args.out_dir = (args.out_dir or root / AUDIT_DIRNAME).expanduser().resolve()
    return args


def safe_prepare_out_dir(out_dir: Path, protected: Iterable[Path]) -> None:
    for path in protected:
        resolved = path.resolve()
        if out_dir == resolved:
            raise SystemExit(f"--out-dir must not be an input path: {out_dir}")
        if resolved.is_dir():
            try:
                out_dir.relative_to(resolved)
            except ValueError:
                pass
            else:
                raise SystemExit(f"--out-dir must not be inside protected input directory: {resolved}")
    out_dir.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    return common.json_default(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    common.write_csv(path, rows, fieldnames=fieldnames)


def to_float(value: Any) -> float | None:
    return common.to_float(value)


def to_int(value: Any) -> int | None:
    return common.to_int(value)


def boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    return None


def finite(value: Any) -> float:
    number = to_float(value)
    return float(number) if number is not None else float("nan")


def percent_label(value: float) -> str:
    return f"{int(round(100 * value)):02d}"


def expected_variants() -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for score in DIAGNOSTIC_SCORES:
        for side in ["low50", "high50"]:
            variants.append(
                VariantSpec(
                    variant_id=f"diag_{score}_{side}",
                    experiment_type="diagnostic_low_high",
                    score_id=score,
                    designation=side,
                    filtering_target="unrestricted_per_hkl_half_split",
                    drop_fraction=None,
                    output_filename=f"diag_{score}_{side}.stream",
                    expected_priority_group="diagnostic",
                )
            )
    for target, drops in TARGET_DROPS.items():
        for score in FILTER_SCORES:
            for drop in drops:
                label = percent_label(drop)
                variants.append(
                    VariantSpec(
                        variant_id=f"filter_{target}_{score}_drop{label}",
                        experiment_type="targeted_filter",
                        score_id=score,
                        designation=f"drop{label}",
                        filtering_target=target,
                        drop_fraction=float(drop),
                        output_filename=f"filter_{target}_{score}_drop{label}.stream",
                        expected_priority_group=TARGET_LABELS[target],
                    )
                )
    return variants


def expected_random_variants() -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for seed in RANDOM_SEEDS:
        for side in ["low50", "high50"]:
            variants.append(
                VariantSpec(
                    variant_id=f"random_diag_seed{seed}_{side}",
                    experiment_type="diagnostic_random_control",
                    score_id=f"random_seed{seed}",
                    designation=side,
                    filtering_target="unrestricted_per_hkl_half_split",
                    drop_fraction=None,
                    output_filename=f"random_diag_seed{seed}_{side}.stream",
                    expected_priority_group="diagnostic_random",
                )
            )
    for drop in [0.05, 0.10, 0.20]:
        label = percent_label(drop)
        for seed in RANDOM_SEEDS:
            variants.append(
                VariantSpec(
                    variant_id=f"random_all_drop{label}_seed{seed}",
                    experiment_type="targeted_random_control",
                    score_id=f"random_seed{seed}",
                    designation=f"drop{label}",
                    filtering_target="all",
                    drop_fraction=float(drop),
                    output_filename=f"random_all_drop{label}_seed{seed}.stream",
                    expected_priority_group="target_A_random",
                )
            )
    for target in ["higheg", "matched"]:
        for drop in [0.20, 0.30, 0.40, 0.50]:
            label = percent_label(drop)
            for seed in RANDOM_SEEDS:
                variants.append(
                    VariantSpec(
                        variant_id=f"random_{target}_drop{label}_seed{seed}",
                        experiment_type="targeted_random_control",
                        score_id=f"random_seed{seed}",
                        designation=f"drop{label}",
                        filtering_target=target,
                        drop_fraction=float(drop),
                        output_filename=f"random_{target}_drop{label}_seed{seed}.stream",
                        expected_priority_group=f"target_{target}_random",
                    )
                )
    return variants


def all_expected_variants(include_random: bool) -> list[VariantSpec]:
    variants = expected_variants()
    if include_random:
        variants.extend(expected_random_variants())
    return variants


def variant_sort_key(variant_id: str) -> tuple[int, int, int, int, str]:
    expected = {spec.variant_id: idx for idx, spec in enumerate(all_expected_variants(include_random=True))}
    return (0, expected[variant_id], 0, 0, variant_id) if variant_id in expected else (1, 9999, 0, 0, variant_id)


def load_metadata(v6_dir: Path) -> dict[str, Any]:
    return {
        "plan_csv": common.read_csv_if_exists(v6_dir / "experiment_plan.csv"),
        "stream_manifest": common.read_csv_if_exists(v6_dir / "stream_manifest.csv"),
        "selection_summary": common.read_csv_if_exists(v6_dir / "per_variant_selection_summary.csv"),
        "stream_rewrite_qc": common.read_csv_if_exists(v6_dir / "stream_rewrite_qc.csv"),
        "scores_json": common.read_json(v6_dir / "scores.json") or [],
        "parameters_json": common.read_json(v6_dir / "parameters.json") or {},
        "validation_json": common.read_json(v6_dir / "validation.json") or {},
        "run_metadata_json": common.read_json(v6_dir / "run_metadata.json") or {},
        "cache_provenance_json": common.read_json(v6_dir / "cache_provenance.json") or {},
        "accepted_population_validation_json": common.read_json(v6_dir / "accepted_population_validation.json") or {},
        "random_control_manifest": common.read_csv_if_exists(v6_dir / "random_control_manifest.csv"),
        "selection_overlap": common.read_csv_if_exists(v6_dir / "selection_overlap.csv"),
        "score_correlations": common.read_csv_if_exists(v6_dir / "score_correlations.csv"),
    }


def dataframe_index(frame: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    if frame.empty or key not in frame.columns:
        return {}
    return {
        str(row[key]): row.to_dict()
        for _, row in frame.iterrows()
        if str(row.get(key, "")) != ""
    }


def has_full_population_random_controls(metadata: dict[str, Any]) -> bool:
    manifest = metadata.get("stream_manifest", pd.DataFrame())
    random_manifest = metadata.get("random_control_manifest", pd.DataFrame())
    plan = metadata.get("plan_csv", pd.DataFrame())
    for frame in [manifest, random_manifest, plan]:
        if isinstance(frame, pd.DataFrame) and not frame.empty and "variant_id" in frame.columns:
            if frame["variant_id"].astype(str).str.startswith("random_").any():
                return True
    return False


def restricted_cache_row_count(metadata: dict[str, Any]) -> int | None:
    candidates = [
        common.find_first_key(metadata.get("validation_json", {}), "cache_rows"),
        common.find_first_key(metadata.get("validation_json", {}), "score_cache_rows"),
        metadata.get("cache_provenance_json", {}).get("row_count") if isinstance(metadata.get("cache_provenance_json"), dict) else None,
    ]
    for value in candidates:
        number = to_int(value)
        if number is not None:
            return int(number)
    return None


def stream_path_from_manifest(v6_dir: Path, spec: VariantSpec, manifest_row: dict[str, Any] | None) -> Path:
    if manifest_row:
        output = str(manifest_row.get("output_stream") or "")
        if output:
            path = Path(output)
            return path if path.is_absolute() else v6_dir / path
    return v6_dir / spec.output_filename


def discover_v6_merges(v6_dir: Path, specs: list[VariantSpec], manifest_index: dict[str, dict[str, Any]]) -> tuple[list[MergeRecord], list[dict[str, Any]]]:
    records: list[MergeRecord] = []
    rows: list[dict[str, Any]] = []
    for spec in specs:
        stream_path = stream_path_from_manifest(v6_dir, spec, manifest_index.get(spec.variant_id))
        merge_dir, candidates = common.locate_merge_dir(v6_dir, stream_path.name)
        records.append(
            MergeRecord(
                label=spec.variant_id,
                variant_id=spec.variant_id,
                source="v6",
                stream_path=stream_path,
                merge_dir=merge_dir,
                candidate_count=len(candidates),
                ambiguous=len(candidates) != 1,
            )
        )
        rows.append(
            {
                "variant_id": spec.variant_id,
                "stream_path": str(stream_path),
                "stream_found": stream_path.is_file(),
                "selected_merge_dir": str(merge_dir or ""),
                "merge_candidate_count": len(candidates),
                "merge_ambiguous": len(candidates) != 1,
                "candidate_dirs": "; ".join(str(path) for path in candidates),
            }
        )
    return records, rows


def parse_all_merges(records: list[MergeRecord], workers: int, logger: AuditLogger) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    global_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def parse_one(record: MergeRecord) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        if record.merge_dir is None:
            return None, [], [{"variant_id": record.variant_id, "label": record.label, "stage": "merge_discovery", "severity": "error", "message": "no merge directory found"}]
        metrics, shells, _settings, local_warnings = common.parse_merge_outputs(record.merge_dir, record.label, record.variant_id)
        metrics["source"] = record.source
        metrics["stream_path"] = str(record.stream_path or "")
        metrics["merge_candidate_count"] = record.candidate_count
        metrics["merge_ambiguous"] = record.ambiguous
        for row in shells:
            row["source"] = record.source
        converted = [
            {
                "variant_id": record.variant_id,
                "label": record.label,
                "stage": "merge_parse",
                "severity": warning.get("severity", "warning"),
                "message": warning.get("message", ""),
                "path": warning.get("path", warning.get("source_file", "")),
            }
            for warning in local_warnings
        ]
        return metrics, shells, converted

    total = len(records)
    with ThreadPoolExecutor(max_workers=min(max(1, workers), max(1, total))) as executor:
        futures = {executor.submit(parse_one, record): record for record in records}
        completed = 0
        for future in as_completed(futures):
            metrics, shells, local_warnings = future.result()
            if metrics is not None:
                global_rows.append(metrics)
                shell_rows.extend(shells)
            warnings.extend(local_warnings)
            completed += 1
            if completed == total or completed % max(1, total // 5) == 0:
                logger.progress("global metric extraction", completed, total)
    global_rows.sort(key=lambda row: variant_sort_key(str(row.get("variant_id", row.get("label", "")))))
    shell_rows.sort(key=lambda row: (variant_sort_key(str(row.get("variant_id", row.get("label", "")))), int(row.get("shell_index") or 0)))
    return global_rows, shell_rows, warnings


def stream_validation(records: list[MergeRecord], metadata: dict[str, Any], workers: int, skip_scan: bool, logger: AuditLogger) -> list[dict[str, Any]]:
    manifest_index = dataframe_index(metadata["stream_manifest"], "variant_id")
    rewrite_index = dataframe_index(metadata["stream_rewrite_qc"], "variant_id")
    rows: list[dict[str, Any]] = []
    scan_results: dict[str, dict[str, Any]] = {}
    if not skip_scan:
        jobs = [(record.variant_id, record.stream_path) for record in records if record.stream_path is not None]
        with ThreadPoolExecutor(max_workers=min(max(1, workers), max(1, len(jobs)))) as executor:
            futures = {executor.submit(common.count_stream_reflections, path): variant for variant, path in jobs if path is not None}
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                scan_results[futures[future]] = future.result()
                completed += 1
                if completed == total or completed % max(1, total // 5) == 0:
                    logger.progress("stream validation", completed, total)
    for record in records:
        manifest = manifest_index.get(record.variant_id, {})
        rewrite = rewrite_index.get(record.variant_id, {})
        path = record.stream_path
        scan = scan_results.get(record.variant_id, {})
        expected_retained = first_nonempty(rewrite.get("kept_observations"), manifest.get("retained_count"))
        scan_count = scan.get("reflection_rows", "")
        count_matches = ""
        if scan_count != "" and expected_retained != "":
            count_matches = to_int(scan_count) == to_int(expected_retained)
        row = {
            "variant_id": record.variant_id,
            "stream_path": str(path or ""),
            "stream_exists": bool(path and path.is_file()),
            "stream_size_bytes": path.stat().st_size if path and path.is_file() else "",
            "manifest_status": manifest.get("status", ""),
            "rewrite_status": rewrite.get("status", ""),
            "requested_removals": first_nonempty(rewrite.get("requested_removals"), manifest.get("selected_or_removed_count")),
            "removed_observations": first_nonempty(rewrite.get("removed_observations"), manifest.get("selected_or_removed_count")),
            "expected_retained_observations": expected_retained,
            "stream_scan_reflection_rows": scan_count,
            "independent_count_matches": count_matches,
            "source_order_preserved": rewrite.get("source_order_preserved", ""),
            "all_requested_keys_found_exactly_once": rewrite.get("all_requested_keys_found_exactly_once", ""),
            "stream_reflection_row_difference_equals_requested_removals": rewrite.get("stream_reflection_row_difference_equals_requested_removals", ""),
        }
        rows.append(row)
    return rows


def first_nonempty(*values: Any) -> Any:
    return common.first_nonempty(*values)


def aggregate_variant_qc(v6_dir: Path, expected_ids: set[str], logger: AuditLogger) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    hkl_path = v6_dir / "per_variant_per_hkl_qc.csv"
    block_path = v6_dir / "per_variant_per_block_qc.csv"
    hkl: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    if hkl_path.is_file():
        usecols = None
        header = pd.read_csv(hkl_path, nrows=0).columns.tolist()
        preferred = [
            "variant_id",
            "experiment_type",
            "filtering_target",
            "score_id",
            "h",
            "k",
            "l",
            "n_observations",
            "n_selected",
            "n_removed",
            "n_omitted_middle",
            "n_eligible",
            "n_eligible_retained",
            "n_high_eg",
            "n_blocks",
            "n_block_observations",
            "actionable",
            "validation_passed",
        ]
        usecols = [name for name in preferred if name in header]
        total_chunks = 0
        for chunk in pd.read_csv(hkl_path, usecols=usecols, chunksize=250_000, low_memory=False):
            total_chunks += 1
            subset = chunk.loc[chunk["variant_id"].astype(str).isin(expected_ids)].copy()
            if subset.empty:
                continue
            for variant, group in subset.groupby(subset["variant_id"].astype(str), sort=False):
                record = hkl[variant]
                record["hkl_qc_rows"] += len(group)
                for col in [
                    "n_observations",
                    "n_selected",
                    "n_removed",
                    "n_omitted_middle",
                    "n_eligible",
                    "n_eligible_retained",
                    "n_high_eg",
                    "n_blocks",
                    "n_block_observations",
                ]:
                    if col in group.columns:
                        record[f"sum_{col}"] += pd.to_numeric(group[col], errors="coerce").fillna(0).sum()
                if "actionable" in group.columns:
                    actionable = group["actionable"].astype(str).str.lower().isin(["true", "1", "yes"])
                    record["actionable_hkl_count"] += int(actionable.sum())
                    record["skipped_hkl_count"] += int((~actionable).sum())
                    if "n_observations" in group.columns:
                        record["skipped_observations"] += pd.to_numeric(group.loc[~actionable, "n_observations"], errors="coerce").fillna(0).sum()
                if "validation_passed" in group.columns:
                    record["hkl_validation_failed_rows"] += int((~group["validation_passed"].astype(str).str.lower().isin(["true", "1", "yes"])).sum())
                remain_col = "n_eligible_retained" if "n_eligible_retained" in group.columns else "n_selected"
                if remain_col in group.columns:
                    values = pd.to_numeric(group[remain_col], errors="coerce").dropna()
                    if not values.empty:
                        record["min_remaining_observations_per_group"] = min(record.get("min_remaining_observations_per_group", float("inf")), float(values.min()))
                        record["max_remaining_observations_per_group"] = max(record.get("max_remaining_observations_per_group", float("-inf")), float(values.max()))
    block: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    if block_path.is_file():
        header = pd.read_csv(block_path, nrows=0).columns.tolist()
        usecols = [name for name in ["variant_id", "h", "k", "l", "block_id", "block_size", "n_removed", "n_retained_in_block", "validation_passed"] if name in header]
        for chunk in pd.read_csv(block_path, usecols=usecols, chunksize=250_000, low_memory=False):
            subset = chunk.loc[chunk["variant_id"].astype(str).isin(expected_ids)].copy()
            if subset.empty:
                continue
            for variant, group in subset.groupby(subset["variant_id"].astype(str), sort=False):
                record = block[variant]
                record["block_qc_rows"] += len(group)
                record["actionable_block_count"] += len(group)
                for col in ["n_removed", "n_retained_in_block"]:
                    if col in group.columns:
                        record[f"sum_{col}"] += pd.to_numeric(group[col], errors="coerce").fillna(0).sum()
                if "validation_passed" in group.columns:
                    record["block_validation_failed_rows"] += int((~group["validation_passed"].astype(str).str.lower().isin(["true", "1", "yes"])).sum())
                if "block_size" in group.columns:
                    sizes = pd.to_numeric(group["block_size"], errors="coerce").dropna()
                    if not sizes.empty:
                        record["block_size_min"] = min(record.get("block_size_min", float("inf")), float(sizes.min()))
                        record["block_size_max"] = max(record.get("block_size_max", float("-inf")), float(sizes.max()))
                if "n_retained_in_block" in group.columns:
                    retained = pd.to_numeric(group["n_retained_in_block"], errors="coerce").dropna()
                    if not retained.empty:
                        record["min_remaining_observations_per_group"] = min(record.get("min_remaining_observations_per_group", float("inf")), float(retained.min()))
                        record["max_remaining_observations_per_group"] = max(record.get("max_remaining_observations_per_group", float("-inf")), float(retained.max()))
    for mapping in [hkl, block]:
        for record in mapping.values():
            for key, value in list(record.items()):
                if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                    record[key] = ""
                elif isinstance(value, np.generic):
                    record[key] = value.item()
    return dict(hkl), dict(block), warnings


def selection_manifest_counts(v6_dir: Path, expected_ids: set[str], logger: AuditLogger) -> dict[str, dict[str, Any]]:
    path = v6_dir / "selected_observations.csv.gz"
    counts: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    if not path.is_file():
        return {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = pd.read_csv(handle, chunksize=250_000, low_memory=False)
        for chunk in reader:
            if "variant_id" not in chunk.columns:
                continue
            subset = chunk.loc[chunk["variant_id"].astype(str).isin(expected_ids)]
            if subset.empty:
                continue
            for variant, group in subset.groupby(subset["variant_id"].astype(str), sort=False):
                record = counts[variant]
                record["selection_manifest_rows"] += len(group)
                if "state" in group.columns:
                    states = group["state"].astype(str)
                    record["removed_manifest_rows"] += int((states == "removed_by_targeted_filter").sum())
                    record["eligible_retained_manifest_rows"] += int((states == "eligible_retained_after_targeted_filter").sum())
                    record["diagnostic_selected_manifest_rows"] += int(states.str.contains("diagnostic_.*half", regex=True).sum())
                    record["diagnostic_omitted_manifest_rows"] += int(states.str.contains("omitted", regex=True).sum())
    return {key: dict(value) for key, value in counts.items()}


def build_selection_counts(
    specs: list[VariantSpec],
    metadata: dict[str, Any],
    hkl_qc: dict[str, dict[str, Any]],
    block_qc: dict[str, dict[str, Any]],
    manifest_counts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_index = dataframe_index(metadata["stream_manifest"], "variant_id")
    selection_index = dataframe_index(metadata["selection_summary"], "variant_id")
    rewrite_index = dataframe_index(metadata["stream_rewrite_qc"], "variant_id")
    validation = metadata["validation_json"]
    source_rows = common.find_first_key(validation, "source_reflection_rows") or ""
    cache_rows = (
        common.find_first_key(validation, "score_cache_rows")
        or common.find_first_key(validation, "cache_rows")
        or (metadata.get("cache_provenance_json", {}) or {}).get("row_count")
        or ""
    )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        manifest = manifest_index.get(spec.variant_id, {})
        summary = selection_index.get(spec.variant_id, {})
        rewrite = rewrite_index.get(spec.variant_id, {})
        hkl = hkl_qc.get(spec.variant_id, {})
        block = block_qc.get(spec.variant_id, {})
        selected = manifest_counts.get(spec.variant_id, {})
        retained = first_nonempty(rewrite.get("kept_observations"), manifest.get("retained_count"), summary.get("retained_count"))
        removed = first_nonempty(rewrite.get("removed_observations"), rewrite.get("requested_removals"), manifest.get("selected_or_removed_count"), summary.get("selected_or_removed_count"))
        source_count = source_rows or rewrite.get("total_reflection_rows_seen", "")
        row = {
            "variant_id": spec.variant_id,
            "experiment_type": spec.experiment_type,
            "score_id": spec.score_id,
            "score_formula": SCORE_FORMULAS.get(spec.score_id, ""),
            "filtering_target": spec.filtering_target,
            "target_group": spec.expected_priority_group,
            "drop_fraction_nominal": spec.drop_fraction if spec.drop_fraction is not None else "",
            "source_accepted_observation_count": source_count,
            "accepted_population_count": first_nonempty(manifest.get("accepted_population_count"), summary.get("accepted_population_count"), cache_rows),
            "source_reflection_row_count": first_nonempty(manifest.get("source_reflection_row_count"), summary.get("source_reflection_row_count"), source_count),
            "v6_cache_or_scored_observation_count": cache_rows,
            "retained_observation_count": retained,
            "removed_or_selected_count": removed,
            "retained_fraction": first_nonempty(manifest.get("retained_fraction"), summary.get("retained_fraction")),
            "removed_fraction_of_all_accepted_observations": first_nonempty(manifest.get("removed_fraction_of_accepted_population"), summary.get("removed_fraction_of_accepted_population"), manifest.get("global_accepted_observation_fraction_removed"), summary.get("global_accepted_observation_fraction_removed")),
            "removed_fraction_of_accepted_population": first_nonempty(manifest.get("removed_fraction_of_accepted_population"), summary.get("removed_fraction_of_accepted_population"), manifest.get("global_accepted_observation_fraction_removed"), summary.get("global_accepted_observation_fraction_removed")),
            "removed_fraction_of_source_rows": first_nonempty(manifest.get("removed_fraction_of_source_rows"), summary.get("removed_fraction_of_source_rows")),
            "removed_fraction_of_eligible_population": first_nonempty(manifest.get("removed_fraction_of_eligible_population"), summary.get("removed_fraction_of_eligible_population")),
            "removed_fraction_of_actionable_population": first_nonempty(manifest.get("removed_fraction_of_actionable_population"), summary.get("removed_fraction_of_actionable_population")),
            "exact_signed_hkl_count": hkl.get("hkl_qc_rows", ""),
            "eligible_observation_count": first_nonempty(hkl.get("sum_n_eligible"), hkl.get("sum_n_high_eg"), hkl.get("sum_n_block_observations")),
            "actionable_observation_count": first_nonempty(manifest.get("actionable_observation_count"), summary.get("actionable_observation_count"), hkl.get("sum_n_block_observations"), hkl.get("sum_n_eligible")),
            "actionable_hkl_count": hkl.get("actionable_hkl_count", ""),
            "actionable_block_count": block.get("actionable_block_count", ""),
            "skipped_hkl_count": hkl.get("skipped_hkl_count", ""),
            "skipped_block_count": "",
            "skipped_observation_count": hkl.get("skipped_observations", ""),
            "min_remaining_observations_per_group": first_nonempty(block.get("min_remaining_observations_per_group"), hkl.get("min_remaining_observations_per_group")),
            "max_remaining_observations_per_group": first_nonempty(block.get("max_remaining_observations_per_group"), hkl.get("max_remaining_observations_per_group")),
            "selection_manifest_rows": selected.get("selection_manifest_rows", ""),
            "removed_manifest_rows": selected.get("removed_manifest_rows", ""),
            "eligible_retained_manifest_rows": selected.get("eligible_retained_manifest_rows", ""),
            "diagnostic_selected_manifest_rows": selected.get("diagnostic_selected_manifest_rows", ""),
            "diagnostic_omitted_manifest_rows": selected.get("diagnostic_omitted_manifest_rows", ""),
            "hkl_validation_failed_rows": hkl.get("hkl_validation_failed_rows", 0),
            "block_validation_failed_rows": block.get("block_validation_failed_rows", 0),
            "block_size_min": block.get("block_size_min", ""),
            "block_size_max": block.get("block_size_max", ""),
        }
        rows.append(row)
    return rows


def diagnostic_pair_results(
    specs: list[VariantSpec],
    global_rows: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
    selection_counts: list[dict[str, Any]],
    selection_overlap: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_by_variant = {str(row.get("variant_id")): row for row in global_rows}
    shell_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shell_rows:
        shell_by_variant[str(row.get("variant_id"))].append(row)
    counts_by_variant = {row["variant_id"]: row for row in selection_counts}
    overlap_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not selection_overlap.empty:
        left_col = "left_variant_id" if "left_variant_id" in selection_overlap.columns else ("left_variant" if "left_variant" in selection_overlap.columns else "")
        right_col = "right_variant_id" if "right_variant_id" in selection_overlap.columns else ("right_variant" if "right_variant" in selection_overlap.columns else "")
        if left_col and right_col:
            for _, row in selection_overlap.iterrows():
                left = str(row[left_col])
                right = str(row[right_col])
                overlap_lookup[(left, right)] = row.to_dict()
                overlap_lookup[(right, left)] = row.to_dict()
    rows: list[dict[str, Any]] = []
    shell_delta_rows: list[dict[str, Any]] = []
    for score in DIAGNOSTIC_SCORES:
        low = f"diag_{score}_low50"
        high = f"diag_{score}_high50"
        low_g = global_by_variant.get(low, {})
        high_g = global_by_variant.get(high, {})
        cc_delta = diff(low_g.get("cc12"), high_g.get("cc12"))
        rsplit_delta = diff(low_g.get("rsplit"), high_g.get("rsplit"))
        snr_delta = diff(low_g.get("snr"), high_g.get("snr"))
        comp_delta = diff(low_g.get("completeness"), high_g.get("completeness"))
        red_delta = diff(low_g.get("redundancy"), high_g.get("redundancy"))
        low_count = to_float(counts_by_variant.get(low, {}).get("removed_or_selected_count"))
        high_count = to_float(counts_by_variant.get(high, {}).get("removed_or_selected_count"))
        overlap = overlap_lookup.get((low, high), {})
        low_shells = {common.boundary_key(row): row for row in shell_by_variant.get(low, []) if common.boundary_key(row) is not None}
        high_shells = {common.boundary_key(row): row for row in shell_by_variant.get(high, []) if common.boundary_key(row) is not None}
        common_keys = sorted(set(low_shells) & set(high_shells))
        cc_shell_deltas = []
        rs_shell_deltas = []
        for idx, key in enumerate(common_keys, start=1):
            low_s = low_shells[key]
            high_s = high_shells[key]
            cc_s = diff(low_s.get("cc12"), high_s.get("cc12"))
            rs_s = diff(low_s.get("rsplit"), high_s.get("rsplit"))
            cc_shell_deltas.append(cc_s)
            rs_shell_deltas.append(rs_s)
            shell_delta_rows.append(
                {
                    "score_id": score,
                    "low_variant_id": low,
                    "high_variant_id": high,
                    "shell_index": idx,
                    "min_invnm": key[0],
                    "max_invnm": key[1],
                    "shell_low_resolution_A": 10.0 / key[0] if key[0] else "",
                    "shell_high_resolution_A": 10.0 / key[1] if key[1] else "",
                    "delta_cc12_low_minus_high": cc_s,
                    "delta_rsplit_low_minus_high": rs_s,
                    "delta_snr_low_minus_high": diff(low_s.get("snr"), high_s.get("snr")),
                    "delta_completeness_low_minus_high": diff(low_s.get("completeness"), high_s.get("completeness")),
                    "delta_redundancy_low_minus_high": diff(low_s.get("redundancy"), high_s.get("redundancy")),
                }
            )
        cc_numeric = [x for x in cc_shell_deltas if isinstance(x, float) and math.isfinite(x)]
        rs_numeric = [x for x in rs_shell_deltas if isinstance(x, float) and math.isfinite(x)]
        harmful_balance = any(abs(value) > 0.2 for value in [snr_delta if isinstance(snr_delta, float) else 0.0, comp_delta if isinstance(comp_delta, float) else 0.0])
        rows.append(
            {
                "score_id": score,
                "low_variant_id": low,
                "high_variant_id": high,
                "low_count": low_count if low_count is not None else "",
                "high_count": high_count if high_count is not None else "",
                "low_high_global_counts_equal": low_count == high_count if low_count is not None and high_count is not None else "",
                "low_high_selection_jaccard": first_nonempty(overlap.get("jaccard"), overlap.get("jaccard_index")),
                "low_high_overlap_count": first_nonempty(overlap.get("overlap_count"), overlap.get("intersection_count")),
                "delta_cc12_low_minus_high": cc_delta,
                "delta_rsplit_low_minus_high": rsplit_delta,
                "delta_snr_low_minus_high": snr_delta,
                "delta_completeness_low_minus_high": comp_delta,
                "delta_redundancy_low_minus_high": red_delta,
                "shells_compared": len(common_keys),
                "cc12_low_favored_shells": sum(1 for x in cc_numeric if x > 0),
                "rsplit_low_favored_shells": sum(1 for x in rs_numeric if x < 0),
                "median_delta_cc12_shell": float(np.median(cc_numeric)) if cc_numeric else "",
                "median_delta_rsplit_shell": float(np.median(rs_numeric)) if rs_numeric else "",
                "harmful_snr_or_completeness_imbalance": harmful_balance,
            }
        )
    rows = rank_diagnostics(rows)
    return rows, shell_delta_rows


def diff(left: Any, right: Any) -> float | str:
    lval = to_float(left)
    rval = to_float(right)
    if lval is None or rval is None:
        return ""
    return float(lval - rval)


def rank_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        cc = finite(row.get("delta_cc12_low_minus_high"))
        rs = finite(row.get("delta_rsplit_low_minus_high"))
        cc_shell = finite(row.get("median_delta_cc12_shell"))
        rs_shell = finite(row.get("median_delta_rsplit_shell"))
        return (cc, -rs, cc_shell, -rs_shell)

    ordered = sorted(rows, key=key, reverse=True)
    ranks = {row["score_id"]: idx for idx, row in enumerate(ordered, start=1)}
    for row in rows:
        row["diagnostic_rank"] = ranks[row["score_id"]]
    rows.sort(key=lambda row: int(row["diagnostic_rank"]))
    return rows


def shell_boundaries_and_deltas(
    shell_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    full_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    boundaries: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shell_rows:
        rows_by_variant[str(row.get("variant_id") or row.get("label"))].append(row)
    scheme_digest_by_variant: dict[str, str] = {}
    for variant, rows in rows_by_variant.items():
        keys = [common.boundary_key(row) for row in sorted(rows, key=lambda item: int(item.get("shell_index") or 0))]
        keys = [key for key in keys if key is not None]
        digest = hashlib.sha1(json.dumps(keys, sort_keys=True).encode("utf-8")).hexdigest()[:16] if keys else ""
        scheme_digest_by_variant[variant] = digest
        for idx, key in enumerate(keys, start=1):
            boundaries.append(
                {
                    "variant_id": variant,
                    "shell_index": idx,
                    "min_invnm": key[0],
                    "max_invnm": key[1],
                    "shell_low_resolution_A": 10.0 / key[0] if key[0] else "",
                    "shell_high_resolution_A": 10.0 / key[1] if key[1] else "",
                    "shell_scheme_id": digest,
                    "shell_count_for_variant": len(keys),
                }
            )
    ref_rows = rows_by_variant.get(full_label, [])
    ref_by_key = {common.boundary_key(row): row for row in ref_rows if common.boundary_key(row) is not None}
    deltas: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant, rows in rows_by_variant.items():
        if variant == full_label:
            continue
        metric_deltas = {"cc12": [], "rsplit": [], "snr": [], "completeness": [], "redundancy": []}
        for row in sorted(rows, key=lambda item: int(item.get("shell_index") or 0)):
            key = common.boundary_key(row)
            ref = ref_by_key.get(key)
            out = dict(row)
            out["reference_variant_id"] = full_label if ref else ""
            for metric in metric_deltas:
                value = diff(row.get(metric), ref.get(metric) if ref else None)
                out[f"delta_{metric}_vs_full"] = value
                if isinstance(value, float) and math.isfinite(value):
                    metric_deltas[metric].append(value)
            deltas.append(out)
        cc = metric_deltas["cc12"]
        rs = metric_deltas["rsplit"]
        outer = deltas[-5:] if len(rows) >= 5 else deltas[-len(rows) :]
        summaries.append(
            {
                "variant_id": variant,
                "shell_scheme_id": scheme_digest_by_variant.get(variant, ""),
                "shell_count": len(rows),
                "shell_boundaries_match_full": bool(ref_by_key) and all(common.boundary_key(row) in ref_by_key for row in rows),
                "improved_cc12_shell_count": sum(1 for value in cc if value > 0),
                "worsened_cc12_shell_count": sum(1 for value in cc if value < 0),
                "median_delta_cc12": float(np.median(cc)) if cc else "",
                "mean_delta_cc12": float(np.mean(cc)) if cc else "",
                "improved_rsplit_shell_count": sum(1 for value in rs if value < 0),
                "worsened_rsplit_shell_count": sum(1 for value in rs if value > 0),
                "median_delta_rsplit": float(np.median(rs)) if rs else "",
                "mean_delta_rsplit": float(np.mean(rs)) if rs else "",
                "best_shell_by_cc12": best_shell(deltas, variant, "delta_cc12_vs_full", maximize=True),
                "worst_shell_by_cc12": best_shell(deltas, variant, "delta_cc12_vs_full", maximize=False),
                "outermost_five_mean_delta_cc12": mean_of_rows(outer, "delta_cc12_vs_full"),
                "outermost_five_mean_delta_rsplit": mean_of_rows(outer, "delta_rsplit_vs_full"),
                "outermost_five_mean_delta_snr": mean_of_rows(outer, "delta_snr_vs_full"),
                "low_resolution_third_mean_delta_cc12": third_mean(deltas, variant, "delta_cc12_vs_full", 0),
                "middle_resolution_third_mean_delta_cc12": third_mean(deltas, variant, "delta_cc12_vs_full", 1),
                "high_resolution_third_mean_delta_cc12": third_mean(deltas, variant, "delta_cc12_vs_full", 2),
            }
        )
    return boundaries, deltas, summaries


def mean_of_rows(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else ""


def best_shell(rows: list[dict[str, Any]], variant: str, key: str, maximize: bool) -> Any:
    subset = [row for row in rows if str(row.get("variant_id")) == variant and to_float(row.get(key)) is not None]
    if not subset:
        return ""
    row = max(subset, key=lambda item: float(item[key])) if maximize else min(subset, key=lambda item: float(item[key]))
    return row.get("shell_index", "")


def third_mean(rows: list[dict[str, Any]], variant: str, key: str, third: int) -> float | str:
    subset = [row for row in rows if str(row.get("variant_id")) == variant]
    subset.sort(key=lambda item: int(item.get("shell_index") or 0))
    if not subset:
        return ""
    splits = np.array_split(subset, 3)
    if third >= len(splits):
        return ""
    values = [to_float(row.get(key)) for row in splits[third]]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else ""


def global_deltas(global_rows: list[dict[str, Any]], full_label: str, selection_counts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {str(row.get("variant_id") or row.get("label")): row for row in global_rows}
    full = by_variant.get(full_label, {})
    counts = {row["variant_id"]: row for row in selection_counts}
    rows: list[dict[str, Any]] = []
    for variant, row in by_variant.items():
        if variant == full_label or not variant.startswith(("diag_", "filter_")):
            continue
        count = counts.get(variant, {})
        out = {
            "variant_id": variant,
            "experiment_type": "diagnostic_low_high" if variant.startswith("diag_") else "targeted_filter",
            "score_id": parse_score_from_variant(variant),
            "filtering_target": parse_target_from_variant(variant),
            "removed_or_selected_count": count.get("removed_or_selected_count", ""),
            "retained_observation_count": count.get("retained_observation_count", ""),
            "removed_fraction_of_all_accepted_observations": count.get("removed_fraction_of_all_accepted_observations", ""),
            "removed_fraction_of_accepted_population": count.get("removed_fraction_of_accepted_population", count.get("removed_fraction_of_all_accepted_observations", "")),
            "removed_fraction_of_source_rows": count.get("removed_fraction_of_source_rows", ""),
            "removed_fraction_of_eligible_population": count.get("removed_fraction_of_eligible_population", ""),
            "removed_fraction_of_actionable_population": count.get("removed_fraction_of_actionable_population", ""),
        }
        for metric in ["cc12", "rsplit", "snr", "completeness", "redundancy"]:
            out[metric] = row.get(metric, "")
            out[f"delta_{metric}_vs_full"] = diff(row.get(metric), full.get(metric))
        rows.append(out)
    rows.sort(key=lambda row: variant_sort_key(row["variant_id"]))
    return rows


def parse_score_from_variant(variant_id: str) -> str:
    if variant_id.startswith("diag_"):
        return variant_id[5:].rsplit("_", 1)[0]
    if variant_id.startswith("filter_"):
        for target in ["matched", "higheg", "all"]:
            prefix = f"filter_{target}_"
            if variant_id.startswith(prefix):
                return variant_id[len(prefix) :].rsplit("_drop", 1)[0]
    return ""


def parse_target_from_variant(variant_id: str) -> str:
    if variant_id.startswith("diag_"):
        return "diagnostic"
    for target in ["matched", "higheg", "all"]:
        if variant_id.startswith(f"filter_{target}_"):
            return target
    return ""


def filter_global_metrics(global_delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in global_delta_rows if row.get("experiment_type") == "targeted_filter"]


def filter_sweeps(filter_rows: list[dict[str, Any]], shell_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shell_by_variant = {row["variant_id"]: row for row in shell_summary}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filter_rows:
        groups[(str(row.get("filtering_target")), str(row.get("score_id")))].append(row)
    out: list[dict[str, Any]] = []
    for (target, score), rows in groups.items():
        rows.sort(key=lambda item: finite(item.get("removed_fraction_of_all_accepted_observations")))
        cc_values = [finite(row.get("cc12")) for row in rows]
        rs_values = [finite(row.get("rsplit")) for row in rows]
        best_cc = max(rows, key=lambda item: finite(item.get("cc12"))) if rows else {}
        best_rs = min(rows, key=lambda item: finite(item.get("rsplit"))) if rows else {}
        for row in rows:
            shell = shell_by_variant.get(row["variant_id"], {})
            out.append(
                {
                    "filtering_target": target,
                    "target_group": TARGET_LABELS.get(target, target),
                    "score_id": score,
                    "variant_id": row["variant_id"],
                    "actual_removed_fraction": row.get("removed_fraction_of_all_accepted_observations", ""),
                    "removed_or_selected_count": row.get("removed_or_selected_count", ""),
                    "cc12": row.get("cc12", ""),
                    "rsplit": row.get("rsplit", ""),
                    "snr": row.get("snr", ""),
                    "redundancy": row.get("redundancy", ""),
                    "delta_cc12_vs_full": row.get("delta_cc12_vs_full", ""),
                    "delta_rsplit_vs_full": row.get("delta_rsplit_vs_full", ""),
                    "shellwise_improved_rsplit_count": shell.get("improved_rsplit_shell_count", ""),
                    "shellwise_worsened_rsplit_count": shell.get("worsened_rsplit_shell_count", ""),
                    "best_fraction_by_cc12": row["variant_id"] == best_cc.get("variant_id"),
                    "best_fraction_by_rsplit": row["variant_id"] == best_rs.get("variant_id"),
                    "cc12_monotonic_non_decreasing_over_sweep": monotonic(cc_values, increasing=True),
                    "rsplit_monotonic_non_increasing_over_sweep": monotonic(rs_values, increasing=False),
                    "interior_optimum_by_cc12": best_cc.get("variant_id") not in {rows[0]["variant_id"], rows[-1]["variant_id"]} if len(rows) > 2 else False,
                    "interior_optimum_by_rsplit": best_rs.get("variant_id") not in {rows[0]["variant_id"], rows[-1]["variant_id"]} if len(rows) > 2 else False,
                }
            )
    out.sort(key=lambda row: (row["filtering_target"], row["score_id"], finite(row["actual_removed_fraction"])))
    return out


def monotonic(values: list[float], increasing: bool) -> bool | str:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return ""
    if increasing:
        return all(b >= a for a, b in zip(clean, clean[1:]))
    return all(b <= a for a, b in zip(clean, clean[1:]))


def score_family_comparisons(diagnostic_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]], shell_summary: list[dict[str, Any]], selection_overlap: pd.DataFrame) -> list[dict[str, Any]]:
    diag_by_score = {row["score_id"]: row for row in diagnostic_rows}
    best_filter_by_score: dict[str, dict[str, Any]] = {}
    for row in filter_rows:
        score = str(row.get("score_id"))
        current = best_filter_by_score.get(score)
        if current is None or finite(row.get("delta_cc12_vs_full")) > finite(current.get("delta_cc12_vs_full")):
            best_filter_by_score[score] = row
    comparisons = [
        ("first_vs_second_coupling_moment", "eg_cmean", "eg_c2mean"),
        ("normalized_vs_unnormalized_second_moment", "eg_c2mean", "eg_m2"),
        ("density_exponent_0_vs_1", "eg_cmean", "eg_d1_cmean"),
        ("density_exponent_1_vs_2", "eg_d1_cmean", "eg_d2_cmean"),
        ("density_exponent_2_vs_3", "eg_d2_cmean", "eg_d3_cmean"),
        ("strong_link_density_vs_first_moment_density", "eg_d2_c2mean", "eg_d2_cmean"),
    ]
    rows: list[dict[str, Any]] = []
    for name, left, right in comparisons:
        left_diag = diag_by_score.get(left, {})
        right_diag = diag_by_score.get(right, {})
        left_filter = best_filter_by_score.get(left, {})
        right_filter = best_filter_by_score.get(right, {})
        rows.append(
            {
                "comparison": name,
                "left_score": left,
                "right_score": right,
                "left_formula": SCORE_FORMULAS.get(left, ""),
                "right_formula": SCORE_FORMULAS.get(right, ""),
                "delta_diagnostic_cc12_separation_left_minus_right": diff(left_diag.get("delta_cc12_low_minus_high"), right_diag.get("delta_cc12_low_minus_high")),
                "delta_diagnostic_rsplit_separation_left_minus_right": diff(left_diag.get("delta_rsplit_low_minus_high"), right_diag.get("delta_rsplit_low_minus_high")),
                "left_best_filter_variant": left_filter.get("variant_id", ""),
                "right_best_filter_variant": right_filter.get("variant_id", ""),
                "delta_best_filter_cc12_left_minus_right": diff(left_filter.get("cc12"), right_filter.get("cc12")),
                "delta_best_filter_rsplit_left_minus_right": diff(left_filter.get("rsplit"), right_filter.get("rsplit")),
                "selection_overlap_source": "selection_overlap.csv" if not selection_overlap.empty else "not_available",
            }
        )
    return rows


def discover_v5_equivalence(root: Path, global_rows: list[dict[str, Any]], shell_rows: list[dict[str, Any]], v6_selection_counts: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> tuple[list[MergeRecord], list[dict[str, Any]]]:
    v5_dir = root / "oridyn_v5_p_lambda_screen_20260716"
    rows: list[dict[str, Any]] = []
    records: list[MergeRecord] = []
    if not v5_dir.is_dir():
        rows.append({"formula": "Eg * D^3 * M / U", "classification": "not_comparable", "reason": "v5 directory not found"})
        return records, rows
    equivalent = common.find_v5_equivalent(v5_dir)
    if not equivalent.get("variant_id"):
        rows.append({"formula": "Eg * D^3 * M / U", "classification": "not_comparable", "reason": equivalent.get("match_method", "not found")})
        return records, rows
    variant = str(equivalent["variant_id"])
    stream = v5_dir / str(equivalent["stream_name"])
    merge_dir, candidates = common.locate_merge_dir(v5_dir, stream.name)
    records.append(MergeRecord("v5_equivalent_p1", variant, "v5_equivalence", stream, merge_dir, len(candidates), len(candidates) != 1))
    v6_counts = {row["variant_id"]: row for row in v6_selection_counts}
    v6_d3 = v6_counts.get("filter_matched_eg_d3_cmean_drop30", {})
    v6_removed = to_float(v6_d3.get("removed_or_selected_count"))
    v5_removed = to_float(equivalent.get("summary_actual_removal_count"))
    sentinel = {}
    if metadata:
        sentinel = (metadata.get("validation_json", {}) or {}).get("v5_sentinel_equivalence", {}) or {}
    if sentinel.get("classification"):
        classification = str(sentinel.get("classification"))
    elif v6_removed == v5_removed:
        classification = "same_eligibility_different_ranking"
    elif v6_removed is not None and v5_removed is not None:
        classification = "different_source_population"
    else:
        classification = "not_comparable"
    rows.append(
        {
            "v6_variant_id": "filter_matched_eg_d3_cmean_drop30",
            "v5_variant_id": variant,
            "v5_stream": str(stream),
            "v5_merge_dir": str(merge_dir or ""),
            "formula_v6": SCORE_FORMULAS["eg_d3_cmean"],
            "formula_v5": equivalent.get("formula", ""),
            "v5_match_method": equivalent.get("match_method", ""),
            "v6_removed_count": v6_removed if v6_removed is not None else "",
            "v5_removed_count": v5_removed if v5_removed is not None else "",
            "v5_expected_actionable_observations": V5_MATCHED_EXPECTED["actionable_observations"],
            "v5_expected_blocks": V5_MATCHED_EXPECTED["actionable_blocks"],
            "v5_expected_removed": V5_MATCHED_EXPECTED["removed"],
            "classification": classification,
            "exact_key_comparison_source": "validation.json:v5_sentinel_equivalence" if sentinel else "",
            "exact_key_missing_count": sentinel.get("missing_v5_keys_from_v6", ""),
            "exact_key_unknown_old_count": sentinel.get("v5_keys_not_in_full_cache", ""),
            "redundancy_difference_explanation": "V6 and V5 remove different numbers of observations if v6_removed_count != v5_removed_count; compare exact keys when selection sets are loaded.",
        }
    )
    return records, rows


def discover_random_controls(search_root: Path, v6_records: list[MergeRecord], v6_selection_counts: list[dict[str, Any]]) -> tuple[list[MergeRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    v6_counts = {row["variant_id"]: row for row in v6_selection_counts}
    candidates: list[MergeRecord] = []
    rows: list[dict[str, Any]] = []
    for path in iter_merge_dirs(search_root, max_depth=3):
        text = path.name.lower()
        if "random" not in text:
            continue
        params = common.read_json(path / "parameters.json") or {}
        stream = common.find_first_key(params, "stream_file")
        label = f"historical_random::{path.name}"
        candidates.append(MergeRecord(label, label, "historical_random", Path(stream) if stream else None, path, 1, False))
        best_match = ""
        classification = "not_comparable"
        reason = "no exact quota/eligibility metadata parsed"
        for record in v6_records:
            variant = record.variant_id
            target = parse_target_from_variant(variant)
            if target == "diagnostic":
                continue
            removed = to_float(v6_counts.get(variant, {}).get("removed_or_selected_count"))
            if target in text and removed is not None and str(int(removed)) in text:
                best_match = variant
                classification = "count_matched_only"
                reason = "name contains target and removal count, but exact eligibility/block matching was not proven"
                break
            if target in text and not best_match:
                best_match = variant
                classification = "same_target_different_count"
                reason = "name contains target but removal count does not match"
        rows.append(
            {
                "random_control_label": label,
                "random_control_merge_dir": str(path),
                "best_v6_condition_match": best_match,
                "classification": classification,
                "reason": reason,
                "oriented_vs_random_delta_allowed": classification == "exact_comparable",
            }
        )
    oriented_rows: list[dict[str, Any]] = []
    return candidates, rows, oriented_rows


def matching_random_prefix(filter_variant_id: str) -> str:
    match = re.match(r"filter_(all|higheg|matched)_(.+)_drop(\d+)$", filter_variant_id)
    if not match:
        return ""
    target, _score, label = match.groups()
    return f"random_{target}_drop{label}_seed"


def generated_random_replicate_results(global_rows: list[dict[str, Any]], selection_counts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {str(row.get("variant_id") or row.get("label")): row for row in global_rows}
    counts = {str(row.get("variant_id")): row for row in selection_counts}
    rows: list[dict[str, Any]] = []
    for variant, oriented in sorted(by_variant.items(), key=lambda item: variant_sort_key(item[0])):
        if not variant.startswith("filter_"):
            continue
        prefix = matching_random_prefix(variant)
        if not prefix:
            continue
        replicates = [by_variant.get(f"{prefix}{seed}", {}) for seed in RANDOM_SEEDS]
        replicates = [row for row in replicates if row]
        if len(replicates) != len(RANDOM_SEEDS):
            continue
        out: dict[str, Any] = {
            "oriented_variant_id": variant,
            "filtering_target": parse_target_from_variant(variant),
            "score_id": parse_score_from_variant(variant),
            "matching_random_variant_ids": ";".join(f"{prefix}{seed}" for seed in RANDOM_SEEDS),
            "random_replicate_count": len(replicates),
            "accepted_observations_removed": counts.get(variant, {}).get("removed_or_selected_count", ""),
            "removed_fraction_of_accepted_population": counts.get(variant, {}).get("removed_fraction_of_accepted_population", counts.get(variant, {}).get("removed_fraction_of_all_accepted_observations", "")),
        }
        for metric in ["cc12", "rsplit", "snr", "completeness", "redundancy"]:
            oriented_value = to_float(oriented.get(metric))
            random_values = [to_float(row.get(metric)) for row in replicates]
            random_values = [value for value in random_values if value is not None]
            out[f"oriented_{metric}"] = oriented_value if oriented_value is not None else ""
            if random_values:
                mean = float(np.mean(random_values))
                out[f"random_{metric}_mean"] = mean
                out[f"random_{metric}_std"] = float(np.std(random_values, ddof=0))
                out[f"random_{metric}_min"] = float(np.min(random_values))
                out[f"random_{metric}_max"] = float(np.max(random_values))
                out[f"delta_{metric}_oriented_minus_random_mean"] = (float(oriented_value) - mean) if oriented_value is not None else ""
                if oriented_value is not None:
                    out[f"oriented_{metric}_above_random_count"] = sum(1 for value in random_values if float(oriented_value) > value)
                    out[f"oriented_{metric}_below_random_count"] = sum(1 for value in random_values if float(oriented_value) < value)
            else:
                out[f"random_{metric}_mean"] = ""
                out[f"random_{metric}_std"] = ""
                out[f"random_{metric}_min"] = ""
                out[f"random_{metric}_max"] = ""
                out[f"delta_{metric}_oriented_minus_random_mean"] = ""
        rows.append(out)
    return rows


def iter_merge_dirs(root: Path, max_depth: int = 3) -> Iterable[Path]:
    root = root.resolve()
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "partialator_results" in path.name and (path / "parameters.json").exists():
            yield path


def halfset_audit(records: list[MergeRecord], hash_halfsets: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.merge_dir is None:
            continue
        params = common.read_json(record.merge_dir / "parameters.json") or {}
        half_keys = {}
        for key in ["seed", "random_seed", "half", "half_seed", "split_seed"]:
            found = common.find_first_key(params, key)
            if found is not None:
                half_keys[key] = found
        for rel in ["crystfel.hkl1", "crystfel.hkl2"]:
            path = record.merge_dir / rel
            fp = common.file_fingerprint(path, hash_file=hash_halfsets)
            rows.append(
                {
                    "variant_id": record.variant_id,
                    "label": record.label,
                    "merge_dir": str(record.merge_dir),
                    "half_file": rel,
                    "half_assignment_seed_keys_json": json.dumps(half_keys, sort_keys=True),
                    "half_assignment_determinism_inferred": "unknown_without_crystfel_internal_audit",
                    "split_by_crystal_or_observation": "not_recorded_in_wrapper_metadata",
                    "thread_or_stream_order_sensitivity": "possible unless CrystFEL split is documented deterministic for this version",
                    **fp,
                }
            )
    return rows


def merge_settings_audit(global_rows: list[dict[str, Any]], full_label: str) -> list[dict[str, Any]]:
    by_variant = {str(row.get("variant_id") or row.get("label")): row for row in global_rows}
    reference = by_variant.get(full_label) or next(iter(by_variant.values()), {})
    fields = [
        "symmetry",
        "partialator_model",
        "iterations",
        "min_measurements",
        "highres",
        "lowres",
        "push_res",
        "min_res",
        "polarisation",
        "no_bscale",
        "disable_pr",
        "normalized_partialator_options",
        "half_set_related_keys_json",
    ]
    rows: list[dict[str, Any]] = []
    for variant, row in sorted(by_variant.items(), key=lambda item: variant_sort_key(item[0])):
        mismatches = []
        out = {
            "variant_id": variant,
            "label": row.get("label", variant),
            "source": row.get("source", ""),
            "merge_dir": row.get("merge_dir", ""),
            "reference_for_settings": reference.get("variant_id", reference.get("label", "")),
            "crystfel_version": parse_metadata_value(Path(str(row.get("merge_dir", ""))) / "metadata_and_outputs.txt", "CrystFEL"),
            "partialator_command": row.get("partialator_command", ""),
            "check_hkl_command_template": "check_hkl crystfel.hkl -p cell.cell --symmetry=4/mmm --lowres=20.0 --highres=0.35 --nshells=20 --shell-file=check_shell.tsv",
            "compare_hkl_cc12_command_template": "compare_hkl crystfel.hkl1 crystfel.hkl2 -p cell.cell --symmetry=4/mmm --lowres=20.0 --highres=0.35 --nshells=20 --fom=CC --shell-file=compare_cc12_shell.tsv",
            "compare_hkl_rsplit_command_template": "compare_hkl crystfel.hkl1 crystfel.hkl2 -p cell.cell --symmetry=4/mmm --lowres=20.0 --highres=0.35 --nshells=20 --fom=Rsplit --shell-file=compare_rsplit_shell.tsv",
        }
        for field in fields:
            equal = str(row.get(field, "")) == str(reference.get(field, ""))
            out[field] = row.get(field, "")
            out[f"{field}_matches_reference"] = equal
            if not equal and variant != str(reference.get("variant_id", reference.get("label", ""))):
                mismatches.append(field)
        out["settings_match_reference_except_input_stream"] = not mismatches
        out["settings_mismatches"] = "; ".join(mismatches)
        rows.append(out)
    return rows


def parse_metadata_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"{key}:"
    for line in common.read_text_lines(path):
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def build_experiment_accounting(
    specs: list[VariantSpec],
    metadata: dict[str, Any],
    merge_records: list[MergeRecord],
    global_rows: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    expected_ids = {spec.variant_id for spec in specs}
    plan = metadata["plan_csv"]
    manifest = metadata["stream_manifest"]
    plan_ids = set(plan["variant_id"].astype(str)) if not plan.empty and "variant_id" in plan.columns else set()
    manifest_index = dataframe_index(manifest, "variant_id")
    record_by_variant = {record.variant_id: record for record in merge_records}
    global_by_variant = {str(row.get("variant_id")): row for row in global_rows}
    shell_count_by_variant = Counter(str(row.get("variant_id")) for row in shell_rows)
    rows: list[dict[str, Any]] = []
    all_ids = sorted(expected_ids | plan_ids | set(manifest_index), key=variant_sort_key)
    for variant_id in all_ids:
        spec = next((item for item in specs if item.variant_id == variant_id), None)
        manifest_row = manifest_index.get(variant_id, {})
        record = record_by_variant.get(variant_id)
        global_row = global_by_variant.get(variant_id, {})
        stream_path = record.stream_path if record else (Path(str(manifest_row.get("output_stream"))) if manifest_row.get("output_stream") else None)
        merge_dir = record.merge_dir if record else None
        missing_required = str(global_row.get("missing_required_files", ""))
        planned = variant_id in plan_ids
        expected = variant_id in expected_ids
        stream_found = bool(stream_path and stream_path.is_file())
        merge_found = merge_dir is not None and merge_dir.is_dir()
        global_metrics_found = all(to_float(global_row.get(metric)) is not None for metric in ["cc12", "rsplit", "redundancy"])
        shell_metrics_found = shell_count_by_variant.get(variant_id, 0) > 0
        failed = (not expected) or (not planned) or (not stream_found) or (not merge_found) or bool(missing_required) or (record.ambiguous if record else True)
        warning = not failed and (not global_metrics_found or not shell_metrics_found)
        status = "failed" if failed else ("warning" if warning else "validated")
        row = {
            "variant_id": variant_id,
            "expected": expected,
            "planned": planned,
            "stream_manifest_status": manifest_row.get("status", ""),
            "stream_path": str(stream_path or ""),
            "stream_found": stream_found,
            "merge_dir": str(merge_dir or ""),
            "merge_found": merge_found,
            "merge_candidate_count": record.candidate_count if record else 0,
            "merge_ambiguous": record.ambiguous if record else True,
            "required_files_present": not bool(missing_required),
            "missing_required_files": missing_required,
            "global_metrics_found": global_metrics_found,
            "shell_metrics_found": shell_metrics_found,
            "shell_count": shell_count_by_variant.get(variant_id, 0),
            "validated": status == "validated",
            "warning": status == "warning",
            "failed": status == "failed",
            "status": status,
        }
        if status != "validated":
            warnings.append({"variant_id": variant_id, "stage": "experiment_accounting", "severity": "error" if failed else "warning", "message": status})
        rows.append(row)
    return rows, warnings


def missing_or_failed_variants(accounting_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in accounting_rows if row.get("status") != "validated"]


def pareto_front(filter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in filter_rows:
        metrics = {
            "cc12": finite(row.get("cc12")),
            "rsplit_neg": -finite(row.get("rsplit")),
            "snr": finite(row.get("snr")),
            "retained": finite(row.get("retained_observation_count")),
            "completeness": finite(row.get("completeness")),
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            continue
        candidates.append((row, metrics))
    front: list[dict[str, Any]] = []
    for row, metrics in candidates:
        dominated = False
        for other, other_metrics in candidates:
            if other is row:
                continue
            ge_all = all(other_metrics[key] >= metrics[key] for key in metrics)
            gt_any = any(other_metrics[key] > metrics[key] for key in metrics)
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            out = dict(row)
            out["pareto_objectives"] = "maximize cc12,snr,retained_observation_count,completeness; minimize rsplit"
            front.append(out)
    front.sort(key=lambda item: (-finite(item.get("cc12")), finite(item.get("rsplit"))))
    return front


def variant_rankings(
    diagnostic_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    shell_summary: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(diagnostic_rows, key=lambda item: int(item.get("diagnostic_rank") or 9999)):
        rows.append({"ranking_type": "diagnostic", "rank": row.get("diagnostic_rank"), "variant_id": row.get("low_variant_id"), "score_id": row.get("score_id"), "basis": "low/high CC1/2 and Rsplit separation", "primary_value": row.get("delta_cc12_low_minus_high")})
    for idx, row in enumerate(sorted(filter_rows, key=lambda item: (finite(item.get("delta_cc12_vs_full")), -finite(item.get("delta_rsplit_vs_full"))), reverse=True), start=1):
        rows.append({"ranking_type": "filtering", "rank": idx, "variant_id": row.get("variant_id"), "score_id": row.get("score_id"), "basis": "global delta CC1/2, then Rsplit", "primary_value": row.get("delta_cc12_vs_full")})
    for idx, row in enumerate(sorted(shell_summary, key=lambda item: (finite(item.get("median_delta_cc12")), -finite(item.get("median_delta_rsplit"))), reverse=True), start=1):
        rows.append({"ranking_type": "shellwise", "rank": idx, "variant_id": row.get("variant_id"), "score_id": parse_score_from_variant(str(row.get("variant_id"))), "basis": "median shell delta CC1/2 and Rsplit", "primary_value": row.get("median_delta_cc12")})
    data_eff = []
    for row in filter_rows:
        frac = finite(row.get("removed_fraction_of_all_accepted_observations"))
        cc = finite(row.get("delta_cc12_vs_full"))
        if math.isfinite(frac) and frac > 0 and math.isfinite(cc):
            item = dict(row)
            item["cc12_gain_per_removed_fraction"] = cc / frac
            data_eff.append(item)
    for idx, row in enumerate(sorted(data_eff, key=lambda item: finite(item.get("cc12_gain_per_removed_fraction")), reverse=True), start=1):
        rows.append({"ranking_type": "data_efficiency", "rank": idx, "variant_id": row.get("variant_id"), "score_id": row.get("score_id"), "basis": "delta CC1/2 per actual removed fraction", "primary_value": row.get("cc12_gain_per_removed_fraction")})
    exact_random = [row for row in random_rows if row.get("classification") == "exact_comparable"]
    for idx, row in enumerate(exact_random, start=1):
        rows.append({"ranking_type": "random_controlled", "rank": idx, "variant_id": row.get("best_v6_condition_match", ""), "basis": "exact comparable random control", "primary_value": ""})
    return rows


def maybe_recalculate_shells(
    records: list[MergeRecord],
    shell_rows: list[dict[str, Any]],
    out_dir: Path,
    disabled: bool,
    logger: AuditLogger,
    reference_variant_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    recalculated_shell_rows: list[dict[str, Any]] = []
    if disabled:
        return rows, recalculated_shell_rows
    keys_by_variant = shell_boundary_keys_by_variant(shell_rows)
    reference_keys = keys_by_variant.get(reference_variant_id, [])
    needing = [
        record
        for record in records
        if record.merge_dir is not None
        and (
            len(keys_by_variant.get(record.variant_id, [])) != 20
            or (reference_keys and keys_by_variant.get(record.variant_id, []) != reference_keys)
        )
    ]
    if not needing:
        return rows, recalculated_shell_rows
    check = shutil.which("check_hkl")
    compare = shutil.which("compare_hkl")
    for record in needing:
        recalc_dir = out_dir / "recalculated_shells" / record.variant_id
        recalc_dir.mkdir(parents=True, exist_ok=True)
        merge_dir = record.merge_dir
        assert merge_dir is not None
        cell = merge_dir / "cell.cell"
        hkl = merge_dir / "crystfel.hkl"
        hkl1 = merge_dir / "crystfel.hkl1"
        hkl2 = merge_dir / "crystfel.hkl2"
        commands = [
            ["check_hkl", str(hkl), "-p", str(cell), "--symmetry=4/mmm", "--lowres=20.0", "--highres=0.35", "--nshells=20", f"--shell-file={recalc_dir / 'check_shell.tsv'}"],
            ["compare_hkl", str(hkl1), str(hkl2), "-p", str(cell), "--symmetry=4/mmm", "--lowres=20.0", "--highres=0.35", "--nshells=20", "--fom=CC", f"--shell-file={recalc_dir / 'compare_cc12_shell.tsv'}"],
            ["compare_hkl", str(hkl1), str(hkl2), "-p", str(cell), "--symmetry=4/mmm", "--lowres=20.0", "--highres=0.35", "--nshells=20", "--fom=Rsplit", f"--shell-file={recalc_dir / 'compare_rsplit_shell.tsv'}"],
        ]
        status = "planned"
        parsed_rows: list[dict[str, Any]] = []
        if not check or not compare:
            status = "skipped_missing_check_hkl_or_compare_hkl"
        else:
            status = "executed"
            for idx, cmd in enumerate(commands, start=1):
                log_path = recalc_dir / f"recalc_{idx}.log"
                with log_path.open("w", encoding="utf-8") as handle:
                    subprocess.run(cmd, check=False, stdout=handle, stderr=subprocess.STDOUT, text=True)
            parsed_rows = parse_recalculated_shell_rows(record, recalc_dir)
            if len(parsed_rows) == 20:
                recalculated_shell_rows.extend(parsed_rows)
            elif parsed_rows:
                status = f"executed_but_parsed_{len(parsed_rows)}_shells"
        rows.append(
            {
                "variant_id": record.variant_id,
                "merge_dir": str(merge_dir),
                "recalc_dir": str(recalc_dir),
                "status": status,
                "existing_shell_count": len(keys_by_variant.get(record.variant_id, [])),
                "existing_boundaries_match_reference": bool(reference_keys) and keys_by_variant.get(record.variant_id, []) == reference_keys,
                "parsed_recalculated_shell_count": len(parsed_rows),
                "recalculated_shells_used": len(parsed_rows) == 20,
                "commands_json": json.dumps(commands),
            }
        )
    return rows, recalculated_shell_rows


def shell_boundary_keys_by_variant(shell_rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shell_rows:
        rows_by_variant[str(row.get("variant_id") or row.get("label"))].append(row)
    out: dict[str, list[tuple[float, float]]] = {}
    for variant, rows in rows_by_variant.items():
        keys = [common.boundary_key(row) for row in sorted(rows, key=lambda item: int(item.get("shell_index") or 0))]
        out[variant] = [key for key in keys if key is not None]
    return out


def parse_recalculated_shell_rows(record: MergeRecord, recalc_dir: Path) -> list[dict[str, Any]]:
    rows = common.join_shells(
        common.parse_shell_table(recalc_dir / "check_shell.tsv", "check_shell"),
        common.parse_shell_table(recalc_dir / "compare_cc12_shell.tsv", "cc12_shell"),
        common.parse_shell_table(recalc_dir / "compare_rsplit_shell.tsv", "rsplit_shell"),
    )
    for row in rows:
        row["label"] = record.label
        row["variant_id"] = record.variant_id
        row["merge_dir"] = str(record.merge_dir or "")
        row["shell_metric_source"] = "recalculated_check_hkl_compare_hkl"
        row["recalculated_shell_dir"] = str(recalc_dir)
    return rows


def replace_shell_rows(original_rows: list[dict[str, Any]], replacement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacement_ids = {str(row.get("variant_id") or row.get("label")) for row in replacement_rows}
    if not replacement_ids:
        return original_rows
    rows = [row for row in original_rows if str(row.get("variant_id") or row.get("label")) not in replacement_ids]
    rows.extend(replacement_rows)
    rows.sort(key=lambda row: (variant_sort_key(str(row.get("variant_id") or row.get("label"))), int(row.get("shell_index") or 0)))
    return rows


def write_required_empty_files(out_dir: Path) -> None:
    for name in REQUIRED_OUTPUT_FILES:
        path = out_dir / name
        if path.exists():
            continue
        if name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        elif name.endswith(".md"):
            path.write_text("# Complete Audit Summary\n\nAudit did not complete.\n", encoding="utf-8")
        elif name.endswith(".log"):
            path.write_text("", encoding="utf-8")
        else:
            path.write_text("\n", encoding="utf-8")


def make_plots(
    out_dir: Path,
    diagnostic_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    shell_deltas: list[dict[str, Any]],
    selection_overlap: pd.DataFrame,
    pareto_rows: list[dict[str, Any]],
    skip: bool,
    warnings: list[dict[str, Any]],
) -> None:
    if skip:
        for name in PLOT_FILES:
            warnings.append({"stage": "plots", "severity": "warning", "message": f"plot skipped by --skip-plots: {name}"})
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        for name in PLOT_FILES:
            warnings.append({"stage": "plots", "severity": "warning", "message": f"matplotlib unavailable for {name}: {exc}"})
        return

    def save_placeholder(name: str, title: str, message: str = "No data available") -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=140)
        plt.close(fig)

    diag = pd.DataFrame(diagnostic_rows)
    if diag.empty:
        for name in ["diagnostic_cc12_low_vs_high.png", "diagnostic_rsplit_low_vs_high.png", "diagnostic_cc12_separation.png", "diagnostic_rsplit_separation.png"]:
            save_placeholder(name, name)
    else:
        for metric, filename, ylabel in [
            ("delta_cc12_low_minus_high", "diagnostic_cc12_separation.png", "low - high CC1/2; positive favors low-score half"),
            ("delta_rsplit_low_minus_high", "diagnostic_rsplit_separation.png", "low - high Rsplit; negative favors low-score half"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_df = diag.sort_values(metric, ascending=False if "cc12" in metric else True)
            ax.bar(plot_df["score_id"], pd.to_numeric(plot_df[metric], errors="coerce"))
            ax.set_ylabel(ylabel)
            ax.set_xlabel("score")
            ax.tick_params(axis="x", rotation=45)
            ax.axhline(0, color="black", linewidth=0.8)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=140)
            plt.close(fig)
        for metric, filename, ylabel in [
            ("delta_cc12_low_minus_high", "diagnostic_cc12_low_vs_high.png", "CC1/2 separation"),
            ("delta_rsplit_low_minus_high", "diagnostic_rsplit_low_vs_high.png", "Rsplit separation"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(range(len(diag)), pd.to_numeric(diag[metric], errors="coerce"))
            ax.set_xticks(range(len(diag)))
            ax.set_xticklabels(diag["score_id"], rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title("Diagnostic low-minus-high; high resolution direction is not applicable")
            ax.axhline(0, color="black", linewidth=0.8)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=140)
            plt.close(fig)

    filters = pd.DataFrame(filter_rows)
    target_names = {"all": "target_A", "higheg": "target_B", "matched": "target_C"}
    metric_labels = {
        "cc12": "CC1/2 (higher is better)",
        "rsplit": "Rsplit (lower is better)",
        "snr": "SNR (higher is better)",
        "redundancy": "Redundancy",
    }
    for target, label in target_names.items():
        for metric in ["cc12", "rsplit", "snr", "redundancy"]:
            name = f"filter_sweep_{label}_{metric}_vs_removed_fraction.png"
            subset = filters.loc[filters.get("filtering_target", pd.Series(dtype=str)).astype(str) == target] if not filters.empty else pd.DataFrame()
            if subset.empty:
                save_placeholder(name, name)
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for score, group in subset.groupby("score_id"):
                group = group.sort_values("removed_fraction_of_all_accepted_observations")
                ax.plot(pd.to_numeric(group["removed_fraction_of_all_accepted_observations"], errors="coerce"), pd.to_numeric(group[metric], errors="coerce"), marker="o", label=score)
            ax.set_xlabel("Actual global fraction removed")
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f"{label}: {metric_labels[metric]}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / name, dpi=140)
            plt.close(fig)

    shell_df = pd.DataFrame(shell_deltas)
    for metric, name, title in [
        ("delta_cc12_vs_full", "shell_heatmap_cc12_delta_vs_full.png", "CC1/2 delta vs full; higher is better"),
        ("delta_rsplit_vs_full", "shell_heatmap_rsplit_delta_vs_full.png", "Rsplit delta vs full; lower is better"),
        ("delta_snr_vs_full", "shell_heatmap_snr_delta_vs_full.png", "SNR delta vs full; higher is better"),
    ]:
        if shell_df.empty or metric not in shell_df.columns:
            save_placeholder(name, title)
            continue
        pivot = shell_df.pivot_table(index="variant_id", columns="shell_index", values=metric, aggfunc="first")
        if pivot.empty:
            save_placeholder(name, title)
            continue
        fig, ax = plt.subplots(figsize=(12, max(4, 0.15 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="none")
        ax.set_title(title + "; shell index increases toward high resolution")
        ax.set_xlabel("Resolution shell (higher index = higher resolution)")
        ax.set_ylabel("Variant")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=140)
        plt.close(fig)

    focus = ["filter_matched_eg_d2_cmean_drop30", "filter_matched_eg_d3_cmean_drop30"]
    for metric, name, label in [("cc12", "focused_shell_curves_cc12.png", "CC1/2"), ("rsplit", "focused_shell_curves_rsplit.png", "Rsplit")]:
        if shell_df.empty:
            save_placeholder(name, name)
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in focus:
            subset = shell_df.loc[shell_df["variant_id"].astype(str) == variant].sort_values("shell_index")
            if not subset.empty and metric in subset.columns:
                ax.plot(subset["shell_index"], pd.to_numeric(subset[metric], errors="coerce"), marker="o", label=variant)
        ax.set_xlabel("Resolution shell (higher index = higher resolution)")
        ax.set_ylabel(label)
        ax.set_title(f"Focused shell curves: {label}; metric direction noted in legend context")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=140)
        plt.close(fig)

    if filters.empty:
        for name in ["metric_improvement_vs_observations_removed.png", "cc12_vs_rsplit_pareto.png", "retained_redundancy_vs_rsplit.png", "retained_redundancy_vs_cc12.png"]:
            save_placeholder(name, name)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pd.to_numeric(filters["removed_or_selected_count"], errors="coerce"), pd.to_numeric(filters["delta_cc12_vs_full"], errors="coerce"))
        ax.set_xlabel("Observations removed")
        ax.set_ylabel("Delta CC1/2 vs full (higher is better)")
        fig.tight_layout()
        fig.savefig(out_dir / "metric_improvement_vs_observations_removed.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pd.to_numeric(filters["rsplit"], errors="coerce"), pd.to_numeric(filters["cc12"], errors="coerce"), label="filters")
        pareto_df = pd.DataFrame(pareto_rows)
        if not pareto_df.empty:
            ax.scatter(pd.to_numeric(pareto_df["rsplit"], errors="coerce"), pd.to_numeric(pareto_df["cc12"], errors="coerce"), label="Pareto", marker="x")
        ax.set_xlabel("Rsplit (lower is better)")
        ax.set_ylabel("CC1/2 (higher is better)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cc12_vs_rsplit_pareto.png", dpi=140)
        plt.close(fig)
        for xmetric, name in [("rsplit", "retained_redundancy_vs_rsplit.png"), ("cc12", "retained_redundancy_vs_cc12.png")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(pd.to_numeric(filters[xmetric], errors="coerce"), pd.to_numeric(filters["redundancy"], errors="coerce"))
            ax.set_xlabel(f"{xmetric} ({'lower' if xmetric == 'rsplit' else 'higher'} is better)")
            ax.set_ylabel("Retained redundancy")
            fig.tight_layout()
            fig.savefig(out_dir / name, dpi=140)
            plt.close(fig)

    if selection_overlap.empty:
        save_placeholder("score_selection_jaccard_matrix.png", "Score-selection Jaccard matrix")
    else:
        left_col = "left_variant_id" if "left_variant_id" in selection_overlap.columns else ("left_variant" if "left_variant" in selection_overlap.columns else "")
        right_col = "right_variant_id" if "right_variant_id" in selection_overlap.columns else ("right_variant" if "right_variant" in selection_overlap.columns else "")
        jac_col = "jaccard" if "jaccard" in selection_overlap.columns else ("jaccard_index" if "jaccard_index" in selection_overlap.columns else "")
        if not left_col or not right_col or not jac_col:
            save_placeholder("score_selection_jaccard_matrix.png", "Score-selection Jaccard matrix", "No recognizable Jaccard columns")
        else:
            names = sorted(set(selection_overlap[left_col].astype(str)) | set(selection_overlap[right_col].astype(str)))[:60]
            matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
            for _, row in selection_overlap.iterrows():
                left = str(row[left_col])
                right = str(row[right_col])
                if left in matrix.index and right in matrix.columns:
                    matrix.loc[left, right] = finite(row[jac_col])
                    matrix.loc[right, left] = finite(row[jac_col])
            fig, ax = plt.subplots(figsize=(10, 8))
            image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, interpolation="none")
            ax.set_title("Selection Jaccard matrix; no missing-value interpolation")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=90, fontsize=4)
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels(matrix.index, fontsize=4)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(out_dir / "score_selection_jaccard_matrix.png", dpi=140)
            plt.close(fig)


def write_summary(
    out_dir: Path,
    summary: dict[str, Any],
    accounting: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    shell_summary: list[dict[str, Any]],
    v5_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
) -> None:
    failed = [row for row in accounting if row.get("status") != "validated"]
    best_diag = diagnostic_rows[0] if diagnostic_rows else {}
    best_filter = max(filter_rows, key=lambda row: finite(row.get("delta_cc12_vs_full")), default={})
    best_target = Counter(row.get("filtering_target") for row in filter_rows[:10]).most_common(1)
    d2 = next((row for row in filter_rows if row.get("variant_id") == "filter_matched_eg_d2_cmean_drop30"), {})
    d3 = next((row for row in filter_rows if row.get("variant_id") == "filter_matched_eg_d3_cmean_drop30"), {})
    concise = {
        "all_88_streams_and_merges_complete": len(failed) == 0,
        "failed_or_warning_variant_count": len(failed),
        "strongest_low_high_score": best_diag.get("score_id", ""),
        "best_filtering_variant_by_cc12_delta": best_filter.get("variant_id", ""),
        "best_filtering_target_top_ranked": best_target[0][0] if best_target else "",
        "best_removal_strength_for_best_filter": best_filter.get("removed_fraction_of_all_accepted_observations", ""),
        "improvements_broad_across_resolution": infer_broad_shell_behavior(shell_summary, best_filter.get("variant_id", "")),
        "d2_vs_d3_matched_drop30": {
            "d2_delta_cc12_vs_full": d2.get("delta_cc12_vs_full", ""),
            "d2_delta_rsplit_vs_full": d2.get("delta_rsplit_vs_full", ""),
            "d3_delta_cc12_vs_full": d3.get("delta_cc12_vs_full", ""),
            "d3_delta_rsplit_vs_full": d3.get("delta_rsplit_vs_full", ""),
        },
        "eg_m2_outperforms_normalized_second_moment": "see score_family_comparisons.csv",
        "historical_random_controls_valid_for_best_conditions": any(row.get("classification") == "exact_comparable" for row in random_rows),
        "v5_v6_matched_drop30_redundancy_difference": v5_rows[0].get("redundancy_difference_explanation", "") if v5_rows else "",
        "robust_results_to_carry_forward": "see variant_rankings.csv and pareto_front.csv",
        "next_experiment": "Prioritize exact random-control generation only for top V6 conditions lacking exact comparable historical controls.",
    }
    summary["concise_answers"] = concise
    (out_dir / "complete_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    lines = [
        "# Complete OriDyn V6 Audit Summary",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Concise Answers",
        "",
    ]
    for key, value in concise.items():
        lines.append(f"- **{key}**: {json.dumps(value, default=json_default)}")
    lines.extend(
        [
            "",
            "## Critical Tables",
            "",
            "- Experiment accounting: `experiment_accounting.csv`",
            "- Selection counts: `selection_counts.csv`",
            "- Global metrics and deltas: `global_metrics_all_variants.csv`, `global_deltas_vs_full.csv`",
            "- Shell metrics and summaries: `shell_metrics_all_variants.csv`, `shell_deltas_vs_full.csv`, `shell_variant_summary.csv`",
            "- V5 and random comparability: `v5_v6_equivalence.csv`, `random_control_comparability.csv`",
            "",
            "## Limitations",
            "",
            "- Historical random controls are classified conservatively unless exact eligibility, block, and quota metadata can be proven from existing files.",
            "- Half-set reproducibility is inferred from logs and output files; the audit does not rerun merges.",
            "- Shell recalculation fallback only runs `check_hkl`/`compare_hkl` when existing shell outputs are missing or incompatible and the commands are available.",
        ]
    )
    (out_dir / "complete_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_broad_shell_behavior(shell_summary: list[dict[str, Any]], variant: str) -> str:
    row = next((item for item in shell_summary if item.get("variant_id") == variant), None)
    if not row:
        return "not_available"
    improved = to_int(row.get("improved_rsplit_shell_count")) or 0
    worsened = to_int(row.get("worsened_rsplit_shell_count")) or 0
    if improved > worsened:
        return "mostly_improved_by_rsplit_shell_count"
    if worsened > improved:
        return "mostly_worse_by_rsplit_shell_count"
    return "mixed_or_tied"


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return ""
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    return {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}


def run_audit(args: argparse.Namespace) -> int:
    args = normalize_args(args)
    safe_prepare_out_dir(args.out_dir, [args.v6_dir, args.source_stream])
    logger = AuditLogger(args.out_dir)
    warnings: list[dict[str, Any]] = []
    stage_total = 21
    metadata = load_metadata(args.v6_dir)
    cache_count = restricted_cache_row_count(metadata)
    if cache_count == RESTRICTED_V6_CACHE_ROWS:
        raise SystemExit(
            f"Refusing to audit restricted pilot cache as complete full-population V6: "
            f"cache_rows={cache_count:,}, expected={FULL_POPULATION_ACCEPTED_OBSERVATIONS:,}"
        )
    include_random = has_full_population_random_controls(metadata)
    specs = all_expected_variants(include_random=include_random)
    expected_ids = {spec.variant_id for spec in specs}
    try:
        logger.stage_start(1, stage_total, "configuration")
        run_metadata: dict[str, Any] = {
            "audit_date_local": now_iso(),
            "project_root": str(Path(__file__).resolve().parents[1]),
            "root": str(args.root),
            "v6_dir": str(args.v6_dir),
            "source_stream": str(args.source_stream),
            "historical_search_root": str(args.historical_search_root),
            "out_dir": str(args.out_dir),
            "git_commit": git_commit(Path(__file__).resolve().parents[1]),
            "package_versions": package_versions(),
            "workers": args.workers,
            "expected_score_variant_count": EXPECTED_VARIANT_COUNT,
            "expected_random_control_count": EXPECTED_FULL_POPULATION_RANDOM_COUNT if include_random else 0,
            "expected_total_variant_count": len(specs),
            "detected_full_population_random_controls": include_random,
            "numeric_thread_env": {name: os.environ.get(name, "") for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]},
            "commands": {"manual_command": "see user-provided command structure"},
        }
        logger.stage_done("configuration")

        logger.stage_start(2, stage_total, "file discovery", len(specs))
        manifest_index = dataframe_index(metadata["stream_manifest"], "variant_id")
        v6_records, merge_discovery_rows = discover_v6_merges(args.v6_dir, specs, manifest_index)
        full_reference, full_candidates = common.discover_full_reference(args.root, args.source_stream)
        extra_records: list[MergeRecord] = []
        full_label = "full_reference"
        if full_reference is not None:
            full_label = full_reference.variant_id
            extra_records.append(MergeRecord(full_reference.label, full_reference.variant_id, "full_reference", full_reference.stream_path, full_reference.merge_dir, 1, False))
        logger.stage_done("file discovery", len(v6_records), len(specs))

        logger.stage_start(3, stage_total, "experiment accounting")
        # Account after parse; stage marker is kept here for requested progress structure.
        logger.stage_done("experiment accounting")

        logger.stage_start(4, stage_total, "stream validation", len(specs))
        stream_rows = stream_validation(v6_records, metadata, args.workers, args.skip_stream_scan, logger)
        logger.stage_done("stream validation", len(stream_rows), len(specs))

        logger.stage_start(5, stage_total, "selection-count audit")
        hkl_qc, block_qc, qc_warnings = aggregate_variant_qc(args.v6_dir, expected_ids, logger)
        warnings.extend(qc_warnings)
        manifest_counts = selection_manifest_counts(args.v6_dir, expected_ids, logger)
        selection_rows = build_selection_counts(specs, metadata, hkl_qc, block_qc, manifest_counts)
        logger.stage_done("selection-count audit", len(selection_rows), len(specs))

        for stage_index, stage_name in [(6, "diagnostic-pair audit"), (7, "target-A audit"), (8, "target-B audit"), (9, "target-C audit")]:
            logger.stage_start(stage_index, stage_total, stage_name)
            logger.stage_done(stage_name)

        logger.stage_start(10, stage_total, "merge-result discovery", len(specs))
        write_csv(args.out_dir / "merge_discovery.csv", merge_discovery_rows)
        write_csv(args.out_dir / "full_reference_candidates.csv", full_candidates)
        logger.stage_done("merge-result discovery", len(merge_discovery_rows), len(specs))

        logger.stage_start(11, stage_total, "global metric extraction", len(v6_records) + len(extra_records))
        global_rows, shell_rows, merge_warnings = parse_all_merges(v6_records + extra_records, args.workers, logger)
        warnings.extend(merge_warnings)
        logger.stage_done("global metric extraction", len(global_rows), len(v6_records) + len(extra_records))

        logger.stage_start(12, stage_total, "shell-boundary validation")
        recalc_rows, recalculated_shell_rows = maybe_recalculate_shells(
            v6_records + extra_records,
            shell_rows,
            args.out_dir,
            args.no_shell_recalculation,
            logger,
            full_label,
        )
        if recalc_rows:
            write_csv(args.out_dir / "shell_recalculation_commands.csv", recalc_rows)
        if recalculated_shell_rows:
            shell_rows = replace_shell_rows(shell_rows, recalculated_shell_rows)
        shell_boundaries, shell_delta_rows, shell_summary_rows = shell_boundaries_and_deltas(shell_rows, global_rows, full_label)
        logger.stage_done("shell-boundary validation", len(shell_boundaries), len(shell_rows))

        logger.stage_start(13, stage_total, "shell parsing or recalculation")
        logger.stage_done("shell parsing or recalculation", len(shell_rows), len(shell_rows))

        accounting_rows, accounting_warnings = build_experiment_accounting(specs, metadata, v6_records, global_rows, shell_rows)
        warnings.extend(accounting_warnings)
        selection_overlap_source = metadata["selection_overlap"]

        diagnostic_rows, diagnostic_shell_rows = diagnostic_pair_results(specs, global_rows, shell_rows, selection_rows, selection_overlap_source)
        delta_rows = global_deltas(global_rows, full_label, selection_rows)
        filter_rows = filter_global_metrics(delta_rows)
        sweep_rows = filter_sweeps(filter_rows, shell_summary_rows)

        logger.stage_start(14, stage_total, "score-family comparisons")
        family_rows = score_family_comparisons(diagnostic_rows, filter_rows, shell_summary_rows, selection_overlap_source)
        logger.stage_done("score-family comparisons", len(family_rows), len(family_rows))

        logger.stage_start(15, stage_total, "selection-overlap analysis")
        overlap_rows = selection_overlap_source.to_dict("records") if not selection_overlap_source.empty else []
        logger.stage_done("selection-overlap analysis", len(overlap_rows), len(overlap_rows))

        logger.stage_start(16, stage_total, "V5 equivalence")
        v5_records, v5_rows = discover_v5_equivalence(args.root, global_rows, shell_rows, selection_rows, metadata)
        if v5_records:
            v5_global_rows, v5_shell_rows, v5_warnings = parse_all_merges(v5_records, args.workers, logger)
            global_rows.extend(v5_global_rows)
            shell_rows.extend(v5_shell_rows)
            warnings.extend(v5_warnings)
        logger.stage_done("V5 equivalence", len(v5_rows), len(v5_rows))

        logger.stage_start(17, stage_total, "random-control matching")
        random_records, random_rows, oriented_random_rows = discover_random_controls(args.historical_search_root, v6_records, selection_rows)
        generated_random_rows = generated_random_replicate_results(global_rows, selection_rows)
        oriented_random_rows.extend(generated_random_rows)
        if generated_random_rows:
            random_rows.append(
                {
                    "random_control_label": "generated_full_population_random_controls",
                    "random_control_merge_dir": str(args.v6_dir),
                    "best_v6_condition_match": "all generated filters",
                    "classification": "exact_comparable",
                    "reason": "corrected V6 builder uses shared matched random controls with identical eligibility and quotas",
                    "oriented_vs_random_delta_allowed": True,
                    "replicate_count": len(RANDOM_SEEDS),
                }
            )
        logger.stage_done("random-control matching", len(random_rows), len(random_rows))

        logger.stage_start(18, stage_total, "half-set audit")
        halfset_rows = halfset_audit(v6_records + extra_records + v5_records, args.hash_halfsets)
        logger.stage_done("half-set audit", len(halfset_rows), len(halfset_rows))

        logger.stage_start(19, stage_total, "rankings")
        pareto_rows = pareto_front(filter_rows)
        ranking_rows = variant_rankings(diagnostic_rows, filter_rows, shell_summary_rows, random_rows)
        logger.stage_done("rankings", len(ranking_rows), len(ranking_rows))

        logger.stage_start(20, stage_total, "plots")
        make_plots(args.out_dir, diagnostic_rows, filter_rows, shell_delta_rows, selection_overlap_source, pareto_rows, args.skip_plots, warnings)
        logger.stage_done("plots", len(PLOT_FILES), len(PLOT_FILES))

        logger.stage_start(21, stage_total, "final report")
        merge_settings_rows = merge_settings_audit(global_rows, full_label)
        missing_rows = missing_or_failed_variants(accounting_rows)
        global_rows.sort(key=lambda row: variant_sort_key(str(row.get("variant_id") or row.get("label"))))
        shell_rows.sort(key=lambda row: (variant_sort_key(str(row.get("variant_id") or row.get("label"))), int(row.get("shell_index") or 0)))

        write_csv(args.out_dir / "experiment_accounting.csv", accounting_rows)
        write_csv(args.out_dir / "stream_validation.csv", stream_rows)
        write_csv(args.out_dir / "selection_counts.csv", selection_rows)
        write_csv(args.out_dir / "diagnostic_pair_results.csv", diagnostic_rows)
        write_csv(args.out_dir / "filter_global_metrics.csv", filter_rows)
        write_csv(args.out_dir / "filter_sweeps.csv", sweep_rows)
        write_csv(args.out_dir / "global_metrics_all_variants.csv", global_rows)
        write_csv(args.out_dir / "global_deltas_vs_full.csv", delta_rows)
        write_csv(args.out_dir / "shell_boundaries.csv", shell_boundaries)
        write_csv(args.out_dir / "shell_metrics_all_variants.csv", shell_rows)
        write_csv(args.out_dir / "shell_deltas_vs_full.csv", shell_delta_rows)
        write_csv(args.out_dir / "shell_diagnostic_low_high_deltas.csv", diagnostic_shell_rows)
        write_csv(args.out_dir / "shell_variant_summary.csv", shell_summary_rows)
        write_csv(args.out_dir / "score_family_comparisons.csv", family_rows)
        write_csv(args.out_dir / "selection_overlap.csv", overlap_rows)
        write_csv(args.out_dir / "v5_v6_equivalence.csv", v5_rows)
        write_csv(args.out_dir / "random_control_comparability.csv", random_rows)
        write_csv(args.out_dir / "oriented_vs_random_results.csv", oriented_random_rows)
        write_csv(args.out_dir / "merge_settings_audit.csv", merge_settings_rows)
        write_csv(args.out_dir / "halfset_audit.csv", halfset_rows)
        write_csv(args.out_dir / "variant_rankings.csv", ranking_rows)
        write_csv(args.out_dir / "pareto_front.csv", pareto_rows)
        write_csv(args.out_dir / "missing_or_failed_variants.csv", missing_rows)
        write_csv(args.out_dir / "warnings.csv", warnings)

        run_metadata.update(
            {
                "selected_full_data_reference": next((row for row in global_rows if row.get("source") == "full_reference"), {}),
                "selected_v5_equivalence_runs": v5_rows,
                "selected_random_controls": random_rows,
                "output_files": REQUIRED_OUTPUT_FILES,
                "plot_files": PLOT_FILES,
                "critical_failure_count": len([row for row in missing_rows if row.get("failed") is True or str(row.get("failed")).lower() == "true"]),
            }
        )
        summary = {
            "run_metadata": run_metadata,
            "counts": {
                "expected_variant_count": EXPECTED_VARIANT_COUNT,
                "expected_random_control_count": EXPECTED_FULL_POPULATION_RANDOM_COUNT if include_random else 0,
                "expected_total_variant_count": len(specs),
                "accounted_variants": len(accounting_rows),
                "validated_variants": sum(1 for row in accounting_rows if row.get("status") == "validated"),
                "warning_or_failed_variants": len(missing_rows),
                "warnings": len(warnings),
            },
        }
        write_summary(args.out_dir, summary, accounting_rows, diagnostic_rows, filter_rows, shell_summary_rows, v5_rows, random_rows)
        (args.out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
        write_required_empty_files(args.out_dir)
        logger.stage_done("final report")
        return 1 if any(row.get("status") == "failed" for row in accounting_rows) else 0
    finally:
        logger.finalize()


def main() -> int:
    return run_audit(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
