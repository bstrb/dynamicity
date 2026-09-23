#!/usr/bin/env python3
"""Read-only audit for the OriDyn V6 matched drop30 result.

The audit inspects existing V5/V6 selection metadata, streams, and merge QC
outputs.  It never rewrites streams, runs Partialator, or modifies existing
experiment directories; all generated artifacts go to a separate audit output
directory.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any, Iterable

import pandas as pd


V6_PRIMARY_VARIANTS = [
    "filter_matched_eg_d2_cmean_drop30",
    "filter_matched_eg_d3_cmean_drop30",
]
V6_SECONDARY_VARIANTS = [
    "filter_all_eg_m2_drop05",
    "filter_all_eg_m2_drop10",
]
V5_EQUIVALENT_LABEL = "v5_equivalent_p1"
FULL_REFERENCE_LABEL = "full_reference"
AUDIT_DIRNAME = "oridyn_v6_matched_drop30_audit_20260717"

METADATA_FILES = [
    "stream_manifest.csv",
    "experiment_plan.csv",
    "experiment_plan.json",
    "per_variant_selection_summary.csv",
    "per_variant_per_hkl_qc.csv",
    "per_variant_per_block_qc.csv",
    "selected_observations.csv.gz",
    "parameters.json",
    "scores.json",
    "validation.json",
    "run_metadata.json",
    "run.log",
]

MERGE_REQUIRED_FILES = [
    "parameters.json",
    "cell.cell",
    "partialator_stdout.log",
    "partialator_stderr.log",
    "crystfel.hkl",
    "crystfel.hkl1",
    "crystfel.hkl2",
    "qc_stats/check_hkl_completeness.log",
    "qc_stats/check_shell.tsv",
    "qc_stats/compare_cc12.log",
    "qc_stats/compare_cc12_shell.tsv",
    "qc_stats/compare_rsplit.log",
    "qc_stats/compare_rsplit_shell.tsv",
]

CANONICAL_V6_EG_D3_CMEAN_TREE = {
    "op": "div",
    "numerator": {
        "op": "mul",
        "args": [
            {"var": "Eg"},
            {"var": "M"},
            {"op": "pow", "base": {"var": "D"}, "exponent": 3.0},
        ],
    },
    "denominator": {"var": "U"},
    "zero_policy": "zero_if_zero_over_zero_else_fail",
}

FLOAT_RE = re.compile(r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?|[-+]?inf|nan", re.IGNORECASE)
REFLECTION_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)(?:\s+|$)")
TIMESTAMP_RE = re.compile(r"(20\d{6}T\d{4,6}|20\d{2}[-_]\d{2}[-_]\d{2}[T_ -]\d{2}[:_-]?\d{2}(?::?\d{2})?)")


@dataclass(frozen=True)
class VariantTarget:
    label: str
    variant_id: str
    stream_name: str
    experiment_dir: Path
    source: str


@dataclass(frozen=True)
class MergeCandidate:
    label: str
    variant_id: str
    stream_path: Path | None
    merge_dir: Path | None
    source: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root.")
    parser.add_argument("--v6-dir", type=Path, default=None, help="V6 output directory. Defaults under --root.")
    parser.add_argument("--v5-dir", type=Path, default=None, help="Likely V5 p/lambda directory. Defaults under --root.")
    parser.add_argument("--source-stream", type=Path, default=None, help="Accepted-observation source stream. Defaults under --root.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Audit output directory. Defaults to --root/{AUDIT_DIRNAME}.",
    )
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Workers for read-only stream scans.")
    parser.add_argument(
        "--skip-stream-scan",
        action="store_true",
        help="Skip independent reflection-row counting of source/filtered streams.",
    )
    parser.add_argument(
        "--skip-selection-sets",
        action="store_true",
        help="Skip exact-key set loading/comparison from selected-observation manifests.",
    )
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    root = args.root.expanduser().resolve()
    args.root = root
    args.v6_dir = (args.v6_dir or root / "oridyn_v6_score_target_filter_map_20260716").expanduser().resolve()
    args.v5_dir = (args.v5_dir or root / "oridyn_v5_p_lambda_screen_20260716").expanduser().resolve()
    args.source_stream = (args.source_stream or root / "MFM300-VIII_cut_20-0_3.stream").expanduser().resolve()
    args.out_dir = (args.out_dir or root / AUDIT_DIRNAME).expanduser().resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
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
    return rounded if abs(float(number) - rounded) < 1e-6 else None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def read_csv_if_exists(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_text_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def parse_expression_tree(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def ensure_audit_output_is_safe(out_dir: Path, protected_paths: Iterable[Path]) -> None:
    for protected in protected_paths:
        try:
            protected_resolved = protected.resolve()
        except OSError:
            protected_resolved = protected
        if out_dir == protected_resolved:
            raise SystemExit(f"--out-dir must not be an existing input path: {out_dir}")
        if protected_resolved.is_dir():
            try:
                out_dir.relative_to(protected_resolved)
            except ValueError:
                pass
            else:
                raise SystemExit(f"--out-dir must not be inside protected input directory: {protected_resolved}")
    out_dir.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def file_fingerprint(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": path.exists(), "sha256": ""}
    if not path.exists():
        return row
    stat = path.stat()
    row.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        }
    )
    if hash_file and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        row["sha256"] = digest.hexdigest()
    return row


def metadata_inventory(v6_dir: Path, v5_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, base, names in [
        ("v6", v6_dir, METADATA_FILES),
        (
            "v5",
            v5_dir,
            [
                "experiment_plan.csv",
                "experiment_plan.json",
                "per_variant_filter_summary.csv",
                "per_variant_per_hkl_qc.csv",
                "per_variant_per_block_qc.csv",
                "selected_removal_observations.csv",
                "parameters.json",
                "scores.json",
                "validation.json",
                "run_metadata.json",
                "run.log",
                "stream_rewrite_qc.csv",
                "summary.csv",
                "score_comparison_by_current_reference.csv",
            ],
        ),
    ]:
        for name in names:
            path = base / name
            row = {"dataset": label, "file": name}
            row.update(file_fingerprint(path))
            rows.append(row)
    return rows


def first_present(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            if str(value) != "":
                return value
    return None


def find_v5_equivalent(v5_dir: Path) -> dict[str, Any]:
    plan = read_csv_if_exists(v5_dir / "experiment_plan.csv")
    summary = read_csv_if_exists(v5_dir / "per_variant_filter_summary.csv")
    if plan.empty:
        return {
            "variant_id": "",
            "stream_name": "",
            "formula": "",
            "match_method": "missing_v5_experiment_plan",
            "candidate_count": 0,
        }

    target_tree = canonical_json(CANONICAL_V6_EG_D3_CMEAN_TREE)
    candidates = []
    for idx, row in plan.iterrows():
        tree = parse_expression_tree(row.get("expression_tree_json"))
        tree_match = canonical_json(tree) == target_tree if tree is not None else False
        formula = str(row.get("formula", ""))
        formula_match = re.sub(r"\s+", "", formula).lower() in {
            "eg*m*d^3/u",
            "eg*m*d**3/u",
        }
        high_eg = to_float(row.get("high_eg_fraction"))
        drop = to_float(row.get("drop_fraction"))
        block_size = to_int(row.get("excitation_block_size"))
        min_final = to_int(row.get("min_final_block_size"))
        min_high = to_int(row.get("min_high_eg_observations"))
        min_remaining = to_int(row.get("min_remaining_per_block"))
        parameter_match = (
            high_eg is not None
            and math.isclose(high_eg, 0.30)
            and drop is not None
            and math.isclose(drop, 0.30)
            and block_size == 10
            and min_final == 5
            and min_high == 10
            and min_remaining == 2
        )
        if (tree_match or formula_match) and parameter_match:
            candidates.append((idx, row, "expression_tree" if tree_match else "formula"))

    if not candidates:
        return {
            "variant_id": "",
            "stream_name": "",
            "formula": "",
            "match_method": "no_formula_and_parameter_match",
            "candidate_count": 0,
        }
    idx, row, method = candidates[0]
    variant = str(first_present(row, ["experiment", "variant", "variant_id"]) or "")
    stream_name = str(first_present(row, ["expected_output_filename", "output_stream", "stream_filename"]) or f"{variant}.stream")
    result: dict[str, Any] = {
        "variant_id": variant,
        "stream_name": Path(stream_name).name,
        "formula": str(row.get("formula", "")),
        "match_method": method,
        "candidate_count": len(candidates),
        "plan_row_index": int(idx),
    }
    if not summary.empty:
        key = "variant" if "variant" in summary.columns else "variant_id"
        match = summary.loc[summary[key].astype(str) == variant]
        if not match.empty:
            result.update({f"summary_{k}": v for k, v in match.iloc[0].to_dict().items()})
    return result


def build_variant_targets(v6_dir: Path, v5_dir: Path, v5_equivalent: dict[str, Any]) -> list[VariantTarget]:
    rows: list[VariantTarget] = []
    for variant_id in [*V6_PRIMARY_VARIANTS, *V6_SECONDARY_VARIANTS]:
        rows.append(
            VariantTarget(
                label=variant_id,
                variant_id=variant_id,
                stream_name=f"{variant_id}.stream",
                experiment_dir=v6_dir,
                source="v6",
            )
        )
    if v5_equivalent.get("variant_id"):
        rows.append(
            VariantTarget(
                label=V5_EQUIVALENT_LABEL,
                variant_id=str(v5_equivalent["variant_id"]),
                stream_name=str(v5_equivalent["stream_name"]),
                experiment_dir=v5_dir,
                source="v5",
            )
        )
    return rows


def stream_path_for(target: VariantTarget) -> Path:
    stream = Path(target.stream_name)
    return stream if stream.is_absolute() else target.experiment_dir / stream.name


def candidate_sort_key(path: Path) -> tuple[int, float, str]:
    complete_score = sum(1 for rel in MERGE_REQUIRED_FILES if (path / rel).exists())
    timestamp = path.stat().st_mtime if path.exists() else 0.0
    return (complete_score, timestamp, path.name)


def locate_merge_dir(experiment_dir: Path, stream_name: str) -> tuple[Path | None, list[Path]]:
    stem = Path(stream_name).stem
    candidates = sorted(experiment_dir.glob(f"{stem}_partialator_results*"))
    if not candidates:
        candidates = sorted(experiment_dir.glob(f"*{stem}*partialator_results*"))
    if not candidates:
        return None, []
    best = sorted(candidates, key=candidate_sort_key, reverse=True)[0]
    return best, candidates


def discover_full_reference(root: Path, source_stream: Path) -> tuple[MergeCandidate | None, list[dict[str, Any]]]:
    stem = source_stream.stem
    patterns = [
        f"{stem}_partialator_results*",
        f"*{stem}*partialator_results*",
        "*full*partialator_results*",
        "*full*",
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                found[str(path.resolve())] = path.resolve()
        for path in root.glob(f"*/{pattern}"):
            if path.is_dir():
                found[str(path.resolve())] = path.resolve()
        for path in root.glob(f"*/*/{pattern}"):
            if path.is_dir():
                found[str(path.resolve())] = path.resolve()
    rows: list[dict[str, Any]] = []
    best: tuple[float, Path] | None = None
    for path in found.values():
        if "oridyn_v5_p_lambda_screen_20260716" in str(path) or "oridyn_v6_score_target_filter_map_20260716" in str(path):
            continue
        metrics, _shells, _settings, warnings = parse_merge_outputs(path, label="candidate_full", variant_id="")
        obs = to_float(metrics.get("observation_count"))
        red = to_float(metrics.get("redundancy"))
        cc12 = to_float(metrics.get("cc12"))
        rsplit = to_float(metrics.get("rsplit"))
        score = 0.0
        if obs is not None:
            score += min(abs(obs - 6_732_955.0) / 6_732_955.0, 1.0)
        else:
            score += 1.0
        if red is not None:
            score += min(abs(red - 534.39) / 534.39, 1.0)
        else:
            score += 1.0
        if cc12 is not None:
            score += min(abs(cc12 - 0.9970285) / 0.01, 1.0)
        if rsplit is not None:
            score += min(abs(rsplit - 5.45) / 5.45, 1.0)
        row = {
            "merge_dir": str(path),
            "candidate_score": score,
            "observation_count": obs,
            "redundancy": red,
            "cc12": cc12,
            "rsplit": rsplit,
            "warning_count": len(warnings),
        }
        rows.append(row)
        if best is None or score < best[0]:
            best = (score, path)
    rows.sort(key=lambda item: (float(item.get("candidate_score") or 999.0), str(item.get("merge_dir"))))
    if best is None:
        return None, rows
    return (
        MergeCandidate(
            label=FULL_REFERENCE_LABEL,
            variant_id=FULL_REFERENCE_LABEL,
            stream_path=source_stream,
            merge_dir=best[1],
            source="full_reference_discovery",
        ),
        rows,
    )


def parse_metadata_headlines(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in read_text_lines(path):
        if match := re.search(r"Completeness:\s*(\S+)%", line, re.IGNORECASE):
            metrics["completeness"] = to_float(match.group(1))
        elif match := re.search(r"Redundancy:\s*(\S+)", line, re.IGNORECASE):
            metrics["redundancy"] = to_float(match.group(1).rstrip("xX"))
        elif match := re.search(r"SNR:\s*(\S+)", line, re.IGNORECASE):
            metrics["snr"] = to_float(match.group(1))
        elif match := re.search(r"CC1/2:\s*(\S+)", line, re.IGNORECASE):
            metrics["cc12"] = to_float(match.group(1))
        elif match := re.search(r"Rsplit:\s*(\S+)", line, re.IGNORECASE):
            metrics["rsplit"] = to_float(match.group(1))
    return metrics


def decimal_places(token: str) -> int | None:
    if "." not in token or "e" in token.lower():
        return None
    return len(token.split(".", 1)[1].rstrip("% "))


def parse_check_hkl_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in read_text_lines(path):
        if match := re.search(r"(\d+)\s+measurements\s+in\s+total", line):
            metrics["observation_count"] = int(match.group(1))
        elif match := re.search(r"(\d+)\s+reflections\s+in\s+total", line):
            metrics["merged_reflection_count"] = int(match.group(1))
        elif match := re.search(r"Overall\s+<snr>\s*=\s*(\S+)", line, re.IGNORECASE):
            metrics["snr"] = to_float(match.group(1))
            metrics["snr_precision"] = f"decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(r"Overall\s+redundancy\s*=\s*(\S+)", line, re.IGNORECASE):
            metrics["redundancy"] = to_float(match.group(1))
            metrics["redundancy_precision"] = f"decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(r"Overall\s+completeness\s*=\s*(\S+)", line, re.IGNORECASE):
            metrics["completeness"] = to_float(match.group(1))
            metrics["completeness_precision"] = f"decimal_places_{decimal_places(match.group(1))}"
    return metrics


def parse_compare_log(path: Path, metric: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in read_text_lines(path):
        if metric == "cc12" and (match := re.search(r"Overall\s+CC\s*=\s*(\S+)", line, re.IGNORECASE)):
            metrics["cc12"] = to_float(match.group(1))
            metrics["cc12_precision"] = f"decimal_places_{decimal_places(match.group(1))}"
        elif metric == "rsplit" and (match := re.search(r"Overall\s+Rsplit\s*=\s*(\S+)", line, re.IGNORECASE)):
            metrics["rsplit"] = to_float(match.group(1))
            metrics["rsplit_precision"] = f"decimal_places_{decimal_places(match.group(1))}"
        elif match := re.search(
            r"Accepted\s+resolution\s+range:\s+(\S+)\s+to\s+(\S+)\s+nm\^-1\s+\((\S+)\s+to\s+(\S+)\s+Angstroms\)",
            line,
            re.IGNORECASE,
        ):
            metrics["low_resolution_limit_A"] = to_float(match.group(3))
            metrics["high_resolution_limit_A"] = to_float(match.group(4))
    return metrics


def numbers_from_line(line: str) -> list[float]:
    values: list[float] = []
    for match in FLOAT_RE.finditer(line):
        value = to_float(match.group(0))
        if value is not None:
            values.append(value)
    return values


def parse_shell_table(path: Path, kind: str) -> list[dict[str, Any]]:
    lines = read_text_lines(path)
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or not re.match(r"[-+.\d]", stripped):
            continue
        values = numbers_from_line(stripped)
        if kind == "check_shell" and len(values) >= 11:
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
                }
            )
        elif kind == "cc12_shell" and len(values) >= 6:
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
                }
            )
        elif kind == "rsplit_shell" and len(values) >= 6:
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
                }
            )
    return rows


def boundary_key(row: dict[str, Any], places: int = 6) -> tuple[float, float] | None:
    min_inv = to_float(row.get("min_invnm"))
    max_inv = to_float(row.get("max_invnm"))
    if min_inv is None or max_inv is None:
        return None
    return (round(min_inv, places), round(max_inv, places))


def join_shells(check_rows: list[dict[str, Any]], cc_rows: list[dict[str, Any]], rsplit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def indexed(rows: list[dict[str, Any]]) -> dict[tuple[float, float], dict[str, Any]]:
        out: dict[tuple[float, float], dict[str, Any]] = {}
        for row in rows:
            key = boundary_key(row)
            if key is not None:
                out[key] = row
        return out

    check_index = indexed(check_rows)
    cc_index = indexed(cc_rows)
    rsplit_index = indexed(rsplit_rows)
    keys = sorted(set(check_index) | set(cc_index) | set(rsplit_index))
    out: list[dict[str, Any]] = []
    for idx, key in enumerate(keys, start=1):
        row: dict[str, Any] = {
            "shell_index": idx,
            "min_invnm": key[0],
            "max_invnm": key[1],
            "shell_lower_resolution_A": 10.0 / key[0] if key[0] else "",
            "shell_upper_resolution_A": 10.0 / key[1] if key[1] else "",
        }
        row.update(check_index.get(key, {}))
        for name in ["cc12", "cc12_nref"]:
            if key in cc_index:
                row[name] = cc_index[key].get(name)
        for name in ["rsplit", "rsplit_nref"]:
            if key in rsplit_index:
                row[name] = rsplit_index[key].get(name)
        out.append(row)
    return out


def find_first_key(data: Any, wanted: str) -> Any:
    if isinstance(data, dict):
        if wanted in data:
            return data[wanted]
        for value in data.values():
            found = find_first_key(value, wanted)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_first_key(value, wanted)
            if found is not None:
                return found
    return None


def extract_merge_settings(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {}
    wrapper = parameters.get("merge_wrapper", {}) if isinstance(parameters.get("merge_wrapper"), dict) else {}
    merging = parameters.get("merging", {}) if isinstance(parameters.get("merging"), dict) else {}
    settings = {
        "stream_file": wrapper.get("stream_file") or find_first_key(parameters, "stream_file"),
        "symmetry": wrapper.get("symmetry") or merging.get("symmetry"),
        "partialator_model": wrapper.get("model") or merging.get("partiality_model"),
        "iterations": wrapper.get("iterations") or merging.get("num_iterations"),
        "min_measurements": wrapper.get("min_measurements") or merging.get("min_measurements_per_unique_reflection"),
        "highres": wrapper.get("highres"),
        "lowres": wrapper.get("lowres"),
        "push_res": wrapper.get("push_res"),
        "min_res": wrapper.get("min_res"),
        "polarisation": wrapper.get("polarisation"),
        "no_bscale": wrapper.get("no_bscale"),
        "disable_pr": wrapper.get("disable_pr"),
        "partialator_command": wrapper.get("partialator_command"),
    }
    half_keys: dict[str, Any] = {}
    for key in ["seed", "random_seed", "half", "half_seed", "split_seed"]:
        found = find_first_key(parameters, key)
        if found is not None:
            half_keys[key] = found
    settings["half_set_related_keys_json"] = canonical_json(half_keys)
    settings["normalized_partialator_options"] = normalize_partialator_command(str(settings.get("partialator_command") or ""))
    return settings


def normalize_partialator_command(command: str) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    normalized: list[str] = []
    skip_next = False
    path_options = {"-o", "--harvest-file", "--log-folder"}
    value_path_prefixes = ("-o=", "--harvest-file=", "--log-folder=")
    for idx, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if idx == 1 and not part.startswith("-"):
            normalized.append("<stream>")
            continue
        if part in path_options:
            normalized.extend([part, "<path>"])
            skip_next = True
            continue
        if part.startswith(value_path_prefixes):
            normalized.append(part.split("=", 1)[0] + "=<path>")
            continue
        normalized.append(part)
    return " ".join(normalized)


def parse_merge_outputs(merge_dir: Path, label: str, variant_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "label": label,
        "variant_id": variant_id,
        "merge_dir": str(merge_dir),
        "merge_dir_exists": merge_dir.is_dir(),
    }
    if not merge_dir.is_dir():
        warnings.append({"label": label, "severity": "error", "message": "merge_dir_missing", "path": str(merge_dir)})
        return metrics, [], {}, warnings
    missing = [rel for rel in MERGE_REQUIRED_FILES if not (merge_dir / rel).exists()]
    metrics["missing_required_files"] = "; ".join(missing)
    metrics["required_files_present"] = not missing
    for rel in missing:
        warnings.append({"label": label, "severity": "warning", "message": f"missing merge output: {rel}", "path": str(merge_dir / rel)})

    params = read_json(merge_dir / "parameters.json")
    settings = extract_merge_settings(params)
    metadata_metrics = parse_metadata_headlines(merge_dir / "metadata_and_outputs.txt")
    check_metrics = parse_check_hkl_log(merge_dir / "qc_stats/check_hkl_completeness.log")
    cc_metrics = parse_compare_log(merge_dir / "qc_stats/compare_cc12.log", "cc12")
    rsplit_metrics = parse_compare_log(merge_dir / "qc_stats/compare_rsplit.log", "rsplit")
    for source in [metadata_metrics, check_metrics, cc_metrics, rsplit_metrics]:
        for key, value in source.items():
            metrics[key] = value

    shell_rows = join_shells(
        parse_shell_table(merge_dir / "qc_stats/check_shell.tsv", "check_shell"),
        parse_shell_table(merge_dir / "qc_stats/compare_cc12_shell.tsv", "cc12_shell"),
        parse_shell_table(merge_dir / "qc_stats/compare_rsplit_shell.tsv", "rsplit_shell"),
    )
    for row in shell_rows:
        row["label"] = label
        row["variant_id"] = variant_id
        row["merge_dir"] = str(merge_dir)
    metrics["number_of_shells"] = len(shell_rows)
    metrics.update(settings)
    return metrics, shell_rows, settings, warnings


def selection_manifest_path(experiment_dir: Path, source: str) -> Path | None:
    if source == "v6":
        path = experiment_dir / "selected_observations.csv.gz"
        return path if path.exists() else None
    path = experiment_dir / "selected_removal_observations.csv"
    return path if path.exists() else None


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def count_selection_rows(path: Path, source: str, variant_ids: set[str], load_sets: bool) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    counts: dict[str, dict[str, Any]] = {
        variant: {
            "selection_manifest_rows": 0,
            "removed_manifest_rows": 0,
            "eligible_retained_manifest_rows": 0,
            "unique_removed_exact_keys": 0,
        }
        for variant in variant_ids
    }
    sets: dict[str, set[str]] = {variant: set() for variant in variant_ids}
    if not path.is_file():
        return counts, sets
    variant_column = "variant_id" if source == "v6" else "variant"
    with open_maybe_gzip(path) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or variant_column not in reader.fieldnames:
            return counts, sets
        for row in reader:
            variant = str(row.get(variant_column, ""))
            if variant not in variant_ids:
                continue
            state = str(row.get("state", "removed_by_targeted_filter"))
            counts[variant]["selection_manifest_rows"] += 1
            removed = source == "v5" or state == "removed_by_targeted_filter"
            if state == "eligible_retained_after_targeted_filter":
                counts[variant]["eligible_retained_manifest_rows"] += 1
            if removed:
                counts[variant]["removed_manifest_rows"] += 1
                if load_sets:
                    key = str(row.get("exact_key_text", ""))
                    if not key:
                        key = exact_key_from_row(row)
                    if key:
                        sets[variant].add(key)
    for variant, key_set in sets.items():
        counts[variant]["unique_removed_exact_keys"] = len(key_set) if load_sets else ""
    return counts, sets


def exact_key_from_row(row: dict[str, Any]) -> str:
    source = str(row.get("source_filename", ""))
    event = normalize_event(row.get("event", ""))
    h = str(row.get("h", ""))
    k = str(row.get("k", ""))
    l = str(row.get("l", ""))
    if not source or not h or not k or not l:
        return ""
    return f"{source}|{event}|{h}|{k}|{l}"


def normalize_event(value: Any) -> str:
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def aggregate_filtered_csv(path: Path, variant_ids: set[str], variant_column: str, numeric_columns: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {variant: {f"sum_{col}": 0 for col in numeric_columns} for variant in variant_ids}
    if not path.is_file():
        return out
    usecols = None
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [column for column in [variant_column, *numeric_columns, "actionable", "validation_passed", "block_size"] if column in header]
    except Exception:
        usecols = None
    for chunk in pd.read_csv(path, chunksize=250_000, usecols=usecols, low_memory=False):
        if variant_column not in chunk.columns:
            continue
        subset = chunk.loc[chunk[variant_column].astype(str).isin(variant_ids)].copy()
        if subset.empty:
            continue
        for variant, group in subset.groupby(subset[variant_column].astype(str), sort=False):
            record = out.setdefault(variant, {f"sum_{col}": 0 for col in numeric_columns})
            record["row_count"] = int(record.get("row_count", 0)) + int(len(group))
            for col in numeric_columns:
                if col in group.columns:
                    record[f"sum_{col}"] = int(record.get(f"sum_{col}", 0)) + int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
            if "actionable" in group.columns:
                record["actionable_true_rows"] = int(record.get("actionable_true_rows", 0)) + int(group["actionable"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            if "validation_passed" in group.columns:
                record["validation_failed_rows"] = int(record.get("validation_failed_rows", 0)) + int((~group["validation_passed"].astype(str).str.lower().isin(["true", "1", "yes"])).sum())
            if "block_size" in group.columns:
                sizes = pd.to_numeric(group["block_size"], errors="coerce").dropna()
                if not sizes.empty:
                    record["block_size_min"] = min(to_float(record.get("block_size_min")) or float("inf"), float(sizes.min()))
                    record["block_size_max"] = max(to_float(record.get("block_size_max")) or float("-inf"), float(sizes.max()))
                    record["block_size_median_last_chunk"] = float(sizes.median())
    return out


def count_stream_reflections(path: Path) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {"stream_path": str(path), "stream_exists": path.is_file(), "reflection_rows": ""}
    if not path.is_file():
        return row
    in_reflections = False
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Reflections measured after indexing" in line:
                in_reflections = True
                continue
            if in_reflections and line.startswith("End of reflections"):
                in_reflections = False
                continue
            if in_reflections and REFLECTION_RE.match(line):
                count += 1
    row["reflection_rows"] = int(count)
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


def load_manifest_rows(v6_dir: Path, v5_dir: Path, targets: list[VariantTarget]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {target.label: {} for target in targets}
    v6_manifest = read_csv_if_exists(v6_dir / "stream_manifest.csv")
    v6_summary = read_csv_if_exists(v6_dir / "per_variant_selection_summary.csv")
    v6_rewrite = read_csv_if_exists(v6_dir / "stream_rewrite_qc.csv")
    v5_summary = read_csv_if_exists(v5_dir / "per_variant_filter_summary.csv")
    v5_rewrite = read_csv_if_exists(v5_dir / "stream_rewrite_qc.csv")

    for target in targets:
        row: dict[str, Any] = {}
        if target.source == "v6":
            for prefix, frame, key in [
                ("stream_manifest", v6_manifest, "variant_id"),
                ("selection_summary", v6_summary, "variant_id"),
                ("stream_rewrite_qc", v6_rewrite, "variant_id"),
            ]:
                if frame.empty or key not in frame.columns:
                    continue
                match = frame.loc[frame[key].astype(str) == target.variant_id]
                if not match.empty:
                    row.update({f"{prefix}_{column}": value for column, value in match.iloc[0].to_dict().items()})
        else:
            for prefix, frame, key in [
                ("filter_summary", v5_summary, "variant"),
                ("stream_rewrite_qc", v5_rewrite, "variant"),
            ]:
                if frame.empty or key not in frame.columns:
                    continue
                match = frame.loc[frame[key].astype(str) == target.variant_id]
                if not match.empty:
                    row.update({f"{prefix}_{column}": value for column, value in match.iloc[0].to_dict().items()})
        records[target.label] = row
    return records


def make_removal_accounting(
    targets: list[VariantTarget],
    manifest_records: dict[str, dict[str, Any]],
    selection_counts: dict[str, dict[str, Any]],
    hkl_aggregates: dict[str, dict[str, Any]],
    block_aggregates: dict[str, dict[str, Any]],
    stream_counts: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    cache_rows = find_first_key(validation, "cache_rows")
    source_rows_meta = find_first_key(validation, "source_reflection_rows")
    source_without_cache = find_first_key(validation, "source_reflection_rows_without_cache_score")
    rows: list[dict[str, Any]] = []
    for target in targets:
        label = target.label
        manifest = manifest_records.get(label, {})
        stream_scan = stream_counts.get(label, {})
        selection = selection_counts.get(target.variant_id, {})
        hkl = hkl_aggregates.get(target.variant_id, {})
        block = block_aggregates.get(target.variant_id, {})
        stream_removed = first_nonempty(
            manifest.get("stream_rewrite_qc_removed_observations"),
            manifest.get("stream_rewrite_qc_requested_removals"),
            manifest.get("selection_summary_selected_or_removed_count"),
            manifest.get("filter_summary_actual_removal_count"),
        )
        row = {
            "label": label,
            "source": target.source,
            "variant_id": target.variant_id,
            "stream_name": target.stream_name,
            "stream_path": str(stream_path_for(target)),
            "removed_by_stream_rewrite_qc": stream_removed,
            "removed_by_selection_manifest_rows": selection.get("removed_manifest_rows", ""),
            "unique_removed_exact_keys_loaded": selection.get("unique_removed_exact_keys", ""),
            "selection_manifest_rows": selection.get("selection_manifest_rows", ""),
            "eligible_retained_manifest_rows": selection.get("eligible_retained_manifest_rows", ""),
            "sum_hkl_qc_n_removed": hkl.get("sum_n_removed", ""),
            "hkl_qc_rows": hkl.get("row_count", ""),
            "sum_block_qc_n_removed": block.get("sum_n_removed", ""),
            "block_qc_rows": block.get("row_count", ""),
            "stream_scan_reflection_rows": stream_scan.get("reflection_rows", ""),
            "source_reflection_rows_from_v6_validation": source_rows_meta if target.source == "v6" else "",
            "v6_cache_rows": cache_rows if target.source == "v6" else "",
            "source_reflection_rows_without_cache_score": source_without_cache if target.source == "v6" else "",
        }
        if target.source == "v6":
            removed = to_float(stream_removed)
            if removed is not None and to_float(source_rows_meta):
                row["fraction_of_source_stream_rows_removed"] = removed / float(source_rows_meta)
            if removed is not None and to_float(cache_rows):
                row["fraction_of_v6_cache_rows_removed"] = removed / float(cache_rows)
        rows.append(row)
    return rows


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if str(value) != "":
            return value
    return ""


def compare_sets(selection_sets: dict[str, set[str]], d3_variant: str, v5_variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("v6_d2_vs_v6_d3", V6_PRIMARY_VARIANTS[0], V6_PRIMARY_VARIANTS[1]),
        ("v6_d3_vs_v5_equivalent", d3_variant, v5_variant),
    ]
    for label, left, right in pairs:
        left_set = selection_sets.get(left, set())
        right_set = selection_sets.get(right, set())
        if not left_set and not right_set:
            rows.append({"comparison": label, "left_variant": left, "right_variant": right, "status": "no_sets_loaded"})
            continue
        intersection = len(left_set & right_set)
        union = len(left_set | right_set)
        rows.append(
            {
                "comparison": label,
                "left_variant": left,
                "right_variant": right,
                "left_count": len(left_set),
                "right_count": len(right_set),
                "intersection_count": intersection,
                "left_only_count": len(left_set - right_set),
                "right_only_count": len(right_set - left_set),
                "jaccard": intersection / union if union else 1.0,
                "exactly_equal": left_set == right_set,
                "status": "compared",
            }
        )
    return rows


def make_design_audit(v6_dir: Path, v5_dir: Path, v5_equivalent: dict[str, Any], block_aggregates: dict[str, dict[str, Any]], hkl_aggregates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v6_parameters = read_json(v6_dir / "parameters.json") or {}
    v6_filtering = v6_parameters.get("filtering_parameters", {}) if isinstance(v6_parameters, dict) else {}
    for variant in V6_PRIMARY_VARIANTS:
        block = block_aggregates.get(variant, {})
        hkl = hkl_aggregates.get(variant, {})
        rows.append(
            {
                "source": "v6",
                "variant_id": variant,
                "formula_or_score": variant.replace("filter_matched_", "").replace("_drop30", ""),
                "high_eg_fraction": v6_filtering.get("high_Eg_fraction"),
                "excitation_block_size": v6_filtering.get("excitation_block_size"),
                "min_final_block_size": v6_filtering.get("min_final_block_size"),
                "min_high_eg_observations": v6_filtering.get("min_high_Eg_observations"),
                "min_remaining": v6_filtering.get("min_remaining"),
                "block_qc_rows": block.get("row_count", ""),
                "sum_block_qc_n_removed": block.get("sum_n_removed", ""),
                "block_size_min": block.get("block_size_min", ""),
                "block_size_max": block.get("block_size_max", ""),
                "validation_failed_block_rows": block.get("validation_failed_rows", ""),
                "hkl_qc_rows": hkl.get("row_count", ""),
                "sum_hkl_qc_n_removed": hkl.get("sum_n_removed", ""),
            }
        )
    if v5_equivalent.get("variant_id"):
        rows.append(
            {
                "source": "v5",
                "variant_id": v5_equivalent.get("variant_id"),
                "formula_or_score": v5_equivalent.get("formula"),
                "high_eg_fraction": v5_equivalent.get("summary_high_eg_fraction"),
                "excitation_block_size": v5_equivalent.get("summary_excitation_block_size"),
                "min_final_block_size": 5,
                "min_high_eg_observations": 10,
                "min_remaining": 2,
                "common_high_Eg_observation_count": v5_equivalent.get("summary_common_high_Eg_observation_count"),
                "common_excitation_block_count": v5_equivalent.get("summary_common_excitation_block_count"),
                "common_actionable_block_count": v5_equivalent.get("summary_common_actionable_block_count"),
                "expected_removal_count": v5_equivalent.get("summary_expected_removal_count"),
                "actual_removal_count": v5_equivalent.get("summary_actual_removal_count"),
                "block_removal_counts_equal": v5_equivalent.get("summary_block_removal_counts_equal"),
                "hkl_removal_counts_equal": v5_equivalent.get("summary_hkl_removal_counts_equal"),
            }
        )
    return rows


def make_merge_candidates(targets: list[VariantTarget], full_reference: MergeCandidate | None) -> tuple[list[MergeCandidate], list[dict[str, Any]]]:
    candidates: list[MergeCandidate] = []
    discovery_rows: list[dict[str, Any]] = []
    for target in targets:
        stream_path = stream_path_for(target)
        merge_dir, all_candidates = locate_merge_dir(target.experiment_dir, target.stream_name)
        discovery_rows.append(
            {
                "label": target.label,
                "variant_id": target.variant_id,
                "stream_path": str(stream_path),
                "selected_merge_dir": str(merge_dir or ""),
                "candidate_count": len(all_candidates),
                "all_candidates": "; ".join(str(path) for path in all_candidates),
            }
        )
        candidates.append(MergeCandidate(target.label, target.variant_id, stream_path, merge_dir, target.source))
    if full_reference is not None:
        candidates.append(full_reference)
    return candidates, discovery_rows


def compare_merge_settings(global_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {str(row.get("label")): row for row in global_rows}
    pairs = [
        ("v6_d2_vs_v6_d3", V6_PRIMARY_VARIANTS[0], V6_PRIMARY_VARIANTS[1]),
        ("v6_d3_vs_v5_equivalent", V6_PRIMARY_VARIANTS[1], V5_EQUIVALENT_LABEL),
    ]
    setting_keys = [
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
    for comparison, left_label, right_label in pairs:
        left = by_label.get(left_label, {})
        right = by_label.get(right_label, {})
        row = {"comparison": comparison, "left_label": left_label, "right_label": right_label}
        mismatches = []
        for key in setting_keys:
            equal = str(left.get(key, "")) == str(right.get(key, ""))
            row[f"{key}_equal"] = equal
            if not equal:
                mismatches.append(key)
        row["all_checked_settings_equal"] = not mismatches
        row["mismatched_settings"] = "; ".join(mismatches)
        rows.append(row)
    return rows


def make_file_fingerprints(candidates: list[MergeCandidate], source_stream: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_row = {"label": "source_stream", "file_role": "source_stream"}
    source_row.update(file_fingerprint(source_stream, hash_file=False))
    rows.append(source_row)
    for candidate in candidates:
        if candidate.stream_path is not None:
            row = {"label": candidate.label, "variant_id": candidate.variant_id, "file_role": "filtered_stream"}
            row.update(file_fingerprint(candidate.stream_path, hash_file=False))
            rows.append(row)
        if candidate.merge_dir is None:
            continue
        for rel in ["parameters.json", "cell.cell", "partialator_stdout.log", "partialator_stderr.log", "crystfel.hkl", "crystfel.hkl1", "crystfel.hkl2"]:
            row = {"label": candidate.label, "variant_id": candidate.variant_id, "file_role": rel}
            row.update(file_fingerprint(candidate.merge_dir / rel, hash_file=False))
            rows.append(row)
    return rows


def delta(value: Any, reference: Any) -> float | str:
    left = to_float(value)
    right = to_float(reference)
    if left is None or right is None:
        return ""
    return left - right


def make_shell_deltas(shell_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label = {}
    for row in shell_rows:
        by_label.setdefault(str(row.get("label")), []).append(row)
    ref = by_label.get(FULL_REFERENCE_LABEL, [])
    ref_by_boundary = {boundary_key(row): row for row in ref if boundary_key(row) is not None}
    deltas: list[dict[str, Any]] = []
    for label in [*V6_PRIMARY_VARIANTS, V5_EQUIVALENT_LABEL, *V6_SECONDARY_VARIANTS]:
        for row in by_label.get(label, []):
            key = boundary_key(row)
            ref_row = ref_by_boundary.get(key)
            out = dict(row)
            out["reference_label"] = FULL_REFERENCE_LABEL if ref_row else ""
            out["delta_cc12_vs_reference"] = delta(row.get("cc12"), ref_row.get("cc12") if ref_row else None)
            out["delta_rsplit_vs_reference"] = delta(row.get("rsplit"), ref_row.get("rsplit") if ref_row else None)
            out["delta_redundancy_vs_reference"] = delta(row.get("redundancy"), ref_row.get("redundancy") if ref_row else None)
            out["delta_snr_vs_reference"] = delta(row.get("snr"), ref_row.get("snr") if ref_row else None)
            out["delta_completeness_vs_reference"] = delta(row.get("completeness"), ref_row.get("completeness") if ref_row else None)
            deltas.append(out)

    d2_by_boundary = {boundary_key(row): row for row in by_label.get(V6_PRIMARY_VARIANTS[0], []) if boundary_key(row) is not None}
    d3_by_boundary = {boundary_key(row): row for row in by_label.get(V6_PRIMARY_VARIANTS[1], []) if boundary_key(row) is not None}
    d2d3: list[dict[str, Any]] = []
    for idx, key in enumerate(sorted(set(d2_by_boundary) & set(d3_by_boundary)), start=1):
        d2 = d2_by_boundary[key]
        d3 = d3_by_boundary[key]
        d2_rsplit = to_float(d2.get("rsplit"))
        d3_rsplit = to_float(d3.get("rsplit"))
        d2_cc = to_float(d2.get("cc12"))
        d3_cc = to_float(d3.get("cc12"))
        d2d3.append(
            {
                "shell_index": idx,
                "min_invnm": key[0],
                "max_invnm": key[1],
                "shell_lower_resolution_A": 10.0 / key[0] if key[0] else "",
                "shell_upper_resolution_A": 10.0 / key[1] if key[1] else "",
                "d2_rsplit": d2_rsplit,
                "d3_rsplit": d3_rsplit,
                "d2_minus_d3_rsplit": delta(d2_rsplit, d3_rsplit),
                "rsplit_winner": "d2" if d2_rsplit is not None and d3_rsplit is not None and d2_rsplit < d3_rsplit else ("d3" if d2_rsplit is not None and d3_rsplit is not None and d3_rsplit < d2_rsplit else "tie_or_missing"),
                "d2_cc12": d2_cc,
                "d3_cc12": d3_cc,
                "d2_minus_d3_cc12": delta(d2_cc, d3_cc),
                "cc12_winner": "d2" if d2_cc is not None and d3_cc is not None and d2_cc > d3_cc else ("d3" if d2_cc is not None and d3_cc is not None and d3_cc > d2_cc else "tie_or_missing"),
            }
        )
    return deltas, d2d3


def summarize_shell_behavior(shell_deltas: list[dict[str, Any]], d2d3_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in [*V6_PRIMARY_VARIANTS, V5_EQUIVALENT_LABEL]:
        subset = [row for row in shell_deltas if row.get("label") == label]
        with_ref = [row for row in subset if to_float(row.get("delta_rsplit_vs_reference")) is not None]
        improved = [row for row in with_ref if to_float(row.get("delta_rsplit_vs_reference")) is not None and float(row["delta_rsplit_vs_reference"]) < 0]
        worsened = [row for row in with_ref if to_float(row.get("delta_rsplit_vs_reference")) is not None and float(row["delta_rsplit_vs_reference"]) > 0]
        outer = subset[-1] if subset else {}
        rows.append(
            {
                "label": label,
                "shell_count": len(subset),
                "shells_with_reference": len(with_ref),
                "rsplit_improved_shells_vs_reference": len(improved),
                "rsplit_worsened_shells_vs_reference": len(worsened),
                "rsplit_improvement_fraction": len(improved) / len(with_ref) if with_ref else "",
                "outer_shell_delta_rsplit_vs_reference": outer.get("delta_rsplit_vs_reference", ""),
                "outer_shell_delta_cc12_vs_reference": outer.get("delta_cc12_vs_reference", ""),
                "outer_shell_delta_redundancy_vs_reference": outer.get("delta_redundancy_vs_reference", ""),
                "outer_shell_delta_snr_vs_reference": outer.get("delta_snr_vs_reference", ""),
            }
        )
    rows.append(
        {
            "label": "d2_vs_d3",
            "shell_count": len(d2d3_rows),
            "rsplit_d2_wins": sum(1 for row in d2d3_rows if row.get("rsplit_winner") == "d2"),
            "rsplit_d3_wins": sum(1 for row in d2d3_rows if row.get("rsplit_winner") == "d3"),
            "cc12_d2_wins": sum(1 for row in d2d3_rows if row.get("cc12_winner") == "d2"),
            "cc12_d3_wins": sum(1 for row in d2d3_rows if row.get("cc12_winner") == "d3"),
        }
    )
    return rows


def make_interpretation_summary(
    removal_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    selection_comparison_rows: list[dict[str, Any]],
    merge_setting_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    shell_summary_rows: list[dict[str, Any]],
    v5_equivalent: dict[str, Any],
) -> dict[str, Any]:
    by_label = {row["label"]: row for row in removal_rows}
    globals_by_label = {str(row.get("label")): row for row in global_rows}
    d2_removed = by_label.get(V6_PRIMARY_VARIANTS[0], {}).get("removed_by_stream_rewrite_qc", "")
    d3_removed = by_label.get(V6_PRIMARY_VARIANTS[1], {}).get("removed_by_stream_rewrite_qc", "")
    v5_removed = by_label.get(V5_EQUIVALENT_LABEL, {}).get("removed_by_stream_rewrite_qc", "") or v5_equivalent.get("summary_actual_removal_count", "")
    d3_vs_v5 = next((row for row in selection_comparison_rows if row.get("comparison") == "v6_d3_vs_v5_equivalent"), {})
    settings_d3_v5 = next((row for row in merge_setting_rows if row.get("comparison") == "v6_d3_vs_v5_equivalent"), {})
    return {
        "questions": {
            "1_v6_matched_drop30_removed": {
                "d2_removed_by_stream_qc": d2_removed,
                "d3_removed_by_stream_qc": d3_removed,
                "note": "Use removal_accounting.csv for stream rows, V6 cache denominator, and selected-manifest counts.",
            },
            "2_v6_design": {
                "status": "see design_audit.csv",
                "v6_design_rows": [row for row in design_rows if row.get("source") == "v6"],
            },
            "3_v6_d3_equivalent_to_v5_p1": {
                "v5_equivalent_variant": v5_equivalent.get("variant_id", ""),
                "v5_equivalent_match_method": v5_equivalent.get("match_method", ""),
                "v6_d3_vs_v5_removed_set_equal": d3_vs_v5.get("exactly_equal", ""),
                "v6_d3_removed_count": d3_vs_v5.get("left_count", ""),
                "v5_removed_count": d3_vs_v5.get("right_count", v5_removed),
            },
            "4_redundancy_discrepancy": {
                "v6_d3_redundancy": globals_by_label.get(V6_PRIMARY_VARIANTS[1], {}).get("redundancy", ""),
                "v5_equivalent_redundancy": globals_by_label.get(V5_EQUIVALENT_LABEL, {}).get("redundancy", ""),
                "likely_denominator_issue": bool(to_float(d3_removed) is not None and to_float(v5_removed) is not None and to_float(d3_removed) != to_float(v5_removed)),
            },
            "5_inputs_settings_halfsets": {
                "merge_settings_equal_d3_vs_v5": settings_d3_v5.get("all_checked_settings_equal", ""),
                "mismatched_settings": settings_d3_v5.get("mismatched_settings", ""),
                "note": "Half-set identity is inferred from settings and hkl1/hkl2 fingerprints; no Partialator half-set regeneration is done.",
            },
            "6_full_precision_global_metrics": {
                label: {
                    "cc12": globals_by_label.get(label, {}).get("cc12", ""),
                    "cc12_precision": globals_by_label.get(label, {}).get("cc12_precision", ""),
                    "rsplit": globals_by_label.get(label, {}).get("rsplit", ""),
                    "rsplit_precision": globals_by_label.get(label, {}).get("rsplit_precision", ""),
                    "redundancy": globals_by_label.get(label, {}).get("redundancy", ""),
                }
                for label in [FULL_REFERENCE_LABEL, *V6_PRIMARY_VARIANTS, V5_EQUIVALENT_LABEL]
            },
            "7_to_9_shell_behavior": shell_summary_rows,
            "10_current_attribution": {
                "can_attribute_to_filtering_only": bool(
                    settings_d3_v5.get("all_checked_settings_equal") is True
                    and d3_vs_v5.get("exactly_equal") is True
                ),
                "note": "If streams/removal sets differ, attribution to filtering alone is not established.",
            },
        }
    }


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    questions = summary.get("questions", {})
    lines = [
        "# OriDyn V6 Matched Drop30 Audit",
        "",
        f"Generated: {now_iso()}",
        "",
        "This is a read-only audit report generated from existing metadata, streams, and merge QC outputs.",
        "",
    ]
    for key, value in questions.items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, sort_keys=True, default=json_default))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    args = normalize_args(args)
    ensure_audit_output_is_safe(args.out_dir, [args.v6_dir, args.v5_dir, args.source_stream])
    print(f"[audit] output directory: {args.out_dir}", flush=True)

    v6_validation = read_json(args.v6_dir / "validation.json") or {}
    v5_equivalent = find_v5_equivalent(args.v5_dir)
    targets = build_variant_targets(args.v6_dir, args.v5_dir, v5_equivalent)
    full_reference, full_reference_candidates = discover_full_reference(args.root, args.source_stream)
    merge_candidates, merge_discovery_rows = make_merge_candidates(targets, full_reference)

    print("[audit] reading selection manifests and QC summaries", flush=True)
    target_ids_by_source: dict[str, set[str]] = {"v6": set(), "v5": set()}
    for target in targets:
        target_ids_by_source.setdefault(target.source, set()).add(target.variant_id)
    selection_counts: dict[str, dict[str, Any]] = {}
    selection_sets: dict[str, set[str]] = {}
    for source, base in [("v6", args.v6_dir), ("v5", args.v5_dir)]:
        path = selection_manifest_path(base, source)
        ids = target_ids_by_source.get(source, set())
        if path is None or not ids:
            continue
        counts, sets = count_selection_rows(path, source, ids, load_sets=not args.skip_selection_sets)
        selection_counts.update(counts)
        selection_sets.update(sets)

    hkl_aggregates = aggregate_filtered_csv(
        args.v6_dir / "per_variant_per_hkl_qc.csv",
        set(V6_PRIMARY_VARIANTS),
        "variant_id",
        ["n_removed", "n_observations", "n_eligible", "n_high_eg", "n_blocks", "n_block_observations"],
    )
    block_aggregates = aggregate_filtered_csv(
        args.v6_dir / "per_variant_per_block_qc.csv",
        set(V6_PRIMARY_VARIANTS),
        "variant_id",
        ["n_removed", "n_retained_in_block"],
    )
    manifest_records = load_manifest_rows(args.v6_dir, args.v5_dir, targets)

    stream_counts: dict[str, dict[str, Any]] = {}
    if not args.skip_stream_scan:
        print("[audit] counting reflection rows in target streams", flush=True)
        stream_jobs = [(target.label, stream_path_for(target)) for target in targets]
        stream_jobs.append(("source_stream", args.source_stream))
        with ThreadPoolExecutor(max_workers=min(args.workers, len(stream_jobs))) as executor:
            futures = {executor.submit(count_stream_reflections, path): label for label, path in stream_jobs}
            for future in as_completed(futures):
                stream_counts[futures[future]] = future.result()

    removal_rows = make_removal_accounting(
        targets,
        manifest_records,
        selection_counts,
        hkl_aggregates,
        block_aggregates,
        stream_counts,
        v6_validation,
    )
    design_rows = make_design_audit(args.v6_dir, args.v5_dir, v5_equivalent, block_aggregates, hkl_aggregates)
    selection_comparison_rows = compare_sets(
        selection_sets,
        V6_PRIMARY_VARIANTS[1],
        str(v5_equivalent.get("variant_id", "")),
    )

    print("[audit] parsing merge QC outputs", flush=True)
    global_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for candidate in merge_candidates:
        if candidate.merge_dir is None:
            warnings.append({"label": candidate.label, "severity": "warning", "message": "no merge directory located"})
            continue
        metrics, shells, _settings, local_warnings = parse_merge_outputs(candidate.merge_dir, candidate.label, candidate.variant_id)
        global_rows.append(metrics)
        shell_rows.extend(shells)
        warnings.extend(local_warnings)

    merge_setting_rows = compare_merge_settings(global_rows)
    shell_deltas, d2d3_rows = make_shell_deltas(shell_rows)
    shell_summary_rows = summarize_shell_behavior(shell_deltas, d2d3_rows)
    fingerprint_rows = make_file_fingerprints(merge_candidates, args.source_stream)
    inventory_rows = metadata_inventory(args.v6_dir, args.v5_dir)

    summary = make_interpretation_summary(
        removal_rows,
        design_rows,
        selection_comparison_rows,
        merge_setting_rows,
        global_rows,
        shell_summary_rows,
        v5_equivalent,
    )
    summary.update(
        {
            "root": str(args.root),
            "v6_dir": str(args.v6_dir),
            "v5_dir": str(args.v5_dir),
            "source_stream": str(args.source_stream),
            "out_dir": str(args.out_dir),
            "created": now_iso(),
        }
    )

    print("[audit] writing audit artifacts", flush=True)
    write_csv(args.out_dir / "metadata_inventory.csv", inventory_rows)
    write_csv(args.out_dir / "merge_discovery.csv", merge_discovery_rows)
    write_csv(args.out_dir / "full_reference_candidates.csv", full_reference_candidates)
    write_csv(args.out_dir / "removal_accounting.csv", removal_rows)
    write_csv(args.out_dir / "design_audit.csv", design_rows)
    write_csv(args.out_dir / "selection_equivalence.csv", selection_comparison_rows)
    write_csv(args.out_dir / "merge_global_metrics.csv", global_rows)
    write_csv(args.out_dir / "merge_settings_comparison.csv", merge_setting_rows)
    write_csv(args.out_dir / "shell_metrics.csv", shell_rows)
    write_csv(args.out_dir / "shell_deltas_vs_reference.csv", shell_deltas)
    write_csv(args.out_dir / "d2_vs_d3_shell_comparison.csv", d2d3_rows)
    write_csv(args.out_dir / "shell_behavior_summary.csv", shell_summary_rows)
    write_csv(args.out_dir / "file_fingerprints.csv", fingerprint_rows)
    write_csv(args.out_dir / "warnings.csv", warnings)
    (args.out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    write_markdown_report(args.out_dir / "audit_report.md", summary)
    return summary


def main() -> int:
    summary = run_audit(parse_args())
    q1 = summary.get("questions", {}).get("1_v6_matched_drop30_removed", {})
    print(
        "[audit] complete: "
        f"d2_removed={q1.get('d2_removed_by_stream_qc', '')}; "
        f"d3_removed={q1.get('d3_removed_by_stream_qc', '')}; "
        f"summary={summary.get('out_dir', '')}/audit_summary.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
