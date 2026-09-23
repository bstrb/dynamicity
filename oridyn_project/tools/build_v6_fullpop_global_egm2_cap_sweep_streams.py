#!/usr/bin/env python3
"""Build V6 full-population global EgM2 cap-sweep filter streams.

This is a V6-only stream builder for testing whether a global high-risk EgM2
tail, protected by a per-signed-HKL safety cap, gives the small 0.5% benefit
while still producing genuinely different 10% and 20% removals.

It reads the existing full_population_cache.sqlite and source stream only.  It
does not run Partialator, merging, indexing, or stream generation unless
--write-streams is supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

import build_v6_fullpop_gentle_egm2_streams as base


DEFAULT_SOURCE_OUT_DIR = base.DEFAULT_SOURCE_OUT_DIR
DEFAULT_SOURCE_STREAM = base.DEFAULT_SOURCE_STREAM
DEFAULT_OUTPUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_fullpop_global_egm2_cap0p8_20260813"
)
DEFAULT_FRACTIONS = ("0.005", "0.010", "0.020", "0.050", "0.100", "0.200")
DEFAULT_MAX_REMOVE_PER_HKL_FRACTION = 0.80
DEFAULT_MIN_RETAINED_PER_HKL = 2
SQL_CHUNK_ROWS = 250_000
CSV_FLOAT_FORMAT = "%.12g"


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


def cap_label(value: float) -> str:
    return f"{float(value):.6g}".rstrip("0").rstrip(".").replace(".", "p")


def drop_label(fraction: float) -> str:
    percent = float(fraction) * 100.0
    if abs(percent - round(percent)) < 1.0e-9 and percent >= 1.0:
        return f"{int(round(percent)):02d}"
    text = f"{percent:.6g}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def parse_fraction_items(value: str | None) -> list[tuple[str, float]]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_FRACTIONS)
    out: list[tuple[str, float]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        fraction = float(item)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise SystemExit(f"Invalid fraction {item!r}; expected 0 < fraction < 1")
        out.append((item, fraction))
    if not out:
        raise SystemExit("--fractions must contain at least one fraction")
    labels = [drop_label(fraction) for _text, fraction in out]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate drop label(s): {duplicates}")
    return sorted(out, key=lambda item: item[1])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fractions", default=",".join(DEFAULT_FRACTIONS))
    parser.add_argument("--max-remove-per-hkl-fraction", type=float, default=DEFAULT_MAX_REMOVE_PER_HKL_FRACTION)
    parser.add_argument("--min-retained-per-hkl", type=int, default=DEFAULT_MIN_RETAINED_PER_HKL)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        return args

    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.fraction_items = parse_fraction_items(args.fractions)
    args.workers = max(1, int(args.workers))
    args.min_retained_per_hkl = max(0, int(args.min_retained_per_hkl))
    args.max_remove_per_hkl_fraction = float(args.max_remove_per_hkl_fraction)
    if not math.isfinite(args.max_remove_per_hkl_fraction) or not (0.0 < args.max_remove_per_hkl_fraction <= 1.0):
        raise SystemExit("--max-remove-per-hkl-fraction must satisfy 0 < value <= 1")
    if args.dry_run and args.write_streams:
        raise SystemExit("Use either --dry-run or --write-streams, not both")
    if not args.write_streams:
        args.dry_run = True
    if not args.source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {args.source_out_dir}")
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    return args


def build_variants(
    fractions: list[tuple[str, float]],
    out_dir: Path,
    max_remove_per_hkl_fraction: float,
) -> list[base.VariantSpec]:
    variants: list[base.VariantSpec] = []
    cap = cap_label(max_remove_per_hkl_fraction)
    for text, fraction in fractions:
        label = drop_label(float(fraction))
        variant_id = f"filter_all_global_eg_m2_cap{cap}_drop{label}"
        output_filename = f"{variant_id}.stream"
        variants.append(
            base.VariantSpec(
                variant_id=variant_id,
                fraction_text=text,
                drop_fraction=float(fraction),
                output_filename=output_filename,
                output_stream=out_dir / output_filename,
            )
        )
    return variants


def planned_output_paths(out_dir: Path, variants: list[base.VariantSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "global_egm2_cap_sweep_manifest.tsv",
        out_dir / "global_egm2_cap_sweep_counts.csv",
        out_dir / "global_egm2_cap_sweep_parameters.json",
        out_dir / "global_egm2_cap_sweep_metadata.json",
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


def max_removable_for_hkl(n_obs: int, max_remove_fraction: float, min_retained: int) -> int:
    n_obs = int(n_obs)
    if n_obs <= 0:
        return 0
    by_fraction = int(math.floor(float(max_remove_fraction) * n_obs))
    by_retained = max(0, n_obs - int(min_retained))
    return int(max(0, min(by_fraction, by_retained)))


def read_hkl_counts(db_file: Path, logger: base.RunLogger) -> dict[tuple[int, int, int], int]:
    total_hkls = base.count_signed_hkls(db_file)
    progress = base.StageProgress(logger, "reading signed-HKL redundancy", total_hkls, "HKLs", workers=1)
    counts: dict[tuple[int, int, int], int] = {}
    with base.connect_readonly(db_file) as conn:
        cursor = conn.execute("SELECT h,k,l,COUNT(*) FROM score_cache GROUP BY h,k,l ORDER BY h,k,l")
        for h, k, l, n_obs in cursor:
            counts[(int(h), int(k), int(l))] = int(n_obs)
            if len(counts) % 10_000 == 0:
                progress.update(len(counts))
    progress.finish(len(counts))
    return counts


def load_rank_arrays(
    db_file: Path,
    accepted_count: int,
    logger: base.RunLogger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    schema = set(base.cache_schema(db_file))
    tie_column = "source_order" if "source_order" in schema else "ordinal"
    query = f"SELECT ordinal,h,k,l,{tie_column},Eg,M2 FROM score_cache"
    ordinal_chunks: list[np.ndarray] = []
    h_chunks: list[np.ndarray] = []
    k_chunks: list[np.ndarray] = []
    l_chunks: list[np.ndarray] = []
    tie_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    rows_read = 0
    progress = base.StageProgress(logger, "reading EgM2 scores", accepted_count, "observations", workers=1)
    with base.connect_readonly(db_file) as conn:
        for chunk in pd.read_sql_query(query, conn, chunksize=SQL_CHUNK_ROWS):
            eg = pd.to_numeric(chunk["Eg"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            m2 = pd.to_numeric(chunk["M2"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            score = eg * m2
            if not np.isfinite(score).all():
                bad = int((~np.isfinite(score)).sum())
                raise SystemExit(f"Nonfinite EgM2 score encountered in full-population cache ({bad:,} rows in chunk)")
            ordinal_chunks.append(chunk["ordinal"].to_numpy(dtype=np.int64, copy=True))
            h_chunks.append(chunk["h"].to_numpy(dtype=np.int32, copy=True))
            k_chunks.append(chunk["k"].to_numpy(dtype=np.int32, copy=True))
            l_chunks.append(chunk["l"].to_numpy(dtype=np.int32, copy=True))
            tie_chunks.append(chunk[tie_column].to_numpy(dtype=np.int64, copy=True))
            score_chunks.append(score.astype(np.float64, copy=True))
            rows_read += int(len(chunk))
            progress.update(rows_read)
    progress.finish(rows_read)
    if rows_read != int(accepted_count):
        raise SystemExit(f"Read {rows_read:,} cache rows, expected {accepted_count:,}")
    logger.log("sorting observations by global EgM2 descending")
    ordinals = np.concatenate(ordinal_chunks)
    h_values = np.concatenate(h_chunks)
    k_values = np.concatenate(k_chunks)
    l_values = np.concatenate(l_chunks)
    tie_values = np.concatenate(tie_chunks)
    scores = np.concatenate(score_chunks)
    order = np.lexsort((tie_values, -scores))
    return ordinals[order], h_values[order], k_values[order], l_values[order], scores[order]


def construct_global_masks(
    db_file: Path,
    variants: list[base.VariantSpec],
    accepted_count: int,
    max_remove_per_hkl_fraction: float,
    min_retained_per_hkl: int,
    logger: base.RunLogger,
) -> tuple[dict[str, base.PackedMask], dict[str, dict[str, Any]], dict[str, Any]]:
    hkl_counts = read_hkl_counts(db_file, logger)
    hkl_capacity = {
        hkl: max_removable_for_hkl(n_obs, max_remove_per_hkl_fraction, min_retained_per_hkl)
        for hkl, n_obs in hkl_counts.items()
    }
    total_capacity = int(sum(hkl_capacity.values()))
    requested_counts = {
        variant.variant_id: int(math.floor(float(variant.drop_fraction) * int(accepted_count)))
        for variant in variants
    }
    max_requested = max(requested_counts.values()) if requested_counts else 0
    if total_capacity < max_requested:
        logger.log(
            "warning: requested maximum removal exceeds cap-limited capacity "
            f"({max_requested:,} requested; {total_capacity:,} possible)"
        )

    ordinals, h_values, k_values, l_values, scores = load_rank_arrays(db_file, accepted_count, logger)
    selected_ordinals: list[int] = []
    selected_hkls: list[tuple[int, int, int]] = []
    selected_scores: list[float] = []
    removed_by_hkl: dict[tuple[int, int, int], int] = {}

    progress = base.StageProgress(logger, "selecting global EgM2 tail with per-HKL cap", len(ordinals), "ranked observations", workers=1)
    scanned = 0
    for ordinal, h, k, l, score in zip(ordinals, h_values, k_values, l_values, scores):
        scanned += 1
        hkl = (int(h), int(k), int(l))
        used = removed_by_hkl.get(hkl, 0)
        cap = hkl_capacity.get(hkl, 0)
        if used < cap:
            selected_ordinals.append(int(ordinal))
            selected_hkls.append(hkl)
            selected_scores.append(float(score))
            removed_by_hkl[hkl] = used + 1
            if len(selected_ordinals) >= max_requested:
                progress.update(scanned, force=True)
                break
        progress.update(scanned)
    progress.finish(scanned)

    selected_ordinal_array = np.asarray(selected_ordinals, dtype=np.int64)
    masks = {variant.variant_id: base.PackedMask(accepted_count) for variant in variants}
    counts: dict[str, dict[str, Any]] = {}
    for variant in variants:
        requested = int(requested_counts[variant.variant_id])
        actual = int(min(requested, len(selected_ordinals)))
        masks[variant.variant_id].set_many(selected_ordinal_array[:actual])
        prefix_hkls = selected_hkls[:actual]
        per_hkl_removed = Counter(prefix_hkls)
        cap_hits = sum(1 for hkl, n_removed in per_hkl_removed.items() if n_removed >= hkl_capacity.get(hkl, 0) > 0)
        if any(n_removed > hkl_capacity.get(hkl, 0) for hkl, n_removed in per_hkl_removed.items()):
            raise SystemExit(f"{variant.variant_id}: per-HKL cap validation failed")
        score_cutoff = selected_scores[actual - 1] if actual else ""
        counts[variant.variant_id] = {
            "signed_hkl_groups": int(len(hkl_counts)),
            "candidate_observation_count": int(accepted_count),
            "requested_removed_count": int(requested),
            "actual_removed_count": int(actual),
            "fill_fraction_of_requested": float(actual / requested) if requested else 1.0,
            "cap_limited_total_capacity": int(total_capacity),
            "selected_hkl_count": int(len(per_hkl_removed)),
            "hkl_groups_at_cap": int(cap_hits),
            "score_cutoff_eg_m2": score_cutoff,
        }

    for variant in variants:
        mask_count = masks[variant.variant_id].count()
        expected = counts[variant.variant_id]["actual_removed_count"]
        if int(mask_count) != int(expected):
            raise SystemExit(f"{variant.variant_id}: mask count {mask_count} != selected count {expected}")

    selection_stats = {
        "signed_hkl_count": int(len(hkl_counts)),
        "accepted_population_count": int(accepted_count),
        "max_remove_per_hkl_fraction": float(max_remove_per_hkl_fraction),
        "min_retained_per_hkl": int(min_retained_per_hkl),
        "cap_limited_total_capacity": int(total_capacity),
        "cap_limited_capacity_fraction_of_accepted": float(total_capacity / max(1, accepted_count)),
        "max_requested_removed_count": int(max_requested),
        "max_selected_removed_count": int(len(selected_ordinals)),
        "ranked_observations_scanned": int(scanned),
    }
    return masks, counts, selection_stats


def make_count_rows(
    variants: list[base.VariantSpec],
    counts: dict[str, dict[str, Any]],
    masks: dict[str, base.PackedMask],
    accepted_count: int,
    source_rows: int | None,
    max_remove_per_hkl_fraction: float,
    min_retained_per_hkl: int,
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
                "ranking_scope": "global",
                "drop_fraction_requested": float(variant.drop_fraction),
                "drop_designation": f"drop{drop_label(float(variant.drop_fraction))}",
                "max_remove_per_hkl_fraction": float(max_remove_per_hkl_fraction),
                "min_retained_per_hkl": int(min_retained_per_hkl),
                "accepted_population_count": int(accepted_count),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "candidate_observation_count": int(count["candidate_observation_count"]),
                "cap_limited_total_capacity": int(count["cap_limited_total_capacity"]),
                "requested_removed_count": int(count["requested_removed_count"]),
                "accepted_observations_removed": int(removed),
                "accepted_observations_retained": int(retained_accepted),
                "fill_fraction_of_requested": count["fill_fraction_of_requested"],
                "selected_hkl_count": int(count["selected_hkl_count"]),
                "hkl_groups_at_cap": int(count["hkl_groups_at_cap"]),
                "score_cutoff_eg_m2": count["score_cutoff_eg_m2"],
                "out_of_analysis_source_rows_retained": "" if source_rows is None else int(source_rows - accepted_count),
                "total_source_rows_retained": "" if source_rows is None else int(source_rows - removed),
                "removed_fraction_of_accepted_population": float(removed / max(1, accepted_count)),
                "removed_fraction_of_source_rows": "" if source_rows is None else float(removed / max(1, source_rows)),
                "mask_sha256": masks[variant.variant_id].digest(),
            }
        )
    return rows


def make_manifest_rows(
    variants: list[base.VariantSpec],
    count_rows: list[dict[str, Any]],
    stream_qc: list[dict[str, Any]],
    write_streams: bool,
) -> list[dict[str, Any]]:
    counts = {row["variant_id"]: row for row in count_rows}
    qc = base.stream_qc_by_variant(stream_qc)
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
                "experiment_type": "fullpop_global_tail_filter",
                "target": "all",
                "ranking_scope": "global",
                "fraction": float(variant.drop_fraction),
                "designation": f"drop{drop_label(float(variant.drop_fraction))}",
                "max_remove_per_hkl_fraction": count["max_remove_per_hkl_fraction"],
                "min_retained_per_hkl": count["min_retained_per_hkl"],
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
    variants: list[base.VariantSpec],
    db_file: Path,
    accepted_count: int,
    source_rows: int | None,
    selection_stats: dict[str, Any],
    counts: dict[str, dict[str, Any]],
    masks: dict[str, base.PackedMask],
    stream_qc: list[dict[str, Any]],
    started_utc: str,
    logger: base.RunLogger,
) -> None:
    logger.log("writing global EgM2 cap-sweep manifests and metadata")
    count_rows = make_count_rows(
        variants,
        counts,
        masks,
        accepted_count,
        source_rows,
        float(args.max_remove_per_hkl_fraction),
        int(args.min_retained_per_hkl),
    )
    manifest_rows = make_manifest_rows(variants, count_rows, stream_qc, bool(args.write_streams))
    write_csv(args.output_dir / "global_egm2_cap_sweep_counts.csv", count_rows)
    write_csv(args.output_dir / "global_egm2_cap_sweep_manifest.tsv", manifest_rows, delimiter="\t")
    parameters = {
        "source_out_dir": str(args.source_out_dir),
        "source_stream": str(args.source_stream),
        "source_cache": str(db_file),
        "output_dir": str(args.output_dir),
        "fractions": [text for text, _fraction in args.fraction_items],
        "workers_requested": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "score_definition": {"score_id": "eg_m2", "formula": "Eg * M2"},
        "candidate_domain": "V6 full-population all-domain scoreable accepted observations from full_population_cache.sqlite",
        "filtering_rule": {
            "target": "all",
            "ranking_scope": "global over all scoreable accepted observations",
            "rank": "Eg*M2 descending; source_order ascending for ties when available, otherwise ordinal ascending",
            "requested_n_remove": "floor(drop_fraction * accepted_population_count)",
            "per_hkl_safety": {
                "grouping": "exact signed h,k,l",
                "max_remove_per_hkl_fraction": float(args.max_remove_per_hkl_fraction),
                "min_retained_per_hkl": int(args.min_retained_per_hkl),
                "max_remove_for_hkl": "min(floor(max_remove_per_hkl_fraction * n_obs), n_obs - min_retained_per_hkl)",
            },
            "symmetry_canonicalization": False,
            "non_selected_observations": "retained unchanged",
            "non_scoreable_source_rows": "retained unchanged",
            "source_order": "preserved by stream rewrite",
        },
    }
    write_json(args.output_dir / "global_egm2_cap_sweep_parameters.json", parameters)
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
            "manifest": str(args.output_dir / "global_egm2_cap_sweep_manifest.tsv"),
            "counts": str(args.output_dir / "global_egm2_cap_sweep_counts.csv"),
            "parameters": str(args.output_dir / "global_egm2_cap_sweep_parameters.json"),
            "metadata": str(args.output_dir / "global_egm2_cap_sweep_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "global_egm2_cap_sweep_metadata.json", metadata)


def run_self_test() -> int:
    project_root = Path(__file__).resolve().parents[1]
    scratch_root = project_root / ".codex_smoke" / "global_egm2_cap_sweep"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True)
    db_file = scratch_root / "full_population_cache.sqlite"
    try:
        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            CREATE TABLE score_cache(
                ordinal INTEGER PRIMARY KEY,
                h INTEGER NOT NULL,
                k INTEGER NOT NULL,
                l INTEGER NOT NULL,
                exact_key_text TEXT NOT NULL,
                source_order INTEGER NOT NULL,
                Eg REAL NOT NULL,
                M2 REAL NOT NULL
            )
            """
        )
        rows: list[tuple[int, int, int, int, str, int, float, float]] = []
        ordinal = 0
        for i in range(10):
            rows.append((ordinal, 1, 0, 0, f"img|1|1|0|0|{i}", ordinal, float(100 - i), 1.0))
            ordinal += 1
        for i in range(10):
            rows.append((ordinal, 2, 0, 0, f"img|1|2|0|0|{i}", ordinal, float(90 - i), 1.0))
            ordinal += 1
        for i in range(4):
            rows.append((ordinal, 3, 0, 0, f"img|1|3|0|0|{i}", ordinal, float(200 - i), 1.0))
            ordinal += 1
        conn.executemany("INSERT INTO score_cache VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()

        out_dir = scratch_root / "out"
        variants = build_variants([("0.50", 0.50), ("0.90", 0.90)], out_dir, 0.80)
        logger = base.RunLogger(None)
        masks, counts, stats = construct_global_masks(
            db_file,
            variants,
            accepted_count=24,
            max_remove_per_hkl_fraction=0.80,
            min_retained_per_hkl=2,
            logger=logger,
        )
        first = counts[variants[0].variant_id]
        second = counts[variants[1].variant_id]
        assert first["requested_removed_count"] == 12
        assert first["actual_removed_count"] == 12
        assert second["requested_removed_count"] == 21
        assert second["actual_removed_count"] == 18
        assert stats["cap_limited_total_capacity"] == 18
        assert masks[variants[0].variant_id].count() == 12
        assert masks[variants[1].variant_id].count() == 18
        print("self-test passed: global ranking, cap limiting, and mask counts are consistent")
        return 0
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    variants = build_variants(args.fraction_items, args.output_dir, float(args.max_remove_per_hkl_fraction))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(planned_output_paths(args.output_dir, variants, bool(args.write_streams)))

    logger = base.RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        logger.log("V6 full-population global EgM2 cap-sweep builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"fractions={','.join(text for text, _fraction in args.fraction_items)}")
        logger.log(f"max_remove_per_hkl_fraction={float(args.max_remove_per_hkl_fraction):.6g}")
        logger.log(f"min_retained_per_hkl={int(args.min_retained_per_hkl)}")
        logger.log(f"workers_requested={int(args.workers)}")

        db_file = base.cache_db_path(args.source_out_dir)
        if not db_file.is_file():
            raise SystemExit(f"full-population cache not found: {db_file}")
        base.require_cache_schema(db_file)
        accepted_count = base.score_cache_count(db_file)
        source_rows = base.source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and source_rows < accepted_count:
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than accepted cache count {accepted_count:,}")

        masks, counts, selection_stats = construct_global_masks(
            db_file,
            variants,
            accepted_count,
            float(args.max_remove_per_hkl_fraction),
            int(args.min_retained_per_hkl),
            logger,
        )
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            stream_qc = base.rewrite_streams(args.source_stream, db_file, variants, masks, logger, source_rows)
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(args, variants, db_file, accepted_count, source_rows, selection_stats, counts, masks, stream_qc, started_utc, logger)
        logger.log("V6 full-population global EgM2 cap-sweep builder complete")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
