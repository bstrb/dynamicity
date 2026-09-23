#!/usr/bin/env python3
"""Build low-drop V6 filter_all EgM2 streams with the original Target-A rule.

This is a narrow companion to build_v6_full_population_sweep.py.  It creates
only filter_all_eg_m2 low-drop variants, using the same full-population cache,
same exact signed-HKL grouping, same EgM2 ordering, and same Target-A per-HKL
removal logic as the original V6 filter_all_eg_m2_drop05/drop10/drop20 runs.

It does not run Partialator or merging.  Stream files are written only with
--write-streams.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
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
    "oridyn_v6_filter_all_egm2_lowdrops_20260813"
)
DEFAULT_FRACTIONS = ("0.005", "0.010", "0.020")

TARGET_A_MIN_OBSERVATIONS = 10
MIN_REMAINING = 2
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
        if fraction > 0.05:
            raise SystemExit(
                f"{item!r} is outside this low-drop exact-match helper. "
                "Use the existing V6 sweep for drop10/drop20; this script is for <=5%."
            )
        out.append((item, fraction))
    if not out:
        raise SystemExit("--fractions must contain at least one fraction")
    labels = [target_a_fraction_label(fraction) for _text, fraction in out]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate Target-A fraction label(s): {duplicates}")
    return out


def target_a_fraction_label(fraction: float) -> str:
    percent = float(fraction) * 100.0
    if abs(percent - round(percent)) < 1.0e-9 and percent >= 1.0:
        return f"{int(round(percent)):02d}"
    text = f"{percent:.6g}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_variants(fractions: list[tuple[str, float]], out_dir: Path) -> list[base.VariantSpec]:
    variants: list[base.VariantSpec] = []
    for text, fraction in fractions:
        label = target_a_fraction_label(float(fraction))
        variant_id = f"filter_all_eg_m2_drop{label}"
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


def planned_output_paths(out_dir: Path, variants: list[base.VariantSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "filter_all_egm2_lowdrop_manifest.tsv",
        out_dir / "filter_all_egm2_lowdrop_counts.csv",
        out_dir / "filter_all_egm2_lowdrop_parameters.json",
        out_dir / "filter_all_egm2_lowdrop_metadata.json",
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


def make_count_rows(
    variants: list[base.VariantSpec],
    counts: dict[str, dict[str, int]],
    masks: dict[str, base.PackedMask],
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
                "experiment_type": "targeted_filter",
                "filtering_target": "all",
                "drop_fraction": float(variant.drop_fraction),
                "designation": f"drop{target_a_fraction_label(float(variant.drop_fraction))}",
                "accepted_population_count": int(accepted_count),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "eligible_observation_count": int(count["candidate_observation_count"]),
                "actionable_observation_count": int(count["actionable_observation_count"]),
                "accepted_observations_removed": int(removed),
                "accepted_observations_retained": int(retained_accepted),
                "out_of_analysis_source_rows_retained": "" if source_rows is None else int(source_rows - accepted_count),
                "total_source_rows_retained": "" if source_rows is None else int(source_rows - removed),
                "removed_fraction_of_accepted_population": float(removed / max(1, accepted_count)),
                "removed_fraction_of_source_rows": "" if source_rows is None else float(removed / max(1, source_rows)),
                "removed_fraction_of_eligible_population": float(removed / max(1, count["candidate_observation_count"])),
                "removed_fraction_of_actionable_population": float(removed / count["actionable_observation_count"]) if count["actionable_observation_count"] else "",
                "target_a_min_observations": TARGET_A_MIN_OBSERVATIONS,
                "min_remaining": MIN_REMAINING,
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
                "score_or_random_control_id": "eg_m2",
                "experiment_type": "targeted_filter",
                "target": "all",
                "fraction": float(variant.drop_fraction),
                "designation": f"drop{target_a_fraction_label(float(variant.drop_fraction))}",
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
        rev = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        status = subprocess.run(["git", "-C", str(project_root), "status", "--short"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
    counts: dict[str, dict[str, int]],
    masks: dict[str, base.PackedMask],
    stream_qc: list[dict[str, Any]],
    started_utc: str,
    logger: base.RunLogger,
) -> None:
    logger.log("writing V6 filter_all EgM2 low-drop manifests and metadata")
    count_rows = make_count_rows(variants, counts, masks, accepted_count, source_rows)
    manifest_rows = make_manifest_rows(variants, count_rows, stream_qc, bool(args.write_streams))
    write_csv(args.output_dir / "filter_all_egm2_lowdrop_counts.csv", count_rows)
    write_csv(args.output_dir / "filter_all_egm2_lowdrop_manifest.tsv", manifest_rows, delimiter="\t")
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
        "compatibility_target": "Original V6 Target-A filter_all_eg_m2_drop05/drop10/drop20 rule",
        "filtering_rule": {
            "target": "all",
            "grouping": "exact signed h,k,l",
            "rank": "Eg*M2 descending, exact_key_text ascending for ties",
            "eligible_hkl": "n_obs >= 10",
            "n_remove": "floor(drop_fraction * n_obs), capped to leave at least 2 observations",
            "min_remaining": MIN_REMAINING,
            "target_a_min_observations": TARGET_A_MIN_OBSERVATIONS,
            "symmetry_canonicalization": False,
            "non_selected_observations": "retained unchanged",
            "non_scoreable_source_rows": "retained unchanged",
        },
    }
    write_json(args.output_dir / "filter_all_egm2_lowdrop_parameters.json", parameters)
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
            "manifest": str(args.output_dir / "filter_all_egm2_lowdrop_manifest.tsv"),
            "counts": str(args.output_dir / "filter_all_egm2_lowdrop_counts.csv"),
            "parameters": str(args.output_dir / "filter_all_egm2_lowdrop_parameters.json"),
            "metadata": str(args.output_dir / "filter_all_egm2_lowdrop_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "filter_all_egm2_lowdrop_metadata.json", metadata)


def main() -> int:
    args = parse_args()
    variants = build_variants(args.fraction_items, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(planned_output_paths(args.output_dir, variants, bool(args.write_streams)))

    logger = base.RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        logger.log("V6 filter_all EgM2 low-drop builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"fractions={','.join(text for text, _fraction in args.fraction_items)}")
        logger.log(f"workers={int(args.workers)}")

        db_file = base.cache_db_path(args.source_out_dir)
        if not db_file.is_file():
            raise SystemExit(f"full-population cache not found: {db_file}")
        base.require_cache_schema(db_file)
        accepted_count = base.score_cache_count(db_file)
        source_rows = base.source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and source_rows < accepted_count:
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than accepted cache count {accepted_count:,}")

        masks, counts, selection_stats = base.construct_masks(db_file, variants, accepted_count, int(args.workers), logger)
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            stream_qc = base.rewrite_streams(args.source_stream, db_file, variants, masks, logger, source_rows)
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(args, variants, db_file, accepted_count, source_rows, selection_stats, counts, masks, stream_qc, started_utc, logger)
        logger.log("V6 filter_all EgM2 low-drop builder complete")
        return 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
