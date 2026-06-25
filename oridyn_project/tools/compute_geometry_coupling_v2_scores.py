#!/usr/bin/env python3
"""Compute isolated v2 geometry-coupling exposure scores from reflection_scores.csv.

This v2 path is separate from the original geometry-trust score that produced
the successful MFM300-V(III) filtering result. It uses signed observations and
does not canonicalize HKLs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oridyn.coupling_exposure_v2 import (
    HKL_COLUMNS,
    KEY_COLUMNS,
    OUTPUT_SCORE_COLUMNS,
    REQUIRED_COLUMNS,
    CouplingV2Params,
    add_v2_normalized_columns,
    missing_required_columns,
    score_frame_coupling_v2,
)


DEFAULT_OUTPUT_CSV = "geometry_coupling_v2_scores.csv"
DEFAULT_SUMMARY_MD = "geometry_coupling_v2_score_summary.md"
DEFAULT_METADATA_JSON = "run_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", required=True, type=Path, help="Existing OriDyn reflection_scores.csv")
    parser.add_argument("--outdir", required=True, type=Path, help="Fresh output directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in --outdir")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--sg0", type=float, default=0.01, help="Excitation-error scale for exp(-(sg/sg0)^2)")
    parser.add_argument("--g0-invA", type=float, default=0.40, help="Reciprocal delta scale for coupling prior")
    parser.add_argument("--hkl-delta-g0", type=float, default=1.5, help="Fallback HKL delta scale if metric fit fails")
    parser.add_argument("--low-order-power", type=float, default=1.5, help="Power p in 1/(1+(q_delta/g0)^p)")
    parser.add_argument("--beta-zone", type=float, default=0.5)
    parser.add_argument("--beta-row", type=float, default=0.5)
    parser.add_argument("--beta-frame", type=float, default=0.25)
    parser.add_argument("--max-edges-per-reflection", type=int, default=64, help="Sum only the top N edge weights")
    parser.add_argument("--target-batch-size", type=int, default=256, help="Target observations scored per vectorized batch")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Frame-parallel workers. Use 0 for a conservative auto setting.",
    )
    parser.add_argument("--progress-every-frames", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test limit on score-table frames")
    args = parser.parse_args()

    if not args.scores_csv.exists():
        raise SystemExit(f"--scores-csv not found: {args.scores_csv}")
    if args.sg0 <= 0:
        raise SystemExit("--sg0 must be > 0")
    if args.g0_invA <= 0:
        raise SystemExit("--g0-invA must be > 0")
    if args.hkl_delta_g0 <= 0:
        raise SystemExit("--hkl-delta-g0 must be > 0")
    if args.low_order_power <= 0:
        raise SystemExit("--low-order-power must be > 0")
    if args.max_edges_per_reflection < 1:
        raise SystemExit("--max-edges-per-reflection must be >= 1")
    if args.target_batch_size < 1:
        raise SystemExit("--target-batch-size must be >= 1")
    if args.progress_every_frames < 1:
        raise SystemExit("--progress-every-frames must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be >= 1 when provided")
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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
    return text


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "csv": args.outdir / args.output_csv,
        "summary": args.outdir / DEFAULT_SUMMARY_MD,
        "metadata": args.outdir / DEFAULT_METADATA_JSON,
    }


def ensure_output(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    blocked = [path for path in paths.values() if path.exists()]
    if blocked and not args.overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing v2 output file(s):\n{formatted}\nUse --overwrite if intended.")
    args.outdir.mkdir(parents=True, exist_ok=True)


def read_scores(scores_csv: Path, max_frames: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(scores_csv, nrows=0).columns
    missing = missing_required_columns(header)
    if missing:
        raise SystemExit(
            "reflection_scores.csv is missing required v2 input column(s): "
            + ", ".join(missing)
            + "\nRequired columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    optional_columns = [column for column in ["assigned_risky_axis"] if column in header]
    usecols = list(dict.fromkeys([*REQUIRED_COLUMNS, *optional_columns]))
    stats: dict[str, Any] = {
        "input_rows_read": 0,
        "max_frames": max_frames,
        "optional_columns_present": optional_columns,
    }
    if max_frames is None:
        table = pd.read_csv(scores_csv, usecols=usecols)
        stats["input_rows_read"] = int(len(table))
    else:
        chunks = []
        frame_order: list[int] = []
        seen_frames: set[int] = set()
        for chunk in pd.read_csv(scores_csv, usecols=usecols, chunksize=250_000):
            stats["input_rows_read"] += int(len(chunk))
            chunks.append(chunk)
            for frame in pd.to_numeric(chunk["frame"], errors="coerce").dropna().astype(int).to_numpy():
                if frame not in seen_frames:
                    seen_frames.add(int(frame))
                    frame_order.append(int(frame))
                    if len(frame_order) >= int(max_frames):
                        break
            if len(frame_order) >= int(max_frames):
                break
        table = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
        keep_frames = set(frame_order[: int(max_frames)])
        table = table[pd.to_numeric(table["frame"], errors="coerce").isin(keep_frames)].copy()
        stats["frames_kept_for_smoke"] = sorted(keep_frames)

    table = clean_scores(table)
    stats["rows_after_cleanup"] = int(len(table))
    stats["frames_after_cleanup"] = int(table["frame"].nunique()) if "frame" in table else 0
    duplicate_mask = table.duplicated(KEY_COLUMNS, keep=False)
    stats["duplicate_key_rows"] = int(duplicate_mask.sum())
    stats["duplicate_keys"] = int(table.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_mask.any() else 0
    if duplicate_mask.any():
        log("Warning: duplicate exact observation keys in score table; keeping first row per key")
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    stats["unique_observation_keys"] = int(len(table))
    return table, stats


def clean_scores(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in ["frame", *HKL_COLUMNS]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    bad_key = out[["frame", *HKL_COLUMNS]].isna().any(axis=1)
    out = out.loc[~bad_key].copy()
    out["frame"] = out["frame"].astype("int64")
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    numeric_columns = [
        "q_invA",
        "sg",
        "same_laue_zone_crowding_norm",
        "systematic_row_risk_norm",
        "frame_axis_risk_norm",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def resolve_workers(requested: int, n_tasks: int) -> int:
    if n_tasks <= 1:
        return 1
    if requested == 0:
        cpu = os.cpu_count() or 1
        # Conservative cap: frame groups are copied to workers, so unlimited CPU
        # can waste RAM on large cSerialED tables.
        return max(1, min(int(cpu), int(n_tasks), 16))
    return max(1, min(int(requested), int(n_tasks)))


def compute_scores(table: pd.DataFrame, params: CouplingV2Params, workers: int, progress_every: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = [(int(frame), group.copy(), params) for frame, group in table.groupby("frame", sort=True)]
    n_tasks = len(tasks)
    used_workers = resolve_workers(workers, n_tasks)
    log(f"Scoring {n_tasks:,} frame(s) with {used_workers} worker(s)")
    scored_frames: list[pd.DataFrame] = []
    frame_stats: list[dict[str, Any]] = []

    completed = 0
    if used_workers == 1:
        for task in tasks:
            scored, stats = score_frame_coupling_v2(task)
            scored_frames.append(scored)
            frame_stats.append(stats)
            completed += 1
            if completed % int(progress_every) == 0 or completed == n_tasks:
                log(f"Scored frames: {completed:,}/{n_tasks:,}")
    else:
        with ProcessPoolExecutor(max_workers=used_workers) as executor:
            futures = [executor.submit(score_frame_coupling_v2, task) for task in tasks]
            for future in as_completed(futures):
                scored, stats = future.result()
                scored_frames.append(scored)
                frame_stats.append(stats)
                completed += 1
                if completed % int(progress_every) == 0 or completed == n_tasks:
                    log(f"Scored frames: {completed:,}/{n_tasks:,}")

    raw_scores = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    if raw_scores.empty:
        return raw_scores, {"workers_used": used_workers, "frame_stats": []}

    # Restore original stable order for exact-key filtering downstream.
    sort_cols = [column for column in ["frame", "source_filename", "event", "h", "k", "l"] if column in raw_scores]
    raw_scores = raw_scores.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    normalized, normalization = add_v2_normalized_columns(raw_scores, params)
    metadata = summarize_frame_stats(frame_stats)
    metadata["workers_used"] = int(used_workers)
    metadata["normalization"] = normalization
    return normalized, metadata


def summarize_frame_stats(frame_stats: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, int] = {}
    projected = 0
    total_observations = 0
    for stats in frame_stats:
        method = str(stats.get("reciprocal_metric_method", "unknown"))
        methods[method] = methods.get(method, 0) + 1
        projected += int(bool(stats.get("reciprocal_metric_projected_to_psd")))
        total_observations += int(stats.get("n_observations", 0))
    return {
        "n_frames_scored": int(len(frame_stats)),
        "frame_observations_scored": int(total_observations),
        "reciprocal_metric_methods": methods,
        "reciprocal_metric_projected_frames": int(projected),
    }


def write_summary(
    path: Path,
    args: argparse.Namespace,
    params: CouplingV2Params,
    score_table: pd.DataFrame,
    output: pd.DataFrame,
    load_stats: dict[str, Any],
    score_metadata: dict[str, Any],
    csv_path: Path,
) -> None:
    lines = [
        "# Geometry-Coupling V2 Score Summary",
        "",
        "## Scope",
        "",
        "- This is a separate v2 score path and does not modify the old successful geometry-trust pipeline.",
        "- Matching/filtering downstream should use exact `source_filename + event + signed h,k,l` keys.",
        "- HKLs are not symmetry-canonicalized.",
        "",
        "## Inputs",
        "",
        f"- Score table: `{args.scores_csv}`",
        f"- Output CSV: `{csv_path}`",
        "",
        "## Required Input Columns",
        "",
        "- " + ", ".join(f"`{column}`" for column in REQUIRED_COLUMNS),
        "",
        "## Equations Implemented",
        "",
        "- Source excitation: `E(h) = exp(-(sg(h) / sg0)^2)` using existing `sg`.",
        "- Reciprocal coupling prior: `1 / (1 + (q_delta / g0_invA)^p)` when a per-frame reciprocal metric can be fit.",
        "- HKL fallback prior: `1 / (1 + (|delta_hkl| / hkl_delta_g0)^p)` if the reciprocal metric fit is unavailable.",
        "- `v2_core`: top-edge sum of `E(h) * coupling_prior(h-g)`, then `log1p`.",
        "- `v2_core_plus_zone`: core with same assigned-Laue-zone edge boost `(1 + beta_zone)`.",
        "- `v2_core_plus_row`: core with target row boost `(1 + beta_row * systematic_row_risk_norm(g))`.",
        "- `v2_full`: same-zone edge boost plus target row boost; `trust_risk_v2_full_norm` also applies soft frame boost `(1 + beta_frame * frame_axis_risk_norm)` before p01-p99 normalization.",
        "- All norm columns use global robust p01-p99 normalization clipped to `[0, 1]`.",
        "",
        "## Parameters",
        "",
    ]
    for key, value in params.to_dict().items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Input rows read: {load_stats.get('input_rows_read', 0):,}",
            f"- Rows after cleanup: {load_stats.get('rows_after_cleanup', 0):,}",
            f"- Frames after cleanup: {load_stats.get('frames_after_cleanup', 0):,}",
            f"- Unique scored observation keys: {load_stats.get('unique_observation_keys', 0):,}",
            f"- Output rows: {len(output):,}",
            "",
            "## Reciprocal Metric Methods",
            "",
        ]
    )
    for method, count in score_metadata.get("reciprocal_metric_methods", {}).items():
        lines.append(f"- `{method}`: {count:,} frame(s)")
    lines.extend(
        [
            "",
            "## Score Distributions",
            "",
            _score_distribution_table(output),
            "",
            "## Output Files",
            "",
            f"- `{csv_path}`",
            f"- `{path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_distribution_table(output: pd.DataFrame) -> str:
    rows = []
    for column in OUTPUT_SCORE_COLUMNS:
        if column not in output:
            continue
        values = pd.to_numeric(output[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "column": column,
                "n_finite": int(len(values)),
                "min": float(values.min()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "p95": float(values.quantile(0.95)) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return "_No score columns were generated._"
    columns = ["column", "n_finite", "min", "median", "p95", "max"]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in df.loc[:, columns].itertuples(index=False):
        values = []
        for key, value in zip(columns, row, strict=True):
            if key in {"column", "n_finite"}:
                values.append(str(value))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(f"{float(value):.6g}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    params: CouplingV2Params,
    load_stats: dict[str, Any],
    score_metadata: dict[str, Any],
    outputs: dict[str, Path],
) -> None:
    payload = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"scores_csv": str(args.scores_csv)},
        "outputs": {name: str(path_value) for name, path_value in outputs.items()},
        "parameters": params.to_dict(),
        "load_stats": load_stats,
        "score_metadata": score_metadata,
        "warnings": [
            "This v2 score path is separate from the original successful geometry-trust pipeline.",
            "The score uses observed same-frame reflections as candidate beams because the existing reflection_scores.csv does not contain unobserved candidate-node lists.",
            "Reciprocal-space delta is estimated from per-frame h,k,l,q_invA; sparse or rank-deficient frames fall back to HKL delta length and are counted in metadata.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = output_paths(args)
    ensure_output(args, paths)

    params = CouplingV2Params(
        sg0=float(args.sg0),
        g0_invA=float(args.g0_invA),
        hkl_delta_g0=float(args.hkl_delta_g0),
        low_order_power=float(args.low_order_power),
        beta_zone=float(args.beta_zone),
        beta_row=float(args.beta_row),
        beta_frame=float(args.beta_frame),
        max_edges_per_reflection=int(args.max_edges_per_reflection),
        target_batch_size=int(args.target_batch_size),
    )

    log(f"Output directory: {args.outdir}")
    log("Loading score table")
    score_table, load_stats = read_scores(args.scores_csv, args.max_frames)
    log(
        "Loaded scores: "
        f"rows={len(score_table):,}, frames={load_stats['frames_after_cleanup']:,}, "
        f"unique_keys={load_stats['unique_observation_keys']:,}"
    )
    log("Computing v2 many-beam coupling exposure scores")
    scored, score_metadata = compute_scores(score_table, params, int(args.workers), int(args.progress_every_frames))

    output_columns = [column for column in ["source_filename", "event", "frame", "h", "k", "l"] if column in scored]
    output_columns.extend(OUTPUT_SCORE_COLUMNS)
    scored[output_columns].to_csv(paths["csv"], index=False)
    write_summary(paths["summary"], args, params, score_table, scored, load_stats, score_metadata, paths["csv"])
    write_metadata(paths["metadata"], args, params, load_stats, score_metadata, paths)

    log("V2 score computation complete")
    print(f"Wrote: {paths['csv']}")
    print(f"Wrote: {paths['summary']}")
    print(f"Wrote: {paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
