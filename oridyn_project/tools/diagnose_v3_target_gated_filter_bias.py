#!/usr/bin/env python3
"""Diagnose excitation bias in direct low-v3 per-HKL filtering."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
DEFAULT_DIRECT_SCORE_COLUMN = "trust_risk_v3_target_gated_shell_norm"
DEFAULT_EG_COLUMN = "target_excitation_Eg"
DEFAULT_GATE_COLUMN = "target_excitation_gate_Gg"
DEFAULT_SG_COLUMN = "sg_target"
DEFAULT_V2_COLUMN = "trust_risk_v2_full_norm"
TRUST_SCORE_CANDIDATES = [
    "badness_v3_excited_lowcoupling_shell_norm",
    "badness_v3_excited_lowcoupling_norm",
]
KEEP_FRACTIONS = [0.80, 0.60, 0.40, 0.20]
OBS_THRESHOLDS = [1, 2, 3, 5, 10]
EG_THRESHOLDS = [0.7, 0.8, 0.9]
DEFAULT_CHUNKSIZE = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-scores", required=True, type=Path, help="V3 score CSV, optionally enriched with trust/badness columns")
    parser.add_argument("--outdir", type=Path, default=None, help="Output diagnostics directory. Defaults to <scores_dir>/diagnostics")
    parser.add_argument("--direct-score-column", default=DEFAULT_DIRECT_SCORE_COLUMN)
    parser.add_argument("--trust-score-column", default=None, help="Optional badness column for new trust selection comparison")
    parser.add_argument("--eg-column", default=DEFAULT_EG_COLUMN)
    parser.add_argument("--gate-column", default=DEFAULT_GATE_COLUMN)
    parser.add_argument("--sg-column", default=DEFAULT_SG_COLUMN)
    parser.add_argument("--v2-risk-column", default=DEFAULT_V2_COLUMN)
    parser.add_argument("--keep-fractions", nargs="+", type=float, default=KEEP_FRACTIONS)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.v3_scores.exists():
        raise SystemExit(f"--v3-scores not found: {args.v3_scores}")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1 when provided")
    fractions = []
    for value in args.keep_fractions:
        fraction = float(value)
        if not np.isfinite(fraction) or not (0.0 < fraction <= 1.0):
            raise SystemExit("--keep-fractions values must satisfy 0 < fraction <= 1")
        fractions.append(fraction)
    args.keep_fractions = sorted(set(fractions), reverse=True)
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def detect_partiality_columns(header: Iterable[str]) -> list[str]:
    out = []
    for column in header:
        lower = str(column).lower()
        if "partial" in lower or lower in {"p", "pr", "partiality_obs"}:
            out.append(str(column))
    return out


def choose_trust_score_column(header: list[str], requested: str | None) -> str | None:
    if requested:
        if requested not in header:
            raise SystemExit(f"--trust-score-column {requested!r} is not present in --v3-scores")
        return requested
    for column in TRUST_SCORE_CANDIDATES:
        if column in header:
            return column
    return None


def require_columns(header: list[str], columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {', '.join(missing)}")


def output_paths(outdir: Path) -> tuple[Path, Path, Path]:
    csv_path = outdir / "v3_direct_low_filter_excitation_bias.csv"
    md_path = outdir / "v3_direct_low_filter_excitation_bias.md"
    json_path = outdir / "v3_direct_low_filter_excitation_bias_metadata.json"
    return csv_path, md_path, json_path


def ensure_outputs(outdir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    csv_path, md_path, json_path = output_paths(outdir)
    blocked = [path for path in (csv_path, md_path, json_path) if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing diagnostic file(s):\n{formatted}\nUse --overwrite if intended.")
    outdir.mkdir(parents=True, exist_ok=True)
    return csv_path, md_path, json_path


def iter_chunks(path: Path, usecols: list[str], chunksize: int, max_rows: int | None) -> Iterable[pd.DataFrame]:
    rows_seen = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        if max_rows is not None:
            remaining = int(max_rows) - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        rows_seen += int(len(chunk))
        yield chunk
        if max_rows is not None and rows_seen >= int(max_rows):
            break


def load_table(args: argparse.Namespace, usecols: list[str]) -> pd.DataFrame:
    chunks = []
    for idx, chunk in enumerate(iter_chunks(args.v3_scores, usecols, args.chunksize, args.max_rows), start=1):
        for column in HKL_COLUMNS:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.loc[~chunk[HKL_COLUMNS].isna().any(axis=1)].copy()
        chunk[HKL_COLUMNS] = chunk[HKL_COLUMNS].astype("int64")
        for column in usecols:
            if column not in KEY_COLUMNS:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunks.append(chunk)
        if idx % 5 == 0:
            log(f"Loaded diagnostic rows: {sum(len(c) for c in chunks):,}")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def q(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.quantile(quantile)) if not numeric.empty else np.nan


def add_quantiles(row: dict[str, Any], prefix: str, values: pd.Series) -> None:
    row[f"{prefix}_q10"] = q(values, 0.10)
    row[f"{prefix}_median"] = q(values, 0.50)
    row[f"{prefix}_q90"] = q(values, 0.90)


def percent_label(fraction: float) -> int:
    return int(round(float(fraction) * 100.0))


def selection_mask(table: pd.DataFrame, score_column: str, keep_fraction: float) -> pd.Series:
    work = table[[*HKL_COLUMNS, "source_filename", "event", score_column]].copy()
    work["_orig_index"] = table.index.to_numpy(dtype=np.int64)
    work = work.dropna(subset=[score_column]).sort_values(
        [*HKL_COLUMNS, score_column, "source_filename", "event"], kind="mergesort"
    )
    grouped = work.groupby(HKL_COLUMNS, sort=False)
    n_total = grouped[score_column].transform("size").astype("int64")
    rank = grouped.cumcount().astype("int64")
    n_keep = np.floor(n_total.to_numpy(dtype=float) * float(keep_fraction)).astype("int64")
    keep = rank.to_numpy(dtype=np.int64) < n_keep
    mask = pd.Series(False, index=table.index)
    mask.loc[work.loc[keep, "_orig_index"].to_numpy(dtype=np.int64)] = True
    return mask


def hkl_count_at_least(table: pd.DataFrame, min_count: int) -> int:
    if table.empty:
        return 0
    counts = table.groupby(HKL_COLUMNS, sort=False).size()
    return int((counts >= int(min_count)).sum())


def hkl_count_with_eg(table: pd.DataFrame, eg_column: str, eg_threshold: float, min_count: int) -> int:
    if table.empty or eg_column not in table:
        return 0
    subset = table.loc[pd.to_numeric(table[eg_column], errors="coerce") > float(eg_threshold)]
    return hkl_count_at_least(subset, int(min_count))


def summarize_selection(
    table: pd.DataFrame,
    method: str,
    score_column: str,
    keep_fraction: float,
    mask: pd.Series,
    metric_columns: list[str],
    partiality_columns: list[str],
    eg_column: str,
) -> dict[str, Any]:
    kept = table.loc[mask].copy()
    removed = table.loc[~mask].copy()
    row: dict[str, Any] = {
        "selection_method": method,
        "score_column": score_column,
        "keep_fraction": float(keep_fraction),
        "keep_tag": f"keep{percent_label(keep_fraction):02d}",
        "total_observations": int(len(table)),
        "kept_observations": int(len(kept)),
        "removed_observations": int(len(removed)),
        "unique_signed_hkls_total": int(table[HKL_COLUMNS].drop_duplicates().shape[0]),
        "unique_signed_hkls_kept": int(kept[HKL_COLUMNS].drop_duplicates().shape[0]) if not kept.empty else 0,
    }
    for threshold in OBS_THRESHOLDS:
        row[f"unique_signed_hkls_with_ge{threshold}_kept_obs"] = hkl_count_at_least(kept, threshold)
    for eg_threshold in EG_THRESHOLDS:
        threshold_label = str(eg_threshold).replace(".", "p")
        for obs_threshold in OBS_THRESHOLDS:
            row[f"unique_signed_hkls_with_ge{obs_threshold}_kept_obs_eg_gt_{threshold_label}"] = hkl_count_with_eg(
                kept, eg_column, eg_threshold, obs_threshold
            )
    for column in metric_columns + partiality_columns:
        add_quantiles(row, f"kept_{column}", kept[column] if column in kept else pd.Series(dtype=float))
        add_quantiles(row, f"removed_{column}", removed[column] if column in removed else pd.Series(dtype=float))
    return row


def markdown_summary(summary: pd.DataFrame, metadata: dict[str, Any]) -> str:
    lines = [
        "# V3 Direct-Low Filtering Excitation Bias Diagnostic",
        "",
        f"Created UTC: `{metadata['created_utc']}`",
        f"Input scores: `{metadata['inputs']['v3_scores']}`",
        f"Partiality-like columns detected: `{', '.join(metadata['detected_columns']['partiality_columns']) or '(none)'}`",
        "",
        "## Selection Summary",
        "",
    ]
    compact_columns = [
        "selection_method",
        "keep_tag",
        "kept_observations",
        "unique_signed_hkls_kept",
        "unique_signed_hkls_with_ge1_kept_obs",
        "unique_signed_hkls_with_ge2_kept_obs",
        "unique_signed_hkls_with_ge5_kept_obs",
        "kept_target_excitation_Eg_median",
        "removed_target_excitation_Eg_median",
        "kept_sg_target_median",
        "removed_sg_target_median",
    ]
    view = summary[[column for column in compact_columns if column in summary.columns]].copy()
    if view.empty:
        lines.append("No rows were summarized.")
    else:
        lines.append(view.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Direct v3-low selection ranks observations by the lowest target-gated risk per signed HKL.",
            "- The simulation uses the same floor(n_observations * keep_fraction) convention as the existing keep-fraction filter.",
            "- Low v3 risk can come from low target excitation, so compare kept versus removed `target_excitation_Eg` and `sg_target`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    header = read_header(args.v3_scores)
    trust_score_column = choose_trust_score_column(header, args.trust_score_column)
    partiality_columns = detect_partiality_columns(header)
    metric_columns = [args.eg_column, args.gate_column, args.sg_column, args.direct_score_column, args.v2_risk_column]
    required = [*KEY_COLUMNS, *metric_columns]
    require_columns(header, required, "v3 scores CSV")
    if trust_score_column is not None:
        require_columns(header, [trust_score_column], "v3 scores CSV")

    outdir = args.outdir if args.outdir is not None else args.v3_scores.parent / "diagnostics"
    csv_path, md_path, json_path = ensure_outputs(outdir, bool(args.overwrite))
    usecols = list(dict.fromkeys([*KEY_COLUMNS, *metric_columns, *partiality_columns, *( [trust_score_column] if trust_score_column else [] )]))

    log(f"Reading diagnostic table from {args.v3_scores}")
    table = load_table(args, usecols)
    log(f"Rows loaded: {len(table):,}")
    methods = [("direct_low_v3", args.direct_score_column)]
    if trust_score_column is not None:
        methods.append(("excited_lowcoupling_badness", trust_score_column))

    rows = []
    for method, score_column in methods:
        for keep_fraction in args.keep_fractions:
            log(f"Simulating {method} {keep_fraction:.2f}")
            mask = selection_mask(table, score_column, keep_fraction)
            rows.append(
                summarize_selection(
                    table,
                    method,
                    score_column,
                    keep_fraction,
                    mask,
                    metric_columns,
                    partiality_columns,
                    args.eg_column,
                )
            )

    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(csv_path, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"v3_scores": str(args.v3_scores)},
        "outputs": {"csv": str(csv_path), "markdown": str(md_path), "metadata": str(json_path)},
        "detected_columns": {"partiality_columns": partiality_columns, "trust_score_column": trust_score_column},
        "selection_convention": "floor(n_observations_for_signed_hkl * keep_fraction); no minimum-one override",
        "keep_fractions": [float(value) for value in args.keep_fractions],
        "max_rows": None if args.max_rows is None else int(args.max_rows),
    }
    md_path.write_text(markdown_summary(summary, metadata), encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"diagnostic_csv: {csv_path}")
    print(f"diagnostic_markdown: {md_path}")
    print(f"metadata_json: {json_path}")
    print(f"methods: {', '.join(name for name, _column in methods)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())