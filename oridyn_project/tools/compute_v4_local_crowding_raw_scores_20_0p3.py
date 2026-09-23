#!/usr/bin/env python3
"""Compute raw v4 local-crowding diagnostic scores for OriDyn 20-0.3 data.

This is score-only diagnostics. It does not normalize scores, reweight sigma, or
write filtered streams. The v2 core score is used only after confirming it is a
local-neighbor term stored as log1p(sum_top_edges(E(h) * C(h-g))). The raw local
neighbor sum is recovered as expm1(manybeam_coupling_v2_core_raw).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
V2_CORE_COLUMN = "manybeam_coupling_v2_core_raw"
DEFAULT_OUTPUT_CSV = "geometry_coupling_v4_local_crowding_raw_scores.csv"
DEFAULT_CHUNKSIZE = 500_000
DEFAULT_SG0 = 0.01

BASE_OPTIONAL_COLUMNS = ["I", "intensity", "sigma", "d_angstrom", "q_invA", "sg"]
OUTPUT_REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "d_angstrom",
    "inv_nm",
    "sg_target",
    "target_excitation_Eg",
    "local_neighbor_sum_raw",
    "local_neighbor_count_effective",
    "local_neighbor_max_Eh",
    "local_crowding_target_gated_raw",
    "local_crowding_target_gated_log1p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-scores", required=True, type=Path, help="20-0.3 geometry_coupling_v2_scores.csv")
    parser.add_argument("--base-scores", required=True, type=Path, help="Keyed 20-0.3 reflection_scores.csv with sg/d metadata")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument("--sg0", type=float, default=DEFAULT_SG0, help="Gaussian excitation scale; defaults to v2 sg0=0.01")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing v4 output files")
    args = parser.parse_args()

    if not args.v2_scores.exists():
        raise SystemExit(f"--v2-scores not found: {args.v2_scores}")
    if not args.base_scores.exists():
        raise SystemExit(f"--base-scores not found: {args.base_scores}")
    if float(args.sg0) <= 0.0:
        raise SystemExit("--sg0 must be > 0")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if args.max_rows is not None and int(args.max_rows) < 1:
        raise SystemExit("--max-rows must be >= 1 when provided")
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


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def require_columns(header: list[str], columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def output_paths(outdir: Path) -> dict[str, Path]:
    csv_path = outdir / DEFAULT_OUTPUT_CSV
    return {
        "csv": csv_path,
        "metadata": csv_path.with_name(f"{csv_path.stem}_metadata.json"),
        "summary": csv_path.with_name(f"{csv_path.stem}_summary.md"),
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    blocked = [path for path in paths.values() if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}\nUse --overwrite if intended.")
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)


def normalize_key_frame(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def keys_equal(v2_chunk: pd.DataFrame, base_chunk: pd.DataFrame) -> bool:
    return v2_chunk.loc[:, KEY_COLUMNS].reset_index(drop=True).equals(base_chunk.loc[:, KEY_COLUMNS].reset_index(drop=True))


def excitation_weight_from_sg(sg: pd.Series, sg0: float) -> pd.Series:
    values = pd.to_numeric(sg, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    scale = max(float(sg0), 1e-12)
    weights = np.exp(-((np.abs(values) / scale) ** 2))
    weights = np.where(np.isfinite(weights), weights, np.nan)
    return pd.Series(weights, index=sg.index, dtype="float64")


def finite_stats(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {"n_finite": 0, "min": None, "q10": None, "median": None, "q90": None, "max": None}
    data = np.concatenate(values).astype(float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"n_finite": 0, "min": None, "q10": None, "median": None, "q90": None, "max": None}
    q10, q50, q90 = np.quantile(data, [0.10, 0.50, 0.90])
    return {
        "n_finite": int(len(data)),
        "min": float(np.min(data)),
        "q10": float(q10),
        "median": float(q50),
        "q90": float(q90),
        "max": float(np.max(data)),
    }


def iter_limited_csv(path: Path, usecols: list[str], chunksize: int, max_rows: int | None):
    remaining = None if max_rows is None else int(max_rows)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        if chunk.empty:
            break
        yield chunk
        if remaining is not None and remaining <= 0:
            break


def compute_scores(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    v2_header = read_header(args.v2_scores)
    base_header = read_header(args.base_scores)
    require_columns(v2_header, [*KEY_COLUMNS, V2_CORE_COLUMN], "v2 scores")
    require_columns(base_header, [*KEY_COLUMNS, "sg"], "base scores")

    base_usecols = [*KEY_COLUMNS]
    for column in BASE_OPTIONAL_COLUMNS:
        if column in base_header and column not in base_usecols:
            base_usecols.append(column)
    v2_usecols = [*KEY_COLUMNS, V2_CORE_COLUMN]

    include_i_column = "I" in base_usecols
    include_intensity_column = "intensity" in base_usecols and not include_i_column
    include_sigma = "sigma" in base_usecols
    include_d = "d_angstrom" in base_usecols
    include_q = "q_invA" in base_usecols
    if not include_d and not include_q:
        raise SystemExit("base scores must contain d_angstrom or q_invA to compute resolution columns")

    tmp_csv = paths["csv"].with_suffix(paths["csv"].suffix + ".tmp")
    if tmp_csv.exists():
        tmp_csv.unlink()

    written = 0
    chunks = 0
    mismatches = 0
    missing_sg = 0
    missing_local = 0
    stats_values = {
        "target_excitation_Eg": [],
        "local_neighbor_sum_raw": [],
        "local_crowding_target_gated_raw": [],
        "local_crowding_target_gated_log1p": [],
    }

    v2_iter = iter_limited_csv(args.v2_scores, v2_usecols, int(args.chunksize), args.max_rows)
    base_iter = iter_limited_csv(args.base_scores, base_usecols, int(args.chunksize), args.max_rows)
    try:
        for chunks, (v2_chunk, base_chunk) in enumerate(zip(v2_iter, base_iter, strict=True), start=1):
            if len(v2_chunk) != len(base_chunk):
                raise RuntimeError(f"v2/base chunk length mismatch at chunk {chunks}: {len(v2_chunk)} vs {len(base_chunk)}")
            v2_work = normalize_key_frame(v2_chunk)
            base_work = normalize_key_frame(base_chunk)
            if not keys_equal(v2_work, base_work):
                mismatches += 1
                bad = (v2_work.loc[:, KEY_COLUMNS].reset_index(drop=True) != base_work.loc[:, KEY_COLUMNS].reset_index(drop=True)).any(axis=1)
                first_bad = int(np.flatnonzero(bad.to_numpy())[0]) if bool(bad.any()) else -1
                raise RuntimeError(f"v2/base exact-key mismatch at chunk {chunks}, row {first_bad}")

            out = v2_work.loc[:, KEY_COLUMNS].copy()
            if include_i_column:
                out["I"] = pd.to_numeric(base_work["I"], errors="coerce")
            elif include_intensity_column:
                out["I"] = pd.to_numeric(base_work["intensity"], errors="coerce")
            if include_sigma:
                out["sigma"] = pd.to_numeric(base_work["sigma"], errors="coerce")

            if include_d:
                d_values = pd.to_numeric(base_work["d_angstrom"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                q_values_for_d = pd.to_numeric(base_work["q_invA"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                d_values = 1.0 / q_values_for_d.where(q_values_for_d > 0)
            out["d_angstrom"] = d_values
            if include_q:
                q_values = pd.to_numeric(base_work["q_invA"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                out["inv_nm"] = 10.0 * q_values
            else:
                out["inv_nm"] = 10.0 / d_values.where(d_values > 0)

            sg_target = pd.to_numeric(base_work["sg"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out["sg_target"] = sg_target
            out["target_excitation_Eg"] = excitation_weight_from_sg(sg_target, float(args.sg0))
            core_raw = pd.to_numeric(v2_work[V2_CORE_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
            local_sum = np.expm1(core_raw.to_numpy(dtype=float))
            local_sum = np.where(np.isfinite(local_sum) & (local_sum >= 0.0), local_sum, np.nan)
            out["local_neighbor_sum_raw"] = local_sum
            out["local_neighbor_count_effective"] = np.nan
            out["local_neighbor_max_Eh"] = np.nan
            out["local_crowding_target_gated_raw"] = out["target_excitation_Eg"].to_numpy(dtype=float) * local_sum
            out["local_crowding_target_gated_log1p"] = np.log1p(out["local_crowding_target_gated_raw"].to_numpy(dtype=float))

            missing_sg += int(out["sg_target"].isna().sum())
            missing_local += int(out["local_neighbor_sum_raw"].isna().sum())
            for column in stats_values:
                values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                if len(values):
                    stats_values[column].append(values)

            output_columns = [*KEY_COLUMNS]
            if "I" in out:
                output_columns.append("I")
            if "sigma" in out:
                output_columns.append("sigma")
            output_columns.extend(OUTPUT_REQUIRED_COLUMNS[len(KEY_COLUMNS) :])
            out.loc[:, output_columns].to_csv(tmp_csv, index=False, mode="w" if written == 0 else "a", header=written == 0)
            written += int(len(out))
            if chunks == 1 or chunks % 5 == 0:
                log(f"Score pass: chunks={chunks:,}, rows_written={written:,}")
    except Exception:
        if tmp_csv.exists():
            tmp_csv.unlink()
        raise

    tmp_csv.replace(paths["csv"])
    return {
        "rows_written": int(written),
        "chunks_processed": int(chunks),
        "key_mismatch_chunks": int(mismatches),
        "missing_sg_target": int(missing_sg),
        "missing_local_neighbor_sum_raw": int(missing_local),
        "base_columns_used": base_usecols,
        "output_columns": pd.read_csv(paths["csv"], nrows=0).columns.tolist(),
        "distributions": {column: finite_stats(values) for column, values in stats_values.items()},
    }


def write_metadata(args: argparse.Namespace, paths: dict[str, Path], stats: dict[str, Any], total_rows: dict[str, int]) -> None:
    payload = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v2_scores": str(args.v2_scores),
            "base_scores": str(args.base_scores),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "parameters": {
            "sg0": float(args.sg0),
            "chunksize": int(args.chunksize),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
        },
        "score_definition": {
            "target_excitation_Eg": "exp(-(sg_target / sg0)^2)",
            "v2_core_term_confirmed": "manybeam_coupling_v2_core_raw is local-only but stored as log1p(sum_top_edges(E(h) * C(h-g)))",
            "local_neighbor_sum_raw": "expm1(manybeam_coupling_v2_core_raw)",
            "local_crowding_target_gated_raw": "target_excitation_Eg * local_neighbor_sum_raw",
            "local_crowding_target_gated_log1p": "log1p(local_crowding_target_gated_raw)",
        },
        "row_counts": total_rows,
        "stats": stats,
        "warnings": [
            "This is a raw score-only diagnostic; no filtering streams are created.",
            "No global normalization or shell normalization is applied.",
            "No v2 full score, row concentration boost, Laue-zone boost, frame-axis boost, trust transform, or badness transform is used.",
            "Exact signed HKL observation matching is validated chunk-by-chunk by source_filename + normalized event + h,k,l.",
            "local_neighbor_count_effective and local_neighbor_max_Eh are not available in the keyed 0p3 inputs and are written as NaN placeholders.",
        ],
    }
    paths["metadata"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(args: argparse.Namespace, paths: dict[str, Path], stats: dict[str, Any], total_rows: dict[str, int]) -> None:
    lines = [
        "# V4 Local Crowding Raw Diagnostic Summary",
        "",
        "This is a score-only diagnostic. It does not filter observations, create streams, or normalize risk.",
        "",
        "## Inputs",
        f"- v2 scores: `{args.v2_scores}`",
        f"- base scores: `{args.base_scores}`",
        "",
        "## Definition",
        "- `manybeam_coupling_v2_core_raw` was confirmed local-only but stored as `log1p(sum_top_edges(E(h) * C(h-g)))`.",
        "- `local_neighbor_sum_raw = expm1(manybeam_coupling_v2_core_raw)`.",
        f"- `target_excitation_Eg = exp(-(sg_target / {float(args.sg0):.6g})^2)`.",
        "- `local_crowding_target_gated_raw = target_excitation_Eg * local_neighbor_sum_raw`.",
        "",
        "## Rows",
        f"- v2 score rows: {total_rows['v2_scores_rows']:,}",
        f"- base score rows: {total_rows['base_scores_rows']:,}",
        f"- output rows: {stats['rows_written']:,}",
        "",
        "## Output Columns",
        "- " + "\n- ".join(stats["output_columns"]),
        "",
        "## Distributions",
    ]
    for column, meta in stats["distributions"].items():
        lines.append(
            f"- `{column}`: n={meta['n_finite']}, min={meta['min']}, q10={meta['q10']}, "
            f"median={meta['median']}, q90={meta['q90']}, max={meta['max']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- No columns ending in `_norm` are written.",
            "- `local_neighbor_count_effective` and `local_neighbor_max_Eh` are included as NaN placeholders because they are not present in the keyed 0p3 inputs and are not cheap to recover from the log-summed v2 core output.",
        ]
    )
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = output_paths(args.outdir)
    ensure_outputs(paths, bool(args.overwrite))
    total_rows = {
        "v2_scores_rows": count_rows(args.v2_scores),
        "base_scores_rows": count_rows(args.base_scores),
    }
    log(f"v2 scores: {args.v2_scores}")
    log(f"base scores: {args.base_scores}")
    log(f"output CSV: {paths['csv']}")
    log(f"sg0: {float(args.sg0):.6g}")
    stats = compute_scores(args, paths)
    write_metadata(args, paths, stats, total_rows)
    write_summary(args, paths, stats, total_rows)
    print("V4 local-crowding raw scores written")
    print(f"output_csv: {paths['csv']}")
    print(f"metadata_json: {paths['metadata']}")
    print(f"summary_md: {paths['summary']}")
    print(f"rows_written: {stats['rows_written']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())