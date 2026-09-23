#!/usr/bin/env python3
"""Compute v3 target-gated geometry-coupling scores from an existing v2 table.

This script is intentionally separate from the v2 scorer. It reads v2 scores,
preserves all v2 columns in the output, and appends target-gated v3 columns.
It does not recompute or modify v2 scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sqlite3
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oridyn.coupling_exposure_v2 import (  # noqa: E402
    HKL_COLUMNS,
    KEY_COLUMNS,
    robust_p01_p99_normalize,
)
from oridyn.geometry import d_spacings_from_q, excitation_error, hkl_lab_vectors, vector_norms  # noqa: E402
from oridyn.stream_parser import parse_crystfel_stream, reciprocal_matrix_from_row  # noqa: E402


DEFAULT_OUTPUT_CSV = "geometry_coupling_v3_target_gated_scores.csv"
DEFAULT_RAW_V2_COLUMN = "manybeam_coupling_v2_full_raw"
DEFAULT_V2_RISK_COLUMN = "trust_risk_v2_full_norm"
DEFAULT_CHUNKSIZE = 500_000
SG_CANDIDATES = ["sg_target", "sg", "s_g", "excitation_error", "ewald_error", "ewald_dist"]
DIAGNOSTIC_HKLS = [(0, 4, 0), (0, 27, 5), (6, 6, 2)]
DIAGNOSTIC_TAIL_N = 10
SHELLS_INV_NM = [
    (0.500, 10.526),
    (10.526, 13.262),
    (13.262, 15.181),
    (15.181, 16.709),
    (16.709, 17.999),
    (17.999, 19.127),
    (19.127, 20.135),
    (20.135, 21.052),
    (21.052, 21.895),
    (21.895, 22.677),
    (22.677, 23.409),
    (23.409, 24.098),
    (24.098, 24.750),
    (24.750, 25.369),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-scores", required=True, type=Path, help="Existing geometry_coupling_v2_scores.csv")
    parser.add_argument(
        "--base-scores",
        type=Path,
        default=None,
        help="Optional base reflection_scores.csv carrying sg/d metadata. Auto-detected from v2 run_metadata.json if omitted.",
    )
    parser.add_argument("--stream", type=Path, default=None, help="Optional CrystFEL stream fallback for sg_target/d-spacing")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path, or output directory")
    parser.add_argument("--raw-v2-column", default=DEFAULT_RAW_V2_COLUMN)
    parser.add_argument("--lambda-ang", type=float, default=0.020)
    parser.add_argument("--target-s0", type=float, default=0.005)
    parser.add_argument("--target-power", type=float, default=2.0)
    parser.add_argument("--target-gate-alpha", type=float, default=1.0)
    parser.add_argument("--target-gate-floor", type=float, default=0.15)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap from the v2 CSV")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing v3 output files")
    parser.add_argument(
        "--v2-risk-column",
        default=DEFAULT_V2_RISK_COLUMN,
        help="Normalized v2 column used only for diagnostics",
    )
    args = parser.parse_args()

    if not args.v2_scores.exists():
        raise SystemExit(f"--v2-scores not found: {args.v2_scores}")
    if args.base_scores is not None and not args.base_scores.exists():
        raise SystemExit(f"--base-scores not found: {args.base_scores}")
    if args.stream is not None and not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if args.lambda_ang <= 0.0:
        raise SystemExit("--lambda-ang must be > 0")
    if args.target_s0 <= 0.0:
        raise SystemExit("--target-s0 must be > 0")
    if args.target_power <= 0.0:
        raise SystemExit("--target-power must be > 0")
    if args.target_gate_alpha <= 0.0:
        raise SystemExit("--target-gate-alpha must be > 0")
    if not 0.0 <= args.target_gate_floor <= 1.0:
        raise SystemExit("--target-gate-floor must be between 0 and 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    if args.max_rows is not None and args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1 when provided")
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def output_csv_path(out: Path) -> Path:
    if out.suffix.lower() == ".csv":
        return out
    return out / DEFAULT_OUTPUT_CSV


def sibling_path(csv_path: Path, suffix: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_{suffix}")


def metadata_source_scores(v2_scores: Path) -> Path | None:
    metadata_path = v2_scores.parent / "run_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source = payload.get("inputs", {}).get("scores_csv")
    if not source:
        return None
    source_path = Path(source)
    return source_path if source_path.exists() else None


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


def normalize_key_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[~out[HKL_COLUMNS].isna().any(axis=1)].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def detect_sg_column(header: Iterable[str]) -> str | None:
    available = set(str(column) for column in header)
    for column in SG_CANDIDATES:
        if column in available:
            return column
    return None


def detect_resolution_method(header: Iterable[str]) -> str | None:
    available = set(str(column) for column in header)
    if "d_angstrom" in available:
        return "d_angstrom"
    if "q_invA" in available:
        return "q_invA"
    return None


def add_resolution_from_columns(table: pd.DataFrame, method: str | None) -> pd.DataFrame:
    out = table.copy()
    if method == "d_angstrom":
        d_values = pd.to_numeric(out["d_angstrom"], errors="coerce").to_numpy(dtype=float)
        out["d_for_shell_angstrom"] = d_values
        out["inv_nm_for_shell"] = np.divide(
            10.0,
            d_values,
            out=np.full_like(d_values, np.nan, dtype=float),
            where=np.isfinite(d_values) & (d_values > 0.0),
        )
    elif method == "q_invA":
        q_values = pd.to_numeric(out["q_invA"], errors="coerce").to_numpy(dtype=float)
        out["inv_nm_for_shell"] = 10.0 * q_values
        out["d_for_shell_angstrom"] = np.divide(
            1.0,
            q_values,
            out=np.full_like(q_values, np.nan, dtype=float),
            where=np.isfinite(q_values) & (q_values > 0.0),
        )
    else:
        out["d_for_shell_angstrom"] = np.nan
        out["inv_nm_for_shell"] = np.nan
    return out


def shell_info(inv_nm: float) -> tuple[int | None, float | None, float | None]:
    if not np.isfinite(inv_nm):
        return None, None, None
    for shell_index, (shell_min, shell_max) in enumerate(SHELLS_INV_NM, start=1):
        is_last = shell_index == len(SHELLS_INV_NM)
        if (shell_min <= inv_nm < shell_max) or (is_last and shell_min <= inv_nm <= shell_max):
            return shell_index, shell_min, shell_max
    return None, None, None


def add_shell_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    shell_indices: list[float] = []
    shell_mins: list[float] = []
    shell_maxes: list[float] = []
    for value in pd.to_numeric(out["inv_nm_for_shell"], errors="coerce").to_numpy(dtype=float):
        shell_index, shell_min, shell_max = shell_info(float(value))
        shell_indices.append(np.nan if shell_index is None else float(shell_index))
        shell_mins.append(np.nan if shell_min is None else float(shell_min))
        shell_maxes.append(np.nan if shell_max is None else float(shell_max))
    out["resolution_shell_index"] = shell_indices
    out["resolution_shell_min_inv_nm"] = shell_mins
    out["resolution_shell_max_inv_nm"] = shell_maxes
    return out


def target_excitation_from_sg(sg_values: np.ndarray, target_s0: float, target_power: float) -> np.ndarray:
    values = np.asarray(sg_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    scale = max(float(target_s0), 1e-12)
    result[finite] = np.exp(-((np.abs(values[finite]) / scale) ** float(target_power)))
    return result


def target_gate_from_excitation(excitation: np.ndarray, gate_alpha: float, gate_floor: float) -> np.ndarray:
    values = np.asarray(excitation, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    clipped = np.clip(values[finite], 0.0, 1.0)
    result[finite] = float(gate_floor) + (1.0 - float(gate_floor)) * (clipped ** float(gate_alpha))
    return result


def iter_score_chunks(path: Path, chunksize: int, max_rows: int | None) -> Iterable[pd.DataFrame]:
    rows_seen = 0
    for chunk in pd.read_csv(path, chunksize=int(chunksize)):
        if max_rows is not None:
            remaining = int(max_rows) - rows_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        rows_seen += int(len(chunk))
        if not chunk.empty:
            yield chunk
        if max_rows is not None and rows_seen >= int(max_rows):
            break


def connect_lookup(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lookup (
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            sg_target REAL,
            d_for_shell_angstrom REAL,
            inv_nm_for_shell REAL,
            PRIMARY KEY (source_filename, event, h, k, l)
        ) WITHOUT ROWID
        """
    )
    return connection


def insert_lookup_records(connection: sqlite3.Connection, records: list[tuple[Any, ...]]) -> None:
    if not records:
        return
    connection.executemany(
        """
        INSERT OR IGNORE INTO lookup
        (source_filename, event, h, k, l, sg_target, d_for_shell_angstrom, inv_nm_for_shell)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )


def build_lookup_from_base(
    base_scores: Path,
    sg_column: str | None,
    resolution_method: str | None,
    chunksize: int,
    db_path: Path,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    connection = connect_lookup(db_path)
    usecols = list(KEY_COLUMNS)
    if sg_column is not None:
        usecols.append(sg_column)
    if resolution_method is not None and resolution_method not in usecols:
        usecols.append(resolution_method)

    stats: dict[str, Any] = {
        "source": str(base_scores),
        "source_type": "base_scores",
        "sg_column": sg_column,
        "resolution_method": resolution_method,
        "rows_read": 0,
        "rows_after_cleanup": 0,
        "lookup_rows_inserted": 0,
    }
    for chunk_index, chunk in enumerate(pd.read_csv(base_scores, usecols=usecols, chunksize=int(chunksize)), start=1):
        stats["rows_read"] += int(len(chunk))
        work = normalize_key_columns(chunk)
        stats["rows_after_cleanup"] += int(len(work))
        if sg_column is not None:
            work["sg_target"] = pd.to_numeric(work[sg_column], errors="coerce")
        else:
            work["sg_target"] = np.nan
        work = add_resolution_from_columns(work, resolution_method)
        records = [
            (
                str(row.source_filename),
                str(row.event),
                int(row.h),
                int(row.k),
                int(row.l),
                none_if_nan(row.sg_target),
                none_if_nan(row.d_for_shell_angstrom),
                none_if_nan(row.inv_nm_for_shell),
            )
            for row in work[
                [*KEY_COLUMNS, "sg_target", "d_for_shell_angstrom", "inv_nm_for_shell"]
            ].itertuples(index=False)
        ]
        insert_lookup_records(connection, records)
        connection.commit()
        stats["lookup_rows_inserted"] = int(connection.execute("SELECT COUNT(*) FROM lookup").fetchone()[0])
        if chunk_index % 5 == 0:
            log(
                "Base lookup progress: "
                f"rows_read={stats['rows_read']:,}, lookup_rows={stats['lookup_rows_inserted']:,}"
            )
    return connection, stats


def none_if_nan(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def build_lookup_from_stream(
    stream_path: Path,
    lambda_angstrom: float,
    db_path: Path,
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    log("Warning: building sg_target lookup from stream; this parser loads stream metadata into memory")
    stream = parse_crystfel_stream(stream_path)
    connection = connect_lookup(db_path)
    stats: dict[str, Any] = {
        "source": str(stream_path),
        "source_type": "stream",
        "lambda_angstrom": float(lambda_angstrom),
        "stream_wavelength_angstrom": float(stream.wavelength_angstrom),
        "frames": int(stream.crystal_table["frame"].nunique()) if not stream.crystal_table.empty else 0,
        "reflections": int(len(stream.reflections)),
        "lookup_rows_inserted": 0,
    }
    if stream.reflections.empty:
        return connection, stats

    crystal_by_frame = stream.crystal_table.set_index("frame", drop=False)
    for frame_number, group in stream.reflections.groupby("frame", sort=True):
        if int(frame_number) not in crystal_by_frame.index:
            continue
        crystal = crystal_by_frame.loc[int(frame_number)]
        reciprocal = reciprocal_matrix_from_row(crystal)
        hkls = group[HKL_COLUMNS].to_numpy(dtype=int)
        g_vectors = hkl_lab_vectors(hkls, reciprocal)
        sg_values = excitation_error(g_vectors, float(lambda_angstrom))
        q_values = vector_norms(g_vectors)
        d_values = d_spacings_from_q(q_values)
        work = group[list(KEY_COLUMNS)].copy()
        work["sg_target"] = sg_values
        work["d_for_shell_angstrom"] = d_values
        work["inv_nm_for_shell"] = np.divide(
            10.0,
            d_values,
            out=np.full_like(d_values, np.nan, dtype=float),
            where=np.isfinite(d_values) & (d_values > 0.0),
        )
        work = normalize_key_columns(work)
        records = [
            (
                str(row.source_filename),
                str(row.event),
                int(row.h),
                int(row.k),
                int(row.l),
                none_if_nan(row.sg_target),
                none_if_nan(row.d_for_shell_angstrom),
                none_if_nan(row.inv_nm_for_shell),
            )
            for row in work.itertuples(index=False)
        ]
        insert_lookup_records(connection, records)
    connection.commit()
    stats["lookup_rows_inserted"] = int(connection.execute("SELECT COUNT(*) FROM lookup").fetchone()[0])
    return connection, stats


def lookup_chunk_values(connection: sqlite3.Connection, chunk: pd.DataFrame) -> pd.DataFrame:
    keys = normalize_key_columns(chunk[list(KEY_COLUMNS)]).reset_index(drop=True)
    keys.insert(0, "row_position", np.arange(len(keys), dtype=np.int64))
    connection.execute("DROP TABLE IF EXISTS chunk_keys")
    keys.to_sql("chunk_keys", connection, index=False, if_exists="replace")
    joined = pd.read_sql_query(
        """
        SELECT
            ck.row_position,
            lu.sg_target AS lookup_sg_target,
            lu.d_for_shell_angstrom AS lookup_d_for_shell_angstrom,
            lu.inv_nm_for_shell AS lookup_inv_nm_for_shell
        FROM chunk_keys AS ck
        LEFT JOIN lookup AS lu
            ON ck.source_filename = lu.source_filename
           AND ck.event = lu.event
           AND ck.h = lu.h
           AND ck.k = lu.k
           AND ck.l = lu.l
        ORDER BY ck.row_position
        """,
        connection,
    )
    return joined


def robust_metadata_from_values(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        series = pd.Series(dtype=float)
    else:
        series = pd.Series(np.concatenate(values).astype(float))
    _normalized, metadata = robust_p01_p99_normalize(series)
    return metadata


def apply_robust_normalization(values: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    result = np.full(numeric.shape, np.nan, dtype=float)
    finite = np.isfinite(numeric)
    if not finite.any():
        return result
    method = metadata.get("method")
    if method == "robust_p01_p99":
        p01 = float(metadata["p01"])
        p99 = float(metadata["p99"])
        result[finite] = np.clip((numeric[finite] - p01) / max(p99 - p01, 1e-12), 0.0, 1.0)
    elif method == "degenerate_p01_p99_to_zero":
        result[finite] = 0.0
    else:
        result[finite] = np.nan
    return result


def enrich_chunk(
    chunk: pd.DataFrame,
    args: argparse.Namespace,
    lookup_connection: sqlite3.Connection | None,
    sg_source_column: str | None,
    sg_source_type: str,
    resolution_method: str | None,
    resolution_source_type: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = chunk.copy()
    stats = {
        "rows": int(len(out)),
        "missing_sg_target": 0,
        "missing_resolution": 0,
        "missing_lookup_sg": 0,
        "missing_lookup_resolution": 0,
    }

    lookup_values = None
    if lookup_connection is not None and (sg_source_type in {"base", "stream"} or resolution_source_type in {"base", "stream"}):
        lookup_values = lookup_chunk_values(lookup_connection, out)

    if sg_source_type == "v2" and sg_source_column is not None:
        out["sg_target"] = pd.to_numeric(out[sg_source_column], errors="coerce")
    elif lookup_values is not None:
        out["sg_target"] = pd.to_numeric(lookup_values["lookup_sg_target"], errors="coerce").to_numpy(dtype=float)
        stats["missing_lookup_sg"] = int(out["sg_target"].isna().sum())
    else:
        out["sg_target"] = np.nan
    stats["missing_sg_target"] = int(out["sg_target"].isna().sum())

    if resolution_source_type == "v2":
        out = add_resolution_from_columns(out, resolution_method)
    elif lookup_values is not None:
        out["d_for_shell_angstrom"] = pd.to_numeric(
            lookup_values["lookup_d_for_shell_angstrom"], errors="coerce"
        ).to_numpy(dtype=float)
        out["inv_nm_for_shell"] = pd.to_numeric(
            lookup_values["lookup_inv_nm_for_shell"], errors="coerce"
        ).to_numpy(dtype=float)
        stats["missing_lookup_resolution"] = int(out["inv_nm_for_shell"].isna().sum())
    else:
        out = add_resolution_from_columns(out, None)
    stats["missing_resolution"] = int(out["inv_nm_for_shell"].isna().sum())

    raw_v2 = pd.to_numeric(out[args.raw_v2_column], errors="coerce").to_numpy(dtype=float)
    sg_values = pd.to_numeric(out["sg_target"], errors="coerce").to_numpy(dtype=float)
    excitation = target_excitation_from_sg(sg_values, args.target_s0, args.target_power)
    gate = target_gate_from_excitation(excitation, args.target_gate_alpha, args.target_gate_floor)
    out["target_excitation_Eg"] = excitation
    out["target_excitation_gate_Gg"] = gate
    out["manybeam_coupling_v3_target_gated_raw"] = gate * raw_v2
    out = add_shell_columns(out)
    return out, stats


def collect_normalization_values(
    args: argparse.Namespace,
    lookup_connection: sqlite3.Connection | None,
    sg_source_column: str | None,
    sg_source_type: str,
    resolution_method: str | None,
    resolution_source_type: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, int]]:
    global_values: list[np.ndarray] = []
    shell_values: dict[int, list[np.ndarray]] = defaultdict(list)
    totals = {
        "rows": 0,
        "finite_v3_raw": 0,
        "missing_sg_target": 0,
        "missing_resolution": 0,
        "missing_lookup_sg": 0,
        "missing_lookup_resolution": 0,
    }
    for chunk_number, chunk in enumerate(iter_score_chunks(args.v2_scores, args.chunksize, args.max_rows), start=1):
        enriched, stats = enrich_chunk(
            chunk,
            args,
            lookup_connection,
            sg_source_column,
            sg_source_type,
            resolution_method,
            resolution_source_type,
        )
        for key, value in stats.items():
            totals[key] += int(value)
        raw = pd.to_numeric(enriched["manybeam_coupling_v3_target_gated_raw"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw)
        totals["finite_v3_raw"] += int(finite.sum())
        if finite.any():
            global_values.append(raw[finite])
        shell_series = pd.to_numeric(enriched["resolution_shell_index"], errors="coerce")
        for shell_index in sorted(shell_series.dropna().astype(int).unique().tolist()):
            mask = (shell_series.to_numpy(dtype=float) == float(shell_index)) & finite
            if mask.any():
                shell_values[int(shell_index)].append(raw[mask])
        if chunk_number % 5 == 0:
            log(f"Normalization pass progress: rows={totals['rows']:,}, finite_v3_raw={totals['finite_v3_raw']:,}")

    global_metadata = robust_metadata_from_values(global_values)
    global_metadata.update(
        {
            "raw_column": "manybeam_coupling_v3_target_gated_raw",
            "normalized_column": "trust_risk_v3_target_gated_norm",
            "scope": "global",
            "method_note": "same robust p01-p99 clipping convention as v2 normalization",
        }
    )
    shell_metadata: dict[int, dict[str, Any]] = {}
    for shell_index, values in sorted(shell_values.items()):
        metadata = robust_metadata_from_values(values)
        shell_min, shell_max = SHELLS_INV_NM[int(shell_index) - 1]
        metadata.update(
            {
                "raw_column": "manybeam_coupling_v3_target_gated_raw",
                "normalized_column": "trust_risk_v3_target_gated_shell_norm",
                "scope": "fixed_resolution_shell_inv_nm",
                "shell_index": int(shell_index),
                "shell_min_inv_nm": float(shell_min),
                "shell_max_inv_nm": float(shell_max),
                "method_note": "robust p01-p99 clipping within fixed requested shell",
            }
        )
        shell_metadata[int(shell_index)] = metadata
    return global_metadata, shell_metadata, totals


def add_normalized_v3_columns(
    table: pd.DataFrame,
    global_metadata: dict[str, Any],
    shell_metadata: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    out = table.copy()
    raw = pd.to_numeric(out["manybeam_coupling_v3_target_gated_raw"], errors="coerce").to_numpy(dtype=float)
    out["trust_risk_v3_target_gated_norm"] = apply_robust_normalization(raw, global_metadata)
    shell_norm = np.full(len(out), np.nan, dtype=float)
    shell_values = pd.to_numeric(out["resolution_shell_index"], errors="coerce").to_numpy(dtype=float)
    for shell_index, metadata in shell_metadata.items():
        mask = np.isfinite(shell_values) & (shell_values.astype(float) == float(shell_index))
        if mask.any():
            shell_norm[mask] = apply_robust_normalization(raw[mask], metadata)
    out["trust_risk_v3_target_gated_shell_norm"] = shell_norm
    return out


def write_v3_scores(
    args: argparse.Namespace,
    csv_path: Path,
    lookup_connection: sqlite3.Connection | None,
    sg_source_column: str | None,
    sg_source_type: str,
    resolution_method: str | None,
    resolution_source_type: str,
    global_metadata: dict[str, Any],
    shell_metadata: dict[int, dict[str, Any]],
) -> tuple[dict[str, int], dict[tuple[int, int, int], list[pd.DataFrame]]]:
    write_header = True
    totals = {
        "rows_written": 0,
        "missing_sg_target": 0,
        "missing_resolution": 0,
        "missing_lookup_sg": 0,
        "missing_lookup_resolution": 0,
    }
    diagnostic_rows: dict[tuple[int, int, int], list[pd.DataFrame]] = defaultdict(list)
    target_set = set(DIAGNOSTIC_HKLS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    for chunk_number, chunk in enumerate(iter_score_chunks(args.v2_scores, args.chunksize, args.max_rows), start=1):
        enriched, stats = enrich_chunk(
            chunk,
            args,
            lookup_connection,
            sg_source_column,
            sg_source_type,
            resolution_method,
            resolution_source_type,
        )
        enriched = add_normalized_v3_columns(enriched, global_metadata, shell_metadata)
        enriched.to_csv(csv_path, index=False, mode="w" if write_header else "a", header=write_header)
        write_header = False
        totals["rows_written"] += int(len(enriched))
        for key in ("missing_sg_target", "missing_resolution", "missing_lookup_sg", "missing_lookup_resolution"):
            totals[key] += int(stats[key])
        for hkl_triplet in target_set:
            hkl_mask = (
                (pd.to_numeric(enriched["h"], errors="coerce") == hkl_triplet[0])
                & (pd.to_numeric(enriched["k"], errors="coerce") == hkl_triplet[1])
                & (pd.to_numeric(enriched["l"], errors="coerce") == hkl_triplet[2])
            )
            if hkl_mask.any():
                diagnostic_rows[hkl_triplet].append(enriched.loc[hkl_mask].copy())
        if chunk_number % 5 == 0:
            log(f"Write pass progress: rows_written={totals['rows_written']:,}")
    return totals, diagnostic_rows


def quantile_summary(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"min": None, "q10": None, "median": None, "q90": None, "max": None}
    return {
        "min": float(numeric.quantile(0.0)),
        "q10": float(numeric.quantile(0.10)),
        "median": float(numeric.quantile(0.50)),
        "q90": float(numeric.quantile(0.90)),
        "max": float(numeric.quantile(1.0)),
    }


def summarize_tail_selection(table: pd.DataFrame, v2_column: str, v3_column: str) -> dict[str, Any]:
    work = table.reset_index(drop=True).copy()
    tail_count = min(DIAGNOSTIC_TAIL_N, int(len(work)))
    if tail_count <= 0:
        return {"tail_n": 0}

    v2_numeric = pd.to_numeric(work[v2_column], errors="coerce") if v2_column in work else pd.Series(np.nan, index=work.index)
    v3_numeric = pd.to_numeric(work[v3_column], errors="coerce") if v3_column in work else pd.Series(np.nan, index=work.index)
    v2_valid = work.loc[v2_numeric.notna()].copy()
    v3_valid = work.loc[v3_numeric.notna()].copy()
    v2_low = set(v2_valid.assign(_risk=v2_numeric.loc[v2_valid.index]).nsmallest(tail_count, "_risk").index)
    v2_high = set(v2_valid.assign(_risk=v2_numeric.loc[v2_valid.index]).nlargest(tail_count, "_risk").index)
    v3_low = set(v3_valid.assign(_risk=v3_numeric.loc[v3_valid.index]).nsmallest(tail_count, "_risk").index)
    v3_high = set(v3_valid.assign(_risk=v3_numeric.loc[v3_valid.index]).nlargest(tail_count, "_risk").index)

    def median_for(indices: set[int], column: str) -> float | None:
        if not indices or column not in work:
            return None
        values = pd.to_numeric(work.loc[sorted(indices), column], errors="coerce").dropna()
        return None if values.empty else float(values.median())

    return {
        "tail_n": int(tail_count),
        "v2_low_count": int(len(v2_low)),
        "v3_low_count": int(len(v3_low)),
        "low_overlap_count": int(len(v2_low & v3_low)),
        "v2_low_only_count": int(len(v2_low - v3_low)),
        "v3_low_only_count": int(len(v3_low - v2_low)),
        "v2_high_count": int(len(v2_high)),
        "v3_high_count": int(len(v3_high)),
        "high_overlap_count": int(len(v2_high & v3_high)),
        "v2_high_only_count": int(len(v2_high - v3_high)),
        "v3_high_only_count": int(len(v3_high - v2_high)),
        "v2_high_median_target_Eg": median_for(v2_high, "target_excitation_Eg"),
        "v3_high_median_target_Eg": median_for(v3_high, "target_excitation_Eg"),
        "v2_high_median_gate_Gg": median_for(v2_high, "target_excitation_gate_Gg"),
        "v3_high_median_gate_Gg": median_for(v3_high, "target_excitation_gate_Gg"),
    }


def build_diagnostics(
    diagnostic_rows: dict[tuple[int, int, int], list[pd.DataFrame]],
    v2_risk_column: str,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    v3_risk_column = "trust_risk_v3_target_gated_norm"
    for hkl_triplet in DIAGNOSTIC_HKLS:
        label = f"{hkl_triplet[0]} {hkl_triplet[1]} {hkl_triplet[2]}"
        if not diagnostic_rows.get(hkl_triplet):
            diagnostics[label] = {"n_observations": 0, "available": False}
            continue
        table = pd.concat(diagnostic_rows[hkl_triplet], ignore_index=True)
        diagnostics[label] = {
            "available": True,
            "n_observations": int(len(table)),
            "v2_risk_column": v2_risk_column,
            "v2_risk": quantile_summary(table[v2_risk_column]) if v2_risk_column in table else None,
            "v3_target_gated_risk_column": v3_risk_column,
            "v3_target_gated_risk": quantile_summary(table[v3_risk_column]),
            "sg_target": quantile_summary(table["sg_target"]),
            "target_excitation_Eg": quantile_summary(table["target_excitation_Eg"]),
            "target_excitation_gate_Gg": quantile_summary(table["target_excitation_gate_Gg"]),
            "low_high_tail_selection": summarize_tail_selection(table, v2_risk_column, v3_risk_column),
        }
    return diagnostics


def print_diagnostics(diagnostics: dict[str, Any]) -> None:
    print("\nDiagnostics for requested HKLs")
    for label, payload in diagnostics.items():
        print(f"HKL {label}:")
        if not payload.get("available"):
            print("  observations: 0 in this run")
            continue
        print(f"  observations: {payload['n_observations']}")
        print(f"  v2 risk: {format_stats(payload.get('v2_risk'))}")
        print(f"  v3 target-gated risk: {format_stats(payload.get('v3_target_gated_risk'))}")
        print(f"  sg_target: {format_stats(payload.get('sg_target'))}")
        print(f"  target Eg: {format_stats(payload.get('target_excitation_Eg'))}")
        tail = payload.get("low_high_tail_selection", {})
        print(
            "  low/high tail selection: "
            f"tail_n={tail.get('tail_n')}, "
            f"low_overlap={tail.get('low_overlap_count')}, "
            f"v2_low_only={tail.get('v2_low_only_count')}, "
            f"v3_low_only={tail.get('v3_low_only_count')}, "
            f"high_overlap={tail.get('high_overlap_count')}, "
            f"v2_high_only={tail.get('v2_high_only_count')}, "
            f"v3_high_only={tail.get('v3_high_only_count')}, "
            f"v2_high_Eg_med={format_optional_float(tail.get('v2_high_median_target_Eg'))}, "
            f"v3_high_Eg_med={format_optional_float(tail.get('v3_high_median_target_Eg'))}"
        )


def format_optional_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(numeric) else f"{numeric:.6g}"


def format_stats(stats: Any) -> str:
    if not isinstance(stats, dict):
        return "NA"
    return ", ".join(f"{key}={format_optional_float(stats.get(key))}" for key in ["min", "q10", "median", "q90", "max"])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def command_text() -> str:
    return shlex.join(sys.argv)


def ensure_outputs(csv_path: Path, overwrite: bool) -> tuple[Path, Path]:
    metadata_path = sibling_path(csv_path, "metadata.json")
    diagnostics_path = sibling_path(csv_path, "diagnostics.json")
    blocked = [path for path in [csv_path, metadata_path, diagnostics_path] if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing v3 output file(s):\n{formatted}\nUse --overwrite if intended.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return metadata_path, diagnostics_path


def main() -> int:
    args = parse_args()
    csv_path = output_csv_path(args.out)
    metadata_path, diagnostics_path = ensure_outputs(csv_path, bool(args.overwrite))

    v2_header = read_header(args.v2_scores)
    missing_v2_keys = [column for column in KEY_COLUMNS if column not in v2_header]
    if missing_v2_keys:
        raise SystemExit("--v2-scores is missing key column(s): " + ", ".join(missing_v2_keys))
    if args.raw_v2_column not in v2_header:
        raise SystemExit(f"--raw-v2-column {args.raw_v2_column!r} is not present in --v2-scores")

    auto_base_scores = metadata_source_scores(args.v2_scores)
    base_scores = args.base_scores or auto_base_scores
    base_header: list[str] = read_header(base_scores) if base_scores is not None and base_scores.exists() else []

    v2_sg_column = detect_sg_column(v2_header)
    base_sg_column = detect_sg_column(base_header) if base_header else None
    v2_resolution_method = detect_resolution_method(v2_header)
    base_resolution_method = detect_resolution_method(base_header) if base_header else None

    warnings: list[str] = []
    if v2_sg_column is not None:
        sg_source_type = "v2"
        sg_source_column = v2_sg_column
    elif base_sg_column is not None and base_scores is not None:
        sg_source_type = "base"
        sg_source_column = base_sg_column
    elif args.stream is not None:
        sg_source_type = "stream"
        sg_source_column = None
        warnings.append("sg_target was not found in v2/base scores; computing from stream UB matrices")
    else:
        raise SystemExit(
            "Could not find sg_target in v2 scores or base scores, and --stream was not provided. "
            "Tried columns: " + ", ".join(SG_CANDIDATES)
        )

    if v2_resolution_method is not None:
        resolution_source_type = "v2"
        resolution_method = v2_resolution_method
    elif base_resolution_method is not None and base_scores is not None:
        resolution_source_type = "base"
        resolution_method = base_resolution_method
    elif args.stream is not None:
        resolution_source_type = "stream"
        resolution_method = None
        warnings.append("Resolution metadata was not found in v2/base scores; computing d-spacing from stream UB matrices")
    else:
        resolution_source_type = "none"
        resolution_method = None
        warnings.append("Resolution metadata was not found; shell-normalized v3 values will be NaN")

    log(f"v2 scores: {args.v2_scores}")
    log(f"base scores: {base_scores if base_scores is not None else '(none)'}")
    log(f"sg_target source: {sg_source_type} column={sg_source_column}")
    log(f"resolution source: {resolution_source_type} method={resolution_method}")
    log(f"output CSV: {csv_path}")

    lookup_connection: sqlite3.Connection | None = None
    lookup_stats: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="oridyn_v3_lookup_", dir=str(csv_path.parent)) as temp_dir:
        db_path = Path(temp_dir) / "target_lookup.sqlite"
        if sg_source_type == "base" or resolution_source_type == "base":
            if base_scores is None:
                raise SystemExit("Internal error: base lookup requested without a base score path")
            log("Building target sg/resolution lookup from base scores")
            lookup_connection, lookup_stats = build_lookup_from_base(
                base_scores,
                base_sg_column if sg_source_type == "base" else None,
                base_resolution_method if resolution_source_type == "base" else None,
                int(args.chunksize),
                db_path,
            )
        elif sg_source_type == "stream" or resolution_source_type == "stream":
            if args.stream is None:
                raise SystemExit("Internal error: stream lookup requested without --stream")
            log("Building target sg/resolution lookup from stream")
            lookup_connection, lookup_stats = build_lookup_from_stream(args.stream, float(args.lambda_ang), db_path)

        log("Pass 1/2: collecting v3 normalization statistics")
        global_metadata, shell_metadata, normalization_totals = collect_normalization_values(
            args,
            lookup_connection,
            sg_source_column,
            sg_source_type,
            resolution_method,
            resolution_source_type,
        )

        if normalization_totals["missing_sg_target"]:
            warnings.append(f"Rows without sg_target: {normalization_totals['missing_sg_target']}")
        if normalization_totals["missing_resolution"]:
            warnings.append(f"Rows without resolution shell metadata: {normalization_totals['missing_resolution']}")

        log("Pass 2/2: writing v3 scores")
        write_totals, diagnostic_rows = write_v3_scores(
            args,
            csv_path,
            lookup_connection,
            sg_source_column,
            sg_source_type,
            resolution_method,
            resolution_source_type,
            global_metadata,
            shell_metadata,
        )

    diagnostics = build_diagnostics(diagnostic_rows, args.v2_risk_column)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_text(),
        "scope": "v3 target-gated score only; v2 inputs are read-only and not modified",
        "inputs": {
            "v2_scores": str(args.v2_scores),
            "base_scores": str(base_scores) if base_scores is not None else None,
            "base_scores_auto_detected_from_v2_metadata": bool(args.base_scores is None and auto_base_scores is not None),
            "stream": str(args.stream) if args.stream is not None else None,
        },
        "outputs": {
            "csv": str(csv_path),
            "metadata": str(metadata_path),
            "diagnostics": str(diagnostics_path),
        },
        "detected_columns": {
            "v2_sg_column": v2_sg_column,
            "base_sg_column": base_sg_column,
            "v2_resolution_method": v2_resolution_method,
            "base_resolution_method": base_resolution_method,
            "raw_v2_column": args.raw_v2_column,
            "v2_risk_column_for_diagnostics": args.v2_risk_column,
        },
        "sources_used": {
            "sg_target": {"source_type": sg_source_type, "column": sg_source_column},
            "resolution": {"source_type": resolution_source_type, "method": resolution_method},
        },
        "equations": {
            "target_excitation_Eg": "exp(-(|sg_target| / target_s0) ** target_power); default power=2 matches v2 excitation_weight_from_sg",
            "target_gate_Gg": "target_gate_floor + (1 - target_gate_floor) * target_excitation_Eg ** target_gate_alpha",
            "manybeam_coupling_v3_target_gated_raw": "target_gate_Gg * raw_v2_column",
            "trust_risk_v3_target_gated_norm": "global robust p01-p99 normalization clipped to [0, 1], same convention as v2",
            "trust_risk_v3_target_gated_shell_norm": "robust p01-p99 normalization within requested fixed 1/nm shells",
        },
        "parameters": {
            "lambda_ang": float(args.lambda_ang),
            "target_s0": float(args.target_s0),
            "target_power": float(args.target_power),
            "target_gate_alpha": float(args.target_gate_alpha),
            "target_gate_floor": float(args.target_gate_floor),
            "chunksize": int(args.chunksize),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
        },
        "fixed_resolution_shells_inv_nm": SHELLS_INV_NM,
        "lookup_stats": lookup_stats,
        "normalization": {
            "global": global_metadata,
            "shells": {str(shell_index): metadata for shell_index, metadata in shell_metadata.items()},
            "pass1_totals": normalization_totals,
        },
        "write_totals": write_totals,
        "warnings": warnings,
        "diagnostics": diagnostics,
    }
    write_json(metadata_path, metadata)
    write_json(diagnostics_path, diagnostics)

    print("\nV3 target-gated scoring complete")
    print(f"output_csv: {csv_path}")
    print(f"metadata_json: {metadata_path}")
    print(f"diagnostics_json: {diagnostics_path}")
    print(f"rows_written: {write_totals['rows_written']:,}")
    print(f"sg_target_source: {sg_source_type} {sg_source_column or ''}".rstrip())
    print(f"resolution_source: {resolution_source_type} {resolution_method or ''}".rstrip())
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print_diagnostics(diagnostics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())