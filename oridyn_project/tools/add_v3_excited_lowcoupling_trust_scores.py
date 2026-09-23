#!/usr/bin/env python3
"""Add excitation-aware low-coupling trust/badness columns to v3 scores.

This is a separate v3-trust experiment. It reads an existing v3 target-gated
score table, preserves all existing columns, and appends trust/badness columns
for filtering. It does not modify v2 or v3 score-generation scripts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oridyn.coupling_exposure_v2 import robust_p01_p99_normalize  # noqa: E402


DEFAULT_OUTPUT_CSV = "geometry_coupling_v3_excited_lowcoupling_scores.csv"
DEFAULT_RISK_COLUMN = "trust_risk_v3_target_gated_shell_norm"
DEFAULT_EG_COLUMN = "target_excitation_Eg"
DEFAULT_CHUNKSIZE = 500_000
SHELL_COLUMN = "resolution_shell_index"
ADDED_COLUMNS = [
    "trust_v3_excited_lowcoupling",
    "badness_v3_excited_lowcoupling",
    "trust_v3_excited_lowcoupling_norm",
    "badness_v3_excited_lowcoupling_norm",
    "trust_v3_excited_lowcoupling_shell_norm",
    "badness_v3_excited_lowcoupling_shell_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-scores", required=True, type=Path, help="Existing v3 target-gated score CSV")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path, or output directory")
    parser.add_argument("--risk-column", default=DEFAULT_RISK_COLUMN)
    parser.add_argument("--eg-column", default=DEFAULT_EG_COLUMN)
    parser.add_argument("--excitation-alpha", type=float, default=1.0)
    parser.add_argument("--coupling-beta", type=float, default=1.0)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.v3_scores.exists():
        raise SystemExit(f"--v3-scores not found: {args.v3_scores}")
    if args.excitation_alpha <= 0.0:
        raise SystemExit("--excitation-alpha must be > 0")
    if args.coupling_beta <= 0.0:
        raise SystemExit("--coupling-beta must be > 0")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def output_csv_path(out: Path) -> Path:
    if out.suffix.lower() == ".csv":
        return out
    return out / DEFAULT_OUTPUT_CSV


def metadata_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_metadata.json")


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def ensure_outputs(csv_path: Path, overwrite: bool) -> Path:
    metadata_path = metadata_path_for(csv_path)
    blocked = [path for path in (csv_path, metadata_path) if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}\nUse --overwrite if intended.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return metadata_path


def require_columns(header: list[str], columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {', '.join(missing)}")


def iter_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=int(chunksize))


def add_raw_trust_columns(chunk: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = chunk.copy()
    eg = pd.to_numeric(out[args.eg_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    risk = pd.to_numeric(out[args.risk_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    eg = eg.clip(lower=0.0, upper=1.0)
    risk = risk.clip(lower=0.0, upper=1.0)
    trust = (eg ** float(args.excitation_alpha)) * ((1.0 - risk) ** float(args.coupling_beta))
    trust = trust.where(np.isfinite(trust), np.nan)
    out["trust_v3_excited_lowcoupling"] = trust.astype(float)
    out["badness_v3_excited_lowcoupling"] = (1.0 - trust).astype(float)
    return out


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy(dtype=float)


def robust_metadata(values: list[np.ndarray], raw_column: str, norm_column: str, scope: str) -> dict[str, Any]:
    series = pd.Series(np.concatenate(values).astype(float)) if values else pd.Series(dtype=float)
    _normalized, metadata = robust_p01_p99_normalize(series)
    metadata.update(
        {
            "raw_column": raw_column,
            "normalized_column": norm_column,
            "scope": scope,
            "method_note": "robust p01-p99 normalization clipped to [0, 1], matching v2 convention",
        }
    )
    return metadata


def collect_normalization_metadata(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[int, dict[str, Any]]], dict[str, int]]:
    trust_values: list[np.ndarray] = []
    badness_values: list[np.ndarray] = []
    shell_trust_values: dict[int, list[np.ndarray]] = {}
    shell_badness_values: dict[int, list[np.ndarray]] = {}
    totals = {
        "rows_read": 0,
        "finite_trust": 0,
        "finite_badness": 0,
        "rows_missing_shell": 0,
    }

    for chunk_index, chunk in enumerate(iter_chunks(args.v3_scores, args.chunksize), start=1):
        work = add_raw_trust_columns(chunk, args)
        totals["rows_read"] += int(len(work))
        trust = finite_values(work["trust_v3_excited_lowcoupling"])
        badness = finite_values(work["badness_v3_excited_lowcoupling"])
        totals["finite_trust"] += int(len(trust))
        totals["finite_badness"] += int(len(badness))
        if len(trust):
            trust_values.append(trust)
        if len(badness):
            badness_values.append(badness)

        if SHELL_COLUMN in work:
            shells = pd.to_numeric(work[SHELL_COLUMN], errors="coerce")
            totals["rows_missing_shell"] += int(shells.isna().sum())
            for shell_index in sorted(shells.dropna().astype(int).unique().tolist()):
                mask = shells.to_numpy(dtype=float) == float(shell_index)
                shell_trust = finite_values(work.loc[mask, "trust_v3_excited_lowcoupling"])
                shell_badness = finite_values(work.loc[mask, "badness_v3_excited_lowcoupling"])
                if len(shell_trust):
                    shell_trust_values.setdefault(int(shell_index), []).append(shell_trust)
                if len(shell_badness):
                    shell_badness_values.setdefault(int(shell_index), []).append(shell_badness)
        else:
            totals["rows_missing_shell"] += int(len(work))

        if chunk_index % 5 == 0:
            log(f"Normalization pass: rows_read={totals['rows_read']:,}, finite_badness={totals['finite_badness']:,}")

    global_trust = robust_metadata(
        trust_values,
        "trust_v3_excited_lowcoupling",
        "trust_v3_excited_lowcoupling_norm",
        "global",
    )
    global_badness = robust_metadata(
        badness_values,
        "badness_v3_excited_lowcoupling",
        "badness_v3_excited_lowcoupling_norm",
        "global",
    )

    shells = {
        "trust": {
            int(shell): robust_metadata(
                values,
                "trust_v3_excited_lowcoupling",
                "trust_v3_excited_lowcoupling_shell_norm",
                f"resolution_shell_{int(shell)}",
            )
            for shell, values in sorted(shell_trust_values.items())
        },
        "badness": {
            int(shell): robust_metadata(
                values,
                "badness_v3_excited_lowcoupling",
                "badness_v3_excited_lowcoupling_shell_norm",
                f"resolution_shell_{int(shell)}",
            )
            for shell, values in sorted(shell_badness_values.items())
        },
    }
    return global_trust, global_badness, shells, totals


def apply_norm(values: pd.Series, metadata: dict[str, Any]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    finite = numeric.notna()
    if not finite.any():
        return out
    if metadata.get("method") == "robust_p01_p99":
        p01 = float(metadata["p01"])
        p99 = float(metadata["p99"])
        out.loc[finite] = ((numeric.loc[finite] - p01) / max(p99 - p01, 1e-12)).clip(lower=0.0, upper=1.0)
    elif metadata.get("method") == "degenerate_p01_p99_to_zero":
        out.loc[finite] = 0.0
    return out


def add_normalized_columns(
    chunk: pd.DataFrame,
    global_trust: dict[str, Any],
    global_badness: dict[str, Any],
    shell_metadata: dict[str, dict[int, dict[str, Any]]],
) -> pd.DataFrame:
    out = chunk.copy()
    out["trust_v3_excited_lowcoupling_norm"] = apply_norm(out["trust_v3_excited_lowcoupling"], global_trust)
    out["badness_v3_excited_lowcoupling_norm"] = apply_norm(out["badness_v3_excited_lowcoupling"], global_badness)
    out["trust_v3_excited_lowcoupling_shell_norm"] = np.nan
    out["badness_v3_excited_lowcoupling_shell_norm"] = np.nan
    if SHELL_COLUMN not in out:
        return out
    shells = pd.to_numeric(out[SHELL_COLUMN], errors="coerce")
    for shell_index, metadata in shell_metadata["trust"].items():
        mask = shells == int(shell_index)
        if mask.any():
            out.loc[mask, "trust_v3_excited_lowcoupling_shell_norm"] = apply_norm(
                out.loc[mask, "trust_v3_excited_lowcoupling"], metadata
            )
    for shell_index, metadata in shell_metadata["badness"].items():
        mask = shells == int(shell_index)
        if mask.any():
            out.loc[mask, "badness_v3_excited_lowcoupling_shell_norm"] = apply_norm(
                out.loc[mask, "badness_v3_excited_lowcoupling"], metadata
            )
    return out


def write_scores(
    args: argparse.Namespace,
    csv_path: Path,
    global_trust: dict[str, Any],
    global_badness: dict[str, Any],
    shell_metadata: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, int]:
    totals = {"rows_written": 0}
    first = True
    for chunk_index, chunk in enumerate(iter_chunks(args.v3_scores, args.chunksize), start=1):
        work = add_raw_trust_columns(chunk, args)
        work = add_normalized_columns(work, global_trust, global_badness, shell_metadata)
        work.to_csv(csv_path, index=False, mode="w" if first else "a", header=first)
        first = False
        totals["rows_written"] += int(len(work))
        if chunk_index % 5 == 0:
            log(f"Write pass: rows_written={totals['rows_written']:,}")
    return totals


def command_text() -> str:
    return shlex.join(sys.argv)


def main() -> int:
    args = parse_args()
    header = read_header(args.v3_scores)
    require_columns(header, [args.eg_column, args.risk_column], "v3 scores CSV")
    csv_path = output_csv_path(args.out)
    metadata_path = ensure_outputs(csv_path, bool(args.overwrite))

    log(f"Input v3 scores: {args.v3_scores}")
    log(f"Output CSV: {csv_path}")
    log(f"Trust formula: ({args.eg_column} ** {args.excitation_alpha}) * ((1 - {args.risk_column}) ** {args.coupling_beta})")
    log("Pass 1/2: collecting normalization metadata")
    global_trust, global_badness, shell_metadata, pass1_totals = collect_normalization_metadata(args)
    log("Pass 2/2: writing enriched score table")
    write_totals = write_scores(args, csv_path, global_trust, global_badness, shell_metadata)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command_text(),
        "scope": "v3 excited-lowcoupling trust/badness experiment; input score files are read-only",
        "inputs": {"v3_scores": str(args.v3_scores)},
        "outputs": {"csv": str(csv_path), "metadata": str(metadata_path)},
        "columns_used": {"target_excitation": args.eg_column, "coupling_risk": args.risk_column},
        "columns_added": ADDED_COLUMNS,
        "formula": {
            "trust_v3_excited_lowcoupling": f"({args.eg_column} ** excitation_alpha) * ((1 - {args.risk_column}) ** coupling_beta)",
            "badness_v3_excited_lowcoupling": "1 - trust_v3_excited_lowcoupling",
            "filtering_interpretation": "lower badness is better; high target excitation and low coupling risk are retained",
        },
        "parameters": {
            "excitation_alpha": float(args.excitation_alpha),
            "coupling_beta": float(args.coupling_beta),
            "chunksize": int(args.chunksize),
        },
        "normalization": {
            "trust_global": global_trust,
            "badness_global": global_badness,
            "shells": {
                "trust": {str(k): v for k, v in shell_metadata["trust"].items()},
                "badness": {str(k): v for k, v in shell_metadata["badness"].items()},
            },
        },
        "pass1_totals": pass1_totals,
        "write_totals": write_totals,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Excited-lowcoupling v3 trust scores written")
    print(f"output_csv: {csv_path}")
    print(f"metadata_json: {metadata_path}")
    print(f"rows_written: {write_totals['rows_written']:,}")
    print("recommended_filter_column: badness_v3_excited_lowcoupling_shell_norm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())