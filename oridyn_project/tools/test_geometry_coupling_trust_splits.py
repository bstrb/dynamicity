#!/usr/bin/env python3
"""Create geometry-coupling trust-risk validation stream splits.

This is an observation-level validation experiment, not a correction model.
It tests whether geometry-derived many-beam coupling proxies identify
observations that are less trustworthy for kinematical merging.

The script keeps/removes stream reflection rows by exact
source_filename + event + signed h,k,l keys. It does not canonicalize
symmetry equivalents and does not modify intensity, sigma, HKL, or metadata.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
PROGRESS_EVERY_REFLECTIONS = 1_000_000
TOP_ROWS = 30

NONSELF_TERMS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

SCORE_MODES = [
    "existing_graph_frame",
    "existing_nonself_mean",
    "equation_proxy_from_existing_terms",
    "flux_balance_geometry",
    "custom_columns",
]

RESOLUTION_BINS = [
    (20.0, 4.0, "20-4A"),
    (4.0, 3.0, "4-3A"),
    (3.0, 2.0, "3-2A"),
    (2.0, 1.5, "2-1.5A"),
    (1.5, 1.09, "1.5-1.09A"),
    (1.09, 0.86, "1.09-0.86A"),
    (0.86, 0.75, "0.86-0.75A"),
    (0.75, 0.68, "0.75-0.68A"),
    (0.68, 0.60, "0.68-0.60A"),
    (0.60, 0.50, "0.60-0.50A"),
    (0.50, 0.40, "0.50-0.40A"),
    (0.40, 0.35, "0.40-0.35A"),
]
RESOLUTION_ORDER = [label for _d_high, _d_low, label in RESOLUTION_BINS] + ["outside"]

STREAM_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_CELL_RE = re.compile(
    rf"^\s*Cell parameters\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+nm,"
    rf"\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+deg"
)
STREAM_UNITCELL_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_UNITCELL_ANGLE_RE = re.compile(rf"^\s*(al|be|ga|alpha|beta|gamma)\s*=\s*({STREAM_FLOAT})\s*deg")

SELECTED_BASE_COLUMNS = [
    "source_filename",
    "event",
    "h",
    "k",
    "l",
    "split",
    "score",
    "trust_risk_raw",
    "trust_risk_norm",
    "d_angstrom",
    "inv_nm",
]

HKL_SUMMARY_COLUMNS = [
    "h",
    "k",
    "l",
    "n_total_eligible",
    "n_selected_each_split",
    "n_dropped_worst",
    "n_random_minus_worst",
    "low_score_min",
    "low_score_median",
    "low_score_max",
    "high_score_min",
    "high_score_median",
    "high_score_max",
    "random_score_median",
    "random_minus_worst_score_median",
    "high_low_median_separation",
]

RESOLUTION_SUMMARY_COLUMNS = [
    "split",
    "resolution_bin",
    "n_observations",
    "n_signed_hkl",
    "median_score",
    "p25_score",
    "p75_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="Original CrystFEL stream")
    parser.add_argument("--scores-csv", required=True, type=Path, help="OriDyn observation-level score table")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-obs-per-hkl", type=int, default=10)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.50,
        help="Per-signed-HKL low/high/random fraction; must satisfy 0 < fraction <= 0.50",
    )
    parser.add_argument(
        "--drop-worst-fraction",
        type=float,
        default=0.20,
        help="Per-signed-HKL high-risk tail removed before random-minus-worst selection",
    )
    parser.add_argument(
        "--score-mode",
        choices=SCORE_MODES,
        default="equation_proxy_from_existing_terms",
    )
    parser.add_argument(
        "--custom-columns",
        default="",
        help="Comma-separated columns for --score-mode custom_columns",
    )
    parser.add_argument(
        "--custom-weights",
        default="",
        help="Comma-separated weights for --score-mode custom_columns; defaults to equal weights",
    )
    parser.add_argument(
        "--write-debug-columns",
        action="store_true",
        help="Include normalized component/debug columns in geometry_trust_selected_keys.csv",
    )
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.scores_csv.exists():
        raise SystemExit(f"--scores-csv not found: {args.scores_csv}")
    if int(args.min_obs_per_hkl) < 2:
        raise SystemExit("--min-obs-per-hkl must be >= 2")
    if not np.isfinite(float(args.fraction)) or not (0.0 < float(args.fraction) <= 0.50):
        raise SystemExit("--fraction must satisfy 0 < fraction <= 0.50")
    if not np.isfinite(float(args.drop_worst_fraction)) or not (0.0 <= float(args.drop_worst_fraction) < 1.0):
        raise SystemExit("--drop-worst-fraction must satisfy 0 <= drop-worst-fraction < 1")
    if float(args.fraction) + float(args.drop_worst_fraction) > 1.0:
        raise SystemExit("--fraction + --drop-worst-fraction must be <= 1 so random-minus-worst can keep the same count")
    if args.score_mode == "custom_columns" and not parse_csv_list(args.custom_columns):
        raise SystemExit("--custom-columns is required when --score-mode custom_columns")
    return args


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_custom_weights(text: str, n_columns: int) -> list[float]:
    if not str(text).strip():
        return [1.0 / float(n_columns)] * n_columns
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(values) != n_columns:
        raise SystemExit("--custom-weights must have the same length as --custom-columns")
    total = float(sum(values))
    if not np.isfinite(total) or total == 0.0:
        raise SystemExit("--custom-weights must sum to a nonzero finite value")
    return [value / total for value in values]


def percent_label(fraction: float) -> int:
    return int(round(float(fraction) * 100.0))


def split_labels_for_args(args: argparse.Namespace) -> dict[str, str]:
    keep_pct = percent_label(float(args.fraction))
    drop_pct = percent_label(float(args.drop_worst_fraction))
    mode = str(args.score_mode)
    return {
        "low": f"low_{mode}_{keep_pct}",
        "high": f"high_{mode}_{keep_pct}",
        "random": f"random_{mode}_{keep_pct}",
        "random_minus_worst": f"random_minus_worst_{mode}_drop{drop_pct}_keep{keep_pct}",
    }


def stream_paths_for_labels(outdir: Path, split_labels: dict[str, str], seed: int) -> dict[str, Path]:
    return {
        split_labels["low"]: outdir / f"{split_labels['low']}.stream",
        split_labels["high"]: outdir / f"{split_labels['high']}.stream",
        split_labels["random"]: outdir / f"{split_labels['random']}_seed{int(seed)}.stream",
        split_labels["random_minus_worst"]: outdir / f"{split_labels['random_minus_worst']}_seed{int(seed)}.stream",
    }


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


def build_key(source: Any, event: Any, h: int, k: int, l: int) -> tuple[str, str, int, int, int]:
    return normalize_source(source), normalize_event(event), int(h), int(k), int(l)


def require_columns(columns: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


def needed_score_columns(header: list[str], args: argparse.Namespace) -> list[str]:
    require_columns(header, KEY_COLUMNS, "scores CSV")
    mode = str(args.score_mode)
    if mode == "flux_balance_geometry":
        raise SystemExit(
            "--score-mode flux_balance_geometry is intentionally not faked here. "
            "It requires per-frame candidate-beam geometry: orientation matrices, candidate HKLs, "
            "excitation errors, reciprocal vectors, and a coupling kernel. Generate those columns "
            "or implement an explicit candidate-beam graph builder before using this mode."
        )
    if mode == "existing_graph_frame":
        required = ["graph_crowding_norm", "frame_axis_risk_norm"]
        require_columns(header, required, "scores CSV")
        return required
    if mode == "existing_nonself_mean":
        available = [column for column in NONSELF_TERMS if column in header]
        if not available:
            raise SystemExit(f"--score-mode existing_nonself_mean needs at least one of: {NONSELF_TERMS}")
        return available
    if mode == "equation_proxy_from_existing_terms":
        require_columns(header, ["graph_crowding_norm"], "scores CSV")
        return [column for column in NONSELF_TERMS if column in header]
    if mode == "custom_columns":
        custom = parse_csv_list(args.custom_columns)
        require_columns(header, custom, "scores CSV")
        return custom
    raise AssertionError(f"Unhandled score mode: {mode}")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def series_stats(series: pd.Series) -> dict[str, float | int]:
    values = numeric_series(series).dropna()
    if values.empty:
        return {
            "finite_count": 0,
            "min": np.nan,
            "p01": np.nan,
            "median": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "finite_count": int(len(values)),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "median": float(values.median()),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def normalize_component(
    table: pd.DataFrame,
    source_column: str,
    output_column: str,
    role: str,
    component_rows: list[dict[str, Any]],
    weight: float | None = None,
    note: str = "",
) -> pd.Series:
    raw = numeric_series(table[source_column])
    stats = series_stats(raw)
    values = raw.dropna()
    if values.empty:
        norm = pd.Series(np.nan, index=table.index, dtype=float)
        method = "no_finite_values"
    elif float(values.min()) >= -1e-12 and float(values.max()) <= 1.0 + 1e-12:
        norm = raw.astype(float)
        method = "already_0_1"
    else:
        p01 = float(values.quantile(0.01))
        p99 = float(values.quantile(0.99))
        if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
            norm = pd.Series(np.nan, index=table.index, dtype=float)
            norm.loc[raw.notna()] = 0.0
            method = "degenerate_p01_p99_to_zero"
        else:
            norm = ((raw - p01) / (p99 - p01)).clip(lower=0.0, upper=1.0)
            method = "robust_p01_p99"
    table[output_column] = norm
    norm_stats = series_stats(norm)
    component_rows.append(
        {
            "output_column": output_column,
            "source_column": source_column,
            "role": role,
            "weight": np.nan if weight is None else float(weight),
            "normalization_method": method,
            "raw_finite_count": stats["finite_count"],
            "raw_min": stats["min"],
            "raw_p01": stats["p01"],
            "raw_median": stats["median"],
            "raw_p99": stats["p99"],
            "raw_max": stats["max"],
            "norm_min": norm_stats["min"],
            "norm_median": norm_stats["median"],
            "norm_max": norm_stats["max"],
            "note": note,
        }
    )
    return norm


def add_missing_zero_component(
    table: pd.DataFrame,
    output_column: str,
    role: str,
    component_rows: list[dict[str, Any]],
    note: str,
) -> pd.Series:
    values = pd.Series(0.0, index=table.index, dtype=float)
    table[output_column] = values
    component_rows.append(
        {
            "output_column": output_column,
            "source_column": "",
            "role": role,
            "weight": np.nan,
            "normalization_method": "missing_default_zero",
            "raw_finite_count": 0,
            "raw_min": np.nan,
            "raw_p01": np.nan,
            "raw_median": np.nan,
            "raw_p99": np.nan,
            "raw_max": np.nan,
            "norm_min": 0.0,
            "norm_median": 0.0,
            "norm_max": 0.0,
            "note": note,
        }
    )
    return values


def normalize_final_score(
    table: pd.DataFrame,
    raw_column: str,
    output_column: str,
    component_rows: list[dict[str, Any]],
    note: str,
) -> pd.Series:
    return normalize_component(
        table,
        source_column=raw_column,
        output_column=output_column,
        role="final_trust_risk_score",
        component_rows=component_rows,
        note=note,
    )


def compute_trust_risk(
    table: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    mode = str(args.score_mode)
    work = table.copy()
    component_rows: list[dict[str, Any]] = []
    debug_columns: list[str] = []
    formula = ""
    mapping: dict[str, str] = {}

    if mode == "existing_graph_frame":
        graph = normalize_component(
            work,
            "graph_crowding_norm",
            "graph_crowding_norm_term",
            "baseline_network_density",
            component_rows,
            weight=0.5,
        )
        frame = normalize_component(
            work,
            "frame_axis_risk_norm",
            "frame_axis_risk_norm_term",
            "baseline_zone_axis_risk",
            component_rows,
            weight=0.5,
        )
        work["trust_risk_raw"] = 0.5 * graph + 0.5 * frame
        formula = "risk = 0.5 * graph_crowding_norm + 0.5 * frame_axis_risk_norm"
        mapping = {
            "graph_crowding_norm": "network/crowding baseline",
            "frame_axis_risk_norm": "zone-axis/frame baseline",
        }
        if args.write_debug_columns:
            debug_columns.extend(["graph_crowding_norm_term", "frame_axis_risk_norm_term"])

    elif mode == "existing_nonself_mean":
        available = [column for column in NONSELF_TERMS if column in work.columns]
        terms = []
        for column in available:
            output = f"{column}_term"
            terms.append(
                normalize_component(
                    work,
                    column,
                    output,
                    "empirical_nonself_component",
                    component_rows,
                    weight=1.0 / len(available),
                )
            )
            if args.write_debug_columns:
                debug_columns.append(output)
        work["trust_risk_raw"] = pd.concat(terms, axis=1).mean(axis=1, skipna=True)
        formula = "risk = mean(available non-self terms)"
        mapping = {column: "empirical non-self geometry/crowding term" for column in available}

    elif mode == "equation_proxy_from_existing_terms":
        graph = normalize_component(
            work,
            "graph_crowding_norm",
            "network_density_term",
            "E-network density",
            component_rows,
        )
        if "same_laue_zone_crowding_norm" in work.columns:
            same_zone = normalize_component(
                work,
                "same_laue_zone_crowding_norm",
                "same_laue_zone_term",
                "same-manifold factor",
                component_rows,
            )
        else:
            same_zone = add_missing_zero_component(
                work,
                "same_laue_zone_term",
                "same-manifold factor",
                component_rows,
                "Column absent; same-zone multiplier defaults to 1.",
            )
        if "systematic_row_risk_norm" in work.columns:
            row = normalize_component(
                work,
                "systematic_row_risk_norm",
                "systematic_row_term",
                "row-channel factor",
                component_rows,
            )
        else:
            row = add_missing_zero_component(
                work,
                "systematic_row_term",
                "row-channel factor",
                component_rows,
                "Column absent; row multiplier defaults to 1.",
            )
        if "frame_axis_risk_norm" in work.columns:
            frame = normalize_component(
                work,
                "frame_axis_risk_norm",
                "frame_axis_term",
                "zone-axis population",
                component_rows,
            )
        else:
            frame = add_missing_zero_component(
                work,
                "frame_axis_term",
                "zone-axis population",
                component_rows,
                "Column absent; frame-axis multiplier defaults to 1.",
            )

        work["same_laue_zone_factor"] = 1.0 + 0.5 * same_zone
        work["row_factor"] = 1.0 + 0.5 * row
        work["zone_axis_factor"] = 1.0 + 0.5 * frame
        work["manybeam_exposure_proxy"] = (
            graph * work["same_laue_zone_factor"] * work["row_factor"] * work["zone_axis_factor"]
        )
        work["trust_risk_raw"] = np.log1p(work["manybeam_exposure_proxy"])
        formula = (
            "manybeam_exposure_proxy = graph_crowding_norm * "
            "(1 + 0.5 * same_laue_zone_crowding_norm) * "
            "(1 + 0.5 * systematic_row_risk_norm) * "
            "(1 + 0.5 * frame_axis_risk_norm); trust_risk = log(1 + manybeam_exposure_proxy)"
        )
        mapping = {
            "graph_crowding_norm": "E-network density",
            "same_laue_zone_crowding_norm": "same-manifold amplification",
            "systematic_row_risk_norm": "systematic-row channel amplification",
            "frame_axis_risk_norm": "zone-axis population amplification",
        }
        debug_columns.extend(
            [
                "network_density_term",
                "same_laue_zone_factor",
                "row_factor",
                "zone_axis_factor",
                "manybeam_exposure_proxy",
            ]
        )

    elif mode == "custom_columns":
        custom_columns = parse_csv_list(args.custom_columns)
        weights = parse_custom_weights(args.custom_weights, len(custom_columns))
        terms = []
        for column, weight in zip(custom_columns, weights, strict=True):
            output = f"{column}_term"
            terms.append(
                normalize_component(
                    work,
                    column,
                    output,
                    "custom_empirical_component",
                    component_rows,
                    weight=weight,
                    note="Custom empirical weighted mean; not a physical flux model.",
                )
                * weight
            )
            if args.write_debug_columns:
                debug_columns.append(output)
        work["trust_risk_raw"] = pd.concat(terms, axis=1).sum(axis=1, skipna=False)
        formula = "risk = weighted mean of normalized custom columns"
        mapping = {column: f"custom empirical column, normalized weight {weight:.6g}" for column, weight in zip(custom_columns, weights, strict=True)}

    else:
        raise AssertionError(f"Unhandled score mode after validation: {mode}")

    work["trust_risk_raw"] = numeric_series(work["trust_risk_raw"])
    normalize_final_score(
        work,
        "trust_risk_raw",
        "trust_risk_norm",
        component_rows,
        "Final score used for low/high/random split selection.",
    )
    work["score"] = work["trust_risk_norm"]
    debug_columns.extend(["trust_risk_raw", "trust_risk_norm"])
    debug_columns = list(dict.fromkeys(debug_columns))
    component_summary = pd.DataFrame.from_records(component_rows)
    score_info = {
        "formula": formula,
        "mapping": mapping,
        "debug_columns": debug_columns,
    }
    return work, component_summary, debug_columns, score_info


def load_scores(path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], dict[str, int]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    score_columns = needed_score_columns(header, args)
    usecols = list(dict.fromkeys([*KEY_COLUMNS, *score_columns]))
    raw = pd.read_csv(path, usecols=usecols)
    stats: dict[str, int] = {"score_rows_loaded": int(len(raw))}
    raw = raw.copy()
    raw["source_filename"] = raw["source_filename"].map(normalize_source)
    raw["event"] = raw["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    bad_hkl = raw[HKL_COLUMNS].isna().any(axis=1)
    if bool(bad_hkl.any()):
        log(f"Dropping {int(bad_hkl.sum()):,} score rows with non-numeric HKLs")
    raw = raw.loc[~bad_hkl].copy()
    raw[HKL_COLUMNS] = raw[HKL_COLUMNS].astype("int64")

    scored, component_summary, debug_columns, score_info = compute_trust_risk(raw, args)
    finite_score = scored["score"].map(np.isfinite)
    eligible = scored.loc[finite_score].copy()
    duplicated = eligible.duplicated(KEY_COLUMNS, keep=False)
    stats.update(
        {
            "score_rows_after_hkl_cleanup": int(len(scored)),
            "eligible_finite_score_rows": int(len(eligible)),
            "duplicate_eligible_key_rows": int(duplicated.sum()),
            "duplicate_eligible_keys": int(eligible.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]),
        }
    )
    if duplicated.any():
        log(
            "Warning: duplicate exact observation keys found in score table; "
            "keeping the first row per key for split selection."
        )
        eligible = eligible.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    stats["eligible_unique_key_rows"] = int(len(eligible))
    return eligible, component_summary, debug_columns, score_info, stats


def load_unit_cell_from_stream(stream_path: Path) -> dict[str, float]:
    """Parse a stream-level or first-crystal unit cell in Angstrom/degrees."""
    cell: dict[str, float] = {}
    in_unit_cell = False
    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "Begin unit cell" in raw_line:
                in_unit_cell = True
                continue
            if in_unit_cell and "End unit cell" in raw_line:
                in_unit_cell = False
                if has_complete_cell(cell):
                    return cell
                continue
            if match := STREAM_CELL_RE.match(raw_line):
                a_nm, b_nm, c_nm, alpha, beta, gamma = (float(match.group(i)) for i in range(1, 7))
                return {
                    "a": 10.0 * a_nm,
                    "b": 10.0 * b_nm,
                    "c": 10.0 * c_nm,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                }
            if in_unit_cell or not line.startswith("----- Begin chunk -----"):
                if match := STREAM_UNITCELL_LENGTH_RE.match(raw_line):
                    cell[match.group(1).lower()] = float(match.group(2))
                    if has_complete_cell(cell):
                        return cell
                    continue
                if match := STREAM_UNITCELL_ANGLE_RE.match(raw_line):
                    key = {"al": "alpha", "be": "beta", "ga": "gamma"}.get(
                        match.group(1).lower(),
                        match.group(1).lower(),
                    )
                    cell[key] = float(match.group(2))
                    if has_complete_cell(cell):
                        return cell
                    continue
    missing = [key for key in ["a", "b", "c", "alpha", "beta", "gamma"] if key not in cell]
    raise SystemExit(f"Could not parse unit cell from stream; missing {missing}")


def has_complete_cell(cell: dict[str, float]) -> bool:
    return all(key in cell for key in ["a", "b", "c", "alpha", "beta", "gamma"])


def reciprocal_basis_from_cell(cell: dict[str, float]) -> np.ndarray:
    """Return reciprocal basis matrix with columns a*, b*, c* in 1/Angstrom."""
    a = float(cell["a"])
    b = float(cell["b"])
    c = float(cell["c"])
    alpha = math.radians(float(cell["alpha"]))
    beta = math.radians(float(cell["beta"]))
    gamma = math.radians(float(cell["gamma"]))
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-12:
        raise SystemExit("Invalid unit cell: sin(gamma) is too small")
    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b * cos_g, b * sin_g, 0.0], dtype=float)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        raise SystemExit("Invalid unit cell: computed c_z^2 <= 0")
    c_vec = np.array([cx, cy, math.sqrt(cz_sq)], dtype=float)
    volume = float(np.dot(a_vec, np.cross(b_vec, c_vec)))
    if abs(volume) < 1e-15:
        raise SystemExit("Invalid unit cell: near-zero volume")
    a_star = np.cross(b_vec, c_vec) / volume
    b_star = np.cross(c_vec, a_vec) / volume
    c_star = np.cross(a_vec, b_vec) / volume
    return np.column_stack([a_star, b_star, c_star])


def d_spacing_angstrom(h: int, k: int, l: int, reciprocal_basis: np.ndarray) -> float:
    hkl = np.array([int(h), int(k), int(l)], dtype=float)
    g_vec = np.asarray(reciprocal_basis, dtype=float) @ hkl
    g_norm = float(np.linalg.norm(g_vec))
    if not np.isfinite(g_norm) or g_norm <= 0.0:
        return np.nan
    return 1.0 / g_norm


def resolution_bin_for_d(d_angstrom: float) -> str:
    if not np.isfinite(d_angstrom):
        return "outside"
    for idx, (d_high, d_low, label) in enumerate(RESOLUTION_BINS):
        if idx == 0:
            if d_low <= d_angstrom <= d_high:
                return label
        elif idx == len(RESOLUTION_BINS) - 1:
            if d_low <= d_angstrom < d_high:
                return label
        elif d_low <= d_angstrom < d_high:
            return label
    return "outside"


def add_resolution_columns(table: pd.DataFrame, cell: dict[str, float]) -> pd.DataFrame:
    out = table.copy()
    basis = reciprocal_basis_from_cell(cell)
    d_by_hkl: dict[tuple[int, int, int], tuple[float, float, str]] = {}
    for h, k, l in out[HKL_COLUMNS].drop_duplicates().itertuples(index=False, name=None):
        d_value = d_spacing_angstrom(int(h), int(k), int(l), basis)
        inv_nm = 10.0 / d_value if np.isfinite(d_value) and d_value > 0.0 else np.nan
        d_by_hkl[(int(h), int(k), int(l))] = (d_value, inv_nm, resolution_bin_for_d(d_value))
    d_values = []
    inv_values = []
    bins = []
    for h, k, l in out[HKL_COLUMNS].itertuples(index=False, name=None):
        d_value, inv_nm, label = d_by_hkl[(int(h), int(k), int(l))]
        d_values.append(d_value)
        inv_values.append(inv_nm)
        bins.append(label)
    out["d_angstrom"] = d_values
    out["inv_nm"] = inv_values
    out["resolution_bin"] = bins
    return out


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def quantile_or_nan(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def selected_frame_for_split(group: pd.DataFrame, split: str, selected_columns: list[str]) -> pd.DataFrame:
    columns_without_split = [column for column in selected_columns if column != "split"]
    out = group.loc[:, columns_without_split].copy()
    out.insert(5, "split", split)
    return out.loc[:, selected_columns]


def key_set_from_frame(frame: pd.DataFrame) -> set[tuple[str, str, int, int, int]]:
    if frame.empty:
        return set()
    keys = frame.loc[:, KEY_COLUMNS].itertuples(index=False, name=None)
    return {(str(source), str(event), int(h), int(k), int(l)) for source, event, h, k, l in keys}


def split_score_stats(group: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame(
            columns=[
                f"{prefix}_score_min",
                f"{prefix}_score_median",
                f"{prefix}_score_max",
            ]
        )
    stats = group.groupby(HKL_COLUMNS, sort=True)["score"].agg(["min", "median", "max"])
    return stats.rename(
        columns={
            "min": f"{prefix}_score_min",
            "median": f"{prefix}_score_median",
            "max": f"{prefix}_score_max",
        }
    )


def split_score_median(group: pd.DataFrame, output_column: str) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame(columns=[output_column])
    stats = group.groupby(HKL_COLUMNS, sort=True)["score"].median().to_frame(output_column)
    return stats


def select_split_keys(
    eligible: pd.DataFrame,
    args: argparse.Namespace,
    split_labels: dict[str, str],
    debug_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[tuple[str, str, int, int, int]]], dict[str, int]]:
    rng = np.random.default_rng(int(args.seed))
    selected_columns = [*SELECTED_BASE_COLUMNS, *[column for column in debug_columns if column not in SELECTED_BASE_COLUMNS]]

    log("Vectorized selection: sorting observations by signed HKL and risk score")
    work = eligible.sort_values([*HKL_COLUMNS, "score", "source_filename", "event"], kind="mergesort").reset_index(drop=True)
    grouped = work.groupby(HKL_COLUMNS, sort=False)
    work["_n_total"] = grouped["score"].transform("size").astype("int64")
    work["_score_rank"] = grouped.cumcount().astype("int64")
    work["_n_select"] = np.floor(work["_n_total"].to_numpy(dtype=float) * float(args.fraction)).astype("int64")
    work["_n_drop"] = np.floor(work["_n_total"].to_numpy(dtype=float) * float(args.drop_worst_fraction)).astype("int64")
    work["_passing"] = (
        (work["_n_total"] >= int(args.min_obs_per_hkl))
        & (work["_n_select"] >= 1)
        & ((work["_n_total"] - work["_n_drop"]) >= work["_n_select"])
    )

    passing_hkls = work.loc[work["_passing"], [*HKL_COLUMNS, "_n_total", "_n_select", "_n_drop"]].drop_duplicates(HKL_COLUMNS)
    n_hkls_seen = int(work[HKL_COLUMNS].drop_duplicates().shape[0])
    n_hkls_passing = int(len(passing_hkls))
    n_observations_in_passing_hkls = int(passing_hkls["_n_total"].sum()) if not passing_hkls.empty else 0

    low_mask = work["_passing"] & (work["_score_rank"] < work["_n_select"])
    high_mask = work["_passing"] & (work["_score_rank"] >= (work["_n_total"] - work["_n_select"]))
    low = work.loc[low_mask].copy()
    high = work.loc[high_mask].copy()

    log("Vectorized selection: drawing random matched controls")
    random_temp = work.loc[work["_passing"], [*HKL_COLUMNS, "_n_select"]].copy()
    random_temp["_rand"] = rng.random(len(random_temp))
    random_temp = random_temp.sort_values([*HKL_COLUMNS, "_rand"], kind="mergesort")
    random_temp["_random_rank"] = random_temp.groupby(HKL_COLUMNS, sort=False).cumcount().astype("int64")
    random_index = random_temp.index[random_temp["_random_rank"] < random_temp["_n_select"]]
    random_group = work.loc[random_index].sort_values([*HKL_COLUMNS, "source_filename", "event"], kind="mergesort").copy()

    log("Vectorized selection: drawing random-minus-worst matched controls")
    remaining_mask = work["_passing"] & (work["_score_rank"] < (work["_n_total"] - work["_n_drop"]))
    minus_temp = work.loc[remaining_mask, [*HKL_COLUMNS, "_n_select"]].copy()
    minus_temp["_rand"] = rng.random(len(minus_temp))
    minus_temp = minus_temp.sort_values([*HKL_COLUMNS, "_rand"], kind="mergesort")
    minus_temp["_minus_rank"] = minus_temp.groupby(HKL_COLUMNS, sort=False).cumcount().astype("int64")
    minus_index = minus_temp.index[minus_temp["_minus_rank"] < minus_temp["_n_select"]]
    minus_group = work.loc[minus_index].sort_values([*HKL_COLUMNS, "source_filename", "event"], kind="mergesort").copy()

    log("Vectorized selection: assembling selected-key table")
    selected = pd.concat(
        [
            selected_frame_for_split(low, split_labels["low"], selected_columns),
            selected_frame_for_split(high, split_labels["high"], selected_columns),
            selected_frame_for_split(random_group, split_labels["random"], selected_columns),
            selected_frame_for_split(minus_group, split_labels["random_minus_worst"], selected_columns),
        ],
        ignore_index=True,
        copy=False,
    )

    log("Vectorized selection: assembling HKL summary")
    hkl_summary = passing_hkls.set_index(HKL_COLUMNS).rename(
        columns={
            "_n_total": "n_total_eligible",
            "_n_select": "n_selected_each_split",
            "_n_drop": "n_dropped_worst",
        }
    )
    minus_counts = minus_group.groupby(HKL_COLUMNS, sort=True).size().to_frame("n_random_minus_worst")
    hkl_summary = hkl_summary.join(minus_counts, how="left")
    hkl_summary["n_random_minus_worst"] = hkl_summary["n_random_minus_worst"].fillna(0).astype("int64")
    hkl_summary = hkl_summary.join(split_score_stats(low, "low"), how="left")
    hkl_summary = hkl_summary.join(split_score_stats(high, "high"), how="left")
    hkl_summary = hkl_summary.join(split_score_median(random_group, "random_score_median"), how="left")
    hkl_summary = hkl_summary.join(split_score_median(minus_group, "random_minus_worst_score_median"), how="left")
    hkl_summary["high_low_median_separation"] = hkl_summary["high_score_median"] - hkl_summary["low_score_median"]
    hkl_summary = hkl_summary.reset_index().loc[:, HKL_SUMMARY_COLUMNS]

    log("Vectorized selection: building stream membership key sets")
    split_keys = {
        split_labels["low"]: key_set_from_frame(low),
        split_labels["high"]: key_set_from_frame(high),
        split_labels["random"]: key_set_from_frame(random_group),
        split_labels["random_minus_worst"]: key_set_from_frame(minus_group),
    }

    stats = {
        "signed_hkls_seen": int(n_hkls_seen),
        "signed_hkls_passing_min_observations": int(n_hkls_passing),
        "eligible_observations_in_passing_hkls": int(n_observations_in_passing_hkls),
        "selected_low_keys": int(len(split_keys[split_labels["low"]])),
        "selected_high_keys": int(len(split_keys[split_labels["high"]])),
        "selected_random_keys": int(len(split_keys[split_labels["random"]])),
        "selected_random_minus_worst_keys": int(len(split_keys[split_labels["random_minus_worst"]])),
    }
    return selected, hkl_summary, split_keys, stats


def summarize_resolution(selected: pd.DataFrame, split_labels: dict[str, str]) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=RESOLUTION_SUMMARY_COLUMNS)
    work = selected.copy()
    if "resolution_bin" not in work.columns:
        work["resolution_bin"] = [resolution_bin_for_d(float(value)) for value in work["d_angstrom"]]
    rows: list[dict[str, Any]] = []
    for split in split_labels.values():
        split_group = work.loc[work["split"] == split]
        for label in RESOLUTION_ORDER:
            group = split_group.loc[split_group["resolution_bin"] == label]
            rows.append(
                {
                    "split": split,
                    "resolution_bin": label,
                    "n_observations": int(len(group)),
                    "n_signed_hkl": int(group[HKL_COLUMNS].drop_duplicates().shape[0]) if not group.empty else 0,
                    "median_score": median_or_nan(group["score"]) if not group.empty else np.nan,
                    "p25_score": quantile_or_nan(group["score"], 0.25) if not group.empty else np.nan,
                    "p75_score": quantile_or_nan(group["score"], 0.75) if not group.empty else np.nan,
                }
            )
    return pd.DataFrame.from_records(rows, columns=RESOLUTION_SUMMARY_COLUMNS)


def parse_reflection_hkl(line: str) -> tuple[int, int, int] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return None
    return h, k, l


def write_split_streams(
    stream_path: Path,
    paths: dict[str, Path],
    split_keys: dict[str, set[tuple[str, str, int, int, int]]],
    split_labels: dict[str, str],
) -> dict[str, Any]:
    handles = {split: path.open("w", encoding="utf-8") for split, path in paths.items()}
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    chunks_seen = 0
    crystals_seen = 0
    stream_observations_seen = 0
    matched_to_any_selected = 0
    kept_counts = {split: 0 for split in split_labels.values()}

    try:
        with stream_path.open("r", encoding="utf-8", errors="replace") as source:
            for raw_line in source:
                line = raw_line.rstrip("\n")

                if "Begin chunk" in line:
                    chunks_seen += 1
                    chunk_source = ""
                    chunk_event = ""
                    current_source = ""
                    current_event = ""
                    in_crystal = False
                    in_reflections = False
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                image_match = STREAM_IMAGE_RE.match(line)
                if image_match:
                    if in_crystal:
                        current_source = normalize_source(image_match.group(1))
                    else:
                        chunk_source = normalize_source(image_match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                event_match = STREAM_EVENT_RE.match(line)
                if event_match:
                    if in_crystal:
                        current_event = normalize_event(event_match.group(1))
                    else:
                        chunk_event = normalize_event(event_match.group(1))
                    for handle in handles.values():
                        handle.write(raw_line)
                    continue

                if "Begin crystal" in line:
                    crystals_seen += 1
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

                if in_crystal and in_reflections:
                    hkl = parse_reflection_hkl(line)
                    if hkl is not None:
                        stream_observations_seen += 1
                        if stream_observations_seen % PROGRESS_EVERY_REFLECTIONS == 0:
                            log(
                                "Stream write progress: "
                                f"reflections={stream_observations_seen:,}, "
                                f"low={kept_counts[split_labels['low']]:,}, "
                                f"high={kept_counts[split_labels['high']]:,}, "
                                f"random={kept_counts[split_labels['random']]:,}, "
                                f"random_minus_worst={kept_counts[split_labels['random_minus_worst']]:,}"
                            )
                        key = build_key(current_source, current_event, *hkl)
                        if any(key in keys for keys in split_keys.values()):
                            matched_to_any_selected += 1
                        for split, keys in split_keys.items():
                            if key in keys:
                                handles[split].write(raw_line)
                                kept_counts[split] += 1
                        continue

                for handle in handles.values():
                    handle.write(raw_line)
    finally:
        for handle in handles.values():
            handle.close()

    stats: dict[str, Any] = {
        "chunks_seen": int(chunks_seen),
        "crystals_seen": int(crystals_seen),
        "stream_observations_seen": int(stream_observations_seen),
        "stream_observations_matching_any_selected_key": int(matched_to_any_selected),
    }
    for split in split_labels.values():
        stats[f"{split}_stream_observations_kept"] = int(kept_counts[split])
        stats[f"{split}_fraction_original_stream_observations_kept"] = float(
            kept_counts[split] / max(stream_observations_seen, 1)
        )
    return stats


def score_summary_by_split(selected: pd.DataFrame, split_labels: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in split_labels.values():
        group = selected.loc[selected["split"] == split]
        rows.append(
            {
                "split": split,
                "n_observations": int(len(group)),
                "score_min": min_or_nan(group["score"]),
                "score_p25": quantile_or_nan(group["score"], 0.25),
                "score_median": median_or_nan(group["score"]),
                "score_p75": quantile_or_nan(group["score"], 0.75),
                "score_max": max_or_nan(group["score"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def markdown_table(table: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 20) -> str:
    if table is None or table.empty:
        return "_No rows._"
    if columns is None:
        columns = table.columns.tolist()
    view = table.loc[:, [column for column in columns if column in table.columns]].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def mapping_lines(mapping: dict[str, str]) -> list[str]:
    if not mapping:
        return ["- No column mapping recorded."]
    return [f"- `{column}`: {meaning}" for column, meaning in mapping.items()]


def write_summary(
    outdir: Path,
    args: argparse.Namespace,
    load_stats: dict[str, int],
    selection_stats: dict[str, int],
    stream_stats: dict[str, Any],
    selected: pd.DataFrame,
    hkl_summary: pd.DataFrame,
    resolution_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    paths: dict[str, Path],
    split_labels: dict[str, str],
    score_info: dict[str, Any],
) -> None:
    split_scores = score_summary_by_split(selected, split_labels)
    top_separation = (
        hkl_summary.sort_values("high_low_median_separation", ascending=False).head(TOP_ROWS)
        if not hkl_summary.empty
        else hkl_summary
    )
    top_dropped = (
        hkl_summary.sort_values(["n_dropped_worst", "high_score_median"], ascending=[False, False]).head(TOP_ROWS)
        if not hkl_summary.empty
        else hkl_summary
    )
    low_label = split_labels["low"]
    high_label = split_labels["high"]
    random_label = split_labels["random"]
    minus_label = split_labels["random_minus_worst"]
    lines = [
        "# Geometry-Coupling Trust-Risk Stream Split",
        "",
        "## Scientific Warning",
        "",
        "- This tests a geometry-coupling approximation to dynamical reliability, not exact dynamical intensities.",
        "- This is an observation-level split, not an intensity correction.",
        "- Stream reflection intensities, sigmas, HKL indices, and non-reflection text are not modified.",
        "- Observations not selected from the score table are removed from all split streams.",
        "- The key presentation test is whether random-minus-worst improves over pure random, not only whether low-risk beats high-risk.",
        "",
        "## Physical Framing",
        "",
        "Simplified coupled-beam amplitude equation:",
        "",
        "```text",
        "dA_g/dz = i * 2*pi * s_g(Omega) * A_g + i * C * sum_h U_(g-h) * A_h",
        "```",
        "",
        "This script does not solve that equation. It computes a trust-risk proxy from existing observation-level geometry terms.",
        "",
        "Generic coupling opportunity:",
        "",
        "```text",
        "K(a -> b | Omega) = E_a(Omega) * E_b(Omega) * V(b-a) * Z(a,b,Omega)",
        "```",
        "",
        "For filtering/splitting trustworthy observations, this experiment uses a nonnegative trust-risk score, not signed flux balance.",
        "",
        "## Score Formula Used",
        "",
        f"- Score mode: `{args.score_mode}`",
        f"- Formula: `{score_info.get('formula', '')}`",
        "",
        "## Column-To-Equation Mapping",
        "",
        *mapping_lines(score_info.get("mapping", {})),
        "",
        "## Inputs",
        "",
        f"- Stream: `{args.stream}`",
        f"- Scores CSV: `{args.scores_csv}`",
        f"- Seed: `{int(args.seed)}`",
        f"- min_obs_per_hkl: `{int(args.min_obs_per_hkl)}`",
        f"- Fraction selected: `{float(args.fraction):.6g}`",
        f"- Drop worst fraction: `{float(args.drop_worst_fraction):.6g}`",
        "",
        "## Counts",
        "",
        f"- Rows loaded: {load_stats['score_rows_loaded']}",
        f"- Eligible finite-score rows: {load_stats['eligible_finite_score_rows']}",
        f"- Unique eligible observation keys: {load_stats['eligible_unique_key_rows']}",
        f"- Signed HKLs passing min obs: {selection_stats['signed_hkls_passing_min_observations']}",
        f"- Eligible observations in passing HKLs: {selection_stats['eligible_observations_in_passing_hkls']}",
        f"- Original stream observations seen: {stream_stats['stream_observations_seen']}",
        f"- Low observations kept: {stream_stats[f'{low_label}_stream_observations_kept']}",
        f"- High observations kept: {stream_stats[f'{high_label}_stream_observations_kept']}",
        f"- Random observations kept: {stream_stats[f'{random_label}_stream_observations_kept']}",
        f"- Random-minus-worst observations kept: {stream_stats[f'{minus_label}_stream_observations_kept']}",
        f"- Low fraction of original stream observations kept: {stream_stats[f'{low_label}_fraction_original_stream_observations_kept']:.6g}",
        f"- High fraction of original stream observations kept: {stream_stats[f'{high_label}_fraction_original_stream_observations_kept']:.6g}",
        f"- Random fraction of original stream observations kept: {stream_stats[f'{random_label}_fraction_original_stream_observations_kept']:.6g}",
        f"- Random-minus-worst fraction of original stream observations kept: {stream_stats[f'{minus_label}_fraction_original_stream_observations_kept']:.6g}",
        "",
        "## Score Summaries",
        "",
        markdown_table(split_scores, max_rows=len(split_scores)),
        "",
        "## Resolution Summaries",
        "",
        markdown_table(resolution_summary, max_rows=len(resolution_summary)),
        "",
        "## Component Normalization Summary",
        "",
        markdown_table(
            component_summary,
            [
                "output_column",
                "source_column",
                "role",
                "weight",
                "normalization_method",
                "raw_finite_count",
                "raw_p01",
                "raw_median",
                "raw_p99",
                "norm_median",
                "note",
            ],
            max_rows=len(component_summary),
        ),
        "",
        "## Top 30 HKLs By High-Low Risk Separation",
        "",
        markdown_table(top_separation, HKL_SUMMARY_COLUMNS, max_rows=TOP_ROWS),
        "",
        "## Top 30 HKLs By Number Of Dropped Worst-Risk Observations",
        "",
        markdown_table(top_dropped, HKL_SUMMARY_COLUMNS, max_rows=TOP_ROWS),
        "",
        "## Output Files",
        "",
        f"- Low stream: `{paths[low_label]}`",
        f"- High stream: `{paths[high_label]}`",
        f"- Random stream: `{paths[random_label]}`",
        f"- Random-minus-worst stream: `{paths[minus_label]}`",
        "- `geometry_trust_selected_keys.csv`",
        "- `geometry_trust_by_hkl.csv`",
        "- `geometry_trust_by_resolution.csv`",
        "- `geometry_trust_component_summary.csv`",
        "- `summary.md`",
        "- `run_metadata.json`",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    outdir: Path,
    args: argparse.Namespace,
    load_stats: dict[str, int],
    selection_stats: dict[str, int],
    stream_stats: dict[str, Any],
    paths: dict[str, Path],
    split_labels: dict[str, str],
    score_info: dict[str, Any],
    component_summary: pd.DataFrame,
) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "stream": str(args.stream),
            "scores_csv": str(args.scores_csv),
        },
        "output_paths": {key: str(value) for key, value in paths.items()},
        "seed": int(args.seed),
        "score_mode": str(args.score_mode),
        "score_formula": score_info.get("formula", ""),
        "column_mapping": score_info.get("mapping", {}),
        "fraction": float(args.fraction),
        "fraction_percent_label": percent_label(float(args.fraction)),
        "drop_worst_fraction": float(args.drop_worst_fraction),
        "drop_worst_percent_label": percent_label(float(args.drop_worst_fraction)),
        "split_labels": split_labels,
        "min_obs_per_hkl": int(args.min_obs_per_hkl),
        "normalization_metadata": component_summary.to_dict(orient="records"),
        "row_counts": {
            **load_stats,
            **selection_stats,
            **stream_stats,
        },
        "warnings": [
            "This tests a geometry-coupling approximation to dynamical reliability, not exact dynamical intensities.",
            "This is an observation-level split, not an intensity correction.",
            "No symmetry canonicalization is applied.",
            "Unmatched or unselected stream observations are removed from all split streams.",
            "The key presentation test is whether random-minus-worst improves over pure random.",
        ],
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    split_labels = split_labels_for_args(args)

    log("Loading score table")
    eligible, component_summary, debug_columns_all, score_info, load_stats = load_scores(args.scores_csv, args)
    debug_columns = debug_columns_all if (args.write_debug_columns or args.score_mode == "equation_proxy_from_existing_terms") else [
        "trust_risk_raw",
        "trust_risk_norm",
    ]
    log(
        "Loaded scores: "
        f"rows={load_stats['score_rows_loaded']:,}, "
        f"finite_score={load_stats['eligible_finite_score_rows']:,}, "
        f"unique_keys={load_stats['eligible_unique_key_rows']:,}"
    )

    log(f"Computed trust-risk score using mode={args.score_mode}")

    log("Parsing stream unit cell for resolution metadata")
    cell = load_unit_cell_from_stream(args.stream)
    eligible = add_resolution_columns(eligible, cell)

    log("Selecting per-HKL low/high/random and random-minus-worst keys")
    selected, hkl_summary, split_keys, selection_stats = select_split_keys(
        eligible,
        args,
        split_labels,
        debug_columns,
    )
    if not selected.empty:
        selected["resolution_bin"] = [resolution_bin_for_d(float(value)) for value in selected["d_angstrom"]]
    resolution_summary = summarize_resolution(selected, split_labels)
    log(
        "Selected split keys: "
        f"hkls={selection_stats['signed_hkls_passing_min_observations']:,}, "
        f"low={selection_stats['selected_low_keys']:,}, "
        f"high={selection_stats['selected_high_keys']:,}, "
        f"random={selection_stats['selected_random_keys']:,}, "
        f"random_minus_worst={selection_stats['selected_random_minus_worst_keys']:,}"
    )

    selected.to_csv(args.outdir / "geometry_trust_selected_keys.csv", index=False)
    hkl_summary.to_csv(args.outdir / "geometry_trust_by_hkl.csv", index=False)
    resolution_summary.to_csv(args.outdir / "geometry_trust_by_resolution.csv", index=False)
    component_summary.to_csv(args.outdir / "geometry_trust_component_summary.csv", index=False)

    paths = stream_paths_for_labels(args.outdir, split_labels, int(args.seed))
    log("Writing split streams")
    stream_stats = write_split_streams(args.stream, paths, split_keys, split_labels)

    write_summary(
        args.outdir,
        args,
        load_stats,
        selection_stats,
        stream_stats,
        selected,
        hkl_summary,
        resolution_summary,
        component_summary,
        paths,
        split_labels,
        score_info,
    )
    write_metadata(args.outdir, args, load_stats, selection_stats, stream_stats, paths, split_labels, score_info, component_summary)

    for path in paths.values():
        print(f"Wrote: {path}")
    print(f"Wrote: {args.outdir / 'geometry_trust_selected_keys.csv'}")
    print(f"Wrote: {args.outdir / 'geometry_trust_by_hkl.csv'}")
    print(f"Wrote: {args.outdir / 'geometry_trust_by_resolution.csv'}")
    print(f"Wrote: {args.outdir / 'geometry_trust_component_summary.csv'}")
    print(f"Wrote: {args.outdir / 'summary.md'}")
    print(f"Wrote: {args.outdir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
