#!/usr/bin/env python3
"""Split a CrystFEL stream by observation-level enhancement-feed risk.

For each signed HKL, this diagnostic selects the lowest fraction, highest
fraction, and a random matched fraction of finite-score observations from joined enhancement-feed
diagnostics. The resulting streams preserve the original stream text except
that measured reflection rows are kept or removed according to exact
source_filename + event + signed h,k,l keys.

No symmetry canonicalization is applied, and intensities/sigmas are not changed.
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
TOP_ROWS = 20

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

SELECTED_COLUMNS = [
    "source_filename",
    "event",
    "h",
    "k",
    "l",
    "split",
    "score",
    "d_angstrom",
    "inv_nm",
]
HKL_SUMMARY_COLUMNS = [
    "h",
    "k",
    "l",
    "n_total_eligible",
    "n_selected_each_split",
    "low_score_min",
    "low_score_median",
    "low_score_max",
    "high_score_min",
    "high_score_median",
    "high_score_max",
    "random_score_median",
]
HKL_TOP_COLUMNS = [
    *HKL_SUMMARY_COLUMNS,
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
    parser.add_argument(
        "--joined-observations-csv",
        required=True,
        type=Path,
        help="Observation-level joined enhancement-feed diagnostics CSV",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--score-column", default="enh_feed_rank_frame")
    parser.add_argument("--min-obs-per-hkl", type=int, default=10)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.50,
        help="Per-signed-HKL fraction to keep in each low/high/random split; must satisfy 0 < fraction <= 0.50",
    )
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.joined_observations_csv.exists():
        raise SystemExit(f"--joined-observations-csv not found: {args.joined_observations_csv}")
    if int(args.min_obs_per_hkl) < 2:
        raise SystemExit("--min-obs-per-hkl must be >= 2")
    if not np.isfinite(float(args.fraction)) or not (0.0 < float(args.fraction) <= 0.50):
        raise SystemExit("--fraction must satisfy 0 < fraction <= 0.50")
    return args


def fraction_percent_label(fraction: float) -> int:
    return int(round(float(fraction) * 100.0))


def split_labels_for_fraction(fraction: float) -> dict[str, str]:
    percent = fraction_percent_label(fraction)
    return {
        "low": f"low_enh_{percent}",
        "high": f"high_enh_{percent}",
        "random": f"random_enh_{percent}",
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


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


def load_joined_diagnostics(path: Path, score_column: str) -> tuple[pd.DataFrame, dict[str, int]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = [*KEY_COLUMNS, score_column]
    optional = [column for column in ["d_angstrom", "inv_nm", "score_matched"] if column in header]
    require_columns(pd.DataFrame(columns=header), required, "joined observations CSV")

    table = pd.read_csv(path, usecols=[*required, *optional])
    stats = {"joined_rows_loaded": int(len(table))}
    table = table.copy()
    table["source_filename"] = table["source_filename"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    bad_hkl = table[HKL_COLUMNS].isna().any(axis=1)
    if bool(bad_hkl.any()):
        log(f"Dropping {int(bad_hkl.sum()):,} joined rows with non-numeric HKLs")
    table = table.loc[~bad_hkl].copy()
    table[HKL_COLUMNS] = table[HKL_COLUMNS].astype("int64")

    table["score"] = pd.to_numeric(table[score_column], errors="coerce")
    if "score_matched" in table.columns:
        matched_mask = parse_bool_series(table["score_matched"])
    else:
        matched_mask = pd.Series(True, index=table.index)
    finite_score = table["score"].map(np.isfinite)
    eligible = table.loc[matched_mask & finite_score].copy()

    duplicated = eligible.duplicated(KEY_COLUMNS, keep=False)
    stats.update(
        {
            "joined_rows_after_hkl_cleanup": int(len(table)),
            "eligible_finite_score_rows": int(len(eligible)),
            "duplicate_eligible_key_rows": int(duplicated.sum()),
            "duplicate_eligible_keys": int(eligible.loc[duplicated, KEY_COLUMNS].drop_duplicates().shape[0]),
        }
    )
    if duplicated.any():
        log(
            "Warning: duplicate exact observation keys found in joined diagnostics; "
            "keeping the first row per key for split selection."
        )
        eligible = eligible.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    stats["eligible_unique_key_rows"] = int(len(eligible))
    return eligible, stats


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


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


def select_split_keys(
    eligible: pd.DataFrame,
    seed: int,
    min_obs_per_hkl: int,
    fraction: float,
    split_labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[tuple[str, str, int, int, int]]], dict[str, int]]:
    rng = np.random.default_rng(int(seed))
    selected_rows: list[dict[str, Any]] = []
    hkl_rows: list[dict[str, Any]] = []
    split_keys: dict[str, set[tuple[str, str, int, int, int]]] = {split: set() for split in split_labels.values()}

    n_hkls_seen = 0
    n_hkls_passing = 0
    n_observations_in_passing_hkls = 0
    work = eligible.sort_values([*HKL_COLUMNS, "score", "source_filename", "event"], kind="mergesort").copy()

    for (h, k, l), group in work.groupby(HKL_COLUMNS, sort=True):
        n_hkls_seen += 1
        n_total = int(len(group))
        if n_total < int(min_obs_per_hkl):
            continue
        n_select = int(math.floor(n_total * float(fraction)))
        if n_select < 1:
            continue
        n_hkls_passing += 1
        n_observations_in_passing_hkls += n_total

        sorted_group = group.sort_values(["score", "source_filename", "event"], kind="mergesort").copy()
        low = sorted_group.head(n_select)
        high = sorted_group.tail(n_select)
        random_indices = rng.choice(sorted_group.index.to_numpy(), size=n_select, replace=False)
        random_group = sorted_group.loc[random_indices].sort_values(["source_filename", "event", "h", "k", "l"], kind="mergesort")

        split_groups = {
            split_labels["low"]: low,
            split_labels["high"]: high,
            split_labels["random"]: random_group,
        }
        for split, split_group in split_groups.items():
            for row in split_group.itertuples(index=False):
                key = build_key(row.source_filename, row.event, int(row.h), int(row.k), int(row.l))
                split_keys[split].add(key)
                selected_rows.append(
                    {
                        "source_filename": str(row.source_filename),
                        "event": str(row.event),
                        "h": int(row.h),
                        "k": int(row.k),
                        "l": int(row.l),
                        "split": split,
                        "score": float(row.score),
                        "d_angstrom": float(row.d_angstrom) if np.isfinite(row.d_angstrom) else np.nan,
                        "inv_nm": float(row.inv_nm) if np.isfinite(row.inv_nm) else np.nan,
                    }
                )

        hkl_rows.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "n_total_eligible": n_total,
                "n_selected_each_split": n_select,
                "low_score_min": min_or_nan(low["score"]),
                "low_score_median": median_or_nan(low["score"]),
                "low_score_max": max_or_nan(low["score"]),
                "high_score_min": min_or_nan(high["score"]),
                "high_score_median": median_or_nan(high["score"]),
                "high_score_max": max_or_nan(high["score"]),
                "random_score_median": median_or_nan(random_group["score"]),
                "high_low_median_separation": median_or_nan(high["score"]) - median_or_nan(low["score"]),
            }
        )

    selected = pd.DataFrame.from_records(selected_rows, columns=SELECTED_COLUMNS)
    hkl_summary = pd.DataFrame.from_records(hkl_rows, columns=HKL_TOP_COLUMNS)
    stats = {
        "signed_hkls_seen": int(n_hkls_seen),
        "signed_hkls_passing_min_observations": int(n_hkls_passing),
        "eligible_observations_in_passing_hkls": int(n_observations_in_passing_hkls),
        "selected_low_keys": int(len(split_keys[split_labels["low"]])),
        "selected_high_keys": int(len(split_keys[split_labels["high"]])),
        "selected_random_keys": int(len(split_keys[split_labels["random"]])),
    }
    return selected, hkl_summary, split_keys, stats


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def quantile_or_nan(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


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
    outdir: Path,
    seed: int,
    split_keys: dict[str, set[tuple[str, str, int, int, int]]],
    split_labels: dict[str, str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = {
        split_labels["low"]: outdir / f"{split_labels['low']}.stream",
        split_labels["high"]: outdir / f"{split_labels['high']}.stream",
        split_labels["random"]: outdir / f"{split_labels['random']}_seed{int(seed)}.stream",
    }
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
                                f"random={kept_counts[split_labels['random']]:,}"
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
    return paths, stats


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


def write_summary(
    outdir: Path,
    args: argparse.Namespace,
    load_stats: dict[str, int],
    selection_stats: dict[str, int],
    stream_stats: dict[str, Any],
    selected: pd.DataFrame,
    hkl_summary: pd.DataFrame,
    resolution_summary: pd.DataFrame,
    paths: dict[str, Path],
    split_labels: dict[str, str],
) -> None:
    split_scores = score_summary_by_split(selected, split_labels)
    top_separation = (
        hkl_summary.sort_values("high_low_median_separation", ascending=False).head(TOP_ROWS)
        if not hkl_summary.empty
        else hkl_summary
    )
    percent = fraction_percent_label(float(args.fraction))
    low_label = split_labels["low"]
    high_label = split_labels["high"]
    random_label = split_labels["random"]
    lines = [
        f"# Enhancement-Feed Observation-Risk {percent}% Stream Split",
        "",
        "## Warning",
        "",
        "- This is an observation-level split, not an intensity correction.",
        "- Stream reflection intensities, sigmas, HKL indices, and non-reflection text are not modified.",
        "- Observations not selected from the joined diagnostics are removed from all three split streams.",
        "",
        "## Inputs",
        "",
        f"- Stream: `{args.stream}`",
        f"- Joined observations CSV: `{args.joined_observations_csv}`",
        f"- Score column: `{args.score_column}`",
        f"- Seed: `{int(args.seed)}`",
        f"- Fraction: `{float(args.fraction):.6g}`",
        f"- min_obs_per_hkl: `{int(args.min_obs_per_hkl)}`",
        "",
        "## Counts",
        "",
        f"- Joined rows loaded: {load_stats['joined_rows_loaded']}",
        f"- Eligible finite-score joined observations: {load_stats['eligible_finite_score_rows']}",
        f"- Unique eligible observation keys: {load_stats['eligible_unique_key_rows']}",
        f"- Signed HKLs passing min observations: {selection_stats['signed_hkls_passing_min_observations']}",
        f"- Eligible observations in passing HKLs: {selection_stats['eligible_observations_in_passing_hkls']}",
        f"- Original stream observations seen: {stream_stats['stream_observations_seen']}",
        f"- Low split observations kept: {stream_stats[f'{low_label}_stream_observations_kept']}",
        f"- High split observations kept: {stream_stats[f'{high_label}_stream_observations_kept']}",
        f"- Random split observations kept: {stream_stats[f'{random_label}_stream_observations_kept']}",
        f"- Low fraction of original stream observations kept: {stream_stats[f'{low_label}_fraction_original_stream_observations_kept']:.6g}",
        f"- High fraction of original stream observations kept: {stream_stats[f'{high_label}_fraction_original_stream_observations_kept']:.6g}",
        f"- Random fraction of original stream observations kept: {stream_stats[f'{random_label}_fraction_original_stream_observations_kept']:.6g}",
        "",
        "## Score Summaries",
        "",
        markdown_table(split_scores),
        "",
        "## Resolution Summaries",
        "",
        markdown_table(resolution_summary, max_rows=len(resolution_summary)),
        "",
        "## Top 20 HKLs By High-Low Score Separation",
        "",
        markdown_table(top_separation, HKL_TOP_COLUMNS, max_rows=TOP_ROWS),
        "",
        "## Output Files",
        "",
        f"- Low stream: `{paths[low_label]}`",
        f"- High stream: `{paths[high_label]}`",
        f"- Random stream: `{paths[random_label]}`",
        "- `enh_feed_obs_split_selected_keys.csv`",
        "- `enh_feed_obs_split_by_hkl.csv`",
        "- `enh_feed_obs_split_by_resolution.csv`",
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
) -> None:
    metadata = {
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {
            "stream": str(args.stream),
            "joined_observations_csv": str(args.joined_observations_csv),
        },
        "output_paths": {key: str(value) for key, value in paths.items()},
        "seed": int(args.seed),
        "score_column": str(args.score_column),
        "fraction": float(args.fraction),
        "fraction_percent_label": fraction_percent_label(float(args.fraction)),
        "split_labels": split_labels,
        "min_obs_per_hkl": int(args.min_obs_per_hkl),
        "row_counts": {
            **load_stats,
            **selection_stats,
            **stream_stats,
        },
        "warnings": [
            "This is an observation-level split, not an intensity correction.",
            "No symmetry canonicalization is applied.",
            "Unmatched or unselected stream observations are removed from all split streams.",
        ],
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    split_labels = split_labels_for_fraction(float(args.fraction))

    log("Loading joined diagnostics")
    eligible, load_stats = load_joined_diagnostics(args.joined_observations_csv, args.score_column)
    log(
        "Loaded joined diagnostics: "
        f"rows={load_stats['joined_rows_loaded']:,}, "
        f"finite_score={load_stats['eligible_finite_score_rows']:,}, "
        f"unique_keys={load_stats['eligible_unique_key_rows']:,}"
    )

    log("Parsing stream unit cell for resolution metadata")
    cell = load_unit_cell_from_stream(args.stream)
    eligible = add_resolution_columns(eligible, cell)

    log(f"Selecting per-HKL low/high/random {float(args.fraction):.6g} observation keys")
    selected, hkl_summary, split_keys, selection_stats = select_split_keys(
        eligible,
        seed=int(args.seed),
        min_obs_per_hkl=int(args.min_obs_per_hkl),
        fraction=float(args.fraction),
        split_labels=split_labels,
    )
    selected["resolution_bin"] = [resolution_bin_for_d(float(value)) for value in selected["d_angstrom"]]
    resolution_summary = summarize_resolution(selected, split_labels)
    log(
        "Selected split keys: "
        f"hkls={selection_stats['signed_hkls_passing_min_observations']:,}, "
        f"low={selection_stats['selected_low_keys']:,}, "
        f"high={selection_stats['selected_high_keys']:,}, "
        f"random={selection_stats['selected_random_keys']:,}"
    )

    selected.loc[:, SELECTED_COLUMNS].to_csv(args.outdir / "enh_feed_obs_split_selected_keys.csv", index=False)
    hkl_summary.loc[:, HKL_SUMMARY_COLUMNS].to_csv(args.outdir / "enh_feed_obs_split_by_hkl.csv", index=False)
    resolution_summary.to_csv(args.outdir / "enh_feed_obs_split_by_resolution.csv", index=False)

    log("Writing split streams")
    paths, stream_stats = write_split_streams(args.stream, args.outdir, int(args.seed), split_keys, split_labels)

    write_summary(
        args.outdir,
        args,
        load_stats,
        selection_stats,
        stream_stats,
        selected,
        hkl_summary,
        resolution_summary,
        paths,
        split_labels,
    )
    write_metadata(args.outdir, args, load_stats, selection_stats, stream_stats, paths, split_labels)

    print(f"Wrote: {paths[split_labels['low']]}")
    print(f"Wrote: {paths[split_labels['high']]}")
    print(f"Wrote: {paths[split_labels['random']]}")
    print(f"Wrote: {args.outdir / 'enh_feed_obs_split_selected_keys.csv'}")
    print(f"Wrote: {args.outdir / 'enh_feed_obs_split_by_hkl.csv'}")
    print(f"Wrote: {args.outdir / 'enh_feed_obs_split_by_resolution.csv'}")
    print(f"Wrote: {args.outdir / 'summary.md'}")
    print(f"Wrote: {args.outdir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
