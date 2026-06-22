#!/usr/bin/env python3
"""Proof-of-concept weak-positive enhancement-feed stream correction.

This standalone diagnostic rewriter uses enhancement-feed partialator diagnostics
to downscale only locally weak signed-HKL observations that are both high-feed
within their frame and part of HKLs with clear positive I_times_weight shifts.

It is intentionally narrow: no symmetry canonicalization is applied, HKL indices
are not changed, and unmatched stream observations are left untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
TARGET_RANK_MIN = 0.90
MIN_RELATIVE_SHIFT_IWEIGHT = 0.50
MIN_TAIL_OBS = 10
MIN_SCALE = 0.10
MAX_SCALE = 1.00
TOP_ROWS = 20
PROGRESS_EVERY = 1_000_000
RESOLUTION_SHELLS_INV_NM = [
    (0.500, 9.211, "20.00-1.09A"),
    (9.211, 11.604, "1.09-0.86A"),
    (11.604, 13.283, "0.86-0.75A"),
    (13.283, 14.620, "0.75-0.68A"),
    (14.620, 15.749, "0.68-0.64A"),
    (15.749, 16.736, "0.64-0.60A"),
    (16.736, 17.618, "0.60-0.57A"),
    (17.618, 18.420, "0.57-0.54A"),
    (18.420, 19.158, "0.54-0.52A"),
    (19.158, 19.843, "0.52-0.50A"),
    (19.843, 20.483, "0.50-0.49A"),
    (20.483, 21.086, "0.49-0.47A"),
    (21.086, 21.656, "0.47-0.46A"),
    (21.656, 22.198, "0.46-0.45A"),
    (22.198, 22.714, "0.45-0.44A"),
    (22.714, 23.208, "0.44-0.43A"),
    (23.208, 23.682, "0.43-0.42A"),
    (23.682, 24.137, "0.42-0.41A"),
    (24.137, 24.576, "0.41-0.407A"),
    (24.576, 25.000, "0.407-0.400A"),
]

STREAM_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_CELL_RE = re.compile(
    rf"^\s*Cell parameters\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+nm,"
    rf"\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+({STREAM_FLOAT})\s+deg"
)
STREAM_UNITCELL_LENGTH_RE = re.compile(rf"^\s*([abc])\s*=\s*({STREAM_FLOAT})\s*A")
STREAM_UNITCELL_ANGLE_RE = re.compile(rf"^\s*(al|be|ga|alpha|beta|gamma)\s*=\s*({STREAM_FLOAT})\s*deg")

SHIFT_REQUIRED_COLUMNS = [
    *HKL_COLUMNS,
    "local_strength_class",
    "relative_shift_Iweight",
    "n_low",
    "n_high",
    "Iweight_low_median",
    "Iweight_high_median",
]
JOINED_REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "enh_feed_rank_frame",
    "enh_feed_raw",
]
SCALED_OUTPUT_COLUMNS = [
    "source_filename",
    "event",
    "h",
    "k",
    "l",
    "old_I",
    "new_I",
    "scale",
    "enh_feed_rank_frame",
    "enh_feed_raw",
    "Iweight_low_median",
    "Iweight_high_median",
    "relative_shift_Iweight",
    "local_strength_class",
    "signed_delta_I",
    "abs_delta_I",
    "relative_delta_I",
    "d_angstrom",
    "inv_nm",
    "resolution_shell",
]
SHELL_SUMMARY_COLUMNS = [
    "resolution_shell",
    "min_inv_nm",
    "max_inv_nm",
    "d_high_A",
    "d_low_A",
    "n_scaled",
    "fraction_of_all_scaled",
    "sum_abs_delta_I",
    "fraction_of_total_abs_delta_I",
    "median_abs_delta_I",
    "median_signed_delta_I",
    "median_relative_delta_I",
    "median_scale",
    "min_scale",
    "max_scale",
]
HKL_MASS_COLUMNS = [
    "h",
    "k",
    "l",
    "d_angstrom",
    "inv_nm",
    "resolution_shell",
    "n_scaled",
    "sum_abs_delta_I",
    "median_abs_delta_I",
    "median_signed_delta_I",
    "median_relative_delta_I",
    "median_scale",
    "min_scale",
    "max_scale",
    "relative_shift_Iweight",
    "Iweight_low_median",
    "Iweight_high_median",
    "local_strength_class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="Original CrystFEL stream to rewrite")
    parser.add_argument(
        "--joined-observations-csv",
        required=True,
        type=Path,
        help="joined_enh_feed_partialator_observations.csv",
    )
    parser.add_argument(
        "--shift-by-hkl-csv",
        required=True,
        type=Path,
        help="enh_feed_shift_by_signed_hkl_with_strength_class.csv",
    )
    parser.add_argument("--output-stream", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    for name in ["stream", "joined_observations_csv", "shift_by_hkl_csv"]:
        path = getattr(args, name)
        if not path.exists():
            raise SystemExit(f"--{name.replace('_', '-')} not found: {path}")
    return args


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def normalize_source(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_event(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def build_key(source: Any, event: Any, h: int, k: int, l: int) -> tuple[str, str, int, int, int]:
    return normalize_source(source), normalize_event(event), int(h), int(k), int(l)


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
                    key = {"al": "alpha", "be": "beta", "ga": "gamma"}.get(match.group(1).lower(), match.group(1).lower())
                    cell[key] = float(match.group(2))
                    if has_complete_cell(cell):
                        return cell
                    continue

            if line.startswith("----- Begin chunk -----") and not has_complete_cell(cell):
                # If no stream-header cell was found, keep scanning for the first crystal
                # "Cell parameters ... nm" line, but do not rely on later metadata.
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


def resolution_shell_for_inv_nm(inv_nm: float) -> str:
    if not np.isfinite(inv_nm):
        return "outside"
    for idx, (min_inv, max_inv, label) in enumerate(RESOLUTION_SHELLS_INV_NM):
        if min_inv <= inv_nm < max_inv:
            return label
        if idx == len(RESOLUTION_SHELLS_INV_NM) - 1 and min_inv <= inv_nm <= max_inv:
            return label
    return "outside"


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


def normalize_hkl_columns(table: pd.DataFrame, label: str) -> pd.DataFrame:
    out = table.copy()
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    bad = out[HKL_COLUMNS].isna().any(axis=1)
    if bool(bad.any()):
        log(f"Dropping {int(bad.sum()):,} {label} row(s) with non-numeric HKL values")
    out = out.loc[~bad].copy()
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def load_target_hkls(path: Path) -> tuple[dict[tuple[int, int, int], dict[str, float | str | int]], pd.DataFrame]:
    table = pd.read_csv(path)
    require_columns(table, SHIFT_REQUIRED_COLUMNS, "shift-by-HKL CSV")
    table = normalize_hkl_columns(table, "shift-by-HKL")

    for column in ["relative_shift_Iweight", "n_low", "n_high", "Iweight_low_median", "Iweight_high_median"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["local_strength_class"] = table["local_strength_class"].astype(str).str.strip().str.lower()

    target = table.loc[
        (table["local_strength_class"] == "weak")
        & (table["relative_shift_Iweight"] > MIN_RELATIVE_SHIFT_IWEIGHT)
        & (table["n_low"] >= MIN_TAIL_OBS)
        & (table["n_high"] >= MIN_TAIL_OBS)
        & (table["Iweight_low_median"] > 0.0)
        & (table["Iweight_high_median"] > table["Iweight_low_median"])
    ].copy()

    target["raw_scale"] = target["Iweight_low_median"] / target["Iweight_high_median"]
    target["scale"] = target["raw_scale"].clip(lower=MIN_SCALE, upper=MAX_SCALE)
    target = target.dropna(subset=["scale"])

    lookup: dict[tuple[int, int, int], dict[str, float | str | int]] = {}
    for row in target.itertuples(index=False):
        key = (int(row.h), int(row.k), int(row.l))
        lookup[key] = {
            "scale": float(row.scale),
            "Iweight_low_median": float(row.Iweight_low_median),
            "Iweight_high_median": float(row.Iweight_high_median),
            "relative_shift_Iweight": float(row.relative_shift_Iweight),
            "local_strength_class": str(row.local_strength_class),
            "n_low": int(row.n_low),
            "n_high": int(row.n_high),
        }
    return lookup, target


def load_joined_observation_lookup(
    path: Path,
    target_hkls: set[tuple[int, int, int]],
) -> tuple[dict[tuple[str, str, int, int, int], dict[str, float]], set[tuple[str, str, int, int, int]], dict[str, int]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    require_columns(pd.DataFrame(columns=header), JOINED_REQUIRED_COLUMNS, "joined observations CSV")
    optional = [column for column in ["score_matched"] if column in header]
    table = pd.read_csv(path, usecols=[*JOINED_REQUIRED_COLUMNS, *optional])
    table = normalize_hkl_columns(table, "joined observations")
    table["source_filename"] = table["source_filename"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["enh_feed_rank_frame"] = pd.to_numeric(table["enh_feed_rank_frame"], errors="coerce")
    table["enh_feed_raw"] = pd.to_numeric(table["enh_feed_raw"], errors="coerce")

    if "score_matched" in table.columns:
        matched_mask = parse_bool_series(table["score_matched"])
    else:
        matched_mask = table["enh_feed_rank_frame"].notna() & table["enh_feed_raw"].notna()

    finite_mask = table["enh_feed_rank_frame"].map(np.isfinite) & table["enh_feed_raw"].map(np.isfinite)
    diagnostic_rows = table.loc[matched_mask & finite_mask].copy()

    diagnostic_key_set: set[tuple[str, str, int, int, int]] = set()
    selected_lookup: dict[tuple[str, str, int, int, int], dict[str, float]] = {}
    duplicate_diagnostic_keys = 0
    duplicate_selected_keys = 0

    for row in diagnostic_rows.itertuples(index=False):
        hkl = (int(row.h), int(row.k), int(row.l))
        key = build_key(row.source_filename, row.event, int(row.h), int(row.k), int(row.l))
        if key in diagnostic_key_set:
            duplicate_diagnostic_keys += 1
        diagnostic_key_set.add(key)
        if hkl not in target_hkls or float(row.enh_feed_rank_frame) < TARGET_RANK_MIN:
            continue
        if key in selected_lookup:
            duplicate_selected_keys += 1
            if float(row.enh_feed_rank_frame) <= selected_lookup[key]["enh_feed_rank_frame"]:
                continue
        selected_lookup[key] = {
            "enh_feed_rank_frame": float(row.enh_feed_rank_frame),
            "enh_feed_raw": float(row.enh_feed_raw),
        }

    stats = {
        "joined_rows_loaded": int(len(table)),
        "diagnostic_rows_with_finite_feed": int(len(diagnostic_rows)),
        "duplicate_diagnostic_keys_seen_after_first": int(duplicate_diagnostic_keys),
        "selected_high_feed_observation_keys": int(len(selected_lookup)),
        "duplicate_selected_keys_seen_after_first": int(duplicate_selected_keys),
    }
    return selected_lookup, diagnostic_key_set, stats


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def parse_reflection_line(line: str) -> tuple[int, int, int, float, float | None] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        intensity = float(parts[3])
    except ValueError:
        return None
    sigma = None
    if len(parts) >= 5:
        try:
            sigma = float(parts[4])
        except ValueError:
            sigma = None
    return h, k, l, intensity, sigma


def update_reflection_line(raw_line: str, new_intensity: float, scale: float, old_sigma: float | None) -> str:
    newline = "\n" if raw_line.endswith("\n") else ""
    body = raw_line[:-1] if newline else raw_line
    parts = body.split()
    if len(parts) < 4:
        return raw_line
    parts[3] = format_stream_float(new_intensity)
    if len(parts) >= 5 and old_sigma is not None:
        parts[4] = format_stream_float(float(old_sigma) * float(scale))
    return " ".join(parts) + newline


def format_stream_float(value: float) -> str:
    return f"{float(value):.6g}"


def rewrite_stream(
    stream_path: Path,
    output_stream: Path,
    selected_observations: dict[tuple[str, str, int, int, int], dict[str, float]],
    diagnostic_keys: set[tuple[str, str, int, int, int]],
    target_hkls: dict[tuple[int, int, int], dict[str, float | str | int]],
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    output_stream.parent.mkdir(parents=True, exist_ok=True)
    scaled_records: list[dict[str, Any]] = []

    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False

    chunks_seen = 0
    crystals_seen = 0
    stream_observations_seen = 0
    matched_to_diagnostics = 0
    scaled_count = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as src, output_stream.open(
        "w",
        encoding="utf-8",
    ) as dst:
        for raw_line in src:
            line = raw_line.rstrip("\n")

            if "Begin chunk" in line:
                chunks_seen += 1
                chunk_source = ""
                chunk_event = ""
                current_source = ""
                current_event = ""
                in_crystal = False
                in_reflections = False
                if chunks_seen % PROGRESS_EVERY == 0:
                    log(f"Stream progress: chunks={chunks_seen:,}, reflections={stream_observations_seen:,}, scaled={scaled_count:,}")
                dst.write(raw_line)
                continue

            image_match = STREAM_IMAGE_RE.match(line)
            if image_match:
                if in_crystal:
                    current_source = normalize_source(image_match.group(1))
                else:
                    chunk_source = normalize_source(image_match.group(1))
                dst.write(raw_line)
                continue

            event_match = STREAM_EVENT_RE.match(line)
            if event_match:
                if in_crystal:
                    current_event = normalize_event(event_match.group(1))
                else:
                    chunk_event = normalize_event(event_match.group(1))
                dst.write(raw_line)
                continue

            if "Begin crystal" in line:
                crystals_seen += 1
                current_source = chunk_source
                current_event = chunk_event
                in_crystal = True
                in_reflections = False
                dst.write(raw_line)
                continue

            if "End crystal" in line:
                in_crystal = False
                in_reflections = False
                dst.write(raw_line)
                continue

            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                dst.write(raw_line)
                continue

            if in_reflections and line.startswith("End of reflections"):
                in_reflections = False
                dst.write(raw_line)
                continue

            if in_crystal and in_reflections:
                parsed = parse_reflection_line(line)
                if parsed is not None:
                    h, k, l, old_intensity, old_sigma = parsed
                    stream_observations_seen += 1
                    if stream_observations_seen % PROGRESS_EVERY == 0:
                        log(
                            "Stream progress: "
                            f"chunks={chunks_seen:,}, reflections={stream_observations_seen:,}, scaled={scaled_count:,}"
                        )
                    key = build_key(current_source, current_event, h, k, l)
                    if key in diagnostic_keys:
                        matched_to_diagnostics += 1
                    selected = selected_observations.get(key)
                    target = target_hkls.get((h, k, l))
                    if selected is not None and target is not None:
                        scale = float(target["scale"])
                        new_intensity = float(old_intensity) * scale
                        dst.write(update_reflection_line(raw_line, new_intensity, scale, old_sigma))
                        scaled_count += 1
                        scaled_records.append(
                            {
                                "source_filename": normalize_source(current_source),
                                "event": normalize_event(current_event),
                                "h": int(h),
                                "k": int(k),
                                "l": int(l),
                                "old_I": float(old_intensity),
                                "new_I": float(new_intensity),
                                "scale": scale,
                                "enh_feed_rank_frame": float(selected["enh_feed_rank_frame"]),
                                "enh_feed_raw": float(selected["enh_feed_raw"]),
                                "Iweight_low_median": float(target["Iweight_low_median"]),
                                "Iweight_high_median": float(target["Iweight_high_median"]),
                                "relative_shift_Iweight": float(target["relative_shift_Iweight"]),
                                "local_strength_class": str(target["local_strength_class"]),
                            }
                        )
                        continue

            dst.write(raw_line)

    scaled = pd.DataFrame.from_records(scaled_records, columns=SCALED_OUTPUT_COLUMNS)
    stats: dict[str, int | float] = {
        "chunks_seen": int(chunks_seen),
        "crystals_seen": int(crystals_seen),
        "stream_observations_seen": int(stream_observations_seen),
        "matched_to_diagnostics": int(matched_to_diagnostics),
        "scaled_observations": int(scaled_count),
        "unmatched_observations": int(stream_observations_seen - matched_to_diagnostics),
        "fraction_stream_observations_scaled": float(scaled_count / max(stream_observations_seen, 1)),
    }
    return scaled, stats


def enrich_scaled_observations(scaled: pd.DataFrame, cell: dict[str, float]) -> pd.DataFrame:
    out = scaled.copy()
    for column in SCALED_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype=float if column not in {"source_filename", "event", "resolution_shell", "local_strength_class"} else object)
    if out.empty:
        return out.loc[:, SCALED_OUTPUT_COLUMNS].copy()

    reciprocal_basis = reciprocal_basis_from_cell(cell)
    old_i = pd.to_numeric(out["old_I"], errors="coerce")
    new_i = pd.to_numeric(out["new_I"], errors="coerce")
    signed_delta = new_i - old_i
    out["signed_delta_I"] = signed_delta
    out["abs_delta_I"] = signed_delta.abs()
    out["relative_delta_I"] = signed_delta / np.maximum(old_i.abs(), 1.0)

    d_values = [
        d_spacing_angstrom(int(row.h), int(row.k), int(row.l), reciprocal_basis)
        for row in out[HKL_COLUMNS].itertuples(index=False)
    ]
    out["d_angstrom"] = d_values
    out["inv_nm"] = np.where(
        pd.to_numeric(out["d_angstrom"], errors="coerce").to_numpy(dtype=float) > 0.0,
        10.0 / pd.to_numeric(out["d_angstrom"], errors="coerce").to_numpy(dtype=float),
        np.nan,
    )
    out["resolution_shell"] = [resolution_shell_for_inv_nm(float(value)) for value in out["inv_nm"]]
    return out.loc[:, SCALED_OUTPUT_COLUMNS].copy()


def summarize_by_resolution_shell(scaled: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_scaled = int(len(scaled))
    total_abs_delta = float(pd.to_numeric(scaled.get("abs_delta_I", pd.Series(dtype=float)), errors="coerce").sum()) if total_scaled else 0.0
    shell_specs = list(RESOLUTION_SHELLS_INV_NM)
    if not scaled.empty and (scaled["resolution_shell"] == "outside").any():
        shell_specs.append((np.nan, np.nan, "outside"))

    for min_inv, max_inv, label in shell_specs:
        group = scaled.loc[scaled["resolution_shell"] == label] if not scaled.empty else scaled
        sum_abs_delta = float(pd.to_numeric(group.get("abs_delta_I", pd.Series(dtype=float)), errors="coerce").sum()) if not group.empty else 0.0
        rows.append(
            {
                "resolution_shell": label,
                "min_inv_nm": float(min_inv) if np.isfinite(min_inv) else np.nan,
                "max_inv_nm": float(max_inv) if np.isfinite(max_inv) else np.nan,
                "d_high_A": 10.0 / float(max_inv) if np.isfinite(max_inv) and max_inv > 0 else np.nan,
                "d_low_A": 10.0 / float(min_inv) if np.isfinite(min_inv) and min_inv > 0 else np.nan,
                "n_scaled": int(len(group)),
                "fraction_of_all_scaled": float(len(group) / max(total_scaled, 1)),
                "sum_abs_delta_I": sum_abs_delta,
                "fraction_of_total_abs_delta_I": float(sum_abs_delta / total_abs_delta) if total_abs_delta > 0.0 else 0.0,
                "median_abs_delta_I": median_or_nan(group.get("abs_delta_I", pd.Series(dtype=float))),
                "median_signed_delta_I": median_or_nan(group.get("signed_delta_I", pd.Series(dtype=float))),
                "median_relative_delta_I": median_or_nan(group.get("relative_delta_I", pd.Series(dtype=float))),
                "median_scale": median_or_nan(group.get("scale", pd.Series(dtype=float))),
                "min_scale": min_or_nan(group.get("scale", pd.Series(dtype=float))),
                "max_scale": max_or_nan(group.get("scale", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame.from_records(rows, columns=SHELL_SUMMARY_COLUMNS)


def summarize_by_hkl_correction_mass(scaled: pd.DataFrame) -> pd.DataFrame:
    if scaled.empty:
        return pd.DataFrame(columns=HKL_MASS_COLUMNS)
    grouped = (
        scaled.groupby(HKL_COLUMNS, sort=True)
        .agg(
            d_angstrom=("d_angstrom", "first"),
            inv_nm=("inv_nm", "first"),
            resolution_shell=("resolution_shell", "first"),
            n_scaled=("scale", "size"),
            sum_abs_delta_I=("abs_delta_I", "sum"),
            median_abs_delta_I=("abs_delta_I", "median"),
            median_signed_delta_I=("signed_delta_I", "median"),
            median_relative_delta_I=("relative_delta_I", "median"),
            median_scale=("scale", "median"),
            min_scale=("scale", "min"),
            max_scale=("scale", "max"),
            relative_shift_Iweight=("relative_shift_Iweight", "first"),
            Iweight_low_median=("Iweight_low_median", "first"),
            Iweight_high_median=("Iweight_high_median", "first"),
            local_strength_class=("local_strength_class", "first"),
        )
        .reset_index()
    )
    return grouped.sort_values("sum_abs_delta_I", ascending=False).loc[:, HKL_MASS_COLUMNS].copy()


def median_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def min_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if not values.empty else np.nan


def max_or_nan(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if not values.empty else np.nan


def scale_stats(scaled: pd.DataFrame) -> dict[str, float | None]:
    if scaled.empty:
        return {"scale_min": None, "scale_median": None, "scale_max": None}
    values = pd.to_numeric(scaled["scale"], errors="coerce").dropna()
    if values.empty:
        return {"scale_min": None, "scale_median": None, "scale_max": None}
    return {
        "scale_min": float(values.min()),
        "scale_median": float(values.median()),
        "scale_max": float(values.max()),
    }


def top_by_scaled_observations(scaled: pd.DataFrame) -> pd.DataFrame:
    if scaled.empty:
        return pd.DataFrame(columns=[*HKL_COLUMNS, "n_scaled", "scale", "relative_shift_Iweight"])
    grouped = (
        scaled.groupby(HKL_COLUMNS, sort=True)
        .agg(
            n_scaled=("scale", "size"),
            scale=("scale", "first"),
            relative_shift_Iweight=("relative_shift_Iweight", "first"),
            Iweight_low_median=("Iweight_low_median", "first"),
            Iweight_high_median=("Iweight_high_median", "first"),
        )
        .reset_index()
    )
    return grouped.sort_values(["n_scaled", "relative_shift_Iweight"], ascending=[False, False]).head(TOP_ROWS)


def top_by_strongest_downscale(target_table: pd.DataFrame, scaled: pd.DataFrame) -> pd.DataFrame:
    if target_table.empty:
        return pd.DataFrame(columns=[*HKL_COLUMNS, "scale", "n_scaled", "relative_shift_Iweight"])
    counts = (
        scaled.groupby(HKL_COLUMNS, sort=True)
        .size()
        .rename("n_scaled")
        .reset_index()
        if not scaled.empty
        else pd.DataFrame(columns=[*HKL_COLUMNS, "n_scaled"])
    )
    table = target_table.merge(counts, on=HKL_COLUMNS, how="left")
    table["n_scaled"] = table["n_scaled"].fillna(0).astype("int64")
    return table.sort_values(["scale", "relative_shift_Iweight"], ascending=[True, False]).head(TOP_ROWS)


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = TOP_ROWS) -> str:
    if table is None or table.empty:
        return "_No rows._"
    view = table.loc[:, [column for column in columns if column in table.columns]].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_summary(
    path: Path,
    args: argparse.Namespace,
    target_table: pd.DataFrame,
    joined_stats: dict[str, int],
    rewrite_stats: dict[str, int | float],
    scaled: pd.DataFrame,
    shell_summary: pd.DataFrame,
    hkl_mass_summary: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = scale_stats(scaled)
    top_scaled = top_by_scaled_observations(scaled)
    top_downscale = top_by_strongest_downscale(target_table, scaled)
    table_columns = [
        "h",
        "k",
        "l",
        "n_scaled",
        "scale",
        "relative_shift_Iweight",
        "Iweight_low_median",
        "Iweight_high_median",
    ]
    shell_columns = [
        "resolution_shell",
        "n_scaled",
        "fraction_of_all_scaled",
        "sum_abs_delta_I",
        "fraction_of_total_abs_delta_I",
        "median_abs_delta_I",
        "median_signed_delta_I",
        "median_relative_delta_I",
        "median_scale",
        "min_scale",
        "max_scale",
    ]
    hkl_mass_columns = [
        "h",
        "k",
        "l",
        "resolution_shell",
        "n_scaled",
        "sum_abs_delta_I",
        "median_abs_delta_I",
        "median_signed_delta_I",
        "median_relative_delta_I",
        "median_scale",
        "relative_shift_Iweight",
    ]

    lines = [
        "# Enhancement-Feed Weak-Positive Stream Scaling Proof Of Concept",
        "",
        "## Scope",
        "",
        "- This is a proof-of-concept stream correction.",
        "- It only tests downscaling locally weak, high-feed observations from signed HKLs with clear positive enhancement-feed I_times_weight shifts.",
        "- It does not canonicalize symmetry equivalents.",
        "- It does not change HKL indices or non-target observations.",
        "- It rewrites only selected stream intensity values, and scales the usual CrystFEL sigma column by the same factor when present.",
        "",
        "## Inputs",
        "",
        f"- Stream: `{args.stream}`",
        f"- Joined observations CSV: `{args.joined_observations_csv}`",
        f"- Shift-by-HKL CSV: `{args.shift_by_hkl_csv}`",
        f"- Output stream: `{args.output_stream}`",
        "",
        "## Target Rules",
        "",
        "- HKL targets require `local_strength_class == weak`, `relative_shift_Iweight > 0.5`, `n_low >= 10`, `n_high >= 10`, `Iweight_low_median > 0`, and `Iweight_high_median > Iweight_low_median`.",
        "- Observation targets additionally require `enh_feed_rank_frame >= 0.9`.",
        "- Per-HKL scale is `clip(Iweight_low_median / Iweight_high_median, 0.10, 1.00)`.",
        "",
        "## Counts",
        "",
        f"- Target weak-positive HKLs: {len(target_table)}",
        f"- Stream observations seen: {rewrite_stats['stream_observations_seen']}",
        f"- Stream observations matched to diagnostics: {rewrite_stats['matched_to_diagnostics']}",
        f"- Stream observations scaled: {rewrite_stats['scaled_observations']}",
        f"- Fraction of stream observations scaled: {rewrite_stats['fraction_stream_observations_scaled']:.6g}",
        f"- Unmatched stream observations: {rewrite_stats['unmatched_observations']}",
        f"- Joined diagnostic rows loaded: {joined_stats['joined_rows_loaded']}",
        f"- Joined diagnostic rows with finite feed values: {joined_stats['diagnostic_rows_with_finite_feed']}",
        f"- Selected high-feed diagnostic observation keys: {joined_stats['selected_high_feed_observation_keys']}",
        "",
        "## Scale Factors",
        "",
        f"- Min scale: {format_optional(stats['scale_min'])}",
        f"- Median scale: {format_optional(stats['scale_median'])}",
        f"- Max scale: {format_optional(stats['scale_max'])}",
        "",
        "## Top Target HKLs By Number Of Scaled Observations",
        "",
        markdown_table(top_scaled, table_columns),
        "",
        "## Top Target HKLs By Strongest Downscale",
        "",
        markdown_table(top_downscale, table_columns),
        "",
        "## Resolution Distribution Of Scaled Corrections",
        "",
        markdown_table(shell_summary, shell_columns, max_rows=len(shell_summary)),
        "",
        "## Top HKLs By Correction Mass",
        "",
        markdown_table(hkl_mass_summary, hkl_mass_columns, max_rows=TOP_ROWS),
        "",
        "## Output Files",
        "",
        f"- Corrected stream: `{args.output_stream}`",
        f"- Scaled-observation CSV: `{path.parent / 'enh_feed_weak_positive_scaled_observations.csv'}`",
        f"- Resolution-shell CSV: `{path.parent / 'enh_feed_scaled_by_resolution_shell.csv'}`",
        f"- HKL correction-mass CSV: `{path.parent / 'enh_feed_scaled_by_hkl_correction_mass.csv'}`",
        f"- Summary: `{path}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def main() -> int:
    args = parse_args()

    log("Loading target weak-positive signed HKLs")
    target_hkls, target_table = load_target_hkls(args.shift_by_hkl_csv)
    log(f"Target weak-positive HKLs: {len(target_hkls):,}")

    log("Loading joined observation diagnostics")
    selected_observations, diagnostic_keys, joined_stats = load_joined_observation_lookup(
        args.joined_observations_csv,
        set(target_hkls),
    )
    log(
        "Joined diagnostics loaded: "
        f"finite={joined_stats['diagnostic_rows_with_finite_feed']:,}, "
        f"selected_high_feed_keys={joined_stats['selected_high_feed_observation_keys']:,}"
    )

    log("Parsing stream unit cell for resolution diagnostics")
    unit_cell = load_unit_cell_from_stream(args.stream)

    log("Rewriting stream")
    scaled, rewrite_stats = rewrite_stream(
        args.stream,
        args.output_stream,
        selected_observations,
        diagnostic_keys,
        target_hkls,
    )
    scaled = enrich_scaled_observations(scaled, unit_cell)
    shell_summary = summarize_by_resolution_shell(scaled)
    hkl_mass_summary = summarize_by_hkl_correction_mass(scaled)

    scaled_csv = args.summary_md.parent / "enh_feed_weak_positive_scaled_observations.csv"
    shell_csv = args.summary_md.parent / "enh_feed_scaled_by_resolution_shell.csv"
    hkl_mass_csv = args.summary_md.parent / "enh_feed_scaled_by_hkl_correction_mass.csv"
    scaled_csv.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(scaled_csv, index=False)
    shell_summary.to_csv(shell_csv, index=False)
    hkl_mass_summary.to_csv(hkl_mass_csv, index=False)
    write_summary(args.summary_md, args, target_table, joined_stats, rewrite_stats, scaled, shell_summary, hkl_mass_summary)

    print(f"Target weak-positive HKLs: {len(target_table)}")
    print(f"Stream observations seen: {rewrite_stats['stream_observations_seen']}")
    print(f"Matched to diagnostics: {rewrite_stats['matched_to_diagnostics']}")
    print(f"Scaled observations: {rewrite_stats['scaled_observations']}")
    print(f"Unmatched observations: {rewrite_stats['unmatched_observations']}")
    print("Resolution shell summary:")
    print(markdown_table(shell_summary, SHELL_SUMMARY_COLUMNS, max_rows=len(shell_summary)))
    print(f"Wrote: {args.output_stream}")
    print(f"Wrote: {scaled_csv}")
    print(f"Wrote: {shell_csv}")
    print(f"Wrote: {hkl_mass_csv}")
    print(f"Wrote: {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
