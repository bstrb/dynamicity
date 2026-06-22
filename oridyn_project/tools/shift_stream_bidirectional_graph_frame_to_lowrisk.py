#!/usr/bin/env python3
"""Aggressively shift high graph/frame-risk observations toward low-risk HKL reference.

This proof-of-concept rewrites a CrystFEL stream by modifying only selected
high-risk reflection intensities. Matching is exact by:

  source_filename + event + signed h + signed k + signed l

No symmetry canonicalization is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional acceleration
    cKDTree = None


SOURCE_COLUMNS = (
    "source_filename",
    "target_source",
    "source",
    "filename",
    "image_filename",
    "image",
    "file",
    "Image filename",
)

UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")
STREAM_CELL_RE = re.compile(
    r"Cell parameters\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+nm,\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+deg"
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an aggressive bidirectional graph/frame-risk corrected "
            "CrystFEL stream by shifting selected high-risk observations toward "
            "their same signed-HKL low-risk reference intensity."
        )
    )
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--unmerged", required=True, type=Path)
    parser.add_argument("--output-stream", required=True, type=Path)
    parser.add_argument("--baseline-low-risk-fraction", type=float, default=0.20)
    parser.add_argument("--shift-tail-fraction", type=float, default=0.10)
    parser.add_argument("--correct-top-risk-fraction", type=float, default=0.20)
    parser.add_argument("--min-relative-shift", type=float, default=0.05)
    parser.add_argument("--lambda-shift", type=float, default=1.0)
    parser.add_argument("--min-obs-all", type=int, default=50)
    parser.add_argument("--local-neighbor-count", type=int, default=30)
    parser.add_argument("--local-min-neighbors", type=int, default=10)
    parser.add_argument("--min-denominator", type=float, default=1.0)
    parser.add_argument("--exclude-partiality-too-small", action="store_true")
    parser.add_argument(
        "--skip-weak-down-nonpositive-iref",
        action="store_true",
        help="Do not target weak-down HKLs whose low-risk reference median I_ref is <= 0",
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--max-events", type=int, default=None, help="Limit crystals/chunks for smoke testing")
    args = parser.parse_args()

    for path_arg in ["stream", "scores", "unmerged"]:
        path = getattr(args, path_arg)
        if not path.exists():
            raise SystemExit(f"--{path_arg} not found: {path}")

    for name in [
        "baseline_low_risk_fraction",
        "shift_tail_fraction",
        "correct_top_risk_fraction",
    ]:
        value = float(getattr(args, name))
        if not (0.0 < value <= 1.0):
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 1]")

    for name in ["min_relative_shift", "lambda_shift", "min_denominator"]:
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")

    if args.min_relative_shift < 0.0:
        raise SystemExit("--min-relative-shift must be >= 0")
    if args.min_denominator <= 0.0:
        raise SystemExit("--min-denominator must be > 0")
    if args.min_obs_all < 1:
        raise SystemExit("--min-obs-all must be >= 1")
    if args.local_neighbor_count < 1:
        raise SystemExit("--local-neighbor-count must be >= 1")
    if args.local_min_neighbors < 1:
        raise SystemExit("--local-min-neighbors must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.max_events is not None and args.max_events < 1:
        raise SystemExit("--max-events must be >= 1 when provided")
    return args


def normalize_source(value: Any) -> str:
    return str(value).strip()


def normalize_event(value: Any) -> str:
    return str(value).strip()


def build_key(source: str, event: str, h: int, k: int, l: int) -> str:
    return f"{normalize_source(source)}\t{normalize_event(event)}\t{int(h)}\t{int(k)}\t{int(l)}"


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [col for col in SOURCE_COLUMNS if col in header]
    if not candidates:
        raise SystemExit(
            "Scores file does not contain a recognized source column. "
            f"Checked {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )
    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for col in candidates:
        if looks_like_source_filename(sample[col]):
            return col
    return candidates[0]


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q25": None, "median": None, "q75": None}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"q25": None, "median": None, "q75": None}
    return {
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def parse_numeric_reflection_line(line: str) -> tuple[int, int, int, float, float, str] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        intensity = float(parts[3])
        fifth = float(parts[4])
    except ValueError:
        return None
    flags = " ".join(parts[5:]) if len(parts) > 5 else ""
    return h, k, l, intensity, fifth, flags


def load_unmerged_observations(
    unmerged_path: Path,
    exclude_partiality_too_small: bool,
    max_events: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    stats = {
        "total_unmerged_observations": 0,
        "kept_unmerged_observations": 0,
        "excluded_partiality_too_small": 0,
        "duplicate_key_count": 0,
        "unmerged_crystals_seen": 0,
    }
    current_source = ""
    current_event = ""
    crystals_seen = 0

    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                if max_events is not None and crystals_seen >= int(max_events):
                    break
                crystals_seen += 1
                stats["unmerged_crystals_seen"] = crystals_seen
                current_source = ""
                current_event = ""
                continue

            if match := UNMERGED_FILENAME_RE.match(line):
                current_source = normalize_source(match.group(1))
                current_event = normalize_event(match.group(2) or "")
                continue

            parsed = parse_numeric_reflection_line(line)
            if parsed is None:
                continue

            h, k, l, intensity, _fifth, flags = parsed
            stats["total_unmerged_observations"] += 1
            if exclude_partiality_too_small and "partiality_too_small" in flags.lower():
                stats["excluded_partiality_too_small"] += 1
                continue

            rows.append(
                {
                    "source_filename": current_source,
                    "event": current_event,
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "I_obs": float(intensity),
                    "key": build_key(current_source, current_event, h, k, l),
                }
            )
            stats["kept_unmerged_observations"] += 1

    table = pd.DataFrame.from_records(rows)
    if not table.empty:
        stats["duplicate_key_count"] = int(table.loc[table.duplicated("key", keep=False), "key"].nunique())
    return table, stats


def load_scores(scores_path: Path, keys_filter: set[str]) -> tuple[pd.DataFrame, int]:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)
    required = [source_column, "event", "h", "k", "l", "graph_crowding_norm", "frame_axis_risk_norm"]
    missing = [col for col in required if col not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    missing_keys = set(keys_filter)
    usecols = list(dict.fromkeys(required))

    for chunk in pd.read_csv(scores_path, usecols=usecols, chunksize=1_000_000):
        total_rows += int(len(chunk))
        for col in ["h", "k", "l", "graph_crowding_norm", "frame_axis_risk_norm"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.dropna(subset=["h", "k", "l", "graph_crowding_norm", "frame_axis_risk_norm"])
        chunk[["h", "k", "l"]] = chunk[["h", "k", "l"]].astype("int64")
        chunk = chunk.rename(columns={source_column: "source_filename"})
        chunk["source_filename"] = chunk["source_filename"].map(normalize_source)
        chunk["event"] = chunk["event"].map(normalize_event)
        chunk["key"] = (
            chunk["source_filename"].astype(str)
            + "\t"
            + chunk["event"].astype(str)
            + "\t"
            + chunk["h"].astype(str)
            + "\t"
            + chunk["k"].astype(str)
            + "\t"
            + chunk["l"].astype(str)
        )
        chunk = chunk[chunk["key"].isin(missing_keys)]
        if chunk.empty:
            continue
        chunk["graph_frame_mean"] = 0.5 * chunk["graph_crowding_norm"] + 0.5 * chunk["frame_axis_risk_norm"]
        chunks.append(chunk[["key", "graph_frame_mean"]])
        missing_keys.difference_update(chunk["key"].astype(str))
        if not missing_keys:
            break

    if not chunks:
        return pd.DataFrame(columns=["key", "graph_frame_mean"]), total_rows
    return pd.concat(chunks, ignore_index=True), total_rows


def load_unit_cell_from_stream(stream_path: Path) -> dict[str, float]:
    in_unit_cell = False
    unit_lines: list[str] = []
    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if "Begin unit cell" in raw:
                in_unit_cell = True
                continue
            if in_unit_cell and "End unit cell" in raw:
                break
            if in_unit_cell:
                unit_lines.append(raw.strip())
            elif match := STREAM_CELL_RE.search(raw):
                a, b, c, alpha, beta, gamma = (float(match.group(i)) for i in range(1, 7))
                return {
                    "a": a,
                    "b": b,
                    "c": c,
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                }

    cell: dict[str, float] = {}
    for line in unit_lines:
        parts = line.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key in {"al", "alpha"}:
            key = "alpha"
        elif key in {"be", "beta"}:
            key = "beta"
        elif key in {"ga", "gamma"}:
            key = "gamma"
        if key not in {"a", "b", "c", "alpha", "beta", "gamma"}:
            continue
        for token in reversed(parts[1:]):
            try:
                cell[key] = float(token)
                break
            except ValueError:
                continue

    missing = [key for key in ["a", "b", "c", "alpha", "beta", "gamma"] if key not in cell]
    if missing:
        raise SystemExit(f"Could not parse unit cell from stream; missing {missing}")
    return {key: float(cell[key]) for key in ["a", "b", "c", "alpha", "beta", "gamma"]}


def reciprocal_basis(cell: dict[str, float]) -> np.ndarray:
    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha = math.radians(cell["alpha"])
    beta = math.radians(cell["beta"])
    gamma = math.radians(cell["gamma"])
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)
    if abs(sg) < 1e-12:
        raise SystemExit("Invalid unit cell: sin(gamma) is too small")

    avec = np.array([a, 0.0, 0.0], dtype=float)
    bvec = np.array([b * cg, b * sg, 0.0], dtype=float)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c * c - cx * cx - cy * cy
    cvec = np.array([cx, cy, math.sqrt(max(0.0, cz_sq))], dtype=float)
    volume = float(np.dot(avec, np.cross(bvec, cvec)))
    if abs(volume) < 1e-12:
        raise SystemExit("Invalid unit cell: volume is too small")
    astar = np.cross(bvec, cvec) / volume
    bstar = np.cross(cvec, avec) / volume
    cstar = np.cross(avec, bvec) / volume
    return np.vstack([astar, bstar, cstar])


def reciprocal_coords(hkl: np.ndarray, cell: dict[str, float]) -> np.ndarray:
    basis = reciprocal_basis(cell)
    return hkl.astype(float) @ basis


@dataclass
class Correction:
    i_ref: float
    direction: str


def compute_hkl_diagnostics(
    joined: pd.DataFrame,
    cell: dict[str, float],
    baseline_low_risk_fraction: float,
    shift_tail_fraction: float,
    correct_top_risk_fraction: float,
    min_relative_shift: float,
    min_obs_all: int,
    local_neighbor_count: int,
    local_min_neighbors: int,
    min_denominator: float,
    skip_weak_down_nonpositive_iref: bool,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Correction]]:
    rows: list[dict[str, object]] = []
    selected_corrections: dict[str, Correction] = {}

    valid = joined.dropna(subset=["I_obs", "graph_frame_mean", "h", "k", "l"]).copy()
    if valid.empty:
        empty = pd.DataFrame(
            columns=[
                "h",
                "k",
                "l",
                "local_class",
                "n_obs",
                "I_ref",
                "I_highrisk",
                "relative_shift",
                "local_neighbor_median_I_ref",
                "n_top_risk_selected",
                "n_shifted",
                "correction_direction",
            ]
        )
        return empty, {"eligible_signed_hkls": 0, "weak_down_target_hkls": 0, "strong_up_target_hkls": 0}, selected_corrections

    valid[["h", "k", "l"]] = valid[["h", "k", "l"]].astype("int64")
    valid["I_obs"] = pd.to_numeric(valid["I_obs"], errors="coerce")
    valid["graph_frame_mean"] = pd.to_numeric(valid["graph_frame_mean"], errors="coerce")
    valid = valid.dropna(subset=["I_obs", "graph_frame_mean"])
    valid = valid.sort_values(["h", "k", "l", "graph_frame_mean"], kind="mergesort")

    hkl_values = valid[["h", "k", "l"]].to_numpy(dtype=np.int64)
    i_values = valid["I_obs"].to_numpy(dtype=float)
    risk_values = valid["graph_frame_mean"].to_numpy(dtype=float)
    start_mask = np.empty(len(valid), dtype=bool)
    start_mask[0] = True
    start_mask[1:] = np.any(hkl_values[1:] != hkl_values[:-1], axis=1)
    starts = np.flatnonzero(start_mask)
    stops = np.r_[starts[1:], len(valid)]
    total_groups = int(len(starts))
    log(f"Preparing signed-HKL diagnostics for {total_groups:,} signed HKLs")

    start_time = time.monotonic()
    for group_idx, (start, stop) in enumerate(zip(starts, stops), start=1):
        if group_idx % max(1, int(progress_every)) == 0:
            elapsed = max(time.monotonic() - start_time, 1e-9)
            rate = group_idx / elapsed
            remaining = (total_groups - group_idx) / rate if rate > 0 else np.nan
            pct = 100.0 * group_idx / max(total_groups, 1)
            log(f"  processed {group_idx:,}/{total_groups:,} HKLs ({pct:.1f}%, {rate:.0f}/s, ETA {format_eta(remaining)})")

        n_obs = int(stop - start)
        if n_obs < int(min_obs_all):
            continue
        h, k, l = (int(x) for x in hkl_values[start])
        i_group = i_values[start:stop]
        risk_group = risk_values[start:stop]
        finite = np.isfinite(i_group) & np.isfinite(risk_group)
        if not bool(finite.all()):
            i_group = i_group[finite]
            risk_group = risk_group[finite]
            n_obs = int(len(i_group))
            if n_obs < int(min_obs_all):
                continue

        n_low = max(1, int(math.ceil(n_obs * baseline_low_risk_fraction)))
        n_high = max(1, int(math.ceil(n_obs * shift_tail_fraction)))
        i_ref = float(np.median(i_group[:n_low]))
        i_highrisk = float(np.median(i_group[-n_high:]))
        denom = max(abs(i_ref), float(min_denominator))
        relative_shift = float((i_highrisk - i_ref) / denom)
        rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "local_class": "unknown",
                "n_obs": n_obs,
                "I_ref": i_ref,
                "I_highrisk": i_highrisk,
                "relative_shift": relative_shift,
                "local_neighbor_median_I_ref": np.nan,
                "n_top_risk_selected": 0,
                "n_shifted": 0,
                "correction_direction": "none",
            }
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "h",
                "k",
                "l",
                "local_class",
                "n_obs",
                "I_ref",
                "I_highrisk",
                "relative_shift",
                "local_neighbor_median_I_ref",
                "n_top_risk_selected",
                "n_shifted",
                "correction_direction",
            ]
        )
        return empty, {"eligible_signed_hkls": 0, "weak_down_target_hkls": 0, "strong_up_target_hkls": 0}, selected_corrections

    per_hkl = pd.DataFrame.from_records(rows)
    log(f"Computing local reciprocal-space medians for {len(per_hkl):,} eligible signed HKLs")
    coords = reciprocal_coords(per_hkl[["h", "k", "l"]].to_numpy(dtype=float), cell)
    i_ref_values = per_hkl["I_ref"].to_numpy(dtype=float)
    neighbor_query_count = min(len(per_hkl), int(local_neighbor_count) + 1)
    local_medians = np.full(len(per_hkl), np.nan, dtype=float)

    if cKDTree is not None and len(per_hkl) > 0:
        tree = cKDTree(coords)
        _distances, neighbor_indices = tree.query(coords, k=neighbor_query_count, workers=-1)
        if neighbor_query_count == 1:
            neighbor_indices = neighbor_indices.reshape(-1, 1)
        for idx in range(len(per_hkl)):
            indices = np.atleast_1d(neighbor_indices[idx])
            indices = indices[indices != idx][: int(local_neighbor_count)]
            if len(indices) >= int(local_min_neighbors):
                local_medians[idx] = float(np.nanmedian(i_ref_values[indices]))
    else:
        for start in range(0, len(per_hkl), 1000):
            stop = min(start + 1000, len(per_hkl))
            distances = np.linalg.norm(coords[None, :, :] - coords[start:stop, None, :], axis=2)
            order = np.argpartition(distances, kth=neighbor_query_count - 1, axis=1)[:, :neighbor_query_count]
            for offset, indices in enumerate(order):
                idx = start + offset
                indices = indices[indices != idx]
                if len(indices) > int(local_neighbor_count):
                    nearest_order = np.argsort(distances[offset, indices])[: int(local_neighbor_count)]
                    indices = indices[nearest_order]
                if len(indices) >= int(local_min_neighbors):
                    local_medians[idx] = float(np.nanmedian(i_ref_values[indices]))

    per_hkl["local_neighbor_median_I_ref"] = local_medians
    known_local = np.isfinite(local_medians)
    per_hkl.loc[known_local, "local_class"] = np.where(
        i_ref_values[known_local] < local_medians[known_local],
        "weak",
        "strong",
    )

    joined_by_hkl = {
        (int(h), int(k), int(l)): group.dropna(subset=["graph_frame_mean", "I_obs"]).copy()
        for (h, k, l), group in joined.groupby(["h", "k", "l"], sort=False)
    }

    weak_down_target_hkls = 0
    strong_up_target_hkls = 0
    weak_down_nonpositive_iref_skipped = 0
    for idx, row in per_hkl.iterrows():
        direction = "none"
        if row["local_class"] == "weak" and float(row["relative_shift"]) > float(min_relative_shift):
            if skip_weak_down_nonpositive_iref and float(row["I_ref"]) <= 0.0:
                direction = "weak_down_skipped_nonpositive_iref"
                weak_down_nonpositive_iref_skipped += 1
                per_hkl.at[idx, "correction_direction"] = direction
                continue
            else:
                direction = "weak_down"
                weak_down_target_hkls += 1
        elif row["local_class"] == "strong" and float(row["relative_shift"]) < -float(min_relative_shift):
            direction = "strong_up"
            strong_up_target_hkls += 1
        if direction == "none":
            continue

        hk = (int(row["h"]), int(row["k"]), int(row["l"]))
        group = joined_by_hkl.get(hk)
        if group is None or group.empty:
            continue
        group = group.sort_values("graph_frame_mean", ascending=False)
        n_select = max(1, int(math.ceil(len(group) * float(correct_top_risk_fraction))))
        selected = group.head(n_select)
        per_hkl.at[idx, "n_top_risk_selected"] = int(len(selected))
        per_hkl.at[idx, "correction_direction"] = direction
        correction = Correction(i_ref=float(row["I_ref"]), direction=direction)
        for key in selected["key"].astype(str):
            selected_corrections[key] = correction

    diagnostics = {
        "eligible_signed_hkls": int(len(per_hkl)),
        "weak_down_target_hkls": int(weak_down_target_hkls),
        "strong_up_target_hkls": int(strong_up_target_hkls),
        "weak_down_nonpositive_iref_hkls_skipped": int(weak_down_nonpositive_iref_skipped),
    }
    return per_hkl, diagnostics, selected_corrections


@dataclass
class StreamReflection:
    line_idx: int
    h: int
    k: int
    l: int
    key: str
    intensity: float
    raw_line: str


def parse_stream_reflection(line: str) -> tuple[int, int, int, float] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
    except ValueError:
        return None


def collect_stream_reflections(
    stream_path: Path,
    max_events: int | None,
) -> tuple[list[str], list[StreamReflection], dict[str, int]]:
    lines: list[str] = []
    reflections: list[StreamReflection] = []
    stats = {"stream_crystals_seen": 0, "total_stream_reflections_seen": 0}

    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    crystals_seen = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            idx = len(lines)
            lines.append(raw)
            line = raw.rstrip("\n")

            if "Begin chunk" in line:
                chunk_source = ""
                chunk_event = ""
                in_reflections = False
                continue
            if match := STREAM_IMAGE_RE.match(line):
                chunk_source = normalize_source(match.group(1))
                if not in_crystal:
                    continue
            if match := STREAM_EVENT_RE.match(line):
                chunk_event = normalize_event(match.group(1))
                if not in_crystal:
                    continue

            if "Begin crystal" in line:
                in_crystal = True
                in_reflections = False
                current_source = chunk_source
                current_event = chunk_event
                crystals_seen += 1
                stats["stream_crystals_seen"] = crystals_seen
                continue
            if "End crystal" in line:
                in_crystal = False
                in_reflections = False
                if max_events is not None and crystals_seen >= int(max_events):
                    break
                continue

            if not in_crystal:
                continue
            if match := STREAM_IMAGE_RE.match(line):
                current_source = normalize_source(match.group(1))
                continue
            if match := STREAM_EVENT_RE.match(line):
                current_event = normalize_event(match.group(1))
                continue
            if "Reflections measured after indexing" in line:
                in_reflections = True
                continue
            if in_reflections and line.startswith("End of reflections"):
                in_reflections = False
                continue
            if in_reflections:
                if not line.strip() or line.lstrip().startswith("h "):
                    continue
                parsed = parse_stream_reflection(line)
                if parsed is None:
                    continue
                h, k, l, intensity = parsed
                stats["total_stream_reflections_seen"] += 1
                reflections.append(
                    StreamReflection(
                        line_idx=idx,
                        h=h,
                        k=k,
                        l=l,
                        key=build_key(current_source, current_event, h, k, l),
                        intensity=float(intensity),
                        raw_line=raw,
                    )
                )

    return lines, reflections, stats


def update_intensity_only(raw_line: str, new_intensity: float) -> str:
    newline = "\n" if raw_line.endswith("\n") else ""
    body = raw_line[:-1] if newline else raw_line
    parts = body.split()
    if len(parts) < 4:
        return raw_line
    parts[3] = f"{new_intensity:.4f}"
    return " ".join(parts) + newline


def main() -> int:
    args = parse_args()

    log("Loading unmerged observations")
    unmerged, unmerged_stats = load_unmerged_observations(
        args.unmerged,
        exclude_partiality_too_small=bool(args.exclude_partiality_too_small),
        max_events=args.max_events,
    )
    if unmerged.empty:
        raise SystemExit("No unmerged observations were loaded")

    log("Loading score table")
    scores, score_rows_read = load_scores(args.scores, set(unmerged["key"].astype(str)))
    if scores.empty:
        raise SystemExit("No matching score rows were loaded")

    log("Joining unmerged observations to scores")
    joined = unmerged.merge(scores, on="key", how="left", indicator=True)
    matched_mask = joined["_merge"] == "both"
    matched_observations = int(matched_mask.sum())
    unmatched_observations = int((~matched_mask).sum())
    if matched_observations == 0:
        raise SystemExit("No matched observations between unmerged data and scores")
    joined = joined.drop(columns=["_merge"])

    log("Loading stream and collecting reflection observations")
    stream_lines, stream_refs, stream_stats = collect_stream_reflections(args.stream, args.max_events)
    if not stream_refs:
        raise SystemExit("No reflections were parsed from the stream")

    log("Computing bidirectional HKL diagnostics")
    cell = load_unit_cell_from_stream(args.stream)
    per_hkl, hkl_diag, selected_corrections = compute_hkl_diagnostics(
        joined=joined.loc[matched_mask].copy(),
        cell=cell,
        baseline_low_risk_fraction=float(args.baseline_low_risk_fraction),
        shift_tail_fraction=float(args.shift_tail_fraction),
        correct_top_risk_fraction=float(args.correct_top_risk_fraction),
        min_relative_shift=float(args.min_relative_shift),
        min_obs_all=int(args.min_obs_all),
        local_neighbor_count=int(args.local_neighbor_count),
        local_min_neighbors=int(args.local_min_neighbors),
        min_denominator=float(args.min_denominator),
        skip_weak_down_nonpositive_iref=bool(args.skip_weak_down_nonpositive_iref),
        progress_every=int(args.progress_every),
    )

    log("Rewriting selected stream intensities")
    out_lines = list(stream_lines)
    total_changed = 0
    unchanged = 0
    weak_down_shifted = 0
    strong_up_shifted = 0
    changed_by_hkl: Counter[tuple[int, int, int]] = Counter()
    abs_changes: list[float] = []
    rel_changes: list[float] = []

    for ref in stream_refs:
        correction = selected_corrections.get(ref.key)
        if correction is None:
            unchanged += 1
            continue
        should_shift = (
            (correction.direction == "weak_down" and ref.intensity > correction.i_ref)
            or (correction.direction == "strong_up" and ref.intensity < correction.i_ref)
        )
        if not should_shift:
            unchanged += 1
            continue
        new_intensity = float(ref.intensity + float(args.lambda_shift) * (correction.i_ref - ref.intensity))
        out_lines[ref.line_idx] = update_intensity_only(ref.raw_line, new_intensity)
        delta = float(new_intensity - ref.intensity)
        abs_changes.append(abs(delta))
        rel_changes.append(abs(delta) / max(abs(ref.intensity), float(args.min_denominator)))
        total_changed += 1
        changed_by_hkl[(ref.h, ref.k, ref.l)] += 1
        if correction.direction == "weak_down":
            weak_down_shifted += 1
        elif correction.direction == "strong_up":
            strong_up_shifted += 1

    if not per_hkl.empty:
        shifted_counts = [
            int(changed_by_hkl.get((int(row.h), int(row.k), int(row.l)), 0))
            for row in per_hkl.itertuples(index=False)
        ]
        per_hkl["n_shifted"] = shifted_counts

    args.output_stream.parent.mkdir(parents=True, exist_ok=True)
    args.output_stream.write_text("".join(out_lines), encoding="utf-8")

    summary = {
        "stream": str(args.stream),
        "scores": str(args.scores),
        "unmerged": str(args.unmerged),
        "output_stream": str(args.output_stream),
        "risk_score": "graph_frame_mean = 0.5 * graph_crowding_norm + 0.5 * frame_axis_risk_norm",
        "baseline_low_risk_fraction": float(args.baseline_low_risk_fraction),
        "shift_tail_fraction": float(args.shift_tail_fraction),
        "correct_top_risk_fraction": float(args.correct_top_risk_fraction),
        "min_relative_shift": float(args.min_relative_shift),
        "lambda_shift": float(args.lambda_shift),
        "min_obs_all": int(args.min_obs_all),
        "local_neighbor_count": int(args.local_neighbor_count),
        "local_min_neighbors": int(args.local_min_neighbors),
        "min_denominator": float(args.min_denominator),
        "exclude_partiality_too_small": bool(args.exclude_partiality_too_small),
        "skip_weak_down_nonpositive_iref": bool(args.skip_weak_down_nonpositive_iref),
        "progress_every": int(args.progress_every),
        "max_events": int(args.max_events) if args.max_events is not None else None,
        **unmerged_stats,
        "score_rows_read": int(score_rows_read),
        "score_rows_matched_to_unmerged_keys": int(len(scores)),
        "matched_observations": matched_observations,
        "unmatched_observations": unmatched_observations,
        **stream_stats,
        **hkl_diag,
        "selected_observation_keys": int(len(selected_corrections)),
        "weak_down_shifted_observations": int(weak_down_shifted),
        "strong_up_shifted_observations": int(strong_up_shifted),
        "total_changed_observations": int(total_changed),
        "unchanged_observations": int(unchanged),
        "absolute_intensity_change": quantiles(abs_changes),
        "relative_intensity_change": quantiles(rel_changes),
        "version": "bidirectional_graph_frame_shift_to_lowrisk_v1",
    }

    outdir = args.output_stream.parent
    per_hkl_out = outdir / "per_hkl_bidirectional_graph_frame_shift.csv"
    diagnostics_out = outdir / "bidirectional_shift_diagnostics.csv"
    summary_out = outdir / "bidirectional_shift_summary.json"

    per_hkl_cols = [
        "h",
        "k",
        "l",
        "local_class",
        "n_obs",
        "I_ref",
        "I_highrisk",
        "relative_shift",
        "local_neighbor_median_I_ref",
        "n_top_risk_selected",
        "n_shifted",
        "correction_direction",
    ]
    per_hkl.loc[:, per_hkl_cols].to_csv(per_hkl_out, index=False)
    with diagnostics_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    writer.writerow([f"{key}.{subkey}", subvalue])
            else:
                writer.writerow([key, value])
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log(f"Wrote corrected stream: {args.output_stream}")
    log(f"Wrote diagnostics: {summary_out}, {per_hkl_out}, {diagnostics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
