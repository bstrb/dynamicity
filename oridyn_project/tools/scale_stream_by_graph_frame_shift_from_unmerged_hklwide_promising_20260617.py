#!/usr/bin/env python3
"""Archive script for the promising 2026-06-17 graph/frame stream scaling run.

This is intentionally the older HKL-wide behavior, kept to reproduce the
promising scaled stream:

  graph_frame_scaled_stream_lambda05/scaled_graph_frame_weak_posshift_lambda05.stream

It rewrites an input stream in place only conceptually: it writes a new stream
where all observations belonging to selected weak signed HKLs are down-scaled
based on a per-HKL comparison of low-risk and high-risk unmerged intensities
joined to OriDyn scores.

Scientific model:
  graph_frame_mean = 0.5 * graph_crowding_norm + 0.5 * frame_axis_risk_norm

Selection key for diagnostics:
  source_filename + event + signed h + signed k + signed l

Rewrite key for this archived promising behavior:
  signed h + signed k + signed l

The script does not run partialator or perform any full-dataset scaling. It only rewrites
the supplied stream using per-HKL scale factors inferred from the supplied unmerged HKL
and score table.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
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
UNMERGED_FLAGGED_RE = re.compile(r"^\s*Flagged:\s*(\S+)\s*$", re.IGNORECASE)
STREAM_IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
STREAM_EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)\s*$")


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale selected CrystFEL stream reflections using graph+frame risk estimates "
            "derived from unmerged intensities and OriDyn scores."
        )
    )
    parser.add_argument("--stream", required=True, type=Path, help="Input CrystFEL .stream file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file")
    parser.add_argument("--output-stream", required=True, type=Path, help="Scaled output .stream file")
    parser.add_argument("--baseline-low-risk-fraction", type=float, default=0.20)
    parser.add_argument("--shift-tail-fraction", type=float, default=0.10)
    parser.add_argument("--scale-top-risk-fraction", type=float, default=0.20)
    parser.add_argument("--min-relative-shift", type=float, default=0.05)
    parser.add_argument("--lambda-scale", type=float, default=0.5)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=1.0)
    parser.add_argument("--min-obs-all", type=int, default=50)
    parser.add_argument("--local-neighbor-count", type=int, default=30)
    parser.add_argument("--local-min-neighbors", type=int, default=10)
    parser.add_argument("--min-denominator", type=float, default=1.0)
    parser.add_argument(
        "--exclude-partiality-too-small",
        action="store_true",
        help="Exclude unmerged observations flagged partiality_too_small",
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--max-events", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument(
        "--scores-chunksize",
        type=int,
        default=1_000_000,
        help="Chunk size when reading the score CSV",
    )
    args = parser.parse_args()

    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if not args.unmerged.exists():
        raise SystemExit(f"--unmerged not found: {args.unmerged}")
    for name in [
        "baseline_low_risk_fraction",
        "shift_tail_fraction",
        "scale_top_risk_fraction",
        "lambda_scale",
        "min_scale",
        "max_scale",
    ]:
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")
    if not (0.0 < args.baseline_low_risk_fraction <= 1.0):
        raise SystemExit("--baseline-low-risk-fraction must be in (0, 1]")
    if not (0.0 < args.shift_tail_fraction <= 1.0):
        raise SystemExit("--shift-tail-fraction must be in (0, 1]")
    if not (0.0 < args.scale_top_risk_fraction <= 1.0):
        raise SystemExit("--scale-top-risk-fraction must be in (0, 1]")
    if args.min_obs_all < 1:
        raise SystemExit("--min-obs-all must be >= 1")
    if args.local_neighbor_count < 1:
        raise SystemExit("--local-neighbor-count must be >= 1")
    if args.local_min_neighbors < 1:
        raise SystemExit("--local-min-neighbors must be >= 1")
    if args.min_denominator <= 0.0:
        raise SystemExit("--min-denominator must be > 0")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.max_events is not None and args.max_events < 1:
        raise SystemExit("--max-events must be >= 1 when provided")
    if args.scores_chunksize < 1:
        raise SystemExit("--scores-chunksize must be >= 1")
    return args


def normalize_source(value: Any) -> str:
    return str(value).strip()


def normalize_event(value: Any) -> str:
    return str(value).strip()


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


def build_key(source: str, event: str, h: int, k: int, l: int) -> str:
    return f"{normalize_source(source)}\t{normalize_event(event)}\t{int(h)}\t{int(k)}\t{int(l)}"


def parse_hkl_line(line: str) -> tuple[int, int, int] | None:
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def parse_stream_reflection_hkl(line: str) -> tuple[int, int, int] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def parse_stream_reflection_numbers(line: str) -> tuple[float, float] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        intensity = float(parts[3])
        sigma = float(parts[4])
    except ValueError:
        return None
    return intensity, sigma


def load_unmerged_observations(
    unmerged_path: Path,
    exclude_partiality_too_small: bool,
    max_events: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    stats = {
        "total_unmerged_observations_seen": 0,
        "unmerged_observations_kept_for_diagnostics": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "duplicate_key_count": 0,
    }

    current_source = ""
    current_event = ""
    current_crystal_flagged = False

    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                current_source = ""
                current_event = ""
                current_crystal_flagged = False
                continue

            if match := UNMERGED_FILENAME_RE.match(line):
                current_source = normalize_source(match.group(1))
                current_event = normalize_event(match.group(2) or "")
                continue

            if match := UNMERGED_FLAGGED_RE.match(line):
                current_crystal_flagged = match.group(1).strip().lower() in {"yes", "y", "true", "1"}
                continue

            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                i_obs = float(parts[3])
                fifth = float(parts[4])
            except ValueError:
                continue

            stats["total_unmerged_observations_seen"] += 1
            flags = " ".join(parts[5:]).strip() if len(parts) > 5 else ""
            flags_lower = flags.lower()
            if current_crystal_flagged:
                stats["excluded_flagged_crystal"] += 1
                continue
            if exclude_partiality_too_small and "partiality_too_small" in flags_lower:
                stats["excluded_partiality_too_small"] += 1
                continue
            if "nan_esd" in flags_lower:
                stats["excluded_nan_esd"] += 1
                continue
            if not np.isfinite(fifth) or fifth <= 0.0:
                stats["excluded_nonpositive_partiality"] += 1
                continue

            rows.append(
                {
                    "source_filename": current_source,
                    "event": current_event,
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "I_obs": float(i_obs),
                    "fifth_column": float(fifth),
                    "reflection_flags": flags,
                    "key": build_key(current_source, current_event, h, k, l),
                }
            )
            stats["unmerged_observations_kept_for_diagnostics"] += 1

            if max_events is not None and stats["unmerged_observations_kept_for_diagnostics"] >= int(max_events):
                break

    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table, stats

    dup_mask = table.duplicated("key", keep=False)
    stats["duplicate_key_count"] = int(table.loc[dup_mask, "key"].nunique())
    return table, stats


def load_scores(scores_path: Path, chunksize: int) -> pd.DataFrame:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)
    required = [source_column, "event", "h", "k", "l", "graph_crowding_norm", "frame_axis_risk_norm"]
    missing = [c for c in required if c not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(scores_path, usecols=list(dict.fromkeys(required)), chunksize=int(chunksize)):
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
        chunk["graph_frame_mean"] = 0.5 * chunk["graph_crowding_norm"] + 0.5 * chunk["frame_axis_risk_norm"]
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["source_filename", "event", "h", "k", "l", "graph_crowding_norm", "frame_axis_risk_norm", "graph_frame_mean", "key"])
    return pd.concat(chunks, ignore_index=True)


def load_unit_cell_from_stream(stream_path: Path) -> dict[str, float]:
    begin_re = re.compile(r"Begin\s+unit\s+cell", re.IGNORECASE)
    end_re = re.compile(r"End\s+unit\s+cell", re.IGNORECASE)
    lines: list[str] = []
    in_block = False
    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if not in_block:
                if begin_re.search(raw):
                    in_block = True
                continue
            if end_re.search(raw):
                break
            lines.append(raw.strip())
    cell: dict[str, float] = {}
    for line in lines:
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) >= 3 and parts[0].lower() in {"a", "b", "c", "alpha", "beta", "gamma", "al", "be", "ga"}:
            key = parts[0].lower()
            if key == "al":
                key = "alpha"
            elif key == "be":
                key = "beta"
            elif key == "ga":
                key = "gamma"
            try:
                cell[key] = float(parts[2]) if parts[1] in {"=", ":"} else float(parts[1])
            except ValueError:
                continue
        elif len(parts) >= 2 and parts[0].lower() in {"a", "b", "c", "alpha", "beta", "gamma", "al", "be", "ga"}:
            key = parts[0].lower()
            if key == "al":
                key = "alpha"
            elif key == "be":
                key = "beta"
            elif key == "ga":
                key = "gamma"
            try:
                cell[key] = float(parts[1])
            except ValueError:
                continue
    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    missing = [k for k in required if k not in cell]
    if missing:
        raise SystemExit(f"Could not parse unit cell from stream header; missing {missing}")
    return {k: float(cell[k]) for k in required}


def reciprocal_metric_tensor(cell: dict[str, float]) -> np.ndarray:
    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha = math.radians(cell["alpha"])
    beta = math.radians(cell["beta"])
    gamma = math.radians(cell["gamma"])
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sa, sb, sg = math.sin(alpha), math.sin(beta), math.sin(gamma)
    volume = a * b * c * math.sqrt(max(1e-16, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg))
    astar = b * c * sa / volume
    bstar = a * c * sb / volume
    cstar = a * b * sg / volume
    # General reciprocal metric tensor in Cartesian-free form.
    return np.array(
        [
            [astar * astar, astar * bstar * cg, astar * cstar * cb],
            [astar * bstar * cg, bstar * bstar, bstar * cstar * ca],
            [astar * cstar * cb, bstar * cstar * ca, cstar * cstar],
        ],
        dtype=float,
    )


def reciprocal_length(h: int, k: int, l: int, gstar: np.ndarray) -> float:
    v = np.array([h, k, l], dtype=float)
    return float(math.sqrt(max(0.0, float(v @ gstar @ v))))


def reciprocal_coords(hkl: np.ndarray, cell: dict[str, float]) -> np.ndarray:
    """Return Cartesian reciprocal coordinates for orthogonal/tetragonal stream cells.

    The current MFM300 stream cell is tetragonal with 90 degree angles. For this
    use case, h/a, k/b, l/c gives the intended local reciprocal-space geometry.
    """

    return np.column_stack(
        [
            hkl[:, 0].astype(float) / float(cell["a"]),
            hkl[:, 1].astype(float) / float(cell["b"]),
            hkl[:, 2].astype(float) / float(cell["c"]),
        ]
    )


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


def compute_hkl_diagnostics(
    joined: pd.DataFrame,
    cell: dict[str, float],
    min_obs_all: int,
    baseline_low_risk_fraction: float,
    shift_tail_fraction: float,
    local_neighbor_count: int,
    local_min_neighbors: int,
    min_denominator: float,
    min_relative_shift: float,
    scale_top_risk_fraction: float,
    lambda_scale: float,
    min_scale: float,
    max_scale: float,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[tuple[int, int, int], float]]:
    gstar = reciprocal_metric_tensor(cell)
    rows: list[dict[str, object]] = []
    scale_factors: list[float] = []
    target_hkl_scales: dict[tuple[int, int, int], float] = {}

    work_rows: list[dict[str, object]] = []

    valid = joined.dropna(subset=["I_obs", "graph_frame_mean", "h", "k", "l"]).copy()
    if valid.empty:
        out = pd.DataFrame(columns=[
            "h", "k", "l", "n_obs", "I_ref", "I_high", "relative_shift", "local_class",
            "local_neighbor_median_I_ref", "graph_frame_scale", "n_scaled", "n_unscaled",
        ])
        return out, {"eligible_signed_hkls": 0, "target_weak_signed_hkls": 0, "scale_factors": []}, target_hkl_scales

    valid[["h", "k", "l"]] = valid[["h", "k", "l"]].astype("int64")
    valid["I_obs"] = pd.to_numeric(valid["I_obs"], errors="coerce")
    valid["graph_frame_mean"] = pd.to_numeric(valid["graph_frame_mean"], errors="coerce")
    valid = valid.sort_values(["h", "k", "l", "graph_frame_mean"], kind="mergesort")

    hkl_values = valid[["h", "k", "l"]].to_numpy(dtype=np.int64)
    i_obs_values = valid["I_obs"].to_numpy(dtype=float)
    risk_values = valid["graph_frame_mean"].to_numpy(dtype=float)
    group_start_mask = np.empty(len(valid), dtype=bool)
    group_start_mask[0] = True
    group_start_mask[1:] = np.any(hkl_values[1:] != hkl_values[:-1], axis=1)
    starts = np.flatnonzero(group_start_mask)
    stops = np.r_[starts[1:], len(valid)]
    total_groups = int(len(starts))
    log(f"  preparing shift diagnostics for {total_groups:,} signed HKLs")

    start_time = time.monotonic()
    for group_idx, (start, stop) in enumerate(zip(starts, stops), start=1):
        if group_idx % max(1, int(progress_every)) == 0:
            elapsed = max(time.monotonic() - start_time, 1e-9)
            rate = group_idx / elapsed
            remaining = (total_groups - group_idx) / rate if rate > 0 else np.nan
            pct = 100.0 * group_idx / max(total_groups, 1)
            log(
                f"  grouped {group_idx:,}/{total_groups:,} signed HKLs "
                f"({pct:.1f}%, {rate:.0f}/s, ETA {format_eta(remaining)})"
            )

        n_obs = int(stop - start)
        if n_obs <= 0:
            continue
        h, k, l = (int(x) for x in hkl_values[start])
        i_obs_group = i_obs_values[start:stop]
        risk_group = risk_values[start:stop]
        finite = np.isfinite(i_obs_group) & np.isfinite(risk_group)
        if not bool(finite.all()):
            i_obs_group = i_obs_group[finite]
            risk_group = risk_group[finite]
            n_obs = int(len(i_obs_group))
            if n_obs <= 0:
                continue

        # The table is already sorted by graph_frame_mean within each HKL.
        n_low = max(1, int(math.ceil(n_obs * float(baseline_low_risk_fraction))))
        n_high = max(1, int(math.ceil(n_obs * float(shift_tail_fraction))))
        i_ref = float(np.median(i_obs_group[:n_low]))
        i_high = float(np.median(i_obs_group[-n_high:]))
        denom = max(abs(i_ref), float(min_denominator))
        rel_shift = float((i_high - i_ref) / denom)
        work_rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "n_obs": n_obs,
                "n_low_ref": n_low,
                "n_high_shift": n_high,
                "I_ref": i_ref,
                "I_high": i_high,
                "relative_shift": rel_shift,
                "graph_frame_mean_median": float(np.median(risk_group)),
                "reciprocal_length": reciprocal_length(h, k, l, gstar),
            }
        )

    if not work_rows:
        out = pd.DataFrame(columns=[
            "h", "k", "l", "n_obs", "I_ref", "I_high", "relative_shift", "local_class",
            "local_neighbor_median_I_ref", "graph_frame_scale", "n_scaled", "n_unscaled",
        ])
        return out, {"eligible_signed_hkls": 0, "target_weak_signed_hkls": 0, "scale_factors": []}, target_hkl_scales

    work = pd.DataFrame.from_records(work_rows)
    log(f"  computing reciprocal-space local medians for {len(work):,} signed HKLs")
    work["local_neighbor_median_I_ref"] = np.nan
    work["local_class"] = "unknown"

    coords = reciprocal_coords(work[["h", "k", "l"]].to_numpy(dtype=float), cell)
    neighbor_query_count = min(len(work), int(local_neighbor_count) + 1)
    i_ref_values = pd.to_numeric(work["I_ref"], errors="coerce").to_numpy(dtype=float)

    if cKDTree is not None:
        tree = cKDTree(coords)
        _distances, neighbor_indices = tree.query(coords, k=neighbor_query_count, workers=-1)
        if neighbor_query_count == 1:
            neighbor_indices = neighbor_indices.reshape(-1, 1)
        local_medians = np.full(len(work), np.nan, dtype=float)
        for idx in range(len(work)):
            indices = np.atleast_1d(neighbor_indices[idx])
            indices = indices[indices != idx][: int(local_neighbor_count)]
            if len(indices) >= int(local_min_neighbors):
                local_medians[idx] = float(np.nanmedian(i_ref_values[indices]))
    else:
        local_medians = np.full(len(work), np.nan, dtype=float)
        for start in range(0, len(work), 1000):
            stop = min(start + 1000, len(work))
            if start and start % max(1000, int(progress_every)) == 0:
                log(f"  local-neighbor fallback processed {start:,}/{len(work):,} HKLs")
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

    work["local_neighbor_median_I_ref"] = local_medians
    known_local = np.isfinite(local_medians)
    work.loc[known_local, "local_class"] = np.where(
        i_ref_values[known_local] < local_medians[known_local],
        "weak",
        "strong",
    )

    work["eligible"] = (
        (work["n_obs"] >= int(min_obs_all))
        & (work["relative_shift"] > float(min_relative_shift))
        & (work["local_class"] == "weak")
    )
    work["graph_frame_scale"] = 1.0
    work["n_top_risk_selected_for_scaling"] = 0
    work["n_scaled"] = 0
    work["n_unscaled"] = work["n_obs"].astype(int)

    joined_by_hkl = {
        (int(h), int(k), int(l)): group.dropna(subset=["graph_frame_mean", "I_obs"]).copy()
        for (h, k, l), group in joined.groupby(["h", "k", "l"], sort=False)
    }
    eligible_rows = work.loc[work["eligible"]]
    log(f"  assigning scale factors for {len(eligible_rows):,} eligible weak signed HKLs")
    for eligible_idx, (idx, row) in enumerate(eligible_rows.iterrows(), start=1):
        if eligible_idx % max(1, int(progress_every)) == 0:
            log(f"  assigned {eligible_idx:,}/{len(eligible_rows):,} eligible scale factors")
        hk = (int(row["h"]), int(row["k"]), int(row["l"]))
        g = joined_by_hkl.get(hk)
        if g is None or g.empty:
            continue
        g = g.sort_values("graph_frame_mean", ascending=False)
        n_scale = max(1, int(math.ceil(len(g) * float(scale_top_risk_fraction))))
        selected = g.head(n_scale)
        i_ref = float(row["I_ref"])
        i_high = float(row["I_high"])
        if not np.isfinite(i_ref) or not np.isfinite(i_high) or abs(i_high) <= 0.0:
            continue
        raw_scale = i_ref / i_high
        scale = 1.0 - float(lambda_scale) * (1.0 - raw_scale)
        scale = float(min(float(max_scale), max(float(min_scale), scale)))
        work.at[idx, "graph_frame_scale"] = scale
        work.at[idx, "n_scaled"] = int(len(selected))
        work.at[idx, "n_unscaled"] = int(len(g) - len(selected))
        work.at[idx, "n_top_risk_selected_for_scaling"] = int(len(selected))
        scale_factors.append(scale)
        target_hkl_scales[hk] = scale

    diagnostics = {
        "eligible_signed_hkls": int(work["eligible"].sum()),
        "target_weak_signed_hkls": int((work["eligible"] & (work["graph_frame_scale"] < 1.0)).sum()),
        "selected_top_risk_observation_keys": int(eligible_rows["n_obs"].sum()) if "n_obs" in eligible_rows else 0,
        "hklwide_target_scale_count": int(len(target_hkl_scales)),
        "scale_factors": scale_factors,
    }
    work = work.drop(columns=["eligible", "reciprocal_length"])
    return work, diagnostics, target_hkl_scales


@dataclass
class StreamReflection:
    line_idx: int
    h: int
    k: int
    l: int
    source_filename: str
    event: str
    key: str
    intensity: float
    sigma: float
    raw_line: str


def collect_stream_reflections(stream_path: Path, max_events: int | None) -> tuple[list[str], list[StreamReflection], dict[str, int]]:
    lines = stream_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    reflections: list[StreamReflection] = []
    stats = {"total_stream_observations_seen": 0}
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_crystal = False
    in_reflections = False
    crystal_count = 0

    for idx, raw in enumerate(lines):
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
            in_crystal = False
            in_reflections = False
            current_source = chunk_source
            current_event = chunk_event
            crystal_count += 1
            if max_events is not None and crystal_count > int(max_events):
                break
            in_crystal = True
            continue
        if "End crystal" in line:
            in_crystal = False
            in_reflections = False
            continue
        if in_crystal:
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
                parsed = parse_stream_reflection_numbers(line)
                hkl = parse_stream_reflection_hkl(line)
                if parsed is None or hkl is None:
                    continue
                stats["total_stream_observations_seen"] += 1
                h, k, l = hkl
                intensity, sigma = parsed
                reflections.append(
                    StreamReflection(
                        line_idx=idx,
                        h=h,
                        k=k,
                        l=l,
                        source_filename=current_source,
                        event=current_event,
                        key=build_key(current_source, current_event, h, k, l),
                        intensity=intensity,
                        sigma=sigma,
                        raw_line=raw,
                    )
                )
    return lines, reflections, stats


def update_reflection_line(raw_line: str, new_intensity: float, new_sigma: float) -> str:
    newline = "\n" if raw_line.endswith("\n") else ""
    body = raw_line[:-1] if newline else raw_line
    parts = body.split()
    if len(parts) < 5:
        return raw_line
    # Preserve the existing column structure as much as practical by reusing token text
    # for everything except the intensity and sigma columns.
    parts[3] = f"{new_intensity:.4f}"
    parts[4] = f"{new_sigma:.4f}"
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
    scores = load_scores(args.scores, chunksize=int(args.scores_chunksize))
    if scores.empty:
        raise SystemExit("No score rows were loaded")

    log("Joining unmerged observations to scores")
    joined = unmerged.merge(scores, on="key", how="left", indicator=True, suffixes=("", "_score"))
    matched = joined["_merge"] == "both"
    matched_observations = int(matched.sum())
    unmatched_observations = int((~matched).sum())
    joined = joined.drop(columns=["_merge"])
    if matched_observations == 0:
        raise SystemExit("No matched observations between unmerged data and scores")

    log("Loading stream and collecting reflection observations")
    stream_lines, stream_refs, stream_stats = collect_stream_reflections(args.stream, args.max_events)
    if not stream_refs:
        raise SystemExit("No reflections were parsed from the stream")

    log("Computing HKL diagnostics and scale factors")
    unit_cell = load_unit_cell_from_stream(args.stream)
    per_hkl, diag, target_hkl_scales = compute_hkl_diagnostics(
        joined=joined.loc[matched].copy(),
        cell=unit_cell,
        min_obs_all=int(args.min_obs_all),
        baseline_low_risk_fraction=float(args.baseline_low_risk_fraction),
        shift_tail_fraction=float(args.shift_tail_fraction),
        local_neighbor_count=int(args.local_neighbor_count),
        local_min_neighbors=int(args.local_min_neighbors),
        min_denominator=float(args.min_denominator),
        min_relative_shift=float(args.min_relative_shift),
        scale_top_risk_fraction=float(args.scale_top_risk_fraction),
        lambda_scale=float(args.lambda_scale),
        min_scale=float(args.min_scale),
        max_scale=float(args.max_scale),
        progress_every=int(args.progress_every),
    )

    log("Rewriting stream reflections")
    out_lines = list(stream_lines)
    scaled_count = 0
    unscaled_count = 0
    scale_values: list[float] = []
    for ref in stream_refs:
        scale = float(target_hkl_scales.get((ref.h, ref.k, ref.l), 1.0))
        if scale != 1.0:
            out_lines[ref.line_idx] = update_reflection_line(ref.raw_line, ref.intensity * scale, ref.sigma * scale)
            scaled_count += 1
            scale_values.append(scale)
        else:
            unscaled_count += 1

    args.output_stream.parent.mkdir(parents=True, exist_ok=True)
    args.output_stream.write_text("".join(out_lines), encoding="utf-8")

    outdir = args.output_stream.parent
    scale_summary = {
        "stream": str(args.stream),
        "scores": str(args.scores),
        "unmerged": str(args.unmerged),
        "output_stream": str(args.output_stream),
        "baseline_low_risk_fraction": float(args.baseline_low_risk_fraction),
        "shift_tail_fraction": float(args.shift_tail_fraction),
        "scale_top_risk_fraction": float(args.scale_top_risk_fraction),
        "min_relative_shift": float(args.min_relative_shift),
        "lambda_scale": float(args.lambda_scale),
        "min_scale": float(args.min_scale),
        "max_scale": float(args.max_scale),
        "min_obs_all": int(args.min_obs_all),
        "local_neighbor_count": int(args.local_neighbor_count),
        "local_min_neighbors": int(args.local_min_neighbors),
        "min_denominator": float(args.min_denominator),
        "progress_every": int(args.progress_every),
        "max_events": int(args.max_events) if args.max_events is not None else None,
        "scores_chunksize": int(args.scores_chunksize),
        "exclude_partiality_too_small": bool(args.exclude_partiality_too_small),
        **unmerged_stats,
        "score_rows_read": int(len(scores)),
        "matched_observations": matched_observations,
        "unmatched_observations": unmatched_observations,
        "total_stream_observations_seen": int(stream_stats["total_stream_observations_seen"]),
        "scaled_stream_observations": int(scaled_count),
        "unscaled_stream_observations": int(unscaled_count),
        **diag,
        "scale_factor_min": float(np.min(scale_values)) if scale_values else 1.0,
        "scale_factor_median": float(np.median(scale_values)) if scale_values else 1.0,
        "scale_factor_max": float(np.max(scale_values)) if scale_values else 1.0,
        "version": "graph_frame_shift_from_unmerged_hklwide_promising_archive_20260617",
    }

    per_hkl_out = outdir / "per_hkl_graph_frame_shift.csv"
    diag_out = outdir / "scale_diagnostics.csv"
    summary_out = outdir / "scale_summary.json"

    per_hkl_cols = [
        "h",
        "k",
        "l",
        "n_obs",
        "n_low_ref",
        "n_high_shift",
        "I_ref",
        "I_high",
        "relative_shift",
        "local_class",
        "local_neighbor_median_I_ref",
        "graph_frame_scale",
        "n_top_risk_selected_for_scaling",
        "n_scaled",
        "n_unscaled",
    ]
    per_hkl.loc[:, per_hkl_cols].to_csv(per_hkl_out, index=False)

    diag_rows = [
        {"metric": key, "value": value}
        for key, value in scale_summary.items()
        if not isinstance(value, (dict, list))
    ]
    pd.DataFrame.from_records(diag_rows).to_csv(diag_out, index=False)
    summary_out.write_text(json.dumps(scale_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log(f"Wrote scaled stream: {args.output_stream}")
    log(f"Wrote diagnostics: {summary_out}, {per_hkl_out}, {diag_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
