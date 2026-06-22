#!/usr/bin/env python3
"""Model-free orientation-shift diagnostics for OriDyn/MFM300.

This script uses only model-free observation data:
- partialator unmerged observations (I_unmerged, partiality)
- OriDyn reflection scores (orientation-risk metrics)
- stream header unit cell (for local reciprocal-space neighborhood classification)
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


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

NONSELF_COMPONENT_COLUMNS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

METRICS_TO_TEST = [
    "nonself_mean",
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
    "nonself_max",
    "self_risk_norm",
    "S_dyn_geom",
    "sigma_dyn_rel",
]

UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether orientation-only metrics predict weak/strong sign-flip intensity "
            "redistribution in model-free observation data."
        )
    )
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator unmerged.hkl")
    parser.add_argument("--stream", required=True, type=Path, help="Original stream path (header unit cell only)")
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder")

    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument(
        "--baseline-top-partiality-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of highest-partiality observations per signed HKL used to compute "
            "I_baseline_hkl for local reciprocal weak/strong classification (default: 0.10)"
        ),
    )
    parser.add_argument("--min-obs-per-hkl", type=int, default=50)

    parser.add_argument(
        "--grouping",
        choices=["signed_hkl", "canonical"],
        default="signed_hkl",
        help="Group observations by signed hkl (default) or canonicalized HKL",
    )
    parser.add_argument(
        "--pointgroup",
        type=str,
        default="1",
        help="Point group for canonical grouping. Used only with --grouping canonical",
    )

    parser.add_argument(
        "--classification-mode",
        choices=["global_median", "local_reciprocal"],
        default="local_reciprocal",
    )
    parser.add_argument("--local-neighbor-count", type=int, default=100)
    parser.add_argument("--local-min-neighbors", type=int, default=30)

    parser.add_argument(
        "--exclude-partiality-too-small",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rows flagged partiality_too_small (default: true)",
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional unmerged-row limit for smoke tests")
    parser.add_argument("--scores-chunksize", type=int, default=1000000)

    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if not args.unmerged.exists():
        raise SystemExit(f"--unmerged not found: {args.unmerged}")
    if not args.stream.exists():
        raise SystemExit(f"--stream not found: {args.stream}")

    if not (0.0 < float(args.tail_fraction) < 0.5):
        raise SystemExit("--tail-fraction must be > 0 and < 0.5")
    if not (0.0 < float(args.baseline_top_partiality_fraction) <= 1.0):
        raise SystemExit("--baseline-top-partiality-fraction must be > 0 and <= 1")
    if int(args.min_obs_per_hkl) < 2:
        raise SystemExit("--min-obs-per-hkl must be >= 2")
    if int(args.local_neighbor_count) < 1:
        raise SystemExit("--local-neighbor-count must be >= 1")
    if int(args.local_min_neighbors) < 1:
        raise SystemExit("--local-min-neighbors must be >= 1")
    if int(args.local_min_neighbors) > int(args.local_neighbor_count):
        raise SystemExit("--local-min-neighbors must be <= --local-neighbor-count")
    if int(args.progress_every) < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.max_rows is not None and int(args.max_rows) < 1:
        raise SystemExit("--max-rows must be >= 1 when provided")
    if int(args.scores_chunksize) < 1:
        raise SystemExit("--scores-chunksize must be >= 1")

    if args.grouping == "canonical" and str(args.pointgroup) not in {"1", "4/mmm"}:
        raise SystemExit("Canonical mode currently supports --pointgroup 1 or 4/mmm")

    return args


def normalize_source(value: Any) -> str:
    return str(value).strip()


def normalize_event(value: Any) -> str:
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def hkl_text(h: int, k: int, l: int) -> str:
    return f"({h},{k},{l})"


def build_key(source: str, event: str, h: int, k: int, l: int) -> str:
    return f"{source}\t{event}\t{int(h)}\t{int(k)}\t{int(l)}"


def canonicalize_hkl(h: int, k: int, l: int, pointgroup: str) -> tuple[int, int, int]:
    if pointgroup == "1":
        return int(h), int(k), int(l)
    if pointgroup == "4/mmm":
        ah = abs(int(h))
        ak = abs(int(k))
        al = abs(int(l))
        return max(ah, ak), min(ah, ak), al
    raise ValueError(f"Unsupported pointgroup: {pointgroup}")


def parse_stream_unit_cell(stream_path: Path) -> dict[str, float]:
    """Read stream header and parse unit-cell parameters, then stop early."""
    log("Reading stream header unit cell")
    cell: dict[str, float] = {}

    def first_float(text: str) -> float | None:
        m = FLOAT_RE.search(text)
        if m is None:
            return None
        return float(m.group(0))

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            low = line.lower()
            if low.startswith("a ="):
                v = first_float(line)
                if v is not None:
                    cell["a"] = v
            elif low.startswith("b ="):
                v = first_float(line)
                if v is not None:
                    cell["b"] = v
            elif low.startswith("c ="):
                v = first_float(line)
                if v is not None:
                    cell["c"] = v
            elif low.startswith("al =") or low.startswith("alpha ="):
                v = first_float(line)
                if v is not None:
                    cell["alpha"] = v
            elif low.startswith("be =") or low.startswith("beta ="):
                v = first_float(line)
                if v is not None:
                    cell["beta"] = v
            elif low.startswith("ga =") or low.startswith("gamma ="):
                v = first_float(line)
                if v is not None:
                    cell["gamma"] = v

            if all(k in cell for k in ["a", "b", "c", "alpha", "beta", "gamma"]):
                log(f"Parsed unit cell by line {line_no:,}: {cell}")
                return cell

            if line.startswith("----- Begin chunk -----"):
                break

    missing = [k for k in ["a", "b", "c", "alpha", "beta", "gamma"] if k not in cell]
    raise SystemExit(f"Failed to parse unit cell from stream header. Missing: {missing}")


def reciprocal_basis_from_cell(cell: dict[str, float]) -> np.ndarray:
    """Return reciprocal basis matrix B with columns a*, b*, c* in 1/Angstrom."""
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
        raise SystemExit("Invalid unit cell: sin(gamma) too small")

    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b * cos_g, b * sin_g, 0.0], dtype=float)

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0.0:
        raise SystemExit("Invalid unit cell: computed c_z^2 <= 0")
    cz = math.sqrt(cz2)
    c_vec = np.array([cx, cy, cz], dtype=float)

    vol = float(np.dot(a_vec, np.cross(b_vec, c_vec)))
    if abs(vol) < 1e-15:
        raise SystemExit("Invalid unit cell: near-zero volume")

    a_star = np.cross(b_vec, c_vec) / vol
    b_star = np.cross(c_vec, a_vec) / vol
    c_star = np.cross(a_vec, b_vec) / vol
    return np.column_stack([a_star, b_star, c_star])


def parse_unmerged_observations(
    unmerged_path: Path,
    exclude_partiality_too_small: bool,
    max_rows: int | None,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    log("Stage 1/4: parsing unmerged observations")

    rows: list[dict[str, Any]] = []
    current_source = ""
    current_event = ""

    reflection_rows_seen = 0
    reflection_rows_kept = 0
    excluded_partiality_too_small = 0
    excluded_nonpositive_partiality = 0
    malformed_rows = 0
    truncated_by_max_rows = False

    with unmerged_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Crystal "):
                continue

            m = UNMERGED_FILENAME_RE.match(line)
            if m:
                current_source = normalize_source(m.group(1))
                current_event = normalize_event(m.group(2) or "")
                continue

            if line.lower().startswith("flagged:"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                i_unmerged = float(parts[3])
                partiality = float(parts[4])
            except ValueError:
                malformed_rows += 1
                continue

            reflection_rows_seen += 1
            flag_text = " ".join(parts[5:]).strip() if len(parts) > 5 else ""
            flag_lower = flag_text.lower()

            if exclude_partiality_too_small and ("partiality_too_small" in flag_lower):
                excluded_partiality_too_small += 1
                if reflection_rows_seen % int(progress_every) == 0:
                    log(
                        "Unmerged progress: "
                        f"seen={reflection_rows_seen:,}, kept={reflection_rows_kept:,}, "
                        f"excluded_partiality_too_small={excluded_partiality_too_small:,}"
                    )
                continue

            if (not np.isfinite(partiality)) or partiality <= 0.0:
                excluded_nonpositive_partiality += 1
                if reflection_rows_seen % int(progress_every) == 0:
                    log(
                        "Unmerged progress: "
                        f"seen={reflection_rows_seen:,}, kept={reflection_rows_kept:,}, "
                        f"excluded_partiality_too_small={excluded_partiality_too_small:,}"
                    )
                continue

            reflection_rows_kept += 1
            rows.append(
                {
                    "source_filename": str(current_source),
                    "event": str(current_event),
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "I_unmerged": float(i_unmerged),
                    "partiality": float(partiality),
                    "I_pr": float(i_unmerged * partiality),
                    "row_flag": flag_text,
                    "unmerged_line_number": int(line_no),
                }
            )

            if reflection_rows_seen % int(progress_every) == 0:
                log(
                    "Unmerged progress: "
                    f"seen={reflection_rows_seen:,}, kept={reflection_rows_kept:,}, "
                    f"excluded_partiality_too_small={excluded_partiality_too_small:,}"
                )

            if max_rows is not None and reflection_rows_kept >= int(max_rows):
                truncated_by_max_rows = True
                log(f"Reached --max-rows={int(max_rows):,}; stopping at line {line_no:,}")
                break

    cols = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "I_unmerged",
        "partiality",
        "I_pr",
        "row_flag",
        "unmerged_line_number",
    ]
    unmerged = pd.DataFrame.from_records(rows, columns=cols)

    if not unmerged.empty:
        unmerged["source_filename"] = unmerged["source_filename"].map(normalize_source)
        unmerged["event"] = unmerged["event"].map(normalize_event)
        unmerged["key"] = (
            unmerged["source_filename"].astype(str)
            + "\t"
            + unmerged["event"].astype(str)
            + "\t"
            + unmerged["h"].astype(str)
            + "\t"
            + unmerged["k"].astype(str)
            + "\t"
            + unmerged["l"].astype(str)
        )
    else:
        unmerged["key"] = pd.Series(dtype=str)

    dup_rows = int(unmerged.duplicated("key", keep=False).sum()) if not unmerged.empty else 0
    dup_keys = int(unmerged.loc[unmerged.duplicated("key", keep=False), "key"].nunique()) if not unmerged.empty else 0

    stats = {
        "unmerged_reflection_rows_seen": int(reflection_rows_seen),
        "unmerged_reflection_rows_kept": int(reflection_rows_kept),
        "excluded_partiality_too_small": int(excluded_partiality_too_small),
        "excluded_nonpositive_partiality": int(excluded_nonpositive_partiality),
        "unmerged_malformed_rows": int(malformed_rows),
        "unmerged_duplicate_key_rows": int(dup_rows),
        "unmerged_duplicate_keys": int(dup_keys),
        "unmerged_truncated_by_max_rows": bool(truncated_by_max_rows),
        "unmerged_unique_keys": int(unmerged["key"].nunique()) if not unmerged.empty else 0,
    }

    log(
        "Stage 1 complete: "
        f"seen={stats['unmerged_reflection_rows_seen']:,}, "
        f"kept={stats['unmerged_reflection_rows_kept']:,}, "
        f"excluded_partiality_too_small={stats['excluded_partiality_too_small']:,}, "
        f"duplicate_key_rows={stats['unmerged_duplicate_key_rows']:,}"
    )

    return unmerged, stats


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [column for column in SOURCE_COLUMNS if column in header]
    if not candidates:
        raise SystemExit(
            "Scores file does not contain a recognized source column. "
            f"Checked {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )

    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for column in candidates:
        if looks_like_source_filename(sample[column]):
            return column
    return candidates[0]


def load_scores_for_keys(
    scores_path: Path,
    keys: set[str],
    chunksize: int,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any], str, list[str]]:
    log("Stage 2/4: loading score rows and matching signed keys")

    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l"]
    missing_required = [c for c in required if c not in header]
    if missing_required:
        raise SystemExit(f"Scores file missing required columns: {missing_required}")

    metric_columns_available = [m for m in METRICS_TO_TEST if m in header]
    include_columns = list(dict.fromkeys([*required, *metric_columns_available, *NONSELF_COMPONENT_COLUMNS]))

    rows_read = 0
    matched_rows = 0
    chunk_count = 0
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(scores_path, usecols=[c for c in include_columns if c in header], chunksize=int(chunksize)):
        chunk_count += 1
        rows_read += len(chunk)

        chunk[source_column] = chunk[source_column].map(normalize_source)
        chunk["event"] = chunk["event"].map(normalize_event)
        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        chunk = chunk.dropna(subset=[source_column, "event", "h", "k", "l"])
        if chunk.empty:
            if rows_read % int(progress_every) < int(chunksize):
                log(f"Scores progress: rows_read={rows_read:,}, matched_rows={matched_rows:,}")
            continue

        chunk = chunk.astype({"h": "int64", "k": "int64", "l": "int64"})
        key_series = (
            chunk[source_column].astype(str)
            + "\t"
            + chunk["event"].astype(str)
            + "\t"
            + chunk["h"].astype(str)
            + "\t"
            + chunk["k"].astype(str)
            + "\t"
            + chunk["l"].astype(str)
        )

        mask = key_series.isin(keys)
        if bool(mask.any()):
            sub = chunk.loc[mask].copy()
            sub["key"] = key_series.loc[mask].to_numpy()
            chunks.append(sub)
            matched_rows += len(sub)

        if rows_read % int(progress_every) < int(chunksize):
            log(f"Scores progress: rows_read={rows_read:,}, matched_rows={matched_rows:,}")

    if not chunks:
        matched_table = pd.DataFrame(columns=["key", source_column, "event", "h", "k", "l", *metric_columns_available])
    else:
        matched_table = pd.concat(chunks, ignore_index=True)

    duplicate_rows = int(matched_table.duplicated("key", keep=False).sum()) if not matched_table.empty else 0
    duplicate_keys = int(matched_table.loc[matched_table.duplicated("key", keep=False), "key"].nunique()) if not matched_table.empty else 0

    if matched_table.empty:
        scores_by_key = pd.DataFrame(columns=["key", "source_filename", "event", "h", "k", "l", *METRICS_TO_TEST])
    else:
        matched_table = matched_table.rename(columns={source_column: "source_filename"})

        agg_map: dict[str, str] = {
            "source_filename": "first",
            "event": "first",
            "h": "first",
            "k": "first",
            "l": "first",
        }

        numeric_cols = [c for c in matched_table.columns if c not in {"key", "source_filename", "event", "h", "k", "l"}]
        for col in numeric_cols:
            matched_table[col] = pd.to_numeric(matched_table[col], errors="coerce")
            agg_map[col] = "median"

        scores_by_key = matched_table.groupby("key", as_index=False).agg(agg_map)

        for component in NONSELF_COMPONENT_COLUMNS:
            if component not in scores_by_key.columns:
                scores_by_key[component] = np.nan

        if "nonself_mean" not in scores_by_key.columns:
            scores_by_key["nonself_mean"] = scores_by_key[NONSELF_COMPONENT_COLUMNS].mean(axis=1, skipna=True)
        if "nonself_max" not in scores_by_key.columns:
            scores_by_key["nonself_max"] = scores_by_key[NONSELF_COMPONENT_COLUMNS].max(axis=1, skipna=True)

        for metric in METRICS_TO_TEST:
            if metric not in scores_by_key.columns:
                scores_by_key[metric] = np.nan

    stats = {
        "score_rows_read": int(rows_read),
        "score_rows_matching_unmerged_keys": int(len(matched_table)),
        "score_unique_matching_keys": int(scores_by_key["key"].nunique()) if not scores_by_key.empty else 0,
        "score_duplicate_key_rows": int(duplicate_rows),
        "score_duplicate_keys": int(duplicate_keys),
        "score_chunks_read": int(chunk_count),
    }

    missing_metrics = [m for m in METRICS_TO_TEST if m not in metric_columns_available and m not in {"nonself_mean", "nonself_max"}]

    log(
        "Stage 2 complete: "
        f"rows_read={stats['score_rows_read']:,}, matched_rows={stats['score_rows_matching_unmerged_keys']:,}, "
        f"unique_keys={stats['score_unique_matching_keys']:,}, duplicate_key_rows={stats['score_duplicate_key_rows']:,}"
    )

    return scores_by_key, stats, source_column, missing_metrics


def add_grouping_columns(df: pd.DataFrame, grouping: str, pointgroup: str) -> pd.DataFrame:
    out = df.copy()

    if grouping == "signed_hkl":
        out["group_h"] = out["h"].astype(int)
        out["group_k"] = out["k"].astype(int)
        out["group_l"] = out["l"].astype(int)
    elif grouping == "canonical":
        canon = out.apply(
            lambda r: canonicalize_hkl(int(r["h"]), int(r["k"]), int(r["l"]), pointgroup),
            axis=1,
            result_type="expand",
        )
        out["group_h"] = canon[0].astype(int)
        out["group_k"] = canon[1].astype(int)
        out["group_l"] = canon[2].astype(int)
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")

    out["group_hkl"] = out.apply(
        lambda r: hkl_text(int(r["group_h"]), int(r["group_k"]), int(r["group_l"])),
        axis=1,
    )
    return out


def compute_group_intensity_stats(
    matched: pd.DataFrame,
    min_obs_per_hkl: int,
    baseline_top_partiality_fraction: float,
) -> pd.DataFrame:
    group_cols = ["group_h", "group_k", "group_l", "group_hkl"]

    grp = (
        matched.groupby(group_cols, as_index=False)
        .agg(
            n_obs=("I_pr", "size"),
            I_hkl_median=("I_pr", "median"),
            h_signed_example=("h", "first"),
            k_signed_example=("k", "first"),
            l_signed_example=("l", "first"),
        )
    )

    baseline_fraction = float(baseline_top_partiality_fraction)
    sort_cols = [*group_cols, "partiality", "unmerged_line_number"]
    ordered = matched.sort_values(
        sort_cols,
        ascending=[True, True, True, True, False, True],
    ).copy()
    ordered["_baseline_rank"] = ordered.groupby(group_cols).cumcount()
    ordered["_baseline_keep_n"] = np.ceil(ordered.groupby(group_cols)["I_pr"].transform("size") * baseline_fraction)
    ordered["_baseline_keep_n"] = ordered["_baseline_keep_n"].clip(lower=1).astype(int)
    baseline_obs = ordered.loc[ordered["_baseline_rank"] < ordered["_baseline_keep_n"]]
    baseline = (
        baseline_obs.groupby(group_cols, as_index=False)
        .agg(
            I_baseline_hkl=("I_pr", "median"),
            baseline_n_obs=("I_pr", "size"),
        )
    )
    grp = grp.merge(baseline, on=group_cols, how="left")
    grp["baseline_top_partiality_fraction"] = baseline_fraction

    grp["eligible_nobs"] = grp["n_obs"] >= int(min_obs_per_hkl)
    grp["local_median_I"] = np.nan
    grp["local_intensity_ratio"] = np.nan
    grp["local_intensity_delta"] = np.nan
    grp["local_neighbor_count"] = 0
    grp["classification"] = "insufficient_obs"
    return grp


def classify_global_median(group_stats: pd.DataFrame, min_obs_per_hkl: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = group_stats.copy()
    eligible = out[out["n_obs"] >= int(min_obs_per_hkl)].copy()
    if eligible.empty:
        return out, {
            "global_median_I_hkl_median": None,
            "eligible_groups": 0,
            "classified_groups": 0,
            "local_groups_with_enough_neighbors": None,
        }

    global_median = float(pd.to_numeric(eligible["I_hkl_median"], errors="coerce").median())

    def classify_row(row: pd.Series) -> str:
        if int(row["n_obs"]) < int(min_obs_per_hkl):
            return "insufficient_obs"
        value = float(row["I_hkl_median"])
        if value < global_median:
            return "weak"
        if value > global_median:
            return "strong"
        return "tie"

    out["classification"] = out.apply(classify_row, axis=1)

    return out, {
        "global_median_I_hkl_median": float(global_median),
        "eligible_groups": int(len(eligible)),
        "classified_groups": int((out["classification"].isin(["weak", "strong"])) .sum()),
        "local_groups_with_enough_neighbors": None,
    }


def classify_local_reciprocal(
    group_stats: pd.DataFrame,
    cell: dict[str, float],
    local_neighbor_count: int,
    local_min_neighbors: int,
    min_obs_per_hkl: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = group_stats.copy()
    eligible = out[out["n_obs"] >= int(min_obs_per_hkl)].copy().reset_index(drop=False)

    if eligible.empty:
        return out, {
            "global_median_I_hkl_median": None,
            "eligible_groups": 0,
            "classified_groups": 0,
            "local_groups_with_enough_neighbors": 0,
        }

    B = reciprocal_basis_from_cell(cell)
    hkl = eligible[["group_h", "group_k", "group_l"]].to_numpy(dtype=float)
    g = hkl @ B.T
    i_baseline = eligible["I_baseline_hkl"].to_numpy(dtype=float)

    n = len(eligible)
    k = min(int(local_neighbor_count), max(0, n - 1))
    local_median = np.full(n, np.nan, dtype=float)
    local_count = np.zeros(n, dtype=int)

    for i in range(n):
        if k <= 0:
            continue
        diff = g - g[i]
        d2 = np.sum(diff * diff, axis=1)
        d2[i] = np.inf
        idx = np.argpartition(d2, k)[:k]
        vals = i_baseline[idx]
        vals = vals[np.isfinite(vals)]
        local_count[i] = int(len(vals))
        if len(vals) >= int(local_min_neighbors):
            local_median[i] = float(np.median(vals))

    eligible["local_median_I"] = local_median
    eligible["local_neighbor_count"] = local_count
    eligible["local_intensity_ratio"] = eligible["I_baseline_hkl"] / eligible["local_median_I"]
    eligible["local_intensity_delta"] = eligible["I_baseline_hkl"] - eligible["local_median_I"]

    def classify_row(row: pd.Series) -> str:
        if not np.isfinite(row["local_median_I"]):
            return "insufficient_neighbors"
        value = float(row["I_baseline_hkl"])
        if not np.isfinite(value):
            return "insufficient_baseline"
        ref = float(row["local_median_I"])
        if value < ref:
            return "weak"
        if value > ref:
            return "strong"
        return "tie"

    eligible["classification"] = eligible.apply(classify_row, axis=1)

    for row in eligible.itertuples(index=False):
        j = int(row.index)
        out.at[j, "local_median_I"] = float(row.local_median_I) if np.isfinite(row.local_median_I) else np.nan
        out.at[j, "local_intensity_ratio"] = float(row.local_intensity_ratio) if np.isfinite(row.local_intensity_ratio) else np.nan
        out.at[j, "local_intensity_delta"] = float(row.local_intensity_delta) if np.isfinite(row.local_intensity_delta) else np.nan
        out.at[j, "local_neighbor_count"] = int(row.local_neighbor_count)
        out.at[j, "classification"] = str(row.classification)

    return out, {
        "global_median_I_hkl_median": None,
        "eligible_groups": int(len(eligible)),
        "classified_groups": int((eligible["classification"].isin(["weak", "strong"])) .sum()),
        "local_groups_with_enough_neighbors": int(np.isfinite(local_median).sum()),
    }


def analyze_metric_shifts(
    matched: pd.DataFrame,
    group_stats: pd.DataFrame,
    metrics: list[str],
    tail_fraction: float,
    min_obs_per_hkl: int,
    classification_mode: str,
    grouping: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    join_cols = [
        "group_h",
        "group_k",
        "group_l",
        "group_hkl",
        "n_obs",
        "I_hkl_median",
        "I_baseline_hkl",
        "baseline_top_partiality_fraction",
        "baseline_n_obs",
        "classification",
        "local_median_I",
        "local_intensity_ratio",
        "local_intensity_delta",
        "local_neighbor_count",
    ]
    merged = matched.merge(group_stats[join_cols], on=["group_h", "group_k", "group_l", "group_hkl"], how="left")

    eligible = group_stats[
        (group_stats["n_obs"] >= int(min_obs_per_hkl))
        & (group_stats["classification"].isin(["weak", "strong"]))
    ].copy()
    eligible_keys = set(zip(eligible["group_h"].astype(int), eligible["group_k"].astype(int), eligible["group_l"].astype(int)))

    grouped_all = {
        key: grp.copy()
        for key, grp in merged.groupby(["group_h", "group_k", "group_l"], sort=False)
        if (int(key[0]), int(key[1]), int(key[2])) in eligible_keys
    }

    per_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for metric in metrics:
        analyzed_keys: set[tuple[int, int, int]] = set()
        zero_spread = 0

        for (gh, gk, gl), grp in grouped_all.items():
            cls = str(grp["classification"].iloc[0])
            i_hkl_median = float(grp["I_hkl_median"].iloc[0])
            i_baseline_hkl = float(grp["I_baseline_hkl"].iloc[0])

            use = grp[[metric, "I_pr"]].copy()
            use[metric] = pd.to_numeric(use[metric], errors="coerce")
            use["I_pr"] = pd.to_numeric(use["I_pr"], errors="coerce")
            use = use.dropna(subset=[metric, "I_pr"])
            n_obs_metric = int(len(use))
            if n_obs_metric < int(min_obs_per_hkl):
                continue

            tail_n = int(math.floor(float(tail_fraction) * float(n_obs_metric)))
            if tail_n < 1:
                continue
            if tail_n * 2 > n_obs_metric:
                tail_n = n_obs_metric // 2
            if tail_n < 1:
                continue

            sorted_use = use.sort_values([metric, "I_pr"], ascending=[True, True])
            low = sorted_use.head(tail_n)
            high = sorted_use.tail(tail_n)

            median_all = float(use["I_pr"].median())
            median_low = float(low["I_pr"].median())
            median_high = float(high["I_pr"].median())

            denom = abs(median_all)
            relative_shift = float((median_high - median_low) / denom) if np.isfinite(denom) and denom > 0.0 else np.nan

            p10 = float(pd.to_numeric(use[metric], errors="coerce").quantile(0.10))
            p90 = float(pd.to_numeric(use[metric], errors="coerce").quantile(0.90))
            spread = float(p90 - p10)
            if np.isfinite(spread) and spread == 0.0:
                zero_spread += 1

            if cls == "weak":
                expected = relative_shift
            elif cls == "strong":
                expected = -relative_shift if np.isfinite(relative_shift) else np.nan
            else:
                expected = np.nan

            analyzed_keys.add((int(gh), int(gk), int(gl)))

            row = {
                "h": int(grp["h"].iloc[0]),
                "k": int(grp["k"].iloc[0]),
                "l": int(grp["l"].iloc[0]),
                "classification": cls,
                "classification_mode": classification_mode,
                "grouping": grouping,
                "I_hkl_median": float(i_hkl_median),
                "I_baseline_hkl": float(i_baseline_hkl) if np.isfinite(i_baseline_hkl) else np.nan,
                "baseline_top_partiality_fraction": (
                    float(grp["baseline_top_partiality_fraction"].iloc[0])
                    if pd.notna(grp["baseline_top_partiality_fraction"].iloc[0])
                    else np.nan
                ),
                "baseline_n_obs": int(grp["baseline_n_obs"].iloc[0]) if pd.notna(grp["baseline_n_obs"].iloc[0]) else 0,
                "n_obs": int(n_obs_metric),
                "local_median_I": float(grp["local_median_I"].iloc[0]) if pd.notna(grp["local_median_I"].iloc[0]) else np.nan,
                "local_intensity_ratio": float(grp["local_intensity_ratio"].iloc[0]) if pd.notna(grp["local_intensity_ratio"].iloc[0]) else np.nan,
                "local_intensity_delta": float(grp["local_intensity_delta"].iloc[0]) if pd.notna(grp["local_intensity_delta"].iloc[0]) else np.nan,
                "local_neighbor_count": int(grp["local_neighbor_count"].iloc[0]) if pd.notna(grp["local_neighbor_count"].iloc[0]) else 0,
                "metric": str(metric),
                "metric_spread": float(spread) if np.isfinite(spread) else np.nan,
                "median_low_risk_I_pr": float(median_low),
                "median_high_risk_I_pr": float(median_high),
                "relative_shift": float(relative_shift) if np.isfinite(relative_shift) else np.nan,
                "expected_signed_shift": float(expected) if np.isfinite(expected) else np.nan,
            }

            if grouping == "canonical":
                row["h_canon"] = int(gh)
                row["k_canon"] = int(gk)
                row["l_canon"] = int(gl)

            per_rows.append(row)

        metric_df = pd.DataFrame([r for r in per_rows if r["metric"] == metric])
        n_hkls = int(len(metric_df))
        weak_n = int((metric_df["classification"] == "weak").sum()) if not metric_df.empty else 0
        strong_n = int((metric_df["classification"] == "strong").sum()) if not metric_df.empty else 0

        if metric_df.empty:
            med_expected = np.nan
            mean_expected = np.nan
            frac_correct = np.nan
            weak_shift = np.nan
            strong_shift = np.nan
            med_spread = np.nan
        else:
            es = pd.to_numeric(metric_df["expected_signed_shift"], errors="coerce")
            med_expected = float(es.median())
            mean_expected = float(es.mean())
            frac_correct = float((es > 0.0).mean())
            weak_shift = float(pd.to_numeric(metric_df.loc[metric_df["classification"] == "weak", "relative_shift"], errors="coerce").median())
            strong_shift = float(pd.to_numeric(metric_df.loc[metric_df["classification"] == "strong", "relative_shift"], errors="coerce").median())
            med_spread = float(pd.to_numeric(metric_df["metric_spread"], errors="coerce").median())

        n_missing_metric = int(len(eligible_keys) - len(analyzed_keys))

        summary_rows.append(
            {
                "metric": metric,
                "classification_mode": classification_mode,
                "grouping": grouping,
                "n_hkls": n_hkls,
                "weak_n": weak_n,
                "strong_n": strong_n,
                "median_expected_signed_shift": med_expected if np.isfinite(med_expected) else np.nan,
                "mean_expected_signed_shift": mean_expected if np.isfinite(mean_expected) else np.nan,
                "frac_correct_sign": frac_correct if np.isfinite(frac_correct) else np.nan,
                "weak_median_relative_shift": weak_shift if np.isfinite(weak_shift) else np.nan,
                "strong_median_relative_shift": strong_shift if np.isfinite(strong_shift) else np.nan,
                "median_metric_spread": med_spread if np.isfinite(med_spread) else np.nan,
                "n_missing_metric": n_missing_metric,
                "n_zero_spread": int(zero_spread),
            }
        )

    per_hkl = pd.DataFrame(per_rows)
    if not per_hkl.empty:
        per_hkl = per_hkl.sort_values(["metric", "classification", "h", "k", "l"]).reset_index(drop=True)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["median_expected_signed_shift", "frac_correct_sign"], ascending=[False, False]).reset_index(drop=True)

    extra = {
        "eligible_non_tie_hkls": int(len(eligible_keys)),
    }
    return summary, per_hkl, extra


def maybe_make_plots(summary_df: pd.DataFrame, per_hkl_df: pd.DataFrame, output_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log(f"Plotting skipped (matplotlib unavailable): {exc}")
        return out

    if not summary_df.empty:
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(summary_df))
        y = pd.to_numeric(summary_df["median_expected_signed_shift"], errors="coerce").to_numpy(dtype=float)
        plt.bar(x, y)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(x, summary_df["metric"].tolist(), rotation=45, ha="right")
        plt.ylabel("median_expected_signed_shift")
        plt.title("Median Expected Signed Shift by Metric")
        plt.tight_layout()
        p1 = output_root / "metric_median_expected_signed_shift.png"
        fig.savefig(p1, dpi=160)
        plt.close(fig)
        out["metric_bar_plot"] = str(p1)

    if not per_hkl_df.empty:
        data = []
        labels = []
        for metric in METRICS_TO_TEST:
            for cls, short in [("weak", "W"), ("strong", "S")]:
                vals = pd.to_numeric(
                    per_hkl_df.loc[(per_hkl_df["metric"] == metric) & (per_hkl_df["classification"] == cls), "relative_shift"],
                    errors="coerce",
                ).dropna()
                if vals.empty:
                    continue
                data.append(vals.to_numpy(dtype=float))
                labels.append(f"{metric}\n{short}")

        if data:
            fig = plt.figure(figsize=(max(12, len(data) * 0.45), 5))
            plt.boxplot(data, showfliers=False)
            plt.axhline(0.0, color="black", linewidth=1)
            plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
            plt.ylabel("relative_shift")
            plt.title("Weak vs Strong Relative Shift Distributions")
            plt.tight_layout()
            p2 = output_root / "weak_strong_relative_shift_boxplot.png"
            fig.savefig(p2, dpi=160)
            plt.close(fig)
            out["weak_strong_boxplot"] = str(p2)

        local_ratio = pd.to_numeric(per_hkl_df["local_intensity_ratio"], errors="coerce").dropna()
        if not local_ratio.empty:
            fig = plt.figure(figsize=(7, 4))
            plt.hist(local_ratio.to_numpy(dtype=float), bins=80)
            plt.axvline(1.0, color="black", linewidth=1)
            plt.xlabel("local_intensity_ratio")
            plt.ylabel("count")
            plt.title("Histogram of Local Intensity Ratio")
            plt.tight_layout()
            p3 = output_root / "local_intensity_ratio_hist.png"
            fig.savefig(p3, dpi=160)
            plt.close(fig)
            out["local_ratio_hist"] = str(p3)

    return out


def main() -> None:
    args = parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    metric_summary_path = args.output_root / "metric_summary.csv"
    per_hkl_path = args.output_root / "per_hkl_metric_shifts.csv"
    metadata_path = args.output_root / "run_metadata.json"

    log("Starting orientation-shift diagnostic")
    log(f"Output root: {args.output_root}")
    log(f"Output file: {metric_summary_path}")
    log(f"Output file: {per_hkl_path}")
    log(f"Output file: {metadata_path}")

    cell = parse_stream_unit_cell(args.stream)

    unmerged, unmerged_stats = parse_unmerged_observations(
        unmerged_path=args.unmerged,
        exclude_partiality_too_small=bool(args.exclude_partiality_too_small),
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )
    if unmerged.empty:
        raise SystemExit("No unmerged observations after filtering")

    keys = set(unmerged["key"].astype(str).tolist())
    scores_by_key, score_stats, source_column, missing_metrics = load_scores_for_keys(
        scores_path=args.scores,
        keys=keys,
        chunksize=args.scores_chunksize,
        progress_every=args.progress_every,
    )

    log("Stage 3/4: joining unmerged observations to scores")
    matched = unmerged.merge(scores_by_key, on="key", how="left", indicator=True, suffixes=("", "_score"))
    matched_rows = int((matched["_merge"] == "both").sum())
    unmatched_rows = int((matched["_merge"] != "both").sum())
    log(f"Matched observations: {matched_rows:,}")
    log(f"Unmatched observations: {unmatched_rows:,}")

    matched = matched.loc[matched["_merge"] == "both"].copy()
    matched = matched.drop(columns=["_merge"])
    if matched.empty:
        raise SystemExit("No matched observations after source+event+signed HKL join")

    matched = add_grouping_columns(matched, grouping=str(args.grouping), pointgroup=str(args.pointgroup))

    group_stats = compute_group_intensity_stats(
        matched,
        min_obs_per_hkl=int(args.min_obs_per_hkl),
        baseline_top_partiality_fraction=float(args.baseline_top_partiality_fraction),
    )

    if str(args.classification_mode) == "global_median":
        group_stats, class_meta = classify_global_median(group_stats, min_obs_per_hkl=int(args.min_obs_per_hkl))
        log(
            "Global classification: "
            f"eligible_groups={class_meta['eligible_groups']:,}, "
            f"classified_groups={class_meta['classified_groups']:,}, "
            f"global_median={class_meta['global_median_I_hkl_median']}"
        )
    else:
        group_stats, class_meta = classify_local_reciprocal(
            group_stats,
            cell=cell,
            local_neighbor_count=int(args.local_neighbor_count),
            local_min_neighbors=int(args.local_min_neighbors),
            min_obs_per_hkl=int(args.min_obs_per_hkl),
        )
        log(
            "Local reciprocal classification: "
            f"eligible_groups={class_meta['eligible_groups']:,}, "
            f"classified_groups={class_meta['classified_groups']:,}, "
            f"groups_with_enough_neighbors={class_meta['local_groups_with_enough_neighbors']:,}"
        )

    summary_df, per_hkl_df, extra = analyze_metric_shifts(
        matched=matched,
        group_stats=group_stats,
        metrics=list(METRICS_TO_TEST),
        tail_fraction=float(args.tail_fraction),
        min_obs_per_hkl=int(args.min_obs_per_hkl),
        classification_mode=str(args.classification_mode),
        grouping=str(args.grouping),
    )

    summary_df.to_csv(metric_summary_path, index=False)
    per_hkl_df.to_csv(per_hkl_path, index=False)

    plot_paths = maybe_make_plots(summary_df, per_hkl_df, args.output_root)

    baseline_n_obs = pd.to_numeric(group_stats["baseline_n_obs"], errors="coerce").dropna()

    metadata = {
        "inputs": {
            "scores": str(args.scores),
            "unmerged": str(args.unmerged),
            "stream": str(args.stream),
            "score_source_column_selected": str(source_column),
        },
        "stream_cell_parameters": cell,
        "outputs": {
            "metric_summary_csv": str(metric_summary_path),
            "per_hkl_metric_shifts_csv": str(per_hkl_path),
            "run_metadata_json": str(metadata_path),
            **plot_paths,
        },
        "baseline": {
            "I_baseline_hkl": "median(I_unmerged * partiality) over top-partiality observations per grouped HKL",
            "baseline_top_partiality_fraction": float(args.baseline_top_partiality_fraction),
            "baseline_n_obs": {
                "min": int(baseline_n_obs.min()) if not baseline_n_obs.empty else 0,
                "median": float(baseline_n_obs.median()) if not baseline_n_obs.empty else 0.0,
                "max": int(baseline_n_obs.max()) if not baseline_n_obs.empty else 0,
            },
        },
        "parameters": {
            "tail_fraction": float(args.tail_fraction),
            "baseline_top_partiality_fraction": float(args.baseline_top_partiality_fraction),
            "local_reciprocal_classification_intensity": "I_baseline_hkl",
            "min_obs_per_hkl": int(args.min_obs_per_hkl),
            "grouping": str(args.grouping),
            "pointgroup": str(args.pointgroup),
            "classification_mode": str(args.classification_mode),
            "local_neighbor_count": int(args.local_neighbor_count),
            "local_min_neighbors": int(args.local_min_neighbors),
            "exclude_partiality_too_small": bool(args.exclude_partiality_too_small),
            "progress_every": int(args.progress_every),
            "max_rows": int(args.max_rows) if args.max_rows is not None else None,
            "scores_chunksize": int(args.scores_chunksize),
        },
        "counts": {
            **unmerged_stats,
            **score_stats,
            "unmerged_rows_for_matching": int(len(unmerged)),
            "matched_rows": int(matched_rows),
            "unmatched_rows": int(unmatched_rows),
            "hkl_groups_total": int(len(group_stats)),
            "hkl_groups_eligible_nobs": int((group_stats["n_obs"] >= int(args.min_obs_per_hkl)).sum()),
            "hkl_groups_classified_weak_or_strong": int((group_stats["classification"].isin(["weak", "strong"])) .sum()),
            "global_median_I_hkl_median": class_meta.get("global_median_I_hkl_median"),
            "local_groups_with_enough_neighbors": class_meta.get("local_groups_with_enough_neighbors"),
            "eligible_non_tie_hkls": int(extra["eligible_non_tie_hkls"]),
            "metric_summary_rows": int(len(summary_df)),
            "per_hkl_metric_shift_rows": int(len(per_hkl_df)),
        },
        "missing_metrics_in_scores": missing_metrics,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("Stage 4/4 complete: outputs written")
    log(f"metric_summary.csv: {metric_summary_path}")
    log(f"per_hkl_metric_shifts.csv: {per_hkl_path}")
    log(f"run_metadata.json: {metadata_path}")


if __name__ == "__main__":
    main()
