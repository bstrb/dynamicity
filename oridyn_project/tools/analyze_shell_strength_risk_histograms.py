#!/usr/bin/env python3
"""Analyze shell-wise weak/strong residual shifts versus OriDyn risk.

This script joins CrystFEL partialator --unmerged-output observations with
OriDyn reflection scores using signed HKL and source/event keys:

    source_filename + event + h + k + l

It treats unmerged column 5 as partiality and computes:

    I_pr = I_unmerged * partiality

Outputs are written under --output-root:
- shell_hkl_classification.csv
- shell_observation_residuals.csv
- shell_strength_summary.csv
- plots/*.png
- README_shell_strength_diagnostics.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare weak/strong HKL residual behavior across resolution shells "
            "versus OriDyn risk."
        )
    )
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output directory for shell-strength diagnostics",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=None,
        help="Optional run_metadata.json containing unit-cell parameters",
    )
    parser.add_argument(
        "--cell",
        nargs=6,
        type=float,
        metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
        help="Fallback unit cell if metadata does not provide one",
    )
    parser.add_argument(
        "--shell-edges",
        nargs="+",
        type=float,
        default=[1.8, 1.4, 1.1, 0.9, 0.7, 0.55, 0.4],
        help="Resolution shell edges in Angstrom, descending order",
    )
    parser.add_argument("--risk-column", default="S_dyn_geom", help="Risk column in scores table")
    parser.add_argument("--score-chunksize", type=int, default=1_000_000, help="CSV chunksize for scores")
    parser.add_argument("--min-obs-per-hkl", type=int, default=20)
    parser.add_argument("--strength-weak-low-q", type=float, default=0.10)
    parser.add_argument("--strength-weak-high-q", type=float, default=0.30)
    parser.add_argument("--strength-strong-low-q", type=float, default=0.70)
    parser.add_argument("--strength-strong-high-q", type=float, default=0.90)
    parser.add_argument("--risk-low-quantile", type=float, default=0.25)
    parser.add_argument("--risk-high-quantile", type=float, default=0.75)
    parser.add_argument("--representative-hkls-per-class", type=int, default=3)
    parser.add_argument("--hist-bins", type=int, default=60)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--smoke-max-reflections",
        type=int,
        default=None,
        help="Limit number of eligible unmerged observations for smoke tests",
    )
    parser.add_argument(
        "--smoke-max-score-rows",
        type=int,
        default=None,
        help="Limit score rows loaded for smoke tests",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Parse/load/join only and print diagnostics; skip summaries and plots",
    )
    return parser.parse_args()


def normalize_source(value: object) -> str:
    return str(value).strip()


def normalize_event(value: object) -> str:
    return str(value).strip()


def source_basename(value: object) -> str:
    return Path(str(value).strip()).name


def looks_like_source_filename(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(1000)
    if values.empty:
        return False
    return bool(values.str.contains(r"\.h5\b|/|\\", regex=True).any())


def choose_score_source_column(scores_path: Path, header: list[str]) -> str:
    candidates = [column for column in SOURCE_COLUMNS if column in header]
    if not candidates:
        raise SystemExit(
            "Score file does not contain a recognized source column. "
            f"Checked {list(SOURCE_COLUMNS)}. Available columns: {header}"
        )
    sample = pd.read_csv(scores_path, usecols=candidates, nrows=1000)
    for column in candidates:
        if looks_like_source_filename(sample[column]):
            return column
    return candidates[0]


def flag_token_mask(text: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))


def parse_unmerged_observations(
    unmerged_path: Path,
    smoke_max_reflections: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []

    stats = {
        "rows_parsed": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "eligible_rows": 0,
        "kept_rows": 0,
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
                current_crystal_flagged = False
                continue

            filename_match = UNMERGED_FILENAME_RE.match(line)
            if filename_match:
                current_source = normalize_source(filename_match.group(1))
                current_event = normalize_event(filename_match.group(2) or "")
                continue

            flagged_match = UNMERGED_FLAGGED_RE.match(line)
            if flagged_match:
                current_crystal_flagged = flagged_match.group(1).strip().lower() in {"yes", "y", "true", "1"}
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
                continue

            stats["rows_parsed"] += 1
            reflection_flag = " ".join(parts[5:]).strip()

            if current_crystal_flagged:
                stats["excluded_flagged_crystal"] += 1
                continue
            if flag_token_mask(reflection_flag, "partiality_too_small"):
                stats["excluded_partiality_too_small"] += 1
                continue
            if flag_token_mask(reflection_flag, "nan_esd"):
                stats["excluded_nan_esd"] += 1
                continue
            if not np.isfinite(partiality) or partiality <= 0.0:
                stats["excluded_nonpositive_partiality"] += 1
                continue

            stats["eligible_rows"] += 1

            rows.append(
                {
                    "source": current_source,
                    "event": current_event,
                    "h": h,
                    "k": k,
                    "l": l,
                    "I_unmerged": i_unmerged,
                    "partiality": partiality,
                    "I_pr": i_unmerged * partiality,
                    "reflection_flag": reflection_flag,
                    "crystal_flagged": current_crystal_flagged,
                }
            )
            stats["kept_rows"] += 1

            if smoke_max_reflections is not None and stats["kept_rows"] >= int(smoke_max_reflections):
                break

    table = pd.DataFrame.from_records(rows)
    if table.empty:
        return table, stats

    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["source_norm"] = table["source"]
    table["source_basename"] = table["source"].map(source_basename)
    table["event_norm"] = table["event"]
    table["h"] = table["h"].astype("int64")
    table["k"] = table["k"].astype("int64")
    table["l"] = table["l"].astype("int64")
    return table, stats


def duplicate_rows(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = df.duplicated(key_columns, keep=False)
    return df.loc[mask].copy()


def duplicate_summary(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=key_columns + ["n_duplicates"])
    out = (
        df.groupby(key_columns, as_index=False)
        .size()
        .rename(columns={"size": "n_duplicates"})
        .sort_values("n_duplicates", ascending=False)
    )
    return out


def write_duplicate_diagnostics(path: Path, label: str, dup_rows: pd.DataFrame, key_columns: list[str]) -> None:
    summary = duplicate_summary(dup_rows, key_columns)
    dup_rows = dup_rows.copy()
    dup_rows.insert(0, "duplicate_set", label)
    summary.insert(0, "duplicate_set", label)
    summary.insert(1, "record_type", "summary")
    dup_rows.insert(1, "record_type", "row")

    for col in summary.columns:
        if col not in dup_rows.columns:
            dup_rows[col] = np.nan
    for col in dup_rows.columns:
        if col not in summary.columns:
            summary[col] = np.nan

    out = pd.concat([summary[dup_rows.columns], dup_rows], ignore_index=True)
    out.to_csv(path, index=False)


def duplicated_column_names(df: pd.DataFrame) -> list[str]:
    dups: list[str] = []
    seen: set[str] = set()
    for col in df.columns[df.columns.duplicated(keep=False)]:
        name = str(col)
        if name not in seen:
            seen.add(name)
            dups.append(name)
    return dups


def cleanup_duplicated_columns_keep_first(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    dup_names = duplicated_column_names(df)
    if dup_names:
        print(
            f"WARNING: duplicate column labels found in {label}: {dup_names}. "
            "Dropping duplicates with keep='first'.",
            file=sys.stderr,
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df, dup_names


def load_scores(
    scores_path: Path,
    risk_column: str,
    chunksize: int,
    smoke_max_score_rows: int | None = None,
) -> tuple[pd.DataFrame, str]:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l", risk_column]
    missing = [col for col in required if col not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    optional_cols = ["sigma_dyn_rel"]
    keep_columns = [source_column, "event", "h", "k", "l", risk_column]
    for c in optional_cols:
        if c in header:
            keep_columns.append(c)
    keep_columns = list(dict.fromkeys(keep_columns))

    chunks: list[pd.DataFrame] = []
    loaded_rows = 0

    reader = pd.read_csv(scores_path, usecols=keep_columns, chunksize=chunksize)
    for chunk in reader:
        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        chunk = chunk.dropna(subset=["h", "k", "l"])
        chunk[["h", "k", "l"]] = chunk[["h", "k", "l"]].astype("int64")

        if smoke_max_score_rows is not None:
            remaining = int(smoke_max_score_rows) - loaded_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        if not chunk.empty:
            chunks.append(chunk)
            loaded_rows += int(len(chunk))

        if smoke_max_score_rows is not None and loaded_rows >= int(smoke_max_score_rows):
            break

    if not chunks:
        return pd.DataFrame(), source_column

    table = pd.concat(chunks, ignore_index=True)
    table = table.rename(columns={source_column: "source"})
    table["source"] = table["source"].map(normalize_source)
    table["event"] = table["event"].map(normalize_event)
    table["source_norm"] = table["source"]
    table["source_basename"] = table["source"].map(source_basename)
    table["event_norm"] = table["event"]
    table[risk_column] = pd.to_numeric(table[risk_column], errors="coerce")
    if "sigma_dyn_rel" in table.columns:
        table["sigma_dyn_rel"] = pd.to_numeric(table["sigma_dyn_rel"], errors="coerce")

    return table, source_column


def join_unmerged_with_scores(
    unmerged: pd.DataFrame,
    scores: pd.DataFrame,
    risk_column: str,
) -> tuple[pd.DataFrame, dict[str, int], str, dict[str, object]]:
    key = ["source_norm", "event_norm", "h", "k", "l"]
    join_debug: dict[str, object] = {}

    unmerged, dup_unmerged_cols = cleanup_duplicated_columns_keep_first(unmerged, "unmerged")
    scores, dup_score_cols = cleanup_duplicated_columns_keep_first(scores, "scores")

    non_payload_cols = {"source", "event", "source_norm", "event_norm", "source_basename"}
    payload_candidates = [col for col in scores.columns if col not in non_payload_cols]
    removed_from_payload = [col for col in payload_candidates if col in key]
    payload_cols = [col for col in payload_candidates if col not in key]
    score_subset_cols = list(dict.fromkeys(key + payload_cols))
    scores_subset = scores.loc[:, score_subset_cols].copy()

    join_debug["duplicated_score_column_names_before_cleanup"] = dup_score_cols
    join_debug["duplicated_unmerged_column_names_before_cleanup"] = dup_unmerged_cols
    join_debug["final_merge_key"] = key
    join_debug["payload_column_count"] = int(len(payload_cols))
    join_debug["hkl_removed_from_payload_cols"] = bool(any(c in {"h", "k", "l"} for c in removed_from_payload))
    join_debug["payload_cols_removed_because_in_key"] = list(dict.fromkeys(removed_from_payload))

    print(
        "JOIN_DEBUG "
        + json.dumps(
            {
                "duplicated_score_column_names_before_cleanup": dup_score_cols,
                "duplicated_unmerged_column_names_before_cleanup": dup_unmerged_cols,
                "final_merge_key": key,
                "payload_column_count": int(len(payload_cols)),
                "hkl_removed_from_payload_cols": bool(any(c in {"h", "k", "l"} for c in removed_from_payload)),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )

    merged = unmerged.merge(
        scores_subset,
        on=key,
        how="left",
        indicator=True,
    )
    merged["match_mode"] = np.where(merged["_merge"] == "both", "primary", "missing")
    merged = merged.drop(columns=["_merge"])

    stats = {
        "unmerged_rows": int(len(unmerged)),
        "score_rows_considered": int(len(scores)),
        "primary_matches": int((merged["match_mode"] == "primary").sum()),
        "basename_matches": 0,
        "missing_after_join": int((merged["match_mode"] == "missing").sum()),
    }

    note = ""
    if stats["missing_after_join"] > 0:
        basename_key = ["source_basename", "event_norm", "h", "k", "l"]
        score_basename_dups = duplicate_rows(scores, basename_key)
        if score_basename_dups.empty:
            missing_idx = merged.index[merged["match_mode"] == "missing"]
            if len(missing_idx) > 0:
                left = merged.loc[
                    missing_idx,
                    [
                        "source",
                        "event",
                        "source_basename",
                        "event_norm",
                        "h",
                        "k",
                        "l",
                        "I_unmerged",
                        "partiality",
                        "I_pr",
                        "reflection_flag",
                        "crystal_flagged",
                        "source_norm",
                    ],
                ].copy()

                score_basename_cols = list(dict.fromkeys(basename_key + payload_cols))
                score_basename = scores.loc[:, score_basename_cols].copy()
                fallback = left.merge(score_basename, on=basename_key, how="left", indicator=True)
                got = fallback["_merge"] == "both"
                if got.any():
                    for col in payload_cols:
                        merged.loc[missing_idx, col] = fallback[col].values
                    matched_idx = missing_idx[got.to_numpy()]
                    merged.loc[matched_idx, "match_mode"] = "basename"
                    stats["basename_matches"] = int(got.sum())
                    stats["missing_after_join"] = int((merged["match_mode"] == "missing").sum())
                    note = "basename fallback applied to unmatched primary keys"
        else:
            note = "basename fallback skipped because basename keys are non-unique in score table"

    merged[risk_column] = pd.to_numeric(merged.get(risk_column), errors="coerce")

    return merged, stats, note, join_debug


def robust_mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def robust_skew(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 3:
        return np.nan
    q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90])
    denom = q90 - q10
    if abs(denom) < 1e-12:
        return np.nan
    return float((q90 + q10 - 2.0 * q50) / denom)


def maybe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def parse_unit_cell_dict(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    if not all(key in candidate for key in required):
        return None
    try:
        return (
            float(candidate["a"]),
            float(candidate["b"]),
            float(candidate["c"]),
            float(candidate["alpha"]),
            float(candidate["beta"]),
            float(candidate["gamma"]),
        )
    except (TypeError, ValueError):
        return None


def load_unit_cell_from_metadata(path: Path | None) -> tuple[float, float, float, float, float, float] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(data, dict):
        direct = parse_unit_cell_dict(data)
        if direct is not None:
            return direct

        for key in ("unit_cell", "cell", "unit_cell_parameters"):
            value = data.get(key)
            if isinstance(value, dict):
                parsed = parse_unit_cell_dict(value)
                if parsed is not None:
                    return parsed

        for parent_key in ("run_params", "parameters", "config", "metadata"):
            parent = data.get(parent_key)
            if isinstance(parent, dict):
                direct_parent = parse_unit_cell_dict(parent)
                if direct_parent is not None:
                    return direct_parent
                for key in ("unit_cell", "cell", "unit_cell_parameters"):
                    value = parent.get(key)
                    if isinstance(value, dict):
                        parsed = parse_unit_cell_dict(value)
                        if parsed is not None:
                            return parsed

    return None


def resolve_unit_cell(
    cli_cell: list[float] | None,
    metadata_path: Path | None,
    scores_path: Path,
) -> tuple[tuple[float, float, float, float, float, float], str]:
    if cli_cell is not None:
        if len(cli_cell) != 6:
            raise SystemExit("--cell requires exactly 6 numbers: a b c alpha beta gamma")
        return tuple(float(x) for x in cli_cell), "cli"

    cell = load_unit_cell_from_metadata(metadata_path)
    if cell is not None:
        return cell, f"metadata:{metadata_path}"

    auto_meta = scores_path.parent / "run_metadata.json"
    if metadata_path is None and auto_meta.exists():
        cell = load_unit_cell_from_metadata(auto_meta)
        if cell is not None:
            return cell, f"metadata:{auto_meta}"

    raise SystemExit(
        "Could not extract unit-cell parameters from metadata. "
        "Provide --cell a b c alpha beta gamma."
    )


def reciprocal_metric_tensor(cell: tuple[float, float, float, float, float, float]) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell

    if min(a, b, c) <= 0.0:
        raise SystemExit("Invalid unit cell: a,b,c must be > 0")

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)

    g = np.array(
        [
            [a * a, a * b * cos_g, a * c * cos_b],
            [a * b * cos_g, b * b, b * c * cos_a],
            [a * c * cos_b, b * c * cos_a, c * c],
        ],
        dtype=float,
    )

    det = float(np.linalg.det(g))
    if not np.isfinite(det) or det <= 0.0:
        raise SystemExit("Invalid unit cell metric tensor (non-positive determinant).")

    return np.linalg.inv(g)


def d_spacing_from_hkl(h: int, k: int, l: int, g_star: np.ndarray) -> float:
    if h == 0 and k == 0 and l == 0:
        return np.nan
    vec = np.array([float(h), float(k), float(l)], dtype=float)
    inv_d_sq = float(vec.T @ g_star @ vec)
    if not np.isfinite(inv_d_sq) or inv_d_sq <= 0.0:
        return np.nan
    return float(1.0 / math.sqrt(inv_d_sq))


def validate_shell_edges(shell_edges: list[float]) -> list[float]:
    if len(shell_edges) < 2:
        raise SystemExit("Need at least two shell edges.")
    edges = [float(x) for x in shell_edges]
    for i in range(len(edges) - 1):
        if not edges[i] > edges[i + 1]:
            raise SystemExit("--shell-edges must be strictly descending, e.g. 1.8 1.4 ... 0.4")
    return edges


def shell_pairs(shell_edges: list[float]) -> list[tuple[float, float]]:
    return [(shell_edges[i], shell_edges[i + 1]) for i in range(len(shell_edges) - 1)]


def format_shell_label(high_d: float, low_d: float) -> str:
    return f"{high_d:.3f}-{low_d:.3f}A"


def shell_slug(label: str) -> str:
    return label.replace("A", "A").replace("-", "_to_").replace(".", "p")


def assign_shell(d_spacing: float, shell_edges: list[float]) -> tuple[str | None, float | None, float | None]:
    if not np.isfinite(d_spacing):
        return None, None, None

    pairs = shell_pairs(shell_edges)
    for idx, (high_d, low_d) in enumerate(pairs):
        is_last = idx == (len(pairs) - 1)
        if is_last:
            in_shell = (d_spacing <= high_d) and (d_spacing >= low_d)
        else:
            in_shell = (d_spacing <= high_d) and (d_spacing > low_d)
        if in_shell:
            return format_shell_label(high_d, low_d), high_d, low_d
    return None, None, None


def compute_hkl_classification(
    matched: pd.DataFrame,
    g_star: np.ndarray,
    shell_edges: list[float],
    min_obs_per_hkl: int,
    weak_low_q: float,
    weak_high_q: float,
    strong_low_q: float,
    strong_high_q: float,
) -> pd.DataFrame:
    grouped = matched.groupby(["h", "k", "l"], as_index=False)
    hkl_table = grouped.agg(
        n_obs=("I_pr", "size"),
        median_I_pr=("I_pr", "median"),
    )

    mad_table = matched.groupby(["h", "k", "l"])["I_pr"].apply(robust_mad).reset_index(name="MAD_I_pr")
    hkl_table = hkl_table.merge(mad_table, on=["h", "k", "l"], how="left")

    hkl_table["d_spacing"] = hkl_table.apply(
        lambda r: d_spacing_from_hkl(int(r["h"]), int(r["k"]), int(r["l"]), g_star),
        axis=1,
    )

    shell_labels: list[str | None] = []
    shell_highs: list[float | None] = []
    shell_lows: list[float | None] = []
    for d_val in hkl_table["d_spacing"].to_numpy(dtype=float):
        label, high_d, low_d = assign_shell(float(d_val), shell_edges)
        shell_labels.append(label)
        shell_highs.append(high_d)
        shell_lows.append(low_d)

    hkl_table["shell_label"] = shell_labels
    hkl_table["shell_d_high"] = shell_highs
    hkl_table["shell_d_low"] = shell_lows

    hkl_table = hkl_table.loc[hkl_table["n_obs"] >= int(min_obs_per_hkl)].copy()
    hkl_table = hkl_table.loc[hkl_table["shell_label"].notna()].copy()

    if hkl_table.empty:
        return hkl_table

    hkl_table["weak_q_low"] = np.nan
    hkl_table["weak_q_high"] = np.nan
    hkl_table["strong_q_low"] = np.nan
    hkl_table["strong_q_high"] = np.nan
    hkl_table["strength_class"] = "other"

    for shell_label, shell_group in hkl_table.groupby("shell_label"):
        idx = shell_group.index
        med = shell_group["median_I_pr"]

        w_lo = float(med.quantile(weak_low_q))
        w_hi = float(med.quantile(weak_high_q))
        s_lo = float(med.quantile(strong_low_q))
        s_hi = float(med.quantile(strong_high_q))

        hkl_table.loc[idx, "weak_q_low"] = w_lo
        hkl_table.loc[idx, "weak_q_high"] = w_hi
        hkl_table.loc[idx, "strong_q_low"] = s_lo
        hkl_table.loc[idx, "strong_q_high"] = s_hi

        weak_mask = (hkl_table.loc[idx, "median_I_pr"] >= w_lo) & (hkl_table.loc[idx, "median_I_pr"] <= w_hi)
        strong_mask = (hkl_table.loc[idx, "median_I_pr"] >= s_lo) & (hkl_table.loc[idx, "median_I_pr"] <= s_hi)
        overlap = weak_mask & strong_mask
        weak_mask = weak_mask & (~overlap)
        strong_mask = strong_mask & (~overlap)

        weak_idx = idx[weak_mask.to_numpy()]
        strong_idx = idx[strong_mask.to_numpy()]

        hkl_table.loc[weak_idx, "strength_class"] = "weak"
        hkl_table.loc[strong_idx, "strength_class"] = "strong"

    return hkl_table


def compute_observation_residuals(
    matched: pd.DataFrame,
    hkl_table: pd.DataFrame,
    risk_column: str,
    risk_low_quantile: float,
    risk_high_quantile: float,
    eps: float,
) -> pd.DataFrame:
    keep_hkl_cols = [
        "h",
        "k",
        "l",
        "n_obs",
        "median_I_pr",
        "MAD_I_pr",
        "d_spacing",
        "shell_label",
        "shell_d_high",
        "shell_d_low",
        "strength_class",
        "weak_q_low",
        "weak_q_high",
        "strong_q_low",
        "strong_q_high",
    ]

    obs = matched.merge(hkl_table[keep_hkl_cols], on=["h", "k", "l"], how="inner")
    if obs.empty:
        return obs

    obs[risk_column] = pd.to_numeric(obs[risk_column], errors="coerce")

    obs["residual"] = obs["I_pr"] - obs["median_I_pr"]
    denom_rel = np.maximum(np.abs(obs["median_I_pr"].to_numpy(dtype=float)), float(eps))
    denom_rob = np.maximum(obs["MAD_I_pr"].to_numpy(dtype=float), float(eps))

    obs["relative_residual"] = obs["residual"].to_numpy(dtype=float) / denom_rel
    obs["robust_residual"] = obs["residual"].to_numpy(dtype=float) / denom_rob
    obs["abs_robust_residual"] = np.abs(obs["robust_residual"])

    q_low = obs.groupby(["h", "k", "l"])[risk_column].transform(lambda s: float(s.quantile(risk_low_quantile)))
    q_high = obs.groupby(["h", "k", "l"])[risk_column].transform(lambda s: float(s.quantile(risk_high_quantile)))
    obs["risk_q_low"] = q_low
    obs["risk_q_high"] = q_high

    obs["risk_group"] = "mid"
    obs.loc[obs[risk_column] <= obs["risk_q_low"], "risk_group"] = "low"
    obs.loc[obs[risk_column] >= obs["risk_q_high"], "risk_group"] = "high"

    obs["hkl"] = (
        "(" + obs["h"].astype(str) + "," + obs["k"].astype(str) + "," + obs["l"].astype(str) + ")"
    )
    return obs


def summarize_shell_strength(
    obs: pd.DataFrame,
    shell_edges: list[float],
    risk_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for high_d, low_d in shell_pairs(shell_edges):
        shell_label = format_shell_label(high_d, low_d)
        for strength_class in ("weak", "strong"):
            subset = obs.loc[
                (obs["shell_label"] == shell_label) & (obs["strength_class"] == strength_class)
            ].copy()

            low = subset.loc[subset["risk_group"] == "low"]
            high = subset.loc[subset["risk_group"] == "high"]

            med_low_rob = float(low["robust_residual"].median()) if not low.empty else np.nan
            med_high_rob = float(high["robust_residual"].median()) if not high.empty else np.nan
            med_low_rel = float(low["relative_residual"].median()) if not low.empty else np.nan
            med_high_rel = float(high["relative_residual"].median()) if not high.empty else np.nan

            rows.append(
                {
                    "shell_label": shell_label,
                    "shell_d_high": high_d,
                    "shell_d_low": low_d,
                    "strength_class": strength_class,
                    "n_hkls": int(subset[["h", "k", "l"]].drop_duplicates().shape[0]),
                    "n_obs": int(len(subset)),
                    "median_low_risk_robust_residual": med_low_rob,
                    "median_high_risk_robust_residual": med_high_rob,
                    "high_minus_low_median_robust_residual": (
                        med_high_rob - med_low_rob
                        if np.isfinite(med_high_rob) and np.isfinite(med_low_rob)
                        else np.nan
                    ),
                    "median_low_risk_relative_residual": med_low_rel,
                    "median_high_risk_relative_residual": med_high_rel,
                    "high_minus_low_median_relative_residual": (
                        med_high_rel - med_low_rel
                        if np.isfinite(med_high_rel) and np.isfinite(med_low_rel)
                        else np.nan
                    ),
                    "spearman_risk_vs_robust_residual": maybe_spearman(
                        subset[risk_column], subset["robust_residual"]
                    ),
                    "spearman_risk_vs_abs_robust_residual": maybe_spearman(
                        subset[risk_column], subset["abs_robust_residual"]
                    ),
                    "robust_skew_high_risk_residuals": robust_skew(high["robust_residual"]),
                    "robust_skew_low_risk_residuals": robust_skew(low["robust_residual"]),
                }
            )

    return pd.DataFrame(rows)


def plot_histograms_by_shell_strength(
    obs: pd.DataFrame,
    shell_edges: list[float],
    plots_dir: Path,
    hist_bins: int,
) -> list[str]:
    files_written: list[str] = []

    for high_d, low_d in shell_pairs(shell_edges):
        shell_label = format_shell_label(high_d, low_d)
        for strength_class in ("weak", "strong"):
            subset = obs.loc[
                (obs["shell_label"] == shell_label) & (obs["strength_class"] == strength_class)
            ]
            if subset.empty:
                continue

            low = subset.loc[subset["risk_group"] == "low"]
            high = subset.loc[subset["risk_group"] == "high"]

            shell_name = shell_slug(shell_label)

            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
            if not low.empty:
                ax.hist(
                    low["robust_residual"],
                    bins=hist_bins,
                    alpha=0.55,
                    color="#2e77bb",
                    label=f"low risk (n={len(low)})",
                )
            if not high.empty:
                ax.hist(
                    high["robust_residual"],
                    bins=hist_bins,
                    alpha=0.55,
                    color="#d43c2f",
                    label=f"high risk (n={len(high)})",
                )
            ax.set_title(f"Robust residual: {shell_label} {strength_class}")
            ax.set_xlabel("robust_residual")
            ax.set_ylabel("count")
            ax.legend(loc="best")
            out = plots_dir / f"{shell_name}_{strength_class}_hist_robust_residual.png"
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            files_written.append(str(out))

            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
            if not low.empty:
                ax.hist(
                    low["relative_residual"],
                    bins=hist_bins,
                    alpha=0.55,
                    color="#2e77bb",
                    label=f"low risk (n={len(low)})",
                )
            if not high.empty:
                ax.hist(
                    high["relative_residual"],
                    bins=hist_bins,
                    alpha=0.55,
                    color="#d43c2f",
                    label=f"high risk (n={len(high)})",
                )
            ax.set_title(f"Relative residual: {shell_label} {strength_class}")
            ax.set_xlabel("relative_residual")
            ax.set_ylabel("count")
            ax.legend(loc="best")
            out = plots_dir / f"{shell_name}_{strength_class}_hist_relative_residual.png"
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            files_written.append(str(out))

    return files_written


def plot_representative_i_pr(
    obs: pd.DataFrame,
    hkl_table: pd.DataFrame,
    shell_edges: list[float],
    plots_dir: Path,
    representative_hkls_per_class: int,
    hist_bins: int,
) -> list[str]:
    files_written: list[str] = []

    for high_d, low_d in shell_pairs(shell_edges):
        shell_label = format_shell_label(high_d, low_d)
        shell_name = shell_slug(shell_label)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=150, sharey=True)
        wrote_any = False

        for idx, strength_class in enumerate(("weak", "strong")):
            ax = axes[idx]
            class_hkls = hkl_table.loc[
                (hkl_table["shell_label"] == shell_label) & (hkl_table["strength_class"] == strength_class)
            ].sort_values("n_obs", ascending=False)

            rep = class_hkls.head(int(representative_hkls_per_class))
            if rep.empty:
                ax.text(0.5, 0.5, "no representative HKLs", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{shell_label} {strength_class}")
                ax.set_xlabel("I_pr")
                ax.set_ylabel("count")
                continue

            for _, row in rep.iterrows():
                h = int(row["h"])
                k = int(row["k"])
                l = int(row["l"])
                values = obs.loc[
                    (obs["shell_label"] == shell_label)
                    & (obs["h"] == h)
                    & (obs["k"] == k)
                    & (obs["l"] == l),
                    "I_pr",
                ]
                if values.empty:
                    continue
                ax.hist(
                    values,
                    bins=hist_bins,
                    histtype="step",
                    linewidth=1.5,
                    label=f"({h},{k},{l}) n={len(values)}",
                )
                wrote_any = True

            ax.set_title(f"{shell_label} {strength_class}")
            ax.set_xlabel("I_pr")
            ax.set_ylabel("count")
            if ax.has_data():
                ax.legend(loc="best", fontsize=7)

        if wrote_any:
            out = plots_dir / f"{shell_name}_representative_I_pr_weak_vs_strong.png"
            fig.tight_layout()
            fig.savefig(out)
            files_written.append(str(out))
        plt.close(fig)

    return files_written


def plot_summary_shift(summary: pd.DataFrame, shell_edges: list[float], plots_dir: Path) -> str | None:
    if summary.empty:
        return None

    ordered_shells = [format_shell_label(high_d, low_d) for high_d, low_d in shell_pairs(shell_edges)]
    pivot = summary.pivot(
        index="shell_label",
        columns="strength_class",
        values="high_minus_low_median_robust_residual",
    )
    pivot = pivot.reindex(ordered_shells)

    if pivot.empty:
        return None

    x = np.arange(len(pivot.index), dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)

    if "weak" in pivot.columns:
        ax.plot(
            x,
            pivot["weak"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.8,
            color="#2e77bb",
            label="weak",
        )
    if "strong" in pivot.columns:
        ax.plot(
            x,
            pivot["strong"].to_numpy(dtype=float),
            marker="s",
            linewidth=1.8,
            color="#d43c2f",
            label="strong",
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("high - low median robust residual")
    ax.set_title("High-minus-low robust residual shift across shells")
    ax.legend(loc="best")

    out = plots_dir / "summary_high_minus_low_median_robust_residual_by_shell.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
    }
    root.mkdir(parents=True, exist_ok=True)
    paths["plots"].mkdir(parents=True, exist_ok=True)
    return paths


def write_readme(
    root: Path,
    args: argparse.Namespace,
    cell: tuple[float, float, float, float, float, float],
    cell_source: str,
    unmerged_stats: dict[str, int],
    join_stats: dict[str, int],
    join_note: str,
    join_debug: dict[str, object],
    hkl_count: int,
    obs_count: int,
    summary_count: int,
    plot_count: int,
) -> None:
    lines: list[str] = []
    lines.append("Shell-strength risk diagnostics")
    lines.append("")
    lines.append(f"unmerged: {args.unmerged}")
    lines.append(f"scores: {args.scores}")
    lines.append(f"risk_column: {args.risk_column}")
    lines.append(f"cell_source: {cell_source}")
    lines.append(
        "cell: "
        + " ".join(
            [
                f"a={cell[0]:.6f}",
                f"b={cell[1]:.6f}",
                f"c={cell[2]:.6f}",
                f"alpha={cell[3]:.6f}",
                f"beta={cell[4]:.6f}",
                f"gamma={cell[5]:.6f}",
            ]
        )
    )
    lines.append(f"shell_edges_A_desc: {args.shell_edges}")
    lines.append(f"min_obs_per_hkl: {args.min_obs_per_hkl}")
    lines.append(
        "strength_quantiles: "
        f"weak=[{args.strength_weak_low_q}, {args.strength_weak_high_q}], "
        f"strong=[{args.strength_strong_low_q}, {args.strength_strong_high_q}]"
    )
    lines.append(f"risk_quantiles: low={args.risk_low_quantile}, high={args.risk_high_quantile}")
    lines.append("")
    lines.append("unmerged_filter_stats:")
    for k, v in unmerged_stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("join_stats:")
    for k, v in join_stats.items():
        lines.append(f"  {k}: {v}")
    if join_note:
        lines.append(f"  note: {join_note}")
    lines.append("")
    lines.append("join_debug:")
    lines.append(
        "  duplicated_score_column_names_before_cleanup: "
        f"{join_debug.get('duplicated_score_column_names_before_cleanup', [])}"
    )
    lines.append(
        "  duplicated_unmerged_column_names_before_cleanup: "
        f"{join_debug.get('duplicated_unmerged_column_names_before_cleanup', [])}"
    )
    lines.append(f"  final_merge_key: {join_debug.get('final_merge_key', [])}")
    lines.append(f"  payload_column_count: {join_debug.get('payload_column_count', 0)}")
    lines.append(f"  hkl_removed_from_payload_cols: {join_debug.get('hkl_removed_from_payload_cols', False)}")
    lines.append(
        "  payload_cols_removed_because_in_key: "
        f"{join_debug.get('payload_cols_removed_because_in_key', [])}"
    )
    lines.append("")
    lines.append(f"shell_hkls_written: {hkl_count}")
    lines.append(f"observations_written: {obs_count}")
    lines.append(f"summary_rows_written: {summary_count}")
    lines.append(f"plot_files_written: {plot_count}")

    out = root / "README_shell_strength_diagnostics.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    shell_edges = validate_shell_edges(args.shell_edges)
    cell, cell_source = resolve_unit_cell(args.cell, args.run_metadata, args.scores)
    g_star = reciprocal_metric_tensor(cell)

    unmerged, unmerged_stats = parse_unmerged_observations(
        args.unmerged,
        smoke_max_reflections=args.smoke_max_reflections,
    )
    if unmerged.empty:
        raise SystemExit("No unmerged rows remained after exclusions.")

    key_cols = ["source_norm", "event_norm", "h", "k", "l"]
    dup_diag_path = out["root"] / "duplicate_diagnostics.csv"

    unmerged_dup = duplicate_rows(unmerged, key_cols)
    if not unmerged_dup.empty:
        write_duplicate_diagnostics(dup_diag_path, "unmerged", unmerged_dup, key_cols)
        raise SystemExit(
            "Duplicate source+event+signed hkl rows found in unmerged observations. "
            f"Diagnostics written: {dup_diag_path}"
        )

    scores, source_col = load_scores(
        args.scores,
        risk_column=args.risk_column,
        chunksize=args.score_chunksize,
        smoke_max_score_rows=args.smoke_max_score_rows,
    )
    if scores.empty:
        raise SystemExit("No score rows were loaded.")

    score_dup = duplicate_rows(scores, key_cols)
    if not score_dup.empty:
        write_duplicate_diagnostics(dup_diag_path, "scores", score_dup, key_cols)
        raise SystemExit(
            "Duplicate source+event+signed hkl rows found in score table. "
            f"Diagnostics written: {dup_diag_path}"
        )

    joined, join_stats, join_note, join_debug = join_unmerged_with_scores(
        unmerged,
        scores,
        risk_column=args.risk_column,
    )

    joined["score_matched"] = joined["match_mode"].isin(["primary", "basename"])
    matched = joined.loc[joined["score_matched"]].copy()

    if args.smoke_only:
        print("SMOKE_ONLY_OK")
        print(f"source_column={source_col}")
        print(f"unmerged_rows={len(unmerged):,}")
        print(f"score_rows={len(scores):,}")
        print(f"joined_rows={len(joined):,}")
        print(f"matched_rows={len(matched):,}")
        print(f"unmerged_stats={json.dumps(unmerged_stats, sort_keys=True)}")
        print(f"join_stats={json.dumps(join_stats, sort_keys=True)}")
        print(f"join_debug={json.dumps(join_debug, sort_keys=True)}")
        return

    if matched.empty:
        raise SystemExit("No matched rows between unmerged observations and scores.")

    hkl_table = compute_hkl_classification(
        matched,
        g_star=g_star,
        shell_edges=shell_edges,
        min_obs_per_hkl=int(args.min_obs_per_hkl),
        weak_low_q=float(args.strength_weak_low_q),
        weak_high_q=float(args.strength_weak_high_q),
        strong_low_q=float(args.strength_strong_low_q),
        strong_high_q=float(args.strength_strong_high_q),
    )
    if hkl_table.empty:
        raise SystemExit("No HKLs remained after shell assignment and min-obs filter.")

    obs = compute_observation_residuals(
        matched,
        hkl_table,
        risk_column=args.risk_column,
        risk_low_quantile=float(args.risk_low_quantile),
        risk_high_quantile=float(args.risk_high_quantile),
        eps=float(args.eps),
    )
    if obs.empty:
        raise SystemExit("No observations remained after HKL-shell merge.")

    summary = summarize_shell_strength(obs, shell_edges=shell_edges, risk_column=args.risk_column)

    hkl_out = out["root"] / "shell_hkl_classification.csv"
    obs_out = out["root"] / "shell_observation_residuals.csv"
    summary_out = out["root"] / "shell_strength_summary.csv"

    hkl_table.to_csv(hkl_out, index=False)
    obs.to_csv(obs_out, index=False)
    summary.to_csv(summary_out, index=False)

    plot_files: list[str] = []
    plot_files.extend(
        plot_histograms_by_shell_strength(
            obs,
            shell_edges=shell_edges,
            plots_dir=out["plots"],
            hist_bins=int(args.hist_bins),
        )
    )
    plot_files.extend(
        plot_representative_i_pr(
            obs,
            hkl_table,
            shell_edges=shell_edges,
            plots_dir=out["plots"],
            representative_hkls_per_class=int(args.representative_hkls_per_class),
            hist_bins=int(args.hist_bins),
        )
    )
    summary_plot = plot_summary_shift(summary, shell_edges=shell_edges, plots_dir=out["plots"])
    if summary_plot is not None:
        plot_files.append(summary_plot)

    write_readme(
        root=out["root"],
        args=args,
        cell=cell,
        cell_source=cell_source,
        unmerged_stats=unmerged_stats,
        join_stats=join_stats,
        join_note=join_note,
        join_debug=join_debug,
        hkl_count=int(len(hkl_table)),
        obs_count=int(len(obs)),
        summary_count=int(len(summary)),
        plot_count=int(len(plot_files)),
    )

    print("DONE")
    print(f"score_source_column={source_col}")
    print(f"joined_rows={len(joined):,}")
    print(f"matched_rows={len(matched):,}")
    print(f"shell_hkls={len(hkl_table):,}")
    print(f"observation_rows={len(obs):,}")
    print(f"summary_rows={len(summary):,}")
    print(f"plots_written={len(plot_files):,}")


if __name__ == "__main__":
    main()
