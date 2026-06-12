#!/usr/bin/env python3
"""Analyze OriDyn component terms against I_pr residual shifts.

This script reuses the validated join logic used in curated HKL diagnostics:
source_filename/source_norm + event + signed h,k,l.

It computes I_pr = I_unmerged * partiality and evaluates per-component
relationships with residual and relative_residual across selected HKLs.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

UNMERGED_FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+(\S+))?\s*$")
UNMERGED_FLAGGED_RE = re.compile(r"^\s*Flagged:\s*(\S+)\s*$", re.IGNORECASE)

DEFAULT_COMPONENT_COLUMNS = [
    "self_risk_norm",
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

DEFAULT_SHELL_EDGES = [1.8, 1.4, 1.1, 0.9, 0.7, 0.55, 0.4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join partiality-weighted unmerged observations with OriDyn scores and "
            "analyze correlations between separate OriDyn component terms and I_pr shifts."
        )
    )
    parser.add_argument("--unmerged", required=True, type=Path, help="partialator --unmerged-output file")
    parser.add_argument("--scores", required=True, type=Path, help="OriDyn reflection_scores.csv")
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=None,
        help="Optional run_metadata.json. If omitted, tries <scores_dir>/run_metadata.json",
    )
    parser.add_argument(
        "--cell",
        nargs=6,
        type=float,
        metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
        default=None,
        help="Fallback unit cell if metadata does not contain one",
    )
    parser.add_argument("--output-root", required=True, type=Path, help="Output folder for component diagnostics")

    parser.add_argument("--risk-column", default="S_dyn_geom", help="Main risk column (default: S_dyn_geom)")
    parser.add_argument("--color-risk-column", default="sigma_dyn_rel", help="Secondary risk-like column")
    parser.add_argument(
        "--component-columns",
        nargs="+",
        default=list(DEFAULT_COMPONENT_COLUMNS),
        help="Component columns from reflection_scores.csv",
    )
    parser.add_argument(
        "--selected-hkl-summary",
        type=Path,
        default=None,
        help="Optional curated_hkl_summary.csv (or HKL text) to restrict analyzed HKLs",
    )

    parser.add_argument("--min-obs-per-hkl", type=int, default=100)
    parser.add_argument(
        "--risk-quantiles",
        nargs=2,
        type=float,
        default=[0.25, 0.75],
        metavar=("LOW_Q", "HIGH_Q"),
        help="Quantiles used for low/high component splits (default: 0.25 0.75)",
    )

    parser.add_argument(
        "--shell-edges",
        nargs="+",
        type=float,
        default=DEFAULT_SHELL_EDGES,
        help="Descending d-spacing shell edges in Angstrom",
    )

    parser.add_argument("--score-chunksize", type=int, default=1_000_000)
    parser.add_argument("--scatter-bins", type=int, default=12)
    parser.add_argument("--max-hkl-scatter-plots", type=int, default=40)
    parser.add_argument("--max-hkls-heatmap", type=int, default=80)
    parser.add_argument("--eps", type=float, default=1e-12)

    parser.add_argument(
        "--smoke-max-reflections",
        type=int,
        default=None,
        help="Read at most this many eligible unmerged observations",
    )
    parser.add_argument(
        "--smoke-max-score-rows",
        type=int,
        default=None,
        help="Read at most this many score rows",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run parse+join+core metrics only; skip heavy plotting",
    )

    args = parser.parse_args()

    if args.min_obs_per_hkl <= 0:
        raise SystemExit("--min-obs-per-hkl must be > 0")
    if args.eps <= 0.0:
        raise SystemExit("--eps must be > 0")
    if args.scatter_bins < 3:
        raise SystemExit("--scatter-bins must be >= 3")
    if args.max_hkl_scatter_plots < 0:
        raise SystemExit("--max-hkl-scatter-plots must be >= 0")
    if args.max_hkls_heatmap <= 0:
        raise SystemExit("--max-hkls-heatmap must be > 0")

    if len(args.shell_edges) < 2:
        raise SystemExit("--shell-edges must contain at least two values")
    if not all(args.shell_edges[i] > args.shell_edges[i + 1] for i in range(len(args.shell_edges) - 1)):
        raise SystemExit("--shell-edges must be strictly descending")

    if len(args.risk_quantiles) != 2:
        raise SystemExit("--risk-quantiles must contain exactly two values")
    q_lo, q_hi = float(args.risk_quantiles[0]), float(args.risk_quantiles[1])
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise SystemExit("--risk-quantiles must satisfy 0 <= low < high <= 1")

    if args.smoke_only:
        if args.smoke_max_reflections is None:
            args.smoke_max_reflections = 50_000
        if args.smoke_max_score_rows is None:
            args.smoke_max_score_rows = 500_000

    return args


def normalize_source(value: object) -> str:
    return str(value).strip()


def normalize_event(value: object) -> str:
    return str(value).strip()


def source_basename(value: object) -> str:
    return Path(str(value).strip()).name


def hkl_key(h: int, k: int, l: int) -> str:
    return f"{int(h)},{int(k)},{int(l)}"


def hkl_slug(h: int, k: int, l: int) -> str:
    return f"h{h:+d}_k{k:+d}_l{l:+d}".replace("+", "p").replace("-", "m")


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


def load_selected_hkls(path: Path | None) -> set[tuple[int, int, int]]:
    if path is None:
        return set()

    if not path.exists():
        raise SystemExit(f"--selected-hkl-summary not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if not all(c in cols for c in ("h", "k", "l")):
            raise SystemExit(f"Selected HKL summary CSV missing h/k/l columns: {path}")
        h_col, k_col, l_col = cols["h"], cols["k"], cols["l"]
        parsed = (
            df[[h_col, k_col, l_col]]
            .dropna()
            .astype({h_col: "int64", k_col: "int64", l_col: "int64"})
            .itertuples(index=False, name=None)
        )
        hkls = {(int(h), int(k), int(l)) for h, k, l in parsed}
    else:
        hkls: set[tuple[int, int, int]] = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = parse_hkl_line(line)
            if parsed is not None:
                hkls.add(parsed)

    if not hkls:
        raise SystemExit(f"No HKLs parsed from {path}")
    return hkls


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


def parse_unmerged_observations(
    unmerged_path: Path,
    selected_hkls: set[tuple[int, int, int]],
    smoke_max_reflections: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []

    stats = {
        "rows_parsed": 0,
        "excluded_flagged_crystal": 0,
        "excluded_partiality_too_small": 0,
        "excluded_nan_esd": 0,
        "excluded_nonpositive_partiality": 0,
        "excluded_not_selected_hkl": 0,
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
            if "partiality_too_small" in reflection_flag.lower():
                stats["excluded_partiality_too_small"] += 1
                continue
            if "nan_esd" in reflection_flag.lower():
                stats["excluded_nan_esd"] += 1
                continue
            if not np.isfinite(partiality) or partiality <= 0.0:
                stats["excluded_nonpositive_partiality"] += 1
                continue
            if selected_hkls and (h, k, l) not in selected_hkls:
                stats["excluded_not_selected_hkl"] += 1
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
    table[["h", "k", "l"]] = table[["h", "k", "l"]].astype("int64")
    return table, stats


def load_scores(
    scores_path: Path,
    risk_column: str,
    color_risk_column: str,
    component_columns: list[str],
    chunksize: int,
    selected_hkls: set[tuple[int, int, int]],
    max_rows: int | None,
) -> tuple[pd.DataFrame, str]:
    header = list(pd.read_csv(scores_path, nrows=0).columns)
    source_column = choose_score_source_column(scores_path, header)

    required = [source_column, "event", "h", "k", "l", risk_column, color_risk_column, *component_columns]
    missing = [c for c in required if c not in header]
    if missing:
        raise SystemExit(f"Scores file missing required column(s): {missing}")

    keep_columns = list(dict.fromkeys([source_column, "event", "h", "k", "l", risk_column, color_risk_column, *component_columns]))

    selected_hkl_keys: set[str] | None = None
    if selected_hkls:
        selected_hkl_keys = {hkl_key(h, k, l) for h, k, l in selected_hkls}

    chunks: list[pd.DataFrame] = []
    loaded_rows = 0

    reader = pd.read_csv(scores_path, usecols=keep_columns, chunksize=chunksize)
    for chunk in reader:
        chunk["h"] = pd.to_numeric(chunk["h"], errors="coerce")
        chunk["k"] = pd.to_numeric(chunk["k"], errors="coerce")
        chunk["l"] = pd.to_numeric(chunk["l"], errors="coerce")
        chunk = chunk.dropna(subset=["h", "k", "l"])
        chunk[["h", "k", "l"]] = chunk[["h", "k", "l"]].astype("int64")

        if selected_hkl_keys is not None:
            keys = chunk["h"].astype(str) + "," + chunk["k"].astype(str) + "," + chunk["l"].astype(str)
            chunk = chunk.loc[keys.isin(selected_hkl_keys)].copy()

        if max_rows is not None:
            remaining = int(max_rows) - loaded_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        if chunk.empty:
            continue

        chunks.append(chunk)
        loaded_rows += int(len(chunk))

        if max_rows is not None and loaded_rows >= int(max_rows):
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

    numeric_cols = [risk_column, color_risk_column, *component_columns]
    for col in numeric_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce")

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

    merged = unmerged.merge(scores_subset, on=key, how="left", indicator=True)
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


def coerce_cell_from_mapping(mapping: object) -> tuple[float, float, float, float, float, float] | None:
    if not isinstance(mapping, dict):
        return None

    required = ["a", "b", "c", "alpha", "beta", "gamma"]
    if not all(key in mapping for key in required):
        return None

    try:
        return (
            float(mapping["a"]),
            float(mapping["b"]),
            float(mapping["c"]),
            float(mapping["alpha"]),
            float(mapping["beta"]),
            float(mapping["gamma"]),
        )
    except (TypeError, ValueError):
        return None


def extract_cell_recursive(node: object, depth: int = 0, max_depth: int = 8) -> tuple[float, float, float, float, float, float] | None:
    if depth > max_depth:
        return None

    direct = coerce_cell_from_mapping(node)
    if direct is not None:
        return direct

    if isinstance(node, dict):
        for key in ("unit_cell", "cell", "unitCell", "unit_cell_parameters"):
            if key in node:
                parsed = extract_cell_recursive(node[key], depth + 1, max_depth)
                if parsed is not None:
                    return parsed
        for value in node.values():
            parsed = extract_cell_recursive(value, depth + 1, max_depth)
            if parsed is not None:
                return parsed

    if isinstance(node, list):
        for value in node:
            parsed = extract_cell_recursive(value, depth + 1, max_depth)
            if parsed is not None:
                return parsed

    return None


def load_cell_from_metadata(path: Path | None) -> tuple[float, float, float, float, float, float] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return extract_cell_recursive(data)


def resolve_cell(
    cli_cell: list[float] | None,
    metadata_path: Path | None,
    scores_path: Path,
) -> tuple[tuple[float, float, float, float, float, float], str]:
    if cli_cell is not None:
        return tuple(float(v) for v in cli_cell), "cli"

    cell = load_cell_from_metadata(metadata_path)
    if cell is not None:
        return cell, f"metadata:{metadata_path}"

    auto_meta = scores_path.parent / "run_metadata.json"
    if metadata_path is None and auto_meta.exists():
        auto_cell = load_cell_from_metadata(auto_meta)
        if auto_cell is not None:
            return auto_cell, f"metadata:{auto_meta}"

    raise SystemExit(
        "Could not extract unit-cell parameters from metadata. "
        "Provide --cell a b c alpha beta gamma."
    )


def reciprocal_metric_tensor(cell: tuple[float, float, float, float, float, float]) -> np.ndarray:
    a, b, c, alpha_deg, beta_deg, gamma_deg = cell
    if min(a, b, c) <= 0.0:
        raise SystemExit("Unit-cell lengths must be positive")

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)

    g = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=float,
    )

    det = float(np.linalg.det(g))
    if not np.isfinite(det) or det <= 0.0:
        raise SystemExit("Invalid unit-cell metric tensor; determinant must be positive")

    return np.linalg.inv(g)


def d_spacing_from_hkl(h: int, k: int, l: int, g_star: np.ndarray) -> float:
    if h == 0 and k == 0 and l == 0:
        return np.nan
    vec = np.array([float(h), float(k), float(l)], dtype=float)
    inv_d_sq = float(vec @ g_star @ vec)
    if not np.isfinite(inv_d_sq) or inv_d_sq <= 0.0:
        return np.nan
    return float(1.0 / math.sqrt(inv_d_sq))


def shell_records(shell_edges: list[float]) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    for idx in range(len(shell_edges) - 1):
        high = float(shell_edges[idx])
        low = float(shell_edges[idx + 1])
        label = f"{high:.3f}-{low:.3f}A"
        records.append({"idx": idx, "high": high, "low": low, "label": label})
    return records


def assign_shell(d_spacing: float, records: list[dict[str, float | str]]) -> tuple[str | None, float, float]:
    if not np.isfinite(d_spacing):
        return None, np.nan, np.nan

    for idx, rec in enumerate(records):
        high = float(rec["high"])
        low = float(rec["low"])
        if idx < len(records) - 1:
            in_shell = d_spacing <= high and d_spacing > low
        else:
            in_shell = d_spacing <= high and d_spacing >= low
        if in_shell:
            return str(rec["label"]), high, low

    return None, np.nan, np.nan


def maybe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan
    v2 = v[mask].to_numpy(dtype=float)
    w2 = w[mask].to_numpy(dtype=float)
    return float(np.average(v2, weights=w2))


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return np.nan

    df = pd.DataFrame({"v": v[mask], "w": w[mask]}).sort_values("v")
    csum = df["w"].cumsum()
    cutoff = 0.5 * float(df["w"].sum())
    idx = int((csum >= cutoff).idxmax())
    return float(df.loc[idx, "v"])


def compute_joined_observations(
    matched: pd.DataFrame,
    g_star: np.ndarray,
    shell_recs: list[dict[str, float | str]],
    min_obs_per_hkl: int,
    eps: float,
    risk_column: str,
    q_low: float,
    q_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hkl_stats = (
        matched.groupby(["h", "k", "l"], as_index=False)
        .agg(n_obs=("I_pr", "size"), median_I_pr=("I_pr", "median"))
    )

    hkl_stats["d_spacing"] = hkl_stats.apply(
        lambda row: d_spacing_from_hkl(int(row["h"]), int(row["k"]), int(row["l"]), g_star),
        axis=1,
    )

    assigned = hkl_stats["d_spacing"].map(lambda d: assign_shell(float(d), shell_recs))
    hkl_stats["shell_label"] = assigned.map(lambda t: t[0])
    hkl_stats["shell_d_high"] = assigned.map(lambda t: t[1])
    hkl_stats["shell_d_low"] = assigned.map(lambda t: t[2])

    hkl_stats = hkl_stats.loc[hkl_stats["shell_label"].notna()].copy()
    hkl_stats["nobs_below_threshold"] = hkl_stats["n_obs"] < int(min_obs_per_hkl)

    keep_hkls = hkl_stats.loc[hkl_stats["n_obs"] >= int(min_obs_per_hkl), ["h", "k", "l"]].copy()
    if keep_hkls.empty:
        raise SystemExit("No HKLs survive --min-obs-per-hkl after shell assignment")

    filtered = matched.merge(keep_hkls, on=["h", "k", "l"], how="inner")
    joined = filtered.merge(hkl_stats, on=["h", "k", "l"], how="left")

    joined["residual"] = joined["I_pr"] - joined["median_I_pr"]
    joined["abs_median_I_pr"] = joined["median_I_pr"].abs()

    shell_abs_med = (
        joined[["h", "k", "l", "shell_label", "abs_median_I_pr"]]
        .drop_duplicates(subset=["h", "k", "l"])
        .groupby("shell_label", as_index=False)["abs_median_I_pr"]
        .median()
        .rename(columns={"abs_median_I_pr": "shell_median_abs_median_I_pr"})
    )

    joined = joined.merge(shell_abs_med, on="shell_label", how="left")

    stable_denom = np.maximum(
        np.maximum(
            joined["abs_median_I_pr"].to_numpy(dtype=float),
            0.1 * joined["shell_median_abs_median_I_pr"].to_numpy(dtype=float),
        ),
        float(eps),
    )
    joined["stable_relative_denominator"] = stable_denom
    joined["relative_residual"] = joined["residual"].to_numpy(dtype=float) / stable_denom

    joined[risk_column] = pd.to_numeric(joined[risk_column], errors="coerce")
    low_map = joined.groupby(["h", "k", "l"])[risk_column].transform(lambda s: float(s.quantile(q_low)))
    high_map = joined.groupby(["h", "k", "l"])[risk_column].transform(lambda s: float(s.quantile(q_high)))
    joined["risk_q_low"] = low_map
    joined["risk_q_high"] = high_map
    joined["risk_group"] = "mid"
    joined.loc[joined[risk_column] <= joined["risk_q_low"], "risk_group"] = "low"
    joined.loc[joined[risk_column] >= joined["risk_q_high"], "risk_group"] = "high"

    return joined, hkl_stats


def compute_component_per_hkl_metrics(
    joined: pd.DataFrame,
    component_columns: list[str],
    q_low: float,
    q_high: float,
    eps: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (h, k, l), group in joined.groupby(["h", "k", "l"], sort=True):
        n_obs = int(len(group))
        median_i = float(group["median_I_pr"].iloc[0])
        shell_label = str(group["shell_label"].iloc[0])
        shell_high = float(group["shell_d_high"].iloc[0])
        shell_low = float(group["shell_d_low"].iloc[0])
        d_spacing = float(group["d_spacing"].iloc[0])
        shell_abs_med = float(group["shell_median_abs_median_I_pr"].iloc[0])

        relative_shift_denom = max(abs(median_i), 0.1 * shell_abs_med, float(eps))

        for comp in component_columns:
            comp_vals = pd.to_numeric(group[comp], errors="coerce")
            valid = comp_vals.notna()
            if not valid.any():
                continue

            c = comp_vals[valid]
            residual = pd.to_numeric(group.loc[valid, "residual"], errors="coerce")
            rel_residual = pd.to_numeric(group.loc[valid, "relative_residual"], errors="coerce")
            i_pr_vals = pd.to_numeric(group.loc[valid, "I_pr"], errors="coerce")

            p05 = float(c.quantile(0.05))
            p50 = float(c.quantile(0.50))
            p95 = float(c.quantile(0.95))
            spread = p95 - p05

            split_low = float(c.quantile(q_low))
            split_high = float(c.quantile(q_high))
            low_mask = c <= split_low
            high_mask = c >= split_high

            low_med_i = float(i_pr_vals[low_mask].median()) if low_mask.any() else np.nan
            high_med_i = float(i_pr_vals[high_mask].median()) if high_mask.any() else np.nan

            shift = high_med_i - low_med_i if np.isfinite(high_med_i) and np.isfinite(low_med_i) else np.nan
            rel_shift = shift / relative_shift_denom if np.isfinite(shift) else np.nan

            rows.append(
                {
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "hkl": hkl_key(int(h), int(k), int(l)),
                    "shell_label": shell_label,
                    "shell_d_high": shell_high,
                    "shell_d_low": shell_low,
                    "d_spacing": d_spacing,
                    "n_obs": n_obs,
                    "median_I_pr": median_i,
                    "shell_median_abs_median_I_pr": shell_abs_med,
                    "stable_relative_denominator": relative_shift_denom,
                    "component": comp,
                    "component_p05": p05,
                    "component_p50": p50,
                    "component_p95": p95,
                    "component_spread": spread,
                    "component_split_low_quantile": q_low,
                    "component_split_high_quantile": q_high,
                    "component_split_low_value": split_low,
                    "component_split_high_value": split_high,
                    "spearman_component_vs_residual": maybe_spearman(c, residual),
                    "spearman_component_vs_relative_residual": maybe_spearman(c, rel_residual),
                    "low_component_median_I_pr": low_med_i,
                    "high_component_median_I_pr": high_med_i,
                    "high_minus_low_component_shift": shift,
                    "relative_high_minus_low_component_shift": rel_shift,
                    "absolute_relative_high_minus_low_component_shift": abs(rel_shift) if np.isfinite(rel_shift) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def compute_component_summary(per_hkl: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for comp, group in per_hkl.groupby("component", sort=True):
        shift = pd.to_numeric(group["high_minus_low_component_shift"], errors="coerce")
        rel_shift = pd.to_numeric(group["relative_high_minus_low_component_shift"], errors="coerce")
        abs_rel_shift = pd.to_numeric(group["absolute_relative_high_minus_low_component_shift"], errors="coerce")

        rho_res = pd.to_numeric(group["spearman_component_vs_residual"], errors="coerce")
        rho_rel = pd.to_numeric(group["spearman_component_vs_relative_residual"], errors="coerce")

        weights = pd.to_numeric(group["n_obs"], errors="coerce").fillna(0.0)

        rows.append(
            {
                "component": comp,
                "n_hkls": int(group[["h", "k", "l"]].drop_duplicates().shape[0]),
                "n_rows": int(len(group)),
                "n_obs_total": int(weights.sum()),
                "median_spearman_component_vs_residual": float(rho_res.median()) if rho_res.notna().any() else np.nan,
                "median_abs_spearman_component_vs_residual": float(rho_res.abs().median()) if rho_res.notna().any() else np.nan,
                "median_spearman_component_vs_relative_residual": float(rho_rel.median()) if rho_rel.notna().any() else np.nan,
                "median_abs_spearman_component_vs_relative_residual": float(rho_rel.abs().median()) if rho_rel.notna().any() else np.nan,
                "median_relative_shift": float(rel_shift.median()) if rel_shift.notna().any() else np.nan,
                "median_abs_relative_shift": float(abs_rel_shift.median()) if abs_rel_shift.notna().any() else np.nan,
                "count_positive_shifts": int((shift > 0).sum()),
                "count_negative_shifts": int((shift < 0).sum()),
                "weighted_mean_relative_shift": weighted_mean(rel_shift, weights),
                "weighted_mean_abs_relative_shift": weighted_mean(abs_rel_shift, weights),
                "weighted_mean_abs_spearman_component_vs_relative_residual": weighted_mean(rho_rel.abs(), weights),
                "weighted_median_relative_shift": weighted_median(rel_shift, weights),
                "weighted_median_abs_relative_shift": weighted_median(abs_rel_shift, weights),
            }
        )

    return pd.DataFrame(rows)


def plot_component_summary_bar(component_summary: pd.DataFrame, out_path: Path) -> bool:
    if component_summary.empty:
        return False

    df = component_summary.sort_values("median_abs_relative_shift", ascending=False).copy()
    x = np.arange(len(df), dtype=float)

    median_abs = pd.to_numeric(df["median_abs_relative_shift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weighted_abs = pd.to_numeric(df["weighted_mean_abs_relative_shift"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(df)), 5.2), dpi=170)
    ax.bar(x - 0.18, median_abs, width=0.35, color="#3a7db8", label="median |relative shift|")
    ax.bar(x + 0.18, weighted_abs, width=0.35, color="#e07a2b", label="n_obs-weighted mean |relative shift|")

    ax.set_xticks(x)
    ax.set_xticklabels(df["component"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("shift magnitude")
    ax.set_title("Component summary: relative shift magnitude")
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_component_hkl_heatmap(per_hkl: pd.DataFrame, max_hkls: int, out_path: Path) -> bool:
    if per_hkl.empty:
        return False

    rank_hkls = (
        per_hkl.groupby(["h", "k", "l", "hkl"], as_index=False)["n_obs"]
        .max()
        .sort_values("n_obs", ascending=False)
        .head(int(max_hkls))
    )
    keep = set(rank_hkls["hkl"].tolist())

    pivot_source = per_hkl.loc[per_hkl["hkl"].isin(keep), ["hkl", "component", "relative_high_minus_low_component_shift"]].copy()
    if pivot_source.empty:
        return False

    pivot = pivot_source.pivot_table(
        index="hkl",
        columns="component",
        values="relative_high_minus_low_component_shift",
        aggfunc="median",
    )

    ordered_hkls = rank_hkls["hkl"].tolist()
    pivot = pivot.reindex(index=ordered_hkls)

    arr = pivot.to_numpy(dtype=float)
    if arr.size == 0:
        return False

    finite = np.isfinite(arr)
    vmax = float(np.nanquantile(np.abs(arr[finite]), 0.95)) if finite.any() else 1.0
    vmax = max(vmax, 1e-6)

    fig_h = max(4.5, 0.28 * len(pivot.index) + 1.6)
    fig_w = max(7.0, 1.0 * len(pivot.columns) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_title("HKL-component relative shift heatmap")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("relative high-low component shift")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def binned_median_trend(x: pd.Series, y: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if mask.sum() < max(5, bins):
        return np.array([]), np.array([])

    xs = xv[mask].to_numpy(dtype=float)
    ys = yv[mask].to_numpy(dtype=float)

    edges = np.quantile(xs, np.linspace(0.0, 1.0, int(bins) + 1))
    edges = np.unique(edges)
    if len(edges) < 4:
        return np.array([]), np.array([])

    centers: list[float] = []
    medians: list[float] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i < (len(edges) - 2):
            sel = (xs >= lo) & (xs < hi)
        else:
            sel = (xs >= lo) & (xs <= hi)
        if int(sel.sum()) < 3:
            continue
        centers.append(float(np.median(xs[sel])))
        medians.append(float(np.median(ys[sel])))

    if not centers:
        return np.array([]), np.array([])
    return np.asarray(centers, dtype=float), np.asarray(medians, dtype=float)


def plot_per_hkl_component_scatter_panels(
    joined: pd.DataFrame,
    per_hkl: pd.DataFrame,
    component_columns: list[str],
    max_hkl_plots: int,
    bins: int,
    out_dir: Path,
) -> int:
    if per_hkl.empty or int(max_hkl_plots) == 0:
        return 0

    top_hkls = (
        per_hkl.groupby(["h", "k", "l", "hkl"], as_index=False)["n_obs"]
        .max()
        .sort_values("n_obs", ascending=False)
        .head(int(max_hkl_plots))
    )

    count = 0
    for _, row in top_hkls.iterrows():
        h = int(row["h"])
        k = int(row["k"])
        l = int(row["l"])
        hkl_name = str(row["hkl"])

        sub = joined.loc[(joined["h"] == h) & (joined["k"] == k) & (joined["l"] == l)].copy()
        if sub.empty:
            continue

        n_comp = len(component_columns)
        ncols = 3
        nrows = int(math.ceil(n_comp / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), dpi=170, squeeze=False)

        for idx, comp in enumerate(component_columns):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r, c]

            x = pd.to_numeric(sub[comp], errors="coerce")
            y = pd.to_numeric(sub["residual"], errors="coerce")
            valid = x.notna() & y.notna()

            if valid.any():
                ax.scatter(x[valid], y[valid], s=10, alpha=0.45, color="#3b6ea8", linewidths=0)
                bx, by = binned_median_trend(x[valid], y[valid], bins=int(bins))
                if bx.size > 0:
                    ax.plot(bx, by, color="#d73027", linewidth=1.8, marker="o", markersize=3)

            rho = maybe_spearman(x, y)
            ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
            ax.set_title(f"{comp}\nrho={rho:.3f}", fontsize=9)
            ax.set_xlabel(comp)
            ax.set_ylabel("residual")

        total_axes = nrows * ncols
        for idx in range(n_comp, total_axes):
            r = idx // ncols
            c = idx % ncols
            axes[r, c].axis("off")

        fig.suptitle(f"HKL {hkl_name} component-vs-residual panels", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

        out_path = out_dir / f"{hkl_slug(h, k, l)}_component_scatter_panels.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        count += 1

    return count


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
        "per_hkl": root / "plots" / "per_hkl",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_readme(
    root: Path,
    args: argparse.Namespace,
    source_col: str,
    cell: tuple[float, float, float, float, float, float],
    cell_source: str,
    selected_hkls_count: int,
    unmerged_stats: dict[str, int],
    join_stats: dict[str, int],
    join_note: str,
    join_debug: dict[str, object],
    n_joined_rows: int,
    n_hkl_metrics: int,
    n_component_rows: int,
    n_summary_rows: int,
    n_scatter_plots: int,
    wrote_summary_plot: bool,
    wrote_heatmap_plot: bool,
) -> None:
    lines: list[str] = []
    lines.append("Sdyn component diagnostics")
    lines.append("")
    lines.append(f"unmerged_file: {args.unmerged}")
    lines.append(f"scores_file: {args.scores}")
    lines.append(f"score_source_column: {source_col}")
    lines.append(f"selected_hkl_summary: {args.selected_hkl_summary}")
    lines.append(f"selected_hkls_count: {selected_hkls_count if selected_hkls_count > 0 else 'all'}")
    lines.append(f"risk_column: {args.risk_column}")
    lines.append(f"color_risk_column: {args.color_risk_column}")
    lines.append(f"component_columns: {args.component_columns}")
    lines.append(f"risk_quantiles: {args.risk_quantiles}")
    lines.append(f"min_obs_per_hkl: {args.min_obs_per_hkl}")
    lines.append(f"shell_edges_A_desc: {args.shell_edges}")
    lines.append(f"cell_source: {cell_source}")
    lines.append("cell(a b c alpha beta gamma): " + " ".join(f"{x:.6g}" for x in cell))
    lines.append("")
    lines.append("Definitions:")
    lines.append("  I_pr = I_unmerged * partiality")
    lines.append("  residual = I_pr - median_I_pr(within HKL)")
    lines.append("  relative_residual = residual / max(abs(median_I_pr), 0.1*median(abs(median_I_pr)) within shell)")
    lines.append("  component splits: bottom/top quantiles from --risk-quantiles")
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
    lines.append("  duplicated_score_column_names_before_cleanup: " + str(join_debug.get("duplicated_score_column_names_before_cleanup", [])))
    lines.append("  duplicated_unmerged_column_names_before_cleanup: " + str(join_debug.get("duplicated_unmerged_column_names_before_cleanup", [])))
    lines.append("  final_merge_key: " + str(join_debug.get("final_merge_key", [])))
    lines.append("  payload_column_count: " + str(join_debug.get("payload_column_count", 0)))
    lines.append("  hkl_removed_from_payload_cols: " + str(join_debug.get("hkl_removed_from_payload_cols", False)))

    lines.append("")
    lines.append(f"joined_component_observations_rows: {n_joined_rows}")
    lines.append(f"component_per_hkl_metrics_rows: {n_component_rows}")
    lines.append(f"component_summary_rows: {n_summary_rows}")
    lines.append(f"hkl_metrics_rows: {n_hkl_metrics}")
    lines.append(f"component_summary_bar_written: {wrote_summary_plot}")
    lines.append(f"component_hkl_heatmap_written: {wrote_heatmap_plot}")
    lines.append(f"per_hkl_scatter_plots_written: {n_scatter_plots}")

    (root / "README_sdyn_component_diagnostics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    selected_hkls = load_selected_hkls(args.selected_hkl_summary)

    metadata_path = args.run_metadata
    if metadata_path is None:
        maybe = args.scores.parent / "run_metadata.json"
        if maybe.exists():
            metadata_path = maybe

    cell, cell_source = resolve_cell(args.cell, metadata_path, args.scores)
    g_star = reciprocal_metric_tensor(cell)
    shell_recs = shell_records([float(x) for x in args.shell_edges])

    unmerged, unmerged_stats = parse_unmerged_observations(
        args.unmerged,
        selected_hkls=selected_hkls,
        smoke_max_reflections=args.smoke_max_reflections,
    )
    if unmerged.empty:
        raise SystemExit("No eligible unmerged observations were parsed")

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
        color_risk_column=args.color_risk_column,
        component_columns=[str(c) for c in args.component_columns],
        chunksize=args.score_chunksize,
        selected_hkls=selected_hkls,
        max_rows=args.smoke_max_score_rows,
    )
    if scores.empty:
        raise SystemExit("No score rows were loaded")

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

    if matched.empty:
        raise SystemExit("No matched rows between unmerged observations and scores")

    q_low, q_high = float(args.risk_quantiles[0]), float(args.risk_quantiles[1])

    joined_component_obs, hkl_metrics = compute_joined_observations(
        matched,
        g_star=g_star,
        shell_recs=shell_recs,
        min_obs_per_hkl=int(args.min_obs_per_hkl),
        eps=float(args.eps),
        risk_column=args.risk_column,
        q_low=q_low,
        q_high=q_high,
    )

    if joined_component_obs.empty:
        raise SystemExit("No joined observations remain after min-obs filtering")

    comp_cols = [str(c) for c in args.component_columns]
    per_hkl = compute_component_per_hkl_metrics(
        joined_component_obs,
        component_columns=comp_cols,
        q_low=q_low,
        q_high=q_high,
        eps=float(args.eps),
    )
    if per_hkl.empty:
        raise SystemExit("No per-component HKL metrics were computed")

    component_summary = compute_component_summary(per_hkl)
    ranked = component_summary.sort_values("median_abs_relative_shift", ascending=False).reset_index(drop=True)

    joined_csv = out["root"] / "joined_component_observations.csv"
    per_hkl_csv = out["root"] / "component_per_hkl_metrics.csv"
    summary_csv = out["root"] / "component_summary.csv"
    ranked_csv = out["root"] / "component_ranked_by_abs_relative_shift.csv"

    joined_component_obs.to_csv(joined_csv, index=False)
    per_hkl.to_csv(per_hkl_csv, index=False)
    component_summary.to_csv(summary_csv, index=False)
    ranked.to_csv(ranked_csv, index=False)

    if args.smoke_only:
        print("SMOKE_ONLY_OK")
        print(f"source_column={source_col}")
        print(f"selected_hkls_count={len(selected_hkls) if selected_hkls else 'all'}")
        print(f"unmerged_rows={len(unmerged):,}")
        print(f"score_rows={len(scores):,}")
        print(f"matched_rows={len(matched):,}")
        print(f"joined_component_rows={len(joined_component_obs):,}")
        print(f"per_hkl_component_rows={len(per_hkl):,}")
        print(f"component_summary_rows={len(component_summary):,}")
        print("join_debug=" + json.dumps(join_debug, sort_keys=True))
        return

    summary_plot = out["plots"] / "component_summary_bar.png"
    heatmap_plot = out["plots"] / "component_hkl_heatmap.png"

    wrote_summary_plot = plot_component_summary_bar(component_summary, summary_plot)
    wrote_heatmap_plot = plot_component_hkl_heatmap(per_hkl, args.max_hkls_heatmap, heatmap_plot)

    scatter_count = plot_per_hkl_component_scatter_panels(
        joined=joined_component_obs,
        per_hkl=per_hkl,
        component_columns=comp_cols,
        max_hkl_plots=int(args.max_hkl_scatter_plots),
        bins=int(args.scatter_bins),
        out_dir=out["per_hkl"],
    )

    write_readme(
        root=out["root"],
        args=args,
        source_col=source_col,
        cell=cell,
        cell_source=cell_source,
        selected_hkls_count=len(selected_hkls),
        unmerged_stats=unmerged_stats,
        join_stats=join_stats,
        join_note=join_note,
        join_debug=join_debug,
        n_joined_rows=int(len(joined_component_obs)),
        n_hkl_metrics=int(len(hkl_metrics)),
        n_component_rows=int(len(per_hkl)),
        n_summary_rows=int(len(component_summary)),
        n_scatter_plots=int(scatter_count),
        wrote_summary_plot=wrote_summary_plot,
        wrote_heatmap_plot=wrote_heatmap_plot,
    )

    print("DONE")
    print(f"source_column={source_col}")
    print(f"selected_hkls_count={len(selected_hkls) if selected_hkls else 'all'}")
    print(f"matched_rows={len(matched):,}")
    print(f"joined_component_rows={len(joined_component_obs):,}")
    print(f"component_per_hkl_rows={len(per_hkl):,}")
    print(f"component_summary_rows={len(component_summary):,}")
    print(f"per_hkl_scatter_plots={scatter_count:,}")


if __name__ == "__main__":
    main()
