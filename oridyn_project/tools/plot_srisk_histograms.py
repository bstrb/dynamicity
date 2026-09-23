#!/usr/bin/env python3
"""Plot presentation-ready histograms of actual S_risk = Eg * M2.

The script reads an existing full-population SQLite score cache.  It does not
run stream generation, merging, Partialator, or refinement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Iterable, Iterator

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_CACHE = DEFAULT_ROOT / "oridyn_v6_full_population_sweep_20260717" / "full_population_cache.sqlite"
DEFAULT_SCORE = "auto"
DEFAULT_CHUNK_SIZE = 500_000
DEFAULT_BINS = 80
DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_HKLS_PER_RESOLUTION_BIN = 2
DEFAULT_EPS = 1.0e-12

HKL_COLUMNS = ("h", "k", "l")
SCORE_TABLE = "score_cache"
UNIT_CELL_RE = re.compile(
    r"^\s*(a|b|c|al|be|ga)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z]+)?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="SQLite full-population score cache")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG/TSV/metadata outputs")
    parser.add_argument(
        "--score",
        "--score-column",
        dest="score",
        default=DEFAULT_SCORE,
        help="Score source: auto, Eg*M2, or an explicit cache column name",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="SQLite rows per streaming chunk")
    parser.add_argument("--max-rows", type=int, default=None, help="Bound analysis to ordinal < max_rows for smoke tests")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Histogram bin count")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Reservoir sample size for display ranges/quantiles")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Epsilon for log10(S_risk + eps)")
    parser.add_argument(
        "--select-high-redundancy-by-resolution",
        action="store_true",
        help="Select high-redundancy signed HKLs from low/mid/high resolution bins when resolution can be computed",
    )
    parser.add_argument(
        "--hkls-per-resolution-bin",
        type=int,
        default=DEFAULT_HKLS_PER_RESOLUTION_BIN,
        help="Example signed HKLs to select per low/mid/high resolution bin",
    )
    parser.add_argument(
        "--source-stream",
        type=Path,
        default=None,
        help="Optional stream path for unit-cell parsing; auto-inferred from cache metadata when omitted",
    )
    args = parser.parse_args()
    args.cache = args.cache.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.source_stream is not None:
        args.source_stream = args.source_stream.expanduser().resolve()
    if int(args.chunk_size) < 1:
        raise SystemExit("--chunk-size must be >= 1")
    if args.max_rows is not None and int(args.max_rows) < 1:
        raise SystemExit("--max-rows must be >= 1 when provided")
    if int(args.bins) < 10:
        raise SystemExit("--bins must be >= 10")
    if int(args.sample_size) < 100:
        raise SystemExit("--sample-size must be >= 100")
    if not math.isfinite(float(args.eps)) or float(args.eps) <= 0.0:
        raise SystemExit("--eps must be a positive finite value")
    if int(args.hkls_per_resolution_bin) < 1:
        raise SystemExit("--hkls-per-resolution-bin must be >= 1")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def connect_readonly(cache: Path) -> sqlite3.Connection:
    if not cache.is_file():
        raise SystemExit(f"--cache not found: {cache}")
    conn = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def table_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({SCORE_TABLE})").fetchall()
    if not rows:
        raise SystemExit(f"SQLite table {SCORE_TABLE!r} was not found")
    return [str(row[1]) for row in rows]


def resolve_score(columns: list[str], requested: str) -> dict[str, Any]:
    column_set = set(columns)
    requested = str(requested).strip()
    stored_candidates = [column for column in ("EgM2", "egm2", "score_eg_m2") if column in column_set]
    if requested.lower() == "auto":
        if len(stored_candidates) > 1:
            raise SystemExit(
                "Ambiguous S_risk columns for --score auto: "
                f"{stored_candidates}. Available columns: {columns}. Use --score COLUMN or --score 'Eg*M2'."
            )
        if len(stored_candidates) == 1:
            column = stored_candidates[0]
            return {"mode": "column", "sql": f'"{column}"', "label": column, "required_columns": [column]}
        if {"Eg", "M2"} <= column_set:
            return {"mode": "expression", "sql": '"Eg" * "M2"', "label": "Eg * M2", "required_columns": ["Eg", "M2"]}
        raise SystemExit(
            "Could not resolve S_risk automatically. Need EgM2/egm2/score_eg_m2 or both Eg and M2. "
            f"Available columns: {columns}"
        )
    if requested.replace(" ", "").lower() in {"eg*m2", "egm2_expr"}:
        missing = sorted({"Eg", "M2"} - column_set)
        if missing:
            raise SystemExit(f"--score Eg*M2 requested but cache is missing {missing}. Available columns: {columns}")
        return {"mode": "expression", "sql": '"Eg" * "M2"', "label": "Eg * M2", "required_columns": ["Eg", "M2"]}
    if requested not in column_set:
        raise SystemExit(f"--score column {requested!r} not found. Available columns: {columns}")
    return {"mode": "column", "sql": f'"{requested}"', "label": requested, "required_columns": [requested]}


def where_clause(max_rows: int | None) -> tuple[str, list[Any]]:
    if max_rows is None:
        return "", []
    return "WHERE ordinal < ?", [int(max_rows)]


def count_rows(conn: sqlite3.Connection, max_rows: int | None) -> int:
    where, params = where_clause(max_rows)
    return int(conn.execute(f"SELECT COUNT(*) FROM {SCORE_TABLE} {where}", params).fetchone()[0])


def iter_score_chunks(
    conn: sqlite3.Connection,
    score_sql: str,
    chunk_size: int,
    max_rows: int | None,
) -> Iterator[np.ndarray]:
    where, params = where_clause(max_rows)
    query = f"SELECT {score_sql} AS srisk FROM {SCORE_TABLE} {where} ORDER BY ordinal"
    cursor = conn.execute(query, params)
    while True:
        rows = cursor.fetchmany(int(chunk_size))
        if not rows:
            break
        yield np.asarray([row[0] for row in rows], dtype=float)


class Reservoir:
    def __init__(self, capacity: int, seed: int = 20260812) -> None:
        self.capacity = int(capacity)
        self.values = np.empty(self.capacity, dtype=float)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        for value in values:
            self.seen += 1
            if self.size < self.capacity:
                self.values[self.size] = float(value)
                self.size += 1
                continue
            slot = int(self.rng.integers(0, self.seen))
            if slot < self.capacity:
                self.values[slot] = float(value)

    def array(self) -> np.ndarray:
        return self.values[: self.size].copy()


def first_pass(
    conn: sqlite3.Connection,
    score_sql: str,
    chunk_size: int,
    max_rows: int | None,
    sample_size: int,
    eps: float,
    total_rows: int,
) -> dict[str, Any]:
    started = time.monotonic()
    reservoir = Reservoir(sample_size)
    processed = 0
    finite = 0
    nonfinite = 0
    negative = 0
    min_score = float("inf")
    max_score = float("-inf")
    min_log = float("inf")
    max_log = float("-inf")
    finite_log = 0
    for chunk in iter_score_chunks(conn, score_sql, chunk_size, max_rows):
        processed += int(len(chunk))
        finite_mask = np.isfinite(chunk)
        values = chunk[finite_mask]
        finite += int(len(values))
        nonfinite += int(len(chunk) - len(values))
        if len(values):
            negative += int(np.sum(values < 0.0))
            min_score = min(min_score, float(np.min(values)))
            max_score = max(max_score, float(np.max(values)))
            reservoir.update(values)
            log_mask = (values + eps) > 0.0
            if np.any(log_mask):
                logs = np.log10(values[log_mask] + eps)
                finite_log += int(len(logs))
                min_log = min(min_log, float(np.min(logs)))
                max_log = max(max_log, float(np.max(logs)))
        if processed == total_rows or processed % max(int(chunk_size), 1) == 0:
            elapsed = max(time.monotonic() - started, 1.0e-9)
            rate = processed / elapsed
            pct = 100.0 * processed / max(1, total_rows)
            eta = (total_rows - processed) / rate if rate > 0 else float("inf")
            log(
                f"first pass: {processed:,}/{total_rows:,} rows ({pct:.1f}%), "
                f"elapsed={format_duration(elapsed)}, rate={rate:,.0f}/s, eta={format_duration(eta)}"
            )
    sample = reservoir.array()
    if sample.size == 0:
        raise SystemExit("No finite S_risk values were found")
    q = np.quantile(sample, [0.50, 0.90, 0.95, 0.99, 0.995])
    log_sample = np.log10(sample[(sample + eps) > 0.0] + eps)
    log_q = np.quantile(log_sample, [0.50, 0.90, 0.95, 0.99, 0.995]) if log_sample.size else np.full(5, np.nan)
    return {
        "rows_processed": int(processed),
        "finite_count": int(finite),
        "nonfinite_count": int(nonfinite),
        "negative_count": int(negative),
        "finite_log_count": int(finite_log),
        "min": float(min_score),
        "max": float(max_score),
        "log_min": float(min_log),
        "log_max": float(max_log),
        "sample_size": int(sample.size),
        "sample_quantiles": {"p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3]), "p99_5": float(q[4])},
        "log_sample_quantiles": {"p50": float(log_q[0]), "p90": float(log_q[1]), "p95": float(log_q[2]), "p99": float(log_q[3]), "p99_5": float(log_q[4])},
    }


def histogram_edges(stats: dict[str, Any], bins: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    min_score = float(stats["min"])
    max_score = float(stats["max"])
    p995 = float(stats["sample_quantiles"]["p99_5"])
    actual_low = min(0.0, min_score) if min_score >= 0.0 else min_score
    display_high = p995 if math.isfinite(p995) and p995 > actual_low else max_score
    if not math.isfinite(display_high) or display_high <= actual_low:
        display_high = actual_low + 1.0
    actual_edges = np.linspace(actual_low, display_high, int(bins) + 1)

    log_low = float(stats["log_min"])
    log_high = float(stats["log_max"])
    if not math.isfinite(log_low) or not math.isfinite(log_high) or log_high <= log_low:
        log_low, log_high = -12.0, 0.0
    log_edges = np.linspace(log_low, log_high, int(bins) + 1)
    edge_info = {
        "actual_display_low": float(actual_low),
        "actual_display_high": float(display_high),
        "actual_hist_tail_is_clipped_to_last_bin": bool(max_score > display_high),
        "log_display_low": float(log_low),
        "log_display_high": float(log_high),
    }
    return actual_edges, log_edges, edge_info


def second_pass_histograms(
    conn: sqlite3.Connection,
    score_sql: str,
    chunk_size: int,
    max_rows: int | None,
    total_rows: int,
    eps: float,
    actual_edges: np.ndarray,
    log_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    started = time.monotonic()
    processed = 0
    actual_counts = np.zeros(len(actual_edges) - 1, dtype=np.int64)
    log_counts = np.zeros(len(log_edges) - 1, dtype=np.int64)
    actual_clipped_high = 0
    log_skipped = 0
    for chunk in iter_score_chunks(conn, score_sql, chunk_size, max_rows):
        processed += int(len(chunk))
        values = chunk[np.isfinite(chunk)]
        if len(values):
            actual_clipped_high += int(np.sum(values > actual_edges[-1]))
            clipped = np.clip(values, actual_edges[0], actual_edges[-1])
            actual_counts += np.histogram(clipped, bins=actual_edges)[0].astype(np.int64)
            log_mask = (values + eps) > 0.0
            log_skipped += int(len(values) - np.sum(log_mask))
            if np.any(log_mask):
                log_values = np.log10(values[log_mask] + eps)
                log_counts += np.histogram(log_values, bins=log_edges)[0].astype(np.int64)
        if processed == total_rows or processed % max(int(chunk_size), 1) == 0:
            elapsed = max(time.monotonic() - started, 1.0e-9)
            rate = processed / elapsed
            pct = 100.0 * processed / max(1, total_rows)
            eta = (total_rows - processed) / rate if rate > 0 else float("inf")
            log(
                f"hist pass: {processed:,}/{total_rows:,} rows ({pct:.1f}%), "
                f"elapsed={format_duration(elapsed)}, rate={rate:,.0f}/s, eta={format_duration(eta)}"
            )
    return actual_counts, log_counts, {"actual_clipped_high": actual_clipped_high, "log_skipped": log_skipped}


def plot_histogram(
    edges: np.ndarray,
    counts: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    quantiles: dict[str, float],
    note: str = "",
) -> None:
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    widths = np.diff(edges)
    ax.bar(edges[:-1], counts, width=widths, align="edge", color="#4969a8", edgecolor="white", linewidth=0.35)
    colors = {"p50": "#202020", "p95": "#c95f35", "p99": "#8b1e3f"}
    labels = {"p50": "median", "p95": "95th percentile", "p99": "99th percentile"}
    for key in ["p50", "p95", "p99"]:
        value = quantiles.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        if edges[0] <= float(value) <= edges[-1]:
            ax.axvline(float(value), color=colors[key], linestyle="--", linewidth=2.0, label=f"{labels[key]} = {value:.3g}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observation count")
    ax.grid(axis="y", alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    if note:
        ax.text(0.01, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=12)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"wrote {path}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def infer_source_stream(cache: Path) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    base = cache.parent
    for name in ["parameters.json", "run_metadata.json", "cache_provenance.json", "accepted_population_validation.json"]:
        payload = read_json(base / name)
        if not payload:
            continue
        candidates = [
            payload.get("source_stream"),
            payload.get("input_stream"),
        ]
        if isinstance(payload.get("source_stream"), dict):
            candidates.append(payload["source_stream"].get("path"))
        nested = payload.get("source_stream_validation")
        if isinstance(nested, dict):
            candidates.append(nested.get("source_stream"))
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate)).expanduser()
            if path.is_file():
                return path.resolve(), warnings
    warnings.append("Could not infer source stream from cache-side metadata; resolution plots will be unbinned unless --source-stream is supplied.")
    return None, warnings


def parse_unit_cell(path: Path) -> tuple[dict[str, float] | None, str]:
    values: dict[str, float] = {}
    in_unit_cell = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if line.startswith("----- Begin unit cell"):
                    in_unit_cell = True
                    continue
                if in_unit_cell and line.startswith("----- End unit cell"):
                    break
                if not in_unit_cell:
                    continue
                match = UNIT_CELL_RE.match(line)
                if not match:
                    continue
                key, raw_value, unit = match.groups()
                value = float(raw_value)
                if key in {"a", "b", "c"}:
                    unit_text = (unit or "A").lower()
                    if unit_text == "nm":
                        value *= 10.0
                    elif unit_text not in {"a", "angstrom", "angstroms"}:
                        return None, f"Unsupported unit-cell unit {unit!r} at line {line_number}"
                values[key] = value
    except OSError as exc:
        return None, f"Could not read source stream {path}: {exc}"
    missing = sorted(set(["a", "b", "c", "al", "be", "ga"]) - set(values))
    if missing:
        return None, f"Could not parse complete unit cell from {path}; missing {missing}"
    return values, ""


def reciprocal_metric_from_cell(cell: dict[str, float]) -> np.ndarray:
    a = float(cell["a"])
    b = float(cell["b"])
    c = float(cell["c"])
    alpha = math.radians(float(cell["al"]))
    beta = math.radians(float(cell["be"]))
    gamma = math.radians(float(cell["ga"]))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("sin(gamma) is too small")
    avec = np.array([a, 0.0, 0.0], dtype=float)
    bvec = np.array([b * math.cos(gamma), b * sin_gamma, 0.0], dtype=float)
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        raise ValueError("computed c-axis z component is nonpositive")
    cvec = np.array([cx, cy, math.sqrt(cz_sq)], dtype=float)
    direct = np.column_stack([avec, bvec, cvec])
    reciprocal = np.linalg.inv(direct).T
    metric = reciprocal.T @ reciprocal
    if not np.isfinite(metric).all():
        raise ValueError("nonfinite reciprocal metric")
    return metric


def d_spacing_from_hkl(h: np.ndarray, k: np.ndarray, l: np.ndarray, metric: np.ndarray) -> np.ndarray:
    hkls = np.column_stack([h, k, l]).astype(float)
    squared = np.einsum("...i,ij,...j->...", hkls, metric, hkls)
    g_norm = np.sqrt(np.clip(squared, 0.0, None))
    return np.divide(1.0, g_norm, out=np.full_like(g_norm, np.nan, dtype=float), where=g_norm > 0.0)


def load_hkl_redundancy(conn: sqlite3.Connection, max_rows: int | None, metric: np.ndarray | None) -> list[dict[str, Any]]:
    if max_rows is not None:
        counts: Counter[tuple[int, int, int]] = Counter()
        cursor = conn.execute(f"SELECT h,k,l FROM {SCORE_TABLE} WHERE ordinal < ? ORDER BY ordinal", [int(max_rows)])
        while True:
            batch = cursor.fetchmany(100_000)
            if not batch:
                break
            counts.update((int(h), int(k), int(l)) for h, k, l in batch)
        rows = [
            {"h": h, "k": k, "l": l, "redundancy": int(n)}
            for (h, k, l), n in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]))
        ]
    else:
        query = f"""
            SELECT h,k,l,COUNT(*) AS redundancy
            FROM {SCORE_TABLE}
            GROUP BY h,k,l
            ORDER BY redundancy DESC,h,k,l
        """
        rows = [{"h": int(h), "k": int(k), "l": int(l), "redundancy": int(n)} for h, k, l, n in conn.execute(query)]
    if metric is not None and rows:
        h = np.asarray([row["h"] for row in rows], dtype=int)
        k = np.asarray([row["k"] for row in rows], dtype=int)
        l = np.asarray([row["l"] for row in rows], dtype=int)
        d_values = d_spacing_from_hkl(h, k, l, metric)
        for row, d_value in zip(rows, d_values):
            row["resolution_A"] = float(d_value) if math.isfinite(float(d_value)) else None
            row["reciprocal_resolution_Ainv"] = float(1.0 / d_value) if math.isfinite(float(d_value)) and d_value > 0.0 else None
    else:
        for row in rows:
            row["resolution_A"] = None
            row["reciprocal_resolution_Ainv"] = None
    return rows


def select_hkls(
    hkl_rows: list[dict[str, Any]],
    by_resolution: bool,
    per_bin: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not hkl_rows:
        return [], {"method": "none", "warnings": ["No signed HKLs found in cache subset"]}
    resolution_available = all(row.get("reciprocal_resolution_Ainv") is not None for row in hkl_rows)
    selected: list[dict[str, Any]] = []
    if by_resolution and resolution_available:
        values = np.asarray([row["reciprocal_resolution_Ainv"] for row in hkl_rows], dtype=float)
        q1, q2 = np.quantile(values[np.isfinite(values)], [1.0 / 3.0, 2.0 / 3.0])
        bins = [
            ("low-resolution", lambda x: x <= q1),
            ("mid-resolution", lambda x: q1 < x <= q2),
            ("high-resolution", lambda x: x > q2),
        ]
        for label, predicate in bins:
            pool = [row for row in hkl_rows if predicate(float(row["reciprocal_resolution_Ainv"]))]
            pool.sort(key=lambda row: (-int(row["redundancy"]), int(row["h"]), int(row["k"]), int(row["l"])))
            for row in pool[:per_bin]:
                item = dict(row)
                item["resolution_bin"] = label
                selected.append(item)
        return selected, {
            "method": "high_redundancy_by_resolution_tertiles",
            "resolution_available": True,
            "reciprocal_resolution_tertile_edges_Ainv": [float(q1), float(q2)],
            "warnings": [],
        }
    hkl_rows = sorted(hkl_rows, key=lambda row: (-int(row["redundancy"]), int(row["h"]), int(row["k"]), int(row["l"])))
    target = max(6, 3 * int(per_bin))
    for row in hkl_rows[:target]:
        item = dict(row)
        item["resolution_bin"] = "resolution unavailable" if not resolution_available else "all-resolution high-redundancy"
        selected.append(item)
    warning = "" if resolution_available else "Resolution unavailable; selected highest-redundancy signed HKLs overall."
    return selected, {"method": "high_redundancy_overall", "resolution_available": resolution_available, "warnings": [warning] if warning else []}


def fetch_scores_for_hkl(
    conn: sqlite3.Connection,
    score_sql: str,
    h: int,
    k: int,
    l: int,
    max_rows: int | None,
) -> np.ndarray:
    where = "WHERE h=? AND k=? AND l=?"
    params: list[Any] = [int(h), int(k), int(l)]
    if max_rows is not None:
        where += " AND ordinal < ?"
        params.append(int(max_rows))
    rows = conn.execute(f"SELECT {score_sql} AS srisk FROM {SCORE_TABLE} {where} ORDER BY ordinal", params).fetchall()
    values = np.asarray([row[0] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summary_stats(values: np.ndarray, eps: float) -> dict[str, Any]:
    if values.size == 0:
        return {}
    q = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
    logs = np.log10(values[(values + eps) > 0.0] + eps)
    out = {
        "min": float(np.min(values)),
        "median": float(q[0]),
        "mean": float(np.mean(values)),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(np.max(values)),
    }
    if logs.size:
        lq = np.quantile(logs, [0.5, 0.9, 0.95, 0.99])
        out.update(
            {
                "log10_min": float(np.min(logs)),
                "log10_median": float(lq[0]),
                "log10_mean": float(np.mean(logs)),
                "log10_p90": float(lq[1]),
                "log10_p95": float(lq[2]),
                "log10_p99": float(lq[3]),
                "log10_max": float(np.max(logs)),
            }
        )
    return out


def plot_hkl_grid(selected: list[dict[str, Any]], values_by_hkl: dict[tuple[int, int, int], np.ndarray], path: Path, log_scale: bool, eps: float) -> None:
    if not selected:
        return
    n = len(selected)
    cols = 3 if n >= 3 else n
    rows = int(math.ceil(n / cols))
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes_array, selected):
        hkl = (int(row["h"]), int(row["k"]), int(row["l"]))
        values = values_by_hkl.get(hkl, np.array([], dtype=float))
        plot_values = np.log10(values[(values + eps) > 0.0] + eps) if log_scale else values
        if plot_values.size:
            ax.hist(plot_values, bins=min(40, max(12, int(math.sqrt(plot_values.size)))), color="#4f9d69", edgecolor="white", linewidth=0.3)
        d_text = "d=n/a" if row.get("resolution_A") is None else f"d={float(row['resolution_A']):.2f} A"
        median = np.median(values) if values.size else float("nan")
        title = f"({hkl[0]}, {hkl[1]}, {hkl[2]}) {d_text}\nN={int(row['redundancy'])}, median S={median:.3g}"
        ax.set_title(title)
        ax.set_xlabel("log10(S_risk + eps)" if log_scale else "actual S_risk")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes_array[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Selected signed-HKL actual S_risk distributions" + (" (log10 view)" if log_scale else ""), fontsize=18)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log(f"wrote {path}")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "h",
        "k",
        "l",
        "resolution_bin",
        "resolution_A",
        "redundancy",
        "min",
        "median",
        "mean",
        "p90",
        "p95",
        "p99",
        "max",
        "log10_min",
        "log10_median",
        "log10_mean",
        "log10_p90",
        "log10_p95",
        "log10_p99",
        "log10_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    log(f"wrote {path}")


def main() -> int:
    args = parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "started_utc": started_utc,
        "script": str(Path(__file__).resolve()),
        "cache": str(args.cache),
        "output_dir": str(args.output_dir),
        "parameters": {
            "score": args.score,
            "chunk_size": int(args.chunk_size),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "bins": int(args.bins),
            "sample_size": int(args.sample_size),
            "eps": float(args.eps),
            "select_high_redundancy_by_resolution": bool(args.select_high_redundancy_by_resolution),
            "hkls_per_resolution_bin": int(args.hkls_per_resolution_bin),
            "source_stream": None if args.source_stream is None else str(args.source_stream),
        },
        "warnings": [],
    }
    log(f"reading cache schema: {args.cache}")
    with connect_readonly(args.cache) as conn:
        columns = table_columns(conn)
        score = resolve_score(columns, args.score)
        total_rows = count_rows(conn, args.max_rows)
        metadata["cache_columns"] = columns
        metadata["resolved_score"] = score
        metadata["rows_to_process"] = int(total_rows)
        log(f"resolved S_risk as {score['label']} using columns {score['required_columns']}")
        log(f"rows to process: {total_rows:,}")
        if total_rows == 0:
            raise SystemExit("No rows selected from score_cache")

        first_stats = first_pass(conn, score["sql"], int(args.chunk_size), args.max_rows, int(args.sample_size), float(args.eps), total_rows)
        actual_edges, log_edges, edge_info = histogram_edges(first_stats, int(args.bins))
        actual_counts, log_counts, hist_stats = second_pass_histograms(
            conn,
            score["sql"],
            int(args.chunk_size),
            args.max_rows,
            total_rows,
            float(args.eps),
            actual_edges,
            log_edges,
        )
        metadata["whole_dataset_stats"] = first_stats
        metadata["histogram_display"] = edge_info
        metadata["histogram_counts"] = hist_stats

        actual_note = ""
        if hist_stats["actual_clipped_high"]:
            actual_note = f"Last bin includes {hist_stats['actual_clipped_high']:,} values above display range"
        plot_histogram(
            actual_edges,
            actual_counts,
            args.output_dir / "whole_dataset_srisk_histogram.png",
            "Whole-dataset actual S_risk histogram (not normalized)",
            "actual S_risk = Eg * M2",
            first_stats["sample_quantiles"],
            actual_note,
        )
        plot_histogram(
            log_edges,
            log_counts,
            args.output_dir / "whole_dataset_log10_srisk_histogram.png",
            "Whole-dataset log10(actual S_risk + eps) histogram",
            "log10(S_risk + eps)",
            first_stats["log_sample_quantiles"],
            "",
        )

        source_stream = args.source_stream
        inferred_warnings: list[str] = []
        if source_stream is None:
            source_stream, inferred_warnings = infer_source_stream(args.cache)
            metadata["warnings"].extend(inferred_warnings)
        metadata["source_stream_used_for_resolution"] = None if source_stream is None else str(source_stream)
        metric = None
        cell = None
        if source_stream is not None:
            cell, warning = parse_unit_cell(source_stream)
            if warning:
                metadata["warnings"].append(warning)
            elif cell is not None:
                try:
                    metric = reciprocal_metric_from_cell(cell)
                except ValueError as exc:
                    metadata["warnings"].append(f"Could not compute reciprocal metric from unit cell: {exc}")
        metadata["unit_cell"] = cell
        metadata["resolution_method"] = "source_stream_unit_cell" if metric is not None else "unavailable"

        log("selecting high-redundancy signed HKL examples")
        hkl_rows = load_hkl_redundancy(conn, args.max_rows, metric)
        selected, selection_meta = select_hkls(
            hkl_rows,
            bool(args.select_high_redundancy_by_resolution),
            int(args.hkls_per_resolution_bin),
        )
        metadata["hkl_selection"] = selection_meta
        metadata["warnings"].extend(selection_meta.get("warnings", []))
        values_by_hkl: dict[tuple[int, int, int], np.ndarray] = {}
        summary_rows: list[dict[str, Any]] = []
        for row in selected:
            hkl = (int(row["h"]), int(row["k"]), int(row["l"]))
            values = fetch_scores_for_hkl(conn, score["sql"], *hkl, args.max_rows)
            values_by_hkl[hkl] = values
            out_row = dict(row)
            out_row.update(summary_stats(values, float(args.eps)))
            summary_rows.append(out_row)
        write_tsv(args.output_dir / "selected_hkl_summary.tsv", summary_rows)
        plot_hkl_grid(selected, values_by_hkl, args.output_dir / "selected_hkl_srisk_histograms.png", False, float(args.eps))
        plot_hkl_grid(selected, values_by_hkl, args.output_dir / "selected_hkl_log10_srisk_histograms.png", True, float(args.eps))

    outputs = {
        "whole_dataset_srisk_histogram": str(args.output_dir / "whole_dataset_srisk_histogram.png"),
        "whole_dataset_log10_srisk_histogram": str(args.output_dir / "whole_dataset_log10_srisk_histogram.png"),
        "selected_hkl_srisk_histograms": str(args.output_dir / "selected_hkl_srisk_histograms.png"),
        "selected_hkl_log10_srisk_histograms": str(args.output_dir / "selected_hkl_log10_srisk_histograms.png"),
        "selected_hkl_summary": str(args.output_dir / "selected_hkl_summary.tsv"),
        "metadata": str(args.output_dir / "srisk_histogram_metadata.json"),
    }
    metadata["outputs"] = outputs
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "srisk_histogram_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {args.output_dir / 'srisk_histogram_metadata.json'}")
    log("done")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
