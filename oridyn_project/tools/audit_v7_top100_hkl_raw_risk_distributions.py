#!/usr/bin/env python3
"""Audit within-HKL raw S_risk distributions for global top-100 V7 rows.

This read-only audit selects the distinct signed HKLs represented in the
global top 100 raw S_risk observations, then extracts all cache rows for those
HKLs and summarizes their raw, log, sqrt, and rank-percentile distributions.

It does not create filtered streams, run Partialator, run merging, or modify
the cache.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_CACHE = (
    DATASET_ROOT
    / "oridyn_v7_geometric_risk_s0_0p005_sig0p03_rcut0p20_20260904"
    / "v7_geometric_risk_cache.sqlite"
)
DEFAULT_OUTPUT_DIR = (
    DATASET_ROOT
    / "oridyn_v7_geometric_risk_s0_0p005_sig0p03_rcut0p20_20260904"
    / "audit"
    / "top100_hkl_raw_risk_distributions"
)
DEFAULT_PREVIOUS_AUDIT = (
    DATASET_ROOT
    / "oridyn_v7_geometric_risk_s0_0p002_rcut0p20_20260903"
    / "audit"
    / "top100_hkl_raw_risk_distributions"
)
DEFAULT_LABEL = "V7b geometric risk, s0=0.005, sigma_C=0.03, r_cut=0.20"
DEFAULT_TOP_N = 100
EPSILON_FOR_LOG10 = 1.0e-300
PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
PERCENTILE_NAMES = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
TAIL_PERCENTILES = [90.0, 95.0, 97.5, 99.0]
REQUIRED_COLUMNS = {
    "source_filename",
    "event",
    "h",
    "k",
    "l",
    "exact_key_text",
    "source_order",
    "E_g",
    "R_env",
    "S_risk",
}
OPTIONAL_COLUMNS = [
    "geometric_neighbour_count",
    "nonzero_neighbour_count",
    "active_neighbour_count",
    "top_neighbour_h",
    "top_neighbour_k",
    "top_neighbour_l",
    "top_neighbour_E_h",
    "top_neighbour_d_gh",
    "top_neighbour_C",
    "top_neighbour_contribution",
    "v6_Eg",
    "v6_M2",
    "v6_S_risk",
]


@dataclass(frozen=True)
class Observation:
    source_filename: str
    event: str
    h: int
    k: int
    l: int
    exact_key_text: str
    source_order: int
    E_g: float
    R_env: float
    S_risk: float
    extra: dict[str, Any]


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--previous-audit-dir", type=Path, default=DEFAULT_PREVIOUS_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.cache = args.cache.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.previous_audit_dir = args.previous_audit_dir.expanduser().resolve() if args.previous_audit_dir else None
    if not args.cache.is_file():
        raise SystemExit(f"Cache not found: {args.cache}")
    if int(args.top_n) < 1:
        raise SystemExit("--top-n must be >= 1")
    args.top_n = int(args.top_n)
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        existing = [item for item in path.iterdir() if item.is_file()]
        if existing and not overwrite:
            preview = "\n".join(str(item) for item in sorted(existing)[:30])
            extra = "" if len(existing) <= 30 else f"\n... and {len(existing) - 30} more"
            raise SystemExit(f"Refusing to overwrite existing audit outputs; pass --overwrite if intentional:\n{preview}{extra}")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sqlite_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def cache_columns(conn: sqlite3.Connection) -> list[str]:
    return [str(row[1]) for row in conn.execute("PRAGMA table_info(v7_score_cache)").fetchall()]


def require_schema(conn: sqlite3.Connection) -> list[str]:
    tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "v7_score_cache" not in tables:
        raise SystemExit("Cache is missing table v7_score_cache")
    columns = cache_columns(conn)
    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise SystemExit(f"v7_score_cache is missing required columns: {missing}")
    return columns


def source_index(source: str) -> str:
    match = re.search(r"_(\d+)\.h5$", str(source))
    return match.group(1) if match else ""


def qstats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {name: math.nan for name in PERCENTILE_NAMES}
    qs = np.percentile(arr, PERCENTILES)
    return {name: float(value) for name, value in zip(PERCENTILE_NAMES, qs)}


def prefixed_qstats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in qstats(values).items()}


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if math.isfinite(float(den)) and float(den) != 0.0 else math.nan


def safe_fraction(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else math.nan


def hkl_key(row: sqlite3.Row | Observation | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(row, tuple):
        return int(row[0]), int(row[1]), int(row[2])
    return int(row["h"] if isinstance(row, sqlite3.Row) else row.h), int(row["k"] if isinstance(row, sqlite3.Row) else row.k), int(row["l"] if isinstance(row, sqlite3.Row) else row.l)


def fetch_top_rows(conn: sqlite3.Connection, top_n: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT source_filename,event,h,k,l,exact_key_text,source_order,E_g,R_env,S_risk
        FROM v7_score_cache
        ORDER BY S_risk DESC, source_order ASC
        LIMIT ?
        """,
        (int(top_n),),
    ).fetchall()


def fetch_observations_for_hkls(conn: sqlite3.Connection, hkls: list[tuple[int, int, int]], columns: list[str]) -> list[Observation]:
    select_columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "exact_key_text",
        "source_order",
        "E_g",
        "R_env",
        "S_risk",
        *[column for column in OPTIONAL_COLUMNS if column in columns],
    ]
    clauses = []
    params: list[int] = []
    for h, k, l in hkls:
        clauses.append("(h=? AND k=? AND l=?)")
        params.extend([int(h), int(k), int(l)])
    query = f"""
        SELECT {','.join(select_columns)}
        FROM v7_score_cache
        WHERE {' OR '.join(clauses)}
        ORDER BY h,k,l,source_order
    """
    rows = conn.execute(query, params).fetchall()
    observations: list[Observation] = []
    for row in rows:
        extra = {column: row[column] for column in select_columns if column in OPTIONAL_COLUMNS}
        observations.append(
            Observation(
                source_filename=str(row["source_filename"]),
                event=str(row["event"]),
                h=int(row["h"]),
                k=int(row["k"]),
                l=int(row["l"]),
                exact_key_text=str(row["exact_key_text"]),
                source_order=int(row["source_order"]),
                E_g=float(row["E_g"]),
                R_env=float(row["R_env"]),
                S_risk=float(row["S_risk"]),
                extra=extra,
            )
        )
    return observations


def group_observations(observations: list[Observation]) -> dict[tuple[int, int, int], list[Observation]]:
    grouped: dict[tuple[int, int, int], list[Observation]] = {}
    for obs in observations:
        grouped.setdefault((obs.h, obs.k, obs.l), []).append(obs)
    for obs_list in grouped.values():
        obs_list.sort(key=lambda obs: (-obs.S_risk, obs.source_order))
    return grouped


def rank_percentile(rank: int, n: int) -> float:
    if n <= 1:
        return 100.0
    return 100.0 * (n - rank) / (n - 1)


def annotated_observations(grouped: dict[tuple[int, int, int], list[Observation]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    top10_rows: list[dict[str, Any]] = []
    for hkl in sorted(grouped):
        obs_list = grouped[hkl]
        scores = np.asarray([obs.S_risk for obs in obs_list], dtype=np.float64)
        mean = float(scores.mean()) if scores.size else math.nan
        std = float(scores.std(ddof=0)) if scores.size else math.nan
        for rank, obs in enumerate(obs_list, start=1):
            percentile = rank_percentile(rank, len(obs_list))
            z = (obs.S_risk - mean) / std if math.isfinite(std) and std > 0.0 else math.nan
            row = {
                "source_index": source_index(obs.source_filename),
                "source_filename": obs.source_filename,
                "event": obs.event,
                "h": obs.h,
                "k": obs.k,
                "l": obs.l,
                "exact_key_text": obs.exact_key_text,
                "source_order": obs.source_order,
                "rank_within_hkl_desc": rank,
                "within_hkl_rank_percentile": percentile,
                "E_g": obs.E_g,
                "R_env": obs.R_env,
                "S_risk": obs.S_risk,
                "log10_S_risk_plus_epsilon": math.log10(obs.S_risk + EPSILON_FOR_LOG10),
                "sqrt_S_risk": math.sqrt(max(obs.S_risk, 0.0)),
                "z_raw_S_risk": z,
            }
            row.update(obs.extra)
            all_rows.append(row)
            if rank <= 10:
                top10_rows.append(row)
    return all_rows, top10_rows


def top_hkl_selection_rows(top_rows: list[sqlite3.Row]) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]]]:
    counts: dict[tuple[int, int, int], int] = {}
    maxima: dict[tuple[int, int, int], float] = {}
    for row in top_rows:
        key = hkl_key(row)
        counts[key] = counts.get(key, 0) + 1
        maxima[key] = max(maxima.get(key, -math.inf), float(row["S_risk"]))
    hkls = sorted(counts)
    rows = [
        {
            "h": h,
            "k": k,
            "l": l,
            "top100_count": counts[(h, k, l)],
            "max_top100_S_risk": maxima[(h, k, l)],
            "selection_note": "signed HKL appears in global top 100 raw V7b S_risk observations",
        }
        for h, k, l in hkls
    ]
    return hkls, rows


def shape_category(row: dict[str, Any]) -> str:
    ratio = float(row.get("S_risk_max_over_median", math.nan))
    p95_ratio = float(row.get("S_risk_p95_over_median", math.nan))
    top1 = float(row.get("top1_S_risk_mass_fraction", math.nan))
    if (math.isfinite(ratio) and ratio >= 1.0e4) or (math.isfinite(top1) and top1 >= 0.10):
        return "very broad right-skew/extreme raw tail"
    if (math.isfinite(ratio) and ratio >= 1.0e2) or (math.isfinite(p95_ratio) and p95_ratio >= 10.0):
        return "broad right-skew"
    if (math.isfinite(ratio) and ratio >= 10.0) or (math.isfinite(p95_ratio) and p95_ratio >= 3.0):
        return "moderate right-skew"
    return "smooth/compact raw distribution"


def distribution_summary(grouped: dict[tuple[int, int, int], list[Observation]], top_counts: dict[tuple[int, int, int], int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hkl in sorted(grouped):
        obs_list = grouped[hkl]
        eg = np.asarray([obs.E_g for obs in obs_list], dtype=np.float64)
        renv = np.asarray([obs.R_env for obs in obs_list], dtype=np.float64)
        srisk = np.asarray([obs.S_risk for obs in obs_list], dtype=np.float64)
        median = float(np.median(srisk)) if srisk.size else math.nan
        mad = float(np.median(np.abs(srisk - median))) if srisk.size else math.nan
        total_mass = float(srisk.sum())
        sorted_scores = np.sort(srisk)[::-1]
        row: dict[str, Any] = {
            "h": hkl[0],
            "k": hkl[1],
            "l": hkl[2],
            "n_observations": len(obs_list),
            "top100_count": top_counts.get(hkl, 0),
        }
        row.update(prefixed_qstats("Eg", eg))
        row.update(prefixed_qstats("R_env", renv))
        row.update(prefixed_qstats("S_risk", srisk))
        row["S_risk_mean"] = float(srisk.mean()) if srisk.size else math.nan
        row["S_risk_std"] = float(srisk.std(ddof=0)) if srisk.size else math.nan
        row["S_risk_median"] = median
        row["S_risk_MAD"] = mad
        row["S_risk_max_over_median"] = safe_ratio(float(np.max(srisk)), median) if srisk.size else math.nan
        row["S_risk_p99_over_median"] = safe_ratio(float(np.percentile(srisk, 99)), median) if srisk.size else math.nan
        row["S_risk_p95_over_median"] = safe_ratio(float(np.percentile(srisk, 95)), median) if srisk.size else math.nan
        row["S_risk_max_minus_median"] = float(np.max(srisk) - median) if srisk.size else math.nan
        row["S_risk_p99_minus_median"] = float(np.percentile(srisk, 99) - median) if srisk.size else math.nan
        row["S_risk_p95_minus_median"] = float(np.percentile(srisk, 95) - median) if srisk.size else math.nan
        row["top1_S_risk_mass_fraction"] = safe_fraction(float(sorted_scores[:1].sum()), total_mass)
        row["top5_S_risk_mass_fraction"] = safe_fraction(float(sorted_scores[:5].sum()), total_mass)
        row["top10_S_risk_mass_fraction"] = safe_fraction(float(sorted_scores[:10].sum()), total_mass)
        row["shape_category"] = shape_category(row)
        rows.append(row)
    return rows


def raw_tail_rows(grouped: dict[tuple[int, int, int], list[Observation]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hkl in sorted(grouped):
        scores = np.asarray([obs.S_risk for obs in grouped[hkl]], dtype=np.float64)
        mean = float(scores.mean()) if scores.size else math.nan
        std = float(scores.std(ddof=0)) if scores.size else math.nan
        z = (scores - mean) / std if math.isfinite(std) and std > 0.0 else np.full(scores.shape, np.nan)
        max_z = float(np.nanmax(z)) if np.any(np.isfinite(z)) else math.nan
        z2 = int(np.count_nonzero(z > 2.0)) if np.any(np.isfinite(z)) else 0
        z3 = int(np.count_nonzero(z > 3.0)) if np.any(np.isfinite(z)) else 0
        z4 = int(np.count_nonzero(z > 4.0)) if np.any(np.isfinite(z)) else 0
        for percentile in TAIL_PERCENTILES:
            threshold = float(np.percentile(scores, percentile)) if scores.size else math.nan
            count_strict = int(np.count_nonzero(scores > threshold)) if scores.size else 0
            count_ge = int(np.count_nonzero(scores >= threshold)) if scores.size else 0
            rows.append(
                {
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "n_observations": int(scores.size),
                    "threshold_type": f"raw_S_risk_p{str(percentile).replace('.', 'p')}",
                    "threshold_value": threshold,
                    "count_above_strict": count_strict,
                    "fraction_above_strict": count_strict / max(int(scores.size), 1),
                    "count_at_or_above": count_ge,
                    "fraction_at_or_above": count_ge / max(int(scores.size), 1),
                    "max_z_raw": max_z,
                    "count_z_raw_gt_2": z2,
                    "count_z_raw_gt_3": z3,
                    "count_z_raw_gt_4": z4,
                    "fraction_z_raw_gt_2": z2 / max(int(scores.size), 1),
                    "fraction_z_raw_gt_3": z3 / max(int(scores.size), 1),
                    "fraction_z_raw_gt_4": z4 / max(int(scores.size), 1),
                }
            )
    return rows


def secondary_summary(grouped: dict[tuple[int, int, int], list[Observation]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hkl in sorted(grouped):
        scores = np.asarray([obs.S_risk for obs in grouped[hkl]], dtype=np.float64)
        log_scores = np.log10(scores + EPSILON_FOR_LOG10)
        sqrt_scores = np.sqrt(np.maximum(scores, 0.0))
        rank_percentiles = np.asarray([rank_percentile(rank, len(scores)) for rank in range(1, len(scores) + 1)], dtype=np.float64)
        row: dict[str, Any] = {
            "h": hkl[0],
            "k": hkl[1],
            "l": hkl[2],
            "n_observations": int(scores.size),
            "epsilon_for_log10": EPSILON_FOR_LOG10,
        }
        row.update(prefixed_qstats("log10_S_risk_plus_epsilon", log_scores))
        row.update(prefixed_qstats("sqrt_S_risk", sqrt_scores))
        row["rank_percentile_min"] = float(np.min(rank_percentiles)) if rank_percentiles.size else math.nan
        row["rank_percentile_median"] = float(np.median(rank_percentiles)) if rank_percentiles.size else math.nan
        row["rank_percentile_max"] = float(np.max(rank_percentiles)) if rank_percentiles.size else math.nan
        row["secondary_note"] = "secondary transformed view only; raw S_risk summaries should be interpreted first"
        rows.append(row)
    return rows


def load_cache_parameters(cache: Path) -> dict[str, Any]:
    meta = cache.parent / "v7_geometric_risk_metadata.json"
    if not meta.is_file():
        return {}
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload.get("parameters", {})


def git_info() -> dict[str, Any]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": rev.returncode == 0,
        "commit": rev.stdout.strip() if rev.returncode == 0 else None,
        "status_short": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def plot_outputs(out_dir: Path, grouped: dict[tuple[int, int, int], list[Observation]], summary_rows: list[dict[str, Any]], label: str, logger: RunLogger) -> None:
    plt, PdfPages = import_matplotlib()
    logger.log("writing top-100 HKL plots")
    hkls = sorted(grouped)
    n_pages = math.ceil(len(hkls) / 12)

    with PdfPages(out_dir / "top100_hkl_raw_Srisk_histograms.pdf") as pdf:
        for page in range(n_pages):
            page_hkls = hkls[page * 12 : (page + 1) * 12]
            fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
            for ax, hkl in zip(axes.flat, page_hkls):
                scores = np.asarray([obs.S_risk for obs in grouped[hkl]], dtype=np.float64)
                ax.hist(scores, bins=min(40, max(10, len(scores) // 3)), color="#356d9a", alpha=0.9)
                ax.set_title(f"{hkl} n={len(scores)}", fontsize=9)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
            for ax in axes.flat[len(page_hkls) :]:
                ax.axis("off")
            fig.suptitle(f"{label}: raw S_risk histograms", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(out_dir / "top100_hkl_raw_Srisk_sorted_curves.pdf") as pdf:
        for page in range(n_pages):
            page_hkls = hkls[page * 12 : (page + 1) * 12]
            fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
            for ax, hkl in zip(axes.flat, page_hkls):
                scores = np.sort(np.asarray([obs.S_risk for obs in grouped[hkl]], dtype=np.float64))[::-1]
                ax.plot(np.arange(1, len(scores) + 1), scores, linewidth=1.2)
                ax.set_title(f"{hkl} n={len(scores)}", fontsize=9)
                ax.set_xlabel("rank desc")
                ax.set_ylabel("S_risk")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            for ax in axes.flat[len(page_hkls) :]:
                ax.axis("off")
            fig.suptitle(f"{label}: sorted raw S_risk curves", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(out_dir / "top100_hkl_log10_Srisk_histograms.pdf") as pdf:
        for page in range(n_pages):
            page_hkls = hkls[page * 12 : (page + 1) * 12]
            fig, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
            for ax, hkl in zip(axes.flat, page_hkls):
                scores = np.asarray([obs.S_risk for obs in grouped[hkl]], dtype=np.float64)
                ax.hist(np.log10(scores + EPSILON_FOR_LOG10), bins=min(40, max(10, len(scores) // 3)), color="#665191", alpha=0.9)
                ax.set_title(f"{hkl} n={len(scores)}", fontsize=9)
                ax.set_xlabel("log10(S_risk + eps)")
            for ax in axes.flat[len(page_hkls) :]:
                ax.axis("off")
            fig.suptitle(f"{label}: secondary log10 S_risk histograms", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

    fig, axes = plt.subplots(5, 5, figsize=(14, 12), constrained_layout=True)
    top25 = sorted(summary_rows, key=lambda row: float(row["S_risk_max"]), reverse=True)[:25]
    for ax, row in zip(axes.flat, top25):
        hkl = (int(row["h"]), int(row["k"]), int(row["l"]))
        scores = np.sort(np.asarray([obs.S_risk for obs in grouped[hkl]], dtype=np.float64))[::-1]
        ax.plot(np.arange(1, len(scores) + 1), scores, linewidth=1.0)
        ax.set_title(f"{hkl}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for ax in axes.flat[len(top25) :]:
        ax.axis("off")
    fig.suptitle(f"{label}: top raw S_risk sorted curves", fontsize=13)
    fig.savefig(out_dir / "top100_hkl_raw_Srisk_small_multiples.png", dpi=180)
    plt.close(fig)

    nobs = np.asarray([row["n_observations"] for row in summary_rows], dtype=np.float64)
    ratios = np.asarray([row["S_risk_p99_over_median"] for row in summary_rows], dtype=np.float64)
    medians = np.asarray([row["S_risk_median"] for row in summary_rows], dtype=np.float64)
    maxima = np.asarray([row["S_risk_max"] for row in summary_rows], dtype=np.float64)
    top1 = np.asarray([row["top1_S_risk_mass_fraction"] for row in summary_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(nobs, ratios, s=28, alpha=0.85)
    ax.set_xlabel("observations per signed HKL")
    ax.set_ylabel("p99 / median raw S_risk")
    ax.set_title(f"{label}: nobs vs raw tail strength")
    ax.set_yscale("log")
    fig.savefig(out_dir / "top100_hkl_nobs_vs_raw_tail_strength.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(medians, maxima, s=28, alpha=0.85)
    ax.set_xlabel("median raw S_risk")
    ax.set_ylabel("max raw S_risk")
    ax.set_title(f"{label}: median vs max raw S_risk")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig(out_dir / "top100_hkl_median_vs_max_raw_Srisk.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    order = np.argsort(top1)[::-1]
    ax.bar(np.arange(len(top1)), top1[order], color="#a05a2c")
    ax.set_xlabel("selected HKLs sorted by top-one mass fraction")
    ax.set_ylabel("fraction of within-HKL S_risk mass")
    ax.set_title(f"{label}: top observation mass fraction")
    fig.savefig(out_dir / "top100_hkl_top_observation_mass_fraction.png", dpi=180)
    plt.close(fig)


def previous_audit_comparison(previous_dir: Path | None) -> dict[str, Any]:
    if previous_dir is None or not previous_dir.is_dir():
        return {"available": False}
    meta_path = previous_dir / "top100_hkl_raw_distribution_metadata.json"
    summary_path = previous_dir / "top100_hkl_raw_distribution_summary.tsv"
    selected_path = previous_dir / "selected_top100_hkls.tsv"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    summary = read_tsv(summary_path)
    selected = read_tsv(selected_path)
    ratios = np.asarray([float(row.get("S_risk_max_over_median", "nan")) for row in summary], dtype=float)
    medians = np.asarray([float(row.get("S_risk_median", "nan")) for row in summary], dtype=float)
    top1 = np.asarray([float(row.get("top1_S_risk_mass_fraction", "nan")) for row in summary], dtype=float)
    l0 = sum(1 for row in selected if int(float(row.get("l", "nan"))) == 0) if selected else None
    return {
        "available": True,
        "path": str(previous_dir),
        "distinct_top100_signed_hkls": meta.get("distinct_top100_signed_hkls", len(selected)),
        "selected_hkl_observation_rows": meta.get("selected_hkl_observation_rows"),
        "median_max_over_median": float(np.nanmedian(ratios)) if ratios.size else None,
        "p90_max_over_median": float(np.nanpercentile(ratios, 90)) if ratios.size else None,
        "median_selected_hkl_median_S_risk": float(np.nanmedian(medians)) if medians.size else None,
        "median_top1_mass_fraction": float(np.nanmedian(top1)) if top1.size else None,
        "l0_selected_hkl_count": l0,
        "selected_hkl_count": len(selected) if selected else None,
    }


def compact_hkl_list(rows: list[dict[str, Any]], key: str, n: int = 8) -> str:
    selected = sorted(rows, key=lambda row: float(row[key]), reverse=True)[:n]
    return "; ".join(
        f"({int(row['h'])},{int(row['k'])},{int(row['l'])}) {key}={float(row[key]):.4g} n={int(row['n_observations'])}"
        for row in selected
    )


def write_notes(
    path: Path,
    args: argparse.Namespace,
    selected_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    top10_rows: list[dict[str, Any]],
    previous: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    l0_count = sum(1 for row in selected_rows if int(row["l"]) == 0)
    l_values = sorted({int(row["l"]) for row in selected_rows})
    total_obs = int(sum(int(row["n_observations"]) for row in summary_rows))
    ratios = np.asarray([float(row["S_risk_max_over_median"]) for row in summary_rows], dtype=np.float64)
    top1 = np.asarray([float(row["top1_S_risk_mass_fraction"]) for row in summary_rows], dtype=np.float64)
    categories: dict[str, int] = {}
    for row in summary_rows:
        categories[str(row["shape_category"])] = categories.get(str(row["shape_category"]), 0) + 1
    sorted_categories = "; ".join(f"{name}: {count}" for name, count in sorted(categories.items(), key=lambda item: (-item[1], item[0])))
    low20 = sum(1 for row in summary_rows if int(row["n_observations"]) < 20)
    low50 = sum(1 for row in summary_rows if int(row["n_observations"]) < 50)
    max_z_examples = sorted(raw_tail_rows_for_notes(summary_rows, path.parent / "top100_hkl_raw_tail_counts.tsv"), key=lambda row: float(row["max_z_raw"]), reverse=True)[:8]
    max_z_text = "; ".join(
        f"({int(row['h'])},{int(row['k'])},{int(row['l'])}) max_z={float(row['max_z_raw']):.3g} z>3={int(row['count_z_raw_gt_3'])}"
        for row in max_z_examples
    )
    if previous.get("available"):
        prev_text = (
            f"Previous V7 had {previous.get('distinct_top100_signed_hkls')} distinct top-100 HKLs and "
            f"{previous.get('selected_hkl_observation_rows')} extracted rows; median max/median ratio "
            f"{float(previous.get('median_max_over_median') or math.nan):.4g}. V7b has {len(selected_rows)} distinct HKLs, "
            f"{total_obs} extracted rows, and median max/median ratio {float(np.nanmedian(ratios)):.4g}."
        )
    else:
        prev_text = "Previous V7 audit summary was not available for comparison."

    lines = [
        "# Top-100 HKL Raw V7b Risk Distribution Audit",
        "",
        f"- Label: `{args.label}`",
        f"- Cache: `{args.cache}`",
        f"- Output: `{args.output_dir}`",
        f"- Cache parameters: `{metadata.get('cache_parameters', {})}`",
        f"- Global top-{args.top_n} raw `S_risk` rows contained `{len(selected_rows)}` distinct signed HKLs.",
        f"- Extracted all observations for those HKLs: `{total_obs}` rows total.",
        "- Raw score was inspected first. Log and square-root summaries are included only as secondary visual/diagnostic views.",
        "",
        "## What the Raw Distributions Look Like",
        f"- Max/median raw `S_risk` ratios across selected HKLs: median `{float(np.nanmedian(ratios)):.4g}`, p90 `{float(np.nanpercentile(ratios, 90)):.4g}`, max `{float(np.nanmax(ratios)):.4g}`.",
        f"- Top-one within-HKL mass fractions: median `{float(np.nanmedian(top1)):.4g}`, max `{float(np.nanmax(top1)):.4g}`.",
        f"- Shape categories from raw summaries: {sorted_categories}.",
        "- Many selected HKLs are chosen because their top observations sit in a narrow high-risk tail relative to the full within-HKL distribution.",
        "",
        "## HKL Baseline vs Within-HKL Tails",
        f"- Highest median raw `S_risk` HKLs: {compact_hkl_list(summary_rows, 'S_risk_median')}.",
        f"- Strongest max/median raw tails: {compact_hkl_list(summary_rows, 'S_risk_max_over_median')}.",
        f"- Largest top-one mass fractions: {compact_hkl_list(summary_rows, 'top1_S_risk_mass_fraction')}.",
        "- Conclusion: the global top rows emphasize a mixture of high-baseline HKLs and sharp within-HKL tails; inspect the sorted raw curves before translating this to a filtering rule.",
        "",
        "## Raw Tail and z-score Diagnostics",
        "- Raw percentile thresholds are directly interpretable and give stable within-HKL tail counts.",
        "- Ordinary raw z-scores are diagnostic only because right-skewed raw distributions pull the mean and standard deviation.",
        f"- Largest ordinary raw z examples: {max_z_text}.",
        "",
        "## Secondary Transformations",
        "- `log10(S_risk + 1e-300)` and `sqrt(S_risk)` are included after raw inspection only. They help visualize tails spanning orders of magnitude but should not be treated as the primary definition without a separate filtering rationale.",
        "",
        "## Too Few Observations",
        f"- HKLs with fewer than 20 observations among the selected top-100 HKLs: `{low20}`.",
        f"- HKLs with fewer than 50 observations: `{low50}`.",
        "",
        "## Answers",
        f"1. The V7b global top-{args.top_n} contains `{len(selected_rows)}` distinct signed HKLs.",
        f"2. The extraction across those HKLs contains `{total_obs}` observations.",
        f"3. Raw `S_risk` distributions are mostly {dominant_shape(categories)}.",
        f"4. Top-100 HKLs have `{l0_count}`/`{len(selected_rows)}` signed HKLs with `l=0`; represented signed `l` values are `{l_values}`.",
        f"5. Compared with the previous V7 audit: {prev_text}",
        "6. The sorted raw curves are the main check for clean high-risk tails; use them together with the raw histograms.",
        "7. Inspect `top100_hkl_raw_Srisk_sorted_curves.pdf`, `top100_hkl_raw_Srisk_histograms.pdf`, and `top100_hkl_median_vs_max_raw_Srisk.png` before choosing V7b cutoffs.",
        "8. This audit does not recommend final cutoffs.",
        "",
        "## Files",
    ]
    for item in sorted(path.parent.iterdir()):
        if item.is_file() and item.name != path.name:
            lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def raw_tail_rows_for_notes(summary_rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    rows = read_tsv(path)
    by_hkl: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["h"]), int(row["k"]), int(row["l"]))
        if key not in by_hkl:
            by_hkl[key] = {
                "h": key[0],
                "k": key[1],
                "l": key[2],
                "max_z_raw": float(row["max_z_raw"]),
                "count_z_raw_gt_3": int(row["count_z_raw_gt_3"]),
            }
    if by_hkl:
        return list(by_hkl.values())
    return [
        {"h": row["h"], "k": row["k"], "l": row["l"], "max_z_raw": math.nan, "count_z_raw_gt_3": 0}
        for row in summary_rows
    ]


def dominant_shape(categories: dict[str, int]) -> str:
    if not categories:
        return "unclassified"
    name, count = sorted(categories.items(), key=lambda item: (-item[1], item[0]))[0]
    if count == 1:
        return name
    return f"{name} ({count} HKLs)"


def summary_fieldnames() -> list[str]:
    return [
        "h",
        "k",
        "l",
        "n_observations",
        "top100_count",
        *[f"Eg_{name}" for name in PERCENTILE_NAMES],
        *[f"R_env_{name}" for name in PERCENTILE_NAMES],
        *[f"S_risk_{name}" for name in PERCENTILE_NAMES],
        "S_risk_mean",
        "S_risk_std",
        "S_risk_median",
        "S_risk_MAD",
        "S_risk_max_over_median",
        "S_risk_p99_over_median",
        "S_risk_p95_over_median",
        "S_risk_max_minus_median",
        "S_risk_p99_minus_median",
        "S_risk_p95_minus_median",
        "top1_S_risk_mass_fraction",
        "top5_S_risk_mass_fraction",
        "top10_S_risk_mass_fraction",
        "shape_category",
    ]


def all_observation_fieldnames(columns: list[str]) -> list[str]:
    return [
        "source_index",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "exact_key_text",
        "source_order",
        "rank_within_hkl_desc",
        "within_hkl_rank_percentile",
        "E_g",
        "R_env",
        "S_risk",
        "log10_S_risk_plus_epsilon",
        "sqrt_S_risk",
        "z_raw_S_risk",
        *[column for column in OPTIONAL_COLUMNS if column in columns],
    ]


def secondary_fieldnames() -> list[str]:
    return [
        "h",
        "k",
        "l",
        "n_observations",
        "epsilon_for_log10",
        *[f"log10_S_risk_plus_epsilon_{name}" for name in PERCENTILE_NAMES],
        *[f"sqrt_S_risk_{name}" for name in PERCENTILE_NAMES],
        "rank_percentile_min",
        "rank_percentile_median",
        "rank_percentile_max",
        "secondary_note",
    ]


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    prepare_output_dir(args.output_dir, args.overwrite)
    logger = RunLogger(args.output_dir / "top100_hkl_raw_distribution_audit.log")
    try:
        logger.log("top-100 HKL raw risk distribution audit start")
        logger.log(f"cache={args.cache}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"label={args.label}")
        conn = sqlite_readonly(args.cache)
        columns = require_schema(conn)
        cache_rows = int(conn.execute("SELECT COUNT(*) FROM v7_score_cache").fetchone()[0])
        logger.log(f"cache rows={cache_rows:,}")
        top_rows = fetch_top_rows(conn, args.top_n)
        if len(top_rows) != args.top_n:
            raise SystemExit(f"Expected {args.top_n} top rows but found {len(top_rows)}")
        hkls, selected_rows = top_hkl_selection_rows(top_rows)
        top_counts = {(int(row["h"]), int(row["k"]), int(row["l"])): int(row["top100_count"]) for row in selected_rows}
        logger.log(f"global top {args.top_n} rows include {len(hkls):,} distinct signed HKLs")
        observations = fetch_observations_for_hkls(conn, hkls, columns)
        conn.close()
        logger.log(f"extracted {len(observations):,} total observations across selected HKLs")
        grouped = group_observations(observations)
        missing_hkls = sorted(set(hkls) - set(grouped))
        if missing_hkls:
            raise SystemExit(f"Missing selected HKLs after extraction: {missing_hkls[:10]}")

        summary_rows = distribution_summary(grouped, top_counts)
        all_rows, top10_rows = annotated_observations(grouped)
        tail_rows = raw_tail_rows(grouped)
        secondary_rows = secondary_summary(grouped)

        write_tsv(args.output_dir / "selected_top100_hkls.tsv", selected_rows, ["h", "k", "l", "top100_count", "max_top100_S_risk", "selection_note"])
        write_tsv(args.output_dir / "top100_hkl_raw_distribution_summary.tsv", summary_rows, summary_fieldnames())
        write_tsv(args.output_dir / "top100_hkl_all_observations_with_raw_scores.tsv", all_rows, all_observation_fieldnames(columns))
        write_tsv(
            args.output_dir / "top100_hkl_raw_tail_counts.tsv",
            tail_rows,
            [
                "h",
                "k",
                "l",
                "n_observations",
                "threshold_type",
                "threshold_value",
                "count_above_strict",
                "fraction_above_strict",
                "count_at_or_above",
                "fraction_at_or_above",
                "max_z_raw",
                "count_z_raw_gt_2",
                "count_z_raw_gt_3",
                "count_z_raw_gt_4",
                "fraction_z_raw_gt_2",
                "fraction_z_raw_gt_3",
                "fraction_z_raw_gt_4",
            ],
        )
        write_tsv(args.output_dir / "top100_hkl_top10_observations_per_hkl.tsv", top10_rows, all_observation_fieldnames(columns))
        write_tsv(args.output_dir / "top100_hkl_secondary_transformed_summary.tsv", secondary_rows, secondary_fieldnames())
        plot_outputs(args.output_dir, grouped, summary_rows, args.label, logger)

        previous = previous_audit_comparison(args.previous_audit_dir)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "label": args.label,
            "cache": str(args.cache),
            "output_dir": str(args.output_dir),
            "previous_audit_dir": str(args.previous_audit_dir) if args.previous_audit_dir else None,
            "previous_audit_comparison": previous,
            "cache_rows": cache_rows,
            "sqlite_schema": columns,
            "cache_parameters": load_cache_parameters(args.cache),
            "top100_rows": int(args.top_n),
            "distinct_top100_signed_hkls": int(len(hkls)),
            "selected_hkl_observation_rows": int(len(observations)),
            "raw_first": True,
            "log_transform_secondary_only": True,
            "epsilon_for_log10": EPSILON_FOR_LOG10,
            "elapsed_seconds": float(time.monotonic() - started),
            "git": git_info(),
        }
        write_notes(args.output_dir / "top100_hkl_raw_distribution_notes.md", args, selected_rows, summary_rows, top10_rows, previous, metadata)
        metadata["elapsed_seconds"] = float(time.monotonic() - started)
        write_json(args.output_dir / "top100_hkl_raw_distribution_metadata.json", metadata)
        logger.log(f"top-100 HKL raw risk distribution audit complete in {time.monotonic() - started:.1f} s")
        return 0
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
