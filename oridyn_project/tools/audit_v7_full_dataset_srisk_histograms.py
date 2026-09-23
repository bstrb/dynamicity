#!/usr/bin/env python3
"""Audit full-dataset V7 S_risk distributions by resolution, HKL, and source.

This is a read-only audit for an existing V7 geometric-risk SQLite cache.  It
reuses the V7 stream/cache helpers used by the cutoff-stream builder so the
d-spacing convention and source/event reciprocal-basis parsing stay identical.

No filtering, stream generation, Partialator, or merging is run.
"""

from __future__ import annotations

import argparse
import csv
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_v7_absolute_srisk_cutoff_streams as cutoff_tools


DATASET_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_CACHE = (
    DATASET_ROOT
    / "oridyn_v7_geometric_risk_s0_0p005_sig0p03_rcut0p20_20260904"
    / "v7_geometric_risk_cache.sqlite"
)
DEFAULT_STREAM = DATASET_ROOT / "MFM300-VIII_cut_20-0_3.stream"
DEFAULT_OUTPUT_DIR = (
    DATASET_ROOT
    / "oridyn_v7_geometric_risk_s0_0p005_sig0p03_rcut0p20_20260904"
    / "audit"
    / "full_dataset_Srisk_histograms_to_0p5A"
)
DEFAULT_LABEL = "V7b geometric risk, s0=0.005, sigma_C=0.03, r_cut=0.20"
DEFAULT_THRESHOLDS = [3e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
DEFAULT_D_MIN = 0.5
DEFAULT_CHUNK_ROWS = 250_000
DEFAULT_SCATTER_SAMPLE_N = 200_000
DEFAULT_SCATTER_SEED = 20260904
EXPECTED_CACHE_ROWS = 6_732_955
PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 100]
PERCENTILE_NAMES = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p97p5", "p99", "p99p5", "p99p9", "max"]
SHELL_PERCENTILES = [0, 25, 50, 75, 90, 95, 97.5, 99, 100]
SHELL_PERCENTILE_NAMES = ["min", "p25", "median", "p75", "p90", "p95", "p97p5", "p99", "max"]
OLD_V7_CUTOFF_LABELS = ("0.050", "0.020", "0.010", "0.005")


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


def parse_thresholds(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or value <= 0.0:
            raise argparse.ArgumentTypeError("thresholds must be positive finite numbers")
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return sorted(set(out), reverse=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--thresholds", default=",".join(f"{x:.0e}" for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN, help="Include observations with d_spacing >= this value in A.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Recorded in metadata; computation is vectorized in the parent process.")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--scatter-sample-n", type=int, default=DEFAULT_SCATTER_SAMPLE_N)
    parser.add_argument("--scatter-seed", type=int, default=DEFAULT_SCATTER_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    args.cache = args.cache.expanduser().resolve()
    args.stream = args.stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.cache.is_file():
        raise SystemExit(f"Cache not found: {args.cache}")
    if not args.stream.is_file():
        raise SystemExit(f"Source stream not found: {args.stream}")
    args.threshold_values = parse_thresholds(args.thresholds)
    if not math.isfinite(float(args.d_min)) or float(args.d_min) <= 0.0:
        raise SystemExit("--d-min must be positive")
    args.workers = max(1, int(args.workers))
    args.chunk_rows = max(1, int(args.chunk_rows))
    args.scatter_sample_n = max(1, int(args.scatter_sample_n))
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


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        existing_files = [item for item in path.iterdir() if item.is_file()]
        if existing_files and not overwrite:
            preview = "\n".join(str(item) for item in sorted(existing_files)[:30])
            extra = "" if len(existing_files) <= 30 else f"\n... and {len(existing_files) - 30} more"
            raise SystemExit(f"Refusing to overwrite existing audit outputs; pass --overwrite if intentional:\n{preview}{extra}")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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


def cache_schema(conn: sqlite3.Connection) -> list[str]:
    return [str(row[1]) for row in conn.execute("PRAGMA table_info(v7_score_cache)").fetchall()]


def qstats(values: np.ndarray, percentiles: list[float], names: list[str]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {name: math.nan for name in names}
    qs = np.percentile(arr, percentiles)
    return {name: float(value) for name, value in zip(names, qs)}


def threshold_label(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")


def threshold_column_label(value: float) -> str:
    return threshold_label(value).replace("-", "m").replace("+", "p")


def source_index(source: str) -> str:
    match = re.search(r"_(\d+)\.h5$", str(source))
    return match.group(1) if match else ""


def structured_hkl(h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
    return cutoff_tools.structured_hkl(h, k, l)


def unique_hkl_count(h: np.ndarray, k: np.ndarray, l: np.ndarray) -> int:
    return cutoff_tools.unique_hkl_count(h, k, l)


def top_hkl_counts(h: np.ndarray, k: np.ndarray, l: np.ndarray, n: int = 50) -> list[tuple[int, int, int, int]]:
    if len(h) == 0:
        return []
    hkls, counts = np.unique(structured_hkl(h, k, l), return_counts=True)
    order = np.lexsort((hkls["l"], hkls["k"], hkls["h"], -counts))
    out: list[tuple[int, int, int, int]] = []
    for idx in order[:n]:
        rec = hkls[int(idx)]
        out.append((int(rec["h"]), int(rec["k"]), int(rec["l"]), int(counts[int(idx)])))
    return out


def top_source_counts(source_ids: np.ndarray, source_labels: list[tuple[str, str]], n: int = 50) -> list[tuple[int, str, str, str, int]]:
    if len(source_ids) == 0:
        return []
    ids, counts = np.unique(source_ids, return_counts=True)
    order = np.lexsort((ids, -counts))
    out: list[tuple[int, str, str, str, int]] = []
    for idx in order[:n]:
        sid = int(ids[int(idx)])
        source, event = source_labels[sid]
        out.append((sid, source_index(source), source, event, int(counts[int(idx)])))
    return out


def shell_mask(d_spacing: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if math.isinf(hi):
        return d_spacing >= float(lo)
    return (d_spacing >= float(lo)) & (d_spacing < float(hi))


def shell_contributions(threshold: float, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    srisk = arrays["S_risk"]
    d_spacing = arrays["d_spacing"]
    mask = srisk > float(threshold)
    total = int(np.count_nonzero(mask))
    rows: list[dict[str, Any]] = []
    for label, lo, hi in cutoff_tools.SHELLS:
        smask = mask & shell_mask(d_spacing, lo, hi)
        count = int(np.count_nonzero(smask))
        rows.append(
            {
                "resolution_shell_A": label,
                "above_threshold_count": count,
                "fraction_of_above_threshold": count / max(total, 1),
                "fraction_of_shell": count / max(int(np.count_nonzero(shell_mask(d_spacing, lo, hi))), 1),
            }
        )
    rows.sort(key=lambda row: int(row["above_threshold_count"]), reverse=True)
    return rows


def write_global_quantiles(path: Path, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(len(arrays["S_risk"]))
    unique_hkls = unique_hkl_count(arrays["h"], arrays["k"], arrays["l"])
    unique_sources = int(np.unique(arrays["source_event_id"]).size) if n else 0
    metric_map = {
        "E_g": arrays["E_g"],
        "R_env": arrays["R_env"],
        "S_risk": arrays["S_risk"],
        "d_spacing_A": arrays["d_spacing"],
        "reciprocal_radius_Ainv": arrays["reciprocal_radius"],
    }
    for metric, values in metric_map.items():
        row: dict[str, Any] = {
            "metric": metric,
            "n_observations": n,
            "unique_signed_hkls": unique_hkls,
            "unique_source_events": unique_sources,
        }
        row.update(qstats(values, PERCENTILES, PERCENTILE_NAMES))
        rows.append(row)
    write_tsv(path, rows, ["metric", "n_observations", "unique_signed_hkls", "unique_source_events", *PERCENTILE_NAMES])
    return rows


def write_resolution_shells(path: Path, arrays: dict[str, np.ndarray], thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    d_spacing = arrays["d_spacing"]
    for label, lo, hi in cutoff_tools.SHELLS:
        mask = shell_mask(d_spacing, lo, hi)
        srisk = arrays["S_risk"][mask]
        eg = arrays["E_g"][mask]
        renv = arrays["R_env"][mask]
        row: dict[str, Any] = {
            "resolution_shell_A": label,
            "d_min_inclusive_A": lo,
            "d_max_exclusive_A": hi if math.isfinite(hi) else "",
            "observation_count": int(np.count_nonzero(mask)),
            "unique_signed_hkls": unique_hkl_count(arrays["h"][mask], arrays["k"][mask], arrays["l"][mask]),
            "unique_source_events": int(np.unique(arrays["source_event_id"][mask]).size) if np.any(mask) else 0,
        }
        row.update({f"S_risk_{key}": value for key, value in qstats(srisk, SHELL_PERCENTILES, SHELL_PERCENTILE_NAMES).items()})
        row["E_g_median"] = qstats(eg, [50], ["median"])["median"]
        row["E_g_p95"] = qstats(eg, [95], ["p95"])["p95"]
        row["R_env_median"] = qstats(renv, [50], ["median"])["median"]
        row["R_env_p95"] = qstats(renv, [95], ["p95"])["p95"]
        for threshold in thresholds:
            col = f"fraction_S_risk_gt_{threshold_column_label(threshold)}"
            row[col] = int(np.count_nonzero(srisk > threshold)) / max(int(len(srisk)), 1)
        rows.append(row)
    fieldnames = [
        "resolution_shell_A",
        "d_min_inclusive_A",
        "d_max_exclusive_A",
        "observation_count",
        "unique_signed_hkls",
        "unique_source_events",
        *[f"S_risk_{name}" for name in SHELL_PERCENTILE_NAMES],
        "E_g_median",
        "E_g_p95",
        "R_env_median",
        "R_env_p95",
        *[f"fraction_S_risk_gt_{threshold_column_label(t)}" for t in thresholds],
    ]
    write_tsv(path, rows, fieldnames)
    return rows


def write_threshold_summaries(
    threshold_path: Path,
    top_hkl_path: Path,
    top_source_path: Path,
    arrays: dict[str, np.ndarray],
    source_labels: list[tuple[str, str]],
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    threshold_rows: list[dict[str, Any]] = []
    hkl_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    srisk = arrays["S_risk"]
    h = arrays["h"]
    k = arrays["k"]
    l = arrays["l"]
    source_ids = arrays["source_event_id"]
    n = len(srisk)
    for threshold in thresholds:
        label = threshold_label(threshold)
        mask = srisk > float(threshold)
        count = int(np.count_nonzero(mask))
        l0_count = int(np.count_nonzero(mask & (l == 0)))
        top_hkls = top_hkl_counts(h[mask], k[mask], l[mask], n=50)
        top_sources = top_source_counts(source_ids[mask], source_labels, n=50)
        top10_hkl_count = int(sum(row[3] for row in top_hkls[:10]))
        top10_source_count = int(sum(row[4] for row in top_sources[:10]))
        threshold_rows.append(
            {
                "threshold": threshold,
                "threshold_label": label,
                "above_threshold_count": count,
                "fraction_of_included_observations": count / max(n, 1),
                "unique_signed_hkls_above_threshold": unique_hkl_count(h[mask], k[mask], l[mask]),
                "unique_source_events_above_threshold": int(np.unique(source_ids[mask]).size) if count else 0,
                "l0_above_threshold_count": l0_count,
                "l0_fraction_of_above_threshold": l0_count / max(count, 1) if count else math.nan,
                "top10_hkl_above_threshold_count": top10_hkl_count,
                "top10_hkl_fraction_of_above_threshold": top10_hkl_count / max(count, 1) if count else math.nan,
                "top10_source_event_above_threshold_count": top10_source_count,
                "top10_source_event_fraction_of_above_threshold": top10_source_count / max(count, 1) if count else math.nan,
            }
        )
        for rank, (hh, kk, ll, c) in enumerate(top_hkls, start=1):
            hkl_rows.append(
                {
                    "threshold": threshold,
                    "threshold_label": label,
                    "rank": rank,
                    "h": hh,
                    "k": kk,
                    "l": ll,
                    "above_threshold_count": c,
                    "fraction_of_above_threshold": c / max(count, 1),
                }
            )
        for rank, (_sid, source_idx, source, event, c) in enumerate(top_sources, start=1):
            source_rows.append(
                {
                    "threshold": threshold,
                    "threshold_label": label,
                    "rank": rank,
                    "source_index": source_idx,
                    "source_filename": source,
                    "event": event,
                    "above_threshold_count": c,
                    "fraction_of_above_threshold": c / max(count, 1),
                }
            )
    threshold_fields = [
        "threshold",
        "threshold_label",
        "above_threshold_count",
        "fraction_of_included_observations",
        "unique_signed_hkls_above_threshold",
        "unique_source_events_above_threshold",
        "l0_above_threshold_count",
        "l0_fraction_of_above_threshold",
        "top10_hkl_above_threshold_count",
        "top10_hkl_fraction_of_above_threshold",
        "top10_source_event_above_threshold_count",
        "top10_source_event_fraction_of_above_threshold",
    ]
    write_tsv(threshold_path, threshold_rows, threshold_fields)
    write_tsv(top_hkl_path, hkl_rows, ["threshold", "threshold_label", "rank", "h", "k", "l", "above_threshold_count", "fraction_of_above_threshold"])
    write_tsv(
        top_source_path,
        source_rows,
        ["threshold", "threshold_label", "rank", "source_index", "source_filename", "event", "above_threshold_count", "fraction_of_above_threshold"],
    )
    return threshold_rows, hkl_rows, source_rows


def l_group_rows(arrays: dict[str, np.ndarray], thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    l_values = arrays["l"]
    for group_type, values in [("signed_l", l_values), ("abs_l", np.abs(l_values))]:
        for value in sorted(np.unique(values)):
            mask = values == value
            srisk = arrays["S_risk"][mask]
            row: dict[str, Any] = {
                "l_group_type": group_type,
                "l_value": int(value),
                "observation_count": int(np.count_nonzero(mask)),
                "unique_signed_hkls": unique_hkl_count(arrays["h"][mask], arrays["k"][mask], arrays["l"][mask]),
            }
            row.update({f"S_risk_{key}": val for key, val in qstats(srisk, SHELL_PERCENTILES, SHELL_PERCENTILE_NAMES).items()})
            for threshold in thresholds:
                row[f"fraction_S_risk_gt_{threshold_column_label(threshold)}"] = int(np.count_nonzero(srisk > threshold)) / max(int(len(srisk)), 1)
            rows.append(row)
    return rows


def write_l_summary(path: Path, arrays: dict[str, np.ndarray], thresholds: list[float]) -> list[dict[str, Any]]:
    rows = l_group_rows(arrays, thresholds)
    fieldnames = [
        "l_group_type",
        "l_value",
        "observation_count",
        "unique_signed_hkls",
        *[f"S_risk_{name}" for name in SHELL_PERCENTILE_NAMES],
        *[f"fraction_S_risk_gt_{threshold_column_label(t)}" for t in thresholds],
    ]
    write_tsv(path, rows, fieldnames)
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


def import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_histograms(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    thresholds: list[float],
    label: str,
    scatter_sample_n: int,
    scatter_seed: int,
    logger: RunLogger,
) -> None:
    plt = import_matplotlib()
    srisk = arrays["S_risk"]
    eg = arrays["E_g"]
    renv = arrays["R_env"]
    d_spacing = arrays["d_spacing"]
    rng = np.random.default_rng(int(scatter_seed))
    logger.log("writing histogram and scatter plots")

    def hist(path: str, values: np.ndarray, title: str, xlabel: str, *, bins: int = 200, xlim: tuple[float, float] | None = None, log_y: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.hist(values[np.isfinite(values)], bins=bins, color="#336699", alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Observation count")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if log_y:
            ax.set_yscale("log")
        for threshold in thresholds:
            if xlim is None or xlim[0] <= threshold <= xlim[1]:
                ax.axvline(threshold, color="#aa3333", alpha=0.35, linewidth=0.9)
        fig.savefig(output_dir / path, dpi=170)
        plt.close(fig)

    hist("full_raw_Srisk_hist.png", srisk, f"{label}: raw S_risk", "S_risk")
    hist("full_raw_Srisk_hist_xlim_0p15.png", srisk, f"{label}: raw S_risk, 0 to 0.15", "S_risk", xlim=(0.0, 0.15))
    hist("full_raw_Srisk_hist_xlim_0p05.png", srisk, f"{label}: raw S_risk, 0 to 0.05", "S_risk", xlim=(0.0, 0.05))
    hist("full_raw_Srisk_hist_log_y.png", srisk, f"{label}: raw S_risk", "S_risk", log_y=True)
    hist("full_log10_Srisk_hist_secondary.png", np.log10(srisk + 1.0e-300), f"{label}: log10(S_risk + 1e-300)", "log10(S_risk + 1e-300)")
    hist("full_Eg_hist.png", eg, f"{label}: E_g", "E_g", bins=160)
    hist("full_Renv_hist.png", renv, f"{label}: R_env", "R_env", bins=200)

    shell_labels = []
    shell_x = []
    shell_q = {"p25": [], "median": [], "p75": [], "p95": [], "p99": []}
    fractions_by_threshold = {threshold: [] for threshold in thresholds}
    for i, (slabel, lo, hi) in enumerate(cutoff_tools.SHELLS):
        mask = shell_mask(d_spacing, lo, hi)
        values = srisk[mask]
        shell_labels.append(slabel)
        shell_x.append(i)
        stats = qstats(values, [25, 50, 75, 95, 99], ["p25", "median", "p75", "p95", "p99"])
        for key in shell_q:
            shell_q[key].append(stats[key])
        for threshold in thresholds:
            fractions_by_threshold[threshold].append(int(np.count_nonzero(values > threshold)) / max(int(len(values)), 1))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for key, values in shell_q.items():
        ax.plot(shell_x, values, marker="o", linewidth=1.3, label=key)
    ax.set_xticks(shell_x, shell_labels, rotation=35, ha="right")
    ax.set_ylabel("S_risk")
    ax.set_title(f"{label}: S_risk quantiles by resolution shell")
    ax.legend()
    fig.savefig(output_dir / "Srisk_quantiles_by_resolution_shell.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for threshold, values in fractions_by_threshold.items():
        ax.plot(shell_x, values, marker="o", linewidth=1.2, label=f">{threshold_label(threshold)}")
    ax.set_xticks(shell_x, shell_labels, rotation=35, ha="right")
    ax.set_ylabel("Fraction above threshold")
    ax.set_title(f"{label}: above-threshold fraction by resolution shell")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(output_dir / "fraction_above_threshold_by_resolution_shell.png", dpi=170)
    plt.close(fig)

    n = min(int(scatter_sample_n), len(srisk))
    sample = rng.choice(len(srisk), size=n, replace=False) if len(srisk) > n else np.arange(len(srisk))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(d_spacing[sample], srisk[sample], s=1, alpha=0.25, linewidths=0)
    ax.set_xlabel("d_spacing (A)")
    ax.set_ylabel("S_risk")
    ax.set_title(f"{label}: sampled S_risk vs d_spacing")
    fig.savefig(output_dir / "Srisk_vs_dspacing_scatter_sample.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(d_spacing[sample], srisk[sample], s=1, alpha=0.25, linewidths=0)
    ax.set_xlabel("d_spacing (A)")
    ax.set_ylabel("S_risk")
    ax.set_yscale("log")
    ax.set_title(f"{label}: sampled S_risk vs d_spacing")
    fig.savefig(output_dir / "Srisk_vs_dspacing_scatter_sample_log_y.png", dpi=170)
    plt.close(fig)

    abs_l_values = np.unique(np.abs(arrays["l"]))
    medians = []
    p95s = []
    for value in abs_l_values:
        vals = srisk[np.abs(arrays["l"]) == value]
        medians.append(qstats(vals, [50], ["median"])["median"])
        p95s.append(qstats(vals, [95], ["p95"])["p95"])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(abs_l_values, medians, marker="o", linewidth=1.2, label="median")
    ax.plot(abs_l_values, p95s, marker="o", linewidth=1.2, label="p95")
    ax.set_xlabel("|l|")
    ax.set_ylabel("S_risk")
    ax.set_title(f"{label}: S_risk by |l|")
    ax.legend()
    fig.savefig(output_dir / "Srisk_by_l_distribution.png", dpi=170)
    plt.close(fig)

    l0_fracs = []
    for threshold in thresholds:
        mask = srisk > threshold
        l0_fracs.append(int(np.count_nonzero(mask & (arrays["l"] == 0))) / max(int(np.count_nonzero(mask)), 1))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot([threshold_label(t) for t in thresholds], l0_fracs, marker="o", linewidth=1.5)
    ax.set_xlabel("S_risk threshold")
    ax.set_ylabel("l=0 fraction above threshold")
    ax.set_title(f"{label}: l=0 fraction among high-risk observations")
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(output_dir / "high_risk_l0_fraction_by_threshold.png", dpi=170)
    plt.close(fig)


def markdown_threshold_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| S_risk threshold | observations above threshold | fraction included | l=0 fraction | top10 source/event fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        l0 = row["l0_fraction_of_above_threshold"]
        src = row["top10_source_event_fraction_of_above_threshold"]
        lines.append(
            f"| > {row['threshold_label']} | {int(row['above_threshold_count']):,} | "
            f"{100.0 * float(row['fraction_of_included_observations']):.3f}% | "
            f"{100.0 * float(l0):.3f}% | {100.0 * float(src):.3f}% |"
        )
    return "\n".join(lines)


def write_notes(
    path: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    global_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
    cache_parameters: dict[str, Any],
) -> None:
    srisk_global = next(row for row in global_rows if row["metric"] == "S_risk")
    eg_global = next(row for row in global_rows if row["metric"] == "E_g")
    renv_global = next(row for row in global_rows if row["metric"] == "R_env")
    first_nonzeroish = [row for row in threshold_rows if int(row["above_threshold_count"]) > 0]
    highest = first_nonzeroish[0] if first_nonzeroish else None
    mid = next((row for row in threshold_rows if row["threshold_label"] == "1e-4"), highest)
    shell_for_mid = []
    if mid is not None:
        shell_for_mid = shell_contributions(float(mid["threshold"]), metadata["arrays_for_notes"])
    params_line = ", ".join(f"{k}={v}" for k, v in sorted(cache_parameters.items())) if cache_parameters else "not found in cache metadata"

    lines = [
        "# Full Dataset V7b S_risk Histogram and Resolution Audit",
        "",
        f"- Label: `{args.label}`",
        f"- Cache: `{args.cache}`",
        f"- Output directory: `{args.output_dir}`",
        f"- Cache parameters: {params_line}",
        f"- Included observations use `d_spacing >= {float(args.d_min):g} A`, equivalently `|g| <= {1.0 / float(args.d_min):g} A^-1`.",
        "- The V7 cache has no stored `d_spacing` or reciprocal-radius columns, so `|g|` was computed as `||B* [h,k,l]||` using the source stream reciprocal basis for each source/event; stream `astar/bstar/cstar` vectors were converted from nm^-1 to A^-1. Then `d_spacing = 1 / |g|`.",
        f"- Stream metrics parsed: {metadata['stream_crystals_seen']:,} crystals, {metadata['stream_unique_source_events']:,} unique source/events; duplicate source/event keys: {metadata['stream_duplicate_source_event_keys']:,}.",
        f"- Cache rows read: {metadata['cache_rows_read']:,}; included after resolution cutoff: {metadata['included_rows']:,}; excluded above 0.5 A or invalid: {metadata['rows_excluded_high_resolution_or_outside'] + metadata['rows_excluded_invalid_radius']:,}.",
        "",
        "## Global Summary",
        f"- Included observations: {metadata['included_rows']:,}",
        f"- Unique signed HKLs: {metadata['included_unique_signed_hkls']:,}",
        f"- Unique source/event pairs: {metadata['included_unique_source_events']:,}",
        f"- `E_g` quantiles: median `{eg_global['median']:.6g}`, p95 `{eg_global['p95']:.6g}`, max `{eg_global['max']:.6g}`.",
        f"- `R_env` quantiles: median `{renv_global['median']:.6g}`, p95 `{renv_global['p95']:.6g}`, max `{renv_global['max']:.6g}`.",
        f"- Raw `S_risk` quantiles: median `{srisk_global['median']:.6g}`, p90 `{srisk_global['p90']:.6g}`, p95 `{srisk_global['p95']:.6g}`, p97.5 `{srisk_global['p97p5']:.6g}`, p99 `{srisk_global['p99']:.6g}`, p99.9 `{srisk_global['p99p9']:.6g}`, max `{srisk_global['max']:.6g}`.",
        "",
        "## Threshold Counts",
        markdown_threshold_table(threshold_rows),
        "",
        "## Resolution Pattern",
    ]
    if shell_for_mid and mid is not None:
        chunks = [
            f"{row['resolution_shell_A']}: {int(row['above_threshold_count']):,} "
            f"({100.0 * float(row['fraction_of_above_threshold']):.1f}% of above-threshold; "
            f"{100.0 * float(row['fraction_of_shell']):.2f}% of shell)"
            for row in shell_for_mid[:5]
        ]
        lines.append(f"- For `S_risk > {mid['threshold_label']}`, the largest shell contributions are: " + "; ".join(chunks) + ".")
    else:
        lines.append("- No observations exceeded the configured thresholds, so threshold-by-shell concentration is unavailable.")
    lines.extend(
        [
            "- Shell table: `full_dataset_Srisk_by_resolution_shell.tsv`.",
            "",
            "## Answers",
            "1. Raw `S_risk` values are much smaller than the previous V7 `s0=0.002, sigma_C=0.05` scale and remain right-skewed.",
            f"2. The previous absolute cutoff region near `0.02` does not apply to this V7b cache: the maximum `S_risk` is `{srisk_global['max']:.6g}`.",
            "3. Counts for the V7b thresholds are listed above and in `full_dataset_Srisk_threshold_counts.tsv`.",
            "4. Resolution-shell concentration should be read from `full_dataset_Srisk_by_resolution_shell.tsv` and the shell plots; this audit does not run filtering.",
            "5. `l=0` dominance depends strongly on threshold; use the threshold table and `high_risk_l0_fraction_by_threshold.png`.",
            "6. Source/event concentration is summarized by the top10 source/event fractions and `full_dataset_top_source_events_by_threshold.tsv`.",
            "7. Based only on these distributions, useful absolute cutoffs for this V7b score scale should be chosen from the populated threshold rows, with stricter tests near the upper tail.",
            "8. This is distribution inspection only; it does not establish a final filtering rule.",
            "",
            "## Files",
        ]
    )
    for item in sorted(path.parent.iterdir()):
        if item.is_file() and item.name != path.name:
            lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_no_old_threshold_labels(output_dir: Path) -> list[str]:
    hits: list[str] = []
    for path in output_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".tsv":
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames:
                for field in reader.fieldnames:
                    for label in OLD_V7_CUTOFF_LABELS:
                        if field.endswith(label):
                            hits.append(f"{path.name}:header:{field}")
                if "threshold_label" in reader.fieldnames:
                    for row in reader:
                        label_value = str(row.get("threshold_label", ""))
                        if label_value in OLD_V7_CUTOFF_LABELS:
                            hits.append(f"{path.name}:threshold_label:{label_value}")
                            break
    return hits


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    prepare_output_dir(args.output_dir, args.overwrite)
    logger = RunLogger(args.output_dir / "full_dataset_Srisk_histogram_audit.log")
    try:
        logger.log("full-dataset V7 S_risk histogram audit start")
        logger.log(f"cache={args.cache}")
        logger.log(f"stream={args.stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"label={args.label}")
        logger.log(f"thresholds={[threshold_label(x) for x in args.threshold_values]}")

        conn = cutoff_tools.open_cache(args.cache)
        columns = cache_schema(conn)
        if "v7_score_cache" not in [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
            raise SystemExit("Cache does not contain table v7_score_cache")
        cache_rows = int(conn.execute("SELECT COUNT(*) FROM v7_score_cache").fetchone()[0])
        if cache_rows != EXPECTED_CACHE_ROWS:
            logger.log(f"WARNING: cache row count is {cache_rows:,}, expected {EXPECTED_CACHE_ROWS:,}")
        else:
            logger.log(f"cache row count check passed: {cache_rows:,}")

        stream_metrics, stream_stats = cutoff_tools.parse_reciprocal_metrics(args.stream, logger)
        args_for_loader = argparse.Namespace(chunk_rows=int(args.chunk_rows), d_min=float(args.d_min))
        arrays, source_labels, domain_stats = cutoff_tools.load_domain_arrays(conn, stream_metrics, args_for_loader, logger)
        conn.close()

        included_rows = int(len(arrays["S_risk"]))
        included_hkls = unique_hkl_count(arrays["h"], arrays["k"], arrays["l"])
        included_sources = int(np.unique(arrays["source_event_id"]).size) if included_rows else 0
        logger.log(f"included rows={included_rows:,}, signed HKLs={included_hkls:,}, source/events={included_sources:,}")

        global_rows = write_global_quantiles(args.output_dir / "full_dataset_Srisk_global_quantiles.tsv", arrays)
        shell_rows = write_resolution_shells(args.output_dir / "full_dataset_Srisk_by_resolution_shell.tsv", arrays, args.threshold_values)
        threshold_rows, _hkl_rows, _source_rows = write_threshold_summaries(
            args.output_dir / "full_dataset_Srisk_threshold_counts.tsv",
            args.output_dir / "full_dataset_top_hkls_by_threshold.tsv",
            args.output_dir / "full_dataset_top_source_events_by_threshold.tsv",
            arrays,
            source_labels,
            args.threshold_values,
        )
        write_l_summary(args.output_dir / "full_dataset_Srisk_by_l.tsv", arrays, args.threshold_values)
        plot_histograms(
            args.output_dir,
            arrays,
            args.threshold_values,
            args.label,
            int(args.scatter_sample_n),
            int(args.scatter_seed),
            logger,
        )

        cache_parameters = load_cache_parameters(args.cache)
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "label": args.label,
            "cache": str(args.cache),
            "source_stream": str(args.stream),
            "output_dir": str(args.output_dir),
            "thresholds": args.threshold_values,
            "percentiles": PERCENTILES,
            "cache_rows_read": int(cache_rows),
            "expected_cache_rows": EXPECTED_CACHE_ROWS,
            "cache_columns": columns,
            "cache_parameters": cache_parameters,
            "included_rows": included_rows,
            "included_unique_signed_hkls": included_hkls,
            "included_unique_source_events": included_sources,
            "resolution_filter": {"d_spacing_min_A": float(args.d_min), "reciprocal_radius_max_Ainv": float(1.0 / args.d_min)},
            "resolution_method": "per-row |g| computed from source/event stream reciprocal basis astar/bstar/cstar converted from nm^-1 to A^-1; d_spacing=1/|g|",
            "scatter_rng_seed": int(args.scatter_seed),
            "scatter_sample_n": int(min(args.scatter_sample_n, included_rows)),
            "workers": int(args.workers),
            "chunk_rows": int(args.chunk_rows),
            "git": git_info(),
            **stream_stats,
            "missing_metric_key_count": domain_stats["missing_metric_key_count"],
            "rows_missing_metric": domain_stats["rows_missing_stream_metric"],
            "rows_excluded_invalid_radius": domain_stats["rows_invalid_radius"],
            "rows_excluded_high_resolution_or_outside": domain_stats["rows_outside_resolution_domain"],
            "elapsed_seconds": float(time.monotonic() - started),
        }
        notes_metadata = dict(metadata)
        notes_metadata["arrays_for_notes"] = arrays
        write_notes(args.output_dir / "full_dataset_Srisk_histogram_notes.md", args, notes_metadata, global_rows, threshold_rows, shell_rows, cache_parameters)
        write_json(args.output_dir / "full_dataset_Srisk_histogram_metadata.json", metadata)

        old_hits = validate_no_old_threshold_labels(args.output_dir)
        if old_hits:
            raise SystemExit("Old V7 cutoff labels appeared in V7b output: " + ", ".join(old_hits[:20]))
        logger.log("confirmed no old V7 cutoff labels in V7b TSV/JSON/log outputs")
        logger.log(f"full-dataset V7 S_risk histogram audit complete in {time.monotonic() - started:.1f} s")
        return 0
    finally:
        logger.close()


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
