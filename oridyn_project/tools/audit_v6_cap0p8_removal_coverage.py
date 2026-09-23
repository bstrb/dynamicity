#!/usr/bin/env python3
"""Audit V6 cap0.8 global EgM2 filtering breadth across signed HKLs.

The script reads existing cap-sweep metadata, merge QC logs, and the original
full-population SQLite score cache.  It reconstructs the global EgM2 removal
prefixes used by tools/build_v6_fullpop_global_egm2_cap_sweep_streams.py, then
writes compact TSV/Markdown audit outputs.

It does not run filtering, stream generation, Partialator, merging, or
refinement.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_fullpop_global_egm2_cap0p8_20260813"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_cap0p8_quick_coverage_audit_20260813"
)

COUNTS_NAME = "global_egm2_cap_sweep_counts.csv"
PARAMETERS_NAME = "global_egm2_cap_sweep_parameters.json"
SQL_CHUNK_ROWS = 250_000
PROGRESS_EVERY_SECONDS = 5.0

SUMMARY_RE = {
    "completeness": re.compile(r"Completeness:\s*([0-9.]+)\s*%"),
    "redundancy": re.compile(r"Redundancy:\s*([0-9.]+)"),
    "snr": re.compile(r"SNR:\s*([0-9.]+)"),
    "cc12": re.compile(r"CC1/2:\s*([0-9.]+)"),
    "rsplit": re.compile(r"Rsplit:\s*([0-9.]+)"),
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_counts(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / COUNTS_NAME
    if not path.is_file():
        raise SystemExit(f"Missing counts file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in counts file: {path}")
    for row in rows:
        row["drop_fraction_requested"] = float(row["drop_fraction_requested"])
        row["accepted_population_count"] = int(float(row["accepted_population_count"]))
        row["source_reflection_row_count"] = int(float(row["source_reflection_row_count"]))
        row["accepted_observations_removed"] = int(float(row["accepted_observations_removed"]))
        row["selected_hkl_count"] = int(float(row["selected_hkl_count"]))
        row["hkl_groups_at_cap"] = int(float(row["hkl_groups_at_cap"]))
        row["max_remove_per_hkl_fraction"] = float(row["max_remove_per_hkl_fraction"])
        row["min_retained_per_hkl"] = int(float(row["min_retained_per_hkl"]))
    return sorted(rows, key=lambda row: row["drop_fraction_requested"])


def variant_from_result_dir(path: Path) -> str:
    name = path.name
    marker = "_partialator_results"
    return name.split(marker, 1)[0] if marker in name else name


def parse_merge_qc(run_dir: Path, count_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {str(row["variant_id"]): row for row in count_rows}
    by_variant: dict[str, dict[str, Any]] = {}
    for result_dir in sorted(run_dir.glob("*partialator_results*")):
        if not result_dir.is_dir():
            continue
        variant = variant_from_result_dir(result_dir)
        summary_file = result_dir / "metadata_and_outputs.txt"
        row: dict[str, Any] = {
            "variant_id": variant,
            "result_dir": str(result_dir),
            "source_file": str(summary_file) if summary_file.is_file() else "",
            "status": "missing_summary",
            "completeness": "",
            "redundancy": "",
            "snr": "",
            "cc12": "",
            "rsplit": "",
        }
        if summary_file.is_file():
            text = summary_file.read_text(encoding="utf-8", errors="replace")
            row["status"] = "parsed"
            for key, pattern in SUMMARY_RE.items():
                match = pattern.search(text)
                if match:
                    row[key] = match.group(1)
        by_variant[variant] = row
    rows: list[dict[str, Any]] = []
    for variant, count_row in wanted.items():
        row = by_variant.get(
            variant,
            {
                "variant_id": variant,
                "result_dir": "",
                "source_file": "",
                "status": "missing_result_dir",
                "completeness": "",
                "redundancy": "",
                "snr": "",
                "cc12": "",
                "rsplit": "",
            },
        )
        row = dict(row)
        row["drop_fraction_requested"] = count_row["drop_fraction_requested"]
        row["removed_observations"] = count_row["accepted_observations_removed"]
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["drop_fraction_requested"]))


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def cache_schema(db_file: Path) -> set[str]:
    with connect_readonly(db_file) as conn:
        return {str(row[1]) for row in conn.execute("PRAGMA table_info(score_cache)").fetchall()}


def require_cache(db_file: Path) -> str:
    if not db_file.is_file():
        raise SystemExit(f"Source cache not found: {db_file}")
    present = cache_schema(db_file)
    required = {"ordinal", "h", "k", "l", "Eg", "M2"}
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"score_cache is missing required column(s): {missing}")
    return "source_order" if "source_order" in present else "ordinal"


def max_removable(n_obs: int, max_fraction: float, min_retained: int) -> int:
    return max(0, min(int(math.floor(float(max_fraction) * int(n_obs))), int(n_obs) - int(min_retained)))


def read_hkl_counts(db_file: Path) -> dict[tuple[int, int, int], int]:
    hkl_counts: dict[tuple[int, int, int], int] = {}
    started = time.monotonic()
    last = started
    log("reading signed-HKL counts from source cache")
    with connect_readonly(db_file) as conn:
        for h, k, l, n_obs in conn.execute("SELECT h,k,l,COUNT(*) FROM score_cache GROUP BY h,k,l ORDER BY h,k,l"):
            hkl_counts[(int(h), int(k), int(l))] = int(n_obs)
            now = time.monotonic()
            if now - last >= PROGRESS_EVERY_SECONDS:
                last = now
                log(f"  read {len(hkl_counts):,} signed HKLs; elapsed={format_duration(now - started)}")
    log(f"read {len(hkl_counts):,} signed HKLs")
    return hkl_counts


def load_ranked_arrays(db_file: Path, accepted_count: int, tie_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = f"SELECT h,k,l,{tie_column},Eg,M2 FROM score_cache"
    h_chunks: list[np.ndarray] = []
    k_chunks: list[np.ndarray] = []
    l_chunks: list[np.ndarray] = []
    tie_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    rows_read = 0
    started = time.monotonic()
    last = started
    log("reading EgM2 scores from source cache")
    with connect_readonly(db_file) as conn:
        for chunk in pd.read_sql_query(query, conn, chunksize=SQL_CHUNK_ROWS):
            eg = pd.to_numeric(chunk["Eg"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            m2 = pd.to_numeric(chunk["M2"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            scores = eg * m2
            if not np.isfinite(scores).all():
                raise SystemExit("Nonfinite EgM2 score encountered in score_cache")
            h_chunks.append(chunk["h"].to_numpy(dtype=np.int32, copy=True))
            k_chunks.append(chunk["k"].to_numpy(dtype=np.int32, copy=True))
            l_chunks.append(chunk["l"].to_numpy(dtype=np.int32, copy=True))
            tie_chunks.append(chunk[tie_column].to_numpy(dtype=np.int64, copy=True))
            score_chunks.append(scores.astype(np.float64, copy=True))
            rows_read += int(len(chunk))
            now = time.monotonic()
            if now - last >= PROGRESS_EVERY_SECONDS:
                last = now
                rate = rows_read / max(1.0e-9, now - started)
                pct = 100.0 * rows_read / max(1, accepted_count)
                eta = (accepted_count - rows_read) / rate if rate > 0 else float("inf")
                log(
                    f"  read {rows_read:,}/{accepted_count:,} rows ({pct:.1f}%); "
                    f"rate={rate:,.0f}/s; eta={format_duration(eta)}"
                )
    if rows_read != int(accepted_count):
        raise SystemExit(f"Read {rows_read:,} score rows, expected {accepted_count:,}")
    log("sorting by global EgM2 descending")
    h_values = np.concatenate(h_chunks)
    k_values = np.concatenate(k_chunks)
    l_values = np.concatenate(l_chunks)
    tie_values = np.concatenate(tie_chunks)
    score_values = np.concatenate(score_chunks)
    order = np.lexsort((tie_values, -score_values))
    return h_values[order], k_values[order], l_values[order], score_values[order]


def reconstruct_selected_hkls(
    db_file: Path,
    count_rows: list[dict[str, Any]],
    hkl_counts: dict[tuple[int, int, int], int],
    tie_column: str,
) -> list[tuple[int, int, int]]:
    accepted_count = int(count_rows[0]["accepted_population_count"])
    max_fraction = float(count_rows[0]["max_remove_per_hkl_fraction"])
    min_retained = int(count_rows[0]["min_retained_per_hkl"])
    max_needed = max(int(row["accepted_observations_removed"]) for row in count_rows)
    capacities = {hkl: max_removable(n_obs, max_fraction, min_retained) for hkl, n_obs in hkl_counts.items()}
    selected: list[tuple[int, int, int]] = []
    used: dict[tuple[int, int, int], int] = {}
    h_values, k_values, l_values, _scores = load_ranked_arrays(db_file, accepted_count, tie_column)
    started = time.monotonic()
    last = started
    log(f"reconstructing selected HKLs up to {max_needed:,} removed observations")
    for index, (h, k, l) in enumerate(zip(h_values, k_values, l_values), start=1):
        hkl = (int(h), int(k), int(l))
        n_used = used.get(hkl, 0)
        if n_used < capacities.get(hkl, 0):
            selected.append(hkl)
            used[hkl] = n_used + 1
            if len(selected) >= max_needed:
                break
        now = time.monotonic()
        if now - last >= PROGRESS_EVERY_SECONDS:
            last = now
            rate = index / max(1.0e-9, now - started)
            log(f"  scanned {index:,} ranked rows; selected {len(selected):,}; rate={rate:,.0f}/s")
    if len(selected) < max_needed:
        log(f"warning: selected only {len(selected):,} rows, but max requested was {max_needed:,}")
    log(f"reconstructed {len(selected):,} selected rows")
    return selected


def concentration_tables(
    count_rows: list[dict[str, Any]],
    selected_hkls: list[tuple[int, int, int]],
    hkl_counts: dict[tuple[int, int, int], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    max_fraction = float(count_rows[0]["max_remove_per_hkl_fraction"])
    min_retained = int(count_rows[0]["min_retained_per_hkl"])
    capacities = {hkl: max_removable(n_obs, max_fraction, min_retained) for hkl, n_obs in hkl_counts.items()}
    for count_row in count_rows:
        variant = str(count_row["variant_id"])
        removed_total = int(count_row["accepted_observations_removed"])
        counter = Counter(selected_hkls[:removed_total])
        removed_counts = sorted(counter.values(), reverse=True)
        affected = len(removed_counts)
        median_removed = float(np.median(removed_counts)) if removed_counts else 0.0
        max_removed = int(removed_counts[0]) if removed_counts else 0
        fractions = [counter[hkl] / hkl_counts[hkl] for hkl in counter]
        max_removed_fraction = max(fractions) if fractions else 0.0
        hkls_at_cap = sum(1 for hkl, n_removed in counter.items() if n_removed >= capacities.get(hkl, 0) > 0)
        hkls_min_retained = sum(1 for hkl, n_removed in counter.items() if hkl_counts[hkl] - n_removed == min_retained)
        top10_fraction = sum(removed_counts[:10]) / removed_total if removed_total else 0.0
        top50_fraction = sum(removed_counts[:50]) / removed_total if removed_total else 0.0
        top100_fraction = sum(removed_counts[:100]) / removed_total if removed_total else 0.0
        summary_rows.append(
            {
                "variant_id": variant,
                "drop_fraction_requested": count_row["drop_fraction_requested"],
                "total_removed_observations": removed_total,
                "affected_signed_hkls": affected,
                "median_removed_per_affected_hkl": f"{median_removed:.3f}",
                "max_removed_per_affected_hkl": max_removed,
                "max_removed_fraction_any_hkl": f"{max_removed_fraction:.6f}",
                "hkls_hitting_80pct_cap": hkls_at_cap,
                "hkls_ending_with_min_retained": hkls_min_retained,
                "min_retained_per_hkl": min_retained,
                "top10_removed_fraction": f"{top10_fraction:.6f}",
                "top50_removed_fraction": f"{top50_fraction:.6f}",
                "top100_removed_fraction": f"{top100_fraction:.6f}",
                "compact_manifest_selected_hkls": count_row["selected_hkl_count"],
                "compact_manifest_hkls_at_cap": count_row["hkl_groups_at_cap"],
            }
        )
        top = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:20]
        for rank, (hkl, n_removed) in enumerate(top, start=1):
            n_obs = hkl_counts[hkl]
            retained = n_obs - n_removed
            top_rows.append(
                {
                    "variant_id": variant,
                    "rank": rank,
                    "h": hkl[0],
                    "k": hkl[1],
                    "l": hkl[2],
                    "source_observations": n_obs,
                    "removed_observations": n_removed,
                    "retained_observations": retained,
                    "removed_fraction": f"{n_removed / n_obs:.6f}",
                    "hit_80pct_cap": int(n_removed >= capacities.get(hkl, 0) > 0),
                    "ended_with_min_retained": int(retained == min_retained),
                }
            )
    return summary_rows, top_rows


def write_recommendation(path: Path, merge_rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> None:
    by_variant = {row["variant_id"]: row for row in coverage_rows}
    qc = {row["variant_id"]: row for row in merge_rows}
    drop0p5 = next((row for row in coverage_rows if abs(float(row["drop_fraction_requested"]) - 0.005) < 1.0e-12), None)
    lines = [
        "# Quick Recommendation",
        "",
        "This audit covers the V6 full-population global EgM2 high-risk tail with max 80% removed per signed HKL and at least 2 retained observations per signed HKL.",
        "",
    ]
    if drop0p5 is None:
        lines.append("No drop0p5 row was found, so the 0.5% case could not be assessed.")
    else:
        variant = str(drop0p5["variant_id"])
        q = qc.get(variant, {})
        top100 = float(drop0p5["top100_removed_fraction"])
        affected = int(drop0p5["affected_signed_hkls"])
        removed = int(drop0p5["total_removed_observations"])
        concentration = "broad" if affected >= 1000 and top100 < 0.10 else "concentrated"
        lines.extend(
            [
                f"- 0.5% case: {removed:,} removed observations across {affected:,} signed HKLs; top 100 HKLs contribute {100.0 * top100:.2f}% of removals.",
                f"- Breadth call: {concentration}.",
                f"- Merge QC at 0.5%: CC1/2={q.get('cc12', 'NA')}, Rsplit={q.get('rsplit', 'NA')}%, SNR={q.get('snr', 'NA')}, redundancy={q.get('redundancy', 'NA')}x.",
            ]
        )
        if concentration == "broad":
            lines.append("- Presentation wording: reasonable to describe as a full-population high-risk-tail diagnostic, but keep the claim modest because only the gentle tail improves.")
        else:
            lines.append("- Presentation wording: describe only as a small high-risk-tail diagnostic, not as a general orientation-risk filtering result.")
    if coverage_rows:
        best_cc = max(merge_rows, key=lambda row: float(row["cc12"] or "nan"))
        lines.extend(
            [
                "",
                f"Best CC1/2 among parsed cap0.8 runs: {best_cc['variant_id']} with CC1/2={best_cc.get('cc12', 'NA')} and Rsplit={best_cc.get('rsplit', 'NA')}%.",
                "Higher fractions are useful as a stress test; they should not be presented as improving filters unless their merge QC beats the full-data reference.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(run_dir: Path, output_dir: Path, source_cache: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count_rows = read_counts(run_dir)
    params = read_json(run_dir / PARAMETERS_NAME)
    if source_cache is None:
        raw_cache = params.get("source_cache")
        if not raw_cache:
            raise SystemExit(f"No source_cache in {run_dir / PARAMETERS_NAME}; pass --source-cache")
        source_cache = Path(raw_cache)
    source_cache = source_cache.expanduser().resolve()
    tie_column = require_cache(source_cache)
    merge_rows = parse_merge_qc(run_dir, count_rows)
    write_tsv(
        output_dir / "merge_qc_table.tsv",
        merge_rows,
        [
            "variant_id",
            "drop_fraction_requested",
            "removed_observations",
            "completeness",
            "redundancy",
            "snr",
            "cc12",
            "rsplit",
            "status",
            "source_file",
            "result_dir",
        ],
    )
    hkl_counts = read_hkl_counts(source_cache)
    selected_hkls = reconstruct_selected_hkls(source_cache, count_rows, hkl_counts, tie_column)
    coverage_rows, top_rows = concentration_tables(count_rows, selected_hkls, hkl_counts)
    write_tsv(output_dir / "removal_coverage_summary.tsv", coverage_rows)
    write_tsv(output_dir / "top_removed_hkls.tsv", top_rows)
    write_recommendation(output_dir / "quick_recommendation.md", merge_rows, coverage_rows)
    log(f"wrote audit outputs to {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-cache", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def self_test() -> int:
    root = Path(__file__).resolve().parents[1] / ".codex_smoke" / "cap0p8_audit"
    if root.exists():
        shutil.rmtree(root)
    run_dir = root / "run"
    out_dir = root / "out"
    run_dir.mkdir(parents=True)
    source_cache = root / "source.sqlite"
    try:
        conn = sqlite3.connect(source_cache)
        conn.execute(
            """
            CREATE TABLE score_cache(
                ordinal INTEGER PRIMARY KEY,
                h INTEGER,
                k INTEGER,
                l INTEGER,
                source_order INTEGER,
                Eg REAL,
                M2 REAL
            )
            """
        )
        rows = []
        ordinal = 0
        for h in [1, 2, 3]:
            for i in range(10):
                rows.append((ordinal, h, 0, 0, ordinal, 100.0 - ordinal, 1.0))
                ordinal += 1
        conn.executemany("INSERT INTO score_cache VALUES (?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        (run_dir / PARAMETERS_NAME).write_text(json.dumps({"source_cache": str(source_cache)}) + "\n", encoding="utf-8")
        with (run_dir / COUNTS_NAME).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "variant_id",
                    "drop_fraction_requested",
                    "accepted_population_count",
                    "source_reflection_row_count",
                    "accepted_observations_removed",
                    "selected_hkl_count",
                    "hkl_groups_at_cap",
                    "max_remove_per_hkl_fraction",
                    "min_retained_per_hkl",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "variant_id": "filter_all_global_eg_m2_cap0p8_drop10",
                    "drop_fraction_requested": "0.10",
                    "accepted_population_count": "30",
                    "source_reflection_row_count": "30",
                    "accepted_observations_removed": "3",
                    "selected_hkl_count": "1",
                    "hkl_groups_at_cap": "0",
                    "max_remove_per_hkl_fraction": "0.8",
                    "min_retained_per_hkl": "2",
                }
            )
        result = run_dir / "filter_all_global_eg_m2_cap0p8_drop10_partialator_results_20260813T0000"
        result.mkdir()
        (result / "metadata_and_outputs.txt").write_text(
            "=== SUMMARY ===\nCompleteness: 99.89%\nRedundancy:   530.00x\nSNR:         16.90\nCC1/2:        0.9971000\nRsplit:       5.44\n",
            encoding="utf-8",
        )
        audit(run_dir, out_dir, source_cache)
        for name in ["merge_qc_table.tsv", "removal_coverage_summary.tsv", "top_removed_hkls.tsv", "quick_recommendation.md"]:
            assert (out_dir / name).is_file(), name
        print("self-test passed: wrote merge QC, coverage summary, top HKLs, and recommendation")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_cache = args.source_cache.expanduser().resolve() if args.source_cache is not None else None
    if not run_dir.is_dir():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    audit(run_dir, output_dir, source_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
