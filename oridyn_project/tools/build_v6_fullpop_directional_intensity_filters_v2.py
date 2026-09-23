#!/usr/bin/env python3
"""Build aggressive V2 directional-intensity full-population filters.

V2 fixes the first directional-intensity proof of concept by ranking broadly
across the finite scoreable population instead of applying hard directional
eligibility gates.  It plans aggressive global removal fractions while obeying
per-HKL safety caps.  Stream files are written only with --write-streams.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from typing import Any, Iterable, Iterator

for _env_name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(_env_name, "1")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_v6_fullpop_directional_intensity_filters as v1


DEFAULT_SOURCE_OUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_full_population_sweep_20260717"
)
DEFAULT_SOURCE_STREAM = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "MFM300-VIII_cut_20-0_3.stream"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_v6_fullpop_directional_intensity_v2_20260811"
)

FAMILY_ORDER = [
    "egm2_global",
    "directional_consistent_soft",
    "directional_extreme_soft",
    "enhancement_soft",
    "depletion_soft",
]
DEFAULT_FAMILIES = tuple(FAMILY_ORDER)
DEFAULT_FRACTIONS = ("0.005", "0.010", "0.020", "0.050", "0.100", "0.150", "0.200")
CONTROL_ORDER = ["matched_random", "low_rank_score", "sign_mismatch_directional"]
DEFAULT_RANDOM_SEEDS = ("1", "2", "3")

EPS = 1.0e-12
ROBUST_MAD_SCALE = 1.4826
MAX_REMOVE_PER_HKL_FRACTION = 0.50
MIN_RETAINED_PER_HKL = 5
MIN_FILL_FRACTION = 0.95
PLAN_BATCH_HKLS = 250
FRAME_FUTURE_MULTIPLIER = 3
MAX_INSERT_ROWS = 50_000

HKL_COLUMNS = ["h", "k", "l"]
RANK_COLUMNS = {
    "egm2_global": "rank_egm2_global",
    "directional_consistent_soft": "rank_directional_consistent_soft",
    "directional_extreme_soft": "rank_directional_extreme_soft",
    "enhancement_soft": "rank_enhancement_soft",
    "depletion_soft": "rank_depletion_soft",
}
RANK_FORMULAS = {
    "egm2_global": "EgM2",
    "directional_consistent_soft": "EgM2 * (1 + clip(max(geom_delta_z*J_resid, 0), 0, 5))",
    "directional_extreme_soft": "EgM2 * (1 + clip(abs(geom_delta_z), 0, 5)) * (1 + clip(abs(J_resid), 0, 5))",
    "enhancement_soft": "EgM2 * (1 + clip(max(geom_delta_z, 0), 0, 5)) * (1 + clip(max(J_resid, 0), 0, 5))",
    "depletion_soft": "EgM2 * (1 + clip(max(-geom_delta_z, 0), 0, 5)) * (1 + clip(max(-J_resid, 0), 0, 5))",
}


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    base_variant_id: str
    family: str
    variant_role: str
    control_type: str
    random_seed: int | None
    fraction_text: str
    fraction: float
    rank_column: str
    rank_formula: str
    output_filename: str
    output_stream: Path


RunLogger = v1.RunLogger
StageProgress = v1.StageProgress
PackedMask = v1.PackedMask


def write_json(path: Path, payload: Any) -> None:
    v1.write_json(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    return v1.read_json(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None, delimiter: str = ",") -> None:
    v1.write_csv(path, rows, fieldnames=fieldnames, delimiter=delimiter)


def parse_fraction_items(value: str | None) -> list[tuple[str, float]]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_FRACTIONS)
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        fraction = float(item)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise SystemExit(f"Invalid drop fraction {item!r}; expected 0 < fraction < 1")
        if item in seen:
            raise SystemExit(f"Duplicate fraction label: {item}")
        out.append((item, fraction))
        seen.add(item)
    if not out:
        raise SystemExit("--fractions must contain at least one value")
    return out


def parse_families(value: str | None) -> list[str]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_FAMILIES)
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(text).split(","):
        family = raw.strip()
        if not family:
            continue
        if family not in FAMILY_ORDER:
            raise SystemExit(f"Unknown family {family!r}; expected one of {', '.join(FAMILY_ORDER)}")
        if family not in seen:
            out.append(family)
            seen.add(family)
    if not out:
        raise SystemExit("--families must contain at least one value")
    return out


def parse_controls(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in str(value).split(","):
        control = raw.strip()
        if not control:
            continue
        if control not in CONTROL_ORDER:
            raise SystemExit(f"Unknown control {control!r}; expected one of {', '.join(CONTROL_ORDER)}")
        if control not in seen:
            out.append(control)
            seen.add(control)
    return out


def parse_seed_items(value: str | None) -> list[int]:
    text = value if value is not None and str(value).strip() else ",".join(DEFAULT_RANDOM_SEEDS)
    out: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        seed = int(item)
        if seed < 0:
            raise SystemExit("--random-seeds values must be >= 0")
        out.append(seed)
    if not out:
        raise SystemExit("--random-seeds must contain at least one seed")
    return out


def fraction_label(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("+"):
        stripped = stripped[1:]
    return stripped.replace(".", "p").replace("-", "m")


def build_variants(
    families: list[str],
    fractions: list[tuple[str, float]],
    controls: list[str],
    random_seeds: list[int],
    out_dir: Path,
) -> tuple[list[VariantSpec], list[VariantSpec]]:
    targeted: list[VariantSpec] = []
    control_variants: list[VariantSpec] = []
    for family in families:
        rank_column = RANK_COLUMNS[family]
        rank_formula = RANK_FORMULAS[family]
        for fraction_text, fraction in fractions:
            base = f"directional_v2_{family}_drop{fraction_label(fraction_text)}"
            targeted.append(
                VariantSpec(
                    variant_id=base,
                    base_variant_id=base,
                    family=family,
                    variant_role="targeted",
                    control_type="",
                    random_seed=None,
                    fraction_text=fraction_text,
                    fraction=float(fraction),
                    rank_column=rank_column,
                    rank_formula=rank_formula,
                    output_filename=f"{base}.stream",
                    output_stream=out_dir / f"{base}.stream",
                )
            )
            if "low_rank_score" in controls:
                control_id = f"{base}_low_rank_score"
                control_variants.append(
                    VariantSpec(
                        variant_id=control_id,
                        base_variant_id=base,
                        family=family,
                        variant_role="control",
                        control_type="low_rank_score",
                        random_seed=None,
                        fraction_text=fraction_text,
                        fraction=float(fraction),
                        rank_column=rank_column,
                        rank_formula=f"control: lowest {rank_formula} within each selected signed HKL",
                        output_filename=f"{control_id}.stream",
                        output_stream=out_dir / f"{control_id}.stream",
                    )
                )
            if "sign_mismatch_directional" in controls:
                control_id = f"{base}_sign_mismatch_directional"
                control_variants.append(
                    VariantSpec(
                        variant_id=control_id,
                        base_variant_id=base,
                        family=family,
                        variant_role="control",
                        control_type="sign_mismatch_directional",
                        random_seed=None,
                        fraction_text=fraction_text,
                        fraction=float(fraction),
                        rank_column=rank_column,
                        rank_formula="control: prefer geom_delta_z*J_resid <= 0 within each selected signed HKL",
                        output_filename=f"{control_id}.stream",
                        output_stream=out_dir / f"{control_id}.stream",
                    )
                )
            if "matched_random" in controls:
                for seed in random_seeds:
                    control_id = f"{base}_matched_random_seed{int(seed)}"
                    control_variants.append(
                        VariantSpec(
                            variant_id=control_id,
                            base_variant_id=base,
                            family=family,
                            variant_role="control",
                            control_type="matched_random",
                            random_seed=int(seed),
                            fraction_text=fraction_text,
                            fraction=float(fraction),
                            rank_column=rank_column,
                            rank_formula=f"control: deterministic random within each selected signed HKL, seed={int(seed)}",
                            output_filename=f"{control_id}.stream",
                            output_stream=out_dir / f"{control_id}.stream",
                        )
                    )
    all_ids = [variant.variant_id for variant in targeted + control_variants]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Internal error: duplicate variant ids")
    return targeted, control_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=DEFAULT_SOURCE_STREAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--fractions", default=",".join(DEFAULT_FRACTIONS))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-streams", action="store_true")
    parser.add_argument("--controls", default="")
    parser.add_argument("--random-seeds", default=",".join(DEFAULT_RANDOM_SEEDS))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.source_stream = args.source_stream.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.family_items = parse_families(args.families)
    args.fraction_items = parse_fraction_items(args.fractions)
    args.control_items = parse_controls(args.controls)
    args.random_seed_items = parse_seed_items(args.random_seeds)
    args.workers = max(1, int(args.workers))
    if args.dry_run and args.write_streams:
        raise SystemExit("Use either --dry-run or --write-streams, not both")
    if not args.write_streams:
        args.dry_run = True
    if not args.source_out_dir.is_dir():
        raise SystemExit(f"--source-out-dir not found: {args.source_out_dir}")
    if not args.source_stream.is_file():
        raise SystemExit(f"--source-stream not found: {args.source_stream}")
    return args


def cache_db_path(source_out_dir: Path) -> Path:
    return source_out_dir / "full_population_cache.sqlite"


def work_db_path(out_dir: Path) -> Path:
    return out_dir / ".directional_v2_work.sqlite"


def connect_readonly(db_file: Path) -> sqlite3.Connection:
    return v1.connect_readonly(db_file)


def connect_work(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def planned_output_paths(out_dir: Path, variants: list[VariantSpec], write_streams: bool) -> list[Path]:
    paths = [
        out_dir / "directional_v2_manifest.tsv",
        out_dir / "directional_v2_counts.csv",
        out_dir / "directional_v2_removed_keys.tsv.gz",
        out_dir / "directional_v2_parameters.json",
        out_dir / "directional_v2_metadata.json",
        out_dir / "run.log",
        work_db_path(out_dir),
    ]
    if write_streams:
        paths.extend(variant.output_stream for variant in variants)
    return paths


def prepare_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(str(path) for path in existing[:20])
        extra = "" if len(existing) <= 20 else f"\n  ... {len(existing) - 20} more"
        raise SystemExit(f"Refusing to overwrite existing output(s):\n  {preview}{extra}")
    if overwrite:
        for path in existing:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                raise SystemExit(f"Refusing to overwrite non-file output path: {path}")


def cache_schema(db_file: Path) -> list[str]:
    with connect_readonly(db_file) as conn:
        return [str(row[1]) for row in conn.execute("PRAGMA table_info(score_cache)").fetchall()]


def require_cache_schema(db_file: Path) -> list[str]:
    present = cache_schema(db_file)
    required = {"ordinal", "source_filename", "event", "h", "k", "l", "exact_key_text", "source_order", "Eg", "D", "M", "M2"}
    missing = sorted(required - set(present))
    if missing:
        raise SystemExit(
            "full_population_cache.sqlite score_cache is missing required V2 columns: "
            f"{missing}. M and D are required for A=M/(D+eps)."
        )
    return present


def score_cache_stats(db_file: Path) -> dict[str, int]:
    with connect_readonly(db_file) as conn:
        row = conn.execute("SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM score_cache").fetchone()
    count = int(row[0])
    min_ordinal = 0 if row[1] is None else int(row[1])
    max_ordinal = -1 if row[2] is None else int(row[2])
    if min_ordinal < 0:
        raise SystemExit(f"Negative score-cache ordinal encountered: {min_ordinal}")
    return {"count": count, "min_ordinal": min_ordinal, "max_ordinal": max_ordinal, "mask_bits": max(count, max_ordinal + 1)}


def count_signed_hkls(db_file: Path) -> int:
    with connect_readonly(db_file) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM (SELECT h,k,l FROM score_cache GROUP BY h,k,l)").fetchone()[0])


def source_reflection_row_count(source_out_dir: Path) -> int | None:
    return v1.source_reflection_row_count(source_out_dir)


def init_work_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE hkl_stats (
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            n_obs INTEGER NOT NULL,
            geom_delta_median REAL NOT NULL,
            geom_delta_mad REAL NOT NULL,
            geom_delta_denom REAL NOT NULL,
            max_remove INTEGER NOT NULL,
            PRIMARY KEY (h,k,l)
        );
        CREATE TABLE features (
            ordinal INTEGER PRIMARY KEY,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            intensity REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL,
            egm2 REAL NOT NULL,
            geom_delta REAL NOT NULL,
            geom_delta_z REAL NOT NULL,
            j_resid REAL NOT NULL,
            local_neighbor_count INTEGER NOT NULL,
            rank_egm2_global REAL NOT NULL,
            rank_directional_consistent_soft REAL NOT NULL,
            rank_directional_extreme_soft REAL NOT NULL,
            rank_enhancement_soft REAL NOT NULL,
            rank_depletion_soft REAL NOT NULL
        );
        CREATE TABLE selected (
            variant_id TEXT NOT NULL,
            base_variant_id TEXT NOT NULL,
            variant_role TEXT NOT NULL,
            control_type TEXT NOT NULL,
            random_seed INTEGER,
            family TEXT NOT NULL,
            global_fraction REAL NOT NULL,
            fraction_label TEXT NOT NULL,
            rank_formula TEXT NOT NULL,
            selection_rank INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            intensity REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL,
            egm2 REAL NOT NULL,
            geom_delta REAL NOT NULL,
            geom_delta_z REAL NOT NULL,
            j_resid REAL NOT NULL,
            local_neighbor_count INTEGER NOT NULL,
            rank_score REAL NOT NULL,
            PRIMARY KEY (variant_id, ordinal)
        );
        """
    )
    conn.commit()


def iter_hkl_batches(db_file: Path, batch_hkls: int = PLAN_BATCH_HKLS) -> Iterator[list[tuple[int, int, int]]]:
    with connect_readonly(db_file) as conn:
        cursor = conn.execute("SELECT h,k,l FROM score_cache GROUP BY h,k,l ORDER BY h,k,l")
        batch: list[tuple[int, int, int]] = []
        for h, k, l in cursor:
            batch.append((int(h), int(k), int(l)))
            if len(batch) >= int(batch_hkls):
                yield batch
                batch = []
        if batch:
            yield batch


def fetch_hkl_batch(db_file: str, hkls: list[tuple[int, int, int]]) -> pd.DataFrame:
    if not hkls:
        return pd.DataFrame(columns=["h", "k", "l", "Eg", "D", "M", "M2"])
    conn = connect_readonly(Path(db_file))
    try:
        frames: list[pd.DataFrame] = []
        for start in range(0, len(hkls), 300):
            chunk = hkls[start : start + 300]
            values = ",".join(["(?,?,?)"] * len(chunk))
            params = [value for hkl in chunk for value in hkl]
            frames.append(
                pd.read_sql_query(
                    f"""
                    WITH batch_hkl(h,k,l) AS (VALUES {values})
                    SELECT sc.h,sc.k,sc.l,sc.Eg,sc.D,sc.M,sc.M2
                    FROM score_cache AS sc
                    JOIN batch_hkl AS b ON sc.h=b.h AND sc.k=b.k AND sc.l=b.l
                    ORDER BY sc.h,sc.k,sc.l
                    """,
                    conn,
                    params=params,
                )
            )
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    finally:
        conn.close()


def worker_hkl_stats(task: tuple[str, list[tuple[int, int, int]]]) -> tuple[list[tuple[Any, ...]], dict[str, int]]:
    db_file, hkls = task
    table = fetch_hkl_batch(db_file, hkls)
    rows: list[tuple[Any, ...]] = []
    stats = {"hkl_count": 0, "observation_count": int(len(table))}
    if table.empty:
        return rows, stats
    for column in ["Eg", "D", "M", "M2"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["Eg", "D", "M", "M2"]].isna().any().any():
        raise RuntimeError("Nonfinite Eg/D/M/M2 encountered while computing geom_delta_z")
    if (table["D"].to_numpy(dtype=float) < 0.0).any():
        raise RuntimeError("Negative D encountered while computing geom_delta_z")
    eg = table["Eg"].to_numpy(dtype=float)
    d = table["D"].to_numpy(dtype=float)
    m = table["M"].to_numpy(dtype=float)
    table = table.assign(geom_delta=m / (d + EPS) - eg)
    for hkl, group in table.groupby(HKL_COLUMNS, sort=False):
        values = group["geom_delta"].to_numpy(dtype=float, copy=False)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        denom = float(ROBUST_MAD_SCALE * mad + EPS)
        n_obs = int(len(values))
        max_remove = int(max(0, min(math.floor(MAX_REMOVE_PER_HKL_FRACTION * n_obs), n_obs - MIN_RETAINED_PER_HKL)))
        rows.append((int(hkl[0]), int(hkl[1]), int(hkl[2]), n_obs, median, mad, denom, max_remove))
        stats["hkl_count"] += 1
    return rows, stats


def populate_hkl_stats(cache_db: Path, work_conn: sqlite3.Connection, workers: int, logger: RunLogger) -> dict[str, Any]:
    total_hkls = count_signed_hkls(cache_db)
    batches = list(iter_hkl_batches(cache_db))
    progress = StageProgress(logger, "v2: computing per-HKL geom_delta statistics", total_hkls, "HKLs", workers, 10.0)
    completed_hkls = 0
    completed_observations = 0
    work_conn.execute("BEGIN")
    try:
        if workers > 1 and len(batches) > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [executor.submit(worker_hkl_stats, (str(cache_db), batch)) for batch in batches]
                for future in as_completed(futures):
                    rows, stats = future.result()
                    work_conn.executemany(
                        """
                        INSERT INTO hkl_stats(h,k,l,n_obs,geom_delta_median,geom_delta_mad,geom_delta_denom,max_remove)
                        VALUES(?,?,?,?,?,?,?,?)
                        """,
                        rows,
                    )
                    completed_hkls += int(stats["hkl_count"])
                    completed_observations += int(stats["observation_count"])
                    progress.update(completed_hkls)
        else:
            for batch in batches:
                rows, stats = worker_hkl_stats((str(cache_db), batch))
                work_conn.executemany(
                    """
                    INSERT INTO hkl_stats(h,k,l,n_obs,geom_delta_median,geom_delta_mad,geom_delta_denom,max_remove)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    rows,
                )
                completed_hkls += int(stats["hkl_count"])
                completed_observations += int(stats["observation_count"])
                progress.update(completed_hkls)
        work_conn.commit()
    except Exception:
        work_conn.rollback()
        raise
    progress.finish(completed_hkls)
    work_conn.execute("CREATE INDEX idx_hkl_stats_limit_v2 ON hkl_stats(h,k,l,max_remove)")
    work_conn.commit()
    return {
        "signed_hkl_count": int(total_hkls),
        "completed_hkls": int(completed_hkls),
        "completed_observations": int(completed_observations),
        "batch_count": int(len(batches)),
    }


def query_cache_rows(conn: sqlite3.Connection, keys: list[str]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for start in range(0, len(keys), 700):
        chunk = keys[start : start + 700]
        values = ",".join(["(?)"] * len(chunk))
        frames.append(
            pd.read_sql_query(
                f"""
                WITH frame_keys(exact_key_text) AS (VALUES {values})
                SELECT sc.ordinal,sc.exact_key_text,sc.source_filename,sc.event,sc.h,sc.k,sc.l,
                       sc.Eg,sc.D,sc.M,sc.M2,
                       hs.n_obs AS hkl_n_obs,hs.geom_delta_median,hs.geom_delta_denom,hs.max_remove
                FROM frame_keys AS fk
                JOIN score_cache AS sc ON sc.exact_key_text=fk.exact_key_text
                JOIN work.hkl_stats AS hs ON hs.h=sc.h AND hs.k=sc.k AND hs.l=sc.l
                ORDER BY sc.exact_key_text
                """,
                conn,
                params=chunk,
            )
        )
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def clip0_5(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 5.0)


def feature_rows_from_joined(joined: pd.DataFrame, metric: np.ndarray) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    required = ["Eg", "D", "M", "M2", "intensity", "geom_delta_median", "geom_delta_denom"]
    for column in required:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
    if joined[required].isna().any().any():
        raise RuntimeError("Nonfinite joined intensity/cache values encountered; cannot rank V2 observations")
    if (joined["D"].to_numpy(dtype=float) < 0.0).any():
        raise RuntimeError("Negative D encountered; cannot compute A=M/(D+eps)")

    eg = joined["Eg"].to_numpy(dtype=float)
    d = joined["D"].to_numpy(dtype=float)
    m = joined["M"].to_numpy(dtype=float)
    m2 = joined["M2"].to_numpy(dtype=float)
    intensity = joined["intensity"].to_numpy(dtype=float)
    egm2 = eg * m2
    geom_delta = m / (d + EPS) - eg
    geom_delta_z = (geom_delta - joined["geom_delta_median"].to_numpy(dtype=float)) / joined["geom_delta_denom"].to_numpy(dtype=float)
    j_values = intensity * eg
    if not np.isfinite(j_values).all():
        raise RuntimeError("Nonfinite J=I*Eg encountered; cannot compute local residuals")
    hkls = joined.loc[:, HKL_COLUMNS].to_numpy(dtype=np.int64)
    j_resid, neighbor_counts = v1.local_j_residuals(hkls, j_values, metric)

    finite = (
        np.isfinite(egm2)
        & np.isfinite(geom_delta)
        & np.isfinite(geom_delta_z)
        & np.isfinite(j_resid)
        & np.isfinite(intensity)
    )
    stats = {
        "joined_rows": int(len(joined)),
        "finite_rankable_rows": int(np.sum(finite)),
        "finite_residual_rows": int(np.isfinite(j_resid).sum()),
        "too_few_neighbor_rows": int((neighbor_counts < v1.MIN_LOCAL_NEIGHBORS).sum()),
    }
    if not np.any(finite):
        return [], stats

    agreement = geom_delta_z * j_resid
    rank_egm2 = egm2
    rank_consistent = egm2 * (1.0 + clip0_5(np.maximum(agreement, 0.0)))
    rank_extreme = egm2 * (1.0 + clip0_5(np.abs(geom_delta_z))) * (1.0 + clip0_5(np.abs(j_resid)))
    rank_enhance = egm2 * (1.0 + clip0_5(np.maximum(geom_delta_z, 0.0))) * (1.0 + clip0_5(np.maximum(j_resid, 0.0)))
    rank_deplete = egm2 * (1.0 + clip0_5(np.maximum(-geom_delta_z, 0.0))) * (1.0 + clip0_5(np.maximum(-j_resid, 0.0)))

    rows: list[tuple[Any, ...]] = []
    finite_positions = np.flatnonzero(finite)
    for pos in finite_positions:
        row = joined.iloc[int(pos)]
        ranks = [rank_egm2[pos], rank_consistent[pos], rank_extreme[pos], rank_enhance[pos], rank_deplete[pos]]
        if not all(math.isfinite(float(value)) for value in ranks):
            continue
        rows.append(
            (
                int(row["ordinal"]),
                str(row["exact_key_text"]),
                str(row["source_filename"]),
                str(row["event"]),
                int(row["h"]),
                int(row["k"]),
                int(row["l"]),
                float(intensity[pos]),
                float(eg[pos]),
                float(d[pos]),
                float(m[pos]),
                float(m2[pos]),
                float(egm2[pos]),
                float(geom_delta[pos]),
                float(geom_delta_z[pos]),
                float(j_resid[pos]),
                int(neighbor_counts[pos]),
                float(rank_egm2[pos]),
                float(rank_consistent[pos]),
                float(rank_extreme[pos]),
                float(rank_enhance[pos]),
                float(rank_deplete[pos]),
            )
        )
    stats["feature_rows"] = int(len(rows))
    return rows, stats


def worker_score_frame(task: tuple[str, str, list[dict[str, Any]], list[list[float]]]) -> dict[str, Any]:
    cache_db, work_db, records_payload, metric_payload = task
    metric = np.asarray(metric_payload, dtype=float)
    stats: dict[str, Any] = {
        "frames": 1,
        "reflection_rows": int(len(records_payload)),
        "joined_rows": 0,
        "missing_cache_rows": 0,
        "finite_residual_rows": 0,
        "too_few_neighbor_rows": 0,
        "finite_rankable_rows": 0,
        "feature_rows": 0,
    }
    if not records_payload:
        return {"feature_rows": [], "stats": stats}
    frame = pd.DataFrame(records_payload)
    if "intensity" not in frame or frame["intensity"].isna().all():
        raise RuntimeError("Frame contains no parsable intensities")

    conn = sqlite3.connect(f"file:{Path(cache_db).resolve()}?mode=ro", uri=True)
    conn.execute("ATTACH DATABASE ? AS work", (str(Path(work_db).resolve()),))
    try:
        cache_rows = query_cache_rows(conn, frame["exact_key_text"].astype(str).tolist())
    finally:
        conn.close()
    if cache_rows.empty:
        stats["missing_cache_rows"] = int(len(frame))
        return {"feature_rows": [], "stats": stats}
    joined = cache_rows.merge(frame.loc[:, ["exact_key_text", "intensity"]], on="exact_key_text", how="left", sort=False, validate="one_to_one")
    stats["missing_cache_rows"] = int(len(frame) - len(joined))
    rows, feature_stats = feature_rows_from_joined(joined, metric)
    stats.update(feature_stats)
    return {"feature_rows": rows, "stats": stats}


def feature_insert_sql() -> str:
    return """
        INSERT OR IGNORE INTO features(
            ordinal,exact_key_text,source_filename,event,h,k,l,intensity,Eg,D,M,M2,egm2,
            geom_delta,geom_delta_z,j_resid,local_neighbor_count,
            rank_egm2_global,rank_directional_consistent_soft,rank_directional_extreme_soft,
            rank_enhancement_soft,rank_depletion_soft
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """


def merge_scan_stats(total: dict[str, Any], stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        total[key] = int(total.get(key, 0)) + int(value)


def drain_frame_futures(
    futures: set[Any],
    work_conn: sqlite3.Connection,
    scan_stats: dict[str, Any],
    progress: StageProgress,
    completed_frames: int,
    force_all: bool = False,
) -> tuple[set[Any], int]:
    if not futures:
        return futures, completed_frames
    if force_all:
        done = set(futures)
    else:
        done, _pending = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        result = future.result()
        rows = result["feature_rows"]
        if rows:
            work_conn.executemany(feature_insert_sql(), rows)
        merge_scan_stats(scan_stats, result["stats"])
        completed_frames += int(result["stats"].get("frames", 0))
        progress.update(completed_frames)
    futures -= done
    return futures, completed_frames


def populate_features(
    cache_db: Path,
    work_db: Path,
    work_conn: sqlite3.Connection,
    source_stream: Path,
    metric: np.ndarray,
    workers: int,
    logger: RunLogger,
) -> dict[str, Any]:
    scan_stats: dict[str, Any] = {
        "frames": 0,
        "reflection_rows": 0,
        "joined_rows": 0,
        "missing_cache_rows": 0,
        "finite_residual_rows": 0,
        "too_few_neighbor_rows": 0,
        "finite_rankable_rows": 0,
        "feature_rows": 0,
    }
    progress = StageProgress(logger, "v2: computing broad directional rank features", None, "frames", workers, 10.0)
    max_pending = max(1, int(workers) * FRAME_FUTURE_MULTIPLIER)
    completed_frames = 0
    work_conn.execute("BEGIN")
    try:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures: set[Any] = set()
                for frame_records in v1.iter_stream_frames(source_stream, logger, 10.0):
                    if not frame_records:
                        continue
                    futures.add(
                        executor.submit(
                            worker_score_frame,
                            (str(cache_db), str(work_db), [record.__dict__ for record in frame_records], metric.tolist()),
                        )
                    )
                    while len(futures) >= max_pending:
                        futures, completed_frames = drain_frame_futures(futures, work_conn, scan_stats, progress, completed_frames)
                while futures:
                    futures, completed_frames = drain_frame_futures(
                        futures, work_conn, scan_stats, progress, completed_frames, force_all=True
                    )
        else:
            for frame_records in v1.iter_stream_frames(source_stream, logger, 10.0):
                result = worker_score_frame((str(cache_db), str(work_db), [record.__dict__ for record in frame_records], metric.tolist()))
                rows = result["feature_rows"]
                if rows:
                    work_conn.executemany(feature_insert_sql(), rows)
                merge_scan_stats(scan_stats, result["stats"])
                completed_frames += int(result["stats"].get("frames", 0))
                progress.update(completed_frames)
        work_conn.commit()
    except Exception:
        work_conn.rollback()
        raise
    progress.finish(completed_frames)

    if int(scan_stats["reflection_rows"]) <= 0:
        raise SystemExit("No reflection intensities were parsed from the source stream")
    if int(scan_stats["joined_rows"]) <= 0:
        raise SystemExit("No parsed stream intensities could be joined to full_population_cache.sqlite exact keys")
    if int(scan_stats["finite_residual_rows"]) <= 0:
        raise SystemExit("No local-neighbour J_resid values could be computed; no silent median fallback is allowed")
    feature_count = int(work_conn.execute("SELECT COUNT(*) FROM features").fetchone()[0])
    if feature_count <= 0:
        raise SystemExit("No finite rankable V2 observations were produced")
    scan_stats["feature_rows_unique"] = int(feature_count)
    logger.log("v2: creating rank indexes for aggressive global selection")
    for family, column in RANK_COLUMNS.items():
        work_conn.execute(f"CREATE INDEX idx_features_{family} ON features({column} DESC, egm2 DESC, exact_key_text)")
    work_conn.execute("CREATE INDEX idx_features_hkl_v2 ON features(h,k,l,exact_key_text)")
    work_conn.execute("CREATE INDEX idx_features_ordinal_v2 ON features(ordinal)")
    work_conn.commit()
    return scan_stats


def hkl_limits_from_work(conn: sqlite3.Connection) -> dict[tuple[int, int, int], int]:
    return {
        (int(h), int(k), int(l)): int(max_remove)
        for h, k, l, max_remove in conn.execute("SELECT h,k,l,max_remove FROM hkl_stats")
    }


def selected_insert_sql() -> str:
    return """
        INSERT OR IGNORE INTO selected(
            variant_id,base_variant_id,variant_role,control_type,random_seed,family,global_fraction,fraction_label,
            rank_formula,selection_rank,ordinal,exact_key_text,source_filename,event,h,k,l,intensity,Eg,D,M,M2,
            egm2,geom_delta,geom_delta_z,j_resid,local_neighbor_count,rank_score
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """


def selected_payload(variant: VariantSpec, selection_rank: int, feature_row: tuple[Any, ...], rank_score: float) -> tuple[Any, ...]:
    return (
        variant.variant_id,
        variant.base_variant_id,
        variant.variant_role,
        variant.control_type,
        None if variant.random_seed is None else int(variant.random_seed),
        variant.family,
        float(variant.fraction),
        variant.fraction_text,
        variant.rank_formula,
        int(selection_rank),
        *feature_row,
        float(rank_score),
    )


def select_family_targets(
    conn: sqlite3.Connection,
    family: str,
    variants: list[VariantSpec],
    scoreable_count: int,
    hkl_limits: dict[tuple[int, int, int], int],
    allow_incomplete: bool,
) -> list[dict[str, Any]]:
    if not variants:
        return []
    rank_column = RANK_COLUMNS[family]
    sorted_variants = sorted(variants, key=lambda item: item.fraction)
    requested = {variant.variant_id: int(math.floor(float(variant.fraction) * int(scoreable_count))) for variant in sorted_variants}
    max_requested = max(requested.values())
    selected_per_hkl: dict[tuple[int, int, int], int] = {}
    selected_count = 0
    batch: list[tuple[Any, ...]] = []
    query = f"""
        SELECT ordinal,exact_key_text,source_filename,event,h,k,l,intensity,Eg,D,M,M2,egm2,
               geom_delta,geom_delta_z,j_resid,local_neighbor_count,{rank_column} AS rank_score
        FROM features
        ORDER BY {rank_column} DESC, egm2 DESC, exact_key_text ASC
    """
    for row in conn.execute(query):
        if selected_count >= max_requested:
            break
        hkl = (int(row[4]), int(row[5]), int(row[6]))
        limit = int(hkl_limits.get(hkl, 0))
        if selected_per_hkl.get(hkl, 0) >= limit:
            continue
        selected_count += 1
        selected_per_hkl[hkl] = selected_per_hkl.get(hkl, 0) + 1
        feature_row = tuple(row[:-1])
        rank_score = float(row[-1])
        for variant in sorted_variants:
            if selected_count <= requested[variant.variant_id]:
                batch.append(selected_payload(variant, selected_count, feature_row, rank_score))
        if len(batch) >= MAX_INSERT_ROWS:
            conn.executemany(selected_insert_sql(), batch)
            batch = []
    if batch:
        conn.executemany(selected_insert_sql(), batch)

    out: list[dict[str, Any]] = []
    for variant in sorted_variants:
        actual = min(int(requested[variant.variant_id]), int(selected_count))
        fill_fraction = float(actual / max(1, int(requested[variant.variant_id])))
        if requested[variant.variant_id] > 0 and fill_fraction < MIN_FILL_FRACTION and not allow_incomplete:
            raise SystemExit(
                f"{variant.variant_id}: actual removals {actual:,} are only {100.0 * fill_fraction:.1f}% "
                f"of requested {requested[variant.variant_id]:,}; rerun with --allow-incomplete to keep this partial stress test"
            )
        out.append(
            {
                "variant_id": variant.variant_id,
                "base_variant_id": variant.base_variant_id,
                "variant_role": variant.variant_role,
                "control_type": variant.control_type,
                "random_seed": "",
                "family": variant.family,
                "rank_formula": variant.rank_formula,
                "global_fraction": float(variant.fraction),
                "fraction_label": variant.fraction_text,
                "requested_removed_count": int(requested[variant.variant_id]),
                "actual_removed_count": int(actual),
                "fill_fraction_of_request": fill_fraction,
            }
        )
    return out


def stable_u64(*items: Any) -> int:
    payload = "|".join(str(item) for item in items).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def target_hkl_counts(conn: sqlite3.Connection, variant_id: str) -> dict[tuple[int, int, int], int]:
    return {
        (int(h), int(k), int(l)): int(n)
        for h, k, l, n in conn.execute(
            "SELECT h,k,l,COUNT(*) FROM selected WHERE variant_id=? GROUP BY h,k,l",
            (variant_id,),
        )
    }


def fetch_features_for_hkl(conn: sqlite3.Connection, hkl: tuple[int, int, int], rank_column: str) -> list[tuple[Any, ...]]:
    h, k, l = hkl
    return list(
        conn.execute(
            f"""
            SELECT ordinal,exact_key_text,source_filename,event,h,k,l,intensity,Eg,D,M,M2,egm2,
                   geom_delta,geom_delta_z,j_resid,local_neighbor_count,{rank_column} AS rank_score
            FROM features
            WHERE h=? AND k=? AND l=?
            """,
            (int(h), int(k), int(l)),
        )
    )


def choose_control_rows(rows: list[tuple[Any, ...]], count: int, variant: VariantSpec, hkl: tuple[int, int, int]) -> list[tuple[int, tuple[Any, ...], float]]:
    if count <= 0:
        return []
    if len(rows) < count:
        raise SystemExit(f"{variant.variant_id}: hkl {hkl} has only {len(rows)} rankable rows for requested control count {count}")
    if variant.control_type == "low_rank_score":
        ordered = sorted(rows, key=lambda row: (float(row[-1]), str(row[1])))
    elif variant.control_type == "matched_random":
        ordered = sorted(rows, key=lambda row: (stable_u64(variant.random_seed, variant.base_variant_id, row[1]), str(row[1])))
    elif variant.control_type == "sign_mismatch_directional":
        def key(row: tuple[Any, ...]) -> tuple[int, float, str]:
            agreement = float(row[14]) * float(row[15])
            mismatch_rank = 0 if agreement <= 0.0 else 1
            directional_strength = abs(float(row[14])) * abs(float(row[15]))
            return (mismatch_rank, -directional_strength, str(row[1]))

        ordered = sorted(rows, key=key)
    else:
        raise SystemExit(f"Unsupported control type: {variant.control_type}")
    return [(idx + 1, tuple(row[:-1]), float(row[-1])) for idx, row in enumerate(ordered[:count])]


def build_control_variant(conn: sqlite3.Connection, variant: VariantSpec) -> dict[str, Any]:
    counts = target_hkl_counts(conn, variant.base_variant_id)
    if not counts:
        raise SystemExit(f"{variant.variant_id}: target variant has no selected rows to match")
    batch: list[tuple[Any, ...]] = []
    total = 0
    for hkl, count in sorted(counts.items()):
        rows = fetch_features_for_hkl(conn, hkl, variant.rank_column)
        chosen = choose_control_rows(rows, int(count), variant, hkl)
        for local_rank, feature_row, rank_score in chosen:
            total += 1
            batch.append(selected_payload(variant, local_rank, feature_row, rank_score))
        if len(batch) >= MAX_INSERT_ROWS:
            conn.executemany(selected_insert_sql(), batch)
            batch = []
    if batch:
        conn.executemany(selected_insert_sql(), batch)
    control_counts = target_hkl_counts(conn, variant.variant_id)
    if control_counts != counts:
        raise SystemExit(f"{variant.variant_id}: control per-HKL removal counts do not match {variant.base_variant_id}")
    requested = int(conn.execute("SELECT COUNT(*) FROM selected WHERE variant_id=?", (variant.base_variant_id,)).fetchone()[0])
    actual = int(conn.execute("SELECT COUNT(*) FROM selected WHERE variant_id=?", (variant.variant_id,)).fetchone()[0])
    return {
        "variant_id": variant.variant_id,
        "base_variant_id": variant.base_variant_id,
        "variant_role": variant.variant_role,
        "control_type": variant.control_type,
        "random_seed": "" if variant.random_seed is None else int(variant.random_seed),
        "family": variant.family,
        "rank_formula": variant.rank_formula,
        "global_fraction": float(variant.fraction),
        "fraction_label": variant.fraction_text,
        "requested_removed_count": int(requested),
        "actual_removed_count": int(actual),
        "fill_fraction_of_request": float(actual / max(1, requested)),
        "control_per_hkl_counts_match_target": True,
    }


def run_selection(
    work_conn: sqlite3.Connection,
    targeted: list[VariantSpec],
    control_variants: list[VariantSpec],
    scoreable_count: int,
    allow_incomplete: bool,
    logger: RunLogger,
) -> list[dict[str, Any]]:
    logger.log("v2: loading per-HKL safety caps")
    hkl_limits = hkl_limits_from_work(work_conn)
    rows: list[dict[str, Any]] = []
    work_conn.execute("BEGIN")
    try:
        progress = StageProgress(logger, "v2: selecting aggressive target removals", len(FAMILY_ORDER), "families", 1, 10.0)
        completed = 0
        for family in FAMILY_ORDER:
            family_variants = [variant for variant in targeted if variant.family == family]
            if family_variants:
                rows.extend(select_family_targets(work_conn, family, family_variants, scoreable_count, hkl_limits, allow_incomplete))
            completed += 1
            progress.update(completed)
        progress.finish(completed)
        if control_variants:
            control_progress = StageProgress(logger, "v2: building matched controls", len(control_variants), "controls", 1, 10.0)
            for idx, variant in enumerate(control_variants, start=1):
                rows.append(build_control_variant(work_conn, variant))
                control_progress.update(idx)
            control_progress.finish(len(control_variants))
        work_conn.commit()
    except Exception:
        work_conn.rollback()
        raise
    work_conn.execute("CREATE INDEX idx_selected_variant_v2 ON selected(variant_id, selection_rank)")
    work_conn.execute("CREATE INDEX idx_selected_ordinal_v2 ON selected(ordinal)")
    work_conn.execute("CREATE INDEX idx_selected_hkl_v2 ON selected(variant_id,h,k,l)")
    work_conn.commit()
    return rows


def selection_hkl_stats(conn: sqlite3.Connection, variant_id: str) -> dict[str, int]:
    rows = list(
        conn.execute(
            """
            SELECT s.h,s.k,s.l,COUNT(*) AS n_removed,hs.max_remove,hs.n_obs
            FROM selected AS s
            JOIN hkl_stats AS hs ON hs.h=s.h AND hs.k=s.k AND hs.l=s.l
            WHERE s.variant_id=?
            GROUP BY s.h,s.k,s.l
            """,
            (variant_id,),
        )
    )
    return {
        "selected_hkl_groups": int(len(rows)),
        "selected_hkl_groups_at_cap": int(sum(1 for _h, _k, _l, n, max_remove, _n_obs in rows if int(n) >= int(max_remove))),
        "max_per_hkl_removed": int(max([int(row[3]) for row in rows] or [0])),
        "safety_violations": int(sum(1 for _h, _k, _l, n, max_remove, n_obs in rows if int(n) > int(max_remove) or int(n_obs) - int(n) < MIN_RETAINED_PER_HKL)),
    }


def selected_mask_digests(conn: sqlite3.Connection, variants: list[VariantSpec], mask_bits: int) -> dict[str, str]:
    digests: dict[str, str] = {}
    for variant in variants:
        mask = PackedMask(mask_bits)
        ordinals = [int(row[0]) for row in conn.execute("SELECT ordinal FROM selected WHERE variant_id=?", (variant.variant_id,))]
        mask.set_many(ordinals)
        digests[variant.variant_id] = mask.digest()
    return digests


def write_removed_keys(conn: sqlite3.Connection, path: Path, logger: RunLogger) -> int:
    total = int(conn.execute("SELECT COUNT(*) FROM selected").fetchone()[0])
    progress = StageProgress(logger, "v2: writing removed exact-key audit", total, "rows", 1, 10.0)
    fieldnames = [
        "variant_id",
        "base_variant_id",
        "variant_role",
        "control_type",
        "random_seed",
        "family",
        "global_fraction",
        "fraction_label",
        "rank_formula",
        "selection_rank",
        "ordinal",
        "exact_key_text",
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "intensity",
        "Eg",
        "D",
        "M",
        "M2",
        "egm2",
        "geom_delta",
        "geom_delta_z",
        "j_resid",
        "local_neighbor_count",
        "rank_score",
    ]
    written = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fieldnames)
        for row in conn.execute(
            """
            SELECT variant_id,base_variant_id,variant_role,control_type,random_seed,family,global_fraction,
                   fraction_label,rank_formula,selection_rank,ordinal,exact_key_text,source_filename,event,h,k,l,
                   intensity,Eg,D,M,M2,egm2,geom_delta,geom_delta_z,j_resid,local_neighbor_count,rank_score
            FROM selected
            ORDER BY variant_id, selection_rank
            """
        ):
            writer.writerow(row)
            written += 1
            progress.update(written)
    progress.finish(written)
    return written


def build_masks_from_selected(
    conn: sqlite3.Connection,
    variants: list[VariantSpec],
    mask_bits: int,
    logger: RunLogger,
) -> dict[str, PackedMask]:
    masks = {variant.variant_id: PackedMask(mask_bits) for variant in variants}
    total = int(conn.execute("SELECT COUNT(*) FROM selected").fetchone()[0])
    progress = StageProgress(logger, "v2: building selected-row masks", total, "rows", 1, 10.0)
    completed = 0
    for variant_id, ordinal in conn.execute("SELECT variant_id, ordinal FROM selected ORDER BY variant_id"):
        masks[str(variant_id)].set_many([int(ordinal)])
        completed += 1
        progress.update(completed)
    progress.finish(completed)
    return masks


def make_count_rows(
    conn: sqlite3.Connection,
    variants: list[VariantSpec],
    selection_rows: list[dict[str, Any]],
    scoreable_count: int,
    source_rows: int | None,
    mask_digests: dict[str, str],
) -> list[dict[str, Any]]:
    by_id = {row["variant_id"]: row for row in selection_rows}
    out: list[dict[str, Any]] = []
    for variant in variants:
        row = dict(by_id.get(variant.variant_id, {}))
        actual = int(row.get("actual_removed_count", 0))
        hkl_stats = selection_hkl_stats(conn, variant.variant_id)
        out.append(
            {
                "variant_id": variant.variant_id,
                "base_variant_id": variant.base_variant_id,
                "variant_role": variant.variant_role,
                "control_type": variant.control_type,
                "random_seed": "" if variant.random_seed is None else int(variant.random_seed),
                "family": variant.family,
                "rank_formula": variant.rank_formula,
                "global_fraction": float(variant.fraction),
                "fraction_label": variant.fraction_text,
                "scoreable_population_count": int(scoreable_count),
                "source_reflection_row_count": "" if source_rows is None else int(source_rows),
                "requested_removed_count": int(row.get("requested_removed_count", 0)),
                "actual_removed_count": actual,
                "actual_removed_fraction_of_scoreable_population": float(actual / max(1, scoreable_count)),
                "actual_removed_fraction_of_source_rows": "" if source_rows is None else float(actual / max(1, source_rows)),
                "fill_fraction_of_request": float(row.get("fill_fraction_of_request", 0.0)),
                "max_remove_per_hkl_fraction": MAX_REMOVE_PER_HKL_FRACTION,
                "min_retained_per_hkl": MIN_RETAINED_PER_HKL,
                **hkl_stats,
                "mask_sha256": mask_digests.get(variant.variant_id, ""),
            }
        )
    return out


def make_manifest_rows(
    variants: list[VariantSpec],
    count_rows: list[dict[str, Any]],
    stream_qc: list[dict[str, Any]],
    write_streams: bool,
) -> list[dict[str, Any]]:
    counts = {row["variant_id"]: row for row in count_rows}
    qc = v1.stream_qc_by_variant(stream_qc)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        count = counts[variant.variant_id]
        qc_row = qc.get(variant.variant_id, {})
        rows.append(
            {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "stream_path": str(variant.output_stream),
                "rank_formula": variant.rank_formula,
                "global_fraction": float(variant.fraction),
                "requested_removed_count": int(count["requested_removed_count"]),
                "actual_removed_count": int(count["actual_removed_count"]),
                "actual_removed_fraction_of_scoreable_population": count["actual_removed_fraction_of_scoreable_population"],
                "actual_removed_fraction_of_source_rows": count["actual_removed_fraction_of_source_rows"],
                "max_remove_per_hkl_fraction": MAX_REMOVE_PER_HKL_FRACTION,
                "min_retained_per_hkl": MIN_RETAINED_PER_HKL,
                "stream_status": qc_row.get("status") or ("generated" if write_streams else "planned_dry_run"),
                "source_order_preserved": qc_row.get("source_order_preserved", True),
                "merge_status": "not_started",
                "base_variant_id": variant.base_variant_id,
                "variant_role": variant.variant_role,
                "control_type": variant.control_type,
                "random_seed": "" if variant.random_seed is None else int(variant.random_seed),
                "stream_qc_removed_observations": qc_row.get("stream_removed_count", ""),
                "stream_qc_kept_observations": qc_row.get("stream_kept_count", ""),
            }
        )
    return rows


def git_info(project_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    try:
        rev = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--short"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return out
    out.update({"available": bool(rev.stdout.strip()), "commit": rev.stdout.strip(), "status_short": status.stdout.splitlines()})
    return out


def write_outputs(
    args: argparse.Namespace,
    variants: list[VariantSpec],
    cache_db: Path,
    cache_columns: list[str],
    cache_stats: dict[str, int],
    source_rows: int | None,
    unit_cell: dict[str, float],
    metric: np.ndarray,
    hkl_stats: dict[str, Any],
    scan_stats: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    removed_keys_count: int,
    stream_qc: list[dict[str, Any]],
    mask_digests: dict[str, str],
    started_utc: str,
    logger: RunLogger,
) -> None:
    logger.log("v2: writing manifests and metadata")
    count_rows = make_count_rows(
        conn=args.work_conn,
        variants=variants,
        selection_rows=selection_rows,
        scoreable_count=int(cache_stats["count"]),
        source_rows=source_rows,
        mask_digests=mask_digests,
    )
    manifest_rows = make_manifest_rows(variants, count_rows, stream_qc, bool(args.write_streams))
    write_csv(args.output_dir / "directional_v2_counts.csv", count_rows)
    write_csv(args.output_dir / "directional_v2_manifest.tsv", manifest_rows, delimiter="\t")
    parameters = {
        "source_out_dir": str(args.source_out_dir),
        "source_stream": str(args.source_stream),
        "source_cache": str(cache_db),
        "output_dir": str(args.output_dir),
        "families": args.family_items,
        "fractions": [text for text, _fraction in args.fraction_items],
        "controls": args.control_items,
        "random_seeds": args.random_seed_items,
        "workers": int(args.workers),
        "dry_run": bool(args.dry_run),
        "write_streams": bool(args.write_streams),
        "allow_incomplete": bool(args.allow_incomplete),
        "overwrite": bool(args.overwrite),
        "score_base": {"score_id": "eg_m2", "formula": "Eg * M2"},
        "directional_geometry_proxy": "A=M/(D+eps); geom_delta=A-Eg; geom_delta_z robust within-HKL z-score",
        "local_intensity_proxy": "J=I*Eg; J_resid=(J_g-weighted_median_C(J_q))/(1.4826*weighted_MAD_C(J_q)+eps)",
        "kernel": {
            "formula": "C(g-q)=exp[-0.5*(dq/sigma_c)^2] for dq<=r_cut, otherwise 0",
            "sigma_c_A_inv": v1.SIGMA_C,
            "r_cut_A_inv": v1.R_CUT,
            "self_coupling_excluded": True,
        },
        "rank_formulas": RANK_FORMULAS,
        "filtering": {
            "requested_removed_count": "floor(global_fraction * full scoreable score_cache row count)",
            "eligibility": "finite EgM2, geom_delta_z, and local-neighbour J_resid; no hard directional thresholds",
            "safety": {
                "max_remove_per_hkl_fraction": MAX_REMOVE_PER_HKL_FRACTION,
                "min_retained_per_hkl": MIN_RETAINED_PER_HKL,
                "skip_safety_violations_and_continue_down_ranking": True,
                "minimum_fill_fraction_without_allow_incomplete": MIN_FILL_FRACTION,
            },
            "exact_key": "source_filename + event + signed h,k,l",
            "symmetry_canonicalization": False,
            "source_order": "preserved by stream-line rewriting",
        },
    }
    write_json(args.output_dir / "directional_v2_parameters.json", parameters)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": started_utc,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "script": str(Path(__file__).resolve()),
        "git": git_info(Path(__file__).resolve().parents[1]),
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "source_files": {
            "source_out_dir": str(args.source_out_dir),
            "source_cache": str(cache_db),
            "source_stream": str(args.source_stream),
            "source_parameters": str(args.source_out_dir / "parameters.json"),
            "source_validation": str(args.source_out_dir / "validation.json"),
        },
        "source_cache_columns": cache_columns,
        "cache_stats": cache_stats,
        "source_reflection_row_count": "" if source_rows is None else int(source_rows),
        "unit_cell_A_deg": unit_cell,
        "reciprocal_metric_A_inv_sq": metric,
        "hkl_stats": hkl_stats,
        "feature_scan_stats": scan_stats,
        "selection_stats": selection_rows,
        "removed_keys_count": int(removed_keys_count),
        "targeted_variant_count": sum(1 for variant in variants if variant.variant_role == "targeted"),
        "control_variant_count": sum(1 for variant in variants if variant.variant_role == "control"),
        "stream_qc": stream_qc,
        "validation": {
            "intensities_parsed": int(scan_stats.get("reflection_rows", 0)) > 0,
            "intensities_joined_to_cache": int(scan_stats.get("joined_rows", 0)) > 0,
            "local_residuals_computed": int(scan_stats.get("finite_residual_rows", 0)) > 0,
            "ranked_broad_population": int(scan_stats.get("feature_rows_unique", 0)) > 0,
            "source_order_preserved": all(bool(row.get("source_order_preserved", True)) for row in stream_qc) if stream_qc else True,
        },
        "outputs": {
            "manifest": str(args.output_dir / "directional_v2_manifest.tsv"),
            "counts": str(args.output_dir / "directional_v2_counts.csv"),
            "removed_keys": str(args.output_dir / "directional_v2_removed_keys.tsv.gz"),
            "parameters": str(args.output_dir / "directional_v2_parameters.json"),
            "metadata": str(args.output_dir / "directional_v2_metadata.json"),
            "run_log": str(args.output_dir / "run.log"),
        },
    }
    write_json(args.output_dir / "directional_v2_metadata.json", metadata)


def main() -> int:
    args = parse_args()
    targeted, controls = build_variants(
        args.family_items,
        args.fraction_items,
        args.control_items,
        args.random_seed_items,
        args.output_dir,
    )
    variants = targeted + controls
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_outputs(planned_output_paths(args.output_dir, variants, bool(args.write_streams)), bool(args.overwrite))

    logger = RunLogger(args.output_dir)
    started_utc = datetime.now(timezone.utc).isoformat()
    work_db = work_db_path(args.output_dir)
    work_conn: sqlite3.Connection | None = None
    try:
        logger.log("full-pop directional intensity V2 builder started")
        logger.log(f"source_out_dir={args.source_out_dir}")
        logger.log(f"source_stream={args.source_stream}")
        logger.log(f"output_dir={args.output_dir}")
        logger.log(f"mode={'write-streams' if args.write_streams else 'dry-run'}")
        logger.log(f"families={','.join(args.family_items)}")
        logger.log(f"fractions={','.join(text for text, _fraction in args.fraction_items)}")
        logger.log(f"controls={','.join(args.control_items) if args.control_items else 'none'}")
        logger.log(f"variants={len(variants)} ({len(targeted)} targeted, {len(controls)} controls)")
        logger.log(f"workers={int(args.workers)}")
        logger.log("aggressive V2 mode: targets are fractions of the full scoreable cache, not tiny thresholded pools")

        cache_db = cache_db_path(args.source_out_dir)
        if not cache_db.is_file():
            raise SystemExit(f"full-population cache not found: {cache_db}")
        cache_columns = require_cache_schema(cache_db)
        cache_stats = score_cache_stats(cache_db)
        if int(cache_stats["count"]) <= 0:
            raise SystemExit("score_cache is empty")
        source_rows = source_reflection_row_count(args.source_out_dir)
        if source_rows is not None and int(source_rows) < int(cache_stats["count"]):
            raise SystemExit(f"source reflection rows {source_rows:,} is smaller than scoreable cache count {cache_stats['count']:,}")

        unit_cell = v1.parse_unit_cell_from_stream(args.source_stream)
        metric = v1.reciprocal_metric_from_cell(unit_cell)
        logger.log(
            "kernel: gaussian C(g-q)=exp[-0.5*(dq/sigma_c)^2], "
            f"sigma_c={v1.SIGMA_C:.6g} A^-1, r_cut={v1.R_CUT:.6g} A^-1, q!=g"
        )

        work_conn = connect_work(work_db)
        args.work_conn = work_conn
        init_work_db(work_conn)
        hkl_stats = populate_hkl_stats(cache_db, work_conn, int(args.workers), logger)
        scan_stats = populate_features(cache_db, work_db, work_conn, args.source_stream, metric, int(args.workers), logger)
        selection_rows = run_selection(work_conn, targeted, controls, int(cache_stats["count"]), bool(args.allow_incomplete), logger)
        removed_keys_count = write_removed_keys(work_conn, args.output_dir / "directional_v2_removed_keys.tsv.gz", logger)
        mask_digests = selected_mask_digests(work_conn, variants, int(cache_stats["mask_bits"]))
        stream_qc: list[dict[str, Any]] = []
        if args.write_streams:
            logger.log("serial stream writing is intentional: it preserves source stream order exactly")
            masks = build_masks_from_selected(work_conn, variants, int(cache_stats["mask_bits"]), logger)
            stream_qc = v1.rewrite_streams(args.source_stream, cache_db, variants, masks, logger, source_rows, 10.0)
        else:
            logger.log("dry-run mode: stream files were not written")
        write_outputs(
            args,
            variants,
            cache_db,
            cache_columns,
            cache_stats,
            source_rows,
            unit_cell,
            metric,
            hkl_stats,
            scan_stats,
            selection_rows,
            removed_keys_count,
            stream_qc,
            mask_digests,
            started_utc,
            logger,
        )
        logger.log("full-pop directional intensity V2 builder complete")
        return 0
    finally:
        if work_conn is not None:
            work_conn.close()
        for sidecar in [work_db, Path(str(work_db) + "-wal"), Path(str(work_db) + "-shm")]:
            if sidecar.exists():
                sidecar.unlink()
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
