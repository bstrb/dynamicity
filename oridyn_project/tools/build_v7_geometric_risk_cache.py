#!/usr/bin/env python3
"""Build a V7 geometric observation-risk cache from existing V6 inputs.

V7 keeps the OriDyn risk idea observation-level, but changes the neighbour
candidate universe from "observed/accepted reflections in the event" to
"predicted local reciprocal-lattice neighbours from the crystal orientation".

Definition:

    S_risk(g) = E(g) * R_env(g)
    R_env(g) = sum_{h != g and d_gh < r_cut} E(h) * C(g,h)
    E(x) = exp[-(s_x / s0)^2]
    C(g,h) = exp[-(d_gh / sigma_C)^2]

where d_gh is the 3D reciprocal-space distance in A^-1, computed from the
stream crystal reciprocal basis, and s_x is the Ewald excitation-error proxy.

This script reads the existing source stream plus the V6 full-population cache.
It writes a new V7 SQLite cache and metadata only.  It does not run filtering,
stream generation, Partialator, merging, or refinement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oridyn.hkl_generation import allowed_by_centering


DATASET_ROOT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524"
)
DEFAULT_SOURCE_OUT_DIR = DATASET_ROOT / "oridyn_v6_full_population_sweep_20260717"
DEFAULT_OUT_DIR = DATASET_ROOT / "oridyn_v7_geometric_risk_20260902"

FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
WAVELENGTH_RE = re.compile(rf"^\s*wavelength\s*=\s*({FLOAT_RE})\s*A")
IMAGE_RE = re.compile(r"^\s*Image filename:\s*(.+?)\s*$")
EVENT_RE = re.compile(r"^\s*Event:\s*(\S+)")
FILENAME_RE = re.compile(r"^\s*Filename:\s*(.+?)(?:\s+//?(\S+))?\s*$")
CENTERING_RE = re.compile(r"^\s*centering\s*=\s*([A-Za-z])")
VECTOR_RE = re.compile(rf"^\s*([abc])star\s*=\s*({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+nm\^-1")

DEFAULT_S0 = 0.002306535801437913
DEFAULT_SIGMA_C = 0.050
DEFAULT_R_CUT = 0.150
DEFAULT_D_MIN = 0.3
DEFAULT_D_MAX = 20.0
DEFAULT_PROGRESS_EVERY = 500
DEFAULT_INSERT_BATCH = 50_000
CONTRIBUTION_FLOOR = 1.0e-12


@dataclass(frozen=True)
class CrystalRecord:
    source_filename: str
    event: str
    crystal_index: int
    wavelength_angstrom: float
    reciprocal_matrix: np.ndarray
    centering: str


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_source(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_event(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def key_to_text(source: Any, event: Any, h: int, k: int, l: int) -> str:
    return f"{normalize_source(source)}|{normalize_event(event)}|{int(h)}|{int(k)}|{int(l)}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def sqlite_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-out-dir", type=Path, default=DEFAULT_SOURCE_OUT_DIR)
    parser.add_argument("--source-stream", type=Path, default=None, help="Defaults to source_stream.path from V6 cache_provenance.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--s0", type=float, default=None, help="Excitation softening. Defaults to V6 provenance sg0.")
    parser.add_argument("--sigma-c", type=float, default=DEFAULT_SIGMA_C)
    parser.add_argument("--r-cut", type=float, default=DEFAULT_R_CUT)
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN, help="Minimum d-spacing for predicted candidate neighbours.")
    parser.add_argument("--d-max", type=float, default=DEFAULT_D_MAX, help="Maximum d-spacing for predicted candidate neighbours.")
    parser.add_argument("--max-crystals", type=int, default=None, help="Optional first-N crystal limit for smoke tests.")
    parser.add_argument("--max-score-rows", type=int, default=None, help="Optional scored-row limit for smoke tests.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Worker processes for per-crystal scoring.")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.self_test:
        return args

    args.source_out_dir = args.source_out_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.cache_db = args.source_out_dir / "full_population_cache.sqlite"
    args.provenance_file = args.source_out_dir / "cache_provenance.json"
    if not args.cache_db.is_file():
        raise SystemExit(f"V6 cache not found: {args.cache_db}")
    provenance = read_json(args.provenance_file)
    if args.source_stream is None:
        raw_stream = provenance.get("source_stream", {}).get("path")
        if not raw_stream:
            raise SystemExit(f"Could not find source_stream.path in {args.provenance_file}")
        args.source_stream = Path(raw_stream)
    args.source_stream = args.source_stream.expanduser().resolve()
    if not args.source_stream.is_file():
        raise SystemExit(f"Source stream not found: {args.source_stream}")
    if args.s0 is None:
        args.s0 = float(provenance.get("geometry_parameters", {}).get("sg0", DEFAULT_S0))
    if float(args.s0) <= 0.0:
        raise SystemExit("--s0 must be > 0")
    if float(args.sigma_c) <= 0.0:
        raise SystemExit("--sigma-c must be > 0")
    if float(args.r_cut) <= 0.0:
        raise SystemExit("--r-cut must be > 0")
    if float(args.d_min) <= 0.0 or float(args.d_max) <= 0.0 or float(args.d_min) > float(args.d_max):
        raise SystemExit("--d-min/--d-max must be positive with d-min <= d-max")
    if args.max_crystals is not None and int(args.max_crystals) < 1:
        raise SystemExit("--max-crystals must be >= 1")
    if args.max_score_rows is not None and int(args.max_score_rows) < 1:
        raise SystemExit("--max-score-rows must be >= 1")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1")
    args.workers = int(args.workers)
    args.progress_every = max(1, int(args.progress_every))
    return args


def iter_stream_crystals(stream_path: Path, max_crystals: int | None = None) -> Iterator[CrystalRecord]:
    wavelength: float | None = None
    centering = "P"
    chunk_source = ""
    chunk_event = ""
    current_source = ""
    current_event = ""
    in_chunk = False
    in_crystal = False
    vectors: dict[str, np.ndarray] = {}
    yielded = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if match := WAVELENGTH_RE.match(raw_line):
                wavelength = float(match.group(1))
                continue
            if match := CENTERING_RE.match(raw_line):
                centering = match.group(1).upper()
                continue
            if raw_line.startswith("----- Begin chunk -----"):
                in_chunk = True
                chunk_source = ""
                chunk_event = ""
                current_source = ""
                current_event = ""
                continue
            if raw_line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                continue
            if not in_chunk:
                continue
            if match := IMAGE_RE.match(raw_line):
                if in_crystal:
                    current_source = normalize_source(match.group(1))
                else:
                    chunk_source = normalize_source(match.group(1))
                continue
            if match := EVENT_RE.match(raw_line):
                if in_crystal:
                    current_event = normalize_event(match.group(1))
                else:
                    chunk_event = normalize_event(match.group(1))
                continue
            if match := FILENAME_RE.match(raw_line):
                parsed_source = normalize_source(match.group(1))
                parsed_event = normalize_event(match.group(2)) if match.group(2) is not None else ""
                if in_crystal:
                    current_source = parsed_source
                    if parsed_event:
                        current_event = parsed_event
                else:
                    chunk_source = parsed_source
                    if parsed_event:
                        chunk_event = parsed_event
                continue
            if raw_line.startswith("--- Begin crystal"):
                in_crystal = True
                vectors = {}
                current_source = chunk_source
                current_event = chunk_event
                continue
            if raw_line.startswith("--- End crystal"):
                if wavelength is None:
                    raise SystemExit("Could not parse wavelength from stream before first crystal")
                missing = [axis for axis in ("a", "b", "c") if axis not in vectors]
                if missing:
                    raise SystemExit(f"Crystal for {current_source}/{current_event} is missing reciprocal vector(s): {missing}")
                reciprocal = np.column_stack([vectors["a"], vectors["b"], vectors["c"]]) / 10.0
                yielded += 1
                yield CrystalRecord(
                    source_filename=current_source,
                    event=current_event,
                    crystal_index=yielded - 1,
                    wavelength_angstrom=float(wavelength),
                    reciprocal_matrix=reciprocal,
                    centering=centering,
                )
                if max_crystals is not None and yielded >= int(max_crystals):
                    return
                in_crystal = False
                continue
            if in_crystal and (match := VECTOR_RE.match(raw_line)):
                vectors[match.group(1)] = np.asarray(
                    [float(match.group(2)), float(match.group(3)), float(match.group(4))],
                    dtype=float,
                )


def fetch_accepted_rows(conn: sqlite3.Connection, source_filename: str, event: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT sc.ordinal,sc.source_filename,sc.event,sc.h,sc.k,sc.l,sc.exact_key_text,sc.source_order,
               sc.sg AS v6_sg,sc.Eg AS v6_Eg,sc.M2 AS v6_M2,sc.Eg*sc.M2 AS v6_S_risk
        FROM accepted AS a
        JOIN score_cache AS sc ON sc.exact_key_text=a.exact_key_text
        WHERE a.source_filename=? AND a.event=?
        ORDER BY a.source_order
        """,
        (normalize_source(source_filename), normalize_event(event)),
    ).fetchall()
    return [dict(row) for row in rows]


def excitation_error(g_vectors: np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    vectors = np.asarray(g_vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)
    k_norm = 1.0 / float(wavelength_angstrom)
    k_in = np.asarray([0.0, 0.0, k_norm], dtype=float)
    return np.linalg.norm(vectors + k_in[None, :], axis=1) - k_norm


def excitation_weight(s: np.ndarray | float, s0: float) -> np.ndarray:
    values = np.asarray(s, dtype=float)
    return np.exp(-np.square(values / max(float(s0), 1.0e-300)))


def offset_candidates(reciprocal: np.ndarray, r_cut: float) -> tuple[np.ndarray, np.ndarray]:
    singular = np.linalg.svd(np.asarray(reciprocal, dtype=float), compute_uv=False)
    min_singular = max(float(np.min(singular)), 1.0e-12)
    bound = int(math.ceil(float(r_cut) / min_singular)) + 1
    offsets: list[tuple[int, int, int]] = []
    vectors: list[np.ndarray] = []
    for dh in range(-bound, bound + 1):
        for dk in range(-bound, bound + 1):
            for dl in range(-bound, bound + 1):
                if dh == 0 and dk == 0 and dl == 0:
                    continue
                delta = np.asarray([dh, dk, dl], dtype=float)
                vec = reciprocal @ delta
                distance = float(np.linalg.norm(vec))
                if 0.0 < distance < float(r_cut):
                    offsets.append((dh, dk, dl))
                    vectors.append(vec)
    if not offsets:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=float)
    order = np.argsort(np.linalg.norm(np.vstack(vectors), axis=1), kind="mergesort")
    return np.asarray(offsets, dtype=np.int64)[order], np.vstack(vectors)[order]


def score_observation(
    row: dict[str, Any],
    reciprocal: np.ndarray,
    wavelength_angstrom: float,
    centering: str,
    offsets: np.ndarray,
    offset_vectors: np.ndarray,
    *,
    s0: float,
    sigma_c: float,
    d_min: float,
    d_max: float,
) -> dict[str, Any]:
    target_hkl = np.asarray([int(row["h"]), int(row["k"]), int(row["l"])], dtype=np.int64)
    target_g = reciprocal @ target_hkl.astype(float)
    s_g = float(excitation_error(target_g, wavelength_angstrom)[0])
    e_g = float(excitation_weight(s_g, s0))
    q_min = 1.0 / float(d_max)
    q_max = 1.0 / float(d_min)

    neighbour_hkls = target_hkl[None, :] + offsets
    if neighbour_hkls.size == 0:
        valid = np.zeros(0, dtype=bool)
    else:
        valid_centering = np.asarray(
            [
                allowed_by_centering(int(h), int(k), int(l), centering)
                and not (int(h) == 0 and int(k) == 0 and int(l) == 0)
                for h, k, l in neighbour_hkls
            ],
            dtype=bool,
        )
        neighbour_g = target_g[None, :] + offset_vectors
        q_norm = np.linalg.norm(neighbour_g, axis=1)
        valid = valid_centering & (q_norm >= q_min) & (q_norm <= q_max)

    selected_hkls = neighbour_hkls[valid]
    selected_vectors = offset_vectors[valid]
    distances = np.linalg.norm(selected_vectors, axis=1) if selected_vectors.size else np.zeros(0, dtype=float)
    neighbour_g = target_g[None, :] + selected_vectors if selected_vectors.size else np.zeros((0, 3), dtype=float)
    s_h = excitation_error(neighbour_g, wavelength_angstrom) if len(selected_hkls) else np.zeros(0, dtype=float)
    e_h = excitation_weight(s_h, s0) if len(selected_hkls) else np.zeros(0, dtype=float)
    c = np.exp(-np.square(distances / max(float(sigma_c), 1.0e-300))) if len(selected_hkls) else np.zeros(0, dtype=float)
    contributions = e_h * c
    r_env = float(np.sum(contributions))
    s_risk = e_g * r_env
    nonzero = int(np.sum(contributions > 0.0))
    active = int(np.sum(contributions > CONTRIBUTION_FLOOR))
    if len(contributions):
        top_idx = int(np.argmax(contributions))
        top_h, top_k, top_l = (int(x) for x in selected_hkls[top_idx])
        top_s = float(s_h[top_idx])
        top_e = float(e_h[top_idx])
        top_d = float(distances[top_idx])
        top_c = float(c[top_idx])
        top_contrib = float(contributions[top_idx])
    else:
        top_h = top_k = top_l = None
        top_s = top_e = top_d = top_c = top_contrib = None

    return {
        "ordinal": int(row["ordinal"]),
        "source_filename": str(row["source_filename"]),
        "event": str(row["event"]),
        "h": int(row["h"]),
        "k": int(row["k"]),
        "l": int(row["l"]),
        "exact_key_text": str(row["exact_key_text"]),
        "source_order": int(row["source_order"]),
        "s_g": s_g,
        "E_g": e_g,
        "R_env": r_env,
        "S_risk": s_risk,
        "geometric_neighbour_count": int(len(selected_hkls)),
        "nonzero_neighbour_count": nonzero,
        "active_neighbour_count": active,
        "top_neighbour_h": top_h,
        "top_neighbour_k": top_k,
        "top_neighbour_l": top_l,
        "top_neighbour_s_h": top_s,
        "top_neighbour_E_h": top_e,
        "top_neighbour_d_gh": top_d,
        "top_neighbour_C": top_c,
        "top_neighbour_contribution": top_contrib,
        "v6_sg": float(row["v6_sg"]),
        "v6_Eg": float(row["v6_Eg"]),
        "v6_M2": float(row["v6_M2"]),
        "v6_S_risk": float(row["v6_S_risk"]),
    }


def score_crystal_task(
    crystal: CrystalRecord,
    accepted_rows: list[dict[str, Any]],
    *,
    s0: float,
    sigma_c: float,
    r_cut: float,
    d_min: float,
    d_max: float,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    offsets, offset_vectors = offset_candidates(crystal.reciprocal_matrix, float(r_cut))
    scored = [
        score_observation(
            row,
            crystal.reciprocal_matrix,
            crystal.wavelength_angstrom,
            crystal.centering,
            offsets,
            offset_vectors,
            s0=float(s0),
            sigma_c=float(sigma_c),
            d_min=float(d_min),
            d_max=float(d_max),
        )
        for row in accepted_rows
    ]
    return int(crystal.crystal_index), os.getpid(), len(scored), scored


def prepare_output(args: argparse.Namespace) -> dict[str, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "db": args.out_dir / "v7_geometric_risk_cache.sqlite",
        "summary": args.out_dir / "v7_geometric_risk_summary.tsv",
        "metadata": args.out_dir / "v7_geometric_risk_metadata.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing V7 outputs; pass --overwrite if intentional:\n{joined}")
    for path in existing:
        path.unlink()
    return paths


def create_output_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE v7_score_cache (
            ordinal INTEGER PRIMARY KEY,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_order INTEGER NOT NULL,
            s_g REAL NOT NULL,
            E_g REAL NOT NULL,
            R_env REAL NOT NULL,
            S_risk REAL NOT NULL,
            geometric_neighbour_count INTEGER NOT NULL,
            nonzero_neighbour_count INTEGER NOT NULL,
            active_neighbour_count INTEGER NOT NULL,
            top_neighbour_h INTEGER,
            top_neighbour_k INTEGER,
            top_neighbour_l INTEGER,
            top_neighbour_s_h REAL,
            top_neighbour_E_h REAL,
            top_neighbour_d_gh REAL,
            top_neighbour_C REAL,
            top_neighbour_contribution REAL,
            v6_sg REAL NOT NULL,
            v6_Eg REAL NOT NULL,
            v6_M2 REAL NOT NULL,
            v6_S_risk REAL NOT NULL
        )
        """
    )
    return conn


INSERT_COLUMNS = [
    "ordinal",
    "source_filename",
    "event",
    "h",
    "k",
    "l",
    "exact_key_text",
    "source_order",
    "s_g",
    "E_g",
    "R_env",
    "S_risk",
    "geometric_neighbour_count",
    "nonzero_neighbour_count",
    "active_neighbour_count",
    "top_neighbour_h",
    "top_neighbour_k",
    "top_neighbour_l",
    "top_neighbour_s_h",
    "top_neighbour_E_h",
    "top_neighbour_d_gh",
    "top_neighbour_C",
    "top_neighbour_contribution",
    "v6_sg",
    "v6_Eg",
    "v6_M2",
    "v6_S_risk",
]


def insert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ",".join(["?"] * len(INSERT_COLUMNS))
    conn.executemany(
        f"INSERT INTO v7_score_cache({','.join(INSERT_COLUMNS)}) VALUES({placeholders})",
        [tuple(row.get(column) for column in INSERT_COLUMNS) for row in rows],
    )


def quantiles_from_db(conn: sqlite3.Connection, column: str) -> dict[str, float]:
    values = np.asarray([row[0] for row in conn.execute(f"SELECT {column} FROM v7_score_cache")], dtype=float)
    if values.size == 0:
        return {}
    qs = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    names = ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]
    return {name: float(value) for name, value in zip(names, qs)}


def write_summary(conn: sqlite3.Connection, path: Path) -> None:
    n = int(conn.execute("SELECT COUNT(*) FROM v7_score_cache").fetchone()[0])
    if n == 0:
        rows = [{"quantity": "n_observations", "value": "0"}]
    else:
        rows: list[dict[str, Any]] = []
        for column in ["E_g", "R_env", "S_risk", "geometric_neighbour_count", "active_neighbour_count", "v6_S_risk"]:
            stats = quantiles_from_db(conn, column)
            rows.append(
                {
                    "quantity": column,
                    "n_observations": n,
                    **{key: f"{value:.12g}" for key, value in stats.items()},
                    "zero_fraction": f"{conn.execute(f'SELECT COUNT(*) FROM v7_score_cache WHERE {column}=0').fetchone()[0] / max(n, 1):.12g}",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "quantity",
            "n_observations",
            "min",
            "p01",
            "p05",
            "p25",
            "median",
            "p75",
            "p95",
            "p99",
            "max",
            "zero_fraction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def git_status() -> dict[str, Any]:
    try:
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        return {
            "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "status_short": status.stdout.splitlines() if status.returncode == 0 else [],
        }
    except Exception as exc:  # pragma: no cover - metadata best effort
        return {"error": str(exc)}


def write_metadata(args: argparse.Namespace, paths: dict[str, Path], stats: dict[str, Any]) -> None:
    provenance = read_json(args.provenance_file)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "formula": {
            "S_risk": "E_g * R_env",
            "R_env": "sum_{h != g and d_gh < r_cut} E_h * C(g,h)",
            "E": "exp[-(s/s0)^2]",
            "C": "exp[-(d_gh/sigma_C)^2]",
            "candidate_universe": "predicted local reciprocal-lattice neighbours from each stream crystal orientation, not only observed/accepted neighbours",
            "exact_signed_hkls": True,
            "symmetry_canonicalization": False,
        },
        "parameters": {
            "s0": float(args.s0),
            "sigma_C": float(args.sigma_c),
            "r_cut": float(args.r_cut),
            "d_min": float(args.d_min),
            "d_max": float(args.d_max),
            "contribution_floor_for_active_count": CONTRIBUTION_FLOOR,
        },
        "inputs": {
            "source_out_dir": str(args.source_out_dir),
            "v6_cache": str(args.cache_db),
            "v6_provenance": str(args.provenance_file),
            "source_stream": str(args.source_stream),
            "v6_geometry_parameters": provenance.get("geometry_parameters", {}),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "run_stats": stats,
        "git": git_status(),
    }
    paths["metadata"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compute_exact_key_hash(conn: sqlite3.Connection) -> str:
    row_hash = hashlib.sha256()
    for (exact_key_text,) in conn.execute("SELECT exact_key_text FROM v7_score_cache ORDER BY source_order"):
        row_hash.update((str(exact_key_text) + "\n").encode("utf-8"))
    return row_hash.hexdigest()


def flush_insert_buffer(out_conn: sqlite3.Connection, insert_buffer: list[dict[str, Any]]) -> None:
    if len(insert_buffer) >= DEFAULT_INSERT_BATCH:
        insert_rows(out_conn, insert_buffer)
        out_conn.commit()
        insert_buffer.clear()


def report_cache_progress(
    started: float,
    *,
    crystals_seen: int,
    crystals_completed: int,
    crystals_with_rows: int,
    scored_rows: int,
    workers: int,
) -> None:
    elapsed = time.monotonic() - started
    seen_rate = crystals_seen / max(elapsed, 1.0e-9)
    completed_rate = crystals_completed / max(elapsed, 1.0e-9)
    row_rate = scored_rows / max(elapsed, 1.0e-9)
    log(
        f"scanned crystals={crystals_seen:,}, completed={crystals_completed:,}, "
        f"matched_crystals={crystals_with_rows:,}, scored_rows={scored_rows:,}, "
        f"scan_rate={seen_rate:,.1f} crystals/s, completion_rate={completed_rate:,.1f} crystals/s, "
        f"score_rate={row_rate:,.0f} rows/s, workers={workers}"
    )


def run_cache(args: argparse.Namespace) -> int:
    paths = prepare_output(args)
    source_conn = sqlite_readonly(args.cache_db)
    out_conn = create_output_db(paths["db"])
    insert_buffer: list[dict[str, Any]] = []
    started = time.monotonic()
    crystals_seen = 0
    crystals_completed = 0
    crystals_with_rows = 0
    scored_rows = 0
    submitted_score_rows = 0
    unmatched_crystals = 0
    worker_pids: set[int] = set()
    max_pending = max(1, int(args.workers) * 2)

    def consume_future(future: Future[tuple[int, int, int, list[dict[str, Any]]]]) -> None:
        nonlocal crystals_completed, scored_rows
        _crystal_index, pid, _row_count, scored = future.result()
        worker_pids.add(int(pid))
        crystals_completed += 1
        scored_rows += len(scored)
        insert_buffer.extend(scored)
        flush_insert_buffer(out_conn, insert_buffer)

    log(f"starting V7 scoring with workers={int(args.workers)}, max_pending={max_pending}")
    if int(args.workers) <= 1:
        for crystal in iter_stream_crystals(args.source_stream, args.max_crystals):
            crystals_seen += 1
            accepted_rows = fetch_accepted_rows(source_conn, crystal.source_filename, crystal.event)
            if not accepted_rows:
                unmatched_crystals += 1
            else:
                crystals_with_rows += 1
                if args.max_score_rows is not None:
                    remaining = int(args.max_score_rows) - submitted_score_rows
                    if remaining <= 0:
                        break
                    accepted_rows = accepted_rows[:remaining]
                _idx, pid, _row_count, scored = score_crystal_task(
                    crystal,
                    accepted_rows,
                    s0=float(args.s0),
                    sigma_c=float(args.sigma_c),
                    r_cut=float(args.r_cut),
                    d_min=float(args.d_min),
                    d_max=float(args.d_max),
                )
                worker_pids.add(int(pid))
                crystals_completed += 1
                submitted_score_rows += len(accepted_rows)
                scored_rows += len(scored)
                insert_buffer.extend(scored)
                flush_insert_buffer(out_conn, insert_buffer)
                if args.max_score_rows is not None and submitted_score_rows >= int(args.max_score_rows):
                    break

            if crystals_seen == 1 or crystals_seen % int(args.progress_every) == 0:
                report_cache_progress(
                    started,
                    crystals_seen=crystals_seen,
                    crystals_completed=crystals_completed,
                    crystals_with_rows=crystals_with_rows,
                    scored_rows=scored_rows,
                    workers=1,
                )
    else:
        pending: set[Future[tuple[int, int, int, list[dict[str, Any]]]]] = set()
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            for crystal in iter_stream_crystals(args.source_stream, args.max_crystals):
                crystals_seen += 1
                accepted_rows = fetch_accepted_rows(source_conn, crystal.source_filename, crystal.event)
                if not accepted_rows:
                    unmatched_crystals += 1
                else:
                    crystals_with_rows += 1
                    if args.max_score_rows is not None:
                        remaining = int(args.max_score_rows) - submitted_score_rows
                        if remaining <= 0:
                            break
                        accepted_rows = accepted_rows[:remaining]
                    submitted_score_rows += len(accepted_rows)
                    pending.add(
                        executor.submit(
                            score_crystal_task,
                            crystal,
                            accepted_rows,
                            s0=float(args.s0),
                            sigma_c=float(args.sigma_c),
                            r_cut=float(args.r_cut),
                            d_min=float(args.d_min),
                            d_max=float(args.d_max),
                        )
                    )
                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            consume_future(future)
                    if args.max_score_rows is not None and submitted_score_rows >= int(args.max_score_rows):
                        break

                if crystals_seen == 1 or crystals_seen % int(args.progress_every) == 0:
                    report_cache_progress(
                        started,
                        crystals_seen=crystals_seen,
                        crystals_completed=crystals_completed,
                        crystals_with_rows=crystals_with_rows,
                        scored_rows=scored_rows,
                        workers=int(args.workers),
                    )

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume_future(future)

    insert_rows(out_conn, insert_buffer)
    out_conn.commit()
    out_conn.execute("CREATE INDEX v7_score_cache_hkl_idx ON v7_score_cache(h,k,l,exact_key_text)")
    out_conn.execute("CREATE INDEX v7_score_cache_source_order_idx ON v7_score_cache(source_order)")
    out_conn.execute("CREATE INDEX v7_score_cache_score_idx ON v7_score_cache(S_risk)")
    out_conn.commit()
    write_summary(out_conn, paths["summary"])
    exact_key_hash = compute_exact_key_hash(out_conn)
    stats = {
        "crystals_seen": int(crystals_seen),
        "crystals_completed": int(crystals_completed),
        "crystals_with_accepted_rows": int(crystals_with_rows),
        "crystals_without_accepted_rows": int(unmatched_crystals),
        "scored_rows": int(scored_rows),
        "submitted_score_rows": int(submitted_score_rows),
        "max_crystals": args.max_crystals,
        "max_score_rows": args.max_score_rows,
        "requested_workers": int(args.workers),
        "worker_pids": sorted(worker_pids),
        "actual_worker_pid_count": len(worker_pids),
        "elapsed_seconds": float(time.monotonic() - started),
        "scored_exact_key_sha256": exact_key_hash,
    }
    write_metadata(args, paths, stats)
    source_conn.close()
    out_conn.close()
    log(f"wrote V7 cache rows={scored_rows:,} to {paths['db']}")
    return 0


def create_smoke_inputs(root: Path) -> tuple[Path, Path, Path]:
    source_out = root / "v6"
    source_out.mkdir(parents=True, exist_ok=True)
    stream = root / "tiny.stream"
    stream.write_text(
        "\n".join(
            [
                "wavelength = 0.025 A",
                "centering = P",
                "----- Begin chunk -----",
                "Filename: img.h5 //1",
                "--- Begin crystal",
                "Cell parameters 1.0 1.0 1.0 nm, 90.0 90.0 90.0 deg",
                "astar = 1.0 0.0 0.0 nm^-1",
                "bstar = 0.0 1.0 0.0 nm^-1",
                "cstar = 0.0 0.0 1.0 nm^-1",
                "Reflections measured after indexing",
                "  1 0 0 10.0 1.0",
                "  2 0 0 11.0 1.0",
                "End of reflections",
                "--- End crystal",
                "----- End chunk -----",
                "----- Begin chunk -----",
                "Filename: img.h5 //2",
                "--- Begin crystal",
                "Cell parameters 1.0 1.0 1.0 nm, 90.0 90.0 90.0 deg",
                "astar = 1.0 0.0 0.0 nm^-1",
                "bstar = 0.0 1.0 0.0 nm^-1",
                "cstar = 0.0 0.0 1.0 nm^-1",
                "Reflections measured after indexing",
                "  1 0 0 10.0 1.0",
                "  2 0 0 11.0 1.0",
                "End of reflections",
                "--- End crystal",
                "----- End chunk -----",
                "",
            ]
        ),
        encoding="utf-8",
    )
    db = source_out / "full_population_cache.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE accepted (
            ordinal INTEGER PRIMARY KEY,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_order INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX accepted_frame_idx ON accepted(source_filename,event)")
    conn.execute(
        """
        CREATE TABLE score_cache (
            ordinal INTEGER PRIMARY KEY,
            source_filename TEXT NOT NULL,
            event TEXT NOT NULL,
            h INTEGER NOT NULL,
            k INTEGER NOT NULL,
            l INTEGER NOT NULL,
            exact_key_text TEXT NOT NULL UNIQUE,
            source_order INTEGER NOT NULL,
            sg REAL NOT NULL,
            abs_sg REAL NOT NULL,
            Eg REAL NOT NULL,
            D REAL NOT NULL,
            U REAL NOT NULL,
            M REAL NOT NULL,
            M2 REAL NOT NULL
        )
        """
    )
    ordinal = 0
    for event in ("1", "2"):
        for hkl in [(1, 0, 0), (2, 0, 0)]:
            ordinal += 1
            h, k, l = hkl
            key = key_to_text("img.h5", event, h, k, l)
            conn.execute("INSERT INTO accepted VALUES(?,?,?,?,?,?,?,?)", (ordinal, "img.h5", event, h, k, l, key, ordinal))
            conn.execute(
                "INSERT INTO score_cache VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ordinal, "img.h5", event, h, k, l, key, ordinal, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            )
    conn.commit()
    conn.close()
    (source_out / "cache_provenance.json").write_text(
        json.dumps(
            {
                "source_stream": {"path": str(stream)},
                "geometry_parameters": {"sg0": 0.02, "sigma_c": 0.05, "r_cut": 0.15},
            }
        ),
        encoding="utf-8",
    )
    return source_out, stream, root / "out"


def run_self_test() -> int:
    smoke_parent = PROJECT_ROOT / ".codex_smoke"
    smoke_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="oridyn_v7_smoke_", dir=str(smoke_parent)))
    try:
        source_out, stream, out_dir = create_smoke_inputs(root)
        args = parse_args(
            [
                "--source-out-dir",
                str(source_out),
                "--source-stream",
                str(stream),
                "--out-dir",
                str(out_dir),
                "--s0",
                "0.02",
                "--sigma-c",
                "0.2",
                "--r-cut",
                "0.25",
                "--d-min",
                "1.0",
                "--d-max",
                "20.0",
                "--max-crystals",
                "2",
                "--workers",
                "2",
                "--overwrite",
            ]
        )
        rc = run_cache(args)
        conn = sqlite3.connect(out_dir / "v7_geometric_risk_cache.sqlite")
        n, positive = conn.execute("SELECT COUNT(*), SUM(CASE WHEN S_risk > 0 THEN 1 ELSE 0 END) FROM v7_score_cache").fetchone()
        conn.close()
        if n != 4 or int(positive) < 2:
            raise AssertionError(f"Unexpected self-test counts: n={n}, positive={positive}")
        print("self-test passed: V7 predicted-neighbour cache wrote positive geometric risk rows with workers=2")
        return rc
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if getattr(args, "self_test", False):
        return run_self_test()
    return run_cache(args)


if __name__ == "__main__":
    raise SystemExit(main())
