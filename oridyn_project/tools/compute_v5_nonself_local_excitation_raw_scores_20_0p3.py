#!/usr/bin/env python3
"""Compute v5 non-self local excitation environment raw scores for OriDyn 20-0.3 data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oridyn.coupling_exposure_v2 import estimate_reciprocal_metric  # noqa: E402


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]
DEFAULT_OUTPUT_CSV = "geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"
DEFAULT_SCORE_COLUMN = "nonself_local_excitation_raw"
DEFAULT_EDGE_WEIGHT = 0.1
DEFAULT_KERNEL = "gaussian"
DEFAULT_SIGMA_C = 0.05
DEFAULT_Q0 = 0.05
DEFAULT_R_CUT = 0.15
DEFAULT_CHUNKSIZE = 500_000
DEFAULT_TARGET_BATCH_SIZE = 256
DEFAULT_PROGRESS_EVERY_FRAMES = 1_000
DEFAULT_SG_UNITS = "A^-1"
PROFILE_RADIUS_RE = re.compile(r"profile_radius\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-zÅ0-9\^-]+)?")
SELECTED_HKLS = [(8, 4, 0), (9, 3, 0), (8, 6, 0), (7, 5, 0), (0, 4, 0), (6, 6, 2)]
V4_DIAGNOSTIC_COLUMNS = ["local_neighbor_sum_raw", "local_crowding_target_gated_raw"]
OUTPUT_COLUMNS = [
    *KEY_COLUMNS,
    "d_angstrom",
    "inv_nm",
    "sg_target",
    "target_excitation_Eg",
    "nonself_neighbor_sum_raw",
    "nonself_neighbor_count_effective",
    "nonself_neighbor_max_Eq",
    "nonself_neighbor_mean_Eq",
    "nonself_neighbor_weighted_mean_dq",
    "nonself_local_excitation_raw",
]


@dataclass(frozen=True)
class V5Params:
    sg0: float
    kernel: str
    sigma_c: float
    q0: float
    r_cut: float
    target_batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="20-0.3 CrystFEL stream used to parse profile_radius")
    parser.add_argument("--v4-scores", required=True, type=Path, help="Existing all-indexed v4 raw score CSV")
    parser.add_argument("--accepted-v4-scores", type=Path, default=None, help="Optional P1 iter1 accepted-only v4 score CSV for diagnostics")
    parser.add_argument("--outdir", required=True, type=Path, help="Independent v5 output directory")
    parser.add_argument("--profile-radius", type=float, default=None, help="Profile radius value used to derive sg0")
    parser.add_argument("--profile-radius-units", choices=["nm^-1", "A^-1"], default="nm^-1")
    parser.add_argument("--sg-units", choices=["A^-1", "nm^-1"], default=DEFAULT_SG_UNITS, help="Units of sg_target values in --v4-scores")
    parser.add_argument("--edge-weight", type=float, default=DEFAULT_EDGE_WEIGHT)
    parser.add_argument("--sg0-override", type=float, default=None, help="Use this sg0 directly instead of profile-radius-derived sg0")
    parser.add_argument("--kernel", choices=["gaussian", "screened_inv2"], default=DEFAULT_KERNEL)
    parser.add_argument("--sigma-c", type=float, default=DEFAULT_SIGMA_C, help="Gaussian coupling width in A^-1")
    parser.add_argument("--q0", type=float, default=DEFAULT_Q0, help="Screened inverse-square coupling scale in A^-1")
    parser.add_argument("--r-cut", type=float, default=DEFAULT_R_CUT, help="Finite coupling cutoff in A^-1")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--target-batch-size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--progress-every-frames", type=int, default=DEFAULT_PROGRESS_EVERY_FRAMES)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test row cap; may truncate the final frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test frame cap")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.stream.is_file():
        raise SystemExit(f"--stream must be a file: {args.stream}")
    if not args.v4_scores.is_file():
        raise SystemExit(f"--v4-scores must be a CSV file: {args.v4_scores}")
    if args.accepted_v4_scores is not None and not args.accepted_v4_scores.is_file():
        raise SystemExit(f"--accepted-v4-scores must be a CSV file: {args.accepted_v4_scores}")
    if args.profile_radius is not None and float(args.profile_radius) <= 0.0:
        raise SystemExit("--profile-radius must be > 0")
    if not (0.0 < float(args.edge_weight) < 1.0):
        raise SystemExit("--edge-weight must satisfy 0 < edge_weight < 1")
    if args.sg0_override is not None and float(args.sg0_override) <= 0.0:
        raise SystemExit("--sg0-override must be > 0")
    if float(args.sigma_c) <= 0.0:
        raise SystemExit("--sigma-c must be > 0")
    if float(args.q0) <= 0.0:
        raise SystemExit("--q0 must be > 0")
    if float(args.r_cut) <= 0.0:
        raise SystemExit("--r-cut must be > 0")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if int(args.target_batch_size) < 1:
        raise SystemExit("--target-batch-size must be >= 1")
    if int(args.progress_every_frames) < 1:
        raise SystemExit("--progress-every-frames must be >= 1")
    if args.max_rows is not None and int(args.max_rows) < 1:
        raise SystemExit("--max-rows must be >= 1 when supplied")
    if args.max_frames is not None and int(args.max_frames) < 1:
        raise SystemExit("--max-frames must be >= 1 when supplied")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def normalize_source(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_event(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    while text.startswith("//"):
        text = text[2:]
    return text


def read_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def require_columns(header: list[str], required: list[str], label: str) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def output_paths(outdir: Path) -> dict[str, Path]:
    csv_path = outdir / DEFAULT_OUTPUT_CSV
    return {
        "csv": csv_path,
        "metadata": outdir / "geometry_coupling_v5_nonself_local_excitation_raw_metadata.json",
        "summary": outdir / "geometry_coupling_v5_nonself_local_excitation_raw_summary.md",
        "selected_hkl_csv": outdir / "selected_hkl_v5_nonself_local_excitation_diagnostics.csv",
        "selected_hkl_md": outdir / "selected_hkl_v5_nonself_local_excitation_diagnostics.md",
        "plot_all_png": outdir / "plots" / "v5_nonself_local_excitation_raw_all_indexed_hist.png",
        "plot_all_pdf": outdir / "plots" / "v5_nonself_local_excitation_raw_all_indexed_hist.pdf",
        "plot_accepted_png": outdir / "plots" / "v5_nonself_local_excitation_raw_p1_iter1_accepted_hist.png",
        "plot_accepted_pdf": outdir / "plots" / "v5_nonself_local_excitation_raw_p1_iter1_accepted_hist.pdf",
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    blocked = [path for path in paths.values() if path.exists()]
    if blocked and not overwrite:
        formatted = "\n".join(f"  {path}" for path in blocked)
        raise SystemExit(f"Refusing to overwrite existing output file(s):\n{formatted}\nUse --overwrite if intended.")
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["plot_all_png"].parent.mkdir(parents=True, exist_ok=True)


def parse_profile_radius_from_stream(path: Path) -> tuple[float | None, str | None, int | None]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = PROFILE_RADIUS_RE.search(line)
            if match:
                units = match.group(2) or "nm^-1"
                units = units.replace("Å", "A")
                if units not in {"nm^-1", "A^-1"}:
                    units = "nm^-1"
                return float(match.group(1)), units, int(line_number)
    return None, None, None


def profile_radius_to_units(value: float, source_units: str, target_units: str) -> float:
    if source_units == target_units:
        return float(value)
    if source_units == "nm^-1" and target_units == "A^-1":
        return 0.1 * float(value)
    if source_units == "A^-1" and target_units == "nm^-1":
        return 10.0 * float(value)
    raise ValueError(f"Unsupported profile-radius conversion {source_units} -> {target_units}")


def resolve_profile_and_sg0(args: argparse.Namespace) -> dict[str, Any]:
    parsed_value, parsed_units, parsed_line = parse_profile_radius_from_stream(args.stream)
    if args.profile_radius is not None:
        profile_value = float(args.profile_radius)
        profile_units = str(args.profile_radius_units)
        profile_source = "cli"
    else:
        profile_value = parsed_value
        profile_units = parsed_units
        profile_source = "stream"

    profile_radius_nm_inv = None
    profile_radius_A_inv = None
    sg0_profile_derived = None
    if profile_value is not None and profile_units is not None:
        profile_radius_nm_inv = profile_radius_to_units(float(profile_value), profile_units, "nm^-1")
        profile_radius_A_inv = profile_radius_to_units(float(profile_value), profile_units, "A^-1")
        radius_in_sg_units = profile_radius_to_units(float(profile_value), profile_units, str(args.sg_units))
        sg0_profile_derived = radius_in_sg_units / math.sqrt(-math.log(float(args.edge_weight)))

    if args.sg0_override is None:
        if sg0_profile_derived is None:
            raise SystemExit(
                "Could not parse profile_radius from --stream and no --profile-radius or --sg0-override was supplied. "
                "Provide --profile-radius or --sg0-override; v5 does not silently default sg0 to 0.01."
            )
        sg0_used = float(sg0_profile_derived)
        sg0_source = "profile_radius"
    else:
        sg0_used = float(args.sg0_override)
        sg0_source = "override"

    return {
        "profile_radius_source": profile_source if profile_value is not None else "not_available",
        "profile_radius_stream_value": parsed_value,
        "profile_radius_stream_units": parsed_units,
        "profile_radius_stream_line": parsed_line,
        "profile_radius_nm_inv": None if profile_radius_nm_inv is None else float(profile_radius_nm_inv),
        "profile_radius_A_inv": None if profile_radius_A_inv is None else float(profile_radius_A_inv),
        "sg_units_detected_or_assumed": f"{args.sg_units} assumed from OriDyn/CrystFEL reciprocal-matrix convention",
        "edge_weight": float(args.edge_weight),
        "sg0_profile_derived": None if sg0_profile_derived is None else float(sg0_profile_derived),
        "sg0_override": None if args.sg0_override is None else float(args.sg0_override),
        "sg0_used": float(sg0_used),
        "sg0_source": sg0_source,
    }


def normalize_v4_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    out["source_filename"] = out["source_filename"].map(normalize_source)
    out["event"] = out["event"].map(normalize_event)
    for column in HKL_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ["d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad_key = out[HKL_COLUMNS].isna().any(axis=1)
    out = out.loc[~bad_key].copy()
    if out.empty:
        return out
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def limited_csv_chunks(path: Path, usecols: list[str], chunksize: int, max_rows: int | None) -> Iterable[pd.DataFrame]:
    remaining = None if max_rows is None else int(max_rows)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= int(len(chunk))
        if chunk.empty:
            break
        yield chunk
        if remaining is not None and remaining <= 0:
            break


def frame_key(table: pd.DataFrame) -> pd.Series:
    return table["source_filename"].astype(str) + "\0" + table["event"].astype(str)


def iter_frame_groups(
    path: Path,
    usecols: list[str],
    chunksize: int,
    max_rows: int | None,
    max_frames: int | None,
) -> Iterable[pd.DataFrame]:
    carry = pd.DataFrame(columns=usecols)
    yielded = 0
    for chunk in limited_csv_chunks(path, usecols, chunksize, max_rows):
        work = normalize_v4_chunk(chunk)
        if not carry.empty:
            work = pd.concat([carry, work], ignore_index=True)
        if work.empty:
            carry = work
            continue
        keys = frame_key(work).to_numpy(dtype=object)
        last_key = keys[-1]
        complete = work.loc[keys != last_key].copy()
        carry = work.loc[keys == last_key].copy()
        if complete.empty:
            continue
        for _key, group in complete.groupby(frame_key(complete), sort=False):
            yield group.reset_index(drop=True)
            yielded += 1
            if max_frames is not None and yielded >= int(max_frames):
                return
    if not carry.empty and (max_frames is None or yielded < int(max_frames)):
        yield carry.reset_index(drop=True)


def excitation_weight_from_sg(sg: np.ndarray, sg0: float) -> np.ndarray:
    values = np.asarray(sg, dtype=float)
    scale = max(float(sg0), 1e-300)
    weights = np.exp(-((np.abs(values) / scale) ** 2))
    return np.where(np.isfinite(weights), weights, 0.0)


def coupling_kernel(dq: np.ndarray, params: V5Params) -> np.ndarray:
    values = np.asarray(dq, dtype=float)
    if params.kernel == "gaussian":
        kernel = np.exp(-0.5 * (values / max(float(params.sigma_c), 1e-300)) ** 2)
    elif params.kernel == "screened_inv2":
        q0_sq = max(float(params.q0), 1e-300) ** 2
        kernel = q0_sq / (values**2 + q0_sq)
    else:
        raise ValueError(f"Unsupported kernel: {params.kernel}")
    kernel = np.where(values <= float(params.r_cut), kernel, 0.0)
    return np.where(np.isfinite(kernel), kernel, 0.0)


def dq_from_delta(delta_hkl: np.ndarray, metric: np.ndarray | None) -> np.ndarray:
    delta = delta_hkl.astype(float)
    if metric is None:
        return np.linalg.norm(delta, axis=2)
    squared = np.einsum("...i,ij,...j->...", delta, metric, delta)
    return np.sqrt(np.clip(squared, 0.0, None))


def score_one_frame(group: pd.DataFrame, params: V5Params) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = group.reset_index(drop=True).copy()
    n = len(work)
    out = work.loc[:, [*KEY_COLUMNS, "d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"]].copy()
    for column in [
        "nonself_neighbor_sum_raw",
        "nonself_neighbor_count_effective",
        "nonself_neighbor_max_Eq",
        "nonself_neighbor_mean_Eq",
        "nonself_neighbor_weighted_mean_dq",
        "nonself_local_excitation_raw",
    ]:
        out[column] = 0.0
    out["nonself_neighbor_weighted_mean_dq"] = np.nan

    source = str(work["source_filename"].iloc[0]) if n else ""
    event = str(work["event"].iloc[0]) if n else ""
    stats: dict[str, Any] = {"source_filename": source, "event": event, "n_observations": int(n)}
    if n <= 1:
        stats.update({"reciprocal_metric_method": "empty_or_singleton_frame", "neighbors_with_nonzero_score": 0})
        return out, stats

    hkls = work.loc[:, HKL_COLUMNS].to_numpy(dtype=np.int64)
    inv_nm = pd.to_numeric(work["inv_nm"], errors="coerce").to_numpy(dtype=float)
    d_values = pd.to_numeric(work["d_angstrom"], errors="coerce").to_numpy(dtype=float)
    q_invA = np.divide(inv_nm, 10.0, out=np.full_like(inv_nm, np.nan, dtype=float), where=np.isfinite(inv_nm))
    missing_q = ~np.isfinite(q_invA)
    q_invA = np.where(missing_q & np.isfinite(d_values) & (d_values > 0.0), 1.0 / d_values, q_invA)
    metric, metric_stats = estimate_reciprocal_metric(hkls, q_invA)
    stats.update(metric_stats)

    sg = pd.to_numeric(work["sg_target"], errors="coerce").to_numpy(dtype=float)
    source_Eq = excitation_weight_from_sg(sg, params.sg0)
    raw = np.zeros(n, dtype=float)
    count_effective = np.zeros(n, dtype=float)
    max_Eq = np.full(n, np.nan, dtype=float)
    mean_Eq = np.full(n, np.nan, dtype=float)
    weighted_mean_dq = np.full(n, np.nan, dtype=float)

    for start in range(0, n, int(params.target_batch_size)):
        stop = min(start + int(params.target_batch_size), n)
        target_hkl = hkls[start:stop]
        delta = hkls[None, :, :] - target_hkl[:, None, :]
        nonself = np.any(delta != 0, axis=2)
        dq = dq_from_delta(delta, metric)
        kernel = coupling_kernel(dq, params)
        contributing = nonself & (kernel > 0.0) & np.isfinite(source_Eq[None, :])
        edge = np.where(contributing, kernel * source_Eq[None, :], 0.0)

        raw_batch = np.sum(edge, axis=1)
        raw[start:stop] = raw_batch
        count_effective[start:stop] = np.sum(np.where(contributing, kernel, 0.0), axis=1)
        neighbor_counts = np.sum(contributing, axis=1)
        if np.any(neighbor_counts):
            eq_matrix = np.where(contributing, source_Eq[None, :], 0.0)
            max_values = np.max(np.where(contributing, source_Eq[None, :], -np.inf), axis=1)
            max_Eq[start:stop] = np.where(neighbor_counts > 0, max_values, np.nan)
            mean_Eq[start:stop] = np.where(neighbor_counts > 0, np.sum(eq_matrix, axis=1) / np.maximum(neighbor_counts, 1), np.nan)
        numerator = np.sum(edge * dq, axis=1)
        weighted_mean_dq[start:stop] = np.divide(numerator, raw_batch, out=np.full_like(raw_batch, np.nan), where=raw_batch > 0.0)

    out["nonself_neighbor_sum_raw"] = raw
    out["nonself_neighbor_count_effective"] = count_effective
    out["nonself_neighbor_max_Eq"] = max_Eq
    out["nonself_neighbor_mean_Eq"] = mean_Eq
    out["nonself_neighbor_weighted_mean_dq"] = weighted_mean_dq
    out["nonself_local_excitation_raw"] = raw
    stats["neighbors_with_nonzero_score"] = int(np.sum(raw > 0.0))
    stats["raw_score_sum"] = float(np.sum(raw))
    return out, stats


def finite_stats(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if not values:
        return {"n_finite": 0, "min": None, "q10": None, "median": None, "q90": None, "max": None}
    data = np.concatenate(values).astype(float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"n_finite": 0, "min": None, "q10": None, "median": None, "q90": None, "max": None}
    q10, q50, q90 = np.quantile(data, [0.10, 0.50, 0.90])
    return {"n_finite": int(len(data)), "min": float(np.min(data)), "q10": float(q10), "median": float(q50), "q90": float(q90), "max": float(np.max(data))}


def compute_scores(args: argparse.Namespace, paths: dict[str, Path], sg0_meta: dict[str, Any]) -> dict[str, Any]:
    header = read_header(args.v4_scores)
    require_columns(header, [*KEY_COLUMNS, "d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"], "v4 scores")
    usecols = [*KEY_COLUMNS, "d_angstrom", "inv_nm", "sg_target", "target_excitation_Eg"]
    params = V5Params(
        sg0=float(sg0_meta["sg0_used"]),
        kernel=str(args.kernel),
        sigma_c=float(args.sigma_c),
        q0=float(args.q0),
        r_cut=float(args.r_cut),
        target_batch_size=int(args.target_batch_size),
    )
    tmp_csv = paths["csv"].with_suffix(paths["csv"].suffix + ".tmp")
    if tmp_csv.exists():
        tmp_csv.unlink()

    written = 0
    frames = 0
    distribution_values: dict[str, list[np.ndarray]] = {column: [] for column in ["target_excitation_Eg", DEFAULT_SCORE_COLUMN, "nonself_neighbor_count_effective", "nonself_neighbor_weighted_mean_dq"]}
    metric_methods: dict[str, int] = {}
    metric_projected = 0
    singleton_frames = 0
    zero_score_rows = 0
    frame_stats_examples: list[dict[str, Any]] = []

    try:
        for group in iter_frame_groups(args.v4_scores, usecols, int(args.chunksize), args.max_rows, args.max_frames):
            scored, frame_stats = score_one_frame(group, params)
            scored.loc[:, OUTPUT_COLUMNS].to_csv(tmp_csv, index=False, mode="w" if written == 0 else "a", header=written == 0)
            written += int(len(scored))
            frames += 1
            method = str(frame_stats.get("reciprocal_metric_method", "unknown"))
            metric_methods[method] = metric_methods.get(method, 0) + 1
            metric_projected += int(bool(frame_stats.get("reciprocal_metric_projected_to_psd")))
            singleton_frames += int(len(scored) <= 1)
            zero_score_rows += int((pd.to_numeric(scored[DEFAULT_SCORE_COLUMN], errors="coerce") <= 0.0).sum())
            if len(frame_stats_examples) < 5:
                frame_stats_examples.append(frame_stats)
            for column in distribution_values:
                values = pd.to_numeric(scored[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                if len(values):
                    distribution_values[column].append(values)
            if frames == 1 or frames % int(args.progress_every_frames) == 0:
                log(f"V5 score pass: frames={frames:,}, rows_written={written:,}")
    except Exception:
        if tmp_csv.exists():
            tmp_csv.unlink()
        raise

    if written == 0:
        raise SystemExit("No rows were scored from --v4-scores")
    tmp_csv.replace(paths["csv"])
    return {
        "rows_written": int(written),
        "frames_scored": int(frames),
        "singleton_frames": int(singleton_frames),
        "zero_score_rows": int(zero_score_rows),
        "metric_methods": metric_methods,
        "metric_projected_frames": int(metric_projected),
        "frame_stats_examples": frame_stats_examples,
        "output_columns": OUTPUT_COLUMNS,
        "distributions": {column: finite_stats(values) for column, values in distribution_values.items()},
    }


def collect_hkl_rows(path: Path, columns: list[str], hkls: list[tuple[int, int, int]], chunksize: int) -> pd.DataFrame:
    selected = set(hkls)
    rows = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=int(chunksize)):
        work = chunk.copy()
        for column in HKL_COLUMNS:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.loc[~work[HKL_COLUMNS].isna().any(axis=1)].copy()
        if work.empty:
            continue
        work[HKL_COLUMNS] = work[HKL_COLUMNS].astype("int64")
        mask = [(int(h), int(k), int(l)) in selected for h, k, l in work.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
        sub = work.loc[mask].copy()
        if not sub.empty:
            if "source_filename" in sub:
                sub["source_filename"] = sub["source_filename"].map(normalize_source)
            if "event" in sub:
                sub["event"] = sub["event"].map(normalize_event)
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def quantile_record(values: pd.Series, prefix: str) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return {f"{prefix}_{name}": None for name in ["min", "q10", "q50", "q90", "max"]}
    q10, q50, q90 = np.quantile(finite, [0.10, 0.50, 0.90])
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q90": float(q90),
        f"{prefix}_max": float(np.max(finite)),
    }


def pearson_or_nan(x: pd.Series, y: pd.Series) -> float | None:
    a = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    b = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 2:
        return None
    return float(np.corrcoef(a.loc[mask].to_numpy(dtype=float), b.loc[mask].to_numpy(dtype=float))[0, 1])


def write_selected_hkl_diagnostics(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    v5_cols = [*KEY_COLUMNS, DEFAULT_SCORE_COLUMN]
    old_v4_cols = [*KEY_COLUMNS, *V4_DIAGNOSTIC_COLUMNS]
    v5 = collect_hkl_rows(paths["csv"], v5_cols, SELECTED_HKLS, int(args.chunksize))
    old = collect_hkl_rows(args.v4_scores, old_v4_cols, SELECTED_HKLS, int(args.chunksize))
    joined = v5.merge(old, on=KEY_COLUMNS, how="left", validate="one_to_one") if not v5.empty else v5

    accepted_keys = pd.DataFrame(columns=KEY_COLUMNS)
    if args.accepted_v4_scores is not None:
        accepted_keys = collect_hkl_rows(args.accepted_v4_scores, KEY_COLUMNS, SELECTED_HKLS, int(args.chunksize)).drop_duplicates(KEY_COLUMNS, keep="first")
    if not accepted_keys.empty and not joined.empty:
        joined = joined.merge(accepted_keys.assign(_p1_iter1_accepted=1), on=KEY_COLUMNS, how="left", validate="one_to_one")
        joined["_p1_iter1_accepted"] = joined["_p1_iter1_accepted"].fillna(0).astype("int8")
    else:
        joined["_p1_iter1_accepted"] = 0

    records = []
    for h, k, l in SELECTED_HKLS:
        sub = joined.loc[(joined["h"] == h) & (joined["k"] == k) & (joined["l"] == l)].copy()
        record: dict[str, Any] = {"h": h, "k": k, "l": l, "n_obs_all_indexed": int(len(sub)), "n_obs_p1_iter1_accepted": int(sub["_p1_iter1_accepted"].sum()) if "_p1_iter1_accepted" in sub else 0}
        record.update(quantile_record(sub.get(DEFAULT_SCORE_COLUMN, pd.Series(dtype=float)), "v5_nonself"))
        record.update(quantile_record(sub.get("local_neighbor_sum_raw", pd.Series(dtype=float)), "old_v4_local_neighbor_sum_raw"))
        record.update(quantile_record(sub.get("local_crowding_target_gated_raw", pd.Series(dtype=float)), "old_v4_target_gated_raw"))
        record["corr_v5_nonself_vs_old_v4_target_gated_raw"] = pearson_or_nan(sub.get(DEFAULT_SCORE_COLUMN, pd.Series(dtype=float)), sub.get("local_crowding_target_gated_raw", pd.Series(dtype=float)))
        records.append(record)
    table = pd.DataFrame.from_records(records)
    table.to_csv(paths["selected_hkl_csv"], index=False)

    lines = [
        "# V5 Selected-HKL Non-Self Local Excitation Diagnostics",
        "",
        f"- v5 scores: `{paths['csv']}`",
        f"- old v4 scores: `{args.v4_scores}`",
        f"- accepted-only v4 scores: `{args.accepted_v4_scores}`" if args.accepted_v4_scores is not None else "- accepted-only v4 scores: none",
        "- exact key: `source_filename + normalized event + signed h,k,l`",
        "",
        "| h | k | l | n all | n accepted | v5 q10 | v5 q50 | v5 q90 | old v4 target q50 | corr v5 vs old target |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples(index=False):
        corr = "nan" if pd.isna(row.corr_v5_nonself_vs_old_v4_target_gated_raw) else f"{float(row.corr_v5_nonself_vs_old_v4_target_gated_raw):.6g}"
        old_q50 = getattr(row, "old_v4_target_gated_raw_q50")
        lines.append(
            f"| {int(row.h)} | {int(row.k)} | {int(row.l)} | {int(row.n_obs_all_indexed)} | {int(row.n_obs_p1_iter1_accepted)} | "
            f"{float(row.v5_nonself_q10):.6g} | {float(row.v5_nonself_q50):.6g} | {float(row.v5_nonself_q90):.6g} | "
            f"{float(old_q50):.6g} | {corr} |"
        )
    paths["selected_hkl_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"selected_hkl_rows": int(len(table)), "selected_hkl_csv": str(paths["selected_hkl_csv"]), "selected_hkl_md": str(paths["selected_hkl_md"])}


def collect_score_values(path: Path, score_column: str, chunksize: int) -> np.ndarray:
    values = []
    for chunk in pd.read_csv(path, usecols=[score_column], chunksize=int(chunksize)):
        arr = pd.to_numeric(chunk[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(arr):
            values.append(arr)
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def load_accepted_keys(path: Path, chunksize: int) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, usecols=KEY_COLUMNS, chunksize=int(chunksize)):
        work = chunk.copy()
        work["source_filename"] = work["source_filename"].map(normalize_source)
        work["event"] = work["event"].map(normalize_event)
        for column in HKL_COLUMNS:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.loc[~work[HKL_COLUMNS].isna().any(axis=1)].copy()
        if not work.empty:
            work[HKL_COLUMNS] = work[HKL_COLUMNS].astype("int64")
            chunks.append(work)
    if not chunks:
        return pd.DataFrame(columns=KEY_COLUMNS)
    return pd.concat(chunks, ignore_index=True).drop_duplicates(KEY_COLUMNS, keep="first")


def collect_accepted_score_values(v5_path: Path, accepted_path: Path, score_column: str, chunksize: int) -> np.ndarray:
    keys = load_accepted_keys(accepted_path, chunksize)
    values = []
    usecols = [*KEY_COLUMNS, score_column]
    for chunk in pd.read_csv(v5_path, usecols=usecols, chunksize=int(chunksize)):
        work = chunk.copy()
        work["source_filename"] = work["source_filename"].map(normalize_source)
        work["event"] = work["event"].map(normalize_event)
        for column in HKL_COLUMNS:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.loc[~work[HKL_COLUMNS].isna().any(axis=1)].copy()
        if work.empty:
            continue
        work[HKL_COLUMNS] = work[HKL_COLUMNS].astype("int64")
        joined = work.merge(keys, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        if not joined.empty:
            arr = pd.to_numeric(joined[score_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(arr):
                values.append(arr)
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def plot_histogram(values: np.ndarray, title: str, xlabel: str, png: Path, pdf: Path) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(finite):
        q01, q99 = np.quantile(finite, [0.01, 0.99])
        plot_values = finite[(finite >= q01) & (finite <= q99)] if q99 > q01 else finite
        ax.hist(plot_values, bins=80, color="#315f8c", alpha=0.86)
        q10, q50, q90 = np.quantile(finite, [0.10, 0.50, 0.90])
        for value, color, label in [(q10, "#777777", "q10"), (q50, "#d1495b", "q50"), (q90, "#777777", "q90")]:
            ax.axvline(float(value), color=color, linewidth=1.2, linestyle="-" if label == "q50" else "--", label=label)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("observations")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return finite_stats([finite])


def write_plots(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    all_values = collect_score_values(paths["csv"], DEFAULT_SCORE_COLUMN, int(args.chunksize))
    plot_stats = {
        "all_indexed": plot_histogram(all_values, "V5 non-self local excitation raw scores: all indexed", DEFAULT_SCORE_COLUMN, paths["plot_all_png"], paths["plot_all_pdf"])
    }
    if args.accepted_v4_scores is not None:
        accepted_values = collect_accepted_score_values(paths["csv"], args.accepted_v4_scores, DEFAULT_SCORE_COLUMN, int(args.chunksize))
        plot_stats["p1_iter1_accepted"] = plot_histogram(
            accepted_values,
            "V5 non-self local excitation raw scores: P1 iter1 accepted",
            DEFAULT_SCORE_COLUMN,
            paths["plot_accepted_png"],
            paths["plot_accepted_pdf"],
        )
    return plot_stats


def write_metadata(args: argparse.Namespace, paths: dict[str, Path], sg0_meta: dict[str, Any], score_stats: dict[str, Any], diagnostics: dict[str, Any], plot_stats: dict[str, Any]) -> None:
    payload = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "v4_scores": str(args.v4_scores),
            "accepted_v4_scores": None if args.accepted_v4_scores is None else str(args.accepted_v4_scores),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "profile_radius_and_sg0": sg0_meta,
        "kernel": {
            "kernel": str(args.kernel),
            "dq_units": "A^-1",
            "sigma_c": float(args.sigma_c),
            "q0": float(args.q0),
            "r_cut": float(args.r_cut),
        },
        "parameters": {
            "chunksize": int(args.chunksize),
            "target_batch_size": int(args.target_batch_size),
            "max_rows": None if args.max_rows is None else int(args.max_rows),
            "max_frames": None if args.max_frames is None else int(args.max_frames),
        },
        "score_definition": {
            "nonself_local_excitation_raw": "sum_{q != g} exp(-(s_q/sg0)^2) * C(|g-q|)",
            "target_excitation_gating": "not used",
            "normalization": "not used",
            "log_transform": "not used",
            "symmetry_canonicalization": "not used; signed HKLs preserved",
        },
        "score_stats": score_stats,
        "diagnostics": diagnostics,
        "plot_stats": plot_stats,
    }
    paths["metadata"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(args: argparse.Namespace, paths: dict[str, Path], sg0_meta: dict[str, Any], score_stats: dict[str, Any], diagnostics: dict[str, Any], plot_stats: dict[str, Any]) -> None:
    dist = score_stats["distributions"].get(DEFAULT_SCORE_COLUMN, {})
    lines = [
        "# V5 Non-Self Local Excitation Raw Score Summary",
        "",
        "This is a score-only diagnostic. It does not filter streams, run partialator, merge, normalize, shell-normalize, or log-transform the primary score.",
        "",
        "## Inputs",
        f"- stream: `{args.stream}`",
        f"- all-indexed v4 score table: `{args.v4_scores}`",
        f"- accepted-only v4 score table: `{args.accepted_v4_scores}`" if args.accepted_v4_scores is not None else "- accepted-only v4 score table: none",
        "",
        "## Definition",
        "- `nonself_local_excitation_raw = sum_{q != g} E(q) * C(g-q)`.",
        "- `E(q) = exp[-(s_q / sg0)^2]` uses the neighboring/source reflection excitation error only.",
        "- `E(g)` is not multiplied into the primary score.",
        "- Target-excitation gating, v2/v3/v4 composite scores, normalization, shell normalization, and log transforms are not used.",
        "- Exact signed HKLs are preserved; no 4/mmm canonicalization is applied.",
        "",
        "## Profile Radius and sg0",
        f"- profile_radius_nm_inv: {sg0_meta['profile_radius_nm_inv']}",
        f"- profile_radius_A_inv: {sg0_meta['profile_radius_A_inv']}",
        f"- sg_units_detected_or_assumed: {sg0_meta['sg_units_detected_or_assumed']}",
        f"- edge_weight: {sg0_meta['edge_weight']}",
        f"- sg0_profile_derived: {sg0_meta['sg0_profile_derived']}",
        f"- sg0_override: {sg0_meta['sg0_override']}",
        f"- sg0_used: {sg0_meta['sg0_used']}",
        f"- sg0_source: {sg0_meta['sg0_source']}",
        "",
        "## Kernel",
        f"- kernel: `{args.kernel}`",
        f"- sigma_c: {float(args.sigma_c):.8g} A^-1",
        f"- q0: {float(args.q0):.8g} A^-1",
        f"- r_cut: {float(args.r_cut):.8g} A^-1",
        "",
        "## Rows",
        f"- frames scored: {score_stats['frames_scored']:,}",
        f"- output rows: {score_stats['rows_written']:,}",
        f"- zero-score rows: {score_stats['zero_score_rows']:,}",
        "",
        "## Distribution",
        f"- `{DEFAULT_SCORE_COLUMN}`: n={dist.get('n_finite')}, min={dist.get('min')}, q10={dist.get('q10')}, median={dist.get('median')}, q90={dist.get('q90')}, max={dist.get('max')}",
        "",
        "## Outputs",
        f"- score CSV: `{paths['csv']}`",
        f"- metadata JSON: `{paths['metadata']}`",
        f"- selected-HKL diagnostic CSV: `{paths['selected_hkl_csv']}`",
        f"- selected-HKL diagnostic markdown: `{paths['selected_hkl_md']}`",
        f"- all-indexed plot PNG: `{paths['plot_all_png']}`",
        f"- accepted-only plot PNG: `{paths['plot_accepted_png']}`" if args.accepted_v4_scores is not None else "- accepted-only plot PNG: not generated",
    ]
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = output_paths(args.outdir)
    ensure_outputs(paths, bool(args.overwrite))
    sg0_meta = resolve_profile_and_sg0(args)
    log(f"profile_radius_nm_inv: {sg0_meta['profile_radius_nm_inv']}")
    log(f"profile_radius_A_inv: {sg0_meta['profile_radius_A_inv']}")
    log(f"edge_weight: {float(args.edge_weight):.6g}")
    log(f"sg0_used: {float(sg0_meta['sg0_used']):.8g} ({sg0_meta['sg0_source']})")
    log(f"kernel: {args.kernel}; sigma_c={float(args.sigma_c):.6g} A^-1; q0={float(args.q0):.6g} A^-1; r_cut={float(args.r_cut):.6g} A^-1")
    log(f"output CSV: {paths['csv']}")
    score_stats = compute_scores(args, paths, sg0_meta)
    log("Writing selected-HKL diagnostics")
    diagnostics = write_selected_hkl_diagnostics(args, paths)
    log("Writing distribution plots")
    plot_stats = write_plots(args, paths)
    write_metadata(args, paths, sg0_meta, score_stats, diagnostics, plot_stats)
    write_summary(args, paths, sg0_meta, score_stats, diagnostics, plot_stats)
    print("V5 non-self local excitation raw scores written")
    print(f"output_csv: {paths['csv']}")
    print(f"metadata_json: {paths['metadata']}")
    print(f"summary_md: {paths['summary']}")
    print(f"selected_hkl_csv: {paths['selected_hkl_csv']}")
    print(f"rows_written: {score_stats['rows_written']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())