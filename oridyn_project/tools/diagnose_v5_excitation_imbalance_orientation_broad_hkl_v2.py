#!/usr/bin/env python3
"""Broad HKL diagnostic for v5 excitation-imbalance/orientation tendencies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # noqa: SIM105
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover - scipy is optional for this diagnostic
    scipy_stats = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_same_hkl_low_high_orientation_examples import (  # noqa: E402
    KEY_COLUMNS,
    HKL_COLUMNS,
    load_stream_orientation_for_selected,
    normalize_event,
    normalize_key_columns,
    normalize_source,
)
from oridyn.axis_prediction import unique_zone_axes  # noqa: E402
from oridyn.geometry import (  # noqa: E402
    axis_angle_deg,
    beam_in_direct_coordinates,
    hkl_lab_vectors,
    reciprocal_matrix_from_cell,
    triplet_label,
)
from oridyn.stream_parser import (  # noqa: E402
    STREAM_MATRIX_COLUMNS,
    STREAM_UNITCELL_ANGLE_RE,
    STREAM_UNITCELL_LENGTH_RE,
    UnitCell,
    reciprocal_matrix_from_row,
)


BASE = Path("/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524")
DEFAULT_STREAM = BASE / "MFM300-VIII_cut_20-0_3.stream"
DEFAULT_V5_SCORES = (
    BASE
    / "oridyn_v5_nonself_local_excitation_raw_20_0p3_20260705"
    / "geometry_coupling_v5_nonself_local_excitation_raw_scores.csv"
)
DEFAULT_ACCEPTED = (
    BASE
    / "oridyn_v4_local_crowding_raw_20_0p3_20260704"
    / "partialator_survivor_mask"
    / "p1_iter1_20260705T1214"
    / "v4_p1_iter1_partialator_survivors_only_scores.csv"
)
DEFAULT_STRENGTH_TABLE = BASE / "MFM300-VIII_cut_20-0_3_partialator_results_20260705T1214" / "crystfel.hkl"
DEFAULT_OUT_DIR = BASE / "oridyn_v5_excitation_imbalance_orientation_broad_hkl_v2_20_0p3_20260712"

DEFAULT_SCORE_COLUMN = "nonself_local_excitation_raw"
DEFAULT_COUPLING_COLUMN = "nonself_neighbor_count_effective"
DEFAULT_TARGET_COLUMN = "target_excitation_Eg"
DEFAULT_SG_COLUMN = "sg_target"
DEFAULT_CHUNKSIZE = 500_000
DEFAULT_SEED = 1
PRIMARY_HKL_TARGET = 75
RESOLUTION_BIN_TARGET = 15
STRENGTH_PER_BIN_TARGET = 5
ANGLE_PER_BIN_TARGET = 5
STRENGTH_CLASSES = ("weak", "medium", "strong")
ANGLE_TERTILE_LABELS = ("cstar_aligned_angle_tertile", "intermediate_angle_tertile", "basal_aligned_angle_tertile")
BASE_CELL_TARGETS = {
    "weak": {"cstar_aligned_angle_tertile": 2, "intermediate_angle_tertile": 2, "basal_aligned_angle_tertile": 1},
    "medium": {"cstar_aligned_angle_tertile": 2, "intermediate_angle_tertile": 1, "basal_aligned_angle_tertile": 2},
    "strong": {"cstar_aligned_angle_tertile": 1, "intermediate_angle_tertile": 2, "basal_aligned_angle_tertile": 2},
}
GEOMETRY_QUOTAS = {
    "min_fully_general": 25,
    "max_axial": 7,
    "max_00l": 8,
    "max_one_index_zero": 20,
    "min_hk0": 12,
    "min_diagonal_special": 12,
}

RESPONSE_COLUMNS = [
    "I_over_merged",
    "I_over_hkl_median",
    "I_robust_z_within_hkl",
    "corrected_intensity_residual",
]
TREND_X_COLUMNS = [
    "excitation_deficit_norm",
    "coupled_excitation_imbalance_raw",
    "nonself_local_excitation_raw",
]
PRIMARY_SLOPE_COLUMN = "slope_corrected_residual_vs_excitation_deficit_norm"


@dataclass(frozen=True)
class IntensityChoice:
    column: str
    sigma_column: str | None
    scale_columns: tuple[str, ...]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--v5-scores", type=Path, default=DEFAULT_V5_SCORES)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--strength-table", type=Path, default=DEFAULT_STRENGTH_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--coupling-column", default=DEFAULT_COUPLING_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--sg-column", default=DEFAULT_SG_COLUMN)
    parser.add_argument("--min-accepted-obs", type=int, default=80)
    parser.add_argument("--min-nonzero-coupling-obs", type=int, default=30)
    parser.add_argument("--min-deficit-spread", type=float, default=1e-4)
    parser.add_argument("--min-target-spread", type=float, default=1e-4)
    parser.add_argument("--target-primary-hkls", type=int, default=PRIMARY_HKL_TARGET)
    parser.add_argument("--symmetry-pairs", type=int, default=10)
    parser.add_argument("--resolution-bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--uvw-max", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()

    for label, path in [
        ("--stream", args.stream),
        ("--v5-scores", args.v5_scores),
        ("--accepted", args.accepted),
        ("--strength-table", args.strength_table),
    ]:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    if int(args.min_accepted_obs) < 1:
        raise SystemExit("--min-accepted-obs must be >= 1")
    if int(args.min_nonzero_coupling_obs) < 1:
        raise SystemExit("--min-nonzero-coupling-obs must be >= 1")
    if int(args.target_primary_hkls) != PRIMARY_HKL_TARGET:
        raise SystemExit(f"--target-primary-hkls must be exactly {PRIMARY_HKL_TARGET} for the v2 hard-quota selector")
    if int(args.symmetry_pairs) < 0:
        raise SystemExit("--symmetry-pairs must be >= 0")
    if int(args.resolution_bins) != 5:
        raise SystemExit("--resolution-bins must be exactly 5 for the v2 hard-quota selector")
    if int(args.uvw_max) < 1:
        raise SystemExit("--uvw-max must be >= 1")
    if int(args.bootstrap_iterations) < 0:
        raise SystemExit("--bootstrap-iterations must be >= 0")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if float(args.min_deficit_spread) < 0.0 or float(args.min_target_spread) < 0.0:
        raise SystemExit("--min-deficit-spread and --min-target-spread must be >= 0")
    return args


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def require_columns(header: list[str], required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"{label} missing required column(s): {missing}")


def hkl_slug(h: int, k: int, l: int) -> str:
    def one(value: int) -> str:
        return f"m{abs(int(value))}" if int(value) < 0 else str(int(value))

    return f"hkl_{one(h)}_{one(k)}_{one(l)}"


def hkl_tuple_from_row(row: Any) -> tuple[int, int, int]:
    if isinstance(row, pd.Series):
        return int(row["h"]), int(row["k"]), int(row["l"])
    return int(row.h), int(row.k), int(row.l)


def hkl_mask(table: pd.DataFrame, hkls: list[tuple[int, int, int]]) -> np.ndarray:
    wanted = pd.MultiIndex.from_tuples(hkls, names=HKL_COLUMNS)
    return pd.MultiIndex.from_frame(table.loc[:, HKL_COLUMNS]).isin(wanted)


def parse_stream_unit_cell(path: Path) -> UnitCell:
    lengths: dict[str, float] = {}
    angles: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if raw.startswith("----- Begin chunk -----"):
                break
            if match := STREAM_UNITCELL_LENGTH_RE.match(raw):
                lengths[match.group(1)] = float(match.group(2))
                continue
            if match := STREAM_UNITCELL_ANGLE_RE.match(raw):
                angles[match.group(1)] = float(match.group(2))
                continue
    missing = [key for key in ["a", "b", "c"] if key not in lengths] + [key for key in ["al", "be", "ga"] if key not in angles]
    if missing:
        raise SystemExit(f"Could not parse complete global unit cell from stream header; missing {missing}")
    return UnitCell(
        a=float(lengths["a"]),
        b=float(lengths["b"]),
        c=float(lengths["c"]),
        alpha=float(angles["al"]),
        beta=float(angles["be"]),
        gamma=float(angles["ga"]),
    )


def read_crystfel_hkl_strength(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    negative = 0
    zero = 0
    positive = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("CrystFEL") or line.startswith("Symmetry:"):
                continue
            if line.startswith("End of reflections"):
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                h = int(parts[0])
                k = int(parts[1])
                l = int(parts[2])
                intensity = float(parts[3])
                if parts[4] == "-":
                    sigma = float(parts[5])
                    nmeas = int(parts[6]) if len(parts) > 6 else np.nan
                else:
                    sigma = float(parts[4])
                    nmeas = int(parts[5])
            except ValueError:
                continue
            negative += int(intensity < 0.0)
            zero += int(intensity == 0.0)
            positive += int(intensity > 0.0)
            rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "merged_intensity_or_Fobs": intensity,
                    "merged_sigma": sigma,
                    "merged_nmeas": nmeas,
                    "strength_source_column": "CrystFEL_I",
                }
            )
    if not rows:
        raise SystemExit(f"No reflection rows could be parsed from strength table: {path}")
    metadata = {
        "strength_source": str(path),
        "strength_metric": "signed full-data merged CrystFEL intensity I",
        "negative_merged_intensity_hkls": int(negative),
        "zero_merged_intensity_hkls": int(zero),
        "positive_merged_intensity_hkls": int(positive),
        "negative_intensity_handling": (
            "Negative merged intensities are retained and participate in signed within-resolution strength percentiles; "
            "they naturally fall into weak tails when their signed intensity is low."
        ),
    }
    return pd.DataFrame.from_records(rows).drop_duplicates(HKL_COLUMNS, keep="first"), metadata


def choose_intensity_columns(header: list[str]) -> IntensityChoice:
    intensity_candidates = [
        "I_scaled",
        "I_corrected",
        "I_pr",
        "I_partiality_corrected",
        "I_unmerged",
    ]
    chosen = next((column for column in intensity_candidates if column in header), None)
    if chosen is None:
        raise SystemExit(f"Accepted table has no usable observation intensity column. Tried {intensity_candidates}")
    sigma_column = next((column for column in ["sigma", "sigma_unmerged", "sigma_I"] if column in header), None)
    scale_columns = tuple(column for column in ["frame_scale", "scale", "scale_factor", "partialator_scale"] if column in header)
    if chosen == "I_unmerged":
        note = "No corrected/scaled observation intensity column was found; using I_unmerged from the accepted survivor table."
    else:
        note = f"Using preferred corrected/scaled observation intensity column {chosen}."
    return IntensityChoice(chosen, sigma_column, scale_columns, note)


def load_accepted_table(path: Path, chunksize: int) -> tuple[pd.DataFrame, IntensityChoice, dict[str, Any]]:
    header = read_header(path)
    require_columns(header, KEY_COLUMNS, "accepted survivor table")
    choice = choose_intensity_columns(header)
    optional = ["partiality", "I_unmerged", choice.column, *choice.scale_columns]
    if choice.sigma_column is not None:
        optional.append(choice.sigma_column)
    optional = list(dict.fromkeys(column for column in optional if column in header))
    usecols = [*KEY_COLUMNS, *optional]
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)), start=1):
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        work["observation_intensity"] = pd.to_numeric(work[choice.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work["I_unmerged"] = (
            pd.to_numeric(work["I_unmerged"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if "I_unmerged" in work.columns
            else np.nan
        )
        work["sigma"] = (
            pd.to_numeric(work[choice.sigma_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if choice.sigma_column is not None
            else np.nan
        )
        work["partiality"] = (
            pd.to_numeric(work["partiality"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if "partiality" in work.columns
            else np.nan
        )
        for scale_column in choice.scale_columns:
            work[scale_column] = pd.to_numeric(work[scale_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        chunks.append(work.loc[:, [*KEY_COLUMNS, "observation_intensity", "I_unmerged", "sigma", "partiality", *choice.scale_columns]].copy())
        if idx == 1 or idx % 10 == 0:
            log(f"Accepted survivor scan: chunks={idx:,}, rows_read={rows_read:,}")
    if not chunks:
        raise SystemExit("Accepted survivor table yielded no usable rows")
    table = pd.concat(chunks, ignore_index=True)
    dup = table.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(dup.sum())
    duplicate_keys = int(table.loc[dup, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact keys in accepted survivor table; keeping first")
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    metadata = {
        "rows_read": int(rows_read),
        "rows_after_key_cleanup": int(len(table)),
        "duplicate_rows": duplicate_rows,
        "duplicate_keys": duplicate_keys,
        "intensity_column_used": choice.column,
        "sigma_column_used": choice.sigma_column,
        "scale_columns_available": list(choice.scale_columns),
        "intensity_choice_note": choice.note,
    }
    return table, choice, metadata


def load_accepted_v5_join(accepted: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = read_header(args.v5_scores)
    required = [*KEY_COLUMNS, args.score_column, args.coupling_column, args.target_column, args.sg_column]
    require_columns(header, required, "v5 score CSV")
    usecols = required
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_after_cleanup = 0
    rows_matched = 0
    accepted_payload = accepted.loc[
        :,
        [
            *KEY_COLUMNS,
            "observation_intensity",
            "I_unmerged",
            "sigma",
            "partiality",
            *[c for c in accepted.columns if c not in [*KEY_COLUMNS, "observation_intensity", "I_unmerged", "sigma", "partiality"]],
        ],
    ].copy()
    for idx, chunk in enumerate(pd.read_csv(args.v5_scores, usecols=usecols, chunksize=int(args.chunksize)), start=1):
        rows_read += int(len(chunk))
        work = normalize_key_columns(chunk)
        for column in [args.score_column, args.coupling_column, args.target_column, args.sg_column]:
            work[column] = pd.to_numeric(work[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=[args.score_column, args.coupling_column, args.target_column])
        rows_after_cleanup += int(len(work))
        matched = work.merge(accepted_payload, on=KEY_COLUMNS, how="inner", sort=False, validate="many_to_one")
        rows_matched += int(len(matched))
        if not matched.empty:
            chunks.append(matched)
        if idx == 1 or idx % 5 == 0:
            log(f"V5/accepted join scan: chunks={idx:,}, v5_rows={rows_read:,}, accepted_matches={rows_matched:,}")
    if not chunks:
        raise SystemExit("No accepted observations matched the v5 score table")
    table = pd.concat(chunks, ignore_index=True)
    dup = table.duplicated(KEY_COLUMNS, keep=False)
    duplicate_rows = int(dup.sum())
    duplicate_keys = int(table.loc[dup, KEY_COLUMNS].drop_duplicates().shape[0]) if duplicate_rows else 0
    if duplicate_rows:
        log("Warning: duplicate exact keys after v5/accepted join; keeping first")
        table = table.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    metadata = {
        "v5_rows_read": int(rows_read),
        "v5_rows_after_cleanup": int(rows_after_cleanup),
        "accepted_v5_matched_rows": int(len(table)),
        "duplicate_join_rows": duplicate_rows,
        "duplicate_join_keys": duplicate_keys,
        "accepted_keys_without_v5_score": int(max(0, len(accepted) - table.loc[:, KEY_COLUMNS].drop_duplicates().shape[0])),
    }
    return table, metadata


def add_dexc_columns(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = table.copy()
    out["nonself_local_excitation_raw"] = pd.to_numeric(out[args.score_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["coupling_sum_raw"] = pd.to_numeric(out[args.coupling_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["target_excitation_Eg"] = pd.to_numeric(out[args.target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    out["target_excitation_error"] = pd.to_numeric(out[args.sg_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    coupling = out["coupling_sum_raw"].to_numpy(dtype=float)
    raw = out["nonself_local_excitation_raw"].to_numpy(dtype=float)
    eg = out["target_excitation_Eg"].to_numpy(dtype=float)
    nonzero = np.isfinite(coupling) & (coupling != 0.0)
    out["neighbor_excitation_weighted_mean"] = np.divide(raw, coupling, out=np.zeros_like(raw, dtype=float), where=nonzero)
    out["coupled_excitation_imbalance_norm"] = out["neighbor_excitation_weighted_mean"].to_numpy(dtype=float) - np.where(nonzero, eg, 0.0)
    out["excitation_deficit_norm"] = np.where(nonzero, eg, 0.0) - out["neighbor_excitation_weighted_mean"].to_numpy(dtype=float)
    out["coupled_excitation_imbalance_raw"] = raw - eg * coupling
    out.loc[~nonzero, ["neighbor_excitation_weighted_mean", "coupled_excitation_imbalance_norm", "excitation_deficit_norm"]] = 0.0
    return out


def four_mmm_orbit_variants(h: int, k: int, l: int) -> set[tuple[int, int, int]]:
    hk_ops = {
        (h, k),
        (k, h),
        (-h, k),
        (h, -k),
        (-k, h),
        (k, -h),
        (-h, -k),
        (-k, -h),
    }
    return {(int(hh), int(kk), int(ll)) for hh, kk in hk_ops for ll in {int(l), -int(l)}}


def four_mmm_orbit_id(h: int, k: int, l: int) -> str:
    canonical = min(four_mmm_orbit_variants(int(h), int(k), int(l)))
    return f"{canonical[0]}_{canonical[1]}_{canonical[2]}"


def hkl_family(h: int, k: int, l: int) -> str:
    h = int(h)
    k = int(k)
    l = int(l)
    nonzero = int(h != 0) + int(k != 0) + int(l != 0)
    if nonzero == 1:
        if l != 0:
            return "00l_axial"
        if h != 0:
            return "h00_axial"
        return "0k0_axial"
    if l == 0:
        return "hk0"
    if nonzero == 2:
        return "one_index_zero_l_nonzero"
    if abs(h) == abs(k):
        return "hhl_diagonal"
    return "fully_general"


def hkl_geometry_class(h: int, k: int, l: int) -> str:
    family = hkl_family(h, k, l)
    if family.endswith("axial"):
        return "axial"
    if family == "hk0":
        return "hk0"
    if family == "one_index_zero_l_nonzero":
        return "one_index_zero_l_nonzero"
    if family == "hhl_diagonal":
        return "diagonal_or_special"
    return "fully_general"


def basal_azimuth_deg(g: np.ndarray, reciprocal: np.ndarray) -> float:
    cstar = reciprocal[:, 2]
    c_unit = cstar / max(float(np.linalg.norm(cstar)), 1e-300)
    a_ref = reciprocal[:, 0] - np.dot(reciprocal[:, 0], c_unit) * c_unit
    if np.linalg.norm(a_ref) < 1e-12:
        a_ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
    e1 = a_ref / np.linalg.norm(a_ref)
    e2 = np.cross(c_unit, e1)
    projection = g - np.dot(g, c_unit) * c_unit
    if np.linalg.norm(projection) < 1e-12:
        return np.nan
    return float((np.degrees(np.arctan2(np.dot(projection, e2), np.dot(projection, e1))) + 360.0) % 360.0)


def add_reciprocal_descriptors(table: pd.DataFrame, reciprocal: np.ndarray) -> pd.DataFrame:
    out = table.copy()
    hkls = out.loc[:, HKL_COLUMNS].to_numpy(dtype=float)
    g_vectors = hkl_lab_vectors(hkls, reciprocal)
    g_norm = np.linalg.norm(g_vectors, axis=1)
    cstar = reciprocal[:, 2]
    cstar_unit = cstar / max(float(np.linalg.norm(cstar)), 1e-300)
    component = g_vectors @ cstar_unit
    out["g_norm_invA"] = g_norm
    out["d_spacing_angstrom"] = np.divide(1.0, g_norm, out=np.full_like(g_norm, np.nan), where=g_norm > 0.0)
    cosang = np.divide(np.abs(component), g_norm, out=np.full_like(g_norm, np.nan), where=g_norm > 0.0)
    cosang = np.clip(cosang, 0.0, 1.0)
    out["angle_to_cstar_deg"] = np.degrees(np.arccos(cosang))
    out["abs_g_cstar_component_over_g"] = cosang
    out["basal_azimuth_deg"] = [basal_azimuth_deg(vec, reciprocal) for vec in g_vectors]
    out["zero_miller_indices"] = (out.loc[:, HKL_COLUMNS].to_numpy(dtype=int) == 0).sum(axis=1)
    out["abs_h_equals_abs_k"] = np.abs(out["h"].to_numpy(dtype=int)) == np.abs(out["k"].to_numpy(dtype=int))
    out["is_axial"] = out["zero_miller_indices"].to_numpy(dtype=int) == 2
    out["is_00l"] = (out["h"].to_numpy(dtype=int) == 0) & (out["k"].to_numpy(dtype=int) == 0) & (out["l"].to_numpy(dtype=int) != 0)
    out["is_one_index_zero"] = out["zero_miller_indices"].to_numpy(dtype=int) == 1
    out["is_fully_general"] = out["zero_miller_indices"].to_numpy(dtype=int) == 0
    out["hkl_family"] = [hkl_family(h, k, l) for h, k, l in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
    out["hkl_geometry_class"] = [hkl_geometry_class(h, k, l) for h, k, l in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
    out["reciprocal_direction_class"] = np.select(
        [out["abs_g_cstar_component_over_g"] <= 0.25, out["abs_g_cstar_component_over_g"] >= 0.75],
        ["basal_or_near_basal", "cstar_directed"],
        default="oblique",
    )
    out["symmetry_orbit_id"] = [four_mmm_orbit_id(h, k, l) for h, k, l in out.loc[:, HKL_COLUMNS].itertuples(index=False, name=None)]
    return out


def qspread(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2:
        return 0.0
    q10, q90 = np.quantile(finite.to_numpy(dtype=float), [0.10, 0.90])
    return float(q90 - q10)


def build_candidate_summary(accepted_v5: pd.DataFrame, strength: pd.DataFrame, reciprocal: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    temp = accepted_v5.copy()
    temp["_nonzero_coupling"] = pd.to_numeric(temp["coupling_sum_raw"], errors="coerce").to_numpy(dtype=float) > 0.0
    grouped = temp.groupby(HKL_COLUMNS, sort=False)
    summary = grouped.agg(
        accepted_observation_count=("nonself_local_excitation_raw", "size"),
        nonzero_coupling_observation_count=("_nonzero_coupling", "sum"),
        median_v5_score=("nonself_local_excitation_raw", "median"),
        q90_v5_score=("nonself_local_excitation_raw", lambda s: float(np.nanquantile(pd.to_numeric(s, errors="coerce"), 0.90))),
        median_target_excitation=("target_excitation_Eg", "median"),
        target_excitation_spread_q90_q10=("target_excitation_Eg", qspread),
        excitation_deficit_norm_spread_q90_q10=("excitation_deficit_norm", qspread),
        median_partiality=("partiality", "median"),
    ).reset_index()
    summary["nonzero_coupling_fraction"] = summary["nonzero_coupling_observation_count"] / summary["accepted_observation_count"].clip(lower=1)
    summary = summary.merge(strength, on=HKL_COLUMNS, how="left", validate="one_to_one")
    summary = add_reciprocal_descriptors(summary, reciprocal)
    finite_intensity = np.isfinite(pd.to_numeric(summary["merged_intensity_or_Fobs"], errors="coerce").to_numpy(dtype=float))
    summary["candidate_pass_finite_intensity"] = finite_intensity
    summary["candidate_pass_min_accepted_obs"] = summary["accepted_observation_count"].to_numpy(dtype=int) >= int(args.min_accepted_obs)
    summary["candidate_pass_min_nonzero_coupling_obs"] = summary["nonzero_coupling_observation_count"].to_numpy(dtype=int) >= int(args.min_nonzero_coupling_obs)
    summary["candidate_pass_deficit_spread"] = summary["excitation_deficit_norm_spread_q90_q10"].to_numpy(dtype=float) >= float(args.min_deficit_spread)
    summary["candidate_pass_target_spread"] = summary["target_excitation_spread_q90_q10"].to_numpy(dtype=float) >= float(args.min_target_spread)
    summary["candidate_pass"] = (
        summary["candidate_pass_finite_intensity"]
        & summary["candidate_pass_min_accepted_obs"]
        & summary["candidate_pass_min_nonzero_coupling_obs"]
        & summary["candidate_pass_deficit_spread"]
        & summary["candidate_pass_target_spread"]
    )
    summary["candidate_rejection_reason"] = ""
    reason_masks = [
        ("missing_or_nonfinite_merged_intensity", ~summary["candidate_pass_finite_intensity"]),
        (f"accepted_obs_lt_{int(args.min_accepted_obs)}", ~summary["candidate_pass_min_accepted_obs"]),
        (f"nonzero_coupling_obs_lt_{int(args.min_nonzero_coupling_obs)}", ~summary["candidate_pass_min_nonzero_coupling_obs"]),
        (f"deficit_spread_lt_{float(args.min_deficit_spread):g}", ~summary["candidate_pass_deficit_spread"]),
        (f"target_spread_lt_{float(args.min_target_spread):g}", ~summary["candidate_pass_target_spread"]),
    ]
    reasons: list[str] = []
    for idx, row in summary.iterrows():
        row_reasons = [label for label, mask in reason_masks if bool(mask.iloc[idx])]
        reasons.append(";".join(row_reasons))
    summary["candidate_rejection_reason"] = reasons
    return assign_resolution_and_strength_bins(summary, int(args.resolution_bins))


def assign_resolution_and_strength_bins(candidates: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    out = candidates.copy()
    eligible = out["candidate_pass"].to_numpy(dtype=bool)
    out["resolution_bin_index"] = np.nan
    out["resolution_bin"] = ""
    if np.any(eligible):
        pct = out.loc[eligible, "g_norm_invA"].rank(method="first", pct=True).to_numpy(dtype=float)
        bin_index = np.minimum(np.floor(pct * int(n_bins)).astype(int), int(n_bins) - 1)
        out.loc[eligible, "resolution_bin_index"] = bin_index
        out.loc[eligible, "resolution_bin"] = [f"res_bin_{idx + 1}_of_{int(n_bins)}" for idx in bin_index]
    merged_i = pd.to_numeric(out["merged_intensity_or_Fobs"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    out["signed_log_merged_intensity"] = np.sign(merged_i) * np.log10(1.0 + np.abs(merged_i))
    out["strength_percentile_within_resolution_bin"] = np.nan
    out["reflection_strength_class"] = "not_classified"
    for _bin, group in out.loc[eligible].groupby("resolution_bin", sort=False):
        pct = group["merged_intensity_or_Fobs"].rank(method="average", pct=True)
        out.loc[group.index, "strength_percentile_within_resolution_bin"] = pct
        out.loc[group.index[pct <= 0.20], "reflection_strength_class"] = "weak"
        out.loc[group.index[(pct >= 0.40) & (pct <= 0.60)], "reflection_strength_class"] = "medium"
        out.loc[group.index[pct >= 0.80], "reflection_strength_class"] = "strong"
        out.loc[group.index[(pct > 0.20) & (pct < 0.40)], "reflection_strength_class"] = "weak_medium_gap"
        out.loc[group.index[(pct > 0.60) & (pct < 0.80)], "reflection_strength_class"] = "medium_strong_gap"
    return assign_angle_tertiles_and_azimuth_sectors(out)


def assign_angle_tertiles_and_azimuth_sectors(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["angle_tertile_index"] = np.nan
    out["angle_tertile"] = ""
    eligible = out["candidate_pass"].to_numpy(dtype=bool)
    for _res_bin, group in out.loc[eligible].groupby("resolution_bin", sort=False):
        ranks = group["angle_to_cstar_deg"].rank(method="first", pct=True)
        idx = np.minimum(np.floor(ranks.to_numpy(dtype=float) * 3).astype(int), 2)
        out.loc[group.index, "angle_tertile_index"] = idx
        out.loc[group.index, "angle_tertile"] = [ANGLE_TERTILE_LABELS[value] for value in idx]
    az = pd.to_numeric(out["basal_azimuth_deg"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    sector = np.floor(((az % 360.0) / 90.0)).astype(float)
    sector[~np.isfinite(az)] = np.nan
    out["basal_azimuth_sector"] = sector
    out.loc[out["angle_tertile"] == "cstar_aligned_angle_tertile", "basal_azimuth_sector"] = np.nan
    return out


def standardized_descriptor_matrix(table: pd.DataFrame) -> np.ndarray:
    az = np.deg2rad(pd.to_numeric(table["basal_azimuth_deg"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    accepted_count = (
        pd.to_numeric(table["accepted_observation_count"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    features = pd.DataFrame(
        {
            "g_norm_invA": pd.to_numeric(table["g_norm_invA"], errors="coerce").to_numpy(dtype=float),
            "signed_log_merged_intensity": pd.to_numeric(table["signed_log_merged_intensity"], errors="coerce").to_numpy(dtype=float),
            "angle_to_cstar_deg": pd.to_numeric(table["angle_to_cstar_deg"], errors="coerce").to_numpy(dtype=float),
            "azimuth_sin": np.sin(az),
            "azimuth_cos": np.cos(az),
            "median_v5_score": pd.to_numeric(table["median_v5_score"], errors="coerce").to_numpy(dtype=float),
            "nonzero_coupling_fraction": pd.to_numeric(table["nonzero_coupling_fraction"], errors="coerce").to_numpy(dtype=float),
            "log_accepted_observation_count": np.log1p(accepted_count),
            "excitation_deficit_norm_spread_q90_q10": pd.to_numeric(
                table["excitation_deficit_norm_spread_q90_q10"], errors="coerce"
            ).to_numpy(dtype=float),
        }
    )
    arr = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    arr = arr.fillna(arr.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    scale = np.nanstd(arr, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (arr - np.nanmean(arr, axis=0)) / scale


def farthest_indices(pool: pd.DataFrame, n: int, already: pd.DataFrame | None = None) -> list[int]:
    if n <= 0 or pool.empty:
        return []
    matrix = standardized_descriptor_matrix(pool)
    selected_positions: list[int] = []
    if already is not None and not already.empty:
        reference = standardized_descriptor_matrix(pd.concat([already, pool], ignore_index=True))
        ref = reference[: len(already)]
        pool_matrix = reference[len(already) :]
        min_dist = np.min(np.linalg.norm(pool_matrix[:, None, :] - ref[None, :, :], axis=2), axis=1) if len(ref) else np.full(len(pool), np.inf)
    else:
        centroid = np.nanmean(matrix, axis=0)
        min_dist = np.linalg.norm(matrix - centroid[None, :], axis=1)
    for _ in range(min(int(n), len(pool))):
        pos = int(np.argmax(min_dist))
        if pos in selected_positions:
            break
        selected_positions.append(pos)
        dist = np.linalg.norm(matrix - matrix[pos][None, :], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected_positions] = -np.inf
    return pool.index.to_numpy()[selected_positions].tolist()


def is_hk0_row(row: pd.Series) -> bool:
    return int(row["h"]) != 0 and int(row["k"]) != 0 and int(row["l"]) == 0


def is_diagonal_special_row(row: pd.Series) -> bool:
    return bool(row.get("abs_h_equals_abs_k", False)) and not bool(row.get("is_axial", False))


def primary_geometry_counts(table: pd.DataFrame) -> dict[str, int]:
    if table.empty:
        return {
            "fully_general": 0,
            "axial": 0,
            "exact_00l": 0,
            "one_index_zero": 0,
            "hk0": 0,
            "diagonal_special": 0,
        }
    return {
        "fully_general": int(table["is_fully_general"].fillna(False).astype(bool).sum()),
        "axial": int(table["is_axial"].fillna(False).astype(bool).sum()),
        "exact_00l": int(table["is_00l"].fillna(False).astype(bool).sum()),
        "one_index_zero": int(table["is_one_index_zero"].fillna(False).astype(bool).sum()),
        "hk0": int(((table["h"].astype(int) != 0) & (table["k"].astype(int) != 0) & (table["l"].astype(int) == 0)).sum()),
        "diagonal_special": int((table["abs_h_equals_abs_k"].fillna(False).astype(bool) & ~table["is_axial"].fillna(False).astype(bool)).sum()),
    }


def projected_geometry_ok(row: pd.Series, selected: pd.DataFrame, remaining_after: int) -> bool:
    counts = primary_geometry_counts(selected)
    projected = {
        "fully_general": counts["fully_general"] + int(bool(row["is_fully_general"])),
        "axial": counts["axial"] + int(bool(row["is_axial"])),
        "exact_00l": counts["exact_00l"] + int(bool(row["is_00l"])),
        "one_index_zero": counts["one_index_zero"] + int(bool(row["is_one_index_zero"])),
        "hk0": counts["hk0"] + int(is_hk0_row(row)),
        "diagonal_special": counts["diagonal_special"] + int(is_diagonal_special_row(row)),
    }
    if projected["axial"] > int(GEOMETRY_QUOTAS["max_axial"]):
        return False
    if projected["exact_00l"] > int(GEOMETRY_QUOTAS["max_00l"]):
        return False
    if projected["one_index_zero"] > int(GEOMETRY_QUOTAS["max_one_index_zero"]):
        return False
    return True


def azimuth_sector_penalty(row: pd.Series, selected: pd.DataFrame) -> float:
    if selected.empty or str(row.get("angle_tertile", "")) == "cstar_aligned_angle_tertile":
        return 0.0
    sector = row.get("basal_azimuth_sector", np.nan)
    if sector is None or not np.isfinite(float(sector)):
        return 0.0
    peers = selected.loc[
        (selected["resolution_bin"] == row["resolution_bin"])
        & (selected["angle_tertile"] == row["angle_tertile"])
        & (selected["basal_azimuth_sector"] == float(sector))
    ]
    return -2.5 * float(len(peers))


def selection_priority(row: pd.Series, selected: pd.DataFrame) -> float:
    counts = primary_geometry_counts(selected)
    priority = 0.0
    if counts["fully_general"] < int(GEOMETRY_QUOTAS["min_fully_general"]) and bool(row["is_fully_general"]):
        priority += 50.0
    if counts["hk0"] < int(GEOMETRY_QUOTAS["min_hk0"]) and is_hk0_row(row):
        priority += 95.0
    if counts["diagonal_special"] < int(GEOMETRY_QUOTAS["min_diagonal_special"]) and is_diagonal_special_row(row):
        priority += 95.0
    if bool(row["is_fully_general"]):
        priority += 10.0
    if bool(row["is_axial"]):
        priority -= 15.0
    if bool(row["is_one_index_zero"]) and not is_hk0_row(row):
        priority -= 30.0
    if bool(row["is_one_index_zero"]) and counts["one_index_zero"] >= int(GEOMETRY_QUOTAS["max_one_index_zero"]) - 4:
        priority -= 20.0
    priority += azimuth_sector_penalty(row, selected)
    return priority


def make_orbit_representatives(candidates: pd.DataFrame) -> pd.DataFrame:
    eligible = candidates.loc[candidates["candidate_pass"]].copy()
    if eligible.empty:
        raise SystemExit("No HKLs passed broad candidate requirements")
    accepted_count = (
        pd.to_numeric(eligible["accepted_observation_count"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    eligible["_representative_score"] = (
        np.log1p(accepted_count)
        + pd.to_numeric(eligible["nonzero_coupling_fraction"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        + 10.0 * pd.to_numeric(eligible["excitation_deficit_norm_spread_q90_q10"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    representatives = (
        eligible.sort_values(["symmetry_orbit_id", "_representative_score"], ascending=[True, False], kind="mergesort")
        .drop_duplicates("symmetry_orbit_id", keep="first")
        .copy()
    )
    return representatives.loc[
        representatives["reflection_strength_class"].isin(STRENGTH_CLASSES)
        & representatives["angle_tertile"].isin(ANGLE_TERTILE_LABELS)
    ].copy()


def quota_cell_availability(candidates: pd.DataFrame, representatives: pd.DataFrame) -> pd.DataFrame:
    eligible = candidates.loc[
        candidates["candidate_pass"]
        & candidates["reflection_strength_class"].isin(STRENGTH_CLASSES)
        & candidates["angle_tertile"].isin(ANGLE_TERTILE_LABELS)
    ]
    res_bins = [f"res_bin_{idx}_of_5" for idx in range(1, 6)]
    rows: list[dict[str, Any]] = []
    for res_bin in res_bins:
        for strength_class in STRENGTH_CLASSES:
            for angle_tertile in ANGLE_TERTILE_LABELS:
                all_count = int(
                    (
                        (eligible["resolution_bin"] == res_bin)
                        & (eligible["reflection_strength_class"] == strength_class)
                        & (eligible["angle_tertile"] == angle_tertile)
                    ).sum()
                )
                rep_count = int(
                    (
                        (representatives["resolution_bin"] == res_bin)
                        & (representatives["reflection_strength_class"] == strength_class)
                        & (representatives["angle_tertile"] == angle_tertile)
                    ).sum()
                )
                target = int(BASE_CELL_TARGETS[strength_class][angle_tertile])
                rows.append(
                    {
                        "resolution_bin": res_bin,
                        "reflection_strength_class": strength_class,
                        "angle_tertile": angle_tertile,
                        "cell_target": target,
                        "candidate_count_all_signed": all_count,
                        "candidate_count_4mmm_orbit_representatives": rep_count,
                        "cell_feasible_before_geometry": rep_count >= target,
                        "deficit_before_geometry": max(0, target - rep_count),
                    }
                )
    return pd.DataFrame(rows)


def adjacent_strength_classes(strength_class: str) -> list[str]:
    if strength_class == "weak":
        return ["medium", "strong"]
    if strength_class == "strong":
        return ["medium", "weak"]
    return ["weak", "strong"]


def choose_quota_candidate(
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    selected_orbits: set[str],
    remaining_after: int,
) -> int | None:
    if pool.empty:
        return None
    pool = pool.loc[~pool["symmetry_orbit_id"].isin(selected_orbits)].copy()
    if pool.empty:
        return None
    allowed_indices = [
        idx
        for idx, row in pool.iterrows()
        if projected_geometry_ok(row, selected, remaining_after)
    ]
    if not allowed_indices:
        return None
    allowed = pool.loc[allowed_indices].copy()
    priorities = pd.Series({idx: selection_priority(allowed.loc[idx], selected) for idx in allowed.index}, dtype=float)
    top_priority = float(priorities.max())
    top = allowed.loc[priorities.index[priorities == top_priority]].copy()
    order = farthest_indices(top, len(top), already=selected)
    return int(order[0]) if order else None


def append_selected_row(selected: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return row.copy()
    return pd.concat([selected, row], ignore_index=False, sort=False)


def min_geometry_deficits(primary: pd.DataFrame) -> dict[str, int]:
    counts = primary_geometry_counts(primary)
    return {
        "fully_general": max(0, int(GEOMETRY_QUOTAS["min_fully_general"]) - counts["fully_general"]),
        "hk0": max(0, int(GEOMETRY_QUOTAS["min_hk0"]) - counts["hk0"]),
        "diagonal_special": max(0, int(GEOMETRY_QUOTAS["min_diagonal_special"]) - counts["diagonal_special"]),
    }


def row_satisfies_min_category(row: pd.Series, category: str) -> bool:
    if category == "fully_general":
        return bool(row["is_fully_general"])
    if category == "hk0":
        return is_hk0_row(row)
    if category == "diagonal_special":
        return is_diagonal_special_row(row)
    raise ValueError(f"Unknown geometry category: {category}")


def max_geometry_caps_ok(primary: pd.DataFrame) -> bool:
    counts = primary_geometry_counts(primary)
    return (
        counts["axial"] <= int(GEOMETRY_QUOTAS["max_axial"])
        and counts["exact_00l"] <= int(GEOMETRY_QUOTAS["max_00l"])
        and counts["one_index_zero"] <= int(GEOMETRY_QUOTAS["max_one_index_zero"])
    )


def repair_min_geometry_quotas(selected: pd.DataFrame, representatives: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    repaired = selected.copy()
    repairs: list[dict[str, Any]] = []
    for _ in range(100):
        deficits = {key: value for key, value in min_geometry_deficits(repaired).items() if value > 0}
        if not deficits:
            return repaired, repairs
        category = max(deficits, key=lambda key: deficits[key])
        selected_orbits = set(str(value) for value in repaired["symmetry_orbit_id"].dropna())
        removal_candidates = repaired.loc[~repaired.apply(lambda row: row_satisfies_min_category(row, category), axis=1)].copy()
        if removal_candidates.empty:
            break
        removal_candidates["_removal_penalty"] = 0
        current_counts = primary_geometry_counts(repaired)
        for other_category, quota_key in [
            ("fully_general", "min_fully_general"),
            ("hk0", "min_hk0"),
            ("diagonal_special", "min_diagonal_special"),
        ]:
            if other_category == category:
                continue
            at_floor = current_counts[other_category] <= int(GEOMETRY_QUOTAS[quota_key])
            if at_floor:
                removal_candidates.loc[
                    removal_candidates.apply(lambda row: row_satisfies_min_category(row, other_category), axis=1),
                    "_removal_penalty",
                ] += 100
        removal_candidates = removal_candidates.sort_values(["_removal_penalty", "selection_order"], ascending=[True, False], kind="mergesort")
        swapped = False
        for out_idx, out_row in removal_candidates.iterrows():
            pool = representatives.loc[
                (representatives["resolution_bin"] == out_row["resolution_bin"])
                & (representatives["reflection_strength_class"] == out_row["reflection_strength_class"])
                & (representatives["angle_tertile"] == out_row["angle_tertile"])
                & representatives.apply(lambda row: row_satisfies_min_category(row, category), axis=1)
                & ~representatives["symmetry_orbit_id"].isin(selected_orbits - {str(out_row["symmetry_orbit_id"])})
            ].copy()
            if pool.empty:
                continue
            for in_idx in farthest_indices(pool, len(pool), already=repaired.drop(index=out_idx)):
                in_row = pool.loc[[in_idx]].copy()
                trial = pd.concat([repaired.drop(index=out_idx), in_row], ignore_index=False, sort=False)
                if trial["symmetry_orbit_id"].duplicated().any() or not max_geometry_caps_ok(trial):
                    continue
                in_row["selection_order"] = out_row["selection_order"]
                in_row["selection_reason"] = f"primary_geometry_repair_{category}"
                in_row["selection_requested_resolution_bin"] = out_row.get("selection_requested_resolution_bin", out_row["resolution_bin"])
                in_row["selection_requested_strength_class"] = out_row.get("selection_requested_strength_class", out_row["reflection_strength_class"])
                in_row["selection_requested_angle_tertile"] = out_row.get("selection_requested_angle_tertile", out_row["angle_tertile"])
                in_row["selection_actual_strength_class"] = in_row["reflection_strength_class"].iloc[0]
                in_row["selection_actual_angle_tertile"] = in_row["angle_tertile"].iloc[0]
                in_row["selection_substitution"] = f"geometry_repair:{category}"
                repaired = pd.concat([repaired.drop(index=out_idx), in_row], ignore_index=False, sort=False)
                repairs.append(
                    {
                        "category": category,
                        "removed_h": int(out_row["h"]),
                        "removed_k": int(out_row["k"]),
                        "removed_l": int(out_row["l"]),
                        "added_h": int(in_row["h"].iloc[0]),
                        "added_k": int(in_row["k"].iloc[0]),
                        "added_l": int(in_row["l"].iloc[0]),
                        "resolution_bin": str(out_row["resolution_bin"]),
                        "reflection_strength_class": str(out_row["reflection_strength_class"]),
                        "angle_tertile": str(out_row["angle_tertile"]),
                    }
                )
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            raise SystemExit(
                f"Could not repair primary HKL geometry quota for {category}; "
                f"current_counts={primary_geometry_counts(repaired)}"
            )
    raise SystemExit(f"Geometry repair did not converge; current_counts={primary_geometry_counts(repaired)}")


def select_primary_hkls_milp(representatives: pd.DataFrame, target_n: int, n_bins: int) -> pd.DataFrame:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local scipy build
        raise SystemExit(f"SciPy MILP fallback is unavailable: {exc}") from exc

    pool = representatives.loc[
        representatives["reflection_strength_class"].isin(STRENGTH_CLASSES)
        & representatives["angle_tertile"].isin(ANGLE_TERTILE_LABELS)
    ].copy()
    if pool.empty:
        raise SystemExit("MILP fallback has no selectable representatives")
    n = len(pool)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(mask: np.ndarray, lb: float, ub: float) -> None:
        rows.append(mask.astype(float))
        lower.append(float(lb))
        upper.append(float(ub))

    add_constraint(np.ones(n, dtype=bool), float(target_n), float(target_n))
    res_bins = [f"res_bin_{idx}_of_{int(n_bins)}" for idx in range(1, int(n_bins) + 1)]
    for res_bin in res_bins:
        res_mask = (pool["resolution_bin"] == res_bin).to_numpy(dtype=bool)
        add_constraint(res_mask, RESOLUTION_BIN_TARGET, RESOLUTION_BIN_TARGET)
        for strength_class in STRENGTH_CLASSES:
            add_constraint(
                (res_mask & (pool["reflection_strength_class"] == strength_class).to_numpy(dtype=bool)),
                STRENGTH_PER_BIN_TARGET,
                STRENGTH_PER_BIN_TARGET,
            )
        for angle_tertile in ANGLE_TERTILE_LABELS:
            add_constraint(
                (res_mask & (pool["angle_tertile"] == angle_tertile).to_numpy(dtype=bool)),
                ANGLE_PER_BIN_TARGET,
                ANGLE_PER_BIN_TARGET,
            )

    add_constraint(pool["is_axial"].fillna(False).to_numpy(dtype=bool), 0, int(GEOMETRY_QUOTAS["max_axial"]))
    add_constraint(pool["is_00l"].fillna(False).to_numpy(dtype=bool), 0, int(GEOMETRY_QUOTAS["max_00l"]))
    add_constraint(pool["is_one_index_zero"].fillna(False).to_numpy(dtype=bool), 0, int(GEOMETRY_QUOTAS["max_one_index_zero"]))
    add_constraint(pool["is_fully_general"].fillna(False).to_numpy(dtype=bool), int(GEOMETRY_QUOTAS["min_fully_general"]), np.inf)
    add_constraint(
        ((pool["h"].astype(int) != 0) & (pool["k"].astype(int) != 0) & (pool["l"].astype(int) == 0)).to_numpy(dtype=bool),
        int(GEOMETRY_QUOTAS["min_hk0"]),
        np.inf,
    )
    add_constraint(
        (pool["abs_h_equals_abs_k"].fillna(False).astype(bool) & ~pool["is_axial"].fillna(False).astype(bool)).to_numpy(dtype=bool),
        int(GEOMETRY_QUOTAS["min_diagonal_special"]),
        np.inf,
    )

    descriptor = standardized_descriptor_matrix(pool)
    diversity_surrogate = np.linalg.norm(descriptor - np.nanmean(descriptor, axis=0)[None, :], axis=1)
    representative_score = pd.to_numeric(pool.get("_representative_score", pd.Series(0.0, index=pool.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score = representative_score + 0.1 * diversity_surrogate
    score += 0.25 * pool["is_fully_general"].fillna(False).to_numpy(dtype=float)
    score += 0.5 * ((pool["h"].astype(int) != 0) & (pool["k"].astype(int) != 0) & (pool["l"].astype(int) == 0)).to_numpy(dtype=float)
    score += 0.5 * (pool["abs_h_equals_abs_k"].fillna(False).astype(bool) & ~pool["is_axial"].fillna(False).astype(bool)).to_numpy(dtype=float)
    score -= 0.1 * pool["is_axial"].fillna(False).to_numpy(dtype=float)
    constraints = LinearConstraint(np.vstack(rows), np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    result = milp(
        c=-score,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n, dtype=float), np.ones(n, dtype=float)),
        constraints=constraints,
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise SystemExit(f"MILP hard-quota fallback failed: {result.message}")
    chosen = pool.loc[np.asarray(result.x) > 0.5].copy()
    if len(chosen) != int(target_n):
        raise SystemExit(f"MILP hard-quota fallback selected {len(chosen)} HKLs, expected {int(target_n)}")
    order = farthest_indices(chosen, len(chosen))
    chosen = chosen.loc[order].copy()
    chosen["selection_order"] = np.arange(1, len(chosen) + 1)
    chosen["selection_reason"] = "primary_milp_hard_quota_fallback"
    chosen["selection_requested_resolution_bin"] = chosen["resolution_bin"]
    chosen["selection_requested_strength_class"] = chosen["reflection_strength_class"]
    chosen["selection_requested_angle_tertile"] = chosen["angle_tertile"]
    chosen["selection_actual_strength_class"] = chosen["reflection_strength_class"]
    chosen["selection_actual_angle_tertile"] = chosen["angle_tertile"]
    chosen["selection_substitution"] = "milp_hard_quota_fallback"
    return chosen


def assert_primary_quotas(primary: pd.DataFrame, target_n: int, n_bins: int) -> None:
    errors: list[str] = []
    if len(primary) != int(target_n):
        errors.append(f"expected {int(target_n)} primary HKLs, selected {len(primary)}")
    duplicate_orbits = int(primary["symmetry_orbit_id"].duplicated().sum()) if "symmetry_orbit_id" in primary.columns else 0
    if duplicate_orbits:
        errors.append(f"selected {duplicate_orbits} duplicate 4/mmm orbit representatives")
    res_bins = [f"res_bin_{idx}_of_{int(n_bins)}" for idx in range(1, int(n_bins) + 1)]
    for res_bin in res_bins:
        count = int((primary["resolution_bin"] == res_bin).sum())
        if count != RESOLUTION_BIN_TARGET:
            errors.append(f"{res_bin}: expected {RESOLUTION_BIN_TARGET} HKLs, selected {count}")
        for strength_class in STRENGTH_CLASSES:
            scount = int(((primary["resolution_bin"] == res_bin) & (primary["reflection_strength_class"] == strength_class)).sum())
            if scount != STRENGTH_PER_BIN_TARGET:
                errors.append(f"{res_bin}/{strength_class}: expected {STRENGTH_PER_BIN_TARGET}, selected {scount}")
        for angle_tertile in ANGLE_TERTILE_LABELS:
            acount = int(((primary["resolution_bin"] == res_bin) & (primary["angle_tertile"] == angle_tertile)).sum())
            if acount != ANGLE_PER_BIN_TARGET:
                errors.append(f"{res_bin}/{angle_tertile}: expected {ANGLE_PER_BIN_TARGET}, selected {acount}")
    counts = primary_geometry_counts(primary)
    if counts["fully_general"] < int(GEOMETRY_QUOTAS["min_fully_general"]):
        errors.append(f"fully_general quota failed: {counts['fully_general']} < {GEOMETRY_QUOTAS['min_fully_general']}")
    if counts["axial"] > int(GEOMETRY_QUOTAS["max_axial"]):
        errors.append(f"axial cap failed: {counts['axial']} > {GEOMETRY_QUOTAS['max_axial']}")
    if counts["exact_00l"] > int(GEOMETRY_QUOTAS["max_00l"]):
        errors.append(f"00l cap failed: {counts['exact_00l']} > {GEOMETRY_QUOTAS['max_00l']}")
    if counts["one_index_zero"] > int(GEOMETRY_QUOTAS["max_one_index_zero"]):
        errors.append(f"one-index-zero cap failed: {counts['one_index_zero']} > {GEOMETRY_QUOTAS['max_one_index_zero']}")
    if counts["hk0"] < int(GEOMETRY_QUOTAS["min_hk0"]):
        errors.append(f"hk0 quota failed: {counts['hk0']} < {GEOMETRY_QUOTAS['min_hk0']}")
    if counts["diagonal_special"] < int(GEOMETRY_QUOTAS["min_diagonal_special"]):
        errors.append(f"diagonal/special quota failed: {counts['diagonal_special']} < {GEOMETRY_QUOTAS['min_diagonal_special']}")
    if errors:
        raise SystemExit("Primary HKL hard-quota validation failed:\n- " + "\n- ".join(errors))


def select_primary_hkls(candidates: pd.DataFrame, target_n: int, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = candidates.copy()
    audit["selected_primary"] = False
    audit["selection_reason"] = ""
    audit["selection_order"] = np.nan
    audit["selection_requested_resolution_bin"] = ""
    audit["selection_requested_strength_class"] = ""
    audit["selection_requested_angle_tertile"] = ""
    audit["selection_actual_strength_class"] = ""
    audit["selection_actual_angle_tertile"] = ""
    audit["selection_substitution"] = ""
    representatives = make_orbit_representatives(audit)
    availability = quota_cell_availability(audit, representatives)
    log("Candidate availability before primary selection:")
    print(availability.to_string(index=False))
    deficient = availability.loc[~availability["cell_feasible_before_geometry"]]
    if not deficient.empty:
        log("Quota cells deficient before geometry; adjacent strength substitutions may be needed:")
        print(deficient.to_string(index=False))

    selected = representatives.iloc[0:0].copy()
    selected_orbits: set[str] = set()
    substitutions: list[dict[str, Any]] = []
    res_bins = [f"res_bin_{idx}_of_{int(n_bins)}" for idx in range(1, int(n_bins) + 1)]
    remaining_strength = {(res_bin, strength): STRENGTH_PER_BIN_TARGET for res_bin in res_bins for strength in STRENGTH_CLASSES}
    remaining_angle = {(res_bin, angle): ANGLE_PER_BIN_TARGET for res_bin in res_bins for angle in ANGLE_TERTILE_LABELS}
    angle_order = {angle: idx for idx, angle in enumerate(reversed(ANGLE_TERTILE_LABELS))}
    planned_slots: list[dict[str, Any]] = []
    for res_bin in res_bins:
        for strength_class in STRENGTH_CLASSES:
            for angle_tertile in ANGLE_TERTILE_LABELS:
                rep_count = int(
                    availability.loc[
                        (availability["resolution_bin"] == res_bin)
                        & (availability["reflection_strength_class"] == strength_class)
                        & (availability["angle_tertile"] == angle_tertile),
                        "candidate_count_4mmm_orbit_representatives",
                    ].iloc[0]
                )
                target = int(BASE_CELL_TARGETS[strength_class][angle_tertile])
                for _ in range(target):
                    planned_slots.append(
                        {
                            "resolution_bin": res_bin,
                            "requested_strength": strength_class,
                            "requested_angle": angle_tertile,
                            "representative_count": rep_count,
                            "cell_target": target,
                        }
                    )
    planned_slots = sorted(
        planned_slots,
        key=lambda item: (
            int(item["representative_count"]),
            -int(item["cell_target"]),
            angle_order[str(item["requested_angle"])],
            str(item["resolution_bin"]),
            str(item["requested_strength"]),
        ),
    )
    selection_order = 0
    for slot in planned_slots:
        res_bin = str(slot["resolution_bin"])
        requested_strength = str(slot["requested_strength"])
        requested_angle = str(slot["requested_angle"])
        if remaining_angle[(res_bin, requested_angle)] <= 0:
            continue
        strength_options = [requested_strength, *adjacent_strength_classes(requested_strength)]
        chosen_idx: int | None = None
        chosen_strength = ""
        selection_note = "primary_exact_resolution_strength_angle_quota"
        for actual_strength in strength_options:
            if remaining_strength.get((res_bin, actual_strength), 0) <= 0:
                continue
            pool = representatives.loc[
                (representatives["resolution_bin"] == res_bin)
                & (representatives["reflection_strength_class"] == actual_strength)
                & (representatives["angle_tertile"] == requested_angle)
            ]
            idx = choose_quota_candidate(pool, selected, selected_orbits, int(target_n) - len(selected) - 1)
            if idx is None:
                continue
            chosen_idx = idx
            chosen_strength = actual_strength
            if actual_strength != requested_strength:
                selection_note = "primary_adjacent_strength_substitution"
                substitutions.append(
                    {
                        "resolution_bin": res_bin,
                        "requested_strength_class": requested_strength,
                        "requested_angle_tertile": requested_angle,
                        "actual_strength_class": actual_strength,
                        "reason": "exact requested resolution-strength-angle cell had no selectable representative",
                    }
                )
            break
        if chosen_idx is None:
            counts = primary_geometry_counts(selected)
            raise SystemExit(
                "No selectable HKL for quota cell "
                f"{res_bin}/{requested_strength}/{requested_angle}; "
                "nearest adjacent strength substitution also failed. "
                f"selected_so_far={len(selected)}, geometry_counts={counts}, "
                f"remaining_strength={remaining_strength}, remaining_angle={remaining_angle}"
            )
        row = representatives.loc[[chosen_idx]].copy()
        row["selection_order"] = selection_order + 1
        row["selection_reason"] = selection_note
        row["selection_requested_resolution_bin"] = res_bin
        row["selection_requested_strength_class"] = requested_strength
        row["selection_requested_angle_tertile"] = requested_angle
        row["selection_actual_strength_class"] = chosen_strength
        row["selection_actual_angle_tertile"] = requested_angle
        row["selection_substitution"] = f"{requested_strength}->{chosen_strength}" if chosen_strength != requested_strength else ""
        selected = append_selected_row(selected, row)
        selected_orbits.add(str(row["symmetry_orbit_id"].iloc[0]))
        remaining_strength[(res_bin, chosen_strength)] -= 1
        remaining_angle[(res_bin, requested_angle)] -= 1
        selection_order += 1
        audit.loc[chosen_idx, "selected_primary"] = True
        audit.loc[chosen_idx, "selection_order"] = selection_order
        audit.loc[chosen_idx, "selection_reason"] = selection_note
        audit.loc[chosen_idx, "selection_requested_resolution_bin"] = res_bin
        audit.loc[chosen_idx, "selection_requested_strength_class"] = requested_strength
        audit.loc[chosen_idx, "selection_requested_angle_tertile"] = requested_angle
        audit.loc[chosen_idx, "selection_actual_strength_class"] = chosen_strength
        audit.loc[chosen_idx, "selection_actual_angle_tertile"] = requested_angle
        if chosen_strength != requested_strength:
            audit.loc[chosen_idx, "selection_substitution"] = f"{requested_strength}->{chosen_strength}"
    if any(value != 0 for value in remaining_strength.values()) or any(value != 0 for value in remaining_angle.values()):
        raise SystemExit(
            "Internal quota accounting failed: "
            f"remaining_strength={remaining_strength}, remaining_angle={remaining_angle}"
        )

    try:
        selected, geometry_repairs = repair_min_geometry_quotas(selected, representatives)
        if geometry_repairs:
            log("Applied same-cell geometry quota repairs:")
            print(pd.DataFrame(geometry_repairs).to_string(index=False))
            substitutions.extend({"geometry_repair": True, **repair} for repair in geometry_repairs)
    except SystemExit as exc:
        log(f"Greedy same-cell geometry repair failed; using MILP hard-quota fallback. Reason: {exc}")
        selected = select_primary_hkls_milp(representatives, int(target_n), int(n_bins))
        substitutions.append({"milp_hard_quota_fallback": True, "reason": str(exc)})
    selected = selected.sort_values("selection_order", kind="mergesort").copy()
    assert_primary_quotas(selected, int(target_n), int(n_bins))
    audit["selected_primary"] = False
    audit["selection_order"] = np.nan
    audit["selection_requested_resolution_bin"] = ""
    audit["selection_requested_strength_class"] = ""
    audit["selection_requested_angle_tertile"] = ""
    audit["selection_actual_strength_class"] = ""
    audit["selection_actual_angle_tertile"] = ""
    audit["selection_substitution"] = ""
    for idx, row in selected.iterrows():
        audit.loc[idx, "selected_primary"] = True
        audit.loc[idx, "selection_order"] = row.get("selection_order", np.nan)
        audit.loc[idx, "selection_reason"] = row.get("selection_reason", "primary_quota_selected")
        audit.loc[idx, "selection_requested_resolution_bin"] = row.get("selection_requested_resolution_bin", row["resolution_bin"])
        audit.loc[idx, "selection_requested_strength_class"] = row.get("selection_requested_strength_class", row["reflection_strength_class"])
        audit.loc[idx, "selection_requested_angle_tertile"] = row.get("selection_requested_angle_tertile", row["angle_tertile"])
        audit.loc[idx, "selection_actual_strength_class"] = row.get("selection_actual_strength_class", row["reflection_strength_class"])
        audit.loc[idx, "selection_actual_angle_tertile"] = row.get("selection_actual_angle_tertile", row["angle_tertile"])
        audit.loc[idx, "selection_substitution"] = row.get("selection_substitution", "")
    audit.loc[audit["candidate_pass"] & ~audit["selected_primary"], "selection_reason"] = "eligible_not_selected"
    audit.loc[~audit["candidate_pass"], "selection_reason"] = audit.loc[~audit["candidate_pass"], "candidate_rejection_reason"]
    if substitutions:
        audit.attrs["selection_substitutions"] = substitutions
    return selected.reset_index(drop=True), audit, availability


def select_symmetry_pairs(candidates: pd.DataFrame, primary: pd.DataFrame, n_pairs: int) -> pd.DataFrame:
    if int(n_pairs) <= 0:
        return pd.DataFrame(columns=[*candidates.columns, "symmetry_pair_id", "symmetry_pair_role"])
    eligible = candidates.loc[candidates["candidate_pass"]].copy()
    rows: list[pd.DataFrame] = []
    used_orbits: set[str] = set()
    primary_orbits = list(primary["symmetry_orbit_id"].dropna().unique())
    preferred_orbits = primary_orbits + [orbit for orbit in eligible["symmetry_orbit_id"].dropna().unique() if orbit not in set(primary_orbits)]
    for orbit in preferred_orbits:
        if orbit in used_orbits:
            continue
        group = eligible.loc[eligible["symmetry_orbit_id"] == orbit].sort_values(
            ["accepted_observation_count", "nonzero_coupling_fraction"], ascending=[False, False], kind="mergesort"
        )
        if len(group) < 2:
            continue
        pair = group.head(2).copy()
        pair_id = f"sympair_{len(rows) + 1:02d}_{orbit}"
        pair["symmetry_pair_id"] = pair_id
        pair["symmetry_pair_role"] = ["member_a", "member_b"][: len(pair)]
        rows.append(pair)
        used_orbits.add(orbit)
        if len(rows) >= int(n_pairs):
            break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*candidates.columns, "symmetry_pair_id", "symmetry_pair_role"])


def combine_selected_hkls(primary: pd.DataFrame, symmetry_pairs: pd.DataFrame) -> pd.DataFrame:
    primary = primary.copy()
    primary["selection_set"] = "primary"
    primary["include_in_primary_aggregate"] = True
    primary["symmetry_pair_id"] = ""
    primary["symmetry_pair_role"] = ""
    if symmetry_pairs.empty:
        return primary
    sym = symmetry_pairs.copy()
    sym["selection_set"] = "symmetry_reproducibility"
    sym["include_in_primary_aggregate"] = False
    combined = pd.concat([primary, sym], ignore_index=True, sort=False)
    key = combined.loc[:, HKL_COLUMNS].astype(str).agg("_".join, axis=1)
    combined["_key"] = key
    merged_rows: list[pd.Series] = []
    for _key, group in combined.groupby("_key", sort=False):
        row = group.iloc[0].copy()
        sets = sorted(set(str(value) for value in group["selection_set"].dropna()))
        row["selection_set"] = "+".join(sets)
        row["include_in_primary_aggregate"] = bool(group["include_in_primary_aggregate"].any())
        pair_ids = [str(value) for value in group.get("symmetry_pair_id", pd.Series(dtype=str)).dropna() if str(value)]
        roles = [str(value) for value in group.get("symmetry_pair_role", pd.Series(dtype=str)).dropna() if str(value)]
        row["symmetry_pair_id"] = ";".join(sorted(set(pair_ids)))
        row["symmetry_pair_role"] = ";".join(sorted(set(roles)))
        merged_rows.append(row)
    return pd.DataFrame(merged_rows).drop(columns=["_key"], errors="ignore").reset_index(drop=True)


def add_intensity_responses(table: pd.DataFrame, selected: pd.DataFrame, scale_columns: tuple[str, ...]) -> pd.DataFrame:
    out = table.copy().reset_index(drop=True)
    merged = out["merged_intensity_or_Fobs"].to_numpy(dtype=float)
    intensity = out["observation_intensity"].to_numpy(dtype=float)
    out["I_over_merged"] = np.divide(
        intensity,
        merged,
        out=np.full_like(intensity, np.nan, dtype=float),
        where=np.isfinite(merged) & (np.abs(merged) > 1e-12),
    )
    hkl_median = out.groupby(HKL_COLUMNS, sort=False)["observation_intensity"].transform("median")
    out["hkl_median_observation_intensity"] = hkl_median
    denom = hkl_median.to_numpy(dtype=float)
    out["I_over_hkl_median"] = np.divide(
        intensity,
        denom,
        out=np.full_like(intensity, np.nan, dtype=float),
        where=np.isfinite(denom) & (np.abs(denom) > 1e-12),
    )
    mad = out.groupby(HKL_COLUMNS, sort=False)["observation_intensity"].transform(lambda s: float(np.nanmedian(np.abs(s - np.nanmedian(s)))))
    out["I_robust_z_within_hkl"] = np.divide(
        intensity - denom,
        1.4826 * mad.to_numpy(dtype=float),
        out=np.full_like(intensity, np.nan, dtype=float),
        where=np.isfinite(mad.to_numpy(dtype=float)) & (mad.to_numpy(dtype=float) > 1e-12),
    )
    residual_values = np.full(len(out), np.nan, dtype=float)
    for _hkl, group in out.groupby(HKL_COLUMNS, sort=False):
        residual_values[group.index.to_numpy(dtype=int)] = robust_residual_for_group(group, scale_columns)
    out["corrected_intensity_residual"] = residual_values
    return out


def robust_residual_for_group(group: pd.DataFrame, scale_columns: tuple[str, ...]) -> np.ndarray:
    y = pd.to_numeric(group["I_over_hkl_median"], errors="coerce").to_numpy(dtype=float)
    covariates = [pd.Series(1.0, index=group.index), group["target_excitation_Eg"], group["partiality"]]
    for scale_column in scale_columns:
        if scale_column in group.columns:
            covariates.append(group[scale_column])
    x = np.column_stack([pd.to_numeric(col, errors="coerce").to_numpy(dtype=float) for col in covariates])
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    residual = np.full(len(group), np.nan, dtype=float)
    if int(mask.sum()) < max(3, x.shape[1] + 1):
        median = np.nanmedian(y)
        residual[mask] = y[mask] - median if np.isfinite(median) else np.nan
        return residual
    beta = huber_irls(x[mask], y[mask])
    residual[mask] = y[mask] - x[mask] @ beta
    return residual


def huber_irls(x: np.ndarray, y: np.ndarray, c: float = 1.345, max_iter: int = 30) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    for _ in range(int(max_iter)):
        resid = y - x @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if not np.isfinite(scale) or scale <= 1e-12:
            break
        weights = np.ones_like(resid)
        large = np.abs(resid) > c * scale
        weights[large] = (c * scale) / np.abs(resid[large])
        wx = x * weights[:, None]
        new_beta, *_ = np.linalg.lstsq(wx, y * weights, rcond=None)
        if np.linalg.norm(new_beta - beta) < 1e-8:
            beta = new_beta
            break
        beta = new_beta
    return beta


def continuous_uvw_text(values: np.ndarray) -> str:
    return f"[{values[0]:.6g} {values[1]:.6g} {values[2]:.6g}]"


def add_orientation_columns(table: pd.DataFrame, stream: Path, uvw_max: int) -> pd.DataFrame:
    log(f"Recovering exact orientation/UVW for {len(table):,} selected accepted observations")
    orientation = load_stream_orientation_for_selected(stream, table.loc[:, KEY_COLUMNS])
    if len(orientation) != len(table):
        found = set(tuple(row) for row in orientation.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
        missing = [tuple(row) for row in table.loc[:, KEY_COLUMNS].itertuples(index=False, name=None) if tuple(row) not in found]
        raise SystemExit(f"Stream orientation lookup missed {len(missing)} exact key(s); first missing: {missing[:3]}")
    joined = table.merge(
        orientation.loc[:, [*KEY_COLUMNS, *STREAM_MATRIX_COLUMNS]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    axes = unique_zone_axes(int(uvw_max))
    xyz: list[np.ndarray] = []
    closest: list[str] = []
    angle: list[float] = []
    for _, row in joined.iterrows():
        reciprocal = reciprocal_matrix_from_row(row)
        beam_uvw = beam_in_direct_coordinates(reciprocal)
        best_axis = min(axes, key=lambda axis: axis_angle_deg(reciprocal, axis))
        xyz.append(beam_uvw)
        closest.append(triplet_label(best_axis))
        angle.append(float(axis_angle_deg(reciprocal, best_axis)))
    arr = np.vstack(xyz) if xyz else np.empty((0, 3))
    joined["continuous_uvw_x"] = arr[:, 0] if len(arr) else []
    joined["continuous_uvw_y"] = arr[:, 1] if len(arr) else []
    joined["continuous_uvw_z"] = arr[:, 2] if len(arr) else []
    joined["continuous_uvw"] = [continuous_uvw_text(value) for value in xyz]
    joined["closest_uvw"] = closest
    joined["closest_zone_axis_angle_deg"] = angle
    return joined


def spearman_with_p(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    a = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    b = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = a.notna() & b.notna()
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    if scipy_stats is not None:
        result = scipy_stats.spearmanr(a.loc[mask].to_numpy(dtype=float), b.loc[mask].to_numpy(dtype=float))
        return float(result.statistic), float(result.pvalue)
    return float(a.loc[mask].rank().corr(b.loc[mask].rank())), np.nan


def partial_spearman(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> float:
    frame = pd.concat([x.rename("_x"), y.rename("_y"), controls], axis=1)
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < max(5, controls.shape[1] + 3):
        return np.nan
    xr = frame["_x"].rank(method="average").to_numpy(dtype=float)
    yr = frame["_y"].rank(method="average").to_numpy(dtype=float)
    cr = np.column_stack([np.ones(len(frame)), *[frame[column].rank(method="average").to_numpy(dtype=float) for column in controls.columns]])
    rx = xr - cr @ np.linalg.lstsq(cr, xr, rcond=None)[0]
    ry = yr - cr @ np.linalg.lstsq(cr, yr, rcond=None)[0]
    if np.std(rx) <= 0.0 or np.std(ry) <= 0.0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def robust_slope(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x.rename("_x"), y.rename("_y")], axis=1).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["_x"].nunique() < 2:
        return np.nan
    xv = frame["_x"].to_numpy(dtype=float)
    yv = frame["_y"].to_numpy(dtype=float)
    if scipy_stats is not None:
        return float(scipy_stats.theilslopes(yv, xv).slope)
    if len(frame) > 250:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(frame), size=250, replace=False)
        xv = xv[idx]
        yv = yv[idx]
    dx = xv[None, :] - xv[:, None]
    dy = yv[None, :] - yv[:, None]
    mask = dx != 0.0
    slopes = dy[mask] / dx[mask]
    return float(np.median(slopes)) if len(slopes) else np.nan


def bootstrap_slope_ci(x: pd.Series, y: pd.Series, iterations: int, seed: int) -> tuple[float, float]:
    frame = pd.concat([x.rename("_x"), y.rename("_y")], axis=1).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if int(iterations) <= 0 or len(frame) < 8 or frame["_x"].nunique() < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    slopes = []
    xv = frame["_x"].to_numpy(dtype=float)
    yv = frame["_y"].to_numpy(dtype=float)
    for _ in range(int(iterations)):
        idx = rng.integers(0, len(frame), size=len(frame))
        sx = pd.Series(xv[idx])
        sy = pd.Series(yv[idx])
        slope = robust_slope(sx, sy)
        if np.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return np.nan, np.nan
    low, high = np.quantile(slopes, [0.025, 0.975])
    return float(low), float(high)


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not np.any(mask):
        return pd.Series(out, index=pvalues.index)
    values = p[mask]
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1, dtype=float)
    adjusted = values[order] * len(values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    target = np.where(mask)[0][order]
    out[target] = adjusted
    return pd.Series(out, index=pvalues.index)


def tertile_labels(values: pd.Series) -> pd.Series:
    rank = values.rank(method="first", pct=True)
    return pd.cut(rank, bins=[0.0, 1 / 3, 2 / 3, 1.0], labels=["low", "medium", "high"], include_lowest=True)


def compute_per_hkl_stats(observations: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []
    for hkl, group in observations.groupby(HKL_COLUMNS, sort=False):
        h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        nonzero = group.loc[pd.to_numeric(group["coupling_sum_raw"], errors="coerce") > 0.0].copy()
        zero = group.loc[pd.to_numeric(group["coupling_sum_raw"], errors="coerce") <= 0.0].copy()
        row: dict[str, Any] = {
            "h": h,
            "k": k,
            "l": l,
            "symmetry_orbit_id": str(group["symmetry_orbit_id"].iloc[0]),
            "selection_set": str(group["selection_set"].iloc[0]),
            "include_in_primary_aggregate": bool(group["include_in_primary_aggregate"].iloc[0]),
            "reflection_strength_class": str(group["reflection_strength_class"].iloc[0]),
            "resolution_bin": str(group["resolution_bin"].iloc[0]),
            "reciprocal_direction_class": str(group["reciprocal_direction_class"].iloc[0]),
            "hkl_geometry_class": str(group["hkl_geometry_class"].iloc[0]),
            "merged_intensity_or_Fobs": float(group["merged_intensity_or_Fobs"].iloc[0]),
            "d_spacing_angstrom": float(group["d_spacing_angstrom"].iloc[0]),
            "g_norm_invA": float(group["g_norm_invA"].iloc[0]),
            "angle_to_cstar_deg": float(group["angle_to_cstar_deg"].iloc[0]),
            "nonzero_coupling_observation_count": int(len(nonzero)),
            "zero_coupling_observation_count": int(len(zero)),
            "zero_coupling_median_corrected_residual": float(zero["corrected_intensity_residual"].median()) if len(zero) else np.nan,
            "n_observations": int(len(group)),
        }
        controls = nonzero.loc[:, ["target_excitation_Eg", "partiality"]]
        for response in RESPONSE_COLUMNS:
            for xcol in TREND_X_COLUMNS:
                rho, p = spearman_with_p(nonzero[response], nonzero[xcol])
                row[f"spearman_{response}_vs_{xcol}"] = rho
                row[f"spearman_p_{response}_vs_{xcol}"] = p
                row[f"partial_spearman_{response}_vs_{xcol}_ctrl_Eg_partiality"] = partial_spearman(nonzero[response], nonzero[xcol], controls)
        slope = robust_slope(nonzero["excitation_deficit_norm"], nonzero["corrected_intensity_residual"])
        ci_low, ci_high = bootstrap_slope_ci(
            nonzero["excitation_deficit_norm"], nonzero["corrected_intensity_residual"], int(args.bootstrap_iterations), int(args.seed) + len(rows)
        )
        row[PRIMARY_SLOPE_COLUMN] = slope
        row["slope_ci95_low"] = ci_low
        row["slope_ci95_high"] = ci_high
        for subset_name, subset in [
            ("high_target_quartile", nonzero.loc[nonzero["target_excitation_Eg"] >= nonzero["target_excitation_Eg"].quantile(0.75)] if len(nonzero) else nonzero),
            ("high_partiality_quartile", nonzero.loc[nonzero["partiality"] >= nonzero["partiality"].quantile(0.75)] if len(nonzero) else nonzero),
        ]:
            row[f"n_{subset_name}"] = int(len(subset))
            row[f"spearman_corrected_residual_vs_deficit_{subset_name}"] = spearman_with_p(
                subset["corrected_intensity_residual"], subset["excitation_deficit_norm"]
            )[0]
            row[f"slope_corrected_residual_vs_deficit_{subset_name}"] = robust_slope(
                subset["excitation_deficit_norm"], subset["corrected_intensity_residual"]
            )
        for split_column, prefix in [("nonself_local_excitation_raw", "v5"), ("coupling_sum_raw", "coupling_sum")]:
            labels = tertile_labels(nonzero[split_column]) if len(nonzero) else pd.Series(dtype=object)
            for label in ["low", "medium", "high"]:
                subset = nonzero.loc[labels == label]
                row[f"n_{prefix}_{label}_tertile"] = int(len(subset))
                row[f"spearman_corrected_residual_vs_deficit_{prefix}_{label}_tertile"] = spearman_with_p(
                    subset["corrected_intensity_residual"], subset["excitation_deficit_norm"]
                )[0]
                row[f"slope_corrected_residual_vs_deficit_{prefix}_{label}_tertile"] = robust_slope(
                    subset["excitation_deficit_norm"], subset["corrected_intensity_residual"]
                )
        rows.append(row)

        for xcol in ["excitation_deficit_norm", "nonself_local_excitation_raw", "coupling_sum_raw"]:
            qlabels = pd.qcut(nonzero[xcol].rank(method="first"), q=min(5, max(1, len(nonzero))), labels=False, duplicates="drop") if len(nonzero) else []
            if len(nonzero):
                work = nonzero.assign(_quintile=np.asarray(qlabels, dtype=float))
                for quintile, subset in work.groupby("_quintile", sort=True):
                    quintile_rows.append(
                        {
                            "h": h,
                            "k": k,
                            "l": l,
                            "quintile_variable": xcol,
                            "quintile_index": int(quintile) + 1,
                            "n": int(len(subset)),
                            "x_min": float(subset[xcol].min()),
                            "x_max": float(subset[xcol].max()),
                            "median_I_over_merged": float(subset["I_over_merged"].median()),
                            "median_I_over_hkl_median": float(subset["I_over_hkl_median"].median()),
                            "median_corrected_intensity_residual": float(subset["corrected_intensity_residual"].median()),
                        }
                    )
    stats = pd.DataFrame.from_records(rows)
    p_col = "spearman_p_corrected_intensity_residual_vs_excitation_deficit_norm"
    if p_col in stats.columns:
        stats["fdr_q_corrected_residual_vs_deficit"] = bh_fdr(stats[p_col])
    return stats, pd.DataFrame.from_records(quintile_rows)


def build_orientation_extremes(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    cases = [
        ("most_negative_Dexc_norm", "coupled_excitation_imbalance_norm", "idxmin"),
        ("most_positive_Dexc_norm", "coupled_excitation_imbalance_norm", "idxmax"),
        ("smallest_excitation_deficit_norm", "excitation_deficit_norm", "idxmin"),
        ("largest_excitation_deficit_norm", "excitation_deficit_norm", "idxmax"),
        ("lowest_v5", "nonself_local_excitation_raw", "idxmin"),
        ("highest_v5", "nonself_local_excitation_raw", "idxmax"),
        ("lowest_corrected_intensity_residual", "corrected_intensity_residual", "idxmin"),
        ("highest_corrected_intensity_residual", "corrected_intensity_residual", "idxmax"),
    ]
    for _hkl, group in observations.groupby(HKL_COLUMNS, sort=False):
        for label, column, op in cases:
            values = pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.dropna().empty:
                continue
            idx = values.idxmin() if op == "idxmin" else values.idxmax()
            row = group.loc[idx].copy()
            row["orientation_case"] = label
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def equal_area_projection(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    vec = np.column_stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float), z.to_numpy(dtype=float)])
    norm = np.linalg.norm(vec, axis=1)
    unit = np.divide(vec, norm[:, None], out=np.zeros_like(vec), where=norm[:, None] > 0.0)
    unit[:, 2] = np.abs(unit[:, 2])
    factor = np.sqrt(2.0 / np.clip(1.0 + unit[:, 2], 1e-12, None))
    return factor * unit[:, 0], factor * unit[:, 1]


def plot_one_hkl(group: pd.DataFrame, out_dir: Path) -> Path:
    h, k, l = hkl_tuple_from_row(group.iloc[0])
    stem = f"{group['reflection_strength_class'].iloc[0]}_{group['resolution_bin'].iloc[0]}_{hkl_slug(h, k, l)}"
    path = out_dir / f"{stem}_broad_dexc_orientation.png"
    xproj, yproj = equal_area_projection(group["continuous_uvw_x"], group["continuous_uvw_y"], group["continuous_uvw_z"])
    nonzero = group.loc[group["coupling_sum_raw"] > 0.0].copy()
    labels = tertile_labels(nonzero["nonself_local_excitation_raw"]) if len(nonzero) else pd.Series(dtype=object)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(group["excitation_deficit_norm"], group["corrected_intensity_residual"], s=12, alpha=0.7)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("excitation_deficit_norm")
    ax.set_ylabel("corrected intensity residual")

    ax = axes[0, 1]
    ax.scatter(group["nonself_local_excitation_raw"], group["corrected_intensity_residual"], s=12, alpha=0.7)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("v5 nonself_local_excitation_raw")
    ax.set_ylabel("corrected intensity residual")

    ax = axes[0, 2]
    sc = ax.scatter(xproj, yproj, c=group["excitation_deficit_norm"], s=12, alpha=0.8, cmap="coolwarm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("equal-area x")
    ax.set_ylabel("equal-area y")
    fig.colorbar(sc, ax=ax, label="excitation_deficit_norm")

    ax = axes[1, 0]
    sc = ax.scatter(xproj, yproj, c=group["corrected_intensity_residual"], s=12, alpha=0.8, cmap="coolwarm")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("equal-area x")
    ax.set_ylabel("equal-area y")
    fig.colorbar(sc, ax=ax, label="corrected residual")

    ax = axes[1, 1]
    if len(nonzero):
        qlabels = pd.qcut(nonzero["excitation_deficit_norm"].rank(method="first"), q=min(5, len(nonzero)), labels=False, duplicates="drop")
        trend = nonzero.assign(_q=qlabels).groupby("_q", sort=True)["corrected_intensity_residual"].median()
        ax.plot(np.arange(1, len(trend) + 1), trend.to_numpy(dtype=float), marker="o")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("excitation-deficit quintile")
    ax.set_ylabel("median corrected residual")

    ax = axes[1, 2]
    if len(nonzero):
        for label in ["low", "medium", "high"]:
            sub = nonzero.loc[labels == label]
            if sub.empty:
                continue
            ax.scatter(sub["excitation_deficit_norm"], sub["corrected_intensity_residual"], s=12, alpha=0.7, label=f"{label} v5")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("excitation_deficit_norm")
    ax.set_ylabel("corrected residual")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"HKL ({h}, {k}, {l}) | {group['reflection_strength_class'].iloc[0]} | {group['reciprocal_direction_class'].iloc[0]}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_plots(observations: pd.DataFrame, out_dir: Path, per_hkl_stats: pd.DataFrame) -> list[Path]:
    hkl_dir = out_dir / "plots" / "per_hkl"
    agg_dir = out_dir / "plots" / "aggregate"
    hkl_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)
    paths = [plot_one_hkl(group, hkl_dir) for _hkl, group in observations.groupby(HKL_COLUMNS, sort=False)]
    primary = per_hkl_stats.loc[per_hkl_stats["include_in_primary_aggregate"].astype(bool)].copy()
    for xcol, xlabel in [
        ("merged_intensity_or_Fobs", "merged intensity"),
        ("d_spacing_angstrom", "d spacing (A)"),
        ("angle_to_cstar_deg", "angle to c* (deg)"),
        ("nonzero_coupling_observation_count", "nonzero-coupling observation count"),
    ]:
        if xcol not in primary.columns or primary.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        ax.scatter(primary[xcol], primary[PRIMARY_SLOPE_COLUMN], s=28, alpha=0.8)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(PRIMARY_SLOPE_COLUMN)
        path = agg_dir / f"primary_slope_vs_{xcol}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def build_symmetry_mate_comparison(per_hkl: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    pairs = selected.loc[selected["symmetry_pair_id"].fillna("") != "", [*HKL_COLUMNS, "symmetry_pair_id", "symmetry_pair_role"]].copy()
    if pairs.empty:
        return pd.DataFrame()
    joined = pairs.merge(per_hkl, on=HKL_COLUMNS, how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for pair_id, group in joined.groupby("symmetry_pair_id", sort=False):
        if len(group) < 2:
            continue
        first, second = group.iloc[0], group.iloc[1]
        rows.append(
            {
                "symmetry_pair_id": pair_id,
                "h1": int(first.h),
                "k1": int(first.k),
                "l1": int(first.l),
                "h2": int(second.h),
                "k2": int(second.k),
                "l2": int(second.l),
                "slope_1": first.get(PRIMARY_SLOPE_COLUMN, np.nan),
                "slope_2": second.get(PRIMARY_SLOPE_COLUMN, np.nan),
                "slope_difference_1_minus_2": first.get(PRIMARY_SLOPE_COLUMN, np.nan) - second.get(PRIMARY_SLOPE_COLUMN, np.nan),
                "rho_1": first.get("spearman_corrected_intensity_residual_vs_excitation_deficit_norm", np.nan),
                "rho_2": second.get("spearman_corrected_intensity_residual_vs_excitation_deficit_norm", np.nan),
            }
        )
    return pd.DataFrame.from_records(rows)


def aggregate_summary(per_hkl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = per_hkl.loc[per_hkl["include_in_primary_aggregate"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for group_col in ["reflection_strength_class", "resolution_bin", "reciprocal_direction_class", "hkl_geometry_class"]:
        if group_col not in primary.columns:
            continue
        for value, group in primary.groupby(group_col, sort=False):
            rows.append(
                {
                    "grouping": group_col,
                    "level": str(value),
                    "n_hkls": int(len(group)),
                    "median_slope": float(group[PRIMARY_SLOPE_COLUMN].median()),
                    "mean_slope": float(group[PRIMARY_SLOPE_COLUMN].mean()),
                    "median_spearman_corrected_residual_vs_deficit": float(
                        group["spearman_corrected_intensity_residual_vs_excitation_deficit_norm"].median()
                    ),
                }
            )
    model = fit_exploratory_model(primary)
    return pd.DataFrame.from_records(rows), model


def fit_exploratory_model(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty or PRIMARY_SLOPE_COLUMN not in primary.columns:
        return pd.DataFrame()
    frame = primary.copy()
    frame["log_abs_merged_intensity"] = np.log1p(np.abs(frame["merged_intensity_or_Fobs"].to_numpy(dtype=float)))
    predictors = frame.loc[:, ["log_abs_merged_intensity", "g_norm_invA", "angle_to_cstar_deg", "nonzero_coupling_observation_count"]].copy()
    for col in ["reciprocal_direction_class", "reflection_strength_class", "resolution_bin"]:
        dummies = pd.get_dummies(frame[col], prefix=col, drop_first=True)
        predictors = pd.concat([predictors, dummies], axis=1)
    x = predictors.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(frame[PRIMARY_SLOPE_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = y.notna() & ~x.isna().any(axis=1)
    if int(mask.sum()) < 5:
        return pd.DataFrame()
    xmat = np.column_stack([np.ones(int(mask.sum())), x.loc[mask].to_numpy(dtype=float)])
    yv = y.loc[mask].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(xmat, yv, rcond=None)
    names = ["intercept", *x.columns.tolist()]
    return pd.DataFrame({"term": names, "coefficient": beta, "model_note": "Exploratory OLS on per-HKL slopes; do not interpret causally."})


def output_observation_columns(table: pd.DataFrame) -> list[str]:
    columns = [
        "source_filename",
        "event",
        "h",
        "k",
        "l",
        "symmetry_orbit_id",
        "selection_set",
        "include_in_primary_aggregate",
        "reflection_strength_class",
        "resolution_bin",
        "merged_intensity_or_Fobs",
        "d_spacing_angstrom",
        "g_norm_invA",
        "reciprocal_direction_class",
        "angle_to_cstar_deg",
        "abs_g_cstar_component_over_g",
        "basal_azimuth_deg",
        "continuous_uvw_x",
        "continuous_uvw_y",
        "continuous_uvw_z",
        "closest_uvw",
        "closest_zone_axis_angle_deg",
        "I_unmerged",
        "observation_intensity",
        "sigma",
        "partiality",
        "target_excitation_Eg",
        "target_excitation_error",
        "nonself_local_excitation_raw",
        "coupling_sum_raw",
        "neighbor_excitation_weighted_mean",
        "coupled_excitation_imbalance_raw",
        "coupled_excitation_imbalance_norm",
        "excitation_deficit_norm",
        "I_over_merged",
        "I_over_hkl_median",
        "I_robust_z_within_hkl",
        "corrected_intensity_residual",
    ]
    return [column for column in columns if column in table.columns]


def print_selection_quota_tables(primary: pd.DataFrame, selected: pd.DataFrame, availability: pd.DataFrame) -> None:
    print("\nCandidate count by resolution x strength x angle tertile (4/mmm representatives):")
    print(
        availability.pivot_table(
            index=["resolution_bin", "reflection_strength_class"],
            columns="angle_tertile",
            values="candidate_count_4mmm_orbit_representatives",
            aggfunc="sum",
            fill_value=0,
        ).to_string()
    )
    print("\nPlanned quota per resolution x strength x angle tertile:")
    print(
        availability.pivot_table(
            index=["resolution_bin", "reflection_strength_class"],
            columns="angle_tertile",
            values="cell_target",
            aggfunc="sum",
            fill_value=0,
        ).to_string()
    )
    print("\nSelected primary HKL counts by resolution bin:")
    print(primary.groupby("resolution_bin").size().to_string())
    print("\nSelected primary HKL counts by resolution x strength:")
    print(primary.groupby(["resolution_bin", "reflection_strength_class"]).size().unstack(fill_value=0).to_string())
    print("\nSelected primary HKL counts by resolution x angle tertile:")
    print(primary.groupby(["resolution_bin", "angle_tertile"]).size().unstack(fill_value=0).to_string())
    print("\nSelected primary HKL counts by reciprocal-direction class:")
    print(primary.groupby("reciprocal_direction_class").size().to_string())
    print("\nSelected primary HKL counts by HKL geometry class:")
    print(primary.groupby("hkl_geometry_class").size().to_string())
    counts = primary_geometry_counts(primary)
    print("\nSelected primary geometry quota counts:")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print("\nSelected primary HKLs:")
    display_columns = [
        "h",
        "k",
        "l",
        "selection_order",
        "reflection_strength_class",
        "resolution_bin",
        "angle_tertile",
        "reciprocal_direction_class",
        "hkl_geometry_class",
        "selection_substitution",
    ]
    print(primary.loc[:, [column for column in display_columns if column in primary.columns]].to_string(index=False))
    if not selected.empty and "selection_set" in selected.columns:
        print("\nSelected signed HKL counts including symmetry-reproducibility set:")
        print(selected.groupby("selection_set").size().to_string())


def write_readme(out_dir: Path, args: argparse.Namespace, intensity_choice: IntensityChoice, strength_meta: dict[str, Any], plot_count: int) -> None:
    lines = [
        "# V5 Excitation-Imbalance Orientation Broad-HKL Diagnostic",
        "",
        "Observation-level diagnostic only: no filtering, merging, Partialator, or refinement is run by this script.",
        "",
        "The excitation-imbalance columns are continuous excitation-contrast descriptors. Their sign is not interpreted as proven gain or loss.",
        "",
        "## Inputs",
        "",
        f"- v5 scores: `{args.v5_scores}`",
        f"- accepted survivor table: `{args.accepted}`",
        f"- full-data merged intensity table: `{args.strength_table}`",
        f"- stream for exact orientation/continuous UVW: `{args.stream}`",
        "",
        "## Intensity Response",
        "",
        f"- observation intensity used: `{intensity_choice.column}`",
        f"- sigma column: `{intensity_choice.sigma_column}`",
        f"- scale covariates used in robust residual model: `{', '.join(intensity_choice.scale_columns) if intensity_choice.scale_columns else 'none found'}`",
        f"- note: {intensity_choice.note}",
        "",
        "## Strength Definition",
        "",
        f"- metric: {strength_meta['strength_metric']}",
        f"- negative merged intensities: {strength_meta['negative_merged_intensity_hkls']:,}",
        f"- handling: {strength_meta['negative_intensity_handling']}",
        "- weak/medium/strong classes are assigned from signed merged-intensity percentiles within resolution bins.",
        "",
        "## Imbalance Formulas",
        "",
        f"- `coupling_sum_raw = {args.coupling_column} = sum C(g-q)` from the existing v5 CSV.",
        f"- `neighbor_excitation_weighted_mean = {args.score_column} / coupling_sum_raw`; set to 0 when coupling sum is 0.",
        "- `coupled_excitation_imbalance_norm = neighbor_excitation_weighted_mean - target_excitation_Eg`; set to 0 when coupling sum is 0.",
        "- `excitation_deficit_norm = target_excitation_Eg - neighbor_excitation_weighted_mean`; set to 0 when coupling sum is 0.",
        "- `coupled_excitation_imbalance_raw = nonself_local_excitation_raw - target_excitation_Eg * coupling_sum_raw`.",
        "",
        "## Outputs",
        "",
    ]
    for name in [
        "candidate_hkl_selection_audit.csv",
        "selected_hkl_strengths_and_geometry.csv",
        "diagnostic_observations.csv",
        "per_hkl_dexc_intensity_stats.csv",
        "per_hkl_quintile_trends.csv",
        "orientation_extremes.csv",
        "symmetry_mate_comparison.csv",
        "aggregate_trend_summary.csv",
        "exploratory_reflection_level_model.csv",
        "run_metadata.json",
    ]:
        lines.append(f"- `{out_dir / name}`")
    lines.append(f"- plots: `{out_dir / 'plots'}` ({plot_count} PNG files)")
    (out_dir / "README_v5_excitation_imbalance_orientation_broad_hkl.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {args.out_dir}")

    log("Parsing reciprocal cell from stream header")
    cell = parse_stream_unit_cell(args.stream)
    reciprocal = reciprocal_matrix_from_cell(cell)

    log("Loading full-data merged intensity table")
    strength, strength_meta = read_crystfel_hkl_strength(args.strength_table)
    log(f"Merged HKLs loaded: {len(strength):,}")

    log("Loading accepted survivor table")
    accepted, intensity_choice, accepted_meta = load_accepted_table(args.accepted, int(args.chunksize))
    log(f"Accepted exact keys loaded: {len(accepted):,}")

    log("Joining accepted observations to existing v5 scores")
    accepted_v5, join_meta = load_accepted_v5_join(accepted, args)
    accepted_v5 = add_dexc_columns(accepted_v5, args)
    log(f"Accepted v5 observations loaded: {len(accepted_v5):,}")

    log("Building broad candidate HKL pool and descriptors")
    candidates = build_candidate_summary(accepted_v5, strength, reciprocal, args)
    primary, audit, availability = select_primary_hkls(candidates, int(args.target_primary_hkls), int(args.resolution_bins))
    sym_pairs = select_symmetry_pairs(candidates, primary, int(args.symmetry_pairs))
    selected = combine_selected_hkls(primary, sym_pairs)
    audit["selected_any"] = pd.MultiIndex.from_frame(audit.loc[:, HKL_COLUMNS]).isin(pd.MultiIndex.from_frame(selected.loc[:, HKL_COLUMNS]))
    audit.loc[audit["selected_any"] & ~audit["selected_primary"], "selection_reason"] = "symmetry_reproducibility_pair"
    availability.to_csv(args.out_dir / "selection_quota_cell_availability.csv", index=False)
    substitutions = audit.attrs.get("selection_substitutions", [])
    if substitutions:
        pd.DataFrame(substitutions).to_csv(args.out_dir / "primary_hkl_selection_substitutions.csv", index=False)
    audit.to_csv(args.out_dir / "candidate_hkl_selection_audit.csv", index=False)
    selected.to_csv(args.out_dir / "selected_hkl_strengths_and_geometry.csv", index=False)

    selected_hkls = [hkl_tuple_from_row(row) for row in selected.itertuples(index=False)]
    log(f"Selected signed HKLs: primary={int(selected['include_in_primary_aggregate'].sum())}, total_with_symmetry={len(selected_hkls)}")
    print_selection_quota_tables(primary, selected, availability)

    if args.selection_only:
        metadata = {
            "command": " ".join(sys.argv),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_only": True,
            "inputs": {
                "stream": str(args.stream),
                "v5_scores": str(args.v5_scores),
                "accepted": str(args.accepted),
                "strength_table": str(args.strength_table),
            },
            "unit_cell": cell.__dict__,
            "accepted_table": accepted_meta,
            "v5_join": join_meta,
            "strength": strength_meta,
            "selection": {
                "target_primary_hkls": int(args.target_primary_hkls),
                "selected_primary_hkls": int(selected["include_in_primary_aggregate"].sum()),
                "selected_total_signed_hkls": int(len(selected)),
                "symmetry_pairs_requested": int(args.symmetry_pairs),
                "symmetry_pairs_selected": int(sym_pairs["symmetry_pair_id"].nunique()) if not sym_pairs.empty else 0,
                "primary_geometry_counts": primary_geometry_counts(primary),
                "substitution_count": int(len(substitutions)),
            },
            "note": "Selection-only run: stopped before exact orientation loading, observation diagnostics, and plotting.",
        }
        (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nSelection-only outputs written to: {args.out_dir}")
        return 0

    log("Preparing selected observation table")
    selected_obs = accepted_v5.loc[hkl_mask(accepted_v5, selected_hkls)].copy()
    selected_metadata_cols = [
        *HKL_COLUMNS,
        "symmetry_orbit_id",
        "selection_set",
        "include_in_primary_aggregate",
        "reflection_strength_class",
        "resolution_bin",
        "angle_tertile",
        "basal_azimuth_sector",
        "merged_intensity_or_Fobs",
        "d_spacing_angstrom",
        "g_norm_invA",
        "reciprocal_direction_class",
        "angle_to_cstar_deg",
        "abs_g_cstar_component_over_g",
        "basal_azimuth_deg",
        "hkl_geometry_class",
        "hkl_family",
        "is_axial",
        "is_00l",
        "is_one_index_zero",
        "is_fully_general",
        "selection_order",
        "selection_reason",
        "selection_substitution",
        "symmetry_pair_id",
        "symmetry_pair_role",
    ]
    selected_obs = selected_obs.merge(selected.loc[:, selected_metadata_cols], on=HKL_COLUMNS, how="left", validate="many_to_one")
    selected_obs = add_intensity_responses(selected_obs, selected, intensity_choice.scale_columns)
    selected_obs = add_orientation_columns(selected_obs, args.stream, int(args.uvw_max))
    selected_obs.loc[:, output_observation_columns(selected_obs)].to_csv(args.out_dir / "diagnostic_observations.csv", index=False)

    log("Computing per-HKL trend diagnostics")
    per_hkl, quintiles = compute_per_hkl_stats(selected_obs, args)
    per_hkl.to_csv(args.out_dir / "per_hkl_dexc_intensity_stats.csv", index=False)
    quintiles.to_csv(args.out_dir / "per_hkl_quintile_trends.csv", index=False)
    extremes = build_orientation_extremes(selected_obs)
    extremes.loc[:, [column for column in ["orientation_case", *output_observation_columns(extremes), "continuous_uvw"] if column in extremes.columns]].to_csv(
        args.out_dir / "orientation_extremes.csv", index=False
    )
    sym_comp = build_symmetry_mate_comparison(per_hkl, selected)
    sym_comp.to_csv(args.out_dir / "symmetry_mate_comparison.csv", index=False)
    aggregate, model = aggregate_summary(per_hkl)
    aggregate.to_csv(args.out_dir / "aggregate_trend_summary.csv", index=False)
    model.to_csv(args.out_dir / "exploratory_reflection_level_model.csv", index=False)

    log("Writing broad diagnostic plots")
    plot_paths = make_plots(selected_obs, args.out_dir, per_hkl)
    write_readme(args.out_dir, args, intensity_choice, strength_meta, len(plot_paths))
    metadata = {
        "command": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stream": str(args.stream),
            "v5_scores": str(args.v5_scores),
            "accepted": str(args.accepted),
            "strength_table": str(args.strength_table),
        },
        "unit_cell": cell.__dict__,
        "accepted_table": accepted_meta,
        "v5_join": join_meta,
        "strength": strength_meta,
        "selection": {
            "target_primary_hkls": int(args.target_primary_hkls),
            "selected_primary_hkls": int(selected["include_in_primary_aggregate"].sum()),
            "selected_total_signed_hkls": int(len(selected)),
            "symmetry_pairs_requested": int(args.symmetry_pairs),
            "symmetry_pairs_selected": int(sym_pairs["symmetry_pair_id"].nunique()) if not sym_pairs.empty else 0,
        },
        "note": "Dexc/deficit columns are excitation-contrast descriptors only, not proven gain/loss scores.",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nWrote outputs to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
