#!/usr/bin/env python3
"""Extract v5 low/high accepted observations and recover nearest/continuous UVW."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_same_hkl_low_high_orientation_examples import (  # noqa: E402
    KEY_COLUMNS,
    load_stream_orientation_for_selected,
    normalize_key_columns,
)
from oridyn.axis_prediction import unique_zone_axes  # noqa: E402
from oridyn.geometry import axis_angle_deg, beam_in_direct_coordinates, triplet_label  # noqa: E402
from oridyn.stream_parser import STREAM_MATRIX_COLUMNS, reciprocal_matrix_from_row  # noqa: E402


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
DEFAULT_OUT_DIR = BASE / "oridyn_v5_allscore_filter_and_50split_20_0p3_20260706" / "low_high_orientation_uvw_examples"
DEFAULT_HKLS = [
    (0, 4, 0),
    (10, -7, 3),
    (0, 27, 5),
    (6, 6, 2),
    (8, 4, 0),
    (9, 3, 0),
    (8, 6, 0),
    (7, 5, 0),
]
SCORE_COLUMN = "nonself_local_excitation_raw"
OUTPUT_COLUMNS = [
    "h",
    "k",
    "l",
    "group",
    "rank",
    "score",
    "source_filename",
    "event",
    "closest_uvw",
    "exact_uvw_or_continuous_uvw",
    "uvw_angle_deg",
    "sg_target",
    "target_excitation_Eg",
    "partiality",
    "I_unmerged",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--v5-scores", type=Path, default=DEFAULT_V5_SCORES)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-name", default="v5_low_high_orientation_uvw_examples.csv")
    parser.add_argument("--n-low", type=int, default=3)
    parser.add_argument("--n-high", type=int, default=3)
    parser.add_argument("--uvw-max", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    for label, path in [("--stream", args.stream), ("--v5-scores", args.v5_scores), ("--accepted", args.accepted)]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    if args.n_low < 1 or args.n_high < 1:
        raise SystemExit("--n-low and --n-high must be >= 1")
    if args.uvw_max < 1:
        raise SystemExit("--uvw-max must be >= 1")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be >= 1")
    return args


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def hkl_mask(table: pd.DataFrame, hkls: list[tuple[int, int, int]]) -> np.ndarray:
    wanted = pd.MultiIndex.from_tuples(hkls, names=["h", "k", "l"])
    return pd.MultiIndex.from_frame(table.loc[:, ["h", "k", "l"]]).isin(wanted)


def load_selected_hkl_rows(path: Path, usecols: list[str], hkls: list[tuple[int, int, int]], chunksize: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        work = normalize_key_columns(chunk)
        work = work.loc[hkl_mask(work, hkls)].copy()
        if not work.empty:
            chunks.append(work)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def load_accepted_subset(path: Path, hkls: list[tuple[int, int, int]], chunksize: int) -> pd.DataFrame:
    header = read_header(path)
    missing = [column for column in KEY_COLUMNS if column not in header]
    if missing:
        raise SystemExit(f"Accepted table is missing exact-key column(s): {missing}")
    usecols = [*KEY_COLUMNS, *[column for column in ["partiality", "I_unmerged"] if column in header]]
    accepted = load_selected_hkl_rows(path, usecols, hkls, chunksize)
    for column in ["partiality", "I_unmerged"]:
        if column not in accepted.columns:
            accepted[column] = np.nan
        accepted[column] = pd.to_numeric(accepted[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return accepted.drop_duplicates(KEY_COLUMNS, keep="first")


def load_v5_subset(path: Path, hkls: list[tuple[int, int, int]], chunksize: int) -> pd.DataFrame:
    header = read_header(path)
    required = [*KEY_COLUMNS, "sg_target", "target_excitation_Eg", SCORE_COLUMN]
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"V5 score table is missing required column(s): {missing}")
    scores = load_selected_hkl_rows(path, required, hkls, chunksize)
    for column in ["sg_target", "target_excitation_Eg", SCORE_COLUMN]:
        scores[column] = pd.to_numeric(scores[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return scores.dropna(subset=[SCORE_COLUMN]).drop_duplicates(KEY_COLUMNS, keep="first")


def select_tail_examples(joined: pd.DataFrame, hkls: list[tuple[int, int, int]], n_low: int, n_high: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for h, k, l in hkls:
        subset = joined.loc[(joined["h"] == h) & (joined["k"] == k) & (joined["l"] == l)].copy()
        if subset.empty:
            raise SystemExit(f"No accepted v5-scored observations found for signed HKL {h} {k} {l}")
        low = subset.sort_values([SCORE_COLUMN, "source_filename", "event"], kind="mergesort").head(int(n_low)).copy()
        high = subset.sort_values([SCORE_COLUMN, "source_filename", "event"], ascending=[False, True, True], kind="mergesort").head(
            int(n_high)
        ).copy()
        low["group"] = "low"
        high["group"] = "high"
        low["rank"] = np.arange(1, len(low) + 1, dtype=int)
        high["rank"] = np.arange(1, len(high) + 1, dtype=int)
        rows.extend([low, high])
    return pd.concat(rows, ignore_index=True)


def continuous_uvw_label(values: np.ndarray) -> str:
    return f"[{values[0]:.6g} {values[1]:.6g} {values[2]:.6g}]"


def add_uvw_columns(selected: pd.DataFrame, stream: Path, uvw_max: int) -> pd.DataFrame:
    orientation = load_stream_orientation_for_selected(stream, selected)
    if len(orientation) != len(selected):
        found = set(tuple(row) for row in orientation.loc[:, KEY_COLUMNS].itertuples(index=False, name=None))
        missing = [tuple(row) for row in selected.loc[:, KEY_COLUMNS].itertuples(index=False, name=None) if tuple(row) not in found]
        raise SystemExit(f"Stream orientation lookup missed {len(missing)} selected exact key(s); first missing: {missing[:3]}")

    joined = selected.merge(
        orientation.loc[:, [*KEY_COLUMNS, *STREAM_MATRIX_COLUMNS]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    axes = unique_zone_axes(int(uvw_max))
    closest: list[str] = []
    continuous: list[str] = []
    angles: list[float] = []
    for _, row in joined.iterrows():
        reciprocal = reciprocal_matrix_from_row(row)
        beam_uvw = beam_in_direct_coordinates(reciprocal)
        best_axis = min(axes, key=lambda axis: axis_angle_deg(reciprocal, axis))
        closest.append(triplet_label(best_axis))
        continuous.append(continuous_uvw_label(beam_uvw))
        angles.append(float(axis_angle_deg(reciprocal, best_axis)))
    joined["closest_uvw"] = closest
    joined["exact_uvw_or_continuous_uvw"] = continuous
    joined["uvw_angle_deg"] = angles
    return joined


def main() -> int:
    args = parse_args()
    accepted = load_accepted_subset(args.accepted, DEFAULT_HKLS, args.chunksize)
    v5 = load_v5_subset(args.v5_scores, DEFAULT_HKLS, args.chunksize)
    joined = v5.merge(accepted, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    selected = select_tail_examples(joined, DEFAULT_HKLS, args.n_low, args.n_high)
    selected = add_uvw_columns(selected, args.stream, args.uvw_max)
    selected["score"] = selected[SCORE_COLUMN]

    out = selected.loc[:, OUTPUT_COLUMNS].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False, max_colwidth=72))
    print(f"output_csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())