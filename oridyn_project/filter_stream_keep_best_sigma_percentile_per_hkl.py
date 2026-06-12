#!/usr/bin/env python3
"""Keep the lowest-risk OriDyn observations per mmm-grouped HKL."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STREAM = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "MFM300-VIII_cut_20-0_3.stream"
)
DEFAULT_SCORES = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "oridyn_row_self_exp_notail/reflection_scores.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/bubl3932/files/MFM300_VIII/"
    "MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/"
    "MFM300-VIII_cut_20-0_3_oridyn_keep_best5pct_per_mmm_hkl.stream"
)

REQUIRED_COLUMNS = ("frame", "h", "k", "l")
OPTIONAL_REPORT_COLUMNS = ("S_dyn_geom", "frame_number", "event", "d_angstrom")


@dataclass(frozen=True)
class ScoreSelection:
    kept_keys: set[tuple[int, int, int, int]]
    summary: pd.DataFrame
    score_rows_read: int
    score_rows_usable: int
    kept_score_rows: int
    duplicate_score_keys: int
    affected_groups: int
    unaffected_groups: int
    affected_observations: int
    condition_description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a CrystFEL stream by keeping only the lowest-score "
            "observations within each mmm-grouped HKL."
        )
    )
    parser.add_argument("--stream", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-fraction", type=float, default=0.05)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--grouping", choices=["mmm"], default="mmm")
    parser.add_argument("--score-column", default="sigma_dyn_rel")
    parser.add_argument(
        "--condition-column",
        default=None,
        help=(
            "Optional group-level summary column used to decide which HKLs are filtered, "
            "for example sigma_p95 or sigma_median. Omit to filter all groups."
        ),
    )
    parser.add_argument(
        "--condition-op",
        choices=["ge", "gt", "le", "lt", "eq", "ne"],
        default=None,
        help="Comparison operator for --condition-column.",
    )
    parser.add_argument(
        "--condition-value",
        type=float,
        default=None,
        help="Numeric threshold for --condition-column.",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path for the per-HKL filter summary CSV.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print detected score columns and exit without filtering.",
    )
    return parser.parse_args()


def summary_path_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_filter_summary.csv")


def read_score_columns(scores_path: Path) -> list[str]:
    return list(pd.read_csv(scores_path, nrows=0).columns)


def inspect_score_columns(columns: list[str], score_column: str) -> None:
    available = set(columns)
    print("Available score columns:")
    print(", ".join(columns))
    print()
    print("Detected columns:")
    print(f"  identity: {', '.join(column for column in REQUIRED_COLUMNS if column in available)}")
    print(f"  score/rank column: {score_column if score_column in available else 'MISSING: ' + score_column}")
    optionals = [column for column in OPTIONAL_REPORT_COLUMNS if column in available]
    print(f"  optional report columns: {', '.join(optionals) if optionals else 'none'}")
    group_columns = ["n_obs", "sigma_min", "sigma_median", "sigma_p95", "sigma_max"]
    if "d_angstrom" in available:
        group_columns.append("d_angstrom_mean")
    if "S_dyn_geom" in available:
        group_columns.extend(
            [
                "S_dyn_geom_min",
                "S_dyn_geom_median",
                "S_dyn_geom_p95",
                "S_dyn_geom_max",
                "S_dyn_geom_spread_p95_minus_median",
                "S_dyn_geom_spread_max_minus_median",
            ]
        )
    print(f"  supported condition columns after grouping: {', '.join(group_columns)}")
    print()


def required_usecols(columns: list[str], score_column: str) -> list[str]:
    required = [*REQUIRED_COLUMNS, score_column]
    missing = [column for column in required if column not in columns]
    if missing:
        raise SystemExit(f"Scores table is missing required column(s): {missing}")
    usecols = [*required, *[column for column in OPTIONAL_REPORT_COLUMNS if column in columns]]
    return list(dict.fromkeys(usecols))


def load_scores(scores_path: Path, columns: list[str], score_column: str, chunksize: int) -> tuple[pd.DataFrame, int]:
    usecols = required_usecols(columns, score_column)
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_usable = 0
    print(f"Reading score table in chunks of {chunksize:,} row(s)...")

    for i, chunk in enumerate(pd.read_csv(scores_path, usecols=usecols, chunksize=chunksize), start=1):
        rows_read += len(chunk)
        table = pd.DataFrame(
            {
                "frame": pd.to_numeric(chunk["frame"], errors="coerce"),
                "h": pd.to_numeric(chunk["h"], errors="coerce"),
                "k": pd.to_numeric(chunk["k"], errors="coerce"),
                "l": pd.to_numeric(chunk["l"], errors="coerce"),
                "score": pd.to_numeric(chunk[score_column], errors="coerce"),
            }
        )
        if "d_angstrom" in chunk:
            table["d_angstrom"] = pd.to_numeric(chunk["d_angstrom"], errors="coerce")
        if "S_dyn_geom" in chunk:
            table["S_dyn_geom"] = pd.to_numeric(chunk["S_dyn_geom"], errors="coerce")

        table = table.dropna(subset=["frame", "h", "k", "l", "score"])
        table = table.loc[np.isfinite(table["score"].to_numpy(dtype=float))]
        table = table.astype({"frame": "int64", "h": "int32", "k": "int32", "l": "int32"})
        table["H"] = table["h"].abs().astype("int32")
        table["K"] = table["k"].abs().astype("int32")
        table["L"] = table["l"].abs().astype("int32")
        chunks.append(table)
        rows_usable += len(table)

        if i == 1 or i % 5 == 0:
            print(f"  chunks read: {i:>3d}, rows read: {rows_read:,}, usable rows: {rows_usable:,}")

    if not chunks:
        raise SystemExit("No usable score rows were found.")

    scores = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(scores):,} usable score row(s).")
    print()
    return scores, rows_read


def keep_count(n_obs: int, keep_fraction: float, min_keep: int) -> int:
    if n_obs <= 0:
        return 0
    return min(n_obs, max(int(min_keep), int(math.ceil(float(n_obs) * float(keep_fraction)))))


def select_lowest_risk_observations(
    scores: pd.DataFrame,
    rows_read: int,
    keep_fraction: float,
    min_keep: int,
    condition_column: str | None,
    condition_op: str | None,
    condition_value: float | None,
) -> ScoreSelection:
    if not 0.0 < keep_fraction <= 1.0:
        raise SystemExit("--keep-fraction must be > 0 and <= 1.")
    if min_keep < 0:
        raise SystemExit("--min-keep must be >= 0.")

    duplicate_keys = int(scores.duplicated(["frame", "h", "k", "l"], keep=False).sum())
    if duplicate_keys:
        print(f"WARNING: found {duplicate_keys:,} score rows with duplicate (frame,h,k,l) keys.")
        print("         The stream filter uses a set of keys, so duplicate score rows collapse to one identity.")
        print()

    group_summary = build_group_summary(scores)
    group_summary, condition_description = apply_group_condition(
        group_summary,
        condition_column=condition_column,
        condition_op=condition_op,
        condition_value=condition_value,
    )
    group_summary["n_keep"] = np.where(
        group_summary["was_conditionally_filtered"],
        group_summary["n_obs"].map(lambda n: keep_count(int(n), keep_fraction, min_keep)),
        group_summary["n_obs"],
    ).astype(int)

    ranked = scores.merge(group_summary[["H", "K", "L", "n_keep"]], on=["H", "K", "L"], how="left")
    ranked = ranked.sort_values(["H", "K", "L", "score", "frame", "h", "k", "l"], kind="mergesort")
    ranked["_rank_in_group"] = ranked.groupby(["H", "K", "L"], sort=False).cumcount()
    kept = ranked.loc[ranked["_rank_in_group"] < ranked["n_keep"]].copy()

    key_columns = ["frame", "h", "k", "l"]
    kept_keys = set(kept.loc[:, key_columns].itertuples(index=False, name=None))
    kept_keys = {(int(frame), int(h), int(k), int(l)) for frame, h, k, l in kept_keys}

    summary = add_keep_summary(group_summary, kept)
    affected = summary["was_conditionally_filtered"]
    return ScoreSelection(
        kept_keys=kept_keys,
        summary=summary,
        score_rows_read=int(rows_read),
        score_rows_usable=int(len(scores)),
        kept_score_rows=int(len(kept)),
        duplicate_score_keys=duplicate_keys,
        affected_groups=int(affected.sum()),
        unaffected_groups=int((~affected).sum()),
        affected_observations=int(summary.loc[affected, "n_obs"].sum()),
        condition_description=condition_description,
    )


def build_group_summary(scores: pd.DataFrame) -> pd.DataFrame:
    grouped = scores.groupby(["H", "K", "L"], sort=False, observed=True)
    summary = grouped.agg(
        n_obs=("score", "size"),
        sigma_min=("score", "min"),
        sigma_median=("score", "median"),
        sigma_p95=("score", lambda x: x.quantile(0.95)),
        sigma_max=("score", "max"),
    ).reset_index()

    if "d_angstrom" in scores:
        d_summary = grouped.agg(d_angstrom_mean=("d_angstrom", "mean")).reset_index()
        summary = summary.merge(d_summary, on=["H", "K", "L"], how="left")

    if "S_dyn_geom" in scores:
        s_summary = grouped.agg(
            S_dyn_geom_min=("S_dyn_geom", "min"),
            S_dyn_geom_median=("S_dyn_geom", "median"),
            S_dyn_geom_p95=("S_dyn_geom", lambda x: x.quantile(0.95)),
            S_dyn_geom_max=("S_dyn_geom", "max"),
        ).reset_index()
        s_summary["S_dyn_geom_spread_p95_minus_median"] = (
            s_summary["S_dyn_geom_p95"] - s_summary["S_dyn_geom_median"]
        )
        s_summary["S_dyn_geom_spread_max_minus_median"] = (
            s_summary["S_dyn_geom_max"] - s_summary["S_dyn_geom_median"]
        )
        summary = summary.merge(s_summary, on=["H", "K", "L"], how="left")

    return summary


def apply_group_condition(
    summary: pd.DataFrame,
    condition_column: str | None,
    condition_op: str | None,
    condition_value: float | None,
) -> tuple[pd.DataFrame, str]:
    supplied = [condition_column is not None, condition_op is not None, condition_value is not None]
    if any(supplied) and not all(supplied):
        raise SystemExit(
            "Provide all condition arguments together: "
            "--condition-column, --condition-op, and --condition-value."
        )

    out = summary.copy()
    if not any(supplied):
        out["was_conditionally_filtered"] = True
        return out, "all HKL groups"

    assert condition_column is not None
    assert condition_op is not None
    assert condition_value is not None
    if condition_column not in out.columns:
        raise SystemExit(
            f"Unknown --condition-column {condition_column!r}. "
            f"Available group columns: {sorted(out.columns)}"
        )

    values = pd.to_numeric(out[condition_column], errors="coerce")
    threshold = float(condition_value)
    if condition_op == "ge":
        mask = values >= threshold
        symbol = ">="
    elif condition_op == "gt":
        mask = values > threshold
        symbol = ">"
    elif condition_op == "le":
        mask = values <= threshold
        symbol = "<="
    elif condition_op == "lt":
        mask = values < threshold
        symbol = "<"
    elif condition_op == "eq":
        mask = values == threshold
        symbol = "=="
    elif condition_op == "ne":
        mask = values != threshold
        symbol = "!="
    else:
        raise SystemExit(f"Unsupported --condition-op {condition_op!r}")

    out["was_conditionally_filtered"] = mask.fillna(False)
    return out, f"{condition_column} {symbol} {threshold:g}"


def add_keep_summary(group_summary: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    summary = group_summary.copy()
    keep_summary = kept.groupby(["H", "K", "L"], sort=False, observed=True).agg(
        n_keep_actual=("score", "size"),
        sigma_keep_max=("score", "max"),
    ).reset_index()
    summary = summary.merge(keep_summary, on=["H", "K", "L"], how="left")
    summary["n_keep_actual"] = summary["n_keep_actual"].fillna(0).astype(int)
    summary["n_removed"] = summary["n_obs"] - summary["n_keep_actual"]
    summary["keep_fraction_actual"] = summary["n_keep_actual"] / summary["n_obs"]

    first_columns = [
        "H",
        "K",
        "L",
        "n_obs",
        "n_keep",
        "n_keep_actual",
        "n_removed",
        "keep_fraction_actual",
        "was_conditionally_filtered",
    ]
    metric_columns = [column for column in summary.columns if column not in first_columns]
    summary = summary.loc[:, first_columns + metric_columns]
    return summary.sort_values(["sigma_median", "sigma_p95", "n_obs"], ascending=[False, False, False])


def parse_reflection_hkl(line: str) -> tuple[int, int, int] | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0])
        k = int(parts[1])
        l = int(parts[2])
        float(parts[3])
        float(parts[4])
    except ValueError:
        return None
    return h, k, l


def filter_stream(
    stream_path: Path,
    output_path: Path,
    kept_keys: set[tuple[int, int, int, int]],
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_chunk = False
    in_crystal = False
    in_reflections = False
    current_frame = -1

    chunks_seen = 0
    crystals_seen = 0
    reflection_rows_seen = 0
    reflection_rows_kept = 0
    reflection_rows_removed = 0

    with stream_path.open("r", encoding="utf-8", errors="replace") as inp, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for raw_line in inp:
            line = raw_line.rstrip("\n")

            if line.startswith("----- Begin chunk -----"):
                in_chunk = True
                in_crystal = False
                in_reflections = False
                chunks_seen += 1
                out.write(raw_line)
                continue

            if line.startswith("----- End chunk -----"):
                in_chunk = False
                in_crystal = False
                in_reflections = False
                out.write(raw_line)
                continue

            if line.startswith("--- Begin crystal"):
                in_crystal = True
                in_reflections = False
                current_frame += 1
                out.write(raw_line)
                continue

            if line.startswith("--- End crystal"):
                if in_crystal:
                    crystals_seen += 1
                in_crystal = False
                in_reflections = False
                out.write(raw_line)
                continue

            if in_crystal and "Reflections measured after indexing" in line:
                in_reflections = True
                out.write(raw_line)
                continue

            if in_reflections and "End of reflections" in line:
                in_reflections = False
                out.write(raw_line)
                continue

            if in_chunk and in_crystal and in_reflections:
                hkl = parse_reflection_hkl(line)
                if hkl is not None:
                    reflection_rows_seen += 1
                    h, k, l = hkl
                    key = (current_frame, h, k, l)
                    if key in kept_keys:
                        reflection_rows_kept += 1
                        out.write(raw_line)
                    else:
                        reflection_rows_removed += 1
                    continue

            out.write(raw_line)

    return {
        "chunks_seen": chunks_seen,
        "crystals_seen": crystals_seen,
        "reflection_rows_seen": reflection_rows_seen,
        "reflection_rows_kept": reflection_rows_kept,
        "reflection_rows_removed": reflection_rows_removed,
    }


def print_summary(
    selection: ScoreSelection,
    stream_stats: dict[str, int],
    keep_fraction: float,
    summary_output: Path,
    output_stream: Path,
) -> None:
    score_kept = len(selection.kept_keys)
    score_removed = selection.score_rows_usable - selection.kept_score_rows
    score_fraction_kept = selection.kept_score_rows / selection.score_rows_usable if selection.score_rows_usable else 0.0

    print("Filter summary:")
    print(f"  score rows read: {selection.score_rows_read:,}")
    print(f"  usable score rows: {selection.score_rows_usable:,}")
    print(f"  duplicate score key rows: {selection.duplicate_score_keys:,}")
    print(f"  mmm HKL groups: {len(selection.summary):,}")
    print(f"  condition: {selection.condition_description}")
    print(f"  affected HKL groups: {selection.affected_groups:,}")
    print(f"  unaffected HKL groups: {selection.unaffected_groups:,}")
    print(f"  affected observations: {selection.affected_observations:,}")
    print(f"  requested keep fraction: {keep_fraction:.6g}")
    print(f"  kept score rows selected: {selection.kept_score_rows:,}")
    print(f"  unique kept stream keys: {score_kept:,}")
    print(f"  removed observations from scores: {score_removed:,}")
    print(f"  retained fraction from scores: {score_fraction_kept:.6f}")
    print(f"  stream chunks seen: {stream_stats['chunks_seen']:,}")
    print(f"  stream crystals seen: {stream_stats['crystals_seen']:,}")
    print(f"  stream reflection rows seen: {stream_stats['reflection_rows_seen']:,}")
    print(f"  stream reflection rows kept: {stream_stats['reflection_rows_kept']:,}")
    print(f"  stream reflection rows removed: {stream_stats['reflection_rows_removed']:,}")
    if stream_stats["reflection_rows_kept"] != score_kept:
        delta = stream_stats["reflection_rows_kept"] - score_kept
        print(f"  WARNING: stream kept count differs from score kept count by {delta:,}.")
        print("           This can happen if the stream and score file do not have identical frame/HKL identities.")
    print(f"  output stream: {output_stream}")
    print(f"  per-HKL filter summary: {summary_output}")
    print()
    print_affected_group_diagnostics(selection.summary)


def print_affected_group_diagnostics(summary: pd.DataFrame, top_n: int = 10) -> None:
    affected = summary.loc[summary["was_conditionally_filtered"]].copy()
    diagnostic_columns = [
        "H",
        "K",
        "L",
        "n_obs",
        "n_keep_actual",
        "n_removed",
        "keep_fraction_actual",
        "sigma_median",
        "sigma_p95",
        "sigma_max",
        "sigma_keep_max",
    ]
    for optional in (
        "S_dyn_geom_median",
        "S_dyn_geom_p95",
        "S_dyn_geom_spread_p95_minus_median",
        "d_angstrom_mean",
    ):
        if optional in affected.columns:
            diagnostic_columns.append(optional)

    print(f"Top affected groups by sigma_median (top {top_n}):")
    if affected.empty:
        print("  no affected groups")
    else:
        print(
            affected.sort_values(["sigma_median", "sigma_p95"], ascending=[False, False])
            .loc[:, diagnostic_columns]
            .head(top_n)
            .to_string(index=False)
        )
    print()

    print(f"Top affected groups by sigma_p95 (top {top_n}):")
    if affected.empty:
        print("  no affected groups")
    else:
        print(
            affected.sort_values(["sigma_p95", "sigma_median"], ascending=[False, False])
            .loc[:, diagnostic_columns]
            .head(top_n)
            .to_string(index=False)
        )
    print()

    sentinel = summary.loc[(summary["H"] == 0) & (summary["K"] == 4) & (summary["L"] == 0)]
    print("Check group (0,4,0):")
    if sentinel.empty:
        print("  group not present")
    else:
        print(sentinel.loc[:, diagnostic_columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    columns = read_score_columns(args.scores)
    inspect_score_columns(columns, args.score_column)

    if args.inspect_only:
        return

    scores, rows_read = load_scores(args.scores, columns, args.score_column, args.chunksize)
    selection = select_lowest_risk_observations(
        scores,
        rows_read=rows_read,
        keep_fraction=args.keep_fraction,
        min_keep=args.min_keep,
        condition_column=args.condition_column,
        condition_op=args.condition_op,
        condition_value=args.condition_value,
    )

    summary_output = args.summary_output or summary_path_for_output(args.output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    selection.summary.to_csv(summary_output, index=False)
    print(f"Wrote per-HKL filter summary: {summary_output}")
    print()

    stream_stats = filter_stream(args.stream, args.output, selection.kept_keys)
    print_summary(selection, stream_stats, args.keep_fraction, summary_output, args.output)


if __name__ == "__main__":
    main()
