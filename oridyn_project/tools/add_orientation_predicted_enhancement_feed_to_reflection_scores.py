#!/usr/bin/env python3
"""Append orientation-predicted enhancement-feed diagnostics to reflection scores.

This standalone OriDyn add-on leaves core OriDyn scoring unchanged. It computes
an enhancement-feed score for observed reflections in a CrystFEL stream, then
left-joins the diagnostic columns onto an existing reflection_scores.csv by
exact source_filename + event + signed h,k,l.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oridyn.geometry import normalize_vector
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.stream_parser import parse_crystfel_stream, parse_crystfel_stream_text, reciprocal_matrix_from_row


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
FEED_COLUMNS = [
    "enh_feed_raw",
    "enh_feed_norm_frame",
    "enh_feed_log_raw",
    "enh_feed_rank_frame",
    "enh_feed_frame_p95_raw",
    "enh_feed_norm_frame_clipped",
    "enh_feed_n_predicted_q",
    "enh_feed_n_valid_paths",
    "enh_feed_top_path_score",
    "enh_feed_s_max",
    "enh_feed_d_min",
    "enh_feed_d_max",
]
PROGRESS_EVERY_FRAMES = 25
LOG_EPSILON = 1e-300
N_TOP_HKLS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--reflection-scores", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--d-min", type=float, default=0.3)
    parser.add_argument("--d-max", type=float, default=20.0)
    parser.add_argument("--s-max", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional first-N indexed-frame limit for smoke tests.")
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Write only key/risk/feed diagnostic columns to enh_feed_observation_scores.csv.",
    )
    parser.add_argument(
        "--write-hkl-spread-summary",
        action="store_true",
        help="Write enh_feed_by_signed_hkl_spread.csv and include HKL spread tables in the large summary.",
    )
    parser.add_argument("--progress-every-frames", type=int, default=PROGRESS_EVERY_FRAMES)
    return parser.parse_args()


def prepare_output_dir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise SystemExit(f"{outdir} exists and is not empty; pass --overwrite to reuse it.")
    outdir.mkdir(parents=True, exist_ok=True)


def parse_stream(path: Path, max_frames: int | None):
    if not path.exists():
        raise SystemExit(f"--stream not found: {path}")
    if max_frames is None:
        try:
            return parse_crystfel_stream(path)
        except ValueError as exc:
            raise SystemExit(f"Could not parse stream cell/orientations: {exc}") from exc

    if max_frames < 1:
        raise SystemExit("--max-frames must be >= 1 when provided.")
    text = stream_prefix_for_first_frames(path, max_frames)
    try:
        return parse_crystfel_stream_text(text, path=str(path))
    except ValueError as exc:
        raise SystemExit(
            "Could not parse cell/orientations from the stream prefix. "
            f"If this stream lacks unit-cell metadata, add a cell option before using this tool. Details: {exc}"
        ) from exc


def stream_prefix_for_first_frames(path: Path, max_frames: int) -> str:
    lines: list[str] = []
    n_crystals = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line)
            if line.startswith("--- End crystal"):
                n_crystals += 1
                if n_crystals >= max_frames:
                    break
    if n_crystals == 0:
        raise SystemExit(f"No indexed crystal blocks found in stream prefix: {path}")
    return "".join(lines)


def load_reflection_scores(path: Path, minimal_output: bool) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"--reflection-scores not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if minimal_output:
        optional = [column for column in ("graph_crowding_norm", "frame_axis_risk_norm") if column in header]
        scores = pd.read_csv(path, usecols=[*KEY_COLUMNS, *optional])
    else:
        scores = pd.read_csv(path)
    missing = [column for column in KEY_COLUMNS if column not in scores.columns]
    if missing:
        raise SystemExit(f"reflection_scores.csv is missing required key column(s): {missing}")

    out = scores.copy()
    out = out.drop(columns=[column for column in FEED_COLUMNS if column in out.columns], errors="ignore")
    out["_reflection_scores_row"] = np.arange(len(out), dtype=np.int64)
    normalize_key_columns(out)

    duplicated = out.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"reflection_scores.csv has duplicate exact observation keys, examples: {examples}")
    return out


def normalize_key_columns(table: pd.DataFrame) -> None:
    table["source_filename"] = table["source_filename"].astype(str)
    table["event"] = table["event"].astype(str)
    for column in ["h", "k", "l"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["h", "k", "l"]].isna().any().any():
        raise SystemExit("Encountered non-numeric h/k/l values in key columns.")
    table[["h", "k", "l"]] = table[["h", "k", "l"]].astype("int64")


def prepare_candidates(stream, d_min: float, d_max: float) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        return generate_candidate_hkls(
            stream.unit_cell,
            d_min,
            d_max,
            hkl_limit=None,
            max_candidates=None,
            centering=stream.unit_cell.centering,
        )
    except ValueError as exc:
        raise SystemExit(
            "Could not generate candidate HKLs from the parsed stream cell. "
            f"If the stream cell is missing or wrong, add a cell option before using this tool. Details: {exc}"
        ) from exc


def hkl_to_G(hkls: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    return np.asarray(hkls, dtype=float) @ np.asarray(reciprocal, dtype=float).T


def excitation_error(G: np.ndarray, wavelength_angstrom: float, beam_direction=(0.0, 0.0, 1.0)) -> np.ndarray:
    vectors = np.asarray(G, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)
    k_norm = 1.0 / float(wavelength_angstrom)
    k_in = normalize_vector(beam_direction) * k_norm
    return (np.sum((vectors + k_in[None, :]) ** 2, axis=1) - k_norm**2) / (2.0 * k_norm)


def excitation_weight(s: np.ndarray, s_max: float) -> np.ndarray:
    return np.exp(-0.5 * (np.asarray(s, dtype=float) / max(float(s_max), 1e-300)) ** 2)


def proxy_strength(G_norm: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-((np.asarray(G_norm, dtype=float) / max(float(scale), 1e-300)) ** 2))


def compute_feed_scores(
    stream,
    candidates: pd.DataFrame,
    d_min: float,
    d_max: float,
    s_max: float,
    progress_every_frames: int,
) -> pd.DataFrame:
    if stream.reflections.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "frame", "frame_number", *FEED_COLUMNS])

    candidate_hkls = candidates[["h", "k", "l"]].to_numpy(dtype=int)
    candidate_lookup = {tuple(int(x) for x in hkl): idx for idx, hkl in enumerate(candidate_hkls)}
    crystal_by_frame = stream.crystal_table.set_index("frame", drop=False)
    frame_groups = list(stream.reflections.groupby("frame", sort=True))
    rows: list[pd.DataFrame] = []
    progress_every = max(int(progress_every_frames), 1)

    for group_idx, (frame, group) in enumerate(frame_groups, start=1):
        if group_idx == 1 or group_idx % progress_every == 0 or group_idx == len(frame_groups):
            print(f"Scoring frame {group_idx}/{len(frame_groups)} (frame={int(frame)}, targets={len(group)})", file=sys.stderr)
        reciprocal = reciprocal_matrix_from_row(crystal_by_frame.loc[int(frame)])
        rows.append(
            score_one_frame(
                group.reset_index(drop=True),
                candidate_hkls,
                candidate_lookup,
                reciprocal,
                stream.wavelength_angstrom,
                d_min,
                d_max,
                s_max,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*KEY_COLUMNS, "frame", "frame_number", *FEED_COLUMNS])


def score_one_frame(
    observed: pd.DataFrame,
    candidate_hkls: np.ndarray,
    candidate_lookup: dict[tuple[int, int, int], int],
    reciprocal: np.ndarray,
    wavelength_angstrom: float,
    d_min: float,
    d_max: float,
    s_max: float,
) -> pd.DataFrame:
    base = observed[[column for column in ["source_filename", "event", "frame", "frame_number", "h", "k", "l"] if column in observed]].copy()
    normalize_key_columns(base)

    candidate_G = hkl_to_G(candidate_hkls, reciprocal)
    candidate_norm = np.linalg.norm(candidate_G, axis=1)
    nonzero = candidate_norm[candidate_norm > 0.0]
    G0 = 3.0 * float(np.min(nonzero)) if nonzero.size else 1.0
    W = proxy_strength(candidate_norm, G0)
    s = excitation_error(candidate_G, wavelength_angstrom)
    E = excitation_weight(s, s_max)
    predicted_q = np.flatnonzero(np.abs(s) <= float(s_max))

    target_hkls = base[["h", "k", "l"]].to_numpy(dtype=int)
    target_G = hkl_to_G(target_hkls, reciprocal)
    target_norm = np.linalg.norm(target_G, axis=1)
    raw = np.zeros(len(base), dtype=float)
    n_paths = np.zeros(len(base), dtype=int)
    top_path = np.zeros(len(base), dtype=float)

    for target_idx, target in enumerate(target_hkls):
        g_G = target_G[target_idx]
        g_norm = target_norm[target_idx]
        for q_idx in predicted_q:
            q = candidate_hkls[q_idx]
            q_key = tuple(int(x) for x in q)
            if q_key == (0, 0, 0):
                continue
            r_key = tuple(int(target[axis] - q[axis]) for axis in range(3))
            if r_key == (0, 0, 0):
                continue
            r_idx = candidate_lookup.get(r_key)
            if r_idx is None:
                continue
            if candidate_norm[q_idx] >= g_norm or candidate_norm[r_idx] >= g_norm:
                continue
            if float(np.dot(candidate_G[q_idx], g_G)) <= 0.0 or float(np.dot(candidate_G[r_idx], g_G)) <= 0.0:
                continue

            contribution = float(E[q_idx] * W[q_idx] * W[r_idx])
            raw[target_idx] += contribution
            n_paths[target_idx] += 1
            if contribution > top_path[target_idx]:
                top_path[target_idx] = contribution

    positive = raw[raw > 0.0]
    frame_p95 = float(np.quantile(positive, 0.95)) if positive.size else 0.0
    denominator = frame_p95 if frame_p95 > 0.0 else 1.0
    if denominator <= 0.0:
        denominator = 1.0

    base["enh_feed_raw"] = raw
    base["enh_feed_norm_frame"] = raw / denominator
    base["enh_feed_log_raw"] = np.log10(raw + LOG_EPSILON)
    base["enh_feed_rank_frame"] = pd.Series(raw).rank(method="average", pct=True).to_numpy(dtype=float)
    base["enh_feed_frame_p95_raw"] = frame_p95
    base["enh_feed_norm_frame_clipped"] = np.minimum(base["enh_feed_norm_frame"].to_numpy(dtype=float), 10.0)
    base["enh_feed_n_predicted_q"] = int(len(predicted_q))
    base["enh_feed_n_valid_paths"] = n_paths
    base["enh_feed_top_path_score"] = top_path
    base["enh_feed_s_max"] = float(s_max)
    base["enh_feed_d_min"] = float(d_min)
    base["enh_feed_d_max"] = float(d_max)
    return base


def append_feed_columns(reflection_scores: pd.DataFrame, feed_scores: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicated = feed_scores.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = feed_scores.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"stream feed scores have duplicate exact observation keys, examples: {examples}")

    joined = reflection_scores.merge(feed_scores[[*KEY_COLUMNS, *FEED_COLUMNS]], on=KEY_COLUMNS, how="left", validate="one_to_one")
    joined = joined.sort_values("_reflection_scores_row").drop(columns=["_reflection_scores_row"])
    n_matches = int(joined["enh_feed_raw"].notna().sum())
    return joined, n_matches


def spearman_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, int(len(frame))
    ranked = frame.rank(method="average")
    return float(ranked["x"].corr(ranked["y"])), int(len(frame))


def format_stats(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"median={float(values.median()):.6g}, p95={float(values.quantile(0.95)):.6g}, max={float(values.max()):.6g}"


def quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(q)) if not numeric.empty else np.nan


def build_hkl_spread_summary(output: pd.DataFrame) -> pd.DataFrame:
    matched = output.loc[output["enh_feed_raw"].notna()].copy()
    columns = [
        "h",
        "k",
        "l",
        "n_obs",
        "enh_feed_raw_min",
        "enh_feed_raw_median",
        "enh_feed_raw_p75",
        "enh_feed_raw_p90",
        "enh_feed_raw_p95",
        "enh_feed_raw_max",
        "enh_feed_raw_iqr",
        "enh_feed_raw_p95_minus_median",
        "enh_feed_log_raw_median",
        "enh_feed_log_raw_p95",
        "enh_feed_rank_frame_median",
        "enh_feed_rank_frame_p95",
    ]
    optional_columns = []
    if "graph_crowding_norm" in matched.columns:
        optional_columns.extend(["graph_crowding_norm_median", "graph_crowding_norm_p95"])
    if "frame_axis_risk_norm" in matched.columns:
        optional_columns.extend(["frame_axis_risk_norm_median", "frame_axis_risk_norm_p95"])
    if matched.empty:
        return pd.DataFrame(columns=[*columns, *optional_columns])

    rows: list[dict[str, float | int]] = []
    for (h, k, l), group in matched.groupby(["h", "k", "l"], sort=True):
        raw = pd.to_numeric(group["enh_feed_raw"], errors="coerce").dropna()
        p25 = float(raw.quantile(0.25)) if not raw.empty else np.nan
        p75 = float(raw.quantile(0.75)) if not raw.empty else np.nan
        median = float(raw.median()) if not raw.empty else np.nan
        p95 = float(raw.quantile(0.95)) if not raw.empty else np.nan
        row: dict[str, float | int] = {
            "h": int(h),
            "k": int(k),
            "l": int(l),
            "n_obs": int(len(group)),
            "enh_feed_raw_min": float(raw.min()) if not raw.empty else np.nan,
            "enh_feed_raw_median": median,
            "enh_feed_raw_p75": p75,
            "enh_feed_raw_p90": float(raw.quantile(0.90)) if not raw.empty else np.nan,
            "enh_feed_raw_p95": p95,
            "enh_feed_raw_max": float(raw.max()) if not raw.empty else np.nan,
            "enh_feed_raw_iqr": p75 - p25 if np.isfinite(p75) and np.isfinite(p25) else np.nan,
            "enh_feed_raw_p95_minus_median": p95 - median if np.isfinite(p95) and np.isfinite(median) else np.nan,
            "enh_feed_log_raw_median": quantile(group["enh_feed_log_raw"], 0.50),
            "enh_feed_log_raw_p95": quantile(group["enh_feed_log_raw"], 0.95),
            "enh_feed_rank_frame_median": quantile(group["enh_feed_rank_frame"], 0.50),
            "enh_feed_rank_frame_p95": quantile(group["enh_feed_rank_frame"], 0.95),
        }
        if "graph_crowding_norm" in group.columns:
            row["graph_crowding_norm_median"] = quantile(group["graph_crowding_norm"], 0.50)
            row["graph_crowding_norm_p95"] = quantile(group["graph_crowding_norm"], 0.95)
        if "frame_axis_risk_norm" in group.columns:
            row["frame_axis_risk_norm_median"] = quantile(group["frame_axis_risk_norm"], 0.50)
            row["frame_axis_risk_norm_p95"] = quantile(group["frame_axis_risk_norm"], 0.95)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = N_TOP_HKLS) -> str:
    if table is None or table.empty:
        return "_No rows._"
    view = table.loc[:, [column for column in columns if column in table.columns]].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_summary(
    outdir: Path,
    stream_path: Path,
    scores_path: Path,
    stream_observations: int,
    score_rows: int,
    n_matches: int,
    n_frames: int,
    output: pd.DataFrame,
    hkl_spread: pd.DataFrame | None,
) -> None:
    matched = output.loc[output["enh_feed_raw"].notna()].copy()
    signed_hkl_count = int(matched[["h", "k", "l"]].drop_duplicates().shape[0]) if not matched.empty else 0
    lines = [
        "# Orientation-Predicted Enhancement Feed Append Summary",
        "",
        "## Inputs",
        "",
        f"- Stream: `{stream_path}`",
        f"- Reflection scores: `{scores_path}`",
        "",
        "## Counts",
        "",
        f"- Stream observations scored: {stream_observations}",
        f"- Reflection score rows: {score_rows}",
        f"- Exact observation matches: {n_matches}",
        f"- Frames scored: {n_frames}",
        f"- Signed HKLs with enhancement scores: {signed_hkl_count}",
        "",
        "## Enhancement Feed Score",
        "",
        "- Candidate source beams are generated from the parsed stream cell and each frame orientation.",
        "- Predicted source beams satisfy `|s(q)| <= s_max`.",
        "- Valid build-up paths obey lower-order forward geometry and contribute `E(q) * W(q) * W(r)`.",
        "- `enh_feed_norm_frame` is `enh_feed_raw / p95_positive_in_frame`.",
        "- This script appends diagnostics only; it does not filter, reweight, or modify OriDyn scoring behavior.",
        "",
        "## Score Distributions",
        "",
        f"- enh_feed_raw: {format_stats(matched['enh_feed_raw'])}",
        f"- enh_feed_norm_frame: {format_stats(matched['enh_feed_norm_frame'])}",
        f"- enh_feed_norm_frame_clipped: {format_stats(matched['enh_feed_norm_frame_clipped'])}",
        f"- enh_feed_rank_frame: {format_stats(matched['enh_feed_rank_frame'])}",
        "",
        "## Spearman Correlations",
        "",
    ]

    if "graph_crowding_norm" in output.columns:
        rho_raw, n_raw = spearman_corr(matched["enh_feed_raw"], matched["graph_crowding_norm"])
        rho_norm, n_norm = spearman_corr(matched["enh_feed_norm_frame"], matched["graph_crowding_norm"])
        lines.append(f"- enh_feed_raw vs graph_crowding_norm: r={rho_raw:.6g}, n={n_raw}")
        lines.append(f"- enh_feed_norm_frame vs graph_crowding_norm: r={rho_norm:.6g}, n={n_norm}")
    else:
        lines.append("- graph_crowding_norm: not present")

    if "frame_axis_risk_norm" in output.columns:
        rho_raw, n_raw = spearman_corr(matched["enh_feed_raw"], matched["frame_axis_risk_norm"])
        rho_norm, n_norm = spearman_corr(matched["enh_feed_norm_frame"], matched["frame_axis_risk_norm"])
        lines.append(f"- enh_feed_raw vs frame_axis_risk_norm: r={rho_raw:.6g}, n={n_raw}")
        lines.append(f"- enh_feed_norm_frame vs frame_axis_risk_norm: r={rho_norm:.6g}, n={n_norm}")
    else:
        lines.append("- frame_axis_risk_norm: not present")

    lines.extend(
        [
            "",
            "Excluded from comparison by design: `self_risk_norm`, `S_dyn_geom`, `sigma_dyn_rel`, and `nonself_mean`.",
            "",
        ]
    )
    summary_text = "\n".join(lines)
    (outdir / "enhancement_feed_summary.md").write_text(summary_text, encoding="utf-8")

    large_lines = [
        summary_text,
        "## Signed-HKL Spread Summary",
        "",
        f"- Signed HKLs summarized: {0 if hkl_spread is None else len(hkl_spread)}",
        "",
    ]
    top_columns = [
        "h",
        "k",
        "l",
        "n_obs",
        "enh_feed_raw_median",
        "enh_feed_raw_p95",
        "enh_feed_raw_p95_minus_median",
        "enh_feed_rank_frame_p95",
    ]
    if hkl_spread is None:
        large_lines.append("_HKL spread summary was not requested. Re-run with `--write-hkl-spread-summary`._")
    else:
        large_lines.extend(
            [
                "### Top 30 HKLs By enh_feed_raw_p95_minus_median",
                "",
                markdown_table(
                    hkl_spread.sort_values("enh_feed_raw_p95_minus_median", ascending=False),
                    top_columns,
                ),
                "",
                "### Top 30 HKLs By enh_feed_raw_p95",
                "",
                markdown_table(
                    hkl_spread.sort_values("enh_feed_raw_p95", ascending=False),
                    top_columns,
                ),
                "",
                "### Top 30 HKLs By enh_feed_rank_frame_p95",
                "",
                markdown_table(
                    hkl_spread.sort_values("enh_feed_rank_frame_p95", ascending=False),
                    top_columns,
                ),
            ]
        )
    (outdir / "enhancement_feed_large_summary.md").write_text("\n".join(large_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.progress_every_frames < 1:
        raise SystemExit("--progress-every-frames must be >= 1.")
    prepare_output_dir(args.outdir, args.overwrite)

    print(f"Parsing stream: {args.stream}", file=sys.stderr)
    stream = parse_stream(args.stream, args.max_frames)
    print(
        f"Parsed {len(stream.crystal_table)} indexed frame(s), {len(stream.reflections)} observed reflection(s)",
        file=sys.stderr,
    )
    print(f"Generating candidate HKLs for {args.d_min:g} <= d <= {args.d_max:g} A", file=sys.stderr)
    candidates, metadata = prepare_candidates(stream, args.d_min, args.d_max)
    print(f"Candidate HKLs: {metadata.get('n_candidates')}", file=sys.stderr)
    print(f"Loading reflection scores: {args.reflection_scores}", file=sys.stderr)
    reflection_scores = load_reflection_scores(args.reflection_scores, args.minimal_output)
    print(f"Loaded {len(reflection_scores)} reflection score row(s)", file=sys.stderr)

    feed_scores = compute_feed_scores(
        stream,
        candidates,
        args.d_min,
        args.d_max,
        args.s_max,
        args.progress_every_frames,
    )
    output, n_matches = append_feed_columns(reflection_scores, feed_scores)
    output_path = args.outdir / ("enh_feed_observation_scores.csv" if args.minimal_output else "reflection_scores_with_enh_feed.csv")
    output.to_csv(output_path, index=False)

    hkl_spread = None
    if args.write_hkl_spread_summary:
        hkl_spread = build_hkl_spread_summary(output)
        hkl_spread.to_csv(args.outdir / "enh_feed_by_signed_hkl_spread.csv", index=False)

    write_summary(
        args.outdir,
        args.stream,
        args.reflection_scores,
        len(feed_scores),
        len(reflection_scores),
        n_matches,
        len(stream.crystal_table),
        output,
        hkl_spread,
    )

    print(f"Stream observations scored: {len(feed_scores)}")
    print(f"Reflection score rows: {len(reflection_scores)}")
    print(f"Exact observation matches: {n_matches}")
    print(f"Wrote: {output_path}")
    if hkl_spread is not None:
        print(f"Wrote: {args.outdir / 'enh_feed_by_signed_hkl_spread.csv'}")
    print(f"Wrote: {args.outdir / 'enhancement_feed_summary.md'}")
    print(f"Wrote: {args.outdir / 'enhancement_feed_large_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
