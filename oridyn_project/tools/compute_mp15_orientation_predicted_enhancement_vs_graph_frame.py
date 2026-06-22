#!/usr/bin/env python3
"""Orientation-predicted enhancement-feed comparison for MP15.

For each indexed frame, candidate HKLs are excited from the frame orientation,
then observed reflections in that frame are scored as feed targets. Results are
joined to OriDyn graph/frame risk terms by exact observation key:
source_filename + event + signed h,k,l. No symmetry canonicalization is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oridyn.geometry import normalize_vector
from oridyn.hkl_generation import generate_candidate_hkls
from oridyn.stream_parser import parse_crystfel_stream, parse_crystfel_stream_text, reciprocal_matrix_from_row


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
GRAPH_COLUMN = "graph_crowding_norm"
FRAME_COLUMN = "frame_axis_risk_norm"
FEED_COLUMNS = ["S_feed_raw", "S_feed_norm_frame"]
MIN_PER_FRAME_CORR_REFLECTIONS = 30
PROGRESS_EVERY_FRAMES = 25
TOP_EXAMPLE_ROWS = 50


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


def load_reflection_scores(path: Path) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        raise SystemExit(f"--reflection-scores not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = [*KEY_COLUMNS, GRAPH_COLUMN]
    missing = [column for column in required if column not in header]
    if missing:
        raise SystemExit(f"reflection_scores.csv is missing required column(s): {missing}")
    has_frame_axis = FRAME_COLUMN in header
    usecols = [*required, FRAME_COLUMN] if has_frame_axis else required
    scores = pd.read_csv(path, usecols=usecols)

    out = scores.copy()
    for col in ["source_filename", "event"]:
        out[col] = out[col].astype(str)
    for col in ["h", "k", "l"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[GRAPH_COLUMN] = pd.to_numeric(out[GRAPH_COLUMN], errors="coerce")
    if has_frame_axis:
        out[FRAME_COLUMN] = pd.to_numeric(out[FRAME_COLUMN], errors="coerce")
    out = out.dropna(subset=["h", "k", "l"])
    out[["h", "k", "l"]] = out[["h", "k", "l"]].astype("int64")

    duplicated = out.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"reflection_scores.csv has duplicate exact observation keys, examples: {examples}")
    return out, has_frame_axis


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
    scale = max(float(s_max), 1e-300)
    return np.exp(-0.5 * (np.asarray(s, dtype=float) / scale) ** 2)


def proxy_strength(G_norm: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-((np.asarray(G_norm, dtype=float) / max(float(scale), 1e-300)) ** 2))


def prepare_candidates(stream, d_min: float, d_max: float) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        candidates, metadata = generate_candidate_hkls(
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
    return candidates, metadata


def compute_frame_scores(stream, candidates: pd.DataFrame, s_max: float) -> pd.DataFrame:
    if stream.reflections.empty:
        return pd.DataFrame(
            columns=[
                *KEY_COLUMNS,
                "frame",
                "frame_number",
                *FEED_COLUMNS,
                "n_predicted_q",
                "n_valid_feed_paths",
            ]
        )
    candidate_hkls = candidates[["h", "k", "l"]].to_numpy(dtype=int)
    candidate_lookup = {tuple(int(x) for x in hkl): idx for idx, hkl in enumerate(candidate_hkls)}
    crystal_by_frame = stream.crystal_table.set_index("frame", drop=False)
    rows: list[pd.DataFrame] = []
    frame_groups = list(stream.reflections.groupby("frame", sort=True))
    for group_idx, (frame, group) in enumerate(frame_groups, start=1):
        if group_idx == 1 or group_idx % PROGRESS_EVERY_FRAMES == 0 or group_idx == len(frame_groups):
            print(f"Scoring frame {group_idx}/{len(frame_groups)} (frame={int(frame)}, targets={len(group)})", file=sys.stderr)
        crystal_row = crystal_by_frame.loc[int(frame)]
        reciprocal = reciprocal_matrix_from_row(crystal_row)
        rows.append(score_one_frame(group.reset_index(drop=True), candidate_hkls, candidate_lookup, reciprocal, stream.wavelength_angstrom, s_max))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score_one_frame(
    observed: pd.DataFrame,
    candidate_hkls: np.ndarray,
    candidate_lookup: dict[tuple[int, int, int], int],
    reciprocal: np.ndarray,
    wavelength_angstrom: float,
    s_max: float,
) -> pd.DataFrame:
    base = observed[[col for col in ["source_filename", "event", "frame", "frame_number", "h", "k", "l"] if col in observed]].copy()
    for col in ["source_filename", "event"]:
        base[col] = base[col].astype(str)
    base[["h", "k", "l"]] = base[["h", "k", "l"]].astype("int64")

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
            raw[target_idx] += float(E[q_idx] * W[q_idx] * W[r_idx])
            n_paths[target_idx] += 1

    positive = raw[raw > 0.0]
    denominator = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    if denominator <= 0.0:
        denominator = 1.0
    base["S_feed_raw"] = raw
    base["S_feed_norm_frame"] = raw / denominator
    base["n_predicted_q"] = int(len(predicted_q))
    base["n_valid_feed_paths"] = n_paths
    return base


def join_scores(feed: pd.DataFrame, reflection_scores: pd.DataFrame) -> pd.DataFrame:
    duplicated = feed.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = feed.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"stream observations have duplicate exact observation keys, examples: {examples}")
    return feed.merge(reflection_scores, on=KEY_COLUMNS, how="inner", validate="one_to_one")


def spearman_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, int(len(frame))
    ranked = frame.rank(method="average")
    return float(ranked["x"].corr(ranked["y"])), int(len(frame))


def risk_columns(has_frame_axis: bool) -> list[str]:
    return [GRAPH_COLUMN, FRAME_COLUMN] if has_frame_axis else [GRAPH_COLUMN]


def add_corr_row(
    rows: list[dict[str, object]],
    scope: str,
    table: pd.DataFrame,
    feed_metric: str,
    risk_metric: str,
    frame: int | None = None,
    source_filename: str | None = None,
    event: str | None = None,
) -> None:
    rho, n_pairs = spearman_corr(table[feed_metric], table[risk_metric])
    rows.append(
        {
            "scope": scope,
            "frame": "" if frame is None else int(frame),
            "source_filename": "" if source_filename is None else source_filename,
            "event": "" if event is None else event,
            "feed_metric": feed_metric,
            "risk_metric": risk_metric,
            "n_pairs": n_pairs,
            "spearman_r": rho,
        }
    )


def build_correlation_summary(joined: pd.DataFrame, has_frame_axis: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    risks = risk_columns(has_frame_axis)
    for feed_metric in FEED_COLUMNS:
        for risk_metric in risks:
            add_corr_row(rows, "all_observations", joined, feed_metric, risk_metric)

    per_frame_start = len(rows)
    for frame, group in joined.groupby("frame", sort=True):
        if len(group) < MIN_PER_FRAME_CORR_REFLECTIONS:
            continue
        source_filename = str(group["source_filename"].iloc[0])
        event = str(group["event"].iloc[0])
        for feed_metric in FEED_COLUMNS:
            for risk_metric in risks:
                add_corr_row(rows, "per_frame", group, feed_metric, risk_metric, int(frame), source_filename, event)

    per_frame = pd.DataFrame.from_records(rows[per_frame_start:])
    if not per_frame.empty:
        for (feed_metric, risk_metric), metric_group in per_frame.groupby(["feed_metric", "risk_metric"], sort=True):
            values = pd.to_numeric(metric_group["spearman_r"], errors="coerce").dropna()
            rows.append(
                {
                    "scope": "per_frame_distribution",
                    "frame": "",
                    "source_filename": "",
                    "event": "",
                    "feed_metric": feed_metric,
                    "risk_metric": risk_metric,
                    "n_pairs": int(values.size),
                    "spearman_r": float(values.median()) if values.size else np.nan,
                    "spearman_min": float(values.min()) if values.size else np.nan,
                    "spearman_p05": float(values.quantile(0.05)) if values.size else np.nan,
                    "spearman_median": float(values.median()) if values.size else np.nan,
                    "spearman_p95": float(values.quantile(0.95)) if values.size else np.nan,
                    "spearman_max": float(values.max()) if values.size else np.nan,
                }
            )
    return pd.DataFrame.from_records(rows)


def write_top_examples(joined: pd.DataFrame, outdir: Path, has_frame_axis: bool) -> None:
    feed = "S_feed_norm_frame"
    graph = GRAPH_COLUMN
    feed_high = float(joined[feed].quantile(0.90))
    feed_low = float(joined[feed].quantile(0.50))
    graph_high = float(joined[graph].quantile(0.90))
    graph_low = float(joined[graph].quantile(0.50))
    columns = [
        "source_filename",
        "event",
        "frame",
        "h",
        "k",
        "l",
        "S_feed_raw",
        "S_feed_norm_frame",
        "n_predicted_q",
        "n_valid_feed_paths",
        GRAPH_COLUMN,
    ]
    if has_frame_axis:
        columns.append(FRAME_COLUMN)
    examples = {
        "top_high_feed_high_graph.csv": joined.loc[(joined[feed] >= feed_high) & (joined[graph] >= graph_high), columns]
        .sort_values([feed, graph], ascending=[False, False])
        .head(TOP_EXAMPLE_ROWS),
        "top_high_feed_low_graph.csv": joined.loc[(joined[feed] >= feed_high) & (joined[graph] <= graph_low), columns]
        .sort_values([feed, graph], ascending=[False, True])
        .head(TOP_EXAMPLE_ROWS),
        "top_high_graph_low_feed.csv": joined.loc[(joined[graph] >= graph_high) & (joined[feed] <= feed_low), columns]
        .sort_values([graph, feed], ascending=[False, True])
        .head(TOP_EXAMPLE_ROWS),
    }
    for filename, table in examples.items():
        table.to_csv(outdir / filename, index=False)


def plot_scatter(joined: pd.DataFrame, outdir: Path, risk_metric: str, filename: str) -> None:
    plot_data = joined.dropna(subset=["S_feed_norm_frame", risk_metric]).copy()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    if plot_data.empty:
        ax.text(0.5, 0.5, "No matched observations", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        scatter = ax.scatter(
            plot_data["S_feed_norm_frame"],
            plot_data[risk_metric],
            c=plot_data["frame"],
            cmap="viridis",
            s=12,
            alpha=0.62,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax, label="frame")
        ax.set_xlabel("S_feed_norm_frame")
        ax.set_ylabel(risk_metric)
        ax.set_title(f"Orientation-predicted feed vs {risk_metric}")
        ax.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=180)
    plt.close(fig)


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if table.empty:
        return "_No rows._"
    view = table.loc[:, [col for col in columns if col in table.columns]].head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_markdown_summary(
    outdir: Path,
    stream_path: Path,
    reflection_scores_path: Path,
    feed_rows: int,
    score_rows: int,
    joined: pd.DataFrame,
    correlations: pd.DataFrame,
    candidate_metadata: dict[str, object],
    d_min: float,
    d_max: float,
    s_max: float,
    max_frames: int | None,
    has_frame_axis: bool,
) -> None:
    global_corr = correlations.loc[correlations["scope"] == "all_observations"]
    per_frame_dist = correlations.loc[correlations["scope"] == "per_frame_distribution"]
    frame_axis_text = "present and compared" if has_frame_axis else "not present in reflection_scores.csv"
    text = f"""# MP15 Orientation-Predicted Feed vs Graph/Frame Risk

## Match Counts

- Stream observations scored: {feed_rows}
- OriDyn reflection score rows loaded: {score_rows}
- Exact observation matches: {len(joined)}
- Max frames: {max_frames if max_frames is not None else "all"}
- Candidate HKLs: {candidate_metadata.get("n_candidates")}
- Candidate bounds: {candidate_metadata.get("hkl_bounds")}
- d range: {d_min:g} to {d_max:g} A
- s_max: {s_max:g} A^-1
- frame_axis_risk_norm: {frame_axis_text}
- Stream: `{stream_path}`
- Reflection scores: `{reflection_scores_path}`

## How The Orientation-Predicted Feed Score Is Computed

- The stream supplies each indexed frame's reciprocal matrix and observed signed target HKLs.
- Candidate HKLs are generated once from the parsed stream cell over `d_min <= d <= d_max`.
- For each frame, candidate reciprocal vectors are mapped through that frame's orientation.
- Candidate source beams `q` are predicted by excitation error `s(q)` and kept when `|s(q)| <= s_max`.
- Source excitation weight is `E(q) = exp(-0.5 * (s(q) / s_max)^2)`.
- The low-order weight is `W(n) = exp(-(|G(n)| / G0)^2)` with `G0 = 3 * min_nonzero_G_norm` for that frame's candidate set.
- Each observed target `g` is scored by valid build-up paths `q + r = g`, where `r` must also be a valid candidate HKL.
- Valid paths require lower-order forward geometry: `|G(q)| < |G(g)|`, `|G(r)| < |G(g)|`, `dot(G(q), G(g)) > 0`, and `dot(G(r), G(g)) > 0`.
- `S_feed_raw(g)` is the sum of `E(q) * W(q) * W(r)`.
- `S_feed_norm_frame(g)` is `S_feed_raw / p95_positive_in_frame`.
- The join to OriDyn uses exact `source_filename + event + signed h,k,l`; no symmetry canonicalization is applied.
- This comparison intentionally excludes `self_risk_norm`, `S_dyn_geom`, `sigma_dyn_rel`, and `nonself_mean`.

## Global Spearman Correlations

{markdown_table(global_corr, ["feed_metric", "risk_metric", "n_pairs", "spearman_r"])}

## Per-Frame Spearman Summary

Frames enter this summary only when they have at least {MIN_PER_FRAME_CORR_REFLECTIONS} matched observations.

{markdown_table(per_frame_dist, ["feed_metric", "risk_metric", "n_pairs", "spearman_min", "spearman_p05", "spearman_median", "spearman_p95", "spearman_max"])}

## Top Example CSVs

- `top_high_feed_high_graph.csv`
- `top_high_feed_low_graph.csv`
- `top_high_graph_low_feed.csv`
"""
    (outdir / "comparison_summary.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.outdir, args.overwrite)

    print(f"Parsing stream: {args.stream}", file=sys.stderr)
    stream = parse_stream(args.stream, args.max_frames)
    print(
        f"Parsed {len(stream.crystal_table)} indexed frame(s), {len(stream.reflections)} observed reflection(s)",
        file=sys.stderr,
    )
    print(f"Generating candidate HKLs for {args.d_min:g} <= d <= {args.d_max:g} A", file=sys.stderr)
    candidates, candidate_metadata = prepare_candidates(stream, args.d_min, args.d_max)
    print(f"Candidate HKLs: {len(candidates)}", file=sys.stderr)
    print(f"Loading reflection scores: {args.reflection_scores}", file=sys.stderr)
    reflection_scores, has_frame_axis = load_reflection_scores(args.reflection_scores)
    print(f"Loaded {len(reflection_scores)} reflection score row(s)", file=sys.stderr)

    feed_scores = compute_frame_scores(stream, candidates, args.s_max)
    joined = join_scores(feed_scores, reflection_scores)

    output_cols = [
        "source_filename",
        "event",
        "frame",
        "frame_number",
        "h",
        "k",
        "l",
        "S_feed_raw",
        "S_feed_norm_frame",
        "n_predicted_q",
        "n_valid_feed_paths",
        GRAPH_COLUMN,
    ]
    if has_frame_axis:
        output_cols.append(FRAME_COLUMN)
    joined[output_cols].to_csv(args.outdir / "framewise_orientation_predicted_feed_vs_graph_frame.csv", index=False)

    correlations = build_correlation_summary(joined, has_frame_axis)
    correlations.to_csv(args.outdir / "correlation_summary.csv", index=False)
    write_top_examples(joined, args.outdir, has_frame_axis)
    plot_scatter(joined, args.outdir, GRAPH_COLUMN, "scatter_feed_vs_graph_crowding.png")
    if has_frame_axis:
        plot_scatter(joined, args.outdir, FRAME_COLUMN, "scatter_feed_vs_frame_axis.png")
    write_markdown_summary(
        args.outdir,
        args.stream,
        args.reflection_scores,
        len(feed_scores),
        len(reflection_scores),
        joined,
        correlations,
        candidate_metadata,
        args.d_min,
        args.d_max,
        args.s_max,
        args.max_frames,
        has_frame_axis,
    )

    print(f"Orientation-predicted feed rows: {len(feed_scores)}")
    print(f"Matched observation rows: {len(joined)}")
    print(f"Frame-axis comparison included: {has_frame_axis}")
    print(f"Outputs written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
