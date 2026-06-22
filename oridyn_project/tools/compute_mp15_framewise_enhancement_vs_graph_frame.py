#!/usr/bin/env python3
"""Framewise observed-reflection enhancement-feed comparison for MP15.

For each indexed frame, this tool computes a simple two-step build-up score
using only observed signed HKLs in that same frame, then joins observations to
OriDyn graph/frame risk terms by exact source_filename + event + signed HKL.
No symmetry canonicalization is applied.
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

from oridyn.stream_parser import parse_crystfel_stream, parse_crystfel_stream_text, reciprocal_matrix_from_row


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
RISK_COLUMNS = ["graph_crowding_norm", "frame_axis_risk_norm"]
FEED_COLUMNS = ["S_feed_raw", "S_feed_norm_frame"]
MIN_PER_FRAME_CORR_REFLECTIONS = 30
PROGRESS_EVERY_FRAMES = 25
TOP_EXAMPLE_ROWS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--reflection-scores", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
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
            raise SystemExit(f"Could not parse stream unit cell/orientations: {exc}") from exc
    if max_frames < 1:
        raise SystemExit("--max-frames must be >= 1 when provided.")

    text = stream_prefix_for_first_frames(path, max_frames)
    try:
        return parse_crystfel_stream_text(text, path=str(path))
    except ValueError as exc:
        raise SystemExit(
            "Could not parse unit cell/orientations from the stream prefix. "
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


def load_reflection_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"--reflection-scores not found: {path}")
    required = [*KEY_COLUMNS, *RISK_COLUMNS]
    try:
        scores = pd.read_csv(path, usecols=required)
    except ValueError as exc:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in required if column not in header]
        raise SystemExit(f"reflection_scores.csv is missing required column(s): {missing}") from exc

    out = scores.copy()
    for col in ["source_filename", "event"]:
        out[col] = out[col].astype(str)
    for col in ["h", "k", "l"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in RISK_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["h", "k", "l"])
    out[["h", "k", "l"]] = out[["h", "k", "l"]].astype("int64")

    dup = out.duplicated(KEY_COLUMNS, keep=False)
    if dup.any():
        examples = out.loc[dup, KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"reflection_scores.csv has duplicate exact observation keys, examples: {examples}")
    return out


def proxy_strength(G_norm: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-((np.asarray(G_norm, dtype=float) / max(float(scale), 1e-300)) ** 2))


def hkl_to_G(hkls: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    return np.asarray(hkls, dtype=float) @ np.asarray(reciprocal, dtype=float).T


def compute_frame_scores(stream) -> pd.DataFrame:
    if stream.reflections.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "frame", "frame_number", *FEED_COLUMNS, "n_feed_paths"])

    crystal_by_frame = stream.crystal_table.set_index("frame", drop=False)
    rows: list[pd.DataFrame] = []
    frame_groups = list(stream.reflections.groupby("frame", sort=True))
    for group_idx, (frame, group) in enumerate(frame_groups, start=1):
        if group_idx == 1 or group_idx % PROGRESS_EVERY_FRAMES == 0 or group_idx == len(frame_groups):
            print(f"Scoring frame {group_idx}/{len(frame_groups)} (frame={int(frame)}, reflections={len(group)})", file=sys.stderr)
        crystal_row = crystal_by_frame.loc[int(frame)]
        reciprocal = reciprocal_matrix_from_row(crystal_row)
        rows.append(score_one_frame(group.reset_index(drop=True), reciprocal))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score_one_frame(group: pd.DataFrame, reciprocal: np.ndarray) -> pd.DataFrame:
    base = group[[col for col in ["source_filename", "event", "frame", "frame_number", "h", "k", "l"] if col in group]].copy()
    for col in ["source_filename", "event"]:
        base[col] = base[col].astype(str)
    base[["h", "k", "l"]] = base[["h", "k", "l"]].astype("int64")

    unique_hkls = base[["h", "k", "l"]].drop_duplicates().reset_index(drop=True)
    hkls = unique_hkls[["h", "k", "l"]].to_numpy(dtype=int)
    if len(hkls) == 0:
        base["S_feed_raw"] = 0.0
        base["S_feed_norm_frame"] = 0.0
        base["n_feed_paths"] = 0
        return base

    G = hkl_to_G(hkls, reciprocal)
    G_norm = np.linalg.norm(G, axis=1)
    nonzero = G_norm[G_norm > 0.0]
    G0 = 3.0 * float(np.min(nonzero)) if nonzero.size else 1.0
    W = proxy_strength(G_norm, G0)
    lookup = {tuple(int(x) for x in hkl): idx for idx, hkl in enumerate(hkls)}

    raw = np.zeros(len(hkls), dtype=float)
    n_paths = np.zeros(len(hkls), dtype=int)
    for target_idx, target in enumerate(hkls):
        target_key = tuple(int(x) for x in target)
        target_G = G[target_idx]
        target_norm = G_norm[target_idx]
        for q_idx, q in enumerate(hkls):
            q_key = tuple(int(x) for x in q)
            if q_key == (0, 0, 0):
                continue
            r_key = tuple(int(target[axis] - q[axis]) for axis in range(3))
            if r_key == (0, 0, 0):
                continue
            r_idx = lookup.get(r_key)
            if r_idx is None:
                continue
            if G_norm[q_idx] >= target_norm or G_norm[r_idx] >= target_norm:
                continue
            if float(np.dot(G[q_idx], target_G)) <= 0.0 or float(np.dot(G[r_idx], target_G)) <= 0.0:
                continue
            raw[target_idx] += float(W[q_idx] * W[r_idx])
            n_paths[target_idx] += 1

    positive = raw[raw > 0.0]
    denominator = float(np.quantile(positive, 0.95)) if positive.size else 1.0
    if denominator <= 0.0:
        denominator = 1.0
    unique_hkls["S_feed_raw"] = raw
    unique_hkls["S_feed_norm_frame"] = raw / denominator
    unique_hkls["n_feed_paths"] = n_paths
    return base.merge(unique_hkls, on=["h", "k", "l"], how="left", validate="many_to_one")


def join_scores(feed: pd.DataFrame, reflection_scores: pd.DataFrame) -> pd.DataFrame:
    dup = feed.duplicated(KEY_COLUMNS, keep=False)
    if dup.any():
        examples = feed.loc[dup, KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"stream observations have duplicate exact observation keys, examples: {examples}")
    return feed.merge(reflection_scores, on=KEY_COLUMNS, how="inner", validate="one_to_one")


def spearman_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, int(len(frame))
    ranked = frame.rank(method="average")
    return float(ranked["x"].corr(ranked["y"])), int(len(frame))


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


def build_correlation_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feed_metric in FEED_COLUMNS:
        for risk_metric in RISK_COLUMNS:
            add_corr_row(rows, "all_observations", joined, feed_metric, risk_metric)

    per_frame_rows_start = len(rows)
    for frame, group in joined.groupby("frame", sort=True):
        if len(group) < MIN_PER_FRAME_CORR_REFLECTIONS:
            continue
        source_filename = str(group["source_filename"].iloc[0])
        event = str(group["event"].iloc[0])
        for feed_metric in FEED_COLUMNS:
            for risk_metric in RISK_COLUMNS:
                add_corr_row(rows, "per_frame", group, feed_metric, risk_metric, int(frame), source_filename, event)

    per_frame = pd.DataFrame.from_records(rows[per_frame_rows_start:])
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


def write_top_examples(joined: pd.DataFrame, outdir: Path) -> None:
    feed = "S_feed_norm_frame"
    graph = "graph_crowding_norm"
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
        "n_feed_paths",
        "graph_crowding_norm",
        "frame_axis_risk_norm",
    ]
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
        ax.set_title(f"Framewise enhancement feed vs {risk_metric}")
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
    max_frames: int | None,
) -> None:
    global_corr = correlations.loc[correlations["scope"] == "all_observations"]
    per_frame_dist = correlations.loc[correlations["scope"] == "per_frame_distribution"]
    text = f"""# MP15 Framewise Enhancement vs Graph/Frame Risk

## Match Counts

- Stream observations scored: {feed_rows}
- OriDyn reflection score rows loaded: {score_rows}
- Exact observation matches: {len(joined)}
- Max frames: {max_frames if max_frames is not None else "all"}
- Stream: `{stream_path}`
- Reflection scores: `{reflection_scores_path}`

## How The Framewise Feed Score Is Computed

- For each indexed frame/event, observed signed HKLs from the stream are used as the only available beams.
- For each target `g`, the script loops over observed `q` in the same frame and forms `r = g - q`.
- A path contributes only if `q` and `r` are nonzero, `r` is also observed in the same frame, both `|G(q)|` and `|G(r)|` are smaller than `|G(g)|`, and both dot products with `G(g)` are positive.
- The low-order weight is `W(n) = exp(-(|G(n)| / G0)^2)` with `G0 = 3 * min_nonzero_G_norm` within that frame.
- `S_feed_raw(g)` is the sum of `W(q) * W(r)` over valid paths.
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
    print(f"Loading reflection scores: {args.reflection_scores}", file=sys.stderr)
    reflection_scores = load_reflection_scores(args.reflection_scores)
    print(f"Loaded {len(reflection_scores)} reflection score row(s)", file=sys.stderr)

    feed_scores = compute_frame_scores(stream)
    joined = join_scores(feed_scores, reflection_scores)
    joined.to_csv(args.outdir / "framewise_enhancement_vs_graph_frame.csv", index=False)

    correlations = build_correlation_summary(joined)
    correlations.to_csv(args.outdir / "correlation_summary.csv", index=False)
    write_top_examples(joined, args.outdir)
    plot_scatter(joined, args.outdir, "graph_crowding_norm", "scatter_feed_vs_graph_crowding.png")
    plot_scatter(joined, args.outdir, "frame_axis_risk_norm", "scatter_feed_vs_frame_axis.png")
    write_markdown_summary(
        args.outdir,
        args.stream,
        args.reflection_scores,
        len(feed_scores),
        len(reflection_scores),
        joined,
        correlations,
        args.max_frames,
    )

    print(f"Framewise feed rows: {len(feed_scores)}")
    print(f"Matched observation rows: {len(joined)}")
    print(f"Outputs written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
