#!/usr/bin/env python3
"""Compare enhancement-feed HKL scores with OriDyn graph/frame risk terms.

The enhancement table is HKL-level. The OriDyn reflection_scores table is
observation-level and is reduced to signed HKL medians before joining. No
symmetry canonicalization is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HKL_COLUMNS = ["h", "k", "l"]
ENHANCEMENT_COLUMNS = [
    "h",
    "k",
    "l",
    "laue_zone",
    "target_excitation_Eg",
    "S_enh_raw",
    "S_enh_norm_global",
    "S_enh_norm_zone",
    "included_in_diffraction_plot",
]
GRAPH_COLUMN = "graph_crowding_norm"
FRAME_COLUMN = "frame_axis_risk_norm"
GRAPH_MEDIAN_COLUMN = "median_graph_crowding_norm"
FRAME_MEDIAN_COLUMN = "median_frame_axis_risk_norm"
ENHANCEMENT_COMPARE_COLUMNS = ["S_enh_raw", "S_enh_norm_zone"]
MIN_CORRELATION_POINTS = 5
TOP_EXAMPLE_ROWS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhancement-csv", required=True, type=Path, help="Path to enhancement_risk_scores.csv")
    parser.add_argument("--reflection-scores", required=True, type=Path, help="Path to OriDyn reflection_scores.csv")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory for joined tables and plots")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise SystemExit(f"{label} is missing required column(s): {missing}")


def load_enhancement(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"--enhancement-csv not found: {path}")
    table = pd.read_csv(path)
    require_columns(table, ENHANCEMENT_COLUMNS, "enhancement CSV")

    out = table[ENHANCEMENT_COLUMNS].copy()
    for col in HKL_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["laue_zone", "target_excitation_Eg", "S_enh_raw", "S_enh_norm_global", "S_enh_norm_zone"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["included_in_diffraction_plot"] = parse_bool_series(out["included_in_diffraction_plot"])
    out = out.dropna(subset=HKL_COLUMNS)
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")
    return out


def load_reflection_scores(path: Path) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        raise SystemExit(f"--reflection-scores not found: {path}")
    table = pd.read_csv(path)
    require_columns(table, [*HKL_COLUMNS, GRAPH_COLUMN], "reflection_scores.csv")

    has_frame_axis = FRAME_COLUMN in table.columns
    keep = [*HKL_COLUMNS, GRAPH_COLUMN]
    if has_frame_axis:
        keep.append(FRAME_COLUMN)

    out = table[keep].copy()
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=HKL_COLUMNS)
    out[HKL_COLUMNS] = out[HKL_COLUMNS].astype("int64")

    agg_spec: dict[str, tuple[str, str]] = {
        "n_observations": (GRAPH_COLUMN, "size"),
        GRAPH_MEDIAN_COLUMN: (GRAPH_COLUMN, "median"),
    }
    if has_frame_axis:
        agg_spec[FRAME_MEDIAN_COLUMN] = (FRAME_COLUMN, "median")
    grouped = out.groupby(HKL_COLUMNS, sort=True).agg(**agg_spec).reset_index()
    return grouped, has_frame_axis


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def prepare_output_dir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise SystemExit(f"{outdir} exists and is not empty; pass --overwrite to reuse it.")
    outdir.mkdir(parents=True, exist_ok=True)


def spearman_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < MIN_CORRELATION_POINTS:
        return np.nan, int(len(frame))
    if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan, int(len(frame))
    return float(frame["x"].rank(method="average").corr(frame["y"].rank(method="average"))), int(len(frame))


def add_correlation_rows(
    rows: list[dict[str, object]],
    table: pd.DataFrame,
    scope: str,
    target_metric: str,
    target_label: str,
    laue_zone: int | None = None,
) -> None:
    for enhancement_metric in ENHANCEMENT_COMPARE_COLUMNS:
        rho, n_pairs = spearman_corr(table[enhancement_metric], table[target_metric])
        rows.append(
            {
                "scope": scope,
                "laue_zone": "" if laue_zone is None else int(laue_zone),
                "enhancement_metric": enhancement_metric,
                "oridyn_metric": target_label,
                "n_pairs": n_pairs,
                "spearman_r": rho,
            }
        )


def correlation_summary(joined: pd.DataFrame, has_frame_axis: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = [(GRAPH_MEDIAN_COLUMN, "median_graph_crowding_norm")]
    if has_frame_axis:
        targets.append((FRAME_MEDIAN_COLUMN, "median_frame_axis_risk_norm"))

    for target_metric, target_label in targets:
        add_correlation_rows(rows, joined, "all_matched", target_metric, target_label)
        plotted = joined.loc[joined["included_in_diffraction_plot"]].copy()
        add_correlation_rows(rows, plotted, "plotted_only", target_metric, target_label)

        for zone, zone_group in joined.groupby("laue_zone", sort=True):
            if len(zone_group.dropna(subset=[target_metric])) >= MIN_CORRELATION_POINTS:
                add_correlation_rows(rows, zone_group, "all_matched_by_laue_zone", target_metric, target_label, int(zone))
        for zone, zone_group in plotted.groupby("laue_zone", sort=True):
            if len(zone_group.dropna(subset=[target_metric])) >= MIN_CORRELATION_POINTS:
                add_correlation_rows(rows, zone_group, "plotted_by_laue_zone", target_metric, target_label, int(zone))

    return pd.DataFrame.from_records(rows)


def write_top_examples(joined: pd.DataFrame, outdir: Path, has_frame_axis: bool) -> dict[str, pd.DataFrame]:
    enhancement_metric = "S_enh_norm_zone"
    graph_metric = GRAPH_MEDIAN_COLUMN
    enh_high = float(joined[enhancement_metric].quantile(0.90))
    enh_low = float(joined[enhancement_metric].quantile(0.50))
    graph_high = float(joined[graph_metric].quantile(0.90))
    graph_low = float(joined[graph_metric].quantile(0.50))

    columns = [
        "h",
        "k",
        "l",
        "laue_zone",
        "included_in_diffraction_plot",
        "target_excitation_Eg",
        "S_enh_raw",
        "S_enh_norm_global",
        "S_enh_norm_zone",
        GRAPH_MEDIAN_COLUMN,
        "n_observations",
    ]
    if has_frame_axis:
        columns.append(FRAME_MEDIAN_COLUMN)

    examples = {
        "top_high_enh_high_graph": joined.loc[
            (joined[enhancement_metric] >= enh_high) & (joined[graph_metric] >= graph_high), columns
        ].sort_values([enhancement_metric, graph_metric], ascending=[False, False]),
        "top_high_enh_low_graph": joined.loc[
            (joined[enhancement_metric] >= enh_high) & (joined[graph_metric] <= graph_low), columns
        ].sort_values([enhancement_metric, graph_metric], ascending=[False, True]),
        "top_high_graph_low_enh": joined.loc[
            (joined[graph_metric] >= graph_high) & (joined[enhancement_metric] <= enh_low), columns
        ].sort_values([graph_metric, enhancement_metric], ascending=[False, True]),
    }
    for name, table in examples.items():
        table.head(TOP_EXAMPLE_ROWS).to_csv(outdir / f"{name}.csv", index=False)
    return {name: table.head(10).copy() for name, table in examples.items()}


def plot_scatter(joined: pd.DataFrame, outdir: Path, y_column: str, y_label: str, output_name: str) -> None:
    plot_data = joined.dropna(subset=["S_enh_norm_zone", y_column, "laue_zone"]).copy()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    if plot_data.empty:
        ax.text(0.5, 0.5, "No matched HKLs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        scatter = ax.scatter(
            plot_data["S_enh_norm_zone"],
            plot_data[y_column],
            c=plot_data["laue_zone"],
            cmap="tab20",
            s=22,
            alpha=0.78,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax, label="Laue zone")
        ax.set_xlabel("S_enh_norm_zone")
        ax.set_ylabel(y_label)
        ax.set_title(f"Enhancement feed vs {y_label}")
        ax.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / output_name, dpi=180)
    plt.close(fig)


def markdown_table(table: pd.DataFrame, columns: list[str], max_rows: int = 8) -> str:
    if table.empty:
        return "_No rows._"
    view = table.loc[:, [col for col in columns if col in table.columns]].head(max_rows).copy()
    for col in view.select_dtypes(include=[np.number]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def strongest_graph_relationship(correlations: pd.DataFrame) -> str:
    main = correlations.loc[
        (correlations["scope"] == "all_matched")
        & (correlations["oridyn_metric"] == "median_graph_crowding_norm")
        & (correlations["enhancement_metric"] == "S_enh_norm_zone")
    ]
    if main.empty or pd.isna(main.iloc[0]["spearman_r"]):
        return "The joined data were insufficient to judge ranking overlap."
    rho = float(main.iloc[0]["spearman_r"])
    abs_rho = abs(rho)
    if abs_rho >= 0.8:
        return f"The enhancement-feed ranking looks broadly similar to graph crowding (Spearman r={rho:.3f})."
    if abs_rho >= 0.5:
        return f"The enhancement-feed ranking partially overlaps graph crowding but is not the same ordering (Spearman r={rho:.3f})."
    return f"The enhancement-feed ranking appears to add a different ordering from graph crowding (Spearman r={rho:.3f})."


def write_markdown_summary(
    outdir: Path,
    enhancement_path: Path,
    reflection_path: Path,
    enhancement_rows: int,
    aggregated_rows: int,
    joined: pd.DataFrame,
    correlations: pd.DataFrame,
    examples: dict[str, pd.DataFrame],
    has_frame_axis: bool,
) -> None:
    graph_corr = correlations.loc[correlations["oridyn_metric"] == "median_graph_crowding_norm"]
    frame_corr = (
        correlations.loc[correlations["oridyn_metric"] == "median_frame_axis_risk_norm"]
        if has_frame_axis
        else pd.DataFrame()
    )
    top_columns = [
        "h",
        "k",
        "l",
        "laue_zone",
        "S_enh_norm_zone",
        GRAPH_MEDIAN_COLUMN,
        "n_observations",
    ]
    if has_frame_axis:
        top_columns.append(FRAME_MEDIAN_COLUMN)

    text = f"""# Enhancement vs Graph/Frame Risk Comparison

## Match Counts

- Enhancement rows: {enhancement_rows}
- Aggregated signed-HKL OriDyn rows: {aggregated_rows}
- Joined signed-HKL rows: {len(joined)}
- Enhancement CSV: `{enhancement_path}`
- Reflection scores CSV: `{reflection_path}`

## How This Is Computed

- The enhancement table is treated as one row per signed `(h,k,l)`.
- `reflection_scores.csv` is observation-level, so `graph_crowding_norm` is aggregated by exact signed `(h,k,l)` using the median.
- `frame_axis_risk_norm` is handled the same way when present.
- The join is an inner join on signed `h,k,l` only. No Laue or point-group symmetry canonicalization is applied.
- Spearman correlation is computed as Pearson correlation of average ranks after dropping non-finite pairs. Rows with fewer than {MIN_CORRELATION_POINTS} finite pairs are reported as `NaN`.
- Top-example CSVs use `S_enh_norm_zone` as the enhancement ranking and median `graph_crowding_norm` as the graph ranking. `high` means top 10 percent; `low` means bottom 50 percent.
- The comparison intentionally excludes `self_risk_norm`, `S_dyn_geom`, `sigma_dyn_rel`, and `nonself_mean`.

## Enhancement vs Graph Crowding

{markdown_table(graph_corr, ["scope", "laue_zone", "enhancement_metric", "oridyn_metric", "n_pairs", "spearman_r"], max_rows=20)}

## Enhancement vs Frame-Axis Risk

{markdown_table(frame_corr, ["scope", "laue_zone", "enhancement_metric", "oridyn_metric", "n_pairs", "spearman_r"], max_rows=20) if has_frame_axis else "_`frame_axis_risk_norm` was not present in the reflection scores table._"}

## Ranking Interpretation

{strongest_graph_relationship(correlations)}

## Top Examples

### High Enhancement, High Graph

{markdown_table(examples["top_high_enh_high_graph"], top_columns)}

### High Enhancement, Low Graph

{markdown_table(examples["top_high_enh_low_graph"], top_columns)}

### High Graph, Low Enhancement

{markdown_table(examples["top_high_graph_low_enh"], top_columns)}
"""
    (outdir / "comparison_summary.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepare_output_dir(args.outdir, args.overwrite)

    enhancement = load_enhancement(args.enhancement_csv)
    oridyn_hkl, has_frame_axis = load_reflection_scores(args.reflection_scores)
    joined = enhancement.merge(oridyn_hkl, on=HKL_COLUMNS, how="inner", validate="one_to_one")

    joined_path = args.outdir / "joined_enhancement_graph_frame.csv"
    joined.to_csv(joined_path, index=False)

    correlations = correlation_summary(joined, has_frame_axis)
    correlations.to_csv(args.outdir / "correlation_summary.csv", index=False)

    examples = write_top_examples(joined, args.outdir, has_frame_axis)
    plot_scatter(joined, args.outdir, GRAPH_MEDIAN_COLUMN, "median graph_crowding_norm", "scatter_enh_vs_graph_crowding.png")
    if has_frame_axis:
        plot_scatter(joined, args.outdir, FRAME_MEDIAN_COLUMN, "median frame_axis_risk_norm", "scatter_enh_vs_frame_axis.png")

    write_markdown_summary(
        args.outdir,
        args.enhancement_csv,
        args.reflection_scores,
        len(enhancement),
        len(oridyn_hkl),
        joined,
        correlations,
        examples,
        has_frame_axis,
    )

    print(f"Enhancement rows: {len(enhancement)}")
    print(f"Aggregated signed-HKL OriDyn rows: {len(oridyn_hkl)}")
    print(f"Joined signed-HKL rows: {len(joined)}")
    print(f"Frame-axis comparison included: {has_frame_axis}")
    print(f"Outputs written to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
