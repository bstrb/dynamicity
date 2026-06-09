"""Plotting utilities for OriDyn output tables."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oridyn-mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_standard_plots(reflection_scores: pd.DataFrame, frame_summary: pd.DataFrame, output_dir: str | Path) -> None:
    """Write the required standard plot set."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_score_distributions(reflection_scores, out)
    plot_frame_risk_trace(frame_summary, out)
    plot_score_term_correlations(reflection_scores, out)


def plot_score_distributions(scores: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        col
        for col in (
            "self_risk_raw",
            "graph_crowding_raw",
            "same_laue_zone_crowding_raw",
            "systematic_row_risk_raw",
            "S_dyn_geom",
            "sigma_dyn_rel",
        )
        if col in scores
    ]
    n_rows = max(len(columns), 1)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, max(3.0, 2.4 * n_rows)))
    axes = np.asarray(axes).reshape(n_rows, 2)
    if not columns:
        axes[0, 0].text(0.5, 0.5, "No reflection scores", ha="center", va="center")
        axes[0, 0].set_axis_off()
        axes[0, 1].set_axis_off()
    for row_idx, col in enumerate(columns):
        ax_full = axes[row_idx, 0]
        ax_tail = axes[row_idx, 1]
        values = pd.to_numeric(scores[col], errors="coerce").dropna()
        if values.empty:
            ax_full.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax_tail.text(0.5, 0.5, "No finite values", ha="center", va="center")
            continue
        ax_full.hist(values, bins=80, color="#4c78a8", alpha=0.88)
        ax_full.set_yscale("log")
        ax_full.set_title(f"{col}: all values")
        ax_full.set_ylabel("count (log)")
        _add_quantile_lines(ax_full, values)

        eps = max(float(values.abs().max()) * 1e-12, 1e-12)
        tail = values.loc[values.abs() > eps]
        if tail.empty:
            ax_tail.text(0.5, 0.5, "Only near-zero values", ha="center", va="center")
            ax_tail.set_axis_off()
            continue
        lower = float(tail.quantile(0.005))
        upper = float(tail.quantile(0.995))
        clipped_tail = tail.loc[(tail >= lower) & (tail <= upper)]
        if clipped_tail.empty:
            clipped_tail = tail
        ax_tail.hist(clipped_tail, bins=80, color="#f58518", alpha=0.88)
        ax_tail.set_yscale("log")
        ax_tail.set_title(f"{col}: nonzero tail, 0.5-99.5 pct")
        ax_tail.set_ylabel("count (log)")
        _add_quantile_lines(ax_tail, clipped_tail)
    for ax in axes[-1, :]:
        ax.set_xlabel("score")
    _save(fig, output_dir / "score_distributions")


def _add_quantile_lines(ax: plt.Axes, values: pd.Series) -> None:
    for quantile, color in ((0.50, "#333333"), (0.95, "#b279a2")):
        value = float(values.quantile(quantile))
        ax.axvline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(value, 0.96, f"p{int(quantile * 100)}", transform=ax.get_xaxis_transform(), fontsize=7, va="top")


def plot_frame_risk_trace(frame_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if frame_summary.empty:
        ax.text(0.5, 0.5, "No frame scores", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_dir / "frame_risk_trace")
        return
    x = frame_summary["frame_number"] if "frame_number" in frame_summary else frame_summary["frame"]
    y_col = "frame_axis_risk_norm" if "frame_axis_risk_norm" in frame_summary else "frame_axis_risk_raw"
    labels = frame_summary["assigned_risky_axis"].astype(str) if "assigned_risky_axis" in frame_summary else pd.Series([""] * len(frame_summary))
    categories = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    colors = labels.map(categories).to_numpy(dtype=float)
    scatter = ax.scatter(x, frame_summary[y_col], c=colors, cmap="tab20", s=18)
    ax.plot(x, frame_summary[y_col], color="#333333", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("frame number")
    ax.set_ylabel(y_col)
    ax.set_title("Frame axis risk")
    if len(categories) <= 12 and len(categories) > 1:
        handles = []
        cmap = scatter.cmap
        norm = scatter.norm
        for label, idx in categories.items():
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(norm(idx)), label=label))
        ax.legend(handles=handles, fontsize=7, loc="best", title="axis")
    _save(fig, output_dir / "frame_risk_trace")


def plot_score_term_correlations(scores: pd.DataFrame, output_dir: Path) -> None:
    columns = [
        col
        for col in (
            "self_risk_norm",
            "graph_crowding_norm",
            "same_laue_zone_crowding_norm",
            "systematic_row_risk_norm",
            "frame_axis_risk_norm",
            "S_dyn_geom",
        )
        if col in scores
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(columns) < 2:
        ax.text(0.5, 0.5, "Not enough score columns", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_dir / "score_term_correlations")
        return
    corr = scores[columns].corr(numeric_only=True).fillna(0.0)
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Score term correlations")
    _save(fig, output_dir / "score_term_correlations")


def plot_residuals(scores_path: str | Path, residuals_path: str | Path, output_dir: str | Path) -> None:
    """Join externally supplied residuals and plot residual magnitude by score."""

    scores = pd.read_csv(scores_path)
    residuals = pd.read_csv(residuals_path)
    join_cols = [col for col in ("frame", "h", "k", "l") if col in scores.columns and col in residuals.columns]
    if not join_cols:
        raise ValueError("Residual table must share frame/h/k/l columns with reflection scores.")
    joined = scores.merge(residuals, on=join_cols, how="inner", suffixes=("", "_residual"))
    residual_col = _find_residual_column(joined)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if joined.empty or residual_col is None:
        ax.text(0.5, 0.5, "No joined residuals", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, out / "residual_vs_score")
        return
    ax.scatter(joined["S_dyn_geom"], joined[residual_col].abs(), s=12, alpha=0.6, color="#f58518")
    ax.set_xlabel("S_dyn_geom")
    ax.set_ylabel(f"abs({residual_col})")
    ax.set_title("External residuals versus geometry score")
    _save(fig, out / "residual_vs_score")


def _find_residual_column(table: pd.DataFrame) -> str | None:
    for column in ("residual", "signed_residual", "abs_residual", "delta", "R"):
        if column in table:
            return column
    numeric = [col for col in table.select_dtypes(include=[np.number]).columns if col not in {"frame", "h", "k", "l", "S_dyn_geom"}]
    return numeric[-1] if numeric else None


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

HKL_COMPONENT_COLUMNS = (
    "self_risk_norm",
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
)


def make_hkl_trace_plots(
    scores: pd.DataFrame,
    output_dir: str | Path,
    score_columns: list[str] | None = None,
    max_hkls: int = 40,
) -> None:
    """Create selected-HKL trajectory plots from `hkl_frame_trajectories.csv`.

    The plots are intentionally diagnostic: they show how geometry-only score
    components change with frame/orientation for a small selected HKL set.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scores.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No HKL trajectory scores", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, out / "hkl_traces_empty")
        return
    label_col = "hkl_label" if "hkl_label" in scores.columns else None
    if label_col is None:
        scores = scores.copy()
        scores["hkl_label"] = [
            f"{int(h)}_{int(k)}_{int(l)}" for h, k, l in scores[["h", "k", "l"]].itertuples(index=False, name=None)
        ]
        label_col = "hkl_label"
    score_columns = score_columns or _default_hkl_score_columns(scores)
    if not score_columns:
        score_columns = ["S_dyn_geom"] if "S_dyn_geom" in scores.columns else []
    _plot_hkl_score_lines(scores, out, score_columns, label_col, max_hkls=max_hkls)
    _plot_hkl_component_lines(scores, out, label_col, max_hkls=max_hkls)
    for score_column in score_columns[:4]:
        if score_column in scores.columns:
            _plot_hkl_heatmap(scores, out, score_column, label_col, max_hkls=max_hkls)
    _plot_component_dominance(scores, out, label_col, max_hkls=max_hkls)


def _default_hkl_score_columns(scores: pd.DataFrame) -> list[str]:
    columns = [col for col in scores.columns if col.startswith("S_dyn_geom")]
    if "S_dyn_geom" in scores.columns:
        columns = ["S_dyn_geom"] + [col for col in columns if col != "S_dyn_geom"]
    return columns[:8]


def _plot_hkl_score_lines(scores: pd.DataFrame, output_dir: Path, score_columns: list[str], label_col: str, max_hkls: int) -> None:
    plot_scores = _limit_hkls_for_plot(scores, label_col, max_hkls)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_col = "frame_number" if "frame_number" in plot_scores.columns else "frame"
    if not score_columns:
        ax.text(0.5, 0.5, "No score columns", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_dir / "hkl_score_traces")
        return
    for label, group in plot_scores.groupby(label_col, sort=False):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[score_columns[0]], linewidth=1.3, alpha=0.85, label=str(label))
    ax.set_xlabel("frame number" if x_col == "frame_number" else "frame")
    ax.set_ylabel(score_columns[0])
    ax.set_title(f"Selected-HKL risk traces: {score_columns[0]}")
    if plot_scores[label_col].nunique() <= 16:
        ax.legend(fontsize=7, ncol=2)
    _save(fig, output_dir / "hkl_score_traces")

    if len(score_columns) > 1:
        for label, group in plot_scores.groupby(label_col, sort=False):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            group = group.sort_values(x_col)
            for column in score_columns:
                if column in group.columns:
                    ax.plot(group[x_col], group[column], linewidth=1.2, label=column)
            ax.set_xlabel("frame number" if x_col == "frame_number" else "frame")
            ax.set_ylabel("score")
            ax.set_title(f"Weight-preset comparison for {label}")
            ax.legend(fontsize=7)
            _save(fig, output_dir / f"hkl_{_safe_filename(label)}_score_preset_comparison")


def _plot_hkl_component_lines(scores: pd.DataFrame, output_dir: Path, label_col: str, max_hkls: int) -> None:
    plot_scores = _limit_hkls_for_plot(scores, label_col, max_hkls)
    x_col = "frame_number" if "frame_number" in plot_scores.columns else "frame"
    columns = [col for col in HKL_COMPONENT_COLUMNS if col in plot_scores.columns]
    if not columns:
        return
    for label, group in plot_scores.groupby(label_col, sort=False):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        group = group.sort_values(x_col)
        for column in columns:
            ax.plot(group[x_col], group[column], linewidth=1.2, label=column)
        ax.set_xlabel("frame number" if x_col == "frame_number" else "frame")
        ax.set_ylabel("normalized component")
        ax.set_title(f"Score components for {label}")
        ax.legend(fontsize=7)
        _save(fig, output_dir / f"hkl_{_safe_filename(label)}_component_trace")


def _plot_hkl_heatmap(scores: pd.DataFrame, output_dir: Path, score_column: str, label_col: str, max_hkls: int) -> None:
    plot_scores = _limit_hkls_for_plot(scores, label_col, max_hkls)
    x_col = "frame_number" if "frame_number" in plot_scores.columns else "frame"
    pivot = plot_scores.pivot_table(index=label_col, columns=x_col, values=score_column, aggfunc="mean")
    if pivot.empty:
        return
    fig_width = max(7.0, min(20.0, 0.16 * pivot.shape[1] + 4.0))
    fig_height = max(3.0, min(18.0, 0.28 * pivot.shape[0] + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
    if pivot.shape[1] <= 30:
        ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns], rotation=90, fontsize=7)
    else:
        ticks = np.linspace(0, pivot.shape[1] - 1, min(10, pivot.shape[1]), dtype=int)
        ax.set_xticks(ticks, [str(pivot.columns[i]) for i in ticks])
    ax.set_xlabel("frame number" if x_col == "frame_number" else "frame")
    ax.set_ylabel("HKL")
    ax.set_title(f"HKL x frame heatmap: {score_column}")
    fig.colorbar(image, ax=ax, label=score_column)
    _save(fig, output_dir / f"hkl_heatmap_{_safe_filename(score_column)}")


def _plot_component_dominance(scores: pd.DataFrame, output_dir: Path, label_col: str, max_hkls: int) -> None:
    columns = [col for col in HKL_COMPONENT_COLUMNS if col in scores.columns]
    if not columns:
        return
    plot_scores = _limit_hkls_for_plot(scores, label_col, max_hkls).copy()
    x_col = "frame_number" if "frame_number" in plot_scores.columns else "frame"
    values = plot_scores[columns].to_numpy(dtype=float)
    dominance = np.argmax(values, axis=1)
    plot_scores["dominant_component"] = [columns[idx] for idx in dominance]
    component_codes = {name: idx for idx, name in enumerate(columns)}
    pivot = plot_scores.pivot_table(
        index=label_col,
        columns=x_col,
        values="dominant_component",
        aggfunc=lambda s: component_codes.get(str(s.iloc[0]), 0),
    )
    if pivot.empty:
        return
    fig_width = max(7.0, min(20.0, 0.16 * pivot.shape[1] + 4.0))
    fig_height = max(3.0, min(18.0, 0.28 * pivot.shape[0] + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", vmin=0, vmax=max(len(columns) - 1, 1))
    ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
    if pivot.shape[1] <= 30:
        ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns], rotation=90, fontsize=7)
    else:
        ticks = np.linspace(0, pivot.shape[1] - 1, min(10, pivot.shape[1]), dtype=int)
        ax.set_xticks(ticks, [str(pivot.columns[i]) for i in ticks])
    ax.set_xlabel("frame number" if x_col == "frame_number" else "frame")
    ax.set_ylabel("HKL")
    ax.set_title("Dominant normalized risk component")
    cbar = fig.colorbar(image, ax=ax, ticks=list(component_codes.values()))
    cbar.ax.set_yticklabels(list(component_codes.keys()))
    _save(fig, output_dir / "hkl_component_dominance")


def _limit_hkls_for_plot(scores: pd.DataFrame, label_col: str, max_hkls: int) -> pd.DataFrame:
    if scores[label_col].nunique() <= max_hkls:
        return scores.copy()
    ranking_col = "S_dyn_geom" if "S_dyn_geom" in scores.columns else None
    if ranking_col is None:
        keep = list(scores[label_col].drop_duplicates().head(max_hkls))
    else:
        keep = (
            scores.groupby(label_col)[ranking_col]
            .max()
            .sort_values(ascending=False)
            .head(max_hkls)
            .index.tolist()
        )
    return scores[scores[label_col].isin(keep)].copy()


def _safe_filename(value: object) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "value"
