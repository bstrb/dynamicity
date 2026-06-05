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
