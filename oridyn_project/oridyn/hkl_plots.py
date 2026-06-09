"""Plots for selected-HKL trajectory exploration."""

from __future__ import annotations

import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oridyn-mpl-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NORM_TERM_COLUMNS = [
    "self_risk_norm",
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

COMPONENT_COLUMNS = [
    "S_self_component",
    "S_graph_component",
    "S_zone_component",
    "S_row_component",
    "S_frame_component",
    "S_interaction_component",
]


def make_hkl_trace_plots(
    scores: pd.DataFrame,
    output_dir: str | Path,
    score_columns: list[str] | None = None,
    max_hkls: int = 50,
) -> None:
    """Create overview plots for selected HKL trajectories across frames."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scores.empty:
        _empty_plot(out / "hkl_score_heatmap", "No HKL trajectory scores")
        return
    table = _with_hkl_labels(scores)
    score_cols = _score_columns(table, score_columns)
    frame_col = "frame_number" if "frame_number" in table else "frame"
    if score_cols:
        _plot_score_heatmap(table, out / "hkl_score_heatmap", score_cols[0], frame_col)
        _plot_score_overview(table, out / "hkl_score_overview", score_cols, frame_col)
    _plot_component_heatmap(table, out / "hkl_component_dominance", frame_col)
    hkl_labels = list(dict.fromkeys(table["hkl_label_plot"].astype(str).tolist()))[: max(0, int(max_hkls))]
    trace_dir = out / "hkl_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for label in hkl_labels:
        sub = table.loc[table["hkl_label_plot"] == label].sort_values(frame_col)
        if score_cols:
            _plot_one_hkl_scores(sub, trace_dir / f"{_safe_name(label)}_scores", label, score_cols, frame_col)
        _plot_one_hkl_components(sub, trace_dir / f"{_safe_name(label)}_components", label, frame_col)


def _with_hkl_labels(scores: pd.DataFrame) -> pd.DataFrame:
    table = scores.copy()
    if "hkl_label" in table:
        labels = table["hkl_label"].fillna("").astype(str)
    else:
        labels = pd.Series([""] * len(table), index=table.index)
    fallback = "(" + table["h"].astype(str) + " " + table["k"].astype(str) + " " + table["l"].astype(str) + ")"
    table["hkl_label_plot"] = labels.where(labels.str.len() > 0, fallback)
    return table


def _score_columns(table: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        return [col for col in requested if col in table]
    cols = []
    if "S_dyn_geom" in table:
        cols.append("S_dyn_geom")
    cols.extend(col for col in table.columns if col.startswith("S_dyn_geom_") and col not in cols)
    return cols


def _plot_score_heatmap(table: pd.DataFrame, stem: Path, score_col: str, frame_col: str) -> None:
    pivot = table.pivot_table(index="hkl_label_plot", columns=frame_col, values=score_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(7.0, 0.18 * pivot.shape[1]), max(3.0, 0.35 * pivot.shape[0])))
    if pivot.empty:
        ax.text(0.5, 0.5, "No heatmap values", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, stem)
        return
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    step = max(1, pivot.shape[1] // 12)
    xticks = list(range(0, pivot.shape[1], step))
    ax.set_xticks(xticks, [str(pivot.columns[i]) for i in xticks], rotation=45, ha="right")
    ax.set_xlabel("frame")
    ax.set_ylabel("HKL")
    ax.set_title(f"Selected-HKL trajectory heatmap: {score_col}")
    fig.colorbar(image, ax=ax, label=score_col)
    _save(fig, stem)


def _plot_score_overview(table: pd.DataFrame, stem: Path, score_cols: list[str], frame_col: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, sub in table.groupby("hkl_label_plot", sort=False):
        ordered = sub.sort_values(frame_col)
        if not ordered.empty:
            ax.plot(ordered[frame_col], ordered[score_cols[0]], linewidth=1.2, alpha=0.75, label=str(label))
    ax.set_xlabel("frame")
    ax.set_ylabel(score_cols[0])
    ax.set_title(f"Selected-HKL risk traces: {score_cols[0]}")
    if table["hkl_label_plot"].nunique() <= 12:
        ax.legend(fontsize=7, loc="best")
    _save(fig, stem)


def _plot_component_heatmap(table: pd.DataFrame, stem: Path, frame_col: str) -> None:
    cols = [col for col in COMPONENT_COLUMNS if col in table]
    fig, ax = plt.subplots(figsize=(8, 4))
    if not cols:
        ax.text(0.5, 0.5, "No component columns", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, stem)
        return
    component_names = [col.replace("S_", "").replace("_component", "") for col in cols]
    rows = []
    labels = []
    for label, sub in table.groupby("hkl_label_plot", sort=False):
        mean_components = sub[cols].mean(numeric_only=True).to_numpy(dtype=float)
        rows.append(mean_components)
        labels.append(str(label))
    matrix = np.vstack(rows) if rows else np.empty((0, len(cols)))
    if matrix.size == 0:
        ax.text(0.5, 0.5, "No component values", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, stem)
        return
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(len(component_names)), component_names, rotation=45, ha="right")
    ax.set_title("Mean score-component contribution per selected HKL")
    fig.colorbar(image, ax=ax, label="mean contribution")
    _save(fig, stem)


def _plot_one_hkl_scores(sub: pd.DataFrame, stem: Path, label: str, score_cols: list[str], frame_col: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for col in score_cols:
        ax.plot(sub[frame_col], sub[col], linewidth=1.4, label=col)
    ax.set_xlabel("frame")
    ax.set_ylabel("score")
    ax.set_title(f"Risk trajectory for {label}")
    ax.legend(fontsize=7, loc="best")
    _save(fig, stem)


def _plot_one_hkl_components(sub: pd.DataFrame, stem: Path, label: str, frame_col: str) -> None:
    cols = [col for col in NORM_TERM_COLUMNS if col in sub]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if not cols:
        ax.text(0.5, 0.5, "No normalized component columns", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, stem)
        return
    for col in cols:
        ax.plot(sub[frame_col], sub[col], linewidth=1.2, label=col.replace("_norm", ""))
    ax.set_xlabel("frame")
    ax.set_ylabel("normalized term")
    ax.set_title(f"Risk-term components for {label}")
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, stem)


def _empty_plot(stem: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    _save(fig, stem)


def _safe_name(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip())
    return safe.strip("_") or "hkl"


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
