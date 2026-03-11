"""Visualization helpers for notebook and script workflows."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import pandas as pd

from .parsers import GXPARMData


def plot_frame_summary(
    frame_summary: pd.DataFrame,
    ax: Axes | None = None,
    y: str = "S_MB",
    title: str | None = None,
) -> Axes:
    """Plot a frame-wise summary metric across the experiment."""

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_summary["frame_number"], frame_summary[y], marker="o", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} across frames")
    ax.grid(True, alpha=0.3)
    return ax


def plot_detector_frame(
    frame_table: pd.DataFrame,
    gxparm: GXPARMData,
    rectangles: Iterable[tuple[float, float, float, float]] | None = None,
    score_column: str = "S_comb",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot one frame on the detector using a static matplotlib scatter plot."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("black")
    norm = Normalize(vmin=float(frame_table[score_column].min()) if not frame_table.empty else 0.0,
                     vmax=float(frame_table[score_column].max()) if not frame_table.empty else 1.0)
    if not frame_table.empty:
        scatter = ax.scatter(
            frame_table["x_px"],
            frame_table["y_px"],
            c=frame_table[score_column],
            s=20.0 + 80.0 * norm(frame_table[score_column]),
            cmap="turbo",
            edgecolors="none",
        )
        plt.colorbar(scatter, ax=ax, label=score_column)
    if rectangles is not None:
        for x1, x2, y1, y2 in rectangles:
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    facecolor="dimgray",
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.3,
                )
            )
    ax.axvline(gxparm.orgx_px, color="white", linewidth=0.5, alpha=0.5)
    ax.axhline(gxparm.orgy_px, color="white", linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, gxparm.detector_nx)
    ax.set_ylim(gxparm.detector_ny, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Detector x / px")
    ax.set_ylabel("Detector y / px")
    ax.set_title(title or "Detector view")
    return ax


def plot_thickness_scan(
    thickness_table: pd.DataFrame,
    frame: int,
    reflections: Sequence[tuple[int, int, int]],
    metric: str = "intensity",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot thickness-dependent behavior for selected reflections at one frame."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    subset = thickness_table[thickness_table["frame"] == frame].copy()
    for h, k, l in reflections:
        curve = subset[(subset["h"] == h) & (subset["k"] == k) & (subset["l"] == l)].sort_values("thickness_nm")
        if curve.empty:
            continue
        ax.plot(curve["thickness_nm"], curve[metric], marker="o", label=f"({h} {k} {l})")
    ax.set_xlabel("Thickness / nm")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Thickness dependence at frame {frame + 1}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def interactive_frame_browser(*args: object, **kwargs: object) -> object:
    """Create a simple ipywidgets frame browser if ipywidgets is available."""

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise RuntimeError("ipywidgets is not installed in this environment.") from exc

    frame_summary: pd.DataFrame = kwargs.pop("frame_summary")
    frame_callback = kwargs.pop("frame_callback")
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(int(frame_summary["frame"].max()), 0),
        step=1,
        description="Frame",
        continuous_update=False,
    )

    def _on_change(change: dict[str, object]) -> None:
        if change.get("name") == "value":
            frame_callback(int(change["new"]))

    slider.observe(_on_change)
    display(slider)
    frame_callback(int(slider.value))
    return slider
