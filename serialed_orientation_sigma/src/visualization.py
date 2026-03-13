from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_orientation_trajectory(frame_summary: pd.DataFrame, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot beam direction in crystal coordinates across frames."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    scatter = ax.scatter(frame_summary["beam_u"], frame_summary["beam_v"], c=frame_summary["frame"], s=20)
    ax.set_xlabel("beam_u")
    ax.set_ylabel("beam_v")
    ax.set_title("Orientation trajectory in crystal coordinates")
    fig.colorbar(scatter, ax=ax, label="frame")
    return fig, ax


def plot_zone_axis_distance(frame_summary: pd.DataFrame, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot nearest-zone-axis distance versus frame."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    ax.plot(frame_summary["frame"], frame_summary["zone_axis_angle"], linewidth=1.5)
    ax.set_xlabel("frame")
    ax.set_ylabel("zone-axis angle (deg)")
    ax.set_title("Nearest zone-axis distance across frames")
    return fig, ax


def plot_dynamical_score_distribution(
    reflection_table: pd.DataFrame,
    score_column: str = "S",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the distribution of reflection-wise dynamical scores."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.hist(reflection_table[score_column], bins=50)
    ax.set_xlabel(score_column)
    ax.set_ylabel("count")
    ax.set_title("Dynamical score distribution")
    return fig, ax


def plot_reflection_score_histogram(
    reflection_table: pd.DataFrame,
    score_column: str = "S",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Alias for a score histogram used in notebook workflows."""
    return plot_dynamical_score_distribution(reflection_table, score_column=score_column, ax=ax)


def plot_detector_map(
    reflection_table: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    c_col: str = "S",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot reflections on detector coordinates colored by score."""
    if x_col not in reflection_table.columns or y_col not in reflection_table.columns:
        raise ValueError("Detector map requires x/y columns in the reflection table.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    scatter = ax.scatter(reflection_table[x_col], reflection_table[y_col], c=reflection_table[c_col], s=12)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Detector map colored by {c_col}")
    fig.colorbar(scatter, ax=ax, label=c_col)
    return fig, ax


def plot_thickness_sensitivity(scan: pd.DataFrame, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot intensity versus thickness for a simulated validation scan."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    ax.plot(scan["thickness_nm"], scan["intensity"], linewidth=1.8)
    ax.set_xlabel("thickness (nm)")
    ax.set_ylabel("simulated intensity")
    h, k, l = int(scan.iloc[0]["h"]), int(scan.iloc[0]["k"]), int(scan.iloc[0]["l"])
    ax.set_title(f"Thickness sensitivity for ({h}, {k}, {l})")
    return fig, ax
