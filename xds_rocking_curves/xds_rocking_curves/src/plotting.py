"""Plotting helpers for rocking-curve analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rocking_curve(
    curve: pd.DataFrame,
    output_path: str | Path,
    normalized: bool = False,
) -> Path:
    """Plot fitted intensity versus frame."""

    y_col = "I_fit_norm" if normalized and "I_fit_norm" in curve.columns else "I_fit"
    fig, ax = plt.subplots(figsize=(8, 4))
    good = curve[curve["fit_success"]].sort_values("frame")
    bad = curve[~curve["fit_success"]].sort_values("frame")
    if not good.empty:
        ax.plot(good["frame"], good[y_col], marker="o", linewidth=1.6, label="successful fit")
        if "sigma_fit" in good.columns and y_col == "I_fit":
            sigma = good["sigma_fit"].to_numpy(dtype=float)
            values = good[y_col].to_numpy(dtype=float)
            if np.isfinite(sigma).any():
                ax.fill_between(good["frame"], values - sigma, values + sigma, alpha=0.15)
    if not bad.empty:
        ax.scatter(bad["frame"], np.zeros(len(bad)), marker="x", label="failed fit")
    title = "Local rocking curve"
    if {"h", "k", "l"}.issubset(curve.columns) and not curve.empty:
        row = curve.iloc[0]
        title += f" for ({int(row['h'])}, {int(row['k'])}, {int(row['l'])})"
    ax.set_xlabel("frame")
    ax.set_ylabel("normalized I_fit" if normalized else "I_fit")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_detector_track(
    predictions: pd.DataFrame,
    curve: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot predicted detector coordinates and fitted centers."""

    fig, ax = plt.subplots(figsize=(6, 6))
    usable = predictions[predictions["on_detector"] & predictions["valid"]]
    ax.plot(usable["x_pred"], usable["y_pred"], linewidth=1.2, label="predicted track")
    relevant = predictions[predictions["is_relevant"] & predictions["on_detector"] & predictions["valid"]]
    if not relevant.empty:
        ax.scatter(relevant["x_pred"], relevant["y_pred"], s=30, label="relevant frames")
    successful = curve[curve["fit_success"]]
    if not successful.empty:
        ax.scatter(successful["x_fit"], successful["y_fit"], s=20, label="fitted centers")
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title("Predicted detector trajectory")
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
