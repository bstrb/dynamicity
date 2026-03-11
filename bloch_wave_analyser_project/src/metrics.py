"""Metric calculations for proxy and thickness-aware modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .constants import EPSILON

FloatArray = NDArray[np.float64]


def two_beam_metric(sg_invA: ArrayLike, xi_angstrom: ArrayLike) -> FloatArray:
    """Compute the proxy two-beam dynamicality metric.

    ``d_2beam = 1 / (1 + (s_g * xi_g)^2)``
    """

    sg = np.asarray(sg_invA, dtype=float)
    xi = np.asarray(xi_angstrom, dtype=float)
    return 1.0 / (1.0 + np.square(sg * xi))


def effective_coupling_multiplicity(eigenvectors: FloatArray, beam_index: int) -> float:
    """Compute the effective coupling multiplicity ``N_eff`` for one beam.

    The browser prototype defines weights as the product of beam and transmitted
    components of each eigenvector. With eigenvectors stored as columns,
    ``w_j = V[beam_index, j] * V[0, j]``.
    """

    weights = eigenvectors[beam_index, :] * eigenvectors[0, :]
    w2 = np.square(weights)
    sum_w2 = float(np.sum(w2))
    sum_w4 = float(np.sum(np.square(w2)))
    if sum_w4 <= EPSILON:
        return 1.0
    return (sum_w2 * sum_w2) / sum_w4


def combined_proxy_score(d_two_beam: ArrayLike, n_eff: ArrayLike) -> FloatArray:
    """Combine two-beam and multi-beam terms into ``S_comb``."""

    return np.asarray(d_two_beam, dtype=float) * np.asarray(n_eff, dtype=float)


def summarize_frame_proxy(frame_table: pd.DataFrame) -> dict[str, float]:
    """Summarize per-reflection proxy metrics for one frame."""

    if frame_table.empty:
        return {
            "n_excited": 0.0,
            "S_2beam": 0.0,
            "S_MB": 0.0,
            "mean_N_eff": 0.0,
            "max_N_eff": 0.0,
        }
    return {
        "n_excited": float(frame_table.shape[0]),
        "S_2beam": float(frame_table["d_2beam"].sum()),
        "S_MB": float(frame_table["S_comb"].sum()),
        "mean_N_eff": float(frame_table["N_eff"].mean()),
        "max_N_eff": float(frame_table["N_eff"].max()),
    }


def thickness_sensitivity_metrics(values: ArrayLike) -> dict[str, float]:
    """Compute simple thickness-sensitivity metrics from a value series."""

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return {
            "thickness_std": np.nan,
            "thickness_cv": np.nan,
            "thickness_max_min_ratio": np.nan,
            "thickness_normalized_range": np.nan,
        }
    mean_value = float(np.mean(data))
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    std_value = float(np.std(data))
    return {
        "thickness_std": std_value,
        "thickness_cv": std_value / (mean_value + EPSILON),
        "thickness_max_min_ratio": max_value / (min_value + EPSILON),
        "thickness_normalized_range": (max_value - min_value) / (mean_value + EPSILON),
    }


def aggregate_reflection_thickness_sensitivity(thickness_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate thickness sensitivity per reflection at fixed frame."""

    if thickness_table.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "frame_number",
                "h",
                "k",
                "l",
                "thickness_std",
                "thickness_cv",
                "thickness_max_min_ratio",
                "thickness_normalized_range",
            ]
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["frame", "frame_number", "h", "k", "l"]
    for keys, group in thickness_table.groupby(group_columns, sort=False):
        metrics = thickness_sensitivity_metrics(group["intensity"].to_numpy(dtype=float))
        row = {column: value for column, value in zip(group_columns, keys, strict=True)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def aggregate_frame_thickness_sensitivity(reflection_sensitivity_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate thickness sensitivity across reflections for each frame."""

    if reflection_sensitivity_table.empty:
        return pd.DataFrame(
            columns=[
                "frame",
                "frame_number",
                "frame_thickness_mean_std",
                "frame_thickness_mean_cv",
                "frame_thickness_mean_normalized_range",
            ]
        )

    grouped = (
        reflection_sensitivity_table.groupby(["frame", "frame_number"], sort=False)
        .agg(
            frame_thickness_mean_std=("thickness_std", "mean"),
            frame_thickness_mean_cv=("thickness_cv", "mean"),
            frame_thickness_mean_normalized_range=("thickness_normalized_range", "mean"),
        )
        .reset_index()
    )
    return grouped
