"""Wilson-like scaling utilities.

The calibration here intentionally mirrors the simple logic from the browser
prototype. Equivalent reflections are merged using a crude symmetry key based on
sorted absolute Miller indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SymmetryKey = tuple[int, int, int]


@dataclass(frozen=True)
class WilsonCalibration:
    """Output from the Wilson-like calibration step."""

    scale_factor: float
    median_amplitude: float
    amplitudes_by_key: dict[SymmetryKey, float]
    merged_table: pd.DataFrame

    def lookup_amplitude(self, h: int, k: int, l: int) -> float:
        """Return ``|F_g|`` for a reflection or the fallback median amplitude."""

        return self.amplitudes_by_key.get(symmetry_key(h, k, l), self.median_amplitude)


def symmetry_key(h: int, k: int, l: int) -> SymmetryKey:
    """Return the simple absolute-value symmetry key used by the HTML prototype."""

    return tuple(sorted((abs(h), abs(k), abs(l)), reverse=True))


def merge_equivalent_reflections(observations: pd.DataFrame) -> pd.DataFrame:
    """Merge reflections using inverse-variance weighting."""

    if observations.empty:
        raise ValueError("Observation table is empty.")

    table = observations.copy()
    table["symmetry_key"] = [symmetry_key(int(h), int(k), int(l)) for h, k, l in table[["h", "k", "l"]].itertuples(index=False)]
    rows: list[dict[str, float | int | SymmetryKey]] = []
    for key, group in table.groupby("symmetry_key", sort=False):
        valid = group[(group["I"] > 0.0) & (group["sigma"] > 0.0)].copy()
        if valid.empty:
            merged_i = 0.1
            n_weighted = 0
        else:
            weights = 1.0 / np.square(valid["sigma"].to_numpy(dtype=float))
            intensities = valid["I"].to_numpy(dtype=float)
            merged_i = float(np.sum(weights * intensities) / np.sum(weights))
            n_weighted = int(valid.shape[0])
        rows.append(
            {
                "symmetry_key": key,
                "merged_I": merged_i,
                "n_observations": int(group.shape[0]),
                "n_weighted": n_weighted,
            }
        )
    return pd.DataFrame.from_records(rows)


def wilson_calibrate(observations: pd.DataFrame, sum_fj2: float) -> WilsonCalibration:
    """Perform the Wilson-like calibration used in the HTML analyser.

    Parameters
    ----------
    observations:
        Reflection observations from ``INTEGRATE.HKL``.
    sum_fj2:
        Composition-derived ``sum_j n_j f_j(0)^2``.
    """

    if sum_fj2 <= 0.0:
        raise ValueError("sum_fj2 must be positive.")

    merged = merge_equivalent_reflections(observations)
    mean_intensity = float(merged["merged_I"].mean())
    scale_factor = mean_intensity / sum_fj2
    amplitudes = np.sqrt(np.maximum(merged["merged_I"].to_numpy(dtype=float) / scale_factor, 0.01))
    merged = merged.copy()
    merged["Fg_abs"] = amplitudes
    median_amplitude = float(np.median(amplitudes))
    amplitudes_by_key = {
        key: float(fg_abs)
        for key, fg_abs in zip(merged["symmetry_key"], merged["Fg_abs"], strict=True)
    }
    return WilsonCalibration(
        scale_factor=scale_factor,
        median_amplitude=median_amplitude,
        amplitudes_by_key=amplitudes_by_key,
        merged_table=merged,
    )
