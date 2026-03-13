from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class WeightingConfig:
    """Parameters controlling score-to-weight conversion."""

    alpha: float = 0.5
    filter_threshold: float | None = None
    min_sigma: float = 1e-6


def baseline_weight_from_sigma(sigma: pd.Series | np.ndarray, min_sigma: float = 1e-6) -> np.ndarray:
    """Convert experimental sigmas into inverse-variance weights."""
    sigma_arr = np.maximum(np.asarray(sigma, dtype=np.float64), float(min_sigma))
    return 1.0 / np.square(sigma_arr)


def apply_sigma_inflation(sigma: pd.Series | np.ndarray, score: pd.Series | np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply multiplicative sigma inflation using `sigma_new = sigma * (1 + alpha * S)`."""
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    score_arr = np.asarray(score, dtype=np.float64)
    return sigma_arr * (1.0 + float(alpha) * score_arr)


def apply_weight_downscaling(
    sigma: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    alpha: float = 0.5,
    min_sigma: float = 1e-6,
) -> np.ndarray:
    """Apply weight down-scaling using `w_new = w / (1 + alpha * S)`."""
    base_weight = baseline_weight_from_sigma(sigma, min_sigma=min_sigma)
    score_arr = np.asarray(score, dtype=np.float64)
    return base_weight / (1.0 + float(alpha) * score_arr)


def apply_filter(score: pd.Series | np.ndarray, threshold: float | None) -> np.ndarray:
    """Return a boolean mask marking reflections retained after score filtering."""
    if threshold is None:
        return np.ones(len(np.asarray(score)), dtype=bool)
    return np.asarray(score, dtype=np.float64) <= float(threshold)


def apply_orientation_aware_weighting(
    reflection_table: pd.DataFrame,
    config: WeightingConfig | None = None,
) -> pd.DataFrame:
    """Add `sigma_new`, `weight_new`, and `keep` columns to a reflection table."""
    cfg = config or WeightingConfig()
    if "S" not in reflection_table.columns:
        raise ValueError("Reflection table must contain an 'S' column before weighting.")
    if "sigma" not in reflection_table.columns:
        raise ValueError("Reflection table must contain a 'sigma' column before weighting.")

    output = reflection_table.copy()
    sigma_base = np.maximum(output["sigma"].to_numpy(dtype=np.float64), cfg.min_sigma)
    score = output["S"].to_numpy(dtype=np.float64)
    output["weight_base"] = baseline_weight_from_sigma(sigma_base, min_sigma=cfg.min_sigma)
    output["sigma_new"] = apply_sigma_inflation(sigma_base, score, alpha=cfg.alpha)
    output["weight_new"] = apply_weight_downscaling(sigma_base, score, alpha=cfg.alpha, min_sigma=cfg.min_sigma)
    output["keep"] = apply_filter(score, cfg.filter_threshold)
    return output
