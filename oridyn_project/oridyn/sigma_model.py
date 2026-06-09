"""Score combination and sigma inflation conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OridynConfig

NORMALIZED_TERM_COLUMNS = (
    "self_risk_norm",
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
)


def combine_score_terms(scores: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    """Combine normalized geometry-only terms into ``S_dyn_geom``."""

    if scores.empty:
        return scores.copy()
    out = scores.copy()
    for column in NORMALIZED_TERM_COLUMNS:
        if column not in out:
            out[column] = 0.0
    weights = config.weights
    out["S_self_component"] = weights.self * out["self_risk_norm"]
    out["S_graph_component"] = weights.graph * out["graph_crowding_norm"]
    out["S_zone_component"] = weights.zone * out["same_laue_zone_crowding_norm"]
    out["S_row_component"] = weights.row * out["systematic_row_risk_norm"]
    out["S_frame_component"] = weights.frame * out["frame_axis_risk_norm"]
    out["S_interaction_component"] = weights.interaction * out["self_risk_norm"] * out["graph_crowding_norm"]
    weighted_sum = (
        out["S_self_component"]
        + out["S_graph_component"]
        + out["S_zone_component"]
        + out["S_row_component"]
        + out["S_frame_component"]
        + out["S_interaction_component"]
    )
    out["S_dyn_geom_weighted_sum"] = weighted_sum
    out["S_dyn_geom_weight_denominator"] = _score_weight_denominator(config)
    if config.score_rescale_by_weights and out["S_dyn_geom_weight_denominator"].iloc[0] > 0.0:
        out["S_dyn_geom"] = weighted_sum / out["S_dyn_geom_weight_denominator"]
    else:
        out["S_dyn_geom"] = weighted_sum
    return out


def add_sigma_model(scores: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    """Convert the final score to optional sigma and weight transformations."""

    if scores.empty:
        return scores.copy()
    out = scores.copy()
    sigma_rel, tail_score, threshold = score_to_sigma_rel(
        out["S_dyn_geom"].to_numpy(dtype=float),
        alpha=config.alpha,
        sigma_map=config.sigma_map,
        sigma_tail_quantile=config.sigma_tail_quantile,
    )
    out["sigma_tail_score"] = tail_score
    out["sigma_tail_threshold"] = threshold
    out["sigma_dyn_rel"] = sigma_rel
    if "sigma" in out:
        sigma = pd.to_numeric(out["sigma"], errors="coerce")
        out["sigma_dyn"] = sigma * out["sigma_dyn_rel"]
        out["weight_dyn"] = np.divide(
            1.0,
            out["sigma_dyn"] ** 2,
            out=np.full(len(out), np.nan, dtype=float),
            where=out["sigma_dyn"].to_numpy(dtype=float) > 0.0,
        )
    return out


def score_to_sigma_rel(
    scores: np.ndarray,
    alpha: float,
    sigma_map: str = "exponential",
    sigma_tail_quantile: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map combined scores to multiplicative sigma factors.

    ``sigma_tail_quantile <= 0`` means no percentile gate: use the nonnegative
    score directly. For positive quantiles, only scores above that percentile
    are inflated, and the active tail is rescaled to ``0..1``.
    """

    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        zeros = np.zeros_like(values, dtype=float)
        return np.ones_like(values, dtype=float), zeros, 0.0

    q = float(sigma_tail_quantile)
    if q < 0.0 or q >= 1.0:
        raise ValueError("sigma_tail_quantile must be in [0, 1). Use 0 for no percentile tail.")

    if q == 0.0:
        threshold = 0.0
        tail_score = np.maximum(values, 0.0)
    else:
        threshold = float(np.quantile(finite, q))
        tail = np.maximum(values - threshold, 0.0)
        scale = max(float(np.max(finite) - threshold), 1e-12)
        tail_score = tail / scale
        tail_score = np.where(np.isfinite(tail_score), tail_score, 0.0)

    strength = float(alpha) * tail_score
    if sigma_map == "linear":
        sigma_rel = 1.0 + strength
    elif sigma_map == "exponential":
        sigma_rel = np.exp(strength)
    else:
        raise ValueError(f"Unknown sigma_map: {sigma_map}")
    sigma_rel = np.where(np.isfinite(sigma_rel), sigma_rel, np.nan)
    return sigma_rel.astype(float), tail_score.astype(float), threshold


def _score_weight_denominator(config: OridynConfig) -> float:
    weights = config.weights
    values = (
        weights.self,
        weights.graph,
        weights.zone,
        weights.row,
        weights.frame,
        weights.interaction,
    )
    return float(sum(weight for weight in values if weight > 0.0))
