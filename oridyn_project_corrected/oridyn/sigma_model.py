"""Score combination and sigma inflation conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OridynConfig


def combine_score_terms(scores: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    """Combine normalized geometry-only terms into ``S_dyn_geom``."""

    if scores.empty:
        return scores.copy()
    out = scores.copy()
    for column in (
        "self_risk_norm",
        "graph_crowding_norm",
        "same_laue_zone_crowding_norm",
        "systematic_row_risk_norm",
        "frame_axis_risk_norm",
    ):
        if column not in out:
            out[column] = 0.0
    weights = config.weights
    out["S_self_component"] = weights.self * out["self_risk_norm"]
    out["S_graph_component"] = weights.graph * out["graph_crowding_norm"]
    out["S_zone_component"] = weights.zone * out["same_laue_zone_crowding_norm"]
    out["S_row_component"] = weights.row * out["systematic_row_risk_norm"]
    out["S_frame_component"] = weights.frame * out["frame_axis_risk_norm"]
    out["S_interaction_component"] = weights.interaction * out["self_risk_norm"] * out["graph_crowding_norm"]
    out["S_dyn_geom"] = (
        out["S_self_component"]
        + out["S_graph_component"]
        + out["S_zone_component"]
        + out["S_row_component"]
        + out["S_frame_component"]
        + out["S_interaction_component"]
    )
    return out


def add_sigma_model(scores: pd.DataFrame, config: OridynConfig) -> pd.DataFrame:
    """Convert the final score to optional sigma and weight transformations."""

    if scores.empty:
        return scores.copy()
    out = scores.copy()
    out["sigma_dyn_rel"] = np.exp(0.5 * float(config.alpha) * out["S_dyn_geom"].to_numpy(dtype=float))
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
