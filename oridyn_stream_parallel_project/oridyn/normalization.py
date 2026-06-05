"""Resolution-shell and robust normalization helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import OridynConfig


REFLECTION_RAW_TO_NORM = {
    "self_risk_raw": "self_risk_norm",
    "graph_crowding_raw": "graph_crowding_norm",
    "same_laue_zone_crowding_raw": "same_laue_zone_crowding_norm",
    "systematic_row_risk_raw": "systematic_row_risk_norm",
}


def assign_resolution_shells(scores: pd.DataFrame, n_shells: int) -> pd.Series:
    """Assign shell IDs by reciprocal length quantiles."""

    if scores.empty or "q_invA" not in scores or n_shells <= 1:
        return pd.Series(np.zeros(len(scores), dtype=int), index=scores.index)
    unique = int(scores["q_invA"].nunique(dropna=True))
    q = max(1, min(int(n_shells), unique))
    if q <= 1:
        return pd.Series(np.zeros(len(scores), dtype=int), index=scores.index)
    shells = pd.qcut(scores["q_invA"], q=q, labels=False, duplicates="drop")
    return shells.fillna(0).astype(int)


def normalize_reflection_terms(scores: pd.DataFrame, config: OridynConfig) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Normalize raw reflection terms within resolution shells."""

    out = scores.copy()
    if out.empty:
        return out, []
    out["resolution_shell"] = assign_resolution_shells(out, config.resolution_shells)
    metadata: list[dict[str, object]] = []
    for raw_col, norm_col in REFLECTION_RAW_TO_NORM.items():
        if raw_col not in out:
            out[norm_col] = 0.0
            continue
        out[norm_col] = 0.0
        for shell, group in out.groupby("resolution_shell", sort=True):
            normalized, params = _normalize_values(group[raw_col].to_numpy(dtype=float), config)
            out.loc[group.index, norm_col] = normalized
            params.update({"column": raw_col, "normalized_column": norm_col, "resolution_shell": int(shell)})
            metadata.append(params)
    return out, metadata


def normalize_frame_terms(frame_scores: pd.DataFrame, config: OridynConfig) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Normalize frame-level axis risk across frames."""

    out = frame_scores.copy()
    if out.empty or "frame_axis_risk_raw" not in out:
        out["frame_axis_risk_norm"] = 0.0
        return out, []
    norm_config = config
    if config.frame_normalization is not None:
        norm_config = replace(config, normalization=config.frame_normalization)
    normalized, params = _normalize_values(out["frame_axis_risk_raw"].to_numpy(dtype=float), norm_config)
    out["frame_axis_risk_norm"] = normalized
    params.update({"column": "frame_axis_risk_raw", "normalized_column": "frame_axis_risk_norm", "scope": "frame"})
    if config.frame_normalization is not None:
        params["frame_normalization_override"] = config.frame_normalization
    return out, [params]


def _normalize_values(values: np.ndarray, config: OridynConfig) -> tuple[np.ndarray, dict[str, object]]:
    if config.normalization == "none":
        return values.astype(float), {"method": "none"}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float), {"method": config.normalization, "empty": True}
    clip = float(config.normalization_clip)
    if config.normalization == "percentile":
        p05 = float(np.percentile(finite, 5))
        p95 = float(np.percentile(finite, 95))
        scale = max(p95 - p05, 1e-12)
        normalized = (values - p05) / scale
        return np.clip(normalized, 0.0, clip), {"method": "percentile", "p05": p05, "p95": p95, "clip": clip}
    if config.normalization == "rank":
        normalized = np.zeros_like(values, dtype=float)
        finite_mask = np.isfinite(values)
        if finite.size == 1:
            return normalized, {"method": "rank", "n": 1, "clip": clip}
        ranks = pd.Series(finite).rank(method="average").to_numpy(dtype=float)
        normalized[finite_mask] = (ranks - 1.0) / max(float(finite.size - 1), 1.0) * clip
        return normalized, {"method": "rank", "n": int(finite.size), "clip": clip}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, 1e-12)
    normalized = (values - median) / scale
    return np.clip(normalized, 0.0, clip), {"method": "median_mad", "median": median, "mad": mad, "clip": clip}
