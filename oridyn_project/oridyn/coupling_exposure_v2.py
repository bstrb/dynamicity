"""Observation-level many-beam coupling-exposure v2 score helpers.

This module is deliberately separate from the original OriDyn graph-crowding
pipeline so successful v1 diagnostic results remain reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["source_filename", "event", "h", "k", "l"]
HKL_COLUMNS = ["h", "k", "l"]

REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "frame",
    "q_invA",
    "sg",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

OUTPUT_SCORE_COLUMNS = [
    "manybeam_coupling_v2_core_raw",
    "trust_risk_v2_core_norm",
    "manybeam_coupling_v2_core_plus_zone_raw",
    "trust_risk_v2_core_plus_zone_norm",
    "manybeam_coupling_v2_core_plus_row_raw",
    "trust_risk_v2_core_plus_row_norm",
    "manybeam_coupling_v2_full_raw",
    "trust_risk_v2_full_norm",
]

VARIANT_RAW_COLUMNS = {
    "core": "manybeam_coupling_v2_core_raw",
    "core_plus_zone": "manybeam_coupling_v2_core_plus_zone_raw",
    "core_plus_row": "manybeam_coupling_v2_core_plus_row_raw",
    "full": "manybeam_coupling_v2_full_raw",
}

VARIANT_NORM_COLUMNS = {
    "core": "trust_risk_v2_core_norm",
    "core_plus_zone": "trust_risk_v2_core_plus_zone_norm",
    "core_plus_row": "trust_risk_v2_core_plus_row_norm",
    "full": "trust_risk_v2_full_norm",
}

AXIS_RE = re.compile(r"[-+]?\d+")


@dataclass(frozen=True)
class CouplingV2Params:
    """Parameters for the v2 coupling-exposure proxy."""

    sg0: float = 0.01
    g0_invA: float = 0.40
    hkl_delta_g0: float = 1.5
    low_order_power: float = 1.5
    beta_zone: float = 0.5
    beta_row: float = 0.5
    beta_frame: float = 0.25
    max_edges_per_reflection: int = 64
    target_batch_size: int = 256

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def missing_required_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return required v2 input columns missing from a score table."""

    have = set(str(column) for column in columns)
    return [column for column in REQUIRED_COLUMNS if column not in have]


def robust_p01_p99_normalize(values: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Normalize finite values to p01-p99, clipped to [0, 1]."""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=values.index, dtype=float), {
            "method": "no_finite_values",
            "finite_count": 0,
            "p01": None,
            "p99": None,
        }
    p01 = float(finite.quantile(0.01))
    p99 = float(finite.quantile(0.99))
    if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
        normalized = pd.Series(0.0, index=values.index, dtype=float)
        normalized.loc[numeric.isna()] = np.nan
        method = "degenerate_p01_p99_to_zero"
    else:
        normalized = ((numeric - p01) / (p99 - p01)).clip(lower=0.0, upper=1.0)
        method = "robust_p01_p99"
    return normalized.astype(float), {
        "method": method,
        "finite_count": int(len(finite)),
        "p01": p01,
        "p99": p99,
        "min": float(finite.min()),
        "median": float(finite.median()),
        "max": float(finite.max()),
    }


def excitation_weight_from_sg(sg: np.ndarray, sg0: float) -> np.ndarray:
    """Soft excitation source weight from the existing OriDyn sg column."""

    scale = max(float(sg0), 1e-12)
    values = np.asarray(sg, dtype=float)
    weights = np.exp(-((np.abs(values) / scale) ** 2))
    return np.where(np.isfinite(weights), weights, 0.0)


def score_frame_coupling_v2(args: tuple[int, pd.DataFrame, CouplingV2Params]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score all observations in one frame."""

    frame, group, params = args
    work = group.reset_index(drop=True).copy()
    n = len(work)
    out = work[
        [column for column in ["source_filename", "event", "frame", "h", "k", "l", "frame_axis_risk_norm"] if column in work]
    ].copy()
    for column in VARIANT_RAW_COLUMNS.values():
        out[column] = 0.0

    stats: dict[str, Any] = {
        "frame": int(frame),
        "n_observations": int(n),
        "reciprocal_metric_method": "empty_frame",
        "reciprocal_metric_rank": 0,
        "reciprocal_metric_projected_to_psd": False,
    }
    if n <= 1:
        return out, stats

    hkls = work[HKL_COLUMNS].to_numpy(dtype=np.int64)
    q = pd.to_numeric(work["q_invA"], errors="coerce").to_numpy(dtype=float)
    sg = pd.to_numeric(work["sg"], errors="coerce").to_numpy(dtype=float)
    source_excitation = excitation_weight_from_sg(sg, params.sg0)
    row_risk = pd.to_numeric(work["systematic_row_risk_norm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    row_boost = 1.0 + float(params.beta_row) * np.clip(row_risk, 0.0, None)

    metric, metric_stats = estimate_reciprocal_metric(hkls, q)
    stats.update(metric_stats)
    axis = _axis_by_row(work)
    laue_n = hkls @ axis if axis is not None else None
    if axis is None:
        stats["laue_zone_method"] = "no_assigned_risky_axis_column_or_parse_failed"
    else:
        stats["laue_zone_method"] = "assigned_risky_axis_dot_hkl"
        stats["assigned_axis_u"] = int(axis[0])
        stats["assigned_axis_v"] = int(axis[1])
        stats["assigned_axis_w"] = int(axis[2])

    core_raw = np.zeros(n, dtype=float)
    zone_raw = np.zeros(n, dtype=float)
    row_raw = np.zeros(n, dtype=float)
    full_raw = np.zeros(n, dtype=float)
    batch_size = max(int(params.target_batch_size), 1)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        target_hkl = hkls[start:stop]
        delta = hkls[None, :, :] - target_hkl[:, None, :]
        nonself = np.any(delta != 0, axis=2)
        coupling_prior = _coupling_prior(delta, metric, params)
        edge_core = coupling_prior * source_excitation[None, :]
        edge_core = np.where(nonself & np.isfinite(edge_core), edge_core, 0.0)

        core_sum = _sum_top_edges(edge_core, int(params.max_edges_per_reflection))
        core_raw[start:stop] = np.log1p(core_sum)

        if laue_n is None:
            zone_edge = edge_core
        else:
            same_zone = laue_n[None, :] == laue_n[start:stop, None]
            zone_edge = edge_core * (1.0 + float(params.beta_zone) * same_zone.astype(float))
        zone_sum = _sum_top_edges(zone_edge, int(params.max_edges_per_reflection))
        zone_raw[start:stop] = np.log1p(zone_sum)

        row_sum = core_sum * row_boost[start:stop]
        row_raw[start:stop] = np.log1p(row_sum)

        full_sum = zone_sum * row_boost[start:stop]
        full_raw[start:stop] = np.log1p(full_sum)

    out["manybeam_coupling_v2_core_raw"] = core_raw
    out["manybeam_coupling_v2_core_plus_zone_raw"] = zone_raw
    out["manybeam_coupling_v2_core_plus_row_raw"] = row_raw
    out["manybeam_coupling_v2_full_raw"] = full_raw
    return out, stats


def add_v2_normalized_columns(scores: pd.DataFrame, params: CouplingV2Params) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add trust-risk normalized columns from v2 raw score columns."""

    out = scores.copy()
    metadata: dict[str, Any] = {}
    for variant, raw_column in VARIANT_RAW_COLUMNS.items():
        norm_column = VARIANT_NORM_COLUMNS[variant]
        trust_raw = pd.to_numeric(out[raw_column], errors="coerce")
        if variant == "full":
            if "frame_axis_risk_norm" in out:
                frame_risk = pd.to_numeric(out["frame_axis_risk_norm"], errors="coerce").fillna(0.0)
            else:
                frame_risk = pd.Series(0.0, index=out.index, dtype=float)
            trust_raw = trust_raw * (1.0 + float(params.beta_frame) * np.clip(frame_risk, 0.0, None))
        out[norm_column], norm_meta = robust_p01_p99_normalize(trust_raw)
        norm_meta["raw_column"] = raw_column
        norm_meta["normalized_column"] = norm_column
        norm_meta["frame_axis_boost_applied"] = bool(variant == "full")
        metadata[variant] = norm_meta
    return out, metadata


def estimate_reciprocal_metric(hkls: np.ndarray, q_invA: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate a reciprocal metric tensor from q^2 = hkl^T M hkl."""

    hkl = np.asarray(hkls, dtype=float)
    q = np.asarray(q_invA, dtype=float)
    finite = np.isfinite(q) & np.all(np.isfinite(hkl), axis=1) & (q > 0.0)
    stats: dict[str, Any] = {
        "reciprocal_metric_method": "hkl_delta_fallback",
        "reciprocal_metric_rank": 0,
        "reciprocal_metric_projected_to_psd": False,
        "reciprocal_metric_fit_points": int(np.sum(finite)),
    }
    if int(np.sum(finite)) < 6:
        stats["reciprocal_metric_reason"] = "fewer_than_6_finite_q_points"
        return None, stats

    x = hkl[finite]
    design = np.column_stack(
        [
            x[:, 0] ** 2,
            x[:, 1] ** 2,
            x[:, 2] ** 2,
            2.0 * x[:, 0] * x[:, 1],
            2.0 * x[:, 0] * x[:, 2],
            2.0 * x[:, 1] * x[:, 2],
        ]
    )
    y = q[finite] ** 2
    try:
        coeff, _residuals, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        stats["reciprocal_metric_reason"] = "lstsq_failed"
        return None, stats

    stats["reciprocal_metric_rank"] = int(rank)
    if int(rank) < 6:
        stats["reciprocal_metric_reason"] = "rank_deficient_metric_fit"
        return None, stats

    metric = np.asarray(
        [
            [coeff[0], coeff[3], coeff[4]],
            [coeff[3], coeff[1], coeff[5]],
            [coeff[4], coeff[5], coeff[2]],
        ],
        dtype=float,
    )
    metric = 0.5 * (metric + metric.T)
    try:
        eigval, eigvec = np.linalg.eigh(metric)
    except np.linalg.LinAlgError:
        stats["reciprocal_metric_reason"] = "eigendecomposition_failed"
        return None, stats
    if not np.all(np.isfinite(eigval)):
        stats["reciprocal_metric_reason"] = "nonfinite_metric_eigenvalues"
        return None, stats
    if float(np.min(eigval)) < 0.0:
        eigval = np.clip(eigval, 1e-12, None)
        metric = (eigvec * eigval[None, :]) @ eigvec.T
        stats["reciprocal_metric_projected_to_psd"] = True
    stats["reciprocal_metric_method"] = (
        "reciprocal_metric_fit_psd_projected"
        if stats["reciprocal_metric_projected_to_psd"]
        else "reciprocal_metric_fit"
    )
    stats["reciprocal_metric_reason"] = ""
    return metric, stats


def _coupling_prior(delta_hkl: np.ndarray, metric: np.ndarray | None, params: CouplingV2Params) -> np.ndarray:
    if metric is None:
        length = np.linalg.norm(delta_hkl.astype(float), axis=2)
        scale = max(float(params.hkl_delta_g0), 1e-12)
    else:
        squared = np.einsum("...i,ij,...j->...", delta_hkl.astype(float), metric, delta_hkl.astype(float))
        length = np.sqrt(np.clip(squared, 0.0, None))
        scale = max(float(params.g0_invA), 1e-12)
    return 1.0 / (1.0 + (length / scale) ** float(params.low_order_power))


def _sum_top_edges(edge_weights: np.ndarray, max_edges: int) -> np.ndarray:
    if edge_weights.size == 0:
        return np.zeros(edge_weights.shape[0], dtype=float)
    if int(max_edges) <= 0 or int(max_edges) >= edge_weights.shape[1]:
        return np.sum(edge_weights, axis=1)
    k = int(max_edges)
    partitioned = np.partition(edge_weights, edge_weights.shape[1] - k, axis=1)[:, -k:]
    return np.sum(partitioned, axis=1)


def _axis_by_row(group: pd.DataFrame) -> np.ndarray | None:
    if "assigned_risky_axis" not in group:
        return None
    labels = group["assigned_risky_axis"].dropna()
    if labels.empty:
        return None
    label = str(labels.iloc[0])
    values = [int(match.group(0)) for match in AXIS_RE.finditer(label)]
    if len(values) != 3:
        return None
    axis = np.asarray(values, dtype=np.int64)
    if not np.any(axis):
        return None
    return axis
