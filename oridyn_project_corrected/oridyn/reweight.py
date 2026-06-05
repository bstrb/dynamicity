"""Post-hoc recombination of normalized OriDyn score terms.

This module deliberately does not recompute geometry. It takes an existing
``reflection_scores.csv`` or ``hkl_frame_trajectories.csv`` table and rebuilds
``S_dyn_geom``-style columns from already-normalized terms. That makes it easy
to ask what happens when self-risk, graph crowding, Laue-zone risk, systematic
rows, or frame-axis risk are emphasized differently.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

from .config import ScoreWeights

TERM_COLUMNS = {
    "self": "self_risk_norm",
    "graph": "graph_crowding_norm",
    "zone": "same_laue_zone_crowding_norm",
    "row": "systematic_row_risk_norm",
    "frame": "frame_axis_risk_norm",
}

DEFAULT_WEIGHT_DICT = ScoreWeights().to_dict()


def load_weight_presets(path: str | Path | None) -> dict[str, dict[str, float]]:
    """Load one or more weight presets from JSON.

    Accepted JSON formats are either a flat mapping::

        {"self": 1, "graph": 2, "zone": 0, "row": 0, "frame": 0, "interaction": 0}

    or a nested mapping::

        {"self_only": {"self": 1, ...}, "graph_heavy": {"graph": 2, ...}}
    """

    if path is None:
        return {"cli": dict(DEFAULT_WEIGHT_DICT)}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Weights JSON must contain a mapping.")
    if any(key in payload for key in DEFAULT_WEIGHT_DICT):
        return {"custom": _clean_weight_dict(payload)}
    presets: dict[str, dict[str, float]] = {}
    for name, values in payload.items():
        if not isinstance(values, dict):
            raise ValueError(f"Preset {name!r} must contain a mapping of weights.")
        presets[str(name)] = _clean_weight_dict(values)
    if not presets:
        raise ValueError("No weight presets found.")
    return presets


def cli_weight_preset(
    w_self: float,
    w_graph: float,
    w_zone: float,
    w_row: float,
    w_frame: float,
    w_interaction: float,
    name: str = "cli",
) -> dict[str, dict[str, float]]:
    """Build a single preset from command-line weight values."""

    return {
        str(name): {
            "self": float(w_self),
            "graph": float(w_graph),
            "zone": float(w_zone),
            "row": float(w_row),
            "frame": float(w_frame),
            "interaction": float(w_interaction),
        }
    }


def reweight_scores(
    scores: pd.DataFrame,
    presets: Mapping[str, Mapping[str, float]],
    alpha: float = 1.0,
    overwrite_single: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``scores`` with new score/sigma columns for presets.

    Parameters
    ----------
    scores:
        Existing OriDyn score table containing normalized term columns.
    presets:
        Mapping from preset name to weights.
    alpha:
        Sigma-inflation strength used in ``sigma_dyn_rel = exp(0.5*alpha*S)``.
    overwrite_single:
        If exactly one preset is supplied, also overwrite canonical
        ``S_dyn_geom`` and ``sigma_dyn_rel`` for convenience.
    """

    out = scores.copy()
    if out.empty:
        return out
    for column in TERM_COLUMNS.values():
        if column not in out:
            out[column] = 0.0
    for name, raw_weights in presets.items():
        weights = _clean_weight_dict(raw_weights)
        suffix = _safe_suffix(name)
        score_col = f"S_dyn_geom_{suffix}"
        sigma_col = f"sigma_dyn_rel_{suffix}"
        self_component = weights["self"] * out[TERM_COLUMNS["self"]]
        graph_component = weights["graph"] * out[TERM_COLUMNS["graph"]]
        zone_component = weights["zone"] * out[TERM_COLUMNS["zone"]]
        row_component = weights["row"] * out[TERM_COLUMNS["row"]]
        frame_component = weights["frame"] * out[TERM_COLUMNS["frame"]]
        interaction_component = weights["interaction"] * out[TERM_COLUMNS["self"]] * out[TERM_COLUMNS["graph"]]
        out[f"S_self_component_{suffix}"] = self_component
        out[f"S_graph_component_{suffix}"] = graph_component
        out[f"S_zone_component_{suffix}"] = zone_component
        out[f"S_row_component_{suffix}"] = row_component
        out[f"S_frame_component_{suffix}"] = frame_component
        out[f"S_interaction_component_{suffix}"] = interaction_component
        out[score_col] = (
            self_component + graph_component + zone_component + row_component + frame_component + interaction_component
        )
        out[sigma_col] = np.exp(0.5 * float(alpha) * out[score_col].to_numpy(dtype=float))
        if "sigma" in out:
            sigma = pd.to_numeric(out["sigma"], errors="coerce")
            out[f"sigma_dyn_{suffix}"] = sigma * out[sigma_col]
            sigma_dyn = out[f"sigma_dyn_{suffix}"].to_numpy(dtype=float)
            out[f"weight_dyn_{suffix}"] = np.divide(
                1.0,
                sigma_dyn**2,
                out=np.full(len(out), np.nan, dtype=float),
                where=sigma_dyn > 0.0,
            )
    if overwrite_single and len(presets) == 1:
        suffix = _safe_suffix(next(iter(presets)))
        out["S_dyn_geom"] = out[f"S_dyn_geom_{suffix}"]
        out["sigma_dyn_rel"] = out[f"sigma_dyn_rel_{suffix}"]
        if f"sigma_dyn_{suffix}" in out:
            out["sigma_dyn"] = out[f"sigma_dyn_{suffix}"]
            out["weight_dyn"] = out[f"weight_dyn_{suffix}"]
    return out


def reweight_scores_file(
    scores_path: str | Path,
    output_path: str | Path,
    presets: Mapping[str, Mapping[str, float]],
    alpha: float = 1.0,
    overwrite_single: bool = True,
) -> pd.DataFrame:
    """Read a score table, reweight it, write it, and return it."""

    table = pd.read_csv(scores_path, dtype={"hkl_label": "string"})
    out = reweight_scores(table, presets, alpha=alpha, overwrite_single=overwrite_single)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def _clean_weight_dict(values: Mapping[str, object]) -> dict[str, float]:
    cleaned = dict(DEFAULT_WEIGHT_DICT)
    for key in cleaned:
        if key in values:
            cleaned[key] = float(values[key])
    return cleaned


def _safe_suffix(name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    suffix = suffix.strip("_") or "preset"
    if suffix[0].isdigit():
        suffix = f"preset_{suffix}"
    return suffix
