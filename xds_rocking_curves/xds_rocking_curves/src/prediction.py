"""Prediction helpers for reflection trajectories across frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .geometry import DetectorGeometry, predict_reflection_on_frame
from .parsers import GXPARMData, SpotData


@dataclass(frozen=True)
class RelevanceConfig:
    """Configuration for selecting relevant frames for one reflection."""

    mode: str = "sg"
    sg_threshold: float = 0.02
    window_half_width: int = 3


def predict_reflection_across_frames(
    gxparm: GXPARMData,
    detector: DetectorGeometry,
    hkl: tuple[int, int, int],
    frames: Iterable[int],
    rotation_sign: float = 1.0,
) -> pd.DataFrame:
    """Predict one reflection across many frames."""

    rows = []
    for frame in frames:
        pred = predict_reflection_on_frame(
            gxparm=gxparm,
            detector=detector,
            hkl=hkl,
            frame_number=float(frame),
            rotation_sign=rotation_sign,
        )
        rows.append(
            {
                "frame": int(frame),
                "phi_deg": pred.phi_deg,
                "x_pred": pred.x_pred,
                "y_pred": pred.y_pred,
                "sg": pred.sg,
                "abs_sg": abs(pred.sg),
                "valid": pred.valid,
                "on_detector": pred.on_detector,
            }
        )
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def select_relevant_frames(
    predictions: pd.DataFrame,
    config: RelevanceConfig | None = None,
) -> pd.DataFrame:
    """Mark relevant frames by excitation error or by a frame window around crossing."""

    cfg = config or RelevanceConfig()
    table = predictions.copy()
    table["is_relevant"] = False
    usable = table[table["valid"] & table["on_detector"]].copy()
    if usable.empty:
        return table

    if cfg.mode == "sg":
        mask = usable["abs_sg"] <= float(cfg.sg_threshold)
        table.loc[usable.index[mask], "is_relevant"] = True
        if not bool(mask.any()):
            best_idx = int(usable["abs_sg"].idxmin())
            table.loc[[best_idx], "is_relevant"] = True
        return table

    if cfg.mode == "window":
        crossing_idx = int(usable["abs_sg"].idxmin())
        crossing_frame = int(table.loc[crossing_idx, "frame"])
        lo = crossing_frame - int(cfg.window_half_width)
        hi = crossing_frame + int(cfg.window_half_width)
        mask = table["frame"].between(lo, hi) & table["valid"] & table["on_detector"]
        table.loc[mask, "is_relevant"] = True
        return table

    raise ValueError(f"Unknown relevance mode: {cfg.mode}")


def annotate_with_spot_validation(
    predictions: pd.DataFrame,
    spots: SpotData | None,
    hkl: tuple[int, int, int],
    max_frame_delta: float = 0.6,
) -> pd.DataFrame:
    """Annotate predictions with nearest SPOT.XDS positions.

    If indexed spots are present, the function first tries to find matching hkl
    values on nearby frames. Otherwise it falls back to the nearest spot in the
    frame neighborhood.
    """

    table = predictions.copy()
    table["spot_x"] = np.nan
    table["spot_y"] = np.nan
    table["spot_distance_px"] = np.nan
    table["spot_indexed_match"] = False
    if spots is None:
        return table

    spot_df = spots.spots.copy()
    indexed = spot_df[spot_df["indexed"]].copy()
    target_hkl = tuple(int(v) for v in hkl)

    for idx, row in table.iterrows():
        frame = float(row["frame"])
        candidates = indexed[
            (indexed["h"] == target_hkl[0])
            & (indexed["k"] == target_hkl[1])
            & (indexed["l"] == target_hkl[2])
            & (np.abs(indexed["z"] - frame) <= max_frame_delta)
        ]
        indexed_match = True
        if candidates.empty:
            indexed_match = False
            candidates = spot_df[np.abs(spot_df["z"] - frame) <= max_frame_delta]
        if candidates.empty:
            continue
        dx = candidates["x"].to_numpy(dtype=float) - float(row["x_pred"])
        dy = candidates["y"].to_numpy(dtype=float) - float(row["y_pred"])
        dist = np.hypot(dx, dy)
        best = candidates.iloc[int(np.argmin(dist))]
        table.at[idx, "spot_x"] = float(best["x"])
        table.at[idx, "spot_y"] = float(best["y"])
        table.at[idx, "spot_distance_px"] = float(np.min(dist))
        table.at[idx, "spot_indexed_match"] = indexed_match
    return table
