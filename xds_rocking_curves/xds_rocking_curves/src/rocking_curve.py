"""Rocking-curve extraction from TIFF image stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .gaussian_fit import GaussianFitConfig, fit_gaussian_patch
from .geometry import DetectorGeometry, GXPARMData
from .image_io import ImageResolver, extract_patch, predicted_center_in_patch, read_image
from .parsers import SpotData
from .prediction import RelevanceConfig, annotate_with_spot_validation, predict_reflection_across_frames, select_relevant_frames


@dataclass(frozen=True)
class RockingCurveConfig:
    """Configuration controlling rocking-curve extraction."""

    patch_half_size: int = 7
    relevance: RelevanceConfig = field(default_factory=RelevanceConfig)
    gaussian_fit: GaussianFitConfig = field(default_factory=GaussianFitConfig)


def build_rocking_curve(
    dataset_name: str,
    thickness_nm: float | None,
    gxparm: GXPARMData,
    detector: DetectorGeometry,
    image_resolver: ImageResolver,
    hkl: tuple[int, int, int],
    config: RockingCurveConfig | None = None,
    rotation_sign: float = 1.0,
    spot_data: SpotData | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict, select, and fit one reflection across a dataset.

    Returns
    -------
    curve_table, prediction_table
        The per-frame fitted curve table and the full prediction table.
    """

    cfg = config or RockingCurveConfig()
    frames = image_resolver.available_frames
    predictions = predict_reflection_across_frames(
        gxparm=gxparm,
        detector=detector,
        hkl=hkl,
        frames=frames,
        rotation_sign=rotation_sign,
    )
    predictions = select_relevant_frames(predictions, cfg.relevance)
    predictions = annotate_with_spot_validation(predictions, spot_data, hkl)

    rows: list[dict[str, object]] = []
    relevant = predictions[predictions["is_relevant"]].copy()
    for pred in relevant.itertuples(index=False):
        frame = int(pred.frame)
        path = image_resolver.path_for_frame(frame)
        image = read_image(path)
        extraction = extract_patch(
            image=image,
            x_pred_px=float(pred.x_pred),
            y_pred_px=float(pred.y_pred),
            half_size=cfg.patch_half_size,
        )
        x_local, y_local = predicted_center_in_patch(float(pred.x_pred), float(pred.y_pred), extraction)
        if not np.isfinite(extraction.patch).any():
            rows.append(
                {
                    "dataset": dataset_name,
                    "thickness_nm": thickness_nm,
                    "frame": frame,
                    "phi_deg": float(pred.phi_deg),
                    "h": int(hkl[0]),
                    "k": int(hkl[1]),
                    "l": int(hkl[2]),
                    "x_pred": float(pred.x_pred),
                    "y_pred": float(pred.y_pred),
                    "x_fit": np.nan,
                    "y_fit": np.nan,
                    "I_fit": np.nan,
                    "bg": np.nan,
                    "sigma_fit": np.nan,
                    "fit_success": False,
                    "fit_message": "Patch contains no finite pixels",
                    "rmse": np.nan,
                    "r_squared": np.nan,
                    "sg": float(pred.sg),
                    "spot_x": float(pred.spot_x) if np.isfinite(pred.spot_x) else np.nan,
                    "spot_y": float(pred.spot_y) if np.isfinite(pred.spot_y) else np.nan,
                    "spot_distance_px": float(pred.spot_distance_px) if np.isfinite(pred.spot_distance_px) else np.nan,
                    "spot_indexed_match": bool(pred.spot_indexed_match),
                    "image_path": str(path),
                }
            )
            continue

        fit = fit_gaussian_patch(extraction.patch, x_local, y_local, config=cfg.gaussian_fit)
        x_fit_global = np.nan
        y_fit_global = np.nan
        if fit.success:
            x_fit_global = float(extraction.x_start_zero + fit.x0 + 1.0)
            y_fit_global = float(extraction.y_start_zero + fit.y0 + 1.0)
        rows.append(
            {
                "dataset": dataset_name,
                "thickness_nm": thickness_nm,
                "frame": frame,
                "phi_deg": float(pred.phi_deg),
                "h": int(hkl[0]),
                "k": int(hkl[1]),
                "l": int(hkl[2]),
                "x_pred": float(pred.x_pred),
                "y_pred": float(pred.y_pred),
                "x_fit": x_fit_global,
                "y_fit": y_fit_global,
                "I_fit": float(fit.integrated_intensity),
                "bg": float(fit.background),
                "sigma_fit": float(fit.integrated_intensity_sigma),
                "fit_success": bool(fit.success),
                "fit_message": fit.message,
                "rmse": float(fit.rmse),
                "r_squared": float(fit.r_squared),
                "sg": float(pred.sg),
                "spot_x": float(pred.spot_x) if np.isfinite(pred.spot_x) else np.nan,
                "spot_y": float(pred.spot_y) if np.isfinite(pred.spot_y) else np.nan,
                "spot_distance_px": float(pred.spot_distance_px) if np.isfinite(pred.spot_distance_px) else np.nan,
                "spot_indexed_match": bool(pred.spot_indexed_match),
                "image_path": str(path),
            }
        )

    curve = pd.DataFrame(rows)
    if not curve.empty:
        finite = curve["I_fit"].replace([np.inf, -np.inf], np.nan)
        if finite.notna().any():
            max_i = float(finite.max())
            if max_i > 0:
                curve["I_fit_norm"] = curve["I_fit"] / max_i
            else:
                curve["I_fit_norm"] = np.nan
        else:
            curve["I_fit_norm"] = np.nan
    return curve, predictions


def save_curve_outputs(
    curve: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save prediction and curve tables to disk."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "rocking_curve.csv"
    prediction_path = out / "prediction_table.csv"
    curve.to_csv(curve_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    return {"curve": curve_path, "predictions": prediction_path}
