"""2D Gaussian fitting for diffraction patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class GaussianFitConfig:
    """Configuration for 2D Gaussian fitting."""

    max_center_shift_px: float = 3.0
    initial_sigma_px: float = 1.5
    min_sigma_px: float = 0.5
    max_sigma_px: float = 6.0


@dataclass(frozen=True)
class GaussianFitResult:
    """Result of a 2D Gaussian fit."""

    success: bool
    message: str
    amplitude: float
    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    theta_rad: float
    background: float
    integrated_intensity: float
    integrated_intensity_sigma: float
    rmse: float
    r_squared: float
    model_patch: np.ndarray | None


def _rotated_elliptical_gaussian(
    yx_coords: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta_rad: float,
    background: float,
) -> np.ndarray:
    y, x = yx_coords
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    x_shift = x - x0
    y_shift = y - y0
    x_rot = cos_t * x_shift + sin_t * y_shift
    y_rot = -sin_t * x_shift + cos_t * y_shift
    expo = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
    return background + amplitude * np.exp(expo)


def _finite_patch_or_raise(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Gaussian fitting expects a 2D patch.")
    if not np.isfinite(arr).any():
        raise ValueError("Patch contains no finite pixels.")
    return arr


def _border_background(patch: np.ndarray) -> float:
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    border = border[np.isfinite(border)]
    if border.size == 0:
        return float(np.nanmedian(patch))
    return float(np.nanmedian(border))


def fit_gaussian_patch(
    patch: np.ndarray,
    x_pred_local: float,
    y_pred_local: float,
    config: GaussianFitConfig | None = None,
) -> GaussianFitResult:
    """Fit a 2D elliptical Gaussian plus a constant background to a patch."""

    cfg = config or GaussianFitConfig()
    data = _finite_patch_or_raise(patch)
    if np.isnan(data).any():
        fill_value = float(np.nanmedian(data))
        data = np.nan_to_num(data, nan=fill_value)

    ny, nx = data.shape
    y_grid, x_grid = np.indices(data.shape, dtype=np.float64)
    background0 = _border_background(data)
    amplitude0 = max(float(np.nanmax(data) - background0), 1.0)
    x0 = float(np.clip(x_pred_local, 0.0, nx - 1.0))
    y0 = float(np.clip(y_pred_local, 0.0, ny - 1.0))

    p0 = np.array(
        [
            amplitude0,
            x0,
            y0,
            float(cfg.initial_sigma_px),
            float(cfg.initial_sigma_px),
            0.0,
            background0,
        ],
        dtype=np.float64,
    )
    lower = np.array(
        [
            0.0,
            x0 - cfg.max_center_shift_px,
            y0 - cfg.max_center_shift_px,
            cfg.min_sigma_px,
            cfg.min_sigma_px,
            -0.5 * np.pi,
            float(np.nanmin(data)) - abs(amplitude0),
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            float(np.nanmax(data) - np.nanmin(data) + amplitude0 + 1.0),
            x0 + cfg.max_center_shift_px,
            y0 + cfg.max_center_shift_px,
            cfg.max_sigma_px,
            cfg.max_sigma_px,
            0.5 * np.pi,
            float(np.nanmax(data) + abs(amplitude0)),
        ],
        dtype=np.float64,
    )
    lower[1] = max(lower[1], 0.0)
    lower[2] = max(lower[2], 0.0)
    upper[1] = min(upper[1], nx - 1.0)
    upper[2] = min(upper[2], ny - 1.0)

    def residuals(params: np.ndarray) -> np.ndarray:
        model = _rotated_elliptical_gaussian(
            (y_grid, x_grid),
            amplitude=params[0],
            x0=params[1],
            y0=params[2],
            sigma_x=params[3],
            sigma_y=params[4],
            theta_rad=params[5],
            background=params[6],
        )
        return (model - data).ravel()

    try:
        result = least_squares(residuals, p0, bounds=(lower, upper), method="trf")
    except Exception as exc:
        return GaussianFitResult(
            success=False,
            message=f"Optimization failed: {exc}",
            amplitude=float("nan"),
            x0=float("nan"),
            y0=float("nan"),
            sigma_x=float("nan"),
            sigma_y=float("nan"),
            theta_rad=float("nan"),
            background=float("nan"),
            integrated_intensity=float("nan"),
            integrated_intensity_sigma=float("nan"),
            rmse=float("nan"),
            r_squared=float("nan"),
            model_patch=None,
        )

    amp, fit_x0, fit_y0, sigma_x, sigma_y, theta, background = result.x
    model_patch = _rotated_elliptical_gaussian(
        (y_grid, x_grid),
        amplitude=amp,
        x0=fit_x0,
        y0=fit_y0,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta_rad=theta,
        background=background,
    )
    residual = model_patch - data
    rss = float(np.sum(residual ** 2))
    tss = float(np.sum((data - float(np.mean(data))) ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    r_squared = 1.0 - rss / tss if tss > 0 else 1.0
    integrated_intensity = float(amp * 2.0 * np.pi * sigma_x * sigma_y)

    intensity_sigma = float("nan")
    dof = data.size - result.x.size
    if dof > 0 and result.jac.size:
        try:
            jac = result.jac
            cov = np.linalg.inv(jac.T @ jac) * (rss / dof)
            grad = np.zeros(result.x.size, dtype=np.float64)
            grad[0] = 2.0 * np.pi * sigma_x * sigma_y
            grad[3] = 2.0 * np.pi * amp * sigma_y
            grad[4] = 2.0 * np.pi * amp * sigma_x
            variance = float(grad @ cov @ grad)
            if variance >= 0.0:
                intensity_sigma = float(np.sqrt(variance))
        except np.linalg.LinAlgError:
            intensity_sigma = float("nan")

    return GaussianFitResult(
        success=bool(result.success),
        message=str(result.message),
        amplitude=float(amp),
        x0=float(fit_x0),
        y0=float(fit_y0),
        sigma_x=float(sigma_x),
        sigma_y=float(sigma_y),
        theta_rad=float(theta),
        background=float(background),
        integrated_intensity=integrated_intensity,
        integrated_intensity_sigma=float(intensity_sigma),
        rmse=rmse,
        r_squared=float(r_squared),
        model_patch=np.asarray(model_patch, dtype=np.float64),
    )
