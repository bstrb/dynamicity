from __future__ import annotations

import numpy as np

from src.gaussian_fit import fit_gaussian_patch


def test_gaussian_fit_recovers_center_and_positive_intensity() -> None:
    ny, nx = 17, 17
    y, x = np.indices((ny, nx), dtype=float)
    true_x = 8.4
    true_y = 7.7
    sigma_x = 1.5
    sigma_y = 1.1
    theta = 0.3
    background = 10.0
    amplitude = 200.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_shift = x - true_x
    y_shift = y - true_y
    x_rot = cos_t * x_shift + sin_t * y_shift
    y_rot = -sin_t * x_shift + cos_t * y_shift
    patch = background + amplitude * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    result = fit_gaussian_patch(patch, x_pred_local=true_x, y_pred_local=true_y)
    assert result.success
    assert abs(result.x0 - true_x) < 0.3
    assert abs(result.y0 - true_y) < 0.3
    assert result.integrated_intensity > 0.0
