from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .geometry import UnitCell, compute_excitation_error, electron_wavelength_angstrom
from .reflection_metrics import estimate_extinction_distance, zone_axis_proximity_from_angle


@dataclass(slots=True)
class ThicknessValidationConfig:
    """Configuration for simplified Bloch-like thickness scans."""

    thickness_min_nm: float = 0.0
    thickness_max_nm: float = 300.0
    thickness_step_nm: float = 5.0
    multi_beam_coupling: float = 0.15
    harmonics: int = 3
    xi_scale_nm: float = 200.0
    structure_factor_decay: float = 0.35
    beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    zone_axis_soft_angle_deg: float = 5.0


def thickness_grid(config: ThicknessValidationConfig) -> np.ndarray:
    """Generate a thickness grid in nanometers."""
    return np.arange(
        float(config.thickness_min_nm),
        float(config.thickness_max_nm) + float(config.thickness_step_nm),
        float(config.thickness_step_nm),
        dtype=np.float64,
    )


def two_beam_intensity(sg: float, xi_g_angstrom: float, thickness_angstrom: np.ndarray) -> np.ndarray:
    """Return a compact two-beam oscillation curve."""
    q = sg * xi_g_angstrom
    omega = np.sqrt(1.0 + q ** 2)
    phase = np.pi * thickness_angstrom * omega / max(xi_g_angstrom, 1e-8)
    return np.sin(phase) ** 2 / (1.0 + q ** 2)


def _deterministic_phases(hkl: Sequence[int], harmonics: int) -> np.ndarray:
    seed = int(abs(hkl[0]) * 73856093 + abs(hkl[1]) * 19349663 + abs(hkl[2]) * 83492791)
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=harmonics)


def simulate_intensity_vs_thickness(
    ub: np.ndarray,
    reflection_hkl: Sequence[int],
    cell: UnitCell,
    thicknesses_nm: np.ndarray | None = None,
    n_excited: float = 0.0,
    zone_axis_angle_deg: float = 90.0,
    config: ThicknessValidationConfig | None = None,
) -> pd.DataFrame:
    """Simulate a simplified Bloch-like thickness response for one reflection."""
    cfg = config or ThicknessValidationConfig()
    t_nm = thicknesses_nm if thicknesses_nm is not None else thickness_grid(cfg)
    t_angstrom = np.asarray(t_nm, dtype=np.float64) * 10.0
    hkl = np.asarray(reflection_hkl, dtype=np.float64).reshape(1, 3)

    wavelength_angstrom = electron_wavelength_angstrom(cell.voltage_kv)
    sg = float(compute_excitation_error(ub, hkl, wavelength_angstrom=wavelength_angstrom, beam_direction=cfg.beam_direction)[0])
    xi_g = float(
        estimate_extinction_distance(
            hkls=hkl,
            cell=cell,
            xi_scale_nm=cfg.xi_scale_nm,
            structure_factor_decay=cfg.structure_factor_decay,
        )[0]
    )
    base = two_beam_intensity(sg, xi_g, t_angstrom)

    zone_proximity = float(zone_axis_proximity_from_angle(zone_axis_angle_deg, cfg.zone_axis_soft_angle_deg))
    crowding_term = np.log1p(max(float(n_excited), 0.0))
    coupling = cfg.multi_beam_coupling * zone_proximity * crowding_term / max(np.log(10.0), 1e-8)

    phases = _deterministic_phases(tuple(int(v) for v in reflection_hkl), cfg.harmonics)
    modulation = np.zeros_like(base)
    for harmonic_index, phase in enumerate(phases, start=1):
        period = xi_g * (1.0 + 0.35 * harmonic_index)
        modulation += np.sin(2.0 * np.pi * t_angstrom / max(period, 1e-8) + phase) / harmonic_index

    intensity = np.clip(base * (1.0 + coupling * modulation) + 0.02 * coupling * np.abs(modulation), 0.0, None)

    return pd.DataFrame(
        {
            "thickness_nm": np.asarray(t_nm, dtype=np.float64),
            "intensity": intensity,
            "sg": sg,
            "xi_g": xi_g,
            "h": int(reflection_hkl[0]),
            "k": int(reflection_hkl[1]),
            "l": int(reflection_hkl[2]),
            "N_excited": float(n_excited),
            "zone_axis_angle": float(zone_axis_angle_deg),
        }
    )


def thickness_sensitivity_metrics(scan: pd.DataFrame) -> dict[str, float]:
    """Compute thickness-sensitivity summary statistics for a scan."""
    intensity = scan["intensity"].to_numpy(dtype=np.float64)
    thickness = scan["thickness_nm"].to_numpy(dtype=np.float64)
    if intensity.size < 2:
        raise ValueError("Thickness scan must contain at least two points.")

    smoothed = intensity
    if intensity.size >= 7:
        smoothed = savgol_filter(intensity, window_length=7, polyorder=2, mode="interp")
    derivative = np.gradient(smoothed, thickness)

    min_intensity = float(np.min(intensity))
    max_intensity = float(np.max(intensity))
    mean_intensity = float(np.mean(intensity))
    std_intensity = float(np.std(intensity))
    ratio = float(max_intensity / min_intensity) if min_intensity > 0 else float("inf")
    derivative_rms = float(np.sqrt(np.mean(derivative ** 2)))
    normalized_std = std_intensity / max(mean_intensity, 1e-8)
    sensitivity = normalized_std + derivative_rms

    return {
        "thickness_std": std_intensity,
        "thickness_range": max_intensity - min_intensity,
        "thickness_cv": normalized_std,
        "thickness_derivative_rms": derivative_rms,
        "thickness_max_min_ratio": ratio,
        "thickness_sensitivity": sensitivity,
    }
