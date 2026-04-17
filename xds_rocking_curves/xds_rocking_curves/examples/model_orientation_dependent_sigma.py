#!/usr/bin/env python3
"""Estimate orientation-dependent extra sigma for thickness trend analysis.

Model idea:
- For each HKL (shared across all 4 datasets), fit y(thickness) with linear trend.
- Collect residuals r = y - y_hat.
- Pool residual spread by orientation bins phi and estimate sigma(phi).
- Define additional orientation sigma as sigma_add(phi) = max(0, sigma(phi) - sigma_global).

Outputs are intended for downstream weighting/error-model design.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers import parse_gxparm


DEFAULT_INPUT = PROJECT_ROOT / "real_data_output" / "LTA_all_reflections_integrate_summary" / "shared_hkl_analysis" / "shared_hkls_long.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "real_data_output" / "LTA_all_reflections_integrate_summary" / "shared_hkl_analysis" / "orientation_sigma_model"
BASE_XDS_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness")
DATASET_ORDER = ["LTA_t1", "LTA_t2", "LTA_t3", "LTA_t4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-orientation-bins", type=int, default=18)
    return parser.parse_args()


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad)


def fit_line_residuals(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.full_like(y, np.nan), np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat
    return resid, float(slope), float(intercept)


def add_phi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phi_deg"] = np.nan

    for ds in DATASET_ORDER:
        gx = parse_gxparm(BASE_XDS_DIR / ds / "GXPARM.XDS")
        mask = out["dataset"] == ds
        if not mask.any():
            continue
        z = pd.to_numeric(out.loc[mask, "z_cal"], errors="coerce").to_numpy(dtype=float)
        phi = gx.phi0_deg + (z - gx.starting_frame) * gx.dphi_deg
        out.loc[mask, "phi_deg"] = phi

    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = add_phi(df)

    df["thickness_nm"] = pd.to_numeric(df["thickness_nm"], errors="coerce")
    df["isig"] = pd.to_numeric(df["isig"], errors="coerce")
    df["I"] = pd.to_numeric(df["I"], errors="coerce")
    df["logI"] = np.log1p(df["I"].clip(lower=0))

    rows_hkl: list[dict[str, float | int]] = []
    rows_obs: list[pd.DataFrame] = []

    grouped = df.sort_values(["h", "k", "l", "thickness_nm"]).groupby(["h", "k", "l"], sort=False)
    for (h, k, l), g in grouped:
        x = g["thickness_nm"].to_numpy(dtype=float)

        y_isig = g["isig"].to_numpy(dtype=float)
        resid_isig, slope_isig, intercept_isig = fit_line_residuals(x, y_isig)

        y_logi = g["logI"].to_numpy(dtype=float)
        resid_logi, slope_logi, intercept_logi = fit_line_residuals(x, y_logi)

        g2 = g.copy()
        g2["resid_isig"] = resid_isig
        g2["resid_logI"] = resid_logi
        rows_obs.append(g2)

        rows_hkl.append(
            {
                "h": int(h),
                "k": int(k),
                "l": int(l),
                "n_points": int(len(g)),
                "phi_mean_deg": float(np.nanmean(g["phi_deg"].to_numpy(dtype=float))),
                "isig_slope_per_nm": slope_isig,
                "isig_intercept": intercept_isig,
                "isig_resid_rms": float(np.sqrt(np.nanmean(resid_isig ** 2))) if np.isfinite(resid_isig).any() else np.nan,
                "logI_slope_per_nm": slope_logi,
                "logI_intercept": intercept_logi,
                "logI_resid_rms": float(np.sqrt(np.nanmean(resid_logi ** 2))) if np.isfinite(resid_logi).any() else np.nan,
            }
        )

    obs = pd.concat(rows_obs, ignore_index=True)
    per_hkl = pd.DataFrame(rows_hkl)

    obs_valid = obs[np.isfinite(obs["phi_deg"])].copy()
    obs_valid["phi_bin"] = pd.qcut(
        obs_valid["phi_deg"],
        q=min(args.n_orientation_bins, obs_valid["phi_deg"].nunique()),
        duplicates="drop",
    )

    global_sigma_isig = robust_sigma(obs_valid["resid_isig"].to_numpy(dtype=float))
    global_sigma_logi = robust_sigma(obs_valid["resid_logI"].to_numpy(dtype=float))

    sigma_rows = []
    for phi_bin, g in obs_valid.groupby("phi_bin", observed=False):
        s_isig = robust_sigma(g["resid_isig"].to_numpy(dtype=float))
        s_logi = robust_sigma(g["resid_logI"].to_numpy(dtype=float))
        phi_mid = float((phi_bin.left + phi_bin.right) / 2.0)
        sigma_rows.append(
            {
                "phi_bin": str(phi_bin),
                "phi_left_deg": float(phi_bin.left),
                "phi_right_deg": float(phi_bin.right),
                "phi_mid_deg": phi_mid,
                "n_points": int(len(g)),
                "sigma_resid_isig": s_isig,
                "sigma_add_isig": max(0.0, s_isig - global_sigma_isig) if np.isfinite(s_isig) and np.isfinite(global_sigma_isig) else np.nan,
                "sigma_resid_logI": s_logi,
                "sigma_add_logI": max(0.0, s_logi - global_sigma_logi) if np.isfinite(s_logi) and np.isfinite(global_sigma_logi) else np.nan,
            }
        )

    sigma_by_phi = pd.DataFrame(sigma_rows).sort_values("phi_mid_deg").reset_index(drop=True)

    # Map expected orientation sigma to each observation and compute normalized residuals.
    obs_model = obs_valid.merge(
        sigma_by_phi[["phi_bin", "sigma_resid_isig", "sigma_resid_logI"]],
        on="phi_bin",
        how="left",
    )
    obs_model["norm_resid_isig"] = obs_model["resid_isig"] / obs_model["sigma_resid_isig"].replace(0, np.nan)
    obs_model["norm_resid_logI"] = obs_model["resid_logI"] / obs_model["sigma_resid_logI"].replace(0, np.nan)

    # Reflection-level excess variability versus orientation-expected residual scale.
    hkl_excess = (
        obs_model.groupby(["h", "k", "l"], as_index=False)
        .agg(
            phi_mean_deg=("phi_deg", "mean"),
            rms_resid_isig=("resid_isig", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            rms_resid_logI=("resid_logI", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            rms_norm_resid_isig=("norm_resid_isig", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            rms_norm_resid_logI=("norm_resid_logI", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
        )
        .copy()
    )
    hkl_excess = hkl_excess.sort_values("rms_norm_resid_isig", ascending=False).reset_index(drop=True)

    # Save outputs.
    per_hkl.to_csv(output_dir / "per_hkl_linear_residual_metrics.csv", index=False)
    obs_model.to_csv(output_dir / "per_observation_with_orientation_residuals.csv", index=False)
    sigma_by_phi.to_csv(output_dir / "sigma_by_orientation_bin.csv", index=False)
    hkl_excess.to_csv(output_dir / "hkl_excess_variability_ranked.csv", index=False)
    hkl_excess.head(500).to_csv(output_dir / "top_500_hkls_excess_variability.csv", index=False)

    # Compact markdown handoff.
    lines = []
    lines.append("# Orientation-Dependent Sigma Model")
    lines.append("")
    lines.append("Residual model per HKL:")
    lines.append("- Fit linear trend vs thickness for `isig` and `logI`.")
    lines.append("- Residuals are pooled by orientation bins (`phi`).")
    lines.append("")
    lines.append(f"Global robust residual sigma (isig): {global_sigma_isig:.6f}")
    lines.append(f"Global robust residual sigma (logI): {global_sigma_logi:.6f}")
    lines.append("")
    lines.append("Recommended additive error model (by orientation bin):")
    lines.append("- `sigma_add_isig(phi) = max(0, sigma_resid_isig(phi) - sigma_global_isig)`")
    lines.append("- `sigma_add_logI(phi) = max(0, sigma_resid_logI(phi) - sigma_global_logI)`")
    lines.append("")
    lines.append(f"HKLs modeled: {len(per_hkl)}")
    lines.append(f"Observations modeled: {len(obs_model)}")
    lines.append("")
    lines.append("Outputs:")
    lines.append("- sigma_by_orientation_bin.csv")
    lines.append("- hkl_excess_variability_ranked.csv")
    lines.append("- top_500_hkls_excess_variability.csv")
    lines.append("- per_hkl_linear_residual_metrics.csv")
    lines.append("- per_observation_with_orientation_residuals.csv")

    (output_dir / "CHATGPT_PRO_ORIENTATION_SIGMA.md").write_text("\n".join(lines) + "\n")

    print(f"Output dir: {output_dir.resolve()}")
    print(f"HKLs modeled: {len(per_hkl)}")
    print(f"Observations modeled: {len(obs_model)}")
    print(f"Global sigma isig: {global_sigma_isig:.6f}")
    print(f"Global sigma logI: {global_sigma_logi:.6f}")


if __name__ == "__main__":
    main()
