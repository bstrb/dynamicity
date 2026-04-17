from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers import parse_integrate_hkl
from src.pipeline import AnalysisConfig, analyze_single_reflection_dataset


LTA_TIFF_TEMPLATE = "/home/bubl3932/files/3DED-DATA/LTA/LTA_t1/image/??????.tiff"
LTA_XDS_DIR = Path("/home/bubl3932/files/3DED-DATA/LTA/xds_uniform_thickness/LTA_t1")

GXPARM_PATH = LTA_XDS_DIR / "GXPARM.XDS"
XDS_INP_PATH = LTA_XDS_DIR / "XDS.INP"
SPOT_XDS_PATH = LTA_XDS_DIR / "SPOT.XDS"
INTEGRATE_HKL_PATH = LTA_XDS_DIR / "INTEGRATE.HKL"

OUTPUT_ROOT = PROJECT_ROOT / "real_data_output" / "LTA_t1_candidates"


def select_candidates(integrate_path: Path, n_candidates: int = 20) -> pd.DataFrame:
    obs = parse_integrate_hkl(integrate_path).observations.copy()
    obs = obs.replace([np.inf, -np.inf], np.nan)
    obs = obs.dropna(subset=["h", "k", "l", "I", "sigma", "x_cal", "y_cal", "z_cal"])
    obs = obs[obs["sigma"] > 0].copy()
    obs["isig"] = obs["I"] / obs["sigma"]
    obs = obs[(obs["I"] > 0) & (obs["isig"] > 4.0)].copy()

    # Keep spots away from detector edge for stable local patch fitting.
    edge_margin = 20.0
    obs = obs[
        (obs["x_cal"] >= edge_margin)
        & (obs["x_cal"] <= 512.0 - edge_margin)
        & (obs["y_cal"] >= edge_margin)
        & (obs["y_cal"] <= 512.0 - edge_margin)
    ].copy()

    if obs.empty:
        raise RuntimeError("No candidate observations remained after filtering.")

    obs["radius_px"] = np.hypot(obs["x_cal"] - 256.0, obs["y_cal"] - 256.0)
    obs["quality"] = np.log1p(obs["I"].clip(lower=0)) + 2.0 * np.log1p(obs["isig"].clip(lower=0))

    # Encourage detector-position diversity by selecting the best few in radial bins.
    obs["r_bin"] = pd.qcut(obs["radius_px"], q=6, labels=False, duplicates="drop")
    per_bin = max(2, n_candidates // max(1, int(obs["r_bin"].nunique())))
    picks = (
        obs.sort_values(["quality"], ascending=False)
        .groupby("r_bin", group_keys=False)
        .head(per_bin)
        .sort_values("quality", ascending=False)
        .head(n_candidates)
        .copy()
    )
    picks["candidate_reason"] = (
        "high I/sigma; away from detector edge; selected for radial diversity"
    )
    return picks.reset_index(drop=True)


def summarize_curve(curve: pd.DataFrame) -> dict[str, float | int]:
    if curve.empty:
        return {
            "n_relevant_frames": 0,
            "n_fit_success": 0,
            "fit_success_rate": 0.0,
            "median_r2": np.nan,
            "median_rmse": np.nan,
            "max_I_fit": np.nan,
        }
    fit_ok = curve["fit_success"].fillna(False).astype(bool)
    r2 = pd.to_numeric(curve["r_squared"], errors="coerce")
    rmse = pd.to_numeric(curve["rmse"], errors="coerce")
    i_fit = pd.to_numeric(curve["I_fit"], errors="coerce")
    n_total = int(len(curve))
    n_ok = int(fit_ok.sum())
    return {
        "n_relevant_frames": n_total,
        "n_fit_success": n_ok,
        "fit_success_rate": float(n_ok / n_total) if n_total else 0.0,
        "median_r2": float(np.nanmedian(r2.to_numpy(dtype=float))) if n_total else np.nan,
        "median_rmse": float(np.nanmedian(rmse.to_numpy(dtype=float))) if n_total else np.nan,
        "max_I_fit": float(np.nanmax(i_fit.to_numpy(dtype=float))) if n_total else np.nan,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = select_candidates(INTEGRATE_HKL_PATH, n_candidates=20)
    candidates_path = OUTPUT_ROOT / "candidate_reflections_selected.csv"
    candidates.to_csv(candidates_path, index=False)

    rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        hkl = (int(row.h), int(row.k), int(row.l))
        frame_hint = int(np.floor(float(row.z_cal) + 0.5))
        run_dir = OUTPUT_ROOT / f"hkl_{hkl[0]}_{hkl[1]}_{hkl[2]}_f{frame_hint}"

        results = analyze_single_reflection_dataset(
            gxparm_path=GXPARM_PATH,
            xds_inp_path=XDS_INP_PATH,
            spot_xds_path=SPOT_XDS_PATH,
            integrate_hkl_path=INTEGRATE_HKL_PATH,
            image_glob=None,
            image_template=LTA_TIFF_TEMPLATE,
            config=AnalysisConfig(
                dataset_name="LTA_t1",
                thickness_nm=100.0,
                hkl=hkl,
                relevance_mode="window",
                window_half_width=5,
                patch_half_size=7,
                max_center_shift_px=3.0,
                initial_sigma_px=1.5,
                min_sigma_px=0.5,
                max_sigma_px=6.0,
                auto_choose_rotation_sign=True,
            ),
            output_dir=run_dir,
        )

        curve_stats = summarize_curve(results.curve)
        rows.append(
            {
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "z_cal_hint": float(row.z_cal),
                "x_cal_hint": float(row.x_cal),
                "y_cal_hint": float(row.y_cal),
                "I_hint": float(row.I),
                "sigma_hint": float(row.sigma),
                "isig_hint": float(row.isig),
                "candidate_reason": str(row.candidate_reason),
                "output_dir": str(run_dir),
                **curve_stats,
            }
        )

    summary = pd.DataFrame(rows)
    # Ranking combines fit success and goodness-of-fit.
    summary["ranking_score"] = (
        summary["fit_success_rate"].fillna(0.0) * 0.7
        + summary["median_r2"].fillna(0.0).clip(lower=0.0, upper=1.0) * 0.3
    )
    summary = summary.sort_values(
        ["ranking_score", "fit_success_rate", "median_r2", "max_I_fit"],
        ascending=False,
    ).reset_index(drop=True)

    summary_path = OUTPUT_ROOT / "reflection_ranking_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.head(5)
    best_path = OUTPUT_ROOT / "best_reflections.csv"
    best.to_csv(best_path, index=False)

    print(f"Selected candidates: {candidates_path}")
    print(f"Ranking summary: {summary_path}")
    print(f"Best reflections: {best_path}")
    if not best.empty:
        print("Top reflections:")
        print(best[["h", "k", "l", "fit_success_rate", "median_r2", "n_relevant_frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
