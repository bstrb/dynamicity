"""Compare predicted thickness trends against real reflection intensities.

This script verifies whether Bloch-wave predicted intensity-vs-thickness behavior
agrees with observed INTEGRATE.HKL reflection trends for one sample family
(e.g., LTA or STW) across folders like:

- data/LTA_t1, data/LTA_t2, data/LTA_t3, data/LTA_t4
- data/STW_t1_360, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parsers import parse_composition, parse_gxparm, parse_integrate_hkl, load_optional_xds_inp
from src.pipeline import AnalysisConfig, run_analysis


def _hkl_equiv_key(h: int, k: int, l: int) -> str:
    vals = sorted((abs(int(h)), abs(int(k)), abs(int(l))))
    return f"{vals[0]}_{vals[1]}_{vals[2]}"


def _hkl_exact_key(h: int, k: int, l: int) -> str:
    return f"{int(h)}_{int(k)}_{int(l)}"


def _make_hkl_key(h: int, k: int, l: int, mode: str) -> str:
    if mode == "exact":
        return _hkl_exact_key(h, k, l)
    return _hkl_equiv_key(h, k, l)


def _discover_thickness_dirs(data_root: Path, sample: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(sample)}_t(\d+)(?:_.*)?$")
    found: list[tuple[int, Path]] = []
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m is None:
            continue
        t_rank = int(m.group(1))
        if (child / "GXPARM.XDS").exists() and (child / "INTEGRATE.HKL").exists():
            found.append((t_rank, child))
    found.sort(key=lambda x: x[0])
    return found


def _parse_thickness_map(text: str | None, thickness_ranks: list[int]) -> dict[int, float]:
    if text is None:
        return {int(r): float(r) for r in thickness_ranks}

    mapping: dict[int, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid thickness map term: {item!r}. Expected form rank:nm")
        left, right = item.split(":", 1)
        rank = int(left.strip())
        nm = float(right.strip())
        mapping[rank] = nm

    missing = [r for r in thickness_ranks if r not in mapping]
    if missing:
        raise ValueError(f"Missing thickness mapping for ranks: {missing}")
    return mapping


def _build_observed_table(thickness_dirs: list[tuple[int, Path]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for t_rank, folder in thickness_dirs:
        gxparm = parse_gxparm(folder / "GXPARM.XDS")
        integrate = parse_integrate_hkl(folder / "INTEGRATE.HKL")
        obs = integrate.observations.copy()
        obs["frame_number"] = obs["frame_est"].astype(int) + 1
        obs["phi_deg"] = gxparm.phi0_deg + gxparm.dphi_deg * (obs["frame_number"] - 1)
        grouped = (
            obs.groupby(["frame_number", "phi_deg", "h", "k", "l"], as_index=False)
            .agg(I_obs=("I", "median"), sigma_obs=("sigma", "median"), n_obs=("I", "size"))
        )
        grouped["thickness_rank"] = int(t_rank)
        grouped["dataset_folder"] = folder.name
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _match_observed_to_target_phi(
    obs: pd.DataFrame,
    target_phi_deg: float,
    phi_tolerance_deg: float,
) -> pd.DataFrame:
    """Select one nearest orientation frame per thickness near target phi."""
    if obs.empty:
        return obs

    frame_phi = (
        obs[["thickness_rank", "frame_number", "phi_deg"]]
        .drop_duplicates()
        .copy()
    )
    frame_phi["phi_abs_err"] = np.abs(frame_phi["phi_deg"] - float(target_phi_deg))
    nearest = (
        frame_phi.sort_values(["thickness_rank", "phi_abs_err", "frame_number"])
        .groupby("thickness_rank", as_index=False)
        .first()
    )
    nearest = nearest[nearest["phi_abs_err"] <= float(phi_tolerance_deg)].copy()
    if nearest.empty:
        return obs.iloc[0:0].copy()

    return obs.merge(
        nearest[["thickness_rank", "frame_number"]],
        on=["thickness_rank", "frame_number"],
        how="inner",
    )


def _gaussian_phi_weight(phi_abs_err: pd.Series, sigma_deg: float) -> pd.Series:
    sigma = max(float(sigma_deg), 1e-6)
    z = phi_abs_err.to_numpy(dtype=float) / sigma
    return pd.Series(np.exp(-0.5 * z * z), index=phi_abs_err.index)


def _aggregate_pred_in_phi_window(
    pred: pd.DataFrame,
    target_phi_deg: float,
    phi_tolerance_deg: float,
) -> pd.DataFrame:
    pw = pred.copy()
    pw["phi_abs_err"] = np.abs(pw["phi_deg"] - float(target_phi_deg))
    pw = pw[pw["phi_abs_err"] <= float(phi_tolerance_deg)].copy()
    if pw.empty:
        return pw

    sigma_deg = max(float(phi_tolerance_deg) / 2.0, 1e-6)
    pw["w_phi"] = _gaussian_phi_weight(pw["phi_abs_err"], sigma_deg=sigma_deg)

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g["w_phi"].to_numpy(dtype=float)
        y = g["I_pred"].to_numpy(dtype=float)
        s = g["S_comb"].to_numpy(dtype=float)
        wsum = float(np.sum(w))
        if wsum <= 0:
            return pd.Series(
                {
                    "I_pred": float(np.median(y)),
                    "S_comb": float(np.max(s)),
                    "h": int(g.iloc[0]["h"]),
                    "k": int(g.iloc[0]["k"]),
                    "l": int(g.iloc[0]["l"]),
                    "n_pred": int(g.shape[0]),
                }
            )
        return pd.Series(
            {
                "I_pred": float(np.sum(w * y) / wsum),
                "S_comb": float(np.sum(w * s) / wsum),
                "h": int(g.iloc[0]["h"]),
                "k": int(g.iloc[0]["k"]),
                "l": int(g.iloc[0]["l"]),
                "n_pred": int(g.shape[0]),
            }
        )

    out = (
        pw.groupby(["thickness_rank", "hkl_key"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    return out


def _aggregate_obs_in_phi_window(
    obs: pd.DataFrame,
    target_phi_deg: float,
    phi_tolerance_deg: float,
) -> pd.DataFrame:
    ow = obs.copy()
    ow["phi_abs_err"] = np.abs(ow["phi_deg"] - float(target_phi_deg))
    ow = ow[ow["phi_abs_err"] <= float(phi_tolerance_deg)].copy()
    if ow.empty:
        return ow

    sigma_deg = max(float(phi_tolerance_deg) / 2.0, 1e-6)
    ow["w_phi"] = _gaussian_phi_weight(ow["phi_abs_err"], sigma_deg=sigma_deg)

    sigma_vals = ow["sigma_obs"].to_numpy(dtype=float)
    positive_sigma = sigma_vals[sigma_vals > 0]
    sigma_floor = float(np.median(positive_sigma)) if positive_sigma.size else 1.0
    ow["w_sigma"] = 1.0 / np.maximum(ow["sigma_obs"].to_numpy(dtype=float), sigma_floor) ** 2
    ow["w"] = ow["w_phi"] * ow["w_sigma"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        w = g["w"].to_numpy(dtype=float)
        y = g["I_obs"].to_numpy(dtype=float)
        wsum = float(np.sum(w))
        if wsum <= 0:
            i_obs = float(np.median(y))
        else:
            i_obs = float(np.sum(w * y) / wsum)
        return pd.Series(
            {
                "I_obs": i_obs,
                "sigma_obs": float(np.median(g["sigma_obs"].to_numpy(dtype=float))),
                "n_obs": int(np.sum(g["n_obs"].to_numpy(dtype=int))),
            }
        )

    out = (
        ow.groupby(["thickness_rank", "hkl_key"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    return out


def _safe_minmax_norm(values: pd.Series) -> pd.Series:
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        return pd.Series(np.full(values.shape[0], 0.5), index=values.index)
    return (values - vmin) / (vmax - vmin)


def _correlation_table(merged: pd.DataFrame) -> pd.DataFrame:
    if "hkl_key" in merged.columns:
        keys = ["frame_number", "hkl_key"]
    else:
        keys = ["frame_number", "h", "k", "l"]
    out_rows: list[dict[str, float | int]] = []

    for key_vals, g in merged.groupby(keys):
        if g["thickness_rank"].nunique() < 3:
            continue
        g = g.sort_values("thickness_rank").copy()

        pred_norm = _safe_minmax_norm(g["I_pred"])
        obs_norm = _safe_minmax_norm(g["I_obs"])

        x = pred_norm.to_numpy(dtype=float)
        y = obs_norm.to_numpy(dtype=float)

        pear = pearsonr(x, y)
        spear = spearmanr(x, y)

        row: dict[str, float | int | str] = {
            "frame_number": int(key_vals[0]),
            "n_thickness": int(g["thickness_rank"].nunique()),
            "pearson_r": float(pear.statistic),
            "pearson_p": float(pear.pvalue),
            "spearman_rho": float(spear.statistic),
            "spearman_p": float(spear.pvalue),
            "mean_S_comb": float(g["S_comb"].mean()),
        }
        if "hkl_key" in merged.columns:
            row["hkl_key"] = str(key_vals[1])
            if all(col in g.columns for col in ["h", "k", "l"]):
                row["h"] = int(g.iloc[0]["h"])
                row["k"] = int(g.iloc[0]["k"])
                row["l"] = int(g.iloc[0]["l"])
        else:
            row["h"] = int(key_vals[1])
            row["k"] = int(key_vals[2])
            row["l"] = int(key_vals[3])
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    return pd.DataFrame.from_records(out_rows).sort_values(
        ["spearman_rho", "pearson_r"], ascending=False
    )


def _save_summary_plots(
    merged: pd.DataFrame,
    corr_table: pd.DataFrame,
    out_dir: Path,
    sample: str,
    frame_number: int | None,
) -> None:
    # Global predicted-vs-observed scatter (normalized per reflection trend)
    if not merged.empty:
        normed = []
        for _, g in merged.groupby(["frame_number", "h", "k", "l"]):
            if g["thickness_rank"].nunique() < 3:
                continue
            g = g.copy()
            g["pred_norm"] = _safe_minmax_norm(g["I_pred"])
            g["obs_norm"] = _safe_minmax_norm(g["I_obs"])
            normed.append(g)
        if normed:
            ndf = pd.concat(normed, ignore_index=True)
            fig, ax = plt.subplots(figsize=(6.6, 6.0))
            ax.scatter(ndf["pred_norm"], ndf["obs_norm"], s=11, alpha=0.35)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_xlabel("Predicted normalized intensity")
            ax.set_ylabel("Observed normalized intensity")
            title_suffix = f"frame {frame_number}" if frame_number is not None else "all frames"
            ax.set_title(f"{sample}: predicted vs observed thickness trends ({title_suffix})")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / f"{sample.lower()}_pred_vs_obs_scatter.png", dpi=220)
            plt.close(fig)

    # Curves for top-correlated reflections
    if corr_table.empty:
        return

    top = corr_table.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax in axes_flat:
        ax.grid(alpha=0.25)

    for i, row in enumerate(top.itertuples(index=False)):
        ax = axes_flat[i]
        mask = (
            (merged["frame_number"] == int(row.frame_number))
            & (merged["h"] == int(row.h))
            & (merged["k"] == int(row.k))
            & (merged["l"] == int(row.l))
        )
        g = merged.loc[mask].sort_values("thickness_rank").copy()
        if g.empty:
            continue

        pred = _safe_minmax_norm(g["I_pred"])
        obs = _safe_minmax_norm(g["I_obs"])
        ax.plot(g["thickness_rank"], pred, marker="o", linewidth=1.8, label="pred")
        ax.plot(g["thickness_rank"], obs, marker="s", linewidth=1.8, label="obs")
        ax.set_title(
            f"F{int(row.frame_number)} ({int(row.h)} {int(row.k)} {int(row.l)})\n"
            f"rho={row.spearman_rho:.2f}, r={row.pearson_r:.2f}",
            fontsize=9,
        )

    for j in range(len(top), len(axes_flat)):
        axes_flat[j].axis("off")

    axes_flat[0].legend(fontsize=8, loc="lower right")
    fig.supxlabel("Thickness rank")
    fig.supylabel("Normalized intensity")
    fig.suptitle(f"{sample}: top reflection trend agreements", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample.lower()}_top_trend_matches.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=["LTA", "STW"], required=True, help="Sample family")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--composition", default="24 Si, 48 O", help="Composition string")
    parser.add_argument("--frame-number", type=int, default=None, help="Optional one-based frame number to verify")
    parser.add_argument(
        "--thickness-map",
        default=None,
        help="Optional rank:nm mapping, e.g. '1:100,2:200,3:350,4:600'",
    )
    parser.add_argument(
        "--phi-tolerance-deg",
        type=float,
        default=0.40,
        help="Max |delta phi| when matching observed frames to target orientation",
    )
    parser.add_argument(
        "--frame-match-mode",
        choices=["nearest", "windowed"],
        default="windowed",
        help="For single-frame verification: nearest-frame matching or phi-window pooled matching.",
    )
    parser.add_argument(
        "--hkl-match-mode",
        choices=["exact", "equiv"],
        default="exact",
        help="Match reflections by exact signed HKL or symmetry-like absolute/permuted equivalent key.",
    )

    parser.add_argument("--dmin", type=float, default=0.6, help="Minimum d-spacing (angstrom)")
    parser.add_argument("--dmax", type=float, default=50.0, help="Maximum d-spacing (angstrom)")
    parser.add_argument(
        "--excitation-tolerance",
        type=float,
        default=1.5e-3,
        help="Excitation tolerance (1/angstrom)",
    )
    parser.add_argument("--filter-untrusted", action="store_true", help="Exclude untrusted regions")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thickness_dirs = _discover_thickness_dirs(data_root, args.sample)
    if len(thickness_dirs) < 3:
        raise RuntimeError(
            f"Need at least 3 thickness folders for {args.sample}. Found {len(thickness_dirs)}."
        )

    thickness_ranks = [t for t, _ in thickness_dirs]
    thickness_map = _parse_thickness_map(args.thickness_map, thickness_ranks)
    thickness_values_nm = [float(thickness_map[r]) for r in thickness_ranks]
    reference_dir = thickness_dirs[0][1]

    gxparm = parse_gxparm(reference_dir / "GXPARM.XDS")
    integrate_ref = parse_integrate_hkl(reference_dir / "INTEGRATE.HKL")
    xds_input = load_optional_xds_inp(reference_dir / "XDS.INP")
    composition = parse_composition(args.composition)

    config = AnalysisConfig(
        dmin_angstrom=args.dmin,
        dmax_angstrom=args.dmax,
        excitation_tolerance_invA=args.excitation_tolerance,
        mode="thickness",
        thickness_nm=np.asarray(thickness_values_nm, dtype=float),
        filter_untrusted=args.filter_untrusted,
    )

    result = run_analysis(
        gxparm=gxparm,
        integrate=integrate_ref,
        composition=composition,
        xds_input=xds_input,
        config=config,
    )

    if result.thickness_long is None or result.thickness_long.empty:
        raise RuntimeError("Pipeline did not produce thickness_long output.")

    pred = result.thickness_long[["frame_number", "phi_deg", "h", "k", "l", "thickness_nm", "intensity", "S_comb"]].copy()
    inv_map = {float(v): int(k) for k, v in thickness_map.items()}
    pred["thickness_rank"] = pred["thickness_nm"].map(lambda v: inv_map.get(float(v), -1)).astype(int)
    pred = pred[pred["thickness_rank"] >= 0].copy()
    pred = pred.rename(columns={"intensity": "I_pred"})
    pred["hkl_key"] = [
        _make_hkl_key(h, k, l, mode=args.hkl_match_mode)
        for h, k, l in pred[["h", "k", "l"]].itertuples(index=False, name=None)
    ]

    obs = _build_observed_table(thickness_dirs)
    if obs.empty:
        raise RuntimeError("Observed table is empty; check INTEGRATE.HKL parsing.")
    obs["hkl_key"] = [
        _make_hkl_key(h, k, l, mode=args.hkl_match_mode)
        for h, k, l in obs[["h", "k", "l"]].itertuples(index=False, name=None)
    ]
    obs["thickness_nm"] = obs["thickness_rank"].map(thickness_map).astype(float)

    if args.frame_number is not None:
        target_frame = int(args.frame_number)
        frame_match = result.frame_summary[result.frame_summary["frame_number"] == target_frame]
        if frame_match.empty:
            raise RuntimeError(f"Requested frame {target_frame} not found in predicted frame summary.")
        target_phi_deg = float(frame_match.iloc[0]["phi_deg"])

        if args.frame_match_mode == "nearest":
            pred = pred[pred["frame_number"] == target_frame].copy()
            pred = (
                pred.groupby(["frame_number", "thickness_rank", "hkl_key"], as_index=False)
                .agg(I_pred=("I_pred", "median"), S_comb=("S_comb", "max"), thickness_nm=("thickness_nm", "first"), h=("h", "first"), k=("k", "first"), l=("l", "first"))
            )
            obs = _match_observed_to_target_phi(
                obs=obs,
                target_phi_deg=target_phi_deg,
                phi_tolerance_deg=args.phi_tolerance_deg,
            )
            obs = (
                obs.groupby(["thickness_rank", "hkl_key"], as_index=False)
                .agg(I_obs=("I_obs", "median"), sigma_obs=("sigma_obs", "median"), n_obs=("n_obs", "sum"), thickness_nm=("thickness_nm", "first"))
            )
        else:
            pred = _aggregate_pred_in_phi_window(
                pred=pred,
                target_phi_deg=target_phi_deg,
                phi_tolerance_deg=args.phi_tolerance_deg,
            )
            obs = _aggregate_obs_in_phi_window(
                obs=obs,
                target_phi_deg=target_phi_deg,
                phi_tolerance_deg=args.phi_tolerance_deg,
            )

        if pred.empty or obs.empty:
            raise RuntimeError(
                "No predicted/observed points after frame matching. "
                "Try increasing --phi-tolerance-deg or using --frame-match-mode nearest."
            )

        if "thickness_nm" not in pred.columns:
            pred = pred.merge(
                pd.DataFrame(
                    {
                        "thickness_rank": list(thickness_map.keys()),
                        "thickness_nm": [float(v) for v in thickness_map.values()],
                    }
                ),
                on="thickness_rank",
                how="left",
            )
        if "thickness_nm" not in obs.columns:
            obs = obs.merge(
                pd.DataFrame(
                    {
                        "thickness_rank": list(thickness_map.keys()),
                        "thickness_nm": [float(v) for v in thickness_map.values()],
                    }
                ),
                on="thickness_rank",
                how="left",
            )

        merged = pred.merge(
            obs,
            on=["hkl_key", "thickness_rank"],
            how="inner",
            suffixes=("_pred", "_obs"),
        )
        merged["frame_number"] = target_frame
    else:
        pred = (
            pred.groupby(["frame_number", "thickness_rank", "hkl_key"], as_index=False)
            .agg(I_pred=("I_pred", "median"), S_comb=("S_comb", "max"), thickness_nm=("thickness_nm", "first"), h=("h", "first"), k=("k", "first"), l=("l", "first"))
        )
        obs = (
            obs.groupby(["frame_number", "thickness_rank", "hkl_key"], as_index=False)
            .agg(I_obs=("I_obs", "median"), sigma_obs=("sigma_obs", "median"), n_obs=("n_obs", "sum"), thickness_nm=("thickness_nm", "first"))
        )
        merged = pred.merge(
            obs,
            on=["frame_number", "hkl_key", "thickness_rank"],
            how="inner",
            suffixes=("_pred", "_obs"),
        )

    corr_table = _correlation_table(merged)

    prefix = f"{args.sample.lower()}"
    if args.frame_number is not None:
        prefix += f"_frame{int(args.frame_number):04d}"

    merged_out = out_dir / f"{prefix}_pred_vs_obs_long.csv"
    corr_out = out_dir / f"{prefix}_reflection_correlations.csv"
    summary_out = out_dir / f"{prefix}_summary.txt"

    merged.to_csv(merged_out, index=False)
    corr_table.to_csv(corr_out, index=False)

    n_pairs = int(merged.shape[0])
    n_reflections = int(corr_table.shape[0])
    med_rho = float(corr_table["spearman_rho"].median()) if n_reflections else float("nan")
    med_r = float(corr_table["pearson_r"].median()) if n_reflections else float("nan")

    summary_lines = [
        f"sample={args.sample}",
        f"frame_number={args.frame_number if args.frame_number is not None else 'all'}",
        f"frame_match_mode={args.frame_match_mode}",
        f"hkl_match_mode={args.hkl_match_mode}",
        f"phi_tolerance_deg={args.phi_tolerance_deg}",
        f"thickness_ranks={thickness_ranks}",
        f"thickness_map_nm={thickness_map}",
        f"merged_points={n_pairs}",
        f"reflections_with_>=3_thickness_points={n_reflections}",
        f"median_spearman_rho={med_rho:.4f}",
        f"median_pearson_r={med_r:.4f}",
        f"merged_table={merged_out}",
        f"correlation_table={corr_out}",
    ]
    summary_out.write_text("\n".join(summary_lines) + "\n")

    _save_summary_plots(
        merged=merged,
        corr_table=corr_table,
        out_dir=out_dir,
        sample=args.sample,
        frame_number=args.frame_number,
    )

    print("\n".join(summary_lines))
    print(f"summary_file={summary_out}")


if __name__ == "__main__":
    main()
