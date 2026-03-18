"""Plot experimental intensities for top varying predicted reflections."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_compare_tables(pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(pattern))
    if not files:
        raise RuntimeError(f"No compare files matched pattern: {pattern}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        required = {"h", "k", "l", "thickness_nm", "I_obs"}
        if not required.issubset(df.columns):
            continue
        df["source_file"] = str(f)
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid compare files with observed intensity columns were found.")

    out = pd.concat(rows, ignore_index=True)
    out["hkl"] = list(zip(out["h"].astype(int), out["k"].astype(int), out["l"].astype(int)))
    return out


def _load_top_hkls(ranked_csv: str, top_n: int) -> pd.DataFrame:
    ranked = pd.read_csv(ranked_csv)
    keep = ranked[["h", "k", "l"]].head(top_n).copy()
    keep["h"] = keep["h"].astype(int)
    keep["k"] = keep["k"].astype(int)
    keep["l"] = keep["l"].astype(int)
    return keep


def _plot_observed_curves(obs_df: pd.DataFrame, top_hkls: pd.DataFrame, output_png: Path) -> pd.DataFrame:
    top_hkls = top_hkls.copy()
    top_hkls["hkl"] = list(zip(top_hkls["h"], top_hkls["k"], top_hkls["l"]))

    cols = 3
    n = len(top_hkls)
    rows = int(np.ceil(max(n, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    coverage_rows = []

    for ax in axes:
        ax.grid(alpha=0.25)

    for i, rec in enumerate(top_hkls.itertuples(index=False)):
        ax = axes[i]
        h, k, l = int(rec.h), int(rec.k), int(rec.l)
        g = obs_df[(obs_df["h"] == h) & (obs_df["k"] == k) & (obs_df["l"] == l)].copy()
        g = g.sort_values("thickness_nm")

        if g.empty:
            ax.set_title(f"({h} {k} {l})\nno observed match", fontsize=9)
            coverage_rows.append({"h": h, "k": k, "l": l, "n_points": 0, "thickness_list_nm": ""})
            continue

        ax.plot(g["thickness_nm"], g["I_obs"], marker="o", linewidth=2.0, color="#1f77b4")
        if "sigma_obs" in g.columns:
            ax.fill_between(
                g["thickness_nm"],
                np.maximum(g["I_obs"] - g["sigma_obs"], 0.0),
                g["I_obs"] + g["sigma_obs"],
                color="#1f77b4",
                alpha=0.15,
                linewidth=0,
            )

        t_list = ",".join(str(int(x)) for x in sorted(g["thickness_nm"].unique()))
        ax.set_title(f"({h} {k} {l})\npoints={len(g)}, t=[{t_list}]", fontsize=9)
        coverage_rows.append({"h": h, "k": k, "l": l, "n_points": int(len(g)), "thickness_list_nm": t_list})

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Thickness (nm)")
    fig.supylabel("Observed intensity (I_obs)")
    fig.suptitle("Experimental intensities for top thickness-varying reflections", y=1.01)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame.from_records(coverage_rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--compare-pattern",
        default="analysis_output/lta_frame0050_compare_t*_*.csv",
        help="Glob pattern for compare tables",
    )
    p.add_argument(
        "--ranked-csv",
        default="analysis_output/lta_frame0050_top_varying_ranked.csv",
        help="Ranked reflections CSV from prediction-based variation ranking",
    )
    p.add_argument("--top-n", type=int, default=12, help="How many top HKLs to plot")
    p.add_argument(
        "--output-prefix",
        default="analysis_output/lta_frame0050_top_varying_observed",
        help="Prefix for output files",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    obs = _load_compare_tables(args.compare_pattern)
    top_hkls = _load_top_hkls(args.ranked_csv, int(args.top_n))

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = prefix.with_name(prefix.name + "_curves.png")
    out_cov = prefix.with_name(prefix.name + "_coverage.csv")

    coverage = _plot_observed_curves(obs, top_hkls, out_png)
    coverage.to_csv(out_cov, index=False)

    print(f"obs_rows={len(obs)}")
    print(f"top_hkls={len(top_hkls)}")
    print(f"plot_png={out_png}")
    print(f"coverage_csv={out_cov}")


if __name__ == "__main__":
    main()
