#!/usr/bin/env python3
"""Plot v4 local-crowding raw diagnostic distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HKL_COLUMNS = ["h", "k", "l"]
DEFAULT_CHUNKSIZE = 500_000
DEFAULT_MAX_SAMPLE_ROWS = 1_000_000
SELECTED_HKLS = [(0, 4, 0), (0, 27, 5), (6, 6, 2)]
SCORE_COLUMNS = [
    "local_neighbor_sum_raw",
    "local_crowding_target_gated_raw",
    "local_crowding_target_gated_log1p",
    "target_excitation_Eg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path, help="geometry_coupling_v4_local_crowding_raw_scores.csv")
    parser.add_argument("--outdir", type=Path, default=None, help="Plot output directory; default: scores parent / plots")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-sample-rows", type=int, default=DEFAULT_MAX_SAMPLE_ROWS)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not args.scores.exists():
        raise SystemExit(f"--scores not found: {args.scores}")
    if int(args.chunksize) < 1:
        raise SystemExit("--chunksize must be >= 1")
    if int(args.max_sample_rows) < 1:
        raise SystemExit("--max-sample-rows must be >= 1")
    if args.outdir is None:
        args.outdir = args.scores.parent / "plots"
    return args


def require_columns(header: list[str], columns: list[str]) -> None:
    missing = [column for column in columns if column not in header]
    if missing:
        raise SystemExit(f"Input score CSV is missing required column(s): {missing}")


def load_sample(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(args.scores, nrows=0).columns.tolist()
    usecols = [column for column in [*HKL_COLUMNS, "inv_nm", *SCORE_COLUMNS] if column in header]
    require_columns(header, [*HKL_COLUMNS, "local_crowding_target_gated_log1p", "target_excitation_Eg", "local_neighbor_sum_raw"])
    rng = np.random.default_rng(int(args.seed))
    chunks = []
    rows_read = 0
    rows_kept = 0
    for chunk_index, chunk in enumerate(pd.read_csv(args.scores, usecols=usecols, chunksize=int(args.chunksize)), start=1):
        rows_read += int(len(chunk))
        remaining = int(args.max_sample_rows) - rows_kept
        if remaining <= 0:
            continue
        if len(chunk) <= remaining:
            sub = chunk.copy()
        else:
            take = rng.choice(len(chunk), size=remaining, replace=False)
            sub = chunk.iloc[np.sort(take)].copy()
        chunks.append(sub)
        rows_kept += int(len(sub))
        if chunk_index == 1 or chunk_index % 5 == 0:
            print(f"Plot sample pass: chunks={chunk_index:,}, rows_read={rows_read:,}, sampled={rows_kept:,}", flush=True)
    sample = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    for column in ["inv_nm", *SCORE_COLUMNS]:
        if column in sample:
            sample[column] = pd.to_numeric(sample[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return sample, {"rows_read": int(rows_read), "rows_sampled": int(len(sample)), "sample_note": "bounded chunk sample for plotting"}


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def save_hist(sample: pd.DataFrame, column: str, path: Path, bins: int = 100) -> None:
    values = finite(sample[column])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values.to_numpy(dtype=float), bins=bins, color="#3b6ea8", alpha=0.85)
    ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hexbin(sample: pd.DataFrame, x: str, y: str, path: Path) -> None:
    work = sample[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(work[x].to_numpy(dtype=float), work[y].to_numpy(dtype=float), gridsize=80, bins="log", mincnt=1, cmap="viridis")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    fig.colorbar(hb, ax=ax, label="log10(count)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hkl_spread(sample: pd.DataFrame, path: Path) -> pd.DataFrame:
    grouped = sample.dropna(subset=["local_crowding_target_gated_log1p"]).groupby(HKL_COLUMNS)[
        "local_crowding_target_gated_log1p"
    ]
    summary = grouped.quantile([0.10, 0.90]).unstack().rename(columns={0.10: "q10", 0.90: "q90"}).reset_index()
    summary["spread_q90_q10"] = summary["q90"] - summary["q10"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(summary["spread_q90_q10"].dropna().to_numpy(dtype=float), bins=100, color="#7a4e9d", alpha=0.85)
    ax.set_title("Per-signed-HKL spread: q90 - q10")
    ax.set_xlabel("q90 - q10 of local_crowding_target_gated_log1p")
    ax.set_ylabel("signed HKLs")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return summary


def save_resolution_summary(sample: pd.DataFrame, path: Path) -> None:
    if "inv_nm" not in sample:
        return
    work = sample.dropna(subset=["inv_nm", "local_crowding_target_gated_log1p"]).copy()
    if work.empty:
        return
    bins = np.linspace(float(work["inv_nm"].min()), float(work["inv_nm"].max()), 40)
    work["inv_nm_bin"] = pd.cut(work["inv_nm"], bins=bins, include_lowest=True)
    grouped = work.groupby("inv_nm_bin", observed=True)
    summary = grouped.agg(
        inv_nm_mid=("inv_nm", "median"),
        median_log1p=("local_crowding_target_gated_log1p", "median"),
        q90_log1p=("local_crowding_target_gated_log1p", lambda values: float(np.nanquantile(values, 0.90))),
        n=("local_crowding_target_gated_log1p", "size"),
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["inv_nm_mid"], summary["median_log1p"], marker="o", label="median")
    ax.plot(summary["inv_nm_mid"], summary["q90_log1p"], marker="o", label="q90")
    ax.set_xlabel("1/d (nm^-1)")
    ax.set_ylabel("local_crowding_target_gated_log1p")
    ax.set_title("Local crowding by resolution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_selected_hkl_panels(sample: pd.DataFrame, outdir: Path) -> list[Path]:
    paths = []
    for h, k, l in SELECTED_HKLS:
        mask = (sample["h"] == h) & (sample["k"] == k) & (sample["l"] == l)
        values = finite(sample.loc[mask, "local_crowding_target_gated_log1p"])
        path = outdir / f"selected_hkl_{h}_{k}_{l}_local_crowding_log1p_hist.png"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values.to_numpy(dtype=float), bins=50, color="#c55a11", alpha=0.85)
        ax.set_title(f"HKL {h} {k} {l}")
        ax.set_xlabel("local_crowding_target_gated_log1p")
        ax.set_ylabel("observations")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_summary(path: Path, sample_stats: dict[str, Any], plot_paths: list[Path], hkl_spread: pd.DataFrame) -> None:
    lines = [
        "# V4 Local Crowding Distribution Plots",
        "",
        f"Rows read: {sample_stats['rows_read']:,}",
        f"Rows sampled for plotting: {sample_stats['rows_sampled']:,}",
        f"Signed HKLs in spread sample: {len(hkl_spread):,}",
        "",
        "## Plots",
    ]
    lines.extend(f"- `{path}`" for path in plot_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sample, sample_stats = load_sample(args)
    plot_paths: list[Path] = []

    for column in SCORE_COLUMNS:
        if column in sample:
            path = args.outdir / f"hist_{column}.png"
            save_hist(sample, column, path)
            plot_paths.append(path)

    path = args.outdir / "hexbin_target_excitation_Eg_vs_local_neighbor_sum_raw.png"
    save_hexbin(sample, "target_excitation_Eg", "local_neighbor_sum_raw", path)
    plot_paths.append(path)

    path = args.outdir / "hexbin_target_excitation_Eg_vs_local_crowding_target_gated_log1p.png"
    save_hexbin(sample, "target_excitation_Eg", "local_crowding_target_gated_log1p", path)
    plot_paths.append(path)

    path = args.outdir / "hist_signed_hkl_spread_q90_q10_local_crowding_log1p.png"
    hkl_spread = save_hkl_spread(sample, path)
    plot_paths.append(path)

    path = args.outdir / "resolution_summary_local_crowding_log1p.png"
    save_resolution_summary(sample, path)
    if path.exists():
        plot_paths.append(path)

    plot_paths.extend(save_selected_hkl_panels(sample, args.outdir))
    summary_path = args.outdir / "plot_summary.md"
    write_summary(summary_path, sample_stats, plot_paths, hkl_spread)
    print("V4 local-crowding plots written")
    print(f"plots_dir: {args.outdir}")
    for path in plot_paths:
        print(f"plot: {path}")
    print(f"summary_md: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())