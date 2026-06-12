#!/usr/bin/env python3
"""Plot selected HKLs using a derived non-self risk score.

This script reads existing joined component observations and derives:

    nonself_mean = mean(
        graph_crowding_norm,
        same_laue_zone_crowding_norm,
        systematic_row_risk_norm,
        frame_axis_risk_norm,
    )

It then generates, per selected HKL:
1) I_pr histogram split by low/high nonself_mean
2) nonself_mean vs residual scatter (residual = I_pr - median_I_pr)
3) binned median trend line on the scatter

No partialator or OriDyn reruns are performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NONSELF_COMPONENT_COLUMNS = [
    "graph_crowding_norm",
    "same_laue_zone_crowding_norm",
    "systematic_row_risk_norm",
    "frame_axis_risk_norm",
]

DEFAULT_EXPLICIT_HKLS = [
    (3, 3, -6),
    (16, 0, 0),
    (-1, 1, 8),
    (0, -12, 0),
    (0, 0, -6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected HKLs from joined_component_observations.csv using "
            "derived nonself_mean score."
        )
    )
    parser.add_argument(
        "--joined-component-observations",
        required=True,
        type=Path,
        help="Path to sdyn_component_diagnostics/joined_component_observations.csv",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output directory (e.g., $OUT/nonself_mean_hkl_figures)",
    )
    parser.add_argument(
        "--explicit-hkls",
        type=Path,
        default=None,
        help="Optional text file with one HKL per line: h k l",
    )
    parser.add_argument(
        "--hkl",
        nargs=3,
        type=int,
        action="append",
        default=None,
        metavar=("H", "K", "L"),
        help="Append explicit HKL triplet; can be provided multiple times",
    )
    parser.add_argument(
        "--use-default-hkls",
        dest="use_default_hkls",
        action="store_true",
        default=True,
        help="Include default HKLs first (default: true)",
    )
    parser.add_argument(
        "--no-default-hkls",
        dest="use_default_hkls",
        action="store_false",
        help="Do not include built-in default HKLs",
    )
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--hist-bins", type=int, default=45)
    parser.add_argument("--trend-bins", type=int, default=10)
    parser.add_argument("--example-max-hkls", type=int, default=4)

    args = parser.parse_args()

    if not (0.0 <= args.low_quantile < args.high_quantile <= 1.0):
        raise SystemExit("Expected 0 <= --low-quantile < --high-quantile <= 1")
    if args.hist_bins < 5:
        raise SystemExit("--hist-bins must be >= 5")
    if args.trend_bins < 3:
        raise SystemExit("--trend-bins must be >= 3")
    if args.example_max_hkls <= 0:
        raise SystemExit("--example-max-hkls must be > 0")

    return args


def parse_hkl_line(line: str) -> tuple[int, int, int] | None:
    text = line.split("#", 1)[0].strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def load_hkls_file(path: Path | None) -> list[tuple[int, int, int]]:
    if path is None:
        return []
    if not path.exists():
        raise SystemExit(f"--explicit-hkls not found: {path}")

    hkls: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_hkl_line(line)
        if parsed is not None:
            hkls.append(parsed)
    return hkls


def dedupe_hkls_preserve_order(items: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def resolve_target_hkls(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    hkls: list[tuple[int, int, int]] = []
    if bool(args.use_default_hkls):
        hkls.extend(DEFAULT_EXPLICIT_HKLS)

    hkls.extend(load_hkls_file(args.explicit_hkls))
    if args.hkl:
        hkls.extend((int(h), int(k), int(l)) for h, k, l in args.hkl)

    hkls = dedupe_hkls_preserve_order(hkls)
    if not hkls:
        raise SystemExit("No explicit HKLs were provided")
    return hkls


def hkl_slug(h: int, k: int, l: int) -> str:
    return f"h{h:+d}_k{k:+d}_l{l:+d}".replace("+", "p").replace("-", "m")


def hkl_text(h: int, k: int, l: int) -> str:
    return f"({h},{k},{l})"


def maybe_spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].corr(frame["y"], method="spearman"))


def binned_median_trend(x: pd.Series, y: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < max(6, bins):
        return np.array([]), np.array([])

    try:
        bucket = pd.qcut(frame["x"], q=int(bins), duplicates="drop")
    except Exception:
        return np.array([]), np.array([])

    grouped = frame.groupby(bucket, observed=True)
    out = grouped.agg(x_mid=("x", "median"), y_mid=("y", "median"), n=("y", "size")).reset_index(drop=True)
    out = out.loc[out["n"] >= 3].copy()
    if out.empty:
        return np.array([]), np.array([])
    return out["x_mid"].to_numpy(dtype=float), out["y_mid"].to_numpy(dtype=float)


def ensure_output_layout(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "plots": root / "plots",
        "per_hkl": root / "plots" / "per_hkl",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def required_columns_missing(df: pd.DataFrame) -> list[str]:
    required = ["h", "k", "l", "I_pr", *NONSELF_COMPONENT_COLUMNS]
    return [c for c in required if c not in df.columns]


def coerce_joined_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["h"] = pd.to_numeric(out["h"], errors="coerce")
    out["k"] = pd.to_numeric(out["k"], errors="coerce")
    out["l"] = pd.to_numeric(out["l"], errors="coerce")
    out["I_pr"] = pd.to_numeric(out["I_pr"], errors="coerce")
    for c in NONSELF_COMPONENT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["h", "k", "l"]).copy()
    out[["h", "k", "l"]] = out[["h", "k", "l"]].astype("int64")

    if "median_I_pr" in out.columns:
        out["median_I_pr"] = pd.to_numeric(out["median_I_pr"], errors="coerce")
    else:
        out["median_I_pr"] = np.nan

    group_median = out.groupby(["h", "k", "l"])["I_pr"].transform("median")
    out["median_I_pr"] = out["median_I_pr"].fillna(group_median)

    out["nonself_mean"] = out[NONSELF_COMPONENT_COLUMNS].mean(axis=1, skipna=True)
    out["residual"] = out["I_pr"] - out["median_I_pr"]
    out["hkl"] = "(" + out["h"].astype(str) + "," + out["k"].astype(str) + "," + out["l"].astype(str) + ")"
    return out


def plot_single_hkl(
    hkl_obs: pd.DataFrame,
    h: int,
    k: int,
    l: int,
    q_low: float,
    q_high: float,
    hist_bins: int,
    trend_bins: int,
    out_path: Path,
) -> tuple[bool, int]:
    if hkl_obs.empty:
        return False, 0

    low = hkl_obs.loc[hkl_obs["nonself_group"] == "low"]
    high = hkl_obs.loc[hkl_obs["nonself_group"] == "high"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=170)

    ax = axes[0]
    if not low.empty:
        ax.hist(low["I_pr"], bins=hist_bins, alpha=0.60, color="#2b8cbe", label=f"low (n={len(low)})")
    if not high.empty:
        ax.hist(high["I_pr"], bins=hist_bins, alpha=0.60, color="#de2d26", label=f"high (n={len(high)})")
    ax.set_title("I_pr histogram split by nonself_mean")
    ax.set_xlabel("I_pr")
    ax.set_ylabel("count")
    if ax.has_data():
        ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    x = pd.to_numeric(hkl_obs["nonself_mean"], errors="coerce")
    y = pd.to_numeric(hkl_obs["residual"], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.any():
        ax.scatter(x[valid], y[valid], s=13, alpha=0.70, color="#4c78a8", linewidths=0)

    bx, by = binned_median_trend(x, y, bins=int(trend_bins))
    if bx.size > 0:
        ax.plot(bx, by, color="#d62728", linewidth=2.0, marker="o", markersize=3.5, label="binned median trend")
        ax.legend(loc="best", fontsize=8)

    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_title("nonself_mean vs residual")
    ax.set_xlabel("nonself_mean")
    ax.set_ylabel("residual = I_pr - median_I_pr")

    rho = maybe_spearman(x, y)
    fig.suptitle(
        (
            f"HKL {hkl_text(h, k, l)} | n={len(hkl_obs)} | "
            f"q_low={q_low:.4f}, q_high={q_high:.4f} | rho={rho:.3f}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return True, int(bx.size)


def plot_example_panel(
    entries: list[dict[str, object]],
    out_path: Path,
    hist_bins: int,
    trend_bins: int,
) -> None:
    if not entries:
        fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=170)
        ax.axis("off")
        ax.text(0.5, 0.5, "No selected HKLs with plottable data", ha="center", va="center")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return

    nrows = len(entries)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, max(3.8 * nrows, 4.0)), dpi=170, squeeze=False)

    for row_idx, entry in enumerate(entries):
        h = int(entry["h"])
        k = int(entry["k"])
        l = int(entry["l"])
        g = entry["obs"]

        low = g.loc[g["nonself_group"] == "low"]
        high = g.loc[g["nonself_group"] == "high"]

        ax_hist = axes[row_idx, 0]
        if not low.empty:
            ax_hist.hist(low["I_pr"], bins=hist_bins, alpha=0.55, color="#2b8cbe", label=f"low n={len(low)}")
        if not high.empty:
            ax_hist.hist(high["I_pr"], bins=hist_bins, alpha=0.55, color="#de2d26", label=f"high n={len(high)}")
        ax_hist.set_title(f"HKL {hkl_text(h, k, l)} I_pr split")
        ax_hist.set_xlabel("I_pr")
        ax_hist.set_ylabel("count")
        if ax_hist.has_data():
            ax_hist.legend(loc="best", fontsize=7)

        ax_scatter = axes[row_idx, 1]
        x = pd.to_numeric(g["nonself_mean"], errors="coerce")
        y = pd.to_numeric(g["residual"], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.any():
            ax_scatter.scatter(x[valid], y[valid], s=12, alpha=0.7, color="#4c78a8", linewidths=0)
        bx, by = binned_median_trend(x, y, bins=int(trend_bins))
        if bx.size > 0:
            ax_scatter.plot(bx, by, color="#d62728", linewidth=1.9, marker="o", markersize=3.2)
        ax_scatter.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
        ax_scatter.set_title(f"HKL {hkl_text(h, k, l)} nonself_mean vs residual")
        ax_scatter.set_xlabel("nonself_mean")
        ax_scatter.set_ylabel("residual")

    fig.suptitle("nonself_mean selected-HKL example panel", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = ensure_output_layout(args.output_root)

    target_hkls = resolve_target_hkls(args)

    joined = pd.read_csv(args.joined_component_observations)
    missing = required_columns_missing(joined)
    if missing:
        raise SystemExit(
            "joined_component_observations.csv missing required column(s): "
            f"{missing}"
        )

    data = coerce_joined_columns(joined)

    targets_set = set(target_hkls)
    selected = data.loc[
        data[["h", "k", "l"]].apply(lambda r: (int(r["h"]), int(r["k"]), int(r["l"])) in targets_set, axis=1)
    ].copy()

    selected["target_requested"] = selected[["h", "k", "l"]].apply(
        lambda r: (int(r["h"]), int(r["k"]), int(r["l"])) in targets_set,
        axis=1,
    )

    summary_rows: list[dict[str, object]] = []
    obs_rows: list[pd.DataFrame] = []
    example_entries: list[dict[str, object]] = []
    per_hkl_plot_count = 0

    for order, (h, k, l) in enumerate(target_hkls, start=1):
        g = selected.loc[(selected["h"] == h) & (selected["k"] == k) & (selected["l"] == l)].copy()

        if g.empty:
            summary_rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "hkl": hkl_text(h, k, l),
                    "hkl_order": order,
                    "status": "missing_hkl",
                    "n_obs_total": 0,
                    "n_obs_valid": 0,
                    "nonself_q_low": np.nan,
                    "nonself_q_high": np.nan,
                    "n_low": 0,
                    "n_high": 0,
                    "median_I_pr": np.nan,
                    "median_low_I_pr": np.nan,
                    "median_high_I_pr": np.nan,
                    "high_minus_low_median_shift": np.nan,
                    "spearman_nonself_vs_residual": np.nan,
                    "trend_points": 0,
                    "plot_file": "",
                }
            )
            continue

        g_valid = g.loc[g["I_pr"].notna() & g["residual"].notna() & g["nonself_mean"].notna()].copy()
        if g_valid.empty:
            g["nonself_group"] = "mid"
            obs_rows.append(g)
            summary_rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": l,
                    "hkl": hkl_text(h, k, l),
                    "hkl_order": order,
                    "status": "no_valid_rows",
                    "n_obs_total": int(len(g)),
                    "n_obs_valid": 0,
                    "nonself_q_low": np.nan,
                    "nonself_q_high": np.nan,
                    "n_low": 0,
                    "n_high": 0,
                    "median_I_pr": float(g["median_I_pr"].dropna().iloc[0]) if g["median_I_pr"].notna().any() else np.nan,
                    "median_low_I_pr": np.nan,
                    "median_high_I_pr": np.nan,
                    "high_minus_low_median_shift": np.nan,
                    "spearman_nonself_vs_residual": np.nan,
                    "trend_points": 0,
                    "plot_file": "",
                }
            )
            continue

        q_low = float(g_valid["nonself_mean"].quantile(float(args.low_quantile)))
        q_high = float(g_valid["nonself_mean"].quantile(float(args.high_quantile)))

        g["nonself_group"] = "mid"
        g.loc[g["nonself_mean"] <= q_low, "nonself_group"] = "low"
        g.loc[g["nonself_mean"] >= q_high, "nonself_group"] = "high"

        g_valid = g.loc[g["I_pr"].notna() & g["residual"].notna() & g["nonself_mean"].notna()].copy()
        low = g_valid.loc[g_valid["nonself_group"] == "low"]
        high = g_valid.loc[g_valid["nonself_group"] == "high"]

        plot_path = out["per_hkl"] / f"{hkl_slug(h, k, l)}.png"
        wrote_plot, trend_points = plot_single_hkl(
            hkl_obs=g_valid,
            h=h,
            k=k,
            l=l,
            q_low=q_low,
            q_high=q_high,
            hist_bins=int(args.hist_bins),
            trend_bins=int(args.trend_bins),
            out_path=plot_path,
        )
        if wrote_plot:
            per_hkl_plot_count += 1

        if wrote_plot and len(example_entries) < int(args.example_max_hkls):
            example_entries.append({"h": h, "k": k, "l": l, "obs": g_valid})

        obs_rows.append(g)

        med_low = float(low["I_pr"].median()) if not low.empty else np.nan
        med_high = float(high["I_pr"].median()) if not high.empty else np.nan

        summary_rows.append(
            {
                "h": h,
                "k": k,
                "l": l,
                "hkl": hkl_text(h, k, l),
                "hkl_order": order,
                "status": "ok",
                "n_obs_total": int(len(g)),
                "n_obs_valid": int(len(g_valid)),
                "nonself_q_low": q_low,
                "nonself_q_high": q_high,
                "n_low": int(len(low)),
                "n_high": int(len(high)),
                "median_I_pr": float(g_valid["median_I_pr"].iloc[0]) if g_valid["median_I_pr"].notna().any() else np.nan,
                "median_low_I_pr": med_low,
                "median_high_I_pr": med_high,
                "high_minus_low_median_shift": med_high - med_low if np.isfinite(med_high) and np.isfinite(med_low) else np.nan,
                "spearman_nonself_vs_residual": maybe_spearman(g_valid["nonself_mean"], g_valid["residual"]),
                "trend_points": int(trend_points),
                "plot_file": str(plot_path),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("hkl_order", ascending=True)
    selected_obs = pd.concat(obs_rows, ignore_index=True) if obs_rows else pd.DataFrame()

    summary_path = out["root"] / "nonself_selected_hkl_summary.csv"
    obs_path = out["root"] / "nonself_selected_hkl_observations.csv"
    panel_path = out["plots"] / "nonself_example_panel.png"

    summary.to_csv(summary_path, index=False)
    selected_obs.to_csv(obs_path, index=False)
    plot_example_panel(
        entries=example_entries,
        out_path=panel_path,
        hist_bins=int(args.hist_bins),
        trend_bins=int(args.trend_bins),
    )

    print("DONE")
    print(f"target_hkls={len(target_hkls):,}")
    print(f"selected_rows={len(selected):,}")
    print(f"per_hkl_plots={per_hkl_plot_count:,}")
    print(f"summary_csv={summary_path}")
    print(f"observations_csv={obs_path}")
    print(f"example_panel={panel_path}")


if __name__ == "__main__":
    main()
