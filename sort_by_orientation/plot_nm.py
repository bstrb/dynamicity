#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_skip_hash(path: Path) -> pd.DataFrame:
    """Read CSV that may include metadata lines starting with '#'."""
    df = pd.read_csv(path, comment="#")
    required = {"N", "M", "Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    for c in ["N", "M", "Score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["N", "M", "Score"]).reset_index(drop=True)


def minmax_norm(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1]."""
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if xmax == xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def plot_normalized(order_idx, N, M, S, out_path: Path, title_prefix: str = ""):
    nN, nM, nS = minmax_norm(N), minmax_norm(M), minmax_norm(S)

    plt.figure(figsize=(12, 6), dpi=120)
    plt.plot(order_idx, nN, marker="o", linewidth=1.5, label="N (normalized)")
    plt.plot(order_idx, nM, marker="s", linewidth=1.5, label="M (normalized)")
    plt.plot(order_idx, nS, marker="^", linewidth=1.5, label="Score (normalized)")

    plt.xlabel("Row order (top → bottom)")
    plt.ylabel("Normalized value [0–1]")
    plt.title(f"{title_prefix}N, M, Score — normalized")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved normalized plot to: {out_path}")


def extract_title_prefix(csv_path: Path) -> str:
    """Grab the first metadata line for a title prefix (best effort)."""
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    return line.strip("# \n") + " — "
                else:
                    break
    except Exception:
        pass
    return ""


def main():
    p = argparse.ArgumentParser(description="Plot normalized N, M, and Score from CSV.")
    p.add_argument("csv", type=Path, help="Input CSV (supports '#' metadata lines).")
    p.add_argument("--out", type=Path, default=Path("nm_score_normalized.png"),
                   help="Output image file (default: nm_score_normalized.png).")
    p.add_argument("--start-index", type=int, default=1,
                   help="Starting x-index (default: 1).")
    args = p.parse_args()

    df = read_csv_skip_hash(args.csv)
    order_idx = np.arange(args.start_index, args.start_index + len(df))
    title_prefix = extract_title_prefix(args.csv)

    plot_normalized(order_idx, df["N"].values, df["M"].values, df["Score"].values,
                    args.out, title_prefix)


if __name__ == "__main__":
    main()
