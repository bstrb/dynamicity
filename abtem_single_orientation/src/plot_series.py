from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _display_image(arr: np.ndarray, log_floor_percentile: float = 1.0, vmax_percentile: float = 99.9) -> np.ndarray:
    """Convert raw intensities to a high-contrast display image.

    Diffraction intensities often span many orders of magnitude. Using log10 with
    robust percentile clipping makes weaker reflections visible.
    """
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=float)

    vals = arr[finite]
    floor = float(np.percentile(vals, log_floor_percentile))
    floor = max(floor, 1e-20)

    log_img = np.log10(np.maximum(arr, floor))
    vmin = float(np.percentile(log_img[finite], 0.5))
    vmax = float(np.percentile(log_img[finite], vmax_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return np.zeros_like(arr, dtype=float)

    img = (log_img - vmin) / (vmax - vmin)
    return np.clip(img, 0.0, 1.0)


def _resample_square(img: np.ndarray, size: int = 320) -> np.ndarray:
    """Resample a 2D image to a square grid for presentation.

    This is a visualization-only transform (not used in simulation outputs).
    """
    h, w = img.shape
    if h == size and w == size:
        return img

    # Resample rows to target width.
    x_src = np.linspace(0.0, 1.0, w)
    x_tgt = np.linspace(0.0, 1.0, size)
    row_resampled = np.vstack([np.interp(x_tgt, x_src, img[r, :]) for r in range(h)])

    # Resample columns to target height.
    y_src = np.linspace(0.0, 1.0, h)
    y_tgt = np.linspace(0.0, 1.0, size)
    sq = np.vstack([np.interp(y_tgt, y_src, row_resampled[:, c]) for c in range(size)]).T
    return sq


def plot_overview(
    index_csv: str | Path,
    output_png: str | Path | None = None,
    display_mode: str = "square_resampled",
    square_size: int = 320,
) -> Path:
    index_csv = Path(index_csv)
    df = pd.read_csv(index_csv).sort_values("thickness_nm").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No patterns found in index CSV.")

    n = len(df)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 3.0 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    for i, rec in enumerate(df.itertuples(index=False)):
        arr = np.load(rec.npy_path)
        img = _display_image(arr)
        if display_mode == "square_resampled":
            img = _resample_square(img, size=square_size)
        elif display_mode == "native":
            pass
        else:
            raise ValueError("display_mode must be 'square_resampled' or 'native'")
        ax = axes[i]
        ax.imshow(img, cmap="magma", origin="lower", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"t={rec.thickness_nm:.0f} nm  ({arr.shape[0]}x{arr.shape[1]})")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if display_mode == "square_resampled":
        title = "abTEM diffraction patterns (square presentation, robust log contrast)"
    else:
        title = "abTEM diffraction patterns (native aspect, robust log contrast)"
    fig.suptitle(title, y=1.01)
    fig.tight_layout()

    if output_png is None:
        output_png = index_csv.parent / "figures" / "patterns_overview.png"
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_png
