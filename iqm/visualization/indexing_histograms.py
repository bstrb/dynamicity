import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import tri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 ; side-effect import

def plot_indexing_rate(folder_path):
    """Generate 3-D surface + 2-D heat-map of indexing rate over (x, y)."""
    # ------------------------------------------------------------------ #
    # 1. Harvest the *.stream files
    # ------------------------------------------------------------------ #
    pattern = os.path.join(folder_path, "*.stream")
    stream_files = glob.glob(pattern)

    records = []
    for path in stream_files:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            x, y = map(float, base.rsplit("_", 2)[-2:])
        except ValueError:
            # skip filenames that don’t match the “…_<x>_<y>.stream” pattern
            continue

        hits = tots = 0
        with open(path, "r") as fh:
            for line in fh:
                if line.startswith("num_reflections"):
                    hits += 1
                elif line.startswith("num_peaks"):
                    tots += 1
        perc = 100 * hits / tots if tots else 0
        records.append((x, y, perc))

    if not records:
        raise RuntimeError("No valid *.stream files found")

    df = pd.DataFrame(records, columns=["x", "y", "rate"])

    # ------------------------------------------------------------------ #
    # 2. 3-D surface plot
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.25)

    ax3d = fig.add_subplot(gs[0], projection="3d")

    cmap = plt.cm.viridis
    norm = plt.Normalize(df.rate.min(), df.rate.max())

    # triangulate irregular (x, y) and plot surface
    tri_obj = tri.Triangulation(df.x, df.y)
    surf = ax3d.plot_trisurf(
        tri_obj,
        df.rate,
        cmap=cmap,
        linewidth=0.2,
        antialiased=True,
        edgecolor="none",
        norm=norm,
        alpha=0.95,
    )

    # pretty extras
    ax3d.set_xlabel("X coordinate shift (pixels)", labelpad=8)
    ax3d.set_ylabel("Y coordinate shift (pixels)", labelpad=8)
    ax3d.set_zlabel("Indexing rate (%)", labelpad=8)
    ax3d.set_title("Indexing rate surface", pad=12, fontsize=12)
    ax3d.view_init(elev=30, azim=135)
    ax3d.xaxis.pane.set_alpha(0.1)
    ax3d.yaxis.pane.set_alpha(0.1)
    ax3d.zaxis.pane.set_alpha(0.1)

    # shared color-bar
    cbar = fig.colorbar(
        surf, ax=ax3d, fraction=0.025, pad=0.08, shrink=0.9, aspect=15
    )
    cbar.set_label("Indexing rate (%)")

    # ------------------------------------------------------------------ #
    # 3. 2-D heat-map (top view)
    # ------------------------------------------------------------------ #
    ax2d = fig.add_subplot(gs[1])

    # create a dense grid for smooth shading
    xi = np.linspace(df.x.min(), df.x.max(), 200)
    yi = np.linspace(df.y.min(), df.y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # interpolate onto grid (nearest because SciPy may not be available)
    from scipy.interpolate import griddata  # if SciPy present; else fallback
    try:
        Zi = griddata(
            points=df[["x", "y"]].values,
            values=df["rate"].values,
            xi=(Xi, Yi),
            method="linear",
        )
    except Exception:
        # fallback: use matplotlib.tri interpolation
        interp = tri.LinearTriInterpolator(tri_obj, df.rate)
        Zi = interp(Xi, Yi)

    hm = ax2d.pcolormesh(
        Xi,
        Yi,
        Zi,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )
    ax2d.set_xlabel("X coordinate shift (pixels)")
    ax2d.set_ylabel("Y coordinate shift (pixels)")
    ax2d.set_title("Heat-map (top view)")

    fig.colorbar(hm, ax=ax2d, fraction=0.046, pad=0.04).set_label("Indexing rate (%)")

    plt.tight_layout()
    plt.show()