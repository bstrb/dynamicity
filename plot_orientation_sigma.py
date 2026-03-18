import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def make_reliability_colormap():
    """
    Custom red -> yellow -> green map.
    Low I/sigma_eff = red
    High I/sigma_eff = green
    """
    return LinearSegmentedColormap.from_list(
        "dyn_reliability",
        [
            (0.00, "#c62828"),  # red
            (0.50, "#f9a825"),  # yellow
            (1.00, "#00a84f"),  # vivid green
        ],
    )


def gaussian_peak(X, Y, x0, y0, amplitude, sigma_xy):
    """
    2D Gaussian peak centered at (x0, y0).
    """
    return amplitude * np.exp(
        -((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma_xy ** 2)
    )


def map_reliability_to_widths(
    reliabilities,
    sigma_vals,
    width_scale=1.0,
    min_width=0.6,
    max_width=6.0,
    curve=0.85,
):
    """
    Map reliability (I/sigma_eff) to visible Gaussian widths.

    Lower reliability -> wider peaks.
    The mapping uses log-reliability percentiles for robust contrast and
    blends a mild sigma_eff term so peaks with larger sigma_eff stay broader.
    """
    rel = np.asarray(reliabilities, dtype=float)
    sig = np.asarray(sigma_vals, dtype=float)

    rel_safe = np.clip(rel, 1e-12, None)
    log_rel = np.log(rel_safe)
    lo, hi = np.percentile(log_rel, [5, 95])

    if np.isclose(lo, hi):
        rel_norm = np.full_like(rel_safe, 0.5)
    else:
        rel_norm = np.clip((log_rel - lo) / (hi - lo), 0.0, 1.0)

    # Invert so low reliability produces wide peaks.
    width_norm = (1.0 - rel_norm) ** curve
    base_width = min_width + width_norm * (max_width - min_width)

    # Keep a small dependence on sigma_eff to preserve uncertainty context.
    sig_ratio = sig / max(np.median(sig), 1e-12)
    sig_factor = np.clip(sig_ratio ** 0.35, 0.75, 1.30)

    widths = np.clip(base_width * sig_factor * width_scale, min_width, max_width)
    return widths


def build_surface_from_peaks(
    peaks,
    grid_resolution=220,
    padding=8.0,
    width_scale=1.0,
    min_width=0.6,
    max_width=6.0,
):
    """
    Build one smooth 3D surface from all peaks.

    Parameters
    ----------
    peaks : list of dict
        Each dict must contain:
            x, y, intensity, sigma_eff
    grid_resolution : int
        Number of grid points in x and y.
    padding : float
        Extra margin around extrema.
    width_scale : float
        Scaling factor converting sigma_eff to visible peak width.
    min_width, max_width : float
        Clamp visible peak widths so the plot remains readable.
    """
    x_vals = np.array([p["x"] for p in peaks], dtype=float)
    y_vals = np.array([p["y"] for p in peaks], dtype=float)

    x_min = np.min(x_vals) - padding
    x_max = np.max(x_vals) + padding
    y_min = np.min(y_vals) - padding
    y_max = np.max(y_vals) + padding

    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X, dtype=float)

    # Prevent undersampling artifacts: peaks narrower than the grid spacing
    # look needle-like or can disappear visually.
    dx = float(x_grid[1] - x_grid[0]) if grid_resolution > 1 else 1.0
    dy = float(y_grid[1] - y_grid[0]) if grid_resolution > 1 else 1.0
    grid_min_width = 1.8 * max(dx, dy)
    effective_min_width = max(min_width, grid_min_width)
    effective_max_width = max(max_width, effective_min_width + 1e-9)

    x_peaks = np.array([float(p["x"]) for p in peaks], dtype=float)
    y_peaks = np.array([float(p["y"]) for p in peaks], dtype=float)
    intensities = np.array([float(p["intensity"]) for p in peaks], dtype=float)
    sigma_vals = np.array([float(p["sigma_eff"]) for p in peaks], dtype=float)

    reliabilities = intensities / np.maximum(sigma_vals, 1e-9)
    widths = map_reliability_to_widths(
        reliabilities,
        sigma_vals,
        width_scale=width_scale,
        min_width=effective_min_width,
        max_width=effective_max_width,
    )

    contributions = np.zeros((len(peaks),) + X.shape, dtype=float)
    for i, (x0, y0, amp, sigma_xy) in enumerate(zip(x_peaks, y_peaks, intensities, widths)):
        contributions[i] = gaussian_peak(X, Y, x0, y0, amp, sigma_xy)

    Z = np.sum(contributions, axis=0)
    return X, Y, Z, reliabilities, widths, contributions


def assign_peak_colors(peaks, reliabilities, cmap, norm):
    """
    Convert each peak reliability into an RGBA color.
    """
    colors = []
    for rel in reliabilities:
        colors.append(cmap(norm(rel)))
    return colors


def plot_diffraction_peaks_3d(
    peaks,
    grid_resolution=220,
    padding=8.0,
    width_scale=1.0,
    min_width=0.6,
    max_width=6.0,
    detector_alpha=0.10,
    surface_alpha=0.95,
    add_floor_contours=True,
    peak_cut_fraction=0.004,
    elev=32,
    azim=-58,
    title="3D Reflection Reliability Map",
    save_path=None,
    dpi=300,
):
    """
    Plot a journal-style 3D diffraction peak landscape.

    Parameters
    ----------
    peaks : list of dict
        Required keys in each dict:
            x, y, intensity, sigma_eff
        Optional key:
            label
    """
    if len(peaks) == 0:
        raise ValueError("The peaks list is empty.")

    for i, p in enumerate(peaks):
        for key in ("x", "y", "intensity", "sigma_eff"):
            if key not in p:
                raise ValueError(f"Peak {i} is missing required key '{key}'.")

    cmap = make_reliability_colormap()

    X, Y, Z, reliabilities, widths, contributions = build_surface_from_peaks(
        peaks=peaks,
        grid_resolution=grid_resolution,
        padding=padding,
        width_scale=width_scale,
        min_width=min_width,
        max_width=max_width,
    )

    rel_min = np.min(reliabilities)
    rel_max = np.max(reliabilities)

    # Avoid zero-width normalization range
    if np.isclose(rel_min, rel_max):
        rel_min = rel_min * 0.95
        rel_max = rel_max * 1.05 + 1e-9

    norm = Normalize(vmin=rel_min, vmax=rel_max)

    peak_colors = np.asarray(assign_peak_colors(peaks, reliabilities, cmap, norm))

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f4f4f1")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f7f7f5")

    # Detector plane
    detector_plane = np.zeros_like(Z)
    ax.plot_surface(
        X,
        Y,
        detector_plane,
        color="#d7d7d4",
        alpha=detector_alpha,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    # Draw each reflection as its own 3D Gaussian bell.
    # This preserves the "individual peak" look while keeping width and color
    # linked to reliability.
    for i, p in enumerate(peaks):
        Zi = contributions[i]
        amp = max(float(p["intensity"]), 1e-9)
        mask = Zi >= amp * peak_cut_fraction

        if not np.any(mask):
            continue

        Xi = np.where(mask, X, np.nan)
        Yi = np.where(mask, Y, np.nan)
        Zi_masked = np.where(mask, Zi, np.nan)

        face = np.zeros(Z.shape + (4,), dtype=float)
        face[..., :3] = peak_colors[i][:3]

        # Slightly higher alpha near peak tops gives nicer bell definition.
        alpha_local = np.clip((Zi / amp) ** 0.45, 0.15, 1.0) * surface_alpha
        face[..., 3] = np.where(mask, alpha_local, 0.0)

        ax.plot_surface(
            Xi,
            Yi,
            Zi_masked,
            facecolors=face,
            linewidth=0,
            antialiased=True,
            shade=True,
            rcount=160,
            ccount=160,
        )

    if add_floor_contours:
        ax.contour(
            X,
            Y,
            np.sum(contributions, axis=0),
            zdir="z",
            offset=0,
            levels=10,
            cmap="Greys",
            linewidths=0.65,
            alpha=0.32,
        )

    z_vals = np.array([p["intensity"] for p in peaks], dtype=float)

    # Axis labels
    ax.set_xlabel("Detector x", labelpad=12)
    ax.set_ylabel("Detector y", labelpad=12)
    ax.set_zlabel("Intensity", labelpad=12)
    ax.set_title(title, pad=18)

    # Clean up panes and grid for a more publication-like look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_box_aspect((1.0, 1.0, 0.42))

    # Better view
    ax.view_init(elev=elev, azim=azim)

    # Set limits
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_zlim(0, np.max(z_vals) * 1.20)

    # Color bar for reliability
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("Reflection reliability  (I / σ_eff)", rotation=90, labelpad=12)
    cbar.outline.set_linewidth(0.6)

    # Optional text legend inside figure
    fig.text(
        0.02,
        0.02,
        "Green: more kinematical / reliable    Yellow: intermediate    Red: more dynamical / uncertain",
        fontsize=10,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Replace this example block with your own peak list
    # x, y          = detector coordinates
    # intensity     = peak height
    # sigma_eff     = effective uncertainty controlling visible peak width
    # label         = optional annotation label
    # -------------------------------------------------------------------------
    peaks = [
    {"x": 610, "y": 613, "intensity": 1.0, "sigma_eff": 1.60, "label": "r01"},
    {"x": 653, "y": 656, "intensity": 1.0, "sigma_eff": 1.60, "label": "r02"},
    {"x": 677, "y": 600, "intensity": 1.0, "sigma_eff": 1.60, "label": "r03"},
    {"x": 635, "y": 556, "intensity": 1.0, "sigma_eff": 1.60, "label": "r04"},
    {"x": 586, "y": 670, "intensity": 1.0, "sigma_eff": 1.60, "label": "r05"},
    {"x": 568, "y": 569, "intensity": 1.0, "sigma_eff": 1.60, "label": "r06"},
    {"x": 543, "y": 626, "intensity": 1.0, "sigma_eff": 1.60, "label": "r07"},
    {"x": 630, "y": 713, "intensity": 1.0, "sigma_eff": 1.60, "label": "r08"},
    {"x": 720, "y": 643, "intensity": 1.0, "sigma_eff": 1.60, "label": "r09"},
    {"x": 591, "y": 513, "intensity": 1.0, "sigma_eff": 1.60, "label": "r10"},
    {"x": 701, "y": 543, "intensity": 1.0, "sigma_eff": 1.60, "label": "r11"},
    {"x": 696, "y": 700, "intensity": 1.0, "sigma_eff": 1.60, "label": "r12"},
    {"x": 658, "y": 500, "intensity": 1.0, "sigma_eff": 1.60, "label": "r13"},
    {"x": 520, "y": 683, "intensity": 1.0, "sigma_eff": 1.60, "label": "r14"},
    {"x": 743, "y": 587, "intensity": 1.0, "sigma_eff": 1.60, "label": "r15"},
    {"x": 563, "y": 726, "intensity": 1.0, "sigma_eff": 1.60, "label": "r16"},
    {"x": 501, "y": 583, "intensity": 1.0, "sigma_eff": 1.60, "label": "r17"},
    {"x": 525, "y": 526, "intensity": 1.0, "sigma_eff": 1.60, "label": "r18"},
    {"x": 477, "y": 639, "intensity": 1.0, "sigma_eff": 1.60, "label": "r19"},

    {"x": 155, "y": 235, "intensity": 1, "sigma_eff": 0.50, "label": "g03"},
    {"x": 39, "y": 519, "intensity": 1, "sigma_eff": 0.50, "label": "g04"},
    {"x": 74, "y": 876, "intensity": 1, "sigma_eff": 0.50, "label": "g05"},
    {"x": 443, "y": 1165, "intensity": 1, "sigma_eff": 0.50, "label": "g06"},
    {"x": 819, "y": 1143, "intensity": 1, "sigma_eff": 0.50, "label": "g07"},
    {"x": 1207, "y": 651, "intensity": 1, "sigma_eff": 0.50, "label": "g08"},
    {"x": 889, "y": 90, "intensity": 1, "sigma_eff": 0.50, "label": "g09"},
]

    plot_diffraction_peaks_3d(
        peaks=peaks,
        grid_resolution=240,
        padding=10.0,
        width_scale=1.0,
        min_width=0.9,
        max_width=7.0,
        detector_alpha=0.10,
        surface_alpha=0.95,
        add_floor_contours=True,
        peak_cut_fraction=0.015,
        elev=30,
        azim=-60,
        title="Orientation-dependent effective reflection uncertainty",
        save_path="reflection_reliability_3d.png",
        dpi=300,
    )