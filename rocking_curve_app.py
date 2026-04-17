"""Interactive electron-diffraction rocking-curve explorer for Jupyter.

This module provides a small ipywidgets + matplotlib app for comparing
kinematical and two-beam dynamical rocking curves.

Notebook usage
--------------
from rocking_curve_app import display_rocking_curve_app

display_rocking_curve_app()
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display


# Short explanatory note shown inside the notebook UI.
PHYSICS_NOTE = (
    "<b>Physical note.</b> The dimensionless excitation variable is "
    "<code>w = \u03be s</code>, where <code>s</code> is the deviation parameter "
    "from the exact Bragg condition and <code>\u03be</code> is the extinction distance. "
    "As thickness <code>t</code> increases, the rocking curve develops more closely spaced "
    "thickness fringes (ripples), while the central peak becomes narrower in <code>w</code>."
)


def _validate_positive(value: float, name: str) -> None:
    """Raise a clear error if a required positive parameter is invalid."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")



def _parse_overlay_thicknesses(raw_text: str) -> list[float]:
    """Parse a comma-separated list of extra thicknesses for overlay plots."""
    if not raw_text.strip():
        return []

    values: list[float] = []
    for chunk in raw_text.split(","):
        item = chunk.strip()
        if not item:
            continue
        value = float(item)
        _validate_positive(value, "Overlay thickness")
        values.append(value)
    return values



def rocking_curve_kinematical(w: np.ndarray, t: float, xi: float) -> np.ndarray:
    """Kinematical rocking-curve intensity.

    I_kin(w, t, xi) = (pi*t/xi)^2 * sinc^2((t/xi)*w)

    NumPy's sinc is normalized as sin(pi x)/(pi x), which matches the formula
    written here in the common physics convention.
    """
    _validate_positive(t, "Thickness t")
    _validate_positive(xi, "Extinction distance xi")

    prefactor = (np.pi * t / xi) ** 2
    argument = (t / xi) * w
    return prefactor * np.sinc(argument) ** 2



def rocking_curve_dynamical(w: np.ndarray, t: float, xi: float) -> np.ndarray:
    """Two-beam dynamical rocking-curve intensity.

    I_dyn(w, t, xi) = (pi*t/xi)^2 * sinc^2((t/xi)*sqrt(1+w^2))
    """
    _validate_positive(t, "Thickness t")
    _validate_positive(xi, "Extinction distance xi")

    prefactor = (np.pi * t / xi) ** 2
    argument = (t / xi) * np.sqrt(1.0 + w**2)
    return prefactor * np.sinc(argument) ** 2



def _plot_curves(
    t: float,
    xi: float,
    w_min: float,
    w_max: float,
    n_points: int,
    show_kinematical: bool,
    show_dynamical: bool,
    overlay_t_values: str,
    overlay_enabled: bool,
) -> None:
    """Draw the interactive rocking-curve plot for the current widget state."""
    plt.close("all")
    _, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)

    try:
        _validate_positive(t, "Thickness t")
        _validate_positive(xi, "Extinction distance xi")

        if n_points < 50:
            raise ValueError("Number of plotted points must be at least 50.")
        if w_max <= w_min:
            raise ValueError("w_max must be greater than w_min.")

        thicknesses = [t]
        if overlay_enabled:
            extras = _parse_overlay_thicknesses(overlay_t_values)
            # Keep ordering stable while removing duplicate values.
            for extra_t in extras:
                if not any(np.isclose(extra_t, existing_t) for existing_t in thicknesses):
                    thicknesses.append(extra_t)

        w = np.linspace(w_min, w_max, n_points)
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(thicknesses)))

        for color, t_value in zip(colors, thicknesses):
            if show_kinematical:
                ax.plot(
                    w,
                    rocking_curve_kinematical(w, t_value, xi),
                    color=color,
                    linewidth=2.2,
                    linestyle="-",
                    label=fr"Kinematical, $t={t_value:g}$",
                )
            if show_dynamical:
                ax.plot(
                    w,
                    rocking_curve_dynamical(w, t_value, xi),
                    color=color,
                    linewidth=2.2,
                    linestyle="--",
                    label=fr"Two-beam dynamical, $t={t_value:g}$",
                )

    except ValueError as exc:
        ax.text(
            0.5,
            0.5,
            f"Input error:\n{exc}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75"},
        )
        ax.set_axis_off()
        plt.show()
        return

    if not show_kinematical and not show_dynamical:
        ax.text(
            0.5,
            0.5,
            "Enable at least one model to display curves.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.75"},
        )
    else:
        ax.legend(loc="best")

    ax.set_xlabel(r"Excitation variable $w = \xi s$", fontsize=12)
    ax.set_ylabel("Intensity (arbitrary units)", fontsize=12)
    ax.set_title("Electron-Diffraction Rocking Curves", fontsize=14)
    ax.grid(True, alpha=0.25)

    plt.show()



def build_rocking_curve_app() -> widgets.Widget:
    """Create the notebook widget layout without displaying it."""
    t_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=10.0,
        step=0.1,
        description="t",
        readout_format=".1f",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    xi_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=10.0,
        step=0.1,
        description="xi",
        readout_format=".1f",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    w_min_slider = widgets.FloatSlider(
        value=-6.0,
        min=-20.0,
        max=0.0,
        step=0.5,
        description="w min",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    w_max_slider = widgets.FloatSlider(
        value=6.0,
        min=0.0,
        max=20.0,
        step=0.5,
        description="w max",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    n_points_slider = widgets.IntSlider(
        value=1200,
        min=100,
        max=5000,
        step=100,
        description="points",
        continuous_update=False,
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    show_kinematical = widgets.Checkbox(value=True, description="Show kinematical")
    show_dynamical = widgets.Checkbox(value=True, description="Show dynamical")
    overlay_enabled = widgets.Checkbox(value=False, description="Overlay extra t values")

    overlay_t_values = widgets.Text(
        value="2, 4, 6",
        description="extra t",
        placeholder="e.g. 2, 4, 6",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )

    controls = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='margin:0 0 6px 0;'>Rocking-Curve Explorer</h3>"
                "<p style='margin:0;'>Compare kinematical and two-beam dynamical "
                "electron-diffraction rocking curves.</p>"
            ),
            widgets.HTML(
                value=(
                    "<div style='padding:10px 12px; background:#f6f8fa; border:1px solid #d0d7de; "
                    "border-radius:8px; line-height:1.4;'>"
                    f"{PHYSICS_NOTE}"
                    "</div>"
                )
            ),
            t_slider,
            xi_slider,
            widgets.HBox([w_min_slider, w_max_slider]),
            n_points_slider,
            widgets.HBox([show_kinematical, show_dynamical]),
            overlay_enabled,
            overlay_t_values,
        ],
        layout=widgets.Layout(gap="8px", width="100%"),
    )

    output = widgets.interactive_output(
        _plot_curves,
        {
            "t": t_slider,
            "xi": xi_slider,
            "w_min": w_min_slider,
            "w_max": w_max_slider,
            "n_points": n_points_slider,
            "show_kinematical": show_kinematical,
            "show_dynamical": show_dynamical,
            "overlay_t_values": overlay_t_values,
            "overlay_enabled": overlay_enabled,
        },
    )

    return widgets.VBox(
        [controls, output],
        layout=widgets.Layout(width="100%", max_width="980px", gap="10px"),
    )



def display_rocking_curve_app() -> widgets.Widget:
    """Display the rocking-curve app in a Jupyter notebook and return it."""
    app = build_rocking_curve_app()
    display(app)
    return app


if __name__ == "__main__":
    print(
        "This module is designed for Jupyter notebooks.\n\n"
        "Example:\n"
        "from rocking_curve_app import display_rocking_curve_app\n"
        "display_rocking_curve_app()"
    )
