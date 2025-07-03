"""
visualisation.py
================
Matplotlib-based 3-D canvas embedded in Qt.

If you need higher frame rates, use *visualisation_pg.py* instead.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from typing import Iterable, Tuple


class Visualization3D(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 6))
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([1, 1, 1])

    # ------------------------------------------------------------------
    def draw_scene(
        self,
        lattice_pts: Iterable[Tuple[int, int, int, np.ndarray]],
        bragg_pts: Iterable[Tuple[int, int, int, np.ndarray]],
        wavelength: float,
    ):
        self.ax.clear()
        r = 1.0 / wavelength

        # Ewald sphere
        u, v = np.mgrid[0:np.pi:60j, 0:2 * np.pi:60j]
        xs = r * np.sin(u) * np.cos(v)
        ys = r * np.sin(u) * np.sin(v)
        zs = r * np.cos(u)
        self.ax.plot_surface(xs, ys, zs, alpha=0.15, linewidth=0)

        # Reciprocal-lattice points
        all_xyz = np.array([vec for *_, vec in lattice_pts])
        self.ax.scatter(all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2],
                        s=8, depthshade=False, color="black")

        # Bragg hits
        hit_xyz = np.array([vec for *_, vec in bragg_pts])
        if hit_xyz.size:
            self.ax.scatter(hit_xyz[:, 0], hit_xyz[:, 1], hit_xyz[:, 2],
                            s=30, depthshade=False, color="red")

        self.ax.set_xlabel(r"$k_x$ (Å$^{-1}$)")
        self.ax.set_ylabel(r"$k_y$")
        self.ax.set_zlabel(r"$k_z$")

        lim = r * 1.2
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])

        self.draw_idle()
