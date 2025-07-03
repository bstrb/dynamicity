"""
visualisation_pg.py
===================
PyQtGraph OpenGL canvas – drop-in replacement for Visualization3D
when you need >60 fps with large reflection sets.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor
from typing import Iterable, Tuple


class Visualization3D(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=6.0)

        self._sphere = None
        self._lattice = gl.GLScatterPlotItem()
        self._bragg = gl.GLScatterPlotItem()

        self.addItem(self._lattice)
        self.addItem(self._bragg)

    # ------------------------------------------------------------------
    def draw_scene(
        self,
        lattice_pts: Iterable[Tuple[int, int, int, np.ndarray]],
        bragg_pts: Iterable[Tuple[int, int, int, np.ndarray]],
        wavelength: float,
    ):
        r = 1.0 / wavelength

        # Build the sphere once
        if self._sphere is None:
            md = gl.MeshData.sphere(rows=30, cols=30, radius=r)
            self._sphere = gl.GLMeshItem(
                meshdata=md, smooth=True,
                color=(0.3, 0.5, 1.0, 0.15),
                shader="shaded", glOptions="additive"
            )
            self.addItem(self._sphere)

        # Lattice points
        all_xyz = np.array([v for *_, v in lattice_pts])
        self._lattice.setData(pos=all_xyz, size=3, color=QColor("black"))

        # Bragg hits
        hit_xyz = np.array([v for *_, v in bragg_pts])
        self._bragg.setData(pos=hit_xyz, size=6, color=QColor("red"))
