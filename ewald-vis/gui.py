"""
gui.py
======
Qt MainWindow, parameter widgets and orchestration.
"""
from __future__ import annotations

import sys
from typing import Dict, Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QLabel, QDoubleSpinBox, QComboBox, QTextEdit, QPushButton
)

from ewald import (
    Cell, make_reciprocal_lattice, bragg_hits,
    electron_wavelength_kv, euler_omega_chi_phi
)
from visualisation_pg import Visualization3D
#   ↳  Replace with `from visualisation_pg import Visualization3D`
#       for the OpenGL canvas.

# -------------------------------------------------------------------- #
# Crystal-system presets
SYSTEM_PRESETS: Dict[str, Dict[str, object]] = {
    "Cubic":      dict(equal_ab=True,  equal_bc=True,  equal_ca=True,
                       alpha=90, beta=90, gamma=90),
    "Tetragonal": dict(equal_ab=True,  equal_bc=False, equal_ca=False,
                       alpha=90, beta=90, gamma=90),
    "Orthorhombic": dict(equal_ab=False, equal_bc=False, equal_ca=False,
                         alpha=90, beta=90, gamma=90),
    "Hexagonal": dict(equal_ab=True, equal_bc=False, equal_ca=False,
                      alpha=90, beta=90, gamma=120),
    "Trigonal":  dict(equal_ab=True,  equal_bc=True,  equal_ca=True,
                      alpha=90, beta=90, gamma=120),
    "Monoclinic": dict(equal_ab=False, equal_bc=False, equal_ca=False,
                       alpha=90, beta=None, gamma=90),   # β variable
    "Triclinic": dict(equal_ab=False, equal_bc=False, equal_ca=False,
                      alpha=None, beta=None, gamma=None),
}

# Lattice-centering reflection filters
CENTERINGS: Dict[str, Callable[[int, int, int], bool]] = {
    "P": lambda h, k, l: True,
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: (h % 2 == k % 2 == l % 2),
    "C": lambda h, k, l: (h + k) % 2 == 0,
    "R": lambda h, k, l: (-h + k + l) % 3 == 0,   # hex/trigonal axes
}

# -------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3-D Ewald-Sphere Explorer (PyQt 6)")
        self._build_ui()
        self._update_spinbox_constraints()
        self._update_scene()

    # ----------------------- UI layout ---------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        outer = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ---------- Left pane: controls & HKL list ----------
        ctl_widget = QWidget()
        ctl_layout = QVBoxLayout(ctl_widget)
        splitter.addWidget(ctl_widget)

        grid = QGridLayout()
        ctl_layout.addLayout(grid)
        row = 0

        # Crystal system
        grid.addWidget(QLabel("Crystal system:"), row, 0)
        self.sys_box = QComboBox()
        self.sys_box.addItems(SYSTEM_PRESETS.keys())
        grid.addWidget(self.sys_box, row, 1); row += 1

        # Accelerating voltage (kV) → λ
        grid.addWidget(QLabel("Voltage (kV):"), row, 0)
        self.kv_sb = self._dspin(50, 300, 1, 300)
        grid.addWidget(self.kv_sb, row, 1); row += 1

        # a, b, c
        for label in "abc":
            grid.addWidget(QLabel(f"{label} (Å):"), row, 0)
            spin = self._dspin(1, 50, 0.01, 10)
            setattr(self, f"{label}_sb", spin)
            grid.addWidget(spin, row, 1)
            row += 1

        # α, β, γ
        for label, attr in zip(("α", "β", "γ"), ("alpha", "beta", "gamma")):
            grid.addWidget(QLabel(f"{label} (°):"), row, 0)
            spin = self._dspin(30, 150, 0.1, 90)
            setattr(self, f"{attr}_sb", spin)
            grid.addWidget(spin, row, 1)
            row += 1

        # Orientation: ω, χ, φ
        for label, attr, rng in (("ω", "omega", 360),
                                 ("χ", "chi",   180),
                                 ("φ", "phi",   360)):
            grid.addWidget(QLabel(f"{label} (°):"), row, 0)
            spin = self._dspin(0, rng, 0.1, 0)
            setattr(self, f"{attr}_sb", spin)
            grid.addWidget(spin, row, 1)
            row += 1

        # HKL range
        grid.addWidget(QLabel("|h|,|k|,|l| max:"), row, 0)
        self.range_sb = self._dspin(1, 25, 1, 5, ints=True)
        grid.addWidget(self.range_sb, row, 1); row += 1

        # Centering
        grid.addWidget(QLabel("Centering:"), row, 0)
        self.cent_box = QComboBox()
        self.cent_box.addItems(CENTERINGS.keys())
        grid.addWidget(self.cent_box, row, 1); row += 1

        # Manual update button (useful when OpenGL canvas is heavy)
        self.update_btn = QPushButton("Update view")
        grid.addWidget(self.update_btn, row, 0, 1, 2); row += 1

        # HKL list
        self.hkl_view = QTextEdit(readOnly=True)
        ctl_layout.addWidget(self.hkl_view, stretch=1)

        # ---------- Right pane: 3-D canvas ----------
        self.canvas = Visualization3D()
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        # --------------- Signal connections ----------------
        self.sys_box.currentIndexChanged.connect(self._update_spinbox_constraints)

        # Live-update widgets
        for sb in (self.kv_sb, self.a_sb, self.b_sb, self.c_sb,
                   self.alpha_sb, self.beta_sb, self.gamma_sb,
                   self.omega_sb, self.chi_sb, self.phi_sb,
                   self.range_sb):
            sb.valueChanged.connect(self._update_scene)
        self.cent_box.currentIndexChanged.connect(self._update_scene)

        # Button
        self.update_btn.clicked.connect(self._update_scene)

    # ------------------------------------------------------------------
    def _dspin(
        self, lo: float, hi: float, step: float, val: float, *, ints: bool = False
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(0 if ints else 3)
        sb.setValue(val)
        return sb

    # ------------------------------------------------------------------
    def _collect_cell(self) -> Cell:
        return Cell(
            a=self.a_sb.value(),
            b=self.b_sb.value() if self.b_sb.isEnabled() else self.a_sb.value(),
            c=self.c_sb.value() if self.c_sb.isEnabled() else self.a_sb.value(),
            alpha=self.alpha_sb.value(),
            beta=self.beta_sb.value(),
            gamma=self.gamma_sb.value(),
        )

    # ----------------- System-preset constraints ----------------------
    def _update_spinbox_constraints(self) -> None:
        preset = SYSTEM_PRESETS[self.sys_box.currentText()]

        # Equal-length constraints
        lock_ab = preset.get("equal_ab", False)
        lock_bc = preset.get("equal_bc", False)
        lock_ca = preset.get("equal_ca", False)

        self.b_sb.setDisabled(lock_ab)
        self.c_sb.setDisabled(lock_ca or (lock_ab and lock_bc))

        if lock_ab:
            self.b_sb.setValue(self.a_sb.value())
        if lock_bc:
            self.c_sb.setValue(self.b_sb.value())
        if lock_ca:
            self.c_sb.setValue(self.a_sb.value())

        # Angles
        for name, sb in (("alpha", self.alpha_sb),
                         ("beta",  self.beta_sb),
                         ("gamma", self.gamma_sb)):
            fixed = preset.get(name)
            if fixed is None:
                sb.setEnabled(True)
            else:
                sb.setEnabled(False)
                sb.setValue(fixed)

        self._update_scene()

    # ------------------ Main redraw routine ---------------------------
    def _update_scene(self) -> None:
        cell = self._collect_cell()
        kv = self.kv_sb.value()
        wavelength = electron_wavelength_kv(kv)   # Å

        hmax = int(self.range_sb.value())
        lattice = make_reciprocal_lattice(cell, h_range=hmax)

        # Orientation
        R = euler_omega_chi_phi(
            self.omega_sb.value(),
            self.chi_sb.value(),
            self.phi_sb.value(),
        )
        lattice_rot = [(h, k, l, R @ g) for h, k, l, g in lattice]

        # Centering filter
        allow = CENTERINGS[self.cent_box.currentText()]
        lattice_rot = [(h, k, l, g) for h, k, l, g in lattice_rot if allow(h, k, l)]

        # Bragg reflections
        hits = bragg_hits(lattice_rot, wavelength)

        # ----- Draw ----------------------------------------------------
        self.canvas.draw_scene(lattice_rot, hits, wavelength)

        # HKL list
        if hits:
            lines = ["h k l     |G| (Å⁻¹)"]
            for h, k, l, g in hits:
                lines.append(f"{h:2d} {k:2d} {l:2d}   {np.linalg.norm(g):7.3f}")
            self.hkl_view.setPlainText("\n".join(lines))
        else:
            self.hkl_view.setPlainText("No reflections within tolerance.")


# ---------------------------------------------------------------------- #
def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
