#!/usr/bin/env python3
"""
ssed_gandalf_iterator_qt.py ‒ Gandalf Indexing GUI (Qt 6)

A drop‑in replacement for the original Tkinter front‑end that ships with
`ssed_gandalf_iterator.py`.  The business logic and CLI invocation remain
identical; this file only swaps the UI layer to Qt 6 (PyQt6).  If you prefer
LGPL bindings you can replace `PyQt6` with `PySide6` – the widget API calls
are the same.

Dependencies
------------
    pip install PyQt6

Usage
-----
    python ssed_gandalf_iterator_qt.py

When the window closes all temporary *indexamajig*/ directories created in the
working directory are removed automatically (same behaviour as the original).
"""
from __future__ import annotations

import sys
import os
import glob
import shutil
import atexit
import signal
from pathlib import Path
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QHBoxLayout,
)

# ---------------------------------------------------------------------------
# Project‑specific import ‒ identical to Tkinter version
# ---------------------------------------------------------------------------
try:
    from gandalf_interations.gandalf_radial_iterator import gandalf_iterator
except ImportError as exc:
    raise SystemExit(
        "Could not import gandalf_iterator. Ensure your PYTHONPATH contains "
        "gandalf_interations/ or install the package if distributed."
    ) from exc

# ---------------------------------------------------------------------------
# House‑keeping helpers
# ---------------------------------------------------------------------------

def cleanup_temp_dirs() -> None:
    """Remove directories in CWD that start with *indexamajig* (same semantics)."""
    for d in glob.glob("indexamajig*"):
        p = Path(d)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"Removed temporary directory: {p}")


aexit_registered = atexit.register(cleanup_temp_dirs)


def _handle_signal(_sig, _frame):
    cleanup_temp_dirs()
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Constants (copied verbatim from Tkinter original)
# ---------------------------------------------------------------------------

default_peakfinder_options = {
    "cxi": "--peaks=cxi",
    "peakfinder9": """--peaks=peakfinder9\n--min-snr-biggest-pix=7\n--min-snr-peak-pix=6\n--min-snr=5\n--min-sig=11\n--min-peak-over-neighbour=-inf\n--local-bg-radius=3""",
    "peakfinder8": """--peaks=peakfinder8\n--threshold=800\n--min-snr=5\n--min-pix-count=2\n--max-pix-count=200\n--local-bg-radius=3\n--min-res=0\n--max-res=1200""",
}

INDEXING_FLAGS: List[str] = ["--indexing=xgandalf", "--integration=rings"]

# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------


class FileBrowseRow(QWidget):
    """A helper widget consisting of a label, line‑edit and *Browse* button."""

    def __init__(self, title: str, file_mode: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.file_mode = file_mode  # "file" | "dir"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(title)
        self.path_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)

        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(browse_btn)

    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        if self.file_mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, self.label.text(), str(Path.cwd()))
        else:
            path = QFileDialog.getExistingDirectory(self, self.label.text(), str(Path.cwd()))
        if path:
            self.path_edit.setText(path)

    # utilities ---------------------------------------------------------
    def text(self) -> str:
        return self.path_edit.text().strip()


class GandalfWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gandalf Indexing GUI – Qt 6")
        self._central = QWidget()
        self.setCentralWidget(self._central)
        self._build_ui()

    # UI ----------------------------------------------------------------
    def _build_ui(self):
        root_layout = QVBoxLayout(self._central)

        # Description ----------------------------------------------------
        desc = (
            "Run the indexamajig command with optional outward centre shifts in a grid.\n"
            "Select .geom and .cell files and choose the input folder with .h5 files to be processed.\n"
            "Set basic parameters such as Output Base (name of your sample), Threads (CPU cores),\n"
            "Max Radius (maximum shift distance), and Step (grid spacing).\n"
            "Configure Peakfinder options, advanced indexing parameters and optionally extra flags.\n"
            "Click 'Run Indexing' to execute indexing iterations with shifted centres until the\n"
            "specified radius."
        )
        lbl_desc = QLabel(desc)
        lbl_desc.setWordWrap(True)
        root_layout.addWidget(lbl_desc)

        # File selection -------------------------------------------------
        file_group = QGroupBox("File Selection")
        fg_layout = QVBoxLayout(file_group)
        self.geom_row = FileBrowseRow("Geometry File (.geom):", "file")
        self.cell_row = FileBrowseRow("Cell File (.cell):", "file")
        self.input_row = FileBrowseRow("Input Folder:", "dir")
        fg_layout.addWidget(self.geom_row)
        fg_layout.addWidget(self.cell_row)
        fg_layout.addWidget(self.input_row)
        root_layout.addWidget(file_group)

        # Basic parameters ----------------------------------------------
        basic_group = QGroupBox("Basic Parameters")
        bg = QGridLayout(basic_group)

        bg.addWidget(QLabel("Output Base:"), 0, 0)
        self.output_base_edit = QLineEdit("Xtal")
        bg.addWidget(self.output_base_edit, 0, 1)

        bg.addWidget(QLabel("Threads:"), 1, 0)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 1024)
        self.threads_spin.setValue(24)
        bg.addWidget(self.threads_spin, 1, 1)

        bg.addWidget(QLabel("Max Radius:"), 2, 0)
        self.max_radius_spin = QDoubleSpinBox()
        self.max_radius_spin.setRange(0.0, 10.0)
        self.max_radius_spin.setDecimals(3)
        self.max_radius_spin.setSingleStep(0.05)
        self.max_radius_spin.setValue(0.1)
        bg.addWidget(self.max_radius_spin, 2, 1)

        bg.addWidget(QLabel("Step:"), 3, 0)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.01, 5.0)
        self.step_spin.setDecimals(3)
        self.step_spin.setSingleStep(0.05)
        self.step_spin.setValue(0.1)
        bg.addWidget(self.step_spin, 3, 1)

        root_layout.addWidget(basic_group)

        # Peakfinder -----------------------------------------------------
        peak_group = QGroupBox("Peakfinder Options")
        pg = QGridLayout(peak_group)

        pg.addWidget(QLabel("Peakfinder:"), 0, 0)
        self.peak_combo = QComboBox()
        self.peak_combo.addItems(["cxi", "peakfinder9", "peakfinder8"])
        pg.addWidget(self.peak_combo, 0, 1)

        pg.addWidget(QLabel("Peakfinder Params:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        self.peak_params_edit = QTextEdit()
        self.peak_params_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        pg.addWidget(self.peak_params_edit, 1, 1)

        # initial text
        self._update_peak_params(self.peak_combo.currentText())
        self.peak_combo.currentTextChanged.connect(self._update_peak_params)

        root_layout.addWidget(peak_group)

        # Advanced indexing ---------------------------------------------
        adv_group = QGroupBox("Advanced Indexing Parameters")
        ag = QGridLayout(adv_group)

        ag.addWidget(QLabel("Min Peaks:"), 0, 0)
        self.min_peaks_spin = QSpinBox()
        self.min_peaks_spin.setRange(1, 1000)
        self.min_peaks_spin.setValue(15)
        ag.addWidget(self.min_peaks_spin, 0, 1)

        ag.addWidget(QLabel("Cell Tolerance:"), 0, 2)
        self.tolerance_edit = QLineEdit("10,10,10,5")
        ag.addWidget(self.tolerance_edit, 0, 3)

        ag.addWidget(QLabel("Sampling Pitch:"), 1, 0)
        self.samp_pitch_spin = QSpinBox()
        self.samp_pitch_spin.setRange(1, 90)
        self.samp_pitch_spin.setValue(5)
        ag.addWidget(self.samp_pitch_spin, 1, 1)

        ag.addWidget(QLabel("Grad Desc Iterations:"), 1, 2)
        self.grad_desc_spin = QSpinBox()
        self.grad_desc_spin.setRange(0, 100)
        self.grad_desc_spin.setValue(1)
        ag.addWidget(self.grad_desc_spin, 1, 3)

        ag.addWidget(QLabel("XGandalf Tolerance:"), 2, 0)
        self.xg_tol_spin = QDoubleSpinBox()
        self.xg_tol_spin.setDecimals(4)
        self.xg_tol_spin.setRange(0.0001, 1.0)
        self.xg_tol_spin.setSingleStep(0.0005)
        self.xg_tol_spin.setValue(0.02)
        ag.addWidget(self.xg_tol_spin, 2, 1)

        ag.addWidget(QLabel("Integration Radius:"), 2, 2)
        self.int_radius_edit = QLineEdit("2,4,10")
        ag.addWidget(self.int_radius_edit, 2, 3)

        root_layout.addWidget(adv_group)

        # Other flags ----------------------------------------------------
        other_group = QGroupBox("Other Extra Flags")
        ov = QVBoxLayout(other_group)
        self.other_flags_edit = QTextEdit()
        self.other_flags_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.other_flags_edit.setPlainText("""--no-revalidate\n--no-half-pixel-shift\n--no-refine\n--no-non-hits-in-stream""")
        ov.addWidget(self.other_flags_edit)
        root_layout.addWidget(other_group)

        # Run button -----------------------------------------------------
        run_btn = QPushButton("Run Indexing")
        run_btn.setStyleSheet("background-color: lightblue; font-weight: bold")
        run_btn.clicked.connect(self._on_run_clicked)
        root_layout.addWidget(run_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_peak_params(self, method: str):
        self.peak_params_edit.setPlainText(default_peakfinder_options.get(method, ""))

    # ------------------------------------------------------------------
    # Run button callback
    # ------------------------------------------------------------------
    def _on_run_clicked(self):
        # Validate file selections --------------------------------------
        geom_file = self.geom_row.text()
        cell_file = self.cell_row.text()
        input_folder = self.input_row.text()
        if not (geom_file and cell_file and input_folder):
            QMessageBox.critical(self, "Missing Information", "Please select Geometry, Cell and Input folder paths.")
            return

        # Basic params ---------------------------------------------------
        output_base = self.output_base_edit.text().strip() or "Xtal"
        threads = int(self.threads_spin.value())
        max_radius = float(self.max_radius_spin.value())
        step = float(self.step_spin.value())

        # Peakfinder -----------------------------------------------------
        peakfinder_method = self.peak_combo.currentText()
        peakfinder_params = [ln.strip() for ln in self.peak_params_edit.toPlainText().splitlines() if ln.strip()]

        # Advanced flags -------------------------------------------------
        advanced_flags = [
            f"--min-peaks={self.min_peaks_spin.value()}",
            f"--tolerance={self.tolerance_edit.text().strip()}",
            f"--xgandalf-sampling-pitch={self.samp_pitch_spin.value()}",
            f"--xgandalf-grad-desc-iterations={self.grad_desc_spin.value()}",
            f"--xgandalf-tolerance={self.xg_tol_spin.value()}",
            f"--int-radius={self.int_radius_edit.text().strip()}",
        ]

        # Other flags list ----------------------------------------------
        other_flags = [ln.strip() for ln in self.other_flags_edit.toPlainText().splitlines() if ln.strip()]

        # Final flags
        flags_list = advanced_flags + other_flags + peakfinder_params + INDEXING_FLAGS

        # Debug printout -------------------------------------------------
        print("Running gandalf_iterator with the following parameters:")
        print("Geometry File:", geom_file)
        print("Cell File:", cell_file)
        print("Input Folder:", input_folder)
        print("Output Base:", output_base)
        print("Threads:", threads)
        print("Max Radius:", max_radius)
        print("Step:", step)
        print("\nPeakfinder Option:", peakfinder_method)
        print("\nAdvanced Flags:")
        for f in advanced_flags:
            print("  ", f)
        print("\nOther Flags:")
        for f in other_flags:
            print("  ", f)
        print("\nCombined Flags:", flags_list)
        self._run_gandalf(
            geom_file,
            cell_file,
            input_folder,
            output_base,
            threads,
            max_radius,
            step,
            flags_list,
        )

    # ------------------------------------------------------------------
    def _run_gandalf(
        self,
        geom: str,
        cell: str,
        input_folder: str,
        output_base: str,
        threads: int,
        max_radius: float,
        step: float,
        extra_flags: List[str],
    ) -> None:
        try:
            gandalf_iterator(
                geom,
                cell,
                input_folder,
                output_base,
                threads,
                max_radius=max_radius,
                step=step,
                extra_flags=extra_flags,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Indexing Error", f"An error occurred during indexing:\n{exc}")
        else:
            QMessageBox.information(self, "Indexing Complete", "Indexing completed successfully.")
            cleanup_temp_dirs()


# ---------------------------------------------------------------------------
# main‑guard
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    win = GandalfWindow()
    win.resize(800, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
