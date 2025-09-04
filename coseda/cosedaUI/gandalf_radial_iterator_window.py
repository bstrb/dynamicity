# coseda/cosedaUI/gandalf_radial_iterator_window.py
from __future__ import annotations

import os
import re
import time
import threading
from pathlib import Path
from typing import Optional, List

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QFileDialog, QWidget, QSizePolicy
)

from coseda.initialize import write_gandalfiteratorsettings
from coseda.gandalf_radial_iterator import (
    # DEFAULT_PEAKFINDER_OPTIONS,
    # INDEXING_FLAGS,
    count_images_in_h5_folder,
    estimate_passes,
    run_gandalf_iterator,
)
# -------------------- constants --------------------

DEFAULT_PEAKFINDER_OPTIONS = {
    "cxi": ["--peaks=cxi"],
    "peakfinder9": [
        "--peaks=peakfinder9",
        "--min-snr-biggest-pix=7",
        "--min-snr-peak-pix=6",
        "--min-snr=5",
        "--min-sig=11",
        "--min-peak-over-neighbour=-inf",
        "--local-bg-radius=3",
    ],
    "peakfinder8": [
        "--peaks=peakfinder8",
        "--threshold=800",
        "--min-snr=5",
        "--min-pix-count=2",
        "--max-pix-count=200",
        "--local-bg-radius=3",
        "--min-res=0",
        "--max-res=1200",
    ],
}

INDEXING_FLAGS: List[str] = ["--indexing=xgandalf", "--integration=rings"]

# -------------------- small helper row --------------------

class FileBrowseRow(QWidget):
    def __init__(self, label: str, mode: str, parent: Optional[QWidget] = None, default: str = ""):
        super().__init__(parent)
        self._mode = mode  # "file" or "dir"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._lab = QLabel(label)
        self._edit = QLineEdit()
        if default:  # set default path if provided
            self._edit.setText(default)

        self._btn = QPushButton("Browse…")
        self._btn.clicked.connect(self._on_browse)

        lay.addWidget(self._lab)
        lay.addWidget(self._edit, 1)
        lay.addWidget(self._btn)

    def _on_browse(self):
        if self._mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, self._lab.text(), str(Path.cwd()))
        else:
            path = QFileDialog.getExistingDirectory(self, self._lab.text(), str(Path.cwd()))
        if path:
            self._edit.setText(path)

    def text(self) -> str:
        return self._edit.text().strip()


# -------------------- worker (backend in a thread) --------------------

class Worker(QObject):
    finished = pyqtSignal(str)          # message
    error = pyqtSignal(str)             # message
    progress = pyqtSignal(int, int)     # done, total
    line = pyqtSignal(str)              # raw stdout line

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        try:
            rc = run_gandalf_iterator(
                geom_file=self.params["geom_file"],
                cell_file=self.params["cell_file"],
                input_folder=self.params["input_folder"],
                output_base=self.params["output_base"],
                threads=self.params["threads"],
                max_radius=self.params["max_radius"],
                step=self.params["step"],
                extra_flags=self.params["extra_flags"],
                progress_callback=lambda done, total: self.progress.emit(done, total),
                line_callback=lambda s: self.line.emit(s),
                cancel_check=self._stop_event.is_set,   # <-- NEW
            )
            if rc == 0:
                self.finished.emit("Indexing finished successfully.")
            elif rc == 130:
                self.error.emit("Indexing was cancelled by user.")
            else:
                self.error.emit(f"Indexing finished with errors (exit code {rc}).")
        except MemoryError:
            self.error.emit("Processing failed: ran out of memory.")
        except Exception as e:
            self.error.emit(str(e))

# -------------------- main dialog --------------------

class GandalfIteratorWindow(QDialog):
    def __init__(self, parent: Optional[Widget] = None, ini_directory: Optional[str] = None, ini_file_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Gandalf Iterator")
        self.setModal(False)
        self.ini_directory = ini_directory
        self.current_input_path = ini_file_path

        self._t0 = None
        self._per_pass_total = 0
        self._passes_total = 1
        self._mirror_stdout = False
        self._mirror_only_progress = True

        self._thread: Optional[QThread] = None
        self._worker: Optional[Worker] = None

        self._build_ui()

    # ---------- UI ----------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Files group
        files_group = QGroupBox("Files")
        files_form = QFormLayout()
        # self.geom_row = FileBrowseRow("Geometry (.geom):", "file")
        # self.cell_row = FileBrowseRow("Cell (.cell):", "file")
        # self.input_row = FileBrowseRow("Input folder:", "dir")
        self.geom_row = FileBrowseRow("Geometry (.geom):", "file", default="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/MFM.geom")
        self.cell_row = FileBrowseRow("Cell (.cell):", "file", default="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038/MFM.cell")
        self.input_row = FileBrowseRow("Input folder:", "dir", default="/home/bubl3932/files/MFM300_VIII/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524/MFM300_UK_2ndGrid_spot_4_220mm_0deg_150nm_50ms_20250524_2038")
        
        files_form.addRow(self.geom_row)
        files_form.addRow(self.cell_row)
        files_form.addRow(self.input_row)
        files_group.setLayout(files_form)
        main_layout.addWidget(files_group)

        # Basic params
        basic_group = QGroupBox("Basic Parameters")
        bg = QFormLayout()

        self.output_edit = QLineEdit("Xtal")
        self.output_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_edit.setMinimumWidth(500)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output base:"))
        output_row.addWidget(self.output_edit)
        output_row_widget = QWidget()
        output_row_widget.setLayout(output_row)
        bg.addRow(output_row_widget)

        self.threads_spin = QSpinBox(); self.threads_spin.setRange(1, 1024); self.threads_spin.setValue(24)
        self.max_radius_spin = QDoubleSpinBox(); self.max_radius_spin.setRange(0.0, 10.0); self.max_radius_spin.setDecimals(3); self.max_radius_spin.setSingleStep(0.05); self.max_radius_spin.setValue(0.0)
        self.step_spin = QDoubleSpinBox(); self.step_spin.setRange(0.01, 5.0); self.step_spin.setDecimals(3); self.step_spin.setSingleStep(0.05); self.step_spin.setValue(0.1)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Threads:"))
        params_row.addWidget(self.threads_spin)
        params_row.addSpacing(10)
        params_row.addWidget(QLabel("Max radius:"))
        params_row.addWidget(self.max_radius_spin)
        params_row.addSpacing(10)
        params_row.addWidget(QLabel("Step:"))
        params_row.addWidget(self.step_spin)
        params_widget = QWidget()
        params_widget.setLayout(params_row)
        bg.addRow(params_widget)
        basic_group.setLayout(bg)
        main_layout.addWidget(basic_group)

        # Peakfinder
        peak_group = QGroupBox("Peakfinder")
        pf_form = QFormLayout()
        peak_row = QHBoxLayout()
        self.peak_combo = QComboBox(); self.peak_combo.addItems(list(DEFAULT_PEAKFINDER_OPTIONS.keys()))
        self.peak_params_edit = QLineEdit()
        self.peak_params_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        peak_row.addWidget(QLabel("Method:"))
        peak_row.addWidget(self.peak_combo)
        peak_row.addSpacing(10)
        peak_row.addWidget(QLabel("Params (auto):"))
        peak_row.addWidget(self.peak_params_edit)
        peak_row_widget = QWidget()
        peak_row_widget.setLayout(peak_row)
        pf_form.addRow(peak_row_widget)
        peak_group.setLayout(pf_form)
        main_layout.addWidget(peak_group)
        self._update_peak_params(self.peak_combo.currentText())
        self.peak_combo.currentTextChanged.connect(self._update_peak_params)

        # Advanced flags
        adv_group = QGroupBox("Advanced Indexing & Other Flags")
        adv_form = QFormLayout()

        self.min_peaks_spin = QSpinBox(); self.min_peaks_spin.setRange(1, 1000); self.min_peaks_spin.setValue(15)
        self.tolerance_edit = QLineEdit("10,10,10,5")
        self.int_radius_edit = QLineEdit("4,5,9")

        min_cell_int_row = QHBoxLayout()
        min_cell_int_row.addWidget(QLabel("Min peaks:"))
        min_cell_int_row.addWidget(self.min_peaks_spin)
        min_cell_int_row.addSpacing(10)
        min_cell_int_row.addWidget(QLabel("Cell tolerance:"))
        min_cell_int_row.addWidget(self.tolerance_edit)
        min_cell_int_row.addSpacing(10)
        min_cell_int_row.addWidget(QLabel("Integration radius:"))
        min_cell_int_row.addWidget(self.int_radius_edit)
        min_cell_int_widget = QWidget()
        min_cell_int_widget.setLayout(min_cell_int_row)

        self.samp_pitch_spin = QSpinBox(); self.samp_pitch_spin.setRange(1, 90); self.samp_pitch_spin.setValue(5)
        self.xg_tol_spin = QDoubleSpinBox(); self.xg_tol_spin.setDecimals(4); self.xg_tol_spin.setRange(0.0001, 1.0); self.xg_tol_spin.setSingleStep(0.0005); self.xg_tol_spin.setValue(0.02)
        self.grad_desc_spin = QSpinBox(); self.grad_desc_spin.setRange(0, 100); self.grad_desc_spin.setValue(1)

        pitch_tol_grad_row = QHBoxLayout()
        pitch_tol_grad_row.addWidget(QLabel("XG sampling pitch:"))
        pitch_tol_grad_row.addWidget(self.samp_pitch_spin)
        pitch_tol_grad_row.addSpacing(10)
        pitch_tol_grad_row.addWidget(QLabel("XG tolerance:"))
        pitch_tol_grad_row.addWidget(self.xg_tol_spin)
        pitch_tol_grad_row.addSpacing(10)
        pitch_tol_grad_row.addWidget(QLabel("Grad-desc iterations:"))
        pitch_tol_grad_row.addWidget(self.grad_desc_spin)
        pitch_tol_grad_widget = QWidget()
        pitch_tol_grad_widget.setLayout(pitch_tol_grad_row)

        self.other_flags_edit = QLineEdit("--no-revalidate --no-half-pixel-shift --no-refine --no-non-hits-in-stream --no-retry --fix-profile-radius=70000000")
        self.other_flags_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.other_flags_edit.setMinimumWidth(500)

        other_flags_row = QHBoxLayout()
        other_flags_row.addWidget(QLabel("Other flags:"))
        other_flags_row.addWidget(self.other_flags_edit)
        other_flags_widget = QWidget()
        other_flags_widget.setLayout(other_flags_row)

        adv_form.addRow(min_cell_int_widget)
        adv_form.addRow(pitch_tol_grad_widget)
        adv_form.addRow(other_flags_widget)
        adv_group.setLayout(adv_form)
        main_layout.addWidget(adv_group)

        # Progress
        prog_group = QGroupBox("Progress")
        v = QVBoxLayout()
        self.progress_label = QLabel("Idle.")
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        v.addWidget(self.progress_label)
        v.addWidget(self.progress_bar)
        prog_group.setLayout(v)
        main_layout.addWidget(prog_group)

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run"); self.run_btn.clicked.connect(self._on_run)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.save_btn = QPushButton("Save Settings"); self.save_btn.clicked.connect(self._on_save_clicked); self.save_btn.setEnabled(bool(self.current_input_path))
        row.addWidget(self.run_btn); row.addWidget(self.stop_btn);row.addWidget(self.save_btn)
        main_layout.addLayout(row)

        # Mirror
        mirror_row = QHBoxLayout()
        mirror_row.addWidget(QLabel("Mirror output:"))
        self.mirror_mode = QComboBox()
        self.mirror_mode.addItems(["Off", "Progress lines", "Full stdout"])
        self.mirror_mode.currentIndexChanged.connect(self._on_mirror_change)
        mirror_row.addWidget(self.mirror_mode)
        mirror_row.addStretch(1)
        main_layout.addLayout(mirror_row)

    # ---------- UI helpers ----------

    def _on_mirror_change(self, idx: int):
        if idx == 0:
            self._mirror_stdout = False
        elif idx == 1:
            self._mirror_stdout = True
            self._mirror_only_progress = True
        else:
            self._mirror_stdout = True
            self._mirror_only_progress = False

    # ADD this method to the class (e.g., below _on_mirror_change)
    def _collect_params_for_ini(self) -> dict:
        peak_params = self.peak_params_edit.text().strip().split()
        other_flags = [f for f in self.other_flags_edit.text().split() if f.strip()]

        return dict(
            geomfile_path=self.geom_row.text().strip(),
            cellfile_path=self.cell_row.text().strip(),
            output_file_base=(self.output_edit.text().strip() or "Xtal"),
            threads=int(self.threads_spin.value()),
            max_radius=float(self.max_radius_spin.value()),
            step=float(self.step_spin.value()),
            peakfinder_method=self.peak_combo.currentText(),
            peakfinder_params=peak_params,
            min_peaks=int(self.min_peaks_spin.value()),
            cell_tolerance=self.tolerance_edit.text().strip(),
            sampling_pitch=int(self.samp_pitch_spin.value()),
            grad_desc_iterations=int(self.grad_desc_spin.value()),
            xgandalf_tolerance=float(self.xg_tol_spin.value()),
            int_radius=self.int_radius_edit.text().strip(),
            other_flags=other_flags,  # NOTE: do not include INDEXING_FLAGS here; keep user extras only
        )


    def _update_peak_params(self, method: str):
        params = DEFAULT_PEAKFINDER_OPTIONS.get(method, [])
        self.peak_params_edit.setText(" ".join(params))

    @staticmethod
    def _fmt_dur(secs: float) -> str:
        secs = max(0, int(secs))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:d}:{m:02d}:{s:02d}"

    # ---------- run/stop ----------

    def _on_run(self):
        geom = self.geom_row.text()
        cell = self.cell_row.text()
        folder = self.input_row.text()
        if not (geom and cell and folder):
            QMessageBox.warning(self, "Missing input", "Please select .geom, .cell and input folder.")
            return
        if not os.path.isdir(folder):
            QMessageBox.critical(self, "Invalid folder", f"Not a directory: {folder}")
            return

        try:
            total, _ = count_images_in_h5_folder(folder)
        except Exception as exc:
            QMessageBox.critical(self, "HDF5 error", f"Failed to scan .h5 files:\n{exc}")
            return

        self._per_pass_total = total
        self._passes_total = estimate_passes(self.max_radius_spin.value(), self.step_spin.value())
        overall_max = max(1, total * self._passes_total)

        self.progress_bar.setRange(0, overall_max)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"0 / {overall_max} images (0.0%) • passes 0/{self._passes_total}")

        advanced_flags = [
            f"--min-peaks={int(self.min_peaks_spin.value())}",
            f"--tolerance={self.tolerance_edit.text().strip()}",
            f"--xgandalf-sampling-pitch={int(self.samp_pitch_spin.value())}",
            f"--xgandalf-grad-desc-iterations={int(self.grad_desc_spin.value())}",
            f"--xgandalf-tolerance={float(self.xg_tol_spin.value())}",
            f"--int-radius={self.int_radius_edit.text().strip()}",
        ]
        other_flags = [f for f in self.other_flags_edit.text().split() if f.strip()]
        pf_flags = DEFAULT_PEAKFINDER_OPTIONS.get(self.peak_combo.currentText(), [])
        combined_flags = advanced_flags + other_flags + pf_flags + INDEXING_FLAGS

        params = dict(
            geom_file=geom,
            cell_file=cell,
            input_folder=folder,
            output_base=(self.output_edit.text().strip() or "Xtal"),
            threads=int(self.threads_spin.value()),
            max_radius=float(self.max_radius_spin.value()),
            step=float(self.step_spin.value()),
            extra_flags=combined_flags,
        )

        # start worker thread
        self._thread = QThread(self)
        self._worker = Worker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.line.connect(self._on_line)
        self._worker.finished.connect(self._on_finished_ok)
        self._worker.error.connect(self._on_finished_err)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_cleanup)

        self._t0 = time.monotonic()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._thread.start()

    def _thread_cleanup(self):
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
        self._worker = None

    def _on_stop_clicked(self):
        if self._worker is not None:
            self._worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.progress_label.setText(self.progress_label.text() + " • Stopping…")

    # ---------- slots from worker ----------

    # ADD this method to the class (e.g., near other slots)
    def _on_save_clicked(self):
        if not self.current_input_path:
            QMessageBox.warning(self, "No INI File", "Cannot determine INI file to save settings.")
            return

        # Basic validation similar to old UI
        geom = self.geom_row.text().strip()
        cell = self.cell_row.text().strip()
        if not geom or not os.path.exists(geom):
            QMessageBox.warning(self, "Invalid Geometry", "Geometry file path is missing or does not exist.")
            return
        if not cell or not os.path.exists(cell):
            QMessageBox.warning(self, "Invalid Cell", "Cell file path is missing or does not exist.")
            return
        if self.step_spin.value() <= 0:
            QMessageBox.warning(self, "Invalid Step", "Step size must be greater than zero.")
            return
        if self.max_radius_spin.value() < 0:
            QMessageBox.warning(self, "Invalid Max Radius", "Max radius must be non-negative.")
            return

        p = self._collect_params_for_ini()
        # Write using the same API as the old script
        write_gandalfiteratorsettings(
            input_path=self.current_input_path,
            geomfile_path=p["geomfile_path"],
            cellfile_path=p["cellfile_path"],
            output_file_base=p["output_file_base"],
            threads=p["threads"],
            max_radius=p["max_radius"],
            step=p["step"],
            peakfinder_method=p["peakfinder_method"],
            peakfinder_params=p["peakfinder_params"],
            min_peaks=p["min_peaks"],
            cell_tolerance=p["cell_tolerance"],
            sampling_pitch=p["sampling_pitch"],
            grad_desc_iterations=p["grad_desc_iterations"],
            xgandalf_tolerance=p["xgandalf_tolerance"],
            int_radius=p["int_radius"],
            other_flags=p["other_flags"],
        )
        QMessageBox.information(self, "Settings Saved", "Gandalf settings have been saved to the INI file.")


    def _on_progress(self, done: int, total: int):
        pct = (done / total * 100.0) if total else 0.0
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        elapsed = time.monotonic() - (self._t0 or time.monotonic())
        self.progress_label.setText(
            f"{done} / {total} images ({pct:.1f}%) • elapsed {self._fmt_dur(elapsed)}"
        )

    def _on_line(self, line: str):
        if not self._mirror_stdout:
            return
        if self._mirror_only_progress:
            if re.search(r"\bimages\s+processed\b", line, flags=re.IGNORECASE):
                print(line, flush=True)
        else:
            print(line, flush=True)

    def _on_finished_ok(self, msg: str):
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        if self.progress_bar.value() < self.progress_bar.maximum():
            self.progress_bar.setValue(self.progress_bar.maximum())
        QMessageBox.information(self, "Gandalf Iterator", msg)

    def _on_finished_err(self, msg: str):
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        QMessageBox.warning(self, "Gandalf Iterator", msg)

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = GandalfIteratorWindow()
    win.show()
    sys.exit(app.exec())
