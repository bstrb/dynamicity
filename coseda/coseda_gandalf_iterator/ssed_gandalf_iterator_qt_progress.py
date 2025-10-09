#!/usr/bin/env python3
"""
ssed_gandalf_iterator_qt_progress.py ‒ Gandalf Indexing GUI (Qt 6) with progress bar

Adds:
  • Progress bar driven by terminal-like output from gandalf_iterator
  • HDF5 total-image counting across all .h5 files in input folder
  • Threaded execution so the UI stays responsive

Dependencies:
    pip install PyQt6 h5py
"""

from __future__ import annotations

import sys
import os
import glob
import shutil
import atexit
import time
import h5py

import signal
import re
from pathlib import Path
from typing import List, Optional

if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("XDG_RUNTIME_DIR", "").startswith("/mnt/wslg"):
    # Running inside WSLg → use Wayland
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
elif os.environ.get("DISPLAY"):
    # DISPLAY already set (e.g. VcXsrv/X410) → use X11
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt6.QtCore import Qt, QProcess, QTimer


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
    QProgressBar,
    QCheckBox,
    QScrollArea
)

# ---------------------------------------------------------------------------
# House-keeping helpers
# ---------------------------------------------------------------------------

def cleanup_temp_dirs() -> None:
    """Remove directories in CWD that start with *indexamajig* (same semantics)."""
    for d in glob.glob("indexamajig*"):
        p = Path(d)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"Removed temporary directory: {p}")

    # Remove mille-data.bin files
    for f in glob.glob("mille-data.bin*"):
        p = Path(f)
        if p.is_file():
            try:
                p.unlink()
                print(f"Removed temporary file: {p}")
            except Exception as e:
                print(f"Could not remove {p}: {e}")

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
    "peakfinder9": """--peaks=peakfinder9
--min-snr-biggest-pix=7
--min-snr-peak-pix=6
--min-snr=5
--min-sig=11
--min-peak-over-neighbour=-inf
--local-bg-radius=3""",
    "peakfinder8": """--peaks=peakfinder8
--threshold=800
--min-snr=5
--min-pix-count=2
--max-pix-count=200
--local-bg-radius=3
--min-res=0
--max-res=1200""",
}

INDEXING_FLAGS: List[str] = ["--indexing=xgandalf", "--integration=rings"]

# ---------------------------------------------------------------------------
# HDF5 image counting
# ---------------------------------------------------------------------------
def count_images_in_h5_folder(folder: str):
    """Count total images from /entry/data/images in all .h5 files in `folder`."""
    import h5py, glob, os
    h5_paths = sorted(glob.glob(os.path.join(folder, "**", "*.h5"), recursive=True))
    total = 0
    per_file = []

    for path in h5_paths:
        n_for_file = 0
        try:
            with h5py.File(path, "r") as f:
                if "/entry/data/images" in f:
                    ds = f["/entry/data/images"]
                    if ds.ndim >= 3:
                        n_for_file = ds.shape[0]
        except Exception:
            n_for_file = 0
        total += n_for_file
        per_file.append((path, n_for_file))

    return total, per_file


# ---------------------------------------------------------------------------
# Qt widgets
# ---------------------------------------------------------------------------

class FileBrowseRow(QWidget):
    """A helper widget consisting of a label, line-edit and *Browse* button."""

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

    def _on_browse(self) -> None:
        if self.file_mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, self.label.text(), str(Path.cwd()))
        else:
            path = QFileDialog.getExistingDirectory(self, self.label.text(), str(Path.cwd()))
        if path:
            self.path_edit.setText(path)

    def text(self) -> str:
        return self.path_edit.text().strip()


class GandalfWindow(QMainWindow):
    """Main application window."""

    progress_line_regex = re.compile(r"^\s*(\d+)\s+images\s+processed\b", re.IGNORECASE)

    # --- timing/throughput helpers ---
    def _fmt_dur(self, secs: float) -> str:
        secs = max(0, int(secs))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:d}:{m:02d}:{s:02d}"

    def _reset_timing(self):
        self._t0 = time.monotonic()
        self._ema_rate = None  # exponential moving average of images/sec

    def _update_rate(self, processed: int, images_per_sec_from_line: Optional[float] = None) -> float:
        # prefer the parser’s images/sec if present; else derive from elapsed
        now = time.monotonic()
        elapsed = max(1e-6, now - getattr(self, "_t0", now))
        raw_rate = images_per_sec_from_line if images_per_sec_from_line and images_per_sec_from_line > 0 else (processed / elapsed if processed > 0 else 0.0)
        # EMA for stability
        alpha = 0.2
        if self._ema_rate is None:
            self._ema_rate = raw_rate
        else:
            self._ema_rate = alpha * raw_rate + (1 - alpha) * self._ema_rate
        return max(1e-6, self._ema_rate)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gandalf Indexing GUI – Qt 6")

        # --- scrollable central area (Option A) ---
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._central = QWidget()                 # this holds your layouts/widgets
        self._scroll.setWidget(self._central)     # put content inside the scroll area
        self.setCentralWidget(self._scroll)

        self._build_ui()  # builds into self._central as before

        # Runtime state
        self._total_images: int = 0

        # Timing & multi-pass state
        self._t0 = None
        self._ema_rate = None
        self._per_pass_total = 0
        self._passes_total = 1
        self._passes_done = 0
        self._prev_processed = 0

        self._cancelling = False


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
        self.max_radius_spin.setValue(0.0)
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
        self.int_radius_edit = QLineEdit("4,5,9")
        ag.addWidget(self.int_radius_edit, 2, 3)

        root_layout.addWidget(adv_group)

        # Other flags ----------------------------------------------------
        other_group = QGroupBox("Other Extra Flags")
        ov = QVBoxLayout(other_group)
        self.other_flags_edit = QTextEdit()
        self.other_flags_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.other_flags_edit.setPlainText("""--no-revalidate
--no-half-pixel-shift
--no-refine
--no-non-hits-in-stream
--no-retry
--fix-profile-radius=70000000
""")
        ov.addWidget(self.other_flags_edit)
        root_layout.addWidget(other_group)

        # Progress bar + status -----------------------------------------
        prog_group = QGroupBox("Progress")
        pl = QVBoxLayout(prog_group)
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)  # will be set after counting
        self.prog_bar.setFormat("%v / %m images (%p%)")
        self.status_lbl = QLabel("Idle.")
        pl.addWidget(self.prog_bar)
        pl.addWidget(self.status_lbl)
        root_layout.addWidget(prog_group)

        # Run / Stop buttons --------------------------------------------------
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Indexing")
        self.run_btn.setStyleSheet("background-color: lightblue; font-weight: bold")
        self.run_btn.clicked.connect(self._on_run_clicked)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #f7c2c2; font-weight: bold")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        root_layout.addLayout(btn_row)

        self.mirror_chk = QCheckBox("Show raw output in terminal")
        self.mirror_chk.setToolTip("Mirror gandalf/indexamajig stdout to this console.")
        self.mirror_chk.toggled.connect(self._on_toggle_mirror)
        root_layout.addWidget(self.mirror_chk)


        self.mirror_progress_only_chk = QCheckBox("Only 'images processed' lines")
        self.mirror_progress_only_chk.setToolTip("When mirroring, print only lines that update progress.")
        self.mirror_progress_only_chk.setChecked(True)
        self.mirror_progress_only_chk.setEnabled(False)  # only meaningful when mirroring is on
        root_layout.addWidget(self.mirror_progress_only_chk)

        # init flag in __init__
        self._mirror_stdout = False
        self._mirror_only_progress = True
        
        # wire up the toggles
        self.mirror_chk.toggled.connect(self._on_toggle_mirror)
        self.mirror_chk.toggled.connect(self.mirror_progress_only_chk.setEnabled)
        self.mirror_progress_only_chk.toggled.connect(self._on_toggle_mirror_only_progress)

    def _on_toggle_mirror(self, checked: bool):
        self._mirror_stdout = checked
    
    def _on_toggle_mirror_only_progress(self, checked: bool):
        self._mirror_only_progress = checked


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_peak_params(self, method: str):
        self.peak_params_edit.setPlainText(default_peakfinder_options.get(method, ""))

    def _set_progress_total(self, total: int):
        self._total_images = max(0, int(total))
        if self._total_images <= 0:
            # Avoid division-by-zero display; use a dummy max and show 0%.
            self.prog_bar.setRange(0, 1)
            self.prog_bar.setValue(0)
            self.status_lbl.setText("No images detected in .h5 files.")
        else:
            self.prog_bar.setRange(0, self._total_images)
            self.prog_bar.setValue(0)
            self.status_lbl.setText(f"0 / {self._total_images} images (0%).")

    def _estimate_total_passes(self, max_radius: float, step: float) -> int:
        """
        Estimate how many centre-shift runs gandalf_iterator will do for a given
        max_radius and step, assuming a circular mask on a square grid and
        including the origin (0,0).
        """
        if step <= 0 or max_radius <= 0:
            return 1
        import math
        r = float(max_radius)
        s = float(step)
        k = int(math.ceil(r / s))
        count = 0
        for ix in range(-k, k + 1):
            for iy in range(-k, k + 1):
                if (ix * s) ** 2 + (iy * s) ** 2 <= r * r + 1e-9:
                    count += 1
        return max(1, count)


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

        if not os.path.isdir(input_folder):
            QMessageBox.critical(self, "Invalid Input Folder", f"Not a directory: {input_folder}")
            return

        # Precompute total images ---------------------------------------
        try:
            total, per_file = count_images_in_h5_folder(input_folder)
        except Exception as exc:
            QMessageBox.critical(self, "HDF5 Scan Error", f"Failed to scan .h5 files:\n{exc}")
            return

        self._set_progress_total(total)

        # Basic params ---------------------------------------------------
        output_base = self.output_base_edit.text().strip() or "Xtal"
        threads = int(self.threads_spin.value())
        max_radius = float(self.max_radius_spin.value())
        step = float(self.step_spin.value())

        # figure out how many passes (centre shifts) we expect
        self._per_pass_total = self._total_images
        self._passes_total = self._estimate_total_passes(max_radius, step)
        self._passes_done = 0
        self._prev_processed = 0

        # scale the progress bar to overall total
        overall_max = max(1, self._per_pass_total * self._passes_total)
        self.prog_bar.setRange(0, overall_max)
        self.prog_bar.setValue(0)
        self.status_lbl.setText(
            f"0 / {overall_max} images (0.0%) • passes 0/{self._passes_total}"
        )


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
        if self._total_images > 0:
            print(f"Total images detected across .h5 files: {self._total_images}")
        else:
            print("Warning: No images were detected across .h5 files; progress will remain at 0%.")

        # Kick off worker thread ----------------------------------------
        self._start_worker(
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

    def _start_worker(
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
        """
        Spawn a separate Python process that imports and runs gandalf_iterator,
        then stream its stdout into our progress parser.
        """
        # Kill any previous process
        try:
            if hasattr(self, "proc") and self.proc is not None:
                self.proc.kill()
        except Exception:
            pass

        # Create a tiny runner script on the fly
        import tempfile, textwrap
        runner_code = textwrap.dedent("""
            import sys
            from gandalf_radial_iterator import gandalf_iterator

            if __name__ == "__main__":
                args = sys.argv[1:]
                if len(args) < 7:
                    sys.stderr.write("Runner expects: geom cell input_folder output_base threads max_radius step [extra flags...]\\n")
                    sys.stderr.flush()
                    sys.exit(2)
                geom, cell, input_folder, output_base, threads, max_radius, step, *extra_flags = args
                gandalf_iterator(
                    geom, cell, input_folder, output_base,
                    int(threads),
                    max_radius=float(max_radius),
                    step=float(step),
                    extra_flags=extra_flags,
                )
        """).lstrip()

        tmp = tempfile.NamedTemporaryFile("w", delete=False, prefix="run_gandalf_", suffix=".py")
        tmp.write(runner_code)
        tmp.flush()
        tmp.close()
        self._runner_path = tmp.name

        # Build child argv (all strings)
        argv = [
            sys.executable,
            self._runner_path,
            str(geom), str(cell), str(input_folder), str(output_base),
            str(threads), str(max_radius), str(step),
            *[str(f) for f in extra_flags],
        ]

        # Create QProcess and merge stderr into stdout
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        # Ensure the child has unbuffered output and can import your modules
        from PyQt6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")

        # Propagate PYTHONPATH (include this process's sys.path)
        sep = os.pathsep
        parent_pp = os.environ.get("PYTHONPATH", "")
        extra_paths = [p for p in sys.path if isinstance(p, str) and p]
        combined_pp = (parent_pp + (sep if parent_pp else "")) + sep.join(extra_paths)
        env.insert("PYTHONPATH", combined_pp)
        self.proc.setProcessEnvironment(env)

        # Run from current working directory (adjust if your iterator expects another cwd)
        self.proc.setWorkingDirectory(os.getcwd())

        # Wire signals -> handlers
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.errorOccurred.connect(self._on_proc_error)

        # Start
        self.proc.start(argv[0], argv[1:])
        if not self.proc.waitForStarted(5000):
            QMessageBox.critical(self, "Launch Error", "Failed to start gandalf subprocess.")
            return
        
        self._stdout_buffer = ""
        self.status_lbl.setText("Indexing started…")

        self._passes_done = 0
        self._prev_processed = 0
        self._reset_timing()

        # UI state
        self._cancelling = False
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_stop_clicked(self):
        """Gracefully stop the running indexing process, escalating if needed."""
        if not hasattr(self, "proc") or self.proc is None:
            return
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            return

        self._cancelling = True
        self.status_lbl.setText("Stopping… (trying graceful interrupt)")

        # 1) Try SIGINT to allow cleanup (Linux/WSL). If that fails, fall back to terminate().
        try:
            pid = int(self.proc.processId())
            if pid > 0:
                try:
                    os.kill(pid, signal.SIGINT)
                except Exception:
                    # Not fatal; try terminate below
                    pass
        except Exception:
            pass

        # 2) Also ask nicely via terminate (SIGTERM on Unix)
        try:
            self.proc.terminate()
        except Exception:
            pass

        # 3) Escalate to SIGKILL if it doesn't exit in time
        def _force_kill():
            if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
                self.status_lbl.setText("Force-stopping…")
                try:
                    self.proc.kill()
                except Exception:
                    pass

        # give it a few seconds to wind down before force-kill
        QTimer.singleShot(5000, _force_kill)

        # UI: disable Stop to prevent repeated clicks
        self.stop_btn.setEnabled(False)


    def _on_proc_output(self):
        """
        Read incremental stdout from the child process, split into lines,
        and feed each line through the same parser we already have.
        """
        data = self.proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if not hasattr(self, "_stdout_buffer"):
            self._stdout_buffer = ""
        self._stdout_buffer += data
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._on_stdout_line(line)

    def _on_proc_finished(self, exitCode: int, exitStatus):
        # Flush any partial line
        if getattr(self, "_stdout_buffer", ""):
            self._on_stdout_line(self._stdout_buffer)
            self._stdout_buffer = ""

        per_pass = self._per_pass_total if getattr(self, "_per_pass_total", 0) else self._total_images
        passes = getattr(self, "_passes_total", 1) or 1
        overall_max = max(1, (per_pass or 0) * passes)

        self.prog_bar.setRange(0, overall_max)
        self.prog_bar.setValue(overall_max if not self._cancelling else self.prog_bar.value())

        # Elapsed & final label
        if getattr(self, "_t0", None) is not None:
            elapsed = time.monotonic() - self._t0
            base = f"passes {passes}/{passes}  •  elapsed {self._fmt_dur(elapsed)}"
        else:
            base = ""

        if self._cancelling:
            self.status_lbl.setText(f"Cancelled at {self.prog_bar.value()} / {overall_max} images.  {base}")
        else:
            self.status_lbl.setText(f"Done. {overall_max} / {overall_max} images (100%).  {base}")

        # UI restore
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Notify
        if self._cancelling:
            QMessageBox.information(self, "Indexing Stopped", "Indexing was cancelled.")
        else:
            if exitCode == 0:
                QMessageBox.information(self, "Indexing Complete", "Indexing finished successfully.")
            else:
                QMessageBox.warning(self, "Indexing Finished with Errors",
                                    f"Subprocess exit code: {exitCode}")

        try:
            if hasattr(self, "_runner_path"):
                Path(self._runner_path).unlink(missing_ok=True)  # Python 3.8+: wrap in try for older
        except Exception:
            pass


        cleanup_temp_dirs()


    def _on_proc_error(self, err):  
        QMessageBox.critical(self, "Indexing Error", f"Subprocess error: {err}")


    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _on_stdout_line(self, line: str):
        """
        Parse lines like:
          '9452 images processed, 5015 hits (53.1%), 3139 indexable (62.6% of hits, 33.2% overall), 3139 crystals, 47.6 images/sec.'
        and update the progress bar accordingly.
        """
        m = self.progress_line_regex.match(line)
        if m:
            try:
                processed = int(m.group(1))
            except Exception:
                processed = None  # type: ignore[assignment]
            if processed is not None:

                # Detect a new pass if processed counter resets or goes backwards
                if processed < self._prev_processed:
                    self._passes_done = min(self._passes_done + 1, self._passes_total - 1)  # clamp
                self._prev_processed = processed

                per_pass = self._per_pass_total if self._per_pass_total > 0 else self._total_images
                # Clamp per-pass processed to avoid spurious overshoots
                cur = min(processed, per_pass) if per_pass > 0 else processed

                overall_done = self._passes_done * per_pass + cur
                overall_max = max(1, per_pass * self._passes_total)
                overall_done = min(overall_done, overall_max)

                self.prog_bar.setRange(0, overall_max)
                self.prog_bar.setValue(overall_done)

                # optional: parse "XX.X images/sec" from the line
                m_rate = re.search(r"([\d.]+)\s*images/sec", line)
                rate = self._update_rate(overall_done, float(m_rate.group(1)) if m_rate else None)

                # elapsed & ETA based on overall
                now = time.monotonic()
                elapsed = now - (self._t0 or now)
                remaining = max(0.0, (overall_max - overall_done) / rate) if rate > 0 else 0.0
                total_est = elapsed + remaining
                pct = (overall_done / overall_max) * 100.0

                self.status_lbl.setText(
                    f"{overall_done} / {overall_max} images ({pct:.1f}%)  •  "
                    f"passes {self._passes_done + 1}/{self._passes_total}  •  "
                    f"elapsed {self._fmt_dur(elapsed)}  •  ETA {self._fmt_dur(remaining)}  "
                    f"(~total {self._fmt_dur(total_est)})"
                )

        if self._mirror_stdout:
            if not self._mirror_only_progress or m:
                print(line)
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# main-guard
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    win = GandalfWindow()
    win.resize(800, 980)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
