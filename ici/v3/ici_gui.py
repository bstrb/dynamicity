#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shlex
import traceback
import subprocess
import re
import time


from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QProgressBar,
)

import ici_orchestrator as orch


class GuiStream:
    """
    Simple stream object that forwards writes to a callback.
    Used to capture stdout/stderr and send to the GUI.
    """
    def __init__(self, callback):
        self._callback = callback

    def write(self, text):
        if text:
            self._callback(text)

    def flush(self):
        # Nothing special required; provided for compatibility.
        pass


class OrchestratorWorker(QObject):
    """
    Worker object that runs orch.main(argv) inside a QThread
    and forwards stdout/stderr and progress to the GUI via signals.
    """
    text_ready = pyqtSignal(str)
    finished = pyqtSignal(int)

    # Progress bar signals (for run_sh.py)
    progress_init = pyqtSignal(int)   # total events
    progress_step = pyqtSignal(int)   # increment (usually 1)
    progress_done = pyqtSignal()      # done / reset

    def __init__(self, argv, parent=None):
        super().__init__(parent)
        self.argv = argv

    def run(self):
        # Redirect stdout/stderr so that orch.OrchestratorRunLogger
        # tees to our GuiStream as "real_stream".
        orig_out, orig_err = sys.stdout, sys.stderr
        gui_stream = GuiStream(self.text_ready.emit)
        sys.stdout = gui_stream
        sys.stderr = gui_stream

        exit_code = 0
        try:
            # orch.main behaves like a CLI main(argv)
            exit_code = orch.main(self.argv)
        except SystemExit as e:
            # orch may raise SystemExit (from run_py / argparse)
            try:
                exit_code = int(e.code)
            except Exception:
                exit_code = 1
        except Exception:
            # Any unexpected exception: log traceback to GUI
            tb = traceback.format_exc()
            self.text_ready.emit(tb)
            exit_code = 1
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err

        self.finished.emit(exit_code)


class OrchestratorMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("SerialED Orchestrator GUI")
        self.resize(1000, 800)

        self._thread = None
        self._worker = None
        self._running = False
        self._stop_requested = False
        self._orig_run_py = None   # original orch.run_py

        # ---------- Scroll area root ----------
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget(scroll)
        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        main_layout = QVBoxLayout(container)

        # ---------- Top group: paths and basic settings ----------
        paths_group = QGroupBox("Paths and basic settings", container)
        paths_layout = QFormLayout(paths_group)

        # Run root
        self.run_root_edit = QLineEdit(paths_group)
        self.run_root_edit.setText(orch.DEFAULT_ROOT)
        run_root_btn = QPushButton("Browse…", paths_group)
        run_root_btn.clicked.connect(self.browse_run_root)
        run_root_row = QHBoxLayout()
        run_root_row.addWidget(self.run_root_edit)
        run_root_row.addWidget(run_root_btn)
        paths_layout.addRow("Run root:", run_root_row)

        # Geom
        self.geom_edit = QLineEdit(paths_group)
        self.geom_edit.setText(orch.DEFAULT_GEOM)
        geom_btn = QPushButton("Browse…", paths_group)
        geom_btn.clicked.connect(self.browse_geom)
        geom_row = QHBoxLayout()
        geom_row.addWidget(self.geom_edit)
        geom_row.addWidget(geom_btn)
        paths_layout.addRow("Geom (.geom):", geom_row)

        # Cell
        self.cell_edit = QLineEdit(paths_group)
        self.cell_edit.setText(orch.DEFAULT_CELL)
        cell_btn = QPushButton("Browse…", paths_group)
        cell_btn.clicked.connect(self.browse_cell)
        cell_row = QHBoxLayout()
        cell_row.addWidget(self.cell_edit)
        cell_row.addWidget(cell_btn)
        paths_layout.addRow("Cell (.cell):", cell_row)

        # HDF5 list
        self.h5_edit = QPlainTextEdit(paths_group)
        # Pre-fill with the default H5 sources, one per line
        self.h5_edit.setPlainText("\n".join(orch.DEFAULT_H5))
        h5_buttons_layout = QHBoxLayout()
        h5_add_btn = QPushButton("Add…", paths_group)
        h5_clear_btn = QPushButton("Clear", paths_group)
        h5_add_btn.clicked.connect(self.add_h5_files)
        h5_clear_btn.clicked.connect(self.clear_h5_files)
        h5_buttons_layout.addWidget(h5_add_btn)
        h5_buttons_layout.addWidget(h5_clear_btn)
        paths_layout.addRow("HDF5 files / globs:", self.h5_edit)
        paths_layout.addRow("", h5_buttons_layout)

        # Max iterations
        self.max_iters_spin = QSpinBox(paths_group)
        self.max_iters_spin.setMinimum(1)
        self.max_iters_spin.setMaximum(10_000)
        self.max_iters_spin.setValue(orch.DEFAULT_MAX_ITERS)
        paths_layout.addRow("Max iterations:", self.max_iters_spin)

        # Jobs (parallelism)
        self.jobs_spin = QSpinBox(paths_group)
        self.jobs_spin.setMinimum(1)
        self.jobs_spin.setMaximum(512)
        self.jobs_spin.setValue(int(orch.DEFAULT_NUM_CPU))
        paths_layout.addRow("Jobs:", self.jobs_spin)

        main_layout.addWidget(paths_group)

        # ---------- Middle group: convergence / propose_next_shifts ----------
        conv_group = QGroupBox("Convergence / propose_next_shifts parameters", container)
        conv_layout = QFormLayout(conv_group)

        # radius_mm
        self.radius_spin = QDoubleSpinBox(conv_group)
        self.radius_spin.setDecimals(6)
        self.radius_spin.setRange(0.0, 10.0)
        self.radius_spin.setSingleStep(0.001)
        self.radius_spin.setValue(float(orch.radius_mm))
        conv_layout.addRow("radius_mm:", self.radius_spin)

        # min_spacing_mm
        self.min_spacing_spin = QDoubleSpinBox(conv_group)
        self.min_spacing_spin.setDecimals(8)
        self.min_spacing_spin.setRange(0.0, 1.0)
        self.min_spacing_spin.setSingleStep(1e-4)
        self.min_spacing_spin.setValue(float(orch.min_spacing_mm))
        conv_layout.addRow("min_spacing_mm:", self.min_spacing_spin)

        # N_conv
        self.N_conv_spin = QSpinBox(conv_group)
        self.N_conv_spin.setMinimum(1)
        self.N_conv_spin.setMaximum(1000)
        self.N_conv_spin.setValue(int(orch.N_conv))
        conv_layout.addRow("N_conv:", self.N_conv_spin)

        # recurring_tol
        self.recurring_tol_spin = QDoubleSpinBox(conv_group)
        self.recurring_tol_spin.setDecimals(6)
        self.recurring_tol_spin.setRange(0.0, 1.0)
        self.recurring_tol_spin.setSingleStep(0.01)
        self.recurring_tol_spin.setValue(float(orch.recurring_tol))
        conv_layout.addRow("recurring_tol:", self.recurring_tol_spin)

        # median_rel_tol
        self.median_rel_tol_spin = QDoubleSpinBox(conv_group)
        self.median_rel_tol_spin.setDecimals(6)
        self.median_rel_tol_spin.setRange(0.0, 1.0)
        self.median_rel_tol_spin.setSingleStep(0.01)
        self.median_rel_tol_spin.setValue(float(orch.median_rel_tol))
        conv_layout.addRow("median_rel_tol:", self.median_rel_tol_spin)

        # noimprove_N
        self.noimprove_N_spin = QSpinBox(conv_group)
        self.noimprove_N_spin.setMinimum(1)
        self.noimprove_N_spin.setMaximum(1000)
        self.noimprove_N_spin.setValue(int(orch.noimprove_N))
        conv_layout.addRow("noimprove_N:", self.noimprove_N_spin)

        # noimprove_eps
        self.noimprove_eps_spin = QDoubleSpinBox(conv_group)
        self.noimprove_eps_spin.setDecimals(6)
        self.noimprove_eps_spin.setRange(0.0, 1.0)
        self.noimprove_eps_spin.setSingleStep(0.01)
        self.noimprove_eps_spin.setValue(float(orch.noimprove_eps))
        conv_layout.addRow("noimprove_eps:", self.noimprove_eps_spin)

        # stability_N
        self.stability_N_spin = QSpinBox(conv_group)
        self.stability_N_spin.setMinimum(1)
        self.stability_N_spin.setMaximum(1000)
        self.stability_N_spin.setValue(int(orch.stability_N))
        conv_layout.addRow("stability_N:", self.stability_N_spin)

        # stability_std
        self.stability_std_spin = QDoubleSpinBox(conv_group)
        self.stability_std_spin.setDecimals(6)
        self.stability_std_spin.setRange(0.0, 1.0)
        self.stability_std_spin.setSingleStep(0.01)
        self.stability_std_spin.setValue(float(orch.stability_std))
        conv_layout.addRow("stability_std:", self.stability_std_spin)

        # done_on_streak_successes
        self.done_streak_succ_spin = QSpinBox(conv_group)
        self.done_streak_succ_spin.setMinimum(1)
        self.done_streak_succ_spin.setMaximum(1000)
        self.done_streak_succ_spin.setValue(int(orch.done_on_streak_successes))
        conv_layout.addRow("done_on_streak_successes:", self.done_streak_succ_spin)

        # done_on_streak_length
        self.done_streak_len_spin = QSpinBox(conv_group)
        self.done_streak_len_spin.setMinimum(1)
        self.done_streak_len_spin.setMaximum(1000)
        self.done_streak_len_spin.setValue(int(orch.done_on_streak_length))
        conv_layout.addRow("done_on_streak_length:", self.done_streak_len_spin)

        # damping factor λ
        self.lambda_spin = QDoubleSpinBox(conv_group)
        self.lambda_spin.setDecimals(6)
        self.lambda_spin.setRange(0.0, 10.0)
        self.lambda_spin.setSingleStep(0.1)
        self.lambda_spin.setValue(float(orch.λ))
        conv_layout.addRow("damping factor (λ):", self.lambda_spin)

        main_layout.addWidget(conv_group)

        # ---------- Flags group ----------
        flags_group = QGroupBox("Indexamajig / xgandalf / integration flags", container)
        flags_layout = QVBoxLayout(flags_group)

        self.flags_edit = QPlainTextEdit(flags_group)
        # Free text, pre-filled from DEFAULT_FLAGS
        self.flags_edit.setPlainText(" ".join(orch.DEFAULT_FLAGS))

        flags_layout.addWidget(QLabel("Flags passed to --flags (free text):", flags_group))
        flags_layout.addWidget(self.flags_edit)

        main_layout.addWidget(flags_group)

        # ---------- Bottom run + stop buttons ----------
        bottom_layout = QHBoxLayout()
        self.run_button = QPushButton("Run orchestration", container)
        self.run_button.clicked.connect(self.on_run_clicked)
        bottom_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop", container)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        self.stop_button.setEnabled(False)
        bottom_layout.addWidget(self.stop_button)

        bottom_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

        # ---------- Progress bar + explanation ----------
        
        self.progress_label = QLabel(
            "Progress (run_sh.py indexing events; only updates while run_sh.py is running):",
            container,
        )
        main_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar(container)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # just percentage inside bar
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.progress_start_time = None

        # ---------- Log output ----------
        self.log_edit = QPlainTextEdit(container)
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_edit.setMinimumHeight(300)  # make it bigger by default
        main_layout.addWidget(self.log_edit, stretch=1)

    # ---------- Path normalization helper ----------

    def _normalize_path(self, path: str) -> str:
        """Expand ~ and make path absolute."""
        return os.path.abspath(os.path.expanduser(path))

    # ---------- Browsers ----------

    def browse_run_root(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select run root directory",
            self.run_root_edit.text() or ".",
        )
        if path:
            self.run_root_edit.setText(path)

    def browse_geom(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .geom file",
            self.geom_edit.text() or ".",
            "Geom files (*.geom);;All files (*)",
        )
        if path:
            self.geom_edit.setText(path)

    def browse_cell(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .cell file",
            self.cell_edit.text() or ".",
            "Cell files (*.cell);;All files (*)",
        )
        if path:
            self.cell_edit.setText(path)

    def add_h5_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select HDF5 files",
            ".",
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not paths:
            return
        current = self.h5_edit.toPlainText().strip()
        lines = [line for line in current.splitlines() if line.strip()]
        lines.extend(paths)
        self.h5_edit.setPlainText("\n".join(lines))

    def clear_h5_files(self):
        self.h5_edit.clear()

    # ---------- Log handling ----------

    def append_text(self, text: str):
        # Append text and scroll to bottom
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.log_edit.insertPlainText(text)
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

    # ---------- Stop handling (flag + run_py monkeypatch) ----------

    def _patch_run_py_for_stop_and_progress(self, worker: OrchestratorWorker):
        """
        Monkey-patch orch.run_py so that it:
          - Checks self._stop_requested
          - Terminates child subprocesses when stop is requested
          - Emits progress signals for run_sh.py
        """
        if self._orig_run_py is not None:
            # already patched
            return

        self._orig_run_py = orch.run_py
        window = self
        w = worker

        def gui_run_py(script: str, args, check: bool = True) -> int:
            is_run_sh = (script == "run_sh.py")

            if is_run_sh:
                cmd = ["python3", "-u", script, *args]
            else:
                cmd = ["python3", script, *args]

            if window._stop_requested:
                raise SystemExit("[GUI] Stop requested before starting subprocess")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if not is_run_sh:
                try:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        if window._stop_requested:
                            proc.terminate()
                            break
                finally:
                    proc.stdout.close()
                rc = proc.wait()
                if check and rc != 0 and not window._stop_requested:
                    raise SystemExit(f"[ERR] {script} exited with {rc}")
                return rc

            # --- Special Case: run_sh.py with GUI progress bar ---
            total_events = None
            done_events = 0
            event_re = re.compile(r"\brunning\s+(\d+)\s+event\b", re.I)

            try:
                for line in proc.stdout:
                    # Detect completed event marker
                    if "__EVENT_DONE__" in line:
                        done_events += 1
                        if total_events is not None:
                            w.progress_step.emit(1)
                        if window._stop_requested:
                            proc.terminate()
                            break
                        continue

                    # Normal line output
                    print(line, end="", flush=True)

                    if window._stop_requested:
                        proc.terminate()
                        break

                    # Detect total number of events
                    m = event_re.search(line)
                    if m and total_events is None:
                        total_events = int(m.group(1))
                        w.progress_init.emit(total_events)
                        continue

            finally:
                proc.stdout.close()
                # Tell GUI to mark progress as done
                w.progress_done.emit()

            rc = proc.wait()
            if check and rc != 0 and not window._stop_requested:
                raise SystemExit(f"[ERR] {script} exited with {rc}")
            return rc

        orch.run_py = gui_run_py

    def _unpatch_run_py(self):
        if self._orig_run_py is not None:
            orch.run_py = self._orig_run_py
            self._orig_run_py = None

    # ---------- Running orchestrator ----------

    def on_run_clicked(self):
        if self._running:
            QMessageBox.warning(
                self,
                "Orchestrator running",
                "An orchestration run is already in progress.",
            )
            return

        argv = self.build_argv()
        if argv is None:
            # build_argv already showed an error
            return

        self.log_edit.clear()
        self.append_text("[GUI] Starting orchestration with arguments:\n")
        self.append_text(" ".join(["ici_orchestrator.py"] + argv) + "\n\n")

        self._running = True
        self._stop_requested = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(True)

        self._thread = QThread(self)
        self._worker = OrchestratorWorker(argv)
        self._worker.moveToThread(self._thread)

        # Patch run_py so we can stop ongoing subprocesses and get progress
        self._patch_run_py_for_stop_and_progress(self._worker)

        self._thread.started.connect(self._worker.run)
        self._worker.text_ready.connect(self.append_text)
        self._worker.finished.connect(self.on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        # Progress connections
        self._worker.progress_init.connect(self.on_progress_init)
        self._worker.progress_step.connect(self.on_progress_step)
        self._worker.progress_done.connect(self.on_progress_done)

        self._thread.start()

    def on_stop_clicked(self):
        if not self._running:
            return
        self._stop_requested = True
        self.stop_button.setEnabled(False)
        self.append_text("\n[GUI] Stop requested. Will abort after current step.\n")

    def on_worker_finished(self, exit_code: int):
        self._running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._unpatch_run_py()
        self.append_text(f"\n[GUI] Orchestrator finished with exit code {exit_code}.\n")

        if exit_code != 0 and not self._stop_requested:
            QMessageBox.warning(
                self,
                "Orchestrator finished with error",
                f"Orchestrator exited with code {exit_code}. "
                f"Check the log above and the orchestrator.log file for details.",
            )

    def _format_eta(self, seconds: float) -> str:
        if seconds is None or seconds <= 0 or seconds == float("inf"):
            return "estimating..."
        total = int(seconds)
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h {m:02d}m {s:02d}s"
        else:
            return f"{m:02d}m {s:02d}s"


    # ---------- Progress bar slots ----------

    def on_progress_init(self, total: int):
        # Called when run_sh.py prints "running N event"
        self.progress_start_time = time.time()

        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")  # keep bar text simple

        self.progress_label.setText(
            f"[run_sh] Indexing: 0% (0 / {total} events), ETA estimating..."
        )

    def on_progress_step(self, step: int):
        # Called once per "__EVENT_DONE__"
        new_val = min(self.progress_bar.value() + step, self.progress_bar.maximum())
        self.progress_bar.setValue(new_val)

        total = self.progress_bar.maximum()
        done = new_val
        if total <= 0:
            return

        percent = 100.0 * done / total

        # ETA
        if self.progress_start_time is not None and done > 0:
            elapsed = time.time() - self.progress_start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else None
            eta_str = self._format_eta(eta)
        else:
            eta_str = "estimating..."

        self.progress_label.setText(
            f"[run_sh] Indexing: {percent:5.1f}% ({done} / {total} events), ETA {eta_str}"
        )

    def on_progress_done(self):
        # Called when run_sh.py subprocess exits
        total = self.progress_bar.maximum()
        if total > 0:
            self.progress_bar.setValue(total)
            self.progress_label.setText(
                f"[run_sh] Indexing finished ({total} / {total} events)"
            )
        else:
            self.progress_label.setText("Progress (last run_sh.py finished)")
        self.progress_start_time = None


    # def build_argv(self):
    #     """
    #     Build a CLI-style argv list for orch.main from the GUI state.
    #     Instead of passing --flags on the CLI (which is awkward because
    #     the flags themselves begin with "-"), we override orch.DEFAULT_FLAGS
    #     directly based on the free-text field.
    #     """
    #     run_root = self.run_root_edit.text().strip()
    #     geom = self.geom_edit.text().strip()
    #     cell = self.cell_edit.text().strip()
    
    def build_argv(self):
        """
        Build a CLI-style argv list for orch.main from the GUI state.
        Instead of passing --flags on the CLI (which is awkward because
        the flags themselves begin with "-"), we override orch.DEFAULT_FLAGS
        directly based on the free-text field.
        """
        run_root = self._normalize_path(self.run_root_edit.text().strip())
        geom     = self._normalize_path(self.geom_edit.text().strip())
        cell     = self._normalize_path(self.cell_edit.text().strip())


        if not run_root:
            QMessageBox.critical(self, "Missing run root", "Please specify a run root directory.")
            return None
        if not geom:
            QMessageBox.critical(self, "Missing geom", "Please specify a .geom file.")
            return None
        if not cell:
            QMessageBox.critical(self, "Missing cell", "Please specify a .cell file.")
            return None

        # HDF5 sources: one per non-empty line
        # h5_lines = [line.strip() for line in self.h5_edit.toPlainText().splitlines() if line.strip()]
        # if not h5_lines:
        #     # If empty, fall back to orchestrator default
        #     h5_lines = list(orch.DEFAULT_H5)


        raw_h5_lines = [line.strip() for line in self.h5_edit.toPlainText().splitlines() if line.strip()]
        if not raw_h5_lines:
            raw_h5_lines = list(orch.DEFAULT_H5)

        h5_lines = [self._normalize_path(p) for p in raw_h5_lines]

        max_iters = self.max_iters_spin.value()
        jobs = self.jobs_spin.value()

        # Convergence parameters
        radius_mm = self.radius_spin.value()
        min_spacing_mm = self.min_spacing_spin.value()
        N_conv = self.N_conv_spin.value()
        recurring_tol = self.recurring_tol_spin.value()
        median_rel_tol = self.median_rel_tol_spin.value()
        noimprove_N = self.noimprove_N_spin.value()
        noimprove_eps = self.noimprove_eps_spin.value()
        stability_N = self.stability_N_spin.value()
        stability_std = self.stability_std_spin.value()
        done_streak_succ = self.done_streak_succ_spin.value()
        done_streak_len = self.done_streak_len_spin.value()
        lambda_val = self.lambda_spin.value()

        # Flags: free text → override orch.DEFAULT_FLAGS directly
        flags_text = self.flags_edit.toPlainText().strip()
        if flags_text:
            try:
                flag_list = shlex.split(flags_text)
            except ValueError as e:
                QMessageBox.critical(
                    self,
                    "Invalid flags",
                    f"Could not parse flags text:\n{e}",
                )
                return None
            # Override the module-level default flags used by ici_orchestrator
            orch.DEFAULT_FLAGS = flag_list

        # Now build argv WITHOUT any explicit --flags, so argparse
        # just uses orch.DEFAULT_FLAGS (which we've possibly overridden).
        argv = []

        argv.extend(["--run-root", run_root])
        argv.extend(["--geom", geom])
        argv.extend(["--cell", cell])

        # HDF5 sources
        argv.append("--h5")
        argv.extend(h5_lines)

        # Max iterations & jobs
        argv.extend(["--max-iters", str(max_iters)])
        argv.extend(["--jobs", str(jobs)])

        # Convergence arguments (mirror ici_orchestrator.main)
        argv.extend(["--radius-mm", str(radius_mm)])
        argv.extend(["--min-spacing-mm", str(min_spacing_mm)])
        argv.extend(["--N-conv", str(N_conv)])
        argv.extend(["--recurring-tol", str(recurring_tol)])
        argv.extend(["--median-rel-tol", str(median_rel_tol)])
        argv.extend(["--noimprove-N", str(noimprove_N)])
        argv.extend(["--noimprove-eps", str(noimprove_eps)])
        argv.extend(["--stability-N", str(stability_N)])
        argv.extend(["--stability-std", str(stability_std)])
        argv.extend(["--done-on-streak-successes", str(done_streak_succ)])
        argv.extend(["--done-on-streak-length", str(done_streak_len)])
        argv.extend(["--damping-factor", str(lambda_val)])

        return argv

    # ---------- Close handling ----------

    def closeEvent(self, event):
        # On close, request stop if something is running.
        if self._running:
            self._stop_requested = True
            self.append_text("\n[GUI] Window closed: stop requested.\n")
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = OrchestratorMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
