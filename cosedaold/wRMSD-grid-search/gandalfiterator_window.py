#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gandalfiterator_window.py (adaptive, lean GUI) — PyQt6 version

A lightweight PyQt6 GUI for the Adaptive Center-Shift Optimization pipeline:
- Choose run root, .geom, .cell, and HDF5 files/directories
- Set adaptive knobs (R_px, s_init_px, K_dir, ...)
- Enter arbitrary indexamajig flags (verbatim pass-through, e.g. "-j 32 --peaks peaks.conf")
- Start the run in a background thread
- Live tail the run_root/log.jsonl and display status
- Show final merged stream path

Dependencies: PyQt6 (pip install pyqt6)

Integrates with:
- gandalfiterator.Params, gandalf_adaptive
"""

from __future__ import annotations
import os
import sys
import json
import shlex
import subprocess
from typing import List, Optional
from pathlib import Path


from PyQt6 import QtCore, QtGui, QtWidgets

# Local imports (ensure these are on PYTHONPATH)
from gandalfiterator import Params, gandalf_adaptive


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _gather_h5_sources(paths: List[str]) -> List[str]:
    out = set()
    for p in paths:
        ap = _abs(p)
        if os.path.isfile(ap) and ap.lower().endswith(".h5"):
            out.add(ap)
        elif os.path.isdir(ap):
            for root, _, files in os.walk(ap):
                for fn in files:
                    if fn.lower().endswith(".h5"):
                        out.add(os.path.join(root, fn))
    return sorted(out)


class LogTailer(QtCore.QThread):
    """Tails a JSONL log file and emits lines as they arrive."""
    lineRead = QtCore.pyqtSignal(str)
    finishedTailing = QtCore.pyqtSignal()

    def __init__(self, log_path: str, parent=None):
        super().__init__(parent)
        self.log_path = log_path
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        waited = 0.0
        while not self._stop and not os.path.exists(self.log_path):
            self.msleep(250)
            waited += 0.25
            if waited > 3600:
                break

        if not os.path.exists(self.log_path):
            self.finishedTailing.emit()
            return

        with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
            while not self._stop:
                pos = f.tell()
                line = f.readline()
                if not line:
                    self.msleep(250)
                    f.seek(pos)
                else:
                    self.lineRead.emit(line.rstrip("\n"))
        self.finishedTailing.emit()


class WorkerThread(QtCore.QThread):
    """Runs gandalf_adaptive in background."""
    startedRun = QtCore.pyqtSignal()
    finishedRun = QtCore.pyqtSignal(str)  # merged stream path (or "")

    def __init__(self, run_root: str, geom: str, cell: str, sources: List[str],
                 params: Params, idx_flags: List[str], parent=None):
        super().__init__(parent)
        self.run_root = run_root
        self.geom = geom
        self.cell = cell
        self.sources = sources
        self.params = params
        self.idx_flags = idx_flags

    def run(self):
        self.startedRun.emit()
        try:
            merged = gandalf_adaptive(
                run_root=self.run_root,
                geom_path=self.geom,
                cell_path=self.cell,
                h5_sources=self.sources,
                params=self.params,
                indexamajig_flags_passthrough=self.idx_flags,
            )
            self.finishedRun.emit(merged or "")
        except Exception:
            self.finishedRun.emit("")

class FileTailer(QtCore.QObject):
    """
    Lightweight file tailer that polls a .err file and emits 'processed' counts
    whenever it sees lines like: 'XX images processed, YY hits ...'
    """
    progressParsed = QtCore.pyqtSignal(int)   # emits 'processed' (int)

    def __init__(self, path: str, poll_ms: int = 250, parent=None):
        super().__init__(parent)
        self.path = path
        self._pos = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)
        import re
        self._re = re.compile(r"^\s*([0-9]+)\s+images\s+processed\b", re.IGNORECASE)

    def start(self):
        self._pos = 0
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _poll(self):
        import os
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                data = f.read()
                if not data:
                    return
                self._pos = f.tell()
        except Exception:
            return

        for raw in data.splitlines():
            try:
                line = raw.decode("utf-8", errors="ignore")
                m = self._re.search(line)
                if m:
                    self.progressParsed.emit(int(m.group(1)))
            except Exception:
                pass


class RunRow(QtWidgets.QWidget):
    """
    A single row in the 'Run Progress' panel: label + QProgressBar + meta label.
    """
    def __init__(self, run_key: str, parent=None):
        super().__init__(parent)
        self.run_key = run_key
        self.total: int | None = None
        self.processed: int = 0

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(8)

        self.lbl = QtWidgets.QLabel(run_key)
        self.lbl.setMinimumWidth(160)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setMinimum(0)
        self.bar.setTextVisible(True)

        self.meta = QtWidgets.QLabel("")
        self.meta.setMinimumWidth(140)

        h.addWidget(self.lbl)
        h.addWidget(self.bar, 1)
        h.addWidget(self.meta)

    def set_total(self, total: int):
        self.total = max(1, int(total))
        self.bar.setMaximum(self.total)
        self._render()

    def update_processed(self, processed: int):
        # Make monotonic: indexamajig may sometimes log duplicates
        self.processed = max(self.processed, int(processed))
        self.bar.setValue(self.processed)
        self._render()

    def _render(self):
        if self.total:
            pct = int(round(100.0 * self.processed / max(1, self.total)))
            self.bar.setFormat(f"%p%   ({self.processed}/{self.total})")
            self.meta.setText(f"{pct}% • {self.run_key}")
        else:
            self.bar.setFormat(f"{self.processed} done")
            self.meta.setText(f"{self.processed} • {self.run_key}")

    def mark_done_if_complete(self):
        if self.total and self.processed >= self.total:
            self.meta.setText(f"Done • {self.run_key}")

    def mark_failed(self):
        self.meta.setText(f"Failed • {self.run_key}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Center Refinement — Gandalf (PyQt6)")
        self.setMinimumSize(980, 720)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---- Paths group ----
        grp_paths = QtWidgets.QGroupBox("Paths")
        layout.addWidget(grp_paths)
        form_paths = QtWidgets.QFormLayout(grp_paths)

        self._run_stats = {}  # key: run_id, val: dict(counts/medians)

        self.runRootEdit = QtWidgets.QLineEdit()
        self.geomEdit = QtWidgets.QLineEdit()
        self.cellEdit = QtWidgets.QLineEdit()
        self.h5List = QtWidgets.QListWidget()
        self.h5List.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        # ---- Defaults for debugging (comment out later) ----
        DEBUG = True
        if DEBUG:
            # DESKTOP
            # default_root = ("/home/bubl3932/files/grid-search-wRMSD-optimization")
            # self.runRootEdit.setText(default_root)
            # self.geomEdit.setText(default_root + "/MFM.geom")
            # self.cellEdit.setText(default_root + "/MFM.cell")
            # self.h5List.addItem(default_root + "/MFM.h5")
            # LAPTOP
            default_root = ("/Users/xiaodong/Desktop/simulations/MFM300-VIII_tI/sim_002")
            self.runRootEdit.setText(default_root)
            self.geomEdit.setText(default_root + "/MFM300-VIII.geom")
            self.cellEdit.setText(default_root + "/MFM300-VIII.cell")
            self.h5List.addItem(default_root + "/sim.h5")
        # ---------------------------------------------------

        btnRunRoot = QtWidgets.QPushButton("Browse…")
        btnGeom = QtWidgets.QPushButton("Browse…")
        btnCell = QtWidgets.QPushButton("Browse…")
        btnAddH5 = QtWidgets.QPushButton("Add .h5 or Folder…")
        btnClearH5 = QtWidgets.QPushButton("Clear")

        hboxRun = QtWidgets.QWidget()
        hb1 = QtWidgets.QHBoxLayout(hboxRun); hb1.setContentsMargins(0,0,0,0)
        hb1.addWidget(self.runRootEdit, 1); hb1.addWidget(btnRunRoot)

        hboxGeom = QtWidgets.QWidget()
        hb2 = QtWidgets.QHBoxLayout(hboxGeom); hb2.setContentsMargins(0,0,0,0)
        hb2.addWidget(self.geomEdit, 1); hb2.addWidget(btnGeom)

        hboxCell = QtWidgets.QWidget()
        hb3 = QtWidgets.QHBoxLayout(hboxCell); hb3.setContentsMargins(0,0,0,0)
        hb3.addWidget(self.cellEdit, 1); hb3.addWidget(btnCell)

        hboxH5Btns = QtWidgets.QWidget()
        hb4 = QtWidgets.QHBoxLayout(hboxH5Btns); hb4.setContentsMargins(0,0,0,0)
        hb4.addWidget(btnAddH5); hb4.addWidget(btnClearH5); hb4.addStretch(1)

        form_paths.addRow("Run root:", hboxRun)
        form_paths.addRow("Geom (.geom):", hboxGeom)
        form_paths.addRow("Cell (.cell):", hboxCell)
        form_paths.addRow("HDF5 sources:", self.h5List)
        form_paths.addRow("", hboxH5Btns)

        # ---- Parameters ----
        grp_params = QtWidgets.QGroupBox("Adaptive Parameters")
        layout.addWidget(grp_params)
        grid = QtWidgets.QGridLayout(grp_params)
        grid.setContentsMargins(9,9,9,9)


        def spinD(default, step, decimals=3, minimum=-1e9, maximum=1e9):
            s = QtWidgets.QDoubleSpinBox()
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setRange(minimum, maximum)
            s.setValue(default)
            s.setMaximumWidth(120) 
            return s

        def spinI(default, step, minimum=1, maximum=10**9):
            s = QtWidgets.QSpinBox()
            s.setRange(minimum, maximum)
            s.setSingleStep(step)
            s.setValue(default)
            s.setMaximumWidth(120)
            return s

        self.R_px = spinD(1.0, 0.1, decimals=3, minimum=0.0)
        self.s_init_px = spinD(0.2, 0.05, decimals=3, minimum=0.0)
        self.K_dir = spinI(10, 1, minimum=1, maximum=360)
        self.s_refine_px = spinD(0.5, 0.1, decimals=3, minimum=0.0)
        self.s_min_px = spinD(0.1, 0.05, decimals=3, minimum=0.0)
        self.eps_rel = spinD(0.007, 0.001, decimals=4, minimum=0.0, maximum=0.5)
        self.N_eval_max = spinI(16, 1, minimum=1, maximum=1000)
        self.tie_tol_rel = spinD(0.01, 0.001, decimals=3, minimum=0.0, maximum=0.2)

        self.chk8 = QtWidgets.QCheckBox("8-connected neighbors")
        self.chk8.setChecked(True)
        self.chkDirectional = QtWidgets.QCheckBox("Directional prioritization")
        self.chkDirectional.setChecked(True)

        labels = ["R_px", "s_init_px", "K_dir", "s_refine_px", "s_min_px",
                  "eps_rel", "N_eval_max", "tie_tol_rel"]
        widgets = [self.R_px, self.s_init_px, self.K_dir, self.s_refine_px,
                   self.s_min_px, self.eps_rel, self.N_eval_max, self.tie_tol_rel]
        for r, (lab, wid) in enumerate(zip(labels, widgets)):
            grid.addWidget(QtWidgets.QLabel(lab + ":"), r, 0)
            grid.addWidget(wid, r, 1)
        grid.addWidget(self.chk8, len(labels), 0, 1, 1)
        grid.addWidget(self.chkDirectional, len(labels), 1, 1, 1)

        # ---- Indexamajig flags ----
        grp_flags = QtWidgets.QGroupBox("indexamajig flags (e.g., -j 32 --peaks peaks.conf)")
        layout.addWidget(grp_flags)
        vbox_flags = QtWidgets.QVBoxLayout(grp_flags)
        self.flagsEdit = QtWidgets.QLineEdit()
        vbox_flags.addWidget(self.flagsEdit)

        # ---- Actions ----
        hboxBtns = QtWidgets.QHBoxLayout()
        layout.addLayout(hboxBtns)
        self.btnStart = QtWidgets.QPushButton("Start Adaptive Run")
        self.btnStart.setDefault(True)
        self.btnQuit = QtWidgets.QPushButton("Quit")
        hboxBtns.addStretch(1)
        hboxBtns.addWidget(self.btnStart)
        hboxBtns.addWidget(self.btnQuit)

        # ---- Log panel ----
        grp_log = QtWidgets.QGroupBox("Run Log (live)")
        layout.addWidget(grp_log, 1)
        vbox_log = QtWidgets.QVBoxLayout(grp_log)
        self.logView = QtWidgets.QPlainTextEdit()
        self.logView.setReadOnly(True)
        self.logView.setMaximumBlockCount(20000)
        vbox_log.addWidget(self.logView)

        self.statusBar().showMessage("Ready.")

        # ---- Signals ----
        btnRunRoot.clicked.connect(self._chooseRunRoot)
        btnGeom.clicked.connect(self._chooseGeom)
        btnCell.clicked.connect(self._chooseCell)
        btnAddH5.clicked.connect(self._addH5OrDir)
        btnClearH5.clicked.connect(self._clearH5List)
        self.btnStart.clicked.connect(self._startRun)
        self.btnQuit.clicked.connect(self.close)

        self.worker: Optional[WorkerThread] = None
        self.tailer: Optional[LogTailer] = None

        self._top_vbox_layout = layout
        self._init_progress_panel()

    def _init_progress_panel(self):
        """Insert a 'Run Progress' scrollable panel above the log."""
        grp_prog = QtWidgets.QGroupBox("Run Progress")
        # We assume `layout` is the top-level VBox in your existing __init__
        # If your variable is named differently, adjust this line accordingly:
        self._top_vbox_layout.addWidget(grp_prog)  # <-- set this reference in __init__

        vbox_prog = QtWidgets.QVBoxLayout(grp_prog)
        self.progScroll = QtWidgets.QScrollArea()
        self.progScroll.setWidgetResizable(True)
        vbox_prog.addWidget(self.progScroll)

        self._progHost = QtWidgets.QWidget()
        self.progScroll.setWidget(self._progHost)

        self._progLayout = QtWidgets.QVBoxLayout(self._progHost)
        self._progLayout.setContentsMargins(6, 6, 6, 6)
        self._progLayout.setSpacing(6)

        # Per-run state (existing)
        self._runRows: dict[str, RunRow] = {}     # run_key -> RunRow
        self._tailers: dict[str, FileTailer] = {} # run_key -> FileTailer

        # NEW: accumulate counts & wRMSD per run (keyed by run_key)
        # schema: { run_key: {"n":int, "idx":int, "wrmsd":List[float], "total": Optional[int]} }
        self._runStats: dict[str, dict] = {}


    def _startErrTailer(self, run_key: str, err_path: str, lst_path: str, total: int | None):
        """Start (or restart) a FileTailer on the given .err file."""
        # Stop existing tailer if present
        old = self._tailers.get(run_key)
        if old:
            try:
                old.stop()
            except Exception:
                pass

        tailer = FileTailer(err_path, parent=self)
        # Connect so each parsed 'processed' count updates the row
        tailer.progressParsed.connect(lambda processed, rk=run_key, lp=lst_path, tot=total:
                                    self._onProgress(rk, processed, tot, lp))
        tailer.start()
        self._tailers[run_key] = tailer

    def _onProgress(self, run_key: str, processed: int, total_hint: int | None, lst_path: str):
        """Update (or create) the UI row for a run."""
        row = self._runRows.get(run_key)
        if not row:
            row = RunRow(run_key)
            self._progLayout.addWidget(row)
            self._runRows[run_key] = row

        # If total unknown, compute once from .lst
        if row.total is None:
            total = total_hint
            if total is None and lst_path:
                try:
                    with open(lst_path, "r") as f:
                        total = sum(1 for ln in f if ln.strip())
                except Exception:
                    total = None
            if total is not None:
                row.set_total(int(total))

        row.update_processed(int(processed))


    # ---- Path pickers ----
    def _chooseRunRoot(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose run root")
        if d:
            self.runRootEdit.setText(d)

    def _chooseGeom(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose .geom", filter="Geom (*.geom);;All Files (*)")
        if fn:
            self.geomEdit.setText(fn)

    def _chooseCell(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose .cell", filter="Cell (*.cell);;All Files (*)")
        if fn:
            self.cellEdit.setText(fn)

    def _addH5OrDir(self):
        dlg = QtWidgets.QFileDialog(self, "Add HDF5 files")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        dlg.setNameFilter("HDF5 (*.h5);;All Files (*)")
        if dlg.exec():
            for p in dlg.selectedFiles():
                self._appendH5Path(p)
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add directory with .h5 files (recursive)")
        if dir_path:
            self._appendH5Path(dir_path)

    def _appendH5Path(self, p: str):
        for i in range(self.h5List.count()):
            if _abs(self.h5List.item(i).text()) == _abs(p):
                return
        self.h5List.addItem(p)

    def _clearH5List(self):
        self.h5List.clear()

    # ---- Run handling ----

    def _ensure_run_bundle(
        self,
        lst_entries,
        geom_path,
        cell_path,
        overlay_path,
        run_dir,
        script_path,
        idx_exec,
        extra_args=None,
    ):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        lst_path = run_dir / Path(overlay_path).with_suffix(".lst").name
        stream_path = run_dir / Path(overlay_path).with_suffix(".stream").name
        script_path = Path(script_path)

        # 1) Write the .lst explicitly
        with open(lst_path, "w") as f:
            for p in lst_entries:
                f.write(str(p) + "\n")

        # 2) Build the exact command (absolute paths)
        cmd = [
            str(Path(idx_exec).resolve()),
            "--lst", str(lst_path.resolve()),
            "--geom", str(Path(geom_path).resolve()),
            "--cell", str(Path(cell_path).resolve()),
            "--out", str(stream_path.resolve()),
        ]
        if extra_args:
            cmd.extend(extra_args)

        # 3) Write a runnable .sh for inspection
        with open(script_path, "w") as sh:
            sh.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            sh.write("echo '[idx] CWD:' $(pwd)\n")
            sh.write("echo '[idx] whoami:' $(whoami)\n")
            sh.write("echo '[idx] running:' " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            sh.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
        os.chmod(script_path, 0o755)

        return {
            "lst": str(lst_path),
            "stream": str(stream_path),
            "script": str(script_path),
            "cmd": cmd,
        }

    def _run_index_command(self, cmd, run_dir):
        proc = subprocess.run(
            cmd,
            cwd=str(run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr


    def _collectParams(self) -> Params:
        return Params(
            R_px=float(self.R_px.value()),
            s_init_px=float(self.s_init_px.value()),
            K_dir=int(self.K_dir.value()),
            s_refine_px=float(self.s_refine_px.value()),
            s_min_px=float(self.s_min_px.value()),
            eps_rel=float(self.eps_rel.value()),
            N_eval_max=int(self.N_eval_max.value()),
            tie_tol_rel=float(self.tie_tol_rel.value()),
            eight_connected=bool(self.chk8.isChecked()),
            directional_refine=bool(self.chkDirectional.isChecked()),
        )

    def _collectSources(self) -> List[str]:
        vals = [self.h5List.item(i).text() for i in range(self.h5List.count())]
        return _gather_h5_sources(vals)

    def _startRun(self):
        run_root = self.runRootEdit.text().strip()
        geom = self.geomEdit.text().strip()
        cell = self.cellEdit.text().strip()
        sources = self._collectSources()
        if not run_root or not geom or not cell or not sources:
            QtWidgets.QMessageBox.warning(self, "Missing inputs",
                                          "Please set run root, geom, cell, and add at least one .h5 or directory.")
            return

        os.makedirs(run_root, exist_ok=True)
        params = self._collectParams()

        flags_str = self.flagsEdit.text().strip()
        try:
            idx_flags = shlex.split(flags_str) if flags_str else []
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Flags error", f"Could not parse flags: {e}")
            return

        log_path = os.path.join(_abs(run_root), "log.jsonl")
        if self.tailer:
            self.tailer.stop()
            self.tailer.wait()
        self.tailer = LogTailer(log_path)
        self.tailer.lineRead.connect(self._onLogLine)
        self.tailer.finishedTailing.connect(lambda: self.statusBar().showMessage("Log tailer finished."))
        self.tailer.start()

        self._setInputsEnabled(False)
        self.logView.appendPlainText("=== Starting adaptive run ===")
        self.statusBar().showMessage("Running…")

        self.worker = WorkerThread(run_root=_abs(run_root),
                                   geom=_abs(geom),
                                   cell=_abs(cell),
                                   sources=sources,
                                   params=params,
                                   idx_flags=idx_flags)
        self.worker.startedRun.connect(lambda: self._onStarted())
        self.worker.finishedRun.connect(self._onFinished)
        self.worker.start()

    def _setInputsEnabled(self, enabled: bool):
        for w in [
            self.runRootEdit, self.geomEdit, self.cellEdit, self.h5List,
            self.R_px, self.s_init_px, self.K_dir, self.s_refine_px, self.s_min_px,
            self.eps_rel, self.N_eval_max, self.tie_tol_rel, self.chk8, self.chkDirectional,
            self.flagsEdit
        ]:
            w.setEnabled(enabled)
        self.btnStart.setEnabled(enabled)

    def _onStarted(self):
        self.logView.appendPlainText("Run thread started.")

    def _onFinished(self, merged_path: str):
        self._setInputsEnabled(True)
        self.statusBar().showMessage("Finished.")
        if self.tailer:
            QtCore.QTimer.singleShot(1000, self.tailer.stop)

        if merged_path:
            self.logView.appendPlainText(f"✓ Merged stream: {merged_path}")
            QtWidgets.QMessageBox.information(self, "Run complete", f"Merged stream written to:\n{merged_path}")
        else:
            self.logView.appendPlainText("No merged stream produced (no FINAL images or failure).")
            QtWidgets.QMessageBox.warning(self, "Run finished", "No merged stream produced.")
            
    def _onLogLine(self, line: str):
        """Handle one line from log.jsonl."""
        # Always echo to the log pane (pretty if JSON)
        obj = None
        try:
            obj = json.loads(line)
            pretty = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
            self.logView.appendPlainText(pretty)
        except Exception:
            self.logView.appendPlainText(line)

        if not isinstance(obj, dict):
            return

        t = obj.get("type")

        if t == "run_started":
            lst = obj.get("lst", "")
            err = obj.get("stderr", "")
            total = obj.get("total")
            run_key = obj.get("run_id") or Path(lst).stem or "run"

            # Ensure row exists
            row = self._runRows.get(run_key)
            if not row:
                row = RunRow(run_key)
                self._progLayout.addWidget(row)
                self._runRows[run_key] = row
            if total is not None:
                try:
                    row.set_total(int(total))
                except Exception:
                    pass

            # Init stats for this run
            self._runStats[run_key] = {"n": 0, "idx": 0, "wrmsd": [], "total": int(total) if total is not None else None}

            # Start tailing the .err for live progress
            self._startErrTailer(run_key, err, lst, total)

        elif t == "candidate_result":
            # Derive run_key from the stream path (engine uses basename(stream) as run_id)
            import os, statistics
            stream_path = obj.get("stream", "") or ""
            run_key = os.path.splitext(os.path.basename(stream_path))[0] or "run"

            st = self._runStats.setdefault(run_key, {"n": 0, "idx": 0, "wrmsd": [], "total": None})

            # Update counts
            st["n"] += 1
            if obj.get("indexed"):
                st["idx"] += 1
                wr = obj.get("wrmsd")
                if isinstance(wr, (int, float)):
                    try:
                        st["wrmsd"].append(float(wr))
                    except Exception:
                        pass

            # Update the corresponding row's meta text
            row = self._runRows.get(run_key)
            if row:
                # prefer row.total if set; otherwise use stats.total; fallback to "?"
                total_disp = row.total or st.get("total") or "?"
                med = None
                try:
                    if st["wrmsd"]:
                        med = statistics.median(st["wrmsd"])
                except Exception:
                    med = None
                txt = f"{row.processed}/{total_disp} • indexed {st['idx']}/{st['n']}"
                if med is not None:
                    txt += f" • med wRMSD={med:.3f}"
                row.meta.setText(txt)

        elif t == "indexamajig_exec":
            # Process end marker; mark done/failed for any visible rows (keep your logic)
            try:
                rc = int(obj.get("returncode") or 0)
            except Exception:
                rc = 1
            for row in self._runRows.values():
                if rc == 0:
                    row.mark_done_if_complete()
                else:
                    row.mark_failed()


def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
