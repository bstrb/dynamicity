#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexIntegrate (SerialED Grid) – COSEDA-style window (patched)
- Keeps the original window, run-dir layout, batch, progress/ETA and tools
- Replaces the placeholder gandalf_iterator with a full implementation:
  • grid of centre shifts (px) → mm using 'res' from .geom
  • per-pass HDF5 perturbation of det_shift_x_mm / det_shift_y_mm
  • run indexamajig with live (unbuffered) output
  • revert HDF5 shifts after each pass (also on exceptions)
  • write streams under <run_dir>/streams/<out_base>_<dx>_<dy>.stream
  • optional "[XY-PASS] i/N ..." marker to help pass counting in UI
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import glob
import shlex
import queue
import signal
import shutil
import tempfile
import textwrap
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Third-party
import h5py  # required by iterator

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QListWidgetItem, QListWidget, QGroupBox, QTabWidget, QProgressBar, QSplitter
)

# ==========================
# VENDORED GANDALF ITERATOR
# ==========================
# The previous version had a placeholder here.
# This is a full, robust implementation adapted from your GI prototype,
# but aligned to this window’s run_dir/streams layout.

def _cleanup_temp_dirs() -> None:
    """Remove common indexamajig scratch and mille files from CWD."""
    try:
        for d in glob.glob("indexamajig*"):
            p = Path(d)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        for f in glob.glob("mille-data.bin*"):
            p = Path(f)
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def _extract_res_px_per_m(geom_file_path: str) -> float:
    """
    Return 'res' from .geom (pixels per meter).
    Example line:  res = 17857.14285714286
    """
    with open(geom_file_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith(";") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == "res":
                return float(v.strip())
    raise ValueError("Could not find 'res =' in geometry file (needed to convert px→mm).")


def _grid_points_in_circle(max_radius: float, step: float) -> List[Tuple[float, float]]:
    """
    Generate (dx,dy) grid points inside a circle of radius `max_radius`
    on a square lattice with spacing `step`, centered on (0,0).
    Sorted by radial distance (origin first).
    """
    if max_radius <= 0.0:
        return [(0.0, 0.0)]
    if step <= 0.0:
        step = max_radius  # degenerate -> single ring

    # decimals to keep based on step
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    pts: List[Tuple[float, float]] = []
    k = int(math.ceil(max_radius / step))
    for ix in range(-k, k + 1):
        for iy in range(-k, k + 1):
            x = round(ix * step, decimals)
            y = round(iy * step, decimals)
            if x*x + y*y <= max_radius*max_radius + 1e-12:
                pts.append((x, y))
    pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    return pts


def _ensure_listfile_for_input(run_dir: Path, input_dir: Path) -> Path:
    """
    Prefer the run's input.lst if already written by the parent window.
    Otherwise, create one (top-level *.h5, absolute paths) inside run_dir.
    """
    lst = run_dir / "input.lst"
    try:
        if lst.exists():
            # Use if it contains at least one non-empty path line
            with open(lst, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        return lst
        # (Re)create from input_dir *.h5 (non-recursive, matching parent’s counter)
        lines: List[str] = []
        for h5 in sorted(input_dir.glob("*.h5")):
            lines.append(str(h5.resolve()))
        lst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return lst
    except Exception as e:
        raise RuntimeError(f"Failed to prepare list file at {lst}: {e}")


def _perturb_det_shifts(listfile: Path, x_mm: float, y_mm: float) -> None:
    """
    Add (x_mm, y_mm) to datasets:
      /entry/data/det_shift_x_mm , /entry/data/det_shift_y_mm
    for each HDF5 path listed in listfile.
    """
    with open(listfile, "r", encoding="utf-8", errors="replace") as f:
        h5s = [ln.strip() for ln in f if ln.strip()]
    for h5path in h5s:
        try:
            with h5py.File(h5path, "r+") as h5f:
                if "entry/data/det_shift_x_mm" not in h5f or "entry/data/det_shift_y_mm" not in h5f:
                    # Gracefully skip if not present
                    continue
                x_ds = h5f["entry/data/det_shift_x_mm"][...]
                y_ds = h5f["entry/data/det_shift_y_mm"][...]
                x_ds += x_mm
                y_ds += y_mm
                h5f["entry/data/det_shift_x_mm"][:] = x_ds
                h5f["entry/data/det_shift_y_mm"][:] = y_ds
        except Exception:
            # Keep going; we will still try to revert
            pass
    print(f"  => Applied shift: Δx={x_mm:.6f} mm, Δy={y_mm:.6f} mm", flush=True)


def _run_indexamajig(
    geomfile: Path,
    listfile: Path,
    cellfile: Path,
    output_stream: Path,
    threads: int,
    extra_flags: Optional[List[str]] = None,
) -> None:
    cmd = [
        "indexamajig",
        "-g", str(geomfile),
        "-i", str(listfile),
        "-o", str(output_stream),
        "-p", str(cellfile),
        "-j", str(threads),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    # inherit stdout/stderr; no capture => lines go straight to parent QProcess
    subprocess.run(cmd, check=True)


def gandalf_iterator(
    geomfile_path: str,
    cellfile_path: str,
    input_path: str,
    output_file_base: str,
    num_threads: int,
    max_radius: float = 0.0,
    step: float = 0.1,
    extra_flags: Optional[List[str]] = None,
) -> int:
    """
    Multi-pass iterator: for each (dx,dy) in a circular grid around (0,0),
    perturb HDF5 detector shifts by Δ (in mm), run indexamajig, then revert.
    Stream paths: <run_dir>/streams/<base>_<dx>_<dy>.stream
    """
    extra_flags = list(extra_flags or [])

    # Derive run_dir from the passed outbase path:  <run_dir>/streams/<base>
    outbase = Path(output_file_base).resolve()
    streams_dir = outbase.parent
    run_dir = streams_dir.parent
    input_dir = Path(input_path).resolve()

    # Prepare list file (re-use run_dir/input.lst if present)
    listfile = _ensure_listfile_for_input(run_dir, input_dir)

    # Convert px → mm from geom 'res' (px/m)
    res_px_per_m = _extract_res_px_per_m(geomfile_path)
    mm_per_px = 1000.0 / float(res_px_per_m)

    # Enumerate grid points (sorted center-out)
    pts = _grid_points_in_circle(max_radius=max(0.0, float(max_radius)),
                                 step=max(0.0, float(step)) if step else 0.1)
    total_passes = max(1, len(pts))

    # Inform parent UI (optional explicit pass marker)
    print(f"[XY-PASS] 0/{total_passes} init", flush=True)
    print(f"[grid] max_radius={max_radius:g} px, step={step:g} px → passes={total_passes}", flush=True)
    print(f"[geom] res={res_px_per_m:g} px/m ⇒ {mm_per_px:.9f} mm/px", flush=True)
    print(f"[i/o] listfile={listfile}  streams_dir={streams_dir}", flush=True)

    # Run all passes
    for i, (dx_px, dy_px) in enumerate(pts, start=1):
        # Build output path <base>_<dx>_<dy>.stream ; keep raw values in name
        out_stream = streams_dir / f"{outbase.name}_{dx_px}_{dy_px}.stream"
        # Emit pass marker for parent progress parser
        print(f"[XY-PASS] {i}/{total_passes} dx={dx_px} dy={dy_px}", flush=True)

        # Convert to mm for HDF5 perturbation
        sx_mm = dx_px * mm_per_px
        sy_mm = dy_px * mm_per_px

        # Apply → run → revert (revert in finally to guarantee cleanup)
        try:
            _perturb_det_shifts(listfile, sx_mm, sy_mm)
            _run_indexamajig(
                geomfile=Path(geomfile_path),
                listfile=listfile,
                cellfile=Path(cellfile_path),
                output_stream=out_stream,
                threads=int(num_threads),
                extra_flags=extra_flags,
            )
        except KeyboardInterrupt:
            # Revert and propagate so parent can stop
            try:
                _perturb_det_shifts(listfile, -sx_mm, -sy_mm)
            finally:
                print("[iterator] interrupted by user", flush=True)
            _cleanup_temp_dirs()
            return 130
        except Exception as e:
            # Revert and continue to next pass (log the error)
            try:
                _perturb_det_shifts(listfile, -sx_mm, -sy_mm)
            finally:
                print(f"[iterator] pass {i}/{total_passes} failed: {e}", flush=True)
            # policy: continue scanning even if one pass fails
            continue
        else:
            # Normal revert after successful run
            _perturb_det_shifts(listfile, -sx_mm, -sy_mm)

    _cleanup_temp_dirs()
    print("[iterator] done.", flush=True)
    return 0
# --- end of vendored iterator ---


# =============
# Data classes
# =============
@dataclass
class RunContext:
    run_dir: Path
    geom_path: Path
    cell_path: Path
    input_dir: Path
    list_path: Path
    out_base: str
    threads: int
    max_radius: float
    step: float
    extra_flags: List[str]
    total_images: int = 0
    estimated_passes: int = 1


# =============
# Utilities
# =============
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def timestamp_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"indexingintegration_{ts}"


def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def count_images_in_h5_folder(folder: Path) -> int:
    """
    Count frames in /entry/data/images across top-level *.h5 files (non-recursive).
    Matches parent progress scaling; the iterator will use the same list.
    """
    total = 0
    for h5path in folder.glob("*.h5"):
        try:
            with h5py.File(h5path, "r") as h5f:
                if "/entry/data/images" in h5f:
                    ds = h5f["/entry/data/images"]
                    if ds.ndim >= 3:
                        total += int(ds.shape[0])
        except Exception:
            pass
    return total


def estimate_grid_points(max_radius_px: float, step_px: float) -> int:
    if step_px <= 0.0 or max_radius_px < 0.0:
        return 0
    if max_radius_px == 0.0:
        return 1
    r2 = max_radius_px * max_radius_px
    s = step_px
    k = int(math.ceil(max_radius_px / s))
    count = 0
    for ix in range(-k, k + 1):
        for iy in range(-k, k + 1):
            if (ix*s) ** 2 + (iy*s) ** 2 <= r2 + 1e-9:
                count += 1
    return max(1, count)


def append_line(widget: QPlainTextEdit, text: str) -> None:
    try:
        widget.appendPlainText(text)
    except Exception:
        try:
            widget.append(text)
        except Exception:
            print(text)


# =========================
# Self-importing runner src
# =========================
GANDALF_RUNNER_CODE = r"""
import sys, os, argparse, importlib.util
os.environ.setdefault("PYTHONUNBUFFERED", "1")

def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location("host_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--host", required=True)
    ap.add_argument("--geom",    required=True)
    ap.add_argument("--cell",    required=True)
    ap.add_argument("--input",   required=True)
    ap.add_argument("--outbase", required=True)
    ap.add_argument("--threads", required=True, type=int)
    ap.add_argument("--radius",  required=True, type=float)
    ap.add_argument("--step",    required=True, type=float)
    args, extra = ap.parse_known_args()

    mod = load_module_from_path(args.host)
    if not hasattr(mod, "gandalf_iterator"):
        sys.stderr.write("gandalf_iterator not found in host file\\n")
        sys.exit(2)

    rc = mod.gandalf_iterator(
        geomfile_path=args.geom,
        cellfile_path=args.cell,
        input_path=args.input,
        output_file_base=args.outbase,
        num_threads=args.threads,
        max_radius=args.radius,
        step=args.step,
        extra_flags=extra or None,
    )
    sys.exit(int(rc) if isinstance(rc, int) else 0)

if __name__ == "__main__":
    main()
"""


# =====================
# Process output reader
# =====================
class ProcessOutputThread(QThread):
    output_received = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, process: subprocess.Popen):
        super().__init__()
        self.process = process

    def run(self):
        try:
            for line in iter(self.process.stdout.readline, b''):
                try:
                    self.output_received.emit(line.decode("utf-8", errors="replace"))
                except Exception:
                    self.output_received.emit(str(line))
        finally:
            rc = self.process.wait()
            self.finished.emit(int(rc))


# ===========
# Main Window
# ===========
class SerialEDIndexIntegrateWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Index & Integrate (SerialED Grid)")
        self.resize(1200, 800)

        # Process / threads
        self.proc: Optional[subprocess.Popen] = None
        self.proc_thread: Optional[ProcessOutputThread] = None
        self.proc_runner_path: Optional[Path] = None

        # Build UI
        self.tabs = QTabWidget()
        self.settings_tab = QWidget()
        self.start_tab = QWidget()
        self._build_settings_tab()
        self._build_start_tab()

        # Left runs panel + right tabs
        self.splitter = QSplitter()
        self.runs_panel = QWidget()
        self._build_runs_panel()
        self.splitter.addWidget(self.runs_panel)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self.splitter)

        # State
        self.run_root = Path.cwd() / "runs"
        ensure_dir(self.run_root)
        self.current_run: Optional[RunContext] = None
        self.selected_run_dir: Optional[Path] = None

        # Progress bookkeeping
        self.per_pass_images = 0
        self.estimated_passes = 1
        self.last_seen_processed = 0
        self.seen_passes = 0

        # Timing / rate (EMA)
        self.run_start_time: Optional[float] = None
        self.ema_rate: Optional[float] = None
        self.ema_alpha = 0.3

        # Batch state
        self.batch_queue: List[dict] = []
        self.batch_active = False

    # -----------------
    # UI: Settings tab
    # -----------------
    def _build_settings_tab(self):
        w = self.settings_tab
        outer = QVBoxLayout(w)

        # Files group
        files_group = QGroupBox("Files")
        files_form = QFormLayout(files_group)

        self.edit_geom = QLineEdit()
        btn_geom = QPushButton("Browse…")
        btn_geom.clicked.connect(lambda: self._browse_to(self.edit_geom, "Geometry (*.geom)"))
        files_form.addRow("Geometry (.geom):", self._hpair(self.edit_geom, btn_geom))

        self.edit_cell = QLineEdit()
        btn_cell = QPushButton("Browse…")
        btn_cell.clicked.connect(lambda: self._browse_to(self.edit_cell, "Cell (*.cell)"))
        files_form.addRow("Cell (.cell):", self._hpair(self.edit_cell, btn_cell))

        self.edit_input_dir = QLineEdit()
        btn_input = QPushButton("Browse…")
        btn_input.clicked.connect(lambda: self._browse_dir(self.edit_input_dir))
        files_form.addRow("Input folder (HDF5s):", self._hpair(self.edit_input_dir, btn_input))

        self.edit_out_base = QLineEdit("Xtal")
        files_form.addRow("Output base:", self.edit_out_base)

        outer.addWidget(files_group)

        # Compute / grid
        idx_group = QGroupBox("Indexing & SerialED Grid")
        idx_form = QFormLayout(idx_group)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 256)
        self.spin_threads.setValue(max(1, os.cpu_count() or 8))
        idx_form.addRow("Threads:", self.spin_threads)

        self.spin_max_radius = QDoubleSpinBox()
        self.spin_max_radius.setDecimals(3)
        self.spin_max_radius.setRange(0.0, 100.0)
        self.spin_max_radius.setSingleStep(0.1)
        self.spin_max_radius.setValue(0.0)
        idx_form.addRow("Max radius (px):", self.spin_max_radius)

        self.spin_step = QDoubleSpinBox()
        self.spin_step.setDecimals(3)
        self.spin_step.setRange(0.001, 10.0)
        self.spin_step.setSingleStep(0.1)
        self.spin_step.setValue(0.1)
        idx_form.addRow("Step (px):", self.spin_step)

        outer.addWidget(idx_group)

        # Flags
        flags_group = QGroupBox("Flags")
        flags_layout = QGridLayout(flags_group)

        flags_layout.addWidget(QLabel("Peakfinder preset:"), 0, 0)
        self.edit_peakfinder = QLineEdit("peakfinder9")
        flags_layout.addWidget(self.edit_peakfinder, 0, 1)

        flags_layout.addWidget(QLabel("Advanced flags:"), 1, 0)
        self.txt_advanced = QPlainTextEdit()
        self.txt_advanced.setPlainText(
            "--min-peaks=15\n"
            "--tolerance=10,10,10,5\n"
            "--xgandalf-sampling-pitch=5\n"
            "--xgandalf-grad-desc-iterations=1\n"
            "--xgandalf-tolerance=0.02\n"
            "--int-radius=4,5,9\n"
        )
        self.txt_advanced.setMinimumHeight(110)
        flags_layout.addWidget(self.txt_advanced, 1, 1)

        flags_layout.addWidget(QLabel("Other flags:"), 2, 0)
        self.txt_other = QPlainTextEdit()
        self.txt_other.setPlainText(
            "--no-revalidate\n"
            "--no-half-pixel-shift\n"
            "--no-refine\n"
            "--no-non-hits-in-stream\n"
            "--no-retry\n"
            "--fix-profile-radius=70000000\n"
            "--indexing=xgandalf\n"
            "--integration=rings\n"
        )
        self.txt_other.setMinimumHeight(110)
        flags_layout.addWidget(self.txt_other, 2, 1)

        outer.addWidget(flags_group)
        self.tabs.addTab(self.settings_tab, "Settings")

    # ---------------
    # UI: Start tab
    # ---------------
    def _build_start_tab(self):
        w = self.start_tab
        outer = QVBoxLayout(w)

        # Buttons (run controls)
        btns = QHBoxLayout()
        self.btn_start_grid = QPushButton("Start SerialED Grid")
        self.btn_start_grid.clicked.connect(self._start_grid_clicked)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_clicked)
        self.btn_stop.setEnabled(False)
        btns.addWidget(self.btn_start_grid)
        btns.addWidget(self.btn_stop)
        outer.addLayout(btns)

        # Progress + pass label
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        outer.addWidget(self.progress)

        row = QHBoxLayout()
        self.lbl_status = QLabel("Ready.")
        self.lbl_pass = QLabel("Pass: -/-")
        self.lbl_elapsed = QLabel("Elapsed: 00:00:00")
        self.lbl_eta = QLabel("ETA: --:--:--")
        self.lbl_rate = QLabel("Rate: -- img/s")
        for wdg in (self.lbl_status, self.lbl_pass, self.lbl_elapsed, self.lbl_eta, self.lbl_rate):
            wdg.setMinimumWidth(160)
        row.addWidget(self.lbl_status)
        row.addWidget(self.lbl_pass)
        row.addWidget(self.lbl_elapsed)
        row.addWidget(self.lbl_eta)
        row.addWidget(self.lbl_rate)
        row.addStretch(1)
        outer.addLayout(row)

        # Command preview
        prev_group = QGroupBox("Command preview")
        prev_lay = QVBoxLayout(prev_group)
        self.txt_preview = QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt_preview.setMaximumHeight(80)
        prev_lay.addWidget(self.txt_preview)
        outer.addWidget(prev_group)

        # Tools row
        tools = QHBoxLayout()
        self.btn_open_streams = QPushButton("Open streams folder")
        self.btn_merge_streams = QPushButton("Merge streams now")
        self.btn_export_run = QPushButton("Export run (.zip)")
        self.btn_open_streams.clicked.connect(self._open_streams_folder)
        self.btn_merge_streams.clicked.connect(self._merge_streams_now_clicked)
        self.btn_export_run.clicked.connect(self._export_run_bundle)
        tools.addWidget(self.btn_open_streams)
        tools.addWidget(self.btn_merge_streams)
        tools.addWidget(self.btn_export_run)
        tools.addStretch(1)
        outer.addLayout(tools)

        # Output log
        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt_output.setMinimumHeight(280)
        outer.addWidget(self.txt_output, 1)

        self.tabs.addTab(self.start_tab, "Start")

    # --------------------
    # Runs panel (left)
    # --------------------
    def _build_runs_panel(self):
        w = self.runs_panel
        outer = QVBoxLayout(w)

        # Run root chooser
        row = QHBoxLayout()
        row.addWidget(QLabel("Run root:"))
        self.edit_run_root = QLineEdit(str(Path.cwd() / "runs"))
        btn_browse_root = QPushButton("…")
        btn_browse_root.setToolTip("Choose run root directory")
        btn_browse_root.clicked.connect(self._choose_run_root)
        row.addWidget(self.edit_run_root, 1)
        row.addWidget(btn_browse_root, 0)
        outer.addLayout(row)

        # Buttons: new/open
        row2 = QHBoxLayout()
        self.btn_new_run = QPushButton("New Run")
        self.btn_new_run.setToolTip("Create a timestamped run folder")
        self.btn_new_run.clicked.connect(self._create_new_run_dir)

        self.btn_open_run = QPushButton("Open Folder")
        self.btn_open_run.setToolTip("Open selected run folder in file manager")
        self.btn_open_run.clicked.connect(self._open_selected_run_dir)
        row2.addWidget(self.btn_new_run)
        row2.addWidget(self.btn_open_run)
        outer.addLayout(row2)

        # Runs list
        self.list_runs = QListWidget()
        self.list_runs.itemSelectionChanged.connect(self._on_select_run)
        outer.addWidget(self.list_runs, 1)

        # Batch queue
        outer.addWidget(QLabel("Batch queue:"))
        self.list_batch = QListWidget()
        outer.addWidget(self.list_batch, 1)

        row3 = QHBoxLayout()
        self.btn_batch_add = QPushButton("Add current settings")
        self.btn_batch_clear = QPushButton("Clear")
        self.btn_batch_start = QPushButton("Start Batch")
        self.btn_batch_add.clicked.connect(self._batch_add_current)
        self.btn_batch_clear.clicked.connect(self._batch_clear)
        self.btn_batch_start.clicked.connect(self._batch_start)
        row3.addWidget(self.btn_batch_add)
        row3.addWidget(self.btn_batch_clear)
        row3.addWidget(self.btn_batch_start)
        outer.addLayout(row3)

        self._refresh_runs_tree()

    # ==============
    # Path helpers
    # ==============
    def _hpair(self, a: QWidget, b: QWidget) -> QWidget:
        cw = QWidget()
        lay = QHBoxLayout(cw)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(a, 1)
        lay.addWidget(b, 0)
        return cw

    def _browse_to(self, lineedit: QLineEdit, filter_str: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", str(Path.cwd()), filter_str)
        if path:
            lineedit.setText(path)

    def _browse_dir(self, lineedit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select folder", str(Path.cwd()))
        if path:
            lineedit.setText(path)

    # ================
    # Run list / open
    # ================
    def _refresh_runs_tree(self):
        self.list_runs.clear()
        root = Path(self.edit_run_root.text().strip())
        if not root.exists():
            return
        for p in sorted(root.glob("indexingintegration_*")):
            if not p.is_dir():
                continue
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.list_runs.addItem(item)

    def _select_run_in_list(self, run_dir: Path):
        for i in range(self.list_runs.count()):
            item = self.list_runs.item(i)
            if Path(item.data(Qt.ItemDataRole.UserRole)) == run_dir:
                self.list_runs.setCurrentItem(item)
                break

    def _on_select_run(self):
        item = self.list_runs.currentItem()
        if not item:
            return
        run_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        # Minimal: reflect into preview; keep UI file pickers user-controlled
        self.selected_run_dir = run_dir
        self._update_command_preview(None)

    def _resolve_active_run_dir(self) -> Optional[Path]:
        if getattr(self, "current_run", None):
            return self.current_run.run_dir
        return getattr(self, "selected_run_dir", None)

    def _open_selected_run_dir(self):
        item = self.list_runs.currentItem()
        if not item:
            return
        run_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(run_dir)])
            elif os.name == "nt":
                os.startfile(str(run_dir))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(run_dir)])
        except Exception as e:
            QMessageBox.warning(self, "Open folder failed", str(e))

    def _choose_run_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select run root", str(Path(self.edit_run_root.text().strip())))
        if path:
            self.run_root = Path(path)
            ensure_dir(self.run_root)
            self.edit_run_root.setText(str(self.run_root))
            self._refresh_runs_tree()

    def _create_new_run_dir(self):
        run_dir = timestamp_run_dir(Path(self.edit_run_root.text().strip()))
        ensure_dir(run_dir)
        ensure_dir(run_dir / "streams")
        write_text(run_dir / "input.lst", "")
        self._refresh_runs_tree()
        self._select_run_in_list(run_dir)

    # =========================
    # Settings → command text
    # =========================
    def _compose_extra_flags(self) -> List[str]:
        flags: List[str] = []
        pf = self.edit_peakfinder.text().strip()
        if pf:
            flags.extend([
                f"--peaks={pf}",
                "--min-snr-biggest-pix=7",
                "--min-snr-peak-pix=6",
                "--min-snr=5",
                "--min-sig=11",
                "--min-peak-over-neighbour=-inf",
                "--local-bg-radius=3",
            ])
        for line in self.txt_advanced.toPlainText().splitlines():
            s = line.strip()
            if s:
                flags.append(s)
        for line in self.txt_other.toPlainText().splitlines():
            s = line.strip()
            if s:
                flags.append(s)
        return flags

    def _build_command_preview(self, ctx: RunContext) -> str:
        base = [
            "indexamajig",
            "-g", str(ctx.geom_path),
            "-i", str(ctx.list_path),
            "-o", f"{ctx.out_base}.stream",
            "-p", str(ctx.cell_path),
            "-j", str(ctx.threads),
        ]
        extras = " ".join(ctx.extra_flags)
        return (
            " ".join(base) + (f" {extras}" if extras else "") +
            f"\n[grid] R={ctx.max_radius:g}px, step={ctx.step:g}px → ~{ctx.estimated_passes} passes"
        )

    def _update_command_preview(self, ctx: Optional[RunContext] = None):
        try:
            if ctx is None:
                # Preview from UI state only
                geom = Path(self.edit_geom.text().strip())
                cell = Path(self.edit_cell.text().strip())
                in_dir = Path(self.edit_input_dir.text().strip())
                out_base = self.edit_out_base.text().strip() or "output"
                threads = int(self.spin_threads.value())
                max_radius = float(self.spin_max_radius.value())
                step = float(self.spin_step.value())
                extras = " ".join(self._compose_extra_flags())

                list_hint = (Path(self.edit_run_root.text().strip()) / "input.lst").as_posix()
                base = [
                    "indexamajig",
                    "-g", geom.as_posix() if geom else "<geom>",
                    "-i", list_hint,
                    "-o", f"{out_base}.stream",
                    "-p", cell.as_posix() if cell else "<cell>",
                    "-j", str(threads),
                ]
                grid_info = f"[grid] R={max_radius:g}px, step={step:g}px → ~{estimate_grid_points(max_radius, step)} passes"
                self.txt_preview.setPlainText(" ".join(base) + (f" {extras}" if extras else "") + f"\n{grid_info}")
                return
            self.txt_preview.setPlainText(self._build_command_preview(ctx))
        except Exception as e:
            self.txt_preview.setPlainText(f"[preview error] {e}")

    # =====================================
    # Start / Stop and runner orchestration
    # =====================================
    def _write_input_list(self, in_dir: Path, list_path: Path) -> int:
        lines: List[str] = [str(p.resolve()) for p in sorted(in_dir.glob("*.h5"))]
        write_text(list_path, "\n".join(lines) + ("\n" if lines else ""))
        total = count_images_in_h5_folder(in_dir)
        append_line(self.txt_output, f"[scan] detected {total} images across {len(lines)} .h5 files")
        return total

    def _start_grid_clicked(self):
        # Validate
        geom = Path(self.edit_geom.text().strip())
        cell = Path(self.edit_cell.text().strip())
        in_dir = Path(self.edit_input_dir.text().strip())
        out_base = self.edit_out_base.text().strip() or "output"
        threads = int(self.spin_threads.value())
        max_radius = float(self.spin_max_radius.value())
        step = float(self.spin_step.value())

        missing = []
        if not geom.is_file(): missing.append("geometry file")
        if not cell.is_file(): missing.append("cell file")
        if not in_dir.is_dir(): missing.append("input folder")
        if step <= 0.0: missing.append("step > 0")
        if max_radius < 0.0: missing.append("radius ≥ 0")
        if missing:
            QMessageBox.warning(self, "Missing/invalid", "Please provide: " + ", ".join(missing))
            return

        extra_flags = self._compose_extra_flags()

        # Create run dir & list
        run_dir = timestamp_run_dir(Path(self.edit_run_root.text().strip()))
        ensure_dir(run_dir)
        list_path = run_dir / "input.lst"
        total_images = self._write_input_list(in_dir, list_path)
        ensure_dir(run_dir / "streams")

        # Progress scale
        est_passes = estimate_grid_points(max_radius, step)
        self.per_pass_images = total_images
        self.estimated_passes = max(1, est_passes)
        self.last_seen_processed = 0
        self.seen_passes = 0

        # Context
        self.current_run = RunContext(
            run_dir=run_dir, geom_path=geom, cell_path=cell,
            input_dir=in_dir, list_path=list_path, out_base=out_base,
            threads=threads, max_radius=max_radius, step=step,
            extra_flags=extra_flags, total_images=total_images,
            estimated_passes=est_passes,
        )
        self._update_command_preview(self.current_run)
        self.lbl_pass.setText(f"Pass: 1/{max(1, est_passes)}")

        append_line(self.txt_output, f"[grid] base='{out_base}', R={max_radius:g}px step={step:g}px → {est_passes} passes")
        append_line(self.txt_output, f"[files] run_dir={run_dir}")
        append_line(self.txt_output, f"[files] input.lst={list_path} (images={total_images})")
        append_line(self.txt_output, f"[files] streams={run_dir/'streams'}")
        append_line(self.txt_output, "[preview] " + self._build_command_preview(self.current_run).replace("\n", " | "))

        # Launch
        self._launch_runner(self.current_run)

    def _stop_clicked(self):
        self._kill_process_group()
        self._cleanup_runner_file()
        self.btn_stop.setEnabled(False)
        self.btn_start_grid.setEnabled(True)
        self.lbl_status.setText("Stopped.")

    def _launch_runner(self, ctx: RunContext):
        # Write temp runner
        fd, runner_path = tempfile.mkstemp(prefix="run_gandalf_", suffix=".py")
        os.close(fd)
        self.proc_runner_path = Path(runner_path)
        with open(self.proc_runner_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(GANDALF_RUNNER_CODE))

        # Build argv
        streams_dir = ctx.run_dir / "streams"
        ensure_dir(streams_dir)
        outbase_for_child = str((streams_dir / ctx.out_base).resolve())
        argv = [
            sys.executable,
            "-u",
            str(self.proc_runner_path),
            "--host", os.path.abspath(__file__),
            "--geom", str(ctx.geom_path),
            "--cell", str(ctx.cell_path),
            "--input", str(ctx.input_dir),
            "--outbase", outbase_for_child,
            "--threads", str(ctx.threads),
            "--radius", str(ctx.max_radius),
            "--step", str(ctx.step),
            *ctx.extra_flags,
        ]

        append_line(self.txt_output, "[launch] " + " ".join(shlex.quote(a) for a in argv))

        # Per-run log
        self.run_log_path = ctx.run_dir / "indexing.log"
        self.run_log_fp = open(self.run_log_path, "w", encoding="utf-8")

        # Spawn (new process group on Unix)
        try:
            self.proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )
        except Exception as e:
            QMessageBox.critical(self, "Launch error", str(e))
            self._cleanup_runner_file()
            return

        # Reader
        self.proc_thread = ProcessOutputThread(self.proc)
        self.proc_thread.output_received.connect(self._on_proc_line)
        self.proc_thread.finished.connect(self._on_proc_finished)
        self.proc_thread.start()

        # UI state
        self.btn_start_grid.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Running…")
        self._reset_progress_bar()

        # Timing start
        self.run_start_time = time.time()
        self.ema_rate = None
        self.lbl_elapsed.setText("Elapsed: 00:00:00")
        self.lbl_eta.setText("ETA: --:--:--")
        self.lbl_rate.setText("Rate: -- img/s")

    def _reset_progress_bar(self):
        total = max(1, self.estimated_passes * max(1, self.per_pass_images))
        self.progress.setMinimum(0)
        self.progress.setMaximum(total)
        self.progress.setValue(0)

    # =====================
    # Process output hooks
    # =====================
    def _on_proc_line(self, text: str):
        append_line(self.txt_output, text.rstrip("\n"))
        try:
            self.run_log_fp.write(text)
            self.run_log_fp.flush()
        except Exception:
            pass

        # Progress: "<N> images processed"
        n = self._parse_images_processed(text)
        if n is not None:
            if n < self.last_seen_processed:
                self.seen_passes += 1
            self.last_seen_processed = n
            total_so_far = self.seen_passes * self.per_pass_images + n
            self.progress.setValue(max(self.progress.value(), total_so_far))

        # Optional explicit pass marker: "[XY-PASS] i/N ..."
        if text.startswith("[XY-PASS]"):
            parts = text.split()
            try:
                frac = parts[1]  # "i/N"
                i, N = frac.split("/")
                self.seen_passes = max(self.seen_passes, int(i) - 1)
                self.estimated_passes = max(self.estimated_passes, int(N))
                self._reset_progress_bar()
            except Exception:
                pass
        self.lbl_pass.setText(f"Pass: {min(self.seen_passes+1, max(1,self.estimated_passes))}/{max(1,self.estimated_passes)}")

        # Rate / ETA
        now = time.time()
        if self.run_start_time:
            rate_in_line = self._parse_images_per_sec(text)
            cur = self.progress.value()
            if rate_in_line and rate_in_line > 0:
                self.ema_rate = (self.ema_alpha * rate_in_line +
                                 (1 - self.ema_alpha) * (self.ema_rate or rate_in_line))
            else:
                elapsed = max(1e-3, now - self.run_start_time)
                derived = cur / elapsed
                if derived > 0:
                    self.ema_rate = (self.ema_alpha * derived +
                                     (1 - self.ema_alpha) * (self.ema_rate or derived))
            self._update_time_labels(now, cur)

    def _parse_images_processed(self, line: str) -> Optional[int]:
        if "images processed" not in line:
            return None
        num = None
        tok = ""
        for ch in line:
            if ch.isdigit():
                tok += ch
            else:
                if tok:
                    num = int(tok)
                    break
        return num

    def _parse_images_per_sec(self, line: str) -> Optional[float]:
        if "images/sec" not in line:
            return None
        try:
            before = line.split("images/sec")[0]
            tokens = before.strip().split()
            for tok in reversed(tokens):
                t = tok.strip(",;")
                try:
                    return float(t)
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _update_time_labels(self, now_ts: float, cur_progress: int):
        elapsed = int(now_ts - (self.run_start_time or now_ts))
        self.lbl_elapsed.setText(f"Elapsed: {self._fmt_hms(elapsed)}")
        if self.ema_rate and self.ema_rate > 0:
            self.lbl_rate.setText(f"Rate: {self.ema_rate:.2f} img/s")
            remaining = max(0, self.progress.maximum() - cur_progress)
            eta_sec = int(remaining / self.ema_rate) if self.ema_rate > 0 else 0
            self.lbl_eta.setText(f"ETA: {self._fmt_hms(eta_sec)}")
        else:
            self.lbl_rate.setText("Rate: -- img/s")
            self.lbl_eta.setText("ETA: --:--:--")

    def _fmt_hms(self, seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _on_proc_finished(self, rc: int):
        append_line(self.txt_output, f"[done] exit code {rc}")
        self.lbl_status.setText("Finished." if rc == 0 else f"Failed (rc={rc})")
        self.btn_stop.setEnabled(False)
        self.btn_start_grid.setEnabled(True)
        try:
            self.run_log_fp.close()
        except Exception:
            pass
        self._cleanup_runner_file()
        self.proc = None
        self.proc_thread = None

        # Reset time labels
        self.lbl_elapsed.setText("Elapsed: 00:00:00")
        self.lbl_eta.setText("ETA: --:--:--")
        self.lbl_rate.setText("Rate: -- img/s")
        self.lbl_pass.setText("Pass: -/-")
        self.run_start_time = None
        self.ema_rate = None

        # Batch continuation
        if self.batch_active:
            QTimer.singleShot(150, self._batch_next)

    # =========
    # Cleanup
    # =========
    def _kill_process_group(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

    def _cleanup_runner_file(self):
        try:
            if self.proc_runner_path and self.proc_runner_path.exists():
                self.proc_runner_path.unlink(missing_ok=True)
        except Exception:
            pass
        self.proc_runner_path = None

    # ========================
    # Streams & export tools
    # ========================
    def _open_streams_folder(self):
        run_dir = self._resolve_active_run_dir()
        if not run_dir:
            QMessageBox.information(self, "Streams", "No run selected. Create/select a run on the left.")
            return
        streams_dir = run_dir / "streams"
        ensure_dir(streams_dir)
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(streams_dir)])
            elif os.name == "nt":
                os.startfile(str(streams_dir))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(streams_dir)])
        except Exception as e:
            QMessageBox.warning(self, "Open streams failed", str(e))

    def _merge_streams_now_clicked(self):
        run_dir = self._resolve_active_run_dir()
        if not run_dir:
            QMessageBox.information(self, "Merge streams", "No run selected.")
            return
        try:
            out_path = self._merge_streams_in_run(run_dir)
            append_line(self.txt_output, f"[merge] merged -> {out_path}")
            QMessageBox.information(self, "Merge streams", f"Merged file:\n{out_path}")
        except Exception as e:
            QMessageBox.warning(self, "Merge streams failed", str(e))

    def _merge_streams_in_run(self, run_dir: Path, out_name: Optional[str] = None) -> Path:
        streams_dir = run_dir / "streams"
        files = sorted(streams_dir.glob("*.stream"))
        if not files:
            raise RuntimeError(f"No .stream files in {streams_dir}")
        if out_name:
            out_path = run_dir / out_name
        else:
            base = None
            if getattr(self, "current_run", None) and self.current_run.run_dir == run_dir:
                base = self.current_run.out_base
            out_path = run_dir / (f"{base}_merged.stream" if base else "merged.stream")
        with open(out_path, "w", encoding="utf-8") as out:
            for i, fpath in enumerate(files, start=1):
                with open(fpath, "r", encoding="utf-8", errors="replace") as inp:
                    for line in inp:
                        out.write(line)
                if i != len(files):
                    out.write("\n")
        return out_path

    def _export_run_bundle(self):
        run_dir = self._resolve_active_run_dir()
        if not run_dir:
            QMessageBox.information(self, "Export run", "No run selected.")
            return
        stage = Path(tempfile.mkdtemp(prefix="run_export_"))
        try:
            staged_run = stage / run_dir.name
            shutil.copytree(run_dir, staged_run)
            # Optionally copy external refs (geom/cell) if outside run_dir
            refs_dir = staged_run / "refs"
            refs_added = False
            geom_path = Path(self.edit_geom.text().strip()) if self.edit_geom.text().strip() else None
            cell_path = Path(self.edit_cell.text().strip()) if self.edit_cell.text().strip() else None
            for p in (geom_path, cell_path):
                if p and p.exists() and not str(p.resolve()).startswith(str(run_dir.resolve())):
                    ensure_dir(refs_dir)
                    shutil.copy2(p, refs_dir / p.name)
                    refs_added = True
            zip_path = run_dir.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=stage, base_dir=run_dir.name)
            note = " (with refs)" if refs_added else ""
            append_line(self.txt_output, f"[export] created {zip_path}{note}")
            QMessageBox.information(self, "Export run", f"Created:\n{zip_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export run failed", str(e))
        finally:
            try:
                shutil.rmtree(stage, ignore_errors=True)
            except Exception:
                pass

    # ==========================
    # Batch queue (simple flow)
    # ==========================
    def _batch_add_current(self):
        data = {
            "geom_path": self.edit_geom.text().strip(),
            "cell_path": self.edit_cell.text().strip(),
            "input_dir": self.edit_input_dir.text().strip(),
            "out_base": self.edit_out_base.text().strip(),
            "threads": int(self.spin_threads.value()),
            "max_radius": float(self.spin_max_radius.value()),
            "step": float(self.spin_step.value()),
            "peakfinder": self.edit_peakfinder.text().strip(),
            "advanced_flags": self.txt_advanced.toPlainText(),
            "other_flags": self.txt_other.toPlainText(),
        }
        self.batch_queue.append(data)
        item = QListWidgetItem(
            f"{data.get('out_base','output')}  |  R={data.get('max_radius',0)}  step={data.get('step',0)}"
        )
        item.setData(Qt.ItemDataRole.UserRole, json.dumps(data))
        self.list_batch.addItem(item)

    def _batch_clear(self):
        self.batch_queue.clear()
        self.list_batch.clear()

    def _batch_start(self):
        if self.batch_active or not self.batch_queue:
            return
        self.batch_active = True
        append_line(self.txt_output, f"[batch] starting {len(self.batch_queue)} job(s)")
        self._batch_next()

    def _batch_next(self):
        if not self.batch_queue:
            append_line(self.txt_output, "[batch] done.")
            self.batch_active = False
            return
        raw = self.batch_queue.pop(0)
        try:
            data = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            data = {}
        # Apply UI and start
        self.edit_geom.setText(data.get("geom_path", ""))
        self.edit_cell.setText(data.get("cell_path", ""))
        self.edit_input_dir.setText(data.get("input_dir", ""))
        self.edit_out_base.setText(data.get("out_base", "output"))
        try: self.spin_threads.setValue(int(data.get("threads", self.spin_threads.value())))
        except Exception: pass
        try: self.spin_max_radius.setValue(float(data.get("max_radius", self.spin_max_radius.value())))
        except Exception: pass
        try: self.spin_step.setValue(float(data.get("step", self.spin_step.value())))
        except Exception: pass
        self.txt_advanced.setPlainText(data.get("advanced_flags", self.txt_advanced.toPlainText()))
        self.txt_other.setPlainText(data.get("other_flags", self.txt_other.toPlainText()))
        # Start
        self._start_grid_clicked()

    # --------------------
    # Little helpers
    # --------------------
    def _fmt_hms(self, seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


# -----------
# main-guard
# -----------
def main() -> None:
    app = QApplication(sys.argv)
    win = SerialEDIndexIntegrateWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
