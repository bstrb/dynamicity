#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexIntegrate (SerialED Grid) – COSEDA-style window
Part 1: Core window, UI (Settings/Start), runner, progress, run dir, flags surface.
- PyQt6 single-file app
- Self-importing runner with parse_known_args (passes unknown --flags through)
- SerialED grid controls (max radius, step)
- Peakfinder/Advanced/Other flags surface
- HDF5 image counting (/entry/data/images)
- Start/Stop with process-group kill and live log
- Per-run folder with input.lst and indexing.log
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
import h5py
import shlex
import signal
import tempfile
import textwrap
import subprocess
from configparser import ConfigParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QDoubleSpinBox, QSplitter, QListWidgetItem,
    QGroupBox, QTabWidget, QProgressBar, QComboBox, QTextEdit, QScrollArea
)

# ==========================
# VENDORED GANDALF ITERATOR
# ==========================
# Paste your *real* gandalf_radial_iterator.py contents here. The entry point
# must be named `gandalf_iterator(geomfile_path, cellfile_path, input_path,
# output_file_base, num_threads, max_radius, step, extra_flags=None)`.
# It should run multiple indexamajig passes (one per (x,y) shift), print progress
# lines to stdout, and return 0 on success.
#
# --- VENDORED GANDALF ITERATOR BEGIN ---

def gandalf_iterator(
    geomfile_path: str,
    cellfile_path: str,
    input_path: str,
    output_file_base: str,
    num_threads: int,
    max_radius: float,
    step: float,
    extra_flags=None,
):
    """
    Run CrystFEL 'indexamajig' over a filled circular grid of center shifts.

    For each (dx, dy) in *pixels*:
      1) Convert to millimeters using geometry 'res' (mm/px = 1000/res).
      2) Apply det_shift_x_mm += dx_mm and det_shift_y_mm += dy_mm (in-place) to all HDF5s listed.
      3) Run indexamajig to write a unique .stream for this pass.
      4) Revert the applied shifts (subtract the same deltas), even on errors.

    Emits progress lines and [XY-PASS] i/N markers for the GUI.
    """
    import sys, subprocess, math
    from pathlib import Path

    # ---- Validate required inputs ----
    geom = Path(geomfile_path)
    cell = Path(cellfile_path)
    input_p = Path(input_path)

    if not geom.is_file():
        print(f"ERROR: geometry file missing: {geom}", file=sys.stderr); return 2
    if not cell.is_file():
        print(f"ERROR: cell file missing: {cell}", file=sys.stderr); return 2
    if not input_p.exists():
        print(f"ERROR: input path missing: {input_p}", file=sys.stderr); return 2

    # ---- Output directory & base name ----
    outbase = Path(output_file_base)
    if outbase.is_absolute():
        out_dir = outbase.parent
        out_base_name = outbase.name
    else:
        base_dir = input_p if input_p.is_dir() else input_p.parent
        out_dir = base_dir / f"xgandalf_iterations_max_radius_{max_radius}_step_{step}"
        out_base_name = outbase.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resulting streamfiles will be saved in {out_dir}")

    # ---- Prepare list.lst (or accept provided list file) ----
    if input_p.is_file():
        list_path = input_p
        print(f"Using provided list file: {list_path}")
    else:
        list_path = input_p / "list.lst"
        try:
            if not list_path.exists():
                exts = ("*.h5", "*.hdf5", "*.cxi")
                hits = []
                for ext in exts:
                    hits.extend(sorted(input_p.glob(ext)))
                hits = [str(p.resolve()) for p in hits]
                list_path.write_text("\n".join(hits) + ("\n" if hits else ""), encoding="utf-8")
                print(f"'list.lst' has been created with {len(hits)} entries at {list_path}")
            else:
                print(f"Using existing list file: {list_path}")
        except Exception as e:
            print(f"WARNING: could not write list.lst: {e}", file=sys.stderr)

    # ---- Build grid: step-lattice points inside circle, sorted radially ----
    grid_pts: list[tuple[float, float]] = []
    if max_radius <= 0 or step <= 0:
        grid_pts = [(0.0, 0.0)]
    else:
        try:
            decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 6
            decimals = min(6, decimals)
        except Exception:
            decimals = 6

        limit = int(math.ceil(max_radius / step))
        R2 = max_radius * max_radius
        pts_set = set()
        for i in range(-limit, limit + 1):
            xi = round(i * step, decimals)
            xi2 = xi * xi
            if xi2 > R2 + 1e-12:
                continue
            for j in range(-limit, limit + 1):
                yj = round(j * step, decimals)
                if xi2 + yj * yj <= R2 + 1e-12:
                    pts_set.add((xi, yj))

        grid_pts = sorted(
            pts_set,
            key=lambda p: (p[0]*p[0] + p[1]*p[1], abs(p[0]) + abs(p[1]), p[0], p[1])
        )
        if (0.0, 0.0) in grid_pts:
            grid_pts.remove((0.0, 0.0))
            grid_pts.insert(0, (0.0, 0.0))

    print(f"Generated {len(grid_pts)} grid points in the circle.")

    # ---- Convert px→mm using geometry 'res' ----
    try:
        res = extract_resolution(str(geom))  # expects a line "res = <float>"
        mm_per_pixel = 1000.0 / float(res)
        print(f"[geom] res={res} → mm_per_pixel={mm_per_pixel}")
    except Exception as e:
        print(f"ERROR: failed to read 'res' from geometry: {e}", file=sys.stderr)
        return 2

    # ---- Handle optional indexamajig path flag; pass through remaining flags ----
    extra_flags = list(extra_flags or [])
    idxamajig_bin = "indexamajig"
    pruned_flags: list[str] = []
    for f in extra_flags:
        if f.startswith("--indexamajig-path="):
            idxamajig_bin = f.split("=", 1)[1].strip() or idxamajig_bin
        else:
            pruned_flags.append(f)
    extra_flags = pruned_flags

    # ---- Main loop: apply shift, run, revert ----
    total = len(grid_pts)
    passes_done = 0

    for (dx_px, dy_px) in grid_pts:
        print(f"Running for pixel shifts x = {dx_px}, y = {dy_px}")
        # Convert pixel shifts to millimeters
        sx_mm = float(dx_px) * mm_per_pixel
        sy_mm = float(dy_px) * mm_per_pixel
        print(f"  => Applying det_shift Δx={sx_mm} mm, Δy={sy_mm} mm")

        stream_name = (
            f"{out_base_name}_{dx_px:.3f}_{dy_px:.3f}.stream"
            .replace("+", "")
            .replace("-0.000", "0.000")
        )
        stream_path = (out_dir / stream_name).resolve()

        # 1) Apply in-place shift
        try:
            perturb_det_shifts(str(list_path), sx_mm, sy_mm)
        except Exception as e:
            print(f"ERROR: failed to apply det_shift ({sx_mm},{sy_mm}) mm: {e}", file=sys.stderr)
            # skip this pass and continue with the next
            continue

        # 2) Run indexamajig and 3) Revert in finally
        try:
            cmd = [
                idxamajig_bin,
                "-g", str(geom),
                "-i", str(list_path),
                "-o", str(stream_path),
                "-p", str(cell),
                "-j", str(num_threads),
                *extra_flags,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                print(f"ERROR: '{idxamajig_bin}' not found. Set it on PATH or use --indexamajig-path=...", file=sys.stderr)
                return 2
            except Exception as e:
                print(f"ERROR: failed to launch indexamajig: {e}", file=sys.stderr)
                return 2

            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line.rstrip())
            except Exception as e:
                print(f"WARNING: error while reading indexamajig output: {e}", file=sys.stderr)

            rc = proc.wait()
            if rc != 0:
                print(f"Error during indexamajig execution (exit {rc}).", file=sys.stderr)
            else:
                passes_done += 1
                print(f"[XY-PASS] {passes_done}/{total} completed -> {stream_path}")

        finally:
            # Always revert the applied shift
            try:
                perturb_det_shifts(str(list_path), -sx_mm, -sy_mm)
                print(f"  => Reverted det_shift Δx={-sx_mm} mm, Δy={-sy_mm} mm")
            except Exception as e:
                print(f"WARNING: failed to revert det_shift: {e}", file=sys.stderr)

    return 0


def extract_resolution(geom_file_path: str) -> float:
    """
    Extracts the resolution value from a geometry file.

    The geometry file is expected to contain a line like:
      res = 17857.14285714286

    Parameters:
        geom_file_path (str): The file path to the geometry file.

    Returns:
        float: The resolution value extracted from the file.

    Raises:
        ValueError: If the resolution value is not found or is invalid.
    """
    with open(geom_file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace
            line = line.strip()
            # Skip comment lines
            if line.startswith(";"):
                continue
            # Look for the resolution line
            if line.startswith("res"):
                parts = line.split("=")
                if len(parts) >= 2:
                    res_str = parts[1].strip()
                    try:
                        return float(res_str)
                    except ValueError:
                        raise ValueError(f"Invalid resolution value: {res_str}")
    raise ValueError("Resolution not found in the geometry file.")

def run_indexamajig(geomfile_path, listfile_path, cellfile_path, output_path, num_threads, extra_flags=None):
    if extra_flags is None:
        extra_flags = []

    cmd = [
        "indexamajig",
        "-g", geomfile_path,
        "-i", listfile_path,
        "-o", output_path,
        "-p", cellfile_path,
        "-j", str(num_threads),
        *extra_flags,
    ]
    # inherit stdout/stderr so QProcess sees lines immediately
    import subprocess
    subprocess.run(cmd, check=True)

def perturb_det_shifts(file_list, x_pert, y_pert):
    """
    Usage: python perturb_shifts.py file_list.lst x_perturbation y_perturbation

    This script adds x_perturbation to entry/data/det_shift_x_mm
    and y_perturbation to entry/data/det_shift_y_mm in each .h5 file
    listed in file_list.lst.
    """

    # Read the .lst file to get list of .h5 files
    with open(file_list, 'r') as f:
        h5_files = [line.strip() for line in f if line.strip()]

    # Loop over .h5 files and apply perturbation
    for h5_file in h5_files:
        # print(f"Processing: {h5_file}")
        with h5py.File(h5_file, 'r+') as h5f:
            # Load existing datasets
            x_data = h5f["entry/data/det_shift_x_mm"][...]
            y_data = h5f["entry/data/det_shift_y_mm"][...]

            # Apply perturbations
            x_data += x_pert
            y_data += y_pert

            # Write updated values back
            h5f["entry/data/det_shift_x_mm"][:] = x_data
            h5f["entry/data/det_shift_y_mm"][:] = y_data

    print(f"  => Applied x shift of {x_pert} mm, y shift of {y_pert} mm")

def list_h5_files(input_path):
    """
    Creates or replaces a 'list.lst' file in the specified input_path directory.
    The file contains the full paths of all files ending with '.h5' in the directory,
    sorted alphabetically.
    
    Args:
        input_path (str): The directory path where '.h5' files are located.
    """
    # Path to the list file
    listfile_path = os.path.join(input_path, 'list.lst')
    
    try:
        # List all .h5 files
        h5_files = [file for file in os.listdir(input_path) if file.endswith('.h5') and os.path.isfile(os.path.join(input_path, file))]
        
        # Sort the list alphabetically
        h5_files_sorted = sorted(h5_files, key=lambda x: x.lower())  # Case-insensitive sorting
        
        # Open the list file in write mode to overwrite if it exists
        with open(listfile_path, 'w') as list_file:
            for file in h5_files_sorted:
                full_path = os.path.join(input_path, file)
                list_file.write(full_path + '\n')
        
        print(f"'list.lst' has been created with {len(h5_files_sorted)} entries at {listfile_path}")
    
    except FileNotFoundError:
        print(f"The directory '{input_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied when accessing '{input_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return listfile_path

# --- VENDORED GANDALF ITERATOR END ---

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


# =========================
# Self-importing runner src
# =========================
GANDALF_RUNNER_CODE = r"""
import sys, os, argparse, importlib.util

def load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("host_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # CRITICAL: register in sys.modules before executing so dataclasses & typing can resolve module
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outbase", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--step", type=float, required=True)
    # Forward everything else verbatim to the iterator
    args, extra = ap.parse_known_args()

    # Import the host module (your GUI file) safely
    mod = load_module_from_path(args.host)

    # Retrieve vendored iterator
    gi = getattr(mod, "gandalf_iterator", None)
    if gi is None:
        print("ERROR: gandalf_iterator not found in host module", file=sys.stderr)
        sys.exit(2)

    # Call it
    rc = gi(
        geomfile_path=args.geom,
        cellfile_path=args.cell,
        input_path=args.input,
        output_file_base=args.outbase,
        num_threads=args.threads,
        max_radius=args.radius,
        step=args.step,
        extra_flags=extra,
    )

    try:
        code = int(rc) if rc is not None else 0
    except Exception:
        code = 0
    sys.exit(code)

if __name__ == "__main__":
    main()
"""

# ---- Peakfinder presets (module-level) ----
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
# -------------------------------------------


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


# ==========
# Utilities
# ==========

def estimate_grid_points(max_radius_px: float, step_px: float) -> int:
    """
    Estimate EXACT number of step-lattice points (x=i*step, y=j*step)
    satisfying x^2 + y^2 <= R^2. Matches gandalf_iterator's grid rule.
    """
    import math

    if step_px <= 0.0 or max_radius_px < 0.0:
        return 0
    if max_radius_px == 0.0:
        return 1  # origin only

    try:
        decimals = max(0, -int(math.floor(math.log10(step_px)))) if step_px > 0 else 6
        decimals = min(6, decimals)
    except Exception:
        decimals = 6

    limit = int(math.ceil(max_radius_px / step_px))
    R2 = max_radius_px * max_radius_px
    count = 0

    for i in range(-limit, limit + 1):
        x = round(i * step_px, decimals)
        x2 = x * x
        if x2 > R2 + 1e-12:
            continue
        # span allowed j for this x
        for j in range(-limit, limit + 1):
            y = round(j * step_px, decimals)
            if x2 + y * y <= R2 + 1e-12:
                count += 1

    return count

def count_images_in_h5_folder(folder: Path) -> int:
    """
    Count images across *.h5/*.hdf5/*.cxi files by summing the first
    dimension of /entry/data/images datasets (when at least 3-D).
    """
    total = 0
    for ext in ("*.h5", "*.hdf5", "*.cxi"):
        for h5path in folder.glob(ext):
            try:
                with h5py.File(h5path, "r") as h5f:
                    if "/entry/data/images" in h5f:
                        ds = h5f["/entry/data/images"]
                        if ds.ndim >= 3:
                            total += int(ds.shape[0])
            except Exception as e:
                print(f"[warn] failed to read {h5path}: {e}")
    return total

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def timestamp_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"indexingintegration_{ts}"


def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def append_line(widget, text: str) -> None:
    try:
        widget.appendPlainText(text)
    except Exception:
        try:
            widget.append(text)
        except Exception:
            print(text)


# ===========
# Main Window
# ===========
class SerialEDIndexIntegrateWindow(QMainWindow):
        
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Index & Integrate (SerialED Grid)")
        self.resize(1200, 800)

        # -------- Core paths / defaults --------
        self.run_root = Path.home() / "indexing_runs"
        self.workspace_root = self.run_root
        ensure_dir(self.run_root)

        # Optional debug defaults for quick testing
        if os.getenv("DEBUG_DEFAULTS", "1") == "1":
            self.default_geom = Path("/home/bubl3932/files/simulations/MFM300-VIII_tI/sim_001/4135627.geom")
            self.default_cell = Path("/home/bubl3932/files/simulations/MFM300-VIII_tI/sim_001/4135627.cell")
            self.default_input = Path("/home/bubl3932/files/simulations/MFM300-VIII_tI/sim_001")
        else:
            self.default_geom = None
            self.default_cell = None
            self.default_input = None

        # Load user prefs once (may override paths above)
        try:
            self._load_prefs()
        except Exception:
            pass

        # -------- State (single source of truth) --------
        self.selected_run_dir: Optional[Path] = None
        self.selected_ini: Optional[Path] = None

        # Progress bookkeeping
        self.per_pass_images = 0
        self.seen_passes = 0
        self.estimated_passes = 1
        self.processed_images_total = 0
        self.last_seen_processed = 0

        # Timing / rate (EMA)
        self.run_start_time = None
        self.ema_rate = None  # images/sec
        self.ema_alpha = 0.3

        # Batch state (manual queue)
        self.batch_queue: list[dict] = []
        self.batch_active = False

        # Workspace batch state
        self._ws_batch_queue: list[Path] = []
        self._ws_batch_mode = False
        self._ws_chain_on_finish = False

        # Child process / runner handles
        self.proc: Optional[subprocess.Popen] = None
        self.proc_thread: Optional[ProcessOutputThread] = None
        self.proc_runner_path: Optional[Path] = None

        # -------- UI skeleton --------
        self.tabs = QTabWidget()
        self.settings_tab = QWidget()
        self.start_tab = QWidget()

        # Build main tabs first (these create widgets like txt_output used elsewhere)
        self._build_settings_tab()
        self._build_start_tab()
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.start_tab, "Start")

        # Editors
        self.cell_tab = QWidget()
        self.geom_tab = QWidget()
        self._build_cell_tab()
        self._build_geom_tab()
        self.tabs.addTab(self.cell_tab, "Cell")
        self.tabs.addTab(self.geom_tab, "Geom")

        # Left-side panels: Workspace + Runs
        self.splitter = QSplitter()
        self.left_container = QWidget()
        left_v = QVBoxLayout(self.left_container)
        left_v.setContentsMargins(0, 0, 0, 0)

        # Workspace panel
        self.workspace_panel = QWidget()
        self._build_workspace_panel()
        left_v.addWidget(self.workspace_panel, 2)

        # Runs panel
        self.runs_panel = QWidget()
        self._build_runs_panel()
        left_v.addWidget(self.runs_panel, 3)

        # Assemble splitter and set as central widget (once)
        self.splitter.addWidget(self.left_container)
        self.splitter.addWidget(self.tabs)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.setCentralWidget(self.splitter)

        # Ensure current paths visible in UI (and preview built)
        # _build_workspace_panel() already populated the tree; refresh once more just in case.
        try:
            self._refresh_workspace_tree()
        except Exception:
            pass

        # Kick initial previews after the event loop starts
        QTimer.singleShot(0, lambda: self._update_command_preview(None))

        # After workspace tree exists, try autoloading first INI's latest run
        QTimer.singleShot(0, self._workspace_autoload_latest_on_startup)


    def _build_workspace_panel(self):
        """Workspace section: choose a workspace root, show INIs and their runs, batch/create actions."""
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
            QTreeWidget, QTreeWidgetItem
        )
        w = self.workspace_panel
        outer = QVBoxLayout(w)

        # Workspace root picker
        row = QHBoxLayout()
        row.addWidget(QLabel("Workspace root:"))
        ws_root_text = str(getattr(self, "workspace_root", self.run_root))
        self.edit_ws_root = QLineEdit(ws_root_text)
        btn_browse_ws = QPushButton("…")
        btn_browse_ws.setToolTip("Choose a directory that contains *.ini files")
        btn_browse_ws.clicked.connect(self._choose_workspace_root)
        row.addWidget(self.edit_ws_root, 1)
        row.addWidget(btn_browse_ws, 0)
        outer.addLayout(row)

        # Tree
        self.tree_ws = QTreeWidget()
        self.tree_ws.setHeaderLabels(["Workspace Files / Runs"])
        self.tree_ws.setColumnCount(1)
        self.tree_ws.currentItemChanged.connect(self._on_ws_tree_selection_changed)
        # Double-click to open run folder
        self.tree_ws.itemDoubleClicked.connect(self._on_ws_item_double_clicked)
        outer.addWidget(self.tree_ws, 1)

        # Buttons (actions)
        row2 = QHBoxLayout()
        self.btn_ws_new_run = QPushButton("New Run for Current INI")
        self.btn_ws_new_run.clicked.connect(self._ws_new_run_for_current_ini)

        self.btn_ws_broadcast = QPushButton("Broadcast settings to runs")
        self.btn_ws_broadcast.setToolTip("Write current Settings to all runs under the selected INI")
        self.btn_ws_broadcast.clicked.connect(self._ws_broadcast_settings_to_runs)

        self.btn_ws_pick_latest = QPushButton("Select newest run")
        self.btn_ws_pick_latest.setToolTip("Locate and select the newest run for the selected INI")
        self.btn_ws_pick_latest.clicked.connect(self._ws_select_newest_run_for_selected_ini)

        self.btn_ws_new_batch = QPushButton("New Batch Run")
        self.btn_ws_new_batch.clicked.connect(self._ws_create_new_batch_group)

        self.btn_ws_start_batch = QPushButton("Start Batch Indexing")
        self.btn_ws_start_batch.clicked.connect(self._ws_start_batch_indexing)

        row2.addWidget(self.btn_ws_new_run)
        row2.addWidget(self.btn_ws_broadcast)
        row2.addWidget(self.btn_ws_pick_latest)
        row2.addWidget(self.btn_ws_new_batch)
        row2.addWidget(self.btn_ws_start_batch)
        outer.addLayout(row2)

        # Initial population
        try:
            self.workspace_root = Path(self.edit_ws_root.text().strip()).resolve()
        except Exception:
            self.workspace_root = Path(str(self.run_root)).resolve()
            self.edit_ws_root.setText(str(self.workspace_root))
        self._refresh_workspace_tree()


    def _on_ws_item_double_clicked(self, item, _column):
        val = item.data(0, Qt.ItemDataRole.UserRole)
        if not val:
            return
        p = Path(val)
        if p.is_dir() and p.name.startswith("indexingintegration_"):
            # open run folder
            try:
                if sys.platform.startswith("darwin"):
                    subprocess.Popen(["open", str(p)])
                elif os.name == "nt":
                    os.startfile(str(p))  # type: ignore[attr-defined]
                else:
                    subprocess.Popen(["xdg-open", str(p)])
            except Exception as e:
                QMessageBox.warning(self, "Open folder failed", str(e))

    def _ws_broadcast_settings_to_runs(self):
        ini = getattr(self, "selected_ini", None)
        if not ini or not Path(ini).exists():
            QMessageBox.information(self, "Broadcast", "Select an INI in the workspace tree first.")
            return
        run_base = Path(self._resolve_run_root(str(ini)))
        if not run_base.exists():
            QMessageBox.information(self, "Broadcast", "Resolved run root does not exist.")
            return

        cfg = self._collect_settings_from_ui()
        runs = [p for p in run_base.glob("indexingintegration_*") if p.is_dir()]
        if not runs:
            QMessageBox.information(self, "Broadcast", "No runs found for the selected INI.")
            return

        ok_settings = 0
        ok_lists = 0
        h5_hint = self._find_h5_path_from_ini(str(ini))  # one-time lookup

        for rd in runs:
            try:
                with open(rd / "index_settings.json", "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                ok_settings += 1
            except Exception as e:
                append_line(self.txt_output, f"[broadcast] settings failed for {rd.name}: {e}")
            # refresh input.lst if we have a hint
            try:
                if h5_hint:
                    with open(rd / "input.lst", "w", encoding="utf-8") as f:
                        f.write(os.path.abspath(h5_hint))
                    ok_lists += 1
            except Exception as e:
                append_line(self.txt_output, f"[broadcast] input.lst failed for {rd.name}: {e}")

        append_line(self.txt_output, f"[broadcast] settings {ok_settings}/{len(runs)}, input.lst {ok_lists}/{len(runs)}")
        QMessageBox.information(self, "Broadcast", f"Wrote settings to {ok_settings} run(s); input.lst to {ok_lists} run(s).")


    def _ws_new_run_for_current_ini(self):
        """Create a new timestamped run for the INI currently selected in the tree."""
        ini = getattr(self, "selected_ini", None)
        if not ini or not Path(ini).exists():
            QMessageBox.information(self, "New Run", "Select an INI in the workspace tree first.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_base = self._resolve_run_root(str(ini))
        run_dir = Path(run_base) / f"indexingintegration_{ts}"
        try:
            ensure_dir(run_dir)
            # Write list file from INI heuristic (or empty)
            lst = run_dir / "input.lst"
            try:
                h5 = self._find_h5_path_from_ini(str(ini))
                with open(lst, "w", encoding="utf-8") as f:
                    f.write(os.path.abspath(h5) if h5 else "")
            except Exception as e:
                QMessageBox.warning(self, "Run", f"Created run dir, but failed to write input.lst:\n{e}")
            # Pre-touch run-local cell/geom (like original)
            cf = run_dir / "cellfile.cell"
            gf = run_dir / "geometry.geom"
            for f in (cf, gf):
                if not f.exists():
                    f.touch()
            # Reflect into UI
            self.selected_run_dir = run_dir
            self.edit_geom.setText(str(gf))
            self.edit_cell.setText(str(cf))
            self.edit_input_dir.setText(str(run_dir))  # iterator scans folder
            self._save_settings_to_run(run_dir)
            self._refresh_workspace_tree()
            self._update_command_preview(None)
        except Exception as e:
            QMessageBox.warning(self, "New Run", str(e))

    def _ws_create_new_batch_group(self):
        """Create a new timestamped run folder with the SAME name for every INI."""
        inis = self._get_workspace_ini_paths()
        if not inis:
            QMessageBox.warning(self, "Batch Run", "No INI files found under the workspace root.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        made = 0
        for ini in inis:
            try:
                rd = Path(self._resolve_run_root(ini)) / f"indexingintegration_{ts}"
                if not rd.is_dir():
                    rd.mkdir(parents=True, exist_ok=True)
                    # pre-touch placeholders
                    (rd / "cellfile.cell").touch(exist_ok=True)
                    (rd / "geometry.geom").touch(exist_ok=True)
                    made += 1
            except Exception:
                continue
        self._refresh_workspace_tree()
        QMessageBox.information(self, "Batch", f"Created batch run 'indexingintegration_{ts}' for {made} file(s).")

    def _choose_workspace_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select workspace root", str(self.workspace_root))
        if not path:
            return
        self.workspace_root = Path(path).resolve()
        self.edit_ws_root.setText(str(self.workspace_root))
        self._refresh_workspace_tree()

    def _refresh_workspace_tree(self):
        """Scan workspace_root for *.ini and list runs under each INI's run-root."""
        from PyQt6.QtWidgets import QTreeWidgetItem
        self.tree_ws.clear()
        root = getattr(self, "workspace_root", None)
        if not root or not Path(root).exists():
            return
        ini_paths = sorted(Path(root).rglob("*.ini"))
        for ini in ini_paths:
            ini_item = QTreeWidgetItem([ini.name])
            ini_item.setData(0, Qt.ItemDataRole.UserRole, str(ini))
            self.tree_ws.addTopLevelItem(ini_item)
            # list runs under resolved run base
            run_base = self._resolve_run_root(str(ini))
            if run_base and Path(run_base).exists():
                for rd in sorted(Path(run_base).glob("indexingintegration_*")):
                    if rd.is_dir():
                        run_item = QTreeWidgetItem([rd.name])
                        run_item.setData(0, Qt.ItemDataRole.UserRole, str(rd))
                        ini_item.addChild(run_item)
        self.tree_ws.expandAll()

    def _on_ws_tree_selection_changed(self, cur, _prev):
        """When a run node is selected, reflect into UI; when an INI is selected, remember it and auto-select its latest run."""
        if not cur:
            return
        val = cur.data(0, Qt.ItemDataRole.UserRole)
        if not val:
            return
        p = Path(val)
        # INI clicked
        if p.suffix.lower() == ".ini" and p.is_file():
            self.selected_ini = p
            # try to auto-select newest run for this INI
            try:
                rd = self._select_newest_run_for_ini(p, also_focus_tree=True)
                if rd:
                    self._apply_run_dir(rd)
                    self._update_command_preview(None)
                else:
                    # no runs yet; try to pre-fill input/geom/cell from INI context
                    h5 = self._find_h5_path_from_ini(str(p))
                    if h5:
                        self.edit_input_dir.setText(str(Path(h5).parent))
                    self._update_command_preview(None)
            except Exception:
                pass
            return

        # Run folder clicked
        if p.is_dir() and p.name.startswith("indexingintegration_"):
            self.selected_run_dir = p
            # load settings.json if present
            s = p / "index_settings.json"
            if s.exists():
                try:
                    with open(s, "r", encoding="utf-8") as f:
                        self._apply_settings_to_ui(json.load(f))
                except Exception as e:
                    append_line(self.txt_output, f"[warn] failed to load settings: {e}")
            else:
                # try to seed sensible defaults (geom/cell from run, input from run or INI)
                self._load_index_settings_or_defaults(p, getattr(self, "selected_ini", None))

            # reflect key file paths
            g = p / "geometry.geom"
            c = p / "cellfile.cell"
            lst = p / "input.lst"
            if g.exists(): self.edit_geom.setText(str(g))
            if c.exists(): self.edit_cell.setText(str(c))
            if lst.exists(): self.edit_input_dir.setText(str(lst.parent))
            self._update_command_preview(None)

    def _load_index_settings_or_defaults(self, run_dir: Path, ini_path: Path | None):
        """
        If index_settings.json is missing:
        - keep current UI values,
        - but try to prefill input folder (from run_dir/input.lst or INI’s h5 location)
        - ensure out_base is at least the run folder name suffix.
        """
        # input folder from run_dir/input.lst (if present and has a line)
        lst = run_dir / "input.lst"
        try:
            if lst.exists():
                with open(lst, "r", encoding="utf-8") as f:
                    first = f.readline().strip()
                if first:
                    self.edit_input_dir.setText(str(Path(first).parent))
            elif ini_path and Path(ini_path).exists():
                h5 = self._find_h5_path_from_ini(str(ini_path))
                if h5:
                    self.edit_input_dir.setText(str(Path(h5).parent))
        except Exception:
            pass

        # sensible out_base default
        if not self.edit_out_base.text().strip():
            try:
                self.edit_out_base.setText(run_dir.name.split("indexingintegration_")[-1] or "output")
            except Exception:
                self.edit_out_base.setText("output")

    def _list_runs_for_ini(self, ini_path: Path) -> list[Path]:
        run_base = Path(self._resolve_run_root(str(ini_path)))
        if not run_base.exists():
            return []
        return [p for p in run_base.glob("indexingintegration_*") if p.is_dir()]

    def _find_newest_run(self, runs: list[Path]) -> Path | None:
        if not runs:
            return None
        # Try by mtime; if equal, fall back to name sort
        runs_sorted = sorted(runs, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
        return runs_sorted[0]

    def _select_newest_run_for_ini(self, ini_path: Path, also_focus_tree: bool = False) -> Path | None:
        runs = self._list_runs_for_ini(ini_path)
        newest = self._find_newest_run(runs)
        if newest and also_focus_tree:
            # walk the tree and select this item if present
            top_count = self.tree_ws.topLevelItemCount()
            for i in range(top_count):
                ini_item = self.tree_ws.topLevelItem(i)
                ini_val = ini_item.data(0, Qt.ItemDataRole.UserRole)
                if ini_val and Path(ini_val) == ini_path:
                    for j in range(ini_item.childCount()):
                        child = ini_item.child(j)
                        val = child.data(0, Qt.ItemDataRole.UserRole)
                        if val and Path(val) == newest:
                            self.tree_ws.setCurrentItem(child)
                            break
                    break
        return newest

    def _ws_select_newest_run_for_selected_ini(self):
        ini = getattr(self, "selected_ini", None)
        if not ini or not Path(ini).exists():
            QMessageBox.information(self, "Workspace", "Select an INI in the workspace tree first.")
            return
        rd = self._select_newest_run_for_ini(ini, also_focus_tree=True)
        if rd:
            self._apply_run_dir(rd)
            append_line(self.txt_output, f"[workspace] selected newest run: {rd.name}")
        else:
            QMessageBox.information(self, "Workspace", "No runs found for this INI yet.")

    def _get_workspace_ini_paths(self) -> list[str]:
        root = getattr(self, "workspace_root", None)
        if not root or not Path(root).exists():
            return []
        return [str(p) for p in sorted(Path(root).rglob("*.ini"))]

    def _resolve_run_root(self, ini_path: str) -> str:
        """Match original logic: run root = base_dir or base_dir/outputfolder if present in INI."""
        try:
            parser = ConfigParser()
            parser.read(ini_path)
            base_dir = os.path.dirname(os.path.abspath(ini_path))
            output_folder = parser.get("Paths", "outputfolder", fallback="").strip()
            return os.path.join(base_dir, output_folder) if output_folder else base_dir
        except Exception:
            return os.path.dirname(os.path.abspath(ini_path))

    def _find_h5_path_from_ini(self, ini_path: str) -> str | None:
        """
        Better heuristic:
        1) Read INI for any path-like values in likely sections/keys.
        2) Search these dirs (and immediate subdirs) for *.h5/*.hdf5/*.cxi.
        3) Fallback: sibling and one-level child dirs of the INI's folder.
        Returns absolute path of the first match or None.
        """
        base = Path(os.path.dirname(os.path.abspath(ini_path)))

        # 1) Collect candidate directories from INI
        cand_dirs: list[Path] = []
        try:
            parser = ConfigParser()
            parser.read(ini_path)
            # common sections/keys used in various pipelines
            likely_keys = {"datafolder", "inputfolder", "datadir", "folder", "path", "h5dir", "cxi_dir"}
            for section in parser.sections():
                for key, val in parser.items(section):
                    v = (val or "").strip().strip('"').strip("'")
                    if not v:
                        continue
                    p = Path(v)
                    # expand relative to INI folder
                    if not p.is_absolute():
                        p = (base / p).resolve()
                    if p.is_dir() and p not in cand_dirs:
                        cand_dirs.append(p)
        except Exception:
            pass

        # Always search the INI's folder first
        cand_dirs.insert(0, base)

        def scan_dir_once(d: Path) -> Optional[Path]:
            for ext in ("*.h5", "*.hdf5", "*.cxi"):
                hits = sorted(d.glob(ext))
                if hits:
                    return hits[0].resolve()
            # immediate subdirs
            for sub in sorted([p for p in d.iterdir() if p.is_dir()]):
                for ext in ("*.h5", "*.hdf5", "*.cxi"):
                    hits = sorted(sub.glob(ext))
                    if hits:
                        return hits[0].resolve()
            return None

        # 2) search candidate dirs
        seen = set()
        for d in cand_dirs:
            if not d.exists():
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            hit = scan_dir_once(d)
            if hit:
                return str(hit)

        # 3) fallback: siblings of INI folder
        for sib in sorted([p for p in base.parent.iterdir() if p.is_dir()]):
            hit = scan_dir_once(sib)
            if hit:
                return str(hit)

        return None

    # -----------------
    # UI: Settings tab
    # -----------------

    def _update_peak_params(self, name: str):
        """
        Fill the Peakfinder params editor based on the selected preset.
        Safe during early init: only triggers preview if txt_preview exists.
        """
        txt = default_peakfinder_options.get(name, "")
        try:
            self.peak_params_edit.blockSignals(True)
            self.peak_params_edit.setPlainText(txt)
        finally:
            self.peak_params_edit.blockSignals(False)
        if hasattr(self, "txt_preview") and self.txt_preview:
            self._update_command_preview(None)
            
    def _toggle_peak_params_visibility(self, name: str):
        """
        Hide the params editor for 'cxi' (only needs --peaks=cxi),
        show it for 'peakfinder9'/'peakfinder8'.
        """
        show = name in ("peakfinder9", "peakfinder8")
        self.peak_params_edit.setVisible(show)

    def _build_settings_tab(self):
        w = self.settings_tab

        # Make the whole settings page scrollable (helps on macOS/smaller windows)
        scroll = QScrollArea(w)
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer = QVBoxLayout(content)  # build page on 'content'

        # ----------------------
        # Files group
        # ----------------------
        files_group = QGroupBox("Files")
        files_form = QFormLayout(files_group)

        # Optional full path to indexamajig
        self.edit_idxmj = QLineEdit("")
        files_form.addRow("indexamajig path (optional):", self.edit_idxmj)

        # Geometry / Cell / Input
        self.edit_geom = QLineEdit()
        if getattr(self, "default_geom", None) and self.default_geom.exists():
            self.edit_geom.setText(str(self.default_geom))
        btn_geom = QPushButton("Browse…")
        btn_geom.clicked.connect(lambda: self._browse_to(self.edit_geom, "Geometry (*.geom)"))
        files_form.addRow("Geometry (.geom):", self._hpair(self.edit_geom, btn_geom))

        self.edit_cell = QLineEdit()
        if getattr(self, "default_cell", None) and self.default_cell.exists():
            self.edit_cell.setText(str(self.default_cell))
        btn_cell = QPushButton("Browse…")
        btn_cell.clicked.connect(lambda: self._browse_to(self.edit_cell, "Cell (*.cell)"))
        files_form.addRow("Cell (.cell):", self._hpair(self.edit_cell, btn_cell))

        self.edit_input_dir = QLineEdit()
        if getattr(self, "default_input", None) and self.default_input.exists():
            self.edit_input_dir.setText(str(self.default_input))
        btn_input = QPushButton("Browse…")
        btn_input.clicked.connect(lambda: self._browse_dir(self.edit_input_dir))
        files_form.addRow("Input folder (HDF5s):", self._hpair(self.edit_input_dir, btn_input))

        self.edit_out_base = QLineEdit("Xtal")
        files_form.addRow("Output base:", self.edit_out_base)

        outer.addWidget(files_group)

        # ----------------------
        # Indexing & Integration
        # ----------------------
        idx_group = QGroupBox("Indexing & Integration")
        idx_form = QFormLayout(idx_group)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 256)
        self.spin_threads.setValue(max(1, os.cpu_count() or 8))
        idx_form.addRow("Threads:", self.spin_threads)

        grid_title = QLabel("SerialED Grid")
        grid_title.setStyleSheet("font-weight:600;")
        idx_form.addRow(grid_title)

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

        # ----------------------
        # Flags (Peakfinder + Advanced + Other)
        # ----------------------
        flags_group = QGroupBox("Flags")
        flags_layout = QGridLayout(flags_group)

        # Peakfinder block (dropdown + multiline params)
        peak_group = QGroupBox("Peakfinder Options")
        pg = QGridLayout(peak_group)

        pg.addWidget(QLabel("Peakfinder:"), 0, 0)
        self.peak_combo = QComboBox()
        self.peak_combo.addItems(["cxi", "peakfinder9", "peakfinder8"])
        # width adapts to content
        self.peak_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        pg.addWidget(self.peak_combo, 0, 1)

        pg.addWidget(QLabel("Peakfinder Params:"), 1, 0, Qt.AlignmentFlag.AlignTop)
        self.peak_params_edit = QTextEdit()
        self.peak_params_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.peak_params_edit.setMinimumHeight(100)  # make it clearly visible
        pg.addWidget(self.peak_params_edit, 1, 1)

        # initial text and live updates
        self._update_peak_params(self.peak_combo.currentText())
        self._toggle_peak_params_visibility(self.peak_combo.currentText())
        self.peak_combo.currentTextChanged.connect(self._update_peak_params)
        self.peak_combo.currentTextChanged.connect(self._toggle_peak_params_visibility)
        self.peak_params_edit.textChanged.connect(lambda: self._update_command_preview(None))

        # add peakfinder group to flags grid
        flags_layout.addWidget(peak_group, flags_layout.rowCount(), 0, 1, 2)

        # Advanced flags (multi-line)
        flags_layout.addWidget(QLabel("Advanced flags:"), flags_layout.rowCount(), 0)
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
        flags_layout.addWidget(self.txt_advanced, flags_layout.rowCount() - 1, 1)

        # Other flags (multi-line)
        flags_layout.addWidget(QLabel("Other flags:"), flags_layout.rowCount(), 0)
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
        flags_layout.addWidget(self.txt_other, flags_layout.rowCount() - 1, 1)

        outer.addWidget(flags_group)

        # Live preview updates when settings change
        for wdg in (self.edit_geom, self.edit_cell, self.edit_input_dir, self.edit_out_base):
            wdg.textChanged.connect(lambda *_: self._update_command_preview(None))
        self.spin_threads.valueChanged.connect(lambda *_: self._update_command_preview(None))
        self.spin_max_radius.valueChanged.connect(lambda *_: self._update_command_preview(None))
        self.spin_step.valueChanged.connect(lambda *_: self._update_command_preview(None))
        self.txt_advanced.textChanged.connect(lambda *_: self._update_command_preview(None))
        self.txt_other.textChanged.connect(lambda *_: self._update_command_preview(None))

        outer.addStretch(1)

        # Mount the scroll area into the tab widget
        tab_layout = QVBoxLayout(w)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)


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

        # Aux: clear log, refresh streams
        self.btn_clear_log = QPushButton("Clear log")
        self.btn_clear_log.clicked.connect(lambda: self.txt_output.setPlainText(""))
        self.btn_refresh_streams = QPushButton("Refresh streams")
        self.btn_refresh_streams.clicked.connect(self._refresh_streams_list)
        btns.addWidget(self.btn_clear_log)
        btns.addWidget(self.btn_refresh_streams)
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
        prev_lay = QHBoxLayout(prev_group)
        self.txt_preview = QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt_preview.setMaximumHeight(80)
        prev_lay.addWidget(self.txt_preview, 1)
        col = QVBoxLayout()
        self.btn_copy_preview = QPushButton("Copy")
        self.btn_copy_preview.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_preview.toPlainText()))
        self.btn_open_streams = QPushButton("Open streams folder")
        self.btn_open_streams.clicked.connect(self._open_streams_folder)
        self.btn_open_latest_stream = QPushButton("Open latest stream")
        self.btn_open_latest_stream.clicked.connect(self._open_latest_stream_file)
        self.btn_merge_streams = QPushButton("Merge streams now")
        self.btn_merge_streams.clicked.connect(self._merge_streams_now_clicked)
        self.btn_export_run = QPushButton("Export run (.zip)")
        self.btn_export_run.clicked.connect(self._export_run_bundle)
        col.addWidget(self.btn_copy_preview)
        col.addWidget(self.btn_open_streams)
        col.addWidget(self.btn_open_latest_stream)
        col.addWidget(self.btn_merge_streams)
        col.addWidget(self.btn_export_run)
        col.addStretch(1)
        prev_lay.addLayout(col)
        outer.addWidget(prev_group)

        # Streams browser
        from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem
        streams_group = QGroupBox("Streams in current run")
        streams_lay = QVBoxLayout(streams_group)
        self.tree_streams = QTreeWidget()
        self.tree_streams.setColumnCount(3)
        self.tree_streams.setHeaderLabels(["File", "Size (MB)", "Modified"])
        self.tree_streams.itemDoubleClicked.connect(self._on_stream_item_double_clicked)
        streams_lay.addWidget(self.tree_streams)
        outer.addWidget(streams_group, 1)

        # Output log
        self.txt_output = QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt_output.setMinimumHeight(220)
        outer.addWidget(self.txt_output, 1)

    def _streams_dir_for_active(self) -> Optional[Path]:
        rd = self._resolve_active_run_dir()
        return (rd / "streams") if rd else None

    def _refresh_streams_list(self):
        from PyQt6.QtWidgets import QTreeWidgetItem
        self.tree_streams.clear()
        sdir = self._streams_dir_for_active()
        if not sdir or not sdir.exists():
            return
        files = sorted(sdir.glob("*.stream"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                sz_mb = p.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                it = QTreeWidgetItem([p.name, f"{sz_mb:.2f}", mtime])
                it.setData(0, Qt.ItemDataRole.UserRole, str(p))
                self.tree_streams.addTopLevelItem(it)
            except Exception:
                continue

    def _on_stream_item_double_clicked(self, item, _col):
        val = item.data(0, Qt.ItemDataRole.UserRole)
        if not val:
            return
        p = Path(val)
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(p)])
            elif os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            QMessageBox.warning(self, "Open stream failed", str(e))

    def _open_latest_stream_file(self):
        sdir = self._streams_dir_for_active()
        if not sdir or not sdir.exists():
            QMessageBox.information(self, "Streams", "No streams folder for the active run.")
            return
        files = list(sdir.glob("*.stream"))
        if not files:
            QMessageBox.information(self, "Streams", "No .stream files found yet.")
            return
        latest = max(files, key=lambda p: p.stat().st_mtime)
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(latest)])
            elif os.name == "nt":
                os.startfile(str(latest))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(latest)])
        except Exception as e:
            QMessageBox.warning(self, "Open latest stream failed", str(e))


    def _build_cell_tab(self):
        lay = QVBoxLayout(self.cell_tab)

        # Toolbar
        bar = QHBoxLayout()
        btn_load = QPushButton("Load from path")
        btn_save = QPushButton("Save to path")
        btn_save_as_run = QPushButton("Save As… (into run dir)")
        bar.addWidget(btn_load)
        bar.addWidget(btn_save)
        bar.addWidget(btn_save_as_run)
        bar.addStretch(1)
        lay.addLayout(bar)

        # Editor
        self.cell_editor = QPlainTextEdit()
        self.cell_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.cell_editor, 1)

        # Wire actions
        btn_load.clicked.connect(self._cell_load_from_path)
        btn_save.clicked.connect(self._cell_save_to_path)
        btn_save_as_run.clicked.connect(self._cell_save_as_into_run)


    def _build_geom_tab(self):
        lay = QVBoxLayout(self.geom_tab)

        # Toolbar
        bar = QHBoxLayout()
        btn_load = QPushButton("Load from path")
        btn_save = QPushButton("Save to path")
        btn_save_as_run = QPushButton("Save As… (into run dir)")
        bar.addWidget(btn_load)
        bar.addWidget(btn_save)
        bar.addWidget(btn_save_as_run)
        bar.addStretch(1)
        lay.addLayout(bar)

        # Editor
        self.geom_editor = QPlainTextEdit()
        self.geom_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self.geom_editor, 1)

        # Wire actions
        btn_load.clicked.connect(self._geom_load_from_path)
        btn_save.clicked.connect(self._geom_save_to_path)
        btn_save_as_run.clicked.connect(self._geom_save_as_into_run)

    # ---- Cell helpers ----
    def _cell_load_from_path(self):
        path = self.edit_cell.text().strip()
        if not path:
            QMessageBox.information(self, "Cell", "Set a cell file path on the Settings tab first.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.cell_editor.setPlainText(f.read())
            append_line(self.txt_output, f"[cell] loaded: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Load cell failed", str(e))

    def _cell_save_to_path(self):
        path = self.edit_cell.text().strip()
        if not path:
            QMessageBox.information(self, "Cell", "Set a cell file path on the Settings tab first.")
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.cell_editor.toPlainText())
            append_line(self.txt_output, f"[cell] saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save cell failed", str(e))

    def _cell_save_as_into_run(self):
        run_dir = getattr(self, "selected_run_dir", None) or (self.current_run.run_dir if getattr(self, "current_run", None) else None)
        if not run_dir:
            QMessageBox.information(self, "Cell", "Select or create a run first (left panel).")
            return
        default = (Path(run_dir) / "cellfile.cell").as_posix()
        path, _ = QFileDialog.getSaveFileName(self, "Save cell into run", default, "Cell (*.cell);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.cell_editor.toPlainText())
            self.edit_cell.setText(path)
            append_line(self.txt_output, f"[cell] saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save cell failed", str(e))

    # ---- Geometry helpers ----
    def _geom_load_from_path(self):
        path = self.edit_geom.text().strip()
        if not path:
            QMessageBox.information(self, "Geometry", "Set a geometry file path on the Settings tab first.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.geom_editor.setPlainText(f.read())
            append_line(self.txt_output, f"[geom] loaded: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Load geometry failed", str(e))

    def _geom_save_to_path(self):
        path = self.edit_geom.text().strip()
        if not path:
            QMessageBox.information(self, "Geometry", "Set a geometry file path on the Settings tab first.")
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.geom_editor.toPlainText())
            append_line(self.txt_output, f"[geom] saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save geometry failed", str(e))

    def _geom_save_as_into_run(self):
        run_dir = getattr(self, "selected_run_dir", None) or (self.current_run.run_dir if getattr(self, "current_run", None) else None)
        if not run_dir:
            QMessageBox.information(self, "Geometry", "Select or create a run first (left panel).")
            return
        default = (Path(run_dir) / "geometry.geom").as_posix()
        path, _ = QFileDialog.getSaveFileName(self, "Save geometry into run", default, "Geometry (*.geom);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.geom_editor.toPlainText())
            self.edit_geom.setText(path)
            append_line(self.txt_output, f"[geom] saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save geometry failed", str(e))

    def _build_runs_panel(self):
        """Left-side runs panel: choose run root, list runs, open and create."""
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
            QListWidget, QListWidgetItem
        )
        w = self.runs_panel
        outer = QVBoxLayout(w)

        # Run root chooser
        row = QHBoxLayout()
        row.addWidget(QLabel("Run root:"))
        self.edit_run_root = QLineEdit(str(self.run_root))
        btn_browse_root = QPushButton("…")
        btn_browse_root.setToolTip("Choose run root directory")
        btn_browse_root.clicked.connect(self._choose_run_root)
        row.addWidget(self.edit_run_root, 1)
        row.addWidget(btn_browse_root, 0)
        outer.addLayout(row)

        # Buttons: new run, open folder
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

        # --- Batch queue (simple scaffold) ---
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

        # Initial population
        self._refresh_runs_tree()

    def _ws_start_batch_indexing(self):
        """Queue and run all INIs with a NEW timestamped run name. Writes settings and seeds input.lst."""
        inis = self._get_workspace_ini_paths()
        if not inis:
            QMessageBox.information(self, "Batch Run", "No INI files found under the workspace root.")
            return
        group = f"indexingintegration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cfg = self._collect_settings_from_ui()
        self._ws_batch_queue = []
        created_dirs = 0
        created_lists = 0

        for ini in inis:
            try:
                run_base = self._resolve_run_root(ini)
                rd = Path(run_base) / group
                rd.mkdir(parents=True, exist_ok=True)
                created_dirs += 1
                # settings
                try:
                    with open(rd / "index_settings.json", "w", encoding="utf-8") as jf:
                        json.dump(cfg, jf, indent=2)
                except Exception:
                    pass
                # seed list from INI heuristic
                try:
                    lst = rd / "input.lst"
                    if not lst.exists():
                        h5 = self._find_h5_path_from_ini(ini)
                        if h5:
                            with open(lst, "w", encoding="utf-8") as f:
                                f.write(os.path.abspath(h5))
                            created_lists += 1
                except Exception:
                    pass
                self._ws_batch_queue.append(rd)
            except Exception:
                continue

        # switch UI to current selection if possible
        if self._ws_batch_queue:
            self._apply_run_dir(self._ws_batch_queue[0])

        self._refresh_workspace_tree()
        # Auto-pick first INI and newest run on startup (if any)
        QTimer.singleShot(0, self._workspace_autoload_latest_on_startup)

        if not self._ws_batch_queue:
            QMessageBox.information(self, "Batch Run", "No runs could be prepared for batch execution.")
            return

        info_bits = [f"created {created_dirs} run folder(s)"]
        if created_lists:
            info_bits.append(f"prepared {created_lists} input.lst file(s)")
        append_line(self.txt_output, "[batch] " + "; ".join(info_bits))

        # begin
        self._ws_batch_mode = True
        self._ws_start_next_in_batch()

    def _workspace_autoload_latest_on_startup(self):
        try:
            # pick first INI
            top = self.tree_ws.topLevelItem(0)
            if not top:
                return
            ini_val = top.data(0, Qt.ItemDataRole.UserRole)
            if not ini_val:
                return
            ini_path = Path(ini_val)
            self.selected_ini = ini_path
            # select newest run for it (and focus)
            rd = self._select_newest_run_for_ini(ini_path, also_focus_tree=True)
            if rd:
                self._apply_run_dir(rd)
            else:
                # no runs yet; try to seed input from INI
                h5 = self._find_h5_path_from_ini(str(ini_path))
                if h5:
                    self.edit_input_dir.setText(str(Path(h5).parent))
            self._update_command_preview(None)
        except Exception:
            pass

    def _apply_run_dir(self, run_dir: Path):
        """Reflect run_dir into the Settings tab and select it everywhere."""
        self.selected_run_dir = Path(run_dir)
        # Prefer local cell/geom if exist
        gf = self.selected_run_dir / "geometry.geom"
        cf = self.selected_run_dir / "cellfile.cell"
        lst = self.selected_run_dir / "input.lst"
        if gf.exists(): self.edit_geom.setText(str(gf))
        if cf.exists(): self.edit_cell.setText(str(cf))
        if lst.exists(): self.edit_input_dir.setText(str(self.selected_run_dir))
        self._update_command_preview(None)
        self._set_window_title_for_run(run_dir)


    def _ws_start_next_in_batch(self):
        if not getattr(self, "_ws_batch_queue", []):
            self._ws_batch_mode = False
            QMessageBox.information(self, "Batch Run", "Batch finished.")
            return
        rd = self._ws_batch_queue.pop(0)
        self._apply_run_dir(rd)
        # trigger a normal grid run (this will save settings, write streams dir, etc.)
        self._start_grid_clicked()
        # chain next on finish
        # (your existing _on_proc_finished already continues your other batch;
        # we keep a separate flag to chain here as well)
        self._ws_chain_on_finish = True

    def _resolve_active_run_dir(self) -> Optional[Path]:
        """Prefer current running context; otherwise use the last selected run."""
        if getattr(self, "current_run", None):
            return self.current_run.run_dir
        return getattr(self, "selected_run_dir", None)

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
        """
        Naive .stream concatenation stub:
        - finds run_dir/streams/*.stream
        - concatenates into run_dir/<out_name or 'merged.stream'>
        Returns the output path.
        """
        streams_dir = run_dir / "streams"
        files = sorted(streams_dir.glob("*.stream"))
        if not files:
            raise RuntimeError(f"No .stream files in {streams_dir}")

        # Choose output name
        if out_name:
            out_path = run_dir / out_name
        else:
            # If we have a current_run with out_base, prefer that for the merged name
            base = None
            if getattr(self, "current_run", None) and self.current_run.run_dir == run_dir:
                base = self.current_run.out_base
            out_path = run_dir / (f"{base}_merged.stream" if base else "merged.stream")

        # Concatenate as text; add a newline between files if needed
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

        # Build a temporary staging dir to ensure external refs are included
        stage = Path(tempfile.mkdtemp(prefix="run_export_"))
        try:
            # Copy run folder contents
            staged_run = stage / run_dir.name
            shutil.copytree(run_dir, staged_run)

            # Bring in external refs (geom/cell) if they are outside run_dir
            refs_dir = staged_run / "refs"
            refs_added = False
            geom_path = Path(self.edit_geom.text().strip()) if self.edit_geom.text().strip() else None
            cell_path = Path(self.edit_cell.text().strip()) if self.edit_cell.text().strip() else None
            for p in (geom_path, cell_path):
                if p and p.exists() and not str(p.resolve()).startswith(str(run_dir.resolve())):
                    ensure_dir(refs_dir)
                    shutil.copy2(p, refs_dir / p.name)
                    refs_added = True

            # Create zip next to the run dir
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
            # Clean staging
            try:
                shutil.rmtree(stage, ignore_errors=True)
            except Exception:
                pass


    def _choose_run_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select run root", str(self.run_root))
        if path:
            self.run_root = Path(path)
            ensure_dir(self.run_root)
            self.edit_run_root.setText(str(self.run_root))
            self._refresh_runs_tree()

    def _create_new_run_dir(self):
        run_dir = timestamp_run_dir(self.run_root)
        ensure_dir(run_dir)
        # Pre-create streams dir and empty input.lst
        ensure_dir(run_dir / "streams")
        write_text(run_dir / "input.lst", "")
        self._refresh_runs_tree()
        # Select it and load (which will reflect paths into UI)
        self._select_run_in_list(run_dir)

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

    def _refresh_runs_tree(self):
        self.list_runs.clear()
        if not self.run_root.exists():
            return
        for p in sorted(self.run_root.glob("indexingintegration_*")):
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
        # Load settings.json if present; populate UI; fill file paths
        settings_path = run_dir / "index_settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._apply_settings_to_ui(data)
            except Exception as e:
                append_line(self.txt_output, f"[warn] failed to load settings: {e}")
        # Populate file widgets to point into this run by default
        # (User can still change them)
        # geom/cell/input folder are stored in settings; leave as-is if present
        self.current_run = None  # clear any old context
        # reflect preview from current UI (no ctx yet)
        self.selected_run_dir = run_dir
        self._update_command_preview(None)


    def _collect_settings_from_ui(self) -> dict:
        return {
            "indexamajig_path": self.edit_idxmj.text().strip(),
            "geom_path": self.edit_geom.text().strip(),
            "cell_path": self.edit_cell.text().strip(),
            "input_dir": self.edit_input_dir.text().strip(),
            "out_base": self.edit_out_base.text().strip(),
            "threads": int(self.spin_threads.value()),
            "max_radius": float(self.spin_max_radius.value()),
            "step": float(self.spin_step.value()),
            "advanced_flags": self.txt_advanced.toPlainText(),
            "other_flags": self.txt_other.toPlainText(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "peakfinder_choice": self.peak_combo.currentText(),
            "peakfinder_params": self.peak_params_edit.toPlainText(),

        }

    def _apply_settings_to_ui(self, data: dict):
        def set_text(le: QLineEdit, key: str):
            if key in data and isinstance(data[key], str):
                le.setText(data[key])

        set_text(self.edit_geom, "geom_path")
        set_text(self.edit_cell, "cell_path")
        set_text(self.edit_input_dir, "input_dir")
        set_text(self.edit_out_base, "out_base")

        if "threads" in data:
            try: self.spin_threads.setValue(int(data["threads"]))
            except: pass
        if "max_radius" in data:
            try: self.spin_max_radius.setValue(float(data["max_radius"]))
            except: pass
        if "step" in data:
            try: self.spin_step.setValue(float(data["step"]))
            except: pass
            
        if "peakfinder" in data and "peakfinder_params" not in data:
            legacy = str(data["peakfinder"]).strip()
            if legacy in default_peakfinder_options:
                self.peak_combo.setCurrentText(legacy)
                self.peak_params_edit.setPlainText(default_peakfinder_options[legacy])
            else:
                self.peak_params_edit.setPlainText(legacy)

        if "peakfinder_choice" in data and isinstance(data["peakfinder_choice"], str):
            self.peak_combo.setCurrentText(data["peakfinder_choice"])

        if "peakfinder_params" in data and isinstance(data["peakfinder_params"], str):
            self.peak_params_edit.setPlainText(data["peakfinder_params"])

        if "advanced_flags" in data and isinstance(data["advanced_flags"], str):
            self.txt_advanced.setPlainText(data["advanced_flags"])
        if "other_flags" in data and isinstance(data["other_flags"], str):
            self.txt_other.setPlainText(data["other_flags"])
        if "indexamajig_path" in data and isinstance(data["indexamajig_path"], str):
            self.edit_idxmj.setText(data["indexamajig_path"])

    def _save_settings_to_run(self, run_dir: Path):
        data = self._collect_settings_from_ui()
        try:
            with open(run_dir / "index_settings.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            append_line(self.txt_output, f"[warn] failed to save settings: {e}")

    def _build_command_preview(self, ctx: RunContext) -> str:
        flags = self._compose_extra_flags()
        base = (
            f"indexamajig -g {ctx.geom_path} -i {ctx.list_path} "
            f"-o {(ctx.run_dir / 'streams' / ctx.out_base)}.stream "
            f"-p {ctx.cell_path} -j {ctx.threads}"
        )
        if flags:
            base += " " + " ".join(flags)
        passes = estimate_grid_points(ctx.max_radius, ctx.step)
        base += f" | [grid] R={ctx.max_radius:g}px, step={ctx.step:g}px → {passes} passes"
        return base

    def _update_command_preview(self, _evt):
        """
        Rebuild the command preview so it matches the filled step-lattice circle used
        by gandalf_iterator and estimate_grid_points. If a RunContext exists, show
        the precise per-run preview (streams path + exact pass count).
        """
        try:
            if not hasattr(self, "txt_preview") or self.txt_preview is None:
                return

            # If we have an active RunContext, show the precise command + grid info.
            if getattr(self, "current_run", None):
                try:
                    self.txt_preview.setPlainText(self._build_command_preview(self.current_run))
                    return
                except Exception as e:
                    self.txt_preview.setPlainText(f"[preview error] {e}")
                    return

            # Fallback simplified preview (no RunContext yet)
            geom = self.edit_geom.text().strip()
            cell = self.edit_cell.text().strip()
            out_base = self.edit_out_base.text().strip()
            threads = str(self.spin_threads.value())

            # Prefer run's list file if available; else input folder path as entered
            if getattr(self, "selected_run_dir", None):
                cand = self.selected_run_dir / "input.lst"
                lst = str(cand) if cand.exists() else self.edit_input_dir.text().strip()
            else:
                lst = self.edit_input_dir.text().strip()

            flags = self._compose_extra_flags()
            base = f"indexamajig -g {geom} -i {lst} -o {out_base}.stream -p {cell} -j {threads}"
            if flags:
                base += " " + " ".join(flags)

            # Exact grid count for step-lattice points within the circle
            try:
                R = float(self.spin_max_radius.value())
                S = float(self.spin_step.value())
                passes = estimate_grid_points(R, S)
                base += f" | [grid] R={R:g}px, step={S:g}px → {passes} passes"
            except Exception:
                pass

            self.txt_preview.setPlainText(base)
        except Exception as e:
            if hasattr(self, "txt_preview") and self.txt_preview:
                self.txt_preview.setPlainText(f"[preview error] {e}")


    def _preflight_validate(self) -> bool:
        """Check that required paths exist and are writable; warn user if not."""
        msgs = []
        gp = Path(self.edit_geom.text().strip())
        cp = Path(self.edit_cell.text().strip())
        ip = Path(self.edit_input_dir.text().strip())
        out_base = self.edit_out_base.text().strip()
        if not gp.is_file(): msgs.append("Geometry file is missing.")
        if not cp.is_file(): msgs.append("Cell file is missing.")
        if not ip.is_dir(): msgs.append("Input folder is missing.")
        if not out_base: msgs.append("Output base is empty.")
        if float(self.spin_step.value()) <= 0: msgs.append("Step must be > 0.")
        if float(self.spin_max_radius.value()) < 0: msgs.append("Max radius must be ≥ 0.")

        rd = self._resolve_active_run_dir() or self.run_root
        try:
            ensure_dir(rd)
            test_path = Path(rd) / ".write_test"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
        except Exception:
            msgs.append(f"Run directory not writable: {rd}")

        if msgs:
            QMessageBox.warning(self, "Preflight", "\n".join(msgs))
            return False
        return True

    def _batch_add_current(self):
        data = self._collect_settings_from_ui()
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
        # Pop next job and apply UI, then start a grid run
        raw = self.batch_queue.pop(0)
        try:
            data = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            data = {}
        self._apply_settings_to_ui(data)
        ensure_dir(self.run_root)
        self._start_grid_clicked()

    # ==============
    # Path pickers
    # ==============
    def _browse_to(self, lineedit: QLineEdit, filter_str: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", str(Path.cwd()), filter_str)
        if path:
            lineedit.setText(path)

    def _browse_dir(self, lineedit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select folder", str(Path.cwd()))
        if path:
            lineedit.setText(path)

    def _hpair(self, a: QWidget, b: QWidget) -> QWidget:
        cw = QWidget()
        lay = QHBoxLayout(cw)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(a, 1)
        lay.addWidget(b, 0)
        return cw

    # =====================
    # Start/Stop click flow
    # =====================
    def _start_grid_clicked(self):
        # Prevent double-start while running
        if self.proc and self.proc.poll() is None:
            QMessageBox.information(self, "Run", "A job is already running.")
            return
        # Validate inputs
        if not self._preflight_validate():
            return

        # Validate inputs
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

        # Compose flags
        extra_flags = self._compose_extra_flags()
        # Create run dir and materialize input.lst
        run_dir = timestamp_run_dir(self.run_root)
        ensure_dir(run_dir)
        list_path = run_dir / "input.lst"
        total_images = self._write_input_list(in_dir, list_path)
        ensure_dir(run_dir / "streams")
        self._save_settings_to_run(run_dir)

        # Progress scale
        est_passes = estimate_grid_points(max_radius, step)
        per_pass = total_images
        self.per_pass_images = per_pass
        self.estimated_passes = max(1, est_passes)
        self.processed_images_total = 0
        self.last_seen_processed = 0
        self.seen_passes = 0

        # Stash context
        self.current_run = RunContext(
            run_dir=run_dir, geom_path=geom, cell_path=cell,
            input_dir=in_dir, list_path=list_path, out_base=out_base,
            threads=threads, max_radius=max_radius, step=step,
            extra_flags=extra_flags, total_images=total_images,
            estimated_passes=est_passes
        )
        # Command preview (precise)
        self._update_command_preview(self.current_run)
        self.lbl_pass.setText(f"Pass: 1/{max(1, self.current_run.estimated_passes)}")


        # Command preview
        append_line(self.txt_output,
                    f"[grid] base='{out_base}', R={max_radius:g}px step={step:g}px → {est_passes} passes")
        append_line(self.txt_output, f"[files] run_dir={run_dir}")
        append_line(self.txt_output, f"[files] input.lst={list_path} (images={total_images})")
        append_line(self.txt_output, f"[files] streams={run_dir/'streams'}")


        # Launch child
        prev = self._build_command_preview(self.current_run)                                        
        append_line(self.txt_output, "[preview] " + prev.replace("\n", " | "))   
        self._launch_runner(self.current_run)

    def _stop_clicked(self):
        if not self.proc or self.proc.poll() is not None:
            self.lbl_status.setText("No active process.")
            return
        ans = QMessageBox.question(self, "Stop run", "Stop the current job?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return
        self._kill_process_group()
        self._cleanup_runner_file()
        self._set_window_title_for_run(self._resolve_active_run_dir())

        self.btn_stop.setEnabled(False)
        self.btn_start_grid.setEnabled(True)
        self.lbl_status.setText("Stopped.")
        self.lbl_pass.setText("Pass: -/-")
        self.lbl_elapsed.setText("Elapsed: 00:00:00")
        self.lbl_eta.setText("ETA: --:--:--")
        self.lbl_rate.setText("Rate: -- img/s")
        self._refresh_streams_list()

    # ==================
    # Compose extra flags
    # ==================
    
    def _compose_extra_flags(self) -> list[str]:
        flags: list[str] = []

        # Optional indexamajig path passthrough (if you added this field)
        idxmj = getattr(self, "edit_idxmj", None)
        if idxmj and idxmj.text().strip():
            flags.append(f"--indexamajig-path={idxmj.text().strip()}")

        # Peakfinder params: take exactly what’s in the text box
        if hasattr(self, "peak_params_edit"):
            for line in self.peak_params_edit.toPlainText().splitlines():
                s = line.strip()
                if s:
                    flags.append(s)

        # Advanced flags (multiline)
        if hasattr(self, "txt_advanced"):
            for line in self.txt_advanced.toPlainText().splitlines():
                s = line.strip()
                if s:
                    flags.append(s)

        # Other flags (multiline)
        if hasattr(self, "txt_other"):
            for line in self.txt_other.toPlainText().splitlines():
                s = line.strip()
                if s:
                    flags.append(s)

        return flags

    # ======================
    # Create list and count
    # ======================

    def _write_input_list(self, in_dir: Path, list_path: Path) -> int:
        """
        Writes an indexamajig-style input list with absolute HDF5 paths.
        Returns total image count for progress scaling.
        """
        lines: List[str] = []
        for ext in ("*.h5", "*.hdf5", "*.cxi"):
            for h5path in sorted(in_dir.glob(ext)):
                lines.append(str(h5path.resolve()))
        write_text(list_path, "\n".join(lines) + ("\n" if lines else ""))
        total = count_images_in_h5_folder(in_dir)
        append_line(self.txt_output, f"[scan] detected {total} images across {len(lines)} files")
        return total


    # ==============
    # Runner launch
    # ==============
        
    def _launch_runner(self, ctx: RunContext):
        # Write a temp runner
        fd, runner_path = tempfile.mkstemp(prefix="run_gandalf_", suffix=".py")
        os.close(fd)
        self.proc_runner_path = Path(runner_path)
        with open(self.proc_runner_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(GANDALF_RUNNER_CODE))

        # Ensure streams dir
        streams_dir = ctx.run_dir / "streams"
        ensure_dir(streams_dir)

        # Build argv (named) + pass-through flags
        outbase_for_child = str((streams_dir / ctx.out_base).resolve())
        argv = [
            sys.executable,
            "-u",
            str(self.proc_runner_path),
            "--host", os.path.abspath(__file__),
            "--geom", str(ctx.geom_path),
            "--cell", str(ctx.cell_path),
            "--input", str(ctx.list_path),
            "--outbase", outbase_for_child,
            "--threads", str(ctx.threads),
            "--radius", str(ctx.max_radius),
            "--step", str(ctx.step),
            *ctx.extra_flags,
        ]

        # Log plan
        append_line(self.txt_output, "[launch] " + " ".join(shlex.quote(a) for a in argv))

        # Create per-run log
        self.run_log_path = ctx.run_dir / "indexing.log"
        self.run_log_fp = open(self.run_log_path, "w", encoding="utf-8")

        # Platform-specific spawn flags
        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if os.name == "nt":
            # Create a new process group for safe CTRL_BREAK
            try:
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            popen_kwargs["preexec_fn"] = os.setsid  # new process group (POSIX)

        # --- begin env patch for indexamajig path ---
        env = os.environ.copy()
        idxmj = self.edit_idxmj.text().strip()
        if idxmj:
            idx_path = Path(idxmj)
            idx_dir = str(idx_path.parent) if idx_path.suffix else str(Path(idxmj))
            # Prepend the directory to PATH so the child can find indexamajig
            env["PATH"] = idx_dir + os.pathsep + env.get("PATH", "")
        popen_kwargs["env"] = env
        # --- end env patch ---

        # Spawn
        try:
            self.proc = subprocess.Popen(argv, **popen_kwargs)
        except Exception as e:
            QMessageBox.critical(self, "Launch error", str(e))
            self._cleanup_runner_file()
            return

        # Reader thread
        self.proc_thread = ProcessOutputThread(self.proc)
        self.proc_thread.output_received.connect(self._on_proc_line)
        self.proc_thread.finished.connect(self._on_proc_finished)
        self.proc_thread.start()

        # UI state
        self.btn_start_grid.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Disable workspace/run actions while running (prevent tree churn)
        for b in (getattr(self, "btn_ws_new_run", None),
                getattr(self, "btn_ws_new_batch", None),
                getattr(self, "btn_ws_start_batch", None),
                getattr(self, "btn_ws_broadcast", None),
                getattr(self, "btn_new_run", None),
                getattr(self, "btn_open_run", None),
                getattr(self, "btn_batch_add", None),
                getattr(self, "btn_batch_clear", None),
                getattr(self, "btn_batch_start", None)):
            if b: b.setEnabled(False)

        self.lbl_status.setText("Running…")
        self._reset_progress_bar()
        self._refresh_streams_list()


        # Timing start
        self.run_start_time = time.time()
        self.ema_rate = None
        self.lbl_elapsed.setText("Elapsed: 00:00:00")
        self.lbl_eta.setText("ETA: --:--:--")
        self.lbl_rate.setText("Rate: -- img/s")


    def _reset_progress_bar(self):
        # Scale 0..(passes*per_pass)
        total = max(1, self.estimated_passes * max(1, self.per_pass_images))
        self.progress.setMinimum(0)
        self.progress.setMaximum(total)
        self.progress.setValue(0)

    # =====================
    # Process output hooks
    # =====================

    def _on_proc_line(self, text: str):
        line = text.rstrip("\n")

        # --- Selective mirroring to GUI log ---
        should_show = True
        if not getattr(self, "_mirror_stdout", True):
            should_show = False
        elif getattr(self, "_mirror_only_progress", True):
            low = line.lower()
            # Show progress lines AND anything that looks like an error/failure
            should_show = (
                ("images processed" in low)
                or ("error" in low)
                or ("failed" in low)
                or ("fatal" in low)
                or low.startswith("[xy-pass]")
            )

        if should_show:
            append_line(self.txt_output, line)

        # --- Always write complete raw output to log file ---
        try:
            self.run_log_fp.write(text)
            self.run_log_fp.flush()
        except Exception:
            pass

        # --- Progress parsing ---
        n = self._parse_images_processed(text)
        if n is not None:
            if n < self.last_seen_processed:
                self.seen_passes += 1
            self.last_seen_processed = n
            total_so_far = self.seen_passes * self.per_pass_images + n
            self.progress.setValue(max(self.progress.value(), total_so_far))

        # --- Pass label updates ---
        if text.startswith("[XY-PASS]"):
            parts = text.split()
            try:
                frac = parts[1]  # "i/N"
                i, N = frac.split("/")
                self.seen_passes = max(self.seen_passes, int(i) - 1)
                self.estimated_passes = max(self.estimated_passes, int(N))
                # Expand max without dropping current progress
                old_val = self.progress.value()
                new_max = max(1, self.estimated_passes * max(1, self.per_pass_images))
                self.progress.setMaximum(new_max)
                self.progress.setValue(min(old_val, new_max))
            except Exception:
                pass

        self.lbl_pass.setText(
            f"Pass: {min(self.seen_passes + 1, max(1, self.estimated_passes))}/{max(1, self.estimated_passes)}"
        )

        # --- Rate / ETA ---
        now = time.time()
        if self.run_start_time:
            rate_in_line = self._parse_images_per_sec(text)
            cur = self.progress.value()
            if rate_in_line and rate_in_line > 0:
                self.ema_rate = (
                    self.ema_alpha * rate_in_line +
                    (1 - self.ema_alpha) * (self.ema_rate or rate_in_line)
                )
            else:
                elapsed = max(1e-3, now - self.run_start_time)
                derived = cur / elapsed
                if derived > 0:
                    self.ema_rate = (
                        self.ema_alpha * derived +
                        (1 - self.ema_alpha) * (self.ema_rate or derived)
                    )
            self._update_time_labels(now, cur)


    def _on_toggle_mirror(self, checked: bool):
        self._mirror_stdout = bool(checked)
        try:
            self.mirror_progress_only_chk.setEnabled(bool(checked))
        except Exception:
            pass

    def _on_toggle_mirror_only_progress(self, checked: bool):
        self._mirror_only_progress = bool(checked)


    def _parse_images_processed(self, line: str) -> Optional[int]:
        if "images processed" not in line:
            return None
        # scan tokens backward until we find an int
        num = None
        toks = line.replace(",", " ").split()
        for i, tok in enumerate(toks):
            if tok.isdigit() and i + 1 < len(toks) and toks[i+1].startswith("images"):
                try:
                    num = int(tok); break
                except Exception:
                    continue
        if num is None:
            # fallback: first integer in the line
            acc = ""
            for ch in line:
                if ch.isdigit(): acc += ch
                elif acc:
                    try: return int(acc)
                    except: return None
            return int(acc) if acc else None
        return num


    def _parse_images_per_sec(self, line: str) -> Optional[float]:
        # light-weight scan for floating number followed by "images/sec"
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

    def _set_window_title_for_run(self, run_dir: Optional[Path]):
        base = "Index & Integrate (SerialED Grid)"
        if run_dir:
            self.setWindowTitle(f"{base} — {run_dir.name}")
        else:
            self.setWindowTitle(base)

    def _on_proc_finished(self, rc: int):
        append_line(self.txt_output, f"[done] exit code {rc}")
        self.lbl_status.setText("Finished." if rc == 0 else f"Failed (rc={rc})")
        self.btn_stop.setEnabled(False)
        self.btn_start_grid.setEnabled(True)
        for b in (getattr(self, "btn_ws_new_run", None),
                getattr(self, "btn_ws_new_batch", None),
                getattr(self, "btn_ws_start_batch", None),
                getattr(self, "btn_ws_broadcast", None),
                getattr(self, "btn_new_run", None),
                getattr(self, "btn_open_run", None),
                getattr(self, "btn_batch_add", None),
                getattr(self, "btn_batch_clear", None),
                getattr(self, "btn_batch_start", None)):
            if b: b.setEnabled(True)

        try:
            self.run_log_fp.close()
        except Exception:
            pass
        self._cleanup_runner_file()
        self.proc = None
        self.proc_thread = None
        self._set_window_title_for_run(self._resolve_active_run_dir())

        # Reset time labels
        self.lbl_elapsed.setText("Elapsed: 00:00:00")
        self.lbl_eta.setText("ETA: --:--:--")
        self.lbl_rate.setText("Rate: -- img/s")
        self.lbl_pass.setText("Pass: -/-")
        self.run_start_time = None
        self.ema_rate = None
        self._refresh_streams_list()

        # Batch continuation
        if self.batch_active:
            QTimer.singleShot(150, self._batch_next)
        if getattr(self, "_ws_chain_on_finish", False):
            self._ws_chain_on_finish = False
            if getattr(self, "_ws_batch_mode", False):
                QTimer.singleShot(200, self._ws_start_next_in_batch)


    # =========
    # Cleanup
    # =========

    def _kill_process_group(self):
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                # Send CTRL_BREAK to the group, then terminate if needed
                try:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    self.proc.wait(timeout=2)
                except Exception:
                    pass
                if self.proc.poll() is None:
                    self.proc.terminate()
            else:
                # POSIX: try graceful group kill, then hard-kill
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


    def _prefs_path(self) -> Path:
        return Path.home() / ".indexintegrate_iterate_window.json"

    def _load_prefs(self):
        try:
            p = self._prefs_path()
            if not p.exists():
                return
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            ws = data.get("workspace_root")
            rr = data.get("run_root")
            if ws and Path(ws).exists():
                self.workspace_root = Path(ws)
                self.edit_ws_root.setText(str(self.workspace_root))
            if rr and Path(rr).exists():
                self.run_root = Path(rr)
                self.edit_run_root.setText(str(self.run_root))
        except Exception:
            pass

    def _save_prefs(self):
        try:
            data = {
                "workspace_root": str(getattr(self, "workspace_root", "")),
                "run_root": str(getattr(self, "run_root", "")),
            }
            with open(self._prefs_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _cleanup_runner_file(self):
        try:
            if self.proc_runner_path and self.proc_runner_path.exists():
                self.proc_runner_path.unlink(missing_ok=True)
        except Exception:
            pass
        self.proc_runner_path = None

    def closeEvent(self, event):
        try:
            self._save_prefs()
        finally:
            super().closeEvent(event)


# =========
# __main__
# =========

def main():
    os.environ["QT_QPA_PLATFORM"] = "xcb" #LINUX
    # os.environ["QT_QPA_PLATFORM"] = "cocoa" #MAC
    app = QApplication(sys.argv)
    w = SerialEDIndexIntegrateWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
