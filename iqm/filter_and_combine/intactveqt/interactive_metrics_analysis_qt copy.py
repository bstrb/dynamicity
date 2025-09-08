#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interactive_metrics_analysis_qt.py — Two-stage per-chunk → global selection

Flow:
  1) Load raw CSV (your new comment-chunk format supported by interactive_iqm.py).
  2) Section 1: apply raw-unit thresholds per metric (≤ sliders; direction-aware
     thresholds are handled in the backend if you choose to wire that later).
  3) Section 2a: Per-chunk normalization (robust_z / zscore / minmax) + weights
     → build a direction-aware combined BADNESS score (lower = better).
  4) Section 2b: Select BEST row per event (min combined).
     - Histogram of best-per-event combined scores.
  5) Section 3: Global normalization of the best-per-event combined score
     (robust_z / zscore / minmax). Apply a global threshold and write filtered CSV.
  6) Convert filtered CSV → .stream.

Dependencies:
  - PyQt6
  - matplotlib (QtAgg backend)
  - interactive_iqm.py (with the new helpers)
  - csv_to_stream.py (write_stream_from_filtered_csv)
"""

from __future__ import annotations

import os
import sys
from functools import partial
from typing import List, Dict

from PyQt6 import QtCore, QtWidgets  # noqa: N812

# Matplotlib backend must be set before pyplot import
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------- Domain helpers (from your modules) ----------------
from csv_to_stream import write_stream_from_filtered_csv
from interactive_iqm import (
    read_metric_csv,
    get_metric_ranges,
    filter_rows,
    write_filtered_csv,
    # new helpers for the two-stage flow:
    normalize_metrics_per_chunk,
    combine_per_chunk_and_select_best,
    global_normalize_metric,
    filter_by_global_metric,
)

metrics_in_order: List[str] = [
    "weighted_rmsd",
    "fraction_outliers",
    "length_deviation",
    "angle_deviation",
    "peak_ratio",
    "percentage_unindexed",
]


class MetricsWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Interactive Metrics Analysis (per-chunk → global)")
        self.resize(1000, 760)

        # central scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
        self._root = QtWidgets.QVBoxLayout(container)

        # State
        self._csv_file: str | None = None
        self._filtered_csv_path: str | None = None
        self._grouped: Dict[str, List[dict]] | None = None
        self._all_rows: List[dict] | None = None
        self._grouped_norm: Dict[str, List[dict]] | None = None
        self._best_rows: List[dict] | None = None
        self._best_rows_global_norm: List[dict] | None = None

        self._build_csv_selector()
        self._analysis: QtWidgets.QWidget | None = None

    # -------------------- 0) CSV selector --------------------
    def _build_csv_selector(self) -> None:
        box = QtWidgets.QGroupBox("1) Select CSV with raw metrics")
        g = QtWidgets.QGridLayout(box)

        self._csv_edit = QtWidgets.QLineEdit()
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_csv)

        btn_load = QtWidgets.QPushButton("Load CSV")
        btn_load.setStyleSheet("background-color: lightblue;")
        btn_load.clicked.connect(self._load_csv)

        g.addWidget(QtWidgets.QLabel("CSV file:"), 0, 0)
        g.addWidget(self._csv_edit, 0, 1)
        g.addWidget(btn_browse, 0, 2)
        g.addWidget(btn_load, 1, 0, 1, 3)

        self._root.addWidget(box)

        help_lab = QtWidgets.QLabel(
            "This tool supports your new CSV format with comment lines:\n"
            "# Image filename: … / # Event: //… followed by rows per event.\n\n"
            "Workflow:\n"
            "  • Section 1: filter by raw metrics (≤ thresholds).\n"
            "  • Section 2: per-chunk normalization + weights → select best per event.\n"
            "  • Section 3: global normalization of the best rows → final filter & save.\n"
        )
        help_lab.setWordWrap(True)
        self._root.addWidget(help_lab)

    def _browse_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select metrics CSV", os.getcwd(), "CSV files (*.csv)"
        )
        if path:
            self._csv_edit.setText(path)

    def _load_csv(self) -> None:
        path = self._csv_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.critical(self, "Error", "Please select a CSV file.")
            return

        try:
            grouped = read_metric_csv(path, group_by_event=True)
            # flatten
            all_rows: List[dict] = []
            for rows in grouped.values():
                all_rows.extend(rows)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._csv_file = path
        self._grouped = grouped
        self._all_rows = all_rows
        self._filtered_csv_path = os.path.join(os.path.dirname(path), "filtered_metrics.csv")

        if self._analysis is not None:
            self._analysis.setParent(None)
        self._analysis = self._build_analysis_ui()
        self._root.addWidget(self._analysis)

    # -------------------- 2) Analysis UI --------------------
    def _build_analysis_ui(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # ------- Section 1: separate metric sliders in raw units -------
        box1 = QtWidgets.QGroupBox("2) Separate metric thresholds (raw units, ≤)")
        g1 = QtWidgets.QGridLayout(box1)

        ranges = get_metric_ranges(self._all_rows, metrics_in_order)
        self._sep_sliders: dict[str, QtWidgets.QSlider] = {}
        self._sep_labels: dict[str, QtWidgets.QLabel] = {}

        for i, m in enumerate(metrics_in_order):
            mn, mx = ranges[m]
            lab = QtWidgets.QLabel(f"{m} ≤")
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setMinimum(0)
            s.setMaximum(1000)
            s.setValue(1000)  # default: wide-open (≤ max)
            s.setFixedWidth(360)
            val_lab = QtWidgets.QLabel(f"{mx:.3f}")
            s.valueChanged.connect(partial(self._update_slider_label, s, val_lab, mn, mx))
            self._sep_sliders[m] = s
            self._sep_labels[m] = val_lab
            g1.addWidget(lab, i, 0)
            g1.addWidget(s, i, 1)
            g1.addWidget(val_lab, i, 2)

        btn_preview_sep = QtWidgets.QPushButton("Preview histograms (after separate thresholds)")
        btn_preview_sep.clicked.connect(self._preview_separate_histograms)
        g1.addWidget(btn_preview_sep, len(metrics_in_order), 0, 1, 3)
        v.addWidget(box1)

        # ------- Section 2: per-chunk normalize + best per event -------
        box2 = QtWidgets.QGroupBox("3) Per-chunk normalization + combined weights → Best per event")
        g2 = QtWidgets.QGridLayout(box2)

        g2.addWidget(QtWidgets.QLabel("Per-chunk normalization method:"), 0, 0)
        self._chunk_norm_combo = QtWidgets.QComboBox()
        self._chunk_norm_combo.addItems(["robust_z", "zscore", "minmax"])
        g2.addWidget(self._chunk_norm_combo, 0, 1)

        self._weight_edits: dict[str, QtWidgets.QLineEdit] = {}
        for i, m in enumerate(metrics_in_order, start=1):
            g2.addWidget(QtWidgets.QLabel(f"{m} weight:"), i, 0)
            e = QtWidgets.QLineEdit("0.0")
            e.setFixedWidth(70)
            self._weight_edits[m] = e
            g2.addWidget(e, i, 1)

        btn_best = QtWidgets.QPushButton("Compute best per event")
        btn_best.setStyleSheet("background-color: lightblue;")
        btn_best.clicked.connect(self._compute_best_per_event)
        g2.addWidget(btn_best, len(metrics_in_order) + 1, 0, 1, 2)

        v.addWidget(box2)

        # ------- Section 3: global normalization & filter -------
        box3 = QtWidgets.QGroupBox("4) Global normalization of best rows → Filter & Save/Stream")
        g3 = QtWidgets.QGridLayout(box3)

        g3.addWidget(QtWidgets.QLabel("Global normalization method:"), 0, 0)
        self._global_norm_combo = QtWidgets.QComboBox()
        self._global_norm_combo.addItems(["robust_z", "zscore", "minmax"])
        g3.addWidget(self._global_norm_combo, 0, 1)

        g3.addWidget(QtWidgets.QLabel("Global threshold (≤ if z-type, ≤ fraction if minmax):"), 1, 0)
        self._global_thr_edit = QtWidgets.QLineEdit("2.0")  # e.g., z ≤ 2.0; for minmax think 0..1
        self._global_thr_edit.setFixedWidth(90)
        g3.addWidget(self._global_thr_edit, 1, 1)

        btn_global = QtWidgets.QPushButton("Global normalize & apply threshold")
        btn_global.setStyleSheet("background-color: lightblue;")
        btn_global.clicked.connect(self._global_normalize_and_filter)
        g3.addWidget(btn_global, 2, 0, 1, 2)

        btn_stream = QtWidgets.QPushButton("Convert filtered CSV → .stream")
        btn_stream.setStyleSheet("background-color: green; color: white;")
        btn_stream.clicked.connect(self._convert_to_stream)
        g3.addWidget(btn_stream, 3, 0, 1, 2)

        v.addWidget(box3)
        v.addStretch(1)
        return w

    # ---------------- helpers ----------------
    def _update_slider_label(self, slider: QtWidgets.QSlider, lab: QtWidgets.QLabel, mn: float, mx: float) -> None:
        val = mn + (mx - mn) * slider.value() / 1000.0
        lab.setText(f"{val:.3f}")

    def _current_separate_thresholds(self) -> Dict[str, float]:
        thr: Dict[str, float] = {}
        ranges = get_metric_ranges(self._all_rows, metrics_in_order)
        for m, sld in self._sep_sliders.items():
            mn, mx = ranges[m]
            thr[m] = mn + (mx - mn) * sld.value() / 1000.0
        return thr

    # ----- Section 1 preview -----
    def _preview_separate_histograms(self) -> None:
        if not self._all_rows:
            return
        thr = self._current_separate_thresholds()
        rows = filter_rows(self._all_rows, thr)
        if not rows:
            QtWidgets.QMessageBox.information(self, "No rows", "No rows pass the separate thresholds.")
            return
        plt.close("all")
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        for i, m in enumerate(metrics_in_order):
            vals = [r[m] for r in rows if r.get(m) is not None]
            axes[i].hist(vals, bins=30)
            axes[i].set_title(m)
        fig.suptitle(f"Filtered rows after separate thresholds (n={len(rows)})")
        plt.tight_layout()
        plt.show()

    # ----- Section 2: per-chunk normalize + best per event -----
    def _compute_best_per_event(self) -> None:
        if self._grouped is None or self._all_rows is None:
            return

        # 1) Apply separate thresholds in raw units
        thr = self._current_separate_thresholds()
        surviving = filter_rows(self._all_rows, thr)
        if not surviving:
            QtWidgets.QMessageBox.information(self, "No rows", "No rows after separate thresholds.")
            return

        # Re-group surviving rows by event_number
        grouped: Dict[str, List[dict]] = {}
        for r in surviving:
            ev = r.get("event_number", "")
            grouped.setdefault(ev, []).append(r)

        # 2) Per-chunk normalization + weights → combined badness + best per event
        method = self._chunk_norm_combo.currentText()
        weights = [float(self._weight_edits[m].text() or 0.0) for m in metrics_in_order]

        grouped_norm = normalize_metrics_per_chunk(grouped, metrics_in_order, method=method)
        best = combine_per_chunk_and_select_best(grouped_norm, metrics_in_order, weights)

        if not best:
            QtWidgets.QMessageBox.information(self, "No best rows", "No best rows could be selected (check weights).")
            return

        self._grouped_norm = grouped_norm
        self._best_rows = best

        # Preview histogram of best-per-event combined_metric
        vals = [r["combined_metric"] for r in best if r.get("combined_metric") is not None]
        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.hist(vals, bins=30)
        plt.title(f"Best-per-event combined_metric ({method}), n={len(vals)}")
        plt.xlabel("combined_metric (badness, lower=better)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        QtWidgets.QMessageBox.information(
            self, "Best rows ready",
            f"Selected best row per event: {len(best)} rows.\nProceed to global normalization."
        )

    # ----- Section 3: global normalize + filter -----
    def _global_normalize_and_filter(self) -> None:
        if not self._best_rows:
            QtWidgets.QMessageBox.warning(self, "Missing step", "Compute best per event first.")
            return
        method = self._global_norm_combo.currentText()
        try:
            thr_val = float(self._global_thr_edit.text())
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Invalid threshold", "Enter a numeric threshold.")
            return

        best_norm, stats = global_normalize_metric(self._best_rows, metric="combined_metric", method=method)
        self._best_rows_global_norm = best_norm

        # Plot normalized
        col = "combined_metric__global"
        vals = [r[col] for r in best_norm if r.get(col) is not None]
        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.hist(vals, bins=30)
        plt.title(f"Global-normalized combined_metric ({method})")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Filter by normalized threshold (≤ by default)
        kept = filter_by_global_metric(best_norm, metric_norm_name=col, threshold=thr_val, keep_low=True)
        if not kept:
            QtWidgets.QMessageBox.information(self, "No rows kept", "No rows pass the global threshold.")
            return

        # Write filtered CSV
        out_csv = self._filtered_csv_path or os.path.join(os.getcwd(), "filtered_metrics.csv")
        write_filtered_csv(kept, out_csv)
        self._filtered_csv_path = out_csv
        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Wrote filtered CSV:\n{out_csv}\n\nYou can now convert to .stream."
        )

    # ----- Convert to stream -----
    def _convert_to_stream(self) -> None:
        if not (self._csv_file and self._filtered_csv_path):
            QtWidgets.QMessageBox.warning(self, "Missing step", "Run global normalization & filter first.")
            return
        try:
            out_dir = os.path.join(os.path.dirname(self._csv_file), "filtered_metrics")
            os.makedirs(out_dir, exist_ok=True)
            out_stream = os.path.join(out_dir, "filtered_metrics.stream")

            # Convert using your helper
            write_stream_from_filtered_csv(
                filtered_csv_path=self._filtered_csv_path,
                output_stream_path=out_stream,
                event_col="event_number",
                streamfile_col="stream_file",
            )
            QtWidgets.QMessageBox.information(self, "Done", f"CSV converted to:\n{out_stream}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Conversion failed", str(exc))


# ----------------------------- entry point -----------------------------
def main() -> None:
    # Run off-screen if no display (e.g., SSH)
    if not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QtWidgets.QApplication(sys.argv)
    win = MetricsWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
