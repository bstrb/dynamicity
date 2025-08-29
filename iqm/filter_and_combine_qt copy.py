#!/usr/bin/env python3
"""
Interactive Metrics Analysis Tool — PyQt6

Replicates the Tk UI:
  1) Load CSV with normalized metrics (grouped by event)
  2) Separate-metric thresholds (sliders) + histograms
  3) Combined metric (weights + threshold) + best-per-event + CSV
  4) Convert filtered CSV to .stream

Requires:
  pip install PyQt6 matplotlib
"""

from __future__ import annotations

import os
import sys
import json, math, time
from typing import Dict, List, Tuple

# --- third-party ---
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog, QGridLayout,
    QSlider, QDoubleSpinBox, QMessageBox, QCheckBox
)

import matplotlib.pyplot as plt

# --- your modules (same as in Tk code) ---
from filter_and_combine.csv_to_stream import write_stream_from_filtered_csv
from filter_and_combine.interactive_iqm import (
    read_metric_csv,
    select_best_results_by_event,
    get_metric_ranges,
    filter_rows,
    write_filtered_csv,
    filter_and_combine,   # helper that can (optionally) prefilter + compute combined metric
)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
metrics_in_order: List[str] = [
    "weighted_rmsd",
    "fraction_outliers",
    "length_deviation",
    "angle_deviation",
    "peak_ratio",
    "percentage_unindexed",
]

SLIDER_STEPS = 1000  # resolution for float sliders


def slider_to_float(v: int, mn: float, mx: float) -> float:
    if mx <= mn:
        return mn
    return mn + (v / SLIDER_STEPS) * (mx - mn)


def float_to_slider(x: float, mn: float, mx: float) -> int:
    if mx <= mn:
        return 0
    t = (x - mn) / (mx - mn)
    return max(0, min(SLIDER_STEPS, int(round(t * SLIDER_STEPS))))


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------
class MetricsQtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Metrics Analysis Tool (Qt6)")

        # state
        self.csv_path: str | None = None
        self.filtered_csv_path: str | None = None
        self.all_rows: List[dict] | None = None

        # built after CSV load
        self.metric_sliders: Dict[str, Tuple[QSlider, QLabel, float, float]] = {}
        self.weight_boxes: Dict[str, QDoubleSpinBox] = {}
        self.combined_thr_slider: QSlider | None = None
        self.combined_thr_lbl: QLabel | None = None
        self.combined_range: Tuple[float, float] | None = None

        # --- scrollable central widget ---
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._central = QWidget()
        self._scroll.setWidget(self._central)
        self.setCentralWidget(self._scroll)

        self.root_layout = QVBoxLayout(self._central)

        self._build_header()
        self._build_file_section()
        self._build_dynamic_container()

    # ---------------- UI sections ----------------
    def _build_header(self):
        text = (
            "Load a CSV file containing normalized metrics. Two analysis modes:\n\n"
            "1) Separate Metrics Filtering:\n"
            "   Set per-metric thresholds (≤). View histograms of passing rows.\n\n"
            "2) Combined Metric:\n"
            "   Set weights for each metric, create a combined metric, then filter by a threshold,\n"
            "   pick best row per event, write CSV, and optionally convert to .stream.\n"
        )
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        self.root_layout.addWidget(lbl)

    def _build_file_section(self):
        grp = QGroupBox("Select CSV with Normalized Metrics")
        g = QGridLayout(grp)

        g.addWidget(QLabel("CSV File:"), 0, 0)
        self.csv_edit = QLineEdit()
        g.addWidget(self.csv_edit, 0, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse)
        g.addWidget(browse_btn, 0, 2)

        load_btn = QPushButton("Load CSV")
        load_btn.setStyleSheet("background: lightblue;")
        load_btn.clicked.connect(self._on_load_csv)
        g.addWidget(load_btn, 1, 0, 1, 3)

        self.root_layout.addWidget(grp)

    def _build_dynamic_container(self):
        # container populated after CSV load
        self.dynamic_wrap = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_wrap)
        self.root_layout.addWidget(self.dynamic_wrap)

    # --------------- callbacks -------------------
    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file with normalized metrics",
            os.getcwd(), "CSV Files (*.csv)"
        )
        if path:
            self.csv_edit.setText(path)

    def _on_load_csv(self):
        path = self.csv_edit.text().strip()
        if not path:
            QMessageBox.critical(self, "Error", "Please select a CSV file.")
            return
        self.csv_path = path
        print(f"Loading CSV: {path}")

        try:
            grouped = read_metric_csv(path, group_by_event=True)
            all_rows: List[dict] = []
            for rows in grouped.values():
                all_rows.extend(rows)
            self.all_rows = all_rows
            print(f"Loaded {len(all_rows)} rows.")

            folder = os.path.dirname(path)
            self.filtered_csv_path = os.path.join(folder, "filtered_metrics.csv")

            self._build_analysis_ui(all_rows)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to read CSV:\n{e}")

    # --------------- build analysis UI ------------
    def _build_analysis_ui(self, all_rows: List[dict]):
        # clear previous
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # SECTION 1: Separate metrics filtering
        self._build_separate_section(all_rows)

        # SECTION 2: Combined metric
        self._build_combined_section(all_rows)

    def _build_separate_section(self, all_rows: List[dict]):
        grp = QGroupBox("Separate Metrics Filtering")
        grid = QGridLayout(grp)

        ranges_dict = get_metric_ranges(all_rows, metrics_in_order)
        self.metric_sliders.clear()

        for row, metric in enumerate(metrics_in_order):
            mn, mx = ranges_dict[metric]
            # default threshold at max (include all)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(SLIDER_STEPS)
            slider.setValue(SLIDER_STEPS)

            val_lbl = QLabel(f"{mx:.4f}")
            grid.addWidget(QLabel(f"{metric} ≤"), row, 0, alignment=Qt.AlignmentFlag.AlignRight)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val_lbl, row, 2)

            def make_updater(_metric=metric, _mn=mn, _mx=mx, _lbl=val_lbl):
                return lambda v: _lbl.setText(f"{slider_to_float(v, _mn, _mx):.4f}")

            slider.valueChanged.connect(make_updater())
            self.metric_sliders[metric] = (slider, val_lbl, mn, mx)

        apply_btn = QPushButton("Apply Separate Metrics Thresholds")
        apply_btn.setStyleSheet("background: lightblue;")
        apply_btn.clicked.connect(self._on_apply_separate)
        grid.addWidget(apply_btn, len(metrics_in_order), 0, 1, 3)

        self.dynamic_layout.addWidget(grp)

    def _build_combined_section(self, all_rows: List[dict]):
        grp = QGroupBox("Combined Metric Creation & Filtering")
        grid = QGridLayout(grp)

        # weights
        self.weight_boxes.clear()
        for i, metric in enumerate(metrics_in_order):
            grid.addWidget(QLabel(f"{metric} Weight:"), i, 0, alignment=Qt.AlignmentFlag.AlignRight)
            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setRange(-1e6, 1e6)
            sb.setSingleStep(0.1)
            sb.setValue(0.0)
            grid.addWidget(sb, i, 1)
            self.weight_boxes[metric] = sb

        # combined threshold slider
        row = len(metrics_in_order)
        grid.addWidget(QLabel("Combined Metric Threshold ≤"), row, 0, alignment=Qt.AlignmentFlag.AlignRight)
        self.combined_thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.combined_thr_slider.setMinimum(0)
        self.combined_thr_slider.setMaximum(SLIDER_STEPS)
        self.combined_thr_slider.setValue(0)
        self.combined_thr_lbl = QLabel("0.0000")
        grid.addWidget(self.combined_thr_slider, row, 1)
        grid.addWidget(self.combined_thr_lbl, row, 2)

        # these will be updated once we compute the combined metric
        self.combined_range = (0.0, 1.0)
        self.combined_thr_slider.valueChanged.connect(self._update_combined_lbl)

        # buttons
        create_btn = QPushButton("Create Combined Metric")
        create_btn.setStyleSheet("background: lightblue;")
        create_btn.clicked.connect(self._on_create_combined_metric)
        grid.addWidget(create_btn, row + 1, 0, 1, 3)

        apply_btn = QPushButton("Apply Combined Metric Threshold (Best Rows)")
        apply_btn.setStyleSheet("background: lightblue;")
        apply_btn.clicked.connect(self._on_apply_combined)
        grid.addWidget(apply_btn, row + 2, 0, 1, 3)

        convert_btn = QPushButton("Convert to Stream")
        convert_btn.setStyleSheet("background: green; color: white;")
        convert_btn.clicked.connect(self._on_convert_to_stream)
        grid.addWidget(convert_btn, row + 3, 0, 1, 3)

        self.dynamic_layout.addWidget(grp)

    # ------------------ actions -------------------
    def _gather_separate_thresholds(self) -> Dict[str, float]:
        thr: Dict[str, float] = {}
        for metric, (slider, _lbl, mn, mx) in self.metric_sliders.items():
            thr[metric] = slider_to_float(slider.value(), mn, mx)
        return thr

    def _gather_weights(self) -> List[float]:
        return [self.weight_boxes[m].value() for m in metrics_in_order]

    # Section 1
    def _on_apply_separate(self):
        if not self.all_rows:
            return
        plt.close('all')

        thresholds = self._gather_separate_thresholds()
        filtered = filter_rows(self.all_rows, thresholds)
        print(f"Separate filtering: {len(self.all_rows)} → {len(filtered)} rows")

        if not filtered:
            QMessageBox.information(self, "No Rows", "No rows passed the thresholds.")
            return

        # histograms
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metrics_in_order):
            vals = [r[metric] for r in filtered if metric in r]
            axes[i].hist(vals, bins=20)
            axes[i].set_title(f"Histogram of {metric}")
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Section 2: create combined metric (no prefilter here)
    def _on_create_combined_metric(self):
        if not self.all_rows:
            return

        weights = self._gather_weights()
        rows_with = filter_and_combine(
            rows=self.all_rows,
            pre_filter=None,
            metrics_to_combine=metrics_in_order,
            weights=weights,
            new_metric_name="combined_metric",
        )
        vals = [r["combined_metric"] for r in rows_with] if rows_with else []
        if not vals:
            QMessageBox.warning(self, "Combined Metric", "Failed to create combined metric (check weights).")
            return

        cmin, cmax = min(vals), max(vals)
        self.combined_range = (cmin, cmax)
        # default slider at max (include everything)
        self.combined_thr_slider.blockSignals(True)
        self.combined_thr_slider.setValue(float_to_slider(cmax, cmin, cmax))
        self.combined_thr_slider.blockSignals(False)
        self._update_combined_lbl(self.combined_thr_slider.value())

        print(f"Combined metric created. Range: [{cmin:.4f}, {cmax:.4f}]")
        QMessageBox.information(self, "Combined Metric", f"Created. Range: [{cmin:.3f}, {cmax:.3f}].")

    def _update_combined_lbl(self, v: int):
        mn, mx = self.combined_range or (0.0, 1.0)
        if self.combined_thr_lbl:
            self.combined_thr_lbl.setText(f"{slider_to_float(v, mn, mx):.4f}")

    # Section 2: apply combined (with prefilter)
    def _on_apply_combined(self):
        if not self.all_rows or not self.filtered_csv_path:
            return
        plt.close('all')

        pre_thresholds = self._gather_separate_thresholds()
        weights = self._gather_weights()
        rows = filter_and_combine(
            rows=self.all_rows,
            pre_filter=pre_thresholds,
            metrics_to_combine=metrics_in_order,
            weights=weights,
            new_metric_name="combined_metric",
        )
        print(f"{len(self.all_rows)} total → {len(rows)} survive separate thresholds")

        if not rows:
            QMessageBox.information(self, "No Rows", "No rows after separate-metric prefilter.")
            return

        mn, mx = self.combined_range or (0.0, 1.0)
        thr = slider_to_float(self.combined_thr_slider.value(), mn, mx)
        by_combined = [r for r in rows if r["combined_metric"] <= thr]
        print(f"{len(by_combined)} remain after combined_metric ≤ {thr:.4f}")

        if not by_combined:
            QMessageBox.information(self, "No Rows", "No rows passed the combined threshold.")
            return

        # best-per-event → CSV + histogram
        grouped: Dict[int, List[dict]] = {}
        for r in by_combined:
            grouped.setdefault(r["event_number"], []).append(r)

        best = select_best_results_by_event(grouped, sort_metric="combined_metric")
        write_filtered_csv(best, self.filtered_csv_path)
        print(f"Wrote CSV → {self.filtered_csv_path}")

        plt.figure(figsize=(8, 6))
        plt.hist([r["combined_metric"] for r in best], bins=20)
        plt.title("Histogram of Best Rows (combined_metric)")
        plt.xlabel("combined_metric")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        QMessageBox.information(self, "Saved", f"Best-per-event CSV written:\n{self.filtered_csv_path}")

    # Convert filtered CSV to .stream
    def _on_convert_to_stream(self):
        if not (self.csv_path and self.filtered_csv_path):
            QMessageBox.critical(self, "Missing", "Load a CSV and produce a filtered CSV first.")
            return

        out_dir = os.path.join(os.path.dirname(self.csv_path), "filtered_metrics")
        os.makedirs(out_dir, exist_ok=True)
        out_stream = os.path.join(out_dir, "filtered_metrics.stream")

        print("Reading filtered CSV…")
        grouped = read_metric_csv(self.filtered_csv_path, group_by_event=True)
        first_event_rows = next(iter(grouped.values()))
        if "combined_metric" in first_event_rows[0]:
            print("combined_metric found → selecting best rows")
            best = select_best_results_by_event(grouped, sort_metric="combined_metric")
            write_filtered_csv(best, self.filtered_csv_path)
            print("CSV overwritten with best rows.")

        print("Writing .stream…")
        write_stream_from_filtered_csv(
            filtered_csv_path=self.filtered_csv_path,
            output_stream_path=out_stream,
            event_col="event_number",
            streamfile_col="stream_file",
        )
        print(f"Done → {out_stream}")
        QMessageBox.information(self, "Stream", f"Saved stream file:\n{out_stream}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = MetricsQtWindow()
    win.resize(900, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
