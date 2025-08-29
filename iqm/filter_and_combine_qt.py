#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Metrics Analysis Tool (Qt6)

Features:
- Load CSV with normalized metrics (grouped by event).
- Separate-metric thresholds (with enable/disable per metric).
- Debounced live pass counts per metric and overall.
- Percentile jump buttons (P50/P90/P95) per metric.
- Combined metric creation from weighted metrics, threshold & best-per-event.
- Convert filtered CSV to .stream.
- Export a reproducibility manifest (JSON) with inputs/outputs/params.

Requires:
  pip install PyQt6 matplotlib
  (and your project modules on PYTHONPATH)
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QPushButton, QLineEdit, QFileDialog,
    QMessageBox, QCheckBox, QSlider, QDoubleSpinBox, QSpinBox
)

# Matplotlib (for histograms)
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# Your modules
from filter_and_combine.interactive_iqm import (
    read_metric_csv,
    select_best_results_by_event,
    get_metric_ranges,
    filter_rows,
    write_filtered_csv,
    filter_and_combine,     # builds combined metric and optional prefilter
)
from filter_and_combine.csv_to_stream import write_stream_from_filtered_csv

APP_VERSION = "0.6.0"

# ----------------------------
# Helpers
# ----------------------------

def compute_percentiles(values: List[float], ps=(50, 90, 95)) -> Dict[int, float]:
    """Simple percentile computation without numpy."""
    out = {}
    if not values:
        return {p: 0.0 for p in ps}
    vals = sorted(values)
    n = len(vals)
    for p in ps:
        if n == 1:
            out[p] = vals[0]
            continue
        rank = (p / 100.0) * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        out[p] = vals[lo] * (1 - frac) + vals[hi] * frac
    return out


class FloatSlider(QWidget):
    """
    A float-aware slider: maps int range [0,1000] to [min,max].
    Exposes: value(), setValue(float), setRange(min,max), valueChanged(float)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._min = 0.0
        self._max = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider)

        self._slider.valueChanged.connect(self._on_int_changed)

        # External signal
        self.valueChanged = self._slider.valueChanged  # we’ll emit manually w/ float via proxy

    def setRange(self, mn: float, mx: float):
        if mx < mn:
            mn, mx = mx, mn
        self._min, self._max = float(mn), float(mx)
        # keep current position consistent
        self.setValue(self.value())

    def value(self) -> float:
        t = self._slider.value() / 1000.0
        return self._min + t * (self._max - self._min)

    def setValue(self, v: float):
        if self._max == self._min:
            self._slider.setValue(0)
            return
        t = (float(v) - self._min) / (self._max - self._min)
        t = max(0.0, min(1.0, t))
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(t * 1000)))
        self._slider.blockSignals(False)

    def setEnabled(self, b: bool) -> None:
        self._slider.setEnabled(b)

    def _on_int_changed(self, _):
        # keep valueChanged signature uniform; consumers will read .value()
        pass

# ----------------------------
# Main Window
# ----------------------------

class MetricsTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Metrics Analysis Tool (Qt6)")
        self.setMinimumSize(1000, 720)

        # Data / state
        self.csv_path: Optional[str] = None
        self.filtered_csv_path: Optional[str] = None
        self.stream_path: Optional[str] = None
        self.all_rows: List[dict] = []

        # Metrics
        self.metrics_in_order: List[str] = [
            "weighted_rmsd",
            "fraction_outliers",
            "length_deviation",
            "angle_deviation",
            "peak_ratio",
            "percentage_unindexed",
        ]

        # Built after CSV load
        # metric -> {enabled_cb, slider(FloatSlider), lbl_pass(QLabel), pbtns(dict), range(min,max), percentiles(dict)}
        self.metric_widgets: Dict[str, dict] = {}
        self.overall_pass_lbl: Optional[QLabel] = None

        # Weights UI per metric
        self.weight_boxes: Dict[str, QDoubleSpinBox] = {}

        # Combined metric slider/range/label
        self.combined_thr_slider: Optional[FloatSlider] = None
        self.combined_thr_lbl: Optional[QLabel] = None
        self.combined_range: Optional[Tuple[float, float]] = None

        # Debounce
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._recompute_live_counts)

        # Last counts for manifest
        self._last_separate_pass_count: Optional[int] = None
        self._last_combined_pass_count: Optional[int] = None

        # Scrollable central widget
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._central = QWidget()
        self._scroll.setWidget(self._central)
        self.setCentralWidget(self._scroll)

        self.root_layout = QVBoxLayout(self._central)
        self.root_layout.setContentsMargins(12, 12, 12, 12)
        self.root_layout.setSpacing(10)

        self._build_header()
        self._build_file_section()
        self._dynamic_container = QVBoxLayout()
        self.root_layout.addLayout(self._dynamic_container)

    # ---------------- UI builders ----------------

    def _build_header(self):
        box = QGroupBox("About")
        v = QVBoxLayout(box)
        lbl = QLabel(
            "Load a CSV with normalized IQM metrics grouped by event.\n"
            "Use per-metric thresholds (with percentiles), then optionally build a weighted combined "
            "metric and select best rows per event. Convert to .stream and export a manifest."
        )
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        self.root_layout.addWidget(box)

    def _build_file_section(self):
        box = QGroupBox("Input / Output")
        g = QGridLayout(box)

        # CSV selector
        g.addWidget(QLabel("CSV File:"), 0, 0)
        self.csv_edit = QLineEdit()
        g.addWidget(self.csv_edit, 0, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_csv)
        g.addWidget(btn_browse, 0, 2)

        self.btn_load = QPushButton("Load CSV")
        self.btn_load.setStyleSheet("font-weight: bold;")
        self.btn_load.clicked.connect(self._on_load_csv)
        g.addWidget(self.btn_load, 1, 0, 1, 3)

        # Output paths (shown after actions)
        self.filtered_lbl = QLabel("Filtered CSV: —")
        self.stream_lbl = QLabel("Stream: —")
        g.addWidget(self.filtered_lbl, 2, 0, 1, 3)
        g.addWidget(self.stream_lbl, 3, 0, 1, 3)

        self.root_layout.addWidget(box)

    def _clear_dynamic(self):
        # Remove any dynamic children
        while self._dynamic_container.count():
            item = self._dynamic_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.metric_widgets.clear()
        self.weight_boxes.clear()
        self.combined_thr_slider = None
        self.combined_thr_lbl = None
        self.combined_range = None
        self.overall_pass_lbl = None

    def _build_metrics_section(self):
        box = QGroupBox("Separate Metrics Filtering")
        v = QVBoxLayout(box)

        # Controls row
        ctrl = QHBoxLayout()
        self.overall_pass_lbl = QLabel("Live pass: —")
        ctrl.addWidget(self.overall_pass_lbl)
        ctrl.addStretch(1)

        btn_select_all = QPushButton("Enable All")
        btn_select_none = QPushButton("Disable All")
        btn_select_all.clicked.connect(lambda: self._set_all_enabled(True))
        btn_select_none.clicked.connect(lambda: self._set_all_enabled(False))
        ctrl.addWidget(btn_select_all)
        ctrl.addWidget(btn_select_none)

        v.addLayout(ctrl)

        # Metrics grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel("Use"), 0, 0)
        grid.addWidget(QLabel("Metric"), 0, 1)
        grid.addWidget(QLabel("Threshold (≤)"), 0, 2)
        grid.addWidget(QLabel("Percentiles"), 0, 3)
        grid.addWidget(QLabel("Pass (live)"), 0, 4)

        ranges = get_metric_ranges(self.all_rows, metrics=self.metrics_in_order)  # {m: (min,max)}
        for r, metric in enumerate(self.metrics_in_order, start=1):
            mn, mx = ranges[metric]
            vals = [row[metric] for row in self.all_rows if metric in row]
            percs = compute_percentiles(vals, ps=(50, 90, 95))

            cb = QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(lambda _, m=metric: self._on_metric_enabled_changed(m))

            name_lbl = QLabel(metric)

            slider = FloatSlider()
            slider.setRange(mn, mx)
            slider.setValue(mx)  # include all by default
            slider.valueChanged.connect(lambda _, m=metric: self._queue_recompute())

            # Percentile buttons
            pbox = QHBoxLayout()
            for tag, p in (("P50", 50), ("P90", 90), ("P95", 95)):
                btn = QPushButton(tag)
                btn.setFixedWidth(50)
                btn.setToolTip(f"Jump to {tag} = {percs[p]:.4g}")
                btn.clicked.connect(lambda _, pv=percs[p], s=slider: (s.setValue(pv), self._queue_recompute()))
                pbox.addWidget(btn)

            pass_lbl = QLabel("—")

            # store widgets
            self.metric_widgets[metric] = {
                "cb": cb,
                "slider": slider,
                "pass_lbl": pass_lbl,
                "min": mn,
                "max": mx,
                "percentiles": percs,
            }

            grid.addWidget(cb, r, 0)
            grid.addWidget(name_lbl, r, 1)
            grid.addLayout(self._h_pack(slider), r, 2)
            grid.addLayout(pbox, r, 3)
            grid.addWidget(pass_lbl, r, 4)

        v.addLayout(grid)

        # Action row for separate filtering & histogram
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply Separate Thresholds + Histograms")
        btn_apply.setStyleSheet("background:#dff0ff;")
        btn_apply.clicked.connect(self._on_apply_separate)
        row.addWidget(btn_apply)
        row.addStretch(1)
        v.addLayout(row)

        self._dynamic_container.addWidget(box)

    def _build_combined_section(self):
        box = QGroupBox("Combined Metric Creation & Filtering")
        v = QVBoxLayout(box)

        grid = QGridLayout()
        grid.addWidget(QLabel("Metric"), 0, 0)
        grid.addWidget(QLabel("Weight"), 0, 1)

        for r, metric in enumerate(self.metrics_in_order, start=1):
            grid.addWidget(QLabel(metric), r, 0)
            w = QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(-1e6, 1e6)
            w.setSingleStep(0.1)
            w.setValue(0.0)
            self.weight_boxes[metric] = w
            grid.addWidget(w, r, 1)

        v.addLayout(grid)

        # Buttons: Create combined metric; then Apply threshold & best-per-event
        btns = QHBoxLayout()
        btn_create = QPushButton("Create / Update Combined Metric")
        btn_create.setStyleSheet("background:#e6ffe6;")
        btn_create.clicked.connect(self._on_create_combined)
        btns.addWidget(btn_create)

        btn_apply_best = QPushButton("Apply Combined Threshold (Best per Event) → CSV")
        btn_apply_best.setStyleSheet("background:#e6f1ff;")
        btn_apply_best.clicked.connect(self._on_apply_combined_best)
        btns.addWidget(btn_apply_best)

        btns.addStretch(1)
        v.addLayout(btns)

        # Combined threshold slider + label
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Combined Threshold ≤"))
        self.combined_thr_slider = FloatSlider()
        self.combined_thr_lbl = QLabel("—")
        thr_row.addWidget(self.combined_thr_slider, 1)
        thr_row.addWidget(self.combined_thr_lbl)
        v.addLayout(thr_row)

        # Convert to stream + Export Manifest
        bottom = QHBoxLayout()
        btn_stream = QPushButton("Convert to Stream")
        btn_stream.setStyleSheet("background:#d7ffd7;")
        btn_stream.clicked.connect(self._on_convert_to_stream)
        bottom.addWidget(btn_stream)

        btn_manifest = QPushButton("Export Manifest (JSON)")
        btn_manifest.setStyleSheet("background:#fff2cc;")
        btn_manifest.clicked.connect(self._on_export_manifest)
        bottom.addWidget(btn_manifest)
        bottom.addStretch(1)

        v.addLayout(bottom)

        self._dynamic_container.addWidget(box)

    # ---------------- actions ----------------

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", os.getcwd(), "CSV files (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def _on_load_csv(self):
        path = self.csv_edit.text().strip()
        if not path:
            QMessageBox.critical(self, "Missing file", "Please choose a CSV file.")
            return
        if not os.path.isfile(path):
            QMessageBox.critical(self, "Not found", f"File not found:\n{path}")
            return

        try:
            grouped = read_metric_csv(path, group_by_event=True)
            rows: List[dict] = []
            for rs in grouped.values():
                rows.extend(rs)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to read CSV:\n{e}")
            return

        self.csv_path = path
        self.all_rows = rows
        self.filtered_csv_path = os.path.join(os.path.dirname(path), "filtered_metrics.csv")
        self.stream_path = None
        self.filtered_lbl.setText(f"Filtered CSV: {self.filtered_csv_path}")
        self.stream_lbl.setText("Stream: —")

        # Rebuild dynamic sections
        self._clear_dynamic()
        self._build_metrics_section()
        self._build_combined_section()
        self._recompute_live_counts()

        QMessageBox.information(self, "Loaded", f"Loaded {len(rows)} rows from\n{path}")

    def _set_all_enabled(self, enabled: bool):
        for meta in self.metric_widgets.values():
            meta["cb"].setChecked(enabled)
        self._queue_recompute()

    def _on_metric_enabled_changed(self, metric: str):
        m = self.metric_widgets[metric]
        enabled = m["cb"].isChecked()
        m["slider"].setEnabled(enabled)
        self._queue_recompute()

    def _gather_enabled_thresholds(self) -> Dict[str, float]:
        """Return {metric: threshold} for enabled metrics."""
        thr = {}
        for metric, meta in self.metric_widgets.items():
            if meta["cb"].isChecked():
                thr[metric] = meta["slider"].value()
        return thr

    def _queue_recompute(self):
        self._debounce.start()

    def _recompute_live_counts(self):
        if not self.all_rows:
            return

        total = len(self.all_rows)

        # Per-metric pass (this metric only)
        for metric, meta in self.metric_widgets.items():
            if not meta["cb"].isChecked():
                meta["pass_lbl"].setText("disabled")
                continue
            thr = meta["slider"].value()
            cnt = sum(1 for r in self.all_rows if metric in r and r[metric] <= thr)
            meta["pass_lbl"].setText(f"≤ {thr:.4g} → {cnt}/{total}")

        # Overall pass (all enabled thresholds)
        thresholds = self._gather_enabled_thresholds()
        if thresholds:
            passed = filter_rows(self.all_rows, thresholds)
            self._last_separate_pass_count = len(passed)
            self.overall_pass_lbl.setText(f"Overall pass (enabled metrics): {len(passed)}/{total}")
        else:
            self._last_separate_pass_count = None
            self.overall_pass_lbl.setText(f"Overall pass (enabled metrics): — / {total}")

    def _on_apply_separate(self):
        if not self.all_rows:
            return
        thresholds = self._gather_enabled_thresholds()
        if not thresholds:
            QMessageBox.information(self, "No metrics enabled", "Enable at least one metric to filter.")
            return

        filtered = filter_rows(self.all_rows, thresholds)
        n0, n1 = len(self.all_rows), len(filtered)

        # Histograms per metric from filtered rows
        if not filtered:
            QMessageBox.information(self, "No rows", "No rows passed current thresholds.")
        else:
            rows = filtered
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            axes = axes.flatten()
            for i, metric in enumerate(self.metrics_in_order):
                vals = [r[metric] for r in rows if metric in r]
                if vals:
                    axes[i].hist(vals, bins=20)
                axes[i].set_title(metric)
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel("Count")
            plt.tight_layout()
            plt.show()

        # Write filtered CSV (separate filtering only as preview)
        write_filtered_csv(filtered, self.filtered_csv_path)
        self.filtered_lbl.setText(f"Filtered CSV: {self.filtered_csv_path}")
        QMessageBox.information(self, "Separate filter",
                                f"{n0} → {n1} rows written to:\n{self.filtered_csv_path}")

    def _enabled_metrics_list(self) -> List[str]:
        return [m for m, meta in self.metric_widgets.items() if meta["cb"].isChecked()]

    def _on_create_combined(self):
        if not self.all_rows:
            return
        enabled_metrics = self._enabled_metrics_list()
        if not enabled_metrics:
            QMessageBox.information(self, "No metrics enabled", "Enable at least one metric to combine.")
            return

        weights_list: List[float] = [self.weight_boxes[m].value() for m in enabled_metrics]

        # Build combined metric on ALL rows (no prefilter here)
        rows_with_metric = filter_and_combine(
            rows=self.all_rows,
            pre_filter=None,
            metrics_to_combine=enabled_metrics,
            weights=weights_list,
            new_metric_name="combined_metric",
        )

        vals = [r["combined_metric"] for r in rows_with_metric if "combined_metric" in r]
        if not vals:
            QMessageBox.warning(self, "No combined metric", "Failed to create combined metric. Check weights.")
            return

        cmin, cmax = min(vals), max(vals)
        self.combined_range = (cmin, cmax)
        self.combined_thr_slider.setRange(cmin, cmax)
        # By default show full range (upper bound)
        self.combined_thr_slider.setValue(cmax)
        self.combined_thr_lbl.setText(f"{cmax:.6g}")

        # Reflect range in label tooltip
        self.combined_thr_lbl.setToolTip(f"Range: [{cmin:.6g}, {cmax:.6g}]")
        self.combined_thr_slider.valueChanged.connect(
            lambda _: self.combined_thr_lbl.setText(f"{self.combined_thr_slider.value():.6g}")
        )

        # Also adjust the combined threshold slider resolution via weight (implicit by FloatSlider)
        QMessageBox.information(self, "Combined metric",
                                f"Combined metric created.\nRange: [{cmin:.6g}, {cmax:.6g}].")

    def _on_apply_combined_best(self):
        if not self.all_rows:
            return
        enabled_metrics = self._enabled_metrics_list()
        if not enabled_metrics:
            QMessageBox.information(self, "No metrics enabled", "Enable at least one metric to combine.")
            return

        # Gather prefilter (separate thresholds) for enabled metrics only
        pre_thresholds: Dict[str, float] = self._gather_enabled_thresholds()

        weights_list: List[float] = [self.weight_boxes[m].value() for m in enabled_metrics]

        # Pre-filter + build combined metric
        pre_filtered = filter_and_combine(
            rows=self.all_rows,
            pre_filter=pre_thresholds,
            metrics_to_combine=enabled_metrics,
            weights=weights_list,
            new_metric_name="combined_metric",
        )

        if not pre_filtered:
            QMessageBox.information(self, "No rows after pre-filter",
                                    "No rows survive the separate-metric thresholds.")
            return

        # Apply combined threshold
        thr = self.combined_thr_slider.value() if self.combined_thr_slider else float("inf")
        by_combined = [r for r in pre_filtered if r.get("combined_metric", float("inf")) <= thr]
        self._last_combined_pass_count = len(by_combined)

        if not by_combined:
            QMessageBox.information(self, "No rows",
                                    f"No rows pass combined_metric ≤ {thr:.6g}.")
            return

        # Best row per event by combined_metric
        grouped = {}
        for r in by_combined:
            grouped.setdefault(r["event_number"], []).append(r)

        best = select_best_results_by_event(grouped, sort_metric="combined_metric")
        write_filtered_csv(best, self.filtered_csv_path)
        self.filtered_lbl.setText(f"Filtered CSV: {self.filtered_csv_path}")

        # Histogram of combined_metric for best
        fig = plt.figure(figsize=(8, 6))
        plt.hist([r["combined_metric"] for r in best], bins=20)
        plt.title("Histogram of Best Rows (combined_metric)")
        plt.xlabel("combined_metric")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        QMessageBox.information(
            self,
            "Combined metric applied",
            f"Best-per-event rows written to:\n{self.filtered_csv_path}\n"
            f"({len(best)} rows)"
        )

    def _on_convert_to_stream(self):
        if not self.filtered_csv_path or not os.path.isfile(self.filtered_csv_path):
            QMessageBox.warning(self, "Missing filtered CSV",
                                "Run a filter step to produce filtered_metrics.csv first.")
            return
        output_dir = os.path.join(os.path.dirname(self.filtered_csv_path), "filtered_metrics")
        os.makedirs(output_dir, exist_ok=True)
        self.stream_path = os.path.join(output_dir, "filtered_metrics.stream")

        try:
            write_stream_from_filtered_csv(
                filtered_csv_path=self.filtered_csv_path,
                output_stream_path=self.stream_path,
                event_col="event_number",
                streamfile_col="stream_file",
            )
        except Exception as e:
            QMessageBox.critical(self, "Stream error", f"Failed to write stream:\n{e}")
            return

        self.stream_lbl.setText(f"Stream: {self.stream_path}")
        QMessageBox.information(self, "Stream written", f"Wrote stream:\n{self.stream_path}")

    def _on_export_manifest(self):
        if not self.csv_path:
            QMessageBox.information(self, "No CSV", "Load a CSV first.")
            return

        manifest = {
            "app": "Interactive Metrics Analysis Tool (Qt6)",
            "version": APP_VERSION,
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_csv": self.csv_path,
            "output_filtered_csv": self.filtered_csv_path,
            "output_stream": self.stream_path,
            "metrics": {},
            "enabled_metrics": self._enabled_metrics_list(),
            "separate_filter_live_pass_count": self._last_separate_pass_count,
            "combined_filter_live_pass_count": self._last_combined_pass_count,
        }

        # thresholds and weights
        for metric, meta in self.metric_widgets.items():
            manifest["metrics"][metric] = {
                "enabled": bool(meta["cb"].isChecked()),
                "threshold_lte": float(meta["slider"].value()) if meta["cb"].isChecked() else None,
                "range": [float(meta["min"]), float(meta["max"])],
                "percentiles": {k: float(v) for k, v in meta["percentiles"].items()},
                "weight": float(self.weight_boxes[metric].value()) if metric in self.weight_boxes else 0.0,
            }

        base_dir = os.path.dirname(self.csv_path)
        out_path = os.path.join(base_dir, "run_manifest.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Manifest error", f"Failed to write manifest:\n{e}")
            return

        QMessageBox.information(self, "Manifest written", f"Wrote:\n{out_path}")

    # ---------------- tiny helpers ----------------

    @staticmethod
    def _h_pack(*widgets):
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            lay.addWidget(w)
        return lay


def main():
    import sys
    app = QApplication(sys.argv)
    w = MetricsTool()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
