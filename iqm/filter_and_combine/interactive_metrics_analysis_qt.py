#!/usr/bin/env python3
"""
interactive_metrics_analysis_qt.py ‒ Interactive Metrics Analysis Tool (Qt 6)

A full PyQt 6 rewrite of the original Tkinter UI.  The underlying data‑handling
functions imported from `filter_and_combine` and `gui_util` are untouched –
only the GUI layer changed.  Feature parity with the Tk version:

* CSV loader + per‑metric sliders (Section 1)
* Combined‑metric builder with weight fields + slider (Section 2)
* Histograms and Matplotlib integration (QtAgg backend)
* Save filtered CSV + optional .stream conversion
* Scrollable central panel

Replace `PyQt6` with `PySide6` if you prefer LGPL bindings – the API calls are
identical.
"""

from __future__ import annotations

import os
import sys
from functools import partial

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: N812 – Qt style

# Ensure Matplotlib uses the QtAgg backend **before** importing pyplot
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402 – after backend set

# ---------------------------------------------------------------------------
#  Domain‑specific helpers (unchanged from the Tk version)
# ---------------------------------------------------------------------------
from csv_to_stream import write_stream_from_filtered_csv
from interactive_iqm import (
    read_metric_csv,
    select_best_results_by_event,
    get_metric_ranges,
    filter_rows,
    write_filtered_csv,
    filter_and_combine,  # ⇐ helper that chains pre‑filter + combine
)

# ---------------------------------------------------------------------------
#  Constants / globals mirroring the Tk script behaviour
# ---------------------------------------------------------------------------
metrics_in_order: list[str] = [
    "weighted_rmsd",
    "fraction_outliers",
    "length_deviation",
    "angle_deviation",
    "peak_ratio",
    "percentage_unindexed",
]


def _resource_path(path: str) -> str:
    """Return *path* unchanged – placeholder for future resource lookup."""
    return path


# ---------------------------------------------------------------------------
#  Main Window class
# ---------------------------------------------------------------------------
class MetricsWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Metrics Analysis Tool (Qt6)")
        self.resize(900, 700)

        # ---------------- central widget: a scroll‑area -------------------
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        # Keep layouts/refs as attrs so we can rebuild after CSV load
        self._root_layout = QtWidgets.QVBoxLayout(container)
        self._csv_file: str | None = None
        self._filtered_csv_path: str | None = None
        self._all_rows: list[dict] | None = None

        self._build_csv_selector()
        self._analysis_widget: QtWidgets.QWidget | None = None

    # ------------------------------------------------------------------
    #  0. CSV selector header
    # ------------------------------------------------------------------
    def _build_csv_selector(self) -> None:
        frm = QtWidgets.QGroupBox("Select CSV with Normalized Metrics")
        lay = QtWidgets.QGridLayout(frm)

        self._csv_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse …")
        browse_btn.clicked.connect(self._browse_csv)

        load_btn = QtWidgets.QPushButton("Load CSV")
        load_btn.setStyleSheet("background-color: lightblue;")
        load_btn.clicked.connect(self._load_csv)

        lay.addWidget(QtWidgets.QLabel("CSV File:"), 0, 0)
        lay.addWidget(self._csv_edit, 0, 1)
        lay.addWidget(browse_btn, 0, 2)
        lay.addWidget(load_btn, 1, 0, 1, 3)

        self._root_layout.addWidget(frm)

        explanation = (
            "Load a CSV file containing *normalized* metrics for indexing quality.\n\n"
            "Section 1 — Separate Metrics:\n"
            "  • Adjust per‑metric thresholds with sliders.\n"
            "  • Histograms update for filtered rows.\n\n"
            "Section 2 — Combined Metric:\n"
            "  • Enter weights for each metric → build a single score.\n"
            "  • Slider applies threshold; best row per event is kept.\n"
            "  • Optionally convert filtered CSV → .stream."
        )
        lab = QtWidgets.QLabel(explanation)
        lab.setWordWrap(True)
        self._root_layout.addWidget(lab)

    # .................................................................
    def _browse_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select CSV file with normalized metrics", os.getcwd(), "CSV files (*.csv)"
        )
        if path:
            self._csv_edit.setText(path)

    # .................................................................
    def _load_csv(self) -> None:
        path = self._csv_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.critical(self, "Error", "Please select a CSV file.")
            return

        self._csv_file = path
        try:
            grouped = read_metric_csv(path, group_by_event=True)
            all_rows: list[dict] = []
            for rows in grouped.values():
                all_rows.extend(rows)
            self._all_rows = all_rows
            self._filtered_csv_path = os.path.join(os.path.dirname(path), "filtered_metrics.csv")
        except Exception as exc:  # noqa: BLE001 – broad but user‑facing
            QtWidgets.QMessageBox.critical(self, "Error loading CSV", str(exc))
            return

        # Build / rebuild analysis panel
        if self._analysis_widget is not None:
            self._analysis_widget.setParent(None)
        self._analysis_widget = self._build_analysis_ui()
        self._root_layout.addWidget(self._analysis_widget)

    # ------------------------------------------------------------------
    #  1. Build analysis UI (runs after CSV load)
    # ------------------------------------------------------------------
    def _build_analysis_ui(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(w)

        # ---------------- Section 1: Separate Metrics -------------------
        sep_box = QtWidgets.QGroupBox("Separate Metrics Filtering")
        sep_lay = QtWidgets.QGridLayout(sep_box)

        ranges = get_metric_ranges(self._all_rows, metrics=metrics_in_order)
        self._sep_sliders: dict[str, QtWidgets.QSlider] = {}
        for row_idx, metric in enumerate(metrics_in_order):
            mn, mx = ranges[metric]
            lab = QtWidgets.QLabel(f"{metric} ≤")
            sld = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            sld.setMinimum(0)
            sld.setMaximum(1000)  # high resolution mapping → actual range later
            sld.setValue(1000)
            sld.setFixedWidth(300)
            self._sep_sliders[metric] = sld

            # Show current numeric value
            val_lab = QtWidgets.QLabel(f"{mx:.3f}")
            sld.valueChanged.connect(partial(self._update_slider_label, sld, val_lab, mn, mx))

            sep_lay.addWidget(lab, row_idx, 0)
            sep_lay.addWidget(sld, row_idx, 1)
            sep_lay.addWidget(val_lab, row_idx, 2)

        apply_sep_btn = QtWidgets.QPushButton("Apply Separate Metrics Thresholds")
        apply_sep_btn.setStyleSheet("background-color: lightblue;")
        apply_sep_btn.clicked.connect(self._apply_separate_thresholds)
        sep_lay.addWidget(apply_sep_btn, len(metrics_in_order), 0, 1, 3)

        vlay.addWidget(sep_box)

        # ---------------- Section 2: Combined Metric --------------------
        comb_box = QtWidgets.QGroupBox("Combined Metric Creation & Filtering")
        comb_lay = QtWidgets.QGridLayout(comb_box)

        self._weight_edits: dict[str, QtWidgets.QLineEdit] = {}
        for row_idx, metric in enumerate(metrics_in_order):
            comb_lay.addWidget(QtWidgets.QLabel(f"{metric} weight:"), row_idx, 0)
            edit = QtWidgets.QLineEdit("0.0")
            edit.setFixedWidth(60)
            self._weight_edits[metric] = edit
            comb_lay.addWidget(edit, row_idx, 1)

        # Combined threshold slider + label
        self._comb_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._comb_slider.setMinimum(0)
        self._comb_slider.setMaximum(1000)
        self._comb_slider.setValue(0)
        self._comb_slider.setFixedWidth(300)
        self._comb_val_lab = QtWidgets.QLabel("0.000")
        self._comb_slider.valueChanged.connect(self._update_comb_label)

        row_thr = len(metrics_in_order)
        comb_lay.addWidget(QtWidgets.QLabel("combined_metric ≤"), row_thr, 0)
        comb_lay.addWidget(self._comb_slider, row_thr, 1)
        comb_lay.addWidget(self._comb_val_lab, row_thr, 2)

        # Buttons
        btn_create = QtWidgets.QPushButton("Create Combined Metric")
        btn_create.setStyleSheet("background-color: lightblue;")
        btn_create.clicked.connect(self._create_combined_metric)
        comb_lay.addWidget(btn_create, row_thr + 1, 0, 1, 3)

        btn_apply = QtWidgets.QPushButton("Apply Combined Metric Threshold (Best Rows)")
        btn_apply.setStyleSheet("background-color: lightblue;")
        btn_apply.clicked.connect(self._apply_combined_metric)
        comb_lay.addWidget(btn_apply, row_thr + 2, 0, 1, 3)

        btn_stream = QtWidgets.QPushButton("Convert to Stream")
        btn_stream.setStyleSheet("background-color: green; color: white;")
        btn_stream.clicked.connect(self._convert_to_stream)
        comb_lay.addWidget(btn_stream, row_thr + 3, 0, 1, 3)

        vlay.addWidget(comb_box)
        vlay.addStretch(1)
        return w

    # ------------------------------------------------------------------
    #  Helpers to reflect slider → numeric label
    # ------------------------------------------------------------------
    def _update_slider_label(self, slider: QtWidgets.QSlider, lab: QtWidgets.QLabel, mn: float, mx: float) -> None:
        val = mn + (mx - mn) * slider.value() / 1000.0
        lab.setText(f"{val:.3f}")

    def _update_comb_label(self) -> None:
        mn, mx = self._comb_range
        val = mn + (mx - mn) * self._comb_slider.value() / 1000.0
        self._comb_val_lab.setText(f"{val:.3f}")

    # ------------------------------------------------------------------
    #  Section 1 actions
    # ------------------------------------------------------------------
    def _apply_separate_thresholds(self) -> None:
        if self._all_rows is None:
            return
        print("\n" + "=" * 50)
        print("SEPARATE METRICS FILTERING")
        ranges = get_metric_ranges(self._all_rows, metrics=metrics_in_order)
        thr: dict[str, float] = {}
        for m, sld in self._sep_sliders.items():
            mn, mx = ranges[m]
            thr[m] = mn + (mx - mn) * sld.value() / 1000.0
        filtered = filter_rows(self._all_rows, thr)
        print(f"{len(self._all_rows)} rows → {len(filtered)} pass thresholds")
        if not filtered:
            QtWidgets.QMessageBox.information(self, "No rows", "No rows passed the thresholds.")
            return

        # Histograms
        plt.close("all")
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()
        for i, metric in enumerate(metrics_in_order):
            values = [r[metric] for r in filtered if metric in r]
            axes[i].hist(values, bins=20)
            axes[i].set_title(f"Histogram of {metric}")
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel("Count")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Section 2: create combined metric
    # ------------------------------------------------------------------
    def _create_combined_metric(self) -> None:
        if self._all_rows is None:
            return
        print("\n" + "=" * 50)
        print("COMBINED METRIC CREATION")

        weights = [float(self._weight_edits[m].text() or 0.0) for m in metrics_in_order]
        rows = filter_and_combine(
            rows=self._all_rows,
            pre_filter=None,
            metrics_to_combine=metrics_in_order,
            weights=weights,
            new_metric_name="combined_metric",
        )
        vals = [r["combined_metric"] for r in rows]
        if not vals:
            QtWidgets.QMessageBox.warning(self, "Failed", "Combined metric creation failed – check weights.")
            return
        mn, mx = min(vals), max(vals)
        self._comb_range = (mn, mx)
        self._comb_slider.setValue(1000)  # start at max
        self._update_comb_label()
        print(f"Combined metric OK  (min={mn:.3f},  max={mx:.3f})")

    # ------------------------------------------------------------------
    def _apply_combined_metric(self) -> None:
        if self._all_rows is None:
            return
        print("\n" + "=" * 50)
        print("COMBINED METRIC FILTERING")

        # Step 1: thresholds from Section 1 sliders
        ranges = get_metric_ranges(self._all_rows, metrics=metrics_in_order)
        pre_thr: dict[str, float] = {}
        for m, sld in self._sep_sliders.items():
            mn, mx = ranges[m]
            pre_thr[m] = mn + (mx - mn) * sld.value() / 1000.0

        # Step 2: weights
        weights = [float(self._weight_edits[m].text() or 0.0) for m in metrics_in_order]

        # Step 3: pre‑filter + combine
        rows = filter_and_combine(
            rows=self._all_rows,
            pre_filter=pre_thr,
            metrics_to_combine=metrics_in_order,
            weights=weights,
            new_metric_name="combined_metric",
        )
        print(f"After pre‑filter: {len(rows)} rows")
        if not rows:
            QtWidgets.QMessageBox.information(self, "No rows", "No rows left after pre‑filter.")
            return

        # Step 4: combined threshold
        mn, mx = self._comb_range
        thr_val = mn + (mx - mn) * self._comb_slider.value() / 1000.0
        rows2 = [r for r in rows if r["combined_metric"] <= thr_val]
        print(f"After combined_metric ≤ {thr_val:.3f}: {len(rows2)} rows")
        if not rows2:
            QtWidgets.QMessageBox.information(self, "No rows", "No rows pass combined threshold.")
            return

        # Step 5: best per event + histogram + CSV write
        grouped = {}
        for r in rows2:
            grouped.setdefault(r["event_number"], []).append(r)
        best = select_best_results_by_event(grouped, sort_metric="combined_metric")
        print(f"Best rows per event: {len(best)} rows → {self._filtered_csv_path}")
        write_filtered_csv(best, self._filtered_csv_path)

        plt.close("all")
        plt.figure(figsize=(8, 6))
        plt.hist([r["combined_metric"] for r in best], bins=20)
        plt.title("Histogram of Best Rows (combined_metric)")
        plt.xlabel("combined_metric")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Convert to .stream
    # ------------------------------------------------------------------
    def _convert_to_stream(self) -> None:
        if not (self._csv_file and self._filtered_csv_path):
            return
        print("\n" + "=" * 50)
        print("CONVERT TO STREAM")

        out_dir = os.path.join(os.path.dirname(self._csv_file), "filtered_metrics")
        os.makedirs(out_dir, exist_ok=True)
        out_stream = os.path.join(out_dir, "filtered_metrics.stream")

        print("Reading filtered CSV …")
        grouped = read_metric_csv(self._filtered_csv_path, group_by_event=True)
        first_event = next(iter(grouped.values()))
        if "combined_metric" in first_event[0]:
            best = select_best_results_by_event(grouped, sort_metric="combined_metric")
            write_filtered_csv(best, self._filtered_csv_path)
            print("Best rows selected and CSV overwritten (combined_metric present).")
        else:
            print("No combined_metric – skipping best‑row selection.")

        print("Writing .stream … →", out_stream)
        write_stream_from_filtered_csv(
            filtered_csv_path=self._filtered_csv_path,
            output_stream_path=out_stream,
            event_col="event_number",
            streamfile_col="stream_file",
        )
        QtWidgets.QMessageBox.information(self, "Done", f"CSV converted to:\n{out_stream}")


# ---------------------------------------------------------------------------
#  Entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    # Head‑less fallback: run off‑screen if no DISPLAY/Wayland found
    if not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QtWidgets.QApplication(sys.argv)
    win = MetricsWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
