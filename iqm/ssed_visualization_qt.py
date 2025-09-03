#!/usr/bin/env python3
# qt_visualizer.py
import sys
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QTextEdit
)

# Import your plotting function (kept backward-compatible)
from visualization.indexing_histograms import plot_indexing_rate


class Worker(QThread):
    progressed = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def run(self):
        try:
            self.progressed.emit(f"Generating visualizations for folder: {self.folder}")
            # Call the provided function; it will create plots with Matplotlib
            plot_indexing_rate(self.folder)
            self.progressed.emit("Visualization completed successfully.")
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualization GUI (PyQt6)")
        self.setMinimumWidth(600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Description
        desc = (
            "Select a folder with indexing output *.stream files from Gandalf iterations with shifted centers.\n"
            "Click 'Generate Visualizations' to extract x/y from filenames, compute indexing rate "
            "(num_reflections / num_peaks * 100), and generate:\n"
            "  • 3D surface (height = indexing rate %)\n"
            "  • 2D heat-map (color = indexing rate %)\n"
        )
        self.desc_label = QLabel(desc)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

        # Folder picker
        row = QHBoxLayout()
        layout.addLayout(row)

        self.folder_label = QLabel("Stream File Folder:")
        row.addWidget(self.folder_label)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(str(Path.cwd()))
        row.addWidget(self.folder_edit, stretch=1)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.on_browse)
        row.addWidget(self.browse_btn)

        # Action button
        self.run_btn = QPushButton("Generate Visualizations")
        self.run_btn.clicked.connect(self.on_run)
        layout.addWidget(self.run_btn)

        # Log area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        layout.addWidget(self.log)

        self.worker: Worker | None = None

    def on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", os.getcwd())
        if folder:
            self.folder_edit.setText(folder)

    def on_run(self):
        folder = self.folder_edit.text().strip() or str(Path.cwd())
        if not Path(folder).exists():
            QMessageBox.critical(self, "Error", "Please select a valid folder.")
            return

        # Disable UI while running
        self.run_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.append_log(f"Starting…\nFolder: {folder}")

        # Start worker thread
        self.worker = Worker(folder)
        self.worker.progressed.connect(self.append_log)
        self.worker.finished_ok.connect(self.on_done_ok)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done_ok(self):
        self.append_log("Done.")
        self.run_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.worker = None

    def on_failed(self, msg: str):
        self.append_log(f"Error: {msg}")
        QMessageBox.critical(self, "Error during visualization", msg)
        self.run_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.worker = None

    def append_log(self, text: str):
        self.log.append(text)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
