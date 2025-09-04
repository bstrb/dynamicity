import os
import h5py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QTextEdit, QMessageBox,
    QComboBox, QFileDialog
)
from PyQt6.QtCore import Qt
from coseda.gandalfiterator import generate_sorted_grid_points
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from coseda.initialize import write_gandalfiteratorsettings

class GandalfIteratorWindow(QDialog):
    def __init__(self, parent, ini_directory, ini_file_path):
        super().__init__(parent)
        self.setWindowTitle("Gandalf Indexing Settings")
        self.setMinimumWidth(600)

        # Store the INI directory and path passed from the main window
        self.ini_directory = ini_directory
        self.current_input_path = ini_file_path

        # Determine maximum dimension from image dataset for XY box/step
        max_dim = None
        if parent and hasattr(parent, 'hdf5_path') and parent.hdf5_path:
            try:
                with h5py.File(parent.hdf5_path, 'r') as f:
                    images = f['entry/data/images']
                    _, height, width = images.shape
                    max_dim = max(width, height)
            except Exception:
                max_dim = None

        self._init_ui(max_dim)


    def _init_ui(self, max_dim=None):
        layout = QVBoxLayout(self)

        # --- File Selection Section ---
        # Geometry file
        geom_layout = QHBoxLayout()
        geom_label = QLabel("Geometry File:")
        self.geom_edit = QLineEdit()
        self.geom_browse_btn = QPushButton("Browse...")
        self.geom_browse_btn.clicked.connect(self.browse_geom)
        geom_layout.addWidget(geom_label)
        geom_layout.addWidget(self.geom_edit)
        geom_layout.addWidget(self.geom_browse_btn)
        layout.addLayout(geom_layout)

        # Cell file
        cell_layout = QHBoxLayout()
        cell_label = QLabel("Cell File:")
        self.cell_edit = QLineEdit()
        self.cell_browse_btn = QPushButton("Browse...")
        self.cell_browse_btn.clicked.connect(self.browse_cell)
        cell_layout.addWidget(cell_label)
        cell_layout.addWidget(self.cell_edit)
        cell_layout.addWidget(self.cell_browse_btn)
        layout.addLayout(cell_layout)

        # Output base
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Base:")
        self.output_base_edit = QLineEdit()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_base_edit)
        layout.addLayout(output_layout)

        # --- Basic Parameters Section ---
        basic_layout = QHBoxLayout()
        threads_label = QLabel("Threads:")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 1024)
        self.threads_spin.setValue(24)
        maxrad_label = QLabel("Max Radius:")
        self.max_radius_spin = QDoubleSpinBox()
        self.max_radius_spin.setRange(0.0, 100.0)
        self.max_radius_spin.setDecimals(4)
        self.max_radius_spin.setValue(0.1)
        step_label = QLabel("Step:")
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.0, 10.0)
        self.step_spin.setDecimals(4)
        self.step_spin.setValue(0.1)
        basic_layout.addWidget(threads_label)
        basic_layout.addWidget(self.threads_spin)
        basic_layout.addWidget(maxrad_label)
        basic_layout.addWidget(self.max_radius_spin)
        basic_layout.addWidget(step_label)
        basic_layout.addWidget(self.step_spin)
        layout.addLayout(basic_layout)

        # --- Peakfinder Options Section ---
        pf_label = QLabel("Peakfinder Method:")
        self.peak_method_combo = QComboBox()
        self.peak_method_combo.addItems(["cxi", "peakfinder9", "peakfinder8"])
        pf_params_label = QLabel("Peakfinder Params:")
        self.peak_params_edit = QTextEdit()
        self.peak_params_edit.setPlaceholderText("Enter one flag per line")
        layout.addWidget(pf_label)
        layout.addWidget(self.peak_method_combo)
        layout.addWidget(pf_params_label)
        layout.addWidget(self.peak_params_edit)

        # --- Advanced Indexing Parameters Section ---
        adv_grid_layout = QHBoxLayout()
        # Min peaks
        minpeaks_label = QLabel("Min Peaks:")
        self.min_peaks_spin = QSpinBox()
        self.min_peaks_spin.setRange(1, 1000)
        self.min_peaks_spin.setValue(15)
        # Cell tolerance
        celltol_label = QLabel("Cell Tolerance:")
        self.cell_tol_edit = QLineEdit()
        self.cell_tol_edit.setText("10,10,10,5")
        # Sampling pitch
        samp_label = QLabel("Sampling Pitch:")
        self.samp_pitch_spin = QSpinBox()
        self.samp_pitch_spin.setRange(1, 90)
        self.samp_pitch_spin.setValue(5)
        # Grad desc iterations
        grad_label = QLabel("Grad Desc Iters:")
        self.grad_iters_spin = QSpinBox()
        self.grad_iters_spin.setRange(0, 100)
        self.grad_iters_spin.setValue(1)
        # XGandalf tolerance
        x_tol_label = QLabel("XG Tol:")
        self.x_tol_spin = QDoubleSpinBox()
        self.x_tol_spin.setRange(0.0001, 1.0)
        self.x_tol_spin.setDecimals(4)
        self.x_tol_spin.setValue(0.02)
        # Integration radius
        int_label = QLabel("Integration Radius:")
        self.int_rad_edit = QLineEdit()
        self.int_rad_edit.setText("2,4,10")

        adv_grid_layout.addWidget(minpeaks_label)
        adv_grid_layout.addWidget(self.min_peaks_spin)
        adv_grid_layout.addWidget(celltol_label)
        adv_grid_layout.addWidget(self.cell_tol_edit)
        adv_grid_layout.addWidget(samp_label)
        adv_grid_layout.addWidget(self.samp_pitch_spin)
        adv_grid_layout.addWidget(grad_label)
        adv_grid_layout.addWidget(self.grad_iters_spin)
        adv_grid_layout.addWidget(x_tol_label)
        adv_grid_layout.addWidget(self.x_tol_spin)
        adv_grid_layout.addWidget(int_label)
        adv_grid_layout.addWidget(self.int_rad_edit)
        layout.addLayout(adv_grid_layout)

        # --- Other Extra Flags Section ---
        other_label = QLabel("Other Flags:")
        self.other_flags_edit = QTextEdit()
        self.other_flags_edit.setPlaceholderText("Enter one flag per line")
        layout.addWidget(other_label)
        layout.addWidget(self.other_flags_edit)

        # Generate pattern button and canvas
        self.generate_btn = QPushButton("Generate Pattern")
        self.generate_btn.clicked.connect(self.generate_pattern)
        layout.addWidget(self.generate_btn)

        self.figure = plt.Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Relative Offsets")
        self.ax.set_xlabel("ΔX")
        self.ax.set_ylabel("ΔY")

        # --- Run and Save Buttons ---
        self.run_btn = QPushButton("Run Indexing")
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

    # --- Browse Methods ---
    def browse_geom(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Geometry File", "", "Geometry Files (*.geom *.json);;All Files (*)")
        if fname:
            self.geom_edit.setText(fname)

    def browse_cell(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select Cell File", "", "Cell Files (*.cell);;All Files (*)")
        if fname:
            self.cell_edit.setText(fname)

    # --- Generate XY Pattern ---
    def generate_pattern(self):
        """Generate the radial XY pattern based on max_radius and step, then display."""
        max_radius = self.max_radius_spin.value()
        step = self.step_spin.value()

        if step <= 0:
            QMessageBox.warning(self, "Invalid Step", "Step size must be greater than zero.")
            return

        if max_radius <= 0:
            QMessageBox.warning(self, "Invalid Max Radius", "Max radius must be greater than zero.")
            return

        # Generate radial pattern centered at (0,0)
        points = generate_sorted_grid_points(max_radius, step)
        if not points:
            QMessageBox.information(self, "No Points", "No grid points generated for the given parameters.")
            return

        xs = [x for x, y in points]
        ys = [y for x, y in points]

        self.ax.clear()
        self.ax.scatter(xs, ys, s=10)
        self.ax.scatter(0, 0, color='red', marker='x')  # center marker
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("Radial Grid Points")
        self.ax.set_xlabel("ΔX")
        self.ax.set_ylabel("ΔY")
        self.canvas.draw()

    # --- Save Settings to INI ---
    def _on_save_clicked(self):
        if not self.current_input_path:
            QMessageBox.warning(self, "No INI File", "Cannot determine INI file to save settings.")
            return

        geom_path = self.geom_edit.text().strip()
        cell_path = self.cell_edit.text().strip()
        output_base = self.output_base_edit.text().strip()

        threads = self.threads_spin.value()
        max_radius = self.max_radius_spin.value()
        step = self.step_spin.value()

        peak_method = self.peak_method_combo.currentText()
        peak_params = [line.strip() for line in self.peak_params_edit.toPlainText().splitlines() if line.strip()]

        min_peaks = self.min_peaks_spin.value()
        cell_tol = self.cell_tol_edit.text().strip()
        samp_pitch = self.samp_pitch_spin.value()
        grad_iters = self.grad_iters_spin.value()
        x_tol = self.x_tol_spin.value()
        int_rad = self.int_rad_edit.text().strip()

        other = [line.strip() for line in self.other_flags_edit.toPlainText().splitlines() if line.strip()]

        write_gandalfiteratorsettings(
            input_path=self.current_input_path,
            geomfile_path=geom_path,
            cellfile_path=cell_path,
            output_file_base=output_base,
            threads=threads,
            max_radius=max_radius,
            step=step,
            peakfinder_method=peak_method,
            peakfinder_params=peak_params,
            min_peaks=min_peaks,
            cell_tolerance=cell_tol,
            sampling_pitch=samp_pitch,
            grad_desc_iterations=grad_iters,
            xgandalf_tolerance=x_tol,
            int_radius=int_rad,
            other_flags=other
        )
        QMessageBox.information(self, "Settings Saved", "Gandalf settings have been saved to the INI file.")

    # --- Run Indexing (placeholder) ---
    def _on_run_clicked(self):
        QMessageBox.information(self, "Run Indexing", "Indexing logic needs to be implemented.")

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ini_dir = "/Users/xiaodong/Desktop/dynamicity/coseda/config"
    ini_file = ini_dir + "/default.ini"
    win = GandalfIteratorWindow(None, ini_dir, ini_file)
    win.show()
    sys.exit(app.exec())
