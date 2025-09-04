from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFormLayout, QCheckBox, QSpinBox, QFileDialog, QMessageBox, QTabWidget, QComboBox, QTextEdit
)
import subprocess
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
import os
import signal

class IndexingControlWindow(QWidget):

    class ProcessOutputThread(QThread):
        output_received = pyqtSignal(str)

        def __init__(self, process):
            super().__init__()
            self.process = process

        def run(self):
            # Read lines until process finishes
            for line in iter(self.process.stdout.readline, b''):
                try:
                    text = line.decode("utf-8")
                except:
                    text = str(line)
                self.output_received.emit(text)
    def __init__(self, main_window, ini_directory=None, h5_path=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.ini_directory = ini_directory
        # h5_path is passed from main_window; ini_path is retrieved from main_window
        self.h5_path = h5_path
        self.ini_path = getattr(main_window, 'full_ini_file_path', None)
        from configparser import ConfigParser
        import os

        self.setWindowTitle("Indexing & Integration")

        # Read ini file and paths
        parser = ConfigParser()
        parser.read(self.ini_path)


        base_dir = os.path.dirname(self.ini_path)
        output_folder = parser.get("Paths", "outputfolder", fallback="")
        output_path = os.path.join(base_dir, output_folder)

        # Detect previous run for cell-file prepopulation (use second newest folder)
        import glob
        run_base = os.path.join(base_dir, output_folder) if output_folder else base_dir
        prev_runs = sorted(glob.glob(os.path.join(run_base, "indexingintegration_*")))
        if len(prev_runs) >= 2:
            target_dir = prev_runs[-2]
            candidate = os.path.join(target_dir, "cellfile.cell")
            self.prev_cell_file = candidate if os.path.exists(candidate) else None
        else:
            self.prev_cell_file = None

        # Create timestamped directory for this run
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_base = output_path if output_folder else base_dir
        self.run_dir = os.path.join(run_base, f"indexingintegration_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # File paths
        self.cell_filepath = os.path.join(self.run_dir, "cellfile.cell")
        # Always place geometry file in run_dir by default
        self.geom_filepath = os.path.join(self.run_dir, "geometry.geom")

        # (Previous run detection and prev_cell_file assignment now handled above)

        # Create list file in run_dir using provided HDF5 path
        self.list_filepath = os.path.join(self.run_dir, "input.lst")
        if self.h5_path:
            try:
                with open(self.list_filepath, "w") as lst_f:
                    lst_f.write(os.path.abspath(self.h5_path))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to write list file: {e}")
        else:
            QMessageBox.critical(self, "Error", "No HDF5 file specified; please select one.")

        # Compute stream file path (not created)
        run_folder_name = os.path.basename(self.run_dir)
        self.stream_filepath = os.path.join(self.run_dir, f"{run_folder_name}.stream")

        # Initialize tab widget
        self.tabs = QTabWidget()
        self._init_index_integrate_tab()
        self._init_cellfile_tab()
        self._init_geomfile_tab()
        self._init_start_indexing_tab()

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # Load initial data for cell and geom tabs
        self._load_cell_file()
        self._load_geom_file()

        # Update command label after all tabs/widgets are initialized
        self._update_command_label()

        # Placeholder for indexing subprocess
        self.indexing_process = None


    # --- Index & Integrate Tab ---
    def _init_index_integrate_tab(self):
        from PyQt6.QtGui import QDoubleValidator
        self.index_tab = QWidget()
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.threads_spin = QSpinBox()
        self.threads_spin.setMinimum(1)
        self.threads_spin.setMaximum(128)
        self.threads_spin.setValue(12)

        # Restore push resolution and integration radius rows, with validators/placeholders
        self.push_res_edit = QLineEdit("0.5")
        self.push_res_edit.setValidator(QDoubleValidator(bottom=0.0))
        self.push_res_edit.setPlaceholderText("e.g. 0.5")
        form_layout.addRow("Push resolution:", self.push_res_edit)
        self.int_radius_edit = QLineEdit("4,5,8")
        self.int_radius_edit.setPlaceholderText("e.g. 4,5,8")
        form_layout.addRow("Integration radius:", self.int_radius_edit)
        self.min_peaks_spin = QSpinBox()
        self.min_peaks_spin.setValue(25)
        form_layout.addRow("Min peaks:", self.min_peaks_spin)

        self.tolerance_edit = QLineEdit("5,5,5,5")
        form_layout.addRow("Tolerance:", self.tolerance_edit)
        # Insert new XGandalf tolerance field after tolerance
        self.xgandalf_tolerance_edit = QLineEdit("0.5")
        form_layout.addRow("XGandalf tolerance:", self.xgandalf_tolerance_edit)
        # connect to update command label
        self.xgandalf_tolerance_edit.textChanged.connect(self._update_command_label)

        self.sampling_pitch_spin = QSpinBox()
        self.sampling_pitch_spin.setValue(5)
        form_layout.addRow("Sampling pitch:", self.sampling_pitch_spin)

        self.min_lat_vec_len_spin = QSpinBox()
        self.min_lat_vec_len_spin.setValue(3)
        form_layout.addRow("Min lattice vec length:", self.min_lat_vec_len_spin)

        self.max_lat_vec_len_spin = QSpinBox()
        self.max_lat_vec_len_spin.setValue(30)
        form_layout.addRow("Max lattice vec length:", self.max_lat_vec_len_spin)

        self.grad_desc_iterations_spin = QSpinBox()
        self.grad_desc_iterations_spin.setValue(2)
        form_layout.addRow("Gradient descent iterations:", self.grad_desc_iterations_spin)

        self.fix_profile_radius_edit = QLineEdit("50000000")
        form_layout.addRow("Fix profile radius:", self.fix_profile_radius_edit)
        # Connect to update command label
        self.fix_profile_radius_edit.textChanged.connect(self._update_command_label)

        # Add checkboxes for optional flags
        self.no_non_hits_cb = QCheckBox("No non-hits in stream")
        self.no_non_hits_cb.setChecked(True)
        form_layout.addRow(self.no_non_hits_cb)
        self.no_non_hits_cb.stateChanged.connect(self._update_command_label)

        self.no_revalidate_cb = QCheckBox("No revalidate")
        self.no_revalidate_cb.setChecked(True)
        form_layout.addRow(self.no_revalidate_cb)
        self.no_revalidate_cb.stateChanged.connect(self._update_command_label)

        self.no_half_pixel_shift_cb = QCheckBox("No half-pixel shift")
        self.no_half_pixel_shift_cb.setChecked(True)
        form_layout.addRow(self.no_half_pixel_shift_cb)
        self.no_half_pixel_shift_cb.stateChanged.connect(self._update_command_label)

        self.no_retry_cb = QCheckBox("No retry")
        self.no_retry_cb.setChecked(True)
        form_layout.addRow(self.no_retry_cb)
        self.no_retry_cb.stateChanged.connect(self._update_command_label)

        self.no_check_cell_cb = QCheckBox("No check cell")
        self.no_check_cell_cb.setChecked(True)
        form_layout.addRow(self.no_check_cell_cb)
        self.no_check_cell_cb.stateChanged.connect(self._update_command_label)

        self.save_index_settings_btn = QPushButton("Save Settings")
        self.save_index_settings_btn.clicked.connect(self._save_index_settings)

        layout.addLayout(form_layout)
        layout.addWidget(self.save_index_settings_btn)
        self.index_tab.setLayout(layout)
        self.tabs.addTab(self.index_tab, "Index & Integrate")

        # Connect signals to update the command label when values change
        self.threads_spin.valueChanged.connect(self._update_command_label)
        self.push_res_edit.textChanged.connect(self._update_command_label)
        self.int_radius_edit.textChanged.connect(self._update_command_label)
        self.min_peaks_spin.valueChanged.connect(self._update_command_label)
        self.tolerance_edit.textChanged.connect(self._update_command_label)
        self.xgandalf_tolerance_edit.textChanged.connect(self._update_command_label)
        self.sampling_pitch_spin.valueChanged.connect(self._update_command_label)
        self.min_lat_vec_len_spin.valueChanged.connect(self._update_command_label)
        self.max_lat_vec_len_spin.valueChanged.connect(self._update_command_label)
        self.grad_desc_iterations_spin.valueChanged.connect(self._update_command_label)
        # Already connected: self.fix_profile_radius_edit.textChanged.connect(self._update_command_label)

    def _save_index_settings(self):
        from coseda.initialize import write_xgandalfsettings
        try:
            import os
            from configparser import ConfigParser

            parser = ConfigParser()
            parser.read(self.ini_path)
            base_dir = os.path.dirname(self.ini_path)
            output_folder = parser.get("Paths", "outputfolder", fallback="")

            # Save updated geom and cell file paths back to ini if changed
            # (No longer using QLineEdit for these fields, so skip updating from edits)

            # pass separate XGandalf tolerance if supported by write_xgandalfsettings
            try:
                write_xgandalfsettings(
                    self.ini_path,
                    tolerance=self.tolerance_edit.text(),
                    xgandalf_tolerance=self.xgandalf_tolerance_edit.text(),
                    sampling_pitch=self.sampling_pitch_spin.value(),
                    min_lattice_vector_length=self.min_lat_vec_len_spin.value(),
                    max_lattice_vector_length=self.max_lat_vec_len_spin.value(),
                    grad_desc_iterations=self.grad_desc_iterations_spin.value(),
                    tolerance_5d=0.2,
                    fix_profile_radius=True
                )
            except TypeError:
                # fallback if original function does not accept xgandalf_tolerance
                write_xgandalfsettings(
                    self.ini_path,
                    tolerance=self.tolerance_edit.text(),
                    sampling_pitch=self.sampling_pitch_spin.value(),
                    min_lattice_vector_length=self.min_lat_vec_len_spin.value(),
                    max_lattice_vector_length=self.max_lat_vec_len_spin.value(),
                    grad_desc_iterations=self.grad_desc_iterations_spin.value(),
                    tolerance_5d=0.2,
                    fix_profile_radius=True
                )
            QMessageBox.information(self, "Saved", "Index & Integrate settings saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # --- Cell File Editor Tab ---
    def _init_cellfile_tab(self):
        self.cell_tab = QWidget()
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Lattice type combobox
        self.lattice_type_combo = QComboBox()
        self.lattice_type_combo.addItems([
            "triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"
        ])
        # Centering combobox
        self.centering_combo = QComboBox()
        self.centering_combo.addItems([
            "P", "A", "B", "C", "I", "F", "R"
        ])
        # Unique axis combobox
        self.unique_axis_combo = QComboBox()
        self.unique_axis_combo.addItems([
            "a", "b", "c"
        ])
        # --- lattice type/unique axis logic ---
        def on_lattice_type_changed(text):
            t = text.lower()
            # Unique axis only for monoclinic
            if t == "monoclinic":
                self.unique_axis_combo.setEnabled(True)
            else:
                self.unique_axis_combo.setCurrentIndex(-1)
                self.unique_axis_combo.setEnabled(False)
            # Auto-set angles to 90 for cubic, tetragonal, orthorhombic
            if t in ("cubic", "tetragonal", "orthorhombic"):
                self.al_edit.setText("90")
                self.be_edit.setText("90")
                self.ga_edit.setText("90")

        self.lattice_type_combo.currentTextChanged.connect(on_lattice_type_changed)
        # Initialize state based on default lattice type
        on_lattice_type_changed(self.lattice_type_combo.currentText())

        self.a_edit = QLineEdit()
        self.b_edit = QLineEdit()
        self.c_edit = QLineEdit()
        self.al_edit = QLineEdit()
        self.be_edit = QLineEdit()
        self.ga_edit = QLineEdit()

        form_layout.addRow("Lattice type:", self.lattice_type_combo)
        form_layout.addRow("Centering:", self.centering_combo)
        form_layout.addRow("Unique axis:", self.unique_axis_combo)
        form_layout.addRow("a:", self.a_edit)
        form_layout.addRow("b:", self.b_edit)
        form_layout.addRow("c:", self.c_edit)
        form_layout.addRow("α:", self.al_edit)
        form_layout.addRow("β:", self.be_edit)
        form_layout.addRow("γ:", self.ga_edit)

        layout.addLayout(form_layout)

        # Timer for debounced saving
        self.cell_timer = QTimer(self)
        self.cell_timer.setSingleShot(True)
        self.cell_timer.timeout.connect(self._save_cell_file)
        # Connect change signals to start timer
        self.lattice_type_combo.currentTextChanged.connect(lambda: self.cell_timer.start(500))
        self.centering_combo.currentTextChanged.connect(lambda: self.cell_timer.start(500))
        self.unique_axis_combo.currentTextChanged.connect(lambda: self.cell_timer.start(500))
        self.a_edit.textChanged.connect(lambda: self.cell_timer.start(500))
        self.b_edit.textChanged.connect(lambda: self.cell_timer.start(500))
        self.c_edit.textChanged.connect(lambda: self.cell_timer.start(500))
        self.al_edit.textChanged.connect(lambda: self.cell_timer.start(500))
        self.be_edit.textChanged.connect(lambda: self.cell_timer.start(500))
        self.ga_edit.textChanged.connect(lambda: self.cell_timer.start(500))

        self.cell_tab.setLayout(layout)
        self.tabs.addTab(self.cell_tab, "Cell File Editor")

    def _load_cell_file(self):
        import os
        # Determine which cell file to load: previous run or new run
        path = self.prev_cell_file if getattr(self, "prev_cell_file", None) and os.path.exists(self.prev_cell_file) else self.cell_filepath
        print(f"DEBUG: _load_cell_file loading from {path}")
        if path and os.path.exists(path):
            try:
                # Read and strip lines
                with open(path, "r") as f:
                    raw_lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
                print(f"DEBUG: raw_lines = {raw_lines}")
                # Skip header if present
                if raw_lines and "version" in raw_lines[0].lower():
                    raw_lines = raw_lines[1:]
                # Parse key/value lines (supports '=' and ':' delimiters)
                cell_vals = {}
                for ln in raw_lines:
                    if "=" in ln:
                        k, v = ln.split("=", 1)
                    elif ":" in ln:
                        k, v = ln.split(":", 1)
                    else:
                        continue
                    cell_vals[k.strip()] = v.strip()
                print(f"DEBUG: parsed cell_vals = {cell_vals}")
                # Populate UI fields
                if "lattice_type" in cell_vals:
                    self.lattice_type_combo.setCurrentText(cell_vals["lattice_type"])
                if "centering" in cell_vals:
                    self.centering_combo.setCurrentText(cell_vals["centering"])
                if "unique_axis" in cell_vals:
                    self.unique_axis_combo.setCurrentText(cell_vals["unique_axis"])
                if "a" in cell_vals:
                    self.a_edit.setText(cell_vals["a"].split()[0])
                if "b" in cell_vals:
                    self.b_edit.setText(cell_vals["b"].split()[0])
                if "c" in cell_vals:
                    self.c_edit.setText(cell_vals["c"].split()[0])
                if "al" in cell_vals:
                    self.al_edit.setText(cell_vals["al"].split()[0])
                if "be" in cell_vals:
                    self.be_edit.setText(cell_vals["be"].split()[0])
                if "ga" in cell_vals:
                    self.ga_edit.setText(cell_vals["ga"].split()[0])
            except Exception as e:
                print(f"DEBUG: exception parsing cell file: {e}")
        # Update command label in case path changed
        self._update_command_label()

    def _save_cell_file(self):
        import os
        from configparser import ConfigParser
        # New logic for field extraction and validation
        lt = self.lattice_type_combo.currentText().strip()
        cent = self.centering_combo.currentText().strip()
        ua = self.unique_axis_combo.currentText().strip() if self.unique_axis_combo.isEnabled() else ""
        a = self.a_edit.text().strip()
        b = self.b_edit.text().strip()
        c = self.c_edit.text().strip()
        al = self.al_edit.text().strip()
        be = self.be_edit.text().strip()
        ga = self.ga_edit.text().strip()

        if not all([lt, cent, a, b, c, al, be, ga]):
            return  # do not save until all required fields are filled
        if self.unique_axis_combo.isEnabled() and not ua:
            return  # do not save until unique axis is set when enabled

        try:
            # Validate numeric fields
            float(a)
            float(b)
            float(c)
            float(al)
            float(be)
            float(ga)
        except ValueError:
            QMessageBox.critical(self, "Error", "Please enter valid numeric values for all cell parameters.")
            return

        if not self.cell_filepath:
            QMessageBox.critical(self, "Error", "Cell file path not defined.")
            return

        try:
            with open(self.cell_filepath, "w") as f:
                f.write("CrystFEL unit cell file version 1.0\n\n")
                # lattice_type
                f.write(f"lattice_type = {lt}\n\n")
                # centering and unique axis
                f.write(f"centering = {cent}\n")
                if ua:
                    f.write(f"unique_axis = {ua}\n\n")
                else:
                    f.write("\n")
                # cell parameters
                f.write(f"a = {a} A\n")
                f.write(f"b = {b} A\n")
                f.write(f"c = {c} A\n")
                f.write(f"al = {al} deg\n")
                f.write(f"be = {be} deg\n")
                f.write(f"ga = {ga} deg\n")
            # After saving, update cellfile entry in INI to point to relative path from base_dir
            parser = ConfigParser()
            parser.read(self.ini_path)
            base_dir = os.path.dirname(self.ini_path)
            rel_path = os.path.relpath(self.cell_filepath, base_dir)
            if not parser.has_section("Paths"):
                parser.add_section("Paths")
            parser.set("Paths", "cellfile", rel_path)
            with open(self.ini_path, "w") as ini_file:
                parser.write(ini_file)
            # Confirmation message removed as requested
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save cell file: {e}")

    # --- Geometry File Editor Tab ---
    def _init_geomfile_tab(self):
        self.geom_tab = QWidget()
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Geometry file fields (CrystFEL .geom format, common keys)
        self.geom_fields = [
            ("wavelength", "Wavelength (Å):"),
            ("adu_per_photon", "ADU per photon:"),
            ("clen", "Camera length (m):"),
            ("res", "Resolution (px/m):"),
            ("data", "Data (data):"),
            ("dim0", "dim0:"),
            ("dim1", "dim1:"),
            ("dim2", "dim2:"),
            ("peak_list", "Peak list:"),
            ("peak_list_type", "Peak list type:"),
            ("detector_shift_x", "Detector shift X:"),
            ("detector_shift_y", "Detector shift Y:"),
            ("min_ss", "min_ss:"),
            ("max_ss", "max_ss:"),
            ("min_fs", "min_fs:"),
            ("max_fs", "max_fs:"),
            ("corner_x", "corner_x:"),
            ("corner_y", "corner_y:"),
            ("fs", "fs:"),
            ("ss", "ss:"),
            ("mask", "Mask:"),
            ("mask_good", "Mask good:"),
            ("mask_bad", "Mask bad:"),
        ]
        self.geom_edits = {}
        for key, label in self.geom_fields:
            edit = QLineEdit()
            self.geom_edits[key] = edit
            form_layout.addRow(label, edit)

        layout.addLayout(form_layout)

        # Timer for debounced saving
        self.geom_timer = QTimer(self)
        self.geom_timer.setSingleShot(True)
        self.geom_timer.timeout.connect(self._save_geom_file)
        for key in self.geom_fields:
            self.geom_edits[key[0]].textChanged.connect(lambda: self.geom_timer.start(500))

        self.geom_tab.setLayout(layout)
        self.tabs.addTab(self.geom_tab, "Geometry File Editor")

    def _load_geom_file(self):
        import os
        if self.geom_filepath and os.path.exists(self.geom_filepath):
            try:
                # Read .geom file as key = value lines, skip comments and blank lines
                with open(self.geom_filepath, "r") as f:
                    lines = f.readlines()
                geom_dict = {}
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith(";") or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        geom_dict[k.strip()] = v.strip()
                for key, _ in self.geom_fields:
                    val = geom_dict.get(key, "")
                    self.geom_edits[key].setText(val)
            except Exception:
                pass
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read(self.ini_path)
        # Fallback: try to extract geometry values from [AcquisitionDetails] if present
        if "AcquisitionDetails" in parser:
            acquisition = parser["AcquisitionDetails"]
            if self.geom_edits["wavelength"].text().strip() == "" and "acceleration_voltage" in acquisition:
                from math import sqrt
                voltage = float(acquisition["acceleration_voltage"])
                h = 6.62607015e-34
                m = 9.10938356e-31
                e = 1.602176634e-19
                c = 299792458
                lambda_m = (h / sqrt(2 * m * e * voltage) / sqrt(1 + (e * voltage) / (2 * m * c**2)) * 10000000000)
                self.geom_edits["wavelength"].setText(str(lambda_m))

            if self.geom_edits["clen"].text().strip() == "" and "camera_length" in acquisition:
                try:
                    cam_len = float(acquisition["camera_length"])
                    if "camera_length_correction" in acquisition:
                        cam_len *= float(acquisition["camera_length_correction"])
                    self.geom_edits["clen"].setText(str(cam_len))
                except ValueError:
                    # Fallback to raw value if parsing fails
                    self.geom_edits["clen"].setText(acquisition["camera_length"])

            if self.geom_edits["res"].text().strip() == "" and "pixels_per_meter" in acquisition:
                self.geom_edits["res"].setText(acquisition["pixels_per_meter"])

            # Set adu_per_photon from acquisition if available and field is empty
            if self.geom_edits["adu_per_photon"].text().strip() == "" and "adu_per_photon" in acquisition:
                self.geom_edits["adu_per_photon"].setText(acquisition["adu_per_photon"])

            try:
                w = int(acquisition["resolution_width"])
                h = int(acquisition["resolution_height"])
                bw = int(acquisition["binning_width"])
                bh = int(acquisition["binning_height"])
                fs = w // bw
                ss = h // bh
                if self.geom_edits["max_fs"].text().strip() == "":
                    self.geom_edits["max_fs"].setText(str(fs - 1))
                if self.geom_edits["max_ss"].text().strip() == "":
                    self.geom_edits["max_ss"].setText(str(ss - 1))
                if self.geom_edits["corner_x"].text().strip() == "":
                    self.geom_edits["corner_x"].setText(str(-fs // 2))
                if self.geom_edits["corner_y"].text().strip() == "":
                    self.geom_edits["corner_y"].setText(str(-ss // 2))
                # --- Add default values for missing fields after setting corner_x, corner_y ---
                if self.geom_edits["min_fs"].text().strip() == "":
                    self.geom_edits["min_fs"].setText("0")
                if self.geom_edits["min_ss"].text().strip() == "":
                    self.geom_edits["min_ss"].setText("0")
                if self.geom_edits["fs"].text().strip() == "":
                    self.geom_edits["fs"].setText("x")
                if self.geom_edits["ss"].text().strip() == "":
                    self.geom_edits["ss"].setText("y")
                if self.geom_edits["mask"].text().strip() == "":
                    self.geom_edits["mask"].setText("/mask")
                if self.geom_edits["mask_good"].text().strip() == "":
                    self.geom_edits["mask_good"].setText("0x01")
                if self.geom_edits["mask_bad"].text().strip() == "":
                    self.geom_edits["mask_bad"].setText("0x00")
                # --- End added defaults ---
            except Exception:
                pass
        # Additional defaults for geometry fields
        if self.geom_edits["data"].text().strip() == "":
            self.geom_edits["data"].setText("/entry/data/images")
        if self.geom_edits["dim0"].text().strip() == "":
            self.geom_edits["dim0"].setText("%")
        if self.geom_edits["dim1"].text().strip() == "":
            self.geom_edits["dim1"].setText("ss")
        if self.geom_edits["dim2"].text().strip() == "":
            self.geom_edits["dim2"].setText("fs")
        if self.geom_edits["peak_list"].text().strip() == "":
            self.geom_edits["peak_list"].setText("/entry/data/")
        if self.geom_edits["peak_list_type"].text().strip() == "":
            self.geom_edits["peak_list_type"].setText("cxi")
        if self.geom_edits["detector_shift_x"].text().strip() == "":
            self.geom_edits["detector_shift_x"].setText("/entry/data/det_shift_x_mm mm")
        if self.geom_edits["detector_shift_y"].text().strip() == "":
            self.geom_edits["detector_shift_y"].setText("/entry/data/det_shift_y_mm mm")

        # Update command label in case path changed
        self._update_command_label()

    def _save_geom_file(self):
        import os
        from configparser import ConfigParser

        # Required fields: wavelength, clen, res
        wl = self.geom_edits["wavelength"].text().strip()
        clen = self.geom_edits["clen"].text().strip()
        res = self.geom_edits["res"].text().strip()
        if not all([wl, clen, res]):
            return  # do not save until required fields are filled

        if not self.geom_filepath:
            QMessageBox.critical(self, "Error", "Geometry file path not defined in INI file.")
            return

        try:
            with open(self.geom_filepath, "w") as f:
                # Header comments
                f.write(";Detector file generated by COSEDA.\n")

                # Core geometry settings
                f.write(f"wavelength  = {wl} A\n")
                val = self.geom_edits["adu_per_photon"].text().strip()
                if val:
                    f.write(f"adu_per_photon = {val}\n")
                f.write(f"clen = {clen} m\n")
                f.write(f"res = {res}\n")

                # Standard defaults and user edits
                keys_order = [
                    "data", "dim0", "dim1", "dim2",
                    "peak_list", "peak_list_type",
                    "detector_shift_x", "detector_shift_y",
                    "min_ss", "max_ss", "min_fs", "max_fs",
                    "corner_x", "corner_y", "fs", "ss",
                    "mask", "mask_good", "mask_bad"
                ]
                for key in keys_order:
                    val = self.geom_edits[key].text().strip()
                    if not val:
                        continue
                    if key in ["min_ss", "max_ss", "min_fs", "max_fs",
                               "corner_x", "corner_y", "fs",
                               "ss", "mask", "mask_good", "mask_bad"]:
                        f.write(f"p0/{key} = {val}\n")
                    else:
                        f.write(f"{key} = {val}\n")

            # Update INI if needed (if paths change)
            parser = ConfigParser()
            parser.read(self.ini_path)
            base_dir = os.path.dirname(self.ini_path)
            rel_path = os.path.relpath(self.geom_filepath, base_dir)
            if not parser.has_section("Paths"):
                parser.add_section("Paths")
            parser.set("Paths", "geomfile", rel_path)

            # Save adu_per_photon to INI AcquisitionDetails
            if not parser.has_section("AcquisitionDetails"):
                parser.add_section("AcquisitionDetails")
            parser.set("AcquisitionDetails", "adu_per_photon", self.geom_edits["adu_per_photon"].text().strip())

            with open(self.ini_path, "w") as ini_file:
                parser.write(ini_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save geometry file: {e}")

    # --- Start Indexing Tab (Placeholder) ---
    def _init_start_indexing_tab(self):
        self.start_tab = QWidget()
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Use QLineEdit for horizontally scrollable, read-only paths
        self.geom_path_edit = QLineEdit(self.geom_filepath or "")
        self.geom_path_edit.setReadOnly(True)
        form_layout.addRow("Geometry file:", self.geom_path_edit)

        self.cell_path_edit = QLineEdit(self.cell_filepath or "")
        self.cell_path_edit.setReadOnly(True)
        form_layout.addRow("Cell file:", self.cell_path_edit)

        self.list_path_edit = QLineEdit(self.list_filepath)
        self.list_path_edit.setReadOnly(True)
        form_layout.addRow("List file:", self.list_path_edit)

        self.stream_path_edit = QLineEdit(self.stream_filepath)
        self.stream_path_edit.setReadOnly(True)
        form_layout.addRow("Output stream:", self.stream_path_edit)

        # Command edit (read-only single-line QLineEdit)
        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        form_layout.addRow("Command:", self.command_edit)

        # Add Start and Stop buttons
        self.start_button = QPushButton("Start Indexing")
        self.start_button.clicked.connect(self._start_indexing)
        form_layout.addRow(self.start_button)

        self.stop_button = QPushButton("Stop Indexing")
        self.stop_button.clicked.connect(self._stop_indexing)
        form_layout.addRow(self.stop_button)

        # Add output_edit QTextEdit for logs/output
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        layout.addLayout(form_layout)
        layout.addWidget(self.output_edit)

        self.start_tab.setLayout(layout)
        self.tabs.addTab(self.start_tab, "Start Indexing")

    def _update_command_label(self):
        # Construct indexamajig command string
        geom = self.geom_filepath
        lst = self.list_filepath
        stream = self.stream_filepath
        cell = self.cell_filepath
        threads = self.threads_spin.value()
        push_res = self.push_res_edit.text().strip()
        int_radius = self.int_radius_edit.text().strip()
        min_peaks = self.min_peaks_spin.value()
        tol = self.tolerance_edit.text().strip()
        xg_tol = self.xgandalf_tolerance_edit.text().strip()
        samp_pitch = self.sampling_pitch_spin.value()
        min_lat = self.min_lat_vec_len_spin.value()
        max_lat = self.max_lat_vec_len_spin.value()
        gd_iters = self.grad_desc_iterations_spin.value()
        fix_profile_radius = self.fix_profile_radius_edit.text().strip()

        # Collect optional flags from checkboxes
        flags = []
        if hasattr(self, "no_non_hits_cb") and self.no_non_hits_cb.isChecked():
            flags.append("--no-non-hits-in-stream")
        if hasattr(self, "no_revalidate_cb") and self.no_revalidate_cb.isChecked():
            flags.append("--no-revalidate")
        if hasattr(self, "no_half_pixel_shift_cb") and self.no_half_pixel_shift_cb.isChecked():
            flags.append("--no-half-pixel-shift")
        if hasattr(self, "no_retry_cb") and self.no_retry_cb.isChecked():
            flags.append("--no-retry")
        if hasattr(self, "no_check_cell_cb") and self.no_check_cell_cb.isChecked():
            flags.append("--no-check-cell")
        flags_str = " ".join(flags)

        cmd = (
            f"indexamajig -g {geom} "
            f"-i {lst} "
            f"-o {stream} "
            f"-j {threads} "
            f"-p {cell} "
            f"--indexing=xgandalf "
            f"--push-res={push_res} "
            f"--integration=rings "
            f"{flags_str} "
            f"--int-radius={int_radius} "
            f"--peaks=cxi "
            f"--max-indexer-threads={threads} "
            f"--min-peaks={min_peaks} "
            f"--xgandalf-tolerance={xg_tol} "
            f"--xgandalf-sampling-pitch={samp_pitch} "
            f"--xgandalf-min-lattice-vector-length={min_lat} "
            f"--xgandalf-max-lattice-vector-length={max_lat} "
            f"--xgandalf-grad-desc-iterations={gd_iters} "
            f"--tolerance={tol} "
            f"--fix-profile-radius={fix_profile_radius}"
        )
        # For QLineEdit, setText instead of setPlainText
        if hasattr(self, "command_edit"):
            self.command_edit.setText(cmd)
        
    def _start_indexing(self):
        # Do not start if already running
        if self.indexing_process and self.indexing_process.poll() is None:
            return
        cmd = self.command_edit.text().strip()
        if not cmd:
            return
        try:
            # Clear output edit if present
            if hasattr(self, "output_edit"):
                self.output_edit.clear()
            # Open log file for writing
            self.log_file = open(os.path.join(self.run_dir, "indexing.log"), "w")
            # Launch subprocess and capture output
            # Use a new process group so that the stop button can terminate all children
            self.indexing_process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                preexec_fn=os.setsid
            )
            # Start a thread to read process output
            self.process_thread = self.ProcessOutputThread(self.indexing_process)
            self.process_thread.output_received.connect(self._append_output)
            self.process_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start indexing:\n{e}")

    def _append_output(self, text):
        # Append to output_edit and write to log file
        if hasattr(self, "output_edit"):
            self.output_edit.append(text.rstrip("\n"))
        if hasattr(self, "log_file"):
            self.log_file.write(text)
            self.log_file.flush()

    def _stop_indexing(self):
        # Use process group termination so that stop button works even if the command spawns children
        if self.indexing_process and self.indexing_process.poll() is None:
            try:
                # first try to terminate the entire process group
                os.killpg(os.getpgid(self.indexing_process.pid), signal.SIGTERM)
                # give it a short grace period
                self.indexing_process.wait(timeout=2)
            except Exception:
                try:
                    self.indexing_process.terminate()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to stop indexing:\n{e}")
            finally:
                self.indexing_process = None
        # Close log file if open
        if hasattr(self, "log_file"):
            try:
                self.log_file.close()
            except:
                pass