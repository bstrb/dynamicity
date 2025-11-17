from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QTabWidget, QComboBox, QTextEdit,
    QTreeWidget, QTreeWidgetItem
)
import subprocess
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
import os
import signal

class IndexingControlWindow(QWidget):

    ini_selection_changed = pyqtSignal(str)  # absolute INI path selected in this window

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
        # Try to locate the workspace file path from the main window
        self.workspace_path = (
            getattr(main_window, 'workspace_file_path', None)
            or getattr(main_window, 'workspace_path', None)
            or getattr(main_window, 'workspacefile', None)
        )
        from configparser import ConfigParser
        import os

        self.setWindowTitle("Indexing & Integration")

        # Read ini file and paths
        parser = ConfigParser()
        parser.read(self.ini_path)
        base_dir = os.path.dirname(self.ini_path)
        import glob
        run_base = self._resolve_run_root(self.ini_path)
        prev_runs = sorted(glob.glob(os.path.join(run_base, "indexingintegration_*")))

        # Do not create a run on open; runs are created explicitly via buttons
        self.run_dir = None
        self.cell_filepath = None
        self.geom_filepath = None
        self.list_filepath = None
        self.stream_filepath = None

        # Initialize tab widget (right side)
        self.tabs = QTabWidget()
        self._init_index_integrate_tab()
        self._init_cellfile_tab()
        self._init_geomfile_tab()
        self._init_start_indexing_tab()

        # Left panel: workspace tree and run creation buttons
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 8, 0)

        left_v.addWidget(QLabel("Workspace Files / Runs"))
        self.runs_tree = QTreeWidget()
        self.runs_tree.setHeaderLabels(["Name"])  # single column
        self.runs_tree.setColumnCount(1)
        left_v.addWidget(self.runs_tree, 1)

        btn_row = QHBoxLayout()
        self.new_run_btn = QPushButton("New Run for Current File")
        self.new_run_btn.clicked.connect(self._create_new_run_for_current_file)
        btn_row.addWidget(self.new_run_btn)
        self.new_batch_run_btn = QPushButton("New Batch Run")
        self.new_batch_run_btn.clicked.connect(self._create_new_batch_run)
        btn_row.addWidget(self.new_batch_run_btn)
        left_v.addLayout(btn_row)

        # Outer layout: left tree + right tabs
        outer = QHBoxLayout()
        outer.addWidget(left_panel, 1)
        outer.addWidget(self.tabs, 3)
        self.setLayout(outer)

        # Populate workspace tree
        self._refresh_runs_tree()
        self._select_tree_for_ini(self.ini_path)

        # Connect tree selection change to handler
        self.runs_tree.currentItemChanged.connect(self._on_runs_tree_selection_changed)

        # Bidirectional selection sync: poll main window for selection changes
        self._last_seen_main_ini = self.ini_path
        self.sync_timer = QTimer(self)
        self.sync_timer.setInterval(700)
        self.sync_timer.timeout.connect(self._sync_from_mainwindow_selection)
        self.sync_timer.start()

        # Load initial data for cell and geom tabs
        if getattr(self, 'prev_cell_file', None) and os.path.exists(self.prev_cell_file):
            # Prefer the latest existing run's cellfile over workspace defaults
            self._load_cell_file()
        else:
            loaded_from_ws = self._load_cell_from_workspace()
            if not loaded_from_ws:
                self._load_cell_file()
        self._load_geom_file()

        # On startup, try to load Index & Integrate settings from the most recent run that actually has them
        try:
            import glob, os as _os
            base_dir = os.path.dirname(self.ini_path)
            run_base = self._resolve_run_root(self.ini_path)
            prev_runs = sorted(
                set(glob.glob(os.path.join(run_base, "indexingintegration_*"))) |
                set(glob.glob(os.path.join(base_dir, "indexingintegration_*")))
            )
            latest_settings_run = None
            for rd in reversed(prev_runs):
                if os.path.exists(os.path.join(rd, 'index_settings.json')):
                    latest_settings_run = rd
                    break
            if latest_settings_run:
                self._load_index_settings_from_run(latest_settings_run)
        except Exception:
            pass

        # Update command label after all tabs/widgets are initialized
        self._update_command_label()

        # Placeholder for indexing subprocess
        self.indexing_process = None
        # Session-level cache of last applied/saved Index & Integrate settings
        self._last_index_settings = None

        # Batch run state
        self._batch_mode = False
        self._batch_queue = []  # list of run_dir paths to process sequentially
        self._proc_poll_timer = QTimer(self)
        self._proc_poll_timer.setInterval(500)
        self._proc_poll_timer.timeout.connect(self._check_process_done)


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
        # Ensure field is editable and accepts floating point numbers (up to 6 decimals)
        self.push_res_edit.setReadOnly(False)
        dv = QDoubleValidator(0.0, 1e12, 6, self)
        dv.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.push_res_edit.setValidator(dv)
        self.push_res_edit.setPlaceholderText("0.5")
        self.push_res_edit.setToolTip("Integrate beyond apparent resolution of each pattern (1/nm)")
        form_layout.addRow("Push resolution:", self.push_res_edit)
        self.int_radius_edit = QLineEdit("4,5,8")
        self.int_radius_edit.setPlaceholderText("e.g. 4,5,8")
        self.int_radius_edit.setToolTip("Outer radii for inner, middle and outer intergration rings (px)")
        form_layout.addRow("Integration radius:", self.int_radius_edit)
        self.min_peaks_spin = QSpinBox()
        self.min_peaks_spin.setValue(25)
        form_layout.addRow("Min peaks:", self.min_peaks_spin)
        self.tolerance_edit = QLineEdit("5,5,5,1.5")
        self.tolerance_edit.setToolTip("Tolerance for unit cell comparison (%), default 5,5,5,1.5")
        form_layout.addRow("Tolerance:", self.tolerance_edit)
        # Insert new XGandalf tolerance field after tolerance
        self.xgandalf_tolerance_edit = QLineEdit("0.02")
        self.xgandalf_tolerance_edit.setToolTip(
            "Relative tolerance of lattice vectors, default 0.02 (2%)"
        )
        form_layout.addRow("XGandalf tolerance:", self.xgandalf_tolerance_edit)

        self.sampling_pitch_spin = QSpinBox()
        self.sampling_pitch_spin.setValue(5)
        self.sampling_pitch_spin.setToolTip(
            "XGANDALF sampling density of reciprocal space, 0-7 (7 is dense)"
        )
        form_layout.addRow("Sampling pitch:", self.sampling_pitch_spin)

        self.min_lat_vec_len_spin = QSpinBox()
        self.min_lat_vec_len_spin.setRange(0, 10000)
        self.min_lat_vec_len_spin.setValue(3)
        self.min_lat_vec_len_spin.setToolTip(
            "Minimum possible lattice vector length (Å), used when no prior cell is provided."
        )
        form_layout.addRow("Min lattice vector length:", self.min_lat_vec_len_spin)

        self.max_lat_vec_len_spin = QSpinBox()
        self.max_lat_vec_len_spin.setRange(0, 10000)
        self.max_lat_vec_len_spin.setValue(30)
        self.max_lat_vec_len_spin.setToolTip(
            "Maximum possible lattice vector length (Å), used when no prior cell is provided."
        )
        form_layout.addRow("Max lattice vector length:", self.max_lat_vec_len_spin)

        self.grad_desc_iterations_spin = QSpinBox()
        self.grad_desc_iterations_spin.setValue(2)
        self.grad_desc_iterations_spin.setToolTip(
            "Number of gradient-descent iterations for XGANDALF 0–5 (5 is many)"
        )
        form_layout.addRow("Gradient descent iterations:", self.grad_desc_iterations_spin)

        self.fix_profile_radius_edit = QLineEdit("50000000")
        form_layout.addRow("Fix profile radius:", self.fix_profile_radius_edit)
        # Connect to update command label
        self.fix_profile_radius_edit.textChanged.connect(self._update_command_label)

        # Add checkboxes for optional flags
        self.no_non_hits_cb = QCheckBox("No non-hits in stream")
        self.no_non_hits_cb.setChecked(True)
        self.no_non_hits_cb.setToolTip("Do not include non-indexed patterns in the output stream. Keeps output smaller.")
        form_layout.addRow(self.no_non_hits_cb)
        self.no_non_hits_cb.stateChanged.connect(self._update_command_label)

        self.no_revalidate_cb = QCheckBox("No revalidate")
        self.no_revalidate_cb.setChecked(True)
        self.no_revalidate_cb.setToolTip("Skip filtering peaks too close to detector edge, doublepeaks or are saturated.")
        form_layout.addRow(self.no_revalidate_cb)
        self.no_revalidate_cb.stateChanged.connect(self._update_command_label)

        self.no_half_pixel_shift_cb = QCheckBox("No half-pixel shift")
        self.no_half_pixel_shift_cb.setChecked(True)
        self.no_half_pixel_shift_cb.setToolTip("Disable half-pixel offset when reading images.")
        form_layout.addRow(self.no_half_pixel_shift_cb)
        self.no_half_pixel_shift_cb.stateChanged.connect(self._update_command_label)

        self.no_retry_cb = QCheckBox("No retry")
        self.no_retry_cb.setChecked(True)
        self.no_retry_cb.setToolTip("Skip retrying with 10% of weakest reflections removed.")
        form_layout.addRow(self.no_retry_cb)
        self.no_retry_cb.stateChanged.connect(self._update_command_label)

        self.no_check_cell_cb = QCheckBox("No check cell")
        self.no_check_cell_cb.setChecked(True)
        self.no_check_cell_cb.setToolTip("Don't check cell parmameters against reference cell.")
        form_layout.addRow(self.no_check_cell_cb)
        self.no_check_cell_cb.stateChanged.connect(self._update_command_label)

        self.no_refine_cb = QCheckBox("No refine")
        self.no_refine_cb.setChecked(True)
        self.no_refine_cb.setToolTip("Don't refine the unit cell after indexing.")
        form_layout.addRow(self.no_refine_cb)
        self.no_refine_cb.stateChanged.connect(self._update_command_label)

        self.save_index_settings_btn = QPushButton("Save Settings")
        self.save_index_settings_btn.clicked.connect(self._save_index_settings)

        layout.addLayout(form_layout)
        layout.addWidget(self.save_index_settings_btn)
        self.index_tab.setLayout(layout)
        self.tabs.addTab(self.index_tab, "Settings")

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


    def _snapshot_index_settings(self):
        """Collect current Index & Integrate settings into a serializable dict."""
        return {
            'threads': self.threads_spin.value(),
            'push_res': self.push_res_edit.text().strip(),
            'int_radius': self.int_radius_edit.text().strip(),
            'min_peaks': self.min_peaks_spin.value(),
            'tolerance': self.tolerance_edit.text().strip(),
            'xg_tolerance': self.xgandalf_tolerance_edit.text().strip(),
            'sampling_pitch': self.sampling_pitch_spin.value(),
            'min_lat_vec_len': int(self.min_lat_vec_len_spin.value()),
            'max_lat_vec_len': int(self.max_lat_vec_len_spin.value()),
            'grad_desc_iterations': self.grad_desc_iterations_spin.value(),
            'fix_profile_radius': self.fix_profile_radius_edit.text().strip(),
            'no_non_hits': bool(self.no_non_hits_cb.isChecked()),
            'no_revalidate': bool(self.no_revalidate_cb.isChecked()),
            'no_half_pixel_shift': bool(self.no_half_pixel_shift_cb.isChecked()),
            'no_retry': bool(self.no_retry_cb.isChecked()),
            'no_check_cell': bool(self.no_check_cell_cb.isChecked()),
            'no_refine': bool(self.no_refine_cb.isChecked()),
        }

    def _apply_index_settings(self, cfg):
        """Apply a previously saved settings dict to UI widgets (signals suppressed)."""
        # Block signals during bulk apply to avoid redundant command rebuilds
        self.threads_spin.blockSignals(True)
        self.push_res_edit.blockSignals(True)
        self.int_radius_edit.blockSignals(True)
        self.min_peaks_spin.blockSignals(True)
        self.tolerance_edit.blockSignals(True)
        self.xgandalf_tolerance_edit.blockSignals(True)
        self.sampling_pitch_spin.blockSignals(True)
        self.min_lat_vec_len_spin.blockSignals(True)
        self.max_lat_vec_len_spin.blockSignals(True)
        self.grad_desc_iterations_spin.blockSignals(True)
        self.fix_profile_radius_edit.blockSignals(True)
        self.no_non_hits_cb.blockSignals(True)
        self.no_revalidate_cb.blockSignals(True)
        self.no_half_pixel_shift_cb.blockSignals(True)
        self.no_retry_cb.blockSignals(True)
        self.no_check_cell_cb.blockSignals(True)
        self.no_refine_cb.blockSignals(True)
        try:
            if 'threads' in cfg: self.threads_spin.setValue(int(cfg['threads']))
            if 'push_res' in cfg: self.push_res_edit.setText(str(cfg['push_res']))
            if 'int_radius' in cfg: self.int_radius_edit.setText(str(cfg['int_radius']))
            if 'min_peaks' in cfg: self.min_peaks_spin.setValue(int(cfg['min_peaks']))
            if 'tolerance' in cfg: self.tolerance_edit.setText(str(cfg['tolerance']))
            if 'xg_tolerance' in cfg: self.xgandalf_tolerance_edit.setText(str(cfg['xg_tolerance']))
            if 'sampling_pitch' in cfg: self.sampling_pitch_spin.setValue(int(cfg['sampling_pitch']))
            if 'min_lat_vec_len' in cfg: self.min_lat_vec_len_spin.setValue(int(cfg['min_lat_vec_len']))
            if 'max_lat_vec_len' in cfg: self.max_lat_vec_len_spin.setValue(int(cfg['max_lat_vec_len']))
            if 'grad_desc_iterations' in cfg: self.grad_desc_iterations_spin.setValue(int(cfg['grad_desc_iterations']))
            if 'fix_profile_radius' in cfg: self.fix_profile_radius_edit.setText(str(cfg['fix_profile_radius']))
            if 'no_non_hits' in cfg: self.no_non_hits_cb.setChecked(bool(cfg['no_non_hits']))
            if 'no_revalidate' in cfg: self.no_revalidate_cb.setChecked(bool(cfg['no_revalidate']))
            if 'no_half_pixel_shift' in cfg: self.no_half_pixel_shift_cb.setChecked(bool(cfg['no_half_pixel_shift']))
            if 'no_retry' in cfg: self.no_retry_cb.setChecked(bool(cfg['no_retry']))
            if 'no_check_cell' in cfg: self.no_check_cell_cb.setChecked(bool(cfg['no_check_cell']))
            if 'no_refine' in cfg: self.no_refine_cb.setChecked(bool(cfg['no_refine']))
        finally:
            # Unblock signals
            self.threads_spin.blockSignals(False)
            self.push_res_edit.blockSignals(False)
            self.int_radius_edit.blockSignals(False)
            self.min_peaks_spin.blockSignals(False)
            self.tolerance_edit.blockSignals(False)
            self.xgandalf_tolerance_edit.blockSignals(False)
            self.sampling_pitch_spin.blockSignals(False)
            self.min_lat_vec_len_spin.blockSignals(False)
            self.max_lat_vec_len_spin.blockSignals(False)
            self.grad_desc_iterations_spin.blockSignals(False)
            self.fix_profile_radius_edit.blockSignals(False)
            self.no_non_hits_cb.blockSignals(False)
            self.no_revalidate_cb.blockSignals(False)
            self.no_half_pixel_shift_cb.blockSignals(False)
            self.no_retry_cb.blockSignals(False)
            self.no_check_cell_cb.blockSignals(False)
            self.no_refine_cb.blockSignals(False)
        self._last_index_settings = cfg
        self._update_command_label()

    def _run_group_key(self):
        """Return the run folder name (e.g., 'indexingintegration_YYYYMMDD_HHMMSS') if current run_dir is set."""
        if not getattr(self, 'run_dir', None):
            return None
        return os.path.basename(self.run_dir.rstrip(os.sep))
    
    def _create_fresh_run_for_current_ini(self, group: str | None = None, seed_input_lst: bool = True) -> str | None:
        """
        Create a brand-new timestamped run directory for the CURRENT INI,
        apply the current GUI settings to that run (cell/geom/settings),
        and point the UI to it. Returns the run_dir or None on failure.
        """
        import os, json
        from datetime import datetime as _dt

        ini = getattr(self, 'ini_path', None)
        if not ini or not os.path.exists(ini):
            return None

        if not group:
            group = f"indexingintegration_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
        run_base = self._resolve_run_root(ini)
        rd = os.path.join(run_base, group)

        try:
            os.makedirs(rd, exist_ok=True)
        except Exception:
            return None

        # Point UI to this run (sets paths like cell/geom/list/stream)
        self._apply_run_dir(rd)

        # Persist current GUI settings (no broadcast)
        cfg = self._snapshot_index_settings()
        try:
            with open(os.path.join(rd, 'index_settings.json'), 'w') as jf:
                json.dump(cfg, jf, indent=2)
        except Exception:
            pass

        # Seed input.lst from the current INI’s detected H5/CXI/EMD if possible
        if seed_input_lst:
            try:
                lst = os.path.join(rd, 'input.lst')
                if not os.path.exists(lst):
                    h5 = self._find_h5_path_from_ini(ini)
                    if h5:
                        with open(lst, 'w') as f:
                            f.write(os.path.abspath(h5))
            except Exception:
                pass

        # Ensure files reflect current UI values
        try: self._save_cell_file()
        except Exception: pass
        try: self._save_geom_file()
        except Exception: pass

        self._update_command_label()
        return rd

    def _broadcast_index_settings(self, cfg):
        try:
            import json, os, glob
            group = self._run_group_key()
            if not group:
                return
            ini_paths = self._get_workspace_ini_paths()
            for ini in ini_paths:
                try:
                    base_dir = os.path.dirname(ini)
                    run_base = self._resolve_run_root(ini)
                    candidates = sorted(
                        set(glob.glob(os.path.join(run_base, group))) |
                        set(glob.glob(os.path.join(base_dir, group)))
                    )
                    for candidate in candidates:
                        if os.path.isdir(candidate):
                            with open(os.path.join(candidate, 'index_settings.json'), 'w') as jf:
                                json.dump(cfg, jf, indent=2)
                except Exception:
                    continue
        except Exception:
            pass

    def _load_index_settings_from_run(self, run_dir=None):
        """Load settings JSON from a run dir (if present) and apply to the UI. Returns True if applied."""
        import os, json
        rd = run_dir or getattr(self, 'run_dir', None)
        if not rd:
            return False
        js = os.path.join(rd, 'index_settings.json')
        if not os.path.exists(js):
            return False
        try:
            with open(js, 'r') as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                self._apply_index_settings(cfg)
                # Remember for sessions where another file has no JSON yet
                self._last_index_settings = cfg
                return True
        except Exception:
            return False
        return False

    def _save_index_settings(self):
        from coseda.initialize import write_xgandalfsettings
        try:
            # Ensure a run exists to anchor settings/paths
            if not getattr(self, 'run_dir', None) or not os.path.isdir(self.run_dir):
                self._create_new_run_for_current_file()
                if not getattr(self, 'run_dir', None) or not os.path.isdir(self.run_dir):
                    raise RuntimeError('Failed to create/select a run directory for saving settings.')
            from configparser import ConfigParser

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
            # Snapshot and broadcast settings to runs with the same timestamped name (batch group)
            cfg = self._snapshot_index_settings()
            self._last_index_settings = cfg
            self._broadcast_index_settings(cfg)
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

        # Workspace integration buttons
        buttons_row = QHBoxLayout()
        self.save_cell_to_workspace_btn = QPushButton("Save Cell to Workspace")
        self.save_cell_to_workspace_btn.clicked.connect(self._save_cell_to_workspace)
        buttons_row.addWidget(self.save_cell_to_workspace_btn)

        self.load_cell_from_workspace_btn = QPushButton("Load from Workspace")
        self.load_cell_from_workspace_btn.clicked.connect(self._load_cell_from_workspace_and_notify)
        buttons_row.addWidget(self.load_cell_from_workspace_btn)

        layout.addLayout(buttons_row)

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
    def _parse_workspace_kv(self):
        """Parse key=value pairs from the workspace file header (commented lines starting with '#')."""
        import os
        kv = {}
        path = getattr(self, 'workspace_path', None)
        if not path or not isinstance(path, str) or not os.path.exists(path):
            return kv
        try:
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s.startswith('#'):
                        # stop at first non-comment – the rest are INI paths
                        # but allow commented settings anywhere; do not break early
                        pass
                    if s.startswith('#') and '=' in s:
                        try:
                            k, v = s[1:].split('=', 1)
                            kv[k.strip()] = v.strip()
                        except Exception:
                            continue
        except Exception:
            return {}
        return kv

    def _load_cell_from_workspace(self) -> bool:
        """Prefill cell tab fields from workspace settings if available. Returns True if applied."""
        kv = self._parse_workspace_kv()
        if not kv:
            return False
        try:
            # Lattice/meta
            lt = kv.get('Cell.lattice_type') or kv.get('lattice_type')
            if lt:
                self.lattice_type_combo.setCurrentText(lt)
            cent = kv.get('Cell.centering') or kv.get('centering')
            if cent:
                self.centering_combo.setCurrentText(cent)
            ua = kv.get('Cell.unique_axis') or kv.get('unique_axis')
            if ua and self.unique_axis_combo.isEnabled():
                self.unique_axis_combo.setCurrentText(ua)

            # Parameters
            def _set(edit, key):
                val = kv.get(f'Cell.{key}') or kv.get(key)
                if val:
                    edit.setText(val.split()[0])
            _set(self.a_edit, 'a')
            _set(self.b_edit, 'b')
            _set(self.c_edit, 'c')
            _set(self.al_edit, 'al')
            _set(self.be_edit, 'be')
            _set(self.ga_edit, 'ga')
            return True
        except Exception:
            return False

    def _load_cell_from_workspace_and_notify(self):
        applied = self._load_cell_from_workspace()
        if applied:
            QMessageBox.information(self, "Workspace", "Cell settings loaded from workspace.")
            self._update_command_label()
        else:
            QMessageBox.warning(self, "Workspace", "No cell settings found in workspace or workspace not available.")

    def _save_cell_to_workspace(self):
        """Write current cell fields into the workspace file header as commented key=value lines."""
        import os
        path = getattr(self, 'workspace_path', None)
        if not path or not isinstance(path, str):
            QMessageBox.critical(self, "Error", "Workspace file path not available.")
            return
        # Gather and validate
        lt = self.lattice_type_combo.currentText().strip()
        cent = self.centering_combo.currentText().strip()
        ua = self.unique_axis_combo.currentText().strip() if self.unique_axis_combo.isEnabled() else ''
        a = self.a_edit.text().strip()
        b = self.b_edit.text().strip()
        c = self.c_edit.text().strip()
        al = self.al_edit.text().strip()
        be = self.be_edit.text().strip()
        ga = self.ga_edit.text().strip()
        required = [lt, cent, a, b, c, al, be, ga]
        if not all(required):
            QMessageBox.critical(self, "Error", "Fill all required cell fields before saving to workspace.")
            return
        try:
            float(a); float(b); float(c); float(al); float(be); float(ga)
        except ValueError:
            QMessageBox.critical(self, "Error", "Non-numeric value in cell parameters.")
            return
        # Read existing lines
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    lines = f.readlines()
            else:
                lines = []
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read workspace file: {e}")
            return
        # Build/replace header key-values
        header_keys = {
            'Cell.lattice_type': lt,
            'Cell.centering': cent,
            # unique_axis only when applicable
        }
        if ua:
            header_keys['Cell.unique_axis'] = ua
        header_keys.update({
            'Cell.a': a,
            'Cell.b': b,
            'Cell.c': c,
            'Cell.al': al,
            'Cell.be': be,
            'Cell.ga': ga,
        })
        # Create a dict of existing commented KV indices to update in place
        kv_indices = {}
        for idx, line in enumerate(lines):
            s = line.strip()
            if s.startswith('#') and '=' in s:
                try:
                    k, _ = s[1:].split('=', 1)
                    k = k.strip()
                    if k in header_keys and k not in kv_indices:
                        kv_indices[k] = idx
                except Exception:
                    continue
        # If file has no lines yet, start with an empty header list
        if not lines:
            lines = []
        # Ensure there is an empty header block at the top: insert/update keys at the beginning before any non-comment line
        # Find the insertion point: before the first non-comment, non-empty line
        insert_at = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                insert_at = i
                break
            insert_at = i + 1
        # Update existing keys
        for k, v in header_keys.items():
            new_line = f"# {k}={v}\n"
            if k in kv_indices:
                lines[kv_indices[k]] = new_line
            else:
                lines.insert(insert_at, new_line)
                insert_at += 1
        # Write back
        try:
            with open(path, 'w') as f:
                f.writelines(lines)
            QMessageBox.information(self, "Workspace", "Cell settings saved to workspace.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write workspace file: {e}")

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

    def _ensure_run_selected_or_latest(self) -> bool:
        if getattr(self, 'run_dir', None) and os.path.isdir(self.run_dir):
            return True
        # Try to find latest run for current INI without relying on INI Paths
        if not self.ini_path or not os.path.exists(self.ini_path):
            return False
        import glob, os as _os
        run_base = self._resolve_run_root(self.ini_path)
        # Also look directly under the INI folder as a fallback
        base_dir = _os.path.dirname(self.ini_path)
        prev_runs = sorted(
            set(glob.glob(_os.path.join(run_base, 'indexingintegration_*'))) |
            set(glob.glob(_os.path.join(base_dir, 'indexingintegration_*')))
        )
        latest_run = prev_runs[-1] if prev_runs else None
        if latest_run and os.path.isdir(latest_run):
            self._apply_run_dir(latest_run)
            return True
        return False

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
            # Try to select the latest run automatically
            if not self._ensure_run_selected_or_latest():
                # No run available; silently abort
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

    def _strip_units_for_ui(self, key: str, val: str) -> str:
        """Return a cleaned value for UI fields by stripping unit suffixes from known keys.
        We keep numbers bare in the QLineEdits and only add units when writing the .geom file."""
        import re
        if not isinstance(val, str):
            return val
        s = val.strip()
        # Strip surrounding quotes if any
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        if key == 'wavelength':
            # Remove a single trailing Å/A token (with optional whitespace)
            s = re.sub(r"\s*(Å|A)\s*$", "", s)
        elif key == 'clen':
            # Remove a single trailing 'm'
            s = re.sub(r"\s*m\s*$", "", s)
        return s

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
                    self.geom_edits[key].setText(self._strip_units_for_ui(key, val))
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
            # Try to select the latest run automatically
            if not self._ensure_run_selected_or_latest():
                # No run available; silently abort
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

        self.list_path_edit = QLineEdit(self.list_filepath or "")
        self.list_path_edit.setReadOnly(True)
        form_layout.addRow("List file:", self.list_path_edit)

        self.stream_path_edit = QLineEdit(self.stream_filepath or "")
        self.stream_path_edit.setReadOnly(True)
        form_layout.addRow("Output stream:", self.stream_path_edit)

        # Command edit (read-only single-line QLineEdit)
        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        form_layout.addRow("Command:", self.command_edit)

        # Add Start (current), Start All (batch) and Stop buttons
        self.start_current_button = QPushButton("Start Current")
        self.start_current_button.clicked.connect(self._start_indexing)
        form_layout.addRow(self.start_current_button)

        self.start_all_button = QPushButton("Start All in Batch")
        self.start_all_button.clicked.connect(self._start_batch_indexing)
        form_layout.addRow(self.start_all_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_indexing)
        form_layout.addRow(self.stop_button)

        # Status label to indicate which run is currently executing (for batch)
        self.batch_status_label = QLabel("")
        form_layout.addRow("Running:", self.batch_status_label)

        # Add output_edit QTextEdit for logs/output
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        layout.addLayout(form_layout)
        layout.addWidget(self.output_edit)

        self.start_tab.setLayout(layout)
        self.tabs.addTab(self.start_tab, "Start Indexing")

    def _start_indexing_for_run(self, run_dir: str) -> bool:
        """Prepare UI/paths for the given run_dir and start indexing. Returns True if started."""
        if not run_dir or not os.path.isdir(run_dir):
            return False
        # Point Start tab to this run and rebuild command
        self._apply_run_dir(run_dir)
        # Ensure command is up to date
        self._update_command_label()
        # Start
        self._start_indexing()
        return True

    def _create_new_batch_run(self):
        """Create a new timestamped run folder with the SAME name for every INI in the workspace.
        Also switch the UI to the current INI's run folder.
        """
        from datetime import datetime as _dt
        group = f"indexingintegration_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
        made = 0
        cur_ini = getattr(self, 'ini_path', None)

        for ini in self._get_workspace_ini_paths():
            try:
                base = self._resolve_run_root(ini)
                rd = os.path.join(base, group)
                if not os.path.isdir(rd):
                    os.makedirs(rd, exist_ok=True)
                    made += 1
                # For convenience, pre-touch placeholder files so later logic can write into them
                try:
                    cf = os.path.join(rd, 'cellfile.cell')
                    gf = os.path.join(rd, 'geometry.geom')
                    # create empty if missing; actual contents will be written when applied/started
                    if not os.path.exists(cf):
                        open(cf, 'a').close()
                    if not os.path.exists(gf):
                        open(gf, 'a').close()
                except Exception:
                    pass
            except Exception:
                continue

        # Point the UI to the current INI's new run, if possible
        if cur_ini and os.path.exists(cur_ini):
            rd_cur = os.path.join(self._resolve_run_root(cur_ini), group)
            if os.path.isdir(rd_cur):
                self._apply_run_dir(rd_cur)

        # Refresh the left tree so the user can see all created runs
        self._refresh_runs_tree()
        # Re-select the current INI in the tree (non-fatal if it fails)
        try:
            self._select_tree_for_ini(cur_ini)
        except Exception:
            pass

        # Status message
        QMessageBox.information(self, "Batch", f"Created batch run '{group}' for {made} file(s).")

    def _start_batch_indexing(self):
        """
        Queue and run all INIs in the workspace with a NEW timestamped run name.
        For each INI, create the run, seed input.lst if possible, and write the
        current GUI settings into index_settings.json. Then execute sequentially.
        """
        from datetime import datetime as _dt
        import os, json

        # Always create a NEW group name each time the button is pressed
        group = f"indexingintegration_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
        cfg = self._snapshot_index_settings()

        self._batch_queue = []
        created_dirs = 0
        created_lists = 0
        ini_paths = self._get_workspace_ini_paths()

        # Prepare runs for all INIs
        for ini in ini_paths:
            try:
                run_base = self._resolve_run_root(ini)
                rd = os.path.join(run_base, group)
                os.makedirs(rd, exist_ok=True)
                created_dirs += 1

                # Write current settings into each run
                try:
                    with open(os.path.join(rd, 'index_settings.json'), 'w') as jf:
                        json.dump(cfg, jf, indent=2)
                except Exception:
                    pass

                # Seed input.lst if possible
                try:
                    lst_path = os.path.join(rd, 'input.lst')
                    if not os.path.exists(lst_path):
                        h5 = self._find_h5_path_from_ini(ini)
                        if h5:
                            with open(lst_path, 'w') as f:
                                f.write(os.path.abspath(h5))
                            created_lists += 1
                except Exception:
                    pass

                # Queue this run
                self._batch_queue.append(rd)
            except Exception:
                continue

        # Switch UI to the current INI’s fresh run for visibility
        cur_ini = getattr(self, 'ini_path', None)
        if cur_ini and os.path.exists(cur_ini):
            rd_cur = os.path.join(self._resolve_run_root(cur_ini), group)
            if os.path.isdir(rd_cur):
                self._apply_run_dir(rd_cur)

        # Refresh tree so user sees the new runs
        try:
            self._refresh_runs_tree()
            try:
                self._select_tree_for_ini(getattr(self, 'ini_path', None))
            except Exception:
                pass
        except Exception:
            pass

        if not self._batch_queue:
            QMessageBox.information(self, "Batch Run", "No runs could be prepared for batch execution.")
            return

        info_bits = [f"created {created_dirs} run folder(s)"]
        if created_lists:
            info_bits.append(f"prepared {created_lists} input.lst file(s)")
        self.batch_status_label.setText("; ".join(info_bits))

        # Kick off the first; the rest chain via _check_process_done
        self._batch_mode = True
        self._start_next_in_batch()

    def _start_next_in_batch(self):
        if not self._batch_queue:
            self.batch_status_label.setText("")
            self._batch_mode = False
            QMessageBox.information(self, "Batch Run", "Batch finished.")
            return
        rd = self._batch_queue.pop(0)
        # Update status and highlight in tree
        self.batch_status_label.setText(rd)
        self._select_tree_item_for_run(rd)
        self._start_indexing_for_run(rd)
        # Begin polling for completion
        self._proc_poll_timer.start()

    def _check_process_done(self):
        # Called periodically while a process is running
        if self.indexing_process is None:
            # Nothing running; if we are in batch mode, schedule next
            if self._batch_mode:
                self._proc_poll_timer.stop()
                self._start_next_in_batch()
            return
        try:
            rc = self.indexing_process.poll()
            if rc is not None:
                # Process finished
                self._proc_poll_timer.stop()
                self.indexing_process = None
                if self._batch_mode:
                    self._start_next_in_batch()
        except Exception:
            # On error, try to progress the batch
            self._proc_poll_timer.stop()
            self.indexing_process = None
            if self._batch_mode:
                self._start_next_in_batch()

    def _select_tree_item_for_run(self, run_dir: str):
        try:
            top_count = self.runs_tree.topLevelItemCount()
            for i in range(top_count):
                top = self.runs_tree.topLevelItem(i)
                if not top:
                    continue
                for j in range(top.childCount()):
                    child = top.child(j)
                    if child and child.toolTip(0) == run_dir:
                        self.runs_tree.setCurrentItem(child)
                        idx = self.runs_tree.indexFromItem(child)
                        if idx.isValid():
                            self.runs_tree.scrollTo(idx)
                        return
        except Exception:
            pass

    def _update_command_label(self):
        """(Re)build the indexamajig command preview.
        If no run is currently selected, try to select the latest run for the
        current INI. If path fields are still empty, prefill them based on the
        run folder so the user always sees a concrete command string.
        """
        import os

        # 1) Ensure we point at some run (latest for current INI if none selected)
        if not getattr(self, 'run_dir', None) or not isinstance(self.run_dir, str) or not os.path.isdir(self.run_dir):
            self._ensure_run_selected_or_latest()

        # 2) Prefill missing path fields from the run_dir so we can always show a command
        rd = getattr(self, 'run_dir', None)
        if rd and os.path.isdir(rd):
            if not getattr(self, 'cell_filepath', None):
                self.cell_filepath = os.path.join(rd, 'cellfile.cell')
            if not getattr(self, 'geom_filepath', None):
                self.geom_filepath = os.path.join(rd, 'geometry.geom')
            if not getattr(self, 'list_filepath', None):
                self.list_filepath = os.path.join(rd, 'input.lst')
            if not getattr(self, 'stream_filepath', None):
                run_folder_name = os.path.basename(rd)
                self.stream_filepath = os.path.join(rd, f'{run_folder_name}.stream')
            # Reflect into Start tab read-only edits if present
            if hasattr(self, 'geom_path_edit') and self.geom_path_edit.text().strip() == '' and getattr(self, 'geom_filepath', ''):
                self.geom_path_edit.setText(self.geom_filepath)
            if hasattr(self, 'cell_path_edit') and self.cell_path_edit.text().strip() == '' and getattr(self, 'cell_filepath', ''):
                self.cell_path_edit.setText(self.cell_filepath)
            if hasattr(self, 'list_path_edit') and self.list_path_edit.text().strip() == '' and getattr(self, 'list_filepath', ''):
                self.list_path_edit.setText(self.list_filepath)
            if hasattr(self, 'stream_path_edit') and self.stream_path_edit.text().strip() == '' and getattr(self, 'stream_filepath', ''):
                self.stream_path_edit.setText(self.stream_filepath)

        # 3) Collect all inputs (may still be empty if no INI/run at all)
        geom = getattr(self, 'geom_filepath', '') or ''
        lst = getattr(self, 'list_filepath', '') or ''
        stream = getattr(self, 'stream_filepath', '') or ''
        cell = getattr(self, 'cell_filepath', '') or ''

        # If we *still* have no context, clear and bail out gracefully
        if not any([geom, lst, stream, cell]):
            if hasattr(self, 'command_edit'):
                self.command_edit.setText('')
            return

        # 4) Read current UI parameter values
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

        # 5) Optional flags
        flags = []
        if hasattr(self, 'no_non_hits_cb') and self.no_non_hits_cb.isChecked():
            flags.append('--no-non-hits-in-stream')
        if hasattr(self, 'no_revalidate_cb') and self.no_revalidate_cb.isChecked():
            flags.append('--no-revalidate')
        if hasattr(self, 'no_half_pixel_shift_cb') and self.no_half_pixel_shift_cb.isChecked():
            flags.append('--no-half-pixel-shift')
        if hasattr(self, 'no_retry_cb') and self.no_retry_cb.isChecked():
            flags.append('--no-retry')
        if hasattr(self, 'no_check_cell_cb') and self.no_check_cell_cb.isChecked():
            flags.append('--no-check-cell')
        if hasattr(self, 'no_refine_cb') and self.no_refine_cb.isChecked():
            flags.append('--no-refine')
        flags_str = ' '.join(flags)

        # 6) Build the command string even if files don't exist yet (paths suffice)
        cmd_parts = [
            f"indexamajig -g {geom}",
            f"-i {lst}",
            f"-o {stream}",
            f"-j {threads}",
            f"-p {cell}",
            f"--indexing=xgandalf",
            f"--push-res={push_res}",
            f"--integration=rings",
            flags_str,
            f"--int-radius={int_radius}",
            f"--peaks=cxi",
            f"--max-indexer-threads={threads}",
            f"--min-peaks={min_peaks}",
            f"--xgandalf-tolerance={xg_tol}",
            f"--xgandalf-sampling-pitch={samp_pitch}",
            f"--xgandalf-min-lattice-vector-length={min_lat}",
            f"--xgandalf-max-lattice-vector-length={max_lat}",
            f"--xgandalf-grad-desc-iterations={gd_iters}",
            f"--tolerance={tol}",
        ]
        if fix_profile_radius:
            cmd_parts.append(f"--fix-profile-radius={fix_profile_radius}")
        # Clean up any accidental double spaces from empty flags_str
        cmd = ' '.join(part for part in cmd_parts if part)

        if hasattr(self, 'command_edit'):
            self.command_edit.setText(cmd)
        
    def _start_indexing(self):
        # Always create a FRESH run with current GUI settings, then start
        if self.indexing_process and self.indexing_process.poll() is None:
            return

        rd = self._create_fresh_run_for_current_ini()
        if not rd:
            return

        cmd = self.command_edit.text().strip()
        if not cmd:
            return

        try:
            if hasattr(self, "output_edit"):
                self.output_edit.clear()
            self.log_file = open(os.path.join(self.run_dir, "indexing.log"), "w")
            self.indexing_process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                preexec_fn=os.setsid
            )
            self.process_thread = self.ProcessOutputThread(self.indexing_process)
            self.process_thread.output_received.connect(self._append_output)
            self.process_thread.start()
            self._proc_poll_timer.start()
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
        # If user stops manually, cancel any remaining batch items
        if self._batch_mode:
            self._proc_poll_timer.stop()
            self._batch_queue = []
            self._batch_mode = False
            self.batch_status_label.setText("")
        # Close log file if open
        if hasattr(self, "log_file"):
            try:
                self.log_file.close()
            except:
                pass
    def _get_workspace_ini_paths(self):
        paths = []
        path = getattr(self, 'workspace_path', None)
        if not path or not isinstance(path, str) or not os.path.exists(path):
            return paths
        try:
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    # treat as an INI path line
                    paths.append(s)
        except Exception:
            pass
        return paths
    
    def _resolve_run_root(self, ini_path: str) -> str:
        """
        Decide where indexingintegration_* runs live without relying on INI Paths.outputfolder.
        Preference: <ini_dir>/output, then <ini_dir>/runs, else <ini_dir>.
        """
        import os
        if not ini_path:
            return os.getcwd()
        base_dir = os.path.dirname(os.path.abspath(ini_path))
        for candidate in (os.path.join(base_dir, 'output'),
                        os.path.join(base_dir, 'runs')):
            if os.path.isdir(candidate):
                return candidate
        return base_dir

    def _refresh_runs_tree(self):
        import glob
        # Pause sync timer during rebuild to avoid races
        timer_active = False
        if hasattr(self, 'sync_timer') and self.sync_timer.isActive():
            timer_active = True
            self.sync_timer.stop()
        try:
            self.runs_tree.clear()
            ini_paths = self._get_workspace_ini_paths()
            for ini in ini_paths:
                try:
                    ini_base = os.path.basename(ini)
                    top = QTreeWidgetItem([ini_base])
                    top.setToolTip(0, ini)
                    self.runs_tree.addTopLevelItem(top)

                    # Determine base for runs
                    # Determine base for runs (do not rely on INI Paths.outputfolder)
                    base_dir = os.path.dirname(ini)
                    run_base = self._resolve_run_root(ini)
                    runs = sorted(
                        set(glob.glob(os.path.join(run_base, 'indexingintegration_*'))) |
                        set(glob.glob(os.path.join(base_dir, 'indexingintegration_*')))
                    )
                    for rd in runs:
                        item = QTreeWidgetItem([os.path.basename(rd)])
                        item.setToolTip(0, rd)
                        top.addChild(item)
                except Exception:
                    continue
            self.runs_tree.expandAll()
        finally:
            if timer_active:
                self.sync_timer.start()

    def _on_current_ini_changed(self):
        """Refresh tab settings when the current INI changes (without creating a new run).
        Prefers the latest run's cellfile; falls back to workspace cell values, then empty/new."""
        if not self.ini_path or not os.path.exists(self.ini_path):
            return
        # Recompute context for this INI
        from configparser import ConfigParser
        parser = ConfigParser()
        try:
            parser.read(self.ini_path)
        except Exception:
            return
        base_dir = os.path.dirname(self.ini_path)
        # Detect latest previous run for this INI without relying on INI Paths
        import glob, os as _os
        run_base = self._resolve_run_root(self.ini_path)
        prev_runs = sorted(
            set(glob.glob(os.path.join(run_base, 'indexingintegration_*'))) |
            set(glob.glob(os.path.join(base_dir, 'indexingintegration_*')))
        )
        latest_run = prev_runs[-1] if prev_runs else None
        # Prefer the newest run that actually has index_settings.json for loading UI settings
        latest_settings_run = None
        for rd in reversed(prev_runs):
            try:
                if os.path.exists(os.path.join(rd, 'index_settings.json')):
                    latest_settings_run = rd
                    break
            except Exception:
                continue
        # Update pointers used by loaders
        self.prev_cell_file = None
        if latest_run:
            candidate = os.path.join(latest_run, 'cellfile.cell')
            if os.path.exists(candidate):
                self.prev_cell_file = candidate
        # Prefer latest run's cellfile; fallback to workspace; then to empty/new
        if getattr(self, 'prev_cell_file', None) and os.path.exists(self.prev_cell_file):
            self._load_cell_file()
        else:
            loaded_ws = self._load_cell_from_workspace()
            if not loaded_ws:
                self._load_cell_file()
        # Try to load geometry from latest run if present; else fall back to existing path/INI-derived defaults
        if latest_run:
            latest_geom = os.path.join(latest_run, 'geometry.geom')
            old_geom_path = getattr(self, 'geom_filepath', None)
            if os.path.exists(latest_geom):
                self.geom_filepath = latest_geom
            self._load_geom_file()
            # Keep geom_filepath pointing at latest if it exists; otherwise restore
            if not os.path.exists(latest_geom) and old_geom_path:
                self.geom_filepath = old_geom_path
        else:
            self._load_geom_file()
        # Also refresh h5 path for command building
        self.h5_path = self._find_h5_path_from_ini(self.ini_path) or self.h5_path
        # Load Index & Integrate settings from the most recent run that actually has them
        if latest_settings_run:
            self._load_index_settings_from_run(latest_settings_run)
        elif getattr(self, '_last_index_settings', None):
            # Apply last known good settings as a sensible default for new files without JSON yet
            self._apply_index_settings(self._last_index_settings)
        # Rebuild command preview with whatever paths we have now
        self._update_command_label()
        # Do NOT refresh the runs tree here; it destroys items and breaks selection highlighting


    def _select_tree_for_ini(self, ini_path: str):
        """Highlight the tree node corresponding to ini_path, if present (safe against rebuilds)."""
        if not ini_path:
            return
        try:
            top_count = self.runs_tree.topLevelItemCount()
            for i in range(top_count):
                top = self.runs_tree.topLevelItem(i)
                if top and top.toolTip(0) == ini_path:
                    self.runs_tree.setCurrentItem(top)
                    # Scroll safely using a model index
                    idx = self.runs_tree.indexFromItem(top)
                    if idx.isValid():
                        self.runs_tree.scrollTo(idx)
                    break
        except RuntimeError:
            # The tree was probably rebuilt; ignore and let next timer tick handle it
            return

    def _on_runs_tree_selection_changed(self, current, previous):
        if current is None:
            return
        parent = current.parent()
        if parent is None:
            # Top-level INI selected -> emit to main window
            ini_path = current.toolTip(0)
            if ini_path and os.path.exists(ini_path):
                # Update our state
                self.ini_path = ini_path
                self.h5_path = self._find_h5_path_from_ini(ini_path) or self.h5_path
                # Emit to main window rather than calling its slots directly
                try:
                    self.ini_selection_changed.emit(ini_path)
                except Exception:
                    pass
                # Refresh right-side tabs for the newly selected INI
                self._on_current_ini_changed()
        else:
            # Child node = a run directory selected -> switch to its parent INI locally, emit to main, then apply run
            run_dir = current.toolTip(0)
            if not run_dir or not os.path.isdir(run_dir):
                return
            parent_ini = current.parent().toolTip(0) if current.parent() else None
            if parent_ini and os.path.exists(parent_ini):
                # Sync selection/state to parent INI locally
                self.ini_path = parent_ini
                self.h5_path = self._find_h5_path_from_ini(parent_ini) or self.h5_path
                try:
                    self.ini_selection_changed.emit(parent_ini)
                except Exception:
                    pass
                # Refresh tabs for the now-current INI
                self._on_current_ini_changed()
            # Finally, point Start tab to the selected run directory
            self._apply_run_dir(run_dir)

    def _apply_run_dir(self, run_dir: str):
        """Point Start tab paths to the selected run directory and update command preview."""
        self.run_dir = run_dir
        self.cell_filepath = os.path.join(run_dir, 'cellfile.cell')
        self.geom_filepath = os.path.join(run_dir, 'geometry.geom')
        # If this run has no cell file yet, write one from current UI values
        if not os.path.exists(self.cell_filepath):
            self._save_cell_file()
        # If this run has saved index settings, load and apply them
        self._load_index_settings_from_run(self.run_dir)
        self.list_filepath = os.path.join(run_dir, 'input.lst')
        run_folder_name = os.path.basename(run_dir)
        self.stream_filepath = os.path.join(run_dir, f'{run_folder_name}.stream')
        if hasattr(self, 'geom_path_edit'):
            self.geom_path_edit.setText(self.geom_filepath)
        if hasattr(self, 'cell_path_edit'):
            self.cell_path_edit.setText(self.cell_filepath)
        if hasattr(self, 'list_path_edit'):
            self.list_path_edit.setText(self.list_filepath)
        if hasattr(self, 'stream_path_edit'):
            self.stream_path_edit.setText(self.stream_filepath)
        # Force loader to use this run's cellfile
        self.prev_cell_file = self.cell_filepath if os.path.exists(self.cell_filepath) else None
        # Optionally load geom/cell into editors if files exist
        self._load_cell_file()
        self._load_geom_file()
        # If this run has no geometry file yet, write one from current UI/INI-derived values
        if not os.path.exists(self.geom_filepath):
            self._save_geom_file()
        self._update_command_label()

    def _sync_from_mainwindow_selection(self):
        """Periodically reflect main window's current INI selection into this window/tree."""
        mw = self.main_window
        current_ini = getattr(mw, 'full_ini_file_path', None)
        if not current_ini or current_ini == self._last_seen_main_ini:
            return
        self._last_seen_main_ini = current_ini
        # Update internal references
        self.ini_path = current_ini
        # Try to select it in the tree, guarding against tree rebuilds
        try:
            self._select_tree_for_ini(current_ini)
        except RuntimeError:
            pass
        self._on_current_ini_changed()

    def _find_h5_path_from_ini(self, ini_path):
        import os, glob
        if not ini_path:
            return None
        try:
            ini_path = os.path.abspath(ini_path)
            base_dir = os.path.dirname(ini_path)
            ini_name = os.path.basename(ini_path)
            stem, _ = os.path.splitext(ini_name)
            # Derive prefix: take part before '_run_' if present, else first two tokens if they look like date_time
            prefix = stem
            if '_run_' in stem:
                prefix = stem.split('_run_')[0]
            else:
                parts = stem.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    prefix = f"{parts[0]}_{parts[1]}"
                elif parts:
                    prefix = parts[0]
            # Candidate directories to search
            base_parent = os.path.dirname(base_dir)
            base_grand = os.path.dirname(base_parent)
            candidates = []
            for d in (base_dir, base_parent, base_grand):
                if d and os.path.isdir(d) and d not in candidates:
                    candidates.append(d)
            patterns = [
                f"{prefix}*.h5", f"{prefix}*.hdf5", f"{prefix}*.cxi",
                f"{prefix}.h5",  f"{prefix}.hdf5",  f"{prefix}.cxi",
            ]
            hits = []
            for root in candidates:
                for pat in patterns:
                    hits.extend(glob.glob(os.path.join(root, pat)))
            hits = sorted(h for h in hits if os.path.isfile(h))
            if hits:
                return hits[0]
            # Shallow scan subdirs of parent that contain the prefix in their name
            if os.path.isdir(base_parent):
                for entry in sorted(os.listdir(base_parent)):
                    sub = os.path.join(base_parent, entry)
                    if os.path.isdir(sub) and prefix in entry:
                        for pat in patterns:
                            cand = sorted(glob.glob(os.path.join(sub, pat)))
                            cand = [h for h in cand if os.path.isfile(h)]
                            if cand:
                                return cand[0]
        except Exception as e:
            print(f"DEBUG: naming-scheme H5 resolve failed for {ini_path}: {e}")
            return None
        print(f"DEBUG: No H5/CXI found by naming scheme for INI: {ini_path} (prefix='{prefix}')")
        return None

    def _create_new_run_for_current_file(self):
        # Require a current INI
        if not self.ini_path or not os.path.exists(self.ini_path):
            QMessageBox.warning(self, 'Run', 'No current INI selected in main window.')
            return
        # Compute run base
        # Compute run base (independent of INI Paths)
        base_dir = os.path.dirname(self.ini_path)
        run_base = self._resolve_run_root(self.ini_path)
        # Create new timestamped run dir
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_run_dir = os.path.join(run_base, f'indexingintegration_{timestamp}')
        try:
            os.makedirs(new_run_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, 'Run', f'Failed to create run directory:\n{e}')
            return
        # Update internal paths to point to this new run
        self.run_dir = new_run_dir
        self.cell_filepath = os.path.join(self.run_dir, 'cellfile.cell')
        self.geom_filepath = os.path.join(self.run_dir, 'geometry.geom')
        self.list_filepath = os.path.join(self.run_dir, 'input.lst')
        run_folder_name = os.path.basename(self.run_dir)
        self.stream_filepath = os.path.join(self.run_dir, f'{run_folder_name}.stream')
        # Write list file; if H5 is not known, prompt the user once
        h5 = self.h5_path or self._find_h5_path_from_ini(self.ini_path)
        if not h5:
            try:
                sel, _ = QFileDialog.getOpenFileName(self, 'Select HDF5/ CXI file', os.path.dirname(self.ini_path), 'Data files (*.h5 *.hdf5 *.cxi);;All files (*)')
                if sel:
                    h5 = sel
            except Exception:
                pass
        try:
            with open(self.list_filepath, 'w') as f:
                if h5:
                    f.write(os.path.abspath(h5))
                else:
                    # Create an empty placeholder; user can fill it later
                    f.write('')
        except Exception as e:
            QMessageBox.warning(self, 'Run', f'Created run dir, but failed to write input.lst:\n{e}')
        # Reflect new paths in the Start tab fields
        if hasattr(self, 'geom_path_edit'):
            self.geom_path_edit.setText(self.geom_filepath)
        if hasattr(self, 'cell_path_edit'):
            self.cell_path_edit.setText(self.cell_filepath)
        if hasattr(self, 'list_path_edit'):
            self.list_path_edit.setText(self.list_filepath)
        if hasattr(self, 'stream_path_edit'):
            self.stream_path_edit.setText(self.stream_filepath)
        # Ensure a cell file exists for the new run (uses current UI values)
        if not os.path.exists(self.cell_filepath):
            self._save_cell_file()
        # Ensure a geometry file exists for the new run (uses current UI/INI-derived values)
        if not os.path.exists(self.geom_filepath):
            self._save_geom_file()
        # Update command string & tree
        self._update_command_label()
        self._refresh_runs_tree()

    def _create_new_batch_run(self):
        ini_paths = self._get_workspace_ini_paths()
        if not ini_paths:
            QMessageBox.warning(self, 'Batch Run', 'No INI files found in the workspace file.')
            return
        from datetime import datetime
        from configparser import ConfigParser
        import os

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        created = 0
        errors = 0
        for ini in ini_paths:
            try:
                parser = ConfigParser()
                parser.read(ini)
                base_dir = os.path.dirname(ini)
                output_folder = parser.get('Paths', 'outputfolder', fallback='')
                run_base = os.path.join(base_dir, output_folder) if output_folder else base_dir
                new_run_dir = os.path.join(run_base, f'indexingintegration_{ts}')
                os.makedirs(new_run_dir, exist_ok=True)

                # 1) Write list file using naming-based H5 resolution (placeholder if unresolved)
                lst_path = os.path.join(new_run_dir, 'input.lst')
                h5 = self._find_h5_path_from_ini(ini)
                try:
                    with open(lst_path, 'w') as f:
                        f.write(os.path.abspath(h5) if h5 else '')
                    print(f"DEBUG: wrote input.lst for {new_run_dir}: '{os.path.abspath(h5) if h5 else ''}'")
                except Exception as e:
                    print(f"DEBUG: failed to write input.lst for {new_run_dir}: {e}")
                    errors += 1
                    continue

                # 2) Write a cell file for this run from current UI values, if valid
                try:
                    lt = self.lattice_type_combo.currentText().strip()
                    cent = self.centering_combo.currentText().strip()
                    ua = self.unique_axis_combo.currentText().strip() if self.unique_axis_combo.isEnabled() else ''
                    a = self.a_edit.text().strip(); b = self.b_edit.text().strip(); c = self.c_edit.text().strip()
                    al = self.al_edit.text().strip(); be = self.be_edit.text().strip(); ga = self.ga_edit.text().strip()
                    ok = True
                    for v in (a, b, c, al, be, ga):
                        try:
                            float(v)
                        except Exception:
                            ok = False; break
                    if ok and lt and cent:
                        with open(os.path.join(new_run_dir, 'cellfile.cell'), 'w') as fcell:
                            fcell.write('CrystFEL unit cell file version 1.0\n\n')
                            fcell.write(f'lattice_type = {lt}\n\n')
                            fcell.write(f'centering = {cent}\n')
                            if ua:
                                fcell.write(f'unique_axis = {ua}\n\n')
                            else:
                                fcell.write('\n')
                            fcell.write(f'a = {a} A\n')
                            fcell.write(f'b = {b} A\n')
                            fcell.write(f'c = {c} A\n')
                            fcell.write(f'al = {al} deg\n')
                            fcell.write(f'be = {be} deg\n')
                            fcell.write(f'ga = {ga} deg\n')
                except Exception as e:
                    print(f"DEBUG: failed to write cellfile.cell for {new_run_dir}: {e}")
                    # not fatal

                # 3) Write a geometry file for this run from current UI/INI-derived values
                try:
                    geom_path = os.path.join(new_run_dir, 'geometry.geom')
                    if not os.path.exists(geom_path):
                        # Temporarily point this window's paths to the new run so _save_geom_file writes there
                        prev_run_dir = getattr(self, 'run_dir', None)
                        prev_geom = getattr(self, 'geom_filepath', '')
                        prev_cell = getattr(self, 'cell_filepath', '')
                        prev_list = getattr(self, 'list_filepath', '')
                        prev_stream = getattr(self, 'stream_filepath', '')
                        try:
                            self.run_dir = new_run_dir
                            self.geom_filepath = geom_path
                            self.cell_filepath = os.path.join(new_run_dir, 'cellfile.cell')
                            self.list_filepath = lst_path
                            run_folder_name = os.path.basename(new_run_dir)
                            self.stream_filepath = os.path.join(new_run_dir, f'{run_folder_name}.stream')
                            self._save_geom_file()
                        finally:
                            self.run_dir = prev_run_dir
                            self.geom_filepath = prev_geom
                            self.cell_filepath = prev_cell
                            self.list_filepath = prev_list
                            self.stream_filepath = prev_stream
                except Exception as e:
                    print(f"DEBUG: failed to write geometry.geom for {new_run_dir}: {e}")
                    # not fatal

                created += 1
            except Exception as e:
                print(f"DEBUG: batch run creation failed for {ini}: {e}")
                errors += 1
                continue

        # Refresh the tree so the new runs appear
        self._refresh_runs_tree()
        # Inform the user
        if created and errors:
            QMessageBox.information(self, 'Batch Run', f'Created {created} run(s) with {errors} error(s).')
        elif created:
            QMessageBox.information(self, 'Batch Run', f'Created {created} run(s).')
        else:
            QMessageBox.warning(self, 'Batch Run', 'No runs were created.')