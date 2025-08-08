#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFormLayout, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, 
    QGroupBox, QTextEdit, QTabWidget, QProgressBar, QCheckBox,
    QComboBox, QSlider, QSizePolicy, QStackedLayout, QPlainTextEdit,
    QProgressDialog, QDialog, QSplitter
)
from astropy import units as u
from astropy.constants import h, c
from astropy.io import fits
import pyqtgraph as pg
import json
import traceback
from datetime import datetime
import copy
import time
import tempfile
import shutil
import threading
import uuid

try:
    import pyxel
    PYXEL_AVAILABLE = True
except ImportError:
    PYXEL_AVAILABLE = False
    print("Warning: Pyxel not available. Pyxel simulation tab will be disabled.")

class PyxelSimulationWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, config, image_data, image_filename, simulation_id):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.image_data = np.array(image_data, copy=True)
        self.image_filename = image_filename
        self.simulation_id = simulation_id
        self.scenario_metadata = None
        
        self._is_cancelled = False
        self._mutex = QMutex()

    def cancel(self):
        with QMutexLocker(self._mutex):
            self._is_cancelled = True

    def is_cancelled(self):
        with QMutexLocker(self._mutex):
            return self._is_cancelled

    def run(self):
        try:
            if self.is_cancelled():
                return
                
            self.progress.emit("Preparing simulation environment...")
            
            temp_base_dir = "pyxel_temp"
            os.makedirs(temp_base_dir, exist_ok=True)
            
            safe_filename = f"validation_{self.simulation_id}_{int(time.time() * 1000)}.fits"
            self.image_filename = os.path.join(temp_base_dir, safe_filename)
            
            self.progress.emit("Saving validation image...")
            
            if np.any(np.isnan(self.image_data)) or np.any(np.isinf(self.image_data)):
                raise ValueError("Invalid image data: contains NaN or infinite values")
            
            if self.image_data.size == 0:
                raise ValueError("Empty image data")
                
            image_to_save = self.image_data.astype(np.float32)
            
            header = fits.Header()
            header['SIMID'] = self.simulation_id
            header['CREATOR'] = 'PyxelWorker'
            header['IMGMIN'] = float(np.min(image_to_save))
            header['IMGMAX'] = float(np.max(image_to_save))
            header['IMGMEAN'] = float(np.mean(image_to_save))
            header['TIMESTAMP'] = datetime.now().isoformat()
            
            hdu = fits.PrimaryHDU(data=image_to_save, header=header)
            hdu.writeto(self.image_filename, overwrite=True)
            
            if not os.path.exists(self.image_filename):
                raise FileNotFoundError(f"Failed to create image file: {self.image_filename}")
            
            test_data = fits.getdata(self.image_filename)
            if not np.allclose(test_data, image_to_save, rtol=1e-5):
                raise ValueError("Image file verification failed - data mismatch")
            
            if self.is_cancelled():
                return
                
            self.progress.emit("Updating Pyxel configuration...")
            
            self.update_load_image_path()
            
            if self.is_cancelled():
                return
                
            self.progress.emit("Running Pyxel simulation...")
            
            detector = self.get_detector_safely()
            if detector is None:
                raise ValueError("No detector found in configuration")
            
            exposure = self.config.exposure
            pipeline = self.config.pipeline
            
            try:
                result_datatree = pyxel.run_mode(mode=exposure, detector=detector, pipeline=pipeline)
                
                if result_datatree is None:
                    raise ValueError("Pyxel simulation returned None")
                
                if "bucket" not in result_datatree:
                    raise ValueError("No bucket data in Pyxel results")
                
                bucket_ds = result_datatree["bucket"].to_dataset()
                
                if "image" not in bucket_ds:
                    raise ValueError("No image data in Pyxel bucket results")
                    
                image_data = bucket_ds["image"].values
                max_adu = np.max(image_data)
                
                print(f"[Worker {self.simulation_id}] Simulation complete:")
                print(f"  Image shape: {image_data.shape}")
                print(f"  ADU range: {np.min(image_data):.2f} to {max_adu:.2f}")
                print(f"  Mean ADU: {np.mean(image_data):.2f}")
                
                if max_adu > 60000:
                    print(f"[Worker {self.simulation_id}] WARNING: High ADU values detected - possible saturation")
                
                self.finished.emit(result_datatree)
                
            except Exception as sim_error:
                raise RuntimeError(f"Pyxel simulation failed: {str(sim_error)}")
            
        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = f"[Simulation {self.simulation_id}] {str(e)}\n\nDetails:\n{error_details}"
            print(f"Worker error: {error_msg}")
            self.error.emit(error_msg)
        finally:
            try:
                if hasattr(self, 'image_filename') and os.path.exists(self.image_filename):
                    os.remove(self.image_filename)
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")

    def get_detector_safely(self):
        detector = None
        
        if hasattr(self.config, 'cmos_detector') and self.config.cmos_detector is not None:
            detector = self.config.cmos_detector
        elif hasattr(self.config, 'detector') and self.config.detector is not None:
            detector = self.config.detector
        elif hasattr(self.config, 'ccd_detector') and self.config.ccd_detector is not None:
            detector = self.config.ccd_detector
        
        return detector

    def update_load_image_path(self):
        image_path = os.path.abspath(self.image_filename)
        
        if hasattr(self.config.pipeline, 'photon_collection') and self.config.pipeline.photon_collection:
            for item in self.config.pipeline.photon_collection:
                if hasattr(item, 'name') and item.name == 'load_image':
                    if not hasattr(item, 'arguments'):
                        item.arguments = {}
                    
                    item.arguments['image_file'] = image_path
                    item.enabled = True
                    
                    print(f"[Worker {self.simulation_id}] Updated load_image path: {image_path}")
                    break
        else:
            raise ValueError("No photon_collection pipeline found or load_image module missing")

class ValidationMainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Validation Tool")
        self.setMinimumSize(1200, 800)
        
        self._simulation_mutex = QMutex()
        self._current_worker = None
        self._simulation_counter = 0
        
        self.emissions_df = pd.DataFrame()
        self.results_df = pd.DataFrame()
        self.sensor_params = self.get_default_sensor_params()
        self.generated_images = {}
        
        self.pyxel_config = None
        self.pyxel_results = None
        self.pyxel_image_cube = None
        self.pyxel_time_points = None
        self.pyxel_analysis_results = None
        
        self.pyxel_results_history = []
        self.current_pyxel_result = None
        
        pg.setConfigOption('imageAxisOrder', 'row-major')
        pg.setConfigOption('useNumba', False)
        
        self.setup_ui()
        self.setup_connections()
        
        self.update_calculated_parameters()
    
    def get_default_sensor_params(self):
        return {
            'aperture_diameter_m': 0.045,
            'focal_length_m': 0.047,
            'pixel_pitch_um': 15.04,
            'sensor_width_px': 1596,
            'sensor_height_px': 2392,
            'exposure_time_s': 1.0,
            'oi_quantum_efficiency': 0.3,
            'oi_filter_transmission': 0.78,
            'al_quantum_efficiency': 0.7,
            'al_filter_transmission': 0.33,
            'oi_wavelength_nm': 777.3,
            'al_wavelength_nm': 395.0
        }

    def clear_simulation_state(self):
        print("Clearing simulation state...")
        
        with QMutexLocker(self._simulation_mutex):
            if self._current_worker is not None and self._current_worker.isRunning():
                print("Cancelling previous simulation...")
                self._current_worker.cancel()
                self._current_worker.wait(3000)
                
        self.pyxel_results = None
        self.pyxel_image_cube = None
        self.pyxel_time_points = None
        self.pyxel_analysis_results = None
        
        temp_dir = "pyxel_temp"
        if os.path.exists(temp_dir):
            try:
                for file in os.listdir(temp_dir):
                    if file.startswith("validation_") and file.endswith(".fits"):
                        file_path = os.path.join(temp_dir, file)
                        if os.path.getmtime(file_path) < time.time() - 60:
                            os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not clean temp files: {e}")
        
        print("Simulation state cleared.")

    def generate_unique_simulation_id(self):
        with QMutexLocker(self._simulation_mutex):
            self._simulation_counter += 1
            return f"sim_{self._simulation_counter}_{uuid.uuid4().hex[:8]}"

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        tabs = QTabWidget()
        
        input_tab = self.create_input_tab()
        tabs.addTab(input_tab, "Input && Parameters")
        
        results_tab = self.create_results_tab()
        tabs.addTab(results_tab, "Results && Analysis")
        
        validation_tab = self.create_validation_tab()
        tabs.addTab(validation_tab, "Validation Plots")
        
        if PYXEL_AVAILABLE:
            pyxel_tab = self.create_pyxel_tab()
            tabs.addTab(pyxel_tab, "Pyxel Simulation")
        else:
            disabled_tab = QWidget()
            disabled_layout = QVBoxLayout(disabled_tab)
            disabled_layout.addWidget(QLabel("Pyxel not available. Please install Pyxel to use this feature."))
            tabs.addTab(disabled_tab, "Pyxel Simulation (Disabled)")
            tabs.setTabEnabled(3, False)
        
        main_layout.addWidget(tabs)
        
        self.status_label = QLabel("Ready to load emissions data")
        self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
        main_layout.addWidget(self.status_label)
    
    def create_input_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        input_group = QGroupBox("Data Input")
        input_layout = QVBoxLayout(input_group)
        
        self.load_emissions_btn = QPushButton("Load Emissions CSV")
        self.load_emissions_btn.setStyleSheet("background-color: #0066cc; font-weight: bold; padding: 10px;")
        input_layout.addWidget(self.load_emissions_btn)
        
        self.data_preview = QTextEdit()
        self.data_preview.setMaximumHeight(150)
        self.data_preview.setPlainText("No data loaded")
        input_layout.addWidget(QLabel("Data Preview:"))
        input_layout.addWidget(self.data_preview)
        
        distance_group = QGroupBox("Distance Settings")
        distance_layout = QFormLayout(distance_group)
        
        self.manual_distance = QDoubleSpinBox()
        self.manual_distance.setRange(0.0001, 50000)
        self.manual_distance.setValue(0.0011)
        self.manual_distance.setSuffix(" km")
        self.manual_distance.setDecimals(4)
        self.manual_distance.setSingleStep(0.0001)
        distance_layout.addRow("Distance:", self.manual_distance)
        
        input_layout.addWidget(distance_group)
        
        sensor_group = QGroupBox("Sensor Parameters")
        sensor_layout = QFormLayout(sensor_group)
        
        self.aperture_input = QDoubleSpinBox()
        self.aperture_input.setRange(0.001, 1.0)
        self.aperture_input.setValue(self.sensor_params['aperture_diameter_m'])
        self.aperture_input.setDecimals(3)
        self.aperture_input.setSuffix(" m")
        sensor_layout.addRow("Aperture Diameter:", self.aperture_input)
        
        self.focal_length_input = QDoubleSpinBox()
        self.focal_length_input.setRange(0.001, 1.0)
        self.focal_length_input.setValue(self.sensor_params['focal_length_m'])
        self.focal_length_input.setDecimals(3)
        self.focal_length_input.setSuffix(" m")
        sensor_layout.addRow("Focal Length:", self.focal_length_input)
        
        self.pixel_pitch_input = QDoubleSpinBox()
        self.pixel_pitch_input.setRange(0.1, 100.0)
        self.pixel_pitch_input.setValue(self.sensor_params['pixel_pitch_um'])
        self.pixel_pitch_input.setDecimals(2)
        self.pixel_pitch_input.setSuffix(" µm")
        sensor_layout.addRow("Pixel Pitch:", self.pixel_pitch_input)
        
        self.sensor_width_input = QSpinBox()
        self.sensor_width_input.setRange(100, 20000)
        self.sensor_width_input.setValue(self.sensor_params['sensor_width_px'])
        self.sensor_width_input.setSuffix(" px")
        sensor_layout.addRow("Sensor Width:", self.sensor_width_input)
        
        self.sensor_height_input = QSpinBox()
        self.sensor_height_input.setRange(100, 20000)
        self.sensor_height_input.setValue(self.sensor_params['sensor_height_px'])
        self.sensor_height_input.setSuffix(" px")
        sensor_layout.addRow("Sensor Height:", self.sensor_height_input)
        
        self.exposure_input = QDoubleSpinBox()
        self.exposure_input.setRange(0.000001, 3600)
        self.exposure_input.setValue(self.sensor_params['exposure_time_s'])
        self.exposure_input.setDecimals(6)
        self.exposure_input.setSuffix(" s")
        sensor_layout.addRow("Exposure Time:", self.exposure_input)
        
        oi_group = QGroupBox("OI Emission Parameters (777.3 nm)")
        oi_layout = QFormLayout(oi_group)
        
        self.oi_qe_input = QDoubleSpinBox()
        self.oi_qe_input.setRange(0.01, 1.0)
        self.oi_qe_input.setValue(self.sensor_params['oi_quantum_efficiency'])
        self.oi_qe_input.setDecimals(3)
        oi_layout.addRow("OI Quantum Efficiency:", self.oi_qe_input)
        
        self.oi_filter_input = QDoubleSpinBox()
        self.oi_filter_input.setRange(0.01, 1.0)
        self.oi_filter_input.setValue(self.sensor_params['oi_filter_transmission'])
        self.oi_filter_input.setDecimals(3)
        oi_layout.addRow("OI Filter Transmission:", self.oi_filter_input)
        
        self.oi_wavelength_input = QDoubleSpinBox()
        self.oi_wavelength_input.setRange(100, 2000)
        self.oi_wavelength_input.setValue(self.sensor_params['oi_wavelength_nm'])
        self.oi_wavelength_input.setDecimals(1)
        self.oi_wavelength_input.setSuffix(" nm")
        oi_layout.addRow("OI Wavelength:", self.oi_wavelength_input)
        
        al_group = QGroupBox("Al Emission Parameters (395.0 nm)")
        al_layout = QFormLayout(al_group)
        
        self.al_qe_input = QDoubleSpinBox()
        self.al_qe_input.setRange(0.01, 1.0)
        self.al_qe_input.setValue(self.sensor_params['al_quantum_efficiency'])
        self.al_qe_input.setDecimals(3)
        al_layout.addRow("Al Quantum Efficiency:", self.al_qe_input)
        
        self.al_filter_input = QDoubleSpinBox()
        self.al_filter_input.setRange(0.01, 1.0)
        self.al_filter_input.setValue(self.sensor_params['al_filter_transmission'])
        self.al_filter_input.setDecimals(3)
        al_layout.addRow("Al Filter Transmission:", self.al_filter_input)
        
        self.al_wavelength_input = QDoubleSpinBox()
        self.al_wavelength_input.setRange(100, 2000)
        self.al_wavelength_input.setValue(self.sensor_params['al_wavelength_nm'])
        self.al_wavelength_input.setDecimals(1)
        self.al_wavelength_input.setSuffix(" nm")
        al_layout.addRow("Al Wavelength:", self.al_wavelength_input)
        
        sensor_layout.addRow(oi_group)
        sensor_layout.addRow(al_group)
        
        param_buttons_layout = QHBoxLayout()
        
        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.setToolTip("Save current parameters to JSON file")
        param_buttons_layout.addWidget(self.save_params_btn)
        
        self.load_params_btn = QPushButton("Load Parameters")
        self.load_params_btn.setToolTip("Load parameters from JSON file")
        param_buttons_layout.addWidget(self.load_params_btn)
        
        self.reset_params_btn = QPushButton("Reset to Defaults")
        self.reset_params_btn.setToolTip("Reset all parameters to default values")
        param_buttons_layout.addWidget(self.reset_params_btn)
        
        sensor_layout.addRow(param_buttons_layout)
        
        self.apply_angular_correction = QCheckBox("Apply Angular Correction (cos factor)")
        self.apply_angular_correction.setChecked(False)
        sensor_layout.addRow(self.apply_angular_correction)
        
        self.angular_cos_factor = QDoubleSpinBox()
        self.angular_cos_factor.setRange(0.0, 1.0)
        self.angular_cos_factor.setValue(1.0)
        self.angular_cos_factor.setDecimals(3)
        self.angular_cos_factor.setSingleStep(0.001)
        self.angular_cos_factor.setEnabled(False)
        self.angular_cos_factor.setToolTip("Cosine factor for angular correction (1.0 = nadir, 0.866 = 30°, 0.5 = 60°)")
        sensor_layout.addRow("Cos Factor:", self.angular_cos_factor)
        
        calc_group = QGroupBox("Calculated Parameters")
        calc_layout = QFormLayout(calc_group)
        
        self.aperture_area_label = QLabel("0.000 m²")
        calc_layout.addRow("Aperture Area:", self.aperture_area_label)
        
        self.pixel_size_label = QLabel("0.00 mm")
        calc_layout.addRow("Pixel Size:", self.pixel_size_label)
        
        self.sensor_size_label = QLabel("0.0 × 0.0 mm")
        calc_layout.addRow("Sensor Physical Size:", self.sensor_size_label)
        
        self.fov_label = QLabel("0.0° × 0.0°")
        calc_layout.addRow("Field of View:", self.fov_label)
        
        self.update_calc_btn = QPushButton("Update Calculated Values")
        calc_layout.addRow(self.update_calc_btn)
        
        sensor_layout.addRow(calc_group)
        
        self.calculate_btn = QPushButton("Calculate Photons && Electrons")
        self.calculate_btn.setStyleSheet("background-color: #ff9900; font-weight: bold; padding: 15px; font-size: 14px;")
        self.calculate_btn.setEnabled(False)
        sensor_layout.addRow(self.calculate_btn)
        
        layout.addWidget(input_group, 1)
        layout.addWidget(sensor_group, 1)
        
        return tab
    
    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        controls_layout = QHBoxLayout()
        
        self.export_results_btn = QPushButton("Export Results to CSV")
        self.export_results_btn.setEnabled(False)
        controls_layout.addWidget(self.export_results_btn)
        
        self.clear_results_btn = QPushButton("Clear Results")
        controls_layout.addWidget(self.clear_results_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.results_table = QTableWidget()
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.results_table)
        
        image_controls_layout = QHBoxLayout()
        
        self.generate_image_btn = QPushButton("Generate Validation Image")
        self.generate_image_btn.setStyleSheet("background-color: #cc6600; font-weight: bold; padding: 8px;")
        self.generate_image_btn.setEnabled(False)
        self.generate_image_btn.setToolTip("Generate sensor image with debris at center pixel - matches main application format")
        image_controls_layout.addWidget(self.generate_image_btn)
        
        self.save_image_btn = QPushButton("Save Image Files")
        self.save_image_btn.setEnabled(False)
        image_controls_layout.addWidget(self.save_image_btn)
        
        self.debris_selector = QComboBox()
        self.debris_selector.setToolTip("Select debris ID for image generation")
        image_controls_layout.addWidget(QLabel("Debris ID:"))
        image_controls_layout.addWidget(self.debris_selector)
        
        self.iteration_selector = QComboBox()
        self.iteration_selector.setToolTip("Select iteration for image generation")
        image_controls_layout.addWidget(QLabel("Iteration:"))
        image_controls_layout.addWidget(self.iteration_selector)
        
        view_note = QLabel("💡 View generated images in 'Validation Plots' tab or run through Pyxel")
        view_note.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        image_controls_layout.addWidget(view_note)
        
        image_controls_layout.addStretch()
        layout.addLayout(image_controls_layout)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(100)
        layout.addWidget(QLabel("Summary Statistics:"))
        layout.addWidget(self.summary_text)
        
        return tab
    
    def create_validation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        plot_controls = QHBoxLayout()
        
        self.plot_type = QComboBox()
        self.plot_type.addItem("Electrons vs Distance", "electrons_distance")
        self.plot_type.addItem("Photons vs Distance", "photons_distance")
        self.plot_type.addItem("Irradiance vs Distance", "irradiance_distance")
        self.plot_type.addItem("Emission Comparison", "emission_comparison")
        self.plot_type.addItem("OI Validation Image", "oi_image")
        self.plot_type.addItem("Al Validation Image", "al_image")
        plot_controls.addWidget(QLabel("Plot Type:"))
        plot_controls.addWidget(self.plot_type)
        
        self.update_plot_btn = QPushButton("Update Plot")
        plot_controls.addWidget(self.update_plot_btn)
        
        self.log_scale_check = QCheckBox("Log Scale")
        self.log_scale_check.setChecked(True)
        plot_controls.addWidget(self.log_scale_check)
        
        plot_controls.addStretch()
        layout.addLayout(plot_controls)
        
        self.plot_stack = QStackedLayout()
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Y Axis')
        self.plot_widget.setLabel('bottom', 'X Axis')
        
        plot_container = QWidget()
        plot_container_layout = QVBoxLayout(plot_container)
        plot_container_layout.addWidget(self.plot_widget)
        
        self.image_widget = pg.ImageView()
        self.image_widget.ui.roiBtn.hide()
        self.image_widget.ui.menuBtn.hide()
        
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        image_container_layout.addWidget(self.image_widget)
        
        self.plot_stack.addWidget(plot_container)
        self.plot_stack.addWidget(image_container)
        
        stack_container = QWidget()
        stack_container.setLayout(self.plot_stack)
        layout.addWidget(stack_container)
        
        return tab
    
    def create_pyxel_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        controls_panel = QGroupBox("Pyxel Simulation Controls")
        controls_panel.setMaximumHeight(200)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setSpacing(3)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        
        yaml_row = QHBoxLayout()
        yaml_row.setSpacing(8)
        
        self.load_yaml_btn = QPushButton("Load YAML")
        self.load_yaml_btn.setStyleSheet("background-color: #0066cc; color: white; font-weight: bold; padding: 6px 12px;")
        self.load_yaml_btn.setMaximumWidth(120)
        yaml_row.addWidget(self.load_yaml_btn)
        
        self.yaml_status_label = QLabel("No YAML configuration loaded")
        self.yaml_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 11px;")
        yaml_row.addWidget(self.yaml_status_label)
        yaml_row.addStretch()
        
        controls_layout.addLayout(yaml_row)
        
        scenario_row = QHBoxLayout()
        scenario_row.setSpacing(8)
        
        scenario_row.addWidget(QLabel("🎯 Scenario:"))
        
        self.pyxel_debris_selector = QComboBox()
        self.pyxel_debris_selector.setToolTip("Select debris ID for simulation")
        self.pyxel_debris_selector.setMaximumWidth(100)
        scenario_row.addWidget(QLabel("Debris:"))
        scenario_row.addWidget(self.pyxel_debris_selector)
        
        self.pyxel_iteration_selector = QComboBox()
        self.pyxel_iteration_selector.setToolTip("Select iteration for simulation")
        self.pyxel_iteration_selector.setMaximumWidth(100)
        scenario_row.addWidget(QLabel("Iter:"))
        scenario_row.addWidget(self.pyxel_iteration_selector)
        
        scenario_row.addStretch()
        controls_layout.addLayout(scenario_row)
        
        sim_row = QHBoxLayout()
        sim_row.setSpacing(8)
        
        sim_row.addWidget(QLabel("Emission:"))
        self.emission_type_selector = QComboBox()
        self.emission_type_selector.addItem("OI Emission", "OI")
        self.emission_type_selector.addItem("Al Emission", "Al")
        self.emission_type_selector.setMaximumWidth(120)
        sim_row.addWidget(self.emission_type_selector)
        
        # Add the auto replace QE checkbox
        self.auto_replace_qe_check = QCheckBox("Auto replace QE")
        self.auto_replace_qe_check.setChecked(True)
        self.auto_replace_qe_check.setToolTip("Automatically replace detector QE based on emission type. Uncheck to use YAML values.")
        sim_row.addWidget(self.auto_replace_qe_check)
        
        self.generate_and_simulate_btn = QPushButton("Generate Image && Simulate")
        self.generate_and_simulate_btn.setStyleSheet("background-color: #ff6600; color: white; font-weight: bold; padding: 8px 15px; font-size: 13px;")
        self.generate_and_simulate_btn.setEnabled(False)
        self.generate_and_simulate_btn.setToolTip("Generate validation image and run Pyxel simulation in one step")
        sim_row.addWidget(self.generate_and_simulate_btn)
        
        self.run_pyxel_btn = QPushButton("▶ Simulate Only")
        self.run_pyxel_btn.setStyleSheet("background-color: #cc6600; color: white; font-weight: bold; padding: 6px 12px;")
        self.run_pyxel_btn.setEnabled(False)
        self.run_pyxel_btn.setToolTip("Run Pyxel simulation on existing validation image")
        self.run_pyxel_btn.setMaximumWidth(120)
        sim_row.addWidget(self.run_pyxel_btn)
        
        sim_row.addStretch()
        controls_layout.addLayout(sim_row)
        
        results_row = QHBoxLayout()
        results_row.setSpacing(8)
        
        self.save_current_csv_btn = QPushButton("💾 Save Current CSV")
        self.save_current_csv_btn.setStyleSheet("background-color: #666666; color: white; padding: 6px 12px;")
        self.save_current_csv_btn.setEnabled(False)
        self.save_current_csv_btn.setMaximumWidth(140)
        results_row.addWidget(self.save_current_csv_btn)
        
        self.save_output_fits_btn = QPushButton("💾 Save Output FITS")
        self.save_output_fits_btn.setStyleSheet("background-color: #9900cc; color: white; padding: 6px 12px;")
        self.save_output_fits_btn.setEnabled(False)
        self.save_output_fits_btn.setMaximumWidth(140)
        self.save_output_fits_btn.setToolTip("Save Pyxel output image as FITS for debugging noise models")
        results_row.addWidget(self.save_output_fits_btn)
        
        self.save_all_csv_btn = QPushButton("📁 Export All CSVs")
        self.save_all_csv_btn.setStyleSheet("background-color: #009900; color: white; padding: 6px 12px;")
        self.save_all_csv_btn.setEnabled(False)
        self.save_all_csv_btn.setMaximumWidth(140)
        self.save_all_csv_btn.setToolTip("Export CSV files for all completed simulations")
        results_row.addWidget(self.save_all_csv_btn)
        
        self.clear_all_results_btn = QPushButton("🗑 Clear All")
        self.clear_all_results_btn.setStyleSheet("background-color: #cc0000; color: white; padding: 6px 12px;")
        self.clear_all_results_btn.setMaximumWidth(100)
        results_row.addWidget(self.clear_all_results_btn)
        
        self.sim_info_label = QLabel("Load YAML config and select scenario to begin")
        self.sim_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        results_row.addWidget(self.sim_info_label)
        results_row.addStretch()
        
        controls_layout.addLayout(results_row)
        
        main_layout.addWidget(controls_panel)
        
        content_splitter = QSplitter(Qt.Horizontal)
        
        history_panel = QGroupBox("Simulation Results History")
        history_panel.setMaximumWidth(400)
        history_panel.setMinimumWidth(350)
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(5, 5, 5, 5)
        
        self.pyxel_history_table = QTableWidget()
        self.pyxel_history_table.setMaximumHeight(200)
        self.pyxel_history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pyxel_history_table.setAlternatingRowColors(True)
        self.pyxel_history_table.setSortingEnabled(True)
        
        history_columns = ['Debris', 'Iter', 'Emission', 'Center ADU', 'Peak ADU', 'SNR', 'Status']
        self.pyxel_history_table.setColumnCount(len(history_columns))
        self.pyxel_history_table.setHorizontalHeaderLabels(history_columns)
        
        history_layout.addWidget(QLabel("📊 Completed Simulations:"))
        history_layout.addWidget(self.pyxel_history_table)
        
        history_controls = QHBoxLayout()
        self.load_selected_result_btn = QPushButton("📖 Load Selected")
        self.load_selected_result_btn.setToolTip("Load selected result from history")
        self.load_selected_result_btn.setEnabled(False)
        history_controls.addWidget(self.load_selected_result_btn)
        
        self.delete_selected_btn = QPushButton("❌ Delete")
        self.delete_selected_btn.setToolTip("Delete selected result from history")
        self.delete_selected_btn.setEnabled(False)
        history_controls.addWidget(self.delete_selected_btn)
        
        history_controls.addStretch()
        history_layout.addLayout(history_controls)
        
        history_layout.addWidget(QLabel("📋 Current Result Summary:"))
        self.pyxel_results_text = QTextEdit()
        self.pyxel_results_text.setPlainText("No simulation results yet")
        self.pyxel_results_text.setFont(QFont("Consolas", 9))
        self.pyxel_results_text.setMaximumHeight(250)
        history_layout.addWidget(self.pyxel_results_text)
        
        yaml_group = QGroupBox("YAML Preview")
        yaml_group.setCheckable(True)
        yaml_group.setChecked(False)
        yaml_group.setMaximumHeight(150)
        yaml_layout = QVBoxLayout(yaml_group)
        yaml_layout.setContentsMargins(5, 5, 5, 5)
        
        self.yaml_preview = QPlainTextEdit()
        self.yaml_preview.setReadOnly(True)
        self.yaml_preview.setPlainText("Load a YAML configuration to see preview")
        self.yaml_preview.setFont(QFont("Consolas", 8))
        yaml_layout.addWidget(self.yaml_preview)
        
        history_layout.addWidget(yaml_group)
        
        image_panel = QGroupBox("Current Simulation Result")
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        image_controls = QHBoxLayout()
        image_controls.addWidget(QLabel("🖼 Current Pyxel Output:"))
        
        image_controls.addWidget(QLabel("Colormap:"))
        self.pyxel_colormap = QComboBox()
        self.pyxel_colormap.addItems(["viridis", "plasma", "inferno", "magma", "hot", "cool", "grey"])
        self.pyxel_colormap.setMaximumWidth(100)
        self.pyxel_colormap.currentTextChanged.connect(self.change_pyxel_colormap)
        image_controls.addWidget(self.pyxel_colormap)
        
        self.auto_levels_btn = QPushButton("Auto Levels")
        self.auto_levels_btn.setMaximumWidth(100)
        self.auto_levels_btn.clicked.connect(self.auto_levels_pyxel_image)
        image_controls.addWidget(self.auto_levels_btn)
        
        self.current_scenario_label = QLabel("No simulation loaded")
        self.current_scenario_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        image_controls.addWidget(self.current_scenario_label)
        
        image_controls.addStretch()
        image_layout.addLayout(image_controls)
        
        self.pyxel_image_widget = pg.ImageView()
        self.pyxel_image_widget.ui.roiBtn.hide()
        self.pyxel_image_widget.ui.menuBtn.hide()
        self.pyxel_image_widget.setPredefinedGradient("viridis")
        
        self.pyxel_image_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.pyxel_image_widget)
        
        content_splitter.addWidget(history_panel)
        content_splitter.addWidget(image_panel)
        content_splitter.setSizes([350, 650])
        
        main_layout.addWidget(content_splitter)
        
        return tab
    
    def setup_connections(self):
        self.load_emissions_btn.clicked.connect(self.load_emissions_data)
        self.calculate_btn.clicked.connect(self.calculate_photon_conversion)
        self.export_results_btn.clicked.connect(self.export_results)
        self.clear_results_btn.clicked.connect(self.clear_results)
        self.update_plot_btn.clicked.connect(self.update_validation_plot)
        
        self.generate_image_btn.clicked.connect(self.generate_validation_images)
        self.save_image_btn.clicked.connect(self.save_image_files)
        
        self.debris_selector.currentTextChanged.connect(self.on_debris_selection_changed)
        self.iteration_selector.currentTextChanged.connect(self.on_iteration_selection_changed)
        
        self.log_scale_check.stateChanged.connect(self.update_validation_plot)
        
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.load_params_btn.clicked.connect(self.load_parameters)
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        self.update_calc_btn.clicked.connect(self.update_calculated_parameters)
        
        if PYXEL_AVAILABLE:
            self.load_yaml_btn.clicked.connect(self.load_yaml_configuration)
            
            self.generate_and_simulate_btn.clicked.connect(self.generate_and_simulate)
            self.run_pyxel_btn.clicked.connect(self.run_pyxel_simulation)
            
            self.pyxel_debris_selector.currentTextChanged.connect(self.on_pyxel_debris_selection_changed)
            self.pyxel_iteration_selector.currentTextChanged.connect(self.on_pyxel_iteration_selection_changed)
            
            self.save_current_csv_btn.clicked.connect(self.save_current_pyxel_csv)
            self.save_output_fits_btn.clicked.connect(self.save_output_fits)
            self.save_all_csv_btn.clicked.connect(self.save_all_pyxel_csvs)
            self.clear_all_results_btn.clicked.connect(self.clear_all_pyxel_results)
            
            self.pyxel_history_table.itemSelectionChanged.connect(self.on_history_selection_changed)
            self.load_selected_result_btn.clicked.connect(self.load_selected_result)
            self.delete_selected_btn.clicked.connect(self.delete_selected_result)
            
            self.auto_levels_btn.clicked.connect(self.auto_levels_pyxel_image)
            self.pyxel_colormap.currentTextChanged.connect(self.change_pyxel_colormap)
        
        for param_input in [self.aperture_input, self.focal_length_input, self.pixel_pitch_input,
                           self.sensor_width_input, self.sensor_height_input]:
            param_input.valueChanged.connect(self.update_calculated_parameters)
        
        self.apply_angular_correction.stateChanged.connect(self.on_angular_correction_changed)
    
    def on_angular_correction_changed(self):
        is_enabled = self.apply_angular_correction.isChecked()
        self.angular_cos_factor.setEnabled(is_enabled)
        
        if is_enabled:
            self.status_label.setText(f"Angular correction enabled (cos factor = {self.angular_cos_factor.value():.3f})")
        else:
            self.status_label.setText("Angular correction disabled")
    
    def on_debris_selection_changed(self):
        selected_debris_id = self.debris_selector.currentData()
        if selected_debris_id is None or self.results_df.empty:
            return
        
        available_iterations = sorted(self.results_df[
            self.results_df['Assembly_ID'] == selected_debris_id
        ]['Iteration'].unique())
        
        self.iteration_selector.clear()
        for iteration in available_iterations:
            self.iteration_selector.addItem(f"Iteration {iteration}", iteration)
        
        if len(available_iterations) > 0:
            self.iteration_selector.setCurrentIndex(0)
        
        self.on_iteration_selection_changed()
    
    def on_iteration_selection_changed(self):
        pass
    
    def update_pyxel_selectors(self):
        if self.results_df.empty:
            self.pyxel_debris_selector.clear()
            self.pyxel_iteration_selector.clear()
            self.generate_and_simulate_btn.setEnabled(False)
            return
        
        self.pyxel_debris_selector.clear()
        debris_ids = sorted(self.results_df['Assembly_ID'].unique())
        for debris_id in debris_ids:
            self.pyxel_debris_selector.addItem(f"Debris {debris_id}", debris_id)
        
        has_config = self.pyxel_config is not None
        has_data = len(debris_ids) > 0
        self.generate_and_simulate_btn.setEnabled(has_config and has_data)
        
        if len(debris_ids) > 0:
            self.pyxel_debris_selector.setCurrentIndex(0)
    
    def on_pyxel_debris_selection_changed(self):
        selected_debris_id = self.pyxel_debris_selector.currentData()
        if selected_debris_id is None or self.results_df.empty:
            return
        
        available_iterations = sorted(self.results_df[
            self.results_df['Assembly_ID'] == selected_debris_id
        ]['Iteration'].unique())
        
        self.pyxel_iteration_selector.clear()
        for iteration in available_iterations:
            self.pyxel_iteration_selector.addItem(f"Iteration {iteration}", iteration)
        
        if len(available_iterations) > 0:
            self.pyxel_iteration_selector.setCurrentIndex(0)
        
        self.on_pyxel_iteration_selection_changed()
    
    def on_pyxel_iteration_selection_changed(self):
        pass
    
    def update_pyxel_detector_qe(self, emission_type):
        if self.pyxel_config is None:
            return False
        
        try:
            if emission_type == "OI":
                target_qe = self.oi_qe_input.value()
                wavelength = self.oi_wavelength_input.value()
            else:
                target_qe = self.al_qe_input.value()
                wavelength = self.al_wavelength_input.value()
            
            detector = None
            if hasattr(self.pyxel_config, 'cmos_detector'):
                detector = self.pyxel_config.cmos_detector
            elif hasattr(self.pyxel_config, 'detector'):
                detector = self.pyxel_config.detector
            elif hasattr(self.pyxel_config, 'ccd_detector'):
                detector = self.pyxel_config.ccd_detector
            
            if detector and hasattr(detector, 'characteristics'):
                detector.characteristics.quantum_efficiency = target_qe
                
                print(f"✓ Updated Pyxel detector QE to {target_qe:.3f} for {emission_type} emission ({wavelength:.1f} nm)")
                return True
            else:
                print("Warning: Could not find detector characteristics to update QE")
                return False
                
        except Exception as e:
            print(f"Error updating Pyxel detector QE: {e}")
            return False

    def generate_and_simulate(self):
        if not PYXEL_AVAILABLE:
            QMessageBox.warning(self, "Pyxel Not Available", "Pyxel is not installed.")
            return
        
        self.clear_simulation_state()
        
        if self.pyxel_config is None:
            QMessageBox.warning(self, "No Configuration", "Please load a YAML configuration first.")
            return
        
        if self.results_df.empty:
            QMessageBox.warning(self, "No Data", "No calculation results available.")
            return
        
        selected_debris_id = self.pyxel_debris_selector.currentData()
        selected_iteration = self.pyxel_iteration_selector.currentData()
        emission_type = self.emission_type_selector.currentData()
        distance_km = self.manual_distance.value()
        
        if selected_debris_id is None or selected_iteration is None:
            QMessageBox.warning(self, "No Selection", "Please select debris ID and iteration.")
            return
        
        with QMutexLocker(self._simulation_mutex):
            if self._current_worker is not None and self._current_worker.isRunning():
                QMessageBox.warning(self, "Simulation Running", "A simulation is already running. Please wait for it to complete.")
                return
        
        try:
            self.status_label.setText("Updating QE and generating image for Pyxel simulation...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            QApplication.processEvents()
            
            # Check if auto replace QE is enabled
            if self.auto_replace_qe_check.isChecked():
                qe_updated = self.update_pyxel_detector_qe(emission_type)
                if not qe_updated:
                    QMessageBox.warning(self, "QE Update Failed", 
                                       "Could not update detector quantum efficiency. Proceeding with default QE.")
            else:
                print(f"Auto replace QE is disabled - using YAML configuration QE values")
                qe_updated = False
            
            W = self.sensor_width_input.value()
            H = self.sensor_height_input.value()
            
            result_row = self.results_df[
                (self.results_df['Assembly_ID'] == selected_debris_id) & 
                (self.results_df['Iteration'] == selected_iteration) &
                (self.results_df['Distance_km'] == distance_km)
            ]
            
            if result_row.empty:
                QMessageBox.warning(self, "Data Not Found", 
                                   f"No data found for Debris {selected_debris_id}, Iteration {selected_iteration} at {distance_km} km")
                return
            
            row_data = result_row.iloc[0]
            
            center_x = W // 2
            center_y = H // 2
            
            if emission_type == "OI":
                photons = row_data['OI_Photons']
                electrons = row_data['OI_Electrons']
            else:
                photons = row_data['Al_Photons']
                electrons = row_data['Al_Electrons']
            
            if np.isnan(photons) or np.isinf(photons) or photons < 0:
                QMessageBox.warning(self, "Invalid Data", f"Invalid photon count: {photons}. Please check input data.")
                return
            
            validation_image = np.zeros((H, W), dtype=np.float64)
            if photons > 0:
                validation_image[center_y, center_x] = photons
            
            if np.any(np.isnan(validation_image)) or np.any(np.isinf(validation_image)):
                QMessageBox.critical(self, "Image Generation Error", "Generated image contains invalid values.")
                return
            
            # Get the applied QE for metadata
            if self.auto_replace_qe_check.isChecked():
                applied_qe = self.oi_qe_input.value() if emission_type == "OI" else self.al_qe_input.value()
            else:
                # Try to get QE from the detector configuration
                try:
                    detector = None
                    if hasattr(self.pyxel_config, 'cmos_detector'):
                        detector = self.pyxel_config.cmos_detector
                    elif hasattr(self.pyxel_config, 'detector'):
                        detector = self.pyxel_config.detector
                    elif hasattr(self.pyxel_config, 'ccd_detector'):
                        detector = self.pyxel_config.ccd_detector
                    
                    if detector and hasattr(detector, 'characteristics'):
                        applied_qe = detector.characteristics.quantum_efficiency
                    else:
                        applied_qe = 0.0  # Unknown QE from YAML
                except:
                    applied_qe = 0.0  # Unknown QE from YAML
            
            scenario_metadata = {
                'debris_id': selected_debris_id,
                'iteration': selected_iteration,
                'distance_km': distance_km,
                'emission_type': emission_type,
                'center_pixel': (center_x, center_y),
                'input_photons': photons,
                'expected_electrons': electrons,
                'sensor_size': (W, H),
                'generation_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'oi_electrons': row_data['OI_Electrons'],
                'al_electrons': row_data['Al_Electrons'],
                'oi_photons': row_data['OI_Photons'],
                'al_photons': row_data['Al_Photons'],
                'oi_emission_wsr': row_data['OI_Emission_Wsr'],
                'al_emission_wsr': row_data['Al_Emission_Wsr'],
                'applied_qe': applied_qe,
                'auto_qe_replaced': self.auto_replace_qe_check.isChecked()
            }
            
            temp_dir = "pyxel_temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            simulation_id = self.generate_unique_simulation_id()
            
            progress_dialog = QProgressDialog("Updating QE and running Pyxel simulation...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Generate & Simulate")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(300)
            progress_dialog.setCancelButton(None)
            progress_dialog.show()
            
            with QMutexLocker(self._simulation_mutex):
                self._current_worker = PyxelSimulationWorker(
                    self.pyxel_config, 
                    validation_image, 
                    "",
                    simulation_id
                )
                self._current_worker.scenario_metadata = scenario_metadata
                
                self._current_worker.finished.connect(
                    lambda result: self.handle_generate_and_simulate_complete(result, progress_dialog, scenario_metadata, simulation_id)
                )
                self._current_worker.error.connect(
                    lambda error: self.handle_pyxel_error(error, progress_dialog, simulation_id)
                )
                self._current_worker.progress.connect(lambda msg: progress_dialog.setLabelText(msg))
                
                self._current_worker.start()
            
            self.generate_and_simulate_btn.setEnabled(False)
            self.run_pyxel_btn.setEnabled(False)
            
            qe_status = f"Applied QE: {applied_qe:.3f}" if self.auto_replace_qe_check.isChecked() else "Using YAML QE"
            
            print(f"Started Pyxel simulation [{simulation_id}] with:")
            print(f"  Emission type: {emission_type}")
            print(f"  Auto replace QE: {self.auto_replace_qe_check.isChecked()}")
            print(f"  {qe_status}")
            print(f"  Input photons: {photons:.3f}")
            print(f"  Expected electrons: {electrons:.3f}")
            print(f"  Image validation: Min={np.min(validation_image):.3f}, Max={np.max(validation_image):.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Error in generate and simulate: {str(e)}")
            self.generate_and_simulate_btn.setEnabled(True)
            self.run_pyxel_btn.setEnabled(True)

    def handle_generate_and_simulate_complete(self, result_datatree, progress_dialog, scenario_metadata, simulation_id):
        progress_dialog.close()
        
        try:
            print(f"Processing simulation results [{simulation_id}]...")
            
            with QMutexLocker(self._simulation_mutex):
                self._current_worker = None
            
            self.pyxel_results = result_datatree
            
            if result_datatree is None:
                raise ValueError("Pyxel simulation returned None")
            
            if "bucket" not in result_datatree:
                raise ValueError("No bucket data in Pyxel results")
            
            bucket_ds = result_datatree["bucket"].to_dataset()
            
            if "image" not in bucket_ds:
                raise ValueError("No image data in Pyxel bucket results")
            
            if "time" not in bucket_ds.coords:
                print("Warning: No time coordinate in bucket data, using default")
                time_points = np.array([0.0])
            else:
                time_points = bucket_ds["time"].values
            
            self.pyxel_image_cube = bucket_ds["image"].values
            self.pyxel_time_points = time_points
            
            if self.pyxel_image_cube.size == 0:
                raise ValueError("Empty image data in Pyxel results")
            
            if np.any(np.isnan(self.pyxel_image_cube)) or np.any(np.isinf(self.pyxel_image_cube)):
                raise ValueError("Invalid values (NaN/Inf) in Pyxel image results")
            
            if len(self.pyxel_image_cube.shape) == 3 and self.pyxel_image_cube.shape[0] > 0:
                final_image = self.pyxel_image_cube[-1]
            elif len(self.pyxel_image_cube.shape) == 2:
                final_image = self.pyxel_image_cube
            else:
                final_image = self.pyxel_image_cube[0] if self.pyxel_image_cube.shape[0] > 0 else np.zeros((100, 100))
            
            max_adu = np.max(final_image)
            if max_adu > 60000:
                print(f"WARNING [{simulation_id}]: High ADU values detected ({max_adu:.2f}) - possible saturation")
            
            analysis_results = self.analyze_pyxel_results(final_image, bucket_ds)
            
            analysis_results.update({
                'simulation_id': simulation_id,
                'emission_type': scenario_metadata['emission_type'],
                'debris_id': scenario_metadata['debris_id'],
                'iteration': scenario_metadata['iteration'],
                'distance_km': scenario_metadata['distance_km'],
                'input_photons': scenario_metadata['input_photons'],
                'expected_electrons': scenario_metadata['expected_electrons'],
                'oi_electrons_available': scenario_metadata['oi_electrons'],
                'al_electrons_available': scenario_metadata['al_electrons'],
                'oi_photons_available': scenario_metadata['oi_photons'],
                'al_photons_available': scenario_metadata['al_photons'],
                'oi_emission_wsr': scenario_metadata['oi_emission_wsr'],
                'al_emission_wsr': scenario_metadata['al_emission_wsr'],
                'applied_qe': scenario_metadata['applied_qe'],
                'auto_qe_replaced': scenario_metadata['auto_qe_replaced'],
                'angular_correction_applied': self.apply_angular_correction.isChecked(),
                'angular_cos_factor': self.angular_cos_factor.value()
            })
            
            complete_result = {
                'simulation_id': simulation_id,
                'scenario_metadata': scenario_metadata,
                'analysis_results': analysis_results,
                'image_cube': self.pyxel_image_cube,
                'time_points': self.pyxel_time_points,
                'final_image': final_image,
                'pyxel_dataset': bucket_ds,
                'completion_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            self.add_result_to_history(complete_result)
            
            self.current_pyxel_result = complete_result
            
            self.display_current_result()
            
            self.save_current_csv_btn.setEnabled(True)
            self.save_output_fits_btn.setEnabled(True)
            self.save_all_csv_btn.setEnabled(len(self.pyxel_results_history) > 0)
            
            self.current_scenario_label.setText(f"Debris {scenario_metadata['debris_id']}, Iter {scenario_metadata['iteration']}, {scenario_metadata['distance_km']:.4f}km, {scenario_metadata['emission_type']}")
            
            self.status_label.setText("✓ Generate && Simulate completed successfully")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
            qe_status = f"Applied QE: {scenario_metadata['applied_qe']:.3f}" if scenario_metadata['auto_qe_replaced'] else f"YAML QE: {scenario_metadata['applied_qe']:.3f}"
            
            print(f"Generate & Simulate complete [{simulation_id}]:")
            print(f"  Input photons: {scenario_metadata['input_photons']:.3f}")
            print(f"  Expected electrons: {scenario_metadata['expected_electrons']:.3f}")
            print(f"  Auto QE replace: {scenario_metadata['auto_qe_replaced']}")
            print(f"  {qe_status}")
            print(f"  Center Pixel ADU: {analysis_results.get('center_pixel_adu', 0):.2f}")
            print(f"  Peak ADU: {analysis_results.get('peak_adu', 0):.2f}")
            print(f"  SNR: {analysis_results.get('snr', 0):.2f}")
            
        except Exception as e:
            error_msg = f"Error processing results [{simulation_id}]:\n{str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Results Error", error_msg)
            self.status_label.setText("Error processing simulation results")
            self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
        
        finally:
            self.generate_and_simulate_btn.setEnabled(True)
            self.run_pyxel_btn.setEnabled(True)

    def handle_pyxel_error(self, error_message, progress_dialog, simulation_id):
        progress_dialog.close()
        
        with QMutexLocker(self._simulation_mutex):
            self._current_worker = None
        
        print(f"Simulation error [{simulation_id}]: {error_message}")
        QMessageBox.critical(self, "Pyxel Simulation Error", f"Pyxel simulation failed [{simulation_id}]:\n{error_message}")
        
        self.status_label.setText("Pyxel simulation failed")
        self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
        
        self.generate_and_simulate_btn.setEnabled(True)
        self.run_pyxel_btn.setEnabled(True)

    def analyze_pyxel_results(self, final_image, result_ds):
        try:
            analysis = {}
            
            analysis['image_shape'] = final_image.shape
            analysis['time_frames'] = len(self.pyxel_time_points) if self.pyxel_time_points is not None else 1
            
            analysis['min_adu'] = float(final_image.min())
            analysis['max_adu'] = float(final_image.max())
            analysis['mean_adu'] = float(final_image.mean())
            analysis['std_adu'] = float(final_image.std())
            analysis['median_adu'] = float(np.median(final_image))
            analysis['peak_adu'] = float(final_image.max())
            analysis['total_adu_count'] = float(final_image.sum())
            
            peak_location = np.unravel_index(np.argmax(final_image), final_image.shape)
            analysis['peak_row'] = int(peak_location[0])
            analysis['peak_col'] = int(peak_location[1])
            
            center_row, center_col = final_image.shape[0] // 2, final_image.shape[1] // 2
            analysis['center_pixel_adu'] = float(final_image[center_row, center_col])
            analysis['center_pixel_row'] = center_row
            analysis['center_pixel_col'] = center_col
            
            non_zero_mask = final_image > 0
            analysis['non_zero_pixels'] = int(np.count_nonzero(final_image))
            analysis['total_pixels'] = int(final_image.size)
            analysis['fill_factor'] = float(analysis['non_zero_pixels'] / analysis['total_pixels'])
            
            background_region_size = 50
            
            y_indices, x_indices = np.ogrid[:final_image.shape[0], :final_image.shape[1]]
            center_mask = ((y_indices - center_row)**2 + (x_indices - center_col)**2) < background_region_size**2
            background_mask = ~center_mask
            
            if np.any(background_mask):
                background_pixels = final_image[background_mask]
                analysis['background_mean'] = float(background_pixels.mean())
                analysis['background_std'] = float(background_pixels.std())
                analysis['background_median'] = float(np.median(background_pixels))
            else:
                analysis['background_mean'] = float(final_image.mean())
                analysis['background_std'] = float(final_image.std())
                analysis['background_median'] = float(np.median(final_image))
            
            signal_level = analysis['peak_adu'] - analysis['background_mean']
            noise_level = analysis['background_std']
            analysis['snr'] = float(signal_level / noise_level) if noise_level > 0 else float('inf')
            
            max_possible_adu = 65535
            analysis['max_possible_adu'] = max_possible_adu
            analysis['dynamic_range_utilization'] = float(analysis['peak_adu'] / max_possible_adu)
            
            saturation_threshold = 0.95 * max_possible_adu
            analysis['saturated_pixels'] = int(np.count_nonzero(final_image >= saturation_threshold))
            analysis['saturation_percentage'] = float(analysis['saturated_pixels'] / analysis['total_pixels'] * 100)
            
            emission_type = self.emission_type_selector.currentData()
            
            if self.auto_replace_qe_check.isChecked():
                if emission_type == "OI":
                    applied_qe = self.oi_qe_input.value()
                else:
                    applied_qe = self.al_qe_input.value()
            else:
                # Try to get QE from detector config
                try:
                    detector = None
                    if hasattr(self.pyxel_config, 'cmos_detector'):
                        detector = self.pyxel_config.cmos_detector
                    elif hasattr(self.pyxel_config, 'detector'):
                        detector = self.pyxel_config.detector
                    elif hasattr(self.pyxel_config, 'ccd_detector'):
                        detector = self.pyxel_config.ccd_detector
                    
                    if detector and hasattr(detector, 'characteristics'):
                        applied_qe = detector.characteristics.quantum_efficiency
                    else:
                        applied_qe = 0.0
                except:
                    applied_qe = 0.0
            
            analysis['emission_type'] = emission_type
            analysis['applied_quantum_efficiency'] = applied_qe
            
            generated_images_metadata = self.generated_images.get('metadata', {})
            if emission_type == "OI":
                input_photons = generated_images_metadata.get('oi_photons', 0)
                expected_electrons = generated_images_metadata.get('oi_electrons', 0)
            else:
                input_photons = generated_images_metadata.get('al_photons', 0) 
                expected_electrons = generated_images_metadata.get('al_electrons', 0)
            
            analysis['input_photons'] = input_photons
            analysis['expected_electrons'] = expected_electrons
            analysis['theoretical_electrons_from_qe'] = input_photons * applied_qe
            
            if input_photons > 0:
                signal_pixels = final_image - analysis['background_mean']
                signal_pixels[signal_pixels < 0] = 0
                total_signal_adu = float(signal_pixels.sum())
                analysis['total_signal_adu'] = total_signal_adu
                
                analysis['conversion_efficiency_adu_per_photon'] = total_signal_adu / input_photons
                
                center_signal_adu = analysis['center_pixel_adu'] - analysis['background_mean']
                center_signal_adu = max(0, center_signal_adu)
                analysis['center_pixel_signal_adu'] = center_signal_adu
                analysis['center_pixel_conversion_efficiency'] = center_signal_adu / input_photons
                
                if expected_electrons > 0:
                    analysis['pyxel_vs_expected_electrons_ratio'] = (center_signal_adu / analysis['conversion_efficiency_adu_per_photon']) / expected_electrons if analysis['conversion_efficiency_adu_per_photon'] > 0 else 0
            else:
                analysis['total_signal_adu'] = 0.0
                analysis['conversion_efficiency_adu_per_photon'] = 0.0
                analysis['center_pixel_signal_adu'] = 0.0
                analysis['center_pixel_conversion_efficiency'] = 0.0
                analysis['pyxel_vs_expected_electrons_ratio'] = 0.0
            
            analysis['sensor_parameters'] = {
                'width_px': self.sensor_width_input.value(),
                'height_px': self.sensor_height_input.value(),
                'pixel_size_um': self.pixel_pitch_input.value(),
                'exposure_time_s': self.exposure_input.value(),
                'applied_qe': applied_qe,
                'oi_quantum_efficiency': self.oi_qe_input.value(),
                'oi_filter_transmission': self.oi_filter_input.value(),
                'al_quantum_efficiency': self.al_qe_input.value(),
                'al_filter_transmission': self.al_filter_input.value(),
                'aperture_diameter_m': self.aperture_input.value(),
                'focal_length_m': self.focal_length_input.value(),
                'auto_qe_replaced': self.auto_replace_qe_check.isChecked()
            }
            
            # Add other sensor data types if available in bucket_ds
            additional_data = {}
            for data_var in result_ds.data_vars:
                if data_var != 'image':  # We already have image
                    try:
                        var_data = result_ds[data_var].values
                        if len(var_data.shape) >= 2:  # 2D or 3D data
                            if len(var_data.shape) == 3:
                                var_final = var_data[-1]  # Last time frame
                            else:
                                var_final = var_data
                            
                            additional_data[f'{data_var}_center_value'] = float(var_final[center_row, center_col])
                            additional_data[f'{data_var}_peak_value'] = float(var_final.max())
                            additional_data[f'{data_var}_mean_value'] = float(var_final.mean())
                    except Exception as e:
                        print(f"Warning: Could not extract {data_var}: {e}")
            
            analysis['additional_sensor_data'] = additional_data
            analysis['analysis_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return analysis
            
        except Exception as e:
            print(f"Error in Pyxel analysis: {e}")
            return {
                'error': str(e),
                'min_adu': float(final_image.min()),
                'max_adu': float(final_image.max()),
                'mean_adu': float(final_image.mean()),
                'analysis_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

    def add_result_to_history(self, result):
        self.pyxel_results_history.append(result)
        self.update_pyxel_history_table()

    def update_pyxel_history_table(self):
        self.pyxel_history_table.setRowCount(len(self.pyxel_results_history))
        
        for row, result in enumerate(self.pyxel_results_history):
            scenario = result['scenario_metadata']
            analysis = result['analysis_results']
            
            self.pyxel_history_table.setItem(row, 0, QTableWidgetItem(str(scenario['debris_id'])))
            self.pyxel_history_table.setItem(row, 1, QTableWidgetItem(str(scenario['iteration'])))
            self.pyxel_history_table.setItem(row, 2, QTableWidgetItem(scenario['emission_type']))
            self.pyxel_history_table.setItem(row, 3, QTableWidgetItem(f"{analysis.get('center_pixel_adu', 0):.1f}"))
            self.pyxel_history_table.setItem(row, 4, QTableWidgetItem(f"{analysis.get('peak_adu', 0):.1f}"))
            self.pyxel_history_table.setItem(row, 5, QTableWidgetItem(f"{analysis.get('snr', 0):.1f}"))
            self.pyxel_history_table.setItem(row, 6, QTableWidgetItem("✓ Complete"))
        
        self.pyxel_history_table.resizeColumnsToContents()

    def display_current_result(self):
        if self.current_pyxel_result is None:
            return
        
        final_image = self.current_pyxel_result['final_image']
        self.pyxel_image_widget.setImage(final_image, autoRange=True, autoLevels=True)
        
        analysis = self.current_pyxel_result['analysis_results']
        self.update_pyxel_results_display(analysis)

    def update_pyxel_results_display(self, analysis):
        summary_lines = []
        
        summary_lines.append(f"=== PYXEL SIMULATION ANALYSIS ===")
        summary_lines.append(f"Simulation ID: {analysis.get('simulation_id', 'N/A')}")
        summary_lines.append(f"Emission Type: {analysis.get('emission_type', 'Unknown')}")
        summary_lines.append(f"Debris ID: {analysis.get('debris_id', 'N/A')}")
        summary_lines.append(f"Iteration: {analysis.get('iteration', 'N/A')}")
        summary_lines.append(f"Distance: {analysis.get('distance_km', 'N/A')} km")
        
        # Show QE information based on auto replace setting
        sensor_params = analysis.get('sensor_parameters', {})
        auto_qe_replaced = sensor_params.get('auto_qe_replaced', True)
        applied_qe = analysis.get('applied_quantum_efficiency', 0)
        
        if auto_qe_replaced:
            summary_lines.append(f"Applied QE (auto): {applied_qe:.3f}")
        else:
            summary_lines.append(f"Applied QE (YAML): {applied_qe:.3f}")
        
        summary_lines.append(f"Analysis Time: {analysis.get('analysis_timestamp', 'N/A')}")
        summary_lines.append("")
        
        # Add the breakdown section similar to the check_adu.py output
        summary_lines.append("=" * 70)
        summary_lines.append("🎯 PIPELINE VALUE BREAKDOWN - LAB COMPARISON")
        summary_lines.append("=" * 70)
        
        # Input image information
        input_photons = analysis.get('input_photons', 0)
        summary_lines.append(f"INPUT IMAGE: {input_photons:.6f} photons")
        summary_lines.append("-" * 70)
        
        # Get additional sensor data for pipeline values
        additional_data = analysis.get('additional_sensor_data', {})
        
        # Pipeline output values with proper formatting
        pipeline_values = [
            ('photon', additional_data.get('photon_center_value', 0), 'Ph'),
            ('charge', additional_data.get('charge_center_value', 0), 'e⁻'),
            ('pixel', additional_data.get('pixel_center_value', 0), 'e⁻'),
            ('signal', additional_data.get('signal_center_value', 0), 'V'),
            ('image', analysis.get('center_pixel_adu', 0), 'adu')
        ]
        
        for var_name, value, unit in pipeline_values:
            summary_lines.append(f"{var_name:<15} {value:<15.6f} {unit}")
        
        summary_lines.append("")
        summary_lines.append(" PRIMARY LAB COMPARISON VALUE:")
        
        # Get the final image variable (ADU)
        final_adu = analysis.get('center_pixel_adu', 0)
        summary_lines.append(f"   Variable: image")
        summary_lines.append(f"   Central Pixel: {final_adu:.6f} adu")
        
        # Calculate conversion factor from input
        if input_photons > 0:
            conversion_factor = final_adu / input_photons
            summary_lines.append(f"   Conversion from input: {conversion_factor:.6f} (adu/photons)")
            summary_lines.append(f"   Input → Output: {input_photons:.6f} → {final_adu:.6f}")
        else:
            summary_lines.append(f"   Conversion from input: N/A (no input photons)")
            summary_lines.append(f"   Input → Output: 0.000000 → {final_adu:.6f}")
        
        summary_lines.append("")
        
        # Continue with existing analysis sections
        summary_lines.append("--- IMAGE PROPERTIES ---")
        summary_lines.append(f"Image Shape: {analysis.get('image_shape', 'N/A')}")
        summary_lines.append(f"Time Frames: {analysis.get('time_frames', 1)}")
        summary_lines.append(f"Total Pixels: {analysis.get('total_pixels', 0):,}")
        summary_lines.append(f"Non-zero Pixels: {analysis.get('non_zero_pixels', 0):,}")
        summary_lines.append(f"Fill Factor: {analysis.get('fill_factor', 0)*100:.3f}%")
        summary_lines.append("")
        
        summary_lines.append("--- ADU STATISTICS ---")
        summary_lines.append(f"Peak ADU: {analysis.get('peak_adu', 0):.2f}")
        summary_lines.append(f"Peak Location: ({analysis.get('peak_row', 0)}, {analysis.get('peak_col', 0)})")
        summary_lines.append(f"CENTER PIXEL ADU: {analysis.get('center_pixel_adu', 0):.2f}")
        summary_lines.append(f"Center Pixel Location: ({analysis.get('center_pixel_row', 0)}, {analysis.get('center_pixel_col', 0)})")
        summary_lines.append(f"Total ADU Count: {analysis.get('total_adu_count', 0):.0f}")
        summary_lines.append(f"Mean ADU: {analysis.get('mean_adu', 0):.3f}")
        summary_lines.append(f"Median ADU: {analysis.get('median_adu', 0):.3f}")
        summary_lines.append(f"Std ADU: {analysis.get('std_adu', 0):.3f}")
        summary_lines.append(f"Min ADU: {analysis.get('min_adu', 0):.2f}")
        summary_lines.append(f"Max ADU: {analysis.get('max_adu', 0):.2f}")
        summary_lines.append("")
        
        summary_lines.append("--- BACKGROUND & NOISE ---")
        summary_lines.append(f"Background Mean: {analysis.get('background_mean', 0):.3f} ADU")
        summary_lines.append(f"Background Std: {analysis.get('background_std', 0):.3f} ADU")
        summary_lines.append(f"Background Median: {analysis.get('background_median', 0):.3f} ADU")
        snr_value = analysis.get('snr', 0)
        snr_display = 'inf' if snr_value == float('inf') else f"{snr_value:.2f}"
        summary_lines.append(f"Signal-to-Noise Ratio: {snr_display}")
        summary_lines.append("")
        
        summary_lines.append("--- PHOTON/ELECTRON CONVERSION ---")
        summary_lines.append(f"Input Photons: {analysis.get('input_photons', 0):.3f}")
        summary_lines.append(f"Expected Electrons: {analysis.get('expected_electrons', 0):.3f}")
        summary_lines.append(f"Theoretical Electrons (QE): {analysis.get('theoretical_electrons_from_qe', 0):.3f}")
        summary_lines.append(f"Total Signal ADU: {analysis.get('total_signal_adu', 0):.2f}")
        summary_lines.append(f"Conversion Efficiency: {analysis.get('conversion_efficiency_adu_per_photon', 0):.3f} ADU/photon")
        summary_lines.append(f"CENTER PIXEL Signal ADU: {analysis.get('center_pixel_signal_adu', 0):.2f}")
        summary_lines.append(f"CENTER PIXEL Conversion: {analysis.get('center_pixel_conversion_efficiency', 0):.3f} ADU/photon")
        summary_lines.append("")
        
        # Add additional sensor data if available
        if additional_data:
            summary_lines.append("--- PIPELINE STAGE VALUES (CENTER PIXEL) ---")
            for key, value in additional_data.items():
                if 'center_value' in key:
                    stage_name = key.replace('_center_value', '').upper()
                    if isinstance(value, float):
                        summary_lines.append(f"{stage_name}: {value:.6f}")
                    else:
                        summary_lines.append(f"{stage_name}: {value}")
            summary_lines.append("")
        
        summary_lines.append("--- DYNAMIC RANGE ---")
        summary_lines.append(f"Max Possible ADU: {analysis.get('max_possible_adu', 65535):,}")
        summary_lines.append(f"Dynamic Range Used: {analysis.get('dynamic_range_utilization', 0)*100:.3f}%")
        summary_lines.append(f"Saturated Pixels: {analysis.get('saturated_pixels', 0):,}")
        summary_lines.append(f"Saturation: {analysis.get('saturation_percentage', 0):.4f}%")
        
        # Add sensor configuration info
        summary_lines.append("")
        summary_lines.append("--- SENSOR CONFIGURATION ---")
        summary_lines.append(f"Sensor Size: {sensor_params.get('width_px', 0)} × {sensor_params.get('height_px', 0)} pixels")
        summary_lines.append(f"Pixel Pitch: {sensor_params.get('pixel_size_um', 0)} µm")
        summary_lines.append(f"Exposure Time: {sensor_params.get('exposure_time_s', 0)} s")
        summary_lines.append(f"Auto QE Replacement: {sensor_params.get('auto_qe_replaced', True)}")
        summary_lines.append(f"Applied QE: {sensor_params.get('applied_qe', 0):.3f}")
        summary_lines.append(f"Aperture: {sensor_params.get('aperture_diameter_m', 0)} m")
        summary_lines.append(f"Focal Length: {sensor_params.get('focal_length_m', 0)} m")
        summary_lines.append(f"Angular Correction: {analysis.get('angular_correction_applied', False)}")
        if analysis.get('angular_correction_applied', False):
            summary_lines.append(f"Angular Cos Factor: {analysis.get('angular_cos_factor', 1.0):.3f}")
        
        self.pyxel_results_text.setPlainText('\n'.join(summary_lines))

    def generate_validation_images(self):
        if self.results_df.empty:
            QMessageBox.warning(self, "No Data", "No calculation results available.")
            return
        
        selected_debris_id = self.debris_selector.currentData()
        selected_iteration = self.iteration_selector.currentData()
        distance_km = self.manual_distance.value()
        
        if selected_debris_id is None or selected_iteration is None:
            QMessageBox.warning(self, "No Selection", "Please select debris ID and iteration.")
            return
        
        try:
            self.status_label.setText("Generating validation images...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            QApplication.processEvents()
            
            W = self.sensor_width_input.value()
            H = self.sensor_height_input.value()
            
            result_row = self.results_df[
                (self.results_df['Assembly_ID'] == selected_debris_id) & 
                (self.results_df['Iteration'] == selected_iteration) &
                (self.results_df['Distance_km'] == distance_km)
            ]
            
            if result_row.empty:
                QMessageBox.warning(self, "Data Not Found", 
                                   f"No data found for Debris {selected_debris_id}, Iteration {selected_iteration} at {distance_km} km")
                return
            
            row_data = result_row.iloc[0]
            
            center_x = W // 2
            center_y = H // 2
            
            oi_image = np.zeros((H, W), dtype=np.float64)
            oi_photons = row_data['OI_Photons']
            if oi_photons > 0:
                oi_image[center_y, center_x] = oi_photons
            
            al_image = np.zeros((H, W), dtype=np.float64)
            al_photons = row_data['Al_Photons']
            if al_photons > 0:
                al_image[center_y, center_x] = al_photons
            
            self.generated_images = {
                'OI': oi_image,
                'Al': al_image,
                'metadata': {
                    'debris_id': selected_debris_id,
                    'iteration': selected_iteration,
                    'distance_km': distance_km,
                    'center_pixel': (center_x, center_y),
                    'oi_photons': oi_photons,
                    'al_photons': al_photons,
                    'oi_electrons': row_data['OI_Electrons'],
                    'al_electrons': row_data['Al_Electrons'],
                    'sensor_size': (W, H),
                    'generation_params': {
                        'aperture_m': self.aperture_input.value(),
                        'exposure_s': self.exposure_input.value(),
                        'oi_qe': self.oi_qe_input.value(),
                        'oi_filter_t': self.oi_filter_input.value(),
                        'al_qe': self.al_qe_input.value(),
                        'al_filter_t': self.al_filter_input.value(),
                        'oi_wavelength_nm': self.oi_wavelength_input.value(),
                        'al_wavelength_nm': self.al_wavelength_input.value()
                    }
                }
            }
            
            self.save_image_btn.setEnabled(True)
            
            if PYXEL_AVAILABLE:
                self.update_pyxel_simulation_status()
            
            self.status_label.setText(f"Images generated: Debris {selected_debris_id}, Iter {selected_iteration}, OI={oi_photons:.1f} photons, Al={al_photons:.1f} photons at center pixel ({center_x},{center_y})")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
            print(f"Generated validation images (PHOTONS for Pyxel):")
            print(f"  Debris ID: {selected_debris_id}")
            print(f"  Iteration: {selected_iteration}")
            print(f"  Distance: {distance_km} km")
            print(f"  Center pixel: ({center_x}, {center_y})")
            print(f"  OI photons: {oi_photons:.3f}")
            print(f"  Al photons: {al_photons:.3f}")
            print(f"  OI electrons (for reference): {row_data['OI_Electrons']:.3f}")
            print(f"  Al electrons (for reference): {row_data['Al_Electrons']:.3f}")
            print(f"  Image size: {W} × {H} pixels")
            
        except Exception as e:
            QMessageBox.critical(self, "Image Generation Error", f"Error generating images: {str(e)}")
            self.status_label.setText(f"Image generation error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")

    def load_emissions_data(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Emissions CSV", "", "CSV Files (*.csv)")
        
        if not filepath:
            return
        
        try:
            self.emissions_df = pd.read_csv(filepath)
            
            required_cols = ["Assembly_ID"]
            iteration_col = "Iteration"
            emission_cols = ["OI_emissions_atomic", "AlI_1_emissions_atomic", "AlI_2_emissions_atomic"]
            
            missing_required = [col for col in required_cols if col not in self.emissions_df.columns]
            available_emissions = [col for col in emission_cols if col in self.emissions_df.columns]
            has_iterations = iteration_col in self.emissions_df.columns
            
            if missing_required:
                QMessageBox.warning(self, "Missing Columns", 
                                   f"Required columns missing: {missing_required}")
                return
            
            if not available_emissions:
                QMessageBox.warning(self, "No Emission Data", 
                                   f"No emission columns found. Expected: {emission_cols}")
                return
            
            if not has_iterations:
                self.emissions_df['Iteration'] = 1
                iteration_info = "No 'Iteration' column found - assuming all data is iteration 1"
            else:
                iteration_info = f"Found iterations: {sorted(self.emissions_df['Iteration'].unique())}"
            
            preview_text = f"Loaded: {len(self.emissions_df)} rows\n"
            preview_text += f"Columns: {list(self.emissions_df.columns)}\n"
            preview_text += f"Available emissions: {available_emissions}\n"
            preview_text += f"Assembly IDs: {sorted(self.emissions_df['Assembly_ID'].unique())}\n"
            preview_text += f"Iterations: {iteration_info}"
            
            self.data_preview.setPlainText(preview_text)
            self.calculate_btn.setEnabled(True)
            self.status_label.setText(f"Loaded {len(self.emissions_df)} emission records with iteration support")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
            print(f"Loaded emissions data: {filepath}")
            print(f"Shape: {self.emissions_df.shape}")
            print(f"Available emission columns: {available_emissions}")
            print(f"Iteration support: {has_iterations}")
            if has_iterations:
                print(f"Available iterations: {sorted(self.emissions_df['Iteration'].unique())}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading file: {str(e)}")
            self.status_label.setText(f"Error loading data: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")

    def calculate_photon_conversion(self):
        if self.emissions_df.empty:
            QMessageBox.warning(self, "No Data", "Please load emissions data first.")
            return
        
        try:
            self.status_label.setText("Calculating photon conversion...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            QApplication.processEvents()
            
            aperture_diameter_m = self.aperture_input.value()
            aperture_area_m2 = np.pi * (aperture_diameter_m / 2)**2
            exposure_time_s = self.exposure_input.value()
            
            oi_quantum_efficiency = self.oi_qe_input.value()
            oi_filter_transmission = self.oi_filter_input.value()
            al_quantum_efficiency = self.al_qe_input.value()
            al_filter_transmission = self.al_filter_input.value()
            
            oi_wavelength_nm = self.oi_wavelength_input.value()
            al_wavelength_nm = self.al_wavelength_input.value()
            
            oi_wavelength = oi_wavelength_nm * u.nm
            al_wavelength = al_wavelength_nm * u.nm
            
            oi_photon_energy_J = (h * c / oi_wavelength).to(u.J).value
            al_photon_energy_J = (h * c / al_wavelength).to(u.J).value
            
            distance_km = self.manual_distance.value()
            
            results = []
            
            for _, emission_row in self.emissions_df.iterrows():
                assembly_id = emission_row["Assembly_ID"]
                iteration = emission_row["Iteration"]
                
                oi_emission = emission_row.get("OI_emissions_atomic", 0.0)
                al1_emission = emission_row.get("AlI_1_emissions_atomic", 0.0)
                al2_emission = emission_row.get("AlI_2_emissions_atomic", 0.0)
                al_combined = al1_emission + al2_emission
                
                distance_m = distance_km * 1000
                
                if oi_emission > 0:
                    oi_results = self.calculate_emission_conversion(
                        oi_emission, distance_m, aperture_area_m2, 
                        exposure_time_s, oi_quantum_efficiency, oi_filter_transmission,
                        oi_photon_energy_J, "OI"
                    )
                else:
                    oi_results = self.get_zero_results("OI")
                
                if al_combined > 0:
                    al_results = self.calculate_emission_conversion(
                        al_combined, distance_m, aperture_area_m2,
                        exposure_time_s, al_quantum_efficiency, al_filter_transmission,
                        al_photon_energy_J, "Al_Combined"
                    )
                else:
                    al_results = self.get_zero_results("Al_Combined")
                
                result = {
                    'Assembly_ID': assembly_id,
                    'Iteration': iteration,
                    'Distance_km': distance_km,
                    'Distance_m': distance_m,
                    
                    'OI_Emission_Wsr': oi_emission,
                    'OI_Irradiance_Wm2': oi_results['irradiance'],
                    'OI_Power_W': oi_results['power'],
                    'OI_Photons': oi_results['photons'],
                    'OI_Electrons': oi_results['electrons'],
                    
                    'Al_Emission_Wsr': al_combined,
                    'Al_Irradiance_Wm2': al_results['irradiance'],
                    'Al_Power_W': al_results['power'],
                    'Al_Photons': al_results['photons'],
                    'Al_Electrons': al_results['electrons'],
                    
                    'Aperture_m': aperture_diameter_m,
                    'Focal_Length_m': self.focal_length_input.value(),
                    'Pixel_Pitch_um': self.pixel_pitch_input.value(),
                    'Sensor_Width_px': self.sensor_width_input.value(),
                    'Sensor_Height_px': self.sensor_height_input.value(),
                    'Exposure_s': exposure_time_s,
                    'OI_QE': oi_quantum_efficiency,
                    'OI_Filter_T': oi_filter_transmission,
                    'Al_QE': al_quantum_efficiency,
                    'Al_Filter_T': al_filter_transmission,
                    'OI_Wavelength_nm': oi_wavelength_nm,
                    'Al_Wavelength_nm': al_wavelength_nm
                }
                
                results.append(result)
            
            self.results_df = pd.DataFrame(results)
            self.update_results_display()
            self.export_results_btn.setEnabled(True)
            
            self.generate_image_btn.setEnabled(True)
            
            self.status_label.setText(f"Calculation complete: {len(results)} results generated with wavelength-specific parameters")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
            print(f"Generated {len(results)} calculation results with wavelength-specific parameters")
            
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", f"Error in calculation: {str(e)}")
            self.status_label.setText(f"Calculation error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")

    def calculate_emission_conversion(self, emission_intensity, distance_m, aperture_area_m2,
                                     exposure_time_s, quantum_efficiency, filter_transmission,
                                     photon_energy_J, emission_type):
        
        irradiance_Wm2 = emission_intensity / (distance_m**2)
        
        power_W = irradiance_Wm2 * aperture_area_m2 * filter_transmission
        
        photons = (power_W * exposure_time_s) / photon_energy_J
        
        electrons = photons * quantum_efficiency
        
        if self.apply_angular_correction.isChecked():
            cos_factor = self.angular_cos_factor.value()
            photons *= cos_factor
            electrons *= cos_factor
        
        return {
            'irradiance': irradiance_Wm2,
            'power': power_W,
            'photons': photons,
            'electrons': electrons
        }

    def get_zero_results(self, emission_type):
        return {
            'irradiance': 0.0,
            'power': 0.0,
            'photons': 0.0,
            'electrons': 0.0
        }

    def update_results_display(self):
        if self.results_df.empty:
            return
        
        self.results_table.clear()
        self.results_table.setRowCount(len(self.results_df))
        self.results_table.setColumnCount(len(self.results_df.columns))
        self.results_table.setHorizontalHeaderLabels(self.results_df.columns.tolist())
        
        for row in range(len(self.results_df)):
            for col in range(len(self.results_df.columns)):
                value = self.results_df.iloc[row, col]
                if isinstance(value, float):
                    if value == 0.0:
                        item_text = "0.000"
                    elif abs(value) < 1e-10:
                        item_text = f"{value:.2e}"
                    elif abs(value) < 1e-6:
                        item_text = f"{value:.3e}"
                    elif abs(value) < 0.001:
                        item_text = f"{value:.9f}"
                    elif abs(value) < 1:
                        item_text = f"{value:.6f}"
                    else:
                        item_text = f"{value:.3f}"
                else:
                    item_text = str(value)
                
                item = QTableWidgetItem(item_text)
                self.results_table.setItem(row, col, item)
        
        self.results_table.resizeColumnsToContents()
        
        self.update_summary_statistics()
        
        self.update_image_selectors()
        
        if PYXEL_AVAILABLE:
            self.update_pyxel_selectors()

    def update_image_selectors(self):
        if self.results_df.empty:
            return
        
        self.debris_selector.clear()
        debris_ids = sorted(self.results_df['Assembly_ID'].unique())
        for debris_id in debris_ids:
            self.debris_selector.addItem(f"Debris {debris_id}", debris_id)
        
        self.generate_image_btn.setEnabled(len(debris_ids) > 0)
        
        if len(debris_ids) > 0:
            self.debris_selector.setCurrentIndex(0)

    def update_summary_statistics(self):
        if self.results_df.empty:
            return
        
        summary = []
        summary.append(f"Total calculations: {len(self.results_df)}")
        summary.append(f"Debris objects: {self.results_df['Assembly_ID'].nunique()}")
        
        if 'Iteration' in self.results_df.columns:
            unique_iterations = sorted(self.results_df['Iteration'].unique())
            summary.append(f"Iterations: {unique_iterations}")
        
        distance = self.manual_distance.value()
        summary.append(f"Distance: {distance:.4f} km")
        
        summary.append("")
        
        oi_electrons = self.results_df['OI_Electrons']
        summary.append(f"OI Electrons - Min: {oi_electrons.min():.2e}, Max: {oi_electrons.max():.2e}, Mean: {oi_electrons.mean():.2e}")
        
        al_electrons = self.results_df['Al_Electrons']
        summary.append(f"Al Electrons - Min: {al_electrons.min():.2e}, Max: {al_electrons.max():.2e}, Mean: {al_electrons.mean():.2e}")
        
        high_oi = len(oi_electrons[oi_electrons > 1000])
        high_al = len(al_electrons[al_electrons > 1000])
        summary.append(f"High signal cases (>1000 e⁻): OI={high_oi}, Al={high_al}")
        
        self.summary_text.setPlainText('\n'.join(summary))

    def export_results(self):
        if self.results_df.empty:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"photon_validation_results_{timestamp}.csv"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Results", default_filename, "CSV Files (*.csv)")
        
        if not filepath:
            return
        
        try:
            self.results_df.to_csv(filepath, index=False)
            
            metadata = {
                'export_timestamp': timestamp,
                'total_results': len(self.results_df),
                'iterations_supported': 'Iteration' in self.results_df.columns,
                'unique_iterations': sorted(self.results_df['Iteration'].unique().tolist()) if 'Iteration' in self.results_df.columns else [1],
                'sensor_parameters': {
                    'aperture_diameter_m': self.aperture_input.value(),
                    'focal_length_m': self.focal_length_input.value(),
                    'pixel_pitch_um': self.pixel_pitch_input.value(),
                    'sensor_width_px': self.sensor_width_input.value(),
                    'sensor_height_px': self.sensor_height_input.value(),
                    'exposure_time_s': self.exposure_input.value(),
                    'oi_quantum_efficiency': self.oi_qe_input.value(),
                    'oi_filter_transmission': self.oi_filter_input.value(),
                    'al_quantum_efficiency': self.al_qe_input.value(),
                    'al_filter_transmission': self.al_filter_input.value(),
                    'oi_wavelength_nm': self.oi_wavelength_input.value(),
                    'al_wavelength_nm': self.al_wavelength_input.value()
                },
                'angular_correction_applied': self.apply_angular_correction.isChecked(),
                'column_descriptions': {
                    'Assembly_ID': 'Debris object identifier',
                    'Iteration': 'Iteration number for this debris scenario',
                    'Distance_km': 'Distance from sensor in kilometers',
                    'OI_Electrons': 'Electron count from OI emission',
                    'Al_Electrons': 'Electron count from Al emission',
                    'OI_Photons': 'Photon count from OI emission',
                    'Al_Photons': 'Photon count from Al emission',
                    'OI_Irradiance_Wm2': 'Irradiance at sensor from OI emission',
                    'Al_Irradiance_Wm2': 'Irradiance at sensor from Al emission',
                    'OI_QE': 'Quantum efficiency for OI wavelength (777.3nm)',
                    'OI_Filter_T': 'Filter transmission for OI wavelength',
                    'Al_QE': 'Quantum efficiency for Al wavelength (395.0nm)',
                    'Al_Filter_T': 'Filter transmission for Al wavelength'
                }
            }
            
            metadata_path = filepath.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            QMessageBox.information(self, "Export Complete", 
                                   f"Results exported successfully!\n\n"
                                   f"Data: {filepath}\n"
                                   f"Metadata: {metadata_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting: {str(e)}")

    def clear_results(self):
        self.results_df = pd.DataFrame()
        self.results_table.clear()
        self.summary_text.clear()
        self.export_results_btn.setEnabled(False)
        self.generate_image_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        self.generated_images = {}
        self.debris_selector.clear()
        self.iteration_selector.clear()
        self.plot_widget.clear()
        self.image_widget.clear()
        
        if PYXEL_AVAILABLE:
            self.clear_all_pyxel_results()
            self.update_pyxel_simulation_status()
            self.pyxel_debris_selector.clear()
            self.pyxel_iteration_selector.clear()

    def update_calculated_parameters(self):
        try:
            aperture_diameter_m = self.aperture_input.value()
            aperture_area_m2 = np.pi * (aperture_diameter_m / 2)**2
            self.aperture_area_label.setText(f"{aperture_area_m2:.6f} m²")
            
            pixel_pitch_um = self.pixel_pitch_input.value()
            pixel_size_mm = pixel_pitch_um / 1000.0
            self.pixel_size_label.setText(f"{pixel_size_mm:.3f} mm")
            
            sensor_width_px = self.sensor_width_input.value()
            sensor_height_px = self.sensor_height_input.value()
            sensor_width_mm = sensor_width_px * pixel_size_mm
            sensor_height_mm = sensor_height_px * pixel_size_mm
            self.sensor_size_label.setText(f"{sensor_width_mm:.1f} × {sensor_height_mm:.1f} mm")
            
            focal_length_m = self.focal_length_input.value()
            if focal_length_m > 0:
                sensor_width_m = sensor_width_mm / 1000.0
                sensor_height_m = sensor_height_mm / 1000.0
                
                h_fov_rad = 2 * np.arctan(sensor_width_m / (2 * focal_length_m))
                v_fov_rad = 2 * np.arctan(sensor_height_m / (2 * focal_length_m))
                
                h_fov_deg = np.degrees(h_fov_rad)
                v_fov_deg = np.degrees(v_fov_rad)
                
                self.fov_label.setText(f"{h_fov_deg:.2f}° × {v_fov_deg:.2f}°")
            else:
                self.fov_label.setText("Invalid focal length")
                
        except Exception as e:
            print(f"Error updating calculated parameters: {e}")

    def load_yaml_configuration(self):
        if not PYXEL_AVAILABLE:
            QMessageBox.warning(self, "Pyxel Not Available", "Pyxel is not installed. Cannot load configuration.")
            return
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Pyxel YAML Configuration", "", "YAML Files (*.yaml *.yml);;All Files (*.*)")
        
        if not filepath:
            return
        
        try:
            self.pyxel_config = pyxel.load(filepath)
            
            with open(filepath, 'r') as f:
                yaml_content = f.read()
            
            self.yaml_preview.setPlainText(yaml_content)
            self.yaml_status_label.setText(f"✓ Loaded: {os.path.basename(filepath)}")
            self.yaml_status_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.update_pyxel_simulation_status()
            
            print(f"Loaded Pyxel configuration: {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "YAML Load Error", f"Error loading YAML configuration:\n{str(e)}")
            self.yaml_status_label.setText(f"✗ Error loading YAML: {str(e)}")
            self.yaml_status_label.setStyleSheet("color: red; font-weight: bold;")

    def update_pyxel_simulation_status(self):
        has_config = self.pyxel_config is not None
        has_data = not self.results_df.empty
        
        if has_config and has_data:
            self.generate_and_simulate_btn.setEnabled(True)
            self.run_pyxel_btn.setEnabled(len(self.generated_images) > 0)
            self.sim_info_label.setText("✓ Ready - Select scenario and Generate && Simulate")
            self.sim_info_label.setStyleSheet("color: green; font-weight: bold;")
        elif not has_config:
            self.generate_and_simulate_btn.setEnabled(False)
            self.run_pyxel_btn.setEnabled(False)
            self.sim_info_label.setText("Load YAML configuration first")
            self.sim_info_label.setStyleSheet("color: orange; font-style: italic;")
        elif not has_data:
            self.generate_and_simulate_btn.setEnabled(False)
            self.run_pyxel_btn.setEnabled(False)
            self.sim_info_label.setText("Calculate photon conversion results first")
            self.sim_info_label.setStyleSheet("color: orange; font-style: italic;")

    def clear_all_pyxel_results(self):
        if not self.pyxel_results_history:
            QMessageBox.information(self, "No Results", "No results to clear.")
            return
        
        reply = QMessageBox.question(self, "Clear All Results", 
                                   f"Clear all {len(self.pyxel_results_history)} simulation results?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.pyxel_results_history = []
            self.current_pyxel_result = None
            
            self.pyxel_image_widget.clear()
            self.pyxel_results_text.setPlainText("No Pyxel simulation results yet")
            self.current_scenario_label.setText("No simulation loaded")
            
            self.update_pyxel_history_table()
            
            self.save_current_csv_btn.setEnabled(False)
            self.save_output_fits_btn.setEnabled(False)
            self.save_all_csv_btn.setEnabled(False)
            self.load_selected_result_btn.setEnabled(False)
            self.delete_selected_btn.setEnabled(False)
            
            self.status_label.setText("All Pyxel results cleared")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def on_history_selection_changed(self):
        selected_rows = self.pyxel_history_table.selectionModel().selectedRows()
        has_selection = len(selected_rows) > 0
        
        self.load_selected_result_btn.setEnabled(has_selection)
        self.delete_selected_btn.setEnabled(has_selection)

    def change_pyxel_colormap(self, colormap_name):
        try:
            self.pyxel_image_widget.setPredefinedGradient(colormap_name)
        except Exception as e:
            print(f"Error changing colormap: {e}")

    def auto_levels_pyxel_image(self):
        try:
            self.pyxel_image_widget.autoLevels()
        except Exception as e:
            print(f"Error auto-leveling image: {e}")

    def run_pyxel_simulation(self):
        if not PYXEL_AVAILABLE:
            QMessageBox.warning(self, "Pyxel Not Available", "Pyxel is not installed.")
            return
        
        if self.pyxel_config is None:
            QMessageBox.warning(self, "No Configuration", "Please load a YAML configuration first.")
            return
        
        if not self.generated_images:
            QMessageBox.warning(self, "No Images", "Please generate validation images first.")
            return
        
        try:
            emission_type = self.emission_type_selector.currentData()
            
            if emission_type not in self.generated_images:
                QMessageBox.warning(self, "Image Not Available", f"No {emission_type} image generated.")
                return
            
            # Check if auto replace QE is enabled
            if self.auto_replace_qe_check.isChecked():
                qe_updated = self.update_pyxel_detector_qe(emission_type)
                if not qe_updated:
                    QMessageBox.warning(self, "QE Update Failed", 
                                       "Could not update detector quantum efficiency. Proceeding with default QE.")
            else:
                print(f"Auto replace QE is disabled - using YAML configuration QE values")
                qe_updated = False
            
            image_data = self.generated_images[emission_type]
            
            temp_dir = "pyxel_temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            metadata = self.generated_images['metadata']
            simulation_id = self.generate_unique_simulation_id()
            
            progress_dialog = QProgressDialog("Running Pyxel simulation...", "Cancel", 0, 0, self)
            progress_dialog.setWindowTitle("Pyxel Simulation")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(300)
            progress_dialog.setCancelButton(None)
            progress_dialog.show()
            
            with QMutexLocker(self._simulation_mutex):
                self._current_worker = PyxelSimulationWorker(self.pyxel_config, image_data, "", simulation_id)
                self._current_worker.finished.connect(lambda result: self.handle_pyxel_complete(result, progress_dialog))
                self._current_worker.error.connect(lambda error: self.handle_pyxel_error(error, progress_dialog, simulation_id))
                self._current_worker.progress.connect(lambda msg: progress_dialog.setLabelText(msg))
                self._current_worker.start()
            
            self.run_pyxel_btn.setEnabled(False)
            self.status_label.setText(f"Running Pyxel simulation on {emission_type} image (Debris {metadata['debris_id']}, Iter {metadata['iteration']})...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"Error starting Pyxel simulation:\n{str(e)}")

    def handle_pyxel_complete(self, result_datatree, progress_dialog):
        progress_dialog.close()
        
        try:
            with QMutexLocker(self._simulation_mutex):
                self._current_worker = None
            
            self.pyxel_results = result_datatree
            
            if "bucket" not in result_datatree:
                raise ValueError("No bucket data in Pyxel results")
            
            bucket_ds = result_datatree["bucket"].to_dataset()
            
            if "image" not in bucket_ds:
                raise ValueError("No image data in Pyxel bucket results")
            
            self.pyxel_image_cube = bucket_ds["image"].values
            if "time" in bucket_ds.coords:
                self.pyxel_time_points = bucket_ds["time"].values
            else:
                self.pyxel_time_points = np.array([0.0])
            
            if len(self.pyxel_image_cube.shape) == 3 and self.pyxel_image_cube.shape[0] > 0:
                final_image = self.pyxel_image_cube[-1]
            elif len(self.pyxel_image_cube.shape) == 2:
                final_image = self.pyxel_image_cube
            else:
                final_image = self.pyxel_image_cube[0] if self.pyxel_image_cube.shape[0] > 0 else np.zeros((100, 100))
            
            self.pyxel_image_widget.setImage(final_image, autoRange=True, autoLevels=True)
            
            analysis_results = self.analyze_pyxel_results(final_image, bucket_ds)
            
            self.pyxel_analysis_results = analysis_results
            
            self.update_pyxel_results_display(analysis_results)
            
            self.save_current_csv_btn.setEnabled(True)
            self.save_output_fits_btn.setEnabled(True)
            
            self.status_label.setText("✓ Pyxel simulation completed successfully")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            
            metadata = self.generated_images.get('metadata', {})
            print(f"Pyxel simulation complete:")
            print(f"  Debris: {metadata.get('debris_id', 'N/A')}, Iteration: {metadata.get('iteration', 'N/A')}")
            print(f"  Output shape: {final_image.shape}")
            print(f"  ADU range: {final_image.min():.2f} to {final_image.max():.2f}")
            print(f"  Center Pixel ADU: {analysis_results.get('center_pixel_adu', 0):.2f}")
            print(f"  Peak ADU: {analysis_results.get('peak_adu', 0):.2f}")
            print(f"  SNR: {analysis_results.get('snr', 0):.2f}")
            
        except Exception as e:
            error_msg = f"Error processing Pyxel results:\n{str(e)}"
            QMessageBox.critical(self, "Results Error", error_msg)
            self.status_label.setText("Error processing Pyxel results")
            self.status_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
        
        finally:
            self.run_pyxel_btn.setEnabled(True)

    def save_image_files(self):
        if not self.generated_images or 'OI' not in self.generated_images:
            QMessageBox.warning(self, "No Images", "No images generated yet. Generate images first.")
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Images", "")
        if not save_dir:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = self.generated_images['metadata']
            
            base_filename = f"validation_debris_{metadata['debris_id']}_iter_{metadata['iteration']}_dist_{metadata['distance_km']:.4f}km_{timestamp}"
            
            oi_image = self.generated_images['OI']
            fits.writeto(os.path.join(save_dir, f"{base_filename}_OI_photons.fits"), 
                        oi_image.astype(np.float32), overwrite=True)
            
            al_image = self.generated_images['Al']
            fits.writeto(os.path.join(save_dir, f"{base_filename}_Al_photons.fits"), 
                        al_image.astype(np.float32), overwrite=True)
            
            np.save(os.path.join(save_dir, f"{base_filename}_OI_photons.npy"), oi_image)
            np.save(os.path.join(save_dir, f"{base_filename}_Al_photons.npy"), al_image)
            
            np.savetxt(os.path.join(save_dir, f"{base_filename}_OI_photon_map.csv"), 
                      oi_image, delimiter=",", fmt='%.6e')
            np.savetxt(os.path.join(save_dir, f"{base_filename}_Al_photon_map.csv"), 
                      al_image, delimiter=",", fmt='%.6e')
            
            full_metadata = {
                'validation_type': 'photon_input_for_pyxel_with_dynamic_qe',
                'generation_timestamp': timestamp,
                'debris_id': metadata['debris_id'],
                'iteration': metadata['iteration'],
                'distance_km': metadata['distance_km'],
                'center_pixel_position': metadata['center_pixel'],
                'sensor_dimensions': metadata['sensor_size'],
                'photon_counts': {
                    'oi_photons': metadata['oi_photons'],
                    'al_photons': metadata['al_photons']
                },
                'electron_counts_reference': {
                    'oi_electrons': metadata['oi_electrons'],
                    'al_electrons': metadata['al_electrons']
                },
                'sensor_parameters': metadata['generation_params'],
                'file_descriptions': {
                    f'{base_filename}_OI_photons.fits': 'OI FITS image with PHOTONS (Pyxel-ready)',
                    f'{base_filename}_Al_photons.fits': 'Al FITS image with PHOTONS (Pyxel-ready)',
                    f'{base_filename}_OI_photons.npy': 'OI NumPy array with photons',
                    f'{base_filename}_Al_photons.npy': 'Al NumPy array with photons',
                    f'{base_filename}_OI_photon_map.csv': 'OI photon count map',
                    f'{base_filename}_Al_photon_map.csv': 'Al photon count map'
                },
                'pyxel_ready': True,
                'notes': f'FITS files contain PHOTONS (correct input for Pyxel) - QE will be applied dynamically based on emission type selection'
            }
            
            with open(os.path.join(save_dir, f"{base_filename}_metadata.json"), 'w') as f:
                json.dump(full_metadata, f, indent=2)
            
            QMessageBox.information(self, "Images Saved", 
                                   f"Validation images saved successfully!\n\n"
                                   f"Location: {save_dir}\n"
                                   f"Base filename: {base_filename}\n\n"
                                   f"Configuration:\n"
                                   f"• Debris: {metadata['debris_id']}\n"
                                   f"• Iteration: {metadata['iteration']}\n"
                                   f"• Distance: {metadata['distance_km']:.4f} km\n"
                                   f"• OI photons: {metadata['oi_photons']:.3f}\n"
                                   f"• Al photons: {metadata['al_photons']:.3f}\n"
                                   f"• Expected OI electrons: {metadata['oi_electrons']:.3f}\n"
                                   f"• Expected Al electrons: {metadata['al_electrons']:.3f}\n\n"
                                   f"Files saved:\n"
                                   f"• FITS images with PHOTONS (Pyxel-ready)\n"
                                   f"• NumPy arrays (.npy)\n"
                                   f"• Photon maps (.csv)\n"
                                   f"• Metadata (.json)")
            
            print(f"Saved validation images to: {save_dir}")
            print(f"Base filename: {base_filename}")
            print(f"IMPORTANT: Images contain PHOTONS, not electrons (correct for Pyxel)")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving images: {str(e)}")

    def save_current_pyxel_csv(self):
        if self.current_pyxel_result is None:
            QMessageBox.warning(self, "No Current Result", "No current simulation result to save.")
            return
        
        scenario = self.current_pyxel_result['scenario_metadata']
        analysis = self.current_pyxel_result['analysis_results']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"pyxel_analysis_debris_{scenario['debris_id']}_iter_{scenario['iteration']}_dist_{scenario['distance_km']:.4f}km_{scenario['emission_type']}_{timestamp}.csv"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Current Pyxel Analysis", default_filename, "CSV Files (*.csv)")
        
        if not filepath:
            return
        
        try:
            self.export_single_result_csv(self.current_pyxel_result, filepath)
            
            QMessageBox.information(self, "CSV Saved", 
                                   f"Current result saved successfully!\n\n"
                                   f"Scenario: Debris {scenario['debris_id']}, Iter {scenario['iteration']}\n"
                                   f"File: {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving CSV: {str(e)}")

    def export_single_result_csv(self, result, filepath):
        scenario = result['scenario_metadata']
        analysis = result['analysis_results']
        
        csv_data = []
        
        csv_data.append(['Section', 'Parameter', 'Value', 'Unit'])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Emission Type', analysis.get('emission_type', ''), ''])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Debris ID', analysis.get('debris_id', ''), ''])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Iteration', analysis.get('iteration', ''), ''])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Distance', analysis.get('distance_km', ''), 'km'])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Applied QE', analysis.get('applied_quantum_efficiency', ''), 'fraction'])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Auto QE Replaced', analysis.get('auto_qe_replaced', ''), 'boolean'])
        csv_data.append(['PYXEL SIMULATION ANALYSIS', 'Analysis Time', analysis.get('analysis_timestamp', ''), ''])
        csv_data.append(['', '', '', ''])
        
        csv_data.append(['IMAGE PROPERTIES', 'Image Shape', f"({analysis.get('image_shape', [0, 0])[0]}, {analysis.get('image_shape', [0, 0])[1]})", 'pixels'])
        csv_data.append(['IMAGE PROPERTIES', 'Time Frames', analysis.get('time_frames', 1), 'frames'])
        csv_data.append(['IMAGE PROPERTIES', 'Total Pixels', f"{analysis.get('total_pixels', 0):,}", 'pixels'])
        csv_data.append(['IMAGE PROPERTIES', 'Non-zero Pixels', f"{analysis.get('non_zero_pixels', 0):,}", 'pixels'])
        csv_data.append(['IMAGE PROPERTIES', 'Fill Factor', f"{analysis.get('fill_factor', 0)*100:.3f}%", 'percent'])
        csv_data.append(['', '', '', ''])
        
        csv_data.append(['ADU STATISTICS', 'Peak ADU', f"{analysis.get('peak_adu', 0):.2f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Peak Location', f"({analysis.get('peak_row', 0)}, {analysis.get('peak_col', 0)})", 'pixels'])
        csv_data.append(['ADU STATISTICS', 'CENTER PIXEL ADU', f"{analysis.get('center_pixel_adu', 0):.2f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Center Pixel Location', f"({analysis.get('center_pixel_row', 0)}, {analysis.get('center_pixel_col', 0)})", 'pixels'])
        csv_data.append(['ADU STATISTICS', 'Total ADU Count', f"{analysis.get('total_adu_count', 0):.0f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Mean ADU', f"{analysis.get('mean_adu', 0):.3f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Median ADU', f"{analysis.get('median_adu', 0):.3f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Std ADU', f"{analysis.get('std_adu', 0):.3f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Min ADU', f"{analysis.get('min_adu', 0):.2f}", 'ADU'])
        csv_data.append(['ADU STATISTICS', 'Max ADU', f"{analysis.get('max_adu', 0):.2f}", 'ADU'])
        csv_data.append(['', '', '', ''])
        
        csv_data.append(['BACKGROUND && NOISE', 'Background Mean', f"{analysis.get('background_mean', 0):.3f}", 'ADU'])
        csv_data.append(['BACKGROUND && NOISE', 'Background Std', f"{analysis.get('background_std', 0):.3f}", 'ADU'])
        csv_data.append(['BACKGROUND && NOISE', 'Background Median', f"{analysis.get('background_median', 0):.3f}", 'ADU'])
        snr_value = analysis.get('snr', 0)
        snr_display = 'inf' if snr_value == float('inf') else f"{snr_value:.2f}"
        csv_data.append(['BACKGROUND && NOISE', 'Signal-to-Noise Ratio', snr_display, 'ratio'])
        csv_data.append(['', '', '', ''])
        
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'Input Photons', f"{analysis.get('input_photons', 0):.3f}", 'photons'])
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'Expected Electrons', f"{analysis.get('expected_electrons', 0):.3f}", 'electrons'])
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'Theoretical Electrons QE', f"{analysis.get('theoretical_electrons_from_qe', 0):.3f}", 'electrons'])
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'Total Signal ADU', f"{analysis.get('total_signal_adu', 0):.2f}", 'ADU'])
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'Conversion Efficiency', f"{analysis.get('conversion_efficiency_adu_per_photon', 0):.3f}", 'ADU/photon'])
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'CENTER PIXEL Signal ADU', f"{analysis.get('center_pixel_signal_adu', 0):.2f}", 'ADU'])
        csv_data.append(['PHOTON ELECTRON CONVERSION', 'CENTER PIXEL Conversion', f"{analysis.get('center_pixel_conversion_efficiency', 0):.3f}", 'ADU/photon'])
        csv_data.append(['', '', '', ''])
        
        additional_data = analysis.get('additional_sensor_data', {})
        if additional_data:
            csv_data.append(['ADDITIONAL SENSOR DATA', '', '', ''])
            for key, value in additional_data.items():
                if isinstance(value, float):
                    csv_data.append(['ADDITIONAL SENSOR DATA', key, f"{value:.3f}", ''])
                else:
                    csv_data.append(['ADDITIONAL SENSOR DATA', key, str(value), ''])
            csv_data.append(['', '', '', ''])
        
        csv_data.append(['DYNAMIC RANGE', 'Max Possible ADU', f"{analysis.get('max_possible_adu', 65535):,}", 'ADU'])
        csv_data.append(['DYNAMIC RANGE', 'Dynamic Range Used', f"{analysis.get('dynamic_range_utilization', 0)*100:.3f}%", 'percent'])
        csv_data.append(['DYNAMIC RANGE', 'Saturated Pixels', f"{analysis.get('saturated_pixels', 0):,}", 'pixels'])
        csv_data.append(['DYNAMIC RANGE', 'Saturation', f"{analysis.get('saturation_percentage', 0):.4f}%", 'percent'])
        csv_data.append(['', '', '', ''])
        
        sensor_params = analysis.get('sensor_parameters', {})
        csv_data.append(['SENSOR PARAMETERS', 'Sensor Width', sensor_params.get('width_px', 0), 'pixels'])
        csv_data.append(['SENSOR PARAMETERS', 'Sensor Height', sensor_params.get('height_px', 0), 'pixels'])
        csv_data.append(['SENSOR PARAMETERS', 'Pixel Size', sensor_params.get('pixel_size_um', 0), 'micrometers'])
        csv_data.append(['SENSOR PARAMETERS', 'Exposure Time', sensor_params.get('exposure_time_s', 0), 'seconds'])
        csv_data.append(['SENSOR PARAMETERS', 'Applied QE', sensor_params.get('applied_qe', 0), 'fraction'])
        csv_data.append(['SENSOR PARAMETERS', 'Auto QE Replaced', sensor_params.get('auto_qe_replaced', True), 'boolean'])
        csv_data.append(['SENSOR PARAMETERS', 'Aperture Diameter', sensor_params.get('aperture_diameter_m', 0), 'meters'])
        csv_data.append(['SENSOR PARAMETERS', 'Focal Length', sensor_params.get('focal_length_m', 0), 'meters'])
        csv_data.append(['SENSOR PARAMETERS', 'Angular Correction Applied', analysis.get('angular_correction_applied', False), 'boolean'])
        csv_data.append(['SENSOR PARAMETERS', 'Angular Cos Factor', analysis.get('angular_cos_factor', 1.0), 'factor'])
        
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

    def save_output_fits(self):
        if self.current_pyxel_result is None:
            QMessageBox.warning(self, "No Current Result", "No current simulation result to save.")
            return
        
        scenario = self.current_pyxel_result['scenario_metadata']
        final_image = self.current_pyxel_result['final_image']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"pyxel_output_debris_{scenario['debris_id']}_iter_{scenario['iteration']}_dist_{scenario['distance_km']:.4f}km_{scenario['emission_type']}_{timestamp}.fits"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Pyxel Output FITS", default_filename, "FITS Files (*.fits)")
        
        if not filepath:
            return
        
        try:
            header = fits.Header()
            header['OBJECT'] = f"Pyxel_Output_Debris_{scenario['debris_id']}"
            header['DEBRIS'] = scenario['debris_id']
            header['ITER'] = scenario['iteration']
            header['DISTANCE'] = (scenario['distance_km'], 'Distance in km')
            header['EMISSION'] = scenario['emission_type']
            header['INPHOTON'] = (scenario['input_photons'], 'Input photons')
            header['EXPELEC'] = (scenario['expected_electrons'], 'Expected electrons')
            header['APPLIEDQE'] = (scenario['applied_qe'], 'Applied quantum efficiency')
            header['AUTOQE'] = (scenario.get('auto_qe_replaced', True), 'Auto QE replacement used')
            header['PIXELX'] = scenario['center_pixel'][0]
            header['PIXELY'] = scenario['center_pixel'][1]
            header['SIMTIME'] = scenario['generation_timestamp']
            header['CREATOR'] = 'Pyxel Validation Tool'
            header['COMMENT'] = 'Output from Pyxel detector simulation'
            
            hdu = fits.PrimaryHDU(data=final_image.astype(np.float32), header=header)
            hdu.writeto(filepath, overwrite=True)
            
            qe_status = "Auto QE" if scenario.get('auto_qe_replaced', True) else "YAML QE"
            
            QMessageBox.information(self, "FITS Saved", 
                                   f"Pyxel output image saved successfully!\n\n"
                                   f"Scenario: Debris {scenario['debris_id']}, Iter {scenario['iteration']}\n"
                                   f"QE Mode: {qe_status} ({scenario['applied_qe']:.3f})\n"
                                   f"Size: {final_image.shape}\n"
                                   f"ADU Range: {final_image.min():.2f} to {final_image.max():.2f}\n"
                                   f"File: {filepath}\n\n"
                                   f"💡 Use this FITS file to check if noise models are working correctly.")
            
            print(f"Saved Pyxel output FITS: {filepath}")
            print(f"  Image shape: {final_image.shape}")
            print(f"  ADU range: {final_image.min():.2f} to {final_image.max():.2f}")
            print(f"  Center pixel ADU: {final_image[final_image.shape[0]//2, final_image.shape[1]//2]:.2f}")
            print(f"  QE Mode: {qe_status}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving FITS: {str(e)}")

    def save_all_pyxel_csvs(self):
        if not self.pyxel_results_history:
            QMessageBox.warning(self, "No Results", "No simulation results to export.")
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory for All Pyxel CSVs", "")
        if not save_dir:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            saved_files = []
            for i, result in enumerate(self.pyxel_results_history):
                scenario = result['scenario_metadata']
                
                filename = f"pyxel_analysis_debris_{scenario['debris_id']}_iter_{scenario['iteration']}_dist_{scenario['distance_km']:.4f}km_{scenario['emission_type']}_{timestamp}.csv"
                filepath = os.path.join(save_dir, filename)
                
                self.export_single_result_csv(result, filepath)
                saved_files.append(filename)
            
            summary_filepath = os.path.join(save_dir, f"pyxel_summary_all_results_{timestamp}.csv")
            self.create_summary_csv(self.pyxel_results_history, summary_filepath)
            saved_files.append(os.path.basename(summary_filepath))
            
            QMessageBox.information(self, "All CSVs Saved", 
                                   f"Saved {len(self.pyxel_results_history)} individual CSV files and 1 summary CSV.\n\n"
                                   f"Location: {save_dir}\n\n"
                                   f"Files: {', '.join(saved_files[:3])}" + 
                                   (f" and {len(saved_files)-3} more..." if len(saved_files) > 3 else ""))
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error saving all CSVs: {str(e)}")

    def create_summary_csv(self, results_list, filepath):
        summary_data = [
            ['Debris_ID', 'Iteration', 'Distance_km', 'Emission_Type', 'Auto_QE_Replaced', 'Input_Photons', 'Expected_Electrons', 
             'Center_Pixel_ADU', 'Peak_ADU', 'Total_ADU_Count', 'SNR', 'Conversion_Efficiency_ADU_per_Photon', 
             'Background_Mean', 'Applied_QE', 'Dynamic_Range_Used_Percent', 'Saturated_Pixels', 'Analysis_Timestamp']
        ]
        
        for result in results_list:
            scenario = result['scenario_metadata']
            analysis = result['analysis_results']
            
            row = [
                scenario.get('debris_id', ''),
                scenario.get('iteration', ''),
                scenario.get('distance_km', ''),
                scenario.get('emission_type', ''),
                scenario.get('auto_qe_replaced', True),
                analysis.get('input_photons', 0),
                analysis.get('expected_electrons', 0),
                analysis.get('center_pixel_adu', 0),
                analysis.get('peak_adu', 0),
                analysis.get('total_adu_count', 0),
                analysis.get('snr', 0),
                analysis.get('conversion_efficiency_adu_per_photon', 0),
                analysis.get('background_mean', 0),
                analysis.get('applied_quantum_efficiency', 0),
                analysis.get('dynamic_range_utilization', 0) * 100,
                analysis.get('saturated_pixels', 0),
                analysis.get('analysis_timestamp', '')
            ]
            summary_data.append(row)
        
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(summary_data)

    def load_selected_result(self):
        selected_rows = self.pyxel_history_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        if row < len(self.pyxel_results_history):
            self.current_pyxel_result = self.pyxel_results_history[row]
            
            self.display_current_result()
            
            scenario = self.current_pyxel_result['scenario_metadata']
            qe_status = "Auto QE" if scenario.get('auto_qe_replaced', True) else "YAML QE"
            self.current_scenario_label.setText(f"Debris {scenario['debris_id']}, Iter {scenario['iteration']}, {scenario['distance_km']:.4f}km, {scenario['emission_type']} ({qe_status})")
            
            self.save_current_csv_btn.setEnabled(True)
            self.save_output_fits_btn.setEnabled(True)
            
            print(f"Loaded result: Debris {scenario['debris_id']}, Iter {scenario['iteration']}, QE mode: {qe_status}")

    def delete_selected_result(self):
        selected_rows = self.pyxel_history_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        if row < len(self.pyxel_results_history):
            result = self.pyxel_results_history[row]
            scenario = result['scenario_metadata']
            
            reply = QMessageBox.question(self, "Delete Result", 
                                       f"Delete result for Debris {scenario['debris_id']}, Iter {scenario['iteration']}?",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                del self.pyxel_results_history[row]
                
                if self.current_pyxel_result == result:
                    self.current_pyxel_result = None
                    self.pyxel_image_widget.clear()
                    self.pyxel_results_text.setPlainText("No simulation results")
                    self.current_scenario_label.setText("No simulation loaded")
                    self.save_current_csv_btn.setEnabled(False)
                    self.save_output_fits_btn.setEnabled(False)
                
                self.update_pyxel_history_table()
                
                self.save_all_csv_btn.setEnabled(len(self.pyxel_results_history) > 0)

    def update_validation_plot(self):
        if self.results_df.empty:
            QMessageBox.warning(self, "No Data", "No results to plot.")
            return
        
        plot_type = self.plot_type.currentData()
        
        if plot_type in ["oi_image", "al_image"]:
            self.plot_stack.setCurrentIndex(1)
            self.plot_validation_image(plot_type)
        else:
            self.plot_stack.setCurrentIndex(0)
            
            self.plot_widget.clear()
            
            if plot_type == "electrons_distance":
                self.plot_electrons_vs_distance()
            elif plot_type == "photons_distance":
                self.plot_photons_vs_distance()
            elif plot_type == "irradiance_distance":
                self.plot_irradiance_vs_distance()
            elif plot_type == "emission_comparison":
                self.plot_emission_comparison()

    def plot_electrons_vs_distance(self):
        self.plot_widget.setLabel('left', 'Electron Count')
        self.plot_widget.setLabel('bottom', 'Distance (km)')
        self.plot_widget.setTitle('Electron Count vs Distance')
        
        use_log = self.log_scale_check.isChecked()
        self.plot_widget.setLogMode(x=use_log, y=use_log)
        
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        
        color_idx = 0
        
        for assembly_id in self.results_df['Assembly_ID'].unique():
            for iteration in self.results_df[self.results_df['Assembly_ID'] == assembly_id]['Iteration'].unique():
                data = self.results_df[
                    (self.results_df['Assembly_ID'] == assembly_id) & 
                    (self.results_df['Iteration'] == iteration)
                ]
                
                oi_color = colors[color_idx % len(colors)]
                oi_curve = self.plot_widget.plot(
                    data['Distance_km'], data['OI_Electrons'], 
                    pen=pg.mkPen(oi_color, width=2), 
                    symbol='o', symbolSize=6, symbolBrush=oi_color,
                    name=f'ID {assembly_id} Iter {iteration} OI'
                )
                
                al_color = colors[(color_idx + 1) % len(colors)]
                al_curve = self.plot_widget.plot(
                    data['Distance_km'], data['Al_Electrons'], 
                    pen=pg.mkPen(al_color, width=2, style=pg.QtCore.Qt.DashLine), 
                    symbol='s', symbolSize=6, symbolBrush=al_color,
                    name=f'ID {assembly_id} Iter {iteration} Al'
                )
                
                color_idx += 2
        
        legend = self.plot_widget.addLegend()

    def plot_photons_vs_distance(self):
        self.plot_widget.setLabel('left', 'Photon Count')
        self.plot_widget.setLabel('bottom', 'Distance (km)')
        self.plot_widget.setTitle('Photon Count vs Distance')
        
        use_log = self.log_scale_check.isChecked()
        self.plot_widget.setLogMode(x=use_log, y=use_log)
        
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        color_idx = 0
        
        for assembly_id in self.results_df['Assembly_ID'].unique():
            for iteration in self.results_df[self.results_df['Assembly_ID'] == assembly_id]['Iteration'].unique():
                data = self.results_df[
                    (self.results_df['Assembly_ID'] == assembly_id) & 
                    (self.results_df['Iteration'] == iteration)
                ]
                
                oi_color = colors[color_idx % len(colors)]
                self.plot_widget.plot(
                    data['Distance_km'], data['OI_Photons'], 
                    pen=pg.mkPen(oi_color, width=2), 
                    symbol='o', symbolSize=6, symbolBrush=oi_color,
                    name=f'ID {assembly_id} Iter {iteration} OI'
                )
                
                al_color = colors[(color_idx + 1) % len(colors)]
                self.plot_widget.plot(
                    data['Distance_km'], data['Al_Photons'], 
                    pen=pg.mkPen(al_color, width=2, style=pg.QtCore.Qt.DashLine), 
                    symbol='s', symbolSize=6, symbolBrush=al_color,
                    name=f'ID {assembly_id} Iter {iteration} Al'
                )
                
                color_idx += 2
        
        self.plot_widget.addLegend()

    def plot_irradiance_vs_distance(self):
        self.plot_widget.setLabel('left', 'Irradiance (W/m²)')
        self.plot_widget.setLabel('bottom', 'Distance (km)')
        self.plot_widget.setTitle('Irradiance vs Distance (Inverse Square Law)')
        
        use_log = self.log_scale_check.isChecked()
        self.plot_widget.setLogMode(x=use_log, y=use_log)
        
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
        color_idx = 0
        
        for assembly_id in self.results_df['Assembly_ID'].unique():
            for iteration in self.results_df[self.results_df['Assembly_ID'] == assembly_id]['Iteration'].unique():
                data = self.results_df[
                    (self.results_df['Assembly_ID'] == assembly_id) & 
                    (self.results_df['Iteration'] == iteration)
                ]
                
                oi_color = colors[color_idx % len(colors)]
                self.plot_widget.plot(
                    data['Distance_km'], data['OI_Irradiance_Wm2'], 
                    pen=pg.mkPen(oi_color, width=2), 
                    symbol='o', symbolSize=6, symbolBrush=oi_color,
                    name=f'ID {assembly_id} Iter {iteration} OI'
                )
                
                al_color = colors[(color_idx + 1) % len(colors)]
                self.plot_widget.plot(
                    data['Distance_km'], data['Al_Irradiance_Wm2'], 
                    pen=pg.mkPen(al_color, width=2, style=pg.QtCore.Qt.DashLine), 
                    symbol='s', symbolSize=6, symbolBrush=al_color,
                    name=f'ID {assembly_id} Iter {iteration} Al'
                )
                
                color_idx += 2
        
        self.plot_widget.addLegend()

    def plot_emission_comparison(self):
        distance_km = self.manual_distance.value()
        data = self.results_df[self.results_df['Distance_km'] == distance_km]
        
        if data.empty:
            distances = self.results_df['Distance_km'].unique()
            distance_km = distances[len(distances)//2]
            data = self.results_df[self.results_df['Distance_km'] == distance_km]
        
        self.plot_widget.setLabel('left', 'Electron Count')
        self.plot_widget.setLabel('bottom', 'Assembly ID + Iteration')
        self.plot_widget.setTitle(f'Emission Comparison at {distance_km:.4f} km')
        
        use_log = self.log_scale_check.isChecked()
        self.plot_widget.setLogMode(x=False, y=use_log)
        
        data_with_labels = data.copy()
        data_with_labels['Combined_Label'] = data_with_labels['Assembly_ID'].astype(str) + '_Iter' + data_with_labels['Iteration'].astype(str)
        
        assembly_iter_labels = data_with_labels['Combined_Label'].values
        oi_electrons = data_with_labels['OI_Electrons'].values
        al_electrons = data_with_labels['Al_Electrons'].values
        
        width = 0.35
        x_positions = np.arange(len(assembly_iter_labels))
        
        oi_bargraph = pg.BarGraphItem(
            x=x_positions - width/2, height=oi_electrons, width=width,
            brush='b', name='OI'
        )
        self.plot_widget.addItem(oi_bargraph)
        
        al_bargraph = pg.BarGraphItem(
            x=x_positions + width/2, height=al_electrons, width=width,
            brush='r', name='Al'
        )
        self.plot_widget.addItem(al_bargraph)
        
        x_labels = [label for label in assembly_iter_labels]
        x_ticks = [(i, label) for i, label in enumerate(x_labels)]
        x_axis = self.plot_widget.getAxis('bottom')
        x_axis.setTicks([x_ticks])
        
        self.plot_widget.addLegend()

    def plot_validation_image(self, plot_type):
        if not self.generated_images:
            empty_image = np.zeros((100, 100))
            self.image_widget.setImage(empty_image)
            return
        
        emission_type = plot_type.replace('_image', '').upper()
        
        if emission_type not in self.generated_images:
            empty_image = np.zeros((100, 100))
            self.image_widget.setImage(empty_image)
            return
        
        image = self.generated_images[emission_type]
        metadata = self.generated_images['metadata']
        
        if np.max(image) > 0:
            center_x, center_y = metadata['center_pixel']
            crop_size = 100
            
            y_start = max(0, center_y - crop_size)
            y_end = min(image.shape[0], center_y + crop_size)
            x_start = max(0, center_x - crop_size)
            x_end = min(image.shape[1], center_x + crop_size)
            
            cropped_image = image[y_start:y_end, x_start:x_end]
            
            self.image_widget.setImage(cropped_image.T, autoRange=True, autoLevels=True)
            
            photon_count = metadata[f"{emission_type.lower()}_photons"]
            electron_count = metadata[f"{emission_type.lower()}_electrons"]
            print(f"Displaying {emission_type} validation image:")
            print(f"  Debris {metadata['debris_id']}, Iter {metadata['iteration']} at {metadata['distance_km']:.4f}km")
            print(f"  {photon_count:.3f} photons at center pixel (for Pyxel)")
            print(f"  {electron_count:.3f} electrons expected (reference)")
            print(f"  Showing {cropped_image.shape} region around center")
        else:
            empty_image = np.zeros((100, 100))
            self.image_widget.setImage(empty_image.T)

    def save_parameters(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"sensor_parameters_{timestamp}.json"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", default_filename, "JSON Files (*.json)")
        
        if not filepath:
            return
        
        try:
            params = {
                'aperture_diameter_m': self.aperture_input.value(),
                'focal_length_m': self.focal_length_input.value(),
                'pixel_pitch_um': self.pixel_pitch_input.value(),
                'sensor_width_px': self.sensor_width_input.value(),
                'sensor_height_px': self.sensor_height_input.value(),
                'exposure_time_s': self.exposure_input.value(),
                'oi_quantum_efficiency': self.oi_qe_input.value(),
                'oi_filter_transmission': self.oi_filter_input.value(),
                'al_quantum_efficiency': self.al_qe_input.value(),
                'al_filter_transmission': self.al_filter_input.value(),
                'oi_wavelength_nm': self.oi_wavelength_input.value(),
                'al_wavelength_nm': self.al_wavelength_input.value(),
                'apply_angular_correction': self.apply_angular_correction.isChecked(),
                'angular_cos_factor': self.angular_cos_factor.value(),
                'auto_replace_qe': self.auto_replace_qe_check.isChecked() if PYXEL_AVAILABLE else True,
                'saved_timestamp': timestamp,
                'parameter_source': 'validation_gui_with_pyxel_thread_safe_auto_qe_toggle'
            }
            
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=2)
                
            QMessageBox.information(self, "Parameters Saved", 
                                f"Parameters saved to: {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving parameters: {str(e)}")

    def load_parameters(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json)")
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
            
            if 'aperture_diameter_m' in params:
                self.aperture_input.setValue(params['aperture_diameter_m'])
            if 'focal_length_m' in params:
                self.focal_length_input.setValue(params['focal_length_m'])
            if 'pixel_pitch_um' in params:
                self.pixel_pitch_input.setValue(params['pixel_pitch_um'])
            if 'sensor_width_px' in params:
                self.sensor_width_input.setValue(params['sensor_width_px'])
            if 'sensor_height_px' in params:
                self.sensor_height_input.setValue(params['sensor_height_px'])
            if 'exposure_time_s' in params:
                self.exposure_input.setValue(params['exposure_time_s'])
            
            if 'oi_quantum_efficiency' in params:
                self.oi_qe_input.setValue(params['oi_quantum_efficiency'])
            elif 'quantum_efficiency' in params:
                self.oi_qe_input.setValue(params['quantum_efficiency'])
                
            if 'oi_filter_transmission' in params:
                self.oi_filter_input.setValue(params['oi_filter_transmission'])
            elif 'filter_transmission' in params:
                self.oi_filter_input.setValue(params['filter_transmission'])
                
            if 'al_quantum_efficiency' in params:
                self.al_qe_input.setValue(params['al_quantum_efficiency'])
            elif 'quantum_efficiency' in params:
                self.al_qe_input.setValue(params['quantum_efficiency'])
                
            if 'al_filter_transmission' in params:
                self.al_filter_input.setValue(params['al_filter_transmission'])
            elif 'filter_transmission' in params:
                self.al_filter_input.setValue(params['filter_transmission'])
            
            if 'oi_wavelength_nm' in params:
                self.oi_wavelength_input.setValue(params['oi_wavelength_nm'])
            if 'al_wavelength_nm' in params:
                self.al_wavelength_input.setValue(params['al_wavelength_nm'])
            if 'apply_angular_correction' in params:
                self.apply_angular_correction.setChecked(params['apply_angular_correction'])
            if 'angular_cos_factor' in params:
                self.angular_cos_factor.setValue(params['angular_cos_factor'])
            
            # Load auto replace QE setting if available
            if PYXEL_AVAILABLE and 'auto_replace_qe' in params:
                self.auto_replace_qe_check.setChecked(params['auto_replace_qe'])
            
            self.update_calculated_parameters()
            
            QMessageBox.information(self, "Parameters Loaded", 
                                f"Parameters loaded from: {filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading parameters: {str(e)}")

    def reset_parameters(self):
        defaults = self.get_default_sensor_params()
        
        self.aperture_input.setValue(defaults['aperture_diameter_m'])
        self.focal_length_input.setValue(defaults['focal_length_m'])
        self.pixel_pitch_input.setValue(defaults['pixel_pitch_um'])
        self.sensor_width_input.setValue(defaults['sensor_width_px'])
        self.sensor_height_input.setValue(defaults['sensor_height_px'])
        self.exposure_input.setValue(defaults['exposure_time_s'])
        self.oi_qe_input.setValue(defaults['oi_quantum_efficiency'])
        self.oi_filter_input.setValue(defaults['oi_filter_transmission'])
        self.al_qe_input.setValue(defaults['al_quantum_efficiency'])
        self.al_filter_input.setValue(defaults['al_filter_transmission'])
        self.oi_wavelength_input.setValue(defaults['oi_wavelength_nm'])
        self.al_wavelength_input.setValue(defaults['al_wavelength_nm'])
        
        self.apply_angular_correction.setChecked(False)
        self.angular_cos_factor.setValue(1.0)
        
        # Reset auto replace QE to default (enabled)
        if PYXEL_AVAILABLE:
            self.auto_replace_qe_check.setChecked(True)
        
        self.update_calculated_parameters()
        
        QMessageBox.information(self, "Reset Complete", "All parameters reset to default values.")


def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("Sensor Validation Tool")
    app.setApplicationVersion("6.0")
    
    window = ValidationMainWindow()
    window.show()
    app.setStyle('Fusion') 
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()