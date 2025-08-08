#!/usr/bin/env python3
"""
Pyxel Sensor Simulator: image processing and simulation tool for space debris analysis
"""
import sys, os, numpy as np
import pandas as pd
from PIL import Image
import re
import traceback
import tempfile
import shutil
from datetime import datetime
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QIcon, QPixmap, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QAction, QFileDialog, QMessageBox,
    QTabWidget, QGroupBox, QLabel, QPushButton, QScrollArea, QVBoxLayout,
    QHBoxLayout, QFormLayout, QInputDialog, QDoubleSpinBox, QSpinBox,
    QCheckBox, QLineEdit, QPlainTextEdit, QComboBox, QDialog, QSlider, QProgressDialog,
    QColorDialog, QSplitter, QListWidget, QListWidgetItem, QTextEdit, QFrame
)
import pyqtgraph as pg
pg.setConfigOptions(
    exitCleanup=False,
    background='k',
    foreground='w',
    antialias=True
)
import pyxel
from pyxel.models.charge_measurement.amplifier_crosstalk import (
    crosstalk_signal_dc,
    crosstalk_signal_ac
)
import matplotlib as mpl
from matplotlib import cm
import os
from astropy.io import fits
import glob
import json
import csv
import copy

def run_exposure_mode(config):
    try:
        exposure = config.exposure
        
        detector = None
        if hasattr(config, 'cmos_detector'):
            detector = config.cmos_detector
        elif hasattr(config, 'detector'):
            detector = config.detector
        elif hasattr(config, 'ccd_detector'):
            detector = config.ccd_detector
        else:
            raise ValueError("No detector found in config")
        
        pipeline = config.pipeline
        
        result_datatree = pyxel.run_mode(mode=exposure, detector=detector, pipeline=pipeline)
        
        return result_datatree
    except Exception as e:
        print(f"Error in run_exposure_mode: {e}")
        raise

def is_esa_viewer_data(filepath):
    if not filepath.lower().endswith('.npy'):
        return False
    filename = os.path.basename(filepath)
    return 'photon_map_' in filename or 'debris' in filename.lower()

def extract_satellite_info(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'photon_map_([^_]+)_(\d+)', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def safe_load_image(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        return QPixmap(file_path)
    except:
        return None

def create_error_dialog(parent, title, message):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setMinimumSize(800, 400)
    
    layout = QVBoxLayout(dialog)
    
    text_edit = QPlainTextEdit()
    text_edit.setPlainText(message)
    text_edit.setReadOnly(True)
    layout.addWidget(text_edit)
    
    button_layout = QHBoxLayout()
    copy_button = QPushButton("Copy to Clipboard")
    copy_button.clicked.connect(lambda: QApplication.clipboard().setText(message))
    
    close_button = QPushButton("Close")
    close_button.clicked.connect(dialog.accept)
    
    button_layout.addWidget(copy_button)
    button_layout.addStretch(1)
    button_layout.addWidget(close_button)
    
    layout.addLayout(button_layout)
    
    return dialog

class StatisticsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Simulation Statistics", parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self.setup_ui()
        self.clear_stats()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.create_basic_stats_section(layout)
        self.create_adu_stats_section(layout)
        self.create_signal_stats_section(layout)
        self.create_quality_stats_section(layout)
        
        layout.addStretch()
        
    def create_basic_stats_section(self, parent_layout):
        group = QGroupBox("Basic Info")
        layout = QFormLayout(group)
        layout.setSpacing(3)
        
        self.filename_label = QLabel("No data")
        self.shape_label = QLabel("N/A")
        self.frames_label = QLabel("N/A")
        
        layout.addRow("File:", self.filename_label)
        layout.addRow("Shape:", self.shape_label)
        layout.addRow("Frames:", self.frames_label)
        
        parent_layout.addWidget(group)
        
    def create_adu_stats_section(self, parent_layout):
        group = QGroupBox("ADU Statistics")
        layout = QFormLayout(group)
        layout.setSpacing(3)
        
        self.peak_adu_label = QLabel("N/A")
        self.center_adu_label = QLabel("N/A")
        self.mean_adu_label = QLabel("N/A")
        self.background_label = QLabel("N/A")
        
        self.peak_adu_label.setStyleSheet("font-weight: bold; color: #ff6600;")
        
        layout.addRow("Peak ADU:", self.peak_adu_label)
        layout.addRow("Center ADU:", self.center_adu_label)
        layout.addRow("Mean ADU:", self.mean_adu_label)
        layout.addRow("Background:", self.background_label)
        
        parent_layout.addWidget(group)
        
    def create_signal_stats_section(self, parent_layout):
        group = QGroupBox("Signal Quality")
        layout = QFormLayout(group)
        layout.setSpacing(3)
        
        self.snr_label = QLabel("N/A")
        self.dynamic_range_label = QLabel("N/A")
        self.fill_factor_label = QLabel("N/A")
        self.saturation_label = QLabel("N/A")
        
        self.snr_label.setStyleSheet("font-weight: bold; color: #00aa44;")
        
        layout.addRow("SNR:", self.snr_label)
        layout.addRow("Dynamic Range:", self.dynamic_range_label)
        layout.addRow("Fill Factor:", self.fill_factor_label)
        layout.addRow("Saturation:", self.saturation_label)
        
        parent_layout.addWidget(group)
        
    def create_quality_stats_section(self, parent_layout):
        group = QGroupBox("Quality Metrics")
        layout = QFormLayout(group)
        layout.setSpacing(3)
        
        self.fwhm_label = QLabel("N/A")
        self.peak_location_label = QLabel("N/A")
        self.total_adu_label = QLabel("N/A")
        self.total_signal_label = QLabel("N/A")
        
        layout.addRow("FWHM:", self.fwhm_label)
        layout.addRow("Peak Location:", self.peak_location_label)
        layout.addRow("Total ADU:", self.total_adu_label)
        layout.addRow("Signal ADU:", self.total_signal_label)
        
        parent_layout.addWidget(group)
        
    def update_stats(self, analysis_results):
        if not analysis_results:
            self.clear_stats()
            return
            
        try:
            filename = analysis_results.get('original_filename', 'N/A')
            if len(filename) > 20:
                filename = "..." + filename[-17:]
            self.filename_label.setText(filename)
            
            shape = analysis_results.get('output_shape', (0, 0))
            self.shape_label.setText(f"{shape[0]} × {shape[1]}")
            
            frames = analysis_results.get('time_frames', 1)
            self.frames_label.setText(str(frames))
            
            peak_adu = analysis_results.get('peak_adu', 0)
            self.peak_adu_label.setText(f"{peak_adu:.1f}")
            
            center_adu = analysis_results.get('center_pixel_adu', 0)
            self.center_adu_label.setText(f"{center_adu:.1f}")
            
            mean_adu = analysis_results.get('mean_adu', 0)
            self.mean_adu_label.setText(f"{mean_adu:.3f}")
            
            background = analysis_results.get('background_mean', 0)
            self.background_label.setText(f"{background:.3f}")
            
            snr = analysis_results.get('snr', 0)
            if snr == float('inf'):
                self.snr_label.setText("∞")
            else:
                self.snr_label.setText(f"{snr:.1f}")
            
            dynamic_range = analysis_results.get('dynamic_range_utilization', 0) * 100
            self.dynamic_range_label.setText(f"{dynamic_range:.1f}%")
            
            fill_factor = analysis_results.get('fill_factor', 0) * 100
            self.fill_factor_label.setText(f"{fill_factor:.4f}%")
            
            saturation = analysis_results.get('saturation_percentage', 0)
            self.saturation_label.setText(f"{saturation:.4f}%")
            
            fwhm = analysis_results.get('fwhm_average', 0)
            if fwhm > 0:
                self.fwhm_label.setText(f"{fwhm:.1f} px")
            else:
                self.fwhm_label.setText("N/A")
            
            peak_row = analysis_results.get('peak_row', 0)
            peak_col = analysis_results.get('peak_col', 0)
            self.peak_location_label.setText(f"({peak_col}, {peak_row})")
            
            total_adu = analysis_results.get('total_adu', 0)
            if total_adu > 1e9:
                self.total_adu_label.setText(f"{total_adu/1e9:.1f}B")
            elif total_adu > 1e6:
                self.total_adu_label.setText(f"{total_adu/1e6:.1f}M")
            elif total_adu > 1e3:
                self.total_adu_label.setText(f"{total_adu/1e3:.1f}k")
            else:
                self.total_adu_label.setText(f"{total_adu:.0f}")
            
            total_signal = analysis_results.get('total_signal_adu', 0)
            if total_signal > 1e9:
                self.total_signal_label.setText(f"{total_signal/1e9:.1f}B")
            elif total_signal > 1e6:
                self.total_signal_label.setText(f"{total_signal/1e6:.1f}M")
            elif total_signal > 1e3:
                self.total_signal_label.setText(f"{total_signal/1e3:.1f}k")
            else:
                self.total_signal_label.setText(f"{total_signal:.0f}")
                
        except Exception as e:
            print(f"Error updating statistics panel: {e}")
            self.clear_stats()
    
    def clear_stats(self):
        self.filename_label.setText("No data")
        self.shape_label.setText("N/A")
        self.frames_label.setText("N/A")
        
        self.peak_adu_label.setText("N/A")
        self.center_adu_label.setText("N/A")
        self.mean_adu_label.setText("N/A")
        self.background_label.setText("N/A")
        
        self.snr_label.setText("N/A")
        self.dynamic_range_label.setText("N/A")
        self.fill_factor_label.setText("N/A")
        self.saturation_label.setText("N/A")
        
        self.fwhm_label.setText("N/A")
        self.peak_location_label.setText("N/A")
        self.total_adu_label.setText("N/A")
        self.total_signal_label.setText("N/A")

class BatchResultsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Batch Processing Results", parent)
        self.setMinimumWidth(320)
        self.setMaximumWidth(380)
        self.batch_summary = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(120)
        self.results_text.setPlainText("No batch processing completed yet")
        self.results_text.setStyleSheet("font-family: monospace; font-size: 9px;")
        layout.addWidget(self.results_text)
        
        self.summary_label = QLabel("No batch summary available")
        self.summary_label.setStyleSheet("font-weight: bold; color: #666; font-size: 10px;")
        self.summary_label.setWordWrap(True)
        self.summary_label.setMaximumHeight(35)
        layout.addWidget(self.summary_label)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        
        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.clicked.connect(self.clear_batch_results)
        self.clear_results_btn.setMaximumWidth(80)
        self.clear_results_btn.setMaximumHeight(25)
        self.clear_results_btn.setStyleSheet("font-size: 9px; padding: 3px;")
        
        self.save_log_btn = QPushButton("Save Log")
        self.save_log_btn.clicked.connect(self.save_results_log)
        self.save_log_btn.setMaximumWidth(80)
        self.save_log_btn.setMaximumHeight(25)
        self.save_log_btn.setStyleSheet("font-size: 9px; padding: 3px;")
        
        button_layout.addWidget(self.clear_results_btn)
        button_layout.addWidget(self.save_log_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def add_completed_file(self, filename, analysis_results):
        current_text = self.results_text.toPlainText()
        if current_text == "No batch processing completed yet":
            current_text = ""
        
        emission_type = analysis_results.get('emission_type', 'Unknown')
        applied_qe = analysis_results.get('applied_qe', 0)
        
        new_entry = (f"✓ {filename} ({emission_type}, QE:{applied_qe:.3f}): "
                    f"Peak: {analysis_results.get('peak_adu', 0):.1f} ADU, "
                    f"SNR: {analysis_results.get('snr', 0):.1f}\n")
        
        self.results_text.setPlainText(current_text + new_entry)
        self._scroll_to_bottom()
    
    def add_error_file(self, filename, error_message):
        current_text = self.results_text.toPlainText()
        if current_text == "No batch processing completed yet":
            current_text = ""
        
        new_entry = f"✗ ERROR {filename}: {error_message}\n"
        self.results_text.setPlainText(current_text + new_entry)
        self._scroll_to_bottom()
    
    def batch_finished(self, summary_stats):
        self.batch_summary = summary_stats
        
        successful_count = summary_stats.get('successful_files', 0)
        total_count = summary_stats.get('total_files', 0)
        emission_type = summary_stats.get('emission_type', 'Unknown')
        applied_qe = summary_stats.get('applied_qe', 0)
        
        summary_text = (f"Summary: {successful_count}/{total_count} successful | "
                       f"{emission_type} emission | QE: {applied_qe:.3f} | "
                       f"Max Peak: {summary_stats.get('max_peak_adu', 0):.1f} ADU")
        self.summary_label.setText(summary_text)
        
        current_text = self.results_text.toPlainText()
        summary = (f"\n=== BATCH COMPLETE ===\n"
                  f"Emission: {emission_type} | QE: {applied_qe:.3f}\n"
                  f"Successful: {successful_count}/{total_count}\n"
                  f"Max Peak ADU: {summary_stats.get('max_peak_adu', 0):.1f}\n"
                  f"Max SNR: {summary_stats.get('max_snr', 0):.1f}\n"
                  f"Completed: {datetime.now().strftime('%H:%M:%S')}\n")
        
        self.results_text.setPlainText(current_text + summary)
        self._scroll_to_bottom()
    
    def clear_batch_results(self):
        self.batch_summary = {}
        self.results_text.setPlainText("No batch processing completed yet")
        self.summary_label.setText("No batch summary available")
    
    def save_results_log(self):
        if self.results_text.toPlainText() == "No batch processing completed yet":
            QMessageBox.warning(self, "No Results", "No batch results to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"batch_results_log_{timestamp}.txt"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Results Log", default_filename, "Text Files (*.txt);;All Files (*.*)")
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("BATCH PROCESSING RESULTS LOG\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(self.results_text.toPlainText())
                
                if self.batch_summary:
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("BATCH SUMMARY\n")
                    f.write("=" * 50 + "\n")
                    for key, value in self.batch_summary.items():
                        f.write(f"{key}: {value}\n")
            
            QMessageBox.information(self, "Log Saved", f"Batch results log saved to:\n{filepath}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving log: {str(e)}")
    
    def _scroll_to_bottom(self):
        cursor = self.results_text.textCursor()
        cursor.movePosition(cursor.End)
        self.results_text.setTextCursor(cursor)
        self.results_text.ensureCursorVisible()
        QApplication.processEvents()

class SingleFrameWorker(QThread):
    """Worker for single frame processing using batch-style logic"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, config, image_data, emission_type, oi_qe, al_qe, auto_qe_enabled, original_data=None):
        super().__init__()
        # Use deep copy just like batch processing
        self.config = copy.deepcopy(config)
        self.image_data = image_data.copy()
        self.original_data = original_data.copy() if original_data is not None else None
        self.emission_type = emission_type
        self.oi_qe = oi_qe
        self.al_qe = al_qe
        self.auto_qe_enabled = auto_qe_enabled
        self.temp_dir = None
        self._simulation_mutex = QMutex()

    def run(self):
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="pyxel_single_")
            
            if self.auto_qe_enabled:
                self.progress.emit("Updating detector QE...")
                qe_updated = self.update_pyxel_detector_qe()
                if qe_updated:
                    self.progress.emit(f"Applied QE: {self.oi_qe if self.emission_type == 'OI' else self.al_qe:.3f}")
                else:
                    self.progress.emit("QE update skipped (manual mode)")
            else:
                self.progress.emit("Using manual QE from YAML...")
            
            self.progress.emit("Preparing image data...")
            fits_file_path = self.ensure_image_available_for_pyxel()
            
            self.progress.emit("Running Pyxel simulation...")
            
            # Use the same simulation approach as batch
            with QMutexLocker(self._simulation_mutex):
                result_datatree = run_exposure_mode(self.config)
            
            if "bucket" not in result_datatree:
                raise ValueError("No bucket data in Pyxel results")
            
            bucket_ds = result_datatree["bucket"].to_dataset()
            
            if "image" not in bucket_ds:
                raise ValueError("No image data in Pyxel bucket results")
            
            image_cube = bucket_ds["image"].values
            if "time" in bucket_ds.coords:
                times = bucket_ds["time"].values
            else:
                times = np.array([0.0])
            
            # Get final frame the same way as batch
            if len(image_cube.shape) == 3 and image_cube.shape[0] > 0:
                final_image = image_cube[-1]
            elif len(image_cube.shape) == 2:
                final_image = image_cube
            else:
                final_image = image_cube[0] if image_cube.shape[0] > 0 else np.zeros((100, 100))
            
            self.progress.emit("Analyzing results...")
            
            # Use the same analysis method as batch
            analysis_results = self.analyze_pyxel_results(final_image, bucket_ds)
            
            # Add emission info the same way as batch
            analysis_results['emission_type'] = self.emission_type
            if self.auto_qe_enabled:
                analysis_results['applied_qe'] = self.oi_qe if self.emission_type == "OI" else self.al_qe
            else:
                analysis_results['applied_qe'] = "YAML"
            analysis_results['oi_qe_setting'] = self.oi_qe
            analysis_results['al_qe_setting'] = self.al_qe
            analysis_results['auto_qe_enabled'] = self.auto_qe_enabled
            
            # Clean up temp file
            try:
                os.remove(fits_file_path)
            except:
                pass
                
            # Return results in the same format
            results = {
                'result_datatree': result_datatree,
                'analysis_results': analysis_results,
                'final_image': final_image,
                'image_cube': image_cube,
                'times': times
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            error_msg = f"Error in single frame processing: {str(e)}"
            self.error.emit(error_msg)
        finally:
            self.cleanup_temp_directory()

    def update_pyxel_detector_qe(self):
        """Use the exact same QE update method as batch processing, but check if auto QE is enabled"""
        if not self.auto_qe_enabled:
            return False
            
        try:
            target_qe = self.oi_qe if self.emission_type == "OI" else self.al_qe
            
            detector = None
            if hasattr(self.config, 'cmos_detector'):
                detector = self.config.cmos_detector
            elif hasattr(self.config, 'detector'):
                detector = self.config.detector
            elif hasattr(self.config, 'ccd_detector'):
                detector = self.config.ccd_detector
            
            if detector and hasattr(detector, 'characteristics'):
                detector.characteristics.quantum_efficiency = target_qe
                
                # Also update the pipeline model directly (like batch)
                if hasattr(self.config.pipeline, 'charge_generation'):
                    for item in self.config.pipeline.charge_generation:
                        if hasattr(item, 'name') and item.name == 'convert_photons':
                            if not hasattr(item, 'arguments'):
                                item.arguments = {}
                            item.arguments['quantum_efficiency'] = target_qe
                            break
                
                return True
            else:
                return False
                
        except Exception as e:
            return False

    def ensure_image_available_for_pyxel(self):
        """Use the exact same FITS creation method as batch processing"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fits_file_path = os.path.join(self.temp_dir, f'single_frame_{timestamp}.fits')
            
            # Prepare data the same way as batch
            data = self.image_data.astype(np.float32)
            if not np.isfinite(data).all():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            fits.writeto(fits_file_path, data, overwrite=True)
            
            self.update_load_image_path(fits_file_path)
            
            return fits_file_path
            
        except Exception as e:
            raise

    def update_load_image_path(self, fits_file_path):
        """Use the exact same path update method as batch processing"""
        fits_file_path = os.path.abspath(fits_file_path)
        
        if hasattr(self.config.pipeline, 'photon_collection') and self.config.pipeline.photon_collection:
            for item in self.config.pipeline.photon_collection:
                if hasattr(item, 'name') and item.name == 'load_image':
                    if not hasattr(item, 'arguments'):
                        item.arguments = {}
                    
                    item.arguments['image_file'] = fits_file_path
                    item.enabled = True
                    break

    def analyze_pyxel_results(self, final_image, bucket_ds):
        """Use the exact same analysis method as batch processing"""
        try:
            analysis = {}
            
            analysis['original_filename'] = 'single_frame_data'
            analysis['original_filepath'] = 'N/A'
            analysis['analysis_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Original data analysis
            if self.original_data is not None:
                analysis['original_shape'] = self.original_data.shape
                analysis['original_min'] = float(self.original_data.min())
                analysis['original_max'] = float(self.original_data.max())
                analysis['original_mean'] = float(self.original_data.mean())
                analysis['original_std'] = float(self.original_data.std())
                analysis['original_nonzero_pixels'] = int(np.count_nonzero(self.original_data))
                analysis['original_total_pixels'] = int(self.original_data.size)
            else:
                # Use current image data if original not available
                analysis['original_shape'] = self.image_data.shape
                analysis['original_min'] = float(self.image_data.min())
                analysis['original_max'] = float(self.image_data.max())
                analysis['original_mean'] = float(self.image_data.mean())
                analysis['original_std'] = float(self.image_data.std())
                analysis['original_nonzero_pixels'] = int(np.count_nonzero(self.image_data))
                analysis['original_total_pixels'] = int(self.image_data.size)
            
            # Output analysis - same as batch
            analysis['output_shape'] = final_image.shape
            analysis['time_frames'] = len(bucket_ds["time"].values) if "time" in bucket_ds else 1
            
            analysis['min_adu'] = float(final_image.min())
            analysis['max_adu'] = float(final_image.max())
            analysis['mean_adu'] = float(final_image.mean())
            analysis['std_adu'] = float(final_image.std())
            analysis['median_adu'] = float(np.median(final_image))
            analysis['peak_adu'] = float(final_image.max())
            
            peak_location = np.unravel_index(np.argmax(final_image), final_image.shape)
            analysis['peak_row'] = int(peak_location[0])
            analysis['peak_col'] = int(peak_location[1])
            
            center_row, center_col = final_image.shape[0] // 2, final_image.shape[1] // 2
            analysis['center_pixel_adu'] = float(final_image[center_row, center_col])
            analysis['center_pixel_row'] = center_row
            analysis['center_pixel_col'] = center_col
            
            analysis['nonzero_pixels'] = int(np.count_nonzero(final_image))
            analysis['total_pixels'] = int(final_image.size)
            analysis['fill_factor'] = float(analysis['nonzero_pixels'] / analysis['total_pixels'])
            
            # Background analysis - same as batch
            corner_size = min(50, min(final_image.shape) // 4)
            corners = [
                final_image[:corner_size, :corner_size],
                final_image[:corner_size, -corner_size:],
                final_image[-corner_size:, :corner_size],
                final_image[-corner_size:, -corner_size:]
            ]
            
            background_pixels = np.concatenate([corner.flatten() for corner in corners])
            analysis['background_mean'] = float(background_pixels.mean())
            analysis['background_std'] = float(background_pixels.std())
            analysis['background_median'] = float(np.median(background_pixels))
            
            # SNR calculation - same as batch
            signal_level = analysis['peak_adu'] - analysis['background_mean']
            noise_level = analysis['background_std']
            analysis['snr'] = float(signal_level / noise_level) if noise_level > 0 else float('inf')
            
            # Dynamic range - same as batch
            max_possible_adu = 65535
            analysis['max_possible_adu'] = max_possible_adu
            analysis['dynamic_range_utilization'] = float(analysis['peak_adu'] / max_possible_adu)
            
            saturation_threshold = 0.95 * max_possible_adu
            analysis['saturated_pixels'] = int(np.count_nonzero(final_image >= saturation_threshold))
            analysis['saturation_percentage'] = float(analysis['saturated_pixels'] / analysis['total_pixels'] * 100)
            
            # Total ADU calculations - same as batch
            total_adu = float(final_image.sum())
            analysis['total_adu'] = total_adu
            
            total_signal_adu = float((final_image - analysis['background_mean']).sum())
            total_signal_adu = max(0, total_signal_adu)
            analysis['total_signal_adu'] = total_signal_adu
            
            center_signal_adu = analysis['center_pixel_adu'] - analysis['background_mean']
            center_signal_adu = max(0, center_signal_adu)
            analysis['center_pixel_signal_adu'] = center_signal_adu
            
            # FWHM analysis - same as batch
            if analysis['peak_adu'] > analysis['background_mean'] + 3 * analysis['background_std']:
                try:
                    peak_row, peak_col = analysis['peak_row'], analysis['peak_col']
                    
                    h_profile = final_image[peak_row, :] - analysis['background_mean']
                    h_profile[h_profile < 0] = 0
                    
                    v_profile = final_image[:, peak_col] - analysis['background_mean']
                    v_profile[v_profile < 0] = 0
                    
                    def calculate_fwhm(profile):
                        if len(profile) == 0 or profile.max() == 0:
                            return 0
                        half_max = profile.max() / 2
                        indices = np.where(profile >= half_max)[0]
                        if len(indices) > 0:
                            return float(indices[-1] - indices[0] + 1)
                        return 0
                    
                    analysis['fwhm_horizontal'] = calculate_fwhm(h_profile)
                    analysis['fwhm_vertical'] = calculate_fwhm(v_profile)
                    analysis['fwhm_average'] = (analysis['fwhm_horizontal'] + analysis['fwhm_vertical']) / 2
                    
                except Exception:
                    analysis['fwhm_horizontal'] = 0.0
                    analysis['fwhm_vertical'] = 0.0
                    analysis['fwhm_average'] = 0.0
            else:
                analysis['fwhm_horizontal'] = 0.0
                analysis['fwhm_vertical'] = 0.0
                analysis['fwhm_average'] = 0.0
            
            # Percentile analysis - same as batch
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                analysis[f'percentile_{p}'] = float(np.percentile(final_image, p))
            
            return analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'original_filename': 'single_frame_error',
                'analysis_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'min_adu': float(final_image.min()) if final_image is not None else 0,
                'max_adu': float(final_image.max()) if final_image is not None else 0,
                'mean_adu': float(final_image.mean()) if final_image is not None else 0
            }

    def cleanup_temp_directory(self):
        """Clean up temporary directory"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            pass

class BatchProcessingWorker(QThread):
    progress = pyqtSignal(str, int, int)
    file_completed = pyqtSignal(str, object)
    error = pyqtSignal(str, str)
    finished = pyqtSignal(object)

    def __init__(self, config, input_dir, output_dir, emission_type, oi_qe, al_qe, auto_qe_enabled):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.emission_type = emission_type
        self.oi_qe = oi_qe
        self.al_qe = al_qe
        self.auto_qe_enabled = auto_qe_enabled
        self.temp_dir = None
        self.csv_writer = None
        self.csv_file = None
        self.summary_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_peak_adu': 0,
            'total_center_adu': 0,
            'max_peak_adu': 0,
            'max_snr': 0,
            'emission_type': emission_type,
            'applied_qe': (oi_qe if emission_type == "OI" else al_qe) if auto_qe_enabled else "YAML",
            'auto_qe_enabled': auto_qe_enabled
        }
        self._simulation_mutex = QMutex()

    def run(self):
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="pyxel_batch_")
            
            if self.auto_qe_enabled:
                self.update_pyxel_detector_qe()
            
            self.initialize_csv_streaming()
            
            npy_files = self.find_all_npy_files()
            total_files = len(npy_files)
            self.summary_stats['total_files'] = total_files
            
            for i, (filepath, relative_path) in enumerate(npy_files):
                try:
                    filename = os.path.basename(filepath)
                    qe_mode = f"QE: {self.summary_stats['applied_qe']:.3f}" if self.auto_qe_enabled else "Manual QE"
                    self.progress.emit(f"Processing {filename}... (Emission: {self.emission_type}, {qe_mode})", i, total_files)
                    
                    data = np.load(filepath, allow_pickle=True)
                    
                    if data.ndim == 3:
                        data = np.sum(data, axis=2)
                    elif data.ndim != 2:
                        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
                    
                    data = data.astype(np.float32)
                    if not np.isfinite(data).all():
                        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    fits_file_path = self.ensure_image_available_for_pyxel(data, filename)
                    
                    with QMutexLocker(self._simulation_mutex):
                        result_datatree = run_exposure_mode(self.config)
                    
                    if "bucket" not in result_datatree:
                        raise ValueError("No bucket data in Pyxel results")
                    
                    bucket_ds = result_datatree["bucket"].to_dataset()
                    
                    if "image" not in bucket_ds:
                        raise ValueError("No image data in Pyxel bucket results")
                    
                    image_cube = bucket_ds["image"].values
                    if "time" in bucket_ds.coords:
                        times = bucket_ds["time"].values
                    else:
                        times = np.array([0.0])
                    
                    if len(image_cube.shape) == 3 and image_cube.shape[0] > 0:
                        final_image = image_cube[-1]
                    elif len(image_cube.shape) == 2:
                        final_image = image_cube
                    else:
                        final_image = image_cube[0] if image_cube.shape[0] > 0 else np.zeros((100, 100))
                    
                    analysis_results = self.analyze_pyxel_results(final_image, bucket_ds, filepath, data)
                    
                    analysis_results['emission_type'] = self.emission_type
                    if self.auto_qe_enabled:
                        analysis_results['applied_qe'] = self.summary_stats['applied_qe']
                    else:
                        analysis_results['applied_qe'] = "YAML"
                    analysis_results['oi_qe_setting'] = self.oi_qe
                    analysis_results['al_qe_setting'] = self.al_qe
                    analysis_results['auto_qe_enabled'] = self.auto_qe_enabled
                    
                    self.write_result_to_csv(analysis_results)
                    self.update_summary_stats(analysis_results)
                    self.save_individual_results(filepath, relative_path, final_image, analysis_results)
                    
                    try:
                        os.remove(fits_file_path)
                    except:
                        pass
                    
                    self.file_completed.emit(filename, analysis_results)
                        
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.error.emit(filename, str(e))
                    self.summary_stats['failed_files'] += 1
                    
                    self.write_error_to_csv(filename, str(e))
                    continue
            
            self.finalize_csv_streaming()
            self.cleanup_temp_directory()
            self.finished.emit(self.summary_stats)
            
        except Exception as e:
            self.error.emit("Batch Process", f"Fatal error in batch processing: {str(e)}")
        finally:
            if self.csv_file:
                try:
                    self.csv_file.close()
                except:
                    pass
            self.cleanup_temp_directory()

    def update_pyxel_detector_qe(self):
        """Update QE only if auto QE is enabled"""
        if not self.auto_qe_enabled:
            return False
            
        try:
            target_qe = self.oi_qe if self.emission_type == "OI" else self.al_qe
            
            detector = None
            if hasattr(self.config, 'cmos_detector'):
                detector = self.config.cmos_detector
            elif hasattr(self.config, 'detector'):
                detector = self.config.detector
            elif hasattr(self.config, 'ccd_detector'):
                detector = self.config.ccd_detector
            
            if detector and hasattr(detector, 'characteristics'):
                detector.characteristics.quantum_efficiency = target_qe
                return True
            else:
                return False
                
        except Exception as e:
            return False

    def initialize_csv_streaming(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qe_mode = "auto" if self.auto_qe_enabled else "manual"
            csv_path = os.path.join(self.output_dir, f"batch_pyxel_analysis_{self.emission_type}_{qe_mode}_{timestamp}.csv")
            
            self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            header = [
                'filename', 'status', 'emission_type', 'applied_qe', 'auto_qe_enabled', 'oi_qe_setting', 'al_qe_setting',
                'analysis_timestamp', 'error_message',
                'original_shape_h', 'original_shape_w', 'original_min', 'original_max', 
                'original_mean', 'original_std', 'original_nonzero_pixels',
                'output_shape_h', 'output_shape_w', 'time_frames',
                'min_adu', 'max_adu', 'mean_adu', 'std_adu', 'median_adu', 'peak_adu',
                'peak_row', 'peak_col', 'center_pixel_adu', 'center_pixel_row', 'center_pixel_col',
                'center_pixel_signal_adu', 'nonzero_pixels', 'fill_factor', 'total_adu', 'total_signal_adu',
                'background_mean', 'background_std', 'background_median', 'snr',
                'dynamic_range_utilization', 'saturated_pixels', 'saturation_percentage',
                'fwhm_horizontal', 'fwhm_vertical', 'fwhm_average',
                'percentile_1', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_50',
                'percentile_75', 'percentile_90', 'percentile_95', 'percentile_99'
            ]
            
            self.csv_writer.writerow(header)
            self.csv_file.flush()
            
        except Exception as e:
            raise

    def write_result_to_csv(self, analysis_results):
        try:
            row = [
                analysis_results.get('original_filename', ''),
                'SUCCESS',
                analysis_results.get('emission_type', ''),
                analysis_results.get('applied_qe', 0),
                analysis_results.get('auto_qe_enabled', False),
                analysis_results.get('oi_qe_setting', 0),
                analysis_results.get('al_qe_setting', 0),
                analysis_results.get('analysis_timestamp', ''),
                '',
                analysis_results.get('original_shape', [0, 0])[0],
                analysis_results.get('original_shape', [0, 0])[1],
                analysis_results.get('original_min', 0),
                analysis_results.get('original_max', 0),
                analysis_results.get('original_mean', 0),
                analysis_results.get('original_std', 0),
                analysis_results.get('original_nonzero_pixels', 0),
                analysis_results.get('output_shape', [0, 0])[0],
                analysis_results.get('output_shape', [0, 0])[1],
                analysis_results.get('time_frames', 1),
                analysis_results.get('min_adu', 0),
                analysis_results.get('max_adu', 0),
                analysis_results.get('mean_adu', 0),
                analysis_results.get('std_adu', 0),
                analysis_results.get('median_adu', 0),
                analysis_results.get('peak_adu', 0),
                analysis_results.get('peak_row', 0),
                analysis_results.get('peak_col', 0),
                analysis_results.get('center_pixel_adu', 0),
                analysis_results.get('center_pixel_row', 0),
                analysis_results.get('center_pixel_col', 0),
                analysis_results.get('center_pixel_signal_adu', 0),
                analysis_results.get('nonzero_pixels', 0),
                analysis_results.get('fill_factor', 0),
                analysis_results.get('total_adu', 0),
                analysis_results.get('total_signal_adu', 0),
                analysis_results.get('background_mean', 0),
                analysis_results.get('background_std', 0),
                analysis_results.get('background_median', 0),
                analysis_results.get('snr', 0),
                analysis_results.get('dynamic_range_utilization', 0),
                analysis_results.get('saturated_pixels', 0),
                analysis_results.get('saturation_percentage', 0),
                analysis_results.get('fwhm_horizontal', 0),
                analysis_results.get('fwhm_vertical', 0),
                analysis_results.get('fwhm_average', 0),
                analysis_results.get('percentile_1', 0),
                analysis_results.get('percentile_5', 0),
                analysis_results.get('percentile_10', 0),
                analysis_results.get('percentile_25', 0),
                analysis_results.get('percentile_50', 0),
                analysis_results.get('percentile_75', 0),
                analysis_results.get('percentile_90', 0),
                analysis_results.get('percentile_95', 0),
                analysis_results.get('percentile_99', 0)
            ]
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            
        except Exception as e:
            pass

    def write_error_to_csv(self, filename, error_message):
        try:
            row = [filename, 'ERROR', self.emission_type, self.summary_stats['applied_qe'], 
                   self.auto_qe_enabled, self.oi_qe, self.al_qe, datetime.now().strftime("%Y%m%d_%H%M%S"), 
                   error_message] + [''] * 47
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            pass

    def update_summary_stats(self, analysis_results):
        try:
            self.summary_stats['successful_files'] += 1
            
            peak_adu = analysis_results.get('peak_adu', 0)
            center_adu = analysis_results.get('center_pixel_adu', 0)
            snr = analysis_results.get('snr', 0)
            
            self.summary_stats['total_peak_adu'] += peak_adu
            self.summary_stats['total_center_adu'] += center_adu
            self.summary_stats['max_peak_adu'] = max(self.summary_stats['max_peak_adu'], peak_adu)
            
            if snr != float('inf'):
                self.summary_stats['max_snr'] = max(self.summary_stats['max_snr'], snr)
                
        except Exception as e:
            pass

    def finalize_csv_streaming(self):
        try:
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qe_mode = "auto" if self.auto_qe_enabled else "manual"
            summary_path = os.path.join(self.output_dir, f"batch_summary_{self.emission_type}_{qe_mode}_{timestamp}.json")
            
            with open(summary_path, 'w') as f:
                json.dump(self.summary_stats, f, indent=2)
            
        except Exception as e:
            pass

    def cleanup_temp_directory(self):
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            pass

    def find_all_npy_files(self):
        npy_files = []
        
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.npy'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.input_dir)
                    npy_files.append((full_path, relative_path))
        
        return sorted(npy_files)

    def ensure_image_available_for_pyxel(self, image_data, original_filename):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_name = os.path.splitext(original_filename)[0]
            fits_file_path = os.path.join(self.temp_dir, f'{base_name}_{timestamp}.fits')
            
            fits.writeto(fits_file_path, image_data, overwrite=True)
            
            self.update_load_image_path(fits_file_path)
            
            return fits_file_path
            
        except Exception as e:
            raise

    def update_load_image_path(self, fits_file_path):
        fits_file_path = os.path.abspath(fits_file_path)
        
        if hasattr(self.config.pipeline, 'photon_collection') and self.config.pipeline.photon_collection:
            for item in self.config.pipeline.photon_collection:
                if hasattr(item, 'name') and item.name == 'load_image':
                    if not hasattr(item, 'arguments'):
                        item.arguments = {}
                    
                    item.arguments['image_file'] = fits_file_path
                    item.enabled = True
                    break

    def analyze_pyxel_results(self, final_image, bucket_ds, original_filepath, original_data):
        try:
            analysis = {}
            
            analysis['original_filename'] = os.path.basename(original_filepath)
            analysis['original_filepath'] = original_filepath
            analysis['analysis_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            analysis['original_shape'] = original_data.shape
            analysis['original_min'] = float(original_data.min())
            analysis['original_max'] = float(original_data.max())
            analysis['original_mean'] = float(original_data.mean())
            analysis['original_std'] = float(original_data.std())
            analysis['original_nonzero_pixels'] = int(np.count_nonzero(original_data))
            analysis['original_total_pixels'] = int(original_data.size)
            
            analysis['output_shape'] = final_image.shape
            analysis['time_frames'] = len(bucket_ds["time"].values) if "time" in bucket_ds else 1
            
            analysis['min_adu'] = float(final_image.min())
            analysis['max_adu'] = float(final_image.max())
            analysis['mean_adu'] = float(final_image.mean())
            analysis['std_adu'] = float(final_image.std())
            analysis['median_adu'] = float(np.median(final_image))
            analysis['peak_adu'] = float(final_image.max())
            
            peak_location = np.unravel_index(np.argmax(final_image), final_image.shape)
            analysis['peak_row'] = int(peak_location[0])
            analysis['peak_col'] = int(peak_location[1])
            
            center_row, center_col = final_image.shape[0] // 2, final_image.shape[1] // 2
            analysis['center_pixel_adu'] = float(final_image[center_row, center_col])
            analysis['center_pixel_row'] = center_row
            analysis['center_pixel_col'] = center_col
            
            analysis['nonzero_pixels'] = int(np.count_nonzero(final_image))
            analysis['total_pixels'] = int(final_image.size)
            analysis['fill_factor'] = float(analysis['nonzero_pixels'] / analysis['total_pixels'])
            
            corner_size = min(50, min(final_image.shape) // 4)
            corners = [
                final_image[:corner_size, :corner_size],
                final_image[:corner_size, -corner_size:],
                final_image[-corner_size:, :corner_size],
                final_image[-corner_size:, -corner_size:]
            ]
            
            background_pixels = np.concatenate([corner.flatten() for corner in corners])
            analysis['background_mean'] = float(background_pixels.mean())
            analysis['background_std'] = float(background_pixels.std())
            analysis['background_median'] = float(np.median(background_pixels))
            
            signal_level = analysis['peak_adu'] - analysis['background_mean']
            noise_level = analysis['background_std']
            analysis['snr'] = float(signal_level / noise_level) if noise_level > 0 else float('inf')
            
            max_possible_adu = 65535
            analysis['max_possible_adu'] = max_possible_adu
            analysis['dynamic_range_utilization'] = float(analysis['peak_adu'] / max_possible_adu)
            
            saturation_threshold = 0.95 * max_possible_adu
            analysis['saturated_pixels'] = int(np.count_nonzero(final_image >= saturation_threshold))
            analysis['saturation_percentage'] = float(analysis['saturated_pixels'] / analysis['total_pixels'] * 100)
            
            total_adu = float(final_image.sum())
            analysis['total_adu'] = total_adu
            
            total_signal_adu = float((final_image - analysis['background_mean']).sum())
            total_signal_adu = max(0, total_signal_adu)
            analysis['total_signal_adu'] = total_signal_adu
            
            center_signal_adu = analysis['center_pixel_adu'] - analysis['background_mean']
            center_signal_adu = max(0, center_signal_adu)
            analysis['center_pixel_signal_adu'] = center_signal_adu
            
            if analysis['peak_adu'] > analysis['background_mean'] + 3 * analysis['background_std']:
                try:
                    peak_row, peak_col = analysis['peak_row'], analysis['peak_col']
                    
                    h_profile = final_image[peak_row, :] - analysis['background_mean']
                    h_profile[h_profile < 0] = 0
                    
                    v_profile = final_image[:, peak_col] - analysis['background_mean']
                    v_profile[v_profile < 0] = 0
                    
                    def calculate_fwhm(profile):
                        if len(profile) == 0 or profile.max() == 0:
                            return 0
                        half_max = profile.max() / 2
                        indices = np.where(profile >= half_max)[0]
                        if len(indices) > 0:
                            return float(indices[-1] - indices[0] + 1)
                        return 0
                    
                    analysis['fwhm_horizontal'] = calculate_fwhm(h_profile)
                    analysis['fwhm_vertical'] = calculate_fwhm(v_profile)
                    analysis['fwhm_average'] = (analysis['fwhm_horizontal'] + analysis['fwhm_vertical']) / 2
                    
                except Exception:
                    analysis['fwhm_horizontal'] = 0.0
                    analysis['fwhm_vertical'] = 0.0
                    analysis['fwhm_average'] = 0.0
            else:
                analysis['fwhm_horizontal'] = 0.0
                analysis['fwhm_vertical'] = 0.0
                analysis['fwhm_average'] = 0.0
            
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                analysis[f'percentile_{p}'] = float(np.percentile(final_image, p))
            
            return analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'original_filename': os.path.basename(original_filepath),
                'analysis_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'min_adu': float(final_image.min()) if final_image is not None else 0,
                'max_adu': float(final_image.max()) if final_image is not None else 0,
                'mean_adu': float(final_image.mean()) if final_image is not None else 0
            }

    def save_individual_results(self, original_filepath, relative_path, final_image, analysis_results):
        try:
            output_subdir = os.path.join(self.output_dir, os.path.dirname(relative_path))
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            base_name = os.path.splitext(os.path.basename(original_filepath))[0]
            timestamp = analysis_results['analysis_timestamp']
            emission_type = analysis_results.get('emission_type', 'unknown')
            qe_mode = "auto" if analysis_results.get('auto_qe_enabled', False) else "manual"
            
            output_fits = os.path.join(output_subdir, f"{base_name}_pyxel_output_{emission_type}_{qe_mode}_{timestamp}.fits")
            fits.writeto(output_fits, final_image.astype(np.float32), overwrite=True)
            
            output_json = os.path.join(output_subdir, f"{base_name}_analysis_{emission_type}_{qe_mode}_{timestamp}.json")
            with open(output_json, 'w') as f:
                json.dump(analysis_results, f, indent=2)
                
        except Exception as e:
            pass

class BatchProcessingTab(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        self.batch_summary = {}
        
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        file_group = QGroupBox("Directory Selection")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(5)
        
        dir_layout = QHBoxLayout()
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setReadOnly(True)
        self.dir_path_edit.setMinimumWidth(300)
        self.browse_dir_btn = QPushButton("Browse Directory")
        self.browse_dir_btn.clicked.connect(self.browse_directory)
        self.browse_dir_btn.setMaximumWidth(120)
        
        dir_layout.addWidget(QLabel("Input Directory:"))
        dir_layout.addWidget(self.dir_path_edit, 1)
        dir_layout.addWidget(self.browse_dir_btn)
        file_layout.addLayout(dir_layout)
        
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_edit.setMinimumWidth(300)
        self.browse_output_btn = QPushButton("Browse Output")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        self.browse_output_btn.setMaximumWidth(120)
        
        output_layout.addWidget(QLabel("Output Directory:"))
        output_layout.addWidget(self.output_dir_edit, 1)
        output_layout.addWidget(self.browse_output_btn)
        file_layout.addLayout(output_layout)
        
        file_list_container = QWidget()
        file_list_layout = QVBoxLayout(file_list_container)
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(3)
        
        file_list_layout.addWidget(QLabel("Files found:"))
        
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        self.file_list.setMaximumHeight(400)
        file_list_layout.addWidget(self.file_list)
        
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(5)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_file_list)
        self.refresh_btn.setMaximumWidth(80)
        
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter files (e.g., *debris*, *.npy)")
        self.filter_edit.textChanged.connect(self.refresh_file_list)
        
        filter_layout.addWidget(self.refresh_btn)
        filter_layout.addWidget(QLabel("Filter:"))
        filter_layout.addWidget(self.filter_edit, 1)
        
        file_list_layout.addLayout(filter_layout)
        file_layout.addWidget(file_list_container)
        
        layout.addWidget(file_group)
        
        qe_group = QGroupBox("Quantum Efficiency Settings")
        qe_layout = QFormLayout(qe_group)
        qe_layout.setSpacing(8)
        qe_layout.setHorizontalSpacing(10)
        
        # Add Auto QE checkbox for batch processing
        self.auto_qe_checkbox = QCheckBox("Automatically apply QE values")
        self.auto_qe_checkbox.setChecked(True)
        self.auto_qe_checkbox.setToolTip("When checked, QE values will be automatically applied to YAML config.\nWhen unchecked, QE values from YAML file will be used.")
        self.auto_qe_checkbox.toggled.connect(self.on_auto_qe_toggled)
        qe_layout.addRow(self.auto_qe_checkbox)
        
        oi_layout = QHBoxLayout()
        self.oi_qe_input = QDoubleSpinBox()
        self.oi_qe_input.setRange(0.001, 1.0)
        self.oi_qe_input.setValue(0.3)
        self.oi_qe_input.setDecimals(3)
        self.oi_qe_input.setMaximumWidth(100)
        self.oi_qe_input.setToolTip("Quantum Efficiency for OI emission (777.3 nm)")
        self.oi_qe_input.valueChanged.connect(self.update_process_button_state)
        oi_layout.addWidget(self.oi_qe_input)
        oi_layout.addWidget(QLabel("(777.3 nm)"))
        oi_layout.addStretch()
        
        al_layout = QHBoxLayout()
        self.al_qe_input = QDoubleSpinBox()
        self.al_qe_input.setRange(0.001, 1.0)
        self.al_qe_input.setValue(0.7)
        self.al_qe_input.setDecimals(3)
        self.al_qe_input.setMaximumWidth(100)
        self.al_qe_input.setToolTip("Quantum Efficiency for Al emission (395.0 nm)")
        self.al_qe_input.valueChanged.connect(self.update_process_button_state)
        al_layout.addWidget(self.al_qe_input)
        al_layout.addWidget(QLabel("(395.0 nm)"))
        al_layout.addStretch()
        
        self.emission_type_combo = QComboBox()
        self.emission_type_combo.addItem("OI Emission (777.3 nm)", "OI")
        self.emission_type_combo.addItem("Al Emission (395.0 nm)", "Al")
        self.emission_type_combo.setToolTip("Select emission type for batch processing")
        self.emission_type_combo.setMaximumWidth(200)
        self.emission_type_combo.currentTextChanged.connect(self.update_process_button_state)
        
        qe_layout.addRow("OI QE:", oi_layout)
        qe_layout.addRow("Al QE:", al_layout)
        qe_layout.addRow("Emission Type:", self.emission_type_combo)
        
        self.qe_info_label = QLabel("QE values will be automatically applied during processing")
        self.qe_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        self.qe_info_label.setWordWrap(True)
        qe_layout.addRow(self.qe_info_label)
        
        layout.addWidget(qe_group)
        
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)
        
        self.progress_label = QLabel("Ready to process")
        self.progress_label.setStyleSheet("font-weight: bold; padding: 4px;")
        self.progress_label.setWordWrap(True)
        status_layout.addWidget(self.progress_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
    
    def on_auto_qe_toggled(self, checked):
        """Update UI when auto QE checkbox is toggled"""
        if checked:
            self.qe_info_label.setText("QE values will be automatically applied during processing")
            self.qe_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        else:
            self.qe_info_label.setText("QE values from YAML configuration will be used (manual mode)")
            self.qe_info_label.setStyleSheet("color: #ff6600; font-style: italic; font-size: 10px; font-weight: bold;")
        
        # Enable/disable QE input controls
        self.oi_qe_input.setEnabled(checked)
        self.al_qe_input.setEnabled(checked)
        self.emission_type_combo.setEnabled(checked)
        
        self.update_process_button_state()
    
    def set_parent_window(self, parent_window):
        self.parent_window = parent_window
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.dir_path_edit.setText(directory)
            self.refresh_file_list()
    
    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            self.update_process_button_state()
    
    def refresh_file_list(self):
        self.file_list.clear()
        
        input_dir = self.dir_path_edit.text()
        if not input_dir or not os.path.exists(input_dir):
            self.update_process_button_state()
            return
        
        filter_pattern = self.filter_edit.text().strip()
        if not filter_pattern:
            filter_pattern = "*.npy"
        
        try:
            npy_files = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith('.npy'):
                        if '*' in filter_pattern or '?' in filter_pattern:
                            import fnmatch
                            if fnmatch.fnmatch(file, filter_pattern):
                                full_path = os.path.join(root, file)
                                relative_path = os.path.relpath(full_path, input_dir)
                                npy_files.append((full_path, relative_path))
                        else:
                            if filter_pattern.lower() in file.lower():
                                full_path = os.path.join(root, file)
                                relative_path = os.path.relpath(full_path, input_dir)
                                npy_files.append((full_path, relative_path))
            
            npy_files.sort()
            
            for full_path, relative_path in npy_files:
                item = QListWidgetItem(relative_path)
                item.setData(Qt.UserRole, full_path)
                item.setToolTip(f"Full path: {full_path}")
                self.file_list.addItem(item)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error scanning directory: {str(e)}")
        
        self.update_process_button_state()
    
    def update_process_button_state(self):
        has_config = self.parent_window and self.parent_window.config is not None
        has_files = self.file_list.count() > 0
        has_output = bool(self.output_dir_edit.text())
        
        if self.parent_window and hasattr(self.parent_window, 'batch_process_button'):
            self.parent_window.batch_process_button.setEnabled(has_config and has_files and has_output)
        
        if not has_config:
            self.progress_label.setText("Load YAML configuration first")
            self.progress_label.setStyleSheet("color: orange; font-weight: bold; padding: 4px;")
        elif not has_files:
            self.progress_label.setText("No files to process")
            self.progress_label.setStyleSheet("color: orange; font-weight: bold; padding: 4px;")
        elif not has_output:
            self.progress_label.setText("Select output directory")
            self.progress_label.setStyleSheet("color: orange; font-weight: bold; padding: 4px;")
        else:
            emission_type = self.emission_type_combo.currentData()
            auto_qe = self.auto_qe_checkbox.isChecked()
            
            if auto_qe:
                qe_value = self.oi_qe_input.value() if emission_type == "OI" else self.al_qe_input.value()
                qe_text = f"QE: {qe_value:.3f}"
            else:
                qe_text = "Manual QE"
            
            self.progress_label.setText(f"Ready to process {self.file_list.count()} files with {emission_type} emission ({qe_text})")
            self.progress_label.setStyleSheet("color: green; font-weight: bold; padding: 4px;")
    
    def start_batch_processing(self):
        if not self.parent_window or not self.parent_window.config:
            QMessageBox.warning(self, "No Configuration", "Please load a YAML configuration first")
            return
        
        input_dir = self.dir_path_edit.text()
        output_dir = self.output_dir_edit.text()
        emission_type = self.emission_type_combo.currentData()
        oi_qe = self.oi_qe_input.value()
        al_qe = self.al_qe_input.value()
        auto_qe_enabled = self.auto_qe_checkbox.isChecked()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Missing Directories", "Please select input and output directories")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Directory Error", f"Cannot create output directory: {str(e)}")
            return
        
        qe_info = f"QE: {oi_qe if emission_type == 'OI' else al_qe:.3f}" if auto_qe_enabled else "Manual QE from YAML"
        
        reply = QMessageBox.question(
            self, "Confirm Batch Processing",
            f"Process all .npy files with {emission_type} emission settings?\n\n"
            f"Input: {input_dir}\n"
            f"Output: {output_dir}\n"
            f"Found: {self.file_list.count()} files\n"
            f"Emission Type: {emission_type}\n"
            f"Auto QE: {'Yes' if auto_qe_enabled else 'No'}\n"
            f"OI QE: {oi_qe:.3f}\n"
            f"Al QE: {al_qe:.3f}\n"
            f"Applied QE: {qe_info}\n\n"
            f"{'QE values will be dynamically applied to YAML configuration.' if auto_qe_enabled else 'QE values from YAML will be used.'}\n"
            f"Results will be streamed directly to CSV for memory efficiency.\n"
            f"This may take a long time!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        qe_mode_text = "auto QE" if auto_qe_enabled else "manual QE"
        self.progress_label.setText(f"Starting batch processing with {emission_type} emission ({qe_mode_text})...")
        self.progress_label.setStyleSheet("color: blue; font-weight: bold; padding: 4px;")
        
        if self.parent_window and hasattr(self.parent_window, 'batch_process_button'):
            self.parent_window.batch_process_button.setEnabled(False)
        
        self.batch_worker = BatchProcessingWorker(
            self.parent_window.config,
            input_dir,
            output_dir,
            emission_type,
            oi_qe,
            al_qe,
            auto_qe_enabled
        )
        
        self.batch_worker.progress.connect(self.update_batch_progress)
        self.batch_worker.file_completed.connect(self.file_processing_completed)
        self.batch_worker.error.connect(self.file_processing_error)
        self.batch_worker.finished.connect(self.batch_processing_finished)
        
        self.batch_worker.start()
    
    def update_batch_progress(self, message, current, total):
        progress_text = f"{message} ({current+1}/{total})"
        self.progress_label.setText(progress_text)
        QApplication.processEvents()
    
    def file_processing_completed(self, filename, analysis_results):
        if hasattr(self.parent_window, 'batch_results_panel'):
            self.parent_window.batch_results_panel.add_completed_file(filename, analysis_results)
        
        QApplication.processEvents()
    
    def file_processing_error(self, filename, error_message):
        if hasattr(self.parent_window, 'batch_results_panel'):
            self.parent_window.batch_results_panel.add_error_file(filename, error_message)
        
        QApplication.processEvents()
    
    def batch_processing_finished(self, summary_stats):
        self.batch_summary = summary_stats
        
        successful_count = summary_stats.get('successful_files', 0)
        total_count = summary_stats.get('total_files', 0)
        emission_type = summary_stats.get('emission_type', 'Unknown')
        applied_qe = summary_stats.get('applied_qe', 0)
        auto_qe_enabled = summary_stats.get('auto_qe_enabled', False)
        
        qe_text = f"QE:{applied_qe:.3f}" if auto_qe_enabled else "Manual QE"
        self.progress_label.setText(f"Batch complete ({emission_type}, {qe_text}): {successful_count}/{total_count} successful")
        self.progress_label.setStyleSheet("color: green; font-weight: bold; padding: 4px;")
        
        if self.parent_window and hasattr(self.parent_window, 'batch_process_button'):
            self.parent_window.batch_process_button.setEnabled(True)
        self.update_process_button_state()
        
        if hasattr(self.parent_window, 'batch_results_panel'):
            self.parent_window.batch_results_panel.batch_finished(summary_stats)
        
        if successful_count > 0:
            qe_mode_text = "automatic QE" if auto_qe_enabled else "manual QE from YAML"
            QMessageBox.information(
                self, "Batch Complete",
                f"Batch processing completed!\n\n"
                f"Emission Type: {emission_type}\n"
                f"QE Mode: {qe_mode_text}\n"
                f"Applied QE: {applied_qe if auto_qe_enabled else 'From YAML'}\n"
                f"Successful: {successful_count}/{total_count}\n"
                f"Results automatically saved to CSV in output directory.\n"
                f"{'QE values were dynamically applied during processing.' if auto_qe_enabled else 'QE values from YAML were used.'}"
            )
        else:
            QMessageBox.warning(
                self, "Batch Issues",
                f"Batch processing completed with issues!\n\n"
                f"Successful: {successful_count}/{total_count}\n"
                f"Check the results log for details"
            )

class PgImageView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.image_view = pg.ImageView()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.setPredefinedGradient("viridis")
        
        self.image_view.getView().setMouseEnabled(x=True, y=True)
        
        self.image_view.scene.sigMouseMoved.connect(self.mouse_moved)
        
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(2, 2, 2, 2)
        
        self.auto_zoom_btn = QPushButton("Auto Zoom")
        self.auto_zoom_btn.setToolTip("Automatically zoom to region with data")
        self.auto_zoom_btn.clicked.connect(self.auto_zoom_to_content)
        self.auto_zoom_btn.setMaximumWidth(120)
        
        self.center_max_btn = QPushButton("Center Max")
        self.center_max_btn.setToolTip("Center view on brightest pixel")
        self.center_max_btn.clicked.connect(self.center_on_maximum)
        self.center_max_btn.setMaximumWidth(120)
        
        self.log_scale_btn = QPushButton("Log Scale")
        self.log_scale_btn.setToolTip("Toggle logarithmic intensity scale")
        self.log_scale_btn.setCheckable(True)
        self.log_scale_btn.clicked.connect(self.toggle_log_scale)
        self.log_scale_btn.setMaximumWidth(120)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "grey", "hot", "cool"])
        self.colormap_combo.setToolTip("Select color map")
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        self.colormap_combo.setMaximumWidth(120)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Reset zoom and position")
        self.reset_btn.clicked.connect(self.reset_view)
        self.reset_btn.setMaximumWidth(100)
        
        self.crosshairs_btn = QPushButton("Crosshairs")
        self.crosshairs_btn.setToolTip("Toggle crosshairs cursor")
        self.crosshairs_btn.setCheckable(True)
        self.crosshairs_btn.clicked.connect(self.toggle_crosshairs)
        self.crosshairs_btn.setMaximumWidth(120)
        
        control_layout.addWidget(QLabel("Tools:"))
        control_layout.addWidget(self.auto_zoom_btn)
        control_layout.addWidget(self.center_max_btn)
        control_layout.addWidget(self.log_scale_btn)
        control_layout.addWidget(self.crosshairs_btn)
        control_layout.addWidget(self.colormap_combo)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch()
        
        self.pixel_info = QLabel("Pixel: (-, -) Value: -")
        self.pixel_info.setStyleSheet("background-color: #f0f0f0; padding: 2px; font-family: monospace;")
        
        layout.addWidget(control_panel)
        layout.addWidget(self.image_view)
        layout.addWidget(self.pixel_info)
        self.setLayout(layout)
        
        self.current_data = None
        self.log_scale_active = False
        self.crosshairs_active = False
        self.crosshairs_v = None
        self.crosshairs_h = None
        
        self.auto_zoom_btn.setEnabled(False)
        self.center_max_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.crosshairs_btn.setEnabled(False)
    
    def toggle_crosshairs(self):
        self.crosshairs_active = self.crosshairs_btn.isChecked()
        
        if self.crosshairs_active:
            pen = pg.mkPen('red', width=1, style=Qt.DashLine)
            self.crosshairs_v = pg.InfiniteLine(angle=90, pen=pen)
            self.crosshairs_h = pg.InfiniteLine(angle=0, pen=pen)
            
            self.image_view.addItem(self.crosshairs_v)
            self.image_view.addItem(self.crosshairs_h)
            
            if self.current_data is not None:
                center_x = self.current_data.shape[1] // 2
                center_y = self.current_data.shape[0] // 2
                self.crosshairs_v.setPos(center_x)
                self.crosshairs_h.setPos(center_y)
        else:
            if self.crosshairs_v is not None:
                self.image_view.removeItem(self.crosshairs_v)
                self.crosshairs_v = None
            if self.crosshairs_h is not None:
                self.image_view.removeItem(self.crosshairs_h)
                self.crosshairs_h = None
    
    def update_crosshairs(self, pos):
        if not self.crosshairs_active or self.current_data is None:
            return
            
        img_item = self.image_view.getImageItem()
        if img_item is None:
            return
            
        mouse_point = img_item.mapFromScene(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        if self.crosshairs_v is not None:
            self.crosshairs_v.setPos(x)
        if self.crosshairs_h is not None:
            self.crosshairs_h.setPos(y)
    
    def mouse_moved(self, pos):
        if self.current_data is None:
            return
            
        self.update_crosshairs(pos)
            
        img_item = self.image_view.getImageItem()
        if img_item is None:
            return
            
        mouse_point = img_item.mapFromScene(pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())
        
        if (0 <= x < self.current_data.shape[1] and 
            0 <= y < self.current_data.shape[0]):
            value = self.current_data[y, x]
            self.pixel_info.setText(f"Pixel: ({x:4d}, {y:4d}) Value: {value:8.3f}")
        else:
            self.pixel_info.setText("Pixel: (-, -) Value: -")
    
    def display_array(self, data: np.ndarray, title: str = "Image"):
        if data is None:
            return
        
        self.current_data = data.copy()
        
        display_data = data.copy()
        if self.log_scale_active and np.any(data > 0):
            epsilon = np.finfo(float).eps
            display_data = np.log10(np.maximum(data, epsilon))
            display_data[data <= 0] = np.nan
        
        self.image_view.setImage(display_data, autoLevels=True)
        
        self.auto_zoom_btn.setEnabled(True)
        self.center_max_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.crosshairs_btn.setEnabled(True)
        
        non_zero_pixels = np.count_nonzero(data)
        total_pixels = data.size
        if non_zero_pixels < total_pixels * 0.1:
            self.auto_zoom_to_content()
    
    def auto_zoom_to_content(self):
        if self.current_data is None:
            return
        
        y_indices, x_indices = np.nonzero(self.current_data)
        
        if len(y_indices) == 0:
            return
        
        padding = max(20, min(self.current_data.shape) // 20)
        
        y_min = max(0, np.min(y_indices) - padding)
        y_max = min(self.current_data.shape[0], np.max(y_indices) + padding)
        x_min = max(0, np.min(x_indices) - padding)
        x_max = min(self.current_data.shape[1], np.max(x_indices) + padding)
        
        self.image_view.getView().setRange(
            xRange=[x_min, x_max],
            yRange=[y_min, y_max],
            padding=0
        )
    
    def center_on_maximum(self):
        if self.current_data is None:
            return
        
        max_idx = np.unravel_index(np.argmax(self.current_data), self.current_data.shape)
        y_max, x_max = max_idx
        
        view_range = self.image_view.getView().viewRange()
        width = view_range[0][1] - view_range[0][0]
        height = view_range[1][1] - view_range[1][0]
        
        self.image_view.getView().setRange(
            xRange=[x_max - width/2, x_max + width/2],
            yRange=[y_max - height/2, y_max + height/2],
            padding=0
        )
        
        max_value = self.current_data[y_max, x_max]
        self.pixel_info.setText(f"Pixel: ({x_max:4d}, {y_max:4d}) Value: {max_value:8.3f} [MAX]")
        self.pixel_info.setStyleSheet("background-color: #ffdddd; padding: 2px; font-family: monospace; font-weight: bold;")
        
        QApplication.instance().processEvents()
        import time
        time.sleep(0.1)
        self.pixel_info.setStyleSheet("background-color: #f0f0f0; padding: 2px; font-family: monospace;")
    
    def toggle_log_scale(self):
        self.log_scale_active = self.log_scale_btn.isChecked()
        if self.current_data is not None:
            self.display_array(self.current_data, "Image")
    
    def change_colormap(self, colormap_name):
        self.image_view.setPredefinedGradient(colormap_name)
    
    def reset_view(self):
        if self.current_data is not None:
            self.image_view.autoRange()

class LoadImageTab(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        
        layout = QVBoxLayout(self)
        
        file_group = QGroupBox("File Loading")
        file_layout = QVBoxLayout(file_group)
        
        status_layout = QFormLayout()
        self.data_label = QLabel("Not loaded")
        status_layout.addRow("Image Data:", self.data_label)
        file_layout.addLayout(status_layout)
        
        control_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.load_numpy_btn = QPushButton("Load NumPy (.npy)")
        self.load_numpy_btn.clicked.connect(self.load_numpy_data)
        
        self.load_fits_btn = QPushButton("Load FITS (.fits)")
        self.load_fits_btn.clicked.connect(self.load_fits_data)
        
        btn_layout.addWidget(self.load_numpy_btn)
        btn_layout.addWidget(self.load_fits_btn)
        control_layout.addLayout(btn_layout)
        
        info_label = QLabel("Load NumPy (.npy) or FITS (.fits) files containing image data. The image will be displayed in the main view.")
        info_label.setStyleSheet("color: blue; font-style: italic;")
        info_label.setWordWrap(True)
        control_layout.addWidget(info_label)
        
        file_layout.addLayout(control_layout)
        layout.addWidget(file_group)
        
        qe_group = QGroupBox("Single Exposure QE Settings")
        qe_layout = QFormLayout(qe_group)
        qe_layout.setSpacing(8)
        qe_layout.setHorizontalSpacing(10)
        
        # Add Auto QE checkbox for single exposure
        self.auto_qe_checkbox = QCheckBox("Automatically apply QE values")
        self.auto_qe_checkbox.setChecked(True)
        self.auto_qe_checkbox.setToolTip("When checked, QE values will be automatically applied to YAML config.\nWhen unchecked, QE values from YAML file will be used.")
        self.auto_qe_checkbox.toggled.connect(self.on_auto_qe_toggled)
        qe_layout.addRow(self.auto_qe_checkbox)
        
        oi_layout = QHBoxLayout()
        self.oi_qe_input = QDoubleSpinBox()
        self.oi_qe_input.setRange(0.001, 1.0)
        self.oi_qe_input.setValue(0.3)
        self.oi_qe_input.setDecimals(3)
        self.oi_qe_input.setMaximumWidth(100)
        self.oi_qe_input.setToolTip("Quantum Efficiency for OI emission (777.3 nm)")
        oi_layout.addWidget(self.oi_qe_input)
        oi_layout.addWidget(QLabel("(777.3 nm)"))
        oi_layout.addStretch()
        
        al_layout = QHBoxLayout()
        self.al_qe_input = QDoubleSpinBox()
        self.al_qe_input.setRange(0.001, 1.0)
        self.al_qe_input.setValue(0.7)
        self.al_qe_input.setDecimals(3)
        self.al_qe_input.setMaximumWidth(100)
        self.al_qe_input.setToolTip("Quantum Efficiency for Al emission (395.0 nm)")
        al_layout.addWidget(self.al_qe_input)
        al_layout.addWidget(QLabel("(395.0 nm)"))
        al_layout.addStretch()
        
        self.emission_type_combo = QComboBox()
        self.emission_type_combo.addItem("OI Emission (777.3 nm)", "OI")
        self.emission_type_combo.addItem("Al Emission (395.0 nm)", "Al")
        self.emission_type_combo.setToolTip("Select emission type for single exposure")
        self.emission_type_combo.setMaximumWidth(200)
        
        qe_layout.addRow("OI QE:", oi_layout)
        qe_layout.addRow("Al QE:", al_layout)
        qe_layout.addRow("Emission Type:", self.emission_type_combo)
        
        self.qe_info_label = QLabel("These QE values will be used for single exposure simulation")
        self.qe_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        self.qe_info_label.setWordWrap(True)
        qe_layout.addRow(self.qe_info_label)
        
        layout.addWidget(qe_group)
        
        exposure_info_label = QLabel("After loading data and setting QE, click 'Perform Exposure' to run Pyxel simulation")
        exposure_info_label.setStyleSheet("color: green; font-style: italic;")
        exposure_info_label.setWordWrap(True)
        layout.addWidget(exposure_info_label)
        
        layout.addStretch()
        
        self.numpy_data = None
    
    def on_auto_qe_toggled(self, checked):
        """Update UI when auto QE checkbox is toggled"""
        if checked:
            self.qe_info_label.setText("These QE values will be used for single exposure simulation")
            self.qe_info_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        else:
            self.qe_info_label.setText("QE values from YAML configuration will be used (manual mode)")
            self.qe_info_label.setStyleSheet("color: #ff6600; font-style: italic; font-size: 10px; font-weight: bold;")
        
        # Enable/disable QE input controls
        self.oi_qe_input.setEnabled(checked)
        self.al_qe_input.setEnabled(checked)
        self.emission_type_combo.setEnabled(checked)
    
    def get_qe_settings(self):
        emission_type = self.emission_type_combo.currentData()
        oi_qe = self.oi_qe_input.value()
        al_qe = self.al_qe_input.value()
        auto_qe_enabled = self.auto_qe_checkbox.isChecked()
        return emission_type, oi_qe, al_qe, auto_qe_enabled
    
    def update_geometry_display(self, config):
        pass
        
    def set_parent_window(self, parent_window):
        self.parent_window = parent_window
    
    def load_fits_data(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open FITS Image Data", "",
                "FITS Files (*.fits *.fit *.fts);;All Files (*.*)"
            )
        if not path:
            return False
            
        try:
            if self.parent_window:
                self.parent_window.statusBar().showMessage("Loading FITS data, please wait...")
                QApplication.processEvents()
            
            with fits.open(path) as hdul:
                data_hdu = None
                for hdu in hdul:
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        data_hdu = hdu
                        break
                
                if data_hdu is None:
                    QMessageBox.warning(self, "Warning", "No image data found in FITS file.")
                    return False
                
                data = data_hdu.data
                header = data_hdu.header
            
            if not isinstance(data, np.ndarray):
                QMessageBox.warning(self, "Warning", "File does not contain a valid array.")
                return False
            
            if data.ndim > 2:
                if data.ndim == 3:
                    response = QMessageBox.question(
                        self, "3D Array Detected",
                        f"Data has shape {data.shape}. Would you like to:\n"
                        "- Yes: Sum across the last dimension\n"
                        "- No: Use only the first slice",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if response == QMessageBox.Yes:
                        if data.shape[2] <= data.shape[0] and data.shape[2] <= data.shape[1]:
                            data = np.sum(data, axis=2)
                        else:
                            data = np.sum(data, axis=0)
                    else:
                        if data.shape[2] <= data.shape[0] and data.shape[2] <= data.shape[1]:
                            data = data[:, :, 0]
                        else:
                            data = data[0, :, :]
                elif data.ndim == 4:
                    data = data[0, 0, :, :]
                else:
                    QMessageBox.warning(self, "Warning", 
                                       f"Unsupported {data.ndim}D array with shape {data.shape}")
                    return False
            
            data = data.astype(np.float32)
            
            if not np.isfinite(data).all():
                response = QMessageBox.question(
                    self, "Data Quality Issue",
                    "Data contains NaN or infinite values. Replace with zeros?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.Yes:
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    return False
            
            return self._process_loaded_data(data, path)
                
        except Exception as e:
            error_details = traceback.format_exc()
            error_dialog = create_error_dialog(self, "Error", f"Failed to load FITS data:\n{str(e)}\n\nDetails:\n{error_details}")
            error_dialog.exec_()
            if self.parent_window:
                self.parent_window.statusBar().showMessage("Error loading FITS data")
            return False
            
    def load_numpy_data(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open NumPy Image Data", "",
                "NumPy Files (*.npy);;All Files (*.*)"
            )
        if not path:
            return False
            
        try:
            if self.parent_window:
                self.parent_window.statusBar().showMessage("Loading NumPy data, please wait...")
                QApplication.processEvents()
            
            data = np.load(path, allow_pickle=True)
            
            if not isinstance(data, np.ndarray):
                QMessageBox.warning(self, "Warning", "File does not contain a valid NumPy array.")
                return False
            
            if data.ndim == 3:
                response = QMessageBox.question(
                    self, "3D Array Detected",
                    f"Data has shape {data.shape}. Would you like to:\n"
                    "- Yes: Sum across the last dimension\n"
                    "- No: Use only the first channel",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.Yes:
                    data = np.sum(data, axis=2)
                else:
                    data = data[:, :, 0]
            elif data.ndim != 2:
                QMessageBox.warning(self, "Warning", 
                                   f"Expected 2D or 3D array, got {data.ndim}D array with shape {data.shape}")
                return False
            
            data = data.astype(np.float32)
            
            if not np.isfinite(data).all():
                response = QMessageBox.question(
                    self, "Data Quality Issue",
                    "Data contains NaN or infinite values. Replace with zeros?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.Yes:
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    return False
            
            return self._process_loaded_data(data, path)
                
        except Exception as e:
            error_details = traceback.format_exc()
            error_dialog = create_error_dialog(self, "Error", f"Failed to load NumPy data:\n{str(e)}\n\nDetails:\n{error_details}")
            error_dialog.exec_()
            if self.parent_window:
                self.parent_window.statusBar().showMessage("Error loading NumPy data")
            return False
    
    def _process_loaded_data(self, data, path):
        if self.parent_window:
            if self.parent_window.original_data is None:
                self.parent_window.original_data = data.copy()
            self.parent_window.data = data.copy()
            self.parent_window.processed_data = None
            
            self.parent_window.display_image_data()
            
            non_zero_pixels = np.count_nonzero(data)
            total_pixels = data.size
            max_value = np.max(data)
            
            emission_type, oi_qe, al_qe, auto_qe_enabled = self.get_qe_settings()
            
            if auto_qe_enabled:
                applied_qe = oi_qe if emission_type == "OI" else al_qe
                qe_text = f"QE: {applied_qe:.3f}"
            else:
                qe_text = "Manual QE"
            
            info_text = (f"Loaded {os.path.basename(path)} (shape={data.shape}, "
                        f"{non_zero_pixels}/{total_pixels} non-zero pixels, max={max_value:.2f}) "
                        f"- Ready for {emission_type} emission ({qe_text})")
            self.parent_window.info_label.setText(info_text)
            self.parent_window.info_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.parent_window.update_batch_button_state()
                
        self.numpy_data = data
        self.data_label.setText(f"Loaded: {os.path.basename(path)} (shape={data.shape})")
        self.data_label.setStyleSheet("color: green;")
        
        if self.parent_window:
            self.parent_window.statusBar().showMessage(f"Image data loaded successfully: {os.path.basename(path)}")
        
        return True

class TimeSeriesViewerDialog(QDialog):
    def __init__(self, parent, image_cube, times, log_transform=False, levels=None):
        super().__init__(parent)
        self.parent = parent
        self.image_cube = image_cube
        self.times = times
        self.log_transform = log_transform
        self.levels = levels
        
        self.setWindowTitle("Time Series Viewer")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.image_view = pg.ImageView()
        
        self.image_view.setImage(
            self.image_cube,
            axes={'t': 0, 'y': 1, 'x': 2},
            xvals=self.times,
            levels=self.levels
        )
        
        self.image_view.setPredefinedGradient("viridis")
        
        layout.addWidget(self.image_view)
        
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.time_label)
        
        self.image_view.timeLine.sigPositionChanged.connect(self.update_time_label)
        self.update_time_label()
        
        button_layout = QHBoxLayout()
        
        self.save_frame_btn = QPushButton("Save Current Frame to Main View")
        self.save_frame_btn.clicked.connect(self.save_current_frame)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(self.save_frame_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def update_time_label(self):
        idx = self.image_view.currentIndex
        if 0 <= idx < len(self.times):
            t = self.times[idx]
            transform_mode = "log₁₀" if self.log_transform else "linear"
            self.time_label.setText(f"Time: {t:g} s   (Display mode: {transform_mode})")
        
    def save_current_frame(self):
        current_index = self.image_view.currentIndex
        
        if 0 <= current_index < len(self.times):
            current_frame = self.image_cube[current_index]
            current_time = self.times[current_index]
            
            self.parent.processed_data = current_frame.copy()
            self.parent.data = current_frame.copy()
            
            self.parent.display_image_data()
            
            self.parent.info_label.setText(f"Saved frame at t={current_time:g}s to main view")
            self.parent.info_label.setStyleSheet("color: green; font-weight: bold;")
            self.parent.statusBar().showMessage(f"Saved frame at t={current_time:g}s to main view")
            
            QMessageBox.information(self, "Frame Saved", f"Frame at t={current_time:g}s has been saved to the main view.")
        else:
            QMessageBox.warning(self, "Warning", "Invalid time index.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pyxel Sensor Simulator")
        self.setWindowIcon(QIcon("pyxel_logo.png"))
        
        self.settings = QSettings("SPOTTA", "PyxelExposureSim")
        self.config = None
        self.data = None
        self.original_data = None
        self.processed_data = None
        self.simulation_in_progress = False
        self.updating_display = False
        
        self.result_dataset = None
        self.image_cube = None
        self.time_points = None
        
        self.current_analysis_results = None
        
        self.setMinimumSize(1400, 900)
        
        self.setup_ui()
        self.create_menus()
        self.statusBar().showMessage("Ready")
        self.load_settings()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        left_widget = self.create_left_panel()
        
        right_widget = self.create_right_panel()
        
        main_layout.addWidget(left_widget, 3)
        main_layout.addWidget(right_widget, 1)
        
        self.add_logos(main_layout)
    
    def create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        image_container = self.create_image_views()
        layout.addWidget(image_container, 4)
        
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setSpacing(8)
        
        buttons_container = self.create_action_buttons()
        buttons_container.setMaximumWidth(380)
        buttons_container.setMinimumWidth(320)
        
        analysis_container = self.create_analysis_buttons()
        analysis_container.setMaximumWidth(380)
        analysis_container.setMinimumWidth(320)
        
        self.batch_results_panel = BatchResultsPanel(self)
        self.batch_results_panel.setMaximumWidth(380)
        self.batch_results_panel.setMinimumWidth(320)
        
        controls_layout.addWidget(buttons_container)
        controls_layout.addWidget(analysis_container)
        controls_layout.addWidget(self.batch_results_panel)
        
        layout.addWidget(controls_container, 0)
        
        self.info_label = QLabel("Ready - Load data and configuration to begin")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold; padding: 8px; background-color: #f0f0f0; border-radius: 4px;")
        layout.addWidget(self.info_label, 0)
        
        return widget
    
    def create_image_views(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setSpacing(5)
        
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        self.original_image_view = PgImageView(self)
        original_layout.addWidget(self.original_image_view)
        
        processed_group = QGroupBox("Processed Image (Pyxel Output)")
        processed_layout = QVBoxLayout(processed_group)
        self.processed_image_view = PgImageView(self)
        processed_layout.addWidget(self.processed_image_view)
        
        layout.addWidget(original_group)
        layout.addWidget(processed_group)
        
        return container
    
    def create_action_buttons(self):
        container = QGroupBox("Image Tools")
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(5)
        
        self.debris_overview_btn = QPushButton("Debris Overview")
        self.debris_overview_btn.clicked.connect(self.show_debris_overview)
        self.debris_overview_btn.setToolTip("Show overview of all debris in image")
        self.debris_overview_btn.setMaximumHeight(28)
        self.debris_overview_btn.setStyleSheet("font-size: 10px; padding: 4px;")
        
        self.pixel_finder_btn = QPushButton("Find Pixels")
        self.pixel_finder_btn.clicked.connect(self.show_pixel_finder)
        self.pixel_finder_btn.setToolTip("Find and highlight pixels above threshold")
        self.pixel_finder_btn.setMaximumHeight(28)
        self.pixel_finder_btn.setStyleSheet("font-size: 10px; padding: 4px;")
        
        row1_layout.addWidget(self.debris_overview_btn)
        row1_layout.addWidget(self.pixel_finder_btn)
        
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(5)
        
        self.stats_btn = QPushButton("Image Stats")
        self.stats_btn.clicked.connect(self.show_image_statistics)
        self.stats_btn.setToolTip("Show detailed image statistics")
        self.stats_btn.setMaximumHeight(28)
        self.stats_btn.setStyleSheet("font-size: 10px; padding: 4px;")
        
        self.time_series_btn = QPushButton("Time Series")
        self.time_series_btn.clicked.connect(self.show_time_series)
        self.time_series_btn.setStyleSheet("background-color: #a83c7c; color: white; font-weight: bold; font-size: 10px; padding: 4px;")
        self.time_series_btn.setEnabled(False)
        self.time_series_btn.setToolTip("View simulation time series data")
        self.time_series_btn.setMaximumHeight(28)
        
        row2_layout.addWidget(self.stats_btn)
        row2_layout.addWidget(self.time_series_btn)
        
        layout.addLayout(row1_layout)
        layout.addLayout(row2_layout)
        
        return container
    
    def create_analysis_buttons(self):
        container = QGroupBox("Analysis & Export")
        layout = QVBoxLayout(container)
        layout.setSpacing(5)
        layout.setContentsMargins(8, 8, 8, 8)
        
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(5)
        
        self.analysis_btn = QPushButton("ADU Analysis")
        self.analysis_btn.clicked.connect(self.show_comprehensive_analysis)
        self.analysis_btn.setToolTip("Show comprehensive ADU analysis from last Pyxel simulation")
        self.analysis_btn.setStyleSheet("background-color: #8b5a3c; color: white; font-weight: bold; font-size: 10px; padding: 4px;")
        self.analysis_btn.setEnabled(False)
        self.analysis_btn.setMaximumHeight(28)
        
        row1_layout.addWidget(self.analysis_btn)
        
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(5)
        
        self.save_processed_btn = QPushButton("Save Processed")
        self.save_processed_btn.clicked.connect(self.save_processed_image)
        self.save_processed_btn.setStyleSheet("background-color: #ca8c3a; color: white; font-weight: bold; font-size: 10px; padding: 4px;")
        self.save_processed_btn.setEnabled(False)
        self.save_processed_btn.setToolTip("Save processed image to file")
        self.save_processed_btn.setMaximumHeight(28)
        
        self.export_analysis_btn = QPushButton("Export CSV")
        self.export_analysis_btn.clicked.connect(self.export_analysis_csv)
        self.export_analysis_btn.setStyleSheet("background-color: #6a8a3a; color: white; font-weight: bold; font-size: 10px; padding: 4px;")
        self.export_analysis_btn.setEnabled(False)
        self.export_analysis_btn.setToolTip("Export analysis results to CSV")
        self.export_analysis_btn.setMaximumHeight(28)
        
        row2_layout.addWidget(self.save_processed_btn)
        row2_layout.addWidget(self.export_analysis_btn)
        
        layout.addLayout(row1_layout)
        layout.addLayout(row2_layout)
        
        return container
    
    def create_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        
        self.tab_widget = QTabWidget()
        
        self.yaml_tab = QWidget()
        yaml_layout = QVBoxLayout(self.yaml_tab)
        self.yaml_text = QPlainTextEdit()
        self.yaml_text.setReadOnly(True)
        self.yaml_text.setStyleSheet("font-family: 'Courier New', monospace; font-size: 10px;")
        yaml_layout.addWidget(self.yaml_text)
        
        self.load_image_tab = LoadImageTab()
        self.load_image_tab.set_parent_window(self)
        
        self.batch_tab = BatchProcessingTab()
        self.batch_tab.set_parent_window(self)
        
        self.statistics_panel = StatisticsPanel(self)
        
        self.tab_widget.addTab(self.yaml_tab, "YAML Config")
        self.tab_widget.addTab(self.load_image_tab, "Load Image")
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")
        self.tab_widget.addTab(self.statistics_panel, "Statistics")
        
        self.tab_widget.setTabToolTip(0, "View and load YAML configuration")
        self.tab_widget.setTabToolTip(1, "Load and visualize image data (NumPy/FITS) with QE controls")
        self.tab_widget.setTabToolTip(2, "Batch process multiple NumPy files with Pyxel and dynamic QE")
        self.tab_widget.setTabToolTip(3, "View simulation statistics and ADU analysis")
        
        layout.addWidget(self.tab_widget, 1)
        
        processing_controls = QGroupBox("Processing Controls")
        controls_layout = QHBoxLayout(processing_controls)
        controls_layout.setSpacing(10)
        
        self.apply_button = QPushButton("Perform Exposure")
        self.apply_button.clicked.connect(self.run_exposure_mode)
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6600; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 10px 15px;
                border-radius: 6px;
                border: none;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #ff8833;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.batch_process_button = QPushButton("Start Batch Processing")
        self.batch_process_button.clicked.connect(self.start_batch_from_main)
        self.batch_process_button.setStyleSheet("""
            QPushButton {
                background-color: #0066cc; 
                color: white; 
                font-weight: bold; 
                font-size: 14px;
                padding: 10px 15px;
                border-radius: 6px;
                border: none;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #0077dd;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.batch_process_button.setEnabled(False)
        
        controls_layout.addWidget(self.apply_button)
        controls_layout.addWidget(self.batch_process_button)
        controls_layout.addStretch()
        
        layout.addWidget(processing_controls, 0)
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        return widget
    
    def add_logos(self, main_layout):
        logo_widget = QWidget()
        logo_layout = QHBoxLayout(logo_widget)
        logo_layout.addStretch()
        
        for logo_file in ["FraunhoferLogo.png", "StrathclydeLogo.png", "ESALogo.png"]:
            if os.path.exists(logo_file):
                lbl = QLabel()
                pm = safe_load_image(logo_file)
                if pm:
                    lbl.setPixmap(pm.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    logo_layout.addWidget(lbl)
        
        logo_widget.setMaximumHeight(120)
    
    def start_batch_from_main(self):
        self.tab_widget.setCurrentWidget(self.batch_tab)
        self.batch_tab.start_batch_processing()
    
    def update_batch_button_state(self):
        if hasattr(self, 'batch_process_button') and hasattr(self, 'batch_tab'):
            has_config = self.config is not None
            has_files = hasattr(self.batch_tab, 'file_list') and self.batch_tab.file_list.count() > 0
            has_output = hasattr(self.batch_tab, 'output_dir_edit') and bool(self.batch_tab.output_dir_edit.text())
            
            self.batch_process_button.setEnabled(has_config and has_files and has_output)

    def on_tab_changed(self, index):
        if index == 2:
            self.batch_tab.update_process_button_state()
            self.update_batch_button_state()
        elif index == 3:
            if hasattr(self, 'current_analysis_results') and self.current_analysis_results:
                self.statistics_panel.update_stats(self.current_analysis_results)

    def show_comprehensive_analysis(self):
        if not self.current_analysis_results:
            QMessageBox.warning(self, "No Analysis", "No analysis results available. Run a Pyxel simulation first.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Comprehensive ADU Analysis")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        analysis_text = QTextEdit()
        analysis_text.setReadOnly(True)
        
        analysis_content = self.format_analysis_results(self.current_analysis_results)
        analysis_text.setPlainText(analysis_content)
        
        layout.addWidget(analysis_text)
        
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(analysis_content))
        
        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(self.export_analysis_csv)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(export_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def format_analysis_results(self, analysis):
        lines = []
        
        lines.append("=== COMPREHENSIVE ADU ANALYSIS ===")
        lines.append(f"Analysis Time: {analysis.get('analysis_timestamp', 'N/A')}")
        lines.append(f"Filename: {analysis.get('original_filename', 'N/A')}")
        lines.append(f"Emission Type: {analysis.get('emission_type', 'N/A')}")
        lines.append(f"Applied QE: {analysis.get('applied_qe', 'N/A')}")
        lines.append(f"Auto QE Mode: {'Yes' if analysis.get('auto_qe_enabled', False) else 'No'}")
        lines.append("")
        
        lines.append("--- ORIGINAL DATA ---")
        lines.append(f"Shape: {analysis.get('original_shape', 'N/A')}")
        lines.append(f"Min/Max/Mean: {analysis.get('original_min', 0):.3f} / {analysis.get('original_max', 0):.3f} / {analysis.get('original_mean', 0):.3f}")
        lines.append(f"Non-zero pixels: {analysis.get('original_nonzero_pixels', 0):,} / {analysis.get('original_total_pixels', 0):,}")
        lines.append("")
        
        lines.append("--- PYXEL OUTPUT (ADU) ---")
        lines.append(f"Output Shape: {analysis.get('output_shape', 'N/A')}")
        lines.append(f"Time Frames: {analysis.get('time_frames', 1)}")
        lines.append(f"Min ADU: {analysis.get('min_adu', 0):.3f}")
        lines.append(f"Max ADU: {analysis.get('max_adu', 0):.3f}")
        lines.append(f"Mean ADU: {analysis.get('mean_adu', 0):.3f}")
        lines.append(f"Median ADU: {analysis.get('median_adu', 0):.3f}")
        lines.append(f"Std ADU: {analysis.get('std_adu', 0):.3f}")
        lines.append("")
        
        lines.append("--- PEAK & CENTER ANALYSIS ---")
        lines.append(f"Peak ADU: {analysis.get('peak_adu', 0):.3f} at ({analysis.get('peak_row', 0)}, {analysis.get('peak_col', 0)})")
        lines.append(f"Center Pixel ADU: {analysis.get('center_pixel_adu', 0):.3f} at ({analysis.get('center_pixel_row', 0)}, {analysis.get('center_pixel_col', 0)})")
        lines.append(f"Center Signal ADU: {analysis.get('center_pixel_signal_adu', 0):.3f}")
        lines.append("")
        
        lines.append("--- BACKGROUND & NOISE ---")
        lines.append(f"Background Mean: {analysis.get('background_mean', 0):.3f} ADU")
        lines.append(f"Background Std: {analysis.get('background_std', 0):.3f} ADU")
        lines.append(f"Background Median: {analysis.get('background_median', 0):.3f} ADU")
        lines.append(f"Signal-to-Noise Ratio: {analysis.get('snr', 0):.2f}")
        lines.append("")
        
        lines.append("--- SIGNAL STATISTICS ---")
        lines.append(f"Non-zero pixels: {analysis.get('nonzero_pixels', 0):,} / {analysis.get('total_pixels', 0):,}")
        lines.append(f"Fill factor: {analysis.get('fill_factor', 0)*100:.3f}%")
        lines.append(f"Total ADU: {analysis.get('total_adu', 0):.3f}")
        lines.append(f"Total signal ADU: {analysis.get('total_signal_adu', 0):.3f}")
        lines.append("")
        
        lines.append("--- DYNAMIC RANGE ---")
        lines.append(f"Max possible ADU: {analysis.get('max_possible_adu', 65535):,}")
        lines.append(f"Dynamic range used: {analysis.get('dynamic_range_utilization', 0)*100:.3f}%")
        lines.append(f"Saturated pixels: {analysis.get('saturated_pixels', 0):,}")
        lines.append(f"Saturation percentage: {analysis.get('saturation_percentage', 0):.4f}%")
        lines.append("")
        
        if analysis.get('fwhm_average', 0) > 0:
            lines.append("--- PSF CHARACTERISTICS ---")
            lines.append(f"FWHM Horizontal: {analysis.get('fwhm_horizontal', 0):.2f} pixels")
            lines.append(f"FWHM Vertical: {analysis.get('fwhm_vertical', 0):.2f} pixels")
            lines.append(f"FWHM Average: {analysis.get('fwhm_average', 0):.2f} pixels")
            lines.append("")
        
        lines.append("--- PERCENTILE ANALYSIS ---")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            lines.append(f"{p:2d}th percentile: {analysis.get(f'percentile_{p}', 0):.3f} ADU")
        
        return "\n".join(lines)
    
    def export_analysis_csv(self):
        if not self.current_analysis_results:
            QMessageBox.warning(self, "No Analysis", "No analysis results to export.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.current_analysis_results.get('original_filename', 'analysis')
        base_name = os.path.splitext(filename)[0]
        emission_type = self.current_analysis_results.get('emission_type', 'unknown')
        auto_qe = "auto" if self.current_analysis_results.get('auto_qe_enabled', False) else "manual"
        default_filename = f"{base_name}_adu_analysis_{emission_type}_{auto_qe}_{timestamp}.csv"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Results", default_filename, "CSV Files (*.csv)")
        
        if not filepath:
            return
        
        try:
            csv_data = []
            csv_data.append(['Parameter', 'Value', 'Unit', 'Category'])
            
            analysis = self.current_analysis_results
            
            csv_data.append(['Original_Filename', analysis.get('original_filename', ''), '', 'File Info'])
            csv_data.append(['Analysis_Timestamp', analysis.get('analysis_timestamp', ''), '', 'File Info'])
            csv_data.append(['Emission_Type', analysis.get('emission_type', ''), '', 'File Info'])
            csv_data.append(['Applied_QE', analysis.get('applied_qe', ''), '', 'File Info'])
            csv_data.append(['Auto_QE_Enabled', analysis.get('auto_qe_enabled', False), '', 'File Info'])
            csv_data.append([''])
            
            csv_data.append(['Original_Shape_H', analysis.get('original_shape', [0, 0])[0], 'pixels', 'Original Data'])
            csv_data.append(['Original_Shape_W', analysis.get('original_shape', [0, 0])[1], 'pixels', 'Original Data'])
            csv_data.append(['Original_Min', analysis.get('original_min', 0), 'value', 'Original Data'])
            csv_data.append(['Original_Max', analysis.get('original_max', 0), 'value', 'Original Data'])
            csv_data.append(['Original_Mean', analysis.get('original_mean', 0), 'value', 'Original Data'])
            csv_data.append(['Original_Std', analysis.get('original_std', 0), 'value', 'Original Data'])
            csv_data.append(['Original_Nonzero_Pixels', analysis.get('original_nonzero_pixels', 0), 'pixels', 'Original Data'])
            csv_data.append([''])
            
            csv_data.append(['Output_Shape_H', analysis.get('output_shape', [0, 0])[0], 'pixels', 'ADU Statistics'])
            csv_data.append(['Output_Shape_W', analysis.get('output_shape', [0, 0])[1], 'pixels', 'ADU Statistics'])
            csv_data.append(['Time_Frames', analysis.get('time_frames', 1), 'frames', 'ADU Statistics'])
            csv_data.append(['Min_ADU', analysis.get('min_adu', 0), 'ADU', 'ADU Statistics'])
            csv_data.append(['Max_ADU', analysis.get('max_adu', 0), 'ADU', 'ADU Statistics'])
            csv_data.append(['Mean_ADU', analysis.get('mean_adu', 0), 'ADU', 'ADU Statistics'])
            csv_data.append(['Median_ADU', analysis.get('median_adu', 0), 'ADU', 'ADU Statistics'])
            csv_data.append(['Std_ADU', analysis.get('std_adu', 0), 'ADU', 'ADU Statistics'])
            csv_data.append(['Peak_ADU', analysis.get('peak_adu', 0), 'ADU', 'ADU Statistics'])
            csv_data.append(['Peak_Row', analysis.get('peak_row', 0), 'pixels', 'ADU Statistics'])
            csv_data.append(['Peak_Col', analysis.get('peak_col', 0), 'pixels', 'ADU Statistics'])
            csv_data.append([''])
            
            csv_data.append(['Center_Pixel_ADU', analysis.get('center_pixel_adu', 0), 'ADU', 'Center Analysis'])
            csv_data.append(['Center_Pixel_Row', analysis.get('center_pixel_row', 0), 'pixels', 'Center Analysis'])
            csv_data.append(['Center_Pixel_Col', analysis.get('center_pixel_col', 0), 'pixels', 'Center Analysis'])
            csv_data.append(['Center_Signal_ADU', analysis.get('center_pixel_signal_adu', 0), 'ADU', 'Center Analysis'])
            csv_data.append([''])
            
            csv_data.append(['Background_Mean', analysis.get('background_mean', 0), 'ADU', 'Background & Noise'])
            csv_data.append(['Background_Std', analysis.get('background_std', 0), 'ADU', 'Background & Noise'])
            csv_data.append(['Background_Median', analysis.get('background_median', 0), 'ADU', 'Background & Noise'])
            csv_data.append(['Signal_to_Noise_Ratio', analysis.get('snr', 0), 'ratio', 'Background & Noise'])
            csv_data.append([''])
            
            csv_data.append(['Nonzero_Pixels', analysis.get('nonzero_pixels', 0), 'pixels', 'Signal Statistics'])
            csv_data.append(['Total_Pixels', analysis.get('total_pixels', 0), 'pixels', 'Signal Statistics'])
            csv_data.append(['Fill_Factor', analysis.get('fill_factor', 0), 'fraction', 'Signal Statistics'])
            csv_data.append(['Total_ADU', analysis.get('total_adu', 0), 'ADU', 'Signal Statistics'])
            csv_data.append(['Total_Signal_ADU', analysis.get('total_signal_adu', 0), 'ADU', 'Signal Statistics'])
            csv_data.append([''])
            
            csv_data.append(['Max_Possible_ADU', analysis.get('max_possible_adu', 65535), 'ADU', 'Dynamic Range'])
            csv_data.append(['Dynamic_Range_Utilization', analysis.get('dynamic_range_utilization', 0), 'fraction', 'Dynamic Range'])
            csv_data.append(['Saturated_Pixels', analysis.get('saturated_pixels', 0), 'pixels', 'Dynamic Range'])
            csv_data.append(['Saturation_Percentage', analysis.get('saturation_percentage', 0), 'percent', 'Dynamic Range'])
            csv_data.append([''])
            
            csv_data.append(['FWHM_Horizontal', analysis.get('fwhm_horizontal', 0), 'pixels', 'PSF'])
            csv_data.append(['FWHM_Vertical', analysis.get('fwhm_vertical', 0), 'pixels', 'PSF'])
            csv_data.append(['FWHM_Average', analysis.get('fwhm_average', 0), 'pixels', 'PSF'])
            csv_data.append([''])
            
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                csv_data.append([f'Percentile_{p}', analysis.get(f'percentile_{p}', 0), 'ADU', 'Percentiles'])
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(csv_data)
            
            auto_qe_status = "automatic" if analysis.get('auto_qe_enabled', False) else "manual (from YAML)"
            QMessageBox.information(self, "Export Complete", 
                                   f"Analysis results exported to:\n{filepath}\n\n"
                                   f"Key metrics:\n"
                                   f"• Emission Type: {analysis.get('emission_type', 'N/A')}\n"
                                   f"• QE Mode: {auto_qe_status}\n"
                                   f"• Applied QE: {analysis.get('applied_qe', 0)}\n"
                                   f"• Peak ADU: {analysis.get('peak_adu', 0):.2f}\n"
                                   f"• Center ADU: {analysis.get('center_pixel_adu', 0):.2f}\n"
                                   f"• SNR: {analysis.get('snr', 0):.2f}\n"
                                   f"• Background: {analysis.get('background_mean', 0):.3f} ADU")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting analysis: {str(e)}")

    def save_processed_image(self):
        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "No processed image to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emission_type = getattr(self, 'current_emission_type', 'unknown')
        auto_qe_mode = "auto" if hasattr(self, 'current_analysis_results') and self.current_analysis_results and self.current_analysis_results.get('auto_qe_enabled', False) else "manual"
        default_filename = f"pyxel_processed_{emission_type}_{auto_qe_mode}_{timestamp}"
        
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Processed Image", default_filename,
            "FITS Files (*.fits);;NumPy Files (*.npy);;CSV Files (*.csv);;PNG Files (*.png);;All Files (*.*)"
        )
        if not path:
            return
        
        try:
            self.statusBar().showMessage(f"Saving processed image to {path}...")
            QApplication.processEvents()
            
            ext = os.path.splitext(path)[1].lower()
            
            if ext == '.fits':
                header = fits.Header()
                if hasattr(self, 'current_analysis_results') and self.current_analysis_results:
                    header['EMISSION'] = (self.current_analysis_results.get('emission_type', 'unknown'), 'Emission type')
                    header['APPLIEDQE'] = (self.current_analysis_results.get('applied_qe', 0), 'Applied quantum efficiency')
                    header['AUTOQE'] = (self.current_analysis_results.get('auto_qe_enabled', False), 'Auto QE mode enabled')
                    header['PEAKADU'] = (self.current_analysis_results.get('peak_adu', 0), 'Peak ADU value')
                    header['CENTERADU'] = (self.current_analysis_results.get('center_pixel_adu', 0), 'Center pixel ADU')
                    header['SNR'] = (self.current_analysis_results.get('snr', 0), 'Signal-to-noise ratio')
                header['CREATOR'] = 'Pyxel Simulator'
                header['TIMESTAMP'] = datetime.now().isoformat()
                
                hdu = fits.PrimaryHDU(data=self.processed_data.astype(np.float32), header=header)
                hdu.writeto(path, overwrite=True)
            elif ext == '.npy':
                np.save(path, self.processed_data)
            elif ext == '.csv':
                np.savetxt(path, self.processed_data, delimiter=',', fmt="%.6f")
            elif ext in ['.png', '.jpg', '.jpeg']:
                normalized = ((self.processed_data - self.processed_data.min()) / 
                             (self.processed_data.max() - self.processed_data.min()) * 255).astype(np.uint8)
                Image.fromarray(normalized).save(path)
            else:
                np.save(path, self.processed_data)
            
            self.info_label.setText(f"Saved processed image: {os.path.basename(path)}")
            self.info_label.setStyleSheet("color: green; font-weight: bold; padding: 8px; background-color: #e8f5e8; border-radius: 4px;")
            self.statusBar().showMessage(f"Processed image saved to {path}")
            
        except Exception as exc:
            error_details = traceback.format_exc()
            error_dialog = create_error_dialog(self, "Error", f"Failed saving processed image:\n{str(exc)}\n\nDetails:\n{error_details}")
            error_dialog.exec_()
            self.statusBar().showMessage("Error saving processed image")

    def show_time_series(self):
        if self.image_cube is None or self.time_points is None:
            QMessageBox.warning(self, "Warning", "No time series data available.\n\nRun a simulation first to generate time series data.")
            return
            
        dialog = TimeSeriesViewerDialog(
            self, 
            self.image_cube, 
            self.time_points,
            log_transform=False
        )
        dialog.exec_()
    
    def update_geometry_display(self, config):
        pass
            
    def show_pixel_finder(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded.")
            return
        
        active_view = self.processed_image_view if self.processed_data is not None else self.original_image_view
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Pixel Finder")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        threshold_group = QGroupBox("Threshold Settings")
        threshold_layout = QFormLayout(threshold_group)
        
        non_zero_data = self.data[self.data > 0]
        if len(non_zero_data) > 0:
            default_threshold = np.percentile(non_zero_data, 50)
            max_val = np.max(self.data)
        else:
            default_threshold = 0.1
            max_val = 1.0
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, max_val)
        self.threshold_spin.setValue(default_threshold)
        self.threshold_spin.setDecimals(6)
        self.threshold_spin.setSingleStep(default_threshold / 10)
        
        threshold_layout.addRow("Threshold Value:", self.threshold_spin)
        
        results_group = QGroupBox("Found Pixels")
        results_layout = QVBoxLayout(results_group)
        
        self.pixel_list = QPlainTextEdit()
        self.pixel_list.setReadOnly(True)
        self.pixel_list.setMaximumHeight(200)
        results_layout.addWidget(self.pixel_list)
        
        button_layout = QHBoxLayout()
        
        find_btn = QPushButton("Find Pixels")
        find_btn.clicked.connect(self.find_pixels_above_threshold)
        
        center_brightest_btn = QPushButton("Center on Brightest")
        center_brightest_btn.clicked.connect(lambda: active_view.center_on_maximum())
        
        auto_zoom_btn = QPushButton("Auto Zoom to Content")
        auto_zoom_btn.clicked.connect(lambda: active_view.auto_zoom_to_content())
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(find_btn)
        button_layout.addWidget(center_brightest_btn)
        button_layout.addWidget(auto_zoom_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addWidget(threshold_group)
        layout.addWidget(results_group)
        layout.addLayout(button_layout)
        
        self.pixel_finder_dialog = dialog
        
        self.find_pixels_above_threshold()
        
        dialog.exec_()
    
    def find_pixels_above_threshold(self):
        if not hasattr(self, 'pixel_finder_dialog') or self.data is None:
            return
        
        threshold = self.threshold_spin.value()
        
        y_indices, x_indices = np.where(self.data >= threshold)
        values = self.data[y_indices, x_indices]
        
        sorted_indices = np.argsort(values)[::-1]
        
        results = []
        results.append(f"Found {len(y_indices)} pixels >= {threshold:.6f}")
        results.append(f"{'Rank':<4} {'X':<6} {'Y':<6} {'Value':<12}")
        results.append("-" * 32)
        
        max_show = min(100, len(y_indices))
        for i in range(max_show):
            idx = sorted_indices[i]
            x, y, val = x_indices[idx], y_indices[idx], values[idx]
            results.append(f"{i+1:<4} {x:<6} {y:<6} {val:<12.6f}")
        
        if len(y_indices) > 100:
            results.append(f"... and {len(y_indices) - 100} more pixels")
        
        self.pixel_list.setPlainText("\n".join(results))
    
    def display_image_data(self):
        if self.original_data is not None:
            self.original_image_view.display_array(self.original_data, "Original")
        elif self.data is not None:
            self.original_image_view.display_array(self.data, "Current Data")
        
        if self.processed_data is not None:
            self.processed_image_view.display_array(self.processed_data, "Processed")
        else:
            self.processed_image_view.display_array(np.zeros((10, 10)), "No processed data")
            
    def show_image_statistics(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded.")
            return
        
        stats = {}
        stats['Shape'] = f"{self.data.shape[0]} x {self.data.shape[1]}"
        stats['Total Pixels'] = self.data.size
        stats['Non-zero Pixels'] = np.count_nonzero(self.data)
        stats['Zero Pixels'] = np.sum(self.data == 0)
        stats['Data Type'] = str(self.data.dtype)
        
        stats['Minimum'] = np.min(self.data)
        stats['Maximum'] = np.max(self.data)
        stats['Mean'] = np.mean(self.data)
        stats['Median'] = np.median(self.data)
        stats['Std Deviation'] = np.std(self.data)
        
        non_zero_data = self.data[self.data > 0]
        if len(non_zero_data) > 0:
            stats['Non-zero Mean'] = np.mean(non_zero_data)
            stats['Non-zero Median'] = np.median(non_zero_data)
            stats['Non-zero Std'] = np.std(non_zero_data)
        else:
            stats['Non-zero Mean'] = 0
            stats['Non-zero Median'] = 0
            stats['Non-zero Std'] = 0
        
        percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'{p}th Percentile'] = np.percentile(self.data, p)
        
        max_loc = np.unravel_index(np.argmax(self.data), self.data.shape)
        min_loc = np.unravel_index(np.argmin(self.data), self.data.shape)
        stats['Max Location (Y,X)'] = f"({max_loc[0]}, {max_loc[1]})"
        stats['Min Location (Y,X)'] = f"({min_loc[0]}, {min_loc[1]})"
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Image Statistics")
        dialog.setMinimumSize(400, 600)
        
        layout = QVBoxLayout(dialog)
        
        stats_text = QPlainTextEdit()
        stats_text.setReadOnly(True)
        
        text_lines = []
        text_lines.append("IMAGE STATISTICS")
        text_lines.append("=" * 40)
        
        text_lines.append("\nBASIC INFORMATION:")
        text_lines.append(f"Shape: {stats['Shape']}")
        text_lines.append(f"Total Pixels: {stats['Total Pixels']:,}")
        text_lines.append(f"Non-zero Pixels: {stats['Non-zero Pixels']:,} ({100*stats['Non-zero Pixels']/stats['Total Pixels']:.2f}%)")
        text_lines.append(f"Zero Pixels: {stats['Zero Pixels']:,}")
        text_lines.append(f"Data Type: {stats['Data Type']}")
        
        text_lines.append("\nVALUE STATISTICS:")
        text_lines.append(f"Minimum: {stats['Minimum']:.6f} at {stats['Min Location (Y,X)']}")
        text_lines.append(f"Maximum: {stats['Maximum']:.6f} at {stats['Max Location (Y,X)']}")
        text_lines.append(f"Mean: {stats['Mean']:.6f}")
        text_lines.append(f"Median: {stats['Median']:.6f}")
        text_lines.append(f"Std Deviation: {stats['Std Deviation']:.6f}")
        
        text_lines.append("\nNON-ZERO STATISTICS:")
        text_lines.append(f"Non-zero Mean: {stats['Non-zero Mean']:.6f}")
        text_lines.append(f"Non-zero Median: {stats['Non-zero Median']:.6f}")
        text_lines.append(f"Non-zero Std: {stats['Non-zero Std']:.6f}")
        
        text_lines.append("\nPERCENTILES:")
        for p in percentiles:
            text_lines.append(f"{p:2d}th: {stats[f'{p}th Percentile']:.6f}")
        
        stats_text.setPlainText("\n".join(text_lines))
        layout.addWidget(stats_text)
        
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(stats_text.toPlainText()))
        
        active_view = self.processed_image_view if self.processed_data is not None else self.original_image_view
        center_max_btn = QPushButton("Center on Maximum")
        center_max_btn.clicked.connect(lambda: active_view.center_on_maximum())
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        
        button_layout.addWidget(copy_btn)
        button_layout.addWidget(center_max_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()

    def show_debris_overview(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded.")
            return
            
        y_indices, x_indices = np.nonzero(self.data)
        
        if len(y_indices) == 0:
            QMessageBox.warning(self, "Warning", "No debris pixels found in image.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Debris Overview")
        dialog.setMinimumSize(900, 600)
        
        layout = QVBoxLayout(dialog)
        
        splitter = QSplitter(Qt.Horizontal)
        
        full_view = pg.ImageView()
        full_view.setImage(self.data)
        full_view.ui.roiBtn.hide()
        full_view.ui.menuBtn.hide()
        full_view.setPredefinedGradient("plasma")
        
        zoom_view = pg.ImageView()
        zoom_view.setImage(self.data)
        zoom_view.ui.roiBtn.hide()
        zoom_view.ui.menuBtn.hide()
        zoom_view.setPredefinedGradient("plasma")
        
        roi = pg.ROI([0, 0], [100, 100], pen=pg.mkPen('r', width=2))
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        full_view.addItem(roi)
        
        padding = 50
        y_min = max(0, np.min(y_indices) - padding)
        y_max = min(self.data.shape[0], np.max(y_indices) + padding)
        x_min = max(0, np.min(x_indices) - padding)
        x_max = min(self.data.shape[1], np.max(x_indices) + padding)
        
        roi.setPos(x_min, y_min)
        roi.setSize([x_max - x_min, y_max - y_min])
        
        def update_zoomed_view():
            zoom_view.setImage(roi.getArrayRegion(self.data, full_view.getImageItem()))
        
        roi.sigRegionChanged.connect(update_zoomed_view)
        update_zoomed_view()
        
        splitter.addWidget(full_view)
        splitter.addWidget(zoom_view)
        splitter.setSizes([400, 500])
        
        layout.addWidget(QLabel("Full Image (Red box shows current debris region)"))
        layout.addWidget(splitter)
        layout.addWidget(QLabel("Drag the red box to explore different areas. Resize it with the corner handles."))
        
        info_text = (f"Found {len(y_indices)} debris pixels\n"
                    f"Bounding box: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]\n"
                    f"Max value: {np.max(self.data[self.data > 0]):.2f}")
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-weight: bold; background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(info_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        
        dialog.exec_()

    def run_exposure_mode(self):
        """New single frame processing using batch-style logic with manual QE control"""
        if self.simulation_in_progress:
            QMessageBox.warning(self, "Warning", "Simulation already in progress.")
            return
            
        if self.config is None:
            QMessageBox.warning(self, "Warning", "Please load a YAML configuration for simulation.")
            return
        
        if self.data is None:
            QMessageBox.warning(self, "Warning", "No image data loaded to process.")
            return
        
        emission_type, oi_qe, al_qe, auto_qe_enabled = self.load_image_tab.get_qe_settings()
        self.current_emission_type = emission_type
        
        try:
            # Store original data if not already stored
            if self.original_data is None and self.data is not None:
                self.original_data = self.data.copy()
            
            # Use the EXACT same approach as batch processing
            self.progress_dialog = QProgressDialog("Running simulation...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowTitle("Exposure Mode Simulation")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(300)
            self.progress_dialog.setCancelButton(None)
            
            self.simulation_in_progress = True
            self.apply_button.setEnabled(False)
            
            # Create worker that uses the same logic as batch processing but with manual QE control
            self.worker = SingleFrameWorker(
                config=self.config,
                image_data=self.data,
                emission_type=emission_type,
                oi_qe=oi_qe,
                al_qe=al_qe,
                auto_qe_enabled=auto_qe_enabled,
                original_data=self.original_data
            )
            
            self.worker.finished.connect(self.handle_single_frame_complete)
            self.worker.error.connect(self.handle_simulation_error)
            self.worker.progress.connect(self.update_progress)
            self.worker.start()
            
            qe_mode = f"QE: {oi_qe if emission_type == 'OI' else al_qe:.3f}" if auto_qe_enabled else "Manual QE"
            self.info_label.setText(f"Simulation in progress ({emission_type} emission, {qe_mode})...")
            self.info_label.setStyleSheet("color: blue; font-weight: bold; padding: 8px; background-color: #e8f0ff; border-radius: 4px;")
            self.statusBar().showMessage("Running exposure mode simulation...")
            
        except Exception as exc:
            self.simulation_in_progress = False
            self.apply_button.setEnabled(True)
            error_details = traceback.format_exc()
            error_dialog = create_error_dialog(self, "Error", f"Failed to start simulation:\n{str(exc)}\n\nDetails:\n{error_details}")
            error_dialog.exec_()

    def handle_single_frame_complete(self, results):
        """Handle completion of single frame simulation using batch-style results"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        self.simulation_in_progress = False
        self.apply_button.setEnabled(True)
        
        try:
            result_datatree = results['result_datatree']
            analysis_results = results['analysis_results']
            final_image = results['final_image']
            image_cube = results['image_cube']
            times = results['times']
            
            # Store results for time series viewing
            self.result_dataset = result_datatree["bucket"].to_dataset() if "bucket" in result_datatree else None
            self.image_cube = image_cube
            self.time_points = times
            self.time_series_btn.setEnabled(True)
            
            # Update display with processed data
            self.updating_display = True
            self.processed_data = final_image.copy()
            self.data = final_image.copy()
            self.display_image_data()
            self.updating_display = False
            
            # Enable action buttons
            self.save_processed_btn.setEnabled(True)
            self.analysis_btn.setEnabled(True)
            self.export_analysis_btn.setEnabled(True)
            
            # Store analysis results
            self.current_analysis_results = analysis_results
            
            # Update statistics panel
            self.statistics_panel.update_stats(self.current_analysis_results)
            self.tab_widget.setCurrentWidget(self.statistics_panel)
            
            # Show completion message
            emission_type = analysis_results.get('emission_type', 'Unknown')
            applied_qe = analysis_results.get('applied_qe', 0)
            auto_qe_enabled = analysis_results.get('auto_qe_enabled', False)
            peak_adu = analysis_results.get('peak_adu', 0)
            center_adu = analysis_results.get('center_pixel_adu', 0)
            snr = analysis_results.get('snr', 0)
            
            qe_text = f"QE:{applied_qe:.3f}" if auto_qe_enabled else "Manual QE"
            self.info_label.setText(f"Simulation complete ({emission_type}, {qe_text})! Peak: {peak_adu:.1f} ADU, Center: {center_adu:.1f} ADU, SNR: {snr:.1f}")
            self.info_label.setStyleSheet("color: green; font-weight: bold; padding: 8px; background-color: #e8f5e8; border-radius: 4px;")
            self.statusBar().showMessage("Exposure mode simulation completed successfully")
            
            # Offer time series view if multiple frames
            if len(times) > 1:
                response = QMessageBox.question(
                    self, "Time Series Available",
                    f"Simulation generated time series data with {len(times)} frames.\n\nWould you like to view the time series?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.Yes:
                    dialog = TimeSeriesViewerDialog(self, image_cube, times)
                    dialog.exec_()
            
        except Exception as e:
            error_details = traceback.format_exc()
            error_dialog = create_error_dialog(self, "Error", f"Error processing simulation results:\n{str(e)}\n\nDetails:\n{error_details}")
            error_dialog.exec_()
            self.statusBar().showMessage("Error processing simulation results")
        
    def handle_simulation_error(self, error_message):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        self.simulation_in_progress = False
        self.updating_display = False
        self.apply_button.setEnabled(True)
        
        error_dialog = create_error_dialog(self, "Simulation Error", error_message)
        error_dialog.exec_()
        self.info_label.setText("Simulation failed.")
        self.info_label.setStyleSheet("color: red; font-weight: bold; padding: 8px; background-color: #ffe8e8; border-radius: 4px;")
        self.statusBar().showMessage("Exposure mode simulation failed")
        
    def update_progress(self, message):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()

    def create_menus(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        open_yaml_action = QAction("Open YAML", self)
        open_yaml_action.triggered.connect(self.open_yaml)
        open_yaml_action.setShortcut("Ctrl+Y")
        file_menu.addAction(open_yaml_action)
        
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)
    
    def open_batch_directory(self):
        self.tab_widget.setCurrentWidget(self.batch_tab)
        self.batch_tab.browse_directory()
        
    def open_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open YAML Config", "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        if not path:
            return
            
        self.statusBar().showMessage(f"Loading YAML from {path}...")
        QApplication.processEvents()
        
        try:
            self.config = pyxel.load(path)
            
            with open(path, 'r') as f:
                txt = f.read()
            self.yaml_text.setPlainText(txt)
            
            self.update_geometry_display(self.config)
            
            self.statusBar().showMessage(f"YAML loaded from {path}")
            self.info_label.setText(f"Loaded YAML: {os.path.basename(path)}")
            self.info_label.setStyleSheet("color: green; font-weight: bold; padding: 8px; background-color: #e8f5e8; border-radius: 4px;")
            
            if hasattr(self, 'batch_tab'):
                self.batch_tab.update_process_button_state()
            self.update_batch_button_state()
            
        except Exception as exc:
            error_details = traceback.format_exc()
            error_dialog = create_error_dialog(self, "Error", f"Failed to load YAML:\n{str(exc)}\n\nDetails:\n{error_details}")
            error_dialog.exec_()
            self.statusBar().showMessage(f"Error loading YAML from {path}")
            self.info_label.setText(f"Error loading YAML: {os.path.basename(path)}")
            self.info_label.setStyleSheet("color: red; font-weight: bold; padding: 8px; background-color: #ffe8e8; border-radius: 4px;")
    
    def open_image_data(self, path=None):
        if path is None:
            result = self.load_image_tab.load_numpy_data()
            if result:
                self.tab_widget.setCurrentWidget(self.load_image_tab)
        else:
            if self.load_image_tab.load_numpy_data(path):
                self.tab_widget.setCurrentWidget(self.load_image_tab)
            
    def get_detector_dimensions(self):
        if self.config:
            try:
                detector = None
                if hasattr(self.config, 'cmos_detector'):
                    detector = self.config.cmos_detector
                elif hasattr(self.config, 'detector'):
                    detector = self.config.detector
                elif hasattr(self.config, 'ccd_detector'):
                    detector = self.config.ccd_detector
                
                if detector and hasattr(detector, 'geometry'):
                    return detector.geometry.row, detector.geometry.col
            except Exception as e:
                pass
        
        if self.data is not None:
            return self.data.shape
        else:
            return 1000, 1000
        
    def save_settings(self):
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/pos", self.pos())
        
        self.settings.setValue("ui/active_tab", self.tab_widget.currentIndex())
        
        self.settings.sync()
        self.statusBar().showMessage("Settings saved successfully")
        
    def load_settings(self):
        wsize = self.settings.value("window/size")
        if wsize:
            self.resize(wsize)
        wpos = self.settings.value("window/pos")
        if wpos:
            self.move(wpos)
            
        active_tab = self.settings.value("ui/active_tab", 0, type=int)
        if 0 <= active_tab < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(active_tab)
            
    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
   main()