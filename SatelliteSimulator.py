#!/usr/bin/env python3
"""
ESA Viewer with debris-tracking camera for visualization and photon conversion
Coordinate system consistency, all units in meters
Proper ECI/ECEF transformations, WGS84 ellipsoid, consistent angular units
"""
import sys, os
import numpy as np
import pandas as pd
import colorsys
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, GCRS, ITRS, EarthLocation
from astropy.constants import h, c
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from PyQt5.QtCore import Qt, QDateTime, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFormLayout, QDoubleSpinBox, QSpinBox, QListWidget, QTabWidget, QGroupBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QDateTimeEdit, QFrame,
    QStackedLayout, QCheckBox, QComboBox, QScrollArea, QSizePolicy, QTextEdit, QDialog,
    QProgressBar,QLineEdit 
)
#from qt_material import apply_stylesheet
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.image as mpimg
import json

# Path to your Blue Marble texture
EARTH_TEXTURE_PATH = r"C:/Users/cib24188/Desktop/Projects/ESA/earth.jpg"

# WGS84 Constants
# Higher precision WGS84 Constants (matching MATLAB's values)
WGS84_A = 6378137.0000000000  # Semi-major axis in meters (exact)
WGS84_F = 1/298.257223563000  # Flattening (higher precision)
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # First eccentricity squared

# Also consider using the exact GM and omega values
WGS84_GM = 3.986004418e14  # m^3/s^2 (exact WGS84 value)
WGS84_OMEGA = 7.292115e-5  # rad/s (Earth rotation rate)

# RGBA palette
_PALETTE = [
    (1.0,0.0,0.0,1.0), (0.0,1.0,0.0,1.0), (0.0,0.0,1.0,1.0),
    (1.0,1.0,0.0,1.0), (1.0,0.0,1.0,1.0), (0.0,1.0,1.0,1.0),
]

# ----------------------------------------------------------------------
# Coordinate Transformation Functions
# ----------------------------------------------------------------------

def eci_to_ecef(r_eci_m, time_utc):
    """
    Convert ECI coordinates to ECEF using astropy
    
    Args:
        r_eci_m: Position vector in ECI frame [m] (3,) numpy array
        time_utc: UTC time (astropy Time object)
    
    Returns:
        r_ecef_m: Position vector in ECEF frame [m] (3,) numpy array
    """
    # Ensure time precision
    if not hasattr(time_utc, 'precision') or time_utc.precision < 9:
        time_utc = Time(time_utc.utc.datetime, scale='utc', precision=9)
    
    # Use higher precision coordinates
    sc_eci = SkyCoord(
        x=r_eci_m[0]*u.m, y=r_eci_m[1]*u.m, z=r_eci_m[2]*u.m,
        frame=GCRS(obstime=time_utc), 
        representation_type='cartesian'
    )
    
    # Transform with explicit precision
    sc_ecef = sc_eci.transform_to(ITRS(obstime=time_utc))
    
    # Extract with higher precision
    r_ecef_m = np.array([
        sc_ecef.x.to_value(u.m),
        sc_ecef.y.to_value(u.m), 
        sc_ecef.z.to_value(u.m)
    ], dtype=np.float64)
    
    return r_ecef_m

def ecef_to_eci(r_ecef_m, time_utc):
    """
    Convert ECEF coordinates to ECI using astropy
    
    Args:
        r_ecef_m: Position vector in ECEF frame [m] (3,) numpy array
        time_utc: UTC time (astropy Time object)
    
    Returns:
        r_eci_m: Position vector in ECI frame [m] (3,) numpy array
    """
    # Convert to astropy SkyCoord in ITRS (ECEF)
    sc_ecef = SkyCoord(x=r_ecef_m[0]*u.m, y=r_ecef_m[1]*u.m, z=r_ecef_m[2]*u.m,
                       frame=ITRS(obstime=time_utc), representation_type='cartesian')
    
    # Transform to GCRS (ECI-like)
    sc_eci = sc_ecef.transform_to(GCRS(obstime=time_utc))
    
    # Extract position in meters
    r_eci_m = np.array([
        sc_eci.x.to(u.m).value,
        sc_eci.y.to(u.m).value,
        sc_eci.z.to(u.m).value
    ])
    
    return r_eci_m

def lla_to_ecef_wgs84(lat_deg, lon_deg, alt_m):
    """
    Convert LLA to ECEF using WGS84 ellipsoid
    
    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees
        alt_m: Altitude in meters above ellipsoid
    
    Returns:
        r_ecef_m: Position vector in ECEF frame [m] (3,) numpy array
    """
    # Convert to radians
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    # WGS84 ellipsoid calculations
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Radius of curvature in prime vertical
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
    
    # ECEF coordinates
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat
    
    return np.array([x, y, z])

def ecef_to_lla_wgs84(r_ecef_m):
    """
    Convert ECEF to LLA using WGS84 ellipsoid
    
    Args:
        r_ecef_m: Position vector in ECEF frame [m] (3,) numpy array
    
    Returns:
        tuple: (lat_deg, lon_deg, alt_m)
    """
    x, y, z = r_ecef_m
    
    # Longitude (straightforward)
    lon_rad = np.arctan2(y, x)
    
    # Iterative solution for latitude and altitude
    p = np.sqrt(x**2 + y**2)
    lat_rad = np.arctan2(z, p * (1 - WGS84_E2))
    
    # Iterate to converge
    for _ in range(5):  # Usually converges in 2-3 iterations
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad)**2)
        alt_m = p / np.cos(lat_rad) - N
        lat_rad = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt_m)))
    
    # Final altitude calculation
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad)**2)
    alt_m = p / np.cos(lat_rad) - N
    
    # Convert to degrees
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    
    return lat_deg, lon_deg, alt_m

# ----------------------------------------------------------------------
# Enhanced Camera and Visibility Functions
# ----------------------------------------------------------------------

def convert_ecef_to_pixel(P_obj_ecef_m, C_ecef_m, P_target_ecef_m, focal_mm, pixel_size_mm, image_resolution):
    """
    Projects 3D ECEF position onto 2D image plane using pinhole camera model
    All inputs in ECEF frame with meters
    
    Args:
        P_obj_ecef_m: Object position in ECEF [m] (3,)
        C_ecef_m: Camera position in ECEF [m] (3,)  
        P_target_ecef_m: Camera look-at target in ECEF [m] (3,)
        focal_mm: Focal length [mm]
        pixel_size_mm: Pixel size (dx, dy) [mm]
        image_resolution: (W, H) in pixels
        
    Returns:
        tuple: (u_px, v_px) pixel coordinates or None if behind camera
    """
    # Calculate forward vector from camera to target (camera's boresight)
    f = P_target_ecef_m - C_ecef_m
    f_norm = np.linalg.norm(f)
    if f_norm == 0: 
        return None
    f /= f_norm
    
    # Define initial world "up" direction for reference
    k = np.array([0.0, 0.0, 1.0])
    
    # Check if camera is pointing nearly parallel to reference up vector
    if abs(np.dot(f, k)) > 0.99:
        k = np.array([1.0, 0.0, 0.0])
    
    # Calculate right vector perpendicular to forward and reference vectors
    r = np.cross(f, k)
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        k = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(f, k)) > 0.99:
            k = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, k)
        r_norm = np.linalg.norm(r)
        if r_norm == 0: 
            return None
    r /= r_norm
    
    # Calculate true up vector perpendicular to forward and right
    u = np.cross(r, f)
    
    # Construct rotation matrix from world to camera coordinates
    R_cam_to_world = np.vstack([r, u, f]).T
    # Transform object point from world to camera coordinates
    P_cam = R_cam_to_world.T @ (P_obj_ecef_m - C_ecef_m)
    
    # Check if object is behind camera plane
    if P_cam[2] <= 1e-6:
        return None
    
    # Extract pixel size in mm
    dx, dy = pixel_size_mm
    W, H = image_resolution
    
    # Project 3D camera coordinates to 2D image plane
    x_prime = P_cam[0] / P_cam[2]
    y_prime = P_cam[1] / P_cam[2]
    
    x_img_mm = focal_mm * x_prime
    y_img_mm = focal_mm * y_prime
    
    # Define image center coordinates
    u0, v0 = W/2.0, H/2.0
    
    # Convert to pixel coordinates
    u_px = (x_img_mm / dx) + u0
    v_px = v0 - (y_img_mm / dy)
    
    return (u_px, v_px)

def calculate_elevation_angle_wgs84(satellite_ecef_m, debris_lat_deg, debris_lon_deg, debris_alt_m):
    """
    Calculate elevation angle using WGS84 ellipsoid
    
    Args:
        satellite_ecef_m: Satellite position in ECEF [m] (3,)
        debris_lat_deg: Debris latitude [degrees]
        debris_lon_deg: Debris longitude [degrees]  
        debris_alt_m: Debris altitude [m]
    
    Returns:
        elevation_angle_deg: Elevation angle [degrees]
    """
    try:
        # Input validation
        if not (-90 <= debris_lat_deg <= 90):
            print(f"WARNING: Invalid latitude {debris_lat_deg}, clamping to [-90, 90]")
            debris_lat_deg = np.clip(debris_lat_deg, -90, 90)
        
        if not (-180 <= debris_lon_deg <= 180):
            print(f"WARNING: Invalid longitude {debris_lon_deg}, wrapping")
            debris_lon_deg = ((debris_lon_deg + 180) % 360) - 180
        
        if debris_alt_m < 0:
            print(f"WARNING: Negative altitude {debris_alt_m}m, setting to 0")
            debris_alt_m = max(0, debris_alt_m)
        
        # Convert debris LLA to ECEF using WGS84
        debris_ecef_m = lla_to_ecef_wgs84(debris_lat_deg, debris_lon_deg, debris_alt_m)
        
        # Vector from debris to satellite
        sat_vector_m = satellite_ecef_m - debris_ecef_m
        sat_distance_m = np.linalg.norm(sat_vector_m)
        
        if sat_distance_m < 1.0:  # Less than 1 meter
            return 90.0
        
        # Calculate local "up" vector (normal to WGS84 ellipsoid at debris location)
        lat_rad = np.radians(debris_lat_deg)
        lon_rad = np.radians(debris_lon_deg)
        
        # WGS84 ellipsoid normal (not spherical!)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # Normal to ellipsoid surface
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
        up_vector = np.array([
            cos_lat * cos_lon,
            cos_lat * sin_lon,
            (1 - WGS84_E2) * sin_lat
        ])
        up_vector /= np.linalg.norm(up_vector)  # Normalize
        
        # Normalize satellite vector
        sat_vector_unit = sat_vector_m / sat_distance_m
        
        # Calculate elevation angle
        cos_zenith = np.dot(sat_vector_unit, up_vector)
        cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
        
        zenith_angle_rad = np.arccos(cos_zenith)
        elevation_angle_deg = 90.0 - np.degrees(zenith_angle_rad)
        
        return elevation_angle_deg
        
    except Exception as e:
        print(f"ERROR in WGS84 elevation calculation: {e}")
        return 45.0

def apply_angular_correction(photons, electrons, satellite_ecef_m, debris_lat_deg, debris_lon_deg, debris_alt_m):
    """
    Apply photon*cos angular correction using WGS84 elevation calculation
    
    Args:
        photons: Number of photons before correction
        electrons: Number of electrons before correction  
        satellite_ecef_m: Satellite position in ECEF [m] (3,)
        debris_lat_deg: Debris latitude [degrees]
        debris_lon_deg: Debris longitude [degrees]
        debris_alt_m: Debris altitude [m]
    
    Returns:
        tuple: (corrected_photons, corrected_electrons, elevation_angle, cos_factor)
    """
    if photons <= 0 and electrons <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate elevation angle using WGS84
    elevation_angle_deg = calculate_elevation_angle_wgs84(
        satellite_ecef_m, debris_lat_deg, debris_lon_deg, debris_alt_m
    )
    
    # Apply cos(α) correction
    if elevation_angle_deg > -10:  # Allow slightly below horizon
        cos_factor = np.cos(np.radians(elevation_angle_deg))
        cos_factor = np.clip(cos_factor, 0.0, 1.0)
        
        corrected_photons = photons * cos_factor
        corrected_electrons = electrons * cos_factor
        
        # Debug output for extreme cases only
        if elevation_angle_deg > 80 or elevation_angle_deg < -5:
            print(f"DEBUG: α={elevation_angle_deg:.1f}°, cos(α)={cos_factor:.3f}, photons: {photons:.1f}→{corrected_photons:.1f}")
    else:
        # Far below horizon
        cos_factor = 0.0
        corrected_photons = 0.0
        corrected_electrons = 0.0
        print(f"DEBUG: α={elevation_angle_deg:.1f}° (below horizon), zeroing signal")
    
    return corrected_photons, corrected_electrons, elevation_angle_deg, cos_factor

def is_visible_wgs84_occlusion(C_ecef_m, P_obj_ecef_m):
    """
    Earth occlusion check using WGS84 ellipsoid
    
    Args:
        C_ecef_m: Camera position in ECEF [m] (3,)
        P_obj_ecef_m: Object position in ECEF [m] (3,)
    
    Returns:
        bool: True if line of sight exists, False if Earth blocks the view
    """
    # Ensure positions are above Earth surface
    C_norm = np.linalg.norm(C_ecef_m)
    P_norm = np.linalg.norm(P_obj_ecef_m)
    
    if C_norm < WGS84_B or P_norm < WGS84_B:  # Below minor axis
        return False
    
    # Vector from camera to object
    d = P_obj_ecef_m - C_ecef_m
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        return True  # Same position
    
    d_unit = d / d_norm
    
    # Find closest point on line to Earth center
    t = -np.dot(C_ecef_m, d_unit)
    
    # Check if closest point is on the line segment
    if t < 0 or t > d_norm:
        return True  # Closest point not on segment
    
    closest_point = C_ecef_m + t * d_unit
    
    # Check if closest point is inside WGS84 ellipsoid
    # For simplicity, use spherical approximation with mean radius
    # For full WGS84, need more complex ellipsoid intersection
    mean_radius = (2 * WGS84_A + WGS84_B) / 3
    closest_dist = np.linalg.norm(closest_point)
    
    return closest_dist >= mean_radius

# ----------------------------------------------------------------------
# Satellite Manager Tab
# ----------------------------------------------------------------------
class SatManagerTab(QWidget):
    """
    Satellite management with proper orbital elements in meters for semi-major axis
    """
    satsChanged = pyqtSignal()
    timeChanged = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sats = []
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        main_hlay = QHBoxLayout(self)
        main_hlay.setSpacing(15)
        main_hlay.setContentsMargins(10, 10, 10, 10)
        
        # Create scrollable area for the left panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(400)
        scroll_area.setMaximumWidth(450)
        
        scroll_content = QWidget()
        left_vlay = QVBoxLayout(scroll_content)
        left_vlay.setSpacing(15)
        left_vlay.setContentsMargins(10, 10, 10, 20)

        # Time Controls Section
        time_box = QGroupBox("Time Settings")
        time_form = QFormLayout()
        time_form.setVerticalSpacing(8)
        time_form.setContentsMargins(15, 15, 15, 15)
        
        self.constellation_time = QDateTimeEdit(QDateTime.currentDateTimeUtc(), self)
        self.constellation_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.constellation_time.setToolTip("Epoch time for satellite constellation generation")
        self.constellation_time.setMinimumHeight(25)
        time_form.addRow("Constellation Epoch:", self.constellation_time)
        
        # Time step buttons
        time_step_widget = QWidget()
        time_step_layout = QHBoxLayout(time_step_widget)
        time_step_layout.setContentsMargins(0, 5, 0, 0)
        time_step_layout.setSpacing(5)
        
        self.step_hour_back_btn = QPushButton("-1h")
        self.step_hour_forward_btn = QPushButton("+1h") 
        self.step_day_forward_btn = QPushButton("+1d")
        self.sync_viz_time_btn = QPushButton("Sync Viz")
        
        for btn in [self.step_hour_back_btn, self.step_hour_forward_btn, 
                   self.step_day_forward_btn, self.sync_viz_time_btn]:
            btn.setMinimumHeight(22)
            btn.setMaximumHeight(25)
            btn.setMaximumWidth(60)
        
        self.step_hour_back_btn.clicked.connect(lambda: self.step_time(-3600))
        self.step_hour_forward_btn.clicked.connect(lambda: self.step_time(3600))
        self.step_day_forward_btn.clicked.connect(lambda: self.step_time(86400))
        self.sync_viz_time_btn.clicked.connect(self.sync_from_viz_time)
        
        time_step_layout.addWidget(self.step_hour_back_btn)
        time_step_layout.addWidget(self.step_hour_forward_btn)
        time_step_layout.addWidget(self.step_day_forward_btn)
        time_step_layout.addWidget(self.sync_viz_time_btn)
        
        time_form.addRow("Quick adjust:", time_step_widget)
        time_box.setLayout(time_form)
        left_vlay.addWidget(time_box)

        # Manual Satellite Section
        manual_box = QGroupBox("Manual Satellite")
        manual_form = QFormLayout()
        manual_form.setVerticalSpacing(6)
        manual_form.setContentsMargins(15, 15, 15, 15)

        # Semi-major axis in meters
        self.a = QDoubleSpinBox(); self.a.setRange(6500000, 50000000); self.a.setValue(7000000); self.a.setMinimumHeight(22)
        self.a.setDecimals(0); self.a.setSuffix(" m")
        self.e = QDoubleSpinBox(); self.e.setRange(0, 0.99); self.e.setDecimals(3); self.e.setMinimumHeight(22)
        self.i = QDoubleSpinBox(); self.i.setRange(0, 180); self.i.setValue(30); self.i.setMinimumHeight(22)
        self.raan = QDoubleSpinBox(); self.raan.setRange(0, 360); self.raan.setValue(0); self.raan.setMinimumHeight(22)
        self.argp = QDoubleSpinBox(); self.argp.setRange(0, 360); self.argp.setValue(0); self.argp.setMinimumHeight(22)
        self.nu = QDoubleSpinBox(); self.nu.setRange(0, 360); self.nu.setValue(0); self.nu.setMinimumHeight(22)

        def add_row(lbl, widget, tip):
            widget.setToolTip(tip)
            manual_form.addRow(lbl, widget)

        add_row("a [m]:", self.a, "Semi-major axis in meters")
        add_row("e:", self.e, "Eccentricity")
        add_row("i [°]:", self.i, "Inclination")
        add_row("RAAN [°]:", self.raan, "Longitude of ascending node")
        add_row("ArgP [°]:", self.argp, "Argument of perigee")
        add_row("ν [°]:", self.nu, "True anomaly")

        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 8, 0, 0)
        button_layout.setSpacing(8)
        
        self.add_btn = QPushButton("Add Satellite")
        self.remove_btn = QPushButton("Remove Selected")
        
        for btn in [self.add_btn, self.remove_btn]:
            btn.setMinimumHeight(25)
            btn.setMaximumHeight(28)
        
        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.remove_btn)
        manual_form.addRow(button_widget)
        manual_box.setLayout(manual_form)
        left_vlay.addWidget(manual_box)

        # Walker Constellation Section
        walker_box = QGroupBox("Walker Constellation (T/P/F)")
        walker_form = QFormLayout()
        walker_form.setVerticalSpacing(6)
        walker_form.setContentsMargins(15, 15, 15, 15)

        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setSpacing(6)
        
        row1_widget = QWidget()
        row1_layout = QHBoxLayout(row1_widget)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(10)
        
        # Left column - Semi-major axis in meters
        left_col = QFormLayout()
        left_col.setVerticalSpacing(4)
        self.walker_a = QDoubleSpinBox(); self.walker_a.setRange(6500000, 50000000); self.walker_a.setValue(7000000); self.walker_a.setMinimumHeight(22)
        self.walker_a.setDecimals(0); self.walker_a.setSuffix(" m")
        self.walker_e = QDoubleSpinBox(); self.walker_e.setRange(0, 0.99); self.walker_e.setDecimals(3); self.walker_e.setValue(0); self.walker_e.setMinimumHeight(22)
        self.walker_i = QDoubleSpinBox(); self.walker_i.setRange(0, 180); self.walker_i.setValue(30); self.walker_i.setMinimumHeight(22)
        left_col.addRow("a [m]:", self.walker_a)
        left_col.addRow("e:", self.walker_e)
        left_col.addRow("i [°]:", self.walker_i)
        
        # Right column
        right_col = QFormLayout()
        right_col.setVerticalSpacing(4)
        self.walker_raan = QDoubleSpinBox(); self.walker_raan.setRange(0, 360); self.walker_raan.setValue(0); self.walker_raan.setMinimumHeight(22)
        self.walker_argp = QDoubleSpinBox(); self.walker_argp.setRange(0, 360); self.walker_argp.setValue(0); self.walker_argp.setMinimumHeight(22)
        self.walker_nu = QDoubleSpinBox(); self.walker_nu.setRange(0, 360); self.walker_nu.setValue(0); self.walker_nu.setMinimumHeight(22)
        right_col.addRow("RAAN [°]:", self.walker_raan)
        right_col.addRow("ArgP [°]:", self.walker_argp)
        right_col.addRow("ν [°]:", self.walker_nu)
        
        left_widget = QWidget()
        left_widget.setLayout(left_col)
        right_widget = QWidget()
        right_widget.setLayout(right_col)
        
        row1_layout.addWidget(left_widget)
        row1_layout.addWidget(right_widget)
        params_layout.addWidget(row1_widget)
        
        walker_form.addRow(params_widget)
        
        # Walker-specific parameters
        self.tw = QSpinBox(); self.tw.setRange(1, 1000); self.tw.setValue(12); self.tw.setMinimumHeight(22)
        self.pw = QSpinBox(); self.pw.setRange(1, 100); self.pw.setValue(3); self.pw.setMinimumHeight(22)
        self.fw = QSpinBox(); self.fw.setRange(0, 100); self.fw.setValue(1); self.fw.setMinimumHeight(22)
        
        walker_form.addRow("T (total):", self.tw)
        walker_form.addRow("P (planes):", self.pw)
        walker_form.addRow("F (phasing):", self.fw)
        
        # Advanced parameters
        self.walker_raan_offset = QDoubleSpinBox(); self.walker_raan_offset.setRange(0, 360); self.walker_raan_offset.setValue(0); self.walker_raan_offset.setMinimumHeight(22)
        walker_form.addRow("RAAN offset [°]:", self.walker_raan_offset)
        
        # Inclination variation
        inc_widget = QWidget()
        inc_layout = QHBoxLayout(inc_widget)
        inc_layout.setContentsMargins(0, 0, 0, 0)
        inc_layout.setSpacing(8)
        
        self.walker_vary_inclination = QCheckBox("Vary inclination")
        self.walker_inclination_spread = QDoubleSpinBox(); self.walker_inclination_spread.setRange(0, 30); self.walker_inclination_spread.setValue(0); self.walker_inclination_spread.setMinimumHeight(22)
        self.walker_inclination_spread.setEnabled(False)
        
        inc_layout.addWidget(self.walker_vary_inclination)
        inc_layout.addWidget(QLabel("Spread [°]:"))
        inc_layout.addWidget(self.walker_inclination_spread)
        walker_form.addRow(inc_widget)

        # Generate button
        self.gen_btn = QPushButton("Generate Walker Constellation")
        self.gen_btn.setMinimumHeight(30)
        self.gen_btn.setMaximumHeight(35)
        self.gen_btn.setStyleSheet("font-weight: bold; background-color: #0066cc; padding: 5px;")
        walker_form.addRow(self.gen_btn)
        
        walker_box.setLayout(walker_form)
        left_vlay.addWidget(walker_box)

        self.walker_vary_inclination.toggled.connect(self.walker_inclination_spread.setEnabled)

        left_vlay.addStretch()
        scroll_area.setWidget(scroll_content)
        main_hlay.addWidget(scroll_area, 1)

        # Satellite List Section
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(5, 0, 0, 0)
        list_layout.setSpacing(5)
        
        list_label = QLabel("Satellite List")
        list_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        list_layout.addWidget(list_label)
        
        self.listw = QListWidget()
        self.listw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.listw.setMinimumWidth(300)
        list_layout.addWidget(self.listw)
        
        main_hlay.addWidget(list_container, 1)

        # Connections
        self.add_btn.clicked.connect(self._add_manual)
        self.remove_btn.clicked.connect(self._remove)
        self.gen_btn.clicked.connect(self._generate_walker)
        self.constellation_time.dateTimeChanged.connect(self._on_time_changed)

    def _on_time_changed(self, datetime):
        """Handle time changes and emit signal for synchronization"""
        self.timeChanged.emit(datetime)
        print(f"Satellite tab time changed: {datetime.toString('yyyy-MM-dd hh:mm:ss')}")

    def step_time(self, seconds):
        """Adjust the constellation time by the specified number of seconds"""
        current_time = self.constellation_time.dateTime()
        new_time = current_time.addSecs(seconds)
        self.constellation_time.setDateTime(new_time)

    def sync_from_viz_time(self):
        """Sync time from visualization tab"""
        print("Manual sync requested from Satellite tab")

    def set_time_from_external(self, datetime):
        """Set time from external source without triggering signals"""
        self.constellation_time.blockSignals(True)
        self.constellation_time.setDateTime(datetime)
        self.constellation_time.blockSignals(False)
        print(f"Satellite tab time synced from external: {datetime.toString('yyyy-MM-dd hh:mm:ss')}")

    def _add_manual(self):
        """Add manually configured satellite with proper units"""
        epoch_time = Time(self.constellation_time.dateTime().toPyDateTime(), scale="utc")
        
        # Convert semi-major axis to correct units for poliastro
        sat = Orbit.from_classical(
            Earth,
            self.a.value() * u.m,  # Now in meters
            self.e.value() * u.one,
            self.i.value() * u.deg,
            self.raan.value() * u.deg,
            self.argp.value() * u.deg,
            self.nu.value() * u.deg,
            epoch=epoch_time
        )
        self._sats.append(sat)
        self._refresh_list()
        self.satsChanged.emit()

    def _remove(self):
        """Remove the selected satellite from the list"""
        i = self.listw.currentRow()
        if 0 <= i < len(self._sats):
            self._sats.pop(i)
            self._refresh_list()
            self.satsChanged.emit()

    def _generate_walker(self):
        """Generate Walker constellation with proper units"""
        T, P, F = self.tw.value(), self.pw.value(), self.fw.value()
        if T % P != 0:
            QMessageBox.warning(self, "Error", "T (total satellites) must be divisible by P (planes).")
            return
        sats_per_plane = T // P

        constellation_dt = self.constellation_time.dateTime().toPyDateTime()
        current_epoch = Time(constellation_dt.replace(microsecond=0), scale="utc")

        # Semi-major axis in meters
        a = self.walker_a.value() * u.m  # Now in meters
        e = self.walker_e.value() * u.one
        inc0 = self.walker_i.value() * u.deg
        raan0 = self.walker_raan.value() * u.deg
        argp = self.walker_argp.value() * u.deg
        nu0 = self.walker_nu.value() * u.deg
        raan_offset = self.walker_raan_offset.value() * u.deg

        vary_inc = self.walker_vary_inclination.isChecked()
        inc_spread = self.walker_inclination_spread.value() * u.deg if vary_inc else 0*u.deg

        new_sats = []
        for p in range(P):
            raan_i = raan0 + (360/P * p) * u.deg + raan_offset

            if vary_inc and P > 1:
                inc_i = inc0 + ((p/(P-1) - 0.5) * inc_spread)
            else:
                inc_i = inc0

            for s in range(sats_per_plane):
                nu_ij = nu0 + (360/sats_per_plane * s) * u.deg + (360/T * p * F) * u.deg

                sat = Orbit.from_classical(Earth, a, e, inc_i, raan_i, argp, nu_ij, epoch=current_epoch)
                new_sats.append(sat)

        self._sats = new_sats
        self._refresh_list()
        self.satsChanged.emit()
        
        print(f"Generated Walker constellation with epoch: {current_epoch.iso}")

    def _refresh_list(self):
        """Update the UI list with the current satellites"""
        self.listw.clear()
        for idx in range(len(self._sats)):
            epoch_str = self._sats[idx].epoch.iso[:19] if hasattr(self._sats[idx], 'epoch') else "Unknown"
            # Display semi-major axis in km for readability
            a_km = self._sats[idx].a.to(u.km).value
            self.listw.addItem(f"Sat {idx+1} (a={a_km:.1f}km, Epoch: {epoch_str})")

    def get_satellites(self):
        """Return a list of (name, satellite) tuples"""
        return [(f"Sat {i+1}", sat) for i,sat in enumerate(self._sats)]

# ----------------------------------------------------------------------
# Debris Tab
# ----------------------------------------------------------------------
class DebrisTab(QWidget):
    """
    Tab for loading and displaying debris data from CSV or Excel files.
    Data should be in meters as per specification
    """
    debrisDataChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)

        btn = QPushButton("Load Debris…"); btn.clicked.connect(self.load)
        lay.addWidget(btn)

        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.table)
        self.df = pd.DataFrame()

    def load(self):
        """Load debris data from a CSV or Excel file"""
        p,_ = QFileDialog.getOpenFileName(self,"Open Debris","","*.csv *.xlsx")
        if not p: return

        try:
            df = pd.read_csv(p) if p.lower().endswith('.csv') else pd.read_excel(p)
            if 'Iter' in df.columns:
                df.rename(columns={'Iter':'Iteration'}, inplace=True)

            cols = ["Time","Iteration","Assembly_ID","Latitude","Longitude","Altitude"]
            
            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                QMessageBox.warning(self,"Missing Columns",
                                    f"The following required columns are missing: {', '.join(missing_cols)}")
                return
                
            self.df = df[cols].copy()
            # Ensure numeric types for relevant columns
            for col in ["Latitude", "Longitude", "Altitude", "Time", "Assembly_ID", "Iteration"]:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            self.df.dropna(subset=["Latitude", "Longitude", "Altitude", "Time", "Assembly_ID"], inplace=True)

            # Altitude should be in meters as specified
            print(f"Loaded debris data: {len(self.df)} rows")
            print(f"Altitude range: {self.df['Altitude'].min():.0f} to {self.df['Altitude'].max():.0f} meters")

            self.table.clear()
            self.table.setRowCount(len(self.df)); self.table.setColumnCount(len(cols))
            self.table.setHorizontalHeaderLabels(cols)

            for i,row in self.df.iterrows():
                for j,c in enumerate(cols):
                    self.table.setItem(i,j,QTableWidgetItem(str(row[c])))

            self.table.resizeColumnsToContents()
            self.debrisDataChanged.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Debris", f"Could not load or process debris file: {str(e)}")
            self.df = pd.DataFrame()
            self.table.clear()
            self.debrisDataChanged.emit()

# ----------------------------------------------------------------------
# Viz Tab with Coordinate Transformations
# ----------------------------------------------------------------------
class VizTab(QWidget):
    """
    3D visualization with proper ECI/ECEF coordinate transformations
    All satellite positions properly converted to ECEF for visualization
    """
    timeChanged = pyqtSignal(object)
    
    def __init__(self, satman, debris_tab, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.satman = satman
        self.debris_tab = debris_tab
        self.debris_tab.debrisDataChanged.connect(self.handle_debris_data_changed)
        self.debris_colors = {}
        self.lla_data = {}
        self.redraw_pending = False
        self.redraw_timer = QTimer()
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.timeout.connect(self.perform_redraw)
        
        self.auto_refresh_timer = QTimer()
        self.auto_refresh_timer.setInterval(2000)
        self.auto_refresh_timer.timeout.connect(self.check_and_redraw)
        self.is_visible = False
        
        self.lla_update_timer = QTimer()
        self.lla_update_timer.setSingleShot(True)
        self.lla_update_timer.timeout.connect(self.generate_lla_data)
        
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Visualisation ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.view_container = QWidget()
        self.view_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.opts['distance'] = 20000
        view_layout.addWidget(self.view)
        
        # Create main view with legend
        main_widget = QWidget()
        main_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self.view_container, 4)
        
        # Legend setup
        legend_scroll = QScrollArea()
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        legend_scroll.setMinimumWidth(180)
        legend_scroll.setMaximumWidth(250)
        legend_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        self.legend = QFrame()
        self.legend.setStyleSheet("background: rgba(0,0,0,0.7); border-radius:4px; color: white;")
        legend_layout = QVBoxLayout(self.legend)
        legend_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("SATELLITE LEGEND")
        title.setStyleSheet("font-size: 14px; color: white;")
        title.setAlignment(Qt.AlignCenter)
        legend_layout.addWidget(title)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #444;")
        legend_layout.addWidget(line)
        
        self.legend_content = QWidget()
        self.legend_content_layout = QVBoxLayout(self.legend_content)
        self.legend_content_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_content_layout.setSpacing(2)
        
        legend_layout.addWidget(self.legend_content)
        legend_scroll.setWidget(self.legend)
        self.legend_layout = legend_layout
        
        main_layout.addWidget(legend_scroll, 1)
        layout.addWidget(main_widget, 1)
        layout.addWidget(self.status_label)
        
        # Earth mesh (uses km for visualization)
        try:
            tex = mpimg.imread(EARTH_TEXTURE_PATH)
            tex = tex.astype(float)/255 if tex.dtype==np.uint8 or tex.max()>1 else tex
            tex = np.flipud(tex)
            h_t, w_t, _ = tex.shape
            md = gl.MeshData.sphere(rows=60, cols=120, radius=6378.0)  # Earth radius in km for visualization
            verts = md.vertexes()
            x,y,z = verts[:,0], verts[:,1], verts[:,2]
            lon = np.arctan2(y,x)
            lat_rad = np.arcsin(z/6378.0)
            
            ui = ((lon / (2 * np.pi) + 0.5) * (w_t - 1)).astype(int)
            vi = ((lat_rad / np.pi + 0.5) * (h_t - 1)).astype(int)
            
            ui = np.clip(ui, 0, w_t - 1)
            vi = np.clip(vi, 0, h_t - 1)
            vc = tex[vi,ui][:,:3]
            md.setVertexColors(vc)
            self.earth = gl.GLMeshItem(meshdata=md, smooth=False,
                                     drawFaces=True, drawEdges=False,
                                     glOptions='opaque')
            self.view.addItem(self.earth)
        except FileNotFoundError:
            print(f"ERROR: Earth texture file not found at {EARTH_TEXTURE_PATH}")
            QMessageBox.critical(self, "Texture Error", f"Earth texture file not found:\n{EARTH_TEXTURE_PATH}\nEarth will not be rendered.")
            self.earth = None
        except Exception as e:
            print(f"Error loading Earth texture: {e}")
            QMessageBox.warning(self, "Texture Error", f"Could not load Earth texture: {e}")
            self.earth = None
        
        # Controls layout
        controls_container = QWidget()
        controls_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        controls_main_layout = QVBoxLayout(controls_container)
        controls_main_layout.setContentsMargins(5, 5, 5, 5)
        controls_main_layout.setSpacing(10)
        
        # Time controls
        time_controls_group = QGroupBox("Time Controls")
        time_controls_layout = QHBoxLayout(time_controls_group)
        
        time_controls_layout.addWidget(QLabel("Satellite Time:"))
        self.snapshot = QDateTimeEdit(QDateTime.currentDateTimeUtc(), self)
        self.snapshot.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        time_controls_layout.addWidget(self.snapshot)
        time_controls_layout.addStretch()
        
        controls_main_layout.addWidget(time_controls_group)
        
        # Debris controls
        debris_controls_layout = QHBoxLayout()
        
        # Debris display mode
        debris_mode_group = QGroupBox("Debris Display Mode")
        debris_mode_layout = QVBoxLayout(debris_mode_group)
        
        self.show_individual_timestep = QCheckBox("Show Individual Timestep")
        self.show_individual_timestep.setChecked(True)
        
        self.show_all_timesteps = QCheckBox("Show All Timesteps")
        self.show_all_timesteps.setChecked(False)
        
        self.show_individual_timestep.toggled.connect(self._on_individual_toggled)
        self.show_all_timesteps.toggled.connect(self._on_all_toggled)
        
        debris_mode_layout.addWidget(self.show_individual_timestep)
        debris_mode_layout.addWidget(self.show_all_timesteps)
        
        debris_controls_layout.addWidget(debris_mode_group)
        
        # Timestep controls
        timestep_group = QGroupBox("Timestep Navigation")
        timestep_group_layout = QVBoxLayout(timestep_group)
        
        timestep_top_layout = QHBoxLayout()
        self.timestep_label = QLabel("Debris Timestep:")
        self.timestep_spinner = QDoubleSpinBox()
        self.timestep_spinner.setDecimals(1)
        self.timestep_spinner.setSingleStep(0.5)
        
        self.update_timestep_spinner_range()
        
        timestep_top_layout.addWidget(self.timestep_label)
        timestep_top_layout.addWidget(self.timestep_spinner)
        timestep_group_layout.addLayout(timestep_top_layout)
        
        timestep_buttons_layout = QHBoxLayout()
        self.prev_timestep_btn = QPushButton("◀ Previous")
        self.next_timestep_btn = QPushButton("Next ▶")
        timestep_buttons_layout.addWidget(self.prev_timestep_btn)
        timestep_buttons_layout.addWidget(self.next_timestep_btn)
        timestep_group_layout.addLayout(timestep_buttons_layout)
        
        debris_controls_layout.addWidget(timestep_group)
        
        # Tracking and display controls
        tracking_group = QGroupBox("Tracking & Display")
        tracking_layout = QVBoxLayout(tracking_group)
        
        tracking_row1 = QHBoxLayout()
        tracking_row1.addWidget(QLabel("Track Debris:"))
        self.tracked_debris = QComboBox()
        self.tracked_debris.addItem("Earth Center (0,0,0)", -1)
        tracking_row1.addWidget(self.tracked_debris)
        tracking_layout.addLayout(tracking_row1)
        
        self.toggle_debris = QCheckBox("Show Debris")
        self.toggle_debris.setChecked(True)
        tracking_layout.addWidget(self.toggle_debris)
        
        debris_controls_layout.addWidget(tracking_group)
        
        controls_main_layout.addLayout(debris_controls_layout)
        
        # Action buttons
        action_buttons_group = QGroupBox("Actions")
        action_buttons_layout = QHBoxLayout(action_buttons_group)
        
        self.update_all_btn = QPushButton("Update Window")
        self.update_all_btn.setStyleSheet("background-color: #ff9900; font-weight: bold; padding: 10px; font-size: 14px;")
        self.update_all_btn.setToolTip("Refreshes debris list, updates visualization, and forces redraw")
        
        action_buttons_layout.addWidget(self.update_all_btn)
        action_buttons_layout.addStretch()
        
        controls_main_layout.addWidget(action_buttons_group)
        layout.addWidget(controls_container)
        
        # Enhanced time change connection
        self.snapshot.dateTimeChanged.connect(self._on_time_changed)
        
        # Connect signals
        self.update_all_btn.clicked.connect(self.update_all)
        self.toggle_debris.stateChanged.connect(self.request_redraw)
        self.show_individual_timestep.toggled.connect(self.request_redraw)
        self.tracked_debris.currentIndexChanged.connect(self.request_redraw)
        self.timestep_spinner.valueChanged.connect(self.request_redraw)
        
        self.prev_timestep_btn.clicked.connect(self.previous_timestep)
        self.next_timestep_btn.clicked.connect(self.next_timestep)
        
        # Initialize lists
        self.orbits = []
        self.dots = []
        self.debris_dots = []
        
        QTimer.singleShot(100, self.update_debris_list)
        self._update_timestep_controls_visibility()

    def _on_time_changed(self, datetime):
        """Handle time changes - emit signal and update LLA data"""
        self.timeChanged.emit(datetime)
        
        self.lla_update_timer.stop()
        self.lla_update_timer.start(300)
        
        self.request_redraw()

    def generate_lla_data(self):
        """
        Generate LLA data with proper coordinate transformations
        Build self.lla_data with positions at t = 0, 0.5, 1, 1.5, 2 s after snapshot time
        """
        self.lla_data.clear()
        sats = self.satman.get_satellites()
        if not sats:
            return

        # Snapshot time, microseconds forced to 0
        snap_dt = self.snapshot.dateTime().toPyDateTime().replace(microsecond=0)
        snap = Time(snap_dt, scale="utc")

        time_grid_sec = [0, 0.5, 1.0, 1.5, 2.0]

        for name, sat in sats:
            lats, lons, alts, ts = [], [], [], []

            for dt_sec in time_grid_sec:
                t = snap + dt_sec * u.s
                dt_from_epoch = (t - sat.epoch).to_value(u.s)
                sat_now = sat.propagate(dt_from_epoch * u.s)
                
                # Get satellite position in ECI (from poliastro)
                r_eci_m = sat_now.rv()[0].to(u.m).value  # ECI position in meters
                
                # Convert ECI to ECEF for proper LLA calculation
                r_ecef_m = eci_to_ecef(r_eci_m, t)
                
                # Convert ECEF to LLA using WGS84
                lat_deg, lon_deg, alt_m = ecef_to_lla_wgs84(r_ecef_m)

                lats.append(lat_deg)
                lons.append(lon_deg)
                alts.append(alt_m / 1000.0)  # Convert to km for display
                ts.append(t)

            self.lla_data[name] = {
                "times": ts,
                "lat": np.array(lats),
                "lon": np.array(lons),
                "alt": np.array(alts)
            }

    def handle_debris_data_changed(self):
        """Slot to handle debris data changes from DebrisTab."""
        self.update_debris_list()
        self.update_timestep_spinner_range()
        self.request_redraw()

    def update_timestep_spinner_range(self):
        if not self.debris_tab.df.empty and 'Time' in self.debris_tab.df.columns:
            timesteps = sorted(pd.to_numeric(self.debris_tab.df['Time'], errors='coerce').dropna().unique())
            if len(timesteps) > 0:
                min_val, max_val = min(timesteps), max(timesteps)
                if min_val <= max_val:
                    self.timestep_spinner.setRange(min_val, max_val)
                    current_val = self.timestep_spinner.value()
                    if not (min_val <= current_val <= max_val) or len(self.orbits) == 0:
                         self.timestep_spinner.setValue(min_val)
                else:
                    self.timestep_spinner.setRange(min_val, min_val)
                    self.timestep_spinner.setValue(min_val)
            else:
                self.timestep_spinner.setRange(0, 100)
                self.timestep_spinner.setValue(0)
        else:
            self.timestep_spinner.setRange(0, 100)
            self.timestep_spinner.setValue(0)

    def _on_individual_toggled(self, checked):
        """Handle individual timestep mode toggle"""
        if checked:
            self.show_all_timesteps.setChecked(False)
        self._update_timestep_controls_visibility()
        self.request_redraw()
    
    def _on_all_toggled(self, checked):
        """Handle all timesteps mode toggle"""
        if checked:
            self.show_individual_timestep.setChecked(False)
        self._update_timestep_controls_visibility()
        self.request_redraw()
    
    def _update_timestep_controls_visibility(self):
        """Show/hide timestep controls based on selected mode"""
        individual_mode = self.show_individual_timestep.isChecked()
        
        self.timestep_label.setVisible(individual_mode)
        self.timestep_spinner.setVisible(individual_mode)
        self.prev_timestep_btn.setVisible(individual_mode)
        self.next_timestep_btn.setVisible(individual_mode)
    
    def update_all(self):
        """Combined function that refreshes debris list, updates visualization, and forces redraw"""
        self.status_label.setText("Updating all components...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()
        
        try:
            self.update_debris_list() 
            QApplication.processEvents()
            
            self.perform_redraw()
            
            sat_count = len(self.satman.get_satellites())
            
            tracked_debris_id_val = self.tracked_debris.currentData()
            tracking_info = ""
            if tracked_debris_id_val is not None and tracked_debris_id_val >= 0:
                tracking_info = f" - Tracking Debris ID {tracked_debris_id_val}"
            
            mode_info = " - All Timesteps" if self.show_all_timesteps.isChecked() else f" - Timestep {self.timestep_spinner.value()}"
                
            self.status_label.setText(f"All updated: {sat_count} satellites{tracking_info}{mode_info}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in update_all: {error_details}")
            self.status_label.setText(f"Error updating: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        QTimer.singleShot(3000, lambda: self.status_label.setText("Visualization ready"))
        
    def update_debris_list(self):
        """Update the tracked debris dropdown with available debris IDs"""
        current_id_val = self.tracked_debris.currentData()
        self.tracked_debris.clear()
        
        self.tracked_debris.addItem("Earth Center (0,0,0)", -1)
        
        if not self.debris_tab.df.empty and 'Assembly_ID' in self.debris_tab.df.columns:
            try:
                debris_ids_series = pd.to_numeric(self.debris_tab.df["Assembly_ID"], errors='coerce').dropna().unique()
                debris_ids = sorted([int(did) for did in debris_ids_series])
                for debris_id_val in debris_ids:
                    self.tracked_debris.addItem(f"Debris ID {debris_id_val}", int(debris_id_val))
                
                if current_id_val is not None:
                    index = self.tracked_debris.findData(current_id_val)
                    if index >= 0:
                        self.tracked_debris.setCurrentIndex(index)
                    elif len(debris_ids) > 0:
                         self.tracked_debris.setCurrentIndex(1)
            except Exception as e:
                print(f"Error updating debris list for tracking: {e}")
        
        self.update_timestep_spinner_range()
    
    def check_and_redraw(self):
        """Auto-refresh if tab is visible and no redraw is pending"""
        if self.is_visible and not self.redraw_pending:
            sats = self.satman.get_satellites()
            if sats and (len(self.orbits) == 0 or len(self.dots) == 0):
                print("Auto-refresh: Detected missing satellite visualization")
                self.request_redraw()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.is_visible = True
        self.auto_refresh_timer.start()
        QTimer.singleShot(100, self.request_redraw)
        
    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.is_visible = False
        self.auto_refresh_timer.stop()
        
    def force_redraw(self):
        """Immediately perform a redraw without delay"""
        self.update_all()
        
    def clear_all_items(self):
        """Clear all items from the scene except Earth"""
        try:
            for itm_list in [self.orbits, self.dots, self.debris_dots]:
                for itm in itm_list:
                    try:
                        self.view.removeItem(itm)
                    except Exception:
                        pass
                itm_list.clear()
            if self.earth is not None:
                items_to_remove = [item for item in self.view.items if item is not self.earth]
                for item in items_to_remove:
                    try:
                        self.view.removeItem(item)
                    except:
                        pass
            else:
                 self.view.clear()
            
            if self.earth and self.earth not in self.view.items:
                 self.view.addItem(self.earth)
            self.orbits = []
            self.dots = []
            self.debris_dots = []
            
            self.view.update()
            
        except Exception as e:
            print(f"Error during clearing scene items: {e}")
        
    def request_redraw(self):
        """Queue a redraw operation instead of performing it immediately"""
        if not self.redraw_pending and self.is_visible:
            self.redraw_pending = True
            self.status_label.setText("Updating visualization...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            QApplication.processEvents()
            self.redraw_timer.start(50)
            
    def perform_redraw(self):
        """Actually perform the redraw operation"""
        if not self.is_visible:
            self.redraw_pending = False
            return
            
        try:
            self.redraw()
            
            tracked_debris_id_val = self.tracked_debris.currentData()
            tracking_info = ""
            if tracked_debris_id_val is not None and tracked_debris_id_val >= 0:
                tracking_info = f" - Tracking Debris ID {tracked_debris_id_val}"
                
            self.status_label.setText(f"Visualization updated successfully{tracking_info}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            self.status_label.setText(f"Error updating: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error in perform_redraw: {e}")
        finally:
            self.redraw_pending = False
            QTimer.singleShot(3000, lambda: self.status_label.setText("Visualization ready"))

    def redraw(self):
        """
        Redraw with proper coordinate transformations
        All satellite positions converted from ECI to ECEF for visualization
        """
        self.clear_all_items()
        
        while self.legend_content_layout.count() > 0:
            item = self.legend_content_layout.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        
        sats = self.satman.get_satellites()
        if not sats:
            return
        
        snap = Time(self.snapshot.dateTime().toPyDateTime(), scale='utc')
        
        self.lla_data.clear()
        
        df = self.debris_tab.df.copy()
        timestep_info_text = "No debris data"
        
        if not df.empty and 'Time' in df.columns:
            if self.show_all_timesteps.isChecked():
                timestep_info_text = "ALL TIMESTEPS"
            else:
                current_timestep = self.timestep_spinner.value()
                try:
                    timesteps = sorted(pd.to_numeric(df['Time'], errors='coerce').dropna().unique())
                except Exception as e:
                    print(f"Error processing timesteps: {e}")
                    timesteps = []
                if timesteps:
                    if current_timestep in timesteps:
                        selected_timestep = current_timestep
                    else:
                        selected_timestep = min(timesteps, key=lambda x: abs(x - current_timestep))
                        if abs(selected_timestep - current_timestep) > 1e-5:
                            self.timestep_spinner.blockSignals(True)
                            self.timestep_spinner.setValue(selected_timestep)
                            self.timestep_spinner.blockSignals(False)
                    
                    df = df[np.isclose(df['Time'], selected_timestep, atol=1e-5)]
                    timestep_info_text = f"Timestep {selected_timestep}"
                else:
                    timestep_info_text = "No valid timesteps in data"
                    df = pd.DataFrame()
        
        tracked_debris_id_val = self.tracked_debris.currentData()
        tracked_debris_position_ecef_km = None
        
        if tracked_debris_id_val is not None and tracked_debris_id_val >= 0 and not df.empty:
            df['Assembly_ID'] = pd.to_numeric(df['Assembly_ID'], errors='coerce')
            tracked_rows = df[df["Assembly_ID"] == tracked_debris_id_val]
            if not tracked_rows.empty:
                track_row = tracked_rows.iloc[0]
                try:
                    lat_deg = float(track_row["Latitude"])
                    lon_deg = float(track_row["Longitude"])
                    alt_m = float(track_row["Altitude"])  # Already in meters
                    
                    # Convert LLA to ECEF using WGS84
                    tracked_debris_position_ecef_m = lla_to_ecef_wgs84(lat_deg, lon_deg, alt_m)
                    tracked_debris_position_ecef_km = tracked_debris_position_ecef_m / 1000.0  # Convert to km for visualization
                except Exception as e:
                    print(f"Error converting tracked debris LLA to ECEF: {e}")
                    tracked_debris_position_ecef_km = None
        
        if tracked_debris_position_ecef_km is not None:
            tracked_entry = QWidget()
            tracked_layout_v = QVBoxLayout(tracked_entry)
            tracked_layout_v.setContentsMargins(0, 10, 0, 10)
            
            tracked_label = QLabel(f"TRACKING DEBRIS ID {tracked_debris_id_val}")
            tracked_label.setStyleSheet("font-size: 14px; color: yellow;")
            tracked_label.setAlignment(Qt.AlignCenter)
            tracked_layout_v.addWidget(tracked_label)
            
            pos_label_text = f"Pos (km): ({tracked_debris_position_ecef_km[0]:.1f}, {tracked_debris_position_ecef_km[1]:.1f}, {tracked_debris_position_ecef_km[2]:.1f})"
            pos_label = QLabel(pos_label_text)
            pos_label.setStyleSheet("color: yellow;")
            pos_label.setAlignment(Qt.AlignCenter)
            tracked_layout_v.addWidget(pos_label)
            
            mode_label = QLabel(f"Mode: {timestep_info_text}")
            mode_label.setStyleSheet("color: yellow;")
            mode_label.setAlignment(Qt.AlignCenter)
            tracked_layout_v.addWidget(mode_label)
            
            line_sep = QFrame()
            line_sep.setFrameShape(QFrame.HLine)
            line_sep.setFrameShadow(QFrame.Sunken)
            line_sep.setStyleSheet("background-color: #444;")
            tracked_layout_v.addWidget(line_sep)
            
            self.legend_content_layout.addWidget(tracked_entry)
        
        # Draw satellites with proper coordinate conversion
        for idx, (name, sat) in enumerate(sats):
            color = _PALETTE[idx % len(_PALETTE)]
            
            try:
                # Generate orbit points in ECI, then convert to ECEF for visualization
                orbit_points_eci = []
                epoch_time = sat.epoch
                for f in np.linspace(0, 1, 100):
                    sat_at_f = sat.propagate(sat.period * f)
                    r_eci_m = sat_at_f.rv()[0].to(u.m).value
                    # Convert each orbit point from ECI to ECEF
                    time_at_f = epoch_time + sat.period * f
                    r_ecef_m = eci_to_ecef(r_eci_m, time_at_f)
                    orbit_points_eci.append(r_ecef_m / 1000.0)  # Convert to km for visualization
                
                pts = np.array(orbit_points_eci)
                orb = gl.GLLinePlotItem(pos=pts, width=1, color=color)
                self.view.addItem(orb)
                self.orbits.append(orb)
            except Exception as e:
                print(f"Error plotting orbit for {name}: {e}")
                
            try:
                # Current satellite position with proper coordinate conversion
                dt = (snap - sat.epoch)
                sat_now = sat.propagate(dt)
                r_eci_m = sat_now.rv()[0].to(u.m).value  # ECI position in meters
                
                # Convert ECI to ECEF for visualization
                r_ecef_m = eci_to_ecef(r_eci_m, snap)
                r_ecef_km = r_ecef_m / 1000.0  # Convert to km for visualization
                
                sphere_mesh = gl.MeshData.sphere(rows=10, cols=10, radius=150)
                dot = gl.GLMeshItem(meshdata=sphere_mesh, smooth=True, color=color, shader='shaded')
                dot.translate(r_ecef_km[0], r_ecef_km[1], r_ecef_km[2])
                self.view.addItem(dot)
                self.dots.append(dot)
                
                r_, g_, b_, a_ = [int(c*255) for c in color]
                entry_widget = QWidget()
                entry_layout_h = QHBoxLayout(entry_widget)
                entry_layout_h.setContentsMargins(0, 2, 0, 2)
                
                sw = QLabel()
                sw.setFixedSize(20, 20)
                sw.setStyleSheet(f"background: rgba({r_},{g_},{b_},{a_}); border:2px solid white;")
                
                pos_text = f"({r_ecef_km[0]:.0f}, {r_ecef_km[1]:.0f}, {r_ecef_km[2]:.0f}) km"
                label = QLabel(f"{name}\n{pos_text}")
                label.setStyleSheet("color: white;")
                
                entry_layout_h.addWidget(sw)
                entry_layout_h.addWidget(label)
                entry_layout_h.addStretch()
                self.legend_content_layout.addWidget(entry_widget)
                
                # Store LLA data using proper coordinate conversion
                lat_deg, lon_deg, alt_m = ecef_to_lla_wgs84(r_ecef_m)
                
                # Generate orbital LLA data
                T = sat.period
                ts = [sat.epoch + T*f for f in np.linspace(0, 1, 30)]
                lats, lons, alts = [], [], []
                for t in ts:
                    dt_t = t - sat.epoch
                    sat_at_t = sat.propagate(dt_t)
                    r_eci_t_m = sat_at_t.rv()[0].to(u.m).value
                    r_ecef_t_m = eci_to_ecef(r_eci_t_m, t)
                    lat_t, lon_t, alt_t = ecef_to_lla_wgs84(r_ecef_t_m)
                    lats.append(lat_t)
                    lons.append(lon_t)
                    alts.append(alt_t / 1000.0)  # Convert to km for display
                
                self.lla_data[name] = {'times':ts, 'lat':np.array(lats), 'lon':np.array(lons), 'alt':np.array(alts)}
                
            except Exception as e:
                 print(f"Error visualizing satellite {name}: {e}")

        # Draw debris (already in ECEF via LLA conversion)
        if self.toggle_debris.isChecked() and not df.empty and 'Assembly_ID' in df.columns:
            df['Assembly_ID'] = pd.to_numeric(df['Assembly_ID'], errors='coerce').dropna()
            df['Assembly_ID'] = df['Assembly_ID'].astype(int)
            
            ids = df['Assembly_ID'].unique().tolist()
            
            if ids:
                debris_header_widget = QWidget()
                debris_header_layout_v = QVBoxLayout(debris_header_widget)
                debris_header_layout_v.setContentsMargins(0, 10, 0, 5)
                
                line_sep_debris = QFrame()
                line_sep_debris.setFrameShape(QFrame.HLine)
                line_sep_debris.setFrameShadow(QFrame.Sunken)
                line_sep_debris.setStyleSheet("background-color: #444;")
                debris_header_layout_v.addWidget(line_sep_debris)
                
                hdr = QLabel(f"DEBRIS ({timestep_info_text})")
                hdr.setStyleSheet("font-size: 12px; color: white;")
                hdr.setAlignment(Qt.AlignCenter)
                debris_header_layout_v.addWidget(hdr)
                self.legend_content_layout.addWidget(debris_header_widget)

            self.debris_colors.clear()
            for i, aid in enumerate(ids):
                if aid == tracked_debris_id_val:
                    self.debris_colors[aid] = (1.0, 1.0, 0.0, 1.0)
                else:
                    self.debris_colors[aid] = _PALETTE[(i + len(sats)) % len(_PALETTE)]

            for j, aid in enumerate(ids):
                col = self.debris_colors.get(aid, _PALETTE[0])
                r_, g_, b_, a_ = [int(c*255) for c in col]
                
                entry_widget_debris = QWidget()
                entry_layout_debris_h = QHBoxLayout(entry_widget_debris)
                entry_layout_debris_h.setContentsMargins(0, 2, 0, 2)
                
                sw_debris = QLabel()
                sw_debris.setFixedSize(20, 20)
                sw_debris.setStyleSheet(f"background: rgba({r_},{g_},{b_},{a_}); border:2px solid white;")
                
                is_tracked = aid == tracked_debris_id_val
                label_text = f"Debris ID {aid}{' (TRACKED)' if is_tracked else ''}"
                label_debris = QLabel(label_text)
                label_debris.setStyleSheet(f"color: {'yellow' if is_tracked else 'white'};")
                
                entry_layout_debris_h.addWidget(sw_debris)
                entry_layout_debris_h.addWidget(label_debris)
                entry_layout_debris_h.addStretch()
                self.legend_content_layout.addWidget(entry_widget_debris)

            for _, r_debris in df.iterrows():
                try:
                    lat_deg = float(r_debris['Latitude'])
                    lon_deg = float(r_debris['Longitude'])
                    alt_m = float(r_debris['Altitude'])  # Already in meters
                    
                    # Convert LLA to ECEF using WGS84
                    pos_ecef_m = lla_to_ecef_wgs84(lat_deg, lon_deg, alt_m)
                    pos_km = pos_ecef_m / 1000.0  # Convert to km for visualization
                    
                    aid = int(r_debris['Assembly_ID'])
                    col_d = self.debris_colors.get(aid, _PALETTE[0])
                    
                    radius_km = 70 if aid == tracked_debris_id_val else 30
                    
                    sphere_mesh_debris = gl.MeshData.sphere(rows=6, cols=6, radius=radius_km)
                    dpt = gl.GLMeshItem(meshdata=sphere_mesh_debris, smooth=True, color=col_d, shader='shaded')
                    dpt.translate(pos_km[0], pos_km[1], pos_km[2])
                    self.view.addItem(dpt)
                    self.debris_dots.append(dpt)
                except Exception as e:
                    print(f"Error rendering debris ID {r_debris.get('Assembly_ID', 'Unknown')}: {e}")
        
        if tracked_debris_position_ecef_km is not None:
            x, y, z = tracked_debris_position_ecef_km
            self.view.opts['center'] = pg.Vector(x, y, z)
        else:
            self.view.opts['center'] = pg.Vector(0,0,0) 
        QApplication.processEvents()
        self.view.update()
    
    def previous_timestep(self):
        """Go to the previous timestep in the data"""
        if self.debris_tab.df.empty or 'Time' not in self.debris_tab.df.columns:
            return
            
        try:
            timesteps = sorted(pd.to_numeric(self.debris_tab.df['Time'], errors='coerce').dropna().unique())
            if not timesteps: return
            
            current_val = self.timestep_spinner.value()
            prev_steps = [t for t in timesteps if t < current_val - 1e-5]
            if prev_steps:
                self.timestep_spinner.setValue(max(prev_steps))
            else:
                self.timestep_spinner.setValue(max(timesteps))
        except Exception as e:
            print(f"Error in previous_timestep: {e}")

    def next_timestep(self):
        """Go to the next timestep in the data"""
        if self.debris_tab.df.empty or 'Time' not in self.debris_tab.df.columns:
            return
        
        try:
            timesteps = sorted(pd.to_numeric(self.debris_tab.df['Time'], errors='coerce').dropna().unique())
            if not timesteps: return
            current_val = self.timestep_spinner.value()
            next_steps = [t for t in timesteps if t > current_val + 1e-5]
            if next_steps:
                self.timestep_spinner.setValue(min(next_steps))
            else:
                self.timestep_spinner.setValue(min(timesteps))
        except Exception as e:
            print(f"Error in next_timestep: {e}")

# ----------------------------------------------------------------------
# Data Tab
# ----------------------------------------------------------------------

class DataTab(QWidget):
    """
    LLA data tab with proper coordinate transformations and CSV export
    Fixed to use consistent 0.5-second timestep intervals for both GUI and CSV
    """
    def __init__(self, viz: VizTab, parent=None):
        super().__init__(parent)
        self.viz = viz

        lay = QVBoxLayout(self)

        # Controls and buttons in one row
        controls_layout = QHBoxLayout()
        
        # Points control
        controls_layout.addWidget(QLabel("Points per satellite:"))
        self.points_spinner = QSpinBox()
        self.points_spinner.setRange(5, 200)  # Increased max range
        self.points_spinner.setValue(20)
        self.points_spinner.setToolTip("Number of points to generate per satellite (0.5s intervals)")
        controls_layout.addWidget(self.points_spinner)
        
        # Time step control
        controls_layout.addWidget(QLabel("Time step:"))
        self.time_step_spinner = QDoubleSpinBox()
        self.time_step_spinner.setRange(0.1, 10.0)
        self.time_step_spinner.setValue(0.5)
        self.time_step_spinner.setSuffix(" sec")
        self.time_step_spinner.setDecimals(1)
        self.time_step_spinner.setSingleStep(0.1)
        self.time_step_spinner.setToolTip("Time interval between data points")
        controls_layout.addWidget(self.time_step_spinner)
        
        # Buttons - simple style like emissions tab
        self.regenerate_btn = QPushButton("Regenerate Data")
        self.regenerate_btn.clicked.connect(self.regenerate_with_points)
        controls_layout.addWidget(self.regenerate_btn)
        
        self.export_btn = QPushButton("Export LLA to CSV…")
        self.export_btn.clicked.connect(self.export_lla_data)
        self.export_btn.setEnabled(False)  # Disabled until data is available
        controls_layout.addWidget(self.export_btn)
        
        controls_layout.addStretch()
        lay.addLayout(controls_layout)

        # Status label
        self.status_label = QLabel("LLA data ready — auto-updates with time changes")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-size: 10px;")
        lay.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.table)

        # Auto-update when time changes
        self.viz.snapshot.dateTimeChanged.connect(self.auto_populate)
        self.viz.satman.satsChanged.connect(self.auto_populate)

    def generate_fixed_timestep_lla_data(self, num_points=None, time_step_sec=None):
        """Generate LLA data with fixed timestep intervals (consistent for GUI and CSV)"""
        if num_points is None:
            # For GUI display, use default 5 points
            num_points = 5
            time_step_sec = 0.5
        else:
            # For CSV export, use user-specified values
            if time_step_sec is None:
                time_step_sec = self.time_step_spinner.value()
        
        self.viz.lla_data.clear()
        sats = self.viz.satman.get_satellites()
        if not sats:
            return

        snap_dt = self.viz.snapshot.dateTime().toPyDateTime().replace(microsecond=0)
        snap = Time(snap_dt, scale="utc")

        print(f"Generating {num_points} LLA points per satellite with {time_step_sec}s intervals")

        for name, sat in sats:
            lats, lons, alts, ts = [], [], [], []

            # Generate points at fixed time intervals
            for i in range(num_points):
                time_offset = i * time_step_sec
                t = snap + time_offset * u.s
                dt_from_epoch = (t - sat.epoch).to_value(u.s)
                sat_now = sat.propagate(dt_from_epoch * u.s)
                
                # Get satellite position in ECI (from poliastro)
                r_eci_m = sat_now.rv()[0].to(u.m).value  # ECI position in meters
                
                # Convert ECI to ECEF for proper LLA calculation
                r_ecef_m = eci_to_ecef(r_eci_m, t)
                
                # Convert ECEF to LLA using WGS84
                lat_deg, lon_deg, alt_m = ecef_to_lla_wgs84(r_ecef_m)

                lats.append(lat_deg)
                lons.append(lon_deg)
                alts.append(alt_m / 1000.0)  # Convert to km for display
                ts.append(t)

            self.viz.lla_data[name] = {
                "times": ts,
                "lat": np.array(lats),
                "lon": np.array(lons),
                "alt": np.array(alts)
            }

            print(f"Generated {len(ts)} points for {name} (timesteps: 0 to {(num_points-1)*time_step_sec:.1f}s)")

    def regenerate_with_points(self):
        """Regenerate UI data with user-specified parameters"""
        try:
            self.status_label.setText("Regenerating UI display data...")
            self.status_label.setStyleSheet("color: orange; font-size: 10px;")
            QApplication.processEvents()
            
            # Generate UI data with user-specified points and time step
            num_points = self.points_spinner.value()
            time_step = self.time_step_spinner.value()
            self.generate_fixed_timestep_lla_data(num_points, time_step)
            self.populate()
            
        except Exception as e:
            print(f"Error in regenerate_with_points: {e}")
            self.status_label.setText(f"Error regenerating data: {e}")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")

    def export_lla_data(self):
        """Export LLA data to CSV file with consistent timestep intervals"""
        sats = self.viz.satman.get_satellites()
        if not sats:
            QMessageBox.warning(self, "No Satellites", "No satellites available to export. Add satellites first.")
            return

        # Get export parameters
        num_points = self.points_spinner.value()
        time_step = self.time_step_spinner.value()

        # Get save location
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        default_filename = f"satellite_lla_data_{num_points}pts_{time_step}s_step_{timestamp}.csv"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save LLA Data", default_filename, "CSV Files (*.csv)"
        )
        
        if not filepath:
            return

        try:
            # Generate fresh LLA data specifically for export with consistent timesteps
            snap_dt = self.viz.snapshot.dateTime().toPyDateTime().replace(microsecond=0)
            snap = Time(snap_dt, scale="utc")
            
            export_data = []
            
            print(f"Generating {num_points} LLA points per satellite for CSV export with {time_step}s intervals")
            
            for name, sat in sats:
                # Generate points at fixed time intervals (same as GUI)
                for i in range(num_points):
                    time_offset = i * time_step
                    t = snap + time_offset * u.s
                    dt_from_epoch = (t - sat.epoch).to_value(u.s)
                    sat_now = sat.propagate(dt_from_epoch * u.s)
                    
                    # Get satellite position in ECI (from poliastro)
                    r_eci_m = sat_now.rv()[0].to(u.m).value  # ECI position in meters
                    
                    # Convert ECI to ECEF for proper LLA calculation
                    r_ecef_m = eci_to_ecef(r_eci_m, t)
                    
                    # Convert ECEF to LLA using WGS84
                    lat_deg, lon_deg, alt_m = ecef_to_lla_wgs84(r_ecef_m)
                    alt_km = alt_m / 1000.0  # Convert to km

                    export_data.append({
                        'Satellite': name,
                        'Time_Offset_Sec': time_offset,
                        'UTC_Time': t.utc.isot,
                        'Latitude_deg': f"{lat_deg:.6f}",
                        'Longitude_deg': f"{lon_deg:.6f}",
                        'Altitude_km': f"{alt_km:.3f}",
                        'Analysis_Time_UTC': self.viz.snapshot.dateTime().toString("yyyy-MM-dd hh:mm:ss")
                    })

            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(filepath, index=False)
            
            # Create metadata file
            metadata_path = filepath.replace('.csv', '_metadata.json')
            metadata = {
                'export_timestamp': timestamp,
                'analysis_time_utc': self.viz.snapshot.dateTime().toString("yyyy-MM-dd hh:mm:ss"),
                'coordinate_system': 'WGS84',
                'total_satellites': len(sats),
                'points_per_satellite': num_points,
                'time_step_seconds': time_step,
                'total_data_points': len(export_data),
                'time_span_seconds': (num_points - 1) * time_step,
                'altitude_units': 'kilometers',
                'coordinate_units': 'degrees',
                'time_format': 'ISO 8601 UTC',
                'timestep_description': f'Fixed {time_step}s intervals: 0, {time_step}, {2*time_step}, ..., {(num_points-1)*time_step}s',
                'data_description': f'Satellite positions at {num_points} timesteps with {time_step}s intervals'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Show summary of timesteps
            timestep_example = [i * time_step for i in range(min(10, num_points))]
            timestep_text = ", ".join(f"{t:.1f}" for t in timestep_example)
            if num_points > 10:
                timestep_text += f", ..., {(num_points-1)*time_step:.1f}"

            QMessageBox.information(
                self, "Export Complete", 
                f"LLA data exported successfully!\n\n"
                f"Data file: {filepath}\n"
                f"Metadata: {metadata_path}\n\n"
                f"Exported {len(export_data)} data points:\n"
                f"• {len(sats)} satellites\n"
                f"• {num_points} points per satellite\n"
                f"• {time_step}s time intervals\n"
                f"• Timesteps (sec): {timestep_text}"
            )
            
            print(f"LLA data exported: {len(export_data)} points ({num_points} per satellite) to {filepath}")
            print(f"Timestep intervals: {time_step}s (0 to {(num_points-1)*time_step:.1f}s)")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
            print(f"Error exporting LLA data: {e}")

    def showEvent(self, event):
        super().showEvent(event)
        self.auto_populate()

    def auto_populate(self):
        """Auto-populate when time changes - uses default 5 points for UI display"""
        self.status_label.setText("Updating LLA data…")
        self.status_label.setStyleSheet("color: orange; font-size: 10px;")
        QApplication.processEvents()

        try:
            # Always use fixed 5 points at 0.5s intervals for UI display (consistent with viz tab)
            self.generate_fixed_timestep_lla_data(num_points=5, time_step_sec=0.5)
            self.populate()
        except Exception as e:
            print(f"Error in auto_populate: {e}")
            self.status_label.setText(f"Error updating LLA data: {e}")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")

    def populate(self):
        """Populate table with LLA data"""
        d = self.viz.lla_data
        if not d:
            self.table.clear()
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("No satellite data available"))
            self.status_label.setText("No LLA data – add satellites first")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            self.export_btn.setEnabled(False)
            return

        total_rows = sum(len(block["lat"]) for block in d.values())
        self.table.clear()
        self.table.setRowCount(total_rows)
        self.table.setColumnCount(6)  # Added Time Offset column
        self.table.setHorizontalHeaderLabels(
            ["Satellite", "Time Offset (s)", "UTC Time", "Lat [°]", "Lon [°]", "Alt [km]"]
        )

        r = 0
        for sat_name, block in d.items():
            for i, (ts, lat, lon, alt) in enumerate(zip(block["times"], block["lat"], block["lon"], block["alt"])):
                # Calculate time offset from first timestep
                time_offset = i * 0.5 if len(block["times"]) == 5 else i * self.time_step_spinner.value()
                
                self.table.setItem(r, 0, QTableWidgetItem(sat_name))
                self.table.setItem(r, 1, QTableWidgetItem(f"{time_offset:.1f}"))
                self.table.setItem(r, 2, QTableWidgetItem(ts.utc.isot))
                self.table.setItem(r, 3, QTableWidgetItem(f"{lat:.6f}"))
                self.table.setItem(r, 4, QTableWidgetItem(f"{lon:.6f}"))
                self.table.setItem(r, 5, QTableWidgetItem(f"{alt:.3f}"))
                r += 1

        self.table.resizeColumnsToContents()

        current_time = self.viz.snapshot.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        ui_points = total_rows // len(d) if len(d) > 0 else 0
        csv_points = self.points_spinner.value()
        csv_time_step = self.time_step_spinner.value()
        
        self.status_label.setText(f"LLA data ready — UI: {ui_points} pts @ 0.5s intervals, CSV: {csv_points} pts @ {csv_time_step}s intervals")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")
        self.export_btn.setEnabled(True)

# ----------------------------------------------------------------------
# Emissions Tab
# ----------------------------------------------------------------------
class EmissionsTab(QWidget):
    """
    Tab for loading and displaying emissions data from CSV or Excel files.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        btn = QPushButton("Load Emissions…"); btn.clicked.connect(self.load)
        lay.addWidget(btn)
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self.table)
        self.df = pd.DataFrame()

    def load(self):
        """Load emissions data from a CSV or Excel file"""
        p,_ = QFileDialog.getOpenFileName(self,"Open Emissions","","*.csv *.xlsx")
        if not p: return

        try:
            df = pd.read_csv(p) if p.lower().endswith('.csv') else pd.read_excel(p)
            if 'Iter' in df.columns:
                df.rename(columns={'Iter':'Iteration'}, inplace=True)
            
            emission_keys = [
                "OI_emissions_atomic",
                "AlI_1_emissions_atomic",
                "AlI_2_emissions_atomic"
            ]
            
            req_cols = ["Iteration", "Assembly_ID"]
            
            found_emission_keys = [key for key in emission_keys if key in df.columns]
            
            missing_req = [c for c in req_cols if c not in df.columns]
            if missing_req:
                QMessageBox.warning(self, "Missing Required Columns", 
                                    f"Missing essential columns for merging: {', '.join(missing_req)}")
                return
            
            if not found_emission_keys:
                QMessageBox.warning(self, "Missing Emission Columns", 
                                    f"No recognized emission columns found. Expected at least one of: {', '.join(emission_keys)}")
            
            columns_to_use = req_cols + found_emission_keys
            columns_to_use = [col for col in columns_to_use if col in df.columns]
            self.df = df[columns_to_use].copy()
            
            self.table.clear()
            self.table.setRowCount(len(self.df))
            self.table.setColumnCount(len(columns_to_use))
            self.table.setHorizontalHeaderLabels(columns_to_use)
            
            for i, (_, row_data) in enumerate(self.df.iterrows()):
                for j, c in enumerate(columns_to_use):
                    self.table.setItem(i, j, QTableWidgetItem(str(row_data[c])))
                    
            self.table.resizeColumnsToContents()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Emissions", f"Could not load or process emissions file: {str(e)}")
            self.df = pd.DataFrame()
            self.table.clear()

# ----------------------------------------------------------------------
# Photon Conversion Tab with Coordinate System
# ----------------------------------------------------------------------
class PhotonConversionTab(QWidget):
    """
    Photon conversion with proper coordinate transformations, WGS84 calculations, background illumination,
    multi-satellite sticky handoff capability, separate QE/filter values for OI and Al emissions,
    and configurable output formats (numpy/FITS)
    """
    def __init__(self, sat_tab, debris_tab, em_tab, viz_tab, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sat_tab, self.debris_tab, self.em_tab, self.viz_tab = sat_tab, debris_tab, em_tab, viz_tab
        self.debris_tab.debrisDataChanged.connect(self.handle_debris_data_change_for_photon)
        
        # Connect to viz tab time changes for automatic timestep updates
        self.viz_tab.timeChanged.connect(self._on_viz_time_changed)
        
        main = QVBoxLayout(self)
        top = QHBoxLayout()
        pg.setConfigOption('imageAxisOrder', 'row-major')
        pg.setConfigOption('useNumba', False)
        
        # Create dual image view layout
        image_container = QHBoxLayout()
        
        # Left image view
        left_image_widget = QWidget()
        left_image_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_image_layout = QVBoxLayout(left_image_widget)
        left_image_layout.setContentsMargins(5, 5, 5, 5)
        
        left_label = QLabel("Left View")
        left_label.setAlignment(Qt.AlignCenter)
        left_image_layout.addWidget(left_label)
        
        self.display_selector_left = QComboBox()
        self.display_selector_left.addItem("OI (777.4 nm)", "OI_emissions_atomic")
        self.display_selector_left.addItem("AlI_1 (e.g. 394.4nm)", "AlI_1_emissions_atomic")
        self.display_selector_left.addItem("AlI_2 (e.g. 396.1nm)", "AlI_2_emissions_atomic")
        self.display_selector_left.addItem("Combined AlI", "combined_al")
        self.display_selector_left.setEnabled(False)
        left_image_layout.addWidget(self.display_selector_left)
        
        self.imv_left = pg.ImageView()
        self.imv_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imv_left.ui.roiBtn.hide()
        self.imv_left.ui.menuBtn.hide()
        left_image_layout.addWidget(self.imv_left)
        
        # Right image view  
        right_image_widget = QWidget()
        right_image_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_image_layout = QVBoxLayout(right_image_widget)
        right_image_layout.setContentsMargins(5, 5, 5, 5)
        
        right_label = QLabel("Right View")
        right_label.setAlignment(Qt.AlignCenter)
        right_image_layout.addWidget(right_label)
        
        self.display_selector_right = QComboBox()
        self.display_selector_right.addItem("OI (777.4 nm)", "OI_emissions_atomic")
        self.display_selector_right.addItem("AlI_1 (e.g. 394.4nm)", "AlI_1_emissions_atomic")
        self.display_selector_right.addItem("AlI_2 (e.g. 396.1nm)", "AlI_2_emissions_atomic") 
        self.display_selector_right.addItem("Combined AlI", "combined_al")
        self.display_selector_right.setCurrentIndex(3)  # Default to Combined AlI
        self.display_selector_right.setEnabled(False)
        right_image_layout.addWidget(self.display_selector_right)
        
        self.imv_right = pg.ImageView()
        self.imv_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.imv_right.ui.roiBtn.hide()
        self.imv_right.ui.menuBtn.hide()
        right_image_layout.addWidget(self.imv_right)
        
        image_container.addWidget(left_image_widget)
        image_container.addWidget(right_image_widget)
        
        top.addLayout(image_container, 3)
        
        # Controls form
        form_group = QGroupBox("Controls")
        form_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        form_group.setMinimumWidth(350)
        form_group.setMaximumWidth(450)
        form = QFormLayout(form_group)
        
        # Satellite selection
        self.sat_selector = QComboBox()
        self.update_satellite_list()
        form.addRow("Select Satellite:", self.sat_selector)
        
        # Debris tracking selection
        self.tracked_debris = QComboBox()
        form.addRow("Track Debris ID:", self.tracked_debris)
        
        # Refresh buttons
        refresh_layout = QHBoxLayout()
        self.refresh_sats_btn = QPushButton("Refresh Satellites")
        self.refresh_debris_btn = QPushButton("Refresh Debris Lists")
        self.refresh_sats_btn.clicked.connect(self.update_satellite_list)
        self.refresh_debris_btn.clicked.connect(self.update_debris_list)
        refresh_layout.addWidget(self.refresh_sats_btn)
        refresh_layout.addWidget(self.refresh_debris_btn)
        form.addRow(refresh_layout)
        
        self.sat_tab.satsChanged.connect(self.update_satellite_list)
        
        # Separator
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        form.addRow(line)
        
        # Physical parameters
        self.ap = QDoubleSpinBox(); self.ap.setDecimals(3); self.ap.setValue(0.04); self.ap.setSuffix(" m"); self.ap.setRange(0.001, 100)  
        self.fl = QDoubleSpinBox(); self.fl.setDecimals(3); self.fl.setValue(0.05); self.fl.setSuffix(" m"); self.fl.setRange(0.001, 100)  
        self.pp = QDoubleSpinBox(); self.pp.setDecimals(2); self.pp.setValue(15.04); self.pp.setSuffix(" µm"); self.pp.setRange(0.1, 100)
        
        self.sensor_width = QSpinBox(); self.sensor_width.setRange(100, 20000); self.sensor_width.setValue(1596); self.sensor_width.setSuffix(" px")
        self.sensor_height = QSpinBox(); self.sensor_height.setRange(100, 20000); self.sensor_height.setValue(2392); self.sensor_height.setSuffix(" px")
        
        self.te = QDoubleSpinBox(); self.te.setDecimals(3); self.te.setValue(1.0); self.te.setSuffix(" s"); self.te.setRange(0.001, 3600)  
        
        # OI Parameters (grouped together)
        self.qe_oi = QDoubleSpinBox(); self.qe_oi.setDecimals(3); self.qe_oi.setSingleStep(0.01); self.qe_oi.setValue(0.3); self.qe_oi.setRange(0.01, 1.0)  
        self.ft_oi = QDoubleSpinBox(); self.ft_oi.setDecimals(3); self.ft_oi.setSingleStep(0.01); self.ft_oi.setValue(0.78); self.ft_oi.setRange(0.01, 1.0)  
        self.ft_oi.setToolTip("Filter transmission efficiency for OI emissions")
        self.lm_oi = QDoubleSpinBox(); self.lm_oi.setDecimals(1); self.lm_oi.setRange(100, 2000); self.lm_oi.setValue(777.3); self.lm_oi.setSuffix(" nm")
        self.lm_oi.setSingleStep(1.0); self.lm_oi.setToolTip("OI emission wavelength")
        
        # Al Parameters (grouped together)
        self.qe_al = QDoubleSpinBox(); self.qe_al.setDecimals(3); self.qe_al.setSingleStep(0.01); self.qe_al.setValue(0.7); self.qe_al.setRange(0.01, 1.0)  
        self.ft_al = QDoubleSpinBox(); self.ft_al.setDecimals(3); self.ft_al.setSingleStep(0.01); self.ft_al.setValue(0.33); self.ft_al.setRange(0.01, 1.0)  
        self.ft_al.setToolTip("Filter transmission efficiency for Al emissions")
        self.lm_al = QDoubleSpinBox(); self.lm_al.setDecimals(1); self.lm_al.setRange(100, 2000); self.lm_al.setValue(395.0); self.lm_al.setSuffix(" nm")
        self.lm_al.setSingleStep(1.0); self.lm_al.setToolTip("Al emission average wavelength")

        # Add parameters to form in organized groups
        for lbl, w in [
            ("Aperture [m]:", self.ap), ("Focal Length [m]:", self.fl), ("Pixel Pitch [µm]:", self.pp),
            ("Sensor Width [px]:", self.sensor_width), ("Sensor Height [px]:", self.sensor_height),
            ("Exposure Time [s]:", self.te),
        ]:
            form.addRow(lbl, w)
        
        # OI Parameters section
        oi_separator = QFrame(); oi_separator.setFrameShape(QFrame.HLine); oi_separator.setFrameShadow(QFrame.Sunken)
        form.addRow(oi_separator)
        oi_label = QLabel("OI Emission Parameters")
        oi_label.setStyleSheet("font-weight: bold; color: #ff6600;")
        form.addRow(oi_label)
        
        form.addRow("OI Quantum Efficiency:", self.qe_oi)
        form.addRow("OI Filter Transmission T:", self.ft_oi)
        form.addRow("OI Wavelength λ [nm]:", self.lm_oi)
        
        # Al Parameters section
        al_separator = QFrame(); al_separator.setFrameShape(QFrame.HLine); al_separator.setFrameShadow(QFrame.Sunken)
        form.addRow(al_separator)
        al_label = QLabel("Al Emission Parameters")
        al_label.setStyleSheet("font-weight: bold; color: #6600ff;")
        form.addRow(al_label)
        
        form.addRow("Al Quantum Efficiency:", self.qe_al)
        form.addRow("Al Filter Transmission T:", self.ft_al)
        form.addRow("Al Avg Wavelength λ [nm]:", self.lm_al)
        
        # Background illumination controls
        bg_separator = QFrame(); bg_separator.setFrameShape(QFrame.HLine); bg_separator.setFrameShadow(QFrame.Sunken)
        form.addRow(bg_separator)
        
        self.enable_background = QCheckBox("Enable Background Illumination")
        self.enable_background.setChecked(False)
        form.addRow(self.enable_background)
        
        self.background_intensity = QDoubleSpinBox()
        self.background_intensity.setDecimals(9)
        self.background_intensity.setRange(0.0, 1e12)  # Allow up to 1 trillion
        self.background_intensity.setValue(1e-6)  # Default: 1 µW/sr
        self.background_intensity.setSuffix(" W/sr")
        self.background_intensity.setEnabled(False)
        self.background_intensity.setToolTip("Background radiance (watts per steradian)\nUses simple conversion without wavelength dependence\nTypical values:\n- Dark space: 1e-9 W/sr\n- Earth vicinity: 1e-6 W/sr\n- Bright Earth: 1e-3 W/sr")
        form.addRow("Background Radiance:", self.background_intensity)
        
        # Connect enable checkbox to enable/disable controls
        self.enable_background.toggled.connect(self.background_intensity.setEnabled)
        
        # Output format controls
        output_separator = QFrame(); output_separator.setFrameShape(QFrame.HLine); output_separator.setFrameShadow(QFrame.Sunken)
        form.addRow(output_separator)
        
        output_label = QLabel("Output Formats")
        output_label.setStyleSheet("font-weight: bold; color: #00aa44;")
        form.addRow(output_label)
        
        self.output_numpy = QCheckBox("Save NumPy arrays (.npy)")
        self.output_numpy.setChecked(True)
        self.output_numpy.setToolTip("Save image arrays as NumPy .npy files")
        form.addRow(self.output_numpy)
        
        self.output_fits = QCheckBox("Save FITS files (.fits)")
        self.output_fits.setChecked(False)
        self.output_fits.setToolTip("Save image arrays as FITS files with metadata headers")
        form.addRow(self.output_fits)
        
        # Save buttons in the form
        save_layout = QHBoxLayout()
        save_layout.setSpacing(2)
        self.save_left_btn = QPushButton("Save Left")
        self.save_right_btn = QPushButton("Save Right")
        self.save_both_btn = QPushButton("Save Both")
        
        save_button_style = "padding: 3px; margin: 1px; font-size: 9px;"
        self.save_left_btn.setStyleSheet(save_button_style)
        self.save_right_btn.setStyleSheet(save_button_style)
        self.save_both_btn.setStyleSheet(save_button_style)
        
        for btn in [self.save_left_btn, self.save_right_btn, self.save_both_btn]:
            btn.setMaximumHeight(22)
            btn.setMinimumHeight(22)
        
        self.save_left_btn.setEnabled(False); self.save_right_btn.setEnabled(False); self.save_both_btn.setEnabled(False)
        self.save_left_btn.clicked.connect(lambda: self.save_results("left"))
        self.save_right_btn.clicked.connect(lambda: self.save_results("right"))
        self.save_both_btn.clicked.connect(lambda: self.save_results("both"))
        save_layout.addWidget(self.save_left_btn); save_layout.addWidget(self.save_right_btn); save_layout.addWidget(self.save_both_btn)
        form.addRow("Save Results:", save_layout)
        
        top.addWidget(form_group, 1)
        main.addLayout(top)
        
        # Controls below images - single frame and batch processing side by side
        controls_below = QHBoxLayout()
        controls_below.setSpacing(10)
        controls_below.setContentsMargins(5, 5, 5, 5)
        
        # Single Frame Controls
        single_frame_group = QGroupBox("Single Frame")
        single_frame_group.setMaximumWidth(300)
        single_frame_layout = QFormLayout(single_frame_group)
        single_frame_layout.setVerticalSpacing(3)
        single_frame_layout.setContentsMargins(8, 8, 8, 8)
        
        # Timestep selection
        self.single_frame_timestep = QDoubleSpinBox()
        self.single_frame_timestep.setDecimals(1)
        self.single_frame_timestep.setSingleStep(0.5)
        self.single_frame_timestep.setSuffix(" s")
        self.single_frame_timestep.setMinimumWidth(100)
        self.single_frame_timestep.setToolTip("Auto-syncs with visualization time changes")
        single_frame_layout.addRow("Debris Timestep:", self.single_frame_timestep)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(3)
        
        self.generate_single_frame_btn = QPushButton("Compute")
        self.generate_single_frame_btn.setStyleSheet("background-color: #0088cc; font-weight: bold; padding: 4px; font-size: 11px;")
        self.generate_single_frame_btn.setMinimumHeight(28)
        self.generate_single_frame_btn.setMaximumHeight(28)
        self.generate_single_frame_btn.clicked.connect(self.generate_single_frame_dual)
        buttons_layout.addWidget(self.generate_single_frame_btn)
        
        self.check_visibility_btn = QPushButton("Visibility")
        self.check_visibility_btn.setStyleSheet("background-color: #00aa44; font-weight: bold; padding: 4px; font-size: 11px;")
        self.check_visibility_btn.setMinimumHeight(28)
        self.check_visibility_btn.setMaximumHeight(28)
        self.check_visibility_btn.clicked.connect(self.check_debris_visibility_timeline)
        buttons_layout.addWidget(self.check_visibility_btn)
        
        single_frame_layout.addRow(buttons_layout)
        
        # Batch Processing Section
        batch_group = QGroupBox("Batch Processing")
        batch_group.setMaximumWidth(350)
        batch_layout = QFormLayout(batch_group)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        batch_layout.setVerticalSpacing(3)
        
        # NEW: Satellite handoff checkbox
        self.enable_satellite_handoff = QCheckBox("Enable Multi-Satellite Handoff")
        self.enable_satellite_handoff.setChecked(False)
        self.enable_satellite_handoff.setToolTip("Automatically switch satellites when current one loses sight")
        batch_layout.addRow(self.enable_satellite_handoff)
        
        self.batch_process_btn = QPushButton("Track Main Body + Longest Survivor")
        self.batch_process_btn.setStyleSheet("background-color: #cc0088; font-weight: bold; padding: 4px; font-size: 10px;")
        self.batch_process_btn.setMinimumHeight(24)
        self.batch_process_btn.setMaximumHeight(24)
        self.batch_process_btn.clicked.connect(self.generate_batch_tracking)
        batch_layout.addRow(self.batch_process_btn)
        
        self.batch_status_label = QLabel("Ready for batch processing")
        self.batch_status_label.setStyleSheet("color: green; font-size: 9px;")
        self.batch_status_label.setAlignment(Qt.AlignLeft)
        batch_layout.addRow(self.batch_status_label)
        
        # Add both groups to horizontal layout
        controls_below.addWidget(single_frame_group)
        controls_below.addWidget(batch_group)
        controls_below.addStretch()  # Push groups to the left
        
        main.addLayout(controls_below)
        
        # Store results
        self.result_data_preview = { "OI_emissions_atomic": None, "AlI_1_emissions_atomic": None, "AlI_2_emissions_atomic": None, "combined_al": None }
        self.result_images_preview = { "OI_emissions_atomic": None, "AlI_1_emissions_atomic": None, "AlI_2_emissions_atomic": None, "combined_al": None }
        self.computed_preview = False
        self.last_generation_metadata = None
        self.last_background_electrons = 0.0
        
        # Connect display selectors
        self.display_selector_left.currentIndexChanged.connect(lambda: self.update_display("left"))
        self.display_selector_right.currentIndexChanged.connect(lambda: self.update_display("right"))
        
        self.update_timestep_range()
        self.update_debris_list()

    def calculate_background_contribution(self, W_px, H_px, exposure_time_s, quantum_efficiency, 
                                        filter_transmission, focal_length_m, pixel_pitch_um, aperture_area_m2):
        """
        Calculate uniform background illumination contribution for each pixel
        Uses simple scaling from radiance to electrons without wavelength conversion
        
        Note: Uses average QE and filter transmission from OI and Al values
        
        Returns:
            background_electrons_per_pixel: Number of background electrons per pixel
        """
        if not self.enable_background.isChecked():
            return 0.0
        
        # Background parameters
        background_radiance_W_per_sr = self.background_intensity.value()  # W/sr
        
        if background_radiance_W_per_sr <= 0:
            return 0.0
        
        # Use average values for background calculation
        avg_quantum_efficiency = (self.qe_oi.value() + self.qe_al.value()) / 2
        avg_filter_transmission = (self.ft_oi.value() + self.ft_al.value()) / 2
        
        # Calculate pixel solid angle
        pixel_size_m = pixel_pitch_um * 1e-6  # Convert µm to meters
        pixel_solid_angle_sr = (pixel_size_m / focal_length_m) ** 2  # steradians
        
        # Calculate background power collected per pixel
        # Power = Radiance × solid_angle × aperture_area × filter_transmission
        background_power_W = background_radiance_W_per_sr * pixel_solid_angle_sr * aperture_area_m2 * avg_filter_transmission
        
        # Simple conversion: assume ~3e18 photons per second per watt (typical visible light)
        # This avoids wavelength-dependent calculations
        photons_per_watt_per_sec = 3e18  # Reasonable assumption for visible light
        
        background_photons_per_pixel = background_power_W * exposure_time_s * photons_per_watt_per_sec
        background_electrons_per_pixel = background_photons_per_pixel * avg_quantum_efficiency
        
        print(f"Background illumination calculation (simplified):")
        print(f"  Radiance: {background_radiance_W_per_sr:.2e} W/sr")
        print(f"  Average QE: {avg_quantum_efficiency:.3f}")
        print(f"  Average filter transmission: {avg_filter_transmission:.3f}")
        print(f"  Pixel solid angle: {pixel_solid_angle_sr:.2e} sr")
        print(f"  Power per pixel: {background_power_W:.2e} W")
        print(f"  Conversion factor: {photons_per_watt_per_sec:.1e} photons/W/s")
        print(f"  Background result: {background_electrons_per_pixel:.3f} e⁻/pixel")
        
        return background_electrons_per_pixel

    def _on_viz_time_changed(self, datetime):
        """Handle time changes from visualization tab"""
        try:
            analysis_time = Time(datetime.toPyDateTime(), scale='utc')
            
            sats = self.sat_tab.get_satellites()
            if sats:
                sat_idx = self.sat_selector.currentData()
                if sat_idx is not None and 0 <= sat_idx < len(sats):
                    _, sensor_orbit = sats[sat_idx]
                    
                    current_satellite_epoch_offset = (analysis_time - sensor_orbit.epoch).to(u.s).value
                    
                    if (self.single_frame_timestep.minimum() <= current_satellite_epoch_offset <= self.single_frame_timestep.maximum()):
                        self.single_frame_timestep.blockSignals(True)
                        self.single_frame_timestep.setValue(current_satellite_epoch_offset)
                        self.single_frame_timestep.blockSignals(False)
                        
                        print(f"Photon tab timestep synced to: {current_satellite_epoch_offset:.1f}s")
                    
        except Exception as e:
            print(f"Error syncing photon tab timestep: {e}")

    def handle_debris_data_change_for_photon(self):
        """Connected to DebrisTab.debrisDataChanged"""
        self.update_debris_list()
        self.update_timestep_range()

    def update_timestep_range(self):
        """Update timestep range based on debris data"""
        if not self.debris_tab.df.empty and 'Time' in self.debris_tab.df.columns:
            try:
                timesteps = sorted(pd.to_numeric(self.debris_tab.df['Time'], errors='coerce').dropna().unique())
                if len(timesteps) > 0:
                    min_t, max_t = min(timesteps), max(timesteps)
                    if min_t <= max_t:
                        self.single_frame_timestep.setRange(min_t, max_t)
                        current_val = self.single_frame_timestep.value()
                        if not (min_t <= current_val <= max_t):
                            self.single_frame_timestep.setValue(min_t)
                    else:
                        self.single_frame_timestep.setRange(min_t, min_t)
                        self.single_frame_timestep.setValue(min_t)
                else:
                    self.single_frame_timestep.setRange(0, 100)
                    self.single_frame_timestep.setValue(0)
            except Exception as e:
                print(f"Error updating timestep range in Photon Tab: {e}")
                self.single_frame_timestep.setRange(0, 100)
                self.single_frame_timestep.setValue(0)
        else:
            self.single_frame_timestep.setRange(0, 100)
            self.single_frame_timestep.setValue(0)
    
    def calculate_fov_angle(self):
        """Calculate field of view angle"""
        focal_length_m = self.fl.value()
        pixel_size_um = self.pp.value()
        sensor_width_px = self.sensor_width.value()
        sensor_height_px = self.sensor_height.value()
        
        if focal_length_m <= 0: return np.pi / 2
        pixel_size_m = pixel_size_um * 1e-6
        sensor_width_m = sensor_width_px * pixel_size_m
        sensor_height_m = sensor_height_px * pixel_size_m
        sensor_diagonal_m = np.sqrt(sensor_width_m**2 + sensor_height_m**2)
        
        half_fov_rad = np.arctan(sensor_diagonal_m / (2 * focal_length_m))
        return half_fov_rad

    def _check_satellite_visibility_for_debris(self, sat_orbit, sat_name, debris_id, timestep, merged_data, analysis_time):
        """
        Check if a specific satellite can see a specific debris at a given timestep
        Returns visibility info including range, elevation, etc.
        """
        try:
            # Get debris data at this timestep
            timestep_tolerance = 0.001
            time_mask = np.isclose(merged_data['Time'], timestep, atol=timestep_tolerance)
            debris_mask = merged_data['Assembly_ID'] == debris_id
            debris_at_timestep = merged_data[time_mask & debris_mask]
            
            if debris_at_timestep.empty:
                return {
                    'visible': False,
                    'reason': 'No debris data at timestep',
                    'range_km': 0,
                    'elevation_deg': 0
                }
            
            debris_row = debris_at_timestep.iloc[0]
            
            # Get debris position in ECEF
            lat_deg = float(debris_row["Latitude"])
            lon_deg = float(debris_row["Longitude"])
            alt_m = float(debris_row["Altitude"])
            P_debris_ecef_m = lla_to_ecef_wgs84(lat_deg, lon_deg, alt_m)
            
            # Get satellite position in ECEF
            dt = (analysis_time - sat_orbit.epoch)
            sat_propagated = sat_orbit.propagate(dt)
            r_sat_eci_m = sat_propagated.rv()[0].to(u.m).value
            r_sat_ecef_m = eci_to_ecef(r_sat_eci_m, analysis_time)
            
            # Check Earth occlusion using WGS84
            if not is_visible_wgs84_occlusion(r_sat_ecef_m, P_debris_ecef_m):
                return {
                    'visible': False,
                    'reason': 'Earth occluded',
                    'range_km': 0,
                    'elevation_deg': 0
                }
            
            # Calculate range
            debris_vector = P_debris_ecef_m - r_sat_ecef_m
            range_m = np.linalg.norm(debris_vector)
            range_km = range_m / 1000.0
            
            # Calculate elevation angle
            elevation_deg = calculate_elevation_angle_wgs84(r_sat_ecef_m, lat_deg, lon_deg, alt_m)
            
            # Check if in FOV and sensor bounds
            W_px = self.sensor_width.value()
            H_px = self.sensor_height.value()
            focal_mm = self.fl.value() * 1e3
            pixel_mm = self.pp.value() * 1e-3
            pixel_size_mm = (pixel_mm, pixel_mm)
            
            # For visibility check, camera points directly at this debris
            pixel = convert_ecef_to_pixel(P_debris_ecef_m, r_sat_ecef_m, P_debris_ecef_m, 
                                        focal_mm, pixel_size_mm, (W_px, H_px))
            
            if pixel is None:
                return {
                    'visible': False,
                    'reason': 'Behind camera',
                    'range_km': range_km,
                    'elevation_deg': elevation_deg
                }
            
            u_px, v_px = pixel
            if not (0 <= u_px < W_px and 0 <= v_px < H_px):
                return {
                    'visible': False,
                    'reason': 'Outside sensor bounds',
                    'range_km': range_km,
                    'elevation_deg': elevation_deg
                }
            
            # VISIBLE!
            return {
                'visible': True,
                'reason': 'Visible',
                'range_km': range_km,
                'elevation_deg': elevation_deg,
                'pixel_x': u_px,
                'pixel_y': v_px
            }
            
        except Exception as e:
            print(f"Error checking visibility for {sat_name} -> debris {debris_id}: {e}")
            return {
                'visible': False,
                'reason': f'Error: {str(e)}',
                'range_km': 0,
                'elevation_deg': 0
            }

    def _find_best_alternative_satellite(self, all_satellites, debris_id, timestep, merged_data, analysis_time, exclude_satellite=None):
        """
        Find the best alternative satellite to view the debris at this timestep
        Prioritizes closest range among visible satellites
        """
        best_satellite = None
        best_range = float('inf')
        best_visibility = None
        
        for sat_name, sat_orbit in all_satellites:
            if exclude_satellite and sat_name == exclude_satellite:
                continue
                
            visibility = self._check_satellite_visibility_for_debris(
                sat_orbit, sat_name, debris_id, timestep, merged_data, analysis_time
            )
            
            if visibility['visible'] and visibility['range_km'] < best_range:
                best_satellite = (sat_name, sat_orbit)
                best_range = visibility['range_km']
                best_visibility = visibility
        
        if best_satellite:
            return {
                'satellite': best_satellite,
                'visibility': best_visibility
            }
        return None

    def generate_batch_tracking(self):
        """Generate batch tracking with optional multi-satellite handoff"""
        try:
            self.batch_process_btn.setEnabled(False)
            self.batch_process_btn.setText("Processing...")
            self.batch_status_label.setText("Initializing batch processing...")
            self.batch_status_label.setStyleSheet("color: orange; font-size: 9px;")
            QApplication.processEvents()
            
            if self.debris_tab.df.empty or self.em_tab.df.empty:
                QMessageBox.warning(self, "Missing Data", "Please load debris and emissions data first.")
                return
            
            sats = self.sat_tab.get_satellites()
            if not sats:
                QMessageBox.warning(self, "No Satellites", "No satellites defined.")
                return
            
            # Use handoff mode or single satellite mode
            use_handoff = self.enable_satellite_handoff.isChecked()
            
            if use_handoff:
                print(f"\n=== MULTI-SATELLITE HANDOFF MODE ENABLED ===")
                print(f"Available satellites: {[name for name, _ in sats]}")
            else:
                print(f"\n=== SINGLE SATELLITE MODE ===")
            
            sat_idx = self.sat_selector.currentData()
            if sat_idx is None or not (0 <= sat_idx < len(sats)):
                sat_idx = 0
            initial_satellite_name, initial_sensor_orbit = sats[sat_idx]
            
            merged = pd.merge(self.debris_tab.df, self.em_tab.df, on=["Iteration", "Assembly_ID"], how="inner")
            if merged.empty:
                QMessageBox.warning(self, "Merge Error", "No matching data between debris and emissions.")
                return
            
            # Find main body and longest survivor
            main_body_id = 1
            main_body_data = merged[merged['Assembly_ID'] == main_body_id]
            
            merged_sorted = merged.sort_values('Time')
            longest_survivor_id = int(merged_sorted.iloc[-1]['Assembly_ID'])
            longest_survivor_data = merged[merged['Assembly_ID'] == longest_survivor_id]
            
            self.batch_status_label.setText(f"Found: Main body ID {main_body_id}, Longest survivor ID {longest_survivor_id}")
            QApplication.processEvents()
            
            # Create output directory
            timestamp_str = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            save_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Batch Processing", "")
            if not save_dir:
                return
            
            # Convert to absolute path and normalize
            save_dir = os.path.abspath(save_dir)
            print(f"Selected output directory (absolute): {save_dir}")
            
            handoff_suffix = "_handoff" if use_handoff else "_single"
            batch_dir = os.path.join(save_dir, f"batch_tracking{handoff_suffix}_{timestamp_str}")
            batch_dir = os.path.abspath(batch_dir)  # Ensure absolute path
            os.makedirs(batch_dir, exist_ok=True)
            
            main_body_dir = os.path.join(batch_dir, f"main_body_ID{main_body_id}")
            survivor_dir = os.path.join(batch_dir, f"longest_survivor_ID{longest_survivor_id}")
            main_body_dir = os.path.abspath(main_body_dir)  # Ensure absolute paths
            survivor_dir = os.path.abspath(survivor_dir)
            os.makedirs(main_body_dir, exist_ok=True)
            os.makedirs(survivor_dir, exist_ok=True)
            
            print(f"Main body ID: {main_body_id} ({len(main_body_data)} timesteps)")
            print(f"Longest survivor ID: {longest_survivor_id} ({len(longest_survivor_data)} timesteps)")
            print(f"Output directory: {batch_dir}")
            print(f"Background illumination: {'ENABLED' if self.enable_background.isChecked() else 'DISABLED'}")
            print(f"Multi-satellite handoff: {'ENABLED' if use_handoff else 'DISABLED'}")
            print(f"Output formats: NumPy={'ENABLED' if self.output_numpy.isChecked() else 'DISABLED'}, FITS={'ENABLED' if self.output_fits.isChecked() else 'DISABLED'}")
            print(f"Separate OI/Al parameters: QE_OI={self.qe_oi.value():.3f}, QE_Al={self.qe_al.value():.3f}, FT_OI={self.ft_oi.value():.3f}, FT_Al={self.ft_al.value():.3f}")
            
            batch_results = {}
            
            for target_name, target_id, target_data, output_dir in [
                ("Main Body", main_body_id, main_body_data, main_body_dir),
                ("Longest Survivor", longest_survivor_id, longest_survivor_data, survivor_dir)
            ]:
                
                if target_data.empty:
                    print(f"No data for {target_name} ID {target_id}, skipping...")
                    continue
                
                self.batch_status_label.setText(f"Processing {target_name} ID {target_id}...")
                QApplication.processEvents()
                
                timesteps = sorted(target_data['Time'].unique())
                print(f"\nProcessing {target_name} ID {target_id}: {len(timesteps)} timesteps")
                
                if use_handoff:
                    target_results = self._process_target_with_handoff(
                        target_id, timesteps, initial_satellite_name, initial_sensor_orbit,
                        sats, merged, output_dir, target_name
                    )
                else:
                    target_results = self._process_target_single_satellite(
                        target_id, timesteps, initial_satellite_name, initial_sensor_orbit,
                        merged, output_dir, target_name
                    )
                
                batch_results[target_name] = {
                    'debris_id': target_id,
                    'timesteps_processed': len(target_results['frames']),
                    'results': target_results['frames'],
                    'handoff_log': target_results.get('handoff_log', []),
                    'coverage_stats': target_results.get('coverage_stats', {})
                }
                
                # Save timeline CSV
                if target_results['frames']:
                    combined_df = pd.concat([pd.DataFrame(result['visible_debris_data']) 
                                           for result in target_results['frames'] if result['visible_debris_data']], 
                                          ignore_index=True)
                    timeline_path = os.path.join(output_dir, f"{target_name.lower().replace(' ', '_')}_timeline.csv")
                    timeline_path = os.path.abspath(timeline_path)  # Ensure absolute path
                    combined_df.to_csv(timeline_path, index=False)
                    
                    # Save handoff log if using handoff mode
                    if use_handoff and target_results.get('handoff_log'):
                        handoff_df = pd.DataFrame(target_results['handoff_log'])
                        handoff_path = os.path.join(output_dir, f"{target_name.lower().replace(' ', '_')}_handoffs.csv")
                        handoff_path = os.path.abspath(handoff_path)  # Ensure absolute path
                        handoff_df.to_csv(handoff_path, index=False)
                        print(f"  Saved handoff log: {handoff_path}")
                    
                    print(f"  Saved timeline: {timeline_path}")
            
            # Create batch summary with separate OI/Al parameters
            summary = {
                'batch_timestamp': timestamp_str,
                'mode': 'multi_satellite_handoff' if use_handoff else 'single_satellite',
                'initial_satellite_name': initial_satellite_name,
                'available_satellites': [name for name, _ in sats],
                'total_targets_processed': len(batch_results),
                'output_formats': {
                    'numpy_enabled': self.output_numpy.isChecked(),
                    'fits_enabled': self.output_fits.isChecked()
                },
                'results_summary': {name: {
                    'debris_id': data['debris_id'], 
                    'timesteps_processed': data['timesteps_processed'],
                    'handoffs': len(data.get('handoff_log', [])),
                    'satellites_used': list(data.get('coverage_stats', {}).keys())
                } for name, data in batch_results.items()},
                'sensor_params': {
                    'aperture_diameter_m': self.ap.value(),
                    'focal_length_m': self.fl.value(),
                    'pixel_pitch_um': self.pp.value(),
                    'sensor_width_px': self.sensor_width.value(),
                    'sensor_height_px': self.sensor_height.value(),
                    'exposure_time_s': self.te.value(),
                    'quantum_efficiency_oi': self.qe_oi.value(),
                    'quantum_efficiency_al': self.qe_al.value(),
                    'filter_transmission_oi': self.ft_oi.value(),
                    'filter_transmission_al': self.ft_al.value()
                },
                'background_illumination': {
                    'enabled': self.enable_background.isChecked(),
                    'radiance_W_per_sr': self.background_intensity.value() if self.enable_background.isChecked() else 0,
                    'conversion_method': 'Simple scaling (3e18 photons/W/s assumption)'
                }
            }
            
            summary_path = os.path.join(batch_dir, "batch_summary.json")
            summary_path = os.path.abspath(summary_path)  # Ensure absolute path
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.batch_status_label.setText(f"Batch processing complete! Results saved to {batch_dir}")
            self.batch_status_label.setStyleSheet("color: green; font-size: 9px;")
            QApplication.processEvents()
            
            total_frames = sum(data['timesteps_processed'] for data in batch_results.values())
            total_handoffs = sum(len(data.get('handoff_log', [])) for data in batch_results.values())
            
            background_status = f"\nBackground: {'ENABLED' if self.enable_background.isChecked() else 'DISABLED'}"
            if self.enable_background.isChecked():
                background_status += f" ({self.background_intensity.value():.2e} W/sr)"
            
            format_status = f"\nOutput formats: NumPy={'YES' if self.output_numpy.isChecked() else 'NO'}, FITS={'YES' if self.output_fits.isChecked() else 'NO'}"
            
            handoff_status = ""
            if use_handoff:
                handoff_status = f"\nSatellite handoffs: {total_handoffs}"
                for target_name, data in batch_results.items():
                    if data.get('coverage_stats'):
                        handoff_status += f"\n{target_name} used satellites: {', '.join(data['coverage_stats'].keys())}"
            
            qe_al_status = f"\nSeparate OI/Al parameters: QE_OI={self.qe_oi.value():.3f}, QE_Al={self.qe_al.value():.3f}, FT_OI={self.ft_oi.value():.3f}, FT_Al={self.ft_al.value():.3f}"
            
            QMessageBox.information(self, "Batch Processing Complete", 
                                   f"Batch tracking completed successfully!\n\n"
                                   f"Mode: {'Multi-satellite handoff' if use_handoff else 'Single satellite'}\n"
                                   f"Main Body ID {main_body_id}: {batch_results.get('Main Body', {}).get('timesteps_processed', 0)} frames\n"
                                   f"Longest Survivor ID {longest_survivor_id}: {batch_results.get('Longest Survivor', {}).get('timesteps_processed', 0)} frames\n"
                                   f"Total frames processed: {total_frames}"
                                   f"{handoff_status}"
                                   f"{background_status}"
                                   f"{format_status}"
                                   f"{qe_al_status}\n\n"
                                   f"Results saved to: {batch_dir}")
            
            print(f"=== BATCH PROCESSING COMPLETED ===")
            print(f"Mode: {'Multi-satellite handoff' if use_handoff else 'Single satellite'}")
            print(f"Total frames processed: {total_frames}")
            if use_handoff:
                print(f"Total satellite handoffs: {total_handoffs}")
            print(f"Output formats: NumPy={self.output_numpy.isChecked()}, FITS={self.output_fits.isChecked()}")
            print(f"Results directory: {batch_dir}")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error in batch processing: {error_msg}")
            QMessageBox.critical(self, "Batch Processing Error", f"An error occurred during batch processing: {str(e)}")
            self.batch_status_label.setText("Batch processing failed")
            self.batch_status_label.setStyleSheet("color: red; font-size: 9px;")
        finally:
            self.batch_process_btn.setEnabled(True)
            self.batch_process_btn.setText("Track Main Body + Longest Survivor")

    def _process_target_with_handoff(self, target_id, timesteps, initial_sat_name, initial_sat_orbit, 
                                   all_satellites, merged_data, output_dir, target_name):
        """Process target with multi-satellite handoff capability"""
        results = {
            'frames': [],
            'handoff_log': [],
            'coverage_stats': {}
        }
        
        # Start with initial (user-selected) satellite
        current_sat_name = initial_sat_name
        current_sat_orbit = initial_sat_orbit
        
        print(f"\n=== HANDOFF TRACKING FOR {target_name} ===")
        print(f"Starting with satellite: {current_sat_name}")
        
        for i, timestep in enumerate(timesteps):
            try:
                self.batch_status_label.setText(f"Processing {target_name}: timestep {i+1}/{len(timesteps)} (t={timestep}) - {current_sat_name}")
                QApplication.processEvents()
                
                analysis_time = self.viz_tab.snapshot.dateTime().toPyDateTime()
                analysis_time_astropy = Time(analysis_time, scale='utc')
                satellite_analysis_time = analysis_time_astropy + timestep * u.s
                
                # Check if current satellite can see the target
                current_visibility = self._check_satellite_visibility_for_debris(
                    current_sat_orbit, current_sat_name, target_id, timestep, 
                    merged_data, satellite_analysis_time
                )
                
                if current_visibility['visible']:
                    # Keep using current satellite
                    satellite_to_use = (current_sat_name, current_sat_orbit)
                    print(f"  t={timestep:.1f}s: Continue with {current_sat_name} (range: {current_visibility['range_km']:.1f}km)")
                else:
                    # Current satellite lost sight - find best alternative
                    print(f"  t={timestep:.1f}s: {current_sat_name} lost sight ({current_visibility['reason']})")
                    
                    best_alternative = self._find_best_alternative_satellite(
                        all_satellites, target_id, timestep, merged_data, 
                        satellite_analysis_time, exclude_satellite=current_sat_name
                    )
                    
                    if best_alternative:
                        # HANDOFF!
                        old_sat = current_sat_name
                        current_sat_name, current_sat_orbit = best_alternative['satellite']
                        satellite_to_use = best_alternative['satellite']
                        
                        # Log handoff
                        handoff_entry = {
                            'timestep': timestep,
                            'from_satellite': old_sat,
                            'to_satellite': current_sat_name,
                            'reason': current_visibility['reason'],
                            'new_range_km': best_alternative['visibility']['range_km'],
                            'new_elevation_deg': best_alternative['visibility']['elevation_deg']
                        }
                        results['handoff_log'].append(handoff_entry)
                        
                        print(f"  t={timestep:.1f}s: HANDOFF {old_sat} → {current_sat_name} (range: {best_alternative['visibility']['range_km']:.1f}km)")
                    else:
                        # No satellite can see it
                        print(f"  t={timestep:.1f}s: No satellite can see debris {target_id}")
                        continue
                
                # Track coverage statistics
                if current_sat_name not in results['coverage_stats']:
                    results['coverage_stats'][current_sat_name] = {
                        'frames_generated': 0,
                        'first_timestep': timestep,
                        'last_timestep': timestep
                    }
                results['coverage_stats'][current_sat_name]['frames_generated'] += 1
                results['coverage_stats'][current_sat_name]['last_timestep'] = timestep
                
                # Generate frame with selected satellite
                frame_result = self._generate_single_batch_frame_with_satellite(
                    timestep, target_id, satellite_to_use, merged_data, output_dir, i
                )
                
                if frame_result:
                    frame_result['satellite_used'] = current_sat_name
                    results['frames'].append(frame_result)
                    print(f"    Frame generated: {frame_result['visible_count']} visible debris")
                
            except Exception as e:
                print(f"Error processing {target_name} timestep {timestep}: {e}")
                continue
        
        # Print handoff summary
        if results['handoff_log']:
            print(f"\n{target_name} Handoff Summary:")
            print(f"  Total handoffs: {len(results['handoff_log'])}")
            for handoff in results['handoff_log']:
                print(f"    t={handoff['timestep']:.1f}s: {handoff['from_satellite']} → {handoff['to_satellite']} ({handoff['reason']})")
        
        print(f"\n{target_name} Coverage Summary:")
        for sat_name, stats in results['coverage_stats'].items():
            duration = stats['last_timestep'] - stats['first_timestep']
            print(f"  {sat_name}: {stats['frames_generated']} frames, t={stats['first_timestep']:.1f}-{stats['last_timestep']:.1f}s ({duration:.1f}s)")
        
        return results

    def _process_target_single_satellite(self, target_id, timesteps, satellite_name, satellite_orbit, 
                                       merged_data, output_dir, target_name):
        """Process target with single satellite (original behavior)"""
        results = {
            'frames': [],
            'handoff_log': [],
            'coverage_stats': {satellite_name: {'frames_generated': 0}}
        }
        
        print(f"\n=== SINGLE SATELLITE TRACKING FOR {target_name} ===")
        print(f"Using satellite: {satellite_name}")
        
        for i, timestep in enumerate(timesteps):
            try:
                self.batch_status_label.setText(f"Processing {target_name}: timestep {i+1}/{len(timesteps)} (t={timestep}) - {satellite_name}")
                QApplication.processEvents()
                
                frame_result = self._generate_single_batch_frame_with_satellite(
                    timestep, target_id, (satellite_name, satellite_orbit), merged_data, output_dir, i
                )
                
                if frame_result:
                    frame_result['satellite_used'] = satellite_name
                    results['frames'].append(frame_result)
                    results['coverage_stats'][satellite_name]['frames_generated'] += 1
                    print(f"  t={timestep:.1f}s: {frame_result['visible_count']} visible debris")
                
            except Exception as e:
                print(f"Error processing {target_name} timestep {timestep}: {e}")
                continue
        
        print(f"\n{target_name} processed: {len(results['frames'])} frames with {satellite_name}")
        return results

    def _generate_single_batch_frame_with_satellite(self, target_timestep, tracked_debris_id, 
                                                  satellite_tuple, merged_data, output_dir, frame_index):
        """Generate single frame using a specific satellite with separate OI/Al parameters and configurable output formats"""
        satellite_name, sensor_orbit = satellite_tuple
        
        timestep_tolerance = 0.001
        time_mask = np.isclose(merged_data['Time'], target_timestep, atol=timestep_tolerance)
        timestep_data = merged_data[time_mask]
        
        if timestep_data.empty:
            return None
        
        tracked_data = timestep_data[timestep_data['Assembly_ID'] == tracked_debris_id]
        if tracked_data.empty:
            return None
        
        # Ensure output_dir is absolute path
        output_dir = os.path.abspath(output_dir)
        
        # Sensor parameters
        W_px = self.sensor_width.value()
        H_px = self.sensor_height.value()
        aperture_diameter_m = self.ap.value()
        aperture_area_m2 = np.pi * (aperture_diameter_m / 2)**2
        focal_length_input_m = self.fl.value()
        focal_length_mm = focal_length_input_m * 1e3
        pixel_pitch_um = self.pp.value()
        pixel_pitch_mm = pixel_pitch_um * 1e-3
        pixel_size_mm_tuple = (pixel_pitch_mm, pixel_pitch_mm)
        exposure_time_s = self.te.value()
        
        # Separate parameters for OI and Al
        quantum_efficiency_oi = self.qe_oi.value()
        quantum_efficiency_al = self.qe_al.value()
        filter_transmission_oi = self.ft_oi.value()
        filter_transmission_al = self.ft_al.value()
        
        fov_half_angle_rad = self.calculate_fov_angle()
        fov_cos_threshold = np.cos(fov_half_angle_rad)
        
        lambda_oi_nm = self.lm_oi.value() * u.nm
        Eph_oi_J = (h * c / lambda_oi_nm).to(u.J).value
        lambda_al_nm = self.lm_al.value() * u.nm
        Eph_al_J = (h * c / lambda_al_nm).to(u.J).value
        
        # Time synchronization & satellite position with proper coordinate conversion
        analysis_time_dt = self.viz_tab.snapshot.dateTime().toPyDateTime()
        analysis_time_astropy = Time(analysis_time_dt, scale='utc')
        current_satellite_epoch_offset = (analysis_time_astropy - sensor_orbit.epoch).to(u.s).value
        effective_propagation_time_s = current_satellite_epoch_offset + target_timestep
        sat_propagated = sensor_orbit.propagate(effective_propagation_time_s * u.s)
        
        # Get satellite position in ECI, convert to ECEF
        C_satellite_eci_m = sat_propagated.r.to(u.m).value
        C_satellite_ecef_m = eci_to_ecef(C_satellite_eci_m, analysis_time_astropy)
        
        # Camera look-at point (tracked debris) with proper LLA to ECEF conversion
        tracked_row = tracked_data.iloc[0]
        lat_tracked = float(tracked_row["Latitude"])
        lon_tracked = float(tracked_row["Longitude"])
        alt_tracked = float(tracked_row["Altitude"])  # Already in meters
        
        P_camera_look_at_ecef_m = lla_to_ecef_wgs84(lat_tracked, lon_tracked, alt_tracked)
        
        # Camera boresight
        vec_sat_to_look_at = P_camera_look_at_ecef_m - C_satellite_ecef_m
        dist_sat_to_look_at = np.linalg.norm(vec_sat_to_look_at)
        if dist_sat_to_look_at == 0:
            return None
        F_camera_boresight_unit_vec = vec_sat_to_look_at / dist_sat_to_look_at
        
        # Initialize image arrays
        img_oi_rgb = np.zeros((H_px, W_px, 3), dtype=np.float64)
        img_al_rgb = np.zeros((H_px, W_px, 3), dtype=np.float64)
        
        # Calculate background contribution per pixel (uses averaged QE and filter transmission)
        background_electrons = self.calculate_background_contribution(
            W_px, H_px, exposure_time_s, (quantum_efficiency_oi + quantum_efficiency_al) / 2, 
            (filter_transmission_oi + filter_transmission_al) / 2, 
            focal_length_input_m, pixel_pitch_um, aperture_area_m2
        )
        
        # Add uniform background to all pixels if enabled
        if self.enable_background.isChecked() and background_electrons > 0:
            background_rgb_value = background_electrons * 0.5  # Scale for visualization
            
            img_oi_rgb[:, :, 0] += background_rgb_value  # Red channel
            img_oi_rgb[:, :, 1] += background_rgb_value  # Green channel
            img_oi_rgb[:, :, 2] += background_rgb_value  # Blue channel
            
            img_al_rgb[:, :, 0] += background_rgb_value  # Red channel
            img_al_rgb[:, :, 1] += background_rgb_value  # Green channel
            img_al_rgb[:, :, 2] += background_rgb_value  # Blue channel
        
        frame_results_oi = []
        frame_results_al = []
        
        # Assign colors
        debris_colors = {}
        unique_ids_in_frame = timestep_data['Assembly_ID'].unique()
        for i, unique_id_val in enumerate(unique_ids_in_frame):
            if unique_id_val == tracked_debris_id:
                debris_colors[unique_id_val] = (1.0, 1.0, 0.0)
            else:
                hue = i / len(unique_ids_in_frame) if len(unique_ids_in_frame) > 0 else 0
                debris_colors[unique_id_val] = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        
        visible_count = 0
        angular_correction_count = 0
        
        # Process all debris at this timestep
        for _, debris_row in timestep_data.iterrows():
            current_debris_id = int(debris_row["Assembly_ID"])
            
            lat_obj = float(debris_row["Latitude"])
            lon_obj = float(debris_row["Longitude"])
            alt_obj = float(debris_row["Altitude"])  # Already in meters
            
            # Convert debris LLA to ECEF using WGS84
            P_obj_ecef_m = lla_to_ecef_wgs84(lat_obj, lon_obj, alt_obj)
            
            # Visibility checks using WGS84 occlusion
            if not is_visible_wgs84_occlusion(C_satellite_ecef_m, P_obj_ecef_m):
                continue
            
            vec_sat_to_obj = P_obj_ecef_m - C_satellite_ecef_m
            dist_sat_to_obj_m = np.linalg.norm(vec_sat_to_obj)
            if dist_sat_to_obj_m == 0:
                continue
            F_obj_direction_unit_vec = vec_sat_to_obj / dist_sat_to_obj_m
            
            dot_product_fov = np.dot(F_camera_boresight_unit_vec, F_obj_direction_unit_vec)
            if dot_product_fov < fov_cos_threshold:
                continue
            
            # Project to pixel coordinates using fixed function
            pixel_coords = convert_ecef_to_pixel(P_obj_ecef_m, C_satellite_ecef_m, P_camera_look_at_ecef_m,
                                            focal_length_mm, pixel_size_mm_tuple, (W_px, H_px))
            
            if pixel_coords is None:
                continue
            u_px, v_px = pixel_coords
            if not (0 <= u_px < W_px and 0 <= v_px < H_px):
                continue
            
            visible_count += 1
            range_km = dist_sat_to_obj_m / 1000.0
            
            # Process OI emissions with WGS84 angular correction and separate parameters
            I_emission_oi = float(debris_row.get("OI_emissions_atomic", 0.0))
            photons_oi = 0
            electrons_oi = 0
            irradiance_oi = 0
            elevation_angle_oi = 0
            cos_factor_oi = 0
            
            if I_emission_oi > 0 and range_km > 0:
                irradiance_oi = I_emission_oi / (dist_sat_to_obj_m**2)
                power_oi = irradiance_oi * aperture_area_m2 * filter_transmission_oi  # Use OI-specific filter
                photons_oi_raw = (power_oi * exposure_time_s) / Eph_oi_J
                electrons_oi_raw = photons_oi_raw * quantum_efficiency_oi  # Use OI-specific QE
                
                # Apply angular correction using WGS84
                photons_oi, electrons_oi, elevation_angle_oi, cos_factor_oi = apply_angular_correction(
                    photons_oi_raw, electrons_oi_raw, C_satellite_ecef_m, lat_obj, lon_obj, alt_obj
                )
                
                if cos_factor_oi > 0:
                    angular_correction_count += 1
            
            # Process Combined Al emissions with WGS84 angular correction and separate parameters
            I_emission_al = 0.0
            I_emission_al += float(debris_row.get("AlI_1_emissions_atomic", 0.0))
            I_emission_al += float(debris_row.get("AlI_2_emissions_atomic", 0.0))
            photons_al = 0
            electrons_al = 0
            irradiance_al = 0
            elevation_angle_al = 0
            cos_factor_al = 0
            
            if I_emission_al > 0 and range_km > 0:
                irradiance_al = I_emission_al / (dist_sat_to_obj_m**2)
                power_al = irradiance_al * aperture_area_m2 * filter_transmission_al  # Use Al-specific filter
                photons_al_raw = (power_al * exposure_time_s) / Eph_al_J
                electrons_al_raw = photons_al_raw * quantum_efficiency_al  # Use Al-specific QE
                
                # Apply angular correction using WGS84
                photons_al, electrons_al, elevation_angle_al, cos_factor_al = apply_angular_correction(
                    photons_al_raw, electrons_al_raw, C_satellite_ecef_m, lat_obj, lon_obj, alt_obj
                )
            
            # Store results with angular correction metadata
            frame_results_oi.append({
                "Time": target_timestep, "Assembly_ID": current_debris_id, "Range_km": range_km,
                "I_Wsr": I_emission_oi, "E_Wm2": irradiance_oi,
                "Photons": photons_oi, "Electrons": electrons_oi,
                "u_px": u_px, "v_px": v_px,
                "Elevation_deg": elevation_angle_oi, "Cos_factor": cos_factor_oi,
                "Satellite": satellite_name
            })
            
            frame_results_al.append({
                "Time": target_timestep, "Assembly_ID": current_debris_id, "Range_km": range_km,
                "I_Wsr": I_emission_al, "E_Wm2": irradiance_al,
                "Photons": photons_al, "Electrons": electrons_al,
                "u_px": u_px, "v_px": v_px,
                "Elevation_deg": elevation_angle_al, "Cos_factor": cos_factor_al,
                "Satellite": satellite_name
            })
            
            # Add to images
            u_idx, v_idx = int(round(u_px)), int(round(v_px))
            if 0 <= u_idx < W_px and 0 <= v_idx < H_px:
                r_val, g_val, b_val = debris_colors.get(current_debris_id, (0.5, 0.5, 0.5))
                
                if electrons_oi > 0:
                    img_oi_rgb[v_idx, u_idx, 0] += electrons_oi * r_val
                    img_oi_rgb[v_idx, u_idx, 1] += electrons_oi * g_val
                    img_oi_rgb[v_idx, u_idx, 2] += electrons_oi * b_val
                
                if electrons_al > 0:
                    img_al_rgb[v_idx, u_idx, 0] += electrons_al * r_val
                    img_al_rgb[v_idx, u_idx, 1] += electrons_al * g_val
                    img_al_rgb[v_idx, u_idx, 2] += electrons_al * b_val
        
        if visible_count == 0:
            return None
        
        # Save files based on selected formats with absolute paths
        frame_basename = f"frame_{frame_index:03d}_t{target_timestep:.1f}_ID{tracked_debris_id}_{satellite_name}"
        
        # Save OI results
        oi_basename = f"{frame_basename}_OI"
        self._save_frame_data(output_dir, oi_basename, img_oi_rgb, frame_results_oi)
        
        # Save Combined Al results
        al_basename = f"{frame_basename}_AlI_combined"
        self._save_frame_data(output_dir, al_basename, img_al_rgb, frame_results_al)
        
        return {
            'timestep': target_timestep,
            'tracked_debris_id': tracked_debris_id,
            'visible_count': visible_count,
            'visible_debris_data': frame_results_oi,
            'frame_index': frame_index,
            'satellite_position_ecef_m': C_satellite_ecef_m.tolist(),
            'camera_look_at_ecef_m': P_camera_look_at_ecef_m.tolist(),
            'angular_correction_count': angular_correction_count,
            'background_electrons_per_pixel': background_electrons if self.enable_background.isChecked() else 0,
            'satellite_used': satellite_name
        }

    def _save_frame_data(self, output_dir, basename, image_array, results_data):
        """Save frame data in selected formats (NumPy and/or FITS) with absolute paths and ASCII-safe headers"""
        # Ensure output_dir is absolute path
        output_dir = os.path.abspath(output_dir)
        
        # Always save CSV results
        csv_path = os.path.join(output_dir, f"{basename}_results.csv")
        csv_path = os.path.abspath(csv_path)  # Ensure absolute path
        print(f"Saving CSV to: {csv_path}")
        pd.DataFrame(results_data).to_csv(csv_path, index=False)
        
        # Save NumPy array if enabled
        if self.output_numpy.isChecked():
            npy_path = os.path.join(output_dir, f"{basename}.npy")
            npy_path = os.path.abspath(npy_path)  # Ensure absolute path
            print(f"Saving NumPy to: {npy_path}")
            np.save(npy_path, image_array)
        
        # Save FITS file if enabled
        if self.output_fits.isChecked():
            fits_path = os.path.join(output_dir, f"{basename}.fits")
            fits_path = os.path.abspath(fits_path)  # Ensure absolute path
            print(f"Saving FITS to: {fits_path}")
            try:
                from astropy.io import fits
                
                # Create FITS header with metadata (ASCII-safe comments)
                header = fits.Header()
                header['CREATED'] = (Time.now().iso, 'File creation timestamp (UTC)')
                header['SOFTWARE'] = ('ESA Viewer - Photon Conversion', 'Analysis software')
                header['UNITS'] = ('electrons', 'Image pixel units')
                header['COMMENT'] = 'Photon conversion simulation results'
                
                # Add sensor parameters to header with ASCII-safe comments
                header['APERTURE'] = (self.ap.value(), 'Aperture diameter [m]')
                header['FOCALLEN'] = (self.fl.value(), 'Focal length [m]')
                header['PIXPITCH'] = (self.pp.value(), 'Pixel pitch [microns]')  # Changed from [μm]
                header['EXPOSURE'] = (self.te.value(), 'Exposure time [s]')
                header['SENSW'] = (self.sensor_width.value(), 'Sensor width [px]')
                header['SENSH'] = (self.sensor_height.value(), 'Sensor height [px]')
                
                # Add emission-specific parameters
                if 'OI' in basename:
                    header['EMISSION'] = ('OI_atomic', 'Emission type')
                    header['WAVELEN'] = (self.lm_oi.value(), 'Wavelength [nm]')
                    header['QE'] = (self.qe_oi.value(), 'Quantum efficiency')
                    header['FILTERTX'] = (self.ft_oi.value(), 'Filter transmission')
                else:
                    header['EMISSION'] = ('AlI_combined', 'Emission type')
                    header['WAVELEN'] = (self.lm_al.value(), 'Average wavelength [nm]')
                    header['QE'] = (self.qe_al.value(), 'Quantum efficiency')
                    header['FILTERTX'] = (self.ft_al.value(), 'Filter transmission')
                
                # Add background info
                header['BKGD_EN'] = (self.enable_background.isChecked(), 'Background illumination enabled')
                if self.enable_background.isChecked():
                    header['BKGD_RAD'] = (self.background_intensity.value(), 'Background radiance [W/sr]')
                
                # Create HDU and save
                hdu = fits.PrimaryHDU(image_array, header=header)
                hdul = fits.HDUList([hdu])
                hdul.writeto(fits_path, overwrite=True)
                hdul.close()
                
                print(f"Successfully saved FITS file: {fits_path}")
                
            except ImportError:
                print(f"Warning: astropy not available for FITS output. Skipping {basename}.fits")
            except Exception as e:
                print(f"Error saving FITS file {basename}.fits: {e}")

    def generate_single_frame_dual(self):
        """Generate single frame with proper coordinate transformations, background illumination, and separate OI/Al parameters"""
        try:
            self.generate_single_frame_btn.setEnabled(False)
            self.generate_single_frame_btn.setText("Computing...")
            QApplication.processEvents()
            
            tracked_debris_id_val = self.tracked_debris.currentData()
            if tracked_debris_id_val is None or tracked_debris_id_val < 0:
                QMessageBox.warning(self, "No Tracked Debris", "Please select a debris object to track and center.")
                return
            
            target_timestep = self.single_frame_timestep.value()
            
            sats = self.sat_tab.get_satellites()
            if not sats: 
                QMessageBox.warning(self, "No Satellites", "No satellites defined.")
                return
            
            sat_idx = self.sat_selector.currentData()
            if sat_idx is None or not (0 <= sat_idx < len(sats)): 
                sat_idx = 0
            satellite_name, sensor_orbit = sats[sat_idx]
            
            if self.debris_tab.df.empty or self.em_tab.df.empty:
                QMessageBox.warning(self, "Missing Data", "Please load debris and emissions data first.")
                return
            
            # Check for required emission columns
            required_oi = "OI_emissions_atomic" in self.em_tab.df.columns
            required_al = "AlI_1_emissions_atomic" in self.em_tab.df.columns or "AlI_2_emissions_atomic" in self.em_tab.df.columns
            
            if not required_oi:
                QMessageBox.warning(self, "Missing OI Data", "OI_emissions_atomic column not found in emissions data.")
                return
            if not required_al:
                QMessageBox.warning(self, "Missing Al Data", "Neither AlI_1_emissions_atomic nor AlI_2_emissions_atomic found in emissions data.")
                return

            merged = pd.merge(self.debris_tab.df, self.em_tab.df, on=["Iteration", "Assembly_ID"], how="inner")
            if merged.empty: 
                QMessageBox.warning(self, "Merge Error", "No matching data between debris and emissions.")
                return

            timestep_tolerance = 0.001
            time_mask = np.isclose(merged['Time'], target_timestep, atol=timestep_tolerance)
            timestep_data = merged[time_mask]
            
            if timestep_data.empty:
                available_times = sorted(merged['Time'].unique())[:5]
                QMessageBox.warning(self, "No Data at Timestep", f"No data at timestep {target_timestep}.\nAvailable starts: {available_times}...")
                return
            
            tracked_data = timestep_data[timestep_data['Assembly_ID'] == tracked_debris_id_val]
            if tracked_data.empty:
                debris_at_ts = timestep_data['Assembly_ID'].unique()
                QMessageBox.warning(self, "Tracked Debris Not Found", f"Debris ID {tracked_debris_id_val} not at t={target_timestep}.\nAvailable IDs: {debris_at_ts}")
                return

            # Sensor and Camera Parameters
            W_px = self.sensor_width.value()
            H_px = self.sensor_height.value()
            aperture_diameter_m = self.ap.value()
            aperture_area_m2 = np.pi * (aperture_diameter_m / 2)**2
            focal_length_input_m = self.fl.value() 
            focal_length_mm = focal_length_input_m * 1e3
            pixel_pitch_um = self.pp.value()
            pixel_pitch_mm = pixel_pitch_um * 1e-3
            pixel_size_mm_tuple = (pixel_pitch_mm, pixel_pitch_mm)
            exposure_time_s = self.te.value()
            
            # Separate parameters for OI and Al
            quantum_efficiency_oi = self.qe_oi.value()
            quantum_efficiency_al = self.qe_al.value()
            filter_transmission_oi = self.ft_oi.value()
            filter_transmission_al = self.ft_al.value()
            
            fov_half_angle_rad = self.calculate_fov_angle()
            fov_cos_threshold = np.cos(fov_half_angle_rad)
            
            lambda_oi_nm = self.lm_oi.value() * u.nm
            Eph_oi_J = (h * c / lambda_oi_nm).to(u.J).value
            lambda_al_nm = self.lm_al.value() * u.nm
            Eph_al_J = (h * c / lambda_al_nm).to(u.J).value
            
            # Time Synchronization & Satellite Position
            analysis_time_dt = self.viz_tab.snapshot.dateTime().toPyDateTime() 
            analysis_time_astropy = Time(analysis_time_dt, scale='utc')
            
            current_satellite_epoch_offset = (analysis_time_astropy - sensor_orbit.epoch).to(u.s).value
            effective_propagation_time_s = current_satellite_epoch_offset + target_timestep
            sat_propagated = sensor_orbit.propagate(effective_propagation_time_s * u.s)
            
            # Get satellite position in ECI, convert to ECEF
            C_satellite_eci_m = sat_propagated.r.to(u.m).value
            C_satellite_ecef_m = eci_to_ecef(C_satellite_eci_m, analysis_time_astropy)

            # Camera Look-at Point (Tracked Debris) with proper LLA to ECEF conversion
            tracked_row = tracked_data.iloc[0]
            lat_tracked = float(tracked_row["Latitude"])
            lon_tracked = float(tracked_row["Longitude"])
            alt_tracked = float(tracked_row["Altitude"])  # Already in meters
            
            P_camera_look_at_ecef_m = lla_to_ecef_wgs84(lat_tracked, lon_tracked, alt_tracked)

            # Camera Boresight (Forward Vector)
            vec_sat_to_look_at = P_camera_look_at_ecef_m - C_satellite_ecef_m
            dist_sat_to_look_at = np.linalg.norm(vec_sat_to_look_at)
            if dist_sat_to_look_at == 0: 
                QMessageBox.warning(self, "Error", "Satellite is at the tracked debris position.")
                return
            F_camera_boresight_unit_vec = vec_sat_to_look_at / dist_sat_to_look_at

            print(f"\n--- Generating Dual Frame (OI + Combined Al) with Separate Parameters ---")
            print(f"  Satellite: {satellite_name} at {C_satellite_ecef_m/1000} km ECEF")
            print(f"  Camera Look-at (Tracked Debris ID {tracked_debris_id_val}): {P_camera_look_at_ecef_m/1000} km ECEF")
            print(f"  Distance to Look-at Point: {dist_sat_to_look_at/1000:.1f} km")
            print(f"  Debris Timestep: {target_timestep}s")
            print(f"  Sensor: {W_px}x{H_px}px, FOV ~ {2*np.degrees(fov_half_angle_rad):.2f}° (diag)")
            print(f"  Background illumination: {'ENABLED' if self.enable_background.isChecked() else 'DISABLED'}")
            print(f"  Output formats: NumPy={'ENABLED' if self.output_numpy.isChecked() else 'DISABLED'}, FITS={'ENABLED' if self.output_fits.isChecked() else 'DISABLED'}")
            print(f"  OI Parameters: QE={quantum_efficiency_oi:.3f}, Filter={filter_transmission_oi:.3f}")
            print(f"  Al Parameters: QE={quantum_efficiency_al:.3f}, Filter={filter_transmission_al:.3f}")

            # Initialize image arrays
            img_oi_rgb = np.zeros((H_px, W_px, 3), dtype=np.float64)
            img_al_rgb = np.zeros((H_px, W_px, 3), dtype=np.float64)
            
            # Calculate background contribution per pixel (uses averaged parameters)
            background_electrons = self.calculate_background_contribution(
                W_px, H_px, exposure_time_s, (quantum_efficiency_oi + quantum_efficiency_al) / 2, 
                (filter_transmission_oi + filter_transmission_al) / 2, 
                focal_length_input_m, pixel_pitch_um, aperture_area_m2
            )
            self.last_background_electrons = background_electrons
            
            # Add uniform background to all pixels if enabled
            if self.enable_background.isChecked() and background_electrons > 0:
                print(f"Adding uniform background: {background_electrons:.3f} e⁻/pixel")
                
                # Add background as white/gray (equal RGB values)
                background_rgb_value = background_electrons * 0.5  # Scale for visualization
                
                img_oi_rgb[:, :, 0] += background_rgb_value  # Red channel
                img_oi_rgb[:, :, 1] += background_rgb_value  # Green channel  
                img_oi_rgb[:, :, 2] += background_rgb_value  # Blue channel
                
                img_al_rgb[:, :, 0] += background_rgb_value  # Red channel
                img_al_rgb[:, :, 1] += background_rgb_value  # Green channel
                img_al_rgb[:, :, 2] += background_rgb_value  # Blue channel
            
            frame_results_oi = []
            frame_results_al = []
            
            debris_colors = {}
            unique_ids_in_frame = timestep_data['Assembly_ID'].unique()
            for i, unique_id_val in enumerate(unique_ids_in_frame):
                if unique_id_val == tracked_debris_id_val: 
                    debris_colors[unique_id_val] = (1.0, 1.0, 0.0)
                else: 
                    hue = i / len(unique_ids_in_frame) if len(unique_ids_in_frame) > 0 else 0
                    debris_colors[unique_id_val] = colorsys.hsv_to_rgb(hue, 0.9, 1.0)

            visible_count = 0
            not_in_fov_count = 0
            behind_cam_count = 0
            occluded_count = 0

            for _, debris_row in timestep_data.iterrows():
                current_debris_id = int(debris_row["Assembly_ID"])
                
                lat_obj = float(debris_row["Latitude"])
                lon_obj = float(debris_row["Longitude"])
                alt_obj = float(debris_row["Altitude"])  # Already in meters
                
                # Convert debris LLA to ECEF using WGS84
                P_obj_ecef_m = lla_to_ecef_wgs84(lat_obj, lon_obj, alt_obj)

                # Visibility check using WGS84 occlusion
                if not is_visible_wgs84_occlusion(C_satellite_ecef_m, P_obj_ecef_m):
                    occluded_count += 1
                    continue

                vec_sat_to_obj = P_obj_ecef_m - C_satellite_ecef_m
                dist_sat_to_obj_m = np.linalg.norm(vec_sat_to_obj)
                if dist_sat_to_obj_m == 0: 
                    continue
                F_obj_direction_unit_vec = vec_sat_to_obj / dist_sat_to_obj_m
                
                dot_product_fov = np.dot(F_camera_boresight_unit_vec, F_obj_direction_unit_vec)
                if dot_product_fov < fov_cos_threshold:
                    not_in_fov_count += 1
                    continue
                
                # Project using coordinate transformation
                pixel_coords = convert_ecef_to_pixel(P_obj_ecef_m, C_satellite_ecef_m, P_camera_look_at_ecef_m, 
                                                    focal_length_mm, pixel_size_mm_tuple, (W_px, H_px))
                
                if pixel_coords is None: 
                    behind_cam_count += 1
                    continue
                u_px, v_px = pixel_coords
                if not (0 <= u_px < W_px and 0 <= v_px < H_px): 
                    not_in_fov_count += 1
                    continue

                visible_count += 1
                range_km = dist_sat_to_obj_m / 1000.0
                
                # Process OI emissions with WGS84 angular correction and separate parameters
                I_emission_oi = float(debris_row.get("OI_emissions_atomic", 0.0))
                photons_oi = 0
                electrons_oi = 0
                irradiance_oi = 0
                elevation_angle_oi = 0
                cos_factor_oi = 0
                if I_emission_oi > 0 and range_km > 0:
                    irradiance_oi = I_emission_oi / (dist_sat_to_obj_m**2)
                    power_oi = irradiance_oi * aperture_area_m2 * filter_transmission_oi  # Use OI-specific filter
                    photons_oi_raw = (power_oi * exposure_time_s) / Eph_oi_J
                    electrons_oi_raw = photons_oi_raw * quantum_efficiency_oi  # Use OI-specific QE
                    
                    # Apply WGS84 angular correction
                    photons_oi, electrons_oi, elevation_angle_oi, cos_factor_oi = apply_angular_correction(
                        photons_oi_raw, electrons_oi_raw, C_satellite_ecef_m, lat_obj, lon_obj, alt_obj
                    )

                # Process Combined Al emissions with WGS84 angular correction and separate parameters
                I_emission_al = 0.0
                I_emission_al += float(debris_row.get("AlI_1_emissions_atomic", 0.0))
                I_emission_al += float(debris_row.get("AlI_2_emissions_atomic", 0.0))
                photons_al = 0
                electrons_al = 0
                irradiance_al = 0
                elevation_angle_al = 0
                cos_factor_al = 0
                if I_emission_al > 0 and range_km > 0:
                    irradiance_al = I_emission_al / (dist_sat_to_obj_m**2)
                    power_al = irradiance_al * aperture_area_m2 * filter_transmission_al  # Use Al-specific filter
                    photons_al_raw = (power_al * exposure_time_s) / Eph_al_J
                    electrons_al_raw = photons_al_raw * quantum_efficiency_al  # Use Al-specific QE
                    
                    # Apply WGS84 angular correction
                    photons_al, electrons_al, elevation_angle_al, cos_factor_al = apply_angular_correction(
                        photons_al_raw, electrons_al_raw, C_satellite_ecef_m, lat_obj, lon_obj, alt_obj
                    )
                
                # Store results for both emissions
                frame_results_oi.append({
                    "Time": target_timestep, "Assembly_ID": current_debris_id, "Range_km": range_km,
                    "I_Wsr": I_emission_oi, "E_Wm2": irradiance_oi,
                    "Photons": photons_oi, "Electrons": electrons_oi,
                    "u_px": u_px, "v_px": v_px
                })
                
                frame_results_al.append({
                    "Time": target_timestep, "Assembly_ID": current_debris_id, "Range_km": range_km,
                    "I_Wsr": I_emission_al, "E_Wm2": irradiance_al,
                    "Photons": photons_al, "Electrons": electrons_al,
                    "u_px": u_px, "v_px": v_px
                })
                
                # Add to images
                u_idx, v_idx = int(round(u_px)), int(round(v_px))
                if 0 <= u_idx < W_px and 0 <= v_idx < H_px:
                    r_val, g_val, b_val = debris_colors.get(current_debris_id, (0.5,0.5,0.5))
                    
                    if electrons_oi > 0:
                        img_oi_rgb[v_idx, u_idx, 0] += electrons_oi * r_val
                        img_oi_rgb[v_idx, u_idx, 1] += electrons_oi * g_val
                        img_oi_rgb[v_idx, u_idx, 2] += electrons_oi * b_val
                    
                    if electrons_al > 0:
                        img_al_rgb[v_idx, u_idx, 0] += electrons_al * r_val
                        img_al_rgb[v_idx, u_idx, 1] += electrons_al * g_val
                        img_al_rgb[v_idx, u_idx, 2] += electrons_al * b_val
                
                if current_debris_id == tracked_debris_id_val:
                    background_info = f", Background: {background_electrons:.3f} e⁻/pixel" if self.enable_background.isChecked() else ""
                    print(f"  Tracked Debris ID {current_debris_id} details in frame:")
                    print(f"    Pixel: ({u_px:.1f}, {v_px:.1f}), Range: {range_km:.1f} km")
                    print(f"    OI Electrons: {electrons_oi:.1f}, Al Electrons: {electrons_al:.1f}{background_info}")

            print(f"  Frame Summary: Visible Debris Points: {visible_count}, Occluded: {occluded_count}, Not in FOV/Sensor: {not_in_fov_count}, Behind Camera: {behind_cam_count}")
            
            if not frame_results_oi and not frame_results_al:
                QMessageBox.warning(self, "No Visible Debris", "No debris was visible in the frame under current parameters.")
                return

            df_oi = pd.DataFrame(frame_results_oi).sort_values(by=['Assembly_ID', 'Time'])
            df_al = pd.DataFrame(frame_results_al).sort_values(by=['Assembly_ID', 'Time'])
            
            background_status = ""
            if self.enable_background.isChecked():
                background_status = f" + {background_electrons:.3f} e⁻/pixel background"
            
            format_status = f"Output formats: NumPy={'ENABLED' if self.output_numpy.isChecked() else 'DISABLED'}, FITS={'ENABLED' if self.output_fits.isChecked() else 'DISABLED'}"
            
            print(f"  Computation complete! Results ready for display and saving.")
            print(f"    OI results: {len(df_oi)} debris points{background_status}")
            print(f"    Combined Al results: {len(df_al)} debris points{background_status}")
            print(f"    {format_status}")

            # Store and display results
            self.result_images_preview["OI_emissions_atomic"] = img_oi_rgb.copy()
            self.result_data_preview["OI_emissions_atomic"] = df_oi.copy()
            self.result_images_preview["combined_al"] = img_al_rgb.copy()
            self.result_data_preview["combined_al"] = df_al.copy()
            self.computed_preview = True
            
            # Store metadata with separate parameters
            self.last_generation_metadata = {
                "satellite_name": satellite_name,
                "tracked_debris_id": tracked_debris_id_val,
                "target_timestep": target_timestep,
                "positions": {
                    "satellite_ecef_m": C_satellite_ecef_m.tolist(),
                    "camera_look_at_ecef_m": P_camera_look_at_ecef_m.tolist()
                },
                "sensor_params": {
                    "aperture_diameter_m": aperture_diameter_m,
                    "focal_length_input_m": focal_length_input_m,
                    "pixel_pitch_um": pixel_pitch_um,
                    "W_px": W_px,
                    "H_px": H_px,
                    "exposure_time_s": exposure_time_s,
                    "quantum_efficiency_oi": quantum_efficiency_oi,
                    "quantum_efficiency_al": quantum_efficiency_al,
                    "filter_transmission_oi": filter_transmission_oi,
                    "filter_transmission_al": filter_transmission_al,
                    "fov_diag_deg": 2*np.degrees(fov_half_angle_rad)
                },
                "wavelengths": {
                    "oi_nm": lambda_oi_nm.value,
                    "al_nm": lambda_al_nm.value
                },
                "background_illumination": {
                    "enabled": self.enable_background.isChecked(),
                    "radiance_W_per_sr": self.background_intensity.value() if self.enable_background.isChecked() else 0,
                    "conversion_method": "Simple scaling (3e18 photons/W/s assumption)",
                    "electrons_per_pixel": background_electrons
                },
                "output_formats": {
                    "numpy_enabled": self.output_numpy.isChecked(),
                    "fits_enabled": self.output_fits.isChecked()
                },
                "stats": {
                    "timestep_data_total": len(timestep_data),
                    "visible_count": visible_count
                }
            }
            
            background_msg = ""
            if self.enable_background.isChecked():
                background_msg = f"\nBackground: {background_electrons:.3f} e⁻/pixel"
            
            param_msg = f"\nSeparate parameters: OI QE={quantum_efficiency_oi:.3f}/Filter={filter_transmission_oi:.3f}, Al QE={quantum_efficiency_al:.3f}/Filter={filter_transmission_al:.3f}"
            
            format_msg = f"\nOutput formats: NumPy={'ENABLED' if self.output_numpy.isChecked() else 'DISABLED'}, FITS={'ENABLED' if self.output_fits.isChecked() else 'DISABLED'}"
            
            QMessageBox.information(self, "Computation Complete", 
                                f"Dual frame computation successful!\n\n"
                                f"Results computed for OI and Combined Al emissions.\n"
                                f"Visible debris: {visible_count} points{background_msg}{param_msg}{format_msg}\n"
                                f"Use the display selectors to view different emissions.\n"
                                f"Use the Save buttons when ready to export files.")
            
            # Enable display selectors and save buttons
            self.display_selector_left.setEnabled(True)
            self.display_selector_right.setEnabled(True)
            self.save_left_btn.setEnabled(True)
            self.save_right_btn.setEnabled(True)
            self.save_both_btn.setEnabled(True)

            # Set displays: OI on left, Combined Al on right
            left_index = self.display_selector_left.findData("OI_emissions_atomic")
            right_index = self.display_selector_right.findData("combined_al")
            
            if left_index >= 0:
                self.display_selector_left.setCurrentIndex(left_index)
                self.update_display("left")
            if right_index >= 0:
                self.display_selector_right.setCurrentIndex(right_index)
                self.update_display("right")

        except Exception as e_main:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error in generate_single_frame_dual: {error_msg}")
            QMessageBox.critical(self, "Frame Generation Error", f"An error occurred: {str(e_main)}")
        finally:
            self.generate_single_frame_btn.setEnabled(True)
            self.generate_single_frame_btn.setText("Compute")
    
    def check_debris_visibility_timeline(self):
        """Check debris visibility timeline"""
        try:
            tracked_debris_id = self.tracked_debris.currentData()
            if tracked_debris_id is None or tracked_debris_id < 0:
                QMessageBox.warning(self, "No Tracked Debris",
                                   "Please select a debris object to check visibility.")
                return
                
            sats = self.sat_tab.get_satellites()
            if not sats:
                QMessageBox.warning(self, "No Satellites", "No satellites defined.")
                return
                
            sat_idx = self.sat_selector.currentData()
            if sat_idx is None or sat_idx >= len(sats):
                sat_idx = 0
                
            satellite_name, sensor = sats[sat_idx]
            
            if self.debris_tab.df.empty:
                QMessageBox.warning(self, "Missing Data", "Please load debris data first.")
                return
                
            tracked_debris_data = self.debris_tab.df[self.debris_tab.df['Assembly_ID'] == tracked_debris_id]
            if tracked_debris_data.empty:
                QMessageBox.warning(self, "Debris Not Found",
                                   f"Debris ID {tracked_debris_id} not found in data.")
                return
                
            # Sensor parameters
            W = self.sensor_width.value()
            H = self.sensor_height.value()
            f_m = self.fl.value() * 1e3  # m to mm
            p_mm = self.pp.value() * 1e-3  # µm to mm
            pix_size_mm = (p_mm, p_mm)
            
            fov_half_angle = self.calculate_fov_angle()
            fov_threshold = np.cos(fov_half_angle)
            
            # Time synchronization
            snap_dt = self.viz_tab.snapshot.dateTime().toPyDateTime()
            analysis_time = Time(snap_dt, scale='utc')
            dt_s = float((analysis_time - sensor.epoch).to_value('s'))
            
            visible_timesteps = []
            
            print(f"\nChecking visibility timeline for Debris ID {tracked_debris_id}")
            print(f"Satellite: {satellite_name}")
            print(f"Total timesteps to check: {len(tracked_debris_data)}")
            
            for _, row in tracked_debris_data.iterrows():
                timestep = float(row['Time'])
                
                # Get debris position using WGS84
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])  
                alt = float(row["Altitude"])  # Already in meters
                P_target_ecef_m = lla_to_ecef_wgs84(lat, lon, alt)
                
                # Propagate satellite and convert ECI to ECEF
                prop_time = timestep + dt_s
                satn = sensor.propagate(prop_time * u.s)
                C_eci_m = satn.rv()[0].to(u.m).value
                C_ecef_m = eci_to_ecef(C_eci_m, analysis_time + timestep * u.s)
                
                # Check Earth occlusion using WGS84
                if not is_visible_wgs84_occlusion(C_ecef_m, P_target_ecef_m):
                    continue
                
                view_vector = P_target_ecef_m - C_ecef_m
                view_distance = np.linalg.norm(view_vector)
                view_direction = view_vector / view_distance
                
                # Camera points at tracked debris
                dot_product = 1.0
                
                # Project to pixel coordinates using coordinate transformation
                pix = convert_ecef_to_pixel(P_target_ecef_m, C_ecef_m, P_target_ecef_m, f_m, pix_size_mm, (W, H))
                
                if pix is not None:
                    u_px, v_px = pix
                    if 0 <= u_px < W and 0 <= v_px < H:
                        Rkm = view_distance / 1000
                        visible_timesteps.append({
                            'time': timestep,
                            'range_km': Rkm,
                            'pixel': (u_px, v_px)
                        })
            
            # Show results
            if visible_timesteps:
                result_text = f"Debris ID {tracked_debris_id} is visible from {satellite_name} at {len(visible_timesteps)} timesteps:\n\n"
                
                for i, vis in enumerate(visible_timesteps[:20]):
                    result_text += f"t={vis['time']:.1f}s: Range={vis['range_km']:.1f}km, Pixel=({vis['pixel'][0]:.0f},{vis['pixel'][1]:.0f})\n"
                
                if len(visible_timesteps) > 20:
                    result_text += f"\n... and {len(visible_timesteps)-20} more timesteps"
                
                # Create dialog to show results
                from PyQt5.QtWidgets import QTextEdit, QDialog, QVBoxLayout, QPushButton
                
                dialog = QDialog(self)
                dialog.setWindowTitle("Debris Visibility Timeline")
                dialog.resize(600, 400)
                
                layout = QVBoxLayout(dialog)
                
                text_edit = QTextEdit()
                text_edit.setPlainText(result_text)
                text_edit.setReadOnly(True)
                layout.addWidget(text_edit)
                
                export_btn = QPushButton("Export to CSV")
                def export_timeline():
                    filepath, _ = QFileDialog.getSaveFileName(
                        dialog, "Save Visibility Timeline",
                        f"visibility_timeline_{satellite_name}_id{tracked_debris_id}.csv",
                        "CSV Files (*.csv)")
                    if filepath:
                        import pandas as pd
                        df = pd.DataFrame(visible_timesteps)
                        df['debris_id'] = tracked_debris_id
                        df['satellite'] = satellite_name
                        df['u_px'] = [p[0] for p in df['pixel']]
                        df['v_px'] = [p[1] for p in df['pixel']]
                        df = df.drop('pixel', axis=1)
                        # Ensure absolute path for saving
                        filepath = os.path.abspath(filepath)
                        df.to_csv(filepath, index=False)
                        QMessageBox.information(dialog, "Exported", f"Timeline saved to {filepath}")
                
                export_btn.clicked.connect(export_timeline)
                layout.addWidget(export_btn)
                
                close_btn = QPushButton("Close")
                close_btn.clicked.connect(dialog.close)
                layout.addWidget(close_btn)
                
                dialog.exec_()
            else:
                QMessageBox.information(self, "No Visibility",
                                       f"Debris ID {tracked_debris_id} is not visible from {satellite_name} at any timestep.")
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error in visibility timeline check: {error_msg}")
            QMessageBox.critical(self, "Error", f"Error checking visibility: {str(e)}")
    
    def update_satellite_list(self):
        """Update the satellite dropdown"""
        current_text = self.sat_selector.currentText()
        self.sat_selector.clear()
        
        satellites = self.sat_tab.get_satellites()
        if satellites:
            for i, (name, _) in enumerate(satellites):
                self.sat_selector.addItem(name, i)
            
            index = self.sat_selector.findText(current_text)
            if index >= 0: self.sat_selector.setCurrentIndex(index)
            elif len(satellites)>0 : self.sat_selector.setCurrentIndex(0)
        else:
            self.sat_selector.addItem("No satellites available", -1)
            
    def update_debris_list(self):
        """Update the tracked debris dropdown"""
        current_id_val = self.tracked_debris.currentData()
        self.tracked_debris.clear()
        self.tracked_debris.addItem("Earth Center (Untracked)", -1)

        if not self.debris_tab.df.empty and 'Assembly_ID' in self.debris_tab.df.columns:
            try:
                debris_ids_series = pd.to_numeric(self.debris_tab.df["Assembly_ID"], errors='coerce').dropna().unique()
                debris_ids = sorted([int(did) for did in debris_ids_series])
                for debris_id_val in debris_ids:
                    self.tracked_debris.addItem(f"Debris ID {debris_id_val}", int(debris_id_val))
                
                if current_id_val is not None:
                    index = self.tracked_debris.findData(current_id_val)
                    if index >= 0: self.tracked_debris.setCurrentIndex(index)
                    elif len(debris_ids) > 0 : self.tracked_debris.setCurrentIndex(1)
            except Exception as e:
                 print(f"Error updating debris list in Photon tab: {e}")

    def update_display(self, side):
        """Update display based on selected emission type"""
        if not self.computed_preview: return
            
        selector = self.display_selector_left if side == "left" else self.display_selector_right
        image_view = self.imv_left if side == "left" else self.imv_right
            
        emission_type_key = selector.currentData()
        current_image = self.result_images_preview.get(emission_type_key)

        if current_image is not None:
            image_view.setImage(current_image.copy(), autoLevels=True, autoRange=True)
        else: 
            image_view.clear()
    
    def save_results(self, which_view):
        """Save computed photon map and result data to files with background illumination metadata, separate OI/Al parameters, and configurable output formats"""
        if not self.computed_preview:
            QMessageBox.warning(self, "No Results", "No results to save. Please generate a frame first.")
            return

        if not hasattr(self, 'last_generation_metadata'):
            QMessageBox.warning(self, "No Metadata", "No generation metadata available. Please generate a frame first.")
            return

        # Check if at least one output format is selected
        if not (self.output_numpy.isChecked() or self.output_fits.isChecked()):
            QMessageBox.warning(self, "No Output Format", "Please select at least one output format (NumPy or FITS).")
            return

        view_map = {"left": (self.display_selector_left, "OI_emissions_atomic"), 
                    "right": (self.display_selector_right, "combined_al")}
        
        views_to_save = [which_view] if which_view != "both" else ["left", "right"]
        saved_any = False

        for side in views_to_save:
            selector, default_emission = view_map[side]
            emission_key = selector.currentData() if selector.currentData() else default_emission
            image_to_save = self.result_images_preview.get(emission_key)
            data_to_save = self.result_data_preview.get(emission_key)

            if image_to_save is None or data_to_save is None or data_to_save.empty:
                QMessageBox.warning(self, "No Data", f"No data for '{selector.currentText() if selector.currentText() else emission_key}' to save.")
                continue

            save_dir = QFileDialog.getExistingDirectory(self, f"Select Output Directory for {side.title()} View", "")
            if not save_dir: 
                continue

            # Convert to absolute path immediately
            save_dir = os.path.abspath(save_dir)
            print(f"Selected save directory (absolute): {save_dir}")
            
            # Verify directory exists and is writable
            if not os.path.exists(save_dir):
                QMessageBox.critical(self, "Directory Error", f"Selected directory does not exist: {save_dir}")
                continue
            
            if not os.access(save_dir, os.W_OK):
                QMessageBox.critical(self, "Permission Error", f"No write permission for directory: {save_dir}")
                continue

            timestamp_str = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
            metadata = self.last_generation_metadata
            sat_name_clean = metadata["satellite_name"].replace(" ", "_")
            
            if emission_key == "OI_emissions_atomic":
                wavelength_str = f"OI_{int(metadata['wavelengths']['oi_nm']):.0f}_nm"
            elif emission_key == "combined_al":
                wavelength_str = f"AlI_combined_{int(metadata['wavelengths']['al_nm']):.0f}_nm"
            else:
                wavelength_str = f"{emission_key}_emission"
            
            background_suffix = "_withBG" if metadata["background_illumination"]["enabled"] else ""
            format_suffix = ""
            if self.output_numpy.isChecked() and self.output_fits.isChecked():
                format_suffix = "_numpy_fits"
            elif self.output_numpy.isChecked():
                format_suffix = "_numpy"
            elif self.output_fits.isChecked():
                format_suffix = "_fits"
            
            base_filename = f"photon_map_{sat_name_clean}_{wavelength_str}_single_frame{background_suffix}{format_suffix}_{timestamp_str}"
            
            try:
                # Save image array in selected formats
                if self.output_numpy.isChecked():
                    npy_path = os.path.join(save_dir, f"{base_filename}.npy")
                    npy_path = os.path.abspath(npy_path)
                    print(f"Saving NumPy array to: {npy_path}")
                    np.save(npy_path, image_to_save)
                    print(f"Successfully saved NumPy array: {npy_path}")
                
                if self.output_fits.isChecked():
                    fits_path = os.path.join(save_dir, f"{base_filename}.fits")
                    fits_path = os.path.abspath(fits_path)
                    print(f"Saving FITS file to: {fits_path}")
                    self._save_fits_file(save_dir, base_filename, image_to_save, emission_key, metadata)
                    print(f"Successfully saved FITS file: {fits_path}")
                
                # Always save results CSV
                csv_path = os.path.join(save_dir, f"{base_filename}_results.csv")
                csv_path = os.path.abspath(csv_path)
                print(f"Saving CSV results to: {csv_path}")
                data_to_save.to_csv(csv_path, index=False)
                print(f"Successfully saved CSV: {csv_path}")
                
                # Create electron map
                H_px, W_px = metadata["sensor_params"]["H_px"], metadata["sensor_params"]["W_px"]
                electron_map_gray = np.zeros((H_px, W_px), dtype=np.float64)
                
                for _, r_map in data_to_save.iterrows():
                    u_idx, v_idx = int(round(r_map['u_px'])), int(round(r_map['v_px']))
                    if 0 <= u_idx < W_px and 0 <= v_idx < H_px:
                        electron_map_gray[v_idx, u_idx] += r_map['Electrons']
                        
                electron_map_path = os.path.join(save_dir, f"{base_filename}_electron_map.csv")
                electron_map_path = os.path.abspath(electron_map_path)
                print(f"Saving electron map to: {electron_map_path}")
                np.savetxt(electron_map_path, electron_map_gray, delimiter=",", fmt='%.3e')
                print(f"Successfully saved electron map: {electron_map_path}")

                # Enhanced metadata with background illumination, separate OI/Al parameters, and output formats
                file_metadata = {
                    "satellite_name": metadata["satellite_name"],
                    "tracked_debris_id": metadata["tracked_debris_id"],
                    "debris_timestep_s": metadata["target_timestep"],
                    "emission_type_selected": "OI (777.3 nm)" if emission_key == "OI_emissions_atomic" else "Combined AlI (396 nm avg)",
                    "emission_data_key": emission_key,
                    "satellite_position_ecef_m": metadata["positions"]["satellite_ecef_m"],
                    "camera_look_at_point_ecef_m": metadata["positions"]["camera_look_at_ecef_m"],
                    "aperture_diameter_m": metadata["sensor_params"]["aperture_diameter_m"],
                    "focal_length_m": metadata["sensor_params"]["focal_length_input_m"],
                    "pixel_pitch_um": metadata["sensor_params"]["pixel_pitch_um"],
                    "sensor_width_px": metadata["sensor_params"]["W_px"],
                    "sensor_height_px": metadata["sensor_params"]["H_px"],
                    "exposure_time_s": metadata["sensor_params"]["exposure_time_s"],
                    "quantum_efficiency_oi": metadata["sensor_params"]["quantum_efficiency_oi"],
                    "quantum_efficiency_al": metadata["sensor_params"]["quantum_efficiency_al"],
                    "filter_transmission_oi": metadata["sensor_params"]["filter_transmission_oi"],
                    "filter_transmission_al": metadata["sensor_params"]["filter_transmission_al"],
                    "effective_fov_diagonal_deg": metadata["sensor_params"]["fov_diag_deg"],
                    "photons_wavelength_source_nm": metadata["wavelengths"]["oi_nm"] if emission_key == "OI_emissions_atomic" else metadata["wavelengths"]["al_nm"],
                    "background_illumination": metadata["background_illumination"],
                    "output_formats": metadata["output_formats"],
                    "generation_timestamp": timestamp_str,
                    "total_debris_at_timestep_in_data": metadata["stats"]["timestep_data_total"],
                    "visible_debris_points_in_frame": metadata["stats"]["visible_count"],
                    "saved_to_directory": save_dir
                }
                
                metadata_path = os.path.join(save_dir, f"{base_filename}_metadata.json")
                metadata_path = os.path.abspath(metadata_path)
                print(f"Saving metadata to: {metadata_path}")
                with open(metadata_path, 'w') as f:
                    json.dump(file_metadata, f, indent=2)
                print(f"Successfully saved metadata: {metadata_path}")
                
                if which_view != "both":
                    background_info = ""
                    if metadata["background_illumination"]["enabled"]:
                        background_info = f"\nBackground: {metadata['background_illumination']['electrons_per_pixel']:.3f} e⁻/pixel"
                    
                    # Get the parameters used for this specific emission type
                    if emission_key == "OI_emissions_atomic":
                        param_info = f"\nOI Parameters: QE={metadata['sensor_params']['quantum_efficiency_oi']:.3f}, Filter={metadata['sensor_params']['filter_transmission_oi']:.3f}"
                    else:
                        param_info = f"\nAl Parameters: QE={metadata['sensor_params']['quantum_efficiency_al']:.3f}, Filter={metadata['sensor_params']['filter_transmission_al']:.3f}"
                    
                    format_info = f"\nOutput formats: NumPy={'SAVED' if self.output_numpy.isChecked() else 'SKIPPED'}, FITS={'SAVED' if self.output_fits.isChecked() else 'SKIPPED'}"
                    
                    QMessageBox.information(self, "File Saved", 
                                          f"{side.title()} view saved successfully!\n\n"
                                          f"Files saved to: {save_dir}\n"
                                          f"Base name: {base_filename}{background_info}{param_info}{format_info}")
                saved_any = True
                
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"Error saving {side} view: {error_msg}")
                QMessageBox.critical(self, "Save Error", f"Error saving {side} view: {str(e)}")
        
        if which_view == "both" and saved_any:
            background_info = ""
            param_info = ""
            format_info = ""
            if hasattr(self, 'last_generation_metadata'):
                metadata = self.last_generation_metadata
                if metadata["background_illumination"]["enabled"]:
                    bg_data = metadata["background_illumination"]
                    background_info = f"\nBackground illumination: {bg_data['electrons_per_pixel']:.3f} e⁻/pixel"
                
                param_info = f"\nSeparate parameters: OI QE={metadata['sensor_params']['quantum_efficiency_oi']:.3f}/Filter={metadata['sensor_params']['filter_transmission_oi']:.3f}, Al QE={metadata['sensor_params']['quantum_efficiency_al']:.3f}/Filter={metadata['sensor_params']['filter_transmission_al']:.3f}"
                
                format_info = f"\nOutput formats: NumPy={'SAVED' if self.output_numpy.isChecked() else 'SKIPPED'}, FITS={'SAVED' if self.output_fits.isChecked() else 'SKIPPED'}"
            
            QMessageBox.information(self, "Files Saved", f"Both views saved successfully to their respective directories.{background_info}{param_info}{format_info}")

    def _save_fits_file(self, save_dir, base_filename, image_array, emission_key, metadata):
        """Save image array as FITS file with proper 2D format"""
        try:
            from astropy.io import fits
            
            # Convert 3D RGB to 2D grayscale for FITS compatibility
            if len(image_array.shape) == 3:
                # Use luminance formula: 0.299*R + 0.587*G + 0.114*B
                grayscale_image = (0.299 * image_array[:, :, 0] + 
                                 0.587 * image_array[:, :, 1] + 
                                 0.114 * image_array[:, :, 2])
                print(f"  Converting 3D RGB {image_array.shape} to 2D grayscale {grayscale_image.shape}")
            else:
                grayscale_image = image_array
            
            # Create minimal FITS header with only ASCII-safe content - no comments to avoid issues
            header = fits.Header()
            header['CREATED'] = Time.now().iso
            header['SOFTWARE'] = 'ESA_Viewer'
            header['UNITS'] = 'electrons'
            header['IMGTYPE'] = '2D_grayscale'  # Indicate this is 2D converted from RGB
            
            # Add sensor parameters - values only, no comments
            header['APERTURE'] = metadata["sensor_params"]["aperture_diameter_m"]
            header['FOCALLEN'] = metadata["sensor_params"]["focal_length_input_m"]
            header['PIXPITCH'] = metadata["sensor_params"]["pixel_pitch_um"]  # Value only, no units in comment
            header['EXPOSURE'] = metadata["sensor_params"]["exposure_time_s"]
            header['SENSW'] = metadata["sensor_params"]["W_px"]
            header['SENSH'] = metadata["sensor_params"]["H_px"]
            header['FOVDIAG'] = metadata["sensor_params"]["fov_diag_deg"]
            
            # Add emission-specific parameters
            if 'OI' in emission_key:
                header['EMISSION'] = 'OI_atomic'
                header['WAVELEN'] = metadata["wavelengths"]["oi_nm"]
                header['QE'] = metadata["sensor_params"]["quantum_efficiency_oi"]
                header['FILTERTX'] = metadata["sensor_params"]["filter_transmission_oi"]
            else:
                header['EMISSION'] = 'AlI_combined'
                header['WAVELEN'] = metadata["wavelengths"]["al_nm"]
                header['QE'] = metadata["sensor_params"]["quantum_efficiency_al"]
                header['FILTERTX'] = metadata["sensor_params"]["filter_transmission_al"]
            
            # Add satellite and debris information
            header['SATELLITE'] = metadata["satellite_name"]
            header['DEBRIS_ID'] = metadata["tracked_debris_id"]
            header['TIMESTEP'] = metadata["target_timestep"]
            
            # Add coordinate information
            header['COORDSYS'] = 'ECI_to_ECEF_WGS84'
            header['SAT_X'] = metadata["positions"]["satellite_ecef_m"][0]
            header['SAT_Y'] = metadata["positions"]["satellite_ecef_m"][1]
            header['SAT_Z'] = metadata["positions"]["satellite_ecef_m"][2]
            header['TGT_X'] = metadata["positions"]["camera_look_at_ecef_m"][0]
            header['TGT_Y'] = metadata["positions"]["camera_look_at_ecef_m"][1]
            header['TGT_Z'] = metadata["positions"]["camera_look_at_ecef_m"][2]
            
            # Add background illumination info
            header['BKGD_EN'] = metadata["background_illumination"]["enabled"]
            if metadata["background_illumination"]["enabled"]:
                header['BKGD_RAD'] = metadata["background_illumination"]["radiance_W_per_sr"]
                header['BKGD_EPX'] = metadata["background_illumination"]["electrons_per_pixel"]
            
            # Add analysis statistics
            header['VIS_CNT'] = metadata["stats"]["visible_count"]
            header['TOT_DEB'] = metadata["stats"]["timestep_data_total"]
            
            # Add output format information
            header['OUT_NPY'] = metadata["output_formats"]["numpy_enabled"]
            header['OUT_FITS'] = metadata["output_formats"]["fits_enabled"]
            
            # Create HDU and save with absolute path - use 2D grayscale image
            hdu = fits.PrimaryHDU(grayscale_image, header=header)
            hdul = fits.HDUList([hdu])
            fits_path = os.path.join(save_dir, f"{base_filename}.fits")
            fits_path = os.path.abspath(fits_path)  # Ensure absolute path
            hdul.writeto(fits_path, overwrite=True)
            hdul.close()
            
            print(f"Saved 2D FITS file: {fits_path}")
            print(f"  Image dimensions: {grayscale_image.shape}")
            print(f"  Data range: {grayscale_image.min():.3e} to {grayscale_image.max():.3e} electrons")
            
        except ImportError:
            print(f"ERROR: astropy not available for FITS output. Cannot save {base_filename}.fits")
            print(f"Install astropy with: pip install astropy")
            QMessageBox.warning(self, "FITS Export Error", 
                              f"Astropy library not available for FITS export.\n"
                              f"Please install astropy to enable FITS output:\n"
                              f"pip install astropy")
        except Exception as e:
            print(f"ERROR saving FITS file {base_filename}.fits: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            QMessageBox.critical(self, "FITS Save Error", f"Error saving FITS file: {str(e)}")
# ----------------------------------------------------------------------
# Visibility Analysis Tab with Coordinate System and UTC Timestamps
# ----------------------------------------------------------------------
class VisibilityAnalysisTab(QWidget):
    """
    Visibility analysis with proper coordinate transformations and WGS84 calculations
    Includes UTC timestamps in tables and export files
    UPDATED: Uses Two-Horizon Access Criterion (rslant <= ac_s + ac_d) for visibility
    """
    def __init__(self, sat_tab, debris_tab, em_tab, photon_tab, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sat_tab = sat_tab
        self.debris_tab = debris_tab
        self.em_tab = em_tab
        self.photon_tab = photon_tab
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Controls
        controls_container = QGroupBox("Analysis Controls")
        controls_main_layout = QHBoxLayout(controls_container)
        
        # Time Controls
        time_group = QGroupBox("Time Settings")
        time_layout = QFormLayout(time_group)
        
        self.snapshot = QDateTimeEdit(QDateTime.currentDateTimeUtc(), self)
        self.snapshot.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        time_layout.addRow("Analysis time:", self.snapshot)
        
        time_step_widget = QWidget()
        time_step_layout = QHBoxLayout(time_step_widget)
        time_step_layout.setContentsMargins(0, 0, 0, 0)
        
        self.step_hour_btn = QPushButton("-1h")
        self.step_hour_btn.clicked.connect(lambda: self.step_time(-3600))
        self.step_hour2_btn = QPushButton("+1h")
        self.step_hour2_btn.clicked.connect(lambda: self.step_time(3600))
        self.step_day_btn = QPushButton("+1d")
        self.step_day_btn.clicked.connect(lambda: self.step_time(86400))
        
        time_step_layout.addWidget(self.step_hour_btn)
        time_step_layout.addWidget(self.step_hour2_btn)
        time_step_layout.addWidget(self.step_day_btn)
        time_layout.addRow("Quick step:", time_step_widget)
        
        controls_main_layout.addWidget(time_group)
        
        # Tracking Controls
        tracking_group = QGroupBox("Tracking & Filtering")
        tracking_layout = QFormLayout(tracking_group)
        
        self.tracked_debris = QComboBox()
        self.tracked_debris.addItem("Earth Center (0,0,0)", -1)
        tracking_layout.addRow("Single-Time Track:", self.tracked_debris)
        
        self.enable_debris_filter = QCheckBox("Filter Debris Objects")
        tracking_layout.addRow(self.enable_debris_filter)
        
        manual_widget = QWidget()
        manual_layout = QHBoxLayout(manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        
        self.manual_debris_input = QLineEdit()
        self.manual_debris_input.setPlaceholderText("e.g., 1,16,25")
        self.manual_debris_input.setEnabled(False)
        self.select_manual_btn = QPushButton("Select")
        self.select_manual_btn.setEnabled(False)
        self.select_manual_btn.clicked.connect(self.select_manual_debris)
        
        manual_layout.addWidget(self.manual_debris_input)
        manual_layout.addWidget(self.select_manual_btn)
        tracking_layout.addRow("Specific IDs:", manual_widget)
        
        # Add info label for duration mode
        duration_info = QLabel("Duration Mode: Camera will track each selected debris separately")
        duration_info.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        duration_info.setWordWrap(True)
        tracking_layout.addRow(duration_info)
        
        controls_main_layout.addWidget(tracking_group)
        
        # Duration Analysis
        duration_group = QGroupBox("Duration Analysis")
        duration_layout = QFormLayout(duration_group)
        
        self.enable_duration = QCheckBox("Enable Duration Analysis")
        duration_layout.addRow(self.enable_duration)
        
        self.duration_seconds = QDoubleSpinBox()
        self.duration_seconds.setRange(1, 86400)
        self.duration_seconds.setValue(558.5)
        self.duration_seconds.setSuffix(" sec")
        self.duration_seconds.setEnabled(False)
        duration_layout.addRow("Duration:", self.duration_seconds)
        
        self.time_step_seconds = QDoubleSpinBox()
        self.time_step_seconds.setRange(0.1, 3600)
        self.time_step_seconds.setValue(0.5)
        self.time_step_seconds.setSuffix(" sec")
        self.time_step_seconds.setEnabled(False)
        duration_layout.addRow("Time Step:", self.time_step_seconds)
        
        self.ignore_timestep_filtering = QCheckBox("Ignore Timestep Filtering")
        self.ignore_timestep_filtering.setEnabled(False)
        self.ignore_timestep_filtering.setToolTip("Use single-time analysis approach for each timestep")
        duration_layout.addRow(self.ignore_timestep_filtering)
        
        self.debris_timestep = QDoubleSpinBox()
        self.debris_timestep.setRange(0, 10000)
        self.debris_timestep.setValue(0.5)
        self.debris_timestep.setSuffix(" sec")
        self.debris_timestep.setDecimals(1)
        self.debris_timestep.setSingleStep(0.5)
        duration_layout.addRow("Single-Time Timestep:", self.debris_timestep)
        
        controls_main_layout.addWidget(duration_group)
        
        # Action Buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.refresh_debris_btn = QPushButton("Refresh Debris")
        self.refresh_debris_btn.clicked.connect(self.update_debris_list)
        
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setStyleSheet("background-color: #ff9900; font-weight: bold;")
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        
        actions_layout.addWidget(self.refresh_debris_btn)
        actions_layout.addWidget(self.analyze_btn)
        actions_layout.addWidget(self.export_btn)
        
        controls_main_layout.addWidget(actions_group)
        layout.addWidget(controls_container)
        
        # Debris selection area
        self.debris_selection_container = QGroupBox("Debris Selection")
        self.debris_selection_container.setVisible(False)
        debris_container_layout = QVBoxLayout(self.debris_selection_container)
        
        quick_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.clear_all_btn = QPushButton("Clear All")
        self.select_all_btn.clicked.connect(self.select_all_debris)
        self.clear_all_btn.clicked.connect(self.clear_all_debris)
        quick_buttons_layout.addWidget(self.select_all_btn)
        quick_buttons_layout.addWidget(self.clear_all_btn)
        quick_buttons_layout.addStretch()
        debris_container_layout.addLayout(quick_buttons_layout)
        
        debris_scroll = QScrollArea()
        debris_scroll.setMaximumHeight(100)
        debris_scroll.setWidgetResizable(True)
        self.debris_list_widget = QWidget()
        self.debris_list_layout = QVBoxLayout(self.debris_list_widget)
        self.debris_list_layout.setContentsMargins(5, 5, 5, 5)
        debris_scroll.setWidget(self.debris_list_widget)
        debris_container_layout.addWidget(debris_scroll)
        
        layout.addWidget(self.debris_selection_container)
        
        self.debris_checkboxes = {}
        
        # Connect filter controls
        self.enable_debris_filter.toggled.connect(self._toggle_debris_filter)
        self.enable_duration.toggled.connect(self._toggle_duration_controls)
        
        # Results tabs
        results_tabs = QTabWidget()
        results_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.sat_view = QTableWidget()
        self.sat_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_tabs.addTab(self.sat_view, "Satellite View")
        
        self.debris_view = QTableWidget()
        self.debris_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_tabs.addTab(self.debris_view, "Debris View")
        
        self.detailed_view = QTableWidget()
        self.detailed_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_tabs.addTab(self.detailed_view, "Visibility Matrix")
        
        self.access_view = QTableWidget()
        self.access_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_tabs.addTab(self.access_view, "Access Periods")
        
        self.stats_view = QTableWidget()
        self.stats_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_tabs.addTab(self.stats_view, "Summary Statistics")
        
        layout.addWidget(results_tabs, 1)
        
        # Status bar
        self.status_label = QLabel("Ready to analyze visibility (Two-Horizon Method)")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.last_results = None
        
        # Initialize
        QTimer.singleShot(100, self.update_debris_list)
        self.debris_tab.debrisDataChanged.connect(self.update_debris_selection_list)
        self.debris_tab.debrisDataChanged.connect(self.update_debris_timestep_range)

    def two_horizon_access_criterion(self, sat_ecef_m, debris_ecef_m):
        """
        Enhanced two-horizon visibility with higher precision
        """
        try:
            # Use double precision throughout
            sat_ecef_m = np.array(sat_ecef_m, dtype=np.float64)
            debris_ecef_m = np.array(debris_ecef_m, dtype=np.float64)
            
            # Convert ECEF to LLA with higher precision
            sat_lat, sat_lon, sat_alt_m = ecef_to_lla_wgs84(sat_ecef_m)
            debris_lat, debris_lon, debris_alt_m = ecef_to_lla_wgs84(debris_ecef_m)
            
            # Use more precise Earth radius calculations
            sat_lat_rad = np.float64(np.radians(sat_lat))
            debris_lat_rad = np.float64(np.radians(debris_lat))
            
            # Higher precision radius calculations
            sin_sat_lat = np.sin(sat_lat_rad)
            sin_debris_lat = np.sin(debris_lat_rad)
            cos_sat_lat = np.cos(sat_lat_rad)
            cos_debris_lat = np.cos(debris_lat_rad)
            
            # More precise N calculations
            N_sat = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_sat_lat**2)
            N_debris = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_debris_lat**2)
            
            # Precise local Earth radius
            re_sat = N_sat * np.sqrt(cos_sat_lat**2 + (1.0 - WGS84_E2)**2 * sin_sat_lat**2)
            re_debris = N_debris * np.sqrt(cos_debris_lat**2 + (1.0 - WGS84_E2)**2 * sin_debris_lat**2)
            
            # Higher precision horizon distances
            ac_s = np.sqrt(2.0 * re_sat * sat_alt_m + sat_alt_m**2)
            ac_d = np.sqrt(2.0 * re_debris * debris_alt_m + debris_alt_m**2)
            
            # Precise slant range
            rslant = np.linalg.norm(debris_ecef_m - sat_ecef_m)
            
            # Use the same comparison as MATLAB (check if MATLAB uses <= or <)
            visible = rslant <= (ac_s + ac_d)  # Try both <= and < to match MATLAB
            
            return visible
            
        except Exception as e:
            print(f"Error in two_horizon_access_criterion: {e}")
            return False
        
    def _toggle_debris_filter(self, checked):
        """Toggle debris filter controls"""
        self.debris_selection_container.setVisible(checked)
        self.manual_debris_input.setEnabled(checked)
        self.select_manual_btn.setEnabled(checked)
        if checked:
            self.update_debris_selection_list()

    def _toggle_duration_controls(self, checked):
        """Toggle duration analysis controls"""
        self.duration_seconds.setEnabled(checked)
        self.time_step_seconds.setEnabled(checked)
        self.ignore_timestep_filtering.setEnabled(checked)

    def update_debris_list(self):
        """Update tracked debris dropdown"""
        current_id = self.tracked_debris.currentData()
        self.tracked_debris.clear()
        
        self.tracked_debris.addItem("Earth Center (0,0,0)", -1)
        
        if not self.debris_tab.df.empty:
            debris_ids = sorted(self.debris_tab.df["Assembly_ID"].unique())
            for debris_id in debris_ids:
                self.tracked_debris.addItem(f"Debris ID {debris_id}", int(debris_id))
                
            if current_id is not None and current_id >= 0:
                index = self.tracked_debris.findData(current_id)
                if index >= 0:
                    self.tracked_debris.setCurrentIndex(index)
        
        self.update_debris_selection_list()
    
    def update_debris_selection_list(self):
        """Update debris selection checkboxes"""
        for checkbox in self.debris_checkboxes.values():
            checkbox.deleteLater()
        self.debris_checkboxes.clear()
        
        while self.debris_list_layout.count():
            child = self.debris_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.debris_tab.df.empty:
            debris_ids = sorted(self.debris_tab.df["Assembly_ID"].unique())
            
            for debris_id in debris_ids:
                checkbox = QCheckBox(f"Debris ID {int(debris_id)}")
                checkbox.setChecked(True)
                self.debris_checkboxes[debris_id] = checkbox
                self.debris_list_layout.addWidget(checkbox)
        
        self.debris_list_layout.addStretch()

    def update_debris_timestep_range(self):
        """Update debris timestep range"""
        if not self.debris_tab.df.empty and 'Time' in self.debris_tab.df.columns:
            try:
                timesteps = sorted(pd.to_numeric(self.debris_tab.df['Time'], errors='coerce').dropna().unique())
                if len(timesteps) > 0:
                    min_t, max_t = min(timesteps), max(timesteps)
                    self.debris_timestep.setRange(min_t, max_t)
                    
                    current_val = self.debris_timestep.value()
                    if not (min_t <= current_val <= max_t):
                        self.debris_timestep.setValue(min_t)
                    
                    print(f"Updated debris timestep range: {min_t} - {max_t}")
                else:
                    self.debris_timestep.setRange(0, 100)
                    self.debris_timestep.setValue(0)
            except Exception as e:
                print(f"Error updating timestep range: {e}")
                self.debris_timestep.setRange(0, 100)
                self.debris_timestep.setValue(0)
        else:
            self.debris_timestep.setRange(0, 100)
            self.debris_timestep.setValue(0)

    def select_manual_debris(self):
        """Select specific debris IDs from manual input"""
        input_text = self.manual_debris_input.text().strip()
        if not input_text:
            QMessageBox.warning(self, "No Input", "Please enter debris IDs (e.g., 1,16,25)")
            return
        
        try:
            requested_ids = []
            for id_str in input_text.split(','):
                id_str = id_str.strip()
                if id_str:
                    debris_id = int(id_str)
                    requested_ids.append(debris_id)
            
            if not requested_ids:
                QMessageBox.warning(self, "Invalid Input", "No valid debris IDs found in input.")
                return
            
            available_ids = list(self.debris_checkboxes.keys())
            found_ids = []
            missing_ids = []
            
            for requested_id in requested_ids:
                if requested_id in available_ids:
                    found_ids.append(requested_id)
                else:
                    missing_ids.append(requested_id)
            
            if not found_ids:
                QMessageBox.warning(self, "IDs Not Found", 
                                   f"None of the requested debris IDs were found in the loaded data.\n"
                                   f"Requested: {requested_ids}\n"
                                   f"Available: {sorted(available_ids)}")
                return
            
            # Clear all selections first
            for checkbox in self.debris_checkboxes.values():
                checkbox.setChecked(False)
            
            # Select only the requested IDs
            for debris_id in found_ids:
                if debris_id in self.debris_checkboxes:
                    self.debris_checkboxes[debris_id].setChecked(True)
            
            message = f"Selected debris IDs: {sorted(found_ids)}"
            if missing_ids:
                message += f"\n\nNot found in data: {sorted(missing_ids)}"
                message += f"\nAvailable IDs: {sorted(available_ids)}"
            
            if self.enable_duration.isChecked():
                message += f"\n\nDuration Mode: Camera will track each selected debris separately"
            
            QMessageBox.information(self, "Selection Complete", message)
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", 
                               f"Invalid debris ID format. Please use numbers separated by commas.\n"
                               f"Example: 1,16,25,100\n\nError: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error parsing debris IDs: {str(e)}")
    
    def select_all_debris(self):
        """Select all debris objects"""
        for checkbox in self.debris_checkboxes.values():
            checkbox.setChecked(True)
    
    def clear_all_debris(self):
        """Clear all debris selections"""
        for checkbox in self.debris_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_debris_ids(self):
        """Get list of selected debris IDs"""
        if not self.enable_debris_filter.isChecked():
            # If debris filtering is disabled, return all available debris
            if not self.debris_tab.df.empty:
                return sorted(self.debris_tab.df["Assembly_ID"].unique())
            return []
        
        # If debris filtering is enabled, return only selected ones
        selected = []
        for debris_id, checkbox in self.debris_checkboxes.items():
            if checkbox.isChecked():
                selected.append(debris_id)
        return sorted(selected)
    
    def step_time(self, seconds):
        """Adjust analysis time"""
        current_time = self.snapshot.dateTime()
        new_time = current_time.addSecs(seconds)
        self.snapshot.setDateTime(new_time)
        
    def run_analysis(self):
        """Run visibility analysis with Two-Horizon Access Criterion"""
        try:
            self.status_label.setText("Running visibility analysis (Two-Horizon Method)...")
            self.export_btn.setEnabled(False)
            QApplication.processEvents()
            
            self.update_debris_timestep_range()
            
            sats = self.sat_tab.get_satellites()
            
            if not sats or self.debris_tab.df.empty or self.em_tab.df.empty:
                QMessageBox.warning(self, "Missing Data", 
                            "Please ensure satellites, debris, and emissions data are loaded.")
                self.status_label.setText("Ready to analyze visibility (Two-Horizon Method)")
                return
            
            selected_debris_ids = self.get_selected_debris_ids()
            if not selected_debris_ids:
                QMessageBox.warning(self, "No Debris Selected", 
                            "Please select at least one debris object to analyze.")
                self.status_label.setText("Ready to analyze visibility (Two-Horizon Method)")
                return
            
            tracked_debris_id = self.tracked_debris.currentData()
            
            print(f"\n=== TWO-HORIZON VISIBILITY ANALYSIS SETUP ===")
            print(f"Selected debris IDs: {selected_debris_ids}")
            print(f"Analysis mode: {'Duration' if self.enable_duration.isChecked() else 'Single time'}")
            print(f"Visibility method: Two-Horizon Access Criterion (rslant <= ac_s + ac_d)")
            
            if self.enable_duration.isChecked():
                print(f"Duration mode: Camera will track each selected debris separately")
                print(f"Ignore timestep filtering: {self.ignore_timestep_filtering.isChecked()}")
            else:
                print(f"Single-time tracked debris ID: {tracked_debris_id}")
                print(f"Single-time debris timestep: {self.debris_timestep.value()}")
            
            try:
                merged_data = pd.merge(self.debris_tab.df, self.em_tab.df, 
                                on=["Iteration", "Assembly_ID"], how="inner")
                if merged_data.empty:
                    QMessageBox.warning(self, "No matching data", 
                                    "No matching rows between debris and emissions data.")
                    return
                
                print(f"Merged data shape: {merged_data.shape}")
                
                # Filter to only selected debris
                merged_data = merged_data[merged_data["Assembly_ID"].isin(selected_debris_ids)]
                if merged_data.empty:
                    QMessageBox.warning(self, "No matching debris", 
                                    "No data found for selected debris objects.")
                    return
                
                print(f"Filtered data shape: {merged_data.shape}")
                print(f"Time range in data: {merged_data['Time'].min()} - {merged_data['Time'].max()}")
                
            except Exception as e:
                QMessageBox.critical(self, "Merge Error", f"Could not merge data: {str(e)}")
                return
            
            # Get sensor parameters from photon tab
            focal_mm = self.photon_tab.fl.value() * 1e3  # m to mm
            pixel_mm = self.photon_tab.pp.value() * 1e-3  # µm to mm
            pixel_size = (pixel_mm, pixel_mm)
            resolution = (self.photon_tab.sensor_width.value(), 
                         self.photon_tab.sensor_height.value())
            W, H = resolution
            
            focal_length_m = self.photon_tab.fl.value()
            pixel_size_um = self.photon_tab.pp.value()
            sensor_width_px = self.photon_tab.sensor_width.value()
            sensor_height_px = self.photon_tab.sensor_height.value()
            
            pixel_size_m = pixel_size_um * 1e-6
            sensor_width_m = sensor_width_px * pixel_size_m
            sensor_height_m = sensor_height_px * pixel_size_m
            sensor_diagonal_m = np.sqrt(sensor_width_m**2 + sensor_height_m**2)
            
            fov_half_angle = np.arctan(sensor_diagonal_m / (2 * focal_length_m))
            fov_threshold = np.cos(fov_half_angle)
            
            h_fov_deg = 2 * np.degrees(np.arctan(sensor_width_m / (2 * focal_length_m)))
            v_fov_deg = 2 * np.degrees(np.arctan(sensor_height_m / (2 * focal_length_m)))
            diag_fov_deg = 2 * np.degrees(fov_half_angle)
            
            print(f"\n=== SENSOR PARAMETERS ===")
            print(f"Focal length: {focal_length_m*1000:.1f}mm")
            print(f"Pixel size: {pixel_size_um}μm")
            print(f"Sensor: {sensor_width_px}×{sensor_height_px}px")
            print(f"FOV: H={h_fov_deg:.2f}°, V={v_fov_deg:.2f}°, Diag={diag_fov_deg:.2f}°")
            print(f"FOV threshold (cos): {fov_threshold:.4f}")
            
            if self.enable_duration.isChecked():
                results = self._run_duration_analysis_multi_track(sats, merged_data, selected_debris_ids, 
                                                                 fov_threshold, focal_mm, pixel_size, resolution)
            else:
                analysis_time = Time(self.snapshot.dateTime().toPyDateTime(), scale='utc')
                results = self._run_single_time_analysis(sats, merged_data, tracked_debris_id,
                                                       fov_threshold, focal_mm, pixel_size, 
                                                       resolution, analysis_time)
            
            results['sensor_params'] = {
                'focal_length_mm': focal_mm,
                'pixel_size_um': pixel_size_um,
                'sensor_width_px': W,
                'sensor_height_px': H,
                'h_fov_deg': h_fov_deg,
                'v_fov_deg': v_fov_deg,
                'diag_fov_deg': diag_fov_deg
            }
            results['selected_debris_ids'] = selected_debris_ids
            results['debris_filter_enabled'] = self.enable_debris_filter.isChecked()
            results['visibility_method'] = 'two_horizon_access_criterion'
            
            self.last_results = results
            self._update_visibility_displays(results)
            self.export_btn.setEnabled(True)
            
            print(f"\n=== TWO-HORIZON ANALYSIS COMPLETE ===")
            print(f"Analyzed {len(selected_debris_ids)} debris objects: {selected_debris_ids}")
            print(f"Method: Two-Horizon Access Criterion")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error: {error_msg}")
            QMessageBox.critical(self, "Error", f"Analysis error: {str(e)}")
            self.status_label.setText("Analysis failed")
    
    def _run_single_time_analysis(self, sats, merged_data, tracked_debris_id, fov_threshold,
                                 focal_mm, pixel_size, resolution, analysis_time):
        """Single time analysis with Two-Horizon method"""
        
        print(f"\n=== SINGLE TIME ANALYSIS (TWO-HORIZON) ===")
        print(f"Analysis time: {analysis_time.iso}")
        
        step_results = self._analyze_timestep_unified(
            sats, merged_data, tracked_debris_id, fov_threshold,
            focal_mm, pixel_size, resolution, analysis_time, None
        )
        
        step_results['analysis_type'] = 'single_time'
        return step_results
    
    def _run_duration_analysis_multi_track(self, sats, merged_data, selected_debris_ids, fov_threshold, 
                                          focal_mm, pixel_size, resolution):
        """
        Duration analysis with Two-Horizon method that tracks each selected debris separately
        """
        
        start_time = Time(self.snapshot.dateTime().toPyDateTime(), scale='utc')
        duration_sec = self.duration_seconds.value()
        step_sec = self.time_step_seconds.value()
        ignore_timestep_filtering = self.ignore_timestep_filtering.isChecked()
        
        # Use actual unique timesteps from the data
        available_debris_times = np.sort(merged_data['Time'].unique())
        print(f"\n=== MULTI-TRACK DURATION ANALYSIS (TWO-HORIZON) ===")
        print(f"Start: {start_time.iso}")
        print(f"Duration: {duration_sec} sec, Step: {step_sec} sec")
        print(f"IGNORE TIMESTEP FILTERING: {ignore_timestep_filtering}")
        print(f"TRACKING TARGETS: {selected_debris_ids}")
        print(f"Available debris timesteps (first 10): {available_debris_times[:10]}")
        print(f"Total available timesteps: {len(available_debris_times)}")
        print(f"Visibility method: Two-Horizon Access Criterion")
        
        # Filter to timesteps within duration window
        max_time = duration_sec
        valid_debris_times = [t for t in available_debris_times if 0 <= t <= max_time]
        
        if not valid_debris_times:
            print(f"ERROR: No debris timesteps found within duration window [0, {max_time}]")
            valid_debris_times = available_debris_times[:min(10, len(available_debris_times))]
            print(f"Using first {len(valid_debris_times)} timesteps as fallback")
        
        time_steps = valid_debris_times
        print(f"Using {len(time_steps)} actual debris timesteps for analysis")
        
        # Initialize combined results structure
        combined_results = {
            'analysis_type': 'duration_multi_track',
            'start_time': start_time,
            'duration_sec': duration_sec,
            'step_sec': step_sec,
            'ignore_timestep_filtering': ignore_timestep_filtering,
            'selected_debris_ids': selected_debris_ids,
            'tracking_targets': selected_debris_ids,
            'actual_time_steps': time_steps,
            'time_steps': time_steps,
            'visibility_timeline': {},
            'access_periods': {},
            'single_time_snapshots': [],
            'tracking_scenarios': {},
            'visibility_method': 'two_horizon_access_criterion'
        }
        
        # Initialize timeline for ALL satellites and ALL debris
        for sat_name, sat in sats:
            combined_results['visibility_timeline'][sat_name] = {}
            combined_results['access_periods'][sat_name] = {}
            for debris_id in selected_debris_ids:
                combined_results['visibility_timeline'][sat_name][debris_id] = []
                combined_results['access_periods'][sat_name][debris_id] = []
        
        # Run analysis for each tracking target
        total_scenarios = len(selected_debris_ids)
        total_steps_per_scenario = len(time_steps)
        
        for scenario_idx, tracking_target_id in enumerate(selected_debris_ids):
            print(f"\n=== TRACKING SCENARIO {scenario_idx + 1}/{total_scenarios}: DEBRIS ID {tracking_target_id} (TWO-HORIZON) ===")
            
            # Store results for this tracking scenario
            scenario_results = {
                'tracking_target': tracking_target_id,
                'visibility_timeline': {},
                'access_periods': {},
                'snapshots': []
            }
            
            for sat_name, sat in sats:
                scenario_results['visibility_timeline'][sat_name] = {}
                scenario_results['access_periods'][sat_name] = {}
                for debris_id in selected_debris_ids:
                    scenario_results['visibility_timeline'][sat_name][debris_id] = []
                    scenario_results['access_periods'][sat_name][debris_id] = []
            
            for step_idx, debris_timestep in enumerate(time_steps):
                overall_progress = ((scenario_idx * total_steps_per_scenario + step_idx) / 
                                  (total_scenarios * total_steps_per_scenario)) * 100
                
                if step_idx % max(1, total_steps_per_scenario // 10) == 0:
                    mode_text = "coords" if ignore_timestep_filtering else "timestep-filtered"
                    self.status_label.setText(f"Multi-track analysis (Two-Horizon {mode_text}): "
                                            f"Tracking debris {tracking_target_id} "
                                            f"({overall_progress:.1f}% total)")
                    QApplication.processEvents()
            
                satellite_analysis_time = start_time + debris_timestep * u.s
                
                print(f"--- Track {tracking_target_id}, step {step_idx+1}/{total_steps_per_scenario}: "
                      f"debris_t={debris_timestep:.1f}s, sat_t={satellite_analysis_time.iso} ---")
                
                step_results = self._analyze_timestep_unified(
                    sats, merged_data, tracking_target_id, fov_threshold, 
                    focal_mm, pixel_size, resolution, satellite_analysis_time, 
                    debris_timestep if not ignore_timestep_filtering else None
                )
                step_results['time_offset'] = debris_timestep
                step_results['tracking_target'] = tracking_target_id
                
                scenario_results['snapshots'].append(step_results)
                
                # Update timeline for this scenario
                for sat_name in step_results['visibility_matrix']:
                    for debris_id in selected_debris_ids:
                        range_km = step_results['visibility_matrix'][sat_name].get(debris_id, 0)
                        is_visible = range_km > 0
                        
                        visibility_point = {
                            'time_offset': debris_timestep,
                            'time_iso': satellite_analysis_time.iso,
                            'visible': is_visible,
                            'range_km': range_km if is_visible else 0,
                            'tracking_target': tracking_target_id
                        }
                        
                        scenario_results['visibility_timeline'][sat_name][debris_id].append(visibility_point)
                        
                        # Also add to combined results for overall view
                        combined_results['visibility_timeline'][sat_name][debris_id].append(visibility_point)
            
            # Calculate access periods for this scenario
            print(f"\n=== CALCULATING ACCESS PERIODS FOR TRACKING {tracking_target_id} (TWO-HORIZON) ===")
            for sat_name in scenario_results['visibility_timeline']:
                print(f"Satellite {sat_name}:")
                for debris_id in selected_debris_ids:
                    timeline = scenario_results['visibility_timeline'][sat_name][debris_id]
                    
                    # Sort timeline by time offset
                    timeline.sort(key=lambda x: x['time_offset'])
                    
                    access_periods = []
                    current_period_start = None
                    
                    for point in timeline:
                        if point['visible'] and current_period_start is None:
                            # Start of new access period
                            current_period_start = point['time_offset']
                        elif not point['visible'] and current_period_start is not None:
                            # End of current access period
                            access_periods.append((current_period_start, point['time_offset']))
                            current_period_start = None
                    
                    # Handle case where visibility continues to the end
                    if current_period_start is not None:
                        access_periods.append((current_period_start, time_steps[-1]))
                    
                    scenario_results['access_periods'][sat_name][debris_id] = access_periods
                    
                    if access_periods:
                        total_access_time = sum(end - start for start, end in access_periods)
                        print(f"  Debris {debris_id}: {len(access_periods)} periods, {total_access_time:.1f}s total")
                        for i, (start, end) in enumerate(access_periods):
                            print(f"    Period {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
                    else:
                        print(f"  Debris {debris_id}: No access periods")
            
            combined_results['tracking_scenarios'][tracking_target_id] = scenario_results
            
            # Add all snapshots to combined results
            combined_results['single_time_snapshots'].extend(scenario_results['snapshots'])
        
        # Calculate combined access periods (merge from all tracking scenarios)
        print(f"\n=== CALCULATING COMBINED ACCESS PERIODS (TWO-HORIZON) ===")
        for sat_name in combined_results['visibility_timeline']:
            for debris_id in selected_debris_ids:
                timeline = combined_results['visibility_timeline'][sat_name][debris_id]
                
                # Consolidate timeline - if ANY tracking scenario sees debris at a timestep, mark as visible
                consolidated_timeline = {}
                for point in timeline:
                    time_offset = point['time_offset']
                    if time_offset not in consolidated_timeline:
                        consolidated_timeline[time_offset] = {
                            'time_offset': time_offset,
                            'time_iso': point['time_iso'],
                            'visible': False,
                            'range_km': 0,
                            'tracking_targets': []
                        }
                    
                    if point['visible']:
                        consolidated_timeline[time_offset]['visible'] = True
                        consolidated_timeline[time_offset]['range_km'] = max(
                            consolidated_timeline[time_offset]['range_km'], 
                            point['range_km']
                        )
                    
                    consolidated_timeline[time_offset]['tracking_targets'].append(
                        point.get('tracking_target', 'Unknown')
                    )
                
                # Sort consolidated timeline by time
                sorted_timeline = sorted(consolidated_timeline.values(), key=lambda x: x['time_offset'])
                
                # Calculate access periods from consolidated timeline
                access_periods = []
                current_period_start = None
                
                for point in sorted_timeline:
                    if point['visible'] and current_period_start is None:
                        current_period_start = point['time_offset']
                    elif not point['visible'] and current_period_start is not None:
                        access_periods.append((current_period_start, point['time_offset']))
                        current_period_start = None
                
                # Handle case where visibility continues to the end
                if current_period_start is not None:
                    access_periods.append((current_period_start, time_steps[-1]))
                
                combined_results['access_periods'][sat_name][debris_id] = access_periods
                
                # Update the combined timeline with consolidated data
                combined_results['visibility_timeline'][sat_name][debris_id] = sorted_timeline
                
                # Debug output for this debris
                if access_periods:
                    total_access_time = sum(end - start for start, end in access_periods)
                    print(f"  Sat {sat_name}, Debris {debris_id}: {len(access_periods)} consolidated periods, {total_access_time:.1f}s total")
                    for i, (start, end) in enumerate(access_periods):
                        print(f"    Period {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s)")
                else:
                    print(f"  Sat {sat_name}, Debris {debris_id}: No access periods")
        
        combined_results['summary'] = self._calculate_duration_summary_multi_track(combined_results)
        
        # Calculate total consolidated access periods for summary
        total_consolidated_periods = 0
        for sat_name in combined_results['access_periods']:
            for debris_id in selected_debris_ids:
                total_consolidated_periods += len(combined_results['access_periods'][sat_name][debris_id])
        
        print(f"\n=== CONSOLIDATION SUMMARY (TWO-HORIZON) ===")
        print(f"Total tracking scenarios: {total_scenarios}")
        print(f"Total consolidated access periods: {total_consolidated_periods}")
        print(f"Time steps per scenario: {total_steps_per_scenario}")
        
        selected_count = len(selected_debris_ids)
        mode_text = "coords" if ignore_timestep_filtering else "timestep-filtered"
        
        self.status_label.setText(f"Multi-track duration analysis (Two-Horizon {mode_text}) complete: "
                                f"{total_scenarios} tracking targets, {total_steps_per_scenario} time steps each, "
                                f"{selected_count} debris analyzed, {total_consolidated_periods} access periods found")
        
        return combined_results

    def _analyze_timestep_unified(self, sats, merged_data, tracked_debris_id, fov_threshold,
                                focal_mm, pixel_size, resolution, analysis_time, debris_timestep):
        """
        Unified analysis method with Two-Horizon Access Criterion
        - Camera points at tracked_debris_id (or Earth center if -1)
        - ALL debris in merged_data are checked for visibility using Two-Horizon method
        """
        
        W, H = resolution
        visibility_matrix = {}
        visibility_reasons = {}
        pixel_positions = {}
        
        time_tolerance = 0.01
        
        stats = {
            "total_debris": 0,
            "beyond_horizon": 0,
            "outside_fov": 0,
            "behind_camera": 0,
            "outside_sensor": 0,
            "visible": 0
        }
        
        # Get all debris IDs to analyze from merged_data
        all_debris_ids = sorted(merged_data['Assembly_ID'].unique())
        
        debug_info = f"Tracking {tracked_debris_id}, timestep {debris_timestep if debris_timestep is not None else 'N/A'}"
        print(f"\n=== UNIFIED ANALYSIS (TWO-HORIZON): {debug_info} ===")
        print(f"Analysis time: {analysis_time.iso}")
        print(f"Analyzing debris IDs: {all_debris_ids}")
        
        for sat_name, sat in sats:
            print(f"\n--- Analyzing satellite: {sat_name} (TWO-HORIZON) ---")
            
            # Get satellite position in ECI, convert to ECEF
            dt = (analysis_time - sat.epoch)
            sat_propagated = sat.propagate(dt)
            r_sat_eci_m = sat_propagated.rv()[0].to(u.m).value
            r_sat_ecef_m = eci_to_ecef(r_sat_eci_m, analysis_time)
            
            # Calculate satellite horizon for debug info
            re = 6371000.0
            sat_distance = np.linalg.norm(r_sat_ecef_m)
            h_s = sat_distance - re
            ac_s = np.sqrt((re + h_s)**2 - re**2)
            
            print(f"Satellite position (ECEF): [{r_sat_ecef_m[0]/1000:.1f}, {r_sat_ecef_m[1]/1000:.1f}, {r_sat_ecef_m[2]/1000:.1f}] km")
            print(f"Satellite altitude: {h_s/1000:.1f} km, horizon distance: {ac_s/1000:.1f} km")
            
            # Determine camera pointing direction ONCE
            if tracked_debris_id >= 0:
                # Camera points at tracked debris
                if debris_timestep is not None:
                    # Use timestep filtering
                    tracked_rows = merged_data[
                        (merged_data["Assembly_ID"] == tracked_debris_id) & 
                        (np.abs(merged_data['Time'] - debris_timestep) <= time_tolerance)
                    ]
                else:
                    # No timestep filtering, just find the tracked debris
                    tracked_rows = merged_data[merged_data["Assembly_ID"] == tracked_debris_id]
                
                if not tracked_rows.empty:
                    row = tracked_rows.iloc[0]
                    lat_deg = float(row["Latitude"])
                    lon_deg = float(row["Longitude"])
                    alt_m = float(row["Altitude"])
                    P_camera_target = lla_to_ecef_wgs84(lat_deg, lon_deg, alt_m)
                    print(f"Camera pointing at tracked debris ID {tracked_debris_id}")
                else:
                    # Fallback to Earth center
                    P_camera_target = np.array([0.0, 0.0, 0.0])
                    print(f"Tracked debris {tracked_debris_id} not found, camera pointing at Earth center")
            else:
                # Camera points at Earth center
                P_camera_target = np.array([0.0, 0.0, 0.0])
                print(f"Camera pointing at Earth center")
            
            # Calculate camera boresight (FIXED for all debris)
            camera_vector = P_camera_target - r_sat_ecef_m
            camera_dist = np.linalg.norm(camera_vector)
            camera_boresight = camera_vector / camera_dist if camera_dist > 0 else np.array([0, 0, -1])
            
            visibility_matrix[sat_name] = {}
            visibility_reasons[sat_name] = {}
            pixel_positions[sat_name] = {}
            
            # Now check ALL debris against this FIXED camera orientation using Two-Horizon method
            for debris_id in all_debris_ids:
                if debris_id != debris_id:  # Skip NaN
                    continue
                
                stats["total_debris"] += 1
                
                # Get debris data for this timestep/time
                if debris_timestep is not None:
                    # Use timestep filtering
                    current_debris_rows = merged_data[
                        (merged_data["Assembly_ID"] == debris_id) & 
                        (np.abs(merged_data['Time'] - debris_timestep) <= time_tolerance)
                    ]
                else:
                    # No timestep filtering, just find the debris
                    current_debris_rows = merged_data[merged_data["Assembly_ID"] == debris_id]

                if current_debris_rows.empty:
                    visibility_matrix[sat_name][debris_id] = 0
                    visibility_reasons[sat_name][debris_id] = "No data at timestep"
                    continue  
                
                row = current_debris_rows.iloc[0]
                
                # Extract position and convert LLA to ECEF using WGS84
                lat_deg = float(row["Latitude"])
                lon_deg = float(row["Longitude"])
                alt_m = float(row["Altitude"])  # Already in meters
                
                P_obj_ecef_m = lla_to_ecef_wgs84(lat_deg, lon_deg, alt_m)
                
                # Debug for tracked debris or first few
                is_tracked = (debris_id == tracked_debris_id)
                debug_this = (len(visibility_matrix[sat_name]) <= 3) or is_tracked
                
                if debug_this:
                    debris_distance = np.linalg.norm(P_obj_ecef_m)
                    h_d = debris_distance - re
                    ac_d = np.sqrt((re + h_d)**2 - re**2)
                    print(f"\n  Debris ID {debris_id} {'(TRACKED)' if is_tracked else ''}:")
                    print(f"    ECEF: [{P_obj_ecef_m[0]/1000:.1f}, {P_obj_ecef_m[1]/1000:.1f}, {P_obj_ecef_m[2]/1000:.1f}] km")
                    print(f"    Altitude: {h_d/1000:.1f} km, horizon: {ac_d/1000:.1f} km")
                
                # 1. Two-Horizon Access Criterion check
                if not self.two_horizon_access_criterion(r_sat_ecef_m, P_obj_ecef_m):
                    visibility_matrix[sat_name][debris_id] = 0
                    visibility_reasons[sat_name][debris_id] = "Beyond horizon (Two-Horizon)"
                    stats["beyond_horizon"] += 1
                    if debug_this:
                        rslant = np.linalg.norm(P_obj_ecef_m - r_sat_ecef_m)
                        print(f"    ✗ Beyond horizon: rslant={rslant/1000:.1f}km > ac_total={(ac_s+ac_d)/1000:.1f}km")
                    continue
                
                # Check if debris is in FOV of the FIXED camera
                debris_vector = P_obj_ecef_m - r_sat_ecef_m
                debris_dist = np.linalg.norm(debris_vector)
                debris_direction = debris_vector / debris_dist if debris_dist > 0 else np.array([0, 0, 1])
                
                # Calculate angle between camera boresight and debris direction
                dot_product = np.dot(camera_boresight, debris_direction)
                
                if debug_this:
                    angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                    print(f"    Range: {debris_dist/1000:.1f} km")
                    print(f"    Angle from boresight: {angle_deg:.2f}°")
                
                # 2. FOV check
                if dot_product < fov_threshold:
                    visibility_matrix[sat_name][debris_id] = 0
                    visibility_reasons[sat_name][debris_id] = "Outside FOV"
                    stats["outside_fov"] += 1
                    if debug_this:
                        print(f"    ✗ Outside FOV")
                    continue
                
                # 3. Camera projection check using FIXED camera target
                pixel = convert_ecef_to_pixel(P_obj_ecef_m, r_sat_ecef_m, P_camera_target, focal_mm, pixel_size, resolution)
                if pixel is None:
                    visibility_matrix[sat_name][debris_id] = 0
                    visibility_reasons[sat_name][debris_id] = "Behind camera"
                    stats["behind_camera"] += 1
                    if debug_this:
                        print(f"    ✗ Behind camera plane")
                    continue
                
                # 4. Sensor bounds check
                u_px, v_px = pixel
                if not (0 <= u_px < W and 0 <= v_px < H):
                    visibility_matrix[sat_name][debris_id] = 0
                    visibility_reasons[sat_name][debris_id] = "Outside sensor"
                    stats["outside_sensor"] += 1
                    if debug_this:
                        print(f"    ✗ Outside sensor bounds: pixel ({u_px:.1f}, {v_px:.1f})")
                    continue
                
                # 5. VISIBLE!
                range_km = debris_dist / 1000.0
                visibility_matrix[sat_name][debris_id] = range_km
                visibility_reasons[sat_name][debris_id] = "Visible"
                pixel_positions[sat_name][debris_id] = (u_px, v_px)
                stats["visible"] += 1
                
                if debug_this:
                    print(f"    ✓ VISIBLE: Range={range_km:.1f}km, Pixel=({u_px:.1f}, {v_px:.1f})")
        
        print(f"\n=== UNIFIED ANALYSIS SUMMARY (TWO-HORIZON): {debug_info} ===")
        print(f"Total analysis checks: {stats['total_debris']}")
        print(f"  VISIBLE: {stats['visible']}")
        print(f"  Beyond horizon (Two-Horizon): {stats['beyond_horizon']}")
        print(f"  Outside FOV: {stats['outside_fov']}")
        print(f"  Behind camera: {stats['behind_camera']}")
        print(f"  Outside sensor: {stats['outside_sensor']}")
        if stats['total_debris'] > 0:
            print(f"Visibility rate: {(stats['visible']/stats['total_debris']*100):.1f}%")
        
        return {
            'visibility_matrix': visibility_matrix,
            'visibility_reasons': visibility_reasons,
            'pixel_positions': pixel_positions,
            'analysis_time': analysis_time,
            'time_offset': debris_timestep if debris_timestep is not None else 0,
            'tracked_debris_id': tracked_debris_id,
            'selected_debris_ids': all_debris_ids,
            'stats': stats,
            'visibility_method': 'two_horizon_access_criterion'
        }

    def _calculate_duration_summary_multi_track(self, time_series_results):
        """Calculate summary statistics for multi-track duration analysis"""
        
        summary = {
            'total_time_steps': len(time_series_results['time_steps']),
            'duration_sec': time_series_results['duration_sec'],
            'step_sec': time_series_results['step_sec'],
            'ignore_timestep_filtering': time_series_results.get('ignore_timestep_filtering', False),
            'tracking_targets': time_series_results.get('tracking_targets', []),
            'total_tracking_scenarios': len(time_series_results.get('tracking_scenarios', {})),
            'visibility_method': time_series_results.get('visibility_method', 'two_horizon_access_criterion'),
            'access_summary': {}
        }
        
        for sat_name in time_series_results['access_periods']:
            summary['access_summary'][sat_name] = {}
            for debris_id in time_series_results['access_periods'][sat_name]:
                access_periods = time_series_results['access_periods'][sat_name][debris_id]
                
                total_access_time = sum(end - start for start, end in access_periods)
                access_percentage = (total_access_time / summary['duration_sec']) * 100 if summary['duration_sec'] > 0 else 0
                
                summary['access_summary'][sat_name][debris_id] = {
                    'num_access_periods': len(access_periods),
                    'total_access_time_sec': total_access_time,
                    'access_percentage': access_percentage,
                    'access_periods': access_periods
                }
        
        return summary

    def _calculate_duration_summary(self, time_series_results):
        """Calculate summary statistics for duration analysis"""
        
        summary = {
            'total_time_steps': len(time_series_results['time_steps']),
            'duration_sec': time_series_results['duration_sec'],
            'step_sec': time_series_results['step_sec'],
            'ignore_timestep_filtering': time_series_results.get('ignore_timestep_filtering', False),
            'visibility_method': time_series_results.get('visibility_method', 'two_horizon_access_criterion'),
            'access_summary': {}
        }
        
        for sat_name in time_series_results['access_periods']:
            summary['access_summary'][sat_name] = {}
            for debris_id in time_series_results['access_periods'][sat_name]:
                access_periods = time_series_results['access_periods'][sat_name][debris_id]
                
                total_access_time = sum(end - start for start, end in access_periods)
                access_percentage = (total_access_time / summary['duration_sec']) * 100 if summary['duration_sec'] > 0 else 0
                
                summary['access_summary'][sat_name][debris_id] = {
                    'num_access_periods': len(access_periods),
                    'total_access_time_sec': total_access_time,
                    'access_percentage': access_percentage,
                    'access_periods': access_periods
                }
        
        return summary
    
    def _update_visibility_displays(self, results):
        """Update all display tables with analysis results"""
        
        if results['analysis_type'] in ['duration', 'duration_multi_track']:
            self._update_duration_displays(results)
        else:
            self._update_single_time_displays(results)
    
    def _update_duration_displays(self, results):
        """Update displays for duration analysis results with UTC timestamps"""
        
        # Update Access Periods view with UTC timestamps
        self.access_view.clear()
        access_data = []
        
        for sat_name in results['access_periods']:
            for debris_id in results['access_periods'][sat_name]:
                periods = results['access_periods'][sat_name][debris_id]
                summary = results['summary']['access_summary'][sat_name][debris_id]
                
                for i, (start, end) in enumerate(periods):
                    # Convert to UTC timestamps
                    start_utc_time = results['start_time'] + start * u.s
                    end_utc_time = results['start_time'] + end * u.s
                    
                    # For multi-track analysis, show which tracking targets contributed
                    tracking_info = ""
                    if results['analysis_type'] == 'duration_multi_track':
                        # Get tracking targets that were active during this period
                        active_targets = set()
                        for scenario_id, scenario_data in results.get('tracking_scenarios', {}).items():
                            scenario_periods = scenario_data.get('access_periods', {}).get(sat_name, {}).get(debris_id, [])
                            for s_start, s_end in scenario_periods:
                                # Check if this scenario period overlaps with the combined period
                                if not (end <= s_start or start >= s_end):  # Periods overlap
                                    active_targets.add(scenario_id)
                        
                        if active_targets:
                            tracking_info = f"Track: {','.join(map(str, sorted(active_targets)))}"
                    
                    access_data.append({
                        'Satellite': sat_name,
                        'Debris_ID': debris_id,
                        'Period': i + 1,
                        'Start_Sec': start,
                        'End_Sec': end,
                        'Start_UTC': start_utc_time.utc.isot,
                        'End_UTC': end_utc_time.utc.isot,
                        'Duration_Sec': end - start,
                        'Total_Access_Sec': summary['total_access_time_sec'],
                        'Access_%': summary['access_percentage'],
                        'Tracking_Info': tracking_info
                    })
        
        if access_data:
            # Updated column headers to include tracking info (11 columns total)
            headers = [
                "Satellite", "Debris ID", "Period #", "Start (sec)", "End (sec)", 
                "Start UTC", "End UTC", "Duration (sec)", "Total Access (sec)", "Access %"
            ]
            
            if results['analysis_type'] == 'duration_multi_track':
                headers.append("Tracking Scenarios")
            
            self.access_view.setColumnCount(len(headers))
            self.access_view.setHorizontalHeaderLabels(headers)
            self.access_view.setRowCount(len(access_data))
            
            for row, data in enumerate(access_data):
                self.access_view.setItem(row, 0, QTableWidgetItem(data['Satellite']))
                self.access_view.setItem(row, 1, QTableWidgetItem(str(data['Debris_ID'])))
                self.access_view.setItem(row, 2, QTableWidgetItem(str(data['Period'])))
                self.access_view.setItem(row, 3, QTableWidgetItem(f"{data['Start_Sec']:.1f}"))
                self.access_view.setItem(row, 4, QTableWidgetItem(f"{data['End_Sec']:.1f}"))
                self.access_view.setItem(row, 5, QTableWidgetItem(data['Start_UTC']))
                self.access_view.setItem(row, 6, QTableWidgetItem(data['End_UTC']))
                self.access_view.setItem(row, 7, QTableWidgetItem(f"{data['Duration_Sec']:.1f}"))
                self.access_view.setItem(row, 8, QTableWidgetItem(f"{data['Total_Access_Sec']:.1f}"))
                self.access_view.setItem(row, 9, QTableWidgetItem(f"{data['Access_%']:.1f}%"))
                
                if results['analysis_type'] == 'duration_multi_track':
                    self.access_view.setItem(row, 10, QTableWidgetItem(data['Tracking_Info']))
                
                # Highlight based on tracking targets
                if results['analysis_type'] == 'duration_multi_track':
                    tracking_targets = results.get('tracking_targets', [])
                    if data['Debris_ID'] in tracking_targets:
                        for col in range(len(headers)):
                            item = self.access_view.item(row, col)
                            if item:
                                item.setBackground(QColor(255, 255, 100))
            
            self.access_view.resizeColumnsToContents()
        
        self._update_duration_stats(results)
        
        if results['single_time_snapshots']:
            last_snapshot = results['single_time_snapshots'][-1]
            self._update_single_time_displays(last_snapshot)

    def _update_duration_stats(self, results):
        """Update statistics for duration analysis with UTC timestamps"""
        summary = results['summary']
        
        total_access_periods = 0
        total_debris_with_access = 0
        max_access_percentage = 0
        
        for sat_name in summary['access_summary']:
            for debris_id in summary['access_summary'][sat_name]:
                access_info = summary['access_summary'][sat_name][debris_id]
                total_access_periods += access_info['num_access_periods']
                if access_info['num_access_periods'] > 0:
                    total_debris_with_access += 1
                max_access_percentage = max(max_access_percentage, access_info['access_percentage'])
        
        # Multi-track specific stats
        multi_track_stats = []
        if results['analysis_type'] == 'duration_multi_track':
            tracking_targets = results.get('tracking_targets', [])
            multi_track_stats = [
                ("Analysis Mode", "Multi-Target Tracking (Two-Horizon)"),
                ("Tracking targets", str(tracking_targets)),
                ("Number of tracking scenarios", len(tracking_targets)),
                ("", "")
            ]
        
        filter_info = []
        if results.get('selected_debris_ids'):
            filter_info = [
                ("Debris filter enabled", results.get('debris_filter_enabled', False)),
                ("Selected debris IDs", str(results.get('selected_debris_ids', [])))
            ]
        
        analysis_mode = "coords (no timestep filtering)" if summary.get('ignore_timestep_filtering', False) else "timestep-filtered"
        visibility_method = results.get('visibility_method', 'unknown')
        
        # Enhanced stats with UTC timestamps
        stats_rows = [
            ("Analysis Type", "Duration Analysis"),
            ("Analysis Mode", analysis_mode),
            ("Visibility Method", f"{visibility_method}"),
            ("Start time (UTC)", results['start_time'].utc.isot),
            ("End time (UTC)", (results['start_time'] + summary['duration_sec'] * u.s).utc.isot),
            ("Duration (sec)", f"{summary['duration_sec']:.1f}"),
            ("Time step (sec)", f"{summary['step_sec']:.1f}"),
            ("Total time steps", summary['total_time_steps']),
            ("", ""),
        ]
        
        if multi_track_stats:
            stats_rows.extend(multi_track_stats)
        
        if filter_info:
            stats_rows.extend(filter_info)
            stats_rows.append(("", ""))
        
        stats_rows.extend([
            ("Total access periods (all satellites)", total_access_periods),
            ("Debris objects with access", total_debris_with_access),
            ("Maximum access percentage", f"{max_access_percentage:.1f}%"),
        ])
        
        self.stats_view.clear()
        self.stats_view.setColumnCount(2)
        self.stats_view.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_view.setRowCount(len(stats_rows))
        
        for i, (metric, value) in enumerate(stats_rows):
            self.stats_view.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_view.setItem(i, 1, QTableWidgetItem(str(value)))
        
        self.stats_view.resizeColumnsToContents()

    def _update_single_time_displays(self, results):
        """Update displays for single-time analysis results"""
        
        visibility_matrix = results['visibility_matrix']
        visibility_reasons = results['visibility_reasons']
        pixel_positions = results['pixel_positions']
        tracked_debris_id = results['tracked_debris_id']
        
        self.detailed_view.clear()
        
        satellite_names = list(visibility_matrix.keys())
        all_debris_ids = set()
        for sat_data in visibility_matrix.values():
            all_debris_ids.update(sat_data.keys())
        debris_ids = sorted(all_debris_ids)
        
        print(f"Updating display for {len(debris_ids)} debris objects: {debris_ids}")
        
        self.detailed_view.setColumnCount(len(satellite_names) + 1)
        self.detailed_view.setRowCount(len(debris_ids))
        
        headers = ["Debris ID"] + satellite_names
        self.detailed_view.setHorizontalHeaderLabels(headers)
        
        for row, debris_id in enumerate(debris_ids):
            id_item = QTableWidgetItem(str(debris_id))
            if debris_id == tracked_debris_id:
                id_item.setBackground(QColor(255, 255, 100))
                id_item.setText(f"{debris_id} (TRACKED)")
            self.detailed_view.setItem(row, 0, id_item)
            
            for col, sat_name in enumerate(satellite_names):
                dist = visibility_matrix[sat_name].get(debris_id, 0)
                reason = visibility_reasons[sat_name].get(debris_id, "")
                
                if dist <= 0:
                    if reason == "Beyond horizon (Two-Horizon)":
                        item = QTableWidgetItem("Beyond horizon")
                        item.setBackground(QColor(255, 200, 200))
                    elif reason == "Outside FOV":
                        item = QTableWidgetItem("Outside FOV")
                        item.setBackground(QColor(240, 240, 240))
                    elif reason == "Behind camera":
                        item = QTableWidgetItem("Behind camera")
                        item.setBackground(QColor(230, 230, 230))
                    elif reason == "Outside sensor":
                        item = QTableWidgetItem("Outside sensor")
                        item.setBackground(QColor(220, 220, 220))
                    else:
                        item = QTableWidgetItem("Not visible")
                        item.setBackground(QColor(240, 240, 240))
                else:
                    pixel_pos = pixel_positions[sat_name].get(debris_id, (0, 0))
                    item = QTableWidgetItem(f"Visible ({dist:.1f} km)\n({pixel_pos[0]:.0f},{pixel_pos[1]:.0f})")
                    strength = min(1.0, 20000/dist) if dist > 0 else 0
                    r = int(255 * (1 - strength))
                    g = 200
                    b = int(100 * (1 - strength))
                    item.setBackground(QColor(r, g, b))
                
                if debris_id == tracked_debris_id:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                
                self.detailed_view.setItem(row, col + 1, item)
        
        self.detailed_view.resizeColumnsToContents()
        
        self._update_satellite_view(satellite_names, debris_ids, visibility_matrix, pixel_positions, tracked_debris_id)
        self._update_debris_view(debris_ids, satellite_names, visibility_matrix, pixel_positions, tracked_debris_id)
        
        if results.get('stats'):
            self._update_single_time_stats(results, satellite_names, debris_ids)
    
    def _update_satellite_view(self, satellite_names, debris_ids, visibility_matrix, pixel_positions, tracked_debris_id):
        """Update satellite-centric view"""
        sat_results = []
        for sat_name in satellite_names:
            visible_debris = []
            for debris_id in debris_ids:
                dist = visibility_matrix[sat_name].get(debris_id, 0)
                if dist > 0:
                    pixel_pos = pixel_positions[sat_name].get(debris_id, (0, 0))
                    visible_debris.append((debris_id, dist, pixel_pos))
            
            sat_results.append((sat_name, len(visible_debris), visible_debris))
        
        self.sat_view.clear()
        self.sat_view.setColumnCount(3)
        self.sat_view.setHorizontalHeaderLabels(["Satellite", "Visible Debris Count", "Visible Debris IDs (with pixel positions)"])
        self.sat_view.setRowCount(len(sat_results))
        
        for i, (sat_name, count, debris_list) in enumerate(sat_results):
            self.sat_view.setItem(i, 0, QTableWidgetItem(sat_name))
            self.sat_view.setItem(i, 1, QTableWidgetItem(str(count)))
            
            debris_ids_text = ""
            for j, (d_id, d_range, d_pixel) in enumerate(debris_list):
                if j > 0:
                    debris_ids_text += ", "
                
                if d_id == tracked_debris_id:
                    debris_ids_text += f"{d_id}*({d_pixel[0]:.0f},{d_pixel[1]:.0f})"
                else:
                    debris_ids_text += f"{d_id}({d_pixel[0]:.0f},{d_pixel[1]:.0f})"
            self.sat_view.setItem(i, 2, QTableWidgetItem(debris_ids_text))
        
        self.sat_view.resizeColumnsToContents()
    
    def _update_debris_view(self, debris_ids, satellite_names, visibility_matrix, pixel_positions, tracked_debris_id):
        """Update debris-centric view"""
        debris_results = {}
        for debris_id in debris_ids:
            debris_results[debris_id] = []
            for sat_name in satellite_names:
                dist = visibility_matrix[sat_name].get(debris_id, 0)
                if dist > 0:
                    pixel_pos = pixel_positions[sat_name].get(debris_id, (0, 0))
                    debris_results[debris_id].append((sat_name, dist, pixel_pos))
        
        self.debris_view.clear()
        self.debris_view.setColumnCount(3)
        self.debris_view.setHorizontalHeaderLabels(["Debris ID", "# Satellites", "Satellites That Can See It"])
        
        # Show ALL debris that were analyzed, not just visible ones
        self.debris_view.setRowCount(len(debris_ids))
        
        for i, debris_id in enumerate(debris_ids):
            sat_list = debris_results[debris_id]
            item = QTableWidgetItem(str(debris_id))
            
            if debris_id == tracked_debris_id:
                item.setBackground(QColor(255, 255, 100))
                item.setText(f"{debris_id} (TRACKED)")
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            
            self.debris_view.setItem(i, 0, item)
            self.debris_view.setItem(i, 1, QTableWidgetItem(str(len(sat_list))))
            
            if sat_list:
                sat_names_str = ", ".join(sat[0] for sat in sat_list)
            else:
                sat_names_str = "None"
            
            self.debris_view.setItem(i, 2, QTableWidgetItem(sat_names_str))
        
        self.debris_view.resizeColumnsToContents()
    
    def _update_single_time_stats(self, results, satellite_names, debris_ids):
        """Update statistics for single-time analysis with UTC timestamp"""
        stats = results['stats']
        tracked_debris_id = results['tracked_debris_id']
        visibility_matrix = results['visibility_matrix']
        analysis_time = results['analysis_time']
        
        debris_results = {}
        for debris_id in debris_ids:
            debris_results[debris_id] = []
            for sat_name in satellite_names:
                dist = visibility_matrix[sat_name].get(debris_id, 0)
                if dist > 0:
                    debris_results[debris_id].append(sat_name)
        
        visible_debris_ids = [debris_id for debris_id, sats in debris_results.items() if sats]
        max_visible_sat = max([len(sats) for sats in debris_results.values()]) if debris_results else 0
        avg_visible_sat = sum([len(sats) for sats in debris_results.values()]) / len(debris_ids) if debris_ids else 0
        
        tracked_stats = []
        if tracked_debris_id is not None and tracked_debris_id >= 0:
            seeing_count = sum(1 for sat_name in satellite_names if 
                            visibility_matrix[sat_name].get(tracked_debris_id, 0) > 0)
            
            tracked_stats = [
                ("Tracked debris ID", tracked_debris_id),
                ("Satellites that can see tracked debris", seeing_count)
            ]
        
        filter_info = []
        if hasattr(results, 'selected_debris_ids'):
            filter_info = [
                ("Debris filter enabled", results.get('debris_filter_enabled', False)),
                ("Selected debris IDs", str(results.get('selected_debris_ids', [])))
            ]
        
        visibility_method = results.get('visibility_method', 'unknown')
        
        stats_rows = [
            ("Analysis Type", "Single Time Point"),
            ("Visibility Method", f"{visibility_method}"),
            ("Analysis time (UTC)", analysis_time.utc.isot),
            ("Analysis time offset (sec)", f"{results.get('time_offset', 0):.1f}"),
            ("Total satellites", len(satellite_names)),
            ("Total debris objects analyzed", len(debris_ids)),
            ("", ""),
        ]
        
        if filter_info:
            stats_rows.extend(filter_info)
            stats_rows.append(("", ""))
        
        stats_rows.extend([
            ("Visibility check failures:", ""),
            ("  Beyond horizon (Two-Horizon)", stats["beyond_horizon"]),
            ("  Outside FOV cone", stats["outside_fov"]),
            ("  Behind camera plane", stats["behind_camera"]),
            ("  Outside sensor bounds", stats["outside_sensor"]),
            ("", ""),
            ("Visible debris objects", stats["visible"]),
            ("Debris objects with no visibility", len(debris_ids) - len(visible_debris_ids)),
            ("Visibility percentage", f"{(stats['visible']/stats['total_debris']*100):.1f}%" if stats['total_debris'] > 0 else "0%"),
            ("", ""),
            ("Max satellites seeing same debris", max_visible_sat),
            ("Avg satellites per analyzed debris", f"{avg_visible_sat:.2f}")
        ])
        
        if tracked_stats:
            stats_rows.extend([("", "")])
            stats_rows.extend(tracked_stats)
        
        self.stats_view.clear()
        self.stats_view.setColumnCount(2)
        self.stats_view.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_view.setRowCount(len(stats_rows))
        
        for i, (metric, value) in enumerate(stats_rows):
            self.stats_view.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_view.setItem(i, 1, QTableWidgetItem(str(value)))
        
        self.stats_view.resizeColumnsToContents()
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other objects to JSON-serializable Python types"""
        import numpy as np
        from astropy.time import Time
        from astropy import units as u
        
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Time):
            return obj.iso
        elif hasattr(obj, 'value') and hasattr(obj, 'unit'):  # Astropy Quantity
            return float(obj.value)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    def export_results(self):
        """Enhanced export for both single-time and duration results"""
        if not self.last_results:
            QMessageBox.warning(self, "No Results", "No analysis results to export.")
            return
            
        save_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if not save_dir:
            return
            
        try:
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            
            if self.last_results.get('analysis_type') in ['duration', 'duration_multi_track']:
                self._export_duration_results(save_dir, timestamp)
            else:
                self._export_single_time_results(save_dir, timestamp)
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting: {str(e)}")
    
    def _export_duration_results(self, save_dir, timestamp):
        """Export duration analysis results with UTC timestamps"""
        selected_debris = self.get_selected_debris_ids()
        debris_suffix = f"_debris_{'-'.join(map(str, selected_debris))}" if len(selected_debris) <= 5 else f"_debris_{len(selected_debris)}selected"
        
        mode_suffix = "_coords" if self.last_results.get('ignore_timestep_filtering', False) else "_filtered"
        
        if self.last_results.get('analysis_type') == 'duration_multi_track':
            mode_suffix += "_multitrack"
        
        base_name = f"visibility_duration_twohorizon{mode_suffix}_{timestamp}{debris_suffix}"
        
        access_data = []
        analysis_start_time = self.last_results['start_time']

        for sat_name in self.last_results['access_periods']:
            for debris_id in self.last_results['access_periods'][sat_name]:
                periods = self.last_results['access_periods'][sat_name][debris_id]
                summary = self.last_results['summary']['access_summary'][sat_name][debris_id]
                
                for i, (start, end) in enumerate(periods):
                    start_utc_time = analysis_start_time + start * u.s
                    end_utc_time = analysis_start_time + end * u.s
                    
                    access_data.append({
                        'Satellite': sat_name,
                        'Debris_ID': debris_id,
                        'Access_Period': i + 1,
                        'Start_Time_UTC': start_utc_time.utc.isot,
                        'End_Time_UTC': end_utc_time.utc.isot,
                        'Start_Time_Sec': start,
                        'End_Time_Sec': end,
                        'Duration_Sec': end - start,
                        'Total_Access_Time_Sec': summary['total_access_time_sec'],
                        'Access_Percentage': summary['access_percentage']
                    })
        
        access_df = pd.DataFrame(access_data)
        access_path = os.path.join(save_dir, f"{base_name}_access_periods.csv")
        access_df.to_csv(access_path, index=False)
        
        timeline_data = []
        for sat_name in self.last_results['visibility_timeline']:
            for debris_id in self.last_results['visibility_timeline'][sat_name]:
                timeline = self.last_results['visibility_timeline'][sat_name][debris_id]
                for point in timeline:
                    timeline_data.append({
                        'Satellite': sat_name,
                        'Debris_ID': debris_id,
                        'Time_Offset_Sec': point['time_offset'],
                        'Time_ISO': point['time_iso'],
                        'Visible': point['visible'],
                        'Range_km': point['range_km'],
                        'Tracking_Target': point.get('tracking_target', 'Unknown')
                    })
        
        timeline_df = pd.DataFrame(timeline_data)
        timeline_path = os.path.join(save_dir, f"{base_name}_full_timeline.csv")
        timeline_df.to_csv(timeline_path, index=False)
        
        metadata = {
            'analysis_type': self.last_results.get('analysis_type', 'duration'),
            'analysis_mode': 'coords' if self.last_results.get('ignore_timestep_filtering', False) else 'timestep_filtered',
            'coordinate_system': 'ECI_to_ECEF_WGS84',
            'visibility_method': self.last_results.get('visibility_method', 'two_horizon_access_criterion'),
            'ignore_timestep_filtering': bool(self.last_results.get('ignore_timestep_filtering', False)),
            'start_time': self.last_results['start_time'].iso,
            'duration_sec': float(self.last_results['duration_sec']),
            'step_sec': float(self.last_results['step_sec']),
            'total_time_steps': int(self.last_results['summary']['total_time_steps']),
            'tracking_targets': [int(x) for x in self.last_results.get('tracking_targets', [])],
            'selected_debris_ids': [int(x) for x in self.last_results.get('selected_debris_ids', [])],
            'debris_filter_enabled': bool(self.last_results.get('debris_filter_enabled', False)),
            'sensor_params': self._make_json_serializable(self.last_results.get('sensor_params', {})),
            'timestamp_generated': timestamp
        }

        metadata = self._make_json_serializable(metadata)

        metadata_path = os.path.join(save_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        analysis_mode_text = "coords (no timestep filtering)" if self.last_results.get('ignore_timestep_filtering', False) else "timestep-filtered"
        
        if self.last_results.get('analysis_type') == 'duration_multi_track':
            analysis_mode_text += " with multi-target tracking"
        
        QMessageBox.information(self, "Export Complete", 
                               f"Duration analysis (Two-Horizon {analysis_mode_text}) exported with UTC timestamps!\n\n"
                               f"Analyzed debris: {selected_debris}\n"
                               f"Tracking targets: {self.last_results.get('tracking_targets', [])}\n"
                               f"Visibility method: Two-Horizon Access Criterion\n"
                               f"Files saved:\n"
                               f"- Access periods: {base_name}_access_periods.csv\n"
                               f"- Full timeline: {base_name}_full_timeline.csv\n"
                               f"- Metadata: {base_name}_metadata.json")
    
    def _export_single_time_results(self, save_dir, timestamp):
        """Export single-time analysis results with UTC timestamps"""
        selected_debris = self.get_selected_debris_ids()
        debris_suffix = f"_debris_{'-'.join(map(str, selected_debris))}" if len(selected_debris) <= 5 else f"_debris_{len(selected_debris)}selected"
        base_name = f"visibility_single_twohorizon_{timestamp}{debris_suffix}"
        
        matrix_data = []
        satellite_names = list(self.last_results['visibility_matrix'].keys())
        all_debris_ids = set()
        for sat_data in self.last_results['visibility_matrix'].values():
            all_debris_ids.update(sat_data.keys())
        debris_ids = sorted(all_debris_ids)
        
        for debris_id in debris_ids:
            row_data = {
                'Debris_ID': debris_id,
                'Analysis_Time_UTC': self.last_results['analysis_time'].utc.isot
            }
            for sat_name in satellite_names:
                dist = self.last_results['visibility_matrix'][sat_name].get(debris_id, 0)
                reason = self.last_results['visibility_reasons'][sat_name].get(debris_id, "")
                if dist > 0:
                    pixel_pos = self.last_results['pixel_positions'][sat_name].get(debris_id, (0, 0))
                    row_data[f"{sat_name}_range_km"] = dist
                    row_data[f"{sat_name}_pixel_x"] = pixel_pos[0]
                    row_data[f"{sat_name}_pixel_y"] = pixel_pos[1]
                else:
                    row_data[f"{sat_name}_range_km"] = 0
                    row_data[f"{sat_name}_pixel_x"] = -1
                    row_data[f"{sat_name}_pixel_y"] = -1
                row_data[f"{sat_name}_status"] = reason
            matrix_data.append(row_data)
        
        matrix_df = pd.DataFrame(matrix_data)
        matrix_path = os.path.join(save_dir, f"{base_name}_matrix.csv")
        matrix_df.to_csv(matrix_path, index=False)
        
        metadata = {
            'analysis_type': 'single_time',
            'coordinate_system': 'ECI_to_ECEF_WGS84',
            'visibility_method': self.last_results.get('visibility_method', 'two_horizon_access_criterion'),
            'analysis_time': self.last_results['analysis_time'].iso,
            'tracked_debris_id': self.last_results['tracked_debris_id'],
            'selected_debris_ids': self.last_results.get('selected_debris_ids', []),
            'debris_filter_enabled': self.last_results.get('debris_filter_enabled', False),
            'sensor_params': self.last_results.get('sensor_params', {}),
            'total_satellites': len(satellite_names),
            'total_debris': len(debris_ids),
            'timestamp_generated': timestamp
        }
        
        metadata_path = os.path.join(save_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        QMessageBox.information(self, "Export Complete", 
                               f"Single-time visibility analysis (Two-Horizon) exported with UTC timestamps!\n\n"
                               f"Analyzed debris: {selected_debris}\n"
                               f"Visibility method: Two-Horizon Access Criterion\n"
                               f"Files saved:\n"
                               f"- Matrix: {base_name}_matrix.csv\n"
                               f"- Metadata: {base_name}_metadata.json")
# ----------------------------------------------------------------------
# Main Window with Coordinate System Support
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    """
    Main application window with proper coordinate system handling
    Enhanced time synchronization and comprehensive tab management
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESA Viewer - Coordinate Systems")
        self.setWindowIcon(QIcon("pyxel_logo.png"))

        self.setMinimumSize(1200, 800)
        
        # Create all tabs with coordinate systems
        sat = SatManagerTab()
        deb = DebrisTab()
        emi = EmissionsTab()
        viz = VizTab(sat, deb)
        data = DataTab(viz)
        pho = PhotonConversionTab(sat, deb, emi, viz)
        vis = VisibilityAnalysisTab(sat, deb, emi, pho)

        # Store references
        self.time_synced_tabs = [viz, vis, pho, sat]
        self._syncing_time = False
        self._syncing_debris = False
        
        self.sat_tab = sat
        self.deb_tab = deb
        self.emi_tab = emi
        self.viz_tab = viz
        self.data_tab = data
        self.pho_tab = pho
        self.vis_tab = vis
        
        # Enhanced time synchronization
        viz.timeChanged.connect(lambda dt: self._sync_time_from_viz(dt))
        viz.snapshot.dateTimeChanged.connect(lambda dt: self._sync_time_from_viz_direct(dt))
        vis.snapshot.dateTimeChanged.connect(lambda dt: self._sync_time_from_vis(dt))
        sat.timeChanged.connect(lambda dt: self._sync_time_from_sat(dt))
        sat.sync_viz_time_btn.clicked.connect(lambda: self._sync_sat_from_viz())
        
        # Satellite change connections
        sat.satsChanged.connect(self._on_satellites_changed)
        
        # Debris data connections
        original_debris_load = deb.load
        deb.load = lambda: self._update_debris_tracking_after_load(original_debris_load)
        deb.debrisDataChanged.connect(self._on_debris_data_changed)
        
        # Debris tracking synchronization
        pho.tracked_debris.currentIndexChanged.connect(
            lambda: self._sync_debris_tracking_from_tab(pho.tracked_debris, "PhotonConversion"))
        viz.tracked_debris.currentIndexChanged.connect(
            lambda: self._sync_debris_tracking_from_tab(viz.tracked_debris, "Visualization"))
        vis.tracked_debris.currentIndexChanged.connect(
            lambda: self._sync_debris_tracking_from_tab(vis.tracked_debris, "VisibilityAnalysis"))
        
        # Tab creation
        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.addTab(sat, "Satellites")
        tabs.addTab(viz, "Visualisation")
        tabs.addTab(data, "LLA Data")
        tabs.addTab(deb, "Debris")
        tabs.addTab(emi, "Emissions")
        tabs.addTab(pho, "Photon Conv.")
        tabs.addTab(vis, "Visibility Analysis")
        
        tabs.currentChanged.connect(lambda i: self._on_tab_changed(i, tabs))
        self.tabs_widget = tabs

        # Central widget setup
        central = QWidget()
        central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(central)
        lay.addWidget(tabs)

        # Logos section
        logo_h = QHBoxLayout()
        logo_h.addStretch()
        
        logo_files = [
            ("FraunhoferLogo.png", "Fraunhofer Institute"),
            ("StrathclydeLogo.png", "University of Strathclyde"), 
            ("ESALogo.png", "European Space Agency")
        ]
        
        for filename, alt_text in logo_files:
            try:
                lbl = QLabel()
                lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                pm = QPixmap(filename)
                if not pm.isNull():
                    lbl.setPixmap(pm.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    lbl.setToolTip(alt_text)
                    logo_h.addWidget(lbl)
                else:
                    print(f"Warning: Could not load logo {filename}")
            except Exception as e:
                print(f"Warning: Error loading logo {filename}: {e}")
        
        logo_h.addStretch()
        lay.addLayout(logo_h)

        self.setCentralWidget(central)
        
        # Styling
        try:
            apply_stylesheet(QApplication.instance() or QApplication(sys.argv), theme="dark_amber.xml")
        except Exception as e:
            print(f"Warning: Could not apply theme: {e}")
        
        # Window sizing
        screen = QApplication.primaryScreen().availableGeometry()
        initial_width = int(screen.width() * 0.85)
        initial_height = int(screen.height() * 0.75)
        
        initial_width = max(initial_width, 1200)
        initial_height = max(initial_height, 800)
        
        self.resize(initial_width, initial_height)
        
        x = (screen.width() - initial_width) // 2
        y = (screen.height() - initial_height) // 2
        self.move(max(0, x), max(0, y))
        
        # Initial setup
        QTimer.singleShot(100, self._perform_initial_setup)
        
        print("ESA Viewer initialized successfully with coordinate systems")

    def _perform_initial_setup(self):
        """Initial setup with coordinate systems"""
        try:
            if self.viz_tab.satman.get_satellites():
                self.viz_tab.generate_lla_data()
                
            self._update_all_debris_lists()
            self.tabs_widget.setCurrentWidget(self.viz_tab)
            
            print("Initial setup completed with coordinate transformations")
        except Exception as e:
            print(f"Warning: Error in initial setup: {e}")

    def _sync_time_from_viz(self, datetime):
        """Sync time from visualization tab"""
        if self._syncing_time:
            return
        self._sync_time_from_viz_direct(datetime)

    def _sync_time_from_viz_direct(self, datetime):
        """Sync time from visualization tab to all other time-enabled tabs"""
        if self._syncing_time:
            return
            
        self._syncing_time = True
        try:
            if hasattr(self.vis_tab, 'snapshot'):
                self.vis_tab.snapshot.blockSignals(True)
                self.vis_tab.snapshot.setDateTime(datetime)
                self.vis_tab.snapshot.blockSignals(False)
            
            if hasattr(self.sat_tab, 'set_time_from_external'):
                self.sat_tab.set_time_from_external(datetime)
                
            print(f"Time synced from Viz tab: {datetime.toString('yyyy-MM-dd hh:mm:ss')}")
            
        except Exception as e:
            print(f"Error in time sync from viz: {e}")
        finally:
            self._syncing_time = False

    def _sync_time_from_vis(self, datetime):
        """Sync time from visibility analysis tab"""
        if self._syncing_time:
            return
            
        self._syncing_time = True
        try:
            if hasattr(self.viz_tab, 'snapshot'):
                self.viz_tab.snapshot.blockSignals(True)
                self.viz_tab.snapshot.setDateTime(datetime)
                self.viz_tab.snapshot.blockSignals(False)
                QTimer.singleShot(100, self.viz_tab.request_redraw)
            
            if hasattr(self.sat_tab, 'set_time_from_external'):
                self.sat_tab.set_time_from_external(datetime)
            
            print(f"Time synced from Visibility tab: {datetime.toString('yyyy-MM-dd hh:mm:ss')}")
            
        except Exception as e:
            print(f"Error in time sync from visibility: {e}")
        finally:
            self._syncing_time = False

    def _sync_time_from_sat(self, datetime):
        """Sync time from satellite tab"""
        if self._syncing_time:
            return
            
        self._syncing_time = True
        try:
            if hasattr(self.viz_tab, 'snapshot'):
                self.viz_tab.snapshot.blockSignals(True)
                self.viz_tab.snapshot.setDateTime(datetime)
                self.viz_tab.snapshot.blockSignals(False)
            
            if hasattr(self.vis_tab, 'snapshot'):
                self.vis_tab.snapshot.blockSignals(True)
                self.vis_tab.snapshot.setDateTime(datetime)
                self.vis_tab.snapshot.blockSignals(False)
            
            print(f"Time synced from Satellite tab: {datetime.toString('yyyy-MM-dd hh:mm:ss')}")
            
        except Exception as e:
            print(f"Error in time sync from satellite tab: {e}")
        finally:
            self._syncing_time = False

    def _sync_sat_from_viz(self):
        """Sync satellite tab time from visualization tab"""
        try:
            viz_time = self.viz_tab.snapshot.dateTime()
            self.sat_tab.set_time_from_external(viz_time)
        except Exception as e:
            print(f"Error syncing satellite tab from viz: {e}")

    def _on_satellites_changed(self):
        """Handle satellite changes"""
        print("Satellites changed - updating all dependent components with coords")
        
        try:
            QTimer.singleShot(50, self.viz_tab.generate_lla_data)
            QTimer.singleShot(150, self.data_tab.auto_populate)
            QTimer.singleShot(100, self._update_all_satellite_lists)
            QTimer.singleShot(200, self.viz_tab.request_redraw)
            
        except Exception as e:
            print(f"Error handling satellite changes: {e}")

    def _update_all_satellite_lists(self):
        """Update satellite lists in all tabs"""
        try:
            if hasattr(self.pho_tab, 'update_satellite_list'):
                self.pho_tab.update_satellite_list()
                
        except Exception as e:
            print(f"Error updating satellite lists: {e}")

    def _on_debris_data_changed(self):
        """Handle debris data changes"""
        print("Debris data changed - updating all dependent components")
        
        try:
            QTimer.singleShot(50, self._update_all_debris_lists)
            QTimer.singleShot(100, self.viz_tab.request_redraw)
            
        except Exception as e:
            print(f"Error handling debris data changes: {e}")

    def _update_all_debris_lists(self):
        """Update debris tracking lists in all relevant tabs"""
        try:
            for tab in [self.pho_tab, self.viz_tab, self.vis_tab]:
                if hasattr(tab, 'update_debris_list'):
                    tab.update_debris_list()
                if hasattr(tab, 'update_debris_selection_list'):
                    tab.update_debris_selection_list()
        except Exception as e:
            print(f"Error updating debris lists: {e}")

    def _on_tab_changed(self, tab_index, tabs_widget):
        """Enhanced tab switching behavior"""
        try:
            current_tab = tabs_widget.widget(tab_index)
            tab_name = tabs_widget.tabText(tab_index)
            
            print(f"Switched to tab: {tab_name}")
            
            if current_tab == self.viz_tab:
                QTimer.singleShot(100, self.viz_tab.request_redraw)
                QTimer.singleShot(50, self.viz_tab.generate_lla_data)
            elif current_tab == self.data_tab:
                QTimer.singleShot(50, self.data_tab.auto_populate)
            elif current_tab == self.vis_tab:
                if hasattr(self.vis_tab, 'update_debris_list'):
                    QTimer.singleShot(50, self.vis_tab.update_debris_list)
            elif current_tab == self.pho_tab:
                QTimer.singleShot(50, self.pho_tab.update_debris_list)
                QTimer.singleShot(50, self.pho_tab.update_satellite_list)
            
            tabs_with_debris = [self.pho_tab, self.viz_tab, self.vis_tab]
            for tab in tabs_with_debris:
                if current_tab == tab and hasattr(tab, 'update_debris_list'):
                    QTimer.singleShot(100, tab.update_debris_list)
                    break
                    
        except Exception as e:
            print(f"Error in tab change handler: {e}")

    def _update_debris_tracking_after_load(self, original_load_func):
        """Enhanced wrapper for debris tab's load function"""
        try:
            result = original_load_func()
            
            print("Debris data loaded - updating all tabs with coordinate handling")
            
            QTimer.singleShot(50, self._update_all_debris_lists)
            QTimer.singleShot(150, self.viz_tab.request_redraw)
            
            return result
            
        except Exception as e:
            print(f"Error in debris tracking update after load: {e}")
            return None
    
    def _sync_debris_tracking_from_tab(self, source_dropdown, source_tab_name):
        """Enhanced debris tracking synchronization"""
        if self._syncing_debris:
            return
            
        self._syncing_debris = True
        try:
            current_id = source_dropdown.currentData()
            
            target_dropdowns = []
            
            if source_tab_name != "PhotonConversion" and hasattr(self.pho_tab, 'tracked_debris'):
                target_dropdowns.append((self.pho_tab.tracked_debris, "PhotonConversion"))
            if source_tab_name != "Visualization" and hasattr(self.viz_tab, 'tracked_debris'):
                target_dropdowns.append((self.viz_tab.tracked_debris, "Visualization"))
            if source_tab_name != "VisibilityAnalysis" and hasattr(self.vis_tab, 'tracked_debris'):
                target_dropdowns.append((self.vis_tab.tracked_debris, "VisibilityAnalysis"))
            
            for dropdown, tab_name in target_dropdowns:
                if hasattr(dropdown, 'findData') and hasattr(dropdown, 'currentIndex'):
                    index = dropdown.findData(current_id)
                    if index >= 0 and index != dropdown.currentIndex():
                        dropdown.blockSignals(True)
                        dropdown.setCurrentIndex(index)
                        dropdown.blockSignals(False)
            
            if source_tab_name == "Visualization":
                QTimer.singleShot(100, self.viz_tab.request_redraw)
            
            print(f"Debris tracking synced from {source_tab_name} tab: ID {current_id}")
            
        except Exception as e:
            print(f"Error in debris tracking sync: {e}")
        finally:
            self._syncing_debris = False

    def closeEvent(self, event):
        """Handle application closing with cleanup"""
        try:
            print("ESA Viewer closing...")
            
            if hasattr(self.viz_tab, 'auto_refresh_timer'):
                self.viz_tab.auto_refresh_timer.stop()
            if hasattr(self.viz_tab, 'redraw_timer'):
                self.viz_tab.redraw_timer.stop()
            if hasattr(self.viz_tab, 'lla_update_timer'):
                self.viz_tab.lla_update_timer.stop()
            
            if hasattr(self.viz_tab, 'lla_data'):
                self.viz_tab.lla_data.clear()
            
            if hasattr(self.viz_tab, 'clear_all_items'):
                self.viz_tab.clear_all_items()
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        event.accept()

    def showEvent(self, event):
        """Handle window show event"""
        super().showEvent(event)
        
        if hasattr(self.viz_tab, 'request_redraw'):
            QTimer.singleShot(200, self.viz_tab.request_redraw)

    def resizeEvent(self, event):
        """Handle window resize event"""
        super().resizeEvent(event)
        
        if hasattr(self.viz_tab, 'view') and hasattr(self.viz_tab, 'is_visible'):
            if self.viz_tab.is_visible:
                QTimer.singleShot(100, self.viz_tab.view.update)

def main():
    """Launch the application with coordinate systems"""
    try:
        app = QApplication(sys.argv)
        
        app.setApplicationName("ESA Viewer - Coordinate Systems")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("ESA Debris Tracking - Coordinates")
        
        win = MainWindow()
        win.show()
        
        print("Starting ESA Viewer with coordinate systems...")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Fatal error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()