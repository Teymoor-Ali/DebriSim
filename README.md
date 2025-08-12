# DebriSim – Space Debris Tracking / Emission Simulator & Sensor Simulation

## Overview
DebriSim is a Python-based application for **space-situational awareness**, debris tracking, and satellite mission planning. It delivers real-time 3D visualisation, precise coordinate-system transformations, and detailed sensor modelling.

### Key Capabilities
- **Satellite Constellation Management** – Design and simulate Walker constellations  
- **Real-time 3D Visualisation** – Interactive Earth-centred scene with accurate orbital mechanics  
- **Debris-Detection Simulation** – Photon-to-electron conversion modelling for optical sensors  
- **Visibility Analysis** – Two-horizon access criterion with Earth-occlusion calculations  
- **Data Pipeline** – Import/export debris and emissions data in multiple formats  
- **Sensor Simulation** – Simulates sensor noise with the Pyxel library  

---

## Features

### Core Modules
| Module               | Description                                                                      |
|----------------------|----------------------------------------------------------------------------------|
| **Satellite Manager**| Define orbital elements, generate Walker constellations, propagate orbits        |
| **3D Visualisation** | Real-time Earth view with satellite tracks and debris rendering                   |
| **Debris Tracking**  | Load & process debris trajectories with precise coordinate transforms             |
| **Photon Conversion**| Simulate sensor response (quantum efficiency, filters, noise)                    |
| **Visibility Analysis**| Compute access periods using advanced geometric algorithms                     |
| **Data Export**      | CSV, NumPy, FITS, plus full metadata preservation                                 |

---

### Python Dependencies
```
numpy==2.2.6
pandas==2.3.0
astropy==7.1.0
poliastro==0.7.0
PyQt5==5.15.11
pyqtgraph==0.13.7
matplotlib==3.10.3
qt-material==2.17
openpyxl==3.1.5
xlrd==2.0.2
scipy==1.16.0
PyOpenGL==3.1.9
pyxel_sim==2.11.2
PyYAML==6.0.2
numba==0.61.2
pillow==11.3.0
```

---

## Installation

### Option 1 — Clone (Recommended)
```bash
git clone https://github.com/yourusername/debrisim.git
cd debrisim

# Set up a virtual environment
python -m venv debrisim-env
# Windows:
debrisim-env\Scripts\activate
# macOS / Linux:
source debrisim-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2 — Direct Download
```bash
# Download the release archive and extract
wget https://github.com/yourusername/debrisim/releases/latest -O debrisim.zip
unzip debrisim.zip && cd debrisim

# (Optional) Download NASA Blue Marble texture
wget https://visibleearth.nasa.gov/images/57752/.../57753l -O earth.jpg
```

---

## Usage — Quick Start
### A) Satellite Simulator
1. **Earth Texture** – Place `earth.jpg` (NASA Blue Marble) in the project root.  
2. **Define Satellites** – *Satellites Tab →* add a manual satellite or generate a **Walker constellation**. Configure `a, e, i, RAAN, ω, ν`.  
3. **Load Debris Data** – *Debris Tab →* load CSV/Excel with columns `Time, Iteration, Assembly_ID, Latitude, Longitude, Altitude`.  
4. **Import Emissions** – *Emissions Tab →* load intensity data (`OI_emissions_atomic`, `AlI_1/2_emissions_atomic`).  
5. **Visualise** – *Visualisation Tab →* inspect the 3‑D Earth scene; choose target & epoch.  
6. **Analyse Visibility** – *Visibility Analysis Tab →* run **Two-Horizon** analysis; export access periods/matrices.  
7. **Photon Conversion Tab**

      **Select Satellite** – Choose observing satellite from dropdown.  
      **Track Debris** – Select debris object to center camera on.  
      **Configure Sensor** – Set aperture, focal length, pixel pitch, sensor size, exposure time.  
      **Set Emission Parameters** – Configure separate quantum efficiency and filter transmission for:
      - **OI** (777 nm) emissions
      - **Al** (395 nm) emissions  
      **Background Illumination** – Optional uniform background *(typical: `1e-6 W/sr` for Earth vicinity)*.  
      **Single Frame** – Click **Compute** to generate dual-view images *(OI left, Al right)* with WGS84 coordinate transformations.  
      **Batch Processing** – Use **Track Main Body + Longest Survivor** for automated multi-timestep analysis with optional satellite handoff (Camera automatically selects any satellite that has the closest view of the debris).  
      **Save Results** – Export as NumPy arrays (`.npy`) and/or FITS files (`.fits`) with full metadata.

### A) Sensor Simulator
1. **Load Detector Config** – *File → Open YAML* and select your `.yaml/.yml` detector file.  
2. **Import Data** – *Load Image Tab →* choose `.npy` or `.fits`. Emission type (OI/Al) is auto-detected.  
3. **Quantum Efficiency** – Use **Auto Mode** (OI=0.3, Al=0.7) or **Manual Mode** (from YAML).  
4. **Run Single Exposure** – Click **Perform Exposure** to simulate sensor response.  
5. **Analyse** – Open **Statistics Tab** for ADU stats; use **Time Series** if available.  
6. **Export** – Click **Export CSV** for results.  
7. **Batch** – *Batch Processing Tab →* select input folder (multiple `.npy`), set output folder, run; results stream to CSV with full metadata.

## Institutions
Developed in collaboration with:

- Fraunhofer Institute  
- University of Strathclyde  
- European Space Agency
