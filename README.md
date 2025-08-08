# DebriSim –  Space Debris Tracking/Emission Simulator & Sensor Simulation
---

## Overview
DebriSim is a Python-based application for **space-situational awareness**, debris tracking, and satellite mission planning. It delivers real-time 3D visualisation, precise coordinate-system transformations, and detailed sensor modelling―all packaged in a responsive GUI.

### Key Capabilities
- **Satellite Constellation Management** – Design and simulate Walker constellations  
- **Real-time 3D Visualisation** – Interactive Earth-centred scene with accurate orbital mechanics  
- **Debris-Detection Simulation** – Photon-to-electron conversion modelling for optical sensors  
- **Visibility Analysis** – Two-horizon access criterion with Earth-occlusion calculations  
- **Data Pipeline** – Import/export debris and emissions data in multiple formats  
- **Sensor Simulation** – Simulator Sensoe Noise with Pyxel Library  

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
PyQt5>=5.15.0
numpy>=1.20.0
pandas>=1.3.0
astropy>=4.0.0
poliastro>=0.15.0
pyqtgraph>=0.12.0
matplotlib>=3.3.0
scipy>=1.7.0
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
debrisim-env\Scriptsctivate
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

## Usage ― Quick Start

1. **Earth Texture**  
   Place `earth.jpg` (NASA Blue Marble) in the project root.

2. **Define Satellites**  
   *Satellites Tab →* Add a manual satellite **or** generate a Walker constellation.  
   Configure `a, e, i, RAAN, ω, ν`.

3. **Load Debris Data**  
   *Debris Tab →* Load CSV/Excel with columns  
   `Time, Iteration, Assembly_ID, Latitude, Longitude, Altitude`.

4. **Import Emissions**  
   *Emissions Tab →* Load intensity data (`OI_emissions_atomic`, `AlI_1/2_emissions_atomic`).

5. **Visualise**  
   *Visualisation Tab →* Inspect the 3-D Earth scene; choose target & epoch.

6. **Analyse Visibility**  
   *Visibility Analysis Tab →* Run Two-Horizon analysis, then export access periods / matrices.

---

## License
DebriSim is released under the **MIT License** – see [`LICENSE`](LICENSE) for full text.

---

## Institutions
Developed in collaboration with:

- Fraunhofer Institute  
- University of Strathclyde
- European Space Agency