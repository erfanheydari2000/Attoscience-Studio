
# Attoscience Studio

**Attoscience Studio** is a PyQt5-based scientific software suite designed for the interactive analysis and visualization of ultrafast light–matter interaction data, with a focus on attosecond pulse generation and high harmonic generation (HHG) from solid-state materials.

It integrates tools for analyzing TDDFT simulation results (e.g., from Octopus), visualizing pulse shapes and spectra, computing minimum pulse widths, and inspecting crystal structure data.

---
# Features

## Ground State Analysis
- **Crystal Structure Visualization**: Examine atomic arrangements in crystal lattices
- **Electronic Band Structure**: Analyze energy dispersion relations with automatic band gap calculation (direct/indirect)
- **Density of States (DOS)**: Visualize electronic state distributions as a function of energy
- **Electron Density Mapping**: Study spatial probability distributions of electrons in materials

## Laser Field Configuration
- **Single & Dual Pulse Setup**: Configure Ex, Ey, Ez components for complex field geometries
- **Vector Potential Analysis**: Time-dependent gauge field calculations (E = -∂A/∂t)
- **Polarization Gating**: Design isolated attosecond pulse generation schemes with precise control over:
  - Laser wavelength, intensity, and optical cycles
  - Ellipticity and rotation angles
  - Carrier envelope phase (CEP)
  - Temporal delays between pulses
  - Sine-square or Gaussian pulse envelopes
- **Gate Width Optimization**: Calculate time-dependent ellipticity ε(t) for optimal gating

## High Harmonic Generation Analysis
- **Time-Domain Current Visualization**: Monitor induced currents (jx, jy, jz) during laser-matter interaction
- **HHG Spectral Analysis**: Fourier transform analysis with both velocity and acceleration forms
- **Harmonic Yield Calculations**: Quantify emission efficiency for specific harmonic orders
- **Phase Analysis**: Extract phase information from harmonic components in radians and degrees
- **Ellipticity Measurements**: Determine circular polarization characteristics of generated harmonics
- **Cutoff Energy Predictions**: Identify maximum photon energies and plateau structures

## Attosecond Pulse Generation & Characterization
- **Dual-Method Pulse Synthesis**: 
  - Method 1: Direct Fourier transform of current components
  - Method 2: Derivative-based transform with frequency weighting
- **Minimum Pulse Width (FWHM) Optimization**: Find shortest achievable pulse durations
- **Gabor Transform Analysis**: Simultaneous time-frequency localization with Gaussian windowing
- **Transform-Limited Pulse Design**: Optimize bandwidth-duration products
- **Spectral Phase Control**: Analyze coherence requirements for pulse formation

## Ultrafast Electron Dynamics
- **Time-Resolved Excitation Tracking**: Monitor excited electron populations over optical cycles
- **Brillouin Zone Mapping**: Visualize k-space distributions of excited electrons and currents
- **Real-Time Animations**: Generate smooth time-evolution movies with:
  - Customizable grid resolution and interpolation
  - Multithreaded processing for responsive UI
  - Export capabilities for presentations
- **Preferential Direction Analysis**: Identify anisotropic excitation patterns

## Mathematical Toolbox
- **Fourier Transform Suite**: 
  - Function-based transforms with user-defined expressions
  - Data file processing with cubic interpolation
  - Inverse transforms for signal reconstruction
- **Unit Conversion System**: Seamless conversion between SI and atomic units for:
  - Intensity (W/cm² ↔ a.u.)
  - Time (seconds ↔ a.u.)
  - Energy (eV ↔ Hartree)
  - Electric fields (V/m ↔ a.u.)
  - Length (meters ↔ Bohr radius)

## Advanced Integration
- **Embedded IPython Console**: Full Python scripting environment for custom analysis
- **Flexible File Format Support**: Compatible with standard TD-DFT output formats
- **High-Performance Visualization**: Matplotlib integration with publication-ready plots
- **Memory-Efficient Processing**: Optimized algorithms for large dataset handling

## User Experience
- **Intuitive Tabbed Interface**: Organized workflow from ground state to attosecond analysis
- **Comprehensive Help System**: Built-in documentation with examples and best practices
- **Real-Time Parameter Feedback**: Immediate visualization of parameter changes
- **Error Handling & Validation**: Robust input checking with helpful error messages
---

## Installation

### Requirements

Python 3.8+  
See `requirements.txt` for all dependencies.

### From Source (Recommended)

```bash
git clone https://github.com/erfan-h/attoscience_studio.git
cd attoscience_studio
pip install .
