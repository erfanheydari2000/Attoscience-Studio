# Attoscience Studio

**Attoscience Studio** is a PyQt5-based scientific software suite designed for the interactive analysis and visualization of ultrafast light–matter interaction data, with a focus on attosecond pulse generation and high harmonic generation (HHG) from solid-state materials.

It integrates tools for analyzing TDDFT simulation results (e.g., from Octopus), visualizing pulse shapes and spectra, computing minimum pulse widths, and inspecting crystal structure data.

---

## Features

**Real-time manitoring**
  While the program is running, CPU usage, memory, network, and Disk I/O information is displayed in real-time.

**Ground State tab**
  **Crystal structure** The user can draw the corresponding unit cell in 3D by providing the parser.log or CIF data files. The parser.log data file is obtained from the Octopus software.
  **Band structure** By providing a bandstructure data file and entering parameters such as Fermi energy, the user can specify the direct/indirect of the band gap and calculate the gap energy, and also draw the band structure.
  **Density of state** Allows users to plot the total density of states for a given material. Users can upload a DOS file (e.g., total-dos.dat) and customize the plot appearance.
  **Electron density** This feature allows users to view and analyze the electron density distribution across different spatial dimensions.
  
**Driving Pulse tab**
  **Electric field** This method allows the visualization of the electric field in two modes: Single Pulse Mode: Plots the electric field components (Ex, Ey, Ez) for a single laser pulse. double Pulse Mode: Plots the combined electric field for two pulses, often used in methods like polarization gating. In this case, the data file contains two separate pulses (Ex1, Ey1, Ez1 for the first pulse and Ex2, Ey2, Ez2 for the second pulse).
  **Vector potential** Similar to "Electric field", it has two options: single pulse and double pulse.
  **Polarization gating field** Allows to explore and configure the features and outputs related to the polarization gating field for generating attosecond pulses.
  **Polarization gate width** This function calculates the time-dependent ellipticity (ε(t)) of two combined laser fields to evaluate the width of the polarization gate.
  
**High Harmonic tab**
  **Total Current** Plots the total current obtained from the TD-DFT calculations.
  **High Harmonic Generation** This function computes the High Harmonic Generation (HHG) spectrum by performing a Fourier Transform on the total current data loaded from a file, converting it from the time domain to the frequency domain.
  **High Harmonic generation Yield** This feature calculates the harmonic yields for specified harmonic orders based on the total current data from a simulation. It performs a Fourier transform on the time-domain current data to compute the harmonic spectra for the x and y polarization components and the total spectrum. 
  **Ellipticity of Harmonics** This module calculates the ellipticity of high-order harmonics based on the provided total current file. 
  **Phase Analyzing** This module analyzes the phase of high-order harmonics from the total current file. 
  
**Attosecond Pulse**  
  **Attosecond Pulse** This module generates attosecond pulses by analyzing the harmonic spectrum using two different methods.
  **Minimum Pulse Width** It is calculated by evaluating the Full Width at Half Maximum (FWHM) of the intensity profile, considering the optimal harmonic orders within a given frequency range. The time corresponding to the maximum intensity is also derived, and the MPW is computed in attoseconds (as).
  **Gabor Transform** his method uses the Gabor transform to analyze the time-frequency distribution of the total current data. The Gabor transform is applied to the time-domain components of the current, which are first adjusted by subtracting the initial value to eliminate offsets.
  
**Electron Dynamics tab**
  **Excited Electrons Over Time** This feature allows users to plot the temporal evolution of excited electrons based on the input simulation data.
  **Excited Electrons in K-Space** Allows to visualize the distribution of excited electrons across the k-space using provided input data files.

**Tool Box tab**
  **Fourier Transform (Function)** Allows users to calculate the Fourier Transform of a user-defined function.
  **Fourier Transform (Data)** Allows users to perform a Fourier Transform on time-domain data loaded from a file. The result is the frequency-domain representation of the data.
  **Inverse Fourier Transform (Function)** This feature computes the inverse Fourier Transform of a user-defined function, reconstructing the original time-domain function from its frequency-domain representation.
  **Inverse Fourier Transform (Data)** This feature analyzes frequency-domain data from a user-selected file and converts it back to the time domain using the inverse Fourier Transform.
  **Unit Conversion** This feature allows users to convert physical quantities between SI units and atomic units for various properties such as intensity, time, energy, electric field, and length.
  
**Options** 
  **Extract Data** is embedded in the parameter input window as a checkbox, and if checked, the calculated data is saved as a .txt data file in the corresponding directory.
  **Plot Settings** Much effort has been made to give the user control over the plots by adjusting things like line width, line color, etc.
  **Window Functions** There are six types of filtering methods for spectra, including "cosine_window", "gaussian", "hanning", "exp_decay", "welch", "bartlett". It will only be applied if the End-of-Pulse value is greater than zero and the corresponding checkbox is checked. The filter is applied from the end of the pulse, so if the user has set the End-of-Pulse value to 10%, for example, the last 10% of the pulse will be filtered. For more information on how to apply the filter, see utils/window_func.py.

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

