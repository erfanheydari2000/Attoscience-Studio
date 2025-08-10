# tool_box/iftdata.py

# Copyright (C) 2024-2025 Erfan Heydari
#
# This file is part of the Attoscience Studio.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os, sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar,
                             QRadioButton, QButtonGroup, QScrollArea, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication
from numpy import trapz
from scipy.integrate import trapezoid, quad
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
####################################################################
#### (3)  Inverse FT from Data (Continuous Frequency and Time) ####
####################################################################
from scipy.interpolate import interp1d
def read_ift_data(file_path):
    data = np.loadtxt(file_path)
    f = data[:, 1]; x_f = data[:, 2]
    
    x_f_interp = interp1d(f, x_f, kind='cubic', fill_value=0, bounds_error=False)
    
    return x_f_interp, f

def get_x_IFT(x_f_interp, f, t):
    real_integrand = lambda f: np.real(x_f_interp(f) * np.exp(2 * np.pi * 1j * f * t))
    comp_integrand = lambda f: np.imag(x_f_interp(f) * np.exp(2 * np.pi * 1j * f * t))
    
    real_part = quad(real_integrand, f[0], f[-1], limit=100)[0]
    comp_part = quad(comp_integrand, f[0], f[-1], limit=100)[0]
    
    return real_part + 1j * comp_part

def ift_data_connector(t, status_label=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select data file in frequency domain")
    if file_path:
        x_f_interp, f = read_ift_data(file_path)
        
        def get_x_IFT_with_f(t):
            return get_x_IFT(x_f_interp, f, t)
        
        x_IFT = np.vectorize(get_x_IFT_with_f)(t)
        
        ift_data_plot(t, x_IFT)
        if status_label:
            status_label.setText("IFT plotted successfully!")
        else:
            print("IFT plotted successfully!")
    else:
        if status_label:
            status_label.setText("Error: file not found or no file selected.")
        else:
            print("Error: file not found or no file selected.")

def ift_data_plot(t, x_IFT):
    Abs_x_IFT = np.abs(x_IFT)
    plt.plot(t, Abs_x_IFT)
    plt.xlabel('Time [s]')
    plt.ylabel('|$x$(t)|')
    plt.show()

class InputDialogIFTData(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inverse Fourier Transform Input Dialog")
        self.setMinimumWidth(550) 
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.t_input = QLineEdit(self)
        self.t_input.setPlaceholderText("Enter time range (e.g., -1,1)")
        form_layout.addRow("Time range (comma-separated):", self.t_input)
        
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.on_submit_ift_data)
        
        layout.addLayout(form_layout)
        layout.addWidget(submit_button)

    def on_submit_ift_data(self):
        try:
            t_range = list(map(float, self.t_input.text().split(',')))
            t = np.linspace(t_range[0], t_range[1], 100)
            if len(t_range) != 2:
                QMessageBox.warning(self, "Error", "Please enter two numbers separated by a comma.")
                return
            
            self.accept()
            
            ift_data_connector(t, status_label=None)
            
            QMessageBox.information(self, "Success", "Function processed and plotted successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")
