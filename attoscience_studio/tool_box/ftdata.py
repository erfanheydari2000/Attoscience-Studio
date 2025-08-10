# tool_box/ftdata.py

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
from matplotlib.widgets import Slider, Button
##----------------------------------------------------
######################################################################
####    (2)  FT from Data (Continuous Time and Frequency)         ####
######################################################################
from scipy.interpolate import interp1d
def read_ft_data(file_path):
    data = np.loadtxt(file_path)
    t = data[:, 1];x_t = data[:, 2]
    
    x_t_interp = interp1d(t, x_t, kind='cubic', fill_value=0, bounds_error=False)
    
    return x_t_interp, t

def get_x_FT(x_t_interp, t, f):
    real_integrand = lambda t: np.real(x_t_interp(t) * np.exp(-2 * np.pi * 1j * f * t))
    comp_integrand = lambda t: np.imag(x_t_interp(t) * np.exp(-2 * np.pi * 1j * f * t))
    
    real_part = quad(real_integrand, t[0], t[-1], limit=100)[0]
    comp_part = quad(comp_integrand, t[0], t[-1], limit=100)[0]
    
    return real_part + 1j * comp_part

def ft_data_connector(f, status_label=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select data file in time domain")
    if file_path:
        x_t_interp, t = read_ft_data(file_path)
        
        def get_x_FT_with_t(f):
            return get_x_FT(x_t_interp, t, f)
        
        x_FT = np.vectorize(get_x_FT_with_t)(f)
        
        ft_data_plot(f, x_FT)
        if status_label:
            status_label.setText("FT plotted successfully!")
        else:
            print("FT plotted successfully!")
    else:
        if status_label:
            status_label.setText("Error: file not found or no file selected.")
        else:
            print("Error: file not found or no file selected.")
   
def ft_data_plot(f, x_FT):
    Abs_x_FT = np.abs(x_FT)
    plt.plot(f, Abs_x_FT)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|$hat{x}$(f)|')
    plt.show()

class InputDialogFTData(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fourier Transform Input Dialog")
        self.setMinimumWidth(550) 
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.f_input = QLineEdit(self)
        self.f_input.setPlaceholderText("Enter frequency range (e.g., -40,40)")
        form_layout.addRow("Frequency range (comma-separated):", self.f_input)
        
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.on_submit_ft_data)
        
        layout.addLayout(form_layout)
        layout.addWidget(submit_button)

    def on_submit_ft_data(self):
        try:
            f_range = list(map(float, self.f_input.text().split(',')))
            f = np.linspace(f_range[0], f_range[1], 100)
            if len(f_range) != 2:
                QMessageBox.warning(self, "Error", "Please enter two numbers separated by a comma.")
                return
            
            self.accept()
            
            ft_data_connector(f, status_label=None)
            
            QMessageBox.information(self, "Success", "Function processed and plotted successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")                             

