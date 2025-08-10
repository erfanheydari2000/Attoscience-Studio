# tool_box/ftfunc.py

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
#### (1)  FT from Function (Continuous Time and Frequency)      ####
####################################################################        
def get_x_FT(x, f, k):
    x_FT_integrand_real = lambda t: np.real(x(t, k) * np.exp(-2 * np.pi * 1j * f * t))
    x_FT_integrand_comp = lambda t: np.imag(x(t, k) * np.exp(-2 * np.pi * 1j * f * t))
    
    x_FT_real = quad(x_FT_integrand_real, -np.inf, np.inf, limit=200)[0]
    x_FT_comp = quad(x_FT_integrand_comp, -np.inf, np.inf, limit=200)[0]
    
    return x_FT_real + 1j * x_FT_comp

def ft_func_plot(x, f, k):
    x_FT = np.vectorize(get_x_FT)(x, f, k)
            
    plt.plot(f, np.abs(x_FT))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$|\hat{x}(f)|$')
    plt.title('Fourier Transform of User-defined Function')
    plt.show()

allowed_names = {
    'np': np,
    'sin': np.sin,
    'cos': np.cos,
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
    'tan': np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    'sinh': np.sinh,
    'cosh': np.cosh,
    'tanh': np.tanh,
    'pi': np.pi,
    'e': np.e
}

class InputDialogFTFunc(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fourier Transform Input Dialog")
        self.setMinimumWidth(750) 
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.func_input = QLineEdit(self)
        self.func_input.setPlaceholderText("Enter function of t and k (e.g., np.exp(-k * t**2) * np.sin(k*t) * t**4)")
        form_layout.addRow("Function x(t, k):", self.func_input)
        
        self.k_input = QLineEdit(self)
        self.k_input.setPlaceholderText("Enter value for k (e.g., 2)")
        form_layout.addRow("Value of k:", self.k_input)
        
        self.f_input = QLineEdit(self)
        self.f_input.setPlaceholderText("Enter frequency range (e.g., -40,40)")
        form_layout.addRow("Frequency range (comma-separated):", self.f_input)
        
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.on_submit_ft_func)
        
        layout.addLayout(form_layout)
        layout.addWidget(submit_button)

    def on_submit_ft_func(self):
        try:
            func_str = self.func_input.text()
            k = float(self.k_input.text())
            f_range = list(map(float, self.f_input.text().split(',')))
            f = np.linspace(f_range[0], f_range[1], 100)
            
            def x(t, k):
                local_vars = {"t": t, "k": k}
                return eval(func_str, {"__builtins__": None}, {**allowed_names, **local_vars})
            
            self.accept()
            
            result = (x, f, k)
            ft_func_plot(*result)
            
            QMessageBox.information(self, "Success", "Function processed and plotted successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")                             

