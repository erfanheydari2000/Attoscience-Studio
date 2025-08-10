# tool_box/iftfunc.py

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
#### (4)  Inverse FT from Function (Continuous Time and Frequency) ####
####################################################################
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
def get_x_inverse_FT(X, t, k):
    x_inv_FT_integrand_real = lambda f: np.real(X(f, k) * np.exp(2 * np.pi * 1j * f * t))
    x_inv_FT_integrand_comp = lambda f: np.imag(X(f, k) * np.exp(2 * np.pi * 1j * f * t))
    
    x_inv_FT_real = quad(x_inv_FT_integrand_real, -np.inf, np.inf, limit=200)[0]
    x_inv_FT_comp = quad(x_inv_FT_integrand_comp, -np.inf, np.inf, limit=200)[0]
    
    return x_inv_FT_real + 1j * x_inv_FT_comp

def inverse_ft_func_plot(X, t, k):
    x_inv_FT = np.vectorize(get_x_inverse_FT)(X, t, k)
    
    plt.plot(t, np.abs(x_inv_FT))
    plt.xlabel('Time [s]')
    plt.ylabel('$|x(t)|$')
    plt.title('Inverse Fourier Transform of User-defined Function')
    plt.show()

class InputDialogInverseFTFunc(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inverse Fourier Transform Input Dialog")
        self.setMinimumWidth(700)
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.func_input = QLineEdit(self)
        self.func_input.setPlaceholderText("Enter function of f and k (e.g., np.exp(-k * f**2) * np.sin(k*f))")
        form_layout.addRow("Function X(f, k):", self.func_input)
        
        self.k_input = QLineEdit(self)
        self.k_input.setPlaceholderText("Enter value for k (e.g., 2)")
        form_layout.addRow("Value of k:", self.k_input)
        
        self.t_input = QLineEdit(self)
        self.t_input.setPlaceholderText("Enter time range (e.g., -10,10)")
        form_layout.addRow("Time range (comma-separated):", self.t_input)
        
        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.on_submit_inverse_ft_func)
        
        layout.addLayout(form_layout)
        layout.addWidget(submit_button)

    def on_submit_inverse_ft_func(self):
        try:
            func_str = self.func_input.text()
            k = float(self.k_input.text())
            t_range = list(map(float, self.t_input.text().split(',')))
            t = np.linspace(t_range[0], t_range[1], 100)
            
            def X(f, k):
                local_vars = {"f": f, "k": k}
                return eval(func_str, {"__builtins__": None}, {**allowed_names, **local_vars})
            
            self.accept()
            
            result = (X, t, k)
            inverse_ft_func_plot(*result)
            
            QMessageBox.information(self, "Success", "Function processed and plotted successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")
