# tool_box/unit.py

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
import webbrowser
import plotly.graph_objects as go
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
#--------------------------------
from attoscience_studio.helper_functions.constants import PhysicalConstants, AtomicUnits
c_light = PhysicalConstants.c_light
hbar = PhysicalConstants.hbar
inverse_alpha_fine = PhysicalConstants.inverse_alpha_fine
elcharge = PhysicalConstants.elcharge
elmass = PhysicalConstants.elmass
r_Bohr = AtomicUnits.r_Bohr
conversion_factor = 3.50944758*1.0e16 # Unit of energy flux (intensity) ==  for peak E field at 1 a.u.
##----------------------------------------------------
class UnitConversionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Unit Conversion")
        self.setFixedSize(300, 300)

        main_layout = QVBoxLayout()

        # Section: Input value
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Enter value:"))
        self.value_input = QLineEdit()
        input_layout.addWidget(self.value_input)
        main_layout.addLayout(input_layout)
        
        # Section: Conversion type dropdown
        main_layout.addWidget(QLabel("Select conversion type:"))
        self.conversion_type = QComboBox()
        self.conversion_type.addItems([
            "Intensity (W/cm²)", 
            "Time", 
            "Energy",
            "Electric Field",
            "Length"
        ])
        main_layout.addWidget(self.conversion_type)

        # Section: Direction of conversion (two-way) in a GroupBox
        direction_groupbox = QGroupBox("Conversion Direction")
        direction_layout = QVBoxLayout()
        
        self.to_atomic_radio = QRadioButton("SI to Atomic")
        self.to_atomic_radio.setChecked(True)
        self.to_si_radio = QRadioButton("Atomic to SI")
        
        direction_layout.addWidget(self.to_atomic_radio)
        direction_layout.addWidget(self.to_si_radio)
        
        direction_groupbox.setLayout(direction_layout)
        main_layout.addWidget(direction_groupbox)

        # Group the radio buttons
        self.direction_group = QButtonGroup()
        self.direction_group.addButton(self.to_atomic_radio)
        self.direction_group.addButton(self.to_si_radio)

        # Section: Convert button
        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.on_submit_unit_conversion)
        self.convert_button.setFixedHeight(40)
        main_layout.addWidget(self.convert_button)

        self.setLayout(main_layout)

    def on_submit_unit_conversion(self):
        try:
            value = float(self.value_input.text())
            conversion_type = self.conversion_type.currentText()
            to_atomic = self.to_atomic_radio.isChecked()

            units = {
                "Intensity (W/cm²)": ("W/cm²", "atomic units"),
                "Time": ("s", "atomic units"),
                "Energy": ("eV", "atomic units"),
                "Electric Field": ("V/m", "atomic units"),
                "Length": ("m", "atomic units")
            }

            if conversion_type == "Intensity (W/cm²)":
                result = value / conversion_factor if to_atomic else value * conversion_factor
            elif conversion_type == "Time":
                TIMEau = (inverse_alpha_fine ** 2) * hbar / (elmass * c_light ** 2)
                result = value / TIMEau if to_atomic else value * TIMEau
            elif conversion_type == "Energy":
                ENERGY_factor = 27.2113962  # 1 a.u. = 27.2113962 eV
                result = value / ENERGY_factor if to_atomic else value * ENERGY_factor
            elif conversion_type == "Electric Field":
                EFIELDau = hbar ** 2 / (elmass * r_Bohr ** 3 * elcharge)
                result = value / EFIELDau if to_atomic else value * EFIELDau
            elif conversion_type == "Length":
                LENGTHau = r_Bohr
                result = value / LENGTHau if to_atomic else value * LENGTHau
            else:
                result = "Invalid conversion type."
                QMessageBox.warning(self, "Error", "Invalid conversion type selected.")
                return

            unit = units[conversion_type][1] if to_atomic else units[conversion_type][0]
            
            QMessageBox.information(self, "Conversion Result", f"Converted value: {result:.4e} {unit}")

        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
