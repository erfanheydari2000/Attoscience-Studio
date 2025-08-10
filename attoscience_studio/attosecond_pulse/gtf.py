# attosecond_pulse/gtf.py

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
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar, QStyle,
                             QRadioButton, QButtonGroup, QScrollArea, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QDoubleSpinBox, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton, QScrollArea, QSlider, QGraphicsOpacityEffect)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QIntValidator
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication, QPropertyAnimation
from numpy import trapz
from scipy.integrate import trapezoid, quad
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from attoscience_studio.resources_rc import *
#--------------------------------
from attoscience_studio.utils.window_func import TotalCurrentFilter
#--------------------------------
from attoscience_studio.helper_functions.constants import PhysicalConstants
Ip_HeV = PhysicalConstants.Ip_HeV
#--------------------------------
from attoscience_studio.utils.status_symbols import Symbols
##----------------------------------------------------
def read_gtf(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.size == 0:
            raise ValueError("The file is empty.")
        reading_step = 3
        t  = data[::reading_step, 1]
        dt = t[1] - t[0]
        jx = data[::reading_step, 2]
        jx = jx - jx[0]
        jy = data[::reading_step, 3]
        jy = jy - jy[0]
    
    except ValueError as e:
        QMessageBox.warning(None, "Error", str(e))
        return
    
    return t, dt, jx, jy
##----------------------------------------------------
def GTF_core(t, dt, hx, hy, w, sigma_gabor):
    L = 6 * sigma_gabor
    half_kernel = int(np.ceil(L / dt))
    n_kernel = 2 * half_kernel + 1

    tau = np.arange(-half_kernel, half_kernel + 1) * dt
    kernel = np.exp(-0.5 * tau**2 / sigma_gabor**2)
    
    Ax = np.zeros((len(t), len(w)), dtype=complex)
    Ay = np.zeros((len(t), len(w)), dtype=complex)
    
    for iw, w_i in enumerate(w):
        Hx_w = hx * np.exp(-1j * w_i * t)
        Hy_w = hy * np.exp(-1j * w_i * t)
        
        Ax[:, iw] = fftconvolve(Hx_w, kernel, mode='same') * dt
        Ay[:, iw] = fftconvolve(Hy_w, kernel, mode='same') * dt
    
    return Ax, Ay
##----------------------------------------------------
def time_frequency(t, dt, jx, jy, lambda0_nm, qstart, qend, g_factor, filtering, window_func):
    w0 = 45.563 / lambda0_nm
    T0 = 2 * np.pi / w0
    dw = w0/2
    
    sigma_gabor = T0 / g_factor

    #---------------------------------------
    EoP = 1.0 - filtering/100
    filter_method = window_func[0]
    WF_param = window_func[1]
    #----
    exponent = 0.0
    sigma = 0.0
    decay_rate = 0.0
    if filter_method=="cosine":
        exponent = WF_param
    elif filter_method=="Gaussian":
        sigma = WF_param
    elif filter_method=="Exponential Decay":
        decay_rate = WF_param
    
    #---------------CALL--------------------
    filter_obj = TotalCurrentFilter(method=filter_method, EoP=EoP, exponent=exponent, sigma=sigma, decay_rate=decay_rate)
    hx, hy = filter_obj.apply_filter(t, jx, jy)
    #---------------------------------------
    
    w = np.arange(qstart * w0, qend * w0 + dw, dw)
    www = (w * Ip_HeV)

    start_time = time.perf_counter()
    #---------------CALL--------------------
    Ax, Ay = GTF_core(t, dt, hx, hy, w, sigma_gabor)
    #---------------------------------------
    end_time = time.perf_counter()
    print(f"GTF_core() execution time: {end_time - start_time:.4f} seconds")

    Ax_abs = np.abs(Ax)
    Ax_log = np.log10(Ax_abs)
    Ay_abs = np.abs(Ay)
    Ay_log = np.log10(Ay_abs)
    Atot_abs = np.sqrt(Ax_abs ** 2 + Ay_abs ** 2)
    Atot_log = np.log10(Atot_abs)
    
    return Ax_log, Ay_log, Atot_log, t, T0, w, w0, www, sigma_gabor

##----------------------------------------------------
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")

def time_frequency_connector(lambda0_nm, qstart, qend, g_factor, filtering, selected_components, window_func, extract_data_option, plot_settings, ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select total_current file")
    if "total_current" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the 'total_current' file.")
        return        
    
    if file_path:
        try:
            start_time = time.perf_counter()
            t, dt, jx, jy = read_gtf(file_path)
            end_time = time.perf_counter()
            print(f"read_gtf() execution time: {end_time - start_time:.4f} seconds")
            
            Ax_log, Ay_log, Atot_log, t, T0, w, w0, www, sigma_gabor = time_frequency(t, dt, jx, jy ,lambda0_nm, qstart, qend, g_factor, filtering, window_func)

            plot_time_frequency(Ax_log, Ay_log, Atot_log, t, T0, w, w0, www, lambda0_nm, qstart, qend, g_factor, selected_components, extract_data_option, plot_settings)

            Time_OC = t/T0
            max_Time_OC = np.max(Time_OC)
            T_SI = T0*2.418884326509*1e-17
            sigma_gabor_SI = sigma_gabor*2.418884326509*1e-17

            timestamp = datetime.now().strftime("[%H:%M:%S]")
            msg = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          HHG log!                          --\n"
                + "-" * 75 + "\n"
                ">>> HHG calculation and visualization successfully completed!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> T [a.u.]: {T0:.12e}\n"
                f">>> T [second]: {T_SI:.12e}\n"
                f">>> w0: {w0:.12e}\n"
                f">>> Max optical cycle: {max_Time_OC}\n"
                f">>> Time window [a.u.]: {sigma_gabor}\n"
                f">>> Time window [s]: {sigma_gabor_SI}\n"
                + "-" * 75
            )
            print_to_console(ipy_console, msg)

        except ValueError as e:
            QMessageBox.warning(None, "Error", str(e))
            return

##----------------------------------------------------
def plot_time_frequency(Ax_log, Ay_log, Atot_log, t, T0, w, w0, www, lambda0_nm, qstart, qend, g_factor, selected_components, extract_data_option, plot_settings):   
    ww = w/w0
    x_min = t[1]/T0
    x_max = t[-1]/T0
    if 'total' in selected_components:
        if 'extract_data' in extract_data_option:
            file_path = "tf_gabor_tot.txt"
            header = (
                f"# Driving wavelength [nm]: {lambda0_nm}\n"
                f"# T [a.u.]:                {T0}\n"
                f"# w0 [a.u.]:               {w0}\n"
                f"# Min HO:                  {qstart}\n"
                f"# Max HO:                  {qend}\n"
                f"# g_factor:                {g_factor}\n"
                "# Atot_log (Intensity [arb.u])\n"
                + '#' * 60
            )
            np.savetxt(file_path, np.column_stack((Atot_log)), header=header, comments='', fmt='%.12e')

        plt.figure()
        plt.imshow(
            Atot_log.T,
            extent=[x_min, x_max, min(ww), max(ww)],
            aspect=plot_settings.get("aspect", 'auto'),
            origin=plot_settings.get("origin", 'lower'),
            cmap=plot_settings.get("cmap", 'jet'),
            interpolation=plot_settings.get("interpolation", 'nearest'),
            vmin=plot_settings.get("vmin", None),
            vmax=plot_settings.get("vmax", None)
        )        
        plt.colorbar(label=plot_settings.get("colorbar_label", 'log_{10} (Intensity) [arb. units]'))
        plt.clim(plot_settings.get("clim_min", None), plot_settings.get("clim_max", None))
        plt.xlabel(plot_settings.get("x_label", "Time [o.c.]"))
        plt.ylabel(plot_settings.get("y_label", "Harmonic order"))
        plt.xlim(left=plot_settings.get("x_min", x_min), right=plot_settings.get("x_max", x_max))
        plt.ylim(bottom=plot_settings.get("y_min", min(ww)), top=plot_settings.get("y_max", max(ww)))
        plt.title(plot_settings.get("graph_title", "Time-Frequency Analysis (Total)"))
        plt.show()
    if 'x' in selected_components:
        if 'extract_data' in extract_data_option:
            file_path = "tf_gabor_x.txt"
            header = (
                f"# Driving wavelength [nm]: {lambda0_nm}\n"
                f"# T [a.u.]:                {T0}\n"
                f"# w0 [a.u.]:               {w0}\n"
                f"# Min HO:                  {qstart}\n"
                f"# Max HO:                  {qend}\n"
                f"# g_factor:                {g_factor}\n"
                "# Ax_log (Intensity [arb.u])\n"
                + '#' * 60
            )
            np.savetxt(file_path, np.column_stack((Ax_log)), header=header, comments='', fmt='%.12e')

        plt.figure()
        plt.imshow(
            Ax_log.T,
            extent=[x_min, x_max, min(ww), max(ww)],
            aspect=plot_settings.get("aspect", 'auto'),
            origin=plot_settings.get("origin", 'lower'),
            cmap=plot_settings.get("cmap", 'jet'),
            interpolation=plot_settings.get("interpolation", 'nearest'),
            vmin=plot_settings.get("vmin", None),
            vmax=plot_settings.get("vmax", None)
        )                   
        plt.colorbar(label=plot_settings.get("colorbar_label", 'log_{10} (Intensity) [arb. units]'))
        plt.clim(plot_settings.get("clim_min", None), plot_settings.get("clim_max", None))
        plt.xlabel(plot_settings.get("x_label", "Time [o.c.]"))
        plt.ylabel(plot_settings.get("y_label", "Harmonic order"))
        plt.xlim(left=plot_settings.get("x_min", x_min), right=plot_settings.get("x_max", x_max))
        plt.ylim(bottom=plot_settings.get("y_min", min(ww)), top=plot_settings.get("y_max", max(ww)))
        plt.title(plot_settings.get("graph_title", "Time-Frequency Analysis (X-component)"))
        plt.show()
    if 'y' in selected_components:
        if 'extract_data' in extract_data_option:
            file_path = "tf_gabor_y.txt"
            header = (
                f"# Driving wavelength [nm]: {lambda0_nm}\n"
                f"# T [a.u.]:                {T0}\n"
                f"# w0 [a.u.]:               {w0}\n"
                f"# Min HO:                  {qstart}\n"
                f"# Max HO:                  {qend}\n"
                f"# g_factor:                {g_factor}\n"
                "# Ay_log (Intensity [arb.u])\n"
                + '#' * 60
            )
            np.savetxt(file_path, np.column_stack((Ay_log)), header=header, comments='', fmt='%.12e')

        plt.figure()
        plt.imshow(
            Ay_log.T,
            extent=[x_min, x_max, min(ww), max(ww)],
            aspect=plot_settings.get("aspect", 'auto'),
            origin=plot_settings.get("origin", 'lower'),
            cmap=plot_settings.get("cmap", 'jet'),
            interpolation=plot_settings.get("interpolation", 'nearest'),
            vmin=plot_settings.get("vmin", None),
            vmax=plot_settings.get("vmax", None)
        )                           
        plt.colorbar(label=plot_settings.get("colorbar_label", 'log_{10} (Intensity) [arb. units]'))
        plt.clim(plot_settings.get("clim_min", None), plot_settings.get("clim_max", None))
        plt.xlabel(plot_settings.get("x_label", "Time [o.c.]"))
        plt.ylabel(plot_settings.get("y_label", "Harmonic order"))
        plt.xlim(left=plot_settings.get("x_min", x_min), right=plot_settings.get("x_max", x_max))
        plt.ylim(bottom=plot_settings.get("y_min", min(ww)), top=plot_settings.get("y_max", max(ww)))
        plt.title(plot_settings.get("graph_title", "Time-Frequency Analysis (Y-component)"))
        plt.show()

##----------------------------------------------------

previous_input_time_frequency = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HHG Parameters")
        self.setFixedSize(1000, 800)
        self.setStyleSheet(self.get_modern_stylesheet())
        self.init_ui()
        self.setup_animations()
        
    def get_modern_stylesheet(self):
        return """
        QDialog {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #f8f9fa, stop: 1 #e9ecef);
            border-radius: 12px;
        }
        
        QGroupBox {
            font-weight: 600;
            font-size: 14px;
            color: #2c3e50;
            border: 2px solid #e3f2fd;
            border-radius: 8px;
            margin: 10px 0;
            padding-top: 15px;
            background: rgba(255, 255, 255, 0.8);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #1976d2;
        }
        
        QLineEdit {
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
            background: white;
            selection-background-color: #2196f3;
        }
        
        QLineEdit:focus {
            border: 2px solid #2196f3;
            /*box-shadow: 0 0 8px rgba(33, 150, 243, 0.3);*/
        }
        
        QSpinBox, QDoubleSpinBox {
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 8px;
            background: white;
            font-size: 13px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #2196f3;
        }
        
        QPushButton {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #2196f3, stop: 1 #1976d2);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 14px;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #42a5f5, stop: 1 #2196f3);
            /*transform: translateY(-1px);*/
        }
        
        QPushButton:pressed {
            background: #1565c0;
        }
        
        QCheckBox {
            font-size: 13px;
            color: #2c3e50;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #bdbdbd;
            border-radius: 4px;
            background: white;
        }
        
        QCheckBox::indicator:checked {
            background: #2196f3;
            border: 2px solid #2196f3;
            image: url(:/icons/select_check_box_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png); /* !!! */
        }
        
        QLabel {
            color: #37474f;
            font-size: 13px;
            font-weight: 500;
        }
        
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollBar {
            background: transparent;
            width: 20px;
            margin: 0px;
            border-radius: 10px;

        }
        """
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        self.create_header(main_layout)
        
        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # content sections
        self.create_required_section(content_layout)
        self.create_optional_section(content_layout)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # button section
        self.create_button_section(main_layout)
        
    def create_header(self, layout):
        header_layout = QHBoxLayout()
        
        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon_label.setPixmap(icon.pixmap(32, 32))  # Set size as needed
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title = QLabel("Gabor transform")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")
        
        subtitle = QLabel("Configure parameters for HHG calculation from total current")
        subtitle.setStyleSheet("font-size: 14px; color: #666; margin: 0;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(2)
        
        header_layout.addWidget(icon_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e0e0e0; margin: 10px 0;")
        layout.addWidget(line)
    
    def create_required_section(self, layout):
        required_group = QGroupBox("Essential Parameters")
        required_layout = QVBoxLayout()
        required_layout.setSpacing(20)

        ##- Basic Parameters Section ---------
        basic_params_group = QGroupBox("Basic Parameters")
        basic_params_layout = QFormLayout()
        basic_params_layout.setSpacing(12)
        basic_params_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # (1) Driving Wavelength
        lambda0_container = QWidget()
        lambda0_layout = QHBoxLayout(lambda0_container)
        lambda0_layout.setContentsMargins(0, 0, 0, 0)
    
        self.lambda0_entry = QLineEdit()
        self.lambda0_entry.setPlaceholderText("Enter driving wavelength (e.g., 2000)")
        self.lambda0_entry.setText(str(previous_input_time_frequency.get("lambda0_nm", "")))
        self.lambda0_entry.setMaxLength(10)

    
        lambda0_unit_label = QLabel("nm")
        lambda0_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
    
        lambda0_layout.addWidget(self.lambda0_entry)
        lambda0_layout.addWidget(lambda0_unit_label)
        
        basic_params_layout.addRow("Driving Wavelength:", lambda0_container)

        # (2) Minimum Harmonic Order
        qstart_container = QWidget()
        qstart_layout = QHBoxLayout(qstart_container)
        qstart_layout.setContentsMargins(0, 0, 0, 0)
    
        self.qstart_entry = QLineEdit()
        self.qstart_entry.setPlaceholderText("Enter minimum harmonic order (e.g., 10)")
        self.qstart_entry.setText(str(previous_input_time_frequency.get("qstart", "")))
        self.qstart_entry.setMaxLength(10)
        
        qstart_unit_label = QLabel("HO")
        qstart_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qstart_layout.addWidget(self.qstart_entry)
        qstart_layout.addWidget(qstart_unit_label)
    
        basic_params_layout.addRow("Minimum Harmonic Order:", qstart_container)
    
        # (3) Maximum Harmonic Order
        qend_container = QWidget()
        qend_layout = QHBoxLayout(qend_container)
        qend_layout.setContentsMargins(0, 0, 0, 0)
    
        self.qend_entry = QLineEdit()
        self.qend_entry.setPlaceholderText("Enter maximum harmonic order (e.g., 20)")
        self.qend_entry.setText(str(previous_input_time_frequency.get("qend", "")))
        self.qend_entry.setMaxLength(10)
        
        qend_unit_label = QLabel("HO")
        qend_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qend_layout.addWidget(self.qend_entry)
        qend_layout.addWidget(qend_unit_label)
    
        basic_params_layout.addRow("Maximum Harmonic Order:", qend_container)

        # (4) g_factor
        g_factor_container = QWidget()
        g_factor_layout = QHBoxLayout(g_factor_container)
        g_factor_layout.setContentsMargins(0, 0, 0, 0)
    
        self.g_factor_entry = QLineEdit()
        self.g_factor_entry.setPlaceholderText("Enter g_factor (e.g., 3)")
        self.g_factor_entry.setText(str(previous_input_time_frequency.get("g_factor", "")))
        self.g_factor_entry.setMaxLength(10)
        
        TAU = Symbols.TAU
        label = f"{Symbols.TAU} = T / g_factor"
        g_factor_unit_label = QLabel(label) #'\u03C4'
        g_factor_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        g_factor_layout.addWidget(self.g_factor_entry)
        g_factor_layout.addWidget(g_factor_unit_label)
    
        basic_params_layout.addRow("G_factor:", g_factor_container)

        #------
        basic_params_group.setLayout(basic_params_layout)
        required_layout.addWidget(basic_params_group)

        ##- Display Options Section -------------------------------------
        display_options_group = QGroupBox("Display Options")
        display_options_layout = QVBoxLayout()
        display_options_layout.setSpacing(15)

        # Components selection
        components_subgroup = QGroupBox("Components")
        components_layout = QHBoxLayout()
        components_layout.setContentsMargins(10, 0, 0, 10)
        components_layout.setSpacing(15)
    
        self.x_component_checkbox = QCheckBox("X Component")
        self.y_component_checkbox = QCheckBox("Y Component")
        self.total_checkbox = QCheckBox("Total")
    
        components_layout.addWidget(self.x_component_checkbox)
        components_layout.addWidget(self.y_component_checkbox)
        components_layout.addWidget(self.total_checkbox)
        components_layout.addStretch()
    
        components_subgroup.setLayout(components_layout)
        display_options_layout.addWidget(components_subgroup)

        # Window Function subgroup --------------------------
        # method (str): Filtering method ("cosine_window", "gaussian", "hanning", "exp_decay" , "welch", "bartlett")
        windowFunc_subgroup = QGroupBox("Window Function")
        windowFunc_main_layout = QVBoxLayout()
        windowFunc_main_layout.setSpacing(10)
    
        # End-of-pulse filtering value at the top
        filter_form_layout = QFormLayout()
        filter_container = QWidget()
        filter_layout = QHBoxLayout(filter_container)
        filter_layout.setContentsMargins(0, 0, 0, 0)

        self.filter_entry = QLineEdit()
        self.filter_entry.setPlaceholderText("0.0")
        self.filter_entry.setText(str(previous_input_time_frequency.get("filtering", "0.0")))
        self.filter_entry.setMaxLength(4)
        
        filter_unit_label = QLabel("%")
        filter_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")

        filter_layout.addWidget(self.filter_entry)
        filter_layout.addWidget(filter_unit_label)
    
        filter_form_layout.addRow("End-of-pulse:", filter_container)
        windowFunc_main_layout.addLayout(filter_form_layout)
        #--------------------------------------------
        # Window function methods in 2 columns, 3 rows
        methods_grid_layout = QGridLayout()
        methods_grid_layout.setSpacing(10)

        #----------------------
        # Left column - methods with parameters
        # Row 0: Cosine + nn parameter
        cosine_container = QWidget()
        cosine_layout = QHBoxLayout(cosine_container)
        cosine_layout.setContentsMargins(110, 0, 0, 0)
        cosine_layout.setSpacing(8)    
    
        self.windowFunc_cosine_checkbox = QCheckBox("Cosine:       ")
        
        self.cosine_nn_entry = QLineEdit()
        self.cosine_nn_entry.setPlaceholderText("Exponent") #***********
        #self.cosine_nn_entry.setText(str(previous_input_spectrum.get("nn", "0")))
        self.cosine_nn_entry.setMaximumWidth(100)
        self.cosine_nn_entry.setEnabled(False)
        self.cosine_nn_entry.setMaxLength(2)
    
        cosine_layout.addWidget(self.windowFunc_cosine_checkbox)
        cosine_layout.addWidget(self.cosine_nn_entry)
        cosine_layout.addStretch()

        methods_grid_layout.addWidget(cosine_container, 0, 0)

        # Row 1: Gaussian + sigma parameter
        gaussian_container = QWidget()
        gaussian_layout = QHBoxLayout(gaussian_container)
        gaussian_layout.setContentsMargins(110, 0, 0, 0)
        gaussian_layout.setSpacing(8)

        self.windowFunc_gaussian_checkbox = QCheckBox("Gaussian:   ")
        
        self.gaussian_sigma_entry = QLineEdit()
        self.gaussian_sigma_entry.setPlaceholderText("sigma")
        #self.gaussian_sigma_entry.setText(str(previous_input_spectrum.get("sigma", "0.0")))
        self.gaussian_sigma_entry.setMaximumWidth(100)
        self.gaussian_sigma_entry.setEnabled(False)
        self.gaussian_sigma_entry.setMaxLength(6)
        
        gaussian_layout.addWidget(self.windowFunc_gaussian_checkbox)
        gaussian_layout.addWidget(self.gaussian_sigma_entry)
        gaussian_layout.addStretch()

        methods_grid_layout.addWidget(gaussian_container, 1, 0)

        # Row 2: Exponential decay + decay_rate parameter
        exp_decay_container = QWidget()
        exp_decay_layout = QHBoxLayout(exp_decay_container)
        exp_decay_layout.setContentsMargins(110, 0, 0, 0)
        exp_decay_layout.setSpacing(8)

        self.windowFunc_exp_decay_checkbox = QCheckBox("Exp. Decay:")
        
        self.exp_decay_rate_entry = QLineEdit()
        self.exp_decay_rate_entry.setPlaceholderText("rate")
        #self.exp_decay_rate_entry.setText(str(previous_input_spectrum.get("rate", "0.0")))
        self.exp_decay_rate_entry.setMaximumWidth(100)
        self.exp_decay_rate_entry.setEnabled(False)
        self.exp_decay_rate_entry.setMaxLength(6)

        exp_decay_layout.addWidget(self.windowFunc_exp_decay_checkbox)
        exp_decay_layout.addWidget(self.exp_decay_rate_entry)
        exp_decay_layout.addStretch()
        
        methods_grid_layout.addWidget(exp_decay_container, 2, 0)
        #---------------------- (left, top, right, bottom)
        # Right column - simple methods without parameters
        # Right column container to add margin
        right_column_container = QWidget()
        right_column_layout = QVBoxLayout(right_column_container)
        right_column_layout.setContentsMargins(0, 0, 110, 0)
        right_column_layout.setSpacing(30)

        self.windowFunc_hanning_checkbox = QCheckBox("Hanning")
        self.windowFunc_welch_checkbox = QCheckBox("Welch")
        self.windowFunc_bartlett_checkbox = QCheckBox("Bartlett")

        right_column_layout.addWidget(self.windowFunc_hanning_checkbox)
        right_column_layout.addWidget(self.windowFunc_welch_checkbox)
        right_column_layout.addWidget(self.windowFunc_bartlett_checkbox)
        right_column_layout.addStretch()

        methods_grid_layout.addWidget(right_column_container, 0, 1, 3, 1)
        #================
        self.windowFunc_group = QButtonGroup()
        self.windowFunc_group.setExclusive(True)

        self.windowFunc_group.addButton(self.windowFunc_cosine_checkbox)
        self.windowFunc_group.addButton(self.windowFunc_gaussian_checkbox)
        self.windowFunc_group.addButton(self.windowFunc_exp_decay_checkbox)
        self.windowFunc_group.addButton(self.windowFunc_hanning_checkbox)
        self.windowFunc_group.addButton(self.windowFunc_welch_checkbox)
        self.windowFunc_group.addButton(self.windowFunc_bartlett_checkbox)
        #================
        self.windowFunc_cosine_checkbox.toggled.connect(self.cosine_nn_entry.setEnabled)
        self.windowFunc_gaussian_checkbox.toggled.connect(self.gaussian_sigma_entry.setEnabled)
        self.windowFunc_exp_decay_checkbox.toggled.connect(self.exp_decay_rate_entry.setEnabled)

        windowFunc_main_layout.addLayout(methods_grid_layout)
        windowFunc_subgroup.setLayout(windowFunc_main_layout)
        display_options_layout.addWidget(windowFunc_subgroup)

        # Additional options
        options_subgroup = QGroupBox("Additional Options")
        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(10, 10, 10, 10)
    
        self.extract_data_checkbox = QCheckBox("Extract Data")
    
        options_layout.addWidget(self.extract_data_checkbox)
        options_layout.addStretch()
    
        ##-------
        options_subgroup.setLayout(options_layout)
        display_options_layout.addWidget(options_subgroup)

        display_options_group.setLayout(display_options_layout)
        required_layout.addWidget(display_options_group)

        ##--------------------------------------------------------
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)

    
    def create_optional_section(self, layout):
        self.plot_options_checkbox = QCheckBox("Advanced Plot Customization") 
        self.plot_options_checkbox.setStyleSheet("font-size: 14px; font-weight: 600; color: #1976d2;")

        layout.addWidget(self.plot_options_checkbox)
        
        self.advanced_group = QGroupBox("Plot Appearance Settings")
        self.advanced_group.setVisible(False)
        
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background: white;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #e0e0e0;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #2196f3;
                color: white;
            }
        """)
        #############
        # Axes tab
        axes_tab = self.create_axes_tab()
        tab_widget.addTab(axes_tab, QIcon(":/icons/finance_27dp_000000_FILL0_wght400_GRAD0_opsz24.png"), "Axes")
        tab_widget.setIconSize(QSize(20, 20))
        #############
        # Styling tab
        styling_tab = self.create_styling_tab()
        tab_widget.addTab(styling_tab, QIcon(":/icons/stylus_note_27dp_000000_FILL0_wght400_GRAD0_opsz24.png"), "Styling")
        #############
        # Labels tab 
        labels_tab = self.create_labels_tab()
        tab_widget.addTab(labels_tab, QIcon(":/icons/label_27dp_000000_FILL0_wght400_GRAD0_opsz24.png"), "Labels")
        #############
        
        advanced_layout = QVBoxLayout(self.advanced_group)
        advanced_layout.addWidget(tab_widget)
        layout.addWidget(self.advanced_group)
        
        ##########################
        self.plot_options_checkbox.stateChanged.connect(self.toggle_advanced_options)
        ##########################
        
    def create_axes_tab(self):
        axes_widget = QWidget()
        axes_layout = QFormLayout(axes_widget)
        axes_layout.setSpacing(12)
        
        # X-axis range
        x_range_widget = QWidget()
        x_range_layout = QHBoxLayout(x_range_widget)
        x_range_layout.setContentsMargins(0, 0, 0, 0)
        
        self.x_min_entry = QLineEdit()
        self.x_min_entry.setPlaceholderText("Min")
        self.x_max_entry = QLineEdit()
        self.x_max_entry.setPlaceholderText("Max")
        
        x_range_layout.addWidget(self.x_min_entry)
        x_range_layout.addWidget(QLabel("to"))
        x_range_layout.addWidget(self.x_max_entry)
        
        axes_layout.addRow("X-axis Range:", x_range_widget)
        
        # Y-axis range
        y_range_widget = QWidget()
        y_range_layout = QHBoxLayout(y_range_widget)
        y_range_layout.setContentsMargins(0, 0, 0, 0)
        
        self.y_min_entry = QLineEdit()
        self.y_min_entry.setPlaceholderText("Min")
        self.y_max_entry = QLineEdit()
        self.y_max_entry.setPlaceholderText("Max")
        
        y_range_layout.addWidget(self.y_min_entry)
        y_range_layout.addWidget(QLabel("to"))
        y_range_layout.addWidget(self.y_max_entry)
        
        axes_layout.addRow("Y-axis Range:", y_range_widget)
        
        # clim
        clim_range_widget = QWidget()
        clim_range_layout = QHBoxLayout(clim_range_widget)
        clim_range_layout.setContentsMargins(0, 0, 0, 0)
        
        self.clim_min_entry = QLineEdit()
        self.clim_min_entry.setPlaceholderText("Min")
        self.clim_max_entry = QLineEdit()
        self.clim_max_entry.setPlaceholderText("Max")
        
        clim_range_layout.addWidget(self.clim_min_entry)
        clim_range_layout.addWidget(QLabel("to"))
        clim_range_layout.addWidget(self.clim_max_entry)
        
        axes_layout.addRow("clim:", clim_range_widget)
    
        return axes_widget
    
    def create_styling_tab(self):
        styling_widget = QWidget()
        styling_layout = QFormLayout(styling_widget)
        styling_layout.setSpacing(12)

        # Colormap
        self.cmap_combo = QComboBox()
        colormaps = [
            "jet", "plasma", "inferno", "magma", "cividis", "gray",
            "viridis", "hot", "cool", "spring", "summer", "autumn", "winter"
        ]
        self.cmap_combo.addItems(colormaps)
        styling_layout.addRow("Colormap:", self.cmap_combo)

        # Aspect ratio
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["auto", "equal"])
        styling_layout.addRow("Aspect Ratio:", self.aspect_combo)

        # Origin
        self.origin_combo = QComboBox()
        self.origin_combo.addItems(["upper", "lower"])
        styling_layout.addRow("Origin:", self.origin_combo)

        # Interpolation
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems([
            "none", "nearest", "bilinear", "bicubic", "spline16",
            "spline36", "hanning", "hamming", "hermite", "kaiser", 
            "quadric", "catrom", "gaussian", "bessel", "mitchell", 
            "sinc", "lanczos"
        ])
        self.interpolation_combo.setCurrentText("nearest")
        styling_layout.addRow("Interpolation:", self.interpolation_combo)

        # Color limits
        vlim_widget = QWidget()
        vlim_layout = QHBoxLayout(vlim_widget)
        vlim_layout.setContentsMargins(0, 0, 0, 0)

        self.vmin_entry = QLineEdit()
        self.vmin_entry.setPlaceholderText("vmin")

        self.vmax_entry = QLineEdit()
        self.vmax_entry.setPlaceholderText("vmax")

        vlim_layout.addWidget(self.vmin_entry)
        vlim_layout.addWidget(QLabel("to"))
        vlim_layout.addWidget(self.vmax_entry)

        styling_layout.addRow("Color Limits:", vlim_widget)

        return styling_widget
        
    def create_labels_tab(self):
        labels_widget = QWidget()
        labels_layout = QFormLayout(labels_widget)
        labels_layout.setSpacing(12)
        
        self.graph_title_entry = QLineEdit()
        self.graph_title_entry.setPlaceholderText("Enter plot title")
        labels_layout.addRow("Graph Title:", self.graph_title_entry)
        
        self.x_label_entry = QLineEdit()
        self.x_label_entry.setPlaceholderText("e.g., Harmonic order")
        labels_layout.addRow("X-axis Label:", self.x_label_entry)
        
        self.y_label_entry = QLineEdit()
        self.y_label_entry.setPlaceholderText("e.g., Intensity [arb.u.]")
        labels_layout.addRow("Y-axis Label:", self.y_label_entry)
       
        # colorbar
        self.colorbar_label_entry = QLineEdit()
        self.colorbar_label_entry.setPlaceholderText("e.g., log_{10} (Intensity) [arb. units]")
        labels_layout.addRow("Colorbar Label:", self.colorbar_label_entry)

        return labels_widget
    
    def create_button_section(self, layout):
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #f5f5f5;
                color: #666;
                border: 2px solid #e0e0e0;
            }
            QPushButton:hover {
                background: #eeeeee;
                border: 2px solid #bdbdbd;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        
        # Submit button
        submit_btn = QPushButton("Generate Plot")
        submit_btn.setIcon(QIcon(":/icons/analytics_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png"))
        submit_btn.setIconSize(QSize(24, 24))
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self.on_submit)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)
    
    ##$$$$$$$$$$$$$
    def setup_animations(self):
        self.fade_effect = QGraphicsOpacityEffect()
        self.advanced_group.setGraphicsEffect(self.fade_effect)
        
        self.fade_animation = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_animation.setDuration(0) # No wasting time !

    def toggle_advanced_options(self, state):
        if state == Qt.Checked:
            self.advanced_group.setVisible(True)
            self.fade_animation.stop()
            try:
                self.fade_animation.finished.disconnect()
            except TypeError:
                pass

            self.fade_animation.setStartValue(0.0)
            self.fade_animation.setEndValue(1.0)
            self.fade_animation.start()

        else:
            self.fade_animation.stop()
            try:
                self.fade_animation.finished.disconnect()
            except TypeError:
                pass

            self.fade_animation.setStartValue(1.0)
            self.fade_animation.setEndValue(0.0)
            self.fade_animation.finished.connect(lambda: self.advanced_group.setVisible(False))
            self.fade_animation.start()

    ##$$$$$$$$$$$$$
    
    def open_color_picker(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.line_color_entry.setText(color.name())
    
    def is_valid_color(self, color):
        try:
            mcolors.to_rgba(color)
            return True
        except ValueError:
            return False
    
    def on_submit(self):
        try:
            lambda0_nm = float(self.lambda0_entry.text())
            if not (0 <= lambda0_nm <= 10000):
                raise ValueError("Driving wavelength must be positive and must be less than 10,000 nm.")
            previous_input_time_frequency["lambda0_nm"] = lambda0_nm
        
            filtering = float(self.filter_entry.text())
            if not (0 <= filtering <= 100):
                raise ValueError("End-of-Pulse must be between 0 [%] and 100 [%].")
            previous_input_time_frequency["filtering"] = filtering
        
            qstart = float(self.qstart_entry.text())
            if qstart <= 0:
                raise ValueError("q must be a positive number.")
            previous_input_time_frequency["qstart"] = qstart
            
            qend = float(self.qend_entry.text())
            if qend <= qstart:
                QMessageBox.warning(dialog, "Invalid input", "Maximum HO (qend) must be greater than Minimum HO (qstart).")
                return
            previous_input_time_frequency["qend"] = qend

            g_factor = float(self.g_factor_entry.text())
            if 15 <= g_factor:
                QMessageBox.warning(
                        self,
                        "Warning: Small Gabor Window",
                        "g_factor is too large — the Gabor window becomes too small and may cause instability."
                    )
            previous_input_time_frequency["g_factor"] = g_factor
            
            
            selected_components = []
            if self.total_checkbox.isChecked():
                selected_components.append('total')
            if self.x_component_checkbox.isChecked():
                selected_components.append('x')
            if self.y_component_checkbox.isChecked():
                selected_components.append('y')

            if not selected_components:
                QMessageBox.warning(dialog, "Invalid input", "Please select at least one spectrum option.")
                return

            plot_settings = {}
            if self.plot_options_checkbox.isChecked():
                
                #line_color = self.line_color_entry.text().strip() or "black"
                #if not self.is_valid_color(line_color):
                #    raise ValueError(f"Invalid color value: {line_color}")
                
                plot_settings = {
                    "x_min": float(self.x_min_entry.text()) if self.x_min_entry.text() else None,
                    "x_max": float(self.x_max_entry.text()) if self.x_max_entry.text() else None,
                    "y_min": float(self.y_min_entry.text()) if self.y_min_entry.text() else None,
                    "y_max": float(self.y_max_entry.text()) if self.y_max_entry.text() else None,
                    "clim_min": float(self.clim_min_entry.text()) if self.clim_min_entry.text() else None,
                    "clim_max": float(self.clim_max_entry.text()) if self.clim_max_entry.text() else None,
                    "graph_title": self.graph_title_entry.text(),
                    "x_label": self.x_label_entry.text(),
                    "y_label": self.y_label_entry.text(),
                    "colorbar_label": self.colorbar_label_entry.text(),
                    "cmap":    self.cmap_combo.currentText(),
                    "aspect":  self.aspect_combo.currentText(),
                    "origin":  self.origin_combo.currentText(),
                    "interpolation": self.interpolation_combo.currentText(),
                    "vmin": float(self.vmin_entry.text()) if self.vmin_entry.text() else None,
                    "vmax": float(self.vmax_entry.text()) if self.vmax_entry.text() else None
                }
            ## ----------------------------------------
            window_func = []
            if self.windowFunc_cosine_checkbox.isChecked():
                window_func.append("cosine")
                window_func.append(float(self.cosine_nn_entry.text()))

            elif self.windowFunc_gaussian_checkbox.isChecked():
                window_func.append("Gaussian")
                window_func.append(float(self.gaussian_sigma_entry.text()))
                
            elif self.windowFunc_exp_decay_checkbox.isChecked():
                window_func.append("Exponential Decay")
                window_func.append(float(self.exp_decay_rate_entry.text()))

            elif self.windowFunc_hanning_checkbox.isChecked():
                window_func.append("Hanning")
                window_func.append(None)
                
            elif self.windowFunc_welch_checkbox.isChecked():
                window_func.append("Welch")
                window_func.append(None)

            elif self.windowFunc_bartlett_checkbox.isChecked():
                window_func.append("Bartlett")
                window_func.append(None)

            else:
                window_func.append("None")
                window_func.append(None)
            ## ----------------------------------------

            extract_data_option = []
            if self.extract_data_checkbox.isChecked():
                extract_data_option.append('extract_data')
            
            # Update
            previous_input_time_frequency.update({"lambda0_nm": lambda0_nm, "filtering": filtering,
                                                  "qstart": qstart, "qend": qend, "g_factor": g_factor})
            
            self.accept()

            # CALL
            time_frequency_connector(lambda0_nm, qstart, qend, g_factor, filtering, selected_components, window_func, extract_data_option, plot_settings, self.parent().ipy_console)
                
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")
            return

def input_dialog_time_frequency(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()



