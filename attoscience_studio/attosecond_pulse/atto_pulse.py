# attosecond_pulse/atto_pulse.py

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
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from matplotlib.widgets import Slider, Button
from joblib import Parallel, delayed
from functools import partial
from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
#--------------------------------
from attoscience_studio.utils.window_func_ATTO import TotalCurrentFilter
#--------------------------------
from attoscience_studio.resources_rc import *
#--------------------------------
from attoscience_studio.helper_functions.constants import AtomicUnits
TIMEau = AtomicUnits.TIMEau
##----------------------------------------------------
def attosecond_pulses(lambda0_nm, qstart, qmax, filtering, attosecond_method, window_func, file_path):
    data = np.loadtxt(file_path)
    
    nrm = 0.9500
    nn = filtering    
    
    w0 = 45.5633 / lambda0_nm
    T = 2 * np.pi / w0
    dw = 0.01
    w = w0 * np.arange(qstart, qmax+ dw, dw) 

    t = data[:, 1]
    dt = t[1] - t[0]
    jx = data[:, 2];jx -= jx[0]  
    jy = data[:, 3];jy -= jy[0]   
    
    djx = np.gradient(jx) / dt; djx -= djx[0]
    djy = np.gradient(jy) / dt; djy -= djy[0]
    
    ith = qstart
    jth = qmax 
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
    hx, hy, dhx, dhy = filter_obj.apply_filter(t, jx, jy, djx, djy)
    #---------------------------------------
    if attosecond_method == 'Method 1':
        ahx = np.zeros(len(w), dtype=complex)
        ahy = np.zeros(len(w), dtype=complex)
        for l in range(len(w)):
            fx = hx * np.exp(-1j * w[l] * t)
            ahx[l] = np.trapz(fx, t)
            fy = hy * np.exp(-1j * w[l] * t)
            ahy[l] = np.trapz(fy, t)
    
        Ix = np.zeros(len(t), dtype=complex)
        Iy = np.zeros(len(t), dtype=complex)
        for j in range(len(t)):
            ffx = ahx * np.exp(1j * w * t[j])
            Ix[j] = np.trapz(ffx, w)
            ffy = ahy * np.exp(1j * w * t[j])
            Iy[j] = np.trapz(ffy, w)
        
        Ix = np.abs(Ix)**2
        Iy = np.abs(Iy)**2
        I  = np.abs(Ix + Iy) ** 2
        
    elif attosecond_method == 'Method 2':
        ajx = np.zeros(len(w), dtype=complex)
        ajy = np.zeros(len(w), dtype=complex)
        for l in range(len(w)):
            gx = dhx * np.exp(-1j * w[l] * t)
            ajx[l] = w[l] * np.trapz(gx, t)
            gy = dhy * np.exp(-1j * w[l] * t)
            ajy[l] = w[l] * np.trapz(gy, t)
    
        Ix = np.zeros(len(t), dtype=complex)
        Iy = np.zeros(len(t), dtype=complex)
        for j in range(len(t)):
            ggx = ajx * np.exp(1j * w * t[j])
            Ix[j] = np.trapz(ggx, w)
            ggy = ajy * np.exp(1j * w * t[j])
            Iy[j] = np.trapz(ggy, w)

        Ix = np.abs(Ix)**2
        Iy = np.abs(Iy)**2    
        I = np.abs(Ix + Iy) ** 2
        
    I_Max_x = max(Ix)
    I_Max_y = max(Iy)
    I_Max = max(I)
    Time_OC = t/T
    return I, Ix, Iy, I_Max, I_Max_x, I_Max_y, Time_OC, T, t
##----------------------------------------------------      
def plot_attosecond_pulse(I, Ix, Iy, I_Max, I_Max_x, I_Max_y, Time_OC, T, t, selected_components, CO_FWHM, lambda0_nm, qstart, qmax, TIMEau, extract_data_option, x_axis_unit, plot_settings):
    if 'total' in selected_components:
        if 'extract_data' in extract_data_option:
            file_path = "pulse_tot.txt"
            header = (
                f"# Driving wavelength [nm]: {lambda0_nm}\n"
                f"# T [a.u.]:                {T}\n"
                f"# Min HO:                  {qstart}\n"
                f"# Max HO:                  {qmax}\n"
                "# t [a.u.], I (Intensity [arb.u])\n"
                + '#' * 60
            )
            np.savetxt(file_path, np.column_stack((t, I)), header=header, comments='', fmt='%.12e')

        plt.figure(1)
        bg_color = plot_settings.get("background_color", "white")
        plt.gcf().patch.set_facecolor(bg_color)

        if x_axis_unit=="Atomic unit":
            plt.plot(t, I, linewidth=plot_settings.get("line_thickness", 1.4),color=plot_settings.get("line_color", "black"))
        else:
            plt.plot(Time_OC, I, linewidth=plot_settings.get("line_thickness", 1.4),color=plot_settings.get("line_color", "black"))
            
        if 'coord' in CO_FWHM:
            idx_tot = np.argmax(I)
            max_I_tot = t[idx_tot]
            half_I_tot = I_Max / 2

            left_idx_candidates_tot  = np.where(I[:idx_tot] <= half_I_tot)[0]
            right_idx_candidates_tot = np.where(I[idx_tot:] <= half_I_tot)[0]

            if left_idx_candidates_tot.size > 0:
                left_idx_tot = left_idx_candidates_tot[-1]
            else:
                left_idx_tot = 0

            if right_idx_candidates_tot.size > 0:
                right_idx_tot = right_idx_candidates_tot[0] + idx_tot
            else:
                right_idx_tot = len(I) - 1
            t_left_tot  = Time_OC[left_idx_tot]
            t_right_tot = Time_OC[right_idx_tot]
            FWHM_tot = t_right_tot - t_left_tot
            
            FWHM_SI_tot = FWHM_tot * T * TIMEau
            FWHM_as_tot = FWHM_SI_tot * 1e18

            plt.plot(max_I_tot / T, I_Max, 'o', markersize=4, markeredgecolor='b', markerfacecolor=[0, 0.7, 0.7])
            plt.plot([t_left_tot, t_right_tot], [half_I_tot, half_I_tot],linewidth=1, color=[0, 0.7, 0.7])
            plt.text((max_I_tot / T)- 2.0, I_Max, f'[{max_I_tot / T:.2e}, {I_Max:.2e}]\nFWHM: {FWHM_as_tot:.2f} as', 
               color=[0, 0.7, 0.7], fontsize=11, fontweight='bold', zorder=5)

        plt.title(plot_settings.get("graph_title", "Attosecond pulse (Total)"))
        plt.xlabel(plot_settings.get("x_label", "Time [a.u.]"))
        plt.ylabel(plot_settings.get("y_label", "Intensity [arb.u.]"))
        
        x_axis = t if x_axis_unit == "Atomic unit" else Time_OC
        plt.xlim(left=plot_settings.get("x_min", np.min(x_axis)), right=plot_settings.get("x_max", np.max(x_axis)))
        plt.ylim(bottom=plot_settings.get("y_min", 0), top=plot_settings.get("y_max", I_Max + I_Max/5))
       
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['top'].set_visible(True)
        plt.box(True)
        plt.show()
        
    if 'x' in selected_components:
        if 'extract_data' in extract_data_option:
            file_path = "pulse_x.txt"
            header = (
                f"# Driving wavelength [nm]: {lambda0_nm}\n"
                f"# T [a.u.]:                {T}\n"
                f"# Min HO:                  {qstart}\n"
                f"# Max HO:                  {qmax}\n"
                "# t [a.u.], Ix (Intensity [arb.u])\n"
                + '#' * 60
            )
            np.savetxt(file_path, np.column_stack((t, Ix)), header=header, comments='', fmt='%.12e')

        plt.figure(2)
        bg_color = plot_settings.get("background_color", "white")
        plt.gcf().patch.set_facecolor(bg_color)

        if x_axis_unit=="Atomic unit":
            plt.plot(t, Ix, linewidth=plot_settings.get("line_thickness", 1.4),color=plot_settings.get("line_color", "black"))
        else:
            plt.plot(Time_OC, Ix, linewidth=plot_settings.get("line_thickness", 1.4),color=plot_settings.get("line_color", "black"))
            
        if 'coord' in CO_FWHM:
            idx_x = np.argmax(Ix)
            max_I_x = t[idx_x]
            half_I_x = I_Max_x / 2

            left_idx_candidates_x  = np.where(Ix[:idx_x] <= half_I_x)[0]
            right_idx_candidates_x = np.where(Ix[idx_x:] <= half_I_x)[0]

            if left_idx_candidates_x.size > 0:
                left_idx_x = left_idx_candidates_x[-1]
            else:
                left_idx_x = 0

            if right_idx_candidates_x.size > 0:
                right_idx_x = right_idx_candidates_x[0] + idx_x
            else:
                right_idx_x = len(Ix) - 1
            t_left_x  = Time_OC[left_idx_x]
            t_right_x = Time_OC[right_idx_x]
            FWHM_x = t_right_x - t_left_x
            FWHM_SI_x = FWHM_x * T * TIMEau
            FWHM_as_x = FWHM_SI_x * 1e18           
            plt.plot(max_I_x / T, I_Max_x, 'o', markersize=4, markeredgecolor='b', markerfacecolor=[0, 0.7, 0.7])
            plt.plot([t_left_x, t_right_x], [half_I_x, half_I_x],linewidth=1, color=[0, 0.7, 0.7])
            plt.text((max_I_x / T) - 2.0, I_Max_x, f'[{max_I_x / T:.2e}, {I_Max_x:.2e}]\nFWHM: {FWHM_as_x:.2f} as', 
               color=[0, 0.7, 0.7], fontsize=11, fontweight='bold', zorder=5)
            
        plt.title(plot_settings.get("graph_title", "Attosecond pulse (X)"))
        plt.xlabel(plot_settings.get("x_label", "Time [a.u.]"))
        plt.ylabel(plot_settings.get("y_label", "Intensity [arb.u.]"))
        
        x_axis = t if x_axis_unit == "Atomic unit" else Time_OC
        plt.xlim(left=plot_settings.get("x_min", np.min(x_axis)), right=plot_settings.get("x_max", np.max(x_axis)))
        plt.ylim(bottom=plot_settings.get("y_min", 0), top=plot_settings.get("y_max", I_Max_x + I_Max_x/5))
     
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['top'].set_visible(True)
        plt.box(True)
        plt.show()

    if 'y' in selected_components:
        if 'extract_data' in extract_data_option:
            file_path = "pulse_y.txt"
            header = (
                f"# Driving wavelength [nm]: {lambda0_nm}\n"
                f"# T [a.u.]:                {T}\n"
                f"# Min HO:                  {qstart}\n"
                f"# Max HO:                  {qmax}\n"
                "# t [a.u.], Iy (Intensity [arb.u])\n"
                + '#' * 60
            )
            np.savetxt(file_path, np.column_stack((t, Iy)), header=header, comments='', fmt='%.12e')

        plt.figure(3)
        bg_color = plot_settings.get("background_color", "white")
        plt.gcf().patch.set_facecolor(bg_color)

        if x_axis_unit=="Atomic unit":
            plt.plot(t, Iy, linewidth=plot_settings.get("line_thickness", 1.4),color=plot_settings.get("line_color", "black"))
        else:
            plt.plot(Time_OC, Iy, linewidth=plot_settings.get("line_thickness", 1.4),color=plot_settings.get("line_color", "black"))

        if 'coord' in CO_FWHM:
            idx_y = np.argmax(Iy)
            max_I_y = t[idx_y]
            half_I_y = I_Max_y / 2

            left_idx_candidates_y  = np.where(Iy[:idx_y] <= half_I_y)[0]
            right_idx_candidates_y = np.where(Iy[idx_y:] <= half_I_y)[0]

            if left_idx_candidates_y.size > 0:
                left_idx_y = left_idx_candidates_y[-1]
            else:
                left_idx_y = 0

            if right_idx_candidates_y.size > 0:
                right_idx_y = right_idx_candidates_y[0] + idx_y
            else:
                right_idx_y = len(Iy) - 1

            t_left_y  = Time_OC[left_idx_y]
            t_right_y = Time_OC[right_idx_y]

            FWHM_y = t_right_y - t_left_y
            FWHM_SI_y = FWHM_y * T * TIMEau
            FWHM_as_y = FWHM_SI_y * 1e18

            plt.plot(max_I_y / T, I_Max_y, 'o', markersize=4, markeredgecolor='b', markerfacecolor=[0, 0.7, 0.7])
            plt.plot([t_left_y, t_right_y], [half_I_y, half_I_y],linewidth=1, color=[0, 0.7, 0.7])
            plt.text((max_I_y / T) - 2.0, I_Max_y, f'[{max_I_y / T:.2e}, {I_Max_y:.2e}]\nFWHM: {FWHM_as_y:.2f} as', 
               color=[0, 0.7, 0.7], fontsize=11, fontweight='bold', zorder=5)

        plt.title(plot_settings.get("graph_title", "Attosecond pulse (Y)"))
        plt.xlabel(plot_settings.get("x_label", "Time [a.u.]"))
        plt.ylabel(plot_settings.get("y_label", "Intensity [arb.u.]"))

        x_axis = t if x_axis_unit == "Atomic unit" else Time_OC
        plt.xlim(left=plot_settings.get("x_min", np.min(x_axis)), right=plot_settings.get("x_max", np.max(x_axis)))
        plt.ylim(bottom=plot_settings.get("y_min", 0), top=plot_settings.get("y_max", I_Max_y + I_Max_y/5))

        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['top'].set_visible(True)
        plt.box(True)
        plt.show()
##----------------------------------------------------         
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")

def atto_plot_connector(lambda0_nm, qstart, qmax, filtering, attosecond_method, x_axis_unit, selected_components, CO_FWHM,
                                extract_data_option, plot_settings, window_func, ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select total_current file")
    if "total_current" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the 'total_current' file.")
        return        
    if file_path:
        try:
            I, Ix, Iy, I_Max, I_Max_x, I_Max_y, Time_OC, T, t = attosecond_pulses(lambda0_nm, qstart, qmax, filtering, attosecond_method, window_func, file_path)
            plot_attosecond_pulse(I, Ix, Iy, I_Max, I_Max_x, I_Max_y, Time_OC, T, t, selected_components, CO_FWHM, lambda0_nm, qstart, qmax, TIMEau, extract_data_option, x_axis_unit, plot_settings)
            max_Time_OC = np.max(Time_OC)
            T_SI = T*2.418884326509*1e-17
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            msg = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          HHG log!                          --\n"
                + "-" * 75 + "\n"
                ">>> HHG calculation and visualization successfully completed!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> T [a.u.]: {T:.12e}\n"
                f">>> T [second]: {T_SI:.12e}\n"
                f">>> Max optical cycle: {max_Time_OC}\n"
                + "-" * 75
            )
            print_to_console(ipy_console, msg)
        except ValueError as e:
            QMessageBox.warning(None, "Error", str(e))
            return
##----------------------------------------------------
previous_input_atto = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Attosecond pulse Parameters")
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
            image: url(:/icons/select_check_box_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png);
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
        
        self.create_header(main_layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        self.create_required_section(content_layout)
        self.create_optional_section(content_layout)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        self.create_button_section(main_layout)
        
    def create_header(self, layout):
        header_layout = QHBoxLayout()
        
        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon_label.setPixmap(icon.pixmap(32, 32))

        title_layout = QVBoxLayout()
        title = QLabel("Attosecond pulse")
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
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e0e0e0; margin: 10px 0;")
        layout.addWidget(line)
    
    def create_required_section(self, layout):
        required_group = QGroupBox("Essential Parameters")
        required_layout = QVBoxLayout()
        required_layout.setSpacing(20)

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
        self.lambda0_entry.setText(str(previous_input_atto.get("lambda0_nm", "")))
        self.lambda0_entry.setMaxLength(10)

    
        lambda0_unit_label = QLabel("nm")
        lambda0_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
    
        lambda0_layout.addWidget(self.lambda0_entry)
        lambda0_layout.addWidget(lambda0_unit_label)
        
        basic_params_layout.addRow("Driving Wavelength:", lambda0_container)

        # (2) Minimum HO
        qstart_container = QWidget()
        qstart_layout = QHBoxLayout(qstart_container)
        qstart_layout.setContentsMargins(0, 0, 0, 0)
    
        self.qstart_entry = QLineEdit()
        self.qstart_entry.setPlaceholderText("Enter min HO (e.g., 10)")
        self.qstart_entry.setText(str(previous_input_atto.get("qstart", "")))
        self.qstart_entry.setMaxLength(10)
        
        qstart_unit_label = QLabel("HO")
        qstart_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qstart_layout.addWidget(self.qstart_entry)
        qstart_layout.addWidget(qstart_unit_label)
    
        basic_params_layout.addRow("Minimum Harmonic Order:", qstart_container)

        # (3) Maximum HO
        qmax_container = QWidget()
        qmax_layout = QHBoxLayout(qmax_container)
        qmax_layout.setContentsMargins(0, 0, 0, 0)
    
        self.qmax_entry = QLineEdit()
        self.qmax_entry.setPlaceholderText("Enter min HO (e.g., 10)")
        self.qmax_entry.setText(str(previous_input_atto.get("qmax", "")))
        self.qmax_entry.setMaxLength(10)
        
        qmax_unit_label = QLabel("HO")
        qmax_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qmax_layout.addWidget(self.qmax_entry)
        qmax_layout.addWidget(qmax_unit_label)
    
        basic_params_layout.addRow("Maximum Harmonic Order:", qmax_container)
    
        # (4) attosecond_method
        attosecond_method_container = QWidget()
        attosecond_method_layout = QHBoxLayout(attosecond_method_container)
        attosecond_method_layout.setContentsMargins(0, 0, 0, 10)
    
        self.attosecond_method_entry = QComboBox()
        self.attosecond_method_entry.addItems(["Method 1", "Method 2"])
        self.attosecond_method_entry.setFixedSize(120, 30)
        self.attosecond_method_entry.setStyleSheet("""QComboBox { min-height: 30px;font-size: 12px;}""")

        
        attosecond_method_spacer = QLabel("")
        attosecond_method_spacer.setStyleSheet("min-width: 30px;")
    
        attosecond_method_layout.addWidget(self.attosecond_method_entry)
        attosecond_method_layout.addWidget(attosecond_method_spacer)
    
        basic_params_layout.addRow("Method:", attosecond_method_container)
        #------
        basic_params_group.setLayout(basic_params_layout)
        required_layout.addWidget(basic_params_group)

        ##- Display Options Section -------------------------------------
        display_options_group = QGroupBox("Display Options")
        display_options_layout = QVBoxLayout()
        display_options_layout.setSpacing(15)

        components_subgroup = QGroupBox("Components")
        components_layout = QHBoxLayout()
        components_layout.setContentsMargins(10, 0, 0, 10)
        components_layout.setSpacing(15)
    
        self.spectrum_x_checkbox = QCheckBox("X Component")
        self.spectrum_y_checkbox = QCheckBox("Y Component")
        self.spectrum_total_checkbox = QCheckBox("Total")
    
        components_layout.addWidget(self.spectrum_x_checkbox)
        components_layout.addWidget(self.spectrum_y_checkbox)
        components_layout.addWidget(self.spectrum_total_checkbox)
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
        self.filter_entry.setText(str(previous_input_atto.get("filtering", "0.0")))
        self.filter_entry.setMaxLength(4)
        
        filter_unit_label = QLabel("%")
        filter_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")

        filter_layout.addWidget(self.filter_entry)
        filter_layout.addWidget(filter_unit_label)
    
        filter_form_layout.addRow("End-of-pulse:", filter_container)
        windowFunc_main_layout.addLayout(filter_form_layout)
        #--------------------------------------------
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
        self.cosine_nn_entry.setPlaceholderText("Exponent")
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

        # X-axis unit selection -----------------------------
        x_axis_subgroup = QGroupBox("X-axis Unit")
        x_axis_layout = QHBoxLayout()
        x_axis_layout.setContentsMargins(10, 10, 10, 10)
    
        self.x_axis_unit_entry = QComboBox()
        self.x_axis_unit_entry.addItems(["Atomic unit", "Optical cycle"])
        self.x_axis_unit_entry.setFixedSize(120, 30)
        self.x_axis_unit_entry.setStyleSheet("""QComboBox { min-height: 30px;font-size: 12px;}""")
        
        x_axis_layout.addWidget(self.x_axis_unit_entry)
        x_axis_layout.addStretch()
    
        x_axis_subgroup.setLayout(x_axis_layout)
        display_options_layout.addWidget(x_axis_subgroup)

        # Additional options
        options_subgroup = QGroupBox("Additional Options")
        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(10, 10, 10, 10)
    
        self.extract_data_checkbox = QCheckBox("Extract Data")
        self.atto_coordinate_checkbox = QCheckBox("Coordinates and FWHM")
    
        options_layout.addWidget(self.extract_data_checkbox)
        options_layout.addWidget(self.atto_coordinate_checkbox)
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
        tab_widget.setIconSize(QSize(20, 20)) # global icon size for all tabs
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
        
        return axes_widget
    
    def create_styling_tab(self):
        styling_widget = QWidget()
        styling_layout = QFormLayout(styling_widget)
        styling_layout.setSpacing(12)
        
        # Color picker for line color
        line_color_widget = QWidget()
        line_color_layout = QHBoxLayout(line_color_widget)
        line_color_layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_color_entry = QLineEdit()
        self.line_color_entry.setPlaceholderText("e.g., blue, #ff5733")
        
        color_picker_btn = QPushButton("🎨")
        color_picker_btn.setFixedSize(40, 32)
        color_picker_btn.clicked.connect(self.open_color_picker)
        
        line_color_layout.addWidget(self.line_color_entry)
        line_color_layout.addWidget(color_picker_btn)
        
        styling_layout.addRow("Line Color:", line_color_widget)
        
        thickness_widget = QWidget()
        thickness_layout = QHBoxLayout(thickness_widget)
        thickness_layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_thickness_spinbox = QDoubleSpinBox()
        self.line_thickness_spinbox.setRange(0.1, 10.0)
        self.line_thickness_spinbox.setValue(1.2)
        self.line_thickness_spinbox.setSuffix(" px")
        
        thickness_slider = QSlider(Qt.Horizontal)
        thickness_slider.setRange(1, 100)
        thickness_slider.setValue(12)
        thickness_slider.valueChanged.connect(lambda v: self.line_thickness_spinbox.setValue(v/10.0))
        
        thickness_layout.addWidget(self.line_thickness_spinbox)
        thickness_layout.addWidget(thickness_slider)
        
        styling_layout.addRow("Line Thickness:", thickness_widget)
        
        self.background_color_entry = QLineEdit()
        self.background_color_entry.setPlaceholderText("e.g., white, black, #f0f0f0")
        styling_layout.addRow("Background Color:", self.background_color_entry)
        
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

        submit_btn = QPushButton("Generate Plot")
        submit_btn.setIcon(QIcon("attoscience_studio/icons/analytics_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png"))
        submit_btn.setIconSize(QSize(24, 24))
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self.on_submit)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)
    
    ##$$$$$$$$$$$$$$$$$$$$$$$$$
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

    ##$$$$
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
            previous_input_atto["lambda0_nm"] = lambda0_nm
        
            filtering = float(self.filter_entry.text())
            if not (0 <= filtering <= 100):
                raise ValueError("End-of-Pulse must be between 0 [%] and 100 [%].")
            previous_input_atto["filtering"] = filtering
        
            qstart = float(self.qstart_entry.text())
            if qstart <= 0:
                raise ValueError("Maximum harmonic order must be a positive number.")
            previous_input_atto["qstart"] = qstart

            qmax = float(self.qmax_entry.text())
            if qmax <= qstart:
                raise ValueError("Invalid input", "Maximum HO must be greater than Minimum HO.")
                return
            previous_input_atto["qmax"] = qmax

            attosecond_method = self.attosecond_method_entry.currentText()
            x_axis_unit = self.x_axis_unit_entry.currentText()

            selected_components = []
            if self.spectrum_x_checkbox.isChecked():
                selected_components.append('x')
            if self.spectrum_y_checkbox.isChecked():
                selected_components.append('y')
            if self.spectrum_total_checkbox.isChecked():
                selected_components.append('total')

            if not selected_components:
                QMessageBox.warning(self, "Invalid input", "Please select at least one component.")
                return
  
            plot_settings = {}
            if self.plot_options_checkbox.isChecked():
                line_color = self.line_color_entry.text().strip() or "black"
                background_color = self.background_color_entry.text().strip() or "white"
                
                if not self.is_valid_color(line_color):
                    raise ValueError(f"Invalid color value: {line_color}")
                
                plot_settings = {
                    "x_min": float(self.x_min_entry.text()) if self.x_min_entry.text() else None,
                    "x_max": float(self.x_max_entry.text()) if self.x_max_entry.text() else None,
                    "y_min": float(self.y_min_entry.text()) if self.y_min_entry.text() else None,
                    "y_max": float(self.y_max_entry.text()) if self.y_max_entry.text() else None,
                    "graph_title": self.graph_title_entry.text(),
                    "x_label": self.x_label_entry.text(),
                    "y_label": self.y_label_entry.text(),
                    "line_color": line_color,
                    "line_thickness": self.line_thickness_spinbox.value(),
                    "background_color": background_color
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

            CO_FWHM = []
            if self.atto_coordinate_checkbox.isChecked():
                CO_FWHM.append('coord')

            self.accept()
            # Update
            previous_input_atto.update({"lambda0_nm": lambda0_nm, "filtering": filtering, "qstart": qstart, "qmax": qmax})
            
            # CALL
            atto_plot_connector(lambda0_nm, qstart, qmax, filtering, attosecond_method, x_axis_unit, selected_components, CO_FWHM,
                                extract_data_option, plot_settings, window_func, self.parent().ipy_console)
                
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def input_dialog_atto(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()




