# driving_filed/vector_potential.py

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
from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from attoscience_studio.resources_rc import *
##----------------------------------------------------
def read_driving_vector_single(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.size == 0:
            raise ValueError("The file is empty.")
        time, Ax, Ay, Az = data[:,1], *data[:, 2:5].T
        dte = time[1] - time[0]
        if np.all(Ax == 0) and np.all(Ay == 0) and np.all(Az == 0):
            raise ValueError("All values are zero.")          
        return time, dte, Ax, Ay, Az
    except Exception as e:
        raise ValueError(f"Failed to read field data: {e}")        
##----------------------------------------------------
def read_driving_vector_dual(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.size == 0:
            raise ValueError("The file is empty.")
        time, Ax1, Ay1, Az1, Ax2, Ay2, Az2 = data[:, 1], *data[:, 2:8].T
        dte = time[1] - time[0]
        if np.all(Ax1 == 0) and np.all(Ay1 == 0) and np.all(Az1 == 0) and np.all(Ax2 == 0) and np.all(Ay2 == 0) and np.all(Az2 == 0):
            raise ValueError("All values are zero.")
        return time, dte, Ax1, Ay1, Az1, Ax2, Ay2, Az2
    except Exception as e:
        raise ValueError(f"Failed to read field data: {e}")

##----------------------------------------------------
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")
##----------------------------------------------------

def las_plot_single(lambda0_nm, plot_options, cf_option, plot_settings, ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select laser file")
    if "laser" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the 'laser' file.")
        return    
    if file_path:
        try:
            time, dte, Ax, Ay, Az = read_driving_vector_single(file_path)
        except ValueError as e:
            QMessageBox.warning(None, "Error", str(e))
            return
    w0 = 45.5633 / lambda0_nm
    T = 2 * np.pi / w0
    T_SI = T*2.418884326509*1e-17
    Time_OC = time/T; max_Time_OC = np.max(Time_OC)
    
    Amax_x = np.max(Ax)
    Amax_y = np.max(Ay)
    Amax_z = np.max(Az)
   
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    msg = (
        f">>> Time                          {timestamp}\n"
        + "-" * 75 + "\n"
        + "--                          Vector potential log!                          --\n"
        + "-" * 75 + "\n"
        ">>> Vector potential visualization successfully completed!\n"
        f">>> File loaded from: {file_path}\n"
        f">>> T [a.u.]: {T:.12e}\n"
        f">>> T [second]: {T_SI:.12e}\n"
        f">>> w0: {w0:.12e}\n"
        f">>> Max optical cycle: {max_Time_OC}\n"
        f">>> Amax_x = {Amax_x:.4e} [a.u.]\n"
        f">>> Amax_y = {Amax_y:.4e} [a.u.]\n"
        f">>> Amax_z = {Amax_z:.4e} [a.u.]\n"
        + "-" * 75
        )
    print_to_console(ipy_console, msg)
    
    if "fieldx" or "fieldy" or "fieldz" or "3D" in cf_option:
        
        fieldx = np.gradient(Ax) / dte / 137.036
        fieldy = np.gradient(Ay) / dte / 137.036
        fieldz = np.gradient(Az) / dte / 137.036

        if "fieldx" in cf_option:
            plt.figure(1)
            plt.plot(Time_OC, fieldx, color='blue', label='$E_{x}$')
            plt.xlabel('Time [o.c.]')
            plt.ylabel('E [a.u.]')           
            plt.legend(loc='upper right')
            plt.box(True)
            plt.show()
            
        if "fieldy" in cf_option:
            plt.figure(2)
            plt.plot(Time_OC, fieldy, color='red', label='$E_{y}$')
            plt.xlabel('Time [o.c.]')
            plt.ylabel('E [a.u.]')           
            plt.legend(loc='upper right')
            plt.box(True)
            plt.show()
            
        if "fieldz" in cf_option:
            plt.figure(3)
            plt.plot(Time_OC, fieldz, color='black', label='$E_{z}$')
            plt.xlabel('Time [o.c.]')
            plt.ylabel('E [a.u.]')
            plt.legend(loc='upper right')
            plt.box(True)
            plt.show()  
    
        if "3D" in cf_option:
            fig = plt.figure(4)
            ax = fig.add_subplot(111, projection='3d')
    
            if np.all(fieldz == 0):
                ax.plot(Time_OC, fieldx, fieldy, color='blue')
                ax.set_xlabel('Time [o.c.]')
                ax.set_ylabel('$E_{x}$ [a.u.]')
                ax.set_zlabel('$E_{y}$ [a.u.]')
                ax.grid(True)
            
            elif np.all(fieldx == 0):
                ax.plot(Time_OC, fieldy, fieldz, color='red')
                ax.set_xlabel('Time [o.c.]')
                ax.set_ylabel('$E_{y}$ [a.u.]')
                ax.set_zlabel('$E_{z}$ [a.u.]')
                ax.grid(True)
                
            elif np.all(fieldy == 0):
                ax.plot(Time_OC, fieldx, fieldz, color='green')
                ax.set_xlabel('Time [o.c.]')
                ax.set_ylabel('$E_{x}$ [a.u.]')
                ax.set_zlabel('$E_{z}$ [a.u.]')
                ax.grid(True)
            plt.show()

    if any([plot_options['Ax'], plot_options['Ay'], plot_options['Az'], plot_options['3D']]):
        if plot_options['Ax']:
            plt.figure(21)
            plt.plot(Time_OC, Ax, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
            plt.title(plot_settings.get("graph_title", "Vector potential (X)"))
            plt.xlabel(plot_settings.get("x_label", "Time [o.c.]"))
            plt.ylabel(plot_settings.get("y_label", "$A_{x}$ [a.u.]"))
            plt.xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
            plt.ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
            #plt.legend(loc='upper right')
            plt.box(True)
            plt.show()
                   
        if plot_options['Ay']:
            plt.figure(22)
            plt.plot(Time_OC, Ay, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
            plt.title(plot_settings.get("graph_title", "Vector potential (Y)"))
            plt.xlabel(plot_settings.get("x_label", "Time [o.c.]"))
            plt.ylabel(plot_settings.get("y_label", "$A_{y}$ [a.u.]"))
            plt.xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
            plt.ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
            #plt.legend(loc='upper right')
            plt.box(True)
            plt.show()
                  
        if plot_options['Az']:
            plt.figure(23)
            plt.plot(Time_OC, Az, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
            plt.title(plot_settings.get("graph_title", "Vector potential (Z)"))
            plt.xlabel(plot_settings.get("x_label", "Time [o.c.]"))
            plt.ylabel(plot_settings.get("y_label", "$A_{z}$ [a.u.]"))
            plt.xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
            plt.ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
            #plt.legend(loc='upper right')
            plt.box(True)
            plt.show()
    
        if plot_options['3D']:
            fig = plt.figure(24)
            ax = fig.add_subplot(111, projection='3d')   
            if np.all(Az == 0):
                ax.plot(Time_OC, Ax, Ay, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
                ax.set_title(plot_settings.get("graph_title", "Vector potential (XY-plane)"))
                ax.set_xlabel(plot_settings.get("x_label", "Time [o.c.]"))
                ax.set_ylabel(plot_settings.get("y_label", "$A_{x}$ [a.u.]"))
                ax.set_zlabel(plot_settings.get("z_label", "$A_{y}$ [a.u.]"))
                ax.set_xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
                ax.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
                ax.set_zlim(bottom=plot_settings.get("z_min", None), top=plot_settings.get("z_max", None))
                ax.grid(True)
                
            elif np.all(Ax == 0):
                ax.plot(Time_OC, Ay, Az, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
                ax.set_title(plot_settings.get("graph_title", "Vector potential (YZ-plane)"))
                ax.set_xlabel(plot_settings.get("x_label", "Time [o.c.]"))
                ax.set_ylabel(plot_settings.get("y_label", "$A_{y}$ [a.u.]"))
                ax.set_zlabel(plot_settings.get("z_label", "$A_{z}$ [a.u.]"))
                ax.set_xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
                ax.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
                ax.set_zlim(bottom=plot_settings.get("z_min", None), top=plot_settings.get("z_max", None))
                ax.grid(True)
                
            elif np.all(Ay == 0):
                ax.plot(Time_OC, Ax, Az, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
                ax.set_title(plot_settings.get("graph_title", "Vector potential (XZ-plane)"))
                ax.set_xlabel(plot_settings.get("x_label", "Time [o.c.]"))
                ax.set_ylabel(plot_settings.get("y_label", "$A_{x}$ [a.u.]"))
                ax.set_zlabel(plot_settings.get("z_label", "$A_{z}$ [a.u.]"))
                ax.set_xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
                ax.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
                ax.set_zlim(bottom=plot_settings.get("z_min", None), top=plot_settings.get("z_max", None))
                ax.grid(True)
            plt.show()

##----------------------------------------------------
def las_plot_dual(lambda0_nm1, lambda0_nm2, cf_option, plot_settings, ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select laser file")
    if "laser" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the 'laser' file.")
        return    
    if file_path:
        try:
            time, dte, Ax1, Ay1, Az1, Ax2, Ay2, Az2 = read_driving_vector_dual(file_path)
        except ValueError as e:
            QMessageBox.warning(None, "Error", str(e))
            return
    w01 = 45.5633 / lambda0_nm1
    w02 = 45.5633 / lambda0_nm2
    T1 = 2 * np.pi / w01
    T2 = 2 * np.pi / w02

    Amax_x1 = np.max(Ax1)
    Amax_y1 = np.max(Ay1)
    Amax_z1 = np.max(Az1)
    Amax_x2 = np.max(Ax2)
    Amax_y2 = np.max(Ay2)
    Amax_z2 = np.max(Az2)
    A_X = Ax1 + Ax2
    A_Y = Ay1 + Ay2
    A_Z = Az1 + Az2

    Time_OC1 = time/T1; max_Time_OC1 = np.max(Time_OC1)
    Time_OC2 = time/T2; max_Time_OC2 = np.max(Time_OC2)
    T1_SI = T1*2.418884326509*1e-17; T2_SI = T2*2.418884326509*1e-17

    timestamp = datetime.now().strftime("[%H:%M:%S]")
    msg = (
        f">>> Time                          {timestamp}\n"
        + "-" * 75 + "\n"
        + "--                         Vector potential log!                          --\n"
        + "-" * 75 + "\n"
        ">>> Vector potential visualization successfully completed!\n"
        f">>> File loaded from: {file_path}\n"
        f">>> T1 [a.u.]: {T1:.12e}\n"
        f">>> T2 [a.u.]: {T2:.12e}\n"
        f">>> T1 [second]: {T1_SI:.12e}\n"
        f">>> T2 [second]: {T2_SI:.12e}\n"
        f">>> w01: {w01:.12e}\n"
        f">>> w02: {w02:.12e}\n"
        f">>> Max optical cycle (first pulse): {max_Time_OC1}\n"
        f">>> Max optical cycle (second pulse): {max_Time_OC2}\n"
        f">>> Amax_x1 = {Amax_x1:.4e} [a.u.]\n"
        f">>> Amax_y1 = {Amax_y1:.4e} [a.u.]\n"
        f">>> Amax_z1 = {Amax_z1:.4e} [a.u.]\n"
        f">>> Amax_x2 = {Amax_x2:.4e} [a.u.]\n"
        f">>> Amax_y2 = {Amax_y2:.4e} [a.u.]\n"
        f">>> Amax_z2 = {Amax_z2:.4e} [a.u.]\n"

        + "-" * 75
        )
    print_to_console(ipy_console, msg)

    if "field" in cf_option:
        fieldx1 = np.gradient(Ax1) / dte / 137.036
        fieldy1 = np.gradient(Ay1) / dte / 137.036
        fieldz1 = np.gradient(Az1) / dte / 137.036  
        fieldx2 = np.gradient(Ax2) / dte / 137.036
        fieldy2 = np.gradient(Ay2) / dte / 137.036
        fieldz2 = np.gradient(Az2) / dte / 137.036

        field_X = (fieldx1 + fieldx2)
        field_Y = (fieldy1 + fieldy2)
        field_Z = (fieldz1 + fieldz2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if np.all(field_Z == 0):
            ax.plot(Time_OC1, field_X, field_Y, linewidth=1.2 ,color="black")
            ax.set_xlabel('Time [o.c.]')
            ax.set_ylabel('$E_{x}$ [a.u.]')
            ax.set_zlabel('$E_{y}$ [a.u.]')
            ax.grid(True)
            
        elif np.all(field_X == 0):
            ax.plot(Time_OC1, field_Y, field_Z, linewidth=1.2 ,color="black")
            ax.set_xlabel('Time [o.c.]')
            ax.set_ylabel('$E_{y}$ [a.u.]')
            ax.set_zlabel('$E_{z}$ [a.u.]')
            ax.grid(True)
            
        elif np.all(field_Y == 0):
            ax.plot(Time_OC1, field_X, field_Z, linewidth=1.2 ,color="black")
            ax.set_xlabel('Time [o.c.]')
            ax.set_ylabel('$E_{x}$ [a.u.]')
            ax.set_zlabel('$E_{z}$ [a.u.]')
            ax.grid(True)
            
        plt.show()
    
    if "A" in cf_option:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if np.all(A_Z == 0):
            ax.plot(Time_OC1, A_X, A_Y, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
            ax.set_title(plot_settings.get("graph_title", "Vector potential (XY-plane)"))
            ax.set_xlabel(plot_settings.get("x_label", "Time [o.c.]"))
            ax.set_ylabel(plot_settings.get("y_label", "$A_{x}$ [a.u.]"))
            ax.set_zlabel(plot_settings.get("z_label", "$A_{y}$ [a.u.]"))
            ax.set_xlim(left=plot_settings.get("x_min", np.min(Time_OC1)), right=plot_settings.get("x_max", np.max(Time_OC1)))
            ax.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
            ax.set_zlim(bottom=plot_settings.get("z_min", None), top=plot_settings.get("z_max", None))
            ax.grid(True)
            
        elif np.all(A_X == 0):
            ax.plot(Time_OC1, A_Y, A_Z, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
            ax.set_title(plot_settings.get("graph_title", "Vector potential (YZ-plane)"))
            ax.set_xlabel(plot_settings.get("x_label", "Time [o.c.]"))
            ax.set_ylabel(plot_settings.get("y_label", "$A_{y}$ [a.u.]"))
            ax.set_zlabel(plot_settings.get("z_label", "$A_{z}$ [a.u.]"))
            ax.set_xlim(left=plot_settings.get("x_min", np.min(Time_OC1)), right=plot_settings.get("x_max", np.max(Time_OC1)))
            ax.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
            ax.set_zlim(bottom=plot_settings.get("z_min", None), top=plot_settings.get("z_max", None))
            ax.grid(True)
            
        elif np.all(A_Y == 0):
            ax.plot(Time_OC1, A_X, A_Z, linewidth=plot_settings.get("line_thickness", 1.2),color=plot_settings.get("line_color", "black"))
            ax.set_title(plot_settings.get("graph_title", "Vector potential (XZ-plane)"))
            ax.set_xlabel(plot_settings.get("x_label", "Time [o.c.]"))
            ax.set_ylabel(plot_settings.get("y_label", "$A_{x}$ [a.u.]"))
            ax.set_zlabel(plot_settings.get("z_label", "$A_{z}$ [a.u.]"))
            ax.set_xlim(left=plot_settings.get("x_min", np.min(Time_OC1)), right=plot_settings.get("x_max", np.max(Time_OC1)))
            ax.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
            ax.set_zlim(bottom=plot_settings.get("z_min", None), top=plot_settings.get("z_max", None))
            ax.grid(True)
            
        plt.show()

##----------------------------------------------------

previous_input_vector_potential = {}
class VectorPotentialDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vector Potential Parameters")
        self.setFixedSize(700, 600)
        self.setStyleSheet(self.get_modern_stylesheet())
        self.selection = None
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
        }
        
        QPushButton:pressed {
            background: #1565c0;
        }
        
        QPushButton.pulse-button {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #4caf50, stop: 1 #388e3c);
            min-height: 50px;
            font-size: 16px;
            margin: 8px;
        }
        
        QPushButton.pulse-button:hover {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #66bb6a, stop: 1 #4caf50);
        }
        
        QPushButton.pulse-button:pressed {
            background: #2e7d32;
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
        
        # Content sections
        self.create_pulse_selection_section(content_layout)
        self.create_parameters_section(content_layout)
        self.create_optional_section(content_layout)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Button section
        self.create_button_section(main_layout)
        
    def create_header(self, layout):
        header_layout = QHBoxLayout()
        
        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon_label.setPixmap(icon.pixmap(32, 32))
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title = QLabel("Vector Potential Analysis")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")
        
        subtitle = QLabel("Configure parameters for vector potential visualization")
        subtitle.setStyleSheet("font-size: 14px; color: #666; margin: 0;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(2)
        
        header_layout.addWidget(icon_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #e0e0e0; margin: 10px 0;")
        layout.addWidget(line)
    
    #-(1)-=======================================
    def create_pulse_selection_section(self, layout):
        pulse_group = QGroupBox("Pulse Configuration")
        pulse_layout = QVBoxLayout()
        pulse_layout.setSpacing(15)
        
        # Question label
        question_label = QLabel("Does your data file contain one or two pulses?")
        question_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #2c3e50; margin-bottom: 10px;")
        pulse_layout.addWidget(question_label)
        
        # Pulse selection buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        self.single_button = QPushButton("Single Pulse")
        self.single_button.clicked.connect(self.select_single) #>>>>>>>>
        
        self.dual_button = QPushButton("Double Pulse")
        self.dual_button.clicked.connect(self.select_dual)     #>>>>>>>>
        
        # Set initial light green color for both buttons
        self.set_button_initial_style()
        
        button_layout.addWidget(self.single_button)
        button_layout.addWidget(self.dual_button)
        
        pulse_layout.addLayout(button_layout)
        pulse_group.setLayout(pulse_layout)
        layout.addWidget(pulse_group)
    
    #-(2)-=======================================
    def create_parameters_section(self, layout):
        self.parameters_group = QGroupBox("Parameters")
        self.parameters_group.setVisible(False) #>>> (initially hidden)
        
        #container for dynamic content
        self.parameters_layout = QVBoxLayout(self.parameters_group)
        
        # both tabs but don't add them yet
        self.single_tab = self.create_single_pulse_tab()
        self.double_tab = self.create_double_pulse_tab()
        
        layout.addWidget(self.parameters_group)
   
    #-(3)-=======================================
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
    #-(3-1)---------------------------------------
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

        # Z-axis range
        z_range_widget = QWidget()
        z_range_layout = QHBoxLayout(z_range_widget)
        z_range_layout.setContentsMargins(0, 0, 0, 0)
        
        self.z_min_entry = QLineEdit()
        self.z_min_entry.setPlaceholderText("Min")
        self.z_max_entry = QLineEdit()
        self.z_max_entry.setPlaceholderText("Max")
        
        z_range_layout.addWidget(self.z_min_entry)
        z_range_layout.addWidget(QLabel("to"))
        z_range_layout.addWidget(self.z_max_entry)
        
        axes_layout.addRow("Z-axis Range (3D plot):", z_range_widget)
        
        return axes_widget
    #-(3-2)---------------------------------------
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
        
        # Line thickness with slider
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
        
        # Background color
        self.background_color_entry = QLineEdit()
        self.background_color_entry.setPlaceholderText("e.g., white, black, #f0f0f0")
        styling_layout.addRow("Background Color:", self.background_color_entry)
        
        return styling_widget
    
    #-(3-3)---------------------------------------
    def create_labels_tab(self):
        labels_widget = QWidget()
        labels_layout = QFormLayout(labels_widget)
        labels_layout.setSpacing(12)
        
        self.graph_title_entry = QLineEdit()
        self.graph_title_entry.setPlaceholderText("Enter plot title")
        labels_layout.addRow("Graph Title:", self.graph_title_entry)
        
        self.x_label_entry = QLineEdit()
        self.x_label_entry.setPlaceholderText("e.g., Time [o.c.]")
        labels_layout.addRow("X-axis Label:", self.x_label_entry)
        
        self.y_label_entry = QLineEdit()
        self.y_label_entry.setPlaceholderText("e.g., Electric Field [a.u.]")
        labels_layout.addRow("Y-axis Label:", self.y_label_entry)

        self.z_label_entry = QLineEdit()
        self.z_label_entry.setPlaceholderText("e.g., Electric Field [a.u.]")
        labels_layout.addRow("Z-axis Label (3D plot):", self.z_label_entry)
       
        return labels_widget
    
    #-(2-1)---------------------------------------
    def create_single_pulse_tab(self):
        single_widget = QWidget()
        single_layout = QVBoxLayout(single_widget)
        single_layout.setSpacing(20)
        
        #-- (1) Wavelength section-------------
        wavelength_group = QGroupBox("Laser Parameters")
        wavelength_layout = QFormLayout()
        wavelength_layout.setSpacing(12)
        
        #-- (2) Driving wavelength -------------
        lambda_container = QWidget()
        lambda_layout = QHBoxLayout(lambda_container)
        lambda_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lambda0_nm_entry = QLineEdit()
        self.lambda0_nm_entry.setPlaceholderText("Enter driving wavelength (e.g., 2000)")
        self.lambda0_nm_entry.setText(str(previous_input_vector_potential.get("lambda0_nm", "")))
        self.lambda0_nm_entry.setMaxLength(10)
        
        lambda_unit_label = QLabel("nm")
        lambda_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        lambda_layout.addWidget(self.lambda0_nm_entry)
        lambda_layout.addWidget(lambda_unit_label)
        
        wavelength_layout.addRow("Driving Wavelength:", lambda_container)
        wavelength_group.setLayout(wavelength_layout)
        single_layout.addWidget(wavelength_group)

        #-- (3) Vector Potential components section-------------
        components_group = QGroupBox("Vector Potential Components")
        components_layout = QVBoxLayout()
        components_layout.setSpacing(15)
 
        # Vector Potential component checkboxes
        field_components_layout = QHBoxLayout()
        field_components_layout.setSpacing(20)
        
        self.checkbox_Ax = QCheckBox("X Component")
        self.checkbox_Ay = QCheckBox("Y Component")
        self.checkbox_Az = QCheckBox("Z Component")
        self.checkbox_3D = QCheckBox("3 Dimension")
        
        field_components_layout.addWidget(self.checkbox_Ax)
        field_components_layout.addWidget(self.checkbox_Ay)
        field_components_layout.addWidget(self.checkbox_Az)
        field_components_layout.addWidget(self.checkbox_3D)
        field_components_layout.addStretch()
                
        components_layout.addLayout(field_components_layout)
        components_group.setLayout(components_layout)
        
        single_layout.addWidget(components_group)

        #-- (4) Corresponding field---------------
        corres_field_group = QGroupBox("Corresponding Field")
        corres_field_layout = QVBoxLayout()
        corres_field_layout.setSpacing(15)

        # Corresponding field checkboxes
        cf_layout = QHBoxLayout()
        cf_layout.setSpacing(20)
        
        self.cf_checkbox_single = QCheckBox("Corresponding field")
                
        cf_layout.addWidget(self.cf_checkbox_single)
        cf_layout.addStretch()
                
        corres_field_layout.addLayout(cf_layout)
        corres_field_group.setLayout(corres_field_layout)
        
        single_layout.addWidget(corres_field_group)
        #-----------------------------------------
        
        single_layout.addStretch()
        
        return single_widget
    
    #-(2-2)---------------------------------------
    def create_double_pulse_tab(self):
        double_widget = QWidget()
        double_layout = QVBoxLayout(double_widget)
        double_layout.setSpacing(20)
        
        #--(1) Wavelength section-------------
        wavelength_group = QGroupBox("Laser Parameters")
        wavelength_layout = QFormLayout()
        wavelength_layout.setSpacing(12)
        
        # First wavelength
        lambda1_container = QWidget()
        lambda1_layout = QHBoxLayout(lambda1_container)
        lambda1_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lambda0_nm1_entry = QLineEdit()
        self.lambda0_nm1_entry.setPlaceholderText("Enter first wavelength (e.g., 2000)")
        self.lambda0_nm1_entry.setText(str(previous_input_vector_potential.get("lambda0_nm1", "")))
        self.lambda0_nm1_entry.setMaxLength(10)
        
        lambda1_unit_label = QLabel("nm")
        lambda1_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        lambda1_layout.addWidget(self.lambda0_nm1_entry)
        lambda1_layout.addWidget(lambda1_unit_label)
        
        wavelength_layout.addRow("Driving Wavelength 1:", lambda1_container)
        
        # Second wavelength
        lambda2_container = QWidget()
        lambda2_layout = QHBoxLayout(lambda2_container)
        lambda2_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lambda0_nm2_entry = QLineEdit()
        self.lambda0_nm2_entry.setPlaceholderText("Enter second wavelength (e.g., 2000)")
        self.lambda0_nm2_entry.setText(str(previous_input_vector_potential.get("lambda0_nm2", "")))
        self.lambda0_nm2_entry.setMaxLength(10)
        
        lambda2_unit_label = QLabel("nm")
        lambda2_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        lambda2_layout.addWidget(self.lambda0_nm2_entry)
        lambda2_layout.addWidget(lambda2_unit_label)
        
        wavelength_layout.addRow("Driving Wavelength 2:", lambda2_container)
        
        wavelength_group.setLayout(wavelength_layout)
        double_layout.addWidget(wavelength_group)
        
        #-- (2) Corresponding field---------------
        corres_field_group = QGroupBox("Corresponding Field")
        corres_field_layout = QVBoxLayout()
        corres_field_layout.setSpacing(15)

        # Corresponding field checkboxes
        cf_layout = QHBoxLayout()
        cf_layout.setSpacing(20)
        
        self.cf_checkbox_double = QCheckBox("Corresponding field")
        
        cf_layout.addWidget(self.cf_checkbox_double)
        cf_layout.addStretch()
                
        corres_field_layout.addLayout(cf_layout)
        corres_field_group.setLayout(corres_field_layout)
        
        double_layout.addWidget(corres_field_group)

        #-----------------------------------------

        double_layout.addStretch()
        
        return double_widget
    
    #==================================================
   
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
        
        # Generate button
        self.generate_btn = QPushButton("Generate Plot")
        self.generate_btn.setIcon(QIcon(":/icons/analytics_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png"))
        self.generate_btn.setIconSize(QSize(24, 24))
        self.generate_btn.setDefault(True)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.on_submit) #>>>>>>>>>
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.generate_btn)
        
        layout.addLayout(button_layout)
    
    def set_button_initial_style(self):
        """Set initial light green style for both buttons"""
        light_green_style = """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #a5d6a7, stop: 1 #81c784);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 16px;
                min-height: 50px;
                margin: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #c8e6c9, stop: 1 #a5d6a7);
            }
        """
        self.single_button.setStyleSheet(light_green_style)
        self.dual_button.setStyleSheet(light_green_style)
    
    def set_button_selected_style(self, selected_button, unselected_button):
        """Set bright green for selected button and light green for unselected"""
        bright_green_style = """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #4caf50, stop: 1 #388e3c);
                color: white;
                border: 2px solid #2e7d32;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 16px;
                min-height: 50px;
                margin: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #66bb6a, stop: 1 #4caf50);
            }
        """
        light_green_style = """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #a5d6a7, stop: 1 #81c784);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 16px;
                min-height: 50px;
                margin: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #c8e6c9, stop: 1 #a5d6a7);
            }
        """
        selected_button.setStyleSheet(bright_green_style)
        unselected_button.setStyleSheet(light_green_style)
        
        self.fade_effect = QGraphicsOpacityEffect()
        self.parameters_group.setGraphicsEffect(self.fade_effect)
        
        self.fade_animation = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_animation.setDuration(0)
        
        # Setup animation for advanced options
        self.advanced_fade_effect = QGraphicsOpacityEffect()
        self.advanced_group.setGraphicsEffect(self.advanced_fade_effect)
        
        self.advanced_fade_animation = QPropertyAnimation(self.advanced_fade_effect, b"opacity")
        self.advanced_fade_animation.setDuration(0)
    
    def setup_animations(self):
        self.fade_effect = QGraphicsOpacityEffect()
        self.parameters_group.setGraphicsEffect(self.fade_effect)
        
        self.fade_animation = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_animation.setDuration(0)
    
    def clear_parameters_layout(self):
        """Clear the parameters layout"""
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
    #--------------------------------------------------
    #--------------------------------------------------
    def select_single(self):
        self.selection = 'single'
        self.show_parameters_section()
        
        # Clear and add only single pulse content
        self.clear_parameters_layout()
        self.parameters_layout.addWidget(self.single_tab)
        
        self.generate_btn.setEnabled(True)
        
        # Update button styles
        self.set_button_selected_style(self.single_button, self.dual_button)
    
    def select_dual(self):
        self.selection = 'dual'
        self.show_parameters_section()
        
        # Clear and add only double pulse content
        self.clear_parameters_layout()
        self.parameters_layout.addWidget(self.double_tab)
        
        self.generate_btn.setEnabled(True)
        
        # Update button styles
        self.set_button_selected_style(self.dual_button, self.single_button)
    #--------------------------------------------------
    #--------------------------------------------------
    def show_parameters_section(self):
        self.parameters_group.setVisible(True)
        self.fade_animation.stop()
        
        try:
            self.fade_animation.finished.disconnect()
        except TypeError:
            pass
        
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()
    
    def toggle_advanced_options(self, state):
        if state == Qt.Checked:
            self.advanced_group.setVisible(True)
            self.advanced_fade_animation.stop()
            try:
                self.advanced_fade_animation.finished.disconnect()
            except TypeError:
                pass

            self.advanced_fade_animation.setStartValue(0.0)
            self.advanced_fade_animation.setEndValue(1.0)
            self.advanced_fade_animation.start()

        else:
            self.advanced_fade_animation.stop()
            try:
                self.advanced_fade_animation.finished.disconnect()
            except TypeError:
                pass

            self.advanced_fade_animation.setStartValue(1.0)
            self.advanced_fade_animation.setEndValue(0.0)
            self.advanced_fade_animation.finished.connect(lambda: self.advanced_group.setVisible(False))
            self.advanced_fade_animation.start()
    
    def open_color_picker(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.line_color_entry.setText(color.name())
    
    def is_valid_color(self, color):
        try:
            mcolors.to_rgba(color)
            return True
        except (ValueError, ImportError):
            return False
    ###.......................................................
    def on_submit(self):
        try:
            print("*************")
            print("self.checkbox_Ax.isChecked():", self.checkbox_Ax.isChecked())
            print("self.cf_checkbox_single.isChecked():", self.cf_checkbox_single.isChecked())
            print("*************")

            plot_settings = {}
            if self.plot_options_checkbox.isChecked():
                line_color = self.line_color_entry.text().strip() or "black"
                
                if not self.is_valid_color(line_color):
                    raise ValueError(f"Invalid color value: {line_color}")
                
                plot_settings = {
                    "x_min": float(self.x_min_entry.text()) if self.x_min_entry.text() else None,
                    "x_max": float(self.x_max_entry.text()) if self.x_max_entry.text() else None,
                    "y_min": float(self.y_min_entry.text()) if self.y_min_entry.text() else None,
                    "y_max": float(self.y_max_entry.text()) if self.y_max_entry.text() else None,
                    "z_min": float(self.z_min_entry.text()) if self.z_min_entry.text() else None,
                    "z_max": float(self.z_max_entry.text()) if self.z_max_entry.text() else None,
                    "graph_title": self.graph_title_entry.text(),
                    "x_label": self.x_label_entry.text(),
                    "y_label": self.y_label_entry.text(),
                    "z_label": self.z_label_entry.text(),
                    "line_color": line_color,
                    "line_thickness": self.line_thickness_spinbox.value(),
                    "background_color": self.background_color_entry.text(),
                }
            #----------------------------------------------------------------------------------------
            if self.selection == 'single':
                lambda0_nm = float(self.lambda0_nm_entry.text())
                if not (0 <= lambda0_nm <= 10000):
                    raise ValueError("Driving wavelength must be positive and must be less than 10,000 nm.")                
                
                plot_options = {
                    'Ax': self.checkbox_Ax.isChecked(),
                    'Ay': self.checkbox_Ay.isChecked(),
                    'Az': self.checkbox_Az.isChecked(),
                    '3D': self.checkbox_3D.isChecked()
                }
                if not any(plot_options.values()):
                    raise ValueError("Invalid input", "Please select at least one vector potential component.")
                
                cf_option = {}
                if self.cf_checkbox_single.isChecked():
                    if plot_options['Ax']:
                        cf_option["fieldx"] = "fieldx"
                    if plot_options['Ay']:
                        cf_option["fieldy"] = "fieldy"
                    if plot_options['Az']:
                        cf_option["fieldz"] = "fieldz"
                    if plot_options['3D']:
                        cf_option["3D"] = "3D"
    
                    print("cf_option: ", cf_option)
                
                
                # Update
                previous_input_vector_potential["lambda0_nm"] = lambda0_nm
                
                self.accept()
                
                # CALL
                las_plot_single(lambda0_nm, plot_options, cf_option, plot_settings, self.parent().ipy_console)

            #----------------------------------------------------------------------------------------
            elif self.selection == 'dual':
                lambda0_nm1 = float(self.lambda0_nm1_entry.text())
                lambda0_nm2 = float(self.lambda0_nm2_entry.text())
                
                if not (0 <= lambda0_nm1 <= 10000) or not (0 <= lambda0_nm2 <= 10000):
                    raise ValueError("Driving wavelength must be positive and must be less than 10,000 nm.")

                # Update
                previous_input_vector_potential.update({
                    "lambda0_nm1": lambda0_nm1,
                    "lambda0_nm2": lambda0_nm2
                })

                cf_option = {"A": "A"}
                if self.cf_checkbox_double.isChecked():
                    cf_option["field"] = "field"
                    
                    print("cf_option: ", cf_option)

                self.accept()
                
                # CALL
                las_plot_dual(lambda0_nm1, lambda0_nm2, cf_option, plot_settings, self.parent().ipy_console)
                
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def select_and_plot_vector_potential(parent):
    dialog = VectorPotentialDialog(parent)
    return dialog.exec_()



            
