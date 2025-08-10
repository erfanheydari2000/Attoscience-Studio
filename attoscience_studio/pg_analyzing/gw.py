# pg_analyzing/gw.py

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
#--------------------------------
from attoscience_studio.helper_functions.constants import PhysicalConstants, AtomicUnits
TIMEau = AtomicUnits.TIMEau
CNST_1_TWcm2 = PhysicalConstants.CNST_1_TWcm2
c_au = PhysicalConstants.c_au
##----------------------------------------------------
def gate_width(lambda1_nm, lambda2_nm, intensity1, intensity2, cycles1, cycles2, eps1, eps2, delay, envelope_name, time_step):
    lambda_um1 = lambda1_nm * 1.0e-3
    lambda_um2 = lambda2_nm * 1.0e-3
    
    fiel_str1 = intensity1 * CNST_1_TWcm2
    fiel_str2 = intensity2 * CNST_1_TWcm2
    
    w01 = 0.045563 / lambda_um1
    w02 = 0.045563 / lambda_um2
    
    T01 = 2 * np.pi / w01
    T02 = 2 * np.pi / w02
        
    Iau1 = np.sqrt(intensity1 * 1.0e12) / np.sqrt(3.509470 * 1.0e16)
    Iau2 = np.sqrt(intensity2 * 1.0e12) / np.sqrt(3.509470 * 1.0e16)
    
    E0_laser1 = np.sqrt(intensity1 * 1e12) / np.sqrt(3.509470 * 1e16)
    E0_laser2 = np.sqrt(intensity2 * 1e12) / np.sqrt(3.509470 * 1e16)
    
    A0_laser1 = Iau1 * c_au / w01
    A0_laser2 = Iau2 * c_au / w02
    
    Ttot1 = cycles1 * T01
    Ttot2 = cycles2 * T02
    Tdelay = delay * T01
    Tst1 = 0 * T01
    
    Tst2 = Tst1 + (Ttot1 / 2) + Tdelay - (Ttot2 / 2)
    Tend1 = Tst1 + Ttot1
    Tend2 = Tst2 + Ttot2
    
    tm1 = Tst1 + Ttot1 / 2
    tm2 = Tst2 + Ttot2 / 2 
    
    dt = time_step
    t = np.arange(0, Tend2 + dt, dt)
    max_iter = int(Tend2 / dt) * 2
    Time_OC = t / T01
    
    def heaviside(x):
        return 1.0 * (x >= 0)
    
    if envelope_name == 'Sine_square':
        envelope_func1 = lambda t: A0_laser1 * np.sin(np.pi * (t - Tst1) / Ttot1)**2 * heaviside(Tend1 - t) * heaviside(t - Tst1)
        envelope_func2 = lambda t: A0_laser2 * np.sin(np.pi * (t - Tst2) / Ttot2)**2 * heaviside(Tend2 - t) * heaviside(t - Tst2)
        
        ellipt_func = lambda t: (envelope_func1(t) - envelope_func2(t)) / (envelope_func1(t) + envelope_func2(t))

    elif envelope_name == 'Gaussian':
        envelope_func1 = lambda t: A0_laser1 * np.exp(-2 * np.log(2) * ((t - tm1) / (Ttot1 / 2))**2)
        envelope_func2 = lambda t: A0_laser2 * np.exp(-2 * np.log(2) * ((t - tm2) / (Ttot2 / 2))**2)
        
        ellipt_func = lambda t: (envelope_func1(t) - envelope_func2(t)) / (envelope_func1(t) + envelope_func2(t))   
    
    time_dep_ellipt = np.array([ellipt_func(ti) for ti in t])
    
    return w01, w02, T01, T02, t, Time_OC, time_dep_ellipt

##----------------------------------------------------
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")
                        
def gw_connector(lambda1_nm, lambda2_nm, intensity1, intensity2, cycles1, cycles2, eps1, eps2, 
                 ellipticity_threshold, delay, envelope_name, time_step, plot_settings, extract_data_option, ipy_console=None):
    try:
                 
        w01, w02, T01, T02, t, Time_OC, time_dep_ellipt = gate_width(lambda1_nm, lambda2_nm, intensity1, intensity2, cycles1, cycles2, eps1, eps2, delay, envelope_name, time_step)
        
        plot_gw(lambda1_nm, lambda2_nm, intensity1, intensity2, cycles1, cycles2, eps1, eps2, ellipticity_threshold, delay, 
                envelope_name, time_step, w01, w02, T01, T02, t, Time_OC, time_dep_ellipt,
                plot_settings, extract_data_option)
 
        T01_SI = T01 * TIMEau
        T02_SI = T02 * TIMEau

        max_Time_OC = np.max(Time_OC)
        
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        msg = (
            f">>> Time                          {timestamp}\n"
            + "-" * 75 + "\n"
            + "--                          GW log!                          --\n"
            + "-" * 75 + "\n"
            ">>> GW calculation and visualization successfully completed\n"
            f">>> T1 [a.u.]: {T01:.12e}\n"
            f">>> T2 [a.u.]: {T02:.12e}\n"
            f">>> T1 [second]: {T01_SI:.12e}\n"
            f">>> T2 [second]: {T02_SI:.12e}\n"
            f">>> w1 [a.u.]: {w01:.12e}\n"
            f">>> w2 [a.u.]: {w02:.12e}\n"
            f">>> Max Time_OC [o.c.]: {max_Time_OC:.12e}\n"
            f">>> Delay factor between two pulses: {delay:.12e}\n"
            + "-" * 75
        )
        print_to_console(ipy_console, msg)

    except ValueError as e:
        QMessageBox.warning(None, "Error", str(e))
        return

##----------------------------------------------------
def plot_gw(lambda1_nm, lambda2_nm, intensity1, intensity2, cycles1, cycles2, eps1, eps2, ellipticity_threshold, delay, 
                envelope_name, time_step, w01, w02, T01, T02, t, Time_OC, time_dep_ellipt,
                plot_settings, extract_data_option):
    if 'extract_data' in extract_data_option:
        file_path = "gw.txt"
        header = (
            f"# Driving wavelength 1 [nm]: {lambda1_nm}\n"
            f"# Driving wavelength 2 [nm]: {lambda2_nm}\n"
            f"# Intensity 1 [TW/cm2]:      {intensity1}\n"
            f"# Intensity 2 [TW/cm2]:      {intensity2}\n"
            f"# Cycles 1:                  {cycles1}\n"
            f"# Cycles 2:                  {cycles2}\n"
            f"# Ellipticity 1:             {eps1}\n"
            f"# Ellipticity 2:             {eps2}\n"
            f"# Delay:                     {delay}\n"
            f"# T01 [a.u.]:                {T01}\n"
            f"# T02 [a.u.]:                {T02}\n"
            f"# w01 [a.u.]:                {w01}\n"
            f"# w02 [a.u.]:                {w02}\n"
            f"# Envelope:                  {envelope_name}\n"
            f"# Time Step:                 {time_step}\n"
            "# t [a.u.],   time_dep_ellipt [a.u.]\n"
            + '#' * 80
        )
        np.savetxt(file_path, np.column_stack((t, time_dep_ellipt)), header=header, comments='', fmt='%.12e')

    abs_time_dep_ellipt = np.abs(time_dep_ellipt)
   
    plt.figure()
    bg_color = plot_settings.get("background_color", "white")
    plt.gcf().patch.set_facecolor(bg_color)
    plt.plot(Time_OC, abs_time_dep_ellipt, color=plot_settings.get("line_color", "darkslategray"), linewidth=plot_settings.get("line_thickness", 1.4), alpha=0.6)
    plt.fill_between(Time_OC, abs_time_dep_ellipt, alpha=0.3)
    plt.axhline(ellipticity_threshold, color=[0.588235294117647, 0.541176470588235, 0.541176470588235], linewidth=0.4)
    plt.title(plot_settings.get("graph_title", "Time Dependent Ellipticity"))
    plt.xlabel(plot_settings.get("x_label", "Time [a.u.]"))
    plt.ylabel(plot_settings.get("y_label", r'|$\xi$|'))
    plt.xlim(left=plot_settings.get("x_min", np.min(Time_OC)), right=plot_settings.get("x_max", np.max(Time_OC)))
    plt.ylim(bottom=plot_settings.get("y_min", 0.0), top=plot_settings.get("y_max", 1.0))
    plt.grid(False)
    plt.show()

##----------------------------------------------------

previous_input_gw = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gate Width")
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
        title = QLabel("Gate Width")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")

        subtitle = QLabel("Enter the parameters to calculate the gate width.")
        subtitle.setStyleSheet("font-size: 13px; color: #666; margin: 0;")

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

        ###--------------------------------
        self._create_pulse_parameters_section(required_layout) ###>>>>>>>
    
        ###--------------------------------
        self._create_timing_envelope_section(required_layout) ###>>>>>>>
    
        ####--------------------------------
        self._create_display_options_section(required_layout) ###>>>>>>>
    
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)

    def _create_pulse_parameters_section(self, parent_layout):
    
        # Pulse 1 Parameters ------------------------
        pulse1_group = QGroupBox("Pulse 1 Parameters")
        pulse1_layout = QFormLayout()
        pulse1_layout.setSpacing(12)
        pulse1_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    
        # pulse 1 parameters
        pulse1_params = [
            ("lambda1_nm", "Driving Wavelength:", "Enter the driving wavelength (e.g., 2000)", "[nm]"),
            ("I_TWcm2_1", "Intensity:", "Enter the intensity (e.g., 1.0)", "[TW/cm^2]"),
            ("Cycles1", "Number of Cycles:", "Enter the number of cycles (e.g., 5)", "[o.c.]"),
            ("eps1", "Ellipticity:", "Enter the ellipticity (e.g., 0.5)", "")
        ]
    
        self._add_parameter_fields(pulse1_layout, pulse1_params, "1")
        pulse1_group.setLayout(pulse1_layout)
        parent_layout.addWidget(pulse1_group)
    
        # Pulse 2 Parameters
        pulse2_group = QGroupBox("Pulse 2 Parameters")
        pulse2_layout = QFormLayout()
        pulse2_layout.setSpacing(12)
        pulse2_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    
        # pulse 2 parameters
        pulse2_params = [
            ("lambda2_nm", "Driving Wavelength:", "Enter the driving wavelength (e.g., 2000)", "[nm]"),
            ("I_TWcm2_2", "Intensity:", "Enter the intensity (e.g., 1.0)", "[TW/cm^2]"),
            ("Cycles2", "Number of Cycles:", "Enter the number of cycles (e.g., 5)", "[o.c.]"),
            ("eps2", "Ellipticity:", "Enter the ellipticity (e.g., 0.5)", "")
        ]
    
        self._add_parameter_fields(pulse2_layout, pulse2_params, "2")
        pulse2_group.setLayout(pulse2_layout)
        parent_layout.addWidget(pulse2_group)

    def _create_timing_envelope_section(self, parent_layout):
        timing_group = QGroupBox("")
        timing_layout = QFormLayout()
        timing_layout.setSpacing(12)
        timing_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    
        # delay
        delay_container = self._create_parameter_container(
            "delay_entry", "delay", "Enter the required delay (e.g., 1.75)", ""
        )
        timing_layout.addRow("Delay:", delay_container)
 
        # ellipticity threshold 
        ellipticity_threshold_container = self._create_parameter_container(
            "ellipticity_threshold_entry", "ellipticity_threshold", "Enter the ellipticity threshold (e.g., 0.2)", ""
        )
        timing_layout.addRow("Ellipticity threshold:", ellipticity_threshold_container)
    
        # Envelope
        envelope_container = QWidget()
        envelope_layout = QHBoxLayout(envelope_container)
        envelope_layout.setContentsMargins(0, 0, 0, 10)
    
        self.envelope_entry = QComboBox()
        self.envelope_entry.addItems(["Sine_square", "Gaussian"])
        self.envelope_entry.setFixedSize(120, 30)
        self.envelope_entry.setStyleSheet("QComboBox { min-height: 30px; font-size: 12px; }")
    
        envelope_spacer = QLabel("")
        envelope_spacer.setStyleSheet("min-width: 30px;")
    
        envelope_layout.addWidget(self.envelope_entry)
        envelope_layout.addWidget(envelope_spacer)
    
        timing_layout.addRow("Envelope:", envelope_container)
    
        timing_group.setLayout(timing_layout)
        parent_layout.addWidget(timing_group)

    def _create_display_options_section(self, parent_layout):
        display_options_group = QGroupBox("Display Options")
        display_options_layout = QHBoxLayout()
        display_options_layout.setSpacing(15)
    
        # Additional options -----> left
        options_subgroup = QGroupBox("Additional Options")
        options_layout = QVBoxLayout()
        options_layout.setContentsMargins(10, 10, 10, 10)
    
        self.extract_data_checkbox = QCheckBox("Extract Data")
        options_layout.addWidget(self.extract_data_checkbox)
        options_layout.addStretch()
    
        options_subgroup.setLayout(options_layout)
        display_options_layout.addWidget(options_subgroup)
    
        # Time step -----> right
        time_step_subgroup = QGroupBox("Time Step Configuration")
        time_step_layout = QVBoxLayout()
        time_step_layout.setContentsMargins(10, 10, 10, 10)
    
        time_step_label = QLabel("Time Step:")
        self.time_step_spinbox = QDoubleSpinBox()
        self.time_step_spinbox.setRange(0.00001, 10.00000)
        self.time_step_spinbox.setValue(previous_input_gw.get("time_step", 0.2))
        self.time_step_spinbox.setSuffix("")
        self.time_step_spinbox.setDecimals(5)
    
        ###------------------
        light_blue_style = """
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            background-color: #ADD8E6;
            border: 1px solid #8AB6D6;
            width: 16px;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #BEE3F7;
        }
        """
        self.time_step_spinbox.setStyleSheet(light_blue_style)
    
        time_step_layout.addWidget(time_step_label)
        time_step_layout.addWidget(self.time_step_spinbox)
        time_step_layout.addStretch()
    
        time_step_subgroup.setLayout(time_step_layout)
        display_options_layout.addWidget(time_step_subgroup)
    
        display_options_group.setLayout(display_options_layout)
        parent_layout.addWidget(display_options_group)

    def _add_parameter_fields(self, layout, params, pulse_num):
        """Helper method to add parameter fields to a layout"""
        for param_key, label, placeholder, unit in params:
            # entry attribute
            if param_key == "lambda1_nm":
                entry_attr_name = "lambda1_nm_entry"
            elif param_key == "lambda2_nm":
                entry_attr_name = "lambda2_nm_entry"
            elif param_key == "I_TWcm2_1":
                entry_attr_name = "intensity1_entry"
            elif param_key == "I_TWcm2_2":
                entry_attr_name = "intensity2_entry"
            elif param_key == "Cycles1":
                entry_attr_name = "cycles1_entry"
            elif param_key == "Cycles2":
                entry_attr_name = "cycles2_entry"
            elif param_key == "eps1":
                entry_attr_name = "eps1_entry"
            elif param_key == "eps2":
                entry_attr_name = "eps2_entry"
            elif param_key == "ellipticity_threshold":
                entry_attr_name = "ellipticity_threshold_entry"
            else:
                entry_attr_name = f"{param_key}_entry"
            
            container = self._create_parameter_container(
                entry_attr_name,
                param_key,
                placeholder,
                unit
            )
            layout.addRow(label, container)

    def _create_parameter_container(self, entry_attr_name, param_key, placeholder, unit):
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
    
        # entry field--------------
        entry = QLineEdit()
        entry.setPlaceholderText(placeholder)
        entry.setText(str(previous_input_gw.get(param_key, "")))
        entry.setMaxLength(5)
        
        ##-----------------------------------
        setattr(self, entry_attr_name, entry)
        ##-----------------------------------
        
        # unit label--------------
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
    
        container_layout.addWidget(entry)
        container_layout.addWidget(unit_label)
    
        return container
    
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
        submit_btn.setIcon(QIcon(":/icons/analytics_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png"))
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
            lambda1_nm = float(self.lambda1_nm_entry.text())
            if not (0 <= lambda1_nm <= 10000):
                raise ValueError("Driving wavelength must be positive and must be less than 10,000 nm.")
            previous_input_gw["lambda1_nm"] = lambda1_nm
    
            lambda2_nm = float(self.lambda2_nm_entry.text())
            if not (0 <= lambda2_nm <= 10000):
                raise ValueError("Driving wavelength must be positive and must be less than 10,000 nm.")
            previous_input_gw["lambda2_nm"] = lambda2_nm
    
            intensity1 = float(self.intensity1_entry.text())
            intensity2 = float(self.intensity2_entry.text())
            if intensity1 <= 0 or intensity2 <= 0:
                raise ValueError("Intensity must be a positive number.")
            previous_input_gw["I_TWcm2_1"] = intensity1
            previous_input_gw["I_TWcm2_2"] = intensity2

            cycles1 = float(self.cycles1_entry.text())
            cycles2 = float(self.cycles2_entry.text())
            if cycles1 <= 0 or cycles2 <= 0:
                raise ValueError("Number of cycle must be a positive number.")
            previous_input_gw["Cycles1"] = cycles1
            previous_input_gw["Cycles2"] = cycles2

            eps1 = float(self.eps1_entry.text())
            eps2 = float(self.eps2_entry.text())
            if not (-1 <= eps1 <= 1) or not (-1 <= eps2 <= 1):
                raise ValueError("Ellipticity must be between -1 and 1.")
            previous_input_gw["eps1"] = eps1
            previous_input_gw["eps2"] = eps2

            ellipticity_threshold = float(self.ellipticity_threshold_entry.text())
            if not (0 <= ellipticity_threshold <= 1):
                raise ValueError("Ellipticity threshold must be between 0 and 1.")
            previous_input_gw["ellipticity_threshold"] = ellipticity_threshold

            delay = float(self.delay_entry.text())
            previous_input_gw["delay"] = delay

            time_step = float(self.time_step_spinbox.value())
            if time_step <= 0.00001:
                raise ValueError("time_step cannot be less than 0.00001.")
            previous_input_gw["time_step"] = time_step

            envelope_name = self.envelope_entry.currentText()
            previous_input_gw["Envelope"] = envelope_name

            plot_settings = {}
            if hasattr(self, 'plot_options_checkbox') and self.plot_options_checkbox.isChecked():
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

            extract_data_option = []
            if self.extract_data_checkbox.isChecked():
                extract_data_option.append('extract_data')

            self.accept()
        
            # Update
            previous_input_gw.update({
                "lambda1_nm": lambda1_nm, 
                "lambda2_nm": lambda2_nm,
                "I_TWcm2_1": intensity1,
                "I_TWcm2_2": intensity2,
                "Cycles1": cycles1,
                "Cycles2": cycles2,
                "eps1": eps1, 
                "eps2": eps2, 
                "ellipticity_threshold": ellipticity_threshold, 
                "delay": delay, 
                "Envelope": envelope_name,  
                "time_step": time_step
            })

            # Call
            gw_connector(lambda1_nm, lambda2_nm, intensity1, intensity2, cycles1, cycles2,
                        eps1, eps2, ellipticity_threshold, delay, envelope_name, time_step, 
                        plot_settings, extract_data_option,
                        self.parent().ipy_console)
                 
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred: {e}")

def input_dialog_gw(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()



