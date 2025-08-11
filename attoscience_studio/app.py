# app.py

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

# This program includes computational code and guidance provided by Mohammad Monfared.

import os, sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import plotly.graph_objects as go
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import animation
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar, 
                             QRadioButton, QButtonGroup, QScrollArea, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QStyle, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QFontMetrics, QPen, QPainter
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication, QSharedMemory, QPropertyAnimation, QEasingCurve, QRect, pyqtProperty
from numpy import trapz
from scipy.integrate import trapezoid, quad
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from matplotlib.widgets import Slider, Button
from joblib import Parallel, delayed
from functools import partial
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
#--------------------------------
from attoscience_studio.styles.styles import *
#--------------------------------
from attoscience_studio.helper_functions.helpers import *
from attoscience_studio.helper_functions.constants import PhysicalConstants, AtomicUnits
#--------------------------------
from attoscience_studio.gs.visualize_parser import *
from attoscience_studio.gs.bstr import *
from attoscience_studio.gs.dos import *
from attoscience_studio.gs.dns import *
#--------------------------------
from attoscience_studio.driving_field.electric_field import *
from attoscience_studio.driving_field.vector_potential import *
#--------------------------------
from attoscience_studio.high_harmonic.total_current import *
from attoscience_studio.high_harmonic.hhg_spectrum import *
from attoscience_studio.high_harmonic.hhg_yield import *
from attoscience_studio.high_harmonic.hhg_ellips import *
from attoscience_studio.high_harmonic.hhg_phs import *
#--------------------------------
from attoscience_studio.attosecond_pulse.atto_pulse import *
from attoscience_studio.attosecond_pulse.gtf import *
from attoscience_studio.attosecond_pulse.find_MPW import *
#--------------------------------
from attoscience_studio.pg_analyzing.pg import *
from attoscience_studio.pg_analyzing.gw import *
#--------------------------------
from attoscience_studio.electron_dynamics.nex import *
from attoscience_studio.electron_dynamics.BZ_Nex import *
from attoscience_studio.electron_dynamics.BZ_Current import *
from attoscience_studio.electron_dynamics.nex_anim import *
#--------------------------------
from attoscience_studio.tool_box.unit import *
from attoscience_studio.tool_box.ftfunc import *
from attoscience_studio.tool_box.ftdata import *
from attoscience_studio.tool_box.iftfunc import *
from attoscience_studio.tool_box.iftdata import *
#--------------------------------
from attoscience_studio.utils.real_time_manitoring import *
from attoscience_studio.utils.single_instance import SingleInstance
from attoscience_studio.utils.anim_controller import AnimationController
from attoscience_studio.utils.status_symbols import Symbols
#--------------------------------
from attoscience_studio.resources_rc import *
#--------------------------------
def open_url(url):
    webbrowser.open_new(url)
#--------------------------------
def is_valid_color(value):
    color = QColor(value)
    return color.isValid()
#----------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Attoscience Studio 1.0.0')
        self.setGeometry(300, 150, 1500, 1100)
        self.setStyleSheet(StyleManager.get_main_window_style())
       
        self.current_theme = "dark"
        self.current_font_size = "Medium"
       
        self.accent_color = StyleManager.ACCENT_COLOR
        
        self.recent_activities = []
        self.data_summaries = []
        
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()
        #############################
        self.info_panel = QTextEdit()
        #############################
        header_layout = QHBoxLayout()

        logo_label = QLabel()
        logo_pixmap = QPixmap(":/icons/logo.png").scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        header_layout.addWidget(logo_label)

        # Tagline
        tagline_layout = QVBoxLayout()
        tagline = QLabel("Attoscience Studio 1.0.0")
        tagline.setFont(StyleManager.HEADER_FONT)
        tagline.setAlignment(Qt.AlignLeft)
        
        tagline.setMinimumSize(100, 5)
        tagline.setStyleSheet(f"color: {StyleManager.TEXT_COLOR};")
        
        tagline.setObjectName("tagline") 
        tagline_layout.addWidget(tagline)

        header_layout.addLayout(tagline_layout)
        header_layout.addStretch(1)

        # Settings Menu
        self.create_settings_menu()
        header_layout.addWidget(self.settings_button)
        self.main_layout.addLayout(header_layout)
        
        #----------------------------------------------------------------
        ## Central Widget:    Tab widget
        #----------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(StyleManager.get_tabs_style())
        self.tabs.setFont(StyleManager.TAB_FONT)
        self.tabs.addTab(self.create_home_tab(), "Home")
        self.tabs.addTab(self.create_ground_state_tab(), "Ground State")
        self.tabs.addTab(self.create_driving_field_tab(), "Driving Pulse")
        self.tabs.addTab(self.create_high_harmonic_tab(), "High Harmonic")
        self.tabs.addTab(self.create_attosecond_pulse_tab(), "Attosecond Pulse")
        self.tabs.addTab(self.electron_dynamics_tab(), "Dynamics")
        self.tabs.addTab(self.create_tool_box_tab(), "Tool Box")
        self.tabs.addTab(self.create_help_tab(), "Help") 

        self.central_layout = QVBoxLayout()
        self.central_layout.addWidget(self.tabs)
        
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.central_layout)
        self.main_layout.addWidget(self.central_widget)
        self.central_widget.setMinimumSize(500, 600)

        self.central_widget.setStyleSheet(StyleManager.get_central_widget())

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)
        
        #----------------------------------------------------------------
        ### IPython terminal 
        #----------------------------------------------------------------
        self.ipy_console = self.make_ipython_widget()
        self.ipy_console.setMinimumHeight(200)  #--->>> self.central_widget.setMinimumSize(,) is the REFERENCE 
        self.main_layout.addWidget(self.ipy_console)

        self.ipy_console.push_variables({
            'main_window': self,
            'label': self.label if hasattr(self, 'label') else None,
            'button': self.button if hasattr(self, 'button') else None,
        })

        self.ipy_console._kernel.shell.run_cell("%config TerminalInteractiveShell.colors = 'Linux'")

    def update_label(self):
        self.label.setText("Label updated!")

    def make_ipython_widget(self):
        kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel = kernel_manager.kernel
        kernel.gui = 'qt'
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        ipython_widget = RichJupyterWidget()
        ####
        ipython_widget.setStyleSheet(StyleManager.get_terminal_style())
        ####
        ipython_widget.kernel_manager = kernel_manager
        ipython_widget.kernel_client = kernel_client

        ipython_widget._kernel = kernel
        def push_variables(variables):
            kernel.shell.push(variables)
        ipython_widget.push_variables = push_variables

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
        ipython_widget.exit_requested.connect(stop)
        return ipython_widget
        
    #----------------------------------------------------------------
    def create_settings_menu(self):
        self.settings_button = QToolButton(self)

        self.settings_button.setIcon(getattr(QtGui, 'QIcon')(':/icons/settings_icon.png'))
        self.settings_button.setIconSize(QSize(18, 18))

        self.settings_button.setStyleSheet(StyleManager.create_settings_menu__button()) ###>>>>>>>>>>>>

        self.settings_menu = QMenu(self)

        self.settings_menu.setStyleSheet(StyleManager.create_settings_menu__itsmenu())  ###>>>>>>>>>>>>

        #self.settings_menu.addAction("Appearance Settings", self.open_appearance_settings)
        #self.settings_menu.addAction("Reset Preferences", self.reset_preferences)
        self.settings_menu.addAction("About", self.show_about_dialog)

        self.settings_button.clicked.connect(self.show_settings_menu) ###>>>>>>>>>>>>

        header_layout = QHBoxLayout()
        header_layout.addStretch(1)
        header_layout.addWidget(self.settings_button)
        self.main_layout.addLayout(header_layout)

    def show_settings_menu(self):
        pos = self.settings_button.mapToGlobal(self.settings_button.rect().bottomRight())
        screen_geometry = QApplication.primaryScreen().availableGeometry()

        menu_width = self.settings_menu.sizeHint().width()
        menu_height = self.settings_menu.sizeHint().height()

        if pos.x() + menu_width > screen_geometry.right():
            pos.setX(screen_geometry.right() - menu_width)

        if pos.y() + menu_height > screen_geometry.bottom():
            pos.setY(screen_geometry.bottom() - menu_height)

        if pos.x() < screen_geometry.left():
            pos.setX(screen_geometry.left())

        if pos.y() < screen_geometry.top():
            pos.setY(screen_geometry.top())

        self.settings_menu.exec_(pos)
        
    def open_appearance_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Appearance Settings")
        dialog.setFixedSize(300, 300)

        theme_label = QLabel("Select Theme:", dialog)
        theme_combobox = QComboBox(dialog)
        theme_combobox.addItems(["Light", "Dark"])
        theme_combobox.setCurrentText("Dark" if self.current_theme == "dark" else "Light")

        font_label = QLabel("Select Font Size:", dialog)
        font_combobox = QComboBox(dialog)
        font_combobox.addItems(["Small", "Medium", "Large"])
        font_combobox.setCurrentText(self.current_font_size)

        color_label = QLabel("Select Accent Color:", dialog)
        color_button = QPushButton("Choose Color", dialog)
        current_color_label = QLabel(f"Current Color: {self.accent_color}", dialog)
        color_button.clicked.connect(lambda: self.pick_accent_color(current_color_label)) ###>>>>>>>>>>>>

        save_button = QPushButton("Save", dialog)
        cancel_button = QPushButton("Cancel", dialog)
        save_button.clicked.connect(lambda: self.save_settings(theme_combobox, font_combobox, self.tabs)) ###>>>>>>>>>>>>
        cancel_button.clicked.connect(dialog.reject)

        layout = QVBoxLayout()
        layout.addWidget(theme_label)
        layout.addWidget(theme_combobox)
        layout.addWidget(font_label)
        layout.addWidget(font_combobox)
        layout.addWidget(color_label)
        layout.addWidget(color_button)
        layout.addWidget(current_color_label)
        layout.addWidget(save_button)
        layout.addWidget(cancel_button)
        dialog.setLayout(layout)

        dialog.exec_()

    def pick_accent_color(self, current_color_label):
        """Open a color picker and update the accent color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.accent_color = color.name()
            current_color_label.setText(f"Current Color: {self.accent_color}")

    def apply_theme(self, font_size, accent_color, is_dark_mode):
        
        main_window_style = StyleManager.get_main_window_style() ###>>>>>>>>>>>>
        
        central_widget_style = StyleManager.get_central_widget() ###>>>>>>>>>>>>

        tab_styles = StyleManager.get_tabs_style() ###>>>>>>>>>>>>
        
        # (I)
        main_window_style = main_window_style.replace(
            "#1e1e1e", "#D9D9D9" if not is_dark_mode else "#1e1e1e"
        )
        main_window_style = main_window_style.replace(
            "#ffffff", "#000000" if not is_dark_mode else "#ffffff"
        )
        # (II)
        central_widget_style = central_widget_style.replace(
            "#121212", "#FFFFFF" if not is_dark_mode else "#121212"
        )
        # (III):
        tab_styles = tab_styles.replace(
            f"background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 {StyleManager.BACKGROUND_COLOR}, stop: 1 {StyleManager.ACCENT_COLOR});",
            f"background-color: {'#D9D9D9' if not is_dark_mode else StyleManager.BACKGROUND_COLOR};"
            f"color: {'#000000' if not is_dark_mode else '#ffffff'};"
        )
          
        self.setStyleSheet(main_window_style)
        self.central_widget.setStyleSheet(central_widget_style)
        self.tabs.setStyleSheet(tab_styles)

    def save_settings(self, theme_combobox, font_combobox, tabs):
        selected_theme = theme_combobox.currentText()
        font_size = font_combobox.currentText()
        font_size_map = {"Small": 10, "Medium": 12, "Large": 14}

        self.current_theme = "dark" if selected_theme == "Dark" else "light"
        self.current_font_size = font_size
        self.accent_color = self.accent_color

        self.apply_theme(font_size_map[font_size], self.accent_color, self.current_theme == "dark") ###>>>>>>>>>>>>


    def reset_preferences(self):
        response = QMessageBox.question(
            self,
            "Reset Preferences",
            "Are you sure you want to reset all preferences to default values?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if response == QMessageBox.Yes:
            self.current_theme = "dark"
            self.current_font_size = "Medium"
            self.accent_color = StyleManager.ACCENT_COLOR #"#3498db"  # Default accent color

            default_font_size = 12
            self.apply_theme(default_font_size, self.accent_color, True)

            self.setStyleSheet(StyleManager.get_main_window_style())

            tagline = self.findChild(QLabel, "tagline")

            tab_widget = self.findChild(QTabWidget, "tab_widget")

            if tagline:
                tagline.setFont(StyleManager.HEADER_FONT)

            if tab_widget:
                tabs.setStyleSheet(StyleManager.get_tabs_style())
                tabs.tabBar().setFont(StyleManager.TAB_FONT)

            QMessageBox.information(self, "Preferences Reset", "Preferences have been reset to default, except for fixed font items.")

    def show_about_dialog(self):
        about_text = """
        <h2>Attoscience Studio</h2>
        <p><b>Version:</b> 1.0.0</p>
        <p>This software is designed for attoscience simulations.
        <p>For more information, visit: 
        <a href='https://github.com/'>GitHub Repository</a></p>
        """
        #my-repo in 
        QMessageBox.about(self, "About", about_text)
        
    #----------------------------------------------------------------
    
    def log_activity(self, activity):
        self.recent_activities.append(activity)
        if len(self.recent_activities) > 5:
            self.recent_activities.pop(0)
        self.update_home_tab()

    def log_data_summaries(self, summary):
        self.data_summaries.append(summary)
        if len(self.data_summaries) > 2:
            self.data_summaries.pop(0)
        self.update_home_tab()

    #----------------------------------------------------------------
    
    def create_home_tab(self):
        widget = QWidget()

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        right_header_label = QLabel("Project Overview")
        right_header_label.setFont(QFont('Arial', 12, QFont.Bold))
        right_header_label.setStyleSheet("color: #f0f0f0; padding: 10px; border-bottom: 2px solid #444444; margin-bottom: 10px;")
        right_header_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_header_label)

        self.activities_layout = QVBoxLayout()
        activities_label = QLabel("Recent Activities")
        activities_label.setFont(QFont('Arial', 12, QFont.Bold))
        activities_label.setStyleSheet("color: #f0f0f0; padding: 10px;")
        self.activities_layout.addWidget(activities_label)

        if self.recent_activities:
            for activity in self.recent_activities[-5:]:
                label = QLabel(f"- {activity}")
                label.setFont(QFont('Arial', 10))
                label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
                self.activities_layout.addWidget(label)
        else:
            label = QLabel("No recent activities.")
            label.setFont(QFont('Arial', 10))
            label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
            self.activities_layout.addWidget(label)
    
        self.summaries_layout = QVBoxLayout()
        summaries_label = QLabel("Data Summaries")
        summaries_label.setFont(QFont('Arial', 12, QFont.Bold))
        summaries_label.setStyleSheet("color: #f0f0f0; padding: 10px;")
        self.summaries_layout.addWidget(summaries_label)

        if self.data_summaries:
            for summary in self.data_summaries[-3:]:
                label = QLabel(f"- {summary}")
                label.setFont(QFont('Arial', 10))
                label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
                self.summaries_layout.addWidget(label)
        else:
            label = QLabel("No data summaries available.")
            label.setFont(QFont('Arial', 10))
            label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
            self.summaries_layout.addWidget(label)

        right_layout.addLayout(self.activities_layout)
        right_layout.addStretch(1)
        right_layout.addLayout(self.summaries_layout)

        right_layout.addStretch(1)
    
        right_widget.setMaximumWidth(400)
        right_widget.setMinimumWidth(350)
    
        ##----------------------------------------------------
        self.system_monitor_widget = SystemMonitorWidget()
        ##----------------------------------------------------
    
        main_layout.addWidget(self.system_monitor_widget)
        main_layout.addWidget(right_widget)

        widget.setLayout(main_layout)
        return widget

    def update_home_tab(self):
        home_tab = self.tabs.widget(0)

        main_layout = home_tab.layout()

        right_widget = main_layout.itemAt(1).widget()
        right_layout = right_widget.layout()

        for layout in [self.activities_layout, self.summaries_layout]:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        recent_activities_label = QLabel("Recent Activities")
        recent_activities_label.setFont(QFont('Arial', 12, QFont.Bold))
        recent_activities_label.setStyleSheet("color: #f0f0f0; padding: 10px;")
        self.activities_layout.addWidget(recent_activities_label)

        if self.recent_activities:
            for activity in reversed(self.recent_activities[-5:]):
                label = QLabel(f"- {activity}")
                label.setFont(QFont('Arial', 10))
                label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
                self.activities_layout.addWidget(label)
        else:
            label = QLabel("No recent activities.")
            label.setFont(QFont('Arial', 10))
            label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
            self.activities_layout.addWidget(label)

        data_summaries_label = QLabel("Data Summaries")
        data_summaries_label.setFont(QFont('Arial', 12, QFont.Bold))
        data_summaries_label.setStyleSheet("color: #f0f0f0; padding: 10px;")
        self.summaries_layout.addWidget(data_summaries_label)

        if self.data_summaries:
            for summary in reversed(self.data_summaries[-3:]):
                label = QLabel(f"- {summary}")
                label.setFont(QFont('Arial', 10))
                label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
                self.summaries_layout.addWidget(label)
        else:
            label = QLabel("No data summaries available.")
            label.setFont(QFont('Arial', 10))
            label.setStyleSheet("color: #bbbbbb; padding-left: 15px;")
            self.summaries_layout.addWidget(label)

    #----------------------------------------------------------------
    def create_ground_state_tab(self):
        buttons = [
            ("Crystal Structure", self.show_crystal_structure, """ EXPERIMENTAL !"""),
            ("Band Structure", self.show_band_structure_dialog, """
                <h3>Band Structure</h3>
                <p>This feature allows you to visualize the electronic band structure of a material. 
                It reads data from a 'bandstructure' file, calculates the band gap, and plots the energy bands relative to the wave vector.</p>
                <h4>Features:</h4>
                <ul>
                    <li>Calculate and display the band gap (direct or indirect).</li>
                    <li>Customizable plot appearance, including axis limits, labels, line colors, and thickness.</li>
                    <li>Supports visualizing multiple bands with user-defined settings.</li>
                </ul>
                <h4>Inputs:</h4>
                <ul>
                    <li><b>Fermi Energy:</b> Specify the Fermi energy in Hartree units to analyze bands above and below this level.</li>
                    <li><b>Number of Bands:</b> Select how many energy bands to include in the plot.</li>
                    <li><b>Plot Customization:</b> Optionally set axis ranges, labels, line properties, and background color.</li>
                </ul>
                <h4>How to Use:</h4>
                <ol>
                    <li>Upload a valid 'bandstructure' file.</li>
                    <li>Provide the required input parameters (Fermi energy and number of bands).</li>
                    <li>Optionally customize plot settings for a tailored visualization.</li>
                    <li>View the calculated band gap details and the plotted band structure.</li>
                </ol>
                <h4>Notes:</h4>
                <ul>
                    <li>Ensure the input file contains correctly formatted band structure data.</li>
                    <li>Default values are used for any unspecified plot settings.</li>
                </ul>
            """),
            ("Density of State", self.density_of_state, """
                <h3>Density of State</h3>
                <p>
                    The Density of State (DOS) panel allows users to plot the total density of states for a given material. 
                    Users can upload a DOS file (e.g., <code>total-dos.dat</code>) and customize the plot appearance with 
                    options for axis limits, graph title, axis labels, line color, line thickness, and background color.
                </p>
                <p>
                    To use this feature:
                    <ol>
                        <li>Click the 'Browse' button to select a valid DOS file.</li>
                        <li>Ensure the file contains valid energy and density data.</li>
                        <li>Adjust plot settings as needed using the provided input dialog.</li>
                        <li>Click 'Submit' to generate the plot.</li>
                    </ol>
                </p>
                <p>
                    The plot will display the energy values on the y-axis (in atomic units) and the total density of states 
                    on the x-axis. Additional options, such as line and background color, can be configured for better 
                    visualization.
                </p>
            """),
            ("Electron Density", self.density, """
                <h3>Electron Density</h3>
                <p>The electron density represents the spatial distribution of electrons within the material. 
                It is a critical property for understanding the electronic structure and behavior of the system. 
                The displayed data typically results from quantum simulations, showing the probability density 
                of finding electrons in specific regions of space. Higher density regions indicate areas with a 
                greater likelihood of electron presence.</p>
                <p><strong>Visualization:</strong> This feature allows users to view and analyze the electron 
                density distribution across different spatial dimensions.</p>
            """)
        ]

        layout, button_widgets = create_buttons_with_info_panel(buttons)
        self.info_panel = layout.itemAt(0).widget()
        #layout = create_buttons_with_info_panel(buttons)
        #self.info_panel = layout.itemAt(0).widget()  # Get the QTextEdit panel

        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    #----------------------------------------------------------------
    
    def create_high_harmonic_tab(self):
        buttons = [
            ('Total Current', self.total_current, """
                <h3>Total Current</h3>
                <p>Plots the total current obtained from the TD-DFT calculations.</p>
                </p>
                <p></p>
            """),
            ('High Harmonic Generation', self.spectrum, """
                <h3>High-order Harmonic Generation spectra</h3>
                <p>This function computes the High Harmonic Generation (HHG) spectrum by performing a Fourier Transform on the total current data loaded from a file, converting it from the time domain to the frequency domain.</p>
                <p>The computation involves:</p>
                <ul>
                    <li>Applying a cosine-shaped filtering window to smooth the current data.</li>
                    <li>Optionally computing the time derivative of the currents:
                        <ul>
                            <li><b>With Derivative:</b> Uses the acceleration form, which emphasizes higher harmonics and relates to emitted radiation directly.</li>
                            <li><b>Without Derivative:</b> Uses the velocity form, providing a direct connection to the charge current and is less sensitive to noise.</li>
                        </ul>
                    </li>
                    <li>Calculating spectral intensities for the x, y, and combined components in the frequency domain.</li>
                </ul>
                <p>The output includes the harmonic spectrum, central laser frequency, Fourier-transformed components, and relevant time and frequency data.</p>
            """),
            ('High-order harmonic generation yield', self.hhg_yield, """
                <h3>High-order Harmonic Generation Yield</h3>
                <p>This feature calculates the harmonic yields for specified harmonic orders based on the total current data from a simulation. It performs a Fourier transform on the time-domain current data to compute the harmonic spectra for the x and y polarization components and the total spectrum.</p>
                <p><b>Method:</b></p>
                <ul>
                    <li>The user selects a <code>total_current</code> file containing time and current data.</li>
                    <li>Depending on the selected mode:
                        <ul>
                            <li><b>With Time Derivative:</b> Numerical differentiation is applied to the current data, smoothed with a Savitzky-Golay filter, to compute the instantaneous rate of change.</li>
                            <li><b>Without Time Derivative:</b> The raw current data is used directly for the calculation.</li>
                        </ul>
                    </li>
                    <li>The harmonic spectra are calculated by integrating over the Fourier transform of the current components.</li>
                    <li>The harmonic yields for the x, y, and total components are integrated over the specified harmonic frequency range.</li>
                </ul>
                <p>The results include total yield, x-component yield, and y-component yield for the selected harmonic range, providing insights into the contributions of different polarization components and harmonic orders.</p>
            """),
            ('Ellipticity of harmonics', self.ellipticity, """
                <h3>Ellipticity of Harmonics</h3>
                <p>This module calculates the ellipticity of high-order harmonics based on the provided total current file.</p>
                <p><b>Method:</b> 
                The spectrum is computed using parameters such as the central wavelength (λ₀), harmonic range (q), filtering options, 
                and time derivative preferences. The calculation involves deriving the right- and left-circularly polarized components (pc_right, pc_left) 
                from the dipole acceleration. Ellipticity (ε) is then computed as:</p>
                <p><b>ε = (|pc_right| - |pc_left|) / (|pc_right| + |pc_left|)</b></p>
                <p>Results include the normalized harmonic spectrum, frequency, and the ellipticity (ε), 
                displayed in a plot for analysis. Data extraction options are also available.</p>
            """),
            ('Phase analysing', self.phase_analysing, """
                <h3>Phase Analysis of High-Order Harmonics</h3>
                <p>This module analyzes the phase of high-order harmonics from the total current file.</p>
                <p><b>Method:</b> 
                The total current data is used to compute the spectrum. 
                The phases of these components, as well as the total phase (Dx + Dy), are calculated and plotted as a function of harmonic frequency.</p>
                <p>Phase values are provided in both radians and degrees, with the intensity of the harmonics also visualized. The results show:</p>
                <ul>
                    <li>Phase of Dx, Dy, and total phase in both radians.</li>
                    <li>Normalized harmonic intensity and time evolution of the harmonic spectrum.</li>
                </ul>
                <p>Data extraction options allow for tailored output to match user needs.</p>
            """),
        ]
        
        layout, button_widgets = create_buttons_with_info_panel(buttons)
        #layout = create_buttons_with_info_panel(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    #----------------------------------------------------------------
    
    def create_attosecond_pulse_tab(self):
        buttons = [
            ('Attosecond Pulse', self.attosecond_pulse, """
                <h3>Attosecond Pulse Generation</h3>
                <p>This module generates attosecond pulses by analyzing the harmonic spectrum using two different methods. The calculation utilizes the total current file, where the time evolution of the current components (jx, jy, jz) is processed to simulate the corresponding high-order harmonic generation (HHG) spectrum.</p>
                <p><b>Method 1:</b> The Fourier components of the current are computed in the frequency domain for both the x and y directions. These components are then used to compute the corresponding harmonic intensities (Ix and Iy), and the total intensity (I) is derived by combining both directions.</p>
                <p><b>Method 2:</b> The current components (jx and jy) are directly integrated over the time-domain, with each harmonic's contribution being weighted by its respective frequency. This method also computes the individual harmonic intensities (Ix and Iy), with the total intensity (I) being calculated in a similar manner to Method 1.</p>
                <p>In both methods, the results include:</p>
                <ul>
                    <li>Individual intensities in the x and y directions (Ix and Iy).</li>
                    <li>Total intensity (I).</li>
                    <li>Maximum values for the x, y, and total intensities (I_Max_x, I_Max_y, I_Max).</li>
                    <li>Time evolution of the harmonic spectrum normalized to the optical cycle.</li>
                </ul>
                <p>The final result is a detailed view of the attosecond pulse generated from the high-order harmonics.</p>
            """),
            ('Minimum Pulse Width', self.find_minimum_pulse_width_FMPW, """
                <h3>Minimum Pulse Width</h3>
                <p>The Minimum Pulse Width (MPW) is a crucial parameter for determining the shortest achievable duration of an attosecond pulse. It is calculated by evaluating the Full Width at Half Maximum (FWHM) of the intensity profile, considering the optimal harmonic orders within a given frequency range. The time corresponding to the maximum intensity is also derived, and the MPW is computed in attoseconds (as).</p>
                <p>For your system, the Minimum Pulse Width is calculated based on the total current data and the specified frequency range. The results provide the FWHM in attoseconds and give insights into the efficiency of the attosecond pulse generation process.</p>
                <p><strong>Optimal Harmonic Orders:</strong></p>
                <ul>
                    <li>Optimal Minimum Harmonic Order: {optimal_qstart}</li>
                    <li>Optimal Maximum Harmonic Order: {optimal_qmax}</li>
                </ul>
                <p><strong>FWHM:</strong> {min_FWHM} [as]</p>
                <p><strong>Time Corresponding to Maximum Intensity (OC):</strong> {OC}</p>
            """),
            ('Gabor Transform', self.time_frequency_gabor, """
                <h3>Gabor Transform</h3>
                <p>This method uses the Gabor transform to analyze the time-frequency distribution of the total current data. The Gabor transform is applied to the time-domain components of the current, which are first adjusted by subtracting the initial value to eliminate offsets.</p>
                <p>The transform is applied over a frequency range defined by the specified harmonic orders <i>qstart</i> and <i>qend</i>, with the corresponding frequency range calculated based on the central frequency <i>lambda0_nm</i>.</p>
                <p>The Gabor window function, with a time-dependent Gaussian envelope, is used to localize the signal in both time and frequency. The result is the log-transformed amplitude spectrum, providing insights into the temporal and spectral features of the total current signal.</p>
                <p>The method also converts the results to appropriate units, with the time width of the Gaussian window controlled by the <i>g_factor</i> parameter, and the output is provided in both the logarithmic and absolute forms for different components (axial and transverse).</p>
            """)
        ]
    
        layout, button_widgets = create_buttons_with_info_panel(buttons)
    
        # store references to the button widgets 
        self.mpw_button = button_widgets[1]  #>> Index 1 == Minimum Pulse Width button
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    #----------------------------------------------------------------

    def create_driving_field_tab(self):
        buttons = [
            ('Electric Field', self.electric_field, """
                <h3>Electric Field</h3>
                <p>This method allows the visualization of the electric field in two modes:</p>
                <ul>
                    <li><strong>Single Pulse Mode:</strong> Plots the electric field components (Ex, Ey, Ez) for a single laser pulse.</li>
                    <li><strong>Dual Pulse Mode:</strong> Plots the combined electric field for two pulses, often used in methods like polarization gating. In this case, the data file contains two separate pulses (Ex1, Ey1, Ez1 for the first pulse and Ex2, Ey2, Ez2 for the second pulse).</li>
                </ul>
                <p>Choose the appropriate mode depending on whether the laser field consists of a single pulse or two pulses.</p>
            """),
            ('Vector Potential', self.vector_potential, """
                <h3>Vector Potential</h3>
                <p>Method:</p>
                <p>The most common approaches to representing the vector potential include:</p>
                <ul>
                    <li><b>Coulomb Gauge (A₀ = 0):</b> This gauge is often used in electrostatics, where the vector potential is defined such that the magnetic field is given by <i>B = ∇ x A</i>, and the electric field is described by the scalar potential alone. This method simplifies the description of the electric field in static conditions.</li>
                    <li><b>Lorenz Gauge:</b> In this approach, the vector potential is defined such that the divergence of the vector potential is related to the scalar potential, <i>∇ · A + (1/c²) ∂Φ/∂t = 0</i>, allowing a more convenient form for wave propagation in vacuum and in light-matter interaction scenarios.</li>
                    <li><b>Time-Dependent Field Approach:</b> For time-dependent simulations, the vector potential is often used to describe the interaction of solid state system with electric fields, such as in the case of laser-matter interactions. The electric field is typically derived as <i>E = -∂A/∂t</i>, where the vector potential varies with time.</li>
                </ul>
                <p>Depending on the physical system and the specific requirements of the simulation, different representations of the vector potential are used.</p>
            """),

            ('Configure Polarization Gating Field', self.pg_analyzing, """
                <h3>Configure Polarization Gating Field</h3>
                <p>This section allows you to explore and configure the features and outputs related to the polarization gating field for generating attosecond pulses. Below are the available functionalities:</p>
                <h4>Features:</h4>
                <ul>
                    <li><strong>Driving Laser Parameters:</strong> Configure two independent laser pulses by specifying their wavelengths (in nm), intensities (in TW/cm²), and the number of optical cycles for each pulse.</li>
                    <li><strong>Ellipticity and Polarization:</strong> Define the ellipticity and rotation angles for each pulse to control their polarization state.</li>
                    <li><strong>Carrier Envelope Phase (CEP):</strong> Adjust the CEP of each pulse to fine-tune the phase relationship between the fields.</li>
                    <li><strong>Delay Configuration:</strong> Introduce a temporal delay between the pulses to modulate their interaction dynamically.</li>
                    <li><strong>Envelope Shape:</strong> Choose from envelope functions such as sine-square or Gaussian to customize the field modulation.</li>
                </ul>
                <h4>Outputs:</h4>
                <ul>
                    <li><strong>Time-Dependent Field Components:</strong> Visualize the Ax and Ay vector potential components over time, including their phase and amplitude characteristics.</li>
                    <li><strong>Envelope Functions:</strong> Plot the envelope functions of both laser pulses, with key metrics like full-width half-maximum (FWHM) displayed.</li>
                    <li><strong>Data Extraction:</strong> Export detailed datasets, including time, vector potentials, and envelope data, for further analysis or integration into other simulations.</li>
                </ul>
                <p>These features and outputs enable comprehensive configuration and visualization of the polarization gating field for optimal attosecond pulse generation.</p>
            """),
            ('Determine Polarization Gate Width', self.gate_width, """
                <h3>Determine Polarization Gate Width</h3>
                <p><b>Method:</b></p>
                <p>This function calculates the time-dependent ellipticity (ε(t)) of two combined laser fields to evaluate the width of the polarization gate. 
                The method involves the following steps:</p>
                <ul>
                    <li>Computing frequencies (ω) and periods (T) of the two laser fields based on their respective wavelengths (λ).</li>
                    <li>Calculating field amplitudes and defining the envelope functions of the fields. Two types of envelopes are supported:
                        <ul>
                            <li><b>sin²:</b> For smoothly varying pulses with gradual turn-on and turn-off behavior.</li>
                            <li><b>Gaussian:</b> For pulses with a bell-shaped intensity profile.</li>
                        </ul>
                    </li>
                    <li>Determining the time-dependent ellipticity as the relative difference between the envelope functions of the two fields.</li>
                </ul>
                <p>The output includes:
                    <ul>
                        <li><b>Optical cycles:</b> Time expressed in optical cycles of the first laser field.</li>
                        <li><b>ε(t):</b> The calculated ellipticity as a function of time.</li>
                    </ul>
                </p>
                <p>Results are passed to the visualization function for plotting, and detailed progress is displayed in the software panel.</p>
            """)
        ]
        
        #layout = create_buttons_with_info_panel(buttons)
        layout, button_widgets = create_buttons_with_info_panel(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    #----------------------------------------------------------------
    
    def electron_dynamics_tab(self):
        buttons = [
            ('Excited Electrons Over Time', self.nex_vs_time, """
                <h3>Excited electrons over time</h3>
                <p><strong>Method:</strong></p>
                <p>
                    This feature allows users to plot the temporal evolution of excited electrons based on the input simulation data.
                    The process involves:
                    <ul>
                        <li><strong>Reading Data:</strong> The software reads the user-selected file containing data. The file must be correctly formatted with time and values in the second and third columns, respectively.</li>
                        <li><strong>Plotting:</strong> The Nex data is plotted against time (scaled to optical cycles), with the maximum value indicated.</li>
                    </ul>
                </p>
                <p><strong>Usage:</strong></p>
                <p>
                    <ol>
                        <li>Select the 'N_ex' file using the file dialog interface. Ensure the file name contains "n_ex" to avoid errors.</li>
                        <li>Enter the driving field wavelength in nanometers and intensity in TW/cm² through the input dialog. Validation ensures these values are positive.</li>
                        <li>The data is processed and displayed, with key metrics like highlighted on the plot.</li>
                    </ol>
                </p>
                <p><strong>Output:</strong></p>
                <p>
                    The graph displays:
                    <ul>
                        <li>N_ex vs. time in optical cycles.</li>
                        <li>The maximum N_{ex} value, marked and annotated on the plot.</li>
                        <li>Information about the driving wavelength and intensity as part of the title.</li>
                    </ul>
                </p>
            """),
            ('Excited electrons across the Brillouin zone', self.nex_distribution, """
                <h3>Excited Electrons in K-Space</h3>
                <p>
                    This module allows you to visualize the distribution of excited electrons 
                    across the Brillouin zone (BZ) using provided input data files.
                </p>
                """),
                
            ('Current across the Brillouin zone', self.curr_distribution, """
                <h3>Current in K-Space</h3>
                <p>
                    This module allows you to visualize the current 
                    across the Brillouin zone (BZ) using provided input data files.
                </p>
                """),
            ('Nex and Current animations', self.current_nex_animation, """ 
                <h3>Animation: Excited Electrons & Current in K-Space</h3>
                <p>
                    This module visualizes the **time evolution** of both the number of excited electrons (N<sub>ex</sub>) and the **current** in k-space, animated according to the applied excitation laser field.
                </p>
    
                <p><strong>Inputs Required:</strong></p>
                <ul>
                    <li><strong>'laser' file:</strong> Absolute path to the laser field file.</li>
                    <li><strong>'output_iter' folder:</strong> Folder containing time-resolved data snapshots.</li>
                </ul>

                <p><strong>Settings:</strong></p>
                <ul>
                    <li><strong>Grid Resolution:</strong> Controls the interpolation density within each data file.</li>
                    <li><strong>Interpolated Frames:</strong> Number of frames interpolated between discrete simulation times (affects animation smoothness).</li>
                </ul>

                <p>
                    <strong>Note:</strong> Higher values for Grid Resolution and Interpolated Frames increase memory usage. If the <em>Save</em> option is enabled, animation frames will also be written to disk, further increasing load. Use caution to avoid crashes.
                </p>

                <p>
                    Multithreading is used to keep the user interface responsive during animation generation and saving. However, due to the processing load, some systems may still experience lag or instability.
                </p>

                <p><strong>Output:</strong></p>
                <ul>
                    <li>Animation preview displayed in the interface (if enabled).</li>
                    <li>Saved animation to file (if the 'Save' checkbox is selected).</li>
                    <li>Progress updates shown in the Python console.</li>
                </ul>

                <p style="color: darkred;">
                    If you experience crashes or unexpected behavior, please contact the developer.
                </p>
            """)
        ]
        layout, button_widgets = create_buttons_with_info_panel(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    #----------------------------------------------------------------
    
    def create_tool_box_tab(self):
        buttons = [
            ('Fourier Transform (Function)', self.ft_func, """
                <h3>Fourier Transform (Function)</h3>
                <p>This feature allows users to calculate the Fourier Transform of a user-defined function.</p>
                <p>The Fourier Transform converts a function from the time domain into its representation in the frequency domain. 
                Users can input a mathematical function of time and a parameter k, specify a value for k, and define a frequency range for the calculation.</p>
                <p>Steps to use this feature:</p>
                <ol>
                    <li>Enter the function as a mathematical expression involving time (t) and k.</li>
                    <li>Provide a numerical value for k.</li>
                    <li>Specify the frequency range as two comma-separated values.</li>
                    <li>Click "Submit" to compute and display the magnitude of the Fourier Transform over the specified frequency range.</li>
                </ol>
                <p>Note: Supported mathematical functions include exponential, trigonometric, logarithmic, and more. Please use Python syntax and functions (e.g., <code>np.sin</code>, <code>np.exp</code>).</p>
            """),
            ('Fourier Transform (Data)', self.ft_data, """
                <h3>Fourier Transform (Data)</h3>
                <p>This feature allows users to perform a Fourier Transform on time-domain data loaded from a file. The result is the frequency-domain representation of the data.</p>
                <p>Steps to use this feature:</p>
                <ol>
                    <li>Click the "Select File" button to upload your time-domain data file. Ensure the file contains two columns: time values and corresponding function values.</li>
                    <li>Enter the desired frequency range as two comma-separated values (e.g., -40,40).</li>
                    <li>Click "Submit" to compute and plot the magnitude of the Fourier Transform over the specified frequency range.</li>
                </ol>
                <p>Additional Details:</p>
                <ul>
                    <li>The tool interpolates the data using cubic interpolation for improved accuracy.</li>
                    <li>The Fourier Transform is computed using numerical integration over the specified frequency range.</li>
                    <li>The resulting plot displays the magnitude of the Fourier Transform as a function of frequency.</li>
                </ul>
                <p>Note: Ensure that the input file format matches the expected structure. The first column should represent time, and the second column should represent the corresponding function values.</p>
            """),
            ('Inverse Fourier Transform (Function)', self.ift_func, """
                <h3>Inverse Fourier Transform (Function)</h3>
                <p>This feature computes the inverse Fourier Transform of a user-defined function, reconstructing the original time-domain function from its frequency-domain representation.</p>
                <p>Steps to use this feature:</p>
                <ol>
                    <li>Enter a function of frequency (f) and parameter (k) in the provided input box. The function must use valid mathematical expressions (e.g., <code>np.exp(-k * f**2) * np.sin(k*f)</code>).</li>
                    <li>Specify the value of the parameter <code>k</code>.</li>
                    <li>Define the time range as two comma-separated values (e.g., <code>-10,10</code>).</li>
                    <li>Click "Submit" to compute and plot the magnitude of the reconstructed time-domain function over the specified range.</li>
                </ol>
                <p>Additional Details:</p>
                <ul>
                    <li>The tool uses numerical integration to compute the inverse Fourier Transform.</li>
                    <li>Ensure that the entered function and inputs are valid. Allowed mathematical operations include exponentials, trigonometric functions, logarithms, and square roots.</li>
                    <li>The resulting plot displays the magnitude of the reconstructed function over time.</li>
                </ul>
                <p>Note: Use valid Python syntax for the function. For example, use <code>np.sin</code> instead of <code>sin</code> for sine functions, and ensure the function operates on the variables <code>f</code> and <code>k</code>.</p>
            """),
            ('Inverse Fourier Transform (Data)', self.ift_data, """
                <h3>Inverse Fourier Transform (Data)</h3>
                <p>This feature analyzes frequency-domain data from a user-selected file and converts it back to the time domain using the inverse Fourier Transform.</p>
                <p>Steps to use this feature:</p>
                <ol>
                    <li>Click on the input field and specify the time range as two comma-separated values (e.g., <code>-1,1</code>).</li>
                    <li>Select the data file containing frequency-domain information when prompted. The file should include at least two columns: frequency values and corresponding data points in the frequency domain.</li>
                    <li>Click "Submit" to compute and visualize the inverse Fourier Transform, which reconstructs the original time-domain signal.</li>
                </ol>
                <p>Additional Details:</p>
                <ul>
                    <li>The tool interpolates the data using cubic interpolation to ensure smooth processing.</li>
                    <li>Numerical integration is performed over the specified frequency range to compute the inverse transform accurately.</li>
                    <li>The resulting plot displays the magnitude of the reconstructed signal over the specified time range.</li>
                </ul>
                <p>Notes:</p>
                <ul>
                    <li>Ensure the data file format is correct and includes valid numerical data for frequency and corresponding values.</li>
                    <li>Large frequency or time ranges may increase computation time.</li>
                </ul>
            """),
            ('Unit Conversion', self.unit_conversion, """
                <h3>Unit Conversion</h3>
                <p>This feature allows users to convert physical quantities between SI units and atomic units for various properties such as intensity, time, energy, electric field, and length.</p>
    
                <p><strong>How to Use:</strong></p>
                <ol>
                    <li>Enter the value to be converted in the input field.</li>
                    <li>Select the type of quantity to convert (e.g., intensity, time, energy, etc.) from the dropdown menu.</li>
                    <li>Choose the direction of conversion:
                        <ul>
                            <li><strong>SI to Atomic:</strong> Converts values from SI units to atomic units.</li>
                            <li><strong>Atomic to SI:</strong> Converts values from atomic units to SI units.</li>
                        </ul>
                    </li>
                    <li>Click the "Convert" button to perform the conversion.</li>
                </ol>
    
                <p><strong>Supported Conversions:</strong></p>
                <ul>
                    <li><strong>Intensity:</strong> Converts between W/cm² and atomic units of intensity.</li>
                    <li><strong>Time:</strong> Converts between seconds (s) and atomic units of time.</li>
                    <li><strong>Energy:</strong> Converts between electron volts (eV) and atomic units of energy.</li>
                    <li><strong>Electric Field:</strong> Converts between volts per meter (V/m) and atomic units of electric field strength.</li>
                    <li><strong>Length:</strong> Converts between meters (m) and atomic units of length.</li>
                </ul>
    
                <p><strong>Additional Information:</strong></p>
                <ul>
                    <li>The conversion factors are based on fundamental physical constants, such as the speed of light, Planck's constant, and the fine-structure constant.</li>
                    <li>Ensure the entered value is a valid numerical input for accurate results.</li>
                    <li>Conversion results are displayed with a precision of up to four decimal places.</li>
                </ul>
            """)
        ]
    
        #layout = create_buttons_with_info_panel(buttons)
        layout, button_widgets = create_buttons_with_info_panel(buttons)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    #----------------------------------------------------------------
    def create_help_tab(self):
        # Main widget and layout
        widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
    
        # Left panel - Help categories
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
    
        # categories header
        categories_header = QLabel("Attoscience Studio")
        categories_header.setFont(QFont('Arial', 14, QFont.Bold))
        categories_header.setStyleSheet("color: #f0f0f0; padding: 10px; border-bottom: 2px solid #444444; margin-bottom: 10px;")
        categories_header.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(categories_header)
    
        # help content display area
        self.help_content = QTextEdit()
        self.help_content.setReadOnly(True)
        self.help_content.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #f0f0f0;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 15px;
                font-size: 11pt;
                line-height: 1.4;
            }
        """)
    
        # Help category buttons
        help_categories = [
            ("Getting Started", self.show_getting_started_help),
            ("Ground State Analysis", self.show_ground_state_help),
            ("Driving Pulse Setup", self.show_driving_pulse_help),
            ("High Harmonic Generation", self.show_hhg_help),
            ("Attosecond Pulse Analysis", self.show_attosecond_help),
            ("Electron Dynamics", self.show_dynamics_help),
            ("Tool Box Utilities", self.show_toolbox_help),
            ("File Formats", self.show_file_formats_help),
            ("Troubleshooting", self.show_troubleshooting_help),
            ("About and Contact", self.show_about_help)
        ]
    
        # category buttons
        for category_name, callback in help_categories:
            btn = QPushButton(category_name)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #f0f0f0;
                    border: 1px solid #555555;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: left;
                    font-size: 11pt;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    border-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #2a2a2a;
                }
            """)
            btn.clicked.connect(callback)
            left_layout.addWidget(btn)
    
        left_layout.addStretch()
        left_widget.setMaximumWidth(250)
        left_widget.setMinimumWidth(200)
    
        # widgets to main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.help_content, stretch=3)
    
        widget.setLayout(main_layout)
    
        # show getting started by default
        self.show_getting_started_help()
    
        return widget

    # help content methods
    def show_getting_started_help(self): 
        content = f"""
        <h2>{Symbols.ATOM} Getting Started with Attoscience Studio</h2>
        
        <h3>Welcome to Attoscience Studio 0.9.0-Beta!</h3>
        <p>This software is designed for analyzing attosecond physics, high harmonic generation, and ultrafast electron dynamics in solid-state systems.</p>
    
        <h3>Quick Start Guide:</h3>
        <ol>
            <li><b>Ground State Analysis:</b> Start by analyzing your material's electronic structure using the Ground State tab</li>
            <li><b>Setup Driving Field:</b> Configure your laser parameters in the Driving Pulse tab</li>
            <li><b>Analyze HHG:</b> Use the High Harmonic tab to study harmonic generation spectra</li>
            <li><b>Generate Attosecond Pulses:</b> Create and analyze attosecond pulses in the dedicated tab</li>
            <li><b>Study Dynamics:</b> Examine electron dynamics across the Brillouin zone</li>
        </ol>
    
        <h3>Key Features:</h3>
        <ul>
            <li> <b>Ground State Properties:</b> Crystal structure, band structure, density of states</li>
            <li> <b>Field Configuration:</b> Electric field, vector potential, polarization gating</li>
            <li> <b>Harmonic Analysis:</b> HHG spectra, yield calculations, phase analysis</li>
            <li> <b>Attosecond Pulses:</b> Pulse generation, minimum pulse width, Gabor transform</li>
            <li> <b>Electron Dynamics:</b> Time evolution, k-space distributions, animations</li>
            <li> <b>Utilities:</b> Fourier transforms, unit conversions, data analysis tools</li>
        </ul>
    
        <h3>IPython Console:</h3>
        <p>The embedded IPython terminal at the bottom allows for scripting and logging to the user.</p>
        """
        self.help_content.setHtml(content)

    def show_ground_state_help(self):
        content = f"""
        <h2>{Symbols.ATOM} Ground State Analysis</h2>
    
        <p>The Ground State tab provides tools for analyzing the electronic structure of materials.</p>
    
        <h3>Crystal Structure</h3>
        <p>Visualize the atomic arrangement in your crystal lattice. This feature is currently experimental and under development.</p>
    
        <h3>Band Structure</h3>
        <p><b>Purpose:</b> Analyze the electronic band structure of your material</p>
        <p><b>Input File:</b> 'bandstructure' file containing k-points and energy eigenvalues</p>
        <p><b>Key Parameters:</b></p>
        <ul>
            <li><b>Fermi Energy:</b> Reference energy level (in Hartree units)</li>
            <li><b>Number of Bands:</b> How many energy bands to include in the plot</li>
            <li><b>Plot Customization:</b> Axis limits, colors, line thickness</li>
        </ul>
        <p><b>Output:</b> Band gap calculation (direct/indirect) and energy dispersion plot</p>
    
        <h3>Density of States (DOS)</h3>
        <p><b>Purpose:</b> Visualize the density of electronic states as a function of energy</p>
        <p><b>Input File:</b> 'total-dos.dat' containing energy and DOS values</p>
        <p><b>Features:</b></p>
        <ul>
            <li>Customizable axis limits and labels</li>
            <li>Adjustable line properties and colors</li>
            <li>Energy values in atomic units</li>
        </ul>
    
        <h3>Electron Density</h3>
        <p><b>Purpose:</b> Analyze the spatial distribution of electrons in the material</p>
        <p>This shows the probability density of finding electrons in different regions of space.</p>
    
        <h3>Tips:</h3>
        <ul>
            <li>Ensure your input files are properly formatted</li>
            <li>Check the band gap value - it affects HHG cutoff energies</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_driving_pulse_help(self):
        content = f"""
        <h2>{Symbols.ATOM} Driving Pulse Configuration</h2>
    
        <p>Configure the laser fields that drive the nonlinear optical processes in your material.</p>
    
        <h3>Electric Field</h3>
        <p><b>Single Pulse Mode:</b> Analyze Ex, Ey, Ez components of a single laser pulse</p>
        <p><b>Dual Pulse Mode:</b> For polarization gating setups with two pulses</p>
        <p><b>Applications:</b> Understanding field strength, polarization, and temporal profile</p>
    
        <h3>Vector Potential</h3>
        <p><b>Gauge Choices:</b></p>
        <ul>
            <li><b>Time-Dependent:</b> E = -∂A/∂t</li>
        </ul>
    
        <h3>Configure Polarization Gating Field</h3>
        <p><b>Purpose:</b> Setup PG fields for isolated attosecond pulse generation</p>
        <p><b>Parameters:</b></p>
        <ul>
            <li><b>Laser Properties:</b> Wavelength (nm), intensity (TW/cm²), optical cycles</li>
            <li><b>Polarization:</b> Ellipticity and rotation angles for each pulse</li>
            <li><b>Phase Control:</b> Carrier envelope phase (CEP) adjustment</li>
            <li><b>Timing:</b> Temporal delay between pulses</li>
            <li><b>Envelope:</b> Sine-square or Gaussian pulse shapes</li>
        </ul>
    
        <h3>Determine Polarization Gate Width</h3>
        <p><b>Method:</b> Calculate time-dependent ellipticity ε(t) = (I₁ - I₂)/(I₁ + I₂)</p>
        <p><b>Applications:</b></p>
        <ul>
            <li>Optimize gate width for single attosecond pulse generation</li>
            <li>Design polarization gating schemes</li>
        </ul>
    
        <h3>Best Practices:</h3>
        <ul>
            <li>For polarization gating:</li>
            <li>Start with fundamental wavelengths (2000 nm, 2000 nm)</li>
            <li> Intensity 1 = Intensity 2 = 0.5 (TW/cm²), delay = 1.75</li>
            <li>Monitor parameters to avoid numerical instabilities</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_hhg_help(self):
        content = f"""
        <h2>{Symbols.ATOM} High Harmonic Generation Analysis</h2>
    
        <p>Analyze the high harmonic spectra generated by intense laser-matter interaction.</p>
    
        <h3>Total Current</h3>
        <p><b>Purpose:</b> Visualize the time-dependent current from TD-DFT calculations</p>
        <p><b>Input:</b> 'total_current' file with time and current components (jx, jy, jz)</p>
        <p><b>Analysis:</b> Shows the time evolution of induced currents during laser interaction</p>
    
        <h3>High Harmonic Generation Spectrum</h3>
        <p><b>Method:</b> Fourier transform of the total current: I(ω) ∝ |FT[j(t)]|²</p>
        <p><b>Options:</b></p>
        <ul>
            <li><b>With Derivative:</b> Acceleration form (∂j/∂t) - emphasizes higher harmonics</li>
            <li><b>Without Derivative:</b> Velocity form - direct current, less noise-sensitive</li>
        </ul>
        <p><b>Features:</b></p>
        <ul>
            <li>Window functions for smooth spectra</li>
        </ul>
    
        <h3>HHG Yield Calculation</h3>
        <p><b>Purpose:</b> Quantify harmonic efficiency for specific orders</p>
        <p><b>Method:</b> Integrate spectral intensity over harmonic frequency ranges</p>
        <p><b>Output:</b></p>
        <ul>
            <li>Total yield (x + y components)</li>
            <li>Individual x and y yields</li>
        </ul>
    
        <h3>Ellipticity of Harmonics</h3>
        <p><b>Formula:</b> ε = (|pc_right| - |pc_left|) / (|pc_right| + |pc_left|)</p>
    
        <h3>Phase Analysis</h3>
        <p><b>Purpose:</b> Extract phase information from harmonic components</p>
        <p><b>Output:</b></p>
        <ul>
            <li>Phase of Dx, Dy components (radians and degrees)</li>
            <li>Total phase (Dx + Dy)</li>
            <li>Phase-resolved harmonic evolution</li>
        </ul>
    
        <h3>Interpretation Tips:</h3>
        <ul>
            <li><b>E_Cutoff Law:</b> Maximum emitted photon energy</li>
            <li><b>Plateau Structure:</b> Look for flat spectral regions</li>
            <li><b>Phase Matching:</b> Consider crystal thickness effects</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_attosecond_help(self):
        content = f"""
        <h2>{Symbols.ATOM} Attosecond Pulse Generation & Analysis</h2>
    
        <p>Generate and characterize attosecond pulses from high harmonic spectra.</p>
    
        <h3>Attosecond Pulse Generation</h3>
        <p><b>Two Methods Available:</b></p>
        <p><b>Method 1:</b></p>
        <ul>
            <li>In this method, the Fourier transform is taken directly from the total current components.</li>
        </ul>
    
        <p><b>Method 2:</b></p>
        <ul>
            <li>In this method, the Fourier transform of the derived components is taken.</li>
            <li>Multiplies the Fourier components by their corresponding angular frequency.</li>
        </ul>
            
        <h3>Minimum Pulse Width (FWHM)</h3>
        <p><b>Purpose:</b> Determine the shortest achievable attosecond pulse duration</p>
        <p><b>Method:</b></p>
        <ul>
            <li>Calculate Full Width at Half Maximum (FWHM) of intensity profile</li>
            <li>Optimize harmonic order range (qstart to qmax)</li>
            <li>Find time corresponding to maximum intensity</li>
        </ul>
        <p><b>Output:</b></p>
        <ul>
            <li>Minimum FWHM in attoseconds</li>
            <li>Optimal harmonic range</li>
        </ul>
    
        <h3>Gabor Transform (Time-Frequency Analysis)</h3>
        <p><b>Purpose:</b> Analyze temporal and spectral localization simultaneously</p>
        <p><b>Method:</b></p>
        <ul>
            <li>Apply Gabor window: G(t,ω) = ∫ j(τ) × g(τ-t) × e^(-iωτ) dτ</li>
            <li>Gaussian envelope: g(t) = exp(-t²/2σ²)</li>
            <li>Window width controlled by g_factor parameter</li>
        </ul>
        <p><b>Applications:</b></p>
        <ul>
            <li>Identify temporal gating mechanisms</li>
            <li>Optimize pulse compression strategies</li>
            <li>Study spectral-temporal correlations</li>
        </ul>
    
        <h3>Physical Insights:</h3>
        <ul>
            <li><b>Transform Limit:</b> Δt × Δω ≥ 1/2 (uncertainty principle)</li>
            <li><b>Bandwidth:</b> Wider harmonic spectrum → shorter pulses</li>
            <li><b>Coherence:</b> Phase-locked harmonics essential for pulse formation</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_dynamics_help(self):
        content = f"""
        <h2>{Symbols.ATOM} Electron Dynamics Analysis</h2>
    
        <p>Study the time evolution of excited electrons and currents in momentum space.</p>
    
        <h3>Excited Electrons Over Time</h3>
        <p><b>Purpose:</b> Track the temporal evolution of excited electron population</p>
        <p><b>Input File:</b> N_ex file containing time and excited electron count</p>
        <p><b>Parameters:</b></p>
        <ul>
            <li><b>Wavelength:</b> Driving field wavelength (nm)</li>
            <li><b>Intensity:</b> Laser intensity (TW/cm²)</li>
        </ul>
        <p><b>Analysis:</b></p>
        <ul>
            <li>Time axis converted to optical cycles</li>
        </ul>
    
        <h3>Excited Electrons Across Brillouin Zone</h3>
        <p><b>Purpose:</b> Visualize k-space distribution of excited electrons</p>
        <p><b>Applications:</b></p>
        <ul>
            <li>Identify preferential excitation directions</li>
            <li>Analyze band structure effects on excitation</li>
        </ul>
    
        <h3>Current Across Brillouin Zone</h3>
        <p><b>Purpose:</b> Map current density in momentum space</p>
        <p><b>Physical Meaning:</b></p>
        <ul>
            <li>Shows directional charge flow</li>
        </ul>
        
        <h3>N_ex and Current Animations</h3>
        <p><b>Purpose:</b> Time-resolved visualization of electron dynamics</p>
        <p><b>Required Inputs:</b></p>
        <ul>
            <li><b>Iteration Directory:</b> 'laser; dtat file (td.general)</li>
            <li><b>Iteration Directory:</b> Iteration directory</li>
        </ul>
    
        <p><b>Settings:</b></p>
        <ul>
            <li><b>Grid Resolution:</b> Interpolation density (higher = smoother, more memory)</li>
            <li><b>Interpolated Frames:</b> Animation smoothness (more frames = smoother motion)</li>
            <li><b>Save Option:</b> Export animation frames to disk</li>
        </ul>
    
        <p><b>{Symbols.WARNING} Performance Notes:</b></p>
        <ul>
            <li>High resolution settings increase memory usage significantly</li>
            <li>Saving animations requires additional disk space</li>
            <li>Multithreading keeps UI responsive but system may still lag</li>
            <li>Contact developer if crashes occur</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_toolbox_help(self):
        content = f"""
        <h2>{Symbols.GEAR} Tool Box Utilities</h2>
    
        <p>Mathematical and analytical tools for data processing.</p>
    
        <h3>Fourier Transform (Function)</h3>
        <p><b>Purpose:</b> Calculate FT of user-defined mathematical functions</p>
        <p><b>Input:</b> Function f(t,k) with parameter k</p>
        <p><b>Syntax:</b> Use Python/NumPy syntax (e.g., np.sin, np.exp)</p>
        <p><b>Example:</b> <code>np.exp(-k*t**2) * np.sin(t)</code></p>
    
        <h3>Fourier Transform (Data)</h3>
        <p><b>Purpose:</b> Transform time-domain data files to frequency domain</p>
        <p><b>Input Format:</b> Two columns [time, amplitude]</p>
        <p><b>Features:</b></p>
        <ul>
            <li>Cubic interpolation for accuracy</li>
            <li>User-defined frequency range</li>
            <li>Numerical integration method</li>
        </ul>
    
        <h3>Inverse Fourier Transform (Function)</h3>
        <p><b>Purpose:</b> Reconstruct time-domain from frequency-domain functions</p>
        <p><b>Formula:</b> f(t) = ∫ F(ω) e^(iωt) dω</p>
        <p><b>Applications:</b></p>
        <ul>
            <li>Signal reconstruction</li>
            <li>Spectral synthesis</li>
        </ul>
    
        <h3>Inverse Fourier Transform (Data)</h3>
        <p><b>Purpose:</b> Convert frequency-domain data back to time domain</p>
        <p><b>Input:</b> Two columns [frequency, spectral_amplitude]</p>
        <p><b>Method:</b> Numerical inverse FT with cubic interpolation</p>
    
        <h3>Unit Conversion</h3>
        <p><b>Supported Conversions:</b></p>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #3c3c3c;">
                <th>Quantity</th>
                <th>SI Unit</th>
                <th>Atomic Unit</th>
            </tr>
            <tr>
                <td>Intensity</td>
                <td>W/cm²</td>
                <td>a.u. intensity</td>
            </tr>
            <tr>
                <td>Time</td>
                <td>seconds (s)</td>
                <td>a.u. time</td>
            </tr>
            <tr>
                <td>Energy</td>
                <td>electron volts (eV)</td>
                <td>Hartree</td>
            </tr>
            <tr>
                <td>Electric Field</td>
                <td>V/m</td>
                <td>a.u. field</td>
            </tr>
            <tr>
                <td>Length</td>
                <td>meters (m)</td>
                <td>Bohr radius</td>
            </tr>
        </table>
    
        <h3>Usage Tips:</h3>
        <ul>
            <li><b>Function Syntax:</b> Always use numpy functions (np.sin, not sin)</li>
            <li><b>File Formats:</b> Ensure clean, numerical data with proper column structure</li>
            <li><b>Frequency Ranges:</b> Choose ranges appropriate for your data sampling</li>
            <li><b>Precision:</b> Results displayed with up to 4 decimal places</li>
        </ul>
    
        """
        self.help_content.setHtml(content)

    def show_file_formats_help(self):
        content = f"""
        <h2>{Symbols.FOLDER} File Formats & Data Structure</h2>
    
        <h3>Ground State Files</h3>
        <p><b>bandstructure:</b></p>
        <ul>
            <li>Format: [coord. kx ky kz (red. coord.), bands:    -- [H]]</li>
            <li>Units: k in reciprocal lattice units, energies in Hartree</li>
            <li>Structure: [kx, ky, kz, E1, E2, E3, ...]</li>
        </ul>
    
        <p><b>total-dos.dat:</b></p>
        <ul>
            <li>Format: Two columns [Energy, DOS]</li>
            <li>Units: Energy in atomic units, DOS in states/energy</li>
            <li>Comment lines starting with # are ignored</li>
        </ul>
    
        <h3>Time-Dependent Files</h3>
        <p><b>total_current:</b></p>
        <ul>
            <li>Format: [Iter, t, I(1), I(2), I(3)]</li>
            <li>Units: Time in atomic units, current in a.u.</li>
            <li>Required for all HHG and attosecond analysis</li>
        </ul>

        <p><b>laser (electric field):</b></p>
        <ul>
            <li>Single pulse format: [Iter, t, E(1), E(2), E(3)]</li>
            <li>Dual pulse format: [Iter, t, E(1), E(2), E(3), E(1), E(2), E(3)]</li>
            <li>Units: Time in a.u., fields in a.u.</li>
        </ul>
    
        <p><b>vector_potential:</b></p>
        <ul>
            <li>Single pulse format: [Iter, t, A(1), A(2), A(3)]</li>
            <li>Dual pulse format: [Iter, t, A(1), A(2), A(3), A(1), A(2), A(3)]</li>
            <li>Units: Time and vector potential in atomic units</li>
        </ul>
    
        <h3>Dynamics Files</h3>
        <p><b>N_ex (excited electrons):</b></p>
        <ul>
            <li>Format: [Iter, t, iter t Nex(t)]</li>
            <li>Must contain "n_ex" in filename for automatic detection</li>
            <li>Units: Time in a.u.</li>
        </ul>
    
        <p><b>output_iter folder:</b></p>
        <ul>
            <li>Contains time-resolved snapshots of k-space data</li>
            <li>Each file represents one time step</li>
            <li>Used for animation generation</li>
        </ul>
    
        <h3>Data Quality Guidelines</h3>
        <ul>
            <li><b>Consistency:</b> Use consistent units throughout</li>
            <li><b>Sampling:</b> Ensure adequate time/frequency resolution</li>
            <li><b>Format:</b> Plain text, space or tab-separated values</li>
            <li><b>Comments:</b> Use # for comment lines</li>
            <li><b>Numerical:</b> Avoid NaN, Inf, or non-numeric entries</li>
        </ul>
    
        <h3>Common Issues & Solutions</h3>
        <p><b>File Not Found:</b></p>
        <ul>
            <li>Check file path and name spelling</li>
            <li>Ensure file is in accessible directory</li>
            <li>Verify file permissions</li>
        </ul>
    
        <p><b>Format Errors:</b></p>
        <ul>
            <li>Remove extra headers or comment lines</li>
            <li>Check for consistent column numbers</li>
            <li>Verify numerical format (no scientific notation issues)</li>
        </ul>
    
        <p><b>Memory Issues:</b></p>
        <ul>
            <li>Large files may cause memory problems</li>
            <li>Consider data subsampling for initial analysis</li>
            <li>Close unused plots and variables</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_troubleshooting_help(self):
        content = f"""
        <h2>{Symbols.WARNING} Troubleshooting Guide</h2>
    
        <h3>Common Error Messages</h3>
    
        <h4>"File format not recognized"</h4>
        <p><b>Causes:</b></p>
        <ul>
            <li>Incorrect file structure or column count</li>
            <li>Non-numeric data in numeric columns</li>
            <li>Inconsistent delimiter usage</li>
        </ul>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Check file format matches expected structure</li>
            <li>Remove header lines or comments within data</li>
            <li>Ensure consistent use of spaces or tabs as delimiters</li>
            <li>Verify all data entries are numeric</li>
        </ul>

        <h4>"Memory Error" or Application Crashes</h4>
        <p><b>Causes:</b></p>
        <ul>
            <li>Large dataset processing (especially animations)</li>
            <li>High resolution settings in visualization</li>
            <li>Multiple large plots open simultaneously</li>
        </ul>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Reduce grid resolution and interpolated frames</li>
            <li>Close unused plots and clear variables</li>
            <li>Process data in smaller chunks</li>
            <li>Restart application if memory usage is high</li>
        </ul>

        <h4>"Invalid function syntax" (Tool Box)</h4>
        <p><b>Common Issues:</b></p>
        <ul>
            <li>Using 'sin' instead of 'np.sin'</li>
            <li>Undefined variables or typos</li>
            <li>Missing parentheses or operators</li>
        </ul>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Always use numpy syntax: np.sin, np.exp, np.cos</li>
            <li>Check variable names match function parameters</li>
            <li>Test simple functions first</li>
        </ul>

        <h3>Performance Issues</h3>
    
        <h4>Slow Animation Generation</h4>
        <p><b>Optimization:</b></p>
        <ul>
            <li>Start with low grid resolution (50-100)</li>
            <li>Use fewer interpolated frames (5-10)</li>
            <li>Disable saving initially for testing</li>
            <li>Monitor system resources</li>
        </ul>

        <h4>Plot Rendering Issues</h4>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Update graphics drivers</li>
            <li>Try different plot backends in IPython</li>
            <li>Reduce plot complexity or data points</li>
            <li>Close other graphics-intensive applications</li>
        </ul>

        <h3>Data Analysis Issues</h3>
    
        <h4>Unexpected Results</h4>
        <p><b>Checklist:</b></p>
        <ul>
            <li>Verify input file units and scaling</li>
            <li>Check parameter values (intensity, wavelength)</li>
            <li>Ensure time/frequency ranges are appropriate</li>
            <li>Compare with known reference calculations</li>
        </ul>

        <h4>Missing Features or Buttons</h4>
        <p><b>Possible Causes:</b></p>
        <ul>
            <li>Feature still in development (marked as "Experimental")</li>
            <li>UI scaling issues on high-DPI displays</li>
            <li>Window size too small</li>
        </ul>

        <h3>IPython Console Issues</h3>
    
        <h4>Console Not Responding</h4>
        <p><b>Solutions:</b></p>
        <ul>
            <li>Try Ctrl+C to interrupt current operation</li>
            <li>Restart the kernel if needed</li>
            <li>Check for infinite loops in custom code</li>
        </ul>

        <h4>Import Errors</h4>
        <p><b>Common Issues:</b></p>
        <ul>
            <li>Missing scientific Python packages</li>
            <li>Version compatibility problems</li>
            <li>Path configuration issues</li>
        </ul>

        <h3>Getting Help</h3>
    
        <h4>Before Contacting Support:</h4>
        <ol>
            <li>Note the exact error message</li>
            <li>Record steps to reproduce the issue</li>
            <li>Check file formats and data quality</li>
            <li>Try with smaller test datasets</li>
            <li>Check system resources (RAM, disk space)</li>
        </ol>

        <h4>Reporting Bugs:</h4>
        <ul>
            <li>Include software version (0.9.0-Beta)</li>
            <li>Describe operating system</li>
            <li>Attach problematic input files (if possible)</li>
            <li>Include screenshots of error messages</li>
        </ul>

        <h3>Best Practices</h3>
    
        <ul>
            <li><b>Start Small:</b> Test with simple cases before complex analysis</li>
            <li><b>Save Work:</b> Export important plots and data regularly</li>
            <li><b>Document:</b> Keep notes on successful parameter combinations</li>
            <li><b>Backup:</b> Keep copies of important input files</li>
            <li><b>Monitor:</b> Watch system resources during heavy calculations</li>
        </ul>
        """
        self.help_content.setHtml(content)

    def show_about_help(self):
        content = f"""
        <h2>{Symbols.EXCLAMATION} About Attoscience Studio</h2>
    
        <h3>Software Information</h3>
        <p><b>Version:</b> 1.0.0</p>
        <p><b>Purpose:</b> Analysis toolkit for attosecond physics, high harmonic generation, and ultrafast electron dynamics in solid-state systems</p>
    
        <h3>Key Capabilities</h3>
        <ul>
            <li> <b>Electronic Structure Analysis:</b> Crystal structure, Band structure, DOS, electron density</li>
            <li> <b>Laser Field Configuration:</b> Single/dual pulse setup, polarization gating</li>
            <li> <b>High Harmonic Generation:</b> Spectral analysis, yield calculations, phase studies</li>
            <li> <b>Attosecond Pulse Generation:</b> Pulse synthesis, duration optimization</li>
            <li> <b>Electron Dynamics:</b> Time-resolved k-space analysis, animations</li>
            <li> <b>Mathematical Tools:</b> Fourier transforms, unit conversions</li>
        </ul>
    
        <h3>Target Applications</h3>
        <ul>
            <li>Time-dependent density functional theory (TD-DFT) analysis</li>
            <li>High harmonic generation in crystals</li>
            <li>Attosecond pulse characterization and optimization</li>
            <li>Ultrafast electron dynamics in condensed matter</li>
            <li>Polarization gating and pulse shaping</li>
        </ul>
    
        <h3>Technical Details</h3>
        <p><b>Programming Language:</b> Python</p>
        <p><b>GUI Framework:</b> PyQt</p>
        <p><b>Scientific Libraries:</b> NumPy, SciPy, Matplotlib</p>
        <p><b>Interactive Console:</b> IPython integration</p>
    
        <h3>System Requirements</h3>
        <ul>
            <li><b>Operating System:</b> Linux</li>
            <li><b>RAM:</b> Minimum 4GB, Recommended 8GB+</li>
            <li><b>Storage:</b> 1GB free space for installation</li>
            <li><b>Graphics:</b> OpenGL support for visualizations</li>
        </ul>
    
        <h3>License & Usage</h3>
        <p>This software is provided for research and educational purposes.</p>
    
        <h3>Acknowledgments</h3>
        <p>This software builds upon research in attosecond science and computational physics. We acknowledge the contributions of the global attosecond physics community.</p>
    
        <h3>Version History</h3>
        <p><b>0.9.0-Beta (Current):</b></p>
        <ul>
            <li>Complete ground state analysis suite</li>
            <li>HHG spectral analysis tools</li>
            <li>Attosecond pulse generation and characterization</li>
            <li>Real-time electron dynamics visualization</li>
            <li>Comprehensive mathematical toolbox</li>
            <li>Integrated IPython console</li>
        </ul>
    
        <h3>Planned Features</h3>
        <ul>
            <li>Enhanced crystal structure visualization</li>
            <li>Advanced filtering and signal processing tools</li>
            <li>Export capabilities for publication-ready figures</li>
            <li>Batch processing for parameter sweeps</li>
            <li>Plugin system for custom analysis modules</li>
        </ul>
    
        <h3>Support & Contact</h3>
        <p>For technical support, bug reports, or feature requests, please contact the developer.</p>
    
        <h3>Contributing</h3>
        <p>We welcome contributions from the scientific community, including:</p>
        <ul>
            <li>Bug reports and feature suggestions</li>
            <li>Code contributions and improvements</li>
            <li>Documentation and tutorial development</li>
            <li>Testing and validation with different systems</li>
        </ul>
    
        <hr>
        <p><i>Attoscience Studio</i></p>
        """
        self.help_content.setHtml(content)

    #----------------------------------------------------------------

    def home(self):
        self.tabs.removeTab(0)
        self.tabs.insertTab(0, self.create_home_tab(), "Home") #QIcon('icons/home.png')
        pass
    
    #&&&&&&&&&&&&
   
    def show_crystal_structure(self):
        input_dialog_CTLS(self)
        self.log_activity("Opened Crystal Structure Dialog")
        self.log_data_summaries("Crystal Structure Data!")
        pass   
        
    def show_band_structure_dialog(self):
        input_dialog_BSTR(self)
        self.log_activity("Opened Band Structure Dialog")
        self.log_data_summaries("Band Structure Data!")
        pass

    def density_of_state(self):
        input_dialog_DOS(self)
        self.log_activity("Opened Density of State Dialog")
        self.log_data_summaries("Density of State Data!")
        pass

    def density(self):
        input_dialog_density(self)
        self.log_activity("Opened Density Dialog")
        self.log_data_summaries("Density Data!")
        pass
    
    #&&&&&&&&&&&&
    
    def total_current(self):
        input_dialog_tot_curr(self)
        self.log_activity("Opened Total Current Dialog")
        self.log_data_summaries("Total Current Data!")
        pass

    def spectrum(self):
        input_dialog_spectrum(self)
        self.log_activity("Opened High Harmonic Spectrum Dialog")
        self.log_data_summaries("Total Current Data!")
        pass

    def hhg_yield(self):
        input_dialog_YIELD(self)
        self.log_activity("Opened High Harmonic Yield Dialog")
        self.log_data_summaries("Total Current Data!")
        pass

    def ellipticity(self):
        input_dialog_ellips(self)
        self.log_activity("Opened High Harmonic Ellipticity Dialog")
        self.log_data_summaries("Total Current Data!")
        pass

    def phase_analysing(self):
        input_dialog_PHASE(self)
        self.log_activity("Opened Phase Analysis of High Harmonics Dialog")
        self.log_data_summaries("Total Current Data!")
        pass
    
    #&&&&&&&&&&&&
    
    def attosecond_pulse(self):
        input_dialog_atto(self)
        self.log_activity("Opened Attosecond Pulse Dialog")
        self.log_data_summaries("Total Current Data!")
        pass        
    
    def find_minimum_pulse_width_FMPW(self):
        input_dialog_FMPW(self)
        self.log_activity("Opened Minimum Pulse Width Dialog")
        self.log_data_summaries("Total Current Data!")
        pass
    ###======================###
    def start_mpw_computation(self, lambda0_nm, qstart, qmax, t, dt, jx, jy):
        # <<Disable UI>>
        self.mpw_button.setEnabled(False)
    
        self.safe_cleanup_thread()

        self.mpw_thread = QThread()
        # CALL ---------------------
        self.mpw_worker = MPWWorker(lambda0_nm, qstart, qmax, t, dt, jx, jy)
        self.mpw_worker.moveToThread(self.mpw_thread)

        self.mpw_worker.finished.connect(self.handle_mpw_result)
        self.mpw_worker.error.connect(self.handle_mpw_error)
        self.mpw_thread.started.connect(self.mpw_worker.run)
        self.mpw_thread.finished.connect(self.safe_cleanup_thread)

        self.mpw_thread.start()

    def handle_mpw_result(self, min_FWHM, optimal_qstart, optimal_qmax, OC, last_OC, max_Time_OC):
        if last_OC < OC <= max_Time_OC:
            warning_message = "Warning: The attosecond pulse's time position is near the end..."
            QMessageBox.warning(self, "FWHM Calculation Warning", warning_message)
        
        messages = [
            f'Full Width at Half Maximum = {min_FWHM:.0f} [as]',
            f'Optimal Minimum HO = {optimal_qstart:.0f}',
            f'Optimal Maximum HO = {optimal_qmax:.0f}',
        ]
        self.show_MPW_information(messages)
        
        self.statusBar().showMessage("Calculation completed", 3000)

    def handle_mpw_error(self, error_msg):
        QMessageBox.critical(self, "Computation Error", error_msg)
        self.statusBar().showMessage("Error in calculation", 5000)

    def safe_cleanup_thread(self):
        worker = getattr(self, 'mpw_worker', None)
        thread = getattr(self, 'mpw_thread', None)
    
        if worker is not None:
            try:
                worker.stop()
                worker.deleteLater()
            except AttributeError:
                pass
    
        if thread is not None:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
            thread.deleteLater()
    
        if hasattr(self, 'mpw_thread'):
            del self.mpw_thread
        if hasattr(self, 'mpw_worker'):
            del self.mpw_worker
    
        self.mpw_button.setEnabled(True)
    
    def closeEvent(self, event):
        if hasattr(self, 'mpw_thread') and self.mpw_thread.isRunning():
            self.safe_cleanup_thread() ##>>>>>>
        event.accept()
        
    def show_MPW_information(self,messages_MPW):
        info_box = QMessageBox()
        info_box.setIcon(QMessageBox.Information)
        info_box.setWindowTitle("MPW Information")
    
        info_msg = '\n'.join(messages_MPW)
        info_box.setText(info_msg)
        font = info_box.font()
        font.setPointSize(10)
        font.setFamily("Arial") 
        info_box.setFont(font)
    
        info_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e1e; 
                padding: 10px;             
            }
            QLabel {
                color: #ffffff;            
            }
            QPushButton {
                background-color: #4caf50; 
                color: white;
                border-radius: 20px;
                padding: 5px 10px;
            }
            QPushButton:hover {
               background-color: #45a049;
            }
        """)
        info_box.exec_()

    ###=====================###
    
    def time_frequency_gabor(self):    
        input_dialog_time_frequency(self)
        self.log_activity("Opened Gabor Transform Dialog")
        self.log_data_summaries("Total Current Data!") #Loaded
        pass
     
    #&&&&&&&&&&&&
            
    def electric_field(self):
        select_and_plot_electric_field(self)
        self.log_activity("Opened Electric Field Dialog")
        self.log_data_summaries("Electric Field (laser) Data!") 
        pass
        
    def vector_potential(self):
        select_and_plot_vector_potential(self)
        self.log_activity("Opened Vector Potential Dialog")
        self.log_data_summaries("Vector Potential (laser) Data!")
        pass

    def pg_analyzing(self):
        input_dialog_pg(self)
        self.log_activity("Opened Configure Polarization Gating Field Dialog")
        pass
        
    def gate_width(self):
        input_dialog_gw(self)
        self.log_activity("Opened Determine Polarization Gate Width Dialog")
        pass
    
    #&&&&&&&&&&&&
    
    def nex_vs_time(self):
        input_dialog_nex(self)
        self.log_activity("Opened Track Excited Electrons Over Time Dialog")
        self.log_data_summaries("Nex Data!")
        pass
        
    def nex_distribution(self):
        input_dialog_bznex(self)
        self.log_activity("Opened excited electrons across the Brillouin zone Dialog")
        self.log_data_summaries("Nex Data!")
        pass

    def curr_distribution(self):
        input_dialog_bzcurr(self)
        self.log_activity("Opened current across the Brillouin zone Dialog")
        self.log_data_summaries("current Data!")
        pass
    ###============================###
    
    def current_nex_animation(self):
        try:
            figure = create_current_nex_analysis(self) ##>>>>>>>>>>>>>>>>>
            if figure:
                self.show_analysis_result(figure)      ##>>>>>>>>>>>>>>>>>
                return True
    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create analysis: {str(e)}")
            return False
    
    def show_analysis_result(self, figure):
        dialog = QDialog(self)
        dialog.setWindowTitle("Current and Nex k-space Analysis")
        dialog.setMinimumSize(1200, 800)
    
        layout = QVBoxLayout(dialog)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
    
        #-----------------------------------------------
        # controller
        controller = AnimationController(figure, canvas, self.ipy_console) ##>>>>>>>>>>>>>>>>>
        dialog.controller = controller
        #-----------------------------------------------
        
        # control buttons
        control_layout = QHBoxLayout()
    
        play_btn = QPushButton("Play")
        pause_btn = QPushButton("Pause")
        reset_btn = QPushButton("Reset")
    
        play_btn.clicked.connect(controller.start_animation)
        pause_btn.clicked.connect(controller.pause_animation)
        reset_btn.clicked.connect(controller.reset_animation)
        
        control_layout.addWidget(play_btn)
        control_layout.addWidget(pause_btn)
        control_layout.addWidget(reset_btn)
        control_layout.addStretch()
    
        layout.addLayout(control_layout)
    
        # close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: (
            controller.stop_animation(),
            dialog.accept()
        ))
        layout.addWidget(close_btn)
    
        dialog.exec_()

    def create_animation_controls(self, figure, canvas, parent_dialog):
        controls_layout = QHBoxLayout()
        play_btn = QPushButton("Play")
        pause_btn = QPushButton("Pause")
        reset_btn = QPushButton("Reset")
    
        controls_layout.addWidget(play_btn)
        controls_layout.addWidget(pause_btn)
        controls_layout.addWidget(reset_btn)
        controls_layout.addStretch()
        
        #-----------------------------------------------
        # << Initialize animation controller >>
        controller = AnimationController(figure, canvas) ##>>>>>>>>>>>>>>>>>
        parent_dialog.animation_controller = controller
        #-----------------------------------------------
        
        # Connect signals
        play_btn.clicked.connect(controller.start_animation)  ##>>>>>>>>>>>>>>>>>
        pause_btn.clicked.connect(controller.pause_animation) ##>>>>>>>>>>>>>>>>>
        reset_btn.clicked.connect(controller.reset_animation) ##>>>>>>>>>>>>>>>>>
    
        return controls_layout

    #&&&&&&&&&&&&
    
    def ft_func(self):
        input_dialog_ft_func = InputDialogFTFunc(self)
        input_dialog_ft_func.exec_()
        self.log_activity("Opened Fourier Transform Function Dialog")
        #self.start_long_task()

    def ft_data(self):
        input_dialog_ft_data = InputDialogFTData(self)
        input_dialog_ft_data.exec_()
        self.log_activity("Opened Fourier Transform Function Dialog")
        #self.start_long_task()

    def ift_func(self):
        input_dialog_ift_func = InputDialogInverseFTFunc(self)
        input_dialog_ift_func.exec_()
        self.log_activity("Opened Fourier Transform Function Dialog")
        #self.start_long_task()

    def ift_data(self):
        input_dialog_ift_data = InputDialogIFTData(self)
        input_dialog_ift_data.exec_()
        self.log_activity("Opened Fourier Transform Function Dialog")
        #self.start_long_task()
        
    def unit_conversion(self):
        unit_conversion_dialog = UnitConversionDialog(self)
        unit_conversion_dialog.exec_()
        self.log_activity("Opened Unit Conversion Dialog")

#----------------------------------------------------------------

class ProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self._glow_opacity = 0.0
        
        self.glow_animation = QPropertyAnimation(self, b"glowOpacity")
        self.glow_animation.setDuration(1500)
        self.glow_animation.setStartValue(0.0)
        self.glow_animation.setEndValue(1.0)
        self.glow_animation.setEasingCurve(QEasingCurve.InOutSine)
        self.glow_animation.setLoopCount(-1)
        self.glow_animation.start()
        
    @pyqtProperty(float)
    def glowOpacity(self):
        return self._glow_opacity
    
    @glowOpacity.setter
    def glowOpacity(self, value):
        self._glow_opacity = value
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        bg_rect = self.rect().adjusted(2, 2, -2, -2)
        painter.setBrush(QBrush(QColor(15, 15, 20, 200)))
        painter.setPen(QPen(QColor(60, 60, 70), 1))
        painter.drawRoundedRect(bg_rect, 12, 12)
        
        # Progress fill
        if self.value() > 0:
            progress_width = int((bg_rect.width() - 4) * self.value() / self.maximum())
            progress_rect = QRect(bg_rect.x() + 2, bg_rect.y() + 2, 
                                progress_width, bg_rect.height() - 4)
            
            # Gradient effect
            gradient_color = QColor(0, 150, 255, int(200 + 55 * self._glow_opacity))
            painter.setBrush(QBrush(gradient_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(progress_rect, 8, 8)
            
            # Glow effect
            glow_color = QColor(0, 200, 255, int(100 * self._glow_opacity))
            painter.setBrush(QBrush(glow_color))
            glow_rect = progress_rect.adjusted(-2, -2, 2, 2)
            painter.drawRoundedRect(glow_rect, 10, 10)
        
        painter.setPen(QPen(QColor(255, 255, 255, 200)))
        font = QFont("Segoe UI", 9, QFont.Medium)
        painter.setFont(font)
        text = f"{self.value()}%"
        text_rect = self.rect()
        painter.drawText(text_rect, Qt.AlignCenter, text)

#----------------------------------------------------------------

class SplashScreen(QSplashScreen):
    def __init__(self, pixmap):
        super().__init__(pixmap, Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)
        
        # overlay widget
        self.overlay = QWidget(self)
        self.overlay.setGeometry(self.rect())
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        # layout
        layout = QVBoxLayout(self.overlay)
        layout.setContentsMargins(50, 50, 50, 80)
        layout.setSpacing(20)
        
        # Spacer
        layout.addStretch()
        
        # App name label
        self.app_label = QLabel("ATTOSCIENCE STUDIO")
        self.app_label.setAlignment(Qt.AlignCenter)
        self.app_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 240);
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 24px;
                font-weight: 300;
                letter-spacing: 3px;
                background: transparent;
                padding: 10px;
            }
        """)
        layout.addWidget(self.app_label)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 180);
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 12px;
                font-weight: 400;
                background: transparent;
                padding: 5px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # progress bar
        self.progress_bar = ProgressBar()  ###>>>>>>>>>>>>>>>>>>
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self.progress_bar)
        
        # Version/loading info
        self.info_label = QLabel("Loading components...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 120);
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 10px;
                font-weight: 400;
                background: transparent;
                padding: 10px;
            }
        """)
        layout.addWidget(self.info_label)
        
        # Animation for app title
        self.title_animation = QPropertyAnimation(self.app_label, b"geometry")
        self.title_animation.setDuration(800)
        self.title_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Fade in animation
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.fade_in_elements)
        self.fade_timer.setSingleShot(True)
        self.fade_timer.start(200)
        
    def fade_in_elements(self):
        self.app_label.setStyleSheet(self.app_label.styleSheet().replace("240", "255"))
        
    def update_progress(self, value, status="Loading..."):
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
        
        if value < 20:
            self.info_label.setText("Initializing core systems...")
        elif value < 40:
            self.info_label.setText("Loading user interface...")
        elif value < 60:
            self.info_label.setText("Preparing workspace...")
        elif value < 80:
            self.info_label.setText("Finalizing setup...")
        else:
            self.info_label.setText("Ready to launch!")
            
    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        overlay_rect = self.rect()
        gradient_brush = QBrush(QColor(10, 15, 25, 100))
        painter.setBrush(gradient_brush)
        painter.setPen(Qt.NoPen)
        painter.drawRect(overlay_rect)
        
        glow_pen = QPen(QColor(0, 150, 255, 50), 2)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(overlay_rect.adjusted(1, 1, -1, -1), 8, 8)

def is_already_running(key="attoscience_studio_instance_lock"):
    shared_memory = QSharedMemory(key)
    
    if shared_memory.attach():
        return True
    
    if not shared_memory.create(1):
        return True
    
    return False
#----------------------------------------------------------------
def main():
    instance_checker = SingleInstance("attoscience_studio_instance")
    if instance_checker.is_running():
        print("Attoscience Studio is already running.")
        sys.exit(0)
    instance_checker.start()
    
    app = QApplication(sys.argv)
    
    ###+++++++++++++++++++++++++++++++++++++++++

    # pixmap
    original_pixmap = QPixmap(":/icons/SplashScreen2.png")
    if original_pixmap.isNull():
        # if image not found
        splash_pix = QPixmap(600, 500)
        splash_pix.fill(QColor(20, 25, 35))
        painter = QPainter(splash_pix)
        painter.setRenderHint(QPainter.Antialiasing)
        
        gradient_brush = QBrush(QColor(25, 35, 50))
        painter.setBrush(gradient_brush)
        painter.drawRect(splash_pix.rect())
        painter.end()
    else:
        splash_pix = original_pixmap.scaled(500, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    # splash screen
    splash = SplashScreen(splash_pix) ###>>>>>>>>>>>>>>>>>>>>>>>
    splash.show()
    
    progress = [0]
    loading_stages = [
        (10, "Initializing..."),
        (25, "Loading modules..."),
        (45, "Setting up workspace..."),
        (65, "Configuring interface..."),
        (85, "Finalizing..."),
        (100, "Complete!")
    ]
    stage_index = [0]
    def update_progress():
        if stage_index[0] < len(loading_stages):
            target_progress, status = loading_stages[stage_index[0]]
            
            if progress[0] < target_progress:
                progress[0] += 2
                splash.update_progress(progress[0], status)
            else:
                stage_index[0] += 1
                
            if progress[0] >= 100:
                QTimer.singleShot(20, lambda: [splash.close(), main_window.show()])
                return
        
        QTimer.singleShot(30, update_progress)
    ###+++++++++++++++++++++++++++++++++++++++++
    
    #-------------------------
    QTimer.singleShot(10, update_progress)
    #-------------------------
    main_window = MainWindow()
    #-------------------------
    splash.finish(main_window)
    #-------------------------
    
    sys.exit(app.exec_())
#----------------------------------------------------------------
if __name__ == '__main__':
    main()
    
 



