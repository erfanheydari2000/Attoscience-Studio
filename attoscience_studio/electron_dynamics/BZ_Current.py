# electron_dynamics/BZ_Current.py

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
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
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
def read_bznex(file_path, file_format):
    try:
        data_i = np.loadtxt(file_path[0])
        data_j = np.loadtxt(file_path[1])
        
        if data_i.size == 0 or data_j.size == 0:
            raise ValueError("The file is empty.")
        
        curr_i = data_i[:, 2]
        curr_j = data_j[:, 2]
        mag_curr = np.sqrt(curr_i**2 + curr_j**2)

        """
        It does not matter which data file we get the K coordinates from because for each of the 
        plan_x, plane_y and plane_z formats, the coordinates in the data files are the same.
        """
        
        ki = data_i[:, 0]
        kj = data_i[:, 1] 
            
        if mag_curr.size == 0 or ki.size == 0 or kj.size == 0:
            raise ValueError("No current(i,j) or k(x,y,z) data found in the file.")
        return ki, kj, mag_curr
    except Exception as e:
        raise ValueError(f"Failed to read current data file: {e}")
##----------------------------------------------------
def grid_interp(ki, kj, mag_curr, A, interp_method):
    try:
        i_grid = np.linspace(np.min(ki), np.max(ki), A)
        j_grid = np.linspace(np.min(kj), np.max(kj), A)
        i_grid, j_grid = np.meshgrid(i_grid, j_grid)
    
        # Interpolate
        nex_interp = griddata((ki, kj), mag_curr, (i_grid, j_grid), method=interp_method)
    
    except Exception as e:   
        raise RuntimeError(f"Grid interpolation failed with method='{interp_method}': {e}") from e
    
    return nex_interp
##----------------------------------------------------
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")

def bzcurr_connector(A, interp_method, file_format, plot_settings, ipy_console=None):
    file_path = []
    if 'plane_x' in file_format:
        file_path_y, _ = QFileDialog.getOpenFileName(None, "Select 'current_kpt-y.kx=0' file")
        file_path.append(file_path_y)
        file_path_z, _ = QFileDialog.getOpenFileName(None, "And select 'current_kpt-z.kx=0' file")
        file_path.append(file_path_z)
        
    elif 'plane_y' in file_format:
        file_path_x, _ = QFileDialog.getOpenFileName(None, "Select 'current_kpt-x.ky=0' file")
        file_path.append(file_path_x)
        file_path_z, _ = QFileDialog.getOpenFileName(None, "And select 'current_kpt-z.ky=0' file")
        file_path.append(file_path_z)

    elif 'plane_z' in file_format:
        file_path_x, _ = QFileDialog.getOpenFileName(None, "Select 'current_kpt-x.kz=0' file")
        file_path.append(file_path_x)
        file_path_y, _ = QFileDialog.getOpenFileName(None, "And select 'current_kpt-y.kz=0' file")
        file_path.append(file_path_y)


    if not any(substring in file_path[0].lower() for substring in [
        "current_kpt-y.kx=0",
        "current_kpt-x.ky=0",
        "current_kpt-x.kz=0"
        ]):
        QMessageBox.warning(None, "File Error", "Please upload the 'current_kpt-y.kx=0' or 'current_kpt-x.ky=0' or 'current_kpt-x.kz=0' file.")
        return
    if not any(substring in file_path[1].lower() for substring in [
        "current_kpt-z.kx=0",
        "current_kpt-z.ky=0",
        "current_kpt-y.kz=0"
        ]):
        QMessageBox.warning(None, "File Error", "Please upload the 'current_kpt-z.kx=0' or 'current_kpt-z.ky=0' or 'current_kpt-y.kz=0' file.")
        return

    if file_path:
        try:
            ki, kj, mag_curr = read_bznex(file_path, file_format)

            curr_interp = grid_interp(ki, kj, mag_curr, A, interp_method)
            
            bzcurr_plot(ki, kj, curr_interp, plot_settings)
   
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            msg = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          Current across the BZ log!                          --\n"
                + "-" * 75 + "\n"
                ">>> BZ curr visualization successfully completed!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> Grid Resolution: {A}\n"
                f">>> Interpolation method: {interp_method}\n"
                + "-" * 75
            )
            print_to_console(ipy_console, msg)
        
        except ValueError as e:
            QMessageBox.warning(None, "Error", str(e))
            return

##----------------------------------------------------
def bzcurr_plot(ki, kj, curr_interp, plot_settings):
    fig = plt.figure()
    extent = (np.min(ki), np.max(ki), np.min(kj), np.max(kj))

    ax1 = fig.add_subplot(1, 1, 1)
    img1 = ax1.imshow(
        curr_interp,
        extent=extent,
        aspect=plot_settings.get("aspect", 'auto'),
        origin=plot_settings.get("origin", 'lower'),
        cmap=plot_settings.get("cmap", 'jet'),
        interpolation=plot_settings.get("interpolation", 'nearest'),
        vmin=plot_settings.get("vmin", None),
        vmax=plot_settings.get("vmax", None)
    )
    plt.colorbar(img1, ax=ax1, label=plot_settings.get("colorbar_label", 'Current'))
    img1.set_clim(plot_settings.get("clim_min", None), plot_settings.get("clim_max", None))

    ax1.set_title(plot_settings.get("graph_title", "Excited electrons"))
    ax1.set_xlabel(plot_settings.get("x_label", r'$\mathregular{K_x\ [2\pi/a]}$'))
    ax1.set_ylabel(plot_settings.get("y_label", r'$\mathregular{K_y\ [2\pi/a]}$'))
    ax1.set_xlim(left=plot_settings.get("x_min", None), right=plot_settings.get("x_max", None))
    ax1.set_ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))

    plt.tight_layout()
    plt.show()
#----------------------------------------------------

previous_values_bzcurr = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Current across the BZ")
        self.setFixedSize(700, 500)
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
        title = QLabel("Current across the BZ")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")

        subtitle = QLabel("Configure parameters to plot current across the Brillouin zone.")
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
        params_group = QGroupBox("Essential Parameters")
        combined_layout = QVBoxLayout()
        combined_layout.setSpacing(20)

        # for up/down button
        light_blue_style = """
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            background-color: #ADD8E6;  /* Light blue */
            border: 1px solid #8AB6D6;
            width: 16px;
        }

        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #BEE3F7;
        }
        """

        # Grid resolution -----------------------------
        params_layout = QHBoxLayout()
        params_layout.setSpacing(20)

        grid_layout = QVBoxLayout()
        grid_label = QLabel("Grid Resolution:")
        self.grid_spinbox = QSpinBox()
        self.grid_spinbox.setRange(1, 1000)
        self.grid_spinbox.setValue(previous_values_bzcurr.get("A", 50))
        self.grid_spinbox.setSuffix(" points")
        self.grid_spinbox.setStyleSheet(light_blue_style)
        grid_layout.addWidget(grid_label)
        grid_layout.addWidget(self.grid_spinbox)

        # Interpolation Method -----------------------------
        interp_method_layout = QVBoxLayout()
        interp_method_label = QLabel("Interpolation Method:")
        self.interp_method_entry = QComboBox()
        self.interp_method_entry.addItems(["cubic", "linear", "nearest"])
        self.interp_method_entry.setStyleSheet("""QComboBox { min-height: 30px; font-size: 12px; }""")

        default_interp_method = previous_values_bzcurr.get("interp_method", "cubic").lower()
        index = self.interp_method_entry.findText(default_interp_method)
        if index >= 0:
            self.interp_method_entry.setCurrentIndex(index)

        interp_method_layout.addWidget(interp_method_label)
        interp_method_layout.addWidget(self.interp_method_entry)

        params_layout.addLayout(grid_layout)
        params_layout.addLayout(interp_method_layout)

        # File format options -----------------------------
        file_format_layout = QHBoxLayout()
        file_format_layout.setSpacing(180)

        self.plane_x_checkbox = QCheckBox("plane-x")
        self.plane_y_checkbox = QCheckBox("plane-y")
        self.plane_z_checkbox = QCheckBox("plane-z")

        # Connect signals to custom handler
        self.plane_x_checkbox.stateChanged.connect(lambda state: self.only_one_checked(self.plane_x_checkbox))
        self.plane_y_checkbox.stateChanged.connect(lambda state: self.only_one_checked(self.plane_y_checkbox))
        self.plane_z_checkbox.stateChanged.connect(lambda state: self.only_one_checked(self.plane_z_checkbox))

        file_format_layout.addWidget(self.plane_x_checkbox)
        file_format_layout.addWidget(self.plane_y_checkbox)
        file_format_layout.addWidget(self.plane_z_checkbox)
        file_format_layout.addStretch()

        file_format_group = QGroupBox("File format")
        file_format_group.setLayout(file_format_layout)        
        
        # Combine both into the final group
        combined_layout.addLayout(params_layout)
        combined_layout.addWidget(file_format_group)

        params_group.setLayout(combined_layout)
        layout.addWidget(params_group)

    ##---helper------------------------------
    def only_one_checked(self, selected):
        if selected.isChecked():
            for checkbox in [self.plane_x_checkbox, self.plane_y_checkbox, self.plane_z_checkbox]:
                if checkbox is not selected:
                    checkbox.setChecked(False)         
    ##----------------------------------------
    
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
            A = self.grid_spinbox.value()

            interp_method = self.interp_method_entry.currentText()
  
            file_format = []
            if self.plane_x_checkbox.isChecked():
                file_format.append('plane_x')
            elif self.plane_y_checkbox.isChecked():
                file_format.append('plane_y')
            elif self.plane_z_checkbox.isChecked():
                file_format.append('plane_z')

            if not file_format:
                QMessageBox.warning(self, "Invalid input", "Please select one file format.")
                return

            plot_settings = {}
            if self.plot_options_checkbox.isChecked():
                
                #line_color = self.line_color_entry.text().strip() or "black"
                #background_color = self.background_color_entry.text().strip() or "white"
                
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

            self.accept()
            
            # Update
            previous_values_bzcurr.update({"A": A})
            
            # CALL
            bzcurr_connector(A, interp_method, file_format, plot_settings, self.parent().ipy_console)
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def input_dialog_bzcurr(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()

