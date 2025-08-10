# gs/dns.py

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
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication, QPropertyAnimation
from numpy import trapz
from scipy.integrate import trapezoid, quad
from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from attoscience_studio.resources_rc import *
##----------------------------------------------------
def read_DENSITY(file_path, selected_formats):
    try:
        data = np.loadtxt(file_path)
        if data.size == 0:
            raise ValueError("The file is empty.")
        i = data[:, 0]
        j = data[:, 1]
        if any(fmt in ["plane_x", "plane_y", "plane_z"] for fmt in selected_formats):
            k = data[:, 2]
        else:
            k = np.nan
        if i.size == 0:
            raise ValueError("No data found in the file.")
        return i, j, k
    except Exception as e:
        raise ValueError(f"Failed to read Density: {e}")
##----------------------------------------------------
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")

def density_connector(selected_formats, plot_settings, ipy_console):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select Density file")
    if not any(substring in file_path.lower() for substring in [
        "density.y=0,z=0",
        "density.x=0,z=0",
        "density.x=0,y=0",
        "density.x=0",
        "density.y=0",
        "density.z=0"
        ]):
        QMessageBox.warning(None, "File Error", "Please upload the density.y=0,z=0 or density.x=0,z=0 or density.x=0,y=0 or density.x=0 or density.y=0 or density.z=0 file.")
        return
    if file_path:
        try:
            i,j,k = read_DENSITY(file_path,selected_formats)
            plot_Density(i,j,k,selected_formats, plot_settings)
            
            num_rows = len(i)
           
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            msg = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          Electron Density log!                          --\n"
                + "-" * 75 + "\n"
                ">>> Electron density plotted successfully!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> num_rows: {num_rows}\n"
                + "-" * 75
            )
            print_to_console(ipy_console, msg)

        except Exception as e:
            print(f"Error parsing Density file: {e}")
            QMessageBox.critical(None, "Error", f"Failed to parse Density file:\n{e}")

##----------------------------------------------------
def plot_Density(i,j,k,selected_formats, plot_settings):   
    if 'axis_x' in selected_formats:
        plt.figure()
        plt.plot(i, j, linewidth=plot_settings.get("line_thickness", 1.2),
                 color=plot_settings.get("line_color", "black"))                 
        plt.xlabel(plot_settings.get("x_label", "x"))
        plt.ylabel(plot_settings.get("y_label", "Density (Re)"))
        plt.title(plot_settings.get("graph_title", "Density (Re)"))
        plt.grid(False)
        plt.xlim(left=plot_settings.get("x_min", np.min(i)), right=plot_settings.get("x_max", np.max(i)))
        plt.ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
        plt.show()

    if 'axis_y' in selected_formats:
        plt.figure()
        plt.plot(i, j, linewidth=plot_settings.get("line_thickness", 1.2),
                 color=plot_settings.get("line_color", "black"))                 
        plt.xlabel(plot_settings.get("x_label", "y"))
        plt.ylabel(plot_settings.get("y_label", "Density (Re)"))
        plt.title(plot_settings.get("graph_title", "Density (Re)"))
        plt.grid(False)
        plt.xlim(left=plot_settings.get("x_min", np.min(i)), right=plot_settings.get("x_max", np.max(i)))
        plt.ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
        plt.show()
        
    if 'axis_z' in selected_formats:
        plt.figure()
        plt.plot(i, j, linewidth=plot_settings.get("line_thickness", 1.2),
                 color=plot_settings.get("line_color", "black"))                 
        plt.xlabel(plot_settings.get("x_label", "z"))
        plt.ylabel(plot_settings.get("y_label", "Density (Re)"))
        plt.title(plot_settings.get("graph_title", "Density (Re)"))
        plt.grid(False)
        plt.xlim(left=plot_settings.get("x_min", np.min(i)), right=plot_settings.get("x_max", np.max(i)))
        plt.ylim(bottom=plot_settings.get("y_min", None), top=plot_settings.get("y_max", None))
        plt.show()

    xi = np.unique(i)
    yj = np.unique(j)
    density_grid = np.full((len(yj), len(xi)), np.nan)
    
    for x, y, density in zip(i, j, k):
        x_idx = np.where(xi == x)[0][0]
        y_idx = np.where(yj == y)[0][0]
        density_grid[y_idx, x_idx] = density
    
    if 'plane_x' in selected_formats:
        plt.figure()
        plt.imshow(density_grid, extent=(xi.min(), xi.max(), yj.min(), yj.max()), 
                   origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Density (Re)')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('Density Distribution (Re) in plane_x')
        plt.show()

    if 'plane_y' in selected_formats:
        plt.figure()
        plt.imshow(density_grid, extent=(xi.min(), xi.max(), yj.min(), yj.max()), 
                   origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Density (Re)')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Density Distribution (Re) in plane_y')
        plt.show()

    if 'plane_z' in selected_formats:
        plt.figure()
        plt.imshow(density_grid, extent=(xi.min(), xi.max(), yj.min(), yj.max()), 
                   origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Density (Re)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Density Distribution (Re) in plane_z')
        plt.show()

##----------------------------------------------------
previous_fermi_energy_H = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Electron Density Parameters")
        self.setFixedSize(650, 450)
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
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title = QLabel("Electron Density Visualization")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")
        
        subtitle = QLabel("Configure format option for electron density plotting")
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

        # Axes Group
        axes_group = QGroupBox("Axes")
        axes_layout = QFormLayout()
    
        self.axis_x_checkbox = QCheckBox("axis_x", self)
        self.axis_y_checkbox = QCheckBox("axis_y", self)
        self.axis_z_checkbox = QCheckBox("axis_z", self)
    
        axes_layout.addRow(self.axis_x_checkbox)
        axes_layout.addRow(self.axis_y_checkbox)
        axes_layout.addRow(self.axis_z_checkbox)
        axes_group.setLayout(axes_layout)

        # Planes Group
        planes_group = QGroupBox("Planes")
        planes_layout = QFormLayout()
    
        self.plane_x_checkbox = QCheckBox("plane_x", self)
        self.plane_y_checkbox = QCheckBox("plane_y", self)
        self.plane_z_checkbox = QCheckBox("plane_z", self)
    
        planes_layout.addRow(self.plane_x_checkbox)
        planes_layout.addRow(self.plane_y_checkbox)
        planes_layout.addRow(self.plane_z_checkbox)
        planes_group.setLayout(planes_layout)

        # Axes and Planes into a horizontal layout
        group_layout = QHBoxLayout()
        group_layout.addWidget(axes_group)
        group_layout.addWidget(planes_group)

        required_layout.addLayout(group_layout)
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
        # Connect checkbox to show/hide advanced options
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
    
    def create_labels_tab(self):
        labels_widget = QWidget()
        labels_layout = QFormLayout(labels_widget)
        labels_layout.setSpacing(12)
        
        self.graph_title_entry = QLineEdit()
        self.graph_title_entry.setPlaceholderText("Enter plot title")
        labels_layout.addRow("Graph Title:", self.graph_title_entry)
        
        self.x_label_entry = QLineEdit()
        self.x_label_entry.setPlaceholderText("e.g., Wave Vector")
        labels_layout.addRow("X-axis Label:", self.x_label_entry)
        
        self.y_label_entry = QLineEdit()
        self.y_label_entry.setPlaceholderText("e.g., Energy [eV]")
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
        
        # Submit button with icon
        submit_btn = QPushButton("Generate Plot")
        submit_btn.setIcon(QIcon(":/icons/analytics_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png"))
        submit_btn.setIconSize(QSize(24, 24))
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self.on_submit)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)
    
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
            selected_formats = []
            if self.axis_x_checkbox.isChecked():
                selected_formats.append('axis_x')
            if self.axis_y_checkbox.isChecked():
                selected_formats.append('axis_y')
            if self.axis_z_checkbox.isChecked():
                selected_formats.append('axis_z')
            if self.plane_x_checkbox.isChecked():
                selected_formats.append('plane_x')
            if self.plane_y_checkbox.isChecked():
                selected_formats.append('plane_y')
            if self.plane_z_checkbox.isChecked():
                selected_formats.append('plane_z')

            if len(selected_formats) != 1:
                QMessageBox.warning(self, "Invalid Input", "Please select exactly one format option.")
                return None, None
            
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
                    "graph_title": self.graph_title_entry.text(),
                    "x_label": self.x_label_entry.text(),
                    "y_label": self.y_label_entry.text(),
                    "line_color": line_color,
                    "line_thickness": self.line_thickness_spinbox.value(),
                    "background_color": self.background_color_entry.text(),
                }
            
            self.accept()
            
            # CALL
            density_connector(selected_formats, plot_settings, self.parent().ipy_console)
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def input_dialog_density(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()

