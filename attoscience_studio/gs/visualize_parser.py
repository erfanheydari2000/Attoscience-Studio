# gs/visualize_parser.py

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

import re
import numpy as np
import sys
import os
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib import gridspec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar,
                             QRadioButton, QButtonGroup, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QScrollArea, QSlider, QStyle, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton, QGraphicsOpacityEffect)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication, QPropertyAnimation
from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from attoscience_studio.parser.parserlog_parser import *
from attoscience_studio.parser.cif_parser import *
from attoscience_studio.utils.atome_styles_size import * 
from attoscience_studio.resources_rc import *
##----------------------------------------------------
def plot_crystal_structure_parserlog(parser, 
                          atom_scale=1.0, 
                          element_colors=None, 
                          element_sizes=None, 
                          figsize=(10, 8),
                          show_axes=False):
    if not parser.scaled_vectors:
        print("Scaled lattice vectors not available. Cannot plot.")
        return
    
    #-----------------------------------------
    cpk_colors = get_cpk_colors()     ###>>>>>
    atomic_radii = get_atomic_radii() ###>>>>>
    #-----------------------------------------
    
    if element_colors is None:
        element_colors = {}
        elements = list(set(atom[0] for atom in parser.reduced_coords))
        
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(elements)))
        
        for i, elem in enumerate(elements):
            element_colors[elem] = cpk_colors.get(elem, color_cycle[i])
    
    if element_sizes is None:
        element_sizes = {}
        base_radius = 600
        for elem in set(atom[0] for atom in parser.reduced_coords):
            radius = atomic_radii.get(elem, 150)
            element_sizes[elem] = atom_scale * base_radius * (radius / 200)
    

    scaled_vectors = parser.scaled_vectors
    reduced_coords = parser.reduced_coords

    wrapped_coords = []
    for atom in reduced_coords:
        symbol = atom[0]
        frac_coords = np.array(atom[1:4])
        wrapped_frac = frac_coords % 1.0
        wrapped_coords.append([symbol] + wrapped_frac.tolist())
    
    # Convert fractional coordinates to Cartesian
    cart_coords = []
    for atom in wrapped_coords:
        symbol = atom[0]
        frac = np.array(atom[1:4])
        cart = np.zeros(3)
        for i in range(3):
            cart += frac[i] * np.array(scaled_vectors[i])
        cart_coords.append([symbol, cart])
    
    # Generate unit cell vertices <<<<<<(8 points)>>>>>>
    vertices_frac = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                vertices_frac.append([i, j, k])
    
    vertices_cart = []
    for vf in vertices_frac:
        vc = np.zeros(3)
        for i in range(3):
            vc += vf[i] * np.array(scaled_vectors[i])
        vertices_cart.append(vc)
    
    # Define edges of the unit cell <<<<<<(12 edges)>>>>>>>>>>
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # unit cell edges-----------------
    for edge in edges:
        p1 = vertices_cart[edge[0]]
        p2 = vertices_cart[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='black', linewidth=1.5, alpha=0.8)
    
    # atoms-----------------
    for symbol, coords in cart_coords:
        ax.scatter(coords[0], coords[1], coords[2], 
                   color=element_colors[symbol],
                   s=element_sizes[symbol],
                   edgecolor='black',
                   alpha=0.9,
                   label=symbol)
    
    ax.set_title(f'Crystal Structure: {", ".join(elements)}', fontsize=14)
    
    # equal aspect ratio-----------------
    min_vals = np.min(vertices_cart, axis=0)
    max_vals = np.max(vertices_cart, axis=0)
    max_range = max(max_vals - min_vals) / 2.0
    
    mid_x = (min_vals[0] + max_vals[0]) * 0.5
    mid_y = (min_vals[1] + max_vals[1]) * 0.5
    mid_z = (min_vals[2] + max_vals[2]) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # legend---------------
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    #----------------
    legend = ax.legend(
        unique_handles,
        unique_labels,
        title='Elements',
        loc='upper left',
        bbox_to_anchor=(1.0, 0.9),
        borderaxespad=0.,
        labelspacing=2.2,  # (default is 0.5)
        fontsize=14,
        ncol=len(unique_labels)
    )
    legend.get_frame().set_visible(False)
    #----------------
    
    ax.grid(False, linestyle=':', alpha=0.5)
    ax.view_init(elev=25, azim=45)

    # lattice parameters----------------
    a, b, c = parser.lattice_parameters

    #----------------
    info_text = f"Lattice parameters (Å):\n"
    info_text += f"  a = {a:.4f}\n"
    info_text += f"  b = {b:.4f}\n"
    info_text += f"  c = {c:.4f}\n\n"

    #-INFO-------------
    info_text += "Atoms in unit cell:\n"
    for atom in parser.reduced_coords:
        element, x, y, z = atom
        info_text += f"  {element:2s}: {x:8.6f} {y:8.6f} {z:8.6f}\n"

    fig.text(
        0.02, 0.5,
        info_text,
        fontsize=12,
        va='center',
        ha='left',
        family='monospace',
        bbox=dict(
            facecolor='whitesmoke',
            edgecolor='lightgray',
            boxstyle='round,pad=0.5'
        )
    )
    #----------------
    if not show_axes:
        ax.set_axis_off()
    #----------------
    plt.tight_layout()
    return fig, ax

##----------------------------------------------------
def plot_crystal_structure_cif(cell_params, atoms_frac, figsize=(10, 8), show_axes=False, azim=30, elev=20):
    # lattice parameters
    a = cell_params['_cell_length_a']
    b = cell_params['_cell_length_b']
    c_val = cell_params['_cell_length_c']
    alpha = np.deg2rad(cell_params['_cell_angle_alpha'])
    beta = np.deg2rad(cell_params['_cell_angle_beta'])
    gamma = np.deg2rad(cell_params['_cell_angle_gamma'])
    
    ###-----------------------------
    # Calculate lattice vectors
    a1 = np.array([a, 0, 0])
    a2 = np.array([0, b, 0])  # gamma = 90°
    cx = 0  # alpha = 90°
    cy = c_val * np.cos(alpha)  # beta = 108.016°
    cz = np.sqrt(c_val**2 - cy**2)
    a3 = np.array([cx, cy, cz])
    
    ###-----------------------------
    """ Transformation matrix for
        fractional to Cartesian coordinates
    """
    M_rows = np.vstack([a1, a2, a3])
    ###-----------------------------
    
    # atom positions to Cartesian coordinates
    atoms_cart = []
    for symbol, x, y, z in atoms_frac:
        frac_coord = np.array([x, y, z])
        cart_coord = np.dot(frac_coord, M_rows)
        atoms_cart.append((symbol, cart_coord))
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    #-----------------------------------------
    cpk_colors = get_cpk_colors()     ###>>>>>
    atomic_radii = get_atomic_radii() ###>>>>>
    #-----------------------------------------
    
    size_scale = 4.9
    
    for symbol, coord in atoms_cart:
        color = cpk_colors.get(symbol, '#FFFFFF')  # default == white
        radius = atomic_radii.get(symbol, 100)     # default radius == 100 if element not found
        size = radius * size_scale
        
        # slightly darker edge
        edge_color = to_rgba(color)
        edge_color = (max(0, edge_color[0]-0.2), 
                      max(0, edge_color[1]-0.2), 
                      max(0, edge_color[2]-0.2), 
                      1.0)
        
        ax.scatter(coord[0], coord[1], coord[2], 
                   s=size, color=color, edgecolor=edge_color, 
                   linewidth=0.5, depthshade=True, alpha=0.9)
    
    # unit cell corners
    corners_frac = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ])
    corners_cart = np.dot(corners_frac, M_rows)
   
    # unit cell boundaries
    edges = [
        [0, 1], [0, 2], [0, 3],
        [1, 4], [1, 5],
        [2, 4], [2, 6],
        [3, 5], [3, 6],
        [4, 7], [5, 7], [6, 7]
    ]

    for start, end in edges:
        xs = [corners_cart[start][0], corners_cart[end][0]]
        ys = [corners_cart[start][1], corners_cart[end][1]]
        zs = [corners_cart[start][2], corners_cart[end][2]]
        ax.plot(xs, ys, zs, 'k-', linewidth=1.2, alpha=0.7)
    
    # perspective and viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # axis styling
    if show_axes:
        ax.set_xlabel('X (Å)', fontsize=10, color='white')
        ax.set_ylabel('Y (Å)', fontsize=10, color='white')
        ax.set_zlabel('Z (Å)', fontsize=10, color='white')
        ax.tick_params(axis='both', colors='white', labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('dimgray')
        ax.yaxis.pane.set_edgecolor('dimgray')
        ax.zaxis.pane.set_edgecolor('dimgray')
        ax.grid(False)
    else:
        ax.set_axis_off()
    
    ax.set_title('Crystal Structure', fontsize=14, color='k', pad=20)
    plt.tight_layout()
    
    legend_elements = []
    seen_symbols = set()
    for symbol, _ in atoms_cart:
        if symbol not in seen_symbols:
            color = cpk_colors.get(symbol, '#FFFFFF')
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='k', 
                                             label=symbol,
                                             markerfacecolor=color, 
                                             markersize=8))
            seen_symbols.add(symbol)
    
    ax.legend(handles=legend_elements, loc='upper right', 
              frameon=True, facecolor='w', edgecolor='w',
              labelcolor='k', fontsize=10)

##----------------------------------------------------
def print_to_console(console: RichJupyterWidget, msg: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{msg}''')")

def CTLS_connector_parser(ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select parser.log file")
    if "parser.log" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the 'parser.log' file.")
        return

    if file_path:
        try:
            parser = CrystalStructureParser(file_path)
            parser.parse()

            fig, ax = plot_crystal_structure_parserlog(
                parser,
                atom_scale=1.2,
                figsize=(12, 6),
                show_axes=False
            )
            plt.show()
            
            a = parser.lattice_parameters[0]
            b = parser.lattice_parameters[1]
            c = parser.lattice_parameters[2]
            scaled_vectors = parser.scaled_vectors
            reduced_coords = parser.reduced_coords

            timestamp = datetime.now().strftime("[%H:%M:%S]")
            msg = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          Parser log!                          --\n"
                + "-" * 75 + "\n"
                ">>> Crystal Structure plotted successfully!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> Lattice parameters (Å):\n"
                f">>> a = {a:.4f}\n"
                f">>> b = {b:.4f}\n"
                f">>> c = {c:.4f}\n"
                f">>> Scaled lattice vectors (Å):\n"
                f">>> {scaled_vectors}\n"
                f">>> Atoms in unit cell:\n"
                f">>> {reduced_coords}\n"
                + "-" * 75
            )
            print_to_console(ipy_console, msg)

        except Exception as e:
            print(f"Error parsing parser.log file: {e}")
            QMessageBox.critical(None, "Error", f"Failed to parse parser.log file:\n{e}")

##----------------------------------------------------
def CTLS_connector_cif(ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select conventional standard CIF file")
    if ".cif" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the '.cif' file.")
        return

    if file_path:
        try:
            cell_params, atoms_frac = parse_cif(file_path)
            
            plot_crystal_structure_cif(cell_params, atoms_frac, 
                               azim=45, elev=25, show_axes=False)
            plt.show()
            
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            msg = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          CIF log!                          --\n"
                + "-" * 75 + "\n"
                ">>> Crystal Structure plotted successfully!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> Lattice parameters (Å):\n"
                f">>> {cell_params}\n"
                f">>> \n"
                f">>> Found {len(atoms_frac)} atoms\n"
                + "-" * 75
            )
            print_to_console(ipy_console, msg)

        except Exception as e:
            print(f"Error parsing cif file: {e}")
            QMessageBox.critical(None, "Error", f"Failed to parse cif file:\n{e}")

##----------------------------------------------------
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crystal Structure Parameters")
        self.setFixedSize(600, 300)
        self.setStyleSheet(self.get_modern_stylesheet())
        self.init_ui()
        
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
        ##self.create_optional_section(content_layout)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # button section
        self.create_button_section(main_layout)
        
    def create_header(self, layout):
        header_layout = QHBoxLayout()

        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon_label.setPixmap(icon.pixmap(32, 32))
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title = QLabel("Crystal Structure Visualization")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")
        
        subtitle = QLabel("Specify the input file type for plotting the crystal structure.")
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
        group_box = QGroupBox("Essential Parameters")
        group_layout = QVBoxLayout()

        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(150)

        self.data_type_parser = QCheckBox("parser.log")
        self.data_type_cif = QCheckBox("Conventional standard CIF")
    
        checkbox_layout.addWidget(self.data_type_parser)
        checkbox_layout.addWidget(self.data_type_cif)
        checkbox_layout.addStretch()
    
        group_layout.addLayout(checkbox_layout)
        group_box.setLayout(group_layout)
    
        layout.addWidget(group_box)
    
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
        submit_btn = QPushButton("Show crystal structure")
        submit_btn.setIcon(QIcon(":/icons/analytics_27dp_1F1F1F_FILL0_wght400_GRAD0_opsz24.png"))
        submit_btn.setIconSize(QSize(24, 24))
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self.on_submit)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)
    
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
            checked_boxes = sum([
                self.data_type_parser.isChecked(),
                self.data_type_cif.isChecked()
            ])

            if checked_boxes == 0:
                QMessageBox.warning(self, "Input Required", "Please select one data type (parser.log or CIF).")
                return
            elif checked_boxes > 1:
                QMessageBox.warning(self, "Only One Allowed", "Please select only one data type — not both.")
                return

            self.accept()

            if self.data_type_parser.isChecked():
                CTLS_connector_parser(self.parent().ipy_console)

            elif self.data_type_cif.isChecked():
                CTLS_connector_cif(self.parent().ipy_console)

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def input_dialog_CTLS(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()






