# electron_dynamics/nex_anim.py

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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.io as sio
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox,
                             QLineEdit, QPushButton, QGroupBox, QLabel, QFrame, QStyle,
                             QFileDialog, QMessageBox, QCheckBox, QSpinBox, QButtonGroup,
                             QDoubleSpinBox, QProgressBar, QApplication, QWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont
from attoscience_studio.resources_rc import *
##----------------------------------------------------
previous_input_current_nex = {}
class CurrentNexAnalysisThread(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    analysis_completed = pyqtSignal(object)  # Will emit the figure object
    error_occurred = pyqtSignal(str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            self.status_updated.emit("Loading laser data...")
            self.progress_updated.emit(10)
           
            ##============================================
            # Extract parameters -------------------------
            base_dir_laser = self.params['base_dir_laser']
            base_dir_iter  = self.params['base_dir_iter']
            lambda0_nm     = self.params['lambda0_nm']
            A              = self.params['A']
            
            num_interpolated_frames = self.params['num_interpolated_frames']
 
            save_animation = self.params['save_animation']
            save_format = self.params['format']
            save_dir = self.params['save_dir']
            
            # Load laser data ---------------------------
            w0 = 45.5633 / lambda0_nm
            T = 2 * np.pi / w0
            file_path = os.path.join(base_dir_laser, 'laser')
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Laser file not found: {file_path}")
                
            laser_data = np.loadtxt(file_path)
            Time = laser_data[:, 1] 
            Ax1, Ay1 = laser_data[:, 2], laser_data[:, 3]
            Ax2, Ay2 = laser_data[:, 5], laser_data[:, 6]
            AX = Ax1 + Ax2
            AY = Ay1 + Ay2
            
            self.status_updated.emit("Processing k-point directories...")
            self.progress_updated.emit(20)
            
            # Get time directories -------------------
            time_dirs = sorted([name for name in os.listdir(base_dir_iter) 
                              if name.startswith('td.')])
            
            if not time_dirs:
                raise ValueError("No td.* directories found in the iteration directory")
            
            file_name_X = 'current_kpt-x.kz=0'
            file_name_Y = 'current_kpt-y.kz=0'
            file_name_nex = 'n_excited_el_kpt.kz=0'
            
            # Initialize grid -------------------------
            first_file = os.path.join(base_dir_iter, time_dirs[0], file_name_X)
            if not os.path.exists(first_file):
                raise FileNotFoundError(f"First k-point file not found: {first_file}")
                
            kpt_data = np.loadtxt(first_file, skiprows=1)
            k_x, k_y = kpt_data[:, 0], kpt_data[:, 1]
            x_grid = np.linspace(np.min(k_x), np.max(k_x), A)
            y_grid = np.linspace(np.min(k_y), np.max(k_y), A)
            x_grid, y_grid = np.meshgrid(x_grid, y_grid)
            
            self.status_updated.emit("Loading and interpolating current data...")
            self.progress_updated.emit(30)
            
            # Process data ---------------------------
            interpolated_data_currents = {}
            interpolated_data_nex = {}
            
            total_dirs = len(time_dirs)
            for i, time_dir in enumerate(time_dirs):
                # Current data ====================
                file_path_X = os.path.join(base_dir_iter, time_dir, file_name_X)
                file_path_Y = os.path.join(base_dir_iter, time_dir, file_name_Y)
                
                if os.path.exists(file_path_X) and os.path.exists(file_path_Y):
                    interpolated_currents = self.load_and_interpolate_data_curr(
                        file_path_X, file_path_Y, x_grid, y_grid)
                    interpolated_data_currents[time_dir] = interpolated_currents
                
                # Nex data ========================
                file_path_nex = os.path.join(base_dir_iter, time_dir, file_name_nex)
                if os.path.exists(file_path_nex):
                    interpolated_nex = self.load_and_interpolate_data_nex(
                        file_path_nex, x_grid, y_grid)
                    interpolated_data_nex[time_dir] = interpolated_nex
                
                progress = 30 + int((i / total_dirs) * 40)
                self.progress_updated.emit(progress)
            
            self.status_updated.emit("Creating interpolated frames...")
            self.progress_updated.emit(70)
            
            # Create arrays and interpolated frames ----------------
            arrays_curr = [interpolated_data_currents[time_dir] for time_dir in time_dirs 
                          if time_dir in interpolated_data_currents]
            arrays_nex = [interpolated_data_nex[time_dir] for time_dir in time_dirs 
                         if time_dir in interpolated_data_nex]
            
            smooth_frames_curr = self.interpolate_frames(arrays_curr, num_interpolated_frames)
            smooth_frames_nex = self.interpolate_frames(arrays_nex, num_interpolated_frames)
            
            # Interpolate time and vector potential ----------------
            target_length = len(smooth_frames_curr)
            interpolated_Time, interpolated_AX, interpolated_AY = self.interpolate_TIME_and_AX_and_AY(Time, AX, AY, target_length)
            interpolated_Time = interpolated_Time / T
            
            self.status_updated.emit("Creating visualization...")
            self.progress_updated.emit(90)
            
            #-------------------------------------------------------
            # Create the figure
            fig = self.create_figure(k_x, k_y, interpolated_Time, interpolated_AX, interpolated_AY, smooth_frames_curr, smooth_frames_nex) ###>>>>>>>>>>>>>>>>>>

            fig.animation_data['save_animation'] = save_animation
            fig.animation_data['format'] = save_format
            fig.animation_data['save_dir'] = save_dir

            #-------------------------------------------------------

            self.status_updated.emit("Analysis completed!")
            self.progress_updated.emit(100)
            self.analysis_completed.emit(fig)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    #=========================================================================
    def load_and_interpolate_data_curr(self, file_path_X, file_path_Y, x_grid, y_grid):
        raw_data_X = np.loadtxt(file_path_X, skiprows=1)
        raw_data_Y = np.loadtxt(file_path_Y, skiprows=1)
        
        kx, ky = raw_data_X[:, 0], raw_data_Y[:, 1]
        currents_X = raw_data_X[:, 2]
        currents_Y = raw_data_Y[:, 2]
        currents = np.sqrt(currents_X**2 + currents_Y**2)

        interpolated_currents = griddata(
            points=(kx, ky),
            values=currents,
            xi=(x_grid, y_grid),
            method='cubic'
        )
        if np.isnan(interpolated_currents).any():
            interpolated_currents = np.nan_to_num(interpolated_currents, nan=0.0)
        return interpolated_currents
    
    def load_and_interpolate_data_nex(self, file_path_nex, x_grid, y_grid):
        raw_data_nex = np.loadtxt(file_path_nex, skiprows=1)
        kx, ky, nex = raw_data_nex[:, 0], raw_data_nex[:, 1], raw_data_nex[:, 2]

        interpolated_nex = griddata(
            points=(kx, ky),
            values=nex,
            xi=(x_grid, y_grid),
            method='cubic'
        )

        if np.isnan(interpolated_nex).any():
            interpolated_nex = np.nan_to_num(interpolated_nex, nan=0.0)
        return interpolated_nex
    
    def interpolate_frames(self, arrays, num_interpolated_frames):
        num_arrays = len(arrays)
        interpolated_frames = []
        
        for i in range(num_arrays - 1):
            frame1 = arrays[i]
            frame2 = arrays[i + 1]

            for t in np.linspace(0, 1, num_interpolated_frames, endpoint=False):
                interpolated_frame = frame1 * (1 - t) + frame2 * t
                interpolated_frames.append(interpolated_frame)

        interpolated_frames.append(arrays[-1])
        return interpolated_frames
    
    def interpolate_TIME_and_AX_and_AY(self, Time, AX, AY, target_length):
        original_indices = np.linspace(0, len(Time) - 1, len(Time))
        new_indices = np.linspace(0, len(Time) - 1, target_length)
        
        interpolated_Time = np.interp(new_indices, original_indices, Time)
        interpolated_AX = np.interp(new_indices, original_indices, AX)
        interpolated_AY = np.interp(new_indices, original_indices, AY)
        
        return interpolated_Time, interpolated_AX, interpolated_AY
   
    #=========================================================================
    
    def create_figure(self, k_x, k_y, interpolated_Time, interpolated_AX, interpolated_AY, smooth_frames_curr, smooth_frames_nex):
        fig = plt.figure(figsize=(12, 8))
        extent = (np.min(k_x), np.max(k_x), np.min(k_y), np.max(k_y))

        # Upper subplot (3D)
        ax1 = fig.add_axes([0.05, 0.55, 0.9, 0.45], projection='3d') # [left, bottom, width, height]
        ax1.plot(interpolated_Time, interpolated_AX, interpolated_AY, 'k', linewidth=3.5)

        ax1.set_xlabel('Time [o.c.]')
        ax1.set_ylabel(r'$\mathregular{A_x\ [a.u.]}$')
        ax1.set_zlabel(r'$\mathregular{A_y\ [a.u.]}$')
        ax1.set_xlim([0, max(interpolated_Time)])
        ax1.set_box_aspect([1.5, 1, 1])
        
        marker, = ax1.plot([], [], [], 'ro', markersize=10, mec='b')
        coordinate_text = fig.text(0.2, 0.75, '', fontsize=12, color='black', ha='center', va='center')
        
        ax1.view_init(elev=30, azim=-60)
        ax1.grid(False)

        # Lower left subplot (Current)
        ax2 = fig.add_axes([0.1, 0.1, 0.35, 0.35])
        img1 = ax2.imshow(smooth_frames_curr[0], extent=extent, origin='lower', 
                         cmap='jet', aspect='equal')
        cbar1 = plt.colorbar(img1, ax=ax2, label='Current', pad=0.02)
        ax2.set_xlabel(r'$\mathregular{K_x\ [2\pi/a]}$')
        ax2.set_ylabel(r'$\mathregular{K_y\ [2\pi/a]}$')

        # Lower right subplot (Nex)
        ax3 = fig.add_axes([0.6, 0.1, 0.35, 0.35])
        img2 = ax3.imshow(smooth_frames_nex[0], extent=extent, origin='lower', 
                         cmap='jet', aspect='equal')
        cbar2 = plt.colorbar(img2, ax=ax3, label='Nex', pad=0.02)
        ax3.set_xlabel(r'$\mathregular{K_x\ [2\pi/a]}$')
        ax3.set_ylabel(r'$\mathregular{K_y\ [2\pi/a]}$')
        
        # Store data for animation
        fig.animation_data = {
            'interpolated_Time': interpolated_Time,
            'interpolated_AX': interpolated_AX,
            'interpolated_AY': interpolated_AY,
            'smooth_frames_curr': smooth_frames_curr,
            'smooth_frames_nex': smooth_frames_nex,
            'k_x': k_x,
            'k_y': k_y,
            'marker': marker,
            'coordinate_text': coordinate_text,
            'img1': img1,
            'img2': img2,
            'cbar1': cbar1,
            'cbar2': cbar2
        }

        return fig

##----------------------------------------------------
class CurrentNexDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Current and Excited Electrons k-space Analysis")
        self.setFixedSize(1000, 900)
        self.setStyleSheet(self.get_modern_stylesheet())
        
        self.analysis_thread = None
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
        
        QPushButton:disabled {
            background: #cccccc;
            color: #666666;
        }
        
        QSpinBox, QDoubleSpinBox {
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
            background: white;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #2196f3;
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
        
        QProgressBar {
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            text-align: center;
            background: white;
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                      stop: 0 #4caf50, stop: 1 #388e3c);
            border-radius: 4px;
        }
        """
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        self.create_header(main_layout)
        
        # Path Configuration
        self.create_path_section(main_layout)
        
        # Parameters
        self.create_parameters_section(main_layout)
        
        # Options
        self.create_options_section(main_layout)
        
        # Progress
        self.create_progress_section(main_layout)
        
        # Buttons
        self.create_button_section(main_layout)


    def create_header(self, layout):
        header_layout = QHBoxLayout()

        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon_label.setPixmap(icon.pixmap(32, 32))
        icon_label.setFixedSize(36, 36)


        # Title and subtitle
        title_layout = QVBoxLayout()
        title = QLabel("Current and Excited Electrons k-space Analysis")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")
        
        subtitle = QLabel("Analyze current density and excited electrons in k-space")
        subtitle.setStyleSheet("font-size: 14px; color: #666; margin: 0;")
        subtitle.setWordWrap(True)
        
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
    
    def create_path_section(self, layout):
        path_group = QGroupBox("Data Directories")
        path_layout = QFormLayout()
        path_layout.setSpacing(12)
        
        # Laser directory
        laser_layout = QHBoxLayout()
        self.laser_dir_entry = QLineEdit()
        self.laser_dir_entry.setPlaceholderText("Select laser data directory (contains 'laser' file)")
        self.laser_dir_entry.setText(previous_input_current_nex.get("base_dir_laser", ""))
        
        laser_browse_btn = QPushButton("Browse")
        laser_browse_btn.clicked.connect(self.browse_laser_directory)
        
        laser_layout.addWidget(self.laser_dir_entry)
        laser_layout.addWidget(laser_browse_btn)
        path_layout.addRow("Laser Directory:", laser_layout)
        
        # Iteration directory
        iter_layout = QHBoxLayout()
        self.iter_dir_entry = QLineEdit()
        self.iter_dir_entry.setPlaceholderText("Select iteration directory (contains td.* subdirectories)")
        self.iter_dir_entry.setText(previous_input_current_nex.get("base_dir_iter", ""))
        
        iter_browse_btn = QPushButton("Browse")
        iter_browse_btn.clicked.connect(self.browse_iter_directory)
        
        iter_layout.addWidget(self.iter_dir_entry)
        iter_layout.addWidget(iter_browse_btn)
        path_layout.addRow("Iteration Directory:", iter_layout)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
    
    def create_parameters_section(self, layout):
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QHBoxLayout()
        params_layout.setSpacing(24)

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

        # Wavelength
        wavelength_layout = QVBoxLayout()
        wavelength_label = QLabel("Driving Wavelength:")
        self.wavelength_spinbox = QDoubleSpinBox()
        self.wavelength_spinbox.setRange(100, 10000)
        self.wavelength_spinbox.setValue(previous_input_current_nex.get("lambda0_nm", 3000))
        self.wavelength_spinbox.setSuffix(" nm")
        self.wavelength_spinbox.setDecimals(1)
        
        self.wavelength_spinbox.setStyleSheet(light_blue_style)
        
        wavelength_layout.addWidget(wavelength_label)
        wavelength_layout.addWidget(self.wavelength_spinbox)

        # Grid resolution
        grid_layout = QVBoxLayout()
        grid_label = QLabel("Grid Resolution:")
        self.grid_spinbox = QSpinBox()
        self.grid_spinbox.setRange(1, 1000)
        self.grid_spinbox.setValue(previous_input_current_nex.get("A", 50))
        self.grid_spinbox.setSuffix(" points")
        
        self.grid_spinbox.setStyleSheet(light_blue_style)
        
        grid_layout.addWidget(grid_label)
        grid_layout.addWidget(self.grid_spinbox)

        # Interpolated frames
        interp_layout = QVBoxLayout()
        interp_label = QLabel("Interpolated Frames:")
        self.interp_spinbox = QSpinBox()
        self.interp_spinbox.setRange(1, 100)
        self.interp_spinbox.setValue(previous_input_current_nex.get("num_interpolated_frames", 16))
        self.interp_spinbox.setSuffix(" frames")
        
        self.interp_spinbox.setStyleSheet(light_blue_style)
        
        interp_layout.addWidget(interp_label)
        interp_layout.addWidget(self.interp_spinbox)

        # Add to the horizontal layout
        params_layout.addLayout(wavelength_layout)
        params_layout.addLayout(grid_layout)
        params_layout.addLayout(interp_layout)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

    def create_options_section(self, layout):
        options_group = QGroupBox("Create Animation")
        options_layout = QHBoxLayout()
        ###------------------------------------
        
        # Save checkbox
        self.save_animation_checkbox = QCheckBox("Save")
        self.save_animation_checkbox.stateChanged.connect(self.toggle_format_enabled)
        options_layout.addWidget(self.save_animation_checkbox)
        ###------------
        
        # combo box for formats
        self.format_combo = QComboBox()
        self.format_combo.setEnabled(self.save_animation_checkbox.isChecked()) ###>>>>>>>>>>>>> CALL

        formats = [
            ("mp4", ":/icons/mp4.png"),
            ("avi", ":/icons/avi-file-icon.png"),
            ("vebm", ":/icons/webm.png"),
            ("gif", ":/icons/gif-file-icon.png"),
            ("mkv", ":/icons/mkv.png")
        ]

        for name, icon_path in formats:
            self.format_combo.addItem(QIcon(icon_path), name)

        # default ---> mp4
        default_format = previous_input_current_nex.get("animation_format", "MP4")
        index = self.format_combo.findText(default_format.upper())
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        
        options_layout.addWidget(self.format_combo)
        
        ###------------
        # --- Save directory section ---
        save_dir_container = QWidget()
        save_dir_layout = QHBoxLayout()
        save_dir_layout.setContentsMargins(0, 0, 0, 0)

        self.save_dir_entry = QLineEdit()
        self.save_dir_entry.setEnabled(self.save_animation_checkbox.isChecked())
        self.save_dir_entry.setPlaceholderText("Select save directory")

        self.save_browse_btn = QPushButton("Browse")
        self.save_browse_btn.setEnabled(self.save_animation_checkbox.isChecked())
        self.save_browse_btn.clicked.connect(self.browse_save_directory)

        save_dir_layout.addWidget(self.save_dir_entry)
        save_dir_layout.addWidget(self.save_browse_btn)
        save_dir_container.setLayout(save_dir_layout)

        options_layout.addWidget(QLabel("Save Directory"))
        options_layout.addWidget(save_dir_container)

        ###------------------------------------
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)


    def create_progress_section(self, layout):
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to start analysis...")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
    
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
        
        # Analyze button
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setIcon(QIcon(":/icons/animation.png"))
        self.analyze_btn.setDefault(True)
        self.analyze_btn.clicked.connect(self.start_analysis)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.analyze_btn)
        
        layout.addLayout(button_layout)
    
    #-- Helpers -----------------------------------------------
    def browse_laser_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Laser Data Directory")
        if directory:
            self.laser_dir_entry.setText(directory)
    #+++++++++++
    def browse_iter_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Iteration Directory")
        if directory:
            self.iter_dir_entry.setText(directory)
    #+++++++++++
    def browse_save_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select save Directory")
        if directory:
            self.save_dir_entry.setText(directory)
    #def toggle_format_enabled(self):
    #    self.format_combo.setEnabled(self.save_animation_checkbox.isChecked())

    def toggle_format_enabled(self, state):
        enabled = (state == Qt.Checked)
        self.format_combo.setEnabled(enabled)
        self.save_dir_entry.setEnabled(enabled)
        self.save_browse_btn.setEnabled(enabled)

    ## ---------------------------------------------------------
    
    def start_analysis(self):
        try:
            laser_dir = self.laser_dir_entry.text().strip()
            iter_dir = self.iter_dir_entry.text().strip()
            
            if not laser_dir or not os.path.exists(laser_dir):
                raise ValueError("Please select a valid laser data directory")
            
            if not iter_dir or not os.path.exists(iter_dir):
                raise ValueError("Please select a valid iteration directory")
            
            # if laser file exists
            laser_file = os.path.join(laser_dir, 'laser')
            if not os.path.exists(laser_file):
                raise ValueError("Laser file not found in the selected directory")

            params = {
                'base_dir_laser': laser_dir,
                'base_dir_iter' : iter_dir,
                'lambda0_nm': self.wavelength_spinbox.value(),
                'A': self.grid_spinbox.value(),
                
                'num_interpolated_frames': self.interp_spinbox.value(),
            }
            
            save_animation = self.save_animation_checkbox.isChecked()
            
            if save_animation:
                params['save_animation'] = True
                params['format'] = self.format_combo.currentText()
                params['save_dir'] = self.save_dir_entry.text().strip()
                print("params['save_dir']", params['save_dir'])

                # Validate save directory
                if not params['save_dir']:
                    raise ValueError("Please select a save directory")
                if not os.path.exists(params['save_dir']):
                    os.makedirs(params['save_dir'], exist_ok=True)
            else:
                params['save_animation'] = False
                params['format'] = 'None'
                params['save_dir'] = 'None'

            # Update
            previous_input_current_nex.update(params)
            
            # Start analysis thread
            self.analysis_thread = CurrentNexAnalysisThread(params) #>>>>>>>>>>>>>>>>>>>>>>>
            self.analysis_thread.progress_updated.connect(self.update_progress)
            self.analysis_thread.status_updated.connect(self.update_status)
            self.analysis_thread.analysis_completed.connect(self.on_analysis_completed)
            self.analysis_thread.error_occurred.connect(self.on_error)
            
            # Update UI
            self.analyze_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.analysis_thread.start()
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def on_analysis_completed(self, figure):
        self.result_figure = figure
        self.accept()
    
    def on_error(self, error_message):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n\n{error_message}")


def create_current_nex_analysis(parent):
    dialog = CurrentNexDialog(parent)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.result_figure
    return None


