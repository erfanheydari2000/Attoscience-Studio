# find_MPW.py
import sys
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import webbrowser
import plotly.graph_objects as go
from matplotlib import gridspec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar,
                             QRadioButton, QButtonGroup, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QScrollArea, QSlider, QStyle, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton, QGraphicsOpacityEffect)
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

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from attoscience_studio.resources_rc import *
##----------------------------------------------------
def read_data_for_MPW(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.size == 0:
            raise ValueError("The file is empty.")
        t = data[:, 1]
        dt = t[1] - t[0]
        jx = data[:, 2] - data[0, 2]
        jy = data[:, 3] - data[0, 3]
        if np.all(jx == 0) and np.all(jy == 0):
            raise ValueError("Both jx and jy values are zero.")           
        return t, dt, jx, jy
    except Exception as e:
        raise ValueError(f"Failed to read field data: {e}")
##----------------------------------------------------
def find_MPW_core(t, dt, jx, jy, lambda0_nm, qstart, qmax):
    w0 = 45.5633 / lambda0_nm
    T = 2 * np.pi / w0    
    dw = 0.1
    Time_OC = t / T
    max_Time_OC = max(Time_OC)
    last_OC = max_Time_OC - 1

    def calculate_FWHM(qstart_inner, qmax_inner):  
    #Currently, only the second method is implemented 
        w = w0 * np.arange(qstart_inner, qmax_inner + dw, dw)
        ajx = np.zeros(len(w), dtype=complex)
        ajy = np.zeros(len(w), dtype=complex)

        for l in range(len(w)):
            gx = jx * np.exp(-1j * w[l] * t)
            ajx[l] = w[l] * np.trapz(gx, t)
            gy = jy * np.exp(-1j * w[l] * t)
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

        idx_tot = np.argmax(I)
        
        OC = Time_OC[idx_tot] #time corresponding to I_Max

        I_Max = max(I)
        half_I_tot = I_Max / 2

        left_idx_candidates_tot = np.where(I[:idx_tot] <= half_I_tot)[0]
        right_idx_candidates_tot = np.where(I[idx_tot:] <= half_I_tot)[0]

        if left_idx_candidates_tot.size > 0:
            left_idx_tot = left_idx_candidates_tot[-1]
        else:
            left_idx_tot = 0

        if right_idx_candidates_tot.size > 0:
            right_idx_tot = right_idx_candidates_tot[0] + idx_tot
        else:
            right_idx_tot = len(I) - 1

        t_left_tot = Time_OC[left_idx_tot]
        t_right_tot = Time_OC[right_idx_tot]
        FWHM_tot = t_right_tot - t_left_tot

        atomic_to_seconds = 2.4188843265857e-17
        FWHM_SI_tot = FWHM_tot * T * atomic_to_seconds
        FWHM_as_tot = FWHM_SI_tot * 1e18
        
        return FWHM_as_tot, OC

    def parallel_calculate(qstart_inner, qmax_inner):
        FWHM, OC = calculate_FWHM(qstart_inner, qmax_inner)
        return FWHM, OC, qstart_inner, qmax_inner
    n_jobs = 4
    results = Parallel(n_jobs=n_jobs)(
        delayed(parallel_calculate)(qstart_inner, qmax_inner)
        for qstart_inner in range(int(qstart), int(qmax - 1))
        for qmax_inner in range(int(qstart_inner + 1), int(qmax + 1))
    )
    
    min_result = min(results, key=lambda x: x[0])
    min_FWHM, OC, optimal_qstart, optimal_qmax = min_result

    return min_FWHM, optimal_qstart, optimal_qmax, OC, last_OC, max_Time_OC

##----------------------------------------------------

class MPWWorker(QObject):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(float, int, int, float, float, float)
    error = pyqtSignal(str)

    def __init__(self, lambda0_nm, qstart, qmax, t, dt, jx, jy):
        super().__init__()
        self.lambda0_nm = lambda0_nm
        self.qstart = qstart
        self.qmax = qmax
        self.t = t
        self.dt = dt
        self.jx = jx
        self.jy = jy

    def run(self):
        try:
            # CALL
            results = find_MPW_core(
                self.t, self.dt, self.jx, self.jy,
                self.lambda0_nm, self.qstart, self.qmax
            )
            self.finished.emit(*results)
        except Exception as e:
            self.error.emit(str(e))

##----------------------------------------------------

previous_input_FMPW = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MPW Parameters")
        self.setFixedSize(650, 450)
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
        
        # Header with icon and title
        self.create_header(main_layout)
        
        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Create content sections
        self.create_required_section(content_layout)
        #self.create_optional_section(content_layout)
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Modern button section
        self.create_button_section(main_layout)
        
    def create_header(self, layout):
        header_layout = QHBoxLayout()

        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
        icon_label.setPixmap(icon.pixmap(32, 32))  # Set size as needed

        title_layout = QVBoxLayout()
        title = QLabel("Find Minimum Pulse Width")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")

        subtitle = QLabel("Configure parameters for finding MPW from total current")
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

        ##- Basic Parameters Section -------------------------------------
        basic_params_group = QGroupBox("") #Basic Parameters#
        basic_params_layout = QFormLayout()
        basic_params_layout.setSpacing(6)
        basic_params_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # (1) Driving Wavelength
        lambda0_container = QWidget()
        lambda0_layout = QHBoxLayout(lambda0_container)
        lambda0_layout.setContentsMargins(0, 0, 0, 0)
    
        self.lambda0_entry = QLineEdit()
        self.lambda0_entry.setPlaceholderText("Enter driving wavelength (e.g., 2000)")
        self.lambda0_entry.setText(str(previous_input_FMPW.get("lambda0_nm", "")))
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
        self.qstart_entry.setPlaceholderText("Minimum harmonic order (e.g., 4)")
        self.qstart_entry.setText(str(previous_input_FMPW.get("qstart", "")))
        self.qstart_entry.setMaxLength(10)
        
        qstart_unit_label = QLabel("HO")
        qstart_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qstart_layout.addWidget(self.qstart_entry)
        qstart_layout.addWidget(qstart_unit_label)
    
        basic_params_layout.addRow("Minimum harmonic order:", qstart_container)

        # (3) Maximum Harmonic Order
        qmax_container = QWidget()
        qmax_layout = QHBoxLayout(qmax_container)
        qmax_layout.setContentsMargins(0, 0, 0, 0)
    
        self.qmax_entry = QLineEdit()
        self.qmax_entry.setPlaceholderText("Maximum harmonic order (e.g., 4)")
        self.qmax_entry.setText(str(previous_input_FMPW.get("qmax", "")))
        self.qmax_entry.setMaxLength(10)
        
        qmax_unit_label = QLabel("HO")
        qmax_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qmax_layout.addWidget(self.qmax_entry)
        qmax_layout.addWidget(qmax_unit_label)
    
        basic_params_layout.addRow("Maximum harmonic order:", qmax_container)
    
        #------
        basic_params_group.setLayout(basic_params_layout)
        required_layout.addWidget(basic_params_group)
        ##--------------------------------------------------------
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)

    def create_button_section(self, layout):
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
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
        
        submit_btn = QPushButton("Find MPW")
        submit_btn.setIcon(QIcon(":/icons/calculate_27dp_000000_FILL0_wght400_GRAD0_opsz24.png"))
        submit_btn.setIconSize(QSize(24, 24))
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self.on_submit)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)

    ##$$$$$$$$$$$$$$$$$$$$$$$$$
    
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
            previous_input_FMPW["lambda0_nm"] = lambda0_nm

            qstart = float(self.qstart_entry.text())
            previous_input_FMPW["qstart"] = qstart
            if qstart <= 0:
                raise ValueError("qstart must be a positive number.")

            qmax = float(self.qmax_entry.text())
            previous_input_FMPW["qmax"] = qmax
            if qmax <= qstart:
                raise ValueError("Maximum HO must be greater than Minimum HO.")

            # Update
            previous_input_FMPW.update({"lambda0_nm": lambda0_nm, "qstart": qstart,"qmax": qmax })

            self.accept()


            file_path, _ = QFileDialog.getOpenFileName(None, "Select total_current file")
            if not file_path:
                return
            if "total_current" not in file_path.lower():
                QMessageBox.warning(dialog, "Error", "Please select total_current file")
                return

            try:
                t, dt, jx, jy = read_data_for_MPW(file_path)
            except Exception as e:
                QMessageBox.warning(dialog, "Error", f"Failed to read file: {str(e)}")
                return


            self.parent().start_mpw_computation(
                lambda0_nm=lambda0_nm,
                qstart=qstart,
                qmax=qmax,
                t=t,
                dt=dt,
                jx=jx,
                jy=jy
            )

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def input_dialog_FMPW(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()   



