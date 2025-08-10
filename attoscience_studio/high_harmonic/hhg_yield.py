# hihg_harmonic/hhg_yield.py

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
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar, QStyle,
                             QRadioButton, QButtonGroup, QScrollArea, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QDoubleSpinBox, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton, QScrollArea, QSlider, QSizePolicy, QGraphicsOpacityEffect)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QPalette, QIcon, QPixmap, QPainter, QColor, QBrush, QIntValidator
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication, QPropertyAnimation, QEasingCurve, pyqtProperty
from numpy import trapz
from scipy.integrate import trapezoid, quad
from scipy.interpolate import griddata
from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from attoscience_studio.resources_rc import *
#--------------------------------
from attoscience_studio.utils.window_func import TotalCurrentFilter
#--------------------------------
from attoscience_studio.helper_functions.constants import PhysicalConstants, AtomicUnits
Ip_HeV = PhysicalConstants.Ip_HeV
##----------------------------------------------------
def read_dtat_file(file_path):
    try:
        data = np.loadtxt(file_path)
        if data.size == 0:
            raise ValueError("The file is empty.")
        t  = data[:, 1]
        jx = data[:, 2]
        jx = jx - jx[0]
        jy = data[:, 3]
        jy = jy - jy[0]
        if np.all(t == 0) and np.all(jx == 0) and np.all(jy == 0):
            raise ValueError("The first, second, and third columns are zero.")   
        return t, jx,jy
        
    except Exception as e:
        raise ValueError(f"Failed to read field data: {e}")

def calcu_YIELD(t, jx, jy, lambda0_nm, filtering, qstart, qend, time_derivative, selected_yields, window_func):
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

    w0 = 45.5633 / lambda0_nm
    T = 2 * np.pi / w0
    dw = 0.001
    wmin = qstart * w0
    wmax = qend * w0
    w_HH = np.arange(wmin, wmax + dw, dw)

    if time_derivative == 'True':
        dhx = np.gradient(hx, t)
        dhy = np.gradient(hy, t)
    elif time_derivative == 'False':
        dhx = hx
        dhy = hy
    
    Nomeg = len(w_HH)
    Dx = np.zeros(Nomeg, dtype=complex)
    Dy = np.zeros(Nomeg, dtype=complex)
    Ttot = 1
    for m in range(Nomeg):
        yx = np.exp(1j * w_HH[m] * t) * dhx
        Dx[m] = trapezoid(yx, t) / Ttot
        yy = np.exp(1j * w_HH[m] * t) * dhy
        Dy[m] = trapezoid(yy, t) / Ttot

    Sx_r = w_HH**2 * np.abs(Dx)**2
    Sx_r[Sx_r <= 0] = 1e-16
    Sx = np.log10(Sx_r)

    Sy_r = w_HH**2 * np.abs(Dy)**2
    Sy_r[Sy_r <= 0] = 1e-16
    Sy = np.log10(Sy_r)
   
    S_r = w_HH**2 * np.abs((Dx) + (Dy))**2
    S_r[S_r <= 0] = 1e-16
    S = np.log10(S_r) 
    
    ww = (w_HH / w0)
    #------------
    messages = []
    if 'total' in selected_yields:
        tot_yield = np.zeros(Nomeg)
        for i in range(Nomeg):
            tot_yield[i] = trapezoid(S_r, ww)
        messages.append(
            f'total yield = {np.sum(tot_yield):.2e}')
    if 'x' in selected_yields:        
        x_yield = np.zeros(Nomeg)    
        for j in range(Nomeg):
            x_yield[j] = trapezoid(Sx_r, ww)
        messages.append(f'x yield = {np.sum(x_yield):.2e}')

    if 'y' in selected_yields:        
        y_yield = np.zeros(Nomeg)    
        for k in range(Nomeg):
            y_yield[k] = trapezoid(Sy_r, ww)
        messages.append(f'y yield = {np.sum(y_yield):.2e}')
        
    return w0, T, Sx, Sy, S, ww, messages

##----------------------------------------------------
def render_latex_formula_to_pixmap(latex_str, dpi=150):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex_str}$", fontsize=12)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp_file.name, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return QPixmap(tmp_file.name)

class ModernMessageDialog(QDialog):
    def __init__(self, title="HHG Yield Results", message="", details="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(700, 500)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                border-radius: 12px;
            }
            QLabel {
                color: #2c3e50;
                font-weight: 500;
            }
            QTextEdit {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
                color: #495057;
                selection-background-color: #007bff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007bff, stop:1 #0056b3);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0056b3, stop:1 #004085);
            }
            QPushButton:pressed {
                background: #004085;
            }
            QFrame#separator {
                background-color: #dee2e6;
                max-height: 1px;
                border: none;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        self.setup_ui(message, details)
        self.setup_animation()
        
    def setup_ui(self, message, details):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Icon (create a simple success icon)
        icon_label = QLabel()
        icon_pixmap = self.create_success_icon()
        icon_label.setPixmap(icon_pixmap)
        icon_label.setFixedSize(48, 48)
        header_layout.addWidget(icon_label)
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        
        title_label = QLabel("HHG Yield Calculation Complete")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 0;")
        
        subtitle_label = QLabel("Results generated successfully")
        subtitle_label.setFont(QFont("Segoe UI", 10))
        subtitle_label.setStyleSheet("color: #6c757d; margin: 0;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Separator
        separator = QFrame()
        separator.setObjectName("separator")
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)
        
        # Content area with scroll
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        #-------------------------------------
        # Message display
        if message:
            message_label = QLabel("Summary:")
            message_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
            message_label.setStyleSheet("color: #495057; margin-bottom: 8px;")
            scroll_layout.addWidget(message_label)
            
            message_text = QTextEdit()
            #message_text.setPlainText(message)
            #- CALL -----------------
            formatted_message = self.format_yield_message_html(message)
            message_text.setHtml(formatted_message)
            #------------------------
            message_text.setMinimumHeight(140)
            message_text.setReadOnly(True)
            scroll_layout.addWidget(message_text)
                
        #-------------------------------------
        # --- Label "Formula"
        formula_title_label = QLabel("Formula:")
        formula_title_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        formula_title_label.setStyleSheet("color: #495057; margin: 16px 0 8px 0;")
        scroll_layout.addWidget(formula_title_label)

        # --- Rendered formula image
        formula_image_label = QLabel()
        #------------
        latex_str = r"I_{\mathrm{HH},i}(n) = \int_{(n-1)\omega}^{(n+1)\omega} \mathrm{HHG}_i(\omega') \, d\omega'"
        #------------
        #- CALL -----
        formula_pixmap = render_latex_formula_to_pixmap(latex_str)
        #------------
        formula_image_label.setPixmap(formula_pixmap)
        formula_image_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(formula_image_label)
        #-------------------------------------

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Button section
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.setFixedSize(100, 36)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    #-------------------------------------
    def format_yield_message_html(self, raw_message):
        import re
        html_message = raw_message.replace("\n", "<br>")
        html_message = re.sub(
            r"(\d\.\d+e[+-]?\d+|\d\.\d+)",
            r"<span style='color:#d63384; font-weight:bold; font-size:12px;'>\1</span>",
            html_message
        )
        html_message = html_message.replace(
            "Yield Results:",
            "<span style='font-weight:600; font-size:12px; color:#212529;'>Yield Results:</span>"
        )
        return f"<div style='font-family:Consolas; font-size:12px; color:#495057;'>{html_message}</div>"
    #-------------------------------------

    def create_success_icon(self):
        pixmap = QPixmap(48, 48)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setBrush(QBrush(QColor("#28a745")))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, 40, 40)
        
        painter.setPen(Qt.white)
        painter.setBrush(Qt.NoBrush)
        pen = painter.pen()
        pen.setWidth(3)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        
        painter.drawLine(15, 24, 21, 30)
        painter.drawLine(21, 30, 33, 18)
        
        painter.end()
        return pixmap
        
    def setup_animation(self):
        self.setWindowOpacity(0.0)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(0)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        QTimer.singleShot(0, self.animation.start)  # No Wasting time

def show_modern_message(title="HHG Yield Results", message="", details="", parent=None): 
    #<<<<<<<    We don't need to show "details" ----> ipy_console   >>>>>>>>>>
    dialog = ModernMessageDialog(title, message, details, parent)
    return dialog.exec_() == QDialog.Accepted

####################################################################
def print_to_console(console: RichJupyterWidget, detailed_log: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{detailed_log}''')")

def yield_connector(lambda0_nm, filtering, qstart, qend, time_derivative, selected_yields, window_func, ipy_console=None):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select total_current file")
    if "total_current" not in file_path.lower():
        QMessageBox.warning(None, "File Error", "Please upload the 'total_current' file.")
        return    
    
    if file_path:
        try:
            t, jx, jy = read_dtat_file(file_path)
            w0, T, Sx, Sy, S, ww, messages = calcu_YIELD(t, jx, jy, lambda0_nm, filtering, qstart, qend, time_derivative, selected_yields, window_func)

            timestamp = datetime.now().strftime("[%H:%M:%S]")
            summary_msg = f"Calculation completed at {timestamp}\n"
            summary_msg += f"File: {file_path.split('/')[-1]}\n"
            summary_msg += f"Frequency range: {qstart} to {qend}\n"
            summary_msg += f"Filtering: {filtering}%\n"
            summary_msg += "\nYield Results:\n" + "\n".join(messages)

            detailed_log = (
                f">>> Time                          {timestamp}\n"
                + "-" * 75 + "\n"
                + "--                          HHG Yield log!                          --\n"
                + "-" * 75 + "\n"
                ">>>  HHG for yield plotted successfully!\n"
                f">>> File loaded from: {file_path}\n"
                f">>> Lambda0: {lambda0_nm} nm\n"
                f">>> Fundamental frequency (w0): {w0:.4f}\n"
                f">>> Period (T): {T:.6f}\n"
                f">>> Filtering: {filtering}%\n"
                f">>> Window function: {window_func[0]} (param: {window_func[1]})\n"
                f">>> Time derivative: {time_derivative}\n"
                f">>> Frequency range: {qstart} to {qend}\n"
                f">>> Selected yields: {', '.join(selected_yields)}\n"
                f">>> Yields: {messages}\n"
                f">>> qstart: {qstart}\n"
                f">>> qend: {qend}\n"
                + "-" * 75
            )
            if ipy_console:
                print_to_console(ipy_console, detailed_log)

            show_modern_message(title="HHG Yield Results",message=summary_msg,details=detailed_log)

        except ValueError as e:
            QMessageBox.warning(None, "Error", str(e))
            return
#########################################################
previous_input_YIELD = {}
class ModernDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HHG yield Parameters")
        self.setFixedSize(1000, 800)
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
        #self.create_optional_section(content_layout)
        
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
        title = QLabel("Hihg Harmonic Yield")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1976d2; margin: 0;")
        
        subtitle = QLabel("Configure parameters for hihg harmonic yield calculation from total current")
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

        ##- Basic Parameters Section ----
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
        self.lambda0_entry.setText(str(previous_input_YIELD.get("lambda0_nm", "")))
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
        self.qstart_entry.setText(str(previous_input_YIELD.get("qstart", "")))
        self.qstart_entry.setMaxLength(10)
        
        qstart_unit_label = QLabel("")
        qstart_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qstart_layout.addWidget(self.qstart_entry)
        qstart_layout.addWidget(qstart_unit_label)
    
        basic_params_layout.addRow("Minimum harmonic order:", qstart_container)

        # (3) Minimum Harmonic Order
        qend_container = QWidget()
        qend_layout = QHBoxLayout(qend_container)
        qend_layout.setContentsMargins(0, 0, 0, 0)
    
        self.qend_entry = QLineEdit()
        self.qend_entry.setPlaceholderText("Maximum harmonic order (e.g., 4)")
        self.qend_entry.setText(str(previous_input_YIELD.get("qend", "")))
        self.qend_entry.setMaxLength(10)
        
        qend_unit_label = QLabel("")
        qend_unit_label.setStyleSheet("color: #666; font-style: italic; min-width: 30px;")
        
        qend_layout.addWidget(self.qend_entry)
        qend_layout.addWidget(qend_unit_label)
    
        basic_params_layout.addRow("Maximum harmonic order:", qend_container)
    
        # (4) Time Derivative
        derivative_container = QWidget()
        derivative_layout = QHBoxLayout(derivative_container)
        derivative_layout.setContentsMargins(0, 0, 0, 10)
    
        self.derivative_entry = QComboBox()
        self.derivative_entry.addItems(["True", "False"])
        self.derivative_entry.setFixedSize(120, 30)
        self.derivative_entry.setStyleSheet("""QComboBox { min-height: 30px;font-size: 12px;}""")

        label = "dJ/dt"
        derivative_spacer = QLabel(label)
        derivative_spacer.setStyleSheet("min-width: 30px;")
    
        derivative_layout.addWidget(self.derivative_entry)
        derivative_layout.addWidget(derivative_spacer)
    
        basic_params_layout.addRow("Time Derivative:", derivative_container)
        #------
        basic_params_group.setLayout(basic_params_layout)
        required_layout.addWidget(basic_params_group)

        ##- Display Options Section -----
        display_options_group = QGroupBox("Display Options")
        display_options_layout = QVBoxLayout()
        display_options_layout.setSpacing(15)

        # Components selection
        components_subgroup = QGroupBox("Components")
        components_layout = QHBoxLayout()
        components_layout.setContentsMargins(10, 0, 0, 10)
        components_layout.setSpacing(15)
    
        self.x_yield_checkbox = QCheckBox("X Component")
        self.y_yield_checkbox = QCheckBox("Y Component")
        self.tot_yield_checkbox = QCheckBox("Total")
    
        components_layout.addWidget(self.x_yield_checkbox)
        components_layout.addWidget(self.y_yield_checkbox)
        components_layout.addWidget(self.tot_yield_checkbox)
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
        self.filter_entry.setText(str(previous_input_YIELD.get("filtering", "0.0")))
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
        ##-------
        #options_subgroup.setLayout(options_layout)
        #display_options_layout.addWidget(options_subgroup)

        display_options_group.setLayout(display_options_layout)
        required_layout.addWidget(display_options_group)
        ##--------------------------------------------------------
        required_group.setLayout(required_layout)
        layout.addWidget(required_group)

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
        submit_btn = QPushButton("Calculate yield")
        submit_btn.setIcon(QIcon(":/icons/calculate_27dp_000000_FILL0_wght400_GRAD0_opsz24.png"))
        submit_btn.setIconSize(QSize(24, 24))
        submit_btn.setDefault(True)
        submit_btn.clicked.connect(self.on_submit)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(submit_btn)
        
        layout.addLayout(button_layout)

    ##$$$$$$$$$$$$$$$
    
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
            previous_input_YIELD["lambda0_nm"] = float(self.lambda0_entry.text())
        
            filtering = float(self.filter_entry.text())
            if not (0 <= filtering <= 100):
                raise ValueError("End-of-Pulse must be between 0 [%] and 100 [%].")
            previous_input_YIELD["filtering"] = float(self.filter_entry.text())
            
            qstart = float(self.qstart_entry.text())
            previous_input_YIELD["qstart"] = float(self.qstart_entry.text())
            if previous_input_YIELD["qstart"] <= 0:
                raise ValueError("qstart must be a positive number.")
            
            
            qend = float(self.qend_entry.text())
            previous_input_YIELD["qend"] = float(self.qend_entry.text())
            if previous_input_YIELD["qend"] <= previous_input_YIELD["qstart"]:
                raise ValueError("Invalid input", "Maximum HO must be greater than Minimum HO.")
                return

            time_derivative = self.derivative_entry.currentText()

            selected_yields = []
            if self.x_yield_checkbox.isChecked():
                selected_yields.append('x')
            if self.y_yield_checkbox.isChecked():
                selected_yields.append('y')
            if self.tot_yield_checkbox.isChecked():
                selected_yields.append('total')

            if not selected_yields:
                QMessageBox.warning(dialog, "Invalid input", "Please select at least one spectrum option.")
                return        
          
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

            result = (previous_input_YIELD["lambda0_nm"], previous_input_YIELD["filtering"], previous_input_YIELD["qstart"],
                      previous_input_YIELD["qend"], time_derivative, selected_yields)

            self.accept()
            # Update
            previous_input_YIELD.update({"lambda0_nm": lambda0_nm, "filtering": filtering, "qstart": qstart, "qend": qend})
            
            # CALL
            yield_connector(*result, window_func, self.parent().ipy_console)
                
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Error: {e}")

def input_dialog_YIELD(parent):
    dialog = ModernDialog(parent)
    return dialog.exec_()


