# helper_functions/helpers.py

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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDialog, QFormLayout, QProgressBar,
                             QRadioButton, QButtonGroup, QScrollArea, QColorDialog, QLineEdit, QMessageBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QSplashScreen, QDoubleSpinBox,
                             QMenu, QLabel, QWidget, QSpinBox, QFrame, QComboBox, QCheckBox, QDialogButtonBox,
                             QTabWidget, QTextEdit, QToolButton)
from PyQt5 import QtGui
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QCoreApplication
##----------------------------------------------------
def create_buttons_with_info_panel(buttons, info_panel_width=450):
    # Main Layout -------
    layout = QHBoxLayout()
    
    # Info Panel --------
    info_panel = QTextEdit()
    info_panel.setReadOnly(True)
    info_panel.setFont(QFont('Arial', 12))
    info_panel.setStyleSheet("""
    QTextEdit {
        background-color: #2b2b2b;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #555;
    }
    QTextEdit:hover {
        color: #00ccff;
        font-size: 12.5pt;
    }
    """)
    info_panel.setText("Hover over a button to see its description here.")
    info_panel.setFixedWidth(info_panel_width)
    layout.addWidget(info_panel)
    # end fucking Panel --------------------------
    
    # Button Panel 
    button_layout = QVBoxLayout()
    button_widgets = []
    
    for button_text, button_function, description in buttons:
        btn = QPushButton(button_text)
        btn.setFont(QFont('Arial', 12))
        btn.setStyleSheet("""
            QPushButton {
                background-color: #3b3b3b;
                color: #ffffff;
                padding: 10px;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        btn.setFixedHeight(50)
        btn.clicked.connect(button_function)
        btn.installEventFilter(btn)
        btn.description = description

        button_layout.addWidget(btn)
        
        button_widgets.append(btn)

        btn.eventFilter = lambda obj, event: handle_hover_event(obj, event, info_panel)

    layout.addLayout(button_layout)
    return layout, button_widgets

##----------------------------------------------------
def handle_hover_event(obj, event, info_panel):
    if event.type() == QEvent.HoverEnter and isinstance(obj, QPushButton):
        info_panel.setText(obj.description)
    elif event.type() == QEvent.HoverLeave:
        pass
    return False
##----------------------------------------------------
def update_info_panel(info_panel, new_text):

    info_panel.clear()
    info_panel.setText(new_text)




