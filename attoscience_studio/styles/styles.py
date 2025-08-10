# styles/styles.py

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

from PyQt5.QtGui import QFont, QGuiApplication
from PyQt5.QtWidgets import QApplication
import sys
##----------------------------------------------------
class StyleManager:
    BACKGROUND_COLOR = "#1e1e1e"
    TEXT_COLOR = "#ffffff"
    SUB_TEXT_COLOR = "#bbbbbb"
    ACCENT_COLOR = "#3b3b3b"
    HOVER_COLOR = "#444444"
    
    # Fonts
    HEADER_FONT = QFont('Arial', 18, QFont.Bold)
    SUB_HEADER_FONT = QFont('Arial', 14)
    FOOTER_FONT = QFont('Arial', 10)
    TAB_FONT = QFont('Arial', 20, QFont.Bold)

    ##----------------------------------------------------
    @classmethod
    def get_main_window_style(cls):
        return f"""
        QMainWindow {{
            background-color: {cls.BACKGROUND_COLOR};
            color: {cls.TEXT_COLOR};
            font-family: Arial;
            
            border: 1px solid #333;
            border-radius: 10px;
        }}
        """

    @classmethod
    def get_tabs_style(cls):
        return f"""
        QTabBar::tab {{
        
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 {cls.BACKGROUND_COLOR}, stop: 1 {cls.ACCENT_COLOR});
            color: {cls.TEXT_COLOR};
            
            padding: 10px;
            font-family: 'Arial';
            font-size: 14px;
            font-weight: bold;
            
            min-width: 100px;
            
            border: 1px solid #444444;
            border-radius: 5px;
            
            margin-right: 3px;  /* Case 2: spacing between tabs */
        }}

        QTabBar::tab:hover {{
            background-color: {cls.HOVER_COLOR};
        }}

        QTabBar::tab:selected {{
            background-color: #2b3a3a;
            color: {cls.TEXT_COLOR};
            border: 2px solid {cls.ACCENT_COLOR};  /* Case 3: highlight selected tab */
            /* box-shadow is not supported in QSS, but this comment shows intent */
        }}
        """

    @classmethod
    def get_footer_style(cls):
        return f"""
            padding: 10px; 
            background-color: #1e1e1e; 
            color: #bbbbbb;
        """

    @classmethod
    def get_central_widget(cls):
        return f"""
            background-color: #121212;
            """

    @classmethod
    def create_settings_menu__button(cls):
        return f"""
        QToolButton {{
            border: none;
            background-color: #1e1e1e;
            color: #ffffff;
        }}
        QToolButton:hover {{
            background-color: #444444;
        }}
        """
    @classmethod
    def create_settings_menu__itsmenu(cls):
        return  f"""
        QMenu {{
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #444444;
        }}
        QMenu::item {{
            padding: 8px 20px;
        }}
        QMenu::item:selected {{
            background-color: #555555;
        }}
        """

    @classmethod
    def get_background_footer_image(cls):
        return f"""
            background-color: #1e1e1e;
            """
    @classmethod
    def get_terminal_style(cls):
        # WARRRM
        return """
        QPlainTextEdit, QTextEdit {
            background-color: #1a1a1a;
            color: #f5f5dc;
            font-family: 'SF Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13pt;
            line-height: 1.6;
            selection-background-color: #3e3e3e;
            selection-color: #ffffff;
            border: 3px solid #404040;
            border-radius: 20px;
            padding: 10px;
        }
        /* Custom scrollbar */
        QScrollBar:vertical {{
            background-color: #21262d;
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: #484f58;
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: #656c76;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
         }} 
        """

