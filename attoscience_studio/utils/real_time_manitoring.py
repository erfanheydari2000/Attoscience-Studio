# utils/real_time_manitoring.py

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

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
import webbrowser
from functools import partial
import psutil
import time
from collections import deque
##----------------------------------------------------
class SystemMonitorChart(QWidget):
    def __init__(self, title="System Monitor", max_points=100, parent=None):
        super().__init__(parent)
        self.title = title
        self.max_points = max_points
        self.data_buffer = deque(maxlen=max_points)
        self.time_buffer = deque(maxlen=max_points)
        self.start_time = time.time()
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 5px;")
        layout.addWidget(self.title_label)
        
        # plot widget with dark theme
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#2B2B2B')
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#E0E0E0', width=1))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#E0E0E0', width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#E0E0E0'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#E0E0E0'))
        
        # grid
        self.plot_widget.showGrid(x=True, y=False, alpha=0.3)
        
        layout.addWidget(self.plot_widget)
        
        # Set fixed size for consistent layout
        #self.setFixedSize(350, 250)
        
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)
        
    def update_plot(self):
        pass
        
    def add_data_point(self, value):
        current_time = time.time() - self.start_time
        self.time_buffer.append(current_time)
        self.data_buffer.append(value)

class CPUMonitorChart(SystemMonitorChart):
    def __init__(self, parent=None):
        super().__init__("CPU Usage", parent=parent)
        self.setup_plot()
        
    def setup_plot(self):
        self.plot_widget.setLabel('left', 'Usage (%)', color='#E0E0E0')
        self.plot_widget.setLabel('bottom', '', color='#E0E0E0')
        self.plot_widget.setYRange(0, 100)
        
        # curve for CPU usage
        self.cpu_curve = self.plot_widget.plot(pen=pg.mkPen(color='#FF6B35', width=3))
    
        # threshold lines
        self.plot_widget.addLine(y=80, pen=pg.mkPen(color='#FF4444', width=2, style=pg.QtCore.Qt.DashLine))
        self.plot_widget.addLine(y=50, pen=pg.mkPen(color='#FFAA00', width=1, style=pg.QtCore.Qt.DotLine))
        
    def update_plot(self):
        # get current CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.add_data_point(cpu_percent)
        
        # Update title with current value
        self.title_label.setText(f"CPU Usage - {cpu_percent:.1f}%")
        
        # Update plot
        if len(self.data_buffer) > 1:
            self.cpu_curve.setData(list(self.time_buffer), list(self.data_buffer))
            
            # <<<Auto-scroll time axis>>>
            if len(self.time_buffer) > 0:
                latest_time = self.time_buffer[-1]
                self.plot_widget.setXRange(max(0, latest_time - 60), latest_time + 5)

class MemoryMonitorChart(SystemMonitorChart):
    def __init__(self, parent=None):
        super().__init__("Memory Usage", parent=parent)
        self.setup_plot()
        
    def setup_plot(self):
        self.plot_widget.setLabel('left', 'Usage (%)', color='#E0E0E0')
        self.plot_widget.setLabel('bottom', '', color='#E0E0E0')
        self.plot_widget.setYRange(0, 100)
        
        # curves for RAM and swap
        self.ram_curve = self.plot_widget.plot(pen=pg.mkPen(color='#00FF88', width=3), name='RAM')
        self.swap_curve = self.plot_widget.plot(pen=pg.mkPen(color='#3F51B5', width=2), name='Swap')
        
        # threshold lines
        self.plot_widget.addLine(y=90, pen=pg.mkPen(color='#FF4444', width=2, style=pg.QtCore.Qt.DashLine))
        self.plot_widget.addLine(y=75, pen=pg.mkPen(color='#FFAA00', width=1, style=pg.QtCore.Qt.DotLine))
        
        # legend
        self.plot_widget.addLegend()
        
    def update_plot(self):
        # Get memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        ram_percent = memory.percent
        swap_percent = swap.percent if swap.total > 0 else 0
        
        self.add_data_point((ram_percent, swap_percent))
        
        # Update title with current values
        self.title_label.setText(f"Memory - RAM: {ram_percent:.1f}% | Swap: {swap_percent:.1f}%")
        
        # Update plot
        if len(self.data_buffer) > 1:
            ram_data = [point[0] for point in self.data_buffer]
            swap_data = [point[1] for point in self.data_buffer]
            
            self.ram_curve.setData(list(self.time_buffer), ram_data)
            self.swap_curve.setData(list(self.time_buffer), swap_data)
            
            # <<<Auto-scroll time axis>>>
            if len(self.time_buffer) > 0:
                latest_time = self.time_buffer[-1]
                self.plot_widget.setXRange(max(0, latest_time - 60), latest_time + 5)

class NetworkMonitorChart(SystemMonitorChart):
    def __init__(self, parent=None):
        super().__init__("Network I/O", parent=parent)
        self.prev_bytes_sent = 0
        self.prev_bytes_recv = 0
        self.setup_plot()
        
    def setup_plot(self):
        self.plot_widget.setLabel('left', 'Speed (KB/s)', color='#E0E0E0')
        self.plot_widget.setLabel('bottom', 'Time (s)', color='#E0E0E0')
        
        # curves for upload and download
        self.upload_curve = self.plot_widget.plot(pen=pg.mkPen(color='#FF4081', width=3), name='Upload')
        self.download_curve = self.plot_widget.plot(pen=pg.mkPen(color='#2196F3', width=3), name='Download')
        
        # legend
        self.plot_widget.addLegend()
        
    def update_plot(self):
        # Get network stats
        net_io = psutil.net_io_counters()
        
        if self.prev_bytes_sent > 0:
            bytes_sent_per_sec = (net_io.bytes_sent - self.prev_bytes_sent)
            bytes_recv_per_sec = (net_io.bytes_recv - self.prev_bytes_recv)
            
            # Convert to KB/s
            upload_speed = bytes_sent_per_sec / 1024
            download_speed = bytes_recv_per_sec / 1024
            
            self.add_data_point((upload_speed, download_speed))
            
            # Update title with current speeds
            self.title_label.setText(f"Network - ↑{upload_speed:.1f} KB/s | ↓{download_speed:.1f} KB/s")
            
            # Update plot
            if len(self.data_buffer) > 1:
                upload_data = [point[0] for point in self.data_buffer]
                download_data = [point[1] for point in self.data_buffer]
                
                self.upload_curve.setData(list(self.time_buffer), upload_data)
                self.download_curve.setData(list(self.time_buffer), download_data)
                
                # Auto-scroll time axis and adjust Y range
                if len(self.time_buffer) > 0:
                    latest_time = self.time_buffer[-1]
                    self.plot_widget.setXRange(max(0, latest_time - 60), latest_time + 5)
                    
                    # Dynamic Y range based on data
                    max_speed = max(max(upload_data[-10:], default=0), max(download_data[-10:], default=0))
                    self.plot_widget.setYRange(0, max(max_speed * 1.2, 10))
        
        self.prev_bytes_sent = net_io.bytes_sent
        self.prev_bytes_recv = net_io.bytes_recv

class DiskMonitorChart(SystemMonitorChart):
    def __init__(self, parent=None):
        super().__init__("Disk I/O", parent=parent)
        self.prev_read_bytes = 0
        self.prev_write_bytes = 0
        self.setup_plot()
        
    def setup_plot(self):
        self.plot_widget.setLabel('left', 'Speed (MB/s)', color='#E0E0E0')
        self.plot_widget.setLabel('bottom', 'Time (s)', color='#E0E0E0')
        
        # curves for read and write
        self.read_curve = self.plot_widget.plot(pen=pg.mkPen(color='#4CAF50', width=3), name='Read')
        self.write_curve = self.plot_widget.plot(pen=pg.mkPen(color='#FF9800', width=3), name='Write')
        
        # legend
        self.plot_widget.addLegend()
        
    def update_plot(self):
        # get disk I/O stats
        disk_io = psutil.disk_io_counters()
        
        if disk_io and self.prev_read_bytes > 0:  # Skip first measurement
            read_bytes_per_sec = (disk_io.read_bytes - self.prev_read_bytes)
            write_bytes_per_sec = (disk_io.write_bytes - self.prev_write_bytes)
            
            # Convert to MB/s
            read_speed = read_bytes_per_sec / (1024 * 1024)
            write_speed = write_bytes_per_sec / (1024 * 1024)
            
            self.add_data_point((read_speed, write_speed))
            
            # Update title with current speeds
            self.title_label.setText(f"Disk I/O - Read: {read_speed:.2f} MB/s | Write: {write_speed:.2f} MB/s")
            
            # Update plot
            if len(self.data_buffer) > 1:
                read_data = [point[0] for point in self.data_buffer]
                write_data = [point[1] for point in self.data_buffer]
                
                self.read_curve.setData(list(self.time_buffer), read_data)
                self.write_curve.setData(list(self.time_buffer), write_data)
                
                # Auto-scroll time axis and adjust Y range
                if len(self.time_buffer) > 0:
                    latest_time = self.time_buffer[-1]
                    self.plot_widget.setXRange(max(0, latest_time - 60), latest_time + 5)
                    
                    # Dynamic Y range based on data
                    max_speed = max(max(read_data[-10:], default=0), max(write_data[-10:], default=0))
                    self.plot_widget.setYRange(0, max(max_speed * 1.2, 1))
        
        if disk_io:
            self.prev_read_bytes = disk_io.read_bytes
            self.prev_write_bytes = disk_io.write_bytes
######################################################################
class SystemMonitorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # Title for the monitoring section
        title_label = QLabel("System Performance Monitor")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #f0f0f0; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # System info section
        info_layout = QHBoxLayout()
        
        # CPU info
        cpu_info = f"CPU: {psutil.cpu_count()} cores"
        cpu_label = QLabel(cpu_info)
        cpu_label.setFont(QFont("Arial", 8))
        cpu_label.setStyleSheet("color: #bbbbbb; padding: 5px;")
        info_layout.addWidget(cpu_label)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = f"RAM: {memory.total / (1024**3):.1f} GB"
        memory_label = QLabel(memory_info)
        memory_label.setFont(QFont("Arial", 8))
        memory_label.setStyleSheet("color: #bbbbbb; padding: 5px;")
        info_layout.addWidget(memory_label)
        
        main_layout.addLayout(info_layout)
        
        # monitoring charts in a 2x2 grid
        charts_layout = QVBoxLayout()
        
        # Top row
        top_row = QHBoxLayout()
        self.cpu_chart = CPUMonitorChart()
        self.memory_chart = MemoryMonitorChart()
        top_row.addWidget(self.cpu_chart)
        top_row.addWidget(self.memory_chart)
        charts_layout.addLayout(top_row)
        
        # Bottom row
        bottom_row = QHBoxLayout()
        self.network_chart = NetworkMonitorChart()
        self.disk_chart = DiskMonitorChart()
        bottom_row.addWidget(self.network_chart)
        bottom_row.addWidget(self.disk_chart)
        charts_layout.addLayout(bottom_row)
        
        main_layout.addLayout(charts_layout)
        
        main_layout.addStretch(1)

        # background
        self.setStyleSheet("background-color: #2B2B2B;") 



