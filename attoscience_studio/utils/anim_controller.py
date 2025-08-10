# utils/anim_controller.py

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

import os
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')  # non-interactive backend for saving

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import animation
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject
import copy

from datetime import datetime
from qtconsole.rich_jupyter_widget import RichJupyterWidget
###------------------------------------------------------------------
def print_to_console(console: RichJupyterWidget, bar: str):
    if hasattr(console, "_kernel_client"):
        console._kernel_client.execute(f"print('''{bar}''')")


class AnimationSaverThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, animation_data, save_format, save_dir, dpi=150, fps=14):
        super().__init__()
        self.animation_data = copy.deepcopy(animation_data)  # Deep copy to avoid conflicts
        self.save_format = save_format
        self.save_dir = save_dir
        self.dpi = dpi
        self.fps = fps
        self.canceled = False
        self.save_fig = None
        self.anim = None

        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for saving

    def run(self):
        try:
            #import matplotlib
            #matplotlib.use('Agg')  # non-interactive backend for saving

            # Create a completely separate figure for saving
            with plt.ioff():  # Turn off interactive mode
                self.save_fig = plt.figure(figsize=(12, 8))
                
                # Recreate the entire plot structure for saving
                self._setup_save_figure()
                
                # Create animation for saving
                self.anim = animation.FuncAnimation(
                    self.save_fig,
                    self._animate_save_frame,
                    frames=len(self.animation_data['smooth_frames_curr']),
                    interval=50,
                    blit=False,
                    repeat=False,
                    cache_frame_data=False
                )
                
                # Create appropriate writer
                writer = self._create_writer()
                                
                # Progress callback
                def progress_callback(current_frame, total_frames):
                    if self.canceled:
                        return False
                    percent = int(100 * current_frame / total_frames)
                    self.progress.emit(percent)
                    return True

                # Save animation
                filename = os.path.join(self.save_dir, f"animation.{self.save_format}")
                self.anim.save(filename, writer=writer, dpi=self.dpi, progress_callback=progress_callback)
                
            if not self.canceled:
                self.finished.emit()
            
        except Exception as e:
            if not self.canceled:
                self.error.emit(str(e))
        finally:
            self._cleanup()
    
    def _setup_save_figure(self):
        """Setup the complete figure structure for saving"""
        # This should mirror your original plot setup
        # You'll need to recreate all the subplots, 3D plots, etc.
        
        # Example structure - adjust based on your actual plot layout:
        gs = self.save_fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 3D subplot (adjust position based on your layout)
        self.save_ax_3d = self.save_fig.add_subplot(gs[0, 0], projection='3d')
        self.save_ax_3d.set_xlabel('Time')
        self.save_ax_3d.set_ylabel('Ax')
        self.save_ax_3d.set_zlabel('Ay')
        
        # Plot the trajectory
        self.save_ax_3d.plot(
            self.animation_data['interpolated_Time'],
            self.animation_data['interpolated_AX'],
            self.animation_data['interpolated_AY'],
            'b-', alpha=0.3
        )
        
        # Create marker
        self.save_marker, = self.save_ax_3d.plot([], [], [], 'ro', markersize=8)
        
        # Add coordinate text
        self.save_coordinate_text = self.save_ax_3d.text2D(0.05, 0.95, '', 
                                                          transform=self.save_ax_3d.transAxes)
        
        # Current plot subplot
        self.save_ax_curr = self.save_fig.add_subplot(gs[0, 1])
        self.save_ax_curr.set_title('Current')
        # Initialize with first frame
        first_curr = self.animation_data['smooth_frames_curr'][0]
        self.save_img1 = self.save_ax_curr.imshow(first_curr, cmap='viridis', aspect='auto')
        self.save_cbar1 = self.save_fig.colorbar(self.save_img1, ax=self.save_ax_curr)
        
        # Next plot subplot  
        self.save_ax_nex = self.save_fig.add_subplot(gs[0, 2])
        self.save_ax_nex.set_title('Next')
        # Initialize with first frame
        first_nex = self.animation_data['smooth_frames_nex'][0]
        self.save_img2 = self.save_ax_nex.imshow(first_nex, cmap='viridis', aspect='auto')
        self.save_cbar2 = self.save_fig.colorbar(self.save_img2, ax=self.save_ax_nex)
    
    def _animate_save_frame(self, frame):
        """Animation function for saving - completely independent"""
        try:
            if self.canceled or frame >= len(self.animation_data['smooth_frames_curr']):
                return []
            
            # Update 3D marker
            x = self.animation_data['interpolated_Time'][frame]
            y = self.animation_data['interpolated_AX'][frame]
            z = self.animation_data['interpolated_AY'][frame]
            self.save_marker.set_data_3d([x], [y], [z])
            self.save_coordinate_text.set_text(
                f"x={x:.3f}\nAx={y:.3f}\nAy={z:.3f}\nFrame={frame+1}/{len(self.animation_data['smooth_frames_curr'])}"
            )
            
            # Update current plot
            curr = self.animation_data['smooth_frames_curr'][frame]
            self.save_img1.set_array(curr)
            self.save_img1.set_clim(vmin=np.nanmin(curr), vmax=np.nanmax(curr))
            
            # Update next plot
            nex = self.animation_data['smooth_frames_nex'][frame]
            self.save_img2.set_array(nex)
            self.save_img2.set_clim(vmin=np.nanmin(nex), vmax=np.nanmax(nex))
            
            return [self.save_marker, self.save_coordinate_text, self.save_img1, self.save_img2]
            
        except Exception as e:
            print(f"Save frame {frame} error: {str(e)}")
            return []

    def _create_writer(self):
        """Create appropriate writer based on format"""
        if self.save_format == 'mp4':
            from matplotlib.animation import FFMpegWriter
            return FFMpegWriter(fps=self.fps)
        elif self.save_format == 'avi':
            from matplotlib.animation import FFMpegWriter
            return FFMpegWriter(fps=self.fps, codec='mpeg4')
        elif self.save_format == 'webm':
            from matplotlib.animation import FFMpegWriter
            return FFMpegWriter(fps=self.fps, codec='libvpx')
        elif self.save_format == 'gif':
            from matplotlib.animation import PillowWriter
            return PillowWriter(fps=self.fps)
        elif self.save_format == 'mkv':
            from matplotlib.animation import FFMpegWriter
            return FFMpegWriter(fps=self.fps, codec='libx264')
        else:
            raise ValueError(f"Unsupported format: {self.save_format}")
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            if self.anim:
                self.anim._stop()
                self.anim = None
            if self.save_fig:
                plt.close(self.save_fig)
                self.save_fig = None
        except:
            pass
    
    def cancel(self):
        self.canceled = True
        self._cleanup()

###------------------------------------------------------------------
class AnimationController(QObject):
    def __init__(self, figure, canvas, ipy_console=None):
        super().__init__()
        self.canvas = canvas
        self.figure = figure
                      
        self.ipy_console = ipy_console
        
        self.anim = None
        self._is_active = False
        self._current_frame = 0
        self._frames_curr = []
        self._setup_complete = False
        
        self.saver_thread = None
        
        # Use a timer to delay setup until canvas is ready
        self.setup_timer = QTimer()
        self.setup_timer.setSingleShot(True)
        self.setup_timer.timeout.connect(self._delayed_setup)
        self.setup_timer.start(100)  # 100ms delay

    def _delayed_setup(self):
        """Delayed setup to ensure canvas is ready"""
        try:
            if not hasattr(self.figure, 'animation_data'):
                return
                
            data = self.figure.animation_data
            self._frames_curr = data['smooth_frames_curr']
            self._total_frames = len(self._frames_curr)
            
            # Store references to all artists for DISPLAY only
            self.marker = data['marker']
            self.coordinate_text = data['coordinate_text']
            self.img1 = data['img1']
            self.img2 = data['img2']
            self.cbar1 = data['cbar1']
            self.cbar2 = data['cbar2']
            
            self._setup_complete = True
        
            # Start BOTH operations simultaneously but independently
            should_save = data.get('save_animation', False)
            if should_save:
                try:
                    self.save_animation(data.get('format', 'mp4'), data.get('save_dir', '.'))
                except Exception as e:
                    print(f"Saving animation failed: {str(e)}")
        
            # ALWAYS start display animation immediately
            self._start_display_animation()
        
        except Exception as e:
            print(f"Animation setup failed: {str(e)}")

    def _start_display_animation(self):
        """Start the display animation"""        
        try:
            # Create animation for display (using every 2nd frame for performance)
            self.anim = animation.FuncAnimation(
                self.figure,
                self._animate_frame,
                frames=range(0, self._total_frames, 2),  # Skip frames for performance
                interval=50,  # 20 FPS
                blit=False,
                repeat=True,  # Allow repeat for display
                cache_frame_data=False
            )
            
            self._is_active = True
            self.canvas.draw()
            
        except Exception as e:
            print(f"Failed to start display animation: {str(e)}")

    def _animate_frame(self, frame):
        """Frame update handler for DISPLAY animation only"""
        if not self._is_active or not self._setup_complete:
            return []
            
        try:
            # Ensure frame index is within bounds
            if frame >= len(self._frames_curr):
                return []
            
            self._current_frame = frame
            data = self.figure.animation_data
            
            # Update 3D marker
            x = data['interpolated_Time'][frame]
            y = data['interpolated_AX'][frame]
            z = data['interpolated_AY'][frame]
            self.marker.set_data_3d([x], [y], [z])
            self.coordinate_text.set_text(
                f"x={x:.3f}\nAx={y:.3f}\nAy={z:.3f}\nFrame={frame+1}/{self._total_frames}"
            )
            
            # Update current plot
            curr = self._frames_curr[frame]
            self.img1.set_array(curr)
            self.img1.set_clim(vmin=np.nanmin(curr), vmax=np.nanmax(curr))
            self.cbar1.update_normal(self.img1)
            
            # Update next plot
            nex = data['smooth_frames_nex'][frame]
            self.img2.set_array(nex)
            self.img2.set_clim(vmin=np.nanmin(nex), vmax=np.nanmax(nex))
            self.cbar2.update_normal(self.img2)
            
            return [self.marker, self.coordinate_text, self.img1, self.img2]
            
        except Exception as e:
            print(f"Frame {frame} error: {str(e)}")
            return []

    def save_animation(self, save_format, save_dir):
        """Save animation in background thread with completely separate objects"""
        if not self._setup_complete:
            return
            
        # Stop any existing saver thread
        if self.saver_thread and self.saver_thread.isRunning():
            self.saver_thread.cancel()
            self.saver_thread.wait(5000)  # Wait up to 5 seconds
            if self.saver_thread.isRunning():
                self.saver_thread.terminate()
                self.saver_thread.wait()
        
        # Create new saver thread with COPIED animation data
        self.saver_thread = AnimationSaverThread(
            self.figure.animation_data,  # This will be deep-copied in the thread
            save_format, 
            save_dir
        )
        
        self.saver_thread.finished.connect(self.on_save_finished) ###>>>>>>
        self.saver_thread.progress.connect(self.on_save_progress) ###>>>>>>
        self.saver_thread.error.connect(self.on_save_error)       ###>>>>>>
        self.saver_thread.start() ###>>>>>>
        
        return self.saver_thread
        
    def on_save_finished(self):
        print("Animation saved successfully")
   
   ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def help_to_console(self, bar):     
        if self.ipy_console:
            print_to_console(self.ipy_console, bar)
        else:
            print("No console available: ", bar)
    
    def draw_ascii_progress_bar(self, percent, width=40):
        done = int(width * percent / 100)
        remaining = width - done
        
        full_block = '\u2588'
        bar = f"[{full_block * done}{'.' * remaining}] {percent:.1f}%"
        self.help_to_console(bar)

    ##$$$
    def on_save_progress(self, percent):
        self.draw_ascii_progress_bar(percent)
   ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    def on_save_error(self, error):
        print(f"Save error: {error}")

    def start_animation(self):
        """Start or resume animation"""
        if self.anim and self._setup_complete:
            self._is_active = True
            try:
                if hasattr(self.anim, 'event_source') and self.anim.event_source:
                    self.anim.event_source.start()
            except Exception as e:
                print(f"Start error: {str(e)}")

    def pause_animation(self):
        """Pause animation"""
        if self.anim and self._setup_complete:
            self._is_active = False
            try:
                if hasattr(self.anim, 'event_source') and self.anim.event_source:
                    self.anim.event_source.stop()
            except Exception as e:
                print(f"Pause error: {str(e)}")

    def reset_animation(self):
        """Reset to first frame"""
        self.pause_animation()
        if self._setup_complete:
            self._current_frame = 0
            self._animate_frame(0)
            self.canvas.draw_idle()

    def stop_animation(self):
        """Complete stop and cleanup"""
        self._is_active = False
        self.pause_animation()
        
        # Stop and cleanup saver thread
        if self.saver_thread and self.saver_thread.isRunning():
            self.saver_thread.cancel()
            self.saver_thread.wait(3000)  # Wait up to 3 seconds
            if self.saver_thread.isRunning():
                self.saver_thread.terminate()
                self.saver_thread.wait()
        
        # Clean up animation
        if self.anim:
            try:
                self.anim._stop()
                if hasattr(self.anim, '_blit_cache'):
                    self.anim._blit_cache.clear()
                self.anim = None
            except Exception as e:
                print(f"Stop error: {str(e)}")
        
        self._setup_complete = False

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.stop_animation()



