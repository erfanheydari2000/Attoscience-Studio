# utils/window_func_ATTO.py

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

import numpy as np
##----------------------------------------------------
class TotalCurrentFilter:
    def __init__(self, method, EoP, exponent, sigma, decay_rate):
        """
        Initialize the filter with user-defined settings.
        Parameters:
            method (str)       : Filtering method ("cosine_window", "gaussian", "hanning", "exp_decay" , "welch", "bartlett" and "None")
            EoP (float)        :  End-of-Pulse
            exponent (float)   : Exponent for cosine filtering
            sigma (float)      : Standard deviation for Gaussian filtering
            decay_rate (float) : Exponential decay rate
        """
        self.method = method
        self.EoP = EoP
        self.exponent = exponent
        self.sigma = sigma
        self.decay_rate = decay_rate

    def apply_filter(self, t, jx, jy, djx, djy):
        """
        Apply the selected filtering method to the input signals.
        Parameters:
            t (array) : Time array
            jx, jy    : Current components
        
        Returns: hx, hy      
        """
        hx = np.copy(jx)
        hy = np.copy(jy)
        dhx = np.copy(djx)
        dhy = np.copy(djy)
        
        if self.method == "cosine":
            return self._cosine_window_filter(t, hx, hy, dhx, dhy)
        elif self.method == "Gaussian":
            return self._gaussian_filter(t, hx, hy, dhx, dhy)
        elif self.method == "Exponential Decay":
            return self._hanning_filter(t, hx, hy, dhx, dhy)
        elif self.method == "Hanning":
            return self._exp_decay_filter(t, hx, hy, dhx, dhy)
        elif self.method == "Welch":
            return self._welch_filter(t, hx, hy, dhx, dhy)            
        elif self.method == "Bartlett":
            return self._bartlett_filter(t, hx, hy, dhx, dhy)   
        else:
            return self._none(t, jx, jy, djx, djy)

    def _cosine_window_filter(self, t, hx, hy, dhx, dhy):
        ii = int(self.EoP * len(t))
        for i in range(ii, len(t)):
            factor = np.cos(0.5 * np.pi * (t[i] - t[ii]) / (t[-1] - t[ii])) ** self.exponent

            hx[i]  *= factor
            hy[i]  *= factor
            dhx[i] *= factor
            dhy[i] *= factor
            
        return hx, hy, dhx, dhy

    def _gaussian_filter(self, t, hx, hy, dhx, dhy):
        ii = int(self.EoP * len(t))
        gauss_window = np.exp(-((t[ii:] - t[ii])**2) / (2 * self.sigma**2))

        hx[ii:] *= gauss_window
        hy[ii:] *= gauss_window
        dhx[ii:] *= gauss_window
        dhy[ii:] *= gauss_window
        
        return hx, hy, dhx, dhy

    def _hanning_filter(self, t, hx, hy, dhx, dhy):
        ii = int(self.EoP * len(t))
        N = len(t) - ii

        j = np.arange(N)
        hann_window = 0.5 * (1 - np.cos((2 * np.pi * j) / N))

        hx[ii:]  *= hann_window
        hy[ii:]  *= hann_window
        dhx[ii:] *= hann_window
        dhy[ii:] *= hann_window
        
        return hx, hy, dhx, dhy

    def _exp_decay_filter(self, t, hx, hy, dhx, dhy):
        ii = int(self.EoP * len(t))
        decay_window = np.exp(-self.decay_rate * (t[ii:] - t[ii]))
        hx[ii:]  *= decay_window
        hy[ii:]  *= decay_window
        dhx[ii:] *= decay_window
        dhy[ii:] *= decay_window
        
        return hx, hy, dhx, dhy

    def _welch_filter(self, t, hx, hy, dhx, dhy):

        ii = int(self.EoP * len(t))
    
        N = len(t) - ii  # from ii to the end
        j = np.arange(N)
        welch_window = 1.0 - ((j - (N - 1) / 2.0) / ((N - 1) / 2.0))**2

        hx[ii:]  *= welch_window
        hy[ii:]  *= welch_window
        dhx[ii:] *= welch_window
        dhy[ii:] *= welch_window
        
        return hx, hy, dhx, dhy

    def _bartlett_filter(self, t, hx, hy, dhx, dhy):
        ii = int(self.EoP * len(t))
        N = len(t) - ii
        j = np.arange(N)
        bartlett_window = 1.0 - np.abs((j - 0.5 * N) / (0.5 * N))
        
        hx[ii:]  *= bartlett_window
        hy[ii:]  *= bartlett_window
        dhx[ii:] *= bartlett_window
        dhy[ii:] *= bartlett_window
        
        return hx, hy, dhx, dhy

    def _none(self, t, hx, hy, dhx, dhy):
        return hx, hy, dhx, dhy

