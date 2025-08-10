# helper_functions/constants.py

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
##----------------------------------------------------
class PhysicalConstants:
    c_ms = 299792458         # (SI) units
    c_au = 137.6             # (G) units
    CNST_1_TWcm2 = 0.6       # V/A
    hbar = 1.0545718e-34
    inverse_alpha_fine = 137.035999139
    c_light = 299792458.0
    elcharge = 1.602176565e-19
    elmass = 9.10938356e-31
    r_Bohr = hbar * inverse_alpha_fine / (c_light * elmass)
    alpha_fine = 1.0 / inverse_alpha_fine
    Ip_HeV = 27.21138602
    mu0 = 4.0 * np.pi * 1e-7
    eps0 = 1.0 / (mu0 * c_light**2)
    r_electron_classical = r_Bohr * (alpha_fine**2)
    Boltzmann_constant = 1.380649e-23
    Avogadro_constant = 6.02214076e23
    universal_gas_constant = Boltzmann_constant * Avogadro_constant
class AtomicUnits:
    r_Bohr = PhysicalConstants.hbar * PhysicalConstants.inverse_alpha_fine / (PhysicalConstants.c_light * PhysicalConstants.elmass)
    TIMEau = (PhysicalConstants.inverse_alpha_fine**2) * PhysicalConstants.hbar / (PhysicalConstants.elmass * PhysicalConstants.c_light**2)
    EFIELDau = PhysicalConstants.hbar**2 / (PhysicalConstants.elmass * r_Bohr**3 * PhysicalConstants.elcharge)
    
    LENGTHau = r_Bohr
    ENERGYau = PhysicalConstants.hbar**2 / (PhysicalConstants.elmass * r_Bohr**2)
