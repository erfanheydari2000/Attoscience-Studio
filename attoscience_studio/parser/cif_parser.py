# parser/cif_parser.py

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
def parse_cif(file_path):
    """Parse CIF file to extract lattice parameters and
       atom positions
    """
    cell_params = {}
    atoms_frac = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            data_lines.append(stripped)
    
    cell_keys = [
        '_cell_length_a', '_cell_length_b', '_cell_length_c',
        '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'
    ]
    
    for line in data_lines:
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key in cell_keys:
            cell_params[key] = float(parts[1])
    
    headers = []
    data_rows = []
    in_atom_loop = False
    found_atom_headers = False
    
    for line in data_lines:
        if line == 'loop_':
            in_atom_loop = True
            headers = []
            continue
            
        if in_atom_loop:
            if line.startswith('_atom_site_'):
                headers.append(line)
            else:
                if headers and not found_atom_headers:
                    required_headers = [
                        '_atom_site_type_symbol',
                        '_atom_site_fract_x',
                        '_atom_site_fract_y',
                        '_atom_site_fract_z'
                    ]
                    if all(h in headers for h in required_headers):
                        found_atom_headers = True
                    else:
                        headers = []
                        in_atom_loop = False
                if found_atom_headers:
                    data_rows.append(line.split())
        else:
            if found_atom_headers:
                break
                
    if not found_atom_headers:
        raise ValueError("Required atom site headers not found in CIF file.")
    
    idx_symbol = headers.index('_atom_site_type_symbol')
    idx_x = headers.index('_atom_site_fract_x')
    idx_y = headers.index('_atom_site_fract_y')
    idx_z = headers.index('_atom_site_fract_z')
    
    for row in data_rows:
        if len(row) < max(idx_symbol, idx_x, idx_y, idx_z) + 1:
            continue
        symbol = row[idx_symbol]
        try:
            x = float(row[idx_x])
            y = float(row[idx_y])
            z = float(row[idx_z])
            atoms_frac.append((symbol, x, y, z))
        except ValueError:
            continue
            
    return cell_params, atoms_frac


