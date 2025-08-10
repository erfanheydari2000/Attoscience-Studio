# parser/parserlog_parser.py

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

import re
import os, sys
import math
import numpy as np
##----------------------------------------------------
class CrystalStructureParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lattice_vectors = []
        self.lattice_parameters = []
        self.reduced_coords = []
        self.scaled_vectors = []

    def parse(self):
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        current_block = None
        for line in lines:
            line = line.strip()
            if "Opened block 'ReducedCoordinates'" in line:
                current_block = 'coords'
            elif "Closed block 'ReducedCoordinates'" in line:
                current_block = None
            elif "Opened block 'LatticeParameters'" in line:
                current_block = 'params'
            elif "Closed block 'LatticeParameters'" in line:
                current_block = None
            elif "Opened block 'LatticeVectors'" in line:
                current_block = 'vectors'
            elif "Closed block 'LatticeVectors'" in line:
                current_block = None

            elif current_block == 'coords':
                m = re.match(r'ReducedCoordinates\[\d+\]\[(\d+)\] = (.+)', line)
                if m:
                    idx, value = int(m.group(1)), m.group(2).replace('"', '')
                    if idx == 0:
                        self.reduced_coords.append([value])
                    else:
                        self.reduced_coords[-1].append(float(value))

            elif current_block == 'params':
                m = re.match(r'LatticeParameters\[\d+\]\[(\d+)\] = (.+)', line)
                if m:
                    self.lattice_parameters.append(float(m.group(2)))

            elif current_block == 'vectors':
                m = re.match(r'LatticeVectors\[\d+\]\[(\d+)\] = (.+)', line)
                if m:
                    idx = int(m.group(1))
                    val = float(m.group(2))
                    if idx == 0:
                        self.lattice_vectors.append([val])
                    else:
                        self.lattice_vectors[-1].append(val)
      
        if (self.lattice_vectors and self.lattice_parameters and 
            len(self.lattice_vectors) == 3 and len(self.lattice_parameters) == 3):
            self.scaled_vectors = []
            for i in range(3):
                vec = np.array(self.lattice_vectors[i])
                length = np.linalg.norm(vec)
                if length > 1e-10:
                    scale = self.lattice_parameters[i] / length
                else:
                    scale = 1.0
                scaled_vec = scale * vec
                self.scaled_vectors.append(scaled_vec.tolist())


