# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re
import logging
from typing import Tuple, List
from dataclasses import dataclass

import numpy as np



LOG = logging.getLogger("parser")



@dataclass
class Segment:
    """
    G-code line segment
    """

    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    width: float
    feedrate: float
    tool_temp: float
    bed_temp: float
    fan_speed: float



class Layer:
    """
    G-code layer
    """

    def __init__(self, number: int = 0):
        self.number = number
        self.segments = []



class Parser():
    """
    G-code parser.
    Returns a list of layers. Each layer contains a list of `Segment` objects.
    """

    def __init__(self):
        self.position = np.array([0, 0, 0], dtype="float32")
        self.extrusion = 0
        self.relative = None
        self.unit_multiplier = 1
        self.feedrate = 100
        self.tool_temp = 0
        self.bed_temp = 0
        self.fan_speed = 0


    def parse_line(self, line, n=None):
        command = None
        comment = None
        values = {}

        # Remove comments
        if ";" in line:
            line, comment = [v.strip() for v in line.split(";", 1)]

        re_command = re.compile(r"([A-Z])\s*([0-9.-]+)")

        for i, match in enumerate(re_command.finditer(line)):
            alpha, number = match.groups()
            if i == 0:
                command = f"{alpha}{number}"
            else:
                values[alpha] = number

        return command, values, comment


    def parse(self, fp):
        layers = []
        extrusion_z_values = set()

        layers.append(Layer(number=len(extrusion_z_values)))
        extrusion_z_values.add(0)

        for n, line in enumerate(fp, 1):
            line = line.rstrip()

            command, values, _comment = self.parse_line(line, n=n)

            if command is None:
                continue

            if command in ("G0", "G1"):
                start_p = np.copy(self.position)
                start_e = self.extrusion
                x_value = None
                y_value = None
                e_value = None

                if self.relative is None:
                    raise NotImplementedError(f"G0: absolute/relative mode not set")
                if "X" in values:
                    x_value = float(values.pop("X"))
                    self.position[0] = self.position[0] * self.relative + x_value
                if "Y" in values:
                    y_value = float(values.pop("Y"))
                    self.position[1] = self.position[1] * self.relative + y_value

                if "Z" in values:
                    z_value = float(values.pop("Z"))
                    self.position[2] = self.position[2] * self.relative + z_value

                if "E" in values:
                    e_value = float(values.pop("E"))
                    self.extrusion = self.extrusion * self.relative + e_value

                if "F" in values:
                    self.feedrate = float(values.pop("F"))

                if values:
                    raise NotImplementedError(f"G0: {values}")

                if (
                        (
                            x_value is not None or
                            y_value is not None
                        ) and
                        self.position[2] not in extrusion_z_values
                ):
                    layers.append(Layer(number=len(extrusion_z_values)))
                    extrusion_z_values.add(self.position[2])

                end_p = np.copy(self.position)
                end_e = self.extrusion

                distance_p = np.linalg.norm(end_p - start_p)
                distance_e = end_e - start_e
                width = distance_e / distance_p if distance_p else 0

                layers[-1].segments.append(Segment(
                    start=start_p,
                    end=end_p,
                    width=width,
                    feedrate=self.feedrate,
                    tool_temp=self.tool_temp,
                    bed_temp=self.bed_temp,
                    fan_speed=self.fan_speed,
                ))

            elif command == "G21":
                self.unit_multiplier = 1
            elif command == "G28":
                if not values:
                    self.position[0] = 0
                    self.position[1] = 0
                    self.position[2] = 0
                else:
                    raise NotImplementedError(f"G28: {values}")
            elif command == "G90":
                if values:
                    raise NotImplementedError(f"G90: {values}")
                self.relative = False
            elif command == "G91":
                if values:
                    raise NotImplementedError(f"G91: {values}")
                self.relative = True
            elif command == "G92":
                if values == {"E": "0"}:
                    self.extrusion = 0
                else:
                    raise NotImplementedError(f"G92: {values}")
            elif command == "M92":
                LOG.debug(f"Ignore set steps per unit")
            elif command in ("M104", "M109"):
                if "S" in values:
                    self.tool_temp = float(values.pop("S"))
                if values:
                    raise NotImplementedError(f"{command}: {values}")
            elif command in ("M140", "M190"):
                if "S" in values:
                    self.bed_temp = float(values.pop("S"))
                if values:
                    raise NotImplementedError(f"{command}: {values}")
            elif command == "M106":
                if "S" in values:
                    self.fan_speed = float(values.pop("S"))
                if values:
                    raise NotImplementedError(f"106: {values}")
            elif command == "M107":
                if values:
                    raise NotImplementedError(f"107: {values}")
                self.fan_speed = 0
            elif command == "M117":
                LOG.info(f"Message: {line}")
            elif command == "M201":
                LOG.debug(f"Ignore adjust acceleration: {command} {values}")
            elif command == "M204":
                LOG.debug(f"Ignore adjust acceleration: {command} {values}")
            else:
                LOG.debug(f"{n}: Ignoring {line}")

        return layers
