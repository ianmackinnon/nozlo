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

import numpy as np


LOG = logging.getLogger("parser")

class Parser():
    """
    G-code parser
    """

    def __init__(self):
        self.position = np.array([0, 0, 0], dtype="float32")
        self.extrusion = 0
        self.relative = None
        self.unit_multiplier = 1
        self.feedrate = 100


    def parse_line(self, line, n=None):

        # Remove comments
        line = line.split(";", 1)[0].strip()

        command = None
        values = {}
        for i, match in enumerate(re.compile("([A-Z])\s*([0-9.-]+)").finditer(line)):
            alpha, number = match.groups()
            if i == 0:
                command = f"{alpha}{number}"
            else:
                values[alpha] = number

        return command, values


    def parse(self, fp):
        lines = []

        for n, line in enumerate(fp, 1):
            line = line.rstrip()

            command, values = self.parse_line(line, n=n)

            if command is None:
                continue


            if command in ("G0", "G1"):

                start_p = np.copy(self.position)
                start_e = self.extrusion

                if self.relative is None:
                    raise NotImplementedError(f"G0: absolute/relative mode not set")
                if "X" in values:
                    self.position[0] = self.position[0] * self.relative + float(values.pop("X"))
                if "Y" in values:
                    self.position[1] = self.position[1] * self.relative + float(values.pop("Y"))

                if "Z" in values:
                    self.position[2] = self.position[2] * self.relative + float(values.pop("Z"))

                if "E" in values:
                    self.extrusion = self.extrusion * self.relative + float(values.pop("E"))

                if "F" in values:
                    self.feedrate = float(values.pop("F"))

                if values:
                    raise NotImplementedError(f"G0: {values}")

                end_p = np.copy(self.position)
                end_e = self.extrusion

                distance_p = np.linalg.norm(end_p - start_p)
                distance_e = end_e - start_e
                width = distance_e / distance_p if distance_p else 0

                lines.append((start_p, end_p, width, self.feedrate))

                # if self.position[2] > 2:
                #     break

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
                else:
                    self.relative = False
            elif command == "G91":
                if values:
                    raise NotImplementedError(f"G91: {values}")
                else:
                    self.relative = True
            elif command == "G92":
                if values == {"E": "0"}:
                    self.extrusion = 0
                else:
                    raise NotImplementedError(f"G92: {values}")
            elif command == "M92":
                LOG.debug(f"Ignore set steps per unit")
            elif command == "M106":
                LOG.debug(f"Ignore enable fan: {values}")
            elif command == "M107":
                LOG.debug(f"Ignore disable fan")
            elif command == "M117":
                LOG.info(f"Message: {line}")
            elif command == "M201":
                LOG.debug(f"Ignore adjust acceleration: {command} {values}")
            elif command == "M204":
                LOG.debug(f"Ignore adjust acceleration: {command} {values}")
            else:
                print(f"{n}: Ignoring {line}")

        return lines
