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
import struct
import logging
from typing import Tuple, BinaryIO, ClassVar
from dataclasses import dataclass

import numpy as np



LOG = logging.getLogger("parser")



@dataclass
class Segment:
    """
    G-code line segment
    """

    struct_format: ClassVar[str] = "fff fff f f f f f"

    start: Tuple[float, float, float]
    end: Tuple[float, float, float]
    width: float
    feedrate: float
    tool_temp: float
    bed_temp: float
    fan_speed: float


    def pack(self):
        return struct.pack(
            self.struct_format,

            self.start[0],
            self.start[1],
            self.start[2],

            self.end[0],
            self.end[1],
            self.end[2],

            self.width,
            self.feedrate,
            self.tool_temp,
            self.bed_temp,
            self.fan_speed,
        )

    @classmethod
    def unpack(cls, buf: bytes):
        segment_data = struct.unpack(cls.struct_format, buf)
        (
            start_x,
            start_y,
            start_z,

            end_x,
            end_y,
            end_z,

            width,
            feedrate,
            tool_temp,
            bed_temp,
            fan_speed,
        ) = segment_data

        return cls(
            start=[start_x, start_y, start_z],
            end=[end_x, end_y, end_z],
            width=width,
            feedrate=feedrate,
            tool_temp=tool_temp,
            bed_temp=bed_temp,
            fan_speed=fan_speed,
        )



class Bbox:
    struct_format = "f fff fff"

    def __init__(self, count=None, min_=None, max_=None):
        if count is None:
            count = 0
        if min_ is None:
            min_ = [0, 0, 0]
        if max_ is None:
            max_ = [0, 0, 0]

        self.count = count
        self.min = np.array(min_, dtype="float32")
        self.max = np.array(max_, dtype="float32")


    def __bool__(self):
        return bool(self.count)


    def update(self, point):
        for i in range(3):
            if not self.count:
                self.min[i] = point[i]
                self.max[i] = point[i]
            else:
                self.min[i] = min(self.min[i], point[i])
                self.max[i] = max(self.max[i], point[i])
        self.count += 1


    @property
    def center(self):
        return ((self.max + self.min) / 2).copy()


    @property
    def size(self):
        return float(np.linalg.norm(self.max - self.min))


    def pack(self):
        return struct.pack(
            self.struct_format,

            self.count,

            self.min[0],
            self.min[1],
            self.min[2],

            self.max[0],
            self.max[1],
            self.max[2],
        )


    @classmethod
    def unpack(cls, buf: bytes):
        segment_data = struct.unpack(cls.struct_format, buf)
        (
            count,

            min_x,
            min_y,
            min_z,

            max_x,
            max_y,
            max_z,
        ) = segment_data

        return cls(
            count=count,
            min_=[min_x, min_y, min_z],
            max_=[max_x, max_y, max_z],
        )



class Layer:
    """
    G-code layer
    """

    struct_format = "IfI"


    def __init__(
            self,
            number: int,
            z: float,
    ):
        self.number = number
        self.z = z

        self.segments = []

        self.bbox_model = None
        self.bbox_total = None
        self.max_segment = None


    def calc_bounds(self):
        max_width = None
        max_feedrate = None
        max_tool_temp = None
        max_bed_temp = None
        max_fan_speed = None

        self.bbox_model = Bbox()
        self.bbox_total = Bbox()

        for segment in self.segments:
            if segment.width and segment.end[0] >= 0 and segment.end[1] >= 0:
                # Extrusion ends in build volume
                self.bbox_model.update(segment.start)
                self.bbox_model.update(segment.end)
            self.bbox_total.update(segment.start)
            self.bbox_total.update(segment.end)

            if max_width is None:
                max_width = segment.width
                max_feedrate = segment.feedrate
                max_tool_temp = segment.tool_temp
                max_bed_temp = segment.bed_temp
                max_fan_speed = segment.fan_speed
            else:
                max_width = max(max_width, segment.width)
                max_feedrate = max(max_feedrate, segment.feedrate)
                max_tool_temp = max(max_tool_temp, segment.tool_temp)
                max_bed_temp = max(max_bed_temp, segment.bed_temp)
                max_fan_speed = max(max_fan_speed, segment.fan_speed)

        self.max_segment = Segment(
            start=[0, 0, 0],
            end=[0, 0, 0],
            width=max_width,
            feedrate=max_feedrate,
            tool_temp=max_tool_temp,
            bed_temp=max_bed_temp,
            fan_speed=max_fan_speed,
        )


    def pack_into(self, out: BinaryIO):
        out.write(struct.pack(
            self.struct_format,
            self.number,
            self.z,
            len(self.segments),
        ))
        for segment in self.segments:
            out.write(segment.pack())
        out.write(self.bbox_model.pack())
        out.write(self.bbox_total.pack())
        out.write(self.max_segment.pack())


    @classmethod
    def unpack_from(cls, in_: BinaryIO):
        layer_size = struct.calcsize(cls.struct_format)
        segment_size = struct.calcsize(Segment.struct_format)
        bbox_size = struct.calcsize(Bbox.struct_format)

        buf = in_.read(layer_size)
        if not buf:
            return None

        layer_data = struct.unpack(cls.struct_format, buf)
        (number, z, length) = layer_data
        layer = cls(
            number=number,
            z=z,
        )
        for s in range(length):
            buf = in_.read(segment_size)
            if not buf:
                break

            layer.segments.append(Segment.unpack(buf))

        buf = in_.read(bbox_size)
        if not buf:
            return None

        layer.bbox_model = Bbox.unpack(buf)

        buf = in_.read(bbox_size)
        if not buf:
            return None

        layer.bbox_total = Bbox.unpack(buf)

        buf = in_.read(segment_size)
        if not buf:
            return None

        layer.max_segment = Segment.unpack(buf)

        return layer



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


    @staticmethod
    def parse_line(line, n=None):
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

        initial_z_value = 0
        layers.append(Layer(
            number=len(extrusion_z_values),
            z=initial_z_value
        ))
        extrusion_z_values.add(initial_z_value)

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

                if (x_value is not None or y_value is not None):
                    if e_value and e_value > 0:
                        if self.position[2] not in extrusion_z_values:
                            layers.append(Layer(
                                number=len(extrusion_z_values),
                                z=self.position[2]
                            ))
                            extrusion_z_values.add(self.position[2])
                        layers[-1].model = True

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
                LOG.debug("Ignore set steps per unit")
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
