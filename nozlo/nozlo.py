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
import time
import math
import shutil
import logging
import colorsys
import datetime
from enum import IntEnum
from typing import Union
from pathlib import Path
from hashlib import sha1
from tempfile import NamedTemporaryFile

import appdirs
import numpy as np
from OpenGL import GL, GLU, GLUT
from OpenGL.GL.shaders import compileShader, compileProgram
import yaml

from nozlo.parser import \
    ParserVersionException, \
    Parser, Model, Bbox



LOG = logging.getLogger("nozlo")


PROFILE = True
UPDATE_DELAY_SECONDS = 1


POSITION_VECTOR_SIZE = 3
VERTEX_SIZE_BYTES = 4

CHANNELS = {
    "progress": {
        "label": "Progress (%)",
        "max": 100,
    },
    "feedrate": {
        "label": "Feedrate (mm/s)",
        "increment": 50,
    },
    "bandwidth": {
        "label": "Bandwidth (KB/s)",
        "increment": 25,
    },
    "fan_speed": {
        "label": "Fan speed (%)",
        "max": 100,
    },
    "tool_temp": {
        "label": "Tool temp. (°C)",
        "increment": 50,
    },
    "bed_temp": {
        "label": "Bed temp. (°C)",
        "increment": 50,
    },
}
DEFAULT_CHANNEL = "feedrate"



def mag_v3f(v):
    return np.linalg.norm(v)



def unit_v3f(v):
    """
    Unit vector
    """

    mag = mag_v3f(v)
    return v.copy() * (1 / mag if mag else 0)



def angle_v3f(v1, v2):
    """
    Angle between two vectors
    """

    return np.arccos(np.clip(np.dot(
        unit_v3f(v1),
        unit_v3f(v2),
    ), -1.0, 1.0))



class SpecialLayer(IntEnum):
    FIRST = -10  # First layer of model
    LAST = -20  # Last layer of model



class Camera():
    up_vector = np.array([0, 0, 1], dtype="float32")

    default_yaw = 45
    default_pitch = 30
    default_distance = 45

    default_ortho = False


    def __init__(self, aim=None, yaw=None, pitch=None, distance=None, ortho=None):
        self.position = np.array([0, 0, 0], dtype="float32")
        self.aim = np.array([0, 0, 0], dtype="float32") if aim is None else aim

        self.yaw = self.default_yaw if yaw is None else yaw
        self.pitch = self.default_pitch if pitch is None else pitch
        self.distance = self.default_distance if distance is None else distance

        self.view_angle = 50
        self.near_plane = 0.1
        self.far_plane = 1000

        self.ortho = self.default_ortho if ortho is None else ortho

        self.update()


    def __repr__(self):
        return f"<Camera: {self.position} -> {self.aim}>"


    def copy(self):
        return Camera(
            aim=self.aim.copy(),
            yaw=self.yaw,
            pitch=self.pitch,
            distance=self.distance,
            ortho=self.ortho,
        )


    def transition(self, other, t):
        tt = 1 - t

        self.aim[0] = other.aim[0] * t + self.aim[0] * tt
        self.aim[1] = other.aim[1] * t + self.aim[1] * tt
        self.aim[2] = other.aim[2] * t + self.aim[2] * tt

        self.yaw = other.yaw * t + self.yaw * tt
        self.pitch = other.pitch * t + self.pitch * tt
        self.distance = other.distance * t + self.distance * tt

        self.update()


    def reset_tumble(self):
        self.yaw = self.default_yaw
        self.pitch = self.default_pitch


    @property
    def up_safe(self):
        up = self.up_vector.copy()

        angle = angle_v3f(self.up_vector, self.position - self.aim)
        if angle == 0:
            up = np.array([
                -math.cos(math.radians(self.yaw)),
                math.sin(math.radians(self.yaw)),
                0
            ], dtype="float32")
        elif angle == math.pi:
            up = np.array([
                math.cos(math.radians(self.yaw)),
                -math.sin(math.radians(self.yaw)),
                0
            ], dtype="float32")

        return up


    def dolly(self, dx, dy, factor):
        position = self.position - self.aim
        horiz = unit_v3f(np.cross(self.up_safe, position))
        vert = unit_v3f(np.cross(position, horiz))
        dolly = horiz * dx + vert * dy
        self.aim += dolly * self.distance * factor


    def zoom(self, dz, factor):
        self.distance *= pow(factor, dz)


    def tumble(self, yaw, pitch):
        if yaw:
            self.yaw += yaw
        if pitch:
            self.pitch += pitch
            limit = 90
            self.pitch = float(np.clip(self.pitch, -limit, limit))


    def frame(self, bbox):
        self.aim = bbox.center
        self.distance = bbox.size


    def update(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        sin_pitch = math.sin(pitch)
        cos_pitch = math.cos(pitch)
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        position = np.array([1, 0, 0], dtype="float32")
        position = position.dot(np.array([
            [cos_pitch, 0, sin_pitch],
            [0, 1, 0],
            [-sin_pitch, 0, cos_pitch],
        ]))
        position = position.dot(np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ]))

        self.position = self.aim + position * self.distance



class Nozlo():
    name = "nozlo"

    reference_color = (0.3, 0.3, 0.3)
    background_color = (0.18, 0.18, 0.18)
    model_color_value: float = 0.6
    move_color_value: float = 0.25

    zoom_factor: float = 0.002
    scroll_factor: float = 1 / 0.9
    heat_lut_size: int = 256

    anim_duration: float = 0.2
    anim_ease: float = 2

    channels = CHANNELS

    def __init__(self):
        # Internal

        self.window = None

        self.program = None
        self.uniform_projection_matrix = None
        self.uniform_modelview_matrix = None

        self.cursor = np.array([0, 0], dtype="int")
        self.button = {v: None for v in range(5)}

        self.width = None
        self.height = None
        self.aspect = None

        self.draw_layer_min: int = 0
        self.draw_layer_max: int = 0
        self.reference_bbox = None

        self.model_path = None
        self.title = None

        self.model_layer_min = None
        self.model_layer_max = None

        self.line_buffer_position = None
        self.line_buffer_progress = None
        self.line_buffer_color = None
        self.locations = {
            "vertex_position": 0,
            "vertex_progress": 1,
            "vertex_color": 2,
        }
        self.line_array = None

        self.line_buffer_length = None
        self.reference_array_chunk = []
        self.model_layer_chunks = []

        self.state = None
        self.last_save_state = None
        self.last_update_time = None

        self.model_channel_max = {}
        self.model_channel_buffer = {}

        self.heat_lut_model = []
        self.heat_lut_move = []
        for i in range(self.heat_lut_size):
            value = i / (self.heat_lut_size - 1)
            self.heat_lut_model.append(self.heat_color(value, self.model_color_value))
            self.heat_lut_move.append(self.heat_color(value, self.move_color_value))

        # Reference

        self.lines_reference = None

        # Model

        self.model = None

        # Display

        self.camera = Camera()
        self.camera_target = Camera()

        self.layer: Union[int, SpecialLayer] = 0
        self.draw_single_layer = False
        self.explode = False
        self.explode_start = 0;
        self.explode_end = 0;
        self.explode_scale = 1

        self.explode_scale_current = self.explode_scale
        self.explode_scale_target = self.explode_scale

        self.channel = DEFAULT_CHANNEL

        # Animation

        self.clock: Union[float, None] = None
        self.elapsed: Union[float, None] = None
        self.anim_t: Union[float, None] = None
        self.anim_d: Union[float, None] = None
        self.reference_layer: Union[float, None] = None

        # Initialise

        self.init_files()

        self.load_reference()


    @staticmethod
    def heat_color(heat, saturation=0.8, value=0.6):
        hue = (4 - 5 * heat) / 6
        color = np.array(colorsys.hls_to_rgb(hue, 0.5, 1))
        mag = np.linalg.norm(np.array(color)) - 0.35

        color = np.array(colorsys.hls_to_rgb(hue, value, saturation))
        color *= pow(mag, -1)

        color = tuple(color.tolist())

        return color


    def max_channel_value(self, channel):
        max_value = self.channels[channel].get("max", None)
        if max_value is not None:
            return max_value

        max_value = getattr(self.model.max_segment, channel)
        increment = self.channels[channel]["increment"]
        return math.ceil(max_value / increment) * increment


    def init_files(self):
        self.cache_dir_path = Path(appdirs.user_cache_dir(self.name))
        self.config_dir_path = Path(appdirs.user_config_dir(self.name))

        for path in [self.cache_dir_path, self.config_dir_path]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                LOG.debug(f"Created directory `{path}`")

        self.config_path = self.config_dir_path / "nozlo.yml"

        if not self.config_path.exists():
            self.save_config()


    def init_program(self):
        vert = f"""\
#version 330 core

uniform mat4 modelview_matrix;
uniform mat4 projection_matrix;
uniform float explode_start;
uniform float explode_end;
uniform float explode_scale;

layout (location = {self.locations['vertex_position']}) in vec3 vertex_position;
layout (location = {self.locations['vertex_progress']}) in vec3 vertex_progress;
layout (location = {self.locations['vertex_color']}) in vec3 vertex_color;

out vec3 color;

void main() {{
  float explode;

explode = 0;
if (explode_start <= vertex_progress[0] && vertex_progress[0] < explode_end) {{
    explode += vertex_progress[0] - explode_start;
}}
if (explode_end <= vertex_progress[0]) {{
    explode += explode_end - explode_start;
}}


  color = vertex_color;
  gl_Position = projection_matrix * modelview_matrix * vec4(
    vertex_position[0],
    vertex_position[1],
    vertex_position[2] + explode * explode_scale,
    1.0
  );
}}
"""

        frag = """\
#version 330 core

in vec3 color;
out vec4 frag_color;

void main() {
  frag_color = vec4(color, 1.0);
}
"""

        self.program = compileProgram(
            compileShader(vert, GL.GL_VERTEX_SHADER),
            compileShader(frag, GL.GL_FRAGMENT_SHADER)
        )

        self.uniform_projection_matrix = GL.glGetUniformLocation(
            self.program, 'projection_matrix')
        self.uniform_modelview_matrix = GL.glGetUniformLocation(
            self.program, "modelview_matrix")
        self.uniform_explode_start = GL.glGetUniformLocation(
            self.program, "explode_start")
        self.uniform_explode_end = GL.glGetUniformLocation(
            self.program, "explode_end")
        self.uniform_explode_scale = GL.glGetUniformLocation(
            self.program, "explode_scale")


    def add_lines_reference(self, line_p, line_c):
        chunk_start = self.line_buffer_length

        for (start, end) in self.lines_reference:
            line_p += [start[0], start[1], start[2]]
            line_p += [end[0], end[1], end[2]]
            line_c += self.reference_color
            line_c += self.reference_color
            self.line_buffer_length += 2

        self.reference_array_chunk = [chunk_start, self.line_buffer_length]


    def update_model_color(self):
        line_c = []

        profile_start = time.time()
        line_c = self.model_channel_buffer[self.channel]
        LOG.debug(f"update model color iterate {time.time() - profile_start:0.2f}")

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_color)
        profile_start = time.time()
        GL.glBufferSubData(
            GL.GL_ARRAY_BUFFER,
            self.model_layer_chunks[0][0] * POSITION_VECTOR_SIZE * VERTEX_SIZE_BYTES,
            np.array(line_c, dtype='float32'),
        )
        LOG.debug(
            f"update model color GL.glBufferSubData {time.time() - profile_start:0.2f}")


    def init_line_buffer(self):
        self.line_array = GL.glGenVertexArrays(1)

        self.line_buffer_position = GL.glGenBuffers(1)
        self.line_buffer_progress = GL.glGenBuffers(1)
        self.line_buffer_color = GL.glGenBuffers(1)


    def load_line_buffer(self):
        LOG.debug("load line buffer start")
        start = time.time()

        line_p = []
        line_t = []
        line_c = []
        self.line_buffer_length = 0

        self.add_lines_reference(line_p, line_c)
        line_t += [0, 0, 0] * self.line_buffer_length

        def segment_layer(time_, layer):
            """
            Return the layer number plus the fraction of the layer completed.
            """

            return layer.number + (
                time_ - layer.max_segment.start_time) / layer.max_segment.duration

        profile_start = time.time()
        for layer in self.model:
            layer_chunk_start = self.line_buffer_length

            for segment in layer:
                line_p += [segment.start[0], segment.start[1], segment.start[2]]
                line_p += [segment.end[0], segment.end[1], segment.end[2]]
                line_t += [segment.start_time,
                           segment_layer(segment.start_time, layer), 0]
                line_t += [segment.start_time + segment.duration,
                           segment_layer(segment.start_time + segment.duration, layer), 0]

            self.line_buffer_length += 2 * len(layer)

            self.model_layer_chunks.append([
                layer_chunk_start,
                self.line_buffer_length
            ])
        LOG.debug(f"analyse model layer sizes {time.time() - profile_start:0.2f}")


        model_lines_buffer_length = (
            self.model_layer_chunks[-1][1] -
            self.model_layer_chunks[0][0]
        )

        line_c += self.model_channel_buffer[self.channel]

        GL.glBindVertexArray(self.line_array)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_position)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(line_p, dtype='float32'),
            GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(
            self.locations["vertex_position"], 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_progress)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(line_t, dtype='float32'),
            GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(
            self.locations["vertex_progress"], 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_color)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(line_c, dtype='float32'),
            GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(
            self.locations["vertex_color"], 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        duration = time.time() - start
        LOG.debug(f"load line buffer end {duration:0.2f}")

        GLUT.glutPostRedisplay()


    # Animation methods

    def anim_start(self):
        self.anim_t = 1.0


    def anim_camera(self):
        if self.anim_t:
            self.camera.transition(self.camera_target, self.anim_d)
        else:
            self.camera = self.camera_target.copy()


    def anim_explode(self):
        if self.anim_t:
            self.explode_scale_current = (
                self.explode_scale_current * (1 - self.anim_d) +
                self.explode_scale_target * self.anim_d
            )
        else:
            self.explode_scale_current = self.explode_scale_target


    # Drawing methods

    def clear(self):
        GL.glClearColor(*self.background_color, 0.0)
        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT |
            GL.GL_DEPTH_BUFFER_BIT
        )


    def render_3d_lines(self):

        # Options

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glDepthMask(True)

        # Projection

        if self.camera.ortho:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            h = self.camera.distance / 2
            w = h * self.aspect
            GL.glOrtho(
                -w, w, -h, h,
                self.camera.near_plane, self.camera.far_plane)
        else:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluPerspective(
                self.camera.view_angle, self.aspect,
                self.camera.near_plane, self.camera.far_plane)

        # Camera

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GLU.gluLookAt(
            *tuple(self.camera.position.tolist()),
            *tuple(self.camera.aim.tolist()),
            *tuple(self.camera.up_safe.tolist()),
        )

        # Shader

        GL.glUseProgram(self.program)

        GL.glUniformMatrix4fv(
            self.uniform_projection_matrix, 1, GL.GL_FALSE,
            GL.glGetFloatv(GL.GL_PROJECTION_MATRIX))
        GL.glUniformMatrix4fv(
            self.uniform_modelview_matrix, 1, GL.GL_FALSE,
            GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX))

        GL.glUniform1f(self.uniform_explode_start, self.explode_start)
        GL.glUniform1f(self.uniform_explode_end, self.explode_end)
        GL.glUniform1f(self.uniform_explode_scale, self.explode_scale_current)

        # Draw Layers

        GL.glLineWidth(1.0)

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)
        GL.glEnableVertexAttribArray(2)

        # Draw reference
        start = self.reference_array_chunk[0]
        end = self.reference_array_chunk[1]
        GL.glDrawArrays(GL.GL_LINES, start, end - start)

        # Draw model layers
        start = self.model_layer_chunks[self.draw_layer_min][0]
        end = self.model_layer_chunks[self.draw_layer_max][1]
        GL.glDrawArrays(GL.GL_LINES, start, end - start)

        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)

        GL.glUseProgram(0)


    @staticmethod
    def text(x, y, text):
        GL.glRasterPos2f(x, y)
        for ch in text:
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_9_BY_15, ord(ch))


    def show_loading_screen(self):
        # Projection

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluOrtho2D(0, self.width, 0, self.height)

        # Camera

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glEnable(GL.GL_BLEND)

        cx = 9
        cy = 15
        ly = int(cy * 1.25)

        message = f"Analysing model..."
        width = cx * len(message)

        x = self.width / 2 - width / 2
        y = ly * 2.5
        margin = 12

        GL.glColor3f(0.12, 0.12, 0.12)

        x1 = x - 2
        y1 = y - 3
        x2 = x1 + width
        y2 = y1 + cy
        x0 = x1 - margin
        y0 = y1 - margin / 2
        x3 = x2 + margin
        y3 = y2 + margin / 2

        GL.glBegin(GL.GL_POLYGON)
        GL.glVertex2f(x0, y1)
        GL.glVertex2f(x1, y0)
        GL.glVertex2f(x2, y0)
        GL.glVertex2f(x3, y1)
        GL.glVertex2f(x3, y2)
        GL.glVertex2f(x2, y3)
        GL.glVertex2f(x1, y3)
        GL.glVertex2f(x0, y2)
        GL.glEnd()

        GL.glColor4f(0.8, 0.8, 0.8, 1)
        self.text(x, y, message)

        GLUT.glutSwapBuffers()


    def render_2d_hud(self):
        # Projection

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluOrtho2D(0, self.width, 0, self.height)

        # Camera

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glDisable(GL.GL_DEPTH_TEST)

        margin = 16

        GL.glEnable(GL.GL_BLEND)

        cx = 9
        cy = 15
        ly = int(cy * 1.25)

        x = self.width - margin - cx * 9.5
        y = self.height - margin - ly * 0.5
        x1 = self.width - margin - cx * 9.5
        x2 = self.width - margin - cx * 6.5

        step = 6
        label = self.channels[self.channel]["label"].rjust(16)
        max_value = self.model_channel_max[self.channel]

        GL.glLineWidth(1.0)
        GL.glColor4f(0.8, 0.8, 0.8, 1)
        x = self.width - margin - cx * 16.5
        self.text(x, y, label)
        x = self.width - margin - cx * 9.5

        for i in range(step):
            t = i / (step - 1)
            value = max_value * (1 - t)
            y -= ly

            color = self.heat_color(value / max_value, value=self.model_color_value)
            GL.glColor3fv(color)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex2f(x1, y + cy * 0.3)
            GL.glVertex2f(x2, y + cy * 0.3)
            GL.glEnd()

            GL.glColor3fv(color)
            self.text(x, y, f"{value:9.1f}")

        x = margin + cx * 0.5
        y = self.height - margin - ly * 0.5

        GL.glColor4f(0.8, 0.8, 0.8, 1)
        self.text(x, y, f"Layer:{self.draw_layer_max:3d}")
        layer = self.model[self.draw_layer_max]
        duration = 0
        for layer in self.model_visible_layers():
            duration += layer.max_segment.duration

        y -= ly
        if layer.segments:
            self.text(x, y, f"Z:{layer.z:7.2f}")

        y -= ly
        duration = re.compile(r"^[0:]{,3}").sub("", str(
            datetime.timedelta(seconds=round(duration))
        ))
        self.text(x, y, f"T:{duration.rjust(7)}")


    # Flow methods

    def tick(self, idle=None):
        now = time.time()
        if self.clock is not None:
            self.elapsed = now - self.clock
        self.clock = now

        if (
                idle and
                self.state and
                self.state != self.last_save_state and
                self.last_update_time and
                self.clock > self.last_update_time + UPDATE_DELAY_SECONDS
        ):
            self.save_state()

        if self.elapsed is not None and self.anim_t is not None:
            last_t = self.anim_t
            self.anim_t -= self.elapsed / self.anim_duration
            self.anim_t = max(0, self.anim_t)
            anim_te = pow(self.anim_t, self.anim_ease)
            last_te = pow(last_t, self.anim_ease)
            self.anim_d = (last_te - anim_te) / last_te
            if self.anim_t == 0:
                self.anim_t = None
                self.update_state()

            self.anim_camera()
            self.anim_explode()


        self.clear()
        self.render_3d_lines()
        self.render_2d_hud()

        GLUT.glutSwapBuffers()


    def _tick(self, idle=None):
        try:
            self.tick(idle=idle)
        except KeyboardInterrupt:
            self.quit()
        except Exception as e:
            self.quit()
            raise e


    def idle(self):
        self._tick(idle=True)


    def display(self):
        self._tick()


    # Control methods

    def update_cursor(self, x, y):
        self.cursor[0] = x
        self.cursor[1] = y


    def set_channel(self, channel):
        if self.channel == channel:
            return

        self.channel = channel
        self.update_model_color()
        self.update_state()


    def nearest_layer_z(self, z):
        last_layer = None
        for layer in self.model:
            if layer.z == z:
                return layer.number
            if layer.z > z:
                if not layer.number:
                    return 0
                dist_below = z - last_layer.z
                layer_height = layer.z - last_layer.z
                return last_layer.number + dist_below / layer_height
            last_layer = layer
        return last_layer.number


    def nearest_layer_duration(self, z):
        last_layer = None
        last_layer_z = None
        for layer in self.model:
            layer_z = layer.max_segment.start_time * self.explode_scale
            if layer_z == z:
                return layer.number
            if layer_z > z:
                if not layer.number:
                    return 0
                dist_below = z - last_layer_z
                layer_height = layer_z - last_layer_z
                return last_layer.number + dist_below / layer_height
            last_layer = layer
            last_layer_z = layer_z
        return last_layer.number


    def current_layer(self):
        if self.layer == SpecialLayer.FIRST:
            return self.model[0]
        if self.layer == SpecialLayer.LAST:
            return self.model[-1]
        return self.model[self.layer]


    def update_camera_position(self):
        self.camera.update()
        self.camera_target = self.camera.copy()
        self.update_state()


    def keyboard(self, key, x, y):
        if ord(key) == 27 or key == b'q':
            self.quit()

        if key == b'a':
            self.frame_reference()
        if key == b'f':
            self.frame_visible_model()
        if key == b'o':
            self.camera.ortho = not self.camera.ortho
            self.update_camera_position()

        if key == b's':
            self.update_model_draw(single=not self.draw_single_layer)
        if key == b'x':
            self.update_model_draw(explode=not self.explode)

        if key == b'0':
            self.set_channel("progress")
        if key == b'1':
            self.set_channel("feedrate")
        if key == b'2':
            self.set_channel("bandwidth")
        if key == b'3':
            self.set_channel("fan_speed")
        if key == b'4':
            self.set_channel("tool_temp")
        if key == b'5':
            self.set_channel("bed_temp")

        if key == b'h':
            self.camera_target.pitch = 0
            self.camera_target.yaw = 180
            self.anim_start()
        if key == b'j':
            self.camera_target.pitch = 0
            self.camera_target.yaw = 90
            self.anim_start()
        if key == b'k':
            self.camera_target.pitch = 0
            self.camera_target.yaw = 0
            self.anim_start()
        if key == b'l':
            self.camera_target.pitch = 0
            self.camera_target.yaw = -90
            self.anim_start()

        if key == b'u':
            self.camera_target.pitch = -90
            self.camera_target.yaw = 90
            self.anim_start()
        if key == b'i':
            self.camera_target.pitch = 90
            self.camera_target.yaw = 90
            self.anim_start()

        if key == b'y':
            self.camera_target.reset_tumble()
            self.anim_start()

        if key == b'-':
            self.explode_scale -= 0.1;
            self.camera.zoom(1, self.scroll_factor)
            self.update_camera_position()
        if key == b'=':
            self.explode_scale += 0.1;
            self.camera.zoom(-1, self.scroll_factor)
            self.update_camera_position()

        if key == b',':
            self.explode_scale /= self.scroll_factor
            self.explode_scale_current = self.explode_scale * self.explode
            self.explode_scale_target = self.explode_scale_current
            self.update_model_draw()
        if key == b'.':
            self.explode_scale *= self.scroll_factor
            self.explode_scale_current = self.explode_scale * self.explode
            self.explode_scale_target = self.explode_scale_current
            self.update_model_draw()

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def special(self, key, x, y):
        if key == GLUT.GLUT_KEY_HOME:
            if self.layer == SpecialLayer.FIRST:
                self.update_model_draw(layer=0)
            else:
                self.update_model_draw(layer=SpecialLayer.FIRST)
        if key == GLUT.GLUT_KEY_END:
            self.update_model_draw(layer=SpecialLayer.LAST)

        if key == GLUT.GLUT_KEY_DOWN:
            self.update_model_draw(layer=self.draw_layer_max - 1)
        if key == GLUT.GLUT_KEY_UP:
            self.update_model_draw(layer=self.draw_layer_max + 1)

        if key == GLUT.GLUT_KEY_PAGE_DOWN:
            self.update_model_draw(layer=self.draw_layer_max - len(self.model) // 5)
        if key == GLUT.GLUT_KEY_PAGE_UP:
            self.update_model_draw(layer=self.draw_layer_max + len(self.model) // 5)

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def model_visible_layers(self):
        """
        Generate of currently displayed model layers.
        """

        for n in range(self.draw_layer_min, self.draw_layer_max + 1):
            yield self.model[n]


    def frame_visible_model(self, anim=True):
        bbox = Bbox()

        for layer in self.model_visible_layers():
            if layer.bbox_model:
                bbox.update(layer.bbox_model.min)
                bbox.update(layer.bbox_model.max)

        if not bbox:
            for layer in self.model_visible_layers():
                bbox.update(layer.bbox_total.min)
                bbox.update(layer.bbox_total.max)

        if self.explode:
            bbox.max[2] += (
                self.explode_end - self.explode_start
            ) * self.explode_scale

        self.camera_target.frame(bbox)

        if anim:
            self.anim_start()
        else:
            self.camera = self.camera_target.copy()
            self.update_camera_position


    def frame_reference(self):
        self.camera_target.frame(self.reference_bbox)
        self.anim_start()


    def move_camera(
            self,
            yaw=0,
            pitch=0,
            dolly_horiz=0,
            dolly_vert=0,
            scroll=0
    ):
        self.camera.dolly(dolly_horiz, dolly_vert, self.zoom_factor)
        self.camera.zoom(scroll, self.scroll_factor)
        self.camera.tumble(yaw, pitch)

        self.update_camera_position()


    def update_model_draw(
            self,
            layer: Union[None, int, SpecialLayer] = None,
            single: Union[None, bool] = None,
            explode: Union[None, bool] = None
    ):
        """
        `layer`: absolute layer number or `-1` for last layer.
        """

        draw_layer_max_ = self.draw_layer_max
        draw_layer_min_ = self.draw_layer_min

        if layer is not None:
            self.layer = layer

        if self.layer == SpecialLayer.FIRST:
            self.draw_layer_max = self.model_layer_min
        elif self.layer == SpecialLayer.LAST:
            self.draw_layer_max = self.model_layer_max
        else:
            self.layer = max(0, min(len(self.model) - 1, self.layer))
            self.draw_layer_max = self.layer

        if single is not None:
            self.draw_single_layer = single

        if explode is not None:
            if explode != self.explode:
                self.explode_scale_current = self.explode_scale * self.explode
                self.explode_scale_target = self.explode_scale * explode
                self.explode = explode
                self.frame_visible_model()

        self.draw_layer_min = self.draw_layer_max if self.draw_single_layer else 0

        if (
                self.draw_layer_min != draw_layer_min_ or
                self.draw_layer_max != draw_layer_max_
        ):
            layer = self.model[self.draw_layer_max]
            self.explode_start = layer.max_segment.start_time
            self.explode_end = layer.max_segment.start_time + layer.max_segment.duration
            self.update_state()



    def mouse(self, button, state, x, y):
        self.button[button] = not state

        if button == 3 and state == 0:
            self.move_camera(scroll=-1)
        if button == 4 and state == 0:
            self.move_camera(scroll=1)

        GLUT.glutPostRedisplay()


    def motion(self, x, y):
        """
        On movement when a button is pressed
        """

        position = np.array([x, y], dtype="int")
        move = position - self.cursor

        yaw = 0
        pitch = 0
        dolly_horiz = 0
        dolly_vert = 0

        if self.button[0]:
            yaw = float(move[0]) * 0.5
            pitch = float(move[1]) * 0.5

        if self.button[2]:
            dolly_horiz = -float(move[0])
            dolly_vert = float(move[1])

        self.move_camera(
            yaw=yaw,
            pitch=pitch,
            dolly_horiz=dolly_horiz,
            dolly_vert=dolly_vert,
        )

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def reshape(self, w, h):
        self.width = w
        self.height = h
        self.aspect = w / h if h else 1
        GL.glViewport(0, 0, w, h)
        GLUT.glutPostRedisplay()


    def save_model_cache(self, model, path):
        with path.open("wb") as fp:
            model.pack_into(fp)


    def load_model_cache(self, path):
        with path.open("rb") as fp:
            model = Model.unpack_from(fp)

        return model


    def load_model(self, gcode_path, cache=True):

        if PROFILE:
            LOG.debug("load model start")
            profile_start = time.time()

        self.title = f"Nozlo: {gcode_path.name}"
        self.model_path = gcode_path.resolve()

        hasher = sha1()
        hasher.update(str(self.model_path).encode())
        model_path_hash = hasher.hexdigest()[:7]
        model_cache_path = (
            self.cache_dir_path /
            f"{self.model_path.stem}.{model_path_hash}.cache"
        )

        self.model = None
        if cache:
            if model_cache_path.exists():
                cache_mtime = model_cache_path.stat().st_mtime
                gcode_mtime = gcode_path.stat().st_mtime
                if cache_mtime >= gcode_mtime:
                    try:
                        cache_start = time.time()
                        self.model = self.load_model_cache(model_cache_path)
                        cache_duration = time.time() - cache_start
                    except ParserVersionException as e:
                        LOG.debug(e)
                    except Exception as e:
                        raise e
                        LOG.error(e)
                    else:
                        if PROFILE:
                            LOG.debug(
                                f"Read from cache in {cache_duration:0.2f}: "
                                f"`{model_cache_path}`")
                else:
                    LOG.debug(f"Cache {cache_mtime} is older than G-code {gcode_mtime}.")

        if self.model is None:
            with gcode_path.open() as fp:
                gcode_start = time.time()
                self.model = Parser().parse(fp)
                gcode_duration = time.time() - gcode_start

            if PROFILE:
                LOG.debug(f"Read from gcode in {gcode_duration:0.2f}: `{gcode_path}`")

            calc_start = time.time()
            layer_start_time = 0
            for layer in self.model:
                layer.calc_bounds(start_time=layer_start_time)
                layer_start_time += layer.max_segment.duration

            self.model.calc_bounds()
            calc_duration = time.time() - calc_start
            if PROFILE:
                LOG.debug(f"Calc layer bounds in {calc_duration:0.2f}")

            self.save_model_cache(self.model, model_cache_path)
            cache_mtime = model_cache_path.stat().st_mtime
            LOG.debug(f"Saved model cache `{model_cache_path}` {cache_mtime}")

        self.model_layer_min = None
        self.model_layer_max = None
        for layer in self.model:
            if layer.bbox_model:
                if self.model_layer_min is None:
                    self.model_layer_min = layer.number
                self.model_layer_max = layer.number

        LOG.info(f"Loaded {len(self.model)} layers, ({self.model_layer_max - self.model_layer_min + 1} containing model).")
        if PROFILE:
            LOG.debug(f"load model end {time.time() - profile_start:0.2f}")

        state = None
        if cache:
            config = self.load_config()
            key = str(self.model_path)
            state = config["models"].get(key, None)


        self.analyse_model()
        self.update_model_draw()

        if state:
            self.set_state(state)
        else:
            self.update_model_draw(layer=SpecialLayer.LAST)
            self.frame_visible_model(anim=False)


    def analyse_model(self):
        self.model_layer_chunks = []

        for channel in self.channels:
            self.model_channel_max[channel] = self.max_channel_value(channel)
            self.model_channel_buffer[channel] = []

        def add_color(channel, value, model, n=1):
            value_index = math.floor(value * (self.heat_lut_size - 1))
            color = (
                self.heat_lut_model[value_index] if model else
                self.heat_lut_move[value_index])
            for _i in range(n):
                self.model_channel_buffer[channel] += color

        profile_start = time.time()
        for layer in self.model:
            for segment in layer:
                for channel in self.channels:
                    if channel == "progress":
                        add_color(
                            channel,
                            (segment.start_time - layer.max_segment.start_time) *
                            1 / layer.max_segment.duration,
                            segment.width
                        )
                        add_color(
                            channel,
                            (segment.start_time + segment.duration -
                             layer.max_segment.start_time) *
                            1 / layer.max_segment.duration,
                            segment.width
                        )
                    else:
                        add_color(
                            channel,
                            getattr(segment, channel) /
                            self.model_channel_max[channel],
                            segment.width,
                            n=2
                        )

        LOG.debug(f"analyse model {time.time() - profile_start:0.2f}")


    def load_reference(self):
        reference_width = 214
        reference_length = 214
        grid_step = 30
        grid_x = 9
        grid_y = 9
        cross_xy = 10

        self.lines_reference = [
            [
                [0, 0, 0],
                [reference_width, 0, 0],
            ],
            [
                [reference_width, 0, 0],
                [reference_width, reference_length, 0],
            ],
            [
                [reference_width, reference_length, 0],
                [0, reference_length, 0],
            ],
            [
                [0, reference_length, 0],
                [0, 0, 0],
            ],
        ]

        cx = reference_width / 2
        cy = reference_length / 2

        for x in range(grid_x):
            px = cx + grid_step * (x - (grid_x - 1) / 2)
            for y in range(grid_y):
                py = cy + grid_step * (y - (grid_y - 1) / 2)
                self.lines_reference += [
                    [
                        [px - cross_xy / 2, py, 0],
                        [px + cross_xy / 2, py, 0],
                    ],
                    [
                        [px, py - cross_xy / 2, 0],
                        [px, py + cross_xy / 2, 0],
                    ],
                ]

        self.reference_bbox = Bbox()
        for (start, end) in self.lines_reference:
            self.reference_bbox.update(start)
            self.reference_bbox.update(end)


    def set_state(self, state) -> None:
        try:
            self.camera.aim[0] = state["aim"][0]
            self.camera.aim[1] = state["aim"][1]
            self.camera.aim[2] = state["aim"][2]
        except KeyError:
            pass

        try:
            self.camera.yaw = state["yaw"]
        except KeyError:
            pass

        try:
            self.camera.pitch = state["pitch"]
        except KeyError:
            pass
        try:
            self.camera.distance = state["distance"]
        except KeyError:
            pass
        try:
            self.camera.ortho = state["ortho"]
        except KeyError:
            pass

        try:
            self.layer = int(state["layer"])
        except KeyError:
            pass
        try:
            self.draw_single_layer = state["single"]
        except KeyError:
            pass
        try:
            self.explode = state["explode"]
        except KeyError:
            pass
        try:
            self.explode_scale = state["explode_scale"]
        except KeyError:
            pass

        self.explode_scale_current = self.explode_scale * self.explode
        self.explode_scale_target = self.explode_scale * self.explode

        try:
            self.channel = state["channel"]
        except KeyError:
            pass

        self.update_model_draw()
        self.update_camera_position()


    def update_state(self) -> None:
        self.state = {
            "aim": [float(v) for v in self.camera.aim],
            "yaw": self.camera.yaw,
            "pitch": self.camera.pitch,
            "distance": self.camera.distance,
            "ortho": self.camera.ortho,
            "layer": int(self.layer),
            "single": self.draw_single_layer,
            "explode": self.explode,
            "explode_scale": self.explode_scale,
            "channel": self.channel,
        }
        self.last_update_time = time.time()


    @staticmethod
    def default_config():
        return {
            "models": {}
        }


    def load_config(self):
        config = None
        try:
            config = yaml.safe_load(self.config_path.read_text())
        except:
            LOG.error(f"Failed to load `{self.config_path}`.")
        else:
            if config is None:
                LOG.error(f"Empty config `{self.config_path}`.")

        if config is None:
            LOG.error(f"Deleted `{self.config_path}`.")
            self.config_path.unlink()
            config = self.default_config()

        return config


    def save_config(self, config=None):
        if config is None:
            config = self.default_config()

        with NamedTemporaryFile("w", suffix=".conf", delete=False) as temp:
            yaml.dump(config, temp)

        shutil.move(temp.name, self.config_path)
        LOG.debug(f"Saved config `{self.config_path}`")


    def save_state(self):
        config = self.load_config()
        key = str(self.model_path)
        config["models"][key] = self.state
        self.save_config(config)
        self.last_save_state = self.state


    def quit(self):
        self.save_state()
        GLUT.glutLeaveMainLoop()


    def run(self):
        GLUT.glutInit()
        GLUT.glutSetOption(GLUT.GLUT_MULTISAMPLE, 4)
        GLUT.glutInitDisplayMode(
            GLUT.GLUT_DOUBLE |
            GLUT.GLUT_DEPTH |
            GLUT.GLUT_MULTISAMPLE
        )
        self.width = 1200
        self.height = 720

        GLUT.glutInitWindowSize(self.width, self.height)
        GLUT.glutInitWindowPosition(100, 100)

        self.window = GLUT.glutCreateWindow(self.title)

        self.init_program()

        self.clear()
        self.show_loading_screen()
        self.init_line_buffer()
        self.load_line_buffer()

        GLUT.glutDisplayFunc(self.display)
        GLUT.glutIdleFunc(self.idle)
        GLUT.glutReshapeFunc(self.reshape)
        GLUT.glutKeyboardFunc(self.keyboard)
        GLUT.glutSpecialFunc(self.special)
        GLUT.glutMouseFunc(self.mouse)
        GLUT.glutMotionFunc(self.motion)
        GLUT.glutPassiveMotionFunc(self.motion)

        GLUT.glutMainLoop()
