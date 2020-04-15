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

import time
import math
import logging
import colorsys

import numpy as np
from OpenGL import GL, GLU, GLUT
from OpenGL.GL.shaders import compileShader, compileProgram

from nozlo.parser import Parser


LOG = logging.getLogger("nozlo")



POSITION_VECTOR_SIZE = 3



class Nozlo():
    bed_color = (0.3, 0.3, 0.3)
    background_color = (0.18, 0.18, 0.18)
    high_feedrate = 100 * 60

    up_vector = np.array([0, 0, 1], dtype="float32")

    def __init__(self):
        self.title = None
        self.window = None

        self.program = None
        self.projection_matrix_uniform = None
        self.modelview_matrix_uniform = None

        self.cursor = np.array([0, 0], dtype="int")
        self.button = {v: None for v in range(5)}

        self.aspect = None

        self.aim = np.array([0, 0, 0], dtype="float32")
        self.yaw = 45;
        self.pitch = 30;

        self.distance = 45;

        self.view_angle = 50
        self.near_plane = 0.1
        self.far_plane = 1000

        self.ortho = False

        self.camera = np.array([0, 0, 0], dtype="float32")

        self.model = None
        self.lines_bed = None

        self.layers = None

        self.draw_layer_max = None
        self.draw_layer_min = None
        self.draw_single_layer = False

        self.model_center = None
        self.model_size = None
        self.bed_center = None
        self.bed_size = None
        self.model_layer_min = None
        self.model_layer_max = None

        self.line_buffer_length = None
        self.line_buffer_position = None
        self.line_buffer_color = None
        self.line_array = None

        self.bed_array_chunk = []
        self.model_layer_chunks = []

        self.max_feedrate = 0

        self.load_bed()


    @staticmethod
    def unit(v):
        """
        Unit vector
        """
        return v / np.linalg.norm(v)


    @classmethod
    def angle(cls, v1, v2):
        """
        Angle between two vectors
        """
        return np.arccos(np.clip(np.dot(
            cls.unit(v1),
            cls.unit(v2),
        ), -1.0, 1.0))


    def init_program(self):
        vert = """\
#version 330 core

uniform mat4 modelview_matrix;
uniform mat4 projection_matrix;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;

out vec3 color;

void main() {
  color = vertex_color;
  gl_Position = projection_matrix * modelview_matrix * vec4(vertex_position, 1.0);
}"""

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

        self.projection_matrix_uniform = GL.glGetUniformLocation(
            self.program, 'projection_matrix')
        self.modelview_matrix_uniform = GL.glGetUniformLocation(
            self.program, "modelview_matrix")


    def add_lines_bed(self, line_p, line_c):
        chunk_start = self.line_buffer_length

        for (start, end) in self.lines_bed:
            line_p += [start[0], start[1], start[2]]
            line_p += [end[0], end[1], end[2]]
            line_c += self.bed_color
            line_c += self.bed_color
            self.line_buffer_length += 2

        self.bed_array_chunk = [chunk_start, self.line_buffer_length]


    @staticmethod
    def hue(value, max_value):
        t = value / max_value if max_value else 0
        return 0.8 * (1 - t)


    def add_lines_model(self, line_p, line_c):
        self.model_layer_chunks = []

        self.max_feedrate = 0
        for layer in self.model:
            for segment in layer.segments:
                self.max_feedrate = max(self.max_feedrate, segment.feedrate)

        increment = 50 * 60
        self.max_feedrate = math.ceil(self.max_feedrate / increment) * increment

        for layer in self.model:
            layer_chunk_start = self.line_buffer_length
            for segment in layer.segments:

                line_p += [segment.start[0], segment.start[1], segment.start[2]]
                line_p += [segment.end[0], segment.end[1], segment.end[2]]

                color = colorsys.hsv_to_rgb(
                    self.hue(segment.feedrate, self.max_feedrate),
                    1,
                    0.8 if segment.width else 0.5
                )

                line_c += color
                line_c += color

                self.line_buffer_length += 2

            self.model_layer_chunks.append([
                layer_chunk_start,
                self.line_buffer_length
            ])

        self.update_model_draw()


    def init_line_buffer(self):
        self.line_array = GL.glGenVertexArrays(1)
        self.line_buffer_position = GL.glGenBuffers(1)
        self.line_buffer_color = GL.glGenBuffers(1)


    def load_line_buffer(self):
        LOG.debug("load line buffer start")
        start = time.time()

        line_p = []
        line_c = []
        self.line_buffer_length = 0

        self.add_lines_bed(line_p, line_c)
        self.add_lines_model(line_p, line_c)

        GL.glBindVertexArray(self.line_array)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_position)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(line_p, dtype='float32'),
            GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_color)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(line_c, dtype='float32'),
            GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        duration = time.time() - start
        LOG.debug(f"load line buffer end {duration:0.2f}")

        GLUT.glutPostRedisplay()


    def render_3d_lines(self):

        # Projection

        if self.ortho:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            h = self.distance / 2
            w = h * self.aspect
            GL.glOrtho(-w, w, -h, h, self.near_plane, self.far_plane)
        else:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glLoadIdentity()
            GLU.gluPerspective(self.view_angle, self.aspect, self.near_plane, self.far_plane)

        # Camera

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        up2 = self.up_vector
        if not self.angle(self.up_vector, self.camera - self.aim):
            up2 = np.array([0, 1, 0])
        GLU.gluLookAt(
            self.camera[0], self.camera[1], self.camera[2],
            self.aim[0], self.aim[1], self.aim[2],
            up2[0], up2[1], up2[2],
        )

        # Shader

        GL.glUseProgram(self.program)

        GL.glUniformMatrix4fv(
            self.projection_matrix_uniform, 1, GL.GL_FALSE,
            GL.glGetFloatv(GL.GL_PROJECTION_MATRIX))
        GL.glUniformMatrix4fv(
            self.modelview_matrix_uniform, 1, GL.GL_FALSE,
            GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX))

        # Options

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glDepthMask(True)

        # Background

        GL.glClearColor(*self.background_color, 0.0)
        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT |
            GL.GL_DEPTH_BUFFER_BIT
        )

        # Draw Layers

        GL.glLineWidth(1.0)

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(1)

        # Draw bed
        start = self.bed_array_chunk[0]
        end = self.bed_array_chunk[1]
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

        GL.glLineWidth(1.0)

        GL.glColor4f(0.8, 0.8, 0.8, 1)
        self.text(x, y, "Feedrate ")

        step = 6
        max_value = self.max_feedrate / 60
        for i in range(step):
            t = i / (step - 1)
            value = max_value * (1 - t)
            hue = self.hue(value, max_value)
            y -= ly

            GL.glColor3fv(colorsys.hsv_to_rgb(hue, 1, 0.8))
            GL.glBegin(GL.GL_LINES)
            GL.glVertex2f(x1, y + cy * 0.3)
            GL.glVertex2f(x2, y + cy * 0.3)
            GL.glEnd()

            GL.glColor3fv(colorsys.hsv_to_rgb(hue, 0.4, 0.8))
            self.text(x, y, f"{value:9.1f}")

        x = margin + cx * 0.5
        y = self.height - margin - ly * 0.5

        GL.glColor4f(0.8, 0.8, 0.8, 1)
        self.text(x, y, f"Layer:{self.draw_layer_max:3d}")
        y -= ly
        layer = self.model[self.draw_layer_max]
        if layer.segments:
            self.text(x, y, f"Z:{layer.z:7.2f}")
        else:
            self.text(x, y, f"Z:   none")


    def display(self):
        LOG.debug("display start")
        start = time.time()

        self.render_3d_lines()
        self.render_2d_hud()

        GLUT.glutSwapBuffers()

        duration = time.time() - start
        LOG.debug(f"display end {duration:0.2f}")


    def _display(self):
        try:
            self.display()
        except KeyboardInterrupt:
            GLUT.glutLeaveMainLoop()


    def update_cursor(self, x, y):
        self.cursor[0] = x
        self.cursor[1] = y


    def keyboard(self, key, x, y):
        if ord(key) == 27 or key == b'q':
            GLUT.glutLeaveMainLoop()

        if key == b'a':
            self.frame_bed()
        if key == b'f':
            self.frame_model()
        if key == b's':
            self.update_model_draw(single=not self.draw_single_layer)
        if key == b'o':
            self.ortho = not self.ortho

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def special(self, key, x, y):
        if key == GLUT.GLUT_KEY_HOME:
            if self.draw_layer_max == self.model_layer_min:
                self.update_model_draw(layer=0)
            else:
                self.update_model_draw(layer=self.model_layer_min)
        if key == GLUT.GLUT_KEY_END:
            if self.draw_layer_max == self.model_layer_max:
                self.update_model_draw(layer=-1)
            else:
                self.update_model_draw(layer=self.model_layer_max)

        if key == GLUT.GLUT_KEY_DOWN:
            self.update_model_draw(layer=self.draw_layer_max - 1)
        if key == GLUT.GLUT_KEY_UP:
            self.update_model_draw(layer=self.draw_layer_max + 1)

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def update_camera_position(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        camera = np.array([1, 0, 0], dtype="float32")
        camera = camera.dot(np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]))
        camera = camera.dot(np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]))

        self.camera = self.aim + camera * self.distance


    def frame_model(self):
        self.aim = self.model_center.copy()
        self.distance = self.model_size

        self.update_camera_position()


    def frame_bed(self):
        self.aim = self.bed_center.copy()
        self.distance = self.bed_size

        self.update_camera_position()


    def move_aim(self, aim):
        self.aim = np.array(aim, dtype="float32")
        self.update_camera_position()


    def move_camera(
            self,
            yaw=0,
            pitch=0,
            dolly_horiz=0,
            dolly_vert=0,
            scroll=0
    ):
        camera = self.camera - self.aim
        horiz = self.unit(np.cross(self.up_vector, camera))
        vert = self.unit(np.cross(camera, horiz))
        dolly = horiz * dolly_horiz + vert * dolly_vert
        self.aim += dolly * self.distance * 0.002

        if scroll < 0:
            self.distance *= 0.9
        if scroll > 0:
            self.distance /= 0.9

        if yaw:
            self.yaw += yaw
        if pitch:
            self.pitch += pitch
            limit = 90 * 0.999
            self.pitch = np.clip(self.pitch, -limit, limit)

        self.update_camera_position()


    def update_model_draw(self, layer=None, single=None):
        """
        `layer`: absolute layer number or `-1` for last layer.
        """

        last = len(self.layers) - 1
        target_min = self.draw_layer_min
        target_max = self.draw_layer_max

        if layer is not None:
            if layer == -1:
                target_max = last
            else:
                target_max = max(0, min(last, layer))

        if single is not None:
            self.draw_single_layer = single

        if self.draw_single_layer:
            target_min = target_max
        else:
            target_min = 0

        if (
                target_max != self.draw_layer_max or
                target_min != self.draw_layer_min
        ):
            self.draw_layer_max = target_max
            self.draw_layer_min = target_min


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


    @staticmethod
    def bbox_init():
        return {
            "center": np.array([0, 0, 0], dtype="float32"),
            "min": np.array([0, 0, 0], dtype="float32"),
            "max": np.array([0, 0, 0], dtype="float32"),
            "_count": 0,
        }


    @staticmethod
    def bbox_update(bbox, point):
        for i in range(3):
            bbox["center"][i] += point[i]
            if not bbox["_count"]:
                bbox["min"][i] = point[i]
                bbox["max"][i] = point[i]
            else:
                bbox["min"][i] = min(bbox["min"][i], point[i])
                bbox["max"][i] = max(bbox["max"][i], point[i])
        bbox["_count"] += 1


    @staticmethod
    def bbox_calc(bbox):
        if bbox["_count"]:
            bbox["center"] /= bbox["_count"]
        del bbox["_count"]



    def load_model(self, gcode_path):
        parser = Parser()

        LOG.debug("load model start")
        profile_start = time.time()

        self.title = f"Nozlo: {gcode_path.name}"

        with gcode_path.open() as fp:
            self.model = parser.parse(fp)

        layers = set()

        bbox = self.bbox_init()

        for layer in self.model:
            layers.add(layer.number)
            for segment in layer.segments:
                if segment.width and segment.end[0] >= 0 and segment.end[1] >= 0:
                    # Extrusion in build volume
                    self.bbox_update(bbox, segment.end)

        self.bbox_calc(bbox)

        self.model_center = bbox["center"]
        self.model_size = np.linalg.norm(bbox["max"] - bbox["min"])

        self.layers = sorted(list(layers))
        self.draw_layer_min = 0
        self.draw_layer_max = (len(self.layers) -1)

        self.model_layer_min = None
        self.model_layer_max = None
        for n, layer in enumerate(self.model):
            if layer.model:
                self.model_layer_max = n
                if self.model_layer_min == None:
                    self.model_layer_min = n

        LOG.info(f"Loaded {len(self.layers)} layers.")
        LOG.debug(f"load model end {time.time() - profile_start:0.2f}")

        self.frame_model()


    def load_bed(self):
        bed_width = 214
        bed_length = 214
        grid_step = 30
        grid_x = 9
        grid_y = 9
        cross_xy = 10

        self.lines_bed = [
            [
                [0, 0, 0],
                [bed_width, 0, 0],
            ],
            [
                [bed_width, 0, 0],
                [bed_width, bed_length, 0],
            ],
            [
                [bed_width, bed_length, 0],
                [0, bed_length, 0],
            ],
            [
                [0, bed_length, 0],
                [0, 0, 0],
            ],
        ]

        cx = bed_width / 2
        cy = bed_length / 2

        for x in range(grid_x):
            px = cx + grid_step * (x - (grid_x - 1) / 2)
            for y in range(grid_y):
                py = cy + grid_step * (y - (grid_y - 1) / 2)
                self.lines_bed += [
                    [
                        [px - cross_xy / 2, py, 0],
                        [px + cross_xy / 2, py, 0],
                    ],
                    [
                        [px, py - cross_xy / 2, 0],
                        [px, py + cross_xy / 2, 0],
                    ],
                ]

        bbox = self.bbox_init()
        for (start, end) in self.lines_bed:
            self.bbox_update(bbox, start)
            self.bbox_update(bbox, end)

        self.bbox_calc(bbox)
        self.bed_center = bbox["center"]
        self.bed_size = np.linalg.norm(bbox["max"] - bbox["min"])


    def idle(self):
        GLUT.glutPostRedisplay()


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
        self.init_line_buffer()
        self.load_line_buffer()

        GLUT.glutDisplayFunc(self._display)
        GLUT.glutIdleFunc(self.idle)
        GLUT.glutReshapeFunc(self.reshape)
        GLUT.glutKeyboardFunc(self.keyboard)
        GLUT.glutSpecialFunc(self.special)
        GLUT.glutMouseFunc(self.mouse)
        GLUT.glutMotionFunc(self.motion)
        GLUT.glutPassiveMotionFunc(self.motion)

        GLUT.glutMainLoop()
