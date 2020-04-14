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

    def __init__(self):
        self.title = None
        self.window = None

        self.program = None
        self.projection_matrix_uniform = None
        self.modelview_matrix_uniform = None

        self.cursor = np.array([0, 0], dtype="int")
        self.button = {v: None for v in range(5)}

        self.aspect = None
        self.view_angle = 50
        self.near_plane = 0.1
        self.far_plane = 1000

        self.camera = np.array([100, -100, 100], dtype="float32")
        self.aim = np.array([0, 0, 0], dtype="float32")
        self.up = np.array([0, 0, 1], dtype="float32")

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

        self.line_buffer_length = None
        self.line_buffer_position = None
        self.line_buffer_color = None
        self.line_array = None

        self.bed_array_chunk = []
        self.model_layer_chunks = []

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


    def add_lines_model(self, line_p, line_c):
        self.model_layer_chunks = []

        for layer in self.model:
            layer_chunk_start = self.line_buffer_length
            for segment in layer.segments:

                line_p += [segment.start[0], segment.start[1], segment.start[2]]
                line_p += [segment.end[0], segment.end[1], segment.end[2]]

                if segment.feedrate < self.high_feedrate:
                    color = colorsys.hsv_to_rgb(
                        0.8 * (1 - (segment.feedrate / self.high_feedrate)),
                        1,
                        0.8 if segment.width else 0.5
                    )
                else:
                    color = (0.8, 0.8, 0.8)

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


    def display(self):
        LOG.debug("display start")
        start = time.time()

        # Projection

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(self.view_angle, self.aspect, self.near_plane, self.far_plane)

        # Camera

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        up2 = self.up
        if not self.angle(self.up, self.camera - self.aim):
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

        # GL.glBindVertexArray(self.line_array)
        # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_position)
        # GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.line_buffer_color)
        # GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

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

        # index = self.draw_index
        # GL.glDrawElements(GL.GL_LINES, len(index), GL.GL_UNSIGNED_INT, index)
        # for n in range(self.draw_layer_min, self.draw_layer_max + 1):
        #     index = self.draw_layer_indices[n]
        #     GL.glDrawElements(GL.GL_LINES, len(index), GL.GL_UNSIGNED_INT, index)

        # index = np.array([0,1,2,3,4,5,6,7], dtype=np.uint32)
        # index = list(range(100))
        # GL.glBindVertexArray(0)

        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)

        GL.glUseProgram(0)

        # Update

        GLUT.glutSwapBuffers()

        duration = time.time() - start
        LOG.debug(f"display end {duration:0.2f}")


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
        if key == b'd':
            self.update_model_draw(toggle_single=True)

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def special(self, key, x, y):
        if key == GLUT.GLUT_KEY_HOME:
            self.update_model_draw(relative="first")
        if key == GLUT.GLUT_KEY_END:
            self.update_model_draw(relative="last")

        if key == GLUT.GLUT_KEY_DOWN:
            self.update_model_draw(layer=-1)
        if key == GLUT.GLUT_KEY_UP:
            self.update_model_draw(layer=1)

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def frame_model(self):
        cam_v = (self.camera - self.aim)
        cam_dist = np.linalg.norm(cam_v)
        cam_v /= cam_dist

        self.aim = self.model_center.copy()
        cam_v2 = cam_v * self.model_size

        self.camera = self.aim + cam_v2


    def frame_bed(self):
        cam_v = (self.camera - self.aim)
        cam_dist = np.linalg.norm(cam_v)
        cam_v /= cam_dist

        self.aim = self.bed_center.copy()
        cam_v2 = cam_v * self.bed_size

        self.camera = self.aim + cam_v2


    def update_camera(
            self,
            tumble_horiz=0,
            tumble_vert=0,
            dolly_horiz=0,
            dolly_vert=0,
            scroll=0
    ):
        cam_v = (self.camera - self.aim)
        cam_dist = np.linalg.norm(cam_v)
        cam_v /= cam_dist

        cam_xy = cam_v * np.array([1, 1, 0])
        cam_z = cam_v[2]
        cam_dist_xy = np.linalg.norm(cam_xy)
        cam_angle_horiz = math.atan2(cam_xy[1], cam_xy[0])
        cam_angle_vert = math.atan2(cam_z, cam_dist_xy)

        if scroll < 0:
            cam_dist *= 0.9
        if scroll > 0:
            cam_dist *= 1.1

        cam_angle_horiz += tumble_horiz
        cam_angle_vert += tumble_vert
        limit = math.pi / 2 * 0.999
        cam_angle_vert = np.clip(cam_angle_vert, -limit, limit)

        cam_v2 = np.array([1, 0, 0], dtype="float32")
        cam_v2 = cam_v2.dot(np.array([
            [math.cos(cam_angle_vert), 0, math.sin(cam_angle_vert)],
            [0, 1, 0],
            [-math.sin(cam_angle_vert), 0, math.cos(cam_angle_vert)],
        ]))
        cam_v2 = cam_v2.dot(np.array([
            [math.cos(cam_angle_horiz), math.sin(cam_angle_horiz), 0],
            [-math.sin(cam_angle_horiz), math.cos(cam_angle_horiz), 0],
            [0, 0, 1],
        ]))
        cam_v2 *= cam_dist

        horiz = self.unit(np.cross(self.up, cam_v))
        vert = self.unit(np.cross(cam_v, horiz))

        dolly = horiz * dolly_horiz + vert * dolly_vert

        self.aim += dolly * cam_dist * 0.002

        camera2 = cam_v2 + self.aim

        self.camera[0] = camera2[0]
        self.camera[1] = camera2[1]
        self.camera[2] = camera2[2]


    def update_model_draw(self, layer=0, relative=None, toggle_single=None):
        last = len(self.layers) - 1
        target_min = self.draw_layer_min
        target_max = self.draw_layer_max
        self.draw_index = []

        if relative == "first":
            target_max = 0
        if relative == "last":
            target_max = last

        target_max += layer
        target_max = max(0, min(last, target_max))

        if toggle_single:
            self.draw_single_layer = not self.draw_single_layer

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
            self.update_camera(scroll=-1)
        if button == 4 and state == 0:
            self.update_camera(scroll=1)

        GLUT.glutPostRedisplay()


    def motion(self, x, y):
        """
        On movement when a button is pressed
        """

        position = np.array([x, y], dtype="int")
        move = position - self.cursor

        tumble_horiz = 0
        tumble_vert = 0
        dolly_horiz = 0
        dolly_vert = 0

        if self.button[0]:
            tumble_horiz = -float(move[0]) * 0.01
            tumble_vert = float(move[1]) * 0.01

        if self.button[2]:
            dolly_horiz = -float(move[0])
            dolly_vert = float(move[1])

        self.update_camera(
            tumble_horiz=tumble_horiz,
            tumble_vert=tumble_vert,
            dolly_horiz=dolly_horiz,
            dolly_vert=dolly_vert,
        )

        self.update_cursor(x, y)
        GLUT.glutPostRedisplay()


    def reshape(self, w, h):
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
                    self.bbox_update(bbox, segment.end)
                    # Extrusion in build volume

        self.bbox_calc(bbox)

        self.model_center = bbox["center"]
        self.model_size = np.linalg.norm(bbox["max"] - bbox["min"])

        self.layers = sorted(list(layers))
        self.draw_layer_min = 0
        self.draw_layer_max = (len(self.layers) -1)

        LOG.info(f"Loaded {len(self.layers)} layers.")
        LOG.debug(f"load model end {time.time() - profile_start:0.2f}")


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


    def run(self):
        GLUT.glutInit()
        GLUT.glutSetOption(GLUT.GLUT_MULTISAMPLE, 4)
        GLUT.glutInitDisplayMode(
            GLUT.GLUT_DOUBLE |
            GLUT.GLUT_ALPHA |
            GLUT.GLUT_DEPTH |
            GLUT.GLUT_STENCIL |
            GLUT.GLUT_MULTISAMPLE
        )
        GLUT.glutInitWindowSize(1200, 720)

        GLUT.glutInitWindowPosition(1200, 720)

        self.window = GLUT.glutCreateWindow(self.title)

        self.frame_model()
        self.init_program()
        self.init_line_buffer()
        self.load_line_buffer()

        GLUT.glutDisplayFunc(self.display)
        GLUT.glutReshapeFunc(self.reshape)
        GLUT.glutKeyboardFunc(self.keyboard)
        GLUT.glutSpecialFunc(self.special)
        GLUT.glutMouseFunc(self.mouse)
        GLUT.glutMotionFunc(self.motion)
        GLUT.glutPassiveMotionFunc(self.motion)

        GLUT.glutMainLoop()
