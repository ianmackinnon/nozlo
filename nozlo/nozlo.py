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

import math
import logging
import colorsys

from OpenGL import GL
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from array import array
import numpy as np

from nozlo.parser import Parser


LOG = logging.getLogger("nozlo")



POSITION_VECTOR_SIZE = 3



class Nozlo():
    def __init__(self):
        self.program = None
        self.projection_matrix_uniform = None
        self.modelview_matrix_uniform = None

        self.cursor = np.array([0, 0], dtype="int")
        self.button = {v: None for v in range(5)}

        self.aspect = None
        self.view_angle = 50
        self.near_plane = 0.1
        self.far_plane = 1000

        self.camera = np.array([214 + 100, -100, 200], dtype="float32")
        self.aim = np.array([107, 107, 0], dtype="float32")
        self.up = np.array([0, 0, 1], dtype="float32")

        self.background = 0.18, 0.18, 0.18, 0.0

        self.layer_buffer_position = None
        self.layer_buffer_color = None
        self.layer_array = None

        self.line_data = None


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

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;

out vec3 color;

void main() {
  color = vertex_color;
  gl_Position = uPMatrix * uMVMatrix * vec4(vertex_position, 1.0);
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

        self.projection_matrix_uniform = glGetUniformLocation(self.program, 'uPMatrix')
        self.modelview_matrix_uniform = glGetUniformLocation(self.program, "uMVMatrix")


    def init_layer_buffer(self):

        self.line_data = []
        color_data = []


        max_feedrate = 0
        for (start, end, width, feedrate) in self.lines:
            max_feedrate = max(max_feedrate, feedrate)

        for (start, end, width, feedrate) in self.lines:
            z_scale = 1
            self.line_data += [start[0], start[1], start[2] * z_scale]
            self.line_data += [end[0], end[1], end[2] * z_scale]
            hue = 0.8 * (1 - (feedrate / max_feedrate))
            value = 0.8 if width else 0.6
            c = colorsys.hsv_to_rgb(hue, 1, value)
            color_data += [c, c, c]
            color_data += [c, c, c]


        self.layer_buffer_position = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.layer_buffer_position)
        glBufferData(
            GL_ARRAY_BUFFER,
            np.array(self.line_data, dtype='float32'),
            GL_STATIC_DRAW
        )

        self.layer_buffer_color = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.layer_buffer_color)
        glBufferData(
            GL_ARRAY_BUFFER,
            np.array(color_data, dtype='float32'),
            GL_STATIC_DRAW
        )



    def display(self):

        # Projection

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.view_angle, self.aspect, self.near_plane, self.far_plane)

        # Camera

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        up2 = self.up
        if not self.angle(self.up, self.camera - self.aim):
            up2 = np.array([0, 1, 0])
        gluLookAt(
            self.camera[0], self.camera[1], self.camera[2],
            self.aim[0], self.aim[1], self.aim[2],
            up2[0], up2[1], up2[2],
        )

        # Shader

        GL.glUseProgram(self.program)

        glUniformMatrix4fv(
            self.projection_matrix_uniform, 1, GL_FALSE,
            glGetFloatv(GL_PROJECTION_MATRIX))
        glUniformMatrix4fv(
            self.modelview_matrix_uniform, 1, GL_FALSE,
            glGetFloatv(GL_MODELVIEW_MATRIX))

        # Background

        glClearColor(*self.background)
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw Layers

        glEnable(GL_MULTISAMPLE)
        # glEnable(GL_BLEND)
        # glEnable(GL_DEPTH_TEST)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_LINE_SMOOTH)
        # glEnable(GL_POLYGON_SMOOTH)
        # glEnable(GL_ALPHA_TEST)
        # glAlphaFunc(GL_ALWAYS, 0)
        # glShadeModel(GL_SMOOTH);
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)

        glLineWidth(1.0)

        self.layer_array = glGenVertexArrays(1)

        glBindVertexArray(self.layer_array)
        glBindBuffer(GL_ARRAY_BUFFER, self.layer_buffer_position)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.layer_buffer_color)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

        length = len(self.line_data) // POSITION_VECTOR_SIZE

        glDrawArrays(GL_LINES, 0, length)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        GL.glUseProgram(0)

        # Update

        glutSwapBuffers()


    def update_cursor(self, x, y):
        self.cursor[0] = x
        self.cursor[1] = y


    def keyboard(self, key, x, y):
        if ord(key) == 27 or key == b'q':
            glutLeaveMainLoop()

        self.update_cursor(x, y)
        glutPostRedisplay()


    def special(self, key, x, y):
        if key == GLUT_KEY_LEFT:
            self.camera[0] -= 1
        if key == GLUT_KEY_RIGHT:
            self.camera[0] += 1
        if key == GLUT_KEY_DOWN:
            self.camera[1] -= 1
        if key == GLUT_KEY_UP:
            self.camera[1] += 1

        self.update_cursor(x, y)
        glutPostRedisplay()


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


    def mouse(self, button, state, x, y):
        self.button[button] = not state

        if button == 3 and state == 0:
            self.update_camera(scroll=-1)
        if button == 4 and state == 0:
            self.update_camera(scroll=1)

        glutPostRedisplay()


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
        glutPostRedisplay()


    def reshape(self, w, h):
        self.aspect = w / h if h else 1;
        glViewport(0, 0, w, h)
        glutPostRedisplay()


    def load(self, gcode_path):
        parser = Parser()

        with gcode_path.open() as fp:
            self.lines = parser.parse(fp)


    def run(self):
        glutInit()
        glutSetOption(GLUT_MULTISAMPLE, 2)
        glutInitDisplayMode(
            GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL | GLUT_MULTISAMPLE)
        glutInitWindowSize(1200, 720)

        glutInitWindowPosition(1200, 720)

        window = glutCreateWindow("Nozlo")

        self.init_program()
        self.init_layer_buffer()

        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.special)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutPassiveMotionFunc(self.motion)

        glutMainLoop();
