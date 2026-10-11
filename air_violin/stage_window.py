"""One native OpenGL window: camera/HUD texture, genuine meshes, and input.

Vision and audio remain independent of this presentation layer. No inference,
file access, audio synthesis, or camera reads happen in the rendering methods.
The procedural mesh and pose math can be tested without importing pyglet.
"""

import ctypes
from pathlib import Path

import numpy as np

from .mesh3d import (Pose, PoseTracker, VIOLIN_GRIP, make_bow_mesh,
                     make_marker_mesh, make_violin_mesh, model_matrix,
                     perspective_matrix)


_TEXTURE_VERTEX = """#version 330 core
layout(location=0) in vec2 position;
layout(location=1) in vec2 texcoord;
out vec2 uv;
void main() { gl_Position=vec4(position,0.0,1.0); uv=texcoord; }
"""
_TEXTURE_FRAGMENT = """#version 330 core
in vec2 uv;
uniform sampler2D picture;
out vec4 frag;
void main() { frag=texture(picture,uv); }
"""
_MESH_VERTEX = """#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
layout(location=3) in float tag;
uniform mat4 model;
uniform mat4 projection;
out vec3 surface_normal;
out vec3 material;
out vec3 object_position;
out float material_tag;
void main() {
    vec4 p=model*vec4(position,1.0);
    p.z-=1100.0;
    gl_Position=projection*p;
    surface_normal=normalize(mat3(model)*normal);
    material=color;
    material_tag=tag;
    object_position=position;
}
"""
_MESH_FRAGMENT = """#version 330 core
in vec3 surface_normal;
in vec3 material;
in vec3 object_position;
in float material_tag;
uniform float active_string;
uniform vec3 active_color;
uniform float intensity;
out vec4 frag;
void main() {
    vec3 n=normalize(surface_normal);
    vec3 light=normalize(vec3(-0.45,0.65,0.85));
    float diffuse=max(dot(n,light),0.0);
    float shine=pow(max(dot(n,normalize(light+vec3(0,0,1))),0.0),32.0);
    vec3 c=material;
    if (material_tag < -0.5) {
        // Restrained longitudinal grain and maple figure, not orange stripes.
        float grain=sin(object_position.x*71.0+sin(object_position.y*1.8)*0.9);
        float figure=sin(object_position.y*9.0+object_position.x*0.8);
        c*=0.99+0.009*grain+0.016*figure;
    }
    bool lit_string=material_tag>0.5 && abs(material_tag-active_string)<0.25;
    if (lit_string) c=active_color;
    c=c*(0.60+0.48*diffuse)+vec3(0.09*shine);
    if (lit_string) c+=active_color*(0.22+0.38*intensity);
    frag=vec4(clamp(c,0.0,1.0),1.0);
}
"""

_STRING_RGB = ((.41, .70, .92), (.45, .88, .71),
               (1., .76, .50), (.94, .59, .77))


class _Geometry:
    """Own explicit GL buffers; deletion occurs while the context is current."""

    def __init__(self, gl, data, indices=None, mesh=False):
        self.gl, self.mesh = gl, mesh
        self.vao, self.vbo, self.ebo = gl.GLuint(), gl.GLuint(), gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        gl.glGenBuffers(1, ctypes.byref(self.vbo))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        data = np.ascontiguousarray(data, dtype=np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data,
                        gl.GL_STATIC_DRAW)
        stride = 40 if mesh else 16
        attributes = ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36)) if mesh else ((0, 2, 0), (1, 2, 8))
        for index, size, offset in attributes:
            gl.glEnableVertexAttribArray(index)
            gl.glVertexAttribPointer(index, size, gl.GL_FLOAT, gl.GL_FALSE,
                                     stride, ctypes.c_void_p(offset))
        self.indexed = indices is not None
        self.count = len(indices) if self.indexed else len(data)
        if self.indexed:
            indices = np.ascontiguousarray(indices, dtype=np.uint32)
            gl.glGenBuffers(1, ctypes.byref(self.ebo))
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes,
                            indices.ctypes.data, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)

    @classmethod
    def from_mesh(cls, gl, mesh):
        data = np.column_stack((mesh.vertices, mesh.normals, mesh.colors, mesh.tags))
        return cls(gl, data, mesh.indices, mesh=True)

    def replace(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        self.gl.glBindBuffer(self.gl.GL_ARRAY_BUFFER, self.vbo)
        self.gl.glBufferData(self.gl.GL_ARRAY_BUFFER, data.nbytes,
                             data.ctypes.data, self.gl.GL_STREAM_DRAW)
        self.count = len(data)

    def draw(self):
        gl = self.gl
        gl.glBindVertexArray(self.vao)
        if self.indexed:
            gl.glDrawElements(gl.GL_TRIANGLES, self.count, gl.GL_UNSIGNED_INT, None)
        else:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.count)

    def close(self):
        gl = self.gl
        if self.ebo.value:
            gl.glDeleteBuffers(1, ctypes.byref(self.ebo))
        if self.vbo.value:
            gl.glDeleteBuffers(1, ctypes.byref(self.vbo))
        if self.vao.value:
            gl.glDeleteVertexArrays(1, ctypes.byref(self.vao))


class StageWindow:
    """OpenGL 3.3 stage, with ASCII-compatible input for the existing controller.

    ``viewport`` passed to present is the camera rectangle in the logical CPU
    canvas, including letterbox offsets. Tracking coordinates are already in
    the displayed camera orientation: this class never mirrors them again.
    """

    def __init__(self, width=920, height=710, headless=False, visible=True):
        # Lazy import keeps headless math tests and OpenCV fallback independent.
        import pyglet
        if headless:
            pyglet.options["headless"] = True
        pyglet.options["shadow_window"] = False
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        self.pyglet, self.gl, self.headless = pyglet, gl, headless
        self.closed, self._disposed = False, False
        self._keys = []
        self.confidence, self.timeout = .65, .18
        self.instrument_size, self.occlusion = 1.0, False
        self._window = None
        self._objects = []
        self._texture = gl.GLuint()
        self._texture_shape = None
        self._canvas_size = (width, height)
        self._fitted = (0, 0, width, height)
        self.violin_pose, self.bow_pose = None, None
        self._violin_tracker, self._bow_tracker = PoseTracker("violin"), PoseTracker("bow")
        try:
            config = gl.Config(major_version=3, minor_version=3, depth_size=24,
                               red_size=8, green_size=8, blue_size=8, alpha_size=8,
                               double_buffer=True)
            self._window = pyglet.window.Window(width=width, height=height,
                caption="Air Violin Studio", config=config, vsync=False,
                visible=visible, resizable=True)
            self._window.switch_to()
            self.renderer_name = gl.gl_info.get_renderer()
            self.renderer_version = ".".join(map(str, gl.gl_info.get_version()))
            self._texture_program = ShaderProgram(Shader(_TEXTURE_VERTEX, "vertex"),
                                                  Shader(_TEXTURE_FRAGMENT, "fragment"))
            self._mesh_program = ShaderProgram(Shader(_MESH_VERTEX, "vertex"),
                                               Shader(_MESH_FRAGMENT, "fragment"))
            quad = np.array(((-1, -1, 0, 1), (1, -1, 1, 1), (1, 1, 1, 0),
                             (-1, -1, 0, 1), (1, 1, 1, 0), (-1, 1, 0, 0)), dtype=np.float32)
            self._quad = _Geometry(gl, quad)
            self._objects.append(self._quad)
            self._patch = _Geometry(gl, np.zeros((3, 4), dtype=np.float32))
            self._objects.append(self._patch)
            self._violin = _Geometry.from_mesh(gl, make_violin_mesh())
            self._objects.append(self._violin)
            self._bow = _Geometry.from_mesh(gl, make_bow_mesh())
            self._objects.append(self._bow)
            self._marker = _Geometry.from_mesh(gl, make_marker_mesh())
            self._objects.append(self._marker)
            gl.glGenTextures(1, ctypes.byref(self._texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
            for name in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, name, gl.GL_LINEAR)
            for name in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, name, gl.GL_CLAMP_TO_EDGE)
            self._texture_program["picture"] = 0
            self._window.push_handlers(on_key_press=self._key_press,
                                       on_close=self._request_close)
        except Exception:
            self.close()
            raise

    def _key_press(self, symbol, modifiers):
        key = self.pyglet.window.key
        named = {key.ESCAPE: 27, key.ENTER: 13, key.NUM_ENTER: 13,
                 key.SPACE: 32, key.BACKSPACE: 8, key.TAB: 9}
        value = named.get(symbol)
        if value is None and 32 <= symbol <= 126:
            value = symbol
            if modifiers & key.MOD_SHIFT and symbol == key.EQUAL:
                value = ord("+")
        if value is not None:
            self._keys.append(value)
        return self.pyglet.event.EVENT_HANDLED

    def _request_close(self):
        self.closed = True
        return self.pyglet.event.EVENT_HANDLED

    def poll(self):
        if not self._disposed:
            self._window.dispatch_events()
        keys, self._keys = self._keys, []
        return keys

    def set_fullscreen(self, value):
        if not self._disposed:
            self._window.set_fullscreen(bool(value))

    def _upload(self, canvas):
        gl = self.gl
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        if canvas.ndim != 3 or canvas.shape[2] != 3:
            raise ValueError("StageWindow expects a uint8 BGR image")
        h, w = canvas.shape[:2]
        self._canvas_size = (w, h)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        if self._texture_shape != (h, w):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0,
                           gl.GL_BGR, gl.GL_UNSIGNED_BYTE, canvas.ctypes.data)
            self._texture_shape = (h, w)
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h,
                              gl.GL_BGR, gl.GL_UNSIGNED_BYTE, canvas.ctypes.data)

    @staticmethod
    def _matrix_uniform(matrix):
        # NumPy stores rows; OpenGL's matrix uniform consumes column-major data.
        return tuple(float(x) for x in matrix.T.reshape(-1))

    def _draw_mesh(self, geometry, model):
        self._mesh_program["model"] = self._matrix_uniform(model)
        geometry.draw()

    def _occlude_palms(self, tracking, calibration, now, viewport):
        """Small image-space palm cutouts, not claimed depth/skin segmentation."""
        if tracking is None or not 0 <= now - tracking.timestamp <= self.timeout:
            return
        import cv2
        cw, ch = self._canvas_size
        ox, oy, w, h = viewport
        for label in (calibration.left_label, calibration.right_label):
            hand = tracking.hand(label, self.confidence)
            if hand is None:
                continue
            points = np.array([(ox + hand.points[i].x * w, oy + hand.points[i].y * h)
                               for i in (0, 1, 2, 5, 9, 13, 17)], dtype=np.float32)
            hull = cv2.convexHull(points).reshape(-1, 2)
            if len(hull) < 3:
                continue
            # Triangulated convex hull is bounded (at most five triangles).
            triangles = np.concatenate([hull[[0, i, i + 1]] for i in range(1, len(hull) - 1)])
            vertices = np.column_stack((triangles[:, 0] / cw * 2 - 1,
                                       1 - triangles[:, 1] / ch * 2,
                                       triangles[:, 0] / cw, triangles[:, 1] / ch))
            self._patch.replace(vertices)
            self._patch.draw()

    def present(self, canvas, tracking, calibration, state, now, demo=False,
                draw_instrument=True, viewport=(20, 64, 880, 560)):
        if self._disposed or self.closed:
            return
        gl = self.gl
        self._window.switch_to()
        self._upload(canvas)
        cw, ch = self._canvas_size
        fw, fh = self._window.get_framebuffer_size()
        fit = min(fw / cw, fh / ch)
        dw, dh = max(1, round(cw * fit)), max(1, round(ch * fit))
        dx, dy = (fw - dw) // 2, (fh - dh) // 2
        self._fitted = (dx, dy, dw, dh)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, fw, fh)
        gl.glClearColor(.015, .018, .025, 1.)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(dx, dy, dw, dh)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        self._texture_program.use()
        self._quad.draw()
        self.violin_pose = self._violin_tracker.update(tracking, calibration, now, viewport,
                                                     self.confidence, self.timeout, self.instrument_size)
        self.bow_pose = self._bow_tracker.update(tracking, calibration, now, viewport,
                                               self.confidence, self.timeout, self.instrument_size)
        if draw_instrument:
            if demo:
                ox, oy, w, h = viewport
                angle = 1.10
                rotation = np.array(((np.cos(angle), -np.sin(angle), 0),
                                     (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
                scale = min(w * .108, h * .15) * self.instrument_size
                self.violin_pose = Pose(np.array((ox + w * .30, oy + h * .39, 0.)),
                                       rotation, scale, now)
                angle = -2.35
                bow_rotation = np.array(((np.cos(angle), -np.sin(angle), 0),
                                         (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
                self.bow_pose = Pose(np.array((ox + w * .80, oy + h * .28, 55.)),
                                    bow_rotation, scale * 1.04, now)
            # Stage-only geometry never covers controls or the performance HUD.
            sx, sy, sw, sh = (20, 64, 880, 500)
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glScissor(dx + round(sx * fit), dy + round((ch - sy - sh) * fit),
                         round(sw * fit), round(sh * fit))
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            self._mesh_program.use()
            self._mesh_program["projection"] = self._matrix_uniform(perspective_matrix(cw, ch))
            selected = state.audio.midi is not None and (state.left_fresh or demo)
            if state.fingering_mode == "fingers" and state.finger is None and not demo:
                selected = False
            self._mesh_program["active_string"] = float(state.string + 1 if selected else 0)
            self._mesh_program["active_color"] = _STRING_RGB[state.string % 4]
            self._mesh_program["intensity"] = float(np.clip(state.audio.intensity, 0, 1))
            if self.violin_pose is not None:
                model = model_matrix(self.violin_pose, cw, ch, VIOLIN_GRIP)
                self._draw_mesh(self._violin, model)
                if selected:
                    marker = np.eye(4)
                    marker[:3, 3] = ((state.string - 1.5) * .05,
                                       -2.28 + np.clip(state.zone, 0, 7) * .18, .42)
                    self._draw_mesh(self._marker, model @ marker)
            if self.bow_pose is not None:
                self._draw_mesh(self._bow, model_matrix(self.bow_pose, cw, ch))
            gl.glDisable(gl.GL_DEPTH_TEST)
            self._texture_program.use()
            # The camera is monocular: a palm hull has no depth. Cutting out
            # its entire bounding region severed the user's violin neck. Keep
            # the complete model unless coarse occlusion is explicitly enabled.
            if self.occlusion and not demo:
                self._occlude_palms(tracking, calibration, now, viewport)
            gl.glDisable(gl.GL_SCISSOR_TEST)
        self._window.flip()

    def read_pixels(self):
        """Explicit screenshot readback; never performed in the normal loop."""
        if self._disposed:
            raise RuntimeError("Stage window is closed")
        gl = self.gl
        self._window.switch_to()
        w, h = self._window.get_framebuffer_size()
        image = np.empty((h, w, 3), dtype=np.uint8)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glReadBuffer(gl.GL_BACK if self.headless else gl.GL_FRONT)
        gl.glReadPixels(0, 0, w, h, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, image.ctypes.data)
        return np.ascontiguousarray(image[::-1])

    def capture(self, path):
        import cv2
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.read_pixels()):
            raise OSError(f"Could not save screenshot: {path}")

    def close(self):
        if self._disposed:
            return
        self.closed, self._disposed = True, True
        if self._window is None:
            return
        self._window.switch_to()
        for geometry in self._objects:
            geometry.close()
        if self._texture.value:
            self.gl.glDeleteTextures(1, ctypes.byref(self._texture))
        for name in ("_mesh_program", "_texture_program"):
            program = getattr(self, name, None)
            if program:
                program.delete()
        self._window.close()
