"""Original procedural 3D instrument meshes and camera-independent pose math.

The models contain actual thickness, surface normals, and closed solid parts.
No downloaded art or graphics runtime is required to generate or test them.
Hand world coordinates estimate relative tilt only: every hand has its own
coordinate origin, so they cannot establish shared scene depth or string contact.
"""

from dataclasses import dataclass
import math
import numpy as np


VIOLIN_GRIP = (0.0, -1.7, 0.32)
WOOD = (0.54, 0.24, 0.075)
EDGE = (0.77, 0.43, 0.16)
EBONY = (0.035, 0.043, 0.048)
IVORY = (0.82, 0.79, 0.65)
SILVER = (0.65, 0.69, 0.72)


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    tags: np.ndarray
    indices: np.ndarray


class _Builder:
    def __init__(self):
        self.vertices, self.normals, self.colors, self.tags = [], [], [], []

    def triangle(self, a, b, c, color, tag=0, outward=None):
        a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            return
        if outward is not None and np.dot(normal, outward) < 0:
            b, c, normal = c, b, -normal
        normal /= length
        self.vertices.extend((a, b, c))
        self.normals.extend((normal, normal, normal))
        self.colors.extend((color, color, color))
        self.tags.extend((tag, tag, tag))

    def quad(self, a, b, c, d, color, tag=0, outward=None):
        self.triangle(a, b, c, color, tag, outward)
        self.triangle(a, c, d, color, tag, outward)

    def finish(self):
        count = len(self.vertices)
        return Mesh(np.asarray(self.vertices, np.float32),
                    np.asarray(self.normals, np.float32),
                    np.asarray(self.colors, np.float32),
                    np.asarray(self.tags, np.float32),
                    np.arange(count, dtype=np.uint32))


def _box(mesh, low, high, color, tag=0):
    lo, hi = np.asarray(low), np.asarray(high)
    # Each face has its own vertices so hard edges have correct lighting normals.
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for side, sign in ((lo[axis], -1), (hi[axis], 1)):
            corners = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = lo.astype(float).copy()
                p[axis] = side
                p[other[0]] = (lo, hi)[u][other[0]]
                p[other[1]] = (lo, hi)[v][other[1]]
                corners.append(p)
            outward = np.zeros(3)
            outward[axis] = sign
            mesh.quad(*corners, color, tag, outward)


def _cylinder(mesh, start, end, radius, color, tag=0, sides=8, end_radius=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return
    axis /= length
    reference = np.array((0., 0., 1.)) if abs(axis[2]) < .8 else np.array((0., 1., 0.))
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.arange(sides) * (2 * math.pi / sides)
    circle = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    a = start + radius * circle
    b = end + (radius if end_radius is None else end_radius) * circle
    for i in range(sides):
        j = (i + 1) % sides
        mesh.quad(a[i], a[j], b[j], b[i], color, tag, circle[i] + circle[j])
        mesh.triangle(start, a[j], a[i], color, tag, -axis)
        mesh.triangle(end, b[i], b[j], color, tag, axis)


def _ellipsoid(mesh, center, radii, color, tag=0, sides=16, rings=8):
    center, radii = np.asarray(center), np.asarray(radii)
    def point(latitude, longitude):
        return center + radii * np.array((math.sin(latitude) * math.cos(longitude),
                                          math.sin(latitude) * math.sin(longitude),
                                          math.cos(latitude)))
    for row in range(rings):
        low, high = row * math.pi / rings, (row + 1) * math.pi / rings
        for col in range(sides):
            left, right = col * 2 * math.pi / sides, (col + 1) * 2 * math.pi / sides
            a, b, c, d = point(low, left), point(low, right), point(high, right), point(high, left)
            mesh.quad(a, b, c, d, color, tag, (a + b + c + d) / 4 - center)


_BODY_Y = np.array((-1.12, -.99, -.76, -.47, -.23, .01, .31, .70, 1.06, 1.33, 1.43))
_BODY_W = np.array((.012, .29, .46, .48, .31, .28, .48, .69, .64, .34, .012))
_BODY_CENTER = .155


def _body_width(y):
    index = np.clip(np.searchsorted(_BODY_Y, y, side="right") - 1, 0, len(_BODY_Y) - 2)
    t = np.clip((y - _BODY_Y[index]) / (_BODY_Y[index + 1] - _BODY_Y[index]), 0, 1)
    t = t * t * (3 - 2 * t)
    return _BODY_W[index] + t * (_BODY_W[index + 1] - _BODY_W[index])


def _surface_z(x, y):
    # Find this point's radial ring on the authored outline, matching the top arch.
    lo, hi = 0., 1.
    for _ in range(20):
        r = (lo + hi) / 2
        boundary_y = _BODY_CENTER + (y - _BODY_CENTER) / max(r, 1e-8)
        inside = (_BODY_Y[0] <= boundary_y <= _BODY_Y[-1] and
                  abs(x) <= r * _body_width(boundary_y))
        if inside:
            hi = r
        else:
            lo = r
    return .13 + .16 * (1 - hi * hi)


def make_violin_mesh() -> Mesh:
    """Generate an original arched, closed violin with four separate string tags."""
    mesh = _Builder()
    ys = np.linspace(_BODY_Y[0], _BODY_Y[-1], 48)
    widths = _body_width(ys)
    outline = np.concatenate((np.column_stack((-widths, ys)),
                              np.column_stack((widths[::-1], ys[::-1]))))
    center = np.array((0., _BODY_CENTER))
    for front in (True, False):
        old = None
        z_center = .29 if front else -.19
        for radius in np.linspace(1 / 6, 1, 6):
            xy = center + radius * (outline - center)
            z = .13 + .16 * (1 - radius * radius) if front else -.14 - .05 * (1 - radius * radius)
            ring = np.column_stack((xy, np.full(len(xy), z)))
            direction = (0, 0, 1 if front else -1)
            for i in range(len(ring)):
                j = (i + 1) % len(ring)
                if old is None:
                    mesh.triangle((0., _BODY_CENTER, z_center), ring[i], ring[j], WOOD, -1, direction)
                else:
                    mesh.quad(old[i], ring[i], ring[j], old[j], WOOD, -1, direction)
            old = ring
    for i, xy in enumerate(outline):
        other = outline[(i + 1) % len(outline)]
        mesh.quad((*xy, -.14), (*other, -.14), (*other, .13), (*xy, .13),
                  WOOD, -1, np.r_[((xy + other) / 2 - center), 0.])
        # Fine raised edge/purfling catches light around the recognizable silhouette.
        _cylinder(mesh, (*xy, .137), (*other, .137), .009, EDGE, sides=4)
    _box(mesh, (-.12, -2.80, .12), (.12, -.50, .30), WOOD, -1)
    _box(mesh, (-.13, -2.78, .30), (.13, -.46, .325), EBONY)
    _box(mesh, (-.11, -3.13, .08), (.11, -2.74, .28), WOOD, -1)
    _ellipsoid(mesh, (0, -3.16, .23), (.16, .22, .13), WOOD, -1)
    spiral = []
    for t in np.linspace(0, 4.5 * math.pi, 34):
        radius = .11 * (1 - t / (5 * math.pi))
        spiral.append((radius * math.cos(t), -3.16 + radius * math.sin(t), .368))
    for a, b in zip(spiral, spiral[1:]):
        _cylinder(mesh, a, b, .009, EDGE, sides=5)
    for side, y in ((-1, -3.02), (1, -2.96), (-1, -2.86), (1, -2.80)):
        _cylinder(mesh, (side * .10, y, .21), (side * .32, y, .21), .025, SILVER)
        _ellipsoid(mesh, (side * .34, y, .21), (.07, .05, .065), EBONY, sides=10, rings=5)
    # Bridge is raised above the arch; two feet leave the central opening visible.
    for side in (-1, 1):
        _box(mesh, (side * .12 - .035, .265, .26), (side * .12 + .035, .335, .49), IVORY)
    _box(mesh, (-.23, .265, .47), (.23, .335, .545), IVORY)
    _box(mesh, (-.115, .69, .27), (.115, 1.09, .32), EBONY)
    _ellipsoid(mesh, (-.35, 1.08, .23), (.28, .26, .065), EBONY, sides=20, rings=6)
    for index, x in enumerate((-.075, -.025, .025, .075)):
        for a, b in (((x, -2.80, .345), (x, .30, .553)),
                     ((x, .30, .553), (x, .98, .335))):
            _cylinder(mesh, a, b, .0055, SILVER, index + 1, sides=6)
    # Dark curved inlays sit on the actual top arch rather than on a flat decal.
    for side in (-1, 1):
        values = np.linspace(0, 1, 32)
        curve = [(side * (.22 + .032 * math.sin(2 * math.pi * t)), -.16 + .83 * t) for t in values]
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            left = ((x0 - .018, y0, _surface_z(x0 - .018, y0) + .003),
                    (x1 - .018, y1, _surface_z(x1 - .018, y1) + .003))
            right = ((x0 + .018, y0, _surface_z(x0 + .018, y0) + .003),
                     (x1 + .018, y1, _surface_z(x1 + .018, y1) + .003))
            mesh.quad(left[0], left[1], right[1], right[0], EBONY, outward=(0, 0, 1))
        for x, y in (curve[0], curve[-1]):
            _ellipsoid(mesh, (x, y, _surface_z(x, y) + .003), (.034, .038, .002), EBONY, sides=10, rings=4)
    return mesh.finish()


def make_bow_mesh() -> Mesh:
    """Actual wood shaft, separate hair bundle, frog, ferrule, and ivory tip."""
    mesh = _Builder()
    shaft = [(x, 0., .035 + .11 * math.sin(math.pi * max(x, 0) / 5.6))
             for x in np.linspace(-.30, 5.60, 32)]
    for a, b in zip(shaft, shaft[1:]):
        _cylinder(mesh, a, b, .027, WOOD, -1, sides=8)
    for y in np.linspace(-.035, .035, 5):
        _cylinder(mesh, (.16, y, -.10), (5.50, y * .7, -.10), .004, IVORY, sides=5)
    _box(mesh, (-.23, -.075, -.22), (.20, .075, .015), EBONY)
    _box(mesh, (.15, -.081, -.20), (.21, .081, -.065), SILVER)
    _cylinder(mesh, (-.38, 0, -.015), (-.23, 0, -.015), .045, SILVER)
    _ellipsoid(mesh, (-.025, .077, -.09), (.045, .003, .045), IVORY, sides=12, rings=6)
    _box(mesh, (5.47, -.035, -.13), (5.60, .035, .036), IVORY)
    return mesh.finish()


def make_marker_mesh() -> Mesh:
    mesh = _Builder()
    _ellipsoid(mesh, (0, 0, 0), (.05, .05, .05), (1., 1., 1.), sides=12, rings=6)
    return mesh.finish()


@dataclass(frozen=True)
class Pose:
    position: np.ndarray  # Canvas pixels x/y; approximate view depth z.
    rotation: np.ndarray  # Proper GL-camera rotation; geometry Y is reflected below.
    scale: float         # Pixels per authored model unit.
    timestamp: float

    def __post_init__(self):
        position = np.array(self.position, dtype=np.float64, copy=True)
        rotation = np.array(self.rotation, dtype=np.float64, copy=True)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Pose requires a 3-vector and a 3x3 rotation")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("Pose must be finite")
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.timestamp):
            raise ValueError("Pose scale and timestamp must be finite; scale must be positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Pose rotation must be orthonormal and proper")
        position.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)


def _orthogonal_rotation(matrix):
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _axis_rotation(axis, angle):
    x, y, z = axis
    cross = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def _smooth_rotation(old, target, alpha, dt):
    relative = old.T @ target
    angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    if angle < 1e-7:
        return old
    if angle > math.pi - 1e-4:
        # A detected basis reversal has no unique cross-product axis. Extract the
        # eigenvector once rather than letting a weighted matrix become singular.
        values, vectors = np.linalg.eig(relative)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    else:
        axis = np.array((relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                         relative[1, 0] - relative[0, 1])) / (2 * math.sin(angle))
    axis /= max(np.linalg.norm(axis), 1e-8)
    step = min(angle * alpha, 8. * dt)
    return _orthogonal_rotation(old @ _axis_rotation(axis, step))


def _world_depth(hand, first, second):
    world = hand.world_points
    if world is None or len(world) != 21:
        return 0.
    array = np.array([(p.x, p.y, p.z) for p in world])
    if not np.isfinite(array).all():
        return 0.
    delta = array[first] - array[second]
    length = np.linalg.norm(delta)
    if not .008 <= length <= .25:
        return 0.
    # MediaPipe z grows away from camera; GL's local front points toward camera.
    return float(np.clip(-delta[2] / length, -.55, .55))


class PoseTracker:
    """Attach the neck or bow grip to the calibrated physical hand, never its side."""

    def __init__(self, kind="violin"):
        if kind not in ("violin", "bow"):
            raise ValueError("Pose kind must be violin or bow")
        self.kind = kind
        self.pose = None

    def reset(self):
        self.pose = None

    def update(self, tracking, calibration, now, viewport, confidence=.65, timeout=.18):
        if len(viewport) == 2:
            width, height = viewport
            offset = np.zeros(2)
        elif len(viewport) == 4:
            x, y, width, height = viewport
            offset = np.array((x, y), dtype=float)
        else:
            raise ValueError("Viewport must be (width, height) or (x, y, width, height)")
        if not all(math.isfinite(v) for v in (*offset, width, height, now)) or width <= 0 or height <= 0:
            self.pose = None
            return None
        if tracking is None or not 0 <= now - tracking.timestamp <= timeout:
            self.pose = None
            return None
        if self.pose is not None and now - self.pose.timestamp > timeout:
            self.pose = None
        label = calibration.left_label if self.kind == "violin" else calibration.right_label
        hand = tracking.hand(label, confidence)
        if hand is None:
            return self.pose
        if self.pose is not None and tracking.timestamp <= self.pose.timestamp:
            return self.pose
        p = np.array([(v.x * width, v.y * height) for v in hand.points]) + offset
        palm = p[5] - p[17]
        palm_span = float(np.linalg.norm(palm))
        body = p[0] - p[9]
        length = float(np.linalg.norm(body))
        if palm_span < 3 or length < 3:
            return self.pose
        grip = (p[2] + p[5]) / 2
        if self.kind == "violin":
            body /= length
            y = np.array((-body[0], body[1], -_world_depth(hand, 0, 9)))
            y /= np.linalg.norm(y)
            x = np.array((body[1], body[0], 0.))
            alignment = np.dot(palm, (body[1], -body[0]))
            x[2] = _world_depth(hand, 5, 17) * (1 if alignment >= 0 else -1)
            x -= y * np.dot(x, y)
            x /= np.linalg.norm(x)
            rotation = np.column_stack((x, y, np.cross(x, y)))
            # Keep the whole body visible below a normal chest-height neck grip;
            # enlarging from fingertip span would also make curling resize it.
            scale = float(np.clip(palm_span * .85, height * .08, height * .15))
        else:
            shaft = np.array((body[1], -body[0]))
            other = tracking.hand(calibration.left_label, confidence)
            if other is None:
                toward = np.array(((calibration.board.x0 + calibration.board.x1) * width / 2,
                                   (calibration.board.y0 + calibration.board.y1) * height / 2)) + offset - grip
            else:
                toward = np.array(((other.points[2].x + other.points[5].x) * width / 2,
                                   (other.points[2].y + other.points[5].y) * height / 2)) + offset - grip
            if np.dot(shaft, toward) < 0:
                shaft *= -1
            x = np.array((shaft[0], -shaft[1], _world_depth(hand, 5, 17) * .5))
            x /= np.linalg.norm(x)
            y = np.array((-x[1], x[0], 0.))
            y /= np.linalg.norm(y)
            rotation = np.column_stack((x, y, np.cross(x, y)))
            scale = float(np.clip(palm_span * .9, height * .055, height * .13))
        position = np.r_[grip, 0.]
        old = self.pose
        if old is not None and tracking.timestamp - old.timestamp <= timeout:
            dt = tracking.timestamp - old.timestamp
            alpha = 1 - math.exp(-dt / .045)
            position = old.position + alpha * (position - old.position)
            scale = old.scale + alpha * (scale - old.scale)
            rotation = _smooth_rotation(old.rotation, rotation, alpha, dt)
        self.pose = Pose(position, rotation, scale, tracking.timestamp)
        return self.pose


def model_matrix(pose, canvas_width, canvas_height, grip=(0., 0., 0.)):
    """Column-vector transform; camera view applies its separate -focal z offset."""
    matrix = np.eye(4)
    matrix[:3, :3] = pose.rotation @ np.diag((pose.scale, -pose.scale, pose.scale))
    position = np.array((pose.position[0] - canvas_width / 2,
                         canvas_height / 2 - pose.position[1], pose.position[2]))
    matrix[:3, 3] = position - matrix[:3, :3] @ np.asarray(grip)
    return matrix.astype(np.float32)


def perspective_matrix(width, height, focal=1100., near=10., far=5000.):
    """Pixel-calibrated perspective: objects at view z=-focal retain pixel scale."""
    if not all(math.isfinite(v) for v in (width, height, focal, near, far)) or min(width, height, focal, near) <= 0 or far <= near:
        raise ValueError("Invalid perspective dimensions or clipping planes")
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0], matrix[1, 1] = 2 * focal / width, 2 * focal / height
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2 * far * near) / (far - near)
    matrix[3, 2] = -1
    return matrix
