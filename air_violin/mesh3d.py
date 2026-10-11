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
WOOD = (0.43, 0.20, 0.085)
EDGE = (0.54, 0.30, 0.14)
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

    def smooth(self, start=0):
        """Average normals only within one continuous authored surface.

        Parts such as bridge/neck and separate front/back retain their own
        normals. Quantized positions join duplicate triangle vertices, avoiding
        visible concentric facets on the arched plates.
        """
        vertices = np.asarray(self.vertices[start:])
        normals = np.asarray(self.normals[start:])
        _, group = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
        sums = np.zeros((int(group.max()) + 1, 3))
        np.add.at(sums, group, normals)
        sums /= np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
        self.normals[start:] = list(sums[group])


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


def _cylinder(mesh, start, end, radius, color, tag=0, sides=8, end_radius=None, caps=True):
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
    first = len(mesh.vertices)
    for i in range(sides):
        j = (i + 1) % sides
        mesh.quad(a[i], a[j], b[j], b[i], color, tag, circle[i] + circle[j])
    mesh.smooth(first)
    if caps:
        for i in range(sides):
            j = (i + 1) % sides
            mesh.triangle(start, a[j], a[i], color, tag, -axis)
            mesh.triangle(end, b[i], b[j], color, tag, axis)


def _ellipsoid(mesh, center, radii, color, tag=0, sides=16, rings=8):
    center, radii = np.asarray(center), np.asarray(radii)
    start = len(mesh.vertices)
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
    mesh.smooth(start)


# Units are approximately 100 mm: 356 mm body, 130 mm neck, ~590 mm
# overall. The narrow C-bouts and four projecting corners distinguish this
# outline from a guitar. Every point is original, authored here.
_BODY_Y = np.array((-1.12, -1.05, -.82, -.51, -.19, .05, .17, .28,
                    .56, .78, .97, 1.06, 1.22, 1.48, 1.81, 2.12, 2.37, 2.44))
_BODY_W = np.array((.012, .34, .69, .825, .79, .61, .68, .49,
                    .535, .515, .57, .76, .83, 1.025, 1.04, .83, .40, .012))
_BODY_CENTER = .66


def _body_width(y):
    index = np.clip(np.searchsorted(_BODY_Y, y, side="right") - 1, 0, len(_BODY_Y) - 2)
    t = np.clip((y - _BODY_Y[index]) / (_BODY_Y[index + 1] - _BODY_Y[index]), 0, 1)
    t = t * t * (3 - 2 * t)
    return _BODY_W[index] + t * (_BODY_W[index + 1] - _BODY_W[index])


def _surface_z(x, y):
    # The plate arch is a continuous surface, not concentric flat platforms.
    # Width-dependent cross arches meet constant-height ribs at the outline.
    fraction = float(np.clip((y - _BODY_Y[0]) / (_BODY_Y[-1] - _BODY_Y[0]), 0, 1))
    longitudinal = max(0., math.sin(math.pi * fraction)) ** .75
    width = max(float(_body_width(y)), 1e-5)
    return .12 + .16 * max(0., 1 - (x / width) ** 2) * longitudinal


def _tapered_block(mesh, y0, y1, half0, half1, z0, z1, color, tag=0):
    """Tapered neck/tail fittings, with a modest crowned upper surface."""
    corners = np.array(((-half0, y0, z0), (half0, y0, z0),
                        (half1, y1, z0), (-half1, y1, z0),
                        (-half0, y0, z1), (half0, y0, z1),
                        (half1, y1, z1), (-half1, y1, z1)))
    for indices, outward in (((0, 1, 2, 3), (0, 0, -1)),
                              ((4, 7, 6, 5), (0, 0, 1)),
                              ((0, 4, 5, 1), (0, -1, 0)),
                              ((3, 2, 6, 7), (0, 1, 0)),
                              ((0, 3, 7, 4), (-1, 0, 0)),
                              ((1, 5, 6, 2), (1, 0, 0))):
        mesh.quad(*(corners[i] for i in indices), color, tag, outward)


def make_violin_mesh() -> Mesh:
    """Original full-size proportions: arched plates, ribs, neck and fittings."""
    mesh = _Builder()
    ys = np.unique(np.r_[np.linspace(_BODY_Y[0], _BODY_Y[-1], 62), _BODY_CENTER])
    widths = _body_width(ys)
    outline = np.concatenate((np.column_stack((-widths, ys)),
                              np.column_stack((widths[::-1], ys[::-1]))))
    center = np.array((0., _BODY_CENTER))
    for front in (True, False):
        start = len(mesh.vertices)
        # Cross-section grid, rather than shrinking concentric copies of the
        # outline: a violin's concave C-bouts are not reliably star-shaped.
        # Homothetic rings can cross those corners and visibly crease the plate.
        rows = []
        for y, width in zip(ys, widths):
            row = []
            for x in np.linspace(-width, width, 17):
                height = _surface_z(x, y)
                z = height if front else -.17 - .75 * (height - .12)
                row.append((x, y, z))
            rows.append(row)
        direction = (0, 0, 1 if front else -1)
        for row in range(len(rows) - 1):
            for col in range(16):
                mesh.quad(rows[row][col], rows[row][col + 1],
                          rows[row + 1][col + 1], rows[row + 1][col], WOOD, -1, direction)
        mesh.smooth(start)
    start = len(mesh.vertices)
    for i, xy in enumerate(outline):
        other = outline[(i + 1) % len(outline)]
        mesh.quad((*xy, -.17), (*other, -.17), (*other, .12), (*xy, .12),
                  WOOD, -1, np.r_[((xy + other) / 2 - center), 0.])
    mesh.smooth(start)
    for i, xy in enumerate(outline):
        other = outline[(i + 1) % len(outline)]
        # Fine wood lip and dark twin purfling, separate from broad body normals.
        _cylinder(mesh, (*xy, .124), (*other, .124), .008, EDGE, sides=4, caps=False)
        inset, next_inset = center + .967 * (xy - center), center + .967 * (other - center)
        _cylinder(mesh, (*inset, .132), (*next_inset, .132), .005, EBONY, sides=4, caps=False)
    _tapered_block(mesh, -2.42, -.91, .10, .15, .10, .315, WOOD, -1)
    _tapered_block(mesh, -2.44, .27, .119, .17, .315, .355, EBONY)
    _box(mesh, (-.124, -2.47, .335), (.124, -2.43, .38), EBONY)
    # Rounded pegbox cheek rails with a recessed central channel.
    for side in (-1, 1):
        _tapered_block(mesh, -3.14, -2.43, .046, .038, .09, .27, WOOD, -1)
        _cylinder(mesh, (side * .093, -3.12, .15), (side * .093, -2.48, .15),
                  .045, WOOD, -1, sides=8)
    _box(mesh, (-.055, -3.10, .255), (.055, -2.54, .265), EBONY)
    _ellipsoid(mesh, (0, -3.27, .19), (.17, .19, .15), WOOD, -1, sides=16, rings=8)
    for side in (-1, 1):
        spiral = []
        for t in np.linspace(0, 4.4 * math.pi, 26):
            radius = .13 * (1 - t / (5.2 * math.pi))
            spiral.append((side * .148, -3.27 + radius * math.cos(t), .19 + radius * math.sin(t)))
        for a, b in zip(spiral, spiral[1:]):
            _cylinder(mesh, a, b, .009, EDGE, sides=5)
    for side, y in ((-1, -3.03), (1, -2.89), (-1, -2.76), (1, -2.62)):
        _cylinder(mesh, (side * .085, y, .18), (side * .27, y, .18), .025, EBONY)
        _ellipsoid(mesh, (side * .29, y, .18), (.065, .036, .068), EBONY, sides=12, rings=6)
    # Separate legs and an arched bridge crown leave visible cutouts.
    for side in (-1, 1):
        _box(mesh, (side * .13 - .035, .795, .255), (side * .13 + .035, .865, .47), IVORY)
        _cylinder(mesh, (side * .12, .83, .44), (side * .045, .83, .57), .027, IVORY, sides=6)
    for x0, x1 in zip(np.linspace(-.23, .23, 9)[:-1], np.linspace(-.23, .23, 9)[1:]):
        z0, z1 = .55 + .055 * (1 - (x0 / .23) ** 2), .55 + .055 * (1 - (x1 / .23) ** 2)
        _box(mesh, (x0, .795, min(z0, z1) - .055), (x1, .865, max(z0, z1)), IVORY)
    _tapered_block(mesh, 1.15, 2.03, .18, .085, .275, .33, EBONY)
    _ellipsoid(mesh, (-.46, 1.95, .27), (.43, .265, .065), EBONY, sides=22, rings=8)
    for index, x in enumerate((-.075, -.025, .025, .075)):
        bridge_x, bridge_z = x * 2.2, .61 - .014 * (index - 1.5) ** 2
        for a, b in (((x, -2.44, .388), (bridge_x, .83, bridge_z)),
                     ((bridge_x, .83, bridge_z), (x * 1.7, 1.24, .355))):
            _cylinder(mesh, a, b, .0065, SILVER, index + 1, sides=6)
        _ellipsoid(mesh, (x * 1.7, 1.25, .358), (.014, .024, .012), SILVER, sides=6, rings=4)
    # Dark narrow f-hole recesses follow the arched plate and include eye/notch.
    for side in (-1, 1):
        values = np.linspace(0, 1, 28)
        curve = [(side * (.40 + .065 * math.sin(2 * math.pi * t)), .28 + 1.03 * t) for t in values]
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            left = ((x0 - .015, y0, _surface_z(x0 - .015, y0) + .002),
                    (x1 - .015, y1, _surface_z(x1 - .015, y1) + .002))
            right = ((x0 + .015, y0, _surface_z(x0 + .015, y0) + .002),
                     (x1 + .015, y1, _surface_z(x1 + .015, y1) + .002))
            mesh.quad(left[0], left[1], right[1], right[0], EBONY, outward=(0, 0, 1))
        for x, y in (curve[0], curve[-1]):
            _ellipsoid(mesh, (x, y, _surface_z(x, y) + .002), (.038, .040, .002), EBONY, sides=10, rings=4)
        x, y = side * .40, .795
        _cylinder(mesh, (x - .040, y, _surface_z(x, y) + .007),
                  (x + .040, y, _surface_z(x, y) + .007), .008, EBONY, sides=5)
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

    def update(self, tracking, calibration, now, viewport, confidence=.65, timeout=.18, size=1.0):
        if len(viewport) == 2:
            width, height = viewport
            offset = np.zeros(2)
        elif len(viewport) == 4:
            x, y, width, height = viewport
            offset = np.array((x, y), dtype=float)
        else:
            raise ValueError("Viewport must be (width, height) or (x, y, width, height)")
        if not all(math.isfinite(v) for v in (*offset, width, height, now, size)) or width <= 0 or height <= 0 or size <= 0:
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
        # A 4/4 violin has a fixed physical size. Palm foreshortening is a poor
        # depth measurement, so it never determines scale. Users can choose a
        # deliberate stage-relative size rather than watching the body shrink.
        scale = min(width * .108, height * .15) * float(np.clip(size, .65, 1.75))
        # Pose is attached to the calibrated physical left hand. Screen side
        # affects playing posture only, never hand identity or pitch mapping.
        board_center = (calibration.board.x0 + calibration.board.x1) / 2
        normalized_x = (grip[0] - offset[0]) / width
        side = 1 if (normalized_x if abs(normalized_x - .5) > .08 else board_center) < .5 else -1
        def violin_axes(left_hand, left_points, left_side):
            wrist_axis = left_points[0] - left_points[9]
            wrist_length = max(float(np.linalg.norm(wrist_axis)), 1.)
            # Body extends toward the torso, almost horizontally. Wrist/palm
            # movement contributes restrained tilt, rather than standing the
            # instrument upright like a cello or rotating it end-on.
            angle = math.atan2(.45, left_side * .89)
            angle += float(np.clip(wrist_axis[0] / wrist_length, -.8, .8)) * .14
            direction = np.array((math.cos(angle), math.sin(angle)))
            y_axis = np.array((-direction[0], direction[1], -_world_depth(left_hand, 0, 9) * .42))
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.array((direction[1], direction[0], _world_depth(left_hand, 5, 17) * .48))
            x_axis -= y_axis * np.dot(x_axis, y_axis)
            x_axis /= np.linalg.norm(x_axis)
            return np.column_stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
        if self.kind == "violin":
            rotation = violin_axes(hand, p, side)
        else:
            other = tracking.hand(calibration.left_label, confidence)
            if other is None:
                toward = np.array(((calibration.board.x0 + calibration.board.x1) * width / 2,
                                   (calibration.board.y0 + calibration.board.y1) * height / 2)) + offset - grip
            else:
                left_points = np.array([(v.x * width, v.y * height) for v in other.points]) + offset
                left_grip = (left_points[2] + left_points[5]) / 2
                left_x = (left_grip[0] - offset[0]) / width
                left_side = 1 if (left_x if abs(left_x - .5) > .08 else board_center) < .5 else -1
                left_rotation = violin_axes(other, left_points, left_side)
                delta = left_rotation @ np.array((0., -( .83 - VIOLIN_GRIP[1]), 0.)) * scale
                # Aim at the bridge/body, not the neck grip. This is visual
                # alignment only; no fabricated contact drives musical sound.
                toward = left_grip + (delta[0], -delta[1]) - grip
            shaft = toward / max(float(np.linalg.norm(toward)), 1.)
            if np.linalg.norm(shaft) < .1:
                shaft = np.array((body[1], -body[0])) / length
            x = np.array((shaft[0], -shaft[1], _world_depth(hand, 5, 17) * .5))
            x /= np.linalg.norm(x)
            y = np.array((-x[1], x[0], 0.))
            y /= np.linalg.norm(y)
            rotation = np.column_stack((x, y, np.cross(x, y)))
            scale *= 1.04
        position = np.r_[grip, 0.]
        old = self.pose
        if old is not None and tracking.timestamp - old.timestamp <= timeout:
            dt = tracking.timestamp - old.timestamp
            alpha = 1 - math.exp(-dt / .045)
            position = old.position + alpha * (position - old.position)
            # Scale changes only through viewport or the explicit size control.
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
