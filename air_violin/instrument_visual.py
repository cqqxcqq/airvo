"""Decorative 2D instrument. Its geometry never determines the musical pitch."""

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class BowPose:
    grip: tuple[float, float]
    angle: float
    length: float
    timestamp: float


def violin_placement(board, bow):
    """Place the body between the playing regions; support either mirrored layout."""
    board_center = (board.x0 + board.x1) / 2
    bow_center = (bow.x0 + bow.x1) / 2
    side = 1 if bow_center >= board_center else -1
    x = float(np.clip(board_center + side * ((board.x1 - board.x0) * .30 + .12), .15, .85))
    y = float(np.clip((board.y0 + board.y1) / 2 + .15, .25, .84))
    return x, y, -.58 * side


class BowAnchor:
    def __init__(self):
        self.pose = None

    def update(self, tracking, calibration, now, width, height, confidence=0.65, timeout=0.18):
        if not tracking or not 0 <= now - tracking.timestamp <= timeout:
            self.pose = None
            return None
        hand = tracking.hand(calibration.right_label, confidence)
        if hand is None:
            # Fade the last position only within the normal tracking timeout.
            if self.pose and now - self.pose.timestamp <= timeout:
                return self.pose
            self.pose = None
            return None
        if self.pose and tracking.timestamp <= self.pose.timestamp:
            return self.pose
        # Near the thumb/index base, rather than their tips: open-hand Auto Bow
        # and pinching Precision Bow should both look like holding the same grip.
        a, b = hand.points[2], hand.points[5]
        grip = ((a.x + b.x) * .5 * width, (a.y + b.y) * .5 * height)
        wrist, middle = hand.points[0], hand.points[9]
        dx, dy = (middle.x - wrist.x) * width, (middle.y - wrist.y) * height
        angle = math.atan2(dy, dx) - math.pi / 2 if math.hypot(dx, dy) > 3 else math.pi
        body_x, body_y, _ = violin_placement(calibration.board, calibration.bow)
        toward_body = (body_x * width - grip[0], body_y * height - grip[1])
        if math.cos(angle) * toward_body[0] + math.sin(angle) * toward_body[1] < 0:
            angle += math.pi  # direction follows calibrated regions, never screen-side identity
        palm = math.hypot((hand.points[5].x - hand.points[17].x) * width,
                          (hand.points[5].y - hand.points[17].y) * height)
        length = float(np.clip(palm * 5.5, width * .28, width * .60))
        old = self.pose
        if old and tracking.timestamp - old.timestamp <= timeout:
            dt = tracking.timestamp - old.timestamp
            alpha = 1 - math.exp(-dt / .035)
            # Unwrap angles so +179 -> -179 degrees does not spin a full circle.
            delta = math.atan2(math.sin(angle - old.angle), math.cos(angle - old.angle))
            angle = old.angle + alpha * delta
            grip = tuple(x + alpha * (y - x) for x, y in zip(old.grip, grip))
            length = old.length + alpha * (length - old.length)
        self.pose = BowPose(grip, angle, length, tracking.timestamp)
        return self.pose


def draw_bow(image, pose, color, intensity=0.0, opacity=1.0):
    if pose is None:
        return
    target = image if opacity >= 1 else image.copy()
    unit = np.array((math.cos(pose.angle), math.sin(pose.angle)))
    normal = np.array((-unit[1], unit[0]))
    grip = np.array(pose.grip)
    def point(distance, offset=0):
        return tuple(np.round(grip + distance * unit + offset * normal).astype(int))
    length = pose.length
    # Frog behind the grip, shaft and hair extending toward the left.
    cv2.line(target, point(-length * .09), point(length * .91, -3), (21, 24, 34), 7, cv2.LINE_AA)
    cv2.line(target, point(-length * .09), point(length * .91, -3), (77, 140, 188), 3, cv2.LINE_AA)
    cv2.line(target, point(0, 9), point(length * .89, 3), (218, 230, 239), 2, cv2.LINE_AA)
    if intensity > .02:
        tone = tuple(int(c * (.40 + .55 * intensity)) for c in color)
        cv2.line(target, point(0, 10), point(length * .89, 4), tone, 1, cv2.LINE_AA)
    cv2.line(target, point(-length * .04, 4), point(length * .025, 4), (37, 41, 45), 12, cv2.LINE_AA)
    cv2.line(target, point(-length * .035, 6), point(length * .015, 6), (179, 185, 192), 2, cv2.LINE_AA)
    cv2.line(target, point(length * .91, -3), point(length * .89, 3), (128, 183, 211), 3, cv2.LINE_AA)
    if opacity < 1:
        cv2.addWeighted(target, opacity, image, 1 - opacity, 0, image)


def draw_violin(image, board, pixel, state, colors, bow_region=None):
    """Recognizable hourglass body, f holes, neck, four strings, and note marker."""
    h, w = image.shape[:2]
    from .config import Calibration
    cx, cy, angle = violin_placement(board, bow_region or Calibration().bow)
    center = np.array(pixel(cx, cy))
    scale = min(w / 880, h / 560) * .85
    rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    def points(items):
        return np.round(np.asarray(items) @ rotation.T * scale + center).astype(np.int32)
    def line(a, b, color, thickness=1):
        cv2.line(image, tuple(points([a])[0]), tuple(points([b])[0]), color, thickness, cv2.LINE_AA)
    # Smooth authored body contour: broad bouts with a narrow C-bout waist.
    sections = [(-103, 0), (-93, 34), (-77, 58), (-45, 63), (-16, 33), (13, 31), (41, 68), (83, 74), (113, 45), (124, 0)]
    ys = np.linspace(-103, 124, 150)
    knots, values = np.array(sections).T
    index = np.clip(np.searchsorted(knots, ys, side="right") - 1, 0, len(knots) - 2)
    amount = (ys - knots[index]) / (knots[index + 1] - knots[index])
    amount = amount * amount * (3 - 2 * amount)
    widths = values[index] + amount * (values[index + 1] - values[index])
    body = points(list(zip(-widths, ys)) + list(zip(widths[::-1], ys[::-1])))
    cv2.fillPoly(image, [body], (38, 90, 139), cv2.LINE_AA)
    cv2.polylines(image, [body], True, (101, 179, 226), max(1, round(2 * scale)), cv2.LINE_AA)
    inner = points(list(zip(-widths * .91, ys * .95)) + list(zip(widths[::-1] * .91, ys[::-1] * .95)))
    cv2.polylines(image, [inner], True, (62, 128, 177), 1, cv2.LINE_AA)
    neck = points([(-13, -235), (13, -235), (15, 18), (-15, 18)])
    cv2.fillPoly(image, [neck], (28, 32, 39), cv2.LINE_AA)
    scroll = tuple(points([(0, -254)])[0])
    cv2.circle(image, scroll, max(4, int(12 * scale)), (66, 132, 182), -1, cv2.LINE_AA)
    cv2.circle(image, scroll, max(2, int(6 * scale)), (25, 61, 92), 2, cv2.LINE_AA)
    for y in (-224, -204):
        line((-22, y), (22, y), (143, 176, 195), 3)
    for x in (-33, 33):
        curve = points([(x - 4, -18), (x + 6, -8), (x + 6, 15), (x - 6, 28), (x - 6, 49), (x + 4, 58)])
        cv2.polylines(image, [curve], False, (16, 36, 55), max(2, round(4 * scale)), cv2.LINE_AA)
    line((-23, 39), (23, 39), (166, 199, 224), max(2, round(4 * scale)))
    cv2.fillPoly(image, [points([(-16, 56), (16, 56), (8, 108), (-8, 108)])], (28, 33, 41), cv2.LINE_AA)
    for index, x in enumerate((-9, -3, 3, 9)):
        active = state.left_fresh and state.finger is not None and state.string == index
        color = colors[index] if active else (166, 188, 209)
        line((x, -231), (x, 99), color, 2 if active else 1)
        if active:
            marker = tuple(points([(x, -222 + state.zone * 32)])[0])
            cv2.circle(image, marker, max(3, round(5 * scale)), color, -1, cv2.LINE_AA)
