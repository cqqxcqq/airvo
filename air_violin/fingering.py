"""Calibrated finger articulation, independent of screen position and bowing.

This estimates a pressed-like pose, not physical pressure or contact. MediaPipe
does not report reliable per-finger visibility; degenerate geometry is unknown.
"""

import math
from .config import Controls, FingerCalibration, Rect, FINGER_FEATURE_VERSION
from .types import Hand, Point

FINGER_JOINTS = ((5, 6, 7, 8), (9, 10, 11, 12),
                 (13, 14, 15, 16), (17, 18, 19, 20))
FINGER_NAMES = ("INDEX", "MIDDLE", "RING", "LITTLE")


def _distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


def _angle(a: Point, joint: Point, b: Point) -> float:
    u = (a.x-joint.x, a.y-joint.y, a.z-joint.z)
    v = (b.x-joint.x, b.y-joint.y, b.z-joint.z)
    denominator = _distance(a, joint) * _distance(b, joint)
    return math.acos(max(-1.0, min(1.0, sum(x*y for x, y in zip(u, v)) / denominator)))


def _vector(a: Point, b: Point) -> tuple[float, float, float]:
    return b.x-a.x, b.y-a.y, b.z-a.z


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x*y for x, y in zip(a, b))


def finger_features(hand: Hand) -> tuple[float | None, ...]:
    """Measure a gentle finger press, including motion at its knuckle (MCP).

    A violin-like press can lower an already curved finger without changing its
    PIP/DIP angles. The previous feature missed that entire movement. A local
    palm plane separates knuckle flexion from finger spreading and camera roll.
    All cues use metric world landmarks, never mixed image x/y and relative z.
    Absolute palm-normal height makes reflecting the hand preserve the feature;
    calibration determines whether a user's comfortable press increases or
    decreases it. These are pose estimates, not pressure measurements.
    """
    points = hand.world_points
    if points is None or len(points) != 21:
        return (None,) * 4
    if not all(math.isfinite(v) for p in points for v in (p.x, p.y, p.z)):
        return (None,) * 4
    palm = _distance(points[5], points[17])
    if not 0.020 <= palm <= 0.160 or _distance(points[0], points[9]) < 0.012:
        return (None,) * 4
    across = _vector(points[17], points[5])
    forward = _vector(points[0], points[9])
    cross = (across[1]*forward[2]-across[2]*forward[1],
             across[2]*forward[0]-across[0]*forward[2],
             across[0]*forward[1]-across[1]*forward[0])
    norm = math.sqrt(_dot(cross, cross))
    if norm < palm*_distance(points[0], points[9])*0.20:
        return (None,) * 4
    normal = tuple(v/norm for v in cross)
    values = []
    for joints in FINGER_JOINTS:
        base, pip, dip, tip = (points[i] for i in joints)
        lengths = (_distance(base, pip), _distance(pip, dip), _distance(dip, tip))
        if any(not 0.002 <= length <= 0.080 for length in lengths):
            values.append(None)
            continue
        knuckle_axis = _vector(points[0], base)
        along_plane = tuple(v-_dot(knuckle_axis, normal)*n for v, n in zip(knuckle_axis, normal))
        along_norm = math.sqrt(_dot(along_plane, along_plane))
        if along_norm < 0.012:
            values.append(None)
            continue
        along = tuple(v/along_norm for v in along_plane)
        first = _vector(base, pip)
        mcp_flexion = math.atan2(abs(_dot(first, normal)), _dot(first, along))
        distal_bend = math.pi - (_angle(base, pip, dip) + _angle(pip, dip, tip))/2
        height = abs(_dot(_vector(base, tip), normal))/sum(lengths)
        shortening = 1.0 - _distance(base, tip) / sum(lengths)
        values.append(0.65*mcp_flexion + 0.20*distal_bend + 0.10*height + 0.05*shortening)
    return tuple(values)


class Fingering:
    def __init__(self, calibration: FingerCalibration | None):
        self.calibration = calibration
        self.reset()

    def reset(self) -> None:
        self.states = [-1] * 4
        self.scores: list[float | None] = [None] * 4
        self.pending: list[int | None] = [None] * 4
        self.pending_since = [0.0] * 4
        self.filtered: list[float | None] = [None] * 4
        self.timestamp = -math.inf
        self.valid_timestamp = -math.inf
        self.note: int | None = None
        self.ambiguous = True

    def _resolve(self, timestamp: float) -> None:
        down = [index for index, state in enumerate(self.states) if state == 1]
        highest = max(down, default=-1)
        # Unknown lower fingers cannot affect a known higher stopped note. An
        # unknown higher finger may stop that note, so it must never imply open.
        certain = all(state == 0 for state in self.states[highest+1:])
        if certain:
            self.ambiguous = False
            self.note = highest + 1
            self.valid_timestamp = timestamp
        else:
            self.ambiguous = True
        self.expire(timestamp)

    def expire(self, now: float) -> int | None:
        timeout = self.calibration.unknown_timeout if self.calibration else 0.080
        # Slow but valid inference is governed by the instrument's tracking
        # watchdog. The shorter hold only applies to an observed uncertain pose.
        if self.ambiguous and now - self.valid_timestamp > timeout:
            self.note = None
        return self.note

    def uncertain(self, timestamp: float) -> None:
        if timestamp > self.timestamp:
            self.timestamp = timestamp
            self.ambiguous = True
            self.states = [-1] * 4
            self.scores = [None] * 4
            self.pending = [None] * 4
            self.filtered = [None] * 4
        self.expire(timestamp)

    def update(self, hand: Hand, timestamp: float) -> int | None:
        if timestamp <= self.timestamp:
            return self.note
        if self.calibration is None or self.calibration.feature_version != FINGER_FEATURE_VERSION:
            self.uncertain(timestamp)
            return None
        features = finger_features(hand)
        dt = timestamp - self.timestamp
        self.timestamp = timestamp
        for index, value in enumerate(features):
            if value is None:
                self.states[index] = -1
                self.scores[index] = self.filtered[index] = None
                self.pending[index] = None
                continue
            lifted = self.calibration.lifted[index]
            pressed = self.calibration.pressed[index]
            score = (value-lifted) / (pressed-lifted)
            # Strongly out-of-calibration poses (a fist or another grip) are
            # ambiguous. Do not extrapolate them into an arbitrary melody.
            if not -1.0 <= score <= 2.2:
                self.states[index] = -1
                self.scores[index] = self.filtered[index] = None
                self.pending[index] = None
                continue
            alpha = 1.0 if self.filtered[index] is None else 1-math.exp(-dt/0.025)
            self.filtered[index] = score if self.filtered[index] is None else self.filtered[index] + alpha*(score-self.filtered[index])
            self.scores[index] = self.filtered[index]
            target = (1 if self.scores[index] >= self.calibration.engage else
                      0 if self.scores[index] <= self.calibration.release else self.states[index])
            if target < 0:
                self.pending[index] = None
            elif target == self.states[index]:
                self.pending[index] = None
            elif target != self.pending[index]:
                self.pending[index], self.pending_since[index] = target, timestamp
                if self.calibration.dwell == 0:
                    self.states[index], self.pending[index] = target, None
            elif timestamp-self.pending_since[index] >= self.calibration.dwell:
                self.states[index], self.pending[index] = target, None
        self._resolve(timestamp)
        return self.note


class BowStringSelector:
    """Vertical wrist bands select strings; horizontal strokes remain bow input."""

    def __init__(self, region: Rect, controls: Controls):
        self.region, self.controls = region, controls
        self.string = 1  # Start on D, matching the first tutorial/practice melody.
        self.pending = None
        self.pending_since = 0.0

    def update(self, wrist: Point, timestamp: float) -> int:
        if not self.region.contains(wrist.x, wrist.y):
            self.pending = None
            return self.string
        v = (wrist.y-self.region.y0) / (self.region.y1-self.region.y0)
        h = self.controls.hysteresis
        target = min(3, int(v*4))
        if (self.string-h)/4 <= v <= (self.string+1+h)/4:
            target = self.string
        if target == self.string:
            self.pending = None
        elif target != self.pending:
            self.pending, self.pending_since = target, timestamp
        elif timestamp-self.pending_since >= self.controls.note_hold:
            self.string, self.pending = target, None
        return self.string
