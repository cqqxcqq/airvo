"""Equal temperament and a five-zone fingerboard with spatial/temporal hysteresis."""

import math
from .config import Controls, Rect
from .types import Point

STRINGS = ("G", "D", "A", "E")
OPEN_MIDI = (55, 62, 69, 76)
EASY_OFFSETS = (0, 2, 4, 5, 7)
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_frequency(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def note_name(midi: int | None) -> str:
    return "--" if midi is None else f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


class Fingerboard:
    def __init__(self, region: Rect, controls: Controls):
        self.region = region
        self.controls = controls
        self.selected: tuple[int, int] | None = None
        self.pending: tuple[int, int] | None = None
        self.pending_since = 0.0
        self.point: Point | None = None
        self.timestamp: float | None = None

    def reset(self) -> None:
        self.selected = self.pending = self.point = self.timestamp = None

    def update(self, point: Point, timestamp: float) -> tuple[int, int] | None:
        if self.timestamp is not None and timestamp <= self.timestamp:
            return self.selected
        if not self.region.contains(point.x, point.y):
            self.reset()
            return None
        if self.point is None or self.timestamp is None:
            self.point = point
        else:
            dt = timestamp - self.timestamp
            alpha = 1 - math.exp(-dt / self.controls.position_tau)
            self.point = Point(self.point.x + alpha * (point.x - self.point.x),
                               self.point.y + alpha * (point.y - self.point.y))
        self.timestamp = timestamp
        u, v = self.region.relative(self.point.x, self.point.y)
        target = (min(3, int(v * 4)), min(4, int(u * 5)))
        if self.selected is None:
            self.selected = target
            return target
        string, zone = self.selected
        h = self.controls.hysteresis
        # Keep each axis independently inside its expanded cell. A small y jitter
        # must not reset a deliberate x note change (and vice versa).
        if (string - h) / 4 <= v <= (string + 1 + h) / 4:
            target = (string, target[1])
        if (zone - h) / 5 <= u <= (zone + 1 + h) / 5:
            target = (target[0], zone)
        if target == self.selected:
            self.pending = None
        elif target != self.pending:
            self.pending = target
            self.pending_since = timestamp
        elif timestamp - self.pending_since >= self.controls.note_hold:
            self.selected = target
            self.pending = None
        return self.selected

    @property
    def midi(self) -> int | None:
        if self.selected is None:
            return None
        string, zone = self.selected
        return OPEN_MIDI[string] + EASY_OFFSETS[zone]
