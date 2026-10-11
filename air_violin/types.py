"""Immutable messages crossing vision, control, and audio boundaries."""

from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0

    def distance(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def mirror_point(point: Point) -> Point:
    """Transform only coordinates, never infer a physical hand from screen side."""
    return Point(1.0 - point.x, point.y, point.z)


@dataclass(frozen=True)
class Hand:
    label: str  # MediaPipe's raw classification, resolved by physical calibration.
    confidence: float
    points: tuple[Point, ...]
    # World landmarks are hand-centered metric predictions, not scene positions.
    # Keep them separate: normalized image z has a different coordinate scale.
    world_points: tuple[Point, ...] | None = None

    @property
    def pinch_ratio(self) -> float:
        # Palm-relative distance is more stable as the performer moves in depth.
        return self.points[4].distance(self.points[8]) / max(
            self.points[5].distance(self.points[17]), 1e-4
        )


@dataclass(frozen=True)
class TrackingFrame:
    timestamp: float  # time.perf_counter() at camera acquisition, seconds.
    hands: tuple[Hand, ...] = ()
    inference_ms: float = 0.0

    def hand(self, label: str, confidence: float = 0.65) -> Hand | None:
        matches = [h for h in self.hands if h.label == label and h.confidence >= confidence]
        # Duplicate handedness or malformed/nonfinite output must not control sound.
        if len(matches) != 1 or len(matches[0].points) != 21:
            return None
        if not all(math.isfinite(v) for p in matches[0].points for v in (p.x, p.y, p.z)):
            return None
        return matches[0]


class PlayState(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    BOWING = "BOWING"
    RELEASING = "RELEASING"


@dataclass(frozen=True)
class AudioState:
    timestamp: float
    midi: int | None = None
    intensity: float = 0.0
    gate: bool = False
    volume: float = 0.65
    bow_direction: int = 0  # nonzero stroke reversals articulate without restarting pitch


@dataclass(frozen=True)
class MusicalState:
    audio: AudioState
    phase: PlayState = PlayState.IDLE
    string: int = 0
    zone: int = 0
    direction: int = 0
    raw_velocity: float = 0.0
    velocity: float = 0.0
    pinch: float = 0.0
    engaged: bool = False
    left_fresh: bool = False
    right_fresh: bool = False
    finger: Point | None = None
    wrist: Point | None = None
    finger_states: tuple[int, ...] = ()  # -1 unknown, 0 lifted, 1 pressed-like.
    finger_scores: tuple[float | None, ...] = ()
    fingering_mode: str = "pointing"
