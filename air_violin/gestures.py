"""Replaceable heuristic bow interpreter; all derivatives use elapsed seconds."""

from dataclasses import dataclass
import math
from .config import Calibration, Controls
from .types import Hand, Point


@dataclass(frozen=True)
class BowState:
    engaged: bool = False
    moving: bool = False
    intensity: float = 0.0
    raw_velocity: float = 0.0
    velocity: float = 0.0
    direction: int = 0
    pinch: float = 0.0
    wrist: Point | None = None


class BowInterpreter:
    def __init__(self, calibration: Calibration, controls: Controls, mode: str = "auto"):
        self.calibration = calibration
        self.controls = controls
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.previous: tuple[float, float] | None = None
        self.speed = 0.0
        self.signed_velocity = 0.0
        self.pinched = False
        self.last_motion = -math.inf
        self.last_intensity = 0.0
        self.state = BowState()

    def set_mode(self, mode: str) -> None:
        if mode not in ("auto", "precision"):
            raise ValueError("Bow mode must be auto or precision")
        self.mode = mode
        self.reset()

    def update(self, hand: Hand, timestamp: float) -> BowState:
        wrist = hand.points[0]
        if self.previous and timestamp <= self.previous[1]:
            return self.state
        if self.previous and timestamp - self.previous[1] > self.controls.tracking_timeout:
            self.reset()
        pinch = hand.pinch_ratio
        if pinch <= self.calibration.pinch_on:
            self.pinched = True
        elif pinch >= self.calibration.pinch_off:
            self.pinched = False
        inside = self.calibration.bow.contains(wrist.x, wrist.y)
        engaged = inside and (self.mode == "auto" or self.pinched)
        raw = 0.0
        if self.previous:
            dt = timestamp - self.previous[1]
            raw = (wrist.x - self.previous[0]) / dt
            # Reject implausible detector teleports; reacquire before sounding.
            if abs(raw) > 6.0 or abs(wrist.x - self.previous[0]) > 0.25:
                self.reset()
                engaged = False
                raw = 0.0
            else:
                alpha = 1 - math.exp(-dt / self.controls.velocity_tau)
                self.speed += alpha * (abs(raw) - self.speed)
                self.signed_velocity += alpha * (raw - self.signed_velocity)
        self.previous = (wrist.x, timestamp)
        # Raw deadzone distinguishes a genuine stop from a smoothing tail.
        meaningful = (engaged and abs(raw) > self.controls.deadzone
                      and abs(self.signed_velocity) > self.controls.deadzone)
        if meaningful:
            self.last_motion = timestamp
            self.last_intensity = min(1.0, self.speed / self.calibration.bow_full_speed)
        moving = engaged and timestamp - self.last_motion <= self.controls.reversal_grace
        if not engaged:
            self.last_motion = -math.inf
        intensity = max(0.08, self.last_intensity) if moving else 0.0
        direction = 1 if self.signed_velocity > self.controls.deadzone else -1 if self.signed_velocity < -self.controls.deadzone else 0
        self.state = BowState(engaged, moving, intensity, raw, self.signed_velocity,
                              direction, pinch, wrist)
        return self.state
