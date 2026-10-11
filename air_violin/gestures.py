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
    sustaining: bool = False  # Assisted phrase is latched; movement is independent.


class BowInterpreter:
    def __init__(self, calibration: Calibration, controls: Controls, mode: str = "auto"):
        self.calibration = calibration
        self.controls = controls
        if mode not in ("assisted", "auto", "precision"):
            raise ValueError("Bow mode must be assisted, auto or precision")
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.previous: tuple[float, float] | None = None
        self.speed = 0.0
        self.signed_velocity = 0.0
        self.pinched = False
        self.last_motion = -math.inf
        self.last_intensity = 0.0
        self.assist_origin: float | None = None
        self.sustaining = False
        self.state = BowState()

    def set_mode(self, mode: str) -> None:
        if mode not in ("assisted", "auto", "precision"):
            raise ValueError("Bow mode must be assisted, auto or precision")
        self.mode = mode
        self.reset()

    def stop_assisted(self) -> None:
        """Stop explicitly, and arm the next stroke from the current position."""
        if self.mode != "assisted":
            return
        self.sustaining = False
        self.assist_origin = self.previous[0] if self.previous else None
        self.last_motion = -math.inf
        self.last_intensity = 0.0
        old = self.state
        inside = old.wrist is not None and self.calibration.bow.contains(old.wrist.x, old.wrist.y)
        self.state = BowState(inside and not self.pinched, False, 0.0, old.raw_velocity,
                              old.velocity, old.direction, old.pinch, old.wrist)

    def toggle_assisted(self, timestamp: float) -> bool:
        """Keyboard sustain respects the same acquisition safety as gestures."""
        if self.mode != "assisted":
            return False
        if self.sustaining:
            self.stop_assisted()
            return False
        if (self.previous is None or not 0 <= timestamp-self.previous[1] <= self.controls.tracking_timeout
                or not self.state.engaged or self.pinched):
            return False
        if self.state.wrist is None or not self.calibration.bow.contains(self.state.wrist.x, self.state.wrist.y):
            return False
        self.sustaining = True
        self.last_intensity = 0.42
        old = self.state
        self.state = BowState(old.engaged, old.moving, self.last_intensity,
                              old.raw_velocity, old.velocity, old.direction,
                              old.pinch, old.wrist, True)
        return True

    def update(self, hand: Hand, timestamp: float) -> BowState:
        wrist = hand.points[0]
        if self.previous and timestamp <= self.previous[1]:
            self.stop_assisted()
            return self.state
        if self.previous and timestamp - self.previous[1] > self.controls.tracking_timeout:
            self.reset()
        pinch = hand.pinch_ratio
        if pinch <= self.calibration.pinch_on:
            self.pinched = True
        elif pinch >= self.calibration.pinch_off:
            self.pinched = False
        inside = self.calibration.bow.contains(wrist.x, wrist.y)
        visible = 0 <= wrist.x <= 1 and 0 <= wrist.y <= 1
        # In assisted mode pinch is an intentional STOP, not engagement. An
        # unpinch merely arms the next stroke; it cannot resume an old phrase.
        engaged = (visible and not self.pinched and (inside or self.sustaining)
                   if self.mode == "assisted" else inside and (self.mode == "auto" or self.pinched))
        if self.mode == "assisted" and (not engaged or self.assist_origin is None):
            self.stop_assisted()
            self.assist_origin = wrist.x
        raw = 0.0
        if self.previous:
            dt = timestamp - self.previous[1]
            raw = (wrist.x - self.previous[0]) / dt
            # Reject implausible detector teleports; reacquire before sounding.
            if abs(raw) > 6.0 or abs(wrist.x - self.previous[0]) > 0.25:
                self.reset()
                engaged = False
                raw = 0.0
                self.assist_origin = wrist.x
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
            if self.mode == "assisted" and self.assist_origin is not None:
                # A real small excursion starts a phrase. Alternating subpixel
                # jitter does not accumulate into a fictitious bow stroke.
                excursion = max(0.012, 0.035 * (self.calibration.bow.x1-self.calibration.bow.x0))
                if abs(wrist.x-self.assist_origin) >= excursion:
                    self.sustaining = True
        moving = engaged and timestamp - self.last_motion <= self.controls.reversal_grace
        if not engaged:
            self.last_motion = -math.inf
        intensity = max(0.08, self.last_intensity) if moving else 0.0
        if self.mode == "assisted" and self.sustaining:
            # A comfortable stationary hand maintains a phrase. Freshness and
            # valid fingering still gate audio in Instrument, never in the GUI.
            intensity = max(0.42, min(0.85, intensity))
        direction = 1 if self.signed_velocity > self.controls.deadzone else -1 if self.signed_velocity < -self.controls.deadzone else 0
        self.state = BowState(engaged, moving, intensity, raw, self.signed_velocity,
                              direction, pinch, wrist, self.sustaining)
        return self.state
