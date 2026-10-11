"""Interactive physical-hand identification, comfortable regions, and sensitivity."""

from dataclasses import replace
import math
import statistics
from .config import Calibration, Rect
from .types import Hand, Point, TrackingFrame

STEPS = (
    ("Your physical LEFT hand", "Raise only your LEFT hand. Keep the other out of view.", "Press SPACE when its landmarks are stable."),
    ("Your physical RIGHT hand", "Raise only your RIGHT hand. Keep the other out of view.", "Press SPACE to confirm its identity."),
    ("Fingerboard: top-left", "Move LEFT index tip to a comfortable upper-left corner.", "Press SPACE. The board will span four string bands."),
    ("Fingerboard: bottom-right", "Move LEFT index tip to the opposite lower-right corner.", "Use a wide, reachable rectangle. Press SPACE."),
    ("Bowing region: top-left", "Move RIGHT wrist to its comfortable upper-left corner.", "Press SPACE to mark the corner."),
    ("Bowing region: bottom-right", "Move RIGHT wrist to the opposite lower-right corner.", "Leave room for horizontal strokes. Press SPACE."),
    ("Precision bow: open hand", "Hold RIGHT thumb and index comfortably apart.", "Press SPACE. Keep your palm facing the camera."),
    ("Precision bow: pinch", "Touch RIGHT thumb and index fingertips gently.", "Press SPACE to set separate on/off thresholds."),
    ("Comfortable bow speed", "Press SPACE, then sweep your RIGHT wrist for 3 seconds.", "Stay inside the bow region and use relaxed strokes."),
)


class Calibrator:
    def __init__(self, mirrored: bool, confidence: float = 0.65):
        self.step = 0
        self.confidence = confidence
        self.value = Calibration(mirrored=mirrored)
        self.message = ""
        self.corner: Point | None = None
        self.open_pinch = 0.0
        self.started: float | None = None
        self.previous: tuple[float, float] | None = None
        self.speeds: list[float] = []
        self.complete = False
        self.recent: list[TrackingFrame] = []

    @property
    def instructions(self) -> tuple[str, str, str]:
        return STEPS[self.step]

    def observe(self, frame: TrackingFrame | None, now: float) -> None:
        if frame is None or now - frame.timestamp > 0.18:
            return
        if not self.recent or frame.timestamp > self.recent[-1].timestamp:
            self.recent.append(frame)
            self.recent = [f for f in self.recent if frame.timestamp - f.timestamp <= 0.45]
        if self.started is None or self.complete:
            return
        hand = frame.hand(self.value.right_label, self.confidence)
        if hand and self.value.bow.contains(hand.points[0].x, hand.points[0].y):
            point = hand.points[0]
            if self.previous and frame.timestamp > self.previous[1]:
                velocity = abs(point.x - self.previous[0]) / (frame.timestamp - self.previous[1])
                if 0.045 < velocity < 4.0:
                    self.speeds.append(velocity)
            self.previous = (point.x, frame.timestamp)
        if now - self.started >= 3.0:
            self.started = None
            self.previous = None
            if len(self.speeds) < 12:
                self.message = "Not enough tracked motion. SPACE to retry relaxed horizontal sweeps."
                self.speeds.clear()
            else:
                speed = sorted(self.speeds)[int(0.80 * (len(self.speeds) - 1))]
                self.value = replace(self.value, bow_full_speed=max(0.15, min(4.0, speed)))
                self.complete = True

    def _stable_hand(self, label: str | None, now: float) -> Hand:
        valid = []
        for frame in self.recent:
            if now - frame.timestamp > 0.45:
                continue
            if label is None:
                if len(frame.hands) != 1:
                    continue
                hand = frame.hand(frame.hands[0].label, self.confidence)
            else:
                hand = frame.hand(label, self.confidence)
            if hand:
                valid.append(hand)
        if len(valid) < 3 or len({h.label for h in valid}) != 1:
            raise ValueError("Hold the requested hand steadily for a moment, then press SPACE.")
        if not self.recent or now - self.recent[-1].timestamp > 0.18:
            raise ValueError("Tracking is stale. Move into camera view and try again.")
        latest = self.recent[-1]
        if (label is None and len(latest.hands) != 1) or latest.hand(valid[-1].label, self.confidence) is None:
            raise ValueError("The requested hand is no longer clearly visible. Try again.")
        for index in (0, 8):
            if any(max(getattr(h.points[index], axis) for h in valid) -
                   min(getattr(h.points[index], axis) for h in valid) > 0.07 for axis in ("x", "y")):
                raise ValueError("Pause your hand briefly to mark a stable position.")
        # Median over a short window reduces single-frame calibration mistakes.
        points = tuple(Point(statistics.median(h.points[i].x for h in valid),
                             statistics.median(h.points[i].y for h in valid)) for i in range(21))
        return Hand(valid[-1].label, min(h.confidence for h in valid), points)

    def advance(self, now: float) -> None:
        if self.started is not None or self.complete:
            return
        try:
            if self.step < 2:
                hand = self._stable_hand(None, now)
                if self.step == 0:
                    other = "Right" if hand.label == "Left" else "Left"
                    self.value = replace(self.value, left_label=hand.label, right_label=other)
                elif hand.label != self.value.right_label:
                    raise ValueError("Same label as your left hand. Show only your physical RIGHT hand.")
            else:
                label = self.value.left_label if self.step < 4 else self.value.right_label
                hand = self._stable_hand(label, now)
                if self.step in (2, 4):
                    self.corner = hand.points[8 if self.step == 2 else 0]
                elif self.step in (3, 5):
                    point = hand.points[8 if self.step == 3 else 0]
                    rect = Rect(self.corner.x, self.corner.y, point.x, point.y)
                    if self.step == 3 and rect.y1 - rect.y0 < 0.24:
                        raise ValueError("Use at least 24% frame height for four comfortable string bands.")
                    self.value = replace(self.value, **{"board" if self.step == 3 else "bow": rect})
                elif self.step == 6:
                    self.open_pinch = hand.pinch_ratio
                elif self.step == 7:
                    closed = hand.pinch_ratio
                    spread = self.open_pinch - closed
                    if spread < 0.25:
                        raise ValueError("Pinch is too similar to open hand. Press C to restart calibration.")
                    self.value = replace(self.value, pinch_on=max(0.01, closed + 0.35 * spread),
                                         pinch_off=closed + 0.60 * spread)
                else:
                    self.started = now
                    self.speeds.clear()
                    self.previous = None
                    self.message = "Sweep now..."
                    return
            self.step += 1
            self.recent.clear()
            self.message = ""
        except (ValueError, TypeError) as exc:
            self.message = str(exc)
