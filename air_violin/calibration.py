"""Interactive physical-hand identification, comfortable regions, and sensitivity."""

from dataclasses import replace
import math
import statistics
from .config import Calibration, FingerCalibration, Rect
from .fingering import finger_features, FINGER_NAMES
from .types import Hand, Point, TrackingFrame

STEPS = (
    ("Your physical LEFT hand", "Raise only your LEFT hand. Keep the other out of view.", "Press SPACE when its landmarks are stable."),
    ("Your physical RIGHT hand", "Raise only your RIGHT hand. Keep the other out of view.", "Press SPACE to confirm its identity."),
    ("Fingerboard: first corner", "Move your physical LEFT index tip to any corner of your playing area.", "Press SPACE. Use a comfortable, reachable area."),
    ("Fingerboard: opposite corner", "Move the SAME index tip diagonally to the opposite corner.", "Either corner order works. Press SPACE."),
    ("Bowing region: first corner", "Move your physical RIGHT wrist to any corner of your bowing area.", "Press SPACE to mark the first corner."),
    ("Bowing region: opposite corner", "Move the SAME wrist diagonally to the opposite corner.", "Leave room for horizontal strokes. Press SPACE."),
    ("Precision bow: open hand", "Hold RIGHT thumb and index comfortably apart.", "Press SPACE. Keep your palm facing the camera."),
    ("Precision bow: pinch", "Touch RIGHT thumb and index fingertips gently.", "Press SPACE to set separate on/off thresholds."),
    ("Comfortable bow speed", "Press SPACE, then sweep your RIGHT wrist for 3 seconds.", "Stay inside the bow region and use relaxed strokes."),
)

FINGER_STEPS = STEPS[:2] + (
    ("Left fingers: comfortably lifted", "Hold LEFT hand like a violin grip, palm visible; all four fingers lifted.", "Keep a natural gentle curve. SPACE records your open pose."),
    ("Left finger 1: INDEX down", "Lower/curl LEFT index as if stopping a string; other fingers lifted.", "Hold the pose still, then press SPACE."),
    ("Left finger 2: MIDDLE down", "Keep index down and lower/curl LEFT middle finger.", "Ring and little remain lifted. Hold still, then SPACE."),
    ("Left finger 3: RING down", "Keep index and middle down; lower/curl LEFT ring finger.", "Little finger remains lifted. Hold still, then SPACE."),
    ("Left finger 4: LITTLE down", "Keep the lower fingers down; lower/curl LEFT little finger.", "Use a relaxed grip rather than a tight fist. Hold still, then SPACE."),
    ("Left fingers: release check", "Lift all four fingers back to your original relaxed open pose.", "SPACE checks that lifting returns to the open string."),
) + STEPS[4:]


class Calibrator:
    def __init__(self, mirrored: bool, confidence: float = 0.65,
                 mode: str = "pointing", initial: Calibration | None = None):
        if mode not in ("pointing", "fingers"):
            raise ValueError("Calibration mode must be pointing or fingers")
        self.mode = mode
        self.steps = FINGER_STEPS if mode == "fingers" else STEPS
        self.step = 0
        self.confidence = confidence
        self.value = initial if initial is not None and initial.mirrored == mirrored else Calibration(mirrored=mirrored)
        if mode == "fingers":
            self.value = replace(self.value, version=2)
        self.lifted = None
        self.pressed = []
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
        return self.steps[self.step]

    @property
    def corner_kind(self) -> str | None:
        if self.mode == "fingers":
            return "bow" if self.step in (8, 9) else None
        return "board" if self.step in (2, 3) else "bow" if self.step in (4, 5) else None

    @property
    def pose_preview(self) -> tuple[float | None, ...]:
        """Live calibrated press scores for the fingers recorded so far."""
        if self.mode != "fingers" or not self.recent or self.lifted is None:
            return (None,) * 4
        latest = self.recent[-1]
        hand = latest.hand(self.value.left_label, self.confidence)
        if hand is None:
            return (None,) * 4
        features = finger_features(hand)
        scores = []
        for index, feature in enumerate(features):
            if feature is None or index >= len(self.pressed):
                scores.append(None)
            else:
                scores.append((feature-self.lifted[index])/(self.pressed[index]-self.lifted[index]))
        return tuple(scores)

    def observe(self, frame: TrackingFrame | None, now: float) -> None:
        if frame is None or not 0 <= now - frame.timestamp <= 0.18:
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
            if not 0 <= now - frame.timestamp <= 0.45:
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
        if not self.recent or not 0 <= now - self.recent[-1].timestamp <= 0.18:
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
                             statistics.median(h.points[i].y for h in valid),
                             statistics.median(h.points[i].z for h in valid)) for i in range(21))
        world_hands = [h for h in valid if h.world_points is not None and len(h.world_points) == 21]
        world = (tuple(Point(*(statistics.median(getattr(h.world_points[i], axis) for h in world_hands)
                               for axis in ("x", "y", "z"))) for i in range(21))
                 if len(world_hands) == len(valid) else None)
        return Hand(valid[-1].label, min(h.confidence for h in valid), points, world)

    def _finger_step(self, now: float) -> None:
        label = self.value.left_label if self.step < 8 else self.value.right_label
        hand = self._stable_hand(label, now)
        if 2 <= self.step <= 7:
            features = finger_features(hand)
            if any(v is None for v in features):
                raise ValueError("All four finger joints must be visible. Face your palm toward the camera.")
            if self.step == 2:
                self.lifted = features
            elif self.step < 7:
                index = self.step-3
                if abs(features[index]-self.lifted[index]) < 0.06:
                    raise ValueError(f"{FINGER_NAMES[index]} pose is too similar. Bend it more clearly, then SPACE.")
                self.pressed.append(features[index])
            else:
                calibration = FingerCalibration(self.lifted, tuple(self.pressed))
                scores = [(value-a)/(b-a) for value, a, b in zip(features, calibration.lifted, calibration.pressed)]
                if any(not -0.75 <= value <= 0.45 for value in scores):
                    raise ValueError("Lift all four fingers to the original relaxed pose; keep your palm visible.")
                self.value = replace(self.value, fingers=calibration)
        elif self.step == 8:
            point = hand.points[0]
            if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                raise ValueError("Keep the wrist inside the camera view.")
            self.corner = point
        elif self.step == 9:
            point = hand.points[0]
            rect = Rect(min(self.corner.x, point.x), min(self.corner.y, point.y),
                        max(self.corner.x, point.x), max(self.corner.y, point.y))
            if rect.y1-rect.y0 < 0.24:
                raise ValueError("Use at least 24% frame height for four bow/string bands.")
            self.value = replace(self.value, bow=rect)
        elif self.step == 10:
            self.open_pinch = hand.pinch_ratio
        elif self.step == 11:
            closed = hand.pinch_ratio
            spread = self.open_pinch-closed
            if spread < 0.25:
                raise ValueError("Open and pinch poses are too similar. C restarts calibration.")
            self.value = replace(self.value, pinch_on=max(0.01, closed+0.35*spread), pinch_off=closed+0.60*spread)
        else:
            self.started, self.previous = now, None
            self.speeds.clear()
            self.message = "Sweep now..."

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
            elif self.mode == "fingers":
                self._finger_step(now)
                if self.started is not None:
                    return
            else:
                label = self.value.left_label if self.step < 4 else self.value.right_label
                hand = self._stable_hand(label, now)
                if self.step in (2, 4):
                    point = hand.points[8 if self.step == 2 else 0]
                    if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                        raise ValueError("Keep the corner inside the camera view, then press SPACE.")
                    self.corner = point
                elif self.step in (3, 5):
                    point = hand.points[8 if self.step == 3 else 0]
                    if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
                        raise ValueError("Keep the opposite corner inside the camera view, then press SPACE.")
                    # Screen direction is easy to confuse in a mirrored view.
                    # Order the bounds, preserving the user's physical-hand binding.
                    rect = Rect(min(self.corner.x, point.x), min(self.corner.y, point.y),
                                max(self.corner.x, point.x), max(self.corner.y, point.y))
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
