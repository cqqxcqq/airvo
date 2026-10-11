"""Interactive physical-hand identification, comfortable regions, and sensitivity."""

from dataclasses import replace
import math
import statistics
from .config import Calibration, FingerCalibration, Rect, FINGER_FEATURE_VERSION
from .fingering import finger_features, FINGER_NAMES, Fingering
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
    ("Left fingers: comfortably lifted", "Hold LEFT hand like a violin grip, palm visible; all four fingers lifted.", "Keep a gentle curve. Hold 1 second, then SPACE."),
    ("Left finger 1: INDEX down", "Lower LEFT index at its knuckle; other fingers lifted.", "A gentle press is enough. Hold 1 second, then SPACE."),
    ("Left finger 2: MIDDLE down", "Keep index down and gently lower LEFT middle finger.", "Ring and little remain lifted. Hold 1 second, then SPACE."),
    ("Left finger 3: RING down", "Keep index and middle down; gently lower LEFT ring finger.", "Little remains lifted. Hold 1 second, then SPACE."),
    ("Left finger 4: LITTLE down", "Keep lower fingers down; gently lower LEFT little finger.", "No fist needed. Hold 1 second, then SPACE."),
    ("Left fingers: play-back check", "Begin with all four fingers down. SPACE starts the check.", "Then lift LITTLE, RING, MIDDLE, INDEX in that order."),
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
        self.lifted_noise = None
        self.pressed = []
        self.pressed_noise = []
        self.candidate_fingers: FingerCalibration | None = None
        self.check: Fingering | None = None
        self.verification_active = False
        self.verification_progress = 0
        self.verification_since: float | None = None
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

    @property
    def verification_note(self) -> str:
        if not self.verification_active:
            return ""
        target = 4-self.verification_progress
        cue = ("Keep all four fingers down" if target == 4 else
               f"Lift {FINGER_NAMES[target]}; keep lower fingers down" if target else
               "Lift INDEX; all four fingers comfortably lifted")
        detected = self.check.note if self.check is not None else None
        return f"{cue}. Detected: {'unknown' if detected is None else detected}; target: {target}"

    @property
    def progress_message(self) -> str:
        if self.verification_active:
            return self.verification_note
        if self.mode == "fingers" and 2 <= self.step <= 6:
            span = self.recent[-1].timestamp-self.recent[0].timestamp if self.recent else 0.0
            return f"Stable pose sample: {min(1.0, span):.1f}/1.0 seconds; SPACE records"
        return self.message

    def observe(self, frame: TrackingFrame | None, now: float) -> None:
        if frame is None or not 0 <= now - frame.timestamp <= 0.18:
            return
        if not self.recent or frame.timestamp > self.recent[-1].timestamp:
            self.recent.append(frame)
            window = 1.15 if self.mode == "fingers" and 2 <= self.step <= 7 else 0.45
            self.recent = [f for f in self.recent if frame.timestamp - f.timestamp <= window]
        if self.verification_active:
            hand = frame.hand(self.value.left_label, self.confidence)
            detected = self.check.update(hand, frame.timestamp) if hand else None
            if hand is None:
                self.check.uncertain(frame.timestamp)
            target = 4-self.verification_progress
            if detected != target or self.check.ambiguous:
                self.verification_since = None
            elif self.verification_since is None:
                self.verification_since = frame.timestamp
            elif frame.timestamp-self.verification_since >= 0.18:
                self.verification_progress += 1
                self.verification_since = None
                if self.verification_progress == 5:
                    self.verification_active = False
                    self.step += 1
                    self.recent.clear()
                    self.message = "Finger press and release verified. Mark a relaxed bowing area."
            return
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
                if self.mode == "fingers":
                    self.value = replace(self.value, fingers=self.candidate_fingers)
                self.complete = True

    def _stable_finger_features(self, now: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Use a full pose window and measure noise, rather than three frames."""
        self._stable_hand(self.value.left_label, now)
        samples = []
        stamps = []
        latest = self.recent[-1].hand(self.value.left_label, self.confidence)
        if latest is None or any(v is None for v in finger_features(latest)):
            raise ValueError("All four finger joints must be visible now. Face your palm toward the camera.")
        for frame in self.recent:
            if not 0 <= now-frame.timestamp <= 1.15:
                continue
            hand = frame.hand(self.value.left_label, self.confidence)
            features = finger_features(hand) if hand else (None,)*4
            if any(v is None for v in features):
                # A fresh image position does not imply fresh metric geometry.
                # Calibrate from a contiguous valid suffix, never old features
                # across a missing/degenerate world-landmark result.
                samples.clear()
                stamps.clear()
                continue
            samples.append(features)
            stamps.append(frame.timestamp)
        if len(samples) < 8 or stamps[-1]-stamps[0] < 0.90:
            raise ValueError("Keep all finger joints visible for one second, then press SPACE.")
        medians = tuple(statistics.median(values) for values in zip(*samples))
        noise = tuple(statistics.median(abs(sample[i]-medians[i]) for sample in samples) for i in range(4))
        if any(value > 0.025 for value in noise):
            raise ValueError("Finger pose is moving or noisy. Pause gently for one second, then SPACE.")
        return medians, noise

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
            features, noise = self._stable_finger_features(now)
            if self.step == 2:
                self.lifted = features
                self.lifted_noise = noise
            elif self.step < 7:
                index = self.step-3
                separation = abs(features[index]-self.lifted[index])
                # Demand separation above observed noise without requiring a
                # large/fatiguing curl: four MADs of the noisier endpoint.
                minimum = max(0.045, 4*max(noise[index], self.lifted_noise[index]))
                if separation < minimum:
                    raise ValueError(f"{FINGER_NAMES[index]} pose is too similar or noisy. Gently lower it at the knuckle, then SPACE.")
                for lower in range(index):
                    score = (features[lower]-self.lifted[lower])/(self.pressed[lower]-self.lifted[lower])
                    if not 0.55 <= score <= 2.2:
                        raise ValueError(f"Keep {FINGER_NAMES[lower]} down while lowering {FINGER_NAMES[index]}.")
                for higher in range(index+1, 4):
                    # These fingers have no pressed calibration yet. Reject a
                    # fist/all-down capture while allowing small natural coupling.
                    tolerance = max(0.10, 0.55*separation, 6*(noise[higher]+self.lifted_noise[higher]))
                    if abs(features[higher]-self.lifted[higher]) > tolerance:
                        raise ValueError(f"Keep {FINGER_NAMES[higher]} lifted; only the requested and lower fingers should be down.")
                self.pressed.append(features[index])
                self.pressed_noise.append(noise[index])
            else:
                calibration = FingerCalibration(self.lifted, tuple(self.pressed), feature_version=FINGER_FEATURE_VERSION)
                scores = [(value-a)/(b-a) for value, a, b in zip(features, calibration.lifted, calibration.pressed)]
                if any(not 0.55 <= value <= 2.2 for value in scores):
                    raise ValueError("Begin the check with all four fingers gently down, then SPACE.")
                self.candidate_fingers = calibration
                self.check = Fingering(calibration)
                self.verification_active = True
                self.verification_progress = 0
                self.verification_since = None
                self.message = "Lift fingers in order to prove notes can change. No bow motion is needed."
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
        if self.started is not None or self.complete or self.verification_active:
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
                if self.started is not None or self.verification_active:
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
