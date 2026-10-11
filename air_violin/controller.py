"""Musical state machine, independent of display rate and audio implementation."""

import math
from .config import Calibration, Controls
from .gestures import BowInterpreter
from .mapping import Fingerboard
from .mapping import OPEN_MIDI, EASY_OFFSETS
from .fingering import Fingering, BowStringSelector
from .types import AudioState, MusicalState, PlayState, TrackingFrame


class Instrument:
    def __init__(self, calibration: Calibration, controls: Controls, bow_mode: str = "auto",
                 *, fingering_mode: str = "pointing", string_override: int | None = None):
        if fingering_mode not in ("pointing", "fingers"):
            raise ValueError("Fingering mode must be pointing or fingers")
        self.controls = controls
        self.calibration = calibration
        self.fingering_mode = fingering_mode
        self.fingering = Fingering(calibration.fingers)
        self.strings = BowStringSelector(calibration.bow, controls)
        self.string_override = None
        self.set_string_override(string_override)
        self.finger_point = None
        self.board = Fingerboard(calibration.board, controls)
        self.bow = BowInterpreter(calibration, controls, bow_mode)
        self.last_frame = self.last_left = self.last_right = -math.inf
        self.delivery_age = 0.0
        self.phase = PlayState.IDLE
        self.release_started = -math.inf
        self.last_midi: int | None = None

    def set_string_override(self, string: int | None) -> None:
        if string is not None and string not in range(4):
            raise ValueError("String override must be G/D/A/E index 0..3 or None")
        self.string_override = string

    def toggle_assisted(self, now: float) -> bool:
        return self.bow.toggle_assisted(now)

    def stop_assisted(self) -> None:
        self.bow.stop_assisted()

    def update(self, frame: TrackingFrame | None, now: float, volume: float = 0.65) -> MusicalState:
        if frame is not None and frame.timestamp < self.last_frame:
            self.bow.stop_assisted()
        if frame is not None and self.last_frame < frame.timestamp <= now:
            self.last_frame = frame.timestamp
            if now - frame.timestamp <= self.controls.tracking_timeout:
                # Motion grace is measured at capture. Account for the measured
                # inference delivery age so an 8 ms heartbeat cannot cancel a
                # freshly delivered moving result. Acquisition watchdogs remain
                # unchanged, and an explicitly stopped result always releases.
                self.delivery_age = min(0.120, max(0.0, now-frame.timestamp))
                left = frame.hand(self.calibration.left_label, self.controls.confidence)
                right = frame.hand(self.calibration.right_label, self.controls.confidence)
                if left:
                    if frame.timestamp - self.last_left > self.controls.tracking_timeout:
                        self.board.reset()
                        self.fingering.reset()
                    self.last_left = frame.timestamp
                    if self.fingering_mode == "fingers":
                        self.fingering.update(left, frame.timestamp)
                        index = self.fingering.note or 1
                        self.finger_point = left.points[(8, 8, 12, 16, 20)[index]]
                    else:
                        self.board.update(left.points[8], frame.timestamp)
                elif self.fingering_mode == "fingers":
                    self.fingering.uncertain(frame.timestamp)
                if right:
                    self.last_right = frame.timestamp
                    bow = self.bow.update(right, frame.timestamp)
                    if (self.bow.mode != "assisted" or not bow.sustaining
                            or (bow.moving and abs(bow.raw_velocity) > self.controls.deadzone)):
                        self.strings.update(right.points[0], frame.timestamp)
                    else:
                        # Lowering a resting assisted hand is ergonomic, not a
                        # string change. Only new horizontal strokes accrue dwell.
                        self.strings.pending = None
        left_fresh = now - self.last_left <= self.controls.tracking_timeout
        right_fresh = now - self.last_right <= self.controls.tracking_timeout
        if not left_fresh:
            self.board.reset()
            self.fingering.reset()
            self.finger_point = None
            self.bow.stop_assisted()
        if not right_fresh:
            self.bow.reset()
        if self.fingering_mode == "fingers":
            note = self.fingering.expire(now)
            string = self.strings.string if self.string_override is None else self.string_override
            zone = note or 0
            midi = OPEN_MIDI[string] + EASY_OFFSETS[zone] if note is not None else None
            selected_point = self.finger_point if note is not None else None
        else:
            midi = self.board.midi
            string, zone = self.board.selected or (0, 0)
            selected_point = self.board.point
        if midi is not None:
            self.last_midi = midi
        elif self.last_midi is not None:
            # Once a played fingering expires, the old assisted phrase cannot
            # restart itself when an occluded hand happens to return.
            self.bow.stop_assisted()
        bow = self.bow.state
        # Also expire the motion grace when inference produces no new results.
        physical_motion = (bow.moving and now-self.bow.last_motion
                           <= self.controls.reversal_grace+self.delivery_age)
        bow_active = bow.sustaining if self.bow.mode == "assisted" else physical_motion
        gate = (midi is not None and left_fresh and right_fresh and bow.engaged and bow_active)
        if gate:
            self.phase = PlayState.BOWING
        elif self.phase == PlayState.BOWING:
            self.phase = PlayState.RELEASING
            self.release_started = now
        elif self.phase == PlayState.RELEASING and now - self.release_started < self.controls.release_time:
            pass
        else:
            self.phase = PlayState.READY if midi is not None and right_fresh else PlayState.IDLE
        # Audio has its own watchdog. GUI heartbeat cannot keep old hands alive.
        freshness = min(self.last_left, self.last_right) if gate else now
        if gate and self.fingering_mode == "fingers":
            freshness = min(freshness, self.fingering.valid_timestamp)
        audio = AudioState(freshness, midi if midi is not None else self.last_midi,
                           bow.intensity if gate else 0.0, gate, volume, bow.direction)
        return MusicalState(audio, self.phase, string, zone, bow.direction,
                            bow.raw_velocity, bow.velocity, bow.pinch, bow.engaged,
                            left_fresh, right_fresh,
                            selected_point,
                            bow.wrist, tuple(self.fingering.states) if self.fingering_mode == "fingers" else (),
                            tuple(self.fingering.scores) if self.fingering_mode == "fingers" else (),
                            self.fingering_mode, bow.sustaining)
