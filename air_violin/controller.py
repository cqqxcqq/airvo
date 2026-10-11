"""Musical state machine, independent of display rate and audio implementation."""

import math
from .config import Calibration, Controls
from .gestures import BowInterpreter
from .mapping import Fingerboard
from .types import AudioState, MusicalState, PlayState, TrackingFrame


class Instrument:
    def __init__(self, calibration: Calibration, controls: Controls, bow_mode: str = "auto"):
        self.controls = controls
        self.calibration = calibration
        self.board = Fingerboard(calibration.board, controls)
        self.bow = BowInterpreter(calibration, controls, bow_mode)
        self.last_frame = self.last_left = self.last_right = -math.inf
        self.phase = PlayState.IDLE
        self.release_started = -math.inf
        self.last_midi: int | None = None

    def update(self, frame: TrackingFrame | None, now: float, volume: float = 0.65) -> MusicalState:
        if frame is not None and self.last_frame < frame.timestamp <= now:
            self.last_frame = frame.timestamp
            if now - frame.timestamp <= self.controls.tracking_timeout:
                left = frame.hand(self.calibration.left_label, self.controls.confidence)
                right = frame.hand(self.calibration.right_label, self.controls.confidence)
                if left:
                    if frame.timestamp - self.last_left > self.controls.tracking_timeout:
                        self.board.reset()
                    self.last_left = frame.timestamp
                    self.board.update(left.points[8], frame.timestamp)
                if right:
                    self.last_right = frame.timestamp
                    self.bow.update(right, frame.timestamp)
        left_fresh = now - self.last_left <= self.controls.tracking_timeout
        right_fresh = now - self.last_right <= self.controls.tracking_timeout
        if not left_fresh:
            self.board.reset()
        if not right_fresh:
            self.bow.reset()
        midi = self.board.midi
        if midi is not None:
            self.last_midi = midi
        bow = self.bow.state
        # Also expire the motion grace when inference produces no new results.
        gate = (midi is not None and left_fresh and right_fresh and bow.engaged
                and now - self.bow.last_motion <= self.controls.reversal_grace)
        if gate:
            self.phase = PlayState.BOWING
        elif self.phase == PlayState.BOWING:
            self.phase = PlayState.RELEASING
            self.release_started = now
        elif self.phase == PlayState.RELEASING and now - self.release_started < self.controls.release_time:
            pass
        else:
            self.phase = PlayState.READY if midi is not None and right_fresh else PlayState.IDLE
        string, zone = self.board.selected or (0, 0)
        # Audio has its own watchdog. GUI heartbeat cannot keep old hands alive.
        freshness = min(self.last_left, self.last_right) if gate else now
        audio = AudioState(freshness, midi if midi is not None else self.last_midi,
                           bow.intensity if gate else 0.0, gate, volume)
        return MusicalState(audio, self.phase, string, zone, bow.direction,
                            bow.raw_velocity, bow.velocity, bow.pinch, bow.engaged,
                            left_fresh, right_fresh, self.board.point, bow.wrist)
