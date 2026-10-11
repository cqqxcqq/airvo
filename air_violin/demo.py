"""Keyboard instrument independent of camera, model, and gesture inference."""

from .mapping import EASY_OFFSETS, OPEN_MIDI
from .types import AudioState, MusicalState, PlayState


class KeyboardInstrument:
    def __init__(self):
        self.string = 1
        self.zone = 0
        self.bowing = False
        self.stroke_until = 0.0

    def key(self, key: int, now: float) -> None:
        if ord("1") <= key <= ord("5"):
            self.zone = key - ord("1")
        elif key == ord("["):
            self.string = max(0, self.string - 1)
        elif key == ord("]"):
            self.string = min(3, self.string + 1)
        elif key == ord(" "):
            self.bowing = not self.bowing
        elif key in (10, 13):
            self.stroke_until = now + 0.7
        elif key == ord("0"):
            self.bowing = False
            self.stroke_until = 0.0

    def update(self, now: float, volume: float) -> MusicalState:
        gate = self.bowing or now < self.stroke_until
        midi = OPEN_MIDI[self.string] + EASY_OFFSETS[self.zone]
        return MusicalState(AudioState(now, midi, 0.65 if gate else 0.0, gate, volume),
                            PlayState.BOWING if gate else PlayState.READY,
                            self.string, self.zone, engaged=gate)
