"""Control heartbeat independent of the UI, with explicit ownership of audio state."""

import threading
import time
from .capture import LatestValue
from .controller import Instrument
from .types import AudioState


class ControlWorker:
    def __init__(self, vision, bus, calibration, controls, mode, volume):
        self.vision, self.bus, self.controls = vision, bus, controls
        self.lock = threading.Lock()
        self.instrument = Instrument(calibration, controls, mode)
        self.calibration, self.mode, self.volume = calibration, mode, volume
        self.enabled = False
        self.latest = LatestValue()
        self.error = ""
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="musical-control", daemon=True)
        self.thread.start()

    def configure(self, calibration, mode, volume, enabled):
        # The same short lock covers publication: disabling returns only after
        # the worker can no longer overwrite keyboard/calibration silence.
        with self.lock:
            if calibration != self.calibration:
                self.instrument = Instrument(calibration, self.controls, mode)
                self.calibration = calibration
                self.latest.put(None)
            elif mode != self.mode or (enabled and not self.enabled):
                self.instrument = Instrument(calibration, self.controls, mode)
                self.latest.put(None)
            if self.enabled and not enabled:
                self.bus.publish(AudioState(time.perf_counter(), volume=volume))
            self.mode, self.volume, self.enabled = mode, volume, enabled

    def _run(self):
        try:
            while not self.stop.is_set():
                with self.lock:
                    if self.enabled:
                        state = self.instrument.update(self.vision.latest.get(), time.perf_counter(), self.volume)
                        self.latest.put(state)
                        self.bus.publish(state.audio)
                self.stop.wait(0.008)
        except Exception as exc:
            self.error = f"Musical controls stopped: {exc}; keyboard mode remains available. Restart for camera controls."
            self.bus.publish(AudioState(time.perf_counter()))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.bus.publish(AudioState(time.perf_counter()))
