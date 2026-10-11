"""Persistent PortAudio output and a lightweight bowed-string-inspired voice.

This is additive synthesis, not an acoustic violin simulation. No vision,
filesystem I/O, device discovery, or blocking locks belong in the callback.
"""

from dataclasses import dataclass
import math
import threading
import time
import numpy as np
from .mapping import midi_frequency
from .types import AudioState


class StateBus:
    """A tiny immutable-message mailbox with a nonblocking audio-side read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = AudioState(time.perf_counter())

    def publish(self, state: AudioState) -> None:
        with self._lock:
            self._state = state

    def read_nowait(self, previous: AudioState) -> AudioState:
        if not self._lock.acquire(blocking=False):
            return previous
        try:
            return self._state
        finally:
            self._lock.release()


def exponential_curve(start: float, target: float, count: int, tau: float, sample_rate: int) -> np.ndarray:
    """Exact one-pole response, continuous across arbitrary callback sizes."""
    return target + (start - target) * np.exp(-np.arange(1, count + 1) / (tau * sample_rate))


class Synthesizer:
    def __init__(self, sample_rate: int = 48000, timeout: float = 0.180, seed: int = 7):
        if sample_rate < 16000:
            raise ValueError("Use a sample rate of at least 16000 Hz")
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.phase = 0.0
        self.frequency = 440.0
        self.amplitude = 0.0
        self.rng = np.random.default_rng(seed)
        self.harmonics = np.arange(1, 25, dtype=np.float64)
        self.weights = 1 / self.harmonics ** 1.10
        self.brightness = 0.0
        self.stroke_transient = 0.0
        self.was_bowing = False
        self.last_direction = 0
        self.previous_noise = 0.0

    def render(self, count: int, state: AudioState, now: float) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        valid_midi = state.midi is not None and 0 <= state.midi <= 127
        target_frequency = midi_frequency(state.midi) if valid_midi else self.frequency
        frequencies = exponential_curve(self.frequency, target_frequency, count, 0.022, self.sample_rate)
        self.frequency = float(frequencies[-1])
        phases = self.phase + (2 * math.pi / self.sample_rate) * np.cumsum(frequencies)
        self.phase = float(phases[-1] % (2 * math.pi))
        fresh = 0 <= now - state.timestamp <= self.timeout
        intensity = float(np.clip(state.intensity, 0, 1)) if math.isfinite(state.intensity) else 0.0
        volume = float(np.clip(state.volume, 0, 1)) if math.isfinite(state.volume) else 0.0
        bowing = state.gate and fresh and valid_midi
        direction = state.bow_direction if state.bow_direction in (-1, 1) else 0
        if bowing and not self.was_bowing:
            self.stroke_transient = .5
        elif bowing and direction and self.last_direction and direction != self.last_direction:
            self.stroke_transient = max(self.stroke_transient, .20)
        if direction and bowing:
            self.last_direction = direction
        if not bowing:
            self.last_direction = 0
        self.was_bowing = bowing
        target_amplitude = 0.65 * volume * intensity if bowing else 0.0
        tau = 0.015 if target_amplitude > self.amplitude else 0.065
        envelope = exponential_curve(self.amplitude, target_amplitude, count, tau, self.sample_rate)
        self.amplitude = float(envelope[-1])
        brightness = exponential_curve(self.brightness, intensity if bowing else 0.0, count, .040, self.sample_rate)
        self.brightness = float(brightness[-1])
        transient = exponential_curve(self.stroke_transient, 0., count, .030, self.sample_rate)
        self.stroke_transient = float(transient[-1])
        output = np.zeros(count, dtype=np.float64)
        normalization = np.zeros(count, dtype=np.float64)
        # Helmholtz motion motivates a rich, saw-like harmonic source. Broad
        # body color and speed-dependent rolloff add character, but these are
        # designed formants, not measured violin resonances or a friction solver.
        # Per-sample curves keep the result independent of callback block size.
        for harmonic, weight in zip(self.harmonics, self.weights):
            partial_frequency = harmonic * frequencies
            body = (1 + .25 * np.exp(-((partial_frequency - 460) / 240) ** 2)
                    + .55 * np.exp(-((partial_frequency - 2400) / 950) ** 2))
            gain = weight * body * np.exp(-(harmonic - 1) / (3 + 14 * brightness))
            gain *= partial_frequency < .45 * self.sample_rate
            output += gain * np.sin(harmonic * phases)
            normalization += gain
        output /= np.maximum(normalization, 1e-12)
        white = self.rng.uniform(-1, 1, count)
        previous = np.concatenate(([self.previous_noise], white[:-1]))
        self.previous_noise = float(white[-1])
        friction = (white - previous) * (.003 * brightness + .015 * transient)
        return ((output * (1 + .08 * transient) + friction) * envelope).astype(np.float32)


@dataclass(frozen=True)
class AudioHealth:
    callbacks: int = 0
    underruns: int = 0
    callback_ms: float = 0.0
    peak: float = 0.0
    error: str = ""


class AudioEngine:
    def __init__(self, bus: StateBus, device: int | str | None = None,
                 sample_rate: int = 48000, block_size: int = 256, timeout: float = 0.180):
        self.bus = bus
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.synth = Synthesizer(sample_rate, timeout)
        self.health = AudioHealth()
        self.stream = None
        self.last_state = AudioState(time.perf_counter())

    def _callback(self, outdata, frames, time_info, status) -> None:
        started = time.perf_counter()
        self.last_state = self.bus.read_nowait(self.last_state)
        previous = self.health
        try:
            samples = self.synth.render(frames, self.last_state, started)
            outdata[:, 0] = samples
            peak = float(np.max(np.abs(samples)))
            error = ""
        except Exception as exc:
            outdata.fill(0)
            peak = 0.0
            error = f"Synth callback: {type(exc).__name__}: {exc}"
        self.health = AudioHealth(previous.callbacks + 1,
                                  previous.underruns + int(bool(status.output_underflow)),
                                  (time.perf_counter() - started) * 1000, peak, error)

    def start(self) -> None:
        import sounddevice as sd
        sd.check_output_settings(device=self.device, channels=1,
                                 dtype="float32", samplerate=self.sample_rate)
        stream = sd.OutputStream(device=self.device, channels=1, dtype="float32",
                                 samplerate=self.sample_rate, blocksize=self.block_size,
                                 latency="low", callback=self._callback)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self.stream = stream

    @property
    def latency_ms(self) -> float | None:
        return float(self.stream.latency) * 1000 if self.stream is not None else None

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
