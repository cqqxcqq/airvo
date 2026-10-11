"""Headless synthetic melody -> gestures -> musical state -> PCM and desktop image.

This is a diagnostic rendering, not a recording feature or a hardware test.
Use --require-model to additionally require real MediaPipe model initialization.
"""

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time
import wave
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from air_violin.audio import AudioEngine, StateBus, Synthesizer
from air_violin.config import Calibration, Controls
from air_violin.controller import Instrument
from air_violin.mapping import EASY_OFFSETS, OPEN_MIDI
from air_violin.models import DEFAULT_MODEL, verify_model
from air_violin.render import Renderer
from air_violin.types import Hand, Point, TrackingFrame
from air_violin.vision import HandTracker


def synthetic_frame(timestamp, string, zone):
    cal = Calibration()
    r = cal.board
    finger = Point(r.x0 + (zone + 0.5) / 5 * (r.x1 - r.x0), r.y0 + (string + 0.5) / 4 * (r.y1 - r.y0))
    left = [Point(0.3, 0.4) for _ in range(21)]
    left[8] = finger
    right = [Point(0.7, 0.5) for _ in range(21)]
    right[0] = Point(0.735 + 0.15 * math.sin(timestamp * 7), 0.5)
    right[5], right[17] = Point(0.65, 0.5), Point(0.75, 0.5)
    right[4], right[8] = Point(0.75, 0.4), Point(0.65, 0.4)
    return TrackingFrame(timestamp, (Hand("Left", 0.98, tuple(left)), Hand("Right", 0.98, tuple(right))))


def validate(output: Path, require_model: bool = False, model: Path = DEFAULT_MODEL):
    sample_rate, block_size = 48000, 256
    synth = Synthesizer(sample_rate)
    cal = Calibration()
    instrument = Instrument(cal, Controls())
    # Ode to Joy's opening phrase, transposed to D major, one playable string.
    zones = (2, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 2, 1, 1)
    duration = 0.45
    blocks, durations, seen = [], [], set()
    state = None
    tracked = None
    last_vision = -1
    for offset in range(0, int((len(zones) * duration + 0.5) * sample_rate), block_size):
        elapsed = offset / sample_rate
        now = 100 + elapsed
        note_index = int(elapsed / duration)
        vision_index = int(elapsed * 30)
        if note_index < len(zones) and vision_index != last_vision:
            last_vision = vision_index
            tracked = synthetic_frame(now, 1, zones[note_index])
        if note_index >= len(zones):
            tracked = TrackingFrame(now)
        state = instrument.update(tracked, now)
        if note_index < len(zones) and elapsed % duration > 0.20:
            expected = OPEN_MIDI[1] + EASY_OFFSETS[zones[note_index]]
            assert state.audio.midi == expected, (note_index, state.audio.midi, expected)
            seen.add(expected)
        started = time.perf_counter()
        blocks.append(synth.render(block_size, state.audio, now))
        durations.append((time.perf_counter() - started) * 1000)
    audio = np.concatenate(blocks)
    assert np.isfinite(audio).all() and 0.05 < abs(audio).max() < 1
    assert len(seen) == 5
    assert abs(audio[-1024:]).max() < 0.01
    output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output / "synthetic-ode.wav"), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767).astype("<i2").tobytes())
    tracked = synthetic_frame(101, 1, 2)
    preview_instrument = Instrument(cal, Controls())
    preview_instrument.update(synthetic_frame(100.96, 1, 2), 100.96)
    state = preview_instrument.update(tracked, 101)
    preview = Renderer().draw(None, state, cal, None, None, False, "auto", AudioEngine(StateBus()),
                              "Synthetic control validation. Webcam and audio hardware have not been tested.")
    if not cv2.imwrite(str(output / "foundation-preview.png"), preview):
        raise OSError("Could not write diagnostic UI image")
    model_result = "UNRUN: official model not installed"
    if model.exists():
        verify_model(model)
        tracker = HandTracker(model)
        try:
            for t in (1.0, 1.04):
                result = tracker.detect(np.zeros((480, 640, 3), np.uint8), t)
                assert result.hands == ()
            model_result = "PASS: real Tasks VIDEO inference on two blank frames (not hand recognition accuracy)"
        finally:
            tracker.close()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "synthetic_melody": "PASS: all five pitches selected through the gesture controller",
        "audio": {"samples": len(audio), "peak": float(abs(audio).max()),
                  "render_block_ms_median": float(np.median(durations)),
                  "render_block_ms_p95": float(np.percentile(durations, 95)),
                  "block_budget_ms": block_size / sample_rate * 1000},
        "mediapipe": model_result,
        "hardware": "UNRUN: webcam, desktop interaction, output device, physical hand calibration, end-to-end latency",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if require_model and not model.exists():
        raise FileNotFoundError("Required MediaPipe model test was not run. Run tools/fetch_model.py, then retry.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()
    try:
        validate(args.output, args.require_model, args.model)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
