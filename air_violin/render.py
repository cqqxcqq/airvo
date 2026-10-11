"""A legible, restrained foundation UI. Elaborate AR scenes belong to milestone 2."""

from collections import deque
import time
import cv2
import numpy as np
from .calibration import Calibrator, STEPS
from .config import Calibration
from .mapping import EASY_OFFSETS, OPEN_MIDI, STRINGS, midi_frequency, note_name
from .types import MusicalState, TrackingFrame

WINDOW = "Air Violin Studio"
COLORS = ((235, 179, 104), (180, 224, 114), (127, 195, 255), (196, 150, 239))  # BGR


def text(image, content, x, y, scale=0.5, color=(218, 225, 235), thickness=1):
    cv2.putText(image, str(content), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wrapped_text(image, content, x, y, width=30, scale=0.43, color=(218, 225, 235)):
    import textwrap
    for line in textwrap.wrap(content, width):
        text(image, line, x, y, scale, color)
        y += 23
    return y


class Renderer:
    def __init__(self):
        self.history = deque(maxlen=8)
        self.previous_note = None
        self.previous_gate = False
        self.previous_frame = 0.0
        self.fps = 0.0
        self.previous_tracking = 0.0
        self.vision_fps = 0.0

    def draw(self, image, state: MusicalState, calibration: Calibration,
             tracking: TrackingFrame | None, wizard: Calibrator | None,
             demo: bool, bow_mode: str, audio, notice: str = "", help_visible: bool = False):
        now = time.perf_counter()
        if self.previous_frame:
            self.fps = 0.9 * self.fps + 0.1 / max(0.001, now - self.previous_frame)
        self.previous_frame = now
        if tracking and tracking.timestamp > self.previous_tracking:
            if self.previous_tracking:
                self.vision_fps = 0.9 * self.vision_fps + 0.1 / max(0.001, tracking.timestamp - self.previous_tracking)
            self.previous_tracking = tracking.timestamp
        selected = state.audio.midi if demo or (state.left_fresh and state.finger is not None) else None
        # Fixed 640x480 interaction canvas, plus dedicated panels outside camera.
        # Resizing is uniform via letterbox so normalized detection stays aligned.
        if image is None:
            canvas = np.full((480, 640, 3), (23, 20, 17), dtype=np.uint8)
        else:
            h, w = image.shape[:2]
            scale = min(640 / w, 480 / h)
            resized = cv2.resize(image, (round(w * scale), round(h * scale)))
            canvas = np.full((480, 640, 3), (23, 20, 17), dtype=np.uint8)
            ox, oy = (640 - resized.shape[1]) // 2, (480 - resized.shape[0]) // 2
            canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        h, w = canvas.shape[:2]
        # Map normalized coordinates through the SAME letterbox as camera pixels.
        if image is None:
            ox = oy = 0
            iw, ih = w, h
        else:
            iw, ih = resized.shape[1], resized.shape[0]
        def pixel(x, y):
            return int(ox + x * iw), int(oy + y * ih)
        board = wizard.value.board if wizard else calibration.board
        overlay = canvas.copy()
        cv2.rectangle(overlay, pixel(board.x0, board.y0), pixel(board.x1, board.y1), (32, 27, 24), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        for s in range(4):
            for z in range(5):
                x0 = board.x0 + (board.x1 - board.x0) * z / 5
                y0 = board.y0 + (board.y1 - board.y0) * s / 4
                x1 = board.x0 + (board.x1 - board.x0) * (z + 1) / 5
                y1 = board.y0 + (board.y1 - board.y0) * (s + 1) / 4
                active = wizard is None and (state.string, state.zone) == (s, z) and selected is not None
                color = COLORS[s] if active else (89, 79, 68)
                cv2.rectangle(canvas, pixel(x0, y0), pixel(x1, y1), color, 2 if active else 1, cv2.LINE_AA)
                px, py = pixel(x0, y0)
                text(canvas, note_name(OPEN_MIDI[s] + EASY_OFFSETS[z]), px + 5, py + 23,
                     0.43, COLORS[s] if active else (205, 203, 193))
        bow = wizard.value.bow if wizard else calibration.bow
        cv2.rectangle(canvas, pixel(bow.x0, bow.y0), pixel(bow.x1, bow.y1), (149, 152, 148), 1, cv2.LINE_AA)
        bx, by = pixel(bow.x0, bow.y0)
        text(canvas, "BOW / " + bow_mode.upper(), bx + 5, by - 8, 0.42)
        for point, color in ((state.finger, (245, 255, 250)), (state.wrist, (142, 220, 254))):
            if point:
                cv2.circle(canvas, pixel(point.x, point.y), 7, color, 2, cv2.LINE_AA)
        if tracking and wizard:
            for hand in tracking.hands:
                for p in hand.points:
                    cv2.circle(canvas, pixel(p.x, p.y), 2, (138, 238, 230), -1)
                x, y = pixel(hand.points[0].x, hand.points[0].y)
                text(canvas, f"raw {hand.label} {hand.confidence:.2f}", x, y + 20, 0.45)
            if wizard.corner and wizard.step in (3, 5):
                cv2.drawMarker(canvas, pixel(wizard.corner.x, wizard.corner.y), (145, 238, 221),
                               cv2.MARKER_CROSS, 20, 2)
        if state.audio.gate and (state.audio.midi != self.previous_note or not self.previous_gate):
            self.history.append(note_name(state.audio.midi))
        self.previous_note, self.previous_gate = state.audio.midi, state.audio.gate
        output = np.full((710, 920, 3), (22, 20, 18), dtype=np.uint8)
        output[90:570, 20:660] = canvas
        text(output, "AIR VIOLIN", 22, 36, 0.88, (239, 245, 250), 2)
        text(output, "STUDIO  /  PLAYABLE FOUNDATION", 23, 62, 0.42, (148, 154, 164))
        text(output, "KEYBOARD DEMO" if demo else "CAMERA / EASY MODE", 678, 37, 0.45, (172, 224, 212))
        text(output, "SELECTED NOTE", 680, 116, 0.43, (147, 156, 165))
        text(output, note_name(selected), 678, 176, 1.4, COLORS[state.string], 2)
        text(output, f"{STRINGS[state.string]} string  |  zone {state.zone + 1}" if selected is not None else "Waiting for a note", 680, 211, 0.52)
        if selected is not None:
            text(output, f"MIDI {selected}   {midi_frequency(selected):.1f} Hz", 680, 238, 0.44)
        text(output, state.phase.value, 680, 284, 0.65, (172, 224, 212), 2)
        text(output, "BOW INTENSITY", 680, 319, 0.42, (147, 156, 165))
        cv2.rectangle(output, (680, 332), (892, 342), (54, 49, 43), -1)
        cv2.rectangle(output, (680, 332), (680 + int(212 * state.audio.intensity), 342), COLORS[state.string], -1)
        text(output, f"Volume {state.audio.volume:.0%}", 680, 369, 0.47)
        text(output, "RECENT NOTES", 680, 411, 0.42, (147, 156, 165))
        for row, notes in enumerate((list(self.history)[-8:-4], list(self.history)[-4:])):
            text(output, "  ".join(notes), 680, 438 + row * 25, 0.40)
        if not demo:
            text(output, f"Left {'tracked' if state.left_fresh else '--'} / Right {'tracked' if state.right_fresh else '--'}", 680, 500, 0.43)
            text(output, f"Bow {state.velocity:+.2f} widths/s", 680, 527, 0.43)
        age = (now - tracking.timestamp) * 1000 if tracking else None
        inference = f"{tracking.inference_ms:.1f} ms" if tracking else "--"
        age_text = f"{age:.0f} ms" if age is not None else "--"
        text(output, f"UI {self.fps:.0f} FPS  |  vision {self.vision_fps:.0f} FPS  |  inference {inference}  |  age {age_text}", 22, 600, 0.42)
        audio_status = "MUTED / NO AUDIO STREAM"
        if audio.stream is not None:
            audio_status = f"Audio {audio.sample_rate} Hz/{audio.block_size} | device {audio.latency_ms:.1f} ms | callback {audio.health.callback_ms:.2f} ms | underruns {audio.health.underruns}"
        text(output, audio_status, 22, 623, 0.42, (153, 203, 211))
        if wizard:
            # Keep the entire camera visible while users mark reachable corners.
            output[90:570, 667:920] = (22, 20, 18)
            title, instruction, action = wizard.instructions
            text(output, f"CALIBRATION {wizard.step + 1}/{len(STEPS)}", 680, 119, 0.43, (172, 224, 212))
            y = wrapped_text(output, title, 680, 164, 22, 0.53)
            y = wrapped_text(output, instruction, 680, y + 22)
            y = wrapped_text(output, action, 680, y + 22)
            if wizard.started is not None:
                wrapped_text(output, f"Sweeping: {max(0, 3 - (now - wizard.started)):.1f} seconds left", 680, y + 22)
            else:
                wrapped_text(output, wizard.message, 680, y + 22, 31, 0.39, (132, 183, 255))
        if notice:
            # Details also go to stderr; truncate the banner to preserve layout.
            text(output, notice[:108], 22, 648, 0.39, (132, 183, 255))
        text(output, "H help   K keyboard   B bow mode   C calibrate   +/- volume   Q quit", 22, 685, 0.47)
        if demo:
            text(output, "1..5 notes   [ / ] string   SPACE sustain toggle   ENTER short bow   0 stop", 24, 553, 0.40)
        if help_visible:
            cv2.rectangle(output, (82, 170), (640, 495), (32, 28, 23), -1)
            lines = ["HOW TO PLAY", "Left INDEX selects a note; preview it before bowing.",
                     "Top to bottom: G / D / A / E. Left to right: +0,2,4,5,7.",
                     "Right WRIST sweeps horizontally inside the bow box.",
                     "B switches Auto Bow / Precision Bow (pinch to engage).",
                     "Stop moving or leave the box to release the sound.",
                     "C repeats physical-hand calibration. K toggles demo.",
                     "Keyboard: [ ] choose string, 1..5 choose note.",
                     "SPACE toggles sustain; ENTER plays a 0.7-second stroke.",
                     "0 stops demo sound. R retries audio after a failure.",
                     "Focus this window for shortcuts. H closes this guide."]
            for i, line in enumerate(lines):
                text(output, line, 102, 204 + i * 26, 0.43 if i else 0.65)
        return output
