"""Stage presentation; calibration keeps its original unclipped camera view."""

from collections import deque
from dataclasses import replace
import time
import cv2
import numpy as np
from .calibration import Calibrator, STEPS
from .config import Calibration
from .effects import MusicalEffects
from .instrument_visual import BowAnchor, BowPose, draw_bow, draw_violin
from .mapping import EASY_OFFSETS, OPEN_MIDI, STRINGS, midi_frequency, note_name
from .scenes import SceneEngine
from .segmentation import composite_person
from .types import MusicalState, TrackingFrame

WINDOW = "Air Violin Studio"
COLORS = ((235, 179, 104), (180, 224, 114), (127, 195, 255), (196, 150, 239))  # BGR


def text(image, content, x, y, scale=0.5, color=(218, 225, 235), thickness=1, outline=False):
    if outline:
        cv2.putText(image, str(content), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (15, 16, 17), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, str(content), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wrapped_text(image, content, x, y, width=30, scale=0.43, color=(218, 225, 235)):
    import textwrap
    for line in textwrap.wrap(content, width):
        text(image, line, x, y, scale, color)
        y += 23
    return y


def physical_hand_caption(raw_label: str, calibration: Calibration | None) -> str:
    if calibration is None:
        return "Hand detected"
    if raw_label == calibration.left_label:
        return "LEFT / notes"
    if raw_label == calibration.right_label:
        return "RIGHT / bow"
    return "Unidentified hand"


class Renderer:
    def __init__(self, scene="studio", custom=None, quality="medium", reduced_motion=False,
                 background=True, blur=False, effect_intensity=.65, confidence=.65, timeout=.18):
        self.scene = SceneEngine(scene, custom, blur)
        self.quality, self.reduced_motion = quality, reduced_motion
        self.background, self.effect_intensity = background, effect_intensity
        self.confidence, self.timeout = confidence, timeout
        self.guides, self.debug = True, False
        self.history = deque(maxlen=8)
        self.previous_note, self.previous_gate = None, False
        self.previous_frame = self.previous_tracking = 0.0
        self.fps = self.vision_fps = 0.0
        self.anchor = BowAnchor()
        self.effects = MusicalEffects()
        self.background_status = "Camera video"

    def cycle_quality(self):
        levels = ("low", "medium", "high")
        self.quality = levels[(levels.index(self.quality) + 1) % 3]

    def draw(self, image, state: MusicalState, calibration: Calibration,
             tracking: TrackingFrame | None, wizard: Calibrator | None,
             demo: bool, bow_mode: str, audio, notice: str = "", help_visible: bool = False,
             mask=None, camera_timestamp=None, segmentation=None):
        now = time.perf_counter()
        if self.previous_frame:
            self.fps = .9 * self.fps + .1 / max(.001, now - self.previous_frame)
        self.previous_frame = now
        if tracking and tracking.timestamp > self.previous_tracking:
            if self.previous_tracking:
                self.vision_fps = .9 * self.vision_fps + .1 / max(.001, tracking.timestamp - self.previous_tracking)
            self.previous_tracking = tracking.timestamp
        selected = state.audio.midi if demo or (state.left_fresh and state.finger is not None) else None
        width, height = (640, 480) if wizard else (880, 560)
        canvas = self.scene.render(width, height, now, self.quality, self.reduced_motion, self.effect_intensity)
        ox = oy = 0
        iw, ih = width, height
        self.background_status = "Scene preview" if demo else "Camera video"
        if image is not None:
            h, w = image.shape[:2]
            # Bound compositing to the baseline camera workload, even with HD capture.
            scale = min(1., 640 / w, 480 / h)
            source = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else image
            if self.background and not wizard and not demo:
                sh, sw = source.shape[:2]
                stage = self.scene.render(sw, sh, now, self.quality, self.reduced_motion, self.effect_intensity)
                source, used = composite_person(source, stage, mask, now, tracking, self.confidence)
                if used and camera_timestamp is not None and abs(camera_timestamp - mask.timestamp) > .25:
                    source = cv2.resize(image, (sw, sh))
                    used = False
                self.background_status = self.scene.label if used else "Camera video - waiting for a fresh person mask"
            h, w = source.shape[:2]
            scale = min(width / w, height / h)
            iw, ih = round(w * scale), round(h * scale)
            resized = cv2.resize(source, (iw, ih))
            ox, oy = (width - iw) // 2, (height - ih) // 2
            canvas[oy:oy + ih, ox:ox + iw] = resized
        def pixel(x, y):
            return int(ox + x * iw), int(oy + y * ih)
        board = wizard.value.board if wizard else calibration.board
        if self.guides or wizard:
            overlay = canvas.copy()
            cv2.rectangle(overlay, pixel(board.x0, board.y0), pixel(board.x1, board.y1), (26, 24, 22), -1)
            cv2.addWeighted(overlay, .18, canvas, .82, 0, canvas)
            for s in range(4):
                for z in range(5):
                    x0 = board.x0 + (board.x1 - board.x0) * z / 5
                    y0 = board.y0 + (board.y1 - board.y0) * s / 4
                    x1 = board.x0 + (board.x1 - board.x0) * (z + 1) / 5
                    y1 = board.y0 + (board.y1 - board.y0) * (s + 1) / 4
                    active = wizard is None and (state.string, state.zone) == (s, z) and selected is not None
                    tone = COLORS[s] if active else (127, 129, 126)
                    cv2.rectangle(canvas, pixel(x0, y0), pixel(x1, y1), tone, 2 if active else 1, cv2.LINE_AA)
                    px, py = pixel(x0, y0)
                    text(canvas, note_name(OPEN_MIDI[s] + EASY_OFFSETS[z]), px + 5, py + 21, .37,
                         tone if active else (218, 223, 224), outline=True)
            bow = wizard.value.bow if wizard else calibration.bow
            cv2.rectangle(canvas, pixel(bow.x0, bow.y0), pixel(bow.x1, bow.y1), (149, 152, 148), 1, cv2.LINE_AA)
            bx, by = pixel(bow.x0, bow.y0)
            text(canvas, "BOW / " + bow_mode.upper(), bx + 5, by - 8, .38, outline=True)
        color = COLORS[state.string]
        if not wizard:
            draw_violin(canvas, board, pixel, state, COLORS, calibration.bow)
            local_pose = self.anchor.update(tracking, calibration, now, iw, ih, self.confidence, self.timeout) if not demo else None
            pose = replace(local_pose, grip=(local_pose.grip[0] + ox, local_pose.grip[1] + oy)) if local_pose else None
            if demo:
                # Explicit demonstration pose, not a claim of webcam tracking.
                pose = BowPose((width * .78, height * .52), 3.10, width * .42, now)
            self.effects.draw(canvas, state, pose, now, color, note_name(selected),
                              self.quality, self.reduced_motion, self.effect_intensity)
            opacity = max(0., min(1., (self.timeout - (now - pose.timestamp)) / .06)) if pose and not demo else 1.
            draw_bow(canvas, pose, color, state.audio.intensity * self.effect_intensity if state.audio.gate else 0., opacity)
            if state.finger and state.left_fresh:
                cv2.circle(canvas, pixel(state.finger.x, state.finger.y), 6, (245, 255, 250), 2, cv2.LINE_AA)
        if tracking and (wizard or self.debug) and 0 <= now - tracking.timestamp <= self.timeout:
            for hand in tracking.hands:
                for p in hand.points:
                    cv2.circle(canvas, pixel(p.x, p.y), 2, (138, 238, 230), -1)
                x, y = pixel(hand.points[0].x, hand.points[0].y)
                binding = wizard.value if wizard and wizard.step >= 1 else None if wizard else calibration
                text(canvas, physical_hand_caption(hand.label, binding), x, y + 20, .40, outline=True)
            if wizard and wizard.corner and wizard.step in (3, 5):
                cv2.drawMarker(canvas, pixel(wizard.corner.x, wizard.corner.y), (145, 238, 221), cv2.MARKER_CROSS, 20, 2)
                label = wizard.value.left_label if wizard.step == 3 else wizard.value.right_label
                hand = tracking.hand(label, wizard.confidence)
                if hand:
                    target = hand.points[8 if wizard.step == 3 else 0]
                    cv2.rectangle(canvas, pixel(min(wizard.corner.x, target.x), min(wizard.corner.y, target.y)),
                                  pixel(max(wizard.corner.x, target.x), max(wizard.corner.y, target.y)), (145, 238, 221), 2)
        if state.audio.gate and (state.audio.midi != self.previous_note or not self.previous_gate):
            self.history.append(note_name(state.audio.midi))
        self.previous_note, self.previous_gate = state.audio.midi, state.audio.gate
        output = np.full((710, 920, 3), (22, 20, 18), dtype=np.uint8)
        text(output, "AIR VIOLIN", 22, 36, .80, (239, 245, 250), 2)
        text(output, "STUDIO", 250, 35, .43, (164, 184, 193))
        mode = "KEYBOARD DEMO" if demo else "CAMERA / EASY"
        text(output, mode + "  |  " + self.quality.upper(), 615, 34, .40, (185, 220, 220))
        origin_y = 90 if wizard else 64
        output[origin_y:origin_y + height, 20:20 + width] = canvas
        if wizard:
            title, instruction, action = wizard.instructions
            text(output, f"CALIBRATION {wizard.step + 1}/{len(STEPS)}", 680, 119, .43, (172, 224, 212))
            y = wrapped_text(output, title, 680, 164, 22, .53)
            if wizard.step == 0:
                y = wrapped_text(output, "Hand identity is assigned when you press SPACE.", 680, y + 12, 31, .39)
            y = wrapped_text(output, instruction, 680, y + 22)
            y = wrapped_text(output, action, 680, y + 22)
            message = f"Sweeping: {max(0, 3 - (now - wizard.started)):.1f} seconds left" if wizard.started is not None else wizard.message
            wrapped_text(output, message, 680, y + 22, 31, .39, (132, 183, 255))
        else:
            # A narrow performance strip; face, hands and instrument lead the view.
            cv2.rectangle(output, (20, 564), (900, 624), (25, 25, 24), -1)
            text(output, note_name(selected), 35, 605, .90, color, 2)
            text(output, f"{STRINGS[state.string]} string" if selected is not None else "Left index: choose a note", 124, 589, .43)
            status = "Bowing" if state.audio.gate else "Move your right hand to bow" if selected is not None else "Ready to play"
            text(output, status, 124, 612, .41, (185, 220, 220))
            text(output, "BOW", 435, 590, .36)
            cv2.rectangle(output, (480, 582), (590, 590), (60, 57, 52), -1)
            cv2.rectangle(output, (480, 582), (480 + int(110 * state.audio.intensity), 590), color, -1)
            text(output, f"{bow_mode.upper()}  |  Volume {state.audio.volume:.0%}", 435, 611, .35)
            text(output, "Recent: " + "  ".join(list(self.history)[-5:]), 625, 590, .36)
            text(output, self.scene.label + (" / background off" if not self.background else ""), 625, 612, .36)
        age = (now - tracking.timestamp) * 1000 if tracking else None
        if self.debug:
            inference = f"{tracking.inference_ms:.1f}ms" if tracking else "--"
            mask_ms = f"{mask.inference_ms:.1f}ms" if mask else "--"
            age_text = f"{age:.0f}ms" if age is not None else "--"
            text(output, f"UI {self.fps:.0f}fps  hands {self.vision_fps:.0f}fps  inference {inference}  age {age_text}  segment {mask_ms}  bow {state.velocity:+.2f}", 22, 641, .36)
            device_latency = f"{audio.latency_ms:.0f}ms" if audio.stream is not None else "muted"
            detail = f"Audio device {device_latency}  callback {audio.health.callback_ms:.2f}ms  underruns {audio.health.underruns}  {state.phase.value}"
            if selected is not None:
                detail += f"  MIDI {selected} / {midi_frequency(selected):.1f}Hz"
            text(output, detail, 22, 658, .36)
        else:
            detail = self.background_status
            if segmentation and segmentation.throttled:
                detail += " / lighter segmentation to protect tracking"
            if audio.stream is None:
                detail += " / audio muted"
            text(output, detail, 22, 644, .38, (170, 191, 198))
        if notice:
            text(output, notice[:115], 22, 675 if self.debug else 661, .35, (132, 183, 255))
        text(output, "H help  G guides  S scene  V background  D debug  F fullscreen  C calibrate  Q quit", 22, 693, .39)
        if help_visible:
            cv2.rectangle(output, (80, 165), (820, 494), (32, 28, 23), -1)
            lines = ["PLAY YOUR AIR VIOLIN", "Left INDEX: select string and note. Right WRIST: sweep horizontally.",
                     "B changes Auto Bow / Precision Bow (pinch to engage).",
                     "Pitch changes stay connected while you keep bowing.",
                     "Stop, leave the bow region, or lose tracking to release safely.",
                     "G shows/hides note guides. S changes scene. V toggles background.",
                     "P cycles LOW / MEDIUM / HIGH. M toggles reduced motion.",
                     "F fullscreen. D diagnostics. C recalibrate. +/- volume. R retry audio.",
                     "K toggles keyboard: [ ] string, 1..5 note, SPACE sustain, ENTER stroke.",
                     "Keyboard 0 stops sound. Focus this window for shortcuts.",
                     "H closes help. Q or ESC quits."]
            for i, line in enumerate(lines):
                text(output, line, 100, 200 + i * 26, .45 if i else .63)
        return output
