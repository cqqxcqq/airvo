"""Desktop lifecycle and keyboard controls; the audio thread never calls this UI."""

import argparse
from contextlib import suppress
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
import cv2
from .audio import AudioEngine, StateBus
from .calibration import Calibrator
from .capture import Camera
from .config import Calibration, Controls, default_config_path, load_calibration, save_calibration
from .controller import Instrument
from .demo import KeyboardInstrument
from .models import DEFAULT_MODEL, verify_model
from .render import Renderer, WINDOW
from .types import AudioState
from .vision import VisionWorker


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Air Violin Studio — milestone 1 desktop instrument")
    p.add_argument("--demo", action="store_true", help="Keyboard instrument; no camera or model required")
    p.add_argument("--silent", action="store_true", help="Explicitly disable hardware audio")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-mirror", action="store_true", help="Changing mirror mode requires recalibration")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--config", type=Path, default=default_config_path())
    p.add_argument("--device", help="Output-device index or substring of its name")
    p.add_argument("--list-audio-devices", action="store_true")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--volume", type=float, default=0.65)
    p.add_argument("--bow-mode", choices=("auto", "precision"), default="auto")
    p.add_argument("--bow-speed", type=float, help="Override full-intensity screen widths/second (0.15..4)")
    p.add_argument("--confidence", type=float, default=0.65)
    p.add_argument("--smoothing-ms", type=float, default=35)
    p.add_argument("--note-hold-ms", type=float, default=50)
    p.add_argument("--bow-deadzone", type=float, default=0.045)
    p.add_argument("--tracking-timeout-ms", type=float, default=180)
    return p


def run(args) -> int:
    if args.list_audio_devices:
        try:
            import sounddevice as sd
            print(sd.query_devices())
            print("Default input/output indices:", sd.default.device)
            return 0
        except Exception as exc:
            print(f"Audio device discovery failed: {exc}. See README troubleshooting.", file=sys.stderr)
            return 2
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No desktop display available. Run python tools/validate.py for headless validation,\n"
              "or launch python -m air_violin --demo on your laptop.", file=sys.stderr)
        return 2
    if not 160 <= args.width <= 1920 or not 120 <= args.height <= 1080:
        raise ValueError("Camera dimensions must be 160..1920 by 120..1080")
    if not 0 <= args.volume <= 1 or not 16000 <= args.sample_rate <= 96000 or not 64 <= args.block_size <= 2048:
        raise ValueError("Volume must be 0..1, sample rate 16000..96000, block size 64..2048")
    controls = Controls(confidence=args.confidence, position_tau=args.smoothing_ms / 1000,
                        note_hold=args.note_hold_ms / 1000, deadzone=args.bow_deadzone,
                        tracking_timeout=args.tracking_timeout_ms / 1000)
    mirrored = not args.no_mirror
    notice = ""
    def report(message):
        nonlocal notice
        notice = message
        print(message, file=sys.stderr)
    calibration = None
    try:
        calibration = load_calibration(args.config, mirrored)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report(f"Calibration not usable: {exc}. Existing file kept; please recalibrate.")
    calibrated = calibration is not None
    if args.bow_speed is not None:
        calibration = replace(calibration or Calibration(mirrored=mirrored), bow_full_speed=args.bow_speed)
    calibration = calibration or Calibration(mirrored=mirrored)
    demo_mode = args.demo
    keyboard = KeyboardInstrument()
    bus = StateBus()
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    audio = AudioEngine(bus, device, args.sample_rate, args.block_size, controls.tracking_timeout)
    camera = vision = wizard = None
    instrument = Instrument(calibration, controls, args.bow_mode)
    renderer = Renderer()
    help_visible = demo_mode
    volume = args.volume
    bow_mode = args.bow_mode
    try:
        if not args.silent:
            try:
                audio.start()
            except Exception as exc:
                report(f"Audio unavailable: {exc}. Visual controls remain usable; R retries audio.")
        else:
            report("Silent mode: no hardware audio output.")
        if not demo_mode:
            try:
                verify_model(args.model)
                camera = Camera(args.camera, args.width, args.height, mirrored, args.backend)
                vision = VisionWorker(camera, args.model, args.confidence)
                if not calibrated:
                    wizard = Calibrator(mirrored, args.confidence)
            except (OSError, ValueError, RuntimeError, IndexError) as exc:
                report(f"Camera mode unavailable: {exc}. Using keyboard demo.")
                demo_mode = True
                help_visible = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 920, 710)
        vision_failure_reported = False
        camera_failure_reported = False
        started = time.perf_counter()
        while True:
            now = time.perf_counter()
            camera_frame = camera.latest.get() if camera else None
            tracking = vision.latest.get() if vision else None
            if vision and vision.error and not vision_failure_reported:
                report(vision.error + "; using keyboard demo.")
                vision_failure_reported = True
                demo_mode, wizard = True, None
            if camera and not camera_failure_reported and (camera.error or (now - started > 5 and camera_frame is None)):
                report(camera.error or "Camera has not delivered a frame. Press K for keyboard demo.")
                camera_failure_reported = True
            if wizard and not demo_mode:
                wizard.observe(tracking, now)
                if wizard.complete:
                    calibration = wizard.value
                    if args.bow_speed is not None:
                        calibration = replace(calibration, bow_full_speed=args.bow_speed)
                    instrument = Instrument(calibration, controls, bow_mode)
                    calibrated = True
                    try:
                        save_calibration(args.config, calibration)
                        report("Calibration saved. Preview a note with your left index, then sweep your right wrist.")
                    except OSError as exc:
                        report(f"Calibration works for this session but was not saved: {exc}")
                    wizard = None
            if demo_mode:
                state = keyboard.update(now, volume)
            elif wizard:
                state = Instrument(wizard.value, controls, bow_mode).update(None, now, volume)
            else:
                state = instrument.update(tracking, now, volume)
            bus.publish(state.audio)
            if audio.health.error and notice != audio.health.error:
                report(audio.health.error)
            image = camera_frame.image if camera_frame and now - camera_frame.timestamp < 0.5 else None
            canvas = renderer.draw(image, state, calibration, tracking, wizard,
                                   demo_mode, bow_mode, audio, notice, help_visible)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(10)
            if key != -1:
                key &= 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("h"):
                help_visible = not help_visible
            elif key in (ord("+"), ord("=")):
                volume = min(1.0, volume + 0.05)
            elif key == ord("-"):
                volume = max(0.0, volume - 0.05)
            elif key == ord("r") and not args.silent and audio.stream is None:
                try:
                    audio.start(); report("Audio output started.")
                except Exception as exc:
                    report(f"Audio retry failed: {exc}")
            elif key == ord("b"):
                bow_mode = "precision" if bow_mode == "auto" else "auto"
                instrument.bow.set_mode(bow_mode)
            elif key == ord("k"):
                keyboard.bowing, keyboard.stroke_until = False, 0.0
                if not demo_mode:
                    demo_mode = True
                elif vision and vision.ready:
                    demo_mode = False
                    instrument = Instrument(calibration, controls, bow_mode)
                    if not calibrated:
                        wizard = Calibrator(mirrored, args.confidence)
                else:
                    report("Camera tracking is unavailable. Restart without --demo after fixing camera/model setup.")
            elif key == ord("c") and not demo_mode:
                wizard = Calibrator(mirrored, args.confidence)
                help_visible = False
            elif wizard and not demo_mode and key == ord(" "):
                wizard.advance(now)
            elif demo_mode:
                keyboard.key(key, now)
        return 0
    finally:
        bus.publish(AudioState(time.perf_counter(), volume=0.0))
        if audio.stream is not None:
            time.sleep(0.10)  # Let the output envelope decay before closing.
        for resource in (audio, vision, camera):
            if resource is not None:
                with suppress(Exception):
                    resource.close()
        with suppress(cv2.error):
            cv2.destroyAllWindows()


def main() -> int:
    try:
        return run(parser().parse_args())
    except (ValueError, OSError, cv2.error) as exc:
        print(f"Air Violin could not start: {exc}", file=sys.stderr)
        return 2
