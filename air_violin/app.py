"""Desktop lifecycle and keyboard controls; the audio thread never calls this UI."""

import argparse
from collections import deque
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
from .models import DEFAULT_MODEL, DEFAULT_SEGMENTER, verify_model
from .render import Renderer, WINDOW
from .runtime import ControlWorker
from .scenes import SCENES
from .segmentation import SegmentationWorker
from .startup import ResourceLoader
from .types import AudioState
from .vision import VisionWorker


class OpenCVWindow:
    """Compatibility transport when an OpenGL 3.3 window is unavailable."""

    def __init__(self):
        self.closed = False
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 920, 710)

    def present(self, canvas, **kwargs):
        cv2.imshow(WINDOW, canvas)

    def poll(self):
        key = cv2.waitKey(10)
        self.closed = (key & 0xFF) in (27, ord("q")) if key != -1 else False
        if not self.closed:
            self.closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        return [key & 0xFF] if key != -1 else []

    def set_fullscreen(self, enabled):
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)

    def close(self):
        cv2.destroyAllWindows()


def open_window(backend, report):
    if backend != "2d":
        try:
            from .stage_window import StageWindow
            window = StageWindow()
            print(f"3D graphics: {window.renderer_name}")
            return window, True
        except Exception as exc:
            report(f"3D graphics unavailable: {exc}. Using 2D; update graphics drivers or use --renderer 2d.")
    return OpenCVWindow(), False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Air Violin Studio — playable visual studio")
    p.add_argument("--demo", action="store_true", help="Keyboard instrument; no camera or model required")
    p.add_argument("--silent", action="store_true", help="Explicitly disable hardware audio")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", choices=("auto", "dshow", "msmf"),
                   default="dshow" if sys.platform == "win32" else "auto",
                   help="Camera API; Windows defaults to DirectShow to avoid slow MSMF startup")
    p.add_argument("--fingering", choices=("fingers", "pointing"), default="fingers")
    p.add_argument("--renderer", choices=("auto", "3d", "2d"), default="auto",
                   help="Native 3D by default; 2D compatibility remains available")
    p.add_argument("--string", choices=("auto", "G", "D", "A", "E"), default="auto",
                   help="Lock a string for finger practice or select by bow-hand height")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-mirror", action="store_true", help="Changing mirror mode requires recalibration")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--segmenter-model", type=Path, default=DEFAULT_SEGMENTER)
    p.add_argument("--scene", choices=SCENES, default="studio")
    p.add_argument("--background", type=Path, help="Custom local JPG/PNG background")
    p.add_argument("--no-background", action="store_true")
    p.add_argument("--background-blur", action="store_true")
    p.add_argument("--quality", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--reduced-motion", action="store_true")
    p.add_argument("--effects", type=float, default=.65, help="Visual-effect intensity 0..1")
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
    if not 0 <= args.volume <= 1 or not 0 <= args.effects <= 1 or not 16000 <= args.sample_rate <= 96000 or not 64 <= args.block_size <= 2048:
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
    fingering_mode = args.fingering
    string_override = None if args.string == "auto" else ("G", "D", "A", "E").index(args.string)
    calibrated = calibration is not None and (fingering_mode == "pointing" or calibration.fingers is not None)
    if args.bow_speed is not None:
        calibration = replace(calibration or Calibration(mirrored=mirrored), bow_full_speed=args.bow_speed)
    calibration = calibration or Calibration(mirrored=mirrored)
    demo_mode = args.demo
    keyboard = KeyboardInstrument()
    bus = StateBus()
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    audio = AudioEngine(bus, device, args.sample_rate, args.block_size, controls.tracking_timeout)
    camera = vision = wizard = segmentation = control_worker = None
    instrument = Instrument(calibration, controls, args.bow_mode,
                            fingering_mode=fingering_mode, string_override=string_override)
    try:
        renderer = Renderer(args.scene, args.background, args.quality, args.reduced_motion,
                            not args.no_background, args.background_blur, args.effects,
                            controls.confidence, controls.tracking_timeout)
    except (OSError, ValueError, cv2.error) as exc:
        report(f"Custom background unavailable: {exc}. Using Studio scene.")
        renderer = Renderer(quality=args.quality, reduced_motion=args.reduced_motion,
                            background=not args.no_background, effect_intensity=args.effects,
                            confidence=controls.confidence, timeout=controls.tracking_timeout)
    help_visible = demo_mode
    volume = args.volume
    bow_mode = args.bow_mode
    window = camera_loader = audio_loader = None
    audio_owned = args.silent
    audio_reported = camera_reported = False
    pending_keys = deque(maxlen=64)
    try:
        # Create the window before any blocking camera/audio-device operation.
        window, renderer.use_3d = open_window(args.renderer, report)
        if renderer.use_3d:
            window.confidence, window.timeout = controls.confidence, controls.tracking_timeout
        renderer.graphics_label = "3D" if renderer.use_3d else "2D compatibility"
        if not args.silent:
            def start_audio():
                audio.start()
                return audio
            audio_loader = ResourceLoader("audio", start_audio)
        else:
            report("Silent mode: no hardware audio output.")
        if not demo_mode:
            def start_camera():
                verify_model(args.model)
                return Camera(args.camera, args.width, args.height, mirrored, args.backend)
            camera_loader = ResourceLoader("camera", start_camera)
        vision_failure_reported = False
        camera_failure_reported = False
        segmentation_failure_reported = False
        control_failure_reported = False
        fullscreen = False
        last_underruns = 0
        priority_until = 0.0
        started = time.perf_counter()
        while True:
            now = time.perf_counter()
            if audio_loader and audio_loader.done.is_set() and not audio_reported:
                audio_owned = audio_loader.take() is not None
                audio_reported = True
                if audio_loader.error:
                    report(f"Audio unavailable: {audio_loader.error}. R retries audio; controls remain usable.")
                else:
                    print(f"Audio ready in {audio_loader.elapsed:.2f}s")
            if camera_loader and camera_loader.done.is_set() and not camera_reported:
                camera = camera_loader.take()
                camera_reported = True
                if camera is None:
                    report(f"Camera unavailable: {camera_loader.error}. Using keyboard demo.")
                    demo_mode, help_visible = True, True
                else:
                    print(f"Camera opened in {camera_loader.elapsed:.2f}s ({args.backend})")
                    vision = VisionWorker(camera, args.model, args.confidence)
                    control_worker = ControlWorker(vision, bus, calibration, controls, bow_mode, volume,
                                                   fingering_mode=fingering_mode, string_override=string_override)
                    segmentation = SegmentationWorker(camera, args.segmenter_model)
                    if not calibrated:
                        wizard = Calibrator(mirrored, args.confidence, mode=fingering_mode, initial=calibration)
            camera_frame = camera.latest.get() if camera else None
            tracking = vision.latest.get() if vision else None
            mask = segmentation.latest.get() if segmentation else None
            if segmentation and segmentation.error and not segmentation_failure_reported:
                report(segmentation.error)
                segmentation_failure_reported = True
            if control_worker and control_worker.error and not control_failure_reported:
                report(control_worker.error)
                control_failure_reported = True
                demo_mode, wizard = True, None
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
                    instrument = Instrument(calibration, controls, bow_mode,
                                            fingering_mode=fingering_mode, string_override=string_override)
                    calibrated = True
                    try:
                        save_calibration(args.config, calibration)
                        report("Calibration saved. Lift/curl left fingers and sweep your right wrist." if fingering_mode == "fingers"
                               else "Calibration saved. Point with your left index and sweep your right wrist.")
                    except OSError as exc:
                        report(f"Calibration works for this session but was not saved: {exc}")
                    wizard = None
            if demo_mode:
                if control_worker:
                    control_worker.configure(calibration, bow_mode, volume, False,
                                             fingering_mode=fingering_mode, string_override=string_override)
                state = keyboard.update(now, volume)
            elif wizard:
                if control_worker:
                    control_worker.configure(calibration, bow_mode, volume, False,
                                             fingering_mode=fingering_mode, string_override=string_override)
                state = Instrument(wizard.value, controls, bow_mode,
                                   fingering_mode=fingering_mode, string_override=string_override).update(None, now, volume)
            else:
                if control_worker:
                    control_worker.configure(calibration, bow_mode, volume, True,
                                             fingering_mode=fingering_mode, string_override=string_override)
                    state = control_worker.latest.get() or instrument.update(None, now, volume)
                else:
                    state = instrument.update(None, now, volume)
            if demo_mode or wizard:
                bus.publish(state.audio)
            if audio.health.underruns > last_underruns:
                priority_until = now + 3
            last_underruns = audio.health.underruns
            pressure = now < priority_until or (tracking is not None and now - tracking.timestamp > .140)
            if segmentation:
                segmentation.configure(renderer.background and not demo_mode and wizard is None,
                                       renderer.quality, pressure)
            if audio.health.error and notice != audio.health.error:
                report(audio.health.error)
            image = camera_frame.image if camera_frame and now - camera_frame.timestamp < 0.5 else None
            startup = ""
            if not demo_mode:
                if camera_loader and not camera_reported:
                    startup = f"Opening webcam {args.camera} ({args.backend})... {camera_loader.elapsed:.1f}s"
                elif vision and not vision.ready and not vision.error:
                    startup = "Loading local hand-tracking model..."
                elif camera and camera_frame is None:
                    startup = "Waiting for the webcam's first frame..."
            if audio_loader and not audio_reported:
                startup += ("  |  " if startup else "") + f"Opening audio output... {audio_loader.elapsed:.1f}s"
            canvas = renderer.draw(image, state, calibration, tracking, wizard,
                                   demo_mode, bow_mode, audio, notice, help_visible, mask,
                                   camera_frame.timestamp if camera_frame else None, segmentation, startup)
            try:
                window.present(canvas, tracking=tracking, calibration=calibration, state=state, now=now,
                               demo=demo_mode, draw_instrument=not (wizard or help_visible or startup),
                               viewport=renderer.camera_viewport)
            except Exception as exc:
                if not renderer.use_3d:
                    raise
                report(f"3D rendering stopped: {exc}. Continuing in 2D compatibility mode.")
                with suppress(Exception):
                    window.close()
                renderer.use_3d = False
                window = OpenCVWindow()
                window.present(canvas)
            pending_keys.extend(window.poll())
            key = pending_keys.popleft() if pending_keys else -1
            if key in (27, ord("q")) or window.closed:
                break
            if renderer.use_3d:
                # Vsync is driver-dependent. Bound UI work explicitly while
                # inference and musical controls continue on their own threads.
                time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - now)))
            if key == ord("h"):
                help_visible = not help_visible
            elif key == ord("g"):
                renderer.guides = not renderer.guides
            elif key == ord("s"):
                renderer.scene.cycle()
            elif key == ord("v"):
                renderer.background = not renderer.background
            elif key == ord("d"):
                renderer.debug = not renderer.debug
            elif key == ord("p"):
                renderer.cycle_quality()
            elif key == ord("m"):
                renderer.reduced_motion = not renderer.reduced_motion
            elif key == ord("f"):
                fullscreen = not fullscreen
                window.set_fullscreen(fullscreen)
            elif key in (ord("+"), ord("=")):
                volume = min(1.0, volume + 0.05)
            elif key == ord("-"):
                volume = max(0.0, volume - 0.05)
            elif key == ord("r") and not args.silent and audio.stream is None and audio_reported:
                audio_loader = ResourceLoader("audio", start_audio)
                audio_reported, audio_owned = False, False
            elif key == ord("b"):
                bow_mode = "precision" if bow_mode == "auto" else "auto"
                instrument.bow.set_mode(bow_mode)
            elif key == ord("k"):
                keyboard.bowing, keyboard.stroke_until = False, 0.0
                if not demo_mode:
                    demo_mode = True
                elif (vision and vision.ready and control_worker and not control_worker.error
                      and not camera.error):
                    demo_mode = False
                    instrument = Instrument(calibration, controls, bow_mode,
                                            fingering_mode=fingering_mode, string_override=string_override)
                    if not calibrated:
                        wizard = Calibrator(mirrored, args.confidence, mode=fingering_mode, initial=calibration)
                else:
                    report("Camera controls are unavailable. Keep keyboard mode or restart after fixing camera/model setup.")
            elif key == ord("c") and not demo_mode:
                wizard = Calibrator(mirrored, args.confidence, mode=fingering_mode, initial=calibration)
                help_visible = False
            elif key == ord("j") and not demo_mode:
                fingering_mode = "pointing" if fingering_mode == "fingers" else "fingers"
                instrument = Instrument(calibration, controls, bow_mode,
                                        fingering_mode=fingering_mode, string_override=string_override)
                if fingering_mode == "fingers" and calibration.fingers is None:
                    wizard = Calibrator(mirrored, args.confidence, mode="fingers", initial=calibration)
                else:
                    wizard = None
                report("Finger controls: curl/lift left fingers; right wrist height selects strings." if fingering_mode == "fingers"
                       else "Pointing controls: left index selects a screen note.")
            elif key in (ord("["), ord("]")) and not demo_mode and fingering_mode == "fingers" and not wizard:
                current = state.string if string_override is None else string_override
                string_override = max(0, min(3, current + (-1 if key == ord("[") else 1)))
                report(f"String locked: {('G', 'D', 'A', 'E')[string_override]}. Backslash restores bow-height selection.")
            elif key == ord("\\") and not demo_mode and fingering_mode == "fingers":
                string_override = None
                report("String selection follows right-wrist height: G / D / A / E, top to bottom.")
            elif wizard and not demo_mode and key == ord(" "):
                wizard.advance(now)
            elif demo_mode:
                keyboard.key(key, now)
        return 0
    finally:
        for loader in (camera_loader, audio_loader):
            if loader:
                loader.close()
        if control_worker:
            control_worker.close()
        bus.publish(AudioState(time.perf_counter(), volume=0.0))
        if audio.stream is not None:
            time.sleep(0.10)  # Let the output envelope decay before closing.
        for resource in (audio if audio_owned else None, segmentation, vision, camera):
            if resource is not None:
                with suppress(Exception):
                    resource.close()
        if window:
            with suppress(Exception):
                window.close()


def main() -> int:
    try:
        return run(parser().parse_args())
    except (ValueError, OSError, cv2.error) as exc:
        print(f"Air Violin could not start: {exc}", file=sys.stderr)
        return 2
