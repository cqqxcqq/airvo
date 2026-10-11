"""Render the real OpenGL pipeline without a camera or audio device.

Synthetic landmarks establish reproducible geometry/texture checks, not proof
that a performer can hold or play the instrument. EGL/Mesa is acceptable here;
the report names the actual graphics driver rather than claiming laptop tests.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from air_violin.config import Calibration
from air_violin.scenes import SceneEngine
from air_violin.stage_window import StageWindow
from air_violin.types import AudioState, Hand, MusicalState, PlayState, Point, TrackingFrame


def synthetic_hands(timestamp, tilt=0.):
    """Anatomically separated palm points; tips do not determine mesh anchoring."""
    hands = []
    for label, center in (("Left", (.34, .42)), ("Right", (.75, .58))):
        cx, cy = center
        coordinates = ((0, .15), (-.07, .07), (-.09, .015), (-.10, -.03), (-.11, -.07),
                       (-.04, -.015), (-.045, -.085), (-.05, -.14), (-.05, -.18),
                       (0, -.04), (0, -.115), (0, -.165), (0, -.21),
                       (.035, -.015), (.04, -.085), (.04, -.13), (.04, -.17),
                       (.07, .02), (.08, -.04), (.08, -.08), (.08, -.11))
        points = tuple(Point(cx + x, cy + y, x * tilt) for x, y in coordinates)
        world = tuple(Point(x * .4, y * .4, x * tilt * .4) for x, y in coordinates)
        # The appended field exists in the updated Tasks adapter; keep this
        # script readable on older data classes during isolated renderer work.
        try:
            hand = Hand(label, .99, points, world_points=world)
        except TypeError:
            hand = Hand(label, .99, points)
        hands.append(hand)
    return TrackingFrame(timestamp, tuple(hands))


def integrated_captures(stage, output, calibration, state, source=None):
    """Exercise the application's real CPU HUD through the native GL window."""
    from air_violin.audio import AudioEngine, StateBus
    from air_violin.calibration import Calibrator
    from air_violin.render import Renderer
    audio = AudioEngine(StateBus())  # No sound device is opened by this check.
    state = replace(state, finger=Point(.29, .24), wrist=Point(.75, .73),
                    fingering_mode="fingers", finger_states=(1, 1, 1, 0),
                    finger_scores=(1., 1., 1., 0.))
    for scene in ("studio", "nebula", "midnight-rain", "forest-of-light"):
        renderer = Renderer(scene=scene, quality="high", reduced_motion=True)
        renderer.use_3d, renderer.guides, renderer.debug = True, False, True
        renderer.graphics_label = stage.renderer_name
        now = time.perf_counter()
        tracking = synthetic_hands(now, tilt=.35)
        current = replace(state, audio=replace(state.audio, timestamp=now))
        canvas = renderer.draw(source, current, calibration, tracking, None, False,
                               "auto", audio, notice="Synthetic pose / headless GL validation")
        stage.present(canvas, tracking, calibration, current, now,
                      viewport=renderer.camera_viewport)
        stage.capture(output / f"integrated-{scene}-3d.png")
    wizard = Calibrator(mirrored=True, mode="fingers", initial=calibration)
    wizard.step = 3
    wizard.message = ("Finger poses are too similar. Show your palm, lift the index, "
                      "then curl it farther before pressing SPACE again.")
    now = time.perf_counter()
    tracking = synthetic_hands(now)
    wizard.observe(tracking, now)
    canvas = renderer.draw(source, state, calibration, tracking, wizard, False,
                           "auto", audio, notice="Synthetic calibration / no webcam")
    stage.present(canvas, tracking, calibration, state, now, draw_instrument=False,
                  viewport=renderer.camera_viewport)
    stage.capture(output / "integrated-finger-calibration.png")


def validate(output, image_path=None):
    output.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((710, 920, 3), np.uint8)
    canvas[:80] = (30, 40, 170)
    canvas[-80:] = (170, 50, 20)
    canvas[:, :40] = (30, 170, 40)
    canvas[:, -40:] = (150, 30, 150)
    calibration = Calibration()
    state = MusicalState(AudioState(time.perf_counter(), midi=67, intensity=.72, gate=True),
                         phase=PlayState.BOWING, string=1, zone=3,
                         left_fresh=True, right_fresh=True)
    stage = StageWindow(headless=True, visible=False)
    try:
        stage.present(canvas, None, calibration, state, time.perf_counter(), draw_instrument=False)
        actual = stage.read_pixels()
        # Detect flipped rows or RGB/BGR confusion in the real uploaded texture.
        for x, y in ((200, 20), (200, 680), (20, 300), (890, 300)):
            assert np.max(np.abs(actual[y, x].astype(int) - canvas[y, x].astype(int))) <= 1
        stage.capture(output / "texture-orientation.png")
        timings = []
        source = cv2.imread(str(image_path)) if image_path else None
        if image_path and source is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        for scene in ("studio", "nebula", "midnight-rain", "forest-of-light"):
            engine = SceneEngine(scene)
            now = time.perf_counter()
            frame = np.zeros((710, 920, 3), np.uint8)
            frame[:] = (20, 23, 29)
            frame[64:624, 20:900] = engine.render(880, 560, now, "high", True, .7)
            if source is not None:
                image = cv2.resize(source, (747, 560))
                frame[64:624, 86:833] = image
            cv2.putText(frame, "AIR VIOLIN / REAL 3D RENDER CHECK", (24, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (224, 229, 238), 1, cv2.LINE_AA)
            cv2.putText(frame, "Synthetic hand pose - not hardware acceptance", (25, 670),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (179, 191, 207), 1, cv2.LINE_AA)
            guides = synthetic_hands(now, tilt=.35)
            for hand in guides.hands:
                for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8),
                              (5, 9, 10, 11, 12), (9, 13, 14, 15, 16),
                              (13, 17, 18, 19, 20), (17, 0)):
                    pts = np.array([(86 + hand.points[i].x * 747,
                                     64 + hand.points[i].y * 560) for i in chain], np.int32)
                    cv2.polylines(frame, [pts], False, (153, 178, 184), 2, cv2.LINE_AA)
            for i in range(8):
                now = time.perf_counter()
                tracking = synthetic_hands(now, tilt=.35)
                before = time.perf_counter()
                stage.present(frame, tracking, calibration,
                              replace(state, audio=replace(state.audio, timestamp=now)),
                              now, viewport=(86, 64, 747, 560))
                stage.gl.glFinish()
                timings.append((time.perf_counter() - before) * 1000)
            assert stage.violin_pose is not None and stage.bow_pose is not None
            rendered = stage.read_pixels()
            assert np.count_nonzero(np.any(rendered != frame, axis=2)) > 1000
            # Geometry cannot draw over the performance strip or upper header.
            assert np.array_equal(rendered[570:620, 20:900], frame[570:620, 20:900])
            stage.capture(output / f"{scene}-3d.png")
            if scene == "studio":
                demo = frame.copy()
                demo[64:624, 20:900] = engine.render(880, 560, now, "high", True, .7)
                stage.present(demo, None, calibration, state, now, demo=True)
                stage.capture(output / "complete-model-demo.png")
        now = time.perf_counter()
        stale = synthetic_hands(now - .2)
        stage.present(frame, stale, calibration, state, now)
        assert stage.violin_pose is None and stage.bow_pose is None
        key = stage.pyglet.window.key
        stage._window.dispatch_event("on_key_press", key.ENTER, 0)
        assert 13 in stage.poll()
        integrated_captures(stage, output, calibration, state, source)
        report = {"pyglet": stage.pyglet.version,
                  "graphics_renderer": stage.renderer_name,
                  "opengl_version": stage.renderer_version,
                  "mode": "headless EGL; synthetic poses; no camera/audio hardware",
                  "texture_orientation": "PASS",
                  "mesh_depth_and_shading": "PASS: actual vertex/index meshes rendered",
                  "tracking_timeout": "PASS",
                  "hud_scissor": "PASS",
                  "application_hud_integration": "PASS: actual Renderer.draw + native GL",
                  "render_completion_median_ms": float(np.median(timings)),
                  "render_completion_p95_ms": float(np.percentile(timings, 95)),
                  "unverified": ["Windows graphics context", "physical hand grip",
                                 "camera tracking", "audio hardware", "real finger occlusion"]}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        stage.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/3d-review"))
    parser.add_argument("--image", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.output, arguments.image), indent=2))
