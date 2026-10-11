from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from air_violin.audio import StateBus, Synthesizer
from air_violin.capture import LatestValue
from air_violin.config import Calibration, Controls
from air_violin.demo import KeyboardInstrument
from air_violin.effects import MusicalEffects
from air_violin.instrument_visual import BowAnchor, BowPose
from air_violin.models import verify_segmenter
from air_violin.render import Renderer
from air_violin.runtime import ControlWorker
from air_violin.scenes import SceneEngine, SCENES, cover_image
from air_violin.segmentation import PersonMask, composite_person
from air_violin.types import AudioState, TrackingFrame
from tests.helpers import hand
from tools.fetch_model import fetch_segmenter


class StageTests(unittest.TestCase):
    def test_bow_points_toward_violin_with_reversed_screen_regions(self):
        from air_violin.config import Rect
        from air_violin.instrument_visual import violin_placement
        from air_violin.types import Point
        import math
        for mirrored, x, board, bow in (
                (True, .78, Rect(.08, .22, .48, .78), Rect(.53, .22, .94, .82)),
                (False, .22, Rect(.52, .22, .92, .78), Rect(.06, .22, .47, .82))):
            cal = replace(Calibration(), mirrored=mirrored, board=board, bow=bow)
            points = list(hand('Right').points)
            points[0], points[9] = Point(x, .55), Point(x, .40)
            points[2], points[5], points[17] = Point(x, .45), Point(x, .45), Point(x + .08, .45)
            pose = BowAnchor().update(TrackingFrame(1, (replace(hand('Right'), points=tuple(points)),)), cal, 1, 640, 480)
            cx, cy, angle = violin_placement(cal.board, cal.bow)
            toward = (cx * 640 - pose.grip[0], cy * 480 - pose.grip[1])
            self.assertGreater(math.cos(pose.angle) * toward[0] + math.sin(pose.angle) * toward[1], 0)
            self.assertEqual(angle < 0, mirrored)
    def test_bow_follows_calibrated_right_hand_in_both_mirror_modes(self):
        for mirrored in (True, False):
            cal = replace(Calibration(), mirrored=mirrored, left_label='Right', right_label='Left')
            selected = hand('Left', wrist=(.75, .5))
            other = hand('Right', wrist=(.20, .5))
            from air_violin.types import Point
            selected_points, other_points = list(selected.points), list(other.points)
            for index in (2, 5):
                selected_points[index] = Point(.75, .4)
                other_points[index] = Point(.20, .4)
            selected = replace(selected, points=tuple(selected_points))
            other = replace(other, points=tuple(other_points))
            frame = TrackingFrame(1, (selected, other))
            pose = BowAnchor().update(frame, cal, 1, 640, 480)
            expected = ((selected.points[2].x + selected.points[5].x) * 320,
                        (selected.points[2].y + selected.points[5].y) * 240)
            self.assertEqual(pose.grip, expected)

    def test_bow_rejects_stale_future_low_confidence_and_duplicate_hands(self):
        cal = Calibration()
        anchor = BowAnchor()
        for frame in (TrackingFrame(0, (hand('Right'),)), TrackingFrame(2, (hand('Right'),)),
                      TrackingFrame(1, (hand('Right', confidence=.2),)),
                      TrackingFrame(1, (hand('Right'), hand('Right')))):
            self.assertIsNone(anchor.update(frame, cal, 1, 640, 480))

    def test_bow_position_expires_after_occlusion(self):
        anchor = BowAnchor()
        pose = anchor.update(TrackingFrame(1, (hand('Right'),)), Calibration(), 1, 640, 480)
        self.assertIsNotNone(pose)
        self.assertIsNotNone(anchor.update(TrackingFrame(1.05), Calibration(), 1.05, 640, 480))
        self.assertIsNone(anchor.update(TrackingFrame(1.25), Calibration(), 1.25, 640, 480))

    def test_bow_smoothing_does_not_spin_at_angle_wrap(self):
        anchor = BowAnchor()
        base = hand('Right')
        from air_violin.types import Point
        poses = []
        for t, dx in ((1, -.001), (1.033, .001)):
            points = list(base.points)
            points[0] = Point(.7, .5)
            points[9] = Point(.7 + dx, .4)
            poses.append(anchor.update(TrackingFrame(t, (replace(base, points=tuple(points)),)), Calibration(), t, 640, 480))
        self.assertLess(abs(poses[1].angle - poses[0].angle), .1)

    def test_zero_one_and_half_masks_composite_expected_pixels(self):
        front = np.full((48, 64, 3), 200, np.uint8)
        back = np.full_like(front, 20)
        for value, expected in ((0., 20), (1., 200), (.5, 110)):
            mask = PersonMask(1, 1, np.full((12, 16), value, np.float32), 1)
            result, used = composite_person(front, back, mask, 1)
            self.assertTrue(used)
            np.testing.assert_allclose(result, expected, atol=1)

    def test_bad_stale_and_future_masks_fall_back_to_camera(self):
        front = np.full((48, 64, 3), 123, np.uint8)
        back = np.zeros_like(front)
        for mask in (None, PersonMask(.5, 0, np.ones((12, 16)), 0),
                     PersonMask(2, 0, np.ones((12, 16)), 0),
                     PersonMask(1, 0, np.empty((0, 0)), 0),
                     PersonMask(1, 0, np.full((12, 16), np.nan), 0)):
            result, used = composite_person(front, back, mask, 1)
            self.assertFalse(used)
            np.testing.assert_array_equal(result, front)

    def test_fresh_hand_landmarks_preserve_fingers_but_stale_ones_do_not(self):
        front = np.full((480, 640, 3), 200, np.uint8)
        back = np.zeros_like(front)
        mask = PersonMask(1, 0, np.zeros((48, 64), np.float32), 0)
        right = hand('Right')
        tip = right.points[8]
        y, x = int(round(tip.y * 480)), int(round(tip.x * 640))
        fresh, _ = composite_person(front, back, mask, 1, TrackingFrame(1, (right,)))
        stale, _ = composite_person(front, back, mask, 1, TrackingFrame(.5, (right,)))
        self.assertEqual(int(fresh[y, x, 0]), 200)
        self.assertEqual(int(stale[y, x, 0]), 0)

    def test_cover_fit_keeps_geometry_and_center_crop(self):
        image = np.zeros((40, 80, 3), np.uint8)
        image[:, 30:50] = 255
        result = cover_image(image, 40, 40)
        self.assertEqual(result.shape, (40, 40, 3))
        self.assertEqual(np.count_nonzero(result[:, :, 0]), 40 * 20)

    def test_all_procedural_presets_and_reduced_motion(self):
        for preset in SCENES:
            scene = SceneEngine(preset)
            a = scene.render(640, 480, 1, reduced_motion=True)
            b = scene.render(640, 480, 100, reduced_motion=True)
            self.assertEqual(a.dtype, np.uint8)
            np.testing.assert_array_equal(a, b)
            self.assertGreater(float(a.std()), 2)

    def test_custom_background_decodes_and_missing_file_is_actionable(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'custom.png'
            cv2.imwrite(str(file), np.full((24, 80, 3), 125, np.uint8))
            scene = SceneEngine(custom=file)
            self.assertEqual(scene.preset, 'custom')
            self.assertEqual(int(scene.render(64, 48, 1)[20, 20, 0]), 125)
            with self.assertRaises(OSError):
                SceneEngine(custom=Path(folder) / 'missing.png')

    def test_effects_are_bounded_and_reduced_motion_clears_motion(self):
        effect = MusicalEffects()
        demo = KeyboardInstrument()
        demo.key(ord(' '), 1)
        image = np.zeros((480, 640, 3), np.uint8)
        for i in range(600):
            t = 1 + i / 30
            state = demo.update(t, .65)
            effect.draw(image, state, BowPose((300, 200), 3.14, 300, t), t, (200, 180, 100), 'D4', 'high')
        self.assertLessEqual(len(effect.trail), 36)
        self.assertLessEqual(len(effect.particles), 64)
        self.assertLessEqual(len(effect.echoes), 4)
        effect.draw(image, state, None, t, (200, 180, 100), 'D4', reduced_motion=True)
        self.assertFalse(effect.trail)
        self.assertFalse(effect.particles)

    def test_segmenter_pin_cannot_be_bypassed_by_modified_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.tflite'
            path.write_bytes(b'changed')
            path.with_suffix('.tflite.sha256').write_text(hashlib.sha256(b'changed').hexdigest())
            with self.assertRaises(ValueError):
                verify_segmenter(path)

    def test_segmenter_download_is_verified_and_reused_offline(self):
        payload = b'\x00\x00\x00\x00TFL3' + b'test model'
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.tflite'
            with (patch('urllib.request.urlopen', return_value=io.BytesIO(payload)) as request,
                  patch('tools.fetch_model.SEGMENTER_SHA256', digest),
                  patch('air_violin.models.SEGMENTER_SHA256', digest)):
                self.assertEqual(fetch_segmenter(path), digest)
                self.assertEqual(fetch_segmenter(path), digest)
                self.assertEqual(request.call_count, 1)

    def test_bad_segmenter_download_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.tflite'
            path.write_bytes(b'keep original')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'bad download')):
                with self.assertRaises(ValueError):
                    fetch_segmenter(path, force=True)
            self.assertEqual(path.read_bytes(), b'keep original')


class BowAudioTests(unittest.TestCase):
    def spectral_centroid(self, intensity):
        synth = Synthesizer()
        state = AudioState(1, 69, intensity, True, 1, 1)
        synth.render(12000, state, 1)
        samples = synth.render(24000, state, 1)
        magnitude = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)
        return float(np.sum(magnitude * frequencies) / magnitude.sum())

    def test_faster_bow_changes_tone_color_not_just_gain(self):
        self.assertGreater(self.spectral_centroid(1), self.spectral_centroid(.2) * 1.2)

    def test_reversal_articulates_without_restarting_envelope_or_phase(self):
        synth = Synthesizer()
        state = AudioState(1, 69, .8, True, .65, 1)
        before = synth.render(12000, state, 1)
        amplitude = synth.amplitude
        after = synth.render(256, replace(state, timestamp=1.1, bow_direction=-1), 1.1)
        self.assertAlmostEqual(synth.amplitude, amplitude, delta=.001)
        self.assertGreater(synth.stroke_transient, .1)
        self.assertLess(abs(float(after[0] - before[-1])), .08)
        synth.render(4800, replace(state, bow_direction=-1), 1)
        self.assertLess(synth.stroke_transient, .01)

    def test_repeated_direction_does_not_retrigger_articulation(self):
        synth = Synthesizer()
        state = AudioState(1, 62, .8, True, .65, 1)
        synth.render(12000, state, 1)
        self.assertLess(synth.stroke_transient, .001)
        synth.render(256, state, 1)
        self.assertLess(synth.stroke_transient, .001)


class ControlThreadTests(unittest.TestCase):
    def test_keyboard_cannot_return_to_a_failed_camera_controller(self):
        from air_violin import app
        from air_violin.capture import CameraFrame
        import contextlib
        camera = SimpleNamespace(latest=LatestValue(), error='', close=unittest.mock.Mock())
        camera.latest.put(CameraFrame(0, time.perf_counter(), np.zeros((480, 640, 3), np.uint8)))
        vision = SimpleNamespace(latest=LatestValue(), ready=True, error='', close=unittest.mock.Mock())
        segmentation = SimpleNamespace(latest=LatestValue(), error='', configure=unittest.mock.Mock(), close=unittest.mock.Mock(), throttled=False)
        worker = SimpleNamespace(latest=LatestValue(), error='synthetic control failure', configure=unittest.mock.Mock(), close=unittest.mock.Mock())
        args = app.parser().parse_args(['--silent', '--renderer', '2d', '--fingering', 'pointing'])
        warnings = io.StringIO()
        with (patch.dict('os.environ', {'DISPLAY': ':test'}),
              patch.object(app, 'load_calibration', return_value=Calibration()),
              patch.object(app, 'verify_model'), patch.object(app, 'Camera', return_value=camera),
              patch.object(app, 'VisionWorker', return_value=vision),
              patch.object(app, 'SegmentationWorker', return_value=segmentation),
              patch.object(app, 'ControlWorker', return_value=worker),
              patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'resizeWindow'),
              patch.object(app.cv2, 'imshow'), patch.object(app.cv2, 'waitKey', side_effect=[ord('k'), ord('q')]),
              patch.object(app.cv2, 'getWindowProperty', return_value=1),
              patch.object(app.cv2, 'destroyAllWindows'), contextlib.redirect_stderr(warnings)):
            self.assertEqual(app.run(args), 0)
        self.assertTrue(worker.configure.call_count >= 2)
        self.assertFalse(any(call.args[-1] for call in worker.configure.call_args_list))
        self.assertIn('Camera controls are unavailable', warnings.getvalue())

    def test_camera_app_scene_keys_source_switch_calibration_and_cleanup(self):
        from air_violin import app
        from air_violin.capture import CameraFrame
        from air_violin.config import save_calibration
        from tools.validate import synthetic_frame
        camera, vision, segmentation = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
        for source in (camera, vision, segmentation):
            source.latest = LatestValue()
            source.error = ''
            source.ready = True
            source.close = unittest.mock.Mock()
        segmentation.configure = unittest.mock.Mock()
        segmentation.throttled = False
        now = time.perf_counter()
        camera.latest.put(CameraFrame(0, now, np.zeros((480, 640, 3), np.uint8)))
        vision.latest.put(synthetic_frame(now, 1, 2))
        segmentation.latest.put(PersonMask(now, 0, np.zeros((48, 64), np.float32), 1))
        keys = iter(map(ord, 'gsvdpmfkkbcq'))
        # Use real control/audio mailboxes; replace only external transports.
        from air_violin.audio import AudioEngine
        bus = StateBus()
        engine = AudioEngine(bus)
        def wait_key(_):
            time.sleep(.015)
            now = time.perf_counter()
            camera.latest.put(CameraFrame(1, now, np.zeros((480, 640, 3), np.uint8)))
            vision.latest.put(synthetic_frame(now, 1, 2))
            return next(keys)
        with tempfile.TemporaryDirectory() as folder:
            args = app.parser().parse_args(['--silent', '--renderer', '2d', '--fingering', 'pointing'])
            args.config = Path(folder) / 'calibration.json'
            save_calibration(args.config, Calibration())
            with (patch.dict('os.environ', {'DISPLAY': ':test'}),
                  patch.object(app, 'AudioEngine', return_value=engine),
                  patch.object(app, 'StateBus', return_value=bus),
                  patch.object(app, 'verify_model'),
                  patch.object(app, 'Camera', return_value=camera),
                  patch.object(app, 'VisionWorker', return_value=vision),
                  patch.object(app, 'SegmentationWorker', return_value=segmentation),
                  patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'resizeWindow'),
                  patch.object(app.cv2, 'imshow'), patch.object(app.cv2, 'waitKey', side_effect=wait_key),
                  patch.object(app.cv2, 'getWindowProperty', return_value=1),
                  patch.object(app.cv2, 'setWindowProperty') as fullscreen,
                  patch.object(app.cv2, 'destroyAllWindows')):
                self.assertEqual(app.run(args), 0)
                fullscreen.assert_called_once()
            self.assertFalse(bus.read_nowait(AudioState(0)).gate)
            for source in (camera, vision, segmentation):
                source.close.assert_called_once()
            self.assertGreater(segmentation.configure.call_count, 5)

    def test_control_continues_without_rendering_then_releases_stale_tracking(self):
        from tools.validate import synthetic_frame
        mailbox = LatestValue()
        bus = StateBus()
        worker = ControlWorker(SimpleNamespace(latest=mailbox), bus, Calibration(), Controls(), 'auto', .65)
        try:
            worker.configure(Calibration(), 'auto', .65, True)
            # No Renderer or app-loop calls: only fresh synthetic camera results.
            start = time.perf_counter()
            saw_bowing = False
            for i in range(15):
                now = time.perf_counter()
                frame = synthetic_frame(now, 1, 2)
                mailbox.put(frame)
                time.sleep(.012)
                state = bus.read_nowait(AudioState(0))
                saw_bowing |= state.gate
            self.assertTrue(saw_bowing)
            deadline = time.perf_counter() + .6
            while time.perf_counter() < deadline:
                state = bus.read_nowait(AudioState(0))
                if not state.gate:
                    break
                time.sleep(.012)
            self.assertFalse(state.gate)
            self.assertEqual(worker.error, '')
        finally:
            worker.close()

    def test_disabling_worker_transfers_audio_ownership_to_keyboard(self):
        mailbox = LatestValue()
        bus = StateBus()
        worker = ControlWorker(SimpleNamespace(latest=mailbox), bus, Calibration(), Controls(), 'auto', .65)
        try:
            worker.configure(Calibration(), 'auto', .65, True)
            worker.configure(Calibration(), 'auto', .65, False)
            demo = AudioState(time.perf_counter(), 69, .7, True)
            bus.publish(demo)
            time.sleep(.035)
            self.assertEqual(bus.read_nowait(AudioState(0)), demo)
        finally:
            worker.close()


if __name__ == '__main__':
    unittest.main()
