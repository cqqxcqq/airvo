"""Exercise actual desktop lifecycle while replacing only hardware transports."""

from dataclasses import replace
import io
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from air_violin import app
from air_violin.capture import CameraFrame, LatestValue
from air_violin.audio import StateBus
from air_violin.config import Calibration, Controls
from air_violin.runtime import ControlWorker
from air_violin.types import AudioState, Point, TrackingFrame
from tests.test_fingering import articulated, finger_calibration


class FakeWindow:
    def __init__(self, keys, ready=lambda: True):
        self.closed = False
        self.keys, self.ready = iter(keys), ready
        self.frames = 0
        self.close = Mock()
        self.set_fullscreen = Mock()

    def present(self, canvas, **kwargs):
        self.frames += 1
        assert canvas.shape == (710, 920, 3)

    def poll(self):
        time.sleep(.005)
        if self.frames > 60:
            raise AssertionError('Application did not finish expected startup/input sequence')
        return [next(self.keys)] if self.ready() else []


class RevisionAppTests(unittest.TestCase):
    def test_assisted_defaults_keep_a_resting_bow_on_d_unless_auto_is_requested(self):
        defaults = app.parser().parse_args([])
        self.assertEqual(defaults.bow_mode, 'assisted')
        self.assertEqual(app.initial_string(defaults), 1)
        self.assertIsNone(app.initial_string(app.parser().parse_args(['--string', 'auto'])))
        self.assertIsNone(app.initial_string(app.parser().parse_args(['--bow-mode', 'auto'])))
        self.assertEqual(app.initial_string(app.parser().parse_args(['--string', 'E'])), 3)

    def test_old_finger_feature_calibration_requires_recalibration(self):
        current = replace(Calibration(), version=2, fingers=finger_calibration())
        obsolete = replace(current, fingers=replace(current.fingers, feature_version=1))
        self.assertTrue(app.has_finger_calibration(current))
        self.assertFalse(app.has_finger_calibration(obsolete))
        self.assertFalse(app.has_finger_calibration(Calibration()))

    def test_camera_sustain_stop_and_resize_keys_reach_their_owners(self):
        ready = threading.Event()
        camera = SimpleNamespace(latest=LatestValue(), error='', close=Mock())
        vision = SimpleNamespace(latest=LatestValue(), error='', ready=True, close=Mock())
        segmentation = SimpleNamespace(latest=LatestValue(), error='', throttled=False,
                                       configure=Mock(), close=Mock())
        worker = SimpleNamespace(latest=LatestValue(), error='', configure=Mock(), close=Mock(),
                                 toggle_assisted=Mock(), stop_assisted=Mock())
        window = FakeWindow(list(map(ord, ' 0.,bbbq')), ready.is_set)
        def make_worker(*_, **kwargs):
            ready.set()
            return worker
        args = app.parser().parse_args(['--silent', '--instrument-size', '1.2'])
        cal = replace(Calibration(), version=2, fingers=finger_calibration())
        with (patch.dict('os.environ', {'DISPLAY': ':test'}),
              patch.object(app, 'open_window', return_value=(window, True)),
              patch.object(app, 'load_calibration', return_value=cal),
              patch.object(app, 'verify_model'), patch.object(app, 'Camera', return_value=camera),
              patch.object(app, 'VisionWorker', return_value=vision),
              patch.object(app, 'SegmentationWorker', return_value=segmentation),
              patch.object(app, 'ControlWorker', side_effect=make_worker)):
            self.assertEqual(app.run(args), 0)
        worker.toggle_assisted.assert_called_once()
        worker.stop_assisted.assert_called_once()
        self.assertAlmostEqual(window.instrument_size, 1.2)
        self.assertEqual({c.args[1] for c in worker.configure.call_args_list}, {'assisted', 'auto', 'precision'})
        self.assertTrue(all(c.kwargs['string_override'] == 1 for c in worker.configure.call_args_list))

    def test_window_draws_while_camera_factory_is_still_blocked(self):
        entered, release = threading.Event(), threading.Event()
        events = []
        window = FakeWindow([ord('q')], entered.is_set)
        def open_window(*_):
            events.append('window')
            return window, False
        def slow_camera(*_):
            events.append('camera')
            entered.set()
            release.wait(2)
            raise RuntimeError('simulated unavailable camera')
        args = app.parser().parse_args(['--silent', '--renderer', '2d'])
        with tempfile.TemporaryDirectory() as folder:
            args.config = Path(folder) / 'unused.json'
            try:
                with (patch.dict('os.environ', {'DISPLAY': ':test'}),
                      patch.object(app, 'open_window', side_effect=open_window),
                      patch.object(app, 'verify_model'),
                      patch.object(app, 'Camera', side_effect=slow_camera)):
                    self.assertEqual(app.run(args), 0)
            finally:
                release.set()
        self.assertEqual(events, ['window', 'camera'])
        self.assertGreaterEqual(window.frames, 1)
        window.close.assert_called_once()

    def test_camera_finger_pointing_switch_and_string_controls_reach_worker(self):
        ready = threading.Event()
        camera = SimpleNamespace(latest=LatestValue(), error='', close=Mock())
        vision = SimpleNamespace(latest=LatestValue(), error='', ready=True, close=Mock())
        segmentation = SimpleNamespace(latest=LatestValue(), error='', throttled=False,
                                       configure=Mock(), close=Mock())
        worker = SimpleNamespace(latest=LatestValue(), error='', configure=Mock(), close=Mock())
        window = FakeWindow(list(map(ord, 'jj]\\bcq')), ready.is_set)
        original_poll = window.poll
        def poll():
            now = time.perf_counter()
            camera.latest.put(CameraFrame(window.frames, now, np.zeros((480, 640, 3), np.uint8)))
            vision.latest.put(TrackingFrame(now, (articulated(), articulated(label='Right'))))
            return original_poll()
        window.poll = poll
        def make_worker(*_, **kwargs):
            ready.set()
            return worker
        args = app.parser().parse_args(['--silent', '--renderer', '2d', '--string', 'D'])
        cal = replace(Calibration(), version=2, fingers=finger_calibration())
        with (patch.dict('os.environ', {'DISPLAY': ':test'}),
              patch.object(app, 'open_window', return_value=(window, False)),
              patch.object(app, 'load_calibration', return_value=cal),
              patch.object(app, 'verify_model'), patch.object(app, 'Camera', return_value=camera),
              patch.object(app, 'VisionWorker', return_value=vision),
              patch.object(app, 'SegmentationWorker', return_value=segmentation),
              patch.object(app, 'ControlWorker', side_effect=make_worker)):
            self.assertEqual(app.run(args), 0)
        configs = worker.configure.call_args_list
        self.assertTrue(any(c.kwargs['fingering_mode'] == 'fingers' for c in configs))
        self.assertTrue(any(c.kwargs['fingering_mode'] == 'pointing' for c in configs))
        self.assertTrue(any(c.kwargs['string_override'] == 2 for c in configs))
        self.assertTrue(any(c.kwargs['string_override'] is None for c in configs))
        self.assertTrue(any(c.args[3] is False for c in configs))  # Calibration releases audio.
        for resource in (camera, vision, segmentation, worker, window):
            resource.close.assert_called_once()

    def test_failed_3d_backend_reports_and_uses_working_2d_transport(self):
        import sys
        fake_gl = SimpleNamespace(StageWindow=Mock(side_effect=RuntimeError('OpenGL 3.3 unavailable')))
        notices = []
        with (patch.dict(sys.modules, {'air_violin.stage_window': fake_gl}),
              patch.object(app, 'OpenCVWindow', return_value='compatibility')):
            window, uses_3d = app.open_window('auto', notices.append)
        self.assertEqual(window, 'compatibility')
        self.assertFalse(uses_3d)
        self.assertIn('OpenGL 3.3 unavailable', notices[0])

    def test_string_shortcut_preserves_a_continuous_finger_bow_in_worker(self):
        cal = replace(Calibration(), version=2, fingers=finger_calibration(dwell=0))
        vision, bus = SimpleNamespace(latest=LatestValue()), StateBus()
        worker = ControlWorker(vision, bus, cal, Controls(), 'auto', .65,
                               fingering_mode='fingers', string_override=1)
        def send(x):
            right = articulated(label='Right')
            points = list(right.points)
            points[0] = Point(x, .5)
            vision.latest.put(TrackingFrame(time.perf_counter(),
                             (articulated(1), replace(right, points=tuple(points)))))
        def await_pitch(midi):
            deadline = time.perf_counter() + .12
            while time.perf_counter() < deadline:
                value = bus.read_nowait(AudioState(0))
                if value.midi == midi and value.gate:
                    return value
                time.sleep(.002)
            self.fail(f'Continuous bowed pitch {midi} did not reach the audio bus')
        try:
            worker.configure(cal, 'auto', .65, True, fingering_mode='fingers', string_override=1)
            send(.65)
            time.sleep(.025)
            send(.70)
            await_pitch(64)
            worker.configure(cal, 'auto', .65, True, fingering_mode='fingers', string_override=2)
            # No new camera frame: changing string must preserve the already
            # confirmed finger and current bow motion, instead of reacquiring.
            after = await_pitch(71)
            self.assertTrue(after.gate)
            self.assertEqual(worker.error, '')
        finally:
            worker.close()
