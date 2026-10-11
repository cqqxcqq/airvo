import base64
import hashlib
import io
import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from air_violin.audio import AudioEngine, StateBus
from air_violin.config import Calibration
from air_violin.demo import KeyboardInstrument
from air_violin.models import verify_model
from air_violin.render import Renderer, physical_hand_caption
from air_violin.vision import convert_result
from tools.fetch_model import fetch_model
from tests.helpers import hand


class IntegrationTests(unittest.TestCase):
    def test_display_uses_calibrated_physical_identity_in_both_mirror_modes(self):
        from dataclasses import replace
        for mirrored in (True, False):
            calibration = replace(Calibration(), mirrored=mirrored, left_label="Right", right_label="Left")
            self.assertEqual(physical_hand_caption("Right", calibration), "LEFT / notes")
            self.assertEqual(physical_hand_caption("Left", calibration), "RIGHT / bow")
        self.assertEqual(physical_hand_caption("Left", Calibration()), "LEFT / notes")
        self.assertEqual(physical_hand_caption("Right", Calibration()), "RIGHT / bow")

    def test_unassigned_hand_does_not_display_a_physical_side(self):
        self.assertEqual(physical_hand_caption("Right", None), "Hand detected")
        self.assertEqual(physical_hand_caption("Left", None), "Hand detected")

    def test_tasks_adapter_preserves_raw_identity_and_capture_timestamp(self):
        source = hand("Right")
        result = SimpleNamespace(hand_landmarks=[source.points],
                                 handedness=[[SimpleNamespace(category_name="Right", score=0.94)]])
        frame = convert_result(result, 42, 17)
        self.assertEqual(frame.timestamp, 42)
        self.assertEqual(frame.inference_ms, 17)
        self.assertEqual(frame.hand("Right").points, source.points)

    def test_letterboxed_render_supports_non_default_camera_aspect(self):
        demo = KeyboardInstrument()
        state = demo.update(1, 0.5)
        for shape in ((480, 640, 3), (360, 640, 3), (640, 480, 3)):
            image = Renderer().draw(np.zeros(shape, np.uint8), state, Calibration(),
                                    None, None, True, "auto", AudioEngine(StateBus()))
            self.assertEqual(image.shape, (710, 920, 3))
            self.assertGreater(np.count_nonzero(image), 10000)

    def test_model_integrity_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.task"
            path.write_bytes(b"test artifact")
            sidecar = path.with_suffix(".task.sha256")
            with self.assertRaises(FileNotFoundError):
                verify_model(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sidecar.write_text(digest)
            with patch("air_violin.models.MODEL_SHA256", digest):
                self.assertEqual(len(verify_model(path)), 64)
                path.write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    verify_model(path)

    def test_downloader_checks_pinned_checksum_and_is_repeatable_offline(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as bundle:
            bundle.writestr("hand_detector.tflite", b"x" * 1_000_000)
            bundle.writestr("hand_landmarks_detector.tflite", b"y" * 1000)
        payload = data.getvalue()
        response = io.BytesIO(payload)
        response.headers = {"x-goog-hash": "md5=" + base64.b64encode(hashlib.md5(payload).digest()).decode()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hand.task"
            digest = hashlib.sha256(payload).hexdigest()
            with (patch("urllib.request.urlopen", return_value=response) as request,
                  patch("tools.fetch_model.MODEL_SHA256", digest),
                  patch("air_violin.models.MODEL_SHA256", digest)):
                digest = fetch_model(path)
                self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
                self.assertEqual(fetch_model(path), digest)
                self.assertEqual(request.call_count, 1)

    def test_download_with_bad_checksum_is_never_installed(self):
        response = io.BytesIO(b"incorrect artifact")
        response.headers = {"x-goog-hash": "md5=" + base64.b64encode(b"0" * 16).decode()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hand.task"
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    fetch_model(path)
            self.assertFalse(path.exists())

    def test_modified_model_and_modified_sidecar_cannot_bypass_the_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hand.task"
            path.write_bytes(b"unexpected model")
            path.with_suffix(".task.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                verify_model(path)

    def test_desktop_loop_demo_keys_and_shutdown_without_hardware(self):
        from air_violin import app
        engine = AudioEngine(StateBus())
        published = []
        engine.bus.publish = published.append
        # Exercise the actual application loop and renderer, substituting only
        # the desktop transport. This does not claim a physical GUI test.
        args = app.parser().parse_args(["--demo", "--silent"])
        with tempfile.TemporaryDirectory() as folder:
            args.config = Path(folder) / "calibration.json"
            with (patch.dict("os.environ", {"DISPLAY": ":test"}),
                  patch.object(app, "AudioEngine", return_value=engine),
                  patch.object(app, "StateBus", return_value=engine.bus),
                  patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "resizeWindow"),
                  patch.object(app.cv2, "imshow") as display,
                  patch.object(app.cv2, "waitKey", side_effect=[ord("h"), ord("3"), ord(" "), ord("0"), ord("q")]),
                  patch.object(app.cv2, "getWindowProperty", return_value=1),
                  patch.object(app.cv2, "destroyAllWindows") as closed,
                  contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(app.run(args), 0)
                self.assertEqual(display.call_count, 5)
                self.assertTrue(any(s.gate and s.midi == 66 for s in published))
                self.assertFalse(published[-1].gate)
                closed.assert_called_once()

    def test_missing_model_and_failed_audio_fall_back_to_visible_demo(self):
        from air_violin import app
        args = app.parser().parse_args([])
        warnings = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            args.config = Path(folder) / "calibration.json"
            args.model = Path(folder) / "missing.task"
            with (patch.dict("os.environ", {"DISPLAY": ":test"}),
                  patch.object(app.AudioEngine, "start", side_effect=RuntimeError("no device")),
                  patch.object(app, "Camera") as camera,
                  patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "resizeWindow"),
                  patch.object(app.cv2, "imshow"), patch.object(app.cv2, "waitKey", return_value=ord("q")),
                  patch.object(app.cv2, "destroyAllWindows"),
                  contextlib.redirect_stderr(warnings)):
                self.assertEqual(app.run(args), 0)
                camera.assert_not_called()
            self.assertIn("Audio unavailable", warnings.getvalue())
            self.assertIn("Using keyboard demo", warnings.getvalue())


if __name__ == "__main__":
    unittest.main()
