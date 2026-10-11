"""MediaPipe Tasks VIDEO API on a worker, processing only the newest camera frame.

VIDEO is synchronous within this worker, avoiding LIVE_STREAM callback races.
Captured timestamps survive inference so delayed results cannot sustain a note.
"""

from pathlib import Path
import threading
import time
import cv2
from .capture import Camera, LatestValue
from .types import Hand, Point, TrackingFrame


def convert_result(result, timestamp: float, inference_ms: float) -> TrackingFrame:
    hands = []
    worlds = getattr(result, "hand_world_landmarks", ())
    for index, (landmarks, categories) in enumerate(zip(result.hand_landmarks, result.handedness)):
        if not categories or len(landmarks) != 21:
            continue
        category = max(categories, key=lambda c: c.score)
        world = worlds[index] if index < len(worlds) else None
        hands.append(Hand(category.category_name, float(category.score),
                          tuple(Point(p.x, p.y, p.z) for p in landmarks),
                          tuple(Point(p.x, p.y, p.z) for p in world)
                          if world is not None and len(world) == 21 else None))
    return TrackingFrame(timestamp, tuple(hands), inference_ms)


class HandTracker:
    def __init__(self, model: Path, confidence: float = 0.65):
        if not model.is_file():
            raise FileNotFoundError(f"Hand model missing: {model}. Run python tools/fetch_model.py")
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def detect(self, image, timestamp: float) -> TrackingFrame:
        height, width = image.shape[:2]
        if width > 640:
            image = cv2.resize(image, (640, round(height * 640 / width)))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        media_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        # Millisecond truncation can collide at high frame rates; enforce ordering.
        timestamp_ms = max(self.last_timestamp_ms + 1, int(timestamp * 1000))
        self.last_timestamp_ms = timestamp_ms
        started = time.perf_counter()
        result = self.landmarker.detect_for_video(media_image, timestamp_ms)
        return convert_result(result, timestamp, (time.perf_counter() - started) * 1000)

    def close(self) -> None:
        self.landmarker.close()


class VisionWorker:
    def __init__(self, camera: Camera, model: Path, confidence: float):
        self.camera = camera
        self.model = model
        self.confidence = confidence
        self.latest = LatestValue()
        self.stop = threading.Event()
        self.error = ""
        self.ready = False
        self.thread = threading.Thread(target=self._run, name="hand-inference", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        tracker = None
        try:
            tracker = HandTracker(self.model, self.confidence)
            self.ready = True
            previous = -1
            while not self.stop.is_set():
                frame = self.camera.latest.get()
                if frame is None or frame.sequence == previous:
                    self.stop.wait(0.002)
                    continue
                previous = frame.sequence
                self.latest.put(tracker.detect(frame.image, frame.timestamp))
        except Exception as exc:
            self.error = f"Hand tracking unavailable: {type(exc).__name__}: {exc}"
        finally:
            self.ready = False
            if tracker is not None:
                tracker.close()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3.0)
