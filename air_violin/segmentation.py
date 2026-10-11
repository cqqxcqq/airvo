"""Optional, rate-limited MediaPipe Tasks person segmentation; never audio work."""

from dataclasses import dataclass
import threading
import time
import cv2
import numpy as np
from .capture import LatestValue
from .models import verify_segmenter

QUALITY = {"low": (256, 8), "medium": (384, 12), "high": (512, 15)}


@dataclass(frozen=True)
class PersonMask:
    timestamp: float
    sequence: int
    alpha: np.ndarray
    inference_ms: float


class PersonSegmenter:
    def __init__(self, model):
        verify_segmenter(model)
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False)
        self.task = vision.ImageSegmenter.create_from_options(options)
        self.last_ms = -1
        self.previous = None

    def detect(self, frame, width=384):
        h, w = frame.image.shape[:2]
        width = min(width, w)
        small = cv2.resize(frame.image, (width, max(1, round(h * width / w))))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        milliseconds = max(self.last_ms + 1, int(frame.timestamp * 1000))
        self.last_ms = milliseconds
        start = time.perf_counter()
        result = self.task.segment_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), milliseconds)
        # This verified landscape model has ONE confidence output, labelled
        # 'selfie': it is person probability, not background probability.
        if not result.confidence_masks or len(result.confidence_masks) != 1:
            raise ValueError("Unexpected person-model output; expected one selfie confidence mask")
        alpha = np.array(result.confidence_masks[0].numpy_view(), dtype=np.float32, copy=True).squeeze(axis=2)
        if not np.isfinite(alpha).all():
            raise ValueError("Nonfinite segmentation mask")
        alpha = cv2.GaussianBlur(np.clip(alpha, 0, 1), (3, 3), 0)
        previous = self.previous
        if previous and previous.alpha.shape == alpha.shape and 0 < frame.timestamp - previous.timestamp < 0.2:
            # Blend only stable pixels: moving hands should not leave old masks.
            stable = np.abs(alpha - previous.alpha) < 0.10
            alpha[stable] = 0.8 * alpha[stable] + 0.2 * previous.alpha[stable]
        alpha.setflags(write=False)
        mask = PersonMask(frame.timestamp, frame.sequence, alpha, (time.perf_counter() - start) * 1000)
        self.previous = mask
        return mask

    def close(self):
        self.task.close()


class SegmentationWorker:
    def __init__(self, camera, model):
        self.camera, self.model = camera, model
        self.latest = LatestValue()
        self.settings = LatestValue()
        self.settings.put((False, "medium", False))
        self.stop = threading.Event()
        self.error, self.ready, self.throttled = "", False, False
        self.thread = threading.Thread(target=self._run, name="person-segmentation", daemon=True)
        self.thread.start()

    def configure(self, enabled, quality, under_pressure=False):
        self.settings.put((enabled, quality, under_pressure))

    def _run(self):
        segmenter = None
        try:
            segmenter = PersonSegmenter(self.model)
            self.ready = True
            previous, next_due = -1, 0.0
            while not self.stop.is_set():
                enabled, quality, pressure = self.settings.get()
                self.throttled = pressure
                width, fps = QUALITY["low" if pressure else quality]
                frame = self.camera.latest.get()
                now = time.perf_counter()
                if not enabled or frame is None or frame.sequence == previous or now < next_due:
                    self.stop.wait(0.010)
                    continue
                previous = frame.sequence
                mask = segmenter.detect(frame, width)
                self.latest.put(mask)
                # Never catch up missed masks; bound duty cycle as well as FPS.
                next_due = now + max(1 / fps, mask.inference_ms / 1000 * 4)
        except Exception as exc:
            self.error = f"Background unavailable: {exc}. Camera video remains usable."
        finally:
            self.ready = False
            if segmenter:
                segmenter.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def composite_person(image, background, mask, timestamp, tracking=None, confidence=0.65):
    """Return (frame, used_mask). No old mask may hide the current camera feed."""
    if (mask is None or not 0 <= timestamp - mask.timestamp <= 0.25
            or mask.alpha.ndim != 2 or mask.alpha.size == 0 or not np.isfinite(mask.alpha).all()):
        return image.copy(), False
    h, w = image.shape[:2]
    alpha = cv2.resize(mask.alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip((alpha - 0.10) / 0.80, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    # Person segmentation can clip fast fingers. Preserve narrow fresh landmark
    # paths, not rectangles or stale hand locations. This is a conservative
    # heuristic, not a guarantee of perfect hair/hand edges.
    if tracking and 0 <= timestamp - tracking.timestamp <= 0.12:
        for label in ("Left", "Right"):
            hand = tracking.hand(label, confidence)
            if hand is None:
                continue
            points = np.array([(round(p.x * w), round(p.y * h)) for p in hand.points], np.int32)
            palm = cv2.convexHull(points[[0, 5, 9, 13, 17]])
            cv2.fillConvexPoly(alpha, palm, 1.0)
            for chain in ((0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
                cv2.polylines(alpha, [points[list(chain)]], False, 1.0, max(3, round(w * 0.008)), cv2.LINE_AA)
    if background.shape != image.shape:
        raise ValueError("Background must match camera dimensions")
    result = image.astype(np.float32) * alpha[..., None] + background.astype(np.float32) * (1 - alpha[..., None])
    return np.clip(result, 0, 255).astype(np.uint8), True
