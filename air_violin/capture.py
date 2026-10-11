"""Latest-frame camera capture; a slow consumer never builds a frame queue."""

from dataclasses import dataclass
import threading
import time
import cv2
import numpy as np


class LatestValue:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None

    def put(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


@dataclass(frozen=True)
class CameraFrame:
    sequence: int
    timestamp: float
    image: np.ndarray


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 mirrored: bool = True, backend: str = "auto"):
        api = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[backend]
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Cannot open webcam {index}. Check camera permission / --camera / --backend.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Some backends ignore this hint.
        self.mirrored = mirrored
        self.latest = LatestValue()
        self.stop = threading.Event()
        self.error = ""
        self.thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        sequence = 0
        failures = 0
        try:
            while not self.stop.is_set():
                ok, image = self.cap.read()
                timestamp = time.perf_counter()
                if not ok:
                    failures += 1
                    if failures >= 10:
                        self.error = "Webcam disconnected or not delivering frames; press K for keyboard demo."
                        break
                    self.stop.wait(0.03)
                    continue
                failures = 0
                if self.mirrored:
                    image = cv2.flip(image, 1)
                self.latest.put(CameraFrame(sequence, timestamp, image))
                sequence += 1
        except Exception as exc:
            self.error = f"Camera error: {exc}"
        finally:
            self.cap.release()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            # A disconnected backend can block read(); release its OS handle.
            self.cap.release()
            self.thread.join(timeout=1.0)
