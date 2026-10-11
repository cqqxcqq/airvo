"""Original procedural scenes and aspect-preserving local image backgrounds."""

import math
from pathlib import Path
import cv2
import numpy as np

SCENES = ("studio", "nebula", "midnight-rain", "forest-of-light")
LABELS = {"studio": "Studio", "nebula": "Nebula", "midnight-rain": "Midnight Rain",
          "forest-of-light": "Forest of Light", "custom": "Custom image"}


def cover_image(image, width, height):
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(image, (max(width, round(w * scale)), max(height, round(h * scale))))
    x, y = (resized.shape[1] - width) // 2, (resized.shape[0] - height) // 2
    return resized[y:y + height, x:x + width].copy()


class SceneEngine:
    def __init__(self, preset="studio", custom=None, blur=False):
        self.preset, self.blur = preset, blur
        self.custom = None
        self.cache = {}
        self.rng = np.random.default_rng(2026)
        self.stars = self.rng.random((72, 3))
        self.rain = self.rng.random((48, 3))
        self.lights = self.rng.random((24, 3))
        if custom:
            path = Path(custom)
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                raise ValueError("Custom background must be JPG or PNG")
            # imdecode works with Windows Unicode paths where imread may fail.
            self.custom = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.custom is None:
                raise ValueError(f"Cannot decode background: {path}")
            self.preset = "custom"

    @property
    def label(self):
        return LABELS[self.preset]

    def cycle(self):
        choices = SCENES + (("custom",) if self.custom is not None else ())
        self.preset = choices[(choices.index(self.preset) + 1) % len(choices)]

    def _base(self, width, height):
        key = self.preset, width, height, self.blur
        if key in self.cache:
            return self.cache[key]
        if self.preset == "custom":
            result = cover_image(self.custom, width, height)
        else:
            y, x = np.mgrid[0:height, 0:width].astype(np.float32)
            x /= width; y /= height
            if self.preset == "studio":
                spot = np.exp(-((x - 0.46) ** 2 / 0.15 + (y - 0.28) ** 2 / 0.65))
                result = np.stack((15 + spot * 22, 17 + spot * 26, 20 + spot * 32), axis=2)
            elif self.preset == "nebula":
                a = np.exp(-((x - 0.25) ** 2 / 0.04 + (y - 0.65) ** 2 / 0.16))
                b = np.exp(-((x - 0.78) ** 2 / 0.08 + (y - 0.25) ** 2 / 0.10))
                result = np.stack((24 + 55 * a + 34 * b, 12 + 20 * a + 12 * b, 16 + 19 * a + 45 * b), axis=2)
            elif self.preset == "midnight-rain":
                light = np.exp(-((x - 0.65) ** 2 / 0.012 + (y - 0.40) ** 2 / 0.4))
                result = np.stack((22 + light * 28, 18 + light * 22, 14 + light * 15), axis=2)
            else:
                light = np.exp(-((x - 0.50) ** 2 / 0.08 + (y - 0.28) ** 2 / 0.22))
                result = np.stack((20 + light * 19, 29 + light * 36, 17 + light * 20), axis=2)
            result = np.clip(result, 0, 255).astype(np.uint8)
            if self.preset == "forest-of-light":
                rng = np.random.default_rng(89)
                for i in range(16):
                    px = int((i / 15 + rng.uniform(-0.03, 0.03)) * width)
                    thickness = int(width * rng.uniform(0.008, 0.020))
                    tone = (13 + i % 4 * 3, 21 + i % 4 * 3, 11 + i % 4 * 3)
                    cv2.line(result, (px, height), (px + int(width * rng.uniform(-0.1, 0.1)), 0), tone, thickness)
                    for branch in (0.18, 0.33, 0.48):
                        py = int(height * branch)
                        cv2.line(result, (px, py), (px + int(width * rng.uniform(-0.12, 0.12)), py - int(height * 0.10)), tone, max(1, thickness // 3))
            elif self.preset == "studio":
                cv2.line(result, (int(width * .08), int(height * .87)), (int(width * .92), int(height * .87)), (44, 48, 53), 1)
        if self.blur:
            result = cv2.GaussianBlur(result, (21, 21), 0)
        # Bound cache across scene/resolution changes; no history of video frames.
        if len(self.cache) >= 8:
            self.cache.clear()
        self.cache[key] = result
        return result

    def render(self, width, height, now, quality="medium", reduced_motion=False, intensity=0.65):
        result = self._base(width, height).copy()
        intensity = float(np.clip(intensity, 0, 1))
        if intensity <= 0:
            return result
        limit = {"low": 16, "medium": 32, "high": 48}[quality]
        t = 0 if reduced_motion else now
        if self.preset == "nebula":
            for x, y, phase in self.stars[:limit]:
                value = int((100 + 45 * math.sin(t * 0.4 + phase * 7)) * intensity)
                cv2.circle(result, (int(x * width), int(y * height)), 1, (value, value, value), -1, cv2.LINE_AA)
        elif self.preset == "midnight-rain" and not reduced_motion:
            for x, y, speed in self.rain[:limit]:
                px, py = int(x * width), int(((y + t * (0.10 + speed * .08)) % 1) * height)
                tone = int(48 * intensity)
                cv2.line(result, (px, py), (px - 3, py + 12), (tone + 14, tone + 10, tone + 6), 1, cv2.LINE_AA)
        elif self.preset == "forest-of-light":
            for x, y, phase in self.lights[:min(limit, 24)]:
                px = int((x + .008 * math.sin(t * .5 + phase * 6)) * width)
                py = int((y + .012 * math.sin(t * .3 + phase * 7)) * height)
                tone = int(100 * intensity)
                cv2.circle(result, (px, py), 2, (tone // 2, tone + 35, tone + 15), -1, cv2.LINE_AA)
        return result
