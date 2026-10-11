"""Original procedural performance spaces; expensive artwork is cached off audio."""

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


def _noise(width, height, seed):
    """Deterministic, multiscale cloud detail, computed only on a cache miss."""
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), np.float32)
    for size, weight in ((6, .57), (18, .28), (48, .15)):
        coarse = rng.random((size, max(2, round(size * width / height)))).astype(np.float32)
        result += weight * cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(result, 0, 1)


def _add_light(image, strength, color):
    """Add broad light in BGR, with no per-frame full-screen blur."""
    return np.clip(image.astype(np.float32) + strength[:, :, None] * np.asarray(color), 0, 255).astype(np.uint8)


def _spark(image, px, py, radius, color):
    """Add a tiny light without painting dark rings over a bright cloud."""
    height, width = image.shape[:2]
    left, top = max(0, px - radius - 1), max(0, py - radius - 1)
    right, bottom = min(width, px + radius + 2), min(height, py + radius + 2)
    if left >= right or top >= bottom:
        return
    patch = np.zeros((bottom - top, right - left, 3), np.uint8)
    cv2.circle(patch, (px - left, py - top), radius, color, -1, cv2.LINE_AA)
    image[top:bottom, left:right] = cv2.add(image[top:bottom, left:right], patch)


class SceneEngine:
    def __init__(self, preset="studio", custom=None, blur=False):
        if preset not in SCENES and not (preset == "custom" and custom):
            raise ValueError(f"Unknown scene: {preset}")
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

    @staticmethod
    def _studio(width, height, x, y):
        warm = np.exp(-((x - .52) ** 2 / .28 + (y - .45) ** 2 / .35))
        image = np.stack((12 + 14 * warm, 15 + 20 * warm, 19 + 27 * warm), axis=2).astype(np.uint8)
        # Recessed acoustic wall, fine ribs, and a quiet brass reveal give the
        # space an architectural scale without competing with the performer.
        for side in (.075, .925):
            cv2.line(image, (int(side * width), int(.10 * height)),
                     (int(side * width), int(.71 * height)), (39, 48, 58), max(1, width // 600))
        for column in np.linspace(.12, .88, 33):
            px = int(column * width)
            cv2.line(image, (px, int(.17 * height)), (px, int(.69 * height)), (20, 27, 34), 1)
        cv2.line(image, (int(.08 * width), int(.70 * height)),
                 (int(.92 * width), int(.70 * height)), (50, 64, 79), 1)
        # Floorboards narrow toward the back wall. Most details are at edges;
        # the central silhouette remains visually calm after compositing.
        floor = y >= .70
        floor_light = np.exp(-((x - .5) ** 2 / .18)) * ((y - .70) / .30)
        floor_colors = np.stack((14 + 13 * floor_light, 20 + 19 * floor_light,
                                 27 + 27 * floor_light), axis=2)
        image[floor] = floor_colors.astype(np.uint8)[floor]
        for endpoint in np.linspace(-.5, 1.5, 17):
            cv2.line(image, (int(.5 * width + (endpoint - .5) * width * .19), int(.70 * height)),
                     (int(endpoint * width), height - 1), (22, 31, 41), 1, cv2.LINE_AA)
        for row in (.74, .80, .89):
            cv2.line(image, (0, int(row * height)), (width - 1, int(row * height)), (24, 34, 44), 1)
        for px in (.16, .84):
            cv2.ellipse(image, (int(px * width), int(.125 * height)),
                        (max(2, int(width * .021)), max(1, int(height * .007))),
                        0, 0, 360, (86, 120, 157), -1, cv2.LINE_AA)
        left_beam = np.exp(-((x - (.16 + .18 * y)) / (.015 + .15 * y)) ** 2) * np.clip((y - .12) * 1.2, 0, .65)
        right_beam = np.exp(-((x - (.84 - .18 * y)) / (.015 + .15 * y)) ** 2) * np.clip((y - .12) * 1.2, 0, .65)
        image = _add_light(image, (left_beam + right_beam) * .65, (15, 28, 45))
        rim = np.exp(-((x - .5) ** 2 / .21 + (y - .88) ** 2 / .007))
        return _add_light(image, rim * .45, (14, 27, 42))

    @staticmethod
    def _nebula(width, height, x, y):
        noise = _noise(width, height, 740)
        fine = _noise(width, height, 221)
        # An oblique cloud ribbon has a bright blue edge and warm dust core;
        # the negative space in the middle keeps a person easy to read.
        ribbon = y - (.75 - .58 * x + .13 * np.sin(x * 7))
        plume = np.exp(-(ribbon ** 2) / (.014 + .018 * fine))
        cloud = plume * np.clip((noise - .16) * 1.45, 0, 1)
        edge = np.exp(-((ribbon + .095) ** 2) / .009) * noise
        core = np.exp(-((x - .83) ** 2 / .034 + (y - .30) ** 2 / .036)) * (.3 + fine)
        void = np.exp(-((x - .49) ** 2 / .12 + (y - .52) ** 2 / .10))
        dust = np.maximum(0, fine - .57) * plume
        image = np.stack((14 + 69 * cloud + 29 * edge + 12 * core,
                          9 + 34 * cloud + 17 * edge + 51 * core,
                          10 + 19 * cloud + 12 * edge + 81 * core), axis=2)
        image *= (1 - .21 * void[:, :, None])
        image += dust[:, :, None] * np.array((20, 28, 39))
        image = np.clip(image, 0, 255).astype(np.uint8)
        rng = np.random.default_rng(440)
        for sx, sy, strength in rng.random((150, 3)):
            color = int(32 + strength * 66)
            _spark(image, int(sx * width), int(sy * height), 1,
                   (color, int(color * .87), int(color * .76)))
        return image

    @staticmethod
    def _rain_city(width, height, x, y):
        glow = np.exp(-((x - .74) ** 2 / .025 + (y - .48) ** 2 / .20))
        image = np.stack((17 + 29 * glow + 8 * y,
                          13 + 19 * glow + 6 * y, 11 + 11 * glow + 3 * y), axis=2).astype(np.uint8)
        rng = np.random.default_rng(1984)
        horizon = int(.70 * height)
        # Two skyline planes and scattered windows; seeded geometry is stable.
        for layer, tone in ((0, (30, 25, 22)), (1, (15, 16, 18))):
            for i in range(19):
                left = int((i / 18 - .035) * width)
                span = int(width * rng.uniform(.035, .080))
                top = int(height * rng.uniform(.30 + layer * .11, .60))
                cv2.rectangle(image, (left, top), (left + span, horizon), tone, -1)
                if layer == 0:
                    continue
                if i % 4 == 0:
                    cv2.line(image, (left + span // 2, top), (left + span // 2, top - int(.025 * height)), tone, 1)
                for wy in range(top + 7, horizon - 4, max(8, height // 35)):
                    for wx in range(left + 4, left + span - 3, max(6, width // 80)):
                        if rng.random() > .77:
                            color = (47, 67, 96) if rng.random() > .35 else (66, 60, 43)
                            cv2.rectangle(image, (wx, wy), (wx + 2, wy + 3), color, -1)
        floor = y >= .70
        image[floor] = np.stack((13 + 10 * (y - .70), 15 + 9 * (y - .70),
                                 18 + 10 * (y - .70)), axis=2).astype(np.uint8)[floor]
        reflection = np.zeros_like(image)
        for sx, color in ((.13, (45, 67, 100)), (.73, (85, 69, 45)), (.90, (58, 84, 121))):
            for band in range(20):
                py = int(height * (.71 + band * .014))
                half = int(width * (.012 + band * .0009) * rng.uniform(.3, 1))
                offset = int(width * rng.uniform(-.009, .009))
                shade = tuple(int(channel * (1 - band / 28)) for channel in color)
                cv2.line(reflection, (int(sx * width) - half + offset, py),
                         (int(sx * width) + half + offset, py), shade, max(1, height // 240), cv2.LINE_AA)
        reflection = cv2.GaussianBlur(reflection, (5, 5), 0)
        image = cv2.add(image, reflection)
        # Lamps sit near the edges, never over the player's face.
        for sx in (.13, .90):
            px = int(sx * width)
            cv2.line(image, (px, int(.34 * height)), (px, horizon), (35, 38, 42), max(1, width // 240))
            cv2.line(image, (px, int(.34 * height)), (px + int(.022 * width), int(.34 * height)), (35, 38, 42), 2)
            lx, ly = sx + .022, .35
            lamp = np.exp(-((x - lx) ** 2 / .0016 + (y - ly) ** 2 / .003))
            image = _add_light(image, lamp, (35, 56, 82))
            cv2.circle(image, (int(lx * width), int(ly * height)), max(1, width // 210), (104, 160, 207), -1, cv2.LINE_AA)
        mist = np.exp(-((y - .69) ** 2) / .012) * (.45 + _noise(width, height, 31) * .55)
        return _add_light(image, mist, (10, 11, 10))

    @staticmethod
    def _forest(width, height, x, y):
        mist = np.exp(-((x - .57) ** 2 / .12 + (y - .45) ** 2 / .25))
        texture = _noise(width, height, 51)
        image = np.stack((20 + 29 * mist, 30 + 40 * mist,
                          17 + 23 * mist), axis=2).astype(np.uint8)
        rng = np.random.default_rng(89)
        # Distant trunks are thin and desaturated, near trunks broad and dark.
        for layer in range(3):
            count = (23, 15, 6)[layer]
            tone = ((37, 53, 35), (22, 37, 23), (10, 21, 14))[layer]
            for i in range(count):
                pos = i / max(1, count - 1) + rng.uniform(-.035, .035)
                # Leave the clearing open behind the torso and instrument.
                if layer == 2 and .22 < pos < .79:
                    continue
                px = int(pos * width)
                lean = int(width * rng.uniform(-.07, .07))
                thickness = max(1, int(width * (.004 + .004 * layer) * rng.uniform(.8, 1.7)))
                root = int(height * (.73 + layer * .09))
                cv2.line(image, (px, root), (px + lean, -10), tone, thickness, cv2.LINE_AA)
                for branch_y in (.15, .29, .44):
                    py = int(branch_y * height)
                    trunk_x = px + int(lean * (1 - branch_y))
                    direction = -1 if pos < .5 else 1
                    tip = (trunk_x + int(width * rng.uniform(.025, .09)) * direction,
                           py - int(height * rng.uniform(.03, .085)))
                    cv2.line(image, (trunk_x, py), tip, tone, max(1, thickness // 3), cv2.LINE_AA)
        # A winding floor path makes the clearing read as a space with depth.
        path = np.array([(int(width * .52), int(height * .67)),
                         (int(width * .62), int(height * .73)),
                         (int(width * .43), int(height * .86)),
                         (int(width * .32), height - 1),
                         (int(width * .72), height - 1),
                         (int(width * .57), int(height * .86)),
                         (int(width * .67), int(height * .74))], np.int32)
        cv2.fillPoly(image, [path], (30, 45, 32), cv2.LINE_AA)
        # Organic leaf silhouettes at the frame edges, not over the face.
        leaves = np.zeros((height, width), np.uint8)
        for _ in range(150):
            px = int(rng.uniform(-.07, 1.07) * width)
            py = int(rng.uniform(-.04, .18) * height)
            radius = max(2, int(width * rng.uniform(.009, .032)))
            cv2.ellipse(leaves, (px, py), (radius, max(2, radius // 2)), rng.uniform(-70, 70), 0, 360, 255, -1)
        foliage = np.array((11, 25, 14), np.float32)
        alpha = leaves[:, :, None].astype(np.float32) / 255
        image = (image * (1 - alpha) + foliage * alpha).astype(np.uint8)
        shafts = np.zeros_like(x)
        for origin in (.65, .78, .90):
            ray = np.exp(-((x - (origin - .31 * y)) / (.017 + .027 * y)) ** 2)
            shafts += ray * np.clip(y - .11, 0, .8)
        image = _add_light(image, shafts * (.2 + .3 * texture), (23, 35, 30))
        haze = np.exp(-((y - .67) ** 2 / .018)) * mist
        return _add_light(image, haze, (10, 18, 13))

    def _base(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Scene dimensions must be positive")
        key = self.preset, width, height, self.blur
        if key in self.cache:
            return self.cache[key]
        if self.preset == "custom":
            result = cover_image(self.custom, width, height)
        else:
            y, x = np.mgrid[0:height, 0:width].astype(np.float32)
            x /= width
            y /= height
            draw = {"studio": self._studio, "nebula": self._nebula,
                    "midnight-rain": self._rain_city, "forest-of-light": self._forest}[self.preset]
            result = draw(width, height, x, y)
            # Quiet frame edges focus attention on the performer. Computed
            # once rather than an expensive vignette on every video frame.
            vignette = np.clip(1 - .19 * ((x - .5) ** 2 + (y - .5) ** 2), .8, 1)
            result = (result * vignette[:, :, None]).astype(np.uint8)
        if self.blur:
            result = cv2.GaussianBlur(result, (21, 21), 0)
        # Bound cache across scene/resolution changes; no camera frame history.
        if len(self.cache) >= 8:
            self.cache.clear()
        result.setflags(write=False)
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
                value = int((96 + 48 * math.sin(t * .4 + phase * 7)) * intensity)
                px, py = int(x * width), int(y * height)
                radius = 1 if phase < .90 else max(1, width // 360)
                _spark(result, px, py, radius, (min(255, value + 22), value + 8, value))
                if phase > .94 and quality == "high":
                    cv2.line(result, (px - 3, py), (px + 3, py), (value // 2, value // 2, value // 2), 1, cv2.LINE_AA)
        elif self.preset == "midnight-rain" and not reduced_motion:
            for x, y, speed in self.rain[:limit]:
                px = int((x - t * .017) % 1 * width)
                py = int(((y + t * (.16 + speed * .10)) % 1) * height)
                tone = int((25 + speed * 22) * intensity)
                length = max(3, int(height * (.012 + speed * .022)))
                cv2.line(result, (px, py), (px - max(1, width // 250), py + length),
                         (tone + 23, tone + 17, tone + 12), 1, cv2.LINE_AA)
        elif self.preset == "forest-of-light":
            for x, y, phase in self.lights[:min(limit, 24)]:
                px = int((x + .007 * math.sin(t * .5 + phase * 6)) * width)
                py = int((.24 + y * .70 + .010 * math.sin(t * .3 + phase * 7)) * height)
                tone = int((78 + 22 * math.sin(t * .8 + phase * 5)) * intensity)
                # A small, restrained halo and bright core, bounded at 24.
                _spark(result, px, py, 3, (tone // 5, tone // 3, tone // 4))
                _spark(result, px, py, 1, (tone // 2, tone + 55, tone + 29))
        return result
