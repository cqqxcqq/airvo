"""Musical event effects with fixed caps; reduced motion stops trails/particles."""

from collections import deque
import math
import cv2


class MusicalEffects:
    def __init__(self):
        self.trail = deque(maxlen=36)
        self.particles = deque(maxlen=64)
        self.echoes = deque(maxlen=4)
        self.previous_gate = False
        self.previous_midi = None
        self.last_particle = -math.inf
        self.last_tracking = -math.inf
        self.serial = 0

    def draw(self, image, state, pose, now, color, note, quality="medium", reduced_motion=False, intensity=.65):
        active = state.audio.gate
        if active and (not self.previous_gate or self.previous_midi != state.audio.midi):
            self.echoes.append((now, note, color))
        self.previous_gate, self.previous_midi = active, state.audio.midi
        if reduced_motion or intensity <= 0:
            self.trail.clear(); self.particles.clear()
            return
        if pose and active and pose.timestamp > self.last_tracking:
            self.trail.append((now, pose.grip, color))
            self.last_tracking = pose.timestamp
        if pose and active and now - self.last_particle > {"low": .18, "medium": .10, "high": .065}[quality]:
            self.serial += 1
            self.particles.append((now, pose.grip, self.serial, color))
            self.last_particle = now
        while self.trail and now - self.trail[0][0] > .40:
            self.trail.popleft()
        while self.particles and now - self.particles[0][0] > 1.2:
            self.particles.popleft()
        trails = list(self.trail)[-{"low": 8, "medium": 18, "high": 36}[quality]:]
        for a, b in zip(trails, trails[1:]):
            fade = max(0, 1 - (now - b[0]) / .4) * intensity
            tone = tuple(int(c * fade) for c in b[2])
            cv2.line(image, tuple(map(int, a[1])), tuple(map(int, b[1])), tone, 2, cv2.LINE_AA)
        for born, position, serial, tone in list(self.particles)[-{"low": 8, "medium": 24, "high": 64}[quality]:]:
            age = now - born
            x = int(position[0] + math.sin(serial * 2.4) * age * 24)
            y = int(position[1] - age * (16 + serial % 5 * 4))
            color_faded = tuple(int(c * max(0, 1 - age / 1.2) * intensity) for c in tone)
            cv2.circle(image, (x, y), 2, color_faded, -1, cv2.LINE_AA)
        for index, (born, content, tone) in enumerate(self.echoes):
            age = now - born
            if 0 <= age < .8:
                color_faded = tuple(int(c * (1 - age / .8) * intensity) for c in tone)
                cv2.putText(image, content, (image.shape[1] - 96, 55 + index * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, color_faded, 1, cv2.LINE_AA)
