from __future__ import annotations

import cv2
import numpy as np

from .. import config


class Wind:
    """A one-shot gust: horizontal streaks that sweep left-to-right across
    the screen and drop out, fired by `trigger()` rather than tied to a
    held/sustained mode. Deliberately not a ParticleSystem subclass - that
    base assumes particles wrap and respawn forever, while a gust plays
    once and ends."""

    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None) -> None:
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()
        self._empty = np.zeros(0, dtype=np.float32)
        self.x = self._empty
        self.y = self._empty
        self.vel_x = self._empty
        self.length = self._empty
        self.age = self._empty

    @property
    def active(self) -> bool:
        return self.x.size > 0

    def trigger(self, direction: int = 1) -> None:
        # New streaks are appended rather than replacing whatever's still
        # playing, so a second swipe (even in the opposite direction)
        # layers a new gust instead of cutting the first one off.
        n = config.WIND_STREAK_COUNT
        speed = self.rng.uniform(config.WIND_MIN_SPEED, config.WIND_MAX_SPEED, n).astype(np.float32)
        new_vel_x = speed if direction >= 0 else -speed
        if direction >= 0:
            new_x = self.rng.uniform(-self.width * 0.3, 0.0, n).astype(np.float32)
        else:
            new_x = self.rng.uniform(self.width, self.width * 1.3, n).astype(np.float32)
        new_y = self.rng.uniform(0, self.height, n).astype(np.float32)
        new_length = self.rng.uniform(config.WIND_MIN_LENGTH, config.WIND_MAX_LENGTH, n).astype(np.float32)
        new_age = np.zeros(n, dtype=np.float32)

        self.x = np.concatenate([self.x, new_x])
        self.y = np.concatenate([self.y, new_y])
        self.vel_x = np.concatenate([self.vel_x, new_vel_x])
        self.length = np.concatenate([self.length, new_length])
        self.age = np.concatenate([self.age, new_age])

    def update(self, dt: float) -> None:
        if not self.active:
            return

        self.x += self.vel_x * dt
        self.age += dt

        alive = (self.age < config.WIND_STREAK_LIFETIME) & (self.x > -self.width) & (self.x < self.width * 2)
        if not np.any(alive):
            self.x = self.y = self.vel_x = self.length = self.age = self._empty
            return
        self.x, self.y, self.vel_x, self.length, self.age = (
            self.x[alive],
            self.y[alive],
            self.vel_x[alive],
            self.length[alive],
            self.age[alive],
        )

    def draw(self, frame: np.ndarray) -> None:
        if not self.active:
            return
        # The tail trails behind the direction of travel, on either side
        # depending on the sign of each streak's own velocity.
        tail_x = self.x - np.sign(self.vel_x) * self.length
        front = np.stack([self.x, self.y], axis=1)
        tail = np.stack([tail_x, self.y], axis=1)
        segments = np.stack([tail, front], axis=1).astype(np.int32)
        cv2.polylines(frame, segments, False, config.WIND_COLOR, config.WIND_THICKNESS, cv2.LINE_AA)
