from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .base import ParticleSystem


class Rain(ParticleSystem):
    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None) -> None:
        super().__init__(width, height, config.RAIN_COUNT, rng or np.random.default_rng())
        self.vel_y = self.rng.uniform(config.RAIN_MIN_SPEED, config.RAIN_MAX_SPEED, self.count).astype(np.float32)
        self.length = self.rng.uniform(config.RAIN_MIN_LENGTH, config.RAIN_MAX_LENGTH, self.count).astype(np.float32)

    def update(self, dt: float) -> None:
        self.y += self.vel_y * dt

        offscreen = self.y > self.height
        self._respawn_at_top(offscreen)
        n = int(offscreen.sum())
        if n:
            self.vel_y[offscreen] = self.rng.uniform(config.RAIN_MIN_SPEED, config.RAIN_MAX_SPEED, n)
            self.length[offscreen] = self.rng.uniform(config.RAIN_MIN_LENGTH, config.RAIN_MAX_LENGTH, n)

    def draw(self, frame: np.ndarray) -> None:
        top = np.stack([self.x, self.y], axis=1)
        bottom = np.stack([self.x, self.y + self.length], axis=1)
        segments = np.stack([top, bottom], axis=1).astype(np.int32)
        cv2.polylines(frame, segments, False, config.RAIN_COLOR, config.RAIN_THICKNESS, cv2.LINE_AA)
