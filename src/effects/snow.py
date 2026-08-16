from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .base import ParticleSystem

_OCTAGON_ANGLES = np.linspace(0, 2 * np.pi, 8, endpoint=False)
_UNIT_OCTAGON = np.stack([np.cos(_OCTAGON_ANGLES), np.sin(_OCTAGON_ANGLES)], axis=1).astype(np.float32)


class Snow(ParticleSystem):
    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None) -> None:
        super().__init__(width, height, config.SNOW_COUNT, rng or np.random.default_rng())
        self.base_x = self.x.copy()
        self.phase = self.rng.uniform(0.0, 2 * np.pi, self.count).astype(np.float32)
        self.sway_amp = self.rng.uniform(config.SNOW_MIN_SWAY, config.SNOW_MAX_SWAY, self.count).astype(np.float32)
        self.sway_freq = self.rng.uniform(
            config.SNOW_MIN_SWAY_FREQ, config.SNOW_MAX_SWAY_FREQ, self.count
        ).astype(np.float32)
        self.radius, self.vel_y = self._depth_attrs(self.count)
        self.t = 0.0

    def _depth_attrs(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        # One depth value per flake drives both size and speed together, so
        # bigger flakes fall faster - the classic near/far parallax cue.
        depth = self.rng.uniform(0.0, 1.0, n).astype(np.float32)
        radius = config.SNOW_MIN_RADIUS + depth * (config.SNOW_MAX_RADIUS - config.SNOW_MIN_RADIUS)
        vel_y = config.SNOW_MIN_SPEED + depth * (config.SNOW_MAX_SPEED - config.SNOW_MIN_SPEED)
        return radius.astype(np.float32), vel_y.astype(np.float32)

    def update(self, dt: float) -> None:
        self.t += dt
        self.y += self.vel_y * dt
        self.x = self.base_x + self.sway_amp * np.sin(self.sway_freq * self.t + self.phase)

        offscreen = self.y > self.height
        n = int(offscreen.sum())
        if n:
            self.base_x[offscreen] = self.rng.uniform(0, self.width, n)
            self.y[offscreen] = self.rng.uniform(-40, 0, n)
            self.phase[offscreen] = self.rng.uniform(0.0, 2 * np.pi, n)
            self.sway_amp[offscreen] = self.rng.uniform(config.SNOW_MIN_SWAY, config.SNOW_MAX_SWAY, n)
            self.sway_freq[offscreen] = self.rng.uniform(config.SNOW_MIN_SWAY_FREQ, config.SNOW_MAX_SWAY_FREQ, n)
            radius, vel_y = self._depth_attrs(n)
            self.radius[offscreen] = radius
            self.vel_y[offscreen] = vel_y

    def draw(self, frame: np.ndarray) -> None:
        centers = np.stack([self.x, self.y], axis=1)[:, None, :]
        offsets = _UNIT_OCTAGON[None, :, :] * self.radius[:, None, None]
        polys = (centers + offsets).astype(np.int32)
        cv2.fillPoly(frame, polys, config.SNOW_COLOR, lineType=cv2.LINE_AA)
