from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .base import Fader


class Tornado:
    """A vortex emanating along an axis running out of the palm toward the
    camera, drawn as continuous spiral arms rather than a swarm of points.

    An arm is a curve that can be evaluated directly: sampling `z` - the
    distance along the axis, 0 at the palm and 1 at full extension - gives
    an angle `base + phase + direction * z * twist` at radius `R(z)`. That
    traces the exact spiral a particle cloud could only imply, which is what
    makes the shape legible, and it costs a handful of strokes instead of
    hundreds of polygons.

    The `direction * z` shear is also the only reason spin direction is
    visible: a rotationally symmetric vortex looks identical whichever way
    it turns, so the curl of the arms has to carry it.

    Triggered by a circular hand gesture and living on a timer, like Wind
    rather than the held finger-count modes, so it layers over whatever
    rain/snow/lightning is running."""

    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None) -> None:
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()

        self.center = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        self.anchor_side: str | None = None
        self.spin_direction = 1

        self._phase = 0.0
        self._time_left = 0.0
        self._fader = Fader(config.TORNADO_FADE_RATE)
        self._layer = np.zeros((height, width, 3), dtype=np.uint8)

        self._inner_radius = width * config.TORNADO_INNER_RADIUS_FRACTION
        self._outer_radius = width * config.TORNADO_OUTER_RADIUS_FRACTION
        self._z = np.linspace(0.0, 1.0, config.TORNADO_ARM_SEGMENTS, dtype=np.float32)
        self._arm_bases = np.arange(config.TORNADO_ARMS, dtype=np.float32) * (2 * np.pi / config.TORNADO_ARMS)
        self._radii = self._radius_at(self._z)

    @property
    def intensity(self) -> float:
        return self._fader.value

    def trigger(self, direction: int, side: str | None, center: np.ndarray) -> None:
        self.spin_direction = 1 if direction >= 0 else -1
        self.anchor_side = side
        self.center = np.asarray(center, dtype=np.float32)
        self._time_left = config.TORNADO_LIFETIME

    def _radius_at(self, z: np.ndarray) -> np.ndarray:
        return self._inner_radius + (self._outer_radius - self._inner_radius) * (
            z**config.TORNADO_FLARE_EXPONENT
        )

    def update(self, dt: float, center: np.ndarray | None = None) -> None:
        self._time_left = max(0.0, self._time_left - dt)
        intensity = self._fader.update(1.0 if self._time_left > 0.0 else 0.0, dt)
        if intensity <= 0.01:
            return

        # Track the hand while it's visible; hold the last position if it
        # drops out, so the vortex doesn't snap to a stale coordinate.
        if center is not None:
            self.center = np.asarray(center, dtype=np.float32)

        self._phase += self.spin_direction * config.TORNADO_SPIN_RATE * dt

    def _arm_points(self, base: float) -> np.ndarray:
        theta = base + self._phase + self.spin_direction * self._z * config.TORNADO_SPIRAL_TWIST
        x = self.center[0] + self._radii * np.cos(theta)
        y = self.center[1] + self._radii * np.sin(theta) * config.TORNADO_TILT_SQUASH
        return np.stack([x, y], axis=1)

    def _brightness(self, z: float) -> float:
        """Ramp up as an arm leaves the palm and dissolve at its outer end,
        so the strokes don't start or stop abruptly."""
        rising = min(z / config.TORNADO_FADE_IN_Z, 1.0) if config.TORNADO_FADE_IN_Z > 0 else 1.0
        tail = 1.0 - max(0.0, z - config.TORNADO_FADE_OUT_Z) / max(1.0 - config.TORNADO_FADE_OUT_Z, 1e-6)
        return float(max(0.0, min(rising, tail)))

    def draw(self, frame: np.ndarray) -> None:
        intensity = self._fader.value
        if intensity <= 0.01:
            return

        layer = self._layer
        layer.fill(0)

        # polylines can't vary stroke width or color along a path, so each
        # arm is drawn as consecutive slices - thin and faint at the palm,
        # thicker and brighter as it flares outward.
        slices = config.TORNADO_ARM_SLICES
        bounds = np.linspace(0, len(self._z) - 1, slices + 1).astype(int)

        for base in self._arm_bases:
            points = self._arm_points(float(base))
            for j in range(slices):
                segment = points[bounds[j] : bounds[j + 1] + 1]
                if len(segment) < 2:
                    continue
                z_mid = float((self._z[bounds[j]] + self._z[bounds[j + 1]]) * 0.5)
                shade = self._brightness(z_mid)
                if shade <= 0.01:
                    continue
                width = config.TORNADO_ARM_WIDTH_INNER + (
                    config.TORNADO_ARM_WIDTH_OUTER - config.TORNADO_ARM_WIDTH_INNER
                ) * z_mid
                color = tuple(float(c) * shade for c in config.TORNADO_COLOR)
                cv2.polylines(
                    layer, [segment.astype(np.int32)], False, color, max(1, int(round(width))), cv2.LINE_AA
                )

        cv2.addWeighted(frame, 1.0, layer, intensity, 0.0, dst=frame)
