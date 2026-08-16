from __future__ import annotations

import cv2
import numpy as np

from .. import config


class Lightning:
    """Bolts aren't particles: each one is a jagged polyline computed once
    via midpoint displacement, then held and redrawn for a fixed number of
    frames before its fingertip - if still raised - earns a fresh strike.
    This intentionally doesn't share ParticleSystem: that base is for large
    uniform arrays of identical points, this is a handful of individually
    shaped polylines plus a full-frame flash."""

    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None) -> None:
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()
        self._bolts: dict[str, tuple[np.ndarray, int]] = {}
        self.flash = 0.0

    def _generate_bolt(self, target: np.ndarray) -> np.ndarray:
        top = np.array([target[0], 0.0], dtype=np.float32)
        points = [top, target]
        total_length = max(float(np.linalg.norm(target - top)), 1.0)
        displacement = config.LIGHTNING_DISPLACEMENT_FRACTION * total_length

        for _ in range(config.LIGHTNING_SUBDIVISIONS):
            new_points = [points[0]]
            for i in range(len(points) - 1):
                p0, p1 = points[i], points[i + 1]
                seg = p1 - p0
                seg_len = np.linalg.norm(seg)
                normal = (
                    np.array([-seg[1], seg[0]], dtype=np.float32) / seg_len
                    if seg_len > 1e-3
                    else np.array([1.0, 0.0], dtype=np.float32)
                )
                mid = (p0 + p1) * 0.5 + normal * self.rng.uniform(-displacement, displacement)
                new_points.append(mid)
                new_points.append(p1)
            points = new_points
            displacement *= config.LIGHTNING_DECAY

        return np.array(points, dtype=np.float32)

    def update(self, targets: dict[str, tuple[float, float]]) -> None:
        # Age every active bolt by one frame. Anything not re-targeted this
        # frame still counts down its remaining lifetime instead of being
        # cut off the instant the gesture changes, giving it a brief tail
        # fade rather than a hard pop.
        for key in list(self._bolts.keys()):
            points, frames_left = self._bolts[key]
            frames_left -= 1
            if frames_left <= 0:
                del self._bolts[key]
            else:
                self._bolts[key] = (points, frames_left)

        # A raised fingertip without a live bolt gets a fresh one - this is
        # what makes a held gesture re-strike on an interval rather than
        # showing one static bolt forever.
        for key, target in targets.items():
            if key not in self._bolts:
                bolt = self._generate_bolt(np.array(target, dtype=np.float32))
                self._bolts[key] = (bolt, config.LIGHTNING_BOLT_LIFETIME_FRAMES)
                self.flash = 1.0

        self.flash *= config.LIGHTNING_FLASH_DECAY

    def draw(self, frame: np.ndarray) -> None:
        if self._bolts:
            glow = np.zeros_like(frame)
            for points, _ in self._bolts.values():
                pts = points.astype(np.int32)
                cv2.polylines(
                    glow, [pts], False, config.LIGHTNING_GLOW_COLOR, config.LIGHTNING_GLOW_THICKNESS, cv2.LINE_AA
                )
                cv2.polylines(
                    glow, [pts], False, config.LIGHTNING_CORE_COLOR, config.LIGHTNING_CORE_THICKNESS, cv2.LINE_AA
                )
            cv2.add(frame, glow, dst=frame)

        if self.flash > 0.01:
            alpha = min(self.flash, 1.0) * config.LIGHTNING_FLASH_MAX_ALPHA
            white = np.full_like(frame, 255)
            cv2.addWeighted(frame, 1.0 - alpha, white, alpha, 0.0, dst=frame)
