from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .. import config


@dataclass
class _Bolt:
    """One discharge: the main channel plus its forked side channels, each
    paired with a width scale. Geometry is generated once at strike time
    and then only aged - the shape doesn't move, it just fades and
    flickers."""

    polylines: list[tuple[np.ndarray, float]]
    target: np.ndarray
    age: float = 0.0
    flicker: float = 1.0

    def intensity(self) -> float:
        t = min(self.age / config.LIGHTNING_BOLT_LIFETIME, 1.0)
        return float((1.0 - t) ** config.LIGHTNING_INTENSITY_FALLOFF) * self.flicker


class Lightning:
    """Bolts aren't particles: each is a jagged polyline built by midpoint
    displacement, held for a fraction of a second, then re-struck while the
    gesture persists. Rendered as a crisp core over a separately blurred
    bloom pass rather than a fat line, which is what separates it from a
    flat two-tone stroke."""

    def __init__(self, width: int, height: int, rng: np.random.Generator | None = None) -> None:
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()
        self._bolts: dict[str, _Bolt] = {}
        self._restrike_cooldowns: dict[str, float] = {}
        self.flash = 0.0

        # Preallocated so the render path doesn't churn a full-frame buffer
        # every frame. The bloom buffer is a fraction of the size.
        self._layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._small_size = (
            max(1, width // config.LIGHTNING_BLOOM_DOWNSCALE),
            max(1, height // config.LIGHTNING_BLOOM_DOWNSCALE),
        )

    def _midpoint_displace(
        self, start: np.ndarray, end: np.ndarray, subdivisions: int, displacement: float
    ) -> np.ndarray:
        points = [start, end]
        for _ in range(subdivisions):
            refined = [points[0]]
            for i in range(len(points) - 1):
                p0, p1 = points[i], points[i + 1]
                seg = p1 - p0
                seg_len = float(np.linalg.norm(seg))
                normal = (
                    np.array([-seg[1], seg[0]], dtype=np.float32) / seg_len
                    if seg_len > 1e-3
                    else np.array([1.0, 0.0], dtype=np.float32)
                )
                mid = (p0 + p1) * 0.5 + normal * self.rng.uniform(-displacement, displacement)
                refined.append(mid.astype(np.float32))
                refined.append(p1)
            points = refined
            displacement *= config.LIGHTNING_DECAY
        return np.array(points, dtype=np.float32)

    def _generate_branches(self, main: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Side channels forking off the main one. Each starts partway down
        the trunk and veers off at an angle, covering a fraction of the
        distance still remaining to the strike point."""
        branches: list[tuple[np.ndarray, float]] = []
        if len(main) < 8:
            return branches

        count = int(self.rng.integers(config.LIGHTNING_BRANCH_MIN, config.LIGHTNING_BRANCH_MAX + 1))
        for _ in range(count):
            i = int(self.rng.integers(len(main) // 5, len(main) - 2))
            start = main[i]
            onward = main[-1] - start
            onward_len = float(np.linalg.norm(onward))
            if onward_len < 20.0:
                continue

            heading = onward / onward_len
            angle = np.radians(self.rng.uniform(*config.LIGHTNING_BRANCH_ANGLE_DEG))
            if self.rng.random() < 0.5:
                angle = -angle
            cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            veered = np.array(
                [heading[0] * cos_a - heading[1] * sin_a, heading[0] * sin_a + heading[1] * cos_a],
                dtype=np.float32,
            )

            length = onward_len * self.rng.uniform(*config.LIGHTNING_BRANCH_LENGTH_FRACTION)
            end = (start + veered * length).astype(np.float32)
            pts = self._midpoint_displace(
                start.copy(),
                end,
                config.LIGHTNING_BRANCH_SUBDIVISIONS,
                length * config.LIGHTNING_DISPLACEMENT_FRACTION * 0.5,
            )
            branches.append((pts, config.LIGHTNING_BRANCH_WIDTH_SCALE))
        return branches

    def _generate_bolt(self, target: np.ndarray) -> _Bolt:
        jitter = self.rng.uniform(-1.0, 1.0) * self.width * config.LIGHTNING_TOP_JITTER_FRACTION
        origin = np.array([target[0] + jitter, 0.0], dtype=np.float32)
        length = max(float(np.linalg.norm(target - origin)), 1.0)

        main = self._midpoint_displace(
            origin, target, config.LIGHTNING_SUBDIVISIONS, length * config.LIGHTNING_DISPLACEMENT_FRACTION
        )
        polylines = [(main, 1.0)] + self._generate_branches(main)
        return _Bolt(polylines=polylines, target=target.copy())

    def update(self, targets: dict[str, tuple[float, float]], dt: float) -> None:
        for key in list(self._restrike_cooldowns):
            self._restrike_cooldowns[key] -= dt

        # Age existing bolts. One that's no longer targeted still lives out
        # its remaining lifetime rather than being cut off mid-flash.
        for key in list(self._bolts):
            bolt = self._bolts[key]
            bolt.age += dt
            if bolt.age >= config.LIGHTNING_BOLT_LIFETIME:
                del self._bolts[key]
            else:
                bolt.flicker = float(self.rng.uniform(*config.LIGHTNING_FLICKER_RANGE))

        for key, target in targets.items():
            if key in self._bolts or self._restrike_cooldowns.get(key, 0.0) > 0.0:
                continue
            self._bolts[key] = self._generate_bolt(np.array(target, dtype=np.float32))
            self._restrike_cooldowns[key] = config.LIGHTNING_BOLT_LIFETIME + self.rng.uniform(
                *config.LIGHTNING_RESTRIKE_DELAY
            )
            self.flash = 1.0

        for key in [k for k in self._restrike_cooldowns if k not in targets and k not in self._bolts]:
            del self._restrike_cooldowns[key]

        self.flash *= config.LIGHTNING_FLASH_DECAY

    def _draw_tapered(
        self, layer: np.ndarray, points: np.ndarray, width_top: float, width_tip: float, color: tuple
    ) -> None:
        """cv2.polylines can't vary stroke width along a path, so the channel
        is drawn as consecutive slices at interpolated widths - thick at the
        cloud end, thin where it lands."""
        steps = config.LIGHTNING_TAPER_STEPS
        bounds = np.linspace(0, len(points) - 1, steps + 1).astype(int)
        for j in range(steps):
            segment = points[bounds[j] : bounds[j + 1] + 1]
            if len(segment) < 2:
                continue
            t = j / max(steps - 1, 1)
            thickness = max(1, int(round(width_top * (1.0 - t) + width_tip * t)))
            cv2.polylines(layer, [segment.astype(np.int32)], False, color, thickness, cv2.LINE_AA)

    def draw(self, frame: np.ndarray) -> None:
        if not self._bolts and self.flash <= 0.01:
            return

        layer = self._layer
        drew_any = False

        if self._bolts:
            layer.fill(0)
            for bolt in self._bolts.values():
                intensity = bolt.intensity()
                if intensity <= 0.02:
                    continue
                drew_any = True
                color = tuple(float(c) * intensity for c in config.LIGHTNING_CORE_COLOR)
                for points, width_scale in bolt.polylines:
                    self._draw_tapered(
                        layer,
                        points,
                        config.LIGHTNING_WIDTH_TOP * width_scale,
                        config.LIGHTNING_WIDTH_TIP * width_scale,
                        color,
                    )
                impact_radius = int(config.LIGHTNING_IMPACT_RADIUS * intensity)
                if impact_radius > 0:
                    cv2.circle(layer, tuple(bolt.target.astype(int)), impact_radius, color, -1, cv2.LINE_AA)

        if drew_any:
            # Bloom: blur at reduced resolution, which is both cheaper than a
            # full-res blur and smoother once the upscale interpolates it.
            small = cv2.resize(layer, self._small_size, interpolation=cv2.INTER_LINEAR)
            cv2.GaussianBlur(small, (0, 0), config.LIGHTNING_BLOOM_SIGMA, dst=small)
            tinted = np.clip(
                small.astype(np.float32) * np.array(config.LIGHTNING_GLOW_TINT, dtype=np.float32), 0, 255
            ).astype(np.uint8)
            glow = cv2.resize(tinted, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(frame, 1.0, glow, config.LIGHTNING_BLOOM_GAIN, 0.0, dst=frame)
            cv2.add(frame, layer, dst=frame)

        if self.flash > 0.01:
            alpha = min(self.flash, 1.0) * config.LIGHTNING_FLASH_MAX_ALPHA
            white = np.full_like(frame, 255)
            cv2.addWeighted(frame, 1.0 - alpha, white, alpha, 0.0, dst=frame)
