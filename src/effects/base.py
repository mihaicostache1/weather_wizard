from __future__ import annotations

import numpy as np


class ParticleSystem:
    """Shared numpy-vectorized mechanics for particles that fall down the
    screen (rain, snow). Per-particle state lives in flat numpy arrays and
    every update touches the whole array at once - no per-particle python
    loops - which is what keeps hundreds of particles cheap enough to run
    alongside real-time hand tracking."""

    def __init__(self, width: int, height: int, count: int, rng: np.random.Generator) -> None:
        self.width = width
        self.height = height
        self.count = count
        self.rng = rng
        self.x = self.rng.uniform(0, width, count).astype(np.float32)
        self.y = self.rng.uniform(0, height, count).astype(np.float32)

    def _respawn_at_top(self, mask: np.ndarray) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        self.x[mask] = self.rng.uniform(0, self.width, n)
        self.y[mask] = self.rng.uniform(-40, 0, n)

    def update(self, dt: float) -> None:
        raise NotImplementedError

    def draw(self, frame: np.ndarray) -> None:
        raise NotImplementedError
