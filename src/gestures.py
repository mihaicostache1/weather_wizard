from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from . import config
from .hand_tracker import Hand

# (mcp_idx, pip_idx, tip_idx) per counted finger. The thumb is deliberately
# excluded: its extension axis runs roughly parallel to the palm rather than
# bending at the knuckle the way the other four fingers do, so the curl-angle
# test below doesn't apply to it, and the gesture contract (1/2/3 fingers)
# never needs it.
_COUNTED_FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


class Mode(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    LIGHTNING = "lightning"


_MODE_BY_COUNT = {
    0: Mode.CLEAR,
    1: Mode.RAIN,
    2: Mode.SNOW,
    3: Mode.LIGHTNING,
}


class ActiveHandSelector:
    """Tracks which hand is in control across frames. `dominant_side` wins
    whenever a fresh pick is needed (start-up, or the previously active hand
    has left the frame). Once a hand is active it stays active - regardless
    of per-frame detection score - until it closes into a fist (0 extended
    fingers) while the other hand has at least one finger raised to take
    over. Requiring the other hand to be doing something avoids the active
    hand flapping back and forth when both are idly closed."""

    def __init__(self, dominant_side: str = "Right") -> None:
        self._dominant_side = dominant_side
        self._active_side: str | None = None

    def update(self, hands: list[Hand]) -> Hand | None:
        by_side = {hand.handedness: hand for hand in hands}

        if self._active_side not in by_side:
            self._active_side = self._dominant_side if self._dominant_side in by_side else next(iter(by_side), None)
        else:
            active_hand = by_side[self._active_side]
            if not extended_fingers(active_hand):
                other_side = "Left" if self._active_side == "Right" else "Right"
                other_hand = by_side.get(other_side)
                if other_hand is not None and extended_fingers(other_hand):
                    self._active_side = other_side

        return by_side.get(self._active_side)


def _curl_angle_deg(points: np.ndarray, mcp_idx: int, pip_idx: int, tip_idx: int) -> float:
    """Angle, in degrees, between the MCP->PIP and PIP->TIP segments. Near 0
    means the finger is straight; large means it's bent back toward the
    palm. Purely local to one finger, so it holds regardless of how the
    whole hand is rotated or how far it is from the camera."""
    v1 = points[pip_idx] - points[mcp_idx]
    v2 = points[tip_idx] - points[pip_idx]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def extended_fingers(hand: Hand) -> list[str]:
    points = hand.landmarks_px
    extended = []
    for name, (mcp_idx, pip_idx, tip_idx) in _COUNTED_FINGERS.items():
        if _curl_angle_deg(points, mcp_idx, pip_idx, tip_idx) < config.EXTENDED_FINGER_ANGLE_DEG:
            extended.append(name)
    return extended


def resolve_finger_count(hands: list[Hand], active_hand: Hand | None) -> int:
    """The active hand's finger count only counts when it's the only hand
    gesturing. If any other visible hand also has fingers extended at the
    same time, the input is ambiguous and treated as no gesture at all."""
    if active_hand is None:
        return 0
    other_hand_gesturing = any(extended_fingers(h) for h in hands if h is not active_hand)
    if other_hand_gesturing:
        return 0
    return len(extended_fingers(active_hand))


def mode_for_count(count: int) -> Mode:
    return _MODE_BY_COUNT.get(count, Mode.CLEAR)


class GestureStabilizer:
    """Only adopts a new mode once it has been the raw reading for `window`
    consecutive frames, so a single misread frame doesn't strobe the
    weather between modes."""

    def __init__(self, window: int = config.GESTURE_DEBOUNCE_FRAMES) -> None:
        self._history: deque[Mode] = deque(maxlen=window)
        self._window = window
        self._stable_mode = Mode.CLEAR

    def update(self, raw_mode: Mode) -> Mode:
        self._history.append(raw_mode)
        if len(self._history) == self._window and all(m == raw_mode for m in self._history):
            self._stable_mode = raw_mode
        return self._stable_mode
