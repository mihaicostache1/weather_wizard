from __future__ import annotations

import cv2
import numpy as np

from . import config
from .gestures import Mode, extended_fingers
from .hand_tracker import FINGERTIP_IDS, HAND_CONNECTIONS, Hand

_FINGERTIP_INDICES = set(FINGERTIP_IDS.values())
_TIP_ID_TO_NAME = {tip_idx: name for name, tip_idx in FINGERTIP_IDS.items()}


def draw_skeleton(
    frame: np.ndarray,
    hands: list[Hand],
    primary_hand: Hand | None = None,
    triggering_hand: Hand | None = None,
    blink_on: bool = False,
) -> None:
    for hand in hands:
        pts = hand.landmarks_px.astype(int)
        extended = set(extended_fingers(hand))
        is_triggering = hand is triggering_hand

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), config.SKELETON_LINE_COLOR, 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(pts):
            if idx in _FINGERTIP_INDICES:
                is_extended = _TIP_ID_TO_NAME[idx] in extended
                if is_extended and is_triggering and blink_on:
                    color = config.FINGERTIP_TRIGGERED_COLOR
                elif is_extended:
                    color = config.FINGERTIP_EXTENDED_COLOR
                else:
                    color = config.FINGERTIP_COLOR
                cv2.circle(frame, (x, y), config.FINGERTIP_RADIUS, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), config.JOINT_RADIUS, config.JOINT_COLOR, -1, cv2.LINE_AA)

        wrist_x, wrist_y = pts[0]
        label = f"{hand.handedness} {hand.score:.2f}"
        if hand is primary_hand:
            label += " (active)"
        cv2.putText(
            frame,
            label,
            (wrist_x - 20, wrist_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            config.HAND_LABEL_COLOR,
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    hand_count: int,
    mirror: bool,
    mode: Mode,
    finger_count: int,
    swipe_fraction: float = 0.0,
) -> None:
    lines = (
        f"fps: {fps:5.1f}   hands: {hand_count}",
        f"mode: {mode.value:<10} fingers: {finger_count}",
        f"swipe: {swipe_fraction:.2f} / {config.SWIPE_MIN_DISTANCE_FRACTION:.2f}",
        f"[m] mirror:{'on' if mirror else 'off'}   [h] hud   [s] screenshot   [q] quit",
    )
    y = 30
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.HUD_TEXT_COLOR, 2, cv2.LINE_AA)
        y += 28
