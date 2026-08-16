from __future__ import annotations

import cv2
import numpy as np

from . import config
from .hand_tracker import FINGERTIP_IDS, HAND_CONNECTIONS, Hand

_FINGERTIP_INDICES = set(FINGERTIP_IDS.values())


def draw_skeleton(frame: np.ndarray, hands: list[Hand]) -> None:
    for hand in hands:
        pts = hand.landmarks_px.astype(int)

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), config.SKELETON_LINE_COLOR, 2, cv2.LINE_AA)

        for idx, (x, y) in enumerate(pts):
            if idx in _FINGERTIP_INDICES:
                cv2.circle(frame, (x, y), config.FINGERTIP_RADIUS, config.FINGERTIP_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), config.JOINT_RADIUS, config.JOINT_COLOR, -1, cv2.LINE_AA)

        wrist_x, wrist_y = pts[0]
        label = f"{hand.handedness} {hand.score:.2f}"
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


def draw_hud(frame: np.ndarray, fps: float, hand_count: int, mirror: bool) -> None:
    lines = (
        f"fps: {fps:5.1f}   hands: {hand_count}",
        f"[m] mirror:{'on' if mirror else 'off'}   [h] hud   [q] quit",
    )
    y = 30
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.HUD_TEXT_COLOR, 2, cv2.LINE_AA)
        y += 28
