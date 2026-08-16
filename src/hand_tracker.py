from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from . import config

# Standard 21-point mediapipe hand topology: (wrist=0, thumb=1-4,
# index=5-8, middle=9-12, ring=13-16, pinky=17-20), plus the palm edges
# tying each finger's base to the next.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGERTIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


@dataclass
class Hand:
    landmarks_px: np.ndarray  # (21, 2) float32 pixel coords in the processed frame
    landmarks_norm: np.ndarray  # (21, 3) float32 mediapipe-normalized x, y, z
    handedness: str  # "Left" or "Right", as perceived by the user
    score: float  # handedness classifier confidence; used as a per-hand quality proxy


class HandTracker:
    def __init__(self) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int, mirrored: bool = True) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[Hand] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            norm = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            px = norm[:, :2] * np.array([w, h], dtype=np.float32)
            category = handedness[0]
            label = category.category_name
            if mirrored:
                # The classifier reads raw pixel chirality. Flipping the
                # frame for the mirror view flips a real hand's chirality
                # too, so the raw label comes out swapped from reality.
                label = "Left" if label == "Right" else "Right"
            hands.append(
                Hand(
                    landmarks_px=px,
                    landmarks_norm=norm,
                    handedness=label,
                    score=category.score,
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
