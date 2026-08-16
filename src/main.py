from __future__ import annotations

import time

import cv2

from . import config
from .hand_tracker import HandTracker
from .overlay import draw_hud, draw_skeleton


def open_camera() -> cv2.VideoCapture:
    for backend in (cv2.CAP_DSHOW, None):
        cap = cv2.VideoCapture(config.CAM_INDEX) if backend is None else cv2.VideoCapture(config.CAM_INDEX, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
            return cap
        cap.release()
    raise RuntimeError("could not open any camera")


def main() -> int:
    cap = open_camera()
    tracker = HandTracker()

    mirror = config.MIRROR_DEFAULT
    hud_visible = True
    fps_ema = 0.0
    start = time.perf_counter()
    prev_tick = start

    try:
        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            hands = tracker.detect(frame, int((now - start) * 1000))

            draw_skeleton(frame, hands)

            dt = now - prev_tick
            prev_tick = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else fps_ema * 0.9 + inst_fps * 0.1

            if hud_visible:
                draw_hud(frame, fps_ema, len(hands), mirror)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("h"):
                hud_visible = not hud_visible
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
