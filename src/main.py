from __future__ import annotations

import time

import cv2
import numpy as np

from . import config
from .effects.base import Fader
from .effects.lightning import Lightning
from .effects.rain import Rain
from .effects.snow import Snow
from .effects.wind import Wind
from .gestures import (
    ActiveHandSelector,
    GestureStabilizer,
    Mode,
    SwipeDetector,
    extended_fingers,
    mode_for_count,
    resolve_finger_count,
)
from .hand_tracker import FINGERTIP_IDS, HandTracker
from .overlay import draw_hud, draw_skeleton


def save_screenshot(frame) -> None:
    config.SCREENSHOT_DIR.mkdir(exist_ok=True)
    path = config.SCREENSHOT_DIR / f"weather_wizard_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)


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
    stabilizer = GestureStabilizer()
    active_hand_selector = ActiveHandSelector()
    swipe_detectors = {"Left": SwipeDetector(), "Right": SwipeDetector()}
    rain: Rain | None = None
    snow: Snow | None = None
    lightning: Lightning | None = None
    rain_layer: np.ndarray | None = None
    snow_layer: np.ndarray | None = None
    rain_fader = Fader(config.EFFECT_FADE_RATE)
    snow_fader = Fader(config.EFFECT_FADE_RATE)

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

            if rain is None:
                h, w = frame.shape[:2]
                rain = Rain(w, h)
                snow = Snow(w, h)
                lightning = Lightning(w, h)
                rain_layer = np.zeros_like(frame)
                snow_layer = np.zeros_like(frame)

            now = time.perf_counter()
            dt = min(now - prev_tick, config.MAX_DT)
            prev_tick = now

            hands = tracker.detect(frame, int((now - start) * 1000), mirrored=mirror)

            primary = active_hand_selector.update(hands)
            finger_count = resolve_finger_count(hands, primary)
            stable_mode = stabilizer.update(mode_for_count(finger_count))

            rain_alpha = rain_fader.update(1.0 if stable_mode is Mode.RAIN else 0.0, dt)
            if rain_alpha > 0.01:
                rain.update(dt)
                rain_layer.fill(0)
                rain.draw(rain_layer)
                cv2.addWeighted(frame, 1.0, rain_layer, rain_alpha, 0.0, dst=frame)

            snow_alpha = snow_fader.update(1.0 if stable_mode is Mode.SNOW else 0.0, dt)
            if snow_alpha > 0.01:
                snow.update(dt)
                snow_layer.fill(0)
                snow.draw(snow_layer)
                cv2.addWeighted(frame, 1.0, snow_layer, snow_alpha, 0.0, dst=frame)

            by_side = {hand.handedness: hand for hand in hands}
            frame_width = frame.shape[1]
            swipe_direction = None
            for side, detector in swipe_detectors.items():
                direction = detector.update(by_side.get(side), now, dt, frame_width)
                if direction is not None:
                    swipe_direction = direction
            if swipe_direction is not None:
                wind.trigger(swipe_direction)
            wind.update(dt)
            wind.draw(frame)

            triggering_hand = primary if stable_mode is not Mode.CLEAR else None
            blink_on = int(now * config.BLINK_HZ * 2) % 2 == 0
            draw_skeleton(frame, hands, primary_hand=primary, triggering_hand=triggering_hand, blink_on=blink_on)

            lightning_targets: dict[str, tuple[float, float]] = {}
            if stable_mode is Mode.LIGHTNING and primary is not None:
                for name in extended_fingers(primary):
                    x, y = primary.landmarks_px[FINGERTIP_IDS[name]]
                    lightning_targets[name] = (float(x), float(y))
            lightning.update(lightning_targets)
            lightning.draw(frame)

            if dt > 0:
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else fps_ema * 0.9 + inst_fps * 0.1

            if hud_visible:
                swipe_fraction = max(d.last_fraction for d in swipe_detectors.values())
                draw_hud(frame, fps_ema, len(hands), mirror, stable_mode, finger_count, swipe_fraction)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("h"):
                hud_visible = not hud_visible
            elif key == ord("s"):
                save_screenshot(frame)
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
