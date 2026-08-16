"""Smoke test - throwaway diagnostic, not part of the final app.

Proves the whole stack end to end before we write a single line of the real
thing:

  1. interpreter + package versions
  2. which mediapipe hand API is available (Tasks vs legacy solutions)
  3. the webcam opens and yields a real frame
  4. HandLandmarker instantiates from the downloaded .task bundle
  5. inference actually runs per frame, at a usable frame rate

Run it with the venv active:  python scripts\\smoke_test.py
Press q or ESC to close the preview window (it also auto-closes after 20s).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

PREVIEW_TIMEOUT_S = 20.0
CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 1280, 720


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_imports():
    """Import the stack and report versions. Returns (np, cv2, mp) or exits."""
    section("1. Interpreter and packages")
    info(f"python {sys.version.split()[0]} ({sys.executable})")

    try:
        import numpy as np
    except ImportError as exc:
        fail(f"numpy import failed: {exc}")
        sys.exit(1)
    ok(f"numpy {np.__version__}")

    try:
        import cv2
    except ImportError as exc:
        fail(f"cv2 import failed: {exc}")
        sys.exit(1)
    ok(f"cv2 {cv2.__version__}")

    try:
        import mediapipe as mp
    except ImportError as exc:
        fail(f"mediapipe import failed: {exc}")
        sys.exit(1)
    ok(f"mediapipe {getattr(mp, '__version__', 'unknown')}")

    return np, cv2, mp


def check_api(mp):
    """Determine which hand-tracking API this mediapipe build exposes."""
    section("2. Available hand API")

    has_tasks = False
    try:
        from mediapipe.tasks.python import vision

        has_tasks = hasattr(vision, "HandLandmarker")
    except ImportError as exc:
        info(f"tasks API not importable: {exc}")

    if has_tasks:
        ok("Tasks API present: mediapipe.tasks.python.vision.HandLandmarker")
    else:
        fail("Tasks API missing - the plan needs revisiting")

    legacy = hasattr(getattr(mp, "solutions", None), "hands")
    info(f"legacy mp.solutions.hands present: {legacy}")

    if not has_tasks and not legacy:
        fail("no usable hand tracking API at all")
        sys.exit(1)

    return has_tasks


def check_model():
    section("3. Model bundle")
    if not MODEL_PATH.exists():
        fail(f"missing {MODEL_PATH}")
        info("download the .task bundle into models/ before running this script")
        sys.exit(1)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        fail(f"{MODEL_PATH.name} is only {size_mb:.2f} MB - download looks truncated")
        sys.exit(1)
    ok(f"{MODEL_PATH.name} ({size_mb:.2f} MB)")


def open_camera(cv2):
    """Try DirectShow first (fast to open on Windows), then the default backend."""
    section("4. Camera")
    attempts = [("CAP_DSHOW", cv2.CAP_DSHOW), ("default", None)]

    for name, backend in attempts:
        cap = cv2.VideoCapture(CAM_INDEX) if backend is None else cv2.VideoCapture(CAM_INDEX, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            grabbed, frame = cap.read()
            if grabbed and frame is not None:
                h, w = frame.shape[:2]
                ok(f"opened camera {CAM_INDEX} via {name} at {w}x{h}")
                return cap
            cap.release()
        info(f"backend {name} did not yield a frame")

    fail("could not open any camera - check that nothing else is using the webcam")
    sys.exit(1)


def build_landmarker(mp):
    section("5. HandLandmarker")
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    ok("landmarker created in VIDEO running mode (num_hands=2)")
    return landmarker


def run_preview(cv2, mp, cap, landmarker):
    section("6. Live inference")
    info(f"preview running - press q or ESC to stop (auto-stops after {PREVIEW_TIMEOUT_S:.0f}s)")

    started = time.perf_counter()
    frames = 0
    max_hands_seen = 0

    while True:
        elapsed = time.perf_counter() - started
        if elapsed > PREVIEW_TIMEOUT_S:
            break

        grabbed, frame = cap.read()
        if not grabbed:
            fail("frame grab failed mid-stream")
            break

        # Mirror so it behaves like a mirror rather than a video call.
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Tasks API wants RGB wrapped in an mp.Image, plus a monotonically
        # increasing timestamp in ms so VIDEO mode can track across frames.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(elapsed * 1000))

        hands = result.hand_landmarks or []
        max_hands_seen = max(max_hands_seen, len(hands))

        # Bare dots only - the real skeleton renderer arrives in step 2.
        for hand in hands:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        frames += 1
        fps = frames / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            frame,
            f"SMOKE TEST  hands={len(hands)}  fps={fps:5.1f}  q=quit",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Weather Wizard - smoke test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    elapsed = time.perf_counter() - started
    if frames:
        ok(f"{frames} frames in {elapsed:.1f}s -> {frames / elapsed:.1f} fps end to end")
    if max_hands_seen:
        ok(f"detected up to {max_hands_seen} hand(s)")
    else:
        info("no hands detected - fine if you never showed one, otherwise worth a retry")


def main() -> int:
    print("Weather Wizard :: smoke test")
    np, cv2, mp = check_imports()
    check_api(mp)
    check_model()

    cap = open_camera(cv2)
    landmarker = None
    try:
        landmarker = build_landmarker(mp)
        run_preview(cv2, mp, cap, landmarker)
    finally:
        cap.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()

    print("\nsmoke test finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
