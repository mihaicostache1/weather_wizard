# Weather Wizard

Real-time hand-gesture-controlled weather, rendered live over your webcam feed. Raise fingers to summon rain, snow, or lightning; the hand skeleton is always tracked and shown on screen.

## Gestures

| Fingers up (thumb doesn't count) | Effect |
|---|---|
| 0, or 4+ (open palm) | Clear |
| 1 | Rain |
| 2 | Snow |
| 3 | Lightning, one bolt per raised fingertip |

Only one hand may gesture at a time — if both hands show extended fingers simultaneously, the input is treated as ambiguous and no effect triggers. The right hand is dominant by default; control hands off to the other hand once the active hand closes into a fist while the other hand raises fingers.

## Setup

Requires Python 3.13 (mediapipe's Windows wheel is not guaranteed to work on newer interpreters).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the hand landmark model bundle:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L -o models\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run

```powershell
python -m src.main
```

Must be run as a module (`-m src.main`) from the project root, not as a direct script path — `src` is a package and uses relative imports.

## Keybinds

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `m` | Toggle mirror |
| `h` | Toggle HUD |
| `s` | Save a screenshot to `screenshots/` |

## Project structure

```
src/
├─ main.py           # camera loop and orchestration
├─ config.py          # every tunable value in the project
├─ hand_tracker.py     # mediapipe HandLandmarker wrapper
├─ gestures.py         # finger counting, active-hand selection, debounce
├─ overlay.py          # skeleton + HUD drawing
└─ effects/
   ├─ base.py          # shared particle mechanics + fade helper
   ├─ rain.py
   ├─ snow.py
   └─ lightning.py     # midpoint-displacement bolt geometry
```

All tunables (particle counts, speeds, colors, debounce timing, fade rate, gesture thresholds) live in `src/config.py`.

## How it works

- **Tracking**: mediapipe's Tasks API `HandLandmarker`, running in `VIDEO` mode for temporal consistency across frames.
- **Finger extension**: measured as the bend angle at each finger's PIP knuckle (angle between the MCP→PIP and PIP→TIP segments), which holds regardless of hand rotation or distance from the camera.
- **Rain/snow**: particle state lives in flat numpy arrays, updated with whole-array arithmetic and drawn in single batched OpenCV calls rather than per-particle Python loops.
- **Lightning**: bolts are generated via midpoint displacement — a straight line from above the fingertip to the fingertip is recursively subdivided, with each new midpoint randomly displaced perpendicular to its segment, by an amount that decays at each level.
- **Cross-mode fade**: switching gestures ramps effects in and out over ~0.3s instead of cutting instantly, via an alpha blend onto a reusable off-screen layer.
