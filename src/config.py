from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720

NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

MIRROR_DEFAULT = True
WINDOW_NAME = "Weather Wizard"

# Colors are BGR (opencv convention), not RGB.
SKELETON_LINE_COLOR = (255, 200, 0)
JOINT_COLOR = (0, 165, 255)
JOINT_RADIUS = 4
FINGERTIP_COLOR = (0, 255, 0)
FINGERTIP_EXTENDED_COLOR = (0, 0, 255)
FINGERTIP_TRIGGERED_COLOR = (255, 0, 180)  # BGR bright purple, blinked in when an effect is live
BLINK_HZ = 3.0
FINGERTIP_RADIUS = 7
HAND_LABEL_COLOR = (255, 255, 255)
HUD_TEXT_COLOR = (0, 255, 255)

# Gesture recognition
GESTURE_DEBOUNCE_FRAMES = 4
EXTENDED_FINGER_ANGLE_DEG = 45.0

# Simulation timing
MAX_DT = 0.05  # clamp so a stalled frame (e.g. alt-tab) doesn't teleport particles
EFFECT_FADE_RATE = 3.0  # intensity units per second (~0.33s for a full fade)

# Rain
RAIN_COUNT = 500
RAIN_MIN_SPEED = 650.0  # px/s
RAIN_MAX_SPEED = 1200.0  # px/s
RAIN_MIN_LENGTH = 10.0
RAIN_MAX_LENGTH = 20.0
RAIN_COLOR = (220, 190, 140)  # BGR pale blue
RAIN_THICKNESS = 1

# Snow
SNOW_COUNT = 300
SNOW_MIN_SPEED = 60.0  # px/s, far flakes
SNOW_MAX_SPEED = 220.0  # px/s, near flakes
SNOW_MIN_RADIUS = 1.5
SNOW_MAX_RADIUS = 4.5
SNOW_MIN_SWAY = 10.0  # px
SNOW_MAX_SWAY = 40.0  # px
SNOW_MIN_SWAY_FREQ = 0.5  # rad/s
SNOW_MAX_SWAY_FREQ = 1.5  # rad/s
SNOW_COLOR = (255, 255, 255)  # BGR white

# Lightning
LIGHTNING_SUBDIVISIONS = 6
LIGHTNING_DISPLACEMENT_FRACTION = 0.45  # relative to bolt length, decays per level
LIGHTNING_DECAY = 0.55
LIGHTNING_TOP_JITTER_FRACTION = 0.12  # how far the origin wanders sideways from straight above the tip

# Forked side channels
LIGHTNING_BRANCH_MIN = 2
LIGHTNING_BRANCH_MAX = 5
LIGHTNING_BRANCH_ANGLE_DEG = (25.0, 55.0)
LIGHTNING_BRANCH_LENGTH_FRACTION = (0.18, 0.45)  # of the remaining distance to the tip
LIGHTNING_BRANCH_SUBDIVISIONS = 4
LIGHTNING_BRANCH_WIDTH_SCALE = 0.55

# The channel tapers from cloud end to strike point
LIGHTNING_WIDTH_TOP = 7.0
LIGHTNING_WIDTH_TIP = 1.5
LIGHTNING_TAPER_STEPS = 8

LIGHTNING_CORE_COLOR = (255, 255, 255)  # BGR white
LIGHTNING_GLOW_TINT = (1.6, 1.0, 1.25)  # BGR multipliers - pushes the halo blue-violet
LIGHTNING_BLOOM_DOWNSCALE = 4  # blur at 1/N resolution: faster and smoother than full-res
LIGHTNING_BLOOM_SIGMA = 6.0
LIGHTNING_BLOOM_GAIN = 1.5

LIGHTNING_IMPACT_RADIUS = 26.0

LIGHTNING_BOLT_LIFETIME = 0.22  # seconds
LIGHTNING_INTENSITY_FALLOFF = 1.6  # exponent on the fade curve
LIGHTNING_FLICKER_RANGE = (0.45, 1.0)
LIGHTNING_RESTRIKE_DELAY = (0.05, 0.22)  # randomized gap so a held gesture doesn't machine-gun

LIGHTNING_FLASH_DECAY = 0.75
LIGHTNING_FLASH_MAX_ALPHA = 0.28

# Swipe detection (wind trigger)
SWIPE_WINDOW_SEC = 0.5
HAND_UP_MAX_TILT_DEG = 60.0  # how far the palm axis may lean from straight up and still count as upright
SWIPE_MIN_DISTANCE_FRACTION = 0.25  # net horizontal wrist movement, as a fraction of frame width, within the window
SWIPE_COOLDOWN_SEC = 1.0

# Wind (swipe-triggered gust)
WIND_STREAK_COUNT = 150
WIND_MIN_SPEED = 1800.0  # px/s
WIND_MAX_SPEED = 3000.0  # px/s
WIND_MIN_LENGTH = 40.0
WIND_MAX_LENGTH = 90.0
WIND_STREAK_LIFETIME = 0.6  # seconds
WIND_COLOR = (200, 200, 200)  # BGR light grey-white
WIND_THICKNESS = 2
