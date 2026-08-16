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
SWIPE_MIN_DISTANCE_FRACTION = 0.28  # net horizontal wrist movement, as a fraction of frame width, within the window
SWIPE_COOLDOWN_SEC = 1.0
SWIPE_MIN_NET_RATIO = 0.6  # net travel vs peak-to-peak; separates a one-way swipe from a circle that comes back

# Circle detection (tornado trigger)
CIRCLE_WINDOW_SEC = 1.6
CIRCLE_MIN_SAMPLES = 8
CIRCLE_MIN_TURNS = 1.6  # full revolutions swept within the window
CIRCLE_MIN_RADIUS_FRACTION = 0.04  # of frame width; below this it's landmark jitter, not a circle
CIRCLE_MAX_RADIUS_VARIATION = 0.9  # std/mean of the radii; loose enough for ovals and drifting loops
CIRCLE_MIN_DIRECTION_CONSISTENCY = 0.7  # share of angular steps that must agree in sign; the main defense
# against straight back-and-forth waving, whose radius spread (~0.58) now clears the shape guard above
CIRCLE_COOLDOWN_SEC = 1.0

# Tornado (circle-triggered vortex, emanating along the axis out of the palm)
TORNADO_LIFETIME = 4.0  # seconds a vortex lasts before fading; re-triggering refreshes it
TORNADO_FADE_RATE = 1.6  # intensity units per second
TORNADO_INNER_RADIUS_FRACTION = 0.02  # radius at the palm, as a fraction of frame width
TORNADO_OUTER_RADIUS_FRACTION = 0.20  # radius at full extension toward the viewer
TORNADO_FLARE_EXPONENT = 1.5  # >1 curves the flare instead of leaving a straight cone
TORNADO_TILT_SQUASH = 0.75  # vertical squash, so the spiral reads as a tilted disc rather than flat-on
TORNADO_ARMS = 4  # spiral arms drawn as continuous curves
TORNADO_ARM_SEGMENTS = 56  # samples along each arm; more is smoother
TORNADO_SPIN_RATE = 4.0  # rad/s the whole spiral rotates
TORNADO_SPIRAL_TWIST = 4.5  # radians of shear across the full extension - sets how tightly the arms curl
TORNADO_ARM_WIDTH_INNER = 1.0  # stroke width at the palm
TORNADO_ARM_WIDTH_OUTER = 4.0  # and at full extension
TORNADO_ARM_SLICES = 12  # polylines can't vary width along a stroke, so each arm is drawn in this many pieces
TORNADO_FADE_IN_Z = 0.12  # arms ramp up over this much of their length
TORNADO_FADE_OUT_Z = 0.75  # and dissolve past here, so they don't end abruptly
TORNADO_COLOR = (200, 195, 185)  # BGR pale dust

# Wind (swipe-triggered gust)
WIND_STREAK_COUNT = 150
WIND_MIN_SPEED = 1800.0  # px/s
WIND_MAX_SPEED = 3000.0  # px/s
WIND_MIN_LENGTH = 40.0
WIND_MAX_LENGTH = 90.0
WIND_STREAK_LIFETIME = 0.6  # seconds
WIND_COLOR = (200, 200, 200)  # BGR light grey-white
WIND_THICKNESS = 2
