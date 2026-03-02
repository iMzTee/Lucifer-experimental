"""constants.py — Arena dimensions, physics constants, boost pad positions.

All values from RocketSim (github.com/ZealanL/RocketSim). Stored as Python
floats; converted to tensors at init time in the modules that need them.
"""

import torch
import math

# ── Arena geometry ──
ARENA_HALF_X = 4096.0       # side walls
ARENA_HALF_Y = 5120.0       # back walls (goal wall Y)
ARENA_HEIGHT = 2048.0        # ceiling
GOAL_HALF_WIDTH = 892.755    # goal opening half-width in X
GOAL_HEIGHT = 642.775        # goal opening height
GOAL_DEPTH = 880.0           # back net depth behind goal line
CORNER_RADIUS = 1024.0       # curved corner radius (simplified to flat)

# ── Ball ──
BALL_RADIUS = 91.25          # soccar ball collision radius
BALL_REST_Z = 93.15          # resting height on flat ground
BALL_MAX_SPEED = 6000.0
BALL_RESTITUTION = 0.6       # bounce coefficient
BALL_DRAG = 0.03             # per-second drag factor
BALL_MASS = 30.0             # relative mass for collision (180/6)
BALL_FRICTION = 0.35          # surface friction coefficient
BALL_MAX_ANG_SPEED = 6.0      # rad/s angular velocity cap

# ── Car (Octane) ──
CAR_HITBOX_LENGTH = 120.507  # full hitbox dimensions
CAR_HITBOX_WIDTH = 86.6994
CAR_HITBOX_HEIGHT = 38.6591
CAR_HITBOX_OFFSET = (13.8757, 0.0, 20.755)
CAR_EFFECTIVE_RADIUS = 75.0   # sphere approximation for collision
CAR_MAX_SPEED = 2300.0
CAR_SUPERSONIC_SPEED = 2200.0  # demo threshold
CAR_MASS = 180.0              # relative mass for collision
CAR_WHEELBASE = 85.0          # front_wheel_x - rear_wheel_x

# ── Ground car physics ──
# Throttle: base acceleration scaled by speed-dependent torque factor
THROTTLE_ACCEL = 1600.0       # uu/s² peak throttle acceleration (at 0 speed)
BRAKE_ACCEL = 3500.0          # uu/s² braking deceleration
COAST_DECEL = 525.0           # uu/s² coasting deceleration (no throttle)
COASTING_BRAKE_FACTOR = 0.15  # fraction of BRAKE_ACCEL for coasting (0.15*3500=525)
STOPPING_SPEED = 25.0         # uu/s below which car fully stops

# Throttle torque factor curve: piecewise linear (speed → factor)
# At 0 speed: full torque; at 1410 uu/s: zero torque (max throttle-only speed)
THROTTLE_TORQUE_SPEEDS = [0.0, 1400.0, 1410.0]
THROTTLE_TORQUE_FACTORS = [1.0, 0.1, 0.0]

# Boost
BOOST_ACCEL_GROUND = 991.667  # uu/s² boost on ground (2975/3)
BOOST_ACCEL_AIR = 1058.333    # uu/s² boost in air (3175/3)
BOOST_CONSUMPTION = 1.0 / 3.0  # boost per second in 0-1 scale (full boost lasts 3s)

# Steering: speed-dependent steer angle curve (piecewise linear)
# abs(speed) → steer_angle (radians) at full steer input
STEER_ANGLE_SPEEDS = [0.0, 500.0, 1000.0, 1500.0, 1750.0, 3000.0]
STEER_ANGLE_VALUES = [0.53356, 0.31930, 0.18203, 0.10570, 0.08507, 0.03454]

# Powerslide steer angle curve (used when handbrake active, interpolated with normal curve)
POWERSLIDE_STEER_SPEEDS = [0.0, 2500.0]
POWERSLIDE_STEER_VALUES = [0.39235, 0.12610]

# Lateral friction curves (slip-angle model)
LAT_FRICTION_CURVE_X = [0.0, 1.0]            # slip ratio breakpoints
LAT_FRICTION_CURVE_Y = [1.0, 0.2]            # friction factor at each slip ratio
HANDBRAKE_LAT_FRICTION_FACTOR = 0.1           # multiply lat friction when full handbrake

# Handbrake analog lerp rates
HANDBRAKE_RISE_RATE = 5.0     # per-second rise when holding
HANDBRAKE_FALL_RATE = 2.0     # per-second fall when released

# ── Air car physics ──
PITCH_TORQUE = 12.46          # rad/s² pitch (130 * CAR_TORQUE_SCALE)
YAW_TORQUE = 9.11             # rad/s² yaw (95 * CAR_TORQUE_SCALE)
ROLL_TORQUE = 38.34           # rad/s² roll (400 * CAR_TORQUE_SCALE)
# Per-axis angular velocity damping rates (per second, from RocketSim CAR_AIR_CONTROL_DAMPING)
# Pitch/yaw damping scales with (1 - |input|): zero when holding full input
# Roll damping is always active regardless of input
PITCH_ANG_DAMPING = 2.876     # 30 * CAR_TORQUE_SCALE
YAW_ANG_DAMPING = 1.917       # 20 * CAR_TORQUE_SCALE
ROLL_ANG_DAMPING = 4.794      # 50 * CAR_TORQUE_SCALE
AIR_THROTTLE_ACCEL = 66.667   # uu/s² air throttle (200/3)
CAR_MAX_ANG_SPEED = 5.5       # rad/s max angular velocity magnitude

# Auto-roll (corrective torque toward surface-aligned orientation)
CAR_AUTOROLL_FORCE = 100.0    # uu/s² (not used directly, informational)
CAR_AUTOROLL_TORQUE = 80.0    # rad/s² corrective roll torque on ground

# Sticky forces (keep car attached to surface)
STICKY_FORCE_GROUND = 0.5     # base sticky force multiplier

# ── Jump / flip ──
JUMP_IMPULSE = 291.667        # uu/s upward impulse on first jump (875/3)
JUMP_HOLD_FORCE = 1458.333    # uu/s² sustained upward force while holding (4375/3)
JUMP_HOLD_TIME = 0.2          # seconds max hold duration
JUMP_MIN_TIME = 0.025         # seconds: force scale reduced during this window
JUMP_MIN_FORCE_SCALE = 0.62   # force multiplier during first JUMP_MIN_TIME
FLIP_IMPULSE = 500.0          # uu/s base horizontal dodge velocity
FLIP_FORWARD_SCALE = 1.0      # dodge impulse scale: forward
FLIP_SIDE_SCALE = 1.9         # dodge impulse scale: sideways
FLIP_BACKWARD_SCALE = 2.5     # dodge impulse scale: backward
FLIP_BACKWARD_X_SCALE = 16.0 / 15.0  # extra backward multiplier (1.067)
FLIP_Z_DAMP = 0.05            # Z velocity multiplier on dodge (~0.65^7)
FLIP_Z_DAMP_PER_TICK = 0.65   # per-tick Z damping during gradual window
FLIP_Z_DAMP_START = 0.15      # seconds after flip: start gradual Z damp
FLIP_Z_DAMP_END = 0.21        # seconds after flip: end gradual Z damp
FLIP_TIMER = 1.25             # seconds after jump before flip expires
FLIP_TORQUE_X = 260.0         # rad/s² flip torque around X axis (roll/yaw)
FLIP_TORQUE_Y = 224.0         # rad/s² flip torque around Y axis (pitch)
FLIP_TORQUE_TIME = 0.65       # seconds of active flip torque
FLIP_PITCH_LOCK_TIME = 0.95   # seconds of air pitch lock after flip
DEMO_RESPAWN_TIME = 3.0       # seconds until respawn after demo

# Auto-flip recovery
CAR_AUTOFLIP_ROLL_THRESH = 2.8     # radians: |roll| threshold to trigger
CAR_AUTOFLIP_TORQUE = 500.0        # rad/s² recovery torque
CAR_AUTOFLIP_TIME = 0.4            # seconds for recovery

# Supersonic tracking (hysteresis)
CAR_SUPERSONIC_ACTIVATE = 2200.0   # speed to activate supersonic
CAR_SUPERSONIC_MAINTAIN = 2100.0   # speed threshold to maintain
CAR_SUPERSONIC_MAINTAIN_TIME = 1.0 # seconds to maintain below threshold

# Ball-car extra impulse (RocketSim)
BALL_CAR_EXTRA_IMPULSE_Z_SCALE = 0.35         # Z component scale
BALL_CAR_EXTRA_IMPULSE_FWD_SCALE = 0.65       # forward adjustment
BALL_CAR_EXTRA_IMPULSE_MAX_SPEED_RATIO = 0.3  # max ratio at high speeds
BALL_CAR_EXTRA_IMPULSE_SPEEDS = [0.0, 500.0, 2300.0, 4600.0]
BALL_CAR_EXTRA_IMPULSE_FACTORS = [0.65, 0.65, 0.55, 0.30]

# Bump velocity curves (speed-dependent bump amounts)
BUMP_GROUND_SPEEDS = [0.0, 500.0, 2300.0]
BUMP_GROUND_FACTORS = [0.5, 0.5, 0.3]
BUMP_AIR_SPEEDS = [0.0, 500.0, 2300.0]
BUMP_AIR_FACTORS = [0.7, 0.7, 0.4]
BUMP_UPWARD_SPEEDS = [0.0, 500.0, 2300.0]
BUMP_UPWARD_FACTORS = [0.6, 0.6, 0.35]

# Gravity
GRAVITY = -650.0              # uu/s² downward

# ── Boost pads ──
# All 34 boost pad positions [x, y, z] from RocketSim
# 6 large pads (100 boost, 10s respawn), 28 small pads (12 boost, 4s respawn)
BOOST_PAD_POSITIONS = torch.tensor([
    # 6 large pads (indices 0-5)
    [-3584.0,    0.0, 73.0],
    [ 3584.0,    0.0, 73.0],
    [-3072.0,  4096.0, 73.0],
    [ 3072.0,  4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [ 3072.0, -4096.0, 73.0],
    # 28 small pads (indices 6-33)
    [    0.0, -4240.0, 70.0],
    [-1792.0, -4184.0, 70.0],
    [ 1792.0, -4184.0, 70.0],
    [-940.0,  -3308.0, 70.0],
    [ 940.0,  -3308.0, 70.0],
    [    0.0, -2816.0, 70.0],
    [-3584.0, -2484.0, 70.0],
    [ 3584.0, -2484.0, 70.0],
    [-1788.0, -2300.0, 70.0],
    [ 1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0],
    [ 2048.0, -1036.0, 70.0],
    [-1024.0,     0.0, 70.0],
    [ 1024.0,     0.0, 70.0],
    [-2048.0,  1036.0, 70.0],
    [ 2048.0,  1036.0, 70.0],
    [-1788.0,  2300.0, 70.0],
    [ 1788.0,  2300.0, 70.0],
    [-3584.0,  2484.0, 70.0],
    [ 3584.0,  2484.0, 70.0],
    [    0.0,  2816.0, 70.0],
    [-940.0,   3308.0, 70.0],
    [ 940.0,   3308.0, 70.0],
    [-1792.0,  4184.0, 70.0],
    [ 1792.0,  4184.0, 70.0],
    [    0.0,  4240.0, 70.0],
    [-3584.0,     0.0, 70.0],  # overlap with large? keep for 34 total
    [ 3584.0,     0.0, 70.0],
], dtype=torch.float32)

N_BOOST_PADS = 34
N_LARGE_PADS = 6
LARGE_PAD_BOOST = 1.0        # full boost
SMALL_PAD_BOOST = 0.12       # 12% boost
LARGE_PAD_RESPAWN = 10.0     # seconds
SMALL_PAD_RESPAWN = 4.0      # seconds
BOOST_PAD_PICKUP_RADIUS = 160.0  # pickup distance (legacy, uniform)
BOOST_PAD_PICKUP_RADIUS_BIG = 160.0    # big pad pickup radius
BOOST_PAD_PICKUP_RADIUS_SMALL = 120.0  # small pad pickup radius

# ── Observation / Reward constants ──
POS_COEF = 1.0 / 2300.0
LIN_VEL_COEF = 1.0 / 2300.0
ANG_VEL_COEF = 1.0 / 3.14159265

BACK_NET_Y = 6000.0
ORANGE_GOAL_BACK = torch.tensor([0.0, BACK_NET_Y, GOAL_HEIGHT / 2], dtype=torch.float32)
BLUE_GOAL_BACK = torch.tensor([0.0, -BACK_NET_Y, GOAL_HEIGHT / 2], dtype=torch.float32)

# ── Agent layout function ──
def get_agent_layout(n_agents):
    """Return layout dict for 1v0 / 1v1 / 2v2 configurations.

    Returns dict with:
        n_agents: total number of agents
        blue_cars: list of car indices on blue team
        orange_cars: list of car indices on orange team
        car_pairs: list of (i,j) pairs for car-car collision
        ally_idx: per-agent index of ally (-1 if none)
        enemy0_idx: per-agent index of first enemy (-1 if none)
        enemy1_idx: per-agent index of second enemy (-1 if none)
        is_orange: per-agent bool
        car_team: list of team assignments (0=blue, 1=orange)
    """
    if n_agents == 1:
        # 1v0: solo blue car, no opponents
        return {
            "n_agents": 1,
            "blue_cars": [0],
            "orange_cars": [],
            "car_pairs": [],
            "ally_idx": [-1],
            "enemy0_idx": [-1],
            "enemy1_idx": [-1],
            "is_orange": [False],
            "car_team": [0],
        }
    elif n_agents == 2:
        # 1v1: one blue (0), one orange (1)
        return {
            "n_agents": 2,
            "blue_cars": [0],
            "orange_cars": [1],
            "car_pairs": [(0, 1)],
            "ally_idx": [-1, -1],
            "enemy0_idx": [1, 0],
            "enemy1_idx": [-1, -1],
            "is_orange": [False, True],
            "car_team": [0, 1],
        }
    else:
        # 2v2: [B0, B1, O0, O1]
        return {
            "n_agents": 4,
            "blue_cars": [0, 1],
            "orange_cars": [2, 3],
            "car_pairs": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            "ally_idx": [1, 0, 3, 2],
            "enemy0_idx": [2, 2, 0, 0],
            "enemy1_idx": [3, 3, 1, 1],
            "is_orange": [False, False, True, True],
            "car_team": [0, 0, 1, 1],
        }

# ── Curriculum ──
STAGE_CONFIG = {
    0: {"tick_skip": 1, "timeout": 3600, "n_agents": 1, "n_envs": 160000},  # Ground Basics
    1: {"tick_skip": 1, "timeout": 3600, "n_agents": 1, "n_envs": 160000},  # Ground Advanced
    2: {"tick_skip": 1, "timeout": 4800, "n_agents": 1, "n_envs": 160000},  # Air Mechanics
    3: {"tick_skip": 1, "timeout": 4800, "n_agents": 2, "n_envs": 80000},   # 1v1 Basics
    4: {"tick_skip": 1, "timeout": 7200, "n_agents": 2, "n_envs": 80000},   # 1v1 Advanced
    5: {"tick_skip": 1, "timeout": 9600, "n_agents": 4, "n_envs": 40000},   # 2v2 Teamwork
}

# Physics tick rate
PHYSICS_HZ = 120
DT = 1.0 / PHYSICS_HZ

# ── Kickoff positions (blue side) ──
KICKOFF_POSITIONS = torch.tensor([
    [-2048.0, -2560.0, 17.0],   # right diagonal
    [ 2048.0, -2560.0, 17.0],   # left diagonal
    [ -256.0, -3840.0, 17.0],   # right back-center
    [  256.0, -3840.0, 17.0],   # left back-center
    [    0.0, -4608.0, 17.0],   # far back (goalie)
], dtype=torch.float32)
KICKOFF_YAWS = torch.tensor([
    math.pi / 4,      # right diagonal: 45° (facing toward center)
    3 * math.pi / 4,  # left diagonal: 135° (facing toward center)
    math.pi / 2,      # right back-center: 90° (facing forward)
    math.pi / 2,      # left back-center: 90° (facing forward)
    math.pi / 2,      # far back: 90° (facing forward)
], dtype=torch.float32)
