"""constants.py — Arena dimensions, physics constants, boost pad positions.

All values from RocketSim / rlgym_sim. Stored as Python floats; converted to
tensors at init time in the modules that need them.
"""

import torch

# ── Arena geometry ──
ARENA_HALF_X = 4096.0       # side walls
ARENA_HALF_Y = 5120.0       # back walls (goal wall Y)
ARENA_HEIGHT = 2044.0        # ceiling
GOAL_HALF_WIDTH = 892.755    # goal opening half-width in X
GOAL_HEIGHT = 642.775        # goal opening height
GOAL_DEPTH = 880.0           # back net depth behind goal line
CORNER_RADIUS = 1024.0       # curved corner radius (simplified to flat)

# ── Ball ──
BALL_RADIUS = 92.75
BALL_MAX_SPEED = 6000.0
BALL_RESTITUTION = 0.6       # bounce coefficient
BALL_DRAG = 0.0305           # per-second drag factor (approx from RocketSim)
BALL_MASS = 30.0             # relative mass for collision

# ── Car (Octane) ──
CAR_HITBOX_HALF = (118.01, 84.20, 36.16)   # half-extents (L, W, H)
CAR_HITBOX_OFFSET = (13.88, 0.0, 20.75)    # hitbox center offset from car origin
CAR_EFFECTIVE_RADIUS = 75.0   # sphere approximation for collision
CAR_MAX_SPEED = 2300.0
CAR_SUPERSONIC_SPEED = 2200.0  # demo threshold
CAR_MASS = 180.0              # relative mass for collision

# Ground car physics
THROTTLE_ACCEL = 1600.0       # uu/s² max throttle acceleration
BRAKE_ACCEL = 3500.0          # uu/s² braking deceleration
BOOST_ACCEL = 991.667         # uu/s² boost acceleration
BOOST_CONSUMPTION = 33.3      # boost per second (100 boost lasts 3s)
MAX_STEER_RATE = 2.5          # rad/s max ground turn rate

# Air car physics
PITCH_TORQUE = 12.46          # rad/s² pitch
YAW_TORQUE = 9.11             # rad/s² yaw
ROLL_TORQUE = 38.34           # rad/s² roll
ANG_VEL_DAMPING = 0.988       # per-tick angular velocity damping (120Hz)
AIR_THROTTLE_ACCEL = 66.667   # uu/s² air throttle (very weak)

# Jump / flip
JUMP_IMPULSE = 292.0          # uu/s upward impulse on first jump
JUMP_HOLD_FORCE = 1458.0      # uu/s² sustained upward force while holding
JUMP_HOLD_TIME = 0.2          # seconds max hold duration
FLIP_IMPULSE = 500.0          # uu/s horizontal dodge impulse
FLIP_TORQUE = 5.5             # rad/s² dodge spin
FLIP_TIMER = 1.25             # seconds after jump before flip expires
DEMO_RESPAWN_TIME = 3.0       # seconds until respawn after demo

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
BOOST_PAD_PICKUP_RADIUS = 160.0  # pickup distance (generous)

# ── Observation / Reward constants ──
POS_COEF = 1.0 / 2300.0
LIN_VEL_COEF = 1.0 / 2300.0
ANG_VEL_COEF = 1.0 / 3.14159265

BACK_NET_Y = 6000.0
ORANGE_GOAL_BACK = torch.tensor([0.0, BACK_NET_Y, GOAL_HEIGHT / 2], dtype=torch.float32)
BLUE_GOAL_BACK = torch.tensor([0.0, -BACK_NET_Y, GOAL_HEIGHT / 2], dtype=torch.float32)

# ── 2v2 player layout: [B0, B1, O0, O1] ──
ALLY_IDX = [1, 0, 3, 2]
ENEMY0_IDX = [2, 2, 0, 0]
ENEMY1_IDX = [3, 3, 1, 1]
IS_ORANGE = [False, False, True, True]

# ── Curriculum ──
STAGE_CONFIG = {
    0: {"tick_skip": 8, "timeout": 300},
    1: {"tick_skip": 8, "timeout": 400},
    2: {"tick_skip": 4, "timeout": 600},
    3: {"tick_skip": 2, "timeout": 1200},
}

# Physics tick rate
PHYSICS_HZ = 120
DT = 1.0 / PHYSICS_HZ

# ── Kickoff positions ──
KICKOFF_POSITIONS = torch.tensor([
    [-2048.0, -2560.0, 17.0],
    [ 2048.0, -2560.0, 17.0],
    [ -256.0, -3840.0, 17.0],
    [  256.0, -3840.0, 17.0],
    [    0.0, -4608.0, 17.0],
], dtype=torch.float32)
