"""arena.py — Arena geometry: wall, floor, ceiling collision for ball and cars.

Simplified model: axis-aligned box with goal openings on ±Y walls.
Curved corners are approximated as flat walls.
"""

import torch
from .constants import (
    ARENA_HALF_X, ARENA_HALF_Y, ARENA_HEIGHT,
    BALL_RADIUS, BALL_RESTITUTION, CAR_EFFECTIVE_RADIUS,
    GOAL_HALF_WIDTH, GOAL_HEIGHT, GOAL_DEPTH, BACK_NET_Y,
)


def arena_collide_ball(state):
    """Clamp ball position to arena bounds, reflect velocity on contact.

    Handles: floor, ceiling, side walls (X), back walls (Y) with goal openings.
    Modifies state.ball_pos and state.ball_vel in-place.
    """
    pos = state.ball_pos   # (E, 3)
    vel = state.ball_vel   # (E, 3)
    R = BALL_RADIUS
    rest = BALL_RESTITUTION

    # ── Floor ──
    floor_pen = R - pos[:, 2]
    floor_hit = floor_pen > 0
    pos[:, 2] = torch.where(floor_hit, torch.tensor(R, device=pos.device), pos[:, 2])
    vel[:, 2] = torch.where(floor_hit, torch.abs(vel[:, 2]) * rest, vel[:, 2])

    # Ball spin on floor (simplified: reduce horizontal vel slightly, add spin)
    # Omitted for now — minimal impact on training

    # ── Ceiling ──
    ceil_pen = pos[:, 2] + R - ARENA_HEIGHT
    ceil_hit = ceil_pen > 0
    pos[:, 2] = torch.where(ceil_hit, torch.tensor(ARENA_HEIGHT - R, device=pos.device), pos[:, 2])
    vel[:, 2] = torch.where(ceil_hit, -torch.abs(vel[:, 2]) * rest, vel[:, 2])

    # ── Side walls (X) ──
    # Right wall (+X)
    right_pen = pos[:, 0] + R - ARENA_HALF_X
    right_hit = right_pen > 0
    pos[:, 0] = torch.where(right_hit, torch.tensor(ARENA_HALF_X - R, device=pos.device), pos[:, 0])
    vel[:, 0] = torch.where(right_hit, -torch.abs(vel[:, 0]) * rest, vel[:, 0])

    # Left wall (-X)
    left_pen = -(pos[:, 0] - R) - ARENA_HALF_X
    left_hit = left_pen > 0
    pos[:, 0] = torch.where(left_hit, torch.tensor(-ARENA_HALF_X + R, device=pos.device), pos[:, 0])
    vel[:, 0] = torch.where(left_hit, torch.abs(vel[:, 0]) * rest, vel[:, 0])

    # ── Back walls (Y) with goal openings ──
    # Ball is in the goal opening if |x| < GOAL_HALF_WIDTH and z < GOAL_HEIGHT
    in_goal_x = torch.abs(pos[:, 0]) < GOAL_HALF_WIDTH
    in_goal_z = pos[:, 2] < GOAL_HEIGHT

    # Orange end (+Y)
    orange_pen = pos[:, 1] + R - ARENA_HALF_Y
    orange_hit = orange_pen > 0
    in_goal = in_goal_x & in_goal_z
    # If in goal opening: don't bounce, let ball enter goal
    # If not in goal opening: bounce off back wall
    bounce_orange = orange_hit & ~in_goal
    pos[:, 1] = torch.where(bounce_orange, torch.tensor(ARENA_HALF_Y - R, device=pos.device), pos[:, 1])
    vel[:, 1] = torch.where(bounce_orange, -torch.abs(vel[:, 1]) * rest, vel[:, 1])

    # Blue end (-Y)
    blue_pen = -(pos[:, 1] - R) - ARENA_HALF_Y
    blue_hit = blue_pen > 0
    bounce_blue = blue_hit & ~in_goal
    pos[:, 1] = torch.where(bounce_blue, torch.tensor(-ARENA_HALF_Y + R, device=pos.device), pos[:, 1])
    vel[:, 1] = torch.where(bounce_blue, torch.abs(vel[:, 1]) * rest, vel[:, 1])

    # ── Goal back net ──
    # If ball is past goal line and inside goal opening, clamp to net
    past_orange = pos[:, 1] > BACK_NET_Y - R
    if past_orange.any():
        pos[:, 1] = torch.where(past_orange, torch.tensor(BACK_NET_Y - R, device=pos.device), pos[:, 1])
        vel[:, 1] = torch.where(past_orange, -torch.abs(vel[:, 1]) * 0.3, vel[:, 1])

    past_blue = pos[:, 1] < -(BACK_NET_Y - R)
    if past_blue.any():
        pos[:, 1] = torch.where(past_blue, torch.tensor(-(BACK_NET_Y - R), device=pos.device), pos[:, 1])
        vel[:, 1] = torch.where(past_blue, torch.abs(vel[:, 1]) * 0.3, vel[:, 1])


def arena_collide_cars(state):
    """Clamp car positions to arena bounds. Simpler than ball — no goal entry.

    Cars can't enter goals. Also handles ground detection.
    """
    pos = state.car_pos   # (E, 4, 3)
    vel = state.car_vel   # (E, 4, 3)
    R = CAR_EFFECTIVE_RADIUS * 0.5  # cars are flatter
    alive = (state.car_is_demoed < 0.5)

    # ── Floor (ground detection) ──
    ground_height = 17.0  # car resting height
    below_ground = pos[:, :, 2] < ground_height
    on_floor = below_ground & alive

    pos[:, :, 2] = torch.where(on_floor, torch.tensor(ground_height, device=pos.device), pos[:, :, 2])
    vel[:, :, 2] = torch.where(on_floor & (vel[:, :, 2] < 0), torch.zeros_like(vel[:, :, 2]), vel[:, :, 2])

    # Set on_ground flag
    near_ground = (pos[:, :, 2] < ground_height + 5.0) & alive
    state.car_on_ground = near_ground.float()

    # Reset jump state when landing
    landing = near_ground & (state.car_has_jumped > 0.5)
    state.car_has_jumped[landing] = 0.0
    state.car_has_flipped[landing] = 0.0
    state.car_has_flip[landing] = 1.0
    state.car_is_jumping[landing] = 0.0
    state.car_jump_timer[landing] = 0.0

    # ── Ceiling ──
    ceil_hit = pos[:, :, 2] > (ARENA_HEIGHT - 30.0)
    pos[:, :, 2] = torch.where(ceil_hit & alive, torch.tensor(ARENA_HEIGHT - 30.0, device=pos.device), pos[:, :, 2])
    vel[:, :, 2] = torch.where(ceil_hit & alive & (vel[:, :, 2] > 0), torch.zeros_like(vel[:, :, 2]), vel[:, :, 2])

    # ── Side walls (X) ──
    right_hit = pos[:, :, 0] > (ARENA_HALF_X - 50.0)
    left_hit = pos[:, :, 0] < -(ARENA_HALF_X - 50.0)
    pos[:, :, 0] = torch.where(right_hit & alive, torch.tensor(ARENA_HALF_X - 50.0, device=pos.device), pos[:, :, 0])
    pos[:, :, 0] = torch.where(left_hit & alive, torch.tensor(-(ARENA_HALF_X - 50.0), device=pos.device), pos[:, :, 0])
    vel[:, :, 0] = torch.where((right_hit & (vel[:, :, 0] > 0)) & alive, torch.zeros_like(vel[:, :, 0]), vel[:, :, 0])
    vel[:, :, 0] = torch.where((left_hit & (vel[:, :, 0] < 0)) & alive, torch.zeros_like(vel[:, :, 0]), vel[:, :, 0])

    # ── Back walls (Y) — cars can't enter goals ──
    orange_hit = pos[:, :, 1] > (ARENA_HALF_Y - 50.0)
    blue_hit = pos[:, :, 1] < -(ARENA_HALF_Y - 50.0)
    pos[:, :, 1] = torch.where(orange_hit & alive, torch.tensor(ARENA_HALF_Y - 50.0, device=pos.device), pos[:, :, 1])
    pos[:, :, 1] = torch.where(blue_hit & alive, torch.tensor(-(ARENA_HALF_Y - 50.0), device=pos.device), pos[:, :, 1])
    vel[:, :, 1] = torch.where((orange_hit & (vel[:, :, 1] > 0)) & alive, torch.zeros_like(vel[:, :, 1]), vel[:, :, 1])
    vel[:, :, 1] = torch.where((blue_hit & (vel[:, :, 1] < 0)) & alive, torch.zeros_like(vel[:, :, 1]), vel[:, :, 1])

    # ── Ground friction (simplified) ──
    on_ground = state.car_on_ground > 0.5
    # Lateral friction: reduce sideways velocity
    right_vec = torch.cross(state.car_up, state.car_fwd, dim=-1)  # (E, 4, 3)
    lateral_speed = (vel * right_vec).sum(dim=-1, keepdim=True)  # (E, 4, 1)
    friction_force = lateral_speed * right_vec * 0.3  # 30% lateral friction per tick
    vel -= friction_force * on_ground.unsqueeze(-1).float()
