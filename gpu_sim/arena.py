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
    """Clamp car positions to arena bounds with multi-surface detection.

    Cars can drive on floor, walls, and ceiling. Surface detection sets
    car_on_ground and car_surface_normal for surface-relative physics.
    """
    pos = state.car_pos   # (E, A, 3)
    vel = state.car_vel   # (E, A, 3)
    alive = (state.car_is_demoed < 0.5)  # (E, A)

    # Surface contact thresholds
    WALL_MARGIN = 50.0    # distance from wall to detect contact
    FLOOR_HEIGHT = 17.0   # car resting height on floor
    CEIL_MARGIN = 30.0    # distance from ceiling
    SURFACE_TOL = 10.0    # tolerance for "near surface" detection

    # ── Clamp positions to arena bounds ──
    # Floor
    floor_hit = pos[:, :, 2] < FLOOR_HEIGHT
    pos[:, :, 2] = torch.where(floor_hit & alive, torch.tensor(FLOOR_HEIGHT, device=pos.device), pos[:, :, 2])
    vel[:, :, 2] = torch.where(floor_hit & alive & (vel[:, :, 2] < 0), torch.zeros_like(vel[:, :, 2]), vel[:, :, 2])

    # Ceiling
    ceil_limit = ARENA_HEIGHT - CEIL_MARGIN
    ceil_hit = pos[:, :, 2] > ceil_limit
    pos[:, :, 2] = torch.where(ceil_hit & alive, torch.tensor(ceil_limit, device=pos.device), pos[:, :, 2])
    vel[:, :, 2] = torch.where(ceil_hit & alive & (vel[:, :, 2] > 0), torch.zeros_like(vel[:, :, 2]), vel[:, :, 2])

    # Right wall (+X)
    right_limit = ARENA_HALF_X - WALL_MARGIN
    right_hit = pos[:, :, 0] > right_limit
    pos[:, :, 0] = torch.where(right_hit & alive, torch.tensor(right_limit, device=pos.device), pos[:, :, 0])
    vel[:, :, 0] = torch.where(right_hit & alive & (vel[:, :, 0] > 0), torch.zeros_like(vel[:, :, 0]), vel[:, :, 0])

    # Left wall (-X)
    left_limit = -(ARENA_HALF_X - WALL_MARGIN)
    left_hit = pos[:, :, 0] < left_limit
    pos[:, :, 0] = torch.where(left_hit & alive, torch.tensor(left_limit, device=pos.device), pos[:, :, 0])
    vel[:, :, 0] = torch.where(left_hit & alive & (vel[:, :, 0] < 0), torch.zeros_like(vel[:, :, 0]), vel[:, :, 0])

    # Orange back wall (+Y)
    orange_limit = ARENA_HALF_Y - WALL_MARGIN
    orange_hit = pos[:, :, 1] > orange_limit
    pos[:, :, 1] = torch.where(orange_hit & alive, torch.tensor(orange_limit, device=pos.device), pos[:, :, 1])
    vel[:, :, 1] = torch.where(orange_hit & alive & (vel[:, :, 1] > 0), torch.zeros_like(vel[:, :, 1]), vel[:, :, 1])

    # Blue back wall (-Y)
    blue_limit = -(ARENA_HALF_Y - WALL_MARGIN)
    blue_hit = pos[:, :, 1] < blue_limit
    pos[:, :, 1] = torch.where(blue_hit & alive, torch.tensor(blue_limit, device=pos.device), pos[:, :, 1])
    vel[:, :, 1] = torch.where(blue_hit & alive & (vel[:, :, 1] < 0), torch.zeros_like(vel[:, :, 1]), vel[:, :, 1])

    # ── Multi-surface detection ──
    # Compute distance to each surface (6 surfaces)
    # Distances are positive = inside arena, smaller = closer to surface
    dist_floor   = pos[:, :, 2] - FLOOR_HEIGHT                # distance above floor
    dist_ceiling = ceil_limit - pos[:, :, 2]                   # distance below ceiling
    dist_right   = right_limit - pos[:, :, 0]                  # distance from right wall
    dist_left    = pos[:, :, 0] - left_limit                   # distance from left wall
    dist_orange  = orange_limit - pos[:, :, 1]                 # distance from orange wall
    dist_blue    = pos[:, :, 1] - blue_limit                   # distance from blue wall

    # Stack distances: (E, A, 6)
    dists = torch.stack([dist_floor, dist_ceiling, dist_right, dist_left, dist_orange, dist_blue], dim=-1)

    # Surface normals: inward-pointing (toward arena center)
    # floor=(0,0,1), ceiling=(0,0,-1), right=(-1,0,0), left=(1,0,0), orange=(0,-1,0), blue=(0,1,0)
    surface_normals = torch.tensor([
        [ 0.0,  0.0,  1.0],  # floor
        [ 0.0,  0.0, -1.0],  # ceiling
        [-1.0,  0.0,  0.0],  # right wall (+X)
        [ 1.0,  0.0,  0.0],  # left wall (-X)
        [ 0.0, -1.0,  0.0],  # orange wall (+Y)
        [ 0.0,  1.0,  0.0],  # blue wall (-Y)
    ], device=pos.device)  # (6, 3)

    # Find nearest surface per car
    nearest_idx = dists.argmin(dim=-1)  # (E, A)
    nearest_dist = dists.gather(-1, nearest_idx.unsqueeze(-1)).squeeze(-1)  # (E, A)

    # Car is on a surface if within tolerance
    on_surface = (nearest_dist < SURFACE_TOL) & alive  # (E, A)
    state.car_on_ground = on_surface.float()

    # Set surface normal for cars on surfaces
    # Index into surface_normals using nearest_idx
    nearest_normal = surface_normals[nearest_idx]  # (E, A, 3)
    # Only update surface normal for cars on a surface; in-air cars keep floor default
    state.car_surface_normal = torch.where(
        on_surface.unsqueeze(-1),
        nearest_normal,
        torch.tensor([0.0, 0.0, 1.0], device=pos.device),
    )

    # ── Landing detection: reset jump state when touching any surface ──
    was_airborne = state.car_has_jumped > 0.5
    landing = on_surface & was_airborne
    state.car_has_jumped[landing] = 0.0
    state.car_has_flipped[landing] = 0.0
    state.car_has_flip[landing] = 1.0
    state.car_is_jumping[landing] = 0.0
    state.car_jump_timer[landing] = 0.0

    # ── Surface friction (handbrake-aware) ──
    on_ground_mask = state.car_on_ground > 0.5
    # Lateral friction: reduce velocity perpendicular to forward and surface normal
    right_vec = torch.cross(state.car_up, state.car_fwd, dim=-1)  # (E, A, 3)
    lateral_speed = (vel * right_vec).sum(dim=-1, keepdim=True)  # (E, A, 1)
    # Normal: 30% lateral friction per tick; Handbrake: 5% (allows powersliding)
    handbrake_on = state.car_handbrake > 0.5  # (E, A)
    friction_coeff = torch.where(handbrake_on, torch.tensor(0.05, device=vel.device),
                                 torch.tensor(0.3, device=vel.device))  # (E, A)
    friction_force = lateral_speed * right_vec * friction_coeff.unsqueeze(-1)
    vel -= friction_force * on_ground_mask.unsqueeze(-1).float()
