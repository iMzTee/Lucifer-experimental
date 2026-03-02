"""arena.py — Arena geometry: wall, floor, ceiling collision for ball and cars.

Simplified model: axis-aligned box with goal openings on ±Y walls.
Curved corners are approximated as flat walls.
"""

import torch
from .constants import (
    ARENA_HALF_X, ARENA_HALF_Y, ARENA_HEIGHT,
    BALL_RADIUS, BALL_RESTITUTION, BALL_FRICTION, BALL_MAX_ANG_SPEED,
    CAR_EFFECTIVE_RADIUS,
    GOAL_HALF_WIDTH, GOAL_HEIGHT, GOAL_DEPTH, BACK_NET_Y,
    LAT_FRICTION_CURVE_X, LAT_FRICTION_CURVE_Y, HANDBRAKE_LAT_FRICTION_FACTOR,
)
from .utils import piecewise_linear

# ── Cached constants (created once on first use, avoid per-tick GPU alloc) ──
_surface_normals = None
_default_normal = None
_friction_curves = {}


def _get_cached_tensors(device):
    """Lazily create and cache constant tensors on the correct device."""
    global _surface_normals, _default_normal
    if _surface_normals is None or _surface_normals.device != device:
        _surface_normals = torch.tensor([
            [ 0.0,  0.0,  1.0],  # floor
            [ 0.0,  0.0, -1.0],  # ceiling
            [-1.0,  0.0,  0.0],  # right wall (+X)
            [ 1.0,  0.0,  0.0],  # left wall (-X)
            [ 0.0, -1.0,  0.0],  # orange wall (+Y)
            [ 0.0,  1.0,  0.0],  # blue wall (-Y)
        ], device=device)
        _default_normal = torch.tensor([0.0, 0.0, 1.0], device=device)
    return _surface_normals, _default_normal


def _get_friction_curves(device):
    """Lazily create and cache friction curve tensors."""
    if device not in _friction_curves:
        _friction_curves[device] = {
            'lat_x': torch.tensor(LAT_FRICTION_CURVE_X, device=device),
            'lat_y': torch.tensor(LAT_FRICTION_CURVE_Y, device=device),
        }
    return _friction_curves[device]


def arena_collide_ball(state):
    """Clamp ball position to arena bounds, reflect velocity on contact.

    Handles: floor, ceiling, side walls (X), back walls (Y) with goal openings.
    Modifies state.ball_pos and state.ball_vel in-place.
    """
    pos = state.ball_pos   # (E, 3)
    vel = state.ball_vel   # (E, 3)
    R = BALL_RADIUS
    rest = BALL_RESTITUTION

    # Pre-compute constant clamp values (Python floats — no GPU alloc)
    ball_floor = R
    ball_ceil = ARENA_HEIGHT - R
    ball_right = ARENA_HALF_X - R
    ball_left = -(ARENA_HALF_X - R)
    ball_orange = ARENA_HALF_Y - R
    ball_blue = -(ARENA_HALF_Y - R)
    net_orange = BACK_NET_Y - R
    net_blue = -(BACK_NET_Y - R)

    ang_vel = state.ball_ang_vel  # (E, 3)

    # ── Floor ──
    floor_hit = pos[:, 2] < ball_floor
    pos[:, 2] = torch.where(floor_hit, ball_floor, pos[:, 2])
    vel[:, 2] = torch.where(floor_hit, torch.abs(vel[:, 2]) * rest, vel[:, 2])
    # Floor friction: normal = +Z, tangential = XY
    if floor_hit.any():
        _apply_ball_surface_friction(vel, ang_vel, floor_hit, 0, 1, 2, R)

    # ── Ceiling ──
    ceil_hit = pos[:, 2] > ball_ceil
    pos[:, 2] = torch.where(ceil_hit, ball_ceil, pos[:, 2])
    vel[:, 2] = torch.where(ceil_hit, -torch.abs(vel[:, 2]) * rest, vel[:, 2])
    if ceil_hit.any():
        _apply_ball_surface_friction(vel, ang_vel, ceil_hit, 0, 1, 2, R)

    # ── Side walls (X) ──
    right_hit = pos[:, 0] > ball_right
    pos[:, 0] = torch.where(right_hit, ball_right, pos[:, 0])
    vel[:, 0] = torch.where(right_hit, -torch.abs(vel[:, 0]) * rest, vel[:, 0])
    if right_hit.any():
        _apply_ball_surface_friction(vel, ang_vel, right_hit, 1, 2, 0, R)

    left_hit = pos[:, 0] < ball_left
    pos[:, 0] = torch.where(left_hit, ball_left, pos[:, 0])
    vel[:, 0] = torch.where(left_hit, torch.abs(vel[:, 0]) * rest, vel[:, 0])
    if left_hit.any():
        _apply_ball_surface_friction(vel, ang_vel, left_hit, 1, 2, 0, R)

    # ── Back walls (Y) with goal openings ──
    in_goal_x = torch.abs(pos[:, 0]) < GOAL_HALF_WIDTH
    in_goal_z = pos[:, 2] < GOAL_HEIGHT
    in_goal = in_goal_x & in_goal_z

    # Orange end (+Y)
    bounce_orange = (pos[:, 1] > ball_orange) & ~in_goal
    pos[:, 1] = torch.where(bounce_orange, ball_orange, pos[:, 1])
    vel[:, 1] = torch.where(bounce_orange, -torch.abs(vel[:, 1]) * rest, vel[:, 1])
    if bounce_orange.any():
        _apply_ball_surface_friction(vel, ang_vel, bounce_orange, 0, 2, 1, R)

    # Blue end (-Y)
    bounce_blue = (pos[:, 1] < ball_blue) & ~in_goal
    pos[:, 1] = torch.where(bounce_blue, ball_blue, pos[:, 1])
    vel[:, 1] = torch.where(bounce_blue, torch.abs(vel[:, 1]) * rest, vel[:, 1])
    if bounce_blue.any():
        _apply_ball_surface_friction(vel, ang_vel, bounce_blue, 0, 2, 1, R)

    # ── Goal back net ──
    past_orange = pos[:, 1] > net_orange
    if past_orange.any():
        pos[:, 1] = torch.where(past_orange, net_orange, pos[:, 1])
        vel[:, 1] = torch.where(past_orange, -torch.abs(vel[:, 1]) * 0.3, vel[:, 1])

    past_blue = pos[:, 1] < net_blue
    if past_blue.any():
        pos[:, 1] = torch.where(past_blue, net_blue, pos[:, 1])
        vel[:, 1] = torch.where(past_blue, torch.abs(vel[:, 1]) * 0.3, vel[:, 1])

    # ── Ball angular velocity cap ──
    ball_ang_speed = ang_vel.norm(dim=-1, keepdim=True)  # (E, 1)
    ang_vel *= torch.where(
        ball_ang_speed > BALL_MAX_ANG_SPEED,
        BALL_MAX_ANG_SPEED / ball_ang_speed.clamp(min=1e-6),
        1.0,
    )


def _apply_ball_surface_friction(vel, ang_vel, hit_mask, t0, t1, n_axis, radius):
    """Apply friction to ball on surface contact, reducing tangential velocity and inducing spin.

    t0, t1: indices of tangential axes
    n_axis: index of the normal axis
    radius: ball radius
    """
    # Tangential velocity components
    friction_amount = BALL_FRICTION
    # Reduce tangential velocity by friction
    scale = max(1.0 - friction_amount, 0.0)
    old_t0 = vel[:, t0].clone()
    old_t1 = vel[:, t1].clone()
    vel[:, t0] = torch.where(hit_mask, vel[:, t0] * scale, vel[:, t0])
    vel[:, t1] = torch.where(hit_mask, vel[:, t1] * scale, vel[:, t1])
    # Induce spin from friction (angular velocity from tangential velocity change)
    dv_t0 = old_t0 - vel[:, t0]  # velocity removed
    dv_t1 = old_t1 - vel[:, t1]
    spin_scale = 1.0 / (radius + 1e-6)
    ang_vel[:, t1] += torch.where(hit_mask, dv_t0 * spin_scale, torch.zeros_like(dv_t0))
    ang_vel[:, t0] -= torch.where(hit_mask, dv_t1 * spin_scale, torch.zeros_like(dv_t1))


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

    # Pre-compute limits (Python floats — no GPU alloc)
    ceil_limit = ARENA_HEIGHT - CEIL_MARGIN
    right_limit = ARENA_HALF_X - WALL_MARGIN
    left_limit = -(ARENA_HALF_X - WALL_MARGIN)
    orange_limit = ARENA_HALF_Y - WALL_MARGIN
    blue_limit = -(ARENA_HALF_Y - WALL_MARGIN)

    # ── Clamp positions to arena bounds ──
    # Floor
    floor_hit = pos[:, :, 2] < FLOOR_HEIGHT
    pos[:, :, 2] = torch.where(floor_hit & alive, FLOOR_HEIGHT, pos[:, :, 2])
    vel[:, :, 2] = torch.where(floor_hit & alive & (vel[:, :, 2] < 0), 0.0, vel[:, :, 2])

    # Ceiling
    ceil_hit = pos[:, :, 2] > ceil_limit
    pos[:, :, 2] = torch.where(ceil_hit & alive, ceil_limit, pos[:, :, 2])
    vel[:, :, 2] = torch.where(ceil_hit & alive & (vel[:, :, 2] > 0), 0.0, vel[:, :, 2])

    # Right wall (+X)
    right_hit = pos[:, :, 0] > right_limit
    pos[:, :, 0] = torch.where(right_hit & alive, right_limit, pos[:, :, 0])
    vel[:, :, 0] = torch.where(right_hit & alive & (vel[:, :, 0] > 0), 0.0, vel[:, :, 0])

    # Left wall (-X)
    left_hit = pos[:, :, 0] < left_limit
    pos[:, :, 0] = torch.where(left_hit & alive, left_limit, pos[:, :, 0])
    vel[:, :, 0] = torch.where(left_hit & alive & (vel[:, :, 0] < 0), 0.0, vel[:, :, 0])

    # Orange back wall (+Y)
    orange_hit = pos[:, :, 1] > orange_limit
    pos[:, :, 1] = torch.where(orange_hit & alive, orange_limit, pos[:, :, 1])
    vel[:, :, 1] = torch.where(orange_hit & alive & (vel[:, :, 1] > 0), 0.0, vel[:, :, 1])

    # Blue back wall (-Y)
    blue_hit = pos[:, :, 1] < blue_limit
    pos[:, :, 1] = torch.where(blue_hit & alive, blue_limit, pos[:, :, 1])
    vel[:, :, 1] = torch.where(blue_hit & alive & (vel[:, :, 1] < 0), 0.0, vel[:, :, 1])

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

    # Surface normals (cached — no per-tick GPU alloc)
    surface_normals, default_normal = _get_cached_tensors(pos.device)

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
        default_normal,
    )

    # ── Landing detection: reset jump state when touching any surface ──
    was_airborne = state.car_has_jumped > 0.5
    landing = on_surface & was_airborne
    state.car_has_jumped[landing] = 0.0
    state.car_has_flipped[landing] = 0.0
    state.car_has_flip[landing] = 1.0
    state.car_is_jumping[landing] = 0.0
    state.car_jump_timer[landing] = 0.0

    # Reset flip torque state on landing
    state.car_is_flipping[landing] = 0.0
    state.car_flip_time[landing] = 0.0
    state.car_flip_rel_torque[landing] = 0.0

    # ── Slip-angle lateral friction model ──
    on_ground_mask = state.car_on_ground > 0.5
    fwd = state.car_fwd  # (E, A, 3)
    right_vec = torch.cross(state.car_up, fwd, dim=-1)  # (E, A, 3)
    right_vec = right_vec / (right_vec.norm(dim=-1, keepdim=True) + 1e-6)

    # Project velocity onto forward and lateral
    fwd_speed = (vel * fwd).sum(dim=-1)        # (E, A)
    lat_speed = (vel * right_vec).sum(dim=-1)   # (E, A)
    abs_fwd = torch.abs(fwd_speed)
    abs_lat = torch.abs(lat_speed)

    # Compute slip ratio: |lat| / (|fwd| + |lat|)
    total_speed = abs_fwd + abs_lat + 1e-6
    slip_ratio = abs_lat / total_speed  # (E, A) in [0, 1]

    # Look up friction factor from slip-angle curve
    fc = _get_friction_curves(pos.device)
    friction_factor = piecewise_linear(slip_ratio, fc['lat_x'], fc['lat_y'])  # (E, A)

    # Handbrake reduces lateral friction
    hb_val = state.car_handbrake_val  # (E, A) analog 0-1
    handbrake_mult = 1.0 - hb_val * (1.0 - HANDBRAKE_LAT_FRICTION_FACTOR)
    friction_factor = friction_factor * handbrake_mult

    # Apply: remove friction_factor fraction of lateral velocity
    lat_removal = lat_speed * friction_factor  # (E, A)
    vel -= (lat_removal.unsqueeze(-1) * right_vec) * on_ground_mask.unsqueeze(-1).float()
