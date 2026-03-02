"""collision.py — Ball-car and car-car collision detection + response.

OBB-sphere model for ball-car collision (oriented bounding box hitbox).
Pair-wise checks for car-car demos and bumps.
Supports variable n_agents (1v0, 1v1, 2v2).
"""

import torch
from .constants import (
    BALL_RADIUS, CAR_EFFECTIVE_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED,
    CAR_SUPERSONIC_SPEED, DEMO_RESPAWN_TIME,
    CAR_HITBOX_LENGTH, CAR_HITBOX_WIDTH, CAR_HITBOX_HEIGHT, CAR_HITBOX_OFFSET,
    BALL_MASS, CAR_MASS, BALL_RESTITUTION,
    BALL_CAR_EXTRA_IMPULSE_Z_SCALE, BALL_CAR_EXTRA_IMPULSE_FWD_SCALE,
    BALL_CAR_EXTRA_IMPULSE_SPEEDS, BALL_CAR_EXTRA_IMPULSE_FACTORS,
    BUMP_GROUND_SPEEDS, BUMP_GROUND_FACTORS,
    BUMP_AIR_SPEEDS, BUMP_AIR_FACTORS,
    BUMP_UPWARD_SPEEDS, BUMP_UPWARD_FACTORS,
)
from .utils import quat_rotate_vector, quat_conjugate, piecewise_linear

# ── Cached hitbox constants (created once per device) ──
_hitbox_cache = {}
_collision_curves = {}


def _get_hitbox_tensors(device):
    """Lazily create and cache hitbox constant tensors on device."""
    if device not in _hitbox_cache:
        _hitbox_cache[device] = {
            'offset': torch.tensor(CAR_HITBOX_OFFSET, dtype=torch.float32, device=device),
            'half_extents': torch.tensor([
                CAR_HITBOX_LENGTH / 2,
                CAR_HITBOX_WIDTH / 2,
                CAR_HITBOX_HEIGHT / 2,
            ], dtype=torch.float32, device=device),
        }
    return _hitbox_cache[device]


def _get_collision_curves(device):
    """Lazily create and cache collision curve tensors."""
    if device not in _collision_curves:
        _collision_curves[device] = {
            'extra_speeds': torch.tensor(BALL_CAR_EXTRA_IMPULSE_SPEEDS, device=device),
            'extra_factors': torch.tensor(BALL_CAR_EXTRA_IMPULSE_FACTORS, device=device),
            'bump_ground_speeds': torch.tensor(BUMP_GROUND_SPEEDS, device=device),
            'bump_ground_factors': torch.tensor(BUMP_GROUND_FACTORS, device=device),
            'bump_air_speeds': torch.tensor(BUMP_AIR_SPEEDS, device=device),
            'bump_air_factors': torch.tensor(BUMP_AIR_FACTORS, device=device),
            'bump_upward_speeds': torch.tensor(BUMP_UPWARD_SPEEDS, device=device),
            'bump_upward_factors': torch.tensor(BUMP_UPWARD_FACTORS, device=device),
        }
    return _collision_curves[device]


def ball_car_collision(state):
    """Detect and resolve ball-car collisions using OBB-sphere model.

    Uses the car's oriented bounding box (hitbox) for collision detection
    and proper mass-based impulse for response.

    Returns: (E, A) float tensor of touches (1.0 where car touched ball).
    Modifies state.ball_pos, state.ball_vel, state.car_vel in-place.
    """
    hb = _get_hitbox_tensors(state.device)
    hitbox_offset = hb['offset']       # (3,)
    half_extents = hb['half_extents']  # (3,)

    # Ball (E, 3) → (E, 1, 3) for agent broadcasting
    ball_pos = state.ball_pos.unsqueeze(1)  # (E, 1, 3)
    ball_vel = state.ball_vel.unsqueeze(1)  # (E, 1, 3)

    car_pos = state.car_pos   # (E, A, 3)
    car_quat = state.car_quat  # (E, A, 4)

    # ── Transform ball into car's local frame ──
    diff = ball_pos - car_pos  # (E, A, 3)
    q_inv = quat_conjugate(car_quat)  # (E, A, 4)
    local_diff = quat_rotate_vector(q_inv, diff)  # (E, A, 3)

    # Offset to hitbox center in local frame
    local_diff = local_diff - hitbox_offset  # (E, A, 3) broadcast (3,)

    # ── Closest point on OBB ──
    closest_local = local_diff.clamp(-half_extents, half_extents)  # (E, A, 3)
    local_sep = local_diff - closest_local  # (E, A, 3)

    # Transform separation back to world frame
    world_sep = quat_rotate_vector(car_quat, local_sep)  # (E, A, 3)

    dist = world_sep.norm(dim=-1)  # (E, A)
    colliding = dist < BALL_RADIUS  # (E, A)

    # Alive check
    alive = state.car_is_demoed < 0.5  # (E, A)
    colliding = colliding & alive

    if not colliding.any():
        return torch.zeros_like(dist)

    # ── Collision normal ──
    # Normal case: ball surface to closest OBB point
    normal = world_sep / (dist.unsqueeze(-1) + 1e-6)  # (E, A, 3)

    # Fallback for ball center inside OBB (dist ≈ 0): use center-to-center
    hitbox_world_offset = quat_rotate_vector(car_quat, hitbox_offset)  # (E, A, 3)
    center_diff = ball_pos - (car_pos + hitbox_world_offset)  # (E, A, 3)
    center_normal = center_diff / (center_diff.norm(dim=-1, keepdim=True) + 1e-6)
    inside_obb = dist < 1e-3
    normal = torch.where(inside_obb.unsqueeze(-1), center_normal, normal)

    # ── Relative velocity and approach check ──
    rel_vel = ball_vel - state.car_vel  # (E, A, 3)
    v_rel_n = (rel_vel * normal).sum(dim=-1)  # (E, A)

    approaching = v_rel_n < 0
    active = colliding & approaching  # (E, A)

    if not active.any():
        return colliding.float()

    # ── Mass-based impulse: j = -(1+e) * v_rel_n / (1/m_ball + 1/m_car) ──
    inv_mass_sum = 1.0 / BALL_MASS + 1.0 / CAR_MASS  # Python float
    j = -(1.0 + BALL_RESTITUTION) * v_rel_n / inv_mass_sum  # (E, A)

    ball_dv = (j / BALL_MASS).unsqueeze(-1) * normal   # (E, A, 3)
    car_dv = -(j / CAR_MASS).unsqueeze(-1) * normal    # (E, A, 3)

    # ── RocketSim extra impulse on ball ──
    curves = _get_collision_curves(state.device)
    car_speed = state.car_vel.norm(dim=-1)  # (E, A)
    extra_factor = piecewise_linear(car_speed, curves['extra_speeds'], curves['extra_factors'])

    # Forward-adjusted impulse direction
    car_fwd = state.car_fwd  # (E, A, 3)
    extra_dir = normal.clone()
    extra_dir = extra_dir + car_fwd * BALL_CAR_EXTRA_IMPULSE_FWD_SCALE
    extra_dir[:, :, 2] = extra_dir[:, :, 2] * BALL_CAR_EXTRA_IMPULSE_Z_SCALE
    extra_dir = extra_dir / (extra_dir.norm(dim=-1, keepdim=True) + 1e-6)

    extra_impulse = extra_dir * (car_speed * extra_factor).unsqueeze(-1)  # (E, A, 3)
    ball_dv = ball_dv + extra_impulse

    active_f = active.unsqueeze(-1).float()  # (E, A, 1)

    # Apply impulse to ball (sum across all colliding cars)
    state.ball_vel += (ball_dv * active_f).sum(dim=1)  # (E, 3)

    # Apply impulse to car
    state.car_vel += car_dv * active_f  # (E, A, 3)

    # ── Speed caps ──
    ball_speed = state.ball_vel.norm(dim=-1, keepdim=True)
    state.ball_vel *= torch.where(
        ball_speed > BALL_MAX_SPEED,
        BALL_MAX_SPEED / ball_speed.clamp(min=1e-6),
        torch.ones_like(ball_speed),
    )
    car_speed = state.car_vel.norm(dim=-1, keepdim=True)
    state.car_vel *= torch.where(
        car_speed > CAR_MAX_SPEED,
        CAR_MAX_SPEED / car_speed.clamp(min=1e-6),
        torch.ones_like(car_speed),
    )

    # ── Separate ball from car (push out of overlap) ──
    overlap = (BALL_RADIUS - dist).clamp(min=0)  # (E, A)
    push = normal * overlap.unsqueeze(-1) * active_f  # (E, A, 3)
    state.ball_pos += push.sum(dim=1)  # (E, 3)

    return colliding.float()


def car_car_collision(state):
    """Check car pairs for demos and bumps.

    Uses layout-defined car_pairs. No-op for 1v0 (empty pairs list).
    """
    pairs = state.layout["car_pairs"]
    if not pairs:
        return

    alive = state.car_is_demoed < 0.5  # (E, A)

    for i, j in pairs:
        both_alive = alive[:, i] & alive[:, j]
        if not both_alive.any():
            continue

        diff = state.car_pos[:, i] - state.car_pos[:, j]  # (E, 3)
        dist = diff.norm(dim=-1)  # (E,)
        contact_dist = CAR_EFFECTIVE_RADIUS * 2.0
        colliding = (dist < contact_dist) & both_alive

        if not colliding.any():
            continue

        # Demo check: opposite teams + supersonic state (uses hysteresis tracking)
        diff_team = state.car_team[:, i] != state.car_team[:, j]
        speed_i = state.car_vel[:, i].norm(dim=-1)
        speed_j = state.car_vel[:, j].norm(dim=-1)

        normal = diff / (dist.unsqueeze(-1) + 1e-6)
        approaching_i = (state.car_vel[:, i] * (-normal)).sum(dim=-1) > 0
        approaching_j = (state.car_vel[:, j] * normal).sum(dim=-1) > 0

        # Use supersonic state flag (with hysteresis) for demo eligibility
        ss_i = state.car_is_supersonic[:, i] > 0.5
        ss_j = state.car_is_supersonic[:, j] > 0.5
        demo_j = colliding & diff_team & ss_i & approaching_i
        demo_i = colliding & diff_team & ss_j & approaching_j

        if demo_j.any():
            state.car_is_demoed[:, j] = torch.where(demo_j, torch.ones_like(state.car_is_demoed[:, j]),
                                                     state.car_is_demoed[:, j])
            state.car_demoed_timer[:, j] = torch.where(demo_j, torch.tensor(DEMO_RESPAWN_TIME, device=state.device),
                                                        state.car_demoed_timer[:, j])
            state.car_vel[demo_j, j] = 0.0
            state.match_demos[:, i] += demo_j.float()

        if demo_i.any():
            state.car_is_demoed[:, i] = torch.where(demo_i, torch.ones_like(state.car_is_demoed[:, i]),
                                                     state.car_is_demoed[:, i])
            state.car_demoed_timer[:, i] = torch.where(demo_i, torch.tensor(DEMO_RESPAWN_TIME, device=state.device),
                                                        state.car_demoed_timer[:, i])
            state.car_vel[demo_i, i] = 0.0
            state.match_demos[:, j] += demo_i.float()

        # Bump physics (non-demo collisions) with speed-dependent curves
        bump = colliding & ~demo_i & ~demo_j
        if bump.any():
            curves = _get_collision_curves(state.device)
            normal_dir = diff / (dist.unsqueeze(-1) + 1e-6)
            rel_vel = state.car_vel[:, i] - state.car_vel[:, j]
            rel_speed_normal = (rel_vel * normal_dir).sum(dim=-1)

            approaching = rel_speed_normal < 0
            active_bump = bump & approaching

            if active_bump.any():
                # Speed-dependent bump factor
                avg_speed = (speed_i + speed_j) * 0.5
                on_ground_i = state.car_on_ground[:, i] > 0.5
                on_ground_j = state.car_on_ground[:, j] > 0.5
                both_ground = on_ground_i & on_ground_j

                # Choose curve based on ground/air state and normal direction
                upward = normal_dir[:, 2] > 0.5
                ground_factor = piecewise_linear(avg_speed, curves['bump_ground_speeds'], curves['bump_ground_factors'])
                air_factor = piecewise_linear(avg_speed, curves['bump_air_speeds'], curves['bump_air_factors'])
                upward_factor = piecewise_linear(avg_speed, curves['bump_upward_speeds'], curves['bump_upward_factors'])

                bump_factor = torch.where(both_ground, ground_factor,
                              torch.where(upward, upward_factor, air_factor))

                bump_impulse = -rel_speed_normal.unsqueeze(-1) * normal_dir * bump_factor.unsqueeze(-1)
                bump_f = active_bump.unsqueeze(-1).float()
                state.car_vel[:, i] += bump_impulse * 0.5 * bump_f
                state.car_vel[:, j] -= bump_impulse * 0.5 * bump_f

                overlap = (contact_dist - dist).clamp(min=0)
                sep = normal_dir * overlap.unsqueeze(-1) * 0.5 * bump_f
                state.car_pos[:, i] += sep
                state.car_pos[:, j] -= sep
