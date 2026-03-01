"""collision.py — Ball-car and car-car collision detection + response.

Simplified sphere-sphere model for ball-car collision.
Pair-wise checks for car-car demos and bumps.
Supports variable n_agents (1v0, 1v1, 2v2).
"""

import torch
from .constants import (
    BALL_RADIUS, CAR_EFFECTIVE_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED,
    CAR_SUPERSONIC_SPEED, DEMO_RESPAWN_TIME,
)


def ball_car_collision(state):
    """Detect and resolve ball-car collisions using sphere-sphere model.

    Returns: (E, A) float tensor of touches (1.0 where car touched ball).
    Modifies state.ball_pos, state.ball_vel in-place.
    """
    A = state.n_agents

    # Expand ball for broadcasting: (E, 1, 3)
    ball_pos = state.ball_pos.unsqueeze(1)  # (E, 1, 3)
    ball_vel = state.ball_vel.unsqueeze(1)  # (E, 1, 3)

    # Offset car position to hitbox center (approximate)
    car_center = state.car_pos + state.car_fwd * 13.88  # hitbox offset
    car_center[:, :, 2] += 20.75  # height offset

    # Distance check
    diff = ball_pos - car_center  # (E, A, 3)
    dist = diff.norm(dim=-1)  # (E, A)
    contact_dist = BALL_RADIUS + CAR_EFFECTIVE_RADIUS

    colliding = dist < contact_dist  # (E, A)

    # Alive check
    alive = state.car_is_demoed < 0.5  # (E, A)
    colliding = colliding & alive

    if not colliding.any():
        return torch.zeros_like(dist)

    # Collision normal (ball - car direction)
    normal = diff / (dist.unsqueeze(-1) + 1e-6)  # (E, A, 3)

    # Relative velocity (ball - car)
    rel_vel = ball_vel - state.car_vel  # (E, A, 3)
    rel_speed = (rel_vel * normal).sum(dim=-1)  # (E, A)

    # Only collide if approaching
    approaching = rel_speed < 0
    active = colliding & approaching  # (E, A)

    if not active.any():
        return colliding.float()

    # Impulse: ball is lighter, so it gets most of the velocity change
    impulse_mag = -1.8 * rel_speed  # (E, A)
    impulse = impulse_mag.unsqueeze(-1) * normal  # (E, A, 3)

    # Apply impulse to ball (sum across all colliding cars)
    active_f = active.unsqueeze(-1).float()  # (E, A, 1)
    ball_impulse = (impulse * active_f).sum(dim=1)  # (E, 3)
    state.ball_vel += ball_impulse

    # Ball speed cap after collision
    ball_speed = state.ball_vel.norm(dim=-1, keepdim=True)
    state.ball_vel *= torch.where(
        ball_speed > BALL_MAX_SPEED,
        BALL_MAX_SPEED / ball_speed.clamp(min=1e-6),
        torch.ones_like(ball_speed),
    )

    # Separate ball from car (push ball out of overlap)
    overlap = (contact_dist - dist).clamp(min=0)  # (E, A)
    push = normal * overlap.unsqueeze(-1) * active_f  # (E, A, 3)
    state.ball_pos += push.sum(dim=1)  # (E, 3)

    # Small pushback on car too (reaction force)
    car_pushback = -normal * overlap.unsqueeze(-1) * active_f * 0.15
    state.car_vel += car_pushback * 50.0 * (1.0 / 120.0)

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

        # Demo check: opposite teams + supersonic speed
        diff_team = state.car_team[:, i] != state.car_team[:, j]
        speed_i = state.car_vel[:, i].norm(dim=-1)
        speed_j = state.car_vel[:, j].norm(dim=-1)

        normal = diff / (dist.unsqueeze(-1) + 1e-6)
        approaching_i = (state.car_vel[:, i] * (-normal)).sum(dim=-1) > 0
        approaching_j = (state.car_vel[:, j] * normal).sum(dim=-1) > 0

        demo_j = colliding & diff_team & (speed_i > CAR_SUPERSONIC_SPEED) & approaching_i
        demo_i = colliding & diff_team & (speed_j > CAR_SUPERSONIC_SPEED) & approaching_j

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

        # Bump physics (non-demo collisions)
        bump = colliding & ~demo_i & ~demo_j
        if bump.any():
            normal_dir = diff / (dist.unsqueeze(-1) + 1e-6)
            rel_vel = state.car_vel[:, i] - state.car_vel[:, j]
            rel_speed_normal = (rel_vel * normal_dir).sum(dim=-1)

            approaching = rel_speed_normal < 0
            active_bump = bump & approaching

            if active_bump.any():
                bump_impulse = -0.8 * rel_speed_normal.unsqueeze(-1) * normal_dir
                bump_f = active_bump.unsqueeze(-1).float()
                state.car_vel[:, i] += bump_impulse * 0.5 * bump_f
                state.car_vel[:, j] -= bump_impulse * 0.5 * bump_f

                overlap = (contact_dist - dist).clamp(min=0)
                sep = normal_dir * overlap.unsqueeze(-1) * 0.5 * bump_f
                state.car_pos[:, i] += sep
                state.car_pos[:, j] -= sep
