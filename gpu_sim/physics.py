"""physics.py — Core physics engine: car dynamics, ball physics, jump/flip mechanics.

All operations are batched tensor ops on TensorState. A single call to
physics_tick() advances all E environments by one 120Hz sub-step.
"""

import torch
from .constants import (
    GRAVITY, CAR_MAX_SPEED, BALL_MAX_SPEED, BALL_RADIUS,
    THROTTLE_ACCEL, BRAKE_ACCEL, BOOST_ACCEL, BOOST_CONSUMPTION,
    MAX_STEER_RATE, PITCH_TORQUE, YAW_TORQUE, ROLL_TORQUE,
    ANG_VEL_DAMPING, AIR_THROTTLE_ACCEL,
    JUMP_IMPULSE, JUMP_HOLD_FORCE, JUMP_HOLD_TIME,
    FLIP_IMPULSE, FLIP_TIMER, DEMO_RESPAWN_TIME, DT,
)
from .utils import quat_integrate, quat_to_fwd_up, safe_normalize


def apply_car_controls(state, actions, dt=DT):
    """Apply throttle, steering, boost, jump/flip to all cars.

    actions: (E, 4, 8) float tensor
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        throttle/steer/pitch/yaw/roll in [-1, 1], jump/boost/handbrake in {0, 1}

    Modifies state in-place.
    """
    E = state.n_envs
    alive = (state.car_is_demoed < 0.5)  # (E, 4)

    throttle = actions[:, :, 0]   # (E, 4)
    steer = actions[:, :, 1]
    pitch_in = actions[:, :, 2]
    yaw_in = actions[:, :, 3]
    roll_in = actions[:, :, 4]
    jump = actions[:, :, 5]
    boost = actions[:, :, 6]
    # handbrake = actions[:, :, 7]  # not used in simplified physics

    on_ground = state.car_on_ground > 0.5    # (E, 4) bool
    in_air = ~on_ground

    # ── Ground physics ──
    # Forward speed along car forward direction
    fwd = state.car_fwd  # (E, 4, 3)
    forward_speed = (state.car_vel * fwd).sum(dim=-1)  # (E, 4)

    # Throttle: accelerate if same direction, brake if opposing
    opposing = (throttle * forward_speed) < 0
    accel_mag = torch.where(
        opposing,
        throttle * BRAKE_ACCEL,
        throttle * THROTTLE_ACCEL * (1.0 - torch.abs(forward_speed) / CAR_MAX_SPEED).clamp(min=0)
    )

    # Apply throttle acceleration (ground only)
    ground_accel = accel_mag.unsqueeze(-1) * fwd  # (E, 4, 3)
    state.car_vel += ground_accel * dt * on_ground.unsqueeze(-1).float()

    # Steering (ground only): change angular velocity around Z axis
    speed_factor = 1.0 - 0.5 * torch.abs(forward_speed) / CAR_MAX_SPEED
    steer_rate = steer * MAX_STEER_RATE * speed_factor  # (E, 4)
    # Set yaw angular velocity on ground
    ground_mask = on_ground.float()
    state.car_ang_vel[:, :, 2] = (
        steer_rate * ground_mask +
        state.car_ang_vel[:, :, 2] * (1.0 - ground_mask)
    )
    # Zero out pitch/roll angular velocity on ground
    state.car_ang_vel[:, :, 0] *= (1.0 - ground_mask)
    state.car_ang_vel[:, :, 1] *= (1.0 - ground_mask)

    # ── Air physics ──
    air_mask = in_air.float()  # (E, 4)

    # Air torques
    air_torque = torch.stack([
        roll_in * ROLL_TORQUE,
        pitch_in * PITCH_TORQUE,
        yaw_in * YAW_TORQUE,
    ], dim=-1)  # (E, 4, 3)

    # Angular velocity damping in air
    state.car_ang_vel *= torch.where(
        in_air.unsqueeze(-1),
        torch.tensor(ANG_VEL_DAMPING, device=state.device),
        torch.tensor(1.0, device=state.device),
    )

    # Apply air torques
    state.car_ang_vel += air_torque * dt * air_mask.unsqueeze(-1)

    # Air throttle (very weak forward push)
    air_throttle = throttle * AIR_THROTTLE_ACCEL
    state.car_vel += (air_throttle.unsqueeze(-1) * fwd * dt * air_mask.unsqueeze(-1))

    # ── Boost (works on ground and air) ──
    boost_active = (boost > 0.5) & (state.car_boost > 0.005) & alive  # (E, 4)
    boost_f = boost_active.float()
    state.car_vel += (BOOST_ACCEL * dt * boost_f.unsqueeze(-1) * fwd)
    state.car_boost -= BOOST_CONSUMPTION * dt * boost_f
    state.car_boost.clamp_(min=0.0, max=1.0)

    # ── Jump mechanics ──
    _apply_jump_flip(state, jump, pitch_in, yaw_in, dt)

    # ── Gravity ──
    state.car_vel[:, :, 2] += GRAVITY * dt

    # ── Speed cap ──
    speed = state.car_vel.norm(dim=-1, keepdim=True)  # (E, 4, 1)
    speed_limited = speed.clamp(min=1e-6)
    state.car_vel *= torch.where(
        speed > CAR_MAX_SPEED,
        CAR_MAX_SPEED / speed_limited,
        torch.ones_like(speed_limited),
    )

    # ── Zero out demoed car physics ──
    demoed = state.car_is_demoed > 0.5
    state.car_vel[demoed] = 0.0
    state.car_ang_vel[demoed] = 0.0


def _apply_jump_flip(state, jump_input, pitch_in, yaw_in, dt):
    """Handle first jump, jump hold, and dodge/flip.

    Modifies state in-place.
    """
    on_ground = state.car_on_ground > 0.5
    jump_pressed = jump_input > 0.5
    alive = state.car_is_demoed < 0.5

    # ── First jump: launch off ground ──
    can_first_jump = on_ground & jump_pressed & (state.car_has_jumped < 0.5) & alive
    if can_first_jump.any():
        up = state.car_up  # (E, 4, 3)
        impulse = up * JUMP_IMPULSE  # (E, 4, 3)
        state.car_vel += impulse * can_first_jump.unsqueeze(-1).float()
        state.car_on_ground[can_first_jump] = 0.0
        state.car_has_jumped[can_first_jump] = 1.0
        state.car_is_jumping[can_first_jump] = 1.0
        state.car_jump_timer[can_first_jump] = 0.0
        state.car_has_flip[can_first_jump] = 1.0

    # ── Jump hold bonus (sustained upward force for first 0.2s) ──
    holding_jump = (
        jump_pressed &
        (state.car_is_jumping > 0.5) &
        (state.car_jump_timer < JUMP_HOLD_TIME) &
        alive
    )
    if holding_jump.any():
        up = state.car_up
        state.car_vel += (up * JUMP_HOLD_FORCE * dt * holding_jump.unsqueeze(-1).float())

    # Release jump button → end jump hold
    released = (~jump_pressed) & (state.car_is_jumping > 0.5)
    state.car_is_jumping[released] = 0.0

    # ── Dodge/flip: second jump in air ──
    can_flip = (
        jump_pressed &
        (~on_ground) &
        (state.car_has_flip > 0.5) &
        (state.car_has_flipped < 0.5) &
        (state.car_has_jumped > 0.5) &
        (~holding_jump) &  # can't flip while still holding first jump
        alive
    )
    if can_flip.any():
        # Dodge direction from stick input
        dodge_dir_y = -pitch_in  # pitch forward = positive Y in car frame
        dodge_dir_x = yaw_in
        dodge_mag = torch.sqrt(dodge_dir_x**2 + dodge_dir_y**2 + 1e-8)

        # If no directional input → stall flip (just cancel downward velocity)
        has_dir = dodge_mag > 0.1
        dodge_dir_x = torch.where(has_dir, dodge_dir_x / dodge_mag, torch.zeros_like(dodge_dir_x))
        dodge_dir_y = torch.where(has_dir, dodge_dir_y / dodge_mag, torch.zeros_like(dodge_dir_y))

        # Convert to world-space dodge impulse using car orientation
        fwd = state.car_fwd  # (E, 4, 3)
        right = torch.cross(state.car_up, fwd, dim=-1)  # (E, 4, 3)
        right = safe_normalize(right)

        # World-space dodge direction
        world_dodge = (fwd * dodge_dir_y.unsqueeze(-1) +
                       right * dodge_dir_x.unsqueeze(-1))  # (E, 4, 3)

        flip_mask = (can_flip & has_dir).unsqueeze(-1).float()
        state.car_vel[:, :, :2] += (world_dodge[:, :, :2] * FLIP_IMPULSE * flip_mask[:, :, :2])

        # Cancel downward velocity on flip
        cancel_mask = can_flip & (state.car_vel[:, :, 2] < 0)
        state.car_vel[:, :, 2] = torch.where(
            cancel_mask, torch.zeros_like(state.car_vel[:, :, 2]), state.car_vel[:, :, 2])

        state.car_has_flipped[can_flip] = 1.0
        state.car_has_flip[can_flip] = 0.0
        state.car_is_jumping[can_flip] = 0.0

    # ── Update jump timer ──
    has_jumped = state.car_has_jumped > 0.5
    state.car_jump_timer += dt * has_jumped.float()

    # Expire flip after 1.25s
    flip_expired = has_jumped & (~on_ground) & (state.car_jump_timer > FLIP_TIMER) & (state.car_has_flipped < 0.5)
    state.car_has_flip[flip_expired] = 0.0


def integrate_positions(state, dt=DT):
    """Update positions from velocities and orientations from angular velocities."""
    alive = (state.car_is_demoed < 0.5).unsqueeze(-1).float()

    # Car position
    state.car_pos += state.car_vel * dt * alive

    # Car orientation (quaternion integration from angular velocity)
    state.car_quat = quat_integrate(state.car_quat, state.car_ang_vel, dt)

    # Ball position
    state.ball_vel[:, 2] += GRAVITY * dt
    # Ball air drag (approximate)
    drag_factor = 1.0 - 0.0305 * dt  # ~0.03 per second
    state.ball_vel *= drag_factor
    # Ball speed cap
    ball_speed = state.ball_vel.norm(dim=-1, keepdim=True)
    state.ball_vel *= torch.where(
        ball_speed > BALL_MAX_SPEED,
        BALL_MAX_SPEED / ball_speed.clamp(min=1e-6),
        torch.ones_like(ball_speed),
    )
    state.ball_pos += state.ball_vel * dt


def update_rotation_vectors(state):
    """Recompute car_fwd and car_up from car_quat."""
    fwd, up = quat_to_fwd_up(state.car_quat)
    state.car_fwd = fwd
    state.car_up = up


def update_demoed_cars(state, dt=DT):
    """Countdown demo respawn timers and respawn cars."""
    demoed = state.car_is_demoed > 0.5
    if not demoed.any():
        return

    state.car_demoed_timer -= dt * demoed.float()
    respawn = demoed & (state.car_demoed_timer <= 0)
    if not respawn.any():
        return

    # Respawn at random-ish positions (center of own half)
    state.car_is_demoed[respawn] = 0.0
    state.car_demoed_timer[respawn] = 0.0
    state.car_boost[respawn] = 0.33

    # Respawn positions: blue at (0, -2500, 17), orange at (0, 2500, 17)
    # Add small random offset to avoid overlap
    is_blue = state.car_team == 0  # (E, 4)
    respawn_blue = respawn & is_blue
    respawn_orange = respawn & (~is_blue)

    if respawn_blue.any():
        n = respawn_blue.sum().item()
        pos = torch.tensor([0.0, -2500.0, 17.0], device=state.device).expand(n, -1).clone()
        pos[:, 0] += torch.rand(n, device=state.device) * 200 - 100
        state.car_pos[respawn_blue] = pos
        state.car_vel[respawn_blue] = 0.0
        state.car_ang_vel[respawn_blue] = 0.0

    if respawn_orange.any():
        n = respawn_orange.sum().item()
        pos = torch.tensor([0.0, 2500.0, 17.0], device=state.device).expand(n, -1).clone()
        pos[:, 0] += torch.rand(n, device=state.device) * 200 - 100
        state.car_pos[respawn_orange] = pos
        state.car_vel[respawn_orange] = 0.0
        state.car_ang_vel[respawn_orange] = 0.0

    # Reset jump/flip state
    state.car_has_jumped[respawn] = 0.0
    state.car_has_flipped[respawn] = 0.0
    state.car_has_flip[respawn] = 1.0
    state.car_is_jumping[respawn] = 0.0
    state.car_jump_timer[respawn] = 0.0
    state.car_on_ground[respawn] = 1.0

    # Reset orientation to upright, facing appropriate direction
    state.car_quat[respawn] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)
