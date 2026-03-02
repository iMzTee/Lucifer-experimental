"""physics.py — Core physics engine: car dynamics, ball physics, jump/flip mechanics.

All operations are batched tensor ops on TensorState. A single call to
physics_tick() advances all E environments by one 120Hz sub-step.

Physics values sourced from RocketSim (github.com/ZealanL/RocketSim).
"""

import torch
from .constants import (
    GRAVITY, CAR_MAX_SPEED, BALL_MAX_SPEED, BALL_DRAG,
    THROTTLE_ACCEL, BRAKE_ACCEL, BOOST_ACCEL_GROUND, BOOST_ACCEL_AIR,
    BOOST_CONSUMPTION, COASTING_BRAKE_FACTOR, STOPPING_SPEED,
    CAR_WHEELBASE, CAR_MAX_ANG_SPEED,
    THROTTLE_TORQUE_SPEEDS, THROTTLE_TORQUE_FACTORS,
    STEER_ANGLE_SPEEDS, STEER_ANGLE_VALUES,
    POWERSLIDE_STEER_SPEEDS, POWERSLIDE_STEER_VALUES,
    HANDBRAKE_RISE_RATE, HANDBRAKE_FALL_RATE,
    PITCH_TORQUE, YAW_TORQUE, ROLL_TORQUE,
    PITCH_ANG_DAMPING, YAW_ANG_DAMPING, ROLL_ANG_DAMPING,
    AIR_THROTTLE_ACCEL, CAR_AUTOROLL_TORQUE,
    STICKY_FORCE_GROUND,
    JUMP_IMPULSE, JUMP_HOLD_FORCE, JUMP_HOLD_TIME,
    JUMP_MIN_TIME, JUMP_MIN_FORCE_SCALE,
    FLIP_IMPULSE, FLIP_FORWARD_SCALE, FLIP_SIDE_SCALE,
    FLIP_BACKWARD_SCALE, FLIP_BACKWARD_X_SCALE, FLIP_Z_DAMP,
    FLIP_Z_DAMP_PER_TICK, FLIP_Z_DAMP_START, FLIP_Z_DAMP_END,
    FLIP_TIMER, FLIP_TORQUE_X, FLIP_TORQUE_Y, FLIP_TORQUE_TIME,
    FLIP_PITCH_LOCK_TIME,
    CAR_AUTOFLIP_ROLL_THRESH, CAR_AUTOFLIP_TORQUE, CAR_AUTOFLIP_TIME,
    CAR_SUPERSONIC_ACTIVATE, CAR_SUPERSONIC_MAINTAIN, CAR_SUPERSONIC_MAINTAIN_TIME,
    DEMO_RESPAWN_TIME, DT,
)
from .utils import (
    quat_integrate, quat_to_fwd_up, safe_normalize,
    quat_from_axis_angle, quat_multiply, quat_normalize,
    piecewise_linear,
)

# ── Cached curve tensors (created once per device, avoid per-tick GPU alloc) ──
_curve_cache = {}


def _get_curves(device):
    """Lazily create and cache piecewise-linear curve tensors on device."""
    if device not in _curve_cache:
        _curve_cache[device] = {
            'throttle_speeds': torch.tensor(THROTTLE_TORQUE_SPEEDS, device=device),
            'throttle_factors': torch.tensor(THROTTLE_TORQUE_FACTORS, device=device),
            'steer_speeds': torch.tensor(STEER_ANGLE_SPEEDS, device=device),
            'steer_angles': torch.tensor(STEER_ANGLE_VALUES, device=device),
            'powerslide_steer_speeds': torch.tensor(POWERSLIDE_STEER_SPEEDS, device=device),
            'powerslide_steer_angles': torch.tensor(POWERSLIDE_STEER_VALUES, device=device),
        }
    return _curve_cache[device]


_piecewise_linear = piecewise_linear  # local alias


def apply_car_controls(state, actions, dt=DT):
    """Apply throttle, steering, boost, jump/flip to all cars.

    actions: (E, A, 8) float tensor
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        throttle/steer/pitch/yaw/roll in [-1, 1], jump/boost/handbrake in {0, 1}

    Modifies state in-place.
    """
    alive = (state.car_is_demoed < 0.5)  # (E, A)
    curves = _get_curves(state.device)

    throttle = actions[:, :, 0]   # (E, A)
    steer = actions[:, :, 1]
    pitch_in = actions[:, :, 2]
    yaw_in = actions[:, :, 3]
    roll_in = actions[:, :, 4]
    jump = actions[:, :, 5]
    boost = actions[:, :, 6]
    handbrake = actions[:, :, 7]
    state.car_handbrake = handbrake

    # ── Handbrake analog lerp (rise at 5/s, fall at 2/s) ──
    hb_val = state.car_handbrake_val
    hb_rising = handbrake > 0.5
    hb_val = torch.where(
        hb_rising,
        (hb_val + HANDBRAKE_RISE_RATE * dt).clamp(max=1.0),
        (hb_val - HANDBRAKE_FALL_RATE * dt).clamp(min=0.0),
    )
    state.car_handbrake_val = hb_val

    on_ground = state.car_on_ground > 0.5    # (E, A) bool
    in_air = ~on_ground

    # ── Ground physics ──
    fwd = state.car_fwd  # (E, A, 3)
    forward_speed = (state.car_vel * fwd).sum(dim=-1)  # (E, A)
    abs_speed = torch.abs(forward_speed)

    # ── Throttle with speed-dependent torque factor ──
    # Torque factor drops from 1.0 at 0 speed to 0.0 at 1410 uu/s
    torque_factor = _piecewise_linear(
        abs_speed, curves['throttle_speeds'], curves['throttle_factors'])
    opposing = (throttle * forward_speed) < 0
    accel_mag = torch.where(
        opposing,
        throttle * BRAKE_ACCEL,
        throttle * THROTTLE_ACCEL * torque_factor,
    )
    ground_accel = accel_mag.unsqueeze(-1) * fwd  # (E, A, 3)
    state.car_vel += ground_accel * dt * on_ground.unsqueeze(-1).float()

    # ── Coasting deceleration (no throttle → friction slows car) ──
    coast_mask = (torch.abs(throttle) < 0.01) & on_ground
    coast_accel = COASTING_BRAKE_FACTOR * BRAKE_ACCEL
    coast_amount = coast_accel * dt
    # Full stop when speed < 25 uu/s
    full_stop = coast_mask & (abs_speed < STOPPING_SPEED)
    state.car_vel[full_stop] = 0.0
    # Decelerate when above stopping speed
    coast_decel_mask = coast_mask & (abs_speed >= STOPPING_SPEED)
    coast_decel = torch.sign(forward_speed) * torch.clamp(abs_speed, max=coast_amount)
    state.car_vel -= (coast_decel * coast_decel_mask.float()).unsqueeze(-1) * fwd

    # ── Steering: speed-dependent steer angle curve ──
    steer_angle = _piecewise_linear(
        abs_speed, curves['steer_speeds'], curves['steer_angles'])
    handbrake_active = handbrake > 0.5  # (E, A)
    if handbrake_active.any():
        powerslide_angle = _piecewise_linear(
            abs_speed, curves['powerslide_steer_speeds'], curves['powerslide_steer_angles'])
        steer_angle = torch.where(handbrake_active, powerslide_angle, steer_angle)

    steer_rate = forward_speed * steer_angle * steer / CAR_WHEELBASE  # (E, A)
    ground_mask = on_ground.float()  # (E, A)

    # Surface-relative steering: angular velocity = steer_rate * surface_normal
    surf_normal = state.car_surface_normal  # (E, A, 3)
    steer_ang_vel = steer_rate.unsqueeze(-1) * surf_normal  # (E, A, 3)

    # On ground: set ang_vel to steering; in air: keep existing ang_vel
    state.car_ang_vel = (
        steer_ang_vel * ground_mask.unsqueeze(-1) +
        state.car_ang_vel * (1.0 - ground_mask.unsqueeze(-1))
    )

    # ── Air physics ──
    air_mask = in_air.float()  # (E, A)

    # Lock air pitch input during flip pitch lock window (0.95s after flip)
    pitch_locked = (state.car_has_flipped > 0.5) & (state.car_flip_time < FLIP_PITCH_LOCK_TIME)
    effective_pitch = torch.where(pitch_locked, torch.zeros_like(pitch_in), pitch_in)

    # Air torques (in car's local frame axes)
    air_torque = torch.stack([
        roll_in * ROLL_TORQUE,
        effective_pitch * PITCH_TORQUE,
        yaw_in * YAW_TORQUE,
    ], dim=-1)  # (E, A, 3)

    # Per-axis angular velocity damping in air (RocketSim model)
    # Pitch/yaw damping scales with (1 - |input|): zero when holding full input
    # Roll damping is always active regardless of input
    if in_air.any():
        fwd = state.car_fwd                                      # (E, A, 3)
        right = torch.cross(state.car_up, fwd, dim=-1)           # (E, A, 3)
        up = state.car_up                                         # (E, A, 3)

        # Project angular velocity onto car-local axes
        ang_pitch = (state.car_ang_vel * right).sum(dim=-1)  # (E, A)
        ang_yaw = (state.car_ang_vel * up).sum(dim=-1)       # (E, A)
        ang_roll = (state.car_ang_vel * fwd).sum(dim=-1)     # (E, A)

        # Input-dependent damping factors
        pitch_damp = PITCH_ANG_DAMPING * (1.0 - torch.abs(pitch_in))  # (E, A)
        yaw_damp = YAW_ANG_DAMPING * (1.0 - torch.abs(yaw_in))       # (E, A)
        roll_damp = ROLL_ANG_DAMPING  # always active (Python float)

        # Subtract damping torques
        damp_vec = (
            (ang_pitch * pitch_damp).unsqueeze(-1) * right +
            (ang_yaw * yaw_damp).unsqueeze(-1) * up +
            (ang_roll * roll_damp).unsqueeze(-1) * fwd
        )  # (E, A, 3)
        state.car_ang_vel -= damp_vec * dt * air_mask.unsqueeze(-1)

    # Apply air torques
    state.car_ang_vel += air_torque * dt * air_mask.unsqueeze(-1)

    # ── Angular velocity cap ──
    ang_speed = state.car_ang_vel.norm(dim=-1, keepdim=True)  # (E, A, 1)
    state.car_ang_vel *= torch.where(
        ang_speed > CAR_MAX_ANG_SPEED,
        CAR_MAX_ANG_SPEED / ang_speed.clamp(min=1e-6),
        1.0,
    )

    # Air throttle (very weak forward push)
    air_throttle = throttle * AIR_THROTTLE_ACCEL
    state.car_vel += (air_throttle.unsqueeze(-1) * fwd * dt * air_mask.unsqueeze(-1))

    # ── Boost (ground/air have different acceleration) ──
    boost_active = (boost > 0.5) & (state.car_boost > 0.005) & alive  # (E, A)
    boost_f = boost_active.float()
    boost_accel = torch.where(on_ground, BOOST_ACCEL_GROUND, BOOST_ACCEL_AIR)
    state.car_vel += (boost_accel * dt * boost_f).unsqueeze(-1) * fwd
    state.car_boost -= BOOST_CONSUMPTION * dt * boost_f
    state.car_boost.clamp_(min=0.0, max=1.0)

    # ── Jump mechanics ──
    _apply_jump_flip(state, jump, pitch_in, yaw_in, forward_speed, dt)

    # ── Gravity (surface-relative when on surface, standard downward in air) ──
    grav = GRAVITY * dt
    on_ground_f = on_ground.float()
    state.car_vel += state.car_surface_normal * grav * on_ground_f.unsqueeze(-1)
    state.car_vel[:, :, 2] += grav * (1.0 - on_ground_f)

    # ── Sticky forces: keep car attached to surface when throttling ──
    throttling = torch.abs(throttle) > 0.01
    sticky_mask = on_ground & throttling & alive
    if sticky_mask.any():
        up_z = state.car_up[:, :, 2]  # (E, A)
        sticky_strength = STICKY_FORCE_GROUND + (1.0 - torch.abs(up_z))
        sticky_force = state.car_surface_normal * (sticky_strength * (-GRAVITY) * dt).unsqueeze(-1)
        state.car_vel += sticky_force * sticky_mask.unsqueeze(-1).float()

    # ── Auto-roll: corrective torque toward surface-aligned orientation ──
    if on_ground.any():
        # Compute the car's right vector
        car_right = torch.cross(state.car_up, state.car_fwd, dim=-1)  # (E, A, 3)
        # How much the car's right vector deviates from surface plane
        # The surface normal component of the right vector
        right_surf = (car_right * state.car_surface_normal).sum(dim=-1)  # (E, A)
        # Apply corrective roll torque proportional to misalignment
        autoroll_torque = -right_surf * CAR_AUTOROLL_TORQUE  # (E, A)
        # Apply along forward direction (roll axis)
        autoroll_mask = on_ground & throttling & alive
        state.car_ang_vel += (autoroll_torque.unsqueeze(-1) * state.car_fwd * dt
                              * autoroll_mask.unsqueeze(-1).float())

    # ── Flip torque (feature 3): apply continuous torque during flip ──
    _apply_flip_torque(state, pitch_in, dt)

    # ── Supersonic tracking (feature 14): hysteresis model ──
    _update_supersonic(state, dt)

    # ── Auto-flip recovery (feature 13) ──
    _apply_autoflip(state, jump, dt)

    # ── Speed cap ──
    speed = state.car_vel.norm(dim=-1, keepdim=True)  # (E, A, 1)
    state.car_vel *= torch.where(speed > CAR_MAX_SPEED, CAR_MAX_SPEED / speed.clamp(min=1e-6), 1.0)

    # ── Surface orientation alignment ──
    if on_ground.any():
        target_up = state.car_surface_normal  # (E, A, 3)
        current_up = state.car_up             # (E, A, 3)
        blended_up = current_up + 0.15 * (target_up - current_up)
        blended_up = safe_normalize(blended_up)
        dot = (current_up * blended_up).sum(dim=-1).clamp(-1.0, 1.0)  # (E, A)
        axis = torch.cross(current_up, blended_up, dim=-1)  # (E, A, 3)
        axis = safe_normalize(axis)
        angle = torch.acos(dot)  # (E, A)
        needs_correction = on_ground & (angle > 0.001)
        if needs_correction.any():
            correction_quat = quat_from_axis_angle(axis, angle)
            corrected = quat_multiply(correction_quat, state.car_quat)
            corrected = quat_normalize(corrected)
            state.car_quat = torch.where(
                needs_correction.unsqueeze(-1),
                corrected,
                state.car_quat,
            )

    # ── Zero out demoed car physics ──
    demoed = state.car_is_demoed > 0.5
    state.car_vel[demoed] = 0.0
    state.car_ang_vel[demoed] = 0.0


def _apply_jump_flip(state, jump_input, pitch_in, yaw_in, forward_speed, dt):
    """Handle first jump, jump hold, and dodge/flip with direction scaling.

    Modifies state in-place.
    """
    on_ground = state.car_on_ground > 0.5
    jump_pressed = jump_input > 0.5
    alive = state.car_is_demoed < 0.5

    # ── First jump: launch along surface normal ──
    can_first_jump = on_ground & jump_pressed & (state.car_has_jumped < 0.5) & alive
    if can_first_jump.any():
        impulse = state.car_surface_normal * JUMP_IMPULSE  # (E, A, 3)
        state.car_vel += impulse * can_first_jump.unsqueeze(-1).float()
        state.car_on_ground[can_first_jump] = 0.0
        state.car_has_jumped[can_first_jump] = 1.0
        state.car_is_jumping[can_first_jump] = 1.0
        state.car_jump_timer[can_first_jump] = 0.0
        state.car_has_flip[can_first_jump] = 1.0

    # ── Jump hold bonus (sustained force for first 0.2s) ──
    holding_jump = (
        jump_pressed &
        (state.car_is_jumping > 0.5) &
        (state.car_jump_timer < JUMP_HOLD_TIME) &
        alive
    )
    if holding_jump.any():
        # Jump pre-min scale: first 0.025s uses 62% force
        jump_scale = torch.where(
            state.car_jump_timer < JUMP_MIN_TIME,
            JUMP_MIN_FORCE_SCALE,
            1.0,
        )
        state.car_vel += (state.car_surface_normal * JUMP_HOLD_FORCE * jump_scale.unsqueeze(-1) * dt
                          * holding_jump.unsqueeze(-1).float())

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
        (~holding_jump) &
        alive
    )
    if can_flip.any():
        # Dodge direction from stick input
        dodge_dir_y = -pitch_in  # pitch forward = positive Y in car frame
        dodge_dir_x = yaw_in
        dodge_mag = torch.sqrt(dodge_dir_x**2 + dodge_dir_y**2 + 1e-8)

        has_dir = dodge_mag > 0.1
        dodge_dir_x = torch.where(has_dir, dodge_dir_x / dodge_mag, 0.0)
        dodge_dir_y = torch.where(has_dir, dodge_dir_y / dodge_mag, 0.0)

        # ── Direction-dependent impulse scaling (RocketSim) ──
        speed_ratio = (torch.abs(forward_speed) / CAR_MAX_SPEED).clamp(max=1.0)

        # Forward/backward component scaling
        is_backward = dodge_dir_y < 0
        fwd_bwd_scale = torch.where(
            is_backward,
            (1.0 + (FLIP_BACKWARD_SCALE - 1.0) * speed_ratio) * FLIP_BACKWARD_X_SCALE,
            1.0 + (FLIP_FORWARD_SCALE - 1.0) * speed_ratio,  # = 1.0 always
        )

        # Side component scaling
        side_scale = 1.0 + (FLIP_SIDE_SCALE - 1.0) * speed_ratio

        # Apply scaling to dodge direction
        scaled_dir_y = dodge_dir_y * fwd_bwd_scale
        scaled_dir_x = dodge_dir_x * side_scale

        # Convert to world-space dodge impulse
        fwd = state.car_fwd  # (E, A, 3)
        right = torch.cross(state.car_up, fwd, dim=-1)  # (E, A, 3)
        right = safe_normalize(right)

        world_dodge = (fwd * scaled_dir_y.unsqueeze(-1) +
                       right * scaled_dir_x.unsqueeze(-1))  # (E, A, 3)

        # Directional dodge: horizontal impulse (XY only)
        flip_mask = (can_flip & has_dir).unsqueeze(-1).float()
        state.car_vel[:, :, :2] += (world_dodge[:, :, :2] * FLIP_IMPULSE * flip_mask[:, :, :2])

        # Double jump (no direction): upward impulse
        double_jump = can_flip & (~has_dir)
        state.car_vel[:, :, 2] += JUMP_IMPULSE * double_jump.float()

        # ── Flip Z-damp: initial clamp on directional dodge ──
        dodge_z_mask = can_flip & has_dir
        state.car_vel[:, :, 2] = torch.where(
            dodge_z_mask,
            state.car_vel[:, :, 2].clamp(min=0) * FLIP_Z_DAMP,
            state.car_vel[:, :, 2],
        )

        # Cancel downward velocity on double-jump
        cancel_mask = double_jump & (state.car_vel[:, :, 2] < 0)
        state.car_vel[:, :, 2] = torch.where(cancel_mask, 0.0, state.car_vel[:, :, 2])

        state.car_has_flipped[can_flip] = 1.0
        state.car_has_flip[can_flip] = 0.0
        state.car_is_jumping[can_flip] = 0.0

        # ── Store flip torque direction for continuous torque ──
        # Torque: pitch component on Y-axis, yaw component on X-axis (roll)
        flip_torque_dir = torch.zeros_like(state.car_flip_rel_torque)  # (E, A, 3)
        # X-axis torque (roll) from yaw input of dodge
        flip_torque_dir[:, :, 0] = dodge_dir_x * FLIP_TORQUE_X
        # Y-axis torque (pitch) from pitch input of dodge
        flip_torque_dir[:, :, 1] = dodge_dir_y * FLIP_TORQUE_Y
        # Store only for cars that did a directional flip
        dir_flip = (can_flip & has_dir).unsqueeze(-1).float()
        state.car_flip_rel_torque = torch.where(
            dir_flip > 0.5,
            flip_torque_dir,
            state.car_flip_rel_torque,
        )
        state.car_is_flipping[can_flip & has_dir] = 1.0
        state.car_flip_time[can_flip] = 0.0

    # ── Update jump timer ──
    has_jumped = state.car_has_jumped > 0.5
    state.car_jump_timer += dt * has_jumped.float()

    # ── Update flip time for flip torque and Z-damp ──
    has_flipped = state.car_has_flipped > 0.5
    state.car_flip_time += dt * has_flipped.float()

    # ── Gradual flip Z-damp (feature 7): apply during 0.15-0.21s window ──
    in_zdamp_window = has_flipped & (state.car_flip_time >= FLIP_Z_DAMP_START) & (state.car_flip_time <= FLIP_Z_DAMP_END)
    if in_zdamp_window.any():
        zdamp_factor = FLIP_Z_DAMP_PER_TICK ** (dt * 120.0)  # scale for dt
        state.car_vel[:, :, 2] = torch.where(
            in_zdamp_window,
            state.car_vel[:, :, 2] * zdamp_factor,
            state.car_vel[:, :, 2],
        )

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
    # Ball air drag
    drag_factor = 1.0 - BALL_DRAG * dt
    state.ball_vel *= drag_factor
    # Ball speed cap
    ball_speed = state.ball_vel.norm(dim=-1, keepdim=True)
    state.ball_vel *= torch.where(
        ball_speed > BALL_MAX_SPEED,
        BALL_MAX_SPEED / ball_speed.clamp(min=1e-6),
        1.0,
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
    is_blue = state.car_team == 0  # (E, A)
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

    # Reset new flip/supersonic/autoflip state
    state.car_is_flipping[respawn] = 0.0
    state.car_flip_time[respawn] = 0.0
    state.car_flip_rel_torque[respawn] = 0.0
    state.car_handbrake_val[respawn] = 0.0
    state.car_is_supersonic[respawn] = 0.0
    state.car_supersonic_time[respawn] = 0.0
    state.car_autoflip_timer[respawn] = 0.0
    state.car_autoflip_torque_scale[respawn] = 0.0

    # Reset orientation to upright, facing appropriate direction
    state.car_quat[respawn] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)


def _apply_flip_torque(state, pitch_in, dt):
    """Apply continuous flip torque during the flip torque window (0.65s).

    Includes flip cancel: if pitch input opposes flip direction, scale torque down.
    """
    flipping = state.car_is_flipping > 0.5
    if not flipping.any():
        return

    # End flip torque after FLIP_TORQUE_TIME
    torque_expired = flipping & (state.car_flip_time > FLIP_TORQUE_TIME)
    state.car_is_flipping[torque_expired] = 0.0

    active = flipping & (state.car_flip_time <= FLIP_TORQUE_TIME)
    if not active.any():
        return

    # Get stored torque direction (in car-local frame)
    torque = state.car_flip_rel_torque  # (E, A, 3) — [roll_torque, pitch_torque, 0]

    # Flip cancel: if pitch input matches flip direction, scale torque Y by (1 - |pitch|)
    # "matches" means same sign as the dodge pitch direction (stored in torque Y)
    flip_pitch_dir = torque[:, :, 1]  # positive = forward flip
    same_dir = (pitch_in * flip_pitch_dir) > 0  # pitch input matches flip direction
    cancel_scale = torch.where(
        same_dir,
        (1.0 - torch.abs(pitch_in)),
        torch.ones_like(pitch_in),
    )

    # Apply cancel only to Y (pitch) component of torque
    scaled_torque = torque.clone()
    scaled_torque[:, :, 1] = torque[:, :, 1] * cancel_scale

    # Convert car-local torque to world-space angular velocity delta
    fwd = state.car_fwd   # (E, A, 3)
    right = torch.cross(state.car_up, fwd, dim=-1)  # (E, A, 3)
    up = state.car_up     # (E, A, 3)

    world_torque = (
        scaled_torque[:, :, 0:1] * fwd +     # roll component along forward
        scaled_torque[:, :, 1:2] * right +    # pitch component along right
        scaled_torque[:, :, 2:3] * up         # yaw component along up (usually 0)
    )

    active_f = active.unsqueeze(-1).float()
    state.car_ang_vel += world_torque * dt * active_f


def _update_supersonic(state, dt):
    """Update supersonic state with hysteresis model.

    Activate at 2200 uu/s, maintain down to 2100 for up to 1s.
    """
    speed = state.car_vel.norm(dim=-1)  # (E, A)
    alive = state.car_is_demoed < 0.5
    is_ss = state.car_is_supersonic > 0.5

    # Activate supersonic
    activate = (~is_ss) & (speed >= CAR_SUPERSONIC_ACTIVATE) & alive
    state.car_is_supersonic[activate] = 1.0
    state.car_supersonic_time[activate] = 0.0

    # Maintain supersonic: above maintain threshold → reset timer
    maintaining = is_ss & (speed >= CAR_SUPERSONIC_MAINTAIN)
    state.car_supersonic_time[maintaining] = 0.0

    # Below maintain threshold → count time
    below = is_ss & (speed < CAR_SUPERSONIC_MAINTAIN)
    state.car_supersonic_time += dt * below.float()

    # Deactivate if below threshold too long or too slow
    deactivate = is_ss & (state.car_supersonic_time > CAR_SUPERSONIC_MAINTAIN_TIME)
    state.car_is_supersonic[deactivate] = 0.0
    state.car_supersonic_time[deactivate] = 0.0


def _apply_autoflip(state, jump_input, dt):
    """Auto-flip recovery when upside down on surface + jump pressed.

    If |roll| > 2.8 rad and on ground and jump pressed, apply recovery torque.
    """
    on_ground = state.car_on_ground > 0.5
    alive = state.car_is_demoed < 0.5
    jump_pressed = jump_input > 0.5

    # Compute roll angle from car_up Z component
    # When upside down, car_up.z < 0, and we can estimate roll from acos
    up_z = state.car_up[:, :, 2]  # (E, A)
    # Approximate roll: if up.z < cos(2.8) ≈ -0.942, car is nearly upside down
    upside_down = up_z < -0.942  # corresponds to |roll| > ~2.8 rad

    # Active autoflip timer
    autoflip_active = state.car_autoflip_timer > 0

    # Start autoflip
    can_start = on_ground & upside_down & jump_pressed & alive & (~autoflip_active)
    if can_start.any():
        state.car_autoflip_timer[can_start] = CAR_AUTOFLIP_TIME
        # Determine roll direction (+1 or -1) based on car's right lean
        right_vec = torch.cross(state.car_up, state.car_fwd, dim=-1)
        lean = right_vec[:, :, 2]  # positive = leaning right
        state.car_autoflip_torque_scale[can_start] = torch.where(
            lean[can_start] >= 0, -1.0, 1.0)

    # Apply autoflip torque
    if autoflip_active.any():
        torque = state.car_autoflip_torque_scale * CAR_AUTOFLIP_TORQUE  # (E, A)
        state.car_ang_vel[:, :, 0] += torque * dt * autoflip_active.float()
        state.car_autoflip_timer -= dt * autoflip_active.float()
        state.car_autoflip_timer.clamp_(min=0.0)
